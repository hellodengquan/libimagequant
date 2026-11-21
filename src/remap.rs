use crate::error::Error;
use crate::image::Image;
use crate::kmeans::Kmeans;
use crate::nearest::Nearest;
use crate::pal::{f_pixel, PalF, PalIndexRemap, Palette, ARGBF};
use crate::quant::QuantizationResult;
use crate::rayoff::*;
use crate::rows::{temp_buf, DynamicRows};
use crate::seacow::{RowBitmap, RowBitmapMut};
use crate::CacheLineAlign;
use core::cell::RefCell;
use core::mem::MaybeUninit;

#[cfg(all(not(feature = "std"), feature = "no_std"))]
use crate::no_std_compat::*;

#[repr(u8)]
#[derive(Eq, PartialEq, Clone, Copy)]
pub enum DitherMapMode {
    None = 0,
    Enabled = 1,
    Always = 2,
}

/// Dithering algorithm used when remapping an image to the palette
///
/// The default is [`DitheringMode::FloydSteinberg`].
#[repr(u8)]
#[derive(Debug, Default, Eq, PartialEq, Clone, Copy)]
#[non_exhaustive]
pub enum DitheringMode {
    /// Floyd-Steinberg error diffusion (the default).
    ///
    /// Highest quality for still images, but the result depends on the order in which pixels are processed.
    #[default]
    FloydSteinberg = 0,
    /// Ordered dithering using a Bayer threshold matrix.
    ///
    /// The output for a pixel depends only on its position and color, so areas that don't change between
    /// frames of an animation get exactly the same palette indices in every frame and don't flicker.
    Ordered = 1,
}

#[derive(Clone)]
pub(crate) struct Remapped {
    pub(crate) int_palette: Palette,
    pub(crate) palette_error: Option<f64>,
}

#[inline(never)]
pub(crate) fn remap_to_palette<'x, 'b: 'x>(px: &mut DynamicRows, background: Option<&mut Image<'_>>, importance_map: Option<&[u8]>, output_pixels: &'x mut RowBitmapMut<'b, MaybeUninit<PalIndexRemap>>, palette: &mut PalF) -> Result<(f64, RowBitmap<'x, PalIndexRemap>), Error> {
    let n = Nearest::new(palette)?;
    let colors = palette.as_slice();
    let palette_len = colors.len();
    if palette_len > PalIndexRemap::MAX as usize + 1 {
        return Err(Error::Unsupported);
    }

    let tls = ThreadLocal::new();
    let width = px.width as usize;
    let per_thread_buffers = move || -> Result<_, Error> { Ok(CacheLineAlign(RefCell::new((Kmeans::new(palette_len)?, temp_buf(width)?, temp_buf(width)?, temp_buf(width)?)))) };

    let tls_tmp1 = tls.get_or_try(per_thread_buffers)?;
    let mut tls_tmp = tls_tmp1.0.borrow_mut();

    let input_rows = px.rows_iter(&mut tls_tmp.1)?;
    let (background, transparent_index) = background.map(|background| {
        (Some(background), n.search(&f_pixel::default(), 0).0 as PalIndexRemap)
    })
    .filter(|&(_, transparent_index)| colors[usize::from(transparent_index)].is_fully_transparent())
    .unwrap_or((None, 0));
    let background = background.map(|bg| bg.px.rows_iter(&mut tls_tmp.1)).transpose()?;

    if background.is_some() {
        tls_tmp.0.update_color(f_pixel::default(), 1., transparent_index as _);
    }

    drop(tls_tmp);

    let remapping_error = output_pixels.rows_mut().enumerate().par_bridge().map(|(row, output_pixels_row)| {
        let mut remapping_error = 0.;
        #[allow(irrefutable_let_patterns)]
        let Ok(tls_res) = tls.get_or_try(per_thread_buffers) else { return f64::NAN };
        let (kmeans, temp_row, temp_row_f, temp_row_f_bg) = &mut *tls_res.0.borrow_mut();

        let output_pixels_row = &mut output_pixels_row[..width];
        let importance_map = importance_map.and_then(|m| m.get(row * width..)).unwrap_or(&[]);
        let row_pixels = &input_rows.row_f_shared(temp_row, temp_row_f, row)[..width];
        let bg_pixels = if let Some(background) = &background  {
            &background.row_f_shared(temp_row, temp_row_f_bg, row)[..width]
        } else { &[] };

        let mut last_match = 0;
        for (col, (inp, out)) in row_pixels.iter().zip(output_pixels_row).enumerate() {
            let (matched, diff) = n.search(inp, last_match as _);
            let matched = matched as PalIndexRemap;
            last_match = matched;
            if let Some(bg) = bg_pixels.get(col) {
                let bg_diff = bg.diff(inp);
                if bg_diff <= diff {
                    remapping_error += f64::from(bg_diff);
                    out.write(transparent_index);
                    continue;
                }
            }
            remapping_error += f64::from(diff);
            out.write(matched);
            let importance = f32::from(importance_map.get(col).copied().unwrap_or(1));
            kmeans.update_color(*inp, importance, matched as _);
        }
        remapping_error
    })
    .sum::<f64>();

    if remapping_error.is_nan() {
        return Err(Error::OutOfMemory);
    }

    if let Some(kmeans) = tls.into_iter()
        .map(|t| t.0.into_inner().0)
        .reduce(Kmeans::merge) { kmeans.finalize(palette); }

    let remapping_error = remapping_error / f64::from(px.width * px.height);
    Ok((remapping_error, unsafe { output_pixels.assume_init() }))
}

fn get_dithered_pixel(dither_level: f32, max_dither_error: f32, thiserr: f_pixel, px: f_pixel) -> f_pixel {
    let s = thiserr.0 * dither_level;
    // This prevents gaudy green pixels popping out of the blue (or red or black! ;)
    let dither_error = s.r.mul_add(s.r, s.g * s.g) + s.b.mul_add(s.b, s.a * s.a);
    if dither_error < 2. / 256. / 256. {
        // don't dither areas that don't have noticeable error — makes file smaller
        return px;
    }

    let mut ratio: f32 = 1.;
    const MAX_OVERFLOW: f32 = 1.1;
    const MAX_UNDERFLOW: f32 = -0.1;
    // allowing some overflow prevents undithered bands caused by clamping of all channels
    if px.r + s.r > MAX_OVERFLOW {
        ratio = ratio.min((MAX_OVERFLOW - px.r) / s.r);
    } else if px.r + s.r < MAX_UNDERFLOW {
        ratio = ratio.min((MAX_UNDERFLOW - px.r) / s.r);
    }
    if px.g + s.g > MAX_OVERFLOW {
        ratio = ratio.min((MAX_OVERFLOW - px.g) / s.g);
    } else if px.g + s.g < MAX_UNDERFLOW {
        ratio = ratio.min((MAX_UNDERFLOW - px.g) / s.g);
    }
    if px.b + s.b > MAX_OVERFLOW {
        ratio = ratio.min((MAX_OVERFLOW - px.b) / s.b);
    } else if px.b + s.b < MAX_UNDERFLOW {
        ratio = ratio.min((MAX_UNDERFLOW - px.b) / s.b);
    }
    if dither_error > max_dither_error {
        ratio *= 0.8;
    }
    f_pixel(ARGBF {
        a: (px.a + s.a).clamp(0., 1.),
        r: s.r.mul_add(ratio, px.r),
        g: s.g.mul_add(ratio, px.g),
        b: s.b.mul_add(ratio, px.b),
    })
}

/// Uses edge/noise map to apply dithering only to flat areas. Dithering on edges creates jagged lines, and noisy areas are "naturally" dithered.
///
///  If `output_image_is_remapped` is true, only pixels noticeably changed by error diffusion will be written to output image.
#[inline(never)]
pub(crate) fn remap_to_palette_floyd(input_image: &mut Image, mut output_pixels: RowBitmapMut<'_, MaybeUninit<PalIndexRemap>>, palette: &PalF, quant: &QuantizationResult, max_dither_error: f32, output_image_is_remapped: bool) -> Result<(), Error> {
    let progress_stage1 = if quant.use_dither_map != DitherMapMode::None { 20 } else { 0 };

    let width = input_image.width();
    let height = input_image.height();

    let mut temp_row = temp_buf(width)?;

    let dither_map = if quant.use_dither_map != DitherMapMode::None {
        input_image.dither_map.as_deref().or(input_image.edges.as_deref()).unwrap_or(&[])
    } else {
        &[]
    };

    let n = Nearest::new(palette)?;
    let palette = palette.as_slice();

    let mut background = input_image.background.as_mut().map(|bg| {
        bg.px.prepare_iter(&mut temp_row, true)?;
        Ok::<_, Error>(&bg.px)
    }).transpose()?;

    let transparent_index = if background.is_some() { n.search(&f_pixel::default(), 0).0 as PalIndexRemap } else { 0 };
    if background.is_some() && !palette[transparent_index as usize].is_fully_transparent() {
        background = None;
    }
    // response to this value is non-linear and without it any value < 0.8 would give almost no dithering
    let mut base_dithering_level = (1. - quant.dither_level).mul_add(-(1. - quant.dither_level), 1.) * (15. / 16.); // prevent small errors from accumulating
    if !dither_map.is_empty() {
        base_dithering_level *= 1. / 255.; // dither_map is in 0-255 scale
    }

    // when using remapping on top of a background, lots of pixels may be transparent, making poor guesses
    // (guesses are only for speed, don't affect visuals)
    let guess_from_remapped_pixels = output_image_is_remapped && background.is_none();

    input_image.px.prepare_iter(&mut temp_row, true)?;
    let input_image_px = &input_image.px;
    let n = &n;

    // Chunks have overhead, so should be big (more than 2 bring diminishing results). Chunks risk causing seams, so should be tall.
    let num_chunks = if quant.single_threaded_dithering { 1 } else { (width * height / 524_288).min(height / 128).max(if height > 128 {2} else {1}).min(num_cpus()) };
    let chunks = output_pixels.chunks((height + num_chunks - 1) / num_chunks).map(CacheLineAlign);
    scope(move |s| {
        let mut chunk_start_row = 0;
        for mut chunk in chunks {
            let chunk_len = chunk.0.len();
            let mut temp_row = temp_buf(width)?;
            let mut input_image_iter = input_image_px.rows_iter_prepared()?;
            let mut background = background.map(|bg| bg.rows_iter_prepared()).transpose()?;
            let mut diffusion = Vec::new();
            let errwidth = width + 2; // +2 saves from checking out of bounds access
            diffusion.try_reserve_exact(errwidth * 2)?;
            diffusion.resize(errwidth * 2, f_pixel::default());

            // restart of dithering creates a seam. this does redundant work to init diffusion state,
            // so that later chunks don't start from scratch
            if chunk_start_row > 2 {
                let mut discard_row = temp_buf(width)?;
                for row in (chunk_start_row - 2) .. chunk_start_row {
                    let row_pixels = input_image_iter.row_f(&mut temp_row, row as _);
                    let bg_pixels = background.as_mut().map(|b| b.row_f(&mut temp_row, row as _)).unwrap_or(&[]);
                    let dither_map = dither_map.get(row * width .. row * width + width).unwrap_or(&[]);
                    let scan_forward = row & 1 == 0;
                    dither_row(row_pixels, &mut discard_row, width as u32, dither_map, base_dithering_level, max_dither_error, n, palette, transparent_index, bg_pixels, guess_from_remapped_pixels, &mut diffusion, scan_forward);
                }
            }
            // parallel remap makes progress not very useful
            if quant.remap_progress(progress_stage1 as f32 + chunk_start_row as f32 * (100. - progress_stage1 as f32) / height as f32) {
                return Err(Error::Aborted);
            }
            s.spawn(move |_| {
                for (chunk_row, output_pixels_row) in chunk.0.rows_mut().enumerate() {
                    let row = chunk_start_row + chunk_row;
                    let row_pixels = input_image_iter.row_f(&mut temp_row, row as _);
                    let bg_pixels = background.as_mut().map(|b| b.row_f(&mut temp_row, row as _)).unwrap_or(&[]);
                    let dither_map = dither_map.get(row * width .. row * width + width).unwrap_or(&[]);
                    let scan_forward = row & 1 == 0;
                    dither_row(row_pixels, output_pixels_row, width as u32, dither_map, base_dithering_level, max_dither_error, n, palette, transparent_index, bg_pixels, guess_from_remapped_pixels, &mut diffusion, scan_forward);
                }
            });
            chunk_start_row += chunk_len;
        }
        Ok(())
    })
}

/// Normalized 8×8 Bayer matrix (values 0-63).
///
/// The threshold of a pixel depends only on its position in the image, so the same input color at
/// the same position always produces the same palette index, regardless of other pixels or frames.
const BAYER8: [u8; 64] = [
     0, 32,  8, 40,  2, 34, 10, 42,
    48, 16, 56, 24, 50, 18, 58, 26,
    12, 44,  4, 36, 14, 46,  6, 38,
    60, 28, 52, 20, 62, 30, 54, 22,
     3, 35, 11, 43,  1, 33,  9, 41,
    51, 19, 59, 27, 49, 17, 57, 25,
    15, 47,  7, 39, 13, 45,  5, 37,
    63, 31, 55, 23, 61, 29, 53, 21,
];

/// Position-only threshold in range [0, 1)
#[inline]
fn bayer_threshold(col: usize, row: usize) -> f32 {
    let n = BAYER8[(row & 7) * 8 + (col & 7)];
    (f32::from(n) + 0.5) / 64.
}

/// Fraction of nearby pixels that should use `second_nearest` instead of `nearest`
/// to represent a color with squared distances `nearest_diff`/`second_diff` to those palette colors.
///
/// Range is `0..=0.5`: `0` when the color is exactly the nearest palette entry,
/// `0.5` when it's equally far from both of them. So the nearest color always dominates.
#[inline]
fn blend_fraction(nearest_diff: f32, second_diff: f32) -> f32 {
    let nearest = nearest_diff.sqrt();
    let second = second_diff.sqrt();
    let sum = nearest + second;
    if sum > 0. { nearest / sum } else { 0. }
}

/// Ordered (Bayer) dithering. Unlike error diffusion, each pixel is processed independently,
/// so the output is a pure function of the pixel's position and color.
///
/// The two palette colors closest to the input are chosen between: the second-nearest one is used
/// for the fraction of pixels determined by the pixel's position in the Bayer matrix and by the
/// dithering level. This approximates the input color in a fixed spatial pattern.
#[inline(never)]
fn dither_row_ordered(row: usize, row_pixels: &[f_pixel], output_pixels_row: &mut [MaybeUninit<PalIndexRemap>], dither_map: &[u8], base_dithering_level: f32, n: &Nearest, palette: &[f_pixel], transparent_index: PalIndexRemap, bg_pixels: &[f_pixel]) {
    let width = row_pixels.len();
    assert_eq!(output_pixels_row.len(), width);

    // The palette index of the previous pixel is only a search hint; `Nearest::search` always
    // returns the true nearest color, so the result is independent of processing order.
    let mut last_match = 0;
    for (col, input_px) in row_pixels.iter().copied().enumerate() {
        let (nearest, nearest_diff) = n.search(&input_px, last_match as _);
        let nearest = nearest as PalIndexRemap;
        last_match = nearest;

        let mut matched = nearest;
        let mut matched_diff = nearest_diff;
        let mut dither_level = base_dithering_level;
        if let Some(&l) = dither_map.get(col) {
            dither_level *= f32::from(l);
        }
        // Dithering never introduces transparency: fully transparent pixels stay fully transparent,
        // and the transparent palette entry is never chosen as the alternate dithering color.
        // That keeps alpha handling consistent with the non-dithered mode, and leaves the transparent
        // index exclusively for background reuse.
        if dither_level > 0. && nearest_diff > 0. && palette.len() > 1
            && !input_px.is_fully_transparent()
            && !palette[usize::from(nearest)].is_fully_transparent()
        {
            let (second, second_diff) = n.search_excluding(&input_px, nearest as _);
            if second_diff.is_finite() && !palette[usize::from(second)].is_fully_transparent() {
                let fraction = blend_fraction(nearest_diff, second_diff);
                if bayer_threshold(col, row) < fraction * dither_level {
                    matched = second as PalIndexRemap;
                    matched_diff = second_diff;
                }
            }
        }

        if let Some(bg_pixel) = bg_pixels.get(col) {
            // Pixels already reproduced by the previous frame at least as well are kept transparent,
            // matching the reuse behavior of the error diffusion mode.
            if bg_pixel.diff(&input_px) <= matched_diff {
                output_pixels_row[col].write(transparent_index);
                continue;
            }
        }
        output_pixels_row[col].write(matched);
    }
}

/// Same as [`remap_to_palette_floyd`], but uses ordered dithering, which produces stable output for animations.
#[inline(never)]
pub(crate) fn remap_to_palette_ordered(input_image: &mut Image, mut output_pixels: RowBitmapMut<'_, MaybeUninit<PalIndexRemap>>, palette: &PalF, quant: &QuantizationResult) -> Result<(), Error> {
    let progress_stage1 = if quant.use_dither_map != DitherMapMode::None { 20 } else { 0 };

    let width = input_image.width();

    let mut temp_row = temp_buf(width)?;

    let dither_map = if quant.use_dither_map != DitherMapMode::None {
        input_image.dither_map.as_deref().or(input_image.edges.as_deref()).unwrap_or(&[])
    } else {
        &[]
    };

    let n = Nearest::new(palette)?;
    let palette = palette.as_slice();

    let mut background = input_image.background.as_mut().map(|bg| {
        bg.px.prepare_iter(&mut temp_row, true)?;
        Ok::<_, Error>(&bg.px)
    }).transpose()?;

    let transparent_index = if background.is_some() { n.search(&f_pixel::default(), 0).0 as PalIndexRemap } else { 0 };
    if background.is_some() && !palette[usize::from(transparent_index)].is_fully_transparent() {
        background = None;
    }
    // The user-visible dithering level is used directly: it's the fraction of pixels that use the
    // alternate palette color. The dither map (edges/noise) attenuates it per pixel.
    let mut base_dithering_level = quant.dither_level;
    if !dither_map.is_empty() {
        base_dithering_level *= 1. / 255.; // dither_map is in 0-255 scale
    }

    let tls = ThreadLocal::new();
    let per_thread_buffers = move || -> Result<_, Error> { Ok(CacheLineAlign(RefCell::new((temp_buf(width)?, temp_buf(width)?, temp_buf(width)?)))) };

    let tls_tmp1 = tls.get_or_try(per_thread_buffers)?;
    let mut tls_tmp = tls_tmp1.0.borrow_mut();
    input_image.px.prepare_iter(&mut tls_tmp.0, true)?;
    let input_rows = input_image.px.rows_iter_prepared()?;
    let background_rows = background.map(|bg| bg.rows_iter_prepared()).transpose()?;
    drop(tls_tmp);

    if quant.remap_progress(progress_stage1 as f32) {
        return Err(Error::Aborted);
    }

    // Rows are independent of each other, so there's no chunking and no seams between chunks
    output_pixels.rows_mut().enumerate().par_bridge().for_each(|(row, output_pixels_row)| {
        #[allow(irrefutable_let_patterns)]
        let Ok(tls_res) = tls.get_or_try(per_thread_buffers) else { return };
        let (temp_row, temp_row_f, temp_row_f_bg) = &mut *tls_res.0.borrow_mut();

        let output_pixels_row = &mut output_pixels_row[..width];
        let dither_map = dither_map.get(row * width .. row * width + width).unwrap_or(&[]);
        let row_pixels = &input_rows.row_f_shared(temp_row, temp_row_f, row)[..width];
        let bg_pixels = if let Some(background_rows) = &background_rows {
            &background_rows.row_f_shared(temp_row, temp_row_f_bg, row)[..width]
        } else { &[] };

        dither_row_ordered(row, row_pixels, output_pixels_row, dither_map, base_dithering_level, &n, palette, transparent_index, bg_pixels);
    });

    if quant.remap_progress(100.) {
        return Err(Error::Aborted);
    }

    Ok(())
}

#[inline(never)]
fn dither_row(row_pixels: &[f_pixel], output_pixels_row: &mut [MaybeUninit<PalIndexRemap>], width: u32, dither_map: &[u8], base_dithering_level: f32, max_dither_error: f32, n: &Nearest, palette: &[f_pixel], transparent_index: PalIndexRemap, bg_pixels: &[f_pixel], guess_from_remapped_pixels: bool, diffusion: &mut [f_pixel], even_row: bool) {
    let width = width as usize;
    assert_eq!(row_pixels.len(), width);
    assert_eq!(output_pixels_row.len(), width);

    let (thiserr, nexterr) = {
        // +2 saves from checking out of bounds access
        let (d1, d2) = diffusion.split_at_mut(width + 2);
        if even_row { (d1, d2) } else { (d2, d1) }
    };

    nexterr.fill_with(f_pixel::default);

    let mut undithered_bg_used = 0u8;
    let mut last_match = 0;
    for x in 0..width {
        let col = if even_row { x } else { width - 1 - x };
        let thiserr = &mut thiserr[col .. col + 3];
        let nexterr = &mut nexterr[col .. col + 3];
        let input_px = row_pixels[col];

        let mut dither_level = base_dithering_level;
        if let Some(&l) = dither_map.get(col) {
            dither_level *= f32::from(l);
        }

        let spx = get_dithered_pixel(dither_level, max_dither_error, thiserr[1], input_px);
        let guessed_match = if guess_from_remapped_pixels {
            unsafe { output_pixels_row[col].assume_init() }
        } else {
            last_match
        };
        let (matched, dither_diff) = n.search(&spx, guessed_match as _);
        let mut matched = matched as PalIndexRemap;
        last_match = matched as PalIndexRemap;
        let mut output_px = palette[last_match as usize];
        if let Some(bg_pixel) = bg_pixels.get(col) {
            // if the background makes better match *with* dithering, it's a definitive win
            let bg_for_dither_diff = spx.diff(bg_pixel);
            if bg_for_dither_diff <= dither_diff {
                output_px = *bg_pixel;
                matched = transparent_index;
            } else if undithered_bg_used > 1 {
                // the undithered fallback can cause artifacts when too many undithered pixels accumulate a big dithering error
                // so periodically ignore undithered fallback to prevent that
                undithered_bg_used = 0;
            } else {
                // if dithering is not applied, there's a high risk of creating artifacts (flat areas, error accumulating badly),
                // OTOH poor dithering disturbs static backgrounds and creates oscilalting frames that break backgrounds
                // back and forth in two differently bad ways
                let max_diff = input_px.diff(bg_pixel);
                let dithered_diff = input_px.diff(&output_px);
                // if dithering is worse than natural difference between frames
                // (this rule dithers moving areas, but does not dither static areas)
                if dithered_diff > max_diff {
                    // then see if an undithered color is closer to the ideal
                    let guessed_px = palette[guessed_match as usize];
                    let undithered_diff = input_px.diff(&guessed_px); // If dithering error is crazy high, don't propagate it that much
                    if undithered_diff < max_diff {
                        undithered_bg_used += 1;
                        output_px = guessed_px;
                        matched = guessed_match;
                    }
                }
            }
        }
        output_pixels_row[col].write(matched);
        let mut err = spx.0 - output_px.0;
        // This prevents weird green pixels popping out of the blue (or red or black! ;)
        if err.r.mul_add(err.r, err.g * err.g) + err.b.mul_add(err.b, err.a * err.a) > max_dither_error {
            err *= 0.75;
        }
        if even_row {
            thiserr[2].0 += err * (7. / 16.);
            nexterr[0].0 += err * (3. / 16.);
            nexterr[1].0 += err * (5. / 16.);
            nexterr[2].0 = err * (1. / 16.);
        } else {
            thiserr[0].0 += err * (7. / 16.);
            nexterr[0].0 = err * (1. / 16.);
            nexterr[1].0 += err * (5. / 16.);
            nexterr[2].0 += err * (3. / 16.);
        }
    }
}

#[test]
fn send() {
    fn is_send<T: Send>() {}

    is_send::<RowBitmapMut<'_, MaybeUninit<u8>>>();
}

#[test]
fn background_to_nop() {
    use crate::RGBA;
    let pixels: Vec<_> = (0..200*200).map(|n| RGBA::new(n as u8, (n/17) as u8, (n/78) as u8, 255)).collect();

    let mut attr = crate::new();
    let mut img = attr.new_image_borrowed(&pixels, 200, 200, 0.).unwrap();
    let img2 = attr.new_image_borrowed(&pixels, 200, 200, 0.).unwrap();
    img.set_background(img2).unwrap();
    img.add_fixed_color(RGBA::new(0,0,0,0)).unwrap();
    attr.set_max_colors(3).unwrap();
    let mut res = attr.quantize(&mut img).unwrap();
    res.set_dithering_level(0.).unwrap();
    let (_, idx) = res.remapped(&mut img).unwrap();
    let first = idx[0];
    assert!(idx.iter().all(|&x| x == first));

    res.set_dithering_level(1.).unwrap();
    let (_, idx) = res.remapped(&mut img).unwrap();
    let first = idx[0];
    assert!(idx.iter().all(|&x| x == first));
}

#[cfg(test)]
fn gradient_image(width: usize, height: usize) -> Vec<crate::RGBA> {
    use crate::RGBA;
    (0..width * height).map(|n| {
        let x = (n % width) as u8;
        let y = (n / width) as u8;
        RGBA::new(x.wrapping_mul(3), y.wrapping_mul(5), 128, 255)
    }).collect()
}

/// Remapping may refine the palette on the first run, so tests that compare outputs call it once
/// to settle the palette, and only then compare.
#[cfg(test)]
fn quantized_result(attr: &crate::Attributes, pixels: &[crate::RGBA], width: usize, height: usize, mode: DitheringMode, level: f32) -> (Image<'static>, QuantizationResult) {
    let mut img = attr.new_image(pixels.to_vec(), width, height, 0.).unwrap();
    let mut res = attr.quantize(&mut img).unwrap();
    res.set_dithering_level(level).unwrap();
    res.set_dithering_mode(mode).unwrap();
    (img, res)
}

#[test]
fn ordered_dithering_is_position_determined() {
    use crate::DitheringMode;

    let width = 64;
    let height = 64;
    let pixels = gradient_image(width, height);
    let mut attr = crate::new();
    attr.set_max_colors(4).unwrap();
    let (mut img, mut res) = quantized_result(&attr, &pixels, width, height, DitheringMode::Ordered, 1.);
    assert_eq!(res.dithering_mode(), DitheringMode::Ordered);

    // settle the palette (the first remap may refine it)
    res.remapped(&mut img).unwrap();
    let (_, first) = res.remapped(&mut img).unwrap();
    let (_, second) = res.remapped(&mut img).unwrap();
    assert_eq!(first, second, "ordered dithering must be reproducible");
}

#[test]
fn ordered_dithering_level_zero_matches_no_dithering() {
    use crate::DitheringMode;

    let width = 40;
    let height = 40;
    let pixels = gradient_image(width, height);
    let mut attr = crate::new();
    attr.set_max_colors(4).unwrap();

    let (mut img, mut res) = quantized_result(&attr, &pixels, width, height, DitheringMode::Ordered, 0.);
    res.remapped(&mut img).unwrap();
    let (_, ordered) = res.remapped(&mut img).unwrap();

    // Dithering level 0 takes the non-dithered code path, same as Floyd-Steinberg at level 0
    res.set_dithering_mode(DitheringMode::FloydSteinberg).unwrap();
    let (_, floyd) = res.remapped(&mut img).unwrap();
    assert_eq!(ordered, floyd, "dithering level 0 must not depend on the mode");
}

#[test]
fn ordered_dithering_static_area_is_stable() {
    use crate::DitheringMode;
    use crate::RGBA;

    // Two "frames" that share a large static area but differ in a moving band
    let width = 48;
    let height = 48;
    let frame = |shift: u8| -> Vec<RGBA> {
        (0..width * height).map(|n| {
            let x = (n % width) as u8;
            let y = (n / width) as u8;
            if (16..32).contains(&y) {
                RGBA::new(x.wrapping_add(shift), 200u8.wrapping_sub(x), 50, 255)
            } else {
                RGBA::new(x.wrapping_mul(4), y.wrapping_mul(3), 100, 255)
            }
        }).collect()
    };

    let mut attr = crate::new();
    attr.set_max_colors(8).unwrap();

    // Use a preset dither map so that remapping doesn't refine the palette between frames
    let dither_map = vec![255u8; width * height];
    let make_image = |shift: u8| {
        let mut img = attr.new_image(frame(shift), width, height, 0.).unwrap();
        img.dither_map = Some(dither_map.clone().into_boxed_slice());
        img
    };

    let mut img = make_image(0);
    let mut res = attr.quantize(&mut img).unwrap();
    res.set_dithering_level(1.).unwrap();
    res.set_dithering_mode(DitheringMode::Ordered).unwrap();

    let mut img1 = make_image(0);
    let (pal1, idx1) = res.remapped(&mut img1).unwrap();
    let mut img2 = make_image(37);
    let (pal2, idx2) = res.remapped(&mut img2).unwrap();
    assert_eq!(pal1, pal2);

    for y in 0..height {
        if (16..32).contains(&y) { continue; }
        for x in 0..width {
            assert_eq!(idx1[y * width + x], idx2[y * width + x], "static pixel at {x},{y} flickered");
        }
    }
}

#[test]
fn ordered_dithering_background_reuse() {
    use crate::DitheringMode;
    use crate::RGBA;

    let width = 32;
    let height = 32;
    let pixels: Vec<_> = (0..width * height).map(|n| {
        let x = (n % width) as u8;
        let y = (n / width) as u8;
        RGBA::new(x.wrapping_mul(5), y.wrapping_mul(5), 90, 255)
    }).collect();

    let mut attr = crate::new();
    attr.set_max_colors(8).unwrap();
    let mut img = attr.new_image(pixels.clone(), width, height, 0.).unwrap();
    let mut background = attr.new_image(pixels, width, height, 0.).unwrap();
    img.add_fixed_color(RGBA::new(0, 0, 0, 0)).unwrap();
    background.add_fixed_color(RGBA::new(0, 0, 0, 0)).unwrap();
    img.set_background(background).unwrap();

    let mut res = attr.quantize(&mut img).unwrap();
    res.set_dithering_level(1.).unwrap();
    res.set_dithering_mode(DitheringMode::Ordered).unwrap();
    let (pal, idx) = res.remapped(&mut img).unwrap();

    let transparent: Vec<_> = pal.iter().enumerate().filter(|(_, c)| c.a == 0).map(|(i, _)| i as u8).collect();
    assert_eq!(1, transparent.len(), "expected a transparent palette entry");
    assert!(idx.iter().all(|&i| i == transparent[0]), "pixels identical to the background must reuse it");
}

#[test]
fn default_dithering_mode_is_floyd_steinberg() {
    use crate::DitheringMode;

    let width = 32;
    let height = 32;
    let pixels = gradient_image(width, height);
    let mut attr = crate::new();
    attr.set_max_colors(4).unwrap();

    let (mut img, mut res) = quantized_result(&attr, &pixels, width, height, DitheringMode::FloydSteinberg, 1.);
    assert_eq!(res.dithering_mode(), DitheringMode::default());
    res.remapped(&mut img).unwrap();
    let (_, floyd_explicit) = res.remapped(&mut img).unwrap();

    // A result that was never told about the mode uses the same (default) algorithm
    let mut img2 = attr.new_image(pixels.clone(), width, height, 0.).unwrap();
    let mut res2 = attr.quantize(&mut img2).unwrap();
    res2.set_dithering_level(1.).unwrap();
    res2.remapped(&mut img2).unwrap();
    let (_, floyd_default) = res2.remapped(&mut img2).unwrap();
    assert_eq!(floyd_default, floyd_explicit, "the default must remain Floyd-Steinberg");
}

#[test]
fn ordered_dithering_respects_dither_map() {
    use crate::DitheringMode;

    let width = 32;
    let height = 32;
    let pixels = gradient_image(width, height);
    let mut attr = crate::new();
    attr.set_max_colors(4).unwrap();

    let render = |map_value: u8| {
        // preset map prevents palette refinement, so both runs use exactly the same palette
        let mut img = attr.new_image(pixels.clone(), width, height, 0.).unwrap();
        img.dither_map = Some(vec![map_value; width * height].into_boxed_slice());
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        res.set_dithering_mode(DitheringMode::Ordered).unwrap();
        res.remapped(&mut img).unwrap().1
    };

    let no_dithering = render(0);
    let full_dithering = render(255);
    assert_ne!(no_dithering, full_dithering, "the edge/noise map must affect ordered dithering");
}

#[test]
fn ordered_dithering_alpha() {
    use crate::DitheringMode;
    use crate::RGBA;

    // Fully transparent and semi-transparent pixels must behave the same as without dithering
    let width = 16;
    let height = 16;
    let pixels: Vec<_> = (0..width * height).map(|n| {
        let x = (n % width) as u8;
        match x % 3 {
            0 => RGBA::new(10, 20, 30, 0),
            1 => RGBA::new(10, 20, 30, 128),
            _ => RGBA::new(x.wrapping_mul(9), 100, 200, 255),
        }
    }).collect();

    let mut attr = crate::new();
    attr.set_max_colors(16).unwrap();
    let mut img = attr.new_image(pixels, width, height, 0.).unwrap();
    let mut res = attr.quantize(&mut img).unwrap();
    res.set_dithering_level(0.).unwrap();
    let (_, plain) = res.remapped(&mut img).unwrap();
    res.set_dithering_level(1.).unwrap();
    res.set_dithering_mode(DitheringMode::Ordered).unwrap();
    let (pal, ordered) = res.remapped(&mut img).unwrap();

    let transparent = pal.iter().position(|c| c.a == 0).map(|i| i as u8);
    assert!(transparent.is_some(), "expected a fully transparent palette entry");
    let transparent = transparent.unwrap();
    let pal_a: Vec<_> = pal.iter().map(|c| c.a).collect();

    for (n, (&ordered_idx, &plain_idx)) in ordered.iter().zip(&plain).enumerate() {
        match n % width % 3 {
            // Fully transparent pixels must stay fully transparent, matching the non-dithered result
            0 => {
                assert_eq!(plain_idx, transparent, "non-dithered fully transparent pixel must be transparent");
                assert_eq!(ordered_idx, transparent, "ordered dithering must keep fully transparent pixels transparent");
            }
            // Semi-transparent pixels are dithered (they may change color), but must not become
            // fully transparent — the transparent index is reserved for real transparency
            1 => {
                assert_ne!(plain_idx, transparent, "non-dithered semi-transparent pixel must not be transparent");
                assert_ne!(pal_a[usize::from(ordered_idx)], 0, "semi-transparent pixel must not become fully transparent");
            }
            // Opaque pixels are dithered as usual
            _ => {
                assert_eq!(pal_a[usize::from(ordered_idx)], 255, "opaque pixel must stay opaque");
            }
        }
    }
}
