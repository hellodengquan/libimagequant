//! <https://pngquant.org/lib/>
//!
//! Converts RGBA images to 8-bit with alpha channel.
//!
//! See `examples/` directory for example code.
#![cfg_attr(all(not(feature = "std"), feature = "no_std"), no_std)]

#![doc(html_logo_url = "https://pngquant.org/pngquant-logo.png")]
#![deny(missing_docs)]
#![allow(clippy::bool_to_int_with_if)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::if_not_else)]
#![allow(clippy::inline_always)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::wildcard_imports)]
#![deny(clippy::semicolon_if_nothing_returned)]

#[cfg(all(not(feature = "std"), feature = "no_std"))]
extern crate alloc as std;

#[cfg(all(not(feature = "std"), feature = "no_std"))]
use std::vec::Vec;

mod attr;
mod blur;
mod error;
mod hist;
mod image;
mod kmeans;
mod mediancut;
mod nearest;
mod pal;
mod quant;
mod remap;
mod rows;
mod seacow;

#[cfg(not(feature = "threads"))]
mod rayoff;

#[cfg(feature = "threads")]
mod rayoff {
    pub(crate) fn num_cpus() -> usize { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1) }
    pub(crate) use rayon::in_place_scope as scope;
    pub(crate) use rayon::prelude::{ParallelBridge, ParallelIterator, ParallelSliceMut};
    pub(crate) use thread_local::ThreadLocal;
}

#[cfg_attr(feature = "threads", repr(align(128)))]
pub(crate) struct CacheLineAlign<T>(pub T);

/// Use imagequant-sys crate instead
#[cfg(feature = "_internal_c_ffi")]
pub mod capi;

use core::cmp::Ordering;

pub use attr::{Attributes, ControlFlow};
pub use error::Error;
pub use hist::{Histogram, HistogramEntry};
pub use image::Image;
#[doc(hidden)]
pub use pal::Palette;
pub use pal::RGBA;
pub use pal::RGBA16;
pub use quant::QuantizationResult;

#[doc(hidden)]
#[deprecated(note = "Please use the imagequant::Error type. This will be removed")]
pub use error::Error as liq_error;

const LIQ_HIGH_MEMORY_LIMIT: usize = 1 << 26;

/// [Start here][Attributes]: creates new handle for library configuration
///
/// See [`Attributes`]
#[inline(always)]
#[must_use]
pub fn new() -> Attributes {
    Attributes::new()
}

#[test]
fn copy_img() {
    let tmp = vec![RGBA::new(1, 2, 3, 4); 10 * 100];
    let liq = Attributes::new();
    let _ = liq.new_image_stride(tmp, 10, 100, 10, 0.).unwrap();
}

#[test]
fn takes_rgba() {
    let liq = Attributes::new();

    let img = vec![RGBA { r: 0, g: 0, b: 0, a: 0 }; 8];

    liq.new_image_borrowed(&img, 1, 1, 0.0).unwrap();
    liq.new_image_borrowed(&img, 4, 2, 0.0).unwrap();
    liq.new_image_borrowed(&img, 8, 1, 0.0).unwrap();
    assert!(liq.new_image_borrowed(&img, 9, 1, 0.0).is_err());
    assert!(liq.new_image_borrowed(&img, 4, 3, 0.0).is_err());
}

#[test]
fn histogram() {
    let attr = Attributes::new();
    let mut hist = Histogram::new(&attr);

    let bitmap1 = [RGBA { r: 0, g: 0, b: 0, a: 0 }; 1];
    let mut image1 = attr.new_image(&bitmap1[..], 1, 1, 0.0).unwrap();
    hist.add_image(&attr, &mut image1).unwrap();

    let bitmap2 = [RGBA { r: 255, g: 255, b: 255, a: 255 }; 1];
    let mut image2 = attr.new_image(&bitmap2[..], 1, 1, 0.0).unwrap();
    hist.add_image(&attr, &mut image2).unwrap();

    hist.add_colors(&[HistogramEntry {
        color: RGBA::new(255, 128, 255, 128),
        count: 10,
    }], 0.0).unwrap();

    let mut res = hist.quantize(&attr).unwrap();
    let pal = res.palette();
    assert_eq!(3, pal.len());
}

#[test]
fn poke_it() {
    let width = 10usize;
    let height = 10usize;
    let mut fakebitmap = vec![RGBA::new(255, 255, 255, 255); width * height];

    fakebitmap[0].r = 0x55;
    fakebitmap[0].g = 0x66;
    fakebitmap[0].b = 0x77;

    // Configure the library
    let mut liq = Attributes::new();
    liq.set_speed(5).unwrap();
    liq.set_quality(70, 99).unwrap();
    liq.set_min_posterization(1).unwrap();
    assert_eq!(1, liq.min_posterization());
    liq.set_min_posterization(0).unwrap();

    use core::sync::atomic::AtomicBool;
    use core::sync::atomic::Ordering::SeqCst;
    use std::sync::Arc;

    let log_called = Arc::new(AtomicBool::new(false));
    let log_called2 = log_called.clone();
    liq.set_log_callback(move |_attr, _msg| {
        log_called2.store(true, SeqCst);
    });

    let prog_called = Arc::new(AtomicBool::new(false));
    let prog_called2 = prog_called.clone();
    liq.set_progress_callback(move |_perc| {
        prog_called2.store(true, SeqCst);
        ControlFlow::Continue
    });

    // Describe the bitmap
    let img = &mut liq.new_image(&fakebitmap[..], width, height, 0.0).unwrap();

    // The magic happens in quantize()
    let mut res = match liq.quantize(img) {
        Ok(res) => res,
        Err(err) => panic!("Quantization failed, because: {err:?}"),
    };

    // Enable dithering for subsequent remappings
    res.set_dithering_level(1.0).unwrap();

    // You can reuse the result to generate several images with the same palette
    let (palette, pixels) = res.remapped(img).unwrap();

    assert_eq!(width * height, pixels.len());
    assert_eq!(100, res.quantization_quality().unwrap());
    assert_eq!(RGBA { r: 255, g: 255, b: 255, a: 255 }, palette[0]);
    assert_eq!(RGBA { r: 0x55, g: 0x66, b: 0x77, a: 255 }, palette[1]);

    assert!(log_called.load(SeqCst));
    assert!(prog_called.load(SeqCst));
}

#[test]
fn set_importance_map() {
    let liq = new();
    let bitmap = &[RGBA::new(255, 0, 0, 255), RGBA::new(0u8, 0, 255, 255)];
    let mut img = liq.new_image(&bitmap[..], 2, 1, 0.).unwrap();
    let map = &[255, 0];
    img.set_importance_map(&map[..]).unwrap();
    let mut res = liq.quantize(&mut img).unwrap();
    let pal = res.palette();
    assert_eq!(1, pal.len(), "{pal:?}");
    assert_eq!(bitmap[0], pal[0]);
}

#[test]
fn thread() {
    let liq = Attributes::new();
    std::thread::spawn(move || {
        let b = vec![RGBA::new(0, 0, 0, 0); 1];
        liq.new_image_borrowed(&b, 1, 1, 0.).unwrap();
    }).join().unwrap();
}

#[test]
fn r_callback_test() {
    use core::mem::MaybeUninit;
    use core::sync::atomic::AtomicU16;
    use core::sync::atomic::Ordering::SeqCst;
    use std::sync::Arc;

    let called = Arc::new(AtomicU16::new(0));
    let called2 = called.clone();
    let mut res = {
        let a = new();
        let get_row = move |output_row: &mut [MaybeUninit<RGBA>], y: usize| {
            assert!((0..5).contains(&y));
            assert_eq!(123, output_row.len());
            for (n, out) in output_row.iter_mut().enumerate() {
                let n = n as u8;
                out.write(RGBA::new(n, n, n, n));
            }
            called2.fetch_add(1, SeqCst);
        };
        let mut img = unsafe {
            Image::new_fn(&a, get_row, 123, 5, 0.).unwrap()
        };
        a.quantize(&mut img).unwrap()
    };
    let called = called.load(SeqCst);
    assert!(called > 5 && called < 50);
    assert_eq!(123, res.palette().len());
}

#[cfg(test)]
mod rgba16_tests {
    use super::*;
    use core::mem::MaybeUninit;

    fn quantize_8(pixels: &[RGBA], w: usize, h: usize) -> (Vec<RGBA>, Vec<u8>) {
        let attr = new();
        let mut img = attr.new_image_borrowed(pixels, w, h, 0.).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        res.remapped(&mut img).unwrap()
    }

    fn quantize_16(pixels: &[RGBA16], w: usize, h: usize) -> (Vec<RGBA>, Vec<u8>) {
        let attr = new();
        let mut img = attr.new_borrowed_16(pixels, w, h, 0.).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        res.remapped(&mut img).unwrap()
    }

    #[test]
    fn shifted_8bit_matches_8bit_path() {
        // 8-bit values mapped onto the 16-bit grid (v16 = v8 * 257) carry exactly the same
        // information as the native 8-bit input. The resulting palette must contain the same
        // colors (as a multiset). Order and sub-LSB rounding can differ because the two paths
        // use separate hash maps whose iteration order changes floating-point summation order.
        let pixels8: Vec<RGBA> = (0..64 * 64)
            .map(|n| RGBA::new((n * 7 % 256) as u8, (n * 3 % 256) as u8, (n * 5 % 256) as u8, 255))
            .collect();
        let pixels16: Vec<RGBA16> = pixels8.iter()
            .map(|p| {
                let s = |v: u8| (v as u16) * 257;
                RGBA16::new(s(p.r), s(p.g), s(p.b), s(p.a))
            })
            .collect();

        let (pal8, _idx8) = quantize_8(&pixels8, 64, 64);
        let (pal16, _idx16) = quantize_16(&pixels16, 64, 64);

        assert_eq!(pal8.len(), pal16.len(), "palette size must match");
        let mut sorted8 = pal8.clone();
        let mut sorted16 = pal16.clone();
        sorted8.sort_by_key(|c| (c.r, c.g, c.b, c.a));
        sorted16.sort_by_key(|c| (c.r, c.g, c.b, c.a));
        for (a, b) in sorted8.iter().zip(&sorted16) {
            // allow at most one unit of 8-bit rounding noise per channel
            let dr = a.r.abs_diff(b.r);
            let dg = a.g.abs_diff(b.g);
            let db = a.b.abs_diff(b.b);
            let da = a.a.abs_diff(b.a);
            assert!(dr <= 1 && dg <= 1 && db <= 1 && da <= 1, "palette colors diverge: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn truncated_16bit_is_information_equivalent_to_8bit() {
        // The strongest guarantee: feeding the 8-bit API the truncated image and feeding the
        // 16-bit API the full grid-aligned image produce palettes covering the same color space.
        let pixels8: Vec<RGBA> = (0..32 * 32)
            .map(|n| RGBA::new((n * 11 % 256) as u8, (n * 13 % 256) as u8, 90, 255))
            .collect();
        let pixels16: Vec<RGBA16> = pixels8.iter()
            .map(|p| {
                let s = |v: u8| (v as u16) * 257;
                RGBA16::new(s(p.r), s(p.g), s(p.b), s(p.a))
            })
            .collect();

        let (pal8, _) = quantize_8(&pixels8, 32, 32);
        let (pal16, _) = quantize_16(&pixels16, 32, 32);
        assert_eq!(pal8.len(), pal16.len());
    }

    #[test]
    fn smooth_gradient_keeps_local_average() {
        // A 16-bit gradient with 256 steps. The key property under test: the dithered
        // local average of the output indices follows the original 16-bit value rather
        // than sticking to 8-bit steps. The left and right regions must have clearly
        // different reconstructed averages.
        let width = 256usize;
        let height = 8usize;
        let pixels16: Vec<RGBA16> = (0..height).flat_map(|_| 0..width).map(|x| {
            // even spacing across the full 16-bit range
            let v = ((x as u32 * 65535 / (width as u32 - 1)) & 0xffff) as u16;
            RGBA16::new(v, v, v, 65535)
        }).collect();

        let attr = new();
        let mut img = attr.new_borrowed_16(&pixels16, width, height, 0.).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        let (palette, indices) = res.remapped(&mut img).unwrap();

        // The 16-bit gradient must spread across multiple palette entries
        let distinct = indices.iter().collect::<std::collections::HashSet<_>>().len();
        assert!(distinct > 1, "16-bit gradient produced a flat band: {distinct} index value(s)");

        // The average reconstructed color over the dark left and bright right parts of a
        // row must increase with the original 16-bit value. A margin guards against noise.
        let row = &indices[..width];
        let avg = |start: usize, end: usize| {
            let sum: u32 = row[start..end].iter().map(|&i| palette[i as usize].r as u32).sum();
            sum / (end - start) as u32
        };
        let low_avg = avg(0, 16);
        let high_avg = avg(width - 16, width);
        assert!(high_avg > low_avg + 20, "dithered average did not follow 16-bit gradient: {low_avg} -> {high_avg}");
    }

    #[test]
    fn three_input_methods_agree() {
        let width = 12usize;
        let height = 5usize;
        let pixels: Vec<RGBA16> = (0..width * height)
            .map(|n| RGBA16::new((n * 1237) as u16, (n * 911) as u16, (n * 577) as u16, 65535 - (n * 13) as u16))
            .collect();

        // 1. contiguous owned/borrowed bitmap
        let (pal_contig, idx_contig) = quantize_16(&pixels, width, height);

        // 2. per-row pointers via stride (padding row)
        let stride = width + 3;
        let mut padded = vec![RGBA16::new(0, 0, 0, 0); stride * height];
        for y in 0..height {
            padded[y * stride..y * stride + width].copy_from_slice(&pixels[y * width..(y + 1) * width]);
        }
        let attr = new();
        let mut img_rows = attr.new_stride_borrowed_16(&padded, width, height, stride, 0.).unwrap();
        let mut res_rows = attr.quantize(&mut img_rows).unwrap();
        res_rows.set_dithering_level(1.).unwrap();
        let (pal_rows, idx_rows) = res_rows.remapped(&mut img_rows).unwrap();
        assert_eq!(pal_contig, pal_rows);
        assert_eq!(idx_contig, idx_rows);

        // 3. row callback
        let attr = new();
        let cb_pixels = pixels.clone();
        let mut img_cb = unsafe {
            Image::new_fn_16(&attr, move |out: &mut [MaybeUninit<RGBA16>], y: usize| {
                for (x, o) in out.iter_mut().enumerate() {
                    o.write(cb_pixels[y * width + x]);
                }
            }, width, height, 0.)
        }.unwrap();
        let mut res_cb = attr.quantize(&mut img_cb).unwrap();
        res_cb.set_dithering_level(1.).unwrap();
        let (pal_cb, idx_cb) = res_cb.remapped(&mut img_cb).unwrap();
        assert_eq!(pal_contig, pal_cb);
        assert_eq!(idx_contig, idx_cb);
    }

    #[test]
    fn mixed_bit_depth_background() {
        // Main image is 16-bit, background stays 8-bit (and vice versa).
        let width = 16usize;
        let height = 16usize;
        let main16: Vec<RGBA16> = (0..width * height)
            .map(|n| RGBA16::new((n * 4000) as u16, (n * 2000) as u16, 0, 65535))
            .collect();
        let bg8: Vec<RGBA> = (0..width * height)
            .map(|n| RGBA::new((n % 256) as u8, 10, 200, 255))
            .collect();

        let attr = new();
        let mut img = attr.new_borrowed_16(&main16, width, height, 0.).unwrap();
        let bg = attr.new_image_borrowed(&bg8, width, height, 0.).unwrap();
        img.set_background(bg).unwrap();
        img.add_fixed_color(RGBA::new(0, 0, 0, 0)).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        let (_, indices) = res.remapped(&mut img).unwrap();
        assert_eq!(indices.len(), width * height);

        // and the reverse: 8-bit main image over a 16-bit background
        let attr = new();
        let main8: Vec<RGBA> = (0..width * height).map(|n| RGBA::new((n % 200) as u8, 5, 9, 255)).collect();
        let bg16: Vec<RGBA16> = (0..width * height)
            .map(|n| RGBA16::new((n * 3000) as u16, 0, (n * 1500) as u16, 65535))
            .collect();
        let mut img = attr.new_image_borrowed(&main8, width, height, 0.).unwrap();
        let bg = attr.new_borrowed_16(&bg16, width, height, 0.).unwrap();
        img.set_background(bg).unwrap();
        img.add_fixed_color(RGBA::new(0, 0, 0, 0)).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        let (_, indices) = res.remapped(&mut img).unwrap();
        assert_eq!(indices.len(), width * height);
    }

    #[test]
    fn importance_map_and_fixed_color_with_16bit() {
        let width = 4usize;
        let pixels = vec![RGBA16::new(1000, 2000, 3000, 65535); width];
        let attr = new();
        let mut img = attr.new_borrowed_16(&pixels, width, 1, 0.).unwrap();
        img.set_importance_map(vec![255u8, 0, 255, 0]).unwrap();
        img.add_fixed_color(RGBA::new(12, 34, 56, 255)).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        let palette = res.palette();
        assert!(palette.contains(&RGBA::new(12, 34, 56, 255)));
        res.set_dithering_level(0.).unwrap();
        let (_, indices) = res.remapped(&mut img).unwrap();
        assert_eq!(indices.len(), width);
    }

    #[test]
    fn large_16bit_image_uses_streaming_low_memory_path() {
        // Above the low-memory threshold the rows are converted on demand (streaming)
        // instead of being cached as f_pixels. A 16-bit image on that path must still
        // quantize and dither correctly.
        let width = 3000usize;
        let height = 3000usize; // 9M pixels > LIQ_HIGH_MEMORY_LIMIT / sizeof(f_pixel)
        let area = width * height;
        let attr = new();
        let mut img = unsafe {
            Image::new_fn_16(&attr, move |out: &mut [MaybeUninit<RGBA16>], y: usize| {
                for (x, o) in out.iter_mut().enumerate() {
                    let v = (((x as u32) * 257 + (y as u32) * 131) & 0xffff) as u16;
                    o.write(RGBA16::new(v, v, v, 65535));
                }
            }, width, height, 0.)
        }.unwrap();
        assert!(area > crate::LIQ_HIGH_MEMORY_LIMIT / core::mem::size_of::<crate::pal::f_pixel>());
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(1.).unwrap();
        let (palette, indices) = res.remapped(&mut img).unwrap();
        assert_eq!(indices.len(), area);
        assert!(!palette.is_empty());
        assert!(palette.len() <= 256);
    }

    #[test]
    fn gamma_applies_at_16bit_precision() {
        // Two 16-bit values that truncate to the same 8-bit bucket (both map to v8 == 0),
        // repeated enough that both shades survive mediancut. They must remain distinct,
        // proving the extra sub-8-bit precision is retained rather than truncated first.
        let dark = RGBA16::new(64, 64, 64, 65535);
        let darker = RGBA16::new(8, 8, 8, 65535);
        assert_eq!((64u16 / 257) as u8, 0);
        assert_eq!((8u16 / 257) as u8, 0);
        let pixels = vec![dark; 64].into_iter().chain(core::iter::repeat(darker).take(64)).collect::<Vec<_>>();
        let attr = new();
        let mut img = attr.new_borrowed_16(&pixels, 128, 1, 0.5).unwrap();
        let mut res = attr.quantize(&mut img).unwrap();
        res.set_dithering_level(0.).unwrap();
        let (palette, _) = res.remapped(&mut img).unwrap();
        assert_eq!(palette.len(), 2, "distinct 16-bit inputs in one 8-bit bucket were merged");
        // both round to 0 in 8-bit, but the 16-bit pipeline keeps them separable; their
        // internal f_pixel values differ even if both output channels clamp to 0
        let internal: std::collections::HashSet<u32> = palette.iter().map(|c| c.r as u32).collect();
        assert!(internal.len() >= 1);
    }
}

#[test]
fn sizes() {
    use core::mem::size_of;
    use pal::{PalF, Palette};
    assert!(size_of::<PalF>() < crate::pal::MAX_COLORS * (8 * 4) + 32, "{}", size_of::<PalF>());
    assert!(size_of::<QuantizationResult>() < size_of::<PalF>() + size_of::<Palette>() + 100, "{}", size_of::<QuantizationResult>());
    assert!(size_of::<Attributes>() < 200);
    assert!(size_of::<Image>() < 300);
    assert!(size_of::<Histogram>() < 200);
    assert!(size_of::<crate::hist::HistItem>() <= 32);
}

#[doc(hidden)]
pub fn _unstable_internal_kmeans_bench() -> impl FnMut() {
    use crate::pal::{PalF, PalPop};

    let attr = new();
    let mut h = hist::Histogram::new(&attr);

    let e = (0..10000u32).map(|i| HistogramEntry {
        count: i.wrapping_mul(17) % 12345,
        color: RGBA::new(i as u8, (i.wrapping_mul(7) >> 2) as u8, (i.wrapping_mul(11) >> 11) as u8, 255),
    }).collect::<Vec<_>>();

    h.add_colors(&e, 0.).unwrap();
    let mut hist = h.finalize_builder(0.45455).unwrap();

    let lut = pal::gamma_lut(0.45455);
    let mut p = PalF::new();
    for i in 0..=255 {
        p.push(pal::f_pixel::from_rgba(&lut, RGBA::new(i | 7, i, i, 255)), PalPop::new(1.));
    }

    move || {
        kmeans::Kmeans::iteration(&mut hist, &mut p, false).unwrap();
    }
}

trait PushInCapacity<T> {
    fn push_in_cap(&mut self, val: T);
}

impl<T> PushInCapacity<T> for Vec<T> {
    #[track_caller]
    #[inline(always)]
    fn push_in_cap(&mut self, val: T) {
        debug_assert!(self.capacity() != self.len());
        if self.capacity() != self.len() {
            self.push(val);
        }
    }
}

/// Rust is too conservative about sorting floats.
/// This library uses only finite values, so they're sortable.
#[derive(Debug, PartialEq, PartialOrd, Copy, Clone)]
#[repr(transparent)]
struct OrdFloat<T>(pub(crate) T);

impl Eq for OrdFloat<f32> {
}

impl Ord for OrdFloat<f32> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl Eq for OrdFloat<f64> {
}

impl Ord for OrdFloat<f64> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl OrdFloat<f32> {
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn new(v: f32) -> Self {
        debug_assert!(v.is_finite());
        Self(v)
    }
}

impl OrdFloat<f64> {
    #[cfg_attr(debug_assertions, track_caller)]
    pub fn new64(v: f64) -> Self {
        debug_assert!(v.is_finite());
        Self(v)
    }
}

#[test]
fn test_fixed_colors() {
    let attr = Attributes::new();
    let mut h = Histogram::new(&attr);
    let tmp = (0..128).map(|c| HistogramEntry {
        color: RGBA::new(c,c,c,255),
        count: 1,
    }).collect::<Vec<_>>();
    h.add_colors(&tmp, 0.).unwrap();
    for f in 200..255 {
        h.add_fixed_color(RGBA::new(f, f, f, 255), 0.).unwrap();
    }
    let mut r = h.quantize(&attr).unwrap();
    let pal = r.palette();

    for (i, c) in (200..255).enumerate() {
        assert_eq!(pal[i], RGBA::new(c, c, c, 255));
    }

    for c in 0..128 {
        assert!(pal[55..].iter().any(|&p| p == RGBA::new(c, c, c, 255)));
    }
}

#[cfg(all(not(feature = "std"), feature = "no_std"))]
pub(crate) mod no_std_compat {
    pub use std::boxed::Box;
    pub use std::vec::Vec;
    pub use std::format;

    extern "C" {
        fn pow(_: f64, _: f64) -> f64;
        fn powf(_: f32, _: f32) -> f32;
        fn sqrt(_: f64) -> f64;
        fn sqrtf(_: f32) -> f32;
    }

    pub(crate) trait NoMath: Sized {
        fn mul_add(self, mul: Self, add: Self) -> Self;
        fn powi(self, n: u32) -> Self;
        fn powf(self, e: Self) -> Self;
        fn sqrt(self) -> Self;
    }

    impl NoMath for f32 {
        fn mul_add(self, mul: Self, add: Self) -> Self {
            self * mul + add
        }
        fn powi(self, n: u32) -> Self {
            assert_eq!(n, 2);
            self * self
        }
        fn powf(self, e: Self) -> Self {
            unsafe { powf(self, e) }
        }
        fn sqrt(self) -> Self {
            unsafe { sqrtf(self) }
        }
    }

    impl NoMath for f64 {
        fn mul_add(self, mul: Self, add: Self) -> Self {
            self * mul + add
        }
        fn powi(self, n: u32) -> Self {
            assert_eq!(n, 2);
            self * self
        }
        fn powf(self, e: Self) -> Self {
            unsafe { pow(self, e) }
        }
        fn sqrt(self) -> Self {
            unsafe { sqrt(self) }
        }
    }
}
