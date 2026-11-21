use crate::error::*;
use crate::image::Image;
use crate::pal::{f_pixel, gamma_lut, gamma_lut16, PalIndex, ARGBF, MAX_COLORS, RGBA, RGBA16};
use crate::quant::QuantizationResult;
use crate::rows::{temp_buf, DynamicRows};
use crate::Attributes;
use core::{fmt, hash, mem};
use core::hash::Hash;

#[cfg(all(not(feature = "std"), feature = "no_std"))]
use std::{format, boxed::Box, vec::Vec};

#[cfg(all(not(feature = "std"), feature = "no_std"))]
use hashbrown::{HashMap, HashSet};

#[cfg(not(all(not(feature = "std"), feature = "no_std")))]
use std::collections::{HashMap, HashSet};


/// Number of pixels in a given color for [`Histogram::add_colors()`]
///
/// Used for building a histogram manually. Otherwise see [`Histogram::add_image()`]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct HistogramEntry {
    /// The color
    pub color: RGBA,
    /// Importance of the color (e.g. number of occurrences)
    pub count: u32,
}

/// Generate one shared palette for multiple images
///
/// If you're converting one image at a time, see [`Attributes::new_image`] instead
pub struct Histogram {
    gamma: Option<f64>,
    fixed_colors: FixedColorsSet,

    /// The key is the RGBA cast to u32
    /// The value is a (boosted) count or 0 if it's a fixed color
    hashmap: HashMap<u32, (u32, RGBA), U32Hasher>,

    /// 16-bit-per-channel colors. The key packs four u16 channels into a u64.
    /// Kept separate so 8-bit inputs keep their exact original behavior.
    hashmap16: HashMap<u64, (u32, RGBA16), U32Hasher>,

    posterize_bits: u8,
    max_histogram_entries: u32,
}

pub(crate) type FixedColorsSet = HashSet<HashColor, U32Hasher>;

#[derive(Clone)]
pub(crate) struct HistItem {
    pub color: f_pixel,
    pub adjusted_weight: f32,
    pub perceptual_weight: f32,
    pub mc_color_weight: f32,
    /// Reused: mc_sort_value during median cut, then likely_palette_index after
    pub sort_val_pal_index_union: u32,
}

impl HistItem {
    #[inline(always)]
    pub fn mc_sort_value(&self) -> u32 {
        self.sort_val_pal_index_union
    }

    #[inline(always)]
    pub fn likely_palette_index(&self) -> PalIndex {
        self.sort_val_pal_index_union as PalIndex
    }

    pub(crate) fn set_likely_palette_index(&mut self, index: PalIndex) {
        self.sort_val_pal_index_union = index as u32;
    }
}

impl fmt::Debug for HistItem {
    #[cold]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HistItem")
            .field("color", &self.color)
            .field("adjusted_weight", &self.adjusted_weight)
            .field("perceptual_weight", &self.perceptual_weight)
            .field("color_weight", &self.mc_color_weight)
            .finish()
    }
}

impl Histogram {
    /// Creates histogram object that will be used to collect color statistics from multiple images.
    ///
    /// All options should be set on `attr` before the histogram object is created. Options changed later may not have effect.
    #[inline]
    #[must_use]
    pub fn new(attr: &Attributes) -> Self {
        Self {
            posterize_bits: attr.posterize_bits(),
            max_histogram_entries: attr.max_histogram_entries,
            fixed_colors: HashSet::with_hasher(U32Hasher(0)),
            hashmap: HashMap::with_hasher(U32Hasher(0)),
            hashmap16: HashMap::with_hasher(U32Hasher(0)),
            gamma: None,
        }
    }

    /// "Learns" colors from the image, which will be later used to generate the palette.
    ///
    /// Fixed colors added to the image are also added to the histogram. If the total number of fixed colors exceeds 256,
    /// this function will fail with `LIQ_BUFFER_TOO_SMALL`.
    #[inline(never)]
    pub fn add_image(&mut self, attr: &Attributes, image: &mut Image) -> Result<(), Error> {
        let width = image.width();
        let height = image.height();
        if image.importance_map.is_none() && attr.use_contrast_maps {
            image.contrast_maps()?;
        }

        self.gamma = image.gamma();

        if !image.fixed_colors.is_empty() {
            self.fixed_colors.extend(image.fixed_colors.iter().copied().enumerate().map(|(idx, rgba)| {
               HashColor { rgba, index: idx as _ }
            }));
        }

        if attr.progress(f32::from(attr.progress_stage1) * 0.40) {
            return Err(Aborted); // bow can free the RGBA source if copy has been made in f_pixels
        }

        let posterize_bits = attr.posterize_bits();
        let surface_area = height * width;
        let estimated_colors = (surface_area / (posterize_bits as usize + if surface_area > 512 * 512 { 7 } else { 5 })).min(250_000);
        self.reserve(estimated_colors);

        if image.px.is_16bit() {
            self.add_pixel_rows16(&image.px, image.importance_map.as_deref(), posterize_bits)?;
        } else {
            self.add_pixel_rows(&image.px, image.importance_map.as_deref(), posterize_bits)?;
        }

        Ok(())
    }

    /// Alternative to `add_image()`. Intead of counting colors in an image, it directly takes an array of colors and their counts.
    ///
    /// This function is only useful if you already have a histogram of the image from another source.
    ///
    /// The gamma may be 0 to mean sRGB. All calls to `add_colors` and `add_fixed_color` should use the same gamma value.
    #[inline(never)]
    pub fn add_colors(&mut self, entries: &[HistogramEntry], gamma: f64) -> Result<(), Error> {
        if entries.is_empty() || entries.len() > 1 << 24 {
            return Err(ValueOutOfRange);
        }

        if !(0. ..1.).contains(&gamma) {
            return Err(ValueOutOfRange);
        }

        if self.gamma.is_none() && gamma > 0. {
            self.gamma = Some(gamma);
        }

        self.reserve(entries.len());

        for e in entries {
            self.add_color(e.color, e.count);
        }

        Ok(())
    }

    /// Add a color guaranteed to be in the final palette
    ///
    /// The gamma may be 0 to mean sRGB. All calls to `add_colors` and `add_fixed_color` should use the same gamma value.
    pub fn add_fixed_color(&mut self, rgba: RGBA, gamma: f64) -> Result<(), Error> {
        if self.fixed_colors.len() >= MAX_COLORS {
            return Err(Unsupported);
        }

        if self.gamma.is_none() && gamma > 0. {
            self.gamma = Some(gamma);
        }

        let idx = self.fixed_colors.len();
        self.fixed_colors.insert(HashColor { rgba, index: idx as _ });

        Ok(())
    }

    /// Generate palette for all images/colors added to the histogram.
    ///
    /// Palette generated using this function won't be improved during remapping.
    /// If you're generating palette for only one image, it's better not to use the `Histogram`.
    #[inline]
    pub fn quantize(&mut self, attr: &Attributes) -> Result<QuantizationResult, Error> {
        self.quantize_internal(attr, true)
    }

    #[inline(never)]
    pub(crate) fn quantize_internal(&mut self, attr: &Attributes, freeze_result_colors: bool) -> Result<QuantizationResult, Error> {
        if self.hashmap.is_empty() && self.hashmap16.is_empty() && self.fixed_colors.is_empty() {
            return Err(Unsupported);
        }

        if attr.progress(0.) { return Err(Aborted); }
        if attr.progress(f32::from(attr.progress_stage1) * 0.89) {
            return Err(Aborted);
        }

        let gamma = self.gamma.unwrap_or(0.45455);
        let hist = self.finalize_builder(gamma).map_err(|_| OutOfMemory)?;

        attr.verbose_print(format!("  made histogram...{} colors found", hist.items.len()));

        QuantizationResult::new(attr, hist, freeze_result_colors, gamma)
    }

    #[inline(always)]
    fn add_color(&mut self, rgba: RGBA, boost: u32) {
        if boost == 0 {
            return;
        }

        let px_int = if rgba.a != 0 {
            self.posterize_mask() & unsafe { RGBAInt { rgba }.int }
        } else { 0 };

        self.hashmap.entry(px_int)
            // it can overflow on images over 2^24 pixels large
            .and_modify(move |e| e.0 = e.0.saturating_add(boost))
            .or_insert((boost, rgba));
    }

    #[inline(always)]
    fn add_color16(&mut self, rgba: RGBA16, boost: u32) {
        if boost == 0 {
            return;
        }

        let px_int = if rgba.a != 0 {
            self.posterize_mask16() & rgba16_key(rgba)
        } else { 0 };

        self.hashmap16.entry(px_int)
            .and_modify(move |e| e.0 = e.0.saturating_add(boost))
            .or_insert((boost, rgba));
    }

    fn reserve(&mut self, entries: usize) {
        let new_entries = entries.saturating_sub(self.hashmap.len() / 3); // assume some will be dupes, if called multiple times
        self.hashmap.reserve(new_entries);
        let new_entries16 = entries.saturating_sub(self.hashmap16.len() / 3);
        self.hashmap16.reserve(new_entries16);
    }

    #[inline(always)]
    const fn posterize_mask(&self) -> u32 {
        let channel_mask = 255 << self.posterize_bits;
        u32::from_ne_bytes([channel_mask, channel_mask, channel_mask, channel_mask])
    }

    /// Posterization mask for four 16-bit channels packed in a u64.
    #[inline(always)]
    const fn posterize_mask16(&self) -> u64 {
        let channel_mask = (65535u64) << self.posterize_bits;
        channel_mask | (channel_mask << 16) | (channel_mask << 32) | (channel_mask << 48)
    }

    /// optionallys et
    fn init_posterize_bits(&mut self, posterize_bits: u8) {
        if self.posterize_bits >= posterize_bits {
            return;
        }
        self.posterize_bits = posterize_bits;
        let new_posterize_mask = self.posterize_mask();

        let new_size = (self.hashmap.len() / 3).max(self.hashmap.capacity() / 5);
        let old_hashmap = mem::replace(&mut self.hashmap, HashMap::with_capacity_and_hasher(new_size, U32Hasher(0)));
        self.hashmap.extend(old_hashmap.into_iter().map(move |(k, v)| {
            (k & new_posterize_mask, v)
        }));

        let new_posterize_mask16 = self.posterize_mask16();
        let new_size16 = (self.hashmap16.len() / 3).max(self.hashmap16.capacity() / 5);
        let old_hashmap16 = mem::replace(&mut self.hashmap16, HashMap::with_capacity_and_hasher(new_size16, U32Hasher(0)));
        self.hashmap16.extend(old_hashmap16.into_iter().map(move |(k, v)| {
            (k & new_posterize_mask16, v)
        }));
    }

    pub(crate) fn add_pixel_rows(&mut self, image: &DynamicRows<'_, '_>, importance_map: Option<&[u8]>, posterize_bits: u8) -> Result<(), Error> {
        let width = image.width as usize;
        let height = image.height as usize;

        debug_assert!(importance_map.map_or(true, |m| m.len() == image.width() * image.height()));

        let mut importance_map = importance_map.unwrap_or(&[]).chunks_exact(width).fuse();
        let image_iter = image.rgba_rows_iter()?;

        let mut temp_row = temp_buf(width)?;
        for row in 0..height {
            let pixels_row = &image_iter.row_rgba(&mut temp_row, row)[..width];
            let importance_map = importance_map.next().map(move |m| &m[..width]).unwrap_or(&[]);
            for (col, px) in pixels_row.iter().copied().enumerate() {
                let boost = importance_map.get(col).copied().unwrap_or(255);
                self.add_color(px, boost.into());
            }
        }
        self.init_posterize_bits(posterize_bits);

        if self.hashmap.len() > self.max_histogram_entries as usize && self.posterize_bits < 3 {
            self.init_posterize_bits(self.posterize_bits + 1);
        }
        Ok(())
    }

    pub(crate) fn add_pixel_rows16(&mut self, image: &DynamicRows<'_, '_>, importance_map: Option<&[u8]>, posterize_bits: u8) -> Result<(), Error> {
        let width = image.width as usize;
        let height = image.height as usize;

        debug_assert!(importance_map.map_or(true, |m| m.len() == image.width() * image.height()));
        debug_assert!(image.is_16bit());

        let mut importance_map = importance_map.unwrap_or(&[]).chunks_exact(width).fuse();
        let image_iter = image.rgba_rows_iter()?;

        let mut temp_row = temp_buf(width)?;
        for row in 0..height {
            let pixels_row = &image_iter.row_rgba16(&mut temp_row, row)[..width];
            let importance_map = importance_map.next().map(move |m| &m[..width]).unwrap_or(&[]);
            for (col, px) in pixels_row.iter().copied().enumerate() {
                let boost = importance_map.get(col).copied().unwrap_or(255);
                self.add_color16(px, boost.into());
            }
        }
        self.init_posterize_bits(posterize_bits);

        if self.hashmap16.len() > self.max_histogram_entries as usize && self.posterize_bits < 3 {
            self.init_posterize_bits(self.posterize_bits + 1);
        }
        Ok(())
    }

    pub(crate) fn finalize_builder(&mut self, gamma: f64) -> Result<HistogramInternal, Error> {
        debug_assert!(gamma > 0.);

        // Fixed colors will be put into normal hashmap, but with very high weight,
        // and temporarily 0 means this fixed max weight
        for &HashColor { rgba, .. } in &self.fixed_colors {
            let px_int = if rgba.a != 0 {
                unsafe { RGBAInt { rgba }.int }
            } else { 0 };

            self.hashmap.insert(px_int, (0, rgba));
        }

        let total_entries = self.hashmap.len() + self.hashmap16.len();
        let mut temp = Vec::new();
        temp.try_reserve_exact(total_entries)?;

        let mut counts = [0; LIQ_MAXCLUSTER];
        temp.extend(self.hashmap.values().map(|&(boost, color)| {
            let cluster_index = ((color.r >> 7) << 3) | ((color.g >> 7) << 2) | ((color.b >> 7) << 1) | (color.a >> 7);
            counts[cluster_index as usize] += 1;

            // fixed colors result in weight == 0.
            let weight = boost as f32;
            TempHistItem::Bit8 { color, weight, cluster_index }
        }));
        temp.extend(self.hashmap16.values().map(|&(boost, color)| {
            let cluster_index = (((color.r >> 15) << 3) | ((color.g >> 15) << 2) | ((color.b >> 15) << 1) | (color.a >> 15)) as u8;
            counts[cluster_index as usize] += 1;

            let weight = boost as f32;
            TempHistItem::Bit16 { color, weight, cluster_index }
        }));

        let mut clusters = [Cluster { begin: 0, end: 0 }; LIQ_MAXCLUSTER];
        let mut next_begin = 0;
        for (cluster, count) in clusters.iter_mut().zip(counts) {
            cluster.begin = next_begin;
            cluster.end = next_begin;
            next_begin += count;
        }

        let mut items = Vec::new();
        items.try_reserve_exact(temp.len())?;
        items.resize(temp.len(), HistItem {
            color: if cfg!(debug_assertions) { f_pixel( ARGBF { r:f32::NAN, g:f32::NAN, b:f32::NAN, a:f32::NAN } ) } else { f_pixel::default() },
            adjusted_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            perceptual_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            mc_color_weight: if cfg!(debug_assertions) { f32::NAN } else { 0. },
            sort_val_pal_index_union: if cfg!(debug_assertions) { !0 } else { 0 },
        });
        let mut items = items.into_boxed_slice();

        // Limit perceptual weight to 1/10th of the image surface area to prevent
        // a single color from dominating all others.
        let max_perceptual_weight = ((0.1 / 255.) * temp.iter().map(|t| f64::from(t.weight())).sum::<f64>()) as f32;

        // The 16-bit table is large, so build it lazily only when a 16-bit color is present
        let lut = gamma_lut(gamma);
        let mut lut16 = None;
        let mut total_perceptual_weight = 0.;
        for temp_item in temp {
            let cluster = &mut clusters[temp_item.cluster_index() as usize];
            let next_index = cluster.end as usize;
            cluster.end += 1;

            // weight == 0 means it's a fixed color
            let item_weight = temp_item.weight();
            let weight = if item_weight > 0. {
                (item_weight * (1. / 255.)).min(max_perceptual_weight)
            } else {
                max_perceptual_weight * 16.
            };
            total_perceptual_weight += f64::from(weight);

            items[next_index].color = match temp_item {
                TempHistItem::Bit8 { color, .. } => f_pixel::from_rgba(&lut, color),
                TempHistItem::Bit16 { color, .. } => {
                    let lut16 = lut16.get_or_insert_with(|| gamma_lut16(gamma));
                    f_pixel::from_rgba16(lut16, color)
                },
            };
            items[next_index].perceptual_weight = weight;
            items[next_index].adjusted_weight = weight;
        }

        let mut fixed_colors: Vec<_> = self.fixed_colors.iter().collect();
        fixed_colors.sort_by_key(|c| c.index); // original order
        let fixed_colors = fixed_colors.iter().map(|c| f_pixel::from_rgba(&lut, c.rgba)).collect();

        Ok(HistogramInternal { items, total_perceptual_weight, clusters, fixed_colors })
    }
}

#[derive(Copy, Clone)]
enum TempHistItem {
    Bit8 {
        color: RGBA,
        weight: f32,
        cluster_index: u8,
    },
    Bit16 {
        color: RGBA16,
        weight: f32,
        cluster_index: u8,
    },
}

impl TempHistItem {
    #[inline(always)]
    const fn weight(&self) -> f32 {
        match self {
            Self::Bit8 { weight, .. } | Self::Bit16 { weight, .. } => *weight,
        }
    }

    #[inline(always)]
    const fn cluster_index(&self) -> u8 {
        match self {
            Self::Bit8 { cluster_index, .. } | Self::Bit16 { cluster_index, .. } => *cluster_index,
        }
    }
}

/// Packs four 16-bit channels into a platform-independent u64 key.
#[inline(always)]
pub(crate) fn rgba16_key(px: RGBA16) -> u64 {
    (u64::from(px.r)) | (u64::from(px.g) << 16) | (u64::from(px.b) << 32) | (u64::from(px.a) << 48)
}

#[repr(C)]
union RGBAInt {
    rgba: RGBA,
    int: u32,
}

/// Clusters form initial boxes for quantization, to ensure extreme colors are better represented
pub const LIQ_MAXCLUSTER: usize = 16;

pub(crate) struct HistogramInternal {
    pub items: Box<[HistItem]>,
    pub total_perceptual_weight: f64,
    pub clusters: [Cluster; LIQ_MAXCLUSTER],
    pub fixed_colors: Box<[f_pixel]>,
}

// Pre-grouped colors
#[derive(Copy, Clone, Debug)]
pub(crate) struct Cluster {
    pub begin: u32,
    pub end: u32,
}

// Simple deterministic hasher for the color hashmap
impl hash::BuildHasher for U32Hasher {
    type Hasher = Self;

    #[inline(always)]
    fn build_hasher(&self) -> Self {
        Self(0)
    }
}

pub(crate) struct U32Hasher(pub u32);
impl hash::Hasher for U32Hasher {
    // magic constant from fxhash. For a single 32-bit key that's all it needs!
    #[inline(always)]
    fn finish(&self) -> u64 { u64::from(self.0).wrapping_mul(0x517cc1b727220a95) }
    #[inline(always)]
    fn write_u32(&mut self, i: u32) { self.0 = i; }
    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        // fold the high and low words so 16-bit RGBA keys (four u16 channels) hash well
        self.0 = (i as u32) ^ ((i >> 32) as u32).rotate_left(13);
    }

    fn write(&mut self, _bytes: &[u8]) { unimplemented!() }
    fn write_u8(&mut self, _i: u8) { unimplemented!() }
    fn write_u16(&mut self, _i: u16) { unimplemented!() }
    fn write_u128(&mut self, _i: u128) { unimplemented!() }
    fn write_usize(&mut self, _i: usize) { unimplemented!() }
    fn write_i8(&mut self, _i: i8) { unimplemented!() }
    fn write_i16(&mut self, _i: i16) { unimplemented!() }
    fn write_i32(&mut self, _i: i32) { unimplemented!() }
    fn write_i64(&mut self, _i: i64) { unimplemented!() }
    fn write_i128(&mut self, _i: i128) { unimplemented!() }
    fn write_isize(&mut self, _i: isize) { unimplemented!() }
}

/// ignores the index
#[derive(PartialEq, Debug)]
pub(crate) struct HashColor {
    pub rgba: RGBA,
    pub index: PalIndex,
}

#[allow(clippy::derived_hash_with_manual_eq)]
impl Hash for HashColor {
    #[inline]
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        let s: &[u8] = self.rgba.as_ref();
        u32::from_ne_bytes(s.try_into().unwrap()).hash(state);
    }
}

impl Eq for HashColor {
}

#[cfg(test)]
mod rgba16_key_tests {
    use super::rgba16_key;
    use crate::RGBA16;

    #[test]
    fn key_packs_all_channels() {
        let px = RGBA16::new(0x1111, 0x2222, 0x3333, 0x4444);
        let key = rgba16_key(px);
        assert_eq!(key & 0xffff, 0x1111, "r channel");
        assert_eq!((key >> 16) & 0xffff, 0x2222, "g channel");
        assert_eq!((key >> 32) & 0xffff, 0x3333, "b channel");
        assert_eq!((key >> 48) & 0xffff, 0x4444, "a channel");
    }

    #[test]
    fn key_is_distinct_and_order_independent() {
        // different colors must not collide in the low bits used by U32Hasher
        let a = rgba16_key(RGBA16::new(1, 2, 3, 65535));
        let b = rgba16_key(RGBA16::new(3, 2, 1, 65535));
        assert_ne!(a, b);
        assert_ne!(a as u32, b as u32, "low-word fold must distinguish r/b swaps");
    }

    #[test]
    fn grid_aligned_key_roundtrips() {
        // v16 = v8 * 257 maps back to v8 via integer division (PNG 16->8 convention)
        for v8 in [0u8, 1, 128, 200, 255] {
            let v16 = v16_scale(v8);
            assert_eq!((v16 / 257) as u8, v8);
        }
    }

    fn v16_scale(v8: u8) -> u16 { (v8 as u16) * 257 }
}
