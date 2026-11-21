use crate::error::Error;
use crate::pal::{f_pixel, gamma_lut, gamma_lut16, RGBA, RGBA16};
use crate::seacow::{Pointer, SeaCow};
use crate::LIQ_HIGH_MEMORY_LIMIT;
use core::mem::{size_of, MaybeUninit};
use core::slice;

#[cfg(all(not(feature = "std"), feature = "no_std"))]
use std::{boxed::Box, vec::Vec};

pub(crate) type RowCallback<'a> = dyn Fn(&mut [MaybeUninit<RGBA>], usize) + Send + Sync + 'a;
pub(crate) type RowCallback16<'a> = dyn Fn(&mut [MaybeUninit<RGBA16>], usize) + Send + Sync + 'a;

pub(crate) enum PixelsSource<'pixels, 'rows> {
    /// The `pixels` field is never read, but it is used to store the rows.
    #[allow(dead_code)]
    Pixels {
        rows: SeaCow<'rows, Pointer<RGBA>>,
        pixels: Option<SeaCow<'pixels, RGBA>>,
    },
    /// 16-bit-per-channel variant of [`PixelsSource::Pixels`]
    #[allow(dead_code)]
    Pixels16 {
        rows: SeaCow<'rows, Pointer<RGBA16>>,
        pixels: Option<SeaCow<'pixels, RGBA16>>,
    },
    Callback(Box<RowCallback<'rows>>),
    Callback16(Box<RowCallback16<'rows>>),
}

impl<'pixels> PixelsSource<'pixels, '_> {
    pub(crate) fn for_pixels(pixels: SeaCow<'pixels, RGBA>, width: u32, height: u32, stride: u32) -> Result<Self, Error> {
        if stride < width || height == 0 || width == 0 {
            return Err(Error::ValueOutOfRange);
        }
        let stride = stride as usize;
        let width = width as usize;
        let height = height as usize;

        let slice = pixels.as_slice();
        let min_area = stride.checked_mul(height).and_then(|a| a.checked_add(width)).ok_or(Error::ValueOutOfRange)? - stride;
        if slice.len() < min_area {
            return Err(Error::BufferTooSmall);
        }

        let rows = SeaCow::boxed(slice.chunks(stride).map(|row| Pointer(row.as_ptr())).take(height).collect());
        Ok(Self::Pixels { rows, pixels: Some(pixels) })
    }

    pub(crate) fn for_pixels16(pixels: SeaCow<'pixels, RGBA16>, width: u32, height: u32, stride: u32) -> Result<Self, Error> {
        if stride < width || height == 0 || width == 0 {
            return Err(Error::ValueOutOfRange);
        }
        let stride = stride as usize;
        let width = width as usize;
        let height = height as usize;

        let slice = pixels.as_slice();
        let min_area = stride.checked_mul(height).and_then(|a| a.checked_add(width)).ok_or(Error::ValueOutOfRange)? - stride;
        if slice.len() < min_area {
            return Err(Error::BufferTooSmall);
        }

        let rows = SeaCow::boxed(slice.chunks(stride).map(|row| Pointer(row.as_ptr())).take(height).collect());
        Ok(Self::Pixels16 { rows, pixels: Some(pixels) })
    }

    pub(crate) const fn is_16bit(&self) -> bool {
        matches!(self, Self::Pixels16 { .. } | Self::Callback16(_))
    }
}

/// A scratch row matching the bit depth of the [`DynamicRows`] it is passed to.
pub(crate) enum TempRow<'a> {
    Bit8(&'a mut [MaybeUninit<RGBA>]),
    Bit16(&'a mut [MaybeUninit<RGBA16>]),
}

/// Owned version of [`TempRow`], stored in per-thread scratch buffers.
pub(crate) enum OwnedTempRow {
    Bit8(Box<[MaybeUninit<RGBA>]>),
    Bit16(Box<[MaybeUninit<RGBA16>]>),
}

impl OwnedTempRow {
    pub(crate) fn for_rows(rows: &DynamicRows<'_, '_>) -> Result<Self, Error> {
        if rows.is_16bit() {
            Ok(Self::Bit16(temp_buf(rows.width())?))
        } else {
            Ok(Self::Bit8(temp_buf(rows.width())?))
        }
    }

    pub(crate) fn as_temp_row(&mut self) -> TempRow<'_> {
        match self {
            Self::Bit8(row) => TempRow::Bit8(row),
            Self::Bit16(row) => TempRow::Bit16(row),
        }
    }
}

/// Gamma lookup table in the image's bit depth.
///
/// Both tables are heap-allocated (the 16-bit one is 256 KiB), so it is built lazily
/// on first row conversion and kept for the lifetime of the row iterator. Boxing both
/// variants keeps the enum small regardless of bit depth.
enum GammaLut {
    Bit8(Box<[f32; 256]>),
    Bit16(Box<[f32]>),
}
pub(crate) struct DynamicRows<'pixels, 'rows> {
    pub(crate) width: u32,
    pub(crate) height: u32,
    f_pixels: Option<Box<[f_pixel]>>,
    pixels: PixelsSource<'pixels, 'rows>,
    pub(crate) gamma: f64,
}

impl Clone for DynamicRows<'_, '_> {
    fn clone(&self) -> Self {
        Self {
            width: self.width,
            height: self.height,
            f_pixels: self.f_pixels.clone(),
            pixels: match &self.pixels {
                PixelsSource::Pixels { rows, pixels } => PixelsSource::Pixels {
                    rows: rows.clone(),
                    pixels: pixels.clone(),
                },
                PixelsSource::Pixels16 { rows, pixels } => PixelsSource::Pixels16 {
                    rows: rows.clone(),
                    pixels: pixels.clone(),
                },
                PixelsSource::Callback(_) => {
                    let area = self.width as usize * self.height as usize;
                    let mut out = Vec::with_capacity(area);
                    let out_rows = out.spare_capacity_mut()[..area].chunks_exact_mut(self.width as usize);
                    for (i, row) in out_rows.enumerate() {
                        self.row_rgba(row, i);
                    }
                    unsafe {
                        out.set_len(area);
                    }
                    let pixels = SeaCow::boxed(out.into_boxed_slice());
                    PixelsSource::for_pixels(pixels, self.width, self.height, self.width).unwrap()
                },
                PixelsSource::Callback16(_) => {
                    let area = self.width as usize * self.height as usize;
                    let mut out = Vec::with_capacity(area);
                    let out_rows = out.spare_capacity_mut()[..area].chunks_exact_mut(self.width as usize);
                    for (i, row) in out_rows.enumerate() {
                        self.row_rgba16(row, i);
                    }
                    unsafe {
                        out.set_len(area);
                    }
                    let pixels = SeaCow::boxed(out.into_boxed_slice());
                    PixelsSource::for_pixels16(pixels, self.width, self.height, self.width).unwrap()
                },
            },
            gamma: self.gamma,
        }
    }
}

pub(crate) struct DynamicRowsIter<'parent, 'pixels, 'rows> {
    px: &'parent DynamicRows<'pixels, 'rows>,
    temp_f_row: Option<Box<[MaybeUninit<f_pixel>]>>,
    gamma_lut: Option<GammaLut>,
}

impl DynamicRowsIter<'_, '_, '_> {
    #[must_use]
    pub fn row_f<'px>(&'px mut self, temp_row: TempRow<'px>, row: usize) -> &'px [f_pixel] {
        debug_assert_eq!(temp_row.len(), self.px.width as usize);
        if let Some(pixels) = self.px.f_pixels.as_ref() {
            let start = self.px.width as usize * row;
            &pixels[start..start + self.px.width as usize]
        } else {
            // Disjoint field borrows: the LUT and the scratch f-row are separate fields
            let lut = self.gamma_lut.as_ref().expect("gamma LUT not initialized");
            match self.temp_f_row.as_mut() {
                Some(t) => match temp_row {
                    TempRow::Bit8(row_pixels) => {
                        let row_pixels = self.px.row_rgba(row_pixels, row);
                        DynamicRows::convert_row_to_f(t, row_pixels, lut)
                    },
                    TempRow::Bit16(row_pixels) => {
                        let row_pixels = self.px.row_rgba16(row_pixels, row);
                        DynamicRows::convert_row_to_f16(t, row_pixels, lut)
                    },
                },
                None => &mut [], // this can't happen
            }
        }
    }

    #[must_use]
    pub fn row_f_shared<'px>(&'px self, temp_row: TempRow<'px>, temp_row_f: &'px mut [MaybeUninit<f_pixel>], row: usize) -> &'px [f_pixel] {
        if let Some(pixels) = self.px.f_pixels.as_ref() {
            &pixels[self.px.width as usize * row..]
        } else {
            let lut = self.gamma_lut.as_ref().expect("gamma LUT not initialized");
            match temp_row {
                TempRow::Bit8(row_pixels) => {
                    let row_pixels = self.px.row_rgba(row_pixels, row);
                    DynamicRows::convert_row_to_f(temp_row_f, row_pixels, lut)
                },
                TempRow::Bit16(row_pixels) => {
                    let row_pixels = self.px.row_rgba16(row_pixels, row);
                    DynamicRows::convert_row_to_f16(temp_row_f, row_pixels, lut)
                },
            }
        }
    }

    #[must_use]
    pub fn row_rgba<'px>(&'px self, temp_row: &'px mut [MaybeUninit<RGBA>], row: usize) -> &'px [RGBA] {
        self.px.row_rgba(temp_row, row)
    }

    #[must_use]
    pub fn row_rgba16<'px>(&'px self, temp_row: &'px mut [MaybeUninit<RGBA16>], row: usize) -> &'px [RGBA16] {
        self.px.row_rgba16(temp_row, row)
    }
}

impl TempRow<'_> {
    fn len(&self) -> usize {
        match self {
            TempRow::Bit8(row) => row.len(),
            TempRow::Bit16(row) => row.len(),
        }
    }
}

impl<'pixels, 'rows> DynamicRows<'pixels, 'rows> {
    #[inline]
    pub(crate) fn new(width: u32, height: u32, pixels: PixelsSource<'pixels, 'rows>, gamma: f64) -> Self {
        debug_assert!(gamma > 0.);
        Self { width, height, f_pixels: None, pixels, gamma }
    }

    #[inline(always)]
    pub(crate) const fn is_16bit(&self) -> bool {
        self.pixels.is_16bit()
    }

    fn row_rgba<'px>(&'px self, temp_row: &'px mut [MaybeUninit<RGBA>], row: usize) -> &'px [RGBA] {
        match &self.pixels {
            PixelsSource::Pixels { rows, .. } => unsafe {
                slice::from_raw_parts(rows.as_slice()[row].0, self.width())
            },
            PixelsSource::Callback(cb) => {
                cb(temp_row, row);
                // cb needs to be marked as unsafe, since it's responsible for initialization :(
                unsafe { slice_assume_init_mut(temp_row) }
            },
            PixelsSource::Pixels16 { .. } | PixelsSource::Callback16(_) => {
                panic!("16-bit image row requested as 8-bit");
            },
        }
    }

    fn row_rgba16<'px>(&'px self, temp_row: &'px mut [MaybeUninit<RGBA16>], row: usize) -> &'px [RGBA16] {
        match &self.pixels {
            PixelsSource::Pixels16 { rows, .. } => unsafe {
                slice::from_raw_parts(rows.as_slice()[row].0, self.width())
            },
            PixelsSource::Callback16(cb) => {
                cb(temp_row, row);
                unsafe { slice_assume_init_mut(temp_row) }
            },
            PixelsSource::Pixels { .. } | PixelsSource::Callback(_) => {
                panic!("8-bit image row requested as 16-bit");
            },
        }
    }

    fn convert_row_to_f<'f>(row_f_pixels: &'f mut [MaybeUninit<f_pixel>], row_pixels: &[RGBA], gamma_lut: &GammaLut) -> &'f mut [f_pixel] {
        assert_eq!(row_f_pixels.len(), row_pixels.len());
        let GammaLut::Bit8(lut) = gamma_lut else { panic!("gamma LUT bit depth mismatch") };
        for (dst, src) in row_f_pixels.iter_mut().zip(row_pixels) {
            dst.write(f_pixel::from_rgba(lut, *src));
        }
        // Safe, just initialized
        unsafe { slice_assume_init_mut(row_f_pixels) }
    }

    fn convert_row_to_f16<'f>(row_f_pixels: &'f mut [MaybeUninit<f_pixel>], row_pixels: &[RGBA16], gamma_lut: &GammaLut) -> &'f mut [f_pixel] {
        assert_eq!(row_f_pixels.len(), row_pixels.len());
        let GammaLut::Bit16(lut) = gamma_lut else { panic!("gamma LUT bit depth mismatch") };
        for (dst, src) in row_f_pixels.iter_mut().zip(row_pixels) {
            dst.write(f_pixel::from_rgba16(lut, *src));
        }
        // Safe, just initialized
        unsafe { slice_assume_init_mut(row_f_pixels) }
    }

    #[must_use]
    fn should_use_low_memory(&self) -> bool {
        self.width() * self.height() > LIQ_HIGH_MEMORY_LIMIT / size_of::<f_pixel>()
    }

    #[inline]
    fn temp_f_row_for_iter(&self) -> Result<Option<Box<[MaybeUninit<f_pixel>]>>, Error> {
        if self.f_pixels.is_some() {
            return Ok(None);
        }
        Ok(Some(temp_buf(self.width())?))
    }

    fn gamma_lut_for_iter(&self) -> Option<GammaLut> {
        if self.f_pixels.is_some() {
            return None;
        }
        Some(if self.is_16bit() {
            GammaLut::Bit16(gamma_lut16(self.gamma))
        } else {
            GammaLut::Bit8(Box::new(gamma_lut(self.gamma)))
        })
    }

    pub fn prepare_iter(&mut self, temp_row: TempRow<'_>, allow_steamed: bool) -> Result<(), Error> {
        debug_assert_eq!(temp_row.len(), self.width());

        if self.f_pixels.is_some() || (allow_steamed && self.should_use_low_memory()) {
            return Ok(());
        }

        let width = self.width();
        let mut f_pixels = temp_buf(width * self.height())?;
        if self.is_16bit() {
            let TempRow::Bit16(temp_row) = temp_row else { panic!("bit depth mismatch") };
            let lut = GammaLut::Bit16(gamma_lut16(self.gamma));
            for (row, f_row) in f_pixels.chunks_exact_mut(width).enumerate() {
                let row_pixels = self.row_rgba16(temp_row, row);
                Self::convert_row_to_f16(f_row, row_pixels, &lut);
            }
        } else {
            let TempRow::Bit8(temp_row) = temp_row else { panic!("bit depth mismatch") };
            let lut = GammaLut::Bit8(Box::new(gamma_lut(self.gamma)));
            for (row, f_row) in f_pixels.chunks_exact_mut(width).enumerate() {
                let row_pixels = self.row_rgba(temp_row, row);
                Self::convert_row_to_f(f_row, row_pixels, &lut);
            }
        }
        // just initialized
        self.f_pixels = Some(unsafe { box_assume_init(f_pixels) });
        Ok(())
    }

    #[inline]
    pub fn rows_iter(&mut self, temp_row: TempRow<'_>) -> Result<DynamicRowsIter<'_, 'pixels, 'rows>, Error> {
        self.prepare_iter(temp_row, true)?;
        Ok(DynamicRowsIter {
            temp_f_row: self.temp_f_row_for_iter()?,
            gamma_lut: self.gamma_lut_for_iter(),
            px: self,
        })
    }

    /// Call `prepare_iter()` first
    #[inline]
    pub fn rows_iter_prepared(&self) -> Result<DynamicRowsIter<'_, 'pixels, 'rows>, Error> {
        Ok(DynamicRowsIter {
            temp_f_row: self.temp_f_row_for_iter()?,
            gamma_lut: self.gamma_lut_for_iter(),
            px: self,
        })
    }

    #[inline]
    pub fn rgba_rows_iter(&self) -> Result<DynamicRowsIter<'_, 'pixels, 'rows>, Error> {
        // This happens when histogram image is recycled
        if let PixelsSource::Pixels { rows, .. } = &self.pixels {
            if rows.as_slice().is_empty() {
                return Err(Error::Unsupported);
            }
        }
        Ok(DynamicRowsIter { px: self, temp_f_row: None, gamma_lut: None })
    }

    #[inline]
    #[allow(clippy::unnecessary_unwrap)]
    pub fn all_rows_f(&mut self) -> Result<&[f_pixel], Error> {
        if self.f_pixels.is_some() {
            // borrow-checker: returning the cached slice while still borrowing self
            return Ok(self.f_pixels.as_ref().unwrap());
        }
        let mut temp_row = OwnedTempRow::for_rows(self)?;
        self.prepare_iter(temp_row.as_temp_row(), false)?;
        self.f_pixels.as_deref().ok_or(Error::Unsupported)
    }

    /// Not recommended
    #[cfg(feature = "_internal_c_ffi")]
    pub(crate) unsafe fn set_memory_ownership(&mut self, own_rows: bool, own_pixels: bool, free_fn: unsafe extern "C" fn(*mut core::ffi::c_void)) -> Result<(), Error> {
        if own_rows {
            match &mut self.pixels {
                PixelsSource::Pixels { rows, .. } => rows.make_owned(free_fn),
                PixelsSource::Pixels16 { rows, .. } => rows.make_owned(free_fn),
                PixelsSource::Callback(_) | PixelsSource::Callback16(_) => return Err(Error::ValueOutOfRange),
            }
        }

        if own_pixels {
            let len = self.width() * self.height();
            match &mut self.pixels {
                PixelsSource::Pixels { pixels: Some(pixels), .. } => pixels.make_owned(free_fn),
                PixelsSource::Pixels16 { pixels: Some(pixels), .. } => pixels.make_owned(free_fn),
                PixelsSource::Pixels { pixels, rows } => {
                    // the row with the lowest address is assumed to be at the start of the bitmap
                    let ptr = rows.as_slice().iter().map(|p| p.0).min().ok_or(Error::Unsupported)?;
                    *pixels = Some(SeaCow::c_owned(ptr.cast_mut(), len, free_fn));
                },
                PixelsSource::Pixels16 { pixels, rows } => {
                    let ptr = rows.as_slice().iter().map(|p| p.0).min().ok_or(Error::Unsupported)?;
                    *pixels = Some(SeaCow::c_owned(ptr.cast_mut(), len, free_fn));
                },
                PixelsSource::Callback(_) | PixelsSource::Callback16(_) => return Err(Error::ValueOutOfRange),
            }
        }
        Ok(())
    }

    pub fn free_histogram_inputs(&mut self) {
        if self.f_pixels.is_some() {
            self.pixels = PixelsSource::Pixels {
                rows: SeaCow::borrowed(&[]),
                pixels: None,
            };
        }
    }

    #[inline(always)]
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width as usize
    }

    #[inline(always)]
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height as usize
    }
}

pub(crate) fn temp_buf<T>(len: usize) -> Result<Box<[MaybeUninit<T>]>, Error> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)?;
    unsafe { v.set_len(len) };
    Ok(v.into_boxed_slice())
}

#[test]
fn send() {
    fn is_send<T: Send>() {}
    fn is_sync<T: Sync>() {}
    is_send::<DynamicRows>();
    is_sync::<DynamicRows>();
    is_send::<PixelsSource>();
    is_sync::<PixelsSource>();
}

#[inline(always)]
unsafe fn box_assume_init<T>(s: Box<[MaybeUninit<T>]>) -> Box<[T]> {
    core::mem::transmute(s)
}

#[inline(always)]
unsafe fn slice_assume_init_mut<T>(s: &mut [MaybeUninit<T>]) -> &mut [T] {
    &mut *(s as *mut [MaybeUninit<T>] as *mut [T])
}
