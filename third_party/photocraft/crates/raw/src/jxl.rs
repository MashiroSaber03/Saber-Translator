//! JPEG XL-compressed DNG tiles and strips (Adobe DNG Specification 1.7, Compression 52546).
//!
//! Each strip or tile holds one JPEG XL codestream (bare or in the ISO/IEC 18181-2 container)
//! whose image is the segment's samples: one channel for CFA data, three for LinearRaw. Samsung
//! Expert RAW writes LinearRaw this way. Decoding is done by `jxl-oxide`, a pure-Rust decoder;
//! like heic-rs in `photocraft-heif`, every call into it runs under `catch_unwind` so a decoder
//! panic on a malformed file becomes [`RawError::Malformed`].
//!
//! The browser build (wasm32) leaves the decoder out: jxl-oxide added 0.88 MB to a wasm that
//! must stay under the per-file cap of static hosts (packaging/web/package.sh), for a format only
//! a few phones write. There, these segments are [`RawError::Unsupported`] with a clear message.

#[cfg(not(target_arch = "wasm32"))]
use jxl_oxide::{AllocTracker, JxlImage};

use crate::Limits;
use crate::error::{RawError, Result};

/// The browser build has no JPEG XL decoder (see the module docs).
#[cfg(target_arch = "wasm32")]
pub(crate) fn decode(_src: &[u8], _w: usize, _h: usize, _samples: usize, _bits: u32, _limits: &Limits) -> Result<Vec<u16>> {
    Err(RawError::unsupported("JPEG XL-compressed DNG (DNG 1.7) can't be opened in the browser version of PhotoCraft; open it in the desktop app"))
}

/// Decodes one segment into `w * h * samples` interleaved samples at `bits` per sample (the
/// TIFF BitsPerSample, which DNG requires the codestream's bit depth to match).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn decode(src: &[u8], w: usize, h: usize, samples: usize, bits: u32, limits: &Limits) -> Result<Vec<u16>> {
    limits.check(w as u64, h as u64, samples as u64 * 2)?;
    // jxl-oxide decodes into f32/i32 planes: allow a few times the output size, and never more
    // than the caller's per-buffer limit.
    let budget = (w as u64).saturating_mul(h as u64).saturating_mul(samples as u64).saturating_mul(16).min(limits.max_alloc);
    let budget = usize::try_from(budget).unwrap_or(usize::MAX);
    guarded(|| {
        let image =
            JxlImage::builder().alloc_tracker(AllocTracker::with_limit(budget)).read(src).map_err(|e| RawError::malformed(format!("JPEG XL raw data: {e}")))?;
        let (jw, jh) = (image.width() as usize, image.height() as usize);
        if (jw, jh) != (w, h) {
            return Err(RawError::malformed(format!("JPEG XL raw tile is {jw}x{jh}, expected {w}x{h}")));
        }
        let render = image.render_frame(0).map_err(|e| RawError::malformed(format!("JPEG XL raw data: {e}")))?;
        let mut stream = render.stream_no_alpha();
        let channels = stream.channels() as usize;
        if channels != samples {
            return Err(RawError::unsupported(format!("JPEG XL raw tile with {channels} channels for {samples} samples per pixel")));
        }
        // Orientation is applied by the stream; DNG tiles are stored upright, so anything else
        // would not fit the tile grid.
        if (stream.width() as usize, stream.height() as usize) != (w, h) {
            return Err(RawError::unsupported("rotated JPEG XL raw tile"));
        }
        let n = w.checked_mul(h).and_then(|p| p.checked_mul(samples)).ok_or_else(|| RawError::malformed("JPEG XL raw tile is too large"))?;
        let mut out = vec![0u16; n];
        if stream.write_to_buffer(&mut out) != n {
            return Err(RawError::malformed("JPEG XL raw data is truncated"));
        }
        // 16-bit samples come back as stored; other depths are scaled to 0..=65535, so scale
        // them back to the TIFF's range (exact for lossless data: 65535 / (2^bits - 1) >= 1).
        if bits < 16 {
            let max = f64::from((1u32 << bits) - 1);
            for v in &mut out {
                *v = (f64::from(*v) * max / 65535.0).round() as u16;
            }
        }
        Ok(out)
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn guarded<T>(f: impl FnOnce() -> Result<T>) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or_else(|_| Err(RawError::malformed("the JPEG XL decoder failed on this file")))
}
