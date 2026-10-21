//! Fallible buffers used by memory-intensive document operations.

use crate::{EngineError, Result};

#[cfg(test)]
thread_local! {
    static FAIL_NEXT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Reserve and initialize a buffer without invoking the allocator's abort-on-OOM path.
pub(crate) fn filled<T: Clone>(len: usize, value: T, purpose: &str) -> Result<Vec<T>> {
    let mut out = capacity(len, purpose)?;
    out.resize(len, value);
    Ok(out)
}

/// Reserve a vector for a known number of operation results.
pub(crate) fn capacity<T>(len: usize, purpose: &str) -> Result<Vec<T>> {
    let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| error(purpose, "requested size overflowed"))?;
    #[cfg(test)]
    if FAIL_NEXT.with(|flag| flag.replace(false)) {
        return Err(error(purpose, "not enough memory"));
    }
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| error(purpose, &format!("not enough memory for {bytes} bytes")))?;
    Ok(out)
}

/// Check an operation boundary where downstream encoders/decoders allocate internally.
/// The test-only failpoint lets command tests prove their transaction boundary.
pub(crate) fn checkpoint(purpose: &str) -> Result<()> {
    #[cfg(test)]
    if FAIL_NEXT.with(|flag| flag.replace(false)) {
        return Err(error(purpose, "not enough memory"));
    }
    let _ = purpose;
    Ok(())
}

/// Copy a region into an explicitly reserved buffer.
pub(crate) fn read_region(surface: &photocraft_raster::Surface, area: photocraft_geom::Rect, purpose: &str) -> Result<Vec<f32>> {
    let len = (area.width() as usize)
        .checked_mul(area.height() as usize)
        .and_then(|n| n.checked_mul(surface.channels()))
        .ok_or_else(|| error(purpose, "requested size overflowed"))?;
    let mut out = filled(len, 0.0, purpose)?;
    surface.read_region_into(area, &mut out);
    Ok(out)
}

pub(crate) fn error(purpose: &str, detail: &str) -> EngineError {
    EngineError::Other(format!("not enough memory for {purpose} ({detail}); the document was not changed"))
}

#[cfg(test)]
pub(crate) fn fail_next_for_test() {
    FAIL_NEXT.with(|flag| flag.set(true));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocation_failure_is_reported_without_panicking() {
        fail_next_for_test();
        let result = filled::<u8>(16, 0, "test image");
        assert!(result.as_ref().is_err_and(|e| e.to_string().contains("not enough memory for test image")));
    }
}
