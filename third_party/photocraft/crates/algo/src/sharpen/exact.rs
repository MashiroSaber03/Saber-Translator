//! Exact separable convolution for Unsharp Mask: stream one source row and reuse
//! the blurred output for the final arithmetic. Each sample retains its tap order.
use crate::image::premultiply;
use crate::{Ctx, Image};
use photocraft_geom::Rect;
use photocraft_raster::Interrupt;

const MAX_SAMPLES: usize = 128 * 1024 * 1024;
// The filter dialog caps radius at 1000. Preserve Details resampling can ask
// for more; permit that while the unchanged kernel's i32 square is safe.
const MAX_RADIUS: f32 = 15_446.0;

pub(crate) fn valid_params(amount: f32, radius: f32, threshold: f32) -> bool {
    amount.is_finite() && threshold.is_finite() && radius.is_finite() && (0.0..=MAX_RADIUS).contains(&radius)
}

pub(crate) fn halo(radius: f32) -> i32 {
    if radius.is_finite() && (0.0..=MAX_RADIUS).contains(&radius) { (radius * 3.0).ceil() as i32 + 1 } else { 0 }
}

pub(crate) fn valid_area(r: Rect) -> bool {
    r.x0 <= r.x1
        && r.y0 <= r.y1
        && r.width() <= 300_000
        && r.height() <= 300_000
        && [r.x0, r.x1, r.y0, r.y1].iter().all(|v| (-1_000_000..=1_000_000).contains(v))
}

fn zeros(len: usize) -> Option<Vec<f32>> {
    if len > MAX_SAMPLES {
        return None;
    }
    let mut data = Vec::new();
    data.try_reserve_exact(len).ok()?;
    data.resize(len, 0.0);
    Some(data)
}

/// Checked contiguous row copy, with the same zero extension as Image::get.
fn read_row(src: &Image, x0: i32, x1: i32, y: i32, dst: &mut [f32]) -> Option<()> {
    dst.fill(0.0);
    if y < src.rect.y0 || y >= src.rect.y1 {
        return Some(());
    }
    let (lo, hi) = (x0.max(src.rect.x0), x1.min(src.rect.x1));
    if lo >= hi {
        return Some(());
    }
    let n = src.ch;
    let count = usize::try_from(i64::from(hi) - i64::from(lo)).ok()?.checked_mul(n)?;
    let offset = usize::try_from(i64::from(lo) - i64::from(x0)).ok()?.checked_mul(n)?;
    let row = usize::try_from(i64::from(y) - i64::from(src.rect.y0)).ok()?;
    let col = usize::try_from(i64::from(lo) - i64::from(src.rect.x0)).ok()?;
    let start = row.checked_mul(src.rect.width() as usize)?.checked_add(col)?.checked_mul(n)?;
    dst.get_mut(offset..offset.checked_add(count)?)?.copy_from_slice(src.data.get(start..start.checked_add(count)?)?);
    Some(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run(src: &Image, out: Rect, ctx: &Ctx, amount: f32, radius: f32, threshold: f32, ctl: &Interrupt) -> Option<Vec<f32>> {
    if ctl.cancelled() || !valid_params(amount, radius, threshold) || !valid_area(out) || !valid_area(src.rect) {
        return None;
    }
    let n = src.ch;
    if n == 0 || n > crate::fxutil::MAXC {
        return None;
    }
    if src.data.len() != (src.rect.width() as usize).checked_mul(src.rect.height() as usize)?.checked_mul(n)? {
        return None;
    }
    if out.is_empty() {
        return Some(Vec::new());
    }
    let kernel = super::gaussian_kernel(radius);
    let r = kernel.len() / 2;
    let (ow, oh) = (out.width() as usize, out.height() as usize);
    let row_len = ow.checked_mul(n)?;
    let th = oh.checked_add(2 * r)?;
    let ww = ow.checked_add(2 * r)?;
    let (x0, x1, y0) = (out.x0.checked_sub(r as i32)?, out.x1.checked_add(r as i32)?, out.y0.checked_sub(r as i32)?);
    let mut row = zeros(ww.checked_mul(n)?)?;
    let mut tmp = zeros(row_len.checked_mul(th)?)?;
    let mut result = zeros(row_len.checked_mul(oh)?)?;
    for (y, dst) in tmp.chunks_exact_mut(row_len).enumerate() {
        if ctl.cancelled() {
            return None;
        }
        read_row(src, x0, x1, y0.checked_add(y as i32)?, &mut row)?;
        premultiply(&mut row, n, ctx.alpha);
        // Tap-major traversal exposes contiguous independent samples to LLVM. No
        // reduction, tap merging, symmetry or fused multiply-add changes the sums.
        for (i, weight) in kernel.iter().enumerate() {
            let start = i.checked_mul(n)?;
            let samples = row.get(start..start.checked_add(row_len)?)?;
            for (d, s) in dst.iter_mut().zip(samples) {
                *d += s * weight;
            }
        }
    }
    let (amount, threshold) = (amount / 100.0, threshold / 255.0);
    let cc = n - usize::from(ctx.alpha);
    for (y, dst) in result.chunks_exact_mut(row_len).enumerate() {
        if ctl.cancelled() {
            return None;
        }
        for (i, weight) in kernel.iter().enumerate() {
            let start = y.checked_add(i)?.checked_mul(row_len)?;
            let samples = tmp.get(start..start.checked_add(row_len)?)?;
            for (d, s) in dst.iter_mut().zip(samples) {
                *d += s * weight;
            }
        }
        let originals = row.get_mut(..row_len)?;
        read_row(src, out.x0, out.x1, out.y0.checked_add(y as i32)?, originals)?;
        for (blurred, original) in dst.chunks_exact_mut(n).zip(originals.chunks_exact(n)) {
            if ctx.alpha && *original.last()? <= 0.0 {
                blurred.copy_from_slice(original);
                continue;
            }
            let a = if ctx.alpha { *blurred.last()? } else { 1.0 };
            for (b, o) in blurred.iter_mut().zip(original).take(cc) {
                // Keep the original unpremultiply cut-off and division, followed by
                // the threshold epsilon and inclusive comparison, at every depth.
                let b_value = if ctx.alpha { if a > 1e-7 { *b / a } else { 0.0 } } else { *b };
                let d = o - b_value;
                *b = if d.abs() + 1e-6 >= threshold { o + amount * d } else { *o };
            }
            if ctx.alpha {
                *blurred.last_mut()? = *original.last()?;
            }
        }
    }
    Some(result)
}
