//! Shape Blur's row-wise convolution. Each sample sees the same f64 additions,
//! in the same span order as the scalar implementation, but contiguous slices
//! let the compiler vectorize without architecture-specific code or unsafe.
use photocraft_geom::Rect;

use crate::{
    BlurShape, Ctx,
    blur2::{Spans, shape_inside},
    fxutil::unpremul_px,
    image::Image,
};

/// The command supports at most 1000 px. Bound direct API input too, before
/// halo arithmetic or O(r²) kernel rasterization. Nonfinite input uses zero radius.
pub(crate) fn radius(r: f32) -> f32 {
    if r.is_finite() { r.clamp(0.0, 1000.0) } else { 0.0 }
}

pub(crate) struct Prepared {
    spans: Spans,
}

impl Prepared {
    pub(crate) fn new(r: f32, shape: BlurShape) -> Self {
        Self { spans: Spans::new(radius(r), shape_inside(shape)) }
    }

    pub(crate) fn run(&self, src: &Image, out: Rect, ctx: &Ctx) -> Vec<f32> {
        self.checked_run(src, out, ctx).unwrap_or_default()
    }

    pub(crate) fn checked_run(&self, src: &Image, out: Rect, ctx: &Ctx) -> Option<Vec<f32>> {
        let n = src.ch;
        if n == 0 || n > crate::fxutil::MAXC || out.is_empty() {
            return Some(Vec::new());
        }
        let w = src.rect.width() as usize;
        let h = src.rect.height() as usize;
        let input_row = w.checked_mul(n)?;
        if src.data.len() != input_row.checked_mul(h)? {
            return None;
        }
        let stride = w.checked_add(1)?.checked_mul(n)?;
        let prefix_len = stride.checked_mul(h)?;
        let row_len = (out.width() as usize).checked_mul(n)?;
        let output_len = row_len.checked_mul(out.height() as usize)?;
        // Bound scratch and output allocations, including hostile public Image/Rect input.
        // Also accommodate direct full-image kernels (36 MP at all eight
        // channels), not just the application's halo-expanded tiles.
        const MAX_SAMPLES: usize = 512 * 1024 * 1024;
        if prefix_len > MAX_SAMPLES || output_len > MAX_SAMPLES {
            return None;
        }
        let mut prefix = Vec::new();
        prefix.try_reserve_exact(prefix_len).ok()?;
        prefix.resize(prefix_len, 0.0f64);
        if input_row != 0 {
            for (s, d) in src.data.chunks_exact(input_row).zip(prefix.chunks_exact_mut(stride)) {
                let mut sum = [0.0f64; crate::fxutil::MAXC];
                for (px, dst) in s.chunks_exact(n).zip(d.get_mut(n..)?.chunks_exact_mut(n)) {
                    let alpha = if ctx.alpha { *px.last()? } else { 1.0 };
                    for (c, ((v, sum), dst)) in px.iter().zip(sum.iter_mut()).zip(dst).enumerate() {
                        // Multiplication stays f32, exactly as premul_window.
                        let v = if ctx.alpha && c + 1 < n { *v * alpha } else { *v };
                        *sum += f64::from(v);
                        *dst = *sum;
                    }
                }
            }
        }
        let mut result = Vec::new();
        result.try_reserve_exact(output_len).ok()?;
        let mut acc = Vec::new();
        acc.try_reserve_exact(row_len).ok()?;
        acc.resize(row_len, 0.0f64);
        let x = i64::from(out.x0) - i64::from(src.rect.x0);
        for y in out.y0..out.y1 {
            acc.fill(0.0);
            for &(dy, a, b) in &self.spans.rows {
                let yy = i64::from(y) - i64::from(src.rect.y0) + i64::from(dy);
                let Ok(yy) = usize::try_from(yy) else {
                    continue;
                };
                if yy >= h {
                    continue;
                }
                let row = prefix.get(yy.checked_mul(stride)?..yy.checked_add(1)?.checked_mul(stride)?)?;
                let lo = x + i64::from(a);
                let hi = x + i64::from(b) + 1;
                let end = out.width() as i64;
                if lo >= 0 && hi + end <= w as i64 {
                    let start_lo = usize::try_from(lo).ok()?.checked_mul(n)?;
                    let start_hi = usize::try_from(hi).ok()?.checked_mul(n)?;
                    let low = row.get(start_lo..start_lo.checked_add(row_len)?)?;
                    let high = row.get(start_hi..start_hi.checked_add(row_len)?)?;
                    for ((acc, hi), lo) in acc.iter_mut().zip(high).zip(low) {
                        *acc += hi - lo;
                    }
                } else {
                    // Direct kernel callers may omit the halo or request pixels outside src.
                    // Retain the scalar kernel's zero extension and fixed normalization.
                    for (i, px) in acc.chunks_exact_mut(n).enumerate() {
                        let low = (lo + i as i64).clamp(0, w as i64) as usize * n;
                        let high = (hi + i as i64).clamp(0, w as i64) as usize * n;
                        for ((acc, hi), lo) in px.iter_mut().zip(row.get(high..high + n)?).zip(row.get(low..low + n)?) {
                            *acc += hi - lo;
                        }
                    }
                }
            }
            for px in acc.chunks_exact(n) {
                let mut sample = [0.0f32; crate::fxutil::MAXC];
                let sample = sample.get_mut(..n)?;
                for (dst, sum) in sample.iter_mut().zip(px) {
                    *dst = (*sum / self.spans.count) as f32;
                }
                unpremul_px(sample, ctx.alpha);
                result.extend_from_slice(sample);
            }
        }
        Some(result)
    }
}
