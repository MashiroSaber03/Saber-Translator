//! Direct Radial Blur, with immutable sampling tables shared by every output tile.
//!
//! Pixel positions and the quality-dependent interval ceilings match the upstream
//! sampler. The hot loop keeps channels together, reads each bilinear tap once,
//! and preserves the upstream alpha arithmetic for every sample.

use photocraft_geom::Rect;
use photocraft_raster::Interrupt;

use crate::image::Image;
use crate::{Ctx, RadialMethod, RadialQuality};

/// Position-independent rotations (Spin) or ray scales (Zoom), prepared once.
pub(crate) struct Plan {
    center: (f32, f32),
    extent: f32,
    method: RadialMethod,
    // Packed tables for interval counts 1..=max_intervals, each with n+1 samples.
    // Best needs at most 8,394,752 pairs (about 64 MiB), reserved fallibly.
    tables: Vec<(f32, f32)>,
    max_intervals: usize,
    valid: bool,
}

impl Plan {
    #[cfg(test)]
    pub(crate) fn new(bounds: Rect, amount: f32, method: RadialMethod, center: (f32, f32)) -> Self {
        Self::with_quality(bounds, amount, method, RadialQuality::Draft, center, &Interrupt::NONE).unwrap_or(Self {
            center: (0.0, 0.0),
            extent: 0.0,
            method,
            tables: Vec::new(),
            max_intervals: 64,
            valid: false,
        })
    }

    #[cfg(test)]
    pub(crate) fn with_quality(
        bounds: Rect,
        amount: f32,
        method: RadialMethod,
        quality: RadialQuality,
        center: (f32, f32),
        ctl: &Interrupt<'_>,
    ) -> Option<Self> {
        Self::prepare(bounds, amount, method, quality, center, None, ctl)
    }

    pub(crate) fn for_area(
        bounds: Rect,
        amount: f32,
        method: RadialMethod,
        quality: RadialQuality,
        center: (f32, f32),
        area: Rect,
        ctl: &Interrupt<'_>,
    ) -> Option<Self> {
        Self::prepare(bounds, amount, method, quality, center, Some(area), ctl)
    }

    #[allow(clippy::too_many_arguments)]
    fn prepare(
        bounds: Rect,
        amount: f32,
        method: RadialMethod,
        quality: RadialQuality,
        center: (f32, f32),
        area: Option<Rect>,
        ctl: &Interrupt<'_>,
    ) -> Option<Self> {
        if !amount.is_finite() || !center.0.is_finite() || !center.1.is_finite() || ctl.cancelled() {
            return None;
        }
        let amount = amount.clamp(0.0, 100.0);
        let center = (bounds.x0 as f32 + bounds.width() as f32 * center.0, bounds.y0 as f32 + bounds.height() as f32 * center.1);
        if !center.0.is_finite() || !center.1.is_finite() {
            return None;
        }
        let extent = match method {
            RadialMethod::Spin => amount.to_radians(),
            RadialMethod::Zoom => amount / 200.0,
        };
        // Only prepare interval counts reachable in the requested output. This
        // avoids preparing all Best tables (64MiB) for a tiny image/selection.
        // Distance from a fixed center is greatest at a rectangle's corners.
        let path = area.map_or(f32::INFINITY, |area| {
            let mut radius = 0.0f32;
            for y in [area.y0, area.y1.saturating_sub(1)] {
                for x in [area.x0, area.x1.saturating_sub(1)] {
                    let dx = x as f32 + 0.5 - center.0;
                    let dy = y as f32 + 0.5 - center.1;
                    radius = radius.max((dx * dx + dy * dy).sqrt());
                }
            }
            extent * radius
        });
        let max_intervals = super::radial_intervals(path, quality);
        let count = max_intervals.checked_mul(max_intervals.checked_add(3)?)?.checked_div(2)?;
        let mut tables = Vec::new();
        tables.try_reserve_exact(count).ok()?;
        for n in 1..=max_intervals {
            if ctl.cancelled() {
                return None;
            }
            for i in 0..=n {
                tables.push(match method {
                    RadialMethod::Spin => ((i as f32 / n as f32 - 0.5) * extent).sin_cos(),
                    RadialMethod::Zoom => (0.0, 1.0 - extent * i as f32 / n as f32),
                });
            }
        }
        Some(Self { center, extent, method, tables, max_intervals, valid: true })
    }

    /// `None` on cancellation or invalid image storage or allocation failure. Sampling beyond
    /// `src.rect` is transparent, including the individual taps at its edges.
    pub(crate) fn filter(&self, src: &Image, out: Rect, ctx: &Ctx, ctl: &Interrupt<'_>) -> Option<Vec<f32>> {
        if !self.valid || ctl.cancelled() || src.ch == 0 {
            return None;
        }
        let source = Source::new(src)?;
        let len = sample_len(out, src.ch)?;
        let mut result = Vec::new();
        result.try_reserve_exact(len).ok()?;
        result.resize(len, 0.0);
        if out.is_empty() {
            return Some(result);
        }
        match src.ch {
            1 => self.dispatch_fixed::<1>(&source, out, ctx.alpha, ctl, &mut result),
            2 => self.dispatch_fixed::<2>(&source, out, ctx.alpha, ctl, &mut result),
            3 => self.dispatch_fixed::<3>(&source, out, ctx.alpha, ctl, &mut result),
            4 => self.dispatch_fixed::<4>(&source, out, ctx.alpha, ctl, &mut result),
            5 => self.dispatch_fixed::<5>(&source, out, ctx.alpha, ctl, &mut result),
            _ => self.dispatch_generic(&source, out, ctx.alpha, ctl, &mut result),
        }?;
        Some(result)
    }

    fn dispatch_fixed<const N: usize>(&self, src: &Source<'_>, out: Rect, alpha: bool, ctl: &Interrupt<'_>, dst: &mut [f32]) -> Option<()> {
        match (alpha, self.method) {
            (true, RadialMethod::Spin) => self.fixed::<N, true, true>(src, out, ctl, dst),
            (false, RadialMethod::Spin) => self.fixed::<N, false, true>(src, out, ctl, dst),
            (true, RadialMethod::Zoom) => self.fixed::<N, true, false>(src, out, ctl, dst),
            (false, RadialMethod::Zoom) => self.fixed::<N, false, false>(src, out, ctl, dst),
        }
    }

    fn fixed<const N: usize, const ALPHA: bool, const SPIN: bool>(&self, src: &Source<'_>, out: Rect, ctl: &Interrupt<'_>, dst: &mut [f32]) -> Option<()> {
        let row_len = (out.width() as usize).checked_mul(N)?;
        for (row_index, row) in dst.chunks_exact_mut(row_len).enumerate() {
            if row_index.is_multiple_of(8) && ctl.cancelled() {
                return None;
            }
            let dy = (i64::from(out.y0) + row_index as i64) as f32 + 0.5 - self.center.1;
            for (column, pixel) in row.as_chunks_mut::<N>().0.iter_mut().enumerate() {
                let dx = (i64::from(out.x0) + column as i64) as f32 + 0.5 - self.center.0;
                let table = self.table(dx, dy)?;
                let mut acc = [0.0; N];
                for &tap in table {
                    let (x, y) = self.position::<SPIN>(dx, dy, tap);
                    sample_fixed::<N, ALPHA>(src, x, y, &mut acc);
                }
                finish(&mut acc, 1.0 / table.len() as f32, ALPHA);
                pixel.copy_from_slice(&acc);
            }
        }
        Some(())
    }

    fn dispatch_generic(&self, src: &Source<'_>, out: Rect, alpha: bool, ctl: &Interrupt<'_>, dst: &mut [f32]) -> Option<()> {
        match (alpha, self.method) {
            (true, RadialMethod::Spin) => self.generic::<true, true>(src, out, ctl, dst),
            (false, RadialMethod::Spin) => self.generic::<false, true>(src, out, ctl, dst),
            (true, RadialMethod::Zoom) => self.generic::<true, false>(src, out, ctl, dst),
            (false, RadialMethod::Zoom) => self.generic::<false, false>(src, out, ctl, dst),
        }
    }

    fn generic<const ALPHA: bool, const SPIN: bool>(&self, src: &Source<'_>, out: Rect, ctl: &Interrupt<'_>, dst: &mut [f32]) -> Option<()> {
        let n = src.image.ch;
        let row_len = (out.width() as usize).checked_mul(n)?;
        let mut sample = Vec::new();
        sample.try_reserve_exact(n).ok()?;
        sample.resize(n, 0.0);
        for (row_index, row) in dst.chunks_exact_mut(row_len).enumerate() {
            if row_index.is_multiple_of(8) && ctl.cancelled() {
                return None;
            }
            let dy = (i64::from(out.y0) + row_index as i64) as f32 + 0.5 - self.center.1;
            for (column, pixel) in row.chunks_exact_mut(n).enumerate() {
                let dx = (i64::from(out.x0) + column as i64) as f32 + 0.5 - self.center.0;
                let table = self.table(dx, dy)?;
                for &tap in table {
                    let (x, y) = self.position::<SPIN>(dx, dy, tap);
                    sample_generic::<ALPHA>(src, x, y, pixel, &mut sample);
                }
                finish(pixel, 1.0 / table.len() as f32, ALPHA);
            }
        }
        Some(())
    }

    #[inline]
    fn table(&self, dx: f32, dy: f32) -> Option<&[(f32, f32)]> {
        let radius = (dx * dx + dy * dy).sqrt();
        let n = ((self.extent * radius).ceil() as usize).clamp(1, self.max_intervals);
        let start = (n - 1).checked_mul(n.checked_add(2)?)?.checked_div(2)?;
        self.tables.get(start..start.checked_add(n.checked_add(1)?)?)
    }

    #[inline]
    fn position<const SPIN: bool>(&self, dx: f32, dy: f32, (s, c): (f32, f32)) -> (f32, f32) {
        if SPIN { (self.center.0 + dx * c - dy * s, self.center.1 + dx * s + dy * c) } else { (self.center.0 + dx * c, self.center.1 + dy * c) }
    }
}

fn sample_len(rect: Rect, channels: usize) -> Option<usize> {
    (rect.width() as usize).checked_mul(rect.height() as usize)?.checked_mul(channels)
}

/// Storage dimensions are checked once, before coordinate-derived offset arithmetic.
struct Source<'a> {
    image: &'a Image,
    width: usize,
}

impl<'a> Source<'a> {
    fn new(image: &'a Image) -> Option<Self> {
        if image.ch == 0 || sample_len(image.rect, image.ch)? != image.data.len() {
            return None;
        }
        Some(Self { image, width: image.rect.width() as usize })
    }

    #[inline]
    fn pixel(&self, x: i64, y: i64) -> Option<&'a [f32]> {
        let b = self.image.rect;
        if x < i64::from(b.x0) || x >= i64::from(b.x1) || y < i64::from(b.y0) || y >= i64::from(b.y1) {
            return None;
        }
        // The rectangular storage length was checked by Source::new, and the
        // coordinates above are inside that rectangle, so these products fit.
        let i = ((y - i64::from(b.y0)) as usize * self.width + (x - i64::from(b.x0)) as usize) * self.image.ch;
        self.image.data.get(i..i + self.image.ch)
    }
}

#[inline]
fn taps(x: f32, y: f32) -> Option<[(i64, i64, f32); 4]> {
    if !x.is_finite() || !y.is_finite() {
        return None;
    }
    let (fx, fy) = (x - 0.5, y - 0.5);
    let (x0, y0) = (fx.floor(), fy.floor());
    let (ax, ay) = (fx - x0, fy - y0);
    // f32 casts saturate, and i64 leaves room for the neighbouring tap even
    // when an adversarial center places a sample outside the i32 image domain.
    let (x0, y0) = (i64::from(x0 as i32), i64::from(y0 as i32));
    Some([(x0, y0, (1.0 - ax) * (1.0 - ay)), (x0 + 1, y0, ax * (1.0 - ay)), (x0, y0 + 1, (1.0 - ax) * ay), (x0 + 1, y0 + 1, ax * ay)])
}

#[inline]
fn sample_fixed<const N: usize, const ALPHA: bool>(src: &Source<'_>, x: f32, y: f32, acc: &mut [f32; N]) {
    let Some(taps) = taps(x, y) else { return };
    let mut sample = [0.0; N];
    for (x, y, weight) in taps {
        if weight <= 0.0 {
            continue;
        }
        let Some(pixel) = src.pixel(x, y) else { continue };
        let Ok(pixel) = <&[f32; N]>::try_from(pixel) else { continue };
        if ALPHA {
            let alpha = pixel.last().copied().unwrap_or(0.0);
            for (sum, &value) in sample.iter_mut().zip(pixel).take(N.saturating_sub(1)) {
                *sum += value * alpha * weight;
            }
            if let Some(sum) = sample.last_mut() {
                *sum += alpha * weight;
            }
        } else {
            for (sum, &value) in sample.iter_mut().zip(pixel) {
                *sum += value * weight;
            }
        }
    }
    accumulate_sample(acc, &sample, ALPHA);
}

#[inline]
fn sample_generic<const ALPHA: bool>(src: &Source<'_>, x: f32, y: f32, acc: &mut [f32], sample: &mut [f32]) {
    let Some(taps) = taps(x, y) else { return };
    sample.fill(0.0);
    for (x, y, weight) in taps {
        if weight <= 0.0 {
            continue;
        }
        let Some(pixel) = src.pixel(x, y) else { continue };
        if ALPHA {
            let alpha = pixel.last().copied().unwrap_or(0.0);
            let colours = sample.len().saturating_sub(1);
            for (sum, &value) in sample.iter_mut().zip(pixel).take(colours) {
                *sum += value * alpha * weight;
            }
            if let Some(sum) = sample.last_mut() {
                *sum += alpha * weight;
            }
        } else {
            for (sum, &value) in sample.iter_mut().zip(pixel) {
                *sum += value * weight;
            }
        }
    }
    accumulate_sample(acc, sample, ALPHA);
}

#[inline]
fn accumulate_sample(acc: &mut [f32], sample: &[f32], alpha: bool) {
    // Retain Image::sample's unpremultiply followed by average_samples'
    // premultiply, including its floating-point rounding and unusual alpha.
    let a = sample.last().copied().unwrap_or(0.0);
    let colours = sample.len().saturating_sub(1);
    for (c, (sum, &value)) in acc.iter_mut().zip(sample).enumerate() {
        *sum += if alpha && c < colours {
            let straight = if a > 0.0 { value / a } else { value };
            straight * a
        } else {
            value
        };
    }
}

#[inline]
fn finish(pixel: &mut [f32], reciprocal: f32, alpha: bool) {
    for value in pixel.iter_mut() {
        *value *= reciprocal;
    }
    if alpha {
        let a = pixel.last().copied().unwrap_or(0.0);
        let colours = pixel.len().saturating_sub(1);
        for value in pixel.iter_mut().take(colours) {
            *value = if a > 1e-7 { *value / a } else { 0.0 };
        }
    }
}

/// Original implementation, retained only as a quality/performance oracle.
#[cfg(test)]
pub(crate) fn reference(src: &Image, out: Rect, ctx: &Ctx, amount: f32, method: RadialMethod, center: (f32, f32)) -> Vec<f32> {
    super::radial(src, out, ctx, amount, method, RadialQuality::Draft, center)
}

#[cfg(test)]
mod tests;
