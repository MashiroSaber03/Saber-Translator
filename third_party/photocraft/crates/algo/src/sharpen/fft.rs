//! Wide exact-kernel Gaussian convolution using FFTs. Summation order changes:
//! positive thresholds and unstable samples retain the sequential f32 fallback.
use crate::{FilterParams, Image};
use photocraft_geom::Rect;
use photocraft_raster::{Interrupt, Surface};
use rayon::prelude::*;
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

const MAX_BYTES: usize = 1 << 30;
const MAX_SIDE: usize = 32_768;
const STRIPE: usize = 32;
type Number = Complex<f64>;

fn buffer<T: Clone>(len: usize, value: T) -> Option<Vec<T>> {
    if len.checked_mul(std::mem::size_of::<T>())? > MAX_BYTES {
        return None;
    }
    let mut data = Vec::new();
    data.try_reserve_exact(len).ok()?;
    data.resize(len, value);
    Some(data)
}

struct Plan {
    size: usize,
    forward: Arc<dyn Fft<f64>>,
    inverse: Arc<dyn Fft<f64>>,
    spectrum: Vec<Number>,
    scratch: usize,
}
impl Plan {
    fn new(needed: usize, kernel: &[f32], planner: &mut FftPlanner<f64>) -> Option<Self> {
        let size = needed.checked_next_power_of_two()?;
        if size > MAX_SIDE || kernel.len() > size {
            return None;
        }
        let forward = planner.plan_fft_forward(size);
        let inverse = planner.plan_fft_inverse(size);
        let scratch = forward.get_inplace_scratch_len().max(inverse.get_inplace_scratch_len());
        let mut spectrum = buffer(size, Number::default())?;
        for (dst, &weight) in spectrum.iter_mut().zip(kernel) {
            dst.re = f64::from(weight) / size as f64;
        }
        forward.process_with_scratch(&mut spectrum, &mut buffer(scratch, Number::default())?);
        Some(Self { size, forward, inverse, spectrum, scratch })
    }
    fn work(&self) -> Option<(Vec<Number>, Vec<Number>)> {
        Some((buffer(self.size, Number::default())?, buffer(self.scratch, Number::default())?))
    }
    fn convolve(&self, data: &mut [Number], scratch: &mut [Number]) -> Option<()> {
        if data.len() != self.size || scratch.len() < self.scratch {
            return None;
        }
        self.forward.process_with_scratch(data, scratch);
        for (d, k) in data.iter_mut().zip(&self.spectrum) {
            *d *= k;
        }
        self.inverse.process_with_scratch(data, scratch);
        Some(())
    }
}

// Bound the complete temporary working set, including all concurrently live
// column gathers, originals, FFT scratch and result stripes. Document tiles are
// excluded, just as in the existing tile scheduler's scratch budget.
fn group_size(resident: usize, worker: usize, count: usize, workers: usize) -> Option<usize> {
    let group = MAX_BYTES.checked_sub(resident)?.checked_div(worker.max(1))?.min(workers).min(count);
    (group > 0).then_some(group)
}

/// `None` declines the optimization, `Some(None)` propagates cancellation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn apply(
    surface: &Surface,
    params: &FilterParams,
    area: Rect,
    selection: Option<&Surface>,
    extent: Option<Rect>,
    ctl: &Interrupt,
) -> Option<Option<Surface>> {
    let FilterParams::UnsharpMask { amount, radius, threshold } = *params else { return None };
    // At a positive threshold, even a tiny sum-order difference can switch the
    // entire sharpen contribution. Zero/negative thresholds always keep it.
    if !(64.0..=1000.0).contains(&radius)
        || threshold > 0.0
        || amount == 0.0
        || !super::exact::valid_params(amount, radius, threshold)
        || !super::exact::valid_area(area)
        || area.is_empty()
        || u64::from(area.width()) * u64::from(area.height()) < 65_536
        || !(1..=crate::fxutil::MAXC).contains(&surface.channels())
        || selection.is_some_and(|s| s.channels() == 0)
        || extent.is_some_and(|e| !super::exact::valid_area(e) || area.intersect(&e) != area)
    {
        return None;
    }
    if ctl.cancelled() {
        return Some(None);
    }
    match run(surface, area, amount, radius, selection, extent, ctl) {
        Some(out) => Some(Some(out)),
        None if ctl.cancelled() => Some(None),
        None => None,
    }
}

#[allow(clippy::too_many_arguments)]
fn run(surface: &Surface, area: Rect, amount: f32, radius: f32, selection: Option<&Surface>, extent: Option<Rect>, ctl: &Interrupt) -> Option<Surface> {
    let (w, h, n) = (area.width() as usize, area.height() as usize, surface.channels());
    let alpha = surface.format().alpha;
    let kernel = super::gaussian_kernel(radius);
    let r = kernel.len() / 2;
    let reach = i32::try_from(r).ok()?;
    let (x0, x1) = (area.x0.checked_sub(reach)?, area.x1.checked_add(reach)?);
    let (y0, y1) = (area.y0.checked_sub(reach)?, area.y1.checked_add(reach)?);
    let (read_x0, read_x1, read_y0, read_y1) = match extent {
        Some(e) => (x0.max(e.x0), x1.min(e.x1), y0.max(e.y0), y1.min(e.y1)),
        None => (x0, x1, y0, y1),
    };
    let rows = usize::try_from(i64::from(read_y1) - i64::from(read_y0)).ok()?;
    let read_width = usize::try_from(i64::from(read_x1) - i64::from(read_x0)).ok()?;
    let row_len = w.checked_mul(n)?;
    let samples = row_len.checked_mul(rows)?;
    let resident = samples.checked_mul(4)?;
    let mut planner = FftPlanner::<f64>::new();
    let horizontal = Plan::new(w.checked_add(2 * r)?, &kernel, &mut planner)?;
    let vertical = Plan::new(h.checked_add(2 * r)?, &kernel, &mut planner)?;
    // A conservative allowance includes FFT plans/twiddles in addition to spectra.
    let plans = horizontal.size.checked_add(vertical.size)?.checked_mul(256)?;
    let resident = resident.checked_add(plans)?;
    let workers = rayon::current_num_threads().max(1);
    let horizontal_worker = read_width.checked_mul(n)?.checked_mul(4)?.checked_add(horizontal.size.checked_add(horizontal.scratch)?.checked_mul(16)?)?;
    let vertical_worker = STRIPE
        .checked_mul(rows)?
        .checked_mul(n)?
        .checked_mul(4)?
        .checked_add(STRIPE.checked_mul(h)?.checked_mul(n)?.checked_mul(12)?)?
        .checked_add(vertical.size.checked_add(vertical.scratch)?.checked_mul(16)?)?;
    let group = group_size(resident, horizontal_worker.max(vertical_worker), w.div_ceil(STRIPE), workers)?;
    let mut tmp = buffer(samples, 0.0f32)?;
    let finished = AtomicUsize::new(0);
    // Read each needed real row once. Outside a repeated extent, horizontal
    // results are repeated lazily by the vertical gather, not transformed again.
    for band in tmp.chunks_mut(row_len.checked_mul(group)?) {
        let first = finished.load(Ordering::Relaxed);
        let ok: Option<Vec<()>> = band
            .par_chunks_exact_mut(row_len)
            .enumerate()
            .map(|(offset, dst)| {
                if ctl.cancelled() {
                    return None;
                }
                let y = read_y0.checked_add(i32::try_from(first.checked_add(offset)?).ok()?)?;
                let mut row = buffer(read_width.checked_mul(n)?, 0.0f32)?;
                // Reservation above covers resize, so the reusable raster read cannot allocate.
                surface.read_region_into(Rect::new(read_x0, y, read_x1, y.checked_add(1)?), &mut row);
                let mut scales = [0.0f64; crate::fxutil::MAXC];
                for px in row.chunks_exact_mut(n) {
                    if px.iter().any(|v| !v.is_finite() || v.abs() > 1e6) || (alpha && !px.last().is_some_and(|a| (0.0..=1.0).contains(a))) {
                        return None;
                    }
                    let a = if alpha { *px.last()? } else { 1.0 };
                    for (c, v) in px.iter_mut().enumerate() {
                        if alpha && c + 1 < n {
                            *v *= a;
                        }
                        let scale = scales.get_mut(c)?;
                        *scale = scale.max(f64::from(v.abs()));
                    }
                }
                let (mut data, mut scratch) = horizontal.work()?;
                for channel in (0..n).step_by(2) {
                    if ctl.cancelled() {
                        return None;
                    }
                    data.fill(Number::default());
                    for (i, d) in data.iter_mut().take(w + 2 * r).enumerate() {
                        let sx = x0.checked_add(i32::try_from(i).ok()?)?.clamp(read_x0, read_x1.checked_sub(1)?);
                        let start = usize::try_from(sx - read_x0).ok()?.checked_mul(n)?;
                        let px = row.get(start..start.checked_add(n)?)?;
                        let sample = |c: usize| -> f64 {
                            match (px.get(c), scales.get(c)) {
                                (Some(v), Some(s)) if *s > 0.0 => f64::from(*v) / s,
                                _ => 0.0,
                            }
                        };
                        *d = Number::new(sample(channel), sample(channel + 1));
                    }
                    horizontal.convolve(&mut data, &mut scratch)?;
                    for (px, d) in dst.chunks_exact_mut(n).zip(data.get(2 * r..2 * r + w)?) {
                        *px.get_mut(channel)? = (d.re * scales.get(channel)?) as f32;
                        if let Some(v) = px.get_mut(channel + 1) {
                            *v = (d.im * scales.get(channel + 1)?) as f32;
                        }
                    }
                }
                finished.fetch_add(1, Ordering::Relaxed);
                Some(())
            })
            .collect();
        ok?;
        ctl.progress(0.5 * finished.load(Ordering::Relaxed) as f32 / rows as f32);
        if ctl.cancelled() {
            return None;
        }
    }
    let mut out = surface.clone();
    let count = w.div_ceil(STRIPE);
    for first in (0..count).step_by(group) {
        let end = first.checked_add(group)?.min(count);
        let results: Option<Vec<_>> = (first..end)
            .into_par_iter()
            .map(|index| {
                if ctl.cancelled() {
                    return None;
                }
                let x = index.checked_mul(STRIPE)?;
                let width = STRIPE.min(w.checked_sub(x)?);
                let mut columns = buffer(width.checked_mul(rows)?.checked_mul(n)?, 0.0f32)?;
                for (y, row) in tmp.chunks_exact(row_len).enumerate() {
                    if y % 64 == 0 && ctl.cancelled() {
                        return None;
                    }
                    for (c, px) in row.get(x * n..(x + width) * n)?.chunks_exact(n).enumerate() {
                        let start = (c * rows + y) * n;
                        columns.get_mut(start..start + n)?.copy_from_slice(px);
                    }
                }
                let rect = Rect::new(area.x0.checked_add(x as i32)?, area.y0, area.x0.checked_add((x + width) as i32)?, area.y1);
                let len = width.checked_mul(h)?.checked_mul(n)?;
                let mut result = buffer(len, 0.0f32)?;
                let mut original = buffer(len, 0.0f32)?;
                surface.read_region_into(rect, &mut original);
                let (mut data, mut scratch) = vertical.work()?;
                for (c, samples) in columns.chunks_exact(rows * n).enumerate() {
                    if ctl.cancelled() {
                        return None;
                    }
                    let mut scales = [0.0f64; crate::fxutil::MAXC];
                    for px in samples.chunks_exact(n) {
                        for (s, v) in scales.iter_mut().zip(px) {
                            *s = s.max(f64::from(v.abs()));
                        }
                    }
                    for channel in (0..n).step_by(2) {
                        if ctl.cancelled() {
                            return None;
                        }
                        data.fill(Number::default());
                        for (i, d) in data.iter_mut().take(h + 2 * r).enumerate() {
                            let sy = y0.checked_add(i as i32)?.clamp(read_y0, read_y1.checked_sub(1)?);
                            let start = usize::try_from(sy - read_y0).ok()?.checked_mul(n)?;
                            let px = samples.get(start..start + n)?;
                            let sample = |c: usize| -> f64 {
                                match (px.get(c), scales.get(c)) {
                                    (Some(v), Some(s)) if *s > 0.0 => f64::from(*v) / s,
                                    _ => 0.0,
                                }
                            };
                            *d = Number::new(sample(channel), sample(channel + 1));
                        }
                        vertical.convolve(&mut data, &mut scratch)?;
                        for (y, d) in data.get(2 * r..2 * r + h)?.iter().enumerate() {
                            let start = (y * width + c) * n;
                            *result.get_mut(start + channel)? = (d.re * scales.get(channel)?) as f32;
                            if let Some(v) = result.get_mut(start + channel + 1).filter(|_| channel + 1 < n) {
                                *v = (d.im * scales.get(channel + 1)?) as f32;
                            }
                        }
                    }
                }
                for (px, o) in result.chunks_exact_mut(n).zip(original.chunks_exact(n)) {
                    if alpha && *o.last()? <= 0.0 {
                        px.copy_from_slice(o);
                        continue;
                    }
                    let a = if alpha { *px.last()? } else { 1.0 };
                    // Near the original unpremultiply cutoff, cancellation/FFT noise
                    // or f32 sum order can expose colour. Decline the whole fast path.
                    let uncertainty = 4.0 * kernel.len() as f32 * f32::EPSILON;
                    if alpha && a <= 1e-7 + uncertainty {
                        return None;
                    }
                    for (v, o) in px.iter_mut().zip(o).take(n - usize::from(alpha)) {
                        let b = if alpha { *v / a } else { *v };
                        *v = o + amount / 100.0 * (o - b);
                        if !v.is_finite() {
                            return None;
                        }
                    }
                    if alpha {
                        *px.last_mut()? = *o.last()?;
                    }
                }
                if let Some(sel) = selection {
                    crate::mix_selection(&mut result, rect, sel, &Image { rect, ch: n, data: original }, alpha);
                }
                Some((rect, result))
            })
            .collect();
        for (rect, data) in results? {
            if ctl.cancelled() {
                return None;
            }
            out.try_write_region(rect, &data).ok()?;
        }
        ctl.progress(0.5 + 0.48 * end as f32 / count as f32);
    }
    if ctl.cancelled() {
        return None;
    }
    out.prune();
    if ctl.cancelled() {
        return None;
    }
    ctl.progress(1.0);
    Some(out)
}

#[cfg(test)]
mod tests;
