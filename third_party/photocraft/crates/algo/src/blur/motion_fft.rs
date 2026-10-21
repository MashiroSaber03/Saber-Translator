//! Overlap-save convolution of the discrete Motion Blur kernel.
//!
//! The spectrum uses the upstream row kernel's merged f32 coefficients, with
//! no intermediate image resampling. Transforms use doubles and keep alpha
//! premultiplied until the final output. The summation order differs from the
//! row kernel, which remains the fallback.

use std::sync::Arc;

use photocraft_geom::Rect;
use photocraft_raster::Interrupt;
use rustfft::{Fft, FftPlanner, num_complex::Complex};

use crate::Image;

const MAX_POINTS: usize = 4 << 20;
const MAX_SIDE: usize = 8192;
const MAX_BYTES: usize = 128 << 20;
const MAX_SAMPLE: f32 = 1e10;
pub(super) const COORDINATE_LIMIT: i32 = 16_384;

type Number = Complex<f64>;

fn buffer<T: Clone>(len: usize, value: T) -> Option<Vec<T>> {
    if len.checked_mul(std::mem::size_of::<T>())? > MAX_BYTES {
        return None;
    }
    let mut result = Vec::new();
    result.try_reserve_exact(len).ok()?;
    result.resize(len, value);
    Some(result)
}

fn dimensions(rect: Rect) -> Option<(usize, usize)> {
    Some((usize::try_from(i64::from(rect.x1) - i64::from(rect.x0)).ok()?, usize::try_from(i64::from(rect.y1) - i64::from(rect.y0)).ok()?))
}

pub(super) fn ordinary_coordinates(rect: Rect) -> bool {
    [rect.x0, rect.y0, rect.x1, rect.y1].iter().all(|v| (-COORDINATE_LIMIT..=COORDINATE_LIMIT).contains(v))
}

// Smooth lengths avoid the almost fourfold padding of a 512px tile plus a
// short halo to a 1024x1024 power-of-two transform. RustFFT supports these
// mixed-radix lengths efficiently.
fn fft_size(needed: usize) -> Option<usize> {
    if needed == 0 || needed > MAX_SIDE {
        return None;
    }
    let mut best = MAX_SIDE;
    let mut a = 1usize;
    while a <= MAX_SIDE {
        let mut b = a;
        while b <= MAX_SIDE {
            let mut c = b;
            while c <= MAX_SIDE {
                if c >= needed {
                    best = best.min(c);
                }
                c = c.checked_mul(5)?;
            }
            b = b.checked_mul(3)?;
        }
        a = a.checked_mul(2)?;
    }
    Some(best)
}

#[derive(Clone, Copy)]
struct Tap {
    x: i32,
    y: i32,
    weight: f64,
}

fn taps(angle: f32, distance: f32, clip: Option<(Rect, Rect)>) -> Option<Vec<Tap>> {
    let distance = distance.abs();
    if !angle.is_finite() || !(64.0..=2000.0).contains(&distance) {
        return None;
    }
    let steps = distance.ceil() as i32;
    let (sin, cos) = angle.to_radians().sin_cos();
    let norm = 1.0 / (steps + 1) as f32;
    let capacity = usize::try_from(steps + 1).ok()?.checked_mul(4)?;
    let mut merged: Vec<(i32, i32, f32)> = Vec::new();
    merged.try_reserve_exact(capacity).ok()?;
    let mut slots = std::collections::HashMap::new();
    slots.try_reserve(capacity).ok()?;
    for i in 0..=steps {
        // Match PR #902's f32 geometry, normalized weights and first-seen merge order.
        let t = i as f32 / steps as f32 - 0.5;
        let (x, y) = (cos * distance * t, -sin * distance * t);
        let (ix, iy) = (x.floor() as i32, y.floor() as i32);
        let (fx, fy) = (x - ix as f32, y - iy as f32);
        for (x, y, weight) in [(ix, iy, (1.0 - fx) * (1.0 - fy)), (ix + 1, iy, fx * (1.0 - fy)), (ix, iy + 1, (1.0 - fx) * fy), (ix + 1, iy + 1, fx * fy)] {
            if weight <= 0.0 {
                continue;
            }
            let slot = *slots.entry((x, y)).or_insert_with(|| {
                merged.push((x, y, 0.0));
                merged.len() - 1
            });
            merged.get_mut(slot)?.2 += weight * norm;
        }
    }
    let mut result = Vec::new();
    result.try_reserve_exact(merged.len()).ok()?;
    for (x, y, weight) in merged {
        if let Some((source, out)) = clip {
            // Omitted reads are transparent; retain the complete kernel's coefficients.
            if i64::from(out.x0) + i64::from(x) >= i64::from(source.x1)
                || i64::from(out.x1) - 1 + i64::from(x) < i64::from(source.x0)
                || i64::from(out.y0) + i64::from(y) >= i64::from(source.y1)
                || i64::from(out.y1) - 1 + i64::from(y) < i64::from(source.y0)
            {
                continue;
            }
        }
        result.push(Tap { x, y, weight: f64::from(weight) });
    }
    Some(result)
}

/// One pixel of the upstream row kernel, without allocating its premultiplied
/// square window. Coefficients and accumulation retain their original f32 order.
fn sample_row_point(src: &Image, x: i32, y: i32, alpha: bool, taps: &[Tap], ctl: &Interrupt) -> Option<[f32; 5]> {
    let (width, height) = dimensions(src.rect)?;
    if !(1..=5).contains(&src.ch) || width.checked_mul(height)?.checked_mul(src.ch)? != src.data.len() || !ordinary_coordinates(src.rect) {
        return None;
    }
    let mut result = [0.0f32; 5];
    for (i, tap) in taps.iter().enumerate() {
        if i % 64 == 0 && ctl.cancelled() {
            return None;
        }
        let (sx, sy) = (x.checked_add(tap.x)?, y.checked_add(tap.y)?);
        let a = if alpha { src.get(sx, sy, src.ch - 1) } else { 1.0 };
        // Each stored coefficient came directly from f32 and converts back
        // exactly. Match the row kernel's weight * premultiplied sample order.
        let weight = tap.weight as f32;
        for (c, value) in result.iter_mut().take(src.ch).enumerate() {
            let sample = src.get(sx, sy, c);
            let premultiplied = if alpha && c + 1 < src.ch { sample * a } else { sample };
            *value += weight * premultiplied;
        }
    }
    if alpha {
        let a = *result.get(src.ch - 1)?;
        for value in result.iter_mut().take(src.ch - 1) {
            *value = if a > 1e-7 { *value / a } else { 0.0 };
        }
    }
    Some(result)
}

/// Conservative native crossover against the merged row kernel. Wide rows can
/// be faster than transforms unless FFT tiles retain enough parallelism. The
/// release matrix in motion_bench covers both winners and declined cases.
#[allow(clippy::too_many_arguments)]
pub(super) fn worthwhile(angle: f32, distance: f32, tile: usize, area: Rect, row_tile: usize, channels: usize, workers: usize) -> Option<bool> {
    if workers < 4 || !(1..=5).contains(&channels) || row_tile == 0 {
        return Some(false);
    }
    let pairs = channels.div_ceil(2);
    // Transforming a channel pair costs almost the same as one channel. Stay
    // on rows for short streaks, with a higher crossover for odd channel counts.
    if !distance.is_finite() || distance.abs() * (channels as f32) < 1500.0 * pairs as f32 {
        return Some(false);
    }
    let taps = taps(angle, distance, None)?;
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (0, 0, 0, 0);
    for tap in &taps {
        min_x = min_x.min(tap.x);
        max_x = max_x.max(tap.x);
        min_y = min_y.min(tap.y);
        max_y = max_y.max(tap.y);
    }
    let sx = usize::try_from(max_x - min_x).ok()?;
    let sy = usize::try_from(max_y - min_y).ok()?;
    let width = fft_size(tile.checked_add(sx)?)?;
    let height = fft_size(tile.checked_add(sy)?)?;
    let points = width.checked_mul(height)?;
    if points > MAX_POINTS {
        return Some(false);
    }
    let (ow, oh) = dimensions(area)?;
    let tiles = ow.div_ceil(tile).checked_mul(oh.div_ceil(tile))?;
    let row_tiles = ow.div_ceil(row_tile).checked_mul(oh.div_ceil(row_tile))?;
    let row_workers = workers.min(row_tiles);
    // Before creating any FFT plans, conservatively estimate the same complete
    // working-set budget as the scheduler. Mixed-radix scratch is bounded here
    // by four times the longest side; the real plan supplies its exact size.
    let source = tile.checked_add(sx)?.checked_mul(tile.checked_add(sy)?)?.checked_mul(channels)?.checked_mul(4)?;
    let result = tile.checked_mul(tile)?.checked_mul(channels)?;
    let scratch = width.max(height).checked_mul(4)?;
    let worker =
        points.checked_mul(2)?.checked_add(scratch)?.checked_mul(16)?.checked_add(source)?.checked_add(result.checked_mul(8)?)?.max(source.checked_mul(2)?);
    let resident = points
        .checked_mul(16)?
        .checked_add(width.checked_add(height)?.checked_mul(256)?)?
        .checked_add(taps.capacity().checked_mul(std::mem::size_of::<Tap>())?)?;
    let metadata = tiles.checked_mul(std::mem::size_of::<(Rect, Rect)>())?;
    let available = (512usize << 20).checked_sub(resident)?.checked_sub(metadata)?;
    let memory_workers = available.checked_div(worker.checked_add(result.checked_mul(4)?)?)?;
    let fft_workers = workers.min(memory_workers).min(tiles);
    // A few large diagonal transforms can lose to well-parallelized rows.
    // Admit them only when their task count gives a clear parallelism advantage.
    let parallel = fft_workers >= row_workers.checked_mul(2)? || (fft_workers >= 4 && fft_workers.checked_mul(2)? >= row_workers);
    let transform_work = points.checked_mul((width.ilog2() + height.ilog2()) as usize)?.checked_mul(tiles)?;
    let row_work = ow.checked_mul(oh)?.checked_mul(taps.len())?;
    Some(parallel && transform_work < row_work)
}

/// Immutable FFT plans and kernel spectrum, shared by every tile of one apply.
pub(super) struct Plan {
    distance: f32,
    width: usize,
    height: usize,
    tile_width: usize,
    tile_height: usize,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
    kernel: Vec<Number>,
    taps: Vec<Tap>,
    row_forward: Arc<dyn Fft<f64>>,
    col_forward: Arc<dyn Fft<f64>>,
    row_inverse: Arc<dyn Fft<f64>>,
    col_inverse: Arc<dyn Fft<f64>>,
    scratch_len: usize,
}

impl Plan {
    pub(super) fn new_with(angle: f32, distance: f32, tile: usize, ctl: &Interrupt) -> Option<Self> {
        Self::build(angle, distance, tile, tile, None, ctl)
    }

    fn build(angle: f32, distance: f32, tile_width: usize, tile_height: usize, clip: Option<(Rect, Rect)>, ctl: &Interrupt) -> Option<Self> {
        if ctl.cancelled() {
            return None;
        }
        if tile_width == 0 || tile_height == 0 || tile_width > 512 || tile_height > 512 {
            return None;
        }
        let taps = taps(angle, distance, clip)?;
        let (mut min_x, mut max_x, mut min_y, mut max_y) = (0, 0, 0, 0);
        for tap in &taps {
            min_x = min_x.min(tap.x);
            max_x = max_x.max(tap.x);
            min_y = min_y.min(tap.y);
            max_y = max_y.max(tap.y);
        }
        let width = fft_size(tile_width.checked_add(usize::try_from(max_x - min_x).ok()?)?)?;
        let height = fft_size(tile_height.checked_add(usize::try_from(max_y - min_y).ok()?)?)?;
        let count = width.checked_mul(height)?;
        if count > MAX_POINTS {
            return None;
        }
        let mut kernel = buffer(count, Number::default())?;
        for tap in &taps {
            // Convolution reverses the displacement of a correlation tap.
            let x = usize::try_from((-tap.x).rem_euclid(i32::try_from(width).ok()?)).ok()?;
            let y = usize::try_from((-tap.y).rem_euclid(i32::try_from(height).ok()?)).ok()?;
            kernel.get_mut(y.checked_mul(width)?.checked_add(x)?)?.re += tap.weight;
        }
        let mut planner = FftPlanner::<f64>::new();
        let row_forward = planner.plan_fft_forward(width);
        let col_forward = planner.plan_fft_forward(height);
        let row_inverse = planner.plan_fft_inverse(width);
        let col_inverse = planner.plan_fft_inverse(height);
        let scratch_len = [
            row_forward.get_inplace_scratch_len(),
            col_forward.get_inplace_scratch_len(),
            row_inverse.get_inplace_scratch_len(),
            col_inverse.get_inplace_scratch_len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let mut result = Self {
            distance,
            width,
            height,
            tile_width,
            tile_height,
            min_x,
            max_x,
            min_y,
            max_y,
            kernel: Vec::new(),
            taps,
            row_forward,
            col_forward,
            row_inverse,
            col_inverse,
            scratch_len,
        };
        let mut transposed = buffer(count, Number::default())?;
        let mut scratch = buffer(scratch_len, Number::default())?;
        result.transform(&mut kernel, &mut transposed, &mut scratch, false, ctl)?;
        result.kernel = kernel;
        Some(result)
    }

    pub(super) fn resident_bytes(&self) -> usize {
        // Include a conservative allowance for the FFT plans' twiddle tables.
        self.kernel.len() * std::mem::size_of::<Number>() + self.taps.capacity() * std::mem::size_of::<Tap>() + (self.width + self.height) * 256
    }

    pub(super) fn worker_bytes(&self, channels: usize, tile: usize) -> Option<usize> {
        let source = tile
            .checked_add(usize::try_from(self.max_x - self.min_x).ok()?)?
            .checked_mul(tile.checked_add(usize::try_from(self.max_y - self.min_y).ok()?)?)?
            .checked_mul(channels)?
            .checked_mul(4)?;
        self.kernel
            .len()
            .checked_mul(2)?
            .checked_add(self.scratch_len)?
            .checked_mul(std::mem::size_of::<Number>())?
            .checked_add(source)?
            .checked_add(tile.checked_mul(tile)?.checked_mul(channels)?.checked_mul(8)?)
    }

    pub(super) fn read_rect(&self, out: Rect) -> Option<Rect> {
        Some(Rect::new(out.x0.checked_add(self.min_x)?, out.y0.checked_add(self.min_y)?, out.x1.checked_add(self.max_x)?, out.y1.checked_add(self.max_y)?))
    }

    fn transform(&self, data: &mut [Number], transposed: &mut [Number], scratch: &mut [Number], inverse: bool, ctl: &Interrupt) -> Option<()> {
        let (rows, cols) = if inverse { (&self.row_inverse, &self.col_inverse) } else { (&self.row_forward, &self.col_forward) };
        // Every row has the planned length, and scratch_len covers all four
        // plans. Those invariants satisfy RustFFT's checked process API.
        for (i, row) in data.chunks_exact_mut(self.width).enumerate() {
            if i % 8 == 0 && ctl.cancelled() {
                return None;
            }
            rows.process_with_scratch(row, scratch);
        }
        transpose(data, transposed, self.width, self.height, ctl)?;
        for (i, row) in transposed.chunks_exact_mut(self.height).enumerate() {
            if i % 8 == 0 && ctl.cancelled() {
                return None;
            }
            cols.process_with_scratch(row, scratch);
        }
        transpose(transposed, data, self.height, self.width, ctl)
    }

    pub(super) fn filter(&self, src: &Image, out: Rect, alpha: bool, ctl: &Interrupt) -> Option<Vec<f32>> {
        let (ow, oh) = dimensions(out)?;
        if out.is_empty() {
            return Some(Vec::new());
        }
        let (sw, sh) = dimensions(src.rect)?;
        if !(1..=5).contains(&src.ch)
            || sw.checked_mul(sh)?.checked_mul(src.ch)? != src.data.len()
            || ow > self.tile_width
            || oh > self.tile_height
            || !ordinary_coordinates(src.rect)
            || !ordinary_coordinates(out)
        {
            return None;
        }
        let mut scales = [0.0f64; 5];
        for (i, pixel) in src.data.chunks_exact(src.ch).enumerate() {
            if i % 4096 == 0 && ctl.cancelled() {
                return None;
            }
            if pixel.iter().any(|value| !value.is_finite() || value.abs() > MAX_SAMPLE)
                || (alpha && !pixel.last().is_some_and(|value| (0.0..=1.0).contains(value)))
            {
                return None;
            }
            let a = if alpha { *pixel.last()? } else { 1.0 };
            for (c, &value) in pixel.iter().enumerate() {
                let value = if alpha && c + 1 < src.ch { value * a } else { value };
                let scale = scales.get_mut(c)?;
                *scale = scale.max(f64::from(value.abs()));
            }
        }
        let mut result = buffer(ow.checked_mul(oh)?.checked_mul(src.ch)?, 0.0f64)?;
        let read = self.read_rect(out)?;
        let used = read.intersect(&src.rect);
        if used.is_empty() || scales.iter().take(src.ch).all(|&scale| scale == 0.0) {
            return buffer(result.len(), 0.0f32);
        }
        let count = self.kernel.len();
        let mut data = buffer(count, Number::default())?;
        let mut transposed = buffer(count, Number::default())?;
        let mut scratch = buffer(self.scratch_len, Number::default())?;
        let norm = 1.0 / count as f64;
        for channel in (0..src.ch).step_by(2) {
            if ctl.cancelled() {
                return None;
            }
            data.fill(Number::default());
            for y in used.y0..used.y1 {
                if (y - used.y0) % 8 == 0 && ctl.cancelled() {
                    return None;
                }
                let source_start =
                    usize::try_from(y - src.rect.y0).ok()?.checked_mul(sw)?.checked_add(usize::try_from(used.x0 - src.rect.x0).ok()?)?.checked_mul(src.ch)?;
                let source_len = usize::try_from(used.x1 - used.x0).ok()?.checked_mul(src.ch)?;
                let row = src.data.get(source_start..source_start.checked_add(source_len)?)?;
                let start = usize::try_from(y - read.y0).ok()?.checked_mul(self.width)?.checked_add(usize::try_from(used.x0 - read.x0).ok()?)?;
                let dst = data.get_mut(start..start.checked_add(source_len / src.ch)?)?;
                for (pixel, dst) in row.chunks_exact(src.ch).zip(dst) {
                    let a = if alpha { *pixel.last()? } else { 1.0 };
                    // Normalize each channel separately before packing a pair.
                    // Otherwise a bright HDR colour's FFT residual could swamp
                    // a small alpha channel sharing the imaginary component.
                    let sample = |c: usize| -> f64 {
                        match (pixel.get(c), scales.get(c)) {
                            (Some(&value), Some(&scale)) if scale > 0.0 => f64::from(if alpha && c + 1 < src.ch { value * a } else { value }) / scale,
                            _ => 0.0,
                        }
                    };
                    *dst = Number::new(sample(channel), sample(channel + 1));
                }
            }
            self.transform(&mut data, &mut transposed, &mut scratch, false, ctl)?;
            for (value, kernel) in data.iter_mut().zip(&self.kernel) {
                *value *= kernel;
            }
            self.transform(&mut data, &mut transposed, &mut scratch, true, ctl)?;
            let x0 = usize::try_from(-self.min_x).ok()?;
            let y0 = usize::try_from(-self.min_y).ok()?;
            for (y, dst) in result.chunks_exact_mut(ow * src.ch).enumerate() {
                let start = y.checked_add(y0)?.checked_mul(self.width)?.checked_add(x0)?;
                let values = data.get(start..start.checked_add(ow)?)?;
                for (pixel, value) in dst.chunks_exact_mut(src.ch).zip(values) {
                    *pixel.get_mut(channel)? = value.re * norm * scales.get(channel)?;
                    if let Some(next) = pixel.get_mut(channel + 1) {
                        *next = value.im * norm * scales.get(channel + 1)?;
                    }
                }
            }
        }
        if alpha {
            // The row kernel accumulates in f32. Recompute very small alpha
            // near its cutoff so a different summation order cannot expose colour.
            let uncertainty = f64::from(self.distance.abs().ceil()) * f64::from(f32::EPSILON) * scales.get(src.ch - 1)?;
            for (i, pixel) in result.chunks_exact_mut(src.ch).enumerate() {
                if i % 16 == 0 && ctl.cancelled() {
                    return None;
                }
                let a = pixel.last_mut()?;
                // Valid alpha is nonnegative. Remove transform roundoff at
                // zero to keep fully transparent regions empty.
                if a.abs() < 1e-14 * scales.get(src.ch - 1)? {
                    *a = 0.0;
                }
                *a = a.clamp(0.0, 1.0);
                let a = *a;
                // The row kernel's f32 sum can straddle the cutoff at long distances.
                // Keep its branch and straight colour exactly for that narrow
                // range, rather than amplifying a tiny alpha difference.
                if a > 0.0 && (a <= 1.01e-7 + uncertainty || (0.99e-7..=1.01e-7).contains(&(a as f32))) {
                    let x = out.x0.checked_add(i32::try_from(i % ow).ok()?)?;
                    let y = out.y0.checked_add(i32::try_from(i / ow).ok()?)?;
                    let original = sample_row_point(src, x, y, alpha, &self.taps, ctl)?;
                    for (dst, value) in pixel.iter_mut().zip(original) {
                        *dst = f64::from(value);
                    }
                    continue;
                }
                for value in pixel.iter_mut().take(src.ch - 1) {
                    *value = if a as f32 > 1e-7 { *value / a } else { 0.0 };
                }
            }
        }
        let mut output = buffer(result.len(), 0.0f32)?;
        for (dst, value) in output.iter_mut().zip(result) {
            *dst = value as f32;
        }
        Some(output)
    }
}

#[cfg(test)]
pub(super) fn row_point_for_test(src: &Image, x: i32, y: i32, alpha: bool, angle: f32, distance: f32) -> Option<Vec<f32>> {
    let taps = taps(angle, distance, None)?;
    let result = sample_row_point(src, x, y, alpha, &taps, &Interrupt::NONE)?;
    Some(result.into_iter().take(src.ch).collect())
}

fn transpose(src: &[Number], dst: &mut [Number], width: usize, height: usize, ctl: &Interrupt) -> Option<()> {
    // Blocking keeps both row-major matrices in cache while transposing.
    for by in (0..height).step_by(32) {
        if ctl.cancelled() {
            return None;
        }
        for bx in (0..width).step_by(32) {
            for y in by..(by + 32).min(height) {
                let start = y.checked_mul(width)?.checked_add(bx)?;
                let row = src.get(start..start.checked_add((width - bx).min(32))?)?;
                for (x, value) in row.iter().enumerate() {
                    *dst.get_mut((bx + x).checked_mul(height)?.checked_add(y)?)? = *value;
                }
            }
        }
    }
    Some(())
}

/// Small-window oracle tests omit taps whose reads must be transparent. The
/// production plan covers the whole kernel and is reused across output tiles.
#[cfg(test)]
pub(super) fn filter(src: &Image, out: Rect, alpha: bool, angle: f32, distance: f32) -> Option<Vec<f32>> {
    if out.is_empty() {
        return Some(Vec::new());
    }
    let (width, height) = dimensions(out)?;
    Plan::build(angle, distance, width, height, Some((src.rect, out)), &Interrupt::NONE)?.filter(src, out, alpha, &Interrupt::NONE)
}
