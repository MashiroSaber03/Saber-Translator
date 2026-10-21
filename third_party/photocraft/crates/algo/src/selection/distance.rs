//! Binary Euclidean distance transform: nearest seeds along columns, then a lower envelope
//! of squared-distance parabolas along rows (Felzenszwalb and Huttenlocher, 2012).

const NO_SEED: u32 = u32::MAX;
// Keep the public API's finite result for a mask with no selected pixel.
const EMPTY_DISTANCE: f32 = 1e10;
const BAND_ROWS: usize = 32;

/// Euclidean distance to the nearest `true` pixel, in row-major order.
///
/// Empty or inconsistent dimensions return an empty buffer. A mask with no seeds returns
/// `1e10` at every pixel, as before. Native passes run in parallel; wasm uses the same scalar
/// kernel. Working storage is one u32 per pixel in addition to the returned f32 buffer, plus
/// a 32-row band and a row envelope per task.
pub fn edt(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    let Some(n) = w.checked_mul(h) else { return Vec::new() };
    if n == 0 || n > isize::MAX as usize / std::mem::size_of::<u32>() || n != inside.len() || u32::try_from(w).is_err() || u32::try_from(h).is_err() {
        return Vec::new();
    }
    if !inside.iter().any(|seed| *seed) {
        return vec![EMPTY_DISTANCE; n];
    }
    if inside.iter().all(|seed| *seed) {
        return vec![0.0; n];
    }
    let Some(band_len) = w.checked_mul(BAND_ROWS) else { return Vec::new() };
    // Column-major storage makes both distance sweeps contiguous. Only the boolean input
    // is read with a stride. Binary costs need two nearest-seed sweeps, not an envelope.
    let mut vertical = vec![NO_SEED; n];
    for_chunks(&mut vertical, h, |x, column| {
        let mut nearest = None;
        for (y, d) in column.iter_mut().enumerate() {
            if inside.get(y * w + x).copied().unwrap_or(false) {
                nearest = Some(y);
            }
            if let Some(seed) = nearest {
                *d = (y - seed) as u32;
            }
        }
        nearest = None;
        for (y, d) in column.iter_mut().enumerate().rev() {
            if *d == 0 {
                nearest = Some(y);
            }
            if let Some(seed) = nearest {
                *d = (*d).min((seed - y) as u32);
            }
        }
    });
    let mut out = vec![0.0; n];
    for_chunks(&mut out, band_len, |band, dst| {
        let y0 = band * BAND_ROWS;
        let rows = dst.len() / w;
        let mut costs = vec![NO_SEED; dst.len()];
        // Copy small column spans into a row band. Reading whole rows straight from the
        // transposed plane would revisit one cache line per pixel on a large canvas.
        for (x, column) in vertical.chunks_exact(h).enumerate() {
            if let Some(span) = column.get(y0..y0 + rows) {
                for (y, d) in span.iter().enumerate() {
                    if let Some(cost) = costs.get_mut(y * w + x) {
                        *cost = *d;
                    }
                }
            }
        }
        let mut sites = vec![0usize; w];
        let mut starts = vec![0.0f64; w];
        for (row, result) in costs.chunks_exact(w).zip(dst.chunks_exact_mut(w)) {
            row_distance(row, result, &mut sites, &mut starts);
        }
    });
    out
}

fn for_chunks<T: Send + Sync>(buf: &mut [T], stride: usize, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    // Avoid starting Rayon tasks for the tiny masks used by tools and thumbnails.
    if buf.len() < 65_536 {
        for (i, chunk) in buf.chunks_mut(stride).enumerate() {
            f(i, chunk);
        }
    } else {
        crate::photo_util::par_rows(buf, stride, 1, f);
    }
}

/// Lower envelope of parabolas with integer vertical distances. Work is O(row length): each
/// site enters and leaves the envelope at most once. Empty columns never become candidates.
fn row_distance(costs: &[u32], out: &mut [f32], sites: &mut [usize], starts: &mut [f64]) {
    let mut count = 0usize;
    for (q, &distance) in costs.iter().enumerate().filter(|(_, d)| **d != NO_SEED) {
        let fq = f64::from(distance).powi(2);
        let mut start = f64::NEG_INFINITY;
        while count > 0 {
            let Some(&p) = sites.get(count - 1) else { return };
            let Some(&dp) = costs.get(p) else { return };
            // (q² - p²)/(2(q-p)) = (q+p)/2. This avoids subtracting large rounded
            // coordinate squares. f64 boundaries retain the half-pixel precision on PSBs.
            start = (fq - f64::from(dp).powi(2)) / (2.0 * (q - p) as f64) + (q as f64 + p as f64) * 0.5;
            if starts.get(count - 1).is_some_and(|s| start > *s) {
                break;
            }
            count -= 1;
        }
        if count == 0 {
            start = f64::NEG_INFINITY;
        }
        if let (Some(site), Some(boundary)) = (sites.get_mut(count), starts.get_mut(count)) {
            *site = q;
            *boundary = start;
        }
        count += 1;
    }
    if count == 0 {
        out.fill(EMPTY_DISTANCE);
        return;
    }
    let mut k = 0usize;
    for (x, d) in out.iter_mut().enumerate() {
        while k + 1 < count && starts.get(k + 1).is_some_and(|s| *s < x as f64) {
            k += 1;
        }
        if let Some(&site) = sites.get(k)
            && let Some(&vertical) = costs.get(site)
        {
            let dx = x as f64 - site as f64;
            *d = (dx * dx + f64::from(vertical).powi(2)).sqrt() as f32;
        }
    }
}

#[cfg(test)]
mod tests;
