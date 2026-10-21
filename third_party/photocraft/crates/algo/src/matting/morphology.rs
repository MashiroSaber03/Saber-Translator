//! Exact flat octagonal morphology: a square followed by a Manhattan ball.

pub(super) fn morph(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    let Some(n) = w.checked_mul(h).filter(|&n| n <= src.len()) else { return Vec::new() };
    if n == 0 || r == 0 {
        return src.to_vec();
    }
    // The old max/min visitation order matters for NaN payloads and signed zero.
    // Keep that path, and small radii where setup cannot pay off.
    if r <= 2 || src[..n].iter().any(|v| v.is_nan() || (*v == 0.0 && v.is_sign_negative())) {
        return repeated(src, w, h, r, grow);
    }
    let mut out = if n >= 1048576 && w >= 256 && h >= 128 && r <= 64 { tiled(&src[..n], w, h, r, grow) } else { dense(&src[..n], w, h, r, grow) };
    out.extend_from_slice(&src[n..]);
    out
}

/// Bound scratch by a 256x128 core plus the radius halo. The returned image is
/// the only full-size allocation. Constant input tiles need no filter scratch.
fn tiled(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    let mut out = vec![0.0; src.len()];
    let run = |band: usize, rows: &mut [f32]| {
        let y = band * 128;
        let height = rows.len() / w;
        let (y0, y1) = (y.saturating_sub(r), y.saturating_add(height).saturating_add(r).min(h));
        for x in (0..w).step_by(256) {
            let width = (w - x).min(256);
            let (x0, x1) = (x.saturating_sub(r), x.saturating_add(width).saturating_add(r).min(w));
            let first = src[y0 * w + x0];
            if (y0..y1).all(|yy| src[yy * w + x0..yy * w + x1].iter().all(|&v| v == first)) {
                for row in rows.chunks_mut(w) {
                    row[x..x + width].fill(first);
                }
                continue;
            }
            let tw = x1 - x0;
            let mut input = Vec::with_capacity(tw * (y1 - y0));
            for yy in y0..y1 {
                input.extend_from_slice(&src[yy * w + x0..yy * w + x1]);
            }
            let filtered = dense(&input, tw, y1 - y0, r, grow);
            for (yy, row) in rows.chunks_mut(w).enumerate() {
                let base = (y - y0 + yy) * tw + x - x0;
                row[x..x + width].copy_from_slice(&filtered[base..base + width]);
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        out.par_chunks_mut(w * h.min(128)).enumerate().for_each(|(band, rows)| run(band, rows));
    }
    #[cfg(target_arch = "wasm32")]
    out.chunks_mut(w * h.min(128)).enumerate().for_each(|(band, rows)| run(band, rows));
    out
}

fn dense(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    // ceil(r/2) unit squares plus floor(r/2) unit crosses. Saturate radii
    // at the image extent: additional growth/erosion cannot reach another pixel.
    let extent = w.max(h) - 1;
    let square = (r / 2 + r % 2).min(extent);
    let diamond = (r / 2).min(w.saturating_add(h).saturating_sub(2));
    let mut cur = square_extreme(src, w, h, square, grow);
    if diamond > 0 {
        let mut next = vec![0.0; src.len()];
        let mut built = 0usize;
        for bit in (0..usize::BITS - diamond.leading_zeros()).rev() {
            if built > 0 {
                cross(&cur, &mut next, w, h, built, grow);
                std::mem::swap(&mut cur, &mut next);
                built *= 2;
            }
            if (diamond >> bit) & 1 != 0 {
                cross(&cur, &mut next, w, h, 1, grow);
                std::mem::swap(&mut cur, &mut next);
                built += 1;
            }
        }
    }
    cur
}

fn square_extreme(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    #[cfg(not(target_arch = "wasm32"))]
    if src.len() >= 262144 {
        return parallel_square(src, w, h, r, grow);
    }
    super::max_square(src, w, h, r, grow)
}

/// Two full-size work planes, reused across blocked transposes. Filtering each
/// column contiguously avoids the full-photo stride of the serial square pass.
#[cfg(not(target_arch = "wasm32"))]
fn parallel_square(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    use rayon::prelude::*;
    let mut rows = vec![0.0; src.len()];
    rows.par_chunks_mut(w)
        .enumerate()
        .for_each_init(|| (Vec::new(), Vec::new()), |(g, hb), (y, out)| super::running_extreme(&src[y * w..(y + 1) * w], out, r.min(w - 1), grow, g, hb));
    let mut cols = vec![0.0; src.len()];
    // Independent bands of up to 32 columns, transposed in 32x32 cache blocks.
    cols.par_chunks_mut(h * w.min(32)).enumerate().for_each(|(band, out)| {
        let x0 = band * 32;
        for y0 in (0..h).step_by(32) {
            let y1 = y0.saturating_add(32).min(h);
            for (x, col) in out.chunks_mut(h).enumerate() {
                for (y, v) in col[y0..y1].iter_mut().enumerate() {
                    *v = rows[(y0 + y) * w + x0 + x];
                }
            }
        }
    });
    cols.par_chunks_mut(h).for_each_init(
        || (Vec::new(), Vec::new(), vec![0.0; h]),
        |(g, hb, out), col| {
            super::running_extreme(col, out, r.min(h - 1), grow, g, hb);
            col.copy_from_slice(out);
        },
    );
    rows.par_chunks_mut(w * h.min(32)).enumerate().for_each(|(band, out)| {
        let y0 = band * 32;
        let height = out.len() / w;
        for x0 in (0..w).step_by(32) {
            for x in x0..x0.saturating_add(32).min(w) {
                let col = &cols[x * h + y0..x * h + y0 + height];
                for (y, &v) in col.iter().enumerate() {
                    out[y * w + x] = v;
                }
            }
        }
    });
    rows
}

/// A radius-2d Manhattan ball is the union of five radius-d balls, centered
/// here and d pixels in each axial direction. Clamp the centers at the canvas
/// edge: every point in the clipped ball still belongs to one of those balls.
fn cross(src: &[f32], dst: &mut [f32], w: usize, h: usize, d: usize, grow: bool) {
    let pick = |a: f32, b: f32| if grow { a.max(b) } else { a.min(b) };
    let row = |y: usize, out: &mut [f32]| {
        let a = &src[y * w..(y + 1) * w];
        let up = y.saturating_sub(d) * w;
        let down = y.saturating_add(d).min(h - 1) * w;
        for (x, v) in out.iter_mut().enumerate() {
            *v = pick(pick(pick(pick(a[x], a[x.saturating_sub(d)]), a[x.saturating_add(d).min(w - 1)]), src[up + x]), src[down + x]);
        }
    };
    if dst.len() >= 262144 {
        crate::photo_util::par_rows(dst, w, 1, row);
    } else {
        dst.chunks_mut(w).enumerate().for_each(|(y, out)| row(y, out));
    }
}

fn repeated(src: &[f32], w: usize, h: usize, r: usize, grow: bool) -> Vec<f32> {
    let mut cur = src.to_vec();
    let mut nxt = cur.clone();
    let pick = |a: f32, b: f32| if grow { a.max(b) } else { a.min(b) };
    for k in 0..r {
        let square = k % 2 == 0;
        for y in 0..h {
            for x in 0..w {
                let mut v = cur[y * w + x];
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if (dx != 0 && dy != 0 && !square) || (dx == 0 && dy == 0) {
                            continue;
                        }
                        let (nx, ny) = (x as i32 + dx, y as i32 + dy);
                        if nx >= 0 && ny >= 0 && (nx as usize) < w && (ny as usize) < h {
                            v = pick(v, cur[ny as usize * w + nx as usize]);
                        }
                    }
                }
                nxt[y * w + x] = v;
            }
        }
        std::mem::swap(&mut cur, &mut nxt);
    }
    cur
}

#[cfg(test)]
mod tests;
