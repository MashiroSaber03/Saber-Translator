//! Content-aware scaling by seam carving (Edit › Content-Aware Scale).
//!
//! S. Avidan, A. Shamir, *Seam Carving for Content-Aware Image Resizing*, SIGGRAPH 2007:
//! a seam is an 8-connected path of one pixel per row (vertical seam) found by dynamic
//! programming over a gradient-magnitude energy map. Shrinking removes the lowest-energy seams one
//! at a time; enlarging finds the `k` lowest seams on a shrinking copy and duplicates them
//! (averaging each with its right neighbour) so the same seam is not stretched repeatedly.
//! Heights are handled by transposing.
//!
//! A protect map (`0..=1`, e.g. an alpha channel or detected skin tones) adds a large energy so
//! seams avoid protected pixels. Buffers are interleaved `w × h × ch` normalised floats of any
//! colour model and depth (alpha is carried like any other channel). Deterministic.

use photocraft_raster::{Cancelled, Interrupt};

/// Energy added per unit of protection: far above any gradient magnitude.
const PROTECT_ENERGY: f32 = 1.0e4;

/// Keep full horizontal detail while reducing the height used to find vertical seams. At a
/// 24 MP working size this cuts the dynamic-programming work by about four times without
/// quantising the output pixels or changing the requested output dimensions.
const SEAM_SEARCH_MAX_HEIGHT: usize = 512;
const PROXY_SHRINK_MIN_HEIGHT: usize = 2_048;
const PROXY_SHRINK_MIN_SEAMS: usize = 64;

/// Gradient-magnitude energy (the paper's e1, `|∂x| + |∂y|` summed over channels). Forward and
/// backward differences are both counted so one-pixel lines (whose central difference is zero)
/// carry energy too.
pub fn energy(w: usize, h: usize, ch: usize, img: &[f32]) -> Vec<f32> {
    let mut e = vec![0.0f32; w * h];
    if w == 0 || h == 0 {
        return e;
    }
    let row = |(y, out): (usize, &mut [f32])| {
        for (x, v) in out.iter_mut().enumerate() {
            *v = energy_at(img, w, w, h, ch, x, y);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        e.par_chunks_mut(w).enumerate().for_each(row);
    }
    #[cfg(target_arch = "wasm32")]
    e.chunks_mut(w).enumerate().for_each(row);
    e
}

/// [`energy`] of one pixel of a `cw`-wide image whose rows are `stride` pixels apart. Shared by
/// the full map and the per-seam updates so both give bit-identical values.
#[inline]
fn energy_at(img: &[f32], stride: usize, cw: usize, h: usize, ch: usize, x: usize, y: usize) -> f32 {
    let (xl, xr) = (x.saturating_sub(1), (x + 1).min(cw - 1));
    let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
    let at = |xx: usize, yy: usize, c: usize| img[(yy * stride + xx) * ch + c];
    let mut s = 0.0;
    for c in 0..ch {
        let v = at(x, y, c);
        s += (at(xr, y, c) - v).abs() + (v - at(xl, y, c)).abs();
        s += (at(x, yd, c) - v).abs() + (v - at(x, yu, c)).abs();
    }
    s * 0.5
}

/// The lowest-energy vertical seam: one x per row.
pub fn find_seam(w: usize, h: usize, energy: &[f32]) -> Vec<usize> {
    if w == 0 || h == 0 || energy.len() < w * h {
        return vec![0; h];
    }
    Dp::default().seam(w, w, h, energy)
}

/// Scratch buffers for the seam search, reused across seams.
#[derive(Default)]
struct Dp {
    cost: Vec<f32>,
    next: Vec<f32>,
    /// Per pixel, where the cheapest path came from: 0 = up-left, 1 = up, 2 = up-right.
    from: Vec<u8>,
}

impl Dp {
    /// The lowest-energy seam of a `cw`-wide, `h`-tall energy map with rows `stride` apart.
    /// Ties go to the pixel straight up, then up-left, then up-right; at the bottom, to the
    /// leftmost end.
    fn seam(&mut self, stride: usize, cw: usize, h: usize, energy: &[f32]) -> Vec<usize> {
        self.cost.clear();
        self.cost.extend_from_slice(&energy[..cw]);
        self.next.resize(cw, 0.0);
        self.from.resize(cw * h, 0);
        for y in 1..h {
            let (cost, next) = (&self.cost[..cw], &mut self.next[..cw]);
            let e = &energy[y * stride..y * stride + cw];
            let from = &mut self.from[y * cw..(y + 1) * cw];
            if cw == 1 {
                (next[0], from[0]) = (cost[0] + e[0], 1);
            } else {
                let (b, d) = if cost[1] < cost[0] { (cost[1], 2) } else { (cost[0], 1) };
                (next[0], from[0]) = (b + e[0], d);
                // The interior has all three parents: no edge tests in the hot loop.
                let inner = next[1..cw - 1].iter_mut().zip(&mut from[1..cw - 1]).zip(&e[1..cw - 1]).zip(cost.windows(3));
                for (((n, f), &ev), c) in inner {
                    let (b, d) = if c[0] < c[1] { (c[0], 0) } else { (c[1], 1) };
                    let (b, d) = if c[2] < b { (c[2], 2) } else { (b, d) };
                    (*n, *f) = (b + ev, d);
                }
                let (b, d) = if cost[cw - 2] < cost[cw - 1] { (cost[cw - 2], 0) } else { (cost[cw - 1], 1) };
                (next[cw - 1], from[cw - 1]) = (b + e[cw - 1], d);
            }
            std::mem::swap(&mut self.cost, &mut self.next);
        }
        let cost = &self.cost[..cw];
        let mut x = (0..cw).min_by(|a, b| cost[*a].total_cmp(&cost[*b])).unwrap_or(0);
        let mut seam = vec![0usize; h];
        for y in (0..h).rev() {
            seam[y] = x;
            if y > 0 {
                x = match self.from[y * cw + x] {
                    0 => x - 1,
                    2 => x + 1,
                    _ => x,
                };
            }
        }
        seam
    }
}

/// Shrinks an image one seam at a time in place. Rows keep their original stride, so removing a
/// seam shifts only the rest of each row (in parallel), and the energy map is updated only next
/// to the removed seam instead of being recomputed: the gradient of a pixel changes only when a
/// neighbour does, which happens within a pixel of the seam in its own row and the rows around it.
struct Carver {
    stride: usize,
    cw: usize,
    h: usize,
    ch: usize,
    img: Vec<f32>,
    prot: Option<Vec<f32>>,
    /// Energy plus protection, as the seam search sees it.
    e: Vec<f32>,
    /// Original column of each pixel, when the caller needs to know which ones were removed.
    idx: Option<Vec<usize>>,
    dp: Dp,
}

impl Carver {
    fn new(w: usize, h: usize, ch: usize, img: Vec<f32>, prot: Option<Vec<f32>>, track: bool) -> Self {
        let mut e = energy(w, h, ch, &img);
        if let Some(p) = &prot {
            for (e, p) in e.iter_mut().zip(p) {
                *e += p.clamp(0.0, 1.0) * PROTECT_ENERGY;
            }
        }
        let idx = track.then(|| (0..h).flat_map(|_| 0..w).collect());
        Carver { stride: w, cw: w, h, ch, img, prot, e, idx, dp: Dp::default() }
    }

    /// Remove the lowest-energy seam. Returns, per row, the removed column (the original column
    /// when tracking).
    fn remove_seam(&mut self) -> Vec<usize> {
        let (stride, cw, h, ch) = (self.stride, self.cw, self.h, self.ch);
        let seam = self.dp.seam(stride, cw, h, &self.e);
        let removed = match &self.idx {
            Some(idx) => seam.iter().enumerate().map(|(y, &x)| idx[y * stride + x]).collect(),
            None => seam.clone(),
        };
        shift_rows(&mut self.img, stride * ch, &seam, ch, cw);
        shift_rows(&mut self.e, stride, &seam, 1, cw);
        if let Some(p) = &mut self.prot {
            shift_rows(p, stride, &seam, 1, cw);
        }
        if let Some(idx) = &mut self.idx {
            shift_rows(idx, stride, &seam, 1, cw);
        }
        self.cw -= 1;
        self.update_energy(&seam);
        removed
    }

    /// Recompute the energy of the pixels whose neighbours changed when `seam` was removed: in
    /// row y, from one left of the leftmost of the seam's x in rows y-1..=y+1 to the rightmost.
    fn update_energy(&mut self, seam: &[usize]) {
        let (stride, cw, h, ch) = (self.stride, self.cw, self.h, self.ch);
        if cw == 0 {
            return;
        }
        let (img, prot) = (&self.img, self.prot.as_deref());
        let row = |(y, e): (usize, &mut [f32])| {
            let near = &seam[y.saturating_sub(1)..(y + 2).min(h)];
            let lo = near.iter().min().map_or(0, |&s| s.saturating_sub(1));
            let hi = near.iter().max().map_or(0, |&s| s.min(cw - 1));
            for x in lo..=hi {
                let mut v = energy_at(img, stride, cw, h, ch, x, y);
                if let Some(p) = prot {
                    v += p[y * stride + x].clamp(0.0, 1.0) * PROTECT_ENERGY;
                }
                e[x] = v;
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            self.e.par_chunks_mut(stride).enumerate().for_each(row);
        }
        #[cfg(target_arch = "wasm32")]
        self.e.chunks_mut(stride).enumerate().for_each(row);
    }

    /// The carved pixels, `cw × h × ch`, rows packed.
    fn into_pixels(self) -> Vec<f32> {
        let (row, used) = (self.stride * self.ch, self.cw * self.ch);
        if row == used {
            return self.img;
        }
        self.img.chunks(row).flat_map(|r| &r[..used]).copied().collect()
    }
}

/// Remove pixel `seam[y]` (of `n` values) from each `row_len`-long row whose first `cw` pixels
/// are in use, by shifting the rest of the row left.
fn shift_rows<T: Copy + Send>(buf: &mut [T], row_len: usize, seam: &[usize], n: usize, cw: usize) {
    let shift = |(r, &x): (&mut [T], &usize)| r.copy_within((x + 1) * n..cw * n, x * n);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        buf.par_chunks_mut(row_len).zip(seam.par_iter()).for_each(shift);
    }
    #[cfg(target_arch = "wasm32")]
    buf.chunks_mut(row_len).zip(seam).for_each(shift);
}

fn transpose(w: usize, h: usize, ch: usize, img: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; img.len()];
    for y in 0..h {
        for x in 0..w {
            let (s, d) = ((y * w + x) * ch, (x * h + y) * ch);
            out[d..d + ch].copy_from_slice(&img[s..s + ch]);
        }
    }
    out
}

/// Change the width of `img` to `new_w` by removing or duplicating vertical seams.
pub fn carve_width(w: usize, h: usize, ch: usize, img: &[f32], protect: Option<&[f32]>, new_w: usize) -> Vec<f32> {
    // Never cancelled; the fallback is unreachable.
    carve_width_with(w, h, ch, img, protect, new_w, &Interrupt::NONE).unwrap_or_else(|_| img.to_vec())
}

/// [`carve_width`] that checks `ctl` before every seam and reports progress per seam.
pub fn carve_width_with(w: usize, h: usize, ch: usize, img: &[f32], protect: Option<&[f32]>, new_w: usize, ctl: &Interrupt) -> Result<Vec<f32>, Cancelled> {
    let new_w = new_w.max(1);
    if new_w == w || w == 0 || h == 0 {
        return Ok(img.to_vec());
    }
    let seams = new_w.abs_diff(w).max(1) as f32;
    if new_w < w {
        if h >= PROXY_SHRINK_MIN_HEIGHT && w - new_w >= PROXY_SHRINK_MIN_SEAMS {
            return carve_width_proxy_height(w, h, ch, img, protect, new_w, ctl);
        }
        let mut carver = Carver::new(w, h, ch, img.to_vec(), protect.map(<[f32]>::to_vec), false);
        while carver.cw > new_w {
            ctl.check()?;
            ctl.progress((w - carver.cw) as f32 / seams);
            carver.remove_seam();
        }
        return Ok(carver.into_pixels());
    }
    // Enlarge in rounds of at most half the current width, each duplicating the k lowest seams.
    let (mut cur, mut cw) = (img.to_vec(), w);
    let mut prot = protect.map(<[f32]>::to_vec);
    while cw < new_w {
        let k = (new_w - cw).min((cw / 2).max(1));
        // Find k seams on a shrinking copy, mapping each back to original columns.
        let mut carver = Carver::new(cw, h, ch, cur.clone(), prot.clone(), true);
        let mut dup: Vec<Vec<usize>> = vec![Vec::with_capacity(k); h];
        for _ in 0..k {
            // A 1 px wide image's only seam is its one column: take it, so it is duplicated.
            if carver.cw == 0 {
                break;
            }
            ctl.check()?;
            ctl.progress((cw + (cw - carver.cw)).saturating_sub(w) as f32 / seams);
            for (d, x) in dup.iter_mut().zip(carver.remove_seam()) {
                d.push(x);
            }
        }
        let added = dup[0].len();
        if added == 0 {
            break;
        }
        let nw = cw + added;
        let mut out = Vec::with_capacity(nw * h * ch);
        let mut nprot = prot.as_ref().map(|_| Vec::with_capacity(nw * h));
        for (y, d) in dup.iter_mut().enumerate() {
            d.sort_unstable();
            let row = &cur[y * cw * ch..(y + 1) * cw * ch];
            let mut di = 0;
            for x in 0..cw {
                out.extend_from_slice(&row[x * ch..(x + 1) * ch]);
                if let (Some(np), Some(p)) = (nprot.as_mut(), prot.as_ref()) {
                    np.push(p[y * cw + x]);
                }
                while di < d.len() && d[di] == x {
                    let xr = (x + 1).min(cw - 1);
                    for c in 0..ch {
                        out.push((row[x * ch + c] + row[xr * ch + c]) * 0.5);
                    }
                    if let (Some(np), Some(p)) = (nprot.as_mut(), prot.as_ref()) {
                        np.push(p[y * cw + x]);
                    }
                    di += 1;
                }
            }
        }
        cur = out;
        prot = nprot;
        cw = nw;
    }
    Ok(cur)
}

/// Find large-image seams on vertically averaged rows, then apply those paths to the original
/// pixels. The proxy has the same width, so each seam still removes exactly one source pixel per
/// row. Keeping output pixels at full resolution avoids the quality loss of resizing the result;
/// only the seam decisions use the coarser vertical signal. Small operations retain the exact
/// full-resolution algorithm above.
fn carve_width_proxy_height(w: usize, h: usize, ch: usize, img: &[f32], protect: Option<&[f32]>, new_w: usize, ctl: &Interrupt) -> Result<Vec<f32>, Cancelled> {
    let step = h.div_ceil(SEAM_SEARCH_MAX_HEIGHT);
    let proxy_h = h.div_ceil(step);
    let row_len = w * ch;
    let proxy_row_len = proxy_h * row_len;
    let mut proxy = vec![0.0; proxy_row_len];
    for (py, out_row) in proxy.chunks_exact_mut(row_len).enumerate() {
        let y0 = py * step;
        let y1 = (y0 + step).min(h);
        let divisor = (y1 - y0) as f32;
        for y in y0..y1 {
            let in_row = &img[y * row_len..(y + 1) * row_len];
            for (out, &sample) in out_row.iter_mut().zip(in_row) {
                *out += sample / divisor;
            }
        }
    }
    let proxy_protect = protect.map(|mask| {
        let mut reduced = vec![0.0; proxy_h * w];
        for (py, out_row) in reduced.chunks_exact_mut(w).enumerate() {
            let y0 = py * step;
            let y1 = (y0 + step).min(h);
            let divisor = (y1 - y0) as f32;
            for y in y0..y1 {
                let in_row = &mask[y * w..(y + 1) * w];
                for (out, &sample) in out_row.iter_mut().zip(in_row) {
                    *out += sample / divisor;
                }
            }
        }
        reduced
    });

    let seam_count = w - new_w;
    let mut carver = Carver::new(w, proxy_h, ch, proxy, proxy_protect, true);
    let mut removed = Vec::with_capacity(seam_count);
    while carver.cw > new_w {
        ctl.check()?;
        removed.push(carver.remove_seam());
        ctl.progress(removed.len() as f32 / seam_count as f32);
    }

    let mut removed_by_row = Vec::with_capacity(proxy_h);
    for proxy_y in 0..proxy_h {
        let mut columns: Vec<usize> = removed.iter().map(|path| path[proxy_y]).collect();
        columns.sort_unstable();
        removed_by_row.push(columns);
    }
    let mut out = Vec::with_capacity(new_w * h * ch);
    let mut columns = vec![0; seam_count];
    for y in 0..h {
        ctl.check()?;
        let proxy_y = (y / step).min(proxy_h - 1);
        let next_proxy_y = (proxy_y + 1).min(proxy_h - 1);
        let blend = (y % step) as f32 / step as f32;
        let lower = &removed_by_row[proxy_y];
        let upper = &removed_by_row[next_proxy_y];
        let mut min_x = 0;
        for (i, x) in columns.iter_mut().enumerate() {
            let interpolated = lower[i] as f32 + (upper[i] as f32 - lower[i] as f32) * blend;
            let max_x = w - (seam_count - i);
            *x = (interpolated.round() as usize).clamp(min_x, max_x);
            min_x = x.saturating_add(1);
        }
        let mut next = 0;
        let row = &img[y * row_len..(y + 1) * row_len];
        for (x, pixel) in row.chunks_exact(ch).enumerate() {
            if columns.get(next) == Some(&x) {
                next += 1;
            } else {
                out.extend_from_slice(pixel);
            }
        }
    }
    ctl.progress(1.0);
    Ok(out)
}

/// Content-aware resize of `img` to `new_w × new_h` (width first, then height by transposing).
pub fn carve(w: usize, h: usize, ch: usize, img: &[f32], protect: Option<&[f32]>, new_w: usize, new_h: usize) -> Vec<f32> {
    // Never cancelled; the fallback is unreachable.
    carve_with(w, h, ch, img, protect, new_w, new_h, &Interrupt::NONE).unwrap_or_else(|_| img.to_vec())
}

/// [`carve`] that checks `ctl` before every seam (a cancel takes effect within one seam) and
/// reports progress over both passes.
#[allow(clippy::too_many_arguments)]
pub fn carve_with(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    protect: Option<&[f32]>,
    new_w: usize,
    new_h: usize,
    ctl: &Interrupt,
) -> Result<Vec<f32>, Cancelled> {
    let (new_w, new_h) = (new_w.max(1), new_h.max(1));
    // Split progress between the passes by their seam counts.
    let (sw, sh) = (new_w.abs_diff(w) as f32, new_h.abs_diff(h) as f32);
    let split = if sw + sh > 0.0 { sw / (sw + sh) } else { 1.0 };
    let cancel = || ctl.cancelled();
    let first = |f: f32| ctl.progress(f * split);
    let wide = carve_width_with(w, h, ch, img, protect, new_w, &Interrupt::new(&cancel, &first))?;
    if new_h == h {
        return Ok(wide);
    }
    // Protection follows the width change by plain resampling (it only steers seams).
    let prot_t = protect.map(|p| transpose(new_w, h, 1, &resize_bilinear(w, h, 1, p, new_w, h)));
    let t = transpose(new_w, h, ch, &wide);
    let second = |f: f32| ctl.progress(split + f * (1.0 - split));
    let tall = carve_width_with(h, new_w, ch, &t, prot_t.as_deref(), new_h, &Interrupt::new(&cancel, &second))?;
    Ok(transpose(new_h, new_w, ch, &tall))
}

/// Plain bilinear resize (the non-content-aware part of the Amount blend).
pub fn resize_bilinear(w: usize, h: usize, ch: usize, img: &[f32], nw: usize, nh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; nw * nh * ch];
    if w == 0 || h == 0 {
        return out;
    }
    for y in 0..nh {
        let fy = ((y as f32 + 0.5) * h as f32 / nh as f32 - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..nw {
            let fx = ((x as f32 + 0.5) * w as f32 / nw as f32 - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            for c in 0..ch {
                let p = |xx: usize, yy: usize| img[(yy * w + xx) * ch + c];
                let top = p(x0, y0) + (p(x1, y0) - p(x0, y0)) * tx;
                let bot = p(x0, y1) + (p(x1, y1) - p(x0, y1)) * tx;
                out[(y * nw + x) * ch + c] = top + (bot - top) * ty;
            }
        }
    }
    out
}

/// Skin-tone protection map (1 = skin) for RGB(A) buffers: the explicit RGB rule of
/// J. Kovač, P. Peer, F. Solina, *Human Skin Colour Clustering for Face Detection*, EUROCON 2003.
pub fn skin_mask(w: usize, h: usize, ch: usize, img: &[f32]) -> Vec<f32> {
    let mut m = vec![0.0f32; w * h];
    if ch < 3 {
        return m;
    }
    for (i, v) in m.iter_mut().enumerate() {
        let p = &img[i * ch..i * ch + 3];
        let (r, g, b) = (p[0] * 255.0, p[1] * 255.0, p[2] * 255.0);
        let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
        if r > 95.0 && g > 40.0 && b > 20.0 && mx - mn > 15.0 && (r - g).abs() > 15.0 && r > g && r > b {
            *v = 1.0;
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The straightforward implementation the per-seam carver replaced (full energy map, fresh
    /// seam search and a new buffer per seam), kept as the oracle for `carver_matches_reference`.
    mod reference {
        use super::super::{PROTECT_ENERGY, resize_bilinear, transpose};

        pub(super) fn energy(w: usize, h: usize, ch: usize, img: &[f32]) -> Vec<f32> {
            let mut e = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (xl, xr) = (x.saturating_sub(1), (x + 1).min(w - 1));
                    let (yu, yd) = (y.saturating_sub(1), (y + 1).min(h - 1));
                    let at = |xx: usize, yy: usize, c: usize| img[(yy * w + xx) * ch + c];
                    let mut s = 0.0;
                    for c in 0..ch {
                        let v = at(x, y, c);
                        s += (at(xr, y, c) - v).abs() + (v - at(xl, y, c)).abs();
                        s += (at(x, yd, c) - v).abs() + (v - at(x, yu, c)).abs();
                    }
                    e[y * w + x] = s * 0.5;
                }
            }
            e
        }
        pub(super) fn find_seam(w: usize, h: usize, energy: &[f32]) -> Vec<usize> {
            let mut cost = energy[..w].to_vec();
            let mut from = vec![0u8; w * h]; // 0 = up-left, 1 = up, 2 = up-right
            let mut next = vec![0.0f32; w];
            for y in 1..h {
                for x in 0..w {
                    let mut best = (cost[x], 1u8);
                    if x > 0 && cost[x - 1] < best.0 {
                        best = (cost[x - 1], 0);
                    }
                    if x + 1 < w && cost[x + 1] < best.0 {
                        best = (cost[x + 1], 2);
                    }
                    next[x] = best.0 + energy[y * w + x];
                    from[y * w + x] = best.1;
                }
                std::mem::swap(&mut cost, &mut next);
            }
            let mut x = (0..w).min_by(|a, b| cost[*a].total_cmp(&cost[*b])).unwrap_or(0);
            let mut seam = vec![0usize; h];
            for y in (0..h).rev() {
                seam[y] = x;
                if y > 0 {
                    x = match from[y * w + x] {
                        0 => x - 1,
                        2 => x + 1,
                        _ => x,
                    };
                }
            }
            seam
        }
        fn remove_seam(w: usize, h: usize, ch: usize, img: &[f32], seam: &[usize]) -> Vec<f32> {
            let mut out = Vec::with_capacity((w - 1) * h * ch);
            for (y, &sx) in seam.iter().enumerate().take(h) {
                let row = &img[y * w * ch..(y + 1) * w * ch];
                out.extend_from_slice(&row[..sx * ch]);
                out.extend_from_slice(&row[(sx + 1) * ch..]);
            }
            out
        }
        pub(super) fn carve_width(w: usize, h: usize, ch: usize, img: &[f32], protect: Option<&[f32]>, new_w: usize) -> Vec<f32> {
            let new_w = new_w.max(1);
            if new_w == w || w == 0 || h == 0 {
                return img.to_vec();
            }
            let protect_e = |e: &mut [f32], p: Option<&Vec<f32>>| {
                if let Some(p) = p {
                    for (e, p) in e.iter_mut().zip(p) {
                        *e += p.clamp(0.0, 1.0) * PROTECT_ENERGY;
                    }
                }
            };
            if new_w < w {
                let (mut cur, mut cw) = (img.to_vec(), w);
                let mut prot = protect.map(<[f32]>::to_vec);
                while cw > new_w {
                    let mut e = energy(cw, h, ch, &cur);
                    protect_e(&mut e, prot.as_ref());
                    let seam = find_seam(cw, h, &e);
                    cur = remove_seam(cw, h, ch, &cur, &seam);
                    prot = prot.map(|p| remove_seam(cw, h, 1, &p, &seam));
                    cw -= 1;
                }
                return cur;
            }
            // Enlarge in rounds of at most half the current width, each duplicating the k lowest seams.
            let (mut cur, mut cw) = (img.to_vec(), w);
            let mut prot = protect.map(<[f32]>::to_vec);
            while cw < new_w {
                let k = (new_w - cw).min((cw / 2).max(1));
                // Find k seams on a shrinking copy, mapping each back to original columns.
                let mut idx: Vec<Vec<usize>> = (0..h).map(|_| (0..cw).collect()).collect();
                let (mut tmp, mut tw) = (cur.clone(), cw);
                let mut tprot = prot.clone();
                let mut dup: Vec<Vec<usize>> = vec![Vec::with_capacity(k); h];
                for _ in 0..k {
                    // A 1 px wide image's only seam is its one column: take it, so it is duplicated.
                    if tw == 0 {
                        break;
                    }
                    let mut e = energy(tw, h, ch, &tmp);
                    protect_e(&mut e, tprot.as_ref());
                    let seam = find_seam(tw, h, &e);
                    for (y, &sx) in seam.iter().enumerate() {
                        dup[y].push(idx[y].remove(sx));
                    }
                    tmp = remove_seam(tw, h, ch, &tmp, &seam);
                    tprot = tprot.map(|p| remove_seam(tw, h, 1, &p, &seam));
                    tw -= 1;
                }
                let added = dup[0].len();
                if added == 0 {
                    break;
                }
                let nw = cw + added;
                let mut out = Vec::with_capacity(nw * h * ch);
                let mut nprot = prot.as_ref().map(|_| Vec::with_capacity(nw * h));
                for (y, d) in dup.iter_mut().enumerate() {
                    d.sort_unstable();
                    let row = &cur[y * cw * ch..(y + 1) * cw * ch];
                    let mut di = 0;
                    for x in 0..cw {
                        out.extend_from_slice(&row[x * ch..(x + 1) * ch]);
                        if let (Some(np), Some(p)) = (nprot.as_mut(), prot.as_ref()) {
                            np.push(p[y * cw + x]);
                        }
                        while di < d.len() && d[di] == x {
                            let xr = (x + 1).min(cw - 1);
                            for c in 0..ch {
                                out.push((row[x * ch + c] + row[xr * ch + c]) * 0.5);
                            }
                            if let (Some(np), Some(p)) = (nprot.as_mut(), prot.as_ref()) {
                                np.push(p[y * cw + x]);
                            }
                            di += 1;
                        }
                    }
                }
                cur = out;
                prot = nprot;
                cw = nw;
            }
            cur
        }

        pub(super) fn carve(w: usize, h: usize, ch: usize, img: &[f32], protect: Option<&[f32]>, new_w: usize, new_h: usize) -> Vec<f32> {
            let (new_w, new_h) = (new_w.max(1), new_h.max(1));
            let wide = carve_width(w, h, ch, img, protect, new_w);
            if new_h == h {
                return wide;
            }
            let prot_t = protect.map(|p| transpose(new_w, h, 1, &resize_bilinear(w, h, 1, p, new_w, h)));
            let t = transpose(new_w, h, ch, &wide);
            let tall = carve_width(h, new_w, ch, &t, prot_t.as_deref(), new_h);
            transpose(new_h, new_w, ch, &tall)
        }
    }

    /// The carver gives bit-identical output to the reference implementation it replaced, over
    /// random shapes (including 1 px wide and tall), channel counts, tie-heavy quantised noise,
    /// protection, shrinking, enlarging past half the width and both axes at once.
    #[test]
    fn carver_matches_reference() {
        let mut seed = 1u32;
        let mut next = |n: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as usize % n
        };
        for case in 0..400u32 {
            let (w, h, ch) = (1 + next(24), 1 + next(18), [1, 3, 4][next(3)]);
            let levels = [2, 5, 255, 65_536][next(4)];
            let img = noise(w * h * ch, levels, 11 + case);
            let protect = (next(3) == 0).then(|| noise(w * h, 3, 101 + case));
            let (nw, nh) = (1 + next(2 * w + 2), 1 + next(2 * h + 2));
            let got = carve(w, h, ch, &img, protect.as_deref(), nw, nh);
            let want = reference::carve(w, h, ch, &img, protect.as_deref(), nw, nh);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "case {case}: {w}x{h}x{ch} -> {nw}x{nh}, levels {levels}, protect {}", protect.is_some());
        }
        // The seam search alone, on raw energy maps full of ties.
        for case in 0..200u32 {
            let (w, h) = (1 + next(40), 1 + next(30));
            let e = noise(w * h, [2, 3, 1000][next(3)], 7 + case);
            assert_eq!(find_seam(w, h, &e), reference::find_seam(w, h, &e), "seam case {case}: {w}x{h}");
            let img = noise(w * h * 3, 7, 3 + case);
            let bits = |v: Vec<f32>| v.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            assert_eq!(bits(energy(w, h, 3, &img)), bits(reference::energy(w, h, 3, &img)), "energy case {case}");
        }
    }

    #[test]
    fn cancelled_carve_stops() {
        let (w, h) = (40, 20);
        let img = vec![0.5f32; w * h * 3];
        let yes = || true;
        assert_eq!(carve_with(w, h, 3, &img, None, 30, 15, &Interrupt::cancel_only(&yes)), Err(Cancelled));
        assert_eq!(carve_with(w, h, 3, &img, None, 30, 15, &Interrupt::NONE).unwrap(), carve(w, h, 3, &img, None, 30, 15));
    }

    /// A flat grey image with a salient vertical bar at x = 10..14 (red/blue columns).
    fn bar(w: usize, h: usize) -> Vec<f32> {
        let mut v = vec![0.5f32; w * h * 3];
        for y in 0..h {
            for x in 10..14 {
                let c = if x % 2 == 0 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] };
                v[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&c);
            }
        }
        v
    }

    /// Columns that are pure red or pure blue top to bottom.
    fn red_columns(w: usize, h: usize, img: &[f32]) -> usize {
        let pure = |p: &[f32]| p[1] < 0.05 && ((p[0] > 0.95 && p[2] < 0.05) || (p[2] > 0.95 && p[0] < 0.05));
        (0..w).filter(|x| (0..h).all(|y| pure(&img[(y * w + x) * 3..(y * w + x) * 3 + 3]))).count()
    }

    #[test]
    fn seam_follows_low_energy() {
        let (w, h) = (5, 4);
        let mut e = vec![9.0f32; w * h];
        for y in 0..h {
            e[y * w + 2] = 0.0;
        }
        assert_eq!(find_seam(w, h, &e), vec![2; 4]);
    }

    #[test]
    fn shrinking_keeps_the_salient_bar() {
        let (w, h) = (40, 12);
        let img = bar(w, h);
        let out = carve(w, h, 3, &img, None, 24, h);
        assert_eq!(out.len(), 24 * h * 3);
        assert_eq!(red_columns(24, h, &out), 4, "the bar survives intact");
        // Plain resampling squashes it instead.
        let plain = resize_bilinear(w, h, 3, &img, 24, h);
        assert!(red_columns(24, h, &plain) < 4);
    }

    #[test]
    fn tall_proxy_shrink_preserves_full_resolution_salient_and_protected_pixels() {
        let (w, h, ch, new_w) = (512, PROXY_SHRINK_MIN_HEIGHT, 3, 384);
        let mut img = bar(w, h);
        let mut protect = vec![0.0; w * h];
        for y in 0..h {
            let i = (y * w + 256) * ch;
            img[i..i + ch].copy_from_slice(&[0.03, 0.87, 0.42]);
            protect[y * w + 256] = 1.0;
        }

        let out = carve_width_with(w, h, ch, &img, Some(&protect), new_w, &Interrupt::NONE).unwrap();

        assert_eq!(out.len(), new_w * h * ch);
        assert_eq!(red_columns(new_w, h, &out), 4, "the full-resolution bar survives");
        for y in 0..h {
            let row = &out[y * new_w * ch..(y + 1) * new_w * ch];
            assert!(row.chunks_exact(ch).any(|px| px == [0.03, 0.87, 0.42]), "protected mark survives row {y}");
        }
    }

    #[test]
    fn tall_proxy_shrink_observes_cancellation_before_modifying_source() {
        let (w, h, ch) = (128, PROXY_SHRINK_MIN_HEIGHT, 3);
        let img = vec![0.5; w * h * ch];
        let cancelled = || true;
        assert_eq!(carve_width_with(w, h, ch, &img, None, 64, &Interrupt::cancel_only(&cancelled)), Err(Cancelled));
    }

    #[test]
    fn enlarging_duplicates_flat_areas() {
        let (w, h) = (30, 8);
        let img = bar(w, h);
        let out = carve(w, h, 3, &img, None, 50, h);
        assert_eq!(out.len(), 50 * h * 3);
        assert_eq!(red_columns(50, h, &out), 4);
    }

    #[test]
    fn height_and_protection() {
        let (w, h) = (20, 30);
        // Horizontal bar: carve height.
        let mut img = vec![0.2f32; w * h];
        for x in 0..w {
            img[12 * w + x] = 1.0;
        }
        let out = carve(w, h, 1, &img, None, w, 18);
        assert_eq!(out.len(), w * 18);
        assert_eq!((0..18).filter(|y| (0..w).all(|x| out[y * w + x] > 0.9)).count(), 1);
        // Protecting a flat region keeps it: the seams go elsewhere.
        let flat = vec![0.5f32; 16 * 4];
        let mut protect = vec![0.0f32; 16 * 4];
        for y in 0..4 {
            protect[y * 16 + 3] = 1.0;
        }
        let mut marked = flat.clone();
        for y in 0..4 {
            marked[y * 16 + 3] = 0.51;
        }
        let out = carve(16, 4, 1, &marked, Some(&protect), 8, 4);
        assert!((0..4).all(|y| out[y * 8..y * 8 + 8].iter().any(|v| (*v - 0.51).abs() < 1e-6)));
    }

    /// Noise quantised to `levels` values, so the energy map is full of exact ties.
    fn noise(len: usize, levels: u32, seed: u32) -> Vec<f32> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s % levels) as f32 / (levels - 1) as f32
            })
            .collect()
    }

    fn fnv(v: &[f32]) -> u64 {
        v.iter().flat_map(|f| f.to_bits().to_le_bytes()).fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
    }

    /// The exact output bits of a spread of carves (shrink, enlarge past half the width, both
    /// axes, protection, 1/3/4 channels, tie-heavy and smooth images), pinned so that speed work
    /// can't change which seams are chosen or how pixels are blended.
    #[test]
    fn output_bits_are_pinned() {
        let cases: [(usize, usize, usize, u32, bool, usize, usize); 8] = [
            (61, 47, 3, 2, false, 40, 47),
            (61, 47, 3, 255, false, 61, 30),
            (61, 47, 4, 4, true, 35, 29),
            (33, 20, 1, 3, false, 80, 20),
            (33, 20, 3, 1000, true, 50, 45),
            (48, 36, 4, 7, false, 70, 18),
            (5, 3, 1, 2, false, 1, 1),
            (40, 40, 3, 2, true, 39, 41),
        ];
        let got: Vec<u64> = cases
            .iter()
            .enumerate()
            .map(|(i, &(w, h, ch, levels, prot, nw, nh))| {
                let img = noise(w * h * ch, levels, 7 + i as u32);
                let protect = prot.then(|| noise(w * h, 3, 99 + i as u32));
                let out = carve(w, h, ch, &img, protect.as_deref(), nw, nh);
                assert_eq!(out.len(), nw * nh * ch);
                fnv(&out)
            })
            .collect();
        assert_eq!(got, PINNED);
    }

    const PINNED: [u64; 8] = [
        0x24ea_9353_02c2_7c58,
        0x5a6e_8554_a0cb_ebb1,
        0x8b18_d63f_396e_21c9,
        0x1db2_23ad_3e6a_a7c5,
        0x721d_6cc5_34e0_3fdf,
        0x961a_ea6d_e512_ec7d,
        0x4d25_767f_9dce_13f5,
        0x2157_9f86_27d6_b835,
    ];

    #[test]
    fn one_pixel_lines_enlarge_by_repeating() {
        // Width 1 has no seam to spare: its one column is repeated. Height 1 likewise (through
        // the transposed pass), alone or with the other axis.
        let col = [0.1f32, 0.2, 0.3, 0.4, 0.5];
        assert_eq!(carve(1, 5, 1, &col, None, 4, 5), col.iter().flat_map(|v| [*v; 4]).collect::<Vec<_>>());
        assert_eq!(carve(5, 1, 1, &col, None, 5, 3), col.repeat(3));
        assert_eq!(carve(1, 1, 3, &[0.7, 0.8, 0.9], None, 3, 2), [0.7, 0.8, 0.9].repeat(6));
        for (w, h, nw, nh) in [(1, 5, 6, 10), (1, 17, 6, 34), (16, 1, 21, 2), (7, 1, 3, 8), (2, 1, 1, 8)] {
            let img = vec![0.5f32; w * h * 4];
            assert_eq!(carve(w, h, 4, &img, Some(&vec![1.0; w * h]), nw, nh).len(), nw * nh * 4, "{w}x{h} -> {nw}x{nh}");
        }
    }

    #[test]
    fn skin_rule() {
        let px = [0.86f32, 0.62, 0.5, 0.2, 0.4, 0.8];
        assert_eq!(skin_mask(2, 1, 3, &px), vec![1.0, 0.0]);
    }
}
