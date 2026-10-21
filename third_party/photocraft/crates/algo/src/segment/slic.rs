//! SLIC superpixels.
//!
//! Implemented from R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua and S. Süsstrunk, "SLIC
//! Superpixels Compared to State-of-the-Art Superpixel Methods", IEEE PAMI 34(11), 2012: k-means
//! in CIELAB + xy restricted to a 2S×2S window around each centre, distance
//! `D = sqrt(d_lab² + (d_xy / S)² m²)`, centres seeded on a grid and moved to the lowest gradient
//! in their 3×3 neighbourhood, then connectivity enforcement (small fragments join a neighbour).

use super::RgbImage;

/// sRGB (`0..=1`) to CIELAB (D65), L in `0..=100`.
pub fn rgb_to_lab(c: [f32; 3]) -> [f32; 3] {
    let lin = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    let (r, g, b) = (lin(c[0]), lin(c[1]), lin(c[2]));
    let x = (0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b) / 0.950_47;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175 * b;
    let z = (0.019_333_9 * r + 0.119_192 * g + 0.950_304_1 * b) / 1.088_83;
    let f = |t: f32| if t > 0.008_856 { t.cbrt() } else { 7.787 * t + 16.0 / 116.0 };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// A superpixel labelling.
#[derive(Clone, Debug)]
pub struct Superpixels {
    pub w: usize,
    pub h: usize,
    /// Label per pixel, `0..count`.
    pub labels: Vec<u32>,
    pub count: usize,
    /// Grid interval S used.
    pub step: f32,
}

/// Segments `img` into about `target` superpixels. `compactness` is SLIC's `m` (10 is typical
/// for CIELAB), `iters` the number of k-means iterations (10 in the paper).
pub fn slic(img: &RgbImage, target: usize, compactness: f32, iters: usize) -> Superpixels {
    let (w, h) = (img.w, img.h);
    let lab: Vec<[f32; 3]> = img.px.iter().map(|p| rgb_to_lab(*p)).collect();
    slic_lab(&lab, w, h, target, compactness, iters)
}

/// [`slic`] on a CIELAB image.
pub fn slic_lab(lab: &[[f32; 3]], w: usize, h: usize, target: usize, compactness: f32, iters: usize) -> Superpixels {
    let n = w * h;
    if n == 0 {
        return Superpixels { w, h, labels: Vec::new(), count: 0, step: 1.0 };
    }
    let s = ((n as f32 / target.max(1) as f32).sqrt()).max(1.0);
    let grad = |x: usize, y: usize| -> f32 {
        let at = |x: usize, y: usize| lab[y.min(h - 1) * w + x.min(w - 1)];
        let (l, r) = (at(x.saturating_sub(1), y), at(x + 1, y));
        let (u, d) = (at(x, y.saturating_sub(1)), at(x, y + 1));
        super::d2(l, r) + super::d2(u, d)
    };
    // Seed centres: [L, a, b, x, y].
    let mut centers: Vec<[f32; 5]> = Vec::new();
    let mut y = s / 2.0;
    while y < h as f32 {
        let mut x = s / 2.0;
        while x < w as f32 {
            let (cx, cy) = (x as usize, y as usize);
            let (mut bx, mut by, mut bg) = (cx, cy, f32::MAX);
            for yy in cy.saturating_sub(1)..=(cy + 1).min(h - 1) {
                for xx in cx.saturating_sub(1)..=(cx + 1).min(w - 1) {
                    let g = grad(xx, yy);
                    if g < bg {
                        (bx, by, bg) = (xx, yy, g);
                    }
                }
            }
            let c = lab[by * w + bx];
            centers.push([c[0], c[1], c[2], bx as f32, by as f32]);
            x += s;
        }
        y += s;
    }
    let mut labels = vec![u32::MAX; n];
    let mut dist = vec![f32::MAX; n];
    let wxy = (compactness / s).powi(2);
    let r = s.ceil() as i64;
    for _ in 0..iters.max(1) {
        dist.fill(f32::MAX);
        for (k, c) in centers.iter().enumerate() {
            let (cx, cy) = (c[3].round() as i64, c[4].round() as i64);
            for yy in (cy - r).max(0)..(cy + r + 1).min(h as i64) {
                for xx in (cx - r).max(0)..(cx + r + 1).min(w as i64) {
                    let i = yy as usize * w + xx as usize;
                    let p = lab[i];
                    let dc = super::d2(p, [c[0], c[1], c[2]]);
                    let (dx, dy) = (xx as f32 - c[3], yy as f32 - c[4]);
                    let d = dc + (dx * dx + dy * dy) * wxy;
                    if d < dist[i] {
                        dist[i] = d;
                        labels[i] = k as u32;
                    }
                }
            }
        }
        let mut sums = vec![[0.0f64; 6]; centers.len()];
        for (i, l) in labels.iter().enumerate() {
            if *l == u32::MAX {
                continue;
            }
            let p = lab[i];
            let s = &mut sums[*l as usize];
            s[0] += p[0] as f64;
            s[1] += p[1] as f64;
            s[2] += p[2] as f64;
            s[3] += (i % w) as f64;
            s[4] += (i / w) as f64;
            s[5] += 1.0;
        }
        for (c, s) in centers.iter_mut().zip(&sums) {
            if s[5] > 0.0 {
                *c = [0, 1, 2, 3, 4].map(|j| (s[j] / s[5]) as f32);
            }
        }
    }
    // Unreached pixels (possible at the far edges): nearest centre by label of a neighbour.
    for i in 0..n {
        if labels[i] == u32::MAX {
            labels[i] = if i > 0 { labels[i - 1] } else { 0 };
        }
    }
    let (labels, count) = enforce_connectivity(&labels, w, h, ((s * s) / 4.0) as usize);
    Superpixels { w, h, labels, count, step: s }
}

/// Relabels 4-connected components; components smaller than `min_size` merge into the adjacent
/// component found before them (as in the SLIC paper's post-processing).
fn enforce_connectivity(labels: &[u32], w: usize, h: usize, min_size: usize) -> (Vec<u32>, usize) {
    let n = w * h;
    let mut out = vec![u32::MAX; n];
    let mut next = 0u32;
    let mut comp = Vec::new();
    for s in 0..n {
        if out[s] != u32::MAX {
            continue;
        }
        // Adjacent already-labelled component (to merge into if this one is small).
        let (sx, sy) = (s % w, s / w);
        let mut adj = None;
        for j in [(sx > 0).then(|| s - 1), (sy > 0).then(|| s - w)].into_iter().flatten() {
            if out[j] != u32::MAX {
                adj = Some(out[j]);
            }
        }
        comp.clear();
        comp.push(s);
        out[s] = next;
        let mut k = 0;
        while k < comp.len() {
            let i = comp[k];
            k += 1;
            let (x, y) = (i % w, i / w);
            for j in [(x > 0).then(|| i - 1), (x + 1 < w).then(|| i + 1), (y > 0).then(|| i - w), (y + 1 < h).then(|| i + w)].into_iter().flatten() {
                if out[j] == u32::MAX && labels[j] == labels[s] {
                    out[j] = next;
                    comp.push(j);
                }
            }
        }
        match adj {
            Some(a) if comp.len() < min_size => {
                for &i in &comp {
                    out[i] = a;
                }
            }
            _ => next += 1,
        }
    }
    (out, next as usize)
}
