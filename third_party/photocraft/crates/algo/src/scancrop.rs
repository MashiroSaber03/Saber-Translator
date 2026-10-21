//! File › Automate › Crop and Straighten Photos: find the photos on a flatbed scan, measure each
//! one's rotation, and cut them out upright.
//!
//! 1. **Background**: the median colour of the image border (a scanner lid is uniform).
//! 2. **Foreground**: pixels whose colour distance to the background exceeds a threshold chosen
//!    by Otsu's method (N. Otsu, *A Threshold Selection Method from Gray-Level Histograms*, IEEE
//!    SMC 1979) on a downsampled copy, then closed and opened morphologically (fills specks
//!    inside photos, drops dust) and split into 8-connected components; tiny ones are dropped.
//! 3. **Rotation**: the minimum-area enclosing rectangle of each component's convex hull
//!    (H. Freeman, R. Shapira, *Determining the Minimum-Area Encasing Rectangle for an
//!    Arbitrary Closed Curve*, CACM 1975: one side is collinear with a hull edge).
//! 4. The rectangle is shrunk by a pixel (to drop the anti-aliased border) and returned in full
//!    resolution; callers resample it upright.

/// A found photo: centre, size (long and short sides as measured along its own axes) and the
/// rotation (degrees, clockwise) that makes it upright.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FoundPhoto {
    pub center: [f64; 2],
    pub width: f64,
    pub height: f64,
    /// Angle of the photo's x axis in the scan (degrees, `-45..=45`).
    pub angle: f64,
    /// Area as a fraction of the scan.
    pub area: f64,
}

impl FoundPhoto {
    /// The four corners in the scan, clockwise from the photo's top-left.
    pub fn corners(&self) -> [[f64; 2]; 4] {
        let (s, c) = self.angle.to_radians().sin_cos();
        let (hw, hh) = (self.width / 2.0, self.height / 2.0);
        [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)].map(|(x, y)| [self.center[0] + c * x - s * y, self.center[1] + s * x + c * y])
    }
}

fn otsu(v: &[f32]) -> f32 {
    const B: usize = 256;
    let mut hist = [0f64; B];
    for x in v {
        hist[((x.clamp(0.0, 1.0)) * (B - 1) as f32) as usize] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let sum: f64 = hist.iter().enumerate().map(|(i, h)| i as f64 * h).sum();
    let (mut wb, mut sb, mut best, mut thr) = (0.0, 0.0, 0.0, 0usize);
    for (i, h) in hist.iter().enumerate() {
        wb += h;
        if wb == 0.0 {
            continue;
        }
        let wf = total - wb;
        if wf == 0.0 {
            break;
        }
        sb += i as f64 * h;
        let (mb, mf) = (sb / wb, (sum - sb) / wf);
        let between = wb * wf * (mb - mf) * (mb - mf);
        if between > best {
            best = between;
            thr = i;
        }
    }
    (thr as f32 + 0.5) / (B - 1) as f32
}

fn morph(w: usize, h: usize, m: &[bool], r: i64, dilate: bool) -> Vec<bool> {
    // Separable square structuring element.
    let pass = |src: &[bool], horizontal: bool| -> Vec<bool> {
        let mut out = vec![false; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut acc = !dilate;
                for d in -r..=r {
                    let (xx, yy) = if horizontal { (x as i64 + d, y as i64) } else { (x as i64, y as i64 + d) };
                    let v = if xx < 0 || yy < 0 || xx >= w as i64 || yy >= h as i64 { false } else { src[yy as usize * w + xx as usize] };
                    if dilate { acc |= v } else { acc &= v }
                }
                out[y * w + x] = acc;
            }
        }
        out
    };
    pass(&pass(m, true), false)
}

fn convex_hull(mut pts: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    pts.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    pts.dedup();
    if pts.len() < 3 {
        return pts;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], *p) <= 0.0 {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], *p) <= 0.0 {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Minimum-area rectangle around a convex hull: (centre, width, height, angle in degrees).
fn min_rect(hull: &[[f64; 2]]) -> ([f64; 2], f64, f64, f64) {
    let mut best = (f64::MAX, [0.0, 0.0], 0.0, 0.0, 0.0);
    let n = hull.len();
    for i in 0..n {
        let (a, b) = (hull[i], hull[(i + 1) % n]);
        let th = (b[1] - a[1]).atan2(b[0] - a[0]);
        let (s, c) = th.sin_cos();
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for p in hull {
            let (u, v) = (c * p[0] + s * p[1], -s * p[0] + c * p[1]);
            x0 = x0.min(u);
            x1 = x1.max(u);
            y0 = y0.min(v);
            y1 = y1.max(v);
        }
        let area = (x1 - x0) * (y1 - y0);
        if area < best.0 {
            let (cu, cv) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
            best = (area, [c * cu - s * cv, s * cu + c * cv], x1 - x0, y1 - y0, th.to_degrees());
        }
    }
    let (_, center, mut w, mut h, mut ang) = best;
    // Normalise to −45..45, swapping the sides when turning by 90°.
    while ang > 45.0 {
        ang -= 90.0;
        std::mem::swap(&mut w, &mut h);
    }
    while ang < -45.0 {
        ang += 90.0;
        std::mem::swap(&mut w, &mut h);
    }
    (center, w, h, ang)
}

/// Finds the photos on a scan given as straight RGBA rows (`w × h`). Photos are returned in
/// reading order (top to bottom, then left to right).
pub fn find_photos(w: usize, h: usize, rgba: &[[f32; 4]]) -> Vec<FoundPhoto> {
    if w < 8 || h < 8 {
        return Vec::new();
    }
    // Work on ≤ 800 px.
    let k = (w.max(h)).div_ceil(800).max(1);
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let mut small = vec![[0.0f32; 3]; sw * sh];
    let mut cnt = vec![0u32; sw * sh];
    for y in 0..h {
        for x in 0..w {
            let i = (y / k) * sw + x / k;
            let p = rgba[y * w + x];
            for c in 0..3 {
                small[i][c] += p[c];
            }
            cnt[i] += 1;
        }
    }
    for (p, c) in small.iter_mut().zip(&cnt) {
        for v in p.iter_mut() {
            *v /= (*c).max(1) as f32;
        }
    }
    // Background = per-channel median of the border.
    let mut border: Vec<[f32; 3]> = Vec::new();
    for x in 0..sw {
        border.push(small[x]);
        border.push(small[(sh - 1) * sw + x]);
    }
    for y in 0..sh {
        border.push(small[y * sw]);
        border.push(small[y * sw + sw - 1]);
    }
    let bg: [f32; 3] = std::array::from_fn(|c| {
        let mut v: Vec<f32> = border.iter().map(|p| p[c]).collect();
        v.sort_by(f32::total_cmp);
        v[v.len() / 2]
    });
    let dist: Vec<f32> = small.iter().map(|p| ((p[0] - bg[0]).powi(2) + (p[1] - bg[1]).powi(2) + (p[2] - bg[2]).powi(2)).sqrt() / 3f32.sqrt()).collect();
    let thr = otsu(&dist).max(0.04);
    let mask: Vec<bool> = dist.iter().map(|d| *d > thr).collect();
    let r = ((sw.max(sh) / 200) as i64).max(1);
    let mask = morph(sw, sh, &morph(sw, sh, &mask, r, true), r, false);
    let mask = morph(sw, sh, &morph(sw, sh, &mask, 1, false), 1, true);
    // Connected components (8-connected).
    let mut label = vec![u32::MAX; sw * sh];
    let mut comps: Vec<Vec<usize>> = Vec::new();
    for s in 0..sw * sh {
        if !mask[s] || label[s] != u32::MAX {
            continue;
        }
        let id = comps.len() as u32;
        let mut stack = vec![s];
        label[s] = id;
        let mut px = Vec::new();
        while let Some(i) = stack.pop() {
            px.push(i);
            let (x, y) = ((i % sw) as i64, (i / sw) as i64);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (xx, yy) = (x + dx, y + dy);
                    if xx < 0 || yy < 0 || xx >= sw as i64 || yy >= sh as i64 {
                        continue;
                    }
                    let j = yy as usize * sw + xx as usize;
                    if mask[j] && label[j] == u32::MAX {
                        label[j] = id;
                        stack.push(j);
                    }
                }
            }
        }
        comps.push(px);
    }
    let total = (sw * sh) as f64;
    let mut out: Vec<FoundPhoto> = comps
        .iter()
        .filter(|c| c.len() as f64 > total * 0.01)
        .map(|c| {
            // Hull of the boundary pixels' corners (in full-resolution px).
            let pts: Vec<[f64; 2]> = c
                .iter()
                .filter(|&&i| {
                    let (x, y) = (i % sw, i / sw);
                    x == 0 || y == 0 || x + 1 == sw || y + 1 == sh || !mask[i - 1] || !mask[i + 1] || !mask[i - sw] || !mask[i + sw]
                })
                .flat_map(|&i| {
                    let (x, y) = ((i % sw) as f64, (i / sw) as f64);
                    [[x, y], [x + 1.0, y], [x, y + 1.0], [x + 1.0, y + 1.0]]
                })
                .map(|p| [p[0] * k as f64, p[1] * k as f64])
                .collect();
            let hull = convex_hull(pts);
            let (center, wd, ht, ang) = min_rect(&hull);
            let inset = k as f64;
            FoundPhoto { center, width: (wd - 2.0 * inset).max(1.0), height: (ht - 2.0 * inset).max(1.0), angle: ang, area: c.len() as f64 / total }
        })
        .collect();
    // Reading order: rows by centre y (bands of a third of the smallest photo), then x.
    let band = out.iter().map(|p| p.height.min(p.width)).fold(f64::MAX, f64::min).max(1.0) / 3.0;
    out.sort_by(|a, b| ((a.center[1] / band).floor()).total_cmp(&(b.center[1] / band).floor()).then(a.center[0].total_cmp(&b.center[0])));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scan: white background with rotated textured rectangles.
    fn scan(w: usize, h: usize, photos: &[FoundPhoto]) -> Vec<[f32; 4]> {
        let mut px = vec![[0.97f32, 0.97, 0.96, 1.0]; w * h];
        for (k, p) in photos.iter().enumerate() {
            let (s, c) = p.angle.to_radians().sin_cos();
            for y in 0..h {
                for x in 0..w {
                    let (dx, dy) = (x as f64 + 0.5 - p.center[0], y as f64 + 0.5 - p.center[1]);
                    let (u, v) = (c * dx + s * dy, -s * dx + c * dy);
                    if u.abs() < p.width / 2.0 && v.abs() < p.height / 2.0 {
                        let t = ((u * 0.05).sin() * (v * 0.07).cos()) as f32;
                        px[y * w + x] = [0.3 + 0.2 * t, 0.2 + 0.1 * k as f32, 0.5 - 0.2 * t, 1.0];
                    }
                }
            }
        }
        px
    }

    #[test]
    fn finds_rotated_photos_in_reading_order() {
        let truth = [
            FoundPhoto { center: [150.0, 120.0], width: 180.0, height: 120.0, angle: 7.0, area: 0.0 },
            FoundPhoto { center: [450.0, 130.0], width: 160.0, height: 130.0, angle: -12.0, area: 0.0 },
            FoundPhoto { center: [300.0, 330.0], width: 220.0, height: 110.0, angle: 3.0, area: 0.0 },
        ];
        let (w, h) = (600, 440);
        let found = find_photos(w, h, &scan(w, h, &truth));
        assert_eq!(found.len(), 3, "{found:?}");
        for (f, t) in found.iter().zip(&truth) {
            assert!((f.angle - t.angle).abs() < 1.0, "{f:?} vs {t:?}");
            assert!((f.center[0] - t.center[0]).abs() < 3.0 && (f.center[1] - t.center[1]).abs() < 3.0, "{f:?}");
            assert!((f.width - t.width).abs() < 6.0 && (f.height - t.height).abs() < 6.0, "{f:?}");
        }
        // Corners are consistent with the centre.
        let c = found[0].corners();
        let mid = [(c[0][0] + c[2][0]) / 2.0, (c[0][1] + c[2][1]) / 2.0];
        assert!((mid[0] - found[0].center[0]).abs() < 1e-9);
        assert!(find_photos(w, h, &vec![[1.0, 1.0, 1.0, 1.0]; w * h]).is_empty());
    }
}
