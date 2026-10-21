//! Click-local red-eye correction: neutralize a red pupil around a click.
//!
//! The kernel never scans a full document. Callers pass the search window plus a
//! read halo that covers a 1 px morphological shrink and a Gaussian of σ = 0.8.

#![allow(clippy::needless_range_loop)]

/// Gaussian σ applied to the pupil mask after a 1 px shrink.
pub const SIGMA: f32 = 0.8;
/// Largest window (search + halo) the engine may allocate, in pixels on a side.
pub const MAX_WINDOW: u32 = 512;
/// Clicks one command may apply in a single history step.
pub const MAX_POINTS: usize = 64;

/// How far outside the search window the mask filter reads.
///
/// `1` (erode) + `ceil(σ × 3)` (Gaussian) + `1` (margin) — recompute if [`SIGMA`] changes.
pub fn halo_radius() -> i32 {
    let blur = (SIGMA * 3.0).ceil();
    let blur = if blur.is_finite() { blur.clamp(0.0, 64.0) as i32 } else { 3 };
    1 + blur + 1
}

/// Search radius in document pixels for a Pupil Size in 1..=100.
pub fn search_radius(pupil_size: f32) -> f32 {
    let p = if pupil_size.is_finite() { pupil_size.clamp(1.0, 100.0) } else { 50.0 };
    (4.0 + 1.20 * p).clamp(8.0, 128.0)
}

/// Side length of the read rect (search + halo) before clamping Pupil Size to 1..=100.
///
/// `None` when `pupil_size` is non-finite or the side would overflow [`MAX_WINDOW`].
pub fn unclamped_window_side(pupil_size: f32) -> Option<u32> {
    if !pupil_size.is_finite() {
        return None;
    }
    let r = 4.0 + 1.20 * pupil_size;
    if !r.is_finite() {
        return None;
    }
    let half = r.abs() + halo_radius() as f32;
    if !half.is_finite() || half > (MAX_WINDOW as f32) {
        return None;
    }
    let side = (2.0 * half + 1.0).ceil();
    if !side.is_finite() || side > MAX_WINDOW as f32 {
        return None;
    }
    Some(side as u32)
}

fn luma(p: [f32; 4]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

fn is_red_eye(p: [f32; 4]) -> bool {
    let (r, g, b) = (p[0], p[1], p[2]);
    if !(r.is_finite() && g.is_finite() && b.is_finite()) {
        return false;
    }
    if r <= g + 0.15 || r <= b + 0.15 {
        return false;
    }
    if luma(p) >= 0.92 {
        return false;
    }
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    max - min >= 0.08
}

fn idx(x: usize, y: usize, w: usize) -> Option<usize> {
    y.checked_mul(w).and_then(|row| row.checked_add(x))
}

fn within_search(dx: i32, dy: i32, radius: f32) -> bool {
    let d = (dx as f32).hypot(dy as f32);
    d.is_finite() && d <= radius
}

fn nearest_seed(mask: &[bool], w: usize, h: usize, origin: (i32, i32), click: (i32, i32), radius: f32) -> Option<usize> {
    let mut best: Option<(usize, i64)> = None;
    for y in 0..h {
        for x in 0..w {
            let Some(i) = idx(x, y, w) else { continue };
            if !mask.get(i).copied().unwrap_or(false) {
                continue;
            }
            let dx = origin.0.saturating_add(x as i32).saturating_sub(click.0);
            let dy = origin.1.saturating_add(y as i32).saturating_sub(click.1);
            if !within_search(dx, dy, radius) {
                continue;
            }
            let d2 = (dx as i64).saturating_mul(dx as i64).saturating_add((dy as i64).saturating_mul(dy as i64));
            match best {
                Some((_, bd)) if d2 >= bd => {}
                _ => best = Some((i, d2)),
            }
        }
    }
    best.map(|(i, _)| i)
}

fn flood_keep(mask: &mut [bool], w: usize, h: usize, seed: usize) {
    let n = mask.len();
    if seed >= n || w == 0 {
        return;
    }
    let mut keep = vec![false; n];
    let mut stack = vec![seed];
    while let Some(i) = stack.pop() {
        let Some(slot) = keep.get_mut(i) else { continue };
        if *slot || !mask.get(i).copied().unwrap_or(false) {
            continue;
        }
        *slot = true;
        let x = i % w;
        let y = i / w;
        if x > 0 {
            stack.push(i - 1);
        }
        if x + 1 < w {
            stack.push(i + 1);
        }
        if y > 0
            && let Some(up) = i.checked_sub(w)
        {
            stack.push(up);
        }
        if y + 1 < h
            && let Some(down) = i.checked_add(w)
        {
            stack.push(down);
        }
        if stack.len() > n.saturating_mul(4) {
            break;
        }
    }
    for (m, k) in mask.iter_mut().zip(keep.iter()) {
        *m = *k;
    }
}

fn erode(mask: &[bool], w: usize, h: usize) -> Vec<bool> {
    let mut out = vec![false; mask.len()];
    for y in 0..h {
        for x in 0..w {
            let Some(i) = idx(x, y, w) else { continue };
            let mut on = true;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = x as i32 + dx;
                    let yy = y as i32 + dy;
                    if xx < 0 || yy < 0 || xx >= w as i32 || yy >= h as i32 {
                        on = false;
                        continue;
                    }
                    let Some(j) = idx(xx as usize, yy as usize, w) else {
                        on = false;
                        continue;
                    };
                    if !mask.get(j).copied().unwrap_or(false) {
                        on = false;
                    }
                }
            }
            if let Some(slot) = out.get_mut(i) {
                *slot = on;
            }
        }
    }
    out
}

fn blur(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let n = w.saturating_mul(h);
    if w == 0 || h == 0 || src.len() < n {
        return vec![0.0; n];
    }
    if !sigma.is_finite() || sigma < 0.1 {
        return src.get(..n.min(src.len())).unwrap_or(src).to_vec();
    }
    let r = (sigma * 3.0).ceil();
    let r = if r.is_finite() { r.clamp(1.0, 32.0) as i32 } else { 3 };
    let mut k = Vec::new();
    let mut sum = 0.0f32;
    for i in -r..=r {
        let wgt = (-(i * i) as f32 / (2.0 * sigma * sigma)).exp();
        if !wgt.is_finite() {
            continue;
        }
        k.push(wgt);
        sum += wgt;
    }
    if k.is_empty() || !sum.is_finite() || sum <= 0.0 {
        return src.get(..n.min(src.len())).unwrap_or(src).to_vec();
    }
    for wgt in &mut k {
        *wgt /= sum;
    }
    let mut tmp = vec![0.0f32; n];
    let mut out = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0f32;
            for (ki, wgt) in k.iter().enumerate() {
                let xx = (x as i32 + ki as i32 - r).clamp(0, w.saturating_sub(1) as i32) as usize;
                let Some(j) = idx(xx, y, w) else { continue };
                a += src.get(j).copied().unwrap_or(0.0) * wgt;
            }
            if let Some(slot) = idx(x, y, w).and_then(|i| tmp.get_mut(i)) {
                *slot = a;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            let mut a = 0.0f32;
            for (ki, wgt) in k.iter().enumerate() {
                let yy = (y as i32 + ki as i32 - r).clamp(0, h.saturating_sub(1) as i32) as usize;
                let Some(j) = idx(x, yy, w) else { continue };
                a += tmp.get(j).copied().unwrap_or(0.0) * wgt;
            }
            if let Some(slot) = idx(x, y, w).and_then(|i| out.get_mut(i)) {
                *slot = a;
            }
        }
    }
    out
}

/// Correct red-eye in `px` (row-major RGBA, including the [`halo_radius`] border).
///
/// `origin` is the document coordinate of `px[0]`. Returns how many pixels changed.
/// Pixels farther than [`search_radius`] from `click` are left untouched.
pub fn apply(px: &mut [[f32; 4]], w: usize, h: usize, click: (i32, i32), origin: (i32, i32), pupil_size: f32, darken: f32) -> u32 {
    let n = w.saturating_mul(h);
    if w == 0 || h == 0 || px.len() < n {
        return 0;
    }
    let radius = search_radius(pupil_size);
    let t_amt = if darken.is_finite() { (darken.clamp(0.0, 100.0) / 100.0).clamp(0.0, 1.0) } else { 0.5 };
    let mut hard = vec![false; n];
    for y in 0..h {
        for x in 0..w {
            let Some(i) = idx(x, y, w) else { continue };
            let Some(p) = px.get(i).copied() else { continue };
            let dx = origin.0.saturating_add(x as i32).saturating_sub(click.0);
            let dy = origin.1.saturating_add(y as i32).saturating_sub(click.1);
            if within_search(dx, dy, radius)
                && is_red_eye(p)
                && let Some(slot) = hard.get_mut(i)
            {
                *slot = true;
            }
        }
    }
    let Some(seed) = nearest_seed(&hard, w, h, origin, click, radius) else {
        return 0;
    };
    flood_keep(&mut hard, w, h, seed);
    let eroded = erode(&hard, w, h);
    let mut soft_in = vec![0.0f32; n];
    for (s, on) in soft_in.iter_mut().zip(eroded.iter()) {
        *s = if *on { 1.0 } else { 0.0 };
    }
    let soft = blur(&soft_in, w, h, SIGMA);
    let mut changed = 0u32;
    for y in 0..h {
        for x in 0..w {
            let Some(i) = idx(x, y, w) else { continue };
            let Some(p) = px.get_mut(i) else { continue };
            let dx = origin.0.saturating_add(x as i32).saturating_sub(click.0);
            let dy = origin.1.saturating_add(y as i32).saturating_sub(click.1);
            if !within_search(dx, dy, radius) {
                continue;
            }
            if luma(*p) >= 0.92 {
                continue;
            }
            let m = soft.get(i).copied().unwrap_or(0.0).clamp(0.0, 1.0);
            let t = m * t_amt;
            if t <= 1.0e-4 {
                continue;
            }
            let (r0, g0, b0, a0) = (p[0], p[1], p[2], p[3]);
            let neut = (g0 + b0) * 0.5;
            let r1 = r0 + (neut - r0) * t;
            let scale = 1.0 - 0.45 * t;
            let (r2, g2, b2) = (r1 * scale, g0 * scale, b0 * scale);
            if [r2, g2, b2].iter().any(|c| !c.is_finite()) {
                continue;
            }
            if (r2 - r0).abs() < 1.0e-6 && (g2 - g0).abs() < 1.0e-6 && (b2 - b0).abs() < 1.0e-6 {
                continue;
            }
            *p = [r2, g2, b2, a0];
            changed = changed.saturating_add(1);
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(w: usize, h: usize, f: impl Fn(usize, usize) -> [f32; 4]) -> Vec<[f32; 4]> {
        let mut px = vec![[0.0; 4]; w * h];
        for y in 0..h {
            for x in 0..w {
                if let Some(slot) = px.get_mut(y * w + x) {
                    *slot = f(x, y);
                }
            }
        }
        px
    }

    fn disc(x: usize, y: usize, cx: f32, cy: f32, r: f32) -> bool {
        (x as f32 - cx).hypot(y as f32 - cy) <= r
    }

    fn portrait_pixel(x: usize, y: usize) -> [f32; 4] {
        // Brown that fails R > G+0.15 so the iris is never in the red-eye mask.
        let iris = [0.38, 0.28, 0.16, 1.0];
        let red = [0.95, 0.10, 0.10, 1.0];
        let white = [1.0, 1.0, 1.0, 1.0];
        if disc(x, y, 16.0, 16.0, 1.2) {
            white
        } else if disc(x, y, 16.0, 16.0, 10.0) {
            red
        } else {
            iris
        }
    }

    #[test]
    fn halo_covers_erode_and_blur_reach() {
        let want = 1 + (SIGMA * 3.0).ceil() as i32 + 1;
        assert_eq!(halo_radius(), want);
        assert_eq!(halo_radius(), 5);
        assert_eq!(search_radius(50.0), 64.0);
        assert_eq!(search_radius(0.0), 8.0);
        assert_eq!(search_radius(101.0), search_radius(100.0));
        assert!(unclamped_window_side(50.0).is_some_and(|s| s < MAX_WINDOW));
        assert!(unclamped_window_side(10_000.0).is_none());
        assert!(unclamped_window_side(f32::NAN).is_none());
    }

    #[test]
    fn red_disc_is_neutralized_and_iris_is_left_alone() {
        let w = 32usize;
        let mut px = sample(w, w, portrait_pixel);
        let iris_before = px[2 * w + 2];
        let n = apply(&mut px, w, w, (16, 16), (0, 0), 50.0, 100.0);
        assert!(n > 0, "corrected {n} pixels");
        // 5 px from the centre: in the red ring, outside the catchlight, inside the eroded mask.
        let after = px[11 * w + 16];
        let neut = (0.10 + 0.10) / 2.0;
        assert!((after[0] - after[1]).abs() < 0.08 && (after[0] - after[2]).abs() < 0.08, "red no longer dominates: {after:?}");
        assert!((after[0] - neut * 0.55).abs() < 0.08, "R ≈ (G+B)/2 then darkened: {after:?}");
        assert!(luma(after) < luma([0.95, 0.10, 0.10, 1.0]) - 0.05, "pupil darkened");
        let iris_after = px[2 * w + 2];
        for c in 0..4 {
            assert!((iris_after[c] - iris_before[c]).abs() < 1e-5, "iris {iris_after:?} vs {iris_before:?}");
        }
        let catch = px[16 * w + 16];
        assert!(catch[0] >= 0.9 && catch[1] >= 0.9 && catch[2] >= 0.9, "catchlight {catch:?}");
    }

    #[test]
    fn gray_pixels_are_not_treated_as_red_eye() {
        let w = 16usize;
        let mut px = sample(w, w, |_, _| [0.4, 0.4, 0.4, 1.0]);
        let n = apply(&mut px, w, w, (8, 8), (0, 0), 50.0, 50.0);
        assert_eq!(n, 0);
        assert_eq!(px[8 * w + 8], [0.4, 0.4, 0.4, 1.0]);
    }

    #[allow(clippy::too_many_arguments)]
    fn apply_tiled(src: &[[f32; 4]], w: usize, h: usize, click: (i32, i32), pupil: f32, darken: f32, tile: usize, halo: i32) -> Vec<[f32; 4]> {
        let mut out = src.to_vec();
        let halo = halo.max(0) as usize;
        let mut y0 = 0;
        while y0 < h {
            let mut x0 = 0;
            while x0 < w {
                let x1 = (x0 + tile).min(w);
                let y1 = (y0 + tile).min(h);
                let rx0 = x0.saturating_sub(halo);
                let ry0 = y0.saturating_sub(halo);
                let rx1 = (x1 + halo).min(w);
                let ry1 = (y1 + halo).min(h);
                let bw = rx1 - rx0;
                let bh = ry1 - ry0;
                let mut buf = vec![[0.0; 4]; bw * bh];
                for yy in 0..bh {
                    for xx in 0..bw {
                        let s = (ry0 + yy) * w + (rx0 + xx);
                        if let Some(slot) = buf.get_mut(yy * bw + xx) {
                            *slot = src.get(s).copied().unwrap_or([0.0; 4]);
                        }
                    }
                }
                apply(&mut buf, bw, bh, click, (rx0 as i32, ry0 as i32), pupil, darken);
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        let b = (yy - ry0) * bw + (xx - rx0);
                        if let Some(slot) = out.get_mut(yy * w + xx) {
                            *slot = buf.get(b).copied().unwrap_or(*slot);
                        }
                    }
                }
                x0 += tile;
            }
            y0 += tile;
        }
        out
    }

    fn max_neighbour_jump(px: &[[f32; 4]], w: usize, h: usize, x: usize) -> f32 {
        let mut m = 0.0f32;
        if x == 0 || x >= w {
            return 0.0;
        }
        for y in 0..h {
            let Some(a) = px.get(y * w + (x - 1)) else { continue };
            let Some(b) = px.get(y * w + x) else { continue };
            let d = (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs());
            if d > m {
                m = d;
            }
        }
        m
    }

    #[test]
    fn tile_seam_matches_an_untiled_reference() {
        let w = 128usize;
        // A red half-plane ending on the 64 px tile boundary: the mask edge is where a short halo fails.
        let src = sample(w, w, |x, _y| if x < 64 { [0.95, 0.10, 0.10, 1.0] } else { [0.38, 0.28, 0.16, 1.0] });
        let mut reference = src.clone();
        apply(&mut reference, w, w, (60, 64), (0, 0), 50.0, 50.0);
        let tiled = apply_tiled(&src, w, w, (60, 64), 50.0, 50.0, 64, halo_radius());
        let mut err = 0.0f32;
        for (a, b) in reference.iter().zip(tiled.iter()) {
            for c in 0..3 {
                err = err.max((a[c] - b[c]).abs());
            }
        }
        assert!(err < 1e-4, "tiled vs reference max err {err}");
        let jump_ref = max_neighbour_jump(&reference, w, w, 64);
        let jump_tiled = max_neighbour_jump(&tiled, w, w, 64);
        assert!((jump_ref - jump_tiled).abs() < 1e-4, "seam jump {jump_tiled} vs {jump_ref}");

        // Reach is erode (1) + ceil(σ×3) (3). A 1 px halo cannot cover that; the documented halo can.
        let short = apply_tiled(&src, w, w, (60, 64), 50.0, 50.0, 64, 1);
        let mut short_err = 0.0f32;
        for (a, b) in reference.iter().zip(short.iter()) {
            for c in 0..3 {
                short_err = short_err.max((a[c] - b[c]).abs());
            }
        }
        assert!(short_err > err + 1e-4, "a halo smaller than the blur reach must diverge (short {short_err}, full {err})");
    }

    #[test]
    fn empty_or_mismatched_buffer_changes_nothing() {
        let mut px: Vec<[f32; 4]> = Vec::new();
        assert_eq!(apply(&mut px, 0, 0, (0, 0), (0, 0), 50.0, 50.0), 0);
        let mut tiny = vec![[0.95, 0.1, 0.1, 1.0]; 4];
        assert_eq!(apply(&mut tiny, 8, 8, (1, 1), (0, 0), 50.0, 50.0), 0);
    }
}
