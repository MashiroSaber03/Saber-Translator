//! Gradient-domain ("Poisson") blending, the core of the Healing Brush and Spot Healing.
//!
//! Method: P. Pérez, M. Gangnet, A. Blake, *Poisson Image Editing*, ACM SIGGRAPH 2003. Seamless
//! cloning finds `f` over a region Ω minimising `∬_Ω |∇f − v|²` with `f = f*` (the destination) on
//! the boundary ∂Ω, which gives the Poisson equation `Δf = div v`. With the guidance field taken as
//! the source gradient, `v = ∇g`, the solution splits as `f = g + h` where `h` is the *membrane*
//! (harmonic) interpolant of the boundary mismatch: `Δh = 0` in Ω, `h = f* − g` on ∂Ω (§3 of the
//! paper). So the healed pixels keep the source's texture (its gradients) while their colour and
//! lighting are pulled to match the destination around the stroke.
//!
//! The Laplace equation is solved on the 4-connected pixel grid by successive over-relaxation (SOR,
//! Gauss–Seidel with over-relaxation), accelerated *cascadically*: the problem is restricted to a
//! half-resolution grid recursively, solved there, bilinearly prolonged as the initial guess, and only
//! a few SOR sweeps are needed per level to remove the remaining high-frequency error. Harmonic
//! functions are smooth, so the coarse guess is already close; this keeps a 300 px dab at a few
//! milliseconds.
//!
//! Buffers are planar-per-call but interleaved at the API: `w × h × ch` normalised floats in any
//! colour model (channels are solved independently, in parallel on native targets).

/// Convergence threshold (max per-sweep update) for the SOR sweeps.
const TOL: f32 = 2e-5;
/// SOR sweeps (cap) and relaxation factor on every level above the coarsest: tuned so a 96×72 disc
/// is within ~1e-3 of a fully converged Gauss–Seidel solve (below one 8-bit step).
const FINE_ITERS: usize = 30;
const FINE_OMEGA: f32 = 1.7;

/// Solve `Δv = 0` on `unknown` pixels with Dirichlet values from the known pixels of `v`
/// (out-of-grid neighbours are ignored, i.e. a Neumann boundary). `v` holds the known values on
/// entry and the solution on exit. Cascadic multigrid + SOR.
pub fn solve_membrane(w: usize, h: usize, unknown: &[bool], v: &mut [f32]) {
    assert_eq!(unknown.len(), w * h);
    assert_eq!(v.len(), w * h);
    solve_rec(w, h, unknown, v, 0, &|| false);
}

fn solve_rec(w: usize, h: usize, unknown: &[bool], v: &mut [f32], depth: usize, stop: &(dyn Fn() -> bool + Sync)) {
    if stop() {
        return;
    }
    let n_unknown = unknown.iter().filter(|u| **u).count();
    if n_unknown == 0 {
        return;
    }
    let known: Vec<f32> = v.iter().zip(unknown).filter(|(_, u)| !**u).map(|(x, _)| *x).collect();
    if known.is_empty() {
        return; // nothing to interpolate from
    }
    if w >= 8 && h >= 8 && n_unknown > 64 && depth < 16 {
        // Restrict: a coarse pixel is known if any of its children is (value = mean of known children).
        let (cw, chh) = (w.div_ceil(2), h.div_ceil(2));
        let mut cu = vec![true; cw * chh];
        let mut cv = vec![0.0f32; cw * chh];
        for cy in 0..chh {
            for cx in 0..cw {
                let (mut s, mut n) = (0.0f32, 0u32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (x, y) = (cx * 2 + dx, cy * 2 + dy);
                    if x < w && y < h && !unknown[y * w + x] {
                        s += v[y * w + x];
                        n += 1;
                    }
                }
                if n > 0 {
                    cu[cy * cw + cx] = false;
                    cv[cy * cw + cx] = s / n as f32;
                }
            }
        }
        solve_rec(cw, chh, &cu, &mut cv, depth + 1, stop);
        // Prolong: bilinear sample of the coarse solution as the initial guess.
        for y in 0..h {
            let fy = ((y as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (chh - 1) as f32);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(chh - 1);
            let ty = fy - y0 as f32;
            for x in 0..w {
                if !unknown[y * w + x] {
                    continue;
                }
                let fx = ((x as f32 + 0.5) / 2.0 - 0.5).clamp(0.0, (cw - 1) as f32);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(cw - 1);
                let tx = fx - x0 as f32;
                let a = cv[y0 * cw + x0] + (cv[y0 * cw + x1] - cv[y0 * cw + x0]) * tx;
                let b = cv[y1 * cw + x0] + (cv[y1 * cw + x1] - cv[y1 * cw + x0]) * tx;
                v[y * w + x] = a + (b - a) * ty;
            }
        }
        sor(w, h, unknown, v, FINE_ITERS, FINE_OMEGA, stop);
    } else {
        let mean = known.iter().sum::<f32>() / known.len() as f32;
        for (x, u) in v.iter_mut().zip(unknown) {
            if *u {
                *x = mean;
            }
        }
        sor(w, h, unknown, v, 2000, 1.85, stop);
    }
}

/// SOR sweeps over the unknown pixels until the largest update drops below [`TOL`].
fn sor(w: usize, h: usize, unknown: &[bool], v: &mut [f32], max_iter: usize, omega: f32, stop: &(dyn Fn() -> bool + Sync)) {
    // Precompute the unknown pixels and which neighbours exist.
    let cells: Vec<(u32, u8)> = (0..w * h)
        .filter(|&i| unknown[i])
        .map(|i| {
            let (x, y) = (i % w, i / w);
            let m = u8::from(x > 0) | (u8::from(x + 1 < w) << 1) | (u8::from(y > 0) << 2) | (u8::from(y + 1 < h) << 3);
            (i as u32, m)
        })
        .collect();
    for _ in 0..max_iter {
        if stop() {
            return;
        }
        let mut max_d = 0.0f32;
        for &(i, m) in &cells {
            let i = i as usize;
            let (s, n) = if m == 0b1111 {
                (v[i - 1] + v[i + 1] + v[i - w] + v[i + w], 4.0)
            } else {
                let mut s = 0.0;
                let mut n = 0.0;
                if m & 1 != 0 {
                    s += v[i - 1];
                    n += 1.0;
                }
                if m & 2 != 0 {
                    s += v[i + 1];
                    n += 1.0;
                }
                if m & 4 != 0 {
                    s += v[i - w];
                    n += 1.0;
                }
                if m & 8 != 0 {
                    s += v[i + w];
                    n += 1.0;
                }
                if n == 0.0 {
                    continue;
                }
                (s, n)
            };
            let d = omega * (s / n - v[i]);
            v[i] += d;
            max_d = max_d.max(d.abs());
        }
        if max_d < TOL {
            break;
        }
    }
}

/// Reference solver for tests: plain Gauss–Seidel from a zero start until the update is below `tol`.
pub fn solve_membrane_reference(w: usize, h: usize, unknown: &[bool], v: &mut [f32], tol: f32, max_iter: usize) {
    for (x, u) in v.iter_mut().zip(unknown) {
        if *u {
            *x = 0.0;
        }
    }
    for _ in 0..max_iter {
        let mut max_d = 0.0f32;
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                if !unknown[i] {
                    continue;
                }
                let (mut s, mut n) = (0.0, 0.0);
                for (nx, ny) in [(x.wrapping_sub(1), y), (x + 1, y), (x, y.wrapping_sub(1)), (x, y + 1)] {
                    if nx < w && ny < h {
                        s += v[ny * w + nx];
                        n += 1.0;
                    }
                }
                if n > 0.0 {
                    let nv = s / n;
                    max_d = max_d.max((nv - v[i]).abs());
                    v[i] = nv;
                }
            }
        }
        if max_d < tol {
            break;
        }
    }
}

/// Map `f` over the channels of an interleaved buffer as planar slices (parallel on native).
fn per_channel(w: usize, h: usize, ch: usize, f: impl Fn(usize) -> Vec<f32> + Sync + Send) -> Vec<f32> {
    #[cfg(not(target_arch = "wasm32"))]
    let planes: Vec<Vec<f32>> = if w * h >= 64 * 64 {
        use rayon::prelude::*;
        (0..ch).into_par_iter().map(&f).collect()
    } else {
        (0..ch).map(&f).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let planes: Vec<Vec<f32>> = (0..ch).map(&f).collect();
    let mut out = vec![0.0f32; w * h * ch];
    for (c, p) in planes.iter().enumerate() {
        for (i, v) in p.iter().enumerate() {
            out[i * ch + c] = *v;
        }
    }
    out
}

/// Seamless cloning (Pérez et al. 2003, guidance field `v = ∇src`): returns `f` with `f = dst` outside
/// `mask` and, inside, `src` plus the membrane that removes the boundary mismatch. So the result keeps
/// the source's texture and takes the destination's colour and lighting.
///
/// `src`/`dst` are interleaved `w × h × ch`; `mask` has `w × h` entries (`true` = healed). Pixels on the
/// grid edge that are in the mask see a Neumann (free) boundary there, so callers should leave a one
/// pixel unmasked margin for a fully Dirichlet problem.
pub fn seamless_clone(w: usize, h: usize, ch: usize, src: &[f32], dst: &[f32], mask: &[bool]) -> Vec<f32> {
    assert_eq!(src.len(), w * h * ch);
    assert_eq!(dst.len(), w * h * ch);
    assert_eq!(mask.len(), w * h);
    per_channel(w, h, ch, |c| {
        let mut hv: Vec<f32> = (0..w * h).map(|i| if mask[i] { 0.0 } else { dst[i * ch + c] - src[i * ch + c] }).collect();
        solve_membrane(w, h, mask, &mut hv);
        (0..w * h).map(|i| if mask[i] { src[i * ch + c] + hv[i] } else { dst[i * ch + c] }).collect()
    })
}

/// Membrane fill (no source texture): harmonic interpolation of the known pixels into `hole`.
pub fn membrane_fill(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool]) -> Vec<f32> {
    per_channel(w, h, ch, |c| {
        let mut v: Vec<f32> = (0..w * h).map(|i| if hole[i] { 0.0 } else { img[i * ch + c] }).collect();
        solve_membrane(w, h, hole, &mut v);
        v
    })
}

/// [`membrane_fill`] that stops at the next SOR sweep once `ctl` is cancelled.
pub fn membrane_fill_with(
    w: usize,
    h: usize,
    ch: usize,
    img: &[f32],
    hole: &[bool],
    ctl: &photocraft_raster::Interrupt,
) -> Result<Vec<f32>, photocraft_raster::Cancelled> {
    ctl.check()?;
    let stop = || ctl.cancelled();
    let out = per_channel(w, h, ch, |c| {
        let mut v: Vec<f32> = (0..w * h).map(|i| if hole[i] { 0.0 } else { img[i * ch + c] }).collect();
        if hole.len() == w * h && v.len() == w * h {
            solve_rec(w, h, hole, &mut v, 0, &stop);
        }
        v
    });
    ctl.check()?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Vec<bool> {
        (0..w * h).map(|i| ((i % w) as f32 - cx).hypot((i / w) as f32 - cy) < r).collect()
    }

    #[test]
    fn membrane_reproduces_linear_boundary() {
        // A linear function is harmonic: the solve must reproduce it exactly inside the hole.
        let (w, h) = (80, 60);
        let unknown = disc(w, h, 40.0, 30.0, 20.0);
        let truth: Vec<f32> = (0..w * h).map(|i| 0.2 + 0.005 * (i % w) as f32 + 0.003 * (i / w) as f32).collect();
        let mut v: Vec<f32> = truth.iter().zip(&unknown).map(|(t, u)| if *u { 0.0 } else { *t }).collect();
        solve_membrane(w, h, &unknown, &mut v);
        let err = v.iter().zip(&truth).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 2e-3, "{err}");
    }

    #[test]
    fn multigrid_matches_reference_solve() {
        let (w, h) = (96, 72);
        let unknown = disc(w, h, 45.0, 36.0, 28.0);
        // Arbitrary, non-harmonic boundary data.
        let init: Vec<f32> = (0..w * h).map(|i| ((i % w) as f32 * 0.21).sin() * 0.3 + ((i / w) as f32 * 0.13).cos() * 0.2 + 0.5).collect();
        let mut fast = init.clone();
        solve_membrane(w, h, &unknown, &mut fast);
        let mut reference = init.clone();
        solve_membrane_reference(w, h, &unknown, &mut reference, 1e-7, 100_000);
        let err = fast.iter().zip(&reference).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 2e-3, "max error vs reference {err}");
        // Known pixels untouched.
        assert!(fast.iter().zip(&init).zip(&unknown).all(|((a, b), u)| *u || a == b));
    }

    #[test]
    fn seamless_clone_takes_destination_colour_and_source_texture() {
        let (w, h, ch) = (64, 64, 3);
        let mask = disc(w, h, 32.0, 32.0, 20.0);
        // Destination: flat reddish; source: bluish with a stripe texture.
        let dst: Vec<f32> = (0..w * h).flat_map(|_| [0.7, 0.3, 0.2]).collect();
        let tex = |x: usize| if (x / 4).is_multiple_of(2) { 0.05 } else { -0.05 };
        let src: Vec<f32> = (0..w * h)
            .flat_map(|i| {
                let t = tex(i % w);
                [0.1 + t, 0.2 + t, 0.8 + t]
            })
            .collect();
        let out = seamless_clone(w, h, ch, &src, &dst, &mask);
        // Mean colour inside the mask matches the destination.
        let inside: Vec<usize> = (0..w * h).filter(|&i| mask[i]).collect();
        for c in 0..ch {
            let m = inside.iter().map(|&i| out[i * ch + c]).sum::<f32>() / inside.len() as f32;
            assert!((m - dst[c]).abs() < 0.02, "channel {c} mean {m}");
        }
        // Source gradients carried: horizontal differences inside match the source's.
        let (mut e, mut n) = (0.0f32, 0);
        for &i in &inside {
            if mask[i + 1] {
                e += ((out[(i + 1) * ch] - out[i * ch]) - (src[(i + 1) * ch] - src[i * ch])).abs();
                n += 1;
            }
        }
        assert!(e / (n as f32) < 0.01, "gradient error {}", e / n as f32);
        // Outside untouched.
        assert_eq!(&out[..3], &dst[..3]);
    }
}
