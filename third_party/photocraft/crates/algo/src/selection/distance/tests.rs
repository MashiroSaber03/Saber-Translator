use super::*;

fn brute(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    let sites: Vec<_> = inside.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| (i % w, i / w)).collect();
    (0..w * h)
        .map(|i| {
            sites
                .iter()
                .map(|&(x, y)| {
                    let dx = (i % w) as f64 - x as f64;
                    let dy = (i / w) as f64 - y as f64;
                    (dx * dx + dy * dy).sqrt() as f32
                })
                .fold(EMPTY_DISTANCE, f32::min)
        })
        .collect()
}

#[test]
fn matches_brute_force_for_every_small_mask() {
    for (w, h) in [(1, 1), (1, 8), (8, 1), (3, 3), (4, 3)] {
        for bits in 0u64..1 << (w * h) {
            let inside: Vec<_> = (0..w * h).map(|i| bits & (1 << i) != 0).collect();
            assert_eq!(edt(&inside, w, h), brute(&inside, w, h), "{w}x{h}: {bits}");
        }
    }
}

#[test]
fn rectangular_and_partial_bands_match_the_previous_transform() {
    for (w, h) in [(1, 101), (101, 1), (17, 33), (73, 65), (9, 255), (513, 137)] {
        for kind in 0..4 {
            let inside: Vec<_> = (0..w * h)
                .map(|i| match kind {
                    0 => false,
                    1 => true,
                    2 => i == 0 || i == w * h - 1,
                    _ => (i * 101 + i / w * 37) % 29 == 0,
                })
                .collect();
            assert_eq!(edt(&inside, w, h), old_edt(&inside, w, h));
        }
    }
}

#[test]
fn large_coordinates_keep_nearby_parabolas_distinct() {
    // Adjacent sites late in a PSB-sized row lost their half-pixel boundaries when
    // the previous transform subtracted f32 coordinate squares.
    for (w, h) in [(65_537, 2), (2, 65_537)] {
        let mut inside = vec![false; w * h];
        if w > h {
            inside[w - 3] = true;
            inside[w - 2] = true;
            inside[2 * w - 1] = true;
        } else {
            inside[(h - 3) * w] = true;
            inside[(h - 2) * w] = true;
            inside[h * w - 1] = true;
        }
        assert_eq!(edt(&inside, w, h), brute(&inside, w, h));
    }
}

#[test]
fn selected_pixels_stay_at_zero_past_coordinate_4096() {
    let inside = vec![true; 6000];
    assert_eq!(edt(&inside, 6000, 1), vec![0.0; 6000]);
    assert!(old_edt(&inside, 6000, 1).iter().any(|d| *d != 0.0), "the previous transform reproduces the regression");
}

#[test]
fn inconsistent_and_empty_dimensions_do_not_panic() {
    for (mask, w, h) in
        [(&[][..], 0, 0), (&[][..], 1, 0), (&[][..], 0, 1), (&[][..], 1, 1), (&[true][..], 2, 1), (&[true, false][..], 1, 1), (&[][..], usize::MAX, 2)]
    {
        assert!(edt(mask, w, h).is_empty());
    }
}

#[test]
fn selection_modifiers_keep_fractional_coverage() {
    // Border is no longer expand − contract (it is Photoshop's soft band, checked against
    // measurements in `selection::tests::border_matches_photoshop`), so only these two remain.
    use crate::selection::{contract, expand};
    let (w, h) = (47, 39);
    let m: Vec<f32> = (0..w * h).map(|i| if (8..31).contains(&(i % w)) && (5..27).contains(&(i / w)) { 0.8 } else { 0.2 }).collect();
    let reference_expand = |m: &[f32], r: f32| {
        let inside: Vec<_> = m.iter().map(|v| *v >= 0.5).collect();
        m.iter().zip(old_edt(&inside, w, h)).map(|(v, d)| v.max((r + 1.0 - d).clamp(0.0, 1.0))).collect::<Vec<_>>()
    };
    for r in [0.0, 1.0, 1.25, 3.75, 16.0] {
        assert_eq!(expand(&m, w, h, r), reference_expand(&m, r));
        let inverse: Vec<_> = m.iter().map(|v| 1.0 - v).collect();
        let contracted: Vec<_> = reference_expand(&inverse, r).iter().map(|v| 1.0 - v).collect();
        assert_eq!(contract(&m, w, h, r), contracted);
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn results_do_not_depend_on_the_worker_count() {
    let (w, h) = (521, 139);
    let inside: Vec<_> = (0..w * h).map(|i| (i * 101 + i / w * 37) % 97 == 0).collect();
    let expected = old_edt(&inside, w, h);
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
        assert_eq!(pool.install(|| edt(&inside, w, h)), expected, "{workers} workers");
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "release comparison on 24 and 36 MP masks"]
#[allow(clippy::assertions_on_constants)] // This opt-in benchmark needs an optimized build.
fn distance_transform_release_comparison() {
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    let workers = std::env::var("EDT_WORKERS").ok().and_then(|s| s.parse().ok()).unwrap_or(16);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
    for (w, h) in [(6000usize, 4000usize), (7200, 5000)] {
        for kind in ["ellipse", "sparse", "dense"] {
            let inside: Vec<_> = (0..w * h)
                .map(|i| {
                    let (x, y) = (i % w, i / w);
                    match kind {
                        "ellipse" => {
                            let dx = x as f64 - w as f64 * 0.5;
                            let dy = y as f64 - h as f64 * 0.5;
                            (dx / (w as f64 * 0.35)).powi(2) + (dy / (h as f64 * 0.3)).powi(2) < 1.0
                        }
                        "sparse" => x % 251 == 17 && y % 241 == 23,
                        _ => (i.wrapping_mul(2_654_435_761) ^ (i >> 8)) % 5 < 2,
                    }
                })
                .collect();
            let before = old_edt(&inside, w, h);
            let after = pool.install(|| edt(&inside, w, h));
            assert!(inside.iter().zip(&after).all(|(seed, distance)| !seed || *distance == 0.0));
            let mut different = 0usize;
            let mut max_error = 0.0f32;
            for (a, b) in before.iter().zip(&after) {
                if a != b {
                    different += 1;
                    max_error = max_error.max((a - b).abs());
                }
            }
            // Validate the large transform independently at corners, seed sites and a grid.
            let sites: Vec<_> = inside.iter().enumerate().filter(|(_, b)| **b).map(|(i, _)| (i % w, i / w)).collect();
            for i in [0, w - 1, w * (h - 1), w * h - 1, w * h / 2 + w / 2] {
                let expected = sites
                    .iter()
                    .map(|&(x, y)| {
                        let dx = (i % w) as f64 - x as f64;
                        let dy = (i / w) as f64 - y as f64;
                        (dx * dx + dy * dy).sqrt() as f32
                    })
                    .fold(EMPTY_DISTANCE, f32::min);
                assert_eq!(after[i], expected);
            }
            drop(sites);
            drop(before);
            drop(after);
            let mut old_ms = Vec::new();
            let mut new_ms = Vec::new();
            for run in 0..3 {
                let mut measure = |fast: bool| {
                    let t = Instant::now();
                    let result = if fast { pool.install(|| edt(black_box(&inside), w, h)) } else { old_edt(black_box(&inside), w, h) };
                    let ms = t.elapsed().as_secs_f64() * 1000.0;
                    black_box(&result);
                    if fast {
                        new_ms.push(ms);
                    } else {
                        old_ms.push(ms);
                    }
                };
                measure(run % 2 == 0);
                measure(run % 2 != 0);
            }
            old_ms.sort_by(f64::total_cmp);
            new_ms.sort_by(f64::total_cmp);
            println!(
                "EDT {w}x{h} {kind} workers={workers} old_ms={old_ms:?} new_ms={new_ms:?} speedup={:.2} different={different} max_error={max_error}",
                old_ms[1] / new_ms[1]
            );
        }
    }
}
/// 1D squared distance transform (Felzenszwalb & Huttenlocher).
fn old_dt1(f: &[f32], out: &mut [f32], v: &mut [usize], z: &mut [f32]) {
    let n = f.len();
    let mut k = 0usize;
    v[0] = 0;
    z[0] = f32::NEG_INFINITY;
    z[1] = f32::INFINITY;
    for q in 1..n {
        loop {
            let p = v[k];
            let s = ((f[q] + (q * q) as f32) - (f[p] + (p * p) as f32)) / (2.0 * (q as f32 - p as f32));
            if s <= z[k] && k > 0 {
                k -= 1;
                continue;
            }
            if s <= z[k] {
                v[0] = q;
                z[0] = f32::NEG_INFINITY;
                z[1] = f32::INFINITY;
                break;
            }
            k += 1;
            v[k] = q;
            z[k] = s;
            z[k + 1] = f32::INFINITY;
            break;
        }
    }
    k = 0;
    for (q, o) in out.iter_mut().enumerate() {
        while z[k + 1] < q as f32 {
            k += 1;
        }
        let d = q as f32 - v[k] as f32;
        *o = d * d + f[v[k]];
    }
}

/// Euclidean distance to the nearest `true` pixel.
fn old_edt(inside: &[bool], w: usize, h: usize) -> Vec<f32> {
    let mut g: Vec<f32> = inside.iter().map(|&b| if b { 0.0 } else { 1e20 }).collect();
    let n = w.max(h).max(1);
    let (mut f, mut o, mut v, mut z) = (vec![0.0; n], vec![0.0; n], vec![0usize; n], vec![0.0f32; n + 1]);
    for x in 0..w {
        for y in 0..h {
            f[y] = g[y * w + x];
        }
        old_dt1(&f[..h], &mut o[..h], &mut v, &mut z);
        for y in 0..h {
            g[y * w + x] = o[y];
        }
    }
    for y in 0..h {
        f[..w].copy_from_slice(&g[y * w..(y + 1) * w]);
        old_dt1(&f[..w], &mut o[..w], &mut v, &mut z);
        for x in 0..w {
            g[y * w + x] = o[x].sqrt();
        }
    }
    g
}
