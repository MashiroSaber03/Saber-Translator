use super::*;
use crate::segment::{FREE, HARD_BG, HARD_FG, RgbImage, grid_cut_weighted};

#[test]
fn merged_starts_equal_all_live_edge_sources() {
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (4, 3)] {
        for bits in 0..1usize << (w * h) {
            let free: Vec<_> = (0..w * h).filter(|i| bits & (1 << i) != 0).collect();
            let mut expected = Vec::new();
            for y in 0..h {
                for x in 0..w {
                    let i = y * w + x;
                    let live = free.contains(&i)
                        || crate::segment::NB.iter().any(|&(dx, dy, _)| {
                            let (xx, yy) = (x as i32 + dx, y as i32 + dy);
                            xx >= 0 && xx < w as i32 && yy < h as i32 && free.contains(&(yy as usize * w + xx as usize))
                        });
                    if live {
                        expected.push(i)
                    }
                }
            }
            assert_eq!(Starts::new(&free, w).collect::<Vec<_>>(), expected, "{w}x{h} bits={bits}");
        }
    }
}

#[test]
fn every_small_constraint_mask_matches_the_previous_graph() {
    for (w, h) in [(1, 1), (1, 6), (6, 1), (2, 3)] {
        let img = RgbImage::from_fn(w, h, |x, y| [(x % 3) as f32 / 2.0, (y % 4) as f32 / 3.0, ((x + y) % 5) as f32 / 4.0]);
        for code in 0..3usize.pow((w * h) as u32) {
            let mut state = code;
            let fixed: Vec<_> = (0..w * h)
                .map(|_| {
                    let v = state % 3;
                    state /= 3;
                    [FREE, HARD_FG, HARD_BG][v]
                })
                .collect();
            let fg: Vec<_> = (0..w * h).map(|i| (i % 3) as f32 * 0.1).collect();
            let bg: Vec<_> = (0..w * h).map(|i| (i % 4) as f32 * 0.15).collect();
            for kind in 0..3 {
                let weight = |d: f32| match kind {
                    0 => 1.0,
                    1 => 0.3 * (-2.0 * d).exp(),
                    _ => 0.02 + 1.0 / (1.0 + (d.sqrt() / 0.05).powf(1.98)),
                };
                assert_eq!(
                    grid_cut_weighted::<true>(&img, &fg, &bg, &fixed, weight),
                    grid_cut_weighted::<false>(&img, &fg, &bg, &fixed, weight),
                    "{w}x{h} constraints={code} weight={kind}"
                );
            }
        }
    }
}

#[test]
fn sparse_graph_keeps_exact_masks_and_avoids_irrelevant_weights() {
    use std::cell::Cell;
    let (w, h) = (101usize, 93usize);
    let img = RgbImage::from_fn(w, h, |x, y| [((x * 13 + y * 7) % 257) as f32 / 256.0; 3]);
    let cost: Vec<_> = (0..w * h).map(|i| (i % 11) as f32 * 0.01).collect();
    for kind in 0..4 {
        let fixed: Vec<_> = (0..w * h)
            .map(|i| {
                if (kind == 0 && i % 503 == 0) || (kind == 1 && i % w >= 48 && i % w <= 52) || (kind == 2 && i / w == 0) || (kind == 3 && i % w == w - 1) {
                    FREE
                } else if i % w < 50 {
                    HARD_FG
                } else {
                    HARD_BG
                }
            })
            .collect();
        let before = Cell::new(0);
        let after = Cell::new(0);
        let old = grid_cut_weighted::<false>(&img, &cost, &cost, &fixed, |d| {
            before.set(before.get() + 1);
            0.3 * (-2.0 * d).exp()
        });
        let new = grid_cut_weighted::<true>(&img, &cost, &cost, &fixed, |d| {
            after.set(after.get() + 1);
            0.3 * (-2.0 * d).exp()
        });
        assert_eq!(new, old, "kind={kind}");
        assert!(after.get() * 5 < before.get(), "kind={kind} counts {} -> {}", before.get(), after.get());
    }
}

#[test]
fn malformed_grid_dimensions_and_short_inputs_return_empty() {
    for img in [
        RgbImage { w: usize::MAX, h: 2, px: Vec::new() },
        RgbImage { w: i32::MAX as usize + 1, h: 0, px: Vec::new() },
        RgbImage { w: 2, h: 2, px: vec![[0.0; 3]; 3] },
    ] {
        assert!(crate::segment::grid_cut(&img, &[], &[], &[], 1.0, 1.0).is_empty());
    }
    let img = RgbImage::new(2, 2);
    let cost = [0.0; 4];
    let fixed = [FREE; 4];
    for (fg, bg, constraints) in [(&cost[..3], &cost[..], &fixed[..]), (&cost[..], &cost[..3], &fixed[..]), (&cost[..], &cost[..], &fixed[..3])] {
        assert!(crate::segment::grid_cut_with(&img, fg, bg, constraints, |_| 1.0).is_empty());
    }
}
