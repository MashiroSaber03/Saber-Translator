use super::*;
use crate::inpaint::{best_offset, hole_bbox, integral, window_count};

#[test]
fn cropped_counts_match_full_image_for_clipped_and_disjoint_windows() {
    let (w, h) = (37, 29);
    for kind in 0..5 {
        let hole: Vec<_> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                match kind {
                    0 => (11..16).contains(&x) && (8..14).contains(&y),
                    1 => x < 3 && y < 4,
                    2 => x >= w - 3 && y >= h - 4,
                    3 => (x == 3 && y == 8) || (x == 31 && y == 25),
                    _ => (x * 37 + y * 71) % 17 == 0,
                }
            })
            .collect();
        let cropped = HoleIntegral::new(w, &hole, hole_bbox(w, h, &hole).unwrap()).unwrap();
        let full = integral(w, h, &hole);
        for x0 in [-50, -1, 0, 3, 11, 15, 36, 40] {
            for x1 in [-2, 0, 3, 13, 16, 37, 50] {
                for y0 in [-30, -1, 0, 8, 13, 28, 35] {
                    for y1 in [-1, 0, 10, 14, 29, 40] {
                        assert_eq!(cropped.count(x0, y0, x1, y1), window_count(&full, w, h, x0, y0, x1, y1), "kind {kind}: {x0},{y0}..{x1},{y1}");
                    }
                }
            }
        }
    }
}

#[test]
fn table_size_depends_on_the_hole_bounds() {
    let (w, h) = (1000, 700);
    let hole: Vec<_> = (0..w * h).map(|i| (503..514).contains(&(i % w)) && (307..320).contains(&(i / w))).collect();
    let table = HoleIntegral::new(w, &hole, hole_bbox(w, h, &hole).unwrap()).unwrap();
    assert_eq!((table.w, table.h, table.data.len()), (11, 13, 12 * 14));
    assert_eq!(table.count(0, 0, 1000, 700), 143);
    assert_eq!(table.count(i32::MIN, i32::MIN, i32::MAX, i32::MAX), 143);
    assert!(HoleIntegral::new(w, &hole, (10, 5, 9, 6)).is_none());
    assert!(HoleIntegral::new(w, &[], (3, 4, 5, 6)).is_none());
    assert!(HoleIntegral::new(w, &[], (0, 0, usize::MAX, 2)).is_none());
}

#[test]
fn proximity_offsets_match_full_tables_at_several_depths_and_channels() {
    let (w, h) = (61, 49);
    for ch in [1, 3, 4, 5] {
        for depth in [8, 16, 32] {
            let img: Vec<_> = (0..w * h * ch)
                .map(|i| {
                    let value = ((i * 101 + i / (w * ch) * 37) % 251) as f32 / 250.0;
                    match depth {
                        8 => (value * 255.0).round() / 255.0,
                        16 => (value * 65535.0).round() / 65535.0,
                        _ => value * 2.0 - 0.5,
                    }
                })
                .collect();
            for kind in 0..5 {
                let hole: Vec<_> = (0..w * h)
                    .map(|i| {
                        let (x, y) = (i % w, i / w);
                        match kind {
                            0 => (27..32).contains(&x) && (20..25).contains(&y),
                            1 => x < 2 && y < 3,
                            2 => (x == 11 && y == 12) || (x == 38 && y == 30),
                            3 => x == 30 && y == 24,
                            _ => false,
                        }
                    })
                    .collect();
                for (ring, radius) in [(1, 5), (3, 17), (5, 30), (1, 49)] {
                    assert_eq!(
                        best_offset(w, h, ch, &img, &hole, ring, radius),
                        old_best_offset(w, h, ch, &img, &hole, ring, radius),
                        "ch {ch} depth {depth} hole {kind} ring {ring} radius {radius}"
                    );
                }
            }
        }
    }
}

#[test]
fn ties_nonfinite_samples_and_full_holes_keep_the_previous_result() {
    let (w, h, ch) = (40, 35, 3);
    let hole: Vec<_> = (0..w * h).map(|i| (18..22).contains(&(i % w)) && (16..20).contains(&(i / w))).collect();
    for value in [0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let img = vec![value; w * h * ch];
        assert_eq!(best_offset(w, h, ch, &img, &hole, 3, 20), old_best_offset(w, h, ch, &img, &hole, 3, 20));
    }
    let full = vec![true; w * h];
    assert_eq!(best_offset(w, h, ch, &vec![0.5; w * h * ch], &full, 3, 20), None);
}

#[test]
#[ignore = "release comparison on 24 and 36 MP images"]
#[allow(clippy::assertions_on_constants)]
fn proximity_integral_release_comparison() {
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    for (w, h, diameters, radius) in [
        (182usize, 182usize, vec![50usize], 66i32),
        (332, 332, vec![100], 116),
        (632, 632, vec![200], 216),
        (932, 932, vec![300], 316),
        (6000, 4000, vec![32, 256], 512),
        (7200, 5000, vec![32, 256], 512),
    ] {
        for diameter in diameters {
            let ch = 3;
            let img: Vec<_> = (0..w * h * ch)
                .map(|i| {
                    let (x, y, c) = ((i / ch) % w, (i / ch) / w, i % ch);
                    ((x % 53 * 7 + y % 47 * 11 + c * 19) % 251) as f32 / 250.0
                })
                .collect();
            let hole: Vec<_> = (0..w * h)
                .map(|i| {
                    let dx = (i % w) as f64 - (w / 2) as f64;
                    let dy = (i / w) as f64 - (h / 2) as f64;
                    dx * dx + dy * dy < (diameter as f64 * 0.5).powi(2)
                })
                .collect();
            let bbox = hole_bbox(w, h, &hole).unwrap();
            let cropped = HoleIntegral::new(w, &hole, bbox).unwrap();
            let old = old_best_offset(w, h, ch, &img, &hole, (diameter / 8).clamp(3, 16), radius);
            assert_eq!(best_offset(w, h, ch, &img, &hole, (diameter / 8).clamp(3, 16), radius), old);
            let mut old_ms = Vec::new();
            let mut new_ms = Vec::new();
            for run in 0..5 {
                let mut measure = |fast: bool| {
                    let start = Instant::now();
                    let result = if fast {
                        best_offset(w, h, ch, black_box(&img), &hole, (diameter / 8).clamp(3, 16), radius)
                    } else {
                        old_best_offset(w, h, ch, black_box(&img), &hole, (diameter / 8).clamp(3, 16), radius)
                    };
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(result, old);
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
                "PROXIMITY {w}x{h} hole_diameter={diameter} old_ms={old_ms:?} new_ms={new_ms:?} speedup={:.2} full_table_bytes={} cropped_table_bytes={} offset={old:?}",
                old_ms[2] / new_ms[2],
                (w + 1) * (h + 1) * 4,
                cropped.data.len() * 4
            );
        }
    }
}
fn old_best_offset(w: usize, h: usize, ch: usize, img: &[f32], hole: &[bool], ring: usize, max_radius: i32) -> Option<(i32, i32)> {
    let (bx0, by0, bx1, by1) = hole_bbox(w, h, hole)?;
    // No valid translation can move farther than the source image's extent.
    // Bound arbitrary public API inputs before constructing i32 search ranges:
    // usize -> i32 truncation and signed increments otherwise overflow or hang.
    let extent = w.max(h).min((i32::MAX / 4) as usize) as i32;
    let max_radius = max_radius.max(0).min(extent);
    if max_radius == 0 {
        return None;
    }
    let ring = ring.max(1).min(extent as usize) as i32;
    let hs = integral(w, h, hole);
    // Sample points: the hole's ring (dilated minus hole), subsampled for speed.
    let mut pts = Vec::new();
    let (x0, y0) = (bx0 as i32 - ring, by0 as i32 - ring);
    let (x1, y1) = (bx1 as i32 + ring, by1 as i32 + ring);
    for y in y0.max(0)..y1.min(h as i32) {
        for x in x0.max(0)..x1.min(w as i32) {
            if hole[y as usize * w + x as usize] {
                continue;
            }
            if window_count(&hs, w, h, x - ring, y - ring, x + ring + 1, y + ring + 1) > 0 {
                pts.push((x, y));
            }
        }
    }
    let stride = (pts.len() / 2000).max(1);
    let pts: Vec<(i32, i32)> = pts.into_iter().step_by(stride).collect();
    let eval = |dx: i32, dy: i32| -> Option<f32> {
        let (hx0, hy0, hx1, hy1) = (x0 + dx, y0 + dy, x1 + dx, y1 + dy);
        let inside = hx0 >= 0 && hy0 >= 0 && hx1 <= w as i32 && hy1 <= h as i32;
        if (dx, dy) == (0, 0) || !inside || window_count(&hs, w, h, hx0, hy0, hx1, hy1) != 0 {
            return None;
        }
        let mut e = 0.0f32;
        for &(x, y) in &pts {
            let (a, b) = ((y as usize * w + x as usize) * ch, ((y + dy) as usize * w + (x + dx) as usize) * ch);
            for c in 0..ch {
                let d = img[a + c] - img[b + c];
                e += d * d;
            }
        }
        // Mild preference for nearer sources on ties.
        Some(e / pts.len().max(1) as f32 * (1.0 + 1e-3 * ((dx as f32).hypot(dy as f32) / max_radius as f32)))
    };
    // Coarse grid over the search window, then a full-resolution refinement around the winner.
    let step = (max_radius / 24).max(1);
    let mut best: Option<((i32, i32), f32)> = None;
    let consider = |dx: i32, dy: i32, best: &mut Option<((i32, i32), f32)>| {
        if let Some(e) = eval(dx, dy)
            && best.is_none_or(|(_, be)| e < be)
        {
            *best = Some(((dx, dy), e));
        }
    };
    let mut dy = -max_radius;
    while dy <= max_radius {
        let mut dx = -max_radius;
        while dx <= max_radius {
            consider(dx, dy, &mut best);
            dx += step;
        }
        dy += step;
    }
    if step > 1
        && let Some(((bx, by), _)) = best
    {
        for dy in (by - step)..=(by + step) {
            for dx in (bx - step)..=(bx + step) {
                consider(dx, dy, &mut best);
            }
        }
    }
    best.map(|b| b.0)
}
