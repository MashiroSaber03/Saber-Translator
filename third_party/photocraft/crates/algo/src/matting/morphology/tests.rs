use super::*;

#[test]
fn every_small_binary_mask_matches_repeated_octagons() {
    for (w, h) in [(1, 1), (1, 7), (7, 1), (2, 3), (3, 3), (4, 3)] {
        for bits in 0..1usize << (w * h) {
            let src: Vec<_> = (0..w * h).map(|i| ((bits >> i) & 1) as f32).collect();
            for r in [0, 1, 2, 3, 4, 5, 7, 10, 13] {
                for grow in [false, true] {
                    assert_eq!(morph(&src, w, h, r, grow), repeated(&src, w, h, r, grow), "{w}x{h} bits={bits} r={r} grow={grow}");
                }
            }
        }
    }
}

#[test]
fn gray_and_hdr_values_match_at_edges_and_every_radius() {
    for (w, h) in [(2, 29), (29, 2), (37, 29), (1, 51), (51, 1)] {
        for depth in [8, 16, 32] {
            let src: Vec<_> = (0..w * h)
                .map(|i| {
                    let v = ((i * 101 + i / w * 37) % 251) as f32 / 250.0;
                    match depth {
                        8 => (v * 255.0).round() / 255.0,
                        16 => (v * 65535.0).round() / 65535.0,
                        _ => v * 4.0 - 1.0,
                    }
                })
                .collect();
            for r in 0..=65 {
                for grow in [false, true] {
                    assert_eq!(morph(&src, w, h, r, grow), repeated(&src, w, h, r, grow), "{w}x{h} depth={depth} r={r} grow={grow}");
                }
            }
        }
    }
}

#[test]
fn impulses_match_the_closed_octagonal_footprint() {
    let (w, h) = (97usize, 83usize);
    for (sx, sy) in [(0, 0), (0, 41), (48, 41), (96, 82), (5, 79)] {
        for r in 1..=65 {
            for grow in [true, false] {
                let mut src = vec![if grow { 0.0 } else { 1.0 }; w * h];
                src[sy * w + sx] = if grow { 1.0 } else { 0.0 };
                let out = morph(&src, w, h, r, grow);
                let (a, b) = (r / 2 + r % 2, r / 2);
                for y in 0..h {
                    for x in 0..w {
                        let affected = x.abs_diff(sx).saturating_sub(a) + y.abs_diff(sy).saturating_sub(a) <= b;
                        let expected = if affected == grow { 1.0 } else { 0.0 };
                        assert_eq!(out[y * w + x], expected, "seed {sx},{sy} r={r} grow={grow} at {x},{y}");
                    }
                }
            }
        }
    }
}

#[test]
fn exceptional_floats_keep_the_old_visitation_order() {
    let (w, h) = (13, 9);
    for special in [f32::from_bits(0x7fc00001), f32::NEG_INFINITY, f32::INFINITY, -0.0] {
        let src: Vec<_> = (0..w * h).map(|i| if i % 7 == 0 { special } else { (i % 11) as f32 / 10.0 }).collect();
        for r in [1, 3, 4, 17, 64] {
            for grow in [false, true] {
                let a = morph(&src, w, h, r, grow);
                let b = repeated(&src, w, h, r, grow);
                assert_eq!(a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), b.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
            }
        }
    }
    let src = vec![f32::from_bits(0x7fc00002); w * h];
    assert_eq!(morph(&src, w, h, 17, true).iter().map(|v| v.to_bits()).collect::<Vec<_>>(), src.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
}

#[test]
fn invalid_sizes_and_large_radii_are_bounded() {
    assert!(morph(&[], usize::MAX, 2, 3, true).is_empty());
    assert!(morph(&[0.0], 2, 2, 3, true).is_empty());
    assert_eq!(morph(&[], 0, 5, usize::MAX, true), Vec::<f32>::new());
    assert_eq!(morph(&[0.0, 0.5, 1.0, 0.1], 2, 2, usize::MAX, true), vec![1.0; 4]);
    assert_eq!(morph(&[0.0, 0.5, 1.0, 0.1, 42.0], 2, 2, 5, true), vec![1.0, 1.0, 1.0, 1.0, 42.0]);
}

#[test]
fn saturated_radius_does_not_pad_a_thin_axis_to_the_long_axis() {
    let (w, h) = (2, 30001);
    let src: Vec<_> = (0..w * h).map(|i| (i % 251) as f32).collect();
    assert_eq!(morph(&src, w, h, usize::MAX, true), vec![250.0; w * h]);
    assert_eq!(morph(&src, w, h, usize::MAX, false), vec![0.0; w * h]);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn worker_counts_are_bit_identical() {
    let (w, h) = (769, 351);
    let src: Vec<_> = (0..w * h).map(|i| ((i * 101 + i / w * 37) % 251) as f32 / 250.0).collect();
    let mut reference = None;
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
        let result = pool.install(|| morph(&src, w, h, 33, false));
        if let Some(a) = &reference { assert_eq!(a, &result) } else { reference = Some(result) }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn parallel_square_matches_serial_for_partial_blocks_and_thin_images() {
    for (w, h) in [(769, 351), (513, 512), (3, 90001), (90001, 3)] {
        let src: Vec<_> = (0..w * h).map(|i| ((i * 101 + i / w * 37) % 251) as f32 / 250.0).collect();
        for r in [0, 1, 3, 16, 32] {
            for grow in [true, false] {
                assert_eq!(parallel_square(&src, w, h, r, grow), crate::matting::max_square(&src, w, h, r, grow), "{w}x{h} r={r} grow={grow}");
            }
        }
    }
}

#[test]
fn shifted_tiles_match_a_whole_mask_at_canvas_edges() {
    use crate::{
        matting::{RefineParams, refine_buffer, refine_mask, region_reader},
        segment::{ImageSampler, RgbImage},
        selection::Region,
    };
    use photocraft_geom::Rect;
    let (w, h) = (611usize, 317usize);
    let img = RgbImage::from_fn(w, h, |_, _| [0.5; 3]);
    let sampler = ImageSampler { img: &img, origin: (0, 0) };
    let mask: Vec<_> = (0..w * h).map(|i| if (i % w < 63 && i / w < 191) || (i % w > 280 && i / w > 171) { 255 } else { 0 }).collect();
    let rect = Rect::new(0, 0, w as i32, h as i32);
    let region = Region { bbox: rect, mask };
    let input: Vec<_> = region.mask.iter().map(|v| *v as f32 / 255.0).collect();
    for shift in [-100.0, -51.0, 51.0, 100.0] {
        // Constant guide keeps this check independent of image detail.
        let p = RefineParams { radius: 32.0, shift_edge: shift, ..Default::default() };
        let whole = refine_buffer(Some(&img), &input, w, h, &p);
        let tiled = refine_mask(&sampler, &region_reader(&region), rect, rect, &p).unwrap();
        for y in 0..h {
            for x in 0..w {
                assert!((tiled.at(x as i32, y as i32) - whole[y * w + x]).abs() <= 1.5 / 255.0, "shift={shift} at {x},{y}");
            }
        }
    }
}

#[test]
fn bounded_tiles_match_dense_windows_on_partial_cores_and_fractional_values() {
    let (w, h) = (1031usize, 1021usize);
    for kind in 0..2 {
        let src: Vec<_> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                if kind == 0 {
                    x as f32 / 1000.0 + y as f32 / 3000.0 - 0.3
                } else if (x.abs_diff(255) + y.abs_diff(127)) < 35 || x == 768 && y == 896 {
                    0.75
                } else {
                    0.0
                }
            })
            .collect();
        for r in [3, 16, 33, 64] {
            for grow in [false, true] {
                assert_eq!(morph(&src, w, h, r, grow), dense(&src, w, h, r, grow), "kind={kind} r={r} grow={grow}");
            }
        }
    }
    for value in [0.0, 0.5, 1.0, f32::INFINITY, f32::NEG_INFINITY] {
        let mut src = vec![value; w * h];
        src.push(42.0);
        for grow in [true, false] {
            assert_eq!(morph(&src, w, h, 64, grow), src);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn bounded_tiles_match_across_worker_counts() {
    let (w, h) = (1031usize, 1021usize);
    let src: Vec<_> = (0..w * h).map(|i| (i % w) as f32 / 1000.0 + (i / w) as f32 / 3000.0).collect();
    let reference = dense(&src, w, h, 64, false);
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();
        assert_eq!(pool.install(|| morph(&src, w, h, 64, false)), reference);
    }
}

#[test]
#[ignore = "release comparison on tiles, 24 and 36 MP masks"]
#[allow(clippy::assertions_on_constants)]
fn octagonal_morphology_release_comparison() {
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    for (w, h) in [(384usize, 384usize), (6000, 4000), (7200, 5000)] {
        let src: Vec<_> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let dx = x as f64 - w as f64 * 0.51;
                let dy = y as f64 - h as f64 * 0.48;
                let d = (dx * dx + dy * dy).sqrt();
                (0.5 + (w.min(h) as f64 * 0.27 - d) / 7.0).clamp(0.0, 1.0) as f32
            })
            .collect();
        for r in [4, 16, 32, 64] {
            for grow in [true, false] {
                let reference = repeated(&src, w, h, r, grow);
                assert_eq!(morph(&src, w, h, r, grow), reference);
                let (mut old_ms, mut new_ms) = (Vec::new(), Vec::new());
                for run in 0..3 {
                    let mut measure = |fast: bool| {
                        let start = Instant::now();
                        let out = if fast { morph(black_box(&src), w, h, r, grow) } else { repeated(black_box(&src), w, h, r, grow) };
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        assert_eq!(out, reference);
                        if fast { new_ms.push(elapsed) } else { old_ms.push(elapsed) }
                    };
                    measure(run % 2 == 0);
                    measure(run % 2 != 0);
                }
                old_ms.sort_by(f64::total_cmp);
                new_ms.sort_by(f64::total_cmp);
                println!("OCTAGON {w}x{h} r={r} grow={grow} old_ms={old_ms:?} new_ms={new_ms:?} speedup={:.2}", old_ms[1] / new_ms[1]);
            }
        }
    }
}

#[test]
#[ignore = "release comparison of the full tiled Select and Mask CPU path"]
#[allow(clippy::assertions_on_constants)]
fn tiled_refinement_release_comparison() {
    use crate::{
        matting::{RefineParams, refine_mask_with, region_reader},
        segment::{ImageSampler, RgbImage, trim_region},
        selection::Region,
    };
    use photocraft_geom::Rect;
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    for (w, h) in [(6000usize, 4000usize), (7200, 5000)] {
        let image = RgbImage::from_fn(w, h, |x, y| {
            let t = x as f32 / w as f32;
            let u = y as f32 / h as f32;
            [0.2 + 0.5 * t, 0.15 + 0.55 * u, 0.3 + 0.15 * ((x / 17 + y / 23) % 3) as f32]
        });
        let sampler = ImageSampler { img: &image, origin: (0, 0) };
        let canvas = Rect::new(0, 0, w as i32, h as i32);
        let mask: Vec<_> = (0..w * h)
            .map(|i| {
                let dx = (i % w) as f64 - w as f64 * 0.51;
                let dy = (i / w) as f64 - h as f64 * 0.48;
                let d = (dx * dx + dy * dy).sqrt();
                ((0.5 + (w.min(h) as f64 * 0.27 - d) / 7.0).clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect();
        let region = trim_region(Region { bbox: canvas, mask }).unwrap();
        let reader = region_reader(&region);
        let p = RefineParams { radius: 64.0, shift_edge: 100.0, ..Default::default() };
        let old = refine_mask_with(&sampler, &reader, region.bbox, canvas, &p, None, repeated).unwrap();
        let new = refine_mask_with(&sampler, &reader, region.bbox, canvas, &p, None, morph).unwrap();
        assert_eq!(new.bbox, old.bbox);
        assert_eq!(new.mask, old.mask);
        let (mut old_ms, mut new_ms) = (Vec::new(), Vec::new());
        for run in 0..3 {
            let mut measure = |fast: bool| {
                let start = Instant::now();
                let out = if fast {
                    refine_mask_with(&sampler, black_box(&reader), region.bbox, canvas, &p, None, morph)
                } else {
                    refine_mask_with(&sampler, black_box(&reader), region.bbox, canvas, &p, None, repeated)
                }
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(out.bbox, old.bbox);
                assert_eq!(out.mask, old.mask);
                if fast { new_ms.push(elapsed) } else { old_ms.push(elapsed) }
            };
            measure(run % 2 == 0);
            measure(run % 2 != 0);
        }
        old_ms.sort_by(f64::total_cmp);
        new_ms.sort_by(f64::total_cmp);
        println!("REFINE {w}x{h} radius=64 shift=100 old_ms={old_ms:?} new_ms={new_ms:?} speedup={:.2}", old_ms[1] / new_ms[1]);
    }
}

#[test]
#[ignore = "release comparison on a dense fractional 24 MP matte"]
#[allow(clippy::assertions_on_constants)]
fn dense_matte_release_comparison() {
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    let (w, h, r) = (6000usize, 4000usize, 64usize);
    let src: Vec<_> = (0..w * h).map(|i| 0.6 * (i % w) as f32 / w as f32 + 0.4 * (i / w) as f32 / h as f32).collect();
    let reference = repeated(&src, w, h, r, true);
    assert_eq!(morph(&src, w, h, r, true), reference);
    let (mut old_ms, mut new_ms) = (Vec::new(), Vec::new());
    for run in 0..3 {
        let mut measure = |fast: bool| {
            let start = Instant::now();
            let out = if fast { morph(black_box(&src), w, h, r, true) } else { repeated(black_box(&src), w, h, r, true) };
            let elapsed = start.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(out, reference);
            if fast { new_ms.push(elapsed) } else { old_ms.push(elapsed) }
        };
        measure(run % 2 == 0);
        measure(run % 2 != 0);
    }
    old_ms.sort_by(f64::total_cmp);
    new_ms.sort_by(f64::total_cmp);
    println!("DENSE {w}x{h} r={r} grow=true old_ms={old_ms:?} new_ms={new_ms:?} speedup={:.2}", old_ms[1] / new_ms[1]);
}
