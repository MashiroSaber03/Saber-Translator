use super::*;
#[test]
#[ignore = "release comparison of actual Quick Selection strokes on 24-36 MP documents"]
#[allow(clippy::assertions_on_constants)]
fn quick_selection_workflow_release_comparison() {
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    for (w, h) in [(6000usize, 4000usize), (7200, 5000)] {
        for (name, size, points, ow, oh) in [
            ("click", 30.0f32, vec![(0.0, 0.0)], 220.0f32, 220.0f32),
            ("long-stroke", 256.0, vec![(-1000.0, 0.0), (1000.0, 0.0)], 2400.0, 1000.0),
            (
                "paint-object",
                180.0,
                vec![(-1000.0, -400.0), (1000.0, -400.0), (1000.0, 0.0), (-1000.0, 0.0), (-1000.0, 400.0), (1000.0, 400.0)],
                2400.0,
                1200.0,
            ),
        ] {
            let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
            let points: Vec<_> = points.iter().map(|&(x, y)| (x + cx, y + cy)).collect();
            let img = RgbImage::from_fn(w, h, |x, y| {
                let inside = (x as f32 - cx).abs() < ow * 0.5 && (y as f32 - cy).abs() < oh * 0.5;
                let noise = ((x * 13 + y * 7) % 31) as f32 / 620.0;
                if inside { [0.8 + noise, 0.1 + noise, 0.1] } else { [0.1, 0.1 + noise, 0.7 + noise] }
            });
            let sampler = ImageSampler { img: &img, origin: (0, 0) };
            let canvas = Rect::new(0, 0, w as i32, h as i32);
            let tune = quick::Tuning::default();
            let old = quick::quick_select_using::<false>(&sampler, canvas, &points, size, &tune).unwrap();
            let new = quick::quick_select_using::<true>(&sampler, canvas, &points, size, &tune).unwrap();
            assert_eq!(new.bbox, old.bbox);
            assert_eq!(new.mask, old.mask);
            let (mut old_ms, mut new_ms) = (Vec::new(), Vec::new());
            for run in 0..3 {
                let mut measure = |fast: bool| {
                    let start = Instant::now();
                    let out = if fast {
                        quick::quick_select_using::<true>(&sampler, canvas, black_box(&points), size, &tune)
                    } else {
                        quick::quick_select_using::<false>(&sampler, canvas, black_box(&points), size, &tune)
                    }
                    .unwrap();
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(out.bbox, old.bbox);
                    assert_eq!(out.mask, old.mask);
                    if fast { new_ms.push(ms) } else { old_ms.push(ms) }
                };
                measure(run % 2 == 0);
                measure(run % 2 != 0);
            }
            old_ms.sort_by(f64::total_cmp);
            new_ms.sort_by(f64::total_cmp);
            println!("QUICK {w}x{h} case={name} brush={size} old_ms={old_ms:?} new_ms={new_ms:?} speedup={:.2} bbox={:?}", old_ms[1] / new_ms[1], old.bbox);
        }
    }
}

#[test]
fn quick_selection_preserves_textures_low_contrast_and_thin_features() {
    let (w, h) = (360usize, 280usize);
    let canvas = Rect::new(0, 0, w as i32, h as i32);
    for kind in 0..6 {
        let img = RgbImage::from_fn(w, h, |x, y| {
            let inside = if kind == 2 {
                (x as f32 - 180.0).hypot(y as f32 - 140.0) < 72.0
            } else if kind == 3 {
                y > 130 && y < 145
            } else {
                x > 60 && x < 300 && y > 60 && y < 220
            };
            let t = ((x * 23 + y * 19) % 101) as f32 / 100.0;
            match kind {
                0 => {
                    if inside {
                        [0.6 + t * 0.1, 0.3, 0.2]
                    } else {
                        [0.4, 0.3 + t * 0.1, 0.2]
                    }
                }
                1 => [0.4 + x as f32 / 1800.0, 0.3 + y as f32 / 1400.0, t * 0.02],
                4 => [t, ((x * 7 + y * 31) % 79) as f32 / 78.0, 0.4],
                5 => [0.4; 3],
                _ => {
                    if inside {
                        [0.8 + t * 0.1, 0.3, 0.2]
                    } else {
                        [0.2, 0.3 + t * 0.1, 0.8]
                    }
                }
            }
        });
        let sampler = ImageSampler { img: &img, origin: (0, 0) };
        for (points, brush) in [(vec![(180.0, 140.0)], 12.0), (vec![(90.0, 140.0), (270.0, 140.0)], 40.0), (vec![(1.0, 1.0), (160.0, 130.0)], 64.0)] {
            for direct_px in [10_000, 300_000] {
                let tune = quick::Tuning { direct_px, ..quick::Tuning::default() };
                let old = quick::quick_select_using::<false>(&sampler, canvas, &points, brush, &tune);
                let new = quick::quick_select_using::<true>(&sampler, canvas, &points, brush, &tune);
                match (old, new) {
                    (Some(a), Some(b)) => {
                        assert_eq!(a.bbox, b.bbox, "kind={kind}");
                        assert_eq!(a.mask, b.mask, "kind={kind} brush={brush} direct={direct_px}");
                    }
                    (None, None) => {}
                    _ => panic!("different empty result kind={kind}"),
                }
            }
        }
    }
}

#[test]
#[ignore = "release comparison through editor raster surfaces on 24-36 MP documents"]
#[allow(clippy::assertions_on_constants)]
fn quick_selection_surface_release_comparison() {
    use photocraft_color::PixelFormat;
    use std::{hint::black_box, time::Instant};
    assert!(!cfg!(debug_assertions), "run with --release");
    for (w, h) in [(6000usize, 4000usize), (7200, 5000)] {
        for format in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F] {
            let (cx, cy) = (w as f32 / 2.0, h as f32 / 2.0);
            let canvas = Rect::new(0, 0, w as i32, h as i32);
            let mut surface = Surface::new(format);
            // Fill the document in bounded row strips, outside the measured strokes.
            for y0 in (0..h).step_by(256) {
                let y1 = (y0 + 256).min(h);
                let data: Vec<_> = (y0..y1)
                    .flat_map(|y| {
                        (0..w).flat_map(move |x| {
                            let noise = ((x * 13 + y * 7) % 31) as f32 / 620.0;
                            if (x as f32 - cx).abs() < 1200.0 && (y as f32 - cy).abs() < 600.0 {
                                [0.8 + noise, 0.1 + noise, 0.1, 1.0]
                            } else {
                                [0.1, 0.1 + noise, 0.7 + noise, 1.0]
                            }
                        })
                    })
                    .collect();
                surface.write_region(Rect::new(0, y0 as i32, w as i32, y1 as i32), &data);
            }
            let sampler = SurfaceSampler(&surface);
            let points = [
                (cx - 1000.0, cy - 400.0),
                (cx + 1000.0, cy - 400.0),
                (cx + 1000.0, cy),
                (cx - 1000.0, cy),
                (cx - 1000.0, cy + 400.0),
                (cx + 1000.0, cy + 400.0),
            ];
            let tune = quick::Tuning::default();
            let old = quick::quick_select_using::<false>(&sampler, canvas, &points, 180.0, &tune).unwrap();
            let new = quick::quick_select_using::<true>(&sampler, canvas, &points, 180.0, &tune).unwrap();
            assert_eq!(new.bbox, old.bbox);
            assert_eq!(new.mask, old.mask);
            let (mut before, mut after) = (Vec::new(), Vec::new());
            for run in 0..5 {
                for fast in [run % 2 == 0, run % 2 != 0] {
                    let start = Instant::now();
                    let out = if fast {
                        quick::quick_select_using::<true>(&sampler, canvas, black_box(&points), 180.0, &tune)
                    } else {
                        quick::quick_select_using::<false>(&sampler, canvas, black_box(&points), 180.0, &tune)
                    }
                    .unwrap();
                    let ms = start.elapsed().as_secs_f64() * 1000.0;
                    assert_eq!(out.bbox, old.bbox);
                    assert_eq!(out.mask, old.mask);
                    if fast { after.push(ms) } else { before.push(ms) }
                }
            }
            before.sort_by(f64::total_cmp);
            after.sort_by(f64::total_cmp);
            println!("SURFACE {w}x{h} format={format:?} old_ms={before:?} new_ms={after:?} speedup={:.2}", before[2] / after[2]);
        }
    }
}
