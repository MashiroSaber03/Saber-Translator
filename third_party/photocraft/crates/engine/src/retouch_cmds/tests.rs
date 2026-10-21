use super::*;
use photocraft_color::ColorMode;

const DEPTHS: [u64; 3] = [8, 16, 32];

#[test]
fn periodic_clone_large_footprint_applies_opacity_once_and_respects_destination_masks() {
    for depth in DEPTHS {
        for lock in [false, true] {
            let mut s = session(8, 8, depth, "rgb");
            paint_layer(&mut s, |x, _| {
                if x < 4 {
                    [1.0, 0.0, 0.0, 1.0]
                } else if x == 7 {
                    [0.0; 4]
                } else {
                    [1.0; 4]
                }
            });
            s.edit("lock", |doc, active| {
                doc.layer_mut(active.unwrap()).unwrap().locks.transparency = lock;
                Ok(())
            })
            .unwrap();
            s.execute("select.rect", json!({"x":5,"y":0,"width":3,"height":8})).unwrap();
            s.execute("paint.cloneStamp", json!({"points":[[6,4]],"offset":[4,0],"size":32,"hardness":100,"opacity":50,"wrap":true})).unwrap();
            assert_eq!(rgba(&s, 4, 4), [1.0; 4], "selection is evaluated at the wrapped destination");
            let opaque = rgba(&s, 6, 4);
            assert!((opaque[1] - 0.5).abs() <= tol(depth), "overlapping tile copies must not multiply opacity: {opaque:?}");
            let transparent = rgba(&s, 7, 4);
            if lock {
                assert_eq!(transparent[3], 0.0, "locked transparent destination stays transparent");
            } else {
                assert!((transparent[3] - 0.5).abs() <= tol(depth), "unlocked transparent destination receives one stroke: {transparent:?}");
                assert_eq!(transparent[0], 1.0);
            }
        }
    }
}

#[test]
fn periodic_clone_selection_uses_native_cmyk_pixels_across_seam() {
    let mut s = session(64, 32, 8, "cmyk");
    paint_layer(&mut s, |x, _| if x < 32 { [0.0, 0.0, 1.0, 1.0] } else { [1.0, 1.0, 0.0, 1.0] });
    s.execute("select.rect", json!({"x":0,"y":0,"width":4,"height":32})).unwrap();
    s.execute("paint.cloneStamp", json!({"points":[[63,16]],"offset":[-23,0],"size":16,"hardness":100,"wrap":true})).unwrap();
    let (inside, outside) = (rgba(&s, 1, 16), rgba(&s, 5, 16));
    assert!(inside[0] > 0.8 && inside[2] < 0.3, "selected wrapped destination becomes yellow: {inside:?}");
    assert!(outside[2] > outside[0] + 0.2, "unselected neighbour remains blue: {outside:?}");
}

#[test]
fn periodic_clone_projection_is_disjoint_and_matches_pixel_modulo() {
    use std::collections::HashSet;
    let tile = Rect::new(0, 0, 8, 6);
    for rect in [Rect::new(-2, -3, 3, 2), Rect::new(-10, -9, 19, 14), Rect::new(7, 5, 10, 8), Rect::new(i32::MIN, i32::MIN, i32::MIN + 3, i32::MIN + 2)] {
        let expected: HashSet<_> = (rect.y0..rect.y1).flat_map(|y| (rect.x0..rect.x1).map(move |x| (x.rem_euclid(8), y.rem_euclid(6)))).collect();
        let mut actual = HashSet::new();
        for r in periodic::project(rect, tile) {
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    assert!(actual.insert((x, y)), "projected rectangles overlap");
                }
            }
        }
        assert_eq!(actual, expected);
    }
}

#[test]
fn periodic_clone_preview_interpolates_across_transformed_source_seam() {
    use crate::presets::clone_source::Mapping;
    for depth in DEPTHS {
        let mut s = session(32, 24, depth, "rgb");
        paint_layer(&mut s, |x, _| if x == 0 { [1.0, 0.0, 0.0, 1.0] } else { [0.0, 0.0, 1.0, 1.0] });
        let before = s.active().unwrap().doc.clone();
        // A rotated mapping samples halfway between the last and first columns.
        let map = Mapping { source: (31.5, 10.0), anchor: (0.0, 0.0), m: [0.0, 1.0, -1.0, 0.0] };
        let r = Rect::new(0, 0, 1, 1);
        let wrapped = clone_preview_periodic(&s, r, &map, "current", true).unwrap();
        assert_eq!(wrapped.px, [[0.5, 0.0, 0.5, 1.0]]);
        let ordinary = clone_preview(&s, r, &map, "current").unwrap();
        assert_eq!(ordinary.px, [[0.0, 0.0, 1.0, 0.5]]);
        assert_eq!(s.active().unwrap().doc, before);
    }
}

#[test]
fn periodic_clone_destination_wraps_edges_and_undoes_at_every_depth() {
    for depth in DEPTHS {
        for wrap in [false, true] {
            let mut s = session(100, 60, depth, "rgb");
            paint_layer(&mut s, |x, y| if x < 24 && y < 24 { [1.0, 0.0, 0.0, 1.0] } else { [1.0; 4] });
            let before = s.active().unwrap().doc.clone();
            s.execute("paint.cloneStamp", json!({"points":[[98,30]],"source":[10,10],"size":16,"hardness":100,"opacity":50,"wrap":wrap})).unwrap();
            for x in [98, 1] {
                let expected = if x == 98 || wrap { [1.0, 0.5, 0.5, 1.0] } else { [1.0; 4] };
                let actual = rgba(&s, x, 30);
                for (a, b) in actual.into_iter().zip(expected) {
                    assert!((a - b).abs() <= tol(depth), "depth={depth} wrap={wrap} x={x}: {actual:?}");
                }
            }
            assert!(s.undo());
            assert_eq!(s.active().unwrap().doc, before, "one undo restores the whole wrapped stroke");
        }
    }
}

#[test]
fn periodic_clone_live_matches_commit_through_a_corner() {
    for depth in DEPTHS {
        let mut s = session(32, 24, depth, "rgb");
        paint_layer(&mut s, texture);
        let before = s.active().unwrap().doc.clone();
        let mut params = json!({"points":[[30,22]],"source":[10,10],"size":16,"hardness":70,"opacity":45,"flow":60,"wrap":true});
        let mut live = LiveRetouch::begin(&s, "paint.cloneStamp", &params).unwrap();
        let mut points = vec![json!([30, 22])];
        for next in [None, Some([34.0, 26.0]), Some([38.0, 28.0]), Some([30.0, 22.0])] {
            if let Some([x, y]) = next {
                live.push(&[StrokePoint::new(x, y, 1.0)]).unwrap();
                points.push(json!([x, y]));
            }
            params["points"] = json!(points);
            let mut committed = Session::new();
            committed.add_document((*before).clone(), None);
            committed.execute("paint.cloneStamp", params.clone()).unwrap();
            let id = s.active().unwrap().active_layer.unwrap();
            let actual = live.doc.layer(id).unwrap().surface().unwrap().read_region(before.bounds());
            let expected = committed.active().unwrap().doc.layer(id).unwrap().surface().unwrap().read_region(before.bounds());
            assert_eq!(actual, expected, "live versus committed pixels at depth {depth}, points {points:?}");
        }
    }
}

#[test]
fn periodic_clone_source_repeats_both_axes_at_every_depth() {
    for depth in DEPTHS {
        let mut s = session(6, 4, depth, "rgb");
        paint_layer(&mut s, |x, y| [x as f32 / 6.0, y as f32 / 4.0, 0.25, 1.0]);
        let st = s.active().unwrap();
        let surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
        let r = Rect::new(-8, -6, 15, 11);
        for which in [SampleLayers::Current, SampleLayers::CurrentAndBelow, SampleLayers::All] {
            let result = periodic::sample(&st.doc, st.active_layer, surf, which, r, "test").unwrap();
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    let expected = surf.rgba(x.rem_euclid(6), y.rem_euclid(4));
                    let actual = to_rgba(&surf.format(), result.px(x, y));
                    for (a, b) in actual.into_iter().zip(expected) {
                        assert!((a - b).abs() <= tol(depth), "{depth}-bit ({x},{y}): {actual:?} != {expected:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn periodic_clone_source_bounds_allocations_and_accepts_far_coordinates() {
    let s = session(6, 4, 8, "rgb");
    let st = s.active().unwrap();
    let surf = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap();
    assert!(periodic::sample(&st.doc, st.active_layer, surf, SampleLayers::Current, Rect::new(0, 0, 100_000, 100_000), "test").is_err());
    for r in [Rect::new(i32::MIN, i32::MIN, i32::MIN + 3, i32::MIN + 2), Rect::new(i32::MAX - 3, i32::MAX - 2, i32::MAX, i32::MAX)] {
        let result = periodic::sample(&st.doc, st.active_layer, surf, SampleLayers::Current, r, "test").unwrap();
        assert_eq!(result.data.len(), 3 * 2 * surf.channels());
        assert!(result.data.iter().all(|v| *v == 1.0));
    }
}

#[test]
fn proximity_match_reads_original_pixels_for_overlapping_offsets() {
    // Shifting source one pixel to the left overlaps the destination in a
    // left-to-right traversal: reading the output would repeat the first value.
    let img = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let mask = [false, true, true, true, false, false];
    let result = proximity_source_patch(&img, 6, 1, 1, &mask, -1, 0).unwrap();
    assert_eq!(result, vec![10.0, 10.0, 20.0, 30.0, 50.0, 60.0]);
    assert_eq!(img, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], "source remains untouched");
}

#[test]
fn proximity_match_preserves_channels_and_unmasked_pixels() {
    // A two-channel image shifted from the row above.
    let img = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let mask = [false, false, true, false];
    let result = proximity_source_patch(&img, 2, 2, 2, &mask, 0, -1).unwrap();
    assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 7.0, 8.0]);
}

#[test]
fn proximity_match_rejects_invalid_translations_without_panicking() {
    let img = vec![1.0, 2.0, 3.0, 4.0];
    let mask = [true, false, false, false];
    assert!(proximity_source_patch(&img, 2, 2, 1, &mask, -1, 0).is_none());
    assert!(proximity_source_patch(&img, 2, 2, 1, &mask, 0, -1).is_none());
    assert!(proximity_source_patch(&img, 2, 2, 1, &mask, i32::MAX, 0).is_none());
    assert!(proximity_source_patch(&img, 2, 2, 1, &mask[..3], 0, 0).is_none());
    assert!(proximity_source_patch(&img, usize::MAX, 2, 1, &mask, 0, 0).is_none());
}

#[test]
fn clone_preview_matches_source_without_mutating_session() {
    use crate::presets::clone_source::Mapping;
    for depth in DEPTHS {
        let mut s = session(32, 32, depth, "rgb");
        paint_layer(&mut s, |x, y| [x as f32 / 32.0, y as f32 / 32.0, 0.25, 1.0]);
        s.execute("cloneSource.set", json!({"source": [8,8]})).unwrap();
        let revision = s.active().unwrap().revision;
        let slots = s.presets.clone.clone();
        let map = Mapping { source: (8.0, 8.0), anchor: (20.0, 20.0), m: [1.0, 0.0, 0.0, 1.0] };
        let buf = clone_preview(&s, Rect::new(20, 20, 21, 21), &map, "current").unwrap();
        let expected = rgba(&s, 8, 8);
        for (actual, expected) in buf.px[0].iter().zip(expected) {
            assert!((actual - expected).abs() < 0.001);
        }
        assert_eq!(s.active().unwrap().revision, revision);
        assert_eq!(s.presets.clone, slots);
        assert!(clone_preview(&s, Rect::new(0, 0, 2048, 2048), &map, "current").is_err());
        assert!(clone_preview(&s, Rect::new(20, 20, 21, 21), &map, "invalid").is_err());
        let invalid = Mapping { source: (f64::NAN, 0.0), ..map };
        assert!(clone_preview(&s, Rect::new(0, 0, 1, 1), &invalid, "current").is_err());
    }
}

fn session(w: u32, h: u32, depth: u64, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "depth": depth, "mode": mode})).unwrap();
    s
}

/// Straight RGBA of the active layer at a pixel.
fn rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
}

fn paint_layer(s: &mut Session, f: impl Fn(i32, i32) -> [f32; 4]) {
    s.edit("setup", |doc, active| {
        let b = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let mut data = Vec::new();
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                data.extend(photocraft_raster::from_rgba(&fmt, f(x, y)));
            }
        }
        surf.write_region(b, &data);
        Ok(())
    })
    .unwrap();
}

fn tol(depth: u64) -> f32 {
    if depth == 8 { 1.0 / 255.0 + 1e-4 } else { 1e-3 }
}

/// A smooth colour gradient plus deterministic fine noise.
fn texture(x: i32, y: i32) -> [f32; 4] {
    let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) as u32 % 1000) as f32 / 1000.0 * 0.06 - 0.03;
    [0.3 + 0.004 * x as f32 + n, 0.5 + 0.003 * y as f32 + n, 0.45 + n, 1.0]
}

#[test]
fn clone_stamp_copies_exactly_at_all_depths() {
    for depth in DEPTHS {
        let mut s = session(96, 64, depth, "rgb");
        paint_layer(&mut s, |x, y| [x as f32 / 96.0, y as f32 / 64.0, ((x * 7 + y * 3) % 11) as f32 / 11.0, 1.0]);
        let r = s.execute("paint.cloneStamp", json!({"points": [[60, 30], [80, 30]], "source": [20, 20], "size": 10, "hardness": 100})).unwrap();
        assert_eq!(r["offset"], json!([-40, -10]));
        assert!(r["nextSource"].is_null(), "aligned keeps the offset");
        for x in [60, 70, 80] {
            let (got, want) = (rgba(&s, x, 30), rgba(&s, x - 40, 20));
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() <= tol(depth), "depth {depth} x {x}: {got:?} vs {want:?}");
            }
        }
        // Outside the brush untouched.
        let far = rgba(&s, 70, 50);
        assert!((far[0] - 70.0 / 96.0).abs() <= tol(depth) && (far[1] - 50.0 / 64.0).abs() <= tol(depth));
        // One undo step.
        assert!(s.undo());
        assert!((rgba(&s, 70, 30)[1] - 30.0 / 64.0).abs() <= tol(depth));
    }
}

#[test]
fn retouch_rejects_brushes_larger_than_the_raster_budget() {
    let mut s = session(32, 32, 8, "rgb");
    let result = s.execute("paint.dodge", json!({"points": [[10, 10]], "size": 1e30}));
    assert!(result.is_err(), "retouch must reject an oversized brush before footprint allocation");
}

#[test]
fn clone_stamp_samples_pre_stroke_state_and_non_aligned_returns_source() {
    let mut s = session(80, 20, 16, "rgb");
    paint_layer(&mut s, |x, _| [x as f32 / 80.0, 0.0, 0.0, 1.0]);
    // Source 4 px to the right of the stroke start, stroke runs right: without pre-stroke sampling the
    // stroke would re-clone its own output and smear.
    let r = s.execute("paint.cloneStamp", json!({"points": [[10, 10], [50, 10]], "source": [14, 10], "size": 6, "hardness": 100, "aligned": false})).unwrap();
    assert_eq!(r["nextSource"], json!([14.0, 10.0]));
    for x in [20, 30, 45] {
        assert!((rgba(&s, x, 10)[0] - (x + 4) as f32 / 80.0).abs() < 1e-3, "x {x}");
    }
    assert!(s.execute("paint.cloneStamp", json!({"points": [[1, 1]]})).is_err(), "source is required");
    // An explicit offset works too.
    s.execute("paint.cloneStamp", json!({"points": [[70, 10]], "offset": [-60, 0], "size": 4, "hardness": 100})).unwrap();
    assert!((rgba(&s, 70, 10)[0] - 14.0 / 80.0).abs() < 1e-3, "samples the current state (x=10 was cloned from x=14)");
}

#[test]
fn clone_stamp_sample_all_layers_onto_empty_layer_and_opacity() {
    let mut s = session(64, 32, 8, "rgb");
    paint_layer(&mut s, |x, _| if x < 32 { [1.0, 0.0, 0.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] });
    s.execute("layer.new.layer", json!({})).unwrap();
    // Current layer is empty: sampling "current" clones transparency (a no-op) …
    s.execute("paint.cloneStamp", json!({"points": [[48, 16]], "source": [16, 16], "size": 8, "hardness": 100})).unwrap();
    assert_eq!(rgba(&s, 48, 16)[3], 0.0);
    // … sampling all layers clones the visible red.
    s.execute("paint.cloneStamp", json!({"points": [[48, 16]], "source": [16, 16], "size": 8, "hardness": 100, "sampleLayer": "all"})).unwrap();
    let p = rgba(&s, 48, 16);
    assert!(p[0] > 0.99 && p[1] < 0.01 && p[3] > 0.99, "{p:?}");
    // currentAndBelow on the bottom layer ignores layers above.
    s.execute("paint.cloneStamp", json!({"points": [[40, 5]], "source": [8, 5], "size": 4, "hardness": 100, "opacity": 50, "sampleLayer": "currentAndBelow"}))
        .unwrap();
    let p = rgba(&s, 40, 5);
    assert!((p[3] - 0.5).abs() < 0.02, "opacity 50 → half alpha {p:?}");
    assert!(s.execute("paint.cloneStamp", json!({"points": [[1, 1]], "source": [0, 0], "sampleLayer": "nope"})).is_err());
}

#[test]
fn clone_stamp_respects_selection_and_cmyk() {
    let mut s = session(64, 32, 8, "cmyk");
    paint_layer(&mut s, |x, _| if x < 32 { [0.0, 0.0, 1.0, 1.0] } else { [1.0, 1.0, 0.0, 1.0] });
    s.execute("select.rect", json!({"x": 40, "y": 0, "width": 8, "height": 32})).unwrap();
    s.execute("paint.cloneStamp", json!({"points": [[36, 16], [56, 16]], "offset": [-30, 0], "size": 8, "hardness": 100})).unwrap();
    // CMYK pixels display through the CMYK profile, so sRGB blue/yellow come back as their
    // (less saturated) in-gamut CMYK equivalents.
    let (inside, outside) = (rgba(&s, 44, 16), rgba(&s, 52, 16));
    assert!(inside[2] > inside[0] + 0.2, "inside the selection: cloned blue {inside:?}");
    assert!(outside[2] < 0.3 && outside[0] > 0.8, "outside the selection: untouched yellow {outside:?}");
}

#[test]
fn healing_brush_takes_destination_colour_and_source_texture() {
    for depth in [8u64, 32] {
        let mut s = session(128, 64, depth, "rgb");
        // Left: dark blue with strong vertical stripes (the texture); right: flat light orange.
        paint_layer(&mut s, |x, _| {
            if x < 64 {
                let t = if (x / 3) % 2 == 0 { 0.08 } else { -0.08 };
                [0.1 + t + 0.1, 0.15 + t + 0.1, 0.5 + t, 1.0]
            } else {
                [0.9, 0.6, 0.3, 1.0]
            }
        });
        s.execute("paint.healingBrush", json!({"points": [[96, 32]], "source": [32, 32], "size": 30, "hardness": 100})).unwrap();
        // Inside the healed disc: mean ≈ destination colour, variance ≈ source texture.
        let pts: Vec<(i32, i32)> = (-8..=8).flat_map(|dy| (-8..=8).map(move |dx| (96 + dx, 32 + dy))).collect();
        let vals: Vec<[f32; 4]> = pts.iter().map(|&(x, y)| rgba(&s, x, y)).collect();
        let mean: Vec<f32> = (0..3).map(|c| vals.iter().map(|v| v[c]).sum::<f32>() / vals.len() as f32).collect();
        for (c, want) in [0.9f32, 0.6, 0.3].iter().enumerate() {
            assert!((mean[c] - want).abs() < 0.06, "depth {depth} channel {c}: mean {} want {want}", mean[c]);
        }
        let var = vals.iter().map(|v| (v[1] - mean[1]).powi(2)).sum::<f32>() / vals.len() as f32;
        assert!(var > 0.003, "texture carried: variance {var}");
        // Horizontal gradient pattern matches the source stripes (period 6).
        let (a, b) = (rgba(&s, 96, 32)[1] - rgba(&s, 95, 32)[1], rgba(&s, 32, 32)[1] - rgba(&s, 31, 32)[1]);
        assert!((a - b).abs() < 0.03, "gradient {a} vs source {b}");
        // Far away untouched.
        assert!((rgba(&s, 125, 3)[0] - 0.9).abs() <= tol(depth));
    }
}

#[test]
fn spot_healing_removes_blemish_all_types() {
    for (depth, kind) in [(8u64, "contentAware"), (16, "contentAware"), (32, "contentAware"), (8, "proximityMatch"), (16, "createTexture")] {
        let mut s = session(160, 120, depth, "rgb");
        paint_layer(&mut s, |x, y| {
            let d = ((x - 80) as f32).hypot((y - 60) as f32);
            if d < 6.0 { [0.05, 0.03, 0.02, 1.0] } else { texture(x, y) }
        });
        let err = |s: &Session| {
            let mut e = 0.0f32;
            let mut n = 0;
            for y in 52..=68 {
                for x in 72..=88 {
                    let (p, t) = (rgba(s, x, y), texture(x, y));
                    e += (0..3).map(|c| (p[c] - t[c]).powi(2)).sum::<f32>();
                    n += 3;
                }
            }
            (e / n as f32).sqrt()
        };
        let before = err(&s);
        s.execute("paint.spotHealing", json!({"points": [[80, 60]], "size": 20, "hardness": 100, "type": kind})).unwrap();
        let after = err(&s);
        assert!(after < 0.05 && after < before * 0.2, "{kind} depth {depth}: rmse {before} → {after}");
        // Far away untouched.
        let (p, t) = (rgba(&s, 10, 10), texture(10, 10));
        assert!((p[0] - t[0]).abs() <= tol(depth) + 1e-3);
    }
}

#[test]
fn dodge_brightens_midtones_more_than_shadows_and_burn_darkens() {
    for depth in DEPTHS {
        let mut s = session(80, 20, depth, "rgb");
        paint_layer(&mut s, |x, _| if x < 40 { [0.5, 0.5, 0.5, 1.0] } else { [0.08, 0.08, 0.08, 1.0] });
        s.execute("paint.dodge", json!({"points": [[5, 10], [75, 10]], "size": 10, "hardness": 100})).unwrap();
        let (mid, shadow) = (rgba(&s, 20, 10)[0] - 0.5, rgba(&s, 60, 10)[0] - 0.08);
        assert!(mid > 0.05, "depth {depth}: midtone +{mid}");
        assert!(mid > 2.0 * shadow.max(0.0), "depth {depth}: midtone +{mid} shadow +{shadow}");
        s.execute("paint.burn", json!({"points": [[5, 15], [35, 15]], "size": 6, "hardness": 100, "exposure": 80})).unwrap();
        assert!(rgba(&s, 20, 15)[0] < 0.45);
        // Shadows range dodges the dark side more than the grey.
        s.execute("paint.dodge", json!({"points": [[5, 3], [75, 3]], "size": 4, "hardness": 100, "range": "shadows"})).unwrap();
        assert!(rgba(&s, 60, 3)[0] - 0.08 > rgba(&s, 20, 3)[0] - 0.5);
    }
    // Grayscale document too.
    let mut s = session(20, 20, 8, "gray");
    s.execute("paint.burn", json!({"points": [[10, 10]], "size": 6, "hardness": 100, "range": "highlights"})).unwrap();
    assert!(rgba(&s, 10, 10)[0] < 0.9);
    assert_eq!(s.active().unwrap().doc.mode, ColorMode::Grayscale);
}

#[test]
fn dodge_does_not_compound_within_a_stroke() {
    // Going back over the same spot in one stroke tones it no further; a second stroke does.
    let once = |pts: Value| {
        let mut s = session(80, 20, 8, "rgb");
        paint_layer(&mut s, |_, _| [0.5, 0.5, 0.5, 1.0]);
        s.execute("paint.dodge", json!({"points": pts, "size": 10, "hardness": 100})).unwrap();
        (rgba(&s, 40, 10)[0], s)
    };
    let (single, mut s) = once(json!([[5, 10], [75, 10]]));
    let (scrubbed, _) = once(json!([[5, 10], [75, 10], [5, 10], [75, 10]]));
    assert!(single > 0.55 && (single - scrubbed).abs() < 1.0 / 255.0, "{single} {scrubbed}");
    s.execute("paint.dodge", json!({"points": [[5, 10], [75, 10]], "size": 10, "hardness": 100})).unwrap();
    assert!(rgba(&s, 40, 10)[0] > single + 0.02);
}

#[test]
fn sponge_reduces_and_increases_saturation() {
    for depth in DEPTHS {
        let mut s = session(40, 20, depth, "rgb");
        paint_layer(&mut s, |_, _| [0.8, 0.4, 0.3, 1.0]);
        let sat = |p: [f32; 4]| photocraft_algo::retouch::saturation([p[0], p[1], p[2]]);
        let s0 = sat(rgba(&s, 10, 10));
        s.execute("paint.sponge", json!({"points": [[2, 10], [38, 10]], "size": 8, "hardness": 100})).unwrap();
        let s1 = sat(rgba(&s, 10, 10));
        assert!(s1 < s0 * 0.8, "depth {depth}: {s0} → {s1}");
        s.execute("paint.sponge", json!({"points": [[2, 3], [38, 3]], "size": 4, "hardness": 100, "mode": "saturate", "vibrance": false})).unwrap();
        assert!(sat(rgba(&s, 10, 3)) > s0);
    }
}

#[test]
fn blur_reduces_local_variance_and_sharpen_increases_it() {
    for depth in [8u64, 16] {
        let mut s = session(60, 30, depth, "rgb");
        paint_layer(&mut s, |x, y| if (x + y) % 2 == 0 { [0.3, 0.3, 0.3, 1.0] } else { [0.7, 0.7, 0.7, 1.0] });
        let var = |s: &Session, x0: i32| {
            let v: Vec<f32> = (10..20).flat_map(|y| (x0..x0 + 10).map(move |x| (x, y))).map(|(x, y)| rgba(s, x, y)[0]).collect();
            let m = v.iter().sum::<f32>() / v.len() as f32;
            v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32
        };
        let v0 = var(&s, 10);
        s.execute("paint.blur", json!({"points": [[5, 15], [25, 15]], "size": 20, "hardness": 100, "strength": 100})).unwrap();
        assert!(var(&s, 10) < v0 * 0.5, "depth {depth}: {v0} → {}", var(&s, 10));
        assert!((var(&s, 45) - v0).abs() < 1e-3, "outside untouched");
        // Sharpen a low-contrast checkerboard: contrast grows; Protect Detail caps it at the local range.
        let mut s = session(60, 30, depth, "rgb");
        paint_layer(&mut s, |x, y| if (x + y) % 2 == 0 { [0.45, 0.45, 0.45, 1.0] } else { [0.55, 0.55, 0.55, 1.0] });
        let v0 = var(&s, 10);
        s.execute("paint.sharpen", json!({"points": [[5, 15], [25, 15]], "size": 20, "hardness": 100, "strength": 100, "protectDetail": false})).unwrap();
        assert!(var(&s, 10) > v0 * 1.5, "depth {depth}: {v0} → {}", var(&s, 10));
        // Pinned at 25%: Protect Detail clamps per dab, so denser dabs creep past the range.
        s.execute("paint.sharpen", json!({"points": [[35, 15], [55, 15]], "size": 20, "hardness": 100, "strength": 100, "spacing": 25})).unwrap();
        let hi = (10..20).flat_map(|y| (40..50).map(move |x| (x, y))).map(|(x, y)| rgba(&s, x, y)[0]).fold(0.0, f32::max);
        assert!(hi <= 0.55 + tol(depth), "protect detail: {hi}");
    }
}

#[test]
fn smudge_moves_colour_along_the_stroke() {
    for depth in DEPTHS {
        let mut s = session(100, 30, depth, "rgb");
        paint_layer(&mut s, |x, _| if x < 30 { [1.0, 0.0, 0.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] });
        s.execute("paint.smudge", json!({"points": [[20, 15], [70, 15]], "size": 12, "hardness": 100, "strength": 80})).unwrap();
        let g = |x| rgba(&s, x, 15)[1];
        assert!(g(36) < 0.6, "depth {depth}: red dragged right ({})", g(36));
        assert!(g(36) < g(65));
        assert!(rgba(&s, 36, 2)[1] > 0.99, "off-stroke untouched");
        s.execute("tools.setColors", json!({"foreground": "#0000ff"})).unwrap();
        s.execute("paint.smudge", json!({"points": [[80, 25], [95, 25]], "size": 6, "hardness": 100, "fingerPainting": true})).unwrap();
        assert!(rgba(&s, 81, 25)[0] < 0.7, "finger painting lays down the foreground colour");
    }
}

#[test]
fn smudge_uses_the_brush_settings_spacing() {
    // #2140: the Brush Settings spacing (the session brush) drives Smudge dabs when the stroke
    // has no `spacing` param, as it drives the Brush.
    let run = |spacing: f32, p: Value| {
        let mut s = session(100, 30, 8, "rgb");
        paint_layer(&mut s, |x, _| if x < 30 { [1.0, 0.0, 0.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] });
        s.tools.brush.spacing = spacing;
        let mut p = p;
        p["points"] = json!([[20, 15], [90, 15]]);
        s.execute("paint.smudge", p).unwrap();
        (30..95).map(|x| rgba(&s, x, 15)[1]).collect::<Vec<_>>()
    };
    let base = json!({"size": 12, "hardness": 100, "strength": 80});
    let dense = run(0.01, base.clone());
    let sparse = run(10.0, base.clone());
    // 1000% of 12 px puts the next dab 120 px away: only the first dab, behind the edge, lands.
    assert!(sparse.iter().all(|g| *g > 0.99), "1000% spacing: nothing dragged past the edge");
    assert!(dense.first().is_some_and(|g| *g < 0.6), "1% spacing: red dragged right ({:?})", dense.first());
    assert_ne!(dense, run(0.25, base.clone()), "1% and 25% differ");
    // An explicit `spacing` param still wins over the Brush Settings value.
    let mut explicit = base;
    explicit["spacing"] = json!(1000);
    assert_eq!(run(0.01, explicit), sparse);
}

#[test]
fn history_brush_restores_earlier_pixels() {
    for depth in [8u64, 32] {
        let mut s = session(40, 20, depth, "rgb");
        s.execute("paint.stroke", json!({"points": [[0, 10], [40, 10]], "size": 12, "hardness": 1.0, "color": "#ff0000"})).unwrap();
        s.execute("paint.stroke", json!({"points": [[0, 10], [40, 10]], "size": 12, "hardness": 1.0, "color": "#00ff00"})).unwrap();
        assert!(rgba(&s, 20, 10)[1] > 0.99);
        // State 1 = after the red stroke.
        s.execute("paint.historyBrush", json!({"points": [[5, 10], [15, 10]], "size": 6, "hardness": 100, "state": 1})).unwrap();
        let p = rgba(&s, 10, 10);
        assert!(p[0] > 0.99 && p[1] < 0.01, "depth {depth}: {p:?}");
        // Default = the Open snapshot: white.
        s.execute("paint.historyBrush", json!({"points": [[25, 10], [35, 10]], "size": 6, "hardness": 100})).unwrap();
        assert!(rgba(&s, 30, 10).iter().all(|v| *v > 0.99));
        assert!(rgba(&s, 20, 10)[1] > 0.99, "between the dabs still green");
        assert!(s.execute("paint.historyBrush", json!({"points": [[5, 10]], "state": 99})).is_err());
    }
}

#[test]
fn history_brush_reads_a_source_when_its_pixels_were_locked() {
    // Given a white layer locked in history state 1, then unlocked and filled red.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "depth": 8, "mode": "rgb", "background": "white"})).unwrap();
    s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();
    s.execute("layer.lockLayers", json!({"pixels": false})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();

    // When History Brush samples the locked historical pixels.
    let result = s.execute("paint.historyBrush", json!({"points": [[8, 8]], "size": 4, "hardness": 100, "state": 1}));

    // Then it paints white; the historical lock limits edits, not reads.
    assert!(result.is_ok(), "{result:?}");
    assert!(rgba(&s, 8, 8).iter().all(|v| *v > 0.99), "{:?}", rgba(&s, 8, 8));
}

#[test]
fn history_brush_refuses_a_locked_destination_when_source_is_unlocked() {
    // Given an unlocked white source and a red, currently locked destination.
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16, "depth": 8, "mode": "rgb", "background": "white"})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("layer.lockLayers", json!({"pixels": true})).unwrap();

    // When History Brush samples the unlocked source and attempts to paint the locked destination.
    let result = s.execute("paint.historyBrush", json!({"points": [[8, 8]], "size": 4, "hardness": 100, "state": 0}));

    // Then the destination lock still rejects the edit and preserves red pixels.
    assert!(result.is_err());
    let pixel = rgba(&s, 8, 8);
    assert!(pixel[0] > 0.99 && pixel[1] < 0.01, "{pixel:?}");
}

#[test]
fn history_brush_errors_when_layer_is_new() {
    let mut s = session(20, 20, 8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    assert!(s.execute("paint.historyBrush", json!({"points": [[5, 5]], "state": 0})).is_err());
}

#[test]
fn retouch_commands_are_registered_and_need_a_pixel_layer() {
    let ids = [
        "paint.cloneStamp",
        "paint.patternStamp",
        "paint.healingBrush",
        "paint.spotHealing",
        "paint.dodge",
        "paint.burn",
        "paint.sponge",
        "paint.blur",
        "paint.sharpen",
        "paint.smudge",
        "paint.historyBrush",
        "paint.remove",
    ];
    for id in ids {
        let spec = crate::commands::find(id).unwrap_or_else(|| panic!("{id} missing"));
        assert!(spec.params.contains("\"points\"") && spec.params.contains("spacing"), "{id}");
        assert!(!Session::new().is_enabled(id));
    }
    let mut s = session(10, 10, 8, "rgb");
    assert!(s.execute("paint.dodge", json!({"points": []})).is_err());
    assert!(s.execute("paint.dodge", json!({"points": [[1, 1]], "range": "bogus"})).is_err());
    assert!(s.execute("paint.spotHealing", json!({"points": [[1, 1]], "type": "bogus"})).is_err());
}

// ---------------------------------------------------------------------------------------------
// Paint target (#207): mask, alpha channel, Quick Mask
// ---------------------------------------------------------------------------------------------

const RETOUCH_IDS: [&str; 12] = [
    "paint.cloneStamp",
    "paint.patternStamp",
    "paint.healingBrush",
    "paint.spotHealing",
    "paint.dodge",
    "paint.burn",
    "paint.sponge",
    "paint.blur",
    "paint.sharpen",
    "paint.smudge",
    "paint.historyBrush",
    "paint.remove",
];

/// Grey noise in 0.3..0.7 (no period along the clone offset used below).
fn gray_noise(x: i32, y: i32) -> f32 {
    0.3 + ((x * 7 + y * 13).rem_euclid(10)) as f32 * 0.04
}

fn texture_surface(surf: &mut Surface, b: Rect) {
    let mut data = Vec::new();
    for y in b.y0..b.y1 {
        for x in b.x0..b.x1 {
            data.push(gray_noise(x, y));
        }
    }
    surf.write_region(b, &data);
}

fn active_layer(s: &Session) -> &photocraft_doc::Layer {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap()
}

fn read_all(surf: &Surface, b: Rect) -> Vec<f32> {
    surf.read_region(b)
}

/// A textured pixel layer with a textured mask and a textured alpha channel. Returns the history
/// index of the state before the mask and channel were textured (for the History Brush).
fn targeted_session() -> (Session, usize) {
    let mut s = session(60, 40, 8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    paint_layer(&mut s, texture);
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    s.execute("channel.new", json!({"fill": "white"})).unwrap();
    let before = s.active().unwrap().history.past_len();
    s.edit("texture", |doc, active| {
        let b = doc.bounds();
        texture_surface(&mut doc.layer_mut(active.unwrap()).unwrap().mask.as_mut().unwrap().surface, b);
        texture_surface(&mut doc.channels[0].surface, b);
        Ok(())
    })
    .unwrap();
    (s, before)
}

fn retouch_params(id: &str, history_state: usize, target: Value) -> Value {
    let mut p = json!({"points": [[20, 20], [40, 20]], "size": 12, "hardness": 100, "strength": 100, "exposure": 100, "target": target});
    match id {
        "paint.cloneStamp" | "paint.healingBrush" => p["offset"] = json!([-13, 3]),
        "paint.historyBrush" => p["state"] = json!(history_state),
        "paint.sponge" => p["mode"] = json!("saturate"),
        _ => {}
    }
    p
}

/// Tools whose stroke must visibly change grey noise (Sponge has no colour to change on grey,
/// healing a noise texture with itself may land on the same values, and Remove completes this
/// periodic noise exactly).
fn changes_gray(id: &str) -> bool {
    !matches!(id, "paint.sponge" | "paint.healingBrush" | "paint.spotHealing" | "paint.remove")
}

#[test]
fn retouch_tools_paint_only_the_targeted_mask() {
    for id in RETOUCH_IDS {
        let (mut s, state) = targeted_session();
        let b = s.active().unwrap().doc.bounds();
        let px0 = read_all(active_layer(&s).surface().unwrap(), b);
        let mask0 = read_all(&active_layer(&s).mask.as_ref().unwrap().surface, b);
        let ch0 = read_all(&s.active().unwrap().doc.channels[0].surface, b);
        s.execute(id, retouch_params(id, state, json!("mask"))).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(read_all(active_layer(&s).surface().unwrap(), b), px0, "{id}: layer pixels untouched");
        assert_eq!(read_all(&s.active().unwrap().doc.channels[0].surface, b), ch0, "{id}: channel untouched");
        if changes_gray(id) {
            assert_ne!(read_all(&active_layer(&s).mask.as_ref().unwrap().surface, b), mask0, "{id}: the mask changed");
        }
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(read_all(&active_layer(&s).mask.as_ref().unwrap().surface, b), mask0, "{id}: one undo step");
    }
}

#[test]
fn retouch_tools_paint_only_the_targeted_alpha_channel() {
    for id in RETOUCH_IDS {
        let (mut s, state) = targeted_session();
        let b = s.active().unwrap().doc.bounds();
        let px0 = read_all(active_layer(&s).surface().unwrap(), b);
        let mask0 = read_all(&active_layer(&s).mask.as_ref().unwrap().surface, b);
        let ch0 = read_all(&s.active().unwrap().doc.channels[0].surface, b);
        s.execute(id, retouch_params(id, state, json!({"channel": 0}))).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(read_all(active_layer(&s).surface().unwrap(), b), px0, "{id}: layer pixels untouched");
        assert_eq!(read_all(&active_layer(&s).mask.as_ref().unwrap().surface, b), mask0, "{id}: mask untouched");
        if changes_gray(id) {
            assert_ne!(read_all(&s.active().unwrap().doc.channels[0].surface, b), ch0, "{id}: the channel changed");
        }
    }
}

#[test]
fn retouch_tools_follow_the_channels_panel_target() {
    // No "target" param: the targeted alpha channel (Channels panel) is painted, like the Brush.
    for id in RETOUCH_IDS.into_iter().filter(|id| changes_gray(id)) {
        let (mut s, state) = targeted_session();
        let b = s.active().unwrap().doc.bounds();
        let px0 = read_all(active_layer(&s).surface().unwrap(), b);
        let ch0 = read_all(&s.active().unwrap().doc.channels[0].surface, b);
        s.active_mut().unwrap().channel_view.target = crate::channel_cmds::ChannelTarget::Alpha(0);
        let mut p = retouch_params(id, state, Value::Null);
        p.as_object_mut().unwrap().remove("target");
        assert!(s.is_enabled(id), "{id}");
        s.execute(id, p).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert_eq!(read_all(active_layer(&s).surface().unwrap(), b), px0, "{id}: layer pixels untouched");
        assert_ne!(read_all(&s.active().unwrap().doc.channels[0].surface, b), ch0, "{id}: the channel changed");
    }
}

#[test]
fn retouch_tools_paint_the_quick_mask() {
    let (mut s, _) = targeted_session();
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 30, "height": 40})).unwrap();
    s.execute("select.editInQuickMaskMode", json!({"on": true})).unwrap();
    let b = s.active().unwrap().doc.bounds();
    let px0 = read_all(active_layer(&s).surface().unwrap(), b);
    let q0 = read_all(&s.active().unwrap().doc.quick_mask.as_ref().unwrap().surface, b);
    s.execute("paint.smudge", json!({"points": [[20, 20], [45, 20]], "size": 12, "hardness": 100, "strength": 90})).unwrap();
    assert_eq!(read_all(active_layer(&s).surface().unwrap(), b), px0, "layer pixels untouched");
    assert_ne!(read_all(&s.active().unwrap().doc.quick_mask.as_ref().unwrap().surface, b), q0, "the Quick Mask changed");
}

#[test]
fn retouch_targets_fail_gracefully() {
    let mut s = session(30, 30, 8, "rgb");
    for id in RETOUCH_IDS {
        let p = |t: Value| json!({"points": [[5, 5], [10, 5]], "size": 6, "state": 0, "offset": [1, 1], "target": t});
        assert!(s.execute(id, p(json!("mask"))).is_err(), "{id}: no mask");
        assert!(s.execute(id, p(json!({"channel": 7}))).is_err(), "{id}: no channel 7");
        assert!(s.execute(id, p(json!("quickMask"))).is_err(), "{id}: not in Quick Mask mode");
        let _ = s.execute(id, p(json!(42)));
        let _ = s.execute(id, p(json!({"channel": "x"})));
        let _ = s.execute(id, json!({"points": [[5, 5]], "sampleAllLayers": "yes"}));
    }
}

// ---------------------------------------------------------------------------------------------
// Sample All Layers (#207)
// ---------------------------------------------------------------------------------------------

/// A Background with `bottom` and an empty layer above it (active).
fn two_layers(bottom: impl Fn(i32, i32) -> [f32; 4]) -> Session {
    let mut s = session(100, 30, 8, "rgb");
    paint_layer(&mut s, bottom);
    s.execute("layer.new.layer", json!({})).unwrap();
    s
}

#[test]
fn sample_all_layers_blur_and_sharpen_work_on_an_empty_layer() {
    let checker = |x: i32, y: i32| if (x + y) % 2 == 0 { [0.4, 0.4, 0.4, 1.0] } else { [0.6, 0.6, 0.6, 1.0] };
    for (id, extra) in [("paint.blur", json!({})), ("paint.sharpen", json!({"protectDetail": false}))] {
        let stroke = |all: bool| {
            let mut p = json!({"points": [[10, 15], [40, 15]], "size": 16, "hardness": 100, "strength": 100, "sampleAllLayers": all});
            p.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            p
        };
        // Off: the empty layer has nothing to blur.
        let mut s = two_layers(checker);
        s.execute(id, stroke(false)).unwrap();
        assert_eq!(rgba(&s, 25, 15)[3], 0.0, "{id}: without Sample All Layers the layer stays empty");
        // On: the layer picks up the blurred / sharpened picture.
        let mut s = two_layers(checker);
        s.execute(id, stroke(true)).unwrap();
        let (a, b) = (rgba(&s, 25, 15), rgba(&s, 26, 15));
        assert!(a[3] > 0.99, "{id}: {a:?}");
        let contrast = (a[0] - b[0]).abs();
        if id == "paint.blur" {
            assert!(contrast < 0.1, "{id}: blurred ({contrast})");
        } else {
            assert!(contrast > 0.25, "{id}: sharpened ({contrast})");
        }
        assert_eq!(rgba(&s, 80, 15)[3], 0.0, "{id}: outside the stroke untouched");
    }
}

#[test]
fn sample_all_layers_smudge_drags_colour_from_below() {
    let edge = |x: i32, _: i32| if x < 30 { [1.0, 0.0, 0.0, 1.0] } else { [1.0, 1.0, 1.0, 1.0] };
    let stroke = |all: bool| json!({"points": [[20, 15], [70, 15]], "size": 12, "hardness": 100, "strength": 80, "sampleAllLayers": all});
    let mut s = two_layers(edge);
    s.execute("paint.smudge", stroke(false)).unwrap();
    assert_eq!(rgba(&s, 40, 15)[3], 0.0, "without Sample All Layers the layer stays empty");
    let mut s = two_layers(edge);
    s.execute("paint.smudge", stroke(true)).unwrap();
    let p = rgba(&s, 36, 15);
    assert!(p[3] > 0.99 && p[1] < 0.6, "red dragged onto the empty layer: {p:?}");
    // A mask target ignores Sample All Layers (nothing to sample from other layers).
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    s.execute("paint.smudge", json!({"points": [[20, 25], [70, 25]], "size": 6, "sampleAllLayers": true, "target": "mask"})).unwrap();
}

#[test]
fn sample_all_layers_spot_healing_heals_onto_an_empty_layer() {
    // #731: the blemish below is healed onto the empty layer.
    for depth in DEPTHS {
        let heal = |all: bool| {
            let mut s = session(100, 30, depth, "rgb");
            paint_layer(&mut s, blemished(46));
            s.execute("layer.new.layer", json!({})).unwrap();
            s.execute("paint.spotHealing", json!({"points": [[50, 24]], "size": 12, "hardness": 100, "sampleAllLayers": all})).unwrap();
            s
        };
        assert_eq!(rgba(&heal(false), 50, 24)[3], 0.0, "depth {depth}: without Sample All Layers the layer stays empty");
        let s = heal(true);
        let (p, t) = (rgba(&s, 50, 24), texture(50, 24));
        assert!(p[3] > 0.99 && (p[0] - t[0]).abs() < 0.1 && (p[1] - t[1]).abs() < 0.1, "depth {depth}: healed, not red: {p:?} want {t:?}");
        assert_eq!(rgba(&s, 80, 15)[3], 0.0, "depth {depth}: outside the stroke untouched");
    }
}

#[test]
fn smudge_across_a_transparent_edge_has_no_dark_fringe() {
    for depth in DEPTHS {
        let mut s = session(100, 30, depth, "rgb");
        s.execute("layer.new.layer", json!({})).unwrap();
        paint_layer(&mut s, |x, _| if x < 30 { [1.0, 0.8, 0.0, 1.0] } else { [0.0, 0.0, 0.0, 0.0] });
        s.execute("paint.smudge", json!({"points": [[20, 15], [70, 15]], "size": 12, "hardness": 50, "strength": 80})).unwrap();
        let mut n = 0;
        for x in 30..80 {
            let p = rgba(&s, x, 15);
            if p[3] > 0.02 {
                n += 1;
                assert!(p[0] > 0.97 && (p[1] - 0.8).abs() < 0.03, "depth {depth} x {x}: {p:?}");
            }
        }
        assert!(n > 5, "depth {depth}: colour dragged into the transparent area");
    }
}

/// `texture` with a solid red blemish over `x0..x0+8`, `y 20..28`.
fn blemished(x0: i32) -> impl Fn(i32, i32) -> [f32; 4] {
    move |x, y| if (x0..x0 + 8).contains(&x) && (20..28).contains(&y) { [1.0, 0.0, 0.0, 1.0] } else { texture(x, y) }
}

/// Every pixel of `rect` (x0, y0, x1, y1) is within `t` of the clean texture.
fn assert_repaired(s: &Session, rect: (i32, i32, i32, i32), t: f32, what: &str) {
    for y in rect.1..rect.3 {
        for x in rect.0..rect.2 {
            let (got, want) = (rgba(s, x, y), texture(x, y));
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() < t, "{what} at {x},{y} channel {c}: {got:?} want {want:?}");
            }
        }
    }
}

#[test]
fn patch_source_repairs_the_selection_with_texture_from_the_drag_target() {
    for depth in DEPTHS {
        let mut s = session(96, 48, depth, "rgb");
        paint_layer(&mut s, blemished(40));
        s.execute("select.rect", json!({"x": 36, "y": 16, "width": 16, "height": 16})).unwrap();
        let r = s.execute("paint.patch", json!({"offset": [-30, 2]})).unwrap();
        assert_eq!(r["offset"], json!([-30, 2]));
        // Gradient plus noise: the membrane fits the source's colour to the surroundings, the noise
        // comes from the source, so the repair is close to (not exactly) the clean texture.
        assert_repaired(&s, (36, 16, 52, 32), 0.07, &format!("depth {depth}"));
        // Outside the selection nothing changed, including the sampled area.
        for (x, y) in [(35, 20), (52, 25), (10, 20), (44, 40)] {
            let (got, want) = (rgba(&s, x, y), texture(x, y));
            assert!((got[0] - want[0]).abs() <= tol(depth), "depth {depth} untouched {x},{y}: {got:?}");
        }
        // One undo step restores the blemish.
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(rgba(&s, 44, 24)[1], 0.0, "depth {depth}: undo brings the blemish back");
    }
}

#[test]
fn patch_content_aware_copies_the_dragged_to_content_into_the_selection() {
    // Photoshop's Patch: Content-Aware (measured on 25.4): only the selection changes; its middle
    // is the dragged-to content (Color 0 keeps its level), the edge band blends.
    for depth in [8, 16] {
        let mut s = session(128, 64, depth, "rgb");
        paint_layer(&mut s, blemished(40));
        s.execute("select.rect", json!({"x": 32, "y": 12, "width": 24, "height": 24})).unwrap();
        let r = s.execute("paint.patch", json!({"offset": [-30, 2], "contentAware": true, "structure": 7, "color": 0})).unwrap();
        assert_eq!(r["contentAware"], json!(true));
        for (x, y) in [(42, 22), (44, 24), (46, 26)] {
            let (got, want) = (rgba(&s, x, y), texture(x - 30, y + 2));
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() <= tol(depth), "depth {depth}: middle copied at {x},{y}: {got:?} vs {want:?}");
            }
        }
        for (x, y) in [(31, 20), (56, 25), (12, 20), (44, 40)] {
            let (got, want) = (rgba(&s, x, y), texture(x, y));
            assert!((got[0] - want[0]).abs() <= tol(depth), "depth {depth}: untouched {x},{y}: {got:?}");
        }
        assert_eq!(s.active().unwrap().history.undo_label(), Some("Patch (Content-Aware)"));
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(rgba(&s, 44, 24)[1], 0.0, "depth {depth}: undo brings the blemish back");
    }
    // Bad options are refused before any work.
    let mut s = session(96, 48, 8, "rgb");
    paint_layer(&mut s, blemished(40));
    s.execute("select.rect", json!({"x": 36, "y": 16, "width": 16, "height": 16})).unwrap();
    for p in [json!({"offset": [-30, 0], "contentAware": true, "structure": 9}), json!({"offset": [-30, 0], "contentAware": true, "color": -1})] {
        assert!(s.execute("paint.patch", p.clone()).is_err(), "{p}");
    }
}

#[test]
fn patch_destination_repairs_the_drag_target_from_the_selection() {
    let mut s = session(96, 48, 16, "rgb");
    paint_layer(&mut s, blemished(40));
    // Select clean texture and drag it onto the blemish.
    s.execute("select.rect", json!({"x": 6, "y": 16, "width": 16, "height": 16})).unwrap();
    s.execute("paint.patch", json!({"offset": [30, 0], "mode": "destination"})).unwrap();
    assert_repaired(&s, (36, 16, 52, 32), 0.07, "destination");
    // The selected (source) pixels stay as they were.
    let (got, want) = (rgba(&s, 10, 20), texture(10, 20));
    assert!((got[1] - want[1]).abs() < 1e-3, "source untouched: {got:?}");
}

#[test]
fn patch_at_the_canvas_edge_stays_opaque_and_matches() {
    // The blemish touches the left edge; the solve must not pull in the empty pixels beyond it.
    let mut s = session(64, 48, 8, "rgb");
    paint_layer(&mut s, blemished(0));
    s.execute("select.rect", json!({"x": 0, "y": 16, "width": 12, "height": 16})).unwrap();
    s.execute("paint.patch", json!({"offset": [30, 0]})).unwrap();
    assert_repaired(&s, (0, 16, 12, 32), 0.07, "edge");
}

#[test]
fn patch_follows_a_feathered_selection_and_a_mask_target() {
    let mut s = session(96, 48, 8, "rgb");
    paint_layer(&mut s, blemished(40));
    s.execute("select.rect", json!({"x": 36, "y": 16, "width": 16, "height": 16, "ellipse": true, "feather": 2})).unwrap();
    s.execute("paint.patch", json!({"offset": [-30, 0]})).unwrap();
    // The blemish (well inside the ellipse) is gone, a corner outside the ellipse is untouched.
    assert_repaired(&s, (42, 22, 46, 26), 0.07, "feathered");
    let (got, want) = (rgba(&s, 36, 16), texture(36, 16));
    assert!((got[0] - want[0]).abs() <= tol(8), "corner outside the ellipse: {got:?}");
    // Patching the layer mask works like any retouch target.
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    s.execute("paint.patch", json!({"offset": [-30, 0], "target": "mask"})).unwrap();
}

#[test]
fn patch_fails_gracefully() {
    let mut s = session(64, 48, 8, "rgb");
    assert!(s.execute("paint.patch", json!({"offset": [10, 0]})).is_err(), "no selection");
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
    for p in [
        json!({}),
        json!({"offset": [10]}),
        json!({"offset": "far"}),
        json!({"offset": [1e300, 0]}),
        json!({"offset": [0, 0]}),
        json!({"offset": [10, 0], "mode": "bogus"}),
        json!({"offset": [60, 0]}),
        json!({"offset": [0, -30]}),
        json!({"offset": [10, 0], "target": "mask"}),
        json!({"offset": [10, 0], "target": {"channel": 7}}),
        json!({"offset": [10, 0], "layer": u64::MAX}),
    ] {
        assert!(s.execute("paint.patch", p.clone()).is_err(), "{p}");
    }
    assert!(s.execute("paint.patch", json!({"offset": [10, 0]})).is_ok());
}

#[test]
fn patch_preview_matches_the_command_and_its_coarse_solve_is_close() {
    let mut s = session(160, 96, 16, "rgb");
    paint_layer(&mut s, blemished(100));
    s.execute("select.rect", json!({"x": 80, "y": 4, "width": 60, "height": 60, "ellipse": true, "feather": 3})).unwrap();
    let p = json!({"offset": [-70, 20]});
    let rev = s.active().unwrap().revision;
    let id = s.active().unwrap().active_layer.unwrap();
    let layer = |d: &Document, x, y| d.layer(id).unwrap().surface().unwrap().rgba(x, y);
    let (exact, dmg) = patch_preview(&s, &p, u64::MAX, 1).unwrap();
    let (coarse, _) = patch_preview(&s, &p, 64, 1).unwrap();
    assert!(dmg.contains(104, 24) && !dmg.contains(20, 24), "{dmg:?}");
    assert_eq!(s.active().unwrap().revision, rev, "the preview records nothing");
    s.execute("paint.patch", p).unwrap();
    let committed = s.active().unwrap().doc.clone();
    let mut worst = 0.0f32;
    for y in 0..96 {
        for x in 0..160 {
            let (e, c, d) = (layer(&exact, x, y), layer(&coarse, x, y), layer(&committed, x, y));
            for k in 0..4 {
                assert!((e[k] - d[k]).abs() < 1e-6, "step 1 is the command at {x},{y}");
                worst = worst.max((c[k] - d[k]).abs());
            }
        }
    }
    // 64 cells over a ~64² patch: an 8× coarser membrane. Measured worst pixel 0.022 (under six
    // 8-bit steps), all in the colour fit: the texture is exact.
    assert!(worst < 0.03, "coarse preview off by {worst}");
    assert!(patch_preview(&s, &json!({"offset": [500, 0]}), 64, 1).is_err());
}

// ---------------------------------------------------------------------------------------------
// Content-Aware Move
// ---------------------------------------------------------------------------------------------

/// `texture` with a solid blue object over `x0..x0+12`, `y0..y0+12`.
fn with_object(x0: i32, y0: i32) -> impl Fn(i32, i32) -> [f32; 4] {
    move |x, y| if (x0..x0 + 12).contains(&x) && (y0..y0 + 12).contains(&y) { [0.05, 0.1, 0.9, 1.0] } else { texture(x, y) }
}

fn is_object(px: [f32; 4]) -> bool {
    px[2] > 0.8 && px[0] < 0.2
}

fn selected(s: &Session, x: i32, y: i32) -> bool {
    s.active().unwrap().doc.selection.as_ref().is_some_and(|sel| sel.sample_channel(x, y, 0) > 0.0)
}

#[test]
fn content_aware_move_moves_the_object_and_fills_its_place_at_all_depths() {
    for depth in DEPTHS {
        let mut s = session(128, 64, depth, "rgb");
        paint_layer(&mut s, with_object(20, 26));
        s.execute("select.rect", json!({"x": 16, "y": 22, "width": 20, "height": 20})).unwrap();
        let r = s.execute("paint.contentAwareMove", json!({"offset": [70, 0]})).unwrap();
        assert_eq!(r["offset"], json!([70, 0]));
        assert_eq!(r["mode"], "move");
        // The object's core is at the new place; its old place holds texture again.
        for (x, y) in [(93, 32), (95, 30), (98, 35)] {
            assert!(is_object(rgba(&s, x, y)), "depth {depth}: object at {x},{y}: {:?}", rgba(&s, x, y));
        }
        for y in 26..38 {
            for x in 20..32 {
                assert!(!is_object(rgba(&s, x, y)), "depth {depth}: old place {x},{y} still shows the object");
            }
        }
        // Far away nothing changed.
        for (x, y) in [(2, 2), (60, 60), (125, 5)] {
            let (got, want) = (rgba(&s, x, y), texture(x, y));
            assert!((got[0] - want[0]).abs() <= tol(depth), "depth {depth}: {x},{y} changed: {got:?}");
        }
        // The selection followed the content.
        assert!(selected(&s, 96, 32) && !selected(&s, 26, 32), "depth {depth}: selection moved");
        // One step: undo restores pixels and selection, redo brings the move back.
        s.execute("edit.undo", json!({})).unwrap();
        assert!(is_object(rgba(&s, 25, 31)) && !is_object(rgba(&s, 95, 31)), "depth {depth}: undo restores the object");
        assert!(selected(&s, 26, 32) && !selected(&s, 96, 32), "depth {depth}: undo restores the selection");
        s.execute("edit.redo", json!({})).unwrap();
        assert!(is_object(rgba(&s, 95, 31)) && !is_object(rgba(&s, 25, 31)), "depth {depth}: redo");
    }
}

#[test]
fn content_aware_move_preview_shows_the_result_without_touching_the_session() {
    use crate::jobs::JobCtx;
    let mut s = session(128, 64, 8, "rgb");
    paint_layer(&mut s, with_object(20, 26));
    s.execute("select.rect", json!({"x": 16, "y": 22, "width": 20, "height": 20})).unwrap();
    let before = s.active().unwrap().revision;
    let p = json!({"offset": [70, 0]});
    let (doc, damage) = content_aware_move_preview(&s, &p).unwrap().run(&JobCtx::new()).unwrap();
    let px = |d: &Document, x, y| d.layers[0].surface().unwrap().rgba(x, y);
    assert!(is_object(px(&doc, 95, 31)) && !is_object(px(&doc, 25, 31)), "the preview shows the move");
    assert!(damage.contains(95, 31) && damage.contains(25, 31), "{damage:?}");
    // The session is untouched.
    assert!(is_object(rgba(&s, 25, 31)) && !is_object(rgba(&s, 95, 31)));
    assert_eq!(s.active().unwrap().revision, before);
    // A cancelled preview stops; a selection past the preview's size and bad params are errors.
    let ctx = JobCtx::new();
    ctx.cancel();
    assert!(matches!(content_aware_move_preview(&s, &p).unwrap().run(&ctx), Err(EngineError::Cancelled)));
    assert!(content_aware_move_preview(&s, &json!({"offset": [0, 0]})).is_err());
    assert!(content_aware_move_preview(&s, &json!({"offset": [70, 0], "rotate": "x"})).is_err());
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 300, "height": 300})).unwrap();
    assert!(content_aware_move_preview(&s, &json!({"offset": [1, 0]})).is_err());
}

#[test]
fn content_aware_move_transform_on_drop_scales_and_rotates_the_content() {
    // A horizontal 16×6 bar in a 36×36 selection, dropped 80 px right and turned 90° with 125%
    // width: it lands upright, 20 px tall, and the blend works around the turned content.
    let bar = |x: i32, y: i32| if (20..36).contains(&x) && (30..36).contains(&y) { [0.05, 0.1, 0.9, 1.0] } else { texture(x, y) };
    for depth in [8, 32] {
        let mut s = session(160, 64, depth, "rgb");
        paint_layer(&mut s, bar);
        s.execute("select.rect", json!({"x": 10, "y": 15, "width": 36, "height": 36})).unwrap();
        let r = s.execute("paint.contentAwareMove", json!({"offset": [80, 0], "rotate": 90, "scaleX": 125})).unwrap();
        assert_eq!(r["transformed"], json!(true));
        for (x, y) in [(108, 24), (108, 33), (108, 42), (106, 30)] {
            assert!(is_object(rgba(&s, x, y)), "depth {depth}: the turned, stretched bar covers {x},{y}: {:?}", rgba(&s, x, y));
        }
        for (x, y) in [(101, 33), (115, 33)] {
            assert!(!is_object(rgba(&s, x, y)), "depth {depth}: no unturned bar at {x},{y}");
        }
        assert!(!is_object(rgba(&s, 28, 33)), "depth {depth}: the old place is filled");
        assert!(selected(&s, 108, 33) && !selected(&s, 28, 33), "depth {depth}: the selection followed");
        // One step: undo brings the bar and the selection back.
        s.execute("edit.undo", json!({})).unwrap();
        assert!(is_object(rgba(&s, 28, 33)) && !is_object(rgba(&s, 108, 24)), "depth {depth}: undo");
        assert!(selected(&s, 28, 33), "depth {depth}: undo restores the selection");
    }
}

#[test]
fn content_aware_extend_keeps_the_original() {
    let mut s = session(128, 64, 8, "rgb");
    paint_layer(&mut s, with_object(20, 26));
    s.execute("select.rect", json!({"x": 16, "y": 22, "width": 20, "height": 20})).unwrap();
    let r = s.execute("paint.contentAwareMove", json!({"offset": [60, -10], "mode": "extend"})).unwrap();
    assert_eq!(r["mode"], "extend");
    assert!(is_object(rgba(&s, 25, 31)), "the original stays");
    assert!(is_object(rgba(&s, 85, 21)), "the copy lands");
    assert_eq!(s.active().unwrap().history.undo_label(), Some("Content-Aware Extend"));
}

#[test]
fn content_aware_move_structure_and_color_shape_the_result() {
    // Structure 7, Color 0: the content is copied exactly but for a band about a patch wide along
    // its edge, which blends into the new surroundings; Structure 3 gives the same, bit for bit
    // (as in Photoshop).
    let moved = |structure: u64| {
        let mut s = session(128, 64, 16, "rgb");
        paint_layer(&mut s, texture);
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 24, "height": 24})).unwrap();
        s.execute("paint.contentAwareMove", json!({"offset": [70, 20], "structure": structure, "mode": "extend"})).unwrap();
        s
    };
    let s = moved(7);
    for (x, y) in [(16, 16), (22, 22), (27, 27)] {
        let (got, want) = (rgba(&s, x + 70, y + 20), texture(x, y));
        for c in 0..4 {
            assert!((got[c] - want[c]).abs() <= tol(16), "strict copy at {x},{y}: {got:?} vs {want:?}");
        }
    }
    let s3 = moved(3);
    for y in 30..54 {
        for x in 80..104 {
            assert_eq!(rgba(&s, x, y), rgba(&s3, x, y), "structure 3 and 7 agree at {x},{y}");
        }
    }
    // Structure 2 widens the re-synthesised band: the edge differs from a plain copy, the centre
    // doesn't. Structure 1 re-synthesises the content too, its centre included.
    let copy = |s: &Session, x: i32, y: i32| (rgba(s, x + 70, y + 20), rgba(s, x, y));
    let s = moved(2);
    let (centre, orig) = copy(&s, 22, 22);
    assert_eq!(centre, orig, "the centre is kept");
    let edge = (0..24).filter(|i| copy(&s, 10 + i, 10).0 != copy(&s, 10 + i, 10).1).count();
    assert!(edge > 12, "the top edge row is blended in ({edge} of 24 pixels differ)");
    let s = moved(1);
    let (centre, orig) = copy(&s, 22, 22);
    assert_ne!(centre, orig, "structure 1 re-synthesises the centre too");

    // Color 10 fits a bright patch to a dark place; Color 0 keeps it bright.
    let mean = |color: u64| {
        let mut s = session(128, 64, 32, "rgb");
        paint_layer(&mut s, |x, _| if x < 64 { [0.8, 0.8, 0.8, 1.0] } else { [0.2, 0.2, 0.2, 1.0] });
        s.execute("select.rect", json!({"x": 20, "y": 20, "width": 24, "height": 24})).unwrap();
        s.execute("paint.contentAwareMove", json!({"offset": [70, 0], "structure": 7, "color": color, "mode": "extend"})).unwrap();
        // The copied middle (the edge band is re-synthesised from the dark surroundings).
        (98..106).map(|x| rgba(&s, x, 32)[0]).sum::<f32>() / 8.0
    };
    let (kept, some, fitted) = (mean(0), mean(3), mean(10));
    assert!((kept - 0.8).abs() < 1e-3, "color 0 keeps the colour: {kept}");
    // Photoshop shifts by up to about 0.04 per step of Color.
    assert!((some - 0.68).abs() < 0.03, "color 3 adapts part of the way: {some}");
    assert!((fitted - 0.4).abs() < 0.03, "color 10 adapts most of the way: {fitted}");
}

#[test]
fn content_aware_move_overlapping_at_the_edge_and_on_other_targets() {
    // A short drag (the content overlaps its old place) from the canvas edge.
    let mut s = session(96, 48, 8, "rgb");
    paint_layer(&mut s, with_object(0, 18));
    s.execute("select.rect", json!({"x": 0, "y": 14, "width": 16, "height": 20})).unwrap();
    s.execute("paint.contentAwareMove", json!({"offset": [8, 0]})).unwrap();
    assert!(is_object(rgba(&s, 14, 24)), "object moved right");
    assert!(!is_object(rgba(&s, 2, 24)), "its old left part is filled: {:?}", rgba(&s, 2, 24));
    assert!(rgba(&s, 0, 24)[3] > 0.99, "the edge stays opaque");

    // On a transparent layer the old place becomes transparent like its surroundings.
    let mut s = session(96, 48, 8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("paint.pencil", json!({"points": [[30, 24]], "size": 10, "color": "#1020e0"})).unwrap();
    s.execute("select.rect", json!({"x": 20, "y": 14, "width": 20, "height": 20})).unwrap();
    s.execute("paint.contentAwareMove", json!({"offset": [40, 0]})).unwrap();
    assert!(rgba(&s, 30, 24)[3] < 0.05, "old place cleared: {:?}", rgba(&s, 30, 24));
    assert!(rgba(&s, 70, 24)[3] > 0.95, "object moved: {:?}", rgba(&s, 70, 24));

    // A layer mask and a grayscale document work like any retouch target.
    s.execute("layer.layerMask.revealAll", json!({})).unwrap();
    s.execute("paint.contentAwareMove", json!({"offset": [-20, 0], "target": "mask"})).unwrap();
    let mut g = session(64, 48, 16, "gray");
    g.execute("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 10})).unwrap();
    g.execute("paint.contentAwareMove", json!({"offset": [30, 20], "structure": 2, "color": 5})).unwrap();
}

#[test]
fn content_aware_move_samples_all_layers_onto_an_empty_layer() {
    let mut s = session(128, 64, 8, "rgb");
    paint_layer(&mut s, with_object(20, 26));
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 16, "y": 22, "width": 20, "height": 20})).unwrap();
    // Without Sample All Layers the empty layer has nothing to move.
    s.execute("paint.contentAwareMove", json!({"offset": [70, 0]})).unwrap();
    assert!(rgba(&s, 95, 31)[3] < 0.01);
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("paint.contentAwareMove", json!({"offset": [70, 0], "sampleAllLayers": true})).unwrap();
    assert!(is_object(rgba(&s, 95, 31)), "the object is painted onto the empty layer: {:?}", rgba(&s, 95, 31));
    let fill = rgba(&s, 25, 31);
    assert!(fill[3] > 0.99 && !is_object(fill), "its old place is covered with fill: {fill:?}");
    assert!(rgba(&s, 2, 2)[3] < 0.01, "elsewhere the layer stays empty");
}

#[test]
fn content_aware_move_fails_gracefully() {
    let mut s = session(64, 48, 8, "rgb");
    assert!(!s.is_enabled("paint.contentAwareMove"), "no selection");
    assert!(s.execute("paint.contentAwareMove", json!({"offset": [10, 0]})).is_err());
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 10, "height": 10})).unwrap();
    assert!(s.is_enabled("paint.contentAwareMove"));
    for p in [
        json!({}),
        json!({"offset": [10]}),
        json!({"offset": "far"}),
        json!({"offset": [1e300, 0]}),
        json!({"offset": [-1e300, 1e300]}),
        json!({"offset": [0, 0]}),
        json!({"offset": [0.4, -0.4]}),
        json!({"offset": [10, 0], "mode": "bogus"}),
        json!({"offset": [10, 0], "structure": 0}),
        json!({"offset": [10, 0], "structure": 8}),
        json!({"offset": [10, 0], "structure": "high"}),
        json!({"offset": [10, 0], "color": -1}),
        json!({"offset": [10, 0], "color": 11}),
        json!({"offset": [10, 0], "color": [1]}),
        json!({"offset": [50, 0]}),
        json!({"offset": [0, -30]}),
        json!({"offset": [10, 0], "target": "mask"}),
        json!({"offset": [10, 0], "target": {"channel": 7}}),
        json!({"offset": [10, 0], "layer": u64::MAX}),
        json!({"offset": [10, 0], "scaleX": 0}),
        json!({"offset": [10, 0], "scaleY": 0.5}),
        json!({"offset": [10, 0], "scaleX": 1e9}),
        json!({"offset": [10, 0], "scaleX": "big"}),
        json!({"offset": [10, 0], "rotate": 1e300}),
        json!({"offset": [10, 0], "rotate": [45]}),
        json!({"offset": [5000, 0], "scaleX": 200}),
        json!({"offset": [0, 0], "scaleX": 100, "rotate": 360}),
    ] {
        assert!(s.execute("paint.contentAwareMove", p.clone()).is_err(), "{p}");
    }
    assert!(s.execute("paint.contentAwareMove", json!({"offset": [10, 0], "structure": 7.0, "color": null})).is_ok());
    // Transform On Drop: scaling in place needs no drag; a big patch may reach past the edge.
    assert!(s.execute("paint.contentAwareMove", json!({"offset": [0, 0], "scaleX": 150, "rotate": -30})).is_ok());
    assert!(s.execute("paint.contentAwareMove", json!({"offset": [20, 0], "scaleX": -10000, "scaleY": 10000})).is_ok());
}

#[test]
fn clone_and_heal_with_a_far_source_are_plain_errors() {
    // Issue #1111: a translation whose offset saturates `Rect::translate` used to sample a
    // narrower window, relabel it full-size and index past the buffer (a panic the dispatch
    // guard turned into an internal error). It is a bad-params error, and the layer is untouched.
    for cmd in ["paint.cloneStamp", "paint.healingBrush"] {
        let mut s = session(40, 30, 8, "rgb");
        paint_layer(&mut s, texture);
        let before = rgba(&s, 10, 10);
        let steps = s.active().unwrap().history.past_len();
        for p in [
            json!({"points": [[10, 10]], "source": [3e9, 0]}),
            json!({"points": [[10, 10]], "source": [0, -3e9]}),
            json!({"points": [[10, 10]], "offset": [2147483647, 0]}),
            json!({"points": [[10, 10], [30, 10]], "source": [2147483640, 0]}),
        ] {
            let err = s.execute(cmd, p.clone()).expect_err(&format!("{cmd} {p}"));
            assert!(matches!(err, EngineError::BadParams { .. }), "{cmd} {p}: {err:?}");
        }
        assert_eq!(rgba(&s, 10, 10), before, "{cmd}: a refused stroke changed the layer");
        assert_eq!(s.active().unwrap().history.past_len(), steps, "{cmd}: a refused stroke recorded a history step");
        // A source far outside the canvas but inside the coordinate range is still fine: it clones
        // transparency.
        s.execute(cmd, json!({"points": [[10, 10]], "source": [1e6, 1e6]})).unwrap();
        s.execute(cmd, json!({"points": [[10, 10], [30, 10]], "source": [2147483000, 0]})).unwrap();
    }
}

fn stamp(s: &mut Session, extra: Value) -> Result<Value> {
    let mut p = json!({"points": [[16, 16]], "size": 16, "hardness": 100, "pattern": "Checkerboard", "scale": 100, "aligned": true});
    if let Some(obj) = extra.as_object() {
        for (k, v) in obj {
            p[k] = v.clone();
        }
    }
    s.execute("paint.patternStamp", p)
}

fn checker_at(x: i32, y: i32) -> f32 {
    if (x.div_euclid(8) + y.div_euclid(8)).rem_euclid(2) == 0 { 0.8 } else { 1.0 }
}

#[test]
fn pattern_stamp_paints_the_tile_at_every_depth_and_undoes() {
    for depth in DEPTHS {
        let mut s = session(64, 48, depth, "rgb");
        let before = rgba(&s, 16, 16);
        let r = stamp(&mut s, json!({"points": [[16, 16]]})).unwrap();
        assert_eq!(r["aligned"], json!(true));
        assert_eq!(r["phase"], json!([0.0, 0.0]));
        assert!(r["pattern"].as_str().is_some_and(|id| !id.is_empty()), "{r}");
        let got = rgba(&s, 16, 16);
        let want = checker_at(16, 16);
        for c in 0..3 {
            assert!((got[c] - want).abs() <= tol(depth), "depth {depth} {got:?} want {want}");
        }
        assert!(got[3] > 0.99, "{got:?}");
        assert_ne!(got, before);
        assert!(s.active().unwrap().doc.patterns.iter().any(|p| p.name == "Checkerboard"));
        assert!(s.undo());
        assert!((rgba(&s, 16, 16)[0] - before[0]).abs() <= tol(depth));
        assert!(s.redo());
        assert!((rgba(&s, 16, 16)[0] - want).abs() <= tol(depth));
    }
    let mut g = session(48, 32, 16, "gray");
    stamp(&mut g, json!({"points": [[16, 16]]})).unwrap();
    let p = rgba(&g, 16, 16);
    assert!((p[0] - checker_at(16, 16)).abs() <= tol(16), "{p:?}");
}

#[test]
fn pattern_stamp_integer_scale_reproduces_the_tile_in_the_dab() {
    let mut s = session(32, 32, 8, "rgb");
    stamp(&mut s, json!({"points": [[8, 8]], "size": 16, "hardness": 100, "phase": [0, 0]})).unwrap();
    for (x, y) in [(8, 8), (10, 8), (8, 12)] {
        let got = rgba(&s, x, y);
        let want = checker_at(x, y);
        assert!((got[0] - want).abs() <= tol(8) && (got[1] - want).abs() <= tol(8), "({x},{y}) {got:?} want {want}");
    }
}

#[test]
fn pattern_stamp_scaled_tiles_have_no_gaps() {
    for scale in [50.0, 200.0] {
        let mut s = session(64, 48, 8, "rgb");
        stamp(&mut s, json!({"points": [[24, 24]], "size": 24, "hardness": 100, "scale": scale, "phase": [0, 0]})).unwrap();
        let mut holes = 0;
        for y in 20..28 {
            for x in 20..28 {
                if rgba(&s, x, y)[3] < 0.5 {
                    holes += 1;
                }
            }
        }
        assert_eq!(holes, 0, "scale {scale}: transparent holes inside the dab");
    }
}

#[test]
fn pattern_stamp_aligned_keeps_document_phase_across_strokes() {
    let mut s = session(80, 40, 8, "rgb");
    stamp(&mut s, json!({"points": [[10, 16]], "size": 8, "hardness": 100, "aligned": true, "phase": [0, 0]})).unwrap();
    let a = rgba(&s, 10, 16);
    stamp(&mut s, json!({"points": [[18, 16]], "size": 8, "hardness": 100, "aligned": true, "phase": [0, 0]})).unwrap();
    let b = rgba(&s, 18, 16);
    assert!((a[0] - checker_at(10, 16)).abs() <= tol(8), "{a:?}");
    assert!((b[0] - checker_at(18, 16)).abs() <= tol(8), "{b:?}");
    assert!((a[0] - b[0]).abs() > 0.05, "aligned wallpaper: (10,16) and (18,16) sit on different cells");
}

#[test]
fn pattern_stamp_unaligned_restarts_at_each_stroke() {
    let mut s = session(80, 40, 8, "rgb");
    let r1 = stamp(&mut s, json!({"points": [[10, 16]], "size": 8, "hardness": 100, "aligned": false})).unwrap();
    assert_eq!(r1["phase"], json!([10.0, 16.0]));
    let a = rgba(&s, 10, 16);
    let r2 = stamp(&mut s, json!({"points": [[40, 16]], "size": 8, "hardness": 100, "aligned": false})).unwrap();
    assert_eq!(r2["phase"], json!([40.0, 16.0]));
    let b = rgba(&s, 40, 16);
    assert!((a[0] - b[0]).abs() <= tol(8), "each stroke's first point is the tile origin: {a:?} vs {b:?}");
}

#[test]
fn pattern_stamp_impressionist_is_seeded_and_replayable() {
    let pts = json!([[20, 20], [36, 22], [52, 20]]);
    let params = json!({"points": pts, "size": 12, "hardness": 80, "pattern": "Bricks", "impressionist": true, "aligned": true});
    let mut a = session(80, 40, 8, "rgb");
    a.execute("paint.patternStamp", params.clone()).unwrap();
    let mut samples = [[0.0f32; 4]; 3];
    for (i, x) in [20, 36, 52].into_iter().enumerate() {
        samples[i] = rgba(&a, x, 20);
    }
    assert!(a.undo());
    a.execute("paint.patternStamp", params.clone()).unwrap();
    for (i, x) in [20, 36, 52].into_iter().enumerate() {
        let got = rgba(&a, x, 20);
        for c in 0..4 {
            assert!((got[c] - samples[i][c]).abs() < 1e-6, "replay x={x} {got:?} vs {:?}", samples[i]);
        }
    }
    let mut b = session(80, 40, 8, "rgb");
    let mut off = params.clone();
    off["impressionist"] = json!(false);
    b.execute("paint.patternStamp", off).unwrap();
    let smooth = rgba(&b, 36, 20);
    let noisy = samples[1];
    assert!((noisy[0] - smooth[0]).abs() > 1e-4 || (noisy[1] - smooth[1]).abs() > 1e-4, "impressionist should jitter the phase: {noisy:?} vs {smooth:?}");
}

#[test]
fn pattern_stamp_tile_seam_matches_an_untilled_reference() {
    let mut s = session(96, 96, 8, "rgb");
    stamp(&mut s, json!({"points": [[64, 48]], "size": 32, "hardness": 100, "phase": [0, 0]})).unwrap();
    let pat = s.patterns.items.iter().find(|p| p.name == "Checkerboard").cloned().expect("builtin");
    let tile = photocraft_compose::pattern::Tile::new(&pat).expect("tile");
    let place = photocraft_compose::pattern::Placement::new(photocraft_geom::Rect::EMPTY, false, (0.0, 0.0), 1.0, 0.0);
    let col = photocraft_geom::Rect::new(64, 40, 65, 56);
    let reference = photocraft_compose::pattern::render(&tile, &place, col);
    for (i, y) in (40..56).enumerate() {
        let got = rgba(&s, 64, y);
        let want = reference[i];
        for c in 0..3 {
            assert!((got[c] - want[c]).abs() <= tol(8), "y={y} {got:?} vs {want:?}");
        }
    }
}

#[test]
fn pattern_stamp_rejects_hostile_input() {
    let mut s = session(64, 48, 8, "rgb");
    assert!(s.execute("paint.patternStamp", json!({"points": [], "pattern": "Checkerboard"})).is_err());
    assert!(s.execute("paint.patternStamp", json!({"pattern": "Checkerboard"})).is_err());
    assert!(s.execute("paint.patternStamp", json!({"points": [[16, 16]], "pattern": "no-such-pattern"})).is_err());
    assert!(s.execute("paint.patternStamp", json!({"points": [[16, 16]], "pattern": "Checkerboard", "size": 1e30})).is_err());
    assert!(s.execute("paint.patternStamp", json!({"points": [[1e7, 0]], "pattern": "Checkerboard", "size": 8})).is_err());
    assert!(s.execute("paint.patternStamp", json!({"points": [[f64::NAN, 0]], "pattern": "Checkerboard", "size": 8})).is_err());
    assert!(s.execute("paint.patternStamp", json!({"points": [[0, 0], [9000, 9000]], "pattern": "Checkerboard", "size": 5000, "hardness": 100})).is_err());
    let fmt = photocraft_color::PixelFormat::new(photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8, false);
    s.patterns.items.push(photocraft_doc::Pattern {
        id: "empty-pat".into(),
        name: "EmptyTile".into(),
        width: 0,
        height: 0,
        surface: photocraft_raster::Surface::new(fmt),
    });
    assert!(s.execute("paint.patternStamp", json!({"points": [[8, 8]], "pattern": "EmptyTile", "size": 8})).is_err());
    s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
    assert!(stamp(&mut s, json!({"points": [[16, 16]]})).is_err());
}

/// The live Clone Stamp shows what the commit paints: the same pixels, fed in pieces or at once,
/// with soft overlapping dabs (built-up coverage) and opacity.
#[test]
fn live_clone_matches_the_commit() {
    let mut s = session(96, 64, 8, "rgb");
    paint_layer(&mut s, texture);
    let p = json!({"points": [[50, 30], [60, 34], [70, 30], [80, 36]], "source": [20, 20], "size": 14, "hardness": 40, "opacity": 70});
    let pts = |v: &[[f64; 2]]| v.iter().map(|q| StrokePoint::new(q[0], q[1], 1.0)).collect::<Vec<_>>();
    let mut live =
        LiveRetouch::begin(&s, "paint.cloneStamp", &json!({"points": [[50, 30]], "source": [20, 20], "size": 14, "hardness": 40, "opacity": 70})).unwrap();
    live.push(&pts(&[[60.0, 34.0], [70.0, 30.0]])).unwrap();
    live.push(&pts(&[[80.0, 36.0]])).unwrap();
    let shown = live.doc.clone();
    s.execute("paint.cloneStamp", p).unwrap();
    let id = s.active().unwrap().active_layer.unwrap();
    let (a, b) = (shown.layer(id).unwrap().surface().unwrap(), s.active().unwrap().doc.layer(id).unwrap().surface().unwrap());
    for y in 20..46 {
        for x in 40..92 {
            let (got, want) = (a.rgba(x, y), b.rgba(x, y));
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() <= tol(8), "({x}, {y}): live {got:?} vs commit {want:?}");
            }
        }
    }
}

// Both preview implementations must reject an entire pointer batch before their renderer,
// coverage, tail or working surface changes. Compare a resumed stroke with a clean control.
macro_rules! live_coordinate_regression {
    ($name:ident, $live:ty, [$($cmd:literal),+]) => {
        #[test]
        fn $name() {
            for depth in DEPTHS {
                let mut s = session(32, 32, depth, "rgb");
                paint_layer(&mut s, texture);
                let st = s.active().unwrap();
                let (doc, revision, history) = (st.doc.clone(), st.revision, st.history.past_len());
                let id = st.active_layer.unwrap();
                for cmd in [$($cmd),+] {
                    let p = json!({"points": [[8, 16]], "source": [4, 8], "size": 4, "hardness": 100});
                    let mut live = <$live>::begin(&s, cmd, &p).unwrap();
                    let mut control = <$live>::begin(&s, cmd, &p).unwrap();
                    // The first regression case fails fast even without the guard, before any
                    // enormous coordinate reaches the path walker.
                    for (x, y) in [
                        (f64::NAN, 16.0),
                        (16.0, f64::NAN),
                        (f64::INFINITY, 16.0),
                        (16.0, f64::NEG_INFINITY),
                        (1e300, 16.0),
                        (16.0, -crate::brush_cmds::MAX_COORD - 1.0),
                    ] {
                        let shown = live.doc.clone();
                        let bounds = live.bounds();
                        let batch = [StrokePoint::new(12.0, 18.0, 1.0), StrokePoint::new(x, y, 1.0)];
                        let err = live.push(&batch).unwrap_err();
                        assert!(matches!(err, EngineError::BadParams { cmd: ref actual, ref msg }
                            if actual == cmd && msg.contains("point coordinates must be finite")));
                        assert!(std::sync::Arc::ptr_eq(&shown, &live.doc), "{cmd}: invalid batch changed the preview");
                        assert_eq!(live.bounds(), bounds, "{cmd}");
                    }
                    let next = [StrokePoint::new(-4.0, 16.0, 1.0), StrokePoint::new(24.0, 16.0, 1.0)];
                    assert_eq!(live.push(&next).unwrap(), control.push(&next).unwrap(), "{cmd}");
                    assert_eq!(live.bounds(), control.bounds(), "{cmd}");
                    let got = live.doc.layer(id).unwrap().surface().unwrap();
                    let want = control.doc.layer(id).unwrap().surface().unwrap();
                    assert_eq!(got.read_region(live.bounds()), want.read_region(control.bounds()), "{cmd}");
                    assert_eq!(got.content_bounds(), want.content_bounds(), "{cmd}");
                }
                let st = s.active().unwrap();
                assert!(std::sync::Arc::ptr_eq(&doc, &st.doc));
                assert_eq!((st.revision, st.history.past_len()), (revision, history));
            }
        }
    };
}

live_coordinate_regression!(
    live_retouch_rejects_invalid_batches_without_changing_the_preview,
    LiveRetouch,
    ["paint.cloneStamp", "paint.healingBrush", "paint.historyBrush"]
);
live_coordinate_regression!(
    live_dab_rejects_invalid_batches_without_changing_the_preview,
    LiveDab,
    ["paint.dodge", "paint.burn", "paint.sponge", "paint.blur", "paint.sharpen", "paint.smudge"]
);

/// The live Dodge shows what the commit paints, also where a stroke doubles back across the
/// cells that remember each pixel's original colour (the live stroke edits one dab at a time).
/// 32-bit, so the live working copy isn't quantised between dabs.
#[test]
fn live_dodge_matches_the_commit_across_cells() {
    let mut s = session(200, 90, 32, "rgb");
    paint_layer(&mut s, texture);
    let all = [[10.0, 40.0], [100.0, 60.0], [180.0, 40.0], [60.0, 30.0], [150.0, 70.0]];
    let pts = |v: &[[f64; 2]]| v.iter().map(|q| StrokePoint::new(q[0], q[1], 1.0)).collect::<Vec<_>>();
    let opts = |p: &[[f64; 2]]| json!({"points": p, "size": 40, "hardness": 30, "exposure": 60, "range": "midtones"});
    let mut live = LiveDab::begin(&s, "paint.dodge", &opts(&all[..1])).unwrap();
    live.push(&pts(&all[1..3])).unwrap();
    live.push(&pts(&all[3..])).unwrap();
    let shown = live.doc.clone();
    s.execute("paint.dodge", opts(&all)).unwrap();
    let id = s.active().unwrap().active_layer.unwrap();
    let (a, b) = (shown.layer(id).unwrap().surface().unwrap(), s.active().unwrap().doc.layer(id).unwrap().surface().unwrap());
    let (mut changed, mut worst) = (false, 0.0f32);
    for y in 0..90 {
        for x in 0..200 {
            let (got, want) = (a.rgba(x, y), b.rgba(x, y));
            changed |= (want[0] - texture(x, y)[0]).abs() > 0.02;
            // The commit also lays the stroke's last dab, which the live stroke shows on release
            // (a soft tip reaches past its 20 px radius).
            let reach = photocraft_paint::tip_reach(20.0, 0.3).ceil() as i32 + 2;
            if (x - 150).pow(2) + (y - 70).pow(2) > reach * reach {
                worst = (0..4).map(|c| (got[c] - want[c]).abs()).fold(worst, f32::max);
            }
        }
    }
    assert!(changed);
    // A soft tip's footprint spans more live cells, so sums land in a different order: within
    // 2e-4 (a twentieth of an 8-bit level).
    assert!(worst <= 2e-4, "live and commit differ by {worst}");
}

#[test]
fn retouch_strokes_refuse_absurd_point_coordinates() {
    // #976: a segment to (1e300, 0) has an infinite length and the dab walker would never stop, so
    // every retouch tool's `points` must stay within ±1e6, as the painting tools' do.
    let cmds = [
        "paint.dodge",
        "paint.burn",
        "paint.sponge",
        "paint.blur",
        "paint.sharpen",
        "paint.smudge",
        "paint.cloneStamp",
        "paint.healingBrush",
        "paint.spotHealing",
        "paint.historyBrush",
        "paint.patternStamp",
        "paint.remove",
    ];
    // A single far point first: cheap even without the bound, so a regression fails fast.
    let far = [json!([[1_000_001, 0]]), json!([[0, 0], [0, -2_000_000]]), json!([[0, 0], [1e300, 0]]), json!([[1e300, 1e300]])];
    for depth in DEPTHS {
        let mut s = session(32, 32, depth, "rgb");
        paint_layer(&mut s, |x, y| [x as f32 / 32.0, y as f32 / 32.0, 0.25, 1.0]);
        s.execute("cloneSource.set", json!({"source": [8, 8]})).unwrap();
        let (before, past) = (rgba(&s, 16, 16), s.active().unwrap().history.past_len());
        for cmd in cmds {
            for points in &far {
                let err = s.execute(cmd, json!({ "points": points })).unwrap_err();
                assert!(err.to_string().contains("within ±"), "{cmd} {points} at {depth}: {err}");
            }
        }
        assert_eq!((rgba(&s, 16, 16), s.active().unwrap().history.past_len()), (before, past), "{depth}: nothing changed");
        // Control: a stroke on the canvas still retouches.
        s.execute("paint.dodge", json!({"points": [[4, 16], [28, 16]], "size": 8})).unwrap();
        assert_ne!(rgba(&s, 16, 16), before, "{depth}");
    }
}

// ---------------------------------------------------------------------------------------------
// Remove Tool
// ---------------------------------------------------------------------------------------------

/// A flat grey layer with a dark square at 40..52 × 24..36.
fn square_session(depth: u64, mode: &str) -> Session {
    let mut s = session(96, 64, depth, mode);
    paint_layer(&mut s, |x, y| if (40..52).contains(&x) && (24..36).contains(&y) { [0.05, 0.05, 0.05, 1.0] } else { [0.5, 0.5, 0.5, 1.0] });
    s
}

/// A ring of points around the square.
fn ring() -> Vec<[f64; 2]> {
    (0..=64).map(|i| f64::from(i) / 64.0 * std::f64::consts::TAU).map(|t| [46.0 + 13.0 * t.cos(), 30.0 + 13.0 * t.sin()]).collect()
}

fn near(a: [f32; 4], b: [f32; 4], tol: f32) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() <= tol)
}

#[test]
fn remove_a_ring_around_an_object_removes_it_in_one_undo_step() {
    for (depth, mode) in [(8, "rgb"), (16, "gray"), (32, "rgb"), (8, "cmyk"), (16, "lab")] {
        let mut s = square_session(depth, mode);
        let bg = rgba(&s, 10, 10);
        assert!(!near(rgba(&s, 46, 30), bg, 0.05), "{mode}/{depth}: the square is there");
        let past = s.active().unwrap().history.past_len();
        let r = s.execute("paint.remove", json!({"points": ring(), "size": 4})).unwrap();
        assert!(r["damage"].is_array(), "{r}");
        for (x, y) in [(46, 30), (40, 24), (51, 35)] {
            assert!(near(rgba(&s, x, y), bg, 0.03), "{mode}/{depth} ({x}, {y}): {:?} vs {bg:?}", rgba(&s, x, y));
        }
        assert_eq!(s.active().unwrap().history.past_len(), past + 1, "{mode}/{depth}: one undo step");
        s.execute("edit.undo", json!({})).unwrap();
        assert!(!near(rgba(&s, 46, 30), bg, 0.05), "{mode}/{depth}: undo brings the square back");
    }
}

#[test]
fn remove_open_strokes_and_the_selection_limit_what_changes() {
    let mut s = square_session(8, "rgb");
    let bg = rgba(&s, 10, 10);
    // Without closeLoops a ring leaves its inside alone.
    s.execute("paint.remove", json!({"points": ring(), "size": 4, "closeLoops": false})).unwrap();
    assert!(!near(rgba(&s, 46, 30), bg, 0.05));
    // A stroke straight over the square removes it.
    s.execute("paint.remove", json!({"points": [[38, 30], [54, 30]], "size": 16})).unwrap();
    assert!(near(rgba(&s, 46, 30), bg, 0.03), "{:?}", rgba(&s, 46, 30));
    // Selected elsewhere, a stroke over the square changes nothing.
    let mut s = square_session(8, "rgb");
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 20})).unwrap();
    let err = s.execute("paint.remove", json!({"points": [[46, 30]], "size": 20})).unwrap_err().to_string();
    assert!(err.contains("check the selection"), "{err}");
    assert!(!near(rgba(&s, 46, 30), bg, 0.05));
}

#[test]
fn remove_on_an_empty_layer_is_a_no_op_without_a_history_step() {
    let mut s = square_session(8, "rgb");
    s.execute("layer.new.layer", json!({})).unwrap();
    let past = s.active().unwrap().history.past_len();
    let v = s.execute("paint.remove", json!({"points": [[38, 30], [54, 30]], "size": 16})).unwrap();
    assert_eq!(v, json!({"changed": false}));
    assert_eq!(s.active().unwrap().history.past_len(), past);
    // With Sample All Layers on, the image below is there to remove from.
    s.execute("paint.remove", json!({"points": [[38, 30], [54, 30]], "size": 16, "sampleAllLayers": true})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), past + 1);
}

#[test]
fn remove_thin_scratches_take_their_surroundings() {
    let mut s = session(80, 40, 16, "rgb");
    paint_layer(&mut s, |x, y| if y == 20 { [1.0, 1.0, 1.0, 1.0] } else { [x as f32 / 80.0, 0.4, 0.6, 1.0] });
    s.execute("paint.remove", json!({"points": [[5, 20], [75, 20]], "size": 2})).unwrap();
    for x in [10, 40, 70] {
        assert!(near(rgba(&s, x, 20), [x as f32 / 80.0, 0.4, 0.6, 1.0], 0.02), "x {x}: {:?}", rgba(&s, x, 20));
    }
}

#[test]
fn remove_bad_params_are_errors() {
    assert!(Session::new().execute("paint.remove", json!({"points": [[1, 1]]})).is_err(), "no document");
    let mut s = square_session(8, "rgb");
    for p in [
        json!({}),
        json!({"points": []}),
        json!({"points": "x"}),
        json!({"points": [[1, 1]], "size": "big", "closeLoops": "yes", "sampleAllLayers": 3}),
        json!({"points": [[1, 1]], "layer": u64::MAX}),
    ] {
        let _ = s.execute("paint.remove", p);
    }
    let err = s.execute("paint.remove", json!({"points": [[-500, -500]], "size": 10})).unwrap_err().to_string();
    assert!(err.contains("outside the canvas"), "{err}");
    // A stroke far off the canvas is refused before its coverage (its whole bounding box) is rendered.
    let err = s.execute("paint.remove", json!({"points": [[0, 0], [99_999, 99_999]], "size": 5000})).unwrap_err().to_string();
    assert!(err.contains("too large"), "{err}");
    let err = s.execute("paint.remove", json!({"points": [[48, 32]], "size": 3000})).unwrap_err().to_string();
    assert!(err.contains("nothing is left"), "{err}");
}

/// Flat (150, 120, 100) with Gaussian film grain of σ 12 levels (the #2104 test image).
fn film_grain(x: i32, y: i32) -> [f32; 4] {
    let u = |k: u32| {
        let mut z = (x as u32).wrapping_mul(0x9E37_79B9) ^ (y as u32).wrapping_mul(0x85EB_CA6B) ^ k.wrapping_mul(0xC2B2_AE35);
        z = (z ^ (z >> 16)).wrapping_mul(0x7FEB_352D);
        z = (z ^ (z >> 15)).wrapping_mul(0x846C_A68B);
        ((z ^ (z >> 16)) as f32 + 1.0) / (u32::MAX as f32 + 2.0)
    };
    let n = (-2.0 * u(1).ln()).sqrt() * (std::f32::consts::TAU * u(2)).cos() * 12.0;
    [(150.0 + n) / 255.0, (120.0 + n) / 255.0, (100.0 + n) / 255.0, 1.0]
}

#[test]
fn spot_healing_keeps_film_grain_through_overlapping_strokes() {
    // #2104: on flat colour with grain the heal must carry the grain, not leave a smooth patch,
    // and must not lose more with each overlapping stroke (each samples the ones before).
    let red = |s: &Session, x0: i32, x1: i32, y0: i32, y1: i32| {
        let v: Vec<f32> = (y0..y1).flat_map(|y| (x0..x1).map(move |x| (x, y))).map(|(x, y)| rgba(s, x, y)[0] * 255.0).collect();
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (m, (v.iter().map(|a| (a - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt())
    };
    for (depth, kind) in [(8u64, "contentAware"), (8, "proximityMatch"), (8, "createTexture"), (16, "createTexture")] {
        let mut s = session(400, 300, depth, "rgb");
        paint_layer(&mut s, film_grain);
        let (m0, sd0) = red(&s, 0, 100, 0, 100);
        for i in 0..8 {
            let y = 100 + 10 * i;
            s.execute("paint.spotHealing", json!({"points": [[140, y], [260, y]], "size": 30, "hardness": 100, "type": kind})).unwrap();
            let (m, sd) = red(&s, 150, 250, y - 4, y + 5);
            assert!(sd > 0.8 * sd0 && sd < 1.25 * sd0, "{kind} depth {depth} stroke {}: grain {sd} vs {sd0} around", i + 1);
            assert!((m - m0).abs() < 3.0, "{kind} depth {depth} stroke {}: mean {m} vs {m0}", i + 1);
        }
    }
}

#[test]
fn remove_several_strokes_apply_as_one_job_and_one_undo_step() {
    let mut s = square_session(8, "rgb");
    let bg = rgba(&s, 10, 10);
    let past = s.active().unwrap().history.past_len();
    // Each stroke covers half of the square; together they remove it.
    let r = s.execute("paint.remove", json!({"strokes": [[[38, 30], [45, 30]], [[47, 30], [54, 30]]], "size": 16})).unwrap();
    assert!(r["damage"].is_array(), "{r}");
    for (x, y) in [(41, 30), (46, 30), (51, 30)] {
        assert!(near(rgba(&s, x, y), bg, 0.03), "({x}, {y}): {:?} vs {bg:?}", rgba(&s, x, y));
    }
    assert_eq!(s.active().unwrap().history.past_len(), past + 1, "one undo step for both strokes");
    for p in [
        json!({"strokes": [], "size": 16}),
        json!({"strokes": "x", "size": 16}),
        json!({"strokes": [[]], "size": 16}),
        json!({"strokes": [[[1, 1]], "x"], "size": 16}),
    ] {
        assert!(s.execute("paint.remove", p).is_err());
    }
}

#[test]
fn remove_to_a_new_layer_leaves_the_target_untouched() {
    let mut s = square_session(8, "rgb");
    let target = s.active().unwrap().active_layer.unwrap();
    let bg = rgba(&s, 10, 10);
    let square = rgba(&s, 46, 30);
    let past = s.active().unwrap().history.past_len();
    let r = s.execute("paint.remove", json!({"points": [[38, 30], [54, 30]], "size": 16, "newLayer": true})).unwrap();
    let nid = LayerId(r["layer"].as_u64().expect("the new layer's id"));
    let d = s.active().unwrap();
    assert_eq!(d.active_layer, Some(nid), "the new layer is active");
    assert_ne!(nid, target);
    let order: Vec<LayerId> = d.doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
    assert_eq!(order, vec![target, nid], "the new layer sits above the target");
    assert_eq!(d.doc.layer(nid).unwrap().name, "Layer 1");
    // The target keeps its pixels; the new layer carries the fill inside the hole only.
    let old = d.doc.layer(target).unwrap().surface().unwrap();
    assert!(near(old.rgba(46, 30), square, 0.001));
    let new = d.doc.layer(nid).unwrap().surface().unwrap();
    assert!(near(new.rgba(46, 30), bg, 0.03), "{:?}", new.rgba(46, 30));
    assert_eq!(new.rgba(10, 10)[3], 0.0, "transparent away from the hole");
    assert_eq!(d.history.past_len(), past + 1);
    s.execute("edit.undo", json!({})).unwrap();
    let d = s.active().unwrap();
    assert!(d.doc.layer(nid).is_none(), "undo removes the layer");
    assert_eq!(d.active_layer, Some(target));
    // On an empty layer with Sample All Layers, the composite is what gets filled, onto a new layer.
    s.execute("layer.new.layer", json!({})).unwrap();
    let empty = s.active().unwrap().active_layer.unwrap();
    let r = s.execute("paint.remove", json!({"points": [[38, 30], [54, 30]], "size": 16, "sampleAllLayers": true, "newLayer": true})).unwrap();
    let nid = LayerId(r["layer"].as_u64().unwrap());
    let d = s.active().unwrap();
    let order: Vec<LayerId> = d.doc.walk().into_iter().map(|(_, _, l)| l.id).collect();
    assert_eq!(order, vec![target, empty, nid]);
    assert!(near(d.doc.layer(nid).unwrap().surface().unwrap().rgba(46, 30), bg, 0.03));
    assert!(near(d.doc.layer(target).unwrap().surface().unwrap().rgba(46, 30), square, 0.001));
}

/// A Clone Stamp or History Brush stroke drawn live and offered to its commit: the commit takes
/// the live pixels, and they are what running the command afresh paints, bit for bit, at every
/// depth, with and without a feathered selection.
#[test]
fn a_live_clone_or_history_stroke_commits_its_own_pixels() {
    let all = [[30.0, 30.0], [50.0, 34.0], [70.0, 28.0], [90.0, 40.0], [60.0, 50.0], [35.0, 44.0], [36.0, 45.0]];
    let pts = |v: &[[f64; 2]]| v.iter().map(|q| StrokePoint::new(q[0], q[1], 1.0)).collect::<Vec<_>>();
    for depth in DEPTHS {
        for (cmd, extra) in [("paint.cloneStamp", json!({"source": [20, 15], "mode": "multiply"})), ("paint.historyBrush", json!({"state": 1}))] {
            for selected in [false, true] {
                let setup = || {
                    let mut s = session(120, 70, depth, "rgb");
                    paint_layer(&mut s, texture);
                    s.execute("paint.stroke", json!({"points": [[0, 40], [120, 40]], "size": 20, "hardness": 1.0, "color": "#c03030"})).unwrap();
                    if selected {
                        s.execute("select.rect", json!({"x": 40, "y": 20, "width": 50, "height": 40, "ellipse": true, "feather": 3})).unwrap();
                    }
                    s
                };
                let params = |v: &[[f64; 2]]| {
                    let mut p = json!({"points": v, "size": 14, "hardness": 40, "opacity": 70});
                    for (k, x) in extra.as_object().unwrap() {
                        p[k] = x.clone();
                    }
                    p
                };
                let mut s = setup();
                let mut live = LiveRetouch::begin(&s, cmd, &params(&all[..1])).unwrap();
                live.push(&pts(&all[1..3])).unwrap();
                live.push(&pts(&all[3..4])).unwrap();
                live.push(&pts(&all[4..])).unwrap();
                s.offer_live_retouch(live).unwrap();
                let r = s.execute(cmd, params(&all)).unwrap();
                s.clear_live_stroke();
                assert_eq!(r["fromLive"], true, "{cmd} {depth}-bit selected={selected}");
                let mut fresh = setup();
                let r = fresh.execute(cmd, params(&all)).unwrap();
                assert!(r.get("fromLive").is_none());
                let (a, b) = (s.active().unwrap(), fresh.active().unwrap());
                let bounds = a.doc.bounds();
                let got = a.doc.layer(a.active_layer.unwrap()).unwrap().surface().unwrap().read_region(bounds);
                let want = b.doc.layer(b.active_layer.unwrap()).unwrap().surface().unwrap().read_region(bounds);
                let diff = got.iter().zip(&want).position(|(x, y)| x.to_bits() != y.to_bits());
                assert!(diff.is_none(), "{cmd} {depth}-bit selected={selected}: first difference at value {diff:?}");
                assert_eq!(a.history.past_len(), b.history.past_len());
            }
        }
    }
}

/// An offered live stroke is taken only by the commit it was drawn for, and only once.
#[test]
fn a_live_retouch_offer_is_taken_only_by_its_own_commit() {
    let mut s = session(96, 64, 8, "rgb");
    paint_layer(&mut s, texture);
    let all = [[50.0, 30.0], [60.0, 34.0], [70.0, 30.0]];
    let params = |v: &[[f64; 2]], opacity: u32| json!({"points": v, "source": [20, 20], "size": 12, "hardness": 50, "opacity": opacity});
    let offer = |s: &mut Session| {
        let live = LiveRetouch::begin(s, "paint.cloneStamp", &params(&all, 70)).unwrap();
        s.offer_live_retouch(live).unwrap();
    };
    let from_live = |r: Value| r.get("fromLive").is_some();
    // Other points, other params, another command: rendered afresh.
    offer(&mut s);
    assert!(!from_live(s.execute("paint.cloneStamp", params(&all[..2], 70)).unwrap()));
    s.execute("edit.undo", json!({})).unwrap();
    offer(&mut s);
    assert!(!from_live(s.execute("paint.cloneStamp", params(&all, 60)).unwrap()));
    s.execute("edit.undo", json!({})).unwrap();
    offer(&mut s);
    let mut history = params(&all, 70);
    history.as_object_mut().unwrap().remove("source");
    assert!(!from_live(s.execute("paint.historyBrush", history).unwrap()));
    s.execute("edit.undo", json!({})).unwrap();
    // The document changed after the stroke started.
    offer(&mut s);
    s.execute("paint.stroke", json!({"points": [[5, 5], [9, 9]], "size": 4})).unwrap();
    assert!(!from_live(s.execute("paint.cloneStamp", params(&all, 70)).unwrap()));
    s.execute("edit.undo", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    // A withdrawn offer, and an offer used once.
    offer(&mut s);
    s.clear_live_stroke();
    assert!(!from_live(s.execute("paint.cloneStamp", params(&all, 70)).unwrap()));
    s.execute("edit.undo", json!({})).unwrap();
    offer(&mut s);
    assert!(from_live(s.execute("paint.cloneStamp", params(&all, 70)).unwrap()));
    s.execute("edit.undo", json!({})).unwrap();
    assert!(!from_live(s.execute("paint.cloneStamp", params(&all, 70)).unwrap()));
    // The Healing Brush blends on release: it has no live pixels to offer.
    let healing = LiveRetouch::begin(&s, "paint.healingBrush", &params(&all, 70)).unwrap();
    assert!(s.offer_live_retouch(healing).is_err());
    // The brush changed during the stroke (the document revision doesn't show it).
    let live = LiveRetouch::begin(&s, "paint.cloneStamp", &params(&all, 70)).unwrap();
    s.execute("tools.setBrush", json!({"brush": {"spacing": 0.5}})).unwrap();
    assert!(s.offer_live_retouch(live).is_err());
}

/// The live stroke renders only the position and pressure of its points, as its commit parses
/// them: a pen's tilt and the times don't change what it shows.
#[test]
fn a_live_retouch_stroke_ignores_tilt_and_times() {
    let mut s = session(64, 48, 8, "rgb");
    paint_layer(&mut s, texture);
    let p = json!({"points": [[20, 20]], "source": [8, 8], "size": 10, "hardness": 50});
    let mut plain = LiveRetouch::begin(&s, "paint.cloneStamp", &p).unwrap();
    let mut pen = LiveRetouch::begin(&s, "paint.cloneStamp", &p).unwrap();
    let pts = [[30.0, 24.0], [44.0, 30.0]];
    plain.push(&pts.map(|q| StrokePoint::new(q[0], q[1], 0.8))).unwrap();
    pen.push(&pts.map(|q| {
        let mut sp = StrokePoint::new(q[0], q[1], 0.8);
        (sp.tilt_x, sp.tilt_y, sp.rotation, sp.time) = (40.0, -20.0, 90.0, 120.0);
        sp
    }))
    .unwrap();
    assert_eq!(plain.points(), pen.points());
    let id = s.active().unwrap().active_layer.unwrap();
    let b = s.active().unwrap().doc.bounds();
    let read = |l: &LiveRetouch| l.doc.layer(id).unwrap().surface().unwrap().read_region(b);
    assert_eq!(read(&plain), read(&pen));
}

#[test]
fn periodic_clone_low_flow_revisits_accumulate_across_tile_aliases() {
    // Independent oracle: paint the same continuous stroke over two physical copies of a
    // tile, then combine the two coverage values using the brush's flow accumulation rule.
    let setup = |width| {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":width,"height":24,"depth":32})).unwrap();
        for x in (8..width).step_by(32) {
            s.execute("paint.pencil", json!({"points":[[x,12]],"size":16,"color":"#ff0000"})).unwrap();
        }
        s
    };
    let params = |wrap| json!({"points":[[22,12],[54,12]],"offset":[-18,0],"size":2,"hardness":100,"flow":20,"opacity":100,"wrap":wrap});
    let green = |s: &Session, x| {
        let st = s.active().unwrap();
        st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, 12)[1]
    };
    let mut physical = setup(64);
    physical.execute("paint.cloneStamp", params(false)).unwrap();
    let (a, b) = (green(&physical, 22), green(&physical, 54));
    assert!(a > 0.0 && a < 1.0 && b > 0.0 && b < 1.0, "oracle needs two partial visits: {a}, {b}");
    let expected = a * b;
    let mut periodic = setup(32);
    periodic.execute("paint.cloneStamp", params(true)).unwrap();
    let actual = green(&periodic, 22);
    assert!((actual - expected).abs() < 0.0001, "repeated visits should build flow: actual green={actual}, expected={expected}, physical copies={a},{b}");
}

#[test]
fn periodic_clone_live_offer_commits_and_undoes_the_same_pixels_as_replay() {
    for depth in [8, 16, 32] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width":32,"height":24,"depth":depth})).unwrap();
        s.execute("paint.pencil", json!({"points":[[8,8]],"size":12,"color":"#d03060"})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let id = s.active().unwrap().active_layer.unwrap();
        let params = json!({"points":[[30,22],[34,26],[38,28],[30,22]],"source":[8,8],"size":16,"hardness":70,"opacity":45,"flow":60,"wrap":true});
        let live = LiveRetouch::begin(&s, "paint.cloneStamp", &params).unwrap();
        let shown = live.doc.layer(id).unwrap().surface().unwrap().read_region(before.bounds());
        s.offer_live_retouch(live).unwrap();
        let result = s.execute("paint.cloneStamp", params.clone()).unwrap();
        assert_eq!(result["fromLive"], true);
        let read = |s: &Session| s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().read_region(before.bounds());
        assert_eq!(read(&s), shown, "offered preview at depth {depth}");
        let mut replay = Session::new();
        replay.add_document((*before).clone(), None);
        replay.execute("paint.cloneStamp", params).unwrap();
        assert_eq!(read(&s), read(&replay), "command replay at depth {depth}");
        assert!(s.undo());
        assert_eq!(s.active().unwrap().doc, before, "one undo at depth {depth}");
    }
}
#[test]
fn periodic_clone_interior_matches_ordinary_brush_masks_and_transparency() {
    let cases = [
        json!({"wetEdges":true}),
        json!({"dualBrush":{"enabled":true,"size":7,"hardness":0.5,"spacing":0.2}}),
        json!({"texture":{"enabled":true,"eachTip":false,"depth":0.6}}),
        json!({"pressureOpacity":true,"pressureSize":false}),
        json!({"wetEdges":true,"dualBrush":{"enabled":true,"size":7},"pressureOpacity":true}),
    ];
    for depth in DEPTHS {
        for brush in &cases {
            let setup = || {
                let mut s = session(96, 64, depth, "rgb");
                paint_layer(&mut s, |x, y| {
                    let mut px = texture(x, y);
                    px[3] = if x < 40 { 0.4 } else { 0.75 };
                    px
                });
                s.execute("tools.setBrush", json!({"brush":brush})).unwrap();
                s.execute("select.rect", json!({"x":48,"y":16,"width":36,"height":36,"feather":2})).unwrap();
                s
            };
            let mut ordinary = setup();
            let mut periodic = setup();
            let mut p = json!({"points":[[55,30,0.4],[65,34,0.8],[72,30,0.3]],"source":[20,20],"size":12,"hardness":65,"flow":35,"opacity":60});
            ordinary.execute("paint.cloneStamp", p.clone()).unwrap();
            p["wrap"] = json!(true);
            periodic.execute("paint.cloneStamp", p).unwrap();
            let read = |s: &Session| {
                let st = s.active().unwrap();
                st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().read_region(st.doc.bounds())
            };
            let (a, b) = (read(&ordinary), read(&periodic));
            let tolerance = if depth == 8 {
                1.1 / 255.0
            } else if depth == 16 {
                1.1 / 65535.0
            } else {
                0.00001
            };
            let error = a.iter().zip(&b).map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
            assert!(error <= tolerance, "depth={depth}, brush={brush}, max native-channel error={error}");
        }
    }
}
