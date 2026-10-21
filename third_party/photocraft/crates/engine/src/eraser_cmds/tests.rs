use super::*;
use photocraft_raster::Surface;

const MODES: [&str; 3] = ["rgb", "cmyk", "gray"];
const DEPTHS: [u32; 3] = [8, 16, 32];

/// White document with a black 20×20 square at (10, 10) on its Background layer.
fn session(mode: &str, depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48, "mode": mode, "depth": depth})).unwrap();
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20, "antiAlias": false})).unwrap();
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s
}

fn layer(s: &Session) -> &Layer {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap()
}

fn surface(s: &Session) -> &Surface {
    layer(s).surface().unwrap()
}

fn alpha(s: &Session, x: i32, y: i32) -> f32 {
    surface(s).rgba(x, y)[3]
}

fn history_len(s: &Session) -> usize {
    s.active().unwrap().history.past_len()
}

#[test]
fn magic_eraser_turns_the_background_into_a_layer_at_every_depth_and_mode() {
    for mode in MODES {
        for depth in DEPTHS {
            let mut s = session(mode, depth);
            let before = surface(&s).clone();
            let h = history_len(&s);
            let r = s.execute("paint.magicEraser", json!({"x": 2, "y": 2, "antiAlias": false})).unwrap();
            assert_eq!(r["erased"], true);
            assert_eq!(history_len(&s), h + 1, "one undo step {mode} {depth}");
            let l = layer(&s);
            assert_eq!(l.name, "Layer 0", "{mode} {depth}");
            assert!(!l.locks.transparency && !l.locks.position);
            assert_eq!(alpha(&s, 2, 2), 0.0, "white erased {mode} {depth}");
            assert_eq!(alpha(&s, 60, 40), 0.0, "contiguous white erased {mode} {depth}");
            assert_eq!(alpha(&s, 20, 20), 1.0, "black square kept {mode} {depth}");
            // The square's colour is untouched in the document's own channels.
            assert_eq!(surface(&s).pixel(20, 20), before.pixel(20, 20));
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(layer(&s).name, "Background");
            assert_eq!(surface(&s), &before);
        }
    }
}

#[test]
fn magic_eraser_options() {
    // Non-contiguous: a second, separate black square goes too.
    let mut s = session("rgb", 8);
    s.execute("select.rect", json!({"x": 40, "y": 10, "width": 10, "height": 10, "antiAlias": false})).unwrap();
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let mut a = s;
    a.execute("paint.magicEraser", json!({"x": 20, "y": 20, "contiguous": true, "antiAlias": false})).unwrap();
    assert_eq!(alpha(&a, 20, 20), 0.0);
    assert_eq!(alpha(&a, 45, 15), 1.0, "contiguous keeps the other square");
    a.execute("edit.undo", json!({})).unwrap();
    a.execute("paint.magicEraser", json!({"x": 20, "y": 20, "contiguous": false, "antiAlias": false})).unwrap();
    assert_eq!(alpha(&a, 45, 15), 0.0, "non-contiguous erases every match");
    assert_eq!(alpha(&a, 2, 2), 1.0);

    // Opacity 50: half erased.
    let mut s = session("rgb", 16);
    s.execute("paint.magicEraser", json!({"x": 2, "y": 2, "opacity": 50, "antiAlias": false})).unwrap();
    assert!((alpha(&s, 2, 2) - 0.5).abs() < 0.01);

    // Tolerance: a near-white grey is erased only within tolerance.
    for (tol, gone) in [(0.0, false), (40.0, true)] {
        let mut s = session("rgb", 8);
        s.execute("select.rect", json!({"x": 0, "y": 40, "width": 64, "height": 8, "antiAlias": false})).unwrap();
        s.execute("edit.fill", json!({"color": "#e6e6e6"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        s.execute("paint.magicEraser", json!({"x": 2, "y": 2, "tolerance": tol, "antiAlias": false})).unwrap();
        assert_eq!(alpha(&s, 5, 44) == 0.0, gone, "tolerance {tol}");
    }

    // The selection limits the erased area.
    let mut s = session("rgb", 8);
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 32, "height": 48, "antiAlias": false})).unwrap();
    s.execute("paint.magicEraser", json!({"x": 2, "y": 2, "antiAlias": false})).unwrap();
    assert_eq!(alpha(&s, 2, 2), 0.0);
    assert_eq!(alpha(&s, 50, 2), 1.0, "outside the selection kept");
}

#[test]
fn magic_eraser_samples_all_layers_and_paints_background_under_a_transparency_lock() {
    // A half-opaque red band over the Background: the composite band is split by the black square,
    // the layer's own band isn't.
    let band = |s: &mut Session| {
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 14, "width": 64, "height": 4, "antiAlias": false})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        s.edit("opacity", |doc, _| {
            doc.layer_mut(id).unwrap().opacity = 0.5;
            Ok(())
        })
        .unwrap();
    };
    let mut s = session("rgb", 8);
    band(&mut s);
    s.execute("paint.magicEraser", json!({"x": 2, "y": 15, "tolerance": 0, "sampleAllLayers": true, "antiAlias": false})).unwrap();
    assert_eq!(alpha(&s, 2, 15), 0.0);
    assert_eq!(alpha(&s, 50, 15), 1.0, "the composite band is split by the black square");
    let mut s = session("rgb", 8);
    band(&mut s);
    s.execute("paint.magicEraser", json!({"x": 2, "y": 15, "tolerance": 0, "antiAlias": false})).unwrap();
    assert_eq!(alpha(&s, 50, 15), 0.0, "the layer's own band is one region");

    // A normal layer with locked transparency gets the background colour instead.
    let mut s = session("rgb", 8);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
    s.execute("tools.setColors", json!({"background": "#00ff00"})).unwrap();
    let id = s.active().unwrap().active_layer.unwrap();
    s.edit("lock", |doc, _| {
        doc.layer_mut(id).unwrap().locks.transparency = true;
        Ok(())
    })
    .unwrap();
    s.execute("paint.magicEraser", json!({"x": 2, "y": 2, "antiAlias": false})).unwrap();
    assert_eq!(surface(&s).rgba(2, 2), [0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn magic_eraser_fails_gracefully() {
    let mut s = session("rgb", 8);
    for p in [json!({}), json!({"x": 1}), json!({"x": "a", "y": 1}), json!({"x": 1, "y": 1, "tolerance": "high"}), json!({"x": 1e300, "y": 1})] {
        assert!(s.execute("paint.magicEraser", p.clone()).is_err(), "{p}");
    }
    // Off the canvas: nothing erased, no history step.
    let h = history_len(&s);
    let r = s.execute("paint.magicEraser", json!({"x": -5, "y": 400})).unwrap();
    assert_eq!(r["erased"], false);
    assert_eq!(history_len(&s), h);
    // Out-of-range numbers clamp.
    s.execute("paint.magicEraser", json!({"x": 2, "y": 2, "tolerance": 9999, "opacity": -3})).unwrap();
    // Locked pixels and non-pixel layers are refused.
    let id = s.active().unwrap().active_layer.unwrap();
    s.edit("lock", |doc, _| {
        doc.layer_mut(id).unwrap().locks.pixels = true;
        Ok(())
    })
    .unwrap();
    assert!(s.execute("paint.magicEraser", json!({"x": 2, "y": 2})).is_err());
    let mut e = Session::new();
    assert!(e.execute("paint.magicEraser", json!({"x": 2, "y": 2})).is_err(), "no document");
}

#[test]
fn background_eraser_erases_the_sampled_colour_at_every_depth_and_mode() {
    for mode in MODES {
        for depth in DEPTHS {
            let mut s = session(mode, depth);
            let h = history_len(&s);
            // Start on white, drag across the black square: only white goes.
            s.execute(
                "paint.backgroundEraser",
                json!({"points": [[4, 20], [40, 20]], "size": 16, "hardness": 1, "sampling": "once", "limits": "discontiguous", "tolerance": 10}),
            )
            .unwrap();
            assert_eq!(history_len(&s), h + 1);
            assert_eq!(layer(&s).name, "Layer 0", "{mode} {depth}");
            assert_eq!(alpha(&s, 6, 20), 0.0, "white under the brush erased {mode} {depth}");
            assert_eq!(alpha(&s, 20, 20), 1.0, "black kept {mode} {depth}");
            assert_eq!(alpha(&s, 36, 20), 0.0, "white past the square erased {mode} {depth}");
            assert_eq!(alpha(&s, 6, 40), 1.0, "outside the brush kept {mode} {depth}");
        }
    }
}

#[test]
fn background_eraser_options() {
    // Continuous sampling erases the black square once the hotspot is over it.
    let mut s = session("rgb", 8);
    s.execute(
        "paint.backgroundEraser",
        json!({"points": [[4, 20], [40, 20]], "size": 16, "hardness": 1, "sampling": "continuous", "limits": "discontiguous", "tolerance": 10}),
    )
    .unwrap();
    assert_eq!(alpha(&s, 20, 20), 0.0);
    // Protect Foreground Color (black) keeps the square even under continuous sampling.
    let mut s = session("rgb", 8);
    s.execute("tools.setColors", json!({"foreground": "#000000"})).unwrap();
    s.execute(
        "paint.backgroundEraser",
        json!({"points": [[4, 20], [40, 20]], "size": 16, "hardness": 1, "limits": "discontiguous", "tolerance": 10, "protectForegroundColor": true}),
    )
    .unwrap();
    assert_eq!(alpha(&s, 20, 20), 1.0);
    assert_eq!(alpha(&s, 6, 20), 0.0);
    // Background swatch: with a black swatch only the square goes.
    let mut s = session("rgb", 8);
    s.execute("tools.setColors", json!({"background": "#000000"})).unwrap();
    s.execute(
        "paint.backgroundEraser",
        json!({"points": [[4, 20], [40, 20]], "size": 16, "hardness": 1, "sampling": "backgroundSwatch", "limits": "discontiguous", "tolerance": 10}),
    )
    .unwrap();
    assert_eq!(alpha(&s, 20, 20), 0.0);
    assert_eq!(alpha(&s, 6, 20), 1.0);
}

#[test]
fn background_eraser_fails_gracefully() {
    let mut s = session("rgb", 8);
    for p in [
        json!({}),
        json!({"points": []}),
        json!({"points": "x"}),
        json!({"points": [[1, 1]], "sampling": "sometimes"}),
        json!({"points": [[1, 1]], "limits": 3}),
        json!({"points": [[1, 1]], "tolerance": "lots"}),
        json!({"points": [[1, 1]], "size": 1e9}),
        json!({"points": [[1e300, 1]]}),
    ] {
        assert!(s.execute("paint.backgroundEraser", p.clone()).is_err(), "{p}");
    }
    // A normal layer with locked transparency is refused.
    s.execute("layer.new.layer", json!({})).unwrap();
    let id = s.active().unwrap().active_layer.unwrap();
    s.edit("lock", |doc, _| {
        doc.layer_mut(id).unwrap().locks.transparency = true;
        Ok(())
    })
    .unwrap();
    assert!(s.execute("paint.backgroundEraser", json!({"points": [[4, 4]]})).is_err());
}

/// `cargo test --release -p photocraft-engine --lib bench_magic_eraser -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_magic_eraser_24mp() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 6000, "height": 4000})).unwrap();
    s.execute("select.rect", json!({"x": 1500, "y": 1000, "width": 3000, "height": 2000, "ellipse": true})).unwrap();
    s.execute("edit.fill", json!({"color": "#204080"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    for (label, p) in [
        ("contiguous, background (~19 MP)", json!({"x": 10, "y": 10})),
        ("non-contiguous", json!({"x": 10, "y": 10, "contiguous": false})),
        ("contiguous, ellipse (~4.7 MP)", json!({"x": 3000, "y": 2000})),
    ] {
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t0 = std::time::Instant::now();
            s.execute("paint.magicEraser", p.clone()).unwrap();
            best = best.min(t0.elapsed().as_secs_f64() * 1000.0);
            s.execute("edit.undo", json!({})).unwrap();
        }
        println!("magic eraser 6000x4000 {label}: {best:.1} ms");
    }
    // Phase breakdown.
    let t0 = std::time::Instant::now();
    let (area, img) = crate::selection_cmds::sample_rgba8(&s, false).unwrap();
    let t1 = t0.elapsed().as_secs_f64() * 1000.0;
    let region = sel::wand_region(&img, area, (10, 10), 32.0, true, true).unwrap();
    let t2 = t0.elapsed().as_secs_f64() * 1000.0;
    let mut surf = surface(&s).clone();
    photocraft_algo::erase::magic_erase(&mut surf, &region, 1.0, None, None);
    let t3 = t0.elapsed().as_secs_f64() * 1000.0;
    println!("phases: sample {t1:.1} ms, wand {:.1} ms, erase {:.1} ms", t2 - t1, t3 - t2);
}
