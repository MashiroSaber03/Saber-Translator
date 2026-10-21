use super::*;
use crate::Session;

fn session(depth: u32, background: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 48, "height": 32, "depth": depth, "background": background})).unwrap();
    s
}

fn active_fill(s: &Session) -> Fill {
    let d = s.active().unwrap();
    match &d.doc.layer(d.active_layer.unwrap()).unwrap().content {
        LayerContent::Fill(f) => f.clone(),
        other => panic!("not a fill layer: {}", other.kind_name()),
    }
}

fn layer_count(s: &Session) -> usize {
    s.active().unwrap().doc.layers.len()
}

const STYLES: [&str; 5] = ["linear", "radial", "angle", "reflected", "diamond"];

#[test]
fn create_makes_a_canvas_aligned_fill_layer_with_the_drag_as_handles() {
    let mut s = session(8, "white");
    let r = s
        .execute(
            CREATE,
            json!({"from": [14, 16], "to": [34, 8], "style": "radial", "stops": [[0, "#ff0000"], [1, "#0000ff"]], "opacity": 50, "mode": "multiply"}),
        )
        .unwrap();
    assert_eq!(layer_count(&s), 2);
    let d = s.active().unwrap();
    let l = d.doc.layer(LayerId(r["layer"].as_u64().unwrap())).unwrap();
    assert_eq!(d.active_layer, Some(l.id));
    assert!(l.mask.is_none(), "no selection, no mask");
    assert_eq!(l.opacity, 0.5);
    assert_eq!(l.blend, photocraft_color::BlendMode::Multiply);
    let Fill::Gradient { style, align, dither, .. } = active_fill(&s) else { panic!() };
    assert_eq!(style, GradientStyle::Radial);
    assert!(!align && dither);
    let g = s.execute(GET, json!({})).unwrap();
    let (from, to) = (g["from"].as_array().unwrap(), g["to"].as_array().unwrap());
    for (v, want) in from.iter().chain(to).zip([14.0, 16.0, 34.0, 8.0]) {
        assert!((v.as_f64().unwrap() - want).abs() < 1e-3, "{g}");
    }
    // One undo step removes the layer.
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(layer_count(&s), 1);
}

#[test]
fn create_masks_the_layer_with_the_selection() {
    let mut s = session(8, "white");
    s.execute("select.rect", json!({"x": 4, "y": 4, "width": 10, "height": 8})).unwrap();
    s.execute(CREATE, json!({"from": [0, 0], "to": [48, 0]})).unwrap();
    let d = s.active().unwrap();
    let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
    let m = l.mask.as_ref().expect("selection becomes the mask");
    assert_eq!(m.value(5, 5), 1.0);
    assert_eq!(m.value(30, 20), 0.0);
    // Still laid out on the canvas, not the mask's bounds.
    assert_eq!(photocraft_compose::fill_frame(l, d.doc.bounds()), d.doc.bounds());
    // The composite outside the selection is the white background.
    let flat = photocraft_compose::flatten(&d.doc);
    assert_eq!(flat.px[20 * 48 + 30], [1.0, 1.0, 1.0, 1.0]);
}

/// Pixel parity: a live gradient composites exactly like painting the same drag.
fn parity(depth: u32, background: &str, extra: Value, select: bool, tol: f32) {
    for style in STYLES {
        for (from, to) in [([10.0, 9.0], [34.0, 22.0]), ([30.0, 16.0], [22.0, 3.0])] {
            let mut p = json!({"from": from, "to": to, "style": style});
            for (k, v) in extra.as_object().unwrap() {
                p[k] = v.clone();
            }
            let run = |live: bool| {
                let mut s = session(depth, background);
                if background != "transparent" {
                    s.execute("layer.new.layer", json!({})).unwrap();
                }
                if select {
                    s.execute("select.rect", json!({"x": 6, "y": 3, "width": 30, "height": 22, "ellipse": true, "antiAlias": true})).unwrap();
                }
                // Linear and Reflected fill layers sample each pixel's top-left corner
                // (compose::effects::gradient_t, fitted on Photoshop's fills); the painted
                // gradient samples centres, so its drag is half a pixel further along.
                let mut p = p.clone();
                if !live && matches!(style, "linear" | "reflected") {
                    p["from"] = json!([from[0] + 0.5, from[1] + 0.5]);
                    p["to"] = json!([to[0] + 0.5, to[1] + 0.5]);
                }
                s.execute(if live { CREATE } else { "paint.gradient" }, p).unwrap();
                photocraft_compose::flatten(&s.active().unwrap().doc)
            };
            let (classic, live) = (run(false), run(true));
            let mut worst = 0.0f32;
            for (a, b) in classic.px.iter().zip(&live.px) {
                for c in 0..4 {
                    worst = worst.max((a[c] * a[3] - b[c] * b[3]).abs());
                }
            }
            assert!(worst <= tol, "{style} {from:?}->{to:?} {extra} select={select} depth={depth}: worst {worst}");
        }
    }
}

#[test]
fn live_gradient_renders_like_the_classic_gradient() {
    let stops = json!({"stops": [[0, "#ff2000"], [0.4, "#20c040"], [1, "#1010e0"]], "dither": false});
    parity(32, "transparent", stops.clone(), false, 1e-4);
    parity(8, "transparent", stops.clone(), false, 0.51 / 255.0);
    parity(8, "white", json!({"stops": [[0, "#ff2000"], [1, "#1010e0"]], "reverse": true, "dither": true}), true, 1.01 / 255.0);
    parity(16, "transparent", json!({"gradient": "Foreground to Transparent", "dither": true}), false, 1.01 / 255.0);
    parity(32, "transparent", json!({"gradient": "Black, White", "transparency": [[0, 65], [1, 0]], "dither": false}), false, 1e-4);
}

/// Issue #465: explicit `transparency` stops replace those of a preset or the current gradient.
#[test]
fn create_honours_explicit_transparency_stops() {
    for extra in [json!({"gradient": "Foreground to Transparent"}), json!({"gradient": "black, white"}), json!({})] {
        let mut s = session(8, "transparent");
        let mut p = json!({"from": [0, 0], "to": [48, 0], "transparency": [[1, 0], [0, 65]]});
        for (k, v) in extra.as_object().unwrap() {
            p[k] = v.clone();
        }
        s.execute(CREATE, p).unwrap();
        let Fill::Gradient { opacity_stops, .. } = active_fill(&s) else { panic!("not a gradient fill") };
        assert_eq!(opacity_stops, vec![(0.0, 0.65), (1.0, 0.0)], "{extra}");
    }
}

#[test]
fn set_moves_handles_and_restyles_in_one_undo_step() {
    let mut s = session(8, "white");
    s.execute(CREATE, json!({"from": [4, 4], "to": [40, 4]})).unwrap();
    let before = active_fill(&s);
    let h0 = s.active().unwrap().history.past_len();
    let g = s.execute(SET, json!({"to": [20, 22]})).unwrap();
    assert!((g["to"][0].as_f64().unwrap() - 20.0).abs() < 1e-3 && (g["from"][0].as_f64().unwrap() - 4.0).abs() < 1e-3, "{g}");
    assert_eq!(s.active().unwrap().history.past_len(), h0 + 1);
    // Changing the style alone keeps the handles where they are.
    let g = s.execute(SET, json!({"style": "angle"})).unwrap();
    assert_eq!(g["style"], "angle");
    assert!((g["to"][1].as_f64().unwrap() - 22.0).abs() < 1e-3, "{g}");
    let g = s.execute(SET, json!({"angle": 45, "scale": 80, "offset": [10, -5], "reverse": true, "dither": false, "midpoints": [0.3]})).unwrap();
    assert_eq!(g["angle"], 45.0);
    assert!((g["scale"].as_f64().unwrap() - 80.0).abs() < 1e-3);
    assert_eq!(g["reverse"], true);
    // Each edit is one step: three undos restore the original.
    for _ in 0..3 {
        s.execute("edit.undo", json!({})).unwrap();
    }
    assert_eq!(active_fill(&s), before);
    // A no-op set adds no history.
    let h = s.active().unwrap().history.past_len();
    s.execute(SET, json!({})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len(), h);
}

#[test]
fn locked_gradient_fill_rejects_content_edits_without_history() {
    let mut s = session(8, "white");
    s.execute(CREATE, json!({"from": [0, 0], "to": [16, 0], "stops": [[0, "#000000"], [1, "#ffffff"]], "dither": false})).unwrap();
    s.execute("layer.lockLayers", json!({"all": true})).unwrap();
    let original = active_fill(&s);
    let locked_history = s.active().unwrap().history.past_len();

    for (command, params) in [(SET, json!({"reverse": true})), (STOP, json!({"action": "color", "index": 0, "color": "#ff0000"}))] {
        let result = s.execute(command, params.clone());
        assert!(matches!(&result, Err(EngineError::Other(message)) if message.contains("locked")), "{command} {params}: {result:?}");
        assert_eq!(active_fill(&s), original, "{command} changed the fill");
        assert_eq!(s.active().unwrap().history.past_len(), locked_history, "{command} added history");
    }

    s.execute("layer.lockLayers", json!({"all": false})).unwrap();
    for (command, params) in [(SET, json!({"reverse": true})), (STOP, json!({"action": "color", "index": 0, "color": "#ff0000"}))] {
        let history = s.active().unwrap().history.past_len();
        s.execute(command, params).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), history + 1, "{command} records one step");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(active_fill(&s), original, "undo restores the fill after {command}");
        assert_eq!(s.active().unwrap().history.past_len(), history);
    }
}

#[test]
fn stops_add_move_recolour_delete_and_midpoints() {
    let mut s = session(8, "white");
    s.execute(CREATE, json!({"from": [0, 0], "to": [48, 0], "stops": [[0, "#000000"], [1, "#ffffff"]], "dither": false})).unwrap();
    let stops = |s: &Session| match active_fill(s) {
        Fill::Gradient { stops, midpoints, opacity_stops, .. } => (stops, midpoints, opacity_stops),
        _ => panic!("not a gradient fill"),
    };
    // Add: the new stop takes the colour the gradient has there.
    let g = s.execute(STOP, json!({"action": "add", "location": 0.5})).unwrap();
    assert_eq!(g["stops"].as_array().unwrap().len(), 3);
    let (st, _, _) = stops(&s);
    assert!((st[1].1.c[0] - 0.5).abs() < 1e-3);
    s.execute(STOP, json!({"action": "color", "index": 1, "color": "#ff0000"})).unwrap();
    assert_eq!(stops(&s).0[1].1.to_rgb(), [1.0, 0.0, 0.0]);
    s.execute(STOP, json!({"action": "midpoint", "index": 0, "location": 0.25})).unwrap();
    assert_eq!(stops(&s).1, vec![0.25, 0.5]);
    // Moving a stop past another re-sorts them (the midpoint travels with its stop).
    s.execute(STOP, json!({"action": "move", "index": 0, "location": 0.75})).unwrap();
    let (st, mids, _) = stops(&s);
    assert_eq!(st.iter().map(|x| x.0).collect::<Vec<_>>(), vec![0.5, 0.75, 1.0]);
    assert_eq!(mids, vec![0.5, 0.25]);
    s.execute(STOP, json!({"action": "delete", "index": 2})).unwrap();
    assert_eq!(stops(&s).0.len(), 2);
    assert!(s.execute(STOP, json!({"action": "delete", "index": 0})).is_err(), "keeps 2 stops");
    // Opacity stops start from an opaque pair.
    s.execute(STOP, json!({"action": "opacity", "kind": "opacity", "index": 1, "opacity": 25})).unwrap();
    assert_eq!(stops(&s).2, vec![(0.0, 1.0), (1.0, 0.25)]);
    s.execute(STOP, json!({"action": "add", "kind": "opacity", "location": 0.5})).unwrap();
    assert!((stops(&s).2[1].1 - 0.625).abs() < 1e-4);
    // Each edit is one undo step.
    let n = s.active().unwrap().history.past_len();
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(s.active().unwrap().history.past_len() + 1, n);
    assert_eq!(stops(&s).2.len(), 2);
}

#[test]
fn preview_equals_the_committed_edit() {
    let mut s = session(8, "white");
    s.execute(CREATE, json!({"from": [4, 4], "to": [40, 20], "style": "diamond"})).unwrap();
    let d = s.active().unwrap();
    let l = d.doc.layer(d.active_layer.unwrap()).unwrap().clone();
    let LayerContent::Fill(f) = &l.content else { panic!() };
    let p = json!({"from": [10, 12], "to": [30, 2]});
    let preview = apply_set(&l, f, d.doc.bounds(), &p, [0.0; 4], [1.0; 4]).unwrap();
    s.execute(SET, p).unwrap();
    assert_eq!(active_fill(&s), preview);
}

#[test]
fn classic_gradient_still_paints_pixels() {
    let mut s = session(8, "white");
    s.execute("paint.gradient", json!({"from": [0, 0], "to": [48, 0], "colors": ["#000000", "#ffffff"], "dither": false})).unwrap();
    assert_eq!(layer_count(&s), 1, "no fill layer");
    let d = s.active().unwrap();
    let px = d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().pixel(1, 5);
    assert!(px[0] < 0.1);
}

#[test]
fn commands_fail_gracefully() {
    let mut empty = Session::new();
    for id in [CREATE, SET, STOP, GET] {
        assert!(empty.execute(id, json!({"from": [0, 0], "to": [1, 1]})).is_err(), "{id} without a document");
    }
    let mut s = session(8, "white");
    // Not a gradient fill layer.
    for id in [SET, STOP, GET] {
        assert!(s.execute(id, json!({})).is_err(), "{id} on a pixel layer");
    }
    for p in [
        json!({}),
        json!({"from": [0, 0]}),
        json!({"from": [0], "to": [1, 1]}),
        json!({"from": [f64::MAX, 0], "to": [1, 1]}),
        json!({"from": ["a", 0], "to": [1, 1]}),
        json!({"from": [0, 0], "to": [1, 1], "style": "spiral"}),
        json!({"from": [0, 0], "to": [1, 1], "style": 3}),
        json!({"from": [0, 0], "to": [1, 1], "gradient": "No Such Preset"}),
        json!({"from": [0, 0], "to": [1, 1], "stops": []}),
        json!({"from": [0, 0], "to": [1, 1], "mode": "nonsense"}),
    ] {
        assert!(s.execute(CREATE, p.clone()).is_err(), "create {p}");
    }
    // A zero-length drag still makes a valid gradient.
    s.execute(CREATE, json!({"from": [3, 3], "to": [3, 3]})).unwrap();
    for p in [
        json!({"layer": u64::MAX}),
        json!({"from": [0]}),
        json!({"to": [1e30, 0]}),
        json!({"angle": "x"}),
        json!({"scale": 0}),
        json!({"scale": -5}),
        json!({"offset": [1]}),
        json!({"offset": ["a", "b"]}),
        json!({"reverse": 1}),
        json!({"stops": [[0, "#000000"]]}),
        json!({"stops": [[0, "#zzzzzz"], [1, "#ffffff"]]}),
        json!({"stops": [[0, [1, 2]], [1, "#ffffff"]]}),
        json!({"stops": "x"}),
        json!({"transparency": [[0]]}),
        json!({"midpoints": [0.5, 0.5, 0.5]}),
        json!({"midpoints": ["x"]}),
        json!({"style": "spiral"}),
        json!({"align": "no"}),
    ] {
        assert!(s.execute(SET, p.clone()).is_err(), "set {p}");
    }
    for p in [
        json!({}),
        json!({"action": "explode"}),
        json!({"action": "add"}),
        json!({"action": "move", "index": 99, "location": 0.5}),
        json!({"action": "delete", "index": 99}),
        json!({"action": "color", "index": 0}),
        json!({"action": "color", "index": 0, "color": 7}),
        json!({"action": "midpoint", "index": 1, "location": 0.5}),
        json!({"action": "midpoint", "index": 0}),
        json!({"action": "add", "kind": "sparkle", "location": 0.5}),
        json!({"action": "color", "kind": "opacity", "index": 0}),
        json!({"action": "opacity", "kind": "opacity", "index": 7, "opacity": 3}),
        json!({"action": "delete", "kind": "opacity", "index": 0}),
        json!({"action": "move", "index": -1, "location": 0.5}),
    ] {
        assert!(s.execute(STOP, p.clone()).is_err(), "stop {p}");
    }
    // Huge-but-finite values clamp instead of failing.
    s.execute(SET, json!({"scale": 1e30, "angle": 1e9, "offset": [1e4, -1e4]})).unwrap();
    let g = s.execute(GET, json!({})).unwrap();
    assert!(g["scale"].as_f64().unwrap().is_finite() && g["from"][0].as_f64().unwrap().is_finite());
    // Scale stays within Photoshop's 10–150 %, also for a handle dragged far away.
    assert!((g["scale"].as_f64().unwrap() - 150.0).abs() < 1e-3, "{g}");
    let g = s.execute(SET, json!({"from": [0, 0], "to": [1e6, 0]})).unwrap();
    assert!((g["scale"].as_f64().unwrap() - 150.0).abs() < 1e-3, "{g}");
    let g = s.execute(SET, json!({"scale": 1})).unwrap();
    assert!((g["scale"].as_f64().unwrap() - 10.0).abs() < 1e-3, "{g}");
}

/// Dragging one handle past the scale range stops it; the other handle stays put (#1243).
#[test]
fn a_clamped_handle_drag_leaves_the_other_handle_alone() {
    let pt = |g: &Value, k: &str| [g[k][0].as_f64().unwrap(), g[k][1].as_f64().unwrap()];
    let close = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-2 && (a[1] - b[1]).abs() < 1e-2;
    for style in STYLES {
        for (from, to, short, short_start) in [([14.0, 16.0], [34.0, 16.0], [15.0, 16.0], [33.0, 16.0]), ([24.0, 6.0], [24.0, 26.0], [24.0, 7.0], [24.0, 25.0])]
        {
            let mut s = session(8, "white");
            s.execute(CREATE, json!({"from": from, "to": to, "style": style})).unwrap();
            // End dragged almost onto the start: far below 10 %.
            let g = s.execute(SET, json!({"to": short})).unwrap();
            assert!((g["scale"].as_f64().unwrap() - 10.0).abs() < 1e-3, "{style}: {g}");
            assert!(close(pt(&g, "from"), from), "{style}: start moved to {:?}", pt(&g, "from"));
            // The end stops on the line towards the pointer.
            let e = pt(&g, "to");
            assert!((e[0] - from[0]) * (short[0] - from[0]) >= 0.0 && (e[1] - from[1]) * (short[1] - from[1]) >= 0.0, "{style}: {e:?}");
            // And the same for the start handle.
            let mut s = session(8, "white");
            s.execute(CREATE, json!({"from": from, "to": to, "style": style})).unwrap();
            let g = s.execute(SET, json!({"from": short_start})).unwrap();
            assert!(close(pt(&g, "to"), to), "{style}: end moved: {g}");
        }
    }
}

/// Issue #958: a stop or segment index far past the end (up to the largest integer) is a
/// `BadParams` error for every action and kind, never an overflow panic, and leaves the gradient
/// and the history as they were.
#[test]
fn stop_rejects_huge_indices_without_panicking() {
    let mut s = session(8, "white");
    s.execute(CREATE, json!({"from": [0, 0], "to": [8, 8], "stops": [[0, "#000000"], [1, "#ffffff"]]})).unwrap();
    let before = active_fill(&s);
    let h = s.active().unwrap().history.past_len();
    for index in [u64::MAX, usize::MAX as u64, u64::MAX - 1, 1 << 32, 2] {
        for p in [
            json!({"action": "midpoint", "index": index, "location": 0.5}),
            json!({"action": "move", "index": index, "location": 0.5}),
            json!({"action": "delete", "index": index}),
            json!({"action": "color", "index": index, "color": "#ff0000"}),
            json!({"action": "move", "kind": "opacity", "index": index, "location": 0.5}),
            json!({"action": "delete", "kind": "opacity", "index": index}),
            json!({"action": "opacity", "kind": "opacity", "index": index, "opacity": 50}),
        ] {
            let r = s.execute(STOP, p.clone());
            assert!(matches!(r, Err(EngineError::BadParams { .. })), "{p}: {r:?}");
            assert_eq!(active_fill(&s), before, "{p} changed the gradient");
            assert_eq!(s.active().unwrap().history.past_len(), h, "{p} added a history step");
        }
    }
    // The last segment of a two-stop gradient is 0; segment 1 does not exist.
    let r = s.execute(STOP, json!({"action": "midpoint", "index": 1, "location": 0.5}));
    assert!(matches!(&r, Err(EngineError::BadParams { msg, .. }) if msg.contains("no segment 1 (there are 1)")), "{r:?}");
    // Control: a valid midpoint edit still applies as one undo step.
    s.execute(STOP, json!({"action": "midpoint", "index": 0, "location": 0.25})).unwrap();
    let Fill::Gradient { midpoints, .. } = active_fill(&s) else { panic!("not a gradient fill") };
    assert_eq!(midpoints, vec![0.25]);
    assert_eq!(s.active().unwrap().history.past_len(), h + 1);
}
