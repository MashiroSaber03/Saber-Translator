//! Merging small layers gives a layer as big as their content in every colour mode. In CMYK and
//! Lab, transparent pixels used to keep the colour of converted transparent black, so Merge
//! Layers, Merge Visible, Stamp Visible and Stamp Down made layers that spanned the whole canvas:
//! Free Transform, the Move tool and alignment then worked on the canvas instead of the content.

use photocraft_doc::{Layer, LayerId};
use photocraft_engine::Session;
use photocraft_engine::transform_cmds::transform_bounds;
use photocraft_geom::Rect;
use serde_json::{Value, json};

const W: u32 = 300;
const H: u32 = 400;
/// Two small shapes far from the canvas edges, and the box around both.
const A: [i32; 4] = [40, 300, 20, 30];
const B: [i32; 4] = [90, 320, 25, 20];
const BOTH: Rect = Rect { x0: 40, y0: 300, x1: 115, y1: 340 };

/// Every mode at the depths it supports (Photoshop has no 32-bit CMYK or Lab).
const CASES: &[(&str, u32)] =
    &[("rgb", 8), ("rgb", 16), ("rgb", 32), ("gray", 8), ("gray", 16), ("gray", 32), ("cmyk", 8), ("cmyk", 16), ("lab", 8), ("lab", 16)];

fn id_of(r: &Value) -> LayerId {
    LayerId(r["layer"].as_u64().unwrap())
}

/// A transparent document with the two shapes on their own layers.
fn scene(mode: &str, depth: u32) -> (Session, LayerId, LayerId) {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": W, "height": H, "mode": mode, "depth": depth, "background": "transparent"})).unwrap();
    let a = id_of(&s.execute("shape.create", json!({"kind": "rect", "rect": A, "fill": "#e07020"})).unwrap());
    let b = id_of(&s.execute("shape.create", json!({"kind": "rect", "rect": B, "fill": "#3060c0"})).unwrap());
    (s, a, b)
}

fn select(s: &mut Session, ids: &[LayerId]) {
    s.execute("layer.select", json!({"layer": ids[0].0})).unwrap();
    for id in &ids[1..] {
        s.execute("layer.select", json!({"layer": id.0, "mode": "add"})).unwrap();
    }
}

fn active(s: &Session) -> &Layer {
    let st = s.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap()
}

/// The merged layer hugs the shapes (anti-aliased edges may add a pixel) and still shows them.
fn assert_tight(s: &Session, what: &str) {
    let doc = &s.active().unwrap().doc;
    let layer = active(s);
    let b = transform_bounds(doc, layer);
    let grown = Rect::new(BOTH.x0 - 1, BOTH.y0 - 1, BOTH.x1 + 1, BOTH.y1 + 1);
    assert!(grown.contains_rect(&b) && b.contains_rect(&Rect::new(BOTH.x0 + 1, BOTH.y0 + 1, BOTH.x1 - 1, BOTH.y1 - 1)), "{what}: bounds {b:?}, want ~{BOTH:?}");
    let surf = layer.surface().unwrap();
    // Inside each shape: opaque. Between them and far away: transparent.
    assert!(surf.rgba(A[0] + 5, A[1] + 5)[3] > 0.99, "{what}: first shape kept");
    assert!(surf.rgba(B[0] + 5, B[1] + 5)[3] > 0.99, "{what}: second shape kept");
    assert!(surf.rgba(70, 310)[3] < 0.01, "{what}: gap stays transparent");
    assert!(surf.rgba(5, 5)[3] < 0.01, "{what}: corner stays transparent");
}

#[test]
fn merge_layers_keeps_content_bounds_in_every_mode() {
    for &(mode, depth) in CASES {
        let (mut s, a, b) = scene(mode, depth);
        select(&mut s, &[a, b]);
        s.execute("layer.mergeLayers", json!({})).unwrap();
        assert_tight(&s, &format!("Merge Layers {mode} {depth}"));
    }
}

#[test]
fn merge_visible_keeps_content_bounds_in_every_mode() {
    for &(mode, depth) in CASES {
        let (mut s, _, _) = scene(mode, depth);
        s.execute("layer.mergeVisible", json!({})).unwrap();
        assert_tight(&s, &format!("Merge Visible {mode} {depth}"));
    }
}

#[test]
fn stamps_keep_content_bounds_in_every_mode() {
    for &(mode, depth) in CASES {
        let (mut s, _, _) = scene(mode, depth);
        s.execute("layer.stampVisible", json!({})).unwrap();
        assert_tight(&s, &format!("Stamp Visible {mode} {depth}"));

        let (mut s, a, b) = scene(mode, depth);
        select(&mut s, &[a, b]);
        s.execute("layer.stampDown", json!({})).unwrap();
        assert_tight(&s, &format!("Stamp Down {mode} {depth}"));
    }
}

/// A fill layer may draw anywhere, so merging one renders the whole canvas; its mask still
/// limits the merged layer to the shapes.
#[test]
fn merging_a_masked_fill_layer_keeps_content_bounds_in_every_mode() {
    let masked_scene = |mode: &str, depth: u32| {
        let (mut s, a, _) = scene(mode, depth);
        s.execute("select.rect", json!({"x": B[0], "y": B[1], "width": B[2], "height": B[3]})).unwrap();
        let fill = id_of(&s.execute("layer.newFillLayer.solidColor", json!({"color": "#3060c0"})).unwrap());
        s.execute("select.deselect", json!({})).unwrap();
        (s, a, fill)
    };
    for &(mode, depth) in CASES {
        let (mut s, a, fill) = masked_scene(mode, depth);
        select(&mut s, &[a, fill]);
        s.execute("layer.mergeLayers", json!({})).unwrap();
        assert_tight(&s, &format!("Merge Layers with a fill layer {mode} {depth}"));

        let (mut s, _, _) = masked_scene(mode, depth);
        s.execute("layer.mergeVisible", json!({})).unwrap();
        assert_tight(&s, &format!("Merge Visible with a fill layer {mode} {depth}"));
    }
}

/// Over an opaque Background, Merge Visible still makes an opaque, canvas-sized Background.
#[test]
fn merge_visible_onto_a_background_stays_opaque() {
    for &(mode, depth) in CASES {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": W, "height": H, "mode": mode, "depth": depth, "background": "white"})).unwrap();
        s.execute("shape.create", json!({"kind": "rect", "rect": A, "fill": "#e07020"})).unwrap();
        s.execute("layer.mergeVisible", json!({})).unwrap();
        let what = format!("{mode} {depth}");
        let doc = &s.active().unwrap().doc;
        let layer = active(&s);
        assert_eq!(layer.name, "Background", "{what}");
        assert_eq!(transform_bounds(doc, layer), Rect::new(0, 0, W as i32, H as i32), "{what}");
        let white = layer.surface().unwrap().rgba(5, 5);
        assert!(white.iter().all(|c| *c > 0.98), "{what}: Background stays white, got {white:?}");
    }
}

/// Undo restores both layers after the fixed merge.
#[test]
fn merge_layers_in_cmyk_is_one_undo_step() {
    let (mut s, a, b) = scene("cmyk", 8);
    select(&mut s, &[a, b]);
    s.execute("layer.mergeLayers", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    let doc = &s.active().unwrap().doc;
    assert!(doc.layer(a).is_some() && doc.layer(b).is_some());
}
