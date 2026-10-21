//! #155: Properties sections share one header style, Quick Actions fit the layer kind, controls
//! fill the panel at every dock width, and the Layers panel's Opacity/Fill rows keep their gap.

use egui::{Rect, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use photocraft_doc::{Layer, LayerContent};
use serde_json::json;

use super::*;
use crate::PhotocraftApp;

/// A document with one layer of each kind; returns the session and the ids by kind.
fn kinds() -> (photocraft_engine::Session, Vec<(&'static str, u64)>) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
    let id = |v: serde_json::Value| v["layer"].as_u64().unwrap();
    let pixel = id(s.execute("layer.new.layer", json!({"name": "Paint"})).unwrap());
    s.execute("select.rect", json!({"x": 10, "y": 10, "width": 100, "height": 60})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#336699"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let shape = id(s.execute("shape.create", json!({"kind": "rect", "rect": [20, 30, 80, 40], "fill": "#ff0000"})).unwrap());
    let ty = id(s.execute("type.create", json!({"text": "Headline", "size": 48, "x": 40, "y": 200})).unwrap());
    (s, vec![("pixel", pixel), ("shape", shape), ("type", ty)])
}

fn layer_of(app: &PhotocraftApp, id: u64) -> Layer {
    app.session.active().unwrap().doc.layer(photocraft_doc::LayerId(id)).unwrap().clone()
}

/// Pro theme with only the Properties group expanded (it then fills the column).
fn harness(session: photocraft_engine::Session, ppp: f32, dock_width: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 1600.0)).with_pixels_per_point(ppp).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(session, crate::Services::default())
    });
    set(&mut h, json!({"dockWidth": dock_width, "dock": {"collapsed": ["color", "navigator", "history", "layers"]}}));
    h
}

fn set(h: &mut Harness<'static, PhotocraftApp>, p: serde_json::Value) {
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new("ui.set", p);
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(8);
}

fn select(h: &mut Harness<'static, PhotocraftApp>, id: u64) {
    h.state_mut().run("layer.select", json!({"layer": id})).unwrap();
    h.run_steps(8);
}

fn headers(h: &Harness<'_, PhotocraftApp>) -> Vec<(String, Rect)> {
    sections_drawn(&h.ctx)
}

#[test]
fn quick_actions_fit_the_layer_kind() {
    let ids = |c: &LayerContent| quick_actions(c).iter().map(|(_, id)| *id).collect::<Vec<_>>();
    let (s, layers) = kinds();
    let app = PhotocraftApp::new(s, crate::Services::default());
    for (kind, id) in layers {
        let l = layer_of(&app, id);
        let all = ids(&l.content);
        assert!(!all.is_empty(), "{kind}");
        match kind {
            "pixel" => {
                assert!(all.contains(&"select.subject"));
                assert!(all.contains(&"layer.removeBackground"));
            }
            _ => assert!(!all.contains(&"select.subject"), "{kind}: Select Subject is for pixel layers"),
        }
        for id in &all {
            assert!(crate::menus::is_live(id), "{kind}: {id} is not a command");
        }
    }
    let ty = app.session.active().unwrap().doc.layers.iter().find(|l| matches!(l.content, LayerContent::Text(_))).unwrap().content.clone();
    assert_eq!(ids(&ty), ["type.convertToShape", "type.convertToParagraphText", "type.convertToPointText", "type.warpText", "type.rasterizeTypeLayer"]);
    assert!(ids(&LayerContent::Adjustment(photocraft_doc::Adjustment::Invert)).is_empty());
}

#[test]
fn type_layer_sections_share_one_header_style_and_actions_fit() {
    let (s, layers) = kinds();
    let ty = layers.iter().find(|(k, _)| *k == "type").unwrap().1;
    let mut h = harness(s, 1.0, 320.0);
    select(&mut h, ty);
    let hs = headers(&h);
    let names: Vec<&str> = hs.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["Transform", "Align and Distribute", "Character", "Paragraph", "Type Options", "Quick Actions"]);
    let first = hs[0].1;
    for (n, r) in &hs {
        assert!((r.left() - first.left()).abs() < 0.5 && (r.width() - first.width()).abs() < 0.5, "{n}: {r:?} vs {first:?}");
        assert!((r.height() - SECTION_H).abs() < 0.5, "{n}: {r:?}");
    }
    // The header names the layer.
    assert!(h.query_by_value("Headline — Type Layer").is_some());
    // Only type-relevant Quick Actions (point type: Convert to Paragraph Text, not Point Text).
    let app = h.state();
    let shown: Vec<&str> = visible_quick_actions(app, &layer_of(app, ty).content).iter().map(|(l, _)| *l).collect();
    assert_eq!(shown, ["Convert to Shape", "Convert to Paragraph Text", "Create Warped Text", "Rasterize Type"]);
    for label in &shown {
        assert!(h.query_by_label(label).is_some(), "{label} button");
    }
    assert!(h.query_by_label("Select Subject").is_none());
}

#[test]
fn pixel_layer_shows_remove_background_quick_action() {
    let (s, layers) = kinds();
    let pixel = layers.iter().find(|(k, _)| *k == "pixel").unwrap().1;
    let mut h = harness(s, 1.0, 320.0);
    select(&mut h, pixel);
    let app = h.state();
    let shown: Vec<&str> = visible_quick_actions(app, &layer_of(app, pixel).content).iter().map(|(_, id)| *id).collect();
    assert!(shown.contains(&"layer.removeBackground"), "{shown:?}");
    assert!(crate::menus::is_live("layer.removeBackground"));
    assert!(h.query_by_label("Remove Background").is_some());
}

#[test]
fn other_kinds_use_the_same_headers() {
    let (s, layers) = kinds();
    let mut h = harness(s, 1.0, 320.0);
    for (kind, id) in layers {
        select(&mut h, id);
        let hs = headers(&h);
        let names: Vec<&str> = hs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.first(), Some(&"Transform"), "{kind}: {names:?}");
        assert_eq!(names.last(), Some(&"Quick Actions"), "{kind}: {names:?}");
        if kind == "shape" {
            assert!(names.contains(&"Appearance") && names.contains(&"Shape"), "{names:?}");
        }
        let first = hs[0].1;
        assert!(hs.iter().all(|(_, r)| (r.left() - first.left()).abs() < 0.5 && (r.height() - SECTION_H).abs() < 0.5), "{kind}: {hs:?}");
    }
}

#[test]
fn type_controls_fill_the_panel_at_every_width_and_scale() {
    for ppp in [1.0, 2.0] {
        for dw in [250.0, 380.0, 520.0] {
            let (s, layers) = kinds();
            let ty = layers.iter().find(|(k, _)| *k == "type").unwrap().1;
            let mut h = harness(s, ppp, dw);
            select(&mut h, ty);
            let hs = headers(&h);
            let header = hs.iter().find(|(n, _)| n == "Character").unwrap().1;
            let what = format!("{ppp}x @ {dw}");
            // Font family and style dropdowns span the section.
            let combos: Vec<Rect> = h
                .query_all_by_role(egui::accesskit::Role::ComboBox)
                .map(|n| n.rect())
                .filter(|r| r.top() >= header.bottom() && r.left() >= header.left() - 1.0)
                .collect();
            assert!(combos.len() >= 2, "{what}: {combos:?}");
            for r in &combos[..2] {
                assert!((r.left() - header.left()).abs() <= 2.0 && (r.right() - header.right()).abs() <= 2.0, "{what}: {r:?} vs {header:?}");
            }
            // Number fields end at the section's right edge (the second column), never past it.
            let fields: Vec<Rect> = h
                .query_all_by_role(egui::accesskit::Role::SpinButton)
                .map(|n| n.rect())
                .filter(|r| r.top() >= header.bottom() && r.left() >= header.left() - 1.0)
                .collect();
            assert!(fields.len() >= 6, "{what}: {} fields", fields.len());
            let right = fields.iter().map(|r| r.right()).fold(f32::MIN, f32::max);
            assert!(right <= header.right() + 0.5 && right >= header.right() - 30.0, "{what}: fields end at {right}, panel {}", header.right());
            // Transform's second column ends where Character's does (one right edge for the panel).
            let transform = hs[0].1;
            let t_right = h
                .query_all_by_role(egui::accesskit::Role::SpinButton)
                .map(|n| n.rect())
                .filter(|r| r.top() >= transform.bottom() && r.bottom() <= header.top())
                .map(|r| r.right())
                .fold(f32::MIN, f32::max);
            // (Both rows have a unit suffix: compare with the size/leading row.)
            // Variable-font axes can precede the size/leading row and have no unit suffix.
            // Compare the intended point-valued controls, retaining the same edge tolerance.
            let lead = ["Font size", "Leading (set to the font size × auto-leading when Auto)"]
                .into_iter()
                .map(|label| h.get_by_role_and_label(egui::accesskit::Role::SpinButton, label).rect().right())
                .fold(f32::MIN, f32::max);
            assert!((t_right - lead).abs() < 0.5, "{what}: Transform fields end at {t_right}, Character at {lead}");
            // Quick Actions buttons span the panel too.
            let qa = h.get_by_label("Rasterize Type").rect();
            assert!(qa.right() >= header.right() - 2.0 && qa.right() <= header.right() + 1.0, "{what}: {qa:?}");
        }
    }
}

#[test]
fn layers_opacity_and_fill_rows_keep_a_gap() {
    for dw in [250.0, 520.0] {
        let (s, layers) = kinds();
        let mut h = Harness::builder().with_size(vec2(1440.0, 1000.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            PhotocraftApp::new(s, crate::Services::default())
        });
        set(&mut h, json!({"dockWidth": dw}));
        select(&mut h, layers[0].1);
        let o = h.query_all_by_label("Opacity:").next().unwrap().rect();
        let f = h.query_all_by_label("Fill:").next().unwrap().rect();
        // Rows are 24 pt high: centres at least a row plus the gap apart.
        assert!(f.center().y - o.center().y >= 24.0 + crate::theme::ROW_GAP - 0.5, "{dw}: {o:?} {f:?}");
    }
}

#[test]
fn field_width_divides_the_row() {
    for avail in [200.0, 300.0, 500.0] {
        let w = field_width(avail, 2, LABEL_W);
        assert!((2.0 * (LABEL_W + LABEL_GAP + w) + COL_GAP - avail).abs() < 1e-3);
    }
    assert_eq!(field_width(10.0, 3, LABEL_W), 36.0);
    assert_eq!(field_width(f32::NAN, 0, LABEL_W), 36.0);
}

#[test]
fn character_and_paragraph_dock_panels_share_the_headers_without_type_options() {
    let (s, layers) = kinds();
    let (ty, px) = (layers.iter().find(|(k, _)| *k == "type").unwrap().1, layers[0].1);
    let mut h = harness(s, 1.0, 320.0);
    // Only the Window › Character group, expanded.
    set(&mut h, json!({"panels": {"character": true}, "dock": {"collapsed": ["color", "navigator", "history", "layers", "properties"]}}));
    select(&mut h, ty);
    for (tab, want) in [(0, "Character"), (1, "Paragraph")] {
        set(&mut h, json!({"dockTabs": {"character": tab}}));
        let hs = headers(&h);
        assert!(hs.iter().all(|(_, r)| (r.height() - SECTION_H).abs() < 0.5), "{hs:?}");
        let names: Vec<&str> = hs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, [want], "tab {tab}");
    }
    // No type layer: the empty-state message, no sections.
    select(&mut h, px);
    assert!(headers(&h).is_empty());
    assert!(h.query_by_value("Select a type layer to edit its character and paragraph settings.").is_some());
}

/// #2085: like Photoshop, Opacity and Fill stack in one right-aligned column: both fields end at
/// the same edge, and both labels end the same gap before them (Fill's label used to touch its field).
#[test]
fn layers_opacity_and_fill_share_one_right_column() {
    for theme in ["pro", "studioLight"] {
        let (s, layers) = kinds();
        let mut h = Harness::builder().with_size(vec2(1440.0, 1000.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            PhotocraftApp::new(s, crate::Services::default())
        });
        set(&mut h, json!({"theme": theme}));
        select(&mut h, layers[0].1);
        // The popup-slider arrow is named after the field; its label is the text left of it on its row.
        let row = |name: &str| {
            let arrow = h.query_all_by_label(name).find(|n| n.accesskit_node().role() == egui::accesskit::Role::Button).unwrap().rect();
            let text = if theme == "pro" { format!("{name}:") } else { name.to_string() };
            let label = h
                .query_all_by_label(&text)
                .filter(|n| n.accesskit_node().role() == egui::accesskit::Role::Label)
                .map(|n| n.rect())
                .find(|r| (r.center().y - arrow.center().y).abs() < 12.0 && r.right() <= arrow.left())
                .unwrap();
            (label, arrow)
        };
        let (ol, oa) = row("Opacity");
        let (fl, fa) = row("Fill");
        assert!((oa.right() - fa.right()).abs() < 0.5, "{theme}: fields end at {} and {}", oa.right(), fa.right());
        assert!((ol.right() - fl.right()).abs() < 0.5, "{theme}: Opacity label ends at {}, Fill label at {}", ol.right(), fl.right());
    }
}

#[test]
fn properties_distribute_buttons_move_selected_layers_and_undo_in_both_axes() {
    use photocraft_doc::LayerId;
    for horizontal in [true, false] {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 300, "height": 300, "background": "transparent"})).unwrap();
        let mut ids = Vec::new();
        for (i, at) in [10, 40, 160].into_iter().enumerate() {
            let id = s.execute("layer.new.layer", json!({"name": format!("Block {i}")})).unwrap()["layer"].as_u64().unwrap();
            let (x, y) = if horizontal { (at, 10) } else { (10, at) };
            s.execute("select.rect", json!({"x": x, "y": y, "width": 10, "height": 10})).unwrap();
            s.execute("edit.fill", json!({"contents": "color", "color": "#336699"})).unwrap();
            s.execute("select.deselect", json!({})).unwrap();
            ids.push(id);
        }
        for (i, id) in ids.iter().copied().enumerate() {
            s.execute("layer.select", json!({"layer": id, "mode": if i == 0 { "replace" } else { "add" }})).unwrap();
        }
        let mut h = harness(s, 1.0, 320.0);
        let button = if horizontal { "Horizontally" } else { "Vertically" };
        // An enabled Properties button invokes the existing command (not a UI-only gesture).
        let before = {
            let st = h.state().session.active().unwrap();
            let b = st.doc.layer(LayerId(ids[1])).unwrap().surface().unwrap().content_bounds();
            if horizontal { b.x0 } else { b.y0 }
        };
        let history = h.state().session.active().unwrap().history.entries().len();
        h.get_by_label(button).click();
        h.run_steps(4);
        let after = {
            let st = h.state().session.active().unwrap();
            let b = st.doc.layer(LayerId(ids[1])).unwrap().surface().unwrap().content_bounds();
            if horizontal { b.x0 } else { b.y0 }
        };
        assert_ne!(after, before, "{button} moved the middle layer");
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), history + 1);
        h.state_mut().session.undo();
        let restored = {
            let st = h.state().session.active().unwrap();
            let b = st.doc.layer(LayerId(ids[1])).unwrap().surface().unwrap().content_bounds();
            if horizontal { b.x0 } else { b.y0 }
        };
        assert_eq!(restored, before, "{button}: undo restores its location");
    }
}

/// #4608: with the Background one of several selected layers (and the active one), Properties
/// shows the Document properties as before, with Align and Distribute above them, and its buttons
/// align the selection: the Background stays, the other layer lines up with it.
#[test]
fn background_in_a_multiple_selection_keeps_align_and_distribute() {
    use photocraft_doc::LayerId;
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 300, "height": 200})).unwrap();
    let bg = s.active().unwrap().doc.layers[0].id.0;
    let block = s.execute("layer.new.layer", json!({"name": "Block"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("select.rect", json!({"x": 20, "y": 30, "width": 40, "height": 20})).unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#336699"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    // Background alone: the Document properties only.
    s.execute("layer.select", json!({"layer": bg})).unwrap();
    let mut h = harness(s, 1.0, 320.0);
    assert!(h.query_by_label("Align Horizontal Centers").is_none(), "one layer: Document properties only");
    // The Background is still the active layer once Block joins the selection.
    h.state_mut().run("layer.select", json!({"layer": block, "mode": "add"})).unwrap();
    h.state_mut().run("layer.select", json!({"layer": bg, "mode": "add"})).unwrap();
    h.run_steps(8);
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(LayerId(bg)));
    assert_eq!(h.state().session.active().unwrap().selected_layers().len(), 2);
    let names: Vec<String> = headers(&h).into_iter().map(|(n, _)| n).collect();
    assert!(names.iter().any(|n| n == "Align and Distribute"), "{names:?}");
    h.get_by_label("Align Horizontal Centers").click();
    h.run_steps(4);
    let st = h.state().session.active().unwrap();
    let b = st.doc.layer(LayerId(block)).unwrap().surface().unwrap().content_bounds();
    assert_eq!((b.x0 + b.x1) / 2, 150, "centred on the Background, which spans the canvas: {b:?}");
    assert_eq!(b.y0, 30, "only horizontally");
}
