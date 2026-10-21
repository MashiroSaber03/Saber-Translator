//! #314: renaming a layer in place works like Photoshop. A click anywhere else, Enter or Tab
//! commits; Esc cancels; only one rename is open at a time, and starting another (double-click,
//! the context menu or Layer › Rename Layer) commits the first. Tab then renames the layer below,
//! ⇧Tab the layer above.

use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2, vec2};
use egui_kittest::Harness;
use serde_json::json;

use super::{recorded, rename, renaming};
use crate::PhotocraftApp;

/// A document with layers "Alpha", "Beta" and "Gamma" (Gamma on top and active).
fn session() -> (photocraft_engine::Session, [u64; 3]) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    let ids = ["Alpha", "Beta", "Gamma"].map(|n| s.execute("layer.new.layer", json!({"name": n})).unwrap()["layer"].as_u64().unwrap());
    (s, ids)
}

fn harness(ppp: f32) -> (Harness<'static, PhotocraftApp>, [u64; 3]) {
    let (s, ids) = session();
    let mut h =
        Harness::builder().with_size(vec2(1440.0, 900.0)).with_pixels_per_point(ppp).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            PhotocraftApp::new(s, crate::Services::default())
        });
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(8);
    (h, ids)
}

fn name(h: &Harness<'_, PhotocraftApp>, layer: u64) -> String {
    h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(layer)).unwrap().name.clone()
}

/// Where `layer`'s name is drawn in the Layers panel.
fn name_at(h: &Harness<'_, PhotocraftApp>, layer: u64) -> Pos2 {
    let row = recorded(&h.ctx).into_iter().find(|r| r.layer == layer).expect("row drawn");
    row.name.map_or(row.row.center(), |r| r.center())
}

fn button(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, pressed: bool) {
    h.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
}

fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    h.run_steps(1);
    button(h, at, true);
    h.run_steps(1);
    button(h, at, false);
    h.run_steps(3);
}

fn double_click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2) {
    h.hover_at(at);
    // Two clicks 2 frames apart (60 fps) are a double click; egui counts a click within 0.6 s of
    // the click before the last as a triple click, so let the clock move past earlier clicks.
    for _ in 0..40 {
        h.step();
    }
    for _ in 0..2 {
        button(h, at, true);
        h.step();
        button(h, at, false);
        h.step();
    }
    h.run_steps(3);
}

fn key(h: &mut Harness<'_, PhotocraftApp>, k: Key) {
    h.event(Event::Key { key: k, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.event(Event::Key { key: k, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
}

fn type_text(h: &mut Harness<'_, PhotocraftApp>, text: &str) {
    h.event(Event::Text(text.into()));
    h.run_steps(2);
}

/// Is `layer`'s rename field on screen (drawn last frame)?
fn field_shown(h: &Harness<'_, PhotocraftApp>, layer: u64) -> bool {
    h.ctx.read_response(egui::Id::new(("layer-rename-field", layer))).is_some()
}

/// A point on the canvas, away from every panel.
fn canvas(h: &Harness<'_, PhotocraftApp>) -> Pos2 {
    let r = h.ctx.content_rect();
    pos2(r.width() * 0.4, r.height() * 0.5)
}

/// Start renaming `layer` by double-clicking its name; the name is selected, so typing replaces it.
fn start(h: &mut Harness<'_, PhotocraftApp>, layer: u64) {
    let at = name_at(h, layer);
    double_click(h, at);
    assert_eq!(renaming(&h.ctx), Some(layer), "double-click starts renaming");
    assert!(field_shown(h, layer));
}

#[test]
fn clicking_elsewhere_commits_the_rename() {
    for ppp in [1.0, 2.0] {
        let (mut h, [a, ..]) = harness(ppp);
        start(&mut h, a);
        type_text(&mut h, "Sky");
        assert_eq!(name(&h, a), "Alpha", "not applied while typing");
        let at = canvas(&h);
        click(&mut h, at);
        assert_eq!(name(&h, a), "Sky", "@{ppp}x: a click elsewhere commits");
        assert_eq!(renaming(&h.ctx), None, "@{ppp}x: and closes the field");
        h.run_steps(2);
        assert!(!field_shown(&h, a));
    }
}

#[test]
fn enter_and_tab_commit_and_escape_cancels() {
    let (mut h, [a, b, c]) = harness(1.0);
    start(&mut h, a);
    type_text(&mut h, "By Enter");
    key(&mut h, Key::Enter);
    assert_eq!((name(&h, a), renaming(&h.ctx)), ("By Enter".into(), None));

    start(&mut h, b);
    type_text(&mut h, "By Tab");
    key(&mut h, Key::Tab);
    assert_eq!((name(&h, b), renaming(&h.ctx)), ("By Tab".into(), Some(a)), "Tab commits and renames the layer below");
    key(&mut h, Key::Escape);

    start(&mut h, c);
    type_text(&mut h, "Never");
    key(&mut h, Key::Escape);
    assert_eq!((name(&h, c), renaming(&h.ctx)), ("Gamma".into(), None), "Esc cancels");
    // Esc only cancelled the rename: the document and the layer selection are untouched.
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(photocraft_doc::LayerId(c)));

    // An empty or unchanged name renames nothing.
    start(&mut h, c);
    type_text(&mut h, "   ");
    key(&mut h, Key::Enter);
    assert_eq!(name(&h, c), "Gamma");
}

#[test]
fn shift_tab_renames_the_layer_above_and_tab_at_the_bottom_ends() {
    let (mut h, [a, b, c]) = harness(1.0);
    start(&mut h, b);
    type_text(&mut h, "Middle");
    h.event(Event::ModifiersChanged(Modifiers::SHIFT));
    h.event(Event::Key { key: Key::Tab, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::SHIFT });
    h.event(Event::Key { key: Key::Tab, physical_key: None, pressed: false, repeat: false, modifiers: Modifiers::SHIFT });
    h.run_steps(1);
    h.event(Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
    assert_eq!((name(&h, b), renaming(&h.ctx)), ("Middle".into(), Some(c)), "⇧Tab renames the layer above");
    // Typing replaces the next layer's name too, as the whole name starts selected.
    type_text(&mut h, "Top");
    key(&mut h, Key::Enter);
    assert_eq!((name(&h, c), renaming(&h.ctx)), ("Top".into(), None));
    // Tab from the lowest layer ends the rename: the Background below isn't renamed in place.
    start(&mut h, a);
    key(&mut h, Key::Tab);
    assert_eq!(renaming(&h.ctx), None);
}

#[test]
fn starting_another_rename_commits_the_first_and_only_one_is_open() {
    for ppp in [1.0, 2.0] {
        let (mut h, [a, b, _]) = harness(ppp);
        start(&mut h, a);
        type_text(&mut h, "First");
        start(&mut h, b);
        assert_eq!(name(&h, a), "First", "@{ppp}x: the first rename was committed");
        assert!(!field_shown(&h, a), "@{ppp}x: its field closed");
        assert!(field_shown(&h, b));
        type_text(&mut h, "Second");
        assert_eq!(rename(&h.ctx).unwrap().text, "Second", "typing goes to the open field");
        key(&mut h, Key::Enter);
        assert_eq!((name(&h, a), name(&h, b)), ("First".into(), "Second".into()));
        assert_eq!(renaming(&h.ctx), None);
    }
}

#[test]
fn layer_rename_layer_from_the_menu_commits_an_open_rename() {
    let (mut h, [a, b, _]) = harness(1.0);
    start(&mut h, a);
    type_text(&mut h, "Kept");
    let ctx = h.ctx.clone();
    crate::menus::invoke(h.state_mut(), &ctx, "layer.renameLayer", json!({"layer": b})).unwrap();
    h.run_steps(3);
    assert_eq!(name(&h, a), "Kept");
    assert_eq!(renaming(&h.ctx), Some(b));
    assert!(field_shown(&h, b) && !field_shown(&h, a));
    type_text(&mut h, "From menu");
    key(&mut h, Key::Enter);
    assert_eq!(name(&h, b), "From menu");
}

#[test]
fn a_rename_ends_when_its_layer_goes_away() {
    let (mut h, [a, ..]) = harness(1.0);
    start(&mut h, a);
    h.state_mut().run("layer.select", json!({"layer": a})).unwrap();
    h.state_mut().run("layer.delete", json!({})).unwrap();
    assert!(h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(a)).is_none());
    h.run_steps(3);
    assert_eq!(renaming(&h.ctx), None);
    // The keyboard is back to the tools: V selects the Move tool.
    key(&mut h, Key::V);
    assert_eq!(h.state().ui.tool, crate::state::Tool::Move);
}

/// #651: a double-click just above or below the name, still over it, renames rather than opening
/// Layer Style.
#[test]
fn double_click_above_or_below_the_name_renames() {
    for ppp in [1.0, 2.0] {
        let (mut h, [a, ..]) = harness(ppp);
        let row = recorded(&h.ctx).into_iter().find(|r| r.layer == a).unwrap();
        let name = row.name.expect("name drawn");
        for y in [row.row.top() + 1.0, row.row.bottom() - 2.0] {
            assert!(!name.expand(2.0).contains(pos2(name.center().x, y)), "@{ppp}x: off the glyphs");
            double_click(&mut h, pos2(name.center().x, y));
            assert_eq!(renaming(&h.ctx), Some(a), "@{ppp}x: y {y} renames");
            assert!(h.state().ui.dialogs.is_empty(), "@{ppp}x: no Layer Style");
            key(&mut h, Key::Escape);
        }
    }
}

/// #350: a double-click on the row outside the name opens Layer Style for that layer; on the
/// name it still renames; on a Smart Object's thumbnail it opens the contents.
#[test]
fn double_click_beside_the_name_opens_layer_style() {
    let (mut h, [a, b, c]) = harness(1.0);
    let styles = |h: &Harness<'_, PhotocraftApp>| h.state().ui.dialogs.iter().filter(|d| d.kind == crate::state::DialogKind::LayerStyle).count();
    // Select Alpha with one click, then double-click the empty part of its row.
    let at = name_at(&h, a);
    click(&mut h, at);
    let row = recorded(&h.ctx).into_iter().find(|r| r.layer == a).unwrap();
    let name = row.name.expect("name drawn");
    let beside = pos2(name.right() + 40.0, row.row.center().y);
    assert!(row.indicators.iter().all(|(_, r)| !r.expand(2.0).contains(beside)) && row.row.contains(beside));
    double_click(&mut h, beside);
    assert_eq!(styles(&h), 1, "Layer Style opened");
    assert_eq!(renaming(&h.ctx), None, "not a rename");
    assert_eq!(h.state().session.active().unwrap().active_layer, Some(photocraft_doc::LayerId(a)));
    h.state_mut().ui.dialogs.clear();
    h.run_steps(2);
    // The name still renames.
    start(&mut h, b);
    assert_eq!(styles(&h), 0);
    key(&mut h, Key::Escape);
    // A Smart Object's thumbnail opens its contents instead of Layer Style.
    h.state_mut().run("layer.select", json!({"layer": c})).unwrap();
    h.state_mut().run("edit.fill", json!({"color": "#808080"})).unwrap();
    h.state_mut().run("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    h.run_steps(4);
    let docs = h.state().session.documents().len();
    let smart = h.state().session.active().unwrap().active_layer.unwrap().0;
    let name = recorded(&h.ctx).into_iter().find(|r| r.layer == smart).and_then(|r| r.name).expect("smart row drawn");
    // The thumbnail sits just left of the name (a Smart Object has no mask thumbnail).
    double_click(&mut h, pos2(name.left() - 18.0, name.center().y));
    assert_eq!(h.state().session.documents().len(), docs + 1, "Edit Contents opened the source");
    assert_eq!(styles(&h), 0);
}
