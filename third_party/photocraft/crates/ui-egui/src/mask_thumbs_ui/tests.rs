//! #153: mask thumbnails, link chains and the red X in Layers rows; #154: the layer mask in
//! Channels, the layer's path in Paths, aspect-correct channel thumbnails, the Paths footer.

use egui::{Modifiers, PointerButton, Pos2, Rect, pos2, vec2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;

use super::{MaskKind, THUMB_VECTOR, recorded, vector_thumb_image};
use crate::PhotocraftApp;

/// A 200×100 document: "Masked" (pixel + vector mask) and a "Badge" shape layer.
fn session() -> (photocraft_engine::Session, u64, u64) {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
    let masked = s.execute("layer.new.layer", json!({"name": "Masked"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("edit.fill", json!({"contents": "color", "color": "#336699"})).unwrap();
    s.execute("layer.layerMask.revealAll", json!({"layer": masked})).unwrap();
    let path = json!({"subpaths": [{"knots": [[20, 10], [120, 10], [120, 90], [20, 90]]}]});
    s.execute("layer.vectorMask.add", json!({"layer": masked, "path": path})).unwrap();
    let shape =
        s.execute("shape.create", json!({"kind": "rect", "rect": [140, 20, 40, 40], "fill": "#cc3333", "name": "Badge"})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.select", json!({"layer": masked})).unwrap();
    (s, masked, shape)
}

fn harness(session: photocraft_engine::Session, tab: usize, ppp: f32, dock_width: f32) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1440.0, 1000.0)).with_pixels_per_point(ppp).with_max_steps(64).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        PhotocraftApp::new(session, crate::Services::default())
    });
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new(
        "ui.set",
        json!({"theme": "promedium", "dockWidth": dock_width, "dockTabs": {"layers": tab}, "dock": {"collapsed": ["color", "properties", "history", "navigator"]}}),
    );
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(8);
    h
}

fn click_with(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, modifiers: Modifiers) {
    h.hover_at(at);
    h.event(egui::Event::ModifiersChanged(modifiers));
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers });
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers });
    h.run_steps(1);
    h.event(egui::Event::ModifiersChanged(Modifiers::NONE));
    h.run_steps(2);
}

fn layer(h: &Harness<'_, PhotocraftApp>, id: u64) -> photocraft_doc::Layer {
    let st = h.state().session.active().unwrap();
    st.doc.layer(photocraft_doc::LayerId(id)).unwrap().clone()
}

#[test]
fn rows_show_both_mask_thumbnails_with_chains_and_cache_the_vector_one() {
    for ppp in [1.0, 2.0] {
        let (s, masked, _) = session();
        let h = harness(s, 0, ppp, 290.0);
        let (thumbs, chains) = recorded(&h.ctx, masked).expect("mask thumbnails drawn");
        assert_eq!(thumbs.iter().map(|(k, _)| *k).collect::<Vec<_>>(), [MaskKind::Pixel, MaskKind::Vector]);
        assert_eq!(chains.len(), 2);
        // layer thumbnail | chain | pixel mask | chain | vector mask, without overlaps.
        for i in 0..2 {
            assert!(chains[i].1.right() <= thumbs[i].1.left() + 0.01, "@{ppp}x chain {i} before its thumbnail");
        }
        assert!(thumbs[0].1.right() <= chains[1].1.left() + 0.01);
        let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == masked).unwrap();
        assert!(row.name.unwrap().left() >= thumbs[1].1.right(), "@{ppp}x the name starts after the masks");
        assert!(h.state().thumbs.contains_key(&(photocraft_doc::LayerId(masked), THUMB_VECTOR)), "vector thumbnail cached");
    }
}

#[test]
fn deleting_the_vector_mask_prunes_its_thumbnail() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let key = (photocraft_doc::LayerId(masked), THUMB_VECTOR);
    assert!(h.state().thumbs.contains_key(&key));
    h.state_mut().run("layer.vectorMask.delete", json!({"layer": masked})).unwrap();
    h.run_steps(3);
    assert!(!h.state().thumbs.contains_key(&key));
}

#[test]
fn footer_trash_deletes_only_the_targeted_mask_linked_or_unlinked() {
    for linked in [true, false] {
        for kind in [MaskKind::Pixel, MaskKind::Vector] {
            let (mut s, masked, shape) = session();
            s.execute("layer.layerMask.linked", json!({"linked": linked})).unwrap();
            s.execute("layer.vectorMask.linked", json!({"linked": linked})).unwrap();
            let before = s.active().unwrap().doc.clone();
            let mut h = harness(s, 0, 1.0, 290.0);
            let r = recorded(&h.ctx, masked).unwrap().0.into_iter().find(|(k, _)| *k == kind).unwrap().1;
            click_with(&mut h, r.center(), Modifiers::NONE);
            let label = if kind == MaskKind::Pixel { "Delete Layer Mask" } else { "Delete Vector Mask" };
            let at = h.get_by_label(label).rect().center();
            click_with(&mut h, at, Modifiers::NONE);
            let l = layer(&h, masked);
            assert!(h.state().ui.dialogs.is_empty(), "mask deletion remains immediate");
            let original = before.layer(photocraft_doc::LayerId(masked)).unwrap();
            assert_eq!(l.content, original.content);
            assert_eq!(layer(&h, shape), *before.layer(photocraft_doc::LayerId(shape)).unwrap());
            if kind == MaskKind::Pixel {
                assert!(l.mask.is_none());
                assert_eq!(l.vector_mask, original.vector_mask);
                assert!(!h.state().ui.mask_target);
            } else {
                assert!(l.vector_mask.is_none());
                assert_eq!(l.mask, original.mask);
                assert!(!h.state().ui.vector_mask_target);
            }
            h.state_mut().run("edit.undo", json!({})).unwrap();
            assert_eq!(*h.state().session.active().unwrap().doc, *before);
            h.state_mut().run("edit.redo", json!({})).unwrap();
            assert_eq!(layer(&h, masked), l);
        }
    }
}

#[test]
fn footer_trash_still_deletes_the_layer_when_its_content_is_targeted() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == masked).unwrap().row;
    click_with(&mut h, pos2(row.left() + 46.0, row.center().y), Modifiers::NONE);
    assert!(!h.state().ui.mask_target && !h.state().ui.vector_mask_target);
    let at = h.get_by_label("Delete layer").rect().center();
    click_with(&mut h, at, Modifiers::NONE);
    assert!(h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(masked)).is_some());
    assert!(h.state().ui.dialogs.iter().any(|d| crate::delete_layer_prompt::owns(&d.fields)));
    h.get_by_label("Delete").click();
    h.run_steps(3);
    assert!(h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(masked)).is_none());
}

/// Like Photoshop, ⌥/Alt-clicking the trash deletes the layer without asking.
#[test]
fn alt_click_on_the_footer_trash_deletes_without_confirmation() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == masked).unwrap().row;
    click_with(&mut h, pos2(row.left() + 46.0, row.center().y), Modifiers::NONE);
    let at = h.get_by_label("Delete layer").rect().center();
    click_with(&mut h, at, Modifiers::ALT);
    assert!(h.state().ui.dialogs.is_empty(), "no confirmation");
    assert!(h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(masked)).is_none());
}

#[test]
fn a_new_mask_is_targeted_so_the_next_footer_trash_keeps_its_layer() {
    let (s, _, shape) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    h.state_mut().run("layer.new.layer", json!({"name": "Fresh mask"})).unwrap();
    let id = h.state().session.active().unwrap().active_layer.unwrap();
    h.run_steps(3);
    let label = format!("Add a mask  (from the selection; {} inverts)", crate::shortcuts::pretty("Alt"));
    let at = h.get_by_label(&label).rect().center();
    click_with(&mut h, at, Modifiers::NONE);
    assert!(h.state().ui.mask_target && !h.state().ui.vector_mask_target);
    assert!(h.state().session.active().unwrap().doc.layer(id).unwrap().mask.is_some());
    let at = h.get_by_label("Delete Layer Mask").rect().center();
    click_with(&mut h, at, Modifiers::NONE);
    assert!(h.state().session.active().unwrap().doc.layer(id).unwrap().mask.is_none());
    assert!(h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(shape)).is_some());
}

/// #4728: with a path selected in the Paths panel, the footer's "Add vector mask" makes the vector
/// mask from that path (Layer › Vector Mask › Current Path) and targets it, so Properties shows
/// its Density and Feather.
#[test]
fn add_vector_mask_uses_the_selected_path_and_targets_the_new_mask() {
    let (s, _, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    h.state_mut().run("layer.new.layer", json!({"name": "Spock"})).unwrap();
    h.state_mut().run("layer.layerMask.revealAll", json!({})).unwrap();
    let id = h.state().session.active().unwrap().active_layer.unwrap();
    let path = json!({"subpaths": [{"closed": true, "knots": [[30, 20], [90, 20], [90, 80], [30, 80]]}]});
    h.state_mut().run("path.set", json!({"name": "work", "path": path})).unwrap();
    h.state_mut().ui.selected_path = Some("work".into());
    let work = h.state().session.active().unwrap().doc.work_path.clone().unwrap();
    h.run_steps(3);
    let at = h.get_by_label("Add vector mask").rect().center();
    click_with(&mut h, at, Modifiers::NONE);
    let l = h.state().session.active().unwrap().doc.layer(id).unwrap().clone();
    assert_eq!(l.vector_mask.map(|v| v.path), Some(work), "the vector mask is the selected path, not Reveal All");
    assert!(h.state().ui.vector_mask_target && !h.state().ui.mask_target, "the new vector mask is targeted");
    assert_eq!(h.state().ui.selected_path.as_deref(), Some("layer"), "the mask's path becomes the active path");
}

#[test]
fn dragging_a_layer_row_to_trash_still_deletes_the_layer_with_a_mask_targeted() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let mask = recorded(&h.ctx, masked).unwrap().0.into_iter().find(|(k, _)| *k == MaskKind::Pixel).unwrap().1;
    click_with(&mut h, mask.center(), Modifiers::NONE);
    // Grab the row by its name: its thumbnails drag their masks (#3113, #3119).
    let from = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == masked).unwrap().name.unwrap().center();
    let to = h.get_by_label("Delete Layer Mask").rect().center();
    h.hover_at(from);
    h.run_steps(1);
    h.event(egui::Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    h.run_steps(1);
    for k in 1..=8 {
        h.event(egui::Event::PointerMoved(from + (to - from) * (k as f32 / 8.0)));
        h.run_steps(1);
    }
    h.event(egui::Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    h.run_steps(3);
    assert!(h.state().session.active().unwrap().doc.layer(photocraft_doc::LayerId(masked)).is_none());
}

/// #3113: dragging the layer mask's thumbnail onto the Trash asks "Apply mask to layer before
/// removing?" and keeps the layer; Apply applies the mask, Delete deletes it.
#[test]
fn dragging_the_mask_thumbnail_to_trash_asks_to_apply_or_delete_it() {
    let (mut s, masked, _) = session();
    // A mask that hides the left half, so applying it changes the pixels.
    s.execute("layer.layerMask.delete", json!({"layer": masked})).unwrap();
    s.execute("select.rect", json!({"x": 100, "y": 0, "width": 100, "height": 100})).unwrap();
    s.execute("layer.layerMask.revealSelection", json!({"layer": masked})).unwrap();
    let mut h = harness(s, 0, 1.0, 290.0);
    let drag_mask = |h: &mut Harness<'_, PhotocraftApp>| {
        let from = recorded(&h.ctx, masked).unwrap().0.into_iter().find(|(k, _)| *k == MaskKind::Pixel).unwrap().1.center();
        click_with(h, from, Modifiers::NONE);
        let to = h.get_by_label("Delete Layer Mask").rect().center();
        h.hover_at(from);
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.run_steps(1);
        for k in 1..=8 {
            h.event(egui::Event::PointerMoved(from + (to - from) * (k as f32 / 8.0)));
            h.run_steps(1);
        }
        h.event(egui::Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(3);
        let d = h.state().ui.dialogs.iter().find(|d| crate::mask_trash_prompt::owns(&d.fields)).expect("the prompt is open");
        assert_eq!(d.fields["message"], "Apply mask to layer before removing?");
        assert!(layer(h, masked).mask.is_some(), "nothing changes before the answer");
        d.id
    };
    let pixels = |h: &Harness<'_, PhotocraftApp>| layer(h, masked).content;
    let before = pixels(&h);
    let past = h.state().session.active().unwrap().history.past_len();
    // Cancel keeps the mask.
    let dialog = drag_mask(&mut h);
    crate::dialogs::cancel(h.state_mut(), dialog).unwrap();
    assert!(layer(&h, masked).mask.is_some());
    // Apply: one step, the mask is gone and the layer's pixels are masked.
    let dialog = drag_mask(&mut h);
    crate::dialogs::confirm(h.state_mut(), dialog).unwrap();
    assert!(layer(&h, masked).mask.is_none());
    assert_ne!(pixels(&h), before, "applied, not deleted");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), past + 1);
    h.state_mut().session.undo();
    h.run_steps(2);
    // Delete: the mask is gone and the pixels are as they were.
    let dialog = drag_mask(&mut h);
    h.state_mut().ui.dialog_mut(dialog).unwrap().fields.insert("choice".into(), json!("delete"));
    crate::dialogs::confirm(h.state_mut(), dialog).unwrap();
    assert!(layer(&h, masked).mask.is_none());
    assert_eq!(pixels(&h), before);
}

/// #3119: dragging the vector mask's thumbnail onto the Trash asks "Delete the vector mask?" and
/// keeps the layer; Cancel keeps the mask, OK deletes only the vector mask in one step.
#[test]
fn dragging_the_vector_mask_thumbnail_to_trash_asks_to_delete_just_the_mask() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let drag_vector_mask = |h: &mut Harness<'_, PhotocraftApp>| {
        let from = recorded(&h.ctx, masked).unwrap().0.into_iter().find(|(k, _)| *k == MaskKind::Vector).unwrap().1.center();
        click_with(h, from, Modifiers::NONE);
        let to = h.get_by_label("Delete Vector Mask").rect().center();
        h.hover_at(from);
        h.run_steps(1);
        h.event(egui::Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        h.run_steps(1);
        for k in 1..=8 {
            h.event(egui::Event::PointerMoved(from + (to - from) * (k as f32 / 8.0)));
            h.run_steps(1);
        }
        h.event(egui::Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        h.run_steps(3);
        let st = h.state().session.active().unwrap();
        assert!(st.doc.layer(photocraft_doc::LayerId(masked)).is_some(), "the layer is not deleted");
        let d = h.state().ui.dialogs.iter().find(|d| crate::mask_trash_prompt::owns(&d.fields)).expect("the prompt is open");
        assert_eq!(d.fields["message"], "Delete the vector mask?");
        assert!(layer(h, masked).vector_mask.is_some(), "nothing changes before the answer");
        d.id
    };
    let past = h.state().session.active().unwrap().history.past_len();
    // Cancel keeps the vector mask.
    let dialog = drag_vector_mask(&mut h);
    crate::dialogs::cancel(h.state_mut(), dialog).unwrap();
    assert!(layer(&h, masked).vector_mask.is_some());
    assert_eq!(h.state().session.active().unwrap().history.past_len(), past);
    // OK: one step, only the vector mask is gone.
    let dialog = drag_vector_mask(&mut h);
    crate::dialogs::confirm(h.state_mut(), dialog).unwrap();
    let after = layer(&h, masked);
    assert!(after.vector_mask.is_none());
    assert!(after.mask.is_some(), "the layer mask stays");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), past + 1);
}

#[test]
fn the_chain_toggles_linking_and_it_survives_psd() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let p = h.get_by_label("Unlink layer mask Masked").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert!(!layer(&h, masked).mask.unwrap().linked);
    let p = h.get_by_label("Unlink vector mask Masked").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert!(!layer(&h, masked).vector_mask.unwrap().linked);
    let doc = (*h.state().session.active().unwrap().doc).clone();
    let bytes = photocraft_io::export(&doc, "m.psd", &Default::default()).unwrap().bytes;
    let back = photocraft_io::import("m.psd", &bytes).unwrap().document;
    let l = back.walk().into_iter().map(|(_, _, l)| l.clone()).find(|l| l.name == "Masked").unwrap();
    assert!(!l.mask.unwrap().linked && !l.vector_mask.unwrap().linked, "unlinked state round-trips PSD");
    // Clicking the (now empty) chain slot links again.
    let p = h.get_by_label("Link layer mask Masked").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert!(layer(&h, masked).mask.unwrap().linked);
}

#[test]
fn shift_click_disables_a_mask() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    for kind in [MaskKind::Pixel, MaskKind::Vector] {
        let r = recorded(&h.ctx, masked).unwrap().0.into_iter().find(|(k, _)| *k == kind).unwrap().1;
        click_with(&mut h, r.center(), Modifiers::SHIFT);
        h.run_steps(2);
    }
    let l = layer(&h, masked);
    assert!(!l.mask.unwrap().enabled && !l.vector_mask.unwrap().enabled);
}

#[test]
fn vector_thumbnail_is_white_inside_grey_outside_at_the_document_aspect() {
    let (s, masked, _) = session();
    let doc = s.active().unwrap().doc.clone();
    let vm = doc.layer(photocraft_doc::LayerId(masked)).unwrap().vector_mask.clone().unwrap();
    let img = vector_thumb_image(&doc, &vm, 64);
    let at = |x: usize, y: usize| img.pixels[y * 64 + x];
    // 200×100 doc in 64 px: 0.32 scale, rows 16..48. The mask covers x 20..120, y 10..90.
    assert_eq!(at(20, 32), egui::Color32::from_gray(255), "inside reveals");
    assert_eq!(at(55, 32), egui::Color32::from_gray(128), "outside hides");
    assert_eq!(at(20, 4), egui::Color32::TRANSPARENT, "letterbox");
}

#[test]
fn channels_list_the_layer_mask_and_keep_the_aspect() {
    let (s, masked, _) = session();
    let mut h = harness(s, 1, 1.0, 290.0);
    let rows = crate::channels_panel::recorded(&h.ctx);
    let names: Vec<_> = rows.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names, ["RGB", "Red", "Green", "Blue", "Masked Mask"], "the mask is a temporary row below the colour channels");
    for (n, r) in &rows {
        assert!((r.width() / r.height() - 2.0).abs() < 0.05, "{n}: 2:1 thumbnail, got {r:?}");
    }
    assert!(!h.state().ui.mask_target);
    let p = h.get_by_label("Masked Mask").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert!(h.state().ui.mask_target, "clicking the mask row targets the mask");
    assert_eq!(crate::canvas::paint_target(h.state()), json!("mask"));
    h.run_steps(2);
    let p = h.get_by_label("RGB").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert!(!h.state().ui.mask_target, "the composite targets the pixels again");
    // A layer without a mask has no mask row.
    h.state_mut().run("layer.layerMask.delete", json!({"layer": masked})).unwrap();
    h.run_steps(3);
    assert!(crate::channels_panel::recorded(&h.ctx).iter().all(|(n, _)| !n.ends_with("Mask")));
}

#[test]
fn paths_list_the_layer_path_above_a_bottom_footer() {
    let (s, masked, shape) = session();
    let mut h = harness(s, 2, 1.0, 290.0);
    let row = h.get_by_label("Masked Vector Mask").rect();
    let footer = crate::vector_ui::paths_footer(&h.ctx).expect("footer drawn");
    assert!(footer.top() > row.bottom(), "buttons below the rows");
    let group = crate::dock::last_rects(&h.ctx).into_iter().find(|(g, _)| *g == crate::dock::Group::Layers).unwrap().1;
    assert!(group.bottom() - footer.bottom() < 16.0, "footer at the panel bottom: footer {footer:?}, group {group:?}");
    assert!(footer.bottom() - row.bottom() > 100.0, "the footer is pinned, not right below the rows");
    h.state_mut().run("layer.select", json!({"layer": shape})).unwrap();
    h.run_steps(3);
    assert!(h.query_by_label("Badge Shape Path").is_some());
    assert!(h.query_by_label("Masked Vector Mask").is_none(), "only the selected layer's path");
    // Selecting the temporary row targets the layer's path: Load as selection uses it.
    let p = h.get_by_label("Badge Shape Path").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert_eq!(h.state().ui.selected_path.as_deref(), Some("layer"));
    // Footer: fill, stroke, load as selection, … (26 pt buttons, 2 pt apart).
    let footer = crate::vector_ui::paths_footer(&h.ctx).unwrap();
    let p = pos2(footer.left() + 2.0 * 28.0 + 13.0, footer.center().y);
    click_with(&mut h, p, Modifiers::NONE);
    let sel = h.state().session.active().unwrap().doc.selection.clone().expect("selection from the shape path");
    let b = sel.content_bounds();
    assert!(b.x0.abs_diff(140) <= 1 && b.width().abs_diff(40) <= 1, "{b:?}");
    let _ = masked;
}

#[test]
fn panels_never_panic_on_odd_documents() {
    // Rows render for a 1×1 and a very wide document, and the fitted thumbnails stay finite.
    for (w, hgt) in [(1, 1), (2000, 3)] {
        for tab in 0..3 {
            let mut s = photocraft_engine::Session::new();
            s.execute("file.new", json!({"width": w, "height": hgt})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            s.execute("layer.layerMask.hideAll", json!({})).unwrap();
            s.execute("layer.vectorMask.revealAll", json!({})).unwrap();
            let _ = harness(s, tab, 1.0, 250.0);
        }
    }
    let (r, uv) = crate::channels_panel::fit_thumb(Rect::from_min_size(pos2(0.0, 0.0), vec2(28.0, 28.0)), 0, 0);
    assert!(r.is_finite() && uv.is_finite() && r.width() > 0.0);
}

// ---------------------------------------------------------------------------------------------
// #196: ⌥-click mask view, ⌘-click vector mask to select, vector-mask target brackets.

/// The session above with a black dot painted into the layer mask at (50, 50).
fn dotted() -> (photocraft_engine::Session, u64, u64) {
    let (mut s, masked, shape) = session();
    s.execute("paint.stroke", json!({"points": [[50, 50]], "size": 20, "hardness": 1.0, "color": "#000000", "target": "mask"})).unwrap();
    (s, masked, shape)
}

fn mask_rect(h: &Harness<'_, PhotocraftApp>, id: u64, kind: MaskKind) -> Rect {
    recorded(&h.ctx, id).unwrap().0.into_iter().find(|(k, _)| *k == kind).unwrap().1
}

/// The layer thumbnail sits left of the first chain (see `paint`).
fn layer_thumb_center(h: &Harness<'_, PhotocraftApp>, id: u64) -> Pos2 {
    let first = recorded(&h.ctx, id).unwrap().0[0].1;
    pos2(first.left() - 4.0 - super::CHAIN_W - first.width() / 2.0, first.center().y)
}

fn mask_view(h: &Harness<'_, PhotocraftApp>) -> serde_json::Value {
    photocraft_engine::inspect::document(h.state().session.active().unwrap())["layerMaskView"].clone()
}

fn canvas_px(h: &Harness<'_, PhotocraftApp>) -> Option<Vec<egui::Color32>> {
    let st = h.state().session.active().unwrap();
    crate::channel_view::render(&st.doc, &st.channel_view, photocraft_geom::Rect::new(0, 0, 200, 100), 1, false, st.active_layer)
}

fn near(a: Rect, b: Rect) -> bool {
    (a.min - b.min).length() < 0.6 && (a.max - b.max).length() < 0.6
}

#[test]
fn alt_click_views_the_mask_in_gray_and_the_layer_thumbnail_returns() {
    for ppp in [1.0, 2.0] {
        let (s, masked, _) = dotted();
        let mut h = harness(s, 0, ppp, 290.0);
        let doc = h.state().session.active().unwrap().doc.clone();
        let r = mask_rect(&h, masked, MaskKind::Pixel);
        click_with(&mut h, r.center(), Modifiers::ALT);
        assert_eq!(mask_view(&h), json!({"layer": masked, "mode": "gray"}), "@{ppp}x");
        assert!(h.state().ui.mask_target, "viewing the mask targets it");
        assert_eq!(crate::canvas::paint_target(h.state()), json!("mask"), "painting paints the mask");
        let px = canvas_px(&h).expect("the canvas shows the mask");
        assert_eq!(px[50 * 200 + 50], egui::Color32::BLACK, "hidden reads black");
        assert_eq!(px[10 * 200 + 150], egui::Color32::WHITE, "revealed reads white");
        assert!(near(super::brackets(&h.ctx).unwrap(), r.expand(3.0)), "brackets on the mask");
        // ⌥-click again: back to the composite.
        click_with(&mut h, r.center(), Modifiers::ALT);
        assert_eq!(mask_view(&h), serde_json::Value::Null);
        assert!(canvas_px(&h).is_none());
        // Into mask view again, then out by clicking the layer thumbnail.
        click_with(&mut h, r.center(), Modifiers::ALT);
        assert_eq!(mask_view(&h)["mode"], "gray");
        let p = layer_thumb_center(&h, masked);
        click_with(&mut h, p, Modifiers::NONE);
        assert_eq!(mask_view(&h), serde_json::Value::Null, "@{ppp}x the layer thumbnail returns");
        assert!(!h.state().ui.mask_target, "and targets the pixels");
        assert!(std::sync::Arc::ptr_eq(&doc, &h.state().session.active().unwrap().doc), "viewing never touched the document");
    }
}

#[test]
fn shift_alt_click_shows_a_rubylith_and_the_channels_eye_toggles_it() {
    let (s, masked, _) = dotted();
    let mut h = harness(s, 0, 1.0, 290.0);
    let r = mask_rect(&h, masked, MaskKind::Pixel);
    click_with(&mut h, r.center(), Modifiers::SHIFT | Modifiers::ALT);
    assert_eq!(mask_view(&h), json!({"layer": masked, "mode": "overlay"}));
    let px = canvas_px(&h).unwrap();
    assert_eq!(px[50 * 200 + 50], egui::Color32::from_rgba_premultiplied(128, 0, 0, 128), "50% red over hidden areas");
    assert_eq!(px[10 * 200 + 150], egui::Color32::TRANSPARENT, "the composite shows through elsewhere");
    assert!(layer(&h, masked).mask.unwrap().enabled, "⇧⌥ doesn't disable the mask");
    // Channels: the mask row's eye shows and hides the overlay.
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dockTabs": {"layers": 1}}));
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(4);
    let p = h.get_by_label("Visibility Masked Mask").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert_eq!(mask_view(&h), serde_json::Value::Null, "the eye hides the overlay");
    let p = h.get_by_label("Visibility Masked Mask").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert_eq!(mask_view(&h)["mode"], "overlay", "and shows it again");
    // In gray view the composite's eye brings the composite back under the overlay.
    h.state_mut().run("view.layerMask", json!({"mode": "gray"})).unwrap();
    h.run_steps(3);
    let p = h.get_by_label("Visibility RGB").rect().center();
    click_with(&mut h, p, Modifiers::NONE);
    assert_eq!(mask_view(&h)["mode"], "overlay");
}

#[test]
fn command_click_a_vector_mask_loads_its_path_as_a_selection() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let r = mask_rect(&h, masked, MaskKind::Vector);
    let sel = |h: &Harness<'_, PhotocraftApp>| h.state().session.active().unwrap().doc.selection.clone();
    click_with(&mut h, r.center(), Modifiers::COMMAND);
    let b = sel(&h).expect("⌘-click selects the path").content_bounds();
    assert_eq!((b.x0, b.y0, b.width(), b.height()), (20, 10, 100, 80));
    assert!(!h.state().ui.vector_mask_target, "⌘-click doesn't retarget");
    // ⌘⇧ add a 4×4 square, ⌘⌥ subtract, ⌘⇧⌥ intersect.
    h.state_mut().run("select.rect", json!({"x": 150, "y": 50, "width": 4, "height": 4})).unwrap();
    click_with(&mut h, r.center(), Modifiers::COMMAND | Modifiers::SHIFT);
    let s1 = sel(&h).unwrap();
    assert!(s1.sample_channel(60, 50, 0) > 0.99 && s1.sample_channel(151, 51, 0) > 0.99, "added");
    h.state_mut().run("select.all", json!({})).unwrap();
    click_with(&mut h, r.center(), Modifiers::COMMAND | Modifiers::ALT);
    let s2 = sel(&h).unwrap();
    assert!(s2.sample_channel(60, 50, 0) < 0.01 && s2.sample_channel(150, 50, 0) > 0.99, "subtracted");
    h.state_mut().run("select.rect", json!({"x": 100, "y": 0, "width": 100, "height": 100})).unwrap();
    click_with(&mut h, r.center(), Modifiers::COMMAND | Modifiers::SHIFT | Modifiers::ALT);
    let b = sel(&h).unwrap().content_bounds();
    assert_eq!((b.x0, b.y0, b.width(), b.height()), (100, 10, 20, 80), "intersected");
}

#[test]
fn clicking_the_vector_mask_targets_it_for_the_path_tools() {
    let (s, masked, _) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    let r = mask_rect(&h, masked, MaskKind::Vector);
    click_with(&mut h, r.center(), Modifiers::NONE);
    assert!(h.state().ui.vector_mask_target && !h.state().ui.mask_target);
    assert_eq!(h.state().ui.selected_path.as_deref(), Some("layer"), "the Paths panel selects its path");
    assert!(near(super::brackets(&h.ctx).unwrap(), r.expand(3.0)), "brackets on the vector mask");
    assert_eq!(crate::canvas::paint_target(h.state()), json!("pixels"));
    // Path Selection moves the vector mask.
    crate::vector_ui::path_selection_finish(h.state_mut(), [30.0, 30.0], [35.0, 32.0]);
    let k = layer(&h, masked).vector_mask.unwrap().path.subpaths[0].knots[0].anchor;
    assert_eq!((k.x, k.y), (25.0, 12.0));
    // The Pen adds a subpath to it.
    h.state_mut().ui.pen = Some(crate::vector_ui::PenPath { knots: vec![[[150.0, 10.0]; 3], [[190.0, 10.0]; 3], [[190.0, 40.0]; 3]], ..Default::default() });
    crate::vector_ui::pen_commit(h.state_mut(), true);
    assert_eq!(layer(&h, masked).vector_mask.unwrap().path.subpaths.len(), 2);
    assert!(h.state().session.active().unwrap().doc.work_path.is_none(), "not a work path");
    h.run_steps(2);
    // The pixel mask and the layer thumbnail take the brackets back.
    let m = mask_rect(&h, masked, MaskKind::Pixel);
    click_with(&mut h, m.center(), Modifiers::NONE);
    assert!(h.state().ui.mask_target && !h.state().ui.vector_mask_target);
    assert!(near(super::brackets(&h.ctx).unwrap(), m.expand(3.0)));
    click_with(&mut h, r.center(), Modifiers::NONE);
    let p = layer_thumb_center(&h, masked);
    click_with(&mut h, p, Modifiers::NONE);
    assert!(!h.state().ui.mask_target && !h.state().ui.vector_mask_target);
    // Deleting the vector mask drops the target.
    click_with(&mut h, r.center(), Modifiers::NONE);
    assert!(h.state().ui.vector_mask_target);
    h.state_mut().run("layer.vectorMask.delete", json!({"layer": masked})).unwrap();
    h.run_steps(2);
    assert!(!h.state().ui.vector_mask_target);
}

#[test]
fn mask_view_gestures_fail_gracefully_without_a_mask() {
    let (s, masked, shape) = session();
    let mut h = harness(s, 0, 1.0, 290.0);
    // ⌥-click on a vector mask is not a mask view.
    let p = mask_rect(&h, masked, MaskKind::Vector).center();
    click_with(&mut h, p, Modifiers::ALT);
    assert_eq!(mask_view(&h), serde_json::Value::Null);
    // A layer without a layer mask: the command fails, nothing changes.
    assert!(h.state_mut().run("view.layerMask", json!({"layer": shape, "mode": "gray"})).is_err());
    assert_eq!(mask_view(&h), serde_json::Value::Null);
    h.run_steps(2);
    // A stale view (layer gone) draws nothing instead of panicking.
    h.state_mut().session.active_mut().unwrap().channel_view.layer_mask = Some(photocraft_engine::mask_view_cmds::LayerMaskView {
        layer: photocraft_doc::LayerId(u64::MAX),
        mode: photocraft_engine::mask_view_cmds::MaskViewMode::Gray,
    });
    assert!(canvas_px(&h).is_none());
    h.run_steps(2);
    // The mask goes while it is shown: the view ends.
    h.state_mut().run("view.layerMask", json!({"layer": masked, "mode": "gray"})).unwrap();
    h.state_mut().run("layer.layerMask.delete", json!({"layer": masked})).unwrap();
    h.run_steps(2);
    assert_eq!(mask_view(&h), serde_json::Value::Null);
    assert_eq!(crate::canvas::paint_target(h.state()), json!("pixels"));
}

/// #1765: `\` shows the active layer's mask as a rubylith and hides it again, as in Photoshop;
/// the mask stays the paint target meanwhile. On a layer without a mask it does nothing, and a
/// focused text field keeps the key.
#[test]
fn backslash_toggles_the_rubylith() {
    let (s, masked, shape) = dotted();
    let mut h = harness(s, 0, 1.0, 290.0);
    let doc = h.state().session.active().unwrap().doc.clone();
    h.key_press(egui::Key::Backslash);
    h.run_steps(2);
    assert_eq!(mask_view(&h), json!({"layer": masked, "mode": "overlay"}), "\\ shows the overlay");
    assert_eq!(crate::canvas::paint_target(h.state()), json!("mask"), "painting paints the mask");
    let px = canvas_px(&h).unwrap();
    assert_eq!(px[50 * 200 + 50], egui::Color32::from_rgba_premultiplied(128, 0, 0, 128), "50% red over hidden areas");
    h.key_press(egui::Key::Backslash);
    h.run_steps(2);
    assert_eq!(mask_view(&h), serde_json::Value::Null, "\\ again hides it");
    assert!(std::sync::Arc::ptr_eq(&doc, &h.state().session.active().unwrap().doc), "a view toggle, not an edit");
    // A layer without a layer mask: nothing happens.
    h.state_mut().run("layer.select", json!({"layer": shape})).unwrap();
    h.run_steps(2);
    h.key_press(egui::Key::Backslash);
    h.run_steps(2);
    assert_eq!(mask_view(&h), serde_json::Value::Null);
    let sc = crate::shortcuts::parse("\\").unwrap();
    assert!(!crate::shortcut_dispatch::Focus::Text.allows(&sc), "typing \\ in a text field stays text");
}

/// #780: clicking the mask thumbnail targets the mask, and ⌘I (Image › Adjustments › Invert)
/// then inverts the mask instead of the layer; on an adjustment layer the menu item is live.
#[test]
fn command_i_inverts_the_targeted_mask() {
    let (s, masked, _) = dotted();
    let mut h = harness(s, 0, 1.0, 290.0);
    let pixels = layer(&h, masked).content;
    let r = mask_rect(&h, masked, MaskKind::Pixel);
    click_with(&mut h, r.center(), Modifiers::NONE);
    assert!(h.state().ui.mask_target);
    h.key_press_modifiers(Modifiers::COMMAND, egui::Key::I);
    h.run_steps(2);
    let l = layer(&h, masked);
    let mask = &l.mask.as_ref().unwrap().surface;
    assert_eq!((mask.sample_channel(50, 50, 0), mask.sample_channel(150, 10, 0)), (1.0, 0.0), "the mask inverted");
    assert!(l.content == pixels, "the layer's pixels are untouched");
    // An adjustment layer's mask: with no pixels of its own, Invert edits the mask whether the
    // panel targets the mask or the icon thumbnail (#4298).
    let app = h.state_mut();
    app.run("layer.newAdjustmentLayer.curves", json!({})).unwrap();
    app.run("layer.layerMask.revealAll", json!({})).unwrap();
    app.ui.mask_target = false;
    assert!(crate::menus::is_enabled(app, "image.adjustments.invert"));
    app.ui.mask_target = true;
    assert!(crate::menus::is_enabled(app, "image.adjustments.invert"));
    app.run("image.adjustments.invert", json!({})).unwrap();
    let st = app.session.active().unwrap();
    let mask = &st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface;
    assert_eq!(mask.sample_channel(10, 10, 0), 0.0, "reveal all → hide all");
}

/// #2420: Selecting a layer mask in the Layers panel toggles the color swatches
/// to the mask's colors (white/black by default), and deselecting toggles back.
#[test]
fn selecting_mask_thumbnail_toggles_swatches_to_mask_colors_and_back() {
    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];
    const BLUE: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];

    let (mut s, masked, _) = dotted();
    s.execute("tools.setColors", json!({"foreground": "#ff0000", "background": "#0000ff"})).unwrap();
    let mut h = harness(s, 0, 1.0, 290.0);

    // Initial state: layer pixels targeted, red foreground, blue background.
    assert!(!h.state().ui.mask_target);
    assert_eq!(h.state().session.tools.foreground, RED);
    assert_eq!(h.state().session.tools.background, BLUE);

    // 1. Click mask thumbnail in Layers panel -> targets mask, swatches toggle to mask colors.
    let r_mask = mask_rect(&h, masked, MaskKind::Pixel);
    click_with(&mut h, r_mask.center(), Modifiers::NONE);
    assert!(h.state().ui.mask_target, "mask thumbnail clicked: mask is targeted");
    assert_eq!(h.state().session.tools.foreground, WHITE);
    assert_eq!(h.state().session.tools.background, BLACK);

    // 2. Click layer thumbnail -> deselects mask, swatches toggle back to pixel colors.
    let r_layer = layer_thumb_center(&h, masked);
    click_with(&mut h, r_layer, Modifiers::NONE);
    assert!(!h.state().ui.mask_target, "layer thumbnail clicked: pixels are targeted");
    assert_eq!(h.state().session.tools.foreground, RED);
    assert_eq!(h.state().session.tools.background, BLUE);

    // 3. Click mask thumbnail again -> swatches toggle back to mask colors.
    click_with(&mut h, r_mask.center(), Modifiers::NONE);
    assert!(h.state().ui.mask_target);
    assert_eq!(h.state().session.tools.foreground, WHITE);
    assert_eq!(h.state().session.tools.background, BLACK);

    // 4. Changing mask colors preserves them when switching away and back.
    h.state_mut().run("tools.setColors", json!({"foreground": "#808080"})).unwrap();
    let grey = h.state().session.tools.foreground;
    assert_eq!(grey, [128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0]);

    click_with(&mut h, r_layer, Modifiers::NONE);
    assert_eq!(h.state().session.tools.foreground, RED, "pixel colors restored");

    click_with(&mut h, r_mask.center(), Modifiers::NONE);
    assert_eq!(h.state().session.tools.foreground, grey, "mask colors restored");

    // 5. Creating a new adjustment layer targets its mask and toggles to mask colors.
    click_with(&mut h, r_layer, Modifiers::NONE);
    assert_eq!(h.state().session.tools.foreground, RED);
    h.state_mut().run("layer.newAdjustmentLayer.curves", json!({})).unwrap();
    assert!(h.state().ui.mask_target, "new adjustment layer targets its mask");
    assert_eq!(h.state().session.tools.foreground, grey);
}

// #3369: a new adjustment or fill layer targets its mask, but Properties shows its settings
// until the mask thumbnail is clicked, as in Photoshop.
#[test]
fn a_new_adjustment_or_fill_layer_shows_its_settings_until_the_mask_thumbnail_is_clicked() {
    for (command, settings) in [("layer.newAdjustmentLayer.curves", Some("Reset to defaults")), ("layer.newFillLayer.solidColor", None)] {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        let mut h = harness(s, 0, 1.0, 290.0);
        let ctx = h.ctx.clone();
        let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "history", "navigator"]}}));
        crate::control::handle(h.state_mut(), &ctx, &req);
        h.state_mut().run(command, json!({})).unwrap();
        h.run_steps(6);
        let id = h.state().session.active().unwrap().active_layer.unwrap().0;
        assert!(h.state().ui.mask_target, "{command}: the new mask is targeted");
        assert_eq!(crate::canvas::paint_target(h.state()), json!("mask"), "{command}: painting edits the mask");
        assert!(h.query_all_by_label("Density").next().is_none(), "{command}: Properties shows the layer, not its mask");
        if let Some(label) = settings {
            assert!(h.query_all_by_label(label).next().is_some(), "{command}: its settings show");
        }
        // Clicking the row (not a thumbnail) keeps the settings.
        let row = h.get_by_label(&layer(&h, id).name).rect().center();
        click_with(&mut h, row, Modifiers::NONE);
        assert!(h.state().ui.mask_target);
        assert!(h.query_all_by_label("Density").next().is_none(), "{command}: a row click keeps the settings");
        // The mask thumbnail switches Properties to the mask.
        let r = mask_rect(&h, id, MaskKind::Pixel);
        click_with(&mut h, r.center(), Modifiers::NONE);
        h.run_steps(2);
        assert!(h.state().ui.mask_target);
        assert!(h.query_all_by_label("Density").next().is_some(), "{command}: the mask thumbnail shows the mask controls");
        if let Some(label) = settings {
            assert!(h.query_all_by_label(label).next().is_none());
        }
    }
}

// #4028: with Adjustments showing in a collapsed Properties group, a new adjustment layer brings
// Properties to the front and expands it, as in Photoshop, so its settings can be edited at once.
#[test]
fn a_new_adjustment_layer_brings_properties_to_the_front() {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 200, "height": 100})).unwrap();
    let mut h = harness(s, 0, 1.0, 290.0);
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dockTabs": {"properties": 1}}));
    crate::control::handle(h.state_mut(), &ctx, &req);
    h.run_steps(2);
    assert!(h.state().ui.dock.is_collapsed(crate::dock::Group::Properties));
    h.state_mut().run("layer.newAdjustmentLayer.curves", json!({})).unwrap();
    h.run_steps(6);
    assert_eq!(h.state().ui.dock_tabs.properties, 0, "Properties is the group's tab");
    assert!(!h.state().ui.dock.is_collapsed(crate::dock::Group::Properties), "the group is expanded");
    assert!(h.query_all_by_label("Reset to defaults").next().is_some(), "the Curves settings show");
}
