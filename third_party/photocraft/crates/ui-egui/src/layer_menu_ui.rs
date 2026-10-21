//! Layers panel right-click menu in Photoshop's order. Items are command ids from the menu catalog;
//! unavailable ones are greyed using the same enablement as the main menus. Items marked as menu
//! invocations (`Value::Null` params) go through `menus::invoke`, so dialogs open like in the menu bar.

use photocraft_doc::{LabelColor, Layer, LayerContent};
use serde_json::{Value, json};

/// One entry: label, command id. `None` = separator.
pub type Entry = Option<(&'static str, &'static str)>;

/// English action key; each menu translates it using its existing display context.
pub(crate) fn mask_toggle_label(enabled: bool) -> &'static str {
    if enabled { "Disable Layer Mask" } else { "Enable Layer Mask" }
}

/// The command behind "Add Layer Mask" (the panel button and this menu). Like Photoshop, an active
/// selection becomes the mask; ⌥ inverts it (Hide Selection, or Hide All without a selection).
pub fn add_mask_command(has_selection: bool, alt: bool) -> &'static str {
    match (has_selection, alt) {
        (true, false) => "layer.layerMask.revealSelection",
        (true, true) => "layer.layerMask.hideSelection",
        (false, false) => "layer.layerMask.revealAll",
        (false, true) => "layer.layerMask.hideAll",
    }
}

/// What the Layers panel "Add a mask" button does for `layer` (#2075). Like Photoshop it never
/// replaces a mask: a layer without a layer mask gets one ([`add_mask_command`]); a layer that
/// already has one gets a vector mask instead (⌥: Hide All); a layer with both (or a shape layer,
/// whose path is already its vector mask, with a layer mask) gets nothing (`None`).
pub fn mask_button_command(layer: Option<&Layer>, has_selection: bool, alt: bool) -> Option<&'static str> {
    let l = layer?;
    if l.mask.is_none() {
        Some(add_mask_command(has_selection, alt))
    } else if l.vector_mask.is_none() && !matches!(l.content, LayerContent::Shape(_)) {
        Some(if alt { "layer.vectorMask.hideAll" } else { "layer.vectorMask.revealAll" })
    } else {
        None
    }
}

/// Commands that act on the active layer only and have no sensible meaning for a selection: Blending
/// Options (one dialog), Copy Layer Style (copy from which layer?) and the per-layer mask items.
/// With several layers selected they would hit whichever layer is active, not the one that was
/// right-clicked, so the menu greys them. Adding a mask, Paste/Clear Layer Style, Rasterize and
/// Export As (one file per layer, #3629) run on every selected layer instead.
pub fn single_layer_only(id: &str) -> bool {
    let adds_mask = matches!(id, "layer.layerMask.revealAll" | "layer.layerMask.hideAll" | "layer.layerMask.revealSelection" | "layer.layerMask.hideSelection");
    matches!(id, "layer.layerStyle.blendingOptions" | "layer.layerStyle.copyLayerStyle") || (id.starts_with("layer.layerMask.") && !adds_mask)
}

/// The context menu entries for a layer (Photoshop 2026 order, trimmed to the layer kind).
pub fn entries(l: &Layer, multi: bool, has_selection: bool) -> Vec<Entry> {
    let mut v: Vec<Entry> = vec![Some((tl!("Blending Options…"), "layer.layerStyle.blendingOptions"))];
    v.push(None);
    v.push(Some((if multi { tl!("Duplicate Layers…") } else { tl!("Duplicate Layer…") }, "layer.duplicate")));
    v.push(Some((if multi { tl!("Delete Layers") } else { tl!("Delete Layer") }, "layer.delete")));
    if !multi {
        v.push(Some((tl!("Group from Layers…"), "layer.groupLayers")));
    }
    v.push(None);
    v.push(Some((tl!("Quick Export As PNG"), "layer.quickExportAsPng")));
    v.push(Some((tl!("Export As…"), "layer.exportAs")));
    v.push(None);
    v.push(Some((tl!("Artboard from Layers…"), "layer.new.artboardFromLayers")));
    v.push(Some((tl!("Frame from Layers…"), "layer.new.frameFromLayers")));
    v.push(None);
    v.push(Some((tl!("Convert to Smart Object"), "layer.smartObjects.convertToSmartObject")));
    if multi {
        // One item for the whole selection, whatever the clicked layer is: Rasterize acts on every
        // selected layer. Edit Contents and Convert to Layers work on one layer, so they are not
        // offered here.
        v.push(Some(("Rasterize Layers", "layer.rasterize.layer")));
    } else {
        match &l.content {
            LayerContent::Text(_) => v.push(Some(("Rasterize Type", "layer.rasterize.type"))),
            LayerContent::Shape(_) => v.push(Some(("Rasterize Layer", "layer.rasterize.shape"))),
            LayerContent::Smart(_) => {
                v.push(Some((tl!("Edit Contents"), "layer.smartObjects.editContents")));
                v.push(Some((tl!("Convert to Layers"), "layer.smartObjects.convertToLayers")));
                v.push(Some(("Rasterize Layer", "layer.rasterize.smartObject")));
            }
            LayerContent::Fill(_) => v.push(Some(("Rasterize Layer", "layer.rasterize.fillContent"))),
            _ => v.push(Some(("Rasterize Layer", "layer.rasterize.layer"))),
        }
    }
    // Photoshop offers it under Rasterize Layer for a layer with a style (it bakes the effects).
    if multi || !l.effects.items.is_empty() {
        v.push(Some(("Rasterize Layer Style", "layer.rasterize.layerStyle")));
    }
    v.push(None);
    // A selection has no single mask to toggle, apply or delete, so its menu offers Add Layer Mask only.
    if let Some(mask) = l.mask.as_ref().filter(|_| !multi) {
        v.push(Some((tl!(mask_toggle_label(mask.enabled)), "layer.layerMask.enabled")));
        v.push(Some((tl!("Apply Layer Mask"), "layer.layerMask.apply")));
        v.push(Some((tl!("Delete Layer Mask"), "layer.layerMask.delete")));
    } else {
        v.push(Some((tl!("Add Layer Mask"), add_mask_command(has_selection, false))));
    }
    v.push(Some((
        if l.clipped { tl!("Release Clipping Mask") } else { tl!("Create Clipping Mask") },
        if l.clipped { "layer.releaseClippingMask" } else { "layer.createClippingMask" },
    )));
    v.push(None);
    v.push(Some((tl!("Link Layers"), "layer.linkLayers")));
    v.push(Some((tl!("Select Linked Layers"), "layer.selectLinkedLayers")));
    v.push(None);
    v.push(Some((tl!("Copy Layer Style"), "layer.layerStyle.copyLayerStyle")));
    v.push(Some((tl!("Paste Layer Style"), "layer.layerStyle.pasteLayerStyle")));
    v.push(Some((tl!("Clear Layer Style"), "layer.layerStyle.clear")));
    v.push(None);
    if multi {
        v.push(Some((tl!("Merge Layers"), "layer.mergeLayers")));
    } else if matches!(l.content, LayerContent::Group(_)) {
        // Photoshop's Merge Group: a group merges itself, not into the layer below (#3832).
        v.push(Some((tl!("Merge Group"), "layer.mergeGroup")));
    } else {
        v.push(Some((tl!("Merge Down"), "layer.mergeDown")));
    }
    v.push(Some((tl!("Merge Visible"), "layer.mergeVisible")));
    v.push(Some((tl!("Flatten Image"), "layer.flattenImage")));
    v
}

/// The layer style menu of an effect row ("Effects" or one effect) under a layer (#2846), in
/// Photoshop's order. It acts on that layer, like Layer ▸ Layer Style.
pub fn effects_entries() -> Vec<Entry> {
    vec![
        Some((tl!("Blending Options…"), "layer.layerStyle.blendingOptions")),
        None,
        Some((tl!("Copy Layer Style"), "layer.layerStyle.copyLayerStyle")),
        Some((tl!("Paste Layer Style"), "layer.layerStyle.pasteLayerStyle")),
        Some((tl!("Clear Layer Style"), "layer.layerStyle.clear")),
        None,
        Some((tl!("Global Light…"), "layer.layerStyle.globalLight")),
        Some((tl!("Create Layer"), "layer.layerStyle.createLayer")),
        Some((tl!("Hide All Effects"), "layer.layerStyle.hideAllEffects")),
        Some((tl!("Scale Effects…"), "layer.layerStyle.scaleEffects")),
    ]
}

/// Render the menu. Pushes `(command, params)` actions; `Value::Null` params mean "invoke like the
/// menu item" (opens the command's dialog when it has one).
pub fn show(app: &crate::PhotocraftApp, ui: &mut egui::Ui, l: &Layer, on_set: bool, actions: &mut Vec<(String, Value)>) -> bool {
    crate::widgets::menu_scroll(ui, |ui| {
        ui.set_min_width(220.0);
        let mut rename = false;
        let has_selection = app.session.active().is_some_and(|s| s.doc.selection.is_some());
        items(app, ui, l, on_set, entries(l, on_set, has_selection), actions);
        ui.separator();
        if ui.button(tl!("Rename Layer…")).clicked() {
            rename = true;
            ui.close();
        }
        color_menu(app, ui, l, on_set, actions);
        rename
    })
}

/// The menu of `l`'s layer-mask thumbnail (#4575), in Photoshop's order. It acts on `l`, which it
/// selects first like the row menu. Photoshop's Mask Options… (the overlay colour) has no
/// command yet, so it is left out.
pub fn show_mask(app: &crate::PhotocraftApp, ui: &mut egui::Ui, l: &Layer, actions: &mut Vec<(String, Value)>) {
    let Some(mask) = &l.mask else { return };
    crate::widgets::menu_scroll(ui, |ui| {
        ui.set_min_width(220.0);
        let edits = vec![
            Some((tl!(mask_toggle_label(mask.enabled)), "layer.layerMask.enabled")),
            Some((tl!("Delete Layer Mask"), "layer.layerMask.delete")),
            Some((tl!("Apply Layer Mask"), "layer.layerMask.apply")),
        ];
        items(app, ui, l, false, edits, actions);
        ui.separator();
        // ⌘⇧-, ⌘⌥- and ⌘⇧⌥-clicking the thumbnail, as menu items.
        let loads =
            [(tl!("Add Mask to Selection"), "add"), (tl!("Subtract Mask from Selection"), "subtract"), (tl!("Intersect Mask with Selection"), "intersect")];
        for (label, operation) in loads {
            if ui.button(label).clicked() {
                actions.push(("layer.select".into(), json!({"layer": l.id.0})));
                actions.push(("select.loadSelection".into(), json!({"channel": "mask", "layer": l.id.0, "operation": operation})));
                ui.close();
            }
        }
        ui.separator();
        if ui.add_enabled(crate::menus::is_enabled(app, "select.selectAndMask"), egui::Button::new(tl!("Select and Mask…"))).clicked() {
            // Like double-clicking the thumbnail with that preference on: it refines the mask.
            actions.push(("layer.select".into(), json!({"layer": l.id.0})));
            actions.push(("ui.maskTarget".into(), json!(true)));
            actions.push(("select.selectAndMask".into(), Value::Null));
            ui.close();
        }
    });
}

/// Render the layer style menu of `l`'s effect rows ([`effects_entries`]); the actions select `l`
/// first when it isn't the active layer.
pub fn show_effects(app: &crate::PhotocraftApp, ui: &mut egui::Ui, l: &Layer, actions: &mut Vec<(String, Value)>) {
    crate::widgets::menu_scroll(ui, |ui| {
        ui.set_min_width(200.0);
        items(app, ui, l, false, effects_entries(), actions);
    });
}

/// The command items of a layer menu, separators collapsed.
fn items(app: &crate::PhotocraftApp, ui: &mut egui::Ui, l: &Layer, on_set: bool, entries: Vec<Entry>, actions: &mut Vec<(String, Value)>) {
    let mut last_sep = true;
    for e in entries {
        if app.services.host_save.is_some() && e.is_some_and(|(_, id)| crate::menus::host_manages(id)) {
            continue;
        }
        match e {
            None => {
                if !last_sep {
                    ui.separator();
                }
                last_sep = true;
            }
            Some((label, id)) => {
                // Skip commands this build doesn't have rather than showing dead items.
                if photocraft_engine::commands::find(id).is_none() && !crate::menu_catalog::CATALOG.iter().any(|m| m.3 == id) {
                    continue;
                }
                last_sep = false;
                // Enablement is exact for the active layer (or the selection); another row is
                // selected first when clicked, so its items stay available.
                let is_active = app.session.active().is_some_and(|s| s.active_layer == Some(l.id));
                let enabled = !(on_set && single_layer_only(id)) && if on_set || is_active { crate::menus::is_enabled(app, id) } else { true };
                if ui.add_enabled(enabled, egui::Button::new(tl!(&label))).clicked() {
                    if !on_set {
                        actions.push(("layer.select".into(), json!({"layer": l.id.0})));
                    }
                    actions.push((id.into(), Value::Null));
                    ui.close();
                }
            }
        }
    }
}

pub(crate) fn color_name(color: LabelColor) -> &'static str {
    crate::i18n::tr_ctx(crate::i18n::current(), "layerLabel", color.label())
}

fn common_color(app: &crate::PhotocraftApp, l: &Layer, on_set: bool) -> Option<LabelColor> {
    if !on_set {
        return Some(l.label);
    }
    let st = app.session.active()?;
    let ids = st.selected_layers();
    let mut labels = ids.iter().filter_map(|id| st.doc.layer(*id).map(|l| l.label));
    let first = labels.next()?;
    labels.all(|c| c == first).then_some(first)
}

fn color_menu(app: &crate::PhotocraftApp, ui: &mut egui::Ui, l: &Layer, on_set: bool, actions: &mut Vec<(String, Value)>) {
    ui.menu_button(tl!("Color"), |ui| {
        color_choices(app, ui, l, on_set, actions);
    });
}

/// The eye column opens the same color choices directly, without the row's full context menu.
pub(crate) fn color_choices(app: &crate::PhotocraftApp, ui: &mut egui::Ui, l: &Layer, on_set: bool, actions: &mut Vec<(String, Value)>) {
    let current = common_color(app, l, on_set);
    crate::widgets::menu_scroll(ui, |ui| {
        ui.set_min_width(170.0);
        for color in LabelColor::ALL {
            if color == LabelColor::Red {
                ui.separator();
            }
            if color_button(ui, color, current == Some(color)).clicked() {
                let params = if on_set {
                    json!({"color": color.id()})
                } else {
                    actions.push(("layer.select".into(), json!({"layer": l.id.0})));
                    json!({"layer": l.id.0, "color": color.id()})
                };
                actions.push(("layer.setLabelColor".into(), params));
                ui.close();
            }
        }
    });
}

fn color_button(ui: &mut egui::Ui, color: LabelColor, checked: bool) -> egui::Response {
    use egui::{Atom, Rect, Stroke, StrokeKind, vec2};
    let t = crate::theme::Tokens::get(ui.ctx());
    let check_id = ui.id().with(("label-check", color.id()));
    let swatch_id = ui.id().with(("label-swatch", color.id()));
    let name = color_name(color);
    let button = egui::Button::new((Atom::custom(check_id, vec2(14.0, 16.0)), Atom::custom(swatch_id, vec2(16.0, 16.0)), name, Atom::grow())).atom_ui(ui);
    if checked && let Some(rect) = button.rect(check_id) {
        crate::icons::paint(ui, rect, "check", 14.0, t.icon);
    }
    if let Some(rect) = button.rect(swatch_id) {
        let rect = Rect::from_center_size(rect.center(), vec2(16.0, 16.0));
        let p = ui.painter();
        if let Some((swatch, _)) = t.layer_label_colors(color) {
            p.rect_filled(rect, t.radius_sm, swatch);
        } else {
            let stroke = Stroke::new(1.0, t.text_dim);
            p.rect_stroke(rect, t.radius_sm, stroke, StrokeKind::Inside);
            let r = rect.shrink(4.0);
            p.line_segment([r.left_top(), r.right_bottom()], stroke);
            p.line_segment([r.right_top(), r.left_bottom()], stroke);
        }
    }
    button.response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, ui.is_enabled(), checked, name));
    button.response
}

#[cfg(test)]
mod color_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_layer_menu_uses_the_same_page_commands_as_the_menu_bar() {
        use egui_kittest::{Harness, kittest::Queryable};
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width":32,"height":24})).unwrap();
        s.execute("layer.new.layer", json!({"name":"Material"})).unwrap();
        s.execute("paint.stroke", json!({"points":[[12,12]],"size":8})).unwrap();
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        let mut app = crate::PhotocraftApp::new(s, Default::default());
        app.services.host_save = Some(Box::new(|| {}));
        let st = app.session.active().unwrap();
        let layer = st.doc.layer(st.active_layer.unwrap()).unwrap().clone();
        let mut h = Harness::builder().with_size(egui::vec2(500.0, 1400.0)).build_ui_state(
            move |ui, app: &mut crate::PhotocraftApp| {
                show(app, ui, &layer, false, &mut Vec::new());
            },
            app,
        );
        h.run_steps(3);
        assert!(h.query_by_label("Duplicate Layer…").is_some());
        assert!(h.query_by_label("Rasterize Layer").is_some());
        assert!(h.query_by_label("Artboard from Layers…").is_none());
        assert!(h.query_by_label("Edit Contents").is_none());
    }

    #[test]
    fn mask_toggle_label_follows_history_and_selected_layer() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        let mut app = crate::PhotocraftApp::new(s, Default::default());
        let toggle = "layer.layerMask.enabled";
        let check = |app: &crate::PhotocraftApp, label: Option<&str>| {
            let st = app.session.active().unwrap();
            let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
            let entry = entries(l, false, st.doc.selection.is_some()).into_iter().flatten().find(|e| e.1 == toggle);
            assert_eq!(entry.map(|e| e.0), label);
            // A selection has no single mask to toggle: its menu never offers the item.
            assert!(entries(l, true, st.doc.selection.is_some()).into_iter().flatten().all(|e| e.1 != toggle));
            let item = crate::menus::menu_items(app).into_iter().find(|i| i.id == toggle).unwrap();
            assert_eq!(item.enabled, label.is_some());
            assert_eq!(item.label, label.unwrap_or("Enable Layer Mask"));
            for lang in [crate::i18n::Lang::EN, crate::i18n::Lang::from_pref("fr")] {
                let layout = crate::native_menu::photocraft_layout(&crate::menus::menu_items(app), lang, lang.code());
                let native = layout.bar.find(toggle).unwrap();
                assert_eq!(native.label, crate::i18n::tr(lang, &item.label));
                assert_eq!(native.enabled, item.enabled);
            }
        };
        check(&app, None);
        app.run("layer.layerMask.revealAll", json!({})).unwrap();
        let masked = app.session.active().unwrap().active_layer.unwrap();
        check(&app, Some("Disable Layer Mask"));
        app.run(toggle, json!({})).unwrap();
        check(&app, Some("Enable Layer Mask"));
        app.run("edit.undo", json!({})).unwrap();
        check(&app, Some("Disable Layer Mask"));
        app.run("edit.redo", json!({})).unwrap();
        check(&app, Some("Enable Layer Mask"));
        app.run("layer.new.layer", json!({})).unwrap();
        check(&app, None);
        app.run("layer.layerMask.revealAll", json!({})).unwrap();
        check(&app, Some("Disable Layer Mask"));
        app.run("layer.select", json!({"layer": masked.0, "mode": "add"})).unwrap();
        check(&app, Some("Enable Layer Mask"));
        app.run("select.rect", json!({"x": 0, "y": 0, "width": 4, "height": 4})).unwrap();
        check(&app, Some("Enable Layer Mask"));
        app.session.active_mut().unwrap().active_layer = None;
        assert!(!crate::menus::is_enabled(&app, toggle));
        let empty = crate::PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        assert!(!crate::menus::is_enabled(&empty, toggle));
    }

    #[test]
    fn mask_toggle_context_menu_follows_the_clicked_layer_and_existing_translations() {
        use egui_kittest::{Harness, kittest::Queryable};

        for language in ["en", "fr"] {
            let lang = crate::i18n::Lang::from_pref(language);
            let _language = crate::i18n::language_scope(lang);
            let mut s = photocraft_engine::Session::new();
            s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            s.execute("layer.layerMask.revealAll", json!({})).unwrap();
            let clicked = s.active().unwrap().active_layer.unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            let other = s.active().unwrap().active_layer.unwrap();
            s.execute("layer.layerMask.revealAll", json!({})).unwrap();
            s.execute("layer.layerMask.enabled", json!({})).unwrap();
            let app = crate::PhotocraftApp::new(s, Default::default());
            // `other` stays the active layer with its mask disabled; the menu is opened over `clicked`,
            // whose mask is enabled, and selects it first (a single-layer menu).
            let mut h = Harness::builder().with_size(egui::vec2(400.0, 900.0)).build_ui_state(
                move |ui, app| {
                    let st = app.session.active().unwrap();
                    let l = st.doc.layer(clicked).unwrap().clone();
                    let mut actions = Vec::new();
                    show(app, ui, &l, false, &mut actions);
                    for (id, params) in actions {
                        crate::menus::invoke(app, ui.ctx(), &id, params).unwrap();
                    }
                },
                app,
            );
            h.run_steps(3);
            let enabled_label = crate::i18n::tr(lang, "Enable Layer Mask");
            let disabled_label = crate::i18n::tr(lang, "Disable Layer Mask");
            // The label follows the mask of the clicked layer, not of the active one.
            assert!(h.query_by_label(enabled_label).is_none());
            h.get_by_label(disabled_label).click();
            h.run_steps(3);
            assert_eq!(h.state().session.active().unwrap().active_layer, Some(clicked));
            assert!(!h.state().session.active().unwrap().doc.layer(clicked).unwrap().mask.as_ref().unwrap().enabled);
            assert!(!h.state().session.active().unwrap().doc.layer(other).unwrap().mask.as_ref().unwrap().enabled);
            h.get_by_label(enabled_label);
        }
    }

    /// With several layers selected there is no single mask to toggle, apply or delete: the menu
    /// offers Add Layer Mask only, whatever the clicked layer has, and does not hit the active layer.
    #[test]
    fn multi_selection_menu_has_no_per_layer_mask_items() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        let st = s.active().unwrap();
        let masked = st.doc.layer(st.active_layer.unwrap()).unwrap().clone();
        let ids = entries(&masked, true, false).into_iter().flatten().map(|e| e.1).collect::<Vec<_>>();
        assert!(ids.contains(&"layer.layerMask.revealAll"));
        for single in ["layer.layerMask.enabled", "layer.layerMask.apply", "layer.layerMask.delete"] {
            assert!(!ids.contains(&single), "{single} acts on one layer, not on a selection");
        }
    }

    #[test]
    fn smart_object_context_menu_edits_the_clicked_layer() {
        use crate::PhotocraftApp;
        use egui::{Event, Modifiers, PointerButton, Pos2, pos2, vec2};
        use egui_kittest::{Harness, kittest::Queryable};

        fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, button: PointerButton) {
            h.hover_at(at);
            h.step();
            for pressed in [true, false] {
                h.event(Event::PointerButton { pos: at, button, pressed, modifiers: Modifiers::NONE });
                h.step();
            }
            h.run_steps(3);
        }

        for ppp in [1.0, 2.0] {
            let mut s = photocraft_engine::Session::new();
            s.execute("file.new", json!({"width": 640, "height": 480})).unwrap();
            s.execute("layer.new.layer", json!({"name": "Smart source"})).unwrap();
            s.execute("paint.stroke", json!({"points": [[100, 100]], "size": 20})).unwrap();
            let smart = s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap()["layer"].as_u64().unwrap();
            s.execute("layer.new.layer", json!({"name": "Other active layer"})).unwrap();
            let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_pixels_per_point(ppp).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(
                move |cc| {
                    PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                    PhotocraftApp::new(s, crate::Services::default())
                },
            );
            let ctx = h.ctx.clone();
            let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
            crate::control::handle(h.state_mut(), &ctx, &req);
            h.run_steps(8);
            let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == smart).expect("layer row drawn").row;
            // Top-level Pro row: 6 pt padding, 28 pt eye column, 24 pt layer thumbnail.
            let at = pos2(row.left() + 6.0 + 28.0 + 12.0, row.center().y);
            click(&mut h, at, PointerButton::Secondary);
            let at = h.get_by_label("Edit Contents").rect().center();
            click(&mut h, at, PointerButton::Primary);
            assert_eq!(h.state().session.active_index(), Some(1), "@{ppp}x: contents tab opens");
            assert!(h.state().session.is_enabled("layer.smartObjects.saveContents"));
            assert!(h.state().session.active().unwrap().doc.walk().iter().any(|(_, _, l)| l.name == "Smart source"));
            h.state_mut().session.set_active(0);
            assert_eq!(h.state().session.active().unwrap().active_layer, Some(photocraft_doc::LayerId(smart)));
        }
    }

    /// #4575: right-clicking a layer-mask thumbnail opens Photoshop's mask menu, not the layer
    /// menu, and its selection items load the clicked layer's mask (not the active layer's).
    #[test]
    fn mask_thumbnail_context_menu_acts_on_the_clicked_mask() {
        use crate::PhotocraftApp;
        use egui::{Event, Modifiers, PointerButton, Pos2, vec2};
        use egui_kittest::{Harness, kittest::Queryable};

        fn click(h: &mut Harness<'_, PhotocraftApp>, at: Pos2, button: PointerButton) {
            h.hover_at(at);
            h.step();
            for pressed in [true, false] {
                h.event(Event::PointerButton { pos: at, button, pressed, modifiers: Modifiers::NONE });
                h.step();
            }
            h.run_steps(3);
        }

        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 100, "height": 60})).unwrap();
        s.execute("layer.new.layer", json!({"name": "Masked"})).unwrap();
        s.execute("select.rect", json!({"x": 60, "y": 0, "width": 40, "height": 60})).unwrap();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        let masked = s.active().unwrap().active_layer.unwrap();
        // Another layer is active, and the selection doesn't overlap the mask.
        s.execute("layer.new.layer", json!({"name": "Other active layer"})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 60})).unwrap();
        let mut h = Harness::builder().with_size(vec2(1440.0, 900.0)).with_step_dt(1.0 / 60.0).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            PhotocraftApp::new(s, crate::Services::default())
        });
        let ctx = h.ctx.clone();
        let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
        crate::control::handle(h.state_mut(), &ctx, &req);
        h.run_steps(8);
        let (thumbs, _) = crate::mask_thumbs_ui::recorded(&h.ctx, masked.0).expect("mask thumbnail drawn");
        let at = thumbs.iter().find(|(k, _)| *k == crate::mask_thumbs_ui::MaskKind::Pixel).unwrap().1.center();
        click(&mut h, at, PointerButton::Secondary);
        assert!(h.query_by_label("Blending Options…").is_none(), "the mask menu, not the layer menu");
        for item in
            ["Disable Layer Mask", "Delete Layer Mask", "Apply Layer Mask", "Subtract Mask from Selection", "Intersect Mask with Selection", "Select and Mask…"]
        {
            h.get_by_label(item);
        }
        let at = h.get_by_label("Add Mask to Selection").rect().center();
        click(&mut h, at, PointerButton::Primary);
        let st = h.state().session.active().unwrap();
        assert_eq!(st.active_layer, Some(masked));
        let sel = st.doc.selection.as_ref().expect("a selection");
        assert!(sel.sample_channel(10, 30, 0) > 0.99, "the selection is kept");
        assert!(sel.sample_channel(80, 30, 0) > 0.99, "the clicked layer's mask is added");
        assert!(sel.sample_channel(40, 30, 0) < 0.01, "nothing else is selected");
    }

    #[test]
    fn a_long_layer_menu_stays_inside_a_short_window_and_scrolls() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 32, "height": 24})).unwrap();
        s.execute("layer.new.layer", json!({"name": "Paint"})).unwrap();
        let app = crate::PhotocraftApp::new(s, crate::Services::default());
        let layer = {
            let st = app.session.active().unwrap();
            st.doc.layer(st.active_layer.unwrap()).unwrap().clone()
        };
        let ctx = egui::Context::default();
        crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        let screen = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 300.0));
        let id = egui::Id::new("layer-menu-probe");
        let mut content = 0.0;
        for _ in 0..4 {
            let input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
            let mut out = ctx.run_ui(input, |ui| {
                // Opened low in the window, like a right-click on a bottom Layers row.
                egui::Area::new(id).order(egui::Order::Foreground).default_pos(egui::pos2(300.0, 250.0)).show(ui.ctx(), |ui| {
                    egui::Frame::menu(ui.style()).show(ui, |ui| {
                        let start = ui.next_widget_position().y;
                        show(&app, ui, &layer, false, &mut Vec::new());
                        content = ui.min_rect().bottom() - start;
                    });
                });
            });
            out.textures_delta.clear();
        }
        let rect = ctx.memory(|m| m.area_rect(id)).expect("the menu was shown");
        assert!(screen.contains_rect(rect), "the menu moves up and stays inside the window: {rect:?}");
        assert!(rect.height() < screen.height(), "{rect:?}");
        let rows = entries(&layer, false, false).iter().flatten().count();
        assert!(rows >= 15, "Photoshop's layer menu is long: {rows} rows");
        assert!(content < rect.height() + 1.0, "the rows scroll inside the menu instead of running off the window");
    }

    #[test]
    fn edit_contents_is_available_only_for_smart_objects() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 32, "height": 24})).unwrap();
        s.execute("layer.new.layer", json!({"name": "Source"})).unwrap();
        let layer = |s: &photocraft_engine::Session| {
            let st = s.active().unwrap();
            st.doc.layer(st.active_layer.unwrap()).unwrap().clone()
        };
        assert!(!entries(&layer(&s), false, false).iter().flatten().any(|e| e.1 == "layer.smartObjects.editContents"));
        s.execute("paint.stroke", json!({"points": [[12, 12]], "size": 8})).unwrap();
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        let ids: Vec<_> = entries(&layer(&s), false, false).into_iter().flatten().map(|e| e.1).collect();
        let edit = ids.iter().position(|id| *id == "layer.smartObjects.editContents").expect("Edit Contents entry");
        assert_eq!(ids[edit + 1..edit + 3], ["layer.smartObjects.convertToLayers", "layer.rasterize.smartObject"]);
    }

    /// Right-clicking inside a multi-layer selection offers one Rasterize item for the whole
    /// selection, whatever kind the clicked layer is, and no single-layer Smart Object items.
    #[test]
    fn multi_selection_menu_rasterizes_the_selection() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 32, "height": 24})).unwrap();
        s.execute("layer.new.layer", json!({"name": "Source"})).unwrap();
        s.execute("paint.stroke", json!({"points": [[12, 12]], "size": 8})).unwrap();
        s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
        let st = s.active().unwrap();
        let smart = st.doc.layer(st.active_layer.unwrap()).unwrap().clone();
        let ids = |multi| entries(&smart, multi, false).into_iter().flatten().map(|e| e.1).collect::<Vec<_>>();
        assert!(ids(false).contains(&"layer.smartObjects.editContents"), "a single smart object keeps its items");
        let multi = ids(true);
        assert!(multi.contains(&"layer.rasterize.layer"));
        for single in ["layer.rasterize.smartObject", "layer.rasterize.shape", "layer.smartObjects.editContents", "layer.smartObjects.convertToLayers"] {
            assert!(!multi.contains(&single), "{single} acts on one layer, not on a selection");
        }
        let label = entries(&smart, true, false).into_iter().flatten().find(|e| e.1 == "layer.rasterize.layer").unwrap().0;
        assert_eq!(label, "Rasterize Layers");
    }

    fn click(h: &mut egui_kittest::Harness<'_, crate::PhotocraftApp>, at: egui::Pos2, button: egui::PointerButton) {
        h.hover_at(at);
        h.step();
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: at, button, pressed, modifiers: egui::Modifiers::NONE });
            h.step();
        }
        h.run_steps(3);
    }

    /// A type layer, an ellipse and a rectangle; the type layer and the ellipse are selected (the
    /// ellipse active), the rectangle stays out. The Layers menu is opened by right-clicking the
    /// type row, which is selected but not the active layer. Returns `[type, ellipse, rectangle]`.
    fn open_multi_selection_menu() -> (egui_kittest::Harness<'static, crate::PhotocraftApp>, [u64; 3]) {
        use crate::PhotocraftApp;
        use egui::{PointerButton, pos2, vec2};

        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 640, "height": 480})).unwrap();
        let text = s.execute("type.create", json!({"text": "Hi", "size": 24, "x": 20, "y": 60})).unwrap()["layer"].as_u64().unwrap();
        let shape = s.execute("shape.create", json!({"kind": "ellipse", "rect": [100, 100, 200, 160]})).unwrap()["layer"].as_u64().unwrap();
        let other = s.execute("shape.create", json!({"kind": "rect", "rect": [300, 100, 380, 160]})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.select", json!({"layer": text})).unwrap();
        s.execute("layer.select", json!({"layer": shape, "mode": "add"})).unwrap();
        let mut h = egui_kittest::Harness::builder()
            .with_size(vec2(1440.0, 900.0))
            .with_pixels_per_point(1.0)
            .with_step_dt(1.0 / 60.0)
            .with_max_steps(64)
            .build_eframe(move |cc| {
                PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                PhotocraftApp::new(s, crate::Services::default())
            });
        let ctx = h.ctx.clone();
        let (req, _rx) = crate::control::ControlRequest::new("ui.set", json!({"dock": {"collapsed": ["color", "properties", "history", "navigator"]}}));
        crate::control::handle(h.state_mut(), &ctx, &req);
        h.run_steps(8);
        let row = crate::layer_row_ui::recorded(&h.ctx).into_iter().find(|r| r.layer == text).expect("layer row drawn").row;
        click(&mut h, pos2(row.left() + 6.0 + 28.0 + 12.0, row.center().y), PointerButton::Secondary);
        (h, [text, shape, other])
    }

    /// The reported bug: right-click a layer inside a multi-layer selection, choose Rasterize, and
    /// only the clicked (or active) layer was converted.
    #[test]
    fn rasterize_from_the_menu_converts_every_selected_layer() {
        use egui_kittest::kittest::Queryable;
        let (mut h, [text, shape, other]) = open_multi_selection_menu();
        let at = h.get_by_label("Rasterize Layers").rect().center();
        click(&mut h, at, egui::PointerButton::Primary);
        let doc = &h.state().session.active().unwrap().doc;
        let raster = |id: u64| matches!(doc.layer(photocraft_doc::LayerId(id)).unwrap().content, photocraft_doc::LayerContent::Raster(_));
        assert!(raster(text) && raster(shape), "both selected layers are rasterized");
        assert!(!raster(other), "an unselected layer is left alone");
        assert_eq!(h.state().session.active().unwrap().selected_layers().len(), 2, "the selection survives");
    }

    /// Items with no single target are greyed for a multi-layer selection (they used to run on
    /// whichever layer happened to be active, not the one right-clicked); the ones that act on the
    /// whole selection, including adding a mask, stay available.
    #[test]
    fn multi_selection_menu_greys_what_acts_on_one_layer() {
        use egui_kittest::kittest::{NodeT, Queryable};
        let (h, _) = open_multi_selection_menu();
        for label in ["Blending Options…", "Copy Layer Style"] {
            assert!(h.get_by_label(label).accesskit_node().is_disabled(), "{label} works on one layer only");
        }
        for label in [
            "Quick Export As PNG",
            "Export As…",
            "Add Layer Mask",
            "Duplicate Layers…",
            "Delete Layers",
            "Convert to Smart Object",
            "Rasterize Layers",
            "Link Layers",
            "Merge Layers",
        ] {
            assert!(!h.get_by_label(label).accesskit_node().is_disabled(), "{label} acts on the selection");
        }
        // Greyed here only for lack of something to act on: no style on the clipboard, no style to clear.
        for label in ["Paste Layer Style", "Clear Layer Style"] {
            assert!(h.get_by_label(label).accesskit_node().is_disabled(), "{label} has nothing to do here");
        }
        // Whatever the clicked layer is, the menu has the same shape: no per-layer mask or Smart Object items.
        for label in ["Disable Layer Mask", "Apply Layer Mask", "Delete Layer Mask", "Edit Contents", "Convert to Layers"] {
            assert!(h.query_by_label(label).is_none(), "{label} is not offered for several layers");
        }
    }

    /// Every entry of the multi-layer menu is either known to act on the selection or greyed by
    /// [`single_layer_only`]; a new single-layer command added to the menu must be classified.
    #[test]
    fn single_layer_only_covers_the_per_layer_commands() {
        for id in [
            "layer.layerStyle.blendingOptions",
            "layer.layerStyle.copyLayerStyle",
            "layer.layerMask.enabled",
            "layer.layerMask.apply",
            "layer.layerMask.delete",
        ] {
            assert!(single_layer_only(id), "{id}");
        }
        for id in [
            "layer.quickExportAsPng",
            "layer.exportAs",
            "layer.duplicate",
            "layer.delete",
            "layer.rasterize.layer",
            "layer.smartObjects.convertToSmartObject",
            "layer.createClippingMask",
            "layer.linkLayers",
            "layer.mergeLayers",
            "layer.layerMask.revealAll",
            "layer.layerMask.hideSelection",
            "layer.layerStyle.pasteLayerStyle",
            "layer.layerStyle.clear",
        ] {
            assert!(!single_layer_only(id), "{id}");
        }
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        let st = s.active().unwrap();
        let masked = st.doc.layer(st.active_layer.unwrap()).unwrap().clone();
        let ids: Vec<_> = entries(&masked, true, false).into_iter().flatten().map(|e| e.1).collect();
        assert!(ids.contains(&"layer.layerMask.revealAll") && !ids.contains(&"layer.layerMask.apply"), "one stable mask item for several layers");
    }

    #[test]
    fn entries_follow_layer_state() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        let st = s.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap().clone();
        let ids: Vec<&str> = entries(&l, false, false).into_iter().flatten().map(|e| e.1).collect();
        assert_eq!(ids.first(), Some(&"layer.layerStyle.blendingOptions"));
        assert!(ids.contains(&"layer.mergeDown") && !ids.contains(&"layer.mergeLayers"));
        assert!(ids.contains(&"layer.layerMask.revealAll"));
        let ids: Vec<&str> = entries(&l, false, true).into_iter().flatten().map(|e| e.1).collect();
        assert!(ids.contains(&"layer.layerMask.revealSelection") && !ids.contains(&"layer.layerMask.revealAll"));
        let ids: Vec<&str> = entries(&l, true, false).into_iter().flatten().map(|e| e.1).collect();
        assert!(ids.contains(&"layer.mergeLayers"));
    }

    /// #3832: a group's right-click menu and the Layer menu offer Photoshop's Merge Group, which
    /// merges the group itself (not into the layer below), and it is enabled.
    #[test]
    fn a_group_offers_merge_group() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("layer.groupLayers", json!({"name": "G"})).unwrap();
        let st = s.active().unwrap();
        let g = st.doc.layer(st.active_layer.unwrap()).unwrap().clone();
        let e: Vec<(&str, &str)> = entries(&g, false, false).into_iter().flatten().collect();
        assert!(e.contains(&("Merge Group", "layer.mergeGroup")), "{e:?}");
        assert!(!e.iter().any(|x| x.0 == "Merge Down"));
        let mut app = crate::PhotocraftApp::new(s, Default::default());
        app.sync_views();
        let items = crate::menus::menu_items(&app);
        let item = items.iter().find(|i| i.id == "layer.mergeLayers").unwrap();
        assert_eq!(item.label, "Merge Group");
        assert!(item.enabled, "Merge Group runs on a group");
        // A plain layer keeps Merge Layers.
        app.run("layer.new.layer", json!({})).unwrap();
        let items = crate::menus::menu_items(&app);
        assert_eq!(items.iter().find(|i| i.id == "layer.mergeLayers").unwrap().label, "Merge Layers");
    }

    #[test]
    fn add_mask_uses_the_selection() {
        assert_eq!(add_mask_command(false, false), "layer.layerMask.revealAll");
        assert_eq!(add_mask_command(false, true), "layer.layerMask.hideAll");
        assert_eq!(add_mask_command(true, true), "layer.layerMask.hideSelection");
        // With a selection the new mask is the selection: inside revealed, outside hidden.
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 4, "height": 10})).unwrap();
        s.execute(add_mask_command(true, false), json!({})).unwrap();
        let st = s.active().unwrap();
        let mask = &st.doc.layer(st.active_layer.unwrap()).unwrap().mask.as_ref().unwrap().surface;
        assert!(mask.pixel(1, 5)[0] > 0.99 && mask.pixel(8, 5)[0] < 0.01);
    }

    /// #2075: clicking the mask button again (with or without ⌥) used to replace the layer's
    /// mask, discarding what was painted into it. Like Photoshop, it now adds a vector mask, and
    /// does nothing once the layer has both.
    #[test]
    fn mask_button_never_replaces_a_mask() {
        for alt in [false, true] {
            let mut s = photocraft_engine::Session::new();
            s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
            s.execute("layer.new.layer", json!({})).unwrap();
            let layer = |s: &photocraft_engine::Session| {
                let st = s.active().unwrap();
                st.doc.layer(st.active_layer.unwrap()).unwrap().clone()
            };
            let first = mask_button_command(Some(&layer(&s)), false, alt).unwrap();
            assert_eq!(first, if alt { "layer.layerMask.hideAll" } else { "layer.layerMask.revealAll" });
            s.execute(first, json!({})).unwrap();
            // Paint a stroke into the mask.
            s.edit("paint mask", |doc, active| {
                let id = (*active).ok_or(photocraft_engine::EngineError::NoDocument)?;
                let m = doc.layer_mut(id).and_then(|l| l.mask.as_mut()).ok_or(photocraft_engine::EngineError::NoDocument)?;
                m.surface.fill_rect(photocraft_geom::Rect::new(2, 2, 5, 5), &[0.5]);
                Ok(())
            })
            .unwrap();
            let painted = layer(&s).mask.unwrap().surface.pixel(3, 3)[0];
            assert!((painted - 0.5).abs() < 0.01);
            // Second click: a vector mask (⌥ hides all), the layer mask is untouched.
            let second = mask_button_command(Some(&layer(&s)), false, alt).unwrap();
            assert_eq!(second, if alt { "layer.vectorMask.hideAll" } else { "layer.vectorMask.revealAll" });
            s.execute(second, json!({})).unwrap();
            let l = layer(&s);
            assert!(l.vector_mask.is_some(), "alt={alt}");
            let mask = l.mask.as_ref().unwrap();
            assert!((mask.surface.pixel(3, 3)[0] - painted).abs() < 1e-6, "alt={alt}: the painted mask was replaced");
            // Third click: nothing left to add.
            assert_eq!(mask_button_command(Some(&l), false, alt), None);
            assert_eq!(mask_button_command(Some(&l), true, alt), None);
        }
        // No layer: nothing to do.
        assert_eq!(mask_button_command(None, false, false), None);
        // A shape layer's path is already its vector mask: after the layer mask, nothing more.
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        s.execute("shape.create", json!({"kind": "rect", "rect": [0, 0, 5, 5]})).unwrap();
        s.execute("layer.layerMask.revealAll", json!({})).unwrap();
        let st = s.active().unwrap();
        let l = st.doc.layer(st.active_layer.unwrap()).unwrap();
        assert_eq!(mask_button_command(Some(l), false, false), None);
    }

    #[test]
    fn every_entry_is_a_known_command() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        let l = s.active().unwrap().doc.layers[0].clone();
        for (_, id) in entries(&l, false, true).into_iter().chain(entries(&l, false, false)).chain(entries(&l, true, false)).chain(effects_entries()).flatten()
        {
            assert!(crate::menu_catalog::CATALOG.iter().any(|m| m.3 == id) || photocraft_engine::commands::find(id).is_some(), "{id}");
        }
    }
}
