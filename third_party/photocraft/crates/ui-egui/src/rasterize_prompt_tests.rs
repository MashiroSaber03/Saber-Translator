use egui::Modifiers;
use photocraft_doc::LayerContent;
use serde_json::json;

use super::*;
use crate::canvas::tool_event;

/// A document whose active layer is a `kind` layer over a white Background.
fn app_with(kind: Kind) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
    app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
    match kind {
        Kind::Type => app.run("type.create", json!({"x": 10, "y": 40, "text": "Hello"})).map(|_| ()),
        Kind::Shape => app.run("shape.create", json!({"kind": "rect", "rect": [10, 10, 60, 40], "fill": "#ff0000"})).map(|_| ()),
        Kind::SmartObject => {
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("edit.fill", json!({"color": "#00ff00"})).unwrap();
            app.run("layer.smartObjects.convertToSmartObject", json!({})).map(|_| ())
        }
        Kind::Fill => app.run("layer.newFillLayer.solidColor", json!({"color": "#00ff00"})).map(|_| ()),
    }
    .unwrap();
    // A new fill layer targets its mask (Photoshop); painting its content means targeting the
    // layer itself, as a click on its thumbnail does.
    app.ui.mask_target = false;
    app.sync_mask_targets();
    app.run("tools.setColors", json!({"foreground": "#0000ff"})).unwrap();
    app
}

fn active_content(app: &PhotocraftApp) -> &'static str {
    let st = app.session.active().unwrap();
    st.doc.layer(st.active_layer.unwrap()).unwrap().content.kind_name()
}

fn past(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().history.past_len()
}

fn click(app: &mut PhotocraftApp, x: f64, y: f64) {
    tool_event(app, ToolEvent::Down { x, y, pressure: 1.0 }, Modifiers::NONE);
    tool_event(app, ToolEvent::Up { x, y }, Modifiers::NONE);
}

fn prompt(app: &PhotocraftApp) -> Option<&crate::state::Dialog> {
    app.ui.dialogs.iter().find(|d| owns(&d.fields))
}

#[test]
fn ok_rasterizes_then_paints_as_two_history_states() {
    for kind in Kind::ALL {
        for tool in [Tool::Brush, Tool::Pencil] {
            let mut app = app_with(kind);
            app.ui.tool = tool;
            let before = past(&app);
            click(&mut app, 30.0, 20.0);
            let d = prompt(&app).unwrap_or_else(|| panic!("{kind:?}/{tool:?}: no prompt"));
            assert_eq!(d.fields["message"], json!(kind.message()));
            assert_eq!(past(&app), before, "nothing happens until the answer");
            let id = d.id;
            let r = crate::dialogs::confirm(&mut app, id).unwrap();
            assert_eq!(r["painted"], json!(true), "{kind:?}/{tool:?}");
            assert_eq!(active_content(&app), "Pixel");
            assert_eq!(past(&app), before + 2, "{kind:?}/{tool:?}: rasterize, then paint");
            let st = app.session.active().unwrap();
            let c = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().rgba(30, 20);
            assert!(c[2] > 0.99 && c[0] < 0.01, "{kind:?}/{tool:?}: painted blue at the click: {c:?}");
            // One undo takes the paint back; the layer stays rasterized.
            app.session.undo();
            assert_eq!(active_content(&app), "Pixel");
            app.session.undo();
            assert_eq!(active_content(&app), kind_name(kind));
        }
    }
}

fn kind_name(k: Kind) -> &'static str {
    match k {
        Kind::Type => "Type",
        Kind::Shape => "Shape",
        Kind::SmartObject => "Smart Object",
        Kind::Fill => "Fill",
    }
}

#[test]
fn cancel_does_nothing() {
    let mut app = app_with(Kind::Type);
    app.ui.tool = Tool::Brush;
    let before = (past(&app), app.session.active().unwrap().doc.clone());
    click(&mut app, 30.0, 20.0);
    let id = prompt(&app).unwrap().id;
    // A second press while it is up doesn't stack another prompt.
    click(&mut app, 40.0, 20.0);
    assert_eq!(app.ui.dialogs.len(), 1);
    app.ui.close_dialog(id).unwrap();
    assert_eq!(past(&app), before.0);
    assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &before.1) || *app.session.active().unwrap().doc == *before.1);
    assert_eq!(active_content(&app), "Type");
}

#[test]
fn only_pixel_painting_asks() {
    let mut app = app_with(Kind::Type);
    for tool in [Tool::Move, Tool::RectMarquee, Tool::Eyedropper, Tool::Hand, Tool::Type] {
        app.ui.tool = tool;
        click(&mut app, 30.0, 20.0);
        assert!(prompt(&app).is_none(), "{tool:?} must not ask");
        app.ui.text_edit = None;
    }
    for tool in [Tool::Eraser, Tool::CloneStamp, Tool::PaintBucket, Tool::Smudge] {
        app.ui.tool = tool;
        click(&mut app, 30.0, 20.0);
        let id = prompt(&app).unwrap_or_else(|| panic!("{tool:?} asks")).id;
        app.ui.close_dialog(id);
    }
    // ⌥-click with the Clone Stamp sets the source.
    app.ui.tool = Tool::CloneStamp;
    tool_event(&mut app, ToolEvent::Down { x: 30.0, y: 20.0, pressure: 1.0 }, Modifiers::ALT);
    assert!(prompt(&app).is_none());
    // Painting the layer's mask needs no rasterizing.
    app.run("layer.layerMask.revealAll", json!({})).unwrap();
    app.ui.mask_target = true;
    app.ui.tool = Tool::Brush;
    click(&mut app, 30.0, 20.0);
    assert!(prompt(&app).is_none());
    assert!(app.session.active().unwrap().doc.layers.iter().any(|l| matches!(l.content, LayerContent::Text(_))));
}

#[test]
fn a_stale_or_bad_prompt_fails_gracefully() {
    let mut app = app_with(Kind::Type);
    app.ui.tool = Tool::Brush;
    click(&mut app, 30.0, 20.0);
    let id = prompt(&app).unwrap().id;
    // The layer was rasterized elsewhere meanwhile: OK reports the error, nothing panics.
    app.run("layer.rasterize.type", json!({})).unwrap();
    assert!(crate::dialogs::confirm(&mut app, id).is_err());
    // Garbage fields.
    for f in [json!({"__rasterize": "plaid", "layer": 1}), json!({"__rasterize": "type"}), json!({"__rasterize": "type", "layer": u64::MAX})] {
        let id = app.ui.open_dialog(crate::state::DialogKind::Command, f.as_object().unwrap().clone());
        assert!(crate::dialogs::confirm(&mut app, id).is_err(), "{f}");
    }
    // A bad tool or point rasterizes without painting.
    let mut app = app_with(Kind::Fill);
    let layer = app.session.active().unwrap().active_layer.unwrap();
    let id = open(&mut app, Kind::Fill, layer, Tool::Brush, [f64::NAN, 1.0, 1.0]);
    app.ui.dialog_mut(id).unwrap().fields.insert("tool".into(), json!("no such tool"));
    let r = crate::dialogs::confirm(&mut app, id).unwrap();
    assert_eq!(r["painted"], json!(false));
    assert_eq!(active_content(&app), "Pixel");
}

#[test]
fn the_control_channel_sees_and_confirms_the_prompt() {
    let mut app = app_with(Kind::Shape);
    let ctx = egui::Context::default();
    app.ui.tool = Tool::Pencil;
    let call = |app: &mut PhotocraftApp, method: &str, params: serde_json::Value| {
        let (req, _rx) = crate::control::ControlRequest::new(method, params);
        match crate::control::handle(app, &ctx, &req) {
            crate::control::Outcome::Done(v) => v,
            _ => panic!("deferred"),
        }
    };
    call(&mut app, "ui.pointer", json!({"events": [{"kind": "down", "x": 30, "y": 20}, {"kind": "up", "x": 30, "y": 20}]}));
    let id = prompt(&app).map(|d| d.id).expect("prompt");
    let r = call(&mut app, "ui.dialog.confirm", json!({"dialog": id}));
    assert_eq!(r["ok"], json!(true), "{r}");
    assert_eq!(active_content(&app), "Pixel");
}

const BLUR: &str = "filter.blur.gaussianBlur";

fn menu(app: &mut PhotocraftApp, id: &str) -> Result<serde_json::Value, String> {
    crate::menus::invoke(app, &egui::Context::default(), id, json!({}))
}

/// Answers the filter prompt, then the frame runs the queued filter.
fn answer(app: &mut PhotocraftApp, id: u64) {
    crate::dialogs::confirm(app, id).unwrap();
    run_pending(app, &egui::Context::default());
}

fn filter_dialog(app: &PhotocraftApp) -> Option<u64> {
    app.ui.dialogs.iter().find(|d| d.fields.get("__command") == Some(&json!(BLUR)) && !owns(&d.fields)).map(|d| d.id)
}

/// #4303: Filter › Gaussian Blur… on a type layer is live and asks to rasterize first.
#[test]
fn a_filter_on_a_type_layer_asks_then_opens_on_the_rasterized_layer() {
    let mut app = app_with(Kind::Type);
    assert!(!app.session.is_enabled(BLUR), "the engine itself still needs pixels");
    assert!(crate::menus::is_enabled(&app, BLUR));
    let before = past(&app);
    menu(&mut app, BLUR).unwrap();
    let d = prompt(&app).expect("the prompt");
    assert_eq!(d.fields["message"], json!(Kind::Type.message()));
    assert!(filter_dialog(&app).is_none(), "the filter waits for the answer");
    assert_eq!((past(&app), active_content(&app)), (before, "Type"));
    let id = d.id;
    answer(&mut app, id);
    assert_eq!(active_content(&app), "Pixel");
    assert_eq!(past(&app), before + 1);
    let blur = filter_dialog(&app).expect("Gaussian Blur opens on the rasterized layer");
    crate::dialogs::confirm(&mut app, blur).unwrap();
    assert_eq!(past(&app), before + 2, "rasterize, then blur");
    assert_eq!(app.session.journal.last().map(|(id, _)| id.as_str()), Some(BLUR));
}

#[test]
fn cancelling_the_filter_prompt_keeps_the_type_layer() {
    let mut app = app_with(Kind::Type);
    let before = (past(&app), app.session.active().unwrap().doc.clone());
    menu(&mut app, BLUR).unwrap();
    // The menu again while it is up doesn't stack another prompt.
    menu(&mut app, BLUR).unwrap();
    assert_eq!(app.ui.dialogs.len(), 1);
    let id = prompt(&app).unwrap().id;
    crate::dialogs::cancel(&mut app, id).unwrap();
    assert!(app.ui.dialogs.is_empty());
    assert_eq!(past(&app), before.0);
    assert_eq!(*app.session.active().unwrap().doc, *before.1);
    assert_eq!(active_content(&app), "Type");
}

/// Convert to Smart Object keeps the text editable and applies the filter as a smart filter.
#[test]
fn convert_to_smart_object_applies_a_smart_filter() {
    let mut app = app_with(Kind::Type);
    menu(&mut app, BLUR).unwrap();
    let id = prompt(&app).unwrap().id;
    app.ui.dialog_mut(id).unwrap().fields.insert("choice".into(), json!("smartObject"));
    answer(&mut app, id);
    assert_eq!(active_content(&app), "Smart Object");
    let blur = filter_dialog(&app).expect("Gaussian Blur opens on the Smart Object");
    crate::dialogs::confirm(&mut app, blur).unwrap();
    let st = app.session.active().unwrap();
    let LayerContent::Smart(sm) = &st.doc.layer(st.active_layer.unwrap()).unwrap().content else { panic!("a Smart Object") };
    assert_eq!(sm.smart_filters.iter().map(|f| f.command.as_str()).collect::<Vec<_>>(), [BLUR]);
}

/// A filter without a dialog runs straight after rasterizing.
#[test]
fn a_filter_without_a_dialog_runs_after_rasterizing() {
    let mut app = app_with(Kind::Type);
    let before = past(&app);
    menu(&mut app, "filter.blur.blurMore").unwrap();
    let id = prompt(&app).unwrap().id;
    answer(&mut app, id);
    assert!(app.ui.dialogs.is_empty());
    assert_eq!(past(&app), before + 2);
    assert_eq!(active_content(&app), "Pixel");
}

#[test]
fn filters_stay_grey_where_rasterizing_would_not_help() {
    // A locked type layer can't be rasterized either.
    let mut app = app_with(Kind::Type);
    app.run("layer.lockLayers", json!({"pixels": true})).unwrap();
    assert!(!crate::menus::is_enabled(&app, BLUR));
    // Only type layers ask: a shape layer's filters stay grey, as before.
    let app = app_with(Kind::Shape);
    assert!(!crate::menus::is_enabled(&app, BLUR));
    // Non-filter commands a type layer greys are untouched.
    let app = app_with(Kind::Type);
    assert_eq!(crate::menus::is_enabled(&app, "edit.fill"), app.session.is_enabled("edit.fill"));
}

/// The dialogs, and the filter a prompt queued, frame by frame.
fn harness(app: PhotocraftApp) -> egui_kittest::Harness<'static, PhotocraftApp> {
    let h = egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).build_ui_state(
        |ui, app| {
            crate::dialogs::show(app, ui.ctx());
            run_pending(app, ui.ctx());
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
    h
}

/// The prompt's buttons, clicked: Rasterize, then the filter.
#[test]
fn the_filter_prompt_offers_convert_rasterize_and_cancel() {
    use egui_kittest::kittest::Queryable;
    let app = app_with(Kind::Type);
    let mut h = harness(app);
    menu(h.state_mut(), BLUR).unwrap();
    h.run_steps(4);
    h.get_by_label("Convert to Smart Object");
    h.get_by_label("Cancel");
    h.get_by_label("Rasterize").click();
    h.run_steps(4);
    assert_eq!(active_content(h.state()), "Pixel");
    assert!(filter_dialog(h.state()).is_some());
}

#[test]
fn the_convert_button_converts_to_a_smart_object() {
    use egui_kittest::kittest::Queryable;
    let app = app_with(Kind::Type);
    let mut h = harness(app);
    menu(h.state_mut(), BLUR).unwrap();
    h.run_steps(4);
    h.get_by_label("Convert to Smart Object").click();
    h.run_steps(4);
    assert_eq!(active_content(h.state()), "Smart Object");
    assert!(filter_dialog(h.state()).is_some());
}

/// The filters with their own window or mode open on the rasterized layer, or on the Smart Object.
#[test]
fn filters_with_their_own_window_open_after_the_prompt() {
    type IsOpen = fn(&PhotocraftApp) -> bool;
    let open: [(&str, IsOpen); 4] = [
        ("filter.liquify", |a| a.distort.liquify.is_some()),
        ("filter.filterGallery", |a| a.distort.gallery.is_some()),
        ("filter.cameraRaw", |a| a.camera_raw.is_some()),
        ("filter.adaptiveWideAngle", |a| a.wide_angle.is_some()),
    ];
    for (filter, is_open) in open {
        for choice in ["rasterize", "smartObject"] {
            let mut app = app_with(Kind::Type);
            assert!(crate::menus::is_enabled(&app, filter), "{filter}");
            menu(&mut app, filter).unwrap();
            assert!(!is_open(&app), "{filter}: waits for the answer");
            let id = prompt(&app).unwrap_or_else(|| panic!("{filter}: no prompt")).id;
            app.ui.dialog_mut(id).unwrap().fields.insert("choice".into(), json!(choice));
            answer(&mut app, id);
            assert!(!app.ui.status_error, "{filter}/{choice}: {}", app.ui.status);
            assert_eq!(active_content(&app), if choice == "rasterize" { "Pixel" } else { "Smart Object" }, "{filter}");
            assert!(is_open(&app), "{filter}/{choice}: its window opens");
        }
    }
}

/// Last Filter asks only when there is a filter to repeat, then repeats it on the rasterized layer.
#[test]
fn last_filter_on_a_type_layer_needs_a_filter_to_repeat() {
    let mut app = app_with(Kind::Type);
    assert!(!crate::menus::is_enabled(&app, "filter.lastFilter"), "nothing to repeat yet");
    app.run("layer.new.layer", json!({})).unwrap();
    app.run("edit.fill", json!({"color": "#00ff00"})).unwrap();
    app.run("filter.blur.blurMore", json!({})).unwrap();
    let st = app.session.active().unwrap();
    let text = st.doc.layers.iter().find(|l| matches!(l.content, LayerContent::Text(_))).unwrap().id;
    app.run("layer.select", json!({"layer": text.0})).unwrap();
    assert!(crate::menus::is_enabled(&app, "filter.lastFilter"));
    let before = past(&app);
    menu(&mut app, "filter.lastFilter").unwrap();
    let id = prompt(&app).expect("the prompt").id;
    answer(&mut app, id);
    assert!(!app.ui.status_error, "{}", app.ui.status);
    assert_eq!(active_content(&app), "Pixel");
    assert_eq!(past(&app), before + 2, "rasterize, then Blur More again");
    assert_eq!(app.session.journal.iter().filter(|(id, _)| id == "filter.blur.blurMore").count(), 2);
}

/// Every Filter menu item live on a pixel layer is live on a type layer and gets there through the
/// prompt (Flame, Tree, Picture Frame already run on a type layer: they render on a new layer).
#[test]
fn every_filter_live_on_pixels_asks_on_a_type_layer() {
    let ids: Vec<&str> = crate::menu_catalog::CATALOG.iter().filter(|e| e.3.starts_with("filter.")).map(|e| e.3).collect();
    let mut pixels = app_with(Kind::Type);
    pixels.run("layer.rasterize.type", json!({})).unwrap();
    let mut asked = 0;
    for id in ids {
        let mut app = app_with(Kind::Type);
        if !crate::menus::is_enabled(&pixels, id) || app.session.is_enabled(id) {
            continue;
        }
        assert!(crate::menus::is_enabled(&app, id), "{id}");
        menu(&mut app, id).unwrap();
        let d = prompt(&app).unwrap_or_else(|| panic!("{id}: no prompt")).id;
        answer(&mut app, d);
        assert!(!app.ui.status_error, "{id}: {}", app.ui.status);
        assert_eq!(active_content(&app), "Pixel", "{id}");
        asked += 1;
    }
    assert!(asked > 40, "{asked} filters");
}

/// Vanishing Point edits pixels only: its prompt offers Rasterize, not Convert to Smart Object.
#[test]
fn vanishing_point_offers_only_rasterize() {
    use egui_kittest::kittest::Queryable;
    let app = app_with(Kind::Type);
    let mut h = harness(app);
    assert!(crate::menus::is_enabled(h.state(), "filter.vanishingPoint"));
    menu(h.state_mut(), "filter.vanishingPoint").unwrap();
    h.run_steps(4);
    assert!(h.query_by_label("Convert to Smart Object").is_none());
    let id = prompt(h.state()).unwrap().id;
    h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("choice".into(), json!("smartObject"));
    assert!(crate::dialogs::confirm(h.state_mut(), id).is_err(), "no conversion behind its back");
    assert_eq!(active_content(h.state()), "Type");
    menu(h.state_mut(), "filter.vanishingPoint").unwrap();
    h.run_steps(4);
    h.get_by_label("Rasterize").click();
    h.run_steps(4);
    assert_eq!(active_content(h.state()), "Pixel");
    assert!(h.state().distort.vanishing.is_some(), "Vanishing Point opens");
}
