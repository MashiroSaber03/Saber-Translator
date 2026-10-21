use super::*;
use crate::{canvas, dialogs, layer_style, theme::ThemeKind};
use egui::{Key, vec2};
use egui_kittest::{Harness, kittest::Queryable};
use std::sync::Arc;

fn app(depth: u32) -> PhotocraftApp {
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    app.run("prefs.set", json!({"path":"interface.language","value":"en"})).unwrap();
    app.run("file.new", json!({"width":32,"height":24,"depth":depth,"background":"#406080"})).unwrap();
    app
}
fn fields(app: &PhotocraftApp, id: u64) -> Map<String, Value> {
    app.ui.dialogs.iter().find(|d| d.id == id).unwrap().fields.clone()
}
fn set_color(app: &mut PhotocraftApp, id: u64, color: &str) {
    app.ui.dialog_mut(id).unwrap().fields.insert("color".into(), json!(color));
}
fn selected(app: &PhotocraftApp, parent: u64) -> String {
    fields(app, parent)["selected"].as_str().unwrap().into()
}

#[test]
fn every_exposed_color_field_uses_the_shared_picker_and_only_updates_its_target() {
    for (kind, _) in layer_style::KINDS {
        for (field, _, p) in spec(kind) {
            if !matches!(p, P::Color) {
                continue;
            }
            let mut app = app(8);
            let parent = layer_style::open(&mut app, Some(kind)).unwrap();
            let effect = selected(&app, parent);
            let original = fields(&app, parent);
            let tools = app.session.tools.clone();
            let doc = app.session.active().unwrap().doc.clone();
            let history = app.session.active().unwrap().history.past_len();
            let picker = open(&mut app, parent, &effect, field).unwrap();
            assert!(color_picker_ui::owns(&fields(&app, picker)));
            assert_eq!(fields(&app, picker)["__label"], "Color Picker (Layer Style Color)");
            set_color(&mut app, picker, "#23a567");
            let shown = preview_fields(&app, parent, &original);
            assert_eq!(super::super::entry(&shown, &effect).unwrap()["params"][field], "#23a567");
            assert_eq!(fields(&app, parent), original);
            dialogs::confirm(&mut app, picker).unwrap();
            let mut expected = original;
            set_param(&mut expected, &effect, field, json!("#23a567"));
            assert_eq!(fields(&app, parent), expected, "{kind}.{field}");
            assert!(Arc::ptr_eq(&app.session.active().unwrap().doc, &doc));
            assert_eq!(app.session.active().unwrap().history.past_len(), history);
            assert_eq!(app.session.tools.foreground, tools.foreground);
            assert_eq!(app.session.tools.background, tools.background);
        }
    }
}

#[test]
fn preview_cancel_ok_parent_cancel_and_apply_are_transactional_at_every_depth() {
    for depth in [8, 16, 32] {
        let mut app = app(depth);
        let parent = layer_style::open(&mut app, Some("colorOverlay")).unwrap();
        let effect = selected(&app, parent);
        layer_style::add_instance(&mut app.ui.dialog_mut(parent).unwrap().fields, "stroke");
        let original = fields(&app, parent);
        let st = app.session.active().unwrap();
        let (doc, history, revision, dirty) = (st.doc.clone(), st.history.past_len(), st.revision, st.is_dirty());
        let picker = open(&mut app, parent, &effect, "color").unwrap();
        set_color(&mut app, picker, "#00ff00");
        let (shown, key) = canvas::display_doc(&mut app, 0);
        assert!(!Arc::ptr_eq(&shown, &doc));
        let again = canvas::display_doc(&mut app, 0);
        assert_eq!(key, again.1);
        assert!(Arc::ptr_eq(&shown, &again.0), "unchanged preview is cached");
        let pixel = photocraft_compose::flatten(&shown).get(3, 3);
        assert!(pixel[1] > pixel[0] && pixel[1] > pixel[2], "live preview is green at {depth} bits");
        let st = app.session.active().unwrap();
        assert_eq!((st.history.past_len(), st.revision, st.is_dirty()), (history, revision, dirty));
        assert!(Arc::ptr_eq(&st.doc, &doc));
        dialogs::cancel(&mut app, picker).unwrap();
        assert_eq!(fields(&app, parent), original);
        assert_ne!(canvas::display_doc(&mut app, 0).1, key);
        let picker = open(&mut app, parent, &effect, "color").unwrap();
        set_color(&mut app, picker, "#00ff00");
        dialogs::confirm(&mut app, picker).unwrap();
        assert_eq!(app.session.active().unwrap().history.past_len(), history);
        dialogs::cancel(&mut app, parent).unwrap();
        assert!(Arc::ptr_eq(&canvas::display_doc(&mut app, 0).0, &doc));
        let parent = layer_style::open(&mut app, Some("colorOverlay")).unwrap();
        let effect = selected(&app, parent);
        let picker = open(&mut app, parent, &effect, "color").unwrap();
        set_color(&mut app, picker, "#00ff00");
        dialogs::confirm(&mut app, picker).unwrap();
        dialogs::confirm(&mut app, parent).unwrap();
        let applied = app.session.active().unwrap().doc.clone();
        let pixel = photocraft_compose::flatten(&applied).get(3, 3);
        assert!(pixel[1] > pixel[0]);
        app.run("edit.undo", json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layer(doc.layers[0].id).unwrap().effects.items.len(), 0);
        app.run("edit.redo", json!({})).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layer(doc.layers[0].id).unwrap().effects, applied.layer(doc.layers[0].id).unwrap().effects);
    }
}

#[test]
fn unchanged_ok_preserves_missing_parameters_and_imported_effect_snapshot() {
    let mut app = app(8);
    app.run("layer.layerStyle.stroke", json!({"size":7,"color":"#123456","add":true})).unwrap();
    app.run("layer.layerStyle.stroke", json!({"size":9,"color":"#abcdef","add":true})).unwrap();
    let parent = layer_style::open(&mut app, Some("stroke")).unwrap();
    let original = fields(&app, parent);
    for effect in ["fx1", "fx2"] {
        let picker = open(&mut app, parent, effect, "color").unwrap();
        dialogs::confirm(&mut app, picker).unwrap();
        assert_eq!(fields(&app, parent), original);
    }
    let picker = open(&mut app, parent, "fx2", "color").unwrap();
    set_color(&mut app, picker, "#223344");
    dialogs::confirm(&mut app, picker).unwrap();
    let current = fields(&app, parent);
    assert_eq!(super::super::entry(&current, "fx1"), super::super::entry(&original, "fx1"));
    assert_eq!(super::super::entry(&current, "fx2").unwrap()["fx"], super::super::entry(&original, "fx2").unwrap()["fx"]);
    dialogs::cancel(&mut app, parent).unwrap();
    let parent = layer_style::open(&mut app, Some("gradientOverlay")).unwrap();
    let effect = selected(&app, parent);
    app.ui.dialog_mut(parent).unwrap().fields.get_mut("effects").unwrap().as_array_mut().unwrap()[0]["params"].as_object_mut().unwrap().remove("from");
    let original = fields(&app, parent);
    let picker = open(&mut app, parent, &effect, "from").unwrap();
    dialogs::confirm(&mut app, picker).unwrap();
    assert_eq!(fields(&app, parent), original);
}

#[test]
fn stale_targets_and_invalid_colors_never_apply_or_leave_preview() {
    for change in ["document", "parent", "effect", "kind", "color", "source", "malformed", "layer", "replaced"] {
        let mut app = app(8);
        let parent = layer_style::open(&mut app, Some("colorOverlay")).unwrap();
        let effect = selected(&app, parent);
        let picker = open(&mut app, parent, &effect, "color").unwrap();
        set_color(&mut app, picker, "#00ff00");
        match change {
            "document" => {
                app.run("file.new", json!({"width":3,"height":3})).unwrap();
            }
            "parent" => {
                app.ui.close_dialog(parent);
            }
            "effect" => {
                app.ui.dialog_mut(parent).unwrap().fields.insert("effects".into(), json!([]));
            }
            "kind" => {
                app.ui.dialog_mut(parent).unwrap().fields["effects"][0]["kind"] = json!("stroke");
            }
            "color" => {
                set_param(&mut app.ui.dialog_mut(parent).unwrap().fields, &effect, "color", json!("#ff00ff"));
            }
            "replaced" => {
                app.ui.dialog_mut(parent).unwrap().fields["effects"][0] =
                    json!({"id":effect,"kind":"colorOverlay","on":true,"params":{"color":"#ff0000","opacity":12}});
            }
            "source" => {
                app.ui.dialog_mut(parent).unwrap().fields["effects"][0]["fx"] = json!({"replacement":true});
            }
            "malformed" => {
                app.ui.dialog_mut(picker).unwrap().fields.insert(TARGET.into(), json!(null));
            }
            _ => {
                app.ui.dialog_mut(parent).unwrap().fields.insert("layer".into(), json!(u64::MAX));
            }
        }
        let current_parent = app.ui.dialogs.iter().find(|d| d.id == parent).cloned();
        assert!(dialogs::confirm(&mut app, picker).is_err(), "{change}");
        assert_eq!(app.ui.dialogs.iter().find(|d| d.id == parent).cloned(), current_parent);
    }
    let mut app = app(8);
    let parent = layer_style::open(&mut app, Some("colorOverlay")).unwrap();
    let effect = selected(&app, parent);
    let original = fields(&app, parent);
    let picker = open(&mut app, parent, &effect, "color").unwrap();
    set_color(&mut app, picker, "bad");
    assert!(dialogs::confirm(&mut app, picker).is_err());
    assert_eq!(fields(&app, parent), original);
    let picker = open(&mut app, parent, &effect, "color").unwrap();
    app.run("file.new", json!({"width":3,"height":3})).unwrap();
    prune(&mut app);
    assert!(!app.ui.dialogs.iter().any(|d| d.id == picker));
}

#[test]
fn preview_off_and_parent_confirmation_and_cancellation_keep_their_contract() {
    let mut app = app(8);
    let parent = layer_style::open(&mut app, Some("colorOverlay")).unwrap();
    app.ui.dialog_mut(parent).unwrap().fields.insert("preview".into(), json!(false));
    let effect = selected(&app, parent);
    let original = fields(&app, parent);
    let doc = app.session.active().unwrap().doc.clone();
    let picker = open(&mut app, parent, &effect, "color").unwrap();
    set_color(&mut app, picker, "#00ff00");
    assert!(Arc::ptr_eq(&canvas::display_doc(&mut app, 0).0, &doc));
    assert!(dialogs::confirm(&mut app, parent).is_err());
    assert_eq!(fields(&app, parent), original);
    assert_eq!(app.ui.dialogs.len(), 2);
    dialogs::cancel(&mut app, parent).unwrap();
    assert!(app.ui.dialogs.is_empty());
}

fn harness() -> Harness<'static, PhotocraftApp> {
    let mut app = app(8);
    layer_style::open(&mut app, Some("colorOverlay")).unwrap();
    let mut h = Harness::builder().with_size(vec2(1500.0, 1000.0)).build_ui_state(
        |ui, app| {
            if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                dialogs::show(app, ui.ctx());
            }
        },
        app,
    );
    PhotocraftApp::setup_context(&h.ctx, ThemeKind::ProMedium);
    h.run_steps(4);
    h
}

#[test]
fn real_swatch_click_opens_shared_dialog_and_enter_escape_only_affect_the_child() {
    for action in [Key::Enter, Key::Escape] {
        let mut h = harness();
        let parent = h.state().ui.dialogs[0].id;
        let before = fields(h.state(), parent);
        h.get_by_role(egui::accesskit::Role::ColorWell).click();
        h.run();
        assert_eq!(h.state().ui.dialogs.len(), 2);
        let picker = h.state().ui.dialogs[1].id;
        set_color(h.state_mut(), picker, "#00ff00");
        h.key_press(action);
        h.run();
        assert_eq!(h.state().ui.dialogs.len(), 1);
        assert_eq!(h.state().ui.dialogs[0].id, parent);
        if action == Key::Escape {
            assert_eq!(fields(h.state(), parent), before);
        } else {
            assert_eq!(super::super::entry(&fields(h.state(), parent), "fx1").unwrap()["params"]["color"], "#00ff00");
        }
    }
}

#[test]
fn eyedropper_edits_only_the_picker_and_invalid_samples_leave_it_unchanged() {
    let mut app = app(8);
    let parent = layer_style::open(&mut app, Some("colorOverlay")).unwrap();
    let effect = selected(&app, parent);
    let original = fields(&app, parent);
    let doc = app.session.active().unwrap().doc.clone();
    let tools = app.session.tools.clone();
    let picker = open(&mut app, parent, &effect, "color").unwrap();
    color_picker_ui::sample_at(&mut app, 4.0, 4.0);
    assert_eq!(fields(&app, picker)["color"], "#406080");
    color_picker_ui::sample_at(&mut app, -100.0, -100.0);
    assert_eq!(fields(&app, picker)["color"], "#406080");
    assert_eq!(fields(&app, parent), original);
    assert!(Arc::ptr_eq(&app.session.active().unwrap().doc, &doc));
    assert_eq!(app.session.tools.foreground, tools.foreground);
    dialogs::cancel(&mut app, picker).unwrap();
}
