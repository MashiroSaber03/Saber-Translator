//! Confirmation for a Layers-panel trash-button click; other deletion routes stay immediate.
use crate::{PhotocraftApp, state::DialogKind};
use serde_json::{Map, Value, json};
const MARK: &str = "__deleteLayer";
const PREF: &str = "layer.delete.confirmation";

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

pub fn intercept(app: &mut PhotocraftApp, command: &str, params: &Value) -> Option<Result<Value, String>> {
    if app.automation_input
        || command != "layer.delete"
        || params.get("__trash").and_then(Value::as_bool) != Some(true)
        || app.session.prefs().dialogs.get(PREF).and_then(|v| v.get("skip")).and_then(Value::as_bool) == Some(true)
    {
        return None;
    }
    let state = app.session.active()?;
    let selected = state.selected_layers();
    let ids = if let Some(id) = params.get("layer").and_then(Value::as_u64) { vec![photocraft_doc::LayerId(id)] } else { selected.clone() };
    if ids.is_empty() {
        return None;
    }
    let names = ids.iter().filter_map(|id| state.doc.layer(*id).map(|layer| layer.name.as_str())).take(4).collect::<Vec<_>>().join(", ");
    let message = if ids.len() == 1 {
        crate::i18n::fmt(tl!("Delete layer ‘{name}’?"), &[("name", &names)])
    } else {
        crate::i18n::fmt(tl!("Delete {count} layers? {names}"), &[("count", &ids.len().to_string()), ("names", &names)])
    };
    let fields = json!({MARK: true, "__command": command, "__label": "Delete Layer", "message": message,
        "document": state.doc.id.0, "selected": selected.iter().map(|id| id.0).collect::<Vec<_>>(), "params": params, "skip": false});
    if app.ui.dialogs.iter().any(|d| owns(&d.fields)) {
        return Some(Ok(json!({"confirmationPending": true})));
    }
    let id = app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default());
    Some(Ok(json!({"dialog": id, "confirmationPending": true})))
}

pub fn body(ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    ui.label(fields.get("message").and_then(Value::as_str).unwrap_or(tl!("Delete layer?")));
    ui.add_space(8.0);
    let mut skip = fields.get("skip").and_then(Value::as_bool).unwrap_or(false);
    ui.checkbox(&mut skip, tl!("Don't show again"));
    fields.insert("skip".into(), json!(skip));
}

pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let state = app.session.active().ok_or(tl!("The document is closed"))?;
    if fields.get("document").and_then(Value::as_u64) != Some(state.doc.id.0)
        || fields.get("selected") != Some(&json!(state.selected_layers().iter().map(|id| id.0).collect::<Vec<_>>()))
    {
        return Err(tl!("The document or selected layers changed. Please request deletion again.").into());
    }
    let command = fields.get("__command").and_then(Value::as_str).ok_or(tl!("Missing delete command"))?;
    if command != "layer.delete" {
        return Err(tl!("Invalid delete command").into());
    }
    let result = app.run_command(command, fields.get("params").cloned().unwrap_or(Value::Null))?;
    if fields.get("skip").and_then(Value::as_bool) == Some(true) {
        app.session.edit_prefs(|prefs| {
            prefs.dialogs.insert(PREF.into(), json!({"skip": true}));
        });
        if let Err(error) = crate::prefs_ui::save_preferences(app) {
            app.ui.status = crate::i18n::fmt(tl!("Layer deleted, but confirmation preference could not be saved: {error}"), &[("error", &error)]);
            app.ui.status_error = true;
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::kittest::Queryable;
    fn make_app() -> PhotocraftApp {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width":8,"height":8})).unwrap();
        s.execute("layer.new.layer", json!({"name":"Ink"})).unwrap();
        PhotocraftApp::new(s, crate::Services::default())
    }
    #[test]
    fn cancel_keeps_layer_and_confirmation_preference() {
        let mut app = make_app();
        let id = app.session.active().unwrap().active_layer.unwrap();
        let result = app.run("layer.delete", json!({"__trash":true})).unwrap();
        let dialog = result["dialog"].as_u64().unwrap();
        assert!(owns(&app.ui.dialogs[0].fields));
        assert!(app.ui.dialogs[0].fields["message"].as_str().unwrap().contains("Ink"));
        app.ui.dialog_mut(dialog).unwrap().fields.insert("skip".into(), json!(true));
        app.ui.close_dialog(dialog);
        assert!(app.session.active().unwrap().doc.layer(id).is_some());
        assert!(!app.session.prefs().dialogs.contains_key(PREF));
    }
    #[test]
    fn confirmation_deletes_once_and_remembers_across_restart() {
        use std::{cell::RefCell, rc::Rc};
        let mut app = make_app();
        let saved = Rc::new(RefCell::new(String::new()));
        let copy = saved.clone();
        app.services.save_prefs = Some(Box::new(move |text| {
            *copy.borrow_mut() = text.to_string();
            Ok(())
        }));
        let target = app.session.active().unwrap().active_layer.unwrap();
        let before = app.session.active().unwrap().history.past_len();
        let dialog = app.run("layer.delete", json!({"__trash":true})).unwrap()["dialog"].as_u64().unwrap();
        app.ui.dialog_mut(dialog).unwrap().fields.insert("skip".into(), json!(true));
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert!(app.session.active().unwrap().doc.layer(target).is_none());
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
        assert!(app.ui.dialogs.is_empty());
        app.session.undo();
        assert!(app.session.active().unwrap().doc.layer(target).is_some());
        let text = saved.borrow().clone();
        assert!(text.contains(PREF));
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width":8,"height":8})).unwrap();
        session.execute("layer.new.layer", json!({"name":"Fresh"})).unwrap();
        let mut fresh = PhotocraftApp::new(session, crate::Services { load_prefs: Some(Box::new(move || Some(text.clone()))), ..Default::default() });
        fresh.run("layer.delete", json!({"__trash":true})).unwrap();
        assert!(fresh.ui.dialogs.is_empty());
    }
    #[test]
    fn changed_document_refuses_confirmation_and_clear_does_not_prompt() {
        let mut app = make_app();
        let dialog = app.run("layer.delete", json!({"__trash":true})).unwrap()["dialog"].as_u64().unwrap();
        app.session.execute("file.new", json!({"width":8,"height":8})).unwrap();
        assert!(crate::dialogs::confirm(&mut app, dialog).is_err());
        assert_eq!(app.session.documents()[0].doc.layer_count(), 2);
        app.run("edit.clear", json!({})).unwrap();
        assert!(app.ui.dialogs.is_empty());
    }

    #[test]
    fn only_trash_click_prompts_and_default_delete_binding_is_unchanged() {
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(1200.0, 800.0)).with_max_steps(64).build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            make_app()
        });
        h.run_steps(4);
        let id = h.state().session.active().unwrap().active_layer.unwrap();
        h.get_by_label("Delete layer").click();
        h.run_steps(4);
        assert!(h.state().ui.dialogs.iter().any(|d| owns(&d.fields)));
        assert!(h.state().session.active().unwrap().doc.layer(id).is_some());
        let dialog = h.state().ui.dialogs[0].id;
        h.state_mut().ui.close_dialog(dialog);
        h.state_mut().run("layer.delete", json!({})).unwrap();
        assert!(h.state().ui.dialogs.is_empty());
        assert!(h.state().session.active().unwrap().doc.layer(id).is_none());
        let clear = photocraft_engine::commands::find("edit.clear").unwrap();
        let delete = photocraft_engine::commands::find("layer.delete").unwrap();
        assert_eq!(clear.shortcut, Some("Delete"));
        assert_eq!(delete.shortcut, None);
    }

    #[test]
    fn confirmation_strings_are_translated_in_every_supported_catalog() {
        for language in crate::i18n::LANGUAGES.iter().filter(|l| l.code != "en") {
            let lang = crate::i18n::Lang::from_code(language.code).unwrap();
            for key in [
                "Delete layer ‘{name}’?",
                "Delete {count} layers? {names}",
                "Delete layer?",
                "Don't show again",
                "The document is closed",
                "The document or selected layers changed. Please request deletion again.",
                "Missing delete command",
                "Invalid delete command",
                "Layer deleted, but confirmation preference could not be saved: {error}",
            ] {
                assert_ne!(crate::i18n::tr(lang, key), key, "{}: {key}", language.code);
            }
        }
    }
}
