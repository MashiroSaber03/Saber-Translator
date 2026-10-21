//! Layer › Duplicate Layer… dialog (#3348): the copy's name and its destination document, as in
//! Photoshop. Only the menu, palette and Layers panel menu open it; ⌘J, Alt-drag and dragging onto
//! the New Layer button still duplicate at once.
use crate::{PhotocraftApp, state::DialogKind};
use serde_json::{Map, Value, json};
const MARK: &str = "__duplicateLayer";

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

pub fn open(app: &mut PhotocraftApp) -> Result<Value, String> {
    let index = app.session.active_index().ok_or("no document")?;
    let st = app.session.active().ok_or("no document")?;
    let selected = st.selected_layers();
    let layer = st.active_layer.and_then(|id| st.doc.layer(id)).ok_or("no active layer")?;
    let mut fields = Map::new();
    fields.insert(MARK.into(), json!(true));
    fields.insert("__command".into(), json!("layer.duplicate"));
    fields.insert("destination".into(), json!(index));
    // Photoshop's document-`Name` starts empty; the engine defaults it when empty.
    fields.insert("documentName".into(), json!(""));
    fields.insert("document".into(), json!(st.doc.id.0));
    fields.insert("selected".into(), json!(selected.iter().map(|id| id.0).collect::<Vec<_>>()));
    // Several selected layers keep their names, as in Photoshop's Duplicate Layers.
    if selected.len() > 1 {
        fields.insert("__label".into(), json!("Duplicate Layers…"));
    } else if layer.is_group() {
        fields.insert("__label".into(), json!("Duplicate Group…"));
        fields.insert("name".into(), json!(st.doc.copy_name(&layer.name)));
    } else {
        fields.insert("__label".into(), json!("Duplicate Layer…"));
        fields.insert("name".into(), json!(st.doc.copy_name(&layer.name)));
    }
    let id = app.ui.open_dialog(DialogKind::Command, fields);
    Ok(json!({"dialog": id}))
}

pub fn body(app: &PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let documents: Vec<(String, String)> = app.session.documents().iter().enumerate().map(|(i, d)| (i.to_string(), d.doc.name.clone())).collect();
    let mut options: Vec<(String, &str)> = documents.iter().map(|(k, n)| (k.clone(), n.as_str())).collect();
    options.push(("new".to_string(), tl!("New")));
    let mut destination = match fields.get("destination") {
        Some(Value::String(s)) if s == "new" => "new".to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    };
    egui::Grid::new("duplicate-layer").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        if let Some(name) = fields.get("name").and_then(Value::as_str) {
            let mut name = name.to_string();
            ui.label(tl!("Name:"));
            ui.add(egui::TextEdit::singleline(&mut name).desired_width(260.0));
            fields.insert("name".into(), json!(name));
            ui.end_row();
        }
        if app.services.host_save.is_some() {
            return;
        }
        ui.label(egui::RichText::new(tl!("Destination")).font(crate::theme::semibold(13.0)));
        ui.end_row();
        ui.label(tl!("Document"));
        crate::widgets::dropdown(ui, "duplicate-layer-document", &mut destination, &options, 260.0);
        ui.end_row();
        // Photoshop's document-`Name` is only there when the destination is a new document.
        if destination == "new" {
            let mut doc_name = fields.get("documentName").and_then(Value::as_str).unwrap_or("").to_string();
            ui.label(tl!("Name:"));
            ui.add(egui::TextEdit::singleline(&mut doc_name).desired_width(260.0));
            fields.insert("documentName".into(), json!(doc_name));
            ui.end_row();
        }
    });
    if destination == "new" {
        fields.insert("destination".into(), json!("new"));
    } else if let Ok(i) = destination.parse::<u64>() {
        fields.insert("destination".into(), json!(i));
    }
}

pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let index = app.session.active_index().ok_or("The document is closed")?;
    let st = app.session.active().ok_or("The document is closed")?;
    if fields.get("document").and_then(Value::as_u64) != Some(st.doc.id.0)
        || fields.get("selected") != Some(&json!(st.selected_layers().iter().map(|id| id.0).collect::<Vec<_>>()))
    {
        return Err("The document or selected layers changed. Choose Duplicate Layer again.".into());
    }
    let mut params = Map::new();
    if let Some(name) = fields.get("name").and_then(Value::as_str) {
        params.insert("name".into(), json!(name));
    }
    match fields.get("destination") {
        Some(Value::String(s)) if s == "new" => {
            if let Some(doc_name) = fields.get("documentName").and_then(Value::as_str).filter(|n| !n.is_empty()) {
                params.insert("documentName".into(), json!(doc_name));
            }
            params.insert("document".into(), json!("new"));
            app.run("layer.duplicate", Value::Object(params))
        }
        Some(Value::Number(n)) => {
            let destination =
                n.as_u64().and_then(|i| usize::try_from(i).ok()).filter(|i| *i < app.session.documents().len()).ok_or("no destination document")?;
            if destination == index {
                app.run("layer.duplicate", Value::Object(params))
            } else {
                params.insert("document".into(), json!(destination));
                app.run("layer.copyToDocument", Value::Object(params))
            }
        }
        _ => Err("no destination document".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make_app() -> PhotocraftApp {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width":8,"height":8,"name":"Poster"})).unwrap();
        s.execute("layer.new.layer", json!({"name":"Ink"})).unwrap();
        PhotocraftApp::new(s, crate::Services::default())
    }
    fn invoke(app: &mut PhotocraftApp) -> u64 {
        let ctx = egui::Context::default();
        crate::menus::invoke(app, &ctx, "layer.duplicate", json!({})).unwrap()["dialog"].as_u64().unwrap()
    }
    fn names(app: &PhotocraftApp, i: usize) -> Vec<String> {
        app.session.documents()[i].doc.walk().into_iter().map(|(_, _, l)| l.name.clone()).collect()
    }

    #[test]
    fn the_menu_opens_the_dialog_and_ok_duplicates_under_the_typed_name() {
        let mut app = make_app();
        let dialog = invoke(&mut app);
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 2, "nothing is copied until OK");
        assert_eq!(app.ui.dialogs[0].fields["name"], "Ink copy");
        app.ui.dialog_mut(dialog).unwrap().fields.insert("name".into(), json!("Shadow"));
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert!(names(&app, 0).contains(&"Shadow".to_string()));
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 3);
    }

    #[test]
    fn embedded_copies_stay_in_the_current_page() {
        let mut app = make_app();
        app.services.host_save = Some(Box::new(|| {}));
        let dialog = invoke(&mut app);
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 3);
        assert!(app.run("layer.duplicate", json!({"document":"new"})).is_err());
        assert!(app.run("layer.copyToDocument", json!({"document":0})).is_err());
        let dialog = invoke(&mut app);
        app.ui.dialog_mut(dialog).unwrap().fields.insert("destination".into(), json!("new"));
        assert!(crate::dialogs::confirm(&mut app, dialog).is_err());
        assert_eq!(app.session.documents().len(), 1);
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 3);
    }

    #[test]
    fn another_destination_document_receives_the_copy() {
        let mut app = make_app();
        app.session.execute("file.new", json!({"width":8,"height":8,"name":"Card"})).unwrap();
        app.session.set_active(0);
        let dialog = invoke(&mut app);
        app.ui.dialog_mut(dialog).unwrap().fields.insert("destination".into(), json!(1));
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert_eq!(app.session.documents()[0].doc.layer_count(), 2);
        assert!(names(&app, 1).contains(&"Ink copy".to_string()));
    }

    #[test]
    fn a_new_destination_makes_a_named_document_holding_just_the_copy() {
        let mut app = make_app();
        let dialog = invoke(&mut app);
        let fields = &mut app.ui.dialog_mut(dialog).unwrap().fields;
        fields.insert("destination".into(), json!("new"));
        fields.insert("documentName".into(), json!("Card"));
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert_eq!(app.session.documents().len(), 2);
        assert_eq!(app.session.active_index(), Some(1));
        assert_eq!(app.session.documents()[1].doc.name, "Card");
        assert_eq!(app.session.documents()[0].doc.layer_count(), 2, "the source is untouched");
        assert!(names(&app, 1).contains(&"Ink copy".to_string()));
    }

    #[test]
    fn a_new_destination_without_a_name_defaults_it_and_refuses_a_dead_index() {
        let mut app = make_app();
        let dialog = invoke(&mut app);
        app.ui.dialog_mut(dialog).unwrap().fields.insert("destination".into(), json!("new"));
        crate::dialogs::confirm(&mut app, dialog).unwrap();
        assert_eq!(app.session.documents()[1].doc.name, "Poster copy");
        let dialog = invoke(&mut app);
        app.ui.dialog_mut(dialog).unwrap().fields.insert("destination".into(), json!(7));
        assert!(crate::dialogs::confirm(&mut app, dialog).is_err());
        assert_eq!(app.session.documents().len(), 2);
    }

    #[test]
    fn a_single_group_gets_the_group_title() {
        let mut app = make_app();
        app.session.execute("layer.new.group", json!({"name": "Pack"})).unwrap();
        let _dialog = invoke(&mut app);
        assert_eq!(app.ui.dialogs[0].fields["__label"], "Duplicate Group…");
    }

    #[test]
    fn a_changed_document_refuses_and_direct_runs_stay_dialog_free() {
        let mut app = make_app();
        let dialog = invoke(&mut app);
        app.session.execute("file.new", json!({"width":8,"height":8})).unwrap();
        assert!(crate::dialogs::confirm(&mut app, dialog).is_err());
        app.session.set_active(0);
        app.run("layer.duplicate", json!({})).unwrap();
        app.run("layer.new.layerViaCopy", json!({})).unwrap();
        assert!(app.ui.dialogs.is_empty());
        assert_eq!(app.session.active().unwrap().doc.layer_count(), 4);
    }
}
