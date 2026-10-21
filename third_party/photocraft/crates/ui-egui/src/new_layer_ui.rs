//! Layer › New › Layer… (⇧⌘N): Photoshop's New Layer dialog. Name, Use Previous Layer to Create
//! Clipping Mask, Color, Mode, Opacity and Fill with neutral color; OK runs `layer.new.layer` with
//! them. The Layers panel's New Layer button makes a layer without asking, and ⌥-click opens the
//! dialog, as in Photoshop.

use photocraft_doc::{BlendMode, LabelColor};
use photocraft_engine::new_layer::neutral_color;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;

const MARK: &str = "__newLayer";
pub const COMMAND: &str = "layer.new.layer";

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

/// Open the dialog for the active document, with the name the layer would get.
pub fn open(app: &mut PhotocraftApp) -> Result<u64, String> {
    let name = app.session.active().ok_or("no document open")?.doc.next_layer_name("Layer");
    let fields = json!({
        MARK: true, "__command": COMMAND, "__label": "New Layer",
        "name": name, "clipToPrevious": false, "color": "none", "mode": "Normal", "opacity": 100.0, "fillNeutral": false,
    });
    Ok(app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default()))
}

fn mode(f: &Map<String, Value>) -> BlendMode {
    f.get("mode")
        .and_then(Value::as_str)
        .and_then(photocraft_engine::commands::blend_from_str)
        .filter(|m| *m != BlendMode::PassThrough)
        .unwrap_or(BlendMode::Normal)
}

fn flag(f: &Map<String, Value>, k: &str) -> bool {
    f.get(k).and_then(Value::as_bool).unwrap_or(false)
}

/// The `layer.new.layer` params the dialog stands for (opacity is a percentage in the dialog).
pub fn params(f: &Map<String, Value>) -> Value {
    let mode = mode(f);
    let opacity = f.get("opacity").and_then(Value::as_f64).filter(|o| o.is_finite()).unwrap_or(100.0).clamp(0.0, 100.0) / 100.0;
    let mut p = json!({
        "clipToPrevious": flag(f, "clipToPrevious"),
        "color": f.get("color").and_then(Value::as_str).and_then(LabelColor::from_id).unwrap_or_default().id(),
        "mode": mode.label(),
        "opacity": opacity,
        // Only modes with a neutral color can fill with one (the checkbox is off for the others).
        "fillNeutral": flag(f, "fillNeutral") && neutral_color(mode).is_some(),
    });
    // An empty name gets the next "Layer N", as the Layers panel button does.
    if let Some(name) = f.get("name").and_then(Value::as_str).filter(|n| !n.trim().is_empty()) {
        p["name"] = json!(name);
    }
    p
}

/// The neutral-fill checkbox's label for `mode`, or the note that it has none.
pub fn neutral_label(mode: BlendMode) -> String {
    let name = tl!(mode.label());
    match neutral_color(mode) {
        Some((_, "white")) => crate::i18n::fmt(tl!("Fill with {mode}-neutral color (white)"), &[("mode", name)]),
        Some((_, "black")) => crate::i18n::fmt(tl!("Fill with {mode}-neutral color (black)"), &[("mode", name)]),
        Some(_) => crate::i18n::fmt(tl!("Fill with {mode}-neutral color (50% gray)"), &[("mode", name)]),
        None => crate::i18n::fmt(tl!("(No neutral color exists for {mode} mode.)"), &[("mode", name)]),
    }
}

pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let label_width = 64.0;
    let row_label = |ui: &mut egui::Ui, text: &str| {
        ui.allocate_ui_with_layout(egui::vec2(label_width, 22.0), egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(text));
    };
    ui.horizontal(|ui| {
        row_label(ui, tl!("Name:"));
        let mut name = f.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        let field = ui.add(egui::TextEdit::singleline(&mut name).desired_width(260.0));
        if field.changed() {
            f.insert("name".into(), json!(name));
        }
        // Photoshop opens on the name, selected, so typing replaces it.
        if !ui.is_sizing_pass() && !flag(f, "__nameFocused") {
            field.request_focus();
            crate::field_tab::select_all(ui.ctx(), field.id, &name);
            f.insert("__nameFocused".into(), json!(true));
        }
    });
    ui.horizontal(|ui| {
        ui.add_space(label_width + ui.spacing().item_spacing.x);
        let mut clip = flag(f, "clipToPrevious");
        if crate::widgets::checkbox(ui, &mut clip, tl!("Use Previous Layer to Create Clipping Mask")).changed() {
            f.insert("clipToPrevious".into(), json!(clip));
        }
    });
    ui.horizontal(|ui| {
        row_label(ui, tl!("Color:"));
        let mut color = f.get("color").and_then(Value::as_str).and_then(LabelColor::from_id).unwrap_or_default();
        let options: Vec<(LabelColor, &str)> = LabelColor::ALL.into_iter().map(|c| (c, crate::layer_menu_ui::color_name(c))).collect();
        if crate::widgets::dropdown(ui, "new-layer-color", &mut color, &options, 140.0) {
            f.insert("color".into(), json!(color.id()));
        }
    });
    let mut m = mode(f);
    ui.horizontal(|ui| {
        row_label(ui, tl!("Mode:"));
        let options: Vec<(BlendMode, &str)> = BlendMode::LAYER_MODES.into_iter().map(|m| (m, m.label())).collect();
        if crate::widgets::dropdown(ui, "new-layer-mode", &mut m, &options, 140.0) {
            f.insert("mode".into(), json!(m.label()));
        }
        ui.add_space(8.0);
        ui.label(tl!("Opacity:"));
        let mut o = f.get("opacity").and_then(Value::as_f64).unwrap_or(100.0) as f32;
        if crate::widgets::value_field(ui, &mut o, 0.0..=100.0, "%", 64.0).changed() {
            f.insert("opacity".into(), json!(o.clamp(0.0, 100.0)));
        }
    });
    ui.horizontal(|ui| {
        ui.add_space(label_width + ui.spacing().item_spacing.x);
        let available = neutral_color(m).is_some();
        let mut fill = available && flag(f, "fillNeutral");
        let changed = ui.add_enabled_ui(available, |ui| crate::widgets::checkbox(ui, &mut fill, &neutral_label(m)).changed()).inner;
        if changed {
            f.insert("fillNeutral".into(), json!(fill));
        }
    });
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    app.run(COMMAND, params(f))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::kittest::{NodeT, Queryable};

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 16, "height": 16})).unwrap();
        app
    }

    fn top(app: &PhotocraftApp) -> &photocraft_doc::Layer {
        app.session.active().unwrap().doc.layers.last().unwrap()
    }

    /// The menu (and its shortcut) opens the dialog; OK makes the layer it describes in one step.
    #[test]
    fn menu_opens_the_dialog_and_ok_makes_the_layer() {
        let mut app = app();
        let ctx = egui::Context::default();
        let layers = app.session.active().unwrap().doc.layers.len();
        let r = crate::menus::invoke(&mut app, &ctx, COMMAND, json!({})).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        assert_eq!(app.session.active().unwrap().doc.layers.len(), layers, "nothing made until OK");
        let d = app.ui.dialogs.iter_mut().find(|d| d.id == id).unwrap();
        assert_eq!(d.fields["name"], "Layer 1");
        for (k, v) in
            [("name", json!("Dodge & Burn")), ("mode", json!("Overlay")), ("opacity", json!(60.0)), ("color", json!("violet")), ("fillNeutral", json!(true))]
        {
            d.fields.insert(k.into(), v);
        }
        let steps = app.session.active().unwrap().history.past_len();
        crate::dialogs::confirm(&mut app, id).unwrap();
        let l = top(&app);
        assert_eq!((l.name.as_str(), l.blend, l.label), ("Dodge & Burn", BlendMode::Overlay, LabelColor::Violet));
        assert!((l.opacity - 0.6).abs() < 1e-6);
        assert!(l.surface().unwrap().tile_count() > 0, "filled with 50% gray");
        assert_eq!(app.session.active().unwrap().history.past_len(), steps + 1);
        // Params given (automation, Actions): no dialog.
        crate::menus::invoke(&mut app, &ctx, COMMAND, json!({"name": "Direct"})).unwrap();
        assert_eq!(top(&app).name, "Direct");
    }

    #[test]
    fn params_follow_photoshop_rules() {
        let f = json!({MARK: true, "name": "  ", "mode": "Hue", "opacity": 250.0, "fillNeutral": true, "color": "plaid", "clipToPrevious": true});
        let p = params(f.as_object().unwrap());
        assert_eq!(p, json!({"clipToPrevious": true, "color": "none", "mode": "Hue", "opacity": 1.0, "fillNeutral": false}));
        assert_eq!(neutral_label(BlendMode::Multiply), "Fill with Multiply-neutral color (white)");
        assert_eq!(neutral_label(BlendMode::Screen), "Fill with Screen-neutral color (black)");
        assert_eq!(neutral_label(BlendMode::SoftLight), "Fill with Soft Light-neutral color (50% gray)");
        assert_eq!(neutral_label(BlendMode::Normal), "(No neutral color exists for Normal mode.)");
    }

    /// The dialog shows Photoshop's controls; the neutral fill is greyed for a mode without one.
    #[test]
    fn the_dialog_shows_its_controls() {
        let mut app = app();
        open(&mut app).unwrap();
        let mut h = egui_kittest::Harness::builder().with_size(egui::vec2(900.0, 600.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                let ctx = ui.ctx().clone();
                if ctx.fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    crate::dialogs::show(app, &ctx);
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(4);
        assert!(h.query_by_label("New Layer").is_some());
        assert!(h.query_by_label("Use Previous Layer to Create Clipping Mask").is_some());
        let fill = h.get_by_label("(No neutral color exists for Normal mode.)");
        assert!(fill.accesskit_node().is_disabled());
        assert!(h.query_by_label("OK").is_some() && h.query_by_label("Cancel").is_some());
        // It opens on the name, selected: typing replaces it.
        h.event(egui::Event::Text("Shade".into()));
        h.run_steps(2);
        assert_eq!(h.state().ui.dialogs[0].fields["name"], "Shade");
    }
}
