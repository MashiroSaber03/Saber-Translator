//! View › Guides › New Guide…: an orientation, a position with a unit and a colour, as in
//! Photoshop.
//!
//! The position opens in the ruler unit ("0 cm") and takes any unit Photoshop's field takes:
//! "50%", "2.5 cm", "1in", "12 pt", "3 pc", "100 px". A bare number is in the ruler unit, as in
//! Photoshop. OK runs `view.newGuide` with the text, so the engine converts it with the document's
//! resolution. A position the engine refuses reopens the dialog with what was typed and the
//! engine's message, as Photoshop keeps New Guide open on a bad position. The
//! dialog is an ordinary one in the UI state, so `ui.inspect`, `ui.dialog.confirm` and
//! `ui.dialog.cancel` drive it. The last successful guide's orientation and colour open next
//! time, with its position shown in the current ruler unit, as Photoshop does.
//!
//! Color lists Photoshop's guide colours (Cyan by default) and Custom…, whose swatch opens the
//! Color Picker. The guide stores the chosen colour itself, so changing the preference guide
//! colour later doesn't recolour it.

use photocraft_engine::edit_cmds::GUIDE_COLORS;
use photocraft_engine::prefs::Unit;
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;

const MARK: &str = "__newGuide";
const COMMAND: &str = "view.newGuide";
const CUSTOM: &str = "custom";
/// New Guide's colour when nothing was remembered, as in Photoshop.
const DEFAULT_COLOR: &str = "cyan";

/// A colour's label in the list (the catalogue holds each one).
pub(crate) fn color_label(name: &str) -> &'static str {
    match name {
        "lightBlue" => tl!("Light Blue"),
        "lightRed" => tl!("Light Red"),
        "green" => tl!("Green"),
        "mediumBlue" => tl!("Medium Blue"),
        "yellow" => tl!("Yellow"),
        "magenta" => tl!("Magenta"),
        "cyan" => tl!("Cyan"),
        "lightGray" => tl!("Light Gray"),
        "black" => tl!("Black"),
        _ => tl!("Custom…"),
    }
}

fn is_color_name(name: &str) -> bool {
    name == CUSTOM || GUIDE_COLORS.iter().any(|(n, ..)| *n == name)
}

/// The Custom… colour as "#rrggbb", starting from Cyan's.
fn custom_color(f: &Map<String, Value>) -> String {
    f.get("customColor")
        .and_then(Value::as_str)
        .filter(|h| photocraft_engine::prefs::parse_hex(h).is_some())
        .map_or_else(|| "#4affff".to_string(), str::to_string)
}

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(MARK)
}

fn ruler_unit(app: &PhotocraftApp) -> Unit {
    app.session.prefs().units_and_rulers.rulers
}

/// "0 cm", "0 px", "0 %": a value in `unit` as the field shows it, with a space before the unit
/// as in Photoshop.
fn with_unit(value: &str, unit: Unit) -> String {
    format!("{value} {}", unit.suffix())
}

/// `px` (document pixels) in the ruler unit, as the field shows it: "14.11 cm". Percent is of the
/// active document's width for a vertical guide and its height for a horizontal one.
fn in_ruler_unit(app: &PhotocraftApp, px: f64, vertical: bool) -> String {
    let units = &app.session.prefs().units_and_rulers;
    let (dpi, extent) = app.session.active().map_or((72.0, 1.0), |st| {
        let d = &st.doc;
        (f64::from(d.resolution_dpi), f64::from(if vertical { d.size.width } else { d.size.height }))
    });
    with_unit(&units.format(px, dpi, extent), units.rulers)
}

/// Open the dialog with the last successful guide's orientation, position (in the current ruler
/// unit) and colour, else horizontal at 0.
pub fn open(app: &mut PhotocraftApp) -> u64 {
    let saved = app.session.prefs().dialogs.get(COMMAND).and_then(Value::as_object).cloned().unwrap_or_default();
    let orientation = saved.get("orientation").and_then(Value::as_str).filter(|o| matches!(*o, "horizontal" | "vertical")).unwrap_or("horizontal");
    let px = saved.get("positionPx").and_then(Value::as_f64).filter(|p| p.is_finite()).unwrap_or(0.0);
    let position = in_ruler_unit(app, px, orientation == "vertical");
    let color = saved.get("color").and_then(Value::as_str).filter(|c| is_color_name(c)).unwrap_or(DEFAULT_COLOR).to_string();
    let custom = custom_color(&saved);
    let mut f = Map::new();
    f.insert(MARK.into(), json!(true));
    f.insert("__command".into(), json!(COMMAND));
    f.insert("__label".into(), json!("New Guide…"));
    f.insert("orientation".into(), json!(orientation));
    f.insert("position".into(), json!(position));
    f.insert("color".into(), json!(color));
    f.insert("customColor".into(), json!(custom));
    app.ui.open_dialog(DialogKind::Command, f)
}

pub fn body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let mut orientation = f.get("orientation").and_then(Value::as_str).unwrap_or("horizontal").to_string();
    let mut position = f.get("position").and_then(Value::as_str).unwrap_or_default().to_string();
    let mut color = f.get("color").and_then(Value::as_str).filter(|c| is_color_name(c)).unwrap_or(DEFAULT_COLOR).to_string();
    if let Some(error) = f.get("__error").and_then(Value::as_str) {
        ui.colored_label(ui.visuals().error_fg_color, error);
        ui.add_space(4.0);
    }
    egui::Grid::new("new-guide-dialog").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
        ui.label(tl!("Orientation"));
        ui.horizontal(|ui| {
            ui.radio_value(&mut orientation, "horizontal".to_string(), tl!("Horizontal"));
            ui.radio_value(&mut orientation, "vertical".to_string(), tl!("Vertical"));
        });
        ui.end_row();
        ui.label(tl!("Position"));
        ui.add(egui::TextEdit::singleline(&mut position).desired_width(120.0));
        ui.end_row();
        ui.label(tl!("Color"));
        ui.horizontal(|ui| {
            // Custom… first, then Photoshop's colours.
            let options: Vec<(String, &str)> =
                std::iter::once(CUSTOM).chain(GUIDE_COLORS.iter().map(|(n, ..)| *n)).map(|n| (n.to_string(), color_label(n))).collect();
            crate::widgets::dropdown(ui, "new-guide-color", &mut color, &options, 140.0);
            if color == CUSTOM {
                let hex = custom_color(f);
                let rgb = photocraft_engine::prefs::parse_hex(&hex).unwrap_or([0x4a, 0xff, 0xff]);
                if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]), tl!("Custom…")).clicked() {
                    f.insert("customColor".into(), json!(hex));
                    crate::color_picker_ui::request_field(f, "customColor", &crate::color_picker_ui::title_for(tl!("Guides")));
                }
                ui.label(egui::RichText::new(hex).monospace());
            }
        });
        ui.end_row();
    });
    f.insert("orientation".into(), json!(orientation));
    f.insert("position".into(), json!(position));
    f.insert("color".into(), json!(color));
}

/// The command's parameters: a bare number gets the ruler unit; anything else goes as typed, for
/// the engine to read (or refuse, keeping the dialog open). Custom… sends its "#rrggbb".
pub fn params(app: &PhotocraftApp, f: &Map<String, Value>) -> Value {
    let orientation = f.get("orientation").and_then(Value::as_str).unwrap_or("horizontal");
    let typed = f.get("position").and_then(Value::as_str).unwrap_or_default().trim();
    let position = match Unit::parse_length(typed) {
        Some((_, None)) => with_unit(typed, ruler_unit(app)),
        _ => typed.to_string(),
    };
    let color = f.get("color").and_then(Value::as_str).filter(|c| is_color_name(c)).unwrap_or(DEFAULT_COLOR);
    let color = if color == CUSTOM { custom_color(f) } else { color.to_string() };
    json!({"orientation": orientation, "position": position, "color": color})
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let params = params(app, f);
    match app.run(COMMAND, params.clone()) {
        Ok(result) => {
            // Remember where the guide landed (so it reopens in whatever the ruler unit is then),
            // the list's choice (not the colour it sent) and the last Custom… colour.
            let vertical = params["orientation"] == "vertical";
            let landed = app.session.active().and_then(|st| if vertical { st.doc.guides.vertical.last() } else { st.doc.guides.horizontal.last() }.copied());
            let remembered = json!({
                "orientation": params["orientation"],
                "positionPx": landed.unwrap_or(0.0),
                "color": f.get("color").cloned().unwrap_or(json!(DEFAULT_COLOR)),
                "customColor": custom_color(f),
            });
            app.session.prefs.edit(|prefs| prefs.dialogs.insert(COMMAND.into(), remembered));
            Ok(result)
        }
        Err(e) => {
            // `dialogs::confirm` has already closed it: open it again with the typed values.
            let mut again = f.clone();
            again.insert("__error".into(), json!(e));
            app.ui.open_dialog(DialogKind::Command, again);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests;
