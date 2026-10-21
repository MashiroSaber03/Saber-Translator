//! Modal Image › Adjustments dialogs (Levels, Curves, Hue/Saturation, …) built from the same
//! editors as the adjustment-layer Properties panel (`adjust_editors`).
//!
//! The dialog's fields are the command's complete parameter set (so automation can read and set
//! them with `ui.dialog.set`) plus private `__` keys. While it is open the canvas previews the
//! settings as a temporary adjustment layer clipped to the target (`adjust_preview`, composited on
//! the GPU); where that can't match the command, the *real command* runs on the proxy document
//! through the filter-preview machinery (`__filter` + `__preview`). OK runs the command once (one
//! history step) and Cancel drops the preview, leaving the document untouched.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::adjust_editors::{self, EditorCx};
use crate::tone::{self, HistSource};

const PREFIX: &str = "image.adjustments.";
const CURVE_PICKER: &str = "__curvePicker";
const CURVE_PICKER_BASE: &str = "__curvePickerBase";

fn picker(value: Option<&Value>) -> Option<photocraft_engine::adjust_params::CurvesEyedropper> {
    use photocraft_engine::adjust_params::CurvesEyedropper;
    match value.and_then(Value::as_str) {
        Some("black") => Some(CurvesEyedropper::Black),
        Some("gray") => Some(CurvesEyedropper::NeutralGray),
        Some("white") => Some(CurvesEyedropper::White),
        _ => None,
    }
}

pub(crate) fn picker_supported(mode: photocraft_doc::ColorMode, eyedropper: photocraft_engine::adjust_params::CurvesEyedropper) -> bool {
    use photocraft_engine::adjust_params::CurvesEyedropper;
    mode == photocraft_doc::ColorMode::Rgb || (mode == photocraft_doc::ColorMode::Grayscale && eyedropper != CurvesEyedropper::NeutralGray)
}

/// The kind of an adjustment dialog with Set Black/Neutral Gray/White Point eyedroppers (Levels,
/// Curves), if `fields` are one.
fn picker_kind(fields: &Map<String, Value>) -> Option<&str> {
    fields.get("__adjust").and_then(Value::as_str).filter(|k| matches!(*k, "levels" | "curves"))
}

/// Whether the top modal has a supported Levels or Curves eyedropper armed.
pub fn picker_armed(app: &PhotocraftApp) -> bool {
    let Some(dialog) = app.ui.dialogs.last().filter(|d| picker_kind(&d.fields).is_some()) else { return false };
    let Some(eyedropper) = picker(dialog.fields.get(CURVE_PICKER)) else { return false };
    app.session.active().is_some_and(|s| picker_supported(s.doc.mode, eyedropper))
}

/// Sample the committed merged document through the regular canvas sampler and update only the
/// open Levels or Curves dialog. Returns true when an armed picker consumed the pointer event,
/// even if the point was outside the image or transparent.
pub fn sample_at(app: &mut PhotocraftApp, x: f64, y: f64) -> bool {
    let Some((id, kind, eyedropper, base)) = app.ui.dialogs.last().and_then(|dialog| {
        let kind = picker_kind(&dialog.fields)?.to_string();
        let eyedropper = picker(dialog.fields.get(CURVE_PICKER))?;
        let base = dialog.fields.get(CURVE_PICKER_BASE).cloned().unwrap_or_else(|| crate::filter_dialog::params_of(&dialog.fields));
        Some((dialog.id, kind, eyedropper, base))
    }) else {
        return false;
    };
    let Some(mode) = app.session.active().map(|s| s.doc.mode) else { return true };
    if !picker_supported(mode, eyedropper) {
        return true;
    }
    let Some(sample) = crate::canvas::composite_color(app, x, y) else { return true };
    let params = match photocraft_engine::adjust_params::adjustment_eyedropper(&kind, &base, eyedropper, sample, mode) {
        Ok(adjustment) => photocraft_engine::adjust_params::to_params(&adjustment),
        Err(error) => {
            app.ui.status = error.to_string();
            return true;
        }
    };
    let Some(dialog) = app.ui.dialog_mut(id) else { return true };
    dialog.fields.insert(CURVE_PICKER_BASE.into(), base);
    if let Value::Object(params) = params {
        dialog.fields.retain(|key, _| key.starts_with("__"));
        dialog.fields.extend(params);
    }
    true
}

pub(crate) fn picker_button(ui: &mut egui::Ui, selected: bool, enabled: bool, tooltip: &str, color: egui::Color32) -> egui::Response {
    let response = ui.add_enabled_ui(enabled, |ui| crate::icons::button(ui, "pipette", 28.0, selected, tooltip)).inner;
    let tokens = crate::theme::Tokens::get(ui.ctx());
    ui.painter().circle_filled(response.rect.right_bottom() - egui::vec2(5.5, 5.5), 3.5, color);
    ui.painter().circle_stroke(response.rect.right_bottom() - egui::vec2(5.5, 5.5), 3.5, egui::Stroke::new(1.0, tokens.field_border));
    response.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, enabled, selected, tooltip));
    response
}

fn picker_controls(ui: &mut egui::Ui, fields: &mut Map<String, Value>, mode: photocraft_doc::ColorMode) {
    use photocraft_engine::adjust_params::CurvesEyedropper;
    let current = picker(fields.get(CURVE_PICKER));
    let mut chosen = None;
    ui.horizontal(|ui| {
        let tokens = crate::theme::Tokens::get(ui.ctx());
        ui.label(egui::RichText::new(tl!("Eyedroppers:")).color(tokens.text_dim).size(12.0));
        // The swatch is the tone the picker maps the sample to (image data, like the Curves
        // gradient bars), not chrome, so it is the same in every theme; its ring is a token.
        for (id, eyedropper, label, tone) in [
            ("black", CurvesEyedropper::Black, tl!("Set Black Point"), 0),
            ("gray", CurvesEyedropper::NeutralGray, tl!("Set Neutral Gray Point"), 128),
            ("white", CurvesEyedropper::White, tl!("Set White Point"), 255),
        ] {
            let enabled = picker_supported(mode, eyedropper);
            if picker_button(ui, current == Some(eyedropper), enabled, label, egui::Color32::from_gray(tone)).clicked() {
                chosen = Some((id, eyedropper));
            }
        }
        if !matches!(mode, photocraft_doc::ColorMode::Rgb | photocraft_doc::ColorMode::Grayscale) {
            ui.label(egui::RichText::new(tl!("Unavailable in this color mode")).color(tokens.text_faint).size(11.0));
        }
    });
    if let Some((id, selected)) = chosen
        && current != Some(selected)
    {
        let base = crate::filter_dialog::params_of(fields);
        fields.insert(CURVE_PICKER.into(), json!(id));
        fields.insert(CURVE_PICKER_BASE.into(), base);
    }
}

/// Whether `command` opens an adjustment dialog (when run without params).
pub fn has_dialog(command: &str) -> bool {
    command.strip_prefix(PREFIX).is_some_and(adjust_editors::has_editor)
}

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key("__adjust")
}

/// Root of the dialog editor's view state in egui memory.
fn dialog_mem(kind: &str) -> egui::Id {
    egui::Id::new(("adjust-dialog", kind))
}

/// Whether holding Alt turns Cancel into Reset for this adjustment dialog (every kind).
pub fn resets(fields: &Map<String, Value>) -> bool {
    owns(fields)
}

/// Alt+Cancel (Reset): engine defaults for the kind; Color Balance keeps Preserve Luminosity and
/// shows Midtones (Photoshop 25.4). Dialog stays open.
pub fn reset(app: &PhotocraftApp, ctx: &egui::Context, fields: &mut Map<String, Value>) {
    if !owns(fields) {
        return;
    }
    let kind = fields.get("__adjust").and_then(Value::as_str).unwrap_or_default().to_string();
    let mode = app.session.active().map_or(photocraft_doc::ColorMode::Rgb, |s| s.doc.mode);
    // Color Balance: Photoshop keeps the user's Preserve Luminosity across Reset.
    let preserve = if kind == "colorBalance" { fields.get("preserveLuminosity").cloned() } else { None };
    let Ok(defaults) = photocraft_engine::adjust_params::default_for(&kind, mode) else { return };
    let values = photocraft_engine::adjust_params::to_params(&defaults);
    // Replace parameter keys; keep private `__` chrome (preview, command, …).
    fields.retain(|k, _| k.starts_with("__"));
    if let Value::Object(v) = values {
        fields.extend(v);
    }
    if let Some(p) = preserve {
        fields.insert("preserveLuminosity".into(), p);
    }
    let mut values = crate::filter_dialog::params_of(fields);
    adjust_editors::reset_adjustment_editor(ctx, &kind, dialog_mem(&kind), &mut values);
    if let Value::Object(v) = values {
        fields.extend(v);
    }
}

/// Opens the dialog for `command` with the kind's neutral settings.
pub fn open(app: &mut PhotocraftApp, command: &str) -> Option<u64> {
    let kind = command.strip_prefix(PREFIX).filter(|k| adjust_editors::has_editor(k))?;
    let spec = photocraft_engine::commands::find(command)?;
    let mode = app.session.active().map_or(photocraft_doc::ColorMode::Rgb, |s| s.doc.mode);
    let defaults = photocraft_engine::adjust_params::default_for(kind, mode).ok()?;
    let mut fields = Map::new();
    fields.insert("__command".into(), json!(command));
    fields.insert("__label".into(), json!(spec.label));
    fields.insert("__adjust".into(), json!(kind));
    fields.insert("__filter".into(), json!(true));
    fields.insert("__preview".into(), json!(true));
    if let Value::Object(p) = photocraft_engine::adjust_params::to_params(&defaults) {
        fields.extend(p);
    }
    Some(app.ui.open_dialog(crate::state::DialogKind::Command, fields))
}

/// Dialog body: the kind's editor over the fields, then the Preview checkbox.
pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let kind = fields.get("__adjust").and_then(Value::as_str).unwrap_or_default().to_string();
    let mode = app.session.active().map_or(photocraft_doc::ColorMode::Rgb, |s| s.doc.mode);
    if matches!(kind.as_str(), "levels" | "curves") {
        picker_controls(ui, fields, mode);
        ui.add_space(4.0);
    }
    let mut values = crate::filter_dialog::params_of(fields);
    let active = app.session.active().and_then(|s| s.active_layer);
    let gray = app.session.active().is_some_and(|s| adjust_editors::is_gray(s.doc.mode));
    let hist = match active {
        Some(id) if adjust_editors::needs_histogram(&kind) => Some(tone::histograms(app, HistSource::Layer(id), adjust_editors::space_of(&values))),
        _ => None,
    };
    let cx = EditorCx {
        mem: dialog_mem(&kind),
        hist,
        gray,
        swatches: adjust_editors::swatches(app),
        gradients: adjust_editors::gradient_presets(app, &kind),
        dialog: true,
        dynamic_color_sliders: app.session.prefs().interface.dynamic_color_sliders,
    };
    let e = adjust_editors::editor(ui, &kind, &mut values, &cx);
    if e.import_gradients {
        crate::preset_files_ui::import_gradients_dialog(app);
    }
    if e.changed
        && let Value::Object(v) = values
    {
        // Replace the parameter keys (an editor may remove one), keep the private ones.
        fields.retain(|k, _| k.starts_with("__"));
        fields.extend(v);
        if picker(fields.get(CURVE_PICKER)).is_some() {
            let base = crate::filter_dialog::params_of(fields);
            fields.insert(CURVE_PICKER_BASE.into(), base);
        }
    }
    let mut preview = fields.get("__preview").and_then(Value::as_bool).unwrap_or(true);
    ui.add_space(6.0);
    if crate::widgets::checkbox(ui, &mut preview, tl!("Preview")).changed() {
        fields.insert("__preview".into(), json!(preview));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui_kittest::Harness;
    use egui_kittest::kittest::{NodeT, Queryable};

    fn app_with_image() -> PhotocraftApp {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 32, "height": 48})).unwrap();
        s.execute("edit.fill", json!({"color": "#b04020"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        PhotocraftApp::new(s, crate::Services::default())
    }

    fn app_in_mode(mode: &str) -> PhotocraftApp {
        let mut session = photocraft_engine::Session::new();
        session.execute("file.new", json!({"width": 32, "height": 24, "mode": mode})).unwrap();
        PhotocraftApp::new(session, crate::Services::default())
    }

    fn arm(app: &mut PhotocraftApp, id: u64, picker: &str) {
        let base = app.ui.dialog_mut(id).map(|dialog| crate::filter_dialog::params_of(&dialog.fields)).unwrap();
        let dialog = app.ui.dialog_mut(id).unwrap();
        dialog.fields.insert(CURVE_PICKER.into(), json!(picker));
        dialog.fields.insert(CURVE_PICKER_BASE.into(), base);
    }

    fn channel_points(app: &PhotocraftApp, key: &str) -> Vec<[f64; 2]> {
        app.ui
            .dialogs
            .last()
            .and_then(|dialog| dialog.fields.get(key))
            .and_then(Value::as_array)
            .map(|points| {
                points
                    .iter()
                    .filter_map(|point| {
                        let point = point.as_array()?;
                        Some([point.first()?.as_f64()?, point.get(1)?.as_f64()?])
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    #[test]
    fn every_adjustment_with_an_editor_has_a_dialog() {
        for kind in adjust_editors::KINDS {
            assert!(has_dialog(&format!("image.adjustments.{kind}")), "{kind}");
        }
        for id in ["image.adjustments.invert", "image.adjustments.selectiveColor", "image.adjustments.desaturate", "filter.blur.gaussianBlur"] {
            assert!(!has_dialog(id), "{id}");
        }
    }

    /// Opens each dialog, renders it, changes a value, then cancels (document untouched) or
    /// confirms (one history step).
    #[test]
    fn dialogs_preview_cancel_and_commit() {
        for kind in adjust_editors::KINDS {
            let cmd = format!("image.adjustments.{kind}");
            let mut harness =
                Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
            PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
            let id = open(harness.state_mut(), &cmd).unwrap();
            harness.run_steps(3);
            let before = harness.state().session.active().unwrap().revision;
            let steps = harness.state().session.active().unwrap().history.past_len();
            // Change the settings the way automation does.
            let sample = photocraft_engine::adjust_params::to_params(
                &photocraft_engine::adjust_params::from_params(kind, &sample_params(kind), None, photocraft_doc::ColorMode::Rgb).unwrap(),
            );
            let d = harness.state_mut().ui.dialog_mut(id).unwrap();
            if let Value::Object(p) = sample {
                d.fields.extend(p);
            }
            harness.run_steps(2);
            // The preview runs the real command on the proxy.
            let fields = harness.state().ui.dialogs[0].fields.clone();
            let doc = harness.state().session.active().unwrap().doc.clone();
            let prev = crate::filter_dialog::preview_document(
                &doc,
                harness.state().session.active().unwrap().active_layer,
                &cmd,
                &crate::filter_dialog::params_of(&fields),
                1,
            );
            assert!(prev.is_some(), "{kind}: preview");
            assert_eq!(harness.state().session.active().unwrap().revision, before, "{kind}: previewing leaves the document alone");
            let r = crate::dialogs::confirm(harness.state_mut(), id);
            assert!(r.is_ok(), "{kind}: {r:?}");
            assert_eq!(harness.state().session.active().unwrap().history.past_len(), steps + 1, "{kind}: one history step");
        }
    }

    fn color_balance_dialog() -> (Harness<'static, PhotocraftApp>, u64) {
        // Real frame times: a double-click's two clicks must land within the double-click time.
        let mut h = Harness::builder()
            .with_size(egui::vec2(1200.0, 900.0))
            .with_step_dt(1.0 / 60.0)
            .build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        let id = open(h.state_mut(), "image.adjustments.colorBalance").unwrap();
        let d = h.state_mut().ui.dialog_mut(id).unwrap();
        for (k, v) in
            [("shadows", json!([10, 0, 0])), ("midtones", json!([40, -30, 20])), ("highlights", json!([0, 0, -15])), ("preserveLuminosity", json!(false))]
        {
            d.fields.insert(k.into(), v);
        }
        h.run_steps(3);
        (h, id)
    }

    fn field(h: &Harness<'static, PhotocraftApp>, id: u64, key: &str) -> Value {
        h.state().ui.dialogs.iter().find(|d| d.id == id).map(|d| d.fields.get(key).cloned().unwrap_or(Value::Null)).unwrap_or(Value::Null)
    }

    fn input(h: &mut Harness<'static, PhotocraftApp>, mods: egui::Modifiers, events: &[egui::Event]) {
        h.event(egui::Event::ModifiersChanged(mods));
        h.run_steps(1);
        for e in events {
            h.event(e.clone());
            h.run_steps(1);
        }
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
        h.run_steps(3);
    }

    /// Photoshop 25.4's Color Balance dialog (measured with the real application): a double-click
    /// on a slider zeroes it whatever the modifiers; the wheel over a slider steps it by 1, with
    /// Shift by 10 (Ctrl/Alt: 1); holding Alt turns Cancel into Reset, which zeroes every tone,
    /// shows Midtones, keeps Preserve Luminosity and leaves the dialog open.
    #[test]
    fn color_balance_dialog_gestures_match_photoshop() {
        use egui_kittest::kittest::Queryable;
        let (mut h, id) = color_balance_dialog();
        let label = h.query_all_by_label("Cyan  ·  Red").last().map(|n| n.rect()).expect("Cyan · Red row");
        let at = egui::pos2(label.left() + 60.0, label.bottom() + 14.0);
        let press = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::ALT };
        input(&mut h, egui::Modifiers::ALT, &[egui::Event::PointerMoved(at), press(true), press(false), press(true), press(false)]);
        assert_eq!(field(&h, id, "midtones"), json!([0.0, -30.0, 20.0]), "Alt+double-click zeroes the slider");
        assert_eq!(field(&h, id, "shadows"), json!([10, 0, 0]), "other tones untouched");
        let wheel =
            |n: f32, m| egui::Event::MouseWheel { unit: egui::MouseWheelUnit::Line, delta: egui::vec2(0.0, n), phase: egui::TouchPhase::Move, modifiers: m };
        input(&mut h, egui::Modifiers::NONE, &[egui::Event::PointerMoved(at), wheel(3.0, egui::Modifiers::NONE)]);
        assert_eq!(field(&h, id, "midtones")[0], json!(3.0), "3 notches up");
        input(&mut h, egui::Modifiers::SHIFT, &[egui::Event::PointerMoved(at), wheel(2.0, egui::Modifiers::SHIFT)]);
        assert_eq!(field(&h, id, "midtones")[0], json!(23.0), "Shift: 10 per notch");
        input(&mut h, egui::Modifiers::CTRL, &[egui::Event::PointerMoved(at), wheel(-1.0, egui::Modifiers::CTRL)]);
        assert_eq!(field(&h, id, "midtones")[0], json!(22.0), "Ctrl: 1 per notch, down");
        // Alt turns Cancel into Reset.
        assert!(h.query_by_label("Cancel").is_some() && h.query_by_label("Reset").is_none());
        h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("highlights".into(), json!([5, 5, 5]));
        let tone = egui::Id::new(("adjust-dialog", "colorBalance")).with("cb-tone");
        h.ctx.data_mut(|d| d.insert_temp(tone, 2usize));
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::ALT));
        h.run_steps(2);
        let reset = h.get_by_label("Reset").rect();
        assert!(h.query_by_label("Cancel").is_none());
        let click =
            |pressed| egui::Event::PointerButton { pos: reset.center(), button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::ALT };
        input(&mut h, egui::Modifiers::ALT, &[egui::Event::PointerMoved(reset.center()), click(true), click(false)]);
        assert!(h.state().ui.dialogs.iter().any(|d| d.id == id), "Reset keeps the dialog open");
        for k in ["shadows", "midtones", "highlights"] {
            assert_eq!(field(&h, id, k), json!([0.0, 0.0, 0.0]), "{k}");
        }
        assert_eq!(field(&h, id, "preserveLuminosity"), json!(false), "Reset keeps Preserve Luminosity");
        assert_eq!(h.ctx.data(|d| d.get_temp::<usize>(tone)), Some(1), "Reset shows Midtones");
        // Without Alt it is Cancel again, and it closes.
        let cancel = h.get_by_label("Cancel").rect();
        let click =
            |pressed| egui::Event::PointerButton { pos: cancel.center(), button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        input(&mut h, egui::Modifiers::NONE, &[egui::Event::PointerMoved(cancel.center()), click(true), click(false)]);
        assert!(h.state().ui.dialogs.is_empty(), "Cancel closes");
    }

    /// Every adjustment kind opts into Alt+Reset; Levels restores engine defaults and stays open.
    #[test]
    fn levels_alt_reset_restores_defaults() {
        use egui_kittest::kittest::Queryable;
        let mut h = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        let id = open(h.state_mut(), "image.adjustments.levels").unwrap();
        h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("inBlack".into(), json!(30));
        h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("gamma".into(), json!(1.3));
        h.run_steps(2);
        assert!(h.query_by_label("Cancel").is_some() && h.query_by_label("Reset").is_none());
        h.event(egui::Event::ModifiersChanged(egui::Modifiers::ALT));
        h.run_steps(2);
        let reset = h.get_by_label("Reset").rect();
        assert!(h.query_by_label("Cancel").is_none());
        let click =
            |pressed| egui::Event::PointerButton { pos: reset.center(), button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::ALT };
        input(&mut h, egui::Modifiers::ALT, &[egui::Event::PointerMoved(reset.center()), click(true), click(false)]);
        assert!(h.state().ui.dialogs.iter().any(|d| d.id == id), "Reset keeps the dialog open");
        let defaults =
            photocraft_engine::adjust_params::to_params(&photocraft_engine::adjust_params::default_for("levels", photocraft_doc::ColorMode::Rgb).unwrap());
        assert_eq!(field(&h, id, "inBlack"), defaults["inBlack"]);
        assert_eq!(field(&h, id, "gamma"), defaults["gamma"]);
    }

    fn sample_params(kind: &str) -> Value {
        match kind {
            "brightnessContrast" => json!({"brightness": 40}),
            "levels" => json!({"inBlack": 30, "gamma": 1.3}),
            "curves" => json!({"points": [[0, 0], [100, 150], [255, 255]]}),
            "exposure" => json!({"exposure": 1}),
            "vibrance" => json!({"vibrance": 50}),
            "hueSaturation" => json!({"reds": {"hue": 40}}),
            "colorBalance" => json!({"midtones": [40, 0, -20]}),
            "blackWhite" => json!({"reds": 150, "tint": true}),
            "photoFilter" => json!({"filter": "cooling80", "density": 60}),
            "channelMixer" => json!({"red": [50, 50, 0, 0]}),
            "posterize" => json!({"levels": 3}),
            "threshold" => json!({"level": 90}),
            "gradientMap" => json!({"stops": [[0, "#200040"], [1, "#ffd080"]]}),
            _ => json!({}),
        }
    }

    #[test]
    fn cancel_leaves_the_document_untouched() {
        let mut harness =
            Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
        PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
        let id = open(harness.state_mut(), "image.adjustments.curves").unwrap();
        harness.run_steps(2);
        harness.state_mut().ui.dialog_mut(id).unwrap().fields.insert("points".into(), json!([[0, 255], [255, 0]]));
        harness.run_steps(2);
        let before = harness.state().session.active().unwrap().doc.clone();
        harness.state_mut().ui.close_dialog(id);
        harness.run_steps(2);
        let after = harness.state().session.active().unwrap().doc.clone();
        assert!(std::sync::Arc::ptr_eq(&before, &after));
        assert!(harness.state().ui.dialogs.is_empty());
    }

    #[test]
    fn gradient_map_dialog_handles_preview_without_committing_until_ok() {
        use egui_kittest::kittest::Queryable;
        let mut h = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
        PhotocraftApp::setup_context(&h.ctx, Default::default());
        let id = open(h.state_mut(), "image.adjustments.gradientMap").unwrap();
        h.run_steps(4);
        let doc = h.state().session.active().unwrap().doc.clone();
        let steps = h.state().session.active().unwrap().history.past_len();
        let strip = h.get_by_label("Gradient stops").rect();
        let start = egui::pos2(strip.left() + 8.0, strip.top() + 36.0);
        let to = start + egui::vec2((strip.width() - 16.0) * 0.3, 0.0);
        h.hover_at(start);
        h.run_steps(1);
        h.drag_at(start);
        h.run_steps(1);
        h.hover_at(to);
        h.run_steps(1);
        h.drop_at(to);
        h.run_steps(3);
        let stops = h.state().ui.dialogs[0].fields["stops"].as_array().unwrap();
        assert!((stops[0][0].as_f64().unwrap() - 0.3).abs() < 0.001);
        assert!(std::sync::Arc::ptr_eq(&doc, &h.state().session.active().unwrap().doc), "dialog edits are previews");
        assert_eq!(h.state().session.active().unwrap().history.past_len(), steps);
        crate::dialogs::confirm(h.state_mut(), id).unwrap();
        h.run_steps(2);
        assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
        h.state_mut().run("edit.undo", json!({})).unwrap();
        assert_eq!(*h.state().session.active().unwrap().doc, *doc);
    }
    #[test]
    fn curves_pencil_drag_draws_the_map_and_confirm_applies_it() {
        let mut h = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        let steps = h.state().session.active().unwrap().history.past_len();
        let id = open(h.state_mut(), "image.adjustments.curves").unwrap();
        let fields = &mut h.state_mut().ui.dialog_mut(id).unwrap().fields;
        fields.insert("pencil".into(), json!(true));
        fields.insert("points".into(), json!((0..256).collect::<Vec<_>>()));
        h.run_steps(4);
        let graph = h.ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::new(("adjust-dialog", "curves")).with("curves-graph"))).unwrap();
        let at = |i: f32, o: f32| egui::pos2(graph.left() + i / 255.0 * graph.width(), graph.bottom() - o / 255.0 * graph.height());
        h.hover_at(at(64.0, 200.0));
        h.run_steps(1);
        h.drag_at(at(64.0, 200.0));
        h.run_steps(1);
        for i in [100.0, 150.0, 192.0] {
            h.hover_at(at(i, 200.0));
            h.run_steps(1);
        }
        h.drop_at(at(192.0, 200.0));
        h.run_steps(3);
        let map = h.state().ui.dialogs[0].fields["points"].as_array().unwrap().clone();
        assert_eq!(map.len(), 256);
        assert!((70..185).all(|k| (map[k].as_f64().unwrap() - 200.0).abs() < 3.0), "{map:?}");
        assert_eq!(map[20].as_f64(), Some(20.0), "outside the stroke the map is untouched");
        crate::dialogs::confirm(h.state_mut(), id).unwrap();
        assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
    }

    #[test]
    fn curves_pointer_drag_previews_until_confirm_and_cancel_preserves_document() {
        for confirm in [false, true] {
            let mut h = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with_image());
            PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
            let original = h.state().session.active().unwrap().doc.clone();
            let steps = h.state().session.active().unwrap().history.past_len();
            let id = open(h.state_mut(), "image.adjustments.curves").unwrap();
            h.state_mut().ui.dialog_mut(id).unwrap().fields.insert("points".into(), json!([[0, 0], [128, 128], [255, 255]]));
            h.run_steps(4);
            let graph = h.ctx.data(|d| d.get_temp::<egui::Rect>(egui::Id::new(("adjust-dialog", "curves")).with("curves-graph"))).unwrap();
            let a = egui::pos2(graph.left() + 128.0 / 255.0 * graph.width(), graph.bottom() - 128.0 / 255.0 * graph.height());
            h.hover_at(a);
            h.run_steps(1);
            h.drag_at(a);
            h.run_steps(1);
            h.hover_at(a + egui::vec2(40.0, -32.0));
            h.run_steps(1);
            h.drop_at(a + egui::vec2(40.0, -32.0));
            h.run_steps(3);
            let points = &h.state().ui.dialogs[0].fields["points"];
            assert_eq!(points.as_array().unwrap().len(), 3);
            assert!(points[1][0].as_f64().unwrap() > 150.0 && points[1][1].as_f64().unwrap() > 150.0, "{points}");
            assert!(std::sync::Arc::ptr_eq(&original, &h.state().session.active().unwrap().doc));
            assert_eq!(h.state().session.active().unwrap().history.past_len(), steps);
            if confirm {
                crate::dialogs::confirm(h.state_mut(), id).unwrap();
                assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
                h.state_mut().run("edit.undo", json!({})).unwrap();
                assert!(std::sync::Arc::ptr_eq(&original, &h.state().session.active().unwrap().doc));
            } else {
                h.state_mut().ui.close_dialog(id);
                h.run_steps(2);
                assert!(std::sync::Arc::ptr_eq(&original, &h.state().session.active().unwrap().doc));
            }
        }
    }

    #[test]
    fn curves_eyedroppers_sample_repeatedly_without_editing_the_document() {
        let mut app = app_with_image();
        let original = app.session.active().unwrap().doc.clone();
        let revision = app.session.active().unwrap().revision;
        let steps = app.session.active().unwrap().history.past_len();
        let id = open(&mut app, "image.adjustments.curves").unwrap();
        arm(&mut app, id, "black");
        assert!(picker_armed(&app));
        assert!(sample_at(&mut app, 8.0, 8.0));
        let first = channel_points(&app, "red");
        assert!((first[0][0] - 176.0).abs() < 0.6 && first[0][1] == 0.0, "{first:?}");
        assert!(sample_at(&mut app, 8.0, 8.0));
        assert_eq!(channel_points(&app, "red"), first, "the same repeated sample is deterministic");
        assert!(sample_at(&mut app, 48.0, 8.0));
        assert_ne!(channel_points(&app, "red"), first);
        assert!(sample_at(&mut app, 8.0, 8.0));
        assert_eq!(channel_points(&app, "red"), first, "repeated samples resolve from the armed baseline");
        let st = app.session.active().unwrap();
        assert!(std::sync::Arc::ptr_eq(&st.doc, &original));
        assert_eq!(st.revision, revision);
        assert_eq!(st.history.past_len(), steps);
    }

    /// #3116: the Levels dialog's Set White Point moves each channel's input white to the sample
    /// (#b04020) and leaves the document alone until OK.
    #[test]
    fn levels_white_point_eyedropper_sets_the_channel_input_whites() {
        let mut app = app_with_image();
        let original = app.session.active().unwrap().doc.clone();
        let id = open(&mut app, "image.adjustments.levels").unwrap();
        arm(&mut app, id, "white");
        assert!(picker_armed(&app));
        assert!(sample_at(&mut app, 8.0, 8.0));
        let in_white = |key: &str| app.ui.dialogs.last().and_then(|d| d.fields.get(key)).and_then(|c| c.get("inWhite")).and_then(Value::as_f64);
        assert_eq!([in_white("red"), in_white("green"), in_white("blue")], [Some(176.0), Some(64.0), Some(32.0)]);
        assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
    }

    #[test]
    fn curves_picker_controls_arm_and_disable_by_document_mode() {
        let harness = |app| {
            let mut h = Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app);
            PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
            open(h.state_mut(), "image.adjustments.curves").unwrap();
            h.run_steps(3);
            h
        };
        let mut rgb = harness(app_in_mode("rgb"));
        for label in ["Set Black Point", "Set Neutral Gray Point", "Set White Point"] {
            assert!(!rgb.get_by_label(label).accesskit_node().is_disabled(), "{label}");
        }
        rgb.get_by_label("Set Black Point").click();
        rgb.run_steps(2);
        assert!(picker_armed(rgb.state()));
        assert_eq!(rgb.state().ui.dialogs.last().and_then(|d| d.fields.get(CURVE_PICKER)).and_then(Value::as_str), Some("black"));

        let gray = harness(app_in_mode("gray"));
        assert!(!gray.get_by_label("Set Black Point").accesskit_node().is_disabled());
        assert!(gray.get_by_label("Set Neutral Gray Point").accesskit_node().is_disabled());
        assert!(!gray.get_by_label("Set White Point").accesskit_node().is_disabled());

        for mode in ["cmyk", "lab"] {
            let h = harness(app_in_mode(mode));
            for label in ["Set Black Point", "Set Neutral Gray Point", "Set White Point"] {
                assert!(h.get_by_label(label).accesskit_node().is_disabled(), "{mode}: {label}");
            }
        }
    }

    #[test]
    fn curves_picker_rejects_outside_and_transparent_samples() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 32, "height": 24, "background": "transparent"})).unwrap();
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        let id = open(&mut app, "image.adjustments.curves").unwrap();
        arm(&mut app, id, "white");
        let before = crate::filter_dialog::params_of(&app.ui.dialog_mut(id).unwrap().fields);
        assert!(sample_at(&mut app, -1.0, 4.0));
        assert_eq!(crate::filter_dialog::params_of(&app.ui.dialog_mut(id).unwrap().fields), before);
        assert!(sample_at(&mut app, 4.0, 4.0));
        assert_eq!(crate::filter_dialog::params_of(&app.ui.dialog_mut(id).unwrap().fields), before);
        assert!(picker_armed(&app), "an invalid sample does not disarm the picker");
    }

    #[test]
    fn curves_picker_cancel_and_confirm_keep_one_transaction() {
        for confirm in [false, true] {
            let mut app = app_with_image();
            let original = app.session.active().unwrap().doc.clone();
            let steps = app.session.active().unwrap().history.past_len();
            let id = open(&mut app, "image.adjustments.curves").unwrap();
            arm(&mut app, id, "white");
            assert!(sample_at(&mut app, 8.0, 8.0));
            assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
            assert_eq!(app.session.active().unwrap().history.past_len(), steps);
            if confirm {
                crate::dialogs::confirm(&mut app, id).unwrap();
                let committed = app.session.active().unwrap().doc.clone();
                assert!(!std::sync::Arc::ptr_eq(&committed, &original));
                assert_eq!(app.session.active().unwrap().history.past_len(), steps + 1);
                app.run("edit.undo", json!({})).unwrap();
                assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
                app.run("edit.redo", json!({})).unwrap();
                assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &committed));
            } else {
                app.ui.close_dialog(id);
                assert!(std::sync::Arc::ptr_eq(&app.session.active().unwrap().doc, &original));
                assert_eq!(app.session.active().unwrap().history.past_len(), steps);
            }
        }
    }

    #[test]
    fn curves_picker_modes_and_depths_are_explicit() {
        use photocraft_engine::adjust_params::CurvesEyedropper;
        assert!(picker_supported(photocraft_doc::ColorMode::Rgb, CurvesEyedropper::NeutralGray));
        assert!(picker_supported(photocraft_doc::ColorMode::Grayscale, CurvesEyedropper::Black));
        assert!(picker_supported(photocraft_doc::ColorMode::Grayscale, CurvesEyedropper::White));
        assert!(!picker_supported(photocraft_doc::ColorMode::Grayscale, CurvesEyedropper::NeutralGray));
        assert!(!picker_supported(photocraft_doc::ColorMode::Cmyk, CurvesEyedropper::Black));
        assert!(!picker_supported(photocraft_doc::ColorMode::Lab, CurvesEyedropper::White));

        for depth in [8, 16, 32] {
            let mut s = photocraft_engine::Session::new();
            s.execute("file.new", json!({"width": 16, "height": 16, "depth": depth})).unwrap();
            s.execute("edit.fill", json!({"color": [0.3142, 0.4821, 0.7137]})).unwrap();
            let mut app = PhotocraftApp::new(s, crate::Services::default());
            let sampled = crate::canvas::composite_color(&mut app, 4.0, 4.0).unwrap();
            let id = open(&mut app, "image.adjustments.curves").unwrap();
            arm(&mut app, id, "gray");
            assert!(sample_at(&mut app, 4.0, 4.0), "RGB @{depth}");
            let red = channel_points(&app, "red");
            let target_level = (f64::from(sampled[0]) * 0.2126 + f64::from(sampled[1]) * 0.7152 + f64::from(sampled[2]) * 0.0722) * 255.0;
            let sample_level = f64::from(sampled[0]) * 255.0;
            let serialized_sample_level = (sample_level * 100.0).round() / 100.0;
            let neutral_anchor = red.iter().find(|point| (point[0] - serialized_sample_level).abs() < 1e-5).unwrap();
            let serialized_target_level = (target_level * 100.0).round() / 100.0;
            assert!((neutral_anchor[1] - serialized_target_level).abs() < 1e-5, "depth={depth} anchor={neutral_anchor:?} sampled={sampled:?}");
            assert!((neutral_anchor[0] - sample_level).abs() <= 0.0051, "depth={depth} anchor={neutral_anchor:?} sampled={}", sampled[0]);
            if depth == 16 || depth == 32 {
                assert!(
                    (sampled[0] * 255.0 - (sampled[0] * 255.0).round()).abs() > 0.01,
                    "depth={depth} sample was quantized to an 8-bit level: {}",
                    sampled[0]
                );
            }
        }
    }

    #[test]
    fn ui_pointer_drives_an_armed_curves_picker() {
        let mut app = app_with_image();
        let id = open(&mut app, "image.adjustments.curves").unwrap();
        arm(&mut app, id, "black");
        let context = egui::Context::default();
        let before = channel_points(&app, "red");
        let (request, _receiver) = crate::control::ControlRequest::new("ui.pointer", json!({"events": [{"kind": "move", "x": 48, "y": 8}]}));
        let _ = crate::control::handle(&mut app, &context, &request);
        assert_eq!(channel_points(&app, "red"), before, "pointer move does not sample");

        let (request, _receiver) = crate::control::ControlRequest::new("ui.pointer", json!({"events": [{"kind": "down", "x": 8, "y": 8}]}));
        let _ = crate::control::handle(&mut app, &context, &request);
        let red = channel_points(&app, "red");
        assert!((red[0][0] - 176.0).abs() < 0.6 && red[0][1] == 0.0, "{red:?}");
        assert_ne!(red, before, "pointer down samples");

        let (request, _receiver) = crate::control::ControlRequest::new("ui.pointer", json!({"events": [{"kind": "up", "x": 48, "y": 8}]}));
        let _ = crate::control::handle(&mut app, &context, &request);
        assert_eq!(channel_points(&app, "red"), red, "pointer up does not sample again");
        assert!(picker_armed(&app));
    }
}
