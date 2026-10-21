//! Properties-panel editors for the Selective Color and Color Lookup adjustment layers.
//!
//! Both commit through `layer.setAdjustment` (which merges these kinds' params into the current
//! values). Selective Color previews live while a slider drags, via `app.live_adjust` carrying
//! the full parameter set; Color Lookup commits on every change (its table isn't in the params).

use photocraft_doc::{Adjustment, LayerId};
use photocraft_engine::adjust_cmds::RANGES;
use serde_json::{Value, json};

use crate::PhotocraftApp;
use crate::theme::Tokens;
use crate::widgets;

const RANGE_LABELS: [&str; 9] = ["Reds", "Yellows", "Greens", "Cyans", "Blues", "Magentas", "Whites", "Neutrals", "Blacks"];

/// The full parameter set of a Selective Color adjustment (every range as `[c, m, y, k]`).
pub fn selective_values(adj: &Adjustment) -> Value {
    let Adjustment::SelectiveColor { relative, adjustments } = adj else {
        return json!({});
    };
    let mut v = json!({"method": if *relative { "relative" } else { "absolute" }});
    for (i, key) in RANGES.iter().enumerate() {
        v[*key] = json!(adjustments[i]);
    }
    v
}

pub fn selective_color_editor(app: &mut PhotocraftApp, ui: &mut egui::Ui, id: LayerId, adj: &Adjustment) {
    let t = Tokens::get(ui.ctx());
    let committed = selective_values(adj);
    let mut values = match &app.live_adjust {
        Some((l, v)) if *l == id && v.get("reds").is_some() => v.clone(),
        _ => committed,
    };
    let key = egui::Id::new(("selc-range", id.0));
    let mut range: usize = ui.data(|d| d.get_temp(key)).unwrap_or(0);
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("Colors")).color(t.text_dim));
        let opts: Vec<(usize, &str)> = RANGE_LABELS.iter().copied().enumerate().collect();
        if widgets::dropdown(ui, &format!("selc-colors-{}", id.0), &mut range, &opts, 150.0) {
            ui.data_mut(|d| d.insert_temp(key, range));
        }
    });
    ui.add_space(4.0);
    let mut live = false;
    let mut commit = false;
    for (k, label) in [tl!("Cyan"), tl!("Magenta"), tl!("Yellow"), tl!("Black")].iter().enumerate() {
        let mut v = values[RANGES[range]][k].as_f64().unwrap_or(0.0) as f32;
        let r = widgets::slider_row(ui, label, &mut v, -100.0..=100.0, "%", None);
        if r.changed() {
            values[RANGES[range]][k] = json!(v.round());
            live = true;
        }
        if r.drag_stopped() || (r.changed() && !r.dragged()) {
            commit = true;
        }
    }
    ui.add_space(4.0);
    let mut method = values["method"].as_str().unwrap_or("relative").to_string();
    ui.horizontal(|ui| {
        for (m, label) in [("relative", tl!("Relative")), ("absolute", tl!("Absolute"))] {
            if ui.radio(method == m, label).clicked() && method != m {
                method = m.to_string();
                values["method"] = json!(m);
                commit = true;
            }
        }
    });
    if live {
        app.live_adjust = Some((id, values.clone()));
    }
    if commit {
        let mut p = values;
        p["layer"] = json!(id.0);
        let _ = app.run("layer.setAdjustment", p);
        app.live_adjust = None;
    }
}

/// Ask the platform for one LUT. Send embedded text through the same engine command on
/// desktop and web, so the result doesn't depend on ambient filesystem permissions.
fn browse_color_lookup(app: &mut PhotocraftApp, layer: LayerId) -> Result<Value, String> {
    let doc = app.active_doc_id()?;
    app.pick_file_bytes_filtered(&["cube", "3dl", "look"], move |app, file_name, bytes| {
        // Reject invalid UTF-8 instead of silently changing the LUT before parsing.
        let data = String::from_utf8(bytes).map_err(|_| format!("{file_name}: LUT must contain UTF-8 text"))?;
        app.refocus(doc)?;
        app.run("layer.setAdjustment", json!({"layer": layer.0, "fileName": file_name, "data": data}))
    })
}

pub fn color_lookup_editor(app: &mut PhotocraftApp, ui: &mut egui::Ui, id: LayerId, adj: &Adjustment) {
    let t = Tokens::get(ui.ctx());
    let Adjustment::ColorLookup { name, lut, size, tetrahedral, dither } = adj else {
        return;
    };
    let builtins = photocraft_engine::adjust_cmds::LOOKS;
    let current =
        if lut.is_none() { "none".to_string() } else { builtins.iter().find(|b| b.1 == name).map_or_else(|| "custom".to_string(), |b| b.0.to_string()) };
    let current = app
        .live_adjust
        .as_ref()
        .filter(|(layer, params)| *layer == id && params.get("lut").is_some())
        .and_then(|(_, params)| params.get("lut").and_then(Value::as_str))
        .map(str::to_owned)
        .unwrap_or(current);
    let custom_label = format!("{name} ({size}³)");
    let mut opts: Vec<(String, &str)> = vec![("load".into(), tl!("Load 3D LUT…")), ("none".into(), tl!("None"))];
    opts.extend(builtins.iter().map(|(id, label)| (id.to_string(), *label)));
    if current == "custom" {
        opts.push(("custom".into(), custom_label.as_str()));
    }
    let mut sel = current.clone();
    let mut params: Option<Value> = None;
    let mut browse = false;
    let cycle_key = egui::Id::new(("color-lookup-cycle", id.0));
    let cycle_input = ui.input(|input| {
        input.raw.events.iter().any(|event| {
            matches!(event, egui::Event::MouseWheel { .. })
                || matches!(event, egui::Event::Key { key: egui::Key::ArrowDown | egui::Key::ArrowUp, pressed: true, .. })
        })
    });
    let mut cycling_preview = false;
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(tl!("3D LUT File")).color(t.text_dim));
        let (chosen, _) = widgets::dropdown_wheel_hovered(ui, &format!("clrl-{}", id.0), &mut sel, &opts, 170.0);
        if !chosen.is_empty() {
            if cycle_input {
                if sel != "load" && sel != "custom" {
                    app.live_adjust = Some((id, json!({"lut": sel})));
                    cycling_preview = true;
                }
            } else if sel == "load" {
                browse = true;
            } else if sel != current && sel != "custom" {
                params = Some(json!({"lut": sel}));
            }
        }
    });
    if cycling_preview {
        let now = ui.input(|input| input.time);
        ui.ctx().data_mut(|data| data.insert_temp(cycle_key, now));
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
    } else if let Some(since) = ui.ctx().data(|data| data.get_temp::<f64>(cycle_key)) {
        if ui.input(|input| input.time - since) >= 0.75 {
            if let Some((layer, preview)) = &app.live_adjust
                && *layer == id
                && preview.get("lut").is_some()
            {
                params = Some(preview.clone());
            }
            ui.ctx().data_mut(|data| data.remove::<f64>(cycle_key));
        } else {
            ui.ctx().request_repaint_after(std::time::Duration::from_millis(100));
        }
    }
    if browse
        && let Err(e) = browse_color_lookup(app, id)
        && e != crate::file_dialog::CANCELLED
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    ui.add_space(4.0);
    let mut tet = *tetrahedral;
    ui.horizontal(|ui| {
        for (on, label) in [(false, tl!("Trilinear")), (true, tl!("Tetrahedral"))] {
            if ui.radio(tet == on, label).clicked() && tet != on {
                tet = on;
                params = Some(json!({"tetrahedral": on}));
            }
        }
    });
    let mut d = *dither;
    if widgets::toggle(ui, &mut d, tl!("Dither")).changed() {
        params = Some(json!({"dither": d}));
    }
    if let Some(mut p) = params {
        if p.get("lut").is_none()
            && let Some((layer, preview)) = &app.live_adjust
            && *layer == id
            && let Some(lut) = preview.get("lut")
        {
            p["lut"] = lut.clone();
        }
        p["layer"] = json!(id.0);
        // Errors (an unreadable LUT file) land in the status bar.
        let _ = app.run("layer.setAdjustment", p);
        app.live_adjust = None;
        ui.ctx().data_mut(|data| data.remove::<f64>(cycle_key));
    }
}

#[cfg(test)]
mod lookup_picker_tests {
    use super::*;
    use crate::file_dialog::{self, FileDialogAnswer, FileDialogRequest};
    use serde_json::json;

    const CUBE: &str = "LUT_3D_SIZE 2\n0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";

    #[test]
    fn chosen_lut_is_embedded_into_the_requested_adjustment_layer() {
        let (dialog, asked) = file_dialog::fake(vec![Some(FileDialogAnswer::Contents("custom.cube".into(), CUBE.as_bytes().to_vec()))]);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { file_dialog: Some(dialog), ..Default::default() });
        app.run("file.new", json!({"width": 16, "height": 12})).unwrap();
        let layer = app.run("layer.newAdjustmentLayer.colorLookup", json!({})).unwrap()["layer"].as_u64().unwrap();
        browse_color_lookup(&mut app, LayerId(layer)).unwrap();
        app.poll_file_dialog(&egui::Context::default(), None);
        assert!(
            matches!(asked.borrow().first(), Some(FileDialogRequest::Open { multiple: false, extensions: Some(exts), .. }) if exts.iter().map(String::as_str).collect::<Vec<_>>() == ["cube", "3dl", "look"])
        );
        app.poll_file_dialog(&egui::Context::default(), None);
        let adj = &app.session.active().unwrap().doc.layer(LayerId(layer)).unwrap().content;
        assert!(
            matches!(adj, photocraft_doc::LayerContent::Adjustment(Adjustment::ColorLookup { name, lut: Some(_), size: 2, .. }) if name == "custom.cube"),
            "{adj:?}"
        );
    }

    #[test]
    fn invalid_lut_does_not_modify_the_existing_adjustment() {
        let (dialog, _) = file_dialog::fake(vec![Some(FileDialogAnswer::Contents("broken.cube".into(), b"invalid".to_vec()))]);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { file_dialog: Some(dialog), ..Default::default() });
        app.run("file.new", json!({"width": 16, "height": 12})).unwrap();
        let layer = app.run("layer.newAdjustmentLayer.colorLookup", json!({})).unwrap()["layer"].as_u64().unwrap();
        let before = app.session.active().unwrap().doc.layer(LayerId(layer)).unwrap().content.clone();
        browse_color_lookup(&mut app, LayerId(layer)).unwrap();
        app.poll_file_dialog(&egui::Context::default(), None);
        app.poll_file_dialog(&egui::Context::default(), None);
        assert_eq!(app.session.active().unwrap().doc.layer(LayerId(layer)).unwrap().content, before);
        assert!(app.ui.status_error && app.ui.status.contains("colorLookup"));
    }

    #[test]
    fn cancelling_the_lut_picker_preserves_the_adjustment() {
        let (dialog, _) = file_dialog::fake(vec![None]);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { file_dialog: Some(dialog), ..Default::default() });
        app.run("file.new", json!({"width": 16, "height": 12})).unwrap();
        let layer = app.run("layer.newAdjustmentLayer.colorLookup", json!({})).unwrap()["layer"].as_u64().unwrap();
        let before = app.session.active().unwrap().doc.layer(LayerId(layer)).unwrap().content.clone();
        browse_color_lookup(&mut app, LayerId(layer)).unwrap();
        app.poll_file_dialog(&egui::Context::default(), None);
        app.poll_file_dialog(&egui::Context::default(), None);
        assert_eq!(app.session.active().unwrap().doc.layer(LayerId(layer)).unwrap().content, before);
        assert!(!app.ui.status_error);
    }
}
