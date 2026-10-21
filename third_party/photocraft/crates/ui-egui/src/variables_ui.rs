//! Front end for Image › Variables (Define… / Data Sets…) and Apply Data Set — Photoshop's
//! data-driven graphics dialog. All editing drives the headless engine commands
//! (`image.variables.define`, `image.variables.dataSets`, `image.applyDataSet`), so the control
//! channel can drive this with `ui.dialog.set` / `confirm` exactly like a human.
//!
//! The dialog keeps its working state as JSON in the `__variables` field, shaped like the engine
//! command params, so `confirm` passes it straight through.

use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::Tokens;

/// This dialog owns a field set when it carries `__variables`.
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key("__variables")
}

/// Alt+Cancel (Reset): restore `__variables` from the open-time snapshot.
pub fn reset(fields: &mut Map<String, Value>) {
    if !owns(fields) {
        return;
    }
    if let Some(orig) = fields.get("__variablesOrig").cloned() {
        fields.insert("__variables".into(), orig);
    }
}

/// Layers available to bind, top to bottom: (id, display name, is-text).
fn layer_options(app: &PhotocraftApp) -> Vec<(u64, String, bool)> {
    let Some(st) = app.session.active() else { return Vec::new() };
    st.doc.walk().iter().map(|(_, _, l)| (l.id.0, l.name.clone(), matches!(l.content, photocraft_doc::LayerContent::Text(_)))).collect()
}

/// Menu entry points. Returns None for ids this module doesn't own.
pub fn menu(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    let has_params = params.as_object().is_some_and(|o| !o.is_empty());
    match id {
        "image.variables.define" | "image.variables.dataSets" if !has_params => {
            let page = if id.ends_with("dataSets") { "dataSets" } else { "define" };
            let mut f = Map::new();
            let state = app.session.active().map(|st| crate::variables_ui::to_state(&st.doc.variables)).unwrap_or_else(|| json!({"defs": [], "dataSets": []}));
            // Snapshot at open so Alt+Reset restores the document's variables.
            f.insert("__variablesOrig".into(), state.clone());
            f.insert("__variables".into(), state);
            f.insert("__page".into(), json!(page));
            f.insert("__label".into(), json!("Variables"));
            Some(Ok(json!({"dialog": app.ui.open_dialog(DialogKind::Command, f)})))
        }
        "image.applyDataSet" if !has_params => {
            // No data set named: open the dialog on the Data Sets page to pick one.
            let mut f = Map::new();
            let state = app.session.active().map(|st| crate::variables_ui::to_state(&st.doc.variables)).unwrap_or_else(|| json!({"defs": [], "dataSets": []}));
            f.insert("__variablesOrig".into(), state.clone());
            f.insert("__variables".into(), state);
            f.insert("__page".into(), json!("dataSets"));
            f.insert("__label".into(), json!("Variables"));
            Some(Ok(json!({"dialog": app.ui.open_dialog(DialogKind::Command, f)})))
        }
        _ => None,
    }
}

/// Serialise the document's variables into the dialog's editable JSON shape.
pub fn to_state(v: &photocraft_doc::Variables) -> Value {
    json!({
        "defs": v.defs.iter().map(|d| {
            let (ty, method, align, clip) = match &d.kind {
                photocraft_doc::VarKind::Visibility => ("visibility", "fit", "center", false),
                photocraft_doc::VarKind::TextReplacement => ("textReplacement", "fit", "center", false),
                photocraft_doc::VarKind::PixelReplacement { method, align, clip } => {
                    (_pixel(), _m(method), _a(align), *clip)
                }
            };
            json!({"name": d.name, "layer": d.layer.0, "type": ty, "method": method, "align": align, "clip": clip})
        }).collect::<Vec<_>>(),
        "dataSets": v.data_sets.iter().map(|s| json!({
            "name": s.name,
            "values": s.values.iter().map(|dv| {
                let (kind, value) = match &dv.value {
                    photocraft_doc::variables::Value::Visibility(b) => ("visibility", json!(b)),
                    photocraft_doc::variables::Value::Text(t) => ("text", json!(t)),
                    photocraft_doc::variables::Value::Pixels(p) => ("pixels", json!(p)),
                };
                json!({"variable": dv.variable, "kind": kind, "value": value})
            }).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

fn _pixel() -> &'static str {
    "pixelReplacement"
}
fn _m(m: &photocraft_doc::PixelMethod) -> &'static str {
    use photocraft_doc::PixelMethod::*;
    match m {
        Fit => "fit",
        Fill => "fill",
        AsIs => "asIs",
        Conform => "conform",
    }
}
fn _a(a: &photocraft_doc::PixelAlign) -> &'static str {
    use photocraft_doc::PixelAlign::*;
    match a {
        TopLeft => "topLeft",
        TopCenter => "topCenter",
        TopRight => "topRight",
        CenterLeft => "centerLeft",
        Center => "center",
        CenterRight => "centerRight",
        BottomLeft => "bottomLeft",
        BottomCenter => "bottomCenter",
        BottomRight => "bottomRight",
    }
}

const TYPES: &[(&str, &str)] = &[("visibility", "Visibility"), ("textReplacement", "Text Replacement"), ("pixelReplacement", "Pixel Replacement")];
const METHODS: &[(&str, &str)] = &[("fit", "Fit"), ("fill", "Fill"), ("conform", "Conform"), ("asIs", "As Is")];

/// Render the dialog. Mutates `fields["__variables"]` and `fields["__page"]`.
pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, fields: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let layers = layer_options(app);
    let mut state = normalize(fields.get("__variables").unwrap_or(&Value::Null));
    let mut page = fields.get("__page").and_then(Value::as_str).unwrap_or("define").to_string();

    ui.set_min_width(420.0);
    // Page switcher.
    ui.horizontal(|ui| {
        for (id, label) in [("define", tl!("Define")), ("dataSets", tl!("Data Sets"))] {
            let sel = page == id;
            if ui.selectable_label(sel, tl!(&label)).clicked() {
                page = id.to_string();
            }
        }
    });
    ui.separator();

    if layers.is_empty() {
        ui.weak(tl!("Open a document to define variables."));
    } else if page == "define" {
        define_page(ui, &t, &layers, &mut state);
    } else {
        data_sets_page(ui, &t, &mut state, fields);
    }

    fields.insert("__variables".into(), state);
    fields.insert("__page".into(), json!(page));
}

/// The dialog state with only object entries in `defs`, `dataSets` and each set's `values`.
/// `ui.dialog.set` can store any JSON in `__variables`, and the pages write fields into entries
/// (`d["name"] = …`), which panics on a number or a string.
fn normalize(state: &Value) -> Value {
    let objects =
        |v: Option<&Value>| -> Vec<Value> { v.and_then(Value::as_array).map(|a| a.iter().filter(|e| e.is_object()).cloned().collect()).unwrap_or_default() };
    let sets: Vec<Value> = objects(state.get("dataSets"))
        .into_iter()
        .map(|mut set| {
            set["values"] = Value::Array(objects(set.get("values")));
            set
        })
        .collect();
    json!({"defs": objects(state.get("defs")), "dataSets": sets})
}

fn define_page(ui: &mut egui::Ui, _t: &Tokens, layers: &[(u64, String, bool)], state: &mut Value) {
    let Some(defs) = state.get_mut("defs").and_then(Value::as_array_mut) else { return };
    ui.label(tl!("Variables bind a layer's visibility, text or pixels to a named data slot."));
    ui.add_space(4.0);
    let mut remove = None;
    for (i, d) in defs.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            // name
            let mut name = d.get("name").and_then(Value::as_str).unwrap_or("").to_string();
            if ui.add(egui::TextEdit::singleline(&mut name).hint_text(tl!("name")).desired_width(110.0)).changed() {
                d["name"] = json!(name);
            }
            // layer
            let mut layer = d.get("layer").and_then(Value::as_u64).unwrap_or(layers[0].0);
            let opts: Vec<(u64, &str)> = layers.iter().map(|(id, n, _)| (*id, n.as_str())).collect();
            if crate::widgets::dropdown(ui, &format!("vl{i}"), &mut layer, &opts, 120.0) {
                d["layer"] = json!(layer);
            }
            // type
            let mut ty = d.get("type").and_then(Value::as_str).unwrap_or("visibility").to_string();
            let mut tsel = ty.clone();
            if crate::widgets::dropdown(ui, &format!("vt{i}"), &mut tsel, &TYPES.iter().map(|(a, b)| (a.to_string(), *b)).collect::<Vec<_>>(), 150.0) {
                ty = tsel.clone();
                d["type"] = json!(ty);
            }
            if ty == "pixelReplacement" {
                let mut m = d.get("method").and_then(Value::as_str).unwrap_or("fit").to_string();
                if crate::widgets::dropdown(ui, &format!("vm{i}"), &mut m, &METHODS.iter().map(|(a, b)| (a.to_string(), *b)).collect::<Vec<_>>(), 80.0) {
                    d["method"] = json!(m);
                }
            }
            if ui.button("−").on_hover_text(tl!("Remove variable")).clicked() {
                remove = Some(i);
            }
        });
    }
    if let Some(i) = remove {
        defs.remove(i);
    }
    ui.add_space(4.0);
    if ui.button(tl!("+ Add variable")).clicked() {
        let n = defs.len() + 1;
        defs.push(json!({"name": format!("var{n}"), "layer": layers[0].0, "type": "visibility", "method": "fit", "align": "center", "clip": false}));
    }
}

fn data_set_index(fields: &Map<String, Value>, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    fields.get("__cur").and_then(Value::as_u64).and_then(|v| usize::try_from(v).ok()).filter(|&i| i < len).unwrap_or(0)
}

fn data_sets_page(ui: &mut egui::Ui, _t: &Tokens, state: &mut Value, fields: &mut Map<String, Value>) {
    let def_meta: Vec<(String, String)> = state
        .get("defs")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .map(|d| {
                    (d.get("name").and_then(Value::as_str).unwrap_or("").to_string(), d.get("type").and_then(Value::as_str).unwrap_or("visibility").to_string())
                })
                .collect()
        })
        .unwrap_or_default();
    if def_meta.is_empty() {
        ui.weak(tl!("Define at least one variable first."));
        return;
    }
    let Some(sets) = state.get_mut("dataSets").and_then(Value::as_array_mut) else { return };
    let mut cur = data_set_index(fields, sets.len());

    ui.horizontal(|ui| {
        if ui.button("◀").clicked() && cur > 0 {
            cur -= 1;
        }
        let name = sets
            .get(cur)
            .and_then(|s| s.get("name"))
            .and_then(Value::as_str)
            .unwrap_or_else(|| {
                let empty = "(none)";
                tl!(empty)
            })
            .to_string();
        ui.label({
            let template = "Data Set: {name}  ({index}/{total})";
            crate::i18n::fmt(
                tl!(template),
                &[("name", &name), ("index", &(if sets.is_empty() { 0 } else { cur + 1 }).to_string()), ("total", &sets.len().to_string())],
            )
        });
        if ui.button("▶").clicked() && cur + 1 < sets.len() {
            cur += 1;
        }
        if ui.button(tl!("New")).clicked() {
            let n = sets.len() + 1;
            sets.push(json!({"name": format!("Data Set {n}"), "values": []}));
            cur = sets.len() - 1;
        }
        if !sets.is_empty() && ui.button(tl!("Delete")).clicked() {
            sets.remove(cur);
            cur = cur.saturating_sub(1);
        }
    });
    ui.separator();

    if let Some(set) = sets.get_mut(cur) {
        // name
        let mut sname = set.get("name").and_then(Value::as_str).unwrap_or("").to_string();
        ui.horizontal(|ui| {
            ui.label(tl!("Name:"));
            if ui.add(egui::TextEdit::singleline(&mut sname).desired_width(200.0)).changed() {
                set["name"] = json!(sname);
            }
        });
        ui.add_space(4.0);
        let Some(values) = set.get_mut("values").and_then(Value::as_array_mut) else { return };
        for (vname, vty) in &def_meta {
            // find or create a value entry for this variable
            let idx = values.iter().position(|v| v.get("variable").and_then(Value::as_str) == Some(vname));
            let i = match idx {
                Some(i) => i,
                None => {
                    let default = match vty.as_str() {
                        "visibility" => json!({"variable": vname, "kind": "visibility", "value": true}),
                        "pixelReplacement" => json!({"variable": vname, "kind": "pixels", "value": ""}),
                        _ => json!({"variable": vname, "kind": "text", "value": ""}),
                    };
                    values.push(default);
                    values.len() - 1
                }
            };
            ui.horizontal(|ui| {
                ui.add_sized([130.0, 18.0], egui::Label::new(vname.as_str()).truncate());
                match vty.as_str() {
                    "visibility" => {
                        let mut on = values[i].get("value").and_then(Value::as_bool).unwrap_or(true);
                        if ui.checkbox(&mut on, tl!("Visible")).changed() {
                            values[i]["value"] = json!(on);
                        }
                    }
                    "pixelReplacement" => {
                        let mut p = values[i].get("value").and_then(Value::as_str).unwrap_or("").to_string();
                        if ui.add(egui::TextEdit::singleline(&mut p).hint_text(tl!("image file path")).desired_width(240.0)).changed() {
                            values[i]["value"] = json!(p);
                        }
                    }
                    _ => {
                        let mut txt = values[i].get("value").and_then(Value::as_str).unwrap_or("").to_string();
                        if ui.add(egui::TextEdit::singleline(&mut txt).desired_width(240.0)).changed() {
                            values[i]["value"] = json!(txt);
                        }
                    }
                }
            });
        }
        ui.add_space(6.0);
        if crate::widgets::secondary_button(ui, tl!("Apply"), 84.0).clicked() {
            fields.insert("__apply".into(), json!(sname));
        }
    }
    fields.insert("__cur".into(), json!(cur));
}

/// OK: save the definitions and data sets, and apply one if the Apply button set `__apply`.
pub fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let state = fields.get("__variables").cloned().unwrap_or_else(|| json!({}));
    let defs = state.get("defs").cloned().unwrap_or_else(|| json!([]));
    let sets = state.get("dataSets").cloned().unwrap_or_else(|| json!([]));
    // Only define when there is at least one named variable (empty define clears).
    app.run("image.variables.define", json!({"defs": defs}))?;
    app.run("image.variables.dataSets", json!({"dataSets": sets}))?;
    if let Some(name) = fields.get("__apply").and_then(Value::as_str) {
        return app.run("image.applyDataSet", json!({"name": name}));
    }
    Ok(json!({"defined": true}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{DataSet, DataValue, LayerId, VarKind, VariableDef, Variables, variables::Value as VV};

    #[test]
    fn to_state_mirrors_the_command_shape() {
        let v = Variables {
            defs: vec![
                VariableDef { name: "show".into(), layer: LayerId(1), kind: VarKind::Visibility },
                VariableDef { name: "title".into(), layer: LayerId(2), kind: VarKind::TextReplacement },
            ],
            data_sets: vec![DataSet { name: "A".into(), values: vec![DataValue { variable: "show".into(), value: VV::Visibility(false) }] }],
            active: None,
        };
        let s = to_state(&v);
        assert_eq!(s["defs"][0]["name"], "show");
        assert_eq!(s["defs"][0]["type"], "visibility");
        assert_eq!(s["defs"][1]["type"], "textReplacement");
        assert_eq!(s["dataSets"][0]["name"], "A");
        assert_eq!(s["dataSets"][0]["values"][0]["kind"], "visibility");
        assert_eq!(s["dataSets"][0]["values"][0]["value"], false);
    }

    #[test]
    fn data_set_index_rejects_out_of_range_control_values() {
        let mut fields = Map::new();
        assert_eq!(data_set_index(&fields, 1), 0);
        fields.insert("__cur".into(), json!(999));
        assert_eq!(data_set_index(&fields, 1), 0);
        fields.insert("__cur".into(), json!(u64::MAX));
        assert_eq!(data_set_index(&fields, 1), 0);
        fields.insert("__cur".into(), json!(1));
        assert_eq!(data_set_index(&fields, 3), 1);
        assert_eq!(data_set_index(&fields, 0), 0);
    }

    #[test]
    fn normalize_keeps_only_object_entries() {
        let s = normalize(&json!({"defs": [1, {"name": "a"}], "dataSets": ["x", {"name": "A", "values": [true, {"variable": "a"}]}, {"values": 3}]}));
        assert_eq!(s, json!({"defs": [{"name": "a"}], "dataSets": [{"name": "A", "values": [{"variable": "a"}]}, {"values": []}]}));
        for bad in [Value::Null, json!(5), json!("s"), json!([1]), json!({"defs": "x", "dataSets": {}})] {
            assert_eq!(normalize(&bad), json!({"defs": [], "dataSets": []}), "{bad}");
        }
    }

    #[test]
    fn alt_reset_restores_open_snapshot() {
        let mut f = Map::new();
        let orig = json!({"defs": [{"name": "a"}], "dataSets": []});
        f.insert("__variables".into(), json!({"defs": [{"name": "edited"}], "dataSets": []}));
        f.insert("__variablesOrig".into(), orig.clone());
        reset(&mut f);
        assert_eq!(f["__variables"], orig);
    }

    #[test]
    fn owns_detects_the_field() {
        let mut f = Map::new();
        assert!(!owns(&f));
        f.insert("__variables".into(), json!({}));
        assert!(owns(&f));
    }
}
