//! Image › Variables and Data Sets, and the File menu's data-set import/export — Photoshop's
//! data-driven graphics. Define variables that bind a layer's visibility, text or pixels to named
//! data slots, then apply a data set (a row of values) to render a variation. Everything is
//! headless and scriptable, so an agent can bind a template once and batch-render many files over
//! the CLI/MCP (`file.export.dataSetsAsFiles`).

use std::sync::Arc;

use photocraft_algo::resample::{Resample, resize_surface, translate_surface};
use photocraft_doc::text::{ParagraphRun, TextRun};
use photocraft_doc::variables::Value as VarValue;
use photocraft_doc::{DataSet, DataValue, Document, LayerContent, LayerId, PixelAlign, PixelMethod, VarKind, VariableDef, Variables};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document".into())
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

// --------------------------------------------------------------------------- parse helpers

fn parse_kind(v: &Value) -> Option<VarKind> {
    match v.get("type").and_then(Value::as_str)? {
        "visibility" => Some(VarKind::Visibility),
        "textReplacement" | "text" => Some(VarKind::TextReplacement),
        "pixelReplacement" | "pixel" => {
            let method = match v.get("method").and_then(Value::as_str) {
                Some("fill") => PixelMethod::Fill,
                Some("asIs") => PixelMethod::AsIs,
                Some("conform") => PixelMethod::Conform,
                _ => PixelMethod::Fit,
            };
            let align = match v.get("align").and_then(Value::as_str) {
                Some("topLeft") => PixelAlign::TopLeft,
                Some("topCenter") => PixelAlign::TopCenter,
                Some("topRight") => PixelAlign::TopRight,
                Some("centerLeft") => PixelAlign::CenterLeft,
                Some("centerRight") => PixelAlign::CenterRight,
                Some("bottomLeft") => PixelAlign::BottomLeft,
                Some("bottomCenter") => PixelAlign::BottomCenter,
                Some("bottomRight") => PixelAlign::BottomRight,
                _ => PixelAlign::Center,
            };
            let clip = v.get("clip").and_then(Value::as_bool).unwrap_or(false);
            Some(VarKind::PixelReplacement { method, align, clip })
        }
        _ => None,
    }
}

fn kind_json(k: &VarKind) -> Value {
    match k {
        VarKind::Visibility => json!({"type": "visibility"}),
        VarKind::TextReplacement => json!({"type": "textReplacement"}),
        VarKind::PixelReplacement { method, align, clip } => {
            let a = format!("{align:?}");
            let align = a[..1].to_lowercase() + &a[1..];
            json!({
                "type": "pixelReplacement",
                "method": method,
                "align": align,
                "clip": clip,
            })
        }
    }
}

/// `a` with `b`'s fields added (both are JSON objects).
fn merged(mut a: Value, b: Value) -> Value {
    if let (Some(ao), Value::Object(bo)) = (a.as_object_mut(), b) {
        ao.extend(bo);
    }
    a
}

fn value_json(v: &VarValue) -> Value {
    match v {
        VarValue::Visibility(b) => json!({"kind": "visibility", "value": b}),
        VarValue::Text(t) => json!({"kind": "text", "value": t}),
        VarValue::Pixels(p) => json!({"kind": "pixels", "value": p}),
    }
}

fn parse_value(v: &Value) -> Option<VarValue> {
    let var = v.get("variable").and_then(Value::as_str)?.to_string();
    let _ = var; // the caller keeps the name
    match v.get("kind").and_then(Value::as_str) {
        Some("visibility") => Some(VarValue::Visibility(v.get("value").and_then(Value::as_bool)?)),
        Some("text") => Some(VarValue::Text(v.get("value").and_then(Value::as_str)?.to_string())),
        Some("pixels") => Some(VarValue::Pixels(v.get("value").and_then(Value::as_str)?.to_string())),
        _ => None,
    }
}

/// Mutate the document's variables without a history step (config, not a canvas edit).
fn set_vars(s: &mut Session, f: impl FnOnce(&mut Variables)) -> Result<()> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let mut doc = (*st.doc).clone();
    f(&mut doc.variables);
    st.doc = Arc::new(doc);
    st.revision += 1;
    Ok(())
}

// --------------------------------------------------------------------------- commands

fn define(s: &mut Session, p: &Value) -> Result<Value> {
    let defs_in = p.get("defs").and_then(Value::as_array).ok_or_else(|| bad("image.variables.define", "need `defs`"))?;
    let doc = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    let mut defs = Vec::with_capacity(defs_in.len());
    for d in defs_in {
        let name = d.get("name").and_then(Value::as_str).ok_or_else(|| bad("image.variables.define", "a def needs `name`"))?;
        let layer = LayerId(d.get("layer").and_then(Value::as_u64).ok_or_else(|| bad("image.variables.define", "a def needs `layer`"))?);
        if doc.layer(layer).is_none() {
            return Err(EngineError::NoLayer(layer));
        }
        let kind = parse_kind(d).ok_or_else(|| bad("image.variables.define", "a def needs a valid `type`"))?;
        if defs.iter().any(|e: &VariableDef| e.name == name) {
            return Err(bad("image.variables.define", format!("duplicate variable `{name}`")));
        }
        defs.push(VariableDef { name: name.to_string(), layer, kind });
    }
    set_vars(s, |v| v.defs = defs)?;
    list(s)
}

fn data_sets(s: &mut Session, p: &Value) -> Result<Value> {
    let rows = p.get("dataSets").and_then(Value::as_array).ok_or_else(|| bad("image.variables.dataSets", "need `dataSets`"))?;
    let mut sets = Vec::with_capacity(rows.len());
    for r in rows {
        let name = r.get("name").and_then(Value::as_str).ok_or_else(|| bad("image.variables.dataSets", "a data set needs `name`"))?;
        let mut values = Vec::new();
        for v in r.get("values").and_then(Value::as_array).into_iter().flatten() {
            let variable = v.get("variable").and_then(Value::as_str).ok_or_else(|| bad("image.variables.dataSets", "a value needs `variable`"))?;
            let value = parse_value(v).ok_or_else(|| bad("image.variables.dataSets", "a value needs `kind` and `value`"))?;
            values.push(DataValue { variable: variable.to_string(), value });
        }
        sets.push(DataSet { name: name.to_string(), values });
    }
    let append = p.get("append").and_then(Value::as_bool).unwrap_or(false);
    set_vars(s, |v| {
        if append {
            v.data_sets.extend(sets);
        } else {
            v.data_sets = sets;
        }
    })?;
    list(s)
}

/// Apply one data set's values to the document (visibility, text, pixels). A history step.
fn apply_to_doc(doc: &mut Document, vars: &Variables, set: &DataSet) -> Result<()> {
    for dv in &set.values {
        let Some(def) = vars.def(&dv.variable) else { continue };
        match (&def.kind, &dv.value) {
            (VarKind::Visibility, VarValue::Visibility(show)) => {
                if let Some(l) = doc.layer_mut(def.layer) {
                    l.visible = *show;
                }
            }
            (VarKind::TextReplacement, VarValue::Text(text)) => {
                let snapshot = doc.clone();
                if let Some(l) = doc.layer_mut(def.layer)
                    && let LayerContent::Text(t) = &mut l.content
                {
                    // As in Photoshop, the new text takes the style of the first character it
                    // replaces (and its paragraph's settings): clearing the runs fell back to the
                    // summary style and lost weight, italic, tracking, caps… (#2742).
                    let style = t.char_runs().into_iter().next().map(|r| r.style).unwrap_or_default();
                    let para = t.paragraph_runs().into_iter().next().map(|r| r.style).unwrap_or_default();
                    t.text = text.clone();
                    t.runs = vec![TextRun { len: t.text.len(), style }];
                    t.paragraphs = vec![ParagraphRun { len: t.text.len(), style: para }];
                    t.sync_summary();
                    // The compositor draws a type layer from its cache: clearing it left the
                    // layer blank until some other edit re-rendered it (#990).
                    crate::type_cmds::refresh(&snapshot, t);
                }
            }
            (VarKind::PixelReplacement { method, align, clip }, VarValue::Pixels(path)) => {
                replace_pixels(doc, def.layer, path, *method, *align, *clip)?;
            }
            _ => {} // value type doesn't match the variable kind: skip
        }
    }
    Ok(())
}

/// Replace a pixel layer's content with an image file, scaled (method) and aligned.
fn replace_pixels(doc: &mut Document, layer: LayerId, path: &str, method: PixelMethod, align: PixelAlign, clip: bool) -> Result<()> {
    let target = doc.layer(layer).and_then(|l| l.surface()).map(|s| s.content_bounds()).filter(|r| !r.is_empty()).unwrap_or_else(|| doc.bounds());
    if target.is_empty() {
        return Ok(());
    }
    let fmt = doc.pixel_format();
    let bytes = std::fs::read(path).map_err(|e| EngineError::Other(format!("read `{path}`: {e}")))?;
    let src = crate::file_cmds::import(path, &bytes)?;
    let img = crate::file_cmds::flattened(&src, fmt);
    let (iw, ih) = (src.size.width.max(1) as f64, src.size.height.max(1) as f64);
    let (tw, th) = (target.width() as f64, target.height() as f64);
    let (sx, sy) = match method {
        PixelMethod::Fit => {
            let k = (tw / iw).min(th / ih);
            (k, k)
        }
        PixelMethod::Fill => {
            let k = (tw / iw).max(th / ih);
            (k, k)
        }
        PixelMethod::Conform => (tw / iw, th / ih),
        PixelMethod::AsIs => (1.0, 1.0),
    };
    let scaled = if (sx - 1.0).abs() > 1e-9 || (sy - 1.0).abs() > 1e-9 { resize_surface(&img, sx, sy, Resample::Bicubic) } else { img };
    let (sw, sh) = (iw * sx, ih * sy);
    // Aligned placement within the target rect.
    let fx = match align {
        PixelAlign::TopLeft | PixelAlign::CenterLeft | PixelAlign::BottomLeft => 0.0,
        PixelAlign::TopRight | PixelAlign::CenterRight | PixelAlign::BottomRight => 1.0,
        _ => 0.5,
    };
    let fy = match align {
        PixelAlign::TopLeft | PixelAlign::TopCenter | PixelAlign::TopRight => 0.0,
        PixelAlign::BottomLeft | PixelAlign::BottomCenter | PixelAlign::BottomRight => 1.0,
        _ => 0.5,
    };
    let dx = (target.x0 as f64 + (tw - sw) * fx).round() as i32;
    let dy = (target.y0 as f64 + (th - sh) * fy).round() as i32;
    let placed = translate_surface(&scaled, dx, dy);
    let placed = if clip { photocraft_algo::resample::crop_surface(&placed, target) } else { placed };
    if let Some(l) = doc.layer_mut(layer)
        && let LayerContent::Raster(sfc) = &mut l.content
    {
        *sfc = placed;
    }
    Ok(())
}

fn apply_data_set(s: &mut Session, p: &Value) -> Result<Value> {
    let (set, idx) = resolve_set(s, p, "image.applyDataSet")?;
    let vars = s.active().ok_or(EngineError::NoDocument)?.doc.variables.clone();
    // Pixel Replacement reads image files, so an untrusted session's gate judges each path.
    if let Some(auth) = s.authorize {
        for dv in &set.values {
            if let VarValue::Pixels(path) = &dv.value {
                auth("image.applyDataSet", &json!({"path": path}))?;
            }
        }
    }
    s.edit(&format!("Apply Data Set \"{}\"", set.name), |doc, _| apply_to_doc(doc, &vars, &set))?;
    set_vars(s, |v| v.active = Some(idx))?;
    Ok(json!({"applied": set.name, "index": idx}))
}

/// Resolve a data set from `{name}` or `{index}`.
fn resolve_set(s: &Session, p: &Value, cmd: &str) -> Result<(DataSet, usize)> {
    let vars = &s.active().ok_or(EngineError::NoDocument)?.doc.variables;
    if let Some(name) = p.get("name").and_then(Value::as_str) {
        let i = vars.data_set_index(name).ok_or_else(|| bad(cmd, format!("no data set `{name}`")))?;
        Ok((vars.data_sets[i].clone(), i))
    } else if let Some(i) = p.get("index").and_then(Value::as_u64) {
        let i = i as usize;
        vars.data_sets.get(i).map(|d| (d.clone(), i)).ok_or_else(|| bad(cmd, format!("no data set at index {i}")))
    } else {
        Err(bad(cmd, "need `name` or `index`"))
    }
}

/// Import data sets from a CSV: the header row holds variable names, each later row is a data set.
/// A value is a boolean (true/false) for a Visibility variable, a path for a Pixel Replacement
/// variable, else text. The first column may be a data-set name (header empty or `DataSet`).
fn import_data_sets(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p.get("path").and_then(Value::as_str).ok_or_else(|| bad("file.import.variableDataSets", "need `path`"))?;
    let text = std::fs::read_to_string(path).map_err(|e| EngineError::Other(format!("read `{path}`: {e}")))?;
    import_data_sets_text(s, &text, p)
}

/// The body of `file.import.variableDataSets` over CSV text already read: automation reads the
/// file beneath its read root and hands the text over (#3216).
pub fn import_data_sets_text(s: &mut Session, text: &str, p: &Value) -> Result<Value> {
    let delim = p.get("delimiter").and_then(Value::as_str).and_then(|d| d.bytes().next()).unwrap_or(b',');
    let vars = s.active().ok_or(EngineError::NoDocument)?.doc.variables.clone();
    let mut records = csv_records(text).filter(|r| !r.trim().is_empty());
    let header: Vec<String> = records.next().map(|h| split_csv(h, delim)).unwrap_or_default();
    if header.is_empty() {
        return Err(bad("file.import.variableDataSets", "empty CSV"));
    }
    let named = header[0].trim().is_empty() || header[0].eq_ignore_ascii_case("dataset");
    let mut sets = Vec::new();
    for (row, record) in records.enumerate() {
        let cells = split_csv(record, delim);
        let name = if named {
            cells.first().cloned().filter(|c| !c.trim().is_empty()).unwrap_or_else(|| format!("Data Set {}", row + 1))
        } else {
            format!("Data Set {}", row + 1)
        };
        let start = if named { 1 } else { 0 };
        let mut values = Vec::new();
        for (ci, var) in header.iter().enumerate().skip(start) {
            let Some(cell) = cells.get(ci) else { continue };
            let Some(def) = vars.def(var) else { continue };
            let value = match def.kind {
                VarKind::Visibility => VarValue::Visibility(matches!(cell.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes" | "visible" | "on")),
                VarKind::PixelReplacement { .. } => VarValue::Pixels(cell.clone()),
                VarKind::TextReplacement => VarValue::Text(cell.clone()),
            };
            values.push(DataValue { variable: var.clone(), value });
        }
        sets.push(DataSet { name, values });
    }
    let n = sets.len();
    set_vars(s, |v| v.data_sets = sets)?;
    Ok(json!({"imported": n}))
}

/// Split logical CSV records without changing line endings inside quoted fields.
fn csv_records(text: &str) -> impl Iterator<Item = &str> {
    let mut chars = text.char_indices();
    let mut start = 0;
    std::iter::from_fn(move || {
        let mut quoted = false;
        for (end, c) in chars.by_ref() {
            if c == '"' {
                // An escaped quote pair toggles twice, leaving the surrounding state intact.
                quoted = !quoted;
            } else if c == '\n' && !quoted {
                let record = text.get(start..end)?;
                start = end.saturating_add(1);
                return Some(record.strip_suffix('\r').unwrap_or(record));
            }
        }
        let record = text.get(start..)?;
        start = text.len();
        (!record.is_empty()).then_some(record)
    })
}

/// Minimal CSV field splitter with double-quote support.
fn split_csv(line: &str, delim: u8) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_q = false;
    let delim = char::from(delim);
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if in_q {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    cur.push('"');
                    let _ = chars.next();
                } else {
                    in_q = false;
                }
            } else {
                cur.push(c);
            }
        } else if c == '"' {
            in_q = true;
        } else if c == delim {
            out.push(std::mem::take(&mut cur));
        } else {
            cur.push(c);
        }
    }
    out.push(cur);
    out
}

/// Apply each data set in turn and export the flattened document, restoring the original after.
/// `{dir, format?("png"), dataSets?[names]}` → `{files:[…],count}`.
fn export_as_files(s: &mut Session, p: &Value) -> Result<Value> {
    let dir = p.get("dir").and_then(Value::as_str).ok_or_else(|| bad("file.export.dataSetsAsFiles", "need `dir`"))?.to_string();
    // The directory appears only once the batch has been planned, as it always did, so a batch
    // rejected for colliding names still leaves no directory behind.
    let mut made_dir = false;
    let names = export_data_sets_with(s, p, |name, bytes| {
        if !made_dir {
            std::fs::create_dir_all(&dir).map_err(|e| EngineError::Other(format!("mkdir `{dir}`: {e}")))?;
            made_dir = true;
        }
        crate::file_cmds::write_file(&format!("{dir}/{name}"), bytes)
    })?;
    let files: Vec<String> = names.iter().map(|name| format!("{dir}/{name}")).collect();
    let count = files.len();
    Ok(json!({"files": files, "count": count}))
}

/// The bodies of `file.export.dataSetsAsFiles`, streamed to `write` as a single-component
/// filename plus its bytes: one flattened file per data set. The whole batch is planned first,
/// so colliding names are rejected before any file is written, and automation writes exactly
/// these bytes beneath its write root (#3216) — the two paths cannot drift. Returns the
/// filenames written, in order.
pub fn export_data_sets_with(s: &Session, p: &Value, mut write: impl FnMut(&str, &[u8]) -> Result<()>) -> Result<Vec<String>> {
    let format = p.get("format").and_then(Value::as_str).unwrap_or("png");
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let original = st.doc.clone();
    let vars = original.variables.clone();
    let wanted: Option<Vec<String>> = p.get("dataSets").and_then(Value::as_array).map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect());
    let sets: Vec<DataSet> = vars.data_sets.iter().filter(|d| wanted.as_ref().is_none_or(|w| w.contains(&d.name))).cloned().collect();
    if sets.is_empty() {
        return Err(bad("file.export.dataSetsAsFiles", "no data sets to export"));
    }
    // Filename template: `{name}` (sanitised data-set name), `{index}` (1-based). Default `{name}`.
    let template = p.get("naming").and_then(Value::as_str).unwrap_or("{name}");
    let doc_stem = original.name.rsplit_once('.').map_or(original.name.as_str(), |(a, _)| a).to_string();
    // Plan the whole batch before writing: sanitised names or a constant template may collide.
    // Match the case-insensitive comparison used by the other batch exporters.
    let mut claimed = std::collections::HashMap::new();
    let mut names = Vec::new();
    for (n, set) in sets.iter().enumerate() {
        let safe: String = set.name.chars().map(|c| if c.is_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
        let stem = template.replace("{name}", &safe).replace("{index}", &format!("{:03}", n + 1)).replace("{document}", &doc_stem);
        let name = format!("{stem}.{format}");
        if let Some(first) = claimed.insert(name.to_lowercase(), &set.name) {
            return Err(bad(
                "file.export.dataSetsAsFiles",
                format!("data sets `{first}` and `{}` resolve to the same output `{name}`; include {{index}} in `naming` to use distinct filenames", set.name),
            ));
        }
        names.push(name);
    }
    let mut written = Vec::new();
    for (set, name) in sets.iter().zip(&names) {
        let mut doc = (*original).clone();
        apply_to_doc(&mut doc, &vars, set)?;
        let opts = photocraft_io::ExportOptions::default();
        let bytes = photocraft_io::export(&doc, format, &opts).map(|r| r.bytes).map_err(|e| EngineError::Other(format!("export `{}`: {e}", set.name)))?;
        write(name, &bytes)?;
        written.push(name.clone());
    }
    Ok(written)
}

fn list(s: &mut Session) -> Result<Value> {
    let v = &s.active().ok_or(EngineError::NoDocument)?.doc.variables;
    Ok(json!({
        "defs": v.defs.iter().map(|d| {
            merged(json!({"name": d.name, "layer": d.layer.0}), kind_json(&d.kind))
        }).collect::<Vec<_>>(),
        "dataSets": v.data_sets.iter().map(|ds| json!({
            "name": ds.name,
            "values": ds.values.iter().map(|dv| {
                merged(json!({"variable": dv.variable}), value_json(&dv.value))
            }).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "active": v.active,
    }))
}

macro_rules! spec {
    ($id:expr, $label:expr, $menu:expr, $params:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: has_doc, journal: true, run: $run }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "image.variables.define",
            "Define…",
            &["Image", "Variables"],
            "{defs:[{name, layer:id, type:visibility|textReplacement|pixelReplacement, method?:fit|fill|asIs|conform, align?, clip?}]} → {defs,dataSets,active}",
            |s, p| define(s, p)
        ),
        spec!(
            "image.variables.dataSets",
            "Data Sets…",
            &["Image", "Variables"],
            "{dataSets:[{name, values:[{variable, kind:visibility|text|pixels, value}]}], append?} → {defs,dataSets,active}",
            |s, p| data_sets(s, p)
        ),
        spec!(
            "image.applyDataSet",
            "Apply Data Set…",
            &["Image"],
            "{name|index} → {applied, index}: sets layer visibility/text/pixels from the data set (one history step)",
            |s, p| apply_data_set(s, p)
        ),
        spec!(
            "file.import.variableDataSets",
            "Variable Data Sets…",
            &["File", "Import"],
            "{path, delimiter?} → {imported}: CSV header = variable names, each row a data set (first column may be the data-set name)",
            |s, p| import_data_sets(s, p)
        ),
        spec!(
            "file.export.dataSetsAsFiles",
            "Data Sets as Files…",
            &["File", "Export"],
            "{dir, format?:png, dataSets?[names], naming?:\"{name}|{index}|{document}\"} → {files,count}: apply each data set and export the flattened document; colliding filenames are rejected before writing (include {index} to disambiguate)",
            |s, p| export_as_files(s, p)
        ),
        CommandSpec {
            id: "variables.list",
            label: "List Variables",
            menu: &[],
            shortcut: None,
            journal: false,
            params: "{} → {defs,dataSets,active}",
            enabled: has_doc,
            run: |s, _| list(s),
        },
    ]
}

#[cfg(test)]
mod tests;
