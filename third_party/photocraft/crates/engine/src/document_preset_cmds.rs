//! Named New Document configurations. The existing preferences services persist these with the
//! other presets; selecting one only returns settings, and document creation stays in `file.new`.

use photocraft_color::{ColorMode, SampleType};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::presets::{always, bad, req_str};
use crate::{Result, Session};

pub(crate) const MAX_DIMENSION: u32 = 300_000;
// What `file.new` creates must reopen: the `.pcraft` loader's default canvas budget covers it (#2935).
const _: () =
    assert!(photocraft_format::MAX_CANVAS_SIDE >= MAX_DIMENSION && photocraft_format::MAX_CANVAS_PIXELS >= MAX_DIMENSION as u64 * MAX_DIMENSION as u64);
pub(crate) const MAX_RESOLUTION: f64 = 30_000.0;
const MAX_PRESETS: usize = 256;
/// File › New › Recent keeps this many settings, like Photoshop's Recent tab.
pub const MAX_RECENT: usize = 20;

/// The formats accepted by `file.new`, also used when validating saved configurations.
pub(crate) fn color_mode(mode: &str) -> Option<ColorMode> {
    match mode {
        "rgb" => Some(ColorMode::Rgb),
        "gray" | "grayscale" => Some(ColorMode::Grayscale),
        "cmyk" => Some(ColorMode::Cmyk),
        "lab" => Some(ColorMode::Lab),
        _ => None,
    }
}

pub(crate) fn sample_type(depth: u64) -> Option<SampleType> {
    match depth {
        8 => Some(SampleType::U8),
        16 => Some(SampleType::U16),
        32 => Some(SampleType::F32),
        _ => None,
    }
}

/// A saved Background Color must not follow later changes to the toolbox colour.
pub(crate) fn background_color(p: &Value, fallback: [f32; 4], cmd: &str) -> Result<[f32; 4]> {
    let c: [f32; 3] = match p.get("backgroundColor") {
        Some(v) => serde_json::from_value(v.clone()).map_err(|_| bad(cmd, "`backgroundColor` must be [r,g,b] in 0..1"))?,
        None => [fallback[0], fallback[1], fallback[2]],
    };
    if !c.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
        return Err(bad(cmd, "`backgroundColor` must be [r,g,b] in 0..1"));
    }
    Ok([c[0], c[1], c[2], 1.0])
}

/// Validate creation before dispatch can commit a floating selection or add a document.
/// Missing fields (including a null params object) keep the command's documented defaults.
pub(crate) fn validate_new_params(s: &Session, p: &Value) -> Result<()> {
    let cmd = "file.new";
    if p.is_null() {
        return Ok(());
    }
    let fields = p.as_object().ok_or_else(|| bad(cmd, "params must be a JSON object"))?;
    for (key, value) in fields {
        let (valid, expected) = match key.as_str() {
            "width" | "height" | "resolution" => {
                let max = if key == "resolution" { MAX_RESOLUTION } else { f64::from(MAX_DIMENSION) };
                (value.as_f64().is_some_and(|n| n.is_finite() && (1.0..=max).contains(&n)), format!("a number within 1..={max}"))
            }
            "mode" => (value.as_str().and_then(color_mode).is_some(), "rgb, gray, grayscale, cmyk or lab".into()),
            "depth" => (value.as_u64().and_then(sample_type).is_some(), "8, 16 or 32".into()),
            "background" => (
                value.as_str().is_some_and(|v| {
                    matches!(v, "white" | "black" | "transparent" | "backgroundColor")
                        || (v.strip_prefix('#').unwrap_or(v).bytes().all(|b| b.is_ascii_hexdigit()) && crate::commands::parse_hex(v).is_some())
                }),
                "white, black, transparent, backgroundColor or a valid hex colour".into(),
            ),
            "backgroundColor" => {
                background_color(p, s.tools.background, cmd)?;
                continue;
            }
            "name" => (value.is_string(), "a string".into()),
            _ => return Err(bad(cmd, format!("unknown parameter `{key}`"))),
        };
        if !valid {
            return Err(bad(cmd, format!("`{key}` must be {expected}")));
        }
    }
    if p.get("background").and_then(Value::as_str) == Some("backgroundColor") {
        background_color(p, s.tools.background, cmd)?;
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentSettings {
    pub width: u32,
    pub height: u32,
    pub resolution: f32,
    pub mode: String,
    pub depth: u8,
    pub background: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub background_color: Option<[f32; 3]>,
    /// Display units only; width/height are always pixels and resolution is always ppi.
    pub unit: String,
    pub resolution_unit: String,
}

impl DocumentSettings {
    fn validate(&self) -> std::result::Result<(), String> {
        if !(1..=MAX_DIMENSION).contains(&self.width) || !(1..=MAX_DIMENSION).contains(&self.height) {
            return Err(format!("Width and height must be within 1..={MAX_DIMENSION} pixels"));
        }
        if !self.resolution.is_finite() || !(1.0..=MAX_RESOLUTION as f32).contains(&self.resolution) {
            return Err(format!("Resolution must be within 1..={MAX_RESOLUTION} ppi"));
        }
        if color_mode(&self.mode).is_none() || sample_type(self.depth.into()).is_none() {
            return Err("Use RGB, Grayscale, CMYK or Lab at 8, 16 or 32 bits".into());
        }
        if !matches!(self.background.as_str(), "white" | "black" | "transparent" | "backgroundColor") && crate::commands::parse_hex(&self.background).is_none()
        {
            return Err("Unknown background choice or invalid colour".into());
        }
        if self.background == "backgroundColor" && self.background_color.is_none() {
            return Err("A saved Background Color needs its colour value".into());
        }
        if self.background_color.is_some_and(|c| !c.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))) {
            return Err("Background colour components must be within 0..1".into());
        }
        if !matches!(self.unit.as_str(), "px" | "in" | "cm" | "mm" | "pt" | "pica") || !matches!(self.resolution_unit.as_str(), "in" | "cm") {
            return Err("Unknown dimension or resolution display unit".into());
        }
        Ok(())
    }

    fn from_params(s: &Session, p: &Value, cmd: &str) -> Result<Self> {
        let mut fields = p.as_object().cloned().ok_or_else(|| bad(cmd, "`settings` must be an object"))?;
        let defaults = json!({"resolution":72.0,"mode":"rgb","depth":8,"background":"white","unit":"px","resolutionUnit":"in"});
        if let Some(defaults) = defaults.as_object() {
            for (k, v) in defaults {
                fields.entry(k.clone()).or_insert_with(|| v.clone());
            }
        }
        // Like file.new, accept pixel counts sent as floats by a UI, but refuse invalid input
        // rather than saving a different configuration.
        for key in ["width", "height"] {
            let n = fields
                .get(key)
                .and_then(Value::as_f64)
                .filter(|n| n.is_finite() && (1.0..=MAX_DIMENSION as f64).contains(n))
                .ok_or_else(|| bad(cmd, format!("`{key}` must be within 1..={MAX_DIMENSION} pixels")))?;
            fields.insert(key.into(), json!(n.round() as u32));
        }
        if fields.get("mode").and_then(Value::as_str) == Some("grayscale") {
            fields.insert("mode".into(), json!("gray"));
        }
        if fields.get("background").and_then(Value::as_str) == Some("backgroundColor") {
            let c = background_color(&Value::Object(fields.clone()), s.tools.background, cmd)?;
            fields.insert("backgroundColor".into(), json!([c[0], c[1], c[2]]));
        }
        let settings: Self = serde_json::from_value(Value::Object(fields)).map_err(|e| bad(cmd, format!("invalid settings: {e}")))?;
        settings.validate().map_err(|e| bad(cmd, e))?;
        Ok(settings)
    }

    /// Creation params deliberately exclude both the preset name and display-only units.
    pub fn command_params(&self) -> Value {
        let mut p = json!({"width":self.width,"height":self.height,"resolution":self.resolution,
            "mode":self.mode,"depth":self.depth,"background":self.background});
        if let Some(c) = self.background_color {
            p["backgroundColor"] = json!(c);
        }
        p
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DocumentPreset {
    pub name: String,
    pub settings: DocumentSettings,
}

fn checked_name(name: &str) -> std::result::Result<String, String> {
    let name = name.trim();
    if name.is_empty() || name.chars().count() > crate::brush_preset_cmds::MAX_NAME {
        return Err("Preset names must contain 1–255 characters after trimming".into());
    }
    Ok(name.into())
}

/// Read entries independently so a damaged preset cannot discard other presets or preferences.
pub(crate) fn load(value: Value) -> Vec<DocumentPreset> {
    let Some(entries) = value.as_array() else { return Vec::new() };
    let mut presets: Vec<DocumentPreset> = Vec::new();
    for entry in entries.iter().take(MAX_PRESETS) {
        let Ok(mut preset) = serde_json::from_value::<DocumentPreset>(entry.clone()) else { continue };
        let Ok(name) = checked_name(&preset.name) else { continue };
        if preset.settings.validate().is_err() || presets.iter().any(|p| p.name.eq_ignore_ascii_case(&name)) {
            continue;
        }
        preset.name = name;
        if preset.settings.mode == "grayscale" {
            preset.settings.mode = "gray".into();
        }
        presets.push(preset);
    }
    presets
}

/// Read the Recent list like [`load`]: a damaged entry is dropped, the rest kept, newest first.
pub(crate) fn load_recent(value: Value) -> Vec<DocumentSettings> {
    let Some(entries) = value.as_array() else { return Vec::new() };
    let mut recent: Vec<DocumentSettings> = Vec::new();
    for entry in entries {
        let Ok(mut settings) = serde_json::from_value::<DocumentSettings>(entry.clone()) else { continue };
        if settings.mode == "grayscale" {
            settings.mode = "gray".into();
        }
        if settings.validate().is_err() || recent.iter().any(|r| r.command_params() == settings.command_params()) {
            continue;
        }
        recent.push(settings);
        if recent.len() >= MAX_RECENT {
            break;
        }
    }
    recent
}

/// Record settings a document was just created with: first in Recent, an equal older entry
/// (the same document, perhaps shown in other units) removed, the oldest dropped past the cap.
fn add_recent(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "document.recent.add";
    let settings = DocumentSettings::from_params(s, p.get("settings").unwrap_or(&Value::Null), CMD)?;
    let recent = &mut s.presets.recent_documents;
    recent.retain(|r| r.command_params() != settings.command_params());
    recent.insert(0, settings);
    recent.truncate(MAX_RECENT);
    s.presets_changed();
    Ok(json!({"recent":s.presets.recent_documents}))
}

fn save(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "document.presets.save";
    let name = checked_name(req_str(p, "name", CMD)?).map_err(|e| bad(CMD, e))?;
    if s.presets.documents.iter().any(|pr| pr.name.eq_ignore_ascii_case(&name)) {
        return Err(bad(CMD, format!("A preset named \"{name}\" already exists; choose another name")));
    }
    if s.presets.documents.len() >= MAX_PRESETS {
        return Err(bad(CMD, format!("At most {MAX_PRESETS} document presets can be saved")));
    }
    let settings = DocumentSettings::from_params(s, p.get("settings").unwrap_or(&Value::Null), CMD)?;
    let preset = DocumentPreset { name, settings };
    let result = json!({"preset":preset});
    s.presets.documents.push(preset);
    s.presets_changed();
    Ok(result)
}

fn get(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "document.presets.get";
    let name = req_str(p, "name", CMD)?;
    let preset = s.presets.documents.iter().find(|pr| pr.name.eq_ignore_ascii_case(name)).ok_or_else(|| bad(CMD, "No such document preset"))?;
    Ok(json!({"preset":preset,"params":preset.settings.command_params()}))
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "document.presets.delete";
    let name = req_str(p, "name", CMD)?;
    let index = s.presets.documents.iter().position(|pr| pr.name.eq_ignore_ascii_case(name)).ok_or_else(|| bad(CMD, "No such document preset"))?;
    s.presets.documents.remove(index);
    s.presets_changed();
    Ok(json!({"deleted":name}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "document.presets.list",
            label: "List Document Presets",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: always,
            run: |s, _| Ok(json!({"presets":s.presets.documents})),
            journal: false,
        },
        CommandSpec {
            id: "document.presets.get",
            label: "Get Document Preset",
            menu: &[],
            shortcut: None,
            params: r#"{"name":str} -> {preset,params} (params for file.new; no document is created)"#,
            enabled: always,
            run: get,
            journal: false,
        },
        CommandSpec {
            id: "document.presets.save",
            label: "Save Document Preset",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str (trimmed, ASCII case-insensitive unique, 1..255 characters),"settings":{"width":px,"height":px,"resolution":ppi=72,"mode":"rgb|gray|cmyk|lab"="rgb","depth":8|16|32=8,"background":"white|black|transparent|backgroundColor|#rrggbb"="white","backgroundColor":[r,g,b]?,"unit":"px|in|cm|mm|pt|pica"="px","resolutionUnit":"in|cm"="in"}}"##,
            enabled: always,
            run: save,
            journal: true,
        },
        CommandSpec {
            id: "document.presets.delete",
            label: "Delete Document Preset",
            menu: &[],
            shortcut: None,
            params: r#"{"name":str}"#,
            enabled: always,
            run: delete,
            journal: true,
        },
        CommandSpec {
            id: "document.recent.list",
            label: "List Recent Document Settings",
            menu: &[],
            shortcut: None,
            params: "{} -> {recent:[settings]} (newest first; each settings.command params minus unit/resolutionUnit go to file.new)",
            enabled: always,
            run: |s, _| Ok(json!({"recent":s.presets.recent_documents})),
            journal: false,
        },
        CommandSpec {
            id: "document.recent.add",
            label: "Add Recent Document Settings",
            menu: &[],
            shortcut: None,
            params: r#"{"settings":{same fields as document.presets.save}} (File › New records each document it creates; newest first, max 20, duplicates collapsed)"#,
            enabled: always,
            run: add_recent,
            // Bookkeeping for File › New, not an editing step an action should replay.
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> Value {
        json!({"width":16,"height":24,"resolution":254.5,"mode":"lab","depth":16,
            "background":"backgroundColor","backgroundColor":[0.125,0.375,0.625],"unit":"mm","resolutionUnit":"cm"})
    }

    fn saved(s: &mut Session, name: &str, settings: Value) -> Value {
        s.execute("document.presets.save", json!({"name":name,"settings":settings})).unwrap()["preset"].clone()
    }

    fn reload(s: &Session) -> Session {
        let mut next = Session::new();
        next.load_prefs_json(&s.prefs_to_json()).unwrap();
        next
    }

    #[test]
    fn every_field_survives_save_reload_and_file_new() {
        for mode in ["rgb", "gray", "cmyk", "lab"] {
            for depth in [8, 16, 32] {
                let mut s = Session::new();
                let mut config = settings();
                config["mode"] = json!(mode);
                config["depth"] = json!(depth);
                let preset = saved(&mut s, "  Product square  ", config.clone());
                assert_eq!(preset["name"], "Product square");
                assert_eq!(preset["settings"], config);
                let mut next = reload(&s);
                assert_eq!(next.presets.documents, s.presets.documents);
                let rev = next.prefs.rev();
                let selected = next.execute("document.presets.get", json!({"name":"product SQUARE"})).unwrap();
                assert_eq!(selected["preset"], preset);
                assert!(next.documents().is_empty(), "save/get do not create a document");
                assert_eq!(next.prefs.rev(), rev, "queries do not dirty preferences");
                let mut params = selected["params"].clone();
                assert!(params.get("name").is_none());
                params["name"] = json!("Actual document");
                next.execute("file.new", params).unwrap();
                let doc = &next.active().unwrap().doc;
                assert_eq!((doc.size.width, doc.size.height, doc.resolution_dpi), (16, 24, 254.5));
                assert_eq!(doc.name, "Actual document");
                assert_eq!(doc.mode, color_mode(mode).unwrap());
                assert_eq!(doc.depth, sample_type(depth).unwrap());
            }
        }
    }

    #[test]
    fn background_choice_and_captured_colour_are_independent_of_the_toolbox() {
        for (background, expected) in [
            ("white", [1.0, 1.0, 1.0, 1.0]),
            ("black", [0.0, 0.0, 0.0, 1.0]),
            ("transparent", [0.0; 4]),
            ("#ff0000", [1.0, 0.0, 0.0, 1.0]),
            ("backgroundColor", [0.125, 0.375, 0.625, 1.0]),
        ] {
            let mut s = Session::new();
            s.tools.background = [0.125, 0.375, 0.625, 1.0];
            saved(&mut s, "Background", json!({"width":4,"height":4,"background":background,"depth":32}));
            let mut s = reload(&s);
            s.tools.background = [1.0, 0.0, 1.0, 1.0];
            let p = s.execute("document.presets.get", json!({"name":"Background"})).unwrap();
            assert_eq!(p["preset"]["settings"]["background"], background);
            s.execute("file.new", p["params"].clone()).unwrap();
            let pixel = s.execute("document.pixel", json!({"x":0,"y":0})).unwrap();
            let rgba = pixel.as_array().unwrap();
            for (actual, expected) in rgba.iter().zip(expected) {
                assert!((actual.as_f64().unwrap() - expected).abs() < 0.001, "{background}: {pixel}");
            }
        }
    }

    #[test]
    fn deletion_marks_preferences_dirty_and_survives_reload() {
        let mut s = Session::new();
        saved(&mut s, "Delete me", settings());
        saved(&mut s, "Keep me", settings());
        let mut s = reload(&s);
        let rev = s.prefs.rev();
        s.execute("document.presets.delete", json!({"name":" delete ME "})).unwrap();
        assert!(s.prefs.rev() > rev);
        let mut s = reload(&s);
        assert_eq!(s.execute("document.presets.list", json!({})).unwrap()["presets"].as_array().unwrap().len(), 1);
        assert_eq!(s.presets.documents[0].name, "Keep me");
        assert!(s.execute("document.presets.get", json!({"name":"Delete me"})).is_err());
        assert!(s.execute("document.presets.delete", json!({"name":"Delete me"})).is_err());
    }

    #[test]
    fn old_preferences_and_malformed_presets_keep_unrelated_settings() {
        let mut original = Session::new();
        original.edit_prefs(|p| p.general.beep_when_done = true);
        original.presets.tool_presets[0].name = "Unrelated tool preset".into();
        let valid = saved(&mut original, "Keep me", settings());
        for invalid in [json!("bad collection"), json!([false, {}, {"name":"broken","settings":{}}])] {
            let mut value = original.prefs_value();
            value["presets"]["documents"] = invalid;
            let mut s = Session::new();
            s.load_prefs_json(&value.to_string()).unwrap();
            assert!(s.prefs().general.beep_when_done);
            assert_eq!(s.presets.tool_presets[0].name, "Unrelated tool preset");
            assert!(s.presets.documents.is_empty());
        }
        let mut value = original.prefs_value();
        let mut bad = valid.clone();
        bad["settings"]["width"] = json!(0);
        value["presets"]["documents"] = json!([bad, valid, valid]);
        let mut s = Session::new();
        s.load_prefs_json(&value.to_string()).unwrap();
        assert_eq!(s.presets.documents.len(), 1, "keep valid entries, discard bad/duplicate entries");
        assert_eq!(s.presets.documents[0].name, "Keep me");
        for without_presets in [false, true] {
            let mut value = original.prefs_value();
            if without_presets {
                value.as_object_mut().unwrap().remove("presets");
            } else {
                value["presets"].as_object_mut().unwrap().remove("documents");
            }
            let mut old = Session::new();
            old.load_prefs_json(&value.to_string()).unwrap();
            assert!(old.presets.documents.is_empty());
            assert!(old.prefs().general.beep_when_done);
        }
    }

    #[test]
    fn bad_input_and_duplicate_names_never_replace_or_mutate_a_snapshot() {
        let mut s = Session::new();
        let mut config = settings();
        saved(&mut s, "Keep me", config.clone());
        config["width"] = json!(99);
        assert_eq!(s.presets.documents[0].settings.width, 16);
        let before = s.prefs_value();
        let rev = s.prefs.rev();
        for name in ["", " \t\n ", "Keep me", " keep ME ", &"x".repeat(256)] {
            assert!(s.execute("document.presets.save", json!({"name":name,"settings":config})).is_err());
        }
        for (key, bad) in [
            ("width", json!(0)),
            ("height", json!(-1)),
            ("width", json!(300_001)),
            ("width", json!("wrong")),
            ("resolution", json!(0)),
            ("resolution", json!(30_001)),
            ("resolution", Value::Null),
            ("mode", json!("indexed")),
            ("depth", json!(64)),
            ("depth", json!("16")),
            ("background", json!("invalid")),
            ("backgroundColor", json!([1, 0])),
            ("backgroundColor", json!([2, 0, 0])),
            ("unit", json!("yards")),
            ("resolutionUnit", json!("px")),
        ] {
            let mut config = settings();
            config[key] = bad;
            assert!(s.execute("document.presets.save", json!({"name":"Bad","settings":config})).is_err(), "{key}");
        }
        for p in [json!({}), Value::Null, json!({"name":12}), json!({"name":"Bad","settings":[]})] {
            assert!(s.execute("document.presets.save", p).is_err());
        }
        assert_eq!(s.prefs_value(), before);
        assert_eq!(s.prefs.rev(), rev);
    }

    #[test]
    fn recent_lists_created_settings_newest_first_capped_and_survives_reload() {
        let mut s = Session::new();
        let add = |s: &mut Session, w: u32, unit: &str| s.execute("document.recent.add", json!({"settings":{"width":w,"height":10,"unit":unit}}));
        for w in 1..=MAX_RECENT as u32 + 5 {
            add(&mut s, w, "px").unwrap();
        }
        // The same document again (shown in other units) moves to the front instead of repeating.
        add(&mut s, 10, "mm").unwrap();
        let mut s = reload(&s);
        let recent = s.execute("document.recent.list", json!({})).unwrap()["recent"].clone();
        let widths: Vec<u64> = recent.as_array().unwrap().iter().map(|r| r["width"].as_u64().unwrap()).collect();
        assert_eq!(widths.len(), MAX_RECENT);
        assert_eq!(&widths[..3], &[10, 25, 24]);
        assert_eq!(widths.iter().filter(|w| **w == 10).count(), 1);
        assert_eq!(recent[0]["unit"], "mm");
        // Bad settings change nothing; damaged stored entries are dropped, the rest kept.
        let rev = s.prefs.rev();
        for bad in [json!({}), json!({"settings":{"width":0,"height":1}}), json!({"settings":"x"}), Value::Null] {
            assert!(s.execute("document.recent.add", bad).is_err());
        }
        assert_eq!(s.prefs.rev(), rev);
        let mut value = s.prefs_value();
        value["presets"]["recent_documents"] = json!([{"width":-1}, recent[1], "junk", recent[1], recent[2]]);
        let mut next = Session::new();
        next.load_prefs_json(&value.to_string()).unwrap();
        assert_eq!(next.presets.recent_documents.iter().map(|r| r.width).collect::<Vec<_>>(), [25, 24]);
        assert!(next.documents().is_empty(), "recording never creates a document");
    }

    #[test]
    fn invalid_explicit_background_colour_does_not_create_a_document() {
        let mut s = Session::new();
        for color in [Value::Null, json!([1, 2]), json!([1, -1, 0]), json!("red")] {
            assert!(s.execute("file.new", json!({"background":"backgroundColor","backgroundColor":color})).is_err());
            assert!(s.documents().is_empty());
        }
    }

    /// What `file.new` can create, `.pcraft` can reopen: the loader's default canvas budget
    /// covers the largest new document (#2935), and such a document round-trips.
    #[test]
    fn the_largest_new_document_reopens_from_pcraft() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": MAX_DIMENSION, "height": MAX_DIMENSION, "background": "transparent"})).unwrap();
        let doc = &s.active().unwrap().doc;
        let bytes = photocraft_format::save_to_bytes(doc, &Default::default()).unwrap();
        let back = photocraft_format::load_from_bytes(&bytes).unwrap();
        assert_eq!(back.size, photocraft_geom::Size::new(MAX_DIMENSION, MAX_DIMENSION));
    }
}
