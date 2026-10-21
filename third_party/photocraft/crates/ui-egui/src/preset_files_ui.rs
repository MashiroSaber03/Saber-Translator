//! Preset files opened like documents (File › Open, drag-and-drop, the Brushes panel's
//! Import Brushes…, the Preset Manager's Load…): Photoshop brushes (`.abr`) go to the brush
//! library through `brush.presets.importAbr`, gradients (`.grd`) to the Gradients panel through
//! `gradient.presets.importGrd`, swatches (`.aco`, `.ase`) to the Swatches panel through
//! `swatches.import`, patterns (`.pat`) to the Patterns panel through `pattern.import`, keyboard
//! shortcut sets (`.kys`) to Edit › Keyboard Shortcuts through
//! [`crate::kys_import`], instead of opening as documents.

use serde_json::{Value, json};

use crate::PhotocraftApp;

/// Extensions handled here rather than by the document importer (`.aco`/`.ase` swatches go to
/// the Swatches panel through [`crate::swatches_ui::import_bytes`]).
/// `.psp` is Photoshop's live `Keyboard Shortcuts.psp` (the same XML as a `.kys`); other `.psp`
/// settings files are refused with a clear message.
pub const PRESET_EXTS: &[&str] = &["abr", "grd", "pat", "aco", "ase", "kys", "psp"];

fn ext(name: &str) -> String {
    std::path::Path::new(name).extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

/// Is `name` a preset file this module imports?
pub fn is_preset_file(name: &str) -> bool {
    PRESET_EXTS.contains(&ext(name).as_str())
}

/// Import from a gradient picker through the same preset command as File › Open.
pub(crate) fn import_gradients_dialog(app: &mut PhotocraftApp) {
    let result = app.pick_file_bytes_filtered(&["grd"], |app, name, bytes| {
        if ext(&name) != "grd" {
            return Err(tl!("Choose a gradient file (.grd).").into());
        }
        open(app, &name, &bytes).ok_or_else(|| tl!("Choose a gradient file (.grd).").to_string())??;
        Ok(Value::Null)
    });
    if let Err(error) = result
        && error != crate::file_dialog::CANCELLED
    {
        app.ui.status = error;
        app.ui.status_error = true;
    }
}

/// Command params for the file: a real path keeps the journal small; browser files arrive as bytes.
fn source(name: &str, bytes: &[u8]) -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    if std::path::Path::new(name).is_absolute() && std::path::Path::new(name).is_file() {
        return json!({"path": name});
    }
    let _ = name;
    json!({"data": photocraft_engine::paint::tile::b64_encode(bytes)})
}

/// Import a preset file; `None` when `name` isn't one.
pub fn open(app: &mut PhotocraftApp, name: &str, bytes: &[u8]) -> Option<Result<(), String>> {
    let (cmd, what) = match ext(name).as_str() {
        "abr" => ("brush.presets.importAbr", "brushes"),
        "grd" => ("gradient.presets.importGrd", "gradients"),
        "pat" => ("pattern.import", "patterns"),
        "aco" | "ase" => return Some(crate::swatches_ui::import_bytes(app, name, bytes, false).map(|_| ())),
        "kys" | "psp" => return Some(crate::kys_import::open_bytes(app, name, bytes).map(|_| ())),
        _ => return None,
    };
    let stem = std::path::Path::new(name).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| tl!("Imported").into());
    let mut p = source(name, bytes);
    p["group"] = json!(stem);
    if what == "brushes" {
        p["select"] = json!(true);
    }
    Some(app.run(cmd, p).map(|r| {
        // A background import (#210): `jobs_ui` reports the count when it lands.
        if r.get("pending").and_then(Value::as_bool) == Some(true) {
            app.ui.status_error = false;
            let template = "Importing {what} from {file}…";
            app.ui.status = crate::i18n::fmt(tl!(template), &[("what", tl!(what)), ("file", &stem)]);
            return;
        }
        let n = r.get("count").and_then(Value::as_u64).unwrap_or(0);
        let warnings: Vec<String> = r.get("warnings").and_then(|w| serde_json::from_value(w.clone()).ok()).unwrap_or_default();
        app.ui.status_error = false;
        let (one, other) = match what {
            "brushes" => ("Imported {n} brush from {file}", "Imported {n} brushes from {file}"),
            "patterns" => ("Imported {n} pattern from {file}", "Imported {n} patterns from {file}"),
            _ => ("Imported {n} gradient from {file}", "Imported {n} gradients from {file}"),
        };
        let imported = crate::i18n::fmt(&crate::i18n::trn(crate::i18n::current(), n, one, other), &[("file", &stem)]);
        app.ui.status = match warnings.first() {
            Some(warning) => {
                let template = "{message} (notes: {n}; {warning})";
                crate::i18n::fmt(tl!(template), &[("message", &imported), ("n", &warnings.len().to_string()), ("warning", warning)])
            }
            None => imported,
        };
        if what == "brushes" {
            // Show them: Brush Settings › Brushes tab.
            app.ui.panels.brush_settings = true;
            app.ui.brush_tab = 1;
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognises_preset_files() {
        assert!(is_preset_file("/a/b/Set.ABR"));
        assert!(is_preset_file("sunsets.grd"));
        assert!(is_preset_file("Brand.ACO"));
        assert!(is_preset_file("exchange.ase"));
        assert!(is_preset_file("Debaser-ColourLibrary-Part6.PAT"));
        assert!(!is_preset_file("photo.psd"));
        assert!(!is_preset_file("abr"));
    }
}
