//! A modal color transaction for a named text style; the canvas preview never edits the session.
use std::sync::Arc;

use photocraft_doc::{Color, DocId, Document};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::{PhotocraftApp, color_picker_ui, state::DialogKind};

const TARGET: &str = "__textStyleColor";

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Kind {
    Character,
    Paragraph,
}

impl Kind {
    fn paragraph(self) -> bool {
        self == Self::Paragraph
    }

    fn command(self) -> &'static str {
        if self.paragraph() { "type.paragraphStyle.set" } else { "type.characterStyle.set" }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Target {
    document: DocId,
    revision: u64,
    kind: Kind,
    id: u32,
    color: String,
}

pub(crate) struct Preview {
    key: u64,
    shown: Result<Arc<Document>, String>,
}

pub(crate) fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(TARGET)
}

fn target(fields: &Map<String, Value>) -> Result<Target, String> {
    serde_json::from_value(fields.get(TARGET).cloned().ok_or("picker has no text style target")?).map_err(|_| "invalid text style color target".into())
}

fn color(doc: &Document, kind: Kind, id: u32) -> Result<Color, String> {
    let attrs = if kind.paragraph() {
        &doc.text_styles.para_style(id).ok_or("no such paragraph style")?.char_attrs
    } else {
        &doc.text_styles.char_style(id).ok_or("no such character style")?.attrs
    };
    serde_json::from_value(attrs.get("color").cloned().ok_or("enable the style's Color attribute first")?).map_err(|_| "invalid text style color".into())
}

fn validate(app: &PhotocraftApp, t: &Target) -> Result<(), String> {
    let st = app.session.active().ok_or("no active document")?;
    if st.doc.id != t.document || st.revision != t.revision {
        return Err("text style color document changed while the picker was open".into());
    }
    let panels = &app.ui.type_panels;
    if !panels.styles || panels.styles_tab != usize::from(t.kind.paragraph()) || panels.options != Some((t.kind.paragraph(), t.id)) {
        return Err("text Style Options closed or changed".into());
    }
    color(&st.doc, t.kind, t.id)?;
    Ok(())
}

/// Validate before changing either the panel state or the dialog stack. Also used by automation.
pub(crate) fn open(app: &mut PhotocraftApp, kind: Kind, id: u32) -> Result<u64, String> {
    if !app.ui.dialogs.is_empty() {
        return Err("close the current dialog before opening a text style color picker".into());
    }
    let st = app.session.active().ok_or("no active document")?;
    let rgb = color(&st.doc, kind, id)?.to_rgb();
    if !rgb.iter().all(|v| v.is_finite()) {
        return Err("invalid text style color".into());
    }
    // Style colors use encoded RGB, just like the existing compact style picker.
    let t = Target { document: st.doc.id, revision: st.revision, kind, id, color: color_picker_ui::hex(rgb) };
    let mut fields = color_picker_ui::initial_fields("dialog", "Color Picker (Text Color)", rgb);
    fields.insert(TARGET.into(), serde_json::to_value(t).map_err(|e| e.to_string())?);
    let panels = &mut app.ui.type_panels;
    panels.styles = true;
    panels.styles_tab = usize::from(kind.paragraph());
    panels.options = Some((kind.paragraph(), id));
    Ok(app.ui.open_dialog(DialogKind::Command, fields))
}

fn chosen(fields: &Map<String, Value>) -> Result<String, String> {
    fields.get("color").and_then(Value::as_str).and_then(color_picker_ui::parse_hex).map(color_picker_ui::hex).ok_or("invalid picker color".into())
}

pub(crate) fn confirm(app: &mut PhotocraftApp, fields: &Map<String, Value>) -> Result<Value, String> {
    let t = target(fields)?;
    validate(app, &t)?;
    let color = chosen(fields)?;
    app.text_style_preview = None;
    if color == t.color {
        return Ok(Value::Null);
    }
    app.run(t.kind.command(), json!({"id": t.id, "attrs": {"color": color}}))
}

pub(crate) fn prune(app: &mut PhotocraftApp) {
    let invalid: Vec<_> =
        app.ui.dialogs.iter().filter(|d| owns(&d.fields) && target(&d.fields).and_then(|t| validate(app, &t)).is_err()).map(|d| d.id).collect();
    app.ui.dialogs.retain(|d| !invalid.contains(&d.id));
    if !app.ui.dialogs.iter().any(|d| owns(&d.fields)) {
        app.text_style_preview = None;
    }
}

/// CPU and GPU canvases use the same isolated document; unchanged frames reuse its Arc.
pub(crate) fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    prune(app);
    let d = app.ui.dialogs.last().filter(|d| owns(&d.fields))?;
    let t = target(&d.fields).ok()?;
    validate(app, &t).ok()?;
    let color = match chosen(&d.fields) {
        Ok(color) => color,
        Err(error) => {
            app.ui.status = error;
            app.ui.status_error = true;
            return None;
        }
    };
    let st = app.session.documents().get(idx)?;
    if st.doc.id != t.document {
        return None;
    }
    if t.color == color {
        app.text_style_preview = None;
        return None;
    }
    let key = (egui::Id::new((t.document.0, t.revision, d.id, t.kind.paragraph(), t.id, &color)).value() & ((1 << 60) - 1)) | (1 << 60);
    if app.text_style_preview.as_ref().map(|p| p.key) != Some(key) {
        let shown = photocraft_engine::type_styles_cmds::preview_options(&st.doc, &json!({"id": t.id, "attrs": {"color": color}}), t.kind.paragraph())
            .map(Arc::new)
            .map_err(|e| e.to_string());
        if let Err(error) = &shown {
            app.ui.status = error.clone();
            app.ui.status_error = true;
        }
        app.text_style_preview = Some(Preview { key, shown });
    }
    app.text_style_preview.as_ref()?.shown.as_ref().ok().map(|doc| (doc.clone(), key))
}

#[cfg(test)]
mod tests;
