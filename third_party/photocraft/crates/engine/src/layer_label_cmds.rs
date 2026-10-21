//! Undoable layer colour labels. A group applies the label to its existing descendants,
//! without changing pixels, visibility, locks or the layer selection.

use photocraft_doc::{LabelColor, LayerId};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const ID: &str = "layer.setLabelColor";

fn bad(message: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: message.into() }
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    if st.selected_layers().is_empty() { Err("no layer selected".into()) } else { Ok(()) }
}

fn set_label(s: &mut Session, p: &Value) -> Result<Value> {
    let p = p.as_object().ok_or_else(|| bad("expected an object"))?;
    let color =
        p.get("color").and_then(Value::as_str).and_then(LabelColor::from_id).ok_or_else(|| {
            bad("`color` must be a layer colour id (none, red, orange, yellow, green, seafoam, blue, indigo, magenta, fuchsia, violet, gray)")
        })?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let roots = match p.get("layer") {
        Some(v) => vec![LayerId(v.as_u64().ok_or_else(|| bad("`layer` must be an unsigned integer"))?)],
        None => st.selected_layers(),
    };
    if roots.is_empty() {
        return Err(bad("no layer selected"));
    }
    let walk = st.doc.walk();
    let mut paths = Vec::with_capacity(roots.len());
    for id in roots {
        let (path, _, _) = walk.iter().find(|(_, _, l)| l.id == id).ok_or(EngineError::NoLayer(id))?;
        paths.push(path);
    }
    // A single walk also deduplicates a selected child whose parent is selected.
    let mut changed = false;
    let targets: Vec<_> = walk
        .iter()
        .filter(|(path, _, _)| paths.iter().any(|root| path.starts_with(root)))
        .map(|(path, _, l)| {
            changed |= l.label != color;
            (path.clone(), l.id)
        })
        .collect();
    if changed {
        s.edit("Layer Color", |doc, _| {
            for (path, id) in &targets {
                doc.layer_at_mut(path).ok_or(EngineError::NoLayer(*id))?.label = color;
            }
            Ok(())
        })?;
        if let Some(st) = s.active_mut() {
            // This is saved document metadata; no canvas pixels need recompositing.
            st.last_damage = Some(photocraft_geom::Rect::EMPTY);
        }
    }
    Ok(json!({"color": color.id(), "layers": targets.iter().map(|(_, id)| id.0).collect::<Vec<_>>(), "changed": changed}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Layer Color",
        menu: &[],
        shortcut: None,
        params: r#"{"color":"none|red|orange|yellow|green|seafoam|blue|indigo|magenta|fuchsia|violet|gray","layer":id?} (no layer: selected layers; groups include descendants)"#,
        enabled,
        run: set_label,
        journal: true,
    }]
}

#[cfg(test)]
mod tests;
