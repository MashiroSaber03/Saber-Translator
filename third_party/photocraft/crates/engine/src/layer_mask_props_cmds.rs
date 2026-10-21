//! Non-destructive pixel-mask properties: density and feather leave the mask pixels intact.

use photocraft_doc::LayerId;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

pub const EDIT: &str = "layer.layerMask.edit";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: EDIT.into(), msg: msg.into() }
}

fn number(p: &Value, name: &str, max: f64) -> Result<Option<f32>> {
    p.get(name)
        .map(|v| {
            v.as_f64()
                .filter(|n| n.is_finite() && (0.0..=max).contains(n))
                .map(|n| n as f32)
                .ok_or_else(|| bad(format!("`{name}` must be a finite number within 0..{max}")))
        })
        .transpose()
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    if !p.is_object() {
        return Err(bad("params must be an object"));
    }
    let density = number(p, "density", 100.0)?.map(|d| d / 100.0);
    let feather = number(p, "feather", 1000.0)?;
    if density.is_none() && feather.is_none() {
        return Err(bad("density or feather is required"));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    if let Some(document) = p.get("document")
        && document.as_u64() != Some(st.doc.id.0)
    {
        return Err(bad("the target document is no longer active"));
    }
    let id = match p.get("layer") {
        Some(v) => LayerId(v.as_u64().ok_or_else(|| bad("`layer` must be an unsigned layer id"))?),
        None => st.active_layer.ok_or_else(|| bad("no layer given and no active layer"))?,
    };
    let layer = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let mask = layer.mask.as_ref().ok_or_else(|| bad("the target layer has no pixel mask"))?;
    let (density, feather) = (density.unwrap_or(mask.density), feather.unwrap_or(mask.feather));
    if (density, feather) != (mask.density, mask.feather) {
        s.edit("Edit Layer Mask", |doc, _| {
            let layer = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            let mask = layer.mask.as_mut().ok_or_else(|| bad("the target layer has no pixel mask"))?;
            (mask.density, mask.feather) = (density, feather);
            Ok(())
        })?;
    }
    Ok(json!({"layer": id.0, "density": density * 100.0, "feather": feather}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: EDIT,
        label: "Edit Layer Mask",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id? (default active),"document":id? (reject a stale dialog),"density":0..100?,"feather":0..1000? (pixels)} → {layer,density,feather}; at least one property required; mask pixels are preserved"##,
        enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
        run: edit,
        journal: true,
    }]
}

#[cfg(test)]
mod tests;
