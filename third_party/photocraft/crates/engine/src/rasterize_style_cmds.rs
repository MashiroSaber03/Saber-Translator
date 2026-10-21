//! Layer › Rasterize › Layer Style (#2245): bake a layer's effects into its pixels.
//!
//! The layer is rendered alone by the compositor (the same path as the canvas) at 100 % opacity
//! in Normal mode, so its effects, fill opacity and masks end up in the pixels; its own opacity,
//! blend mode, knockout and Blend If stay layer properties and apply to the baked pixels. For a
//! Normal layer whose effects blend Normal the document looks the same before and after. An
//! effect in another mode (a Multiply shadow, a Screen glow) is evaluated against the
//! transparent backdrop of the isolated layer, so it can look different afterwards; no single
//! layer of pixels can blend one part Multiply and another Screen.

use photocraft_color::BlendMode;
use photocraft_doc::{Knockout, Layer, LayerContent, LayerId};
use photocraft_raster::{Surface, from_rgba_into};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const ID: &str = "layer.rasterize.layerStyle";

/// Whether Rasterize › Layer Style applies to `l`: visible effects on a layer whose content can
/// become pixels (not a group or an adjustment layer).
pub fn can_bake(l: &Layer) -> bool {
    photocraft_compose::effects::has_effects(l) && !matches!(l.content, LayerContent::Group(_) | LayerContent::Adjustment(_))
}

/// Some selected layer (or the active one) has effects to bake.
fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let ids = d.selected_layers().into_iter().chain(d.active_layer);
    if ids.filter_map(|id| d.doc.layer(id)).any(can_bake) { Ok(()) } else { Err("no selected layer has layer effects to rasterize".into()) }
}

/// Bake one layer (type, shape, fill and Smart Object content is rasterized first, in the same
/// history step). Returns false when the layer has nothing to bake.
fn bake_one(s: &mut Session, id: LayerId, key: &str) -> Result<bool> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    if !d.doc.layer(id).is_some_and(can_bake) {
        return Ok(false);
    }
    if let Some(kind) = crate::extra_cmds::kind_of(s, id) {
        crate::extra_cmds::rasterize_one(s, id, Some(kind), key)?;
    }
    s.coalesce_request = Some(key.to_string());
    let r = s.edit("Rasterize Layer Style", |doc, _| {
        let l = doc.layer(id).cloned().ok_or(EngineError::NoLayer(id))?;
        if !matches!(l.content, LayerContent::Raster(_)) {
            return Err(EngineError::Other("the layer could not be rasterized".into()));
        }
        let canvas = doc.bounds();
        let fmt = doc.pixel_format();
        // The pixels (also those past the canvas) and everything the effects reach from them.
        let area = photocraft_compose::layer_bounds(&l, canvas).inflate(photocraft_compose::effects::margin(&l).max(0));
        let mut alone = l.clone();
        alone.visible = true;
        alone.opacity = 1.0;
        alone.blend = BlendMode::Normal;
        alone.clipped = false;
        // These depend on the layers beneath and stay on the layer, applied to the baked pixels.
        alone.advanced.knockout = Knockout::None;
        alone.blend_if = Default::default();
        alone.excluded_channels = 0;
        let mut data = Vec::new();
        if !area.is_empty() {
            // Render the layer on its own with the document's settings (mode, depth, global
            // light, patterns), then put the real stack back.
            let stack = std::mem::replace(&mut doc.layers, vec![alone]);
            let buf = photocraft_compose::render(doc, area);
            doc.layers = stack;
            let n = fmt.channels();
            data = vec![0.0f32; buf.px.len() * n];
            for (p, out) in buf.px.iter().zip(data.chunks_exact_mut(n)) {
                from_rgba_into(&fmt, *p, out);
            }
        }
        let mut px = Surface::new(fmt);
        if !area.is_empty() {
            px.write_region(area, &data);
        }
        px.prune();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        l.content = LayerContent::Raster(px);
        l.fill_cache = None;
        // Fill opacity, the masks and the effects are in the pixels now.
        l.fill_opacity = 1.0;
        l.mask = None;
        l.vector_mask = None;
        l.effects.items.clear();
        l.effects.psd_raw = None;
        l.effects.reference = None;
        l.effects.enabled = true;
        l.advanced.blend_interior = false;
        l.advanced.layer_mask_hides_effects = false;
        l.advanced.vector_mask_hides_effects = false;
        Ok(())
    });
    s.coalesce_request = None;
    r?;
    Ok(true)
}

/// `{"layer":id?}`: with a layer, just that one; otherwise every selected layer with effects, in
/// one history step.
fn rasterize_layer_style(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = crate::layer_multi_cmds::targets(s, p)?;
    let many = ids.len() > 1;
    let key = crate::extra_cmds::step_key(s, "rasterizeLayerStyle");
    let mut done = Vec::new();
    for id in ids {
        if bake_one(s, id, &key)? {
            done.push(id.0);
        }
    }
    let Some(&first) = done.first() else {
        return Err(EngineError::Other(if many { "none of the selected layers has layer effects" } else { "the layer has no layer effects" }.into()));
    };
    let active = s.active().and_then(|d| d.active_layer).map(|id| id.0).filter(|id| done.contains(id));
    Ok(json!({"layer": active.unwrap_or(first), "layers": done}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: "Layer Style",
        menu: &["Layer", "Rasterize"],
        shortcut: None,
        params: r##"{"layer":id?} (no layer: every selected layer with effects; bakes the effects, fill opacity and masks into the pixels)"##,
        enabled,
        run: rasterize_layer_style,
        journal: true,
    }]
}

#[cfg(test)]
mod tests;
