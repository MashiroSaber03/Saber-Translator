//! View › layer mask (#196): ⌥-click a layer-mask thumbnail shows the mask as a grayscale image
//! instead of the composite; ⇧⌥-click (or the eye on the Channels panel's temporary mask row)
//! shows it as a rubylith overlay over the composite.
//!
//! This is view state on [`crate::channel_cmds::ChannelView::layer_mask`], like the Channels
//! panel's eyes: no history step, a clean document stays clean, and it never changes a pixel.
//! While a mask is shown, pixel commands without an explicit `target` edit that mask
//! ([`crate::channel_cmds`]'s target injection), so painting in mask view paints the mask.
//! The view ends when its layer stops being the active layer or loses its mask ([`fix`]).

use photocraft_doc::LayerId;
use serde::Serialize;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{DocState, EngineError, Result, Session};

pub const ID: &str = "view.layerMask";

/// `\` (#1765): show or hide the active layer's mask as a rubylith, Photoshop's key for it.
pub const OVERLAY_ID: &str = "layer.toggleMaskOverlay";

/// How a shown layer mask is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum MaskViewMode {
    /// The mask alone, as a grayscale image (⌥-click).
    Gray,
    /// The composite with the mask as a red overlay over hidden areas (⇧⌥-click, `\`).
    Overlay,
}

/// Which layer's mask is shown, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct LayerMaskView {
    pub layer: LayerId,
    pub mode: MaskViewMode,
}

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// The mask view of `st` when it is live: its layer is the active layer and still has a mask.
pub fn current(st: &DocState) -> Option<LayerMaskView> {
    let v = st.channel_view.layer_mask?;
    let live = st.active_layer == Some(v.layer) && st.doc.layer(v.layer).is_some_and(|l| l.mask.is_some());
    live.then_some(v)
}

/// Drop a mask view that no longer applies (another layer became active, the layer or its mask
/// was deleted, undo).
pub(crate) fn fix(st: &mut DocState) {
    if st.channel_view.layer_mask.is_some() && current(st).is_none() {
        st.channel_view.layer_mask = None;
    }
}

fn mode_name(v: Option<LayerMaskView>) -> &'static str {
    match v.map(|v| v.mode) {
        None => "off",
        Some(MaskViewMode::Gray) => "gray",
        Some(MaskViewMode::Overlay) => "overlay",
    }
}

/// The view as JSON (`{"layer","mode"}`, or `null` when off); part of `document.inspect`.
pub fn view_json(st: &DocState) -> Value {
    match current(st) {
        Some(v) => json!({"layer": v.layer.0, "mode": mode_name(Some(v))}),
        None => Value::Null,
    }
}

/// `{"layer":id?,"mode":"off"|"gray"|"overlay"|"toggleGray"|"toggleOverlay"="toggleGray"}`.
fn set_view(s: &mut Session, p: &Value) -> Result<Value> {
    let mode = match p.get("mode") {
        None | Some(Value::Null) => "toggleGray",
        Some(v) => v.as_str().ok_or_else(|| bad("`mode` must be a string"))?,
    };
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let layer = match p.get("layer") {
        None | Some(Value::Null) => st.active_layer.ok_or_else(|| bad("no layer given and no active layer"))?,
        Some(v) => LayerId(v.as_u64().ok_or_else(|| bad("`layer` must be a layer id"))?),
    };
    let l = st.doc.layer(layer).ok_or(EngineError::NoLayer(layer))?;
    let has_mask = l.mask.is_some();
    let name = l.name.clone();
    let shown = current(st).filter(|v| v.layer == layer).map(|v| v.mode);
    let want = match mode {
        "off" => None,
        "gray" => Some(MaskViewMode::Gray),
        "overlay" => Some(MaskViewMode::Overlay),
        "toggleGray" => (shown != Some(MaskViewMode::Gray)).then_some(MaskViewMode::Gray),
        "toggleOverlay" => (shown != Some(MaskViewMode::Overlay)).then_some(MaskViewMode::Overlay),
        other => return Err(bad(format!("unknown mode `{other}` (off|gray|overlay|toggleGray|toggleOverlay)"))),
    };
    if want.is_some() && !has_mask {
        return Err(bad(format!("layer \"{name}\" has no layer mask")));
    }
    // ⌥-clicking another layer's mask selects that layer, as in Photoshop.
    if want.is_some() && s.active().and_then(|st| st.active_layer) != Some(layer) {
        s.select_layer(layer)?;
    }
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let next = want.map(|mode| LayerMaskView { layer, mode });
    // "off" for a layer whose mask isn't the one shown leaves the other view alone.
    let applies = next.is_some() || st.channel_view.layer_mask.is_some_and(|v| v.layer == layer);
    if applies && st.channel_view.layer_mask != next {
        st.channel_view.layer_mask = next;
        // Hiding the mask while every colour channel is off would leave a blank canvas.
        crate::channel_cmds::fix_view(st);
        // A view change: redraw without recompositing, and keep a clean document clean.
        let clean = st.saved_revision == st.revision;
        st.revision += 1;
        if clean {
            st.saved_revision = st.revision;
        }
        st.last_damage = Some(photocraft_geom::Rect::EMPTY);
    }
    let v = current(st);
    Ok(json!({"layer": layer.0, "mode": mode_name(v.filter(|v| v.layer == layer))}))
}

/// `\` needs a layer mask on the active layer; without one Photoshop's key does nothing.
fn active_has_mask(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    let layer = st.active_layer.and_then(|id| st.doc.layer(id)).ok_or("no active layer")?;
    if layer.mask.is_some() { Ok(()) } else { Err(format!("layer \"{}\" has no layer mask", layer.name)) }
}

/// `{"layer":id?}`: [`ID`] with mode `toggleOverlay` (a `mode` param is ignored).
fn toggle_overlay(s: &mut Session, p: &Value) -> Result<Value> {
    let layer = match p {
        Value::Object(m) => m.get("layer").cloned().unwrap_or(Value::Null),
        Value::Null => Value::Null,
        _ => return Err(EngineError::BadParams { cmd: OVERLAY_ID.into(), msg: "expected an object".into() }),
    };
    set_view(s, &json!({"layer": layer, "mode": "toggleOverlay"}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: ID,
            label: "View Layer Mask",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"mode":"off"|"gray"|"overlay"|"toggleGray"|"toggleOverlay"="toggleGray"} (gray: the mask alone, ⌥-click its thumbnail; overlay: red rubylith over the composite, ⇧⌥-click or `\`; view state, not an undo step; painting while shown paints the mask)"##,
            enabled: has_doc,
            run: set_view,
            journal: false,
        },
        CommandSpec {
            id: OVERLAY_ID,
            label: "Toggle Layer Mask Overlay",
            menu: &[],
            shortcut: Some("\\"),
            params: r##"{"layer":id?} (shows or hides the active layer's mask as a 50% red rubylith over the composite, the `\` key; view state, not an undo step; painting while shown paints the mask)"##,
            enabled: active_has_mask,
            run: toggle_overlay,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests;
