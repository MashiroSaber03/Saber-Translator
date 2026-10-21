//! Copy and Paste with one channel targeted in the Channels panel (#2307), as in Photoshop.
//!
//! - Copy with a colour channel (Red, Green, Blue, …) or an alpha channel targeted copies that
//!   channel's values as a grayscale clip, instead of the whole layer.
//! - Paste with a colour channel targeted writes the clip's luminosity into that channel of the
//!   active layer, instead of making a new layer. The other channels and the layer's transparency
//!   stay as they are. Pasting into an alpha channel, a layer mask or the Quick Mask goes through
//!   [`crate::edit_cmds::paste_to_target`].
//! - Cut with a colour channel targeted copies that channel and fills the cut area of that channel
//!   only with the background colour (#2698).

use photocraft_color::{ColorMode, PixelFormat};
use photocraft_doc::{Layer, LayerId};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::channel_cmds::ChannelTarget;
use crate::edit_cmds::Clip;
use crate::{EngineError, Result, Session};

/// The colour channel a paste writes into: the Channels panel targets one and the params name no
/// other target.
pub(crate) fn paste_color_target(s: &Session, p: &Value) -> Option<usize> {
    if p.get("target").is_some() {
        return None;
    }
    match s.active()?.channel_view.target {
        ChannelTarget::Color(k) => Some(k),
        _ => None,
    }
}

/// Copy the targeted colour or alpha channel as a grayscale clip, inside the selection (or the
/// whole canvas). `Ok(None)` when no single channel is targeted: Copy copies the layer.
pub(crate) fn copy(s: &mut Session) -> Result<Option<Value>> {
    let Some(d) = s.active() else { return Ok(None) };
    let doc = &d.doc;
    let canvas = doc.bounds();
    // The surface the values come from, and which of its samples is the channel.
    let (src, k) = match d.channel_view.target {
        ChannelTarget::Composite | ChannelTarget::LayerMask | ChannelTarget::VectorMask => return Ok(None),
        ChannelTarget::Color(k) => {
            let id = d.active_layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
            let surf = doc.layer(id).and_then(Layer::surface).ok_or_else(|| EngineError::Other("the active layer has no pixels".into()))?;
            (surf, k)
        }
        ChannelTarget::Alpha(i) => {
            let c = doc.channels.get(i).ok_or_else(|| EngineError::Other(format!("no alpha channel {i} (document has {})", doc.channels.len())))?;
            (&c.surface, 0)
        }
    };
    let n = src.format().channels();
    if k >= n {
        return Err(EngineError::Other(format!("no colour channel {k} in this layer")));
    }
    let sel = doc.selection.as_ref();
    let area = sel.map_or(canvas, |m| m.content_bounds().intersect(&canvas));
    if area.is_empty() {
        return Err(EngineError::Other("Could not copy: the selected area is empty".into()));
    }
    let fmt = PixelFormat::new(ColorMode::Grayscale, doc.pixel_format().sample, true);
    let values = crate::allocation::read_region(src, area, "copying a channel")?;
    let cover = sel.map(|m| crate::allocation::read_region(m, area, "copying a channel selection").map(|v| (v, m.format().channels().max(1)))).transpose()?;
    let len = (values.len() / n).checked_mul(2).ok_or_else(|| EngineError::Other("the channel is too large to copy".into()))?;
    let mut px = crate::allocation::capacity(len, "copying a channel")?;
    for (i, p) in values.chunks_exact(n).enumerate() {
        px.push(p.get(k).copied().unwrap_or(0.0));
        px.push(cover.as_ref().map_or(1.0, |(m, mk)| m.get(i * mk).copied().unwrap_or(0.0)));
    }
    let mut surface = Surface::new(fmt);
    surface.try_write_region(area, &px)?;
    surface.prune();
    let bounds = surface.content_bounds();
    if bounds.is_empty() {
        return Err(EngineError::Other("Could not copy: the selected area is empty".into()));
    }
    s.clipboard = Some(Clip { surface, bounds, layers: None, resolution: Some(doc.resolution_dpi) });
    Ok(Some(json!({"bounds": [bounds.x0, bounds.y0, bounds.width(), bounds.height()], "channel": k})))
}

/// Paste the clip's luminosity into colour channel `k` of the active layer, placed as a paste
/// places it. The clip's transparency, and `limit` (Paste Into / Outside), let the old values show
/// through. One history step.
pub(crate) fn paste(s: &mut Session, p: &Value, k: usize, in_place: bool, limit: Option<&Surface>, label: &str) -> Result<Value> {
    let clip = s.clipboard.clone().ok_or_else(|| EngineError::Other("the clipboard is empty".into()))?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = d.doc.bounds();
    let id = d.active_layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    let (dx, dy) = crate::edit_cmds::paste_offset(&clip, canvas, p, in_place);
    let src = crate::edit_cmds::shifted(&clip.surface, dx, dy);
    let area = src.content_bounds().intersect(&canvas);
    s.edit(label, |doc, _| {
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let n = surf.format().channels();
        if k >= n {
            return Err(EngineError::Other(format!("no colour channel {k} in this layer")));
        }
        if !area.is_empty() {
            let mut rgba = vec![[0.0f32; 4]; area.width() as usize * area.height() as usize];
            src.read_rgba_into(area, &mut rgba);
            let weight = limit.map(|m| (m.read_region(area), m.format().channels().max(1)));
            let mut px = surf.read_region(area);
            for (i, (dst, s)) in px.chunks_exact_mut(n).zip(&rgba).enumerate() {
                let w = s[3] * weight.as_ref().map_or(1.0, |(m, mk)| m.get(i * mk).copied().unwrap_or(0.0));
                let v = photocraft_color::convert::rgb_to_gray([s[0], s[1], s[2]]);
                if let Some(c) = dst.get_mut(k) {
                    *c += (v - *c) * w.clamp(0.0, 1.0);
                }
            }
            surf.write_region(area, &px);
            surf.prune();
        }
        doc.selection = None;
        Ok(())
    })?;
    Ok(json!({"layer": id.0, "channel": k, "offset": [dx, dy]}))
}

/// Cut with colour channel `k` of layer `id` targeted (#2698): copy the channel as [`copy`] does,
/// then fill the selected area (or the canvas) of that channel only with the background colour's
/// value for it, as Photoshop does. The other channels and the layer's transparency stay as they
/// are. One history step.
pub(crate) fn cut(s: &mut Session, id: LayerId, k: usize) -> Result<Value> {
    let r = copy(s)?.ok_or_else(|| EngineError::Other("no colour channel is targeted".into()))?;
    let bg = s.tools.background;
    s.edit("Cut Pixels", |doc, _| crate::channel_cmds::clear_color_channel(doc, id, k, bg))?;
    Ok(r)
}

/// Copy reads only: with an alpha channel targeted it needs no pixel layer.
pub(crate) fn copies_alpha_channel(s: &Session) -> bool {
    s.active().is_some_and(|st| matches!(st.channel_view.target, ChannelTarget::Alpha(i) if i < st.doc.channels.len()))
}

#[cfg(test)]
#[path = "channel_clip/tests.rs"]
mod tests;
