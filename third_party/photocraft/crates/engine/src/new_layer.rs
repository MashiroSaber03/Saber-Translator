//! Layer › New › Layer: Photoshop's New Layer options (name, Use Previous Layer to Create Clipping
//! Mask, color label, mode, opacity, and Fill with neutral color), applied in one history step.

use photocraft_doc::{BlendMode, LabelColor, Layer};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

const CMD: &str = "layer.new.layer";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

/// The color that leaves everything below unchanged in `mode` (Photoshop's "Fill with
/// Multiply-neutral color (white)"), as RGB, and its name. `None` for modes that have none.
pub fn neutral_color(mode: BlendMode) -> Option<([f32; 3], &'static str)> {
    use BlendMode::*;
    match mode {
        Darken | Multiply | ColorBurn | LinearBurn | DarkerColor | Divide => Some(([1.0; 3], "white")),
        Lighten | Screen | ColorDodge | LinearDodge | LighterColor | Difference | Exclusion | Subtract => Some(([0.0; 3], "black")),
        Overlay | SoftLight | HardLight | VividLight | LinearLight | PinLight => Some(([0.5; 3], "50% gray")),
        _ => None,
    }
}

/// `layer.new.layer`: a pixel layer above the active one (below it with `position: below`, the
/// Layers panel's ⌘/Ctrl-click), joining the clip group it lands in.
pub fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let flag = |k: &str| match p.get(k) {
        None | Some(Value::Null) => Ok(false),
        Some(v) => v.as_bool().ok_or_else(|| bad(format!("`{k}` must be true or false"))),
    };
    let clip = flag("clipToPrevious")?;
    let fill_neutral = flag("fillNeutral")?;
    let mode = match p.get("mode") {
        None | Some(Value::Null) => BlendMode::Normal,
        Some(v) => {
            let name = v.as_str().ok_or_else(|| bad("`mode` must be a blend mode name"))?;
            crate::commands::blend_from_str(name).filter(|m| *m != BlendMode::PassThrough).ok_or_else(|| bad(format!("unknown layer blend mode `{name}`")))?
        }
    };
    let opacity = match p.get("opacity") {
        None | Some(Value::Null) => 1.0,
        Some(v) => v.as_f64().filter(|o| o.is_finite()).ok_or_else(|| bad("`opacity` must be a number from 0 to 1"))?.clamp(0.0, 1.0) as f32,
    };
    let label = match p.get("color") {
        None | Some(Value::Null) => LabelColor::None,
        Some(v) => v.as_str().and_then(LabelColor::from_id).ok_or_else(|| bad(format!("`color` must be a layer color id, not {v}")))?,
    };
    let neutral =
        if fill_neutral { Some(neutral_color(mode).ok_or_else(|| bad(format!("{} has no neutral color to fill with", mode.label())))?.0) } else { None };
    let below = match p.get("position") {
        None | Some(Value::Null) => false,
        Some(v) => match v.as_str() {
            Some("above") => false,
            Some("below") => true,
            _ => return Err(bad("`position` must be above or below")),
        },
    };
    let name = match p.get("name") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_str().ok_or_else(|| bad("`name` must be a string"))?.to_string()),
    };
    let id = s.edit("New Layer", |doc, active| {
        let name = name.clone().unwrap_or_else(|| doc.next_layer_name("Layer"));
        let mut layer = Layer::raster(name, doc.pixel_format());
        layer.blend = mode;
        layer.opacity = opacity;
        layer.label = label;
        if let Some(rgb) = neutral {
            let bounds = doc.bounds();
            let surface = layer.surface_mut().ok_or_else(|| bad("a new pixel layer has no pixels"))?;
            let px = photocraft_raster::from_rgba(&surface.format(), [rgb[0], rgb[1], rgb[2], 1.0]);
            surface.fill_rect(bounds, &px);
        }
        // Nothing may sit under the Background: below it the layer lands above it instead.
        let id = match *active {
            Some(a) if below && doc.layer(a).is_some() && !doc.is_background(a) => doc.insert_below(Some(a), layer),
            _ => doc.insert_above(*active, layer),
        };
        crate::clip_group::settle(doc, &[id], &Default::default(), clip);
        *active = Some(id);
        Ok(id)
    })?;
    crate::layer_multi_cmds::note_inert_insert(s, id);
    Ok(json!({ "layer": id.0 }))
}

#[cfg(test)]
mod tests {
    use photocraft_doc::{BlendMode, LabelColor};
    use photocraft_geom::Rect;
    use serde_json::json;

    use super::neutral_color;
    use crate::Session;

    /// A document with a busy Background: every pixel a different colour.
    fn busy(mode: &str, depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32, "mode": mode, "depth": depth})).unwrap();
        s.edit("busy", |doc, active| {
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            let fmt = surf.format();
            for y in 0..32 {
                for x in 0..32 {
                    let rgba = [x as f32 / 31.0, y as f32 / 31.0, ((x * 7 + y * 3) % 32) as f32 / 31.0, 1.0];
                    surf.fill_rect(Rect::new(x, y, x + 1, y + 1), &photocraft_raster::from_rgba(&fmt, rgba));
                }
            }
            Ok(())
        })
        .unwrap();
        s
    }

    fn max_diff(a: &photocraft_compose::Buffer, b: &photocraft_compose::Buffer) -> f32 {
        a.px.iter().zip(&b.px).flat_map(|(p, q)| p.iter().zip(q).map(|(x, y)| (x - y).abs())).fold(0.0, f32::max)
    }

    /// Fill with neutral color leaves the composite unchanged in every mode that offers it, at
    /// every depth; the wrong neutral (gray for Multiply) does change it.
    #[test]
    fn neutral_fills_leave_the_image_unchanged() {
        for (mode, depth) in [("rgb", 8), ("rgb", 16), ("rgb", 32), ("gray", 8), ("gray", 16)] {
            let s0 = busy(mode, depth);
            let doc0 = s0.active().unwrap().doc.clone();
            let shown = photocraft_compose::render(&doc0, doc0.bounds());
            for m in BlendMode::LAYER_MODES.into_iter().filter(|m| neutral_color(*m).is_some()) {
                let mut s = busy(mode, depth);
                s.execute("layer.new.layer", json!({"mode": m.label(), "fillNeutral": true})).unwrap();
                let doc = &s.active().unwrap().doc;
                let top = doc.layers.last().unwrap();
                assert_eq!(top.blend, m);
                assert!(top.surface().unwrap().tile_count() > 0, "{m:?} filled");
                let d = max_diff(&photocraft_compose::render(doc, doc.bounds()), &shown);
                // 8-bit stores 50% gray as 128 (0.502), which can move a pixel by one level.
                let tolerance = if depth == 8 { 1.0 / 255.0 + 1e-6 } else { 1e-3 };
                assert!(d <= tolerance, "{mode}/{depth}: {m:?} changed the image by {d}");
            }
            let mut s = busy(mode, depth);
            s.execute("layer.new.layer", json!({"mode": "Multiply"})).unwrap();
            s.execute("edit.fill", json!({"contents": "gray"})).unwrap();
            let doc = &s.active().unwrap().doc;
            assert!(max_diff(&photocraft_compose::render(doc, doc.bounds()), &shown) > 0.1, "{mode}/{depth}: the control changes it");
        }
        for m in [BlendMode::Normal, BlendMode::Dissolve, BlendMode::HardMix, BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity] {
            assert!(neutral_color(m).is_none(), "{m:?}");
        }
    }

    #[test]
    fn options_apply_in_one_step() {
        let mut s = busy("rgb", 8);
        let steps = s.active().unwrap().history.past_len();
        let id =
            s.execute("layer.new.layer", json!({"name": "Shade", "mode": "Overlay", "opacity": 0.4, "color": "blue", "fillNeutral": true})).unwrap()["layer"]
                .as_u64()
                .unwrap();
        let st = s.active().unwrap();
        assert_eq!(st.history.past_len(), steps + 1);
        let l = st.doc.layer(photocraft_doc::LayerId(id)).unwrap();
        assert_eq!((l.name.as_str(), l.blend, l.opacity, l.label), ("Shade", BlendMode::Overlay, 0.4, LabelColor::Blue));
        assert_eq!(st.active_layer, Some(l.id));
        // Defaults: what the Layers panel button makes.
        s.execute("layer.new.layer", json!({})).unwrap();
        let st = s.active().unwrap();
        let l = st.doc.layers.last().unwrap();
        assert_eq!((l.blend, l.opacity, l.label, l.clipped), (BlendMode::Normal, 1.0, LabelColor::None, false));
        assert_eq!(l.surface().unwrap().tile_count(), 0);
    }

    /// `position: below` (⌘/Ctrl-click on the panel button) lands under the active layer; the
    /// Background stays at the bottom, so below it the layer lands above it.
    #[test]
    fn position_below_lands_under_the_active_layer() {
        let mut s = busy("rgb", 8);
        let names = |s: &Session| s.active().unwrap().doc.layers.iter().map(|l| l.name.clone()).collect::<Vec<_>>();
        s.execute("layer.new.layer", json!({"name": "A"})).unwrap();
        let below = s.execute("layer.new.layer", json!({"name": "Below", "position": "below"})).unwrap()["layer"].as_u64().unwrap();
        assert_eq!(names(&s), ["Background", "Below", "A"]);
        assert_eq!(s.active().unwrap().active_layer, Some(photocraft_doc::LayerId(below)));
        s.execute("layer.new.layer", json!({"name": "Above", "position": "above"})).unwrap();
        assert_eq!(names(&s), ["Background", "Below", "Above", "A"]);
        let bg = s.active().unwrap().doc.layers[0].id.0;
        s.execute("layer.select", json!({"layer": bg})).unwrap();
        s.execute("layer.new.layer", json!({"name": "OverBg", "position": "below"})).unwrap();
        assert_eq!(names(&s), ["Background", "OverBg", "Below", "Above", "A"]);
    }

    /// Bad params are errors, never panics, and change nothing.
    #[test]
    fn bad_params_are_errors() {
        let mut s = busy("rgb", 8);
        let n = s.active().unwrap().doc.layers.len();
        for p in [
            json!({"mode": "Pass Through"}),
            json!({"mode": "Sparkle"}),
            json!({"mode": 3}),
            json!({"opacity": "half"}),
            json!({"color": "plaid"}),
            json!({"color": 1}),
            json!({"fillNeutral": true}),
            json!({"fillNeutral": true, "mode": "Hue"}),
            json!({"fillNeutral": "yes"}),
            json!({"clipToPrevious": 1}),
            json!({"name": 7}),
            json!({"position": "sideways"}),
            json!({"position": 1}),
        ] {
            assert!(s.execute("layer.new.layer", p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.active().unwrap().doc.layers.len(), n);
        // Out-of-range opacity is clamped, as layer.setProps does.
        s.execute("layer.new.layer", json!({"opacity": 7})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.last().unwrap().opacity, 1.0);
    }
}
