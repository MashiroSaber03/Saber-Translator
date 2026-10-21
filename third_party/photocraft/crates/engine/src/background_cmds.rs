//! Layer › New › Background from Layer, and the rule that keeps the Background at the bottom.
//!
//! Photoshop's Background is the locked, opaque layer at the bottom of the stack. Background from
//! Layer turns the selected layer into it: transparent pixels take the background colour and the
//! layer drops to the bottom. While a document has a Background nothing reorders it and no layer
//! can go below it (Layer › Arrange, Reverse, dragging in the Layers panel); Layer from Background
//! (`layer.new.layerFromBackground`) turns it back into a normal layer that moves freely.

use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.new.backgroundFromLayer";

/// The document's Background layer, if it has one (always the bottom layer).
pub(crate) fn background_id(doc: &Document) -> Option<LayerId> {
    doc.background().map(|l| l.id)
}

/// Refuses an edit that moved the Background `bg` (taken before the edit) off the bottom of the
/// stack, or put another layer below it. Called at the end of a `Session::edit` closure, so an
/// `Err` leaves the document and history untouched.
pub(crate) fn keep_background_at_bottom(bg: Option<LayerId>, doc: &Document) -> Result<()> {
    match bg {
        Some(id) if doc.layers.first().map(|l| l.id) != Some(id) => Err(EngineError::Other(
            "the Background layer stays at the bottom of the stack (Layer › New › Layer from Background makes it a normal layer)".into(),
        )),
        _ => Ok(()),
    }
}

/// Can `l` become the Background? Groups and adjustment layers have no pixels of their own.
fn convertible(l: &Layer) -> std::result::Result<(), String> {
    match l.content {
        LayerContent::Group(_) | LayerContent::Adjustment(_) => {
            Err(format!("{} {} layer can't become the Background", l.content.article(), l.content.kind_name()))
        }
        _ => Ok(()),
    }
}

fn can_convert(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if background_id(&d.doc).is_some() {
        return Err("the document already has a Background layer".into());
    }
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    convertible(l)
}

fn background_from_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let id = match p.get("layer") {
        None | Some(Value::Null) => crate::commands::layer_param(s, p)?,
        Some(v) => LayerId(v.as_u64().ok_or_else(|| EngineError::BadParams { cmd: CMD.into(), msg: "`layer` must be a layer id".into() })?),
    };
    let bg = s.tools.background;
    s.edit("Background From Layer", |doc, active| {
        if background_id(doc).is_some() {
            return Err(EngineError::Other("the document already has a Background layer".into()));
        }
        let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        convertible(l).map_err(EngineError::Other)?;
        // Render the layer alone, as it looks (mask, opacity and effects applied), over the
        // background colour: the Background is opaque and plain.
        let mut only = l.clone();
        only.visible = true;
        only.clipped = false;
        let visible = l.visible;
        let fmt = doc.pixel_format();
        let rest = std::mem::replace(&mut doc.layers, vec![only]);
        let pixels = photocraft_compose::flatten_to_surface(doc, fmt, Some([bg[0], bg[1], bg[2]]));
        doc.layers = rest;
        let mut out = Layer::raster("Background", fmt);
        out.id = id;
        out.visible = visible;
        out.locks.transparency = true;
        out.locks.position = true;
        *crate::pixels_mut(&mut out)? = pixels;
        doc.remove(id).ok_or(EngineError::NoLayer(id))?;
        doc.layers.insert(0, out);
        *active = Some(id);
        Ok(())
    })?;
    // The Background is the one targeted layer now.
    s.select_layer(id)?;
    Ok(json!({"layer": id.0}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Background From Layer",
        menu: &["Layer", "New"],
        shortcut: None,
        params: r##"{"layer":id?} (the active layer by default; transparent pixels take the background colour and the layer drops to the bottom as the locked Background)"##,
        enabled: can_convert,
        journal: true,
        run: background_from_layer,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_geom::Rect;

    /// Bottom to top: Layer 0 (the old Background, now a normal layer), A, B. A has a 4×4 red
    /// square at (2, 2); the rest of it is transparent.
    fn session(depth: u32) -> (Session, [LayerId; 3]) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 12, "depth": depth})).unwrap();
        s.execute("layer.new.layerFromBackground", json!({})).unwrap();
        let l0 = doc(&s).layers[0].id;
        let id = |s: &mut Session, name: &str| LayerId(s.execute("layer.new.layer", json!({"name": name})).unwrap()["layer"].as_u64().unwrap());
        let a = id(&mut s, "A");
        s.execute("select.rect", json!({"x": 2, "y": 2, "width": 4, "height": 4})).unwrap();
        s.execute("edit.fill", json!({"contents": "color", "color": "#ff0000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let b = id(&mut s, "B");
        (s, [l0, a, b])
    }

    fn doc(s: &Session) -> &Document {
        &s.active().unwrap().doc
    }

    fn order(s: &Session) -> Vec<LayerId> {
        doc(s).layers.iter().map(|l| l.id).collect()
    }

    fn pixel(s: &Session, x: i32, y: i32) -> Vec<f32> {
        doc(s).layers[0].surface().unwrap().read_region(Rect::new(x, y, x + 1, y + 1))
    }

    /// A Background (the white one `file.new` makes) with A and B above it.
    fn with_background() -> (Session, [LayerId; 3]) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 12})).unwrap();
        let bg = doc(&s).layers[0].id;
        let a = LayerId(s.execute("layer.new.layer", json!({"name": "A"})).unwrap()["layer"].as_u64().unwrap());
        let b = LayerId(s.execute("layer.new.layer", json!({"name": "B"})).unwrap()["layer"].as_u64().unwrap());
        (s, [bg, a, b])
    }

    #[test]
    fn opacity_fill_or_a_blend_mode_turn_the_background_into_a_layer() {
        // Like Photoshop: the Background takes none of them, so it becomes "Layer 0" first.
        for (key, value) in [("opacity", json!(0.5)), ("fill", json!(0.5)), ("blend", json!("Multiply"))] {
            let (mut s, [bg, ..]) = with_background();
            s.execute("layer.setProps", json!({"layer": bg.0, key: value})).unwrap();
            let l = doc(&s).layer(bg).unwrap();
            assert!(!doc(&s).is_background(bg) && l.name == "Layer 0", "{key}");
            assert!(l.opacity < 1.0 || l.fill_opacity < 1.0 || l.blend == photocraft_color::BlendMode::Multiply, "{key}: applied");
        }
        // Values that leave its look alone keep it the Background.
        let (mut s, [bg, ..]) = with_background();
        s.execute("layer.setProps", json!({"layer": bg.0, "opacity": 1.0, "fill": 1.0, "blend": "Normal", "visible": false})).unwrap();
        assert!(doc(&s).is_background(bg));
    }

    #[test]
    fn select_and_mask_output_to_a_layer_mask_turns_the_background_into_a_layer() {
        let (mut s, [bg, ..]) = with_background();
        s.execute("layer.select", json!({"layer": bg.0})).unwrap();
        s.execute("select.rect", json!({"x": 2, "y": 2, "width": 6, "height": 6})).unwrap();
        s.execute("select.refineEdge", json!({"output": "layerMask"})).unwrap();
        let l = doc(&s).layer(bg).unwrap();
        assert!(l.mask.is_some() && !doc(&s).is_background(bg) && l.name == "Layer 0");
    }

    #[test]
    fn makes_the_layer_the_background_at_the_bottom() {
        for depth in [8, 16, 32] {
            let (mut s, [l0, a, b]) = session(depth);
            s.tools.background = [0.0, 0.0, 1.0, 1.0];
            s.execute("layer.select", json!({"layer": a.0})).unwrap();
            assert!(s.is_enabled(CMD), "{depth}");
            assert_eq!(s.execute(CMD, json!({})).unwrap()["layer"], a.0);
            assert_eq!(order(&s), [a, l0, b], "{depth}: drops to the bottom");
            let l = &doc(&s).layers[0];
            assert!(doc(&s).is_background(l.id) && l.name == "Background", "{depth}");
            assert_eq!(s.active().unwrap().active_layer, Some(a));
            // Its pixels stay; transparency takes the background colour (blue).
            let red = pixel(&s, 3, 3);
            let blue = pixel(&s, 10, 10);
            assert!((red[0] - 1.0).abs() < 1e-3 && red[2].abs() < 1e-3 && (red[3] - 1.0).abs() < 1e-3, "{depth}: {red:?}");
            assert!(blue[0].abs() < 1e-3 && (blue[2] - 1.0).abs() < 1e-3 && (blue[3] - 1.0).abs() < 1e-3, "{depth}: {blue:?}");
            // Now there is a Background: the item greys, Layer from Background is back.
            assert!(!s.is_enabled(CMD));
            assert!(s.execute(CMD, json!({"layer": b.0})).is_err(), "{depth}: one Background per document");
            assert!(s.is_enabled("layer.new.layerFromBackground"));
            // One step: undo restores the layer where it was, transparent.
            assert!(s.undo());
            assert_eq!(order(&s), [l0, a, b], "{depth}");
            let l = doc(&s).layer(a).unwrap();
            assert!(!doc(&s).is_background(l.id) && l.name == "A");
            assert!(s.redo());
            assert_eq!(order(&s), [a, l0, b], "{depth}");
            assert!(doc(&s).is_background(doc(&s).layers[0].id));
        }
    }

    #[test]
    fn a_layer_in_a_group_leaves_it_and_keeps_its_look() {
        let (mut s, [l0, a, b]) = session(8);
        s.execute("layer.select", json!({"layer": b.0})).unwrap();
        s.execute("layer.groupLayers", json!({})).unwrap();
        s.execute("layer.setProps", json!({"layer": b.0, "opacity": 0.5, "visible": false})).unwrap();
        s.execute("layer.select", json!({"layer": b.0})).unwrap();
        s.execute(CMD, json!({})).unwrap();
        let d = doc(&s);
        assert_eq!(d.layers[0].id, b);
        assert!(d.layers[1..].iter().all(|l| l.id != b));
        assert_eq!(d.layers[1].id, l0);
        assert_eq!(d.layers[2].id, a);
        let bg = &d.layers[0];
        assert!(d.is_background(bg.id) && !bg.visible, "visibility is kept");
        assert_eq!(bg.opacity, 1.0);
    }

    #[test]
    fn refuses_groups_adjustments_missing_layers_and_bad_params() {
        // No document.
        let mut s = Session::new();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());
        // Already a Background: unavailable, for any layer.
        let (mut s, [bg, a, _]) = with_background();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({"layer": a.0})).is_err());
        assert!(s.execute(CMD, json!({"layer": bg.0})).is_err());
        // Bad params.
        let (mut s, [_, a, _]) = session(8);
        for p in [json!({"layer": "A"}), json!({"layer": -1}), json!({"layer": 1.5}), json!({"layer": u64::MAX}), json!({"layer": [a.0]})] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        assert!(doc(&s).background().is_none());
        // A group or an adjustment layer can't become the Background.
        s.execute("layer.select", json!({"layer": a.0})).unwrap();
        s.execute("layer.groupLayers", json!({})).unwrap();
        assert!(!s.is_enabled(CMD), "the group is active");
        assert!(s.execute(CMD, json!({})).is_err());
        s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());
        assert!(doc(&s).background().is_none());
    }

    #[test]
    fn the_background_stays_at_the_bottom() {
        let (mut s, [bg, a, b]) = with_background();
        // Arrange: the Background doesn't move, and nothing goes below it. (The menu greys these
        // for the Background in the shell's enable rules; here an agent asks anyway.)
        for cmd in ["layer.arrange.bringForward", "layer.arrange.bringToFront", "layer.arrange.sendBackward", "layer.arrange.sendToBack"] {
            assert!(s.execute(cmd, json!({"layer": bg.0})).is_err(), "{cmd}");
            assert_eq!(order(&s), [bg, a, b], "{cmd}");
        }
        assert!(s.execute("layer.arrange.sendBackward", json!({"layer": a.0})).is_err());
        s.execute("layer.arrange.sendToBack", json!({"layer": b.0})).unwrap();
        assert_eq!(order(&s), [bg, b, a], "Send to Back stops above the Background");
        // Dragging in the Layers panel (layer.moveTo).
        assert!(s.execute("layer.moveTo", json!({"layer": bg.0, "target": a.0, "position": "above"})).is_err());
        assert!(s.execute("layer.moveTo", json!({"layer": a.0, "target": bg.0, "position": "below"})).is_err());
        assert!(s.execute("layer.moveTo", json!({"layer": a.0, "target": bg.0, "position": "below", "copy": true})).is_err());
        assert!(s.execute("layer.moveTo", json!({"layers": [bg.0, a.0], "target": b.0, "position": "above"})).is_err());
        assert_eq!(order(&s), [bg, b, a]);
        s.execute("layer.moveTo", json!({"layer": a.0, "target": bg.0, "position": "above"})).unwrap();
        assert_eq!(order(&s), [bg, a, b], "above the Background is fine");
        // ⌥-dragging the Background copies it above; the copy is a normal layer.
        s.execute("layer.moveTo", json!({"layer": bg.0, "target": b.0, "position": "above", "copy": true})).unwrap();
        assert_eq!(doc(&s).layers.len(), 4);
        assert_eq!(doc(&s).layers[0].id, bg);
        assert!(s.undo());
        // Reverse with the Background among the selected layers.
        s.execute("layer.select", json!({"layer": bg.0})).unwrap();
        s.execute("layer.select", json!({"layer": b.0, "mode": "add"})).unwrap();
        assert!(s.execute("layer.arrange.reverse", json!({})).is_err());
        assert_eq!(order(&s), [bg, a, b]);
        // As a normal layer it moves freely again.
        s.execute("layer.new.layerFromBackground", json!({})).unwrap();
        s.execute("layer.arrange.bringToFront", json!({"layer": bg.0})).unwrap();
        assert_eq!(order(&s), [a, b, bg]);
    }
}
