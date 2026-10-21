//! Layer › New › Frame from Layers. A frame clips its contents to a rectangle, so a template can
//! hold an image (or placeholder) that is cropped to the frame. We build it from existing
//! primitives: group the selected layers, then clip the group with a rectangular layer mask the
//! size of their bounds — the same visual result Photoshop's Frame tool produces for this command.
//! `layer.new.frame` is the Frame tool's (K) command: a rectangular or elliptical frame drawn on
//! the canvas, empty or holding the pixel layer it was drawn over. Headless and scriptable like
//! every other command.

use photocraft_doc::{Layer, LayerContent, LayerId, LayerMask, PixelFormat};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    match s.active() {
        Some(d) if !d.selected_layers.is_empty() || d.active_layer.is_some() => Ok(()),
        Some(_) => Err("select a layer".into()),
        None => Err("no document".into()),
    }
}

/// A reveal mask that is on inside `rect` and off (hidden) everywhere else.
fn clip_mask(rect: Rect) -> LayerMask {
    let mut mask = LayerMask::hide_all();
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    if w > 0 && h > 0 {
        mask.surface.write_region(rect, &vec![1.0f32; w * h]);
        mask.surface.prune();
    }
    // Default to GRAY8 like the other mask constructors.
    debug_assert_eq!(mask.surface.format(), PixelFormat::GRAY8);
    mask
}

/// Groups the selected layers, then turns the group into a frame, as one history step: one undo
/// takes the whole command back (#496).
fn frame_from_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let mut gp = serde_json::Map::new();
    if let Some(name) = p.get("name") {
        gp.insert("name".into(), name.clone());
    }
    let journal = s.journal.len();
    let grouped = s.execute("layer.new.groupFromLayers", Value::Object(gp))?;
    match frame_group(s, &grouped) {
        Ok(r) => {
            // Fold the grouping step into the frame step.
            if let Some(st) = s.active_mut() {
                st.history.purge_last();
            }
            Ok(r)
        }
        Err(e) => {
            // The group couldn't be framed: take the grouping back, leaving nothing to redo and
            // no grouping step to record.
            s.undo();
            s.journal.truncate(journal);
            if let Some(st) = s.active_mut() {
                st.history.clear_redo();
            }
            Err(e)
        }
    }
}

/// Clips the group `layer.new.groupFromLayers` returned (`grouped`) to its contents' bounds.
fn frame_group(s: &mut Session, grouped: &Value) -> Result<Value> {
    let gid = LayerId(grouped.get("layer").and_then(Value::as_u64).ok_or_else(|| EngineError::Other("grouping did not return a layer".into()))?);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let rect =
        crate::layer_multi_cmds::layer_bounds(d.doc.layer(gid).ok_or(EngineError::NoLayer(gid))?).filter(|r| !r.is_empty()).unwrap_or_else(|| d.doc.bounds());

    s.edit("Frame from Layers", |doc, _| {
        let g = doc.layer_mut(gid).ok_or(EngineError::NoLayer(gid))?;
        g.mask = Some(clip_mask(rect));
        Ok(())
    })?;
    Ok(json!({"layer": gid.0, "frame": [rect.x0, rect.y0, rect.width(), rect.height()]}))
}

/// The largest frame side (px) `layer.new.frame` accepts, as Photoshop's canvas limit.
const MAX_FRAME_SIDE: i32 = 300_000;

/// A reveal mask that is on inside the ellipse inscribed in `rect` and off everywhere else, with
/// 4x4-supersampled edges (as the Elliptical Marquee). Only pixels inside `area` are written.
fn ellipse_mask(rect: Rect, area: Rect) -> LayerMask {
    let mut mask = LayerMask::hide_all();
    let (cx, cy) = ((f64::from(rect.x0) + f64::from(rect.x1)) / 2.0, (f64::from(rect.y0) + f64::from(rect.y1)) / 2.0);
    let (rx, ry) = (f64::from(rect.width().max(1)) / 2.0, f64::from(rect.height().max(1)) / 2.0);
    let inside = |x: f64, y: f64| {
        let (dx, dy) = ((x - cx) / rx, (y - cy) / ry);
        dx * dx + dy * dy <= 1.0
    };
    let cut = rect.intersect(&area);
    for y in cut.y0..cut.y1 {
        for x in cut.x0..cut.x1 {
            let (fx, fy) = (f64::from(x), f64::from(y));
            let n = (0..16).filter(|i| inside(fx + (f64::from(i % 4) + 0.5) / 4.0, fy + (f64::from(i / 4) + 0.5) / 4.0)).count();
            if n > 0 {
                mask.surface.write_pixel(x, y, &[n as f32 / 16.0]);
            }
        }
    }
    mask.surface.prune();
    mask
}

/// Frame tool (K): a new frame — a group clipped to the drawn rectangle or ellipse — as one
/// history step. With `layer` (a pixel layer, not the Background), that layer becomes the frame's
/// content, as when a frame is drawn over a selected pixel layer in Photoshop; else it is empty.
fn new_frame(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.new.frame";
    let bad = |msg: String| EngineError::BadParams { cmd: CMD.into(), msg };
    let get = |k: &str| crate::commands::int_i32(CMD, p, k)?.ok_or_else(|| bad(format!("missing `{k}`")));
    let (x, y, w, h) = (get("x")?, get("y")?, get("width")?, get("height")?);
    if !(1..=MAX_FRAME_SIDE).contains(&w) || !(1..=MAX_FRAME_SIDE).contains(&h) {
        return Err(bad(format!("`width` and `height` must be 1..={MAX_FRAME_SIDE}")));
    }
    let (Some(x1), Some(y1)) = (x.checked_add(w), y.checked_add(h)) else {
        return Err(bad("the frame is outside the 32-bit coordinate range".into()));
    };
    let rect = Rect::new(x, y, x1, y1);
    let ellipse = match p.get("shape").map(|v| v.as_str()) {
        None | Some(Some("rectangle")) => false,
        Some(Some("ellipse")) => true,
        Some(_) => return Err(bad("`shape` must be \"rectangle\" or \"ellipse\"".into())),
    };
    let content = match p.get("layer") {
        None | Some(Value::Null) => None,
        Some(v) => Some(LayerId(v.as_u64().ok_or_else(|| bad(format!("`layer` = {v} is not a layer id")))?)),
    };
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let gid = s.edit("New Frame", |doc, active| {
        let mask = if ellipse { ellipse_mask(rect, doc.bounds()) } else { clip_mask(rect.intersect(&doc.bounds())) };
        let name = name.unwrap_or_else(|| doc.next_layer_name("Frame"));
        let gid = match content {
            Some(id) => {
                let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
                if !matches!(l.content, LayerContent::Raster(_)) || doc.is_background(id) {
                    return Err(bad("only a pixel layer (not the Background) can go in a frame".into()));
                }
                let gid = doc.insert_above(Some(id), Layer::group(name, vec![]));
                let child = doc.remove(id).ok_or(EngineError::NoLayer(id))?;
                doc.layer_mut(gid).and_then(Layer::children_mut).ok_or(EngineError::NoLayer(gid))?.push(child);
                crate::layer_multi_cmds::check_group_depth(doc, "New Frame")?;
                gid
            }
            None => doc.insert_above(*active, Layer::group(name, vec![])),
        };
        doc.layer_mut(gid).ok_or(EngineError::NoLayer(gid))?.mask = Some(mask);
        *active = Some(gid);
        Ok(gid)
    })?;
    crate::layer_multi_cmds::reselect(s, vec![gid], Some(gid));
    Ok(json!({"layer": gid.0, "frame": [x, y, w, h], "shape": if ellipse { "ellipse" } else { "rectangle" }}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "layer.new.frame",
            label: "New Frame",
            menu: &[],
            shortcut: None,
            params: r#"{"x":i32,"y":i32,"width":u32,"height":u32,"shape":"rectangle|ellipse"="rectangle","layer":id?,"name":str?} → {layer, frame:[x,y,w,h], shape}: Frame tool; a group clipped to the shape, holding pixel layer `layer` if given"#,
            enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            journal: true,
            run: new_frame,
        },
        CommandSpec {
            id: "layer.new.frameFromLayers",
            label: "Frame from Layers",
            menu: &["Layer", "New"],
            shortcut: None,
            params: r#"{"name":str?} → {layer, frame:[x,y,w,h]}: groups the selected layers and clips them to a frame rect"#,
            enabled: has_layer,
            journal: true,
            run: |s, p| frame_from_layers(s, p),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Layer, LayerContent};
    use photocraft_geom::Rect;

    fn session() -> (Session, LayerId, LayerId) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 80, "height": 60})).unwrap();
        let (a, b) = s
            .edit("setup", |doc, _| {
                let fmt = doc.pixel_format();
                let mut a = Layer::raster("A", fmt);
                a.surface_mut().unwrap().fill_rect(Rect::new(4, 4, 24, 24), &photocraft_raster::from_rgba(&fmt, [1.0, 0.0, 0.0, 1.0]));
                let mut b = Layer::raster("B", fmt);
                b.surface_mut().unwrap().fill_rect(Rect::new(30, 20, 50, 40), &photocraft_raster::from_rgba(&fmt, [0.0, 0.0, 1.0, 1.0]));
                let (ia, ib) = (a.id, b.id);
                doc.layers.push(a);
                doc.layers.push(b);
                Ok((ia, ib))
            })
            .unwrap();
        (s, a, b)
    }

    #[test]
    fn frames_selected_layers_into_a_masked_group() {
        let (mut s, a, b) = session();
        s.execute("layer.select", json!({"layer": a.0, "mode": "replace"})).unwrap();
        s.execute("layer.select", json!({"layer": b.0, "mode": "add"})).unwrap();
        let r = s.execute("layer.new.frameFromLayers", json!({"name": "Frame 1"})).unwrap();
        let gid = LayerId(r["layer"].as_u64().unwrap());
        // frame rect = union of the two layers' bounds (4,4)..(50,40).
        let frame = r["frame"].as_array().unwrap();
        assert_eq!(frame[2].as_i64().unwrap(), 50 - 4); // width
        assert_eq!(frame[3].as_i64().unwrap(), 40 - 4); // height
        let st = s.active().unwrap();
        let g = st.doc.layer(gid).unwrap();
        assert!(matches!(g.content, LayerContent::Group(_)), "it's a group");
        assert!(g.mask.is_some(), "the group has a clip mask");
        // Undoable (last step is the frame).
        assert_eq!(st.history.entries().last().map(|e| e.as_str()), Some("Frame from Layers"));
    }

    #[test]
    fn frame_from_layers_is_one_history_step() {
        // #496: grouping and framing undo together.
        let (mut s, a, b) = session();
        let entries = s.active().unwrap().history.entries();
        let layers = |s: &Session| s.active().unwrap().doc.layers.iter().map(|l| l.id).collect::<Vec<_>>();
        let before = layers(&s);
        s.execute("layer.select", json!({"layer": a.0, "mode": "replace"})).unwrap();
        s.execute("layer.select", json!({"layer": b.0, "mode": "add"})).unwrap();
        let gid = LayerId(s.execute("layer.new.frameFromLayers", json!({})).unwrap()["layer"].as_u64().unwrap());
        let mut framed = entries.clone();
        framed.push("Frame from Layers".into());
        assert_eq!(s.active().unwrap().history.entries(), framed);
        assert!(s.undo());
        assert_eq!(s.active().unwrap().history.entries(), entries);
        assert_eq!(layers(&s), before, "one undo leaves no group behind");
        assert!(s.redo());
        assert!(s.active().unwrap().doc.layer(gid).is_some_and(|g| g.mask.is_some()), "redo brings back the framed group");
    }

    /// Frame tool: a rectangular frame is an empty group clipped to the drawn rect, undone in one
    /// step; an elliptical one hides the corners; drawn over a pixel layer, it holds that layer.
    #[test]
    fn new_frame_clips_a_group_to_the_drawn_shape() {
        let (mut s, a, _) = session();
        let count = |s: &Session| s.active().unwrap().doc.layers.len();
        let before = count(&s);
        let r = s.execute("layer.new.frame", json!({"x": 10, "y": 10, "width": 40, "height": 30})).unwrap();
        let gid = LayerId(r["layer"].as_u64().unwrap());
        let st = s.active().unwrap();
        let g = st.doc.layer(gid).unwrap();
        assert!(matches!(&g.content, LayerContent::Group(c) if c.children.is_empty()), "an empty frame");
        let m = g.mask.as_ref().unwrap();
        assert_eq!((m.value(10, 10), m.value(49, 39), m.value(9, 10), m.value(50, 39)), (1.0, 1.0, 0.0, 0.0));
        assert_eq!((g.name.as_str(), st.active_layer), ("Frame 1", Some(gid)));
        assert!(s.undo());
        assert_eq!(count(&s), before, "one undo removes the frame");

        let r = s.execute("layer.new.frame", json!({"x": 10, "y": 10, "width": 40, "height": 30, "shape": "ellipse", "layer": a.0})).unwrap();
        let gid = LayerId(r["layer"].as_u64().unwrap());
        let g = s.active().unwrap().doc.layer(gid).unwrap();
        assert!(matches!(&g.content, LayerContent::Group(c) if c.children.iter().map(|l| l.id).eq([a])), "the frame holds layer A");
        let m = g.mask.as_ref().unwrap();
        assert_eq!((m.value(30, 25), m.value(10, 10)), (1.0, 0.0), "centre shown, corner hidden");
        for bad in [json!({"x": 0, "y": 0, "width": 0, "height": 5}), json!({"x": 0, "y": 0, "width": 5, "height": 5, "shape": "star"}), json!({"x": 0})] {
            assert!(s.execute("layer.new.frame", bad).is_err());
        }
    }

    #[test]
    fn needs_a_selected_layer() {
        let mut s = Session::new();
        assert!(s.execute("layer.new.frameFromLayers", json!({})).is_err());
    }
}
