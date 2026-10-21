//! Stamping (#217): merge layers into a layer while keeping the originals.
//!
//! - **Stamp Visible** (⌘⌥⇧E, or ⌥ + Layer › Merge Visible): every visible layer composited into a
//!   new layer above the active one.
//! - **Stamp Down** (⌘⌥E, or ⌥ + Layer › Merge Down): a copy of the active layer merged into the
//!   pixel layer below it; with several layers selected, a copy of them merged into a new layer
//!   above the top one.
//!
//! Photoshop has no menu items for these (they are the ⌥ variants of the merge commands), so they
//! are commands with default shortcuts: rebindable in Edit › Keyboard Shortcuts (Layer), recorded
//! in actions and drivable by agents. The composite is rendered in bands
//! ([`photocraft_compose::flatten_to_surface`]), so no full-size float copy of a large document is
//! held, and tiles left fully transparent are pruned.

use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document".into())
}

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document")?;
    st.active_layer.filter(|id| st.doc.layer(*id).is_some()).map(|_| ()).ok_or_else(|| "no active layer".into())
}

fn object(p: &Value, cmd: &str) -> Result<()> {
    match p {
        Value::Object(_) | Value::Null => Ok(()),
        _ => Err(EngineError::BadParams { cmd: cmd.into(), msg: "expected an object".into() }),
    }
}

/// The document's pixel format with an alpha channel (a stamp has transparency).
fn with_alpha(doc: &Document) -> PixelFormat {
    let f = doc.pixel_format();
    PixelFormat::new(f.mode, f.sample, true)
}

/// A new pixel layer holding the composite of `doc`'s visible layers, rendered in bands.
fn stamp_layer(doc: &Document, name: String) -> Result<Layer> {
    let fmt = with_alpha(doc);
    let mut layer = Layer::raster(name, fmt);
    *crate::pixels_mut(&mut layer)? = photocraft_compose::flatten_to_surface(doc, fmt, None);
    Ok(layer)
}

/// Make `id` the only selected layer and the active one.
fn select_only(s: &mut Session, id: LayerId) {
    if let Some(st) = s.active_mut() {
        st.selected_layers = vec![id];
        st.active_layer = Some(id);
        st.layer_anchor = Some(id);
        crate::fix_selection(st);
    }
}

fn stamp_visible(s: &mut Session, p: &Value) -> Result<Value> {
    object(p, "layer.stampVisible")?;
    let id = s.edit("Stamp Visible", |doc, active| {
        if !doc.walk().iter().any(|(path, _, l)| !l.is_group() && (1..=path.len()).all(|n| doc.layer_at(&path[..n]).is_some_and(|a| a.visible))) {
            return Err(EngineError::Other("there are no visible layers to stamp".into()));
        }
        let layer = stamp_layer(doc, doc.next_layer_name("Layer"))?;
        let id = doc.insert_above(*active, layer);
        *active = Some(id);
        Ok(id)
    })?;
    select_only(s, id);
    Ok(json!({ "layer": id.0 }))
}

fn stamp_down(s: &mut Session, p: &Value) -> Result<Value> {
    object(p, "layer.stampDown")?;
    let sel = crate::layer_multi_cmds::selected(s);
    if sel.len() >= 2 {
        return stamp_selected(s, &sel);
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let id = st.active_layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
    s.edit("Stamp Down", |doc, _| {
        let path = doc.path_of(id).ok_or(EngineError::NoLayer(id))?;
        let Some((&idx, parent)) = path.split_last() else { return Err(EngineError::NoLayer(id)) };
        let below_idx = idx.checked_sub(1).ok_or_else(|| EngineError::Other("there is no layer below to stamp into".into()))?;
        let mut below = parent.to_vec();
        below.push(below_idx);
        let lower = doc.layer_at(&below).ok_or(EngineError::NoLayer(id))?.clone();
        let upper = doc.layer(id).ok_or(EngineError::NoLayer(id))?.clone();
        if !matches!(lower.content, LayerContent::Raster(_)) {
            return Err(EngineError::Other("the layer below is not a pixel layer".into()));
        }
        let locks = doc.effective_locks(lower.id);
        if locks.all || locks.pixels {
            return Err(EngineError::Other("the layer below is locked".into()));
        }
        if !upper.visible {
            return Err(EngineError::Other("the layer is hidden".into()));
        }
        // The two layers on their own, composited in bands; the lower one keeps its id, name and
        // blend mode, as Merge Down does.
        let mut tmp = doc.clone();
        let mut base = lower.clone();
        base.visible = true;
        tmp.layers = vec![base, upper];
        let fmt = doc.pixel_format();
        let fmt = if lower.locks.transparency { fmt } else { PixelFormat::new(fmt.mode, fmt.sample, true) };
        let surf = photocraft_compose::flatten_to_surface(&tmp, fmt, None);
        let mut merged = Layer::new(lower.name.clone(), LayerContent::Raster(surf));
        merged.id = lower.id;
        merged.blend = lower.blend;
        merged.visible = lower.visible;
        merged.locks = lower.locks;
        merged.opacity = 1.0;
        *doc.layer_at_mut(&below).ok_or(EngineError::NoLayer(lower.id))? = merged;
        Ok(())
    })?;
    Ok(json!({ "layer": id.0 }))
}

/// Several layers selected: a copy of them merged into a new layer above the top one.
fn stamp_selected(s: &mut Session, sel: &[LayerId]) -> Result<Value> {
    let id = s.edit("Stamp Layers", |doc, active| {
        let ids = crate::layer_multi_cmds::top_level(doc, sel);
        let top = *ids.last().ok_or_else(|| EngineError::Other("no layers selected".into()))?;
        let mut solo = doc.clone();
        solo.layers = ids.iter().filter_map(|id| doc.layer(*id)).filter(|l| l.visible).cloned().collect();
        if solo.layers.is_empty() {
            return Err(EngineError::Other("the selected layers are all hidden".into()));
        }
        let name = format!("{} (merged)", doc.layer(top).map_or("Layer", |l| l.name.as_str()));
        let layer = stamp_layer(&solo, name)?;
        let id = doc.insert_above(Some(top), layer);
        *active = Some(id);
        Ok(id)
    })?;
    select_only(s, id);
    Ok(json!({ "layer": id.0 }))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "layer.stampVisible",
            label: "Stamp Visible",
            menu: &[],
            shortcut: Some("Cmd+Alt+Shift+E"),
            params: r##"{} → {layer} (every visible layer merged into a new layer above the active one; the originals stay)"##,
            enabled: has_doc,
            journal: true,
            run: stamp_visible,
        },
        CommandSpec {
            id: "layer.stampDown",
            label: "Stamp Down",
            menu: &[],
            shortcut: Some("Cmd+Alt+E"),
            params: r##"{} → {layer} (a copy of the active layer merged into the pixel layer below; several selected: into a new layer above them)"##,
            enabled: has_layer,
            journal: true,
            run: stamp_down,
        },
    ]
}

/// The ⌥ variant of a Layer menu merge command (⌥ + Merge Down / Merge Visible / Merge Layers).
pub fn alt_variant(id: &str) -> Option<&'static str> {
    match id {
        "layer.mergeVisible" => Some("layer.stampVisible"),
        "layer.mergeDown" | "layer.mergeLayers" => Some("layer.stampDown"),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 64 × 64 at `depth`: a red Background, a blue square (16..48) on "Blue", and "Hidden" (green
    /// all over, hidden). The Blue layer is active.
    fn session(depth: u32) -> (Session, LayerId, LayerId, LayerId) {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64, "depth": depth, "background": "#ff0000"})).unwrap();
        let bg = s.active().unwrap().doc.layers[0].id;
        s.execute("layer.new.layer", json!({"name": "Hidden"})).unwrap();
        s.execute("edit.fill", json!({"color": "#00ff00"})).unwrap();
        let hidden = s.active().unwrap().active_layer.unwrap();
        s.execute("layer.setProps", json!({"layer": hidden.0, "visible": false})).unwrap();
        s.execute("layer.new.layer", json!({"name": "Blue"})).unwrap();
        s.execute("select.rect", json!({"x": 16, "y": 16, "width": 32, "height": 32})).unwrap();
        s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let blue = s.active().unwrap().active_layer.unwrap();
        (s, bg, hidden, blue)
    }

    fn rgba(s: &Session, id: LayerId, x: i32, y: i32) -> [f32; 4] {
        s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().rgba(x, y)
    }

    fn ids(s: &Session) -> Vec<LayerId> {
        s.active().unwrap().doc.walk().iter().map(|(_, _, l)| l.id).collect()
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
    }

    #[test]
    fn stamp_visible_adds_the_composite_above_the_active_layer() {
        for depth in [8, 16, 32] {
            let (mut s, bg, hidden, blue) = session(depth);
            let before = ids(&s);
            let r = s.execute("layer.stampVisible", json!({})).unwrap();
            let id = LayerId(r["layer"].as_u64().unwrap());
            // The originals stay; the stamp sits right above the active layer and is selected.
            assert_eq!(ids(&s), [bg, hidden, blue, id], "{depth}-bit");
            let st = s.active().unwrap();
            assert_eq!(st.active_layer, Some(id));
            assert_eq!(st.selected_layers, vec![id]);
            let l = st.doc.layer(id).unwrap();
            assert!(l.visible && matches!(l.content, LayerContent::Raster(_)));
            assert!(l.name.starts_with("Layer "), "{}", l.name);
            assert!(l.surface().unwrap().format().alpha);
            // The visible composite: blue square on red; the hidden green layer is left out.
            assert!(close(rgba(&s, id, 30, 30), [0.0, 0.0, 1.0, 1.0]), "{depth}: {:?}", rgba(&s, id, 30, 30));
            assert!(close(rgba(&s, id, 2, 2), [1.0, 0.0, 0.0, 1.0]));
            assert!(close(rgba(&s, blue, 2, 2), [0.0, 0.0, 0.0, 0.0]), "Blue is untouched");
            // One undo step.
            s.undo();
            assert_eq!(ids(&s), before);
        }
    }

    #[test]
    fn stamp_visible_respects_opacity_blend_and_groups() {
        let (mut s, bg, _, blue) = session(8);
        s.execute("layer.setProps", json!({"layer": blue.0, "opacity": 0.5})).unwrap();
        s.execute("layer.groupLayers", json!({"name": "G"})).unwrap();
        // The active layer is in a group: the stamp goes into the group, above it.
        s.execute("layer.select", json!({"layer": blue.0})).unwrap();
        let id = LayerId(s.execute("layer.stampVisible", json!({})).unwrap()["layer"].as_u64().unwrap());
        let doc = &s.active().unwrap().doc;
        let path = doc.path_of(id).unwrap();
        assert_eq!(path.len(), 2, "inside the group");
        assert_eq!(doc.path_of(blue).unwrap().last().map(|i| i + 1), path.last().copied());
        let p = rgba(&s, id, 30, 30);
        assert!((p[0] - 0.5).abs() < 0.02 && (p[2] - 0.5).abs() < 0.02 && p[3] > 0.99, "50 % blue over red: {p:?}");
        assert!(doc.layer(bg).is_some());
    }

    #[test]
    fn stamp_visible_fails_gracefully() {
        let mut s = Session::new();
        assert!(s.execute("layer.stampVisible", json!({})).is_err(), "no document");
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        let bg = s.active().unwrap().doc.layers[0].id;
        s.execute("layer.setProps", json!({"layer": bg.0, "visible": false})).unwrap();
        let rev = s.active().unwrap().revision;
        assert!(s.execute("layer.stampVisible", json!({})).is_err(), "nothing visible");
        assert_eq!(s.active().unwrap().revision, rev, "no change");
        s.execute("layer.setProps", json!({"layer": bg.0, "visible": true})).unwrap();
        for bad in [json!([1, 2]), json!("x"), json!(3)] {
            assert!(s.execute("layer.stampVisible", bad.clone()).is_err(), "{bad}");
            assert!(s.execute("layer.stampDown", bad).is_err());
        }
        assert!(s.execute("layer.stampVisible", Value::Null).is_ok());
    }

    #[test]
    fn stamp_down_merges_a_copy_into_the_layer_below() {
        let (mut s, bg, hidden, blue) = session(16);
        // Blue is above Hidden (hidden): stamp into a visible pixel layer instead.
        s.execute("layer.setProps", json!({"layer": hidden.0, "visible": true})).unwrap();
        s.execute("layer.stampDown", json!({})).unwrap();
        assert_eq!(ids(&s), [bg, hidden, blue], "no layer added or removed");
        assert!(close(rgba(&s, hidden, 30, 30), [0.0, 0.0, 1.0, 1.0]), "the blue square was stamped into Hidden");
        assert!(close(rgba(&s, hidden, 2, 2), [0.0, 1.0, 0.0, 1.0]));
        assert!(close(rgba(&s, blue, 30, 30), [0.0, 0.0, 1.0, 1.0]), "Blue stays");
        assert_eq!(s.active().unwrap().active_layer, Some(blue));
        s.undo();
        assert!(close(rgba(&s, hidden, 30, 30), [0.0, 1.0, 0.0, 1.0]));
    }

    #[test]
    fn stamp_down_refuses_what_it_cannot_do() {
        let (mut s, bg, hidden, blue) = session(8);
        let rev = s.active().unwrap().revision;
        // The bottom layer has nothing below.
        s.execute("layer.select", json!({"layer": bg.0})).unwrap();
        assert!(s.execute("layer.stampDown", json!({})).is_err());
        // A locked layer below.
        s.execute("layer.select", json!({"layer": blue.0})).unwrap();
        s.execute("layer.setProps", json!({"layer": hidden.0, "locks": {"pixels": true}})).unwrap();
        assert!(s.execute("layer.stampDown", json!({})).is_err());
        s.execute("layer.setProps", json!({"layer": hidden.0, "locks": {"pixels": false}})).unwrap();
        // A hidden active layer.
        s.execute("layer.setProps", json!({"layer": blue.0, "visible": false})).unwrap();
        assert!(s.execute("layer.stampDown", json!({})).is_err());
        s.execute("layer.setProps", json!({"layer": blue.0, "visible": true})).unwrap();
        // A fill layer below is not a pixel layer.
        s.execute("layer.select", json!({"layer": hidden.0})).unwrap();
        s.execute("layer.newFillLayer.solidColor", json!({"color": "#ffffff"})).unwrap();
        s.execute("layer.select", json!({"layer": blue.0})).unwrap();
        assert!(s.execute("layer.stampDown", json!({})).is_err());
        assert!(s.active().unwrap().revision > rev);
        let mut empty = Session::new();
        assert!(empty.execute("layer.stampDown", json!({})).is_err(), "no document");
    }

    #[test]
    fn stamp_with_several_layers_selected_makes_a_new_layer() {
        let (mut s, bg, hidden, blue) = session(8);
        s.execute("layer.setProps", json!({"layer": hidden.0, "visible": true})).unwrap();
        s.execute("layer.select", json!({"layer": hidden.0})).unwrap();
        s.execute("layer.select", json!({"layer": blue.0, "mode": "add"})).unwrap();
        let id = LayerId(s.execute("layer.stampDown", json!({})).unwrap()["layer"].as_u64().unwrap());
        assert_eq!(ids(&s), [bg, hidden, blue, id]);
        assert!(close(rgba(&s, id, 30, 30), [0.0, 0.0, 1.0, 1.0]));
        assert!(close(rgba(&s, id, 2, 2), [0.0, 1.0, 0.0, 1.0]), "only the selected layers, not the red Background");
        assert_eq!(s.active().unwrap().doc.layer(id).unwrap().name, "Blue (merged)");
    }

    #[test]
    fn shortcuts_are_photoshops_and_rebindable() {
        let spec = |id: &str| crate::commands::find(id).unwrap();
        assert_eq!(spec("layer.stampVisible").shortcut, Some("Cmd+Alt+Shift+E"));
        assert_eq!(spec("layer.stampDown").shortcut, Some("Cmd+Alt+E"));
        let mut s = Session::new();
        s.execute("edit.keyboardShortcuts", json!({"set": {"layer.stampVisible": "F7"}})).unwrap();
        assert_eq!(s.prefs().shortcut("layer.stampVisible", Some("Cmd+Alt+Shift+E")), Some("F7"));
        assert_eq!(alt_variant("layer.mergeVisible"), Some("layer.stampVisible"));
        assert_eq!(alt_variant("layer.mergeDown"), Some("layer.stampDown"));
        assert_eq!(alt_variant("layer.flattenImage"), None);
    }

    /// Release timing and peak memory of a 24 MP Stamp Visible:
    /// `cargo test --release -p photocraft-engine --lib stamp_visible_24mp -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn stamp_visible_24mp() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 6000, "height": 4000})).unwrap();
        for k in 0..3 {
            s.execute("layer.new.layer", json!({})).unwrap();
            s.execute("select.rect", json!({"x": 500 * k, "y": 400 * k, "width": 3000, "height": 2000})).unwrap();
            s.execute("edit.fill", json!({"color": [0.2 * k as f32, 0.5, 0.8, 1.0]})).unwrap();
        }
        s.execute("select.deselect", json!({})).unwrap();
        let t = std::time::Instant::now();
        s.execute("layer.stampVisible", json!({})).unwrap();
        println!("stampVisible 6000x4000, 4 layers: {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    }
}
