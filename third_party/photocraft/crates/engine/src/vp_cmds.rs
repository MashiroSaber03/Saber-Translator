//! Filter › Vanishing Point: perspective planes (drawn, or torn off an edge of another plane),
//! pasting an image onto a plane and the perspective clone stamp, as one replayable command
//! ([`photocraft_algo::vanishing`]). The edit lands on the active pixel layer, or on a new layer
//! with `"newLayer": true` (Photoshop's advice for non-destructive Vanishing Point work).

use photocraft_algo::vanishing::{Edge, PlaneCheck, Scene, VpPlane, clone_stroke, paste};
use photocraft_doc::{Document, Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

pub const VP: &str = "filter.vanishingPoint";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: VP.into(), msg: msg.into() }
}

fn pt(v: &Value) -> Option<[f64; 2]> {
    Some([v.get(0)?.as_f64()?, v.get(1)?.as_f64()?])
}

/// Builds the scene from `"planes"`.
pub fn scene(p: &Value, canvas: Rect) -> Result<Scene> {
    let planes = p
        .get("planes")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty())
        .ok_or_else(|| bad("pass `planes`: [{\"corners\":[[x,y]×4]}, {\"from\":0,\"edge\":\"top\",\"angle\":90}]"))?;
    let center = [(canvas.x0 + canvas.x1) as f64 / 2.0, (canvas.y0 + canvas.y1) as f64 / 2.0];
    let corners = |v: &Value| -> Option<[[f64; 2]; 4]> {
        let a = v.get("corners")?.as_array()?;
        Some([pt(a.first()?)?, pt(a.get(1)?)?, pt(a.get(2)?)?, pt(a.get(3)?)?])
    };
    // A drawn quad must be convex (Photoshop's red plane): a bow-tie or dented quad has no
    // perspective to paste or clone along.
    let drawn = |i: usize, c: [[f64; 2]; 4]| -> Result<VpPlane> {
        let pl = VpPlane { corners: c };
        if pl.check(center) == PlaneCheck::Invalid {
            return Err(bad(format!("plane {i}: the four corners must form a convex quad")));
        }
        Ok(pl)
    };
    let first = planes.first().and_then(corners).ok_or_else(|| bad("the first plane needs four `corners`"))?;
    let first = drawn(0, first)?;
    let focal = p.get("focalLength").and_then(Value::as_f64).filter(|f| *f > 0.0);
    let mut sc = Scene::new(first, center, focal, canvas.width().max(canvas.height()) as f64);
    for (i, pl) in planes.iter().enumerate().skip(1) {
        if let Some(c) = corners(pl) {
            sc.planes.push(drawn(i, c)?);
            continue;
        }
        let from = pl
            .get("from")
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .filter(|f| *f < sc.planes.len())
            .ok_or_else(|| bad(format!("plane {i}: `corners` or `from` (an earlier plane)")))?;
        let edge = match pl.get("edge").and_then(Value::as_str).unwrap_or("top") {
            "top" => Edge::Top,
            "right" => Edge::Right,
            "bottom" => Edge::Bottom,
            "left" => Edge::Left,
            e => return Err(bad(format!("plane {i}: unknown edge `{e}`"))),
        };
        let angle = pl.get("angle").and_then(Value::as_f64).unwrap_or(90.0);
        let depth = pl.get("depth").and_then(Value::as_f64).unwrap_or(1.0);
        sc.tear_off(from, edge, angle, depth).ok_or_else(|| bad(format!("plane {i} would lie behind the camera")))?;
    }
    if sc.planes.iter().any(|p| p.homography().is_none()) {
        return Err(bad("a plane is degenerate"));
    }
    Ok(sc)
}

fn has_layer(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    match d.active_layer.and_then(|id| d.doc.layer(id)).map(|l| &l.content) {
        Some(LayerContent::Raster(_)) => Ok(()),
        Some(other) => Err(format!("Vanishing Point needs a pixel layer (active layer is {} {} layer)", other.article(), other.kind_name())),
        None => Err("no active layer".into()),
    }
}

/// Composites `top` over `dst` (premultiplied source-over), same format.
fn over(dst: &mut Surface, top: &Surface) {
    let b = top.content_bounds();
    if b.is_empty() {
        return;
    }
    let top = if top.format() == dst.format() { top.clone() } else { top.convert(dst.format()) };
    let fmt = dst.format();
    let n = fmt.channels();
    let (t, mut d) = (top.read_region(b), dst.read_region(b));
    for (tp, dp) in t.chunks_exact(n).zip(d.chunks_exact_mut(n)) {
        if !fmt.alpha {
            dp.copy_from_slice(tp);
            continue;
        }
        let (ta, da) = (tp[n - 1], dp[n - 1]);
        let oa = ta + da * (1.0 - ta);
        for c in 0..n - 1 {
            dp[c] = if oa > 0.0 { (tp[c] * ta + dp[c] * da * (1.0 - ta)) / oa } else { 0.0 };
        }
        dp[n - 1] = oa;
    }
    dst.write_region(b, &d);
}

fn vanishing_point(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let canvas = st.doc.bounds();
    let sc = scene(p, canvas)?;
    // Paste sources resolved up front (layer pixels or the clipboard).
    let mut pastes: Vec<(usize, Surface, Rect, [f64; 2], f64)> = Vec::new();
    for (k, item) in p.get("paste").and_then(Value::as_array).cloned().unwrap_or_default().iter().enumerate() {
        let plane = item.get("plane").and_then(Value::as_u64).unwrap_or(0) as usize;
        if plane >= sc.planes.len() {
            return Err(bad(format!("paste {k}: no plane {plane}")));
        }
        let (surf, rect) = match item.get("layer").and_then(Value::as_u64) {
            Some(l) => {
                let sf = st.doc.layer(LayerId(l)).and_then(Layer::surface).ok_or(EngineError::NoLayer(LayerId(l)))?;
                (sf.clone(), sf.content_bounds())
            }
            None => {
                let clip = s.clipboard.as_ref().ok_or_else(|| bad(format!("paste {k}: nothing on the clipboard (or pass `layer`)")))?;
                (clip.surface.clone(), clip.surface.content_bounds())
            }
        };
        let at = item.get("at").and_then(pt).unwrap_or([0.25, 0.25]);
        let width = item.get("width").and_then(Value::as_f64).unwrap_or(0.5).clamp(0.01, 10.0);
        pastes.push((plane, surf, rect, at, width));
    }
    let clones: Vec<Value> = p.get("clone").and_then(Value::as_array).cloned().unwrap_or_default();
    let new_layer = p.get("newLayer").and_then(Value::as_bool).unwrap_or(false);
    let mut dabs = 0usize;
    let target = s.edit("Vanishing Point", |doc: &mut Document, active| {
        let id = if new_layer {
            let l = Layer::raster(doc.next_layer_name("Vanishing Point"), doc.pixel_format());
            let nid = doc.insert_above(*active, l);
            *active = Some(nid);
            nid
        } else {
            active.ok_or_else(|| EngineError::Other("no active layer".into()))?
        };
        // A new layer clones from the visible image; otherwise from the layer itself.
        let base: Option<Surface> = new_layer.then(|| crate::file_cmds::flattened(doc, doc.pixel_format()));
        let locks = doc.effective_locks(id);
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        if locks.all || locks.pixels {
            return Err(EngineError::Other(format!("layer \"{}\" is locked", l.name)));
        }
        let surf = l.surface_mut().ok_or_else(|| EngineError::Other("Vanishing Point needs a pixel layer".into()))?;
        for (plane, src, rect, at, width) in &pastes {
            if let Some(out) = paste(&sc, *plane, src, *rect, *at, *width)? {
                over(surf, &out);
            }
        }
        for (k, c) in clones.iter().enumerate() {
            let source = c.get("source").and_then(pt).ok_or_else(|| bad(format!("clone {k}: `source` [x,y]")))?;
            let points: Vec<[f64; 2]> = c.get("points").and_then(Value::as_array).map(|a| a.iter().filter_map(pt).collect()).unwrap_or_default();
            let size = c.get("size").and_then(Value::as_f64).unwrap_or(40.0).clamp(1.0, 2000.0);
            let hardness = c.get("hardness").and_then(Value::as_f64).unwrap_or(50.0).clamp(0.0, 100.0) / 100.0;
            let opacity = (c.get("opacity").and_then(Value::as_f64).unwrap_or(100.0).clamp(0.0, 100.0) / 100.0) as f32;
            // Sample from the original pixels (merged with what's been pasted).
            let mut work = match &base {
                Some(b) => {
                    let mut w = b.clone();
                    over(&mut w, surf);
                    w
                }
                None => surf.clone(),
            };
            let before = work.clone();
            dabs += clone_stroke(&sc, &mut work, source, &points, size, hardness, opacity);
            // Copy back only what the stroke changed.
            let b = work.content_bounds().union(&before.content_bounds());
            if !b.is_empty() {
                let (nw, ow) = (work.read_region(b), before.read_region(b));
                let n = surf.format().channels();
                let mut cur = surf.read_region(b);
                for i in 0..cur.len() / n {
                    if nw[i * n..(i + 1) * n] != ow[i * n..(i + 1) * n] {
                        cur[i * n..(i + 1) * n].copy_from_slice(&nw[i * n..(i + 1) * n]);
                    }
                }
                surf.write_region(b, &cur);
            }
        }
        surf.prune();
        Ok(id)
    })?;
    let planes: Vec<Value> = sc
        .planes
        .iter()
        .enumerate()
        .map(|(i, pl)| {
            let (lu, lv) = sc.lifted(i).lengths();
            json!({"corners": pl.corners, "aspect": lu / lv})
        })
        .collect();
    Ok(json!({"layer": target.0, "focal": sc.focal, "planes": planes, "pasted": pastes.len(), "dabs": dabs}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: VP,
        label: "Vanishing Point…",
        menu: &["Filter"],
        shortcut: Some("Cmd+Alt+V"),
        params: r##"{"planes":[{"corners":[[x,y]×4]} | {"from":plane,"edge":"top|right|bottom|left","angle":deg=90,"depth":1}],"focalLength":px=0,"paste":[{"plane":0,"layer":id? (else the clipboard),"at":[u,v]=[0.25,0.25],"width":0..1=0.5}],"clone":[{"source":[x,y],"points":[[x,y]],"size":px=40,"hardness":0..100=50,"opacity":0..100=100}],"newLayer":bool=false}"##,
        enabled: has_layer,
        run: vanishing_point,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planes_paste_and_clone() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 400, "height": 300})).unwrap();
        s.execute("layer.new.layer", json!({"name": "floor"})).unwrap();
        s.execute("edit.fill", json!({"color": "#404040"})).unwrap();
        let floor = s.active().unwrap().active_layer.unwrap();
        // A trapezoid "floor" receding to the top.
        let planes = json!([{"corners": [[120, 120], [280, 120], [360, 280], [40, 280]]}, {"from": 0, "edge": "top", "angle": 90}]);
        s.execute("layer.new.layer", json!({"name": "sticker"})).unwrap();
        s.edit("sticker", |doc, active| {
            doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap().fill_rect(Rect::new(0, 0, 40, 20), &[1.0, 0.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        let sticker = s.active().unwrap().active_layer.unwrap();
        s.execute("layer.select", json!({"layer": floor.0})).unwrap();
        let r = s.execute(VP, json!({"planes": planes, "paste": [{"plane": 0, "layer": sticker.0, "at": [0.3, 0.4], "width": 0.4}]})).unwrap();
        assert_eq!(r["planes"].as_array().unwrap().len(), 2);
        assert!(r["focal"].as_f64().unwrap() > 50.0, "{r}");
        let l = s.active().unwrap().doc.layer(floor).unwrap().surface().unwrap().clone();
        // The paste covers the plane's middle and not its far corner.
        let h = VpPlane { corners: [[120.0, 120.0], [280.0, 120.0], [360.0, 280.0], [40.0, 280.0]] }.homography().unwrap();
        let (x, y) = h.apply(0.5, 0.41);
        assert!(l.rgba(x as i32, y as i32)[0] > 0.9, "{:?}", l.rgba(x as i32, y as i32));
        assert!(l.rgba(130, 125)[0] < 0.5);
        s.undo();
        // Clone: copy the bright stripe painted near the bottom to the top of the floor.
        s.edit("stripe", |doc, _| {
            doc.layer_mut(floor).unwrap().surface_mut().unwrap().fill_rect(Rect::new(180, 250, 220, 265), &[0.0, 1.0, 0.0, 1.0]);
            Ok(())
        })
        .unwrap();
        s.execute("layer.select", json!({"layer": floor.0})).unwrap();
        let r = s
            .execute(VP, json!({"planes": planes, "clone": [{"source": [200, 257], "points": [[200, 160], [205, 160]], "size": 12, "hardness": 80}]}))
            .unwrap();
        assert!(r["dabs"].as_u64().unwrap() >= 1, "{r}");
        let l = s.active().unwrap().doc.layer(floor).unwrap().surface().unwrap();
        assert!(l.rgba(200, 160)[1] > 0.8, "{:?}", l.rgba(200, 160));
        assert!(s.execute(VP, json!({})).is_err());
        assert!(s.execute(VP, json!({"planes": [{"corners": [[0, 0], [1, 1]]}]})).is_err());
        let r = s.execute(VP, json!({"planes": planes, "newLayer": true})).unwrap();
        assert_ne!(r["layer"].as_u64().unwrap(), floor.0);
    }

    #[test]
    fn rejects_planes_that_are_not_convex() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 200, "height": 200, "depth": 16})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        let bowtie = json!([[20, 20], [180, 180], [180, 20], [20, 180]]);
        let dent = json!([[20, 20], [180, 20], [60, 60], [20, 180]]);
        let square = json!([[20, 20], [180, 20], [180, 180], [20, 180]]);
        for planes in [json!([{"corners": bowtie}]), json!([{"corners": square}, {"corners": dent}])] {
            let e = s.execute(VP, json!({"planes": planes})).unwrap_err().to_string();
            assert!(e.contains("convex"), "{e}");
        }
        assert!(s.execute(VP, json!({"planes": [{"corners": square}]})).is_ok());
    }
}
