//! Select › Sky, Select › Isolate Layers and Select › Transform Selection.
//!
//! * `select.sky` runs the classical sky segmentation of `photocraft_algo::segment::sky` on the
//!   composite (or the active layer) and combines it with the selection.
//! * `select.isolateLayers` toggles the Layers panel's isolation filter: only the layers selected
//!   when it was turned on are listed ([`crate::DocState::isolated_layers`], view state).
//! * `select.transformSelection` transforms the selection outline/mask only (not pixels): a
//!   quad (distort / perspective), an affine matrix, scale / rotate / skew / move about a
//!   reference point, or a warp (same params as `edit.transform.warp`).

use photocraft_algo::segment::sky::{self, SkyParams};
use photocraft_algo::segment::{Sampler, SurfaceSampler};
use photocraft_algo::selection::{SelectionMode, combine_region};
use photocraft_algo::transform::{Homography, Interp};
use photocraft_algo::warp::warp_mesh_gray;
use photocraft_doc::{Document, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}
fn has_selection(s: &Session) -> std::result::Result<(), String> {
    s.active().filter(|d| d.doc.selection.is_some()).map(|_| ()).ok_or_else(|| "no selection".into())
}
fn has_layer(s: &Session) -> std::result::Result<(), String> {
    s.active().and_then(|d| d.active_layer).map(|_| ()).ok_or_else(|| "no active layer".into())
}
fn f(p: &Value, k: &str, d: f64) -> f64 {
    p.get(k).and_then(Value::as_f64).unwrap_or(d)
}
fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

struct CompositeSampler<'a>(&'a Document);

impl Sampler for CompositeSampler<'_> {
    fn rgba(&self, r: Rect) -> Vec<[f32; 4]> {
        photocraft_compose::render(self.0, r).px
    }
}

// ---------------------------------------------------------------- Sky

fn select_sky(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let canvas = doc.bounds();
    let params = SkyParams { threshold: f(p, "threshold", 50.0) as f32, softness: f(p, "softness", 50.0) as f32 };
    let all = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(true);
    let layer = d.active_layer.and_then(|id| doc.layer(id)).and_then(|l| l.surface());
    let region = match (all, layer) {
        (false, Some(surf)) => sky::select_sky(&SurfaceSampler(surf), canvas, params),
        _ => sky::select_sky(&CompositeSampler(&doc), canvas, params),
    };
    let m = p.get("mode").and_then(Value::as_str).unwrap_or("replace");
    let m = SelectionMode::parse(if m == "new" { "replace" } else { m });
    let Some(region) = region else {
        // Photoshop reports "No sky was detected" and leaves the selection alone.
        return Ok(json!({ "selected": doc.selection.is_some(), "changed": false, "coverage": 0.0, "message": "No sky was detected" }));
    };
    let total: f64 = region.mask.iter().map(|v| f64::from(*v) / 255.0).sum();
    let coverage = total / (canvas.width() as f64 * canvas.height() as f64).max(1.0);
    let b = region.bbox;
    let selected = s.edit("Select Sky", |doc, _| {
        doc.selection = combine_region(doc.selection.as_ref(), Some(&region), m);
        Ok(doc.selection.is_some())
    })?;
    Ok(json!({ "selected": selected, "changed": true, "coverage": coverage, "bounds": [b.x0, b.y0, b.width() as i32, b.height() as i32] }))
}

// ---------------------------------------------------------------- Isolate Layers

fn isolate_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let on = p.get("on").and_then(Value::as_bool).unwrap_or(st.isolated_layers.is_empty());
    st.isolated_layers = if on { st.selected_layers() } else { Vec::new() };
    st.revision += 1;
    Ok(json!({ "isolated": !st.isolated_layers.is_empty(), "layers": st.isolated_layers.iter().map(|l| l.0).collect::<Vec<_>>() }))
}

/// Whether a Layers panel row for `id` is listed under isolation (ancestors of an isolated layer
/// stay visible so it can be shown in place).
pub fn isolation_shows(doc: &Document, isolated: &[LayerId], id: LayerId) -> bool {
    if isolated.is_empty() || isolated.contains(&id) {
        return true;
    }
    fn contains(l: &photocraft_doc::Layer, set: &[LayerId]) -> bool {
        match &l.content {
            photocraft_doc::LayerContent::Group(g) => g.children.iter().any(|c| set.contains(&c.id) || contains(c, set)),
            _ => false,
        }
    }
    doc.layer(id).is_some_and(|l| contains(l, isolated))
}

// ---------------------------------------------------------------- Transform Selection

fn reference_point(p: &Value, r: [f64; 4]) -> [f64; 2] {
    let (x0, y0, x1, y1) = (r[0], r[1], r[2], r[3]);
    let (cx, cy) = ((x0 + x1) / 2.0, (y0 + y1) / 2.0);
    match p.get("reference") {
        Some(Value::Array(a)) if a.len() >= 2 => [a[0].as_f64().unwrap_or(cx), a[1].as_f64().unwrap_or(cy)],
        Some(Value::String(k)) => match k.as_str() {
            "topLeft" => [x0, y0],
            "top" => [cx, y0],
            "topRight" => [x1, y0],
            "left" => [x0, cy],
            "right" => [x1, cy],
            "bottomLeft" => [x0, y1],
            "bottom" => [cx, y1],
            "bottomRight" => [x1, y1],
            _ => [cx, cy],
        },
        _ => [cx, cy],
    }
}

fn quad(p: &Value) -> Option<[[f64; 2]; 4]> {
    let a = p.get("quad").or_else(|| p.get("corners"))?.as_array()?;
    if a.len() != 4 {
        return None;
    }
    let mut q = [[0.0; 2]; 4];
    for (i, v) in a.iter().enumerate() {
        let c = v.as_array()?;
        q[i] = [c.first()?.as_f64()?, c.get(1)?.as_f64()?];
    }
    Some(q)
}

/// The homography of the scale / rotate / skew / move params about the reference point.
fn affine_params(p: &Value, r: [f64; 4]) -> Homography {
    let [px, py] = reference_point(p, r);
    let (sx, sy) = (f(p, "scaleX", 100.0) / 100.0, f(p, "scaleY", 100.0) / 100.0);
    let a = f(p, "rotate", 0.0).to_radians();
    let (kx, ky) = (f(p, "skewX", 0.0).to_radians().tan(), f(p, "skewY", 0.0).to_radians().tan());
    let (dx, dy) = (f(p, "dx", f(p, "x", 0.0)), f(p, "dy", f(p, "y", 0.0)));
    let t = |x: f64, y: f64| Homography([1.0, 0.0, x, 0.0, 1.0, y, 0.0, 0.0, 1.0]);
    let (sa, ca) = a.sin_cos();
    let rot = Homography([ca, -sa, 0.0, sa, ca, 0.0, 0.0, 0.0, 1.0]);
    let skew = Homography([1.0, kx, 0.0, ky, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let scale = Homography([sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0]);
    t(px + dx, py + dy).mul(&rot).mul(&skew).mul(&scale).mul(&t(-px, -py))
}

fn transform_selection(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "select.transformSelection";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let sel = d.doc.selection.as_ref().ok_or_else(|| EngineError::Other("no selection".into()))?;
    let b = sel.content_bounds();
    let rect = match p.get("rect").and_then(Value::as_array) {
        Some(a) if a.len() == 4 => {
            let v: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
            [v[0], v[1], v[2], v[3]]
        }
        _ => [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64],
    };
    let interp = Interp::parse(p.get("interpolation").and_then(Value::as_str).unwrap_or("bilinear"));
    let is_warp = ["warp", "style", "mesh", "grid"].iter().any(|k| p.get(*k).is_some());
    if is_warp {
        let r = Rect::new(rect[0].floor() as i32, rect[1].floor() as i32, rect[2].ceil() as i32, rect[3].ceil() as i32);
        let w = crate::warp_cmds::warp_from_params(CMD, p, [r.x0 as f64, r.y0 as f64, r.x1 as f64, r.y1 as f64])?;
        if w.is_identity() {
            return Ok(json!({"changed": false}));
        }
        let selected = s.edit("Transform Selection", |doc, _| {
            if let Some(sel) = &doc.selection {
                doc.selection = Some(warp_mesh_gray(sel, &|x, y| w.map(x, y), interp)).filter(|s| !s.content_bounds().is_empty());
            }
            Ok(doc.selection.is_some())
        })?;
        return Ok(json!({"changed": true, "selected": selected, "bounds": bounds_json(s)}));
    }
    let h = if let Some(q) = quad(p) {
        Homography::rect_to_quad(rect, q).ok_or_else(|| bad(CMD, "degenerate quad"))?
    } else if let Some(m) = p.get("matrix").and_then(Value::as_array) {
        let v: Vec<f64> = m.iter().filter_map(Value::as_f64).collect();
        if v.len() != 6 {
            return Err(bad(CMD, "matrix must be [a, b, c, d, e, f]"));
        }
        Homography([v[0], v[2], v[4], v[1], v[3], v[5], 0.0, 0.0, 1.0])
    } else {
        affine_params(p, rect)
    };
    if h.inverse().is_none() {
        return Err(bad(CMD, "the transform collapses the selection"));
    }
    if h.0.iter().zip(Homography::IDENTITY.0).all(|(a, b)| (a - b).abs() < 1e-12) {
        return Ok(json!({"changed": false, "selected": true}));
    }
    let selected = s.edit("Transform Selection", |doc, _| {
        if let Some(sel) = &doc.selection {
            crate::transform_cmds::check_budget(sel.content_bounds(), &h, "selection")?;
            doc.selection = Some(crate::transform_cmds::warp_gray(sel, &h, interp)?).filter(|s| !s.content_bounds().is_empty());
        }
        Ok(doc.selection.is_some())
    })?;
    Ok(json!({"changed": true, "selected": selected, "bounds": bounds_json(s)}))
}

fn bounds_json(s: &Session) -> Value {
    match s.active().and_then(|d| d.doc.selection.as_ref()).map(|s| s.content_bounds()) {
        Some(r) => json!([r.x0, r.y0, r.width() as i32, r.height() as i32]),
        None => Value::Null,
    }
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "select.sky",
            label: "Sky",
            menu: &["Select"],
            shortcut: None,
            params: r##"{"mode":"replace|add|subtract|intersect"="replace","sampleAllLayers":bool=true,"threshold":0..100=50 (higher = stricter),"softness":0..100=50 (edge softness)} → {selected, changed, coverage (0–1 of the canvas), bounds [x,y,w,h]}"##,
            enabled: has_doc,
            run: select_sky,
            journal: true,
        },
        CommandSpec {
            id: "select.isolateLayers",
            label: "Isolate Layers",
            menu: &["Select"],
            shortcut: None,
            params: r##"{"on":bool? (default: toggle)} — Layers panel lists only the selected layers → {isolated, layers}"##,
            enabled: has_layer,
            run: isolate_layers,
            journal: false,
        },
        CommandSpec {
            id: "select.transformSelection",
            label: "Transform Selection",
            menu: &["Select"],
            shortcut: None,
            params: r##"{"rect":[x0,y0,x1,y1]? (frame; default = selection bounds),"quad"|"corners":[[x,y]×4]? (where the frame's corners go: distort/perspective),"matrix":[a,b,c,d,e,f]?,"scaleX":%=100,"scaleY":%=100,"rotate":deg=0,"skewX":deg=0,"skewY":deg=0,"dx":px=0,"dy":px=0,"reference":"center|topLeft|top|topRight|left|right|bottomLeft|bottom|bottomRight"|[x,y]="center","style"|"mesh"|"grid"|"warp":… (warp, as edit.transform.warp),"interpolation":"bilinear|bicubic|nearest"="bilinear"}"##,
            enabled: has_selection,
            run: transform_selection,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sel_bounds(s: &Session) -> Option<Rect> {
        s.active().unwrap().doc.selection.as_ref().map(|s| s.content_bounds())
    }

    fn landscape(depth: u64) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 160, "height": 100, "depth": depth})).unwrap();
        s.edit("paint", |doc, active| {
            let l = doc.layer_mut(active.unwrap()).unwrap();
            let surf = l.surface_mut().unwrap();
            let n = surf.channels();
            for y in 0..100 {
                for x in 0..160 {
                    let (cx, cy) = (112.0f32, 35.0f32);
                    let crown = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() < 18.0;
                    let nz = (((x * 7919 + y * 104_729) % 97) as f32 / 97.0 - 0.5) * 0.25;
                    let c = if crown {
                        [0.1 + nz * 0.5, 0.35 + nz, 0.1]
                    } else if y < 60 {
                        let t = y as f32 / 60.0;
                        [0.25 + 0.4 * t, 0.5 + 0.3 * t, 0.9 + 0.05 * t]
                    } else {
                        [0.35 + nz, 0.3 + nz, 0.15]
                    };
                    let mut px = vec![1.0f32; n];
                    px[..3].copy_from_slice(&c);
                    surf.write_pixel(x, y, &px);
                }
            }
            Ok(())
        })
        .unwrap();
        s
    }

    #[test]
    fn sky_selects_the_sky_at_every_depth() {
        for depth in [8, 16, 32] {
            let mut s = landscape(depth);
            let r = s.execute("select.sky", json!({})).unwrap();
            assert_eq!(r["changed"], json!(true), "{depth}");
            let cov = r["coverage"].as_f64().unwrap();
            assert!(cov > 0.45 && cov < 0.62, "depth {depth}: coverage {cov}");
            let sel = s.active().unwrap().doc.selection.clone().unwrap();
            assert!(sel.sample_channel(10, 10, 0) > 0.9, "{depth}");
            assert!(sel.sample_channel(10, 90, 0) < 0.1, "{depth}");
            assert!(sel.sample_channel(112, 35, 0) < 0.1, "{depth}: tree");
            s.undo();
            assert!(s.active().unwrap().doc.selection.is_none());
        }
    }

    #[test]
    fn sky_without_sky_changes_nothing() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 40, "height": 40})).unwrap();
        s.execute("edit.fill", json!({"color": "#504020"})).ok();
        let r = s.execute("select.sky", json!({})).unwrap();
        assert_eq!(r["changed"], json!(false));
        assert!(s.active().unwrap().doc.selection.is_none());
    }

    #[test]
    fn isolate_layers_toggles() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 20, "height": 20})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        let active = s.active().unwrap().active_layer.unwrap();
        let r = s.execute("select.isolateLayers", json!({})).unwrap();
        assert_eq!(r["isolated"], json!(true));
        assert_eq!(r["layers"], json!([active.0]));
        let d = s.active().unwrap();
        let bg = d.doc.layers[0].id;
        assert!(isolation_shows(&d.doc, &d.isolated_layers, active));
        assert!(!isolation_shows(&d.doc, &d.isolated_layers, bg));
        assert_eq!(crate::inspect::document(d)["isolatedLayers"], json!([active.0]));
        let r = s.execute("select.isolateLayers", json!({})).unwrap();
        assert_eq!(r["isolated"], json!(false));
        assert!(s.active().unwrap().isolated_layers.is_empty());
    }

    fn rect_sel() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 100, "height": 100})).unwrap();
        s.execute("select.rect", json!({"x": 20, "y": 30, "width": 20, "height": 10})).unwrap();
        s
    }

    #[test]
    fn transform_selection_scales_rotates_and_moves() {
        let mut s = rect_sel();
        assert_eq!(sel_bounds(&s), Some(Rect::new(20, 30, 40, 40)));
        let px0 = s.active().unwrap().doc.layers[0].surface().unwrap().read_region(Rect::new(0, 0, 100, 100));
        s.execute("select.transformSelection", json!({"scaleX": 200, "scaleY": 200})).unwrap();
        let b = sel_bounds(&s).unwrap();
        assert!((b.x0 - 10).abs() <= 1 && (b.x1 - 50).abs() <= 1 && (b.y0 - 25).abs() <= 1 && (b.y1 - 45).abs() <= 1, "{b:?}");
        // Pixels untouched.
        assert_eq!(s.active().unwrap().doc.layers[0].surface().unwrap().read_region(Rect::new(0, 0, 100, 100)), px0);
        s.undo();
        assert_eq!(sel_bounds(&s), Some(Rect::new(20, 30, 40, 40)));
        s.execute("select.transformSelection", json!({"rotate": 90})).unwrap();
        let b = sel_bounds(&s).unwrap();
        assert!(b.width().abs_diff(10) <= 2 && b.height().abs_diff(20) <= 2, "{b:?}");
        s.undo();
        s.execute("select.transformSelection", json!({"dx": 30, "dy": -10, "reference": "topLeft"})).unwrap();
        assert_eq!(sel_bounds(&s), Some(Rect::new(50, 20, 70, 30)));
    }

    #[test]
    fn transform_selection_perspective_and_warp() {
        let mut s = rect_sel();
        s.execute("select.transformSelection", json!({"corners": [[25, 30], [35, 30], [45, 40], [15, 40]]})).unwrap();
        let sel = s.active().unwrap().doc.selection.clone().unwrap();
        assert!(sel.sample_channel(17, 39, 0) > 0.5 && sel.sample_channel(21, 31, 0) < 0.5);
        s.undo();
        let r = s.execute("select.transformSelection", json!({"style": "arc", "bend": 50})).unwrap();
        assert_eq!(r["changed"], json!(true));
        assert_ne!(sel_bounds(&s), Some(Rect::new(20, 30, 40, 40)));
        assert!(s.execute("select.transformSelection", json!({"scaleX": 0})).is_err());
    }

    #[test]
    fn transform_selection_needs_a_selection() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
        assert!(!s.is_enabled("select.transformSelection"));
    }
}

#[cfg(test)]
mod hang_tests {
    use super::*;

    #[test]
    fn transform_selection_with_extreme_params_terminates() {
        // Regression (panic_hunt): a huge off-canvas transform used to allocate a buffer spanning
        // the old + new bounds and hang. It must return quickly without crashing.
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 24, "height": 16})).unwrap();
        s.execute("select.rect", json!({"x": 1, "y": 1, "width": 6, "height": 5})).unwrap();
        // dx/dy move the selection far off-canvas; scaleX enlarges it — the old code allocated a
        // buffer spanning the original + moved bounds (~1e5 × 1e5) and hung.
        let r = s.execute("select.transformSelection", json!({"dx": 1e5, "dy": -1e5, "scaleX": 300.0}));
        assert!(r.is_ok(), "far transform returns without hanging: {r:?}");
        // The document is still usable.
        assert!(s.execute("select.all", json!({})).is_ok());
    }
}
