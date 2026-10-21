//! Color Sampler points (#1046): numbered colour readings pinned to canvas pixels and listed in
//! the Info panel, so a fixed position can be compared instead of chasing the live pointer.
//!
//! Points are per-document view state ([`crate::DocState::color_samplers`]): they belong to their
//! document, are discarded when it closes, and are not saved to `.pcraft` or PSD. Placing, moving
//! and removing them changes no pixels and adds no History step, so they are not journaled. Each
//! point reads one composite pixel (no averaging); its values match the Info panel's pointer
//! readout (RGB 0–255 and the naive device CMYK percentages, without an ICC profile).

use photocraft_doc::Document;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{DocState, EngineError, Result, Session};

/// Photoshop's Color Sampler limit: at most 10 points per document.
pub const MAX_SAMPLERS: usize = 10;

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// Naive device CMYK (no ICC profile), in percent, as the Info panel shows it. 32-bit samples can
/// lie outside 0..1, so each channel is clamped first (NaN reads as 0), like the RGB readout.
pub fn device_cmyk(r: f32, g: f32, b: f32) -> [i32; 4] {
    let unit = |v: f32| if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
    let (r, g, b) = (unit(r), unit(g), unit(b));
    let k = 1.0 - r.max(g).max(b);
    let f = |v: f32| if k >= 1.0 { 0.0 } else { (1.0 - v - k) / (1.0 - k) };
    [f(r), f(g), f(b), k].map(|v| (v * 100.0).round() as i32)
}

/// The Info-panel values for a sample point at document pixel `pos`: composite RGB 0–255 and
/// naive device CMYK percentages. One composite pixel, no averaging.
pub fn readout(doc: &Document, pos: [f64; 2]) -> ([i32; 3], [i32; 4]) {
    let rgba = crate::sample_cmds::sample_color(doc, None, pos[0], pos[1], 1, crate::sample_cmds::SampleLayers::All);
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as i32;
    ([q(rgba[0]), q(rgba[1]), q(rgba[2])], device_cmyk(rgba[0], rgba[1], rgba[2]))
}

/// One point as the list command reports it.
pub fn point_json(doc: &Document, index: usize, pos: [f64; 2]) -> Value {
    let (rgb, cmyk) = readout(doc, pos);
    json!({"index": index, "number": index + 1, "position": [pos[0], pos[1]], "rgb": rgb, "cmyk": cmyk})
}

fn list_json(doc: &Document, points: &[[f64; 2]]) -> Value {
    json!({
        "max": MAX_SAMPLERS,
        "samplers": points.iter().enumerate().map(|(i, p)| point_json(doc, i, *p)).collect::<Vec<_>>(),
    })
}

fn coord(cmd: &str, p: &Value, key: &str) -> Result<f64> {
    p.get(key).and_then(Value::as_f64).filter(|v| v.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a finite number (document pixels)")))
}

/// The pixel a point at document coordinate `(x, y)` sits on, if it is inside the canvas. Whole
/// pixels only: the click's fractional position floors to the pixel it lands on.
fn pixel(doc: &Document, x: f64, y: f64) -> Option<[f64; 2]> {
    let (px, py) = (x.floor(), y.floor());
    if px < f64::from(i32::MIN) || px > f64::from(i32::MAX) || py < f64::from(i32::MIN) || py > f64::from(i32::MAX) {
        return None;
    }
    doc.bounds().contains(px as i32, py as i32).then_some([px, py])
}

fn index_param(cmd: &str, p: &Value) -> Result<usize> {
    p.get("index").and_then(Value::as_u64).and_then(|n| usize::try_from(n).ok()).ok_or_else(|| bad(cmd, "pass `index` (0-based)"))
}

/// Apply a Color Sampler change to the active document: view state, so no history step and no
/// pixels. Bumps the revision (keeping a clean document clean) so UIs refresh.
fn change<R>(s: &mut Session, f: impl FnOnce(&mut DocState) -> Result<R>) -> Result<R> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let r = f(st)?;
    let clean = st.saved_revision == st.revision;
    st.revision += 1;
    if clean {
        st.saved_revision = st.revision;
    }
    st.last_damage = Some(photocraft_geom::Rect::EMPTY);
    Ok(r)
}

fn add(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "view.colorSamplers.add";
    let x = coord(CMD, p, "x")?;
    let y = coord(CMD, p, "y")?;
    change(s, |st| {
        let pos = pixel(&st.doc, x, y).ok_or_else(|| EngineError::Other("the sample point is outside the canvas".into()))?;
        if st.color_samplers.len() >= MAX_SAMPLERS {
            return Err(EngineError::Other(format!("Color Sampler limit reached: at most {MAX_SAMPLERS} points per document")));
        }
        st.color_samplers.push(pos);
        let i = st.color_samplers.len() - 1;
        Ok(json!({"index": i, "count": st.color_samplers.len(), "point": point_json(&st.doc, i, pos)}))
    })
}

fn move_point(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "view.colorSamplers.move";
    let i = index_param(CMD, p)?;
    let x = coord(CMD, p, "x")?;
    let y = coord(CMD, p, "y")?;
    change(s, |st| {
        let n = st.color_samplers.len();
        if i >= n {
            return Err(bad(CMD, format!("no sample point {i} (the document has {n})")));
        }
        let pos = pixel(&st.doc, x, y).ok_or_else(|| EngineError::Other("the sample point is outside the canvas".into()))?;
        st.color_samplers[i] = pos;
        Ok(json!({"index": i, "point": point_json(&st.doc, i, pos)}))
    })
}

fn delete(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "view.colorSamplers.delete";
    let i = index_param(CMD, p)?;
    change(s, |st| {
        let n = st.color_samplers.len();
        if i >= n {
            return Err(bad(CMD, format!("no sample point {i} (the document has {n})")));
        }
        st.color_samplers.remove(i);
        Ok(json!({"deleted": 1, "remaining": st.color_samplers.len()}))
    })
}

fn clear(s: &mut Session, _p: &Value) -> Result<Value> {
    change(s, |st| {
        let n = st.color_samplers.len();
        st.color_samplers.clear();
        Ok(json!({"deleted": n, "remaining": 0}))
    })
}

fn list(s: &mut Session, _p: &Value) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    Ok(list_json(&st.doc, &st.color_samplers))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "view.colorSamplers.add",
            label: "Add Color Sampler",
            menu: &[],
            shortcut: None,
            params: r##"{"x":f64,"y":f64} → the new point {index, number, position, rgb, cmyk}; errors at the 10-point limit or off the canvas"##,
            enabled: has_doc,
            run: add,
            journal: false,
        },
        CommandSpec {
            id: "view.colorSamplers.move",
            label: "Move Color Sampler",
            menu: &[],
            shortcut: None,
            params: r##"{"index":n,"x":f64,"y":f64}"##,
            enabled: has_doc,
            run: move_point,
            journal: false,
        },
        CommandSpec {
            id: "view.colorSamplers.delete",
            label: "Delete Color Sampler",
            menu: &[],
            shortcut: None,
            params: r##"{"index":n}"##,
            enabled: has_doc,
            run: delete,
            journal: false,
        },
        CommandSpec {
            id: "view.colorSamplers.clear",
            label: "Clear Color Samplers",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: has_doc,
            run: clear,
            journal: false,
        },
        CommandSpec {
            id: "view.colorSamplers.list",
            label: "Color Samplers",
            menu: &[],
            shortcut: None,
            params: r##"{} → {max, samplers:[{index, number, position, rgb, cmyk}]}"##,
            enabled: has_doc,
            run: list,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_doc::{Layer, Size};
    use photocraft_geom::Rect;

    /// A 100×80 white document.
    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 100, "height": 80, "background": "white"})).unwrap();
        s
    }

    fn list(s: &mut Session) -> Value {
        s.execute("view.colorSamplers.list", json!({})).unwrap()
    }

    fn points(s: &mut Session) -> Vec<Value> {
        list(s)["samplers"].as_array().cloned().unwrap_or_default()
    }

    #[test]
    fn add_reads_the_composite_pixel_and_lists_it() {
        let mut s = session();
        let r = s.execute("view.colorSamplers.add", json!({"x": 5, "y": 6})).unwrap();
        assert_eq!(r["index"], 0);
        assert_eq!(r["count"], 1);
        assert_eq!(r["point"]["number"], 1);
        assert_eq!(r["point"]["position"], json!([5.0, 6.0]));
        assert_eq!(r["point"]["rgb"], json!([255, 255, 255]));
        assert_eq!(r["point"]["cmyk"], json!([0, 0, 0, 0]));
        let l = list(&mut s);
        assert_eq!(l["max"], 10);
        assert_eq!(l["samplers"].as_array().unwrap().len(), 1);
        assert_eq!(l["samplers"][0]["position"], json!([5.0, 6.0]));
    }

    #[test]
    fn fractional_clicks_floor_to_the_pixel() {
        let mut s = session();
        let r = s.execute("view.colorSamplers.add", json!({"x": 5.9, "y": 6.1})).unwrap();
        assert_eq!(r["point"]["position"], json!([5.0, 6.0]));
    }

    #[test]
    fn add_reads_an_actual_painted_pixel() {
        // A black pixel at (3, 3) of an otherwise white document.
        let mut s = session();
        s.edit("paint", |doc, _| {
            let surf = doc.layers[0].surface_mut().unwrap();
            let black = photocraft_raster::from_rgba(&surf.format(), [0.0, 0.0, 0.0, 1.0]);
            surf.fill_rect(Rect::new(3, 3, 4, 4), &black);
            Ok(())
        })
        .unwrap();
        let r = s.execute("view.colorSamplers.add", json!({"x": 3, "y": 3})).unwrap();
        assert_eq!(r["point"]["rgb"], json!([0, 0, 0]));
        assert_eq!(r["point"]["cmyk"], json!([0, 0, 0, 100]));
        // Painting over the point changes its readout (the list is computed live).
        s.edit("paint", |doc, _| {
            let surf = doc.layers[0].surface_mut().unwrap();
            let red = photocraft_raster::from_rgba(&surf.format(), [1.0, 0.0, 0.0, 1.0]);
            surf.fill_rect(Rect::new(3, 3, 4, 4), &red);
            Ok(())
        })
        .unwrap();
        assert_eq!(points(&mut s)[0]["rgb"], json!([255, 0, 0]));
    }

    #[test]
    fn ten_points_then_the_limit() {
        let mut s = session();
        for i in 0..10 {
            s.execute("view.colorSamplers.add", json!({"x": i, "y": 0})).unwrap();
        }
        assert_eq!(points(&mut s).len(), 10);
        let err = s.execute("view.colorSamplers.add", json!({"x": 20, "y": 0})).unwrap_err().to_string();
        assert!(err.contains("limit"), "{err}");
        assert_eq!(points(&mut s).len(), 10);
    }

    #[test]
    fn move_delete_and_clear() {
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 5, "y": 5})).unwrap();
        s.execute("view.colorSamplers.add", json!({"x": 50, "y": 50})).unwrap();
        let r = s.execute("view.colorSamplers.move", json!({"index": 0, "x": 10, "y": 20})).unwrap();
        assert_eq!(r["point"]["position"], json!([10.0, 20.0]));
        assert_eq!(points(&mut s)[0]["position"], json!([10.0, 20.0]));
        s.execute("view.colorSamplers.delete", json!({"index": 1})).unwrap();
        assert_eq!(points(&mut s).len(), 1);
        // The remaining point is the moved #1.
        assert_eq!(points(&mut s)[0]["position"], json!([10.0, 20.0]));
        let r = s.execute("view.colorSamplers.clear", json!({})).unwrap();
        assert_eq!(r["deleted"], 1);
        assert!(points(&mut s).is_empty());
    }

    #[test]
    fn placing_and_moving_add_no_history_and_change_no_pixels() {
        let mut s = session();
        let before = s.active().unwrap().doc.layers[0].surface().unwrap().pixel(5, 5);
        let past = s.active().unwrap().history.past_len();
        s.execute("view.colorSamplers.add", json!({"x": 5, "y": 5})).unwrap();
        s.execute("view.colorSamplers.move", json!({"index": 0, "x": 9, "y": 9})).unwrap();
        s.execute("view.colorSamplers.delete", json!({"index": 0})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), past, "no history steps");
        assert_eq!(s.active().unwrap().doc.layers[0].surface().unwrap().pixel(5, 5), before, "no pixels changed");
        assert!(!s.active().unwrap().is_dirty(), "a clean document stays clean");
    }

    #[test]
    fn points_belong_to_their_document() {
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 5, "y": 5})).unwrap();
        s.execute("file.new", json!({"width": 40, "height": 40, "background": "white"})).unwrap();
        assert!(points(&mut s).is_empty(), "the new document has no points");
        s.execute("document.activate", json!({"document": 0})).unwrap();
        assert_eq!(points(&mut s).len(), 1, "the first document kept its point");
        // Closing the second document leaves the first's points alone.
        s.execute("file.close", json!({"document": 1})).unwrap();
        assert_eq!(points(&mut s).len(), 1);
        // Closing the first discards its points with it.
        s.execute("file.close", json!({"document": 0})).unwrap();
        assert!(s.execute("view.colorSamplers.list", json!({})).is_err(), "no document open");
    }

    #[test]
    fn a_crop_drops_points_outside_the_new_canvas() {
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 5, "y": 5})).unwrap();
        s.execute("view.colorSamplers.add", json!({"x": 50, "y": 50})).unwrap();
        s.execute("image.crop", json!({"x": 0, "y": 0, "width": 10, "height": 10})).unwrap();
        let p = points(&mut s);
        assert_eq!(p.len(), 1, "the outside point was removed");
        assert_eq!(p[0]["position"], json!([5.0, 5.0]));
        assert_eq!(s.take_samplers_removed(), 1, "one removed point reported");
        assert_eq!(s.take_samplers_removed(), 0, "reported once");
    }

    #[test]
    fn a_crop_moves_points_with_their_pixels() {
        // A crop from a non-zero origin moves the canvas: the point follows its pixel.
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 50, "y": 40})).unwrap();
        s.execute("image.crop", json!({"x": 40, "y": 30, "width": 20, "height": 20})).unwrap();
        let p = points(&mut s);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0]["position"], json!([10.0, 10.0]), "moved by (-40, -30)");
        assert_eq!(s.take_samplers_removed(), 0, "nothing removed");
    }

    #[test]
    fn a_canvas_size_shift_moves_points() {
        // Canvas Size anchored bottom-right shifts the canvas origin by (+20, +20).
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 30, "y": 30})).unwrap();
        s.execute("image.canvasSize", json!({"width": 120, "height": 100, "anchor": "bottomRight"})).unwrap();
        let p = points(&mut s);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0]["position"], json!([50.0, 50.0]), "moved by the canvas shift");
    }

    #[test]
    fn a_right_angle_rotation_moves_points() {
        // 90° clockwise about the centre: the doc centre stays put.
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 50, "y": 40})).unwrap();
        s.execute("image.rotation.arbitrary", json!({"angle": 90, "direction": "cw"})).unwrap();
        let p = points(&mut s);
        assert_eq!(p.len(), 1);
        let pos = p[0]["position"].as_array().unwrap();
        let (x, y) = (pos[0].as_f64().unwrap(), pos[1].as_f64().unwrap());
        // The new canvas is 80×100, so the centre is (40, 50).
        assert!((x - 40.0).abs() < 1e-6 && (y - 50.0).abs() < 1e-6, "{pos:?}");
    }

    #[test]
    fn a_rotated_crop_moves_points() {
        // The crop frame is centred on the point, so the point lands at the new canvas centre.
        let mut s = session();
        s.execute("view.colorSamplers.add", json!({"x": 50, "y": 40})).unwrap();
        s.execute("image.crop", json!({"x": 40, "y": 30, "width": 20, "height": 20, "angle": 90})).unwrap();
        let p = points(&mut s);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0]["position"], json!([10.0, 10.0]));
    }

    #[test]
    fn bad_params_and_no_document_are_errors_not_panics() {
        let mut s = Session::new();
        for id in ["view.colorSamplers.add", "view.colorSamplers.move", "view.colorSamplers.delete", "view.colorSamplers.clear", "view.colorSamplers.list"] {
            assert!(s.execute(id, json!({})).is_err(), "{id} without a document");
        }
        let mut s = session();
        for (id, p) in [
            ("view.colorSamplers.add", json!({})),
            ("view.colorSamplers.add", json!({"x": "a", "y": 1})),
            ("view.colorSamplers.add", json!({"x": 1e300, "y": 1})),
            ("view.colorSamplers.add", json!({"x": 5, "y": f64::NAN})),
            ("view.colorSamplers.add", json!({"x": -1, "y": 5})),
            ("view.colorSamplers.add", json!({"x": 5, "y": 1000})),
            ("view.colorSamplers.move", json!({"index": 0, "x": 1, "y": 1})),
            ("view.colorSamplers.delete", json!({"index": 0})),
            ("view.colorSamplers.delete", json!({})),
            ("view.colorSamplers.move", json!({"index": 4_294_967_296u64, "x": 1, "y": 1})),
        ] {
            assert!(matches!(s.execute(id, p.clone()), Err(EngineError::BadParams { .. }) | Err(EngineError::Other(_))), "{id} {p}");
        }
        // An off-canvas point is refused and places nothing.
        assert!(points(&mut s).is_empty());
    }

    #[test]
    fn device_cmyk_matches_the_info_panel_formula() {
        assert_eq!(device_cmyk(1.0, 1.0, 1.0), [0, 0, 0, 0]);
        assert_eq!(device_cmyk(0.0, 0.0, 0.0), [0, 0, 0, 100]);
        assert_eq!(device_cmyk(1.0, 0.0, 0.0), [0, 100, 100, 0]);
        assert_eq!(device_cmyk(0.5, 0.5, 0.5), [0, 0, 0, 50]);
    }

    /// 32-bit samples can lie outside 0..1 (or be NaN); the CMYK readout clamps them like the RGB
    /// readout does instead of reporting K:-50% or C:120% (#4191).
    #[test]
    fn device_cmyk_clamps_out_of_range_and_nan_samples() {
        assert_eq!(device_cmyk(1.5, 1.5, 1.5), [0, 0, 0, 0]);
        assert_eq!(device_cmyk(2.0, 0.0, 0.0), [0, 100, 100, 0]);
        let in_range = |c: [i32; 4]| c.iter().all(|v| (0..=100).contains(v));
        for (r, g, b) in [(-0.2, 0.5, 0.5), (-1.0, -1.0, -1.0), (f32::NAN, 0.5, 0.5), (f32::NAN, f32::NAN, f32::NAN), (f32::INFINITY, f32::NEG_INFINITY, 0.5)] {
            let c = device_cmyk(r, g, b);
            assert!(in_range(c), "{r} {g} {b} -> {c:?}");
        }
        assert_eq!(device_cmyk(f32::NAN, f32::NAN, f32::NAN), [0, 0, 0, 100]);
    }

    #[test]
    fn hdr_pixel_on_a_32_bit_document_reads_in_range() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8, "depth": 32, "background": "white"})).unwrap();
        s.edit("paint", |doc, _| {
            let surf = doc.layers[0].surface_mut().unwrap();
            let hdr = photocraft_raster::from_rgba(&surf.format(), [2.0, 2.0, 2.0, 1.0]);
            surf.fill_rect(Rect::new(3, 3, 4, 4), &hdr);
            Ok(())
        })
        .unwrap();
        let r = s.execute("view.colorSamplers.add", json!({"x": 3, "y": 3})).unwrap();
        assert_eq!(r["point"]["rgb"], json!([255, 255, 255]));
        assert_eq!(r["point"]["cmyk"], json!([0, 0, 0, 0]));
    }

    #[test]
    fn points_work_at_every_depth_and_mode() {
        for (mode, depth) in [("rgb", 8), ("rgb", 16), ("rgb", 32), ("grayscale", 8), ("cmyk", 8), ("lab", 8)] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 20, "height": 20, "mode": mode, "depth": depth, "background": "white"})).unwrap();
            let r = s.execute("view.colorSamplers.add", json!({"x": 10, "y": 10})).unwrap();
            let rgb = r["point"]["rgb"].as_array().unwrap();
            assert_eq!(rgb.len(), 3);
            assert!(rgb.iter().all(|v| v.as_i64().is_some_and(|n| (0..=255).contains(&n))), "{mode} {depth}: {rgb:?}");
        }
    }

    /// The mode and depth only affect the composite the readout samples, not the point model.
    #[test]
    fn document_helpers_are_not_disturbed() {
        let mut d = Document::new("t", Size::new(4, 4), ColorMode::Rgb, SampleType::U8);
        d.layers.push(Layer::raster("Background", d.pixel_format()));
        assert_eq!(pixel(&d, 0.0, 0.0), Some([0.0, 0.0]));
        assert_eq!(pixel(&d, 3.9, 3.9), Some([3.0, 3.0]));
        assert_eq!(pixel(&d, 4.0, 0.0), None);
    }
}
