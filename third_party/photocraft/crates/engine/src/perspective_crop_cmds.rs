//! The Perspective Crop tool's commit: `image.perspectiveCrop`.
//!
//! Four corners (TL, TR, BR, BL in document px) mark a quad that is a rectangle seen in perspective
//! (a photographed document, a facade). The whole document is mapped through the homography taking
//! that quad onto an upright W x H rectangle, which becomes the new canvas, in one undo step:
//! every raster layer, layer mask, alpha channel and the Quick Mask is resampled through the map
//! (per destination tile, in parallel, bounded by the new canvas), the Background stays a
//! Background over the background colour where the quad reaches past the image, and pixels outside
//! the quad are deleted (Photoshop's Perspective Crop has no Delete Cropped Pixels option).
//!
//! Type, shape, fill and smart object layers are rasterized first: a projective map has no
//! editable equivalent for them. Vector masks, paths, notes, slices and guides follow the affine
//! map closest to the homography (exact when the quad is a parallelogram).

use photocraft_algo::transform::{Homography, Interp, convex_quad, quad_size, quad_to_rect, warp_surface_clipped};
use photocraft_color::PixelFormat;
use photocraft_doc::{Document, Layer, LayerContent, LayerId, Size};
use photocraft_geom::{Affine, Rect};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const CMD: &str = "image.perspectiveCrop";
/// The largest output side (the `file.new`, Canvas Size and Crop limit).
const MAX_SIDE: u32 = 300_000;
/// Corner coordinates beyond this are refused (far past any canvas).
const MAX_COORD: f64 = 1e7;
/// The highest resolution (ppi) the crop may set, as for Image Size.
const MAX_RESOLUTION: f64 = 10_000.0;

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// `corners`: four `[x, y]` points, TL, TR, BR, BL, forming a convex quad clockwise on screen.
fn corners(p: &Value) -> Result<[[f64; 2]; 4]> {
    let shape = || bad("`corners` must be four [x, y] points: top-left, top-right, bottom-right, bottom-left");
    let list = p.get("corners").and_then(Value::as_array).filter(|a| a.len() == 4).ok_or_else(shape)?;
    let mut q = [[0.0; 2]; 4];
    for (slot, c) in q.iter_mut().zip(list) {
        let xy = c.as_array().filter(|a| a.len() == 2).ok_or_else(shape)?;
        let (Some(x), Some(y)) = (xy.first().and_then(Value::as_f64), xy.get(1).and_then(Value::as_f64)) else { return Err(shape()) };
        if !(x.is_finite() && y.is_finite()) || x.abs() > MAX_COORD || y.abs() > MAX_COORD {
            return Err(bad(format!("corner [{x}, {y}] must be finite and within ±{MAX_COORD} px")));
        }
        *slot = [x, y];
    }
    if !convex_quad(&q) {
        return Err(bad(
            "the corners must form a convex quadrilateral at least 1 px in area, in order top-left, top-right, bottom-right, bottom-left (clockwise on screen)",
        ));
    }
    Ok(q)
}

/// Output size: `width`/`height` in whole px; one alone keeps the quad's proportions; neither
/// takes the quad's mean side lengths ([`quad_size`]).
fn out_size(p: &Value, q: &[[f64; 2]; 4]) -> Result<(u32, u32)> {
    let side = |key: &str| -> Result<Option<f64>> {
        match p.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(v) => v
                .as_f64()
                .filter(|n| n.is_finite() && (1.0..=f64::from(MAX_SIDE)).contains(&n.round()))
                .map(|n| Some(n.round()))
                .ok_or_else(|| bad(format!("`{key}` = {v} must be a number of pixels from 1 to {MAX_SIDE}"))),
        }
    };
    let (qw, qh) = quad_size(q);
    let (w, h) = match (side("width")?, side("height")?) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None) => (w, w * qh / qw),
        (None, Some(h)) => (h * qw / qh, h),
        (None, None) => (qw, qh),
    };
    let px = |v: f64| -> Result<u32> {
        let v = v.round().max(1.0);
        if !v.is_finite() || v > f64::from(MAX_SIDE) {
            return Err(bad(format!("the output would be {v} px on a side; the limit is {MAX_SIDE}")));
        }
        Ok(v as u32)
    };
    Ok((px(w)?, px(h)?))
}

fn resolution(p: &Value) -> Result<Option<f32>> {
    match p.get("resolution") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let r = v.as_f64().filter(|r| r.is_finite() && (1.0..=MAX_RESOLUTION).contains(r));
            Ok(Some(r.ok_or_else(|| bad(format!("`resolution` = {v} must be a number of pixels per inch from 1 to {MAX_RESOLUTION}")))? as f32))
        }
    }
}

fn interpolation(p: &Value) -> Result<Interp> {
    match p.get("interpolation") {
        None | Some(Value::Null) => Ok(Interp::Bicubic),
        Some(Value::String(s)) if ["nearest", "nearestNeighbor", "bilinear", "bicubic"].contains(&s.as_str()) => Ok(Interp::parse(s)),
        Some(v) => Err(bad(format!("`interpolation` = {v} must be \"bicubic\", \"bilinear\" or \"nearest\""))),
    }
}

/// The affine map closest to the crop for geometry that can't take a homography (vector masks,
/// paths, notes, guides): the quad's centroid goes to the output's centre and its mean horizontal
/// and vertical edges to the output's sides. Exact when the quad is a parallelogram.
fn nearest_affine(q: &[[f64; 2]; 4], w: f64, h: f64) -> Affine {
    let [tl, tr, br, bl] = *q;
    let u = [(tr[0] - tl[0] + br[0] - bl[0]) / 2.0, (tr[1] - tl[1] + br[1] - bl[1]) / 2.0];
    let v = [(bl[0] - tl[0] + br[0] - tr[0]) / 2.0, (bl[1] - tl[1] + br[1] - tr[1]) / 2.0];
    let det = u[0] * v[1] - v[0] * u[1];
    if !det.is_finite() || det.abs() < 1e-12 {
        return Affine::default();
    }
    let (m00, m01, m10, m11) = (w * v[1] / det, -w * v[0] / det, -h * u[1] / det, h * u[0] / det);
    let c = [(tl[0] + tr[0] + br[0] + bl[0]) / 4.0, (tl[1] + tr[1] + br[1] + bl[1]) / 4.0];
    Affine { m: [m00, m10, m01, m11, w / 2.0 - (m00 * c[0] + m01 * c[1]), h / 2.0 - (m10 * c[0] + m11 * c[1])] }
}

/// A one-channel surface (mask, alpha channel, Quick Mask) through `h`, kept inside `clip`; what
/// the map uncovers takes the surface's default (a mask's reveal or hide).
fn warp_gray(s: &Surface, h: &Homography, interp: Interp, clip: Rect) -> Result<Surface> {
    const WHAT: &str = "a perspective crop of a mask or channel";
    let default = s.default_pixel().first().copied().unwrap_or(0.0);
    let fmt = s.format();
    let mut out = Surface::with_default(fmt, &[default]);
    let src = s.content_bounds();
    if src.is_empty() {
        return Ok(out);
    }
    // Content with alpha 1 over a transparent outside, so the edges blend into the default.
    let v = crate::allocation::read_region(s, src, WHAT)?;
    let mut lifted = crate::allocation::filled(v.len().checked_mul(2).ok_or_else(|| bad("the mask is too large"))?, 1.0, WHAT)?;
    for (g, px) in v.iter().zip(lifted.as_chunks_mut::<2>().0) {
        px[0] = *g;
    }
    let mut tmp = Surface::new(PixelFormat::new(fmt.mode, fmt.sample, true));
    tmp.try_write_region(src, &lifted)?;
    let warped = warp_surface_clipped(&tmp, src, h, interp, clip)?;
    let b = warped.content_bounds().intersect(&clip);
    if !b.is_empty() {
        let px = crate::allocation::read_region(&warped, b, WHAT)?;
        let flat: Vec<f32> = px.as_chunks::<2>().0.iter().map(|[g, a]| g * a + default * (1.0 - a)).collect();
        out.try_write_region(b, &flat)?;
    }
    out.prune();
    Ok(out)
}

/// Maps every layer's pixels and masks through `h` into `clip`.
/// `background_id` is the id of the document's Background, if it has one.
fn warp_layers(layers: &mut [Layer], h: &Homography, interp: Interp, clip: Rect, bg: &[f32], background_id: Option<LayerId>) -> Result<()> {
    for l in layers {
        crate::allocation::checkpoint("a perspective crop")?;
        let background = background_id == Some(l.id);
        match &mut l.content {
            LayerContent::Group(g) => warp_layers(&mut g.children, h, interp, clip, bg, None)?,
            LayerContent::Raster(surf) => {
                let warped = warp_surface_clipped(surf, surf.content_bounds(), h, interp, clip)?;
                *surf = if background {
                    // The Background stays opaque: the mapped pixels over the background colour.
                    let mut base = Surface::new(surf.format());
                    base.fill_rect(clip, bg);
                    crate::transform_cmds::composite_over(&mut base, &warped);
                    base.prune();
                    base
                } else {
                    warped
                };
            }
            // Adjustment layers have no pixels; the rest was rasterized before.
            _ => {}
        }
        if let Some(m) = l.mask.as_mut() {
            m.surface = warp_gray(&m.surface, h, interp, clip)?;
        }
        l.fill_cache = None;
    }
    Ok(())
}

fn apply(doc: &mut Document, q: &[[f64; 2]; 4], h: &Homography, size: (u32, u32), interp: Interp, bg: [f32; 4]) -> Result<()> {
    let (w, ht) = size;
    let clip = Rect::from_xywh(0, 0, w, ht);
    let bg = photocraft_raster::from_rgba(&doc.pixel_format(), bg);
    let background_id = doc.background().map(|l| l.id);
    warp_layers(&mut doc.layers, h, interp, clip, &bg, background_id)?;
    for ch in doc.channels.iter_mut().chain(doc.quick_mask.as_mut()) {
        ch.surface = warp_gray(&ch.surface, h, interp, clip)?;
    }
    doc.selection = None;
    crate::canvas_geom::transform_geometry(doc, &nearest_affine(q, f64::from(w), f64::from(ht)));
    doc.size = Size::new(w, ht);
    crate::canvas_geom::refresh(doc, crate::canvas_geom::Refresh::Shapes);
    Ok(())
}

fn perspective_crop(s: &mut Session, p: &Value) -> Result<Value> {
    let q = corners(p)?;
    let (w, h) = out_size(p, &q)?;
    let res = resolution(p)?;
    let interp = interpolation(p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    crate::image_cmds::check_resample_budget(CMD, &d.doc, w, h, "")?;
    let rasterize = d
        .doc
        .walk()
        .iter()
        .any(|(_, _, l)| matches!(l.content, LayerContent::Text(_) | LayerContent::Shape(_) | LayerContent::Fill(_) | LayerContent::Smart(_)));
    let map = quad_to_rect(&q, f64::from(w), f64::from(h)).ok_or_else(|| bad("the corners don't define a perspective"))?;
    let bg = s.tools.background;
    crate::analysis_cmds::compound(s, "Perspective Crop", |s| {
        if rasterize {
            crate::extra_cmds::rasterize_all(s)?;
        }
        s.edit("Perspective Crop", |doc, _| {
            apply(doc, &q, &map, (w, h), interp, bg)?;
            if let Some(r) = res {
                doc.resolution_dpi = r;
            }
            Ok(())
        })
    })?;
    let dpi = s.active().map(|d| d.doc.resolution_dpi);
    Ok(json!({"width": w, "height": h, "resolution": dpi, "corners": q}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Perspective Crop",
        menu: &[],
        shortcut: None,
        params: r##"{"corners":[[x,y],[x,y],[x,y],[x,y]] (top-left, top-right, bottom-right, bottom-left; a convex quad, document px),"width":px,"height":px,"resolution":ppi,"interpolation":"bicubic|bilinear|nearest"="bicubic"} → {width,height,resolution,corners} (the quad is mapped onto an upright width x height canvas, all layers, one undo step; no width/height: the quad's mean side lengths, one alone keeps its proportions; type, shape, fill and smart object layers are rasterized)"##,
        enabled: has_doc,
        run: perspective_crop,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, SampleType};
    use photocraft_doc::Color;

    /// A document whose Background shows a W x H ramp (red = u / W, green = v / H) seen in
    /// perspective through the quad `q`: pixel (x, y) holds the ramp at the rectified point.
    fn skewed_ramp(depth: SampleType, q: &[[f64; 2]; 4], w: f64, h: f64) -> Session {
        let to_doc = Homography::rect_to_quad([0.0, 0.0, w, h], *q).unwrap();
        let to_out = to_doc.inverse().unwrap();
        let mut doc = Document::with_background("ramp", Size::new(160, 120), ColorMode::Rgb, depth, Color::WHITE);
        let fmt = doc.pixel_format();
        let mut px = Vec::new();
        for y in 0..120 {
            for x in 0..160 {
                let (u, v) = to_out.apply(x as f64 + 0.5, y as f64 + 0.5);
                px.extend(photocraft_raster::from_rgba(&fmt, [(u / w).clamp(0.0, 1.0) as f32, (v / h).clamp(0.0, 1.0) as f32, 0.5, 1.0]));
            }
        }
        doc.layers[0].surface_mut().unwrap().write_region(Rect::from_xywh(0, 0, 160, 120), &px);
        let mut s = Session::new();
        s.add_document(doc, None);
        s
    }

    /// The ramp comes out straight: every output pixel holds the ramp value of its own position
    /// (within 1 px), at 8 and 16 bits, and one undo restores the document.
    #[test]
    fn perspective_crop_rectifies_a_skewed_ramp() {
        let q = [[30.0, 12.0], [140.0, 25.0], [128.0, 110.0], [14.0, 95.0]];
        let (w, h) = (100.0, 80.0);
        for depth in [SampleType::U8, SampleType::U16] {
            let mut s = skewed_ramp(depth, &q, w, h);
            let r = s.execute(CMD, json!({"corners": q, "width": w, "height": h})).unwrap();
            assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(100), Some(80)));
            let doc = &s.active().unwrap().doc;
            assert_eq!((doc.size.width, doc.size.height), (100, 80));
            let surf = doc.layers[0].surface().unwrap();
            let (fmt, n) = (surf.format(), surf.channels());
            let out = surf.read_region(Rect::from_xywh(0, 0, 100, 80));
            for y in (2..78).step_by(5) {
                for x in (2..98).step_by(5) {
                    let i = (y * 100 + x) * n;
                    let p = photocraft_raster::to_rgba(&fmt, &out[i..i + n]);
                    let (u, v) = (p[0] as f64 * w, p[1] as f64 * h);
                    assert!((u - (x as f64 + 0.5)).abs() < 1.0 && (v - (y as f64 + 0.5)).abs() < 1.0, "{depth:?} ({x}, {y}) shows ({u:.2}, {v:.2})");
                }
            }
            s.execute("edit.undo", json!({})).unwrap();
            assert_eq!(s.active().unwrap().doc.size, Size::new(160, 120), "one undo step");
        }
    }

    /// Bad quads and params are refused with an error, and the document is untouched.
    #[test]
    fn perspective_crop_rejects_bad_quads() {
        let mut s = skewed_ramp(SampleType::U8, &[[0.0, 0.0], [160.0, 0.0], [160.0, 120.0], [0.0, 120.0]], 160.0, 120.0);
        for p in [
            json!({}),
            json!({"corners": [[0, 0], [10, 0], [10, 10]]}),
            json!({"corners": [[0, 0], [50, 0], [100, 0], [0, 50]]}),
            json!({"corners": [[0, 0], [50, 50], [50, 0], [0, 50]]}),
            json!({"corners": [[0, 0], [0, 50], [50, 50], [50, 0]]}),
            json!({"corners": [[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]}),
            json!({"corners": [[0, 0], [1e300, 0], [50, 50], [0, 50]]}),
            json!({"corners": [[0, 0], ["a", 0], [50, 50], [0, 50]]}),
            json!({"corners": [[0, 0], [50, 0], [50, 50], [0, 50]], "width": 0}),
            json!({"corners": [[0, 0], [50, 0], [50, 50], [0, 50]], "width": 1e9, "height": 10}),
            json!({"corners": [[0, 0], [50, 0], [50, 50], [0, 50]], "resolution": -3}),
            json!({"corners": [[0, 0], [50, 0], [50, 50], [0, 50]], "interpolation": 3}),
        ] {
            assert!(s.execute(CMD, p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.active().unwrap().doc.size, Size::new(160, 120));
    }
}
