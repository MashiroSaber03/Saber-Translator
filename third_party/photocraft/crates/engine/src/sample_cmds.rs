//! Eyedropper sampling (#1649): Photoshop's Sample Size (Point Sample, or the average of a
//! 3×3 … 101×101 square centred on the clicked pixel) and Sample (which layers the colour comes
//! from). The Eyedropper tool, the painting tools' ⌥-click eyedropper and the Info panel read
//! colours through [`sample_color`], so they share one setting, as in Photoshop.
//!
//! Averages are taken in the document's own colour model and bit depth: the composite under the
//! square is converted to the document's channels (Gray, Lab; CMYK documents are composited in
//! RGB, so their composite RGB is averaged, see [`averaging_model`]), averaged weighted by alpha,
//! rounded to the document's depth (8 and 16 bit; 32 bit keeps the float), then returned as the
//! straight-alpha RGBA the compositor and the tool colours use.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// The Sample Size choices: 1 = Point Sample, otherwise the side of the averaged square.
pub const SAMPLE_SIZES: [u32; 7] = [1, 3, 5, 11, 31, 51, 101];

/// Which layers the Eyedropper samples (its options bar's Sample menu).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SampleLayers {
    /// The active layer alone.
    Current,
    /// The active layer and everything below it.
    CurrentAndBelow,
    /// The whole visible composite (Photoshop's default).
    #[default]
    All,
    /// [`Self::CurrentAndBelow`] with adjustment layers left out.
    CurrentAndBelowNoAdjustments,
    /// [`Self::All`] with adjustment layers left out.
    AllNoAdjustments,
}

impl SampleLayers {
    /// Every choice with its param id, in the options bar's order.
    pub const ALL: [(SampleLayers, &'static str); 5] = [
        (SampleLayers::Current, "current"),
        (SampleLayers::CurrentAndBelow, "currentAndBelow"),
        (SampleLayers::All, "all"),
        (SampleLayers::AllNoAdjustments, "allNoAdjustments"),
        (SampleLayers::CurrentAndBelowNoAdjustments, "currentAndBelowNoAdjustments"),
    ];

    /// The choice named `id` (the `sampleLayer` param), `None` for an unknown one.
    pub fn parse(id: &str) -> Option<Self> {
        Self::ALL.iter().find(|(_, n)| *n == id).map(|(v, _)| *v)
    }

    fn no_adjustments(self) -> bool {
        matches!(self, SampleLayers::AllNoAdjustments | SampleLayers::CurrentAndBelowNoAdjustments)
    }
}

/// The pixels a sample of `size` at document point (`x`, `y`) averages: the `size`×`size` square
/// centred on the pixel under the point, clipped to `canvas`. `None` when the point is off the
/// canvas (or not finite), or `size` is not one of [`SAMPLE_SIZES`].
pub fn sample_rect(x: f64, y: f64, size: u32, canvas: Rect) -> Option<Rect> {
    if !(x.is_finite() && y.is_finite() && SAMPLE_SIZES.contains(&size)) {
        return None;
    }
    let (fx, fy) = (x.floor(), y.floor());
    if fx < f64::from(i32::MIN) || fx > f64::from(i32::MAX) || fy < f64::from(i32::MIN) || fy > f64::from(i32::MAX) {
        return None;
    }
    let (px, py) = (fx as i32, fy as i32);
    if !canvas.contains(px, py) {
        return None;
    }
    let r = (size / 2) as i32;
    let square = Rect::new(px.saturating_sub(r), py.saturating_sub(r), px.saturating_add(r).saturating_add(1), py.saturating_add(r).saturating_add(1));
    Some(square.intersect(&canvas))
}

/// Round `v` to what a channel of `depth` can store (32 bit keeps the float).
fn quantize(v: f32, depth: SampleType) -> f32 {
    let mut b = [0u8; 4];
    photocraft_color::write_sample(&mut b, depth, 0, v);
    photocraft_color::read_sample(&b, depth, 0)
}

/// The alpha-weighted average of straight-alpha RGBA pixels `px`, taken in the channels of
/// colour model `mode` at bit depth `depth`: each pixel is converted to `mode`, the colour
/// channels are averaged weighted by alpha (so transparent pixels don't darken the result), the
/// mean is rounded to `depth` and converted back to RGBA. Alpha is the mean coverage. Fully
/// transparent (or empty) input averages to transparent black.
///
/// CMYK conversions use the CMYK space active on this thread
/// (`photocraft_color::convert::with_cmyk_space`).
pub fn average(px: &[[f32; 4]], mode: ColorMode, depth: SampleType) -> [f32; 4] {
    let fmt = PixelFormat::new(mode, depth, false);
    let n = fmt.channels().min(8);
    let mut sum = [0.0f64; 8];
    let mut alpha = 0.0f64;
    let mut native = [0.0f32; 8];
    for p in px {
        let a = if p[3].is_finite() { p[3].clamp(0.0, 1.0) } else { 0.0 };
        if a <= 0.0 || p[..3].iter().any(|v| !v.is_finite()) {
            continue;
        }
        let k = photocraft_raster::from_rgba_into(&fmt, *p, &mut native).min(n);
        for (s, v) in sum.iter_mut().zip(&native[..k]) {
            *s += f64::from(*v) * f64::from(a);
        }
        alpha += f64::from(a);
    }
    if px.is_empty() || alpha <= 0.0 {
        return [0.0; 4];
    }
    let mean: Vec<f32> = sum[..n].iter().map(|s| quantize((s / alpha) as f32, depth)).collect();
    let [r, g, b, _] = photocraft_raster::to_rgba(&fmt, &mean);
    [r, g, b, quantize((alpha / px.len() as f64) as f32, depth)]
}

/// The document that renders what `which` samples: the whole document, or a copy with the layers
/// it leaves out hidden. `active` is the active layer (`None`: Current Layer samples nothing).
fn sampled_document(doc: &Document, active: Option<LayerId>, which: SampleLayers) -> Option<std::borrow::Cow<'_, Document>> {
    use std::borrow::Cow;
    match which {
        SampleLayers::All => return Some(Cow::Borrowed(doc)),
        SampleLayers::Current => {
            // The active layer on its own, as if it were the only layer: its pixels, not how its
            // group, blend mode or opacity show it in the composite.
            let mut layer = doc.layer(active?)?.clone();
            layer.visible = true;
            layer.clipped = false;
            layer.opacity = 1.0;
            layer.fill_opacity = 1.0;
            layer.blend = photocraft_color::BlendMode::Normal;
            let mut solo = doc.clone();
            solo.layers = vec![layer];
            return Some(Cow::Owned(solo));
        }
        _ => {}
    }
    let mut copy = doc.clone();
    let walk: Vec<(Vec<usize>, LayerId, bool)> = doc.walk().into_iter().map(|(p, _, l)| (p, l.id, matches!(l.content, LayerContent::Adjustment(_)))).collect();
    if matches!(which, SampleLayers::CurrentAndBelow | SampleLayers::CurrentAndBelowNoAdjustments) {
        // Everything composited after (above) the active layer in the bottom-to-top walk, except
        // its own contents: the walk lists a group before its children.
        let pos = walk.iter().position(|(_, id, _)| Some(*id) == active)?;
        let own = walk.get(pos).map(|(p, _, _)| p.clone()).unwrap_or_default();
        for (path, _, _) in walk.get(pos + 1..).unwrap_or_default().iter().filter(|(p, _, _)| !p.starts_with(&own)) {
            if let Some(l) = copy.layer_at_mut(path) {
                l.visible = false;
            }
        }
    }
    if which.no_adjustments() {
        for (path, _, _) in walk.iter().filter(|(_, _, adj)| *adj) {
            if let Some(l) = copy.layer_at_mut(path) {
                l.visible = false;
            }
        }
    }
    Some(Cow::Owned(copy))
}

/// The colour the Eyedropper picks at document point (`x`, `y`) with Sample Size `size` (see
/// [`SAMPLE_SIZES`]) from the layers `which` names, as straight-alpha RGBA. A Point Sample is the
/// composite pixel itself; larger sizes are the [`average`] of the square around it, clipped to
/// the canvas. Transparent black off the canvas, over transparency, or when `which` needs an
/// active layer and there is none.
pub fn sample_color(doc: &Document, active: Option<LayerId>, x: f64, y: f64, size: u32, which: SampleLayers) -> [f32; 4] {
    let Some(rect) = sample_rect(x, y, size, doc.bounds()) else { return [0.0; 4] };
    let Some(src) = sampled_document(doc, active, which) else { return [0.0; 4] };
    let buf = photocraft_compose::render(&src, rect);
    if size == 1 {
        return buf.px.first().copied().unwrap_or_default();
    }
    average(&buf.px, averaging_model(doc), doc.depth)
}

/// The colour model the averages are taken in: the document's, except CMYK. CMYK documents are
/// composited in RGB (see `photocraft_compose`), and the profile's RGB → CMYK → RGB round trip
/// is not exact (deep shadows move by several levels), so averaging in CMYK would make a 3×3
/// sample of a flat area differ from its Point Sample. Their composite RGB is averaged instead.
fn averaging_model(doc: &Document) -> ColorMode {
    match doc.pixel_format().mode {
        ColorMode::Cmyk => ColorMode::Rgb,
        m => m,
    }
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

/// The `size` param: 1 (Point Sample) or an averaged square's side; "point" is accepted too.
pub fn size_param(cmd: &str, p: &Value) -> Result<u32> {
    match p.get("size") {
        None | Some(Value::Null) => Ok(1),
        Some(Value::String(s)) if s == "point" => Ok(1),
        Some(v) => v
            .as_u64()
            .and_then(|n| u32::try_from(n).ok())
            .filter(|n| SAMPLE_SIZES.contains(n))
            .ok_or_else(|| bad(cmd, format!("`size` = {v} must be \"point\" or one of {SAMPLE_SIZES:?}"))),
    }
}

/// The `sampleLayer` param (default `all`).
pub fn layers_param(cmd: &str, p: &Value) -> Result<SampleLayers> {
    match p.get("sampleLayer") {
        None | Some(Value::Null) => Ok(SampleLayers::All),
        Some(Value::String(s)) => SampleLayers::parse(s).ok_or_else(|| {
            let ids: Vec<&str> = SampleLayers::ALL.iter().map(|(_, n)| *n).collect();
            bad(cmd, format!("unknown sampleLayer `{s}` ({})", ids.join("|")))
        }),
        Some(v) => Err(bad(cmd, format!("`sampleLayer` = {v} must be a string"))),
    }
}

fn coord(cmd: &str, p: &Value, key: &str) -> Result<f64> {
    p.get(key).and_then(Value::as_f64).filter(|v| v.is_finite()).ok_or_else(|| bad(cmd, format!("`{key}` must be a finite number (document pixels)")))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "document.sampleColor",
        label: "Sample Color",
        menu: &[],
        shortcut: None,
        params: r##"{"x":f64,"y":f64,"size":"point"|1|3|5|11|31|51|101=1,"sampleLayer":"current"|"currentAndBelow"|"all"|"allNoAdjustments"|"currentAndBelowNoAdjustments"="all"} → [r,g,b,a] (the Eyedropper's Sample Size and Sample: the composite pixel, or the average of the size×size square centred on it, clipped to the canvas; transparent off the canvas)"##,
        enabled: |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
        run: |s: &mut Session, p: &Value| {
            const CMD: &str = "document.sampleColor";
            let (x, y) = (coord(CMD, p, "x")?, coord(CMD, p, "y")?);
            let size = size_param(CMD, p)?;
            let which = layers_param(CMD, p)?;
            let d = s.active().ok_or(EngineError::NoDocument)?;
            Ok(json!(sample_color(&d.doc, d.active_layer, x, y, size, which)))
        },
        journal: false,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Adjustment, Layer, Size};

    /// A `w`×`h` document of `mode`/`depth` whose background is `px(x, y)` (RGBA, converted to
    /// the document's channels).
    fn doc(w: u32, h: u32, mode: ColorMode, depth: SampleType, px: impl Fn(i32, i32) -> [f32; 4]) -> Document {
        let mut d = Document::new("t", Size::new(w, h), mode, depth);
        let mut bg = Layer::raster("Background", d.pixel_format());
        let s = bg.surface_mut().unwrap();
        let fmt = s.format();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                s.write_pixel(x, y, &photocraft_raster::from_rgba(&fmt, px(x, y)));
            }
        }
        d.layers.push(bg);
        d
    }

    fn close(a: [f32; 4], b: [f32; 4], tol: f32) -> bool {
        a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= tol)
    }

    #[test]
    fn sample_rect_centres_the_square_and_clips_to_the_canvas() {
        let canvas = Rect::new(0, 0, 20, 10);
        assert_eq!(sample_rect(5.5, 5.5, 1, canvas), Some(Rect::new(5, 5, 6, 6)));
        assert_eq!(sample_rect(5.9, 5.1, 3, canvas), Some(Rect::new(4, 4, 7, 7)));
        assert_eq!(sample_rect(10.0, 5.0, 5, canvas), Some(Rect::new(8, 3, 13, 8)));
        // Corners and edges: the square is clipped.
        assert_eq!(sample_rect(0.0, 0.0, 5, canvas), Some(Rect::new(0, 0, 3, 3)));
        assert_eq!(sample_rect(19.5, 9.5, 11, canvas), Some(Rect::new(14, 4, 20, 10)));
        assert_eq!(sample_rect(10.0, 5.0, 101, canvas), Some(canvas));
        // Off the canvas, non-finite, or not a Photoshop size: nothing.
        assert_eq!(sample_rect(-0.5, 3.0, 3, canvas), None);
        assert_eq!(sample_rect(20.0, 3.0, 3, canvas), None);
        assert_eq!(sample_rect(f64::NAN, 3.0, 3, canvas), None);
        assert_eq!(sample_rect(1e300, 3.0, 3, canvas), None);
        assert_eq!(sample_rect(3.0, 3.0, 4, canvas), None);
        assert_eq!(sample_rect(3.0, 3.0, 0, canvas), None);
        // At the i32 limits the square saturates rather than overflowing.
        let huge = Rect::new(i32::MAX - 4, i32::MAX - 4, i32::MAX, i32::MAX);
        assert_eq!(sample_rect(f64::from(i32::MAX - 1), f64::from(i32::MAX - 1), 101, huge), Some(huge));
    }

    #[test]
    fn average_weights_by_alpha_and_ignores_garbage() {
        let px = [[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0, 1.0]];
        let a = average(&px, ColorMode::Rgb, SampleType::F32);
        assert!(close(a, [0.5, 0.0, 0.5, 0.5], 1e-6), "{a:?}");
        // Half-transparent red next to opaque blue: red counts half.
        let a = average(&[[1.0, 0.0, 0.0, 0.5], [0.0, 0.0, 1.0, 1.0]], ColorMode::Rgb, SampleType::F32);
        assert!(close(a, [1.0 / 3.0, 0.0, 2.0 / 3.0, 0.75], 1e-6), "{a:?}");
        assert_eq!(average(&[], ColorMode::Rgb, SampleType::U8), [0.0; 4]);
        assert_eq!(average(&[[0.3, 0.3, 0.3, 0.0]], ColorMode::Rgb, SampleType::U8), [0.0; 4]);
    }

    #[test]
    fn average_rounds_to_the_document_depth() {
        // 0, 0.1, 1 averages to 0.3667: 94/255 at 8 bit, 24030/65535 at 16 bit, the float at 32.
        let px = [[0.0, 0.0, 0.0, 1.0], [0.1, 0.1, 0.1, 1.0], [1.0, 1.0, 1.0, 1.0]];
        let a8 = average(&px, ColorMode::Rgb, SampleType::U8);
        assert_eq!(a8[0], 94.0 / 255.0);
        let a16 = average(&px, ColorMode::Rgb, SampleType::U16);
        assert_eq!(a16[0], 24030.0 / 65535.0);
        let a32 = average(&px, ColorMode::Rgb, SampleType::F32);
        assert!((a32[0] - 1.1 / 3.0).abs() < 1e-6);
        assert!(a8[0] != a32[0] && a16[0] != a32[0]);
    }

    #[test]
    fn average_is_taken_in_the_document_colour_model() {
        // Gray: the mean of the gray values.
        let a = average(&[[0.2, 0.2, 0.2, 1.0], [0.6, 0.6, 0.6, 1.0]], ColorMode::Grayscale, SampleType::F32);
        assert!(close(a, [0.4, 0.4, 0.4, 1.0], 1e-5), "{a:?}");
        // CMYK and Lab: the mean of the CMYK / Lab values, not of the RGB ones.
        let (red, blue) = ([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]);
        for mode in [ColorMode::Cmyk, ColorMode::Lab] {
            let fmt = PixelFormat::new(mode, SampleType::F32, false);
            let (r, b) = (photocraft_raster::from_rgba(&fmt, red), photocraft_raster::from_rgba(&fmt, blue));
            let mean: Vec<f32> = r.iter().zip(&b).map(|(x, y)| (x + y) / 2.0).collect();
            let want = photocraft_raster::to_rgba(&fmt, &mean);
            let got = average(&[red, blue], mode, SampleType::F32);
            assert!(close(got, want, 1e-5), "{mode:?}: {got:?} vs {want:?}");
            assert!(!close(got, [0.5, 0.0, 0.5, 1.0], 0.02), "{mode:?} averaged in RGB: {got:?}");
        }
    }

    #[test]
    fn averages_use_the_document_model_except_cmyk() {
        let model = |mode| averaging_model(&Document::new("m", Size::new(1, 1), mode, SampleType::U8));
        assert_eq!(model(ColorMode::Rgb), ColorMode::Rgb);
        assert_eq!(model(ColorMode::Grayscale), ColorMode::Grayscale);
        assert_eq!(model(ColorMode::Lab), ColorMode::Lab);
        assert_eq!(model(ColorMode::Bitmap), ColorMode::Grayscale);
        assert_eq!(model(ColorMode::Cmyk), ColorMode::Rgb);
    }

    /// A 9×9 document: white, except a black 3×3 block at (0..3, 0..3).
    fn block(mode: ColorMode, depth: SampleType) -> Document {
        doc(9, 9, mode, depth, |x, y| if x < 3 && y < 3 { [0.0, 0.0, 0.0, 1.0] } else { [1.0; 4] })
    }

    #[test]
    fn sample_sizes_average_the_square_around_the_pixel_at_every_depth_and_mode() {
        for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab] {
            for depth in SampleType::ALL {
                let d = block(mode, depth);
                let all = SampleLayers::All;
                let tol = if depth == SampleType::U8 { 1.5 / 255.0 } else { 2e-3 };
                // Point Sample: the composite pixel itself (black, as the document shows it).
                let p = sample_color(&d, None, 1.5, 1.5, 1, all);
                let white = sample_color(&d, None, 7.0, 7.0, 1, all);
                assert!(p[0] < 0.05 && p[3] == 1.0, "{mode:?} {depth:?} point {p:?}");
                // 3×3 at (3, 3): one black pixel of nine. Gray, RGB: 8/9 white.
                let s = sample_color(&d, None, 3.5, 3.5, 3, all);
                let want = sample_color(&d, None, 3.5, 3.5, 1, all);
                assert!(s[0] < want[0] && s[3] == 1.0, "{mode:?} {depth:?} 3x3 {s:?}");
                if matches!(mode, ColorMode::Rgb | ColorMode::Grayscale) {
                    assert!(close(s, [8.0 / 9.0, 8.0 / 9.0, 8.0 / 9.0, 1.0], tol), "{mode:?} {depth:?} 3x3 {s:?}");
                }
                // 5×5 at the corner: clipped to 3×3, all black.
                let c = sample_color(&d, None, 0.0, 0.0, 5, all);
                assert!(close(c, p, tol), "{mode:?} {depth:?} corner {c:?} vs {p:?}");
                // 101×101 covers the canvas: 9 of 81 black.
                if matches!(mode, ColorMode::Rgb | ColorMode::Grayscale) {
                    let w = sample_color(&d, None, 4.0, 4.0, 101, all);
                    assert!(close(w, [72.0 / 81.0, 72.0 / 81.0, 72.0 / 81.0, 1.0], tol), "{mode:?} {depth:?} 101 {w:?}");
                }
                // A uniform area averages to its own colour (round trips through the model).
                let u = sample_color(&d, None, 7.0, 7.0, 3, all);
                assert!(close(u, white, tol), "{mode:?} {depth:?} uniform {u:?} vs {white:?}");
                // Off the canvas: nothing.
                assert_eq!(sample_color(&d, None, -1.0, 4.0, 11, all), [0.0; 4]);
            }
        }
    }

    #[test]
    fn current_and_below_keeps_an_active_groups_contents() {
        // A white background and a group holding a red layer; the group is the active layer.
        let mut d = doc(3, 3, ColorMode::Rgb, SampleType::U8, |_, _| [1.0; 4]);
        let mut red = Layer::raster("red", d.pixel_format());
        red.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 3, 3), &[1.0, 0.0, 0.0, 1.0]);
        let group = Layer::group("group", vec![red]);
        let gid = group.id;
        d.layers.push(group);
        for which in [SampleLayers::CurrentAndBelow, SampleLayers::CurrentAndBelowNoAdjustments] {
            let c = sample_color(&d, Some(gid), 1.5, 1.5, 1, which);
            assert!(close(c, [1.0, 0.0, 0.0, 1.0], 1.5 / 255.0), "{which:?}: {c:?}");
        }
    }

    #[test]
    fn sample_layers_choose_what_is_averaged() {
        // Background white; a red 3x3 layer in the middle; an Invert adjustment on top.
        let mut d = doc(9, 9, ColorMode::Rgb, SampleType::U8, |_, _| [1.0; 4]);
        let bg = d.layers[0].id;
        let mut red = Layer::raster("red", d.pixel_format());
        red.surface_mut().unwrap().fill_rect(Rect::new(3, 3, 6, 6), &[1.0, 0.0, 0.0, 1.0]);
        let red = d.insert_above(Some(bg), red);
        let inv = d.insert_above(Some(red), Layer::new("invert", LayerContent::Adjustment(Adjustment::Invert)));
        let (cx, cy) = (4.5, 4.5);
        // 5×5 around the centre: 9 red pixels of 25.
        let s = |d: &Document, active, which| sample_color(d, active, cx, cy, 5, which);
        let close8 = |a: [f32; 4], b: [f32; 4]| close(a, b, 1.5 / 255.0);
        // Current Layer = the red layer alone: red, 9/25 covered.
        let cur = s(&d, Some(red), SampleLayers::Current);
        assert!(close8(cur, [1.0, 0.0, 0.0, 9.0 / 25.0]), "{cur:?}");
        // Current & Below: red over white, not inverted.
        let below = s(&d, Some(red), SampleLayers::CurrentAndBelow);
        assert!(close8(below, [1.0, 16.0 / 25.0, 16.0 / 25.0, 1.0]), "{below:?}");
        // All Layers: inverted (cyan over black).
        let all = s(&d, Some(red), SampleLayers::All);
        assert!(close8(all, [0.0, 9.0 / 25.0, 9.0 / 25.0, 1.0]), "{all:?}");
        // No Adjustments: as if the Invert were hidden.
        assert!(close8(s(&d, Some(red), SampleLayers::AllNoAdjustments), below));
        assert!(close8(s(&d, Some(inv), SampleLayers::CurrentAndBelowNoAdjustments), below));
        assert!(close8(s(&d, Some(inv), SampleLayers::CurrentAndBelow), all));
        // Current Layer with an adjustment layer or no active layer: nothing to sample.
        assert_eq!(s(&d, Some(inv), SampleLayers::Current), [0.0; 4]);
        assert_eq!(s(&d, None, SampleLayers::Current), [0.0; 4]);
        assert_eq!(s(&d, None, SampleLayers::CurrentAndBelow), [0.0; 4]);
        // Current Layer ignores how the composite shows the layer: hidden, 50% opacity, Multiply.
        let l = d.layer_mut(red).unwrap();
        l.visible = false;
        l.opacity = 0.5;
        l.blend = photocraft_color::BlendMode::Multiply;
        assert!(close8(s(&d, Some(red), SampleLayers::Current), cur));
    }

    #[test]
    fn the_command_validates_its_params() {
        let mut s = Session::new();
        assert!(s.execute("document.sampleColor", json!({"x": 1, "y": 1})).is_err(), "no document");
        s.execute("file.new", json!({"width": 8, "height": 8, "background": "white"})).unwrap();
        let v: Vec<f32> = serde_json::from_value(s.execute("document.sampleColor", json!({"x": 4, "y": 4, "size": 3})).unwrap()).unwrap();
        assert_eq!(v, vec![1.0; 4]);
        let v: Vec<f32> =
            serde_json::from_value(s.execute("document.sampleColor", json!({"x": 4, "y": 4, "size": "point", "sampleLayer": "current"})).unwrap()).unwrap();
        assert_eq!(v, vec![1.0; 4]);
        let v: Vec<f32> = serde_json::from_value(s.execute("document.sampleColor", json!({"x": -40, "y": 4, "size": 101})).unwrap()).unwrap();
        assert_eq!(v, vec![0.0; 4]);
        for bad in [
            json!({}),
            json!({"x": 1}),
            json!({"x": "a", "y": 1}),
            json!({"x": 1, "y": 1, "size": 4}),
            json!({"x": 1, "y": 1, "size": -3}),
            json!({"x": 1, "y": 1, "size": "huge"}),
            json!({"x": 1, "y": 1, "size": 1e40}),
            json!({"x": 1, "y": 1, "sampleLayer": "below"}),
            json!({"x": 1, "y": 1, "sampleLayer": 3}),
            json!([1, 2]),
        ] {
            assert!(matches!(s.execute("document.sampleColor", bad.clone()), Err(EngineError::BadParams { .. })), "{bad}");
        }
    }
}
