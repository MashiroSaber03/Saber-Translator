//! Affinity documents (`.af`, `.afdesign`, `.afphoto`, `.afpub`): the native layers, curves,
//! shapes, artboards, text, images and masks `photocraft-affinity` reads, mapped to PhotoCraft
//! layers (groups, shape layers, type layers, embedded smart objects, pixel layers, layer and
//! vector masks). What could not be mapped is listed in the warnings. A file whose native data
//! can't be read opens as its embedded preview, with a warning saying why.

use std::sync::Arc;

use photocraft_affinity::model::{self as af, Align, Blend, Kind, Pixels};
use photocraft_affinity::paint::{self as ap, Cap, GradientKind, Join};
use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::text::{CharStyle, ParagraphRun, ParagraphStyle, TextAlign, TextRun, TextShape};
use photocraft_doc::vector::{Knot, LineCap, LineJoin, Path, ShapeLayer, ShapeStroke, StrokeAlign, Subpath, VectorMask};
use photocraft_doc::{
    Artboard, ArtboardBackground, Document, Fill, GradientStyle, Group, Layer, LayerContent, LayerMask, Locks, SmartObject, SmartSource, TextLayer,
};
use photocraft_geom::{Affine, Point, Rect, Size};
use photocraft_raster::Surface;

use crate::{ImportResult, IoError};

pub use photocraft_affinity::is_affinity;

/// Extensions recognized as Affinity documents.
pub const EXTENSIONS: &[&str] = &["af", "afphoto", "afdesign", "afpub"];

/// Pixels an image or mask is resampled into, after cropping to the canvas.
const MAX_PIXELS: u64 = 64 << 20;
/// Canvas side in pixels.
const MAX_SIDE: f64 = 300_000.0;

pub fn has_extension(name: &str) -> bool {
    name.rsplit(['/', '\\']).next().and_then(|n| n.rsplit_once('.')).is_some_and(|(_, ext)| EXTENSIONS.iter().any(|e| ext.eq_ignore_ascii_case(e)))
}

pub(crate) fn import(name: &str, bytes: &[u8]) -> Result<ImportResult, IoError> {
    match photocraft_affinity::read(bytes, photocraft_affinity::Limits::default()) {
        Ok(d) => {
            let mut b = Builder::new(name, &d)?;
            b.document(&d)?;
            let mut warnings: Vec<String> = d.warnings.iter().map(|w| format!("Affinity: not imported or approximated: {w}.")).collect();
            warnings.extend(b.warnings.iter().map(|w| format!("Affinity: {w}.")));
            // Type layers draw from their model, within the import's raster bound: the sizes and
            // frames come from an untrusted file.
            let mut document = b.doc;
            crate::text_import::prepare(&mut document);
            // Save never writes back over the Affinity file.
            Ok(ImportResult { document, warnings, source_read_only: true, preview_only: false })
        }
        Err(native) => preview(name, bytes, &native),
    }
}

/// The embedded preview, for files whose native document can't be read.
fn preview(name: &str, bytes: &[u8], native: &photocraft_affinity::Error) -> Result<ImportResult, IoError> {
    let p = photocraft_affinity::preview(bytes).map_err(|e| IoError::Unsupported(format!("{native}; its embedded preview can't be read either ({e})")))?;
    let opts = photocraft_codecs::DecodeOptions {
        limits: photocraft_codecs::Limits { max_width: 4096, max_height: 4096, max_pixels: 4096 * 4096, max_alloc: 128 << 20 },
        ..Default::default()
    };
    let img = photocraft_codecs::decode_with(p.png, &opts)?;
    let mut r = crate::flat::image_to_document(name, &img)?;
    if let Some(layer) = r.document.layers.first_mut() {
        layer.name = "Affinity preview".into();
        layer.locks = Default::default();
    }
    r.source_read_only = true;
    r.preview_only = true;
    r.warnings.insert(
        0,
        format!(
            "The native Affinity document could not be read ({native}). Opened only its embedded {}×{} PNG preview, which may be smaller than the document: no layers, vectors, text or pages. Save a new copy; the Affinity source cannot be saved back. Export PSD or PNG from Affinity for full-resolution artwork.",
            p.width, p.height
        ),
    );
    Ok(r)
}

struct Builder {
    doc: Document,
    warnings: Vec<String>,
    /// Document pixels (in the canvas) per Affinity document pixel: always 1; spreads are offset.
    format: PixelFormat,
    canvas: Rect,
}

fn affine(a: af::Affine) -> Affine {
    Affine { m: a.0 }
}

fn point(p: af::Point) -> Point {
    Point::new(p.x, p.y)
}

impl Builder {
    fn new(name: &str, d: &af::Document) -> Result<Self, IoError> {
        // Spreads (Publisher pages) side by side, 64 px apart.
        let mut w: f64 = 0.0;
        let mut h: f64 = 0.0;
        for (i, s) in d.spreads.iter().enumerate() {
            w += (s.bounds.x1 - s.bounds.x0).max(1.0) + if i > 0 { 64.0 } else { 0.0 };
            h = h.max((s.bounds.y1 - s.bounds.y0).max(1.0));
        }
        if !(w.is_finite() && h.is_finite()) || w > MAX_SIDE || h > MAX_SIDE {
            return Err(IoError::Unsupported("the Affinity document is larger than 300 000 pixels on a side".into()));
        }
        let size = Size::new(w.ceil().max(1.0) as u32, h.ceil().max(1.0) as u32);
        let mut doc = Document::new(name, size, ColorMode::Rgb, SampleType::U8);
        doc.resolution_dpi = d.dpi as f32;
        let canvas = Rect::new(0, 0, size.width as i32, size.height as i32);
        Ok(Self { doc, warnings: Vec::new(), format: PixelFormat::RGBA8, canvas })
    }

    fn warn(&mut self, w: &str) {
        if !self.warnings.iter().any(|x| x == w) {
            self.warnings.push(w.to_string());
        }
    }

    fn document(&mut self, d: &af::Document) -> Result<(), IoError> {
        let pages = d.spreads.len() > 1 || d.spreads.iter().any(|s| s.pages.len() > 1);
        let mut x = 0.0;
        let mut layers = Vec::new();
        for (i, spread) in d.spreads.iter().enumerate() {
            let to_doc = Affine::translate(x - spread.bounds.x0, -spread.bounds.y0);
            let mut children: Vec<Layer> = spread.nodes.iter().filter_map(|n| self.layer(n, to_doc, true)).collect();
            if pages {
                // Each page becomes an artboard holding the art drawn on it.
                let rects: Vec<af::Rect> = if spread.pages.is_empty() { vec![spread.bounds] } else { spread.pages.clone() };
                let page = rects.first().copied().unwrap_or(spread.bounds);
                let r = self.bbox(to_doc, page);
                let mut g = Layer::new(
                    format!("Page {}", i + 1),
                    LayerContent::Group(Group { children: std::mem::take(&mut children), expanded: true, artboard: None }),
                );
                if let LayerContent::Group(gr) = &mut g.content {
                    gr.artboard = Some(Artboard { rect: r, background: ArtboardBackground::White, preset: String::new() });
                }
                if rects.len() > 1 {
                    self.warn("facing pages import as one artboard per spread");
                }
                layers.push(g);
            } else {
                layers.append(&mut children);
            }
            x += (spread.bounds.x1 - spread.bounds.x0).max(1.0) + 64.0;
        }
        if d.spreads.first().is_some_and(|s| !s.transparent) && !self.doc_has_artboards(&layers) {
            // An opaque page: a white background below the art.
            let mut bg = Surface::with_default(self.format, &[1.0, 1.0, 1.0, 1.0]);
            bg.prune();
            let mut l = Layer::new("Background", LayerContent::Raster(bg));
            l.locks.transparency = true;
            l.locks.position = true;
            layers.insert(0, l);
        }
        self.doc.layers = layers;
        Ok(())
    }

    fn doc_has_artboards(&self, layers: &[Layer]) -> bool {
        layers.iter().any(|l| matches!(&l.content, LayerContent::Group(g) if g.artboard.is_some()))
    }

    /// Integer bounding box of `r` mapped by `m`, clipped to the canvas.
    fn bbox(&self, m: Affine, r: af::Rect) -> Rect {
        let pts = [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)].map(|(x, y)| m.apply(Point::new(x, y)));
        let fold = |f: fn(f64, f64) -> f64, init: f64, g: fn(&Point) -> f64| pts.iter().map(g).fold(init, f);
        let x0 = fold(f64::min, f64::INFINITY, |p| p.x).floor();
        let y0 = fold(f64::min, f64::INFINITY, |p| p.y).floor();
        let x1 = fold(f64::max, f64::NEG_INFINITY, |p| p.x).ceil();
        let y1 = fold(f64::max, f64::NEG_INFINITY, |p| p.y).ceil();
        let c = |v: f64, lo: i32, hi: i32| (v.clamp(f64::from(lo), f64::from(hi))) as i32;
        Rect::new(
            c(x0, self.canvas.x0, self.canvas.x1),
            c(y0, self.canvas.y0, self.canvas.y1),
            c(x1, self.canvas.x0, self.canvas.x1),
            c(y1, self.canvas.y0, self.canvas.y1),
        )
    }

    fn blend(&mut self, b: Blend) -> BlendMode {
        match b {
            Blend::Normal => BlendMode::Normal,
            Blend::PassThrough => BlendMode::PassThrough,
            Blend::Darken => BlendMode::Darken,
            Blend::DarkerColor => BlendMode::DarkerColor,
            Blend::Multiply => BlendMode::Multiply,
            Blend::ColorBurn => BlendMode::ColorBurn,
            Blend::Lighten => BlendMode::Lighten,
            Blend::LighterColor => BlendMode::LighterColor,
            Blend::Screen => BlendMode::Screen,
            Blend::ColorDodge => BlendMode::ColorDodge,
            Blend::Add => BlendMode::LinearDodge,
            Blend::Overlay => BlendMode::Overlay,
            Blend::SoftLight => BlendMode::SoftLight,
            Blend::HardLight => BlendMode::HardLight,
            Blend::VividLight => BlendMode::VividLight,
            Blend::PinLight => BlendMode::PinLight,
            Blend::LinearLight => BlendMode::LinearLight,
            Blend::HardMix => BlendMode::HardMix,
            Blend::Difference => BlendMode::Difference,
            Blend::Exclusion => BlendMode::Exclusion,
            Blend::Subtract => BlendMode::Subtract,
            Blend::Hue => BlendMode::Hue,
            Blend::Saturation => BlendMode::Saturation,
            Blend::Luminosity => BlendMode::Luminosity,
            Blend::Color => BlendMode::Color,
            Blend::Average | Blend::Negation | Blend::Reflect | Blend::Glow | Blend::Erase => {
                self.warn("the Average, Negation, Reflect, Glow and Erase blend modes import as Normal");
                BlendMode::Normal
            }
        }
    }

    fn base(&mut self, n: &af::Node, content: LayerContent) -> Layer {
        let mut l = Layer::new(if n.name.is_empty() { default_name(&n.kind) } else { n.name.clone() }, content);
        l.visible = n.visible;
        l.opacity = n.opacity as f32;
        l.blend = self.blend(n.blend);
        if n.locked {
            l.locks = Locks { all: true, ..Locks::default() };
        }
        l
    }

    fn group(&mut self, n: &af::Node, children: Vec<Layer>) -> Layer {
        // Pass-through and Normal (isolated) groups map one to one.
        self.base(n, LayerContent::Group(Group { children, expanded: false, artboard: None }))
    }

    /// One Affinity node and its subtree; `to_doc` maps Affinity document pixels to the canvas.
    fn layer(&mut self, n: &af::Node, to_doc: Affine, top: bool) -> Option<Layer> {
        let children: Vec<Layer> = n.children.iter().filter_map(|c| self.layer(c, to_doc, false)).collect();
        let mut l = match &n.kind {
            Kind::Layer | Kind::Group => self.group(n, children),
            Kind::Unsupported if children.is_empty() => return None,
            Kind::Unsupported => self.group(n, children),
            Kind::Shape { path, fills, strokes, even_odd } => {
                let path = self.path(path, to_doc, *even_odd);
                let shape = self.shape(path.clone(), fills, strokes, to_doc);
                if children.is_empty() {
                    self.base(n, LayerContent::Shape(shape))
                } else {
                    // A shape clips its children: a group masked by the outline, the shape at its bottom.
                    let mut own = Layer::new(if n.name.is_empty() { "Shape".to_string() } else { n.name.clone() }, LayerContent::Shape(shape));
                    own.opacity = 1.0;
                    let mut kids = vec![own];
                    kids.extend(children);
                    let mut g = self.group(n, kids);
                    g.vector_mask = Some(VectorMask { path, enabled: true, linked: true, density: 1.0, feather: 0.0 });
                    if !strokes.is_empty() {
                        self.warn("strokes of shapes that clip other layers are clipped to the shape");
                    }
                    g
                }
            }
            Kind::Artboard { rect, background } => {
                let r = self.bbox(to_doc, *rect);
                let bg = match background.first() {
                    Some(ap::Paint::Solid(c)) => {
                        let c = color(*c);
                        if c.alpha <= 0.0 { ArtboardBackground::Transparent } else { ArtboardBackground::Custom(c) }
                    }
                    Some(_) => {
                        self.warn("artboard backgrounds other than a solid colour import as transparent");
                        ArtboardBackground::Transparent
                    }
                    None => ArtboardBackground::Transparent,
                };
                let mut g = self.group(n, children);
                if top {
                    if let LayerContent::Group(gr) = &mut g.content {
                        gr.artboard = Some(Artboard { rect: r, background: bg, preset: String::new() });
                    }
                } else {
                    // PhotoCraft artboards are top-level groups: a nested one clips by a vector mask.
                    let p = Path::new(vec![Subpath::polygon(&[
                        (f64::from(r.x0), f64::from(r.y0)),
                        (f64::from(r.x1), f64::from(r.y0)),
                        (f64::from(r.x1), f64::from(r.y1)),
                        (f64::from(r.x0), f64::from(r.y1)),
                    ])]);
                    g.vector_mask = Some(VectorMask { path: p, enabled: true, linked: true, density: 1.0, feather: 0.0 });
                    self.warn("artboards inside groups import as masked groups");
                }
                g
            }
            Kind::Text(t) => {
                let layer = self.text(t, to_doc);
                let l = self.base(n, LayerContent::Text(layer));
                if children.is_empty() {
                    l
                } else {
                    let mut kids = vec![l];
                    kids.extend(children);
                    self.group(n, kids)
                }
            }
            Kind::Image(img) => match self.image(img, to_doc) {
                Some(content) => self.base(n, content),
                None if children.is_empty() => return None,
                None => self.group(n, children),
            },
        };
        if let Some(mask) = &n.mask {
            l.vector_mask = Some(VectorMask { path: self.path(mask, to_doc, false), enabled: true, linked: true, density: 1.0, feather: 0.0 });
        }
        if let Some(mask) = &n.pixel_mask
            && let Some(surface) = self.mask(mask, to_doc)
        {
            l.mask = Some(LayerMask { surface, enabled: true, linked: true, density: 1.0, feather: 0.0 });
        }
        Some(l)
    }

    fn path(&self, p: &af::Path, to_doc: Affine, even_odd: bool) -> Path {
        let m = to_doc;
        let subpaths = p
            .subpaths
            .iter()
            .map(|s| {
                // Knots: each anchor with the control points of the segments arriving and leaving.
                let mut knots: Vec<Knot> = Vec::with_capacity(s.segments.len() + 1);
                let start = m.apply(point(s.start));
                knots.push(Knot::corner(start.x, start.y));
                for [c1, c2, e] in &s.segments {
                    if let Some(prev) = knots.last_mut() {
                        prev.out_ctrl = m.apply(point(*c1));
                    }
                    let e = m.apply(point(*e));
                    let mut k = Knot::corner(e.x, e.y);
                    k.in_ctrl = m.apply(point(*c2));
                    knots.push(k);
                }
                // A closed subpath that ends where it starts: the last knot is the first.
                if s.closed
                    && knots.len() > 1
                    && let (Some(f), Some(l)) = (knots.first().copied(), knots.last().copied())
                    && (f.anchor.x - l.anchor.x).abs() < 1e-9
                    && (f.anchor.y - l.anchor.y).abs() < 1e-9
                {
                    knots.pop();
                    if let Some(first) = knots.first_mut() {
                        first.in_ctrl = l.in_ctrl;
                    }
                }
                Subpath { closed: s.closed, knots, op: Default::default() }
            })
            .collect();
        let mut path = Path::new(subpaths);
        if even_odd {
            // One component filled even-odd, so inner subpaths cut holes as in Affinity.
            path.fill_rule = photocraft_doc::vector::FillRule::EvenOdd;
            for s in path.subpaths.iter_mut().skip(1) {
                s.op = photocraft_doc::vector::PathOp::Join;
            }
        }
        path
    }

    fn fill(&mut self, p: &ap::Paint, frame: Rect, to_doc: Affine) -> Option<Fill> {
        match p {
            ap::Paint::None => None,
            ap::Paint::Solid(c) => Some(Fill::Solid(color(*c))),
            ap::Paint::Gradient(g) => {
                let style = match g.kind {
                    GradientKind::Linear => GradientStyle::Linear,
                    GradientKind::Radial => GradientStyle::Radial,
                    GradientKind::Conical => GradientStyle::Angle,
                };
                let m = to_doc.mul(&affine(g.transform));
                let (o, e, v) = (m.apply(Point::new(0.0, 0.0)), m.apply(Point::new(1.0, 0.0)), m.apply(Point::new(0.0, 1.0)));
                let (u, w) = ((e.x - o.x).hypot(e.y - o.y), (v.x - o.x).hypot(v.y - o.y));
                if style == GradientStyle::Radial && u > 0.0 && (w / u - 1.0).abs() > 0.01 {
                    self.warn("elliptical gradients import as circular ones");
                }
                let (angle, scale, offset) =
                    photocraft_compose::gradient_fill::from_handles(style, [o.x as f32, o.y as f32], [e.x as f32, e.y as f32], frame, 0.0);
                let stops = g.stops.iter().map(|s| (s.offset as f32, Color { alpha: 1.0, ..color(s.color) })).collect();
                let opacity_stops = g.stops.iter().map(|s| (s.offset as f32, s.color.alpha() as f32)).collect();
                let midpoints = g.stops.iter().take(g.stops.len().saturating_sub(1)).map(|s| s.half_point() as f32).collect();
                Some(Fill::Gradient { stops, angle, scale, style, reverse: false, opacity_stops, midpoints, offset, dither: false, align: true })
            }
        }
    }

    fn shape(&mut self, path: Path, fills: &[ap::Paint], strokes: &[ap::Stroke], to_doc: Affine) -> ShapeLayer {
        let frame = path
            .control_bounds()
            .map(|(x0, y0, x1, y1)| Rect::new(x0.floor() as i32, y0.floor() as i32, x1.ceil() as i32, y1.ceil() as i32))
            .unwrap_or(self.canvas);
        if fills.len() > 1 || strokes.len() > 1 {
            self.warn("only one fill and one stroke per shape import");
        }
        let fill = fills.first().and_then(|f| self.fill(f, frame, to_doc));
        let stroke = strokes.first().and_then(|s| {
            let paint = self.fill(&s.paint, frame, to_doc)?;
            if s.behind {
                self.warn("strokes drawn behind their fill import in front of it");
            }
            Some(ShapeStroke {
                width: s.width as f32,
                paint,
                opacity: 1.0,
                align: match s.align {
                    ap::Align::Center => StrokeAlign::Center,
                    ap::Align::Inside => StrokeAlign::Inside,
                    ap::Align::Outside => StrokeAlign::Outside,
                },
                cap: match s.cap {
                    Cap::Butt => LineCap::Butt,
                    Cap::Round => LineCap::Round,
                    Cap::Square => LineCap::Square,
                },
                join: match s.join {
                    Join::Miter => LineJoin::Miter,
                    Join::Round => LineJoin::Round,
                    Join::Bevel => LineJoin::Bevel,
                },
                miter_limit: s.miter_limit as f32,
                // In multiples of the width, as Photoshop stores them.
                dashes: s.dash.as_ref().map(|(d, _)| d.iter().map(|v| (v / s.width.max(1e-9)) as f32).collect()).unwrap_or_default(),
                dash_offset: s.dash.as_ref().map_or(0.0, |(_, o)| (o / s.width.max(1e-9)) as f32),
            })
        });
        let mut sh = ShapeLayer { path, fill, stroke, live: None, cache: None, psd_raw: None };
        sh.cache = Some(photocraft_vector::render_shape(&sh, self.format, self.canvas));
        sh
    }

    fn text(&mut self, t: &af::Text, to_doc: Affine) -> TextLayer {
        let origin = match t.frame {
            Some(f) => Point::new(f.x0, f.y0),
            None => point(t.anchor),
        };
        let m = to_doc.mul(&affine(t.transform)).mul(&Affine::translate(origin.x, origin.y));
        let k = m.determinant().abs().sqrt().max(1e-9);
        let transform = m.mul(&Affine::scale(1.0 / k));
        let dpi = self.doc.resolution_dpi.max(1.0) as f64;
        let mut text = String::new();
        let mut runs = Vec::new();
        for r in &t.runs {
            // Affinity separates paragraphs with U+2029 and lines with U+2028.
            let s = r.text.replace(['\u{2029}', '\u{2028}'], "\r");
            let style = CharStyle {
                font_family: r.family.clone(),
                postscript_name: (!r.postscript.is_empty()).then(|| r.postscript.clone()),
                weight: r.weight.clamp(100, 900) as u16,
                italic: r.italic,
                size_pt: (r.size * k * 72.0 / dpi) as f32,
                color: match &r.fill {
                    ap::Paint::Solid(c) => color(*c),
                    _ => Color::BLACK,
                },
                tracking: (r.tracking * 1000.0) as f32,
                leading_pt: r.leading.map(|l| (l * k * 72.0 / dpi) as f32),
                ..CharStyle::default()
            };
            text.push_str(&s);
            runs.push(TextRun { len: s.len(), style });
        }
        let align = match t.align {
            Align::Left => TextAlign::Left,
            Align::Center => TextAlign::Center,
            Align::Right => TextAlign::Right,
            Align::Justify => TextAlign::JustifyLeft,
        };
        let mut layer = TextLayer {
            text: text.clone(),
            font_family: runs.first().map(|r| r.style.font_family.clone()).unwrap_or_default(),
            size_pt: runs.first().map_or(12.0, |r| r.style.size_pt),
            color: runs.first().map_or(Color::BLACK, |r| r.style.color),
            transform,
            runs,
            paragraphs: vec![ParagraphRun { len: text.len(), style: ParagraphStyle { align, ..ParagraphStyle::default() } }],
            ..TextLayer::default()
        };
        if let Some(f) = t.frame {
            layer.shape = TextShape::Box { x: 0.0, y: 0.0, width: ((f.x1 - f.x0) * k) as f32, height: ((f.y1 - f.y0) * k) as f32 };
        }
        layer
    }

    /// Straight RGBA8 pixels of an image (decoding an embedded file), with its size.
    fn pixels(&mut self, img: &af::Image) -> Option<(u32, u32, Vec<u8>)> {
        match &img.pixels {
            Pixels::Rgba8(v) => Some((img.width, img.height, v.clone())),
            Pixels::Encoded(bytes) => {
                let opts = photocraft_codecs::DecodeOptions {
                    limits: photocraft_codecs::Limits { max_width: 65_535, max_height: 65_535, max_pixels: MAX_PIXELS, max_alloc: 512 << 20 },
                    ..Default::default()
                };
                match photocraft_codecs::decode_with(bytes, &opts) {
                    Ok(d) => {
                        let (w, h) = d.dimensions();
                        Some((w, h, d.to_rgba8()))
                    }
                    Err(_) => {
                        self.warn("an embedded image that could not be decoded was left out");
                        None
                    }
                }
            }
        }
    }

    fn image(&mut self, img: &af::Image, to_doc: Affine) -> Option<LayerContent> {
        let (w, h, rgba) = self.pixels(img)?;
        // The pixel grid maps onto the Affinity image's own size, whatever the stored file's.
        let m = to_doc
            .mul(&affine(img.transform))
            .mul(&Affine { m: [f64::from(img.width) / f64::from(w.max(1)), 0.0, 0.0, f64::from(img.height) / f64::from(h.max(1)), 0.0, 0.0] });
        let surface = self.resample(w, h, &rgba, m, 4)?;
        Some(match &img.pixels {
            // A placed image keeps its file, as a smart object placed by the same transform.
            Pixels::Encoded(bytes) => LayerContent::Smart(SmartObject::new(
                SmartSource::Embedded { file_name: "Affinity image".into(), bytes: Arc::new(bytes.clone()) },
                m,
                Some(surface),
            )),
            Pixels::Rgba8(_) => LayerContent::Raster(surface),
        })
    }

    fn mask(&mut self, img: &af::Image, to_doc: Affine) -> Option<Surface> {
        let (w, h, rgba) = self.pixels(img)?;
        let grey: Vec<u8> = rgba.as_chunks::<4>().0.iter().map(|p| p[0]).collect();
        let m = to_doc.mul(&affine(img.transform));
        self.resample(w, h, &grey, m, 1)
    }

    /// Bilinear resampling of `src` (`w`×`h`, `channels` 4 = straight RGBA8 or 1 = grey) mapped
    /// by `m` (source pixels to canvas pixels) into a surface; exact copy when `m` is an integer
    /// translation. Pixels outside the source are transparent (grey: 0, hidden).
    fn resample(&mut self, w: u32, h: u32, src: &[u8], m: Affine, channels: usize) -> Option<Surface> {
        let format = if channels == 1 { PixelFormat::GRAY8 } else { self.format };
        let mut out = if channels == 1 { Surface::with_default(format, &[0.0]) } else { Surface::new(format) };
        let r = self.bbox(m, af::Rect { x0: 0.0, y0: 0.0, x1: f64::from(w), y1: f64::from(h) });
        if r.width() == 0 || r.height() == 0 {
            return Some(out);
        }
        if u64::from(r.width()) * u64::from(r.height()) > MAX_PIXELS {
            self.warn("images larger than 64 megapixels on the canvas were left out");
            return None;
        }
        let [a, b, c, d, e, f] = m.m;
        let exact = a == 1.0 && d == 1.0 && b == 0.0 && c == 0.0 && e.fract() == 0.0 && f.fract() == 0.0;
        let inv = m.inverse()?;
        let (rw, rh) = (r.width() as usize, r.height() as usize);
        let mut buf = vec![0u8; rw * rh * channels];
        let at = |x: i64, y: i64, ch: usize| -> f64 {
            if x < 0 || y < 0 || x >= i64::from(w) || y >= i64::from(h) {
                return 0.0;
            }
            f64::from(src.get((y as usize * w as usize + x as usize) * channels + ch).copied().unwrap_or(0))
        };
        for (row, out_row) in buf.chunks_exact_mut(rw * channels).enumerate() {
            for (col, px) in out_row.chunks_exact_mut(channels).enumerate() {
                let p = inv.apply(Point::new(f64::from(r.x0) + col as f64 + 0.5, f64::from(r.y0) + row as f64 + 0.5));
                if exact {
                    let (x, y) = (p.x.floor() as i64, p.y.floor() as i64);
                    for (ch, v) in px.iter_mut().enumerate() {
                        *v = at(x, y, ch) as u8;
                    }
                    continue;
                }
                let (fx, fy) = (p.x - 0.5, p.y - 0.5);
                let (x0, y0) = (fx.floor() as i64, fy.floor() as i64);
                let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
                let weights =
                    [((x0, y0), (1.0 - tx) * (1.0 - ty)), ((x0 + 1, y0), tx * (1.0 - ty)), ((x0, y0 + 1), (1.0 - tx) * ty), ((x0 + 1, y0 + 1), tx * ty)];
                if channels == 1 {
                    px[0] = weights.iter().map(|((x, y), wgt)| at(*x, *y, 0) * wgt).sum::<f64>().round().clamp(0.0, 255.0) as u8;
                } else {
                    // Interpolate premultiplied colour so transparent pixels don't bleed.
                    let alpha: f64 = weights.iter().map(|((x, y), wgt)| at(*x, *y, 3) * wgt).sum();
                    for (ch, v) in px.iter_mut().enumerate().take(3) {
                        let pm: f64 = weights.iter().map(|((x, y), wgt)| at(*x, *y, ch) * at(*x, *y, 3) / 255.0 * wgt).sum();
                        *v = if alpha > 0.0 { (pm * 255.0 / alpha).round().clamp(0.0, 255.0) as u8 } else { 0 };
                    }
                    px[3] = alpha.round().clamp(0.0, 255.0) as u8;
                }
            }
        }
        out.write_interleaved(r, &buf);
        out.prune();
        Some(out)
    }
}

fn default_name(k: &Kind) -> String {
    match k {
        Kind::Layer => "Layer",
        Kind::Group => "Group",
        Kind::Artboard { .. } => "Artboard",
        Kind::Shape { .. } => "Shape",
        Kind::Text(_) => "Text",
        Kind::Image(_) => "Image",
        Kind::Unsupported => "Group",
    }
    .into()
}

fn color(c: ap::Color) -> Color {
    let f = |v: f64| v.clamp(0.0, 1.0) as f32;
    match c {
        ap::Color::Rgb { r, g, b, a } => Color::rgba(f(r), f(g), f(b), f(a)),
        ap::Color::Cmyk { c, m, y, k, a } => {
            let cmyk = Color { mode: ColorMode::Cmyk, c: [f(c), f(m), f(y), f(k)], alpha: f(a) };
            let [r, g, b] = cmyk.to_rgb();
            Color::rgba(r, g, b, f(a))
        }
        ap::Color::Gray { v, a } => Color::rgba(f(v), f(v), f(v), f(a)),
        ap::Color::Lab { l, a, b, alpha } => {
            let lab = Color { mode: ColorMode::Lab, c: [f(l / 100.0), f((a + 128.0) / 255.0), f((b + 128.0) / 255.0), 0.0], alpha: f(alpha) };
            let [r, g, bl] = lab.to_rgb();
            Color::rgba(r, g, bl, f(alpha))
        }
    }
}
