//! Rasterizes a [`TextLayout`] into a document-space [`Surface`] of any pixel format.

use photocraft_color::{Color, PixelFormat, SampleType};
use photocraft_doc::text::AntiAlias;
use photocraft_geom::{Affine, Rect};
use photocraft_raster::Surface;
use skrifa::instance::{LocationRef, NormalizedCoord, Size};
use skrifa::outline::DrawSettings;
use skrifa::{GlyphId, MetadataProvider};

use crate::layout::{GlyphOrient, PlacedGlyph, TextLayout};
use crate::raster::{Bounds, Coverage, LineSink, Pen, Xform, rect_to};
use crate::warp::{Placement, Warp};

/// Faux-italic slant in degrees (Photoshop-like).
pub const FAUX_ITALIC_DEG: f32 = 12.0;
/// Faux-bold ring radius as a fraction of the font size (tuned so stem weight stays close to the old summed-coverage look).
pub const FAUX_BOLD_RADIUS: f32 = 0.036;
/// Largest raster we produce (pixels), as a guard against absurd sizes.
const MAX_PIXELS: u64 = 256 * 1024 * 1024;

/// Rendered text: pixels in document space plus the covered rectangle.
pub struct Rendered {
    pub surface: Surface,
    pub rect: Rect,
}

/// Bends text-space lines with a [`Warp`], then maps them through `post` into `inner`.
struct WarpSink<'a, S: LineSink> {
    inner: &'a mut S,
    warp: &'a Warp,
    post: Xform,
}

impl<S: LineSink> LineSink for WarpSink<'_, S> {
    fn line(&mut self, p0: (f64, f64), p1: (f64, f64)) {
        let len = ((p1.0 - p0.0).powi(2) + (p1.1 - p0.1).powi(2)).sqrt();
        let n = ((len / self.warp.max_segment()).ceil() as usize).clamp(1, 256);
        let map = |t: f64| {
            let (x, y) = self.warp.apply(p0.0 + (p1.0 - p0.0) * t, p0.1 + (p1.1 - p0.1) * t);
            self.post.apply(x, y)
        };
        let mut a = map(0.0);
        for i in 1..=n {
            let b = map(i as f64 / n as f64);
            self.inner.line(a, b);
            a = b;
        }
    }
}

/// Draws every glyph and decoration of `layout` whose style colour is `color` (or all of them
/// when `color` is `None`) into `sink`, through `transform` (text space → sink space), bending
/// the outlines with `warp` first (in text space).
fn draw(layout: &TextLayout, transform: &Xform, sink: &mut impl LineSink, only_color: Option<&Color>, warp: Option<&Warp>) {
    draw_pass(layout, transform, sink, only_color, warp, Pass::All);
}

/// Which part of the layout [`draw_pass`] emits. Faux bold draws each glyph at several offsets;
/// summing those into one coverage buffer saturates the antialiased edge pixels (hard, pixelated
/// stems), so the rasteriser draws every offset into its own buffer and merges them with `max`.
#[derive(Clone, Copy, PartialEq)]
enum Pass {
    /// Everything, all offsets (bounds, outlines).
    All,
    /// Non-bold glyphs and decorations.
    Plain,
    /// Bold glyphs at one offset (index into [`BOLD_OFFSETS`]).
    Bold(usize),
}

const BOLD_OFFSETS: [(f64, f64); 8] = [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.7, 0.7), (-0.7, -0.7), (0.7, -0.7), (-0.7, 0.7)];

fn is_bold(layout: &TextLayout, g: &PlacedGlyph) -> bool {
    let face_bold = layout.faces.get(g.face as usize).is_some_and(|f| f.embolden);
    layout.styles.get(g.style as usize).is_some_and(|s| s.faux_bold) || face_bold
}

fn draw_pass(layout: &TextLayout, transform: &Xform, sink: &mut impl LineSink, only_color: Option<&Color>, warp: Option<&Warp>, pass: Pass) {
    for g in &layout.glyphs {
        let st = &layout.styles[g.style as usize];
        if only_color.is_some_and(|c| c != &st.color) {
            continue;
        }
        let face = &layout.faces[g.face as usize];
        let Ok(font) = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index) else {
            continue;
        };
        let Some(outline) = font.outline_glyphs().get(GlyphId::new(g.id)) else {
            continue;
        };
        let coords: Vec<NormalizedCoord> = face.coords.iter().map(|&c| NormalizedCoord::from_bits(c)).collect();
        let mut glyph = glyph_xform(layout, g, face.skew_deg);
        // Type on a path moves each glyph rigidly: turned to the curve at its centre.
        let mut bend = warp;
        if let Some(w) = warp {
            match w.place_glyph(glyph_center(layout, g, &font, face.size_px, &coords)) {
                Placement::Bend => {}
                Placement::Hidden => continue,
                Placement::Frame(m) => {
                    glyph = Xform(m).mul(&glyph);
                    bend = None;
                }
            }
        }
        let bold = st.faux_bold || face.embolden;
        let r = (face.size_px * FAUX_BOLD_RADIUS) as f64;
        let single = [(0.0, 0.0)];
        let offsets: &[(f64, f64)] = match (pass, bold) {
            (Pass::Plain, true) => continue,
            (Pass::Bold(_), false) => continue,
            (Pass::Bold(k), true) => match BOLD_OFFSETS.get(k..=k) {
                Some(o) => o,
                None => continue,
            },
            (Pass::All, true) => &BOLD_OFFSETS,
            (_, false) => &single,
        };
        for &(ox, oy) in offsets {
            let local = glyph.mul(&Xform([1.0, 0.0, 0.0, 1.0, ox * r, oy * r]));
            let settings = DrawSettings::unhinted(Size::new(face.size_px), LocationRef::new(&coords));
            match bend {
                Some(w) => {
                    let mut ws = WarpSink { inner: &mut *sink, warp: w, post: *transform };
                    let mut pen = Pen::new(&mut ws, local);
                    if outline.draw(settings, &mut pen).is_ok() {
                        skrifa::outline::OutlinePen::close(&mut pen);
                    }
                }
                None => {
                    let mut pen = Pen::new(&mut *sink, transform.mul(&local));
                    if outline.draw(settings, &mut pen).is_ok() {
                        skrifa::outline::OutlinePen::close(&mut pen);
                    }
                }
            }
        }
    }
    if matches!(pass, Pass::Bold(_)) {
        return;
    }
    for d in &layout.decorations {
        let st = &layout.styles[d.style as usize];
        if only_color.is_some_and(|c| c != &st.color) {
            continue;
        }
        match warp {
            Some(w) => {
                let mut ws = WarpSink { inner: &mut *sink, warp: w, post: *transform };
                rect_to(&mut ws, &Xform::IDENTITY, d.x0 as f64, d.y0 as f64, d.x1 as f64, d.y1 as f64);
            }
            None => rect_to(sink, transform, d.x0 as f64, d.y0 as f64, d.x1 as f64, d.y1 as f64),
        }
    }
}

/// Font units (scaled to px, y up) → text space for a placed glyph: scales, faux italic,
/// baseline shift, and the vertical-type orientation.
fn glyph_xform(layout: &TextLayout, g: &PlacedGlyph, face_skew_deg: f32) -> Xform {
    let Some(st) = layout.styles.get(g.style as usize) else { return Xform([1.0, 0.0, 0.0, -1.0, g.x as f64, g.y as f64]) };
    let hs = if st.horizontal_scale > 0.0 { st.horizontal_scale } else { 1.0 } as f64;
    let vs = if st.vertical_scale > 0.0 { st.vertical_scale } else { 1.0 } as f64;
    let skew_deg = if st.faux_italic { FAUX_ITALIC_DEG } else { 0.0 } + face_skew_deg;
    let skew = (skew_deg as f64).to_radians().tan() * vs;
    let shift = (st.baseline_shift_pt * layout.px_per_pt) as f64;
    let (x, y) = (g.x as f64, g.y as f64);
    match g.orient {
        GlyphOrient::Horizontal => Xform([hs, 0.0, skew, -vs, x, y - shift]),
        // Baseline shift moves upright glyphs in vertical type to the right.
        GlyphOrient::Upright => Xform([hs, 0.0, skew, -vs, x + shift, y]),
        // 90° clockwise: the glyph's up direction (and its baseline shift) points right.
        GlyphOrient::Rotated => Xform([0.0, 1.0, -1.0, 0.0, x, y]).mul(&Xform([hs, 0.0, skew, -vs, 0.0, -shift])),
    }
}

/// The horizontal centre (text space) of a placed glyph: the pen position plus half its advance.
fn glyph_center(layout: &TextLayout, g: &PlacedGlyph, font: &skrifa::FontRef, size_px: f32, coords: &[NormalizedCoord]) -> f64 {
    let hs = layout.styles.get(g.style as usize).map_or(1.0, |st| if st.horizontal_scale > 0.0 { st.horizontal_scale } else { 1.0 });
    let advance = match g.orient {
        GlyphOrient::Horizontal => font.glyph_metrics(Size::new(size_px), LocationRef::new(coords)).advance_width(GlyphId::new(g.id)).unwrap_or(0.0),
        _ => 0.0,
    };
    f64::from(g.x) + f64::from(advance * hs) / 2.0
}

/// The deformation of a type layer: the path it runs along, or its Warp Text settings (None
/// when neither applies).
pub fn layout_deform(layout: &TextLayout, layer: &photocraft_doc::TextLayer) -> Option<Warp> {
    Warp::on_path(&layer.shape).or_else(|| layout_warp(layout, layer.warp.as_ref()))
}

/// The warp to apply to `layout` (None when `warp` is absent, `warpNone` or flat).
pub fn layout_warp(layout: &TextLayout, warp: Option<&photocraft_doc::text::TextWarp>) -> Option<Warp> {
    Warp::new(warp?, layout.bounds()?)
}

/// Colour components in `format`'s model (without alpha).
fn color_in(format: &PixelFormat, c: &Color) -> Vec<f32> {
    let n = format.mode.color_channels();
    if c.mode == format.mode {
        c.c[..n].to_vec()
    } else {
        let [r, g, b] = c.to_rgb();
        let mut v = photocraft_raster::from_rgba(&PixelFormat { alpha: false, ..*format }, [r, g, b, 1.0]);
        v.truncate(n);
        v
    }
}

/// Document-space bounds of the drawn text (integer pixel rectangle).
pub fn ink_rect(layout: &TextLayout, transform: &Affine) -> Rect {
    ink_rect_warped(layout, transform, None)
}

fn rect_from_bounds([x0, y0, x1, y1]: [f64; 4]) -> Rect {
    if ![x0, y0, x1, y1].into_iter().all(f64::is_finite) {
        return Rect::EMPTY;
    }

    let min = f64::from(i32::MIN);
    let max = f64::from(i32::MAX);
    let (x0, y0, x1, y1) =
        ((x0.floor() - 1.0).clamp(min, max), (y0.floor() - 1.0).clamp(min, max), (x1.ceil() + 1.0).clamp(min, max), (y1.ceil() + 1.0).clamp(min, max));
    if x1 <= x0 || y1 <= y0 || x1 - x0 > MAX_PIXELS as f64 || y1 - y0 > MAX_PIXELS as f64 {
        return Rect::EMPTY;
    }

    Rect::new(x0 as i32, y0 as i32, x1 as i32, y1 as i32)
}

/// [`ink_rect`] of warped text.
pub fn ink_rect_warped(layout: &TextLayout, transform: &Affine, warp: Option<&Warp>) -> Rect {
    let mut b = Bounds::default();
    draw(layout, &Xform(transform.m), &mut b, None, warp);
    b.rect.map(rect_from_bounds).unwrap_or(Rect::EMPTY)
}

/// Rasterizes `layout` through `transform` (text space → document pixels). The result always
/// has an alpha channel; colour is written in `format`'s colour model and sample depth.
pub fn rasterize(layout: &TextLayout, transform: &Affine, format: PixelFormat, antialias: AntiAlias) -> Rendered {
    rasterize_warped(layout, transform, format, antialias, None)
}

/// [`rasterize`] with the glyph outlines bent by `warp` (Type › Warp Text).
pub fn rasterize_warped(layout: &TextLayout, transform: &Affine, format: PixelFormat, antialias: AntiAlias, warp: Option<&Warp>) -> Rendered {
    let format = PixelFormat { alpha: true, ..format };
    let rect = ink_rect_warped(layout, transform, warp);
    let (w, h) = (rect.width() as usize, rect.height() as usize);
    let mut surface = Surface::new(format);
    if w == 0 || h == 0 || (w as u64) * (h as u64) > MAX_PIXELS {
        return Rendered { surface, rect: Rect::new(0, 0, 0, 0) };
    }
    let n = format.mode.color_channels();
    let stride = n + 1;
    // Premultiplied accumulation.
    let mut acc = vec![0.0f32; w * h * stride];
    let xf = Xform([1.0, 0.0, 0.0, 1.0, -rect.x0 as f64, -rect.y0 as f64]).mul(&Xform(transform.m));
    let mut colors: Vec<Color> = Vec::new();
    for st in &layout.styles {
        if !colors.contains(&st.color) {
            colors.push(st.color);
        }
    }
    for c in &colors {
        let mut cov = Coverage::new(w, h);
        draw_pass(layout, &xf, &mut cov, Some(c), warp, Pass::Plain);
        let mut cov = cov.finish();
        if layout.glyphs.iter().any(|g| is_bold(layout, g)) {
            for k in 0..BOLD_OFFSETS.len() {
                let mut bc = Coverage::new(w, h);
                draw_pass(layout, &xf, &mut bc, Some(c), warp, Pass::Bold(k));
                for (a, b) in cov.iter_mut().zip(bc.finish()) {
                    *a = a.max(b);
                }
            }
        }
        let comps = color_in(&format, c);
        let a = c.alpha.clamp(0.0, 1.0);
        for (i, &cv) in cov.iter().enumerate() {
            let cv = if antialias == AntiAlias::None { if cv >= 0.5 { 1.0 } else { 0.0 } } else { cv };
            let s = cv * a;
            if s <= 0.0 {
                continue;
            }
            let px = &mut acc[i * stride..(i + 1) * stride];
            let keep = 1.0 - s;
            for j in 0..n {
                px[j] = comps[j] * s + px[j] * keep;
            }
            px[n] = s + px[n] * keep;
        }
    }
    // Unpremultiply.
    for px in acc.chunks_exact_mut(stride) {
        let a = px[n];
        if a > 0.0 {
            for v in &mut px[..n] {
                let straight = *v / a;
                // Float text can carry HDR and signed colours. Keep integer output bounded
                // and retain the old handling of non-finite intermediate values.
                *v = if format.sample == SampleType::F32 && straight.is_finite() { straight } else { straight.clamp(0.0, 1.0) };
            }
        }
    }
    surface.write_region(rect, &acc);
    surface.prune();
    Rendered { surface, rect }
}

/// One element of a glyph outline in document space (see [`outlines`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PathEl {
    MoveTo([f64; 2]),
    LineTo([f64; 2]),
    QuadTo([f64; 2], [f64; 2]),
    CurveTo([f64; 2], [f64; 2], [f64; 2]),
    Close,
}

/// Records outline commands through a text→document map (and an optional warp).
struct Recorder<'a> {
    glyph: Xform,
    post: Xform,
    warp: Option<&'a Warp>,
    out: Vec<PathEl>,
}

impl Recorder<'_> {
    fn map(&self, x: f32, y: f32) -> [f64; 2] {
        let (tx, ty) = self.glyph.apply(f64::from(x), f64::from(y));
        let (tx, ty) = match self.warp {
            Some(w) => w.apply(tx, ty),
            None => (tx, ty),
        };
        let (dx, dy) = self.post.apply(tx, ty);
        [dx, dy]
    }
}

impl skrifa::outline::OutlinePen for Recorder<'_> {
    fn move_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.out.push(PathEl::MoveTo(p));
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.map(x, y);
        self.out.push(PathEl::LineTo(p));
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let (c, p) = (self.map(cx0, cy0), self.map(x, y));
        self.out.push(PathEl::QuadTo(c, p));
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let (a, b, p) = (self.map(cx0, cy0), self.map(cx1, cy1), self.map(x, y));
        self.out.push(PathEl::CurveTo(a, b, p));
    }
    fn close(&mut self) {
        self.out.push(PathEl::Close);
    }
}

/// Glyph outlines of `layout` in document space (through `transform`, bent by `warp`), one
/// element list per glyph, for Type › Create Work Path and Convert to Shape. Faux bold is not
/// applied (Photoshop also converts the regular outline); faux italic and scaling are. Under a
/// warp, control points are mapped directly, which is exact for straight segments and a close
/// approximation for curves.
pub fn outlines(layout: &TextLayout, transform: &Affine, warp: Option<&Warp>) -> Vec<Vec<PathEl>> {
    let mut glyphs = Vec::new();
    for g in &layout.glyphs {
        let face = &layout.faces[g.face as usize];
        let Ok(font) = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index) else {
            continue;
        };
        let Some(outline) = font.outline_glyphs().get(GlyphId::new(g.id)) else {
            continue;
        };
        let coords: Vec<NormalizedCoord> = face.coords.iter().map(|&c| NormalizedCoord::from_bits(c)).collect();
        let mut glyph = glyph_xform(layout, g, face.skew_deg);
        let mut bend = warp;
        if let Some(w) = warp {
            match w.place_glyph(glyph_center(layout, g, &font, face.size_px, &coords)) {
                Placement::Bend => {}
                Placement::Hidden => continue,
                Placement::Frame(m) => {
                    glyph = Xform(m).mul(&glyph);
                    bend = None;
                }
            }
        }
        let mut rec = Recorder { glyph, post: Xform(transform.m), warp: bend, out: Vec::new() };
        let settings = DrawSettings::unhinted(Size::new(face.size_px), LocationRef::new(&coords));
        if outline.draw(settings, &mut rec).is_ok() && !rec.out.is_empty() {
            glyphs.push(rec.out);
        }
    }
    glyphs
}

#[cfg(test)]
mod tests {
    use super::{MAX_PIXELS, rect_from_bounds};
    use photocraft_geom::Rect;

    #[test]
    fn bounds_are_clipped_without_overflow_and_oversized_rectangles_rejected() {
        let min = f64::from(i32::MIN);
        let max = f64::from(i32::MAX);
        let cases = [
            ([min, 0.0, min + 8.0, 8.0], Rect::new(i32::MIN, -1, i32::MIN + 9, 9)),
            ([max - 8.0, 0.0, max, 8.0], Rect::new(i32::MAX - 9, -1, i32::MAX, 9)),
            ([-f64::MAX, 0.0, f64::MAX, 8.0], Rect::EMPTY),
            ([0.0, 0.0, MAX_PIXELS as f64, 8.0], Rect::EMPTY),
            ([0.25, -3.25, 10.1, 8.8], Rect::new(-1, -5, 12, 10)),
        ];

        for (bounds, expected) in cases {
            let first = rect_from_bounds(bounds);
            assert_eq!(first, expected, "bounds: {bounds:?}");
            assert_eq!(rect_from_bounds(bounds), first, "bounds: {bounds:?}");
        }
    }
}
