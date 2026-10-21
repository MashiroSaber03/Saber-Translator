//! The document model PhotoCraft imports: spreads, artboards, layers, groups, curves, shapes,
//! text and images, with transforms composed into document space. Anything else is reported
//! by [`Document::warnings`] instead of being dropped silently.

use std::collections::{BTreeMap, HashSet};

use crate::paint::{self, Paint, Stroke};
use crate::stream::{ObjId, Stream, Tag};
use crate::{Archive, Error, Limits, stream};

/// Affine map `x' = a·x + c·y + e`, `y' = b·x + d·y + f` (kurbo's coefficient order).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine(pub [f64; 6]);

impl Affine {
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    /// Affinity's `Xfrm`, stored row by row: (a, c, e, b, d, f).
    pub(crate) fn from_xfrm(m: [f64; 6]) -> Self {
        Self([m[0], m[3], m[1], m[4], m[2], m[5]])
    }

    pub fn then(self, outer: Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [oa, ob, oc, od, oe, of] = outer.0;
        Self([oa * a + oc * b, ob * a + od * b, oa * c + oc * d, ob * c + od * d, oa * e + oc * f + oe, ob * e + od * f + of])
    }

    pub fn apply(&self, p: Point) -> Point {
        let [a, b, c, d, e, f] = self.0;
        Point { x: a * p.x + c * p.y + e, y: b * p.x + d * p.y + f }
    }

    /// Geometric mean of the axis scales: how much a stroke width or font size grows.
    pub fn scale(&self) -> f64 {
        let [a, b, c, d, ..] = self.0;
        (a * d - b * c).abs().sqrt()
    }

    pub fn is_finite(&self) -> bool {
        self.0.iter().all(|v| v.is_finite())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

/// One cubic Bézier subpath, in document pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct SubPath {
    pub start: Point,
    /// (control 1, control 2, end) per segment.
    pub segments: Vec<[Point; 3]>,
    pub closed: bool,
}

/// An outline in document pixels.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Path {
    pub subpaths: Vec<SubPath>,
}

impl Path {
    pub fn transformed(&self, m: Affine) -> Self {
        Self {
            subpaths: self
                .subpaths
                .iter()
                .map(|s| SubPath { start: m.apply(s.start), segments: s.segments.iter().map(|seg| seg.map(|p| m.apply(p))).collect(), closed: s.closed })
                .collect(),
        }
    }
}

/// Affinity's blend modes, numbered by the `Blnd` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Blend {
    #[default]
    Normal,
    PassThrough,
    Darken,
    DarkerColor,
    Multiply,
    ColorBurn,
    Lighten,
    LighterColor,
    Screen,
    ColorDodge,
    Add,
    Overlay,
    SoftLight,
    HardLight,
    VividLight,
    PinLight,
    LinearLight,
    HardMix,
    Difference,
    Exclusion,
    Subtract,
    Hue,
    Saturation,
    Luminosity,
    Color,
    Average,
    Negation,
    Reflect,
    Glow,
    Erase,
}

impl Blend {
    /// The enum (id, version) pairs observed in a document whose layers are named after their mode.
    fn from_enum(id: u16, version: u16) -> Option<Self> {
        use Blend::*;
        Some(match (id, version) {
            (0, _) => Normal,
            (1, 0) => Darken,
            (2, 1) => DarkerColor,
            (2, 0) => Multiply,
            (3, 0) => ColorBurn,
            (4, 0) => Lighten,
            (6, 1) => LighterColor,
            (5, 0) => Screen,
            (6, 0) => ColorDodge,
            (7, 0) => Add,
            (8, 0) => Overlay,
            (9, 0) => SoftLight,
            (10, 0) => HardLight,
            (11, 0) => VividLight,
            (12, 0) => PinLight,
            (15, 1) => LinearLight,
            (13, 0) => HardMix,
            (14, 0) => Difference,
            (15, 0) => Exclusion,
            (16, 0) => Subtract,
            (17, 0) => Hue,
            (18, 0) => Saturation,
            (19, 0) => Luminosity,
            (20, 0) => Color,
            (21, 0) => Average,
            (22, 0) => Negation,
            (23, 0) => Reflect,
            (24, 0) => Glow,
            (25, 0) => Erase,
            _ => return None,
        })
    }
}

/// A run of text with one set of character attributes.
#[derive(Debug, Clone, PartialEq)]
pub struct TextRun {
    pub text: String,
    /// PostScript name and family of the chosen font.
    pub postscript: String,
    pub family: String,
    /// CSS weight (400 regular, 700 bold).
    pub weight: i64,
    pub italic: bool,
    /// Font size in document pixels, before the node's transform.
    pub size: f64,
    /// Tracking in em.
    pub tracking: f64,
    /// Fixed line pitch in document pixels, when the run overrides automatic leading.
    pub leading: Option<f64>,
    pub fill: Paint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
    Justify,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub runs: Vec<TextRun>,
    pub align: Align,
    /// Artistic text: the first baseline's anchor point in node space (start, centre or end of the
    /// line according to `align`). Frame text: the frame box in node space.
    pub anchor: Point,
    pub frame: Option<Rect>,
    /// Node space to document pixels.
    pub transform: Affine,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pixels {
    /// Straight-alpha RGBA, 8 bits per channel, row by row.
    Rgba8(Vec<u8>),
    /// The original file a placed image was made from (PNG, JPEG…), as embedded.
    Encoded(Vec<u8>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Pixels,
    /// Pixel space (0..w, 0..h) to document pixels.
    pub transform: Affine,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Kind {
    /// Layer container (`Scop`).
    Layer,
    Group,
    /// Artboard: a rectangle that clips its children; `background` paints behind them.
    Artboard {
        rect: Rect,
        background: Vec<Paint>,
    },
    /// Curve, parametric shape or compound shape: an outline with its fills and strokes.
    /// Children of a shape are clipped by its outline (painted over its fills, under its strokes).
    Shape {
        path: Path,
        fills: Vec<Paint>,
        strokes: Vec<Stroke>,
        even_odd: bool,
    },
    Text(Text),
    Image(Image),
    /// Something this reader does not import; its children are still imported.
    Unsupported,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    pub class: Tag,
    pub name: String,
    pub visible: bool,
    pub locked: bool,
    pub opacity: f64,
    pub blend: Blend,
    pub kind: Kind,
    /// Vector mask in document pixels: the node shows only inside it (nonzero fill).
    pub mask: Option<Path>,
    /// Pixel mask: grey levels (white shows the node, black hides it); outside it the node is hidden.
    pub pixel_mask: Option<Image>,
    /// Bottom to top.
    pub children: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spread {
    /// Spread bounds in document pixels (the canvas, or the union of the artboards).
    pub bounds: Rect,
    /// Pages of a Publisher-style spread (one, or two facing pages), in document pixels.
    pub pages: Vec<Rect>,
    /// Transparent background; otherwise the page is white.
    pub transparent: bool,
    /// Bottom to top.
    pub nodes: Vec<Node>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    /// Pixels per inch: all geometry is in document pixels at this resolution.
    pub dpi: f64,
    pub spreads: Vec<Spread>,
    /// Application that last saved the file, as stored (`Affinity 3.0.2`), when present.
    pub saved_by: Option<String>,
    /// Content that was not imported or was approximated, one line per kind, deduplicated.
    pub warnings: Vec<String>,
}

/// Deepest layer nesting followed; deeper content is reported, not imported.
const MAX_TREE_DEPTH: usize = 128;
/// Total layer nodes imported.
const MAX_NODES: usize = 500_000;

/// Read a document: archive, `doc.dat` and the model.
pub fn read(bytes: &[u8], limits: Limits) -> Result<Document, Error> {
    let mut archive = Archive::open(bytes, limits)?;
    let doc = archive.read("doc.dat")?;
    let s = stream::parse(&doc)?;
    Reader::new(&s, &mut archive).document()
}

pub(crate) struct Reader<'s, 'a, 'b> {
    pub(crate) s: &'s Stream,
    pub(crate) archive: &'b mut Archive<'a>,
    pub(crate) dpi: f64,
    warnings: BTreeMap<String, usize>,
    nodes: usize,
    active: HashSet<ObjId>,
}

impl<'s, 'a, 'b> Reader<'s, 'a, 'b> {
    fn new(s: &'s Stream, archive: &'b mut Archive<'a>) -> Self {
        Self { s, archive, dpi: 72.0, warnings: BTreeMap::new(), nodes: 0, active: HashSet::new() }
    }

    pub(crate) fn warn(&mut self, what: impl Into<String>) {
        *self.warnings.entry(what.into()).or_default() += 1;
    }

    fn document(mut self) -> Result<Document, Error> {
        let s = self.s;
        let root = s.root;
        self.dpi = s.obj(root, b"UVCn").and_then(|u| s.f64(u, b"UPPI")).filter(|d| *d > 0.0 && *d < 100_000.0).unwrap_or(72.0);
        let saved_by = s.obj(root, b"NVer").map(|v| {
            let product = s.str(v, b"Prod").unwrap_or("Affinity");
            match (s.int(v, b"Majr"), s.int(v, b"Minr"), s.int(v, b"Bild")) {
                (Some(a), Some(b), Some(c)) => format!("{product} {a}.{b}.{c}"),
                _ => product.to_string(),
            }
        });
        let doc = s.obj(root, b"DocR").ok_or(Error::Malformed("document has no root node"))?;
        let mut spreads = Vec::new();
        for spread in s.objs(doc, b"Chld") {
            if !s.is(spread, b"Sprd") {
                self.warn("an unknown kind of page");
                continue;
            }
            let pages = pages(s, spread);
            let default_size = s.floats::<2>(doc, b"DfSz").filter(|[w, h]| *w > 0.0 && *h > 0.0).map(|[w, h]| Rect { x0: 0.0, y0: 0.0, x1: w, y1: h });
            let bounds = s.floats::<4>(spread, b"SprB").map(rect).or_else(|| pages.iter().copied().reduce(union)).or(default_size).unwrap_or(Rect {
                x0: 0.0,
                y0: 0.0,
                x1: 0.0,
                y1: 0.0,
            });
            let transparent = s.bool(spread, b"SprT").unwrap_or(false);
            let (nodes, mask) = self.children(spread, Affine::IDENTITY, 0)?;
            if mask.is_some() {
                self.warn("mask layers directly on a page (imported without them)");
            }
            spreads.push(Spread { bounds, pages, transparent, nodes });
        }
        if spreads.is_empty() {
            return Err(Error::Malformed("document has no pages"));
        }
        if !s.objs(doc, b"MpCh").is_empty() {
            self.warn("master pages (their content is not placed on the pages that use them)");
        }
        let warnings = self.warnings.into_iter().map(|(w, n)| if n > 1 { format!("{w} ({n}×)") } else { w }).collect();
        Ok(Document { dpi: self.dpi, spreads, saved_by, warnings })
    }

    /// The children of `parent`, and the pixel mask a mask layer among them puts on the parent.
    fn children(&mut self, parent: ObjId, world: Affine, depth: usize) -> Result<(Vec<Node>, Option<Image>), Error> {
        let s = self.s;
        let mut out = Vec::new();
        let mut mask = None;
        for c in s.objs(parent, b"Chld") {
            if s.is(c, b"MRst") {
                // A mask layer inside a group or layer masks that container.
                if s.bool(c, b"Visi") == Some(false) {
                    continue;
                }
                let local = s.floats::<6>(c, b"Xfrm").map(Affine::from_xfrm).unwrap_or(Affine::IDENTITY);
                match crate::raster::mask(self, c, local.then(world))? {
                    Some(m) if mask.is_none() => mask = Some(m),
                    Some(_) => self.warn("several pixel masks on one layer (only the first is used)"),
                    None => {}
                }
                continue;
            }
            if let Some(n) = self.node(c, world, depth + 1)? {
                out.push(n);
            }
        }
        Ok((out, mask))
    }

    fn node(&mut self, id: ObjId, parent: Affine, depth: usize) -> Result<Option<Node>, Error> {
        let s = self.s;
        if depth > MAX_TREE_DEPTH {
            self.warn("layers nested deeper than 128 levels");
            return Ok(None);
        }
        self.nodes += 1;
        if self.nodes > MAX_NODES {
            return Err(Error::Limit("too many layers"));
        }
        // A node that contains itself (a shared object linked back into its own subtree).
        if !self.active.insert(id) {
            self.warn("a layer that contains itself");
            return Ok(None);
        }
        let local = s.floats::<6>(id, b"Xfrm").map(Affine::from_xfrm).unwrap_or(Affine::IDENTITY);
        let world = local.then(parent);
        let class = s.class(id).unwrap_or(Tag(0));
        let name = s.str(id, b"Desc").unwrap_or_default().to_string();
        let visible = s.bool(id, b"Visi").unwrap_or(true);
        let locked = s.bool(id, b"Edtb").is_some_and(|e| !e);
        let opacity = s.f64(id, b"Opac").map_or(1.0, |o| o.clamp(0.0, 1.0));
        let blend = match s.enumeration(id, b"Blnd") {
            None if s.bool(id, b"PasT") != Some(false) && is_container(class) => Blend::PassThrough,
            None => Blend::Normal,
            Some((i, v)) => Blend::from_enum(i, v).unwrap_or_else(|| {
                self.warn("an unknown blend mode (imported as Normal)");
                Blend::Normal
            }),
        };
        let kind = if !world.is_finite() {
            self.warn("a layer with an invalid transform");
            Kind::Unsupported
        } else {
            self.kind(id, class, world, parent)?
        };
        // A compound's children are its operands, already merged into its outline.
        let (children, child_mask) = if class == Tag::of(b"Comp") { (Vec::new(), None) } else { self.children(id, world, depth)? };
        let (mask, attached_mask) = self.attached_masks(id, world)?;
        let pixel_mask = attached_mask.or(child_mask);
        self.active.remove(&id);
        if s.objs(id, b"FiEf").iter().any(|e| s.bool(*e, b"Enab") != Some(false)) {
            self.warn("layer effects (shadows, glows, outlines, bevels…)");
        }
        Ok(Some(Node { class, name, visible, locked, opacity, blend, kind, mask, pixel_mask, children }))
    }

    fn kind(&mut self, id: ObjId, class: Tag, world: Affine, parent: Affine) -> Result<Kind, Error> {
        let s = self.s;
        Ok(match &class.0.to_be_bytes() {
            b"Scop" => Kind::Layer,
            b"Grup" => Kind::Group,
            b"PCrv" | b"ShpN" | b"Comp" | b"SNEN" | b"SNRR" | b"ShRN" => {
                // Current .af boards carry artboard properties (aprp) in phrp; older
                // documents use ABEn. Converted curve boards keep those properties too.
                // Other property classes also use phrp, so the field alone is not a marker.
                if s.bool(id, b"ABEn") == Some(true) || s.obj(id, b"phrp").is_some_and(|p| s.is(p, b"aprp")) {
                    if class != Tag::of(b"ShpN") {
                        self.warn("a converted artboard outline (imported at its bounding box)");
                    }
                    return Ok(self.artboard(id, world));
                }
                let Some((local, even_odd)) = crate::geometry::outline(self, id, class, world) else {
                    self.warn(format!("a {} that could not be read", describe(class)));
                    return Ok(Kind::Unsupported);
                };
                let (fills, strokes) = paint::node_paint(self, id, world);
                Kind::Shape { path: local.transformed(world), fills, strokes, even_odd }
            }
            b"TxtA" | b"TxtF" | b"TxtC" => match crate::text::read(self, id, world) {
                Some(t) => Kind::Text(t),
                None => {
                    self.warn("text whose story could not be read (master-page text, or an unknown layout)");
                    Kind::Unsupported
                }
            },
            b"ImgN" | b"Rstr" => match crate::raster::node_image(self, id, world)? {
                Some(img) => Kind::Image(img),
                None => Kind::Unsupported,
            },
            b"EmbN" => match crate::raster::embedded(self, id, world)? {
                Some(img) => {
                    self.warn("embedded documents and symbols (imported as pictures of them)");
                    Kind::Image(img)
                }
                None => {
                    self.warn("embedded documents and symbols without a cached picture");
                    Kind::Unsupported
                }
            },
            b"FRst" => {
                // A fill layer paints its whole bitmap area; its gradient lives in page space.
                let b = s.obj(id, b"Bitm");
                let size = |t: &[u8; 4]| b.and_then(|b| s.int(b, t)).map(|v| v as f64).filter(|v| *v > 0.0);
                match (size(b"BmpW"), size(b"BmpH")) {
                    (Some(w), Some(h)) => {
                        let local = Path { subpaths: vec![crate::shapes::rectangle(0.0, 0.0, w, h)] };
                        let fills = paint::fills(self, id, parent, b"BFFl");
                        Kind::Shape { path: local.transformed(world), fills, strokes: Vec::new(), even_odd: false }
                    }
                    _ => {
                        self.warn("fill layers without an area");
                        Kind::Unsupported
                    }
                }
            }
            _ => {
                self.warn(format!("{} layers", describe(class)));
                Kind::Unsupported
            }
        })
    }

    /// Vector shapes attached to a node (`AdCh`) mask it: it shows only inside their outlines;
    /// an attached mask layer is a pixel mask. Adjustments attached the same way are reported.
    fn attached_masks(&mut self, id: ObjId, world: Affine) -> Result<(Option<Path>, Option<Image>), Error> {
        let s = self.s;
        let attached = s.objs(id, b"AdCh");
        let mut mask: Option<Path> = None;
        let mut pixels: Option<Image> = None;
        for a in attached {
            let class = s.class(a).unwrap_or(Tag(0));
            if s.bool(a, b"Visi") == Some(false) {
                continue;
            }
            if class == Tag::of(b"PCrv") || class == Tag::of(b"ShpN") || class == Tag::of(b"Comp") {
                let local = s.floats::<6>(a, b"Xfrm").map(Affine::from_xfrm).unwrap_or(Affine::IDENTITY).then(world);
                if let Some((p, _)) = crate::geometry::outline(self, a, class, local) {
                    mask.get_or_insert_with(Path::default).subpaths.extend(p.transformed(local).subpaths);
                }
            } else if class == Tag::of(b"MRst") {
                let local = s.floats::<6>(a, b"Xfrm").map(Affine::from_xfrm).unwrap_or(Affine::IDENTITY).then(world);
                match crate::raster::mask(self, a, local)? {
                    Some(m) if pixels.is_none() => pixels = Some(m),
                    Some(_) => self.warn("several pixel masks on one layer (only the first is used)"),
                    None => {}
                }
            } else {
                self.warn(format!("{} layers attached to other layers (imported without them)", describe(class)));
            }
        }
        Ok((mask, pixels))
    }

    fn artboard(&mut self, id: ObjId, world: Affine) -> Kind {
        let s = self.s;
        let b = s.floats::<4>(id, b"ShpB").map(rect).unwrap_or(Rect { x0: 0.0, y0: 0.0, x1: 0.0, y1: 0.0 });
        let corners = [Point { x: b.x0, y: b.y0 }, Point { x: b.x1, y: b.y0 }, Point { x: b.x1, y: b.y1 }, Point { x: b.x0, y: b.y1 }].map(|p| world.apply(p));
        let [a, b2, c, d, ..] = world.0;
        if b2.abs() > 1e-9 || c.abs() > 1e-9 || a <= 0.0 || d <= 0.0 {
            self.warn("a rotated or flipped artboard (imported at its bounding box)");
        }
        let xs = corners.map(|p| p.x);
        let ys = corners.map(|p| p.y);
        let rect = Rect {
            x0: xs.iter().copied().fold(f64::INFINITY, f64::min),
            y0: ys.iter().copied().fold(f64::INFINITY, f64::min),
            x1: xs.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            y1: ys.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        };
        let background = paint::fills(self, id, world, b"BFFl");
        Kind::Artboard { rect, background }
    }
}

/// Page rectangles: `SpMd.PagR[].rctp` (Publisher 2, Affinity 3) or `SprB` split into `PagC`
/// equal pages (Publisher 1 facing spreads).
fn pages(s: &Stream, spread: ObjId) -> Vec<Rect> {
    if let Some(md) = s.obj(spread, b"SpMd") {
        let v: Vec<Rect> = s.objs(md, b"PagR").into_iter().filter_map(|p| s.floats::<4>(p, b"rctp").map(rect)).filter(|r| r.x1 > r.x0 && r.y1 > r.y0).collect();
        if !v.is_empty() {
            return v;
        }
    }
    match (s.floats::<4>(spread, b"SprB").map(rect), s.int(spread, b"PagC")) {
        (Some(b), Some(n @ 2..=16)) => {
            let w = (b.x1 - b.x0) / n as f64;
            (0..n).map(|i| Rect { x0: b.x0 + w * i as f64, x1: b.x0 + w * (i + 1) as f64, ..b }).collect()
        }
        _ => Vec::new(),
    }
}

pub(crate) fn union(a: Rect, b: Rect) -> Rect {
    Rect { x0: a.x0.min(b.x0), y0: a.y0.min(b.y0), x1: a.x1.max(b.x1), y1: a.y1.max(b.y1) }
}

fn is_container(class: Tag) -> bool {
    class == Tag::of(b"Scop") || class == Tag::of(b"Grup")
}

pub(crate) fn rect(v: [f64; 4]) -> Rect {
    Rect { x0: v[0], y0: v[1], x1: v[2], y1: v[3] }
}

/// A readable name for a layer class in warnings.
pub(crate) fn describe(class: Tag) -> &'static str {
    match &class.0.to_be_bytes() {
        b"PCrv" => "curve",
        b"ShpN" | b"SNEN" | b"SNRR" | b"ShRN" => "shape",
        b"Comp" => "compound",
        b"Rstr" => "pixel",
        b"ImgN" => "image",
        b"EmbN" => "embedded document",
        b"MPIN" => "master page instance",
        b"FRst" => "fill",
        b"MRst" => "mask",
        b"Slic" | b"SlcP" => "slice",
        b"3DLA" => "3D",
        [_, _, b'R', b'A'] => "adjustment",
        _ => "unknown",
    }
}
