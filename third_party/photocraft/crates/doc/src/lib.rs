//! The Photocraft document model: pure data, no rendering, no UI.
//!
//! Layer order: inside every group (and the root), `children[0]` is the **bottom** layer, matching
//! compositing order and the PSD file order. UIs display the list reversed.
//!
//! Non-destructive kinds (adjustment, fill, text, shape, smart object) and colour modes beyond RGB are
//! part of the model from day one, even where rendering support lands later (architecture §1.1).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod adjust;
pub mod advanced;
pub mod analysis;
pub mod blend_if;
pub mod comps;
pub mod effects;
pub mod mode;
pub mod pattern;
pub mod slices;
pub mod text;
pub mod text_styles;
pub mod variables;
pub mod vector;
pub mod video;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

pub use adjust::Adjustment;
pub use advanced::{AdvancedBlending, Knockout};
pub use analysis::{CountGroup, Measurement, MeasurementScale, Note, Ruler};
pub use blend_if::{BlendIf, BlendRange};
pub use comps::{Artboard, ArtboardBackground, CompAppearance, CompLayerState, LayerComp};
pub use effects::{
    Bevel, BevelContour, BevelStyle, BevelTechnique, BevelTexture, Contour, Effect, FxCommon, FxPaint, GlobalLight, Glow, GlowSource, GlowTechnique, Gradient,
    GradientStyle, Satin, Shadow, StrokeFx, StrokePosition,
};
pub use mode::{ColorTable, Duotone, DuotoneInk, StackMode};
pub use pattern::Pattern;
pub use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
pub use photocraft_geom::{Affine, Rect, Size};
pub use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};
pub use slices::{Slice, SliceKind, SliceOrigin, Slices};
pub use text_styles::TextStyles;
pub use variables::{DataSet, DataValue, PixelAlign, PixelMethod, VarKind, VariableDef, Variables};
pub use vector::{
    ClippingPath, FillRule, Knot, LineCap, LineJoin, LiveShape, NamedPath, Path, PathOp, ShapeLayer, ShapeStroke, StrokeAlign, Subpath, VectorMask,
};
pub use video::{Timeline, VideoData, VideoSource};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

impl LayerId {
    pub fn fresh() -> Self {
        LayerId(next_id())
    }
}

/// Identity of the source contents shared by smart-object instances, independent of their bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SmartContentsId(pub u64);

impl SmartContentsId {
    pub fn fresh() -> Self {
        SmartContentsId(next_id())
    }
}

/// Advance the process-wide id counter (shared by layer, document and smart-contents ids) past `max`, so ids
/// loaded from a file never collide with ids minted later.
pub fn ensure_ids_above(max: u64) {
    NEXT_ID.fetch_max(max.saturating_add(1), Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DocId(pub u64);

impl DocId {
    pub fn fresh() -> Self {
        DocId(next_id())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Locks {
    pub transparency: bool,
    pub pixels: bool,
    pub position: bool,
    pub artboard: bool,
    pub all: bool,
}

impl Locks {
    /// Set the lock a command names (`transparency`, `pixels`, `position`, `artboard`, `all`);
    /// false for an unknown name. Lock Image Pixels includes Lock Transparent Pixels, so, as in
    /// Photoshop, locking the pixels clears the transparency lock, and that one stays off while the
    /// pixels are locked (#4892).
    pub fn set(&mut self, key: &str, on: bool) -> bool {
        match key {
            "transparency" => self.transparency = on && !self.pixels,
            "pixels" => {
                self.pixels = on;
                if on {
                    self.transparency = false;
                }
            }
            "position" => self.position = on,
            "artboard" => self.artboard = on,
            "all" => self.all = on,
            _ => return false,
        }
        true
    }

    /// Whether any lock is set: Photoshop then shows the lock icon on the layer's row.
    pub fn any(self) -> bool {
        self.transparency || self.pixels || self.position || self.artboard || self.all
    }

    pub fn union(self, o: Locks) -> Locks {
        Locks {
            transparency: self.transparency || o.transparency,
            pixels: self.pixels || o.pixels,
            position: self.position || o.position,
            artboard: self.artboard || o.artboard,
            all: self.all || o.all,
        }
    }
}

/// Photoshop layer colour labels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LabelColor {
    #[default]
    None,
    Red,
    Orange,
    Yellow,
    Green,
    Blue,
    Violet,
    Gray,
    Seafoam,
    Indigo,
    Magenta,
    Fuchsia,
}

impl LabelColor {
    /// Display order, independent of the PSD sheet-colour indices.
    pub const ALL: [Self; 12] = [
        Self::None,
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Seafoam,
        Self::Blue,
        Self::Indigo,
        Self::Magenta,
        Self::Fuchsia,
        Self::Violet,
        Self::Gray,
    ];

    /// Stable, language-independent command and inspection value.
    pub fn id(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Red => "red",
            Self::Orange => "orange",
            Self::Yellow => "yellow",
            Self::Green => "green",
            Self::Seafoam => "seafoam",
            Self::Blue => "blue",
            Self::Indigo => "indigo",
            Self::Magenta => "magenta",
            Self::Fuchsia => "fuchsia",
            Self::Violet => "violet",
            Self::Gray => "gray",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::None => "No Color",
            Self::Red => "Red",
            Self::Orange => "Orange",
            Self::Yellow => "Yellow",
            Self::Green => "Green",
            Self::Seafoam => "Seafoam",
            Self::Blue => "Blue",
            Self::Indigo => "Indigo",
            Self::Magenta => "Magenta",
            Self::Fuchsia => "Fuchsia",
            Self::Violet => "Violet",
            Self::Gray => "Gray",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|c| c.id() == id)
    }
}

#[cfg(test)]
mod label_color_tests {
    use super::LabelColor;

    #[test]
    fn layer_color_ids_and_existing_serialized_names_stay_stable() {
        for (name, color) in [
            ("None", LabelColor::None),
            ("Red", LabelColor::Red),
            ("Orange", LabelColor::Orange),
            ("Yellow", LabelColor::Yellow),
            ("Green", LabelColor::Green),
            ("Blue", LabelColor::Blue),
            ("Violet", LabelColor::Violet),
            ("Gray", LabelColor::Gray),
        ] {
            assert_eq!(serde_json::to_value(color).unwrap(), name);
            assert_eq!(serde_json::from_value::<LabelColor>(serde_json::json!(name)).unwrap(), color);
        }
        for c in LabelColor::ALL {
            assert_eq!(LabelColor::from_id(c.id()), Some(c));
            assert_eq!(serde_json::from_value::<LabelColor>(serde_json::to_value(c).unwrap()).unwrap(), c);
        }
        for id in ["", "Red", "0", "unknown", "🔴"] {
            assert_eq!(LabelColor::from_id(id), None);
        }
        assert!(serde_json::from_value::<LabelColor>(serde_json::json!("Unknown")).is_err());
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerMask {
    /// Grayscale surface; untouched pixels read as `default` (0 = hide, 1 = reveal).
    pub surface: Surface,
    pub enabled: bool,
    pub linked: bool,
    pub density: f32,
    pub feather: f32,
}

impl LayerMask {
    pub fn reveal_all() -> Self {
        Self { surface: Surface::with_default(PixelFormat::GRAY8, &[1.0]), enabled: true, linked: true, density: 1.0, feather: 0.0 }
    }
    pub fn hide_all() -> Self {
        Self { surface: Surface::with_default(PixelFormat::GRAY8, &[0.0]), ..Self::reveal_all() }
    }
    /// Effective mask value at a pixel, including density.
    pub fn value(&self, x: i32, y: i32) -> f32 {
        if !self.enabled {
            return 1.0;
        }
        let v = self.surface.sample_channel(x, y, 0);
        1.0 - self.density * (1.0 - v)
    }

    /// Effective mask values over `r` (row-major) into `out`, including
    /// density; all ones when disabled.
    pub fn values_into(&self, r: Rect, out: &mut Vec<f32>) {
        let n = r.width() as usize * r.height() as usize;
        if !self.enabled {
            out.clear();
            out.resize(n, 1.0);
            return;
        }
        self.surface.read_region_into(r, out);
        let ch = self.surface.channels();
        if ch > 1 {
            let v: Vec<f32> = out.chunks_exact(ch).map(|p| p[0]).collect();
            *out = v;
        }
        if self.density < 1.0 {
            for v in out.iter_mut() {
                *v = 1.0 - self.density * (1.0 - *v);
            }
        }
    }
}

/// Layer styles. Parsed PSD effect data is kept raw until the effects engine lands (M9).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Effects {
    pub enabled: bool,
    pub items: Vec<Effect>,
    /// Original PSD `lfx2` (`lfxs` on groups) block data (object-effects version + descriptor),
    /// written back verbatim on PSD export. Not duplicated in `Layer::psd_blocks`.
    pub psd_raw: Option<Arc<Vec<u8>>>,
    /// Effects reference point (PSD `fxrp`): where linked patterns (Pattern Overlay, pattern
    /// strokes and glows) anchor their tiling. It moves with the layer. `None` = the layer's
    /// top-left.
    pub reference: Option<(f64, f64)>,
}

/// Deepest group nesting a document may hold: a layer inside this many nested groups is the
/// deepest legal one. Everything that walks the layer tree (engine lookups, the layers panel,
/// PSD export, `.pcraft` save and load) recurses once per level, so deeper trees risk
/// overflowing the 1 MiB main-thread stacks of Windows and wasm. PSD import and the engine
/// commands that nest layers (`layer.groupLayers`, `layer.moveTo`, Artboard from Layers) enforce
/// this up front; `photocraft-format` refuses to save deeper trees.
pub const MAX_GROUP_DEPTH: usize = 100;

#[derive(Clone, Debug, PartialEq)]
pub struct Group {
    pub children: Vec<Layer>,
    pub expanded: bool,
    /// Set when this (top-level) group is an artboard: children are clipped to its rect.
    pub artboard: Option<Artboard>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Fill {
    Solid(Color),
    /// Gradient between stops at `angle` degrees (`style` geometry, optionally reversed), laid
    /// out in the layer's frame (or the canvas when `align` is off), its centre moved by `offset`.
    Gradient {
        stops: Vec<(f32, Color)>,
        angle: f32,
        scale: f32,
        style: GradientStyle,
        reverse: bool,
        /// Opacity stops `(location, opacity)` (both `0..=1`); empty = each colour stop's alpha.
        /// When set, a pixel's alpha is the colour's alpha times the opacity at that location.
        #[serde(default)]
        opacity_stops: Vec<(f32, f32)>,
        /// Colour midpoints, one per segment between consecutive (sorted) colour stops: where
        /// the segment is half-way (`0..=1` of the segment). Missing entries are 0.5.
        #[serde(default)]
        midpoints: Vec<f32>,
        /// Centre offset as a fraction of the frame (PSD `Ofst`).
        #[serde(default)]
        offset: (f32, f32),
        /// Dither (breaks 8-bit banding with one level of noise).
        #[serde(default)]
        dither: bool,
        /// "Align with layer": lay out in the layer's frame (its masks' bounds), else the canvas.
        #[serde(default = "effects::yes")]
        align: bool,
    },
    /// A pattern from [`Document::patterns`] (looked up by `id`, then `name`).
    Pattern {
        name: String,
        scale: f32,
        #[serde(default)]
        id: String,
        /// Rotation in degrees (counter-clockwise).
        #[serde(default)]
        angle: f32,
        /// "Link with Layer": the pattern origin follows the layer's frame (else the canvas).
        #[serde(default = "effects::yes")]
        link: bool,
        /// Phase (origin offset) in pixels.
        #[serde(default)]
        phase: (f32, f32),
    },
}

impl Fill {
    /// A gradient fill with the classic settings: opaque stops at their alpha, 50 % midpoints,
    /// centred, no dither, aligned with the layer.
    pub fn gradient(stops: Vec<(f32, Color)>, angle: f32, scale: f32, style: GradientStyle, reverse: bool) -> Fill {
        Fill::Gradient { stops, angle, scale, style, reverse, opacity_stops: Vec::new(), midpoints: Vec::new(), offset: (0.0, 0.0), dither: false, align: true }
    }

    /// This fill with its colours in `mode`'s model (see [`Color::in_mode`]): a fill layer's
    /// colours are in its document's mode, which the compositors read them as.
    pub fn in_mode(mut self, mode: ColorMode) -> Fill {
        match &mut self {
            Fill::Solid(c) => *c = c.in_mode(mode),
            Fill::Gradient { stops, .. } => stops.iter_mut().for_each(|(_, c)| *c = c.in_mode(mode)),
            Fill::Pattern { .. } => {}
        }
        self
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct TextLayer {
    pub text: String,
    pub font_family: String,
    pub size_pt: f32,
    pub color: Color,
    pub transform: Affine,
    /// Rasterized appearance (from PSD or our text engine).
    pub cache: Option<Surface>,
    /// Data of the PSD `TySh` block. On export it replaces the `TySh` entry in
    /// [`Layer::psd_blocks`] (the text engine updates this field when it edits text).
    pub psd_raw: Option<Arc<Vec<u8>>>,
    /// Character style runs (see [`text`]). Empty = one run built from `font_family`, `size_pt`
    /// and `color`, which otherwise mirror the first run (summary for simple UIs).
    pub runs: Vec<text::TextRun>,
    /// Paragraph style runs. Empty = default paragraph style.
    pub paragraphs: Vec<text::ParagraphRun>,
    /// Point or paragraph (box) text.
    pub shape: text::TextShape,
    /// Keep automatic wrapping but draw all lines/columns beyond the paragraph frame.
    /// The frame remains layout geometry; this never changes detection or repair bounds.
    pub overflow_visible: bool,
    pub block_align: text::BlockAlign,
    pub orientation: text::Orientation,
    pub antialias: text::AntiAlias,
    /// Warp Text settings (applied to the glyph outlines when rendering).
    pub warp: Option<text::TextWarp>,
}

impl Default for TextLayer {
    fn default() -> Self {
        Self {
            text: String::new(),
            font_family: String::new(),
            size_pt: 12.0,
            color: Color::BLACK,
            transform: Affine::IDENTITY,
            cache: None,
            psd_raw: None,
            runs: Vec::new(),
            paragraphs: Vec::new(),
            shape: text::TextShape::Point,
            overflow_visible: false,
            block_align: text::BlockAlign::Start,
            orientation: text::Orientation::Horizontal,
            antialias: text::AntiAlias::Smooth,
            warp: None,
        }
    }
}

impl TextLayer {
    /// Character runs covering exactly `text.len()` bytes, each on a char boundary.
    pub fn char_runs(&self) -> Vec<text::TextRun> {
        let base = text::CharStyle { font_family: self.font_family.clone(), size_pt: self.size_pt, color: self.color, ..Default::default() };
        normalize_runs(&self.text, self.runs.iter().map(|r| (r.len, r.style.clone())).collect(), base)
            .into_iter()
            .map(|(len, style)| text::TextRun { len, style })
            .collect()
    }
    /// Paragraph runs covering exactly `text.len()` bytes, with an optional zero-length
    /// final run for an empty paragraph after a trailing line break.
    pub fn paragraph_runs(&self) -> Vec<text::ParagraphRun> {
        let mut runs = normalize_runs(&self.text, self.paragraphs.iter().map(|r| (r.len, r.style.clone())).collect(), text::ParagraphStyle::default());
        // The final empty paragraph owns the synthetic terminal paragraph marker in PSD
        // text. Its formatting has no source bytes; do not lose that last run when normalizing.
        if self.text.ends_with(['\r', '\n'])
            && let Some((last, preceding)) = self.paragraphs.split_last()
            && last.len <= 1
            && preceding.iter().try_fold(0usize, |n, r| n.checked_add(r.len)) == Some(self.text.len())
        {
            runs.push((0, last.style.clone()));
        }
        runs.into_iter().map(|(len, style)| text::ParagraphRun { len, style }).collect()
    }
    /// Copies the first run's family, size and colour into the summary fields.
    pub fn sync_summary(&mut self) {
        if let Some(r) = self.runs.first() {
            self.font_family = r.style.font_family.clone();
            self.size_pt = r.style.size_pt;
            self.color = r.style.color;
        }
    }
}

/// Stretches/truncates `(len, style)` runs to cover `text` exactly, snapping to char boundaries
/// and dropping empty runs (keeps one run for empty text).
fn normalize_runs<S: Clone>(text: &str, runs: Vec<(usize, S)>, base: S) -> Vec<(usize, S)> {
    let total = text.len();
    if total == 0 {
        return vec![(0, runs.into_iter().next().map_or(base, |r| r.1))];
    }
    let mut out: Vec<(usize, S)> = Vec::new();
    let mut at = 0usize;
    for (len, style) in runs {
        if at >= total {
            break;
        }
        let mut end = at.saturating_add(len).min(total);
        while !text.is_char_boundary(end) {
            end += 1;
        }
        if end > at {
            out.push((end - at, style));
            at = end;
        }
    }
    if at < total {
        match out.last_mut() {
            Some(last) => last.0 += total - at,
            None => out.push((total, base.clone())),
        }
    }
    if out.is_empty() {
        out.push((0, base));
    }
    out
}

#[derive(Clone, Debug, PartialEq)]
pub struct SmartObject {
    /// Ordinary layer duplicates share this identity; independent contents receive a fresh one.
    pub contents_id: SmartContentsId,
    /// Embedded source file bytes (PSD, PNG, …) or linked path.
    pub source: SmartSource,
    pub transform: Affine,
    pub smart_filters: Vec<SmartFilter>,
    /// Rendered appearance.
    pub cache: Option<Surface>,
    /// Data of the PSD placed-layer block (`SoLd`, `PlLd` or `SoLE`). On export
    /// it replaces that entry in [`Layer::psd_blocks`].
    pub psd_raw: Option<Arc<Vec<u8>>>,
    /// Layer › Smart Filter › Disable Smart Filters turns the whole stack off
    /// without touching each filter's own visibility.
    pub filters_enabled: bool,
    /// The smart filter mask (one per smart object, as in Photoshop): where the
    /// filter stack's result shows over the unfiltered content.
    pub filter_mask: Option<LayerMask>,
    /// Edit › Transform › Warp on the smart object, in source-image coordinates; applied before
    /// `transform` when re-rendering from the source, so warping stays lossless.
    pub warp: Option<photocraft_geom::warp::Warp>,
    /// Layer › Smart Objects › Stack Mode: when set, the source's top-level layers are combined
    /// per pixel with this statistic instead of composited.
    pub stack_mode: Option<StackMode>,
    /// Distort / Perspective (or an imported placement whose corners aren't a parallelogram): the
    /// full projective map from source pixels to document pixels, row-major 3×3. It overrides
    /// `transform`, which then holds its affine approximation at the source origin.
    pub perspective: Option<[f64; 9]>,
}

impl SmartObject {
    /// A smart object with no filters and no PSD data.
    pub fn new(source: SmartSource, transform: Affine, cache: Option<Surface>) -> Self {
        Self {
            contents_id: SmartContentsId::fresh(),
            source,
            transform,
            smart_filters: Vec::new(),
            cache,
            psd_raw: None,
            filters_enabled: true,
            filter_mask: None,
            warp: None,
            stack_mode: None,
            perspective: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum SmartSource {
    Embedded { file_name: String, bytes: Arc<Vec<u8>> },
    Linked { path: String },
}

/// A filter applied non-destructively: a command id plus its parameters (data, not code).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SmartFilter {
    pub command: String,
    pub params: serde_json::Value,
    pub blend: BlendMode,
    pub opacity: f32,
    pub visible: bool,
}

/// Pixels rendered by another application (Photoshop) for a fill layer.
///
/// Used by the compositor instead of rendering [`Fill`] itself, but only while
/// `fill` still equals the layer's current fill (editing the fill invalidates
/// the cache automatically).
#[derive(Clone, Debug, PartialEq)]
pub struct FillCache {
    /// The fill these pixels were rendered for.
    pub fill: Fill,
    /// Rendered pixels in document coordinates.
    pub surface: Surface,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerContent {
    Raster(Surface),
    Group(Group),
    Adjustment(Adjustment),
    Fill(Fill),
    Text(TextLayer),
    Shape(ShapeLayer),
    Smart(SmartObject),
}

impl LayerContent {
    pub fn kind_name(&self) -> &'static str {
        match self {
            LayerContent::Raster(_) => "Pixel",
            LayerContent::Group(_) => "Group",
            LayerContent::Adjustment(_) => "Adjustment",
            LayerContent::Fill(_) => "Fill",
            LayerContent::Text(_) => "Type",
            LayerContent::Shape(_) => "Shape",
            LayerContent::Smart(_) => "Smart Object",
        }
    }

    /// English indefinite article for this layer kind ("an" for Adjustment, "a" for all others).
    pub fn article(&self) -> &'static str {
        match self {
            LayerContent::Adjustment(_) => "an",
            _ => "a",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub locks: Locks,
    pub blend: BlendMode,
    /// Layer opacity (applies to content and effects).
    pub opacity: f32,
    /// Fill opacity (applies to content but not effects).
    pub fill_opacity: f32,
    /// Clipped to the layer below (clipping mask).
    pub clipped: bool,
    pub mask: Option<LayerMask>,
    /// Vector mask (Layer › Vector Mask), applied together with `mask`. Shape layers keep their
    /// outline in [`ShapeLayer::path`] instead.
    pub vector_mask: Option<VectorMask>,
    pub effects: Effects,
    pub label: LabelColor,
    pub content: LayerContent,
    /// Preserved PSD additional-layer-info blocks (key, data) for lossless
    /// round-trip of anything not modelled above: vector masks, blending
    /// options, layer version, and the original adjustment/fill blocks (the
    /// PSD writer reuses an adjustment or fill block verbatim while it still
    /// decodes to the layer's current parameters, so Levels/Curves/Hue-Sat
    /// extras survive). Keys regenerated from the fields above (`luni`,
    /// `lyid`, `lsct`, `lsdk`, `iOpa`, `lspf`, `lclr`, `lfx2`) are not stored.
    pub psd_blocks: Vec<([u8; 4], Arc<Vec<u8>>)>,
    /// PSD layer id (`lyid`), kept so ids stay stable across round trips.
    /// Cleared by [`Layer::duplicate`].
    pub psd_id: Option<u32>,
    /// Photoshop-rendered pixels for fill layers (see [`FillCache`]).
    pub fill_cache: Option<FillCache>,
    /// Layer › Link Layers: layers sharing a link group move together. `None` = not linked.
    pub link_group: Option<u64>,
    /// Blending Options › Advanced Blending › Channels: bit `i` set = colour channel `i` of the
    /// document's mode (R, G, B / C, M, Y, K / L, a, b / Gray) is left out of blending, so the
    /// backdrop's value is kept there. 0 = every channel blends (the default). PSD `brst`.
    pub excluded_channels: u32,
    /// Blending Options › Blend If: value ranges of this layer and of the layers beneath it
    /// outside which the layer's pixels are hidden. Default = everything blends. PSD layer-record
    /// blending ranges.
    pub blend_if: BlendIf,
    /// Blending Options › Advanced Blending: knockout, blend interior effects / clipped layers as
    /// group, transparency shapes layer, layer / vector mask hides effects. Default = Photoshop's.
    pub advanced: AdvancedBlending,
    /// Layer › Video Layers frame stack (None for a normal layer).
    pub video: Option<VideoData>,
}

impl Layer {
    /// Whether the layer has the Background's own properties: an unclipped pixel layer with
    /// transparency and position locked. Only the bottom layer of a document with them is the
    /// Background: use [`Document::background`] / [`Document::is_background`] to identify it.
    pub fn has_background_traits(&self) -> bool {
        self.locks.transparency && self.locks.position && !self.clipped && matches!(self.content, LayerContent::Raster(_))
    }

    pub fn new(name: impl Into<String>, content: LayerContent) -> Self {
        Layer {
            id: LayerId::fresh(),
            name: name.into(),
            visible: true,
            locks: Locks::default(),
            blend: BlendMode::Normal,
            opacity: 1.0,
            fill_opacity: 1.0,
            clipped: false,
            mask: None,
            vector_mask: None,
            effects: Effects { enabled: true, ..Default::default() },
            label: LabelColor::None,
            content,
            psd_blocks: Vec::new(),
            psd_id: None,
            fill_cache: None,
            link_group: None,
            excluded_channels: 0,
            blend_if: BlendIf::default(),
            advanced: AdvancedBlending::default(),
            video: None,
        }
    }
    pub fn raster(name: impl Into<String>, format: PixelFormat) -> Self {
        Self::new(name, LayerContent::Raster(Surface::new(format)))
    }
    pub fn group(name: impl Into<String>, children: Vec<Layer>) -> Self {
        let mut l = Self::new(name, LayerContent::Group(Group { children, expanded: true, artboard: None }));
        l.blend = BlendMode::PassThrough;
        l
    }
    pub fn is_group(&self) -> bool {
        matches!(self.content, LayerContent::Group(_))
    }
    pub fn children(&self) -> Option<&[Layer]> {
        match &self.content {
            LayerContent::Group(g) => Some(&g.children),
            _ => None,
        }
    }
    pub fn children_mut(&mut self) -> Option<&mut Vec<Layer>> {
        match &mut self.content {
            LayerContent::Group(g) => Some(&mut g.children),
            _ => None,
        }
    }
    pub fn surface(&self) -> Option<&Surface> {
        match &self.content {
            LayerContent::Raster(s) => Some(s),
            LayerContent::Text(t) => t.cache.as_ref(),
            LayerContent::Shape(s) => s.cache.as_ref(),
            LayerContent::Smart(s) => s.cache.as_ref(),
            _ => None,
        }
    }
    pub fn surface_mut(&mut self) -> Option<&mut Surface> {
        match &mut self.content {
            LayerContent::Raster(s) => Some(s),
            _ => None,
        }
    }
    /// Deep copy with fresh ids (for Duplicate Layer).
    pub fn duplicate(&self) -> Layer {
        let mut l = self.clone();
        l.reassign_ids();
        l
    }
    fn reassign_ids(&mut self) {
        self.id = LayerId::fresh();
        self.psd_id = None;
        if let Some(ch) = self.children_mut() {
            for c in ch {
                c.reassign_ids();
            }
        }
    }
}

/// Ruler guides in document pixels. f64 like the rest of the view geometry: a guide placed in
/// ruler units (1 cm at 300 ppi is 118.110236… px) or read from a PSD's 1/32 px positions keeps
/// its exact place.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Guides {
    pub horizontal: Vec<f64>,
    pub vertical: Vec<f64>,
    /// Guides given their own colour (View › Guides › New Guide… › Color). The rest use
    /// Preferences › Guides, Grid & Slices › Guide colour.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub colors: Vec<GuideColor>,
}

/// One guide's own colour. It is keyed by orientation and position, not by index, so code that
/// drops or reorders guide positions (crop, artboards, guide layouts) can never move a colour onto
/// another guide: a guide whose entry no longer matches simply uses the preference colour.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GuideColor {
    pub vertical: bool,
    pub position: f64,
    /// sRGB, 0–255.
    pub rgb: [u8; 3],
}

impl Guides {
    /// The guide's own colour, if it has one.
    pub fn color(&self, vertical: bool, position: f64) -> Option<[u8; 3]> {
        self.colors.iter().find(|c| c.vertical == vertical && c.position == position).map(|c| c.rgb)
    }

    /// Give the guide at `position` its own colour (`None` returns it to the preference colour).
    pub fn set_color(&mut self, vertical: bool, position: f64, rgb: Option<[u8; 3]>) {
        self.colors.retain(|c| !(c.vertical == vertical && c.position == position));
        if let Some(rgb) = rgb {
            self.colors.push(GuideColor { vertical, position, rgb });
        }
    }

    /// Drop colours whose guide no longer exists (after a guide moved, was deleted or cleared).
    pub fn prune_colors(&mut self) {
        let (h, v) = (&self.horizontal, &self.vertical);
        self.colors.retain(|c| if c.vertical { v.contains(&c.position) } else { h.contains(&c.position) });
    }
}

/// Which areas an alpha channel's overlay colour marks (Channel Options › Color Indicates).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ColorIndicates {
    /// Colour shows over black (unselected) pixels: Photoshop's default.
    #[default]
    MaskedAreas,
    /// Colour shows over white (selected) pixels.
    SelectedAreas,
}

/// An extra (alpha or spot) channel, also used for the temporary Quick Mask channel.
///
/// The surface is grayscale at the document depth; white = selected (1.0), black = masked.
#[derive(Clone, Debug, PartialEq)]
pub struct AlphaChannel {
    pub name: String,
    pub surface: Surface,
    /// Spot colour channels carry their ink colour and solidity.
    pub spot: Option<(Color, f32)>,
    /// Overlay colour shown when the channel is viewed together with the composite.
    pub color: Color,
    /// Overlay opacity (0..1).
    pub opacity: f32,
    pub indicates: ColorIndicates,
}

impl AlphaChannel {
    /// Photoshop's default overlay: red at 50% over masked areas.
    pub const DEFAULT_COLOR: Color = Color::rgb(1.0, 0.0, 0.0);

    pub fn new(name: impl Into<String>, surface: Surface) -> Self {
        Self { name: name.into(), surface, spot: None, color: Self::DEFAULT_COLOR, opacity: 0.5, indicates: ColorIndicates::MaskedAreas }
    }

    /// Overlay coverage (0..1, before opacity) for channel value `v`: spot channels show ink
    /// where the channel is white, alpha channels follow [`ColorIndicates`].
    pub fn overlay_coverage(&self, v: f32) -> f32 {
        if self.spot.is_some() || self.indicates == ColorIndicates::SelectedAreas { v } else { 1.0 - v }
    }
}

/// A raw PSD global tagged block: (signature, key, data).
pub type PsdGlobalBlock = ([u8; 4], [u8; 4], Arc<Vec<u8>>);

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Metadata {
    pub xmp: Option<String>,
    pub exif: Option<Arc<Vec<u8>>>,
    /// Free-form image text (PNG tEXt/zTXt/iTXt, TIFF ASCII tags), including duplicate keys.
    pub text: Vec<(String, String)>,
    /// Raw PSD image resources we don't model yet: (id, name, data), for lossless round-trip.
    pub psd_resources: Vec<(u16, String, Arc<Vec<u8>>)>,
    /// Raw PSD global additional-layer-info blocks: (signature, key, data),
    /// e.g. `lnk2` embedded smart-object files, `Patt` patterns, `Txt2`.
    pub psd_global_blocks: Vec<PsdGlobalBlock>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub id: DocId,
    pub name: String,
    pub size: Size,
    pub resolution_dpi: f32,
    pub mode: ColorMode,
    pub depth: SampleType,
    pub icc_profile: Option<Arc<Vec<u8>>>,
    /// Bottom-to-top.
    pub layers: Vec<Layer>,
    pub channels: Vec<AlphaChannel>,
    pub guides: Guides,
    /// The rulers' zero point in document pixels (`[x, y]`), dragged from the ruler corner. The
    /// rulers, the Info panel and other position readouts measure from it; `[0, 0]` (the
    /// image's top-left) is the default.
    pub ruler_origin: [f64; 2],
    /// Active selection as a grayscale coverage surface (None = no selection).
    pub selection: Option<Surface>,
    pub metadata: Metadata,
    /// Global light used by layer effects (PSD resources 1037/1049).
    pub global_light: GlobalLight,
    /// Saved paths (Paths panel), top to bottom.
    pub paths: Vec<NamedPath>,
    /// The (unsaved) work path.
    pub work_path: Option<Path>,
    /// Clipping path used on export (names one of `paths`).
    pub clipping_path: Option<ClippingPath>,
    /// Select › Edit in Quick Mask Mode: the temporary "Quick Mask" channel while the mode is on.
    /// Part of the document so entering and leaving are history steps (as in Photoshop).
    pub quick_mask: Option<AlphaChannel>,
    /// Quick Mask was entered with no selection, so an untouched white mask (Color Indicates
    /// Masked Areas) leaves no selection rather than Select All.
    pub quick_mask_unselected: bool,
    /// Patterns stored with the document (PSD `Patt`/`Pat2`/`Pat3`): those its layers use, plus
    /// any imported with it.
    pub patterns: Vec<Pattern>,
    /// Indexed Color palette (Image › Mode › Color Table). Set while `mode` is Indexed.
    pub color_table: Option<ColorTable>,
    /// Duotone inks. Set while `mode` is Duotone.
    pub duotone: Option<Duotone>,
    /// Window › Layer Comps, top to bottom as listed in the panel.
    pub layer_comps: Vec<LayerComp>,
    /// Id of the comp applied last (PSD `lastAppliedComp`); None = the document's own state.
    pub last_applied_comp: Option<u32>,
    /// "Last Document State": layer state saved when a comp is applied over the document's own
    /// state, restored by `layerComp.restoreLastDocumentState`. Its id is 0.
    pub last_document_state: Option<LayerComp>,
    /// Image › Analysis: measurement scale, Count tool groups and the Ruler line.
    pub measurement: Measurement,
    /// Note tool annotations (PSD `Anno`).
    pub notes: Vec<Note>,
    /// Character and paragraph styles (Window › Character Styles / Paragraph Styles).
    pub text_styles: TextStyles,
    /// Web slices (Slice tool, layer-based slices; PSD resource 1050). Auto slices are derived.
    pub slices: Slices,
    /// Image › Variables and Data Sets (data-driven graphics).
    pub variables: Variables,
    /// Window › Timeline (None until a video timeline is created).
    pub timeline: Option<Timeline>,
    /// How the document's lossy file was encoded (its JPEG quality, WebP lossy or lossless), so
    /// a save writes it back the same way rather than at fixed defaults.
    pub file_encoding: FileEncoding,
}

/// The encoder settings a document's file was opened or saved with: what File › Save and the
/// Save As defaults reuse for that format. Empty for a new document or a file without lossy
/// options. Qualities are on the codecs' 1–100 scale.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct FileEncoding {
    /// JPEG quality (estimated from the file's quantisation tables on open).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub jpeg_quality: Option<u8>,
    /// WebP: `Some(true)` lossless (VP8L), `Some(false)` lossy (VP8 at [`Self::webp_quality`]).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webp_lossless: Option<bool>,
    /// Lossy WebP quality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub webp_quality: Option<u8>,
}

impl FileEncoding {
    pub fn is_empty(&self) -> bool {
        *self == FileEncoding::default()
    }
}

/// Where a layer lives in the tree: indices from the root down.
pub type LayerPath = Vec<usize>;

impl Document {
    pub fn new(name: impl Into<String>, size: Size, mode: ColorMode, depth: SampleType) -> Self {
        Document {
            id: DocId::fresh(),
            name: name.into(),
            size,
            resolution_dpi: 72.0,
            mode,
            depth,
            icc_profile: None,
            layers: Vec::new(),
            channels: Vec::new(),
            guides: Guides::default(),
            ruler_origin: [0.0, 0.0],
            selection: None,
            metadata: Metadata::default(),
            global_light: GlobalLight::default(),
            paths: Vec::new(),
            work_path: None,
            clipping_path: None,
            quick_mask: None,
            quick_mask_unselected: false,
            patterns: Vec::new(),
            color_table: None,
            duotone: None,
            layer_comps: Vec::new(),
            last_applied_comp: None,
            last_document_state: None,
            measurement: Measurement::default(),
            notes: Vec::new(),
            text_styles: TextStyles::default(),
            slices: Slices::default(),
            variables: Variables::default(),
            timeline: None,
            file_encoding: FileEncoding::default(),
        }
    }

    /// New document with a filled "Background" layer (like File → New).
    pub fn with_background(name: impl Into<String>, size: Size, mode: ColorMode, depth: SampleType, fill: Color) -> Self {
        let mut d = Self::new(name, size, mode, depth);
        let mut bg = Layer::raster("Background", d.pixel_format());
        bg.locks.transparency = true;
        bg.locks.position = true;
        if let Some(s) = bg.surface_mut() {
            let rgba = fill.to_rgb();
            let px = photocraft_raster::from_rgba(&s.format(), [rgba[0], rgba[1], rgba[2], fill.alpha]);
            s.fill_rect(d.bounds(), &px);
        }
        d.layers.push(bg);
        d
    }

    /// Pixel format for new raster layers in this document.
    pub fn pixel_format(&self) -> PixelFormat {
        let mode = match self.mode {
            ColorMode::Indexed | ColorMode::Multichannel => ColorMode::Rgb,
            ColorMode::Bitmap | ColorMode::Duotone => ColorMode::Grayscale,
            m => m,
        };
        PixelFormat::new(mode, self.depth, true)
    }

    pub fn bounds(&self) -> Rect {
        Rect::from_size(self.size)
    }

    /// Nesting depth of the deepest layer: 0 for a root-level layer, 1 inside one group, and so
    /// on. Computed over an explicit stack, so it cannot overflow on any tree it measures.
    pub fn max_group_depth(&self) -> usize {
        let mut max = 0;
        let mut stack: Vec<(&[Layer], usize)> = vec![(&self.layers, 0)];
        while let Some((layers, depth)) = stack.pop() {
            if layers.is_empty() {
                continue;
            }
            max = max.max(depth);
            for l in layers {
                if let Some(ch) = l.children()
                    && !ch.is_empty()
                {
                    stack.push((ch, depth + 1));
                }
            }
        }
        max
    }

    /// Depth-first walk yielding `(path, depth, layer)` bottom-to-top.
    pub fn walk(&self) -> Vec<(LayerPath, usize, &Layer)> {
        fn rec<'a>(layers: &'a [Layer], prefix: &mut LayerPath, out: &mut Vec<(LayerPath, usize, &'a Layer)>) {
            for (i, l) in layers.iter().enumerate() {
                prefix.push(i);
                out.push((prefix.clone(), prefix.len() - 1, l));
                if let Some(ch) = l.children() {
                    rec(ch, prefix, out);
                }
                prefix.pop();
            }
        }
        let mut out = Vec::new();
        rec(&self.layers, &mut Vec::new(), &mut out);
        out
    }

    pub fn layer_count(&self) -> usize {
        self.walk().len()
    }

    pub fn path_of(&self, id: LayerId) -> Option<LayerPath> {
        fn rec(layers: &[Layer], id: LayerId, path: &mut LayerPath) -> bool {
            for (i, l) in layers.iter().enumerate() {
                path.push(i);
                if l.id == id || l.children().is_some_and(|ch| rec(ch, id, path)) {
                    return true;
                }
                path.pop();
            }
            false
        }
        let mut path = Vec::new();
        rec(&self.layers, id, &mut path).then_some(path)
    }

    /// The locks in force on the layer at `path`: its own and those of every group around it,
    /// since locking a group locks its contents.
    pub fn locks_at(&self, path: &[usize]) -> Locks {
        (1..=path.len()).filter_map(|n| self.layer_at(path.get(..n)?)).fold(Locks::default(), |a, l| a.union(l.locks))
    }

    /// Whether the layer at `path` is shown: it and every group around it are visible.
    pub fn shown_at(&self, path: &[usize]) -> bool {
        (1..=path.len()).all(|n| path.get(..n).and_then(|p| self.layer_at(p)).is_some_and(|l| l.visible))
    }

    /// [`Self::locks_at`] for the layer `id`.
    pub fn effective_locks(&self, id: LayerId) -> Locks {
        self.path_of(id).map_or_else(Locks::default, |p| self.locks_at(&p))
    }

    /// The Background layer, identified structurally as Photoshop does, never by its name (a file
    /// from a German or Japanese Photoshop names it "Hintergrund" or "背景"): the bottom layer of
    /// the stack, a pixel layer that is not clipped, with transparency and position locked (a PSD
    /// stores it without a transparency channel and transparency-protected; File › New and opaque
    /// flat files create it the same way).
    pub fn background(&self) -> Option<&Layer> {
        self.layers.first().filter(|l| Layer::has_background_traits(l))
    }

    /// [`Self::background`], mutably.
    pub fn background_mut(&mut self) -> Option<&mut Layer> {
        self.layers.first_mut().filter(|l| Layer::has_background_traits(l))
    }

    /// Whether `id` is the document's Background layer ([`Self::background`]).
    pub fn is_background(&self, id: LayerId) -> bool {
        self.background().is_some_and(|b| b.id == id)
    }

    pub fn layer(&self, id: LayerId) -> Option<&Layer> {
        fn rec(layers: &[Layer], id: LayerId) -> Option<&Layer> {
            layers.iter().find_map(|l| if l.id == id { Some(l) } else { l.children().and_then(|ch| rec(ch, id)) })
        }
        rec(&self.layers, id)
    }

    pub fn layer_mut(&mut self, id: LayerId) -> Option<&mut Layer> {
        let path = self.path_of(id)?;
        self.layer_at_mut(&path)
    }

    /// A [`Layer::link_group`] id no layer uses yet: one above the highest stored id, or, when a
    /// loaded file already holds `u64::MAX` (#961), the smallest free id from 1. Never wraps, so a
    /// new link can't join an unrelated group. `None` only if every id is taken.
    pub fn unused_link_group(&self) -> Option<u64> {
        let mut used: Vec<u64> = self.walk().iter().filter_map(|(_, _, l)| l.link_group).collect();
        used.sort_unstable();
        used.dedup();
        if let Some(next) = used.last().map_or(Some(1), |g| g.checked_add(1)) {
            return Some(next);
        }
        let mut free = 1u64;
        for g in used {
            if g > free {
                break;
            }
            if g == free {
                free = free.checked_add(1)?;
            }
        }
        Some(free)
    }

    pub fn layer_at(&self, path: &[usize]) -> Option<&Layer> {
        let (first, rest) = path.split_first()?;
        let mut cur = self.layers.get(*first)?;
        for &i in rest {
            cur = cur.children()?.get(i)?;
        }
        Some(cur)
    }

    pub fn layer_at_mut(&mut self, path: &[usize]) -> Option<&mut Layer> {
        let (first, rest) = path.split_first()?;
        let mut cur = self.layers.get_mut(*first)?;
        for &i in rest {
            cur = cur.children_mut()?.get_mut(i)?;
        }
        Some(cur)
    }

    /// The sibling list containing `path` and the index within it.
    fn siblings_mut(&mut self, path: &[usize]) -> Option<(&mut Vec<Layer>, usize)> {
        let (&last, parent) = path.split_last()?;
        if parent.is_empty() {
            return Some((&mut self.layers, last));
        }
        let p = self.layer_at_mut(parent)?;
        Some((p.children_mut()?, last))
    }

    /// Insert `layer` directly above the layer `above` (or at the top of the root if None).
    pub fn insert_above(&mut self, above: Option<LayerId>, layer: Layer) -> LayerId {
        let id = layer.id;
        match above.and_then(|a| self.path_of(a)).and_then(|path| self.siblings_mut(&path)) {
            Some((sib, idx)) => sib.insert(idx + 1, layer),
            None => self.layers.push(layer),
        }
        id
    }

    /// Insert `layer` directly below the layer `below` (or at the bottom of the root if None).
    pub fn insert_below(&mut self, below: Option<LayerId>, layer: Layer) -> LayerId {
        let id = layer.id;
        match below.and_then(|b| self.path_of(b)).and_then(|path| self.siblings_mut(&path)) {
            Some((sib, idx)) => sib.insert(idx, layer),
            None => self.layers.insert(0, layer),
        }
        id
    }

    pub fn remove(&mut self, id: LayerId) -> Option<Layer> {
        let path = self.path_of(id)?;
        let (sib, idx) = self.siblings_mut(&path)?;
        Some(sib.remove(idx))
    }

    /// Move a layer up (+1) or down (-1) among its siblings. Returns false at the ends.
    pub fn shift(&mut self, id: LayerId, delta: i32) -> bool {
        let Some(path) = self.path_of(id) else { return false };
        let Some((sib, idx)) = self.siblings_mut(&path) else { return false };
        let to = idx as i64 + delta as i64;
        if to < 0 || to >= sib.len() as i64 {
            return false;
        }
        let l = sib.remove(idx);
        sib.insert(to as usize, l);
        true
    }

    /// Next unused "Layer N" name.
    pub fn next_layer_name(&self, base: &str) -> String {
        let names: std::collections::HashSet<&str> = self.walk().into_iter().map(|(_, _, l)| l.name.as_str()).collect();
        // At most `names.len() + 1` candidates are needed, so the search always succeeds.
        (1..=names.len() + 1).map(|n| format!("{base} {n}")).find(|n| !names.contains(n.as_str())).unwrap_or_else(|| base.to_string())
    }

    /// Name for a copy of a layer named `name`, unique in the document: "Layer 1 copy", then
    /// "Layer 1 copy 2", "Layer 1 copy 3"… A name that already ends in "copy" (or "copy N") is
    /// numbered from its root rather than growing another "copy".
    pub fn copy_name(&self, name: &str) -> String {
        let root = match name.rsplit_once(" copy") {
            Some((root, rest)) if rest.is_empty() || rest.strip_prefix(' ').is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())) => root,
            _ => name,
        };
        let names: std::collections::HashSet<&str> = self.walk().into_iter().map(|(_, _, l)| l.name.as_str()).collect();
        let first = format!("{root} copy");
        if !names.contains(first.as_str()) {
            return first;
        }
        // At most `names.len() + 1` candidates are needed, so the search always succeeds.
        (2..=names.len() + 2).map(|n| format!("{root} copy {n}")).find(|n| !names.contains(n.as_str())).unwrap_or(first)
    }

    /// Top-most layer id, useful as the default active layer.
    pub fn top_layer(&self) -> Option<LayerId> {
        self.layers.last().map(|l| l.id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc() -> Document {
        Document::with_background("t", Size::new(100, 50), ColorMode::Rgb, SampleType::U8, Color::WHITE)
    }

    #[test]
    fn any_lock_counts_every_lock() {
        assert!(!Locks::default().any());
        let each = [
            Locks { transparency: true, ..Locks::default() },
            Locks { pixels: true, ..Locks::default() },
            Locks { position: true, ..Locks::default() },
            Locks { artboard: true, ..Locks::default() },
            Locks { all: true, ..Locks::default() },
        ];
        assert!(each.iter().all(|k| k.any()));
    }

    #[test]
    fn unused_link_group_never_wraps() {
        let mut d = doc();
        assert_eq!(d.unused_link_group(), Some(1));
        let with = |d: &mut Document, groups: &[u64]| {
            d.layers.truncate(1);
            for g in groups {
                let mut l = Layer::raster("l", PixelFormat::RGBA8);
                l.link_group = Some(*g);
                d.layers.push(l);
            }
        };
        with(&mut d, &[3, 7]);
        assert_eq!(d.unused_link_group(), Some(8));
        with(&mut d, &[u64::MAX - 1]);
        assert_eq!(d.unused_link_group(), Some(u64::MAX));
        // A stored maximal id would wrap to 0 (#961): take the smallest free id instead.
        with(&mut d, &[u64::MAX]);
        assert_eq!(d.unused_link_group(), Some(1));
        with(&mut d, &[0, 1, 2, 4, u64::MAX]);
        assert_eq!(d.unused_link_group(), Some(3));
    }

    #[test]
    fn new_document_has_locked_background() {
        let d = doc();
        assert_eq!(d.layers.len(), 1);
        let bg = &d.layers[0];
        assert_eq!(bg.name, "Background");
        assert!(bg.locks.transparency);
        let s = bg.surface().unwrap();
        assert_eq!(s.pixel(0, 0), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.pixel(99, 49), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(s.pixel(100, 0), vec![0.0; 4]);
    }

    /// The Background is the locked bottom pixel layer, whatever it is called.
    #[test]
    fn background_is_structural_not_named() {
        let mut d = doc();
        let id = d.layers[0].id;
        assert!(d.is_background(id));
        for name in ["Hintergrund", "背景", "Arrière-plan"] {
            d.layers[0].name = name.into();
            assert_eq!(d.background().map(|l| l.id), Some(id), "{name}");
        }
        // A layer called "Background" without the locks, or above the bottom, is not it.
        let mut top = Layer::raster("Background", d.pixel_format());
        top.locks.transparency = true;
        top.locks.position = true;
        let top_id = top.id;
        d.layers.push(top);
        assert!(!d.is_background(top_id));
        d.layers[0].locks.position = false;
        assert!(d.background().is_none());
        d.layers[0].locks.position = true;
        d.layers[0].clipped = true;
        assert!(d.background_mut().is_none());
    }

    #[test]
    fn max_group_depth_counts_nesting_without_recursing() {
        let mut d = doc();
        let bg = d.layers[0].id;
        let mut chain = Layer::raster("L", d.pixel_format());
        for _ in 0..3 {
            chain = Layer::group("G", vec![chain]);
        }
        d.insert_above(Some(bg), chain);
        assert_eq!(d.max_group_depth(), 3);
        // An unfilled group does not add a level (nothing lives inside it).
        d.insert_above(Some(bg), Layer::group("empty", vec![]));
        assert_eq!(d.max_group_depth(), 3);
    }

    #[test]
    fn pixel_format_follows_mode_and_depth() {
        let d = Document::new("c", Size::new(1, 1), ColorMode::Cmyk, SampleType::U16);
        assert_eq!(d.pixel_format(), PixelFormat::new(ColorMode::Cmyk, SampleType::U16, true));
        let b = Document::new("b", Size::new(1, 1), ColorMode::Bitmap, SampleType::U8);
        assert_eq!(b.pixel_format().mode, ColorMode::Grayscale);
    }

    #[test]
    fn text_runs_with_maximal_lengths_normalize_to_text_length() {
        let t = TextLayer {
            text: "ab".into(),
            runs: vec![
                text::TextRun { len: 1, style: text::CharStyle::default() },
                text::TextRun { len: usize::MAX, style: text::CharStyle { size_pt: 24.0, ..Default::default() } },
            ],
            paragraphs: vec![
                text::ParagraphRun { len: 1, style: text::ParagraphStyle::default() },
                text::ParagraphRun { len: usize::MAX, style: text::ParagraphStyle { align: text::TextAlign::Center, ..Default::default() } },
            ],
            ..Default::default()
        };

        let chars = t.char_runs();
        assert_eq!(chars.iter().map(|r| r.len).sum::<usize>(), t.text.len());
        assert_eq!(chars.len(), 2);
        assert_eq!(chars[1].len, 1);
        assert_eq!(chars[1].style.size_pt, 24.0);

        let paragraphs = t.paragraph_runs();
        assert_eq!(paragraphs.iter().map(|r| r.len).sum::<usize>(), t.text.len());
        assert_eq!(paragraphs.len(), 2);
        assert_eq!(paragraphs[1].len, 1);
        assert_eq!(paragraphs[1].style.align, text::TextAlign::Center);
    }

    #[test]
    fn insert_remove_shift() {
        let mut d = doc();
        let bg = d.layers[0].id;
        let a = d.insert_above(Some(bg), Layer::raster("A", d.pixel_format()));
        let b = d.insert_above(Some(bg), Layer::raster("B", d.pixel_format()));
        // order bottom->top: bg, B, A
        let names: Vec<_> = d.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["Background", "B", "A"]);
        assert!(d.shift(b, 1));
        assert_eq!(d.layers[2].id, b);
        assert!(!d.shift(b, 1));
        assert_eq!(d.remove(a).unwrap().name, "A");
        assert!(d.layer(a).is_none());
    }

    #[test]
    fn insert_below_lands_under_the_reference() {
        let mut d = doc();
        let bg = d.layers[0].id;
        let a = d.insert_above(Some(bg), Layer::raster("A", d.pixel_format()));
        let b = d.insert_below(Some(a), Layer::raster("B", d.pixel_format()));
        let names: Vec<_> = d.layers.iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["Background", "B", "A"]);
        assert_eq!(d.layers[1].id, b);
    }

    #[test]
    fn groups_walk_and_paths() {
        let mut d = doc();
        let inner = Layer::raster("inner", d.pixel_format());
        let inner_id = inner.id;
        let g = Layer::group("G", vec![inner]);
        let gid = d.insert_above(None, g);
        let walk: Vec<_> = d.walk().into_iter().map(|(p, depth, l)| (p, depth, l.name.clone())).collect();
        assert_eq!(walk, vec![(vec![0], 0, "Background".into()), (vec![1], 0, "G".into()), (vec![1, 0], 1, "inner".into())]);
        assert_eq!(d.path_of(inner_id), Some(vec![1, 0]));
        assert_eq!(d.layer(gid).unwrap().blend, BlendMode::PassThrough);
        // insert above a nested layer stays inside the group
        let n = d.insert_above(Some(inner_id), Layer::raster("n", d.pixel_format()));
        assert_eq!(d.path_of(n), Some(vec![1, 1]));
        assert_eq!(d.layer_count(), 4);
    }

    /// `layer` and `path_of` search the tree directly (no `walk`): they must agree with it at
    /// every depth, and miss cleanly.
    #[test]
    fn layer_lookup_agrees_with_walk() {
        let mut d = doc();
        let format = d.pixel_format();
        let leaf = |n: &str| Layer::raster(n, format);
        let deep = Layer::group("A", vec![leaf("a0"), Layer::group("B", vec![leaf("b0"), Layer::group("C", vec![leaf("c0")]), leaf("b2")]), leaf("a2")]);
        d.insert_above(None, deep);
        d.insert_above(None, leaf("top"));
        let walk: Vec<(LayerPath, LayerId)> = d.walk().into_iter().map(|(p, _, l)| (p, l.id)).collect();
        assert_eq!(walk.len(), 10);
        for (path, id) in &walk {
            assert_eq!(d.path_of(*id).as_ref(), Some(path));
            assert_eq!(d.layer(*id).map(|l| l.id), Some(*id));
            assert_eq!(d.layer_mut(*id).map(|l| l.id), Some(*id));
        }
        let missing = Layer::raster("elsewhere", d.pixel_format()).id;
        assert!(d.path_of(missing).is_none() && d.layer(missing).is_none() && d.layer_mut(missing).is_none());
    }

    #[test]
    fn a_locked_group_locks_its_contents() {
        let mut d = doc();
        let inner = Layer::raster("inner", d.pixel_format());
        let inner_id = inner.id;
        let mut g = Layer::group("G", vec![Layer::group("H", vec![inner])]);
        g.locks.position = true;
        let gid = d.insert_above(None, g);
        d.layer_mut(inner_id).unwrap().locks.pixels = true;
        let k = d.effective_locks(inner_id);
        assert!(k.position && k.pixels && !k.all);
        assert!(!d.effective_locks(gid).pixels, "a child's lock doesn't lock its group");
        assert_eq!(d.effective_locks(LayerId(u64::MAX)), Locks::default());
    }

    #[test]
    fn duplicate_assigns_fresh_ids_recursively() {
        let g = Layer::group("G", vec![Layer::raster("x", PixelFormat::RGBA8)]);
        let dup = g.duplicate();
        assert_ne!(g.id, dup.id);
        assert_ne!(g.children().unwrap()[0].id, dup.children().unwrap()[0].id);
        assert_eq!(dup.children().unwrap()[0].name, "x");
    }

    #[test]
    fn duplicate_clears_psd_id_but_keeps_blocks() {
        let mut l = Layer::raster("x", PixelFormat::RGBA8);
        l.psd_id = Some(7);
        l.psd_blocks.push((*b"vmsk", Arc::new(vec![1])));
        let d = l.duplicate();
        assert_eq!(d.psd_id, None);
        assert_eq!(d.psd_blocks, l.psd_blocks);
    }

    #[test]
    fn layer_names_increment() {
        let mut d = doc();
        assert_eq!(d.next_layer_name("Layer"), "Layer 1");
        d.insert_above(None, Layer::raster("Layer 1", d.pixel_format()));
        assert_eq!(d.next_layer_name("Layer"), "Layer 2");
    }

    #[test]
    fn mask_density() {
        let mut m = LayerMask::hide_all();
        assert_eq!(m.value(5, 5), 0.0);
        m.density = 0.5;
        assert!((m.value(5, 5) - 0.5).abs() < 1e-6);
        m.enabled = false;
        assert_eq!(m.value(5, 5), 1.0);
    }

    #[test]
    fn snapshot_clone_is_independent() {
        let mut d = doc();
        let snap = d.clone();
        let bg = d.layers[0].id;
        d.layer_mut(bg).unwrap().surface_mut().unwrap().write_pixel(0, 0, &[0.0, 0.0, 0.0, 1.0]);
        assert_eq!(snap.layers[0].surface().unwrap().pixel(0, 0), vec![1.0; 4]);
    }

    #[test]
    fn layer_content_article_and_kind_name() {
        assert_eq!(LayerContent::Adjustment(Adjustment::Invert).article(), "an");
        assert_eq!(LayerContent::Adjustment(Adjustment::Invert).kind_name(), "Adjustment");
        assert_eq!(LayerContent::Raster(Surface::new(PixelFormat::RGBA8)).article(), "a");
        assert_eq!(LayerContent::Group(Group { children: Vec::new(), expanded: true, artboard: None }).article(), "a");
    }
}

#[cfg(test)]
mod id_tests {
    #[test]
    fn ensure_ids_above_advances_counter() {
        let far = super::LayerId::fresh().0 + 1000;
        super::ensure_ids_above(far);
        assert!(super::LayerId::fresh().0 > far);
        super::ensure_ids_above(3); // never moves backwards
        assert!(super::DocId::fresh().0 > far);
    }
}
