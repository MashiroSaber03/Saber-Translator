//! Channels: alpha and spot channels, Select › Save / Load Selection, Quick Mask, channel
//! targeting and visibility (the Channels panel), Image › Apply Image / Calculations, and the
//! Channels panel menu (Split / Merge Channels, Merge Spot Channel).
//!
//! Alpha channels live in [`Document::channels`] (white = selected). Quick Mask is a temporary
//! channel in [`Document::quick_mask`], so entering and leaving it are history steps like in
//! Photoshop. Which channel is targeted and which are visible is view state on
//! [`DocState::channel_view`] (not history), like the active layer.
//!
//! Pixel commands (`paint.*`, `filter.*`, `image.adjustments.*`, `edit.fill`) take
//! `"target": {"channel": i}` or `"target": "quickMask"`; [`Session::execute`] fills that in from the
//! targeted channel when the caller didn't pass a target. Targeting a single colour channel keeps
//! those commands from touching the other colour channels of the layer ([`restrict_to_color`]).

use photocraft_algo::selection::{self as sel, SelectionMode};
use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, blend::blend_channel};
use photocraft_doc::{AlphaChannel, ColorIndicates, Document, Layer, LayerContent, LayerId};
use photocraft_geom::{Rect, TileCoord};
use photocraft_raster::{Surface, from_rgba, to_rgba};
use serde::Serialize;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{DocState, EngineError, Result, Session};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------------------------------
// View state

/// What pixel commands edit (Channels panel click / ⌘2…⌘9).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase", tag = "kind", content = "index")]
pub enum ChannelTarget {
    /// The layers (all colour channels). In Quick Mask mode this edits the Quick Mask.
    #[default]
    Composite,
    /// One colour channel of the active layer (index into the pixel format's colour channels).
    Color(usize),
    /// An alpha or spot channel (index into [`Document::channels`]).
    Alpha(usize),
    /// The active layer's layer mask.
    LayerMask,
    /// The active layer's vector mask.
    VectorMask,
}

/// Per-document Channels panel state: the target and the eye toggles. Not part of history.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChannelView {
    pub target: ChannelTarget,
    /// Hidden colour channels (missing entries are visible).
    pub color_hidden: Vec<bool>,
    /// Visible alpha channels (missing entries are hidden).
    pub alpha_visible: Vec<bool>,
    pub quick_mask_hidden: bool,
    /// ⌥-click view of the active layer's mask (#196); see [`crate::mask_view_cmds`].
    pub layer_mask: Option<crate::mask_view_cmds::LayerMaskView>,
}

impl ChannelView {
    pub fn color_visible(&self, k: usize) -> bool {
        !self.color_hidden.get(k).copied().unwrap_or(false)
    }
    pub fn alpha_shown(&self, i: usize) -> bool {
        self.alpha_visible.get(i).copied().unwrap_or(false)
    }
    pub fn set_color_visible(&mut self, k: usize, v: bool) {
        if self.color_hidden.len() <= k {
            self.color_hidden.resize(k + 1, false);
        }
        self.color_hidden[k] = !v;
    }
    pub fn set_alpha_visible(&mut self, i: usize, v: bool) {
        if self.alpha_visible.len() <= i {
            self.alpha_visible.resize(i + 1, false);
        }
        self.alpha_visible[i] = v;
    }
    /// Number of visible colour channels out of `colors`.
    pub fn visible_colors(&self, colors: usize) -> usize {
        (0..colors).filter(|k| self.color_visible(*k)).count()
    }
    /// True when the canvas shows just the normal composite (nothing extra to draw).
    pub fn is_plain(&self, doc: &Document) -> bool {
        let colors = color_count(doc);
        self.visible_colors(colors) == colors
            && !(0..doc.channels.len()).any(|i| self.alpha_shown(i))
            && (doc.quick_mask.is_none() || self.quick_mask_hidden)
            && self.shown_layer_mask(doc).is_none()
    }
    /// The layer mask shown on the canvas (#196) and how, if its layer still has a mask.
    pub fn shown_layer_mask<'a>(&self, doc: &'a Document) -> Option<(&'a photocraft_doc::LayerMask, crate::mask_view_cmds::MaskViewMode)> {
        let v = self.layer_mask?;
        Some((doc.layer(v.layer)?.mask.as_ref()?, v.mode))
    }
}

/// Session-wide Quick Mask Options (double-click the Quick Mask button in Photoshop).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct QuickMaskOptions {
    pub color: Color,
    pub opacity: f32,
    pub indicates: ColorIndicates,
}

impl Default for QuickMaskOptions {
    fn default() -> Self {
        Self { color: AlphaChannel::DEFAULT_COLOR, opacity: 0.5, indicates: ColorIndicates::MaskedAreas }
    }
}

/// Keep the view valid after the document changed (undo, deleted channels, mode changes).
pub(crate) fn fix_view(st: &mut DocState) {
    let n = st.doc.channels.len();
    let colors = color_count(&st.doc);
    let quick = st.doc.quick_mask.is_some();
    {
        let v = &mut st.channel_view;
        v.alpha_visible.truncate(n);
        v.color_hidden.truncate(colors);
        match v.target {
            ChannelTarget::Alpha(i) if i >= n => v.target = ChannelTarget::Composite,
            ChannelTarget::Color(k) if k >= colors || colors < 2 => v.target = ChannelTarget::Composite,
            ChannelTarget::LayerMask if st.active_layer.and_then(|id| st.doc.layer(id)).is_none_or(|l| l.mask.is_none()) => {
                v.target = ChannelTarget::Composite;
            }
            ChannelTarget::VectorMask if st.active_layer.and_then(|id| st.doc.layer(id)).is_none_or(|l| l.vector_mask.is_none()) => {
                v.target = ChannelTarget::Composite;
            }
            _ => {}
        }
    }
    crate::mask_view_cmds::fix(st);
    // An alpha/spot eye, Quick Mask, or the active layer's mask can be the only thing on the
    // canvas (so R, G and B can all be off). Otherwise the colour channels come back.
    let active = st.active_layer;
    let shown = {
        let doc = &st.doc;
        let v = &st.channel_view;
        (0..n).any(|i| v.alpha_shown(i)) || (quick && !v.quick_mask_hidden) || active.is_some_and(|id| doc.layer(id).is_some_and(|l| l.mask.is_some()))
    };
    if st.channel_view.visible_colors(colors) == 0 && !shown {
        st.channel_view.color_hidden.clear();
    }
}

/// Bump the revision for a view-only change without dirtying a clean document.
fn touch(st: &mut DocState) {
    let clean = st.saved_revision == st.revision;
    st.revision += 1;
    st.view_revision += 1;
    if clean {
        st.saved_revision = st.revision;
    }
    st.last_damage = None;
}

// ---------------------------------------------------------------------------------------------
// Channel names and formats

/// Colour channel names of a document mode (the rows under the composite).
pub fn color_names(mode: ColorMode) -> &'static [&'static str] {
    match mode {
        ColorMode::Cmyk => &["Cyan", "Magenta", "Yellow", "Black"],
        ColorMode::Lab => &["Lightness", "a", "b"],
        ColorMode::Grayscale => &["Gray"],
        _ => &["Red", "Green", "Blue"],
    }
}

/// Name of the composite row.
pub fn composite_name(mode: ColorMode) -> &'static str {
    match mode {
        ColorMode::Cmyk => "CMYK",
        ColorMode::Lab => "Lab",
        ColorMode::Grayscale => "Gray",
        _ => "RGB",
    }
}

fn edit_mode(doc: &Document) -> ColorMode {
    doc.pixel_format().mode
}

pub fn color_count(doc: &Document) -> usize {
    edit_mode(doc).color_channels()
}

/// Pixel format of alpha channels: grayscale at the document depth, no transparency.
pub fn channel_format(doc: &Document) -> PixelFormat {
    PixelFormat::new(ColorMode::Grayscale, doc.depth, false)
}

/// Next free "`base` N" name (Alpha 1, Spot Color 2, …).
pub fn next_alpha_name(doc: &Document, base: &str) -> String {
    // `channels.len() + 1` candidates always include a free one.
    (1..=doc.channels.len() + 1).map(|n| format!("{base} {n}")).find(|n| !doc.channels.iter().any(|c| &c.name == n)).unwrap_or_else(|| base.to_string())
}

// ---------------------------------------------------------------------------------------------
// Planes: one grayscale value per canvas pixel

fn read_plane(s: &Surface, r: Rect) -> Vec<f32> {
    let mut v = Vec::new();
    s.read_region_into(r, &mut v);
    let ch = s.channels();
    if ch > 1 { v.chunks_exact(ch).map(|p| p[0]).collect() } else { v }
}

fn plane_surface(vals: &[f32], r: Rect, fmt: PixelFormat) -> Surface {
    let mut s = Surface::new(fmt);
    let clamped: Vec<f32> = vals.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    s.write_region(r, &clamped);
    s.prune();
    s
}

/// Make every canvas pixel of `s` explicit so filters and adjustments that work on the content
/// bounds also see the untouched (default-valued) parts of a channel.
fn materialize(s: &mut Surface, r: Rect) {
    let v = s.read_region(r);
    s.write_region(r, &v);
}

fn luminosity(rgba: [f32; 4]) -> f32 {
    0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
}

// ---------------------------------------------------------------------------------------------
// Channel references

/// A channel named in params: `"composite"`, a colour name (`"red"`, `"cyan"`, `"lightness"`…),
/// an alpha index (number) or name, `"quickMask"`, `"selection"`, `"transparency"`, `"mask"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChanRef {
    Composite,
    Color(usize),
    Alpha(usize),
    QuickMask,
    Selection,
    Transparency,
    LayerMask,
    /// The layer's vector mask, rasterized (⌘-click its thumbnail, #196).
    VectorMask,
}

pub fn parse_ref(v: &Value, doc: &Document) -> Option<ChanRef> {
    let mode = edit_mode(doc);
    match v {
        Value::Number(n) => n.as_u64().map(|i| ChanRef::Alpha(i as usize)),
        Value::Object(o) => {
            if let Some(k) = o.get("color").and_then(Value::as_u64) {
                Some(ChanRef::Color(k as usize))
            } else {
                o.get("alpha").and_then(Value::as_u64).map(|i| ChanRef::Alpha(i as usize))
            }
        }
        Value::String(s) => {
            let l = s.to_ascii_lowercase();
            match l.as_str() {
                "composite" | "merged" => return Some(ChanRef::Composite),
                "quickmask" | "quick mask" => return Some(ChanRef::QuickMask),
                "selection" => return Some(ChanRef::Selection),
                "transparency" => return Some(ChanRef::Transparency),
                "mask" | "layermask" => return Some(ChanRef::LayerMask),
                "vectormask" | "vector mask" => return Some(ChanRef::VectorMask),
                _ => {}
            }
            if l == composite_name(mode).to_ascii_lowercase() {
                return Some(ChanRef::Composite);
            }
            if let Some(k) = color_names(mode).iter().position(|n| n.eq_ignore_ascii_case(&l)) {
                return Some(if color_names(mode).len() == 1 { ChanRef::Composite } else { ChanRef::Color(k) });
            }
            doc.channels.iter().position(|c| c.name == *s).map(ChanRef::Alpha)
        }
        _ => None,
    }
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn doc_index(s: &Session, p: &Value, key: &str) -> Result<usize> {
    match crate::commands::int(p, key).and_then(|v| u64::try_from(v).ok()) {
        Some(i) if (i as usize) < s.documents().len() => Ok(i as usize),
        Some(i) => Err(EngineError::Other(format!("no open document {i}"))),
        None => s.active_index().ok_or(EngineError::NoDocument),
    }
}

/// Alpha channel index from `p[key]` (number or name), checked against the document.
fn alpha_index(doc: &Document, p: &Value, key: &str, cmd: &str) -> Result<usize> {
    match p.get(key).and_then(|v| parse_ref(v, doc)) {
        Some(ChanRef::Alpha(i)) if i < doc.channels.len() => Ok(i),
        Some(ChanRef::Alpha(i)) => Err(bad(cmd, format!("no alpha channel {i} (document has {})", doc.channels.len()))),
        _ => Err(bad(cmd, format!("`{key}` must name an alpha channel (index or name)"))),
    }
}

// ---------------------------------------------------------------------------------------------
// Sources (Load Selection, Apply Image, Calculations)

/// Native pixel values of a layer or of the merged image over the canvas.
fn native_pixels(doc: &Document, layer: Option<LayerId>) -> Result<(Vec<f32>, usize)> {
    let r = doc.bounds();
    let fmt = doc.pixel_format();
    let n = fmt.channels();
    match layer {
        Some(id) => {
            let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            let s = l.surface().ok_or_else(|| EngineError::Other(format!("layer \"{}\" has no pixels", l.name)))?;
            if s.format() == fmt {
                return Ok((s.read_region(r), n));
            }
            let raw = s.read_region(r);
            let sf = s.format();
            Ok((raw.chunks_exact(sf.channels()).flat_map(|p| from_rgba(&fmt, to_rgba(&sf, p))).collect(), n))
        }
        None => {
            let buf = photocraft_compose::render(doc, r);
            Ok((buf.px.iter().flat_map(|p| from_rgba(&fmt, *p)).collect(), n))
        }
    }
}

/// `"layer"` param of a source: a layer id, a layer name (the topmost of that name, as a loaded
/// Photoshop action refers to it), or `"merged"` (default). An unknown name is an error.
fn source_layer(doc: &Document, p: &Value, cmd: &str) -> Result<Option<LayerId>> {
    match p.get("layer") {
        Some(Value::Number(n)) => Ok(n.as_u64().map(LayerId)),
        Some(Value::String(name)) if !name.is_empty() && name != "merged" => doc
            .walk()
            .into_iter()
            .rev()
            .find(|(_, _, l)| l.name == *name)
            .map(|(_, _, l)| Some(l.id))
            .ok_or_else(|| bad(cmd, format!("no source layer named \"{name}\""))),
        _ => Ok(None),
    }
}

/// Planes of channel `r` from document `doc` (`layer` = None: merged). A composite gives one
/// plane per colour channel; everything else gives one plane.
fn ref_planes(doc: &Document, layer: Option<LayerId>, active: Option<LayerId>, r: ChanRef) -> Result<Vec<Vec<f32>>> {
    let area = doc.bounds();
    let colors = color_count(doc);
    Ok(match r {
        ChanRef::Composite | ChanRef::Color(_) => {
            let (px, n) = native_pixels(doc, layer)?;
            let planes: Vec<Vec<f32>> = (0..colors).map(|c| px.chunks_exact(n).map(|p| p[c]).collect()).collect();
            match r {
                ChanRef::Color(k) => vec![planes.get(k).cloned().ok_or_else(|| EngineError::Other(format!("no colour channel {k}")))?],
                _ => planes,
            }
        }
        ChanRef::Alpha(i) => vec![read_plane(&doc.channels.get(i).ok_or_else(|| EngineError::Other(format!("no alpha channel {i}")))?.surface, area)],
        ChanRef::QuickMask => vec![read_plane(&doc.quick_mask.as_ref().ok_or_else(|| EngineError::Other("not in Quick Mask mode".into()))?.surface, area)],
        ChanRef::Selection => vec![sel::mask_from_surface(doc.selection.as_ref(), area)],
        ChanRef::Transparency => {
            let id = layer.or(active).ok_or_else(|| EngineError::Other("transparency needs a layer".into()))?;
            let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            match l.surface() {
                Some(s) => {
                    let mut px = vec![[0.0f32; 4]; area.width() as usize * area.height() as usize];
                    s.read_rgba_into(area, &mut px);
                    vec![px.iter().map(|p| p[3]).collect()]
                }
                // A shape can have no pixel cache yet (for example after loading a document).
                // Use its existing renderer rather than treating it as a canvas-wide fill.
                None => match &l.content {
                    LayerContent::Shape(shape) => vec![
                        photocraft_vector::CompiledShape::new(shape, photocraft_vector::DEFAULT_TOLERANCE, area)
                            .render_rgba(area)
                            .into_iter()
                            .map(|p| p[3])
                            .collect(),
                    ],
                    // Adjustment and fill layers cover the whole canvas.
                    _ => vec![vec![1.0; area.width() as usize * area.height() as usize]],
                },
            }
        }
        ChanRef::LayerMask => {
            let id = layer.or(active).ok_or_else(|| EngineError::Other("layer mask needs a layer".into()))?;
            let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            let m = l.mask.as_ref().ok_or_else(|| EngineError::Other(format!("layer \"{}\" has no mask", l.name)))?;
            vec![read_plane(&m.surface, area)]
        }
        ChanRef::VectorMask => {
            let id = layer.or(active).ok_or_else(|| EngineError::Other("vector mask needs a layer".into()))?;
            let l = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
            let m = l.vector_mask.as_ref().ok_or_else(|| EngineError::Other(format!("layer \"{}\" has no vector mask", l.name)))?;
            // The path's shape, whether or not the mask is enabled (as Photoshop loads it).
            let shape = photocraft_doc::VectorMask { enabled: true, density: 1.0, feather: 0.0, ..m.clone() };
            vec![photocraft_vector::vector_mask_values(&shape, area)]
        }
    })
}

/// One grayscale plane (composites reduce to luminosity, as in Calculations).
fn gray_plane(doc: &Document, layer: Option<LayerId>, active: Option<LayerId>, r: ChanRef) -> Result<Vec<f32>> {
    let mut planes = ref_planes(doc, layer, active, r)?;
    if planes.len() == 1
        && let Some(only) = planes.pop()
    {
        return Ok(only);
    }
    let fmt = doc.pixel_format();
    let n = planes.first().map_or(0, Vec::len);
    let mut px = vec![0.0f32; fmt.channels()];
    Ok((0..n)
        .map(|i| {
            for (c, p) in planes.iter().enumerate() {
                px[c] = p[i];
            }
            if fmt.alpha {
                px[planes.len()] = 1.0;
            }
            luminosity(to_rgba(&fmt, &px))
        })
        .collect())
}

/// A source spec `{document?, layer?, channel, invert?}`, reduced to planes over the canvas of
/// the active document (sizes must match, as in Photoshop).
fn source(s: &Session, p: &Value, cmd: &str, gray: bool) -> Result<Vec<Vec<f32>>> {
    let di = doc_index(s, p, "document")?;
    let st = &s.documents()[di];
    let target = s.active().ok_or(EngineError::NoDocument)?;
    if st.doc.size != target.doc.size {
        return Err(bad(cmd, format!("source document \"{}\" is {}×{}; it must match the target size", st.doc.name, st.doc.size.width, st.doc.size.height)));
    }
    let r = p.get("channel").map_or(Some(ChanRef::Composite), |v| parse_ref(v, &st.doc)).ok_or_else(|| bad(cmd, "unknown `channel`"))?;
    let layer = source_layer(&st.doc, p, cmd)?;
    let mut planes = if gray { vec![gray_plane(&st.doc, layer, st.active_layer, r)?] } else { ref_planes(&st.doc, layer, st.active_layer, r)? };
    // Guard every caller (`.remove(0)` / `src[0]`) against an empty source.
    if planes.is_empty() {
        return Err(bad(cmd, "the source has no channels"));
    }
    if p.get("invert").and_then(Value::as_bool).unwrap_or(false) {
        for pl in &mut planes {
            for v in pl.iter_mut() {
                *v = 1.0 - *v;
            }
        }
    }
    Ok(planes)
}

// ---------------------------------------------------------------------------------------------
// Blending (Apply Image / Calculations)

/// Apply Image / Calculations blending: the separable layer modes plus Add and Subtract with
/// scale and offset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Blending {
    Mode(BlendMode),
    Add { scale: f32, offset: f32 },
    Subtract { scale: f32, offset: f32 },
}

impl Blending {
    pub fn parse(p: &Value) -> std::result::Result<Self, String> {
        let name = p.get("blending").and_then(Value::as_str).unwrap_or("multiply");
        let scale = p.get("scale").and_then(Value::as_f64).unwrap_or(1.0).clamp(1.0, 2.0) as f32;
        let offset = p.get("offset").and_then(Value::as_f64).unwrap_or(0.0).clamp(-255.0, 255.0) as f32 / 255.0;
        match name.to_ascii_lowercase().as_str() {
            "add" => Ok(Blending::Add { scale, offset }),
            "subtract" => Ok(Blending::Subtract { scale, offset }),
            other => match crate::commands::blend_from_str(other) {
                Some(m) if m.is_separable() && !matches!(m, BlendMode::Dissolve | BlendMode::PassThrough) => Ok(Blending::Mode(m)),
                _ => Err(format!("unsupported blending `{other}`")),
            },
        }
    }

    /// Blend `top` (the source) onto `base` (the target).
    pub fn apply(self, base: f32, top: f32) -> f32 {
        match self {
            Blending::Mode(m) => blend_channel(m, base, top),
            Blending::Add { scale, offset } => (base + top) / scale + offset,
            Blending::Subtract { scale, offset } => (base - top) / scale + offset,
        }
        .clamp(0.0, 1.0)
    }
}

/// A source spec from `p[key]`, or from flat `<prefix>Document` / `<prefix>Layer` /
/// `<prefix>Channel` / `<prefix>Invert` keys (what the generic dialog form sends).
fn source_spec(p: &Value, key: &str, prefix: &str) -> Option<Value> {
    if let Some(v) = p.get(key) {
        return Some(v.clone());
    }
    let mut m = serde_json::Map::new();
    for (k, field) in [("document", "Document"), ("layer", "Layer"), ("channel", "Channel"), ("invert", "Invert")] {
        match p.get(format!("{prefix}{field}")) {
            None | Some(Value::Null) => {}
            Some(Value::String(s)) if s.is_empty() || (k == "layer" && s == "merged") => {}
            Some(v) => {
                m.insert(k.into(), v.clone());
            }
        }
    }
    (!m.is_empty()).then_some(Value::Object(m))
}

fn mask_plane(s: &Session, p: &Value, cmd: &str) -> Result<Option<Vec<f32>>> {
    match source_spec(p, "mask", "mask") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => Ok(None),
        Some(m) if m.get("channel").is_none_or(|c| c == "none") => Ok(None),
        Some(m) => Ok(Some(source(s, &m, cmd, true)?.remove(0))),
    }
}

// ---------------------------------------------------------------------------------------------
// Targets for pixel commands

/// Where a pixel command writes, from its `"target"` param.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Target {
    Pixels,
    Mask,
    Alpha(usize),
    QuickMask,
}

pub(crate) fn target_of(p: &Value) -> Target {
    match p.get("target") {
        Some(Value::String(s)) => match s.as_str() {
            "mask" => Target::Mask,
            "quickMask" => Target::QuickMask,
            _ => Target::Pixels,
        },
        Some(Value::Object(o)) => o.get("channel").and_then(Value::as_u64).map_or(Target::Pixels, |i| Target::Alpha(i as usize)),
        _ => Target::Pixels,
    }
}

/// The target is an alpha channel or the Quick Mask (no layer involved).
pub(crate) fn is_channel_target(p: &Value) -> bool {
    matches!(target_of(p), Target::Alpha(_) | Target::QuickMask)
}

/// The surface a pixel command writes to and whether the layer's transparency is locked.
pub(crate) fn target_surface<'a>(doc: &'a mut Document, layer: Option<LayerId>, p: &Value) -> Result<(&'a mut Surface, bool)> {
    match target_of(p) {
        Target::Alpha(i) => {
            let n = doc.channels.len();
            doc.channels.get_mut(i).map(|c| (&mut c.surface, false)).ok_or_else(|| EngineError::Other(format!("no alpha channel {i} (document has {n})")))
        }
        Target::QuickMask => doc.quick_mask.as_mut().map(|c| (&mut c.surface, false)).ok_or_else(|| EngineError::Other("not in Quick Mask mode".into())),
        t => {
            let id = layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
            let lock = doc.effective_locks(id).transparency && t == Target::Pixels;
            Ok((crate::commands::paint_surface(doc, id, p)?, lock))
        }
    }
}

pub(crate) fn read_target_surface<'a>(doc: &'a Document, layer: Option<LayerId>, p: &Value) -> Result<&'a Surface> {
    match target_of(p) {
        Target::Alpha(i) => {
            let n = doc.channels.len();
            doc.channels.get(i).map(|c| &c.surface).ok_or_else(|| EngineError::Other(format!("no alpha channel {i} (document has {n})")))
        }
        Target::QuickMask => doc.quick_mask.as_ref().map(|c| &c.surface).ok_or_else(|| EngineError::Other("not in Quick Mask mode".into())),
        Target::Pixels => {
            let id = layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
            doc.layer(id).and_then(Layer::surface).ok_or(EngineError::NoLayer(id))
        }
        Target::Mask => {
            let id = layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
            doc.layer(id).and_then(|l| l.mask.as_ref()).map(|m| &m.surface).ok_or(EngineError::NoLayer(id))
        }
    }
}

/// The grayscale surface filters and adjustments edit instead of the layer's pixels (made explicit
/// over the canvas), if the params target one: an alpha channel, the Quick Mask, or the mask of
/// `layer`.
pub(crate) fn channel_surface_for_filter<'a>(doc: &'a mut Document, layer: Option<LayerId>, p: &Value) -> Result<Option<&'a mut Surface>> {
    if target_of(p) == Target::Pixels {
        return Ok(None);
    }
    let b = doc.bounds();
    let (surf, _) = target_surface(doc, layer, p)?;
    materialize(surf, b);
    Ok(Some(surf))
}

/// Commands that edit a targeted layer mask in place of the layer: the Image › Adjustments table,
/// the filters and the fills (Edit › Fill and its keys). A mask target enables them on any layer
/// with a mask, so ⌘I inverts an adjustment layer's mask (#780) and Fill fills it (#2456) as in
/// Photoshop.
fn edits_mask(id: &str) -> bool {
    id.strip_prefix("image.adjustments.").is_some_and(|kind| crate::adjust_params::default_for(kind, ColorMode::Rgb).is_ok())
        || crate::filters::params_for(id, &Value::Null).is_some()
        || id == "edit.fill"
        // Copying a targeted mask (#2928).
        || id == "edit.copy"
        || crate::fill_key_cmds::IDS.contains(&id)
}

/// The precondition of `id` when `p` targets the active layer's mask (`"target":"mask"`): the
/// layer needs a mask. None when another precondition applies.
pub(crate) fn mask_target_enabled(s: &Session, id: &str, p: &Value) -> Option<std::result::Result<(), String>> {
    if target_of(p) != Target::Mask || !edits_mask(id) {
        return None;
    }
    Some(crate::active_layer_of(s).and_then(|l| if l.mask.is_some() { Ok(()) } else { Err("the active layer has no layer mask".into()) }))
}

/// Whether command `id` edits the targeted channel or mask when its params name no `"target"`
/// (filters, adjustments, paint and fill commands); the shell adds its mask target to these.
pub fn follows_target(id: &str) -> bool {
    // Moving the selected pixels (⌘-drag, Move tool) moves the targeted mask's, not the layer's.
    // Moving a layer with its unlinked mask targeted moves the mask alone (`layer.translate`).
    routed(id) || id == "select.float" || id == "layer.translate"
}

/// Commands whose target follows the Channels panel.
fn routed(id: &str) -> bool {
    id.starts_with("filter.")
        || id == "plugin.run"
        || id.starts_with("image.adjustments.")
        || matches!(id, "paint.stroke" | "paint.pencil" | "paint.bucket" | "paint.gradient" | "paint.mixerBrush" | "edit.fill" | "image.applyImage")
        || crate::fill_key_cmds::IDS.contains(&id)
        // Pasting into a targeted mask or channel (#1035); Copy takes a targeted mask (#2928).
        || matches!(id, "edit.copy" | "edit.paste" | "edit.pasteSpecial.pasteInPlace" | "edit.pasteSpecial.pasteInto" | "edit.pasteSpecial.pasteOutside")
        // Edit › Transform › Rotate/Flip turn a targeted mask or channel (#2835).
        || matches!(
            id,
            "edit.transform.rotate180"
                | "edit.transform.rotate90Cw"
                | "edit.transform.rotate90Ccw"
                | "edit.transform.flipHorizontal"
                | "edit.transform.flipVertical"
        )
        || matches!(
            id,
            "paint.cloneStamp"
                | "paint.patternStamp"
                | "paint.healingBrush"
                | "paint.spotHealing"
                | "paint.dodge"
                | "paint.burn"
                | "paint.sponge"
                | "paint.blur"
                | "paint.sharpen"
                | "paint.smudge"
                | "paint.historyBrush"
                | "paint.redEye"
                | "paint.remove"
        )
}

/// Fill in `"target"` from the targeted channel (or Quick Mask mode) when the caller gave none.
pub(crate) fn inject_target(s: &Session, id: &str, params: Value) -> Value {
    if !routed(id) || params.get("target").is_some() {
        return params;
    }
    let Some(st) = s.active() else { return params };
    let t = match st.channel_view.target {
        ChannelTarget::Alpha(i) if i < st.doc.channels.len() => json!({ "channel": i }),
        ChannelTarget::Composite if st.doc.quick_mask.is_some() => json!("quickMask"),
        ChannelTarget::LayerMask => json!("mask"),
        // Viewing the active layer's mask (⌥-click): pixel commands edit the mask.
        ChannelTarget::Composite if crate::mask_view_cmds::current(st).is_some() => json!("mask"),
        _ => return params,
    };
    match params {
        Value::Object(mut m) => {
            m.insert("target".into(), t);
            Value::Object(m)
        }
        _ => json!({ "target": t }),
    }
}

/// The colour channel pixel commands are limited to, if a single colour channel is targeted.
pub(crate) fn color_restriction(s: &Session, id: &str, params: &Value) -> Option<usize> {
    if !routed(id) || is_channel_target(params) || crate::commands::is_mask_target(params) {
        return None;
    }
    match s.active()?.channel_view.target {
        ChannelTarget::Color(k) => Some(k),
        _ => None,
    }
}

/// After a pixel edit with colour channel `k` targeted, put the other channels (and the layer's
/// transparency) of layer `id` back as they were. Only changed tiles are touched.
pub(crate) fn restrict_to_color(before: &Document, after: &mut Document, id: LayerId, k: usize) {
    let Some(old) = before.layer(id).and_then(Layer::surface) else { return };
    let Some(new) = after.layer_mut(id).and_then(Layer::surface_mut) else { return };
    if old.format() != new.format() {
        return;
    }
    let mut coords: Vec<TileCoord> = new.tiles().filter(|(c, t)| old.tile(**c).is_none_or(|o| !std::sync::Arc::ptr_eq(o, t))).map(|(c, _)| *c).collect();
    coords.extend(old.tiles().filter(|(c, _)| new.tile(**c).is_none()).map(|(c, _)| *c));
    let n = new.channels();
    for c in coords {
        let r = c.rect();
        let o = old.read_region(r);
        let mut v = new.read_region(r);
        for (pn, po) in v.chunks_exact_mut(n).zip(o.chunks_exact(n)) {
            for ch in 0..n {
                if ch != k {
                    pn[ch] = po[ch];
                }
            }
        }
        new.write_region(r, &v);
    }
    new.prune();
}

/// The colour channel Clear is limited to: the Channels panel targets a single one.
pub(crate) fn targeted_color(s: &Session) -> Option<usize> {
    match s.active()?.channel_view.target {
        ChannelTarget::Color(k) => Some(k),
        _ => None,
    }
}

/// Clear (Delete) with colour channel `k` of layer `id` targeted (#2720): fill the selected area
/// (or the canvas) of that channel only with the background colour's value for it, as Photoshop
/// does. A channel has no transparency of its own, so the other channels and the layer's alpha
/// stay; partly selected pixels blend.
pub(crate) fn clear_color_channel(doc: &mut Document, id: LayerId, k: usize, background: [f32; 4]) -> Result<()> {
    let canvas = doc.bounds();
    let sel = doc.selection.clone();
    let area = sel.as_ref().map_or(canvas, |m| m.content_bounds().intersect(&canvas));
    let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
    let fmt = surf.format();
    let n = fmt.channels();
    // The background colour in the layer's colour model, so `k` picks the right component.
    let Some(v) = from_rgba(&fmt, [background[0], background[1], background[2], 1.0]).get(k).copied().filter(|_| k < n) else {
        return Err(EngineError::Other(format!("no colour channel {k} in this layer")));
    };
    if area.is_empty() {
        return Ok(());
    }
    let cover =
        sel.as_ref().map(|m| crate::allocation::read_region(m, area, "clearing a channel selection").map(|v| (v, m.format().channels().max(1)))).transpose()?;
    let mut px = crate::allocation::read_region(surf, area, "clearing a channel")?;
    for (i, p) in px.chunks_exact_mut(n).enumerate() {
        let w = cover.as_ref().map_or(1.0, |(m, mk)| m.get(i * mk).copied().unwrap_or(0.0));
        if let Some(c) = p.get_mut(k) {
            *c += (v - *c) * w.clamp(0.0, 1.0);
        }
    }
    surf.try_write_region(area, &px)?;
    surf.prune();
    Ok(())
}

/// The alpha channel or Quick Mask the Channels panel targets, as `"target"` params: Clear
/// (Delete) fills it instead of the layer (#2734).
pub(crate) fn targeted_channel(s: &Session) -> Option<Value> {
    let st = s.active()?;
    let t = match st.channel_view.target {
        ChannelTarget::Alpha(i) if i < st.doc.channels.len() => json!({ "channel": i }),
        ChannelTarget::Composite if st.doc.quick_mask.is_some() => json!("quickMask"),
        _ => return None,
    };
    Some(json!({ "target": t }))
}

/// Clear (Delete) with an alpha channel or the Quick Mask targeted (`target` from
/// [`targeted_channel`]): fill the selected area (or the canvas) of it with the background colour,
/// as gray, as Photoshop does. Partly selected pixels blend.
pub(crate) fn clear_channel(doc: &mut Document, target: &Value, background: [f32; 4]) -> Result<()> {
    let canvas = doc.bounds();
    let sel = doc.selection.clone();
    let area = sel.as_ref().map_or(canvas, |m| m.content_bounds().intersect(&canvas));
    if area.is_empty() {
        return Ok(());
    }
    let (surf, _) = target_surface(doc, None, target)?;
    crate::fill_cmds::fill_color(surf, area, [background[0], background[1], background[2], 1.0], sel.as_ref())
}

/// The active document targets an alpha channel or is in Quick Mask mode, so pixel commands
/// work without a pixel layer.
pub(crate) fn edits_channel(s: &Session) -> bool {
    s.active().is_some_and(|st| {
        st.doc.quick_mask.is_some() && st.channel_view.target == ChannelTarget::Composite
            || matches!(st.channel_view.target, ChannelTarget::Alpha(i) if i < st.doc.channels.len())
    })
}

// ---------------------------------------------------------------------------------------------
// Enabled predicates

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}
fn has_channels(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if s.active().is_some_and(|d| !d.doc.channels.is_empty()) { Ok(()) } else { Err("the document has no alpha channels".into()) }
}
fn has_selection(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if s.active().is_some_and(|d| d.doc.selection.is_some()) { Ok(()) } else { Err("no selection".into()) }
}
fn has_spot(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if s.active().is_some_and(|d| d.doc.channels.iter().any(|c| c.spot.is_some())) { Ok(()) } else { Err("the document has no spot channels".into()) }
}
fn can_split(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if color_count(&d.doc) + d.doc.channels.len() < 2 { Err("only one channel to split".into()) } else { Ok(()) }
}
fn can_merge(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    if mergeable(s).len() >= 2 { Ok(()) } else { Err("Merge Channels needs at least two open grayscale documents of the same size".into()) }
}

// ---------------------------------------------------------------------------------------------
// Save / Load Selection

fn operation<'a>(p: &'a Value, key: &str) -> &'a str {
    p.get(key).and_then(Value::as_str).unwrap_or("new")
}

fn save_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "select.saveSelection";
    let op = p.get("operation").and_then(Value::as_str).unwrap_or("new").to_string();
    if !matches!(op.as_str(), "new" | "replace" | "add" | "subtract" | "intersect") {
        return Err(bad(cmd, format!("unknown operation `{op}`")));
    }
    let di = doc_index(s, p, "document")?;
    let src = s.active().ok_or(EngineError::NoDocument)?;
    let area = src.doc.bounds();
    let selection = sel::mask_from_surface(src.doc.selection.as_ref(), area);
    if s.documents()[di].doc.size != src.doc.size {
        return Err(bad(cmd, "the destination document must have the same size"));
    }
    let channel = p.get("channel").cloned().unwrap_or(Value::Null);
    let name = p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).map(str::to_string);
    let prev = s.active_index();
    s.set_active(di);
    let r = s.edit("Save Selection", |doc, _| {
        let fmt = channel_format(doc);
        let existing = match &channel {
            Value::Null => None,
            Value::String(x) if x == "new" => None,
            v => match parse_ref(v, doc) {
                Some(ChanRef::Alpha(i)) if i < doc.channels.len() => Some(i),
                _ => return Err(bad(cmd, "`channel` must be \"new\" or an existing alpha channel")),
            },
        };
        match (existing, op.as_str()) {
            (None, _) | (_, "new") => {
                let name = name.clone().unwrap_or_else(|| next_alpha_name(doc, "Alpha"));
                doc.channels.push(AlphaChannel::new(name, plane_surface(&selection, area, fmt)));
                Ok(doc.channels.len() - 1)
            }
            (Some(i), op) => {
                let old = read_plane(&doc.channels[i].surface, area);
                let vals: Vec<f32> = match op {
                    "add" => old.iter().zip(&selection).map(|(a, b)| a.max(*b)).collect(),
                    "subtract" => old.iter().zip(&selection).map(|(a, b)| (a - b).max(0.0)).collect(),
                    "intersect" => old.iter().zip(&selection).map(|(a, b)| a.min(*b)).collect(),
                    _ => selection.clone(),
                };
                doc.channels[i].surface = plane_surface(&vals, area, fmt);
                if let Some(n) = &name {
                    doc.channels[i].name = n.clone();
                }
                Ok(i)
            }
        }
    });
    if let Some(i) = prev {
        s.set_active(i);
    }
    let i = r?;
    let name = s.documents()[di].doc.channels[i].name.clone();
    Ok(json!({ "channel": i, "name": name, "document": di }))
}

fn load_selection(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "select.loadSelection";
    let mode = match operation(p, "operation") {
        "new" | "replace" => SelectionMode::Replace,
        "add" => SelectionMode::Add,
        "subtract" => SelectionMode::Subtract,
        "intersect" => SelectionMode::Intersect,
        other => return Err(bad(cmd, format!("unknown operation `{other}`"))),
    };
    if p.get("channel").is_none() {
        return Err(bad(cmd, "missing `channel`"));
    }
    // `"targeted"`: the Channels panel's target, as a Photoshop action's "load the target
    // channel" step records it.
    let mut targeted = p.clone();
    if p.get("channel").and_then(Value::as_str) == Some("targeted") {
        let st = s.active().ok_or(EngineError::NoDocument)?;
        targeted["channel"] = match st.channel_view.target {
            ChannelTarget::Composite => json!("composite"),
            ChannelTarget::Color(k) => json!({ "color": k }),
            ChannelTarget::Alpha(i) => json!(i),
            ChannelTarget::LayerMask => json!("mask"),
            ChannelTarget::VectorMask => json!("vectorMask"),
        };
    }
    let p = &targeted;
    // Composite and colour channels load their luminosity / values, as ⌘-clicking them does.
    let plane = source(s, p, cmd, true)?.into_iter().next().ok_or_else(|| bad(cmd, "the channel has no pixels"))?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let label = match p.get("channel").and_then(|v| parse_ref(v, doc)) {
        Some(ChanRef::Transparency) => "Load Transparency",
        Some(ChanRef::LayerMask) => "Load Layer Mask",
        _ => "Load Selection",
    };
    let selected = s.edit(label, |doc, _| {
        doc.selection = sel::combine(doc.selection.as_ref(), &plane, doc.bounds(), mode);
        Ok(doc.selection.is_some())
    })?;
    Ok(json!({ "selected": selected }))
}

// ---------------------------------------------------------------------------------------------
// Quick Mask

/// Quick Mask channel for the current selection: white where selected. No selection is an empty
/// mask that shows no colour: white (nothing masked, as in Photoshop) when the colour indicates
/// masked areas, black (nothing selected) when it indicates selected areas. White and "no
/// selection" are told apart by [`Document::quick_mask_unselected`].
fn quick_mask_from(doc: &Document, o: &QuickMaskOptions) -> AlphaChannel {
    let area = doc.bounds();
    let fmt = channel_format(doc);
    let surface = match &doc.selection {
        Some(s) => plane_surface(&sel::mask_from_surface(Some(s), area), area, fmt),
        None => unselected_surface(area, fmt, o.indicates),
    };
    AlphaChannel { color: o.color, opacity: o.opacity, indicates: o.indicates, ..AlphaChannel::new("Quick Mask", surface) }
}

/// The colourless, nothing-selected Quick Mask for `indicates`.
fn unselected_surface(area: Rect, fmt: PixelFormat, indicates: ColorIndicates) -> Surface {
    let mut s = Surface::new(fmt);
    if indicates != ColorIndicates::SelectedAreas {
        s.fill_rect(area, &[1.0]);
    }
    s
}

/// Every pixel of `vals` reads as `target` on the 8-bit grid.
fn uniform(vals: &[f32], target: f32) -> bool {
    vals.iter().all(|v| (v - target).abs() <= 0.5 / 255.0)
}

/// The selection a Quick Mask channel turns back into. All white is Select All, unless Quick
/// Mask was entered with no selection and the white is its untouched masked-areas start.
fn selection_from(doc: &Document, ch: &AlphaChannel) -> Option<Surface> {
    let area = doc.bounds();
    let vals = read_plane(&ch.surface, area);
    if doc.quick_mask_unselected && ch.indicates != ColorIndicates::SelectedAreas && uniform(&vals, 1.0) {
        return None;
    }
    sel::combine(None, &vals, area, SelectionMode::Replace)
}

fn quick_mask(s: &mut Session, p: &Value) -> Result<Value> {
    let on = s.active().ok_or(EngineError::NoDocument)?.doc.quick_mask.is_some();
    let want = p.get("on").and_then(Value::as_bool).unwrap_or(!on);
    if want == on {
        return Ok(json!({ "quickMask": on }));
    }
    let opts = s.quick_mask_options;
    if want {
        s.edit("Edit in Quick Mask", |doc, _| {
            doc.quick_mask = Some(quick_mask_from(doc, &opts));
            doc.quick_mask_unselected = doc.selection.is_none();
            doc.selection = None;
            Ok(())
        })?;
        if let Some(st) = s.active_mut() {
            st.channel_view.target = ChannelTarget::Composite;
            st.channel_view.quick_mask_hidden = false;
        }
    } else {
        s.edit("Exit Quick Mask", |doc, _| {
            if let Some(q) = doc.quick_mask.take() {
                doc.selection = selection_from(doc, &q);
            }
            Ok(())
        })?;
    }
    Ok(json!({ "quickMask": want }))
}

// ---------------------------------------------------------------------------------------------
// Channel management

fn indicates_param(v: Option<&Value>) -> Option<ColorIndicates> {
    match v?.as_str()? {
        "masked" | "maskedAreas" => Some(ColorIndicates::MaskedAreas),
        "selected" | "selectedAreas" => Some(ColorIndicates::SelectedAreas),
        _ => None,
    }
}

fn color_value(p: &Value, key: &str) -> Option<Color> {
    p.get(key)?;
    let c = crate::commands::color_param(p, key, [1.0, 0.0, 0.0, 1.0]);
    Some(Color::rgb(c[0], c[1], c[2]))
}

fn new_channel(s: &mut Session, p: &Value) -> Result<Value> {
    let fill = p.get("fill").and_then(Value::as_str).unwrap_or("black").to_string();
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let indicates = indicates_param(p.get("indicates"));
    let color = color_value(p, "color");
    let opacity = p.get("opacity").and_then(Value::as_f64).map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
    let i = s.edit("New Channel", |doc, _| {
        let area = doc.bounds();
        let fmt = channel_format(doc);
        let surface = match fill.as_str() {
            "white" => plane_surface(&vec![1.0; area.width() as usize * area.height() as usize], area, fmt),
            "selection" => plane_surface(&sel::mask_from_surface(doc.selection.as_ref(), area), area, fmt),
            _ => Surface::new(fmt),
        };
        let mut ch = AlphaChannel::new(name.clone().unwrap_or_else(|| next_alpha_name(doc, "Alpha")), surface);
        if let Some(v) = indicates {
            ch.indicates = v;
        }
        if let Some(c) = color {
            ch.color = c;
        }
        if let Some(o) = opacity {
            ch.opacity = o;
        }
        doc.channels.push(ch);
        Ok(doc.channels.len() - 1)
    })?;
    // Photoshop targets and shows a new channel on its own.
    target(s, ChannelTarget::Alpha(i));
    Ok(json!({ "channel": i }))
}

fn new_spot(s: &mut Session, p: &Value) -> Result<Value> {
    let color = color_value(p, "color").unwrap_or(Color::rgb(1.0, 0.0, 0.0));
    let solidity = p.get("solidity").and_then(Value::as_f64).unwrap_or(0.0).clamp(0.0, 100.0) as f32 / 100.0;
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let from_sel = p.get("fromSelection").and_then(Value::as_bool).unwrap_or(true);
    let i = s.edit("New Spot Channel", |doc, _| {
        let area = doc.bounds();
        let fmt = channel_format(doc);
        let surface = if from_sel && doc.selection.is_some() {
            plane_surface(&sel::mask_from_surface(doc.selection.as_ref(), area), area, fmt)
        } else {
            Surface::new(fmt)
        };
        let name = name.clone().unwrap_or_else(|| next_alpha_name(doc, "Spot Color"));
        doc.channels.push(AlphaChannel { spot: Some((color, solidity)), ..AlphaChannel::new(name, surface) });
        Ok(doc.channels.len() - 1)
    })?;
    if let Some(st) = s.active_mut() {
        st.channel_view.set_alpha_visible(i, true);
        touch(st);
    }
    Ok(json!({ "channel": i }))
}

fn duplicate_channel(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.duplicate";
    let src = s.active().ok_or(EngineError::NoDocument)?;
    let area = src.doc.bounds();
    let r = p.get("channel").and_then(|v| parse_ref(v, &src.doc)).ok_or_else(|| bad(cmd, "missing or unknown `channel`"))?;
    let mut ch = match r {
        ChanRef::Alpha(i) => src.doc.channels.get(i).cloned().ok_or_else(|| bad(cmd, format!("no alpha channel {i}")))?,
        ChanRef::QuickMask => src.doc.quick_mask.clone().ok_or_else(|| bad(cmd, "not in Quick Mask mode"))?,
        other => {
            let plane = gray_plane(&src.doc, None, src.active_layer, other)?;
            AlphaChannel::new(String::new(), plane_surface(&plane, area, channel_format(&src.doc)))
        }
    };
    let base = match r {
        ChanRef::Alpha(_) | ChanRef::QuickMask => ch.name.clone(),
        ChanRef::Color(k) => color_names(edit_mode(&src.doc))[k].to_string(),
        _ => composite_name(edit_mode(&src.doc)).to_string(),
    };
    ch.name = p.get("name").and_then(Value::as_str).map_or_else(|| format!("{base} copy"), str::to_string);
    if p.get("invert").and_then(Value::as_bool).unwrap_or(false) {
        let v: Vec<f32> = read_plane(&ch.surface, area).into_iter().map(|v| 1.0 - v).collect();
        ch.surface = plane_surface(&v, area, ch.surface.format());
    }
    match p.get("document") {
        Some(Value::String(x)) if x == "new" => {
            // Photoshop makes a multichannel document; ours is grayscale with the channel as pixels.
            let doc = gray_document(&ch.name, &src.doc, &read_plane(&ch.surface, area));
            let i = s.add_document(doc, None);
            Ok(json!({ "document": i }))
        }
        _ => {
            let di = doc_index(s, p, "document")?;
            if s.documents()[di].doc.size != src.doc.size {
                return Err(bad(cmd, "the destination document must have the same size"));
            }
            let prev = s.active_index();
            s.set_active(di);
            let r = s.edit("Duplicate Channel", |doc, _| {
                let fmt = channel_format(doc);
                if ch.surface.format() != fmt {
                    ch.surface = ch.surface.convert(fmt);
                }
                doc.channels.push(ch);
                Ok(doc.channels.len() - 1)
            });
            if let Some(i) = prev {
                s.set_active(i);
            }
            Ok(json!({ "channel": r?, "document": di }))
        }
    }
}

/// A one-layer grayscale document whose pixels are `plane` (Split Channels, Duplicate Channel
/// to a new document).
fn gray_document(name: &str, like: &Document, plane: &[f32]) -> Document {
    let mut d = Document::new(name, like.size, ColorMode::Grayscale, like.depth);
    d.resolution_dpi = like.resolution_dpi;
    let fmt = d.pixel_format();
    let mut bg = Layer::raster("Background", fmt);
    bg.locks.transparency = true;
    bg.locks.position = true;
    if let Some(s) = bg.surface_mut() {
        let data: Vec<f32> = plane.iter().flat_map(|v| [v.clamp(0.0, 1.0), 1.0]).collect();
        s.write_region(like.bounds(), &data);
        s.prune();
    }
    d.layers.push(bg);
    d
}

fn delete_channel(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.delete";
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let i = match p.get("channel") {
        Some(_) => alpha_index(doc, p, "channel", cmd)?,
        None => match s.active().ok_or(EngineError::NoDocument)?.channel_view.target {
            ChannelTarget::Alpha(i) => i,
            _ => return Err(bad(cmd, "missing `channel`")),
        },
    };
    // `edit` validates the view against the shorter channel list before we can remap it.
    // Keep the original indices and eyes so a surviving final channel is not discarded.
    let view = s.active().ok_or(EngineError::NoDocument)?.channel_view.clone();
    s.edit("Delete Channel", |doc, _| {
        doc.channels.remove(i);
        Ok(())
    })?;
    if let Some(st) = s.active_mut() {
        st.channel_view = view;
        let v = &mut st.channel_view;
        if i < v.alpha_visible.len() {
            v.alpha_visible.remove(i);
        }
        v.target = match v.target {
            ChannelTarget::Alpha(t) if t == i => {
                v.color_hidden.clear();
                ChannelTarget::Composite
            }
            ChannelTarget::Alpha(t) if t > i => ChannelTarget::Alpha(t - 1),
            t => t,
        };
        fix_view_now(st);
    }
    Ok(Value::Null)
}

fn fix_view_now(st: &mut DocState) {
    fix_view(st);
    touch(st);
}

fn rename_channel(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.rename";
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let i = alpha_index(doc, p, "channel", cmd)?;
    let name = p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad(cmd, "missing `name`"))?.to_string();
    s.edit("Rename Channel", |doc, _| {
        doc.channels[i].name = name;
        Ok(())
    })?;
    Ok(Value::Null)
}

fn move_channel(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.move";
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let n = doc.channels.len();
    let i = alpha_index(doc, p, "channel", cmd)?;
    let to = p.get("to").and_then(Value::as_u64).map(|v| (v as usize).min(n - 1)).ok_or_else(|| bad(cmd, "missing `to`"))?;
    if to == i {
        return Ok(json!({ "channel": i }));
    }
    s.edit("Channel Order", |doc, _| {
        let c = doc.channels.remove(i);
        doc.channels.insert(to, c);
        Ok(())
    })?;
    if let Some(st) = s.active_mut() {
        let v = &mut st.channel_view;
        v.alpha_visible.resize(n, false);
        let vis = v.alpha_visible.remove(i);
        v.alpha_visible.insert(to, vis);
        let remap = |t: usize| {
            if t == i {
                to
            } else if i < t && t <= to {
                t - 1
            } else if to <= t && t < i {
                t + 1
            } else {
                t
            }
        };
        if let ChannelTarget::Alpha(t) = v.target {
            v.target = ChannelTarget::Alpha(remap(t));
        }
        touch(st);
    }
    Ok(json!({ "channel": to }))
}

/// Make `t` the edit target and show it the way Photoshop does when its row is clicked.
fn target(s: &mut Session, t: ChannelTarget) {
    if let Some(st) = s.active_mut() {
        let colors = color_count(&st.doc);
        let n = st.doc.channels.len();
        let v = &mut st.channel_view;
        v.target = t;
        match t {
            // Like clicking RGB in Photoshop: the alpha overlays go away with their eyes.
            ChannelTarget::Composite => {
                v.color_hidden.clear();
                v.alpha_visible.clear();
            }
            ChannelTarget::LayerMask | ChannelTarget::VectorMask => v.color_hidden.clear(),
            ChannelTarget::Color(k) => {
                v.color_hidden = (0..colors).map(|c| c != k).collect();
                v.alpha_visible.clear();
            }
            ChannelTarget::Alpha(i) => {
                v.color_hidden = vec![true; colors];
                v.alpha_visible = (0..n).map(|c| c == i).collect();
            }
        }
        fix_view_now(st);
    }
    s.tools.target_mask(t == ChannelTarget::LayerMask);
}

fn parse_target(s: &Session, p: &Value, cmd: &str) -> Result<ChannelTarget> {
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    match p.get("channel").map_or(Some(ChanRef::Composite), |v| parse_ref(v, doc)) {
        Some(ChanRef::Composite) => Ok(ChannelTarget::Composite),
        Some(ChanRef::Color(k)) if k < color_count(doc) && color_count(doc) > 1 => Ok(ChannelTarget::Color(k)),
        Some(ChanRef::Alpha(i)) if i < doc.channels.len() => Ok(ChannelTarget::Alpha(i)),
        Some(ChanRef::LayerMask) => {
            let active = crate::active_layer_of(s).map_err(|e| bad(cmd, e))?;
            if active.mask.is_none() {
                return Err(bad(cmd, "the active layer has no layer mask"));
            }
            Ok(ChannelTarget::LayerMask)
        }
        Some(ChanRef::VectorMask) => {
            let active = crate::active_layer_of(s).map_err(|e| bad(cmd, e))?;
            if active.vector_mask.is_none() {
                return Err(bad(cmd, "the active layer has no vector mask"));
            }
            Ok(ChannelTarget::VectorMask)
        }
        _ => Err(bad(cmd, "`channel` must be \"composite\", \"mask\", \"vectorMask\", a colour channel name or an alpha channel")),
    }
}

fn target_cmd(s: &mut Session, p: &Value) -> Result<Value> {
    let t = parse_target(s, p, "channel.target")?;
    target(s, t);
    Ok(list(s))
}

/// The channel shortcuts' modifier as the platform writes it, for error messages (`⌘4`, `Ctrl+4`).
const SLOT_KEY: &str = if cfg!(target_os = "macos") { "⌘" } else { "Ctrl+" };

/// ⌘2 composite, ⌘3… colour channels then alpha channels (Photoshop CC defaults).
fn target_slot(s: &mut Session, slot: usize) -> Result<Value> {
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let colors = color_count(doc);
    let shown_colors = if colors > 1 { colors } else { 0 };
    let t = match slot {
        2 => ChannelTarget::Composite,
        n if n - 3 < shown_colors => ChannelTarget::Color(n - 3),
        n if n - 3 - shown_colors < doc.channels.len() => ChannelTarget::Alpha(n - 3 - shown_colors),
        _ => return Err(EngineError::Other(format!("no channel for {SLOT_KEY}{slot}"))),
    };
    target(s, t);
    Ok(list(s))
}

fn set_visible(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.setVisible";
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let r = p.get("channel").and_then(|v| parse_ref(v, doc)).ok_or_else(|| bad(cmd, "missing or unknown `channel`"))?;
    let colors = color_count(doc);
    let n = doc.channels.len();
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let v = &mut st.channel_view;
    let visible = |cur: bool| p.get("visible").and_then(Value::as_bool).unwrap_or(!cur);
    match r {
        ChanRef::Composite => {
            let all = v.visible_colors(colors) == colors;
            let on = visible(all);
            v.color_hidden = vec![!on; colors];
        }
        ChanRef::Color(k) if k < colors => {
            let on = visible(v.color_visible(k));
            v.set_color_visible(k, on);
        }
        ChanRef::Alpha(i) if i < n => {
            let on = visible(v.alpha_shown(i));
            v.set_alpha_visible(i, on);
        }
        ChanRef::QuickMask => v.quick_mask_hidden = !visible(!v.quick_mask_hidden),
        _ => return Err(bad(cmd, "that channel has no visibility")),
    }
    fix_view_now(st);
    Ok(list(s))
}

fn options(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.options";
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let r = p.get("channel").and_then(|v| parse_ref(v, doc)).ok_or_else(|| bad(cmd, "missing or unknown `channel`"))?;
    let name = p.get("name").and_then(Value::as_str).map(str::trim).filter(|n| !n.is_empty()).map(str::to_string);
    let color = color_value(p, "color");
    let opacity = p.get("opacity").and_then(Value::as_f64).map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
    let solidity = p.get("solidity").and_then(Value::as_f64).map(|v| (v as f32 / 100.0).clamp(0.0, 1.0));
    let kind = p.get("indicates").and_then(Value::as_str).map(str::to_string);
    let apply = move |ch: &mut AlphaChannel| -> Result<()> {
        if let Some(n) = &name {
            ch.name = n.clone();
        }
        match kind.as_deref() {
            Some("spot") => {
                let (c, sol) = ch.spot.unwrap_or((ch.color, 0.0));
                ch.spot = Some((color.unwrap_or(c), solidity.unwrap_or(sol)));
                return Ok(());
            }
            Some(k) => {
                ch.indicates = indicates_param(Some(&json!(k))).ok_or_else(|| bad(cmd, format!("unknown `indicates` {k}")))?;
                ch.spot = None;
            }
            None => {}
        }
        match &mut ch.spot {
            Some((c, sol)) => {
                *c = color.unwrap_or(*c);
                *sol = solidity.or(opacity).unwrap_or(*sol);
            }
            None => {
                ch.color = color.unwrap_or(ch.color);
                ch.opacity = opacity.unwrap_or(ch.opacity);
            }
        }
        Ok(())
    };
    match r {
        ChanRef::Alpha(i) if i < doc.channels.len() => {
            s.edit("Channel Options", |doc, _| apply(&mut doc.channels[i]))?;
        }
        ChanRef::QuickMask => {
            // Quick Mask Options persist for the session; the live Quick Mask follows them.
            let mut probe = AlphaChannel {
                color: s.quick_mask_options.color,
                opacity: s.quick_mask_options.opacity,
                indicates: s.quick_mask_options.indicates,
                ..AlphaChannel::new("Quick Mask", Surface::new(PixelFormat::GRAY8))
            };
            apply(&mut probe)?;
            if probe.spot.is_some() {
                return Err(bad(cmd, "Quick Mask can't be a spot channel"));
            }
            s.quick_mask_options = QuickMaskOptions { color: probe.color, opacity: probe.opacity, indicates: probe.indicates };
            if s.active().is_some_and(|d| d.doc.quick_mask.is_some()) {
                let o = s.quick_mask_options;
                s.edit("Quick Mask Options", |doc, _| {
                    let area = doc.bounds();
                    let fmt = channel_format(doc);
                    let unselected = doc.quick_mask_unselected;
                    if let Some(q) = &mut doc.quick_mask {
                        // An untouched no-selection mask stays colourless (nothing selected)
                        // when the colour switches between masked and selected areas.
                        if unselected && q.indicates != o.indicates {
                            let empty = if q.indicates == ColorIndicates::SelectedAreas { 0.0 } else { 1.0 };
                            if uniform(&read_plane(&q.surface, area), empty) {
                                q.surface = unselected_surface(area, fmt, o.indicates);
                            }
                        }
                        q.color = o.color;
                        q.opacity = o.opacity;
                        q.indicates = o.indicates;
                    }
                    Ok(())
                })?;
            }
        }
        _ => return Err(bad(cmd, "`channel` must be an alpha/spot channel or \"quickMask\"")),
    }
    Ok(list(s))
}

/// Merge Spot Channel: flattens the image and prints the spot ink into the colour channels.
fn merge_spot(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.mergeSpot";
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let i = match p.get("channel") {
        Some(_) => alpha_index(doc, p, "channel", cmd)?,
        None => match s.active().ok_or(EngineError::NoDocument)?.channel_view.target {
            ChannelTarget::Alpha(i) => i,
            _ => doc.channels.iter().position(|c| c.spot.is_some()).ok_or_else(|| bad(cmd, "no spot channel"))?,
        },
    };
    let (ink, solidity) = doc.channels[i].spot.ok_or_else(|| bad(cmd, "not a spot channel"))?;
    s.edit("Merge Spot Channel", |doc, active| {
        let area = doc.bounds();
        let fmt = doc.pixel_format();
        let comp = photocraft_compose::render(doc, area);
        let k = read_plane(&doc.channels[i].surface, area);
        let ink = ink.to_rgb();
        // Ink over the image: opaque at 100% solidity, multiplied (transparent ink) at 0%.
        let data: Vec<f32> = comp
            .px
            .iter()
            .zip(&k)
            .flat_map(|(c, &v)| {
                let rgb: [f32; 3] = std::array::from_fn(|j| {
                    let over = solidity * ink[j] + (1.0 - solidity) * c[j] * ink[j];
                    c[j] + (over - c[j]) * v.clamp(0.0, 1.0)
                });
                from_rgba(&fmt, [rgb[0], rgb[1], rgb[2], 1.0])
            })
            .collect();
        let mut bg = Layer::raster("Background", fmt);
        bg.locks.transparency = true;
        bg.locks.position = true;
        if let Some(sf) = bg.surface_mut() {
            sf.write_region(area, &data);
            sf.prune();
        }
        let id = bg.id;
        doc.layers = vec![bg];
        doc.channels.remove(i);
        *active = Some(id);
        Ok(())
    })?;
    if let Some(st) = s.active_mut() {
        st.channel_view = ChannelView::default();
        fix_view_now(st);
    }
    Ok(Value::Null)
}

/// Split Channels: one grayscale document per colour and alpha channel; closes the original.
fn split(s: &mut Session, p: &Value) -> Result<Value> {
    let idx = s.active_index().ok_or(EngineError::NoDocument)?;
    let doc = s.documents()[idx].doc.clone();
    let area = doc.bounds();
    let (px, n) = native_pixels(&doc, None)?;
    let mode = edit_mode(&doc);
    let names = color_names(mode);
    let short = |name: &str| name.chars().next().map(|c| c.to_ascii_uppercase().to_string()).unwrap_or_default();
    let mut docs: Vec<Document> = Vec::new();
    for (c, name) in names.iter().enumerate() {
        let plane: Vec<f32> = px.chunks_exact(n).map(|p| p[c]).collect();
        docs.push(gray_document(&format!("{}_{}", doc.name, short(name)), &doc, &plane));
    }
    for ch in &doc.channels {
        docs.push(gray_document(&format!("{}_{}", doc.name, ch.name), &doc, &read_plane(&ch.surface, area)));
    }
    if p.get("closeOriginal").and_then(Value::as_bool).unwrap_or(true) {
        s.close(idx);
    }
    let made: Vec<usize> = docs.into_iter().map(|d| s.add_document(d, None)).collect();
    Ok(json!({ "documents": made }))
}

/// Open grayscale documents with the active document's size, active first.
fn mergeable(s: &Session) -> Vec<usize> {
    let Some(a) = s.active() else { return Vec::new() };
    let ok = |d: &DocState| d.doc.mode == ColorMode::Grayscale && d.doc.size == a.doc.size;
    let mut v: Vec<usize> = s.active_index().into_iter().filter(|_| ok(a)).collect();
    v.extend(s.documents().iter().enumerate().filter(|(i, d)| Some(*i) != s.active_index() && ok(d)).map(|(i, _)| i));
    v
}

/// Merge Channels: grayscale documents of one size become the channels of an RGB, CMYK or Lab
/// document (extra documents become alpha channels); the sources close.
fn merge(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "channel.merge";
    let mode = match p.get("mode").and_then(Value::as_str).unwrap_or("rgb") {
        "rgb" => ColorMode::Rgb,
        "cmyk" => ColorMode::Cmyk,
        "lab" => ColorMode::Lab,
        other => return Err(bad(cmd, format!("unknown mode `{other}`"))),
    };
    let srcs: Vec<usize> = match p.get("documents").and_then(Value::as_array) {
        Some(a) => a
            .iter()
            .map(|v| v.as_u64().map(|i| i as usize).filter(|i| *i < s.documents().len()).ok_or_else(|| bad(cmd, "bad document index")))
            .collect::<Result<_>>()?,
        None => mergeable(s).into_iter().take(mode.color_channels()).collect(),
    };
    if srcs.len() < mode.color_channels() {
        return Err(bad(cmd, format!("{mode:?} needs {} grayscale documents, got {}", mode.color_channels(), srcs.len())));
    }
    // Each source closes by its *current* index, so a repeated index would shift the later
    // removals onto a document that was never named (#934). Refuse it rather than guess.
    let mut seen = std::collections::HashSet::new();
    if let Some(dup) = srcs.iter().find(|i| !seen.insert(**i)) {
        return Err(bad(cmd, format!("document {dup} is listed more than once")));
    }
    let Some(first) = srcs.first().and_then(|&i| s.documents().get(i)).map(|d| d.doc.clone()) else {
        return Err(bad(cmd, "bad document index"));
    };
    let mut planes = Vec::new();
    for &i in &srcs {
        let d = &s.documents()[i].doc;
        if d.size != first.size || d.mode != ColorMode::Grayscale {
            return Err(bad(cmd, format!("\"{}\" is not a grayscale document of the same size", d.name)));
        }
        planes.push(gray_plane(d, None, None, ChanRef::Composite)?);
    }
    let mut out = Document::new(p.get("name").and_then(Value::as_str).unwrap_or("Untitled"), first.size, mode, first.depth);
    out.resolution_dpi = first.resolution_dpi;
    let fmt = out.pixel_format();
    let cc = mode.color_channels();
    let npx = planes[0].len();
    let mut data = Vec::with_capacity(npx * fmt.channels());
    for i in 0..npx {
        for pl in &planes[..cc] {
            data.push(pl[i].clamp(0.0, 1.0));
        }
        data.push(1.0);
    }
    let mut bg = Layer::raster("Background", fmt);
    bg.locks.transparency = true;
    bg.locks.position = true;
    if let Some(sf) = bg.surface_mut() {
        sf.write_region(out.bounds(), &data);
        sf.prune();
    }
    out.layers.push(bg);
    let area = out.bounds();
    let cf = channel_format(&out);
    for (k, pl) in planes[cc..].iter().enumerate() {
        out.channels.push(AlphaChannel::new(format!("Alpha {}", k + 1), plane_surface(pl, area, cf)));
    }
    let mut close: Vec<usize> = srcs.clone();
    close.sort_unstable();
    for i in close.into_iter().rev() {
        s.close(i);
    }
    let i = s.add_document(out, None);
    Ok(json!({ "document": i }))
}

// ---------------------------------------------------------------------------------------------
// Apply Image / Calculations

fn apply_image(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "image.applyImage";
    let blending = Blending::parse(p).map_err(|e| bad(cmd, e))?;
    let opacity = p.get("opacity").and_then(Value::as_f64).unwrap_or(100.0).clamp(0.0, 100.0) as f32 / 100.0;
    let preserve = p.get("preserveTransparency").and_then(Value::as_bool).unwrap_or(false);
    let src_spec = source_spec(p, "source", "source").unwrap_or_else(|| json!({}));
    let src = source(s, &src_spec, cmd, false)?;
    let mask = mask_plane(s, p, cmd)?;
    let layer = s.active().and_then(|d| d.active_layer);
    s.edit("Apply Image", |doc, _| {
        let area = doc.bounds();
        let selection = doc.selection.as_ref().map(|m| sel::mask_from_surface(Some(m), area));
        let weight = |i: usize| opacity * mask.as_ref().map_or(1.0, |m| m[i]) * selection.as_ref().map_or(1.0, |m| m[i]);
        if is_channel_target(p) {
            // A grayscale target gets the luminosity of a colour source (one plane).
            let srcp =
                if src.len() == 1 { src[0].clone() } else { (0..src[0].len()).map(|i| src.iter().map(|pl| pl[i]).sum::<f32>() / src.len() as f32).collect() };
            let (surf, _) = target_surface(doc, None, p)?;
            let base = read_plane(surf, area);
            let out: Vec<f32> = base.iter().enumerate().map(|(i, &b)| b + (blending.apply(b, srcp[i]) - b) * weight(i)).collect();
            *surf = plane_surface(&out, area, surf.format());
            return Ok(());
        }
        let id = layer.ok_or_else(|| EngineError::Other("no active layer".into()))?;
        let fmt = doc.pixel_format();
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let sf = surf.format();
        if sf != fmt {
            return Err(EngineError::Other("the target layer's format doesn't match the document".into()));
        }
        let n = sf.channels();
        let colors = sf.mode.color_channels();
        let mut v = surf.read_region(area);
        for (i, px) in v.chunks_exact_mut(n).enumerate() {
            let base_a = if sf.alpha { px[colors].clamp(0.0, 1.0) } else { 1.0 };
            let mut w = weight(i);
            if preserve {
                w *= base_a;
            }
            if w <= 0.0 {
                continue;
            }
            // Painted over the layer like a brush: the blend only acts where the layer has
            // pixels (where it is transparent the source shows as is), and the alpha grows by
            // source-over, so applying to an empty layer copies the source in any mode.
            let out_a = if preserve { base_a } else { w + base_a * (1.0 - w) };
            for (c, b) in px[..colors].iter_mut().enumerate() {
                let top = src.get(c).unwrap_or(&src[0])[i];
                let blended = top + (blending.apply(*b, top) - top) * base_a;
                *b = if preserve { *b + (blended - *b) * w } else { (w * blended + base_a * (1.0 - w) * *b) / out_a };
            }
            if sf.alpha {
                px[colors] = out_a;
            }
        }
        surf.write_region(area, &v);
        surf.prune();
        Ok(())
    })?;
    Ok(Value::Null)
}

fn calculations(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "image.calculations";
    let blending = Blending::parse(p).map_err(|e| bad(cmd, e))?;
    let opacity = p.get("opacity").and_then(Value::as_f64).unwrap_or(100.0).clamp(0.0, 100.0) as f32 / 100.0;
    let s1 = source(s, &source_spec(p, "source1", "source1").unwrap_or_else(|| json!({})), cmd, true)?.remove(0);
    let s2 = source(s, &source_spec(p, "source2", "source2").unwrap_or_else(|| json!({})), cmd, true)?.remove(0);
    let mask = mask_plane(s, p, cmd)?;
    // Source 1 is blended onto Source 2.
    let out: Vec<f32> =
        s2.iter().zip(&s1).enumerate().map(|(i, (&b, &t))| b + (blending.apply(b, t) - b) * opacity * mask.as_ref().map_or(1.0, |m| m[i])).collect();
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    match p.get("result").and_then(Value::as_str).unwrap_or("newChannel") {
        "newChannel" => {
            let i = s.edit("Calculations", |doc, _| {
                let name = name.clone().unwrap_or_else(|| next_alpha_name(doc, "Alpha"));
                let area = doc.bounds();
                let fmt = channel_format(doc);
                doc.channels.push(AlphaChannel::new(name, plane_surface(&out, area, fmt)));
                Ok(doc.channels.len() - 1)
            })?;
            Ok(json!({ "channel": i }))
        }
        "newDocument" => {
            let like = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
            let d = gray_document(name.as_deref().unwrap_or("Untitled"), &like, &out);
            let i = s.add_document(d, None);
            Ok(json!({ "document": i }))
        }
        "selection" => {
            let selected = s.edit("Calculations", |doc, _| {
                doc.selection = sel::combine(None, &out, doc.bounds(), SelectionMode::Replace);
                Ok(doc.selection.is_some())
            })?;
            Ok(json!({ "selected": selected }))
        }
        other => Err(bad(cmd, format!("unknown result `{other}` (newChannel|newDocument|selection)"))),
    }
}

// ---------------------------------------------------------------------------------------------
// State as JSON

/// The Channels panel rows and view state of a document (also part of `inspect`).
pub fn channels_json(st: &DocState) -> Value {
    let doc = &st.doc;
    let mode = edit_mode(doc);
    let v = &st.channel_view;
    let colors = color_count(doc);
    json!({
        "composite": composite_name(mode),
        "colors": color_names(mode).iter().enumerate().map(|(k, n)| json!({ "name": n, "visible": v.color_visible(k) })).collect::<Vec<_>>(),
        "alpha": doc.channels.iter().enumerate().map(|(i, c)| json!({
            "index": i,
            "name": c.name,
            "visible": v.alpha_shown(i),
            "spot": c.spot.map(|(col, sol)| json!({ "color": col.to_rgb(), "solidity": sol })),
            "color": c.color.to_rgb(),
            "opacity": c.opacity,
            "indicates": c.indicates,
        })).collect::<Vec<_>>(),
        "quickMask": doc.quick_mask.as_ref().map(|q| json!({ "visible": !v.quick_mask_hidden, "color": q.color.to_rgb(), "opacity": q.opacity, "indicates": q.indicates })),
        "target": v.target,
        "compositeVisible": v.visible_colors(colors) == colors,
        "layerMaskView": crate::mask_view_cmds::view_json(st),
    })
}

fn list(s: &Session) -> Value {
    s.active().map(channels_json).unwrap_or(Value::Null)
}

// ---------------------------------------------------------------------------------------------
// Registry

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $sc:expr, $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: $sc, params: $params, enabled: $en, run: $run, journal: true }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "select.saveSelection",
            "Save Selection…",
            ["Select"],
            None,
            r##"{"name":str?,"channel":"new"|index|name="new","document":index?,"operation":"new|replace|add|subtract|intersect"="new"}"##,
            has_selection,
            save_selection
        ),
        spec!(
            "select.loadSelection",
            "Load Selection…",
            ["Select"],
            None,
            r##"{"channel":index|name|"composite"|"red|green|blue|…"|"transparency"|"mask"|"vectorMask"|"quickMask"|"selection"|"targeted" (the Channels panel's target),"layer":id?,"document":index?,"invert":bool=false,"operation":"new|add|subtract|intersect"="new"}"##,
            has_doc,
            load_selection
        ),
        spec!("select.editInQuickMaskMode", "Edit in Quick Mask Mode", ["Select"], Some("Q"), r##"{"on":bool?=toggle}"##, has_doc, quick_mask),
        spec!(
            "image.applyImage",
            "Apply Image…",
            ["Image"],
            None,
            r##"{"source":{"document":index?,"layer":id|name|"merged"="merged","channel":"composite"|"red|…"|index|"selection"|"transparency"|"mask"="composite","invert":bool=false},"blending":"normal|multiply|screen|overlay|softLight|hardLight|colorDodge|colorBurn|darken|lighten|difference|exclusion|linearBurn|linearDodge|add|subtract|…"="multiply","opacity":0..100=100,"scale":1..2=1,"offset":-255..255=0,"preserveTransparency":bool=false,"mask":{"document","layer","channel","invert"}?,"sourceChannel|sourceDocument|sourceLayer|sourceInvert|maskChannel|…":flat form of source/mask?}"##,
            has_apply_target,
            apply_image
        ),
        spec!(
            "image.calculations",
            "Calculations…",
            ["Image"],
            None,
            r##"{"source1":{"document","layer","channel","invert"},"source2":{…},"blending":"multiply|…"="multiply","opacity":0..100=100,"scale":1..2=1,"offset":-255..255=0,"mask":{…}?,"result":"newChannel|newDocument|selection"="newChannel","name":str?}"##,
            has_doc,
            calculations
        ),
        spec!(
            "channel.new",
            "New Channel…",
            [],
            None,
            r##"{"name":str?,"fill":"black|white|selection"="black","color":"#rrggbb"="#ff0000","opacity":0..100=50,"indicates":"masked|selected"="masked"}"##,
            has_doc,
            new_channel
        ),
        spec!(
            "channel.newSpot",
            "New Spot Channel…",
            [],
            None,
            r##"{"name":str?,"color":"#rrggbb"="#ff0000","solidity":0..100=0,"fromSelection":bool=true}"##,
            has_doc,
            new_spot
        ),
        spec!(
            "channel.duplicate",
            "Duplicate Channel…",
            [],
            None,
            r##"{"channel":index|name|"red|…"|"composite"|"quickMask","name":str?,"document":index|"new"?=active,"invert":bool=false}"##,
            has_doc,
            duplicate_channel
        ),
        spec!("channel.delete", "Delete Channel", [], None, r##"{"channel":index|name?=targeted}"##, has_channels, delete_channel),
        spec!("channel.rename", "Rename Channel", [], None, r##"{"channel":index|name,"name":str}"##, has_channels, rename_channel),
        spec!("channel.move", "Reorder Channel", [], None, r##"{"channel":index|name,"to":index}"##, has_channels, move_channel),
        spec!(
            "channel.target",
            "Target Channel",
            [],
            None,
            r##"{"channel":"composite"|"mask"|"vectorMask"|"red|green|blue|…"|index|name="composite"}"##,
            has_doc,
            target_cmd
        ),
        spec!(
            "channel.setVisible",
            "Channel Visibility",
            [],
            None,
            r##"{"channel":"composite"|"red|…"|index|name|"quickMask","visible":bool?=toggle}"##,
            has_doc,
            set_visible
        ),
        spec!(
            "channel.options",
            "Channel Options…",
            [],
            None,
            r##"{"channel":index|name|"quickMask","name":str?,"indicates":"masked|selected|spot"?,"color":"#rrggbb"?,"opacity":0..100?,"solidity":0..100?}"##,
            has_doc,
            options
        ),
        spec!("channel.mergeSpot", "Merge Spot Channel", [], None, r##"{"channel":index|name?=targeted or first spot}"##, has_spot, merge_spot),
        spec!("channel.split", "Split Channels", [], None, r##"{"closeOriginal":bool=true}"##, can_split, split),
        spec!(
            "channel.merge",
            "Merge Channels…",
            [],
            None,
            r##"{"mode":"rgb|cmyk|lab"="rgb","documents":[index,…]?=open grayscale docs of the active size,"name":str?}"##,
            can_merge,
            merge
        ),
        CommandSpec {
            id: "channel.list",
            label: "Channels",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: has_doc,
            run: |s, _| Ok(list(s)),
            journal: false,
        },
        spec!("channel.target.composite", "Target Composite Channel", [], Some("Cmd+2"), "{}", has_doc, |s, _| target_slot(s, 2)),
        // ⌘\ targets the active layer's mask, as clicking its thumbnail does.
        spec!("channel.target.layerMask", "Target Layer Mask", [], Some("Cmd+\\"), "{}", has_layer_mask, |s, _| target_cmd(
            s,
            &serde_json::json!({"channel": "mask"})
        )),
        // ⌘⌥2 loads the image's luminosity as a selection (⌘-click on the composite channel).
        spec!("select.loadLuminosity", "Load Luminosity", [], Some("Cmd+Alt+2"), "{}", has_doc, |s, _| load_selection(
            s,
            &serde_json::json!({"channel": "composite"})
        )),
        spec!("channel.target.slot3", "Target Channel 3", [], Some("Cmd+3"), "{}", slot_ok::<3>, |s, _| target_slot(s, 3)),
        spec!("channel.target.slot4", "Target Channel 4", [], Some("Cmd+4"), "{}", slot_ok::<4>, |s, _| target_slot(s, 4)),
        spec!("channel.target.slot5", "Target Channel 5", [], Some("Cmd+5"), "{}", slot_ok::<5>, |s, _| target_slot(s, 5)),
        spec!("channel.target.slot6", "Target Channel 6", [], Some("Cmd+6"), "{}", slot_ok::<6>, |s, _| target_slot(s, 6)),
        spec!("channel.target.slot7", "Target Channel 7", [], Some("Cmd+7"), "{}", slot_ok::<7>, |s, _| target_slot(s, 7)),
        spec!("channel.target.slot8", "Target Channel 8", [], Some("Cmd+8"), "{}", slot_ok::<8>, |s, _| target_slot(s, 8)),
        spec!("channel.target.slot9", "Target Channel 9", [], Some("Cmd+9"), "{}", slot_ok::<9>, |s, _| target_slot(s, 9)),
    ]
}

/// Target Layer Mask: the active layer has a mask.
fn has_layer_mask(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    let layer = st.active_layer.and_then(|id| st.doc.layer(id)).ok_or("no active layer")?;
    layer.mask.as_ref().map(|_| ()).ok_or_else(|| "the active layer has no layer mask".to_string())
}

fn slot_ok<const N: usize>(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let colors = color_count(&d.doc);
    let shown = if colors > 1 { colors } else { 0 };
    if N - 3 < shown + d.doc.channels.len() { Ok(()) } else { Err(format!("no channel for {SLOT_KEY}{N}")) }
}

fn has_apply_target(s: &Session) -> std::result::Result<(), String> {
    if edits_channel(s) {
        return Ok(());
    }
    let d = s.active().ok_or("no document open")?;
    let l = d.active_layer.and_then(|id| d.doc.layer(id)).ok_or("no active layer")?;
    if matches!(l.content, LayerContent::Raster(_)) {
        Ok(())
    } else {
        Err(format!("Apply Image needs a pixel layer (active layer is {} {} layer)", l.content.article(), l.content.kind_name()))
    }
}
