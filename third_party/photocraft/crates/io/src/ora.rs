//! OpenRaster (`.ora`) read and write. Layout from the public OpenRaster specification
//! (https://www.openraster.org/, `stack.xml` 0.0.x): a ZIP whose first, stored entry is
//! `mimetype` = `image/openraster`, a `stack.xml` layer tree, one PNG per layer, and a
//! `mergedimage.png` / `Thumbnails/thumbnail.png` rendering.
//!
//! Read: raster layers (PNG, any position and size), groups (isolated and pass-through),
//! names, opacity, visibility, `svg:*` and Krita's `krita:*` composite operations, and the
//! `edit-locked` / `alpha-preserve` flags. Everything else is skipped with a warning.
//! Write: raster layers and groups as above; other layer kinds are written as their rendered
//! pixels, layer masks are applied to the pixels, and whatever ORA cannot hold is warned about.

#[cfg(test)]
mod tests;

use photocraft_codecs::{self as codecs, ChannelLayout, Format, Image, SampleType as CSample};
use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_format::zip::{ZipReader, ZipWriter};
use photocraft_geom::{Rect, Size};
use photocraft_raster::{Interrupt, Surface};

use crate::{ExportResult, ImportResult, IoError};

type Result<T> = std::result::Result<T, IoError>;

/// The `mimetype` entry of an OpenRaster archive.
pub(crate) const MIMETYPE: &[u8] = b"image/openraster";
/// Decoded pixels across every layer (and the stack.xml) are capped, as for PDN.
pub(crate) const MAX_PIXEL_BYTES: usize = 1 << 30;
/// Canvas side limit (Photoshop's PSB limit).
pub(crate) const MAX_SIDE: u32 = 300_000;
/// Layers and groups in one document.
pub(crate) const MAX_NODES: usize = 8192;
const MAX_STACK_XML: usize = 16 << 20;
const MAX_ENTRY: usize = 512 << 20;
const THUMBNAIL_SIDE: u32 = 256;

fn invalid(message: impl Into<String>) -> IoError {
    IoError::Ora(message.into())
}

/// The payload of a ZIP's first entry when it is a stored file named `mimetype`, which is how
/// OpenRaster and Krita mark their archives (so the type is readable without unzipping).
pub(crate) fn zip_mimetype(bytes: &[u8]) -> Option<&[u8]> {
    let u16_at = |o: usize| bytes.get(o..o + 2).and_then(|s| <[u8; 2]>::try_from(s).ok()).map(u16::from_le_bytes);
    let u32_at = |o: usize| bytes.get(o..o + 4).and_then(|s| <[u8; 4]>::try_from(s).ok()).map(u32::from_le_bytes);
    if u32_at(0)? != 0x0403_4b50 || u16_at(8)? != 0 {
        return None;
    }
    let size = usize::try_from(u32_at(18)?).ok()?;
    let name_len = usize::from(u16_at(26)?);
    let extra_len = usize::from(u16_at(28)?);
    if bytes.get(30..30 + name_len)? != b"mimetype" || size > 256 {
        return None;
    }
    let start = 30 + name_len + extra_len;
    bytes.get(start..start.checked_add(size)?)
}

/// `true` for an OpenRaster archive (by its `mimetype` entry).
pub(crate) fn is_ora(bytes: &[u8]) -> bool {
    zip_mimetype(bytes) == Some(MIMETYPE)
}

/// Blend modes by Krita composite-op id: Krita's `.kra` `compositeop` attribute and the
/// `krita:<id>` composite operations Krita writes into ORA for modes SVG has no name for.
/// Returns the mode and whether it only approximates Krita's formula.
pub(crate) fn krita_blend(id: &str) -> Option<(BlendMode, bool)> {
    use BlendMode as B;
    Some(match id {
        "normal" => (B::Normal, false),
        "multiply" => (B::Multiply, false),
        "screen" => (B::Screen, false),
        "overlay" => (B::Overlay, false),
        "darken" => (B::Darken, false),
        "lighten" => (B::Lighten, false),
        "dodge" => (B::ColorDodge, false),
        "burn" => (B::ColorBurn, false),
        "hard_light" => (B::HardLight, false),
        "soft_light" => (B::SoftLight, false),
        // The W3C soft light curve, close to but not Photoshop's.
        "soft_light_svg" => (B::SoftLight, true),
        "diff" => (B::Difference, false),
        "exclusion" => (B::Exclusion, false),
        "linear_burn" => (B::LinearBurn, false),
        "linear_dodge" | "add" => (B::LinearDodge, false),
        "subtract" => (B::Subtract, false),
        "divide" => (B::Divide, false),
        "vivid_light" => (B::VividLight, false),
        "linear light" => (B::LinearLight, false),
        "pin_light" => (B::PinLight, false),
        "hard_mix_photoshop" => (B::HardMix, false),
        // Krita's own Hard Mix thresholds differently from Photoshop's.
        "hard mix" => (B::HardMix, true),
        "darker color" => (B::DarkerColor, false),
        "lighter color" => (B::LighterColor, false),
        "hue" => (B::Hue, false),
        "saturation" => (B::Saturation, false),
        "color" => (B::Color, false),
        "luminize" => (B::Luminosity, false),
        "dissolve" => (B::Dissolve, false),
        "reflect" => (B::Reflect, false),
        "glow" => (B::Glow, false),
        "negation" => (B::Negation, false),
        "xor" => (B::Xor, true),
        _ => return None,
    })
}

/// The Krita composite-op id written for `mode` (`None` for modes Krita has no counterpart for).
pub(crate) fn krita_id(mode: BlendMode) -> Option<&'static str> {
    use BlendMode as B;
    Some(match mode {
        B::Normal | B::PassThrough => "normal",
        B::Dissolve => "dissolve",
        B::Darken => "darken",
        B::Multiply => "multiply",
        B::ColorBurn => "burn",
        B::LinearBurn => "linear_burn",
        B::DarkerColor => "darker color",
        B::Lighten => "lighten",
        B::Screen => "screen",
        B::ColorDodge => "dodge",
        B::LinearDodge => "linear_dodge",
        B::LighterColor => "lighter color",
        B::Overlay => "overlay",
        B::SoftLight => "soft_light",
        B::HardLight => "hard_light",
        B::VividLight => "vivid_light",
        B::LinearLight => "linear light",
        B::PinLight => "pin_light",
        B::HardMix => "hard_mix_photoshop",
        B::Difference => "diff",
        B::Exclusion => "exclusion",
        B::Subtract => "subtract",
        B::Divide => "divide",
        B::Hue => "hue",
        B::Saturation => "saturation",
        B::Color => "color",
        B::Luminosity => "luminize",
        B::Reflect => "reflect",
        B::Glow => "glow",
        B::Negation => "negation",
        B::Xor | B::PaintNetColorBurn | B::PaintNetColorDodge => return None,
    })
}

/// The SVG compositing names of the OpenRaster specification.
const SVG_OPS: [(&str, BlendMode); 15] = [
    ("svg:src-over", BlendMode::Normal),
    ("svg:multiply", BlendMode::Multiply),
    ("svg:screen", BlendMode::Screen),
    ("svg:overlay", BlendMode::Overlay),
    ("svg:darken", BlendMode::Darken),
    ("svg:lighten", BlendMode::Lighten),
    ("svg:color-dodge", BlendMode::ColorDodge),
    ("svg:color-burn", BlendMode::ColorBurn),
    ("svg:hard-light", BlendMode::HardLight),
    ("svg:soft-light", BlendMode::SoftLight),
    ("svg:difference", BlendMode::Difference),
    ("svg:color", BlendMode::Color),
    ("svg:luminosity", BlendMode::Luminosity),
    ("svg:hue", BlendMode::Hue),
    ("svg:saturation", BlendMode::Saturation),
];

/// A composite-op attribute as a blend mode; unknown names fall back to Normal with a warning.
fn composite_op(op: &str, layer: &str, warnings: &mut Warnings) -> BlendMode {
    if let Some((_, mode)) = SVG_OPS.iter().find(|(name, _)| *name == op) {
        if op == "svg:soft-light" {
            warnings.note("svg:soft-light (the W3C curve) opens as Photoshop's Soft Light, which differs slightly");
        }
        return *mode;
    }
    match op {
        "svg:plus" => return BlendMode::LinearDodge,
        // Not in the specification's list, but written by several applications.
        "svg:exclusion" => return BlendMode::Exclusion,
        _ => {}
    }
    if let Some((mode, approx)) = op.strip_prefix("krita:").and_then(krita_blend) {
        if approx {
            warnings.note(&format!("Krita's {op} blend mode is approximated by {}", mode.label()));
        }
        return mode;
    }
    warnings.layer(layer, &format!("composite operation {op} is not supported; it opens as Normal"));
    BlendMode::Normal
}

/// Warnings gathered during import, each distinct message once; per-layer ones are grouped.
#[derive(Default)]
pub(crate) struct Warnings {
    general: Vec<String>,
    per_layer: Vec<(String, Vec<String>)>,
}

impl Warnings {
    pub(crate) fn note(&mut self, message: &str) {
        if !self.general.iter().any(|m| m == message) {
            self.general.push(message.to_string());
        }
    }
    pub(crate) fn layer(&mut self, layer: &str, message: &str) {
        match self.per_layer.iter_mut().find(|(m, _)| m == message) {
            Some((_, layers)) => {
                if !layers.iter().any(|l| l == layer) {
                    layers.push(layer.to_string());
                }
            }
            None => self.per_layer.push((message.to_string(), vec![layer.to_string()])),
        }
    }
    pub(crate) fn finish(self) -> Vec<String> {
        let mut out = self.general;
        for (message, layers) in self.per_layer {
            let shown: Vec<String> = layers.iter().take(5).map(|l| format!("\"{l}\"")).collect();
            let more = layers.len().saturating_sub(5);
            let tail = if more > 0 { format!(" and {more} more") } else { String::new() };
            out.push(format!("{} ({}{tail})", message, shown.join(", ")));
        }
        out
    }
}

/// A number attribute; a missing one is `default`, an unparsable one an error.
fn num(node: roxmltree::Node<'_, '_>, name: &str, default: f64) -> Result<f64> {
    match node.attribute(name) {
        None => Ok(default),
        Some(v) => v.trim().parse::<f64>().ok().filter(|v| v.is_finite()).ok_or_else(|| invalid(format!("invalid {name} attribute `{v}`"))),
    }
}

fn int(node: roxmltree::Node<'_, '_>, name: &str) -> Result<i32> {
    let v = num(node, name, 0.0)?.round();
    if v.abs() > f64::from(MAX_SIDE) * 4.0 {
        return Err(invalid(format!("{name} offset {v} is out of range")));
    }
    Ok(v as i32)
}

fn flag(node: roxmltree::Node<'_, '_>, name: &str) -> bool {
    node.attribute(name).is_some_and(|v| matches!(v.trim(), "true" | "1"))
}

/// Shared node properties: name, visibility, opacity and composite operation.
fn props(node: roxmltree::Node<'_, '_>, layer: &mut Layer, index: usize, warnings: &mut Warnings) -> Result<()> {
    layer.name = node.attribute("name").filter(|name| !name.trim().is_empty()).map_or_else(|| format!("Layer {index}"), str::to_string);
    layer.visible = node.attribute("visibility").is_none_or(|v| v.trim() != "hidden");
    layer.opacity = num(node, "opacity", 1.0)?.clamp(0.0, 1.0) as f32;
    layer.blend = composite_op(node.attribute("composite-op").map_or("svg:src-over", str::trim), &layer.name, warnings);
    if flag(node, "edit-locked") {
        layer.locks.all = true;
    }
    if flag(node, "alpha-preserve") {
        layer.locks.transparency = true;
    }
    Ok(())
}

struct Reader<'a, 'c> {
    zip: ZipReader<'a>,
    ctl: &'c Interrupt<'c>,
    nodes: usize,
    budget: usize,
    warnings: Warnings,
    /// Decoded layer PNGs, converted to the document depth once every layer is known.
    images: Vec<Image>,
}

/// A layer as read, before its pixels are converted to the document's format.
enum Node {
    Raster { layer: Layer, image: Option<usize>, x: i32, y: i32 },
    Group { layer: Layer, children: Vec<Node> },
}

impl Reader<'_, '_> {
    fn png(&mut self, src: &str, layer: &str) -> Result<Option<usize>> {
        let Some(entry) = self.zip.find(src).cloned() else {
            self.warnings.layer(layer, "the layer's image is missing from the archive; the layer opens empty");
            return Ok(None);
        };
        let data = self.zip.read(&entry, MAX_ENTRY).map_err(|e| invalid(format!("{src}: {e}")))?;
        let mut opts = codecs::DecodeOptions::default();
        opts.limits.max_width = MAX_SIDE;
        opts.limits.max_height = MAX_SIDE;
        opts.limits.max_alloc = self.budget as u64;
        let image = codecs::decode_with(&data, &opts).map_err(|e| invalid(format!("{src}: {e}")))?;
        let (w, h) = image.dimensions();
        let rgba16 = (w as usize).checked_mul(h as usize).and_then(|n| n.checked_mul(8)).ok_or_else(|| invalid("layer image size overflow"))?;
        self.budget = self.budget.checked_sub(rgba16).ok_or_else(|| invalid("decoded layers exceed the 1 GiB limit"))?;
        if !matches!(codecs::detect(&data), Some(Format::Png)) {
            self.warnings.note("some layer images are not PNG, as OpenRaster expects; they were read anyway");
        }
        self.images.push(image);
        Ok(Some(self.images.len() - 1))
    }

    fn stack(&mut self, stack: roxmltree::Node<'_, '_>, depth: usize) -> Result<Vec<Node>> {
        if depth > photocraft_doc::MAX_GROUP_DEPTH {
            return Err(IoError::Unsupported(format!("layer groups nested deeper than {}", photocraft_doc::MAX_GROUP_DEPTH)));
        }
        let mut out = Vec::new();
        for child in stack.children().filter(roxmltree::Node::is_element) {
            self.ctl.check().map_err(|_| IoError::Cancelled)?;
            self.nodes += 1;
            if self.nodes > MAX_NODES {
                return Err(invalid(format!("more than {MAX_NODES} layers")));
            }
            let index = self.nodes;
            match child.tag_name().name() {
                "layer" => {
                    let mut layer = Layer::raster("", PixelFormat::RGBA8);
                    props(child, &mut layer, index, &mut self.warnings)?;
                    let (x, y) = (int(child, "x")?, int(child, "y")?);
                    let image = match child.attribute("src") {
                        Some(src) => self.png(src.trim(), &layer.name.clone())?,
                        None => {
                            self.warnings.layer(&layer.name, "the layer has no image; it opens empty");
                            None
                        }
                    };
                    out.push(Node::Raster { layer, image, x, y });
                }
                "stack" => {
                    let mut layer = Layer::group("", Vec::new());
                    props(child, &mut layer, index, &mut self.warnings)?;
                    // `auto` (the default) isolates only when something needs it: an
                    // OpenRaster group without its own blend mode is a pass-through group.
                    let isolate = child.attribute("isolation").map(str::trim) == Some("isolate");
                    if !isolate && layer.blend == BlendMode::Normal {
                        layer.blend = BlendMode::PassThrough;
                    }
                    let children = self.stack(child, depth + 1)?;
                    out.push(Node::Group { layer, children });
                }
                other => {
                    let name = child.attribute("name").unwrap_or(other).to_string();
                    self.warnings.layer(&name, &format!("OpenRaster <{other}> elements are not supported and were skipped"));
                }
            }
        }
        Ok(out)
    }
}

/// Sample depth of the document: the deepest layer image (8 or 16 bit; float stays float).
fn document_depth(images: &[Image]) -> SampleType {
    images.iter().fold(SampleType::U8, |d, i| match (d, i.sample_type()) {
        (SampleType::F32, _) | (_, CSample::F32 | CSample::F16) => SampleType::F32,
        (SampleType::U16, _) | (_, CSample::U16) => SampleType::U16,
        _ => SampleType::U8,
    })
}

pub(crate) fn csample(depth: SampleType) -> CSample {
    crate::flat::csample(depth)
}

/// Builds the document tree bottom-to-top (stack.xml lists the topmost node first).
fn build(nodes: Vec<Node>, images: &[Image], fmt: PixelFormat, ctl: &Interrupt<'_>) -> Result<Vec<Layer>> {
    let mut out = Vec::with_capacity(nodes.len());
    for node in nodes.into_iter().rev() {
        ctl.check().map_err(|_| IoError::Cancelled)?;
        match node {
            Node::Raster { mut layer, image, x, y } => {
                let mut surface = Surface::new(fmt);
                if let Some(image) = image.and_then(|i| images.get(i)) {
                    let (w, h) = image.dimensions();
                    let px = image.convert(ChannelLayout::Rgba, csample(fmt.sample));
                    let rect = Rect::new(x, y, x.saturating_add(w as i32), y.saturating_add(h as i32));
                    if rect.width() != w || rect.height() != h || px.data().len() != (w as usize) * (h as usize) * fmt.bytes_per_pixel() {
                        return Err(invalid("layer image does not fit its position"));
                    }
                    surface.write_interleaved(rect, px.data());
                    surface.prune();
                }
                layer.content = LayerContent::Raster(surface);
                out.push(layer);
            }
            Node::Group { mut layer, children } => {
                let children = build(children, images, fmt, ctl)?;
                if let Some(c) = layer.children_mut() {
                    *c = children;
                }
                out.push(layer);
            }
        }
    }
    Ok(out)
}

pub(crate) fn import(name: &str, bytes: &[u8], ctl: &Interrupt<'_>) -> Result<ImportResult> {
    let zip = ZipReader::new(bytes).map_err(|e| invalid(e.to_string()))?;
    let mime = zip.read_by_name("mimetype", 256).map_err(|e| invalid(e.to_string()))?;
    if mime.trim_ascii() != MIMETYPE {
        return Err(invalid("the archive's mimetype is not image/openraster"));
    }
    let xml = zip.read_by_name("stack.xml", MAX_STACK_XML).map_err(|e| invalid(e.to_string()))?;
    let xml = std::str::from_utf8(&xml).map_err(|_| invalid("stack.xml is not UTF-8"))?;
    let tree = roxmltree::Document::parse(xml).map_err(|e| invalid(format!("stack.xml: {e}")))?;
    let image = tree.root_element();
    if image.tag_name().name() != "image" {
        return Err(invalid("stack.xml has no <image> root"));
    }
    let side = |attr: &str| -> Result<u32> {
        let v = num(image, attr, 0.0)?;
        if !(1.0..=f64::from(MAX_SIDE)).contains(&v) || v.fract() != 0.0 {
            return Err(invalid(format!("image {attr} {v} must be a whole number of pixels from 1 to {MAX_SIDE}")));
        }
        Ok(v as u32)
    };
    let (w, h) = (side("w")?, side("h")?);
    if (w as usize).saturating_mul(h as usize).saturating_mul(8) > MAX_PIXEL_BYTES {
        return Err(invalid("the canvas exceeds the 1 GiB pixel limit"));
    }
    let root = image.children().find(|n| n.has_tag_name("stack")).ok_or_else(|| invalid("stack.xml has no root <stack>"))?;
    let mut reader = Reader { zip, ctl, nodes: 0, budget: MAX_PIXEL_BYTES, warnings: Warnings::default(), images: Vec::new() };
    let nodes = reader.stack(root, 1)?;
    ctl.progress(0.8);
    let depth = document_depth(&reader.images);
    if depth == SampleType::F32 {
        reader.warnings.note("float layer images open as a 32-bit float document");
    }
    let fmt = PixelFormat::new(ColorMode::Rgb, depth, true);
    let mut document = Document::new(name, Size::new(w, h), ColorMode::Rgb, depth);
    document.layers = build(nodes, &reader.images, fmt, ctl)?;
    let xres = num(image, "xres", 72.0)?;
    let yres = num(image, "yres", xres)?;
    if xres > 0.0 && xres < 1.0e6 {
        document.resolution_dpi = xres as f32;
        reader.warnings.general.extend(crate::unequal_resolution_warning(xres, yres));
    }
    if document.layers.is_empty() {
        reader.warnings.note("the OpenRaster file has no layers");
    }
    Ok(ImportResult { document, warnings: reader.warnings.finish(), source_read_only: false, preview_only: false })
}

// ---------------------------------------------------------------------------------------------
// Export

/// Attribute text with XML's five special characters escaped (and control characters dropped).
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' => out.push_str("&#9;"),
            '\n' => out.push_str("&#10;"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// The composite-op attribute for `mode`: the SVG name where the specification has one, else
/// Krita's (`krita:<id>`, which Krita reads back and others treat as Normal).
fn op_name(mode: BlendMode) -> Option<String> {
    match mode {
        BlendMode::PassThrough => Some("svg:src-over".into()),
        BlendMode::LinearDodge => Some("svg:plus".into()),
        m => SVG_OPS.iter().find(|(_, s)| *s == m).map(|(n, _)| (*n).to_string()).or_else(|| krita_id(m).map(|id| format!("krita:{id}"))),
    }
}

struct Writer<'a> {
    doc: &'a Document,
    zip: ZipWriter,
    xml: String,
    next: usize,
    sample: CSample,
    warnings: Warnings,
    /// Only collect the warnings ([`losses`]): no pixels are rendered or encoded.
    dry: bool,
}

/// Straight RGBA floats as interleaved PNG samples at `sample` depth.
fn encode_rgba(px: &[[f32; 4]], sample: CSample) -> Vec<u8> {
    match sample {
        CSample::U16 => px.iter().flat_map(|p| p.map(|v| ((v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16).to_ne_bytes())).flatten().collect(),
        _ => px.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)).collect(),
    }
}

fn png(w: u32, h: u32, px: &[[f32; 4]], sample: CSample) -> Result<Vec<u8>> {
    let image = Image::from_raw(w, h, ChannelLayout::Rgba, sample, encode_rgba(px, sample))?;
    let opts = codecs::EncodeOptions { embed_icc: false, ..Default::default() };
    Ok(codecs::encode(&image, Format::Png, &opts)?)
}

/// `px` (row-major over `rect`) cut down to the pixels that aren't fully transparent, with
/// transparent pixels zeroed, so masked-out or empty margins aren't written.
fn crop(rect: Rect, mut px: Vec<[f32; 4]>) -> (Rect, Vec<[f32; 4]>) {
    let w = rect.width() as usize;
    if w == 0 {
        return (Rect::EMPTY, Vec::new());
    }
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for (i, p) in px.iter_mut().enumerate() {
        if p[3] <= 0.0 {
            *p = [0.0; 4];
            continue;
        }
        let (x, y) = (i % w, i / w);
        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
    }
    if x0 >= x1 {
        return (Rect::EMPTY, Vec::new());
    }
    if (x0, y0, x1) == (0, 0, w) && y1 == px.len() / w {
        return (rect, px);
    }
    let out = (y0..y1).flat_map(|y| px.get(y * w + x0..y * w + x1).unwrap_or_default().iter().copied()).collect();
    (Rect::new(rect.x0 + x0 as i32, rect.y0 + y0 as i32, rect.x0 + x1 as i32, rect.y0 + y1 as i32), out)
}

impl Writer<'_> {
    fn common(&mut self, layer: &Layer) -> String {
        let mut attrs = format!(
            "name=\"{}\" visibility=\"{}\" opacity=\"{}\"",
            escape(&layer.name),
            if layer.visible { "visible" } else { "hidden" },
            // OpenRaster has only one opacity for groups. For ordinary isolated groups,
            // fold fill opacity into that attribute; groups have no pixel payload in which
            // to bake it as raster layers do (#3048).
            if matches!(&layer.content, LayerContent::Group(_)) { (layer.opacity * layer.fill_opacity).clamp(0.0, 1.0) } else { layer.opacity.clamp(0.0, 1.0) }
        );
        let op = op_name(layer.blend).unwrap_or_else(|| {
            self.warnings.layer(&layer.name, &format!("the {} blend mode has no OpenRaster equivalent; it was saved as Normal", layer.blend.label()));
            "svg:src-over".into()
        });
        attrs.push_str(&format!(" composite-op=\"{op}\""));
        if layer.locks.all || layer.locks.pixels {
            attrs.push_str(" edit-locked=\"true\"");
        }
        if layer.locks.transparency {
            attrs.push_str(" alpha-preserve=\"true\"");
        }
        if layer.clipped {
            self.warnings.layer(&layer.name, "OpenRaster has no clipping masks; these layers were saved unclipped");
        }
        if !layer.effects.items.is_empty() || layer.effects.psd_raw.is_some() {
            self.warnings.layer(&layer.name, "layer styles are not saved to OpenRaster");
        }
        if layer.vector_mask.as_ref().is_some_and(|m| m.enabled) {
            self.warnings.layer(&layer.name, "vector masks are not saved to OpenRaster");
        }
        if layer.fill_opacity < 1.0 {
            if matches!(&layer.content, LayerContent::Group(_)) {
                self.warnings.layer(&layer.name, "fill opacity was combined into the layer opacity");
            } else {
                self.warnings.layer(&layer.name, "fill opacity was applied to the saved layer pixels");
            }
        }
        attrs
    }

    /// What saving a non-group layer's pixels loses: a type, shape, fill or smart object layer
    /// becomes pixels, and its pixel mask is applied to them.
    fn content_warnings(&mut self, layer: &Layer) {
        if !matches!(layer.content, LayerContent::Raster(_)) {
            self.warnings.layer(&layer.name, &format!("{} layers are saved to OpenRaster as pixels", layer.content.kind_name().to_lowercase()));
        }
        if let Some(mask) = layer.mask.as_ref().filter(|m| m.enabled) {
            if mask.feather > 0.0 {
                self.warnings.layer(&layer.name, "layer mask feathering is not applied in OpenRaster");
            }
            self.warnings.layer(&layer.name, "layer masks were applied to the layer pixels");
        }
    }

    /// The layer's own pixels (straight RGBA floats) and their bounds; its pixel mask applied.
    fn pixels(&mut self, layer: &Layer) -> (Rect, Vec<[f32; 4]>) {
        let canvas = self.doc.bounds();
        let (rect, mut buf) = match &layer.content {
            LayerContent::Raster(s) => {
                let r = s.content_bounds();
                (r, photocraft_compose::surface_to_buffer(s, r).px)
            }
            _ => {
                // Rendered as it shows on its own, before the layer's blending options.
                let mut solo = layer.clone();
                solo.visible = true;
                solo.opacity = 1.0;
                solo.fill_opacity = 1.0;
                solo.blend = BlendMode::Normal;
                solo.clipped = false;
                solo.mask = None;
                solo.effects = Default::default();
                let r = photocraft_compose::composite_bounds(&solo, canvas).unwrap_or(canvas).intersect(&canvas);
                (r, photocraft_compose::render_layer(&solo, r).px)
            }
        };
        if let Some(mask) = layer.mask.as_ref().filter(|m| m.enabled) {
            let mut values = Vec::new();
            mask.values_into(rect, &mut values);
            for (p, m) in buf.iter_mut().zip(values) {
                p[3] *= m;
            }
        }
        if layer.fill_opacity < 1.0 {
            for p in &mut buf {
                p[3] *= layer.fill_opacity.clamp(0.0, 1.0);
            }
        }
        crop(rect, buf)
    }

    fn layers(&mut self, layers: &[Layer], depth: usize) -> Result<()> {
        if depth > photocraft_doc::MAX_GROUP_DEPTH {
            return Err(IoError::Unsupported(format!("layer groups nested deeper than {}", photocraft_doc::MAX_GROUP_DEPTH)));
        }
        // stack.xml lists the topmost layer first.
        for layer in layers.iter().rev() {
            match &layer.content {
                LayerContent::Group(g) => {
                    let attrs = self.common(layer);
                    let isolation = if layer.blend == BlendMode::PassThrough { "auto" } else { "isolate" };
                    if layer.mask.as_ref().is_some_and(|m| m.enabled) {
                        self.warnings.layer(&layer.name, "group masks are not saved to OpenRaster");
                    }
                    self.xml.push_str(&format!("<stack {attrs} isolation=\"{isolation}\">"));
                    self.layers(&g.children, depth + 1)?;
                    self.xml.push_str("</stack>");
                }
                LayerContent::Adjustment(_) => {
                    self.warnings.layer(&layer.name, "adjustment layers cannot be saved to OpenRaster and were left out (the merged image keeps their look)");
                }
                _ => {
                    let attrs = self.common(layer);
                    self.content_warnings(layer);
                    if self.dry {
                        continue;
                    }
                    let (rect, px) = self.pixels(layer);
                    let (rect, px) = if rect.is_empty() { (Rect::from_xywh(0, 0, 1, 1), vec![[0.0; 4]]) } else { (rect, px) };
                    let src = format!("data/layer{}.png", self.next);
                    self.next += 1;
                    let data = png(rect.width(), rect.height(), &px, self.sample)?;
                    self.zip.add(&src, &data)?;
                    self.xml.push_str(&format!("<layer {attrs} src=\"{src}\" x=\"{}\" y=\"{}\"/>", rect.x0, rect.y0));
                }
            }
        }
        Ok(())
    }
}

pub(crate) fn export(doc: &Document) -> Result<ExportResult> {
    if !matches!(doc.mode, ColorMode::Rgb | ColorMode::Grayscale) {
        return Err(IoError::Unsupported("OpenRaster holds RGB layers; convert the document to RGB or Grayscale first".into()));
    }
    let sample = match doc.depth {
        SampleType::U8 => CSample::U8,
        _ => CSample::U16,
    };
    let mut zip = ZipWriter::new();
    // The mimetype must be the first entry, stored, so the type is readable at a fixed offset.
    zip.add("mimetype", MIMETYPE)?;
    let mut w = Writer { doc, zip, xml: String::new(), next: 0, sample, warnings: Warnings::default(), dry: false };
    document_warnings(doc, &mut w.warnings);
    w.xml.push_str(&format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<image version=\"0.0.6\" w=\"{}\" h=\"{}\" xres=\"{}\" yres=\"{}\"><stack>",
        doc.size.width,
        doc.size.height,
        doc.resolution_dpi.round().max(1.0),
        doc.resolution_dpi.round().max(1.0)
    ));
    w.layers(&doc.layers, 1)?;
    w.xml.push_str("</stack></image>\n");
    let Writer { mut zip, xml, warnings, .. } = w;
    zip.add("stack.xml", xml.as_bytes())?;
    let merged = photocraft_compose::flatten(doc);
    zip.add("mergedimage.png", &png(doc.size.width, doc.size.height, &merged.px, sample)?)?;
    let thumb = photocraft_compose::thumbnail(doc, THUMBNAIL_SIDE);
    let thumb = Image::from_raw(thumb.width, thumb.height, ChannelLayout::Rgba, CSample::U8, thumb.pixels)?;
    zip.add("Thumbnails/thumbnail.png", &codecs::encode(&thumb, Format::Png, &codecs::EncodeOptions::default())?)?;
    Ok(ExportResult { bytes: zip.finish()?, warnings: warnings.finish() })
}

/// What saving `doc` to OpenRaster loses that isn't in a layer.
fn document_warnings(doc: &Document, warnings: &mut Warnings) {
    if doc.depth == SampleType::F32 {
        warnings.note("32-bit float layers are saved as 16-bit PNGs");
    }
    if !doc.channels.is_empty() {
        warnings.note("alpha channels are not saved to OpenRaster");
    }
    if !doc.metadata.text.is_empty() || doc.metadata.xmp.is_some() || doc.metadata.exif.is_some() {
        warnings.note("metadata (XMP, EXIF, text) is not saved to OpenRaster");
    }
    // Readers assume sRGB, so an sRGB document loses nothing by the missing tag.
    let srgb = |bytes: &[u8]| photocraft_cms::Profile::parse(bytes).is_ok_and(|p| p.same_colors(photocraft_cms::Builtin::Srgb.profile()));
    if doc.icc_profile.as_deref().is_some_and(|b| !srgb(b)) {
        warnings.note("the colour profile is not saved to OpenRaster; readers assume sRGB");
    }
    if doc.mode == ColorMode::Grayscale {
        warnings.note("grayscale layers are saved as RGB");
    }
}

/// Everything saving `doc` to OpenRaster would lose (the warnings [`export`] gives), without
/// rendering or encoding anything. Empty when an OpenRaster file holds the whole document, so it
/// can be the document's own file; otherwise a save to OpenRaster is a copy (#3806).
pub fn losses(doc: &Document) -> Vec<String> {
    if !matches!(doc.mode, ColorMode::Rgb | ColorMode::Grayscale) {
        return vec!["OpenRaster holds RGB layers; convert the document to RGB or Grayscale first".into()];
    }
    let mut w = Writer { doc, zip: ZipWriter::new(), xml: String::new(), next: 0, sample: CSample::U8, warnings: Warnings::default(), dry: true };
    document_warnings(doc, &mut w.warnings);
    if let Err(e) = w.layers(&doc.layers, 1) {
        w.warnings.note(&e.to_string());
    }
    w.warnings.finish()
}
