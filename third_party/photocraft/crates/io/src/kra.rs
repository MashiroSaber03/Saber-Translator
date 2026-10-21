//! Krita (`.kra`, `.krz`) import. Layout observed from files Krita writes (no Krita code used):
//! a ZIP whose first, stored entry is `mimetype` = `application/x-krita`, a `maindoc.xml` node
//! tree, one tiled pixel file per paint layer and mask under `<image name>/layers/`, the
//! image's ICC profile in `<image name>/annotations/icc`, and Krita's rendering in
//! `mergedimage.png`.
//!
//! Tiled pixel files are a short text header (`VERSION 2`, `TILEWIDTH`, `TILEHEIGHT`,
//! `PIXELSIZE`, `DATA <tiles>`) followed by, per tile, a `x,y,LZF,<bytes>` line and that many
//! bytes: a flag (1 = LZF-compressed, 0 = raw) and the tile's pixels stored as byte planes
//! (every pixel's first byte, then every pixel's second byte, ...). RGB pixels are in
//! Krita's BGRA order, 16-bit samples little-endian.
//!
//! Imported: paint layers (8 and 16-bit RGBA and gray), groups (isolated and pass-through),
//! names, visibility, opacity, blend modes, locks, layer offsets and transparency masks.
//! Every other node kind (vector, filter, fill, file and clone layers, filter, transform and
//! colorize masks) is left out with a warning; Krita's merged image still shows their look.
//! Documents in a colour space we don't read open as Krita's merged image, flagged as a preview.

#[cfg(test)]
mod tests;
mod tiles;

use photocraft_color::{BlendMode, PixelFormat};
use photocraft_doc::{Document, Layer, LayerContent, LayerMask};
use photocraft_format::zip::ZipReader;
use photocraft_geom::Size;
use photocraft_raster::{Interrupt, Surface};

use crate::ora::{MAX_NODES, MAX_PIXEL_BYTES, MAX_SIDE, Warnings, krita_blend};
use crate::{ImportResult, IoError};
use tiles::{Space, TileFile};

type Result<T> = std::result::Result<T, IoError>;

/// The `mimetype` entry of a Krita archive.
pub(crate) const MIMETYPE: &[u8] = b"application/x-krita";
pub(crate) const EXTENSIONS: [&str; 2] = ["kra", "krz"];
const MAX_MAINDOC: usize = 32 << 20;
const MAX_LAYER_FILE: usize = 1 << 30;

fn invalid(message: impl Into<String>) -> IoError {
    IoError::Kra(message.into())
}

/// `true` for a Krita archive (by its `mimetype` entry).
pub(crate) fn is_kra(bytes: &[u8]) -> bool {
    crate::ora::zip_mimetype(bytes).is_some_and(|mime| mime.trim_ascii() == MIMETYPE)
}

fn attr<'a>(node: roxmltree::Node<'a, '_>, name: &str) -> Option<&'a str> {
    node.attribute(name).map(str::trim)
}

fn number(node: roxmltree::Node<'_, '_>, name: &str, default: f64) -> Result<f64> {
    match attr(node, name) {
        None | Some("") => Ok(default),
        Some(v) => v.parse::<f64>().ok().filter(|v| v.is_finite()).ok_or_else(|| invalid(format!("invalid {name} attribute `{v}`"))),
    }
}

fn offset(node: roxmltree::Node<'_, '_>, name: &str) -> Result<i32> {
    let v = number(node, name, 0.0)?.round();
    if v.abs() > f64::from(MAX_SIDE) * 4.0 {
        return Err(invalid(format!("{name} offset {v} is out of range")));
    }
    Ok(v as i32)
}

fn truthy(node: roxmltree::Node<'_, '_>, name: &str) -> bool {
    matches!(attr(node, name), Some("1" | "true"))
}

/// Child elements named `tag` inside the node's `<container>` element (`layers` or `masks`).
fn children<'a, 'i>(node: roxmltree::Node<'a, 'i>, container: &str, tag: &'static str) -> impl Iterator<Item = roxmltree::Node<'a, 'i>> {
    node.children().filter(move |c| c.has_tag_name(container)).flat_map(move |c| c.children().filter(move |n| n.has_tag_name(tag)))
}

struct Reader<'a, 'c> {
    zip: ZipReader<'a>,
    ctl: &'c Interrupt<'c>,
    /// `<image name>/layers/`.
    dir: String,
    /// The document's pixel format (layers in another supported space are converted to it).
    fmt: PixelFormat,
    nodes: usize,
    budget: usize,
    warnings: Warnings,
}

impl Reader<'_, '_> {
    fn file(&self, name: &str) -> Result<Option<Vec<u8>>> {
        let path = format!("{}{name}", self.dir);
        match self.zip.find(&path) {
            None => Ok(None),
            Some(e) => self.zip.read(e, MAX_LAYER_FILE).map(Some).map_err(|err| invalid(format!("{path}: {err}"))),
        }
    }

    /// The default pixel Krita stores next to a tiled file, decoded to `space` samples.
    fn default_pixel(&self, name: &str, space: Space) -> Result<Option<Vec<f32>>> {
        let Some(bytes) = self.file(&format!("{name}.defaultpixel"))? else { return Ok(None) };
        if bytes.len() != space.pixel_size() {
            return Err(invalid(format!("{name}.defaultpixel has {} bytes, expected {}", bytes.len(), space.pixel_size())));
        }
        Ok(Some(space.normalised(&bytes)))
    }

    /// A tiled pixel file in `space` as a surface in `fmt`, offset by (`dx`, `dy`).
    fn surface(&mut self, name: &str, space: Space, fmt: PixelFormat, dx: i32, dy: i32) -> Result<Option<Surface>> {
        let Some(bytes) = self.file(name)? else { return Ok(None) };
        let file = TileFile::parse(&bytes)?;
        if file.pixel_size != space.pixel_size() {
            return Err(invalid(format!("{name}: pixel size {} does not match its colour space", file.pixel_size)));
        }
        let cost = file.decoded_bytes(fmt.bytes_per_pixel())?;
        self.budget = self.budget.checked_sub(cost).ok_or_else(|| invalid("decoded layers exceed the 1 GiB limit"))?;
        let mut surface = match self.default_pixel(name, space)? {
            Some(px) if px.iter().any(|v| *v != 0.0) => Surface::with_default(fmt, &space.to_format(&px, fmt)),
            _ => Surface::new(fmt),
        };
        file.decode_into(&mut surface, space, dx, dy, self.ctl)?;
        surface.prune();
        Ok(Some(surface))
    }

    fn props(&mut self, node: roxmltree::Node<'_, '_>, layer: &mut Layer) -> Result<()> {
        layer.name = attr(node, "name").filter(|n| !n.is_empty()).map_or_else(|| format!("Layer {}", self.nodes), str::to_string);
        layer.visible = attr(node, "visible") != Some("0");
        layer.opacity = (number(node, "opacity", 255.0)? / 255.0).clamp(0.0, 1.0) as f32;
        if truthy(node, "locked") {
            layer.locks.all = true;
        }
        let op = attr(node, "compositeop").unwrap_or("normal");
        layer.blend = match krita_blend(op) {
            Some((mode, approx)) => {
                if approx {
                    self.warnings.note(&format!("Krita's {op} blend mode is approximated by {}", mode.label()));
                }
                mode
            }
            None => {
                self.warnings.layer(&layer.name, &format!("Krita's {op} blend mode is not supported; it opens as Normal"));
                BlendMode::Normal
            }
        };
        if attr(node, "channelflags").is_some_and(|f| f.contains('0')) {
            self.warnings.layer(&layer.name, "channel locks (Krita's per-channel blending) are ignored");
        }
        if truthy(node, "inheritalpha") {
            self.warnings.layer(&layer.name, "Inherit Alpha is not supported; these layers blend normally");
        }
        if attr(node, "keyframes").is_some_and(|k| !k.is_empty()) {
            self.warnings.layer(&layer.name, "animated layers open with the frame Krita saved in the layer");
        }
        Ok(())
    }

    /// The node's transparency masks, combined into one layer mask.
    fn masks(&mut self, node: roxmltree::Node<'_, '_>, layer: &mut Layer, dx: i32, dy: i32) -> Result<()> {
        for mask in children(node, "masks", "mask") {
            self.ctl.check().map_err(|_| IoError::Cancelled)?;
            let kind = attr(mask, "nodetype").unwrap_or("");
            match kind {
                "transparencymask" => {}
                // A layer's local selection: not part of its look.
                "selectionmask" => continue,
                other => {
                    let what = match other {
                        "filtermask" => "filter masks",
                        "transformmask" => "transform masks",
                        "colorizemask" => "colorize masks",
                        _ => "unknown masks",
                    };
                    self.warnings.layer(&layer.name, &format!("{what} are not supported and were left out"));
                    continue;
                }
            }
            if layer.mask.is_some() {
                self.warnings.layer(&layer.name, "only the first transparency mask of a layer is kept");
                continue;
            }
            let Some(file) = attr(mask, "filename") else { continue };
            let (mx, my) = (dx.saturating_add(offset(mask, "x")?), dy.saturating_add(offset(mask, "y")?));
            match self.surface(&format!("{file}.pixelselection"), Space::Alpha8, PixelFormat::GRAY8, mx, my)? {
                Some(surface) => {
                    layer.mask = Some(LayerMask { surface, enabled: attr(mask, "visible") != Some("0"), ..LayerMask::reveal_all() });
                }
                None => self.warnings.layer(&layer.name, "a transparency mask's pixels are missing from the file"),
            }
        }
        Ok(())
    }

    /// The `<layers>` of `node`, bottom-to-top (maindoc.xml lists the topmost first).
    fn layers(&mut self, node: roxmltree::Node<'_, '_>, depth: usize, dx: i32, dy: i32) -> Result<Vec<Layer>> {
        if depth > photocraft_doc::MAX_GROUP_DEPTH {
            return Err(IoError::Unsupported(format!("layer groups nested deeper than {}", photocraft_doc::MAX_GROUP_DEPTH)));
        }
        let mut out = Vec::new();
        for child in children(node, "layers", "layer") {
            self.ctl.check().map_err(|_| IoError::Cancelled)?;
            self.nodes += 1;
            if self.nodes > MAX_NODES {
                return Err(invalid(format!("more than {MAX_NODES} layers")));
            }
            let (x, y) = (dx.saturating_add(offset(child, "x")?), dy.saturating_add(offset(child, "y")?));
            let kind = attr(child, "nodetype").unwrap_or("");
            let mut layer = match kind {
                "paintlayer" => {
                    let mut layer = Layer::raster("", self.fmt);
                    self.props(child, &mut layer)?;
                    let space = attr(child, "colorspacename").and_then(Space::from_name);
                    match (space, attr(child, "filename")) {
                        (Some(space), Some(file)) => match self.surface(file, space, self.fmt, x, y)? {
                            Some(s) => layer.content = LayerContent::Raster(s),
                            None => self.warnings.layer(&layer.name, "the layer's pixels are missing from the file; it opens empty"),
                        },
                        (None, _) => {
                            let cs = attr(child, "colorspacename").unwrap_or("?");
                            self.warnings.layer(&layer.name, &format!("layers in Krita's {cs} colour space are not supported and were left out"));
                            continue;
                        }
                        (_, None) => self.warnings.layer(&layer.name, "the layer names no pixel file; it opens empty"),
                    }
                    layer
                }
                "grouplayer" => {
                    let mut layer = Layer::group("", Vec::new());
                    self.props(child, &mut layer)?;
                    if truthy(child, "passthrough") {
                        if layer.blend != BlendMode::Normal {
                            self.warnings.layer(&layer.name, "a pass-through group's blend mode is ignored, as in Krita");
                        }
                        layer.blend = BlendMode::PassThrough;
                    }
                    // A group's own x/y is not applied to its children: Krita moves a group by
                    // moving their pixels, and its merged image ignores the group offset.
                    let kids = self.layers(child, depth + 1, dx, dy)?;
                    if let LayerContent::Group(g) = &mut layer.content {
                        g.children = kids;
                        g.expanded = !truthy(child, "collapsed");
                    }
                    layer
                }
                other => {
                    let name = attr(child, "name").unwrap_or(other).to_string();
                    let what = match other {
                        "shapelayer" => "vector layers",
                        "adjustmentlayer" => "filter layers",
                        "generatorlayer" => "fill layers",
                        "filelayer" => "file layers",
                        "clonelayer" => "clone layers",
                        _ => "unknown layer kinds",
                    };
                    self.warnings.layer(&name, &format!("Krita {what} are not supported yet and were left out"));
                    continue;
                }
            };
            self.masks(child, &mut layer, x, y)?;
            out.push(layer);
        }
        out.reverse();
        Ok(out)
    }
}

/// Krita's merged image as a one-layer stand-in, for documents we can't read layer by layer.
fn preview(name: &str, zip: &ZipReader<'_>, why: String) -> Result<ImportResult> {
    let png = zip.read_by_name("mergedimage.png", MAX_LAYER_FILE).map_err(|e| invalid(format!("{why}, and the merged image can't be read: {e}")))?;
    let mut r = crate::flat::import_flat(name, &png)?;
    r.document.name = name.to_string();
    r.warnings.insert(0, format!("{why}; the file opens as Krita's merged image, without its layers"));
    r.source_read_only = true;
    r.preview_only = true;
    Ok(r)
}

pub(crate) fn import(name: &str, bytes: &[u8], ctl: &Interrupt<'_>) -> Result<ImportResult> {
    let zip = ZipReader::new(bytes).map_err(|e| invalid(e.to_string()))?;
    let mime = zip.read_by_name("mimetype", 256).map_err(|e| invalid(e.to_string()))?;
    if mime.trim_ascii() != MIMETYPE {
        return Err(invalid("the archive's mimetype is not application/x-krita"));
    }
    let xml = zip.read_by_name("maindoc.xml", MAX_MAINDOC).map_err(|e| invalid(e.to_string()))?;
    let xml = std::str::from_utf8(&xml).map_err(|_| invalid("maindoc.xml is not UTF-8"))?;
    // maindoc.xml declares Krita's DTD by public id; roxmltree only reads internal subsets and
    // guards entity expansion.
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    let tree = roxmltree::Document::parse_with_options(xml, opts).map_err(|e| invalid(format!("maindoc.xml: {e}")))?;
    let image = tree.root_element().children().find(|n| n.has_tag_name("IMAGE")).ok_or_else(|| invalid("maindoc.xml has no IMAGE"))?;
    let side = |a: &str| -> Result<u32> {
        let v = number(image, a, 0.0)?;
        if !(1.0..=f64::from(MAX_SIDE)).contains(&v) || v.fract() != 0.0 {
            return Err(invalid(format!("image {a} {v} must be a whole number of pixels from 1 to {MAX_SIDE}")));
        }
        Ok(v as u32)
    };
    let (w, h) = (side("width")?, side("height")?);
    if (w as usize).saturating_mul(h as usize).saturating_mul(8) > MAX_PIXEL_BYTES {
        return Err(invalid("the canvas exceeds the 1 GiB pixel limit"));
    }
    let cs = attr(image, "colorspacename").unwrap_or("");
    let Some(space) = Space::from_name(cs) else {
        return preview(name, &zip, format!("Krita's {cs} colour space is not supported yet"));
    };
    let (mode, depth) = space.model();
    let image_name = attr(image, "name").unwrap_or("");
    let mut reader = Reader {
        zip,
        ctl,
        dir: format!("{image_name}/layers/"),
        fmt: PixelFormat::new(mode, depth, true),
        nodes: 0,
        budget: MAX_PIXEL_BYTES,
        warnings: Warnings::default(),
    };
    let layers = reader.layers(image, 1, 0, 0)?;
    ctl.progress(0.9);
    let mut document = Document::new(name, Size::new(w, h), mode, depth);
    document.layers = layers;
    let icc = format!("{image_name}/annotations/icc");
    if let Some(e) = reader.zip.find(&icc) {
        let profile = reader.zip.read(e, 16 << 20).map_err(|err| invalid(format!("{icc}: {err}")))?;
        if !profile.is_empty() {
            document.icc_profile = Some(std::sync::Arc::new(profile));
        }
    }
    let (xres, yres) = (number(image, "x-res", 72.0)?, number(image, "y-res", 72.0)?);
    if xres > 0.0 && xres < 1.0e6 {
        document.resolution_dpi = xres as f32;
        if let Some(w) = crate::unequal_resolution_warning(xres, yres) {
            reader.warnings.note(&w);
        }
    }
    if document.layers.is_empty() {
        reader.warnings.note("no layers of the Krita document could be imported");
    }
    if image.children().any(|n| n.has_tag_name("assistants") || n.has_tag_name("ReferenceImages")) {
        reader.warnings.note("drawing assistants and reference images were left out");
    }
    Ok(ImportResult { document, warnings: reader.warnings.finish(), source_read_only: true, preview_only: false })
}
