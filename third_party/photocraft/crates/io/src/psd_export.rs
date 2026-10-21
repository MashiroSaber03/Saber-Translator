//! [`Document`] → PSD/PSB.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent, LayerMask, SmartContentsId};
use photocraft_psd::file::{GlobalLayerMask, LayerInfoPlacement};
use photocraft_psd::layer::{BlendingRanges, ChannelData, LayerFlags, LayerInfo, LayerMask as PsdMask, MaskData, MaskParameters};
use photocraft_psd::resources::{ImageResource, ResolutionInfo, ids, version_info_resource};
use photocraft_psd::{
    BlendMode as PsdBlend, ColorMode as PsdMode, Compression, Header, ImageData, LayerRecord, PsdFile, Rect as PsdRect, SectionType, TaggedBlock, Version,
};
use photocraft_raster::{Interrupt, Surface};

use crate::adjust_map;
use crate::blocks;
use crate::pixels::{deinterleave, encode_be, psd_depth};
use crate::psd_import::REGENERATED;

/// Blocks that define a layer's content as something other than pixels: fill (`SoCo`, `GdFl`,
/// `PtFl`), placed / smart object (`SoLd`, `PlLd`, `SoLE`), type (`TySh`) and shape fill (`vscg`).
/// A raster layer never writes them, nor adjustment blocks.
const RASTER_STALE_KEYS: [&[u8; 4]; 8] = [b"SoCo", b"GdFl", b"PtFl", b"SoLd", b"PlLd", b"SoLE", b"TySh", b"vscg"];

const PSD_MAX_FILE_SIZE: u64 = 2 * 1024 * 1024 * 1024;
const PSD_ESTIMATE_FIXED_OVERHEAD: u64 = 1024 * 1024;
const PSD_ESTIMATE_LAYER_OVERHEAD: u64 = 64 * 1024;

/// Options for [`document_to_psd_with`].
#[derive(Debug, Clone)]
pub struct PsdExportOptions {
    /// Write PSB even when the document fits PSD limits.
    pub force_psb: bool,
    /// Matte the merged image against white where it is translucent, as Photoshop does
    /// (`false` keeps straight colour under the alpha: for containers that store the composite
    /// with its own alpha, such as a layered TIFF).
    pub merged_matte: bool,
    /// Compression of the merged image (32-bit float documents are always written raw).
    pub merged_compression: Compression,
    /// Write the real merged image (Preferences › File Handling › Maximize PSD and PSB File
    /// Compatibility). `false` writes Photoshop's white placeholder instead and records
    /// `has_real_merged_data = false` in resource 1057: a smaller file and a faster save, but
    /// readers that only show the merged image see white. A document without layer records always
    /// gets the real image, since its pixels live nowhere else.
    pub maximize_compatibility: bool,
}

impl Default for PsdExportOptions {
    fn default() -> Self {
        PsdExportOptions { force_psb: false, merged_matte: true, merged_compression: Compression::Rle, maximize_compatibility: true }
    }
}

struct Ex {
    fmt: PixelFormat,
    /// Document resolution (for generated type-layer data).
    dpi: f32,
    /// Character/paragraph styles written into every type layer's engine data.
    text_styles: photocraft_doc::TextStyles,
    mask_fmt: PixelFormat,
    cc: usize,
    cmyk: bool,
    version: Version,
    next_id: u32,
    /// Reserved imported layer and generated divider IDs.
    used_ids: std::collections::HashSet<u32>,
    /// PSD ids assigned to document layers (kept from `psd_id` when unique).
    layer_ids: std::collections::HashMap<photocraft_doc::LayerId, u32>,
    /// The generated `Txt2` object number of each type layer (a preserved `TySh` keeps its own
    /// `TextIndex`, others get the next free number; see `photocraft_text::psd::build_txt2`),
    /// written into the `TySh` data this export re-serializes.
    text_index: std::collections::HashMap<photocraft_doc::LayerId, i32>,
    canvas: photocraft_geom::Rect,
    records: Vec<LayerRecord>,
    warnings: Vec<String>,
    /// Guides (artboard blocks list the guides inside each board).
    guides: photocraft_doc::Guides,
    /// Layer comps to regenerate into each layer's `cmls`; None = preserved data still matches.
    comps: Option<(Vec<photocraft_doc::LayerComp>, Option<photocraft_doc::LayerComp>)>,
    /// Smart objects: embedded files and filter caches for the global blocks.
    smart: SmartOut,
    /// The document's patterns, which Pattern Fill layers are rendered from.
    patterns: Vec<photocraft_doc::Pattern>,
    /// The layer written as Photoshop's Background (see [`psd_background`]).
    background: Option<photocraft_doc::LayerId>,
}

/// The layer Photoshop should open as its Background: the bottom top-level layer when it is a
/// locked (transparency and position), unclipped, unmasked raster that is opaque over the whole
/// canvas at full opacity and Normal, as a Background always is. Photoshop recognises a
/// Background by the bottom layer having no transparency channel (measured on Photoshop 27.11:
/// with the `-1` channel written, 317 of 427 re-exported corpus files reopened with the
/// Background as an ordinary layer).
fn psd_background(doc: &Document) -> Option<photocraft_doc::LayerId> {
    let l = doc.layers.first()?;
    let LayerContent::Raster(s) = &l.content else { return None };
    let locked = l.locks.transparency && l.locks.position;
    let plain = !l.clipped && l.mask.is_none() && l.vector_mask.is_none() && l.opacity >= 1.0 && l.fill_opacity >= 1.0;
    let normal = l.blend == photocraft_doc::BlendMode::Normal;
    if !(locked && plain && normal) {
        return None;
    }
    let canvas = doc.bounds();
    if canvas.is_empty() || s.content_bounds() != canvas {
        return None;
    }
    let n = s.channels();
    let opaque = s.read_region(canvas).chunks_exact(n).all(|p| p.last().is_some_and(|a| *a >= 1.0));
    opaque.then_some(l.id)
}

/// How deep embedded documents are exported inside each other before a smart object is written
/// as pixels instead.
const MAX_NESTING: u32 = 8;

/// An embedded smart-object source as written to `lnk2`.
#[derive(Clone, Debug)]
struct Source {
    uuid: String,
    /// Pixel size of the source document.
    size: (f64, f64),
    dpi: f64,
}

fn contents_uuid(contents_id: SmartContentsId, bytes: &[u8]) -> String {
    let mut identity = blake3::Hasher::new();
    identity.update(&contents_id.0.to_le_bytes());
    identity.update(bytes);
    crate::smart_map::uuid_from(identity.finalize().as_bytes())
}

/// Smart-object state collected while emitting layers, written into the global blocks at the end.
#[derive(Default)]
struct SmartOut {
    /// Nesting depth of this export (embedded documents are exported recursively).
    depth: u32,
    /// The document's preserved global blocks (embedded files and filter caches from PSD import).
    globals: Vec<photocraft_doc::PsdGlobalBlock>,
    /// uuids of the files in the preserved linked-layer blocks.
    known: std::collections::HashSet<String>,
    /// uuids the layers reference.
    used: std::collections::HashSet<String>,
    /// Unreferenced preserved files may be dropped (every smart layer's file id is known).
    prune_ok: bool,
    /// Files to embed.
    add: Vec<crate::linked::LinkedFile>,
    /// Sources converted so far, by contents identity and bytes (`Err` = can't be embedded).
    converted: std::collections::HashMap<(SmartContentsId, [u8; 32]), Result<Source, String>>,
    /// Each placed layer has its own id, even when it duplicates an imported layer.
    placed_used: std::collections::HashSet<String>,
    /// Original instances reserve their imported placement regardless of layer order.
    placed_owners: std::collections::HashMap<String, photocraft_doc::LayerId>,
    /// Legacy records without a placed id stay unchanged for the original instance.
    unplaced_owners: std::collections::HashMap<SmartContentsId, photocraft_doc::LayerId>,
    /// The imported identity that keeps each original linked UUID.
    linked_contents: std::collections::HashMap<String, SmartContentsId>,
    /// Independent copies of preserved linked items, including unavailable external sources.
    linked_copies: std::collections::BTreeMap<(String, SmartContentsId), photocraft_doc::PsdGlobalBlock>,
    /// Embedded documents decoded while converting them, by uuid (for the filter caches).
    docs: std::collections::HashMap<String, Document>,
    /// Source composites (document pixel format, source space) by uuid; `None` = unreadable.
    composites: std::collections::HashMap<String, Option<(std::sync::Arc<Surface>, photocraft_geom::Rect)>>,
    /// Preserved `FEid` items (by placed id).
    old_fx: Vec<photocraft_psd::filter_effects::FilterEffectsItem>,
    /// Placed ids whose preserved `FEid` item stays.
    keep_fx: std::collections::HashSet<String>,
    /// Regenerated `FEid` items.
    new_fx: Vec<photocraft_psd::filter_effects::FilterEffectsItem>,
}

/// `FEid` data padded to 4 bytes inside the block length: Photoshop counts the padding in the
/// length and can't open a file whose `FEid` is padded outside it.
fn padded_fx(fx: &photocraft_psd::filter_effects::FilterEffects) -> Vec<u8> {
    let mut d = fx.to_bytes();
    d.resize(d.len().next_multiple_of(4), 0);
    d
}

const LINK_KEYS: [&[u8; 4]; 4] = [b"lnk2", b"lnk3", b"lnkD", b"lnkE"];
const FX_KEYS: [&[u8; 4]; 2] = [b"FEid", b"FXid"];

impl SmartOut {
    fn new(doc: &Document, depth: u32) -> Self {
        let globals = doc.metadata.psd_global_blocks.clone();
        let known = globals.iter().filter(|(_, k, _)| LINK_KEYS.contains(&k)).flat_map(|(_, _, d)| crate::linked::block_uuids(d)).collect();
        let old_fx = globals
            .iter()
            .filter(|(_, k, _)| FX_KEYS.contains(&k))
            .filter_map(|(_, _, d)| photocraft_psd::filter_effects::FilterEffects::parse(d).ok())
            .flat_map(|fx| fx.items)
            .collect();
        let mut linked_contents = std::collections::HashMap::new();
        let mut placed_owners = std::collections::HashMap::new();
        let mut unplaced_owners = std::collections::HashMap::new();
        let mut layers = doc.walk();
        // Duplicating a layer clears psd_id and gives it a newer runtime id.
        layers.sort_by_key(|(_, _, layer)| (layer.psd_id.is_none(), layer.id));
        for (_, _, layer) in layers {
            if let LayerContent::Smart(sm) = &layer.content
                && let photocraft_doc::SmartSource::Linked { path } = &sm.source
                && let Some(placed) = sm.psd_raw.as_ref().and_then(|d| crate::smart_map::parse_sold(d))
                && placed.idnt == *path
            {
                linked_contents.entry(path.clone()).or_insert(sm.contents_id);
                if placed.placed.is_empty() {
                    unplaced_owners.entry(sm.contents_id).or_insert(layer.id);
                } else {
                    placed_owners.entry(placed.placed).or_insert(layer.id);
                }
            }
        }
        SmartOut { depth, globals, known, prune_ok: true, old_fx, linked_contents, placed_owners, unplaced_owners, ..Default::default() }
    }

    /// The document's global blocks with the linked-layer files and filter caches brought up to
    /// date: unreferenced files dropped (when every reference is known), new files appended,
    /// filter caches kept or regenerated per smart object. Unchanged blocks stay byte-identical.
    fn finish(&self, blocks: Vec<photocraft_doc::PsdGlobalBlock>) -> Vec<photocraft_doc::PsdGlobalBlock> {
        use std::sync::Arc;
        let keep = |u: &str| !self.prune_ok || self.used.contains(u);
        let mut out = Vec::with_capacity(blocks.len() + 2);
        let mut added = false;
        let mut fx_placed = false;
        for (sig, key, data) in blocks {
            if LINK_KEYS.contains(&&key) {
                let add: &[crate::linked::LinkedFile] = if !added && &key == b"lnk2" { &self.add } else { &[] };
                added |= &key == b"lnk2";
                match crate::linked::rebuild_block(&data, &keep, add) {
                    None => out.push((sig, key, data)),
                    Some(d) if d.is_empty() => {}
                    Some(d) => out.push((sig, key, Arc::new(d))),
                }
            } else if FX_KEYS.contains(&&key) {
                let Ok(mut fx) = photocraft_psd::filter_effects::FilterEffects::parse(&data) else {
                    out.push((sig, key, data));
                    continue;
                };
                let before = fx.items.len();
                fx.items.retain(|i| self.keep_fx.contains(&i.id));
                let fresh = !fx_placed && !self.new_fx.is_empty();
                if fresh {
                    fx.items.extend(self.new_fx.iter().cloned());
                    fx_placed = true;
                }
                if fx.items.len() == before && !fresh {
                    out.push((sig, key, data));
                } else if !fx.items.is_empty() {
                    out.push((sig, key, Arc::new(padded_fx(&fx))));
                }
            } else {
                out.push((sig, key, data));
            }
        }
        if !added && !self.add.is_empty() {
            let data: Vec<u8> = self.add.iter().flat_map(crate::linked::encode_linked_file).collect();
            out.push((*b"8BIM", *b"lnk2", Arc::new(data)));
        }
        for (sig, key, data) in self.linked_copies.values() {
            if let Some((_, _, block)) = out.iter_mut().find(|(s, k, _)| s == sig && k == key) {
                Arc::make_mut(block).extend_from_slice(data);
            } else {
                out.push((*sig, *key, data.clone()));
            }
        }
        if !fx_placed && !self.new_fx.is_empty() {
            let fx = photocraft_psd::filter_effects::FilterEffects { version: 3, items: self.new_fx.clone() };
            out.push((*b"8BIM", *b"FEid", Arc::new(padded_fx(&fx))));
            if !out.iter().any(|(_, k, _)| k == b"FMsk") {
                // Filter mask overlay: RGB red at 50 %, Photoshop's default.
                out.push((*b"8BIM", *b"FMsk", Arc::new(vec![0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 50])));
            }
        }
        out
    }
}

/// The source size of a smart object whose file we can't decode: its rendered bounds taken back
/// through the inverse transform (the source's top-left is the origin).
fn size_from_layer(sm: &photocraft_doc::SmartObject) -> (f64, f64) {
    let r = sm.cache.as_ref().map(Surface::content_bounds).filter(|r| !r.is_empty());
    let (Some(r), Some(inv)) = (r, sm.transform.inverse()) else { return (1.0, 1.0) };
    let [a, b, c, d, e, f] = inv.m;
    let (mut w, mut h) = (1.0f64, 1.0f64);
    for (x, y) in [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)] {
        let (x, y) = (f64::from(x), f64::from(y));
        w = w.max((a * x + c * y + e).round());
        h = h.max((b * x + d * y + f).round());
    }
    (w, h)
}

fn upsert(raw: &mut Vec<([u8; 4], Vec<u8>)>, key: &[u8; 4], data: Vec<u8>, before: Option<&[u8; 4]>) {
    if let Some(e) = raw.iter_mut().find(|(k, _)| k == key) {
        e.1 = data;
        return;
    }
    match before.and_then(|b| raw.iter().position(|(k, _)| k == b)) {
        Some(i) => raw.insert(i, (*key, data)),
        None => raw.push((*key, data)),
    }
}

fn h_of(r: photocraft_geom::Rect) -> usize {
    r.height() as usize
}

fn psd_mode(m: ColorMode) -> PsdMode {
    match m {
        ColorMode::Grayscale => PsdMode::Grayscale,
        ColorMode::Cmyk => PsdMode::Cmyk,
        ColorMode::Lab => PsdMode::Lab,
        _ => PsdMode::Rgb,
    }
}

fn to_psd_rect(r: photocraft_geom::Rect) -> PsdRect {
    PsdRect { top: r.y0, left: r.x0, bottom: r.y1, right: r.x1 }
}

fn q255(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

impl Ex {
    fn compression(&self) -> Compression {
        if self.fmt.sample == SampleType::F32 { Compression::ZipPrediction } else { Compression::Rle }
    }

    fn encode(&self, id: i16, plane: &[u8], w: usize, h: usize) -> ChannelData {
        let depth = psd_depth(self.fmt.sample);
        // Photoshop writes empty layer/mask channels as just the raw compression code.
        // A ZIP stream for a zero-sized channel makes readers try to decompress an
        // invalid image extent (notably psd-tools for 32-bit adjustment layers).
        let compression = if w == 0 || h == 0 { Compression::Raw } else { self.compression() };
        ChannelData::encode(id, compression, plane, w, h, depth, self.version)
            .or_else(|_| ChannelData::encode(id, Compression::Raw, plane, w, h, depth, self.version))
            .unwrap_or(ChannelData { id, compression: Some(Compression::Raw), data: plane.to_vec() })
    }

    fn empty_channels(&self) -> Vec<ChannelData> {
        (-1..self.cc as i16).map(|id| self.encode(id, &[], 0, 0)).collect()
    }

    fn pixel_channels(&self, s: &Surface, name: &str) -> (PsdRect, Vec<ChannelData>, Vec<String>) {
        let mut warnings = Vec::new();
        let s = if s.format() != self.fmt {
            warnings.push(format!("layer \"{name}\": pixels converted from {:?} to {:?}", s.format(), self.fmt));
            s.convert(self.fmt)
        } else {
            s.clone()
        };
        let r = s.content_bounds();
        if r.is_empty() {
            return (PsdRect::default(), self.empty_channels(), warnings);
        }
        // The interleaved gather is the sequential pass for full-canvas layers, so it runs
        // in row bands on all cores and the band results concatenate in order.
        let row_len = r.width() as usize;
        let bands = crate::pixels::bands(h_of(r));
        let parts =
            crate::pixels::par_map(bands, |rows| s.to_interleaved(photocraft_geom::Rect::new(r.x0, r.y0 + rows.start as i32, r.x1, r.y0 + rows.end as i32)));
        let mut bytes = Vec::with_capacity(row_len * h_of(r) * s.format().bytes_per_pixel());
        for part in parts {
            bytes.extend_from_slice(&part);
        }
        if self.fmt.mode == ColorMode::Lab && self.fmt.sample == SampleType::U16 {
            crate::pixels::lab16_chroma(&mut bytes, self.cc + 1, false);
        }
        let mut invert = vec![self.cmyk; self.cc];
        invert.push(false);
        let planes = deinterleave(&bytes, self.cc + 1, self.fmt.sample, &invert);
        drop(bytes);
        let (w, h) = (r.width() as usize, r.height() as usize);
        // Alpha (-1) first, then the colour channels; each compressed on its own thread.
        let order: Vec<(i16, &Vec<u8>)> =
            std::iter::once((-1, &planes[self.cc])).chain(planes.iter().take(self.cc).enumerate().map(|(c, p)| (c as i16, p))).collect();
        let ch = crate::pixels::par_map(order, |(id, plane)| self.encode(id, plane, w, h));
        (to_psd_rect(r), ch, warnings)
    }

    fn mask(&self, m: &LayerMask, vector: Option<(f32, f32)>) -> (MaskData, Option<ChannelData>) {
        let s = if m.surface.format() != self.mask_fmt { m.surface.convert(self.mask_fmt) } else { m.surface.clone() };
        let default = s.default_pixel().first().copied().unwrap_or(0.0);
        let r = s.content_bounds();
        let (w, h) = (r.width() as usize, r.height() as usize);
        let plane = if r.is_empty() { Vec::new() } else { deinterleave(&s.to_interleaved(r), 1, self.mask_fmt.sample, &[false]).remove(0) };
        let mut flags = 0u8;
        if !m.linked {
            flags |= PsdMask::FLAG_RELATIVE;
        }
        if !m.enabled {
            flags |= PsdMask::FLAG_DISABLED;
        }
        let parameters = mask_parameters(Some((m.density, m.feather)), vector);
        if parameters.is_some() {
            flags |= PsdMask::FLAG_PARAMETERS;
        }
        let pm = PsdMask {
            rect: if r.is_empty() { PsdRect::default() } else { to_psd_rect(r) },
            default_color: q255(default),
            flags,
            trailing: if parameters.is_none() { vec![0, 0] } else { Vec::new() },
            parameters,
            real: None,
            real_first: true,
        };
        (MaskData::Mask(pm), Some(self.encode(-2, &plane, w, h)))
    }

    /// Prefer numbers above imported layer IDs, wrapping safely at u32::MAX.
    fn fresh_id(&mut self) -> u32 {
        loop {
            self.next_id = self.next_id.wrapping_add(1).max(1);
            if self.used_ids.insert(self.next_id) {
                return self.next_id;
            }
        }
    }

    fn layer_id(&mut self, l: &Layer) -> u32 {
        match self.layer_ids.get(&l.id) {
            Some(id) => *id,
            None => self.fresh_id(),
        }
    }

    fn record(&mut self, l: &Layer, extra: Vec<TaggedBlock>, pixels: Option<&Surface>) -> LayerRecord {
        let id = self.layer_id(l);
        let (rect, mut channels, warnings) = match pixels {
            Some(s) => self.pixel_channels(s, &l.name),
            None => (PsdRect::default(), self.empty_channels(), Vec::new()),
        };
        self.warnings.extend(warnings);
        let mut mask = MaskData::None;
        // Vector mask density/feather live in the mask parameters, with or without a user mask.
        let vector = l.vector_mask.as_ref().map(|vm| (vm.density, vm.feather));
        if let Some(m) = &l.mask {
            let (md, ch) = self.mask(m, vector);
            mask = md;
            channels.extend(ch);
        } else if let Some(parameters) = mask_parameters(None, vector) {
            // Parameters only: no user mask channel, so nothing hides on re-import.
            mask = MaskData::Mask(PsdMask {
                rect: PsdRect::default(),
                default_color: 255,
                flags: PsdMask::FLAG_PARAMETERS,
                parameters: Some(parameters),
                real: None,
                trailing: Vec::new(),
                real_first: true,
            });
        }
        // A Background carries no transparency channel: that is how Photoshop tells it apart.
        if self.background == Some(l.id) {
            channels.retain(|c| c.id != -1);
        }
        // As Photoshop writes them: bit 4 is meaningful, and set for layers whose pixels don't
        // show (adjustment layers).
        let mut flags = LayerFlags(LayerFlags::BIT4_USEFUL);
        if matches!(l.content, LayerContent::Adjustment(_)) {
            flags.0 |= LayerFlags::PIXEL_DATA_IRRELEVANT;
        }
        flags.set_hidden(!l.visible);
        flags.set_transparency_protected(l.locks.transparency);
        let mut blocks = vec![TaggedBlock::unicode_name(&l.name), TaggedBlock::layer_id(id)];
        let lspf = blocks::lspf_from_locks(&l.locks);
        if lspf != 0 {
            blocks.push(TaggedBlock::protection(lspf));
        }
        if l.label != photocraft_doc::LabelColor::None {
            blocks.push(TaggedBlock::sheet_color(blocks::label_index(l.label)));
        }
        blocks.push(TaggedBlock::fill_opacity(q255(l.fill_opacity)));
        // Effects: the original lfx2 while the effects are unchanged (or
        // could not be decoded at all); otherwise regenerated.
        let fx = &l.effects;
        for effect in &fx.items {
            use photocraft_doc::effects::Effect;
            let modes = match effect {
                Effect::DropShadow(s) | Effect::InnerShadow(s) => [Some(s.common.blend), None],
                Effect::OuterGlow(g) | Effect::InnerGlow(g) => [Some(g.common.blend), None],
                Effect::Stroke(s) => [Some(s.common.blend), None],
                Effect::ColorOverlay { common, .. } | Effect::GradientOverlay { common, .. } | Effect::PatternOverlay { common, .. } => {
                    [Some(common.blend), None]
                }
                Effect::Satin(s) => [Some(s.common.blend), None],
                Effect::BevelEmboss(b) => [Some(b.highlight.blend), Some(b.shadow.blend)],
            };
            for mode in modes.into_iter().flatten() {
                if !mode.has_psd_equivalent() {
                    self.warnings.push(format!(
                        "layer \"{}\", {}: {} has no PSD equivalent and was written as Normal; save as .pcraft to preserve it",
                        l.name,
                        effect.label(),
                        mode.label()
                    ));
                }
            }
        }
        let lfx2 = match effects_unchanged(l) {
            Some(true) | None if fx.psd_raw.is_some() => fx.psd_raw.as_ref().map(|r| r.to_vec()),
            _ => (!fx.items.is_empty()).then(|| crate::effects_map::write_lfx2(fx.enabled, &fx.items)),
        };
        if let Some(d) = lfx2 {
            let key = if l.is_group() { *b"lfxs" } else { *b"lfx2" };
            blocks.push(TaggedBlock::new(key, d));
        }
        blocks.extend(extra);
        // Layer-level blocks keep their pad byte inside the length, as Photoshop writes them:
        // readers such as psd-tools do not skip a pad after an odd length, so a regenerated
        // odd-length `lfx2` misaligned every block after it (#200).
        for b in &mut blocks {
            if b.data.len() % 2 == 1 && b.padding.is_none() {
                b.data.push(0);
            }
        }
        if !l.blend.has_psd_equivalent() {
            self.warnings.push(format!(
                "layer \"{}\": {} has no PSD equivalent and was written as Normal; save as .pcraft to preserve it",
                l.name,
                l.blend.label()
            ));
        }
        LayerRecord {
            rect,
            channels,
            blend_mode: PsdBlend::from_key(l.blend.psd_key()),
            opacity: q255(l.opacity),
            clipping: u8::from(l.clipped),
            flags,
            filler: 0,
            mask,
            blending_ranges: crate::blocks::ranges_from_blend_if(&l.blend_if, self.cc),
            name: legacy_name(&l.name),
            blocks,
            extra_trailing: Vec::new(),
        }
    }

    /// Content blocks for `l`: preserved raw blocks (with `psd_raw`
    /// overrides) plus regenerated adjustment/fill blocks. A preserved
    /// adjustment or fill block is reused verbatim while it still decodes to
    /// the layer's current parameters; otherwise it is regenerated.
    fn content_blocks(&mut self, l: &Layer) -> Vec<TaggedBlock> {
        let mut raw: Vec<([u8; 4], Vec<u8>)> = l.psd_blocks.iter().filter(|(k, _)| !REGENERATED.contains(&k)).map(|(k, d)| (*k, d.to_vec())).collect();
        // A stale multi-effects block would override the regenerated lfx2.
        if effects_unchanged(l) == Some(false) {
            raw.retain(|(k, _)| k != b"lmfx");
        }
        let mut regenerated: Vec<([u8; 4], Vec<u8>)> = Vec::new();
        // `keys` is in the importer's priority order (`psd_raw` came from the first key present),
        // not file order: Photoshop writes `PlLd` before `SoLd`, and overwriting the first match
        // in file order put `soLD` data under the `PlLd` key (#200).
        let set_principal = |raw: &mut Vec<([u8; 4], Vec<u8>)>, keys: &[&[u8; 4]], data: Option<&std::sync::Arc<Vec<u8>>>| {
            let Some(d) = data else { return };
            match keys.iter().find_map(|k| raw.iter().position(|(rk, _)| rk == *k)) {
                Some(i) => raw[i].1 = d.to_vec(),
                None => raw.push((*keys[0], d.to_vec())),
            }
        };
        let fill_rule = |raw: &mut Vec<([u8; 4], Vec<u8>)>, regenerated: &mut Vec<([u8; 4], Vec<u8>)>, f: &photocraft_doc::Fill| {
            let keep = raw.iter().any(|(k, d)| matches!(k, b"SoCo" | b"GdFl" | b"PtFl") && blocks::parse_fill(k, d).as_ref() == Some(f));
            if !keep {
                raw.retain(|(k, _)| !matches!(k, b"SoCo" | b"GdFl" | b"PtFl"));
                // Photoshop pads layer-level blocks to a multiple of 4 inside the length (every
                // one in its own files is). A fill descriptor of 2 mod 4 bytes, as a pattern
                // name of the wrong length gives, left the record's extra data unaligned.
                let (key, mut data) = blocks::write_fill(f);
                data.resize(data.len().next_multiple_of(4), 0);
                regenerated.push((key, data));
            }
        };
        match &l.content {
            LayerContent::Adjustment(a) => {
                let channels = match self.fmt.mode {
                    ColorMode::Rgb => adjust_map::Channels::Rgb,
                    ColorMode::Grayscale => adjust_map::Channels::Gray,
                    ColorMode::Cmyk => adjust_map::Channels::Cmyk,
                    ColorMode::Lab => adjust_map::Channels::Lab,
                    _ => adjust_map::Channels::Other,
                };
                let cged = raw.iter().find(|(k, _)| k == b"CgEd").map(|(_, d)| d.clone());
                let keep = raw.iter().any(|(k, d)| {
                    // A previously saved PhotoCraft v2 Photo Filter block may be 18 bytes.
                    // It parses here, but Photoshop requires four-byte alignment, so repair
                    // that legacy record on re-save rather than preserving the broken bytes.
                    let legacy_phfl = k == b"phfl" && d.starts_with(&[0, 2]) && d.len() % 4 != 0;
                    !legacy_phfl && adjust_map::ADJUSTMENT_KEYS.contains(&k) && adjust_map::parse(k, d, cged.as_deref(), channels) == *a
                });
                // A Gradient Map whose stops are unchanged keeps the file's own block (stops in
                // the document's CMYK/Lab/Gray space, smoothness, method) with Reverse and Dither
                // updated in place.
                let keep = keep
                    || (matches!(a, photocraft_doc::Adjustment::GradientMap { .. })
                        && raw.iter_mut().any(|(k, d)| k == b"grdm" && adjust_map::retarget_gradient_map(d, a, channels)));
                if !keep {
                    raw.retain(|(k, _)| !adjust_map::ADJUSTMENT_KEYS.contains(&k) && k != b"CgEd");
                    let w = adjust_map::write(a);
                    if w.is_empty() {
                        self.warnings.push(format!("layer \"{}\": {} adjustment is not yet written to PSD", l.name, a.label()));
                    }
                    // Photoshop's `clrL` descriptor has no interpolation setting, so the
                    // reopened layer renders trilinear.
                    if matches!(a, photocraft_doc::Adjustment::ColorLookup { tetrahedral: true, .. }) {
                        self.warnings.push(format!(
                            "layer \"{}\": Color Lookup tetrahedral interpolation is saved as trilinear (PSD has no interpolation setting)",
                            l.name
                        ));
                    }
                    regenerated.extend(w);
                }
            }
            LayerContent::Fill(f) => fill_rule(&mut raw, &mut regenerated, f),
            LayerContent::Shape(sh) => {
                if let Some(f) = &sh.fill {
                    fill_rule(&mut raw, &mut regenerated, f);
                }
                let (w, h) = (self.canvas.width(), self.canvas.height());
                crate::vector_map::shape_blocks(sh, &mut raw, w, h, self.dpi);
            }
            LayerContent::Text(t) => {
                // Text layers without PSD data (created here) get a generated TySh.
                let generated = t.psd_raw.is_none().then(|| std::sync::Arc::new(photocraft_text::psd::build_tysh(t, self.dpi, None)));
                let src = t.psd_raw.as_ref().or(generated.as_ref());
                // Character/paragraph style sheets from the document's styles.
                let styled = src.and_then(|d| crate::text_styles_map::export_tysh(d, t, &self.text_styles, self.dpi)).map(std::sync::Arc::new);
                // `TextIndex` names the layer's object in the document's `Txt2` (where the
                // auto-kern mode lives); left at 0, every type layer would claim object 0 (#1348).
                // Keep an unchanged preserved TySh byte-for-byte (corpus_tysh_lossless).
                // Duplicate or out-of-range file TextIndex values must be renumbered,
                // however, even when no style sheet needed rebuilding: otherwise two
                // layers may read the same Txt2 object and acquire the wrong kerning.
                let idx = self.text_index.get(&l.id).copied().unwrap_or(0);
                let rewrite = styled.as_ref().or(generated.as_ref()).or_else(|| src.filter(|d| photocraft_text::psd::text_index(d.as_slice()) != Some(idx)));
                let fresh = rewrite.and_then(|d| photocraft_text::psd::set_text_index(d.as_slice(), idx)).map(std::sync::Arc::new);
                set_principal(&mut raw, &[b"TySh"], fresh.as_ref().or(styled.as_ref().or(src)));
                if !raw.iter().any(|(k, _)| k == b"TySh") {
                    self.warnings.push(format!("layer \"{}\": text layer written as pixels (no TySh data)", l.name));
                }
            }
            LayerContent::Smart(sm) => {
                let parsed = sm.psd_raw.as_deref().and_then(|d| crate::smart_map::parse_sold(d));
                if sm.psd_raw.is_some() && parsed.is_none() && !self.plld_edited(sm) {
                    // Placed-layer data that isn't `SoLd` (a CS2 `PlLd` record, or one we can't
                    // read) and still says what the layer is: kept verbatim.
                    self.smart.prune_ok = false;
                    set_principal(&mut raw, smart_keys(sm.psd_raw.as_deref().map(Vec::as_slice)), sm.psd_raw.as_ref());
                } else {
                    // An edited CS2 `PlLd`-only layer is written like any other smart object
                    // (`SoLd` and a regenerated `PlLd`), with its current placement (#2465).
                    self.smart_blocks(l, sm, parsed, &mut raw);
                }
            }
            // A pixel layer carries no content-defining blocks: a fill, placed (smart), type or
            // shape-fill block left over from before Rasterize would make a reader (ours or
            // Photoshop's) reopen the layer as that kind and discard the pixels. A shape stroke
            // (`vstk`) beside a vector mask makes a shape too; without one, Photoshop keeps it on
            // pixel layers, so it stays. A vector mask (`vmsk`/`vsms`, `vogk`) stays.
            LayerContent::Raster(_) => {
                let masked = l.vector_mask.is_some();
                raw.retain(|(k, _)| !(RASTER_STALE_KEYS.contains(&k) || adjust_map::ADJUSTMENT_KEYS.contains(&k) || k == b"CgEd" || (k == b"vstk" && masked)));
            }
            LayerContent::Group(_) => {}
        }
        // Effects reference point: written from the field (in place, keeping block order).
        match l.effects.reference {
            Some((x, y)) => {
                let data: Vec<u8> = x.to_be_bytes().into_iter().chain(y.to_be_bytes()).collect();
                match raw.iter_mut().find(|(k, _)| k == b"fxrp") {
                    Some(e) => e.1 = data,
                    None => raw.push((*b"fxrp", data)),
                }
            }
            None => raw.retain(|(k, _)| k != b"fxrp"),
        }
        // Advanced Blending channel restrictions: written from the field (in place).
        match crate::blocks::brst_data(l.excluded_channels) {
            Some(data) => match raw.iter_mut().find(|(k, _)| k == b"brst") {
                Some(e) => e.1 = data,
                None => raw.push((*b"brst", data)),
            },
            None => raw.retain(|(k, _)| k != b"brst"),
        }
        // Advanced Blending: written from the field (in place when the block was imported).
        crate::blocks::put_advanced(&l.advanced, &mut raw);
        if !matches!(l.content, LayerContent::Shape(_)) {
            self.vector_mask_block(l, &mut raw);
        }
        crate::comps_map::artboard_block(&self.guides, l, &mut raw);
        if let Some((comps, last)) = &self.comps {
            // Ids are assigned to every layer before emitting.
            let id = self.layer_ids.get(&l.id).copied().unwrap_or(0);
            crate::comps_map::set_cmls(&mut raw, crate::comps_map::write_cmls(comps, last.as_ref(), l, id));
        }
        regenerated.into_iter().chain(raw).map(|(k, d)| TaggedBlock::new(k, d)).collect()
    }

    /// Keeps, regenerates or removes the `vmsk`/`vsms` block of a non-shape layer.
    fn vector_mask_block(&mut self, l: &Layer, raw: &mut Vec<([u8; 4], Vec<u8>)>) {
        let (w, h) = (self.canvas.width(), self.canvas.height());
        let pos = raw.iter().position(|(k, _)| k == b"vsms" || k == b"vmsk");
        match (&l.vector_mask, pos) {
            (None, Some(_)) => raw.retain(|(k, _)| k != b"vsms" && k != b"vmsk"),
            (None, None) => {}
            (Some(vm), Some(i)) if crate::vector_map::vector_mask_block_matches(&raw[i].1, vm, w, h) => {}
            (Some(vm), pos) => {
                let data = crate::vector_map::vector_mask_bytes(vm, w, h);
                match pos {
                    Some(i) => raw[i].1 = data,
                    None => raw.push((*b"vmsk", data)),
                }
            }
        }
    }

    /// Placed-layer blocks (`PlLd` + `SoLd`) of a smart object, its embedded file (`lnk2`) and
    /// filter cache (`FEid`). An imported placed layer that nothing changed is written verbatim;
    /// otherwise the blocks are regenerated (from the imported descriptor where the source is
    /// the same). Without a source that can be embedded the layer is written as pixels.
    fn smart_blocks(&mut self, l: &Layer, sm: &photocraft_doc::SmartObject, template: Option<crate::smart_map::Placed>, raw: &mut Vec<([u8; 4], Vec<u8>)>) {
        use crate::smart_map::{FilterStack, PlacedSpec, filter_fx, plld_bytes, sold_bytes, uuid_from};
        let src = match self.smart_source(sm, template.as_ref()) {
            Ok(s) => s,
            Err(e) => {
                raw.retain(|(k, _)| !matches!(k, b"SoLd" | b"PlLd" | b"SoLE"));
                self.warnings.push(format!("layer \"{}\": smart object written as pixels ({e})", l.name));
                return;
            }
        };
        self.smart.used.insert(src.uuid.clone());
        let stack = FilterStack {
            enabled: sm.filters_enabled,
            filters: sm.smart_filters.clone(),
            mask_enabled: sm.filter_mask.as_ref().is_none_or(|m| m.enabled),
            mask_linked: sm.filter_mask.as_ref().is_some_and(|m| m.linked),
        };
        let same_source = template.as_ref().filter(|t| t.idnt == src.uuid);
        let unchanged = same_source.zip(sm.psd_raw.as_ref()).is_some_and(|(t, data)| self.placed_unchanged(t, data, sm, &stack));
        let placed = match same_source {
            Some(t) if t.placed.is_empty() && unchanged && self.smart.unplaced_owners.get(&sm.contents_id) == Some(&l.id) => String::new(),
            Some(t) if self.smart.placed_owners.get(&t.placed) == Some(&l.id) && self.smart.placed_used.insert(t.placed.clone()) => t.placed.clone(),
            _ => {
                let seed = format!("{}:{}", src.uuid, l.id.0);
                let mut suffix = 0u64;
                loop {
                    let id = uuid_from(if suffix == 0 { seed.clone() } else { format!("{seed}:{suffix}") }.as_bytes());
                    if !self.smart.placed_owners.contains_key(&id) && self.smart.placed_used.insert(id.clone()) {
                        break id;
                    }
                    suffix += 1;
                }
            }
        };
        if let (Some(t), Some(data)) = (same_source, sm.psd_raw.as_ref())
            && t.placed == placed
            && unchanged
        {
            let keys: &[&[u8; 4]] = if raw.iter().any(|(k, _)| k == b"SoLd") { &[b"SoLd"] } else { &[b"SoLE", b"SoLd"] };
            match keys.iter().find_map(|k| raw.iter().position(|(rk, _)| rk == *k)) {
                Some(i) => raw[i].1 = data.to_vec(),
                None => raw.push((*b"SoLd", data.to_vec())),
            }
            if !stack.filters.is_empty() {
                self.smart.keep_fx.insert(t.placed.clone());
            }
            return;
        }
        let size = same_source.and_then(|t| crate::smart_map::stored_size(&t.descriptor)).unwrap_or(src.size);
        let size = if size.0 > 0.0 && size.1 > 0.0 { size } else { size_from_layer(sm) };
        let mut warnings = Vec::new();
        let fx = (!stack.filters.is_empty()).then(|| filter_fx(&stack, &l.name, &mut warnings));
        let spec = PlacedSpec {
            idnt: &src.uuid,
            placed: &placed,
            transform: sm.transform,
            perspective: sm.perspective,
            size,
            dpi: src.dpi,
            warp: sm.warp.as_ref(),
            filter_fx: fx,
        };
        let sold = sold_bytes(same_source.map(|t| &t.descriptor), &spec, &mut warnings);
        let plld = plld_bytes(&spec, &mut warnings);
        warnings.dedup();
        let key = if same_source.is_some() && raw.iter().any(|(k, _)| k == b"SoLE") && !raw.iter().any(|(k, _)| k == b"SoLd") { *b"SoLE" } else { *b"SoLd" };
        raw.retain(|(k, _)| k == &key || !matches!(k, b"SoLd" | b"SoLE"));
        upsert(raw, b"PlLd", plld, Some(&key));
        upsert(raw, &key, sold, None);
        if !stack.filters.is_empty() {
            let bounds = sm.cache.as_ref().map_or(self.canvas, |c| c.content_bounds().union(&self.canvas));
            let item = match (sm.stack_mode, self.source_composite(&src.uuid)) {
                (None, Some((img, img_bounds))) => {
                    let unfiltered = match &sm.perspective {
                        Some(p) => {
                            photocraft_algo::warp::place_source_projective(&img, img_bounds, &photocraft_algo::transform::Homography(*p), sm.warp.as_ref())
                        }
                        None => photocraft_algo::warp::place_source(&img, img_bounds, &sm.transform, sm.warp.as_ref()),
                    };
                    match unfiltered {
                        Ok(unfiltered) => crate::smart_map::feid_item(&placed, &unfiltered, sm.filter_mask.as_ref(), bounds, self.fmt),
                        Err(e) => {
                            warnings.push(format!("smart filter mask was not re-rendered for PSD export ({e})"));
                            None
                        }
                    }
                }
                _ => None,
            };
            match item {
                Some(i) => self.smart.new_fx.push(i),
                // Without the unfiltered pixels Photoshop re-renders from the source itself; only
                // the filter mask has nowhere to go.
                None if sm.filter_mask.is_some() => self.warnings.push(format!("layer \"{}\": the smart filter mask was not written to PSD", l.name)),
                None => {}
            }
        }
        self.warnings.extend(warnings.into_iter().map(|w| if w.starts_with("layer ") { w } else { format!("layer \"{}\": {w}", l.name) }));
    }

    /// Whether a CS2 placed layer (`PlLd` with no `SoLd`, #2465) no longer matches its record:
    /// moved, transformed, warped, filtered or given other contents. `false` for a record that
    /// can't be read (or whose warp can't), or whose file isn't embedded: its contents were never
    /// available, so it is kept.
    fn plld_edited(&self, sm: &photocraft_doc::SmartObject) -> bool {
        let Some(data) = sm.psd_raw.as_deref() else { return false };
        let Some((id, corners)) = blocks::parse_plld(data) else { return false };
        // Without its embedded file the contents were never available: keep the record as it
        // is, rather than resolve the id (bytes from the file) as a path on disk.
        if !self.smart.known.contains(&id) {
            return false;
        }
        let warp = match blocks::parse_plld_warp(data) {
            blocks::PlldWarp::Unreadable => return false,
            blocks::PlldWarp::None => None,
            blocks::PlldWarp::Warp(w) => Some(w),
        };
        if sm.source != (photocraft_doc::SmartSource::Linked { path: id.clone() })
            || sm.warp != warp
            || !sm.smart_filters.is_empty()
            || sm.filter_mask.is_some()
        {
            return true;
        }
        // Placed as the importer places it (the same source size, from the same file, and the
        // same frame: a custom warp's control-point box, #3677).
        let size = crate::linked::source_size_in_blocks(self.smart.globals.iter().map(|(_, k, d)| (k, d.as_slice())), &id, &photocraft_raster::Interrupt::NONE);
        blocks::plld_transform(&corners, size, warp.as_ref()) != (sm.transform, sm.perspective)
    }

    /// Whether an imported placed layer still says what the smart object is: transform, warp,
    /// filter stack and filter mask as stored.
    fn placed_unchanged(&self, t: &crate::smart_map::Placed, data: &[u8], sm: &photocraft_doc::SmartObject, stack: &crate::smart_map::FilterStack) -> bool {
        if t.transform != sm.transform || blocks::parse_placed_warp(b"SoLd", data) != sm.warp || t.stack.clone().unwrap_or_default() != *stack {
            return false;
        }
        if stack.filters.is_empty() {
            return true;
        }
        let old = self.smart.old_fx.iter().find(|i| i.id == t.placed);
        let mask = old.map(|i| crate::smart_map::mask_from_item(i, self.mask_fmt.sample, stack));
        match mask {
            Some(Ok(m)) => m == sm.filter_mask,
            Some(Err(_)) => false,
            None => sm.filter_mask.is_none(),
        }
    }

    /// The `lnk2` file a smart object's source is written as: a preserved embedded file, or a new
    /// one (a `.pcraft` source becomes a PSB of the nested document; image files are embedded as
    /// they are; a linked file is read and embedded).
    fn smart_source(&mut self, sm: &photocraft_doc::SmartObject, template: Option<&crate::smart_map::Placed>) -> Result<Source, String> {
        use photocraft_doc::SmartSource;
        match &sm.source {
            // A file of the imported document, or one its placed-layer data names (a missing
            // link stays a smart object, as in Photoshop).
            SmartSource::Linked { path } if self.smart.known.contains(path) || template.is_some_and(|t| t.idnt == *path) => {
                let stored = template.filter(|t| t.idnt == *path).and_then(|t| crate::smart_map::stored_size(&t.descriptor));
                let geometry = stored.map(|size| (size, f64::from(self.dpi))).or_else(|| {
                    let meta = photocraft_doc::Metadata { psd_global_blocks: self.smart.globals.clone(), ..Default::default() };
                    let f = crate::linked::find_linked_file(&meta, path)?;
                    crate::linked::source_geometry(&f.file_name, &f.bytes, &photocraft_raster::Interrupt::NONE).ok()
                });
                let (size, dpi) = geometry.unwrap_or((size_from_layer(sm), f64::from(self.dpi)));
                let original = self.smart.linked_contents.entry(path.clone()).or_insert(sm.contents_id);
                // An absent Idnt is not a shared-source identity; import treats each as fresh.
                let uuid = if path.is_empty() || *original == sm.contents_id {
                    path.clone()
                } else {
                    let uuid = contents_uuid(sm.contents_id, path.as_bytes());
                    let key = (path.clone(), sm.contents_id);
                    if !self.smart.linked_copies.contains_key(&key) {
                        let copy = self
                            .smart
                            .globals
                            .iter()
                            .find_map(|(sig, k, data)| {
                                LINK_KEYS
                                    .contains(&k)
                                    .then(|| crate::linked::copy_item(data, path, &uuid))
                                    .flatten()
                                    .map(|data| (*sig, *k, std::sync::Arc::new(data)))
                            })
                            .ok_or_else(|| format!("the linked source {path} has no readable record to copy independently"))?;
                        self.smart.linked_copies.insert(key, copy);
                    }
                    uuid
                };
                Ok(Source { uuid, size, dpi })
            }
            SmartSource::Linked { path } => {
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let bytes = std::fs::read(path).map_err(|e| format!("the linked file {path} can't be read: {e}"))?;
                    let name = path.rsplit(['/', '\\']).next().unwrap_or(path).to_string();
                    self.warnings.push(format!("the linked smart object {name} was embedded (PSD export keeps no external links)"));
                    self.embed(sm.contents_id, &name, &bytes)
                }
                #[cfg(target_arch = "wasm32")]
                Err(format!("the linked file {path} can't be read here"))
            }
            SmartSource::Embedded { file_name, bytes } => self.embed(sm.contents_id, file_name, bytes),
        }
    }

    /// The composite of the source `uuid` (an embedded file of this export or of the imported
    /// document) in the document's pixel format, its top-left at the origin; decoded once.
    fn source_composite(&mut self, uuid: &str) -> Option<(std::sync::Arc<Surface>, photocraft_geom::Rect)> {
        if let Some(c) = self.smart.composites.get(uuid) {
            return c.clone();
        }
        let doc = match self.smart.docs.get(uuid) {
            Some(d) => Some(d.clone()),
            None => {
                let file = self.smart.add.iter().find(|f| f.uuid == uuid).cloned().or_else(|| {
                    let meta = photocraft_doc::Metadata {
                        psd_global_blocks: self.smart.globals.iter().chain(self.smart.linked_copies.values()).cloned().collect(),
                        ..Default::default()
                    };
                    crate::linked::find_linked_file(&meta, uuid)
                });
                file.and_then(|f| crate::import(&f.file_name, &f.bytes).ok()).map(|r| r.document)
            }
        };
        let c = doc.map(|d| {
            let mut s = Surface::new(self.fmt);
            let fmt = self.fmt;
            let _ = photocraft_compose::render_bands(&d, d.bounds(), 0, |band| -> Result<(), ()> {
                let mut vals = Vec::with_capacity(band.px.len() * fmt.channels());
                let mut v = [0.0f32; 5];
                for p in &band.px {
                    let n = photocraft_raster::from_rgba_into(&fmt, *p, &mut v);
                    vals.extend_from_slice(&v[..n]);
                }
                s.write_region(band.rect, &vals);
                Ok(())
            });
            s.prune();
            (std::sync::Arc::new(s), d.bounds())
        });
        self.smart.composites.insert(uuid.to_string(), c.clone());
        c
    }

    /// Adds `bytes` as an embedded file (converted to PSB when it is a `.pcraft` bundle), once
    /// per contents identity and bytes. Equal bytes in independent objects stay independent.
    fn embed(&mut self, contents_id: SmartContentsId, file_name: &str, bytes: &[u8]) -> Result<Source, String> {
        let key = (contents_id, *blake3::hash(bytes).as_bytes());
        if let Some(r) = self.smart.converted.get(&key) {
            return r.clone();
        }
        let r = self.convert_source(contents_id, file_name, bytes);
        self.smart.converted.insert(key, r.clone());
        r
    }

    fn convert_source(&mut self, contents_id: SmartContentsId, file_name: &str, bytes: &[u8]) -> Result<Source, String> {
        let (name, data, size, dpi, nested) = if photocraft_format::is_pcraft(bytes) {
            if self.smart.depth >= MAX_NESTING {
                return Err(format!("smart objects nest more than {MAX_NESTING} deep"));
            }
            let doc = photocraft_format::load_from_bytes(bytes).map_err(|e| format!("its contents can't be read: {e}"))?;
            let (file, warnings) =
                document_to_psd_nested(&doc, &PsdExportOptions { force_psb: true, ..Default::default() }, self.smart.depth + 1, &Interrupt::NONE);
            let data = file.to_bytes().map_err(|e| format!("its contents can't be written: {e}"))?;
            let stem = file_name.rsplit_once('.').map_or(file_name, |(a, _)| a);
            self.warnings.extend(warnings.into_iter().map(|w| format!("smart object {stem}: {w}")));
            let size = (f64::from(doc.size.width), f64::from(doc.size.height));
            (format!("{stem}.psb"), data, size, f64::from(doc.resolution_dpi), Some(doc))
        } else {
            // Formats we can't decode (vector PDF/AI…) are still embedded; their size comes from the layer.
            let (size, dpi) =
                crate::linked::source_geometry(file_name, bytes, &photocraft_raster::Interrupt::NONE).unwrap_or(((0.0, 0.0), f64::from(self.dpi)));
            (file_name.to_string(), bytes.to_vec(), size, dpi, None)
        };
        let uuid = contents_uuid(contents_id, &data);
        if let Some(d) = nested {
            self.smart.docs.insert(uuid.clone(), d);
        }
        if !self.smart.known.contains(&uuid) && !self.smart.add.iter().any(|f| f.uuid == uuid) {
            self.smart.add.push(crate::linked::LinkedFile { uuid: uuid.clone(), file_name: name, bytes: data });
        }
        Ok(Source { uuid, size, dpi })
    }

    /// Pixels for a fill layer: Photoshop's cached rendering while valid,
    /// otherwise our own rendering of the fill over the canvas.
    fn fill_pixels(&self, l: &Layer, f: &photocraft_doc::Fill) -> Surface {
        if let Some(c) = &l.fill_cache
            && c.fill == *f
        {
            return c.surface.clone();
        }
        if self.cmyk
            && let Some(s) =
                photocraft_compose::gradient_fill::render_cmyk_fill(f, self.canvas, photocraft_compose::fill_frame(l, self.canvas), self.fmt.sample)
        {
            return s;
        }
        // In the frame the layer's masks give it, like the compositor (masks are stored apart).
        // Readers composite these pixels (ours keeps them as the fill's rendering), so a pattern
        // fill must be rendered from the document's patterns, not left transparent (#1907).
        let buf = photocraft_compose::render_fill_content(l, f, self.canvas, &self.patterns);
        let mut s = Surface::new(self.fmt);
        let vals: Vec<f32> = buf.px.iter().flat_map(|p| photocraft_raster::from_rgba(&self.fmt, *p)).collect();
        s.write_region(self.canvas, &vals);
        s.prune();
        s
    }

    /// Write the records of `layers` (and their groups' children). `tick` runs after each one:
    /// it reports progress and returns `false` to stop early (a cancelled save, #3389).
    fn emit(&mut self, layers: &[Layer], tick: &mut dyn FnMut() -> bool) {
        for l in layers {
            let extra = self.content_blocks(l);
            match &l.content {
                LayerContent::Group(g) => {
                    let id = self.fresh_id();
                    self.records.push(LayerRecord {
                        channels: self.empty_channels(),
                        blending_ranges: BlendingRanges::full(self.cc),
                        name: b"</Layer group>".to_vec(),
                        blocks: vec![
                            TaggedBlock::unicode_name("</Layer group>"),
                            TaggedBlock::layer_id(id),
                            TaggedBlock::section_divider(SectionType::BoundingDivider, None, None),
                        ],
                        ..Default::default()
                    });
                    self.emit(&g.children, tick);
                    let kind = if g.expanded { SectionType::OpenFolder } else { SectionType::ClosedFolder };
                    let lsct = TaggedBlock::section_divider(kind, Some(PsdBlend::from_key(l.blend.psd_key())), None);
                    let r = self.record(l, std::iter::once(lsct).chain(extra).collect(), None);
                    self.records.push(r);
                }
                LayerContent::Raster(s) => {
                    let r = self.record(l, extra, Some(s));
                    self.records.push(r);
                }
                LayerContent::Adjustment(_) => {
                    let r = self.record(l, extra, None);
                    self.records.push(r);
                }
                LayerContent::Fill(f) => {
                    let px = self.fill_pixels(l, f);
                    let r = self.record(l, extra, Some(&px));
                    self.records.push(r);
                }
                LayerContent::Text(t) => {
                    let r = self.record(l, extra, t.cache.as_ref());
                    self.records.push(r);
                }
                LayerContent::Shape(sh) => {
                    let r = self.record(l, extra, sh.cache.as_ref());
                    self.records.push(r);
                }
                LayerContent::Smart(sm) => {
                    let r = self.record(l, extra, sm.cache.as_ref());
                    self.records.push(r);
                }
            }
            if !tick() {
                return;
            }
        }
    }
}

/// Resource 1026 (Layer › Link Layers): one u16 group id per layer record, in the order
/// [`Ex::emit`] writes records (a group's bounding divider, its children, then the group).
/// Imported ids that fit in a u16 are kept, so unchanged files write the same bytes; otherwise
/// groups are renumbered by first appearance. `None` when no layer is linked.
fn link_group_resource(layers: &[Layer]) -> Option<Vec<u8>> {
    fn walk(layers: &[Layer], out: &mut Vec<Option<u64>>) {
        for l in layers {
            if let LayerContent::Group(g) = &l.content {
                out.push(None);
                walk(&g.children, out);
            }
            out.push(l.link_group);
        }
    }
    let mut per = Vec::new();
    walk(layers, &mut per);
    if per.iter().all(Option::is_none) {
        return None;
    }
    let fits = per.iter().flatten().all(|&g| (1..=u64::from(u16::MAX)).contains(&g));
    let mut seen: Vec<u64> = Vec::new();
    let ids = per.iter().map(|g| match *g {
        None => 0u16,
        Some(g) if fits => g as u16,
        Some(g) => {
            let i = seen.iter().position(|&s| s == g).unwrap_or_else(|| {
                seen.push(g);
                seen.len() - 1
            });
            u16::try_from(i + 1).unwrap_or(u16::MAX)
        }
    });
    Some(ids.flat_map(u16::to_be_bytes).collect())
}

/// Whether the layer's effects still equal what its preserved `lmfx`/`lfx2`
/// decodes to. `None` when there is nothing preserved or it can't be decoded.
/// Block keys a smart object's `psd_raw` may be written under, in priority order. The data says
/// which it is: `PlLd` holds a `plcL` structure, `SoLd`/`SoLE` a `soLD` one (Adobe PSD spec,
/// "Placed Layer" and "Placed Layer Data"). Writing one under the other's key makes the block
/// unreadable.
fn smart_keys(data: Option<&[u8]>) -> &'static [&'static [u8; 4]] {
    match data.and_then(|d| d.get(..4)) {
        Some(b"plcL") => &[b"PlLd"],
        Some(b"soLD") => &[b"SoLd", b"SoLE"],
        _ => &[b"SoLd", b"PlLd", b"SoLE"],
    }
}

fn effects_unchanged(l: &Layer) -> Option<bool> {
    let src = l.psd_blocks.iter().find(|(k, _)| k == b"lmfx").map(|(_, d)| d.clone()).or_else(|| l.effects.psd_raw.clone())?;
    let (m, items) = crate::effects_map::parse_lfx2(&src)?;
    Some(m == l.effects.enabled && items == l.effects.items)
}

pub(crate) fn legacy_name(s: &str) -> Vec<u8> {
    s.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c as u8 } else { b'?' }).take(255).collect()
}

pub(crate) fn unicode_names_resource(names: &[&str]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in names {
        let units: Vec<u16> = n.encode_utf16().chain(std::iter::once(0)).collect();
        v.extend_from_slice(&(units.len() as u32).to_be_bytes());
        for u in units {
            v.extend_from_slice(&u.to_be_bytes());
        }
    }
    v
}

pub(crate) fn pascal_names_resource(names: &[&str]) -> Vec<u8> {
    let mut v = Vec::new();
    for n in names {
        let b = legacy_name(n);
        v.push(b.len() as u8);
        v.extend_from_slice(&b);
    }
    v
}

pub(crate) fn guides_resource(doc: &Document) -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&1u32.to_be_bytes());
    v.extend_from_slice(&576u32.to_be_bytes());
    v.extend_from_slice(&576u32.to_be_bytes());
    let n = doc.guides.horizontal.len() + doc.guides.vertical.len();
    v.extend_from_slice(&(n as u32).to_be_bytes());
    for (list, dir) in [(&doc.guides.vertical, 0u8), (&doc.guides.horizontal, 1u8)] {
        for p in list {
            v.extend_from_slice(&((p * 32.0).round() as i32).to_be_bytes());
            v.push(dir);
        }
    }
    v
}

/// Path resources: saved paths (2000+), the work path (1025) and the clipping path name (2999),
/// the last only when the imported one doesn't already name the document's clipping path.
pub(crate) fn path_resources(doc: &Document, warnings: &mut Vec<String>) -> Vec<ImageResource> {
    use crate::vector_map::{CLIPPING_PATH, WORK_PATH, path_from_resource, path_to_records};
    let mut out = Vec::new();
    let (w, h) = (doc.size.width, doc.size.height);
    let max = (*crate::vector_map::SAVED_PATHS.end() - *crate::vector_map::SAVED_PATHS.start() + 1) as usize;
    if doc.paths.len() > max {
        warnings.push(format!("only {max} saved paths fit in PSD; the rest were dropped"));
    }
    for (i, p) in doc.paths.iter().take(max).enumerate() {
        let data = match &p.psd_raw {
            Some(r) if path_from_resource(r, w, h).as_ref() == Some(&p.path) => r.to_vec(),
            _ => path_to_records(&p.path, w, h).to_bytes(),
        };
        let mut r = ImageResource::new(crate::vector_map::SAVED_PATHS.start() + i as u16, data);
        r.name = legacy_name(&p.name);
        out.push(r);
    }
    if let Some(wp) = &doc.work_path {
        out.push(ImageResource::new(WORK_PATH, path_to_records(wp, w, h).to_bytes()));
    }
    let raw_clip = doc.metadata.psd_resources.iter().find(|(id, _, _)| *id == CLIPPING_PATH);
    let raw_name = raw_clip.map(|(_, _, d)| {
        let n = usize::from(d.first().copied().unwrap_or(0));
        d.get(1..1 + n).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default()
    });
    if let Some(c) = &doc.clipping_path
        && raw_name.as_deref() != Some(c.name.as_str())
    {
        // Pascal name, then flatness as 16.16 fixed (best effort; see vector_map docs).
        let mut data = vec![0u8];
        let name = legacy_name(&c.name);
        data[0] = name.len() as u8;
        data.extend_from_slice(&name);
        data.extend_from_slice(&((c.flatness.max(0.0) * 65536.0) as u32).to_be_bytes());
        out.push(ImageResource::new(CLIPPING_PATH, data));
    }
    out
}

/// Whether an imported (unmodelled) resource still describes the document and is written back.
pub(crate) fn keep_imported_resource(doc: &Document, id: u16, data: &[u8]) -> bool {
    // Measurement scale / count resources: only while they still describe the document.
    if !crate::annotations_map::keep_resource(doc, id) {
        return false;
    }
    // A preserved clipping-path resource is only valid while it names the current one.
    if id == crate::vector_map::CLIPPING_PATH {
        let n = usize::from(data.first().copied().unwrap_or(0));
        let raw_name = data.get(1..1 + n).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
        return doc.clipping_path.as_ref().is_some_and(|c| c.name == raw_name);
    }
    true
}

/// A document-level (global) tagged block. Photoshop pads these to a multiple of 4, and readers
/// such as psd-tools step to the next block that way: an even pad after `CAI ` (77 bytes)
/// misaligned every block after it (#200).
pub(crate) fn global_block(signature: [u8; 4], key: [u8; 4], data: &[u8]) -> TaggedBlock {
    let mut tb = TaggedBlock::new(key, data.to_vec());
    tb.signature = signature;
    tb.padding = Some(vec![0; (4 - data.len() % 4) % 4]);
    tb
}

fn cmyk_of(fmt: &PixelFormat) -> bool {
    fmt.mode == ColorMode::Cmyk
}

/// The merged image's planes (colour channels, then alpha), big-endian, matted against white
/// when `matte`; with whether any pixel needs the alpha channel and whether any is below 1.
/// `ctl` gets the fraction of rows done after each band; once it is cancelled the rest of the
/// image is left blank (the caller discards it).
fn merged_planes(doc: &Document, fmt: &PixelFormat, cmyk: bool, matte: bool, ctl: &Interrupt) -> (Vec<u8>, bool, bool) {
    let sample = fmt.sample;
    let cc = fmt.mode.color_channels();
    let bps = sample.bytes();
    let n = doc.size.area() as usize;
    let plane = n * bps;
    let mut planes = vec![0u8; plane * (cc + 1)];
    let (mut has_alpha, mut translucent) = (false, false);
    let white = photocraft_raster::from_rgba(fmt, [1.0, 1.0, 1.0, 1.0]);
    let canvas = doc.bounds();
    let w = canvas.width() as usize;
    let space = photocraft_compose::cmyk_space(doc);
    let lab16 = fmt.mode == ColorMode::Lab && sample == SampleType::U16;
    let rows = canvas.height().max(1) as f32;
    let _ = photocraft_compose::render_bands(doc, canvas, 0, |band| -> Result<(), ()> {
        if ctl.cancelled() {
            return Err(());
        }
        // Retain alpha whenever it differs from opaque at the stored precision.
        has_alpha |= band.px.iter().any(|p| match sample {
            SampleType::U8 => q255(p[3]) < 255,
            SampleType::U16 => (p[3].clamp(0.0, 1.0) * 65535.0 + 0.5) as u16 != u16::MAX,
            SampleType::F32 => p[3] != 1.0,
        });
        translucent |= band.px.iter().any(|p| p[3] < 1.0);
        let start = (band.rect.y0 - canvas.y0) as usize * w;
        // Converted and encoded on all cores, then copied into each plane.
        let parts = crate::pixels::par_map(crate::pixels::bands(band.px.len()), |range| {
            // The composite came through the document's CMYK profile: convert back through it too.
            photocraft_color::convert::with_cmyk_space(space.as_ref(), || {
                let mut out: Vec<Vec<u8>> = vec![Vec::with_capacity(range.len() * bps); cc + 1];
                let mut v = [0.0f32; 5];
                for p in &band.px[range.clone()] {
                    photocraft_raster::from_rgba_into(fmt, *p, &mut v);
                    for c in 0..=cc {
                        // Matte against white like Photoshop (see `pixels::matte`).
                        let m = if c < cc && matte { crate::pixels::matte(v[c], v[cc], white[c]) } else { v[c] };
                        let x = if cmyk && c < cc { 1.0 - m } else { m };
                        // 16-bit Lab a*/b* use Photoshop's 0..65280 scale (`pixels::lab16_chroma`).
                        let x = if lab16 && (c == 1 || c == 2) { x * (crate::pixels::LAB16_CHROMA_MAX / 65535.0) } else { x };
                        encode_be(x, sample, &mut out[c]);
                    }
                }
                (range.start, out)
            })
        });
        for (at, part) in parts {
            for (c, src) in part.iter().enumerate() {
                let o = c * plane + (start + at) * bps;
                planes[o..o + src.len()].copy_from_slice(src);
            }
        }
        ctl.progress((band.rect.y1 - canvas.y0) as f32 / rows);
        Ok(())
    });
    (planes, has_alpha, translucent)
}

/// The merged image of a file saved without Maximize Compatibility: Photoshop's white placeholder
/// in the document's colour channels, with no alpha. Encoded like [`merged_planes`], so CMYK white
/// is no ink and Lab white is L 100 with neutral a*/b*.
fn placeholder_planes(doc: &Document, fmt: &PixelFormat) -> Vec<u8> {
    let cc = fmt.mode.color_channels();
    let cmyk = cmyk_of(fmt);
    let lab16 = fmt.mode == ColorMode::Lab && fmt.sample == SampleType::U16;
    let space = photocraft_compose::cmyk_space(doc);
    let white = photocraft_color::convert::with_cmyk_space(space.as_ref(), || photocraft_raster::from_rgba(fmt, [1.0, 1.0, 1.0, 1.0]));
    let n = doc.size.area() as usize;
    let mut planes = Vec::with_capacity(n * fmt.sample.bytes() * cc);
    for c in 0..cc {
        let v = white.get(c).copied().unwrap_or(1.0);
        let x = if cmyk { 1.0 - v } else { v };
        let x = if lab16 && (c == 1 || c == 2) { x * (crate::pixels::LAB16_CHROMA_MAX / 65535.0) } else { x };
        let mut sample = Vec::with_capacity(4);
        encode_be(x, fmt.sample, &mut sample);
        planes.extend(sample.iter().copied().cycle().take(n * sample.len()));
    }
    planes
}

/// Converts a document to a PSD file model (see [`document_to_psd_with`]).
pub fn document_to_psd(doc: &Document) -> PsdFile {
    document_to_psd_with(doc, &PsdExportOptions::default()).0
}

/// Converts a document to a PSD/PSB file model, returning warnings about
/// anything that could not be represented. The merged composite is rendered
/// with `photocraft_compose::flatten`.
pub fn document_to_psd_with(doc: &Document, opts: &PsdExportOptions) -> (PsdFile, Vec<String>) {
    document_to_psd_nested(doc, opts, 0, &Interrupt::NONE)
}

/// [`document_to_psd_with`] for a background save (#3389): reports progress through `ctl` after
/// each layer record and each band of the merged composite. Once `ctl` is cancelled it stops
/// early and returns an **incomplete** model: callers check `ctl` and discard it.
pub(crate) fn document_to_psd_interruptible(doc: &Document, opts: &PsdExportOptions, ctl: &Interrupt) -> (PsdFile, Vec<String>) {
    document_to_psd_nested(doc, opts, 0, ctl)
}

/// How much of a PSD conversion the merged composite weighs, in layer records: rendering it
/// (all layers composited) and encoding its planes cost about two full-canvas layers.
const MERGED_WEIGHT: f32 = 2.0;
/// The part of the merged composite's share filled once its bands are rendered and converted; the
/// rest is compressing its planes.
const MERGED_RENDERED: f32 = 0.8;

/// Adds headroom for compressed channel streams and their per-row length tables.
fn estimate_encoded_size(raw_bytes: u64, rows: u64, structural_bytes: u64) -> Option<u64> {
    raw_bytes.checked_add(raw_bytes.checked_add(63)?.checked_div(64)?)?.checked_add(rows.checked_mul(8)?)?.checked_add(structural_bytes)
}

fn add_plane_estimate(raw_bytes: &mut u64, rows: &mut u64, width: u64, height: u64, channels: u64, bytes_per_sample: u64) -> Option<()> {
    let plane_bytes = width.checked_mul(height)?.checked_mul(channels)?.checked_mul(bytes_per_sample)?;
    let plane_rows = height.checked_mul(channels)?;
    *raw_bytes = raw_bytes.checked_add(plane_bytes)?;
    *rows = rows.checked_add(plane_rows)?;
    Some(())
}

fn exceeds_psd_size_limit(estimated_size: Option<u64>) -> bool {
    estimated_size.is_none_or(|size| size > PSD_MAX_FILE_SIZE)
}

fn surface_dimensions(surface: &Surface) -> Option<(u64, u64)> {
    let bounds = surface.content_bounds();
    Some((u64::from(bounds.width()), u64::from(bounds.height())))
}

pub(crate) fn estimate_psd_size(doc: &Document) -> Option<u64> {
    let mut raw_bytes = 0;
    let mut rows = 0;
    let mut structural_bytes = PSD_ESTIMATE_FIXED_OVERHEAD;

    let width = u64::from(doc.size.width);
    let height = u64::from(doc.size.height);

    if doc.mode == ColorMode::Multichannel {
        let channels = u64::try_from(doc.channels.len().clamp(1, 56)).ok()?;
        let sample_bytes = u64::try_from(if doc.depth == SampleType::F32 { SampleType::U16.bytes() } else { doc.depth.bytes() }).ok()?;
        add_plane_estimate(&mut raw_bytes, &mut rows, width, height, channels, sample_bytes)?;
    } else {
        let format = doc.pixel_format();
        let color_channels = u64::try_from(format.mode.color_channels()).ok()?;
        let bytes_per_sample = u64::try_from(format.sample.bytes()).ok()?;
        let layer_channels = color_channels.checked_add(1)?;

        for (_, _, layer) in doc.walk() {
            structural_bytes = structural_bytes.checked_add(PSD_ESTIMATE_LAYER_OVERHEAD)?.checked_add(u64::try_from(layer.name.len()).ok()?)?;
            for (_, data) in &layer.psd_blocks {
                structural_bytes = structural_bytes.checked_add(u64::try_from(data.len()).ok()?)?;
            }

            let surface = match &layer.content {
                LayerContent::Fill(fill) => {
                    let mut fill_width = width;
                    let mut fill_height = height;
                    if let Some(cache) = &layer.fill_cache
                        && cache.fill == *fill
                    {
                        let (cache_width, cache_height) = surface_dimensions(&cache.surface)?;
                        fill_width = fill_width.max(cache_width);
                        fill_height = fill_height.max(cache_height);
                    }
                    add_plane_estimate(&mut raw_bytes, &mut rows, fill_width, fill_height, layer_channels, bytes_per_sample)?;
                    None
                }
                _ => layer.surface(),
            };
            if let Some(surface) = surface {
                let (surface_width, surface_height) = surface_dimensions(surface)?;
                add_plane_estimate(&mut raw_bytes, &mut rows, surface_width, surface_height, layer_channels, bytes_per_sample)?;
            }
            if let Some(mask) = &layer.mask {
                let (mask_width, mask_height) = surface_dimensions(&mask.surface)?;
                add_plane_estimate(&mut raw_bytes, &mut rows, mask_width, mask_height, 1, bytes_per_sample)?;
            }
            if let LayerContent::Smart(smart) = &layer.content
                && let photocraft_doc::SmartSource::Embedded { bytes, .. } = &smart.source
            {
                structural_bytes = structural_bytes.checked_add(u64::try_from(bytes.len()).ok()?)?;
            }
        }

        let max_extra_channels = 56usize.saturating_sub(format.mode.color_channels() + 1);
        let mut extra_channels = doc.channels.len().min(max_extra_channels);
        if doc.quick_mask.is_some() && extra_channels < max_extra_channels {
            extra_channels += 1;
        }
        let merged_channels = color_channels.checked_add(1)?.checked_add(u64::try_from(extra_channels).ok()?)?;
        add_plane_estimate(&mut raw_bytes, &mut rows, width, height, merged_channels, bytes_per_sample)?;
        for channel in doc.channels.iter().take(extra_channels) {
            structural_bytes = structural_bytes.checked_add(u64::try_from(channel.name.len()).ok()?)?;
        }
    }

    structural_bytes = structural_bytes
        .checked_add(u64::try_from(doc.icc_profile.as_ref().map_or(0, |profile| profile.len())).ok()?)?
        .checked_add(u64::try_from(doc.metadata.xmp.as_ref().map_or(0, String::len)).ok()?)?
        .checked_add(u64::try_from(doc.metadata.exif.as_ref().map_or(0, |data| data.len())).ok()?)?;
    for (_, name, data) in &doc.metadata.psd_resources {
        structural_bytes = structural_bytes.checked_add(u64::try_from(name.len()).ok()?)?.checked_add(u64::try_from(data.len()).ok()?)?;
    }
    for (_, _, data) in &doc.metadata.psd_global_blocks {
        structural_bytes = structural_bytes.checked_add(u64::try_from(data.len()).ok()?)?;
    }

    estimate_encoded_size(raw_bytes, rows, structural_bytes)
}

/// The signature a kept tagged block is written with. `8B64` exists only in PSB files: a PSD
/// with an `8B64` block (kept from a PSB source, e.g. its global `FMsk` or `cinf`) is refused by
/// Photoshop as "not compatible with this version" (measured in Photoshop 27.11).
fn block_signature(sig: [u8; 4], version: Version) -> [u8; 4] {
    if version == Version::Psd && sig == *b"8B64" { *b"8BIM" } else { sig }
}

pub(crate) fn psd_version(doc: &Document, force_psb: bool, size_exceeds_limit: bool) -> Version {
    let exceeds_dimensions = doc.size.width > 30_000 || doc.size.height > 30_000;
    if force_psb || exceeds_dimensions || size_exceeds_limit { Version::Psb } else { Version::Psd }
}

pub(crate) fn psd_size_exceeds_limit(doc: &Document) -> bool {
    exceeds_psd_size_limit(estimate_psd_size(doc))
}

/// [`document_to_psd_with`] for a document embedded `depth` smart objects deep.
fn document_to_psd_nested(doc: &Document, opts: &PsdExportOptions, depth: u32, ctl: &Interrupt) -> (PsdFile, Vec<String>) {
    let mut export = std::borrow::Cow::Borrowed(doc);
    for (_, _, layer) in doc.walk() {
        let LayerContent::Text(text) = &layer.content else { continue };
        if text.overflow_visible || text.block_align != photocraft_doc::text::BlockAlign::Start {
            let layout = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).layout(text, doc.resolution_dpi);
            if let Some(fitted) = photocraft_text::psd::visible_overflow_for_psd(text, &layout)
                && let Some(target) = export.to_mut().layer_mut(layer.id)
            {
                target.content = LayerContent::Text(fitted);
            }
        }
    }
    let doc = export.as_ref();
    if doc.mode == ColorMode::Multichannel {
        return crate::multichannel_map::document_to_psd(doc, opts.force_psb);
    }
    let fmt = doc.pixel_format();
    let sample = fmt.sample;
    let cc = fmt.mode.color_channels();
    let big = doc.size.width > 30_000 || doc.size.height > 30_000;
    let size_exceeds_limit = !opts.force_psb && !big && psd_size_exceeds_limit(doc);
    let version = psd_version(doc, opts.force_psb, size_exceeds_limit);
    let mut ex = Ex {
        fmt,
        dpi: doc.resolution_dpi,
        text_styles: doc.text_styles.clone(),
        text_index: Default::default(),
        mask_fmt: PixelFormat::new(ColorMode::Grayscale, sample, false),
        cc,
        cmyk: fmt.mode == ColorMode::Cmyk,
        version,
        next_id: 0,
        used_ids: Default::default(),
        layer_ids: Default::default(),
        canvas: doc.bounds(),
        records: Vec::new(),
        warnings: Vec::new(),
        guides: doc.guides.clone(),
        comps: (!crate::comps_map::comps_unchanged(doc)).then(|| (doc.layer_comps.clone(), doc.last_document_state.clone())),
        smart: SmartOut::new(doc, depth),
        patterns: doc.patterns.clone(),
        background: psd_background(doc),
    };
    if big && !opts.force_psb {
        ex.warnings.push("document exceeds 30000 px; written as PSB".into());
    } else if size_exceeds_limit && !opts.force_psb {
        ex.warnings.push("estimated encoded size exceeds 2 GB; written as PSB".into());
    }
    if fmt.mode != doc.mode {
        ex.warnings.push(format!("{:?} document written as {:?}", doc.mode, fmt.mode));
    }
    // Assign layer ids up front (stable across round trips): keep unique
    // `psd_id`s, number the rest after them; dividers come after all.
    {
        let walk = doc.walk();
        let mut used = std::collections::HashSet::new();
        for (_, _, l) in &walk {
            if let Some(id) = l.psd_id
                && used.insert(id)
            {
                ex.layer_ids.insert(l.id, id);
            }
        }
        let mut next = 1u32;
        for (_, _, l) in &walk {
            if let std::collections::hash_map::Entry::Vacant(e) = ex.layer_ids.entry(l.id) {
                while used.contains(&next) {
                    next += 1;
                }
                used.insert(next);
                e.insert(next);
            }
        }
        ex.next_id = used.iter().copied().max().unwrap_or(0);
        ex.used_ids = used;
        // Text objects' number in the generated `Txt2` (#1348): a preserved `TySh` keeps the
        // number it already names — its `Txt2` object must sit at that slot for `apply_txt2` —
        // and new layers get the next free number.
        let mut used_txt = std::collections::HashSet::new();
        let mut next_txt = 0i32;
        ex.text_index = Default::default();
        for (_, _, l) in &walk {
            let LayerContent::Text(t) = &l.content else {
                continue;
            };
            let kept = t.psd_raw.as_deref().and_then(|d| photocraft_text::psd::text_index(d));
            let index = match kept {
                Some(i) if used_txt.insert(i) => i,
                _ => {
                    while used_txt.contains(&next_txt) {
                        next_txt += 1;
                    }
                    used_txt.insert(next_txt);
                    next_txt
                }
            };
            ex.text_index.insert(l.id, index);
        }
    }
    // Progress: one unit per layer record, then the merged composite (#3389).
    let records = doc.walk().len() as f32;
    let units = records + MERGED_WEIGHT;
    let layers_share = records / units;
    let mut done = 0.0f32;
    let mut tick = || {
        done += 1.0;
        ctl.progress(done / units);
        !ctl.cancelled()
    };
    let space = photocraft_compose::cmyk_space(doc);
    photocraft_color::convert::with_cmyk_space(space.as_ref(), || ex.emit(&doc.layers, &mut tick));

    // Without Maximize Compatibility the merged image is a white placeholder and the composite
    // isn't rendered at all. A document without layer records keeps the real image: its pixels
    // live nowhere else.
    let real_merged = opts.maximize_compatibility || ex.records.is_empty();
    let (mut planes, has_alpha) = if real_merged {
        // Merged composite, rendered and encoded in bands (no full-size float composite). Matting
        // against white only changes pixels with alpha < 1; if some are slightly translucent but
        // all round to opaque (so no alpha channel is written), encode once more without the matte.
        let merged_end = layers_share + (1.0 - layers_share) * MERGED_RENDERED;
        let (mut planes, has_alpha, translucent) =
            crate::with_span(ctl, layers_share, merged_end, |ctl| merged_planes(doc, &fmt, cmyk_of(&fmt), opts.merged_matte, ctl));
        if !has_alpha && translucent && opts.merged_matte {
            planes = merged_planes(doc, &fmt, cmyk_of(&fmt), false, &Interrupt::cancel_only(&|| ctl.cancelled())).0;
        }
        (planes, has_alpha)
    } else {
        (placeholder_planes(doc, &fmt), false)
    };
    let n = doc.size.area() as usize;
    if !has_alpha {
        planes.truncate(cc * n * sample.bytes());
    }
    let canvas = doc.bounds();
    let max_extra = 56 - cc - usize::from(has_alpha);
    if doc.channels.len() > max_extra {
        ex.warnings.push(format!("only {max_extra} alpha channels fit in PSD; the rest were dropped"));
    }
    let mut extra: Vec<_> = doc.channels.iter().take(max_extra).collect();
    // Saved in Quick Mask mode: the mask is written as the last extra channel and flagged by
    // resource 1022 (quick mask info: channel id, initially-empty flag), as Photoshop does.
    let quick_mask_id = match &doc.quick_mask {
        Some(q) if extra.len() < max_extra => {
            extra.push(q);
            Some((cc + usize::from(has_alpha) + extra.len() - 1) as u16)
        }
        Some(_) => {
            ex.warnings.push("Quick Mask was dropped because the PSD channel limit was reached".into());
            None
        }
        None => None,
    };
    for a in &extra {
        let s = if a.surface.format() != ex.mask_fmt { a.surface.convert(ex.mask_fmt) } else { a.surface.clone() };
        let bytes = s.to_interleaved(canvas);
        planes.extend(deinterleave(&bytes, 1, sample, &[false]).remove(0));
    }
    let channels = (cc + usize::from(has_alpha) + extra.len()) as u16;
    let header = Header::new(version, doc.size.width, doc.size.height, channels, psd_depth(sample), psd_mode(fmt.mode));
    let mcomp = if sample == SampleType::F32 { Compression::Raw } else { opts.merged_compression };
    let image_data = ImageData::encode(mcomp, &planes, &header)
        .or_else(|_| ImageData::encode(Compression::Raw, &planes, &header))
        .unwrap_or(ImageData { compression: Compression::Raw, data: planes });
    ctl.progress(1.0);

    // Resources.
    let mut resources = vec![
        // hRes/vRes are always pixels per inch. The display unit written here is inches.
        ImageResource::new(ids::RESOLUTION_INFO, ResolutionInfo::from_dpi(f64::from(doc.resolution_dpi)).to_bytes()),
        ImageResource::new(ids::GLOBAL_ANGLE, (doc.global_light.angle.round() as i32).to_be_bytes().to_vec()),
        ImageResource::new(ids::GLOBAL_ALTITUDE, (doc.global_light.altitude.round() as i32).to_be_bytes().to_vec()),
    ];
    if let Some(icc) = &doc.icc_profile {
        resources.push(ImageResource::new(ids::ICC_PROFILE, icc.to_vec()));
    }
    if !doc.guides.horizontal.is_empty() || !doc.guides.vertical.is_empty() {
        resources.push(ImageResource::new(1032, guides_resource(doc)));
    }
    if !extra.is_empty() {
        // One entry per extra channel, the merged transparency included (as in every file
        // Photoshop opens; one short, Photoshop refuses the file with "unexpected end-of-file").
        let names: Vec<&str> = has_alpha.then_some("Transparency").into_iter().chain(extra.iter().map(|a| a.name.as_str())).collect();
        resources.push(ImageResource::new(1006, pascal_names_resource(&names)));
        resources.push(ImageResource::new(1045, unicode_names_resource(&names)));
        resources.push(ImageResource::new(crate::channel_map::DISPLAY_INFO, crate::channel_map::display_info(&extra, has_alpha)));
    }
    if let Some(id) = quick_mask_id {
        let mut data = id.to_be_bytes().to_vec();
        data.push(0);
        resources.push(ImageResource::new(crate::channel_map::QUICK_MASK_INFO, data));
    }
    resources.extend(path_resources(doc, &mut ex.warnings));
    if let Some(groups) = link_group_resource(&doc.layers) {
        resources.push(ImageResource::new(ids::LAYER_GROUP_INFO, groups));
    }
    // The pixels are saved as they are shown: never let a reader rotate them again. The XMP and
    // EXIF resolution follow the ResolutionInfo resource, so no copy contradicts it (#1691).
    let ppi = Some((doc.resolution_dpi, doc.resolution_dpi)).filter(|d| d.0 > 0.0);
    if let Some(x) = &doc.metadata.xmp {
        resources.push(ImageResource::new(ids::XMP, photocraft_codecs::export_xmp(x, ppi).as_bytes().to_vec()));
    }
    if let Some(e) = &doc.metadata.exif {
        resources.push(ImageResource::new(ids::EXIF, photocraft_codecs::export_exif(e, ppi).into_owned()));
    }
    let mut global_blocks = Vec::new();
    let preserved = ex.smart.finish(crate::annotations_map::export_blocks(doc, crate::pattern_map::export_global_blocks(doc)));
    // The type layers' `Txt2` is regenerated (the auto-kern modes live there, and a preserved
    // block goes stale as text is edited): one object per `TextIndex`, at the slot each layer's
    // `TySh` names (unused slots stay empty; see `photocraft_text::psd::build_txt2`). The block
    // it replaces is handed over so a kept object keeps everything the file held beyond its text
    // and style runs (Photoshop's `/21 /1` glyph pen positions, …).
    let text_layers: Vec<(i32, &photocraft_doc::TextLayer)> = doc
        .walk()
        .into_iter()
        .filter_map(|(_, _, l)| match &l.content {
            LayerContent::Text(t) => Some((ex.text_index.get(&l.id).copied().unwrap_or(0), t)),
            _ => None,
        })
        .collect();
    let previous = (!text_layers.is_empty()).then(|| preserved.iter().find(|(_, key, _)| key == b"Txt2").map(|(_, _, data)| data.to_vec())).flatten();
    // A generated Txt2 contains kerning runs but not Photoshop's document/font/frame model.
    // Photoshop rejects that incomplete block on open. Only update an existing model;
    // new text stays editable through TySh, with Optical falling back to Metrics on edit.
    if previous.is_none() && photocraft_text::psd::needs_txt2(&text_layers) {
        ex.warnings.push("New optical text uses Metrics kerning when edited in Photoshop; its saved appearance and editable text are preserved.".into());
    }
    let txt2 = previous.as_deref().map(|data| photocraft_text::psd::build_txt2(&text_layers, Some(data)));
    for (sig, key, data) in preserved {
        if txt2.is_some() && key == *b"Txt2" {
            continue;
        }
        // A kept layer block (one the import could not read, #2700) describes layers the
        // document doesn't have. The document's own layers are written below, and a file holds
        // one layer list: a second `Lr16`/`Lr32` would contradict it.
        if !ex.records.is_empty() && matches!(&key, b"Lr16" | b"Lr32" | b"Layr") {
            continue;
        }
        global_blocks.push(global_block(block_signature(sig, version), key, &data));
    }
    if let Some(txt2) = txt2 {
        let mut tb = TaggedBlock::new(*b"Txt2", txt2.clone());
        tb.signature = *b"8BIM";
        tb.padding = Some(vec![0; (4 - txt2.len() % 4) % 4]);
        global_blocks.push(tb);
    }
    let comps_resource = ex.comps.is_some().then(|| crate::comps_map::write_comps_resource(doc));
    let (slices_resource, slices_warning) = crate::slices_map::export_resource(doc, &ex.layer_ids);
    if let Some(warning) = slices_warning {
        ex.warnings.push(warning);
    }
    for (id, name, data) in &doc.metadata.psd_resources {
        // Layer comps: the preserved list while unchanged, else regenerated (or dropped) below.
        if *id == crate::comps_map::LAYER_COMPS && comps_resource.is_some() {
            continue;
        }
        // Slices: the preserved resource while unchanged, else regenerated (or dropped) below.
        if *id == crate::slices_map::SLICES && slices_resource.is_some() {
            continue;
        }
        if !keep_imported_resource(doc, *id, data) {
            continue;
        }
        let mut r = ImageResource::new(*id, data.to_vec());
        r.name = legacy_name(name);
        resources.push(r);
    }
    if let Some(data) = crate::annotations_map::fresh_scale_resource(doc) {
        resources.push(ImageResource::new(crate::annotations_map::MEASUREMENT_SCALE, data));
    }
    if let Some(Some(data)) = comps_resource {
        resources.push(ImageResource::new(crate::comps_map::LAYER_COMPS, data));
    }
    if let Some(Some(data)) = slices_resource {
        resources.push(ImageResource::new(crate::slices_map::SLICES, data));
    }
    resources.push(version_info_resource(real_merged));

    let has_layers = !ex.records.is_empty();
    let placement = match (has_layers, sample) {
        (true, SampleType::U16) => LayerInfoPlacement::GlobalBlock { index: 0, signature: *b"8BIM", key: *b"Lr16", padding: None },
        (true, SampleType::F32) => LayerInfoPlacement::GlobalBlock { index: 0, signature: *b"8BIM", key: *b"Lr32", padding: None },
        _ => LayerInfoPlacement::Section,
    };
    let records = std::mem::take(&mut ex.records);
    let mut layer_info = has_layers.then_some(LayerInfo { merged_alpha: has_alpha, layers: records, padding: None });
    if let Some(info) = &mut layer_info {
        // Photoshop pads the layer info to 4 bytes (an even pad misaligns readers after a global
        // `Lr16`/`Lr32` block, #200). A length that cannot be computed keeps the even default.
        // A global block's length is the exact data length and its padding to 4 follows the block:
        // readers that check that length exactly (Unity's PSD Importer) reject it otherwise (#4654).
        if matches!(placement, LayerInfoPlacement::GlobalBlock { .. }) {
            info.padding = Some(Vec::new());
        } else if info.pad_to(version, 4).is_err() {
            info.padding = None;
        }
    }
    // 32-bit files carry Photoshop's HDR toning records as Color Mode Data; Photoshop will not
    // open a 32-bit file without them (#291).
    let color_mode_data = photocraft_psd::hdr::color_mode_data_for_depth(header.depth);
    let file = PsdFile {
        header,
        color_mode_data,
        resources,
        layer_info,
        layer_info_placement: placement,
        global_layer_mask: (has_layers || !global_blocks.is_empty()).then(GlobalLayerMask::default),
        global_blocks,
        layer_mask_trailing: Vec::new(),
        image_data,
    };
    (file, ex.warnings)
}

/// PSD mask parameters for a user mask's and a vector mask's (density, feather); `None` when
/// every value is the default (density 1, feather 0).
fn mask_parameters(user: Option<(f32, f32)>, vector: Option<(f32, f32)>) -> Option<MaskParameters> {
    let density = |v: Option<(f32, f32)>| v.and_then(|(d, _)| (d < 1.0).then(|| q255(d)));
    let feather = |v: Option<(f32, f32)>| v.and_then(|(_, f)| (f != 0.0).then_some(f64::from(f)));
    let p =
        MaskParameters { flags: 0, user_density: density(user), user_feather: feather(user), vector_density: density(vector), vector_feather: feather(vector) };
    let flags = u8::from(p.user_density.is_some())
        | u8::from(p.user_feather.is_some()) << 1
        | u8::from(p.vector_density.is_some()) << 2
        | u8::from(p.vector_feather.is_some()) << 3;
    (flags != 0).then_some(MaskParameters { flags, ..p })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document(width: u32, height: u32) -> Document {
        Document::new("size estimate", photocraft_geom::Size::new(width, height), ColorMode::Rgb, SampleType::U8)
    }

    /// Photoshop opens a PSD's bottom layer as its Background only when that layer has no
    /// transparency channel (measured on Photoshop 27.11: with the channel written, re-exported
    /// files reopened with the Background as an ordinary layer).
    #[test]
    fn a_background_is_written_without_a_transparency_channel() {
        use photocraft_color::Color;
        let size = photocraft_geom::Size::new(8, 6);
        let mut doc = Document::with_background("bg", size, ColorMode::Rgb, SampleType::U8, Color::rgb(0.2, 0.4, 0.6));
        doc.layers.push(Layer::raster("Layer 1", doc.pixel_format()));
        let file = document_to_psd(&doc);
        let layers = &file.layer_info.as_ref().unwrap().layers;
        let ids = |i: usize| layers[i].channels.iter().map(|c| c.id).collect::<Vec<_>>();
        assert_eq!(ids(0), vec![0, 1, 2], "the Background has no -1 channel");
        assert_eq!(layers[0].flags.0, LayerFlags::BIT4_USEFUL | LayerFlags::TRANSPARENCY_PROTECTED, "as Photoshop writes it");
        assert!(ids(1).contains(&-1), "other layers keep their transparency");
        // It comes back as the same locked, opaque Background.
        let back = crate::import("bg.psd", &file.to_bytes().unwrap()).unwrap().document;
        let bg = &back.layers[0];
        assert!(bg.locks.transparency && bg.locks.position, "{:?}", bg.locks);
        assert_eq!(bg.surface().unwrap().read_region(back.bounds()), doc.layers[0].surface().unwrap().read_region(doc.bounds()));
        // A bottom layer with transparency (or unlocked) is not a Background.
        let mut holey = doc.clone();
        holey.layers[0].surface_mut().unwrap().write_pixel(2, 2, &[0.0, 0.0, 0.0, 0.0]);
        assert!(document_to_psd(&holey).layer_info.unwrap().layers[0].channels.iter().any(|c| c.id == -1));
        let mut unlocked = doc.clone();
        unlocked.layers[0].locks.position = false;
        assert!(document_to_psd(&unlocked).layer_info.unwrap().layers[0].channels.iter().any(|c| c.id == -1));
    }

    /// A flat 1-bit Bitmap file (no layers) opens in Photoshop as its Background; so does it in
    /// PhotoCraft, and it is written back as one (no transparency channel).
    #[test]
    fn a_flat_bitmap_file_opens_and_saves_as_a_background() {
        let file = photocraft_psd::testgen::merged_only(Version::Psd, PsdMode::Bitmap, 1, Compression::Raw, 8, 4);
        let doc = crate::import("flat.psd", &file.to_bytes().unwrap()).unwrap().document;
        let bg = &doc.layers[0];
        assert!(bg.name == "Background" && bg.locks.transparency && bg.locks.position, "{:?}", bg.locks);
        let out = document_to_psd(&doc);
        assert!(out.layer_info.unwrap().layers[0].channels.iter().all(|c| c.id >= 0));
    }

    #[test]
    fn group_divider_id_wraps_without_colliding_with_imported_layer_ids() {
        let mut doc = document(8, 8);
        let mut group = Layer::group("Imported group", vec![Layer::raster("Child", doc.pixel_format())]);
        group.psd_id = Some(u32::MAX);
        doc.layers.push(group);

        let file = document_to_psd(&doc);
        let ids: Vec<u32> = file.layers().iter().filter_map(|layer| layer.layer_id()).collect();
        let unique: std::collections::HashSet<u32> = ids.iter().copied().collect();
        assert_eq!(ids.len(), 3, "group, child and closing divider");
        assert_eq!(unique.len(), ids.len(), "no generated ID may collide");
        assert!(unique.contains(&u32::MAX), "preserve the imported group ID");
        assert!(unique.contains(&1), "reuse the first unused positive ID on wrap");

        let encoded = file.to_bytes().unwrap();
        let parsed = PsdFile::from_bytes(&encoded).unwrap();
        let reloaded_ids: std::collections::HashSet<u32> = parsed.layers().iter().filter_map(|layer| layer.layer_id()).collect();
        assert_eq!(unique, reloaded_ids);
    }

    #[test]
    fn unreadable_lr16_warns_on_import_and_is_not_saved_twice() {
        // #2700: a 16-bit file whose Lr16 block claims five layers but holds none.
        let mut file = photocraft_psd::testgen::merged_only(Version::Psd, PsdMode::Rgb, 16, Compression::Raw, 8, 6);
        file.global_layer_mask = Some(GlobalLayerMask::default());
        file.global_blocks.push(TaggedBlock::new(*b"Lr16", vec![0, 5, 0, 0]));
        let imported = crate::import("broken.psd", &file.to_bytes().unwrap()).unwrap();
        assert!(imported.warnings.iter().any(|w| w.contains("Lr16") && w.contains("could not be read")), "{:?}", imported.warnings);

        let saved = PsdFile::from_bytes(&document_to_psd(&imported.document).to_bytes().unwrap()).unwrap();
        let ours = matches!(saved.layer_info_placement, LayerInfoPlacement::GlobalBlock { key, .. } if key == *b"Lr16");
        let kept = saved.global_blocks.iter().filter(|b| b.key == *b"Lr16").count();
        assert_eq!((ours, kept), (true, 0), "exactly one Lr16 block: the document's own layers");
        assert_eq!(saved.layers().len(), 1);
    }

    #[test]
    fn photo_filter_psd_is_four_byte_aligned_and_round_trips() {
        let color = [60000u16, 30000, 0].map(|x| x as f32 / 65535.0);
        let adjustment = photocraft_doc::Adjustment::PhotoFilter { color, density: 0.14, preserve_luminosity: true };
        let mut doc =
            Document::with_background("Photo Filter PSD", photocraft_geom::Size::new(32, 24), ColorMode::Rgb, SampleType::U8, photocraft_doc::Color::WHITE);
        let mut layer = Layer::new("Warming Filter", LayerContent::Adjustment(adjustment.clone()));
        doc.layers.push(layer.clone());

        let assert_phfl = |file: &PsdFile| {
            let records = &file.layer_info.as_ref().unwrap().layers;
            let data =
                records.iter().flat_map(|r| r.blocks.iter()).find(|b| b.key == *b"phfl").map(|b| b.data.as_slice()).expect("Photo Filter adjustment data");
            assert_eq!(data.len(), 20, "PSD layer block length includes four-byte padding");
            assert_eq!(&data[0..2], &[0, 2], "version 2 RGB Photo Filter");
            assert_eq!(&data[17..], &[0, 0, 0], "three padding bytes after the 17-byte payload");
        };

        let exported = document_to_psd(&doc);
        assert_phfl(&exported);
        let encoded = exported.to_bytes().unwrap();
        let parsed = PsdFile::from_bytes(&encoded).unwrap();
        assert_phfl(&parsed);
        let (restored, _) = crate::psd_import::psd_to_document(&parsed);
        assert!(restored.layers.iter().any(|l| l.content == LayerContent::Adjustment(adjustment.clone())));

        // Importing a legacy 18-byte block must still work; saving it again should repair it
        // even when the adjustment values have not changed (normally raw blocks are reused).
        let mut legacy = crate::adjust_map::write(&adjustment)[0].1.clone();
        legacy.truncate(18);
        layer.psd_blocks = vec![(*b"phfl", std::sync::Arc::new(legacy))];
        doc.layers.pop();
        doc.layers.push(layer);
        let repaired = document_to_psd(&doc);
        assert_phfl(&repaired);
        let repaired_bytes = repaired.to_bytes().unwrap();
        assert_phfl(&PsdFile::from_bytes(&repaired_bytes).unwrap());
    }

    #[test]
    fn photo_filter_version_3_is_kept_until_edited() {
        // The #3777 block (Photoshop CS2, XYZ colour): saved back byte-exact while unchanged.
        let v3 = vec![0, 3, 0, 0, 0x5e, 0x7f, 0, 0, 0x61, 0xaa, 0, 0, 0x0e, 0xf3, 0, 0, 0, 0x3f, 1, 0];
        let adjustment = crate::adjust_map::parse(b"phfl", &v3, None, crate::adjust_map::Channels::Rgb);
        let photocraft_doc::Adjustment::PhotoFilter { color, .. } = adjustment else { panic!("{adjustment:?}") };
        let mut doc =
            Document::with_background("Photo Filter v3", photocraft_geom::Size::new(8, 8), ColorMode::Rgb, SampleType::U8, photocraft_doc::Color::WHITE);
        let mut layer = Layer::new("Photo Filter 1", LayerContent::Adjustment(adjustment.clone()));
        layer.psd_blocks = vec![(*b"phfl", std::sync::Arc::new(v3.clone()))];
        doc.layers.push(layer);
        let phfl = |doc: &Document| {
            let file = PsdFile::from_bytes(&document_to_psd(doc).to_bytes().unwrap()).unwrap();
            let records = &file.layer_info.as_ref().unwrap().layers;
            records.iter().flat_map(|r| r.blocks.iter()).find(|b| b.key == *b"phfl").map(|b| b.data.clone()).expect("Photo Filter adjustment data")
        };
        assert_eq!(phfl(&doc), v3);
        // Once edited it is written as version 2 (RGB), and the colour survives.
        let edited = photocraft_doc::Adjustment::PhotoFilter { color, density: 0.5, preserve_luminosity: true };
        if let Some(l) = doc.layers.last_mut() {
            l.content = LayerContent::Adjustment(edited.clone());
        }
        let saved = phfl(&doc);
        assert_eq!(&saved[..2], &[0, 2]);
        let reread = crate::adjust_map::parse(b"phfl", &saved, None, crate::adjust_map::Channels::Rgb);
        let photocraft_doc::Adjustment::PhotoFilter { color: back, density, .. } = reread else { panic!("{reread:?}") };
        assert!(back.iter().zip(color).all(|(b, c)| (b - c).abs() <= 1.0 / 65535.0), "{back:?} vs {color:?}");
        assert_eq!(density, 0.5);
    }

    #[test]
    fn size_limit_boundary_keeps_limit_and_switches_above_it() {
        assert!(!exceeds_psd_size_limit(Some(PSD_MAX_FILE_SIZE)));
        assert!(exceeds_psd_size_limit(Some(PSD_MAX_FILE_SIZE + 1)));
        assert!(exceeds_psd_size_limit(None));
    }

    #[test]
    fn encoded_size_estimator_rejects_arithmetic_overflow() {
        assert_eq!(estimate_encoded_size(u64::MAX, 0, 0), None);
        assert_eq!(estimate_encoded_size(0, u64::MAX, 0), None);
        assert_eq!(estimate_encoded_size(0, 0, u64::MAX), Some(u64::MAX));
        assert!(exceeds_psd_size_limit(estimate_psd_size(&document(u32::MAX, u32::MAX))));
    }

    #[test]
    fn large_encoded_size_selects_psb_without_changing_explicit_choice() {
        let large = document(30_000, 30_000);
        assert!(estimate_psd_size(&large).is_some_and(|size| size > PSD_MAX_FILE_SIZE));
        let large_size_exceeds_limit = psd_size_exceeds_limit(&large);
        assert_eq!(psd_version(&large, false, large_size_exceeds_limit), Version::Psb);

        let small = document(1, 1);
        let small_size_exceeds_limit = psd_size_exceeds_limit(&small);
        assert_eq!(psd_version(&small, false, small_size_exceeds_limit), Version::Psd);
        assert_eq!(psd_version(&small, true, small_size_exceeds_limit), Version::Psb);
    }
    #[test]
    fn kept_psb_blocks_are_written_as_8bim_in_a_psd() {
        // Photoshop 27.11 refuses a PSD holding `8B64` blocks ("not compatible with this
        // version"); they came from re-saving a PSB as PSD with its global FMsk and cinf kept.
        let mut doc = document(4, 4);
        doc.metadata.psd_global_blocks = vec![
            (*b"8B64", *b"FMsk", std::sync::Arc::new(vec![0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0, 0, 50])),
            (*b"8BIM", *b"Patt", std::sync::Arc::new(Vec::new())),
        ];
        let signatures = |force_psb| {
            let opts = PsdExportOptions { force_psb, ..PsdExportOptions::default() };
            let bytes = document_to_psd_with(&doc, &opts).0.to_bytes().unwrap();
            let file = PsdFile::from_bytes(&bytes).unwrap();
            file.global_blocks.iter().map(|b| (b.signature, b.key)).collect::<Vec<_>>()
        };
        assert_eq!(signatures(false), vec![(*b"8BIM", *b"FMsk"), (*b"8BIM", *b"Patt")]);
        assert_eq!(signatures(true), vec![(*b"8B64", *b"FMsk"), (*b"8BIM", *b"Patt")], "a PSB keeps 8B64");
    }

    /// A layered document with transparency in the merged image and an alpha channel: the name
    /// and DisplayInfo resources hold one entry per extra channel, the merged transparency first,
    /// as in every file Photoshop opens (one short, Photoshop 27.11 refused the file with "Could
    /// not import the clipboard because an unexpected end-of-file was encountered"). The alpha
    /// channel keeps its own name and options on the way back.
    #[test]
    fn alpha_names_count_the_merged_transparency() {
        let mut doc = document(8, 6);
        let mut layer = Layer::raster("Layer 1", doc.pixel_format());
        layer.surface_mut().unwrap().write_pixel(1, 1, &[1.0, 0.0, 0.0, 1.0]);
        doc.layers.push(layer);
        let mut a = photocraft_doc::AlphaChannel::new("Alpha 1", photocraft_raster::Surface::new(PixelFormat::GRAY8));
        a.opacity = 0.5;
        doc.channels.push(a);
        let file = document_to_psd(&doc);
        let header_extra = usize::from(file.header.channels) - 3;
        let names = |id| file.resources.iter().find(|r| r.id == id).map(|r| r.data.clone()).unwrap();
        let pascal = names(1006);
        let (mut count, mut i) = (0, 0);
        while i < pascal.len() {
            count += 1;
            i += 1 + usize::from(pascal[i]);
        }
        assert_eq!((header_extra, count), (2, 2), "transparency + Alpha 1");
        assert_eq!(names(crate::channel_map::DISPLAY_INFO).len(), 4 + 2 * 13);
        let back = crate::import("a.psd", &file.to_bytes().unwrap()).unwrap().document;
        assert_eq!(back.channels.len(), 1);
        assert_eq!(back.channels[0].name, "Alpha 1");
        assert!((back.channels[0].opacity - 0.5).abs() < 1e-6);
    }
}
