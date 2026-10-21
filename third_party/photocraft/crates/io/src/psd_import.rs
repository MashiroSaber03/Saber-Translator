//! PSD/PSB → [`Document`].

use std::sync::Arc;

use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{
    AlphaChannel, Document, Effects, FillCache, Group, Layer, LayerContent, LayerMask, ShapeLayer, SmartContentsId, SmartObject, SmartSource, TextLayer,
};
use photocraft_geom::{Rect, Size, TILE_SIZE};
use photocraft_psd::layer::{CHANNEL_REAL_USER_MASK, CHANNEL_TRANSPARENCY, CHANNEL_USER_MASK};
use photocraft_psd::resources::ids;
use photocraft_psd::tagged::BlockData;
use photocraft_psd::{ColorMode as PsdMode, LayerNode, LayerRecord, PsdFile, TaggedBlock};
use photocraft_raster::Surface;

use crate::adjust_map::{self, ADJUSTMENT_KEYS};
use crate::blocks;
use crate::pixels::{interleave, max_sample, sample_for_depth, zero_sample};

/// Resources that are mapped to document fields or regenerated on export.
pub(crate) const MAPPED_RESOURCES: [u16; 19] = [
    ids::GLOBAL_ANGLE,
    ids::GLOBAL_ALTITUDE,
    ids::RESOLUTION_INFO,
    ids::ICC_PROFILE,
    ids::XMP,
    ids::EXIF,
    ids::VERSION_INFO,
    ids::THUMBNAIL,
    ids::THUMBNAIL_PS4,
    ids::LAYER_STATE,
    ids::LAYER_GROUP_INFO,
    1032, // grid and guides
    1045, // unicode alpha names
    1006, // pascal alpha names
    1069, // layer selection ids
    1072, // layer group(s) enabled id
    crate::channel_map::DISPLAY_INFO,
    crate::channel_map::DISPLAY_INFO_OLD,
    crate::channel_map::QUICK_MASK_INFO,
];

/// Blocks regenerated from document fields on export; not kept in
/// `Layer::psd_blocks`.
pub(crate) const REGENERATED: [&[u8; 4]; 9] = [b"luni", b"lyid", b"lsct", b"lsdk", b"iOpa", b"lspf", b"lclr", b"lfx2", b"lfxs"];

pub(crate) struct Ctx<'a> {
    pub file: &'a PsdFile,
    pub fmt: PixelFormat,
    pub mask_fmt: PixelFormat,
    pub cc: usize,
    pub cmyk: bool,
    pub warnings: Vec<String>,
    /// Document resolution (type sizes are converted to points with it).
    pub dpi: f32,
    /// The parsed `Txt2` block (type settings EngineData lacks, e.g. optical kerning).
    pub txt2: Option<photocraft_text::engine_data::Value>,
    /// Smart-filter caches from the global `FEid`/`FXid` blocks (filter masks, by placed id).
    pub filter_effects: Vec<photocraft_psd::filter_effects::FilterEffectsItem>,
    /// Placed layers naming the same source share editable contents, not their placement.
    pub smart_contents: std::collections::HashMap<String, SmartContentsId>,
    /// Source sizes of `PlLd`-only placed layers, by source id (read once per embedded file).
    pub plld_sizes: std::collections::HashMap<String, Option<(f64, f64)>>,
    /// Cancellation and progress for background opens (checked per layer record).
    pub ctl: photocraft_raster::Interrupt<'a>,
    /// Layer records decoded so far, out of `total` (progress).
    pub done: usize,
    pub total: usize,
}

fn doc_mode(m: PsdMode) -> Option<ColorMode> {
    Some(match m {
        PsdMode::Bitmap => ColorMode::Bitmap,
        PsdMode::Grayscale => ColorMode::Grayscale,
        PsdMode::Indexed => ColorMode::Indexed,
        PsdMode::Rgb => ColorMode::Rgb,
        PsdMode::Cmyk => ColorMode::Cmyk,
        PsdMode::Multichannel => ColorMode::Multichannel,
        PsdMode::Duotone => ColorMode::Duotone,
        PsdMode::Lab => ColorMode::Lab,
        PsdMode::Unknown(_) => return None,
    })
}

// Real-mask metadata without a -3 channel still selects the synthetic -2 mask.
fn selected_real_mask(rec: &LayerRecord) -> Option<photocraft_psd::RealMask> {
    rec.channel(CHANNEL_REAL_USER_MASK)?;
    rec.layer_mask()?.real
}

/// The fill of a plain shape layer (fill block + vector path, no stroke) to import as a fill layer
/// with a vector mask: when its mask parameters give the vector mask a density below 100 % or a
/// feather (`.1` true), or when it fills with a pattern.
fn soft_shape_fill(rec: &LayerRecord, has_vector: bool, fill_key: Option<&[u8; 4]>) -> Option<(photocraft_doc::Fill, bool)> {
    if !has_vector || rec.block(b"vstk").is_some() || rec.block(b"vscg").is_some() {
        return None;
    }
    let k = fill_key?;
    let fill = rec.block(k).and_then(|b| blocks::parse_fill(k, &b.data))?;
    let p = rec.layer_mask().and_then(|m| m.parameters);
    let soft = p.is_some_and(|p| p.vector_density.is_some_and(|d| d < 255) || p.vector_feather.is_some_and(|f| f > 0.0));
    // A pattern-filled shape can't be rasterized on its own (the compositor resolves the
    // document's patterns for fill layers), and its stored pixels may be the unclipped fill
    // (our own export writes fill layers so, #1907): it is a pattern fill layer seen through
    // its vector mask.
    let pattern = matches!(fill, photocraft_doc::Fill::Pattern { .. });
    (soft || pattern).then_some((fill, soft))
}

/// Where a placed layer's source goes: read from `SoLd`/`SoLE`, or from a CS2 `PlLd` record.
/// An empty `id` leaves the contents unavailable (the stored pixels stay the rendering).
#[derive(Default)]
struct Placement {
    id: String,
    transform: photocraft_geom::Affine,
    perspective: Option<[f64; 9]>,
    warp: Option<photocraft_geom::warp::Warp>,
}

impl Ctx<'_> {
    fn warn(&mut self, s: impl Into<String>) {
        self.warnings.push(s.into());
    }

    /// Source id, transform, (Distort/Perspective) projective placement and warp of a placed
    /// layer that has only a `PlLd` block, as Photoshop CS2 and older write it (#2465). `PlLd`
    /// stores the placed corners but not the source size, which comes from the embedded file.
    fn plld_placement(&mut self, data: &[u8], name: &str) -> Placement {
        let Some((id, p)) = blocks::parse_plld(data) else {
            self.warn(format!("layer \"{name}\": its placed-layer data (PlLd) could not be read; the smart object's contents are unavailable"));
            return Placement::default();
        };
        let warp = match blocks::parse_plld_warp(data) {
            blocks::PlldWarp::None => None,
            blocks::PlldWarp::Warp(w) => Some(w),
            blocks::PlldWarp::Unreadable => {
                // Re-rendering from the source would unwarp the layer: keep it as saved (its
                // warped pixels), as before PlLd was read.
                self.warn(format!(
                    "layer \"{name}\": its smart object warp (PlLd) could not be read; the layer keeps its warped pixels but its contents are unavailable"
                ));
                return Placement::default();
            }
        };
        let (file, ctl) = (self.file, self.ctl);
        let size = *self
            .plld_sizes
            .entry(id.clone())
            .or_insert_with(|| crate::linked::source_size_in_blocks(file.global_blocks.iter().map(|b| (&b.key, b.data.as_slice())), &id, &ctl));
        // Without the source (missing or unreadable), it is taken as placed at 100 %.
        let (transform, perspective) = blocks::plld_transform(&p, size, warp.as_ref());
        Placement { id, transform, perspective, warp }
    }

    fn channel_plane(&mut self, rec: &LayerRecord, id: i16, name: &str) -> Option<Vec<u8>> {
        rec.channel(id)?;
        match rec.decode_channel(id, self.file.header.depth, self.file.header.version) {
            Ok(p) => Some(p),
            Err(e) => {
                self.warn(format!("layer \"{name}\": channel {id} could not be decoded ({e}); treated as empty"));
                None
            }
        }
    }

    /// [`Self::channel_plane`] without the warnings sink, so a layer's channels decode
    /// concurrently (their RLE rows are the bulk of the open time on full-canvas layers).
    fn channel_plane_decoded(&self, rec: &LayerRecord, id: i16) -> Option<Vec<u8>> {
        rec.channel(id)?;
        rec.decode_channel(id, self.file.header.depth, self.file.header.version).ok()
    }

    fn record_surface(&mut self, rec: &LayerRecord, name: &str) -> Surface {
        let r = rec.rect;
        if r.is_empty() || r.size().is_err() {
            return Surface::new(self.fmt);
        }
        let (w, h) = r.size().unwrap_or((0, 0));
        let s = self.fmt.sample;
        let ids: Vec<i16> = (0..self.cc as i16).chain([CHANNEL_TRANSPARENCY]).collect();
        let planes: Vec<Option<Vec<u8>>> = if cfg!(target_arch = "wasm32") {
            ids.iter().map(|&id| self.channel_plane(rec, id, name)).collect()
        } else {
            // The channels decode independently; their RLE rows are the bulk of the open
            // time for full-canvas layers, so they run side by side and the warnings land
            // in the same order afterwards.
            // par_map reuses the crate's worker pool (one OS thread per channel per layer
            // would spawn thousands on a file with many layers).
            let decoded: Vec<Option<Vec<u8>>> = crate::pixels::par_map(ids.to_vec(), |id| self.channel_plane_decoded(rec, id));
            let mut planes: Vec<Option<Vec<u8>>> = Vec::with_capacity(decoded.len());
            for (&id, p) in ids.iter().zip(decoded) {
                if p.is_none() && rec.channel(id).is_some() {
                    self.warn(format!("layer \"{name}\": channel {id} could not be decoded; treated as empty"));
                }
                planes.push(p);
            }
            planes
        };
        // A channel that is in the file but could not be decoded leaves the layer empty, as the
        // warning says. Filling in for it would paint an opaque black layer over the document.
        if ids.iter().zip(planes.iter()).any(|(&id, p)| p.is_none() && rec.channel(id).is_some()) {
            return Surface::new(self.fmt);
        }
        // Only decoded channel data justifies a buffer: the decoders bound their output by the
        // bytes in the file, but a bare rectangle can declare 300000² pixels in a few bytes (#755).
        if planes.iter().all(Option::is_none) {
            self.warn(format!("layer \"{name}\": no channel data for its {w}x{h} bounds; treated as empty"));
            return Surface::new(self.fmt);
        }
        let refs: Vec<Option<&[u8]>> = planes.iter().map(|p| p.as_deref()).collect();
        let mut fill: Vec<Vec<u8>> = vec![zero_sample(s); self.cc];
        fill.push(max_sample(s));
        let mut invert = vec![self.cmyk; self.cc];
        invert.push(false);
        let mut bytes = interleave(&refs, &fill, w * h, s, &invert);
        if self.fmt.mode == ColorMode::Lab && s == SampleType::U16 {
            crate::pixels::lab16_chroma(&mut bytes, self.cc + 1, true);
        }
        let mut surf = Surface::from_interleaved(self.fmt, Rect::new(r.left, r.top, r.right, r.bottom), &bytes);
        surf.prune();
        surf
    }

    fn record_mask(&mut self, rec: &LayerRecord, name: &str) -> Option<LayerMask> {
        let m = rec.layer_mask()?;
        let (id, rect, default, flags) = match selected_real_mask(rec) {
            Some(real) => (CHANNEL_REAL_USER_MASK, real.rect, real.background, real.flags),
            _ => {
                rec.channel(CHANNEL_USER_MASK)?;
                (CHANNEL_USER_MASK, m.rect, m.default_color, m.flags)
            }
        };
        let mut surface = Surface::with_default(self.mask_fmt, &[f32::from(default) / 255.0]);
        if !rect.is_empty()
            && let Ok((w, h)) = rect.size()
            && let Some(plane) = self.channel_plane(rec, id, name)
        {
            let s = self.mask_fmt.sample;
            let bytes = interleave(&[Some(&plane)], &[zero_sample(s)], w * h, s, &[false]);
            surface.write_interleaved(Rect::new(rect.left, rect.top, rect.right, rect.bottom), &bytes);
            surface.prune();
        }
        let params = m.parameters;
        Some(LayerMask {
            surface,
            enabled: flags & 2 == 0,
            linked: flags & 1 == 0,
            density: params.and_then(|p| p.user_density).map_or(1.0, |d| f32::from(d) / 255.0),
            feather: params.and_then(|p| p.user_feather).map_or(0.0, |f| f as f32),
        })
    }

    /// The filter mask of the smart object whose `placed` id is `placed`, from the `FEid` cache:
    /// document coordinates, white beyond its bounds (`filterMaskExtendWithWhite`). An all-white
    /// mask is no mask.
    fn filter_mask(&mut self, placed: &str, stack: &crate::smart_map::FilterStack, name: &str) -> Option<LayerMask> {
        let item = self.filter_effects.iter().find(|i| i.id == placed)?;
        match crate::smart_map::mask_from_item(item, self.mask_fmt.sample, stack) {
            Ok(m) => m,
            Err(e) => {
                self.warn(format!("layer \"{name}\": {e}; the filter mask was ignored"));
                None
            }
        }
    }

    fn apply_common(&mut self, l: &mut Layer, rec: &LayerRecord, blend_override: Option<photocraft_psd::BlendMode>) {
        l.visible = rec.is_visible();
        l.opacity = f32::from(rec.opacity) / 255.0;
        l.fill_opacity = f32::from(rec.fill_opacity()) / 255.0;
        l.clipped = rec.clipping != 0;
        let key = blend_override.unwrap_or(rec.blend_mode).key();
        l.blend = match BlendMode::from_psd_key(key) {
            Some(b) => b,
            None => {
                self.warn(format!("layer \"{}\": unknown blend mode {:?}; using Normal", l.name, String::from_utf8_lossy(&key)));
                BlendMode::Normal
            }
        };
        if let Some(Ok(BlockData::Protection(v))) = rec.block(b"lspf").and_then(TaggedBlock::parsed) {
            l.locks = blocks::locks_from_lspf(v);
        }
        if rec.flags.transparency_protected() {
            l.locks.transparency = true;
        }
        if let Some(Ok(BlockData::SheetColor(v))) = rec.block(b"lclr").and_then(TaggedBlock::parsed) {
            l.label = blocks::label_from_index(v);
        }
        // `lmfx` (multiple instances per kind) supersedes `lfx2` when present;
        // it stays in `psd_blocks`, `lfx2` in `Effects::psd_raw`.
        // Groups store their effects under `lfxs` (same layout as `lfx2`).
        let single = rec.block(b"lfx2").or_else(|| rec.block(b"lfxs"));
        if let Some(fx) = rec.block(b"lmfx").or(single) {
            let (enabled, items) = crate::effects_map::parse_lfx2(&fx.data).unwrap_or_else(|| (blocks::effects_enabled(&fx.data), Vec::new()));
            l.effects = Effects { enabled, items, psd_raw: single.map(|b| Arc::new(b.data.clone())), reference: None };
        } else if let Some(fx) = rec.block(b"lrFX")
            && !rec.section_type().is_folder()
            && let Some((enabled, items)) = crate::effects_map::parse_lrfx(&fx.data)
        {
            // Legacy effects only on non-group layers (Photoshop ignores
            // them on groups, see psd-tools effects/shape-fx.psd).
            l.effects = Effects { enabled, items, psd_raw: None, reference: None };
        }
        // Effects reference point (`fxrp`: two f64, x then y); also left in `psd_blocks`, where
        // export overwrites it from the field.
        if let Some(b) = rec.block(b"fxrp")
            && let (Some(x), Some(y)) = (b.data.get(..8), b.data.get(8..16))
        {
            let f = |s: &[u8]| f64::from_be_bytes(s.try_into().unwrap_or([0; 8]));
            l.effects.reference = Some((f(x), f(y)));
        }
        // Advanced Blending channel restrictions (`brst`: the u32 ids of channels left out);
        // also kept in `psd_blocks`, where export rewrites it from the field.
        if let Some(b) = rec.block(b"brst") {
            l.excluded_channels = crate::blocks::parse_brst(&b.data);
        }
        // Advanced Blending (`knko`, `infx`, `clbl`, `tsly`, `lmgm`, `vmgm`); the blocks stay in
        // `psd_blocks`, where export rewrites them from the field in place.
        l.advanced = crate::blocks::advanced_from_blocks(|k| rec.block(k).map(|b| b.data.as_slice()));
        // Blend If lives in the layer record's blending ranges.
        l.blend_if = crate::blocks::blend_if_from_ranges(&rec.blending_ranges);
        l.psd_id = rec.layer_id();
        let name = l.name.clone();
        l.mask = self.record_mask(rec, &name);
    }

    fn layer_from_record(&mut self, rec: &LayerRecord) -> Layer {
        let name = rec.name();
        let fill_key = [b"SoCo", b"GdFl", b"PtFl"].into_iter().find(|k| rec.block(k).is_some());
        let vector_key = [b"vsms", b"vmsk"].into_iter().find(|k| rec.block(k).is_some());
        let adj_key = ADJUSTMENT_KEYS.into_iter().find(|k| rec.block(k).is_some());
        // `SoLE` (a linked smart object) before `PlLd`: a `PlLd` alone is Photoshop CS2's record.
        let smart_key = [b"SoLd", b"SoLE", b"PlLd"].into_iter().find(|k| rec.block(k).is_some());
        let blocks = preserved_blocks(rec);
        // `psd_raw` shares the Arc of the matching `psd_blocks` entry.
        let principal = |k: &[u8; 4]| blocks.iter().find(|(bk, _)| bk == k).map(|(_, d)| d.clone());
        let mut fill_cache = None;

        let content = if let Some(k) = adj_key {
            let data = rec.block(k).map(|b| b.data.clone()).unwrap_or_default();
            let cged = rec.block(b"CgEd").map(|b| &b.data[..]);
            let adj = adjust_map::parse(
                k,
                &data,
                cged,
                match self.fmt.mode {
                    ColorMode::Rgb => adjust_map::Channels::Rgb,
                    ColorMode::Grayscale => adjust_map::Channels::Gray,
                    ColorMode::Cmyk => adjust_map::Channels::Cmyk,
                    ColorMode::Lab => adjust_map::Channels::Lab,
                    _ => adjust_map::Channels::Other,
                },
            );
            // Kept verbatim for saving, but it renders as nothing: say so rather than open the
            // layer silently without its effect (#1763).
            if matches!(adj, photocraft_doc::Adjustment::Unsupported { .. }) {
                let why = adjust_map::unreadable_reason(k, &data).map(|r| format!(" ({r})")).unwrap_or_default();
                self.warn(format!(
                    "layer \"{name}\": its {} settings could not be read{why}; the layer is kept as saved but has no effect",
                    adjust_map::label(k)
                ));
            }
            LayerContent::Adjustment(adj)
        } else if rec.block(b"TySh").is_some() {
            // Typed model from TySh/EngineData (photocraft-text); Photoshop's pixels stay the cache.
            let data = rec.block(b"TySh").map(|b| b.data.clone()).unwrap_or_default();
            let mut t = photocraft_text::psd::text_layer_from_tysh(&data, self.dpi).unwrap_or_else(|| {
                let (text, transform) = blocks::parse_tysh(&data).unwrap_or_default();
                TextLayer { text, transform, ..Default::default() }
            });
            if let Some(txt2) = &self.txt2 {
                photocraft_text::psd::apply_txt2(&mut t, &data, txt2);
                photocraft_text::psd::apply_txt2_frame(&mut t, &data, txt2);
            }
            // Photoshop's pixels are the cache only when the file has some: ag-psd, GIMP and
            // other writers leave type layers without image data (Photoshop re-renders them on
            // open), and an empty cache would show nothing until the layer is edited; the import
            // renders such a layer from its model instead (`text_import::prepare`). When the
            // engine text itself is blank, Photoshop draws nothing either: keep the empty pixels
            // (corpus: text/path-wave-open.psd).
            let cache = self.record_surface(rec, &name);
            let drawn = !cache.content_bounds().is_empty() || photocraft_text::psd::engine_text_is_blank(&data);
            t.cache = drawn.then_some(cache);
            t.psd_raw = principal(b"TySh");
            LayerContent::Text(t)
        } else if let Some(k) = smart_key {
            let Placement { id, transform, perspective, warp } = match rec.block(k) {
                Some(b) if k == b"PlLd" => self.plld_placement(&b.data, &name),
                Some(b) => {
                    let (id, transform) = blocks::parse_smart(k, &b.data);
                    // Distort / Perspective: the fourth corner (the affine `transform` drops it).
                    let perspective = blocks::parse_smart_perspective(k, &b.data);
                    Placement { id, transform, perspective, warp: blocks::parse_placed_warp(k, &b.data) }
                }
                None => Placement::default(),
            };
            let contents_id =
                if id.is_empty() { SmartContentsId::fresh() } else { *self.smart_contents.entry(id.clone()).or_insert_with(SmartContentsId::fresh) };
            // Smart filters (`filterFX` in the placed-layer data) and their mask (`FEid`).
            let placed = rec.block(b"SoLd").or_else(|| rec.block(b"SoLE")).and_then(|b| crate::smart_map::parse_sold(&b.data));
            let stack = placed.as_ref().and_then(|p| p.stack.clone()).unwrap_or_default();
            let filter_mask = match &placed {
                Some(p) if !stack.filters.is_empty() => self.filter_mask(&p.placed, &stack, &name),
                _ => None,
            };
            LayerContent::Smart(SmartObject {
                contents_id,
                source: SmartSource::Linked { path: id },
                transform,
                smart_filters: stack.filters,
                cache: Some(self.record_surface(rec, &name)),
                psd_raw: principal(k),
                filters_enabled: stack.enabled,
                filter_mask,
                warp,
                stack_mode: None,
                perspective,
            })
        } else if let Some((f, soft)) = soft_shape_fill(rec, vector_key.is_some(), fill_key) {
            // A shape whose vector mask has a density or feather is a fill layer seen through a
            // soft vector mask: the fill shows beyond the path, which the stored pixels (the
            // shape alone) lack. Import it as exactly that. A pattern fill keeps the stored
            // pixels as its rendering (a soft mask's pixels lack the fill beyond the path).
            if !soft && matches!(f, photocraft_doc::Fill::Pattern { .. }) && !rec.rect.is_empty() {
                fill_cache = Some(FillCache { fill: f.clone(), surface: self.record_surface(rec, &name) });
            }
            LayerContent::Fill(f)
        } else if (vector_key.is_some() && (fill_key.is_some() || rec.block(b"vstk").is_some())) || rec.block(b"vscg").is_some() {
            let fill = fill_key.and_then(|k| rec.block(k).and_then(|b| blocks::parse_fill(k, &b.data)));
            let mut sh = ShapeLayer { fill, cache: Some(self.record_surface(rec, &name)), psd_raw: vector_key.and_then(principal), ..Default::default() };
            let lookup = |k: &[u8; 4]| rec.block(k).map(|b| b.data.clone());
            crate::vector_map::shape_from_blocks(&mut sh, &lookup, self.file.header.width, self.file.header.height, self.dpi);
            // No stored pixels (32-bit documents, files saved without layer pixels): render the
            // shape from its path and fill, as Photoshop does when it opens the file.
            if (rec.rect.is_empty() || rec.rect.size().is_err()) && (!sh.path.subpaths.is_empty() || sh.path.inverted) {
                let canvas = Rect::new(0, 0, self.file.header.width as i32, self.file.header.height as i32);
                sh.cache = Some(photocraft_vector::render_shape(&sh, self.fmt, canvas));
            }
            LayerContent::Shape(sh)
        } else if let Some(k) = fill_key {
            match rec.block(k).and_then(|b| blocks::parse_fill(k, &b.data)) {
                Some(f) => {
                    // Keep Photoshop's rendering of the fill so it composites exactly.
                    if !rec.rect.is_empty() {
                        fill_cache = Some(FillCache { fill: f.clone(), surface: self.record_surface(rec, &name) });
                    }
                    LayerContent::Fill(f)
                }
                None => {
                    self.warn(format!("layer \"{name}\": unreadable {} fill; imported as pixels", String::from_utf8_lossy(k)));
                    LayerContent::Raster(self.record_surface(rec, &name))
                }
            }
        } else {
            LayerContent::Raster(self.record_surface(rec, &name))
        };
        let rendered = matches!(content, LayerContent::Shape(_) | LayerContent::Text(_) | LayerContent::Smart(_));
        let mut l = Layer::new(name, content);
        l.psd_blocks = blocks;
        l.fill_cache = fill_cache;
        self.apply_common(&mut l, rec, None);
        // Mask flag bit 3: the user mask was rendered from vector data. For
        // shape/text/smart layers the cached pixels already include that
        // coverage, so applying it again would double-mask.
        if rendered && selected_real_mask(rec).is_none() && rec.layer_mask().is_some_and(|m| m.flags & 8 != 0) {
            l.mask = None;
        }
        if !matches!(l.content, LayerContent::Shape(_)) {
            self.apply_vector_mask(&mut l, rec);
        }
        l
    }

    /// Typed vector mask from `vsms`/`vmsk` (non-shape layers). A user mask that Photoshop
    /// rendered from the vector data (flag bit 3, no separate real mask) is dropped: the
    /// compositor rasterizes the vector mask itself.
    fn apply_vector_mask(&mut self, l: &mut Layer, rec: &LayerRecord) {
        let Some(b) = rec.block(b"vsms").or_else(|| rec.block(b"vmsk")) else { return };
        let Some(mut vm) = crate::vector_map::vector_mask_from_block(&b.data, self.file.header.width, self.file.header.height) else { return };
        if let Some(m) = rec.layer_mask() {
            if let Some(p) = m.parameters {
                vm.density = p.vector_density.map_or(1.0, |d| f32::from(d) / 255.0);
                vm.feather = p.vector_feather.map_or(0.0, |f| f as f32);
            }
            if m.flags & 8 != 0 && selected_real_mask(rec).is_none() {
                l.mask = None;
            }
        }
        l.vector_mask = Some(vm);
    }

    fn build(&mut self, nodes: &[LayerNode], depth: usize) -> Vec<Layer> {
        let layers = self.file.layers();
        let ctl = self.ctl;
        nodes
            .iter()
            // A cancelled open stops decoding; the caller discards the partial document.
            .take_while(|_| !ctl.cancelled())
            .map(|n| match n {
                LayerNode::Layer { index } => {
                    let l = self.layer_from_record(&layers[*index]);
                    self.done += 1;
                    // Layers are 10–95 % of an open (the merged image and channels the rest).
                    self.ctl.progress(0.1 + 0.85 * self.done as f32 / self.total.max(1) as f32);
                    l
                }
                LayerNode::Group { index, children, .. } => {
                    let rec = &layers[*index];
                    // The recursion here (and in every later consumer of the tree) is bounded by
                    // the document model's nesting cap; a deeper subtree is not imported.
                    let children = if depth >= photocraft_doc::MAX_GROUP_DEPTH {
                        self.warn(format!(
                            "group `{}` nests deeper than {} groups; its contents were not imported",
                            rec.name(),
                            photocraft_doc::MAX_GROUP_DEPTH
                        ));
                        Vec::new()
                    } else {
                        self.build(children, depth + 1)
                    };
                    let sd = rec.section_divider();
                    let expanded = sd.is_none_or(|s| s.kind != photocraft_psd::SectionType::ClosedFolder);
                    let artboard = crate::comps_map::ARTBOARD_KEYS.iter().find_map(|k| rec.block(k)).and_then(|b| crate::comps_map::parse_artboard(&b.data));
                    let mut l = Layer::new(rec.name(), LayerContent::Group(Group { children, expanded, artboard }));
                    l.psd_blocks = preserved_blocks(rec);
                    self.apply_common(&mut l, rec, sd.and_then(|s| s.blend_mode));
                    self.apply_vector_mask(&mut l, rec);
                    l
                }
            })
            .collect()
    }
}

/// Sets `Layer::link_group` from resource 1026's per-record ids (`nodes` and `layers` correspond).
fn apply_link_groups(nodes: &[LayerNode], layers: &mut [Layer], ids: &[u16]) {
    fn rec(nodes: &[LayerNode], layers: &mut [Layer], ids: &[u16], depth: usize) {
        for (n, l) in nodes.iter().zip(layers.iter_mut()) {
            let index = match n {
                LayerNode::Layer { index } => *index,
                LayerNode::Group { index, children, .. } => {
                    // Matches the importer's nesting cap: deeper children were not imported.
                    if depth < photocraft_doc::MAX_GROUP_DEPTH
                        && let LayerContent::Group(g) = &mut l.content
                    {
                        rec(children, &mut g.children, ids, depth + 1);
                    }
                    *index
                }
            };
            l.link_group = ids.get(index).copied().filter(|&g| g != 0).map(u64::from);
        }
    }
    rec(nodes, layers, ids, 0);
}

/// Rejects a file whose groups nest deeper than the document model's cap, with the actionable
/// limit error. Nesting past the cap could never be saved (.pcraft refuses it).
pub(crate) fn check_group_depth(file: &PsdFile) -> Result<(), crate::IoError> {
    if group_depth(file) > photocraft_doc::MAX_GROUP_DEPTH {
        return Err(crate::IoError::Unsupported(format!("layer groups nested deeper than {}", photocraft_doc::MAX_GROUP_DEPTH)));
    }
    Ok(())
}

/// Deepest group nesting of the file's layer records (a layer inside this many groups is the
/// deepest), computed like `layer_tree`'s stack — iteratively, so no file can make it recurse.
pub(crate) fn group_depth(file: &PsdFile) -> usize {
    use photocraft_psd::tagged::SectionType;
    let (mut open, mut depth) = (0usize, 0usize);
    for rec in file.layers() {
        match rec.section_type() {
            SectionType::BoundingDivider => {
                open += 1;
                depth = depth.max(open);
            }
            SectionType::OpenFolder | SectionType::ClosedFolder => open = open.saturating_sub(1),
            _ => {}
        }
    }
    depth
}

fn preserved_blocks(rec: &LayerRecord) -> Vec<([u8; 4], Arc<Vec<u8>>)> {
    rec.blocks.iter().filter(|b| !REGENERATED.contains(&&b.key)).map(|b| (b.key, Arc::new(b.data.clone()))).collect()
}

fn be_to_ne_plane(plane: &[u8], s: SampleType, n: usize) -> Vec<u8> {
    interleave(&[Some(plane)], &[zero_sample(s)], n, s, &[false])
}

fn unicode_names(data: &[u8]) -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 4 <= data.len() {
        let n = u32::from_be_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]]) as usize;
        at += 4;
        let Some(b) = data.get(at..at + n * 2) else { break };
        let units: Vec<u16> = b.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes([c[0], c[1]])).collect();
        out.push(String::from_utf16_lossy(&units).trim_end_matches('\0').to_string());
        at += n * 2;
    }
    out
}

fn parse_guides(data: &[u8], doc: &mut Document) {
    let rd = |at: usize| data.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let Some(count) = rd(12) else { return };
    for i in 0..count as usize {
        let at = 16 + i * 5;
        let (Some(loc), Some(&dir)) = (rd(at), data.get(at + 4)) else { break };
        let pos = f64::from(loc as i32) / 32.0;
        if dir == 1 {
            doc.guides.horizontal.push(pos);
        } else {
            doc.guides.vertical.push(pos);
        }
    }
}

/// Converts a parsed PSD into a document. Never fails: problems become warnings.
/// A flattened file's merged image as its layer: like Photoshop, an opaque one is the locked
/// Background (Photoshop 27.11 opens psd-tools' flat Bitmap and Indexed files that way), so it is
/// written back as one.
fn flat_background(s: Surface) -> Layer {
    let opaque = {
        let r = s.content_bounds();
        let n = s.channels();
        !r.is_empty() && s.read_region(r).chunks_exact(n).all(|p| p.last().is_some_and(|a| *a >= 1.0))
    };
    let mut l = Layer::new("Background", LayerContent::Raster(s));
    l.locks.transparency = opaque;
    l.locks.position = opaque;
    l
}

/// Photoshop opens a file's bottom layer as its Background when that layer's record has no
/// transparency (`-1`) channel, whatever its name and lock flags say: it is then locked and shown
/// as "Background" (measured on Photoshop 27.11 with files from psd-tools and ag-psd whose bottom
/// layer is called "Фон", "bg" or "Layer 1" and carries no locks).
fn mark_background(file: &PsdFile, layers: &mut [Layer]) {
    let Some(rec) = file.layers().first() else { return };
    let Some(bottom) = layers.first_mut() else { return };
    let opaque_record = !rec.channels.is_empty() && rec.channels.iter().all(|c| c.id >= 0);
    if opaque_record && matches!(bottom.content, LayerContent::Raster(_)) && !bottom.clipped {
        bottom.locks.transparency = true;
        bottom.locks.position = true;
    }
}

pub fn psd_to_document(file: &PsdFile) -> (Document, Vec<String>) {
    // Never cancelled, so always `Some`; the fallback is unreachable.
    psd_to_document_with(file, &photocraft_raster::Interrupt::NONE)
        .unwrap_or_else(|| (Document::new("Untitled", Size::new(1, 1), ColorMode::Rgb, SampleType::U8), Vec::new()))
}

/// [`psd_to_document`] for a background open: checks `ctl` per layer record and reports
/// progress. `None` when cancelled.
pub fn psd_to_document_with(file: &PsdFile, ctl: &photocraft_raster::Interrupt) -> Option<(Document, Vec<String>)> {
    let h = &file.header;
    let mut warnings = Vec::new();
    let mode = doc_mode(h.color_mode).unwrap_or_else(|| {
        warnings.push(format!("unknown color mode {}; importing as RGB", h.color_mode.as_u16()));
        ColorMode::Rgb
    });
    let layered = matches!(mode, ColorMode::Grayscale | ColorMode::Rgb | ColorMode::Cmyk | ColorMode::Lab) && h.depth != 1;
    let multichannel = mode == ColorMode::Multichannel && h.depth != 1;
    let depth = if layered || multichannel { sample_for_depth(h.depth) } else { SampleType::U8 };
    let mut doc = Document::new("Untitled", Size::new(h.width, h.height), mode, depth);

    // Resources.
    for r in &file.resources {
        match r.id {
            ids::RESOLUTION_INFO => {
                if let Ok(ri) = photocraft_psd::ResolutionInfo::from_bytes(&r.data) {
                    // Fixed 16.16 hRes/vRes are pixels per inch. The unit shorts only
                    // say how to show that number (1 = px/in, 2 = px/cm).
                    let (x, y) = (ri.h_res(), ri.v_res());
                    doc.resolution_dpi = x as f32;
                    warnings.extend(crate::unequal_resolution_warning(x, y));
                }
            }
            ids::ICC_PROFILE => doc.icc_profile = Some(Arc::new(r.data.clone())),
            ids::XMP => doc.metadata.xmp = Some(String::from_utf8_lossy(&r.data).into_owned()),
            ids::EXIF => doc.metadata.exif = Some(Arc::new(r.data.clone())),
            1032 => parse_guides(&r.data, &mut doc),
            ids::GLOBAL_ANGLE | ids::GLOBAL_ALTITUDE => {
                if let Some(b) = r.data.get(..4) {
                    let v = i32::from_be_bytes([b[0], b[1], b[2], b[3]]) as f32;
                    if r.id == ids::GLOBAL_ANGLE {
                        doc.global_light.angle = v;
                    } else {
                        doc.global_light.altitude = v;
                    }
                }
            }
            id if crate::vector_map::SAVED_PATHS.contains(&id) => match crate::vector_map::path_from_resource(&r.data, h.width, h.height) {
                Some(path) => doc.paths.push(photocraft_doc::NamedPath {
                    name: String::from_utf8_lossy(&r.name).into_owned(),
                    path,
                    psd_raw: Some(Arc::new(r.data.clone())),
                }),
                None => doc.metadata.psd_resources.push((id, String::from_utf8_lossy(&r.name).into_owned(), Arc::new(r.data.clone()))),
            },
            crate::vector_map::WORK_PATH => {
                doc.work_path = crate::vector_map::path_from_resource(&r.data, h.width, h.height);
            }
            crate::vector_map::CLIPPING_PATH => {
                // Kept raw (layout beyond the name is not modelled); the name is exposed.
                let n = usize::from(r.data.first().copied().unwrap_or(0));
                let name = r.data.get(1..1 + n).map(|b| String::from_utf8_lossy(b).into_owned()).unwrap_or_default();
                doc.clipping_path = Some(photocraft_doc::ClippingPath { name, flatness: 0.0 });
                doc.metadata.psd_resources.push((r.id, String::from_utf8_lossy(&r.name).into_owned(), Arc::new(r.data.clone())));
            }
            id if MAPPED_RESOURCES.contains(&id) => {}
            id => doc.metadata.psd_resources.push((id, String::from_utf8_lossy(&r.name).into_owned(), Arc::new(r.data.clone()))),
        }
    }
    for b in &file.global_blocks {
        doc.metadata.psd_global_blocks.push((b.signature, b.key, Arc::new(b.data.clone())));
    }
    // A 16/32-bit file whose layer block can't be parsed would otherwise open as a flattened
    // file without a word (#2700). The raw block is kept, but a PSD save writes the document's
    // own layers in its place (see `psd_export`).
    if let Some((b, e)) = file.unreadable_layer_block() {
        warnings.push(format!(
            "the layers ({} block) could not be read ({e}); only the merged image was opened. Saving as PSD replaces the unreadable layer data",
            b.key_str()
        ));
    }
    doc.patterns = crate::pattern_map::from_global_blocks(&doc);
    // Notes (`Anno`) and the measurement scale (resource 1074); raw data stays for verbatim export.
    doc.notes = crate::annotations_map::notes_from_blocks(&doc);
    if crate::annotations_map::anno_unreadable(&doc) {
        warnings.push("the notes (Anno block) could not be read; they are kept unchanged on PSD save unless a note is added".into());
    }
    if let Some(scale) = crate::annotations_map::raw_scale(&doc) {
        doc.measurement.scale = scale;
    }

    let fmt = doc.pixel_format();
    let cc = fmt.mode.color_channels();
    let mut cx = Ctx {
        file,
        fmt,
        mask_fmt: PixelFormat::new(ColorMode::Grayscale, depth, false),
        cc,
        cmyk: fmt.mode == ColorMode::Cmyk,
        warnings,
        dpi: doc.resolution_dpi,
        txt2: file.global_blocks.iter().find(|b| &b.key == b"Txt2").and_then(|b| photocraft_text::psd::parse_txt2(&b.data)),
        filter_effects: Vec::new(),
        smart_contents: Default::default(),
        plld_sizes: Default::default(),
        ctl: *ctl,
        done: 0,
        total: file.layers().len(),
    };
    for b in file.global_blocks.iter().filter(|b| matches!(&b.key, b"FEid" | b"FXid")) {
        // Smart-filter caches can be large: a cancelled open stops between blocks.
        if ctl.cancelled() {
            return None;
        }
        match photocraft_psd::filter_effects::FilterEffects::parse(&b.data) {
            Ok(fx) => cx.filter_effects.extend(fx.items),
            Err(e) => cx.warn(format!("smart filter masks ({}) could not be read: {e}", b.key_str())),
        }
    }
    if ctl.cancelled() {
        return None;
    }

    let (w, hh) = (h.width as usize, h.height as usize);
    let n = w * hh;
    let canvas = Rect::new(0, 0, h.width as i32, h.height as i32);
    // The merged composite is decoded only when something consumes it: a flattened or
    // multichannel file (it *is* the image), or extra alpha/spot channels behind the colour
    // ones. A normal layered Photoshop save carries a merged composite nobody reads - roughly
    // half the file's bytes, decoded and thrown away on every open before this.
    let extra = usize::from(h.channels) > cc + usize::from(file.merged_has_alpha());
    let merged = if !layered || file.layers().is_empty() || extra {
        let m = file.decode_merged();
        if ctl.cancelled() {
            return None;
        }
        ctl.progress(0.1);
        if let Err(e) = &m {
            cx.warn(format!("merged image could not be decoded: {e}"));
        }
        Some(m)
    } else {
        None
    };

    if multichannel {
        // Ink channels only (no layers); see `multichannel_map`.
        if let Some(Ok(all)) = &merged {
            let names = file.resource(1045).map(|r| unicode_names(&r.data)).unwrap_or_default();
            crate::multichannel_map::import(file, all, &mut doc, &names, &mut cx.warnings);
        }
        if !file.layers().is_empty() {
            cx.warn(format!("Multichannel documents have no layers: {} layer records were not imported", file.layers().len()));
        }
    } else if layered {
        // Capped like `build` (#2940): a group past the document's nesting cap keeps no children,
        // so no file can make this tree, or anything walking or dropping it, recurse deeper.
        let tree = file.layer_tree_max_depth(photocraft_doc::MAX_GROUP_DEPTH);
        // Colours stored in the document's CMYK space (Gradient Map stops…) convert through
        // its own profile, as the compositor and the PSD writer read them.
        let space = photocraft_compose::cmyk_space(&doc);
        doc.layers = photocraft_color::convert::with_cmyk_space(space.as_ref(), || cx.build(&tree, 0));
        // Photoshop's Background: the bottom record stored without a transparency channel (-1),
        // locked whatever its (localized) name and whether or not the file carries an `lspf`
        // block. Its name is kept.
        mark_background(file, &mut doc.layers);
        // Layer › Link Layers: resource 1026 holds one group id per layer record (0 = unlinked).
        if let Some(Ok(photocraft_psd::resources::ResourceData::LayerGroupInfo(groups))) =
            file.resources.iter().find(|r| r.id == ids::LAYER_GROUP_INFO).and_then(photocraft_psd::resources::ImageResource::parsed)
        {
            apply_link_groups(&tree, &mut doc.layers, &groups);
        }
        if doc.layers.is_empty()
            && let Some(Ok(all)) = &merged
        {
            // Flattened file: the merged image becomes the background layer.
            let plane = h.row_bytes() * hh;
            let alpha_idx = if file.merged_has_alpha() { Some(cc) } else { None };
            let mut planes: Vec<Option<&[u8]>> = (0..cc).map(|c| all.get(c * plane..(c + 1) * plane)).collect();
            planes.push(alpha_idx.and_then(|a| all.get(a * plane..(a + 1) * plane)));
            let mut fill = vec![zero_sample(depth); cc];
            fill.push(max_sample(depth));
            let mut inv = vec![cx.cmyk; cc];
            inv.push(false);
            let mut bytes = interleave(&planes, &fill, n, depth, &inv);
            if fmt.mode == ColorMode::Lab && depth == SampleType::U16 {
                crate::pixels::lab16_chroma(&mut bytes, cc + 1, true);
            }
            let mut s = Surface::from_interleaved(fmt, canvas, &bytes);
            if alpha_idx.is_some() {
                // Undo Photoshop's white matting of the merged image.
                // A band of tile rows at a time (no full-size float copy of the image).
                let white = photocraft_raster::from_rgba(&fmt, [1.0, 1.0, 1.0, 1.0]);
                let mut vals = Vec::new();
                let mut y = canvas.y0;
                while y < canvas.y1 {
                    let band = Rect::new(canvas.x0, y, canvas.x1, y.saturating_add(TILE_SIZE).min(canvas.y1));
                    s.read_region_into(band, &mut vals);
                    for px in vals.chunks_exact_mut(cc + 1) {
                        let a = px[cc];
                        for c in 0..cc {
                            px[c] = if a <= 0.0 { 0.0 } else { crate::pixels::unmatte(px[c], a, white[c]) };
                        }
                    }
                    s.write_region(band, &vals);
                    y = band.y1;
                }
            }
            s.prune();
            let mut bg = Layer::new("Background", LayerContent::Raster(s));
            if alpha_idx.is_none() {
                bg.locks.transparency = true;
                bg.locks.position = true;
            }
            doc.layers.push(bg);
        }
    } else {
        if !file.layers().is_empty() {
            cx.warn(format!("{:?} documents are imported flattened: {} layer records were not imported", h.color_mode, file.layers().len()));
        } else {
            cx.warn(format!("{:?} {}-bit document converted to {:?} 8-bit for editing", h.color_mode, h.depth, fmt.mode));
        }
        if h.color_mode == PsdMode::Indexed && file.color_mode_data.len() >= 768 {
            // Planar palette: 256 reds, 256 greens, 256 blues (the Color Table).
            let m = &file.color_mode_data;
            let colors = (0..256).map(|i| [m[i], m[256 + i], m[512 + i]]).collect();
            doc.color_table = Some(photocraft_doc::ColorTable { colors, transparent: None });
        } else if h.color_mode == PsdMode::Duotone && !file.color_mode_data.is_empty() {
            // The duotone ink block is undocumented: keep it raw; the image shows as its gray plate.
            doc.duotone =
                Some(photocraft_doc::Duotone { inks: vec![photocraft_doc::DuotoneInk::new("Black", [0.0; 3])], psd_raw: Some(file.color_mode_data.clone()) });
            cx.warn("duotone inks are not interpreted (shown as grayscale)");
        } else if !file.color_mode_data.is_empty() {
            cx.warn("color mode data is not preserved");
        }
        let rgba = file.composite_rgba8().or_else(|_| {
            // Multichannel: show the first channels as RGB.
            let all = merged.clone().ok_or_else(|| photocraft_psd::PsdError::Invalid("no merged image".into()))??;
            let plane = h.row_bytes() * hh;
            let mut data = vec![255u8; n * 4];
            for c in 0..3.min(usize::from(h.channels)) {
                let p = photocraft_psd::pixels::plane_to_u8(&all[c * plane..(c + 1) * plane], h.depth, w, hh)?;
                for i in 0..n {
                    data[i * 4 + c] = p[i];
                }
            }
            Ok::<_, photocraft_psd::PsdError>(photocraft_psd::RgbaImage { left: 0, top: 0, width: h.width, height: h.height, data })
        });
        if let Ok(img) = rgba {
            let mut s = Surface::new(fmt);
            let vals: Vec<f32> = img
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .flat_map(|p| {
                    let v = photocraft_raster::from_rgba(&fmt, [p[0], p[1], p[2], p[3]].map(|x| f32::from(x) / 255.0));
                    v.into_iter()
                })
                .collect();
            s.write_region(canvas, &vals);
            s.prune();
            doc.layers.push(flat_background(s));
        }
    }

    // Extra (alpha / spot) channels of the merged image.
    if layered && let Some(Ok(all)) = &merged {
        let first = cc + usize::from(file.merged_has_alpha());
        let plane = h.row_bytes() * hh;
        let names = file.resource(1045).map(|r| unicode_names(&r.data)).unwrap_or_default();
        // Photoshop lists every extra channel, the merged transparency first when there is one.
        let skip = usize::from(file.merged_has_alpha() && names.len() == usize::from(h.channels).saturating_sub(cc));
        let names = names.get(skip..).unwrap_or_default();
        for (k, idx) in (first..usize::from(h.channels)).enumerate() {
            let Some(p) = all.get(idx * plane..(idx + 1) * plane) else { break };
            let mut s = Surface::new(cx.mask_fmt);
            s.write_interleaved(canvas, &be_to_ne_plane(p, depth, n));
            s.prune();
            let name = names.get(k).cloned().unwrap_or_else(|| format!("Alpha {}", k + 1));
            doc.channels.push(AlphaChannel::new(name, s));
        }
        // Channel options (overlay colour, opacity, colour indicates) and spot inks.
        if let Some(r) = file.resource(crate::channel_map::DISPLAY_INFO) {
            crate::channel_map::apply_display_info(&r.data, true, skip, &mut doc.channels);
        } else if let Some(r) = file.resource(crate::channel_map::DISPLAY_INFO_OLD) {
            crate::channel_map::apply_display_info(&r.data, false, skip, &mut doc.channels);
        }
        // Saved in Quick Mask mode: that channel becomes the document's Quick Mask again.
        if let Some(r) = file.resource(crate::channel_map::QUICK_MASK_INFO)
            && r.data.len() >= 2
            && let Some(k) = usize::from(u16::from_be_bytes([r.data[0], r.data[1]])).checked_sub(first)
            && k < doc.channels.len()
        {
            doc.quick_mask = Some(doc.channels.remove(k));
        }
    }

    // Layer comps (resource 1065 + per-layer `cmls`); the raw data stays for verbatim export.
    let raw_comps = doc.metadata.psd_resources.iter().find(|(id, _, _)| *id == crate::comps_map::LAYER_COMPS).map(|(_, _, d)| d.clone());
    let comp_warnings;
    (doc.layer_comps, doc.last_applied_comp, doc.last_document_state, comp_warnings) =
        crate::comps_map::decode_comps(raw_comps.as_deref().map(Vec::as_slice), &doc);
    cx.warnings.extend(comp_warnings);
    // Slices (resource 1050), after layer ids are known; the raw data stays for verbatim export.
    crate::slices_map::import(&mut doc);
    // Character and paragraph styles from the type layers' engine data.
    cx.warnings.extend(crate::text_styles_map::import(&mut doc));

    if ctl.cancelled() {
        return None;
    }
    ctl.progress(1.0);
    Some((doc, cx.warnings))
}
