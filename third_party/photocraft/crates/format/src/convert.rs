//! Document ⇄ manifest conversion. Binary data is routed through a
//! [`Sink`] (save) or [`Fetch`] (load) keyed by BLAKE3 hash.

use std::sync::Arc;

use photocraft_color::{PixelFormat, SampleType};
use photocraft_doc::{
    AlphaChannel, CompAppearance, CompLayerState, DocId, Document, Effects, FillCache, Group, Layer, LayerComp, LayerContent, LayerId, LayerMask, Metadata,
    NamedPath, Pattern, ShapeLayer, SmartContentsId, SmartObject, SmartSource, TextLayer,
};
use photocraft_geom::{TILE_SIZE, TileCoord};
use photocraft_raster::{Surface, Tile, decode_pixel, encode_pixel};

use crate::manifest::*;
use crate::{FormatError, Result};

/// Deepest group nesting a bundle may hold — the one nesting limit every consumer of the layer
/// tree shares; see [`photocraft_doc::MAX_GROUP_DEPTH`].
pub use photocraft_doc::MAX_GROUP_DEPTH;

fn too_deep() -> FormatError {
    FormatError::LimitExceeded(format!("layer groups nested deeper than {MAX_GROUP_DEPTH}"))
}

/// Refuses to save a document the loader would reject for its group nesting.
pub(crate) fn check_nesting(layers: &[Layer]) -> Result<()> {
    // Bounded recursion: it stops at the first layer past the limit.
    fn too_deep_at(layers: &[Layer], depth: usize) -> bool {
        (depth > MAX_GROUP_DEPTH && !layers.is_empty())
            || layers.iter().any(|l| matches!(&l.content, LayerContent::Group(g) if too_deep_at(&g.children, depth + 1)))
    }
    if too_deep_at(layers, 0) { Err(too_deep()) } else { Ok(()) }
}

pub(crate) trait Sink {
    /// Register a tile, returning its hash.
    fn tile(&mut self, format: PixelFormat, tile: &Arc<Tile>) -> Hash;
    fn blob(&mut self, data: &Arc<Vec<u8>>) -> Hash;
}

pub(crate) trait Fetch {
    /// Canonical (little-endian) tile bytes of the given expected length.
    fn tile(&mut self, hash: &str, len: usize) -> Result<Vec<u8>>;
    fn blob(&mut self, hash: &str) -> Result<Arc<Vec<u8>>>;
}

pub(crate) fn hex(bytes: &[u8]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}

pub(crate) fn unhex(s: &str) -> Result<Vec<u8>> {
    if !s.len().is_multiple_of(2) {
        return Err(FormatError::corrupt(format!("bad hex `{s}`")));
    }
    (0..s.len())
        .step_by(2)
        .map(|i| {
            let pair = s.get(i..i + 2).unwrap_or("x");
            // `from_str_radix` accepts a leading `+`; reject anything that is not
            // two ASCII hex digits so a malformed string fails as corrupt (#1818).
            if !pair.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(FormatError::corrupt(format!("bad hex `{s}`")));
            }
            u8::from_str_radix(pair, 16).map_err(|_| FormatError::corrupt(format!("bad hex `{s}`")))
        })
        .collect()
}

fn key4(s: &str) -> Result<[u8; 4]> {
    unhex(s)?.try_into().map_err(|_| FormatError::corrupt(format!("bad 4-byte key `{s}`")))
}

pub(crate) fn is_valid_hash(h: &str) -> bool {
    h.len() == 64 && h.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Convert native-endian sample bytes to little-endian in place (and back:
/// the swap is an involution).
pub(crate) fn swap_to_le(bytes: &mut [u8], sample: SampleType) {
    if cfg!(target_endian = "big") {
        match sample {
            SampleType::U8 => {}
            SampleType::U16 => bytes.as_chunks_mut::<2>().0.iter_mut().for_each(|c| c.reverse()),
            SampleType::F32 => bytes.as_chunks_mut::<4>().0.iter_mut().for_each(|c| c.reverse()),
        }
    }
}

pub(crate) fn tile_len(format: &PixelFormat) -> usize {
    (TILE_SIZE * TILE_SIZE) as usize * format.bytes_per_pixel()
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

fn surface_m(s: &Surface, sink: &mut dyn Sink) -> SurfaceM {
    let f = s.format();
    let mut dp = vec![0u8; f.bytes_per_pixel()];
    encode_pixel(&f, &s.default_pixel(), &mut dp);
    swap_to_le(&mut dp, f.sample);
    SurfaceM { format: f, default: hex(&dp), tiles: s.tiles().map(|(c, t)| TileRef { tx: c.tx, ty: c.ty, hash: sink.tile(f, t) }).collect() }
}

fn opt_blob(b: &Option<Arc<Vec<u8>>>, sink: &mut dyn Sink) -> Option<Hash> {
    b.as_ref().map(|b| sink.blob(b))
}

fn video_m(v: &photocraft_doc::VideoData, sink: &mut dyn Sink) -> crate::manifest::VideoDataM {
    crate::manifest::VideoDataM {
        frames: v.frames.iter().map(|f| surface_m(f, sink)).collect(),
        original_frames: v.original_frames.as_ref().map(|frames| frames.iter().map(|f| surface_m(f, sink)).collect()),
        source: v.source.clone(),
        fps: v.fps,
        show_altered: v.show_altered,
    }
}

fn layer_m(l: &Layer, sink: &mut dyn Sink) -> LayerM {
    let content = match &l.content {
        LayerContent::Raster(s) => ContentM::Raster { surface: surface_m(s, sink) },
        LayerContent::Group(g) => {
            ContentM::Group { children: g.children.iter().map(|c| layer_m(c, sink)).collect(), expanded: g.expanded, artboard: g.artboard.clone() }
        }
        LayerContent::Adjustment(a) => ContentM::Adjustment { adjustment: a.clone() },
        LayerContent::Fill(f) => ContentM::Fill { fill: f.clone() },
        LayerContent::Text(t) => ContentM::Text {
            text: t.text.clone(),
            font_family: t.font_family.clone(),
            size_pt: t.size_pt,
            color: t.color,
            transform: t.transform,
            cache: t.cache.as_ref().map(|s| surface_m(s, sink)),
            psd_raw: opt_blob(&t.psd_raw, sink),
            runs: t.runs.clone(),
            paragraphs: t.paragraphs.clone(),
            shape: t.shape.clone(),
            overflow_visible: t.overflow_visible,
            block_align: t.block_align,
            orientation: t.orientation,
            antialias: t.antialias,
            warp: t.warp.clone(),
        },
        LayerContent::Shape(s) => ContentM::Shape {
            fill: s.fill.clone(),
            cache: s.cache.as_ref().map(|c| surface_m(c, sink)),
            psd_raw: opt_blob(&s.psd_raw, sink),
            path: s.path.clone(),
            stroke: s.stroke.clone(),
            live: s.live.clone(),
        },
        LayerContent::Smart(s) => ContentM::Smart {
            contents_id: Some(s.contents_id.0),
            source: match &s.source {
                SmartSource::Embedded { file_name, bytes } => SmartSourceM::Embedded { file_name: file_name.clone(), blob: sink.blob(bytes) },
                SmartSource::Linked { path } => SmartSourceM::Linked { path: path.clone() },
            },
            transform: s.transform,
            smart_filters: s.smart_filters.clone(),
            cache: s.cache.as_ref().map(|c| surface_m(c, sink)),
            psd_raw: opt_blob(&s.psd_raw, sink),
            filters_enabled: s.filters_enabled,
            filter_mask: s.filter_mask.as_ref().map(|m| MaskM {
                surface: surface_m(&m.surface, sink),
                enabled: m.enabled,
                linked: m.linked,
                density: m.density,
                feather: m.feather,
            }),
            warp: s.warp.clone(),
            stack_mode: s.stack_mode,
            perspective: s.perspective,
        },
    };
    LayerM {
        id: l.id.0,
        name: l.name.clone(),
        visible: l.visible,
        locks: l.locks,
        blend: l.blend,
        opacity: l.opacity,
        fill_opacity: l.fill_opacity,
        clipped: l.clipped,
        mask: l.mask.as_ref().map(|m| MaskM {
            surface: surface_m(&m.surface, sink),
            enabled: m.enabled,
            linked: m.linked,
            density: m.density,
            feather: m.feather,
        }),
        vector_mask: l.vector_mask.clone(),
        effects: EffectsM {
            enabled: l.effects.enabled,
            items: l.effects.items.clone(),
            psd_raw: opt_blob(&l.effects.psd_raw, sink),
            reference: l.effects.reference,
        },
        label: l.label,
        content,
        psd_blocks: l.psd_blocks.iter().map(|(k, d)| (hex(k), sink.blob(d))).collect(),
        psd_id: l.psd_id,
        fill_cache: l.fill_cache.as_ref().map(|fc| FillCacheM { fill: fc.fill.clone(), surface: surface_m(&fc.surface, sink) }),
        link_group: l.link_group,
        excluded_channels: l.excluded_channels,
        blend_if: l.blend_if.clone(),
        advanced: l.advanced,
        video: l.video.as_ref().map(|v| video_m(v, sink)),
    }
}

fn channel_m(c: &AlphaChannel, sink: &mut dyn Sink) -> ChannelM {
    // Destructured so a new `AlphaChannel` field fails to compile here until it is saved.
    let AlphaChannel { name, surface, spot, color, opacity, indicates } = c;
    ChannelM { name: name.clone(), surface: surface_m(surface, sink), spot: *spot, color: *color, opacity: *opacity, indicates: *indicates }
}

pub(crate) fn doc_m(d: &Document, sink: &mut dyn Sink) -> DocM {
    DocM {
        id: d.id.0,
        name: d.name.clone(),
        size: d.size,
        resolution_dpi: d.resolution_dpi,
        mode: d.mode,
        depth: d.depth,
        icc_profile: opt_blob(&d.icc_profile, sink),
        layers: d.layers.iter().map(|l| layer_m(l, sink)).collect(),
        channels: d.channels.iter().map(|c| channel_m(c, sink)).collect(),
        guides: d.guides.clone(),
        ruler_origin: d.ruler_origin,
        selection: d.selection.as_ref().map(|s| surface_m(s, sink)),
        metadata: MetadataM {
            xmp: d.metadata.xmp.clone(),
            exif: opt_blob(&d.metadata.exif, sink),
            text: d.metadata.text.clone(),
            psd_resources: d.metadata.psd_resources.iter().map(|(id, n, b)| (*id, n.clone(), sink.blob(b))).collect(),
            psd_global_blocks: d.metadata.psd_global_blocks.iter().map(|(s, k, b)| (hex(s), hex(k), sink.blob(b))).collect(),
        },
        global_light: d.global_light,
        paths: d.paths.iter().map(|p| NamedPathM { name: p.name.clone(), path: p.path.clone(), psd_raw: opt_blob(&p.psd_raw, sink) }).collect(),
        work_path: d.work_path.clone(),
        clipping_path: d.clipping_path.clone(),
        quick_mask: d.quick_mask.as_ref().map(|c| channel_m(c, sink)),
        quick_mask_unselected: d.quick_mask_unselected,
        patterns: d
            .patterns
            .iter()
            .map(|p| PatternM { id: p.id.clone(), name: p.name.clone(), width: p.width, height: p.height, surface: surface_m(&p.surface, sink) })
            .collect(),
        color_table: d.color_table.clone(),
        duotone: d.duotone.clone(),
        variables: d.variables.clone(),
        timeline: d.timeline.clone(),
        file_encoding: d.file_encoding,
        layer_comps: d.layer_comps.iter().map(|c| comp_m(c, sink)).collect(),
        last_applied_comp: d.last_applied_comp,
        last_document_state: d.last_document_state.as_ref().map(|c| comp_m(c, sink)),
        measurement: d.measurement.clone(),
        notes: d.notes.clone(),
        text_styles: d.text_styles.clone(),
        slices: d.slices.clone(),
    }
}

fn effects_m(e: &Effects, sink: &mut dyn Sink) -> EffectsM {
    EffectsM { enabled: e.enabled, items: e.items.clone(), psd_raw: opt_blob(&e.psd_raw, sink), reference: e.reference }
}

fn comp_m(c: &LayerComp, sink: &mut dyn Sink) -> LayerCompM {
    // Destructured so a new `LayerComp` field fails to compile here until it is saved.
    let LayerComp { id, name, comment, apply_visibility, apply_position, apply_appearance, states } = c;
    LayerCompM {
        id: *id,
        name: name.clone(),
        comment: comment.clone(),
        apply_visibility: *apply_visibility,
        apply_position: *apply_position,
        apply_appearance: *apply_appearance,
        states: states
            .iter()
            .map(|s| {
                let CompLayerState { layer, visible, position, appearance } = s;
                CompStateM {
                    layer: layer.0,
                    visible: *visible,
                    position: *position,
                    appearance: appearance.as_ref().map(|a| {
                        let CompAppearance { blend, opacity, fill_opacity, effects } = a;
                        CompAppearanceM { blend: *blend, opacity: *opacity, fill_opacity: *fill_opacity, effects: effects_m(effects, sink) }
                    }),
                }
            })
            .collect(),
    }
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------

pub(crate) struct Loader<'a> {
    pub fetch: &'a mut dyn Fetch,
    pub preserve_ids: bool,
    /// Stored layer id → loaded id (differs when ids are remapped), for layer comp states.
    pub id_map: std::collections::HashMap<u64, LayerId>,
    pub contents_id_map: std::collections::HashMap<u64, SmartContentsId>,
    pub legacy_linked_contents: std::collections::HashMap<String, SmartContentsId>,
    /// Decoded tiles by content hash and sample type. Saves store a tile once however often it
    /// is referenced (a solid fill shares one tile), so a load decodes and charges the byte
    /// budget once per distinct tile and hands every reference the same copy-on-write `Arc`
    /// (#3315). Grouped by sample type: in-memory byte order depends on it.
    pub tiles: std::collections::HashMap<SampleType, std::collections::HashMap<Hash, Arc<Tile>>>,
}

impl Loader<'_> {
    fn video(&mut self, m: &crate::manifest::VideoDataM) -> Result<photocraft_doc::VideoData> {
        let frames = m.frames.iter().map(|f| self.surface(f)).collect::<Result<Vec<_>>>()?;
        if m.original_frames.as_ref().is_some_and(|o| o.len() != frames.len()) {
            return Err(FormatError::corrupt("video original frame count differs from edited frame count"));
        }
        let original_frames = m.original_frames.as_ref().map(|source| source.iter().map(|f| self.surface(f)).collect::<Result<Vec<_>>>()).transpose()?;
        Ok(photocraft_doc::VideoData { frames, original_frames, source: m.source.clone(), fps: m.fps, show_altered: m.show_altered })
    }

    fn surface(&mut self, m: &SurfaceM) -> Result<Surface> {
        let f = m.format;
        let mut dp = unhex(&m.default)?;
        if dp.len() != f.bytes_per_pixel() {
            return Err(FormatError::corrupt("default pixel has wrong size for its format"));
        }
        swap_to_le(&mut dp, f.sample);
        let mut s = Surface::with_default(f, &decode_pixel(&f, &dp));
        let len = tile_len(&f);
        // Both edges of a tile's rectangle must fit in i32 (`TileCoord::rect` multiplies by the
        // tile size): -8388608 ..= 8388606. A damaged manifest can name any index (#938).
        let fits = |i: i32| i.checked_mul(TILE_SIZE).is_some() && i.checked_add(1).and_then(|e| e.checked_mul(TILE_SIZE)).is_some();
        for t in &m.tiles {
            if !fits(t.tx) || !fits(t.ty) {
                return Err(FormatError::corrupt(format!("tile ({}, {}) is outside the coordinate range", t.tx, t.ty)));
            }
            let c = TileCoord::new(t.tx, t.ty);
            if s.tile(c).is_some() {
                return Err(FormatError::corrupt(format!("duplicate tile ({}, {})", t.tx, t.ty)));
            }
            if let Some(tile) = self.tiles.get(&f.sample).and_then(|m| m.get(&t.hash)) {
                // One hash names one byte string, so a cached tile of another length means
                // the same hash was referenced from formats of different pixel sizes.
                if tile.bytes().len() != len {
                    return Err(FormatError::corrupt(format!("tile {} is referenced with different sizes", t.hash)));
                }
                s.put_tiles([(c, tile.clone())]);
                continue;
            }
            let mut bytes = self.fetch.tile(&t.hash, len)?;
            swap_to_le(&mut bytes, f.sample);
            s.tile_mut(c).bytes_mut().copy_from_slice(&bytes);
            if let Some(tile) = s.tile(c) {
                self.tiles.entry(f.sample).or_default().insert(t.hash.clone(), tile.clone());
            }
        }
        Ok(s)
    }

    fn opt_surface(&mut self, m: &Option<SurfaceM>) -> Result<Option<Surface>> {
        m.as_ref().map(|s| self.surface(s)).transpose()
    }

    fn opt_blob(&mut self, h: &Option<Hash>) -> Result<Option<Arc<Vec<u8>>>> {
        h.as_ref().map(|h| self.fetch.blob(h)).transpose()
    }

    fn id(&mut self, raw: u64) -> LayerId {
        let id = if self.preserve_ids { LayerId(raw) } else { LayerId::fresh() };
        self.id_map.insert(raw, id);
        id
    }

    fn contents_id(&mut self, raw: Option<u64>, source: &SmartSourceM) -> SmartContentsId {
        if let Some(raw) = raw {
            return *self.contents_id_map.entry(raw).or_insert_with(|| if self.preserve_ids { SmartContentsId(raw) } else { SmartContentsId::fresh() });
        }
        // Legacy embedded blobs may be deduplicated despite representing independent objects.
        // Only a nonempty linked path provides an existing shared-source identity.
        if let SmartSourceM::Linked { path } = source
            && !path.is_empty()
        {
            return *self.legacy_linked_contents.entry(path.clone()).or_insert_with(SmartContentsId::fresh);
        }
        SmartContentsId::fresh()
    }

    fn comp(&mut self, m: &LayerCompM) -> Result<LayerComp> {
        let mut states = Vec::with_capacity(m.states.len());
        for s in &m.states {
            let appearance = match &s.appearance {
                Some(a) => Some(CompAppearance {
                    blend: a.blend,
                    opacity: a.opacity,
                    fill_opacity: a.fill_opacity,
                    effects: Effects {
                        enabled: a.effects.enabled,
                        items: a.effects.items.clone(),
                        psd_raw: self.opt_blob(&a.effects.psd_raw)?,
                        reference: a.effects.reference,
                    },
                }),
                None => None,
            };
            // States of layers deleted before the save keep their (now dangling) id, which
            // `layerComp.updateWarnings` reports; it must not collide with a live layer.
            let layer = self.id_map.get(&s.layer).copied().unwrap_or(LayerId(u64::MAX - s.layer.min(u64::MAX / 2)));
            states.push(CompLayerState { layer, visible: s.visible, position: s.position, appearance });
        }
        Ok(LayerComp {
            id: m.id,
            name: m.name.clone(),
            comment: m.comment.clone(),
            apply_visibility: m.apply_visibility,
            apply_position: m.apply_position,
            apply_appearance: m.apply_appearance,
            states,
        })
    }

    fn layer(&mut self, m: &LayerM, depth: usize) -> Result<Layer> {
        if depth > MAX_GROUP_DEPTH {
            return Err(too_deep());
        }
        // Children first, from this small frame: the conversion's large locals stay off the stack
        // that grows with each nesting level.
        let mut children = Vec::new();
        if let ContentM::Group { children: kids, .. } = &m.content {
            children.reserve(kids.len());
            for c in kids {
                children.push(self.layer(c, depth + 1)?);
            }
        }
        self.layer_with(m, children)
    }

    /// One layer, given its already converted group children.
    #[inline(never)]
    fn layer_with(&mut self, m: &LayerM, children: Vec<Layer>) -> Result<Layer> {
        let content = match &m.content {
            ContentM::Raster { surface } => LayerContent::Raster(self.surface(surface)?),
            ContentM::Group { expanded, artboard, .. } => LayerContent::Group(Group { artboard: artboard.clone(), children, expanded: *expanded }),
            ContentM::Adjustment { adjustment } => LayerContent::Adjustment(adjustment.clone()),
            ContentM::Fill { fill } => LayerContent::Fill(fill.clone()),
            ContentM::Text {
                text,
                font_family,
                size_pt,
                color,
                transform,
                cache,
                psd_raw,
                runs,
                paragraphs,
                shape,
                overflow_visible,
                block_align,
                orientation,
                antialias,
                warp,
            } => LayerContent::Text(TextLayer {
                text: text.clone(),
                font_family: font_family.clone(),
                size_pt: *size_pt,
                color: *color,
                transform: *transform,
                cache: self.opt_surface(cache)?,
                psd_raw: self.opt_blob(psd_raw)?,
                runs: runs.clone(),
                paragraphs: paragraphs.clone(),
                shape: shape.clone(),
                overflow_visible: *overflow_visible,
                block_align: *block_align,
                orientation: *orientation,
                antialias: *antialias,
                warp: warp.clone(),
            }),
            ContentM::Shape { fill, cache, psd_raw, path, stroke, live } => LayerContent::Shape(ShapeLayer {
                path: path.clone(),
                fill: fill.clone(),
                stroke: stroke.clone(),
                live: live.clone(),
                cache: self.opt_surface(cache)?,
                psd_raw: self.opt_blob(psd_raw)?,
            }),
            ContentM::Smart { contents_id, source, transform, smart_filters, cache, psd_raw, filters_enabled, filter_mask, warp, stack_mode, perspective } => {
                LayerContent::Smart(SmartObject {
                    contents_id: self.contents_id(*contents_id, source),
                    source: match source {
                        SmartSourceM::Embedded { file_name, blob } => SmartSource::Embedded { file_name: file_name.clone(), bytes: self.fetch.blob(blob)? },
                        SmartSourceM::Linked { path } => SmartSource::Linked { path: path.clone() },
                    },
                    transform: *transform,
                    smart_filters: smart_filters.clone(),
                    cache: self.opt_surface(cache)?,
                    psd_raw: self.opt_blob(psd_raw)?,
                    filters_enabled: *filters_enabled,
                    filter_mask: match filter_mask {
                        Some(mm) => Some(LayerMask {
                            surface: self.surface(&mm.surface)?,
                            enabled: mm.enabled,
                            linked: mm.linked,
                            density: mm.density,
                            feather: mm.feather,
                        }),
                        None => None,
                    },
                    warp: warp.clone().filter(|w| w.mesh.as_ref().is_none_or(|m| m.is_valid())),
                    stack_mode: *stack_mode,
                    perspective: perspective.filter(|p| p.iter().all(|v| v.is_finite())),
                })
            }
        };
        let mask = match &m.mask {
            Some(mm) => {
                Some(LayerMask { surface: self.surface(&mm.surface)?, enabled: mm.enabled, linked: mm.linked, density: mm.density, feather: mm.feather })
            }
            None => None,
        };
        let fill_cache = match &m.fill_cache {
            Some(fc) => Some(FillCache { fill: fc.fill.clone(), surface: self.surface(&fc.surface)? }),
            None => None,
        };
        let mut psd_blocks = Vec::with_capacity(m.psd_blocks.len());
        for (k, h) in &m.psd_blocks {
            psd_blocks.push((key4(k)?, self.fetch.blob(h)?));
        }
        Ok(Layer {
            id: self.id(m.id),
            name: m.name.clone(),
            visible: m.visible,
            locks: m.locks,
            blend: m.blend,
            opacity: m.opacity,
            fill_opacity: m.fill_opacity,
            clipped: m.clipped,
            mask,
            vector_mask: m.vector_mask.clone(),
            effects: Effects {
                enabled: m.effects.enabled,
                items: m.effects.items.clone(),
                psd_raw: self.opt_blob(&m.effects.psd_raw)?,
                reference: m.effects.reference,
            },
            label: m.label,
            content,
            psd_blocks,
            psd_id: m.psd_id,
            fill_cache,
            link_group: m.link_group,
            excluded_channels: m.excluded_channels,
            blend_if: m.blend_if.clone(),
            advanced: m.advanced,
            video: m.video.as_ref().map(|v| self.video(v)).transpose()?,
        })
    }

    pub(crate) fn document(&mut self, m: &DocM) -> Result<Document> {
        // Reserve every stored id before allocating identities missing from older bundles. If an
        // id is implausibly large, remap the whole document while retaining shared contents groups.
        if self.preserve_ids {
            fn max_ids(layers: &[LayerM], depth: usize, max: &mut u64) -> Result<()> {
                if depth > MAX_GROUP_DEPTH && !layers.is_empty() {
                    return Err(too_deep());
                }
                for l in layers {
                    *max = (*max).max(l.id);
                    match &l.content {
                        ContentM::Group { children, .. } => max_ids(children, depth + 1, max)?,
                        ContentM::Smart { contents_id: Some(id), .. } => *max = (*max).max(*id),
                        _ => {}
                    }
                }
                Ok(())
            }
            let mut max = m.id;
            max_ids(&m.layers, 0, &mut max)?;
            self.preserve_ids = reserve_ids_through(max);
        }
        let layers = m.layers.iter().map(|l| self.layer(l, 0)).collect::<Result<Vec<_>>>()?;
        let mut channels = Vec::with_capacity(m.channels.len());
        for c in &m.channels {
            channels.push(self.channel(c)?);
        }
        let quick_mask = m.quick_mask.as_ref().map(|c| self.channel(c)).transpose()?;
        let mut md = Metadata {
            xmp: m.metadata.xmp.clone(),
            exif: self.opt_blob(&m.metadata.exif)?,
            text: m.metadata.text.clone(),
            psd_resources: Vec::new(),
            psd_global_blocks: Vec::new(),
        };
        for (id, n, h) in &m.metadata.psd_resources {
            md.psd_resources.push((*id, n.clone(), self.fetch.blob(h)?));
        }
        for (s, k, h) in &m.metadata.psd_global_blocks {
            md.psd_global_blocks.push((key4(s)?, key4(k)?, self.fetch.blob(h)?));
        }
        let mut paths = Vec::with_capacity(m.paths.len());
        for p in &m.paths {
            paths.push(NamedPath { name: p.name.clone(), path: p.path.clone(), psd_raw: self.opt_blob(&p.psd_raw)? });
        }
        let mut patterns = Vec::with_capacity(m.patterns.len());
        for p in &m.patterns {
            // Sizes come from the file: a pattern with no tiles can claim any size, and renderers
            // allocate per pixel.
            let max = photocraft_doc::pattern::MAX_EDGE;
            if p.width == 0 || p.height == 0 {
                return Err(FormatError::corrupt(format!("pattern `{}` is {}×{} px", p.name, p.width, p.height)));
            }
            if p.width > max || p.height > max {
                return Err(FormatError::LimitExceeded(format!("pattern `{}` is {}×{} px (max {max} per side)", p.name, p.width, p.height)));
            }
            patterns.push(Pattern { id: p.id.clone(), name: p.name.clone(), width: p.width, height: p.height, surface: self.surface(&p.surface)? });
        }
        let layer_comps = m.layer_comps.iter().map(|c| self.comp(c)).collect::<Result<Vec<_>>>()?;
        let last_document_state = m.last_document_state.as_ref().map(|c| self.comp(c)).transpose()?;
        // Layer-based slices follow their layer through id remapping; a dangling one becomes a
        // user slice (its rect stays where it was).
        let mut slices = m.slices.clone();
        for sl in &mut slices.list {
            if let Some(l) = sl.layer {
                sl.layer = self.id_map.get(&l.0).copied();
                if sl.layer.is_none() && sl.origin == photocraft_doc::SliceOrigin::Layer {
                    sl.origin = photocraft_doc::SliceOrigin::User;
                }
            }
        }
        let id = if self.preserve_ids { DocId(m.id) } else { DocId::fresh() };
        Ok(Document {
            id,
            name: m.name.clone(),
            size: m.size,
            resolution_dpi: m.resolution_dpi,
            mode: m.mode,
            depth: m.depth,
            icc_profile: self.opt_blob(&m.icc_profile)?,
            layers,
            channels,
            guides: m.guides.clone(),
            ruler_origin: m.ruler_origin,
            selection: self.opt_surface(&m.selection)?,
            metadata: md,
            global_light: m.global_light,
            paths,
            work_path: m.work_path.clone(),
            clipping_path: m.clipping_path.clone(),
            quick_mask,
            quick_mask_unselected: m.quick_mask_unselected,
            patterns,
            color_table: m.color_table.clone(),
            duotone: m.duotone.clone(),
            variables: m.variables.clone(),
            timeline: m.timeline.clone(),
            file_encoding: m.file_encoding,
            layer_comps,
            last_applied_comp: m.last_applied_comp,
            last_document_state,
            measurement: m.measurement.clone(),
            notes: m.notes.clone(),
            text_styles: m.text_styles.clone(),
            slices,
        })
    }

    fn channel(&mut self, c: &ChannelM) -> Result<AlphaChannel> {
        Ok(AlphaChannel { name: c.name.clone(), surface: self.surface(&c.surface)?, spot: c.spot, color: c.color, opacity: c.opacity, indicates: c.indicates })
    }
}

/// Advance the global id counter past `max` so ids loaded from a file never
/// collide with ids minted later in this process. Returns `false` if `max`
/// is implausibly far ahead (the caller then remaps ids instead).
pub(crate) fn reserve_ids_through(max: u64) -> bool {
    let probe = LayerId::fresh().0;
    if max < probe {
        return true;
    }
    if max - probe > 10_000_000 {
        return false;
    }
    photocraft_doc::ensure_ids_above(max);
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unhex_rejects_leading_plus() {
        // `u8::from_str_radix` accepts a leading `+`; unhex must not (#1818).
        assert!(unhex("+5").is_err());
        assert!(unhex("ab+5").is_err());
        assert!(unhex("+5cd").is_err());
    }

    #[test]
    fn unhex_rejects_non_hex_characters() {
        assert!(unhex("zz").is_err());
        assert!(unhex(" 1").is_err());
        assert!(unhex("g0").is_err());
    }

    #[test]
    fn unhex_accepts_valid_hex() {
        assert_eq!(unhex("00").unwrap(), vec![0x00]);
        assert_eq!(unhex("ff").unwrap(), vec![0xff]);
        assert_eq!(unhex("0123456789abcdef").unwrap(), vec![0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef]);
        assert_eq!(unhex("FF").unwrap(), vec![0xff]);
    }

    #[test]
    fn unhex_rejects_odd_length() {
        assert!(unhex("0").is_err());
        assert!(unhex("abc").is_err());
    }
}
