//! Smart Objects and Smart Filters: Layer › Smart Objects, Layer › Smart Filter and
//! Filter › Convert for Smart Filters.
//!
//! A smart object keeps its **source** (embedded file bytes, or a linked path) plus a
//! `transform` and a stack of smart filters; its `cache` is the rendered appearance the
//! compositors draw. The engine owns re-rendering (compose stays a pure function of the
//! document):
//!
//! ```text
//! source bytes ──decode + composite (cached by content hash)──► source image (source pixels)
//!              ──transform (exact shift, or bicubic warp)──────► placed (document pixels)
//!              ──each visible smart filter, blended (blend/opacity)──► filtered
//!              ──filter mask mix (unfiltered ↔ filtered)───────► cache
//! ```
//!
//! The placed stage and the result after each filter are cached too (`stack_cache`), so a
//! re-render starts from the deepest stage it shares with an earlier one.
//!
//! Converting a layer embeds it as an in-memory `.pcraft` bundle of a nested document (exactly
//! lossless, layered, any depth/model). PSD placed layers keep their source in the preserved
//! global `lnk2` block (found by uuid), and keep Photoshop's rendering until something changes,
//! so an unedited PSD still round-trips byte for byte. PSD export (`photocraft_io::smart_map`)
//! writes every smart object back as one: its source embedded in `lnk2` (a `.pcraft` source as a
//! PSB), its smart filters in `filterFX` and its filter mask in `FEid`.

use std::sync::{Arc, Mutex};

use photocraft_algo::resample::translate_surface;
use photocraft_algo::transform::Homography;
use photocraft_color::{BlendMode, PixelFormat};
use photocraft_doc::{DocId, Document, Layer, LayerContent, LayerId, LayerMask, Metadata, SmartContentsId, SmartFilter, SmartObject, SmartSource};
use photocraft_geom::{Affine, Rect, Size};
use photocraft_raster::Surface;
use serde_json::{Value, json};
use stack_cache::{Stage, StageKey};

use crate::commands::{CommandSpec, blend_from_str, layer_param};
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

/// An open Edit Contents document and the shared contents it updates when saved or closed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmartLink {
    pub child: DocId,
    pub parent: DocId,
    pub layer: LayerId,
    pub contents_id: SmartContentsId,
}

/// PSD placed-layer keys that describe the smart object's source; stale once we change it.
const PLACED_KEYS: [&[u8; 4]; 3] = [b"SoLd", b"PlLd", b"SoLE"];

/// Command id of a Photoshop smart filter PhotoCraft does not implement (kept verbatim from PSD).
pub use photocraft_io::smart_map::UNSUPPORTED_FILTER;

fn other(msg: impl Into<String>) -> EngineError {
    EngineError::Other(msg.into())
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

// ---------- source encoding / decoding ----------

/// Encodes a document as embedded smart-object contents: an in-memory `.pcraft` bundle.
pub fn encode_source(doc: &Document) -> Result<Vec<u8>> {
    photocraft_format::save_to_bytes(doc, &Default::default()).map_err(|e| other(format!("can't encode smart object contents: {e}")))
}

/// Decodes smart-object contents (a `.pcraft` bundle, PSD/PSB, or any flat image format).
pub fn decode_source(file_name: &str, bytes: &[u8]) -> Result<Document> {
    crate::file_cmds::import(file_name, bytes).map_err(|e| other(format!("can't read smart object contents \"{file_name}\": {e}")))
}

fn base_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

fn read_file(path: &str) -> Option<Vec<u8>> {
    if path.is_empty() { None } else { photocraft_format::read_file(std::path::Path::new(path)).ok() }
}

/// The source file of a smart object: embedded bytes, the PSD's embedded linked-layer data (PSD
/// import keeps the placed layer's uuid as the path), or a linked file on disk.
pub fn source_bytes(meta: &Metadata, src: &SmartSource) -> Option<(String, Arc<Vec<u8>>)> {
    embedded_source_bytes(meta, src).or_else(|| match src {
        SmartSource::Linked { path } => read_file(path).map(|b| (base_name(path), Arc::new(b))),
        SmartSource::Embedded { .. } => None,
    })
}

/// Memory-only resolution also covers Photoshop's embedded files addressed by a `Linked` UUID.
fn embedded_source_bytes(meta: &Metadata, src: &SmartSource) -> Option<(String, Arc<Vec<u8>>)> {
    match src {
        SmartSource::Embedded { file_name, bytes } => Some((file_name.clone(), bytes.clone())),
        SmartSource::Linked { path } => photocraft_io::linked::find_linked_file(meta, path).map(|f| (f.file_name, Arc::new(f.bytes))),
    }
}

/// The actual path is stored in the document, not in the command's params. Recheck it before
/// reading, reusing embedded PSD bytes rather than parsing their global blocks a second time.
fn authorized_source_bytes(s: &Session, cmd: &str, meta: &Metadata, src: &SmartSource) -> Result<Option<(String, Arc<Vec<u8>>)>> {
    if let Some(source) = embedded_source_bytes(meta, src) {
        return Ok(Some(source));
    }
    let SmartSource::Linked { path } = src else { return Ok(None) };
    if let Some(gate) = s.authorize {
        gate(cmd, &json!({"path": path}))?;
    }
    Ok(read_file(path).map(|b| (base_name(path), Arc::new(b))))
}

/// Recheck a source's actual path before an untrusted command reads it: the path lives in the
/// document, not in the command's params. Embedded PSD UUIDs never reach the filesystem.
fn authorize_source(s: &Session, cmd: &str, doc: &Document, src: &SmartSource) -> Result<()> {
    if let (Some(gate), SmartSource::Linked { path }) = (s.authorize, src)
        && embedded_source_bytes(&doc.metadata, src).is_none()
    {
        gate(cmd, &json!({"path": path}))?;
    }
    Ok(())
}

/// A decoded, composited source in its own pixel space (`bounds` starts at the origin).
#[derive(Clone, Debug)]
pub struct SourceImage {
    pub surface: Arc<Surface>,
    pub bounds: Rect,
}

type CacheKey = ([u8; 32], PixelFormat);

struct CacheEntry {
    key: CacheKey,
    image: SourceImage,
    decodes: u32,
}

/// Decoded source composites by content hash, most recent last. Re-filtering or re-transforming
/// a smart object never decodes its contents again.
static SOURCE_CACHE: Mutex<Vec<CacheEntry>> = Mutex::new(Vec::new());
const CACHE_ENTRIES: usize = 32;
const CACHE_BYTES: usize = 1 << 30;

fn cache_key(bytes: &[u8], fmt: PixelFormat) -> CacheKey {
    (*blake3::hash(bytes).as_bytes(), fmt)
}

fn cache_get(key: &CacheKey) -> Option<SourceImage> {
    let mut c = SOURCE_CACHE.lock().ok()?;
    let i = c.iter().position(|e| e.key == *key)?;
    let e = c.remove(i);
    let img = e.image.clone();
    c.push(e);
    Some(img)
}

fn cache_put(key: CacheKey, image: SourceImage) {
    let Ok(mut c) = SOURCE_CACHE.lock() else { return };
    let decodes = match c.iter().position(|e| e.key == key) {
        Some(i) => c.remove(i).decodes + 1,
        None => 1,
    };
    c.push(CacheEntry { key, image, decodes });
    let size = |e: &CacheEntry| e.image.bounds.width() as usize * e.image.bounds.height() as usize * e.key.1.channels() * 4;
    while c.len() > CACHE_ENTRIES || (c.len() > 1 && c.iter().map(size).sum::<usize>() > CACHE_BYTES) {
        c.remove(0);
    }
}

/// How many times these contents were decoded at this pixel format (0 = never / evicted).
pub fn source_decode_count(bytes: &[u8], fmt: PixelFormat) -> u32 {
    let key = cache_key(bytes, fmt);
    SOURCE_CACHE.lock().ok().and_then(|c| c.iter().find(|e| e.key == key).map(|e| e.decodes)).unwrap_or(0)
}

fn buffer_to_surface(buf: &photocraft_compose::Buffer, fmt: PixelFormat) -> Surface {
    let n = fmt.channels();
    let mut data = vec![0.0f32; buf.px.len() * n];
    for (p, out) in buf.px.iter().zip(data.chunks_exact_mut(n)) {
        photocraft_raster::from_rgba_into(&fmt, *p, out);
    }
    let mut s = Surface::new(fmt);
    if !buf.rect.is_empty() {
        s.write_region(buf.rect, &data);
    }
    s.prune();
    s
}

/// An SVG source rendered at `scale` times its own size (cached per scale), so a vector smart
/// object stays sharp however it is placed.
pub fn svg_source_image(bytes: &[u8], fmt: PixelFormat, scale: f64) -> Result<SourceImage> {
    let mut h = blake3::Hasher::new();
    h.update(bytes);
    h.update(&scale.to_bits().to_le_bytes());
    let key = (*h.finalize().as_bytes(), fmt);
    if let Some(img) = cache_get(&key) {
        return Ok(img);
    }
    let buf = photocraft_io::svg::rasterize(bytes, scale as f32).map_err(|e| other(format!("can't render the SVG smart object: {e}")))?;
    let img = SourceImage { surface: Arc::new(buffer_to_surface(&buf, fmt)), bounds: buf.rect };
    cache_put(key, img.clone());
    Ok(img)
}

/// The scale a vector source is rasterised at for its placement: the larger axis scale of the
/// transform in 1/16 steps (so nudges reuse the cache), between 1/16 and 64.
fn vector_scale(sm: &SmartObject) -> f64 {
    let [a, b, c, d, ..] = sm.transform.m;
    let k = (a * a + b * b).sqrt().max((c * c + d * d).sqrt());
    if !k.is_finite() {
        return 1.0;
    }
    ((k * 16.0).ceil() / 16.0).clamp(1.0 / 16.0, 64.0)
}

/// The composited source image of `bytes` in pixel format `fmt` (decoded once, then cached).
pub fn source_image(file_name: &str, bytes: &[u8], fmt: PixelFormat) -> Result<SourceImage> {
    let key = cache_key(bytes, fmt);
    if let Some(img) = cache_get(&key) {
        return Ok(img);
    }
    let doc = decode_source(file_name, bytes)?;
    let buf = photocraft_compose::flatten(&doc);
    let img = SourceImage { surface: Arc::new(buffer_to_surface(&buf, fmt)), bounds: doc.bounds() };
    cache_put(key, img.clone());
    Ok(img)
}

/// Layer › Smart Objects › Stack Mode: the source's layers (or, when it holds a single group, the
/// group's layers) combined per pixel with `mode`. Cached like [`source_image`], keyed by mode.
pub fn stack_image(file_name: &str, bytes: &[u8], fmt: PixelFormat, mode: photocraft_doc::StackMode) -> Result<SourceImage> {
    use photocraft_algo::stack::Stat;
    let mut key = cache_key(bytes, fmt);
    key.0 = *blake3::hash(&[&key.0[..], b"stack:", mode.id().as_bytes()].concat()).as_bytes();
    if let Some(img) = cache_get(&key) {
        return Ok(img);
    }
    let doc = decode_source(file_name, bytes)?;
    let layers: &[Layer] = match doc.layers.as_slice() {
        [only] if only.is_group() => only.children().unwrap_or_default(),
        all => all,
    };
    let bounds = doc.bounds();
    let frames: Vec<Vec<[f32; 4]>> = layers
        .iter()
        .filter(|l| l.visible)
        .map(|l| {
            let mut one = l.clone();
            one.blend = BlendMode::Normal;
            one.clipped = false;
            photocraft_compose::render_layer(&one, bounds).px
        })
        .collect();
    if frames.is_empty() {
        return Err(EngineError::Other("the smart object's contents have no visible layers".into()));
    }
    let stat = match mode {
        photocraft_doc::StackMode::Entropy => Stat::Entropy,
        photocraft_doc::StackMode::Kurtosis => Stat::Kurtosis,
        photocraft_doc::StackMode::Maximum => Stat::Maximum,
        photocraft_doc::StackMode::Mean => Stat::Mean,
        photocraft_doc::StackMode::Median => Stat::Median,
        photocraft_doc::StackMode::Minimum => Stat::Minimum,
        photocraft_doc::StackMode::Range => Stat::Range,
        photocraft_doc::StackMode::Skewness => Stat::Skewness,
        photocraft_doc::StackMode::StandardDeviation => Stat::StandardDeviation,
        photocraft_doc::StackMode::Summation => Stat::Summation,
        photocraft_doc::StackMode::Variance => Stat::Variance,
    };
    let buf = photocraft_compose::Buffer { rect: bounds, px: photocraft_algo::stack::combine(&frames, stat) };
    let img = SourceImage { surface: Arc::new(buffer_to_surface(&buf, fmt)), bounds };
    cache_put(key, img.clone());
    Ok(img)
}

// ---------- rendering ----------

/// `top` blended over `base` with `mode` at `opacity` (a smart filter's blending options).
fn blend_surfaces(base: &Surface, top: &Surface, mode: BlendMode, opacity: f32) -> Surface {
    let fmt = base.format();
    let area = base.content_bounds().union(&top.content_bounds());
    let mut out = base.clone();
    if area.is_empty() {
        return out;
    }
    let n = fmt.channels();
    let (b, t) = (base.read_region(area), top.read_region(area));
    let mut o = vec![0.0f32; b.len()];
    for ((bp, tp), op) in b.chunks_exact(n).zip(t.chunks_exact(n)).zip(o.chunks_exact_mut(n)) {
        let rgba = photocraft_color::blend::composite(mode, photocraft_raster::to_rgba(&fmt, bp), photocraft_raster::to_rgba(&fmt, tp), opacity);
        photocraft_raster::from_rgba_into(&fmt, rgba, op);
    }
    out.write_region(area, &o);
    out.prune();
    out
}

/// Mixes `filtered` over `base` through the filter mask (premultiplied, any channel layout).
fn mask_mix(base: &Surface, filtered: &Surface, mask: &LayerMask) -> Surface {
    let fmt = base.format();
    let area = base.content_bounds().union(&filtered.content_bounds());
    let mut out = filtered.clone();
    if area.is_empty() {
        return out;
    }
    let n = fmt.channels();
    let (b, f) = (base.read_region(area), filtered.read_region(area));
    let mut k = Vec::new();
    mask.values_into(area, &mut k);
    let mut o = vec![0.0f32; b.len()];
    for (i, ((bp, fp), op)) in b.chunks_exact(n).zip(f.chunks_exact(n)).zip(o.chunks_exact_mut(n)).enumerate() {
        let m = k.get(i).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        if !fmt.alpha {
            for c in 0..n {
                op[c] = bp[c] + (fp[c] - bp[c]) * m;
            }
            continue;
        }
        let a = n - 1;
        let oa = bp[a] + (fp[a] - bp[a]) * m;
        op[a] = oa;
        for c in 0..a {
            let pm = bp[c] * bp[a] + (fp[c] * fp[a] - bp[c] * bp[a]) * m;
            op[c] = if oa > 0.0 { (pm / oa).clamp(0.0, 1.0) } else { 0.0 };
        }
    }
    out.write_region(area, &o);
    out.prune();
    out
}

/// Runs the smart filter stack on the placed (unfiltered) content.
pub fn apply_smart_filters(placed: &Surface, sm: &SmartObject, canvas: Rect) -> Surface {
    if !sm.filters_enabled || !sm.smart_filters.iter().any(|f| f.visible) {
        return placed.clone();
    }
    let mut cur = placed.clone();
    // Edge pixels repeat at the canvas ∪ placed-content edge, as for layer filters.
    let extent = canvas.union(&placed.content_bounds());
    for f in sm.smart_filters.iter().filter(|f| f.visible) {
        cur = apply_smart_filter(f, &cur, canvas, extent);
    }
    match sm.filter_mask.as_ref().filter(|m| m.enabled) {
        Some(m) => mask_mix(placed, &cur, m),
        None => cur,
    }
}

/// One visible smart filter over `cur`, blended with its blend mode and opacity.
fn apply_smart_filter(f: &SmartFilter, cur: &Surface, canvas: Rect, extent: Rect) -> Surface {
    // Unknown ids (e.g. Photoshop filters we don't implement) leave the pixels alone.
    let Some(out) = crate::filters::apply_filter_to_surface(&f.command, &f.params, cur, canvas, None, extent) else { return cur.clone() };
    if f.blend == BlendMode::Normal && f.opacity >= 1.0 { out } else { blend_surfaces(cur, &out, f.blend, f.opacity.clamp(0.0, 1.0)) }
}

/// The source placed in document pixels, before any smart filter.
fn place(name: &str, bytes: &[u8], fmt: PixelFormat, sm: &SmartObject) -> Result<Surface> {
    // A vector source is rendered at the placement's scale, and the placement is divided by
    // that scale so the rendered pixels land where the source units would.
    if sm.stack_mode.is_none() && sm.warp.is_none() && photocraft_io::svg::is_svg(bytes) {
        let k = vector_scale(sm);
        let img = svg_source_image(bytes, fmt, k)?;
        return Ok(match &sm.perspective {
            Some(p) => {
                let [h0, h1, h2, h3, h4, h5, h6, h7, h8] = *p;
                let h = Homography([h0 / k, h1 / k, h2, h3 / k, h4 / k, h5, h6 / k, h7 / k, h8]);
                photocraft_algo::warp::place_source_projective(&img.surface, img.bounds, &h, None)?
            }
            None => {
                let [a, b, c, d, e, f] = sm.transform.m;
                let t = Affine { m: [a / k, b / k, c / k, d / k, e, f] };
                photocraft_algo::warp::place_source(&img.surface, img.bounds, &t, None)?
            }
        });
    }
    let img = match sm.stack_mode {
        Some(mode) => stack_image(name, bytes, fmt, mode)?,
        None => source_image(name, bytes, fmt)?,
    };
    // Through the warp (source space) and the transform in one pass; whole-pixel moves are exact.
    Ok(match &sm.perspective {
        Some(p) => photocraft_algo::warp::place_source_projective(&img.surface, img.bounds, &Homography(*p), sm.warp.as_ref())?,
        None => photocraft_algo::warp::place_source(&img.surface, img.bounds, &sm.transform, sm.warp.as_ref())?,
    })
}

/// The box of a smart object's source in source pixels (what its warp's `us`/`vs` span), or `None`
/// when the source can't be read.
pub(crate) fn source_box(doc: &Document, sm: &SmartObject) -> Option<[f64; 4]> {
    let (name, bytes) = source_bytes(&doc.metadata, &sm.source)?;
    let fmt = doc.pixel_format();
    let img = match sm.stack_mode {
        Some(mode) => stack_image(&name, &bytes, fmt, mode),
        None => source_image(&name, &bytes, fmt),
    }
    .ok()?;
    let r = img.bounds;
    Some([f64::from(r.x0), f64::from(r.y0), f64::from(r.x1), f64::from(r.y1)])
}

/// Renders a smart object from its source. `Ok(None)` when the source is unavailable (a missing
/// linked file, or a PSD placed layer without embedded data): callers keep the existing cache.
pub fn render(doc: &Document, sm: &SmartObject) -> Result<Option<Surface>> {
    render_in(doc, sm, &JobCtx::new())
}

/// [`render`], reporting progress per smart filter to `ctx` and stopping between filters once it
/// is cancelled. Gives the same pixels as [`apply_smart_filters`] on the placed source, but
/// starts from the deepest stage of the stack already rendered (see [`stack_cache`]).
fn render_in(doc: &Document, sm: &SmartObject, ctx: &JobCtx) -> Result<Option<Surface>> {
    let Some((name, bytes)) = source_bytes(&doc.metadata, &sm.source) else { return Ok(None) };
    let (fmt, canvas) = (doc.pixel_format(), doc.bounds());
    let placed_key = stack_cache::placed_key(&bytes, fmt, sm, canvas);
    let placed = || -> Result<Stage> {
        if let Some(stage) = stack_cache::get(&placed_key) {
            return Ok(stage);
        }
        let surface = place(&name, &bytes, fmt, sm)?;
        // Edge pixels repeat at the canvas ∪ placed-content edge, as for layer filters.
        let stage = Stage { extent: canvas.union(&surface.content_bounds()), surface };
        stack_cache::put(placed_key, &stage);
        Ok(stage)
    };
    let filters: Vec<&SmartFilter> = sm.smart_filters.iter().filter(|f| sm.filters_enabled && f.visible).collect();
    if filters.is_empty() {
        return Ok(Some(placed()?.surface));
    }
    let keys: Vec<StageKey> = filters
        .iter()
        .scan(placed_key, |key, f| {
            *key = stack_cache::filter_key(key, f);
            Some(*key)
        })
        .collect();
    let deepest = keys.iter().enumerate().rev().find_map(|(i, key)| stack_cache::get(key).map(|stage| (i + 1, stage)));
    let (done, mut cur) = match deepest {
        Some(hit) => hit,
        None => (0, placed()?),
    };
    let n = filters.len() as f32;
    for (i, (f, key)) in filters.iter().zip(&keys).enumerate().skip(done) {
        ctx.check()?;
        ctx.progress(i as f32 / n, "");
        cur = Stage { surface: apply_smart_filter(f, &cur.surface, canvas, cur.extent), extent: cur.extent };
        stack_cache::put(*key, &cur);
    }
    ctx.progress(1.0, "");
    Ok(Some(match sm.filter_mask.as_ref().filter(|m| m.enabled) {
        Some(m) => mask_mix(&placed()?.surface, &cur.surface, m),
        None => cur.surface,
    }))
}

/// Bytes held by rendered smart filter stages.
pub fn stack_cache_bytes() -> usize {
    stack_cache::bytes()
}

/// Drop every rendered smart filter stage (Edit › Purge › All). Returns the bytes released.
pub fn purge_stack_cache() -> usize {
    stack_cache::purge()
}

/// The smart object's pixels below smart filter `index` (the input that filter edits): the placed
/// source and the filters under it, without the filter mask. `Ok(None)` when the source is
/// unavailable.
pub fn render_below_filter(doc: &Document, sm: &SmartObject, index: usize) -> Result<Option<Surface>> {
    let mut below = sm.clone();
    below.smart_filters.truncate(index);
    below.filter_mask = None;
    render(doc, &below)
}

/// Re-renders smart layer `l` (which lives in `doc`) in place. Returns false if its source is
/// unavailable (the cache is then left alone).
pub fn refresh_layer(doc: &Document, l: &mut Layer) -> Result<bool> {
    refresh_layer_in(doc, l, &JobCtx::new())
}

fn refresh_layer_in(doc: &Document, l: &mut Layer, ctx: &JobCtx) -> Result<bool> {
    let LayerContent::Smart(sm) = &mut l.content else { return Ok(false) };
    match render_in(doc, sm, ctx)? {
        Some(px) => {
            sm.cache = Some(px);
            Ok(true)
        }
        None => Ok(false),
    }
}

/// Re-renders the smart object `id` of `doc`.
pub fn refresh(doc: &mut Document, id: LayerId) -> Result<bool> {
    refresh_in(doc, id, &JobCtx::new())
}

fn refresh_in(doc: &mut Document, id: LayerId, ctx: &JobCtx) -> Result<bool> {
    let mut l = doc.layer(id).cloned().ok_or(EngineError::NoLayer(id))?;
    if !matches!(l.content, LayerContent::Smart(_)) {
        return Err(other("the layer is not a smart object"));
    }
    let ok = refresh_layer_in(doc, &mut l, ctx)?;
    if ok && let Some(dst) = doc.layer_mut(id) {
        dst.content = l.content;
    }
    Ok(ok)
}

fn refresh_or_fail(doc: &mut Document, id: LayerId) -> Result<()> {
    refresh_or_fail_in(doc, id, &JobCtx::new())
}

fn refresh_or_fail_in(doc: &mut Document, id: LayerId, ctx: &JobCtx) -> Result<()> {
    if refresh_in(doc, id, ctx)? {
        Ok(())
    } else {
        Err(other("the smart object's contents are unavailable (missing linked file?), so it can't be re-rendered"))
    }
}

fn smart(doc: &Document, id: LayerId) -> Result<&SmartObject> {
    match &doc.layer(id).ok_or(EngineError::NoLayer(id))?.content {
        LayerContent::Smart(sm) => Ok(sm),
        _ => Err(other("the layer is not a smart object")),
    }
}

fn smart_mut(doc: &mut Document, id: LayerId) -> Result<&mut SmartObject> {
    match &mut doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.content {
        LayerContent::Smart(sm) => Ok(sm),
        _ => Err(other("the layer is not a smart object")),
    }
}

fn mask_from_selection(sel: &Surface) -> LayerMask {
    LayerMask { surface: sel.clone(), enabled: true, linked: true, density: 1.0, feather: 0.0 }
}

/// Appends a smart filter (from a `filter.*` command) and re-renders. A selection becomes the
/// filter mask, as in Photoshop. Without a renderable source the filter is applied to the cached
/// pixels (and still recorded).
pub(crate) fn add_smart_filter(doc: &mut Document, id: LayerId, sf: photocraft_doc::SmartFilter, selection: Option<&Surface>) -> Result<()> {
    let canvas = doc.bounds();
    let renderable = source_bytes(&doc.metadata, &smart(doc, id)?.source).is_some();
    let sm = smart_mut(doc, id)?;
    if renderable {
        if let Some(sel) = selection
            && sm.filter_mask.is_none()
        {
            sm.filter_mask = Some(mask_from_selection(sel));
        }
        sm.smart_filters.push(sf);
        return refresh_or_fail(doc, id);
    }
    let bounds = selection.map(Surface::content_bounds).filter(|b| !b.is_empty()).unwrap_or(canvas);
    let cache = sm.cache.as_ref().ok_or_else(|| other("smart object has no pixels"))?;
    let out = crate::filters::apply_filter_to_surface(&sf.command, &sf.params, cache, bounds, selection, canvas)
        .ok_or_else(|| other(format!("unknown filter {}", sf.command)))?;
    sm.cache = Some(out);
    sm.smart_filters.push(sf);
    Ok(())
}

/// Rounds float noise off a composed transform (scale 0.1 then 10 should be exactly 1), so
/// repeated transforms land back on exact, lossless placements.
pub(crate) fn snap_affine(a: Affine) -> Affine {
    Affine { m: a.m.map(|v| if (v - v.round()).abs() < 1e-9 { v.round() } else { v }) }
}

/// Where the smart object's source pixels land in the document: its projective map (Distort,
/// Perspective) or its affine transform.
pub fn placement(sm: &SmartObject) -> Homography {
    sm.perspective.map(Homography).unwrap_or_else(|| affine_homography(&sm.transform))
}

pub(crate) fn affine_homography(a: &Affine) -> Homography {
    let [a, b, c, d, e, f] = a.m;
    Homography([a, c, e, b, d, f, 0.0, 0.0, 1.0])
}

/// Sets where the source lands. An affine map is stored exactly in `transform` (no
/// `perspective`); a projective one in `perspective`, with `transform` its affine approximation at
/// the source origin (for code that only needs scale and position).
pub(crate) fn set_placement(sm: &mut SmartObject, h: Homography) {
    let m = h.0;
    let n = if m[8].abs() > 1e-12 { m.map(|v| v / m[8]) } else { m };
    if n[6].abs() < 1e-12 && n[7].abs() < 1e-12 {
        sm.transform = snap_affine(Affine { m: [n[0], n[3], n[1], n[4], n[2], n[5]] });
        sm.perspective = None;
        return;
    }
    let h = Homography(n);
    let (o, x, y) = (h.apply(0.0, 0.0), h.apply(1.0, 0.0), h.apply(0.0, 1.0));
    sm.transform = Affine { m: [x.0 - o.0, x.1 - o.1, y.0 - o.0, y.1 - o.1, o.0, o.1] };
    sm.perspective = Some(n);
}

/// Composes `a` (document → document) onto the smart object's placement.
pub(crate) fn transform_placement(sm: &mut SmartObject, a: &Affine) {
    match sm.perspective {
        None => sm.transform = snap_affine(a.mul(&sm.transform)),
        Some(_) => set_placement(sm, affine_homography(a).mul(&placement(sm))),
    }
}

/// Moves a smart object by whole pixels without re-rendering.
pub(crate) fn shift_smart(sm: &mut SmartObject, dx: i32, dy: i32) {
    transform_placement(sm, &Affine::translate(dx as f64, dy as f64));
    if let Some(c) = &mut sm.cache {
        *c = crate::layer_multi_cmds::shift_surface(c, dx, dy);
    }
    if let Some(m) = &mut sm.filter_mask {
        m.surface = crate::layer_multi_cmds::shift_surface(&m.surface, dx, dy);
    }
}

/// Forget PSD placed-layer data that no longer describes the smart object (its source changed):
/// PSD export then writes fresh placed-layer blocks and embeds the new source.
fn detach_psd(l: &mut Layer) {
    if let LayerContent::Smart(sm) = &mut l.content {
        sm.psd_raw = None;
    }
    l.psd_blocks.retain(|(k, _)| !PLACED_KEYS.contains(&k));
}

// ---------- conversion ----------

/// Moves a layer (any kind) by whole pixels without re-rendering anything.
fn shift_layer(l: &mut Layer, dx: i32, dy: i32) {
    if dx == 0 && dy == 0 {
        return;
    }
    let a = Affine::translate(dx as f64, dy as f64);
    let mv = |s: &Surface| translate_surface(s, dx, dy);
    if let Some(m) = &mut l.mask {
        m.surface = mv(&m.surface);
    }
    if let Some(vm) = &mut l.vector_mask {
        vm.path = vm.path.transform(&a);
    }
    if let Some(fc) = &mut l.fill_cache {
        fc.surface = mv(&fc.surface);
    }
    match &mut l.content {
        LayerContent::Raster(s) => *s = mv(s),
        LayerContent::Text(t) => {
            t.transform = a.mul(&t.transform);
            if let Some(c) = &mut t.cache {
                *c = mv(c);
            }
        }
        LayerContent::Shape(sh) => {
            crate::vector_cmds::transform_shape(sh, &a);
            if let Some(c) = &mut sh.cache {
                *c = mv(c);
            }
        }
        LayerContent::Smart(sm) => shift_smart(sm, dx, dy),
        LayerContent::Group(g) => g.children.iter_mut().for_each(|c| shift_layer(c, dx, dy)),
        LayerContent::Adjustment(_) | LayerContent::Fill(_) => {}
    }
}

fn subtree_bounds(l: &Layer, canvas: Rect) -> Rect {
    match &l.content {
        LayerContent::Group(g) => g.children.iter().map(|c| subtree_bounds(c, canvas)).fold(Rect::EMPTY, |a, b| a.union(&b)),
        LayerContent::Adjustment(_) | LayerContent::Fill(_) => canvas,
        _ => l.surface().map_or(Rect::EMPTY, Surface::content_bounds),
    }
}

fn any_layer(l: &Layer, f: &dyn Fn(&Layer) -> bool) -> bool {
    f(l) || l.children().is_some_and(|c| c.iter().any(|c| any_layer(c, f)))
}

/// Builds the smart-object layer replacing `l` in `doc`: the layer (with its mask, effects and
/// fill opacity) becomes a nested document cropped to its rendered bounds; visibility, opacity,
/// blend mode, clipping and label stay on the smart layer. The cache is the exact composite, so
/// the document looks the same before and after.
pub fn layer_to_smart(doc: &Document, l: &Layer) -> Result<Layer> {
    let background = doc.is_background(l.id);
    let mut inner = l.clone();
    inner.visible = true;
    inner.opacity = 1.0;
    inner.clipped = false;
    inner.locks = Default::default();
    if !(inner.is_group() && inner.blend == BlendMode::PassThrough) {
        inner.blend = BlendMode::Normal;
    }
    if background {
        inner.name = "Layer 0".into();
    }
    let mut out = layers_to_smart(doc, &l.name, vec![inner])?;
    if background {
        out.name = "Layer 0".into();
    }
    out.visible = l.visible;
    out.opacity = l.opacity;
    out.blend = if l.blend == BlendMode::PassThrough { BlendMode::Normal } else { l.blend };
    out.clipped = l.clipped;
    out.label = l.label;
    if !background {
        out.locks = l.locks;
    }
    Ok(out)
}

/// Builds a smart-object layer named `name` whose nested document holds `layers` (bottom to top)
/// at its top level, as Photoshop does when several layers are converted: no wrapper group. The
/// nested document is cropped to the rendered bounds and the cache is the exact composite (the
/// same as the layers inside a pass-through group). The smart layer is visible, opaque and Normal.
pub fn layers_to_smart(doc: &Document, name: &str, layers: Vec<Layer>) -> Result<Layer> {
    let canvas = doc.bounds();
    let mut sub = Document::new(format!("{name}.pcraft"), doc.size, doc.mode, doc.depth);
    sub.resolution_dpi = doc.resolution_dpi;
    sub.icc_profile = doc.icc_profile.clone();
    sub.global_light = doc.global_light;
    // Pattern fills and pattern layer effects reference document-level patterns by ID.
    // Preserve those resources in the embedded document before rendering or saving it.
    sub.patterns = doc.patterns.clone();
    if layers.iter().any(|l| any_layer(l, &|x| matches!(x.content, LayerContent::Smart(_)))) {
        // Nested PSD placed layers find their embedded files here.
        sub.metadata.psd_global_blocks = doc.metadata.psd_global_blocks.clone();
    }
    let has_fx = layers.iter().any(|l| any_layer(l, &|x| x.effects.enabled && !x.effects.items.is_empty()));
    let region = layers.iter().map(|l| subtree_bounds(l, canvas)).fold(canvas, |a, b| a.union(&b)).inflate(if has_fx { 256 } else { 0 });
    sub.layers = layers;

    let buf = photocraft_compose::render(&sub, region);
    let w = region.width() as usize;
    let mut b = Rect::EMPTY;
    for (i, p) in buf.px.iter().enumerate() {
        if p[3] > 0.0 {
            let (x, y) = (region.x0 + (i % w) as i32, region.y0 + (i / w) as i32);
            b = b.union(&Rect::new(x, y, x + 1, y + 1));
        }
    }
    if b.is_empty() {
        return Err(other("the layer is empty, so there is nothing to convert"));
    }
    let fmt = doc.pixel_format();
    let mut cache = buffer_to_surface(&buf, fmt);
    cache = photocraft_algo::resample::crop_surface(&cache, b);
    cache.prune();

    sub.layers.iter_mut().for_each(|l| shift_layer(l, -b.x0, -b.y0));
    sub.size = Size::new(b.width(), b.height());
    let bytes = encode_source(&sub)?;
    // Seed the decode cache with the composite we already have.
    cache_put(cache_key(&bytes, fmt), SourceImage { surface: Arc::new(translate_surface(&cache, -b.x0, -b.y0)), bounds: sub.bounds() });

    let source = SmartSource::Embedded { file_name: sub.name.clone(), bytes: Arc::new(bytes) };
    Ok(Layer::new(name, LayerContent::Smart(SmartObject::new(source, Affine::translate(b.x0 as f64, b.y0 as f64), Some(cache)))))
}

fn convert(s: &mut Session, p: &Value) -> Result<Value> {
    let ids = match p.get("layer").and_then(Value::as_u64) {
        Some(id) => vec![LayerId(id)],
        None => crate::layer_multi_cmds::selected(s),
    };
    let new_id = s.edit("Convert to Smart Object", |doc, active| {
        let ids = crate::layer_multi_cmds::top_level(doc, &ids);
        let top = *ids.last().ok_or_else(|| other("no layer selected"))?;
        let l = doc.layer(top).ok_or(EngineError::NoLayer(top))?;
        let so = if ids.len() == 1 {
            layer_to_smart(doc, l)?
        } else {
            let children = ids.iter().map(|id| doc.layer(*id).cloned().ok_or(EngineError::NoLayer(*id))).collect::<Result<Vec<_>>>()?;
            layers_to_smart(doc, &l.name, children)?
        };
        let new_id = so.id;
        // As with Group Layers, place the result in the top selected root's parent. Insert
        // before removal so paths shifting in other parents cannot discard unselected content.
        doc.insert_above(Some(top), so);
        for id in ids {
            doc.remove(id).ok_or(EngineError::NoLayer(id))?;
        }
        *active = Some(new_id);
        Ok(new_id)
    })?;
    crate::layer_multi_cmds::reselect(s, vec![new_id], Some(new_id));
    Ok(json!({"layer": new_id.0}))
}

// ---------- contents ----------

fn via_copy(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    s.edit("New Smart Object via Copy", |doc, active| {
        let src = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        let mut copy = src.duplicate();
        copy.name = doc.copy_name(&src.name);
        if let LayerContent::Smart(sm) = &mut copy.content {
            // Independent contents: resolve to embedded bytes and drop the shared PSD uuid.
            sm.contents_id = SmartContentsId::fresh();
            if let Some((file_name, bytes)) = source_bytes(&doc.metadata, &sm.source) {
                sm.source = SmartSource::Embedded { file_name, bytes };
            }
        } else {
            return Err(other("the layer is not a smart object"));
        }
        detach_psd(&mut copy);
        let new_id = doc.insert_above(Some(id), copy);
        *active = Some(new_id);
        Ok(json!({"layer": new_id.0}))
    })
}

fn path_param<'a>(cmd: &str, p: &'a Value) -> Result<&'a str> {
    p.get("path").and_then(Value::as_str).filter(|s| !s.is_empty()).ok_or_else(|| bad(cmd, "pass `path`"))
}

/// Instances sharing contents in this document, including those inside groups.
fn instances(doc: &Document, contents_id: SmartContentsId) -> Vec<LayerId> {
    doc.walk().into_iter().filter_map(|(_, _, l)| matches!(&l.content, LayerContent::Smart(sm) if sm.contents_id == contents_id).then_some(l.id)).collect()
}

/// Swaps shared contents, keeping each instance's transform and filters, and re-renders.
fn set_source(s: &mut Session, p: &Value, label: &str, keep_psd: bool, make: impl FnOnce(&Metadata, &SmartSource) -> Result<SmartSource>) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let new = make(&doc.metadata, &smart(doc, id)?.source)?;
    set_shared_source(s, id, label, keep_psd, new)
}

// Keep the shared-instance transaction out of the per-command source-builder instantiations.
fn set_shared_source(s: &mut Session, id: LayerId, label: &str, keep_psd: bool, new: SmartSource) -> Result<Value> {
    let (parent, contents_id) = s.edit(label, |doc, _| {
        let contents_id = smart(doc, id)?.contents_id;
        for instance in instances(doc, contents_id) {
            smart_mut(doc, instance)?.source = new.clone();
            if !keep_psd {
                detach_psd(doc.layer_mut(instance).ok_or(EngineError::NoLayer(instance))?);
            }
            refresh_or_fail(doc, instance)?;
        }
        Ok((doc.id, contents_id))
    })?;
    // An old editor must not overwrite newly replaced/relinked contents on close. Keep its
    // document open, but the next Edit Contents opens the replacement source.
    s.smart_links.retain(|l| l.parent != parent || l.contents_id != contents_id);
    Ok(json!({"layer": id.0}))
}

fn replace_contents(s: &mut Session, p: &Value) -> Result<Value> {
    let path = path_param("layer.smartObjects.replaceContents", p)?.to_string();
    let bytes = photocraft_format::read_file(std::path::Path::new(&path)).map_err(|e| other(format!("can't read {path}: {e}")))?;
    let id = layer_param(s, p)?;
    replace_contents_bytes(s, id, &base_name(&path), bytes)
}

/// Replace Contents from a file's name and bytes: what the desktop and web file pickers hand over.
pub fn replace_contents_bytes(s: &mut Session, id: LayerId, file_name: &str, bytes: Vec<u8>) -> Result<Value> {
    decode_source(file_name, &bytes)?; // fail before touching the document
    set_shared_source(s, id, "Replace Contents", false, SmartSource::Embedded { file_name: file_name.to_string(), bytes: Arc::new(bytes) })
}

/// The file name and bytes of smart object `id`'s contents.
pub fn contents_of(doc: &Document, id: LayerId) -> Result<(String, Arc<Vec<u8>>)> {
    source_bytes(&doc.metadata, &smart(doc, id)?.source).ok_or_else(|| other("the smart object's contents are unavailable"))
}

/// Writes the smart object's file as it is. To a path with another extension (a `.png` for
/// `.psb` or `.pcraft` contents) the contents are opened and exported in that format instead.
fn export_contents(s: &mut Session, p: &Value) -> Result<Value> {
    let path = path_param("layer.smartObjects.exportContents", p)?;
    let id = layer_param(s, p)?;
    let (name, bytes) = contents_of(&s.active().ok_or(EngineError::NoDocument)?.doc, id)?;
    let ext = |f: &str| {
        let e = std::path::Path::new(f).extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase);
        e.map(|e| match e.as_str() {
            "jpeg" => "jpg".to_string(),
            "tiff" => "tif".to_string(),
            _ => e,
        })
    };
    if ext(path).is_none() || ext(path) == ext(&name) {
        crate::file_cmds::write_file(path, &bytes)?;
        return Ok(json!({"path": path, "fileName": name, "bytes": bytes.len(), "converted": false}));
    }
    let contents = decode_source(&name, &bytes)?;
    let r = photocraft_io::export(&contents, path, &Default::default()).map_err(|e| other(format!("can't export the contents as {}: {e}", base_name(path))))?;
    crate::file_cmds::write_file(path, &r.bytes)?;
    Ok(json!({"path": path, "fileName": name, "bytes": r.bytes.len(), "converted": true, "warnings": r.warnings}))
}

fn edit_contents(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let parent = st.doc.id;
    let contents_id = smart(&st.doc, id)?.contents_id;
    // Already open for editing: switch to that document rather than opening another copy.
    if let Some(index) =
        s.smart_links.iter().find(|l| l.parent == parent && l.contents_id == contents_id).and_then(|l| s.documents().iter().position(|d| d.doc.id == l.child))
    {
        s.set_active(index);
        return Ok(json!({"document": index, "parentLayer": id.0}));
    }
    let source = &smart(&st.doc, id)?.source;
    let (name, bytes) = authorized_source_bytes(s, "layer.smartObjects.editContents", &st.doc.metadata, source)?
        .ok_or_else(|| other("the smart object's contents are unavailable"))?;
    let mut child = decode_source(&name, &bytes)?;
    // Bundles keep their document id; each open copy needs its own.
    child.id = DocId::fresh();
    child.name = name;
    let index = s.add_document(child, None);
    // Admission may replace an ID already owned by another open document.
    let child_id = s.documents().get(index).ok_or(EngineError::NoDocument)?.doc.id;
    s.smart_links.retain(|l| l.child != child_id);
    s.smart_links.push(SmartLink { child: child_id, parent, layer: id, contents_id });
    Ok(json!({"document": index, "parentLayer": id.0}))
}

/// Writes an Edit Contents document back into all its parent instances (one undoable step in the
/// parent) and marks it saved. Returns false if `index` isn't an Edit Contents document.
pub fn commit_child(s: &mut Session, index: usize) -> Result<bool> {
    let Some(st) = s.docs.get(index) else { return Ok(false) };
    let child = st.doc.clone();
    let Some(link) = s.smart_links.iter().find(|l| l.child == child.id).copied() else { return Ok(false) };
    let parent = s.docs.iter().position(|d| d.doc.id == link.parent).ok_or_else(|| other("the smart object's document was closed"))?;
    let bytes = Arc::new(encode_source(&child)?);
    let prev = s.active;
    s.active = Some(parent);
    let r = s.edit("Edit Contents", |doc, _| {
        let ids = instances(doc, link.contents_id);
        if ids.is_empty() {
            return Err(other("the smart object's instances were removed"));
        }
        for id in ids {
            let sm = smart_mut(doc, id)?;
            let name = match &sm.source {
                SmartSource::Embedded { file_name, .. } => file_name.as_str(),
                SmartSource::Linked { .. } => child.name.as_str(),
            };
            let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
            sm.source = SmartSource::Embedded { file_name: format!("{stem}.pcraft"), bytes: bytes.clone() };
            detach_psd(doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?);
            refresh_or_fail(doc, id)?;
        }
        Ok(())
    });
    s.active = prev;
    r?;
    if let Some(st) = s.docs.get_mut(index) {
        st.saved_revision = st.revision;
    }
    Ok(true)
}

/// Called before a document closes: an edited Edit Contents document updates its parent.
pub(crate) fn on_close(s: &mut Session, index: usize) {
    let Some(st) = s.docs.get(index) else { return };
    let id = st.doc.id;
    if st.is_dirty() && s.smart_links.iter().any(|l| l.child == id) {
        let _ = commit_child(s, index);
    }
    s.smart_links.retain(|l| l.child != id && l.parent != id);
}

fn is_smart_child(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if s.smart_links.iter().any(|l| l.child == d.doc.id) { Ok(()) } else { Err("this document isn't smart object contents".into()) }
}

fn update_all(s: &mut Session) -> Result<Value> {
    let ids: Vec<LayerId> = s
        .active()
        .ok_or(EngineError::NoDocument)?
        .doc
        .walk()
        .into_iter()
        .filter(|(_, _, l)| matches!(&l.content, LayerContent::Smart(sm) if matches!(sm.source, SmartSource::Linked { .. })))
        .map(|(_, _, l)| l.id)
        .collect();
    let mut updated = Vec::new();
    s.edit("Update All Modified Content", |doc, _| {
        for id in &ids {
            if refresh(doc, *id)? {
                updated.push(id.0);
            }
        }
        Ok(())
    })?;
    Ok(json!({"updated": updated}))
}

// ---------- smart filter editing ----------

/// Edits the smart filters of a smart object and re-renders it. The edit itself is instant, so its
/// errors (a bad index, say) are reported at once; the re-render runs as a background job when
/// started with `Session::start`, so the window keeps drawing (with progress, cancellable) while
/// a long filter stack recomputes.
fn edit_filters(s: &mut Session, p: &Value, label: &str, f: impl FnOnce(&mut SmartObject) -> Result<()>) -> Result<Value> {
    let id = layer_param(s, p)?;
    let mut edited = smart(&s.active().ok_or(EngineError::NoDocument)?.doc, id)?.clone();
    f(&mut edited)?;
    crate::jobs::edit_job(
        s,
        label,
        move |doc, _, ctx| {
            *smart_mut(doc, id)? = edited;
            refresh_or_fail_in(doc, id, ctx)
        },
        move |()| json!({"layer": id.0}),
    )
}

fn filter_index(cmd: &str, p: &Value, sm: &SmartObject) -> Result<usize> {
    let n = sm.smart_filters.len();
    if n == 0 {
        return Err(other("the smart object has no smart filters"));
    }
    match p.get("index") {
        None => Ok(n - 1),
        Some(v) => v.as_u64().map(|i| i as usize).filter(|i| *i < n).ok_or_else(|| bad(cmd, format!("`index` must be 0..{n} (0 = bottom filter)"))),
    }
}

fn set_filter_visible(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.smartFilter.setVisible";
    edit_filters(s, p, "Smart Filter Visibility", |sm| {
        let i = filter_index(CMD, p, sm)?;
        let f = &mut sm.smart_filters[i];
        f.visible = p.get("visible").and_then(Value::as_bool).unwrap_or(!f.visible);
        Ok(())
    })
}

fn set_filter_params(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.smartFilter.setParams";
    let new = p.get("params").cloned().ok_or_else(|| bad(CMD, "pass `params` (merged into the filter's parameters)"))?;
    edit_filters(s, p, "Edit Smart Filter", |sm| {
        let i = filter_index(CMD, p, sm)?;
        let f = &mut sm.smart_filters[i];
        if f.command == photocraft_io::smart_map::UNSUPPORTED_FILTER {
            return Err(bad(CMD, "this smart filter isn't implemented in PhotoCraft: it is kept as is (it can be hidden, moved or deleted)"));
        }
        match (&mut f.params, new) {
            (Value::Object(old), Value::Object(n)) => old.extend(n),
            (slot, n) => *slot = n,
        }
        Ok(())
    })
}

fn blending_options(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.smartFilter.blendingOptions";
    let blend = match p.get("blend").and_then(Value::as_str) {
        Some(b) => Some(blend_from_str(b).filter(|m| *m != BlendMode::PassThrough).ok_or_else(|| bad(CMD, format!("unknown blend mode `{b}`")))?),
        None => None,
    };
    let opacity = p.get("opacity").and_then(Value::as_f64).map(|o| if o > 1.0 { o / 100.0 } else { o } as f32);
    edit_filters(s, p, "Smart Filter Blending Options", |sm| {
        let i = filter_index(CMD, p, sm)?;
        let f = &mut sm.smart_filters[i];
        if let Some(b) = blend {
            f.blend = b;
        }
        if let Some(o) = opacity {
            f.opacity = o.clamp(0.0, 1.0);
        }
        Ok(())
    })
}

fn delete_filter(s: &mut Session, p: &Value) -> Result<Value> {
    edit_filters(s, p, "Delete Smart Filter", |sm| {
        let i = filter_index("layer.smartFilter.delete", p, sm)?;
        sm.smart_filters.remove(i);
        Ok(())
    })
}

fn move_filter(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.smartFilter.move";
    edit_filters(s, p, "Move Smart Filter", |sm| {
        let i = filter_index(CMD, p, sm)?;
        let n = sm.smart_filters.len();
        let to = p.get("to").and_then(Value::as_u64).map(|t| t as usize).filter(|t| *t < n).ok_or_else(|| bad(CMD, format!("`to` must be 0..{n}")))?;
        let f = sm.smart_filters.remove(i);
        sm.smart_filters.insert(to, f);
        Ok(())
    })
}

/// Dragging smart filters onto another smart object in the Layers panel (#4271): they move there,
/// or are copied with `copy` (⌥-drag in Photoshop). One filter (`filter`) lands on top of the
/// target's filters; all of them (the Smart Filters row) are added on top too, with the filter
/// mask when the target has none. Both smart objects are re-rendered in one undo step.
fn transfer_filters(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "layer.smartFilter.transfer";
    let id_of = |key: &str| p.get(key).and_then(Value::as_u64).map(LayerId).ok_or_else(|| bad(CMD, format!("`{key}` must be a layer id")));
    let (from, to) = (id_of("from")?, id_of("to")?);
    if from == to {
        return Err(bad(CMD, "the smart filters are already on that layer"));
    }
    // The other `layer.smartFilter.*` commands name a filter `index`; here a guessed `index`
    // would otherwise be dropped and every filter moved.
    if p.get("index").is_some() {
        return Err(bad(CMD, "name the filter with `filter` (no `filter`: all of them)"));
    }
    let copy = p.get("copy").and_then(Value::as_bool).unwrap_or(false);
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    let src = smart(doc, from)?;
    smart(doc, to).map_err(|_| bad(CMD, "smart filters can only be dropped on a smart object"))?;
    let n = src.smart_filters.len();
    if n == 0 {
        return Err(other("the smart object has no smart filters"));
    }
    let index = match p.get("filter") {
        None | Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().map(|i| i as usize).filter(|i| *i < n).ok_or_else(|| bad(CMD, format!("`filter` must be 0..{n} (0 = bottom filter)")))?),
    };
    let (filters, mask) = match index {
        Some(i) => (vec![src.smart_filters[i].clone()], None),
        None => (src.smart_filters.clone(), src.filter_mask.clone()),
    };
    let label = match (copy, index.is_some()) {
        (true, true) => "Copy Smart Filter",
        (true, false) => "Copy Smart Filters",
        (false, true) => "Move Smart Filter",
        (false, false) => "Move Smart Filters",
    };
    crate::jobs::edit_job(
        s,
        label,
        move |doc, _, ctx| {
            let dst = smart_mut(doc, to)?;
            dst.smart_filters.extend(filters);
            if dst.filter_mask.is_none() {
                dst.filter_mask = mask;
            }
            refresh_or_fail_in(doc, to, ctx)?;
            if !copy {
                let src = smart_mut(doc, from)?;
                match index {
                    Some(i) => {
                        src.smart_filters.remove(i);
                    }
                    None => {
                        src.smart_filters.clear();
                        src.filter_mask = None;
                    }
                }
                refresh_or_fail_in(doc, from, ctx)?;
            }
            Ok(())
        },
        move |()| json!({"from": from.0, "to": to.0}),
    )
}

// ---------- enablement ----------

fn active_smart(s: &Session) -> std::result::Result<&SmartObject, String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    match &d.doc.layer(id).ok_or("no active layer")?.content {
        LayerContent::Smart(sm) => Ok(sm),
        other => Err(format!("the active layer is {} {} layer, not a smart object", other.article(), other.kind_name())),
    }
}
fn has_smart(s: &Session) -> std::result::Result<(), String> {
    active_smart(s).map(|_| ())
}
fn has_linked(s: &Session) -> std::result::Result<(), String> {
    match active_smart(s)?.source {
        SmartSource::Linked { .. } => Ok(()),
        _ => Err("the smart object is embedded".into()),
    }
}
fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}
fn has_smart_filters(s: &Session) -> std::result::Result<(), String> {
    if active_smart(s)?.smart_filters.is_empty() { Err("the smart object has no smart filters".into()) } else { Ok(()) }
}
fn has_filter_mask(s: &Session) -> std::result::Result<(), String> {
    active_smart(s)?.filter_mask.as_ref().map(|_| ()).ok_or_else(|| "the smart object has no filter mask".into())
}
fn convertible(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    d.doc.layer(id).map(|_| ()).ok_or_else(|| "no active layer".into())
}
fn not_smart(s: &Session) -> std::result::Result<(), String> {
    convertible(s)?;
    if active_smart(s).is_ok() { Err("the layer is already a smart object".into()) } else { Ok(()) }
}

macro_rules! spec {
    ($id:literal, $label:literal, $menu:expr, $params:literal, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: $params, enabled: $en, run: $run, journal: true }
    };
}

const SO: &[&str] = &["Layer", "Smart Objects"];
const SF: &[&str] = &["Layer", "Smart Filter"];

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "layer.smartObjects.convertToSmartObject",
            "Convert to Smart Object",
            SO,
            r##"{"layer":id?} (default: combine selected layers)"##,
            convertible,
            convert
        ),
        spec!(
            "filter.convertForSmartFilters",
            "Convert for Smart Filters",
            &["Filter"],
            r##"{"layer":id?} (default: combine selected layers)"##,
            not_smart,
            convert
        ),
        spec!("layer.smartObjects.newSmartObjectViaCopy", "New Smart Object via Copy", SO, r##"{"layer":id?}"##, has_smart, via_copy),
        spec!("layer.smartObjects.rasterize", "Rasterize", SO, r##"{"layer":id?}"##, has_smart, |s, p| s.execute("layer.rasterize.smartObject", p.clone())),
        spec!(
            "layer.smartObjects.editContents",
            "Edit Contents",
            SO,
            r##"{"layer":id?} → opens the contents as a new document; saving (layer.smartObjects.saveContents) or closing it updates the smart object"##,
            has_smart,
            edit_contents
        ),
        spec!("layer.smartObjects.saveContents", "Save Contents", &[], "{} (in an Edit Contents document)", is_smart_child, |s, _| {
            let i = s.active_index().ok_or(EngineError::NoDocument)?;
            Ok(json!({"updated": commit_child(s, i)?}))
        }),
        spec!("layer.smartObjects.replaceContents", "Replace Contents…", SO, r##"{"layer":id?,"path":str}"##, has_smart, replace_contents),
        spec!(
            "layer.smartObjects.exportContents",
            "Export Contents…",
            SO,
            r##"{"layer":id?,"path":str (another extension than the contents' own exports them in that format)}"##,
            has_smart,
            export_contents
        ),
        spec!("layer.smartObjects.relinkToFile", "Relink to File…", SO, r##"{"layer":id?,"path":str}"##, has_smart, |s, p| {
            let path = path_param("layer.smartObjects.relinkToFile", p)?.to_string();
            set_source(s, p, "Relink to File", false, |_, _| Ok(SmartSource::Linked { path }))
        }),
        spec!("layer.smartObjects.updateModifiedContent", "Update Modified Content", SO, r##"{"layer":id?}"##, has_linked, |s, p| {
            let id = layer_param(s, p)?;
            s.edit("Update Modified Content", |doc, _| refresh_or_fail(doc, id))?;
            Ok(json!({"layer": id.0}))
        }),
        spec!(
            "layer.smartObjects.updateChanged",
            "Update Changed Linked Content",
            &[],
            "{} — re-render the active document's linked smart objects whose files changed on disk since they were last seen (size or modification time); the desktop app does this by itself every two seconds, as Photoshop does while a document is open. Files seen for the first time are only remembered. One Update Modified Content undo step per file → {updated: [layer ids]}",
            |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            crate::link_watch::update_changed
        ),
        spec!(
            "layer.smartObjects.updateAllModifiedContent",
            "Update All Modified Content",
            SO,
            "{}",
            |s| s.active().map(|_| ()).ok_or_else(|| "no document open".into()),
            |s, _| update_all(s)
        ),
        spec!(
            "layer.smartObjects.convertToLayers",
            "Convert to Layers",
            SO,
            r##"{"layer":id?} (contents unpacked at the placement: one layer, or a group named after the smart object; smart filters are discarded)"##,
            has_smart,
            unpack::convert_to_layers
        ),
        spec!("layer.smartObjects.convertToEmbedded", "Convert to Embedded", SO, r##"{"layer":id?}"##, has_linked, |s, p| {
            set_source(s, p, "Convert to Embedded", true, |meta, src| {
                let (file_name, bytes) = source_bytes(meta, src).ok_or_else(|| other("the linked file can't be read"))?;
                Ok(SmartSource::Embedded { file_name, bytes })
            })
        }),
        spec!(
            "layer.smartObjects.convertToLinked",
            "Convert to Linked…",
            SO,
            r##"{"layer":id?,"path":str} (writes the contents there)"##,
            has_smart,
            |s, p| {
                let path = path_param("layer.smartObjects.convertToLinked", p)?.to_string();
                set_source(s, p, "Convert to Linked", false, |meta, src| {
                    let (_, bytes) = source_bytes(meta, src).ok_or_else(|| other("the smart object's contents are unavailable"))?;
                    crate::file_cmds::write_file(&path, &bytes)?;
                    Ok(SmartSource::Linked { path })
                })
            }
        ),
        // Smart filters
        spec!(
            "layer.smartFilter.disableSmartFilters",
            "Disable Smart Filters",
            SF,
            r##"{"layer":id?,"enabled":bool? (default: toggle)}"##,
            has_smart_filters,
            |s, p| {
                edit_filters(s, p, "Disable Smart Filters", |sm| {
                    sm.filters_enabled = p.get("enabled").and_then(Value::as_bool).unwrap_or(!sm.filters_enabled);
                    Ok(())
                })
            }
        ),
        spec!("layer.smartFilter.deleteFilterMask", "Delete Filter Mask", SF, r##"{"layer":id?}"##, has_filter_mask, |s, p| {
            edit_filters(s, p, "Delete Filter Mask", |sm| {
                sm.filter_mask = None;
                Ok(())
            })
        }),
        spec!(
            "layer.smartFilter.disableFilterMask",
            "Disable Filter Mask",
            SF,
            r##"{"layer":id?,"enabled":bool? (default: toggle)}"##,
            has_filter_mask,
            |s, p| {
                edit_filters(s, p, "Disable Filter Mask", |sm| {
                    if let Some(m) = &mut sm.filter_mask {
                        m.enabled = p.get("enabled").and_then(Value::as_bool).unwrap_or(!m.enabled);
                    }
                    Ok(())
                })
            }
        ),
        spec!(
            "layer.smartFilter.blendingOptions",
            "Blending Options…",
            SF,
            r##"{"layer":id?,"index":u32? (0 = bottom; default top),"blend":str?,"opacity":0..1?}"##,
            has_smart_filters,
            blending_options
        ),
        spec!("layer.smartFilter.clearSmartFilters", "Clear Smart Filters", SF, r##"{"layer":id?}"##, has_smart_filters, |s, p| {
            edit_filters(s, p, "Clear Smart Filters", |sm| {
                sm.smart_filters.clear();
                sm.filter_mask = None;
                Ok(())
            })
        }),
        spec!(
            "layer.smartFilter.setVisible",
            "Show/Hide Smart Filter",
            &[],
            r##"{"layer":id?,"index":u32?,"visible":bool? (default: toggle)}"##,
            has_smart_filters,
            set_filter_visible
        ),
        spec!(
            "layer.smartFilter.setParams",
            "Edit Smart Filter",
            &[],
            r##"{"layer":id?,"index":u32?,"params":{…} (merged)}"##,
            has_smart_filters,
            set_filter_params
        ),
        spec!("layer.smartFilter.delete", "Delete Smart Filter", &[], r##"{"layer":id?,"index":u32?}"##, has_smart_filters, delete_filter),
        spec!("layer.smartFilter.move", "Move Smart Filter", &[], r##"{"layer":id?,"index":u32?,"to":u32}"##, has_smart_filters, move_filter),
        spec!(
            "layer.smartFilter.transfer",
            "Move or Copy Smart Filters",
            &[],
            r##"{"from":id,"to":id,"filter":u32? (default: all, with the filter mask),"copy":bool=false}"##,
            has_doc,
            transfer_filters
        ),
    ]
}

mod stack_cache;
mod unpack;

#[cfg(test)]
mod tests;
