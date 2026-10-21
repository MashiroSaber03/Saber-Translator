//! Live canvas preview for the Image › Adjustments dialogs (`adjust_dialog`).
//!
//! While a dialog is open on a pixel layer the canvas shows the committed document with a
//! temporary adjustment layer (the dialog's settings, the selection as its mask) clipped to the
//! target layer. That composite equals what the destructive command will produce (see the tests:
//! within 1/255 for every kind, depth and mode below), but it is drawn by the regular canvas path,
//! so on the GPU canvas a change re-runs the wgpu compositor instead of the command on a CPU
//! proxy (the layer textures stay resident; just the adjustment's uniforms / LUT change):
//! - over the target's area within the selection only (`switch_region`), and of that only what
//!   the view shows; the rest catches up as the view moves (`uncovered`);
//! - zoomed out on a large document, on a copy reduced to about the screen's resolution, built
//!   once per session ([`gpu_proxy`]), unless Preferences › Performance › Low Resolution Previews
//!   is off.
//!
//! OK still runs the real command (one history step); Cancel just drops the preview. The base
//! document (with the preview layer) and the region are built once per dialog session; a change
//! only swaps the layer's adjustment in a shallow copy.
//!
//! Cases where an adjustment-layer composite differs from the destructive result, or where it
//! would not be cheap, keep the CPU proxy preview (`canvas::ensure_filter_preview`), see
//! [`unsupported`]:
//! - the target is not a plain pixel layer (smart object, type, shape, group: the command edits
//!   a smart object's filters or refuses), or is itself clipped (a clipped adjustment would act on
//!   the whole clipping group, not just the target);
//! - the target uses Blend If (its "This Layer" ranges test the layer's adjusted colour);
//! - the target's untouched area is visible (a surface without alpha whose content doesn't cover
//!   the canvas: the command only edits content);
//! - the command targets an alpha channel, the Quick Mask or a single colour channel;
//! - CMYK, Lab and the other non-RGB modes except Grayscale: the command converts each pixel to
//!   RGB, adjusts and converts back (CMYK even changes when the settings are neutral), which a
//!   composite in display RGB can't reproduce;
//! - in Grayscale documents, kinds that make colour from gray (Vibrance, Color Balance, Black &
//!   White, Photo Filter, Channel Mixer, Gradient Map, colorized Hue/Saturation): the command
//!   folds each pixel back to gray, the composite would carry the colour through;
//! - on the GPU canvas, documents the wgpu compositor doesn't cover (it would fall back to a
//!   full CPU composite per change); on the CPU canvas, documents above the proxy size.

use std::sync::Arc;

use photocraft_color::ColorMode;
use photocraft_doc::{Adjustment, DocId, Document, Layer, LayerContent, LayerId, LayerMask};
use photocraft_geom::Rect;
use serde_json::Value;

use crate::PhotocraftApp;

/// Id of the temporary preview layer. Fixed so the GPU compositor keeps its mask (the selection)
/// resident from one change to the next. Never reaches a committed document.
pub const PREVIEW_LAYER: LayerId = LayerId(u64::MAX - 77);

/// Tag of the preview keys handed to the canvas (`canvas::display_doc`).
const TAG: u64 = 1 << 59;
/// Bits of a preview key that identify the session (document state + target); below them the
/// settings hash.
const SESSION_SHIFT: u32 = 40;

/// Why `target` in `doc` can't preview as a clipped adjustment layer, or None when it can.
pub fn unsupported(doc: &Document, target: LayerId) -> Option<&'static str> {
    if !matches!(doc.mode, ColorMode::Rgb | ColorMode::Grayscale) {
        return Some("colour mode composites through display RGB");
    }
    let Some(l) = doc.layer(target) else { return Some("no target layer") };
    let LayerContent::Raster(s) = &l.content else { return Some("not a pixel layer") };
    if l.clipped {
        return Some("the target is clipped");
    }
    if photocraft_compose::blend_if_active(l, doc.mode) {
        return Some(tl!("Blend If"));
    }
    let shows_default = !s.format().alpha || s.default_pixel().last().is_some_and(|a| *a > 0.0);
    if shows_default && !s.content_bounds().contains_rect(&doc.bounds()) {
        return Some("untouched area is visible");
    }
    None
}

/// Whether `adj` maps gray to gray. Others make colour in a grayscale document, which the command
/// folds back to gray per pixel while the composite carries it to the end.
fn keeps_gray(adj: &Adjustment) -> bool {
    match adj {
        Adjustment::BrightnessContrast { .. }
        | Adjustment::Levels { .. }
        | Adjustment::Curves { .. }
        | Adjustment::Exposure { .. }
        | Adjustment::Posterize { .. }
        | Adjustment::Threshold { .. }
        | Adjustment::Invert => true,
        Adjustment::HueSaturation { colorize, .. } => !colorize,
        _ => false,
    }
}

/// `doc` with a placeholder temporary adjustment layer clipped to `target` (the selection as its
/// mask): built once per dialog session, then [`with_settings`] per change. Errors when the target
/// isn't eligible ([`unsupported`]).
pub fn base_document(doc: &Document, target: LayerId) -> Result<Document, String> {
    if let Some(why) = unsupported(doc, target) {
        return Err(why.into());
    }
    let mut l = Layer::new(tl!("Adjustment Preview"), LayerContent::Adjustment(Adjustment::Invert));
    l.id = PREVIEW_LAYER;
    l.clipped = true;
    if let Some(sel) = &doc.selection {
        l.mask = Some(LayerMask { surface: sel.clone(), enabled: true, linked: false, density: 1.0, feather: 0.0 });
    }
    let mut out = doc.clone();
    out.insert_above(Some(target), l);
    Ok(out)
}

/// A [`base_document`] with the dialog's settings in its preview layer (a shallow copy: pixel
/// tiles are shared). Errors on invalid settings and on kinds the layer can't reproduce here.
pub fn with_settings(base: &Document, kind: &str, params: &Value) -> Result<Document, String> {
    set_adjustment(base, PREVIEW_LAYER, kind, params)
}

fn set_adjustment(base: &Document, layer: LayerId, kind: &str, params: &Value) -> Result<Document, String> {
    let adj = photocraft_engine::adjust_params::from_params(kind, params, None, base.mode).map_err(|e| e.to_string())?;
    if base.mode == ColorMode::Grayscale && !keeps_gray(&adj) {
        return Err("colour result in a grayscale document".into());
    }
    let mut out = base.clone();
    let l = out.layer_mut(layer).ok_or("no preview layer")?;
    l.content = LayerContent::Adjustment(adj);
    Ok(out)
}

/// `doc` with the dialog's adjustment as a temporary layer clipped to `target` (the selection as
/// its mask): [`base_document`] then [`with_settings`].
pub fn preview_document(doc: &Document, target: LayerId, kind: &str, params: &Value) -> Result<Document, String> {
    with_settings(&base_document(doc, target)?, kind, params)
}

/// The canvas area the preview can change: the target's content within the selection, grown by
/// the reach of layer effects (the target's and its groups').
pub fn region(doc: &Document, target: LayerId) -> Rect {
    let Some(s) = doc.layer(target).and_then(Layer::surface) else { return Rect::EMPTY };
    let mut r = s.content_bounds().intersect(&doc.bounds());
    if let Some(sel) = &doc.selection {
        r = r.intersect(&sel.content_bounds());
    }
    if r.is_empty() {
        return r;
    }
    r.inflate(crate::canvas::effect_reach(&doc.layers)).intersect(&doc.bounds())
}

/// Per-dialog preview state on the app.
pub struct AdjustPreview {
    doc: DocId,
    revision: u64,
    /// Session part of the keys (document, revision, target, channel target).
    session: u64,
    /// The session's [`base_document`], or None when it previews on the CPU proxy.
    base: Option<Arc<Document>>,
    region: Rect,
    /// Hash of the kind, settings and Preview checkbox (0: no dialog).
    hash: u64,
    /// Whether the layer path handles these settings (false: CPU proxy preview).
    layer_path: bool,
    /// The preview document and its key, or None (proxy path, or the Preview checkbox is off).
    shown: Option<(Arc<Document>, u64)>,
    /// When the settings last changed (ms), for throttling secondary views.
    changed_ms: f64,
    /// The dialog's kind and parameters.
    settings: (String, Value),
    /// Reduced-resolution copy for zoomed-out views (see [`gpu_proxy`]).
    proxy: Option<GpuProxy>,
    /// The main canvas texture shows preview `key` only within this rect (the view when it was
    /// composited, see [`switch_damage`]); elsewhere in the region it is older.
    covered: Option<(u64, Rect)>,
}

/// A [`base_document`] scaled down by `k` for zoomed-out GPU previews, with its own document and
/// layer ids (so its textures live beside the full-size document's), plus the preview built from
/// it for the current settings.
struct GpuProxy {
    k: u32,
    base: Arc<Document>,
    /// (settings hash, preview).
    shown: Option<(u64, Arc<Document>)>,
    /// Settings hash last composited into the proxy's texture.
    drawn: Option<u64>,
}

/// The open adjustment dialog: (kind, params, Preview on).
fn dialog(app: &PhotocraftApp) -> Option<(String, Value, bool)> {
    let d = app.ui.dialogs.iter().find(|d| crate::adjust_dialog::owns(&d.fields))?;
    let kind = d.fields.get("__adjust")?.as_str()?.to_string();
    let preview = d.fields.get("__preview").and_then(Value::as_bool).unwrap_or(true);
    Some((kind, crate::filter_dialog::params_of(&d.fields), preview))
}

/// FNV-1a over `parts`, never 0.
fn hash(parts: &[&[u8]]) -> u64 {
    parts.iter().flat_map(|p| p.iter()).fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)) | 1
}

/// Whether the layer path can draw `doc` cheaply on this app's canvas.
fn cheap(app: &PhotocraftApp, doc: &Document) -> bool {
    match &app.gpu {
        Some(g) => g.supports(doc),
        None => crate::proxy::preview_factor(doc, crate::proxy::reduced_previews(app)) <= 1,
    }
}

/// Bring the cached preview up to date for document `idx` (the active one) and return it.
fn update(app: &mut PhotocraftApp, idx: usize) -> Option<&AdjustPreview> {
    if app.session.active_index() != Some(idx) {
        return None;
    }
    let Some((kind, params, preview)) = dialog(app) else {
        // Keep a closed dialog's session for the canvas's switch back (see `switch_region`) but
        // never show it.
        if let Some(p) = app.adjust_preview.as_mut() {
            (p.hash, p.layer_path, p.shown) = (0, false, None);
        }
        return None;
    };
    let st = app.session.documents().get(idx)?;
    let (doc_id, revision, target) = (st.doc.id, st.revision, st.active_layer?);
    // Commands that target a channel (Channels panel / Quick Mask) or a layer mask keep the proxy
    // preview.
    let channel_target = st.channel_view.target != photocraft_engine::channel_cmds::ChannelTarget::Composite
        || st.doc.quick_mask.is_some()
        || !app.with_mask_target(&format!("image.adjustments.{kind}"), Value::Null).is_null();
    let session = hash(&[&doc_id.0.to_le_bytes(), &revision.to_le_bytes(), &target.0.to_le_bytes(), &[u8::from(channel_target)]]);
    let h = hash(&[kind.as_bytes(), params.to_string().as_bytes(), &[u8::from(preview)]]);
    if let Some(p) = &app.adjust_preview
        && (p.doc, p.session, p.hash) == (doc_id, session, h)
    {
        return app.adjust_preview.as_ref();
    }
    if app.adjust_preview.as_ref().is_none_or(|p| (p.doc, p.session) != (doc_id, session)) {
        let doc = st.doc.clone();
        let base = if channel_target { None } else { base_document(&doc, target).ok().filter(|b| cheap(app, b)).map(Arc::new) };
        let region = if base.is_some() { region(&doc, target) } else { Rect::EMPTY };
        app.adjust_preview = Some(AdjustPreview {
            doc: doc_id,
            revision,
            session,
            base,
            region,
            hash: 0,
            layer_path: false,
            shown: None,
            changed_ms: crate::gpu_canvas::now_ms(),
            settings: (String::new(), Value::Null),
            proxy: None,
            covered: None,
        });
    }
    let p = app.adjust_preview.as_mut()?;
    let built = p.base.as_ref().and_then(|b| with_settings(b, &kind, &params).ok());
    let key = TAG | ((session & 0x7_ffff) << SESSION_SHIFT) | (h & ((1 << SESSION_SHIFT) - 1));
    p.layer_path = built.is_some();
    p.shown = built.filter(|_| preview).map(|d| (Arc::new(d), key));
    p.hash = h;
    p.changed_ms = crate::gpu_canvas::now_ms();
    p.settings = (kind, params);
    app.adjust_preview.as_ref()
}

/// The document the canvas shows for document `idx` while an adjustment dialog previews through
/// the layer path: (document, preview key).
pub fn display_doc(app: &mut PhotocraftApp, idx: usize) -> Option<(Arc<Document>, u64)> {
    update(app, idx)?.shown.clone()
}

/// Whether the open adjustment dialog previews through the layer path (so the CPU proxy preview
/// must stay off).
pub fn on_layer(app: &mut PhotocraftApp, idx: usize) -> bool {
    update(app, idx).is_some_and(|p| p.layer_path)
}

/// The canvas area to recomposite when document `idx` (at `revision`) switches from the preview
/// key `from` to `to` (raw keys, display transform removed): both must be this dialog's previews
/// or the committed document (0). None means "everything".
pub fn switch_region(app: &PhotocraftApp, doc: DocId, revision: u64, from: u64, to: u64) -> Option<Rect> {
    let p = app.adjust_preview.as_ref()?;
    if p.doc != doc || p.revision != revision {
        return None;
    }
    let prefix = TAG | ((p.session & 0x7_ffff) << SESSION_SHIFT);
    let ours = |k: u64| k == 0 || k >> SESSION_SHIFT == prefix >> SESSION_SHIFT;
    (ours(from) && ours(to) && (from != 0 || to != 0)).then_some(p.region)
}

/// The key of the preview currently shown, if any.
pub fn shown_key(app: &PhotocraftApp) -> Option<u64> {
    app.adjust_preview.as_ref()?.shown.as_ref().map(|s| s.1)
}

/// The part of a switch's region (from [`switch_region`]) to composite now: to a preview, only
/// what the view shows (`visible`, document pixels), remembering it so [`uncovered`] catches up
/// as the view moves; back to the committed document (key 0), the whole region.
pub fn switch_damage(app: &mut PhotocraftApp, to: u64, region: Rect, visible: Rect) -> Rect {
    let Some(p) = app.adjust_preview.as_mut() else { return region };
    if to == 0 {
        p.covered = None;
        return region;
    }
    let shown = region.intersect(&visible);
    p.covered = Some((to, shown));
    shown
}

/// Where the main canvas texture (showing preview `key`) is out of date within `visible`: the
/// region's visible part when the view moved past what was composited. Marks it covered.
pub fn uncovered(app: &mut PhotocraftApp, doc: DocId, key: u64, visible: Rect) -> Option<Rect> {
    let p = app.adjust_preview.as_mut().filter(|p| p.doc == doc)?;
    let (k, cov) = p.covered?;
    if k != key {
        return None;
    }
    let want = p.region.intersect(&visible);
    if want.is_empty() || cov.contains_rect(&want) {
        return None;
    }
    p.covered = Some((k, want));
    Some(want)
}

/// True while the preview's settings changed within the last `ms` milliseconds (secondary views
/// such as the Navigator wait for the settings to settle instead of recompositing every change).
pub fn settling(app: &PhotocraftApp, ms: f64) -> bool {
    app.adjust_preview.as_ref().is_some_and(|p| p.shown.is_some() && crate::gpu_canvas::now_ms() - p.changed_ms < ms)
}

/// Documents above this size preview zoomed-out views on a reduced copy ([`gpu_proxy`]).
const PROXY_MIN_PIXELS: u64 = 4_000_000;
/// Proxy document ids: the document's id with this bit flipped.
const PROXY_DOC_BIT: u64 = 1 << 60;
/// Proxy layer ids: the layer's id with this bit flipped (document layer ids are small counters;
/// [`PREVIEW_LAYER`] has it set).
const PROXY_LAYER_BIT: u64 = 1 << 62;

fn flip_ids(layers: &mut [Layer]) {
    for l in layers {
        l.id = LayerId(l.id.0 ^ PROXY_LAYER_BIT);
        if let Some(ch) = l.children_mut() {
            flip_ids(ch);
        }
    }
}

/// Proxy factor for a view at `zoom` (physical screen pixels per document pixel): the largest
/// power of two that keeps the proxy at least at the screen's resolution; 1 when the view is too
/// close for a proxy.
pub fn proxy_factor(zoom: f32) -> u32 {
    if zoom.is_nan() || zoom <= 0.0 {
        return 1;
    }
    let mut k = 1u32;
    while k < 64 && (k * 2) as f32 * zoom <= 1.0 {
        k *= 2;
    }
    k
}

/// A [`base_document`] scaled down by `k` with its own document and layer ids (so the GPU keeps
/// its textures beside the full-size document's).
pub fn proxy_base(base: &Document, k: u32) -> Document {
    let mut d = photocraft_compose::proxy::proxy_document(base, k);
    d.id = photocraft_doc::DocId(base.id.0 ^ PROXY_DOC_BIT);
    flip_ids(&mut d.layers);
    d
}

/// [`with_settings`] for a [`proxy_base`].
pub fn proxy_with_settings(proxy: &Document, kind: &str, params: &Value) -> Result<Document, String> {
    set_adjustment(proxy, LayerId(PREVIEW_LAYER.0 ^ PROXY_LAYER_BIT), kind, params)
}

/// A frame of the zoomed-out preview: composite `doc` (the proxy preview, its own document id)
/// into its GPU texture over `damage` (None: everything), then draw it scaled by `k`.
pub struct ProxyFrame {
    pub k: u32,
    pub doc: Arc<Document>,
    pub damage: Option<Rect>,
    /// False when the texture already shows this preview.
    pub stale: bool,
    hash: u64,
}

/// The zoomed-out preview for document `idx` at `zoom`, when the dialog previews through the layer
/// and the view is far enough out that a reduced copy shows the same detail as the full-size
/// composite: each change then composites ~1/k² of the pixels. Built once per session and factor.
pub fn gpu_proxy(app: &mut PhotocraftApp, idx: usize, zoom: f32) -> Option<ProxyFrame> {
    if !crate::proxy::reduced_previews(app) {
        return None;
    }
    let gpu = app.gpu.clone()?;
    update(app, idx)?;
    let p = app.adjust_preview.as_mut()?;
    p.shown.as_ref()?;
    let base = p.base.clone()?;
    let k = proxy_factor(zoom);
    if k < 2 || base.size.area() <= PROXY_MIN_PIXELS || !photocraft_compose::proxy::proxy_faithful(&base) {
        return None;
    }
    if p.proxy.as_ref().is_none_or(|g| g.k != k) {
        let d = proxy_base(&base, k);
        if !gpu.supports(&d) {
            return None;
        }
        p.proxy = Some(GpuProxy { k, base: Arc::new(d), shown: None, drawn: None });
    }
    let (kind, params) = &p.settings;
    let h = p.hash;
    let g = p.proxy.as_mut()?;
    if g.shown.as_ref().is_none_or(|(sh, _)| *sh != h) {
        let d = proxy_with_settings(&g.base, kind, params).ok()?;
        g.shown = Some((h, Arc::new(d)));
    }
    let doc = g.shown.as_ref()?.1.clone();
    let r = p.region;
    let kk = k as i32;
    let scaled = Rect::new(r.x0.div_euclid(kk), r.y0.div_euclid(kk), r.x1.div_euclid(kk) + 1, r.y1.div_euclid(kk) + 1).intersect(&doc.bounds());
    let damage = g.drawn.map(|_| scaled);
    Some(ProxyFrame { k, doc, damage, stale: g.drawn != Some(h), hash: h })
}

/// Record that `frame` was composited into the proxy's texture.
pub fn proxy_drawn(app: &mut PhotocraftApp, frame: &ProxyFrame) {
    if let Some(g) = app.adjust_preview.as_mut().and_then(|p| p.proxy.as_mut())
        && g.k == frame.k
    {
        g.drawn = Some(frame.hash);
    }
}

pub(crate) fn retain_documents(app: &mut PhotocraftApp) {
    // Native reopen can reuse the ID and revision; neither the old base document nor its
    // coverage and upload markers may survive after that document leaves the session.
    if app.adjust_preview.as_ref().is_some_and(|p| !app.session.documents().iter().any(|st| st.doc.id == p.doc)) {
        app.adjust_preview = None;
    }
}

/// GPU texture keys the preview uses besides the documents' own (keep them alive).
pub fn gpu_keys(app: &PhotocraftApp) -> Option<u64> {
    let p = app.adjust_preview.as_ref().filter(|p| p.hash != 0)?;
    p.proxy.as_ref().map(|g| g.base.id.0)
}

#[cfg(test)]
#[path = "adjust_preview_tests.rs"]
mod tests;
