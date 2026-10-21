//! Filter › Filter Gallery…: a full-window dialog like Photoshop's: a big preview on the left,
//! the category folders with thumbnails in the middle, and on the right OK / Cancel, the
//! selected effect's filter and settings, and the effect-layer stack (new, delete, reorder,
//! show/hide). The stack is applied bottom to top.
//!
//! The preview shows what OK applies at the preview's zoom, like Photoshop's: the stack runs at
//! full resolution on the visible part of the layer (plus the filters' reach), and is then
//! averaged down to the zoom (#2502). On the desktop it renders off the UI thread with a progress
//! bar, so a 24 MP layer at Fit in View never freezes the window. OK runs
//! `filter.filterGallery {effects:[...]}`, one history step (a smart filter on smart objects).
//! Everything is drivable over the control channel with `filter.filterGallery {"ui": {...}}`
//! while the dialog is open.

use std::hash::{Hash, Hasher};

use egui::{Align2, Color32, FontId, Rect as ERect, Sense, Stroke, TextureHandle, pos2, vec2};
use photocraft_algo::{FilterParams, GALLERY_CATEGORIES, GalleryFilter};
use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use photocraft_raster::{Interrupt, Surface};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::edit_history::{EditHistory, Step};
use crate::filter_dialog::{Kind, parse_spec};
use crate::theme::Tokens;
use crate::widgets;

const MID_W: f32 = 300.0;
const RIGHT_W: f32 = 290.0;
const THUMB: [usize; 2] = [80, 56];

/// One effect layer of the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectLayer {
    pub filter: GalleryFilter,
    pub params: Map<String, Value>,
    pub visible: bool,
}

impl EffectLayer {
    fn new(filter: GalleryFilter) -> Self {
        EffectLayer { filter, params: default_params(filter), visible: true }
    }
    fn to_json(&self) -> Value {
        json!({"filter": self.filter.key(), "params": self.params, "visible": self.visible})
    }
}

/// The filter's defaults in command-param form (choices by name).
fn default_params(f: GalleryFilter) -> Map<String, Value> {
    let mut m = Map::new();
    for p in parse_spec(f.params_doc()) {
        let v = match p.kind {
            Kind::Range { default, .. } => json!(default),
            Kind::Choice { options, default } => {
                let initial = default.as_deref().or_else(|| options.first().map(String::as_str)).unwrap_or("");
                json!(initial)
            }
            Kind::Bool(b) => json!(b),
            _ => continue,
        };
        m.insert(p.key, v);
    }
    m
}

pub struct GalleryDialog {
    pub layer: LayerId,
    layer_name: String,
    canvas: Rect,
    src: Surface,
    /// Apply order: index 0 runs first (shown at the bottom of the list).
    pub effects: Vec<EffectLayer>,
    pub selected: usize,
    open: [bool; 6],
    foreground: [f32; 4],
    background: [f32; 4],
    /// Screen px per document px (0 = fit) and the view centre (document px).
    pub zoom: f32,
    center: [f32; 2],
    tex: Option<TextureHandle>,
    /// The key of the preview in `tex` and the document rect it covers.
    shown: Option<(u64, Rect)>,
    /// The preview rendering off the UI thread (desktop), and the latest wanted key with when it
    /// was first wanted (ms), to cancel a stale render once the settings settle.
    #[cfg(not(target_arch = "wasm32"))]
    job: Option<PreviewJob>,
    latest: Option<(u64, f64)>,
    /// A key whose render failed: not retried until the settings or view change.
    failed: Option<u64>,
    thumb_src: Surface,
    thumbs: Vec<Option<TextureHandle>>,
    pub preview_ms: f64,
    history: EditHistory<Vec<EffectLayer>>,
}

impl GalleryDialog {
    pub fn describe(&self) -> Value {
        json!({
            "layer": self.layer.0,
            "effects": self.effects.iter().map(EffectLayer::to_json).collect::<Vec<_>>(),
            "selected": self.selected,
            "zoom": if self.zoom > 0.0 { json!(self.zoom) } else { json!("fit") },
            "previewMs": self.preview_ms,
            "rendering": self.rendering().is_some(),
        })
    }

    /// The command params for the current stack.
    pub fn params(&self) -> Value {
        json!({"effects": self.effects.iter().map(EffectLayer::to_json).collect::<Vec<_>>(), "foreground": self.foreground, "background": self.background})
    }

    /// The progress (`0.0..=1.0`) of a preview still rendering.
    fn rendering(&self) -> Option<f32> {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(j) = &self.job {
            return Some(j.ctl.fraction());
        }
        None
    }

    fn cur(&mut self) -> &mut EffectLayer {
        let i = self.selected.min(self.effects.len() - 1);
        &mut self.effects[i]
    }

    fn set_filter(&mut self, f: GalleryFilter) {
        let e = self.cur();
        if e.filter != f {
            *e = EffectLayer { visible: e.visible, ..EffectLayer::new(f) };
        }
    }

    fn add(&mut self) {
        // Like Photoshop: the new effect layer starts as a copy of the selected one, above it.
        let e = self.cur().clone();
        self.selected = (self.selected + 1).min(self.effects.len());
        self.effects.insert(self.selected, e);
    }

    fn delete(&mut self) {
        if self.effects.len() > 1 {
            self.effects.remove(self.selected.min(self.effects.len() - 1));
            self.selected = self.selected.saturating_sub(1).min(self.effects.len() - 1);
        }
    }

    pub(crate) fn step(&mut self, step: Step) {
        if self.history.step(&mut self.effects, step) {
            self.selected = self.selected.min(self.effects.len().saturating_sub(1));
        }
    }

    fn move_to(&mut self, from: usize, to: usize) {
        if from < self.effects.len() && to < self.effects.len() && from != to {
            let e = self.effects.remove(from);
            self.effects.insert(to, e);
            self.selected = to;
        }
    }
}

/// A nearest-neighbour copy of `r` of `surf`, every `k`-th pixel, placed at the origin.
fn sample(surf: &Surface, r: Rect, k: u32) -> Surface {
    let k = k.max(1) as i32;
    let (w, h) = ((r.width() as i32 + k - 1) / k, (r.height() as i32 + k - 1) / k);
    let mut out = Surface::new(surf.format());
    if w <= 0 || h <= 0 {
        return out;
    }
    let n = surf.channels();
    let mut data = Vec::with_capacity((w * h) as usize * n);
    for j in 0..h {
        let y = r.y0 + j * k;
        let row = surf.read_region(Rect::new(r.x0, y, r.x1, y + 1));
        for i in 0..w {
            let o = (i * k) as usize * n;
            data.extend_from_slice(&row[o..o + n]);
        }
    }
    out.write_region(Rect::new(0, 0, w, h), &data);
    out
}

/// Runs a stack on a small surface (the preview and thumbnails).
fn render(src: &Surface, params: &Value) -> Surface {
    let effects = photocraft_engine::gallery_cmds::effects_from_json(params).unwrap_or_default();
    let b = src.content_bounds().union(&Rect::new(0, 0, 1, 1));
    let b = Rect::new(0, 0, b.x1, b.y1);
    photocraft_algo::apply_in(src, &FilterParams::FilterGallery { effects }, b, b, None, b)
}

/// The preview of `vis` at 1/`k` (#2502): the stack runs at full resolution exactly as OK runs it
/// (same bounds, same canvas-edge handling, textures and noise seeded by document position), so
/// the pixels equal the same crop of the applied result; then `k`×`k` blocks are averaged.
/// `None` when cancelled.
fn preview_pixels(src: &Surface, canvas: Rect, params: &Value, vis: Rect, k: u32, ctl: &Interrupt) -> Option<Surface> {
    let effects = photocraft_engine::gallery_cmds::effects_from_json(params).unwrap_or_default();
    let fp = FilterParams::FilterGallery { effects };
    let content = src.content_bounds();
    let area = photocraft_algo::output_area(&fp, content, canvas, None).intersect(&vis);
    let full = if area.is_empty() { src.clone() } else { photocraft_algo::apply_in_with(src, &fp, area, canvas, None, canvas.union(&content), ctl)? };
    Some(reduce(&full, vis, k))
}

/// `r` of `surf` scaled down by `k`, each pixel the average of a `k`×`k` block (colour weighted by
/// alpha), placed at the origin.
fn reduce(surf: &Surface, r: Rect, k: u32) -> Surface {
    let k = k.clamp(1, 1 << 16) as usize;
    let (bw, bh) = (r.width() as usize, r.height() as usize);
    let (w, h) = (bw.div_ceil(k), bh.div_ceil(k));
    let mut out = Surface::new(surf.format());
    if w == 0 || h == 0 {
        return out;
    }
    let n = surf.channels().max(1);
    let alpha = surf.format().alpha;
    let colours = if alpha { n - 1 } else { n };
    // One output row from a band of `k` source rows.
    let row = |j: usize, out: &mut [f32]| {
        let y0 = r.y0.saturating_add((j * k) as i32);
        let band = surf.read_region(Rect::new(r.x0, y0, r.x1, y0.saturating_add(k as i32).min(r.y1)));
        let mut acc = vec![0.0f64; w * n];
        let mut count = vec![0u32; w];
        for (i, px) in band.chunks_exact(n).enumerate() {
            let x = (i % bw) / k;
            let (Some(sum), Some(c)) = (acc.get_mut(x * n..x * n + n), count.get_mut(x)) else { continue };
            let a = if alpha { px[n - 1] as f64 } else { 1.0 };
            for (s, v) in sum.iter_mut().zip(&px[..colours]) {
                *s += *v as f64 * a;
            }
            if alpha {
                sum[n - 1] += a;
            }
            *c += 1;
        }
        for ((sum, c), o) in acc.chunks_exact(n).zip(&count).zip(out.chunks_exact_mut(n)) {
            let a = if alpha { sum[n - 1] } else { *c as f64 };
            for (o, v) in o.iter_mut().zip(&sum[..colours]) {
                *o = if a > 0.0 { (v / a) as f32 } else { 0.0 };
            }
            if alpha {
                o[n - 1] = (a / (*c).max(1) as f64) as f32;
            }
        }
    };
    let mut data = vec![0.0f32; w * h * n];
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(w * n).enumerate().for_each(|(j, out)| row(j, out));
    }
    #[cfg(target_arch = "wasm32")]
    for (j, out) in data.chunks_mut(w * n).enumerate() {
        row(j, out);
    }
    out.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
    out
}

fn image(surf: &Surface, w: usize, h: usize) -> egui::ColorImage {
    let mut px = vec![[0u8; 4]; w * h];
    for y in 0..h {
        surf.read_rgba8_into(Rect::new(0, y as i32, w as i32, y as i32 + 1), &mut px[y * w..(y + 1) * w]);
    }
    let alpha = surf.format().alpha;
    egui::ColorImage::new([w, h], px.iter().map(|p| Color32::from_rgba_unmultiplied(p[0], p[1], p[2], if alpha { p[3] } else { 255 })).collect())
}

/// Opens the dialog on the active layer, starting from the last gallery stack used.
pub fn open(app: &mut PhotocraftApp) -> Result<(), String> {
    let (layer, src, _) = crate::distort_ui::active_pixels(app)?;
    let st = app.session.active().ok_or("no document")?;
    let canvas = st.doc.bounds();
    let layer_name = st.doc.layer(layer).map(|l| l.name.clone()).unwrap_or_default();
    let last = app.session.journal.iter().rev().find(|(id, _)| id == "filter.filterGallery").map(|(_, p)| p.clone());
    let mut effects: Vec<EffectLayer> = last
        .as_ref()
        .and_then(|p| p.get("effects").and_then(Value::as_array).cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|e| {
            let f = GalleryFilter::from_key(e.get("filter")?.as_str()?)?;
            let mut l = EffectLayer::new(f);
            if let Some(Value::Object(m)) = e.get("params") {
                l.params.extend(m.clone());
            }
            l.visible = e.get("visible").and_then(Value::as_bool).unwrap_or(true);
            Some(l)
        })
        .collect();
    if effects.is_empty() {
        effects.push(EffectLayer::new(GalleryFilter::ColoredPencil));
    }
    // Thumbnail source: the middle of the layer, about 4× the thumbnail size.
    let cb = src.content_bounds().intersect(&canvas);
    let cb = if cb.is_empty() { canvas } else { cb };
    let k = ((cb.width() as usize / (THUMB[0] * 4)).min(cb.height() as usize / (THUMB[1] * 4))).max(1) as u32;
    let (tw, th) = (THUMB[0] as i32 * k as i32, THUMB[1] as i32 * k as i32);
    let (cx, cy) = ((cb.x0 + cb.x1) / 2, (cb.y0 + cb.y1) / 2);
    let thumb_src = sample(&src, Rect::new(cx - tw / 2, cy - th / 2, cx - tw / 2 + tw, cy - th / 2 + th), k);
    let sel = effects.len() - 1;
    let open_cats = std::array::from_fn(|i| effects[sel].filter.category() == GALLERY_CATEGORIES[i]);
    let history = EditHistory::new(&effects);
    app.distort.gallery = Some(GalleryDialog {
        layer,
        layer_name,
        canvas,
        src,
        effects,
        selected: sel,
        open: open_cats,
        foreground: app.session.tools.foreground,
        background: app.session.tools.background,
        zoom: 0.0,
        center: [(canvas.x0 + canvas.x1) as f32 / 2.0, (canvas.y0 + canvas.y1) as f32 / 2.0],
        tex: None,
        shown: None,
        #[cfg(not(target_arch = "wasm32"))]
        job: None,
        latest: None,
        failed: None,
        thumb_src,
        thumbs: vec![None; GalleryFilter::ALL.len()],
        preview_ms: 0.0,
        history,
    });
    Ok(())
}

/// OK: runs `filter.filterGallery` with the stack (one history step).
pub fn commit(app: &mut PhotocraftApp) -> Result<Value, String> {
    let Some(d) = app.distort.gallery.take() else { return Err("the Filter Gallery is not open".into()) };
    if !d.effects.iter().any(|e| e.visible) {
        return Ok(json!({"committed": false}));
    }
    let mut p = d.params();
    p["layer"] = json!(d.layer.0);
    app.run("filter.filterGallery", p)
}

/// Control channel: `filter.filterGallery {"ui": {...}}` while the dialog is open.
pub fn control(app: &mut PhotocraftApp, ui: &Value) -> Result<Value, String> {
    let flag = |k: &str| ui.get(k).and_then(Value::as_bool) == Some(true);
    if flag("commit") {
        let r = commit(app)?;
        return Ok(json!({"committed": true, "result": r}));
    }
    if flag("cancel") {
        app.distort.gallery = None;
        return Ok(json!({"cancelled": true}));
    }
    let d = app.distort.gallery.as_mut().ok_or("the Filter Gallery is not open")?;
    if let Some(i) = ui.get("select").and_then(Value::as_u64) {
        d.selected = (i as usize).min(d.effects.len() - 1);
    }
    if flag("add") {
        d.add();
    }
    if flag("delete") {
        d.delete();
    }
    if let Some(i) = ui.get("toggle").and_then(Value::as_u64)
        && let Some(e) = d.effects.get_mut(i as usize)
    {
        e.visible = !e.visible;
    }
    if let Some(m) = ui.get("move") {
        let g = |k: &str| m.get(k).and_then(Value::as_u64).map(|v| v as usize);
        if let (Some(a), Some(b)) = (g("from"), g("to")) {
            d.move_to(a, b);
        }
    }
    if let Some(k) = ui.get("filter").and_then(Value::as_str) {
        let f = GalleryFilter::from_key(k).ok_or_else(|| format!("unknown gallery filter `{k}`"))?;
        d.set_filter(f);
    }
    if let Some(Value::Object(m)) = ui.get("params") {
        d.cur().params.extend(m.clone());
    }
    d.history.settle(&d.effects, false);
    match ui.get("zoom") {
        Some(Value::String(s)) if s == "fit" => d.zoom = 0.0,
        // The gallery's own preview zoom, not the canvas (`zoom_levels` doesn't apply): its − / +
        // buttons and menu stop at 400 %, an agent may set up to 3200 %.
        Some(v) if v.is_number() => d.zoom = (v.as_f64().unwrap_or(1.0) as f32).clamp(0.01, 32.0),
        _ => {}
    }
    Ok(d.describe())
}

/// A preview rendering on a worker thread: what it shows, its result and its cancellation.
#[cfg(not(target_arch = "wasm32"))]
struct PreviewJob {
    key: u64,
    vis: Rect,
    rx: std::sync::mpsc::Receiver<Result<Option<(egui::ColorImage, f64)>, String>>,
    ctl: photocraft_engine::jobs::JobCtx,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for PreviewJob {
    /// A closed dialog or a superseded render stops at its next tile.
    fn drop(&mut self) {
        self.ctl.cancel();
    }
}

/// The preview's zoom (screen px per document px), the visible document rect and the factor the
/// texture is reduced by. Fit in View also recentres the view.
fn view(d: &mut GalleryDialog, area: ERect) -> (f32, Rect, u32) {
    let (cw, ch) = (d.canvas.width().max(1) as f32, d.canvas.height().max(1) as f32);
    let zoom = if d.zoom > 0.0 { d.zoom } else { ((area.width() - 24.0) / cw).min((area.height() - 24.0) / ch).clamp(0.005, 1.0) };
    if d.zoom <= 0.0 {
        d.center = [d.canvas.x0 as f32 + cw / 2.0, d.canvas.y0 as f32 + ch / 2.0];
    }
    let (hw, hh) = (area.width() / zoom / 2.0, area.height() / zoom / 2.0);
    let vis =
        Rect::new((d.center[0] - hw).floor() as i32, (d.center[1] - hh).floor() as i32, (d.center[0] + hw).ceil() as i32, (d.center[1] + hh).ceil() as i32)
            .intersect(&d.canvas);
    // At least one texture pixel per screen pixel; the GPU's linear filter does the rest.
    let k = (1.0 / zoom + 1e-3).floor().clamp(1.0, 4096.0) as u32;
    (zoom, vis, k)
}

/// Keeps the preview texture current with the stack and the view; returns where it is drawn.
/// With `background` the render runs off the UI thread and the last preview stays up (moving
/// with the view) until it is done; otherwise it renders inline (web, tests, inline sessions).
fn update_preview(d: &mut GalleryDialog, ctx: &egui::Context, area: ERect, background: bool) -> Option<ERect> {
    let (zoom, vis, k) = view(d, area);
    let params = d.params();
    let mut h = std::collections::hash_map::DefaultHasher::new();
    params.to_string().hash(&mut h);
    (vis.x0, vis.y0, vis.x1, vis.y1, k).hash(&mut h);
    let key = h.finish();
    let now = crate::gpu_canvas::now_ms();
    if d.latest.is_none_or(|(k0, _)| k0 != key) {
        d.latest = Some((key, now));
    }
    #[cfg(not(target_arch = "wasm32"))]
    poll_job(d, ctx);
    let wanted = d.shown.is_none_or(|(k0, _)| k0 != key) && !vis.is_empty() && d.failed != Some(key);
    #[cfg(not(target_arch = "wasm32"))]
    if wanted && background {
        start_job(d, ctx, key, vis, k, &params, now);
    } else if d.job.as_ref().is_some_and(|j| j.key != key) {
        // Back to what is already shown: the render for other settings is no longer needed.
        d.job = None;
    }
    if wanted && !background {
        let t0 = crate::gpu_canvas::now_ms();
        if let Some(out) = preview_pixels(&d.src, d.canvas, &params, vis, k, &Interrupt::NONE) {
            let img = image(&out, reduced(vis.width(), k), reduced(vis.height(), k));
            show_image(d, ctx, img, key, vis, crate::gpu_canvas::now_ms() - t0);
        }
    }
    let (_, r) = d.shown?;
    let at = |x: i32, y: i32| area.center() + vec2((x as f32 - d.center[0]) * zoom, (y as f32 - d.center[1]) * zoom);
    Some(ERect::from_min_max(at(r.x0, r.y0), at(r.x1, r.y1)))
}

/// A length of `n` document pixels reduced by `k` (partial blocks count).
fn reduced(n: u32, k: u32) -> usize {
    (n as usize).div_ceil(k.max(1) as usize)
}

fn show_image(d: &mut GalleryDialog, ctx: &egui::Context, img: egui::ColorImage, key: u64, vis: Rect, ms: f64) {
    match &mut d.tex {
        Some(t) => t.set(img, egui::TextureOptions::LINEAR),
        None => d.tex = Some(ctx.load_texture("gallery-preview", img, egui::TextureOptions::LINEAR)),
    }
    d.preview_ms = ms;
    d.shown = Some((key, vis));
}

/// Takes a finished render: shown when it completed, dropped when cancelled, remembered when it
/// failed (so a failing stack isn't retried every frame).
#[cfg(not(target_arch = "wasm32"))]
fn poll_job(d: &mut GalleryDialog, ctx: &egui::Context) {
    let Some(job) = &d.job else { return };
    let result = match job.rx.try_recv() {
        Ok(r) => r,
        Err(std::sync::mpsc::TryRecvError::Empty) => return,
        Err(std::sync::mpsc::TryRecvError::Disconnected) => Err("the Filter Gallery preview stopped".into()),
    };
    let Some(job) = d.job.take() else { return };
    match result {
        Ok(Some((img, ms))) => show_image(d, ctx, img, job.key, job.vis, ms),
        Ok(None) => {}
        Err(_) => d.failed = Some(job.key),
    }
}

/// One render at a time: a running render for older settings finishes (so a slider drag keeps
/// updating) unless the settings have been unchanged for a moment, then it is cancelled.
#[cfg(not(target_arch = "wasm32"))]
fn start_job(d: &mut GalleryDialog, ctx: &egui::Context, key: u64, vis: Rect, k: u32, params: &Value, now: f64) {
    if let Some(job) = &d.job {
        let settled = d.latest.is_some_and(|(k0, t)| k0 == key && now - t >= crate::filter_preview_worker::SETTLE_MS);
        if job.key != key && settled {
            job.ctl.cancel();
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(30));
        return;
    }
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let ctl = photocraft_engine::jobs::JobCtx::new();
    let (work, src, canvas, params, repaint) = (ctl.clone(), d.src.clone(), d.canvas, params.clone(), ctx.clone());
    let spawned = std::thread::Builder::new().name("gallery-preview".into()).spawn(move || {
        let run = || {
            let t0 = crate::gpu_canvas::now_ms();
            let out = work.stage(0.0, 1.0, "", |ctl| preview_pixels(&src, canvas, &params, vis, k, ctl))?;
            let img = image(&out, reduced(vis.width(), k), reduced(vis.height(), k));
            Some((img, crate::gpu_canvas::now_ms() - t0))
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)).map_err(|_| "the Filter Gallery preview failed".to_string());
        let _ = tx.send(result);
        repaint.request_repaint();
    });
    match spawned {
        Ok(_) => {
            d.job = Some(PreviewJob { key, vis, rx, ctl });
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
        Err(_) => d.failed = Some(key),
    }
}

/// Renders up to `budget` missing thumbnails (spread over frames so opening stays instant).
fn update_thumbs(d: &mut GalleryDialog, ctx: &egui::Context, budget: usize) {
    let mut done = 0;
    for (i, f) in GalleryFilter::ALL.iter().enumerate() {
        if d.thumbs[i].is_some() || !d.open[GALLERY_CATEGORIES.iter().position(|c| *c == f.category()).unwrap_or(0)] {
            continue;
        }
        if done == budget {
            ctx.request_repaint();
            return;
        }
        let p = json!({"effects": [{"filter": f.key()}], "foreground": d.foreground, "background": d.background});
        let out = render(&d.thumb_src, &p);
        let b = d.thumb_src.content_bounds();
        let img = image(&out, b.x1.max(1) as usize, b.y1.max(1) as usize);
        d.thumbs[i] = Some(ctx.load_texture(format!("gallery-thumb-{}", f.key()), img, egui::TextureOptions::LINEAR));
        done += 1;
    }
}

/// Draws the dialog (a full-window layer over the app).
pub fn show(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let t = Tokens::get(ctx);
    let screen = ctx.content_rect();
    let mut action: Option<&str> = None;
    let background = app.background_jobs;
    egui::Area::new(egui::Id::new("gallery-dialog")).order(egui::Order::Foreground).fixed_pos(screen.min).show(ctx, |ui| {
        let Some(d) = app.distort.gallery.as_mut() else { return };
        d.history.settle(&d.effects, ctx.input(|i| i.pointer.any_down()));
        let (full, _) = ui.allocate_exact_size(screen.size(), Sense::hover());
        let painter = ui.painter().clone();
        painter.rect_filled(full, 0.0, t.chrome);
        let title = ERect::from_min_size(full.min, vec2(full.width(), 30.0));
        painter.rect_filled(title, 0.0, t.dock);
        painter.line_segment([title.left_bottom(), title.right_bottom()], Stroke::new(1.0, t.separator));
        let pct = if d.zoom > 0.0 { format!("{:.0}%", d.zoom * 100.0) } else { tl!("Fit").into() };
        let name = d.effects.get(d.selected).map_or("", |e| tl!(e.filter.name()));
        painter.text(title.center(), Align2::CENTER_CENTER, format!("{name} ({}, {pct})", d.layer_name), FontId::proportional(13.0), t.text);
        let body = ERect::from_min_max(pos2(full.left(), title.bottom()), full.max);
        let right = ERect::from_min_size(pos2(body.right() - RIGHT_W, body.top()), vec2(RIGHT_W, body.height()));
        let mid = ERect::from_min_size(pos2(right.left() - MID_W, body.top()), vec2(MID_W, body.height()));
        let area = ERect::from_min_max(body.min, pos2(mid.left(), body.bottom() - 30.0));
        let zoom_bar = ERect::from_min_max(pos2(body.left(), area.bottom()), pos2(mid.left(), body.bottom()));

        // ---- Preview ----
        painter.rect_filled(area, 0.0, t.canvas);
        let shown = update_preview(d, ctx, area, background);
        let clip = painter.with_clip_rect(area);
        if let (Some(tex), Some(r)) = (&d.tex, shown) {
            widgets::checker(&clip, r.intersect(area), 8.0);
            clip.image(tex.id(), r, ERect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            clip.rect_stroke(r, 0.0, Stroke::new(1.0, t.separator), egui::StrokeKind::Outside);
        }
        let resp = ui.interact(area, egui::Id::new("gallery-preview"), Sense::drag());
        if resp.dragged() && d.zoom > 0.0 {
            let dl = resp.drag_delta() / d.zoom;
            d.center = [d.center[0] - dl.x, d.center[1] - dl.y];
        }
        // ---- Zoom bar ----
        painter.rect_filled(zoom_bar, 0.0, t.dock);
        let mut zb = ui.new_child(egui::UiBuilder::new().max_rect(zoom_bar.shrink2(vec2(10.0, 3.0))).layout(egui::Layout::left_to_right(egui::Align::Center)));
        let steps = [0.0f32, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0];
        if crate::icons::button(&mut zb, "minus", 22.0, false, tl!("Zoom out")).clicked() {
            d.zoom = steps.iter().rev().copied().find(|&s| s > 0.0 && s < d.zoom.max(0.0) - 1e-3).unwrap_or(0.0);
        }
        if crate::icons::button(&mut zb, "plus", 22.0, false, tl!("Zoom in")).clicked() {
            d.zoom = steps.iter().copied().find(|&s| s > d.zoom + 1e-3).unwrap_or(4.0);
        }
        let mut z = d.zoom;
        let opts: Vec<(f32, &str)> =
            vec![(0.0, tl!("Fit in View")), (0.125, "12.5%"), (0.25, "25%"), (0.5, "50%"), (1.0, "100%"), (2.0, "200%"), (4.0, "400%")];
        if widgets::dropdown(&mut zb, "gallery-zoom", &mut z, &opts, 110.0) {
            d.zoom = z;
        }
        if let Some(p) = d.rendering() {
            // Like Photoshop's: a progress bar while the preview renders (the last one stays up).
            let (br, _) = zb.allocate_exact_size(vec2(90.0, 6.0), Sense::hover());
            crate::jobs_ui::bar(&zb, br, (p > 0.0).then_some(p), &t);
        } else {
            zb.label(egui::RichText::new(format!("{} {:.0} ms", tl!("Preview"), d.preview_ms)).color(t.text_faint).size(11.0));
        }

        // ---- Category folders with thumbnails ----
        painter.rect_filled(mid, 0.0, t.dock);
        painter.line_segment([mid.left_top(), mid.left_bottom()], Stroke::new(1.0, t.separator));
        update_thumbs(d, ctx, 6);
        let mut mu = ui.new_child(egui::UiBuilder::new().max_rect(mid.shrink2(vec2(8.0, 8.0))));
        egui::ScrollArea::vertical().id_salt("gallery-tree").show(&mut mu, |ui| {
            for (ci, cat) in GALLERY_CATEGORIES.iter().enumerate() {
                let icon = if d.open[ci] { "folder-open" } else { "folder" };
                let r = ui.horizontal(|ui| {
                    let (ir, _) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::hover());
                    crate::icons::paint(ui, ir, icon, 14.0, t.icon);
                    ui.add(egui::Label::new(egui::RichText::new(tl!(*cat)).color(t.text)).selectable(false).sense(Sense::click()))
                        .on_hover_cursor(egui::CursorIcon::PointingHand)
                });
                if r.inner.clicked() {
                    d.open[ci] = !d.open[ci];
                }
                if !d.open[ci] {
                    continue;
                }
                let members: Vec<(usize, GalleryFilter)> = GalleryFilter::ALL.iter().copied().enumerate().filter(|(_, f)| f.category() == *cat).collect();
                egui::Grid::new(format!("gallery-cat-{ci}")).spacing([6.0, 6.0]).show(ui, |ui| {
                    for (n, (i, f)) in members.iter().enumerate() {
                        let selected = d.effects.get(d.selected).is_some_and(|e| e.filter == *f);
                        let cell = ui.vertical(|ui| {
                            let (r, resp) = ui.allocate_exact_size(vec2(THUMB[0] as f32, THUMB[1] as f32), Sense::click());
                            match &d.thumbs[*i] {
                                Some(tex) => {
                                    ui.painter().image(tex.id(), r, ERect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                                }
                                None => {
                                    ui.painter().rect_filled(r, 2.0, t.field);
                                }
                            }
                            let stroke = if selected { Stroke::new(2.0, t.accent) } else { Stroke::new(1.0, t.separator) };
                            ui.painter().rect_stroke(r, 0.0, stroke, egui::StrokeKind::Outside);
                            ui.add_sized(
                                [THUMB[0] as f32, 14.0],
                                egui::Label::new(egui::RichText::new(tl!(f.name())).size(10.5).color(if selected { t.text } else { t.text_dim })).truncate(),
                            );
                            resp
                        });
                        if cell.inner.clicked() {
                            d.set_filter(*f);
                        }
                        if n % 3 == 2 {
                            ui.end_row();
                        }
                    }
                });
                ui.add_space(4.0);
            }
        });

        // ---- Right: buttons, filter settings, effect layers ----
        painter.rect_filled(right, 0.0, t.dock);
        painter.line_segment([right.left_top(), right.left_bottom()], Stroke::new(1.0, t.separator));
        let inner = right.shrink2(vec2(12.0, 10.0));
        let mut ru = ui.new_child(egui::UiBuilder::new().max_rect(inner));
        ru.spacing_mut().item_spacing.y = 6.0;
        ru.horizontal(|ui| {
            if let Some(role) = widgets::dialog_buttons(
                ui,
                &[
                    widgets::DialogButton::new(widgets::ButtonRole::Default, tl!("OK"), 120.0),
                    widgets::DialogButton::new(widgets::ButtonRole::Cancel, tl!("Cancel"), 120.0),
                ],
            ) {
                action = Some(if role == widgets::ButtonRole::Default { "ok" } else { "cancel" });
            }
        });
        ru.add_space(4.0);
        let mut cur = d.effects[d.selected].filter;
        let names: Vec<(GalleryFilter, &str)> = GalleryFilter::ALL.iter().map(|f| (*f, f.name())).collect();
        if widgets::dropdown(&mut ru, "gallery-filter", &mut cur, &names, inner.width() - 8.0) {
            d.set_filter(cur);
        }
        widgets::hairline(&mut ru);
        let list_h = 190.0;
        let params_h = (inner.height() - 110.0 - list_h).max(80.0);
        egui::ScrollArea::vertical().id_salt("gallery-params").max_height(params_h).show(&mut ru, |ui| {
            let sel = d.selected;
            let e = &mut d.effects[sel];
            for p in parse_spec(e.filter.params_doc()) {
                let label = crate::filter_dialog::label(&p.key);
                match p.kind {
                    Kind::Range { min, max, default } => {
                        let mut v = e.params.get(&p.key).and_then(Value::as_f64).unwrap_or(default as f64) as f32;
                        widgets::slider_row(ui, &label, &mut v, min..=max, "", None);
                        e.params.insert(p.key, json!(v.round()));
                    }
                    Kind::Choice { options, default } => {
                        let initial = default.as_deref().or_else(|| options.first().map(String::as_str)).unwrap_or("");
                        let mut c = e.params.get(&p.key).and_then(Value::as_str).unwrap_or(initial).to_string();
                        let labels: Vec<String> = options.iter().map(|o| crate::filter_dialog::label(o)).collect();
                        let opts: Vec<(String, &str)> = options.iter().cloned().zip(labels.iter().map(String::as_str)).collect();
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim));
                            widgets::dropdown(ui, &format!("gallery-{}", p.key), &mut c, &opts, 140.0);
                        });
                        e.params.insert(p.key, json!(c));
                    }
                    Kind::Bool(b0) => {
                        let mut b = e.params.get(&p.key).and_then(Value::as_bool).unwrap_or(b0);
                        widgets::checkbox(ui, &mut b, &label);
                        e.params.insert(p.key, json!(b));
                    }
                    Kind::Json if p.key == "glowColor" => {
                        let c = e
                            .params
                            .get("glowColor")
                            .and_then(Value::as_array)
                            .map(|a| a.iter().filter_map(Value::as_f64).map(|v| v as f32).collect::<Vec<_>>())
                            .filter(|v| v.len() >= 3);
                        let neon = photocraft_algo::GalleryEffect::new(GalleryFilter::NeonGlow).color;
                        let mut rgb = c.map_or([neon[0], neon[1], neon[2]], |v| [v[0], v[1], v[2]]);
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(tl!("Glow Color")).color(t.text_dim));
                            crate::widgets::color_edit_button_rgb(ui, &mut rgb);
                        });
                        e.params.insert("glowColor".into(), json!([rgb[0], rgb[1], rgb[2], 1.0]));
                    }
                    _ => {}
                }
            }
            if e.filter.uses_colours() {
                ui.label(egui::RichText::new(tl!("Uses the foreground and background colours.")).color(t.text_faint).size(11.0));
            }
        });
        // Effect layers (top of the list = applied last).
        let list_top = inner.bottom() - list_h;
        let lr = ERect::from_min_max(pos2(inner.left(), list_top), inner.max);
        let mut lu = ui.new_child(egui::UiBuilder::new().max_rect(lr));
        widgets::hairline(&mut lu);
        egui::ScrollArea::vertical().id_salt("gallery-layers").max_height(list_h - 44.0).show(&mut lu, |ui| {
            let mut toggle = None;
            for i in (0..d.effects.len()).rev() {
                let sel = i == d.selected;
                let (row, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
                if sel {
                    ui.painter().rect_filled(row, 2.0, t.accent_soft);
                }
                let eye = ERect::from_min_size(row.min + vec2(4.0, 4.0), vec2(16.0, 16.0));
                crate::icons::paint(ui, eye, if d.effects[i].visible { "eye" } else { "eye-off" }, 14.0, t.icon);
                ui.painter().text(
                    row.left_center() + vec2(28.0, 0.0),
                    Align2::LEFT_CENTER,
                    tl!(d.effects[i].filter.name()),
                    FontId::proportional(12.5),
                    if d.effects[i].visible { t.text } else { t.text_faint },
                );
                if resp.clicked() {
                    if resp.interact_pointer_pos().is_some_and(|p| p.x < eye.right() + 4.0) {
                        toggle = Some(i);
                    } else {
                        d.selected = i;
                    }
                }
            }
            if let Some(i) = toggle {
                d.effects[i].visible = !d.effects[i].visible;
            }
        });
        lu.horizontal(|ui| {
            if crate::icons::button(ui, "chevron-up", 24.0, false, tl!("Move effect layer up")).clicked() {
                let s = d.selected;
                d.move_to(s, (s + 1).min(d.effects.len() - 1));
            }
            if crate::icons::button(ui, "chevron-down", 24.0, false, tl!("Move effect layer down")).clicked() {
                let s = d.selected;
                d.move_to(s, s.saturating_sub(1));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if crate::icons::button(ui, "trash", 24.0, false, tl!("Delete effect layer")).clicked() {
                    d.delete();
                }
                if crate::icons::button(ui, "file-plus", 24.0, false, tl!("New effect layer")).clicked() {
                    d.add();
                }
            });
        });
    });
    match action {
        Some("ok") => {
            let _ = commit(app);
        }
        Some("cancel") => app.distort.gallery = None,
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", json!({"width": 90, "height": 60, "depth": 16})).unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, active| {
                let s = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
                for y in 0..60 {
                    for x in 0..90 {
                        let v = ((x / 6 + y / 6) % 2) as f32;
                        s.write_pixel(x, y, &[v, 0.4, 1.0 - v, 1.0]);
                    }
                }
                Ok(())
            })
            .unwrap();
        app.sync_views();
        app
    }

    #[test]
    fn stack_editing_and_commit_match_the_engine() {
        let ctx = egui::Context::default();
        let mut app = app();
        let r = crate::distort_ui::menu(&mut app, &ctx, "filter.filterGallery", &json!({})).unwrap().unwrap();
        assert_eq!(r["gallery"]["effects"].as_array().unwrap().len(), 1);
        control(&mut app, &json!({"filter": "cutout", "params": {"numberOfLevels": 3}})).unwrap();
        control(&mut app, &json!({"add": true})).unwrap();
        control(&mut app, &json!({"filter": "texturizer", "params": {"texture": "brick", "relief": 20}})).unwrap();
        control(&mut app, &json!({"add": true})).unwrap();
        control(&mut app, &json!({"filter": "glowingEdges"})).unwrap();
        control(&mut app, &json!({"toggle": 2})).unwrap();
        let d = control(&mut app, &json!({"move": {"from": 1, "to": 0}})).unwrap();
        let keys: Vec<&str> = d["effects"].as_array().unwrap().iter().map(|e| e["filter"].as_str().unwrap()).collect();
        assert_eq!(keys, ["texturizer", "cutout", "glowingEdges"]);
        assert_eq!(d["effects"][2]["visible"], json!(false));
        // Preview pixels = engine result on the same pixels.
        let params = app.distort.gallery.as_ref().unwrap().params();
        let before = app.session.active().unwrap().history.past_len();
        control(&mut app, &json!({"commit": true})).unwrap();
        assert!(app.distort.gallery.is_none());
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1);
        let (id, p) = app.session.journal.last().cloned().unwrap();
        assert_eq!(id, "filter.filterGallery");
        assert_eq!(p["effects"], params["effects"]);
        // Re-opening starts from the last stack.
        crate::distort_ui::menu(&mut app, &ctx, "filter.filterGallery", &json!({})).unwrap().unwrap();
        assert_eq!(app.distort.gallery.as_ref().unwrap().effects.len(), 3);
        control(&mut app, &json!({"delete": true})).unwrap();
        assert_eq!(app.distort.gallery.as_ref().unwrap().effects.len(), 2);
        control(&mut app, &json!({"cancel": true})).unwrap();
        assert_eq!(app.session.active().unwrap().history.past_len(), before + 1, "cancel records nothing");
    }

    #[test]
    fn preview_and_thumbnails_render() {
        let ctx = egui::Context::default();
        let mut app = app();
        open(&mut app).unwrap();
        let d = app.distort.gallery.as_mut().unwrap();
        update_preview(d, &ctx, ERect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 300.0)), false);
        assert!(d.tex.is_some() && d.shown.is_some());
        d.open = [true; 6];
        update_thumbs(d, &ctx, 100);
        assert!(d.thumbs.iter().all(Option::is_some));
        // Preview equals the engine's algorithm on the sampled pixels.
        let src = sample(&d.src, d.canvas, 1);
        let a = render(&src, &d.params());
        let mut s2 = photocraft_engine::Session::new();
        s2.add_document(app.session.active().unwrap().doc.as_ref().clone(), None);
        s2.select_layer(d.layer).unwrap();
        s2.execute("filter.filterGallery", d.params()).unwrap();
        let st = s2.active().unwrap();
        let b = st.doc.layer(d.layer).unwrap().surface().unwrap().read_region(d.canvas);
        let worst = a.read_region(d.canvas).iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "preview differs from the result by {worst}");
    }

    /// A 240×160 layer with gradients, edges and fine detail, so every gallery filter has
    /// something to draw.
    fn photo_app() -> PhotocraftApp {
        let (w, h) = (240, 160);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.session.execute("file.new", json!({"width": w, "height": h, "depth": 16})).unwrap();
        app.session.execute("layer.new.layer", json!({})).unwrap();
        app.session
            .edit("paint", |doc, active| {
                let s = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
                for y in 0..h {
                    for x in 0..w {
                        let (u, v) = (x as f32 / w as f32, y as f32 / h as f32);
                        let disc = (((x - 150) * (x - 150) + (y - 70) * (y - 70)) < 45 * 45) as i32 as f32;
                        let stripes = ((x + y) / 11 % 2) as f32;
                        s.write_pixel(x, y, &[u * 0.8 + disc * 0.2, 0.25 + 0.5 * stripes * (1.0 - disc), v * (1.0 - disc) + disc * 0.9, 1.0]);
                    }
                }
                Ok(())
            })
            .unwrap();
        app.sync_views();
        app
    }

    /// What OK applies: `filter.filterGallery` with the dialog's stack on a copy of the document.
    fn applied(app: &PhotocraftApp, d: &GalleryDialog) -> Surface {
        let mut s2 = photocraft_engine::Session::new();
        s2.add_document(app.session.active().unwrap().doc.as_ref().clone(), None);
        s2.select_layer(d.layer).unwrap();
        s2.execute("filter.filterGallery", d.params()).unwrap();
        s2.active().unwrap().doc.layer(d.layer).unwrap().surface().unwrap().clone()
    }

    /// Mean of `k`×`k` blocks (opaque layers), written independently of [`reduce`].
    fn box_mean(surf: &Surface, r: Rect, k: i32) -> Vec<f32> {
        let mut out = Vec::new();
        let mut px = [0.0f32; 4];
        for by in (r.y0..r.y1).step_by(k as usize) {
            for bx in (r.x0..r.x1).step_by(k as usize) {
                let mut sum = [0.0f32; 4];
                let mut n = 0.0;
                for y in by..(by + k).min(r.y1) {
                    for x in bx..(bx + k).min(r.x1) {
                        surf.read_pixel(x, y, &mut px);
                        sum.iter_mut().zip(px).for_each(|(s, v)| *s += v);
                        n += 1.0;
                    }
                }
                out.extend(sum.map(|s| s / n));
            }
        }
        out
    }

    /// Mean absolute RGB difference.
    fn mean_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());
        let (sum, n) = a.chunks(4).zip(b.chunks(4)).fold((0.0, 0.0), |(s, n), (p, q)| (s + (0..3).map(|c| (p[c] - q[c]).abs()).sum::<f32>(), n + 3.0));
        sum / n
    }

    #[test]
    fn zoomed_out_preview_matches_the_applied_result_scaled_down() {
        // #2502: below 100 % the preview filtered every k-th pixel with the same params, so
        // strokes, grain and textures showed several times too large.
        let ctx = egui::Context::default();
        let mut app = photo_app();
        open(&mut app).unwrap();
        let area = ERect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 300.0));
        for f in ["graphicPen", "charcoal", "reticulation", "stainedGlass", "spatter", "chalkCharcoal"] {
            control(&mut app, &json!({"filter": f, "zoom": 0.25})).unwrap();
            let reference = {
                let d = app.distort.gallery.as_ref().unwrap();
                box_mean(&applied(&app, d), d.canvas, 4)
            };
            let d = app.distort.gallery.as_mut().unwrap();
            let (_, vis, k) = view(d, area);
            assert_eq!((vis, k), (d.canvas, 4), "{f}: at 25 % the whole layer shows, averaged 4×4");
            assert!(update_preview(d, &ctx, area, false).is_some());
            let shown = preview_pixels(&d.src, d.canvas, &d.params(), vis, k, &Interrupt::NONE).unwrap();
            let (w, h) = (reduced(vis.width(), k) as i32, reduced(vis.height(), k) as i32);
            let diff = mean_diff(&shown.read_region(Rect::new(0, 0, w, h)), &reference);
            // The old preview (every 4th pixel, filtered at that size) for comparison.
            let old = mean_diff(&render(&sample(&d.src, vis, k), &d.params()).read_region(Rect::new(0, 0, w, h)), &reference);
            assert!(diff < 1e-3, "{f}: preview differs from the applied result by {diff}");
            assert!(old > 20.0 * diff.max(1e-3), "{f}: the old preview differed by only {old}");
        }
    }

    #[test]
    fn a_region_renders_the_same_pixels_as_the_full_result() {
        // Zoomed in, only the visible part renders: noise, cells and textures are seeded by
        // document position, and the filters read their reach around the region.
        let mut app = photo_app();
        open(&mut app).unwrap();
        let region = Rect::new(37, 21, 151, 117);
        for f in GalleryFilter::ALL {
            control(&mut app, &json!({"filter": f.key()})).unwrap();
            let d = app.distort.gallery.as_ref().unwrap();
            let full = applied(&app, d).read_region(region);
            let part = preview_pixels(&d.src, d.canvas, &d.params(), region, 1, &Interrupt::NONE).unwrap();
            let part = part.read_region(Rect::new(0, 0, region.width() as i32, region.height() as i32));
            let worst = full.iter().zip(&part).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
            assert!(worst < 1e-3, "{}: region differs from the full result by {worst}", f.key());
        }
    }

    #[test]
    fn reduce_averages_blocks_weighted_by_alpha() {
        let mut s = Surface::new(photocraft_color::PixelFormat::new(photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::F32, true));
        s.write_pixel(0, 0, &[1.0, 0.0, 0.0, 1.0]);
        s.write_pixel(1, 0, &[0.0, 1.0, 0.0, 0.0]); // transparent: its colour doesn't count
        s.write_pixel(2, 0, &[0.2, 0.4, 0.6, 1.0]); // a partial block at the edge
        let r = reduce(&s, Rect::new(0, 0, 3, 2), 2).read_region(Rect::new(0, 0, 2, 1));
        assert_eq!(r, [1.0, 0.0, 0.0, 0.25, 0.2, 0.4, 0.6, 0.5]);
        assert_eq!(reduce(&s, Rect::EMPTY, 2).content_bounds(), Rect::EMPTY);
    }

    #[test]
    fn background_preview_renders_off_the_ui_thread() {
        let ctx = egui::Context::default();
        let mut app = photo_app();
        open(&mut app).unwrap();
        control(&mut app, &json!({"filter": "graphicPen"})).unwrap();
        let d = app.distort.gallery.as_mut().unwrap();
        let area = ERect::from_min_size(pos2(0.0, 0.0), vec2(400.0, 300.0));
        assert!(update_preview(d, &ctx, area, true).is_none(), "nothing to show until the first render is done");
        assert!(d.describe()["rendering"].as_bool().unwrap());
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        while d.rendering().is_some() {
            assert!(std::time::Instant::now() < deadline, "the preview never finished");
            std::thread::sleep(std::time::Duration::from_millis(5));
            update_preview(d, &ctx, area, true);
        }
        assert!(d.tex.is_some() && update_preview(d, &ctx, area, true).is_some());
        // A new setting starts a new render; the last preview stays up meanwhile.
        let before = d.cur().params.get("strokeLength").cloned().unwrap();
        d.cur().params.insert("strokeLength".into(), json!(5));
        assert!(update_preview(d, &ctx, area, true).is_some());
        assert!(d.rendering().is_some());
        // Back to the settings on screen: that render is dropped (and cancelled).
        d.cur().params.insert("strokeLength".into(), before);
        assert!(update_preview(d, &ctx, area, true).is_some());
        assert!(d.rendering().is_none());
        d.cur().params.insert("strokeLength".into(), json!(5));
        update_preview(d, &ctx, area, true);
        // Closing the dialog cancels it.
        app.distort.gallery = None;
    }

    /// Preview cost on 24 MP: `cargo test -p photocraft-ui-egui --lib bench_gallery_preview -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_gallery_preview() {
        let (w, h) = (6000, 4000);
        let canvas = Rect::new(0, 0, w, h);
        let mut s = Surface::new(photocraft_color::PixelFormat::new(photocraft_color::ColorMode::Rgb, photocraft_color::SampleType::U8, true));
        s.fill_rect(canvas, &[0.5, 0.4, 0.3, 1.0]);
        for y in (0..h).step_by(9) {
            s.fill_rect(Rect::new(0, y, w, y + 3), &[0.9, 0.8, 0.2, 1.0]);
        }
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        std::hint::black_box(reduce(&s, canvas, 8));
        println!("{:>14}: {:>7.0} ms", "reduce 1/8", ms(t));
        for f in ["graphicPen", "charcoal", "reticulation", "stainedGlass", "spatter", "roughPastels"] {
            let p = json!({"effects": [{"filter": f}]});
            let t = std::time::Instant::now();
            std::hint::black_box(render(&sample(&s, canvas, 8), &p));
            let old = ms(t);
            let t = std::time::Instant::now();
            std::hint::black_box(preview_pixels(&s, canvas, &p, canvas, 8, &Interrupt::NONE));
            let fit = ms(t);
            let t = std::time::Instant::now();
            std::hint::black_box(preview_pixels(&s, canvas, &p, Rect::new(2400, 1600, 3500, 2400), 1, &Interrupt::NONE));
            let screen = ms(t);
            println!("{f:>14}: old fit {old:>5.0} ms, new fit {fit:>6.0} ms, new 100 % (1100x800) {screen:>5.0} ms");
        }
    }
}
