//! Histograms for the tone editors (Levels, Curves, Threshold) and the Histogram panel.
//!
//! The adjustment editors themselves live in `adjust_editors`; this module computes and caches
//! the histograms they draw behind their controls, in the channels of the adjustment's space
//! (RGB, CMYK ink brightness, Lab), and draws them.

use std::sync::Arc;

use egui::{Color32, Rect, Sense, vec2};
use photocraft_doc::adjust::ToneSpace;
use photocraft_doc::{Document, Layer, LayerId};

use crate::PhotocraftApp;
use crate::theme::Tokens;

const CHANNELS: [(&str, &str); 4] = [("rgb", "RGB"), ("red", "Red"), ("green", "Green"), ("blue", "Blue")];

static EMPTY: [u32; 256] = [0; 256];

/// 256-bin histograms: the composite then one per channel of the space (RGB: luminosity, R, G, B;
/// CMYK: composite, C, M, Y, K as channel brightness; Lab: L, a, b). `tag` identifies what they
/// were computed from (see [`HistSource`]).
#[derive(Clone, Debug, Default)]
pub struct Histograms {
    pub tag: u64,
    pub ch: Vec<[u32; 256]>,
}

impl std::ops::Index<usize> for Histograms {
    type Output = [u32; 256];
    fn index(&self, i: usize) -> &[u32; 256] {
        self.ch.get(i).unwrap_or(&EMPTY)
    }
}

/// What a tone editor's histogram shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HistSource {
    /// The image below an adjustment layer (what the adjustment receives).
    BelowLayer(LayerId),
    /// One layer on its own (the target of a destructive Image › Adjustments command).
    Layer(LayerId),
}

fn tag(source: HistSource, space: ToneSpace) -> u64 {
    let (k, id) = match source {
        HistSource::BelowLayer(l) => (1u64, l.0),
        HistSource::Layer(l) => (2, l.0),
    };
    let s = match space {
        ToneSpace::Rgb => 0u64,
        ToneSpace::Cmyk => 1,
        ToneSpace::Lab => 2,
    };
    id.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (k << 60) ^ (s << 56)
}

/// Shows only the branch of the layer tree holding `id`. Returns whether `id` is in `layers`.
fn isolate(layers: &mut [Layer], id: LayerId) -> bool {
    let mut found = false;
    for l in layers.iter_mut() {
        let here = l.id == id || l.children_mut().is_some_and(|c| isolate(c, id));
        l.visible = here;
        found |= here;
    }
    found
}

/// Histograms of `doc` as seen by `source`, in the channels of `space`.
pub fn compute_histograms(doc: &Document, source: HistSource, space: ToneSpace) -> Histograms {
    let mut d = doc.clone();
    match source {
        HistSource::BelowLayer(hide) => {
            if let Some(l) = d.layer_mut(hide) {
                l.visible = false;
            }
        }
        HistSource::Layer(id) => {
            isolate(&mut d.layers, id);
        }
    }
    let img = photocraft_compose::thumbnail(&d, 384);
    let n = match space {
        ToneSpace::Rgb => 4,
        ToneSpace::Cmyk => 5,
        ToneSpace::Lab => 3,
    };
    let mut h = vec![[0u32; 256]; n];
    let bin = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as usize;
    for p in img.pixels.as_chunks::<4>().0 {
        if p[3] == 0 {
            continue;
        }
        match space {
            ToneSpace::Rgb => {
                for c in 0..3 {
                    h[c + 1][p[c] as usize] += 1;
                }
                h[0][((p[0] as u32 + p[1] as u32 + p[2] as u32) / 3) as usize] += 1;
            }
            ToneSpace::Cmyk => {
                let rgb = [p[0], p[1], p[2]].map(|v| f32::from(v) / 255.0);
                let ink = photocraft_color::convert::rgb_to_cmyk(rgb);
                for (c, v) in ink.iter().enumerate() {
                    h[c + 1][bin(1.0 - v)] += 1;
                }
                h[0][bin(1.0 - ink.iter().sum::<f32>() / 4.0)] += 1;
            }
            ToneSpace::Lab => {
                let rgb = [p[0], p[1], p[2]].map(|v| f32::from(v) / 255.0);
                let l = photocraft_color::convert::srgb_to_lab(rgb);
                h[0][bin(l[0] / 100.0)] += 1;
                h[1][bin((l[1] + 128.0) / 255.0)] += 1;
                h[2][bin((l[2] + 128.0) / 255.0)] += 1;
            }
        }
    }
    Histograms { tag: tag(source, space), ch: h }
}

/// Cached histograms for a tone editor; recomputed only when something other than the editor's
/// own commits changed the document (see [`keep_after_commit`]).
pub fn histograms(app: &mut PhotocraftApp, source: HistSource, space: ToneSpace) -> Arc<Histograms> {
    let Some(st) = app.session.active() else { return Arc::new(Histograms::default()) };
    let (doc_id, rev) = (st.doc.id, st.revision);
    let want = tag(source, space);
    let layer = match source {
        HistSource::BelowLayer(l) | HistSource::Layer(l) => l,
    };
    let pixels_kept = view_only(st);
    if let Some((d, l, r, h)) = app.tone_hist.as_mut()
        && *d == doc_id
        && *l == layer
        && (*r == rev || (*r + 1 == rev && pixels_kept))
        && h.tag == want
    {
        *r = rev;
        return h.clone();
    }
    let t0 = crate::gpu_canvas::now_ms();
    let h = Arc::new(compute_histograms(&st.doc, source, space));
    app.perf.span("histogram", crate::gpu_canvas::now_ms() - t0);
    app.tone_hist = Some((doc_id, layer, rev, h.clone()));
    h
}

/// Whether the latest revision changed no pixels (selecting a layer, say): cached histograms of
/// the previous revision still hold (#125).
fn view_only(st: &photocraft_engine::DocState) -> bool {
    st.last_damage.is_some_and(|r| r.is_empty())
}

/// After an editor commits its own adjustment layer, the image below it is unchanged: keep the
/// cached histogram valid for the new revision. Call with the revision seen before the commit.
pub fn keep_after_commit(app: &mut PhotocraftApp, layer: LayerId, before: Option<u64>) {
    let now = app.session.active().map(|s| s.revision);
    if let (Some((_, l, r, _)), Some(before), Some(now)) = (app.tone_hist.as_mut(), before, now)
        && *l == layer
        && *r == before
    {
        *r = now;
    }
}

pub fn draw_histogram(p: &egui::Painter, r: Rect, h: &[u32; 256], color: Color32) {
    // Scale to a high percentile so a single spike (e.g. pure white) doesn't flatten the rest.
    let mut sorted: Vec<u32> = h.to_vec();
    sorted.sort_unstable();
    let top = sorted.get(250).copied().unwrap_or(1).max(1) as f32 * 1.1;
    let mut pts = vec![egui::pos2(r.left(), r.bottom())];
    for (i, v) in h.iter().enumerate() {
        let x = r.left() + (i as f32 + 0.5) / 256.0 * r.width();
        let y = r.bottom() - (*v as f32 / top).min(1.0) * r.height();
        pts.push(egui::pos2(x, y));
    }
    pts.push(egui::pos2(r.right(), r.bottom()));
    // One mesh of vertical strips: no anti-aliasing feather, so no seams between bins.
    let mut mesh = egui::Mesh::default();
    for w in pts.windows(2).skip(1).take(256) {
        let (a, b) = (w[0], w[1]);
        let i = mesh.vertices.len() as u32;
        for q in [egui::pos2(a.x, r.bottom()), a, b, egui::pos2(b.x, r.bottom())] {
            mesh.colored_vertex(q, color);
        }
        mesh.add_triangle(i, i + 1, i + 2);
        mesh.add_triangle(i, i + 2, i + 3);
    }
    p.add(mesh);
}

/// Display colour of channel `name` (curve, histogram and gradient bars).
pub fn channel_color(name: &str, t: &Tokens) -> Color32 {
    match name {
        "red" => Color32::from_rgb(235, 80, 80),
        "green" => Color32::from_rgb(80, 205, 95),
        "blue" => Color32::from_rgb(90, 140, 255),
        "cyan" => Color32::from_rgb(40, 200, 230),
        "magenta" => Color32::from_rgb(225, 70, 200),
        "yellow" => Color32::from_rgb(235, 210, 40),
        "a" => Color32::from_rgb(220, 90, 140),
        "b" => Color32::from_rgb(200, 180, 60),
        _ => t.text,
    }
}

fn channel_picker(ui: &mut egui::Ui, id: egui::Id, label: &str) -> usize {
    let mut ch: usize = ui.data(|d| d.get_temp(id)).unwrap_or(0);
    ui.horizontal(|ui| {
        let t = Tokens::get(ui.ctx());
        ui.label(egui::RichText::new(tl!(&label)).color(t.text_dim).size(12.0));
        let opts: Vec<(usize, &str)> = CHANNELS.iter().enumerate().map(|(i, (_, l))| (i, *l)).collect();
        crate::widgets::dropdown(ui, &format!("{id:?}-ch"), &mut ch, &opts, 110.0);
    });
    ui.data_mut(|d| d.insert_temp(id, ch));
    ch
}

// ---------------------------------------------------------------------------------------------
// Histogram panel

/// Histogram panel (Window › Histogram): whole-image histogram with Photoshop's statistics.
/// Recomputed at most every 250 ms while the document keeps changing (e.g. during painting).
pub fn histogram_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else {
        ui.label(egui::RichText::new(tl!("No document")).color(t.text_faint));
        return;
    };
    let (doc_id, rev) = (st.doc.id, st.revision);
    let now = crate::gpu_canvas::now_ms();
    if view_only(st)
        && let Some((d, r, _, _)) = app.doc_hist.as_mut()
        && *d == doc_id
        && *r + 1 == rev
    {
        *r = rev;
    }
    let stale = !matches!(&app.doc_hist, Some((d, r, _, _)) if *d == doc_id && *r == rev);
    let due = app.doc_hist.as_ref().is_none_or(|(d, _, at, _)| *d != doc_id || now - at > 250.0);
    if stale && due {
        // Computed off the UI thread (doc clone + proxy + pixel counts cost tens of ms on
        // large documents): the panel keeps showing the previous histogram with the existing
        // "cached data" warning until the worker delivers.
        let pending = app.hist_job.as_ref().is_some_and(|(d, r, _)| *d == doc_id && *r == rev);
        if !pending {
            let (tx, rx) = std::sync::mpsc::channel();
            // No threads on the web (and a spawn can fail natively): compute inline then, and
            // deliver through the same channel so one receive path lands both.
            #[cfg(not(target_arch = "wasm32"))]
            let spawned = {
                let doc = st.doc.clone();
                let tx = tx.clone();
                let ctx = ui.ctx().clone();
                std::thread::Builder::new()
                    .name("histogram".into())
                    .spawn(move || {
                        let t0 = crate::gpu_canvas::now_ms();
                        let h = Arc::new(compute_histograms(&doc, HistSource::BelowLayer(LayerId(u64::MAX)), ToneSpace::Rgb));
                        let _ = tx.send((doc.id, crate::gpu_canvas::now_ms() - t0, h));
                        ctx.request_repaint();
                    })
                    .is_ok()
            };
            #[cfg(target_arch = "wasm32")]
            let spawned = false;
            if !spawned {
                let t0 = crate::gpu_canvas::now_ms();
                let h = Arc::new(compute_histograms(&st.doc, HistSource::BelowLayer(LayerId(u64::MAX)), ToneSpace::Rgb));
                let _ = tx.send((doc_id, crate::gpu_canvas::now_ms() - t0, h));
            }
            app.hist_job = Some((doc_id, rev, rx));
        }
    } else if stale {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(260));
    }
    // A finished worker lands its histogram; the revision check keeps a stale result from
    // overwriting a newer one. A worker that died without sending clears the job so the next
    // frame schedules a new one.
    if let Some((jd, jr, rx)) = &app.hist_job {
        match rx.try_recv() {
            Ok((_, ms, h)) => {
                app.perf.span("histogram", ms);
                if *jd == doc_id {
                    app.doc_hist = Some((doc_id, *jr, crate::gpu_canvas::now_ms(), h));
                }
                app.hist_job = None;
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => app.hist_job = None,
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
        }
    }
    let Some((_, _, _, h)) = app.doc_hist.clone() else { return };
    let size = app.session.active().map_or(photocraft_doc::Size::new(0, 0), |s| s.doc.size);
    // Statistics come from a downsampled cache (like Photoshop's cache levels).
    let level = (size.width.max(size.height) as f32 / 384.0).max(1.0).log2().ceil() as u32 + 1;
    let ch = channel_picker(ui, egui::Id::new("histogram-panel-ch"), "Channel:");
    let w = ui.available_width();
    let (r, _) = ui.allocate_exact_size(vec2(w, 100.0), Sense::hover());
    let p = ui.painter_at(r);
    p.rect_filled(r, 0.0, t.field);
    let color = if ch == 0 { Color32::from_gray(if t.pro { 190 } else { 70 }) } else { channel_color(CHANNELS[ch.min(3)].0, &t) };
    draw_histogram(&p, r, &h[ch], color);
    if stale {
        // Photoshop's "cached data" warning triangle while the histogram lags the image.
        crate::icons::paint(ui, Rect::from_min_size(r.right_top() + vec2(-18.0, 2.0), vec2(16.0, 16.0)), "triangle-alert", 12.0, t.warning);
    }
    let hist = &h[ch];
    let n: u64 = hist.iter().map(|v| *v as u64).sum();
    if n == 0 {
        return;
    }
    let mean = hist.iter().enumerate().map(|(i, v)| i as f64 * *v as f64).sum::<f64>() / n as f64;
    let var = hist.iter().enumerate().map(|(i, v)| (i as f64 - mean).powi(2) * *v as f64).sum::<f64>() / n as f64;
    let mut acc = 0u64;
    let median = hist.iter().position(|v| {
        acc += *v as u64;
        acc * 2 >= n
    });
    ui.add_space(4.0);
    egui::Grid::new("hist-stats").num_columns(2).spacing(vec2(12.0, 2.0)).show(ui, |ui| {
        for (k, v) in [
            (tl!("Mean:"), format!("{mean:.2}")),
            (tl!("Std Dev:"), format!("{:.2}", var.sqrt())),
            (tl!("Median:"), median.unwrap_or(0).to_string()),
            (tl!("Pixels:"), (size.width as u64 * size.height as u64).to_string()),
            (tl!("Cache Level:"), level.to_string()),
        ] {
            ui.label(egui::RichText::new(k).color(t.text_dim).size(11.5));
            ui.label(egui::RichText::new(v).font(crate::theme::mono(11.5)).color(t.text));
            ui.end_row();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn histogram_ignores_the_edited_adjustment() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap(); // white
        s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
        let st = s.active().unwrap();
        let h = compute_histograms(&st.doc, HistSource::BelowLayer(st.active_layer.unwrap()), ToneSpace::Rgb);
        assert!(h[0][255] > 0 && h[0][0] == 0, "sees the white image, not the inverted result");
        assert_eq!(h[9], [0; 256], "out-of-range channels are empty, not a panic");
    }

    #[test]
    fn layer_histogram_sees_only_that_layer_and_spaces_have_their_channels() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 32, "height": 32})).unwrap(); // white background
        let bg = s.active().unwrap().doc.layers[0].id;
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 32, "height": 32})).unwrap();
        s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
        let st = s.active().unwrap();
        let h = compute_histograms(&st.doc, HistSource::Layer(bg), ToneSpace::Rgb);
        assert!(h[0][255] > 0 && h[0][0] == 0, "the black layer above is hidden");
        assert_eq!(compute_histograms(&st.doc, HistSource::Layer(bg), ToneSpace::Cmyk).ch.len(), 5);
        assert_eq!(compute_histograms(&st.doc, HistSource::Layer(bg), ToneSpace::Lab).ch.len(), 3);
    }
}

#[cfg(test)]
mod hist_async_tests {
    use super::*;
    use crate::PhotocraftApp;
    use serde_json::json;

    /// The panel schedules the histogram on a worker and keeps showing the previous data
    /// until it lands: right after the trigger the job exists (nothing computed in this
    /// frame), the receiver delivers, and a stale result cannot overwrite a newer revision.
    #[test]
    fn panel_histogram_is_computed_off_the_ui_thread() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        let _ = &ctx;
        app.run("file.new", json!({"width": 640, "height": 480})).unwrap();
        let (doc_id, rev) = {
            let st = app.session.active().unwrap();
            (st.doc.id, st.revision)
        };
        // Drive one panel frame through a minimal egui pass.
        let ctx = egui::Context::default();
        ctx.begin_pass(Default::default());
        let panel_ui = egui::Ui::new(
            ctx.clone(),
            egui::Id::new("hist-test"),
            egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 200.0))),
        );
        let mut panel_ui = panel_ui;
        histogram_panel(&mut app, &mut panel_ui);
        // A fast worker (or the inline fallback) may already have landed the result this frame.
        if let Some(job) = app.hist_job.take() {
            assert_eq!(job.0, doc_id);
            assert_eq!(job.1, rev);
            // The worker delivers (its repaint request is a no-op outside a frame).
            let (did, _, h) = job.2.recv().expect("the worker delivers the histogram");
            assert_eq!(did, doc_id);
            assert!(!h.ch.is_empty(), "the histogram has data");
            // Delivering it for the revision it was computed from lands in the cache.
            app.doc_hist = Some((doc_id, rev, crate::gpu_canvas::now_ms(), h.clone()));
        }
        assert_eq!(app.doc_hist.as_ref().map(|h| (h.0, h.1)), Some((doc_id, rev)));
        // …and a newer revision keeps the old entry stale instead of being overwritten by it.
        app.run("layer.new.layer", json!({})).unwrap();
        let rev2 = app.session.active().unwrap().revision;
        assert_ne!(rev, rev2);
        assert_ne!(app.doc_hist.as_ref().unwrap().1, rev2, "stale after an edit");
    }

    /// A worker that died without sending (its sender dropped) clears the job, so the next
    /// frame schedules a new one instead of waiting forever.
    #[test]
    fn a_dead_histogram_worker_clears_the_job() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 48})).unwrap();
        let (doc_id, rev) = {
            let st = app.session.active().unwrap();
            (st.doc.id, st.revision)
        };
        let (tx, rx) = std::sync::mpsc::channel();
        drop(tx);
        app.hist_job = Some((doc_id, rev, rx));
        let ctx = egui::Context::default();
        ctx.begin_pass(Default::default());
        let mut ui = egui::Ui::new(
            ctx.clone(),
            egui::Id::new("hist-dead"),
            egui::UiBuilder::new().max_rect(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(300.0, 200.0))),
        );
        histogram_panel(&mut app, &mut ui);
        assert!(app.hist_job.is_none(), "a disconnected job is dropped");
        assert!(app.doc_hist.is_none(), "nothing landed from the dead worker");
    }
}
