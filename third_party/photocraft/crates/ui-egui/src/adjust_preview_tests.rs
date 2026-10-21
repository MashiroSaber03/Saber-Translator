use super::*;
use photocraft_color::{BlendMode, Color, PixelFormat, SampleType};
use photocraft_doc::Size;
use serde_json::json;

const KINDS: [&str; 13] = crate::adjust_editors::KINDS;

/// Settings that move every kind well away from neutral.
pub(crate) fn sample(kind: &str) -> Value {
    match kind {
        "brightnessContrast" => json!({"brightness": 40, "contrast": 30}),
        "levels" => json!({"inBlack": 30, "inWhite": 220, "gamma": 1.3, "outBlack": 10}),
        "curves" => json!({"points": [[0, 20], [90, 150], [255, 230]]}),
        "exposure" => json!({"exposure": 0.8, "offset": 0.02, "gamma": 0.9}),
        "vibrance" => json!({"vibrance": 50, "saturation": 20}),
        "hueSaturation" => json!({"hue": 25, "saturation": 30, "reds": {"hue": 40}}),
        "colorBalance" => json!({"midtones": [40, 0, -20], "shadows": [0, 20, 0]}),
        "blackWhite" => json!({"reds": 150, "tint": true}),
        "photoFilter" => json!({"filter": "cooling80", "density": 60}),
        "channelMixer" => json!({"red": [50, 50, 0, 0], "blue": [0, 20, 90, 5]}),
        "posterize" => json!({"levels": 3}),
        "threshold" => json!({"level": 90}),
        "gradientMap" => json!({"stops": [[0, "#200040"], [1, "#ffd080"]]}),
        _ => json!({}),
    }
}

fn rnd(seed: u32, i: u32) -> f32 {
    let mut h = seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

fn fill_noise(s: &mut photocraft_raster::Surface, r: Rect, seed: u32, min_alpha: f32) {
    let fmt = s.format();
    let ch = fmt.channels();
    let n = r.width() * r.height();
    let mut data = Vec::with_capacity(n as usize * ch);
    for i in 0..n {
        for c in 0..ch {
            let v = rnd(seed + c as u32 * 7919, i);
            let alpha = fmt.alpha && c == ch - 1;
            data.push(if alpha { min_alpha + (1.0 - min_alpha) * v } else { v });
        }
    }
    s.write_region(r, &data);
}

/// Background, the target (semi-transparent noise, 80 % Multiply, layer mask), a layer clipped to
/// the target and a Screen layer on top: what the preview layer is inserted into.
pub(crate) fn document(mode: ColorMode, depth: SampleType, selection: bool) -> (Document, LayerId) {
    let (w, h) = (40u32, 32u32);
    let mut doc = Document::with_background("t", Size::new(w, h), mode, depth, Color::rgba(0.7, 0.6, 0.5, 1.0));
    let fmt = doc.pixel_format();
    if let Some(s) = doc.layers[0].surface_mut() {
        fill_noise(s, Rect::from_xywh(0, 0, w, h / 2), 3, 1.0);
    }
    let mut target = Layer::raster("target", fmt);
    if let Some(s) = target.surface_mut() {
        fill_noise(s, Rect::from_xywh(4, 2, 30, 26), 11, 0.2);
    }
    target.opacity = 0.8;
    target.blend = BlendMode::Multiply;
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::from_xywh(0, 0, 10, 10), &[0.3]);
    target.mask = Some(m);
    let id = target.id;
    let mut clip = Layer::raster("clipped", fmt);
    if let Some(s) = clip.surface_mut() {
        fill_noise(s, Rect::from_xywh(20, 10, 12, 12), 17, 0.5);
    }
    clip.clipped = true;
    clip.opacity = 0.6;
    let mut top = Layer::raster("top", fmt);
    if let Some(s) = top.surface_mut() {
        fill_noise(s, Rect::from_xywh(0, 20, 40, 12), 23, 0.0);
    }
    top.blend = BlendMode::Screen;
    doc.layers.extend([target, clip, top]);
    if selection {
        let mut sel = photocraft_raster::Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::from_xywh(8, 4, 24, 20), &[1.0]);
        sel.fill_rect(Rect::from_xywh(8, 4, 6, 20), &[0.5]);
        doc.selection = Some(sel);
    }
    (doc, id)
}

/// The document after running the real destructive command on `target`.
pub(crate) fn destructive(doc: &Document, target: LayerId, kind: &str, params: &Value) -> Option<Document> {
    let mut s = photocraft_engine::Session::new();
    s.add_document(doc.clone(), None);
    s.select_layer(target).ok()?;
    s.execute(&format!("image.adjustments.{kind}"), params.clone()).ok()?;
    Some((*s.active()?.doc).clone())
}

/// Largest premultiplied difference between two composites.
pub(crate) fn max_diff(a: &Document, b: &Document) -> f32 {
    let (ra, rb) = (photocraft_compose::render(a, a.bounds()), photocraft_compose::render(b, b.bounds()));
    let pm = |p: &[f32; 4]| [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]];
    ra.px.iter().zip(&rb.px).map(|(x, y)| (0..4).map(|c| (pm(x)[c] - pm(y)[c]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max)
}

/// Kinds that make colour from gray: grayscale documents preview them on the CPU proxy.
const GRAY_COLOUR: [&str; 6] = ["vibrance", "colorBalance", "blackWhite", "photoFilter", "channelMixer", "gradientMap"];

#[test]
fn preview_composite_matches_the_destructive_command() {
    let tol = 1.0 / 255.0 + 1e-4;
    let mut worst = (0.0f32, String::new());
    let mut fails = Vec::new();
    for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for selection in [false, true] {
                let (doc, target) = document(mode, depth, selection);
                for kind in KINDS {
                    for params in [sample(kind), json!({})] {
                        let what = format!("{kind} {mode:?} {depth:?} selection {selection} {params}");
                        let want = destructive(&doc, target, kind, &params);
                        let got = preview_document(&doc, target, kind, &params);
                        let gray_colour = mode == ColorMode::Grayscale && GRAY_COLOUR.contains(&kind);
                        let (Some(want), Ok(got)) = (want, got) else {
                            assert!(gray_colour, "{what}: command and preview disagree on validity");
                            continue;
                        };
                        assert!(!gray_colour, "{what}: should use the proxy preview");
                        let d = max_diff(&got, &want);
                        if d > worst.0 {
                            worst = (d, what.clone());
                        }
                        if d > tol {
                            fails.push(format!("{what}: preview differs from the command by {:.2}/255", d * 255.0));
                        }
                    }
                }
            }
        }
    }
    eprintln!("worst preview difference {:.3}/255 ({})", worst.0 * 255.0, worst.1);
    assert!(fails.is_empty(), "{}", fails.join("\n"));
}

#[test]
fn preview_leaves_the_document_alone_and_marks_its_layer() {
    let (doc, target) = document(ColorMode::Rgb, SampleType::U8, true);
    let p = preview_document(&doc, target, "curves", &sample("curves")).unwrap();
    assert_eq!(p.layer_count(), doc.layer_count() + 1);
    let l = p.layer(PREVIEW_LAYER).unwrap();
    assert!(l.clipped && l.mask.is_some());
    // Directly above the target, below the layer already clipped to it.
    let path = p.path_of(PREVIEW_LAYER).unwrap();
    assert_eq!(p.layer_at(&[path[0] - 1]).unwrap().id, target);
    assert!(doc.layer(PREVIEW_LAYER).is_none());
    // Region: the target's content within the selection.
    assert_eq!(region(&doc, target), Rect::from_xywh(8, 4, 24, 20));
}

#[test]
fn ineligible_targets_use_the_proxy_preview() {
    let (mut doc, target) = document(ColorMode::Rgb, SampleType::U8, false);
    let clipped = doc.layers[2].id;
    assert_eq!(unsupported(&doc, clipped), Some("the target is clipped"));
    assert_eq!(unsupported(&doc, LayerId(u64::MAX - 1)), Some("no target layer"));
    doc.layers.push(Layer::new("adj", LayerContent::Adjustment(photocraft_doc::Adjustment::Invert)));
    let adj = doc.layers[4].id;
    assert_eq!(unsupported(&doc, adj), Some("not a pixel layer"));
    assert!(unsupported(&doc, target).is_none());
    // Bad settings are an error, not a panic.
    assert!(preview_document(&doc, target, "levels", &json!({"gamma": "x"})).is_err());
    assert!(preview_document(&doc, target, "nope", &json!({})).is_err());
    assert!(preview_document(&doc, target, "curves", &json!([1, 2])).is_err());
    // CMYK and Lab adjust through RGB per pixel: proxy preview.
    for mode in [ColorMode::Cmyk, ColorMode::Lab] {
        let (d, t) = document(mode, SampleType::U8, false);
        assert!(unsupported(&d, t).is_some(), "{mode:?}");
    }
    let (gray, t) = document(ColorMode::Grayscale, SampleType::U16, false);
    assert!(preview_document(&gray, t, "hueSaturation", &json!({"colorize": true, "hue": 30, "saturation": 40})).is_err());
    assert!(preview_document(&gray, t, "hueSaturation", &json!({"lightness": 20})).is_ok());
}

/// The canvas recomposites only `region` between previews: nothing outside it may change.
#[test]
fn preview_changes_nothing_outside_its_region() {
    for selection in [false, true] {
        let (doc, target) = document(ColorMode::Rgb, SampleType::U8, selection);
        let r = region(&doc, target);
        let before = photocraft_compose::render(&doc, doc.bounds());
        for kind in KINDS {
            let p = preview_document(&doc, target, kind, &sample(kind)).unwrap();
            let after = photocraft_compose::render(&p, doc.bounds());
            for (i, (a, b)) in before.px.iter().zip(&after.px).enumerate() {
                let (x, y) = ((i % 40) as i32, (i / 40) as i32);
                if !r.contains(x, y) {
                    assert_eq!(a, b, "{kind} selection {selection}: ({x}, {y}) outside {r:?}");
                }
            }
        }
    }
}

fn app_with(doc: Document, target: LayerId) -> PhotocraftApp {
    let mut s = photocraft_engine::Session::new();
    s.add_document(doc, None);
    s.select_layer(target).unwrap();
    PhotocraftApp::new(s, crate::Services::default())
}

/// The dialog previews through the layer (no CPU proxy run), each change only swaps the
/// preview document; Cancel leaves the document untouched, OK is one history step.
#[test]
fn dialog_previews_through_the_layer() {
    use egui_kittest::Harness;
    for confirm in [false, true] {
        let (doc, target) = document(ColorMode::Rgb, SampleType::U16, true);
        let mut harness =
            Harness::builder().with_size(egui::vec2(1200.0, 900.0)).build_ui_state(|ui, app| crate::dialogs::show(app, ui.ctx()), app_with(doc, target));
        PhotocraftApp::setup_context(&harness.ctx, crate::theme::ThemeKind::ALL[0]);
        let id = crate::adjust_dialog::open(harness.state_mut(), "image.adjustments.levels").unwrap();
        harness.run_steps(2);
        let committed = harness.state().session.active().unwrap().doc.clone();
        let steps = harness.state().session.active().unwrap().history.past_len();
        let app = harness.state_mut();
        assert!(on_layer(app, 0));
        let (first, k1) = display_doc(app, 0).unwrap();
        assert!(first.layer(PREVIEW_LAYER).is_some());
        // Unchanged settings: the cached preview, same key.
        let (again, k1b) = display_doc(app, 0).unwrap();
        assert!(Arc::ptr_eq(&first, &again) && k1 == k1b);
        // A change: new key, and the canvas only recomposites the target's area.
        let d = app.ui.dialog_mut(id).unwrap();
        d.fields.insert("gamma".into(), json!(1.6));
        let (second, k2) = display_doc(app, 0).unwrap();
        assert_ne!(k1, k2);
        let rev = app.session.active().unwrap().revision;
        let doc_id = committed.id;
        assert_eq!(switch_region(app, doc_id, rev, k1, k2), Some(Rect::from_xywh(8, 4, 24, 20)));
        assert_eq!(switch_region(app, doc_id, rev, 0, k2), Some(Rect::from_xywh(8, 4, 24, 20)));
        assert_eq!(switch_region(app, doc_id, rev, 1 << 41, k2), None, "another preview kind");
        assert_eq!(switch_region(app, doc_id, rev + 1, k1, k2), None, "another revision");
        // Only the visible part composites now; the rest when the view gets there.
        let region = Rect::from_xywh(8, 4, 24, 20);
        let (corner, all) = (Rect::from_xywh(0, 0, 16, 16), Rect::from_xywh(0, 0, 40, 32));
        assert_eq!(switch_damage(app, k2, region, corner), Rect::new(8, 4, 16, 16));
        assert_eq!(uncovered(app, doc_id, k2, corner), None);
        assert_eq!(uncovered(app, doc_id, k1, all), None, "another preview");
        assert_eq!(uncovered(app, doc_id, k2, all), Some(region));
        assert_eq!(uncovered(app, doc_id, k2, all), None);
        assert_eq!(switch_damage(app, 0, region, corner), region, "back to the document: everything");
        assert_eq!(uncovered(app, doc_id, k2, all), None);
        // The preview equals the command with the dialog's settings.
        let params = crate::filter_dialog::params_of(&app.ui.dialog_mut(id).unwrap().fields);
        let want = destructive(&committed, target, "levels", &params).unwrap();
        assert!(max_diff(&second, &want) <= 1.0 / 255.0);
        // Preview off: the committed document.
        app.ui.dialog_mut(id).unwrap().fields.insert("__preview".into(), json!(false));
        assert!(display_doc(app, 0).is_none());
        assert!(on_layer(app, 0), "still the layer path, just hidden");
        harness.run_steps(2);
        assert!(Arc::ptr_eq(&harness.state().session.active().unwrap().doc, &committed), "previewing leaves the document alone");
        if confirm {
            crate::dialogs::confirm(harness.state_mut(), id).unwrap();
            let st = harness.state().session.active().unwrap();
            assert_eq!(st.history.past_len(), steps + 1, "one history step");
            assert!(st.doc.layer(PREVIEW_LAYER).is_none());
            assert!(max_diff(&st.doc, &want) <= 1e-6, "OK runs the real command");
        } else {
            harness.state_mut().ui.close_dialog(id);
            harness.run_steps(2);
            let st = harness.state().session.active().unwrap();
            assert!(Arc::ptr_eq(&st.doc, &committed), "Cancel leaves the document untouched");
            assert_eq!(st.history.past_len(), steps);
        }
        let app = harness.state_mut();
        assert!(display_doc(app, 0).is_none() && !on_layer(app, 0));
    }
}

/// Targets the layer path can't reproduce keep the CPU proxy preview.
#[test]
fn ineligible_dialogs_keep_the_proxy_preview() {
    // A clipped target.
    let (doc, _) = document(ColorMode::Rgb, SampleType::U8, false);
    let clipped = doc.layers[2].id;
    let mut app = app_with(doc, clipped);
    crate::adjust_dialog::open(&mut app, "image.adjustments.curves").unwrap();
    assert!(!on_layer(&mut app, 0) && display_doc(&mut app, 0).is_none());
    // An alpha-channel target.
    let (doc, target) = document(ColorMode::Rgb, SampleType::U8, false);
    let mut app = app_with(doc, target);
    app.session.execute("select.all", json!({})).unwrap();
    app.session.execute("select.saveSelection", json!({})).unwrap();
    if let Some(st) = app.session.active_mut() {
        st.channel_view.target = photocraft_engine::channel_cmds::ChannelTarget::Alpha(0);
    }
    crate::adjust_dialog::open(&mut app, "image.adjustments.levels").unwrap();
    assert!(!on_layer(&mut app, 0));
}

#[test]
fn proxy_factor_follows_the_zoom() {
    assert_eq!(proxy_factor(1.0), 1);
    assert_eq!(proxy_factor(0.6), 1);
    assert_eq!(proxy_factor(0.5), 2);
    assert_eq!(proxy_factor(0.2), 4);
    assert_eq!(proxy_factor(0.1), 8);
    assert_eq!(proxy_factor(0.0001), 64);
    assert_eq!(proxy_factor(0.0), 1);
    assert_eq!(proxy_factor(f32::NAN), 1);
}

/// The zoomed-out proxy keeps the preview layer (with the dialog's settings) under its own ids.
#[test]
fn proxy_preview_has_its_own_ids() {
    let (doc, target) = document(ColorMode::Rgb, SampleType::U8, true);
    let base = base_document(&doc, target).unwrap();
    let proxy = proxy_base(&base, 2);
    assert_eq!((proxy.size.width, proxy.size.height), (20, 16));
    assert_ne!(proxy.id, doc.id);
    assert!(proxy.walk().iter().all(|(_, _, l)| doc.layer(l.id).is_none() && l.id != PREVIEW_LAYER));
    let p = proxy_with_settings(&proxy, "curves", &sample("curves")).unwrap();
    let full = with_settings(&base, "curves", &sample("curves")).unwrap();
    let (a, b) = (photocraft_compose::render(&p, p.bounds()), photocraft_compose::render(&full, full.bounds()));
    // Nearest-neighbour proxy: pixel (x, y) shows the full composite at (2x, 2y).
    for y in 0..16 {
        for x in 0..20 {
            let (pa, pb) = (a.px[y * 20 + x], b.px[y * 2 * 40 + x * 2]);
            assert!((0..4).all(|c| (pa[c] - pb[c]).abs() < 2.0 / 255.0), "({x}, {y}) {pa:?} vs {pb:?}");
        }
    }
    assert!(proxy_with_settings(&proxy, "levels", &json!({"gamma": "x"})).is_err());
}
