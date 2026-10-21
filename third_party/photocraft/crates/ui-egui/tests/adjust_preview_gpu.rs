//! GPU parity of the Image › Adjustments dialog preview (`adjust_preview`): the document with
//! the temporary clipped adjustment layer must composite on the wgpu compositor like on the CPU
//! reference, and like the destructive command's result, for every kind with a dialog. Skips
//! without an adapter that renders 32-bit float targets (like `crates/gpu/tests/parity.rs`).

use eframe::wgpu;
use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerId, LayerMask, Size};
use photocraft_geom::Rect;
use photocraft_gpu::{Compositor, render_to_vec};
use photocraft_ui_egui::adjust_preview::{PREVIEW_LAYER, preview_document};
use serde_json::{Value, json};

fn block_on<F: std::future::Future>(f: F) -> F::Output {
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    let mut f = std::pin::pin!(f);
    loop {
        if let std::task::Poll::Ready(v) = f.as_mut().poll(&mut cx) {
            return v;
        }
        std::thread::yield_now();
    }
}

fn gpu() -> Option<(wgpu::Device, wgpu::Queue, Compositor)> {
    let adapter = match block_on(wgpu::Instance::default().request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skipping adjustment preview GPU parity: no adapter ({e})");
            return None;
        }
    };
    if Compositor::preferred_acc_format(&adapter) != wgpu::TextureFormat::Rgba32Float {
        eprintln!("skipping adjustment preview GPU parity: adapter can't render Rgba32Float");
        return None;
    }
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    let comp = Compositor::try_new_with_format(&device, wgpu::TextureFormat::Rgba32Float).ok()?;
    Some((device, queue, comp))
}

fn rnd(seed: u32, i: u32) -> f32 {
    let mut h = seed.wrapping_mul(0x9e37_79b9) ^ i.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    (h & 0xffff) as f32 / 65535.0
}

fn noise(l: &mut Layer, r: Rect, seed: u32, min_alpha: f32) {
    let s = l.surface_mut().unwrap();
    let fmt = s.format();
    let ch = fmt.channels();
    let data: Vec<f32> = (0..r.width() * r.height())
        .flat_map(|i| {
            (0..ch).map(move |c| {
                let v = rnd(seed + c as u32 * 7919, i);
                if fmt.alpha && c == ch - 1 { min_alpha + (1.0 - min_alpha) * v } else { v }
            })
        })
        .collect();
    s.write_region(r, &data);
}

fn document(mode: ColorMode, depth: SampleType, selection: bool) -> (Document, LayerId) {
    let mut doc = Document::with_background("t", Size::new(48, 40), mode, depth, Color::rgba(0.7, 0.6, 0.5, 1.0));
    let fmt = doc.pixel_format();
    let mut target = Layer::raster("target", fmt);
    noise(&mut target, Rect::from_xywh(4, 2, 38, 34), 11, 0.2);
    target.opacity = 0.8;
    target.blend = BlendMode::Multiply;
    let mut m = LayerMask::reveal_all();
    m.surface.fill_rect(Rect::from_xywh(0, 0, 12, 12), &[0.3]);
    target.mask = Some(m);
    let id = target.id;
    let mut clip = Layer::raster("clipped", fmt);
    noise(&mut clip, Rect::from_xywh(20, 10, 14, 14), 17, 0.5);
    clip.clipped = true;
    clip.opacity = 0.6;
    let mut top = Layer::raster("top", fmt);
    noise(&mut top, Rect::from_xywh(0, 24, 48, 16), 23, 0.0);
    top.blend = BlendMode::Screen;
    doc.layers.extend([target, clip, top]);
    if selection {
        let mut sel = photocraft_raster::Surface::new(PixelFormat::GRAY8);
        sel.fill_rect(Rect::from_xywh(8, 4, 30, 26), &[1.0]);
        sel.fill_rect(Rect::from_xywh(8, 4, 8, 26), &[0.5]);
        doc.selection = Some(sel);
    }
    (doc, id)
}

fn sample(kind: &str) -> Value {
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

fn pm(p: &[f32; 4]) -> [f32; 4] {
    [p[0] * p[3], p[1] * p[3], p[2] * p[3], p[3]]
}

fn diffs(a: &[[f32; 4]], b: &[[f32; 4]]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| (0..4).map(|c| (pm(x)[c] - pm(y)[c]).abs()).fold(0.0, f32::max)).collect()
}

/// Largest difference, ignoring up to `skip` outliers.
fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]], skip: usize) -> f32 {
    let mut d = diffs(a, b);
    d.sort_by(|x, y| y.total_cmp(x));
    d.get(skip).copied().unwrap_or(0.0)
}

#[test]
fn preview_layer_composites_on_the_gpu_like_the_command() {
    let Some((device, queue, mut comp)) = gpu() else { return };
    // GPU vs CPU of the same document (as the compositor parity tests), and GPU preview vs the
    // command's CPU result (adds the command's rounding to the layer depth).
    let (tol_gpu, tol_cmd) = (2.0 / 255.0, 2.5 / 255.0);
    let mut worst = 0.0f32;
    for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for selection in [false, true] {
                let (doc, target) = document(mode, depth, selection);
                for kind in photocraft_ui_egui::adjust_editors::KINDS {
                    let params = sample(kind);
                    let Ok(preview) = preview_document(&doc, target, kind, &params) else {
                        // Grayscale kinds that make colour preview on the CPU proxy.
                        assert_eq!(mode, ColorMode::Grayscale, "{kind}");
                        continue;
                    };
                    assert!(preview.layer(PREVIEW_LAYER).is_some());
                    let what = format!("{kind} {mode:?} {depth:?} selection {selection}");
                    let b = preview.bounds();
                    // The GPU holds 16- and 32-bit layers as half floats: Threshold and Posterize
                    // may flip a few pixels sitting right at a step (1 % allowed).
                    let skip = if matches!(kind, "threshold" | "posterize") && depth != SampleType::U8 { (b.width() * b.height()) as usize / 100 } else { 0 };
                    let on_gpu = render_to_vec(&mut comp, &device, &queue, &preview, b).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let on_cpu = photocraft_compose::render(&preview, b).px;
                    let d = max_diff(&on_gpu, &on_cpu, skip);
                    assert!(d <= tol_gpu, "{what}: GPU vs CPU {:.2}/255", d * 255.0);
                    let mut s = photocraft_engine::Session::new();
                    s.add_document(doc.clone(), None);
                    s.select_layer(target).unwrap();
                    s.execute(&format!("image.adjustments.{kind}"), params.clone()).unwrap();
                    let cmd = photocraft_compose::render(&s.active().unwrap().doc, b).px;
                    let d = max_diff(&on_gpu, &cmd, skip);
                    worst = worst.max(d);
                    assert!(d <= tol_cmd, "{what}: GPU preview vs command {:.2}/255", d * 255.0);
                }
            }
        }
    }
    eprintln!("worst GPU preview vs command: {:.3}/255", worst * 255.0);
}

/// The zoomed-out proxy preview (own document and layer ids) composites on the GPU like the CPU.
#[test]
fn proxy_preview_composites_on_the_gpu() {
    use photocraft_ui_egui::adjust_preview::{base_document, proxy_base, proxy_with_settings};
    let Some((device, queue, mut comp)) = gpu() else { return };
    for selection in [false, true] {
        let (doc, target) = document(ColorMode::Rgb, SampleType::U8, selection);
        let proxy = proxy_base(&base_document(&doc, target).unwrap(), 2);
        // Render the full document too, so both share the compositor's resident textures.
        render_to_vec(&mut comp, &device, &queue, &doc, doc.bounds()).unwrap();
        for kind in photocraft_ui_egui::adjust_editors::KINDS {
            let p = proxy_with_settings(&proxy, kind, &sample(kind)).unwrap();
            let on_gpu = render_to_vec(&mut comp, &device, &queue, &p, p.bounds()).unwrap();
            let on_cpu = photocraft_compose::render(&p, p.bounds()).px;
            let d = max_diff(&on_gpu, &on_cpu, 0);
            assert!(d <= 2.0 / 255.0, "{kind} selection {selection}: {:.2}/255", d * 255.0);
        }
    }
}
