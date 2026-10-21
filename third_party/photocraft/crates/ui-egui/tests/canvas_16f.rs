//! 16/32-bit documents get an `Rgba16Float` canvas texture (#46): smooth gradients keep more than
//! 256 levels, 32-bit values above 1.0 survive for View › 32-bit Preview Options, and 8-bit
//! documents keep their `Rgba8Unorm` texture byte for byte. Skips when there is no GPU adapter or
//! it can't render to `Rgba16Float`.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use eframe::egui_wgpu::RenderState;
use eframe::wgpu::TextureFormat;
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent, Size};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::gpu_canvas::{self, GpuCanvas};
use serde_json::json;

use photocraft_testkit::gpu_lock;

/// A headless GPU canvas on a device created like the app's, if the adapter has a float canvas.
fn canvas() -> Option<(GpuCanvas, RenderState)> {
    let rs =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(gpu_canvas::wgpu_setup(), Default::default()))).ok();
    let Some(rs) = rs else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    if !gpu_canvas::supports_f16_canvas(&rs.adapter) || std::env::var("PHOTOCRAFT_CANVAS_F16").as_deref() == Ok("0") {
        eprintln!("skipping: adapter can't render/filter Rgba16Float");
        return None;
    }
    Some((GpuCanvas::new(&rs), rs))
}

/// A one-layer `w`×`h` RGB document of `depth` whose columns ramp from `lo` to `hi` (gray).
fn gradient(depth: SampleType, w: u32, h: u32, lo: f32, hi: f32) -> Document {
    let mut d = Document::new("gradient", Size::new(w, h), ColorMode::Rgb, depth);
    let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, depth, true));
    for x in 0..w {
        let v = lo + (hi - lo) * x as f32 / (w - 1) as f32;
        s.fill_rect(Rect::new(x as i32, 0, x as i32 + 1, h as i32), &[v, v, v, 1.0]);
    }
    d.layers.push(Layer::new("ramp", LayerContent::Raster(s)));
    d
}

/// Distinct red values along row `y` of read-back texels.
fn levels(texels: &[[f32; 4]], w: u32, y: u32) -> usize {
    let row = &texels[(y * w) as usize..((y + 1) * w) as usize];
    row.iter().map(|p| p[0].to_bits()).collect::<HashSet<_>>().len()
}

/// The stored texels of a document's canvas texture against its CPU composite (premultiplied).
fn max_error(doc: &Document, texels: &[[f32; 4]]) -> f32 {
    let buf = photocraft_compose::render(doc, doc.bounds());
    buf.px.iter().zip(texels).map(|(c, t)| (0..4).map(|i| ((if i < 3 { c[i] * c[3] } else { c[3] }) - t[i]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max)
}

#[test]
fn sixteen_bit_gradient_keeps_more_than_256_levels() {
    let _gpu = gpu_lock();
    let Some((g, _rs)) = canvas() else { return };
    let doc = gradient(SampleType::U16, 4096, 4, 0.0, 1.0);
    let key = doc.id.0;
    // GPU compositor (or the CPU fallback where it doesn't apply).
    let r = g.refresh(key, &doc, None, None);
    let (format, size, texels) = g.read_texels(key).expect("read back");
    assert_eq!(format, TextureFormat::Rgba16Float, "refresh path {}", r.kind);
    assert_eq!(size, [4096, 4]);
    let n = levels(&texels, 4096, 0);
    eprintln!("{} refresh: {n} levels", r.kind);
    assert!(n > 256, "{n} levels: banded");
    assert!(max_error(&doc, &texels) < 2e-3, "texels differ from the composite");
    // The CPU compositor's banded upload.
    let doc2 = gradient(SampleType::U16, 4096, 4, 0.0, 1.0);
    g.upload_composite(doc2.id.0, &doc2, None);
    let (format, _, texels) = g.read_texels(doc2.id.0).expect("read back");
    assert_eq!(format, TextureFormat::Rgba16Float);
    let n = levels(&texels, 4096, 3);
    assert!(n > 256, "CPU path: {n} levels");
    assert!(max_error(&doc2, &texels) < 2e-3, "CPU path texels differ from the composite");
    // Memory: 8 bytes per texel plus mips.
    let (_, bytes) = g.texture_info(doc2.id.0).expect("info");
    assert!((4096 * 4 * 8..4096 * 4 * 8 * 3 / 2).contains(&bytes), "{bytes} bytes");
}

#[test]
fn damage_rect_uploads_into_the_float_texture() {
    let _gpu = gpu_lock();
    let Some((g, _rs)) = canvas() else { return };
    let mut doc = gradient(SampleType::U16, 512, 64, 0.0, 1.0);
    let key = doc.id.0;
    g.upload_composite(key, &doc, None);
    // Repaint a patch and upload only it, as the CPU path does after a brush stroke.
    if let LayerContent::Raster(s) = &mut doc.layers[0].content {
        s.fill_rect(Rect::new(100, 10, 140, 30), &[0.123_45, 0.5, 0.987_65, 1.0]);
    }
    let r = g.refresh(key, &doc, Some(Rect::new(100, 10, 140, 30)), None);
    assert!(r.kind.ends_with("rect"), "{}", r.kind);
    let (_, _, texels) = g.read_texels(key).expect("read back");
    let p = texels[20 * 512 + 120];
    assert!((p[0] - 0.123_45).abs() < 1e-3 && (p[2] - 0.987_65).abs() < 1e-3, "{p:?}");
    assert!(max_error(&doc, &texels) < 2e-3);
}

#[test]
fn eight_bit_documents_keep_their_rgba8_texture() {
    let _gpu = gpu_lock();
    let Some((g, _rs)) = canvas() else { return };
    let doc = gradient(SampleType::U8, 1024, 4, 0.0, 1.0);
    // CPU path: exactly the premultiplied RGBA8 of the composite, as before.
    g.upload_composite(doc.id.0, &doc, None);
    let (format, _, texels) = g.read_texels(doc.id.0).expect("read back");
    assert_eq!(format, TextureFormat::Rgba8Unorm);
    let want = gpu_canvas::premultiply_rgba8(&photocraft_compose::render(&doc, doc.bounds()).px);
    let got: Vec<u8> = texels.iter().flat_map(|p| p.map(|v| (v * 255.0).round() as u8)).collect();
    assert!(got == want, "8-bit texels changed");
    assert!(levels(&texels, 1024, 0) <= 256);
    // GPU path: still RGBA8.
    let doc2 = gradient(SampleType::U8, 1024, 4, 0.0, 1.0);
    g.refresh(doc2.id.0, &doc2, None, None);
    assert_eq!(g.texture_info(doc2.id.0).map(|i| i.0), Some(TextureFormat::Rgba8Unorm));
    // Converting to 16 bits switches the texture to float.
    let mut doc3 = doc2.clone();
    doc3.depth = SampleType::U16;
    if let LayerContent::Raster(s) = &mut doc3.layers[0].content {
        *s = s.convert(PixelFormat::new(ColorMode::Rgb, SampleType::U16, true));
    }
    g.refresh(doc3.id.0, &doc3, Some(Rect::new(0, 0, 4, 4)), None);
    assert_eq!(g.texture_info(doc3.id.0).map(|i| i.0), Some(TextureFormat::Rgba16Float));
}

#[test]
fn format_follows_depth_and_budget() {
    let _gpu = gpu_lock();
    let Some((g, _rs)) = canvas() else { return };
    assert_eq!(g.format_for(SampleType::U8, [4000, 4000]), TextureFormat::Rgba8Unorm);
    assert_eq!(g.format_for(SampleType::U16, [7360, 4912]), TextureFormat::Rgba16Float);
    assert_eq!(g.format_for(SampleType::F32, [7360, 4912]), TextureFormat::Rgba16Float);
    // Over the budget (a 14000² 16-bit document): 8-bit, so it stays on the GPU.
    if std::env::var_os("PHOTOCRAFT_CANVAS_F16").is_none() {
        assert_eq!(g.format_for(SampleType::U16, [14000, 14000]), TextureFormat::Rgba8Unorm);
    }
}

#[test]
fn thirty_two_bit_values_above_one_are_kept() {
    let _gpu = gpu_lock();
    let Some((g, _rs)) = canvas() else { return };
    let doc = gradient(SampleType::F32, 256, 2, 0.0, 4.0);
    g.refresh(doc.id.0, &doc, None, None);
    let (format, _, texels) = g.read_texels(doc.id.0).expect("read back");
    assert_eq!(format, TextureFormat::Rgba16Float);
    let last = texels[255][0];
    assert!((last - 4.0).abs() < 0.01, "{last}");
    let doc2 = gradient(SampleType::F32, 256, 2, 0.0, 4.0);
    g.upload_composite(doc2.id.0, &doc2, None);
    let (_, _, texels) = g.read_texels(doc2.id.0).expect("read back");
    assert!((texels[255][0] - 4.0).abs() < 0.01, "CPU path {}", texels[255][0]);
}

/// A rendered screen: width and RGBA8 rows.
struct Screen {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

impl Screen {
    fn green(&self, x: u32, y: u32) -> u8 {
        self.px[((y * self.w + x) * 4 + 1) as usize]
    }
}

/// Renders the app with `doc` open after `setup` commands (saving the PNG as `name` in the temp
/// dir), or None when the GPU can't be used.
fn screen(doc: Document, setup: &[(&str, serde_json::Value)], name: &str) -> Option<Screen> {
    let ok = Arc::new(AtomicBool::new(false));
    let ok2 = ok.clone();
    let setup: Vec<(String, serde_json::Value)> = setup.iter().map(|(a, b)| (a.to_string(), b.clone())).collect();
    let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
        egui_kittest::Harness::builder().with_size(egui::vec2(1000.0, 700.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            if let Some(rs) = cc.wgpu_render_state.as_ref() {
                ok2.store(gpu_canvas::supports_f16_canvas(&rs.adapter) && std::env::var("PHOTOCRAFT_CANVAS_F16").as_deref() != Ok("0"), Ordering::SeqCst);
                app.set_wgpu(rs.clone());
            }
            app.session.open_document(doc, None);
            app
        })
    }));
    let Ok(mut harness) = built else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    if !ok.load(Ordering::SeqCst) {
        eprintln!("skipping: adapter can't render/filter Rgba16Float");
        return None;
    }
    // Outside the catch_unwind: a failing setup command fails the test instead of reading as
    // "no GPU adapter" (#760).
    for (id, p) in setup {
        harness.state_mut().run(&id, p).unwrap_or_else(|e| panic!("setup command {id}: {e}"));
    }
    harness.run_steps(6);
    let img = harness.render().expect("render");
    let out = std::env::temp_dir().join(name);
    let _ = img.save(&out);
    eprintln!("screen: {}", out.display());
    Some(Screen { w: img.width(), h: img.height(), px: img.into_raw() })
}

/// Distinct gray levels along a screen row inside `x0..x1`, and the largest step between
/// neighbouring pixels.
fn screen_levels(img: &Screen, y: u32, x0: u32, x1: u32) -> (usize, i32) {
    let row: Vec<u8> = (x0..x1).map(|x| img.green(x, y)).collect();
    let steps = row.windows(2).map(|w| (w[1] as i32 - w[0] as i32).abs()).max().unwrap_or(0);
    (row.iter().collect::<HashSet<_>>().len(), steps)
}

#[test]
fn sixteen_bit_gradient_on_screen() {
    let _gpu = gpu_lock();
    // A 16-bit ramp drawn by the app's canvas: smooth, monotonic, no visible steps.
    let doc = gradient(SampleType::U16, 2048, 1024, 0.0, 1.0);
    let Some(img) = screen(doc, &[], "photocraft-canvas-16f-gradient.png") else { return };
    let (n, step) = screen_levels(&img, img.h / 2, 150, 600);
    assert!(n > 100 && step <= 3, "{n} levels, max step {step}");
}

#[test]
fn thirty_two_bit_preview_exposes_values_above_one() {
    let _gpu = gpu_lock();
    // A dark 32-bit ramp brightened +4 stops by View › 32-bit Preview Options: with an 8-bit
    // canvas texture its 0..1/16 range has 16 codes (banding); the float texture keeps it smooth.
    let doc = gradient(SampleType::F32, 2048, 1024, 0.0, 1.0 / 16.0);
    let Some(img) = screen(doc, &[("view.thirtyTwoBitPreviewOptions", json!({"exposure": 4.0, "gamma": 1.0}))], "photocraft-canvas-16f-hdr-ramp.png") else {
        return;
    };
    let (n, _) = screen_levels(&img, img.h / 2, 150, 600);
    assert!(n > 40, "{n} levels: banded");
    // Values above 1.0 are brought into range by a negative exposure instead of clipped: 2.0
    // at −3 stops shows as sRGB(lin(2.0) / 8) rather than sRGB(lin(1.0) / 8).
    let mut flat = Document::new("hdr", Size::new(512, 512), ColorMode::Rgb, SampleType::F32);
    let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::F32, true));
    s.fill_rect(Rect::new(0, 0, 512, 512), &[2.0, 2.0, 2.0, 1.0]);
    flat.layers.push(Layer::new("bright", LayerContent::Raster(s)));
    let Some(img) = screen(flat, &[("view.thirtyTwoBitPreviewOptions", json!({"exposure": -3.0, "gamma": 1.0}))], "photocraft-canvas-16f-hdr-flat.png") else {
        return;
    };
    let got = img.green(img.w * 2 / 5, img.h / 2) as f32;
    let dec = |v: f32| if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    let enc = |v: f32| if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
    let want = enc(dec(2.0) / 8.0) * 255.0;
    let clipped = enc(dec(1.0) / 8.0) * 255.0;
    eprintln!("shown {got}, want {want:.1} (clipped would be {clipped:.1})");
    assert!((got - want).abs() <= 3.0, "shown {got}, want {want:.1} (clipped {clipped:.1})");
}
