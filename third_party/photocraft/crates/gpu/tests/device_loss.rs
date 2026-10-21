//! The compositor on a lost device (#243): renders return `Unsupported` (so callers use the CPU
//! compositor) instead of panicking with "Buffer 'pc_compose_uniforms' is invalid". Skips when no
//! GPU adapter exists.

use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent};
use photocraft_geom::{Rect, Size};
use photocraft_gpu::{Compositor, DeviceHealth, Fault, render_to_vec};
use photocraft_raster::Surface;

/// Concurrent wgpu instances in one process segfault on some drivers (see `parity.rs`).
static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skipping device-loss tests: no adapter ({e})");
            return None;
        }
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some((adapter, device, queue))
}

fn doc() -> Document {
    let mut d = Document::with_background("t", Size::new(600, 400), ColorMode::Rgb, SampleType::U8, Color::rgba(0.9, 0.9, 0.9, 1.0));
    let mut s = Surface::new(d.pixel_format());
    s.fill_rect(Rect::new(20, 20, 500, 300), &[0.8, 0.1, 0.1, 0.7]);
    d.layers.push(Layer::new("red", LayerContent::Raster(s)));
    d
}

#[test]
fn injected_fault_skips_the_gpu() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let Ok(mut comp) = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)) else { return };
    let health = DeviceHealth::watch(&device);
    comp.set_health(health.clone());
    let d = doc();
    assert!(render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).is_ok());
    assert!(comp.fault().is_none());
    health.mark(Fault::Lost("injected".into()));
    let e = render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).unwrap_err();
    assert!(e.0.contains("GPU device lost: injected"), "{e:?}");
}

#[test]
fn destroyed_device_returns_an_error_not_a_panic() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let Ok(mut comp) = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)) else { return };
    let health = DeviceHealth::watch(&device);
    comp.set_health(health.clone());
    let d = doc();
    assert!(render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).is_ok());
    device.destroy();
    // The first render after the loss runs into it (uploads, uniforms, submit, readback);
    // every one after sees the flag and touches nothing.
    let mut d2 = d.clone();
    d2.layers[1].opacity = 0.4;
    for _ in 0..3 {
        assert!(render_to_vec(&mut comp, &device, &queue, &d2, d2.bounds()).is_err());
    }
    assert!(health.fault().is_some(), "loss not detected");
    assert!(health.errors() > 0 || health.fault().is_some_and(|f| f.is_lost()));
}

#[test]
fn health_wait_on_a_destroyed_device_is_safe() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((_adapter, device, _queue)) = device() else { return };
    let health = DeviceHealth::watch(&device);
    assert!(health.wait(&device, None));
    device.destroy();
    // Polling a destroyed device panics inside wgpu; the wait records a loss instead.
    health.wait(&device, None);
    health.poll(&device);
    assert!(!health.is_ok());
    assert!(!health.wait(&device, None));
}

fn assert_unwatched_destroyed_device(encode_only: bool) {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let mut comp = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
    let d = Document::with_background("synthetic", Size::new(1, 1), ColorMode::Rgb, SampleType::U8, Color::rgba(0.9, 0.9, 0.9, 1.0));
    assert!(render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).is_ok());
    let mut encoder = device.create_command_encoder(&Default::default());
    device.destroy();
    let mut changed = d.clone();
    changed.layers[0].opacity = 0.4;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if encode_only {
            comp.encode(&device, &queue, &mut encoder, &changed, changed.bounds(), &mut |_, _| {})
        } else {
            comp.render(&device, &queue, &changed, changed.bounds(), |_, _| {})
        }
    }));
    assert!(result.is_ok(), "an unwatched device failure must return Unsupported instead of panicking");
    let error = result.unwrap().expect_err("a destroyed device must not report successful GPU work");
    assert!(error.0.contains("GPU work failed"), "{error}");
}

#[test]
fn an_unwatched_destroyed_device_returns_an_error_without_panicking() {
    assert_unwatched_destroyed_device(false);
}

#[test]
fn an_unwatched_encode_on_a_destroyed_device_returns_an_error_without_panicking() {
    assert_unwatched_destroyed_device(true);
}

#[test]
fn loss_during_an_unwatched_render_returns_an_error_without_panicking() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let mut comp = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
    comp.set_texture_limit(256);
    let d = Document::with_background("synthetic", Size::new(512, 1), ColorMode::Rgb, SampleType::U8, Color::rgba(0.9, 0.9, 0.9, 1.0));
    let mut chunks = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        comp.render(&device, &queue, &d, d.bounds(), |_, _| {
            chunks += 1;
            if chunks == 1 {
                // A driver reset can arrive after the entry-point health check too.
                device.destroy();
            }
        })
    }));
    assert!(chunks > 0, "the injection must occur after GPU work began");
    assert!(result.is_ok(), "device loss mid-render must return Unsupported instead of panicking");
    assert!(result.unwrap().is_err(), "a partially failed render must not report success");
}

#[test]
fn a_captured_render_error_preserves_the_device_for_a_later_render() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let mut comp = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
    let d = Document::with_background("synthetic", Size::new(1, 1), ColorMode::Rgb, SampleType::U8, Color::rgba(0.9, 0.9, 0.9, 1.0));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        comp.render(&device, &queue, &d, d.bounds(), |_, _| {
            // Reject a buffer above the device's valid capability without actually allocating it.
            let _ = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("synthetic_invalid_buffer"),
                size: device.limits().max_buffer_size + 1,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        })
    }));
    assert!(result.is_ok(), "GPU validation errors must return Unsupported instead of panicking");
    let error = result.unwrap().expect_err("an invalid allocation must fail this render");
    assert!(error.0.contains("synthetic_invalid_buffer"), "{error}");
    let out = render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).expect("the next fitting render keeps working");
    let expected = photocraft_compose::flatten(&d);
    for (cpu, gpu) in expected.px[0].iter().zip(out[0]) {
        assert!((cpu - gpu).abs() <= 1.0 / 255.0, "the healthy GPU control differs: {cpu} vs {gpu}");
    }
}

fn assert_recovery_after_invalid_encoder(encode_only: bool) {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let mut comp = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
    let mut d = doc();
    d.layers[1].effects.items = vec![photocraft_doc::Effect::default_drop_shadow()];
    // Pipeline creation errors are not proof of missing hardware capabilities: check the
    // adapter and the features actually enabled on this device before allowing a skip.
    let required = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
    let features = device.features();
    for format in [wgpu::TextureFormat::R32Float, wgpu::TextureFormat::R16Float] {
        let adapter_usages = adapter.get_texture_format_features(format).allowed_usages;
        let guaranteed_usages = format.guaranteed_format_features(features).allowed_usages;
        if !adapter_usages.contains(required)
            || (!features.contains(wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES) && !guaranteed_usages.contains(required))
        {
            eprintln!(
                "skipping effect-map recovery test: {:?} {format:?} lacks {required:?}; adapter usages={adapter_usages:?}, guaranteed usages={guaranteed_usages:?}, device features={features:?}",
                adapter.get_info().backend,
            );
            return;
        }
    }
    comp.supports(&d).expect("a capable adapter must support the recovery fixture's effect-map pipelines");
    let src = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let dst = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4, usage: wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut injected = false;
    let mut sink = |encoder: &mut wgpu::CommandEncoder, _: photocraft_gpu::ChunkOut<'_>| {
        injected = true;
        // Misaligned copies invalidate the encoder, discarding all recorded effect-map work.
        encoder.copy_buffer_to_buffer(&src, 0, &dst, 0, 1);
        if encode_only {
            // encode does not finish its caller-owned encoder; force an immediate error too.
            let _ = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("synthetic_invalid_buffer"),
                size: device.limits().max_buffer_size + 1,
                usage: wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    };
    let result = if encode_only {
        let mut encoder = device.create_command_encoder(&Default::default());
        comp.encode(&device, &queue, &mut encoder, &d, d.bounds(), &mut sink)
    } else {
        comp.render(&device, &queue, &d, d.bounds(), sink)
    };
    assert!(injected, "the invalid encoder injection must reach the sink");
    let error = result.expect_err("invalid encoder must fail");
    assert!(error.0.contains("GPU work failed"), "the injected GPU error must be captured: {error}");
    let recovered = render_to_vec(&mut comp, &device, &queue, &d, d.bounds()).unwrap();
    let mut fresh = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
    let expected = render_to_vec(&mut fresh, &device, &queue, &d, d.bounds()).unwrap();
    for (a, b) in recovered.iter().zip(expected.iter()) {
        for (a, b) in a.iter().zip(b.iter()) {
            assert!((a - b).abs() <= 1.0 / 255.0, "recovered cached pixels differ: {a} vs {b}");
        }
    }
}

#[test]
fn render_recovers_effect_maps_after_invalid_encoder() {
    assert_recovery_after_invalid_encoder(false);
}

#[test]
fn encode_recovers_effect_maps_after_invalid_encoder() {
    assert_recovery_after_invalid_encoder(true);
}

#[test]
fn loss_during_intermediate_unwatched_submissions_returns_an_error() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = device() else { return };
    let mut comp = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
    comp.set_texture_limit(256);
    let mut d = Document::with_background("synthetic", Size::new(2048, 1), ColorMode::Rgb, SampleType::U8, Color::rgba(0.9, 0.9, 0.9, 1.0));
    for _ in 0..300 {
        let mut layer = d.layers[0].clone();
        layer.id = photocraft_doc::LayerId::fresh();
        layer.opacity = 0.5;
        d.layers.push(layer);
    }
    let mut chunks = 0;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        comp.render(&device, &queue, &d, d.bounds(), |_, _| {
            chunks += 1;
            if chunks == 1 {
                device.destroy();
            }
        })
    }));
    assert!(chunks > 0);
    assert!(result.is_ok(), "intermediate submit polling must not escape as a panic");
    assert!(result.unwrap().is_err());
}
