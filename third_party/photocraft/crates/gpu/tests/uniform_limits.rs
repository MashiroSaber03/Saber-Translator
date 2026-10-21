//! A deep stack that cannot fit the device's uniform buffer must fall back without panicking.

use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, Fill, Layer, LayerContent};
use photocraft_geom::Size;
use photocraft_gpu::{Compositor, DeviceHealth, plan, render_to_vec};

static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn fills(count: usize) -> Document {
    fills_size(count, Size::new(1, 1))
}

fn fills_size(count: usize, size: Size) -> Document {
    let mut doc = Document::new("Small deep stack", size, ColorMode::Rgb, SampleType::U8);
    for _ in 0..count {
        doc.layers.push(Layer::new("Translucent fill", LayerContent::Fill(Fill::Solid(Color::rgba(0.25, 0.5, 0.75, 0.5)))));
    }
    doc
}

fn constrained_device() -> Option<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())) {
        Ok(adapter) => adapter,
        Err(error) => {
            eprintln!("skipping uniform-limit test: no GPU adapter ({error})");
            return None;
        }
    };
    let limits =
        wgpu::Limits { max_buffer_size: 65_536, max_storage_buffer_binding_size: 65_536, max_uniform_buffer_binding_size: 65_536, ..Default::default() };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_limits: limits, ..Default::default() }))
        .expect("the supported reduced device limits must create a device");
    assert_eq!(device.limits().max_buffer_size, 65_536);
    Some((adapter, device, queue))
}

fn compositor(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Compositor {
    let mut compositor =
        Compositor::try_new_with_format(device, Compositor::preferred_acc_format(adapter)).expect("the reduced-limit device supports the compositor");
    compositor.set_texture_limit(256);
    compositor
}

fn healthy_control(compositor: &mut Compositor, device: &wgpu::Device, queue: &wgpu::Queue) {
    let small = fills(2);
    let expected = photocraft_compose::flatten(&small);
    let actual = render_to_vec(compositor, device, queue, &small, small.bounds()).expect("a fitting stack renders on the same device");
    assert_eq!(actual.len(), 1);
    for (cpu, gpu) in expected.px[0].iter().zip(actual[0]) {
        assert!((cpu - gpu).abs() <= 1.0 / 255.0, "the healthy small-stack CPU control differs: {cpu} vs {gpu}");
    }
}

#[test]
fn a_deep_stack_above_the_device_buffer_limit_returns_an_error_without_panicking() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = constrained_device() else { return };
    let mut compositor = compositor(&adapter, &device);
    healthy_control(&mut compositor, &device, &queue);
    let deep = fills(150);
    eprintln!(
        "uniform-limit reproduction: backend={:?}, buffer limit={}, page={}, passes={}",
        adapter.get_info().backend,
        device.limits().max_buffer_size,
        compositor.page_size(),
        plan(&deep).unwrap().passes.len()
    );
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| render_to_vec(&mut compositor, &device, &queue, &deep, deep.bounds())));
    assert!(result.is_ok(), "rendering an oversized uniform buffer must return Unsupported, not panic");
    assert!(result.unwrap().is_err(), "a stack above the device's buffer limit must return Unsupported");
}

#[test]
fn an_oversized_stack_falls_back_without_faulting_the_watched_device() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = constrained_device() else { return };
    let health = DeviceHealth::watch(&device);
    let mut compositor = compositor(&adapter, &device);
    compositor.set_health(health.clone());
    healthy_control(&mut compositor, &device, &queue);
    let deep = fills(150);
    let fallback = match render_to_vec(&mut compositor, &device, &queue, &deep, deep.bounds()) {
        Ok(_) => panic!("the oversized stack must return Unsupported for CPU fallback"),
        Err(_) => photocraft_compose::flatten(&deep),
    };
    for (actual, expected) in fallback.px[0].iter().zip([0.25, 0.5, 0.75, 1.0]) {
        assert!((actual - expected).abs() <= 1.0 / 255.0, "CPU fallback preserves the translucent-fill composite: {actual} vs {expected}");
    }
    let small = fills(2);
    let next = render_to_vec(&mut compositor, &device, &queue, &small, small.bounds());
    eprintln!("watched uniform-limit reproduction: errors={}, fault={:?}", health.errors(), health.fault());
    eprintln!("next fitting stack after fallback: {next:?}");
    assert!(health.is_ok(), "the unsupported stack must leave the device healthy: {:?}", health.fault());
    assert_eq!(health.errors(), 0, "no invalid GPU operation is issued before falling back");
    healthy_control(&mut compositor, &device, &queue);
}

#[test]
fn the_exact_buffer_limit_still_renders_with_cpu_parity() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = constrained_device() else { return };
    let health = DeviceHealth::watch(&device);
    let mut compositor = compositor(&adapter, &device);
    compositor.set_health(health.clone());
    let boundary = fills(127);
    assert_eq!(plan(&boundary).unwrap().passes.len(), 255, "one chunk plus255passes uses exactly65536bytes");
    let expected = photocraft_compose::flatten(&boundary);
    let actual = render_to_vec(&mut compositor, &device, &queue, &boundary, boundary.bounds()).expect("the exact supported buffer boundary is accepted");
    assert_eq!(actual.len(), 1);
    for (cpu, gpu) in expected.px[0].iter().zip(actual[0]) {
        assert!((cpu - gpu).abs() <= 1.0 / 255.0, "the boundary render differs from CPU: {cpu} vs {gpu}");
    }
    assert_eq!(health.errors(), 0);
    assert!(health.is_ok());
}

#[test]
fn later_oversized_page_cells_are_rejected_before_render_or_encode_calls_the_sink() {
    let _lock = GPU_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some((adapter, device, queue)) = constrained_device() else { return };
    let health = DeviceHealth::watch(&device);
    let doc = fills_size(126, Size::new(4096, 2048));
    let region = photocraft_geom::Rect::new(2047, 0, 4096, 2048);
    assert_eq!(plan(&doc).unwrap().passes.len(), 253);
    // Default pages are2048, chunks1024: the narrow first page has2chunks and fits;
    // the later full page has4chunks and would exceed the same65536byte device limit.
    for encode_only in [false, true] {
        let mut compositor = Compositor::try_new_with_format(&device, Compositor::preferred_acc_format(&adapter)).unwrap();
        compositor.set_health(health.clone());
        assert_eq!(compositor.page_size(), 2048);
        let mut calls = 0;
        let mut sink = |_: &mut wgpu::CommandEncoder, _: photocraft_gpu::ChunkOut<'_>| calls += 1;
        let result = if encode_only {
            let mut encoder = device.create_command_encoder(&Default::default());
            compositor.encode(&device, &queue, &mut encoder, &doc, region, &mut sink)
        } else {
            compositor.render(&device, &queue, &doc, region, &mut sink)
        };
        assert!(result.is_err(), "a later cell's uniform requirement must trigger CPU fallback");
        assert_eq!(calls, 0, "preflight must reject the whole render before any partial sink output");
        assert_eq!(health.errors(), 0);
        assert!(health.is_ok());
    }
}
