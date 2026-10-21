//! Read every mip, rather than only level zero: a clean 100% view does not prove that
//! reduced previews are valid. Keep all cases on one device (some drivers dislike concurrent
//! test devices), and use synthetic pixels instead of event photographs.

use super::*;
use photocraft_compose::Buffer;
use photocraft_doc::SampleType;
use photocraft_geom::Rect;

fn read_mips(gpu: &GpuCanvas, key: u64) -> Vec<Vec<[f32; 4]>> {
    let renderer = gpu.rs.renderer.read();
    let doc = &renderer.callback_resources.get::<Resources>().unwrap().docs[&key];
    assert_eq!(doc.tiles.len(), 1);
    let tile = &doc.tiles[0];
    let (device, queue) = (&gpu.rs.device, &gpu.rs.queue);
    let bpp = texel_bytes(doc.format) as usize;
    (0..tile.levels.len())
        .map(|level| {
            let (w, h) = ((doc.size[0] >> level).max(1), (doc.size[1] >> level).max(1));
            let row = (w as usize * bpp).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mip_test_readback"),
                size: (row * h as usize) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &tile.texture, mip_level: level as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo {
                    buffer: &buffer,
                    layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row as u32), rows_per_image: Some(h) },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            queue.submit([encoder.finish()]);
            buffer.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
            assert!(gpu.health.wait(device, None));
            let data = buffer.slice(..).get_mapped_range().unwrap();
            let mut pixels = Vec::new();
            for y in 0..h as usize {
                for x in 0..w as usize {
                    let offset = y * row + x * bpp;
                    pixels.push(texel_to_f32(doc.format, &data[offset..offset + bpp]));
                }
            }
            pixels
        })
        .collect()
}

#[test]
fn canvas_mips_preserve_opaque_pixels_and_partial_updates() {
    let rs = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| egui_kittest::wgpu::create_render_state(wgpu_setup(), Default::default())));
    let Ok(rs) = rs else {
        eprintln!("skipping mip readback: no GPU adapter");
        return;
    };
    eprintln!("mip readback adapter: {:?}", rs.adapter.get_info());
    let gpu = GpuCanvas::new(&rs);
    // Both mip paths: straight into the level, and the DX12 scratch-texture workaround.
    for separate in [false, true] {
        gpu.rs.renderer.write().callback_resources.get_mut::<Resources>().unwrap().separate_mip_targets = separate;
        check_mips(&gpu);
    }
}

fn check_mips(gpu: &GpuCanvas) {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        for (w, h) in [(1791, 1083), (513, 1), (1, 513)] {
            let mut full = Buffer::filled(Rect::from_xywh(0, 0, w, h), [0.25, 0.5, 0.75, 1.0]);
            // Reusing the key also covers replacing textures after a size or depth change.
            gpu.upload_buffer_full(1, &full, depth);
            for (level, pixels) in read_mips(gpu, 1).iter().enumerate() {
                for pixel in pixels {
                    assert!((pixel[3] - 1.0).abs() < 0.001, "{depth:?} {w}x{h} mip {level}: {pixel:?}");
                    for (actual, expected) in pixel.iter().zip([0.25, 0.5, 0.75, 1.0]) {
                        assert!((actual - expected).abs() < 0.005, "{depth:?} {w}x{h} mip {level}: {pixel:?}");
                    }
                }
            }
            let patch = Rect::from_xywh((w / 2) as i32, (h / 2) as i32, (w / 3).max(1), (h / 3).max(1));
            let changed = Buffer::filled(patch, [1.0, 0.0, 0.0, 1.0]);
            assert!(gpu.upload_buffer_rect(1, &changed));
            for y in patch.y0..patch.y1 {
                for x in patch.x0..patch.x1 {
                    full.px[y as usize * w as usize + x as usize] = [1.0, 0.0, 0.0, 1.0];
                }
            }
            gpu.upload_buffer_full(2, &full, depth);
            let partial = read_mips(gpu, 1);
            let rebuilt = read_mips(gpu, 2);
            for (level, (actual, expected)) in partial.iter().zip(&rebuilt).enumerate() {
                for (a, b) in actual.iter().zip(expected) {
                    assert!((a[3] - 1.0).abs() < 0.001, "partial {depth:?} mip {level}: {a:?}");
                    for (a, b) in a.iter().zip(b) {
                        assert!((a - b).abs() < 0.005, "partial/full mismatch {depth:?} {w}x{h} mip {level}: {a} != {b}");
                    }
                }
            }
        }
    }
}
