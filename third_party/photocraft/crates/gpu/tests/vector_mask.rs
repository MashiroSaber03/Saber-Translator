//! Vector masks on the GPU: the planner samples the layer's combined mask (pixel mask × the
//! rasterised path, `compose::masks::combined_mask`); results must match the CPU compositor.

use photocraft_color::{BlendMode, Color, ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Adjustment, Document, Effect, Layer, LayerContent, LayerMask, Path, Size, Subpath, VectorMask};
use photocraft_geom::Rect;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue, photocraft_gpu::Compositor)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    // Exact comparison needs 32-bit float targets (see parity.rs); skip on adapters without them.
    if photocraft_gpu::Compositor::preferred_acc_format(&adapter) != wgpu::TextureFormat::Rgba32Float {
        return None;
    }
    let comp = photocraft_gpu::Compositor::try_new_with_format(&device, wgpu::TextureFormat::Rgba32Float).ok()?;
    Some((device, queue, comp))
}

fn check(g: &mut (wgpu::Device, wgpu::Queue, photocraft_gpu::Compositor), d: &Document, what: &str) {
    let cpu = photocraft_compose::flatten(d);
    let out = photocraft_gpu::render_to_vec(&mut g.2, &g.0, &g.1, d, d.bounds()).unwrap_or_else(|e| panic!("{what}: {e}"));
    let worst = cpu
        .px
        .iter()
        .zip(&out)
        .map(|(c, o)| (0..4).map(|k| if k == 3 { (c[3] - o[3]).abs() } else { (c[k] * c[3] - o[k] * o[3]).abs() }).fold(0.0f32, f32::max))
        .fold(0.0f32, f32::max);
    // One 8-bit code at most; the epsilon absorbs float noise on an exact one-code step
    // (the combined mask is cached as U16 since #2417).
    assert!(worst <= 1.0 / 255.0 + 1e-6, "{what}: max diff {:.2}/255", worst * 255.0);
}

fn tri(inverted: bool, density: f32) -> VectorMask {
    let mut p = Path::new(vec![Subpath::polygon(&[(3.5, 2.0), (40.0, 6.3), (12.0, 30.7)])]);
    p.inverted = inverted;
    let mut vm = VectorMask::new(p);
    vm.density = density;
    vm
}

fn painted(name: &str, fmt: PixelFormat, rgba: [f32; 4]) -> Layer {
    let mut l = Layer::raster(name, fmt);
    l.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 44, 34), &photocraft_raster::from_rgba(&fmt, rgba));
    l
}

#[test]
fn vector_masks_match_the_cpu() {
    let Some(mut g) = gpu() else { return };
    for depth in [SampleType::U8, SampleType::U16] {
        for (inverted, density) in [(false, 1.0), (true, 1.0), (false, 0.6)] {
            let mut d = Document::with_background("v", Size::new(44, 34), ColorMode::Rgb, depth, Color::WHITE);
            let fmt = d.pixel_format();
            // Raster layer with a vector mask and a pixel mask.
            let mut l = painted("a", fmt, [0.9, 0.2, 0.1, 0.8]);
            l.vector_mask = Some(tri(inverted, density));
            let mut m = LayerMask::reveal_all();
            m.surface.fill_rect(Rect::new(20, 0, 44, 34), &[0.3]);
            m.density = 0.9;
            l.mask = Some(m);
            l.blend = BlendMode::Multiply;
            d.layers.push(l);
            // A group (isolated and pass-through) with a vector mask.
            let mut inner = painted("b", fmt, [0.1, 0.6, 0.9, 1.0]);
            inner.opacity = 0.7;
            let mut grp = Layer::group("g", vec![inner]);
            grp.vector_mask = Some(tri(!inverted, density));
            grp.blend = BlendMode::Normal;
            d.layers.push(grp.clone());
            grp.blend = BlendMode::PassThrough;
            d.layers.push(grp);
            // An adjustment layer and an effect layer with vector masks.
            let mut adj = Layer::new("inv", LayerContent::Adjustment(Adjustment::Invert));
            adj.vector_mask = Some(tri(inverted, density));
            d.layers.push(adj);
            let mut fx = painted("fx", fmt, [0.2, 0.8, 0.3, 1.0]);
            fx.surface_mut().unwrap().fill_rect(Rect::new(0, 0, 44, 34), &photocraft_raster::from_rgba(&fmt, [0.0; 4]));
            fx.surface_mut().unwrap().fill_rect(Rect::new(8, 8, 30, 26), &photocraft_raster::from_rgba(&fmt, [0.2, 0.8, 0.3, 1.0]));
            fx.vector_mask = Some(tri(inverted, density));
            fx.effects.items = vec![Effect::default_drop_shadow()];
            d.layers.push(fx);
            check(&mut g, &d, &format!("{depth:?} inverted {inverted} density {density}"));
            // Editing the path re-rasterises the mask.
            if let Some(vm) = &mut d.layers[1].vector_mask {
                vm.path.subpaths[0].knots[1].anchor.x = 30.0;
                vm.path.subpaths[0].knots[1].in_ctrl.x = 30.0;
                vm.path.subpaths[0].knots[1].out_ctrl.x = 30.0;
            }
            check(&mut g, &d, "edited path");
            // Disabling it shows the layer unmasked by the path.
            d.layers[1].vector_mask.as_mut().unwrap().enabled = false;
            check(&mut g, &d, "disabled");
        }
    }
}
