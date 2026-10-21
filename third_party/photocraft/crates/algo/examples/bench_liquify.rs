//! Liquify / Puppet / Perspective timings on a 24 MP image.
//!
//! ```sh
//! cargo run --release -p photocraft-algo --example bench_liquify
//! ```

use std::time::Instant;

use photocraft_algo::liquify::{LiquifyField, LiquifyStroke, LiquifyTool, ProxyImage, apply_liquify, auto_cell};
use photocraft_algo::perspective::{PerspectiveMap, Plane};
use photocraft_algo::puppet::{PuppetDensity, PuppetMode, PuppetPin, PuppetWarp, deform, render};
use photocraft_algo::transform::Interp;
use photocraft_algo::warp::warp_mesh_surface;
use photocraft_color::PixelFormat;
use photocraft_geom::Rect;
use photocraft_raster::Surface;

fn main() {
    let (w, h) = (6000, 4000);
    let b = Rect::new(0, 0, w, h);
    let mut s = Surface::new(PixelFormat::RGBA8);
    let mut row = vec![0.0f32; w as usize * 4];
    for y in 0..h {
        for x in 0..w as usize {
            let v = (((x / 37) + (y as usize / 41)) % 2) as f32;
            row[x * 4..x * 4 + 4].copy_from_slice(&[v, x as f32 / w as f32, y as f32 / h as f32, 1.0]);
        }
        s.write_region(Rect::new(0, y, w, y + 1), &row);
    }
    let cell = auto_cell(b);
    println!("24 MP RGBA8, field cell {cell} px");

    // Interactive: proxy + per-dab update.
    let t = Instant::now();
    let proxy = ProxyImage::new(&s, b, 1600);
    println!("proxy {}×{} built in {:.1} ms", proxy.w, proxy.h, t.elapsed().as_secs_f64() * 1e3);
    let mut out = vec![[0u8; 4]; proxy.w * proxy.h];
    let t = Instant::now();
    proxy.render(&LiquifyField::new(b, cell), [0, 0, proxy.w, proxy.h], &mut out);
    println!("full proxy render {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let mut field = LiquifyField::new(b, cell);
    println!("field alloc {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
    let mut st = LiquifyStroke::new(LiquifyTool::ForwardWarp, 600.0);
    let mut worst = 0.0f64;
    let mut total = 0.0;
    let n = 60;
    let mut prev = [2000.0, 2000.0, 1.0];
    field.stroke_begin(&st, prev);
    for k in 1..=n {
        let p = [2000.0 + k as f64 * 25.0, 2000.0 + (k as f64 * 0.3).sin() * 200.0, 1.0];
        let t = Instant::now();
        let dirty = field.stroke_segment(&st, prev, p);
        let r = proxy.proxy_rect(dirty);
        proxy.render(&field, r, &mut out);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        worst = worst.max(ms);
        total += ms;
        prev = p;
    }
    println!("forward-warp segment (600 px brush, 25 px moves) + proxy update: mean {:.2} ms, worst {:.2} ms", total / n as f64, worst);
    // One 3000 px drag with a 200 px brush (a straight leg, then a diagonal one), as a slow
    // (every 2 px) and a fast (every 60 px) pointer reports it: the field cost of the whole
    // stroke, and how far the two results differ.
    let mut drag = LiquifyStroke::new(LiquifyTool::ForwardWarp, 200.0);
    let mut fields = Vec::new();
    let along = |d: f64| if d <= 1500.0 { vec![1500.0 + d, 2000.0] } else { vec![3000.0 + (d - 1500.0) * 0.6, 2000.0 + (d - 1500.0) * 0.8] };
    for step in [2.0, 60.0] {
        drag.points = (0..=(3000.0 / step) as usize).map(|k| along(k as f64 * step)).collect();
        let t = Instant::now();
        let mut f = LiquifyField::new(b, cell);
        f.apply_stroke(&drag);
        println!("forward-warp 3000 px drag (200 px brush), an event every {step} px: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
        fields.push(f);
    }
    let diff = fields[0].d.iter().zip(&fields[1].d).map(|(p, q)| (p[0] - q[0]).abs().max((p[1] - q[1]).abs())).fold(0.0f32, f32::max);
    println!("  slow vs fast pointer: fields differ by at most {diff:.3} px (max displacement {:.1} px)", fields[0].max_displacement());
    st.tool = LiquifyTool::Bloat;
    let t = Instant::now();
    let mut d = Rect::EMPTY;
    for _ in 0..20 {
        d = field.stroke_segment(&st, [3000.0, 1500.0, 1.0], [3000.0, 1500.0, 1.0]);
    }
    println!("bloat dab alone: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    let t = Instant::now();
    for _ in 0..20 {
        proxy.render(&field, proxy.proxy_rect(d), &mut out);
    }
    println!("dab-sized proxy update alone: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);
    let t = Instant::now();
    for _ in 0..20 {
        let d = field.stroke_segment(&st, [3000.0, 1500.0, 1.0], [3000.0, 1500.0, 1.0]);
        proxy.render(&field, proxy.proxy_rect(d), &mut out);
    }
    println!("bloat dab (600 px) + proxy update: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 20.0);

    let t = Instant::now();
    let res = apply_liquify(&s, &field);
    println!("apply on 24 MP (touched tiles only): {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    // Worst case: every tile displaced.
    let mut all = LiquifyField::new(b, cell);
    for v in all.d.iter_mut() {
        *v = [1.3, -0.7];
    }
    let t = Instant::now();
    let _ = apply_liquify(&s, &all);
    println!("apply on 24 MP (whole field displaced): {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    drop(res);

    // Puppet warp on a 24 MP layer.
    let pw = PuppetWarp {
        pins: vec![
            PuppetPin { src: [1000.0, 2000.0], dst: [1000.0, 2000.0], rotate: None, depth: 0 },
            PuppetPin { src: [5000.0, 2000.0], dst: [4800.0, 1200.0], rotate: None, depth: 0 },
            PuppetPin { src: [3000.0, 3500.0], dst: [3100.0, 3500.0], rotate: None, depth: 0 },
        ],
        mode: PuppetMode::Normal,
        density: PuppetDensity::Normal,
        expansion: 2.0,
    };
    let t = Instant::now();
    let (solver, v, order) = deform(&s, b, &pw, photocraft_algo::puppet::ITERATIONS);
    println!(
        "puppet mesh {} verts / {} tris + ARAP ({} rounds): {:.1} ms",
        solver.mesh.verts.len(),
        solver.mesh.tris.len(),
        photocraft_algo::puppet::ITERATIONS,
        t.elapsed().as_secs_f64() * 1e3
    );
    let dst: Vec<[f64; 2]> = pw.pins.iter().map(|p| p.dst).collect();
    let t = Instant::now();
    let _ = solver.solve(&dst, &[None; 3], PuppetMode::Normal, 4, Some(&v));
    println!("puppet interactive re-solve (4 warm rounds): {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
    let t = Instant::now();
    let _ = render(&s, b, &solver.mesh, &v, &order, Interp::Bicubic);
    println!("puppet render 24 MP: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);

    let q = [[0.0, 0.0], [6000.0, 0.0], [6000.0, 4000.0], [0.0, 4000.0]];
    let m = PerspectiveMap::new(&[Plane { src: q, dst: [[400.0, 0.0], [5600.0, 200.0], [6000.0, 4000.0], [0.0, 4000.0]] }]).unwrap();
    let t = Instant::now();
    let _ = warp_mesh_surface(&s, b, &|x, y| m.map(x, y), Interp::Bicubic);
    println!("perspective warp 24 MP: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
}
