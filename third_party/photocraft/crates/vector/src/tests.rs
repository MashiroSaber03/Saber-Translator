use photocraft_color::{Color, PixelFormat, SampleType};
use photocraft_doc::{Fill, FillRule, LineCap, LineJoin, Path, PathOp, ShapeLayer, ShapeStroke, StrokeAlign, Subpath, VectorMask};
use photocraft_geom::Rect;

use crate::*;

fn area(v: &[f32]) -> f64 {
    v.iter().map(|x| f64::from(*x)).sum()
}

fn cov(path: &Path, rect: Rect) -> Vec<f32> {
    path_coverage(path, rect)
}

#[test]
fn circle_area_matches_analytic() {
    for r in [3.3, 10.0, 47.5, 200.25] {
        let c = 256.0 + 0.37;
        let p = shapes::ellipse(c - r, c - r, 2.0 * r, 2.0 * r);
        let got = area(&cov(&p, Rect::new(0, 0, 512, 512)));
        let want = std::f64::consts::PI * r * r;
        // The 4-arc Bézier circle overshoots the true circle by up to 0.027% of r.
        assert!((got - want).abs() / want < 0.005, "r={r}: {got} vs {want}");
        if r >= 10.0 {
            assert!((got - want).abs() / want < 0.0015, "r={r}: {got} vs {want}");
        }
    }
}

#[test]
fn rect_area_exact_and_rotated() {
    let p = shapes::rect(10.3, 20.7, 100.45, 50.2);
    let got = area(&cov(&p, Rect::new(0, 0, 200, 100)));
    assert!((got - 100.45 * 50.2).abs() < 1e-2, "{got}");
    // Rotated square: area preserved.
    let rot = p.transform(&photocraft_geom::Affine::rotate(0.4).then(&photocraft_geom::Affine::translate(60.0, 30.0)));
    let got = area(&cov(&rot, Rect::new(-100, -100, 400, 400)));
    assert!((got - 100.45 * 50.2).abs() / (100.45 * 50.2) < 1e-4, "{got}");
}

#[test]
fn coverage_is_tile_independent() {
    let p = shapes::polygon(3.0, 5.0, 90.0, 80.0, 7, 0.45);
    let full = cov(&p, Rect::new(0, 0, 100, 100));
    for (x0, y0) in [(0, 0), (13, 29), (50, 50)] {
        let r = Rect::new(x0, y0, x0 + 37, y0 + 23);
        let part = cov(&p, r);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let a = full[(y * 100 + x) as usize];
                let b = part[((y - y0) * 37 + (x - x0)) as usize];
                assert!((a - b).abs() < 1e-5);
            }
        }
    }
}

/// Pentagram drawn as one self-intersecting subpath.
fn pentagram() -> Path {
    let pts: Vec<(f64, f64)> = (0..5)
        .map(|i| {
            let a = -std::f64::consts::FRAC_PI_2 + std::f64::consts::TAU * (i * 2 % 5) as f64 / 5.0;
            (50.0 + 40.0 * a.cos(), 50.0 + 40.0 * a.sin())
        })
        .collect();
    Path::new(vec![Subpath::polygon(&pts)])
}

#[test]
fn even_odd_vs_non_zero() {
    let r = Rect::new(0, 0, 100, 100);
    let nz = cov(&pentagram(), r);
    let mut eo_path = pentagram();
    eo_path.fill_rule = FillRule::EvenOdd;
    let eo = cov(&eo_path, r);
    // Centre: filled with non-zero (winding 2), empty with even-odd.
    assert_eq!(nz[50 * 100 + 50], 1.0);
    assert_eq!(eo[50 * 100 + 50], 0.0);
    // A star tip is filled with both.
    assert_eq!(nz[16 * 100 + 50], 1.0);
    assert_eq!(eo[16 * 100 + 50], 1.0);
    assert!(area(&nz) > area(&eo) + 500.0);
    // Non-zero area − even-odd area = the inner pentagon (circumradius R·cos72°/cos36°).
    let r_in = 40.0 * 72f64.to_radians().cos() / 36f64.to_radians().cos();
    let inner_pent = 2.5 * r_in * r_in * (72f64.to_radians()).sin();
    assert!(((area(&nz) - area(&eo)) - inner_pent).abs() / inner_pent < 0.01, "{} vs {inner_pent}", area(&nz) - area(&eo));
}

fn two_squares(op: PathOp) -> Path {
    Path::new(vec![shapes::rect(10.0, 10.0, 40.0, 40.0).subpaths.remove(0), shapes::rect(30.0, 30.0, 40.0, 40.0).subpaths.remove(0).with_op(op)])
}

#[test]
fn boolean_ops_areas() {
    let r = Rect::new(0, 0, 100, 100);
    let a = |op| area(&cov(&two_squares(op), r));
    assert!((a(PathOp::Combine) - (1600.0 * 2.0 - 400.0)).abs() < 1e-2);
    assert!((a(PathOp::Subtract) - 1200.0).abs() < 1e-2);
    assert!((a(PathOp::Intersect) - 400.0).abs() < 1e-2);
    assert!((a(PathOp::Exclude) - 2400.0).abs() < 1e-2);
    // Order matters: the first subpath is always combined.
    let mut p = two_squares(PathOp::Combine);
    p.subpaths[0].op = PathOp::Subtract;
    assert!((area(&cov(&p, r)) - 2800.0).abs() < 1e-2);
    // Inversion.
    let mut inv = two_squares(PathOp::Intersect);
    inv.inverted = true;
    assert!((area(&cov(&inv, r)) - (10000.0 - 400.0)).abs() < 1e-2);
    // Donut: circle minus a smaller circle (exact booleans, even with overlapping AA edges).
    let mut d = shapes::ellipse(10.0, 10.0, 80.0, 80.0);
    d.subpaths.push(shapes::ellipse(30.0, 30.0, 40.0, 40.0).subpaths.remove(0).with_op(PathOp::Subtract));
    let want = std::f64::consts::PI * (1600.0 - 400.0);
    assert!((area(&cov(&d, r)) - want).abs() / want < 0.002);
}

fn stroke_area(path: &Path, st: &StrokeStyle) -> f64 {
    let r = stroke_rasterizer(path, st, DEFAULT_TOLERANCE);
    area(&r.render(Rect::new(-50, -50, 250, 250)))
}

#[test]
fn stroke_widths_caps_and_joins() {
    let line = Path::new(vec![Subpath::polyline(&[(20.0, 50.0), (120.0, 50.0)])]);
    let st = |w: f64, cap| StrokeStyle { width: w, cap, ..Default::default() };
    // Butt: exactly length × width.
    for w in [1.0, 2.5, 10.0] {
        assert!((stroke_area(&line, &st(w, LineCap::Butt)) - 100.0 * w).abs() < 1e-2, "w={w}");
    }
    // Square adds w/2 at both ends; round adds a disc.
    assert!((stroke_area(&line, &st(10.0, LineCap::Square)) - 110.0 * 10.0).abs() < 1e-2);
    let round = stroke_area(&line, &st(10.0, LineCap::Round));
    assert!((round - (1000.0 + std::f64::consts::PI * 25.0)).abs() < 0.5, "{round}");
    // Right-angle polyline: miter fills the full corner square, bevel half, round a quarter disc.
    let l = Path::new(vec![Subpath::polyline(&[(20.0, 20.0), (120.0, 20.0), (120.0, 120.0)])]);
    let j = |join| stroke_area(&l, &StrokeStyle { width: 10.0, join, ..Default::default() });
    let base = 2.0 * 100.0 * 10.0;
    let miter = j(LineJoin::Miter);
    let bevel = j(LineJoin::Bevel);
    let roundj = j(LineJoin::Round);
    // Two 100×10 bars share the inner 5×5 quarter; the outer 5×5 corner is the join.
    assert!((miter - (base - 25.0 + 25.0)).abs() < 1e-2, "{miter}");
    assert!((bevel - (base - 25.0 + 12.5)).abs() < 1e-2, "{bevel}");
    assert!((roundj - (base - 25.0 + std::f64::consts::PI * 25.0 / 4.0)).abs() < 0.1, "{roundj}");
    // Miter limit: a very sharp angle falls back to bevel.
    let sharp = Path::new(vec![Subpath::polyline(&[(0.0, 0.0), (100.0, 5.0), (0.0, 10.0)])]);
    let lim = |m| stroke_area(&sharp, &StrokeStyle { width: 4.0, join: LineJoin::Miter, miter_limit: m, ..Default::default() });
    assert!(lim(100.0) > lim(4.0) + 10.0);
}

#[test]
fn closed_rect_stroke_and_alignment() {
    let sq = shapes::rect(20.0, 20.0, 100.0, 100.0);
    let st = StrokeStyle { width: 10.0, ..Default::default() };
    // Centred miter stroke of a closed square: outer 110² minus inner 90².
    assert!((stroke_area(&sq, &st) - (110.0f64.powi(2) - 90.0f64.powi(2))).abs() < 1e-2);
    let shape = |align| ShapeLayer {
        path: sq.clone(),
        fill: None,
        stroke: Some(ShapeStroke { width: 10.0, paint: Fill::Solid(Color::BLACK), align, ..Default::default() }),
        ..Default::default()
    };
    let alpha_area = |sh: &ShapeLayer| {
        let rgba = CompiledShape::new(sh, DEFAULT_TOLERANCE, Rect::new(0, 0, 150, 150)).render_rgba(Rect::new(0, 0, 150, 150));
        rgba.iter().map(|p| f64::from(p[3])).sum::<f64>()
    };
    assert!((alpha_area(&shape(StrokeAlign::Inside)) - (100.0f64.powi(2) - 80.0f64.powi(2))).abs() < 0.05);
    assert!((alpha_area(&shape(StrokeAlign::Outside)) - (120.0f64.powi(2) - 100.0f64.powi(2))).abs() < 0.05);
    assert!((alpha_area(&shape(StrokeAlign::Center)) - (110.0f64.powi(2) - 90.0f64.powi(2))).abs() < 0.05);
}

#[test]
fn aligned_dashes_keep_width_units_and_open_paths_stay_centered() {
    let clip = Rect::new(0, 0, 100, 100);
    let square = shapes::rect(10.0, 10.0, 80.0, 80.0);
    for (align, y, other_y) in [(StrokeAlign::Inside, 11, 7), (StrokeAlign::Center, 9, 6), (StrokeAlign::Outside, 7, 11)] {
        let sh = ShapeLayer {
            path: square.clone(),
            stroke: Some(ShapeStroke { width: 4.0, align, dashes: vec![2.0, 2.0], ..Default::default() }),
            ..Default::default()
        };
        let rgba = CompiledShape::new(&sh, DEFAULT_TOLERANCE, clip).render_rgba(clip);
        assert!(rgba[(y * 100 + 13) as usize][3] > 0.99);
        assert!(rgba[(y * 100 + 21) as usize][3] < 0.01);
        assert!(rgba[(other_y * 100 + 13) as usize][3] < 0.01);
    }
    let open = Path::new(vec![Subpath::polyline(&[(10.0, 50.0), (90.0, 50.0)])]);
    let render = |align| {
        let sh =
            ShapeLayer { path: open.clone(), stroke: Some(ShapeStroke { width: 4.0, align, cap: LineCap::Round, ..Default::default() }), ..Default::default() };
        CompiledShape::new(&sh, DEFAULT_TOLERANCE, clip).render_rgba(clip)
    };
    assert_eq!(render(StrokeAlign::Center), render(StrokeAlign::Inside));
    assert_eq!(render(StrokeAlign::Center), render(StrokeAlign::Outside));
}

#[test]
fn dashes_cover_expected_fraction() {
    let line = Path::new(vec![Subpath::polyline(&[(0.0, 50.0), (120.0, 50.0)])]);
    let st = StrokeStyle { width: 4.0, dashes: vec![8.0, 4.0], ..Default::default() };
    // 120 px = 10 periods of 12 px, 8 on each.
    assert!((stroke_area(&line, &st) - 80.0 * 4.0).abs() < 1e-2);
    // Photoshop-style dash units (multiples of width) via stroke_style.
    let s = stroke_style(&ShapeStroke { width: 4.0, dashes: vec![2.0, 1.0], ..Default::default() });
    assert_eq!(s.dashes, vec![8.0, 4.0]);
    // Round caps on zero-length dashes draw dots.
    let dots = StrokeStyle { width: 4.0, cap: LineCap::Round, dashes: vec![0.0, 12.0], ..Default::default() };
    let a = stroke_area(&line, &dots);
    let one = std::f64::consts::PI * 4.0;
    assert!(a > 9.0 * one && a < 11.5 * one, "{a}");
}

#[test]
fn render_shape_formats_and_paint() {
    let sh = ShapeLayer {
        path: shapes::rect(2.0, 2.0, 10.0, 10.0),
        fill: Some(Fill::Solid(Color::rgb(1.0, 0.0, 0.0))),
        stroke: Some(ShapeStroke { width: 2.0, paint: Fill::Solid(Color::rgb(0.0, 0.0, 1.0)), ..Default::default() }),
        ..Default::default()
    };
    for sample in SampleType::ALL {
        for mode in [photocraft_color::ColorMode::Rgb, photocraft_color::ColorMode::Grayscale, photocraft_color::ColorMode::Cmyk] {
            let fmt = PixelFormat::new(mode, sample, true);
            let s = render_shape(&sh, fmt, Rect::new(0, 0, 20, 20));
            assert_eq!(s.format(), fmt);
            assert_eq!(s.content_bounds(), Rect::new(1, 1, 13, 13));
            let centre = photocraft_raster::to_rgba(&fmt, &s.pixel(7, 7));
            let edge = photocraft_raster::to_rgba(&fmt, &s.pixel(2, 7));
            assert!((centre[3] - 1.0).abs() < 1e-3);
            if mode == photocraft_color::ColorMode::Rgb {
                assert!(centre[0] > 0.99 && centre[2] < 0.01);
                assert!(edge[2] > 0.99, "{edge:?}");
            }
        }
    }
    // Clip: nothing outside.
    let s = render_shape(&sh, PixelFormat::RGBA8, Rect::new(0, 0, 5, 5));
    assert_eq!(s.content_bounds(), Rect::new(1, 1, 5, 5));
}

#[test]
fn gradient_fill_spans_bounds() {
    let sh = ShapeLayer {
        path: shapes::rect(0.0, 0.0, 100.0, 10.0),
        fill: Some(Fill::gradient(vec![(0.0, Color::BLACK), (1.0, Color::WHITE)], 0.0, 1.0, photocraft_doc::GradientStyle::Linear, false)),
        ..Default::default()
    };
    let rgba = CompiledShape::new(&sh, DEFAULT_TOLERANCE, Rect::new(0, 0, 100, 10)).render_rgba(Rect::new(0, 0, 100, 10));
    assert!(rgba[0][0] < 0.02 && rgba[99][0] > 0.98 && (rgba[50][0] - 0.505).abs() < 0.02);
}

/// A gradient fill with every non-default field, rendered through `Paint`: the expectations
/// below are hand-derived (the compositor's own rendering is checked against them by the
/// parity tests in `photocraft-compose`).
fn grad(fill: Fill, canvas: Rect) -> Vec<[f32; 4]> {
    let sh = ShapeLayer { path: shapes::rect(0.0, 0.0, 100.0, 10.0), fill: Some(fill), ..Default::default() };
    let c = Rect::new(0, 0, canvas.x1.min(100), canvas.y1.min(10));
    CompiledShape::new(&sh, DEFAULT_TOLERANCE, canvas).render_rgba(c)
}

#[test]
fn gradient_midpoints_opacity_stops_offset_and_sorted_stops() {
    // Linear 0° over the path's 100 × 10 bounds: t is the pixel's left corner / 100, so pixel
    // x sits at t = x / 100.
    let g = |midpoints: Vec<f32>, opacity_stops: Vec<(f32, f32)>, offset: (f32, f32), stops: Vec<(f32, Color)>| Fill::Gradient {
        stops,
        angle: 0.0,
        scale: 1.0,
        style: photocraft_doc::GradientStyle::Linear,
        reverse: false,
        opacity_stops,
        midpoints,
        offset,
        dither: false,
        align: true,
    };
    // Midpoint 0.25: the colour is half-way at t = 0.25 (pixel 25).
    let px = grad(g(vec![0.25], vec![], (0.0, 0.0), vec![(0.0, Color::BLACK), (1.0, Color::WHITE)]), Rect::new(0, 0, 100, 10));
    assert!((px[25][0] - 0.5).abs() < 1e-3, "{}", px[25][0]);
    // Opacity stop ramp 1 → 0.5: alpha at t = 0.25 is 0.875.
    let px = grad(g(vec![], vec![(0.0, 1.0), (1.0, 0.5)], (0.0, 0.0), vec![(0.0, Color::BLACK), (1.0, Color::WHITE)]), Rect::new(0, 0, 100, 10));
    assert!((px[25][3] - 0.875).abs() < 1e-3, "{}", px[25][3]);
    // Centre offset half a frame right: t = 0 moves to x = 50, so pixel 75 sits at t = 0.25.
    let px = grad(g(vec![], vec![], (0.5, 0.0), vec![(0.0, Color::BLACK), (1.0, Color::WHITE)]), Rect::new(0, 0, 100, 10));
    assert!(px[25][0] < 0.01 && (px[75][0] - 0.25).abs() < 1e-3, "{} {}", px[25][0], px[75][0]);
    // Unsorted stops are sorted by location: black still left, white right.
    let px = grad(g(vec![], vec![], (0.0, 0.0), vec![(1.0, Color::WHITE), (0.0, Color::BLACK)]), Rect::new(0, 0, 100, 10));
    assert!(px[1][0] < 0.02 && px[98][0] > 0.97, "{} {}", px[1][0], px[98][0]);
}

#[test]
fn gradient_without_align_with_layer_uses_the_canvas_frame() {
    let canvas = Rect::new(0, 0, 200, 10);
    let g = |align: bool| Fill::Gradient {
        stops: vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
        angle: 0.0,
        scale: 1.0,
        style: photocraft_doc::GradientStyle::Linear,
        reverse: false,
        opacity_stops: vec![],
        midpoints: vec![],
        offset: (0.0, 0.0),
        dither: false,
        align,
    };
    // Aligned: the 100-px path bounds are the frame, pixel 55 sits at t = 0.55.
    let px = grad(g(true), canvas);
    assert!((px[55][0] - 0.55).abs() < 1e-3, "{}", px[55][0]);
    // Not aligned: the 200-px canvas is the frame, the same pixel sits at t = 0.275.
    let px = grad(g(false), canvas);
    assert!((px[55][0] - 0.275).abs() < 1e-3, "{}", px[55][0]);
}

#[test]
fn gradient_dither_is_deterministic_and_one_level() {
    let g = |dither: bool| Fill::Gradient {
        stops: vec![(0.0, Color::BLACK), (1.0, Color::WHITE)],
        angle: 30.0,
        scale: 1.3,
        style: photocraft_doc::GradientStyle::Radial,
        reverse: false,
        opacity_stops: vec![],
        midpoints: vec![],
        offset: (0.1, -0.1),
        dither,
        align: true,
    };
    let a = grad(g(true), Rect::new(0, 0, 100, 10));
    let b = grad(g(true), Rect::new(0, 0, 100, 10));
    assert_eq!(a, b, "dither must be deterministic");
    let plain = grad(g(false), Rect::new(0, 0, 100, 10));
    let mut max = 0.0f32;
    let mut changed = 0;
    for (p, q) in a.iter().zip(&plain) {
        max = max.max((p[0] - q[0]).abs());
        changed += usize::from((p[0] - q[0]).abs() > 1e-6);
    }
    assert!(changed > 0 && max <= 1.0 / 255.0 + 1e-6, "changed={changed} max={max}");
}

#[test]
fn coverage_surface_and_vector_mask_values() {
    let p = shapes::rect(10.0, 10.0, 20.0, 20.0);
    let r = fill_rasterizer(&p, DEFAULT_TOLERANCE);
    let s = coverage_surface(&r, PixelFormat::GRAY8, Rect::new(0, 0, 300, 300));
    assert_eq!(s.content_bounds(), Rect::new(10, 10, 30, 30));
    assert_eq!(s.tile_count(), 1);
    let mut inv = p.clone();
    inv.inverted = true;
    let si = coverage_surface(&fill_rasterizer(&inv, DEFAULT_TOLERANCE), PixelFormat::GRAY8, Rect::new(0, 0, 40, 40));
    assert_eq!(si.sample_channel(0, 0, 0), 1.0);
    assert_eq!(si.sample_channel(15, 15, 0), 0.0);
    let mut vm = VectorMask::new(p);
    vm.density = 0.25;
    let v = vector_mask_values(&vm, Rect::new(0, 0, 40, 40));
    assert!((v[0] - 0.75).abs() < 1e-6 && (v[15 * 40 + 15] - 1.0).abs() < 1e-6);
    vm.enabled = false;
    assert!(vector_mask_values(&vm, Rect::new(0, 0, 4, 4)).iter().all(|x| *x == 1.0));
    // Empty vector masks reveal all (hide all when inverted).
    let mut empty = VectorMask::new(Path::default());
    assert!(vector_mask_values(&empty, Rect::new(0, 0, 4, 4)).iter().all(|x| *x == 1.0));
    empty.path.inverted = true;
    assert!(vector_mask_values(&empty, Rect::new(0, 0, 4, 4)).iter().all(|x| *x == 0.0));
}

#[test]
fn sequential_vector_masks_match_parallel_rendering() {
    let rect = Rect::new(-3, -7, 510, 250);
    let mut path = shapes::ellipse(10.25, 8.75, 470.0, 220.0);
    path.subpaths.push(shapes::rect(200.5, 90.25, 45.0, 50.0).subpaths.remove(0).with_op(PathOp::Subtract));
    for inverted in [false, true] {
        for density in [0.0, 0.25, 1.0] {
            path.inverted = inverted;
            let mut mask = VectorMask::new(path.clone());
            mask.density = density;
            let compiled = CompiledVectorMask::new(&mask);
            assert_eq!(compiled.render_sequential(rect), compiled.render(rect));
        }
    }
    let compiled = CompiledVectorMask::new(&VectorMask::new(path.clone()));
    for empty in [Rect::EMPTY, Rect::new(1, 1, 1, 10), Rect::new(1, 1, 10, 1)] {
        assert_eq!(compiled.render_sequential(empty), compiled.render(empty));
    }
    let mut empty = VectorMask::new(Path::default());
    for inverted in [false, true] {
        empty.path.inverted = inverted;
        let compiled = CompiledVectorMask::new(&empty);
        assert_eq!(compiled.render_sequential(rect), compiled.render(rect));
    }
    let mut disabled = VectorMask::new(path);
    disabled.enabled = false;
    let compiled = CompiledVectorMask::new(&disabled);
    assert_eq!(compiled.render_sequential(rect), compiled.render(rect));
    assert_eq!(compiled.render_sequential(Rect::EMPTY), compiled.render(Rect::EMPTY));
}

fn iou(a: &[f32], b: &[f32]) -> f64 {
    let (mut i, mut u) = (0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        i += f64::from(x.min(*y));
        u += f64::from(x.max(*y));
    }
    i / u.max(1e-9)
}

#[test]
fn trace_roundtrip_iou() {
    let rect = Rect::new(0, 0, 160, 120);
    // Antialiased donut + a hard rectangle, as a selection would hold.
    let mut src = shapes::ellipse(10.0, 10.0, 90.0, 90.0);
    src.subpaths.push(shapes::ellipse(35.0, 35.0, 40.0, 40.0).subpaths.remove(0).with_op(PathOp::Subtract));
    src.subpaths.push(shapes::rect(110.0, 20.0, 40.0, 80.0).subpaths.remove(0));
    let mask = cov(&src, rect);
    for tol in [0.5, 1.0, 2.0] {
        let traced = trace::trace_mask(&mask, rect, 0.0, tol);
        assert_eq!(traced.subpaths.len(), 3, "tol {tol}");
        let back = cov(&traced, rect);
        let v = iou(&mask, &back);
        assert!(v > 0.98, "tol {tol}: IoU {v}");
        let knots: usize = traced.subpaths.iter().map(|s| s.knots.len()).sum();
        assert!(knots < 60, "tol {tol}: {knots} knots");
    }
    // Hard-edged (binary) mask.
    let hard: Vec<f32> = mask.iter().map(|v| if *v >= 0.5 { 1.0 } else { 0.0 }).collect();
    let back = cov(&trace::trace_mask(&hard, rect, 0.0, 1.0), rect);
    assert!(iou(&hard, &back) > 0.98);
}

/// Performance on a 6016² canvas (run with `--release --ignored --nocapture`).
#[test]
#[ignore]
fn perf_6016() {
    let canvas = Rect::new(0, 0, 6016, 6016);
    // A complex path: 400-point star spiral with curves (Bézier knots).
    let mut knots = Vec::new();
    let n = 2000;
    for i in 0..n {
        let t = i as f64 / n as f64 * std::f64::consts::TAU;
        let r = 2400.0 + 500.0 * (t * 37.0).sin() + 200.0 * (t * 101.0).cos();
        let (x, y) = (3008.0 + r * t.cos(), 3008.0 + r * t.sin());
        let d = 6.0;
        knots.push(photocraft_doc::Knot::smooth(
            photocraft_geom::Point::new(x, y),
            photocraft_geom::Point::new(x - d * t.sin(), y + d * t.cos()),
            photocraft_geom::Point::new(x + d * t.sin(), y - d * t.cos()),
        ));
    }
    let mut path = Path::new(vec![Subpath { closed: true, knots, op: PathOp::Combine }]);
    path.subpaths.push(shapes::ellipse(2000.0, 2000.0, 2016.0, 2016.0).subpaths.remove(0).with_op(PathOp::Exclude));
    let sh = ShapeLayer {
        path: path.clone(),
        fill: Some(Fill::Solid(Color::rgb(0.2, 0.5, 0.9))),
        stroke: Some(ShapeStroke { width: 12.0, join: LineJoin::Round, dashes: vec![3.0, 2.0], ..Default::default() }),
        ..Default::default()
    };
    let t = std::time::Instant::now();
    let r = fill_rasterizer(&path, DEFAULT_TOLERANCE);
    let c = r.render(canvas);
    let fill_ms = t.elapsed().as_secs_f64() * 1000.0;
    let st = stroke_style(sh.stroke.as_ref().unwrap());
    let t = std::time::Instant::now();
    let lines = flatten_path(&path, 0.04);
    let polys = stroke_polygons(&lines, &st, 0.01);
    eprintln!(
        "flatten+pieces {:.0} ms: {} polylines {} pts, {} pieces {} verts",
        t.elapsed().as_secs_f64() * 1000.0,
        lines.len(),
        lines.iter().map(|l| l.pts.len()).sum::<usize>(),
        polys.len(),
        polys.iter().map(|p| p.len()).sum::<usize>()
    );
    let t = std::time::Instant::now();
    let sr = stroke_rasterizer(&path, &st, DEFAULT_TOLERANCE);
    let sc = sr.render(canvas);
    let stroke_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = std::time::Instant::now();
    let s = render_shape(&sh, PixelFormat::RGBA8, canvas);
    let shape_ms = t.elapsed().as_secs_f64() * 1000.0;
    let t = std::time::Instant::now();
    let s16 = render_shape(&sh, PixelFormat::RGBA16, canvas);
    let shape16_ms = t.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "6016²: fill coverage {fill_ms:.0} ms (area {:.0}), stroke coverage {stroke_ms:.0} ms (area {:.0}), shape→RGBA8 surface {shape_ms:.0} ms ({} tiles), RGBA16 {shape16_ms:.0} ms",
        area(&c),
        area(&sc),
        s.tile_count()
    );
    assert!(s16.tile_count() > 0);
}

#[test]
fn joined_subpaths_fill_as_one_component() {
    // An outer square and an inner one wound the other way: as one component (PSD operation
    // -1 after the first record) the inner one is a hole; as separate "combine" shapes it isn't.
    let outer = Subpath::polygon(&[(2.0, 2.0), (18.0, 2.0), (18.0, 18.0), (2.0, 18.0)]);
    let inner = Subpath::polygon(&[(6.0, 6.0), (6.0, 14.0), (14.0, 14.0), (14.0, 6.0)]);
    let r = Rect::new(0, 0, 20, 20);
    let joined = Path::new(vec![outer.clone(), inner.clone().with_op(PathOp::Join)]);
    let v = cov(&joined, r);
    assert_eq!(v[10 * 20 + 10], 0.0, "hole");
    assert_eq!(v[4 * 20 + 4], 1.0);
    assert!((area(&v) - (256.0 - 64.0)).abs() < 1e-3);
    let separate = Path::new(vec![outer.clone(), inner.clone()]);
    assert_eq!(cov(&separate, r)[10 * 20 + 10], 1.0);
    assert_eq!(joined.components(), vec![0..2]);
    assert_eq!(separate.components(), vec![0..1, 1..2]);
    // A component after a joined one starts afresh with its own operation.
    let cut = Subpath::polygon(&[(0.0, 0.0), (20.0, 0.0), (20.0, 4.0), (0.0, 4.0)]).with_op(PathOp::Subtract);
    let p = Path::new(vec![outer, inner.with_op(PathOp::Join), cut]);
    assert_eq!(p.components(), vec![0..2, 2..3]);
    let v = cov(&p, r);
    assert_eq!(v[3 * 20 + 10], 0.0);
    assert_eq!(v[5 * 20 + 10], 1.0);
    // A lone joined subpath acts as the first component.
    let lone = Path::new(vec![Subpath::polygon(&[(2.0, 2.0), (8.0, 2.0), (8.0, 8.0)]).with_op(PathOp::Join)]);
    assert_eq!(cov(&lone, r)[3 * 20 + 6], 1.0);
}

/// Content bounds of a filled shape rendered as a layer (every depth) and as a selection
/// (every depth); they must all equal `want`.
fn assert_rendered_bounds(path: &Path, want: Rect, what: &str) {
    let canvas = Rect::new(0, 0, 1920, 1080);
    let sh = ShapeLayer { path: path.clone(), fill: Some(Fill::Solid(Color::rgb(0.0, 0.0, 1.0))), ..Default::default() };
    for sample in SampleType::ALL {
        let fmt = PixelFormat::new(photocraft_color::ColorMode::Rgb, sample, true);
        assert_eq!(render_shape(&sh, fmt, canvas).content_bounds(), want, "{what}: layer {sample:?}");
        let sel = coverage_surface(&fill_rasterizer(path, DEFAULT_TOLERANCE), PixelFormat::new(photocraft_color::ColorMode::Grayscale, sample, false), canvas);
        assert_eq!(sel.content_bounds(), want, "{what}: selection {sample:?}");
    }
}

#[test]
fn curved_shape_bounds_are_exact() {
    // Issue #2537: f32 round-off left ~1e-7 coverage just right of curved edges, so curved
    // shapes reported bounds one pixel too wide. Rectangles never showed it.
    let p = |x: f64, y: f64| photocraft_geom::Point::new(x, y);
    // An asymmetric Bézier blob whose extremes lie on its anchors (axis-parallel handles).
    let blob = Path::new(vec![Subpath {
        closed: true,
        knots: vec![
            photocraft_doc::Knot::smooth(p(60.0, 0.0), p(30.0, 0.0), p(80.0, 0.0)),
            photocraft_doc::Knot::smooth(p(100.0, 45.0), p(100.0, 20.0), p(100.0, 70.0)),
            photocraft_doc::Knot::smooth(p(40.0, 100.0), p(75.0, 100.0), p(15.0, 100.0)),
            photocraft_doc::Knot::smooth(p(0.0, 55.0), p(0.0, 85.0), p(0.0, 25.0)),
        ],
        op: Default::default(),
    }]);
    let cases = [
        ("ellipse", shapes::ellipse(0.0, 0.0, 100.0, 100.0), Rect::new(0, 0, 100, 100)),
        ("ellipse 600", shapes::ellipse(660.0, 240.0, 600.0, 600.0), Rect::new(660, 240, 1260, 840)),
        ("ellipse at x.5", shapes::ellipse(10.5, 10.0, 100.0, 100.0), Rect::new(10, 10, 111, 110)),
        ("rounded rect", shapes::rounded_rect(0.0, 0.0, 100.0, 100.0, [20.0; 4]), Rect::new(0, 0, 100, 100)),
        // Outer vertices at x = 50 ± 50·cos 18° (2.45, 97.55), lowest at y = 50 + 50·sin 54° (90.45).
        ("star", shapes::polygon(0.0, 0.0, 100.0, 100.0, 5, 0.5), Rect::new(2, 0, 98, 91)),
        ("custom path", blob, Rect::new(0, 0, 100, 100)),
        ("rect", shapes::rect(0.0, 0.0, 100.0, 100.0), Rect::new(0, 0, 100, 100)),
    ];
    for (what, path, want) in &cases {
        assert_rendered_bounds(path, *want, what);
    }
}

#[test]
fn thin_antialiased_edges_still_count_toward_bounds() {
    // A quarter-pixel overhang and a 1-px bar straddling two columns are real coverage.
    assert_rendered_bounds(&shapes::rect(0.0, 0.0, 100.25, 100.0), Rect::new(0, 0, 101, 100), "quarter overhang");
    assert_rendered_bounds(&shapes::rect(10.5, 0.0, 1.0, 50.0), Rect::new(10, 0, 12, 50), "1-px bar");
    assert_rendered_bounds(&shapes::rect(0.0, 0.0, 100.01, 10.0), Rect::new(0, 0, 101, 10), "1% sliver");
    // At 32-bit even a 1e-4 sliver is kept: only round-off is dropped.
    let r = fill_rasterizer(&shapes::rect(0.0, 0.0, 100.0001, 10.0), DEFAULT_TOLERANCE);
    let c = r.render(Rect::new(0, 0, 102, 10));
    assert!(c[100] > 5e-5 && c[101] == 0.0, "{} {}", c[100], c[101]);
}
