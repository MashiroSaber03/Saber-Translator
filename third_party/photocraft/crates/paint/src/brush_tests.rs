//! Brush-engine tests: tips, spacing, dynamics, masks, smoothing, determinism, formats.

use photocraft_color::{BlendMode, ColorMode, PixelFormat, SampleType};
use photocraft_geom::{Point, Rect};
use photocraft_raster::Surface;

use crate::brush::*;
use crate::dynamics::{Smoother, smooth_points};
use crate::render::{BrushContext, StrokeRenderer, mask_combine, render_stroke, texture_combine};
use crate::tile::GrayTile;
use crate::{Dab, Stroke, StrokePoint, dabs};

fn brush() -> BrushSettings {
    BrushSettings { pressure_size: false, spacing: 0.25, ..Default::default() }
}

fn line(x0: f64, x1: f64, y: f64) -> Vec<StrokePoint> {
    vec![StrokePoint::new(x0, y, 1.0), StrokePoint::new(x1, y, 1.0)]
}

fn dabs_of(b: &BrushSettings, pts: &[StrokePoint]) -> Vec<Dab> {
    dabs(&Stroke { brush: b.clone(), points: pts.to_vec() })
}

fn raster(b: &BrushSettings, d: &Dab) -> (Rect, Vec<f32>) {
    let ctx = BrushContext::new(b);
    let r = ctx.dab_rect(d, false);
    let mut v = Vec::new();
    ctx.rasterize(d, false, r, &mut v);
    (r, v)
}

fn at(r: &(Rect, Vec<f32>), x: i32, y: i32) -> f32 {
    r.1[((y - r.0.y0) * r.0.width() as i32 + (x - r.0.x0)) as usize]
}

fn paint(b: &BrushSettings, pts: &[StrokePoint], w: i32, h: i32) -> Surface {
    let mut s = Surface::new(PixelFormat::RGBA32F);
    let _ = w + h;
    render_stroke(&mut s, b, pts, None, false, 1.0);
    s
}

fn alpha_sum(s: &Surface, r: Rect) -> f32 {
    let mut t = 0.0;
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            t += s.rgba(x, y)[3];
        }
    }
    t
}

// ---------- tips ----------

#[test]
fn round_tip_roundness_and_angle() {
    let b = brush();
    let mut d = Dab::round(Point::new(50.0, 50.0), 20.0, 1.0);
    d.roundness = 0.25;
    let r = raster(&b, &d);
    // Wide along x, thin along y.
    assert!(at(&r, 50 + 15, 50) > 0.99);
    assert!(at(&r, 50, 50 + 8) < 0.01);
    assert!(at(&r, 50, 50 + 3) > 0.99);
    // Rotated 90° counter-clockwise: now tall.
    d.angle = std::f32::consts::FRAC_PI_2;
    let r = raster(&b, &d);
    assert!(at(&r, 50, 50 - 15) > 0.99);
    assert!(at(&r, 50 + 8, 50) < 0.01);
    // 45°: the long axis runs up-right (y-down image space: x+, y−).
    d.angle = std::f32::consts::FRAC_PI_4;
    let r = raster(&b, &d);
    assert!(at(&r, 60, 40) > 0.99, "{}", at(&r, 60, 40));
    assert!(at(&r, 60, 60) < 0.01);
}

#[test]
fn hardness_profile_is_monotonic() {
    for h in [0.0f32, 0.5, 1.0] {
        let b = BrushSettings { hardness: h, ..brush() };
        let r = raster(&b, &Dab::round(Point::new(50.0, 50.0), 20.0, 1.0));
        // A soft tip paints past its 20 px radius, as Photoshop's does, out to its reach.
        let n = crate::tip_reach(20.0, h).ceil() as i32 + 1;
        let prof: Vec<f32> = (0..n).map(|i| at(&r, 50 + i, 50)).collect();
        assert!(prof.windows(2).all(|w| w[1] <= w[0] + 1e-6), "h={h} {prof:?}");
        assert!(prof[0] > 0.99);
        assert!(*prof.last().unwrap() < 0.002, "h={h} {prof:?}");
        let mid = at(&r, 60, 50);
        if h == 1.0 {
            assert!(mid > 0.99);
        } else if h == 0.0 {
            assert!(mid > 0.2 && mid < 0.8, "{mid}");
        }
    }
}

#[test]
fn hard_edge_is_sharp() {
    // A hardness of 1 is a steep rim: fully opaque inside the nominal radius, empty just past
    // it, with only the anti-aliasing band in between.
    let b = BrushSettings { hardness: 1.0, ..brush() };
    let r = raster(&b, &Dab::round(Point::new(50.0, 50.0), 20.0, 1.0));
    let prof: Vec<f32> = (0..22).map(|i| at(&r, 50 + i, 50)).collect();
    let band = prof.iter().filter(|&&a| (0.01..0.99).contains(&a)).count();
    assert!(band <= 3, "hardness 1 rim is {band} px wide: {prof:?}");
    assert!(prof[17] > 0.99, "translucent inside the rim: {prof:?}");
    assert!(*prof.last().unwrap() < 0.01, "{prof:?}");
    // A soft tip spreads the same crossing over many pixels instead.
    let soft = BrushSettings { hardness: 0.0, ..brush() };
    let rs = raster(&soft, &Dab::round(Point::new(50.0, 50.0), 20.0, 1.0));
    let profs: Vec<f32> = (0..22).map(|i| at(&rs, 50 + i, 50)).collect();
    let soft_band = profs.iter().filter(|&&a| (0.01..0.99).contains(&a)).count();
    assert!(soft_band >= 10, "soft rim is only {soft_band} px: {profs:?}");
}

#[test]
fn sampled_tip_is_scaled_and_oriented() {
    // Tip: left half painted.
    let tip = GrayTile::from_fn(16, 16, |x, _| if x < 8 { 1.0 } else { 0.0 });
    let b = BrushSettings { tip: TipShape::Sampled(tip), ..brush() };
    let d = Dab::round(Point::new(50.0, 50.0), 20.0, 1.0);
    let r = raster(&b, &d);
    assert!(at(&r, 40, 50) > 0.99 && at(&r, 60, 50) < 0.01);
    // Flip X mirrors it.
    let r = raster(&b, &Dab { flip_x: true, ..d });
    assert!(at(&r, 40, 50) < 0.01 && at(&r, 60, 50) > 0.99);
    // Rotated 90° CCW: the painted (left) half moves to the bottom.
    let r = raster(&b, &Dab { angle: std::f32::consts::FRAC_PI_2, ..d });
    assert!(at(&r, 50, 60) > 0.99 && at(&r, 50, 40) < 0.01);
    // Heavily downscaled uses mips: still sensible coverage.
    let r = raster(&b, &Dab::round(Point::new(50.0, 50.0), 2.0, 1.0));
    assert!(r.1.iter().sum::<f32>() > 1.0);
}

#[test]
fn pencil_is_aliased() {
    let b = BrushSettings { aliased: true, hardness: 0.0, size: 9.0, ..brush() };
    let s = paint(&b, &line(10.0, 50.0, 20.5), 64, 40);
    for y in 10..32 {
        for x in 0..64 {
            let a = s.rgba(x, y)[3];
            assert!(a == 0.0 || a == 1.0, "({x},{y}) = {a}");
        }
    }
    assert_eq!(s.rgba(30, 20)[3], 1.0);
    // Anti-aliased brush has partial pixels.
    let s = paint(&BrushSettings { aliased: false, ..b }, &line(10.0, 50.0, 20.3), 64, 40);
    assert!((10..32).any(|y| (0..64).any(|x| (0.01..0.99).contains(&s.rgba(x, y)[3]))));
}

#[test]
fn pencil_dabs_sit_on_the_pixel_grid() {
    use crate::render::{grid_center, grid_square};
    // Odd sizes centre on the pixel holding the point, even ones on the nearest corner.
    assert_eq!(grid_center(10.0, 5.0, 1.0), (10.5, 5.5));
    assert_eq!(grid_center(10.9, 5.2, 3.0), (10.5, 5.5));
    assert_eq!(grid_center(10.4, 5.6, 2.0), (10.0, 6.0));
    assert_eq!(grid_square(10.2, 5.7, 1.0), [10.0, 5.0, 11.0, 6.0]);
    assert_eq!(grid_square(10.2, 5.7, 4.0), [8.0, 4.0, 12.0, 8.0]);
    assert_eq!(grid_square(f64::NAN, 0.0, f32::NAN).len(), 4);
    // A 1 px pencil paints exactly the pixel under each point, even on a pixel corner, with
    // no partial alpha.
    let b = BrushSettings { aliased: true, hardness: 1.0, size: 1.0, ..brush() };
    let s = paint(&b, &[StrokePoint::new(7.0, 4.0, 1.0)], 16, 16);
    for y in 0..16 {
        for x in 0..16 {
            assert_eq!(s.rgba(x, y)[3], if (x, y) == (7, 4) { 1.0 } else { 0.0 }, "({x},{y})");
        }
    }
    // A horizontal 1 px line is one pixel thick and unbroken.
    let s = paint(&b, &line(2.0, 20.0, 9.3), 24, 16);
    for x in 2..20 {
        assert_eq!(s.rgba(x, 9)[3], 1.0, "x {x}");
        assert_eq!(s.rgba(x, 8)[3] + s.rgba(x, 10)[3], 0.0, "x {x}");
    }
    // A 4 px dab stays inside its cursor square.
    let b = BrushSettings { size: 4.0, ..b };
    let s = paint(&b, &[StrokePoint::new(10.2, 5.7, 1.0)], 24, 16);
    let [x0, y0, x1, y1] = grid_square(10.2, 5.7, 4.0);
    for y in 0..16 {
        for x in 0..24 {
            let inside = (x as f64) >= x0 && (x as f64) < x1 && (y as f64) >= y0 && (y as f64) < y1;
            assert!(inside || s.rgba(x, y)[3] == 0.0, "({x},{y}) outside the square is painted");
        }
    }
    assert_eq!(s.rgba(9, 5)[3], 1.0);
}

#[test]
fn pencil_1px_line_is_a_clean_8_connected_staircase() {
    // #2139: a 1 px Pencil line is one pixel per column (or row, when steep) on the ideal line
    // between the end pixels: no doubled pixels at the steps.
    let b = BrushSettings { aliased: true, hardness: 1.0, size: 1.0, ..brush() };
    let cases = [((2.5, 2.5), (22.5, 9.5)), ((3.5, 30.5), (10.5, 5.5)), ((1.2, 1.7), (25.8, 4.1)), ((5.5, 5.5), (17.5, 17.5)), ((20.3, 3.6), (4.9, 12.2))];
    for ((ax, ay), (bx, by)) in cases {
        let s = paint(&b, &[StrokePoint::new(ax, ay, 1.0), StrokePoint::new(bx, by, 1.0)], 40, 40);
        let painted: Vec<(i32, i32)> = (0..40).flat_map(|y| (0..40).map(move |x| (x, y))).filter(|&(x, y)| s.rgba(x, y)[3] > 0.0).collect();
        let (a, e) = ((ax.floor(), ay.floor()), (bx.floor(), by.floor()));
        let steep = (e.1 - a.1).abs() > (e.0 - a.0).abs();
        let major = |p: (f64, f64)| if steep { (p.1, p.0) } else { (p.0, p.1) };
        let ((m0, n0), (m1, n1)) = (major(a), major(e));
        assert_eq!(painted.len() as f64, (m1 - m0).abs() + 1.0, "{a:?}->{e:?}: one pixel per step, got {painted:?}");
        for &(x, y) in &painted {
            let (m, n) = major((f64::from(x), f64::from(y)));
            let ideal = n0 + (n1 - n0) * (m - m0) / (m1 - m0);
            assert!((n - ideal).abs() <= 0.5 + 1e-9, "{a:?}->{e:?}: ({x},{y}) is off the line");
        }
        let columns: std::collections::BTreeSet<i64> = painted.iter().map(|&(x, y)| major((f64::from(x), f64::from(y))).0 as i64).collect();
        assert_eq!(columns.len(), painted.len(), "{a:?}->{e:?}: a column holds two pixels");
    }
}

// ---------- spacing ----------

#[test]
fn spacing_sets_dab_count() {
    for (spacing, len, expect) in [(0.25f32, 100.0, 21usize), (1.0, 100.0, 6), (0.5, 200.0, 21)] {
        let b = BrushSettings { spacing, size: 20.0, ..brush() };
        assert_eq!(dabs_of(&b, &line(0.0, len, 0.0)).len(), expect, "spacing {spacing}");
    }
    // Spacing carries across segments: a polyline gives the same count as the straight line.
    let b = BrushSettings { spacing: 0.5, size: 20.0, ..brush() };
    let pts: Vec<StrokePoint> = (0..=20).map(|i| StrokePoint::new(i as f64 * 10.0, 0.0, 1.0)).collect();
    assert_eq!(dabs_of(&b, &pts).len(), 21);
    let d = dabs_of(&b, &pts);
    assert!(d.windows(2).all(|w| ((w[1].center.x - w[0].center.x) - 10.0).abs() < 1e-6));
}

#[test]
fn stroke_has_no_seams() {
    // Overlapping dabs must fuse into one solid band: a seam is a periodic gap or a darker
    // join, so the minimum coverage inside the stroke stays at the opacity.
    for spacing in [0.1f32, 0.25, 0.5] {
        let b = BrushSettings { spacing, size: 20.0, ..brush() };
        let s = paint(&b, &line(20.0, 180.0, 100.5), 200, 200);
        for y in 98..=102 {
            for x in 40..=160 {
                let a = s.rgba(x, y)[3];
                assert!(a > 0.99, "spacing {spacing} seam at ({x},{y}): {a}");
            }
        }
    }
}

// ---------- dynamics ----------

#[test]
fn pressure_controls_size_with_minimum() {
    let b = BrushSettings {
        size: 40.0,
        shape_dynamics: ShapeDynamics {
            enabled: true,
            size: Dynamic { control: Control::PenPressure, minimum: 0.25, ..Default::default() },
            ..Default::default()
        },
        ..brush()
    };
    let pts = vec![StrokePoint::new(0.0, 0.0, 0.0), StrokePoint::new(200.0, 0.0, 1.0)];
    let d = dabs_of(&b, &pts);
    assert!((d[0].radius - 5.0).abs() < 1e-3, "min diameter 25 % of 40 → r 5, got {}", d[0].radius);
    assert!((d.last().unwrap().radius - 20.0).abs() < 0.5);
    assert!(d.windows(2).all(|w| w[1].radius >= w[0].radius - 1e-4));
}

#[test]
fn fade_ramps_over_steps() {
    let b = BrushSettings {
        size: 20.0,
        spacing: 0.5,
        shape_dynamics: ShapeDynamics {
            enabled: true,
            size: Dynamic { control: Control::Fade, fade_steps: 10, minimum: 0.0, ..Default::default() },
            ..Default::default()
        },
        ..brush()
    };
    let d = dabs_of(&b, &line(0.0, 300.0, 0.0));
    assert!((d[0].radius - 10.0).abs() < 1e-3);
    assert!(d[5].radius < d[0].radius * 0.6, "{}", d[5].radius);
    assert!(d.iter().skip(10).all(|x| x.radius <= 0.5 + 1e-6));
    // Fade on opacity (Transfer).
    let b = BrushSettings {
        transfer: Transfer { enabled: true, opacity: Dynamic { control: Control::Fade, fade_steps: 4, ..Default::default() }, ..Default::default() },
        ..brush()
    };
    let d = dabs_of(&b, &line(0.0, 100.0, 0.0));
    assert_eq!(d[0].opacity, 1.0);
    assert_eq!(d[4].opacity, 0.0);
}

#[test]
fn tilt_direction_and_rotation_drive_angle() {
    let mk = |c: Control| BrushSettings { shape_dynamics: ShapeDynamics { enabled: true, angle: Dynamic::controlled(c), ..Default::default() }, ..brush() };
    let mut p = StrokePoint::new(0.0, 0.0, 1.0);
    p.tilt_x = 0.0;
    p.tilt_y = -45.0; // pen leaning "up" the screen
    let pts = vec![p, StrokePoint { x: 50.0, ..p }];
    let d = dabs_of(&mk(Control::PenTilt), &pts);
    assert!((d[0].angle.to_degrees() - 90.0).abs() < 1e-3, "{}", d[0].angle.to_degrees());
    // Direction: travelling down the screen = -90° (counter-clockwise positive).
    let pts = vec![StrokePoint::new(0.0, 0.0, 1.0), StrokePoint::new(0.0, 50.0, 1.0)];
    let d = dabs_of(&mk(Control::Direction), &pts);
    assert!(d.iter().all(|x| (x.angle.to_degrees() + 90.0).abs() < 1e-3));
    // Rotation.
    let mut q = StrokePoint::new(0.0, 0.0, 1.0);
    q.rotation = 30.0;
    let d = dabs_of(&mk(Control::Rotation), &[q, StrokePoint { x: 20.0, ..q }]);
    assert!((d[0].angle.to_degrees() - 30.0).abs() < 1e-3);
    // Pen tilt on size: upright pen full size, flat pen minimum.
    let b = BrushSettings {
        shape_dynamics: ShapeDynamics { enabled: true, size: Dynamic::controlled(Control::PenTilt), ..Default::default() },
        size: 20.0,
        ..brush()
    };
    let mut flat = StrokePoint::new(0.0, 0.0, 1.0);
    flat.tilt_x = 90.0;
    let d = dabs_of(&b, &[flat]);
    assert!(d[0].radius <= 0.5 + 1e-6);
}

#[test]
fn jitter_is_deterministic_per_seed() {
    let b = BrushSettings {
        size: 30.0,
        seed: 42,
        shape_dynamics: ShapeDynamics {
            enabled: true,
            size: Dynamic::jitter(1.0),
            angle: Dynamic::jitter(1.0),
            roundness: Dynamic { jitter: 1.0, minimum: 0.2, ..Default::default() },
            flip_x_jitter: true,
            flip_y_jitter: true,
            ..Default::default()
        },
        scattering: Scattering { enabled: true, scatter: Dynamic::jitter(2.0), both_axes: true, count: 3, count_jitter: Dynamic::jitter(1.0) },
        ..brush()
    };
    let pts = line(0.0, 300.0, 50.0);
    let a = dabs_of(&b, &pts);
    assert_eq!(a, dabs_of(&b, &pts));
    let c = dabs_of(&BrushSettings { seed: 43, ..b.clone() }, &pts);
    assert_ne!(a, c);
    // Jitter actually varies things, within their ranges.
    let radii: Vec<f32> = a.iter().map(|d| d.radius).collect();
    assert!(radii.iter().any(|&r| r < 10.0) && radii.iter().all(|&r| r <= 15.0 + 1e-4));
    assert!(a.iter().all(|d| d.roundness >= 0.2 - 1e-6 && d.roundness <= 1.0));
    assert!(a.iter().any(|d| d.flip_x) && a.iter().any(|d| !d.flip_x));
    // Chunked feeding gives identical dabs.
    let mut r = StrokeRenderer::new(&b, None, 1.0).record_dabs();
    let many: Vec<StrokePoint> = (0..=30).map(|i| StrokePoint::new(i as f64 * 10.0, 50.0, 1.0)).collect();
    for ch in many.chunks(4) {
        r.push(ch);
    }
    r.finish();
    assert_eq!(r.dabs(), dabs_of(&b, &many).as_slice());
}

#[test]
fn scattering_stays_in_bounds() {
    for both in [false, true] {
        let b = BrushSettings {
            size: 20.0,
            scattering: Scattering { enabled: true, scatter: Dynamic::jitter(2.0), both_axes: both, count: 4, ..Default::default() },
            ..brush()
        };
        let d = dabs_of(&b, &line(0.0, 400.0, 100.0));
        // count 4 per spacing step (5 px) → 81 steps.
        assert_eq!(d.len(), 81 * 4);
        let max_off = 2.0 * 10.0;
        assert!(d.iter().all(|x| (x.center.y - 100.0).abs() <= max_off + 1e-6));
        assert!(d.iter().any(|x| (x.center.y - 100.0).abs() > max_off * 0.5), "scatter too small");
        if !both {
            // Perpendicular only: x stays on the spacing grid.
            assert!(d.iter().all(|x| (x.center.x / 5.0 - (x.center.x / 5.0).round()).abs() < 1e-6));
        } else {
            assert!(d.iter().any(|x| (x.center.x / 5.0 - (x.center.x / 5.0).round()).abs() > 0.1));
        }
    }
}

#[test]
fn zero_scatter_keeps_dabs_on_the_stroke() {
    // Scatter at 0 % must not move a dab, even with the section switched on.
    let on = BrushSettings { scattering: Scattering { enabled: true, ..Default::default() }, ..brush() };
    for b in [brush(), on] {
        let d = dabs_of(&b, &line(10.0, 90.0, 40.5));
        assert!(d.len() > 4);
        assert!(
            d.iter().all(|x| (x.center.y - 40.5).abs() < 1e-9),
            "scatter moved dabs off the stroke: {:?}",
            d.iter().map(|x| x.center.y).collect::<Vec<_>>()
        );
    }
}

#[test]
fn color_dynamics_ranges() {
    let base = BrushSettings { color: [1.0, 0.0, 0.0, 1.0], background: [0.0, 0.0, 1.0, 1.0], seed: 5, ..brush() };
    // FG/BG jitter mixes between the two colours only.
    let b = BrushSettings { color_dynamics: ColorDynamics { enabled: true, fg_bg: Dynamic::jitter(1.0), ..Default::default() }, ..base.clone() };
    let d = dabs_of(&b, &line(0.0, 200.0, 0.0));
    assert!(d.iter().all(|x| x.color[1].abs() < 1e-5 && (x.color[0] + x.color[2] - 1.0).abs() < 1e-3 || x.color[0].max(x.color[2]) >= 0.999));
    assert!(d.iter().any(|x| x.color[2] > 0.5) && d.iter().any(|x| x.color[0] > 0.5));
    // Hue jitter keeps saturation/brightness of a pure colour.
    let b = BrushSettings { color_dynamics: ColorDynamics { enabled: true, hue_jitter: 1.0, ..Default::default() }, ..base.clone() };
    let d = dabs_of(&b, &line(0.0, 200.0, 0.0));
    for x in &d {
        let mx = x.color[0].max(x.color[1]).max(x.color[2]);
        let mn = x.color[0].min(x.color[1]).min(x.color[2]);
        assert!((mx - 1.0).abs() < 1e-4 && mn.abs() < 1e-4, "{:?}", x.color);
    }
    assert!(d.iter().any(|x| x.color[1] > 0.5 || x.color[2] > 0.5));
    // Purity -1 desaturates completely.
    let b = BrushSettings { color_dynamics: ColorDynamics { enabled: true, purity: -1.0, ..Default::default() }, ..base.clone() };
    let d = dabs_of(&b, &line(0.0, 20.0, 0.0));
    assert!(d.iter().all(|x| (x.color[0] - x.color[1]).abs() < 1e-5 && (x.color[1] - x.color[2]).abs() < 1e-5));
    // Per-stroke: one colour for all dabs.
    let b = BrushSettings { color_dynamics: ColorDynamics { enabled: true, per_tip: false, hue_jitter: 1.0, ..Default::default() }, ..base.clone() };
    let d = dabs_of(&b, &line(0.0, 200.0, 0.0));
    assert!(d.windows(2).all(|w| w[0].color == w[1].color));
    // Per-tip colours land in the pixels.
    let b =
        BrushSettings { size: 10.0, spacing: 3.0, color_dynamics: ColorDynamics { enabled: true, fg_bg: Dynamic::jitter(1.0), ..Default::default() }, ..base };
    let s = paint(&b, &line(10.0, 310.0, 20.0), 330, 40);
    let colours: Vec<[f32; 4]> = (0..11).map(|i| s.rgba(10 + i * 30, 20)).collect();
    assert!(colours.iter().any(|c| c[2] > 0.6) && colours.iter().any(|c| c[0] > 0.6), "{colours:?}");
}

#[test]
fn transfer_opacity_caps_and_flow_builds() {
    // Pressure on opacity caps coverage.
    let b =
        BrushSettings { transfer: Transfer { enabled: true, opacity: Dynamic::controlled(Control::PenPressure), ..Default::default() }, size: 10.0, ..brush() };
    let pts = vec![StrokePoint::new(10.0, 10.0, 0.4), StrokePoint::new(90.0, 10.0, 0.4)];
    let s = paint(&b, &pts, 100, 20);
    assert!((s.rgba(50, 10)[3] - 0.4).abs() < 0.01, "{}", s.rgba(50, 10)[3]);
    // Low flow builds up with overlap but never past the opacity.
    let b = BrushSettings { flow: 0.1, opacity: 0.8, size: 10.0, ..brush() };
    let one = paint(&b, &[StrokePoint::new(50.0, 10.0, 1.0)], 100, 20).rgba(50, 10)[3];
    let many = paint(&b, &line(40.0, 60.0, 10.0), 100, 20).rgba(50, 10)[3];
    assert!((one - 0.08).abs() < 0.01, "{one}");
    assert!(many > one * 3.0 && many <= 0.8 + 1e-4, "{many}");
    // Flow jitter varies the dab alpha deterministically.
    let b = BrushSettings { transfer: Transfer { enabled: true, flow: Dynamic::jitter(1.0), ..Default::default() }, ..brush() };
    let d = dabs_of(&b, &line(0.0, 100.0, 0.0));
    assert!(d.iter().any(|x| x.alpha < 0.5) && d.iter().all(|x| x.alpha <= 1.0));
}

#[test]
fn build_up_accumulates_over_time() {
    let b = BrushSettings { build_up: true, build_up_rate: 20.0, flow: 0.05, size: 20.0, hardness: 1.0, ..brush() };
    let hold = |ms: f64| {
        let mut a = StrokePoint::new(50.0, 50.0, 1.0);
        let mut z = a;
        a.time = 0.0;
        z.time = ms;
        let s = paint(&b, &[a, z], 100, 100);
        (dabs_of(&b, &[a, z]).len(), s.rgba(50, 50)[3])
    };
    let (n0, a0) = hold(0.0);
    let (n1, a1) = hold(500.0);
    let (n2, a2) = hold(2000.0);
    assert_eq!((n0, n1, n2), (1, 11, 41));
    assert!(a0 < a1 && a1 < a2, "{a0} {a1} {a2}");
    // Without build-up, lingering adds nothing.
    let nb = BrushSettings { build_up: false, ..b };
    let mut a = StrokePoint::new(50.0, 50.0, 1.0);
    let z = StrokePoint { time: 2000.0, ..a };
    a.time = 0.0;
    assert_eq!(dabs_of(&nb, &[a, z]).len(), 1);
}

#[test]
fn huge_airbrush_time_gap_emits_bounded_catch_up() {
    let b = BrushSettings { build_up: true, build_up_rate: 1000.0, ..brush() };
    let a = StrokePoint { time: 0.0, ..StrokePoint::new(20.0, 20.0, 1.0) };
    let brief_pause = StrokePoint { time: 200.0, ..a };
    assert_eq!(dabs_of(&b, &[a, brief_pause]).len(), 201, "a 200 ms pause should retain its airbrush build-up");

    let z = StrokePoint { time: f64::MAX, ..a };
    let d = dabs_of(&b, &[a, z]);
    assert!(d.len() <= 2, "a huge pause should resume with one dab, not replay its time backlog: {}", d.len());
    assert_eq!(d.last().map(|dab| dab.center), Some(Point::new(20.0, 20.0)));

    let infinite = StrokePoint { time: f64::INFINITY, ..a };
    assert!(dabs_of(&b, &[a, infinite]).len() <= 1);
}

#[test]
fn unbounded_brush_dimensions_are_safe_in_the_public_renderer() {
    let brush = BrushSettings { size: f32::MAX, dual_brush: DualBrush { enabled: true, size: f32::INFINITY, ..Default::default() }, ..brush() };
    let ctx = BrushContext::new(&brush);
    assert_eq!(ctx.brush.size, crate::MAX_BRUSH_SIZE);
    assert_eq!(ctx.brush.dual_brush.size, 0.5);

    let d = dabs_of(&brush, &[StrokePoint::new(10.0, 10.0, 1.0)]);
    assert!(d.iter().all(|dab| dab.radius <= crate::MAX_BRUSH_SIZE / 2.0));
}

#[test]
fn texture_modulates_coverage() {
    let tile = GrayTile::from_fn(8, 8, |x, _| if x < 4 { 1.0 } else { 0.0 });
    let tex = |mode: MaskMode, depth: f32, each_tip: bool| BrushSettings {
        size: 40.0,
        texture: Texture { enabled: true, pattern: Pattern::Tile(tile.clone()), mode, depth, each_tip, ..Default::default() },
        ..brush()
    };
    for each_tip in [false, true] {
        let s = paint(&tex(MaskMode::Multiply, 1.0, each_tip), &line(20.0, 80.0, 30.0), 100, 60);
        assert!(s.rgba(46, 30)[3] < 0.1, "dark texel blocks paint");
        assert!(s.rgba(41, 30)[3] > 0.9, "light texel passes");
        let half = paint(&tex(MaskMode::Multiply, 0.5, each_tip), &line(20.0, 80.0, 30.0), 100, 60);
        let v = half.rgba(46, 30)[3];
        if each_tip {
            // Per-tip texturing builds up across overlapping dabs.
            assert!(v > 0.5 && v < 1.0, "{v}");
        } else {
            assert!((v - 0.5).abs() < 0.02, "{v}");
        }
    }
    // Brightness and invert.
    let mut b = tex(MaskMode::Multiply, 1.0, false);
    b.texture.invert = true;
    let s = paint(&b, &line(20.0, 80.0, 30.0), 100, 60);
    assert!(s.rgba(46, 30)[3] > 0.9 && s.rgba(41, 30)[3] < 0.1);
    b.texture.invert = false;
    b.texture.brightness = 1.0;
    let s = paint(&b, &line(20.0, 80.0, 30.0), 100, 60);
    // +150 lifts a black texel by 150 levels of 255, not to white (#3340).
    let lifted = s.rgba(46, 30)[3];
    assert!((lifted - 150.0 / 255.0).abs() < 0.02, "{lifted}");
    // All modes keep zero coverage at zero and stay in range.
    for m in [
        MaskMode::Multiply,
        MaskMode::Subtract,
        MaskMode::Darken,
        MaskMode::Overlay,
        MaskMode::ColorDodge,
        MaskMode::ColorBurn,
        MaskMode::LinearBurn,
        MaskMode::HardMix,
        MaskMode::LinearHeight,
        MaskMode::Height,
    ] {
        for t in [0.0, 0.3, 1.0] {
            assert_eq!(mask_combine(m, 0.0, t, 1.0), 0.0, "{m:?}");
            let v = mask_combine(m, 0.7, t, 0.8);
            assert!((0.0..=1.0).contains(&v), "{m:?}");
        }
        for t in [0.0, 0.3, 1.0] {
            assert_eq!(texture_combine(m, 0.0, t, 1.0), 0.0, "{m:?}");
            let v = texture_combine(m, 0.7, t, 0.8);
            assert!((0.0..=1.0).contains(&v), "{m:?}");
        }
        assert!((0.0..=1.0).contains(&texture_combine(m, 0.7, f32::NAN, f32::INFINITY)), "{m:?}");
        // Full-white texture never reduces multiply-like masks below the brush.
        assert!(mask_combine(m, 0.6, 1.0, 1.0) >= 0.6 - 1e-5 || matches!(m, MaskMode::Overlay), "{m:?}");
    }
    // Depth jitter per tip is deterministic and within [min depth, depth].
    let mut b = tex(MaskMode::Multiply, 0.8, true);
    b.texture.depth_jitter = Dynamic { jitter: 1.0, minimum: 0.25, ..Default::default() };
    let d = dabs_of(&b, &line(0.0, 200.0, 0.0));
    assert!(d.iter().all(|x| x.depth <= 0.8 + 1e-6 && x.depth >= 0.8 * 0.25 - 1e-6));
    assert!(d.iter().any(|x| x.depth < 0.6));
}

/// Texture Brightness and Contrast as Photoshop applies them, measured by stroking a 0..255 grey
/// ramp pattern in Multiply mode at 100 % depth (the stroke's opacity is then the pattern value):
/// (setting, ramp value, opacity read back), in levels of 255 (#3340).
#[test]
fn texture_brightness_and_contrast_match_photoshop() {
    let level = |brightness: f32, contrast: f32, invert: bool, v: f32| {
        // The model's ranges: brightness ±150 → ±1; contrast -50..0 → -1..0, 0..100 → 0..1.
        let c = if contrast < 0.0 { contrast / 50.0 } else { contrast / 100.0 };
        let t = Texture { brightness: brightness / 150.0, contrast: c, invert, ..Default::default() };
        t.level(v / 255.0) * 255.0
    };
    for (b, c, inv, v, want) in [
        (87.0, 0.0, false, 8.0, 95.0),
        (87.0, 0.0, false, 200.0, 255.0),
        (-87.0, 0.0, false, 104.0, 17.0),
        (150.0, 0.0, false, 8.0, 158.0),
        (0.0, 50.0, false, 104.0, 81.0),
        (0.0, 50.0, false, 136.0, 145.0),
        (0.0, -50.0, false, 8.0, 67.0),
        (0.0, -50.0, false, 248.0, 188.0),
        (0.0, 100.0, false, 120.0, 0.0),
        (0.0, 100.0, false, 136.0, 255.0),
        // Positive contrast after the shift, negative contrast before it.
        (40.0, 50.0, false, 104.0, 161.0),
        (40.0, 50.0, false, 40.0, 33.0),
        (-40.0, -30.0, false, 120.0, 82.0),
        (-40.0, -30.0, false, 248.0, 172.0),
        // Invert comes last.
        (40.0, 0.0, true, 8.0, 207.0),
        (0.0, 50.0, true, 72.0, 238.0),
    ] {
        let got = level(b, c, inv, v);
        assert!((got - want).abs() <= 2.0, "brightness {b}, contrast {c}, invert {inv}, ramp {v}: {got} (Photoshop {want})");
    }
}

/// The Texture modes that differ from the Dual Brush's, against single Photoshop dabs over a
/// 0..255 ramp: (mode, depth, brush coverage, ramp value, opacity read back), in levels of 255.
#[test]
fn texture_modes_match_photoshop() {
    use MaskMode::*;
    for (m, d, v, t, want) in [
        // Height: a low depth paints the dark valleys only, more depth fills the pattern.
        (Height, 0.0, 255.0, 8.0, 0.0),
        (Height, 0.01, 255.0, 8.0, 28.0),
        (Height, 0.05, 255.0, 8.0, 148.0),
        (Height, 0.05, 255.0, 136.0, 20.0),
        (Height, 0.13, 255.0, 8.0, 255.0),
        (Height, 0.13, 255.0, 152.0, 244.0),
        (Height, 0.13, 255.0, 248.0, 148.0),
        (Height, 0.13, 186.0, 88.0, 201.0),
        (Height, 0.13, 142.0, 104.0, 116.0),
        (Height, 0.25, 255.0, 248.0, 255.0),
        // Linear Height: graded by the pattern while the reach stays below its top.
        (LinearHeight, 0.01, 255.0, 152.0, 15.0),
        (LinearHeight, 0.05, 255.0, 8.0, 151.0),
        (LinearHeight, 0.05, 255.0, 152.0, 63.0),
        (LinearHeight, 0.10, 255.0, 72.0, 240.0),
        (LinearHeight, 0.13, 101.0, 120.0, 82.0),
        (LinearHeight, 0.13, 142.0, 104.0, 130.0),
        // Subtract: the light parts take paint away.
        (Subtract, 1.0, 255.0, 8.0, 247.0),
        (Subtract, 1.0, 255.0, 248.0, 7.0),
        (Subtract, 0.5, 224.0, 72.0, 188.0),
        // Color Burn / Color Dodge below full depth.
        (ColorBurn, 0.5, 224.0, 72.0, 207.0),
        (ColorDodge, 0.5, 99.0, 120.0, 130.0),
        // Unchanged modes.
        (Multiply, 0.13, 255.0, 8.0, 223.0),
        (Darken, 0.5, 255.0, 8.0, 131.0),
    ] {
        let got = texture_combine(m, v / 255.0, t / 255.0, d) * 255.0;
        assert!((got - want).abs() <= 3.0, "{m:?} depth {d}, coverage {v}, ramp {t}: {got} (Photoshop {want})");
    }
}

/// #3340: an imported Height-mode brush at a low depth with a darkened pattern painted nothing.
#[test]
fn low_depth_height_texture_paints() {
    let tile = GrayTile::from_fn(8, 8, |x, y| if (x + y) % 2 == 0 { 0.45 } else { 0.6 });
    for mode in [MaskMode::Height, MaskMode::LinearHeight] {
        let b = BrushSettings {
            size: 40.0,
            texture: Texture { enabled: true, pattern: Pattern::Tile(tile.clone()), mode, depth: 0.13, brightness: -87.0 / 150.0, ..Default::default() },
            ..brush()
        };
        let s = paint(&b, &line(20.0, 80.0, 30.0), 100, 60);
        let a = s.rgba(50, 30)[3];
        assert!(a > 0.95, "{mode:?}: {a}");
    }
}

#[test]
fn dual_brush_intersects() {
    // Dual tip: small dabs far apart → the stroke survives only where they land.
    let b = BrushSettings { size: 30.0, dual_brush: DualBrush { enabled: true, size: 6.0, spacing: 5.0, hardness: 1.0, ..Default::default() }, ..brush() };
    let s = paint(&b, &line(20.0, 200.0, 30.0), 220, 60);
    // Dual dabs every 30 px starting at x = 20.
    assert!(s.rgba(20, 30)[3] > 0.9 && s.rgba(50, 30)[3] > 0.9);
    assert!(s.rgba(35, 30)[3] < 0.01, "between dual dabs: {}", s.rgba(35, 30)[3]);
    // Outside the primary (y beyond its radius) nothing, even if dual covers.
    assert_eq!(s.rgba(20, 50)[3], 0.0);
    // Without dual, the stroke is continuous.
    let s = paint(&BrushSettings { dual_brush: DualBrush::default(), ..b }, &line(20.0, 200.0, 30.0), 220, 60);
    assert!(s.rgba(35, 30)[3] > 0.9);
}

#[test]
fn wet_edges_shape_the_stroke_not_each_dab() {
    // #2088: Wet Edges darkens the rim of the whole stroke. Along the centre of a straight stroke
    // of overlapping dabs the paint is even (no chain of per-dab rings), and the stroke's sides are
    // darker than its centre.
    let b = BrushSettings { size: 40.0, hardness: 0.8, wet_edges: true, ..brush() };
    let s = paint(&b, &line(20.0, 220.0, 50.0), 240, 100);
    let centre: Vec<f32> = (60..180).map(|x| s.rgba(x, 50)[3]).collect();
    let (lo, hi) = centre.iter().fold((f32::MAX, f32::MIN), |(lo, hi), &a| (lo.min(a), hi.max(a)));
    assert!(hi - lo < 0.02, "centre line is even: {lo}..{hi}");
    assert!(hi < 0.6, "wet interior is lighter: {hi}");
    for x in [80, 120, 160] {
        let side = s.rgba(x, 50 + 16)[3];
        assert!(side > hi + 0.15, "the side at x={x} is darker than the centre: {side} vs {hi}");
    }
}

#[test]
fn wet_edges_and_noise() {
    let b = BrushSettings { size: 40.0, hardness: 0.0, wet_edges: true, ..brush() };
    let s = paint(&b, &line(20.0, 120.0, 50.0), 140, 100);
    let centre = s.rgba(70, 50)[3];
    assert!(centre < 0.6, "wet interior is lighter: {centre}");
    // Overlap doesn't build up past a single dab.
    let single = paint(&b, &[StrokePoint::new(70.0, 50.0, 1.0)], 140, 100).rgba(70, 50)[3];
    assert!((centre - single).abs() < 0.05, "{centre} vs {single}");
    // Rim is darker than the interior for a hard tip.
    let hb = BrushSettings { hardness: 1.0, ..b };
    let d = paint(&hb, &[StrokePoint::new(50.0, 50.0, 1.0)], 100, 100);
    assert!(d.rgba(50 + 18, 50)[3] > d.rgba(50, 50)[3] + 0.2);
    // Noise: soft tip coverage becomes grainy (more pixels near 0/1).
    let soft = BrushSettings { size: 40.0, hardness: 0.0, ..brush() };
    let a = paint(&soft, &[StrokePoint::new(50.0, 50.0, 1.0)], 100, 100);
    let n = paint(&BrushSettings { noise: true, ..soft }, &[StrokePoint::new(50.0, 50.0, 1.0)], 100, 100);
    let extreme = |s: &Surface| {
        (30..70)
            .flat_map(|y| (30..70).map(move |x| (x, y)))
            .filter(|&(x, y)| {
                let v = s.rgba(x, y)[3];
                !(0.15..=0.85).contains(&v)
            })
            .count()
    };
    assert!(extreme(&n) > extreme(&a) + 100, "{} vs {}", extreme(&n), extreme(&a));
}

#[test]
fn smoothing_modes() {
    let zig: Vec<StrokePoint> = (0..40).map(|i| StrokePoint::new(i as f64 * 5.0, if i % 2 == 0 { 0.0 } else { 6.0 }, 1.0)).collect();
    let dev = |pts: &[StrokePoint]| pts.iter().map(|p| (p.y - 3.0).abs()).sum::<f64>() / pts.len() as f64;
    // Exponential smoothing reduces the zig-zag.
    let cfg = Smoothing { amount: 0.8, ..Default::default() };
    let sm = smooth_points(&cfg, 1.0, &zig);
    assert!(dev(&sm[5..sm.len() - 1]) < dev(&zig) * 0.5);
    // Catch-up on end finishes at the last input point; without it, the stroke ends short.
    assert_eq!(sm.last().map(|p| (p.x, p.y)), zig.last().map(|p| (p.x, p.y)));
    let no = smooth_points(&Smoothing { catch_up_on_end: false, ..cfg.clone() }, 1.0, &zig);
    assert!(no.last().unwrap().x < zig.last().unwrap().x - 1.0);
    // Pulled string: jitter smaller than the string moves nothing; the brush trails by the string.
    let ps = Smoothing { amount: 0.2, pulled_string: true, catch_up_on_end: false, ..Default::default() };
    let jitter: Vec<StrokePoint> = (0..20).map(|i| StrokePoint::new(if i % 2 == 0 { 0.0 } else { 5.0 }, 0.0, 1.0)).collect();
    assert_eq!(smooth_points(&ps, 1.0, &jitter).len(), 1, "within the 20 px string: no motion");
    let far = smooth_points(&ps, 1.0, &[StrokePoint::new(0.0, 0.0, 1.0), StrokePoint::new(100.0, 0.0, 1.0)]);
    assert!((far.last().unwrap().x - 80.0).abs() < 1e-4);
    // Adjust for zoom: at 200 % zoom the string is half as long in image pixels.
    let far = smooth_points(&ps, 2.0, &[StrokePoint::new(0.0, 0.0, 1.0), StrokePoint::new(100.0, 0.0, 1.0)]);
    assert!((far.last().unwrap().x - 90.0).abs() < 1e-4);
    // Stroke catch-up: a pause (timestamps) lets the brush converge on the pointer.
    let mut a = StrokePoint::new(0.0, 0.0, 1.0);
    a.time = 0.0;
    let b = StrokePoint { x: 100.0, time: 16.0, ..a };
    let c = StrokePoint { time: 500.0, ..b };
    let mut s = Smoother::new(&Smoothing { amount: 0.8, catch_up: true, catch_up_on_end: false, ..Default::default() }, 1.0);
    let mut out = Vec::new();
    for p in [a, b, c] {
        s.push(p, &mut out);
    }
    assert!(out.last().unwrap().x > 99.0, "{}", out.last().unwrap().x);
    let mut s = Smoother::new(&Smoothing { amount: 0.8, catch_up: false, catch_up_on_end: false, ..Default::default() }, 1.0);
    let mut out = Vec::new();
    for p in [a, b, c] {
        s.push(p, &mut out);
    }
    assert!(out.last().unwrap().x < 70.0, "{}", out.last().unwrap().x);
}

#[test]
fn disabled_smoothing_keeps_its_settings_and_paints_unsmoothed_at_every_depth() {
    let pts = vec![StrokePoint::new(10.0, 20.0, 1.0), StrokePoint::new(30.0, 8.0, 0.5), StrokePoint::new(50.0, 20.0, 0.8)];
    for pulled_string in [false, true] {
        let smoothing = Smoothing { enabled: false, amount: 0.8, pulled_string, ..Default::default() };
        let got = smooth_points(&smoothing, 1.0, &pts);
        assert_eq!(got.len(), pts.len());
        for (a, b) in got.iter().zip(&pts) {
            assert_eq!((a.x, a.y, a.pressure), (b.x, b.y, b.pressure));
        }
        let b = BrushSettings { size: 9.0, smoothing: smoothing.clone(), ..brush() };
        let zero = BrushSettings { smoothing: Smoothing { amount: 0.0, ..smoothing.clone() }, ..b.clone() };
        for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F] {
            let (mut off, mut reference) = (Surface::new(fmt), Surface::new(fmt));
            render_stroke(&mut off, &b, &pts, None, false, 1.0);
            render_stroke(&mut reference, &zero, &pts, None, false, 1.0);
            for y in 0..40 {
                for x in 0..64 {
                    assert_eq!(off.pixel(x, y), reference.pixel(x, y), "{fmt:?} at {x},{y}");
                }
            }
        }
        // Disabling is not an amount=0 rewrite: saving and re-enabling recover the same mode.
        let mut back: Smoothing = serde_json::from_value(serde_json::to_value(&smoothing).unwrap()).unwrap();
        assert_eq!(back, smoothing);
        back.enabled = true;
        assert!(smooth_points(&back, 1.0, &pts).iter().zip(&pts).any(|(a, b)| a.x != b.x || a.y != b.y));
    }
    // Existing brush JSON has no switch and must retain its old smoothing behavior.
    let legacy: Smoothing = serde_json::from_str(r#"{"amount":0.8,"pulledString":true}"#).unwrap();
    assert!(legacy.enabled && legacy.amount == 0.8 && legacy.pulled_string);
}

#[test]
fn smoothing_string_runs_from_the_lagging_brush_to_the_pointer() {
    // #3479: the canvas draws the line between the brush and the pointer.
    let cfg = Smoothing { amount: 0.8, ..Default::default() };
    let mut s = Smoother::new(&cfg, 1.0);
    let mut out = Vec::new();
    assert_eq!(s.string(), None, "no stroke yet");
    s.push(StrokePoint::new(0.0, 0.0, 1.0), &mut out);
    assert_eq!(s.string(), None, "the first point is the brush");
    s.push(StrokePoint::new(100.0, 0.0, 1.0), &mut out);
    let (brush, pointer) = s.string().unwrap();
    assert!(brush[0] > 0.0 && brush[0] < 100.0, "the brush lags: {brush:?}");
    assert_eq!(pointer, [100.0, 0.0]);
    let mut off = Smoother::new(&Smoothing { amount: 0.0, ..cfg }, 1.0);
    off.push(StrokePoint::new(0.0, 0.0, 1.0), &mut out);
    off.push(StrokePoint::new(100.0, 0.0, 1.0), &mut out);
    assert_eq!(off.string(), None, "smoothing off: no string");
}

#[test]
fn smoothing_pulls_by_time_not_by_how_often_the_device_reports() {
    // The pointer jumps 100 px and is held there for 32 ms, reported at 1000 Hz or about 60 Hz.
    let cfg = Smoothing { amount: 0.8, catch_up: true, catch_up_on_end: false, ..Default::default() };
    let run = |interval: f64| {
        let mut s = Smoother::new(&cfg, 1.0);
        let mut out = Vec::new();
        s.push(StrokePoint::new(0.0, 0.0, 1.0), &mut out);
        let mut t = 0.0;
        while t < 32.0 {
            t += interval;
            s.push(StrokePoint { time: t, ..StrokePoint::new(100.0, 0.0, 1.0) }, &mut out);
        }
        out.last().map_or(0.0, |p| p.x)
    };
    let (fast, slow) = (run(1.0), run(16.0));
    // Two 16 ms steps of a 0.2 pull: 1 - 0.8² = 36 % of the way.
    assert!((slow - 36.0).abs() < 1e-4, "{slow}");
    assert!((fast - slow).abs() < 1e-4, "1000 Hz {fast} vs 60 Hz {slow}");
    // Without timestamps each point is one step, as before.
    let mut s = Smoother::new(&cfg, 1.0);
    let mut out = Vec::new();
    for x in [0.0, 100.0] {
        s.push(StrokePoint::new(x, 0.0, 1.0), &mut out);
    }
    assert!((out.last().map_or(0.0, |p| p.x) - 20.0).abs() < 1e-4);
}

#[test]
fn smoothing_without_catch_up_stops_while_the_pointer_pauses() {
    let cfg = Smoothing { amount: 0.8, catch_up: false, catch_up_on_end: false, ..Default::default() };
    let mut s = Smoother::new(&cfg, 1.0);
    let mut out = Vec::new();
    s.push(StrokePoint::new(0.0, 0.0, 1.0), &mut out);
    s.push(StrokePoint { time: 16.0, ..StrokePoint::new(100.0, 0.0, 1.0) }, &mut out);
    let n = out.len();
    for t in [32.0, 48.0, 64.0] {
        s.push(StrokePoint { time: t, ..StrokePoint::new(100.0, 0.0, 1.0) }, &mut out);
    }
    assert_eq!(out.len(), n, "held still: the brush doesn't move");
}

#[test]
fn pose_overrides_stylus_values() {
    let b = BrushSettings {
        size: 40.0,
        pose: Pose { enabled: true, pressure: 0.5, override_pressure: true, ..Default::default() },
        shape_dynamics: ShapeDynamics { enabled: true, size: Dynamic::controlled(Control::PenPressure), ..Default::default() },
        ..brush()
    };
    let d = dabs_of(&b, &line(0.0, 50.0, 0.0));
    assert!(d.iter().all(|x| (x.radius - 10.0).abs() < 1e-4));
}

#[test]
fn chunked_rendering_matches_one_shot() {
    let b = BrushSettings {
        size: 24.0,
        seed: 9,
        shape_dynamics: ShapeDynamics { enabled: true, size: Dynamic::jitter(0.5), angle: Dynamic::jitter(1.0), ..Default::default() },
        scattering: Scattering { enabled: true, scatter: Dynamic::jitter(1.0), both_axes: true, count: 2, ..Default::default() },
        texture: Texture { enabled: true, depth: 0.5, ..Default::default() },
        color_dynamics: ColorDynamics { enabled: true, hue_jitter: 0.5, ..Default::default() },
        color: [1.0, 0.0, 0.0, 1.0],
        ..brush()
    };
    let pts: Vec<StrokePoint> = (0..60).map(|i| StrokePoint::new(20.0 + i as f64 * 4.0, 60.0 + (i as f64 * 0.3).sin() * 30.0, 1.0)).collect();
    let mut one = Surface::new(PixelFormat::RGBA8);
    let d1 = render_stroke(&mut one, &b, &pts, None, false, 1.0);
    let mut inc = Surface::new(PixelFormat::RGBA8);
    let pre = inc.clone();
    let mut r = StrokeRenderer::new(&b, Some(inc.format()), 1.0);
    for ch in pts.chunks(7) {
        r.push(ch);
        r.composite(&pre, &mut inc, None, false, false);
    }
    r.finish();
    r.composite(&pre, &mut inc, None, false, false);
    assert_eq!(d1, r.bounds());
    for y in d1.y0..d1.y1 {
        for x in d1.x0..d1.x1 {
            assert_eq!(one.pixel(x, y), inc.pixel(x, y), "({x},{y})");
        }
    }
}

#[test]
fn paints_every_depth_and_model() {
    let b = BrushSettings { size: 8.0, color: [1.0, 0.0, 0.0, 1.0], ..brush() };
    for mode in [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk] {
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for alpha in [true, false] {
                let fmt = PixelFormat::new(mode, sample, alpha);
                let mut s = Surface::with_default(fmt, &photocraft_raster::from_rgba(&fmt, [1.0, 1.0, 1.0, 1.0]));
                let fancy = BrushSettings {
                    shape_dynamics: ShapeDynamics { enabled: true, size: Dynamic::jitter(0.3), ..Default::default() },
                    color_dynamics: ColorDynamics { enabled: true, brightness_jitter: 0.1, ..Default::default() },
                    texture: Texture { enabled: true, depth: 0.2, ..Default::default() },
                    ..b.clone()
                };
                for br in [&b, &fancy] {
                    render_stroke(&mut s, br, &line(5.0, 40.0, 5.0), None, false, 1.0);
                }
                let c = s.rgba(20, 5);
                let expect = photocraft_raster::to_rgba(&fmt, &photocraft_raster::from_rgba(&fmt, [1.0, 0.0, 0.0, 1.0]));
                assert!((c[0] - expect[0]).abs() < 0.15 && (c[1] - expect[1]).abs() < 0.12, "{fmt:?} {c:?} vs {expect:?}");
                assert_eq!(s.pixel(20, 30), s.pixel(5000, 5000), "{fmt:?} untouched");
            }
        }
    }
}

#[test]
fn blend_mode_and_dissolve() {
    let mut s = Surface::new(PixelFormat::RGBA32F);
    s.fill_rect(Rect::new(0, 0, 100, 20), &[0.5, 0.5, 0.5, 1.0]);
    let b = BrushSettings { size: 10.0, color: [0.5, 0.5, 0.5, 1.0], mode: BlendMode::Multiply, ..brush() };
    render_stroke(&mut s, &b, &line(10.0, 90.0, 10.0), None, false, 1.0);
    assert!((s.rgba(50, 10)[0] - 0.25).abs() < 1e-4);
    let mut s = Surface::new(PixelFormat::RGBA32F);
    let b = BrushSettings { size: 30.0, opacity: 0.5, mode: BlendMode::Dissolve, ..brush() };
    render_stroke(&mut s, &b, &line(20.0, 80.0, 20.0), None, false, 1.0);
    let vals: Vec<f32> = (30..70).map(|x| s.rgba(x, 20)[3]).collect();
    assert!(vals.iter().all(|&v| v == 0.0 || v == 1.0) && vals.contains(&0.0) && vals.contains(&1.0));
}

#[test]
fn presets_render_and_round_trip() {
    let presets = crate::presets::builtin();
    assert!(presets.len() >= 12);
    for p in &presets {
        let j = serde_json::to_string(p).unwrap();
        let back: BrushPreset = serde_json::from_str(&j).unwrap();
        assert_eq!(&back, p, "{}", p.name);
        let mut s = Surface::new(PixelFormat::RGBA8);
        let pts: Vec<StrokePoint> = (0..20).map(|i| StrokePoint { time: i as f64 * 16.0, ..StrokePoint::new(40.0 + i as f64 * 8.0, 60.0, 0.8) }).collect();
        let dmg = render_stroke(&mut s, &p.brush, &pts, None, false, 1.0);
        assert!(!dmg.is_empty(), "{}", p.name);
        assert!(alpha_sum(&s, dmg) > 10.0, "{} painted nothing", p.name);
    }
    // Partial JSON fills defaults.
    let b: BrushSettings = serde_json::from_str(r#"{"size": 50, "scattering": {"enabled": true, "scatter": {"jitter": 2}}, "pressure_size": false}"#).unwrap();
    assert_eq!(b.size, 50.0);
    assert!(b.scattering.enabled && b.scattering.count == 1 && !b.pressure_size);
    assert_eq!(b.hardness, 1.0);
}

#[test]
fn protect_texture_keeps_pattern() {
    let presets = crate::presets::builtin();
    let chalk = crate::presets::find(&presets, "chalk").unwrap().brush.clone();
    let canvas = crate::presets::find(&presets, "Canvas Texture").unwrap().brush.clone();
    let cur = BrushSettings { protect_texture: true, ..chalk.clone() };
    let next = canvas.clone().with_protected_texture(&cur);
    assert_eq!(next.texture.pattern, chalk.texture.pattern);
    assert_eq!(next.texture.mode, canvas.texture.mode);
    assert_eq!(canvas.clone().with_protected_texture(&chalk).texture.pattern, canvas.texture.pattern);
}

#[test]
fn tail_preview_shows_the_stroke_as_finishing_it_would() {
    // Live previews draw the smoothing catch-up tail before the stroke ends: compositing the
    // tail preview over the incremental composite gives exactly the finished stroke.
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U16, true);
    let b = BrushSettings { size: 9.0, smoothing: Smoothing { amount: 0.7, ..Default::default() }, ..brush() };
    let pts: Vec<StrokePoint> = (0..12).map(|i| StrokePoint::new(5.0 + i as f64 * 9.0, 20.0 + (i % 3) as f64 * 7.0, 1.0)).collect();
    let pre = Surface::new(fmt);
    let mut live = pre.clone();
    let mut r = StrokeRenderer::new(&b, Some(fmt), 1.0);
    r.push(&pts);
    r.composite(&pre, &mut live, None, false, false);
    let mut tail = r.tail_preview().expect("the brush lags: there is a tail");
    tail.composite(&pre, &mut live, None, false, false);
    let mut done = pre.clone();
    render_stroke(&mut done, &b, &pts, None, false, 1.0);
    assert!((0..60).all(|y| (0..120).all(|x| live.rgba(x, y) == done.rgba(x, y))));
    // Without smoothing (and past the first dab) finishing adds nothing.
    let mut r = StrokeRenderer::new(&brush(), Some(fmt), 1.0);
    r.push(&pts);
    assert!(r.tail_preview().is_none());
}

#[test]
fn live_union_of_passes_matches_merging_the_finished_passes_after_every_chunk() {
    // Symmetric painting previews the stroke and its mirrors with `composite_union_live`, which
    // redraws only changed tiles; after every chunk the target must be exactly what finishing
    // every pass and merging them with `composite_union_many` gives (the commit takes these pixels).
    let base = BrushSettings { size: 14.0, hardness: 0.6, opacity: 0.6, ..brush() };
    let mut brushes = vec![base.clone()];
    brushes.push(BrushSettings { smoothing: Smoothing { amount: 0.7, ..Default::default() }, ..base.clone() });
    let mut b = base.clone();
    b.dual_brush.enabled = true;
    b.dual_brush.size = 6.0;
    b.dual_brush.scatter = 1.5;
    brushes.push(b);
    let mut b = base.clone();
    b.color_dynamics.enabled = true;
    b.color_dynamics.hue_jitter = 0.5;
    brushes.push(b);
    brushes.push(BrushSettings { wet_edges: true, ..base.clone() });
    brushes.push(BrushSettings { mode: BlendMode::Dissolve, ..base.clone() });
    brushes.push(BrushSettings { erase: true, ..base.clone() });
    let mut b = base.clone();
    b.texture.enabled = true;
    b.texture.each_tip = false;
    brushes.push(b);
    let stroke: Vec<StrokePoint> = (0..40).map(|i| StrokePoint::new(30.0 + i as f64 * 3.7, 40.0 + 25.0 * (i as f64 / 5.0).sin(), 1.0)).collect();
    // Mirrors about x = 100 and y = 75; the first point sits near the vertical axis so passes overlap.
    let mirror = |fx: bool, fy: bool| -> Vec<StrokePoint> {
        stroke.iter().map(|p| StrokePoint::new(if fx { 200.0 - p.x } else { p.x }, if fy { 150.0 - p.y } else { p.y }, p.pressure)).collect()
    };
    let paths = [stroke.clone(), mirror(true, false), mirror(false, true), mirror(true, true)];
    let area = Rect::new(-20, -20, 240, 190);
    let mut sel = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, false));
    let sel_rect = Rect::new(0, 0, 160, 120);
    sel.write_region(sel_rect, &(0..160 * 120).map(|i| ((i % 160) as f32 / 120.0).min(1.0)).collect::<Vec<_>>());
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let fmt = PixelFormat::new(ColorMode::Rgb, sample, true);
        let mut pre = Surface::new(fmt);
        let fill = Rect::new(10, 10, 190, 140);
        let n = fmt.channels();
        let px: Vec<f32> =
            (0..fill.width() * fill.height()).flat_map(|i| [(i % 7) as f32 / 7.0, 0.5, (i % 13) as f32 / 13.0, 0.8].into_iter().take(n)).collect();
        pre.write_region(fill, &px);
        for (bi, b) in brushes.iter().enumerate() {
            // (passes included at each chunk, selection): two passes from the start; four, of
            // which two join part-way.
            for (four, selection) in [(false, None), (true, None), (true, Some(&sel))] {
                let mut renderers: Vec<StrokeRenderer> = (0..4).map(|_| StrokeRenderer::new(b, Some(fmt), 1.0)).collect();
                let mut live = crate::render::UnionLive::default();
                let mut target = pre.clone();
                let (mut at, mut chunk, mut merged) = (0, 1, 0);
                while at < stroke.len() {
                    let end = (at + chunk).min(stroke.len());
                    for (r, path) in renderers.iter_mut().zip(&paths) {
                        r.push(&path[at..end]);
                    }
                    let count = if four && at >= 12 { 4 } else { 2 };
                    let full = merged != count;
                    merged = count;
                    let mut passes: Vec<&mut StrokeRenderer> = renderers.iter_mut().take(count).collect();
                    StrokeRenderer::composite_union_live(&mut passes, &mut live, full, &pre, &mut target, selection, false);
                    let finished: Vec<StrokeRenderer> = renderers
                        .iter()
                        .take(count)
                        .map(|r| {
                            let mut r = r.clone();
                            r.finish();
                            r
                        })
                        .collect();
                    let mut expected = pre.clone();
                    finished[0].composite_union_many(&finished[1..], &pre, &mut expected, selection, false);
                    assert!(
                        target.read_region(area) == expected.read_region(area),
                        "{sample:?}, brush {bi}, four {four}, selection {}, points ..{end}",
                        selection.is_some()
                    );
                    assert_eq!(live.bounds(), finished.iter().fold(Rect::EMPTY, |r, f| r.union(&f.bounds())));
                    at = end;
                    chunk = chunk % 3 + 1;
                }
            }
        }
    }
}

// ---------- Tilt Scale, Brush Projection, speed spacing, locks, mixer transfer ----------

/// Coverage extent (pixels above half coverage) of a rasterised dab along x and y through its centre.
fn extents(b: &BrushSettings, d: &Dab) -> (usize, usize) {
    let r = raster(b, d);
    let (cx, cy) = (d.center.x.floor() as i32, d.center.y.floor() as i32);
    let w = (r.0.x0..r.0.x1).filter(|&x| at(&r, x, cy) > 0.5).count();
    let h = (r.0.y0..r.0.y1).filter(|&y| at(&r, cx, y) > 0.5).count();
    (w, h)
}

fn pen(x: f64, y: f64, tilt_x: f32, tilt_y: f32, rotation: f32) -> StrokePoint {
    StrokePoint { tilt_x, tilt_y, rotation, ..StrokePoint::new(x, y, 1.0) }
}

#[test]
fn tilt_scale_squashes_the_tip_height_with_pen_tilt() {
    // Size control on Pen Tilt with a 100 % minimum, so only Tilt Scale changes the shape.
    let mk = |scale: f32| BrushSettings {
        size: 40.0,
        shape_dynamics: ShapeDynamics {
            enabled: true,
            size: Dynamic { control: Control::PenTilt, minimum: 1.0, ..Default::default() },
            tilt_scale: scale,
            ..Default::default()
        },
        ..brush()
    };
    let upright = [pen(50.0, 50.0, 0.0, 0.0, 0.0)];
    let tilted = [pen(50.0, 50.0, 45.0, 0.0, 0.0)];
    // 0 % (the default) and an upright pen leave the tip round.
    let d = dabs_of(&mk(0.0), &tilted)[0];
    assert_eq!(d.roundness, 1.0);
    let d = dabs_of(&mk(2.0), &upright)[0];
    assert_eq!(d.roundness, 1.0);
    // 100 % at 45° tilt: height × (1 − 0.5 × 0.5) = 0.75; 200 %: 0.5.
    let d = dabs_of(&mk(1.0), &tilted)[0];
    assert!((d.roundness - 0.75).abs() < 1e-4, "{}", d.roundness);
    let d2 = dabs_of(&mk(2.0), &tilted)[0];
    assert!((d2.roundness - 0.5).abs() < 1e-4, "{}", d2.roundness);
    let (w, h) = extents(&mk(2.0), &d2);
    assert!(w >= 38 && (h as i32 - 20).abs() <= 2, "{w}×{h}");
    // Only with the size control on Pen Tilt (Photoshop greys the slider otherwise).
    let mut b = mk(2.0);
    b.shape_dynamics.size.control = Control::PenPressure;
    assert_eq!(dabs_of(&b, &tilted)[0].roundness, 1.0);
    // Rendered strokes differ.
    let a = paint(&mk(0.0), &[pen(20.0, 40.0, 60.0, 0.0, 0.0), pen(120.0, 40.0, 60.0, 0.0, 0.0)], 0, 0);
    let c = paint(&mk(2.0), &[pen(20.0, 40.0, 60.0, 0.0, 0.0), pen(120.0, 40.0, 60.0, 0.0, 0.0)], 0, 0);
    let area = Rect::new(0, 0, 160, 80);
    assert!(alpha_sum(&c, area) < alpha_sum(&a, area) * 0.75, "{} vs {}", alpha_sum(&c, area), alpha_sum(&a, area));
}

#[test]
fn brush_projection_foreshortens_along_tilt_and_turns_with_rotation() {
    let mk = |on: bool| BrushSettings { size: 40.0, shape_dynamics: ShapeDynamics { enabled: true, brush_projection: on, ..Default::default() }, ..brush() };
    // Tilted 60° towards +x: the tip is foreshortened to cos 60° = ½ along x.
    let pts = [pen(50.0, 50.0, 60.0, 0.0, 0.0)];
    let d = dabs_of(&mk(true), &pts)[0];
    assert!((d.proj_scale - 0.5).abs() < 1e-4 && d.proj_angle.abs() < 1e-5, "{d:?}");
    let (w, h) = extents(&mk(true), &d);
    assert!((w as i32 - 20).abs() <= 2 && h >= 38, "{w}×{h}");
    // Tilted along y instead: foreshortened vertically.
    let d = dabs_of(&mk(true), &[pen(50.0, 50.0, 0.0, 60.0, 0.0)])[0];
    let (w, h) = extents(&mk(true), &d);
    assert!(w >= 38 && (h as i32 - 20).abs() <= 2, "{w}×{h}");
    // Off: the tilt doesn't touch the tip.
    let d = dabs_of(&mk(false), &pts)[0];
    assert_eq!((d.proj_scale, d.angle), (1.0, 0.0));
    assert_eq!(extents(&mk(false), &d), (40, 40));
    // Barrel rotation turns the tip (an elliptical one shows it).
    let mut b = mk(true);
    b.roundness = 0.5;
    let d = dabs_of(&b, &[pen(50.0, 50.0, 0.0, 0.0, 90.0)])[0];
    assert!((d.angle.to_degrees() - 90.0).abs() < 1e-3);
    let (w, h) = extents(&b, &d);
    assert!(w < h, "{w}×{h}");
    // Dual-brush dabs are never projected.
    let ctx = BrushContext::new(&mk(true));
    let dd = Dab { proj_scale: 0.3, ..Dab::round(Point::new(50.0, 50.0), 20.0, 1.0) };
    let (r, mut a, mut c) = (ctx.dab_rect(&dd, true), Vec::new(), Vec::new());
    ctx.rasterize(&dd, true, r, &mut a);
    ctx.rasterize(&Dab { proj_scale: 1.0, ..dd }, true, r, &mut c);
    assert_eq!(a, c);
}

#[test]
fn spacing_off_spaces_dabs_by_pointer_speed() {
    let mut b = BrushSettings { size: 20.0, spacing_enabled: false, ..brush() };
    let timed = |ms: f64| vec![StrokePoint { time: 0.0, ..StrokePoint::new(0.0, 10.0, 1.0) }, StrokePoint { time: ms, ..StrokePoint::new(100.0, 10.0, 1.0) }];
    // 100 px in 400 ms: a dab every 8 ms = every 2 px; in 40 ms: every 20 px.
    let slow = dabs_of(&b, &timed(400.0)).len();
    let fast = dabs_of(&b, &timed(40.0)).len();
    assert_eq!((slow, fast), (51, 6));
    // Never denser than half a pixel, however slow.
    assert!(dabs_of(&b, &timed(1.0e9)).len() <= 201);
    // No timestamps (a ⇧-click line): spaced by distance like Spacing on, not a dab per point.
    let pts = [StrokePoint::new(0.0, 10.0, 1.0), StrokePoint::new(90.0, 10.0, 1.0), StrokePoint::new(100.0, 10.0, 1.0)];
    let checked = BrushSettings { spacing_enabled: true, ..b.clone() };
    assert_eq!(dabs_of(&b, &pts).len(), dabs_of(&checked, &pts).len());
    assert!(dabs_of(&b, &pts).len() > 3);
    // Checked: fixed spacing (25 % of 20 px = 5 px) whatever the speed.
    b.spacing_enabled = true;
    assert_eq!(dabs_of(&b, &timed(400.0)).len(), dabs_of(&b, &timed(40.0)).len());
    // Chunked input gives the same dabs.
    b.spacing_enabled = false;
    let pts: Vec<StrokePoint> = (0..20).map(|i| StrokePoint { time: f64::from(i) * 13.0, ..StrokePoint::new(f64::from(i) * 7.0, 10.0, 1.0) }).collect();
    let one = dabs_of(&b, &pts);
    let mut g = crate::dynamics::DabGenerator::new(&b, 1.0);
    let (mut out, mut dual) = (Vec::new(), Vec::new());
    for c in pts.chunks(3) {
        g.push(c, &mut out, &mut dual);
    }
    g.finish(&mut out, &mut dual);
    assert_eq!(one, out);
}

#[test]
fn locked_sections_survive_picking_a_preset() {
    let presets = crate::presets::builtin();
    let chalk = crate::presets::find(&presets, "chalk").unwrap().brush.clone();
    let mut cur = BrushSettings {
        scattering: Scattering { enabled: true, count: 5, ..Default::default() },
        transfer: Transfer { enabled: true, wetness: Dynamic::jitter(0.4), ..Default::default() },
        noise: true,
        ..brush()
    };
    // Unlocked: the preset's sections win.
    let next = chalk.clone().picked_over(&cur);
    assert_eq!(next.scattering, chalk.scattering);
    assert_eq!(next.noise, chalk.noise);
    // Locked: the current sections stay, and so do the locks.
    cur.locks = SectionLocks { scattering: true, transfer: true, noise: true, ..Default::default() };
    let next = chalk.clone().picked_over(&cur);
    assert_eq!(next.scattering, cur.scattering);
    assert_eq!(next.transfer, cur.transfer);
    assert!(next.noise);
    assert_eq!(next.locks, cur.locks);
    // Unlocked sections still come from the preset.
    assert_eq!(next.texture, chalk.texture);
    assert_eq!(next.tip, chalk.tip);
}

#[test]
fn transfer_wetness_and_mix_vary_per_dab() {
    let b = BrushSettings {
        transfer: Transfer {
            enabled: true,
            wetness: Dynamic::controlled(Control::PenPressure),
            mix: Dynamic { jitter: 1.0, minimum: 0.25, ..Default::default() },
            ..Default::default()
        },
        ..brush()
    };
    let pts = [StrokePoint::new(0.0, 0.0, 0.3), StrokePoint::new(60.0, 0.0, 0.3)];
    let d = dabs_of(&b, &pts);
    assert!(d.iter().all(|x| (x.wet - 0.3).abs() < 1e-5));
    assert!(d.iter().all(|x| (0.25..=1.0).contains(&x.mix)) && d.iter().any(|x| x.mix < 0.9));
    // Transfer off: as set.
    let d = dabs_of(&brush(), &pts);
    assert!(d.iter().all(|x| x.wet == 1.0 && x.mix == 1.0));
    // New fields round-trip and old JSON loads with neutral values.
    let full = BrushSettings {
        spacing_enabled: false,
        shape_dynamics: ShapeDynamics { tilt_scale: 1.5, brush_projection: true, ..Default::default() },
        locks: SectionLocks { texture: true, ..Default::default() },
        mixer: crate::mixer::MixerSettings { wet: 0.9, sample_all_layers: true, ..Default::default() },
        ..b
    };
    let back: BrushSettings = serde_json::from_value(serde_json::to_value(&full).unwrap()).unwrap();
    assert_eq!(back, full);
    let old: BrushSettings = serde_json::from_str(r#"{"size": 12, "shapeDynamics": {"enabled": true}, "transfer": {"enabled": true}}"#).unwrap();
    assert!(old.spacing_enabled && !old.shape_dynamics.brush_projection && old.shape_dynamics.tilt_scale == 0.0 && old.locks == SectionLocks::default());
    assert_eq!(old.mixer, crate::mixer::MixerSettings::default());
}

/// Primary and Dual Brush dabs of a stroke (the generator, as the renderer drives it).
fn all_dabs(b: &BrushSettings, pts: &[StrokePoint]) -> (Vec<Dab>, Vec<Dab>) {
    let mut g = crate::dynamics::DabGenerator::new(b, 1.0);
    let (mut dabs, mut duals) = (Vec::new(), Vec::new());
    g.push(pts, &mut dabs, &mut duals);
    g.finish(&mut dabs, &mut duals);
    (dabs, duals)
}

/// A Scatter amount far beyond the 1000 % maximum (or non-finite), on the primary tip or the
/// Dual Brush, used to push dab centres to infinity and overflow the dab rectangle (#977). The
/// renderer caps it at the maximum, so dabs stay finite and within reach, and painting works at
/// every bit depth.
#[test]
fn huge_scatter_is_capped_and_renders_at_every_depth() {
    let pts = line(10.0, 40.0, 10.0);
    let with = |j: f32| BrushSettings {
        size: 20.0,
        scattering: Scattering { enabled: true, scatter: Dynamic::jitter(j), both_axes: true, count: 4, ..Default::default() },
        dual_brush: DualBrush { enabled: true, size: 6.0, scatter: j, both_axes: true, ..Default::default() },
        ..brush()
    };
    let capped = all_dabs(&with(MAX_SCATTER), &pts);
    // Primary reach: MAX_SCATTER radii of 10 px; dual reach: MAX_SCATTER radii of 3 px.
    let near = |ds: &[Dab], r: f64| ds.iter().all(|d| (d.center.x - 25.0).abs() <= 15.0 + r && (d.center.y - 10.0).abs() <= r);
    assert!(near(&capped.0, f64::from(MAX_SCATTER) * 10.0) && near(&capped.1, f64::from(MAX_SCATTER) * 3.0));
    for j in [MAX_SCATTER + 0.01, 1e6, 1e38, f32::MAX, f32::INFINITY] {
        let b = with(j);
        assert_eq!(all_dabs(&b, &pts), capped, "scatter {j} paints like the maximum");
        let ctx = BrushContext::new(&b);
        for d in &capped.0 {
            let r = ctx.dab_rect(d, false);
            // A 20 px soft tip reaches about 1.6 radii (see `tip_reach`).
            let side = 2 * (crate::tip_reach(10.0, b.hardness).ceil() as u32 + 1) + 2;
            assert!(!r.is_empty() && r.width() <= side && r.height() <= side, "{r:?} vs {side}");
        }
        for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, sample, true));
            let damage = render_stroke(&mut s, &b, &pts, None, false, 1.0);
            assert!(!damage.is_empty() && damage.x0 > -200 && damage.x1 < 250, "scatter {j} at {sample:?}: {damage:?}");
        }
    }
    // A NaN amount scatters nothing, as a zero amount does.
    let none = all_dabs(&with(0.0), &pts);
    assert_eq!(all_dabs(&with(f32::NAN), &pts), none);
    assert!(none.0.iter().all(|d| d.center.y == 10.0));
}

/// Control: an ordinary Scatter amount above 100 % still scatters, unchanged by the cap (#977).
#[test]
fn ordinary_scatter_is_unchanged_by_the_cap() {
    let b = BrushSettings {
        size: 20.0,
        scattering: Scattering { enabled: true, scatter: Dynamic::jitter(2.0), both_axes: true, count: 3, ..Default::default() },
        dual_brush: DualBrush { enabled: true, size: 6.0, scatter: 1.5, both_axes: true, ..Default::default() },
        ..brush()
    };
    assert_eq!(b.bounded_for_render(), b);
    let (dabs, duals) = all_dabs(&b, &line(10.0, 200.0, 50.0));
    assert!(dabs.iter().any(|d| (d.center.y - 50.0).abs() > 10.0) && dabs.iter().all(|d| (d.center.y - 50.0).abs() <= 20.0));
    assert!(duals.iter().any(|d| (d.center.y - 50.0).abs() > 1.0) && duals.iter().all(|d| (d.center.y - 50.0).abs() <= 4.5));
}

/// The dab rectangle of a hostile centre (non-finite or beyond `i32`), from a direct caller of the
/// public API, saturates instead of overflowing (#977).
#[test]
fn dab_rect_of_a_far_or_non_finite_centre_does_not_overflow() {
    let ctx = BrushContext::new(&brush());
    for c in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 1e300, -1e300, f64::from(i32::MAX), f64::from(i32::MIN)] {
        let d = Dab::round(Point::new(c, c), 10.0, 1.0);
        for r in [ctx.dab_rect(&d, false), ctx.dab_rect(&d, true), crate::retouch::dab_rect(&d)] {
            assert!(r.x0 <= r.x1 && r.y0 <= r.y1, "centre {c}: {r:?}");
        }
    }
    let huge = Dab::round(Point::new(5.0, 5.0), f32::INFINITY, 1.0);
    let r = crate::retouch::dab_rect(&huge);
    assert!(r.x0 <= r.x1 && r.y0 <= r.y1, "{r:?}");
}

/// Alpha down from a horizontal line at `y = 100.5` on an f32 surface, one value per pixel row.
fn cross_section(b: &BrushSettings, pts: &[StrokePoint], x: i32) -> Vec<f32> {
    let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::F32, true));
    render_stroke(&mut s, b, pts, None, false, 1.0);
    (100..170).map(|y| s.rgba(x, y)[3]).collect()
}

/// Rows from where the profile drops below 90 % to where it drops below 10 %.
fn soft_edge_width(profile: &[f32]) -> usize {
    let below = |v: f32| profile.iter().position(|a| *a < v).unwrap_or(profile.len());
    below(0.1).saturating_sub(below(0.9))
}

#[test]
fn a_soft_stroke_keeps_most_of_its_tips_softness() {
    // The default brush (10 % spacing), 100 px, hardness 0. Overlapping dabs build up (Flow); a
    // Gaussian-like tip keeps most of its area faint, so the stroke keeps about two thirds of the
    // tip's soft edge (half with the old smoothstep falloff).
    let b = BrushSettings { size: 100.0, hardness: 0.0, pressure_size: false, ..Default::default() };
    let stroke = cross_section(&b, &line(0.0, 400.0, 100.5), 200);
    let dab = cross_section(&b, &[StrokePoint::new(200.5, 100.5, 1.0)], 200);
    let (s, d) = (soft_edge_width(&stroke), soft_edge_width(&dab));
    assert!(d >= 20 && s * 100 >= d * 65, "stroke soft edge {s} px vs tip {d} px");
}

#[test]
fn a_tiny_dabs_coverage_does_not_depend_on_where_it_lands_in_a_pixel() {
    let total = |x: f64, y: f64| {
        let b = BrushSettings { size: 1.0, hardness: 1.0, pressure_size: false, ..Default::default() };
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, SampleType::F32, true));
        render_stroke(&mut s, &b, &[StrokePoint::new(x, y, 1.0)], None, false, 1.0);
        (0..10).flat_map(|py| (0..10).map(move |px| (px, py))).map(|(px, py)| s.rgba(px, py)[3]).sum::<f32>()
    };
    let sums: Vec<f32> = [(5.5, 5.5), (5.75, 5.5), (5.0, 5.0), (5.3, 5.8)].iter().map(|&(x, y)| total(x, y)).collect();
    let (lo, hi) = sums.iter().fold((f32::MAX, 0.0f32), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
    assert!(hi - lo < 0.03 * hi, "total coverage by sub-pixel position: {sums:?}");
}

#[test]
fn the_tip_matches_photoshops_measured_dabs() {
    use crate::tip_falloff;
    // Photoshop 27.11, single 40 px dabs (R = 20): mean alpha over rings 0.1 R wide starting at
    // the given fraction of the radius (area-weighted, as measured).
    let measured: [(f32, [(f32, f32); 4]); 3] = [
        (0.0, [(0.2, 0.861), (0.5, 0.509), (0.9, 0.129), (1.2, 0.027)]),
        (0.5, [(0.6, 0.97), (0.8, 0.561), (1.0, 0.135), (1.2, 0.012)]),
        (1.0, [(0.8, 1.0), (0.9, 0.821), (1.2, 0.0), (1.3, 0.0)]),
    ];
    for (h, rings) in measured {
        for (lo, want) in rings {
            let (mut sum, mut area) = (0.0, 0.0);
            for k in 0..64 {
                let r = (lo + 0.1 * (k as f32 + 0.5) / 64.0) * 20.0;
                sum += tip_falloff(r, 20.0, h) * r;
                area += r;
            }
            let got = sum / area;
            assert!((got - want).abs() < 0.05, "hardness {h} ring at {lo} R: {got} vs {want}");
        }
    }
    assert_eq!(tip_falloff(0.0, 10.0, 0.0), 1.0);
    assert_eq!(tip_falloff(4.0, 10.0, 0.5), 1.0, "inside the hard core");
    let v: Vec<f32> = (0..=40).map(|i| tip_falloff(i as f32 * 0.5, 10.0, 0.0)).collect();
    assert!(v.windows(2).all(|w| w[1] <= w[0]), "{v:?}");
    assert_eq!(tip_falloff(crate::tip_reach(10.0, 0.0) * 4.0, 10.0, 0.0), 0.0, "far past the reach");
}

#[test]
fn half_coverage_radius_is_the_tips_50_percent_contour() {
    use crate::{half_coverage_radius, tip_falloff};
    // The Normal Brush Tip cursor ring (#2744): where the soft falloff crosses 50 %.
    for h in [0.0, 0.25, 0.5, 0.75, 0.99] {
        let d = half_coverage_radius(10.5, h);
        assert!((tip_falloff(d, 10.5, h) - 0.5).abs() < 1e-3, "hardness {h}: ring at {d}");
    }
    // A hard tip's ring is its edge.
    assert!((half_coverage_radius(10.5, 1.0) - 10.5).abs() < 0.5);
    assert!(half_coverage_radius(10.5, f32::NAN).is_finite());
}

/// A soft, low-flow stroke (the banding case) on a white `fmt` surface, and its first channel.
fn soft_low_flow(fmt: PixelFormat) -> Surface {
    let mut s = Surface::with_default(fmt, &[1.0, 1.0, 1.0, 1.0]);
    let b = BrushSettings { size: 120.0, hardness: 0.0, flow: 0.05, opacity: 0.3, color: [0.0, 0.0, 0.0, 1.0], pressure_size: false, ..Default::default() };
    render_stroke(&mut s, &b, &line(0.0, 300.0, 100.0), None, false, 1.0);
    s
}

/// Largest error, in 8-bit levels, of `got`'s row averages against `want`'s (red channel). Along a
/// straight horizontal stroke every pixel of a row has the same true value, so plain rounding
/// gives the whole row one wrong level (a band) while dithering keeps the row's average true.
fn worst_row_error(got: &Surface, want: &Surface) -> f32 {
    let mut worst = 0.0f32;
    for y in 30..170 {
        let (mut g, mut w) = (0.0, 0.0);
        for x in 100..200 {
            g += got.rgba(x, y)[0];
            w += want.rgba(x, y)[0];
        }
        worst = worst.max((g - w).abs() / 100.0 * 255.0);
    }
    worst
}

#[test]
fn an_8bit_soft_stroke_is_dithered_instead_of_banded() {
    let reference = soft_low_flow(PixelFormat::new(ColorMode::Rgb, SampleType::F32, true));
    let eight = soft_low_flow(PixelFormat::RGBA8);
    let err = worst_row_error(&eight, &reference);
    assert!(err < 0.15, "row averages are {err} levels off the float render");
}

#[test]
fn dithering_never_moves_an_exact_colour_or_an_untouched_pixel() {
    let mut s = Surface::with_default(PixelFormat::RGBA8, &[1.0, 1.0, 1.0, 1.0]);
    let b = BrushSettings { size: 20.0, hardness: 1.0, color: [0.2, 0.4, 0.6, 1.0], pressure_size: false, ..Default::default() };
    render_stroke(&mut s, &b, &line(10.0, 90.0, 50.0), None, false, 1.0);
    let q = |v: f32| (v * 255.0).round() as i32;
    for x in 15..85 {
        let p = s.rgba(x, 50);
        assert_eq!((q(p[0]), q(p[1]), q(p[2])), (q(0.2), q(0.4), q(0.6)), "solid paint at x {x}: {p:?}");
    }
    assert_eq!(s.rgba(50, 5), [1.0, 1.0, 1.0, 1.0], "far from the stroke: untouched");
}

#[test]
fn an_aliased_stroke_at_partial_opacity_is_one_flat_colour() {
    // 30% of red 219/41/61 over white lands between 8-bit levels (244.2, 190.7, 196.8); dithering
    // used to scatter the Pencil's line over several neighbouring levels.
    let mut s = Surface::with_default(PixelFormat::RGBA8, &[1.0, 1.0, 1.0, 1.0]);
    let color = [219.0 / 255.0, 41.0 / 255.0, 61.0 / 255.0, 1.0];
    let b = BrushSettings { size: 1.0, aliased: true, opacity: 0.3, color, pressure_size: false, ..Default::default() };
    render_stroke(&mut s, &b, &line(2.0, 60.0, 5.0), None, false, 1.0);
    let first = s.rgba(3, 5);
    assert_ne!(first, [1.0, 1.0, 1.0, 1.0], "the line was painted");
    for x in 3..60 {
        assert_eq!(s.rgba(x, 5), first, "pixel {x} of the Pencil line");
    }
}
