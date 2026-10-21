//! Frozen scalar rasterizer used only as a pixel oracle for the computed-tip fast path.
use super::*;
use crate::tile::GrayTile;
use photocraft_geom::Point;

impl BrushContext {
    fn rasterize_reference(&self, d: &Dab, dual: bool, rect: Rect, out: &mut Vec<f32>, mut depth: Option<&mut Vec<f32>>) {
        let (w, h) = (rect.width() as usize, rect.height() as usize);
        out.clear();
        out.resize(w * h, 0.0);
        if let Some(dp) = depth.as_deref_mut() {
            dp.clear();
            dp.resize(w * h, 0.0);
        }
        let b = &self.brush;
        let (hardness, mips) = if dual { (b.dual_brush.hardness, self.dual_tip.as_deref()) } else { (b.hardness, self.tip.as_deref()) };
        let aliased = b.aliased && !dual;
        let wet = b.wet_edges && !dual && depth.is_none();
        let noise = b.noise && !dual;
        let tex_tip = !dual && b.texture.enabled && b.texture.each_tip && self.texture.is_some();
        let (cx, cy) = if aliased { grid_center(d.center.x, d.center.y, 2.0 * d.radius) } else { (d.center.x as f32, d.center.y as f32) };
        let (sn, cs) = d.angle.sin_cos();
        let (fx, fy) = (if d.flip_x { -1.0 } else { 1.0 }, if d.flip_y { -1.0 } else { 1.0 });
        // Brush Projection: stretch the sampling coordinates along the tilt direction, which
        // foreshortens the tip there by `proj_scale`.
        let proj = (!dual && d.proj_scale < 0.999).then(|| {
            let (ps, pc) = d.proj_angle.sin_cos();
            (pc, ps, 1.0 / d.proj_scale.max(0.05) - 1.0)
        });
        let r = d.radius;
        let ro = d.roundness.max(0.5 / r).min(1.0);
        let rm = (r * ro).max(0.5);
        let reach = if aliased || mips.is_some() { rm } else { crate::tip_reach(rm, hardness).max(rm) };
        let reach2 = (reach + 0.5) * (reach + 0.5);
        // Sampled-tip mapping.
        let (level, inv_scale, tw, th) = match mips {
            Some(m) => {
                let (w0, h0, _) = &m.levels[0];
                let big = *w0.max(h0) as f32;
                (m.level_for(2.0 * r * ro.max(0.25)), big / (2.0 * r), *w0 as f32, *h0 as f32)
            }
            None => (0, 1.0, 1.0, 1.0),
        };
        // A pixel offset from the dab centre in tip space: to y-up, project, rotate by -angle, flip.
        let to_tip = |dx: f32, dy: f32| {
            let (mut ux, mut uy) = (dx, -dy);
            if let Some((ax, ay, k)) = proj {
                let t = (ux * ax + uy * ay) * k;
                ux += t * ax;
                uy += t * ay;
            }
            ((ux * cs + uy * sn) * fx, (-ux * sn + uy * cs) * fy)
        };
        // A tip a few pixels across takes each pixel's covered area (a 4×4 grid inside it):
        // sampling only the pixel centre made a tiny dab's total coverage swing with its sub-pixel
        // position, and thin lines beaded.
        let small = mips.is_none() && !aliased && rm < SMALL_TIP_RADIUS;
        let small_reach2 = (reach + 0.75) * (reach + 0.75);
        for yy in 0..h {
            let y = rect.y0 + yy as i32;
            let dy = y as f32 + 0.5 - cy;
            for xx in 0..w {
                let x = rect.x0 + xx as i32;
                let dx = x as f32 + 0.5 - cx;
                let (u, v) = to_tip(dx, dy);
                let (mut val, rn) = match mips {
                    // Block: the whole-pixel square, hard (the Eraser's Block mode).
                    _ if aliased && b.square => {
                        if dx.abs() >= r || dy.abs() >= r {
                            continue;
                        }
                        (1.0, 0.0)
                    }
                    None if small => {
                        let d2 = (u * ro).powi(2) + v * v;
                        if d2 >= small_reach2 {
                            continue;
                        }
                        let mut sum = 0.0;
                        for sy in SUBPIXEL {
                            for sx in SUBPIXEL {
                                let (su, sv) = to_tip(dx + sx, dy + sy);
                                let sd = ((su * ro).powi(2) + sv * sv).sqrt();
                                sum += tip_falloff(sd, rm, hardness);
                            }
                        }
                        (sum / (SUBPIXEL.len() * SUBPIXEL.len()) as f32, d2.sqrt() / rm)
                    }
                    None => {
                        let d2 = (u * ro).powi(2) + v * v;
                        if d2 >= reach2 {
                            continue;
                        }
                        let dd = d2.sqrt();
                        let val = if aliased { if dd <= rm { 1.0 } else { 0.0 } } else { dab_coverage(dd, rm, hardness) };
                        (val, dd / rm)
                    }
                    Some(m) => {
                        let (px, py) = (u * inv_scale + tw / 2.0, -(v / ro) * inv_scale + th / 2.0);
                        if px < -1.0 || py < -1.0 || px > tw + 1.0 || py > th + 1.0 {
                            continue;
                        }
                        let s = m.sample_clamped(level, px / tw, py / th);
                        let val = if aliased { if s >= 0.5 { 1.0 } else { 0.0 } } else { s };
                        (val, 1.0 - s)
                    }
                };
                if val <= 0.0 {
                    continue;
                }
                if noise && val < 1.0 {
                    let n = hash2(x, y, NOISE_SALT);
                    let hard = if n < val { 1.0 } else { 0.0 };
                    val += (hard - val) * 0.7;
                }
                if tex_tip {
                    val = mask_combine(b.texture.mode, val, self.texture_at(x, y), d.depth);
                }
                if wet {
                    val *= wet_factor(rn);
                }
                if let Some(slot) = depth.as_deref_mut().and_then(|dp| dp.get_mut(yy * w + xx)) {
                    *slot = (1.0 - rn).clamp(0.0, 1.0);
                }
                out[yy * w + xx] = val * d.alpha;
            }
        }
    }
}

fn compare(ctx: &BrushContext, dab: &Dab, dual: bool, rect: Rect, with_depth: bool) {
    // Reused, dirty buffers also check clearing and resizing.
    let (mut actual, mut expected) = (vec![0.75; 19], vec![0.75; 19]);
    let (mut actual_depth, mut expected_depth) = (vec![0.25; 23], vec![0.25; 23]);
    ctx.rasterize_into(dab, dual, rect, &mut actual, with_depth.then_some(&mut actual_depth));
    ctx.rasterize_reference(dab, dual, rect, &mut expected, with_depth.then_some(&mut expected_depth));
    for (label, a, b) in [("coverage", actual, expected), ("depth", actual_depth, expected_depth)] {
        assert_eq!(a.len(), b.len());
        let mismatch = a.iter().zip(&b).enumerate().find(|(_, (a, b))| a.to_bits() != b.to_bits());
        assert!(mismatch.is_none(), "{label} mismatch {mismatch:?}; {dab:?}; dual={dual}, rect={rect:?}");
    }
}

#[test]
fn computed_tips_match_scalar_bits() {
    for aliased in [false, true] {
        for hardness in [-0.5, 0.0, 0.1, 0.5, 0.9999, 1.0, 1.5] {
            let ctx = BrushContext::new(&BrushSettings { aliased, hardness, ..Default::default() });
            for radius in [0.5, 1.0, 2.99, 3.0, 3.01, 15.0, 31.5] {
                for roundness in [0.0, 0.1, 0.4, 1.0] {
                    for alpha in [-0.5, -0.0, 0.0, 0.37, 1.0] {
                        let mut d = Dab::round(Point::new(-31.25, 63.75), radius, alpha);
                        d.roundness = roundness;
                        d.flip_x = true;
                        d.flip_y = true;
                        compare(&ctx, &d, false, ctx.dab_rect(&d, false), false);
                    }
                }
            }
        }
    }
}

#[test]
fn large_corner_dabs_keep_off_canvas_coverage() {
    for (hardness, aliased) in [(0.0, false), (1.0, false), (1.0, true)] {
        let ctx = BrushContext::new(&BrushSettings { hardness, aliased, ..Default::default() });
        let d = Dab::round(Point::new(-1050.25, -1050.75), 1500.0, 0.73);
        let rect = ctx.dab_rect(&d, false);
        compare(&ctx, &d, false, rect, false);
        let mut coverage = Vec::new();
        ctx.rasterize(&d, false, rect, &mut coverage);
        // The layer is an infinite plane: paint outside the visible document must survive.
        assert!(coverage.iter().enumerate().any(|(i, v)| *v > 0.0 && rect.x0 + ((i % rect.width() as usize) as i32) < 0));
        compare(&ctx, &d, false, Rect::new(0, 0, 32, 32), false);
        compare(&ctx, &d, false, Rect::new(0, 0, 0, 0), false);
    }
}

#[test]
fn feature_and_exceptional_dabs_match_scalar_bits() {
    let tip = GrayTile::from_fn(8, 5, |x, y| ((x + y) % 7) as f32 / 6.0);
    for feature in 0..7 {
        let mut b = BrushSettings { hardness: 0.25, ..Default::default() };
        match feature {
            0 => b.noise = true,
            1 => b.wet_edges = true,
            2 => {
                b.aliased = true;
                b.square = true;
            }
            3 => b.tip = TipShape::Sampled(tip.clone()),
            4 => {
                b.texture.enabled = true;
                b.texture.each_tip = true;
                b.texture.pattern = Pattern::Tile(tip.clone());
            }
            5 => b.dual_brush.tip = TipShape::Sampled(tip.clone()),
            _ => {}
        }
        let ctx = BrushContext::new(&b);
        for dual in [false, true] {
            for with_depth in [false, true] {
                for (angle, proj_scale) in [(0.0, 1.0), (0.7, 1.0), (0.0, 0.6)] {
                    let mut d = Dab::round(Point::new(63.5, -0.25), 15.0, 0.47);
                    d.angle = angle;
                    d.roundness = 0.4;
                    d.proj_scale = proj_scale;
                    compare(&ctx, &d, dual, ctx.dab_rect(&d, dual), with_depth);
                }
            }
        }
    }
    let ctx = BrushContext::new(&BrushSettings::default());
    for value in [f32::NAN, f32::from_bits(0x7f80_0123), f32::INFINITY, f32::NEG_INFINITY] {
        for field in 0..4 {
            let mut d = Dab::round(Point::new(0.0, 0.0), 15.0, 1.0);
            match field {
                0 => d.alpha = value,
                1 => d.angle = value,
                2 => d.radius = value,
                _ => d.roundness = value,
            }
            compare(&ctx, &d, false, Rect::new(-4, -4, 4, 4), false);
        }
        let ctx = BrushContext::new(&BrushSettings { hardness: value, ..Default::default() });
        compare(&ctx, &Dab::round(Point::new(0.0, 0.0), 15.0, 1.0), false, Rect::new(-16, -16, 16, 16), false);
    }
}

#[test]
fn computed_tips_match_at_pixel_and_float_boundaries() {
    for aliased in [false, true] {
        for center in [Point::new(0.0, -0.0), Point::new(0.5, 0.5), Point::new(255.9999, -256.0001), Point::new(16_777_216.25, -16_777_216.75)] {
            for radius in [3.0, 3.5, 15.25, 113.75] {
                for hardness in [0.0, 0.6, 1.0] {
                    let ctx = BrushContext::new(&BrushSettings { hardness, aliased, ..Default::default() });
                    let mut d = Dab::round(center, radius, 0.137);
                    d.roundness = 0.375;
                    compare(&ctx, &d, false, ctx.dab_rect(&d, false), false);
                }
            }
        }
    }
}
