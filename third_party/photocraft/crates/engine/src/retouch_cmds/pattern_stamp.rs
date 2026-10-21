//! Pattern Stamp: paint with the current pattern through a brush stroke.
//!
//! Aligned strokes lock the tile origin in document space. Unaligned strokes restart the origin at
//! the first point of that stroke. Impressionist jitters each dab's phase with a seeded RNG so a
//! replay of the same points is identical.

use photocraft_color::{BlendMode, PixelFormat};
use photocraft_compose::pattern::{Placement, Tile, render};
use photocraft_doc::Pattern;
use photocraft_geom::Rect;
use photocraft_paint::retouch::{Footprint, Region, apply_coverage, apply_dab_stroke, over_native, stroke_coverage};
use photocraft_paint::{Stroke, StrokePoint};
use photocraft_raster::{Surface, from_rgba_into, to_rgba};
use serde_json::{Value, json};

use super::{bad, blend_param, damage_json, flag, has_pixel_layer, parse_brush};
use crate::{Result, Session};

/// A single stroke's accumulated coverage must stay at or under this many pixels (8192², so a
/// stroke corner to corner across a 24–36 MP canvas fits). Larger footprints return an error
/// instead of allocating.
const MAX_STROKE_PIXELS: u64 = 67_108_864;

const STREAM_PHASE_X: u64 = 101;
const STREAM_PHASE_Y: u64 = 102;

pub(super) fn enabled(s: &Session) -> std::result::Result<(), String> {
    has_pixel_layer(s)?;
    if crate::pattern_cmds::resolve(s, "").is_none() { Err("no pattern selected".into()) } else { Ok(()) }
}

pub(super) fn run(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "paint.patternStamp";
    let (mut stroke, id) = parse_brush(s, p, CMD)?;
    let mode = blend_param(p, CMD)?;
    let aligned = flag(p, "aligned", true);
    let impressionist = flag(p, "impressionist", false);
    let pat = crate::pattern_cmds::resolve_param(s, CMD, p)?;
    if pat.is_empty() {
        return Err(bad(CMD, "the pattern is empty"));
    }
    let scale = stamp_scale(p, CMD)?;
    let angle = stamp_angle(p, CMD)?;
    let Some(first) = stroke.points.first() else {
        return Err(bad(CMD, "`points` is empty"));
    };
    let phase = stamp_phase(p, CMD, aligned, first)?;
    let tile = Tile::new(&pat).ok_or_else(|| bad(CMD, "the pattern is empty or too large to paint"))?;
    reject_huge_stroke(CMD, &stroke)?;

    let seed = p.get("seed").and_then(Value::as_u64).unwrap_or_else(|| {
        let bytes = p.get("points").map(std::string::ToString::to_string).unwrap_or_default();
        photocraft_paint::rng::seed_from_bytes(bytes.as_bytes())
    });
    stroke.brush.seed = seed;

    let dmg = s.edit("Pattern Stamp", |doc, _| {
        crate::pattern_cmds::ensure_in_doc(doc, &pat);
        let pre = doc.clone();
        let (surf, lock) = crate::channel_cmds::target_surface(doc, id, p)?;
        let sel = pre.selection.as_ref();
        if impressionist {
            paint_impressionist(surf, &stroke, sel, lock, &tile, &pat, phase, scale, angle, seed, mode)
        } else {
            paint_aligned(surf, &stroke, sel, lock, &tile, phase, scale, angle, mode, CMD)
        }
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = Some(dmg);
    }
    Ok(json!({
        "damage": damage_json(dmg),
        "phase": [phase.0, phase.1],
        "aligned": aligned,
        "pattern": pat.id,
    }))
}

fn stamp_scale(p: &Value, cmd: &str) -> Result<f32> {
    match p.get("scale") {
        None => Ok(1.0),
        Some(v) => {
            let s = v.as_f64().ok_or_else(|| bad(cmd, "`scale` must be a number (1..1000 %)"))?;
            if !s.is_finite() || !(1.0..=1000.0).contains(&s) {
                return Err(bad(cmd, "scale must be 1..1000 %"));
            }
            Ok((s as f32) / 100.0)
        }
    }
}

fn stamp_angle(p: &Value, cmd: &str) -> Result<f32> {
    match p.get("angle") {
        None => Ok(0.0),
        Some(v) => {
            let a = v.as_f64().ok_or_else(|| bad(cmd, "`angle` must be a number (degrees)"))?;
            if !a.is_finite() {
                return Err(bad(cmd, "`angle` must be finite"));
            }
            Ok(a as f32)
        }
    }
}

fn stamp_phase(p: &Value, cmd: &str, aligned: bool, first: &StrokePoint) -> Result<(f32, f32)> {
    match p.get("phase") {
        None => {
            if aligned {
                Ok((0.0, 0.0))
            } else {
                Ok((first.x as f32, first.y as f32))
            }
        }
        Some(v) => {
            let a = v.as_array().ok_or_else(|| bad(cmd, "`phase` must be [px, py]"))?;
            let x = a.first().and_then(Value::as_f64).ok_or_else(|| bad(cmd, "`phase` must be [px, py]"))?;
            let y = a.get(1).and_then(Value::as_f64).ok_or_else(|| bad(cmd, "`phase` must be [px, py]"))?;
            if !x.is_finite() || !y.is_finite() {
                return Err(bad(cmd, "`phase` coordinates must be finite"));
            }
            Ok((x as f32, y as f32))
        }
    }
}

fn reject_huge_stroke(cmd: &str, stroke: &Stroke) -> Result<()> {
    let pad = f64::from(stroke.brush.size.max(1.0)).mul_add(0.5, 4.0);
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for q in &stroke.points {
        min_x = min_x.min(q.x);
        min_y = min_y.min(q.y);
        max_x = max_x.max(q.x);
        max_y = max_y.max(q.y);
    }
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return Err(bad(cmd, "point coordinates must be finite"));
    }
    let w = (max_x - min_x + pad * 2.0).ceil().max(1.0);
    let h = (max_y - min_y + pad * 2.0).ceil().max(1.0);
    if !w.is_finite() || !h.is_finite() || w > 1.0e9 || h > 1.0e9 {
        return Err(bad(cmd, format!("stroke too large (at most {MAX_STROKE_PIXELS} px)")));
    }
    let area = (w as u64).saturating_mul(h as u64);
    if area > MAX_STROKE_PIXELS {
        return Err(bad(cmd, format!("stroke too large ({w:.0}×{h:.0} px, at most {MAX_STROKE_PIXELS})")));
    }
    Ok(())
}

fn reject_bounds(cmd: &str, bounds: Rect) -> Result<()> {
    if bounds.is_empty() {
        return Ok(());
    }
    let (w, h) = (u64::from(bounds.width()), u64::from(bounds.height()));
    let area = w.saturating_mul(h);
    if area > MAX_STROKE_PIXELS {
        return Err(bad(cmd, format!("stroke too large ({w}×{h} px, at most {MAX_STROKE_PIXELS})")));
    }
    Ok(())
}

fn placement(phase: (f32, f32), scale: f32, angle: f32) -> Placement {
    Placement::new(Rect::EMPTY, false, phase, scale, angle)
}

fn rgba_to_region(fmt: &PixelFormat, rect: Rect, rgba: &[[f32; 4]]) -> Region {
    let n = fmt.channels();
    let mut paint = Region::new(rect, n);
    let mut enc = [0.0f32; 8];
    for (i, px) in rgba.iter().enumerate() {
        from_rgba_into(fmt, *px, &mut enc);
        let Some(start) = i.checked_mul(n) else { continue };
        let Some(end) = start.checked_add(n) else { continue };
        if let (Some(dst), Some(src)) = (paint.data.get_mut(start..end), enc.get(..n)) {
            dst.copy_from_slice(src);
        }
    }
    paint
}

#[allow(clippy::too_many_arguments)]
fn paint_aligned(
    surf: &mut Surface,
    stroke: &Stroke,
    sel: Option<&Surface>,
    lock: bool,
    tile: &Tile,
    phase: (f32, f32),
    scale: f32,
    angle: f32,
    mode: BlendMode,
    cmd: &str,
) -> Result<Rect> {
    let (bounds, cov) = stroke_coverage(stroke);
    reject_bounds(cmd, bounds)?;
    if bounds.is_empty() {
        return Ok(Rect::EMPTY);
    }
    let place = placement(phase, scale, angle);
    let rgba = render(tile, &place, bounds);
    let paint = rgba_to_region(&surf.format(), bounds, &rgba);
    Ok(apply_coverage(surf, bounds, &cov, stroke.brush.opacity, sel, lock, &paint, mode))
}

#[allow(clippy::too_many_arguments)]
fn paint_impressionist(
    surf: &mut Surface,
    stroke: &Stroke,
    sel: Option<&Surface>,
    lock: bool,
    tile: &Tile,
    pat: &Pattern,
    phase: (f32, f32),
    scale: f32,
    angle: f32,
    seed: u64,
    mode: BlendMode,
) -> Result<Rect> {
    let fmt = surf.format();
    let tw = pat.width as f32 * scale;
    let th = pat.height as f32 * scale;
    // Halo is the dab footprint only: Impressionist jitters phase, it does not blur.
    Ok(apply_dab_stroke(surf, stroke, sel, lock, 0, |work, fp| {
        stamp_dab(work, &fmt, tile, fp, phase, scale, angle, seed, tw, th, mode);
    }))
}

fn footprint_cov(fp: &Footprint, x: i32, y: i32) -> f32 {
    if x < fp.rect.x0 || y < fp.rect.y0 {
        return 0.0;
    }
    let dx = x.wrapping_sub(fp.rect.x0) as u32;
    let dy = y.wrapping_sub(fp.rect.y0) as u32;
    let w = fp.rect.width();
    let idx = (dy as usize).saturating_mul(w as usize).saturating_add(dx as usize);
    fp.cov.get(idx).copied().unwrap_or(0.0)
}

#[allow(clippy::too_many_arguments)]
fn stamp_dab(
    work: &mut Region,
    fmt: &PixelFormat,
    tile: &Tile,
    fp: &Footprint,
    phase: (f32, f32),
    scale: f32,
    angle: f32,
    seed: u64,
    tw: f32,
    th: f32,
    mode: BlendMode,
) {
    let r = fp.rect.intersect(&work.rect);
    if r.is_empty() {
        return;
    }
    let i = fp.index as u64;
    let jx = photocraft_paint::rng::rand_signed(seed, i, STREAM_PHASE_X) * tw;
    let jy = photocraft_paint::rng::rand_signed(seed, i, STREAM_PHASE_Y) * th;
    let place = placement((phase.0 + jx, phase.1 + jy), scale, angle);
    let rgba = render(tile, &place, r);
    let rw = r.width() as usize;
    let n = work.ch;
    let mut enc = [0.0f32; 8];
    for y in r.y0..r.y1 {
        for x in r.x0..r.x1 {
            let k = footprint_cov(fp, x, y);
            if k <= 0.0 {
                continue;
            }
            let ox = (x - r.x0) as usize;
            let oy = (y - r.y0) as usize;
            let Some(px) = rgba.get(oy.saturating_mul(rw).saturating_add(ox)) else { continue };
            from_rgba_into(fmt, *px, &mut enc);
            let dst = work.px_mut(x, y);
            if mode == BlendMode::Normal {
                if let Some(src) = enc.get(..n) {
                    over_native(fmt, dst, src, k, false);
                }
            } else {
                let d = to_rgba(fmt, dst);
                let o = photocraft_color::blend::composite(mode, d, *px, k);
                from_rgba_into(fmt, o, &mut enc);
                if let Some(src) = enc.get(..n) {
                    dst.copy_from_slice(src);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stroke(points: &[(f64, f64)], size: f32) -> Stroke {
        let brush = photocraft_paint::BrushSettings { size, ..Default::default() };
        Stroke { brush, points: points.iter().map(|&(x, y)| StrokePoint::new(x, y, 1.0)).collect() }
    }

    #[test]
    fn a_diagonal_across_a_24_mp_canvas_is_not_too_large() {
        assert!(reject_huge_stroke("t", &stroke(&[(0.0, 0.0), (6015.0, 3999.0)], 100.0)).is_ok());
        assert!(reject_bounds("t", Rect::from_xywh(-60, -60, 6136, 4120)).is_ok());
        assert!(reject_huge_stroke("t", &stroke(&[(0.0, 0.0), (9000.0, 9000.0)], 100.0)).is_err());
        assert!(reject_bounds("t", Rect::from_xywh(0, 0, 9000, 9000)).is_err());
    }
}
