//! Scatter each source streak once, retaining the old nearest-source tie rule.

use super::{read_px, wind_reach};
use crate::fxutil::{MAXC, luma};
use crate::image::Image;
use crate::noise::hash01;
use crate::{Ctx, WindMethod};
use photocraft_geom::Rect;

const MAX_ROW: usize = 1_048_576;

pub(super) fn streaks(src: &Image, out: Rect, ctx: &Ctx, method: WindMethod, from_right: bool, seed: u32) -> Option<Vec<f32>> {
    if method == WindMethod::Stagger || src.ch == 0 || src.ch > MAXC {
        return None;
    }
    let n = src.ch;
    let w = out.width() as usize;
    if w == 0 || w > MAX_ROW || out.is_empty() {
        return None;
    }
    let lmax = wind_reach(method) as i32;
    let d = if from_right { -1 } else { 1 };
    // Include the upstream halo and one more source for the edge difference.
    let start = if from_right { out.x1.checked_add(lmax - 1)? } else { out.x0.checked_sub(lmax)? };
    start.checked_sub(d)?;
    let sources = w.checked_add(lmax as usize)?;
    let mut best = Vec::new();
    best.try_reserve_exact(w).ok()?;
    best.resize(w, (0.0f32, lmax + 1));
    let mut res = src.crop(out);
    let mut tmp = [0.0f32; MAXC];
    let strength = if method == WindMethod::Blast { 8.0 } else { 4.0 };
    for y in out.y0..out.y1 {
        best.fill((0.0, lmax + 1));
        read_px(src, start - d, y, &mut tmp[..n]);
        let mut prev = luma(ctx, &tmp[..n]);
        for step in 0..sources {
            let s = i32::try_from(i64::from(start) + i64::from(d) * i64::try_from(step).ok()?).ok()?;
            read_px(src, s, y, &mut tmp[..n]);
            let lum = luma(ctx, &tmp[..n]);
            let edge = ((lum - prev).abs() * strength).min(1.0);
            prev = lum;
            if edge <= 0.0 {
                continue;
            }
            let len = (hash01(s, y, 44, seed).powi(2) * lmax as f32).max(1.0);
            for k in 1..=(len.floor() as i32).min(lmax) {
                let x = s.checked_add(d * k)?;
                if x < out.x0 || x >= out.x1 {
                    continue;
                }
                let weight = edge * (1.0 - k as f32 / (len + 1.0));
                let hit = best.get_mut((x - out.x0) as usize)?;
                // Sources are visited downwind; equal weights prefer the smallest original k.
                if weight > hit.0 || (weight == hit.0 && k < hit.1) {
                    *hit = (weight, k);
                }
            }
        }
        let row = (y - out.y0) as usize * w * n;
        for (x, &(weight, k)) in best.iter().enumerate() {
            if weight > 0.0 {
                let sx = i32::try_from(i64::from(out.x0) + x as i64 - i64::from(d * k)).ok()?;
                read_px(src, sx, y, &mut tmp[..n]);
                let px = res.get_mut(row + x * n..row + (x + 1) * n)?;
                for (v, s) in px.iter_mut().zip(&tmp) {
                    *v += (*s - *v) * weight;
                }
            }
        }
    }
    Some(res)
}
