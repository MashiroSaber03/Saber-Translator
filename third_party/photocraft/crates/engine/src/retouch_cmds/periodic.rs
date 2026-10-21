//! Bounded reads from the repeating document tile used by Pattern Preview.

use super::*;

/// At most four disjoint rectangles in the document tile. Projecting the dirty area separately
/// from the stroke bounds avoids scanning the whole document when a small dab crosses a seam.
pub(super) fn project(rect: Rect, tile: Rect) -> Vec<Rect> {
    if rect.is_empty() || tile.is_empty() {
        return Vec::new();
    }
    let axis = |a: i32, b: i32, lo: i32, hi: i32| {
        let period = i64::from(hi) - i64::from(lo);
        let length = i64::from(b) - i64::from(a);
        if length >= period {
            return vec![(lo, hi)];
        }
        let start = i64::from(lo) + (i64::from(a) - i64::from(lo)).rem_euclid(period);
        let end = start + length;
        if end <= i64::from(hi) { vec![(start as i32, end as i32)] } else { vec![(start as i32, hi), (lo, (i64::from(lo) + end - i64::from(hi)) as i32)] }
    };
    let xs = axis(rect.x0, rect.x1, tile.x0, tile.x1);
    let ys = axis(rect.y0, rect.y1, tile.y0, tile.y1);
    ys.into_iter().flat_map(|(y0, y1)| xs.iter().map(move |&(x0, x1)| Rect::new(x0, y0, x1, y1))).collect()
}

/// Composite periodic copies of one stroke, restoring each dirty patch from pre-stroke pixels.
/// Each dab wraps once, then builds flow towards its opacity ceiling. Stroke opacity and
/// selection are applied once after all dabs, including repeated visits through other tiles.
pub(super) struct Composite<'a> {
    pub target: &'a mut Surface,
    pub before: &'a Surface,
    pub tile: Rect,
    pub opacity: f32,
    pub selection: Option<&'a Surface>,
    pub lock: bool,
    pub mode: BlendMode,
}

impl Composite<'_> {
    pub fn draw(&mut self, dirty: Rect, brush: &BrushSettings, points: &[StrokePoint], mut paint: impl FnMut(Rect) -> Result<Region>) -> Result<Rect> {
        let ctx = photocraft_paint::BrushContext::new(brush);
        let mut generator = photocraft_paint::dynamics::DabGenerator::new(brush, 1.0);
        let (mut primary, mut dual) = (Vec::new(), Vec::new());
        generator.push(points, &mut primary, &mut dual);
        generator.finish(&mut primary, &mut dual);
        let primary: Vec<_> = primary
            .into_iter()
            .map(|d| {
                let r = ctx.dab_rect(&d, false);
                (d, r)
            })
            .collect();
        let dual: Vec<_> = dual
            .into_iter()
            .map(|d| {
                let r = ctx.dab_rect(&d, true);
                (d, r)
            })
            .collect();
        let mut damage = Rect::EMPTY;
        let mut work = 0_u64;
        for rect in project(dirty, self.tile) {
            let mut y = rect.y0;
            while y < rect.y1 {
                let y1 = y.saturating_add(64).min(rect.y1);
                let mut x = rect.x0;
                while x < rect.x1 {
                    let x1 = x.saturating_add(64).min(rect.x1);
                    let dst = Rect::new(x, y, x1, y1);
                    let mut stroke = PatchStroke::new(dst, self.target.channels());
                    let mut dual_stroke = PatchStroke::new(dst, 1);
                    let alpha = alpha_index(&self.target.format());
                    for (dab, bounds) in &primary {
                        if !project(*bounds, self.tile).iter().any(|r| !r.intersect(&dst).is_empty()) {
                            continue;
                        }
                        let footprint = self.fold_dab(&ctx, dab, *bounds, dst, false, &mut paint, &mut work)?;
                        stroke.accumulate(dab, &footprint, alpha);
                    }
                    for (dab, bounds) in &dual {
                        if !project(*bounds, self.tile).iter().any(|r| !r.intersect(&dst).is_empty()) {
                            continue;
                        }
                        let footprint = self.fold_dab(&ctx, dab, *bounds, dst, true, &mut paint, &mut work)?;
                        let mut dab = *dab;
                        dab.opacity = 1.0;
                        dual_stroke.accumulate(&dab, &footprint, None);
                    }
                    stroke.normalize(alpha);
                    for py in dst.y0..dst.y1 {
                        for px in dst.x0..dst.x1 {
                            let i = stroke.paint.index(px, py);
                            let depth = brush.wet_edges.then(|| stroke.depth.get(i).copied().unwrap_or(0.0));
                            let dual = brush.dual_brush.enabled.then(|| dual_stroke.coverage.get(i).copied().unwrap_or(0.0));
                            if let Some(c) = stroke.coverage.get_mut(i) {
                                *c = ctx.stroke_mask_wet(*c, depth, dual, px, py);
                            }
                        }
                    }
                    self.target.write_region(dst, &self.before.read_region(dst));
                    apply_coverage(self.target, dst, &stroke.coverage, self.opacity, self.selection, self.lock, &stroke.paint, self.mode);
                    damage = damage.union(&dst);
                    x = x1;
                }
                y = y1;
            }
        }
        Ok(damage)
    }

    #[allow(clippy::too_many_arguments)]
    fn fold_dab(
        &self,
        ctx: &photocraft_paint::BrushContext,
        dab: &photocraft_paint::Dab,
        bounds: Rect,
        dst: Rect,
        dual: bool,
        paint: &mut impl FnMut(Rect) -> Result<Region>,
        work: &mut u64,
    ) -> Result<PatchStroke> {
        let mut result = PatchStroke::new(dst, if dual { 1 } else { self.target.channels() });
        let (w, h) = (i64::from(self.tile.width()), i64::from(self.tile.height()));
        let nx0 = (i64::from(bounds.x0) - i64::from(dst.x1)).div_euclid(w) + 1;
        let nx1 = (i64::from(bounds.x1) - 1 - i64::from(dst.x0)).div_euclid(w);
        let ny0 = (i64::from(bounds.y0) - i64::from(dst.y1)).div_euclid(h) + 1;
        let ny1 = (i64::from(bounds.y1) - 1 - i64::from(dst.y0)).div_euclid(h);
        let (mut values, mut depths) = (Vec::new(), Vec::new());
        let wet = ctx.brush.wet_edges && !dual;
        for ny in ny0..=ny1 {
            for nx in nx0..=nx1 {
                let (dx, dy) = (nx * w, ny * h);
                let r = Rect::new(
                    (i64::from(dst.x0) + dx).max(i64::from(bounds.x0)) as i32,
                    (i64::from(dst.y0) + dy).max(i64::from(bounds.y0)) as i32,
                    (i64::from(dst.x1) + dx).min(i64::from(bounds.x1)) as i32,
                    (i64::from(dst.y1) + dy).min(i64::from(bounds.y1)) as i32,
                );
                *work = work.saturating_add(r.width() as u64 * r.height() as u64);
                if *work > 64 * 1024 * 1024 {
                    return Err(bad("paint.cloneStamp", "wrapped stroke exceeds its coverage budget"));
                }
                ctx.rasterize_into(dab, dual, r, &mut values, wet.then_some(&mut depths));
                let mut sampled = None;
                for sy in r.y0..r.y1 {
                    for sx in r.x0..r.x1 {
                        let source_index = ((sy - r.y0) as usize) * r.width() as usize + (sx - r.x0) as usize;
                        let value = values.get(source_index).copied().unwrap_or(0.0);
                        if value <= 0.0 {
                            continue;
                        }
                        let (tx, ty) = ((i64::from(sx) - dx) as i32, (i64::from(sy) - dy) as i32);
                        let i = result.paint.index(tx, ty);
                        if let Some(depth) = result.depth.get_mut(i) {
                            *depth = depth.max(depths.get(source_index).copied().unwrap_or(0.0));
                        }
                        let Some(previous) = result.coverage.get_mut(i) else { continue };
                        if value <= *previous {
                            continue;
                        }
                        if !dual {
                            if sampled.is_none() {
                                sampled = Some(paint(r)?);
                            }
                            if let Some(source) = &sampled {
                                result.paint.px_mut(tx, ty).copy_from_slice(source.px(sx, sy));
                            }
                        }
                        *previous = value;
                    }
                }
            }
        }
        Ok(result)
    }
}

struct PatchStroke {
    coverage: Vec<f32>,
    depth: Vec<f32>,
    // Native colours weighted by coverage and source alpha; alpha weighted by coverage.
    paint: Region,
}

impl PatchStroke {
    fn new(rect: Rect, channels: usize) -> Self {
        let len = rect.width() as usize * rect.height() as usize;
        Self { coverage: vec![0.0; len], depth: vec![0.0; len], paint: Region::new(rect, channels) }
    }

    fn accumulate(&mut self, dab: &photocraft_paint::Dab, footprint: &PatchStroke, alpha: Option<usize>) {
        let rect = self.paint.rect;
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let i = self.paint.index(x, y);
                let Some(c) = self.coverage.get_mut(i) else { continue };
                let value = footprint.coverage.get(i).copied().unwrap_or(0.0);
                if value <= 0.0 {
                    continue;
                }
                if let Some(depth) = self.depth.get_mut(i) {
                    *depth = depth.max(footprint.depth.get(i).copied().unwrap_or(0.0));
                }
                if dab.opacity <= *c {
                    continue;
                }
                let next = *c + value * (dab.opacity - *c);
                let weight = next - *c;
                let source = footprint.paint.px(x, y);
                let sa = alpha.and_then(|a| source.get(a)).copied().unwrap_or(1.0);
                for (channel, (dst, src)) in self.paint.px_mut(x, y).iter_mut().zip(source).enumerate() {
                    *dst += if Some(channel) == alpha { weight * sa } else { weight * sa * src };
                }
                *c = next;
            }
        }
    }

    fn normalize(&mut self, alpha: Option<usize>) {
        let rect = self.paint.rect;
        for y in rect.y0..rect.y1 {
            for x in rect.x0..rect.x1 {
                let i = self.paint.index(x, y);
                let coverage = self.coverage.get(i).copied().unwrap_or(0.0);
                let px = self.paint.px_mut(x, y);
                let weight = alpha.and_then(|a| px.get(a)).copied().unwrap_or(coverage);
                for (channel, value) in px.iter_mut().enumerate() {
                    let denominator = if Some(channel) == alpha { coverage } else { weight };
                    *value = if denominator > 0.0 { *value / denominator } else { 0.0 };
                }
            }
        }
    }
}

pub(super) fn sample(pre: &Document, id: Option<LayerId>, surf: &Surface, which: SampleLayers, rect: Rect, cmd: &str) -> Result<Region> {
    let tile = pre.bounds();
    if tile.is_empty() {
        return Err(bad(cmd, "cannot repeat an empty document"));
    }
    if rect.is_empty() {
        return Ok(Region::new(Rect::EMPTY, surf.channels()));
    }
    let (w, h, n) = (rect.width() as usize, rect.height() as usize, surf.channels());
    let len = w
        .checked_mul(h)
        .and_then(|v| v.checked_mul(n))
        .filter(|&v| v <= 16 * 1024 * 1024)
        .ok_or_else(|| bad(cmd, "periodic source window exceeds its 64 MiB budget"))?;
    let mut out = Region { rect, ch: n, data: vec![0.0; len] };
    // Split at tile seams. Read only the required rectangles, including the bilinear apron,
    // rather than allocating or compositing the entire document for a small clone dab.
    let mut dy = 0;
    while dy < h {
        let sy = (i64::from(rect.y0) + dy as i64 - i64::from(tile.y0)).rem_euclid(i64::from(tile.height())) as i32 + tile.y0;
        let rows = (h - dy).min((tile.y1 - sy) as usize);
        let mut dx = 0;
        while dx < w {
            let sx = (i64::from(rect.x0) + dx as i64 - i64::from(tile.x0)).rem_euclid(i64::from(tile.width())) as i32 + tile.x0;
            let cols = (w - dx).min((tile.x1 - sx) as usize);
            let part = super::sample(pre, id, surf, which, Rect::new(sx, sy, sx + cols as i32, sy + rows as i32));
            for (dst, src) in out.data.chunks_exact_mut(w * n).skip(dy).take(rows).zip(part.data.chunks_exact(cols * n)) {
                if let Some(dst) = dst.get_mut(dx * n..(dx + cols) * n) {
                    dst.copy_from_slice(src);
                }
            }
            dx += cols;
        }
        dy += rows;
    }
    Ok(out)
}
