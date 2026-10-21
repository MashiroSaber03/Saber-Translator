//! Select and Mask's Refine Edge Brush: strokes that set where edge refinement happens.
//!
//! Painting (`add`) marks an area as unknown: its coverage is estimated from the image, so soft
//! detail such as hair beyond the edge-detection band is picked up. Painting with `add = false`
//! (Photoshop's Alt) removes the refinement there and keeps the original edge. Strokes apply in
//! order; a later one paints over an earlier one.
//!
//! The estimate projects each painted pixel's colour on the line between the mean colours of the
//! known foreground and background around it (normalised convolution over a box twice the brush
//! size), the colour-line model of A. Levin, D. Lischinski and Y. Weiss, "A Closed-Form Solution
//! to Natural Image Matting" (CVPR 2006), with one foreground and one background colour per window.

use photocraft_geom::Rect;

use super::box_mean;
use crate::segment::RgbImage;

/// One Refine Edge Brush stroke, in document pixels.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeStroke {
    /// The stroke's path (a single point paints a dab).
    pub points: Vec<(f32, f32)>,
    /// Brush diameter in pixels.
    pub size: f32,
    /// Add to the refinement area (true) or restore the original edge (false).
    pub add: bool,
}

/// Brush diameters the strokes may use.
pub const SIZE_RANGE: std::ops::RangeInclusive<f32> = 1.0..=5000.0;

/// The painted area: signed coverage (`+1` refined, `−1` restored, `0` untouched) over `bbox`.
#[derive(Clone, Debug, PartialEq)]
pub struct EdgeBrush {
    bbox: Rect,
    /// Coverage × 127, row-major over `bbox`.
    v: Vec<i8>,
    /// Bounds of the added (positive) coverage.
    added: Rect,
    /// Box radius of the colour estimate (pixels).
    window: usize,
}

/// Coverage (0–1) of a round brush of radius `r` swept from `a` to `b`, at pixel centre `p`.
fn capsule(p: (f32, f32), a: (f32, f32), b: (f32, f32), r: f32) -> f32 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    let (cx, cy) = (a.0 + t * dx, a.1 + t * dy);
    let d = ((p.0 - cx).powi(2) + (p.1 - cy).powi(2)).sqrt();
    (r + 0.5 - d).clamp(0.0, 1.0)
}

/// The pixels a disc of radius `r` at `p` can touch, as a (possibly empty) rectangle.
fn dab_bounds(p: (f32, f32), r: f32) -> Rect {
    let lo = |v: f32| (v - r - 1.0).floor().clamp(i32::MIN as f32 / 2.0, i32::MAX as f32 / 2.0) as i32;
    let hi = |v: f32| (v + r + 1.0).ceil().clamp(i32::MIN as f32 / 2.0, i32::MAX as f32 / 2.0) as i32;
    Rect::new(lo(p.0), lo(p.1), hi(p.0), hi(p.1))
}

/// Most pixels the strokes of one [`EdgeBrush`] may sweep (Σ length × size + size² per stroke):
/// about eleven times a 24 MP canvas.
pub const MAX_SWEPT_AREA: f64 = (1u64 << 28) as f64;

/// The area `strokes` sweep (see [`MAX_SWEPT_AREA`]); non-finite strokes count as infinite.
pub fn swept_area(strokes: &[EdgeStroke]) -> f64 {
    strokes
        .iter()
        .map(|s| {
            let size = f64::from(s.size.clamp(*SIZE_RANGE.start(), *SIZE_RANGE.end()));
            let len: f64 = s.points.windows(2).filter_map(|w| Some(f64::from((w.get(1)?.0 - w.first()?.0).hypot(w.get(1)?.1 - w.first()?.1)))).sum();
            if len.is_finite() { len * size + size * size } else { f64::INFINITY }
        })
        .sum()
}

/// The stroke's path as short segments: points closer than a quarter of the radius to the last
/// kept one are dropped, and long segments are split at most `r` long, so each segment's box
/// stays close to the area it paints.
fn segments(points: &[(f32, f32)], r: f32) -> Vec<((f32, f32), (f32, f32))> {
    let Some(&first) = points.first() else { return Vec::new() };
    let min_step = (r / 4.0).max(0.5);
    let max_step = r.max(2.0);
    let mut kept = vec![first];
    for p in points.iter().skip(1) {
        let last = kept.last().copied().unwrap_or(first);
        if (p.0 - last.0).hypot(p.1 - last.1) >= min_step {
            kept.push(*p);
        }
    }
    if let (Some(last), Some(end)) = (kept.last().copied(), points.last().copied())
        && last != end
    {
        kept.push(end);
    }
    if kept.len() == 1 {
        return vec![(first, first)];
    }
    let mut out = Vec::new();
    for w in kept.windows(2) {
        let (Some(&a), Some(&b)) = (w.first(), w.get(1)) else { continue };
        let pieces = ((b.0 - a.0).hypot(b.1 - a.1) / max_step).ceil().clamp(1.0, 1e6) as usize;
        let at = |k: usize| {
            let t = k as f32 / pieces as f32;
            (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t)
        };
        out.extend((0..pieces).map(|k| (at(k), at(k + 1))));
    }
    out
}

impl EdgeBrush {
    /// Rasterises `strokes` (clipped to `canvas`). `None` when they paint nothing or sweep more
    /// than [`MAX_SWEPT_AREA`].
    pub fn new(strokes: &[EdgeStroke], canvas: Rect) -> Option<EdgeBrush> {
        let usable = |s: &&EdgeStroke| s.size.is_finite() && !s.points.is_empty() && s.points.iter().all(|p| p.0.is_finite() && p.1.is_finite());
        if swept_area(strokes) > MAX_SWEPT_AREA {
            return None;
        }
        let strokes: Vec<&EdgeStroke> = strokes.iter().filter(usable).collect();
        let stroke_bounds = |s: &EdgeStroke| -> Rect {
            let r = s.size.clamp(*SIZE_RANGE.start(), *SIZE_RANGE.end()) / 2.0;
            s.points.iter().fold(Rect::EMPTY, |acc, p| acc.union(&dab_bounds(*p, r))).intersect(&canvas)
        };
        let bbox = strokes.iter().fold(Rect::EMPTY, |acc, s| acc.union(&stroke_bounds(s)));
        if bbox.is_empty() {
            return None;
        }
        let bw = bbox.width() as usize;
        let mut v = vec![0i8; bw.checked_mul(bbox.height() as usize)?];
        let mut cov: Vec<f32> = Vec::new();
        let mut max_add = 0.0f32;
        for s in strokes {
            let sb = stroke_bounds(s);
            if sb.is_empty() {
                continue;
            }
            let r = s.size.clamp(*SIZE_RANGE.start(), *SIZE_RANGE.end()) / 2.0;
            if s.add {
                max_add = max_add.max(2.0 * r);
            }
            let sw = sb.width() as usize;
            cov.clear();
            cov.resize(sw * sb.height() as usize, 0.0);
            // The stroke's coverage: the most any of its segments gives (a single dab for one point).
            for (a, b) in segments(&s.points, r) {
                let seg = dab_bounds(a, r).union(&dab_bounds(b, r)).intersect(&sb);
                for y in seg.y0..seg.y1 {
                    for x in seg.x0..seg.x1 {
                        let c = capsule((x as f32 + 0.5, y as f32 + 0.5), a, b, r);
                        if let Some(slot) = cov.get_mut((y - sb.y0) as usize * sw + (x - sb.x0) as usize)
                            && c > *slot
                        {
                            *slot = c;
                        }
                    }
                }
            }
            let target = if s.add { 1.0 } else { -1.0 };
            for y in sb.y0..sb.y1 {
                for x in sb.x0..sb.x1 {
                    let c = cov.get((y - sb.y0) as usize * sw + (x - sb.x0) as usize).copied().unwrap_or(0.0);
                    if c <= 0.0 {
                        continue;
                    }
                    if let Some(slot) = v.get_mut((y - bbox.y0) as usize * bw + (x - bbox.x0) as usize) {
                        let cur = f32::from(*slot) / 127.0;
                        *slot = ((cur + (target - cur) * c) * 127.0).round().clamp(-127.0, 127.0) as i8;
                    }
                }
            }
        }
        if v.iter().all(|c| *c == 0) {
            return None;
        }
        let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        for (row, line) in v.chunks_exact(bw).enumerate() {
            let (Some(first), Some(last)) = (line.iter().position(|c| *c > 0), line.iter().rposition(|c| *c > 0)) else { continue };
            let y = bbox.y0 + row as i32;
            (x0, x1, y0, y1) = (x0.min(bbox.x0 + first as i32), x1.max(bbox.x0 + last as i32 + 1), y0.min(y), y1.max(y + 1));
        }
        let added = if x0 < x1 { Rect::new(x0, y0, x1, y1) } else { Rect::EMPTY };
        Some(EdgeBrush { bbox, v, added, window: max_add.ceil().clamp(4.0, 64.0) as usize })
    }

    /// Where the strokes paint (added or restored).
    pub fn bounds(&self) -> Rect {
        self.bbox
    }

    /// Where the strokes add to the refinement area (empty when they only restore).
    pub fn added_bounds(&self) -> Rect {
        self.added
    }

    /// Box radius of the colour estimate (it reads up to four times as far).
    pub fn window(&self) -> usize {
        self.window
    }

    /// Signed coverage at document pixel `(x, y)` (zero outside the painted area).
    pub fn at(&self, x: i32, y: i32) -> f32 {
        if !self.bbox.contains(x, y) {
            return 0.0;
        }
        let i = (y - self.bbox.y0) as usize * self.bbox.width() as usize + (x - self.bbox.x0) as usize;
        self.v.get(i).map_or(0.0, |c| f32::from(*c) / 127.0)
    }

    /// Signed coverage over `r` (row-major).
    pub fn read(&self, r: Rect) -> Vec<f32> {
        (r.y0..r.y1).flat_map(|y| (r.x0..r.x1).map(move |x| self.at(x, y))).collect()
    }
}

/// Coverage estimated from the image where `brush` (signed, `w × h`) adds: each pixel's colour
/// projected on the line between the mean known foreground and background colours within
/// `window` px (or `4 × window` where there are none that close). Known means outside the painted area and (nearly) fully in or out of `m`. Pixels
/// with no foreground or background nearby, or no contrast between them, keep `m`.
pub(super) fn estimate(img: &RgbImage, m: &[f32], brush: &[f32], w: usize, h: usize, window: usize) -> Vec<f32> {
    let n = w * h;
    let known = |want_fg: bool| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let (mv, bv) = (m.get(i).copied().unwrap_or(0.0), brush.get(i).copied().unwrap_or(0.0));
                let ok = bv <= 0.0 && if want_fg { mv >= 0.98 } else { mv <= 0.02 };
                if ok { 1.0 } else { 0.0 }
            })
            .collect()
    };
    let (wf, wb) = (known(true), known(false));
    // Normalised convolution of the known colours: weight sums and weighted colour sums, over the
    // window and (where the window finds none) four times as far.
    let means = |wt: &[f32], r: usize| -> (Vec<f32>, Vec<Vec<f32>>) {
        let sums = (0..3)
            .map(|c| {
                let v: Vec<f32> = img.px.iter().zip(wt).map(|(p, k)| p.get(c).copied().unwrap_or(0.0) * k).collect();
                box_mean(&v, w, h, r)
            })
            .collect();
        (box_mean(wt, w, h, r), sums)
    };
    let near = [means(&wf, window), means(&wb, window)];
    let far = [means(&wf, window * 4), means(&wb, window * 4)];
    // The mean known colour at `i` (0 = foreground, 1 = background).
    let mean = |k: usize, i: usize| -> Option<[f32; 3]> {
        let pick = |(d, s): &(Vec<f32>, Vec<Vec<f32>>)| -> Option<[f32; 3]> {
            let d = d.get(i).copied().filter(|d| *d >= 1e-4)?;
            Some([0, 1, 2].map(|c| s.get(c).and_then(|v| v.get(i)).copied().unwrap_or(0.0) / d))
        };
        near.get(k).and_then(pick).or_else(|| far.get(k).and_then(pick))
    };
    let mut out = m.to_vec();
    for (i, o) in out.iter_mut().enumerate() {
        if brush.get(i).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let (Some(fm), Some(bm), Some(px)) = (mean(0, i), mean(1, i), img.px.get(i)) else { continue };
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for ((f, b), p) in fm.iter().zip(&bm).zip(px) {
            num += (p - b) * (f - b);
            den += (f - b) * (f - b);
        }
        if den > 1e-4 {
            *o = (num / den).clamp(0.0, 1.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matting::{RefineParams, refine_mask_brushed, region_reader};
    use crate::segment::ImageSampler;
    use crate::selection::Region;

    /// Strands of the subject's colour (every 4th column from x = 60) beyond the selection's
    /// edge at x = 40: brushing over them adds them, Alt-brushing the edge keeps it as it was.
    #[test]
    fn brush_picks_up_strands_beyond_the_band_and_restores_the_edge() {
        let (w, h) = (120usize, 60usize);
        let (fg, bg) = ([0.9f32, 0.7, 0.2], [0.1f32, 0.3, 0.8]);
        let strand = |x: usize| x < 40 || ((60..84).contains(&x) && x.is_multiple_of(4));
        let img = RgbImage::from_fn(w, h, |x, _| if strand(x) { fg } else { bg });
        let s = ImageSampler { img: &img, origin: (0, 0) };
        let canvas = Rect::new(0, 0, w as i32, h as i32);
        let reg = Region { bbox: canvas, mask: (0..w * h).map(|i| if i % w < 40 { 255 } else { 0 }).collect() };
        let p = RefineParams { radius: 4.0, ..Default::default() };
        let refine = |strokes: &[EdgeStroke]| {
            let brush = EdgeBrush::new(strokes, canvas);
            refine_mask_brushed(&s, &region_reader(&reg), reg.bbox, canvas, &p, brush.as_ref()).expect("refined")
        };
        let plain = refine(&[]);
        assert!(plain.at(64, 30) < 0.01, "the band alone stops short of the strands");
        let over_strands = EdgeStroke { points: vec![(70.0, 10.0), (70.0, 50.0)], size: 30.0, add: true };
        let brushed = refine(std::slice::from_ref(&over_strands));
        assert!(brushed.at(64, 30) > 0.9 && brushed.at(72, 30) > 0.9, "strands added: {}", brushed.at(64, 30));
        assert!(brushed.at(66, 30) < 0.1, "the background between them stays out: {}", brushed.at(66, 30));
        assert!(brushed.at(20, 30) > 0.99 && brushed.at(100, 30) < 0.01, "unpainted areas keep the refined selection");
        // Restoring paints the refinement out: with Shift Edge 0 the edge is the original one.
        let soft = RefineParams { radius: 4.0, feather: 0.0, ..Default::default() };
        let blurred = RgbImage::from_fn(w, h, |x, _| {
            let t = ((x as f32 - 36.0) / 8.0).clamp(0.0, 1.0);
            [0, 1, 2].map(|c| fg[c] * (1.0 - t) + bg[c] * t)
        });
        let sb = ImageSampler { img: &blurred, origin: (0, 0) };
        let refined = refine_mask_brushed(&sb, &region_reader(&reg), reg.bbox, canvas, &soft, None).expect("refined");
        assert!(refined.at(41, 30) > 0.1, "the band softens the blurred edge");
        let restore = [EdgeStroke { points: vec![(40.0, 0.0), (40.0, 60.0)], size: 20.0, add: false }];
        let restored = refine_mask_brushed(&sb, &region_reader(&reg), reg.bbox, canvas, &soft, EdgeBrush::new(&restore, canvas).as_ref()).expect("refined");
        assert!(restored.at(41, 30) < 0.01 && restored.at(38, 30) > 0.99, "the original edge is back");
    }

    #[test]
    fn strokes_paint_in_order_and_clip_to_the_canvas() {
        let canvas = Rect::new(0, 0, 100, 50);
        let add = EdgeStroke { points: vec![(10.0, 25.0), (90.0, 25.0)], size: 10.0, add: true };
        let b = EdgeBrush::new(std::slice::from_ref(&add), canvas).expect("painted");
        assert!(b.at(50, 25) > 0.99 && b.at(50, 40) == 0.0);
        assert_eq!(b.window(), 10);
        // A later subtract stroke paints over the add.
        let sub = EdgeStroke { points: vec![(50.0, 25.0)], size: 20.0, add: false };
        let b = EdgeBrush::new(&[add.clone(), sub], canvas).expect("painted");
        assert!(b.at(50, 25) < -0.99 && b.at(20, 25) > 0.99);
        // Off the canvas, non-finite or empty strokes paint nothing.
        let off = EdgeStroke { points: vec![(-500.0, -500.0)], size: 10.0, add: true };
        let nan = EdgeStroke { points: vec![(f32::NAN, 3.0)], size: 10.0, add: true };
        let empty = EdgeStroke { points: vec![], size: 10.0, add: true };
        assert!(EdgeBrush::new(&[off, nan, empty], canvas).is_none());
    }
}
