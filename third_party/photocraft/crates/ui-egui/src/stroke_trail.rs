//! The translucent footprint trail the retouching tools (Dodge, Blur, Clone Stamp, …) and Quick
//! Selection show while dragging: the union of the brush's round footprint along the drag.
//!
//! It is a coverage mask in document space, stamped segment by segment as the pointer moves and
//! drawn as one texture over the document. The Spot Healing Brush's trail is instead the coverage
//! of the brush's own dabs ([`Trail::feed_dabs`]): hardness, pressure and shape dynamics show as
//! they heal (#2364). It used to be an egui polyline as wide as the brush:
//! egui tessellates a polyline whose width is much larger than its steps (any zoomed-in drag)
//! into hard pie wedges fanning out across the canvas (#189), and translucent overlaps doubled up.

use egui::{Color32, ColorImage, Rect, TextureHandle, TextureOptions, pos2};
use photocraft_paint::{BrushSettings, StrokePoint, StrokeRenderer};

/// Longest side of the mask, in mask pixels (big documents are stamped at a reduced scale).
const MAX_SIDE: u32 = 1024;

/// A drag's footprint mask.
pub struct Trail {
    /// Document size the mask covers.
    doc: [u32; 2],
    /// Document pixels per mask pixel.
    k: f32,
    w: usize,
    h: usize,
    /// Coverage 0..=255 per mask pixel.
    cov: Vec<u8>,
    /// Drag points stamped so far.
    fed: usize,
    /// Mask pixels changed since the last upload (x0, y0, x1, y1), if any.
    dirty: Option<[usize; 4]>,
    tex: Option<TextureHandle>,
    /// The stroke's dabs, when the trail shows their coverage ([`Self::feed_dabs`]).
    dabs: Option<Dabs>,
}

/// The brush stroke behind a [`Trail::feed_dabs`] trail.
struct Dabs {
    renderer: StrokeRenderer,
    /// Document pixels the last "finish now" tail (a lone first dab, smoothing catch-up) covered.
    tail: photocraft_geom::Rect,
}

impl Trail {
    pub fn new(doc: [u32; 2]) -> Self {
        let k = doc[0].max(doc[1]).div_ceil(MAX_SIDE).max(1);
        let (w, h) = (doc[0].div_ceil(k).max(1) as usize, doc[1].div_ceil(k).max(1) as usize);
        Self { doc, k: k as f32, w, h, cov: vec![0; w * h], fed: 0, dirty: None, tex: None, dabs: None }
    }

    /// Coverage at a mask pixel (tests).
    pub fn coverage(&self, x: usize, y: usize) -> u8 {
        if x < self.w && y < self.h { self.cov.get(y * self.w + x).copied().unwrap_or(0) } else { 0 }
    }

    /// Document pixels per mask pixel.
    pub fn scale(&self) -> f32 {
        self.k
    }

    /// Stamp the drag points not seen yet: round footprints of `diameter` document pixels joined
    /// into a capsule per segment (coverage is the max, so overlaps don't add up).
    pub fn feed(&mut self, points: &[[f64; 3]], diameter: f32) {
        if self.fed >= points.len() {
            return;
        }
        let k = f64::from(self.k);
        let r = (f64::from(diameter.max(1.0)) / 2.0) / k;
        let mask = |p: &[f64; 3]| [p[0] / k, p[1] / k];
        for i in self.fed..points.len() {
            let Some(b) = points.get(i).map(mask) else { break };
            let a = i.checked_sub(1).and_then(|j| points.get(j)).map_or(b, mask);
            self.capsule(a, b, r);
        }
        self.fed = points.len();
    }

    /// Show the coverage the brush's own dabs leave along the drag, as the stroke commit renders
    /// it (`photocraft_paint::retouch::stroke_coverage`): soft edges for a soft brush, the size and
    /// shape each dab takes from pen pressure and dynamics. `brush` is asked for once, on the first
    /// call. What finishing the stroke now would add (the press's lone first dab) shows too.
    pub fn feed_dabs(&mut self, points: &[[f64; 3]], brush: impl FnOnce() -> BrushSettings) {
        if self.dabs.is_some() && self.fed >= points.len() {
            return;
        }
        let mut dabs = self.dabs.take().unwrap_or_else(|| Dabs { renderer: StrokeRenderer::new(&brush(), None, 1.0), tail: photocraft_geom::Rect::EMPTY });
        let new: Vec<StrokePoint> = points
            .get(self.fed..)
            .unwrap_or(&[])
            .iter()
            .filter(|p| p.iter().all(|v| v.is_finite()) && p[0].abs() < 1e7 && p[1].abs() < 1e7)
            .map(|p| StrokePoint::new(p[0], p[1], p[2] as f32))
            .collect();
        self.fed = points.len();
        dabs.renderer.push(&new);
        let mut changed = dabs.renderer.take_dirty_rect().union(&dabs.tail);
        let tail = dabs.renderer.tail_preview().map(|mut t| (t.take_dirty_rect(), t));
        dabs.tail = tail.as_ref().map_or(photocraft_geom::Rect::EMPTY, |(r, _)| *r);
        changed = changed.union(&dabs.tail);
        let tail = tail.map(|(_, t)| t);
        let renderer = &dabs.renderer;
        self.restamp(changed, |x, y| renderer.coverage_at(x, y).max(tail.as_ref().map_or(0.0, |t| t.coverage_at(x, y))));
        self.dabs = Some(dabs);
    }

    /// Recompute the mask pixels over the document rectangle `r` from `cov` (document pixel →
    /// coverage 0..=1), sampled at each mask pixel's centre.
    fn restamp(&mut self, r: photocraft_geom::Rect, cov: impl Fn(i32, i32) -> f32) {
        if r.is_empty() {
            return;
        }
        let k = f64::from(self.k);
        let x0 = (f64::from(r.x0) / k).floor().max(0.0) as usize;
        let y0 = (f64::from(r.y0) / k).floor().max(0.0) as usize;
        let x1 = (f64::from(r.x1) / k).ceil().clamp(0.0, self.w as f64) as usize;
        let y1 = (f64::from(r.y1) / k).ceil().clamp(0.0, self.h as f64) as usize;
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        for y in y0..y1 {
            let dy = ((y as f64 + 0.5) * k).floor() as i32;
            for x in x0..x1 {
                let dx = ((x as f64 + 0.5) * k).floor() as i32;
                let c = cov(dx, dy);
                let c = if c.is_finite() { (c.clamp(0.0, 1.0) * 255.0).round() as u8 } else { 0 };
                if let Some(v) = self.cov.get_mut(y * self.w + x) {
                    *v = c;
                }
            }
        }
        let d = self.dirty.get_or_insert([x0, y0, x1, y1]);
        *d = [d[0].min(x0), d[1].min(y0), d[2].max(x1), d[3].max(y1)];
    }

    fn capsule(&mut self, a: [f64; 2], b: [f64; 2], r: f64) {
        if !(a.iter().chain(&b).all(|v| v.is_finite()) && r.is_finite()) {
            return;
        }
        let x0 = (a[0].min(b[0]) - r - 1.0).floor().max(0.0);
        let y0 = (a[1].min(b[1]) - r - 1.0).floor().max(0.0);
        let x1 = (a[0].max(b[0]) + r + 1.0).ceil().min(self.w as f64);
        let y1 = (a[1].max(b[1]) + r + 1.0).ceil().min(self.h as f64);
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        let (x0, y0, x1, y1) = (x0 as usize, y0 as usize, x1 as usize, y1 as usize);
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len2 = dx * dx + dy * dy;
        for y in y0..y1 {
            for x in x0..x1 {
                let (px, py) = (x as f64 + 0.5 - a[0], y as f64 + 0.5 - a[1]);
                let t = if len2 > 0.0 { ((px * dx + py * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
                let d = (px - t * dx).hypot(py - t * dy);
                // One mask pixel of anti-aliasing at the edge.
                let c = ((r + 0.5 - d).clamp(0.0, 1.0) * 255.0).round() as u8;
                if let Some(v) = self.cov.get_mut(y * self.w + x)
                    && c > *v
                {
                    *v = c;
                }
            }
        }
        let d = self.dirty.get_or_insert([x0, y0, x1, y1]);
        *d = [d[0].min(x0), d[1].min(y0), d[2].max(x1), d[3].max(y1)];
    }

    fn image(&self, r: [usize; 4], color: Color32) -> ColorImage {
        let [x0, y0, x1, y1] = r;
        let mut px = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            for x in x0..x1 {
                let c = self.coverage(x, y);
                px.push(Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), ((u16::from(c) * u16::from(color.a())) / 255) as u8));
            }
        }
        ColorImage::new([x1 - x0, y1 - y0], px)
    }

    /// Upload what changed and draw the trail over the document (`doc_rect` = the document's
    /// screen rectangle, `flip` = mirrored view), tinted `color` where fully covered.
    pub fn draw(&mut self, painter: &egui::Painter, doc_rect: Rect, flip: bool, color: Color32) {
        let opts = TextureOptions::LINEAR;
        let dirty = self.dirty.take();
        if self.tex.is_none() {
            let img = self.image([0, 0, self.w, self.h], color);
            self.tex = Some(painter.ctx().load_texture("stroke-trail", img, opts));
        } else if let Some(r) = dirty {
            let img = self.image(r, color);
            if let Some(t) = self.tex.as_mut() {
                t.set_partial([r[0], r[1]], img, opts);
            }
        }
        let Some(t) = &self.tex else { return };
        // The mask's last column/row may reach past the document (sizes rounded up).
        let (u, v) = (self.doc[0] as f32 / (self.w as f32 * self.k), self.doc[1] as f32 / (self.h as f32 * self.k));
        let uv = if flip { Rect::from_min_max(pos2(u, 0.0), pos2(0.0, v)) } else { Rect::from_min_max(pos2(0.0, 0.0), pos2(u, v)) };
        painter.image(t.id(), doc_rect, uv, Color32::WHITE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_trail_is_the_union_of_round_footprints_along_the_drag() {
        let mut t = Trail::new([200, 100]);
        // Many tiny steps with a wide brush (what zoomed-in drags send).
        let pts: Vec<[f64; 3]> = (0..200).map(|i| [40.0 + f64::from(i) * 0.5, 50.0 + (f64::from(i) * 0.3).sin(), 1.0]).collect();
        t.feed(&pts[..100], 40.0);
        t.feed(&pts, 40.0);
        assert_eq!(t.coverage(90, 50), 255, "on the path");
        assert_eq!(t.coverage(90, 50 + 15), 255, "inside the footprint");
        assert_eq!(t.coverage(90, 50 + 25), 0, "outside the footprint: no wedges reaching out");
        assert_eq!(t.coverage(5, 50), 0, "nothing behind the start");
        assert_eq!(t.coverage(170, 50), 0, "nothing past the end");
        // Every covered pixel lies within the brush radius of the path.
        for y in 0..100 {
            for x in 0..200 {
                if t.coverage(x, y) > 0 {
                    let d = pts.iter().map(|p| (x as f64 + 0.5 - p[0]).hypot(y as f64 + 0.5 - p[1])).fold(f64::MAX, f64::min);
                    assert!(d <= 21.5, "({x}, {y}) is {d} px from the path");
                }
            }
        }
    }

    #[test]
    fn hostile_points_and_huge_documents_are_safe() {
        let mut t = Trail::new([300_000, 2]);
        assert!(t.scale() > 1.0 && t.w <= MAX_SIDE as usize);
        t.feed(&[[f64::NAN, 1.0, 1.0], [f64::INFINITY, -1e300, 1.0], [-5.0, -5.0, 1.0], [1e12, 1.0, 1.0]], 1e9);
        t.feed(&[], 10.0);
        let mut e = Trail::new([0, 0]);
        e.feed(&[[0.0, 0.0, 1.0]], 0.0);
        assert!(e.coverage(0, 0) > 0 && e.coverage(5, 5) == 0);
    }
}
