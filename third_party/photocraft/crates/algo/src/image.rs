//! Interleaved `f32` image regions and sampling helpers.

use photocraft_geom::Rect;
use photocraft_raster::Surface;

/// How samples outside an image are read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// Zero (transparent).
    Transparent,
    /// Clamp to the nearest edge pixel.
    Repeat,
    /// Wrap around.
    Wrap,
}

/// A rectangle of interleaved, normalized samples.
#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub rect: Rect,
    /// Channels per pixel (alpha last when the surface has alpha).
    pub ch: usize,
    pub data: Vec<f32>,
}

impl Image {
    /// Reads `rect` from a surface.
    pub fn read(s: &Surface, rect: Rect) -> Self {
        Image { rect, ch: s.channels(), data: s.read_region(rect) }
    }

    /// Reads `rect`, repeating the nearest pixel of `extent` for samples outside it. Photoshop
    /// treats the canvas edge this way when filtering a layer, so a blur doesn't fade a
    /// full-canvas layer to transparent at the document edge.
    pub fn read_clamped(s: &Surface, rect: Rect, extent: Rect) -> Self {
        let inner = rect.intersect(&extent);
        if inner.is_empty() || inner == rect {
            return Self::read(s, rect);
        }
        let src = Self::read(s, inner);
        let mut img = Self::new(rect, src.ch);
        let ch = src.ch;
        let w = rect.width() as usize;
        for y in rect.y0..rect.y1 {
            let sy = y.clamp(inner.y0, inner.y1 - 1);
            let row = (y - rect.y0) as usize * w * ch;
            for x in rect.x0..rect.x1 {
                let sx = x.clamp(inner.x0, inner.x1 - 1);
                let d = row + (x - rect.x0) as usize * ch;
                img.data[d..d + ch].copy_from_slice(src.px(sx, sy));
            }
        }
        img
    }

    /// New zeroed image.
    pub fn new(rect: Rect, ch: usize) -> Self {
        Image { rect, ch, data: vec![0.0; rect.width() as usize * rect.height() as usize * ch] }
    }

    #[inline]
    pub fn w(&self) -> i32 {
        self.rect.width() as i32
    }

    #[inline]
    fn index(&self, x: i32, y: i32) -> usize {
        ((y - self.rect.y0) as usize * self.rect.width() as usize + (x - self.rect.x0) as usize) * self.ch
    }

    /// Sample at integer coordinates; 0 outside.
    #[inline]
    pub fn get(&self, x: i32, y: i32, c: usize) -> f32 {
        if self.rect.contains(x, y) { self.data[self.index(x, y) + c] } else { 0.0 }
    }

    /// Samples of the pixels of a one-pixel-high rectangle (must be inside).
    #[inline]
    pub fn row(&self, r: Rect) -> &[f32] {
        let i = self.index(r.x0, r.y0);
        &self.data[i..i + r.width() as usize * self.ch]
    }

    /// Pixel slice at `(x, y)` (must be inside).
    #[inline]
    pub fn px(&self, x: i32, y: i32) -> &[f32] {
        let i = self.index(x, y);
        &self.data[i..i + self.ch]
    }

    /// Sample with an edge policy relative to `area`.
    #[inline]
    pub fn get_edge(&self, x: i32, y: i32, c: usize, edge: Edge, area: Rect) -> f32 {
        if area.is_empty() {
            return 0.0;
        }
        if area.contains(x, y) {
            return self.get(x, y, c);
        }
        match edge {
            Edge::Transparent => 0.0,
            Edge::Repeat => self.get(x.clamp(area.x0, area.x1 - 1), y.clamp(area.y0, area.y1 - 1), c),
            Edge::Wrap => {
                let (w, h) = (area.width().max(1) as i32, area.height().max(1) as i32);
                self.get(area.x0 + (x - area.x0).rem_euclid(w), area.y0 + (y - area.y0).rem_euclid(h), c)
            }
        }
    }

    /// Bilinear sample of the premultiplied pixel at `(x, y)` (pixel centres
    /// at `.5`), written to `out` (straight colour, alpha last).
    pub fn sample(&self, x: f32, y: f32, edge: Edge, area: Rect, alpha: bool, out: &mut [f32]) {
        let (fx, fy) = (x - 0.5, y - 0.5);
        let (x0, y0) = (fx.floor(), fy.floor());
        let (ax, ay) = (fx - x0, fy - y0);
        let (x0, y0) = (x0 as i32, y0 as i32);
        let n = self.ch;
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let mut acc_a = 0.0;
        for (dx, dy, wgt) in [(0, 0, (1.0 - ax) * (1.0 - ay)), (1, 0, ax * (1.0 - ay)), (0, 1, (1.0 - ax) * ay), (1, 1, ax * ay)] {
            if wgt <= 0.0 {
                continue;
            }
            let a = if alpha { self.get_edge(x0 + dx, y0 + dy, n - 1, edge, area) } else { 1.0 };
            for (c, o) in out.iter_mut().enumerate().take(if alpha { n - 1 } else { n }) {
                *o += self.get_edge(x0 + dx, y0 + dy, c, edge, area) * a * wgt;
            }
            acc_a += a * wgt;
        }
        if alpha {
            if acc_a > 0.0 {
                for o in out.iter_mut().take(n - 1) {
                    *o /= acc_a;
                }
            }
            out[n - 1] = acc_a;
        }
    }

    /// Copy of the samples inside `out` (for identity passes).
    pub fn crop(&self, out: Rect) -> Vec<f32> {
        let mut v = Vec::with_capacity(out.width() as usize * out.height() as usize * self.ch);
        for y in out.y0..out.y1 {
            for x in out.x0..out.x1 {
                for c in 0..self.ch {
                    v.push(self.get(x, y, c));
                }
            }
        }
        v
    }
}

/// Premultiplies colour channels by alpha (in place). No-op without alpha.
pub fn premultiply(data: &mut [f32], ch: usize, alpha: bool) {
    if !alpha {
        return;
    }
    for px in data.chunks_exact_mut(ch) {
        let a = px[ch - 1];
        for v in px.iter_mut().take(ch - 1) {
            *v *= a;
        }
    }
}

/// Inverse of [`premultiply`].
pub fn unpremultiply(data: &mut [f32], ch: usize, alpha: bool) {
    if !alpha {
        return;
    }
    for px in data.chunks_exact_mut(ch) {
        let a = px[ch - 1];
        for v in px.iter_mut().take(ch - 1) {
            *v = if a > 1e-7 { *v / a } else { 0.0 };
        }
    }
}
