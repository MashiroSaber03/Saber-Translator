//! Magnetic Lasso edge tracing: a selection border that snaps to the strongest edges near the
//! pointer.
//!
//! Between two fastening points the border is the cheapest 8-connected pixel path, where pixels on
//! a strong, well-localised edge are cheap and the rest are expensive: the "live wire" of
//! Mortensen & Barrett, *Intelligent Scissors for Image Composition* (SIGGRAPH 1995). A link costs
//! a weighted sum of an edge-strength term, an edge-centre term (non-maximum suppression across the
//! edge, in place of the paper's Laplacian zero crossings) and a gradient-direction term that keeps
//! the border running along the edge. The search stays in a corridor, the pixels within the
//! detection width of the pointer's path, and a small pull towards that path keeps the border on
//! the pointer where there is no edge, so it never jumps to an edge the pointer did not go near.
//! Points on an edge are placed at its sub-pixel centre, so the border lands on the edge rather than
//! half a pixel beside it.
//!
//! Pixels are read lazily in blocks through a caller-supplied fetch (a layer's pixels or the
//! composite), so tracing on a 36 MP image reads only the area around the border.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::f32::consts::{PI, SQRT_2};

use photocraft_geom::Rect;

/// Side of the cached pixel blocks (px).
const BLOCK: i32 = 128;
/// Blocks kept before the cache starts over (128² RGBA8 is 64 KiB, so at most 64 MiB).
const MAX_BLOCKS: usize = 1024;
/// Longest stretch of the pointer's path searched at once (px). Longer paths are traced in pieces
/// that meet on an edge, which bounds every search to a small area.
const PIECE: f64 = 256.0;
/// Longest pointer path [`Tracer::trace`] follows (px); a longer one is returned as is.
pub const MAX_GUIDE_LENGTH: f64 = 262_144.0;
/// Detection width limits (px), as in Photoshop's options bar.
pub const WIDTH_RANGE: (f64, f64) = (1.0, 256.0);

// Link cost weights: Mortensen & Barrett's 0.43 (edge centre) / 0.43 (strength) / 0.14
// (direction), a cost per pixel of length so the border never wanders along an edge for free, and
// the pull towards the pointer's path.
const W_CENTRE: f32 = 0.43;
const W_STRENGTH: f32 = 0.43;
const W_DIRECTION: f32 = 0.14;
const W_LENGTH: f32 = 0.02;
const W_GUIDE: f32 = 0.4;
/// The direction term where either pixel has no edge direction (the mean of the term).
const NEUTRAL_DIRECTION: f32 = 2.0 / 3.0;

/// The 8 neighbours, in the order `Search::from` records them.
const DIRS: [(i32, i32); 8] = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)];

/// Detection settings: the Magnetic Lasso options bar.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Settings {
    /// Detection width (px): the border follows only edges this close to the pointer's path.
    pub width: f64,
    /// Contrast (0..1): steps weaker than this, in every channel, are not edges.
    pub contrast: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { width: 10.0, contrast: 0.1 }
    }
}

impl Settings {
    /// Settings clamped to their ranges (width 1..256 px, contrast 1..100 %); NaN takes the default.
    pub fn new(width: f64, contrast: f32) -> Self {
        let d = Settings::default();
        let width = if width.is_finite() { width.clamp(WIDTH_RANGE.0, WIDTH_RANGE.1) } else { d.width };
        let contrast = if contrast.is_finite() { contrast.clamp(0.01, 1.0) } else { d.contrast };
        Settings { width, contrast }
    }

    /// The corridor's radius: the detection width plus room for an 8-connected path.
    fn radius(&self) -> f64 {
        self.width + 1.5
    }
}

/// Reads straight-alpha RGBA8 pixels of a rectangle inside the image, row-major. A result of the
/// wrong length reads as transparent.
pub type Fetch<'a> = dyn FnMut(Rect) -> Vec<[u8; 4]> + 'a;

/// Edge tracing over one image, caching the pixels it has read.
#[derive(Debug)]
pub struct Tracer {
    bounds: Rect,
    blocks: HashMap<(i32, i32), Vec<[u8; 4]>>,
}

impl Tracer {
    /// A tracer for an image covering `bounds` (document pixels).
    pub fn new(bounds: Rect) -> Self {
        Tracer { bounds, blocks: HashMap::new() }
    }

    pub fn bounds(&self) -> Rect {
        self.bounds
    }

    /// Pixels read so far (for tests and diagnostics).
    pub fn cached_pixels(&self) -> usize {
        self.blocks.values().map(Vec::len).sum()
    }

    /// Where a fastening point near `p` goes: the centre of the most prominent edge within the
    /// detection width, or `p` itself (kept inside the image) when no edge there reaches the
    /// contrast. `None` when `p` is not a finite point or the image is empty.
    pub fn snap(&mut self, fetch: &mut Fetch, p: [f64; 2], s: Settings) -> Option<[f64; 2]> {
        let s = Settings::new(s.width, s.contrast);
        let p = self.clamp_point(p)?;
        let r = s.width;
        let (px, py) = self.pixel(p);
        let ri = r.ceil() as i32 + 1;
        let area = Rect::new(px.saturating_sub(ri), py.saturating_sub(ri), px.saturating_add(ri + 1), py.saturating_add(ri + 1)).intersect(&self.bounds);
        let f = self.field(fetch, area);
        let mut best: Option<(f32, [f64; 2])> = None;
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                let Some(i) = f.index(x, y) else { continue };
                let strength = f.strength.get(i).copied().unwrap_or(0.0);
                if strength < s.contrast || !f.ridge.get(i).copied().unwrap_or(false) {
                    continue;
                }
                let d = dist([x as f64 + 0.5, y as f64 + 0.5], p);
                if d > r {
                    continue;
                }
                // The strongest edge, a little in favour of the nearer one.
                let score = strength * (1.0 - 0.25 * (d / r) as f32);
                if best.is_none_or(|(b, _)| score > b) {
                    best = Some((score, f.point(i, x, y, s.contrast)));
                }
            }
        }
        Some(best.map_or(p, |(_, q)| q))
    }

    /// The border from `from` to `to` along the edges near `guide`, the pointer's path between
    /// them (`from` and `to` are its ends whatever it holds). It starts at `from` and ends at `to`
    /// exactly (both kept inside the image). Empty when either is not a finite point or the image
    /// is empty.
    pub fn trace(&mut self, fetch: &mut Fetch, from: [f64; 2], to: [f64; 2], guide: &[[f64; 2]], s: Settings) -> Vec<[f64; 2]> {
        let s = Settings::new(s.width, s.contrast);
        let (Some(from), Some(to)) = (self.clamp_point(from), self.clamp_point(to)) else { return Vec::new() };
        let mut g = Vec::with_capacity(guide.len().saturating_add(2));
        g.push(from);
        g.extend(guide.iter().filter_map(|q| self.clamp_point(*q)));
        g.push(to);
        // A coarse guide is enough for the corridor, and much cheaper to draw.
        let g = simplify(&g, 0.5);
        if length(&g) > MAX_GUIDE_LENGTH {
            return g;
        }
        let pieces = split(&g, PIECE);
        let mut out = vec![from];
        let mut cur = from;
        let last = pieces.len().saturating_sub(1);
        for (k, piece) in pieces.iter().enumerate() {
            let target = if k == last { to } else { piece.last().and_then(|q| self.snap(fetch, *q, s)).unwrap_or(to) };
            out.extend(self.trace_piece(fetch, cur, target, piece, s).into_iter().skip(1));
            cur = target;
        }
        // Within half a pixel: drops the staircase of pixel steps, keeps the curve of an edge.
        simplify(&out, 0.5)
    }

    /// One piece of [`Tracer::trace`]: the cheapest path from `a` to `b` in the corridor around
    /// `guide`, from `a` to `b` inclusive.
    fn trace_piece(&mut self, fetch: &mut Fetch, a: [f64; 2], b: [f64; 2], guide: &[[f64; 2]], s: Settings) -> Vec<[f64; 2]> {
        let rad = s.radius();
        let (ax, ay) = self.pixel(a);
        let (bx, by) = self.pixel(b);
        let mut bbox = Rect::new(ax.min(bx), ay.min(by), ax.max(bx) + 1, ay.max(by) + 1);
        for q in guide {
            let (x, y) = self.pixel(*q);
            bbox = bbox.union(&Rect::new(x, y, x + 1, y + 1));
        }
        let area = bbox.inflate(rad.ceil() as i32 + 1).intersect(&self.bounds);
        let f = self.field(fetch, area);
        let mut near = corridor(area, guide, rad);
        let (Some(start), Some(goal)) = (f.index(ax, ay), f.index(bx, by)) else { return vec![a, b] };
        for i in [start, goal] {
            if let Some(v) = near.get_mut(i) {
                *v = 0.0;
            }
        }
        let Some(cells) = search(&f, &near, (rad * rad) as f32, start, goal, s.contrast) else { return vec![a, b] };
        let mut pts: Vec<[f64; 2]> = cells.iter().map(|&i| f.point_at(i, s.contrast)).collect();
        match pts.len() {
            0 | 1 => vec![a, b],
            n => {
                pts[0] = a;
                pts[n - 1] = b;
                pts
            }
        }
    }

    /// `p` kept inside the image, or `None` if it is not a finite point or the image is empty.
    fn clamp_point(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let b = self.bounds;
        if b.is_empty() || !p[0].is_finite() || !p[1].is_finite() {
            return None;
        }
        Some([p[0].clamp(b.x0 as f64, b.x1 as f64), p[1].clamp(b.y0 as f64, b.y1 as f64)])
    }

    /// The pixel under a point (clamped to the image).
    fn pixel(&self, p: [f64; 2]) -> (i32, i32) {
        let b = self.bounds;
        let c = |v: f64, lo: i32, hi: i32| if v.is_finite() { (v.floor() as i32).clamp(lo, (hi - 1).max(lo)) } else { lo };
        (c(p[0], b.x0, b.x1), c(p[1], b.y0, b.y1))
    }

    /// Edge features over `area` (inside the image).
    fn field(&mut self, fetch: &mut Fetch, area: Rect) -> Field {
        // Sobel reads one ring beyond the gradient, and non-maximum suppression one beyond the area.
        let ext = area.inflate(2).intersect(&self.bounds);
        let px = self.pixels(fetch, ext);
        Field::new(area, ext, &px)
    }

    /// Straight RGBA8 pixels of `r` (inside the image), through the block cache.
    fn pixels(&mut self, fetch: &mut Fetch, r: Rect) -> Vec<[u8; 4]> {
        let w = r.width() as usize;
        let mut out = vec![[0u8; 4]; w * r.height() as usize];
        if r.is_empty() {
            return out;
        }
        if self.blocks.len() > MAX_BLOCKS {
            self.blocks.clear();
        }
        let (bx0, by0) = (r.x0.div_euclid(BLOCK), r.y0.div_euclid(BLOCK));
        let (bx1, by1) = ((r.x1 - 1).div_euclid(BLOCK), (r.y1 - 1).div_euclid(BLOCK));
        for by in by0..=by1 {
            for bx in bx0..=bx1 {
                let br = Rect::new(
                    bx.saturating_mul(BLOCK),
                    by.saturating_mul(BLOCK),
                    bx.saturating_add(1).saturating_mul(BLOCK),
                    by.saturating_add(1).saturating_mul(BLOCK),
                )
                .intersect(&self.bounds);
                if br.is_empty() {
                    continue;
                }
                let n = br.width() as usize * br.height() as usize;
                let block = self.blocks.entry((bx, by)).or_insert_with(|| {
                    let v = fetch(br);
                    if v.len() == n { v } else { vec![[0; 4]; n] }
                });
                let ov = br.intersect(&r);
                let len = ov.width() as usize;
                for y in ov.y0..ov.y1 {
                    let src = (y - br.y0) as usize * br.width() as usize + (ov.x0 - br.x0) as usize;
                    let dst = (y - r.y0) as usize * w + (ov.x0 - r.x0) as usize;
                    if let (Some(s), Some(d)) = (block.get(src..src + len), out.get_mut(dst..dst + len)) {
                        d.copy_from_slice(s);
                    }
                }
            }
        }
        out
    }
}

/// Edge features of the pixels in an area.
struct Field {
    area: Rect,
    w: usize,
    /// Edge strength: the steepest channel's Sobel gradient, scaled so a sharp step of height `h`
    /// reads `h` (capped at 1).
    strength: Vec<f32>,
    /// Unit gradient of that channel (zero on flat pixels).
    grad: Vec<[f32; 2]>,
    /// The pixel is the strongest across its edge (non-maximum suppression).
    ridge: Vec<bool>,
    /// From the pixel centre to the edge's sub-pixel centre, along the gradient.
    offset: Vec<[f32; 2]>,
}

impl Field {
    /// Features over `area` from the straight RGBA8 pixels `px` of `ext` (the area plus up to two
    /// pixels around it, inside the image; beyond it the edge pixels repeat).
    fn new(area: Rect, ext: Rect, px: &[[u8; 4]]) -> Field {
        let (ew, eh) = (ext.width() as i32, ext.height() as i32);
        // Premultiplied channels: an opaque shape on transparency has an edge, transparent
        // pixels of different colours do not.
        let ch: Vec<[f32; 4]> = px
            .iter()
            .map(|p| {
                let a = f32::from(p[3]) / 255.0;
                [f32::from(p[0]) / 255.0 * a, f32::from(p[1]) / 255.0 * a, f32::from(p[2]) / 255.0 * a, a]
            })
            .collect();
        let at = |x: i32, y: i32| -> [f32; 4] {
            let (x, y) = ((x - ext.x0).clamp(0, (ew - 1).max(0)), (y - ext.y0).clamp(0, (eh - 1).max(0)));
            ch.get(y as usize * ew.max(0) as usize + x as usize).copied().unwrap_or_default()
        };
        // Gradients over the area plus one ring (non-maximum suppression looks across it).
        let g1 = area.inflate(1).intersect(&ext);
        let gw = g1.width() as usize;
        let mut st = vec![0.0f32; gw * g1.height() as usize];
        let mut gd = vec![[0.0f32; 2]; st.len()];
        for y in g1.y0..g1.y1 {
            for x in g1.x0..g1.x1 {
                let (nw, n, ne) = (at(x - 1, y - 1), at(x, y - 1), at(x + 1, y - 1));
                let (w, e) = (at(x - 1, y), at(x + 1, y));
                let (sw, s, se) = (at(x - 1, y + 1), at(x, y + 1), at(x + 1, y + 1));
                let mut best = (0.0f32, [0.0f32; 2]);
                for c in 0..4 {
                    let gx = (ne[c] + 2.0 * e[c] + se[c]) - (nw[c] + 2.0 * w[c] + sw[c]);
                    let gy = (sw[c] + 2.0 * s[c] + se[c]) - (nw[c] + 2.0 * n[c] + ne[c]);
                    let m2 = gx * gx + gy * gy;
                    if m2 > best.0 {
                        best = (m2, [gx, gy]);
                    }
                }
                let mag = best.0.sqrt();
                let i = (y - g1.y0) as usize * gw + (x - g1.x0) as usize;
                if let (Some(sv), Some(gv)) = (st.get_mut(i), gd.get_mut(i)) {
                    *sv = (mag / 4.0).min(1.0);
                    *gv = if mag > 1e-6 { [best.1[0] / mag, best.1[1] / mag] } else { [0.0; 2] };
                }
            }
        }
        let s_at =
            |x: i32, y: i32| -> f32 { if g1.contains(x, y) { st.get((y - g1.y0) as usize * gw + (x - g1.x0) as usize).copied().unwrap_or(0.0) } else { 0.0 } };
        let w = area.width() as usize;
        let n = w * area.height() as usize;
        let mut f = Field { area, w, strength: vec![0.0; n], grad: vec![[0.0; 2]; n], ridge: vec![false; n], offset: vec![[0.0; 2]; n] };
        for y in area.y0..area.y1 {
            for x in area.x0..area.x1 {
                let s0 = s_at(x, y);
                let g = if g1.contains(x, y) { gd.get((y - g1.y0) as usize * gw + (x - g1.x0) as usize).copied().unwrap_or_default() } else { [0.0; 2] };
                let i = (y - area.y0) as usize * w + (x - area.x0) as usize;
                if let Some(v) = f.strength.get_mut(i) {
                    *v = s0;
                }
                if let Some(v) = f.grad.get_mut(i) {
                    *v = g;
                }
                if g == [0.0; 2] {
                    continue;
                }
                // The neighbours across the edge: the gradient rounded to one of 8 directions.
                let q = |v: f32, o: f32| if v.abs() >= 0.4142 * o.abs() { v.signum() as i32 } else { 0 };
                let (dx, dy) = (q(g[0], g[1]), q(g[1], g[0]));
                let (sm, sp) = (s_at(x - dx, y - dy), s_at(x + dx, y + dy));
                if s0 > 0.0 && s0 >= sm && s0 >= sp {
                    if let Some(v) = f.ridge.get_mut(i) {
                        *v = true;
                    }
                    // The peak of the parabola through the three strengths.
                    let denom = sm - 2.0 * s0 + sp;
                    let t = if denom < -1e-6 { (0.5 * (sm - sp) / denom).clamp(-0.5, 0.5) } else { 0.0 };
                    if let Some(v) = f.offset.get_mut(i) {
                        *v = [t * dx as f32, t * dy as f32];
                    }
                }
            }
        }
        f
    }

    fn h(&self) -> usize {
        self.strength.len() / self.w.max(1)
    }

    fn index(&self, x: i32, y: i32) -> Option<usize> {
        self.area.contains(x, y).then(|| (y - self.area.y0) as usize * self.w + (x - self.area.x0) as usize)
    }

    /// Where the border runs through pixel `i` at (`x`, `y`): the edge's sub-pixel centre on an edge
    /// at least `contrast` strong, else the pixel centre.
    fn point(&self, i: usize, x: i32, y: i32, contrast: f32) -> [f64; 2] {
        let on_edge = self.ridge.get(i).copied().unwrap_or(false) && self.strength.get(i).copied().unwrap_or(0.0) >= contrast;
        let o = if on_edge { self.offset.get(i).copied().unwrap_or_default() } else { [0.0; 2] };
        [x as f64 + 0.5 + f64::from(o[0]), y as f64 + 0.5 + f64::from(o[1])]
    }

    fn point_at(&self, i: usize, contrast: f32) -> [f64; 2] {
        let (x, y) = ((i % self.w.max(1)) as i32 + self.area.x0, (i / self.w.max(1)) as i32 + self.area.y0);
        self.point(i, x, y, contrast)
    }
}

/// Squared distance from each pixel centre of `area` to the polyline `guide`, infinite beyond
/// `rad`: the corridor the border may use.
fn corridor(area: Rect, guide: &[[f64; 2]], rad: f64) -> Vec<f32> {
    let w = area.width() as usize;
    let mut near = vec![f32::INFINITY; w * area.height() as usize];
    let r2 = rad * rad;
    let single;
    let segs: &[[f64; 2]] = match guide {
        [p] => {
            single = [*p, *p];
            &single
        }
        _ => guide,
    };
    for s in segs.windows(2) {
        let (a, b) = (s[0], s[1]);
        let lo = |u: f64, v: f64| (u.min(v) - rad).floor() as i32;
        let hi = |u: f64, v: f64| (u.max(v) + rad).ceil() as i32 + 1;
        let r = Rect::new(lo(a[0], b[0]), lo(a[1], b[1]), hi(a[0], b[0]), hi(a[1], b[1])).intersect(&area);
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let d2 = dist2_segment([x as f64 + 0.5, y as f64 + 0.5], a, b);
                if d2 <= r2
                    && let Some(v) = near.get_mut((y - area.y0) as usize * w + (x - area.x0) as usize)
                {
                    *v = v.min(d2 as f32);
                }
            }
        }
    }
    near
}

/// The cheapest 8-connected path from `start` to `goal` through the corridor `near` (squared
/// distances to the guide, `rad2` at its rim), as field indices from `start` to `goal`.
fn search(f: &Field, near: &[f32], rad2: f32, start: usize, goal: usize, contrast: f32) -> Option<Vec<usize>> {
    let (w, h) = (f.w, f.h());
    let n = w * h;
    // Cost of entering each pixel, per pixel of length, and its edge tangent.
    let mut node = vec![f32::INFINITY; n];
    let mut tangent = vec![[0.0f32; 2]; n];
    for i in 0..n {
        let d2 = near.get(i).copied().unwrap_or(f32::INFINITY);
        if !d2.is_finite() {
            continue;
        }
        let s = f.strength.get(i).copied().unwrap_or(0.0);
        let on_edge = s >= contrast && f.ridge.get(i).copied().unwrap_or(false);
        // No pull below the contrast, full pull from twice the contrast, a little more for
        // stronger edges still.
        let good = ((s - contrast) / contrast.max(0.02)).clamp(0.0, 1.0) * (0.6 + 0.4 * s.min(1.0));
        // Grows linearly away from the pointer's path, so the border hugs it where there is no edge.
        let pull = (d2 / rad2.max(1.0)).sqrt().min(1.0);
        let centre = if on_edge { 0.0 } else { 1.0 };
        if let Some(v) = node.get_mut(i) {
            *v = W_CENTRE * centre + W_STRENGTH * (1.0 - good) + W_LENGTH + W_GUIDE * pull;
        }
        if s >= 0.5 * contrast
            && let (Some(t), Some(g)) = (tangent.get_mut(i), f.grad.get(i))
        {
            *t = [g[1], -g[0]];
        }
    }
    let mut dist = vec![f32::INFINITY; n];
    let mut from = vec![u8::MAX; n];
    *dist.get_mut(start)? = 0.0;
    let mut heap = BinaryHeap::new();
    // Costs are finite and non-negative, so their bit patterns sort like the values.
    heap.push(Reverse((0.0f32.to_bits(), start)));
    while let Some(Reverse((bits, i))) = heap.pop() {
        let d = f32::from_bits(bits);
        if d > dist.get(i).copied().unwrap_or(f32::INFINITY) {
            continue;
        }
        if i == goal {
            break;
        }
        let (x, y) = ((i % w) as i32, (i / w) as i32);
        let tp = tangent.get(i).copied().unwrap_or_default();
        for (k, &(dx, dy)) in DIRS.iter().enumerate() {
            let (nx, ny) = (x + dx, y + dy);
            if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                continue;
            }
            let j = ny as usize * w + nx as usize;
            let cost = node.get(j).copied().unwrap_or(f32::INFINITY);
            if !cost.is_finite() {
                continue;
            }
            let len = if dx != 0 && dy != 0 { SQRT_2 } else { 1.0 };
            let tq = tangent.get(j).copied().unwrap_or_default();
            let nd = d + cost * len + W_DIRECTION * direction_cost(tp, tq, dx, dy, len);
            if let (Some(dj), Some(fj)) = (dist.get_mut(j), from.get_mut(j))
                && nd < *dj
            {
                *dj = nd;
                *fj = k as u8;
                heap.push(Reverse((nd.to_bits(), j)));
            }
        }
    }
    if !dist.get(goal).is_some_and(|d| d.is_finite()) {
        return None;
    }
    let mut cells = vec![goal];
    let mut cur = goal;
    while cur != start && cells.len() <= n {
        let &(dx, dy) = DIRS.get(usize::from(*from.get(cur)?))?;
        let (x, y) = ((cur % w) as i32 - dx, (cur / w) as i32 - dy);
        if x < 0 || y < 0 {
            return None;
        }
        cur = y as usize * w + x as usize;
        cells.push(cur);
    }
    cells.reverse();
    Some(cells)
}

/// The gradient-direction term of a link from a pixel with edge tangent `tp` to its neighbour at
/// (`dx`, `dy`) with tangent `tq`: 0 running straight along an edge, up to 1 cutting across it or
/// turning sharply (Mortensen & Barrett's f_D).
fn direction_cost(tp: [f32; 2], tq: [f32; 2], dx: i32, dy: i32, len: f32) -> f32 {
    if tp == [0.0; 2] || tq == [0.0; 2] {
        return NEUTRAL_DIRECTION;
    }
    let mut l = [dx as f32 / len, dy as f32 / len];
    if tp[0] * l[0] + tp[1] * l[1] < 0.0 {
        l = [-l[0], -l[1]];
    }
    let dp = (tp[0] * l[0] + tp[1] * l[1]).clamp(-1.0, 1.0);
    let dq = (l[0] * tq[0] + l[1] * tq[1]).clamp(-1.0, 1.0);
    2.0 / (3.0 * PI) * (dp.acos() + dq.acos())
}

/// The polyline `g` cut into pieces at most `piece` long; each piece starts where the last ended.
fn split(g: &[[f64; 2]], piece: f64) -> Vec<Vec<[f64; 2]>> {
    let Some(&first) = g.first() else { return Vec::new() };
    let mut out = Vec::new();
    let mut cur = vec![first];
    let mut acc = 0.0;
    for w in g.windows(2) {
        let (mut a, b) = (w[0], w[1]);
        let mut seg = dist(a, b);
        while acc + seg > piece && seg > 0.0 {
            let t = ((piece - acc) / seg).clamp(0.0, 1.0);
            let m = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
            cur.push(m);
            out.push(std::mem::replace(&mut cur, vec![m]));
            a = m;
            seg = dist(a, b);
            acc = 0.0;
        }
        cur.push(b);
        acc += seg;
    }
    out.push(cur);
    out
}

/// Ramer–Douglas–Peucker: drops points within `tol` of the line through their neighbours that
/// stay, keeping both ends.
pub fn simplify(p: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    let n = p.len();
    if n < 3 {
        return p.to_vec();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let tol2 = tol * tol;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b <= a + 1 {
            continue;
        }
        let (Some(&pa), Some(&pb)) = (p.get(a), p.get(b)) else { continue };
        let mut far = (0.0, a);
        for (k, q) in p.iter().enumerate().take(b).skip(a + 1) {
            let d2 = dist2_segment(*q, pa, pb);
            if d2 > far.0 {
                far = (d2, k);
            }
        }
        if far.0 > tol2 {
            if let Some(k) = keep.get_mut(far.1) {
                *k = true;
            }
            stack.push((a, far.1));
            stack.push((far.1, b));
        }
    }
    p.iter().zip(&keep).filter(|(_, k)| **k).map(|(q, _)| *q).collect()
}

/// Length of a polyline.
pub fn length(p: &[[f64; 2]]) -> f64 {
    p.windows(2).map(|w| dist(w[0], w[1])).sum()
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

fn dist2_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (vx, vy) = (b[0] - a[0], b[1] - a[1]);
    let l2 = vx * vx + vy * vy;
    let t = if l2 > 0.0 { (((p[0] - a[0]) * vx + (p[1] - a[1]) * vy) / l2).clamp(0.0, 1.0) } else { 0.0 };
    let (dx, dy) = (p[0] - (a[0] + t * vx), p[1] - (a[1] + t * vy));
    dx * dx + dy * dy
}

#[cfg(test)]
mod tests;
