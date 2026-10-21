//! Anti-aliased polygon rasterizer with exact area coverage and boolean combination.
//!
//! Each pixel row is cut into horizontal sub-bands at every edge endpoint and every edge
//! crossing inside the row, so that within a sub-band all active edges span its full height and
//! keep their left-to-right order. Walking the edges in that order with one winding counter per
//! *component* tells, between any two consecutive edges, whether the region is inside the
//! combined shape (non-zero or even-odd per component, folded with the path operations). The
//! boundary edges of the inside intervals are then accumulated as signed trapezoid areas ("area
//! to the right of the edge", +1 for a left boundary, −1 for a right boundary) into a per-row
//! cell buffer whose prefix sum is the exact covered area of every pixel. This is the classic
//! scanline/area-coverage technique (see e.g. FreeType's gray rasterizer or libart), written
//! from first principles here; it needs no sub-sampling and handles overlapping components
//! exactly.

use photocraft_doc::{FillRule, PathOp};
use photocraft_geom::Rect;

/// Coverage below this is f32 round-off in the cell prefix sum, not area: it is well under half
/// a 16-bit step (7.6e-6), so 8- and 16-bit output is unchanged by dropping it.
const RESIDUE: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
struct Edge {
    x0: f64,
    y0: f64,
    y1: f64,
    /// +1 when the source segment runs downwards (y increasing), −1 upwards.
    dir: i32,
    comp: u32,
    dxdy: f64,
}

impl Edge {
    #[inline]
    fn x_at(&self, y: f64) -> f64 {
        self.x0 + (y - self.y0) * self.dxdy
    }
}

/// A combination step: a component (set of polygons filled with `rule`) and how it folds into
/// the result so far.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Component {
    pub op: PathOp,
    pub rule: FillRule,
}

/// A compiled set of polygons, ready to render coverage for any rectangle.
#[derive(Clone, Debug, Default)]
pub struct Rasterizer {
    /// Sorted by `y0`.
    edges: Vec<Edge>,
    comps: Vec<Component>,
    inverted: bool,
    bounds: Option<(f64, f64, f64, f64)>,
    /// Every component combines and none is inverted: inside = any component inside.
    union_only: bool,
}

impl Rasterizer {
    /// Empty rasterizer (renders nothing, or everything when `inverted`: the final result is
    /// inverted).
    pub fn new(inverted: bool) -> Self {
        Rasterizer { inverted, union_only: true, ..Default::default() }
    }

    /// Adds a component made of closed polygons (open input is closed implicitly).
    pub fn add_component(&mut self, polys: &[Vec<(f64, f64)>], op: PathOp, rule: FillRule) {
        let comp = self.comps.len() as u32;
        self.comps.push(Component { op, rule });
        if op != PathOp::Combine && comp != 0 {
            self.union_only = false;
        }
        for poly in polys {
            let n = poly.len();
            if n < 2 {
                continue;
            }
            for i in 0..n {
                let a = poly[i];
                let b = poly[(i + 1) % n];
                if !(a.0.is_finite() && a.1.is_finite() && b.0.is_finite() && b.1.is_finite()) {
                    continue;
                }
                self.bounds = Some(match self.bounds {
                    None => (a.0.min(b.0), a.1.min(b.1), a.0.max(b.0), a.1.max(b.1)),
                    Some(bb) => (bb.0.min(a.0.min(b.0)), bb.1.min(a.1.min(b.1)), bb.2.max(a.0.max(b.0)), bb.3.max(a.1.max(b.1))),
                });
                if a.1 == b.1 {
                    continue;
                }
                let (p, q, dir) = if a.1 < b.1 { (a, b, 1) } else { (b, a, -1) };
                self.edges.push(Edge { x0: p.0, y0: p.1, y1: q.1, dir, comp, dxdy: (q.0 - p.0) / (q.1 - p.1) });
            }
        }
        self.edges.sort_by(|a, b| a.y0.total_cmp(&b.y0));
    }

    /// Pixel bounds of everything that may be covered (`None` = nothing; the whole plane
    /// when inverted is reported as `None` too — callers clip to their own area). The upper
    /// edge is clipped to `i32::MAX`, the largest representable exclusive rectangle edge.
    pub fn pixel_bounds(&self) -> Option<Rect> {
        if self.inverted {
            return None;
        }
        let (x0, y0, x1, y1) = self.bounds?;
        let r = Rect::new(x0.floor() as i32, y0.floor() as i32, (x1.ceil() as i32).saturating_add(1), (y1.ceil() as i32).saturating_add(1));
        (!r.is_empty()).then_some(r)
    }

    pub fn is_inverted(&self) -> bool {
        self.inverted
    }

    /// Whether the combined shape is inside given per-component winding numbers.
    #[inline]
    fn inside(&self, wind: &[i32], union_count: i32) -> bool {
        if self.union_only {
            return (union_count > 0) ^ self.inverted;
        }
        let mut acc = false;
        for (i, c) in self.comps.iter().enumerate() {
            let v = match c.rule {
                FillRule::NonZero => wind[i] != 0,
                FillRule::EvenOdd => wind[i] & 1 != 0,
            };
            // Photoshop applies the first component as "combine".
            let op = if i == 0 { PathOp::Combine } else { c.op };
            acc = match op {
                // A joined component (only reachable when built by hand) combines.
                PathOp::Combine | PathOp::Join => acc | v,
                PathOp::Subtract => acc & !v,
                PathOp::Intersect => acc & v,
                PathOp::Exclude => acc ^ v,
            };
        }
        acc ^ self.inverted
    }

    fn comp_inside(&self, comp: usize, w: i32) -> bool {
        match self.comps[comp].rule {
            FillRule::NonZero => w != 0,
            FillRule::EvenOdd => w & 1 != 0,
        }
    }

    /// Coverage (`0..=1`) of every pixel of `rect`, row-major. Rows are rendered in parallel
    /// strips on native targets.
    pub fn render(&self, rect: Rect) -> Vec<f32> {
        let w = rect.width() as usize;
        let h = rect.height() as usize;
        let mut out = vec![0.0f32; w * h];
        if w == 0 || h == 0 {
            return out;
        }
        #[cfg(not(target_arch = "wasm32"))]
        const STRIP: usize = 32;
        #[cfg(not(target_arch = "wasm32"))]
        if h > STRIP && w * h > 64 * 1024 {
            use rayon::prelude::*;
            out.par_chunks_mut(w * STRIP).enumerate().for_each(|(i, chunk)| {
                let y0 = rect.y0 + (i * STRIP) as i32;
                let rows = chunk.len() / w;
                self.render_into(Rect::new(rect.x0, y0, rect.x1, y0 + rows as i32), chunk);
            });
            return out;
        }
        self.render_into(rect, &mut out);
        out
    }

    /// Renders `rect` into `out` (`rect.width() * rect.height()` values), sequentially.
    pub fn render_into(&self, rect: Rect, out: &mut [f32]) {
        let w = rect.width() as usize;
        let mut st = RowState::new(self.comps.len(), w);
        // Edges are sorted by y0; `next` is the first edge not yet activated.
        let top = f64::from(rect.y0);
        let mut next = self.edges.partition_point(|e| e.y0 < top);
        st.active.extend((0..next).filter(|&i| self.edges[i].y1 > top));
        for (row, y) in (rect.y0..rect.y1).enumerate() {
            let (ya, yb) = (f64::from(y), f64::from(y) + 1.0);
            st.active.retain(|&i| self.edges[i].y1 > ya);
            while next < self.edges.len() && self.edges[next].y0 < yb {
                if self.edges[next].y1 > ya {
                    st.active.push(next);
                }
                next += 1;
            }
            let acc = &mut st.acc;
            acc.iter_mut().for_each(|v| *v = 0.0);
            if st.active.is_empty() {
                if self.inverted {
                    out[row * w..(row + 1) * w].iter_mut().for_each(|v| *v = 1.0);
                } else {
                    out[row * w..(row + 1) * w].iter_mut().for_each(|v| *v = 0.0);
                }
                continue;
            }
            // Sub-band boundaries: the row edges plus every edge endpoint inside the row.
            st.cuts.clear();
            st.cuts.push(ya);
            st.cuts.push(yb);
            for &i in &st.active {
                let e = &self.edges[i];
                if e.y0 > ya && e.y0 < yb {
                    st.cuts.push(e.y0);
                }
                if e.y1 > ya && e.y1 < yb {
                    st.cuts.push(e.y1);
                }
            }
            st.cuts.sort_by(f64::total_cmp);
            st.cuts.dedup();
            let cuts = std::mem::take(&mut st.cuts);
            for c in cuts.windows(2) {
                self.band(c[0], c[1], rect.x0, &mut st, 0);
            }
            st.cuts = cuts;
            let row_out = &mut out[row * w..(row + 1) * w];
            let mut sum = 0.0f32;
            for (o, a) in row_out.iter_mut().zip(st.acc.iter()) {
                sum += *a;
                // Opposite boundaries don't cancel exactly in f32, leaving ~1e-7 just past a
                // shape's right edge; that residue would count as content (#2537). Snap it (and
                // a NaN) to 0.
                *o = if sum >= RESIDUE { sum.min(1.0) } else { 0.0 };
            }
        }
    }

    /// Accumulates the sub-band `[a, b)` of the current row.
    fn band(&self, a: f64, b: f64, x0: i32, st: &mut RowState, depth: u32) {
        if b - a <= 1e-12 {
            return;
        }
        st.band.clear();
        let mid = 0.5 * (a + b);
        for &i in &st.active {
            let e = &self.edges[i];
            if e.y0 <= mid && e.y1 >= mid {
                let (xa, xb) = (e.x_at(a), e.x_at(b));
                st.band.push(BandEdge { xa, xb, xm: 0.5 * (xa + xb), idx: i as u32 });
            }
        }
        if st.band.is_empty() {
            if self.inverted {
                add_full(&mut st.acc, b - a);
            }
            return;
        }
        st.band.sort_by(|p, q| p.xm.total_cmp(&q.xm));
        // An inversion of the order at the band's top or bottom means two edges cross inside
        // it: split there so the order is constant in every piece.
        if depth < 24 {
            for k in 1..st.band.len() {
                let (p, q) = (st.band[k - 1], st.band[k]);
                let top = p.xa - q.xa;
                let bot = p.xb - q.xb;
                if top > 1e-9 || bot > 1e-9 {
                    // Crossing where (p - q)(y) = 0, linear in y over the band.
                    let t = if (top - bot).abs() > 1e-15 { top / (top - bot) } else { 0.5 };
                    let y = a + (b - a) * t.clamp(0.0, 1.0);
                    let y = if y <= a + 1e-9 || y >= b - 1e-9 { mid } else { y };
                    self.band(a, y, x0, st, depth + 1);
                    self.band(y, b, x0, st, depth + 1);
                    return;
                }
            }
        }
        st.wind.iter_mut().for_each(|w| *w = 0);
        let mut union_count = 0i32;
        let mut inside = self.inverted;
        if inside {
            add_full(&mut st.acc, b - a);
        }
        let n = st.band.len();
        for k in 0..n {
            let be = st.band[k];
            let e = &self.edges[be.idx as usize];
            let c = e.comp as usize;
            if self.union_only {
                let was = self.comp_inside(c, st.wind[c]);
                st.wind[c] += e.dir;
                let now = self.comp_inside(c, st.wind[c]);
                union_count += i32::from(now) - i32::from(was);
            } else {
                st.wind[c] += e.dir;
            }
            let now = self.inside(&st.wind, union_count);
            if now != inside {
                let w = if now { 1.0 } else { -1.0 };
                draw(&mut st.acc, x0, be.xa, a, be.xb, b, w);
                inside = now;
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct BandEdge {
    xa: f64,
    xb: f64,
    xm: f64,
    idx: u32,
}

struct RowState {
    active: Vec<usize>,
    cuts: Vec<f64>,
    band: Vec<BandEdge>,
    wind: Vec<i32>,
    /// Cell accumulator: `width + 1` cells (the extra one absorbs spill-over).
    acc: Vec<f32>,
}

impl RowState {
    fn new(comps: usize, width: usize) -> Self {
        RowState { active: Vec::new(), cuts: Vec::new(), band: Vec::new(), wind: vec![0; comps.max(1)], acc: vec![0.0; width + 1] }
    }
}

/// Coverage `h` over the whole row (a boundary left of the rectangle).
#[inline]
fn add_full(acc: &mut [f32], h: f64) {
    acc[0] += h as f32;
}

/// Adds `w` × the area to the right of the segment `(xa, ya) → (xb, yb)` (`ya < yb`, both inside
/// one pixel row) to the cell accumulator whose cell 0 is column `x0`.
fn draw(acc: &mut [f32], x0: i32, xa: f64, ya: f64, xb: f64, yb: f64, w: f64) {
    let width = (acc.len() - 1) as f64;
    let left = f64::from(x0);
    // Work in rectangle-relative x.
    let (mut ua, mut ub) = (xa - left, xb - left);
    let dy = yb - ya;
    if dy <= 0.0 {
        return;
    }
    // Order by x; the vertical extent per unit x is what matters.
    if ua > ub {
        std::mem::swap(&mut ua, &mut ub);
    }
    let dx = ub - ua;
    let h_of = |u0: f64, u1: f64| if dx > 1e-12 { dy * (u1 - u0) / dx } else { dy };
    // Part left of the rectangle: full coverage of every column.
    if ua < 0.0 {
        let cut = ub.min(0.0);
        acc[0] += (w * h_of(ua, cut)) as f32;
        if ub <= 0.0 {
            return;
        }
        ua = 0.0;
    }
    if ua >= width {
        return;
    }
    let end = ub.min(width);
    if dx <= 1e-12 {
        let c = ua.floor();
        let f = ua - c;
        let ci = c as usize;
        acc[ci] += (w * dy * (1.0 - f)) as f32;
        acc[ci + 1] += (w * dy * f) as f32;
        return;
    }
    let mut u = ua;
    while u < end {
        let c = u.floor();
        let v = (c + 1.0).min(end);
        let h = h_of(u, v);
        let m = 0.5 * (u + v) - c;
        let ci = c as usize;
        acc[ci] += (w * h * (1.0 - m)) as f32;
        acc[ci + 1] += (w * h * m) as f32;
        if v <= u {
            break;
        }
        u = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coverage_surface;
    use photocraft_color::PixelFormat;

    fn square(x: f64, y: f64, s: f64) -> Vec<(f64, f64)> {
        vec![(x, y), (x + s, y), (x + s, y + s), (x, y + s)]
    }

    fn sum(v: &[f32]) -> f64 {
        v.iter().map(|x| f64::from(*x)).sum()
    }

    #[test]
    fn axis_aligned_square_exact() {
        let mut r = Rasterizer::new(false);
        r.add_component(&[square(2.25, 3.5, 4.0)], PathOp::Combine, FillRule::NonZero);
        let c = r.render(Rect::new(0, 0, 10, 10));
        assert!((sum(&c) - 16.0).abs() < 1e-4);
        assert!((c[3 * 10 + 2] - 0.75 * 0.5).abs() < 1e-5);
        assert!((c[5 * 10 + 4] - 1.0).abs() < 1e-6);
        assert_eq!(c[0], 0.0);
    }

    #[test]
    fn clipped_rect_matches_full_render() {
        let mut r = Rasterizer::new(false);
        r.add_component(&[vec![(1.3, 0.7), (17.9, 5.2), (8.1, 19.6)]], PathOp::Combine, FillRule::NonZero);
        let full = r.render(Rect::new(0, 0, 20, 20));
        let part = r.render(Rect::new(7, 4, 13, 15));
        for y in 4..15 {
            for x in 7..13 {
                let a = full[y * 20 + x];
                let b = part[(y - 4) * 6 + (x - 7)];
                assert!((a - b).abs() < 1e-5, "{x},{y}: {a} vs {b}");
            }
        }
        // Triangle area by the shoelace formula.
        let area = 0.5 * ((17.9f64 - 1.3) * (19.6 - 0.7) - (8.1 - 1.3) * (5.2 - 0.7)).abs();
        assert!((sum(&full) - area).abs() < 1e-3);
    }

    #[test]
    fn inverted_empty_is_full() {
        let r = Rasterizer::new(true);
        assert!(r.render(Rect::new(0, 0, 4, 4)).iter().all(|v| *v == 1.0));
    }

    #[test]
    fn pixel_bounds_clip_upper_edge_at_i32_max() {
        let mut r = Rasterizer::new(false);
        let max = f64::from(i32::MAX);
        r.add_component(&[square(max - 2.0, max - 2.0, 4.0)], PathOp::Combine, FillRule::NonZero);

        assert_eq!(r.pixel_bounds(), Some(Rect::new(i32::MAX - 2, i32::MAX - 2, i32::MAX, i32::MAX)));

        // Raster work clipped to a normal document area stays empty and does not use the
        // extreme bounds as an allocation size.
        let clipped = coverage_surface(&r, PixelFormat::GRAY8, Rect::new(0, 0, 8, 8));
        assert_eq!(clipped.tile_count(), 0);
    }

    #[test]
    fn pixel_bounds_beyond_i32_max_are_empty() {
        let mut r = Rasterizer::new(false);
        let max = f64::from(i32::MAX);
        r.add_component(&[square(max + 10.0, max + 10.0, 4.0)], PathOp::Combine, FillRule::NonZero);

        assert_eq!(r.pixel_bounds(), None);
    }

    #[test]
    fn pixel_bounds_keep_normal_boundary_rounding() {
        let mut r = Rasterizer::new(false);
        r.add_component(&[square(1.0, 2.0, 3.0)], PathOp::Combine, FillRule::NonZero);

        assert_eq!(r.pixel_bounds(), Some(Rect::new(1, 2, 5, 6)));
    }
}
