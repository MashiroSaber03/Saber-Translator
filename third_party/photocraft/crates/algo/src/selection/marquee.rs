//! Marquee shapes built straight into tiles, and the tile-wise combination of two selections
//! (#2888, part of #211).
//!
//! Both produce exactly the bytes the per-pixel versions did (an ellipse written with
//! `Surface::write_pixel`, a combination read and written through whole-canvas `f32` buffers),
//! but work a tile at a time in parallel, keep only the tiles that hold something (as
//! `Surface::prune` would leave them) and share one tile for every fully selected one.

use std::sync::Arc;

use photocraft_color::{PixelFormat, SampleType, read_sample, write_sample};
use photocraft_geom::{Rect, TILE_SIZE, TileCoord};
use photocraft_raster::{Surface, Tile};

use crate::photo_util::par_map;

const TS: usize = TILE_SIZE as usize;

/// The byte a GRAY8 surface stores for `v`.
fn encode8(v: f32) -> u8 {
    let mut b = [0u8];
    write_sample(&mut b, SampleType::U8, 0, v);
    let [b] = b;
    b
}

/// The value a GRAY8 surface reads for `b`.
fn decode8(b: u8) -> f32 {
    read_sample(&[b], SampleType::U8, 0)
}

/// An ellipse inscribed in a rectangle, tested in `f32` exactly as the Elliptical Marquee
/// always has (so its anti-aliased edge keeps the same bytes).
struct Ellipse {
    cx: f32,
    cy: f32,
    rx: f32,
    ry: f32,
}

impl Ellipse {
    fn new(r: Rect) -> Self {
        Self {
            cx: (i64::from(r.x0) + i64::from(r.x1)) as f32 / 2.0,
            cy: (i64::from(r.y0) + i64::from(r.y1)) as f32 / 2.0,
            rx: r.width() as f32 / 2.0,
            ry: r.height() as f32 / 2.0,
        }
    }

    #[inline]
    fn inside(&self, x: f32, y: f32) -> bool {
        let (dx, dy) = ((x - self.cx) / self.rx, (y - self.cy) / self.ry);
        dx * dx + dy * dy <= 1.0
    }

    /// Whether every point of pixel `(x, y)` that the coverage samples is inside: its four
    /// corners (anti-aliased) or its centre. Along a row the pixels where this holds form one
    /// run: each test is monotonic in the distance from the centre (IEEE rounding is monotonic).
    #[inline]
    fn full(&self, x: i32, y: i32, aa: bool) -> bool {
        let (fx, fy) = (x as f32, y as f32);
        if !aa {
            return self.inside(fx + 0.5, fy + 0.5);
        }
        self.inside(fx, fy) && self.inside(fx + 1.0, fy) && self.inside(fx, fy + 1.0) && self.inside(fx + 1.0, fy + 1.0)
    }

    /// Coverage of pixel `(x, y)` in sixteenths: 16 when [`Ellipse::full`], else (anti-aliased)
    /// the number of its 4×4 sub-samples inside.
    #[inline]
    fn coverage(&self, x: i32, y: i32, aa: bool) -> u8 {
        if self.full(x, y, aa) {
            return 16;
        }
        if !aa {
            return 0;
        }
        let (fx, fy) = (x as f32, y as f32);
        (0..16u8).filter(|i| self.inside(fx + ((i % 4) as f32 + 0.5) / 4.0, fy + ((i / 4) as f32 + 0.5) / 4.0)).count() as u8
    }
}

/// One row of the ellipse: nothing outside `lo..hi`, fully selected in `full_lo..full_hi`, and
/// the edge pixels in between (`lo..full_lo`, `full_hi..hi`) need the exact coverage test.
#[derive(Clone, Copy)]
struct RowSpan {
    lo: i32,
    full_lo: i32,
    full_hi: i32,
    hi: i32,
}

/// Bounds row `y` of `e` within `cut.x0..cut.x1` from the exact (f64) ellipse, widened by a
/// margin that covers the f32 tests' rounding. The bounds are then checked with the exact tests
/// (the fully selected run at both ends, nothing just outside); any doubt evaluates the whole row.
fn row_span(e: &Ellipse, y: i32, cut: Rect, aa: bool) -> RowSpan {
    let exact = RowSpan { lo: cut.x0, full_lo: cut.x1, full_hi: cut.x1, hi: cut.x1 };
    let (cx, cy, rx, ry) = (f64::from(e.cx), f64::from(e.cy), f64::from(e.rx), f64::from(e.ry));
    // Every point the tests sample in this row lies in `y..y + 1`.
    let (ya, yb) = (f64::from(y), f64::from(y) + 1.0);
    let near = if cy < ya { ya - cy } else { (cy - yb).max(0.0) };
    let far = (ya - cy).abs().max((yb - cy).abs());
    let half = |d: f64| {
        let t = 1.0 - (d / ry) * (d / ry);
        if t >= 0.0 { rx * t.sqrt() } else { -1.0 }
    };
    let (outer, inner) = (half(near).max(0.0), half(far));
    // Near the top and bottom a rounding error δ in the test moves the edge by up to rx·√δ.
    let m = 2.0 + rx * 1e-3;
    if !(outer.is_finite() && inner.is_finite() && m.is_finite()) {
        return exact;
    }
    let px = |v: f64| v.clamp(f64::from(cut.x0), f64::from(cut.x1)) as i32;
    let lo = px((cx - outer - m).floor());
    let hi = px((cx + outer + m).floor() + 1.0);
    let (mut full_lo, mut full_hi) = (px((cx - inner + m).ceil()), px((cx + inner - m).floor()));
    if inner < 0.0 || full_lo >= full_hi {
        (full_lo, full_hi) = (hi, hi);
    }
    let empty = |x: i32| e.coverage(x, y, aa) == 0;
    let ok = (lo <= cut.x0 || empty(lo - 1)) && (hi >= cut.x1 || empty(hi)) && (full_lo >= full_hi || (e.full(full_lo, y, aa) && e.full(full_hi - 1, y, aa)));
    if ok { RowSpan { lo, full_lo, full_hi, hi } } else { exact }
}

/// The Elliptical Marquee's shape: the ellipse inscribed in `r`, cut to `clip`, as a GRAY8
/// selection. Anti-aliased edge pixels hold the fraction of their 4×4 sub-samples inside;
/// without anti-aliasing a pixel is selected when its centre is inside.
pub fn ellipse_surface(r: Rect, clip: Rect, anti_alias: bool) -> Surface {
    let mut s = Surface::new(PixelFormat::GRAY8);
    let cut = r.intersect(&clip);
    if cut.is_empty() {
        return s;
    }
    let e = Ellipse::new(r);
    let enc: [u8; 17] = std::array::from_fn(|n| encode8(n as f32 / 16.0));
    let one = encode8(1.0);
    let rows = par_map(cut.height() as usize, |i| row_span(&e, cut.y0 + i as i32, cut, anti_alias));
    let span = |y: i32| rows.get(usize::try_from(y - cut.y0).ok()?);
    let full = s.solid_tile(&[1.0]);
    let blank = s.solid_tile(&[0.0]);
    let coords: Vec<TileCoord> = cut.tiles().collect();
    let built = par_map(coords.len(), |k| {
        let tc = *coords.get(k)?;
        let rect = tc.rect();
        let tr = rect.intersect(&cut);
        if tr == rect && (tr.y0..tr.y1).all(|y| span(y).is_some_and(|s| s.full_lo <= tr.x0 && s.full_hi >= tr.x1)) {
            return Some((tc, Arc::clone(&full)));
        }
        let mut tile: Option<Tile> = None;
        for y in tr.y0..tr.y1 {
            let Some(sp) = span(y) else { continue };
            let row = (y - rect.y0) as usize * TS;
            let at = |x: i32| row + (x - rect.x0) as usize;
            let (a, b) = (sp.full_lo.max(tr.x0), sp.full_hi.min(tr.x1));
            if a < b
                && let Some(d) = tile.get_or_insert_with(|| Tile::clone(&blank)).bytes_mut().get_mut(at(a)..at(b))
            {
                d.fill(one);
            }
            for x in (sp.lo.max(tr.x0)..sp.full_lo.min(tr.x1)).chain(sp.full_hi.max(tr.x0)..sp.hi.min(tr.x1)) {
                let n = e.coverage(x, y, anti_alias);
                if n > 0
                    && let Some(d) = tile.get_or_insert_with(|| Tile::clone(&blank)).bytes_mut().get_mut(at(x))
                {
                    *d = enc.get(usize::from(n)).copied().unwrap_or(one);
                }
            }
        }
        tile.map(|t| (tc, Arc::new(t)))
    });
    s.put_tiles(built.into_iter().flatten());
    s
}

/// `f(a, b)` per pixel over `area` for two selections, as a GRAY8 selection holding nothing
/// outside `area`: the same bytes as reading both over `area`, mapping and writing the result.
/// GRAY8 inputs (every selection) go tile by tile in parallel through a table of all 256×256
/// byte pairs; a tile `f` leaves unchanged stays shared with its input.
pub fn combine_surfaces(a: &Surface, b: &Surface, area: Rect, f: impl Fn(f32, f32) -> f32) -> Surface {
    let mut out = Surface::new(PixelFormat::GRAY8);
    if area.is_empty() {
        return out;
    }
    if a.format() != PixelFormat::GRAY8 || b.format() != PixelFormat::GRAY8 {
        let (ra, rb) = (a.read_region(area), b.read_region(area));
        let data: Vec<f32> = ra.iter().zip(&rb).map(|(x, y)| f(*x, *y)).collect();
        if data.len() == area.width() as usize * area.height() as usize {
            out.write_region(area, &data);
            out.prune();
        }
        return out;
    }
    let lut: Vec<u8> = (0..=u16::MAX).map(|i| encode8(f(decode8((i >> 8) as u8), decode8(i as u8)))).collect();
    let pair = |x: u8, y: u8| lut.get(usize::from(x) << 8 | usize::from(y)).copied().unwrap_or(0);
    let (da, db) = (a.default_bytes().first().copied().unwrap_or(0), b.default_bytes().first().copied().unwrap_or(0));
    // Whether `f` passes one side through where the other side has no tile (e.g. Add, Subtract).
    let keeps_a = (0..=255u8).all(|v| pair(v, db) == v);
    let keeps_b = (0..=255u8).all(|v| pair(da, v) == v);
    let (full, blank) = (out.solid_tile(&[1.0]), out.solid_tile(&[0.0]));
    let (row_a, row_b) = ([da; TS], [db; TS]);
    let coords: Vec<TileCoord> = area.tiles().collect();
    let built = par_map(coords.len(), |k| {
        let tc = *coords.get(k)?;
        let rect = tc.rect();
        let tr = rect.intersect(&area);
        let (ta, tb) = (a.tile(tc), b.tile(tc));
        if tr == rect {
            let pass = match (ta, tb) {
                (Some(t), None) if keeps_a => Some(t),
                (None, Some(t)) if keeps_b => Some(t),
                (None, None) if pair(da, db) == 0 => return None,
                _ => None,
            };
            if let Some(t) = pass {
                return t.bytes().iter().any(|v| *v != 0).then(|| (tc, Arc::clone(t)));
            }
        }
        let mut tile = Tile::clone(&blank);
        let data = tile.bytes_mut();
        let (mut lo, mut hi) = (if tr == rect { u8::MAX } else { 0 }, 0u8);
        let (x0, x1) = ((tr.x0 - rect.x0) as usize, (tr.x1 - rect.x0) as usize);
        for ly in (tr.y0 - rect.y0) as usize..(tr.y1 - rect.y0) as usize {
            let (s, e) = (ly * TS + x0, ly * TS + x1);
            let src_a = ta.and_then(|t| t.bytes().get(s..e)).unwrap_or(&row_a[..x1 - x0]);
            let src_b = tb.and_then(|t| t.bytes().get(s..e)).unwrap_or(&row_b[..x1 - x0]);
            let Some(dst) = data.get_mut(s..e) else { continue };
            for (d, (x, y)) in dst.iter_mut().zip(src_a.iter().zip(src_b)) {
                *d = pair(*x, *y);
                (lo, hi) = (lo.min(*d), hi.max(*d));
            }
        }
        (hi > 0).then(|| (tc, if lo == u8::MAX { Arc::clone(&full) } else { Arc::new(tile) }))
    });
    out.put_tiles(built.into_iter().flatten());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Elliptical Marquee as it was: every pixel tested and written one at a time.
    fn old_ellipse(r: Rect, area: Rect, aa: bool) -> Surface {
        let cut = r.intersect(&area);
        let mut shape = Surface::new(PixelFormat::GRAY8);
        let (cx, cy) = ((r.x0 + r.x1) as f32 / 2.0, (r.y0 + r.y1) as f32 / 2.0);
        let (rx, ry) = (r.width() as f32 / 2.0, r.height() as f32 / 2.0);
        let inside = |x: f32, y: f32| {
            let (dx, dy) = ((x - cx) / rx, (y - cy) / ry);
            dx * dx + dy * dy <= 1.0
        };
        for y in cut.y0..cut.y1 {
            for x in cut.x0..cut.x1 {
                let (fx, fy) = (x as f32, y as f32);
                let corners = [(fx, fy), (fx + 1.0, fy), (fx, fy + 1.0), (fx + 1.0, fy + 1.0)].iter().filter(|(a, b)| inside(*a, *b)).count();
                let cov = if !aa {
                    if inside(fx + 0.5, fy + 0.5) { 1.0 } else { 0.0 }
                } else if corners == 4 {
                    1.0
                } else {
                    let n = (0..16).filter(|i| inside(fx + ((i % 4) as f32 + 0.5) / 4.0, fy + ((i / 4) as f32 + 0.5) / 4.0)).count();
                    n as f32 / 16.0
                };
                if cov > 0.0 {
                    shape.write_pixel(x, y, &[cov]);
                }
            }
        }
        shape
    }

    /// Combining two selections as it was: whole-area `f32` buffers.
    fn old_combine(a: &Surface, b: &Surface, area: Rect, f: impl Fn(f32, f32) -> f32) -> Surface {
        let (ra, rb) = (a.read_region(area), b.read_region(area));
        let data: Vec<f32> = ra.iter().zip(&rb).map(|(x, y)| f(*x, *y)).collect();
        let mut out = Surface::new(PixelFormat::GRAY8);
        out.write_region(area, &data);
        out.prune();
        out
    }

    #[test]
    fn marquee_matches_the_per_pixel_version() {
        let area = Rect::new(0, 0, 1100, 700);
        let shapes = [
            Rect::from_xywh(40, 30, 1000, 620), // whole tiles inside
            Rect::from_xywh(3, 5, 1, 1),
            Rect::from_xywh(10, 10, 2, 7),
            Rect::from_xywh(100, 200, 37, 301),
            Rect::from_xywh(255, 255, 258, 2),
            Rect::from_xywh(-300, -200, 900, 600),   // dragged past the canvas
            Rect::from_xywh(-500, -500, 2100, 1700), // larger than the canvas
            Rect::from_xywh(1090, 690, 40, 40),
            Rect::from_xywh(2000, 0, 50, 50), // entirely outside
        ];
        for r in shapes {
            for aa in [true, false] {
                let new = ellipse_surface(r, area, aa);
                assert!(new == old_ellipse(r, area, aa), "ellipse {r:?} aa={aa}");
            }
        }
        // The other modes, against a rectangle and an anti-aliased ellipse with a partial edge.
        let mut rect = Surface::new(PixelFormat::GRAY8);
        rect.fill_rect(Rect::from_xywh(200, 100, 600, 520), &[1.0]);
        let ell = old_ellipse(Rect::from_xywh(-60, 250, 700, 500), area, true);
        let ops: [fn(f32, f32) -> f32; 3] = [|a, b| a.max(b), |a, b| a * (1.0 - b), |a, b| a.min(b)];
        for (a, b) in [(&rect, &ell), (&ell, &rect), (&ell, &ell)] {
            for f in ops {
                assert!(combine_surfaces(a, b, area, f) == old_combine(a, b, area, f));
            }
        }
    }
}
