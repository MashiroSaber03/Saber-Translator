//! Geometry primitives shared by every Photocraft crate.
//!
//! Document space is integer pixels with the origin at the canvas top-left.
//! Layers may extend beyond the canvas (negative coordinates are valid).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use serde::{Deserialize, Serialize};

pub mod warp;

/// Edge length of a raster tile in pixels.
pub const TILE_SIZE: i32 = 256;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Size {
    pub width: u32,
    pub height: u32,
}

impl Size {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
    pub fn area(self) -> u64 {
        self.width as u64 * self.height as u64
    }
    pub fn is_empty(self) -> bool {
        self.width == 0 || self.height == 0
    }
}

/// Integer rectangle, half-open: `[x0, x1) × [y0, y1)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect {
    pub x0: i32,
    pub y0: i32,
    pub x1: i32,
    pub y1: i32,
}

impl Rect {
    pub const EMPTY: Rect = Rect { x0: 0, y0: 0, x1: 0, y1: 0 };

    pub const fn new(x0: i32, y0: i32, x1: i32, y1: i32) -> Self {
        Self { x0, y0, x1, y1 }
    }
    pub fn from_xywh(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self::new(x, y, x.saturating_add(w as i32), y.saturating_add(h as i32))
    }
    pub fn from_size(size: Size) -> Self {
        Self::from_xywh(0, 0, size.width, size.height)
    }
    /// Horizontal extent; 0 when `x1 <= x0`. Exact even for i32-extreme edges.
    pub fn width(&self) -> u32 {
        if self.x1 > self.x0 { self.x1.abs_diff(self.x0) } else { 0 }
    }
    /// Vertical extent; 0 when `y1 <= y0`. Exact even for i32-extreme edges.
    pub fn height(&self) -> u32 {
        if self.y1 > self.y0 { self.y1.abs_diff(self.y0) } else { 0 }
    }
    pub fn size(&self) -> Size {
        Size::new(self.width(), self.height())
    }
    pub fn is_empty(&self) -> bool {
        self.x1 <= self.x0 || self.y1 <= self.y0
    }
    /// `other` lies entirely inside `self` (an empty `other` is contained in anything).
    pub fn contains_rect(&self, other: &Rect) -> bool {
        other.is_empty() || (other.x0 >= self.x0 && other.y0 >= self.y0 && other.x1 <= self.x1 && other.y1 <= self.y1)
    }
    pub fn contains(&self, x: i32, y: i32) -> bool {
        x >= self.x0 && x < self.x1 && y >= self.y0 && y < self.y1
    }
    pub fn intersect(&self, o: &Rect) -> Rect {
        let r = Rect::new(self.x0.max(o.x0), self.y0.max(o.y0), self.x1.min(o.x1), self.y1.min(o.y1));
        if r.is_empty() { Rect::EMPTY } else { r }
    }
    pub fn union(&self, o: &Rect) -> Rect {
        if self.is_empty() {
            return *o;
        }
        if o.is_empty() {
            return *self;
        }
        Rect::new(self.x0.min(o.x0), self.y0.min(o.y0), self.x1.max(o.x1), self.y1.max(o.y1))
    }
    pub fn translate(&self, dx: i32, dy: i32) -> Rect {
        Rect::new(self.x0.saturating_add(dx), self.y0.saturating_add(dy), self.x1.saturating_add(dx), self.y1.saturating_add(dy))
    }
    pub fn inflate(&self, d: i32) -> Rect {
        Rect::new(self.x0.saturating_sub(d), self.y0.saturating_sub(d), self.x1.saturating_add(d), self.y1.saturating_add(d))
    }
    /// All tiles overlapping this rect.
    pub fn tiles(&self) -> impl Iterator<Item = TileCoord> + use<> {
        let (a, b) = if self.is_empty() {
            (TileCoord::new(0, 0), TileCoord::new(-1, -1))
        } else {
            (TileCoord::containing(self.x0, self.y0), TileCoord::containing(self.x1 - 1, self.y1 - 1))
        };
        (a.ty..=b.ty).flat_map(move |ty| (a.tx..=b.tx).map(move |tx| TileCoord::new(tx, ty)))
    }
}

/// Index of a tile in the infinite tile grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TileCoord {
    pub tx: i32,
    pub ty: i32,
}

impl TileCoord {
    pub const fn new(tx: i32, ty: i32) -> Self {
        Self { tx, ty }
    }
    pub fn containing(x: i32, y: i32) -> Self {
        Self::new(x.div_euclid(TILE_SIZE), y.div_euclid(TILE_SIZE))
    }
    pub fn rect(&self) -> Rect {
        Rect::from_xywh(self.tx * TILE_SIZE, self.ty * TILE_SIZE, TILE_SIZE as u32, TILE_SIZE as u32)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// 2D affine transform `[a c e; b d f; 0 0 1]` (column vectors).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Affine {
    pub m: [f64; 6],
}

impl Default for Affine {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Affine {
    pub const IDENTITY: Affine = Affine { m: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0] };

    pub fn translate(dx: f64, dy: f64) -> Self {
        Affine { m: [1.0, 0.0, 0.0, 1.0, dx, dy] }
    }
    pub fn scale(s: f64) -> Self {
        Affine { m: [s, 0.0, 0.0, s, 0.0, 0.0] }
    }
    pub fn rotate(radians: f64) -> Self {
        let (s, c) = radians.sin_cos();
        Affine { m: [c, s, -s, c, 0.0, 0.0] }
    }
    /// `self * other`: apply `other` first, then `self`.
    pub fn then(&self, other: &Affine) -> Affine {
        other.mul(self)
    }
    pub fn mul(&self, o: &Affine) -> Affine {
        let [a, b, c, d, e, f] = self.m;
        let [oa, ob, oc, od, oe, of] = o.m;
        Affine { m: [a * oa + c * ob, b * oa + d * ob, a * oc + c * od, b * oc + d * od, a * oe + c * of + e, b * oe + d * of + f] }
    }
    pub fn apply(&self, p: Point) -> Point {
        let [a, b, c, d, e, f] = self.m;
        Point::new(a * p.x + c * p.y + e, b * p.x + d * p.y + f)
    }
    pub fn determinant(&self) -> f64 {
        self.m[0] * self.m[3] - self.m[1] * self.m[2]
    }
    pub fn inverse(&self) -> Option<Affine> {
        let det = self.determinant();
        if det.abs() < 1e-12 {
            return None;
        }
        let [a, b, c, d, e, f] = self.m;
        let inv = 1.0 / det;
        Some(Affine { m: [d * inv, -b * inv, -c * inv, a * inv, (c * f - d * e) * inv, (b * e - a * f) * inv] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rect_basics() {
        let r = Rect::from_xywh(10, 20, 30, 40);
        assert_eq!(r.width(), 30);
        assert_eq!(r.height(), 40);
        assert!(r.contains(10, 20));
        assert!(!r.contains(40, 20));
        assert!(Rect::new(5, 5, 5, 10).is_empty());
    }

    #[test]
    fn intersect_and_union() {
        let a = Rect::new(0, 0, 10, 10);
        let b = Rect::new(5, 5, 15, 15);
        assert_eq!(a.intersect(&b), Rect::new(5, 5, 10, 10));
        assert_eq!(a.union(&b), Rect::new(0, 0, 15, 15));
        assert_eq!(a.intersect(&Rect::new(20, 20, 30, 30)), Rect::EMPTY);
        assert_eq!(Rect::EMPTY.union(&a), a);
    }

    #[test]
    fn tile_coords_handle_negatives() {
        assert_eq!(TileCoord::containing(0, 0), TileCoord::new(0, 0));
        assert_eq!(TileCoord::containing(255, 255), TileCoord::new(0, 0));
        assert_eq!(TileCoord::containing(256, 0), TileCoord::new(1, 0));
        assert_eq!(TileCoord::containing(-1, -1), TileCoord::new(-1, -1));
        assert_eq!(TileCoord::containing(-256, -257), TileCoord::new(-1, -2));
    }

    #[test]
    fn rect_tiles_enumeration() {
        let r = Rect::new(-10, 0, 300, 10);
        let t: Vec<_> = r.tiles().collect();
        assert_eq!(t, vec![TileCoord::new(-1, 0), TileCoord::new(0, 0), TileCoord::new(1, 0)]);
        assert_eq!(Rect::EMPTY.tiles().count(), 0);
        assert_eq!(Rect::new(0, 0, 256, 256).tiles().count(), 1);
        assert_eq!(Rect::new(0, 0, 257, 257).tiles().count(), 4);
    }

    #[test]
    fn affine_inverse_roundtrip() {
        let t = Affine::translate(5.0, -3.0).then(&Affine::rotate(0.7)).then(&Affine::scale(2.5));
        let inv = t.inverse().unwrap();
        let p = Point::new(12.0, -7.5);
        let q = inv.apply(t.apply(p));
        assert!((q.x - p.x).abs() < 1e-9 && (q.y - p.y).abs() < 1e-9);
        assert!(Affine::scale(0.0).inverse().is_none());
    }

    #[test]
    fn affine_then_order() {
        let t = Affine::translate(10.0, 0.0).then(&Affine::scale(2.0));
        // translate first, then scale
        assert_eq!(t.apply(Point::new(1.0, 0.0)), Point::new(22.0, 0.0));
    }

    #[test]
    fn translate_and_inflate_saturate_on_overflow() {
        let r = Rect::new(i32::MAX - 10, i32::MIN + 10, i32::MAX, i32::MIN + 20);
        let t = r.translate(100, -100);
        assert_eq!(t.x0, i32::MAX);
        assert_eq!(t.x1, i32::MAX);
        assert_eq!(t.y0, i32::MIN);
        assert_eq!(t.y1, i32::MIN);
        let grown = Rect::new(i32::MAX - 5, i32::MIN + 5, i32::MAX - 1, i32::MIN + 10).inflate(10);
        assert_eq!(grown.x1, i32::MAX);
        assert_eq!(grown.y0, i32::MIN);
    }

    #[test]
    fn width_and_height_do_not_overflow_on_extreme_edges() {
        // Inverted extreme rect: empty, so 0 (x1 - x0 used to overflow and panic).
        let r = Rect::new(i32::MAX, i32::MAX, i32::MIN, i32::MIN);
        assert_eq!((r.width(), r.height()), (0, 0));
        assert!(r.is_empty());
        // Widest possible rect: the true extent fits in u32.
        let r = Rect::new(i32::MIN, i32::MIN + 20, i32::MAX, i32::MAX - 5);
        assert_eq!(r.width(), u32::MAX);
        assert_eq!(r.height(), u32::MAX - 25);
        assert_eq!(r.size(), Size::new(u32::MAX, u32::MAX - 25));
    }
}
