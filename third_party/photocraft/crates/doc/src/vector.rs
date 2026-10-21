//! Vector data: paths (Pen tool), shape layers, saved/work paths and vector masks. Pure data;
//! rasterization lives in `photocraft-vector`, PSD mapping in `photocraft-io`.
//!
//! Coordinates are **document pixels** (f64, y down), cubic Bézier. A subpath is a list of knots;
//! segment `i` runs from `knots[i].anchor` via `knots[i].out_ctrl` and `knots[i+1].in_ctrl` to
//! `knots[i+1].anchor` (closed subpaths add the segment back to the first knot). A knot whose
//! control points equal its anchor is a corner with straight segments.
//!
//! Path operations follow Photoshop: every subpath is filled on its own (with the path's
//! [`FillRule`]) and folded, in order, into the result with its [`PathOp`]; the first subpath
//! always acts as "combine" (see [`Path::effective_op`]; verified against Photoshop's pixels in
//! the PSD corpus). Subpaths marked [`PathOp::Join`] belong to the shape component before them
//! and are filled together with it (a custom shape's holes). [`Path::inverted`] inverts the final area. A vector mask without subpaths
//! reveals everything (Photoshop's fresh "Add Vector Mask").

use std::sync::Arc;

use photocraft_color::Color;
use photocraft_geom::{Affine, Point};
use serde::{Deserialize, Serialize};

use crate::{Fill, Surface};

/// A Bézier knot (anchor + two control points).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Knot {
    pub anchor: Point,
    /// Control point of the segment arriving at this knot.
    pub in_ctrl: Point,
    /// Control point of the segment leaving this knot.
    pub out_ctrl: Point,
    /// Smooth (linked) knot: the Pen tool keeps both handles collinear.
    #[serde(default)]
    pub smooth: bool,
}

impl Knot {
    /// A corner knot with retracted handles.
    pub fn corner(x: f64, y: f64) -> Self {
        let p = Point::new(x, y);
        Knot { anchor: p, in_ctrl: p, out_ctrl: p, smooth: false }
    }
    /// A smooth knot with explicit handles.
    pub fn smooth(anchor: Point, in_ctrl: Point, out_ctrl: Point) -> Self {
        Knot { anchor, in_ctrl, out_ctrl, smooth: true }
    }
    pub fn transform(&self, a: &Affine) -> Knot {
        Knot { anchor: a.apply(self.anchor), in_ctrl: a.apply(self.in_ctrl), out_ctrl: a.apply(self.out_ctrl), smooth: self.smooth }
    }
}

/// How a subpath combines with the subpaths before it (Photoshop's path operations).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathOp {
    /// Combine shapes (union).
    #[default]
    Combine,
    /// Subtract front shape.
    Subtract,
    /// Intersect shape areas.
    Intersect,
    /// Exclude overlapping shapes (xor).
    Exclude,
    /// Part of the previous subpath's shape component (PSD operation -1 after a component's
    /// first record): the component's subpaths are filled together, with the path's fill rule,
    /// so inner subpaths wound the other way cut holes (custom shapes).
    Join,
}

/// Winding rule used to fill each subpath.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Subpath {
    pub closed: bool,
    pub knots: Vec<Knot>,
    #[serde(default)]
    pub op: PathOp,
}

impl Subpath {
    /// Closed polygon through `pts` (corner knots).
    pub fn polygon(pts: &[(f64, f64)]) -> Self {
        Subpath { closed: true, knots: pts.iter().map(|&(x, y)| Knot::corner(x, y)).collect(), op: PathOp::Combine }
    }
    /// Open polyline through `pts`.
    pub fn polyline(pts: &[(f64, f64)]) -> Self {
        Subpath { closed: false, ..Self::polygon(pts) }
    }
    pub fn with_op(mut self, op: PathOp) -> Self {
        self.op = op;
        self
    }
    /// Cubic segments `[p0, c1, c2, p1]`.
    pub fn segments(&self) -> Vec<[Point; 4]> {
        let n = self.knots.len();
        if n < 2 {
            return Vec::new();
        }
        let count = if self.closed { n } else { n - 1 };
        (0..count)
            .map(|i| {
                let a = &self.knots[i];
                let b = &self.knots[(i + 1) % n];
                [a.anchor, a.out_ctrl, b.in_ctrl, b.anchor]
            })
            .collect()
    }
}

/// A vector path: subpaths plus how they are filled and combined.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Path {
    pub subpaths: Vec<Subpath>,
    #[serde(default)]
    pub fill_rule: FillRule,
    /// The final area is inverted (PSD vector-mask invert flag / "initial fill rule" = 1).
    #[serde(default)]
    pub inverted: bool,
}

impl Path {
    pub fn new(subpaths: Vec<Subpath>) -> Self {
        Path { subpaths, fill_rule: FillRule::NonZero, inverted: false }
    }
    pub fn is_empty(&self) -> bool {
        self.subpaths.iter().all(|s| s.knots.is_empty())
    }
    /// The operation actually applied for subpath `i`: Photoshop applies the first subpath as
    /// "combine" whatever its stored operation (a lone "intersect" shape still draws).
    pub fn effective_op(&self, i: usize) -> PathOp {
        if i == 0 { PathOp::Combine } else { self.subpaths[i].op }
    }
    /// Shape components: ranges of subpaths filled together (a subpath starts a new component
    /// unless it is a [`PathOp::Join`]), each folded in with its first subpath's
    /// [`Path::effective_op`].
    pub fn components(&self) -> Vec<std::ops::Range<usize>> {
        let mut out: Vec<std::ops::Range<usize>> = Vec::new();
        for (i, s) in self.subpaths.iter().enumerate() {
            match out.last_mut() {
                Some(r) if s.op == PathOp::Join => r.end = i + 1,
                _ => out.push(i..i + 1),
            }
        }
        out
    }
    pub fn transform(&self, a: &Affine) -> Path {
        let mut p = self.clone();
        for s in &mut p.subpaths {
            for k in &mut s.knots {
                *k = k.transform(a);
            }
        }
        p
    }
    /// Bounds of all anchors and control points `(x0, y0, x1, y1)` (a superset of the curve).
    pub fn control_bounds(&self) -> Option<(f64, f64, f64, f64)> {
        let mut it = self.subpaths.iter().flat_map(|s| s.knots.iter()).flat_map(|k| [k.anchor, k.in_ctrl, k.out_ctrl]);
        let first = it.next()?;
        Some(it.fold((first.x, first.y, first.x, first.y), |b, p| (b.0.min(p.x), b.1.min(p.y), b.2.max(p.x), b.3.max(p.y))))
    }
}

/// A document-level path (Paths panel entry).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NamedPath {
    pub name: String,
    pub path: Path,
    /// Original PSD path resource data; reused on export while it still decodes to `path`.
    pub psd_raw: Option<Arc<Vec<u8>>>,
}

/// The document's clipping path (PSD resource 2999): which saved path clips on export.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ClippingPath {
    pub name: String,
    /// Output flatness in device pixels (0 = printer default).
    pub flatness: f32,
}

/// A layer's vector mask (Layer › Vector Mask).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorMask {
    pub path: Path,
    pub enabled: bool,
    /// Moves with the layer.
    pub linked: bool,
    /// Density `0..=1` (Properties panel), like [`crate::LayerMask::density`].
    pub density: f32,
    /// Feather radius in px (stored; rendered unfeathered for now).
    pub feather: f32,
}

impl VectorMask {
    pub fn new(path: Path) -> Self {
        VectorMask { path, enabled: true, linked: true, density: 1.0, feather: 0.0 }
    }
}

/// Where a stroke sits relative to the path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StrokeAlign {
    Inside,
    #[default]
    Center,
    Outside,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineCap {
    #[default]
    Butt,
    Round,
    Square,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineJoin {
    #[default]
    Miter,
    Round,
    Bevel,
}

/// Shape stroke (Photoshop's shape "Stroke" options, not the layer-style stroke).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ShapeStroke {
    /// Width in px.
    pub width: f32,
    pub paint: Fill,
    pub opacity: f32,
    pub align: StrokeAlign,
    pub cap: LineCap,
    pub join: LineJoin,
    /// Miter limit as a multiple of the width (Photoshop default 100 → effectively always
    /// mitered; SVG-style 4 is common elsewhere).
    pub miter_limit: f32,
    /// Dash pattern in multiples of the stroke width (Photoshop's convention); empty = solid.
    pub dashes: Vec<f32>,
    /// Dash offset in multiples of the width.
    pub dash_offset: f32,
}

impl Default for ShapeStroke {
    fn default() -> Self {
        ShapeStroke {
            width: 3.0,
            paint: Fill::Solid(Color::BLACK),
            opacity: 1.0,
            align: StrokeAlign::Center,
            cap: LineCap::Butt,
            join: LineJoin::Miter,
            miter_limit: 100.0,
            dashes: Vec::new(),
            dash_offset: 0.0,
        }
    }
}

impl ShapeStroke {
    /// Limits accepted by interactive commands. Importers retain original descriptor data;
    /// renderers must additionally bound their work for arbitrary file-provided geometry.
    pub fn validate(&self) -> Result<(), &'static str> {
        if !self.width.is_finite() || !(0.0..=1_000_000.0).contains(&self.width) {
            return Err("stroke width must be finite and between 0 and 1000000 px");
        }
        if !self.opacity.is_finite() || !(0.0..=1.0).contains(&self.opacity) {
            return Err("stroke opacity must be between 0 and 100%");
        }
        if !self.miter_limit.is_finite() || !(1.0..=500.0).contains(&self.miter_limit) {
            return Err("miter limit must be between 1 and 500");
        }
        if !self.dash_offset.is_finite() || self.dash_offset.abs() > 1_000_000.0 {
            return Err("dash offset must be finite and within ±1000000 widths");
        }
        if self.dashes.len() > 32 || self.dashes.iter().any(|d| !d.is_finite() || !(0.0..=10000.0).contains(d)) {
            return Err("dash pattern must contain at most 32 finite lengths between 0 and 10000 widths");
        }
        if !self.dashes.is_empty() && self.dashes.iter().sum::<f32>() < 0.01 {
            return Err("dash pattern total must be at least 0.01 widths");
        }
        Ok(())
    }
}

/// Parametric ("live") shape the path was generated from, kept for re-editing (Properties panel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum LiveShape {
    /// Rectangle `[x, y, w, h]` with corner radii `[tl, tr, br, bl]` (all zero = sharp).
    Rect {
        rect: [f64; 4],
        radii: [f64; 4],
    },
    Ellipse {
        rect: [f64; 4],
    },
    /// Regular polygon or star inscribed in `rect`; `star_ratio` = inner/outer radius (1 = polygon).
    Polygon {
        rect: [f64; 4],
        sides: u32,
        star_ratio: f64,
    },
    /// Line from `from` to `to` drawn as a filled bar `weight` px thick.
    Line {
        from: [f64; 2],
        to: [f64; 2],
        weight: f64,
    },
}

/// A shape layer: a filled and/or stroked path.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShapeLayer {
    pub path: Path,
    /// `None` = no fill (Photoshop's fill "none").
    pub fill: Option<Fill>,
    pub stroke: Option<ShapeStroke>,
    pub live: Option<LiveShape>,
    /// Rasterized appearance (from PSD or `photocraft-vector`).
    pub cache: Option<Surface>,
    /// Data of the PSD vector mask block (`vsms`, else `vmsk`). On export it replaces that entry
    /// in [`crate::Layer::psd_blocks`] while it still decodes to `path`.
    pub psd_raw: Option<Arc<Vec<u8>>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segments_open_and_closed() {
        let s = Subpath::polygon(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]);
        assert_eq!(s.segments().len(), 3);
        let o = Subpath::polyline(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]);
        assert_eq!(o.segments().len(), 2);
        assert_eq!(o.segments()[1][3], Point::new(10.0, 10.0));
    }

    #[test]
    fn first_op_is_combine_and_bounds() {
        let p = Path::new(vec![Subpath::polygon(&[(1.0, 2.0), (5.0, 2.0), (5.0, 9.0)]).with_op(PathOp::Intersect)]);
        assert_eq!(p.effective_op(0), PathOp::Combine);
        assert_eq!(p.control_bounds(), Some((1.0, 2.0, 5.0, 9.0)));
        let t = p.transform(&Affine::translate(1.0, 1.0));
        assert_eq!(t.control_bounds(), Some((2.0, 3.0, 6.0, 10.0)));
    }

    #[test]
    fn serde_roundtrip() {
        let p = Path::new(vec![Subpath::polygon(&[(0.0, 0.0), (3.0, 4.0)]).with_op(PathOp::Exclude)]);
        let s = ShapeStroke { dashes: vec![2.0, 1.0], ..Default::default() };
        let j = serde_json::to_string(&(&p, &s, &LiveShape::Rect { rect: [0.0; 4], radii: [1.0; 4] })).unwrap();
        let back: (Path, ShapeStroke, LiveShape) = serde_json::from_str(&j).unwrap();
        assert_eq!(back.0, p);
        assert_eq!(back.1, s);
        let v: Path = serde_json::from_str(r#"{"subpaths":[{"closed":true,"knots":[]}]}"#).unwrap();
        assert_eq!(v.subpaths[0].op, PathOp::Combine);
    }
}
