//! Snapping (View › Snap / Snap To) and smart guides, as pure geometry.
//!
//! A [`SnapTargets`] set is built once per gesture from the document (guides, grid, document
//! bounds and centre, layer edges and centres, selection edges), then every pointer move asks it
//! to snap a point ([`SnapTargets::snap_point`]: marquee corners, crop handles, guide positions,
//! pen anchors, transform handles) or a moving rectangle ([`SnapTargets::snap_rect`]: the Move
//! tool, Free Transform drags). Distances are in document pixels; callers convert their screen
//! threshold with the zoom (`8 px / zoom`). Results carry [`SnapLine`]s so the UI can draw what
//! was snapped to (smart guides are the layer-to-layer subset of these, drawn in magenta).

use photocraft_doc::{Document, LayerContent, LayerId};
use serde::Serialize;

/// What a snap target came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum SnapKind {
    Guide,
    Grid,
    DocumentEdge,
    DocumentCenter,
    LayerEdge,
    LayerCenter,
    Selection,
}

impl SnapKind {
    /// Layer-to-layer and document-centre alignments are what Photoshop shows as smart guides.
    pub fn is_smart(self) -> bool {
        matches!(self, SnapKind::LayerEdge | SnapKind::LayerCenter | SnapKind::DocumentCenter)
    }
}

/// A line to snap to: `pos` on one axis (x for vertical lines), `span` its extent on the other
/// axis (for drawing alignment lines between objects).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub pos: f64,
    pub kind: SnapKind,
    pub span: (f64, f64),
}

/// A snapped alignment, for drawing: a vertical line at x = `pos` from y = `from` to `to`
/// (or horizontal at y = `pos`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct SnapLine {
    pub vertical: bool,
    pub pos: f64,
    pub from: f64,
    pub to: f64,
    pub kind: SnapKind,
}

/// Which target families are active (View › Snap To).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SnapOptions {
    pub guides: bool,
    pub grid: bool,
    pub layers: bool,
    pub document: bool,
    pub selection: bool,
    /// Grid subdivision spacing in document pixels, independently for x and y.
    pub grid_step: [f64; 2],
}

impl Default for SnapOptions {
    fn default() -> Self {
        Self { guides: true, grid: false, layers: true, document: true, selection: true, grid_step: [18.0; 2] }
    }
}

/// Snap targets on both axes.
#[derive(Clone, Debug, Default)]
pub struct SnapTargets {
    /// Vertical lines (x positions).
    pub x: Vec<Target>,
    /// Horizontal lines (y positions).
    pub y: Vec<Target>,
    /// Grid subdivision spacing on the x and y axes (document px).
    pub grid: Option<[f64; 2]>,
    /// Extent used for grid lines and guides ([x0, y0, x1, y1]).
    pub extent: [f64; 4],
}

/// Bounds of a layer for snapping: what Free Transform would show (`None` for empty layers,
/// groups and maskless adjustment layers).
pub fn layer_rect(doc: &Document, id: LayerId) -> Option<[f64; 4]> {
    rect_of(doc, doc.layer(id)?)
}

fn rect_of(doc: &Document, l: &photocraft_doc::Layer) -> Option<[f64; 4]> {
    if l.is_group() || (matches!(l.content, LayerContent::Adjustment(_)) && l.mask.is_none()) {
        return None;
    }
    let b = crate::transform_cmds::transform_bounds(doc, l);
    (!b.is_empty()).then_some([b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64])
}

impl SnapTargets {
    /// Targets of `doc` for the enabled families; layers in `exclude` (the ones being moved)
    /// are skipped.
    pub fn from_document(doc: &Document, opts: &SnapOptions, exclude: &[LayerId]) -> Self {
        let (w, h) = (doc.size.width as f64, doc.size.height as f64);
        let mut t = SnapTargets { extent: [0.0, 0.0, w, h], ..Default::default() };
        if opts.document {
            t.add_rect([0.0, 0.0, w, h], SnapKind::DocumentEdge, SnapKind::DocumentCenter);
        }
        if opts.guides {
            for g in &doc.guides.vertical {
                t.x.push(Target { pos: *g, kind: SnapKind::Guide, span: (0.0, h) });
            }
            for g in &doc.guides.horizontal {
                t.y.push(Target { pos: *g, kind: SnapKind::Guide, span: (0.0, w) });
            }
        }
        if opts.grid && opts.grid_step.iter().all(|step| step.is_finite() && *step > 0.0) {
            t.grid = Some(opts.grid_step);
        }
        if opts.layers {
            for (_, _, l) in doc.walk() {
                if !l.visible || exclude.contains(&l.id) {
                    continue;
                }
                if let Some(r) = rect_of(doc, l) {
                    t.add_rect(r, SnapKind::LayerEdge, SnapKind::LayerCenter);
                }
            }
        }
        if opts.selection
            && let Some(sel) = &doc.selection
        {
            let b = sel.content_bounds();
            if !b.is_empty() {
                let r = [b.x0 as f64, b.y0 as f64, b.x1 as f64, b.y1 as f64];
                t.x.push(Target { pos: r[0], kind: SnapKind::Selection, span: (r[1], r[3]) });
                t.x.push(Target { pos: r[2], kind: SnapKind::Selection, span: (r[1], r[3]) });
                t.y.push(Target { pos: r[1], kind: SnapKind::Selection, span: (r[0], r[2]) });
                t.y.push(Target { pos: r[3], kind: SnapKind::Selection, span: (r[0], r[2]) });
            }
        }
        t
    }

    /// Add a rectangle's edges and centre lines.
    pub fn add_rect(&mut self, r: [f64; 4], edge: SnapKind, center: SnapKind) {
        let (xs, ys) = ((r[1], r[3]), (r[0], r[2]));
        self.x.push(Target { pos: r[0], kind: edge, span: xs });
        self.x.push(Target { pos: r[2], kind: edge, span: xs });
        self.x.push(Target { pos: (r[0] + r[2]) / 2.0, kind: center, span: xs });
        self.y.push(Target { pos: r[1], kind: edge, span: ys });
        self.y.push(Target { pos: r[3], kind: edge, span: ys });
        self.y.push(Target { pos: (r[1] + r[3]) / 2.0, kind: center, span: ys });
    }

    /// Only the targets for which `keep` holds (e.g. smart guides: layer alignments).
    pub fn filtered(&self, keep: impl Fn(SnapKind) -> bool) -> SnapTargets {
        SnapTargets {
            x: self.x.iter().copied().filter(|t| keep(t.kind)).collect(),
            y: self.y.iter().copied().filter(|t| keep(t.kind)).collect(),
            grid: self.grid.filter(|_| keep(SnapKind::Grid)),
            extent: self.extent,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty() && self.y.is_empty() && self.grid.is_none()
    }

    /// The closest target to `v` within `tol` on one axis (`vertical` = x positions). Guides win
    /// ties over other kinds, as in Photoshop.
    pub fn nearest(&self, vertical: bool, v: f64, tol: f64) -> Option<Target> {
        let list = if vertical { &self.x } else { &self.y };
        let mut best: Option<(f64, Target)> = None;
        let mut consider = |t: Target| {
            let d = (t.pos - v).abs();
            if d > tol {
                return;
            }
            let better = match best {
                None => true,
                Some((bd, bt)) => d < bd - 1e-9 || ((d - bd).abs() <= 1e-9 && t.kind == SnapKind::Guide && bt.kind != SnapKind::Guide),
            };
            if better {
                best = Some((d, t));
            }
        };
        list.iter().copied().for_each(&mut consider);
        if let Some(steps) = self.grid {
            let step = steps[usize::from(!vertical)];
            let g = (v / step).round() * step;
            let span = if vertical { (self.extent[1], self.extent[3]) } else { (self.extent[0], self.extent[2]) };
            consider(Target { pos: g, kind: SnapKind::Grid, span });
        }
        best.map(|(_, t)| t)
    }

    /// Snap a point: each axis independently to its nearest target within `tol`.
    pub fn snap_point(&self, p: [f64; 2], tol: f64) -> ([f64; 2], Vec<SnapLine>) {
        let mut out = p;
        let mut lines = Vec::new();
        if let Some(t) = self.nearest(true, p[0], tol) {
            out[0] = t.pos;
            lines.push(line(true, t, p[1], p[1]));
        }
        if let Some(t) = self.nearest(false, p[1], tol) {
            out[1] = t.pos;
            lines.push(line(false, t, out[0], out[0]));
        }
        (out, lines)
    }

    /// Snap a moving rectangle `[x0, y0, x1, y1]`: on each axis the edge or centre closest to a
    /// target decides the correction. Returns the offset to add and the alignments made (every
    /// anchor that lines up after the correction, so equal-size neighbours show both edges).
    pub fn snap_rect(&self, r: [f64; 4], tol: f64) -> ([f64; 2], Vec<SnapLine>) {
        let mut delta = [0.0, 0.0];
        let mut lines = Vec::new();
        for (axis, vertical) in [(0usize, true), (1usize, false)] {
            let (a, b) = (r[axis], r[axis + 2]);
            let anchors = [a, (a + b) / 2.0, b];
            let mut best: Option<(f64, f64)> = None; // (|d|, d)
            for v in anchors {
                if let Some(t) = self.nearest(vertical, v, tol) {
                    let d = t.pos - v;
                    if best.is_none_or(|(bd, _)| d.abs() < bd) {
                        best = Some((d.abs(), d));
                    }
                }
            }
            let Some((_, d)) = best else { continue };
            delta[axis] = d;
            let other = if vertical { (r[1], r[3]) } else { (r[0], r[2]) };
            for v in anchors {
                let moved = v + d;
                let list = if vertical { &self.x } else { &self.y };
                let hit = list.iter().copied().filter(|t| (t.pos - moved).abs() < 1e-6).min_by_key(|t| u8::from(t.kind != SnapKind::Guide));
                let hit = hit.or_else(|| {
                    self.grid
                        .filter(|steps| {
                            let step = steps[axis];
                            ((moved / step).round() * step - moved).abs() < 1e-6
                        })
                        .map(|_| Target { pos: moved, kind: SnapKind::Grid, span: other })
                });
                if let Some(t) = hit {
                    lines.push(line(vertical, t, other.0, other.1));
                }
            }
        }
        (delta, lines)
    }
}

/// Alignment line through target `t` covering both its span and the snapped object (`a..b`).
fn line(vertical: bool, t: Target, a: f64, b: f64) -> SnapLine {
    SnapLine { vertical, pos: t.pos, from: t.span.0.min(a).min(b), to: t.span.1.max(a).max(b), kind: t.kind }
}

/// Union of rectangles (`None` when empty).
pub fn union(rects: impl IntoIterator<Item = [f64; 4]>) -> Option<[f64; 4]> {
    rects.into_iter().reduce(|a, b| [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])])
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::{Color, ColorMode, Layer, SampleType, Size};

    fn targets() -> SnapTargets {
        let mut t = SnapTargets { extent: [0.0, 0.0, 400.0, 300.0], ..Default::default() };
        t.add_rect([0.0, 0.0, 400.0, 300.0], SnapKind::DocumentEdge, SnapKind::DocumentCenter);
        t.x.push(Target { pos: 120.0, kind: SnapKind::Guide, span: (0.0, 300.0) });
        t
    }

    #[test]
    fn point_snaps_within_threshold_only() {
        let t = targets();
        let (p, lines) = t.snap_point([123.0, 151.0], 4.0);
        assert_eq!(p, [120.0, 150.0]);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].kind, SnapKind::Guide);
        assert_eq!(lines[1].kind, SnapKind::DocumentCenter);
        let (p, lines) = t.snap_point([130.0, 160.0], 4.0);
        assert_eq!(p, [130.0, 160.0]);
        assert!(lines.is_empty());
    }

    #[test]
    fn threshold_scales_with_zoom() {
        // 8 screen px: at 400% that is 2 document px, at 25% it is 32.
        let t = targets();
        assert!(t.nearest(true, 125.0, 8.0 / 4.0).is_none());
        assert_eq!(t.nearest(true, 145.0, 8.0 / 0.25).map(|t| t.pos), Some(120.0));
    }

    #[test]
    fn guides_win_ties() {
        let mut t = targets();
        t.x.push(Target { pos: 120.0, kind: SnapKind::LayerEdge, span: (0.0, 10.0) });
        assert_eq!(t.nearest(true, 121.0, 5.0).unwrap().kind, SnapKind::Guide);
    }

    #[test]
    fn grid_snaps_to_nearest_line() {
        let t = SnapTargets { grid: Some([25.0; 2]), extent: [0.0, 0.0, 100.0, 100.0], ..Default::default() };
        assert_eq!(t.snap_point([48.0, 61.0], 3.0).0, [50.0, 61.0]);
        assert_eq!(t.nearest(false, 74.0, 3.0).unwrap().kind, SnapKind::Grid);
    }

    #[test]
    fn rectangular_grid_snaps_independently_on_both_axes() {
        // A 1280×720 document, 20% major spacing and two subdivisions:
        // vertical lines every 128 px, horizontal lines every 72 px.
        let t = SnapTargets { grid: Some([128.0, 72.0]), extent: [0.0, 0.0, 1280.0, 720.0], ..Default::default() };
        assert_eq!(t.snap_point([127.0, 71.0], 2.0).0, [128.0, 72.0]);
        assert_eq!(t.nearest(true, 255.0, 2.0).map(|t| t.pos), Some(256.0));
        assert_eq!(t.nearest(false, 143.0, 2.0).map(|t| t.pos), Some(144.0));
        let (delta, lines) = t.snap_rect([126.0, 70.0, 160.0, 100.0], 3.0);
        assert_eq!(delta, [2.0, 2.0]);
        assert!(lines.iter().any(|l| l.vertical && l.pos == 128.0 && l.kind == SnapKind::Grid));
        assert!(lines.iter().any(|l| !l.vertical && l.pos == 72.0 && l.kind == SnapKind::Grid));
    }

    #[test]
    fn rect_snaps_by_closest_anchor() {
        let t = targets();
        // A 50-wide box whose right edge is 2 px left of the guide: moves right by 2.
        let (d, lines) = t.snap_rect([68.0, 10.0, 118.0, 40.0], 4.0);
        assert_eq!(d[0], 2.0);
        assert!(lines.iter().any(|l| l.vertical && l.pos == 120.0));
        // Centre near the document centre (200, 150): centres both ways.
        let (d, lines) = t.snap_rect([148.0, 102.0, 248.0, 202.0], 4.0);
        assert_eq!(d, [2.0, -2.0]);
        assert!(lines.iter().any(|l| l.kind == SnapKind::DocumentCenter && l.vertical));
        assert!(lines.iter().any(|l| l.kind == SnapKind::DocumentCenter && !l.vertical));
    }

    #[test]
    fn smart_guide_lines_span_both_objects() {
        let mut t = SnapTargets::default();
        t.add_rect([300.0, 200.0, 350.0, 260.0], SnapKind::LayerEdge, SnapKind::LayerCenter);
        // Moving box above-left whose left edge is near x = 300.
        let (d, lines) = t.snap_rect([302.0, 10.0, 330.0, 40.0], 5.0);
        assert_eq!(d[0], -2.0);
        let l = lines.iter().find(|l| l.vertical && l.pos == 300.0).unwrap();
        assert_eq!((l.from, l.to), (10.0, 260.0));
        assert!(l.kind.is_smart());
    }

    #[test]
    fn targets_from_document() {
        let mut doc = Document::with_background("s", Size::new(200, 100), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        doc.guides.vertical.push(40.0);
        let mut l = Layer::raster("box", doc.pixel_format());
        l.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::new(10, 20, 30, 60), &[1.0, 0.0, 0.0, 1.0]);
        let id = doc.insert_above(None, l);
        let opts = SnapOptions::default();
        let t = SnapTargets::from_document(&doc, &opts, &[]);
        assert!(t.x.iter().any(|t| t.kind == SnapKind::Guide && t.pos == 40.0));
        assert!(t.x.iter().any(|t| t.kind == SnapKind::LayerEdge && t.pos == 30.0));
        assert!(t.y.iter().any(|t| t.kind == SnapKind::LayerCenter && t.pos == 40.0));
        assert!(t.x.iter().any(|t| t.kind == SnapKind::DocumentCenter && t.pos == 100.0));
        // The moving layer is excluded; guides off removes guides.
        let t = SnapTargets::from_document(&doc, &SnapOptions { guides: false, ..opts }, &[id]);
        assert!(!t.x.iter().any(|t| t.kind == SnapKind::Guide || (t.kind == SnapKind::LayerEdge && t.pos == 30.0)));
        assert_eq!(layer_rect(&doc, id), Some([10.0, 20.0, 30.0, 60.0]));
        assert_eq!(union([[0.0, 0.0, 1.0, 1.0], [2.0, -1.0, 3.0, 0.5]]), Some([0.0, -1.0, 3.0, 1.0]));
    }
}
