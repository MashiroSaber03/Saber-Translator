//! Filter › Liquify: a dense displacement field edited by brush strokes, then used to resample
//! the layer once.
//!
//! The field stores, for every node of a regular grid (one node per `cell` document pixels), the
//! offset `d` such that output pixel `p` shows the source at `p + d(p)` (an inverse map, so
//! applying it never leaves holes). Strokes are plain data ([`LiquifyStroke`]: tool, brush and
//! points), applied in order by [`LiquifyField::apply_stroke`]; replaying the same strokes on the
//! same bounds gives the same field bit for bit, which is what makes the filter a replayable
//! command and a smart filter.
//!
//! Every tool is written as "where does the new output at `p` come from in the current output":
//! `q = M(p)` (Forward Warp: `p - kΔ`, Twirl: a rotation about the brush centre, Pucker/Bloat: a
//! scaling about it), so the new field is `d'(p) = q + d(q) - p`, sampled bilinearly from the
//! field as it was before the dab. Reconstruct scales `d` toward zero and Smooth relaxes it toward
//! its local mean. Freeze/Thaw paint a mask that scales every tool's strength down.
//!
//! [`apply_liquify`] upsamples the field with Catmull-Rom (bicubic) per pixel and samples the
//! source (premultiplied, bicubic), tile by tile in parallel; tiles where the field is zero are
//! copied untouched. [`ProxyImage::render`] is the cheap preview path: bilinear field and
//! bilinear source on a downsampled RGBA copy, re-rendered only where a dab changed things.

use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde::{Deserialize, Serialize};

/// The Liquify tools (Photoshop's left tool strip, minus Hand/Zoom/Face).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LiquifyTool {
    #[default]
    ForwardWarp,
    Reconstruct,
    Smooth,
    TwirlCw,
    TwirlCcw,
    Pucker,
    Bloat,
    PushLeft,
    Freeze,
    Thaw,
    LassoMask,
    /// The Reconstruct… button: scales the whole (unfrozen) field toward zero by `amount` %.
    ReconstructAll,
    /// Mask Options › Mask All / None / Invert All (whole-field freeze edits).
    FreezeAll,
    ThawAll,
    InvertFreeze,
}

impl LiquifyTool {
    pub const ALL: [LiquifyTool; 11] = [
        LiquifyTool::ForwardWarp,
        LiquifyTool::Reconstruct,
        LiquifyTool::Smooth,
        LiquifyTool::TwirlCw,
        LiquifyTool::TwirlCcw,
        LiquifyTool::Pucker,
        LiquifyTool::Bloat,
        LiquifyTool::PushLeft,
        LiquifyTool::Freeze,
        LiquifyTool::Thaw,
        LiquifyTool::LassoMask,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LiquifyTool::ForwardWarp => "Forward Warp",
            LiquifyTool::Reconstruct => "Reconstruct",
            LiquifyTool::Smooth => "Smooth",
            LiquifyTool::TwirlCw => "Twirl Clockwise",
            LiquifyTool::TwirlCcw => "Twirl Counterclockwise",
            LiquifyTool::Pucker => "Pucker",
            LiquifyTool::Bloat => "Bloat",
            LiquifyTool::PushLeft => "Push Left",
            LiquifyTool::Freeze => "Freeze Mask",
            LiquifyTool::Thaw => "Thaw Mask",
            LiquifyTool::LassoMask => "Freeze Lasso",
            LiquifyTool::ReconstructAll => "Reconstruct All",
            LiquifyTool::FreezeAll => "Mask All",
            LiquifyTool::ThawAll => "Mask None",
            LiquifyTool::InvertFreeze => "Invert Mask",
        }
    }

    /// Whole-field operations (no brush, no points).
    pub fn is_global(self) -> bool {
        matches!(self, LiquifyTool::ReconstructAll | LiquifyTool::FreezeAll | LiquifyTool::ThawAll | LiquifyTool::InvertFreeze)
    }

    /// Tools that act while the brush is held still (each recorded point is a dab).
    pub fn is_stationary(self) -> bool {
        !matches!(self, LiquifyTool::ForwardWarp | LiquifyTool::PushLeft | LiquifyTool::LassoMask)
    }
}

fn d_size() -> f64 {
    100.0
}
fn d_density() -> f64 {
    50.0
}
fn d_pressure() -> f64 {
    100.0
}
fn d_rate() -> f64 {
    80.0
}

/// One brush stroke: the tool, its brush options and the points it passed through
/// (`[x, y]` or `[x, y, pressure 0..1]`, document pixels).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LiquifyStroke {
    #[serde(default)]
    pub tool: LiquifyTool,
    /// Brush diameter (px).
    #[serde(default = "d_size")]
    pub size: f64,
    /// How far the full effect reaches toward the brush edge, 0 (only the centre; feathered)
    /// ..100 (most of the brush). It always fades to nothing at the edge ([`falloff`]).
    #[serde(default = "d_density")]
    pub density: f64,
    /// Overall strength, 0..100.
    #[serde(default = "d_pressure")]
    pub pressure: f64,
    /// Speed of the stationary tools, 0..100.
    #[serde(default = "d_rate")]
    pub rate: f64,
    #[serde(default)]
    pub points: Vec<Vec<f64>>,
    /// `reconstructAll` only: how much to restore, 0..100. `lassoMask`: 1.0/none freezes the
    /// polygon, 0.0 thaws it.
    #[serde(default)]
    pub amount: Option<f64>,
}

impl LiquifyStroke {
    pub fn new(tool: LiquifyTool, size: f64) -> Self {
        LiquifyStroke { tool, size, density: d_density(), pressure: d_pressure(), rate: d_rate(), points: Vec::new(), amount: None }
    }

    /// Validates the stroke (finite numbers, 2 or 3 per point).
    pub fn validate(&self) -> Result<(), String> {
        if !(self.size.is_finite() && self.size >= 1.0 && self.size <= 15000.0) {
            return Err(format!("brush size {} out of range 1..15000", self.size));
        }
        for p in &self.points {
            if !(2..=3).contains(&p.len()) || p.iter().any(|v| !v.is_finite()) {
                return Err("stroke points are [x, y] or [x, y, pressure]".into());
            }
        }
        Ok(())
    }
}

/// The displacement field plus the freeze mask, over `bounds` (document pixels).
#[derive(Clone, Debug)]
pub struct LiquifyField {
    pub bounds: Rect,
    /// Document pixels per field node.
    pub cell: f64,
    pub w: usize,
    pub h: usize,
    /// Inverse displacement per node, row-major.
    pub d: Vec<[f32; 2]>,
    /// Freeze mask per node (0 = free, 1 = frozen).
    pub freeze: Vec<f32>,
    /// Where the brush stroke in progress stands, so dab spacing runs on across its segments.
    cursor: Option<StrokeCursor>,
}

/// Two fields are equal when their displacement and freeze mask are: the stroke cursor is
/// transient brush state, not part of the result.
impl PartialEq for LiquifyField {
    fn eq(&self, o: &Self) -> bool {
        self.bounds == o.bounds && self.cell == o.cell && self.w == o.w && self.h == o.h && self.d == o.d && self.freeze == o.freeze
    }
}

/// The brush stroke in progress: the point its last segment ended at, its last dab and how far
/// the brush has travelled along the path since that dab.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StrokeCursor {
    end: [f64; 2],
    last_dab: [f64; 2],
    carry: f64,
}

/// Distance between dabs along a stroke for brush radius `r`: a twentieth of the radius, so a
/// drag is many small pushes and its edge stays smooth.
fn dab_spacing(r: f64) -> f64 {
    (r * 0.05).max(0.5)
}

/// Per-dab strength of a stationary tool (Twirl, Pucker, Bloat, Reconstruct, Smooth, Freeze,
/// Thaw) while it moves: scaled to the spacing so its effect per distance travelled stays what it
/// was with dabs a fifth of the radius apart.
fn moving_weight(r: f64) -> f64 {
    dab_spacing(r) / (r * 0.2).max(0.5)
}

/// The default field resolution for `bounds`: 2 px per node up to 4 MP, 4 px beyond.
pub fn auto_cell(bounds: Rect) -> f64 {
    if bounds.size().area() <= 4_200_000 { 2.0 } else { 4.0 }
}

impl LiquifyField {
    pub fn new(bounds: Rect, cell: f64) -> Self {
        let cell = cell.clamp(1.0, 64.0);
        let w = (f64::from(bounds.width()) / cell).ceil() as usize + 2;
        let h = (f64::from(bounds.height()) / cell).ceil() as usize + 2;
        LiquifyField { bounds, cell, w, h, d: vec![[0.0; 2]; w * h], freeze: vec![0.0; w * h], cursor: None }
    }

    /// Document position of node (i, j) (node 0 sits on the first pixel centre).
    #[inline]
    pub fn node_pos(&self, i: usize, j: usize) -> [f64; 2] {
        [f64::from(self.bounds.x0) + 0.5 + i as f64 * self.cell, f64::from(self.bounds.y0) + 0.5 + j as f64 * self.cell]
    }

    /// Field coordinates of a document point.
    #[inline]
    fn to_grid(&self, x: f64, y: f64) -> (f64, f64) {
        ((x - f64::from(self.bounds.x0) - 0.5) / self.cell, (y - f64::from(self.bounds.y0) - 0.5) / self.cell)
    }

    pub fn is_identity(&self) -> bool {
        self.d.iter().all(|v| v[0] == 0.0 && v[1] == 0.0)
    }

    /// Largest displacement magnitude (px).
    pub fn max_displacement(&self) -> f32 {
        self.d.iter().map(|v| v[0].hypot(v[1])).fold(0.0, f32::max)
    }

    /// Bilinear displacement at a document point (zero outside the field).
    pub fn sample(&self, x: f64, y: f64) -> [f64; 2] {
        let (gx, gy) = self.to_grid(x, y);
        bilinear(&self.d, self.w, self.h, gx, gy)
    }

    /// Freeze value at a document point (bilinear).
    pub fn freeze_at(&self, x: f64, y: f64) -> f32 {
        let (gx, gy) = self.to_grid(x, y);
        let gx = gx.clamp(0.0, (self.w - 1) as f64);
        let gy = gy.clamp(0.0, (self.h - 1) as f64);
        let (i, j) = ((gx as usize).min(self.w - 2), (gy as usize).min(self.h - 2));
        let (fx, fy) = ((gx - i as f64) as f32, (gy - j as f64) as f32);
        let at = |i: usize, j: usize| self.freeze[j * self.w + i];
        (at(i, j) * (1.0 - fx) + at(i + 1, j) * fx) * (1.0 - fy) + (at(i, j + 1) * (1.0 - fx) + at(i + 1, j + 1) * fx) * fy
    }

    /// Applies one stroke. Returns the document rectangle whose output changed.
    pub fn apply_stroke(&mut self, s: &LiquifyStroke) -> Rect {
        self.cursor = None;
        match s.tool {
            LiquifyTool::LassoMask => return self.apply_lasso(s),
            LiquifyTool::ReconstructAll => {
                let k = (s.amount.unwrap_or(100.0) / 100.0).clamp(0.0, 1.0) as f32;
                for (v, f) in self.d.iter_mut().zip(&self.freeze) {
                    let m = 1.0 - k * (1.0 - f);
                    v[0] *= m;
                    v[1] *= m;
                }
                return self.bounds;
            }
            LiquifyTool::FreezeAll => self.freeze.fill(1.0),
            LiquifyTool::ThawAll => self.freeze.fill(0.0),
            LiquifyTool::InvertFreeze => self.freeze.iter_mut().for_each(|f| *f = 1.0 - *f),
            _ => {}
        }
        if s.tool.is_global() {
            return self.bounds;
        }
        let pts: Vec<[f64; 3]> = s.points.iter().filter(|p| p.len() >= 2).map(|p| [p[0], p[1], p.get(2).copied().unwrap_or(1.0)]).collect();
        let Some(&first) = pts.first() else { return Rect::EMPTY };
        let mut dirty = self.stroke_begin(s, first);
        for w in pts.windows(2) {
            dirty = dirty.union(&self.stroke_segment(s, w[0], w[1]));
        }
        dirty
    }

    /// Lasso: sets the freeze mask to `amount` (1.0 when none; 0.0 thaws) at every field node
    /// inside the closed polygon in `s.points` (even-odd rule; non-finite points are skipped).
    /// A scanline fill: each node row costs one pass over the edges, then only the nodes inside
    /// are written. Returns the document rectangle the mask changed in.
    fn apply_lasso(&mut self, s: &LiquifyStroke) -> Rect {
        let poly: Vec<[f64; 2]> = s.points.iter().filter_map(|p| Some([*p.first()?, *p.get(1)?])).filter(|p| p[0].is_finite() && p[1].is_finite()).collect();
        if poly.len() < 3 || self.w == 0 || self.h == 0 {
            return Rect::EMPTY;
        }
        let (mut px0, mut py0, mut px1, mut py1) = (f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for p in &poly {
            px0 = px0.min(p[0]);
            py0 = py0.min(p[1]);
            px1 = px1.max(p[0]);
            py1 = py1.max(p[1]);
        }
        let (_, gy0) = self.to_grid(px0, py0);
        let (_, gy1) = self.to_grid(px1, py1);
        let last_row = self.h.saturating_sub(1) as f64;
        let last_col = self.w.saturating_sub(1) as f64;
        let (j0, j1) = (gy0.ceil().max(0.0), gy1.floor().min(last_row));
        if j1 < j0 {
            return Rect::EMPTY;
        }
        let v = if s.amount.unwrap_or(1.0) > 0.5 { 1.0f32 } else { 0.0 };
        let mut xs: Vec<f64> = Vec::new();
        for j in (j0 as usize)..=(j1 as usize) {
            let y = self.node_pos(0, j)[1];
            xs.clear();
            let Some(mut prev) = poly.last().copied() else { return Rect::EMPTY };
            for &a in &poly {
                if (a[1] > y) != (prev[1] > y) {
                    xs.push(a[0] + (y - a[1]) * (prev[0] - a[0]) / (prev[1] - a[1]));
                }
                prev = a;
            }
            xs.sort_by(f64::total_cmp);
            // Even-odd: nodes with xs[2k] <= x < xs[2k + 1] are inside.
            for &[xa, xb] in xs.as_chunks::<2>().0 {
                let (ga, _) = self.to_grid(xa, y);
                let (gb, _) = self.to_grid(xb, y);
                let (i0, i1) = (ga.ceil().max(0.0), (gb.ceil() - 1.0).min(last_col));
                if i1 < i0 {
                    continue;
                }
                let row = j * self.w;
                if let Some(cells) = self.freeze.get_mut(row + i0 as usize..=row + i1 as usize) {
                    cells.fill(v);
                }
            }
        }
        Rect::new(px0.floor() as i32, py0.floor() as i32, px1.ceil() as i32, py1.ceil() as i32).intersect(&self.bounds)
    }

    /// Starts a stroke at `p` (`[x, y, pressure]`): stationary tools dab once there. Interactive
    /// callers use this plus [`Self::stroke_segment`] per new point, which is exactly what
    /// [`Self::apply_stroke`] does with the whole point list.
    pub fn stroke_begin(&mut self, s: &LiquifyStroke, p: [f64; 3]) -> Rect {
        self.cursor = Some(StrokeCursor { end: [p[0], p[1]], last_dab: [p[0], p[1]], carry: 0.0 });
        if s.tool.is_stationary() && !s.tool.is_global() { self.dab(s, [p[0], p[1]], [0.0, 0.0], p[2].clamp(0.0, 1.0), 1.0) } else { Rect::EMPTY }
    }

    /// Continues a stroke from `a` to `b`. Dabs fall every [`dab_spacing`] of distance travelled
    /// along the whole stroke, carried from one segment to the next, so the result is the same
    /// however finely the pointer path was sampled (a fast and a slow mouse give one warp). A
    /// stationary tool held still dabs once at `b`. When `a` is not where the previous segment
    /// ended, the stroke starts afresh at `a`.
    pub fn stroke_segment(&mut self, s: &LiquifyStroke, a: [f64; 3], b: [f64; 3]) -> Rect {
        self.walk_segment(s, a, b, true)
    }

    /// [`Self::stroke_segment`]; `skip` leaves out the dabs that cannot reach a field node (they
    /// would change nothing), which tests compare with the full walk.
    fn walk_segment(&mut self, s: &LiquifyStroke, a: [f64; 3], b: [f64; 3], skip: bool) -> Rect {
        let mut dirty = Rect::EMPTY;
        let (pa, pb) = (a[2].clamp(0.0, 1.0), b[2].clamp(0.0, 1.0));
        let stationary = s.tool.is_stationary();
        let cur = match self.cursor {
            Some(c) if c.end == [a[0], a[1]] => c,
            _ => StrokeCursor { end: [a[0], a[1]], last_dab: [a[0], a[1]], carry: 0.0 },
        };
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let len = dx.hypot(dy);
        if len == 0.0 {
            self.cursor = Some(cur);
            return if stationary { self.dab(s, [b[0], b[1]], [0.0, 0.0], pb, 1.0) } else { dirty };
        }
        // NaN or a length that overflowed: nothing sensible to walk; the next segment restarts.
        if !len.is_finite() {
            self.cursor = None;
            return dirty;
        }
        let r = (s.size / 2.0).max(0.5);
        let spacing = dab_spacing(r);
        let weight = if stationary { moving_weight(r) } else { 1.0 };
        // Distance along this segment of its first dab, then one every `spacing`.
        let first = (spacing - cur.carry).clamp(0.0, spacing);
        // Distances are compared with a little slack: a dab that rounding puts a hair past the
        // segment's end still falls on it, so cutting one segment in two never adds or drops a dab.
        let slack = spacing * 1e-6;
        if first > len + slack {
            self.cursor = Some(StrokeCursor { end: [b[0], b[1]], last_dab: cur.last_dab, carry: cur.carry + len });
            return dirty;
        }
        let last_k = ((len - first + slack) / spacing).floor().max(0.0);
        let at = |dist: f64| [a[0] + dx * dist / len, a[1] + dy * dist / len];
        let (mut k0, mut k1) = (0.0f64, last_k);
        if skip {
            match self.reachable_span(a, [dx, dy], len, r) {
                // A dragging dab is centred one spacing behind its position: pad by that and a step.
                Some((d0, d1)) => {
                    k0 = k0.max(((d0 - spacing - first) / spacing).floor() - 1.0);
                    k1 = k1.min(((d1 + spacing - first) / spacing).ceil() + 1.0);
                }
                None => k1 = -1.0,
            }
        }
        if k0 <= k1 {
            let mut prev = if k0 > 0.0 { at(first + (k0 - 1.0) * spacing) } else { cur.last_dab };
            for k in (k0 as u64)..=(k1 as u64) {
                let dist = (first + k as f64 * spacing).min(len);
                let c = at(dist);
                let delta = [c[0] - prev[0], c[1] - prev[1]];
                // Forward Warp centres the dab where the brush was (it drags what is under it).
                let centre = if stationary { c } else { prev };
                dirty = dirty.union(&self.dab(s, centre, delta, pa + (pb - pa) * dist / len, weight));
                prev = c;
            }
        }
        let last = (first + last_k * spacing).min(len);
        self.cursor = Some(StrokeCursor { end: [b[0], b[1]], last_dab: at(last), carry: (len - last).clamp(0.0, spacing) });
        dirty
    }

    /// The distances along a segment from `a` (direction `dir`, length `len`) whose points lie
    /// within a dab radius `r` plus two nodes of the node grid, or `None` when none do. Only dabs
    /// near that span can touch a node; a segment to a far-off point has trillions of others
    /// (#937).
    fn reachable_span(&self, a: [f64; 3], dir: [f64; 2], len: f64, r: f64) -> Option<(f64, f64)> {
        if self.w == 0 || self.h == 0 {
            return None;
        }
        let pad = r + 2.0 * self.cell;
        let lo = [f64::from(self.bounds.x0) + 0.5 - pad, f64::from(self.bounds.y0) + 0.5 - pad];
        let hi = [lo[0] + (self.w - 1) as f64 * self.cell + 2.0 * pad, lo[1] + (self.h - 1) as f64 * self.cell + 2.0 * pad];
        let (mut u0, mut u1) = (0.0f64, 1.0f64);
        for ax in 0..2 {
            let d = dir[ax];
            if d == 0.0 {
                if a[ax].is_nan() || a[ax] < lo[ax] || a[ax] > hi[ax] {
                    return None;
                }
                continue;
            }
            let (v0, v1) = ((lo[ax] - a[ax]) / d, (hi[ax] - a[ax]) / d);
            u0 = u0.max(v0.min(v1));
            u1 = u1.min(v0.max(v1));
        }
        if u0.is_nan() || u1.is_nan() || u0 > u1 {
            return None;
        }
        Some((u0 * len, u1 * len))
    }

    /// One dab at `c` with brush motion `delta`, its strength scaled by `weight`. Returns the
    /// document rect it affected.
    fn dab(&mut self, s: &LiquifyStroke, c: [f64; 2], delta: [f64; 2], point_pressure: f64, weight: f64) -> Rect {
        if s.tool == LiquifyTool::LassoMask {
            return Rect::EMPTY; // the lasso is polygon-only; never a brush dab
        }
        let r = (s.size / 2.0).max(0.5);
        let strength = (s.pressure / 100.0).clamp(0.0, 1.0) * point_pressure * weight;
        let rate = (s.rate / 100.0).clamp(0.0, 1.0);
        let density = (s.density / 100.0).clamp(0.0, 1.0);
        if strength <= 0.0 {
            return Rect::EMPTY;
        }
        let (gx0, gy0) = self.to_grid(c[0] - r, c[1] - r);
        let (gx1, gy1) = self.to_grid(c[0] + r, c[1] + r);
        let i0 = gx0.floor().max(0.0) as usize;
        let j0 = gy0.floor().max(0.0) as usize;
        let i1 = (gx1.ceil().max(-1.0) as isize).min(self.w as isize - 1);
        let j1 = (gy1.ceil().max(-1.0) as isize).min(self.h as isize - 1);
        if i1 < i0 as isize || j1 < j0 as isize {
            return Rect::EMPTY;
        }
        let (i1, j1) = (i1 as usize, j1 as usize);
        // Snapshot of the affected nodes plus a margin (sampling reads around them).
        let m = ((delta[0].abs().max(delta[1].abs()) + r * 0.2) / self.cell).ceil() as usize + 2;
        let (si0, sj0) = (i0.saturating_sub(m), j0.saturating_sub(m));
        let (si1, sj1) = ((i1 + m).min(self.w - 1), (j1 + m).min(self.h - 1));
        let sw = si1 - si0 + 1;
        let sh = sj1 - sj0 + 1;
        let mut snap = Vec::with_capacity(sw * sh);
        for j in sj0..=sj1 {
            snap.extend_from_slice(&self.d[j * self.w + si0..=j * self.w + si1]);
        }
        let old = |gx: f64, gy: f64| -> [f64; 2] {
            let (lx, ly) = (gx - si0 as f64, gy - sj0 as f64);
            if lx >= 0.0 && ly >= 0.0 && lx <= (sw - 1) as f64 && ly <= (sh - 1) as f64 {
                bilinear(&snap, sw, sh, lx, ly)
            } else {
                // Outside the snapshot nothing changed: read the live field.
                bilinear(&self.d, self.w, self.h, gx, gy)
            }
        };
        let mut writes: Vec<(usize, [f32; 2], f32)> = Vec::with_capacity((i1 - i0 + 1) * (j1 - j0 + 1));
        let inv_cell = 1.0 / self.cell;
        for j in j0..=j1 {
            for i in i0..=i1 {
                let p = self.node_pos(i, j);
                let (dx, dy) = (p[0] - c[0], p[1] - c[1]);
                let t2 = (dx * dx + dy * dy) / (r * r);
                if t2 >= 1.0 {
                    continue;
                }
                let fall = falloff(t2, density);
                let idx = j * self.w + i;
                let k_raw = strength * fall;
                if s.tool == LiquifyTool::Freeze || s.tool == LiquifyTool::Thaw {
                    let f = self.freeze[idx];
                    let v = if s.tool == LiquifyTool::Freeze { (f as f64 + k_raw).min(1.0) } else { (f as f64 - k_raw).max(0.0) };
                    writes.push((idx, self.d[idx], v as f32));
                    continue;
                }
                let k = k_raw * (1.0 - f64::from(self.freeze[idx]));
                if k <= 0.0 {
                    continue;
                }
                let cur = self.d[idx];
                let new = match s.tool {
                    LiquifyTool::Reconstruct => {
                        let a = (1.0 - k * (0.1 + 0.4 * rate)) as f32;
                        [cur[0] * a, cur[1] * a]
                    }
                    LiquifyTool::Smooth => {
                        let (gx, gy) = ((i - si0) as isize, (j - sj0) as isize);
                        let at = |x: isize, y: isize| snap[(y.clamp(0, sh as isize - 1) as usize) * sw + x.clamp(0, sw as isize - 1) as usize];
                        let mut mean = [0.0f32; 2];
                        for (ox, oy) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)] {
                            let v = at(gx + ox, gy + oy);
                            mean[0] += v[0] / 8.0;
                            mean[1] += v[1] / 8.0;
                        }
                        let a = (k * (0.1 + 0.4 * rate)) as f32;
                        [cur[0] + (mean[0] - cur[0]) * a, cur[1] + (mean[1] - cur[1]) * a]
                    }
                    _ => {
                        let q = match s.tool {
                            LiquifyTool::ForwardWarp => [p[0] - k * delta[0], p[1] - k * delta[1]],
                            // Left of the stroke direction (y down): (dx, dy) → (dy, -dx).
                            LiquifyTool::PushLeft => [p[0] - k * delta[1], p[1] + k * delta[0]],
                            LiquifyTool::TwirlCw | LiquifyTool::TwirlCcw => {
                                let th = k * rate * 0.12 * if s.tool == LiquifyTool::TwirlCw { 1.0 } else { -1.0 };
                                // Content turns by +th, so sample at the point turned by -th.
                                let (sn, cs) = (-th).sin_cos();
                                [c[0] + dx * cs - dy * sn, c[1] + dx * sn + dy * cs]
                            }
                            LiquifyTool::Pucker => {
                                let a = 1.0 + k * rate * 0.06;
                                [c[0] + dx * a, c[1] + dy * a]
                            }
                            _ => {
                                let a = 1.0 - k * rate * 0.06;
                                [c[0] + dx * a, c[1] + dy * a]
                            }
                        };
                        let (qx, qy) = ((q[0] - p[0]) * inv_cell + i as f64, (q[1] - p[1]) * inv_cell + j as f64);
                        let dq = old(qx, qy);
                        [(q[0] + dq[0] - p[0]) as f32, (q[1] + dq[1] - p[1]) as f32]
                    }
                };
                writes.push((idx, new, self.freeze[idx]));
            }
        }
        for (idx, d, f) in writes {
            self.d[idx] = d;
            self.freeze[idx] = f;
        }
        let pad = r.ceil() as i32 + 2;
        Rect::new(c[0].floor() as i32 - pad, c[1].floor() as i32 - pad, c[0].ceil() as i32 + pad, c[1].ceil() as i32 + pad).intersect(&self.bounds)
    }

    /// Replays strokes on a fresh field.
    pub fn from_strokes(bounds: Rect, cell: f64, strokes: &[LiquifyStroke]) -> Self {
        let mut f = LiquifyField::new(bounds, cell);
        for s in strokes {
            f.apply_stroke(s);
        }
        f
    }

    /// Catmull-Rom (bicubic) displacement at grid coordinates.
    fn sample_cubic(&self, gx: f64, gy: f64, wx: &[f64; 4], wy: &[f64; 4]) -> [f64; 2] {
        let (ix, iy) = (gx.floor() as isize, gy.floor() as isize);
        let mut acc = [0.0f64; 2];
        for (jj, wyj) in wy.iter().enumerate() {
            let y = (iy - 1 + jj as isize).clamp(0, self.h as isize - 1) as usize;
            let row = &self.d[y * self.w..(y + 1) * self.w];
            for (ii, wxi) in wx.iter().enumerate() {
                let x = (ix - 1 + ii as isize).clamp(0, self.w as isize - 1) as usize;
                let k = wxi * wyj;
                acc[0] += f64::from(row[x][0]) * k;
                acc[1] += f64::from(row[x][1]) * k;
            }
        }
        acc
    }

    /// Whether any node influencing document rect `r` is displaced.
    fn touches(&self, r: Rect) -> bool {
        let (gx0, gy0) = self.to_grid(f64::from(r.x0), f64::from(r.y0));
        let (gx1, gy1) = self.to_grid(f64::from(r.x1), f64::from(r.y1));
        let i0 = (gx0.floor() as isize - 2).clamp(0, self.w as isize - 1) as usize;
        let j0 = (gy0.floor() as isize - 2).clamp(0, self.h as isize - 1) as usize;
        let i1 = (gx1.ceil() as isize + 2).clamp(0, self.w as isize - 1) as usize;
        let j1 = (gy1.ceil() as isize + 2).clamp(0, self.h as isize - 1) as usize;
        (j0..=j1).any(|j| self.d[j * self.w + i0..=j * self.w + i1].iter().any(|v| v[0] != 0.0 || v[1] != 0.0))
    }
}

/// A dab's strength at squared distance `t2` (in brush radii, 0..1) for Density `density` (0..1):
/// 1 at the centre, and down to 0 at the rim with a flat slope at every density, as in
/// Photoshop, where Density sets how far the full effect reaches toward the edge (0: only the
/// centre, the rest feathered; 1: most of the brush) but the effect always fades out at it.
/// A brush that stopped short of 0 at its rim moved the pixels inside it and left those just
/// outside, tearing the image along every dab's circle (#2606).
pub fn falloff(t2: f64, density: f64) -> f64 {
    let t2 = t2.clamp(0.0, 1.0);
    let density = if density.is_nan() { 0.5 } else { density.clamp(0.0, 1.0) };
    // Feathered: (1 - t²)³. Dense: (1 - t⁴)², flat in the middle. Both are 0 with zero slope at
    // t = 1, and their slopes stay small enough that a dab every fifth of the radius never folds
    // the image over itself.
    let feathered = (1.0 - t2).powi(3);
    let dense = (1.0 - t2 * t2).powi(2);
    feathered + (dense - feathered) * density
}

fn bilinear(d: &[[f32; 2]], w: usize, h: usize, gx: f64, gy: f64) -> [f64; 2] {
    if !(gx > -1.0 && gy > -1.0 && gx < w as f64 && gy < h as f64) {
        return [0.0, 0.0];
    }
    let gx = gx.clamp(0.0, (w - 1) as f64);
    let gy = gy.clamp(0.0, (h - 1) as f64);
    let (i, j) = ((gx as usize).min(w.saturating_sub(2)), (gy as usize).min(h.saturating_sub(2)));
    let (fx, fy) = (gx - i as f64, gy - j as f64);
    let i1 = (i + 1).min(w - 1);
    let j1 = (j + 1).min(h - 1);
    let at = |x: usize, y: usize| d[y * w + x];
    let (a, b, c, e) = (at(i, j), at(i1, j), at(i, j1), at(i1, j1));
    let l = |k: usize| (f64::from(a[k]) * (1.0 - fx) + f64::from(b[k]) * fx) * (1.0 - fy) + (f64::from(c[k]) * (1.0 - fx) + f64::from(e[k]) * fx) * fy;
    [l(0), l(1)]
}

fn catmull_rom(t: f64) -> [f64; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1.0, -1.5 * t3 + 2.0 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]
}

/// Resamples `src` through the field: inside `field.bounds` each pixel shows the source at
/// `p + d(p)` (bicubic field, bicubic premultiplied source, edge pixels repeat past the field
/// bounds); everything outside the bounds is kept. Same format as `src`.
pub fn apply_liquify(src: &Surface, field: &LiquifyField) -> Surface {
    let fmt = src.format();
    let n = fmt.channels();
    let alpha = fmt.alpha;
    let b = field.bounds;
    let mut out = src.clone();
    if field.is_identity() || b.is_empty() {
        return out;
    }
    let maxd = f64::from(field.max_displacement()).ceil() as i32 + 3;
    let tiles: Vec<Rect> = b.tiles().map(|t| t.rect().intersect(&b)).filter(|r| !r.is_empty() && field.touches(*r)).collect();
    let work = |t: &Rect| -> Option<(Rect, Vec<f32>)> {
        let w = t.width() as usize;
        let h = t.height() as usize;
        // Field per pixel (bicubic), rows share their y weights.
        let mut disp = vec![[0.0f64; 2]; w * h];
        let (mut fx0, mut fy0, mut fx1, mut fy1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        let mut moved = false;
        for y in 0..h {
            let py = f64::from(t.y0) + y as f64 + 0.5;
            let gy = (py - f64::from(b.y0) - 0.5) / field.cell;
            let wy = catmull_rom(gy - gy.floor());
            for x in 0..w {
                let px = f64::from(t.x0) + x as f64 + 0.5;
                let gx = (px - f64::from(b.x0) - 0.5) / field.cell;
                let wx = catmull_rom(gx - gx.floor());
                let d = field.sample_cubic(gx, gy, &wx, &wy);
                if d[0] != 0.0 || d[1] != 0.0 {
                    moved = true;
                }
                disp[y * w + x] = d;
                let (sx, sy) = (px + d[0], py + d[1]);
                fx0 = fx0.min(sx);
                fy0 = fy0.min(sy);
                fx1 = fx1.max(sx);
                fy1 = fy1.max(sy);
            }
        }
        if !moved {
            return None;
        }
        let foot =
            Rect::new(fx0.floor() as i32 - 3, fy0.floor() as i32 - 3, fx1.ceil() as i32 + 3, fy1.ceil() as i32 + 3).intersect(&t.inflate(maxd)).intersect(&b);
        if foot.is_empty() {
            return None;
        }
        let mut px = src.read_region(foot);
        if alpha {
            for p in px.chunks_exact_mut(n) {
                let al = p[n - 1];
                for v in &mut p[..n - 1] {
                    *v *= al;
                }
            }
        }
        let fw = foot.width() as usize;
        let at = |x: i32, y: i32| -> usize {
            let x = x.clamp(foot.x0, foot.x1 - 1);
            let y = y.clamp(foot.y0, foot.y1 - 1);
            ((y - foot.y0) as usize * fw + (x - foot.x0) as usize) * n
        };
        let snap = |t: f64| if (t - t.round()).abs() < 1e-7 { t.round() } else { t };
        let mut outp = vec![0.0f32; w * h * n];
        let mut acc = [0.0f64; 8];
        for y in 0..h {
            for x in 0..w {
                let d = disp[y * w + x];
                let u = snap(f64::from(t.x0) + x as f64 + d[0]);
                let v = snap(f64::from(t.y0) + y as f64 + d[1]);
                let (ix, iy) = (u.floor() as i32, v.floor() as i32);
                let (wx, wy) = (catmull_rom(u - f64::from(ix)), catmull_rom(v - f64::from(iy)));
                acc[..n].fill(0.0);
                for (j, wyj) in wy.iter().enumerate() {
                    for (i, wxi) in wx.iter().enumerate() {
                        let k = wxi * wyj;
                        if k == 0.0 {
                            continue;
                        }
                        let o = at(ix - 1 + i as i32, iy - 1 + j as i32);
                        for (c, s) in acc.iter_mut().enumerate().take(n) {
                            *s += f64::from(px[o + c]) * k;
                        }
                    }
                }
                let o = (y * w + x) * n;
                if alpha {
                    let al = acc[n - 1].clamp(0.0, 1.0);
                    outp[o + n - 1] = al as f32;
                    for c in 0..n - 1 {
                        outp[o + c] = if al > 0.0 { (acc[c] / al).clamp(0.0, 1.0) as f32 } else { 0.0 };
                    }
                } else {
                    for c in 0..n {
                        outp[o + c] = acc[c].clamp(0.0, 1.0) as f32;
                    }
                }
            }
        }
        Some((*t, outp))
    };
    #[cfg(not(target_arch = "wasm32"))]
    let done: Vec<(Rect, Vec<f32>)> = {
        use rayon::prelude::*;
        tiles.par_iter().filter_map(work).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let done: Vec<(Rect, Vec<f32>)> = tiles.iter().filter_map(work).collect();
    for (r, v) in done {
        out.write_region(r, &v);
    }
    if alpha {
        out.prune();
    }
    out
}

/// A downsampled, premultiplied RGBA copy of a layer for the interactive preview.
#[derive(Clone, Debug)]
pub struct ProxyImage {
    /// Document region the proxy covers (the field bounds).
    pub bounds: Rect,
    /// Document pixels per proxy pixel.
    pub scale: f64,
    pub w: usize,
    pub h: usize,
    /// Premultiplied RGBA, row-major.
    pub px: Vec<[f32; 4]>,
}

const MAX_FULL_RES_SOURCE_PIXELS: usize = 8 * 1024 * 1024;

impl ProxyImage {
    /// Box-downsamples `src` over `bounds` so the longer side is at most `max_side`.
    pub fn new(src: &Surface, bounds: Rect, max_side: usize) -> Self {
        let side = bounds.width().max(bounds.height()).max(1) as f64;
        let scale = (side / max_side.max(1) as f64).max(1.0).ceil();
        let w = ((f64::from(bounds.width()) / scale).ceil() as usize).max(1);
        let h = ((f64::from(bounds.height()) / scale).ceil() as usize).max(1);
        let k = scale as i32;
        let mut px = vec![[0.0f32; 4]; w * h];
        let fmt = src.format();
        let fill_rows = |y: usize, row: &mut [[f32; 4]]| {
            let r = Rect::new(bounds.x0, bounds.y0 + y as i32 * k, bounds.x0 + w as i32 * k, (bounds.y0 + (y as i32 + 1) * k).min(bounds.y1));
            let mut buf = vec![[0.0f32; 4]; r.width() as usize * r.height() as usize];
            src.read_rgba_into(r, &mut buf);
            let rw = r.width() as usize;
            for (x, o) in row.iter_mut().enumerate() {
                let mut acc = [0.0f32; 4];
                let mut cnt = 0.0f32;
                for yy in 0..r.height() as usize {
                    for xx in x * k as usize..((x + 1) * k as usize).min(rw) {
                        let p = buf[yy * rw + xx];
                        let a = if fmt.alpha { p[3] } else { 1.0 };
                        acc[0] += p[0] * a;
                        acc[1] += p[1] * a;
                        acc[2] += p[2] * a;
                        acc[3] += a;
                        cnt += 1.0;
                    }
                }
                if cnt > 0.0 {
                    *o = acc.map(|v| v / cnt);
                }
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            use rayon::prelude::*;
            px.par_chunks_mut(w).enumerate().for_each(|(y, row)| fill_rows(y, row));
        }
        #[cfg(target_arch = "wasm32")]
        px.chunks_mut(w).enumerate().for_each(|(y, row)| fill_rows(y, row));
        ProxyImage { bounds, scale, w, h, px }
    }

    /// Renders proxy pixels `r` (proxy coordinates) through the field into `out` (premultiplied
    /// RGBA8, `w × h`): bilinear field, bilinear source.
    pub fn render(&self, field: &LiquifyField, r: [usize; 4], out: &mut [[u8; 4]]) {
        let [x0, y0, x1, y1] = [r[0].min(self.w), r[1].min(self.h), r[2].min(self.w), r[3].min(self.h)];
        if x0 >= x1 || y0 >= y1 {
            return;
        }
        let s = self.scale;
        let (bx, by) = (f64::from(self.bounds.x0), f64::from(self.bounds.y0));
        let row = |y: usize, line: &mut [[u8; 4]]| {
            let py = by + (y as f64 + 0.5) * s;
            for (x, slot) in line.iter_mut().enumerate().take(x1).skip(x0) {
                let px = bx + (x as f64 + 0.5) * s;
                let d = field.sample(px, py);
                let (u, v) = ((px + d[0] - bx) / s - 0.5, (py + d[1] - by) / s - 0.5);
                let (iu, iv) = (u.floor(), v.floor());
                let (fu, fv) = ((u - iu) as f32, (v - iv) as f32);
                let at = |x: f64, y: f64| self.px[(y.clamp(0.0, (self.h - 1) as f64) as usize) * self.w + x.clamp(0.0, (self.w - 1) as f64) as usize];
                let (a, b, c, e) = (at(iu, iv), at(iu + 1.0, iv), at(iu, iv + 1.0), at(iu + 1.0, iv + 1.0));
                *slot = std::array::from_fn(|k| {
                    let v = (a[k] * (1.0 - fu) + b[k] * fu) * (1.0 - fv) + (c[k] * (1.0 - fu) + e[k] * fu) * fv;
                    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
                });
            }
        };
        let rows = &mut out[y0 * self.w..y1 * self.w];
        // A dab's few thousand pixels are faster on this thread than waking the pool.
        #[cfg(not(target_arch = "wasm32"))]
        if (x1 - x0) * (y1 - y0) > 150_000 {
            use rayon::prelude::*;
            rows.par_chunks_mut(self.w).enumerate().for_each(|(i, line)| row(y0 + i, line));
            return;
        }
        rows.chunks_mut(self.w).enumerate().for_each(|(i, line)| row(y0 + i, line));
    }

    /// Renders a document-pixel-resolution preview for `region` into the corresponding part of
    /// `out`, whose dimensions cover `bounds`. Source reads are limited to the dirty region plus
    /// the displacement range that can affect it, so zoomed-in brush updates do not flatten the
    /// whole layer. `source_scratch` is reused by the caller between updates.
    pub fn render_full_resolution(
        src: &Surface,
        field: &LiquifyField,
        bounds: Rect,
        region: Rect,
        out: &mut [[u8; 4]],
        source_scratch: &mut Vec<[f32; 4]>,
    ) -> bool {
        let bounds = bounds.intersect(&field.bounds);
        let region = region.intersect(&bounds);
        let (width, height) = (bounds.width() as usize, bounds.height() as usize);
        let Some(out_len) = width.checked_mul(height) else { return false };
        if region.is_empty() || out.len() != out_len || width == 0 || height == 0 {
            return false;
        }

        // Bilinear field interpolation is a convex combination of its surrounding nodes. The
        // extrema of those nodes therefore bound every source sample in this output rectangle.
        let (mut min_dx, mut min_dy, mut max_dx, mut max_dy) = (f32::INFINITY, f32::INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        let (gx0, gy0) = field.to_grid(f64::from(region.x0) + 0.5, f64::from(region.y0) + 0.5);
        let (gx1, gy1) = field.to_grid(f64::from(region.x1) - 0.5, f64::from(region.y1) - 0.5);
        let (i0, j0) = (gx0.floor().max(0.0) as usize, gy0.floor().max(0.0) as usize);
        let (i1, j1) = (gx1.ceil().max(0.0) as usize, gy1.ceil().max(0.0) as usize);
        for j in j0..=j1.min(field.h.saturating_sub(1)) {
            for i in i0..=i1.min(field.w.saturating_sub(1)) {
                let Some(v) = j.checked_mul(field.w).and_then(|row| row.checked_add(i)).and_then(|idx| field.d.get(idx)) else { continue };
                min_dx = min_dx.min(v[0]);
                min_dy = min_dy.min(v[1]);
                max_dx = max_dx.max(v[0]);
                max_dy = max_dy.max(v[1]);
            }
        }
        if !min_dx.is_finite() || !min_dy.is_finite() || !max_dx.is_finite() || !max_dy.is_finite() {
            return false;
        }

        // Source coordinates are pixel centres. Add one sample on each side for bilinear
        // interpolation, then clamp to the document just as the regular proxy renderer does.
        let x0 = (f64::from(region.x0) + f64::from(min_dx)).floor() - 1.0;
        let y0 = (f64::from(region.y0) + f64::from(min_dy)).floor() - 1.0;
        let x1 = (f64::from(region.x1 - 1) + f64::from(max_dx)).ceil() + 2.0;
        let y1 = (f64::from(region.y1 - 1) + f64::from(max_dy)).ceil() + 2.0;
        let canvas = field.bounds;
        let src_rect = Rect::new(
            x0.max(f64::from(canvas.x0)).min(f64::from(canvas.x1)) as i32,
            y0.max(f64::from(canvas.y0)).min(f64::from(canvas.y1)) as i32,
            x1.max(f64::from(canvas.x0)).min(f64::from(canvas.x1)) as i32,
            y1.max(f64::from(canvas.y0)).min(f64::from(canvas.y1)) as i32,
        );
        if src_rect.is_empty() {
            return false;
        }
        let (sw, sh) = (src_rect.width() as usize, src_rect.height() as usize);
        let Some(source_len) = sw.checked_mul(sh) else { return false };
        if source_len > MAX_FULL_RES_SOURCE_PIXELS {
            return false;
        }
        if source_scratch.try_reserve(source_len.saturating_sub(source_scratch.len())).is_err() {
            return false;
        }
        source_scratch.resize(source_len, [0.0; 4]);
        src.read_rgba_into(src_rect, source_scratch);
        if !src.format().alpha {
            for pixel in source_scratch.iter_mut() {
                pixel[3] = 1.0;
            }
        }
        for pixel in source_scratch.iter_mut() {
            let alpha = pixel[3];
            pixel[0] *= alpha;
            pixel[1] *= alpha;
            pixel[2] *= alpha;
        }

        let x0 = (region.x0 - bounds.x0) as usize;
        let x1 = (region.x1 - bounds.x0) as usize;
        let y0 = (region.y0 - bounds.y0) as usize;
        let y1 = (region.y1 - bounds.y0) as usize;
        let source_at = |x: f64, y: f64| -> [f32; 4] {
            let fx = (x - 0.5).clamp(f64::from(canvas.x0), f64::from(canvas.x1 - 1));
            let fy = (y - 0.5).clamp(f64::from(canvas.y0), f64::from(canvas.y1 - 1));
            let ix = fx.floor();
            let iy = fy.floor();
            let (tx, ty) = ((fx - ix) as f32, (fy - iy) as f32);
            let sx0 = (ix as i32).clamp(src_rect.x0, src_rect.x1 - 1);
            let sy0 = (iy as i32).clamp(src_rect.y0, src_rect.y1 - 1);
            let sx1 = (sx0 + 1).min(src_rect.x1 - 1);
            let sy1 = (sy0 + 1).min(src_rect.y1 - 1);
            let at = |sx: i32, sy: i32| -> [f32; 4] {
                let ox = (sx - src_rect.x0) as usize;
                let oy = (sy - src_rect.y0) as usize;
                source_scratch.get(oy * sw + ox).copied().unwrap_or([0.0; 4])
            };
            let (a, b, c, d) = (at(sx0, sy0), at(sx1, sy0), at(sx0, sy1), at(sx1, sy1));
            std::array::from_fn(|channel| (a[channel] * (1.0 - tx) + b[channel] * tx) * (1.0 - ty) + (c[channel] * (1.0 - tx) + d[channel] * tx) * ty)
        };
        let render_rows = |first_y: usize, rows: &mut [[u8; 4]]| {
            for (offset, row) in rows.chunks_mut(width).enumerate() {
                let y = bounds.y0 + (first_y + offset) as i32;
                let Some(dest) = row.get_mut(x0..x1) else { continue };
                for (offset_x, pixel) in dest.iter_mut().enumerate() {
                    let x = bounds.x0 + (x0 + offset_x) as i32;
                    let p = [f64::from(x) + 0.5, f64::from(y) + 0.5];
                    let d = field.sample(p[0], p[1]);
                    let rgba = source_at(p[0] + d[0], p[1] + d[1]);
                    *pixel = rgba.map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                }
            }
        };
        let Some(rows) = out.get_mut(y0 * width..y1 * width) else { return false };
        #[cfg(not(target_arch = "wasm32"))]
        if (x1 - x0).saturating_mul(y1 - y0) > 150_000 {
            use rayon::prelude::*;
            rows.par_chunks_mut(width).enumerate().for_each(|(offset, line)| render_rows(y0 + offset, line));
            return true;
        }
        rows.chunks_mut(width).enumerate().for_each(|(offset, line)| render_rows(y0 + offset, line));
        true
    }

    /// The proxy pixel rectangle covering document rect `r`.
    pub fn proxy_rect(&self, r: Rect) -> [usize; 4] {
        let s = self.scale;
        let f = |v: i32, o: i32| (f64::from(v - o) / s).max(0.0);
        [
            f(r.x0, self.bounds.x0).floor() as usize,
            f(r.y0, self.bounds.y0).floor() as usize,
            (f(r.x1, self.bounds.x0).ceil() as usize + 1).min(self.w),
            (f(r.y1, self.bounds.y0).ceil() as usize + 1).min(self.h),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::{ColorMode, PixelFormat, SampleType};

    /// The segment walk before #937: every dab, whether or not it can reach the field.
    fn full_walk(f: &mut LiquifyField, s: &LiquifyStroke, a: [f64; 3], b: [f64; 3]) -> Rect {
        f.walk_segment(s, a, b, false)
    }

    #[test]
    fn a_segment_walks_only_the_dabs_that_can_reach_the_field() {
        // #937: a segment to a far-off point walked one dab per spacing step along its whole
        // length (~3e14 for 1e15 px). Skipping the dabs that can't reach a field node must leave
        // every result bit-identical to the full walk.
        use LiquifyTool::*;
        let bounds = Rect::new(-7, 3, 89, 67);
        let segments: [([f64; 3], [f64; 3]); 10] = [
            ([10.0, 20.0, 1.0], [70.0, 50.0, 0.5]),        // inside
            ([-900.0, 30.0, 1.0], [40.0, 31.0, 1.0]),      // in from far left
            ([40.0, 30.0, 1.0], [40.0, 4000.0, 0.2]),      // out downwards
            ([-500.0, -500.0, 1.0], [600.0, 600.0, 1.0]),  // diagonal through
            ([-300.0, -9.0, 1.0], [300.0, -9.0, 1.0]),     // parallel, just above (the brush reaches)
            ([-300.0, -400.0, 1.0], [300.0, -400.0, 1.0]), // parallel, far above (reaches nothing)
            ([88.5, -50.0, 1.0], [88.5, 120.0, 1.0]),      // vertical, along the right edge
            ([200.0, 200.0, 1.0], [201.0, 900.0, 1.0]),    // entirely outside
            ([0.0, 0.0, 1.0], [0.25, 0.1, 1.0]),           // shorter than one step
            ([60.0, 40.0, 1.0], [-2000.0, 45.0, 1.0]),     // continues the stroke before it, out left
        ];
        for tool in [ForwardWarp, PushLeft, Reconstruct, Smooth, TwirlCw, TwirlCcw, Pucker, Bloat, Freeze, Thaw] {
            for size in [6.0, 40.0] {
                for (a, b) in segments {
                    let s = LiquifyStroke::new(tool, size);
                    let (mut fast, mut full) = (LiquifyField::new(bounds, 2.0), LiquifyField::new(bounds, 2.0));
                    // Some displacement first, so Reconstruct and Smooth have something to undo.
                    for f in [&mut fast, &mut full] {
                        f.stroke_segment(&LiquifyStroke::new(ForwardWarp, 30.0), [20.0, 20.0, 1.0], [60.0, 40.0, 1.0]);
                    }
                    let dirty = fast.stroke_segment(&s, a, b);
                    assert_eq!(dirty, full_walk(&mut full, &s, a, b), "{tool:?} {size} {a:?} {b:?}");
                    assert!(fast.d == full.d && fast.freeze == full.freeze, "{tool:?} {size} {a:?} {b:?}");
                }
            }
        }
    }

    /// The same path sampled by `parts` pointer events per corner-to-corner leg, unevenly.
    fn sampled(path: &[[f64; 3]], parts: usize) -> Vec<Vec<f64>> {
        let mut out = vec![path[0].to_vec()];
        for w in path.windows(2) {
            let (a, b) = (w[0], w[1]);
            // Uneven steps, as a real pointer gives: t grows quadratically.
            for k in 1..=parts {
                let t = (k * k) as f64 / (parts * parts) as f64;
                out.push((0..3).map(|i| a[i] + (b[i] - a[i]) * t).collect());
            }
        }
        out
    }

    /// #2606: dabs restarted at every pointer event, so a fast mouse (few far-apart events) and a
    /// slow one (many close ones) warped the same path differently. Spacing now runs on along the
    /// whole stroke: any sampling of a path gives the same field.
    #[test]
    fn the_result_does_not_depend_on_how_finely_the_path_was_sampled() {
        use LiquifyTool::*;
        let path = [[20.0, 20.0, 1.0], [62.0, 26.0, 0.6], [70.0, 50.0, 0.9], [28.0, 55.0, 1.0], [30.0, 30.0, 1.0]];
        for tool in [ForwardWarp, PushLeft, TwirlCw, Bloat, Reconstruct, Smooth, Freeze] {
            for size in [12.0, 40.0] {
                let field = |parts: usize| {
                    let mut warp = LiquifyStroke::new(ForwardWarp, 30.0);
                    warp.points = vec![vec![30.0, 30.0], vec![50.0, 40.0]];
                    let mut s = LiquifyStroke::new(tool, size);
                    s.points = sampled(&path, parts);
                    // Some displacement first, so Reconstruct and Smooth have something to change.
                    LiquifyField::from_strokes(bounds(), 2.0, &[warp, s])
                };
                let coarse = field(1);
                for parts in [3, 17, 60] {
                    let fine = field(parts);
                    let dd = coarse.d.iter().zip(&fine.d).map(|(p, q)| (p[0] - q[0]).abs().max((p[1] - q[1]).abs())).fold(0.0f32, f32::max);
                    let df = coarse.freeze.iter().zip(&fine.freeze).map(|(p, q)| (p - q).abs()).fold(0.0f32, f32::max);
                    assert!(dd < 1e-3 && df < 1e-4, "{tool:?} {size}px, {parts} events per leg: field differs by {dd} px, mask by {df}");
                }
            }
        }
    }

    /// A long drag whose events land exactly on dab positions, along a diagonal: rounding must not
    /// add or drop the dab at a segment's end, or the deep displacement a long Forward Warp builds
    /// up differs by a whole dab between a slow and a fast pointer.
    #[test]
    fn a_long_diagonal_drag_gives_one_field_at_any_event_spacing() {
        let field = |step: f64| {
            let mut s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 40.0);
            s.points = (0..=(300.0 / step) as usize).map(|k| vec![40.0 + k as f64 * step * 0.6, 20.0 + k as f64 * step * 0.8]).collect();
            LiquifyField::from_strokes(Rect::new(0, 0, 320, 300), 2.0, &[s])
        };
        let (slow, fast) = (field(2.0), field(60.0));
        let dd = slow.d.iter().zip(&fast.d).map(|(p, q)| (p[0] - q[0]).abs().max((p[1] - q[1]).abs())).fold(0.0f32, f32::max);
        assert!(slow.max_displacement() > 50.0, "the drag builds up a deep displacement");
        assert!(dd < 1e-3, "slow and fast pointer fields differ by {dd} px");
    }

    #[test]
    fn a_stroke_interrupted_by_a_jump_restarts_its_spacing() {
        // A segment that does not start where the last one ended begins a new run of dabs there.
        let s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 20.0);
        let mut f = LiquifyField::new(bounds(), 2.0);
        f.stroke_begin(&s, [10.0, 10.0, 1.0]);
        f.stroke_segment(&s, [10.0, 10.0, 1.0], [10.3, 10.0, 1.0]);
        let mut g = LiquifyField::new(bounds(), 2.0);
        f.stroke_segment(&s, [40.0, 30.0, 1.0], [50.0, 30.0, 1.0]);
        g.stroke_begin(&s, [40.0, 30.0, 1.0]);
        g.stroke_segment(&s, [40.0, 30.0, 1.0], [50.0, 30.0, 1.0]);
        assert_eq!(f, g);
        // Hostile points: nothing panics and the next segment starts afresh.
        f.stroke_segment(&s, [f64::NAN, 3.0, 1.0], [4.0, f64::INFINITY, 1.0]);
        f.stroke_segment(&s, [4.0, 3.0, f64::NAN], [9.0, 3.0, 1.0]);
    }

    #[test]
    fn a_segment_to_a_far_off_point_returns() {
        // The issue's request: about 3.3e14 steps before the fix. Also a segment whose length
        // overflows f64.
        let mut f = LiquifyField::new(Rect::new(0, 0, 24, 16), 2.0);
        let s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 30.0);
        let dirty = f.stroke_segment(&s, [0.0, 0.0, 1.0], [1e15, 0.0, 1.0]);
        assert!(!dirty.is_empty() && dirty.x1 <= 64, "{dirty:?}");
        f.stroke_segment(&s, [-1e308, 5.0, 1.0], [1e308, 5.0, 1.0]);
        f.stroke_segment(&LiquifyStroke::new(LiquifyTool::Bloat, 30.0), [5.0, -1e15, 1.0], [5.0, 1e15, 1.0]);
    }

    fn sample(st: SampleType) -> Surface {
        let mut s = Surface::new(PixelFormat::new(ColorMode::Rgb, st, true));
        for y in 0..64 {
            for x in 0..96 {
                let v = ((x / 4 + y / 4) % 2) as f32;
                s.write_pixel(x, y, &[v, (x as f32) / 96.0, (y as f32) / 64.0, 1.0]);
            }
        }
        s
    }

    fn bounds() -> Rect {
        Rect::new(0, 0, 96, 64)
    }

    fn worst(a: &Surface, b: &Surface, r: Rect) -> f32 {
        a.read_region(r).iter().zip(b.read_region(r)).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    fn lasso(points: &[[f64; 2]], amount: Option<f64>) -> LiquifyStroke {
        let mut s = LiquifyStroke::new(LiquifyTool::LassoMask, 1.0);
        s.points = points.iter().map(|p| p.to_vec()).collect();
        s.amount = amount;
        s
    }

    /// Reference even-odd test (ray casting), to check the scanline fill against.
    fn inside(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
        let mut inside = false;
        let mut j = poly.len() - 1;
        for (i, a) in poly.iter().enumerate() {
            let b = poly[j];
            if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < (b[0] - a[0]) * (p[1] - a[1]) / (b[1] - a[1]) + a[0] {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    #[test]
    fn lasso_freezes_exactly_the_nodes_inside_a_concave_polygon() {
        // A concave "C" with a notch, at a non-integer cell size.
        let poly = [[10.0, 8.0], [70.0, 8.0], [70.0, 20.0], [30.0, 22.5], [33.0, 40.0], [80.0, 41.0], [60.0, 58.0], [9.0, 55.0]];
        for cell in [1.0, 2.0, 3.5] {
            let mut f = LiquifyField::new(bounds(), cell);
            let dirty = f.apply_stroke(&lasso(&poly, None));
            assert_eq!(dirty, Rect::new(9, 8, 80, 58));
            for j in 0..f.h {
                for i in 0..f.w {
                    let want = if inside(f.node_pos(i, j), &poly) { 1.0 } else { 0.0 };
                    assert_eq!(f.freeze[j * f.w + i], want, "cell {cell}: node ({i}, {j}) at {:?}", f.node_pos(i, j));
                }
            }
            assert!(f.freeze.contains(&1.0));
            // Strokes replay to the same field (undo rebuilds from the list).
            assert_eq!(LiquifyField::from_strokes(bounds(), cell, &[lasso(&poly, None)]), f);
        }
    }

    #[test]
    fn lasso_thaws_with_amount_zero() {
        let mut f = LiquifyField::new(bounds(), 2.0);
        f.apply_stroke(&LiquifyStroke::new(LiquifyTool::FreezeAll, 1.0));
        assert!(f.freeze.iter().all(|&v| v == 1.0));
        let square = [[20.0, 20.0], [40.0, 20.0], [40.0, 40.0], [20.0, 40.0]];
        f.apply_stroke(&lasso(&square, Some(0.0)));
        assert_eq!(f.freeze_at(30.0, 30.0), 0.0, "thawed inside");
        assert_eq!(f.freeze_at(5.0, 5.0), 1.0, "still frozen outside");
    }

    #[test]
    fn lasso_ignores_degenerate_and_hostile_polygons() {
        let empty = LiquifyField::new(bounds(), 2.0);
        let cases: [&[[f64; 2]]; 6] = [
            &[],
            &[[10.0, 10.0], [50.0, 50.0]],
            &[[f64::NAN, 10.0], [50.0, f64::INFINITY], [20.0, 30.0]],
            // Entirely outside the canvas, on every side.
            &[[-50.0, -50.0], [-10.0, -50.0], [-10.0, -10.0]],
            &[[200.0, 10.0], [300.0, 10.0], [250.0, 50.0]],
            &[[10.0, 1e12], [50.0, 1e12], [30.0, 2e12]],
        ];
        for poly in cases {
            let mut f = empty.clone();
            assert_eq!(f.apply_stroke(&lasso(poly, None)), Rect::EMPTY, "{poly:?}");
            assert_eq!(f, empty, "{poly:?}");
        }
        // Huge coordinates around the canvas clamp to it instead of overflowing.
        let mut f = empty.clone();
        f.apply_stroke(&lasso(&[[-1e15, -1e15], [1e15, -1e15], [1e15, 1e15], [-1e15, 1e15]], None));
        assert!(f.freeze.iter().all(|&v| v == 1.0));
        // A NaN point among good ones is skipped.
        let mut f = empty.clone();
        f.apply_stroke(&lasso(&[[10.0, 10.0], [f64::NAN, 0.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]], None));
        assert_eq!(f.freeze_at(30.0, 30.0), 1.0);
    }

    #[test]
    fn the_brush_fades_to_nothing_at_its_rim_at_every_density() {
        for density in [0.0, 0.25, 0.5, 0.75, 1.0, f64::NAN, -3.0, 7.0] {
            assert_eq!(falloff(0.0, density), 1.0, "full strength at the centre ({density})");
            assert_eq!(falloff(1.0, density), 0.0, "nothing at the rim ({density})");
            // Flat at the rim: the last 2 % of the radius moves at most a couple of percent.
            assert!(falloff(0.98f64.powi(2), density) < 0.02, "{density}: {}", falloff(0.98f64.powi(2), density));
            let mut prev = 1.0;
            for k in 1..=100 {
                let v = falloff(k as f64 / 100.0, density);
                assert!(v <= prev && v >= 0.0, "{density}: not falling at t² = {k}%");
                prev = v;
            }
        }
        // Density widens the strong core: more strength at every radius, as in Photoshop.
        for k in 1..100 {
            let t2 = k as f64 / 100.0;
            assert!(falloff(t2, 1.0) >= falloff(t2, 0.5) && falloff(t2, 0.5) >= falloff(t2, 0.0), "t² = {t2}");
        }
        assert!(falloff(0.5, 1.0) > 0.5 && falloff(0.5, 0.0) < 0.2);
    }

    /// #2606: the dab's strength used to stop at 1 - Density at its rim, so a Forward Warp moved
    /// everything just inside the brush and nothing just outside, tearing the image along every
    /// dab's circle (a hard-edged block at Density 0, half strength at the default 50).
    #[test]
    fn forward_warp_leaves_the_brush_rim_in_place() {
        // A drag of one dab spacing (a twentieth of the radius): one dab, centred where it started.
        let (r, drag) = (20.0, 1.0);
        for density in [0.0, 50.0, 100.0] {
            let mut s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 2.0 * r);
            s.density = density;
            s.points = vec![vec![48.0, 32.0], vec![48.0 + drag, 32.0]];
            let f = LiquifyField::from_strokes(bounds(), 1.0, &[s]);
            let mut rim = 0;
            for j in 0..f.h {
                for i in 0..f.w {
                    let p = f.node_pos(i, j);
                    let t = (p[0] - 48.0).hypot(p[1] - 32.0) / r;
                    if (0.97..1.0).contains(&t) {
                        let d = f.d[j * f.w + i];
                        assert!(f64::from(d[0].hypot(d[1])) < 0.03 * drag, "density {density}: a node at {t:.3} r moved {d:?}");
                        rim += 1;
                    }
                }
            }
            assert!(rim > 20, "{rim} rim nodes");
            assert!(f.sample(48.0, 32.0)[0] < -0.9 * drag, "density {density}: the centre follows the brush");
        }
    }

    #[test]
    fn no_strokes_is_identity_at_every_depth() {
        for st in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let s = sample(st);
            let f = LiquifyField::new(bounds(), 2.0);
            assert_eq!(apply_liquify(&s, &f), s);
            // A field with exactly-zero strokes (zero pressure) is still identity.
            let mut st0 = LiquifyStroke::new(LiquifyTool::ForwardWarp, 30.0);
            st0.pressure = 0.0;
            st0.points = vec![vec![40.0, 30.0], vec![60.0, 30.0]];
            let f = LiquifyField::from_strokes(bounds(), 2.0, &[st0]);
            assert!(f.is_identity());
        }
    }

    #[test]
    fn full_resolution_preview_matches_the_unit_scale_proxy_and_partial_updates() {
        let canvas = Rect::new(0, 0, 96, 64);
        for sample_type in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let source = sample(sample_type);
            let mut stroke = LiquifyStroke::new(LiquifyTool::ForwardWarp, 32.0);
            stroke.points = vec![vec![38.0, 30.0, 1.0], vec![52.0, 34.0, 0.8]];
            let mut field = LiquifyField::from_strokes(canvas, 2.0, &[stroke]);

            let proxy = ProxyImage::new(&source, canvas, 96);
            assert_eq!(proxy.scale, 1.0);
            let mut want = vec![[0; 4]; proxy.w * proxy.h];
            proxy.render(&field, [0, 0, proxy.w, proxy.h], &mut want);

            let mut got = vec![[0; 4]; proxy.w * proxy.h];
            let mut scratch = Vec::new();
            assert!(ProxyImage::render_full_resolution(&source, &field, canvas, canvas, &mut got, &mut scratch));
            assert_eq!(got, want, "full-resolution render at {sample_type:?}");

            let mut more = LiquifyStroke::new(LiquifyTool::Bloat, 16.0);
            more.points = vec![vec![43.0, 32.0, 1.0]];
            let dirty = field.apply_stroke(&more);
            proxy.render(&field, proxy.proxy_rect(dirty), &mut want);
            assert!(ProxyImage::render_full_resolution(&source, &field, canvas, dirty, &mut got, &mut scratch));
            assert_eq!(got, want, "partial update at {sample_type:?}");
        }
    }

    #[test]
    fn forward_warp_displaces_along_the_stroke_and_reconstruct_restores() {
        let mut s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 40.0);
        s.points = vec![vec![40.0, 32.0], vec![52.0, 32.0]];
        let mut f = LiquifyField::from_strokes(bounds(), 2.0, &[s]);
        // Content moved right: the output at the end point samples from the left.
        let d = f.sample(52.0, 32.0);
        assert!(d[0] < -4.0 && d[1].abs() < 0.5, "{d:?}");
        // A pixel-level check: a marker column follows the stroke.
        let mut img = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::F32, true));
        img.fill_rect(bounds(), &[0.0, 1.0]);
        img.fill_rect(Rect::new(40, 28, 42, 36), &[1.0, 1.0]);
        let out = apply_liquify(&img, &f);
        let brightest = (30..70).max_by(|a, b| out.pixel(*a, 32)[0].total_cmp(&out.pixel(*b, 32)[0])).unwrap();
        assert!(brightest > 44, "marker moved right: {brightest}");
        // Reconstruct All restores the original exactly.
        let mut rec = LiquifyStroke::new(LiquifyTool::ReconstructAll, 10.0);
        rec.amount = Some(100.0);
        f.apply_stroke(&rec);
        assert!(f.is_identity());
        assert_eq!(apply_liquify(&img, &f), img);
    }

    #[test]
    fn reconstruct_brush_reduces_the_field_and_freeze_protects() {
        let mut s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 40.0);
        s.points = vec![vec![30.0, 32.0], vec![45.0, 32.0]];
        let mut f = LiquifyField::from_strokes(bounds(), 2.0, std::slice::from_ref(&s));
        let before = f.max_displacement();
        let mut r = LiquifyStroke::new(LiquifyTool::Reconstruct, 80.0);
        r.points = vec![vec![40.0, 32.0]; 30];
        f.apply_stroke(&r);
        assert!(f.max_displacement() < before * 0.2, "{} vs {before}", f.max_displacement());
        // Frozen area doesn't move.
        let mut fr = LiquifyStroke::new(LiquifyTool::Freeze, 60.0);
        fr.density = 100.0;
        fr.points = vec![vec![40.0, 32.0]];
        let mut g = LiquifyField::new(bounds(), 2.0);
        g.apply_stroke(&fr);
        assert!(g.freeze_at(40.0, 32.0) > 0.99);
        g.apply_stroke(&s);
        assert!(g.sample(40.0, 32.0)[0].abs() < 1e-3);
        // Thaw unfreezes.
        let mut th = fr.clone();
        th.tool = LiquifyTool::Thaw;
        g.apply_stroke(&th);
        assert!(g.freeze_at(40.0, 32.0) < 0.01);
        // Mask All / Invert / None.
        g.apply_stroke(&LiquifyStroke::new(LiquifyTool::FreezeAll, 1.0));
        assert!(g.freeze.iter().all(|f| *f == 1.0));
        g.apply_stroke(&LiquifyStroke::new(LiquifyTool::InvertFreeze, 1.0));
        assert!(g.freeze.iter().all(|f| *f == 0.0));
        g.apply_stroke(&fr);
        g.apply_stroke(&LiquifyStroke::new(LiquifyTool::InvertFreeze, 1.0));
        assert!(g.freeze_at(40.0, 32.0) < 0.01 && g.freeze_at(5.0, 5.0) > 0.99);
        g.apply_stroke(&LiquifyStroke::new(LiquifyTool::ThawAll, 1.0));
        assert!(g.freeze.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn twirl_pucker_bloat_and_push_left_have_the_right_sense() {
        let at = |tool: LiquifyTool, pts: Vec<Vec<f64>>| {
            let mut s = LiquifyStroke::new(tool, 40.0);
            s.points = pts;
            LiquifyField::from_strokes(bounds(), 2.0, &[s])
        };
        let c = vec![vec![48.0, 32.0]; 10];
        // Pucker pulls content in: the output samples outward (d points away from the centre).
        let f = at(LiquifyTool::Pucker, c.clone());
        assert!(f.sample(58.0, 32.0)[0] > 0.1);
        let f = at(LiquifyTool::Bloat, c.clone());
        assert!(f.sample(58.0, 32.0)[0] < -0.1);
        // Twirl clockwise (y down): content right of centre moves down, so it samples from above.
        let f = at(LiquifyTool::TwirlCw, c.clone());
        assert!(f.sample(58.0, 32.0)[1] < -0.1, "{:?}", f.sample(58.0, 32.0));
        let f = at(LiquifyTool::TwirlCcw, c);
        assert!(f.sample(58.0, 32.0)[1] > 0.1);
        // Push Left while dragging up moves content left (samples from the right).
        let f = at(LiquifyTool::PushLeft, vec![vec![48.0, 44.0], vec![48.0, 20.0]]);
        assert!(f.sample(48.0, 32.0)[0] > 0.5, "{:?}", f.sample(48.0, 32.0));
        // Smooth reduces sharp field differences.
        let mut s = LiquifyStroke::new(LiquifyTool::ForwardWarp, 12.0);
        s.points = vec![vec![40.0, 32.0], vec![50.0, 32.0]];
        let mut g = LiquifyField::from_strokes(bounds(), 2.0, &[s]);
        let m0 = g.max_displacement();
        let mut sm = LiquifyStroke::new(LiquifyTool::Smooth, 60.0);
        sm.points = vec![vec![45.0, 32.0]; 20];
        g.apply_stroke(&sm);
        assert!(g.max_displacement() < m0);
    }

    #[test]
    fn replay_is_deterministic_and_depths_agree() {
        let mut s = LiquifyStroke::new(LiquifyTool::TwirlCw, 50.0);
        s.points = vec![vec![48.0, 32.0, 0.5], vec![50.0, 33.0, 1.0], vec![52.0, 34.0]];
        let a = LiquifyField::from_strokes(bounds(), 2.0, std::slice::from_ref(&s));
        let b = LiquifyField::from_strokes(bounds(), 2.0, std::slice::from_ref(&s));
        assert_eq!(a, b);
        let r8 = apply_liquify(&sample(SampleType::U8), &a);
        let r32 = apply_liquify(&sample(SampleType::F32), &a);
        assert!(worst(&r8, &r32, bounds()) < 2.0 / 255.0 + 1e-4);
    }

    #[test]
    fn proxy_matches_the_full_render_roughly() {
        let src = sample(SampleType::U8);
        let mut s = LiquifyStroke::new(LiquifyTool::Bloat, 40.0);
        s.points = vec![vec![48.0, 32.0]; 8];
        let f = LiquifyField::from_strokes(bounds(), 2.0, &[s]);
        let p = ProxyImage::new(&src, bounds(), 96);
        assert_eq!((p.w, p.h, p.scale), (96, 64, 1.0));
        let mut out = vec![[0u8; 4]; p.w * p.h];
        p.render(&f, [0, 0, p.w, p.h], &mut out);
        let full = apply_liquify(&src, &f);
        let mut bad = 0;
        for y in 0..64 {
            for x in 0..96 {
                let q = full.pixel(x, y);
                let o = out[y as usize * 96 + x as usize];
                if (f32::from(o[1]) / 255.0 - q[1]).abs() > 0.1 {
                    bad += 1;
                }
            }
        }
        assert!(bad < 96 * 64 / 50, "{bad}");
    }
}
