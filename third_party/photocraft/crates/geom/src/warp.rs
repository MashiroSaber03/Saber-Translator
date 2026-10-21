//! Warp geometry shared by Edit › Transform › Warp, smart-object warps and Warp Text.
//!
//! Two kinds of warp, both forward maps from a source box to document space:
//! - **Styles** ([`WarpStyle`], Photoshop's fifteen presets plus bend and horizontal/vertical
//!   distortion, the PSD `warp` descriptor's `warpStyle`/`warpValue`/`warpPerspective*`). Each
//!   is a single bicubic Bezier patch whose control grid follows the bend: measured from
//!   Photoshop's renders (see [`preset_grid`]), not taken from Adobe's code. Every style is the
//!   identity at bend 0 and continuous in the bend.
//! - **Custom meshes** ([`BezierMesh`]): a grid of bicubic Bezier patches. The default is one
//!   patch (a 4×4 control grid); split warps add patch rows/columns by exact de Casteljau
//!   subdivision, so splitting never changes the shape.
//!
//! Pure geometry: resampling lives in `photocraft-algo::warp`.

use std::f64::consts::PI;

use serde::{Deserialize, Serialize};

/// Warp style: none, a custom mesh, or one of Photoshop's presets (menu order).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum WarpStyle {
    #[default]
    None,
    Custom,
    Arc,
    ArcLower,
    ArcUpper,
    Arch,
    Bulge,
    ShellLower,
    ShellUpper,
    Flag,
    Wave,
    Fish,
    Rise,
    Fisheye,
    Inflate,
    Squeeze,
    Twist,
}

impl WarpStyle {
    /// The fifteen presets in Photoshop's menu order.
    pub const PRESETS: [WarpStyle; 15] = [
        WarpStyle::Arc,
        WarpStyle::ArcLower,
        WarpStyle::ArcUpper,
        WarpStyle::Arch,
        WarpStyle::Bulge,
        WarpStyle::ShellLower,
        WarpStyle::ShellUpper,
        WarpStyle::Flag,
        WarpStyle::Wave,
        WarpStyle::Fish,
        WarpStyle::Rise,
        WarpStyle::Fisheye,
        WarpStyle::Inflate,
        WarpStyle::Squeeze,
        WarpStyle::Twist,
    ];

    /// Every style (none, custom, presets).
    pub fn all() -> impl Iterator<Item = WarpStyle> {
        [WarpStyle::None, WarpStyle::Custom].into_iter().chain(Self::PRESETS)
    }

    /// PSD `warpStyle` enum value (`warpArc`, `warpCustom`, `warpNone`…).
    pub fn psd_name(self) -> &'static str {
        match self {
            WarpStyle::None => "warpNone",
            WarpStyle::Custom => "warpCustom",
            WarpStyle::Arc => "warpArc",
            WarpStyle::ArcLower => "warpArcLower",
            WarpStyle::ArcUpper => "warpArcUpper",
            WarpStyle::Arch => "warpArch",
            WarpStyle::Bulge => "warpBulge",
            WarpStyle::ShellLower => "warpShellLower",
            WarpStyle::ShellUpper => "warpShellUpper",
            WarpStyle::Flag => "warpFlag",
            WarpStyle::Wave => "warpWave",
            WarpStyle::Fish => "warpFish",
            WarpStyle::Rise => "warpRise",
            WarpStyle::Fisheye => "warpFisheye",
            WarpStyle::Inflate => "warpInflate",
            WarpStyle::Squeeze => "warpSqueeze",
            WarpStyle::Twist => "warpTwist",
        }
    }

    /// Short id used by commands (`arc`, `arcLower`, `custom`, `none`).
    pub fn id(self) -> &'static str {
        let p = self.psd_name().trim_start_matches("warp");
        // Lower-case first letter without allocating: every id is listed.
        match p {
            "None" => "none",
            "Custom" => "custom",
            "Arc" => "arc",
            "ArcLower" => "arcLower",
            "ArcUpper" => "arcUpper",
            "Arch" => "arch",
            "Bulge" => "bulge",
            "ShellLower" => "shellLower",
            "ShellUpper" => "shellUpper",
            "Flag" => "flag",
            "Wave" => "wave",
            "Fish" => "fish",
            "Rise" => "rise",
            "Fisheye" => "fisheye",
            "Inflate" => "inflate",
            "Squeeze" => "squeeze",
            _ => "twist",
        }
    }

    /// Menu label (`Arc Lower`).
    pub fn label(self) -> &'static str {
        match self {
            WarpStyle::None => "None",
            WarpStyle::Custom => "Custom",
            WarpStyle::Arc => "Arc",
            WarpStyle::ArcLower => "Arc Lower",
            WarpStyle::ArcUpper => "Arc Upper",
            WarpStyle::Arch => "Arch",
            WarpStyle::Bulge => "Bulge",
            WarpStyle::ShellLower => "Shell Lower",
            WarpStyle::ShellUpper => "Shell Upper",
            WarpStyle::Flag => "Flag",
            WarpStyle::Wave => "Wave",
            WarpStyle::Fish => "Fish",
            WarpStyle::Rise => "Rise",
            WarpStyle::Fisheye => "Fisheye",
            WarpStyle::Inflate => "Inflate",
            WarpStyle::Squeeze => "Squeeze",
            WarpStyle::Twist => "Twist",
        }
    }

    /// Parses a short id, a PSD name or a label (case-insensitive, spaces ignored).
    pub fn parse(s: &str) -> Option<WarpStyle> {
        let k: String = s.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_ascii_lowercase();
        Self::all().find(|w| w.id().to_ascii_lowercase() == k || w.psd_name().to_ascii_lowercase() == k)
    }

    pub fn is_preset(self) -> bool {
        !matches!(self, WarpStyle::None | WarpStyle::Custom)
    }
}

/// A preset style prepared for one box: its control grid and the box it bends.
#[derive(Clone, Copy, Debug)]
pub struct StyleWarp {
    /// The 4 × 4 Bezier control grid in box units (0..1, y down), row-major.
    grid: Grid,
    /// Box origin and size.
    o: (f64, f64),
    size: (f64, f64),
}

/// A single bicubic Bezier patch's control points, `[row][column]`, in box units.
type Grid = [[[f64; 2]; 4]; 4];

impl StyleWarp {
    /// Prepares `style` with bend / distortions in percent (-100..100) for `[x0, y0, x1, y1]`.
    /// `None` for non-presets, an all-zero warp or an empty box.
    pub fn new(style: WarpStyle, bend: f64, h_distort: f64, v_distort: f64, vertical: bool, bounds: [f64; 4]) -> Option<StyleWarp> {
        if !style.is_preset() {
            return None;
        }
        let [x0, y0, x1, y1] = bounds;
        let (hw, hh) = ((x1 - x0) / 2.0, (y1 - y0) / 2.0);
        if !(hw > 0.0 && hh > 0.0) {
            return None;
        }
        let pct = |v: f64| if v.is_finite() { (v / 100.0).clamp(-1.0, 1.0) } else { 0.0 };
        let (b, hd, vd) = (pct(bend), pct(h_distort), pct(v_distort));
        if b == 0.0 && hd == 0.0 && vd == 0.0 {
            return None;
        }
        let grid = preset_grid(style, b, hd, vd, vertical, hh / hw);
        Some(StyleWarp { grid, o: (x0, y0), size: (2.0 * hw, 2.0 * hh) })
    }

    /// Longest segment (px) worth sending through [`StyleWarp::apply`] unsplit.
    pub fn max_segment(&self) -> f64 {
        (self.size.0.min(self.size.1) / 32.0).max(0.5)
    }

    /// Maps a point.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (u, v) = ((x - self.o.0) / self.size.0, (y - self.o.1) / self.size.1);
        let (bu, bv) = (bernstein(u), bernstein(v));
        let (mut px, mut py) = (0.0, 0.0);
        for (row, wv) in self.grid.iter().zip(bv) {
            for (p, wu) in row.iter().zip(bu) {
                px += wu * wv * p[0];
                py += wu * wv * p[1];
            }
        }
        (self.o.0 + px * self.size.0, self.o.1 + py * self.size.1)
    }

    /// The style as one Bezier patch over its box (every preset is one exactly).
    pub fn mesh(&self) -> BezierMesh {
        let points = self.grid.iter().flatten().map(|p| [self.o.0 + p[0] * self.size.0, self.o.1 + p[1] * self.size.1]).collect();
        BezierMesh { us: vec![0.0, 1.0], vs: vec![0.0, 1.0], points }
    }
}

/// The rest grid (no warp).
fn rest_grid() -> Grid {
    std::array::from_fn(|j| std::array::from_fn(|i| [i as f64 / 3.0, j as f64 / 3.0]))
}

/// A horizontal edge at height `y` (box units, 0..1 across) bent into a circular arc of `|b|·π`
/// that keeps its chord, bowing up when `up == (b > 0)`. `h` is the box's height over its width:
/// the arc is circular in pixels.
fn edge_arc(y: f64, b: f64, up: bool, h: f64) -> [[f64; 2]; 4] {
    if b.abs() < 1e-9 {
        return std::array::from_fn(|i| [i as f64 / 3.0, y]);
    }
    let th = b.abs() * PI;
    let k = 4.0 / 3.0 * (th / 4.0).tan() * 0.5 / (th / 2.0).sin();
    let s = if (b > 0.0) == up { -1.0 } else { 1.0 };
    let (c, n) = ((th / 2.0).cos(), (th / 2.0).sin());
    let dy = s * k * n / h;
    [[0.0, y], [k * c, y + dy], [1.0 - k * c, y + dy], [1.0, y]]
}

/// Arc: every row is a circular arc of `|b|·π` around one centre; the edge on the inside of the
/// bend keeps its chord, and the rows' radii grow by the box height towards the other edge.
/// Worked in units of the box width (`h` = height / width) so the arcs are circular in pixels.
fn arc_grid(b: f64, h: f64) -> Grid {
    if b.abs() < 1e-9 {
        return rest_grid();
    }
    let up = b > 0.0;
    let th = b.abs() * PI;
    let r = 0.5 / (th / 2.0).sin();
    let k = 4.0 / 3.0 * (th / 4.0).tan();
    let s = if up { -1.0 } else { 1.0 };
    let cy = if up { h + r * (th / 2.0).cos() } else { -r * (th / 2.0).cos() };
    std::array::from_fn(|j| {
        let v = j as f64 / 3.0;
        let big = r + h * if up { 1.0 - v } else { v };
        let at = |a: f64| [0.5 + big * a.sin(), cy + s * big * a.cos()];
        let tangent = |a: f64| [a.cos(), -s * a.sin()];
        let (p0, p3, t0, t1) = (at(-th / 2.0), at(th / 2.0), tangent(-th / 2.0), tangent(th / 2.0));
        let l = k * big;
        let pts = [p0, [p0[0] + l * t0[0], p0[1] + l * t0[1]], [p3[0] - l * t1[0], p3[1] - l * t1[1]], p3];
        pts.map(|p| [p[0], p[1] / h])
    })
}

/// Rows blended from `top` to `bottom`.
fn blend_rows(top: [[f64; 2]; 4], bottom: [[f64; 2]; 4]) -> Grid {
    std::array::from_fn(|j| {
        let t = j as f64 / 3.0;
        std::array::from_fn(|i| [top[i][0] + (bottom[i][0] - top[i][0]) * t, top[i][1] + (bottom[i][1] - top[i][1]) * t])
    })
}

/// Photoshop's warp presets as Bezier control grids, measured on Photoshop 27.11: every preset at
/// bend ±50 and ±100, Arc and Bulge with ±50 horizontal and vertical distortion, and the vertical
/// orientation, each fitted from scripted renders of a coordinate-coded grid image (every preset
/// is a single bicubic patch, within 0.5 px). These grids put Photoshop's pixels back within
/// 1 px (median) on all 70 cases, and on 32 more on 256 × 96 and 96 × 256 boxes, vertical, and
/// with both distortions: the arc family stays circular in pixels, the other offsets scale with
/// the box, and a vertical warp is the horizontal one with the axes (and distortions) swapped.
/// `b`, `hd`, `vd` are -1..1; `aspect` is the box's height over its width.
fn preset_grid(style: WarpStyle, b: f64, hd: f64, vd: f64, vertical: bool, aspect: f64) -> Grid {
    // Height over width of the box the bend runs along (the transposed box when vertical); in
    // that frame the two distortions trade places too.
    let h = if vertical { 1.0 / aspect } else { aspect };
    let (hd, vd) = if vertical { (vd, hd) } else { (hd, vd) };
    let mut g = rest_grid();
    match style {
        WarpStyle::Arc => g = arc_grid(b, h),
        WarpStyle::Arch => g = std::array::from_fn(|j| edge_arc(j as f64 / 3.0, b, true, h)),
        WarpStyle::Bulge => g = blend_rows(edge_arc(0.0, b, true, h), edge_arc(1.0, b, false, h)),
        WarpStyle::ArcLower => g = blend_rows(g[0], edge_arc(1.0, b, false, h)),
        WarpStyle::ArcUpper => g = blend_rows(edge_arc(0.0, b, true, h), g[3]),
        // The bulging edge and the outer points of the row next to it come from Arc (bent the
        // other way for Shell Lower); the rest stays put.
        WarpStyle::ShellLower => {
            let a = arc_grid(-b, h);
            g[3] = a[3];
            g[2][0] = a[2][0];
            g[2][3] = a[2][3];
        }
        WarpStyle::ShellUpper => {
            let a = arc_grid(b, h);
            g[0] = a[0];
            g[1][0] = a[1][0];
            g[1][3] = a[1][3];
        }
        WarpStyle::Flag => {
            for row in &mut g {
                row[1][1] -= 2.0 * b;
                row[2][1] += 2.0 * b;
            }
        }
        WarpStyle::Wave => {
            for row in &mut g[1..3] {
                row[1][1] += 4.0 / 3.0 * b;
                row[2][1] -= 4.0 / 3.0 * b;
            }
        }
        WarpStyle::Fish => {
            for (row, f) in g.iter_mut().zip([-2.0, -2.0 / 3.0, 2.0 / 3.0, 2.0]) {
                row[1][1] += f * b;
                row[2][1] -= f * b;
            }
        }
        WarpStyle::Rise => {
            for row in &mut g {
                row[0][1] += 2.0 * b;
                row[1][1] += 2.0 * b;
            }
        }
        WarpStyle::Fisheye => {
            for (j, i, dx, dy) in [(1, 1, -1.0, -1.0), (1, 2, 1.0, -1.0), (2, 1, -1.0, 1.0), (2, 2, 1.0, 1.0)] {
                g[j][i][0] += dx * 2.0 / 3.0 * b;
                g[j][i][1] += dy * 2.0 / 3.0 * b;
            }
        }
        // Edge midpoints move out (Inflate) or the sides in (Squeeze) by b/3, the inner points by b/9.
        WarpStyle::Inflate | WarpStyle::Squeeze => {
            let s = if style == WarpStyle::Inflate { 1.0 } else { -1.0 };
            for p in &mut g[0][1..3] {
                p[1] -= b / 3.0;
            }
            for p in &mut g[3][1..3] {
                p[1] += b / 3.0;
            }
            for row in &mut g[1..3] {
                row[0][0] -= s * b / 3.0;
                row[3][0] += s * b / 3.0;
            }
            for (j, i, dx, dy) in [(1, 1, -1.0, -1.0), (1, 2, 1.0, -1.0), (2, 1, -1.0, 1.0), (2, 2, 1.0, 1.0)] {
                g[j][i][0] += s * dx * b / 9.0;
                g[j][i][1] += dy * b / 9.0;
            }
        }
        // The inner points turn round the centre (the other way for a negative bend).
        WarpStyle::Twist => {
            let a = b.abs();
            if b >= 0.0 {
                g[1][1][0] += a;
                g[1][2][1] += a;
                g[2][1][1] -= a;
                g[2][2][0] -= a;
            } else {
                g[1][1][1] += a;
                g[1][2][0] -= a;
                g[2][1][0] += a;
                g[2][2][1] -= a;
            }
        }
        WarpStyle::None | WarpStyle::Custom => {}
    }
    // Vertical distortion: the top and bottom rows scale about their end points' midpoint (by
    // 1 ∓ vd) and the middle rows follow by the blend of those two moves.
    if vd != 0.0 {
        let moved = |row: [[f64; 2]; 4], s: f64| -> [[f64; 2]; 4] {
            let m = [(row[0][0] + row[3][0]) / 2.0, (row[0][1] + row[3][1]) / 2.0];
            std::array::from_fn(|i| [m[0] + (row[i][0] - m[0]) * s - row[i][0], m[1] + (row[i][1] - m[1]) * s - row[i][1]])
        };
        let (d0, d3) = (moved(g[0], 1.0 - vd), moved(g[3], 1.0 + vd));
        for (j, row) in g.iter_mut().enumerate() {
            let t = j as f64 / 3.0;
            for (i, p) in row.iter_mut().enumerate() {
                p[0] += d0[i][0] + (d3[i][0] - d0[i][0]) * t;
                p[1] += d0[i][1] + (d3[i][1] - d0[i][1]) * t;
            }
        }
    }
    // Horizontal distortion: each column scales about its end points' midpoint by 1 + hd·(2i/3 − 1).
    if hd != 0.0 {
        for i in 0..4 {
            let s = 1.0 + hd * (2.0 * i as f64 / 3.0 - 1.0);
            let m = [(g[0][i][0] + g[3][i][0]) / 2.0, (g[0][i][1] + g[3][i][1]) / 2.0];
            for row in &mut g {
                row[i] = [m[0] + (row[i][0] - m[0]) * s, m[1] + (row[i][1] - m[1]) * s];
            }
        }
    }
    // Vertical orientation: the same grid with the axes swapped.
    if vertical {
        g = std::array::from_fn(|j| std::array::from_fn(|i| [g[i][j][1], g[i][j][0]]));
    }
    g
}

fn bernstein(t: f64) -> [f64; 4] {
    let s = 1.0 - t;
    [s * s * s, 3.0 * s * s * t, 3.0 * s * t * t, t * t * t]
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// De Casteljau split of a cubic at `t`: 7 points (left 0..=3, right 3..=6).
fn split_cubic(p: [[f64; 2]; 4], t: f64) -> [[f64; 2]; 7] {
    let a = lerp(p[0], p[1], t);
    let b = lerp(p[1], p[2], t);
    let c = lerp(p[2], p[3], t);
    let d = lerp(a, b, t);
    let e = lerp(b, c, t);
    let f = lerp(d, e, t);
    [p[0], a, d, f, e, c, p[3]]
}

/// Inverse of the Bernstein interpolation matrix at t = 0, 1/3, 2/3, 1 (rows: control point,
/// columns: sample). Exact rational values.
const BERN_INV: [[f64; 4]; 4] = [[1.0, 0.0, 0.0, 0.0], [-5.0 / 6.0, 3.0, -1.5, 1.0 / 3.0], [1.0 / 3.0, -1.5, 3.0, -5.0 / 6.0], [0.0, 0.0, 0.0, 1.0]];

/// A grid of bicubic Bezier patches over the unit square of a source box.
///
/// `us`/`vs` are the patch boundaries in the box's normalised coordinates (`0..=1`, increasing,
/// first 0 and last 1). Control points are a `(3·(us.len()-1)+1) × (3·(vs.len()-1)+1)` grid in
/// row-major order, in output (document) coordinates; neighbouring patches share their edge row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BezierMesh {
    pub us: Vec<f64>,
    pub vs: Vec<f64>,
    pub points: Vec<[f64; 2]>,
}

impl BezierMesh {
    /// The identity mesh over `bounds` with `cols × rows` patches (Photoshop's default is 1 × 1,
    /// a 4 × 4 control grid).
    pub fn identity(bounds: [f64; 4], cols: usize, rows: usize) -> BezierMesh {
        let (cols, rows) = (cols.clamp(1, 64), rows.clamp(1, 64));
        let us: Vec<f64> = (0..=cols).map(|i| i as f64 / cols as f64).collect();
        let vs: Vec<f64> = (0..=rows).map(|i| i as f64 / rows as f64).collect();
        Self::fit(&|s, t| [bounds[0] + s * (bounds[2] - bounds[0]), bounds[1] + t * (bounds[3] - bounds[1])], us, vs)
    }

    /// Fits a mesh with patch boundaries `us`/`vs` to the map `f` (normalised → output) by
    /// interpolating it at the patch's thirds. Exact for bicubic maps (affine maps included).
    pub fn fit(f: &dyn Fn(f64, f64) -> [f64; 2], us: Vec<f64>, vs: Vec<f64>) -> BezierMesh {
        let (pc, pr) = (us.len() - 1, vs.len() - 1);
        let (nx, ny) = (3 * pc + 1, 3 * pr + 1);
        let mut points = vec![[0.0; 2]; nx * ny];
        for pj in 0..pr {
            for pi in 0..pc {
                let mut s = [[[0.0; 2]; 4]; 4]; // s[row j][col i]
                for (j, row) in s.iter_mut().enumerate() {
                    for (i, v) in row.iter_mut().enumerate() {
                        let u = us[pi] + (us[pi + 1] - us[pi]) * i as f64 / 3.0;
                        let t = vs[pj] + (vs[pj + 1] - vs[pj]) * j as f64 / 3.0;
                        *v = f(u, t);
                    }
                }
                for j in 0..4 {
                    for i in 0..4 {
                        let mut acc = [0.0; 2];
                        for (a, wa) in BERN_INV[j].iter().enumerate() {
                            for (b, wb) in BERN_INV[i].iter().enumerate() {
                                let k = wa * wb;
                                acc[0] += k * s[a][b][0];
                                acc[1] += k * s[a][b][1];
                            }
                        }
                        points[(3 * pj + j) * nx + 3 * pi + i] = acc;
                    }
                }
            }
        }
        BezierMesh { us, vs, points }
    }

    /// Control points per row.
    pub fn nx(&self) -> usize {
        3 * (self.us.len() - 1) + 1
    }
    /// Control point rows.
    pub fn ny(&self) -> usize {
        3 * (self.vs.len() - 1) + 1
    }
    pub fn point(&self, i: usize, j: usize) -> [f64; 2] {
        self.points[j * self.nx() + i]
    }

    /// Checks the invariants (knots and point count); malformed data from files is rejected.
    pub fn is_valid(&self) -> bool {
        let knots_ok = |k: &[f64]| k.len() >= 2 && k[0] == 0.0 && k[k.len() - 1] == 1.0 && k.windows(2).all(|w| w[1] > w[0]);
        knots_ok(&self.us)
            && knots_ok(&self.vs)
            && self.points.len() == self.nx() * self.ny()
            && self.points.iter().all(|p| p[0].is_finite() && p[1].is_finite())
    }

    fn locate(knots: &[f64], s: f64) -> (usize, f64) {
        let n = knots.len() - 1;
        let s = s.clamp(0.0, 1.0);
        let mut i = 0;
        while i + 1 < n && s > knots[i + 1] {
            i += 1;
        }
        let w = knots[i + 1] - knots[i];
        (i, if w > 0.0 { (s - knots[i]) / w } else { 0.0 })
    }

    /// Evaluates the surface at normalised `(s, t)`.
    pub fn eval(&self, s: f64, t: f64) -> [f64; 2] {
        let (pi, a) = Self::locate(&self.us, s);
        let (pj, b) = Self::locate(&self.vs, t);
        let (ba, bb) = (bernstein(a), bernstein(b));
        let nx = self.nx();
        let mut out = [0.0; 2];
        for (j, wb) in bb.iter().enumerate() {
            for (i, wa) in ba.iter().enumerate() {
                let p = self.points[(3 * pj + j) * nx + 3 * pi + i];
                let k = wa * wb;
                out[0] += p[0] * k;
                out[1] += p[1] * k;
            }
        }
        out
    }

    /// Splits the patch column containing `s` at `s` (a vertical split line). Exact: the surface
    /// is unchanged. Returns false when `s` is on or too close to an existing boundary.
    pub fn split_u(&mut self, s: f64) -> bool {
        let (pi, a) = Self::locate(&self.us, s);
        if !(0.01..=0.99).contains(&a) {
            return false;
        }
        let (nx, ny) = (self.nx(), self.ny());
        let mut pts = Vec::with_capacity((nx + 3) * ny);
        for j in 0..ny {
            let row = &self.points[j * nx..(j + 1) * nx];
            pts.extend_from_slice(&row[..3 * pi]);
            let seg = [row[3 * pi], row[3 * pi + 1], row[3 * pi + 2], row[3 * pi + 3]];
            pts.extend_from_slice(&split_cubic(seg, a));
            pts.extend_from_slice(&row[3 * pi + 4..]);
        }
        self.points = pts;
        self.us.insert(pi + 1, s.clamp(0.0, 1.0));
        true
    }

    /// Splits the patch row containing `t` at `t` (a horizontal split line). Exact.
    pub fn split_v(&mut self, t: f64) -> bool {
        self.transpose();
        let ok = self.split_u(t);
        self.transpose();
        ok
    }

    /// Removes interior vertical boundary `k` (1..us.len()-1), merging its two patch columns.
    /// Exact when the patches came from an unedited split; otherwise the outer control points are
    /// kept and the inner handles extrapolated.
    pub fn remove_split_u(&mut self, k: usize) -> bool {
        if k == 0 || k + 1 >= self.us.len() {
            return false;
        }
        let a = (self.us[k] - self.us[k - 1]) / (self.us[k + 1] - self.us[k - 1]);
        let (nx, ny) = (self.nx(), self.ny());
        let mut pts = Vec::with_capacity((nx - 3) * ny);
        let c0 = 3 * (k - 1);
        for j in 0..ny {
            let row = &self.points[j * nx..(j + 1) * nx];
            let (p0, q1, r2, p3) = (row[c0], row[c0 + 1], row[c0 + 5], row[c0 + 6]);
            // De Casteljau inverse: q1 = lerp(p0, p1, a), r2 = lerp(p2, p3, a).
            let p1 = [(q1[0] - (1.0 - a) * p0[0]) / a, (q1[1] - (1.0 - a) * p0[1]) / a];
            let p2 = [(r2[0] - a * p3[0]) / (1.0 - a), (r2[1] - a * p3[1]) / (1.0 - a)];
            pts.extend_from_slice(&row[..c0]);
            pts.extend_from_slice(&[p0, p1, p2, p3]);
            pts.extend_from_slice(&row[c0 + 7..]);
        }
        self.points = pts;
        self.us.remove(k);
        true
    }

    /// Removes interior horizontal boundary `k`.
    pub fn remove_split_v(&mut self, k: usize) -> bool {
        self.transpose();
        let ok = self.remove_split_u(k);
        self.transpose();
        ok
    }

    fn transpose(&mut self) {
        let (nx, ny) = (self.nx(), self.ny());
        let mut pts = vec![[0.0; 2]; nx * ny];
        for j in 0..ny {
            for i in 0..nx {
                pts[i * ny + j] = self.points[j * nx + i];
            }
        }
        self.points = pts;
        std::mem::swap(&mut self.us, &mut self.vs);
    }

    /// Applies `f` to every control point (exact for affine maps).
    pub fn map_points(&self, f: impl Fn([f64; 2]) -> [f64; 2]) -> BezierMesh {
        BezierMesh { us: self.us.clone(), vs: self.vs.clone(), points: self.points.iter().map(|p| f(*p)).collect() }
    }

    /// Normalised `(s, t)` whose surface point is nearest `p` (coarse search, then Newton
    /// steps on the patch). Used to place split lines where the user clicks.
    pub fn param_at(&self, p: [f64; 2]) -> (f64, f64) {
        let n = 48;
        let mut best = (0.5, 0.5, f64::MAX);
        for j in 0..=n {
            for i in 0..=n {
                let (s, t) = (i as f64 / n as f64, j as f64 / n as f64);
                let q = self.eval(s, t);
                let d = (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2);
                if d < best.2 {
                    best = (s, t, d);
                }
            }
        }
        let (mut s, mut t) = (best.0, best.1);
        for _ in 0..12 {
            let h = 1e-5;
            let q = self.eval(s, t);
            let (qs, qt) = (self.eval(s + h, t), self.eval(s, t + h));
            let (a, b, c, d) = ((qs[0] - q[0]) / h, (qt[0] - q[0]) / h, (qs[1] - q[1]) / h, (qt[1] - q[1]) / h);
            let det = a * d - b * c;
            if det.abs() < 1e-12 {
                break;
            }
            let (ex, ey) = (p[0] - q[0], p[1] - q[1]);
            s = (s + (d * ex - b * ey) / det).clamp(0.0, 1.0);
            t = (t + (a * ey - c * ex) / det).clamp(0.0, 1.0);
        }
        (s, t)
    }

    /// Pulls the surface point at normalised `(s, t)` by `d`, as dragging the grid anywhere inside
    /// a patch does in other editors. `base` is the control grid at the start of the drag.
    ///
    /// Two parts. The surrounding grid follows loosely: every interior anchor within about two cells
    /// moves by a share of `d` that fades with distance. The picture's outer border stays pinned
    /// (the picture keeps filling its frame) unless the grab is within a
    /// few percent of it, and then it follows more and more. Each anchor
    /// takes its whole 3 x 3 block of control points along, so every grid line stays smooth through
    /// its anchors and the neighbouring cells and dividers bend with the pull. Then the four inner
    /// control points of the grabbed patch make up the rest, so the grabbed point lands exactly at
    /// its start + `d`. Grabbing near an anchor moves that anchor more, and the inner points less.
    pub fn pull(&mut self, base: &[[f64; 2]], s: f64, t: f64, d: [f64; 2]) {
        /// Within this distance of the mesh border (in 0..1 box units) a grab starts to drag the
        /// border with it; further in, the border stays exactly where it is.
        const BORDER_ZONE: f64 = 0.12;
        /// Falloff radius, in the grabbed patch's size.
        const REACH: f64 = 1.8;
        if base.len() != self.points.len() || !d.iter().all(|v| v.is_finite()) {
            return;
        }
        self.points.copy_from_slice(base);
        let (cols, rows) = (self.us.len() - 1, self.vs.len() - 1);
        let (pi, a) = Self::locate(&self.us, s);
        let (pj, b) = Self::locate(&self.vs, t);
        let (ba, bb) = (bernstein(a), bernstein(b));
        let nx = self.nx();
        let anchor = |ai: usize, aj: usize| base.get(3 * aj * nx + 3 * ai).copied().unwrap_or([0.0; 2]);
        let at0 = self.eval(s, t);
        let target = [at0[0] + d[0], at0[1] + d[1]];
        // The grabbed patch's size and its "interior-ness": 1 in the middle, 0 on its edges.
        let mut size = 1.0f64;
        for k in 0..2 {
            let vals = [anchor(pi, pj)[k], anchor(pi + 1, pj)[k], anchor(pi, pj + 1)[k], anchor(pi + 1, pj + 1)[k]];
            let (lo, hi) = vals.iter().fold((f64::MAX, f64::MIN), |m, v| (m.0.min(*v), m.1.max(*v)));
            size = size.max(hi - lo);
        }
        let radius = REACH * size;
        let inner_w = (ba[1] + ba[2]) * (bb[1] + bb[2]);
        let interior = (inner_w / 0.5625).clamp(0.0, 1.0);
        let to_border = s.min(1.0 - s).min(t.min(1.0 - t));
        let near_border = (1.0 - to_border / BORDER_ZONE).clamp(0.0, 1.0);
        // How much of the pull each corner of the grabbed patch is responsible for.
        // A corner on the picture's border only takes part once the grab is near that border.
        // The picture's four corners never move in a pull (a grab near the border bends the
        // border through its handles instead).
        let mesh_corner = |ai: usize, aj: usize| (ai == 0 || ai == cols) && (aj == 0 || aj == rows);
        let reach = |ai: usize, aj: usize, wa: f64, wb: f64| {
            let border = ai == 0 || ai == cols || aj == 0 || aj == rows;
            if mesh_corner(ai, aj) { 0.0 } else { wa * wb * if border { near_border } else { 1.0 } }
        };
        let corner = [
            (pi, pj, reach(pi, pj, ba[0] + ba[1], bb[0] + bb[1])),
            (pi + 1, pj, reach(pi + 1, pj, ba[2] + ba[3], bb[0] + bb[1])),
            (pi, pj + 1, reach(pi, pj + 1, ba[0] + ba[1], bb[2] + bb[3])),
            (pi + 1, pj + 1, reach(pi + 1, pj + 1, ba[2] + ba[3], bb[2] + bb[3])),
        ];
        // Share of `d` for each anchor.
        let mut share = vec![0.0f64; (cols + 1) * (rows + 1)];
        for aj in 0..=rows {
            for ai in 0..=cols {
                let p = anchor(ai, aj);
                let x = ((p[0] - at0[0]).hypot(p[1] - at0[1]) / radius).clamp(0.0, 1.0);
                let fall = 0.5 * (1.0 + (std::f64::consts::PI * x).cos());
                let border = ai == 0 || ai == cols || aj == 0 || aj == rows;
                let edge = if mesh_corner(ai, aj) {
                    0.0
                } else if border {
                    near_border
                } else {
                    1.0
                };
                share[aj * (cols + 1) + ai] = fall * edge;
            }
        }
        // The grabbed patch's own corners: near one, that corner follows the pointer.
        for (ai, aj, w) in corner {
            let f = &mut share[aj * (cols + 1) + ai];
            *f = f.max((1.0 - interior) * w);
        }
        // Every control point moves with the anchor it is closest to.
        for j in 0..self.ny() {
            for i in 0..nx {
                let (ai, aj) = (((i + 1) / 3).min(cols), ((j + 1) / 3).min(rows));
                let k = share[aj * (cols + 1) + ai];
                let p = &mut self.points[j * nx + i];
                p[0] += d[0] * k;
                p[1] += d[1] * k;
            }
        }
        // Close the gap exactly, with three groups of control points that share the work:
        // - the four inner points of the grabbed patch (all of it in the middle of the patch,
        //   little near a grid line, where they barely reach the surface);
        // - the corner blocks (an anchor with its handles, moving as a whole so lines stay smooth);
        // - when the grab is near the picture's border, that border's own handles: the border
        //   bends into a curve and its corners stay put.
        // Closing nudges the tangents across the grid lines, so they are evened out again and the
        // gap re-closed.
        let norm: f64 = [1, 2].iter().flat_map(|j| [1, 2].iter().map(move |i| (ba[*i] * bb[*j]).powi(2))).sum();
        let corner_norm: f64 = corner.iter().map(|c| c.2 * c.2).sum();
        // With no free corner to help (a one-patch grid, grabbed away from its border) the four
        // inner points do all of it.
        let inner_part = if corner_norm > 1e-12 { (inner_w / 0.15).min(1.0) } else { 1.0 };
        // Handles along the sides of the grabbed patch that lie on the picture's border.
        let mut edge: Vec<(usize, f64)> = Vec::new();
        for k in [1usize, 2] {
            if pi == 0 {
                edge.push(((3 * pj + k) * nx + 3 * pi, ba[0] * bb[k]));
            }
            if pi + 1 == cols {
                edge.push(((3 * pj + k) * nx + 3 * pi + 3, ba[3] * bb[k]));
            }
            if pj == 0 {
                edge.push((3 * pj * nx + 3 * pi + k, ba[k] * bb[0]));
            }
            if pj + 1 == rows {
                edge.push(((3 * pj + 3) * nx + 3 * pi + k, ba[k] * bb[3]));
            }
        }
        let edge_norm: f64 = edge.iter().map(|e| e.1 * e.1).sum();
        let edge_part = if edge_norm > 1e-12 { near_border } else { 0.0 };
        let (inner_share, corner_share) = ((1.0 - edge_part) * inner_part, (1.0 - edge_part) * (1.0 - inner_part));
        for _ in 0..60 {
            let now = self.eval(s, t);
            let rem = [target[0] - now[0], target[1] - now[1]];
            if rem[0].hypot(rem[1]) < 1e-9 {
                break;
            }
            if inner_share > 0.0 && norm > 0.0 {
                for j in [1usize, 2] {
                    for i in [1usize, 2] {
                        let k = ba[i] * bb[j] / norm * inner_share;
                        let p = &mut self.points[(3 * pj + j) * nx + 3 * pi + i];
                        p[0] += rem[0] * k;
                        p[1] += rem[1] * k;
                    }
                }
            }
            if edge_part > 0.0 {
                for (idx, w) in &edge {
                    if let Some(p) = self.points.get_mut(*idx) {
                        let k = w / edge_norm * edge_part;
                        p[0] += rem[0] * k;
                        p[1] += rem[1] * k;
                    }
                }
            }
            if corner_share > 0.0 && corner_norm > 0.0 {
                for (ai, aj, w) in corner {
                    let k = w / corner_norm * corner_share;
                    for dj in -1..=1isize {
                        for di in -1..=1isize {
                            let (i, j) = ((3 * ai) as isize + di, (3 * aj) as isize + dj);
                            if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < self.ny() {
                                let p = &mut self.points[j as usize * nx + i as usize];
                                p[0] += rem[0] * k;
                                p[1] += rem[1] * k;
                            }
                        }
                    }
                }
            }
            for _ in 0..3 {
                self.smooth_lines((3 * pi).saturating_sub(3), 3 * pi + 7, (3 * pj).saturating_sub(3), 3 * pj + 7);
            }
        }
    }

    /// Makes the tangent equal on both sides of every interior grid line, for the control points
    /// in columns `i0..i1` and rows `j0..j1` (exclusive ends, clamped to the grid): each pair of
    /// handles takes the average of the two tangents it had.
    fn smooth_lines(&mut self, i0: usize, i1: usize, j0: usize, j1: usize) {
        let nx = self.nx();
        let (i1, j1) = (i1.min(nx), j1.min(self.ny()));
        let hu = |k: usize| self.us[k + 1] - self.us[k];
        let hv = |k: usize| self.vs[k + 1] - self.vs[k];
        // (left, centre, right, span left, span right)
        let mut triples = Vec::new();
        for a in 1..self.us.len() - 1 {
            if 3 * a >= i0 && 3 * a < i1 {
                for j in j0..j1 {
                    triples.push((j * nx + 3 * a - 1, j * nx + 3 * a, j * nx + 3 * a + 1, hu(a - 1), hu(a)));
                }
            }
        }
        for b in 1..self.vs.len() - 1 {
            if 3 * b >= j0 && 3 * b < j1 {
                for i in i0..i1 {
                    triples.push(((3 * b - 1) * nx + i, 3 * b * nx + i, (3 * b + 1) * nx + i, hv(b - 1), hv(b)));
                }
            }
        }
        for (l, c, r, hl, hr) in triples {
            let (pl, pc, pr) = (self.points[l], self.points[c], self.points[r]);
            let tan = [((pc[0] - pl[0]) / hl + (pr[0] - pc[0]) / hr) / 2.0, ((pc[1] - pl[1]) / hl + (pr[1] - pc[1]) / hr) / 2.0];
            self.points[l] = [pc[0] - tan[0] * hl, pc[1] - tan[1] * hl];
            self.points[r] = [pc[0] + tan[0] * hr, pc[1] + tan[1] * hr];
        }
    }

    /// Drags control point `index` by `d` from the grid `base` it had when the drag began, keeping
    /// the surface smooth across the grid lines:
    /// - an anchor moves with the eight points around it, as one block;
    /// - a handle moves with the two points beside it across its line, and the handle opposite
    ///   it on the same anchor turns the other way (with its two neighbours), so the line keeps
    ///   passing straight through the anchor instead of creasing there.
    pub fn drag_point(&mut self, base: &[[f64; 2]], index: usize, d: [f64; 2]) {
        if base.len() != self.points.len() || index >= base.len() || !d.iter().all(|v| v.is_finite()) {
            return;
        }
        self.points.copy_from_slice(base);
        let (nx, ny) = (self.nx(), self.ny());
        let (i, j) = (index % nx, index / nx);
        let mut shift = |i: isize, j: isize, d: [f64; 2]| {
            if i >= 0 && j >= 0 && (i as usize) < nx && (j as usize) < ny {
                let p = &mut self.points[j as usize * nx + i as usize];
                p[0] += d[0];
                p[1] += d[1];
            }
        };
        let (ii, jj) = (i as isize, j as isize);
        // Spans either side of interior knot `k`.
        let span = |knots: &[f64], k: usize| (k > 0 && k + 1 < knots.len()).then(|| (knots[k] - knots[k - 1], knots[k + 1] - knots[k]));
        match (i % 3, j % 3) {
            (0, 0) => {
                for dj in -1..=1 {
                    for di in -1..=1 {
                        shift(ii + di, jj + dj, d);
                    }
                }
            }
            // A handle along a vertical line: it takes the two points beside it across the line.
            (0, r) => {
                for di in -1..=1 {
                    shift(ii + di, jj, d);
                }
                // The opposite handle of the same anchor turns the other way.
                if let Some((hl, hr)) = span(&self.vs, j / 3 + usize::from(r == 2)) {
                    let (other, k) = if r == 1 { (jj - 2, hl / hr) } else { (jj + 2, hr / hl) };
                    for di in -1..=1 {
                        shift(ii + di, other, [-d[0] * k, -d[1] * k]);
                    }
                }
            }
            // A handle along a horizontal line, likewise.
            (r, 0) => {
                for dj in -1..=1 {
                    shift(ii, jj + dj, d);
                }
                if let Some((hl, hr)) = span(&self.us, i / 3 + usize::from(r == 2)) {
                    let (other, k) = if r == 1 { (ii - 2, hl / hr) } else { (ii + 2, hr / hl) };
                    for dj in -1..=1 {
                        shift(other, jj + dj, [-d[0] * k, -d[1] * k]);
                    }
                }
            }
            // An inner point of a patch has no handle of its own.
            _ => shift(ii, jj, d),
        }
    }

    /// Bounding box of the control points (the surface lies inside it).
    pub fn control_bounds(&self) -> [f64; 4] {
        self.points.iter().fold([f64::MAX, f64::MAX, f64::MIN, f64::MIN], |b, p| [b[0].min(p[0]), b[1].min(p[1]), b[2].max(p[0]), b[3].max(p[1])])
    }
}

/// A complete warp: what Photoshop's `warp` descriptor stores, plus the custom mesh.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Warp {
    pub style: WarpStyle,
    /// Bend in percent (-100..100), presets only.
    #[serde(default)]
    pub bend: f64,
    #[serde(default)]
    pub h_distort: f64,
    #[serde(default)]
    pub v_distort: f64,
    /// Warp along the vertical axis.
    #[serde(default)]
    pub vertical: bool,
    /// The source box `[x0, y0, x1, y1]` the warp is defined over.
    pub bounds: [f64; 4],
    /// Control mesh for [`WarpStyle::Custom`] (output coordinates).
    #[serde(default)]
    pub mesh: Option<BezierMesh>,
}

impl Warp {
    /// No warp over `bounds`.
    pub fn none(bounds: [f64; 4]) -> Warp {
        Warp { style: WarpStyle::None, bend: 0.0, h_distort: 0.0, v_distort: 0.0, vertical: false, bounds, mesh: None }
    }

    /// A preset style.
    pub fn preset(style: WarpStyle, bend: f64, bounds: [f64; 4]) -> Warp {
        Warp { style, bend, ..Warp::none(bounds) }
    }

    /// A custom warp from a mesh.
    pub fn custom(mesh: BezierMesh, bounds: [f64; 4]) -> Warp {
        Warp { style: WarpStyle::Custom, mesh: Some(mesh), ..Warp::none(bounds) }
    }

    fn style_warp(&self) -> Option<StyleWarp> {
        StyleWarp::new(self.style, self.bend, self.h_distort, self.v_distort, self.vertical, self.bounds)
    }

    /// Whether the warp maps every point of its box onto itself (within `1e-9` px).
    pub fn is_identity(&self) -> bool {
        match self.style {
            WarpStyle::None => true,
            WarpStyle::Custom => match &self.mesh {
                None => true,
                Some(m) => {
                    let id = BezierMesh::fit(&|s, t| self.box_point(s, t), m.us.clone(), m.vs.clone());
                    m.points.iter().zip(&id.points).all(|(a, b)| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9)
                }
            },
            _ => self.style_warp().is_none(),
        }
    }

    fn box_point(&self, s: f64, t: f64) -> [f64; 2] {
        let b = self.bounds;
        [b[0] + s * (b[2] - b[0]), b[1] + t * (b[3] - b[1])]
    }

    /// Maps a source point (inside the box) to output space.
    pub fn map(&self, x: f64, y: f64) -> (f64, f64) {
        match self.style {
            WarpStyle::None => (x, y),
            WarpStyle::Custom => match &self.mesh {
                Some(m) => {
                    let b = self.bounds;
                    let (w, h) = (b[2] - b[0], b[3] - b[1]);
                    if w <= 0.0 || h <= 0.0 {
                        return (x, y);
                    }
                    let p = m.eval((x - b[0]) / w, (y - b[1]) / h);
                    (p[0], p[1])
                }
                None => (x, y),
            },
            _ => match self.style_warp() {
                Some(sw) => sw.apply(x, y),
                None => (x, y),
            },
        }
    }

    /// The warp as a control mesh (a preset is its own Bezier patch split into `cols × rows`;
    /// custom warps return their own mesh).
    pub fn to_mesh(&self, cols: usize, rows: usize) -> BezierMesh {
        if let (WarpStyle::Custom, Some(m)) = (self.style, &self.mesh) {
            return m.clone();
        }
        let (cols, rows) = (cols.clamp(1, 64), rows.clamp(1, 64));
        // A preset is one Bezier patch: split it exactly.
        if let Some(sw) = self.style_warp() {
            let mut m = sw.mesh();
            for i in 1..cols {
                m.split_u(i as f64 / cols as f64);
            }
            for j in 1..rows {
                m.split_v(j as f64 / rows as f64);
            }
            return m;
        }
        let us: Vec<f64> = (0..=cols).map(|i| i as f64 / cols as f64).collect();
        let vs: Vec<f64> = (0..=rows).map(|i| i as f64 / rows as f64).collect();
        BezierMesh::fit(
            &|s, t| {
                let p = self.box_point(s, t);
                let (x, y) = self.map(p[0], p[1]);
                [x, y]
            },
            us,
            vs,
        )
    }

    /// The same warp followed by the affine map `[a, b, c, d, e, f]` (`x' = a·x + c·y + e`,
    /// `y' = b·x + d·y + f`), as a custom mesh (exact for custom warps).
    pub fn then_affine(&self, m: [f64; 6]) -> Warp {
        let mesh = self.to_mesh(4, 4).map_points(|p| [m[0] * p[0] + m[2] * p[1] + m[4], m[1] * p[0] + m[3] * p[1] + m[5]]);
        Warp::custom(mesh, self.bounds)
    }

    /// Approximate output bounds (sampled; includes the control mesh for custom warps).
    pub fn output_bounds(&self) -> [f64; 4] {
        let n = 32;
        let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
        for j in 0..=n {
            for i in 0..=n {
                let p = self.box_point(i as f64 / n as f64, j as f64 / n as f64);
                let (x, y) = self.map(p[0], p[1]);
                b = [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)];
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: [f64; 4] = [10.0, 20.0, 110.0, 70.0];

    /// The largest mismatch between the tangents on the two sides of any interior grid line.
    fn tangent_defect(m: &BezierMesh) -> f64 {
        let nx = m.nx();
        let mut worst: f64 = 0.0;
        for a in 1..m.us.len() - 1 {
            let (hl, hr) = (m.us[a] - m.us[a - 1], m.us[a + 1] - m.us[a]);
            for j in 0..m.ny() {
                let (l, c, r) = (m.points[j * nx + 3 * a - 1], m.points[j * nx + 3 * a], m.points[j * nx + 3 * a + 1]);
                for k in 0..2 {
                    worst = worst.max(((c[k] - l[k]) / hl - (r[k] - c[k]) / hr).abs());
                }
            }
        }
        for b in 1..m.vs.len() - 1 {
            let (hl, hr) = (m.vs[b] - m.vs[b - 1], m.vs[b + 1] - m.vs[b]);
            for i in 0..nx {
                let (l, c, r) = (m.points[(3 * b - 1) * nx + i], m.points[3 * b * nx + i], m.points[(3 * b + 1) * nx + i]);
                for k in 0..2 {
                    worst = worst.max(((c[k] - l[k]) / hl - (r[k] - c[k]) / hr).abs());
                }
            }
        }
        worst
    }

    #[test]
    fn dragging_an_anchor_or_handle_keeps_the_surface_smooth_across_the_lines() {
        let mut base = BezierMesh::identity([0.0, 0.0, 300.0, 300.0], 1, 1);
        for s in [0.3, 0.55, 0.9] {
            assert!(base.split_u(s));
            assert!(base.split_v(s));
        }
        let nx = base.nx();
        // Every kind of point on a grid line, interior and border: anchors, handles each way.
        for (i, j) in [(6, 6), (7, 6), (5, 6), (6, 7), (6, 5), (3, 9), (9, 3), (0, 6), (6, 0), (4, 3), (3, 4), (12, 12)] {
            let mut m = base.clone();
            m.drag_point(&base.points, j * nx + i, [23.0, -31.0]);
            assert_eq!(m.points[j * nx + i], [base.points[j * nx + i][0] + 23.0, base.points[j * nx + i][1] - 31.0], "({i}, {j}) lands on the pointer");
            assert!(tangent_defect(&m) < 1e-6, "({i}, {j}): {}", tangent_defect(&m));
        }
    }

    /// Fraction of the surface where the map folds over itself (its Jacobian is not positive).
    fn fold_fraction(m: &BezierMesh) -> f64 {
        let (n, h) = (80, 1e-4);
        let mut bad = 0;
        for j in 0..n {
            for i in 0..n {
                let (s, t) = ((i as f64 + 0.5) / n as f64, (j as f64 + 0.5) / n as f64);
                let q = m.eval(s, t);
                let (qs, qt) = (m.eval(s + h, t), m.eval(s, t + h));
                if (qs[0] - q[0]) * (qt[1] - q[1]) - (qt[0] - q[0]) * (qs[1] - q[1]) <= 0.0 {
                    bad += 1;
                }
            }
        }
        bad as f64 / (n * n) as f64
    }

    #[test]
    fn a_pull_within_a_quarter_cell_never_folds_the_surface() {
        // 75 px cells: dragging a quarter of a cell, any direction, anywhere in the grid must not tear the picture.
        let base = BezierMesh::identity([0.0, 0.0, 300.0, 300.0], 4, 4);
        for (s, t) in [(0.42, 0.42), (0.75, 0.2), (0.2, 0.2), (0.5, 0.5), (0.85, 0.7), (0.3, 0.7)] {
            for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.7, 0.7), (-0.7, 0.7)] {
                let mut m = base.clone();
                m.pull(&base.points, s, t, [dx * 19.0, dy * 19.0]);
                assert_eq!(fold_fraction(&m), 0.0, "({s}, {t}) by ({dx}, {dy}) quarter cells");
            }
        }
    }

    #[test]
    fn a_pull_in_the_middle_leaves_the_border_where_it_is() {
        let mut base = BezierMesh::identity([0.0, 0.0, 600.0, 800.0], 1, 1);
        for s in [0.33, 0.66] {
            assert!(base.split_u(s));
        }
        assert!(base.split_v(0.5));
        let (nx, ny) = (base.nx(), base.ny());
        let on_border = |i: usize, j: usize| i == 0 || j == 0 || i == nx - 1 || j == ny - 1;
        let mut m = base.clone();
        m.pull(&base.points, 0.5, 0.5, [-180.0, 40.0]);
        for j in 0..ny {
            for i in 0..nx {
                if on_border(i, j) {
                    let (a, b) = (m.points[j * nx + i], base.points[j * nx + i]);
                    assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6, "border point ({i}, {j}) moved");
                }
            }
        }
        // The pull itself still lands exactly.
        let (before, after) = (base.eval(0.5, 0.5), m.eval(0.5, 0.5));
        assert!((after[0] - before[0] + 180.0).abs() < 1e-4 && (after[1] - before[1] - 40.0).abs() < 1e-4);
    }

    #[test]
    fn a_pull_anywhere_away_from_the_border_never_moves_the_border() {
        // The default one-patch grid and a cut one; grabs all over, none within 12% of an edge.
        let one = BezierMesh::identity([0.0, 0.0, 600.0, 800.0], 1, 1);
        let mut cut = one.clone();
        assert!(cut.split_u(0.4) && cut.split_u(0.7) && cut.split_v(0.35) && cut.split_v(0.8));
        for base in [one, cut] {
            let (nx, ny) = (base.nx(), base.ny());
            for (s, t) in [(0.5, 0.5), (0.2, 0.5), (0.8, 0.3), (0.3, 0.85), (0.14, 0.14), (0.86, 0.5)] {
                for d in [[-400.0, 0.0], [250.0, -90.0], [0.0, 300.0]] {
                    let mut m = base.clone();
                    m.pull(&base.points, s, t, d);
                    for j in 0..ny {
                        for i in 0..nx {
                            if i == 0 || j == 0 || i == nx - 1 || j == ny - 1 {
                                let (a, b) = (m.points[j * nx + i], base.points[j * nx + i]);
                                assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6, "({s}, {t}) by {d:?} moved border point ({i}, {j})");
                            }
                        }
                    }
                    let (before, after) = (base.eval(s, t), m.eval(s, t));
                    assert!((after[0] - before[0] - d[0]).abs() < 1e-3 && (after[1] - before[1] - d[1]).abs() < 1e-3, "({s}, {t}) by {d:?}: {after:?}");
                }
            }
        }
    }

    #[test]
    fn pulling_on_the_border_bows_it_and_leaves_the_corners_put() {
        // A one-patch grid, grabbed on the middle of its right edge and pulled 170 px out.
        let base = BezierMesh::identity([0.0, 0.0, 800.0, 1200.0], 1, 1);
        let mut m = base.clone();
        m.pull(&base.points, 1.0, 0.5, [170.0, 0.0]);
        let at = m.eval(1.0, 0.5);
        assert!((at[0] - 970.0).abs() < 1e-3 && (at[1] - 600.0).abs() < 1e-3, "{at:?}");
        for (s, t) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let (a, b) = (m.eval(s, t), base.eval(s, t));
            assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6, "corner ({s}, {t}) moved");
        }
        // The other three edges did not move.
        for k in 0..=10 {
            let u = k as f64 / 10.0;
            for (s, t) in [(0.0, u), (u, 0.0), (u, 1.0)] {
                let (a, b) = (m.eval(s, t), base.eval(s, t));
                assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6, "edge point ({s}, {t}) moved");
            }
        }
        // Only the right edge's two handles carry the pull.
        let nx = base.nx();
        let moved: Vec<usize> = (0..m.points.len()).filter(|k| m.points[*k] != base.points[*k]).collect();
        assert_eq!(moved, vec![nx + 3, 2 * nx + 3], "{moved:?}");
        assert_eq!(fold_fraction(&m), 0.0);
    }

    #[test]
    fn pull_keeps_the_surface_smooth_across_the_grid_lines() {
        // A grid cut unevenly both ways, then pulled hard inside one cell and then another.
        let mut base = BezierMesh::identity([0.0, 0.0, 300.0, 300.0], 1, 1);
        for s in [0.3, 0.55, 0.9] {
            assert!(base.split_u(s));
            assert!(base.split_v(s));
        }
        assert!(tangent_defect(&base) < 1e-9, "cuts are exact");
        for (s, t, d) in [(0.42, 0.42, [60.0, -35.0]), (0.75, 0.2, [-40.0, 50.0]), (0.1, 0.95, [30.0, 30.0])] {
            let mut m = base.clone();
            m.pull(&base.points, s, t, d);
            assert!(tangent_defect(&m) < 1e-6, "({s}, {t}): {}", tangent_defect(&m));
            let before = base.eval(s, t);
            let after = m.eval(s, t);
            assert!((after[0] - before[0] - d[0]).abs() < 1e-4 && (after[1] - before[1] - d[1]).abs() < 1e-4, "{after:?}");
        }
    }

    #[test]
    fn pull_moves_the_grabbed_surface_point_exactly_and_stays_local() {
        let base = BezierMesh::identity(B, 3, 2);
        for (s, t) in [(0.5, 0.5), (0.1, 0.9), (0.4, 0.25), (0.0, 0.5), (1.0, 0.3)] {
            let mut m = base.clone();
            let before = base.eval(s, t);
            m.pull(&base.points, s, t, [7.0, -4.5]);
            let after = m.eval(s, t);
            assert!((after[0] - before[0] - 7.0).abs() < 1e-4 && (after[1] - before[1] + 4.5).abs() < 1e-4, "({s}, {t}): {after:?}");
        }
        // The pull stays near the grabbed patch: the far end of the grid does not move.
        let mut m = base.clone();
        m.pull(&base.points, 0.1, 0.25, [5.0, 5.0]);
        let nx = base.nx();
        let moved = |k: usize| (m.points[k][0] - base.points[k][0]).abs() > 1e-6 || (m.points[k][1] - base.points[k][1]).abs() > 1e-6;
        assert!((0..base.ny()).all(|j| !moved(j * nx + nx - 1)), "last column stays put");
        assert!((0..nx).any(moved), "something moved");
        // Pulling again from the same base replaces the pull (a drag is not cumulative).
        m.pull(&base.points, 0.5, 0.25, [0.0, 0.0]);
        assert!(m.points.iter().zip(&base.points).all(|(a, b)| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9));
    }

    fn close(a: (f64, f64), b: (f64, f64), eps: f64) -> bool {
        (a.0 - b.0).abs() < eps && (a.1 - b.1).abs() < eps
    }

    #[test]
    fn styles_parse_and_round_trip_names() {
        for s in WarpStyle::all() {
            assert_eq!(WarpStyle::parse(s.id()), Some(s));
            assert_eq!(WarpStyle::parse(s.psd_name()), Some(s));
            assert_eq!(WarpStyle::parse(s.label()), Some(s), "{s:?}");
        }
        assert_eq!(WarpStyle::parse("spiral"), None);
    }

    #[test]
    fn presets_at_bend_zero_are_identity() {
        for s in WarpStyle::PRESETS {
            let w = Warp::preset(s, 0.0, B);
            assert!(w.is_identity());
            assert_eq!(w.map(33.0, 44.0), (33.0, 44.0));
        }
    }

    #[test]
    fn identity_mesh_is_exact_and_fit_reproduces_affine() {
        let m = BezierMesh::identity(B, 1, 1);
        assert_eq!((m.nx(), m.ny()), (4, 4));
        let w = Warp::custom(m, B);
        assert!(w.is_identity());
        assert!(close(w.map(37.5, 61.25), (37.5, 61.25), 1e-9));
        // Affine maps fit exactly.
        let f = |s: f64, t: f64| [3.0 + 2.0 * s - t, 1.0 + 0.5 * s + 4.0 * t];
        let m = BezierMesh::fit(&f, vec![0.0, 1.0], vec![0.0, 0.4, 1.0]);
        for (s, t) in [(0.1, 0.9), (0.5, 0.2), (0.77, 0.41)] {
            let p = m.eval(s, t);
            let e = f(s, t);
            assert!((p[0] - e[0]).abs() < 1e-9 && (p[1] - e[1]).abs() < 1e-9);
        }
    }

    #[test]
    fn splits_are_exact_and_removable() {
        let mut m = BezierMesh::identity(B, 1, 1);
        // Bend it: move an inner control point.
        m.points[5] = [50.0, 10.0];
        let before: Vec<[f64; 2]> = (0..=10).flat_map(|j| (0..=10).map(move |i| (i, j))).map(|(i, j)| m.eval(i as f64 / 10.0, j as f64 / 10.0)).collect();
        let orig = m.clone();
        assert!(m.split_u(0.3));
        assert!(m.split_v(0.6));
        assert!(!m.split_u(0.3), "existing boundary");
        assert_eq!((m.nx(), m.ny()), (7, 7));
        assert!(m.is_valid());
        let after: Vec<[f64; 2]> = (0..=10).flat_map(|j| (0..=10).map(move |i| (i, j))).map(|(i, j)| m.eval(i as f64 / 10.0, j as f64 / 10.0)).collect();
        for (a, b) in before.iter().zip(&after) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
        let (ls, lt) = m.param_at(m.eval(0.42, 0.77));
        assert!((ls - 0.42).abs() < 1e-6 && (lt - 0.77).abs() < 1e-6, "{ls},{lt}");
        assert!(m.remove_split_u(1));
        assert!(m.remove_split_v(1));
        assert_eq!(m.us, orig.us);
        for (a, b) in m.points.iter().zip(&orig.points) {
            assert!((a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
        }
    }

    const SQUARE: [f64; 4] = [128.0, 128.0, 384.0, 384.0];
    const WIDE: [f64; 4] = [128.0, 208.0, 384.0, 304.0];
    const TALL: [f64; 4] = [208.0, 128.0, 304.0, 384.0];

    /// Photoshop 27.11, measured: a 256 px grid image whose colours encode each pixel's own
    /// position, warped in a 512 px document over the box (128, 128)–(384, 384). Each pair is a
    /// source point and where Photoshop put it.
    #[test]
    fn presets_match_photoshop() {
        type Case = (WarpStyle, f64, f64, f64, bool, [([f64; 2], [f64; 2]); 3]);
        let cases: [Case; 9] = [
            (WarpStyle::Arc, 50.0, 0.0, 0.0, false, [([272.5, 155.5], [296.5, 104.5]), ([331.5, 182.5], [424.5, 168.5]), ([259.5, 157.5], [264.5, 104.5])]),
            (WarpStyle::Arc, -100.0, 0.0, 0.0, false, [([277.5, 354.5], [344.5, 472.5]), ([140.5, 218.5], [40.5, 168.5]), ([172.5, 237.5], [56.5, 264.5])]),
            (WarpStyle::Flag, 50.0, 0.0, 0.0, false, [([136.5, 143.5], [136.5, 120.5]), ([136.5, 159.5], [136.5, 136.5]), ([136.5, 175.5], [136.5, 152.5])]),
            (WarpStyle::Twist, -50.0, 0.0, 0.0, false, [([148.5, 205.5], [152.5, 216.5]), ([197.5, 212.5], [200.5, 232.5]), ([134.5, 212.5], [136.5, 216.5])]),
            (
                WarpStyle::ShellLower,
                50.0,
                0.0,
                0.0,
                false,
                [([248.5, 136.5], [248.5, 136.5]), ([264.5, 136.5], [264.5, 136.5]), ([232.5, 136.5], [232.5, 136.5])],
            ),
            (WarpStyle::Fish, -100.0, 0.0, 0.0, false, [([296.5, 355.5], [296.5, 440.5]), ([344.5, 298.5], [344.5, 344.5]), ([328.5, 356.5], [328.5, 472.5])]),
            (WarpStyle::Bulge, 50.0, 50.0, 0.0, false, [([293.5, 147.5], [296.5, 88.5]), ([293.5, 375.5], [296.5, 440.5]), ([340.5, 281.5], [344.5, 296.5])]),
            (WarpStyle::Arc, 50.0, 0.0, -50.0, false, [([181.5, 350.5], [184.5, 328.5]), ([291.5, 372.5], [280.5, 344.5]), ([232.5, 349.5], [232.5, 312.5])]),
            (WarpStyle::Arc, 50.0, 0.0, 0.0, true, [([155.5, 272.5], [104.5, 296.5]), ([182.5, 331.5], [168.5, 424.5]), ([157.5, 259.5], [104.5, 264.5])]),
        ];
        // Wide (256 × 96 at 128, 208) and tall (96 × 256 at 208, 128) boxes, and a vertical warp
        // with distortion.
        type Boxed = (WarpStyle, f64, f64, f64, bool, [f64; 4], [([f64; 2], [f64; 2]); 2]);
        let boxed: [Boxed; 6] = [
            (WarpStyle::Arc, 50.0, 0.0, 0.0, false, WIDE, [([144.5, 269.6], [120.5, 264.5]), ([271.5, 220.42], [280.5, 168.5])]),
            (WarpStyle::Bulge, 50.0, 0.0, 0.0, false, TALL, [([249.11, 374.5], [248.5, 392.5]), ([249.11, 360.5], [248.5, 376.5])]),
            (WarpStyle::Flag, 50.0, 0.0, 0.0, true, WIDE, [([137.5, 296.42], [184.5, 296.5]), ([153.5, 296.42], [200.5, 296.5])]),
            (WarpStyle::Rise, 50.0, 0.0, 0.0, true, TALL, [([250.97, 152.5], [344.5, 152.5]), ([301.26, 264.5], [344.5, 264.5])]),
            (WarpStyle::Bulge, 60.0, 20.0, -10.0, false, WIDE, [([208.5, 224.15], [200.5, 184.5]), ([219.5, 277.42], [216.5, 296.5])]),
            (WarpStyle::Arc, 50.0, 30.0, 0.0, true, SQUARE, [([331.5, 327.5], [296.5, 360.5]), ([318.5, 240.5], [264.5, 232.5])]),
        ];
        let all =
            cases.iter().map(|c| (c.0, c.1, c.2, c.3, c.4, SQUARE, c.5.to_vec())).chain(boxed.iter().map(|c| (c.0, c.1, c.2, c.3, c.4, c.5, c.6.to_vec())));
        for (style, bend, hd, vd, vertical, bounds, pairs) in all {
            let w = Warp { h_distort: hd, v_distort: vd, vertical, ..Warp::preset(style, bend, bounds) };
            for (src, want) in pairs {
                let got = w.map(src[0], src[1]);
                let d = (got.0 - want[0]).hypot(got.1 - want[1]);
                assert!(d < 1.5, "{style:?} {bend} h{hd} v{vd} vertical {vertical} {bounds:?}: {src:?} -> {got:?}, Photoshop {want:?}");
            }
        }
    }

    #[test]
    fn preset_fits_closely_as_mesh_and_affine_composes() {
        let w = Warp::preset(WarpStyle::Bulge, 40.0, B);
        let m = Warp::custom(w.to_mesh(4, 4), B);
        for (x, y) in [(20.0, 30.0), (60.0, 45.0), (100.0, 65.0)] {
            assert!(close(w.map(x, y), m.map(x, y), 0.5), "{:?} vs {:?}", w.map(x, y), m.map(x, y));
        }
        let t = Warp::custom(BezierMesh::identity(B, 1, 1), B).then_affine([1.0, 0.0, 0.0, 1.0, 5.0, -2.0]);
        assert!(close(t.map(50.0, 50.0), (55.0, 48.0), 1e-9));
        let ob = Warp::none(B).output_bounds();
        assert!((ob[0] - 10.0).abs() < 1e-9 && (ob[3] - 70.0).abs() < 1e-9);
    }
}
