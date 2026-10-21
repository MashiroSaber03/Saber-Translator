//! Warp Text: bends the glyph outlines of a type layer with one of Photoshop's fifteen warp
//! styles (the PSD `warp` descriptor's `warpStyle`).
//!
//! The style math lives in `photocraft_geom::warp` (shared with Edit › Transform › Warp): our
//! own closed-form approximations of each style's look (observed behaviour, not Adobe's
//! formulas), the identity at `bend = 0` and continuous in `bend`. The mapping is evaluated per
//! point; the renderer subdivides long segments so straight edges bend.

use photocraft_doc::text::{TextShape, TextWarp};
use photocraft_doc::vector::Subpath;
use photocraft_geom::warp::{StyleWarp, WarpStyle};

/// Warp styles in Photoshop's menu order: (PSD `warpStyle` value, short id used by commands).
pub const STYLES: [(&str, &str); 15] = [
    ("warpArc", "arc"),
    ("warpArcLower", "arcLower"),
    ("warpArcUpper", "arcUpper"),
    ("warpArch", "arch"),
    ("warpBulge", "bulge"),
    ("warpShellLower", "shellLower"),
    ("warpShellUpper", "shellUpper"),
    ("warpFlag", "flag"),
    ("warpWave", "wave"),
    ("warpFish", "fish"),
    ("warpRise", "rise"),
    ("warpFisheye", "fisheye"),
    ("warpInflate", "inflate"),
    ("warpSqueeze", "squeeze"),
    ("warpTwist", "twist"),
];

/// PSD style name (`warpArc`) for a short id (`arc`, case-insensitive) or a PSD name.
pub fn psd_style(id: &str) -> Option<&'static str> {
    let l = id.to_ascii_lowercase();
    if l == "none" || l == "warpnone" {
        return Some("warpNone");
    }
    STYLES.iter().find(|(psd, short)| psd.to_ascii_lowercase() == l || short.to_ascii_lowercase() == l).map(|(psd, _)| *psd)
}

/// Short id (`flag`, `none`) for a PSD style name (`warpFlag`, `warpNone`): the reverse of [`psd_style`].
pub fn short_style(psd: &str) -> Option<&'static str> {
    if psd == "warpNone" {
        return Some("none");
    }
    STYLES.iter().find(|(p, _)| *p == psd).map(|(_, short)| *short)
}

/// Where a glyph goes when the layout is bent: see [`Warp::place_glyph`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Placement {
    /// Not path text: bend the outline point by point with [`Warp::apply`].
    Bend,
    /// Beyond either end of the path: the glyph is not drawn.
    Hidden,
    /// Path text: the glyph keeps its shape and is moved rigidly by this text-space map
    /// (`[a, b, c, d, e, f]`, applied to text-space points).
    Frame([f64; 6]),
}

/// A warp prepared for one layout: the shared style math of
/// [`photocraft_geom::warp::StyleWarp`] (also used by Edit › Transform › Warp) over the text box,
/// or a path the type runs along.
#[derive(Clone, Debug)]
pub struct Warp(Kind);

#[derive(Clone, Debug)]
enum Kind {
    Style(Box<StyleWarp>),
    Path(PathFrame),
}

impl Warp {
    /// Prepares `w` for text whose bounds are `[x0, y0, x1, y1]` (text space). `None` for
    /// `warpNone`, unknown styles, an all-zero warp or empty bounds.
    pub fn new(w: &TextWarp, bounds: [f32; 4]) -> Option<Warp> {
        let style = WarpStyle::parse(&w.style).filter(|s| s.is_preset() && s.psd_name() == w.style)?;
        StyleWarp::new(style, f64::from(w.value), f64::from(w.horizontal_distortion), f64::from(w.vertical_distortion), !w.horizontal, bounds.map(f64::from))
            .map(|s| Warp(Kind::Style(Box::new(s))))
    }

    /// The deformation of type on a path (`None` for any other shape, or a path without length).
    pub fn on_path(shape: &TextShape) -> Option<Warp> {
        let TextShape::Path { path, start, flip } = shape else { return None };
        PathFrame::new(path, *start, *flip).map(|f| Warp(Kind::Path(f)))
    }

    /// Longest segment (text-space px) worth sending through [`Warp::apply`] unsplit.
    pub fn max_segment(&self) -> f64 {
        match &self.0 {
            Kind::Style(s) => s.max_segment(),
            Kind::Path(_) => 2.0,
        }
    }

    /// Maps a text-space point.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        match &self.0 {
            Kind::Style(s) => s.apply(x, y),
            Kind::Path(p) => p.apply(x, y),
        }
    }

    /// How to place a glyph whose horizontal centre is at text-space `cx`.
    pub fn place_glyph(&self, cx: f64) -> Placement {
        match &self.0 {
            Kind::Style(_) => Placement::Bend,
            Kind::Path(p) => p.frame(cx).map_or(Placement::Hidden, Placement::Frame),
        }
    }
}

/// Most cubic segments of a type path that are followed (a hostile file cannot make this large).
const MAX_PATH_SEGMENTS: usize = 4096;

/// A flattened path with arc lengths: text-space `x` is arc length from `start`, `y` the
/// distance to the left of the direction of travel (the side the glyph tops point to).
#[derive(Clone, Debug)]
struct PathFrame {
    pts: Vec<(f64, f64)>,
    /// Arc length at each point.
    cum: Vec<f64>,
    start: f64,
}

impl PathFrame {
    fn new(path: &Subpath, start: f32, flip: bool) -> Option<PathFrame> {
        let mut pts: Vec<(f64, f64)> = Vec::new();
        for seg in path.segments().iter().take(MAX_PATH_SEGMENTS) {
            let p = seg.map(|q| (q.x, q.y));
            if p.iter().any(|q| !q.0.is_finite() || !q.1.is_finite()) {
                return None;
            }
            let poly = (p[1].0 - p[0].0).hypot(p[1].1 - p[0].1) + (p[2].0 - p[1].0).hypot(p[2].1 - p[1].1) + (p[3].0 - p[2].0).hypot(p[3].1 - p[2].1);
            let n = ((poly / 2.0).ceil() as usize).clamp(4, 128);
            if pts.is_empty() {
                pts.push(p[0]);
            }
            for i in 1..=n {
                let t = i as f64 / n as f64;
                let u = 1.0 - t;
                let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                pts.push((a * p[0].0 + b * p[1].0 + c * p[2].0 + d * p[3].0, a * p[0].1 + b * p[1].1 + c * p[2].1 + d * p[3].1));
            }
        }
        // Drop repeated points so every piece has a direction.
        pts.dedup_by(|b, a| (b.0 - a.0).abs() < 1e-9 && (b.1 - a.1).abs() < 1e-9);
        if flip {
            pts.reverse();
        }
        let mut cum = Vec::with_capacity(pts.len());
        let mut acc = 0.0;
        for (i, p) in pts.iter().enumerate() {
            if let Some(q) = i.checked_sub(1).and_then(|j| pts.get(j)) {
                acc += (p.0 - q.0).hypot(p.1 - q.1);
            }
            cum.push(acc);
        }
        (pts.len() >= 2 && acc.is_finite() && acc > 1e-6 && start.is_finite()).then(|| PathFrame { pts, cum, start: f64::from(start) })
    }

    fn length(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }

    /// Point and unit tangent at arc length `s`; past either end the end tangent continues straight.
    fn at(&self, s: f64) -> ((f64, f64), (f64, f64)) {
        let last = self.pts.len() - 1;
        let i = self.cum.partition_point(|&c| c < s).clamp(1, last);
        let (a, b) = (self.pts[i - 1], self.pts[i]);
        let len = (b.0 - a.0).hypot(b.1 - a.1).max(1e-12);
        let t = ((b.0 - a.0) / len, (b.1 - a.1) / len);
        let along = s - self.cum[i - 1];
        ((a.0 + t.0 * along, a.1 + t.1 * along), t)
    }

    fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let (p, t) = self.at(self.start + x);
        (p.0 - t.1 * y, p.1 + t.0 * y)
    }

    /// The rigid map for a glyph centred at text-space `cx`; `None` outside the path.
    fn frame(&self, cx: f64) -> Option<[f64; 6]> {
        let s = self.start + cx;
        if !(0.0..=self.length()).contains(&s) {
            return None;
        }
        let (p, t) = self.at(s);
        Some([t.0, t.1, -t.1, t.0, p.0 - t.0 * cx, p.1 - t.1 * cx])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warp(style: &str, value: f32) -> Warp {
        let w = TextWarp { style: psd_style(style).unwrap().into(), value, horizontal_distortion: 0.0, vertical_distortion: 0.0, horizontal: true };
        Warp::new(&w, [0.0, -40.0, 200.0, 10.0]).unwrap()
    }

    /// #218: the Warp Text dialog shows a layer's PSD style by its short id.
    #[test]
    fn short_ids_round_trip_with_psd_styles() {
        assert_eq!(short_style("warpFlag"), Some("flag"));
        assert_eq!(short_style("warpNone"), Some("none"));
        assert_eq!(short_style("warpCustom"), None);
        for (psd, short) in STYLES {
            assert_eq!(short_style(psd), Some(short));
            assert_eq!(psd_style(short_style(psd).unwrap()), Some(psd));
        }
    }

    #[test]
    fn styles_resolve_and_none_is_identity() {
        assert_eq!(psd_style("Arc"), Some("warpArc"));
        assert_eq!(psd_style("warpTwist"), Some("warpTwist"));
        assert_eq!(psd_style("none"), Some("warpNone"));
        assert_eq!(psd_style("spiral"), None);
        let none = TextWarp { style: "warpNone".into(), value: 50.0, ..Default::default() };
        assert!(Warp::new(&none, [0.0, 0.0, 10.0, 10.0]).is_none());
        let flat = TextWarp { style: "warpArc".into(), value: 0.0, ..Default::default() };
        assert!(Warp::new(&flat, [0.0, 0.0, 10.0, 10.0]).is_none());
    }

    #[test]
    fn every_style_is_continuous_and_fixes_the_centre_column() {
        for (_, id) in STYLES {
            let w = warp(id, 50.0);
            let (x, y) = w.apply(100.0, -15.0);
            assert!(x.is_finite() && y.is_finite(), "{id}");
            // Tiny bends barely move anything.
            let t = warp(id, 0.5);
            let (x2, y2) = t.apply(20.0, 0.0);
            assert!((x2 - 20.0).abs() < 2.0 && y2.abs() < 2.0, "{id}: {x2},{y2}");
        }
    }

    #[test]
    fn arc_raises_the_middle_relative_to_the_ends() {
        let w = warp("arc", 50.0);
        let mid = w.apply(100.0, -15.0);
        let end = w.apply(0.0, -15.0);
        // As in Photoshop: the bottom edge keeps its corners, the middle rises and the ends of
        // the upper rows swing out and down along the arc.
        let corner = w.apply(0.0, 10.0);
        assert!(corner.0.abs() < 1e-9 && (corner.1 - 10.0).abs() < 1e-9, "{corner:?}");
        assert!(mid.1 < -15.0, "the middle rises: {mid:?}");
        assert!(end.1 > mid.1 + 5.0, "ends drop below the centre: {end:?}");
        assert!(end.0 < 0.0, "ends swing out along the arc: {end:?}");
        // A negative bend smiles.
        let s = warp("arc", -50.0);
        assert!(s.apply(0.0, -15.0).1 < -20.0);
    }

    #[test]
    fn bulge_and_squeeze_change_height_in_the_middle() {
        let b = warp("bulge", 50.0);
        assert!(b.apply(100.0, -40.0).1 < -50.0);
        // Squeeze (Photoshop) pinches the sides in while the top and bottom bow out a little.
        let s = warp("squeeze", 50.0);
        assert!(s.apply(0.0, -15.0).0 > 10.0, "{:?}", s.apply(0.0, -15.0));
        assert!(s.apply(100.0, -40.0).1 < -40.0);
    }
}
