//! The pixel grid of the CPU canvas path, matching the GPU canvas shader (`gpu_canvas.rs`):
//! shown above 500% zoom, one device pixel wide, lightening a dark pixel by about a quarter
//! (darkening a light one), and only over pixels that have content, never over empty checker.
//! The line colour fades from white to black across a luma band ([`light_share`]) instead of
//! switching at one cut, so an image whose tones sit near the middle gets an even grid rather
//! than cells that flip between bright and dark lines.
//!
//! The shader reads each pixel's alpha as it draws. Here the same rule runs on the composite of
//! the part of the document in view, turned into merged line segments. Both the composite and
//! the segments are cached per document revision and per padded region, so panning inside the
//! padding costs nothing and the work is bounded by the viewport at 500% or more, never by the
//! document (AGENTS.md rule 8).

use egui::{Color32, Painter, Stroke};
use photocraft_doc::Document;
use photocraft_geom::Rect as DRect;

use crate::PhotocraftApp;
use crate::canvas::ViewXform;

/// The grid shows above this zoom (1.0 = 100%). Shared with the GPU canvas.
pub const MIN_ZOOM: f32 = 5.0;
/// Opacity of the grid line over a fully opaque dark pixel. Shared with the GPU canvas.
pub const STRENGTH: f32 = 0.25;
/// Light pixels get a black line this much weaker than the white one on dark pixels. The WGSL
/// in `gpu_canvas.rs` spells the same number.
pub const LIGHT_SCALE: f32 = 0.64;
/// The line is fully white at or below the first luma and fully black (at `LIGHT_SCALE`
/// strength) at or above the second; between them the two blend. The WGSL spells the same band.
const FADE_LUMA: [f32; 2] = [0.35, 0.75];
/// The share of black in the line is kept to this many steps, so noisy mid-tone pixels fall into
/// a few colours and equal-colour runs still merge into long segments.
const FADE_STEPS: f32 = 8.0;
/// Luminance of the checker under a see-through pixel when deciding light or dark; between the
/// default checker's two greys.
const CHECKER_LUMA: f32 = 0.9;
/// Cached regions snap to this many document pixels, so a small pan stays inside the cache.
const REGION_SNAP: i32 = 64;
/// Most segments kept. A document that is a fine lattice of transparent holes would otherwise
/// produce one segment per cell; past this the grid is simply left off the remainder.
const MAX_SEGMENTS: usize = 60_000;

/// Whether the grid is drawn at `zoom`, in device (physical) pixels per document pixel: the
/// canvas view zoom on the CPU path, the shader's point scale times `pixels_per_point` on the GPU.
pub fn shows_at(zoom: f32) -> bool {
    zoom > MIN_ZOOM
}

/// How light the line should be for a pixel showing `luma`: 0 is a white line, 1 a black one.
/// Smoothstep across [`FADE_LUMA`], rounded to [`FADE_STEPS`]. Non-finite luma reads as dark.
pub fn light_share(luma: f32) -> f32 {
    let luma = if luma.is_finite() { luma } else { 0.0 };
    let x = ((luma - FADE_LUMA[0]) / (FADE_LUMA[1] - FADE_LUMA[0])).clamp(0.0, 1.0);
    (x * x * (3.0 - 2.0 * x) * FADE_STEPS).round() / FADE_STEPS
}

/// One grid line piece in document pixels, axis-aligned, with its colour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridSeg {
    pub from: [i32; 2],
    pub to: [i32; 2],
    pub color: Color32,
}

/// The grid of a document region, valid for one revision and preview.
pub struct GridLines {
    revision: u64,
    preview_key: u64,
    region: DRect,
    segs: Vec<GridSeg>,
}

/// The line colour over a pixel with straight-alpha `rgba` (0..1), or `None` where there is
/// nothing to draw on: transparent pixels show checker, which stays clean.
pub fn cell_color(rgba: [f32; 4]) -> Option<Color32> {
    let a = rgba[3];
    if a.is_nan() || a <= 0.0 {
        return None;
    }
    let a = a.min(1.0);
    let luma = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2];
    let shown = luma * a + CHECKER_LUMA * (1.0 - a);
    let t = light_share(shown);
    // A white layer and a black layer over the pixel, as one grey with the combined alpha.
    let (white, black) = (STRENGTH * a * (1.0 - t), STRENGTH * a * LIGHT_SCALE * t);
    let total = white + black;
    let alpha = (total * 255.0).round() as u8;
    if alpha == 0 {
        return None;
    }
    let grey = (white / total * 255.0).round() as u8;
    Some(Color32::from_rgba_unmultiplied(grey, grey, grey, alpha))
}

/// Merge consecutive cells of equal colour along one line into `emit(start, end, colour)` runs.
fn runs(len: i32, color_at: impl Fn(i32) -> Option<Color32>, mut emit: impl FnMut(i32, i32, Color32)) {
    let mut open: Option<(i32, Color32)> = None;
    for i in 0..len {
        let c = color_at(i);
        match (open, c) {
            (Some((_, oc)), Some(c)) if oc == c => {}
            _ => {
                if let Some((s, oc)) = open {
                    emit(s, i, oc);
                }
                open = c.map(|c| (i, c));
            }
        }
    }
    if let Some((s, oc)) = open {
        emit(s, len, oc);
    }
}

/// The grid segments for the straight-alpha `px` covering `rect` (row-major). Each pixel owns
/// its left and top edge, as in the shader, so the document's right and bottom edges carry no
/// line. Runs of equal colour merge, so an opaque region costs one segment per line.
pub fn segments(px: &[[f32; 4]], rect: DRect) -> Vec<GridSeg> {
    let (w, h) = (rect.width() as i32, rect.height() as i32);
    let mut out = Vec::new();
    if w == 0 || h == 0 || px.len() < w as usize * h as usize {
        return out;
    }
    let at = |x: i32, y: i32| px.get(y as usize * w as usize + x as usize).and_then(|p| cell_color(*p));
    for x in 0..w {
        if out.len() >= MAX_SEGMENTS {
            return out;
        }
        runs(h, |y| at(x, y), |s, e, color| out.push(GridSeg { from: [rect.x0 + x, rect.y0 + s], to: [rect.x0 + x, rect.y0 + e], color }));
    }
    for y in 0..h {
        if out.len() >= MAX_SEGMENTS {
            return out;
        }
        runs(w, |x| at(x, y), |s, e, color| out.push(GridSeg { from: [rect.x0 + s, rect.y0 + y], to: [rect.x0 + e, rect.y0 + y], color }));
    }
    out.truncate(MAX_SEGMENTS);
    out
}

/// The part of the document to cache for a view showing `visible`: that area within `bounds`,
/// grown outward to multiples of [`REGION_SNAP`].
pub fn cache_region(visible: DRect, bounds: DRect) -> DRect {
    let v = visible.intersect(&bounds);
    if v.is_empty() {
        return DRect::EMPTY;
    }
    let down = |n: i32| n.div_euclid(REGION_SNAP) * REGION_SNAP;
    let up = |n: i32| down(n.saturating_add(REGION_SNAP - 1));
    DRect::new(down(v.x0), down(v.y0), up(v.x1), up(v.y1)).intersect(&bounds)
}

impl GridLines {
    fn build(doc: &Document, revision: u64, preview_key: u64, region: DRect) -> Self {
        let segs = if region.is_empty() { Vec::new() } else { segments(&photocraft_compose::render(doc, region).px, region) };
        Self { revision, preview_key, region, segs }
    }

    /// Whether this still holds the grid of `need` at the given revision and preview.
    fn covers(&self, revision: u64, preview_key: u64, need: &DRect) -> bool {
        self.revision == revision && self.preview_key == preview_key && self.region.contains_rect(need)
    }
}

/// Draw the grid of document `idx` over the CPU-composited canvas. `visible` is the document
/// area in view (document pixels); `output` the display the canvas is drawn for.
pub fn paint(app: &mut PhotocraftApp, painter: &Painter, xf: &ViewXform, idx: usize, output: Option<u32>, visible: DRect) {
    let Some((id, revision)) = app.session.documents().get(idx).map(|st| (st.doc.id, st.revision)) else { return };
    let (doc, preview_key) = crate::canvas::display_doc(app, idx);
    let need = visible.intersect(&doc.bounds());
    if need.is_empty() {
        return;
    }
    let Some(cache) = app.canvases.get_mut(&crate::canvas::cache_key(id, output)) else { return };
    if !cache.grid.as_ref().is_some_and(|g| g.covers(revision, preview_key, &need)) {
        cache.grid = Some(GridLines::build(&doc, revision, preview_key, cache_region(visible, doc.bounds())));
    }
    let Some(grid) = &cache.grid else { return };
    for s in &grid.segs {
        let (a, b) = (xf.to_screen(s.from[0] as f32, s.from[1] as f32), xf.to_screen(s.to[0] as f32, s.to[1] as f32));
        painter.line_segment([a, b], Stroke::new(1.0, s.color));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPAQUE_DARK: [f32; 4] = [0.1, 0.1, 0.1, 1.0];
    const OPAQUE_LIGHT: [f32; 4] = [0.9, 0.9, 0.9, 1.0];

    #[test]
    fn grid_starts_above_five_hundred_percent() {
        assert!(!shows_at(5.0));
        assert!(!shows_at(1.0));
        assert!(shows_at(5.01));
        assert!(shows_at(128.0));
    }

    #[test]
    fn transparent_pixels_get_no_line() {
        assert_eq!(cell_color([0.0, 0.0, 0.0, 0.0]), None);
        assert_eq!(cell_color([1.0, 1.0, 1.0, 0.0]), None);
        assert_eq!(cell_color([0.5, 0.5, 0.5, f32::NAN]), None);
        // Almost invisible pixels round to no line rather than a faint speck.
        assert_eq!(cell_color([0.0, 0.0, 0.0, 0.001]), None);
    }

    #[test]
    fn dark_pixels_get_a_quarter_white_and_light_ones_less_black() {
        assert_eq!(cell_color(OPAQUE_DARK), Some(Color32::from_white_alpha(64)));
        assert_eq!(cell_color(OPAQUE_LIGHT), Some(Color32::from_black_alpha(41)));
    }

    #[test]
    fn line_strength_follows_alpha_like_the_shader() {
        // Half-clear black over the checker reads dark (0.45): mostly a white line at about half
        // strength (a little black is already blended in).
        let half = cell_color([0.0, 0.0, 0.0, 0.5]).unwrap();
        assert!((28..=32).contains(&half.a()) && half.r() as f32 > 0.8 * half.a() as f32, "{half:?}");
        // Half-clear white reads light (0.95): a black line at half the light-pixel strength.
        assert_eq!(cell_color([1.0, 1.0, 1.0, 0.5]), Some(Color32::from_black_alpha(20)));
        let full = cell_color([0.0, 0.0, 0.0, 1.0]);
        assert_eq!(full, Some(Color32::from_white_alpha(64)));
        // Opaque white keeps a full-strength black line: the grid never vanishes on a blank page.
        assert_eq!(cell_color([1.0, 1.0, 1.0, 1.0]), Some(Color32::from_black_alpha(41)));
    }

    #[test]
    fn out_of_range_values_do_not_panic() {
        // NaN colour with a valid alpha reads as dark; alpha past 1 is clamped.
        assert_eq!(cell_color([f32::INFINITY, f32::NEG_INFINITY, f32::NAN, 5.0]), Some(Color32::from_white_alpha(64)));
        assert_eq!(cell_color([0.0; 4]), None);
    }

    #[test]
    fn line_colour_fades_across_mid_tones_instead_of_flipping() {
        // Walk the whole luma range: the line's net effect on a pixel of that luma (white pulls
        // up, black pulls down) must change by only a little between neighbouring tones, and the
        // band edges keep the pure white and pure black lines.
        assert_eq!(light_share(0.2), 0.0);
        assert_eq!(light_share(0.9), 1.0);
        assert_eq!(light_share(f32::NAN), 0.0);
        let mut prev: Option<f32> = None;
        for i in 0..=100 {
            let l = i as f32 / 100.0;
            let c = cell_color([l, l, l, 1.0]).unwrap_or(Color32::TRANSPARENT);
            // Un-premultiply to the over colour and its alpha, then the change it makes.
            let a = c.a() as f32 / 255.0;
            let over = if a > 0.0 { c.r() as f32 / 255.0 / a } else { 0.0 };
            let delta = a * (over - l);
            if let Some(p) = prev {
                assert!((delta - p).abs() < 0.06, "luma {l}: line effect jumps from {p} to {delta}");
            }
            prev = Some(delta);
        }
    }

    #[test]
    fn mid_tone_noise_gets_near_identical_lines() {
        // Luma 0.51 to 0.59 straddles the old 0.55 cut: it gave a span of about 0.2 here.
        let strengths: Vec<f32> = (51..=59)
            .map(|i| {
                let l = i as f32 / 100.0;
                let c = cell_color([l, l, l, 1.0]).unwrap();
                let a = c.a() as f32 / 255.0;
                a * (c.r() as f32 / 255.0 / a - l)
            })
            .collect();
        let (lo, hi) = strengths.iter().fold((f32::MAX, f32::MIN), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
        assert!(hi - lo < 0.08, "line effect over mid tones spans {lo}..{hi}");
        assert!(hi.abs() < 0.06 && lo.abs() < 0.06, "mid-tone lines should nearly vanish: {strengths:?}");
    }

    fn grid(w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) -> Vec<[f32; 4]> {
        (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| f(x, y)).collect()
    }

    #[test]
    fn opaque_region_is_one_segment_per_line() {
        let rect = DRect::from_xywh(0, 0, 6, 4);
        let segs = segments(&grid(6, 4, |_, _| OPAQUE_DARK), rect);
        assert_eq!(segs.len(), 6 + 4);
        let white = Color32::from_white_alpha(64);
        assert!(segs.contains(&GridSeg { from: [0, 0], to: [0, 4], color: white }));
        assert!(segs.contains(&GridSeg { from: [5, 0], to: [5, 4], color: white }));
        assert!(segs.contains(&GridSeg { from: [0, 3], to: [6, 3], color: white }));
        // No closing line on the right or bottom edge of the region.
        assert!(!segs.iter().any(|s| s.from[0] == 6 || s.from[1] == 4));
    }

    #[test]
    fn empty_region_draws_nothing() {
        let rect = DRect::from_xywh(0, 0, 8, 8);
        assert!(segments(&grid(8, 8, |_, _| [0.0; 4]), rect).is_empty());
    }

    #[test]
    fn a_lone_pixel_owns_its_left_and_top_edges() {
        let rect = DRect::from_xywh(10, 20, 5, 5);
        let px = grid(5, 5, |x, y| if (x, y) == (2, 3) { OPAQUE_DARK } else { [0.0; 4] });
        let segs = segments(&px, rect);
        let white = Color32::from_white_alpha(64);
        assert_eq!(segs.len(), 2);
        assert!(segs.contains(&GridSeg { from: [12, 23], to: [12, 24], color: white }));
        assert!(segs.contains(&GridSeg { from: [12, 23], to: [13, 23], color: white }));
    }

    #[test]
    fn lines_stop_where_the_pixels_do() {
        // Left half opaque, right half transparent: vertical lines only on the left half and
        // the horizontal ones only as long as the opaque part.
        let rect = DRect::from_xywh(0, 0, 8, 2);
        let px = grid(8, 2, |x, _| if x < 4 { OPAQUE_DARK } else { [0.0; 4] });
        let segs = segments(&px, rect);
        assert_eq!(segs.len(), 4 + 2);
        assert!(segs.iter().all(|s| s.to[0] <= 4));
        assert!(segs.contains(&GridSeg { from: [0, 1], to: [4, 1], color: Color32::from_white_alpha(64) }));
    }

    #[test]
    fn runs_split_where_the_colour_changes() {
        let rect = DRect::from_xywh(0, 0, 1, 4);
        let px = vec![OPAQUE_DARK, OPAQUE_DARK, OPAQUE_LIGHT, OPAQUE_LIGHT];
        let segs: Vec<_> = segments(&px, rect).into_iter().filter(|s| s.from[0] == s.to[0]).collect();
        assert_eq!(segs.len(), 2);
        assert!(segs.contains(&GridSeg { from: [0, 0], to: [0, 2], color: Color32::from_white_alpha(64) }));
        assert!(segs.contains(&GridSeg { from: [0, 2], to: [0, 4], color: Color32::from_black_alpha(41) }));
    }

    #[test]
    fn a_checkered_lattice_is_capped() {
        let n = 400u32;
        let rect = DRect::from_xywh(0, 0, n, n);
        let px = grid(n, n, |x, y| if (x + y) % 2 == 0 { OPAQUE_DARK } else { [0.0; 4] });
        assert_eq!(segments(&px, rect).len(), MAX_SEGMENTS);
    }

    #[test]
    fn short_or_empty_buffers_are_ignored() {
        assert!(segments(&[], DRect::from_xywh(0, 0, 4, 4)).is_empty());
        assert!(segments(&[OPAQUE_DARK; 3], DRect::from_xywh(0, 0, 4, 4)).is_empty());
        assert!(segments(&[], DRect::EMPTY).is_empty());
    }

    #[test]
    fn cache_region_snaps_outward_and_stays_in_the_document() {
        let bounds = DRect::new(0, 0, 1000, 500);
        assert_eq!(cache_region(DRect::new(70, 10, 130, 90), bounds), DRect::new(64, 0, 192, 128));
        assert_eq!(cache_region(DRect::new(-50, -50, 2000, 2000), bounds), bounds);
        assert_eq!(cache_region(DRect::new(900, 400, 1200, 700), bounds), DRect::new(896, 384, 1000, 500));
        assert!(cache_region(DRect::new(2000, 2000, 2100, 2100), bounds).is_empty());
    }

    #[test]
    fn cached_grid_serves_a_pan_inside_its_region_only() {
        let g = GridLines { revision: 3, preview_key: 7, region: DRect::new(64, 0, 192, 128), segs: Vec::new() };
        assert!(g.covers(3, 7, &DRect::new(70, 10, 130, 90)));
        assert!(!g.covers(4, 7, &DRect::new(70, 10, 130, 90)));
        assert!(!g.covers(3, 8, &DRect::new(70, 10, 130, 90)));
        assert!(!g.covers(3, 7, &DRect::new(60, 10, 130, 90)));
    }
}
