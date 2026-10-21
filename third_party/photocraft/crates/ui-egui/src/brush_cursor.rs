//! The painting cursor for brushes with a tip shape: sampled tip outlines and rotated or
//! flattened round tips. Everything else falls back to the plain circle (`tool_cursor`).

use std::sync::Arc;

use egui::{CursorIcon, Painter, Pos2, Vec2};
use photocraft_paint::brush::{BrushSettings, TipShape};

/// Below this on-screen radius an outline is unreadable: show the crosshair instead.
const MIN_RADIUS: f32 = 3.0;

#[derive(Clone)]
struct Cached {
    key: Key,
    rings: Option<Arc<Vec<Vec<[f32; 2]>>>>,
}

/// The brush fields an outline depends on.
#[derive(Clone)]
struct Key {
    tip: TipShape,
    angle: f32,
    roundness: f32,
    hardness: f32,
    flip_x: bool,
    flip_y: bool,
    full: bool,
}

impl Key {
    fn of(b: &BrushSettings, full: bool) -> Self {
        Self { tip: b.tip.clone(), angle: b.angle, roundness: b.roundness, hardness: b.hardness, flip_x: b.flip_x, flip_y: b.flip_y, full }
    }
    fn matches(&self, b: &BrushSettings, full: bool) -> bool {
        self.tip == b.tip
            && self.angle == b.angle
            && self.roundness == b.roundness
            && self.hardness == b.hardness
            && self.flip_x == b.flip_x
            && self.flip_y == b.flip_y
            && self.full == full
    }
}

/// The screen vectors of one tip radius (`size / 2` document pixels) along the document's x and
/// y axes: the view's zoom, rotation, flip and pixel aspect, which the outline follows as the
/// stroke does.
pub(crate) fn tip_axes(xf: &crate::canvas::ViewXform, size: f32) -> [Vec2; 2] {
    let r = if size.is_finite() { size.max(0.0) / 2.0 } else { 0.0 };
    [xf.map_vec(Vec2::new(r, 0.0)), xf.map_vec(Vec2::new(0.0, r))]
}

/// Rings in radius units (document orientation) on screen, through [`tip_axes`].
fn on_screen(rings: &[Vec<[f32; 2]>], axes: [Vec2; 2]) -> Vec<Vec<Vec2>> {
    rings.iter().map(|r| r.iter().map(|p| axes[0] * p[0] + axes[1] * p[1]).collect()).collect()
}

/// The brush cursor at `at`. `radius` is the tip's screen radius (size / 2 at the current zoom),
/// for the plain circle; `axes` comes from [`tip_axes`], for a traced outline. `normal` is
/// Preferences › Cursors "Normal Brush Tip" (the 50 % contour, not the full tip).
pub(crate) fn cursor(painter: &Painter, at: Pos2, brush: &BrushSettings, radius: f32, axes: [Vec2; 2], normal: bool, centre: bool) -> CursorIcon {
    if let Some(rings) = rings(painter.ctx(), brush, !normal) {
        let pts = on_screen(&rings, axes);
        let extent = pts.iter().flatten().fold(0.0f32, |m, p| m.max(p.length()));
        if extent < MIN_RADIUS {
            return crate::tool_cursor::crosshair(painter, at, 4.0, 0.0);
        }
        return crate::tool_cursor::outline(painter, at, pts, centre);
    }
    let r = if normal { photocraft_paint::half_coverage_radius(radius, brush.hardness).max(1.0) } else { radius };
    crate::tool_cursor::circle(painter, at, r, centre)
}

/// The tip outline in radius units, cached across frames; `None` when the plain circle says it all
/// (a plain round tip) or the tip has no traceable shape.
pub(crate) fn rings(ctx: &egui::Context, brush: &BrushSettings, full: bool) -> Option<Arc<Vec<Vec<[f32; 2]>>>> {
    if matches!(brush.tip, TipShape::Round) && brush.angle == 0.0 && brush.roundness >= 1.0 {
        return None;
    }
    let id = egui::Id::new("photocraft-tip-outline");
    if let Some(c) = ctx.data(|d| d.get_temp::<Cached>(id))
        && c.key.matches(brush, full)
    {
        return c.rings;
    }
    let traced = photocraft_paint::outline::tip_outline(brush, full).map(Arc::new);
    ctx.data_mut(|d| d.insert_temp(id, Cached { key: Key::of(brush, full), rings: traced.clone() }));
    traced
}

/// Which texels of the document-pixel `rect` (`x0, y0, x1, y1`, row-major) lie inside the tip
/// `rings` (radius units, even-odd so holes stay clear) when the tip is centred on `at` with
/// `radius` document pixels. The Clone Source overlay uses it to clip to a shaped tip.
pub(crate) fn tip_mask(rings: &[Vec<[f32; 2]>], at: [f64; 2], radius: f64, rect: [i32; 4]) -> Vec<bool> {
    let [x0, y0, x1, y1] = rect;
    let (w, h) = (usize::try_from(x1 - x0).unwrap_or(0), usize::try_from(y1 - y0).unwrap_or(0));
    let mut mask = vec![false; w.saturating_mul(h)];
    let mut hits: Vec<f64> = Vec::new();
    for row in 0..h {
        let y = f64::from(y0) + row as f64 + 0.5;
        hits.clear();
        for ring in rings {
            for (i, a) in ring.iter().enumerate() {
                let Some(b) = ring.get((i + 1) % ring.len()) else { continue };
                let (ax, ay) = (at[0] + f64::from(a[0]) * radius, at[1] + f64::from(a[1]) * radius);
                let (bx, by) = (at[0] + f64::from(b[0]) * radius, at[1] + f64::from(b[1]) * radius);
                if (ay <= y) != (by <= y) {
                    hits.push(ax + (y - ay) / (by - ay) * (bx - ax));
                }
            }
        }
        hits.sort_by(f64::total_cmp);
        for pair in hits.chunks(2) {
            let [a, b] = pair else { continue };
            let (from, to) = (a - f64::from(x0) - 0.5, b - f64::from(x0) - 0.5);
            let start = from.ceil().max(0.0) as usize;
            let end = ((to.floor() + 1.0).max(0.0) as usize).min(w);
            for x in start..end {
                if let Some(m) = mask.get_mut(row * w + x) {
                    *m = true;
                }
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::canvas::ViewXform;
    use egui::{Rect, pos2};

    fn view(rotation: f32, flip: bool, aspect: f32) -> ViewXform {
        ViewXform { rect: Rect::from_min_size(pos2(0.0, 0.0), egui::vec2(400.0, 300.0)), zoom: 2.0, center: [50.0, 40.0], flip, rotation, aspect }
    }

    fn close(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 1e-4
    }

    /// The axes are the screen offsets of the document points one radius right of and below the
    /// pointer, whatever the view does.
    #[test]
    fn tip_axes_follow_the_view_like_to_screen() {
        for (rotation, flip, aspect) in [(0.0, false, 1.0), (30.0, false, 1.0), (-90.0, true, 1.0), (45.0, true, 2.0), (200.0, false, 0.5)] {
            let xf = view(rotation, flip, aspect);
            let [ex, ey] = tip_axes(&xf, 20.0);
            let o = xf.to_screen(60.0, 30.0);
            assert!(close(ex, xf.to_screen(70.0, 30.0) - o), "x axis at {rotation}°, flip {flip}, aspect {aspect}");
            assert!(close(ey, xf.to_screen(60.0, 40.0) - o), "y axis at {rotation}°, flip {flip}, aspect {aspect}");
        }
    }

    /// An unrotated, unflipped view gives what upstream's radius scaling gave.
    #[test]
    fn a_plain_view_scales_the_outline_by_the_screen_radius() {
        let rings = vec![vec![[1.0, 0.0], [0.0, 0.5], [-1.0, 0.0]]];
        let pts = on_screen(&rings, tip_axes(&view(0.0, false, 1.0), 20.0));
        let want = [Vec2::new(20.0, 0.0), Vec2::new(0.0, 10.0), Vec2::new(-20.0, 0.0)];
        assert!(pts[0].iter().zip(want).all(|(a, b)| close(*a, b)), "{pts:?}");
    }

    /// A tip pointing along the document's x axis points down the screen in a view turned 90°,
    /// and left in a flipped one.
    #[test]
    fn the_outline_turns_and_flips_with_the_view() {
        let rings = vec![vec![[1.0, 0.0]]];
        let turned = on_screen(&rings, tip_axes(&view(90.0, false, 1.0), 20.0));
        assert!(close(turned[0][0], Vec2::new(0.0, 20.0)), "{turned:?}");
        let flipped = on_screen(&rings, tip_axes(&view(0.0, true, 1.0), 20.0));
        assert!(close(flipped[0][0], Vec2::new(-20.0, 0.0)), "{flipped:?}");
    }

    #[test]
    fn hostile_sizes_give_finite_axes() {
        for size in [f32::NAN, f32::INFINITY, -5.0, 0.0] {
            let [ex, ey] = tip_axes(&view(30.0, true, 1.0), size);
            assert!(ex.x.is_finite() && ex.y.is_finite() && ey.x.is_finite() && ey.y.is_finite(), "size {size}");
        }
    }

    /// The overlay clip follows the tip: a square ring covers its own texels, a hole stays clear,
    /// and a rotated flat tip leaves the corners of the brush square outside.
    #[test]
    fn tip_mask_follows_the_ring_shape() {
        let square = vec![[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]];
        let hole = vec![[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]];
        let mask = tip_mask(&[square.clone(), hole], [10.0, 10.0], 10.0, [0, 0, 20, 20]);
        let at = |x: usize, y: usize| mask[y * 20 + x];
        assert!(at(6, 10) && at(14, 10) && at(10, 6) && at(10, 14), "inside the square");
        assert!(!at(3, 10) && !at(10, 16) && !at(0, 0), "outside the square");
        assert!(!at(10, 10) && at(8, 10), "the hole is clear");
        assert_eq!(mask.iter().filter(|m| **m).count(), 10 * 10 - 2 * 2, "10x10 square minus the 2x2 hole");
        assert!(tip_mask(&[square], [10.0, 10.0], 10.0, [5, 5, 5, 9]).is_empty(), "an empty rect is harmless");
    }
}
