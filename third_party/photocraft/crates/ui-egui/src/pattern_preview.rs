//! View › Pattern Preview (#1067): the canvas shows the document repeated around itself as a
//! seamless tile, so what leaves one edge is seen coming back at the opposite one. The copies
//! reuse the document's display texture (no compositing per copy); selection, guides and other
//! overlays stay on the document itself.

use std::ops::RangeInclusive;

/// Most tiles drawn along each axis, the document included: 15 on each side of the view's centre
/// tile (at most 960 copies, each one textured quad per visible texture tile). A tiny document
/// zoomed far out would otherwise need thousands of copies per frame; past this the pattern stops.
pub const MAX_PER_AXIS: i32 = 31;

/// The copies of a `size` document that cover `visible` (the document-pixel area the view shows:
/// x0, y0, x1, y1), as tile indices: copy `[i, j]` is the document moved by `i` widths and `j`
/// heights. The document itself (`[0, 0]`) is not listed. Row by row, left to right. When more
/// than [`MAX_PER_AXIS`] tiles would fit along an axis, the run is centred on the view.
pub fn copies(visible: [f64; 4], size: [u32; 2]) -> Vec<[i32; 2]> {
    let (Some(xs), Some(ys)) = (axis(visible[0], visible[2], size[0]), axis(visible[1], visible[3], size[1])) else {
        return Vec::new();
    };
    ys.flat_map(|j| xs.clone().map(move |i| [i, j])).filter(|&c| c != [0, 0]).collect()
}

/// Document offset of copy `c` in pixels of a `size` document.
pub fn offset(c: [i32; 2], size: [u32; 2]) -> [f32; 2] {
    [c[0] as f32 * size[0] as f32, c[1] as f32 * size[1] as f32]
}

/// Tile indices along one axis that overlap `lo..hi` for tiles `len` pixels long.
fn axis(lo: f64, hi: f64, len: u32) -> Option<RangeInclusive<i32>> {
    if len == 0 || !lo.is_finite() || !hi.is_finite() || hi <= lo {
        return None;
    }
    let n = f64::from(len);
    // Far past any real view; keeps the ± MAX_PER_AXIS arithmetic inside i32.
    let index = |v: f64| v.clamp(-1e9, 1e9) as i32;
    let first = index((lo / n).floor());
    let last = index((hi / n).ceil()) - 1;
    if last.saturating_sub(first) < MAX_PER_AXIS {
        return Some(first..=last);
    }
    let mid = index(((lo + hi) / 2.0 / n).floor());
    let half = MAX_PER_AXIS / 2;
    Some(mid - half..=mid + half)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_filling_the_view_has_no_copies() {
        // Zoomed in: the view shows part of a 1000 × 800 document.
        assert!(copies([100.0, 100.0, 600.0, 500.0], [1000, 800]).is_empty());
        // Exactly the document.
        assert!(copies([0.0, 0.0, 1000.0, 800.0], [1000, 800]).is_empty());
    }

    #[test]
    fn fit_view_gets_the_neighbours_that_show() {
        // A 100 × 100 document centred in a view 300 doc px wide and 100 tall: one copy each side.
        assert_eq!(copies([-100.0, 0.0, 200.0, 100.0], [100, 100]), vec![[-1, 0], [1, 0]]);
        // A view slightly larger than the document all round: the eight neighbours.
        let c = copies([-10.0, -10.0, 110.0, 110.0], [100, 100]);
        assert_eq!(c, vec![[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]);
    }

    #[test]
    fn zoomed_out_fills_the_view() {
        // 200 × 100 document, zoomed out so the view shows 1000 × 500 doc px around it.
        let c = copies([-400.0, -200.0, 600.0, 300.0], [200, 100]);
        // Columns -2..=2, rows -2..=2, minus the document itself.
        assert_eq!(c.len(), 5 * 5 - 1);
        assert!(c.contains(&[-2, -2]) && c.contains(&[2, 2]) && !c.contains(&[0, 0]));
        assert_eq!(offset([-2, 1], [200, 100]), [-400.0, 100.0]);
    }

    #[test]
    fn panned_view_only_draws_what_shows() {
        // Panned right so the document is off screen to the left: only copies to its right.
        let c = copies([250.0, 10.0, 450.0, 90.0], [100, 100]);
        assert_eq!(c, vec![[2, 0], [3, 0], [4, 0]]);
        // Panned up-left past the corner.
        assert_eq!(copies([-150.0, -150.0, -60.0, -60.0], [100, 100]), vec![[-2, -2], [-1, -2], [-2, -1], [-1, -1]]);
    }

    #[test]
    fn copies_are_capped_and_centred_on_the_view() {
        // A 1 px document zoomed out over a view thousands of pixels wide.
        let c = copies([-2000.0, -1000.0, 2000.0, 1000.0], [1, 1]);
        assert_eq!(c.len(), (MAX_PER_AXIS * MAX_PER_AXIS - 1) as usize);
        let half = MAX_PER_AXIS / 2;
        assert!(c.iter().all(|&[i, j]| (-half..=half).contains(&i) && (-half..=half).contains(&j)));
        // Panned far away: the run follows the view's centre, not the document.
        let c = copies([1_000_000.0, 0.0, 1_004_000.0, 1.0], [1, 1]);
        assert_eq!(c.len(), MAX_PER_AXIS as usize);
        assert!(c.iter().all(|&[i, j]| j == 0 && (1_002_000 - half..=1_002_000 + half).contains(&i)));
    }

    #[test]
    fn hostile_input_draws_nothing() {
        assert!(copies([0.0, 0.0, 10.0, 10.0], [0, 10]).is_empty());
        assert!(copies([f64::NAN, 0.0, 10.0, 10.0], [10, 10]).is_empty());
        assert!(copies([0.0, 0.0, f64::INFINITY, 10.0], [10, 10]).is_empty());
        assert!(copies([10.0, 0.0, 0.0, 10.0], [10, 10]).is_empty());
        // Huge but finite: capped, no overflow.
        assert_eq!(copies([-1e300, -1e300, 1e300, 1e300], [1, 1]).len(), (MAX_PER_AXIS * MAX_PER_AXIS - 1) as usize);
    }
}
