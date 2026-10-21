//! The canvas zoom range and Photoshop's Zoom In / Zoom Out ladder, measured on Photoshop 25.4:
//!
//! - the zoom runs up to 12800 %, and down to where the document's shorter side is 2 px
//!   (4 × 4 px: 50 %; 2000 × 1500 px: 0.1333 %);
//! - Zoom In / Zoom Out (⌘+ / ⌘−) and a Zoom tool click step along [`LADDER`], from any zoom to
//!   the next step (880 %: in to 1200 %, out to 800 %); they go no lower than 0.1 %, or the
//!   document's minimum when that is higher;
//! - ⌥ + wheel multiplies the zoom by 1.1 a notch (`wheel_nav`), and a notch that would pass a
//!   limit stops exactly on it.
//!
//! Every way of setting the zoom (wheel, Zoom tool, menu, zoom fields, Navigator, control
//! channel) clamps through here, so no control can pull the zoom back to a narrower range of
//! its own.

/// 12800 %.
pub const MAX: f32 = 128.0;

/// Zoom In / Zoom Out steps (1 = 100 %).
pub const LADDER: [f32; 33] = [
    0.001,
    0.002,
    0.003,
    0.004,
    0.005,
    0.007,
    0.01,
    0.015,
    0.02,
    0.03,
    0.04,
    0.05,
    0.0625,
    1.0 / 12.0,
    0.125,
    1.0 / 6.0,
    0.25,
    1.0 / 3.0,
    0.5,
    2.0 / 3.0,
    1.0,
    2.0,
    3.0,
    4.0,
    5.0,
    6.0,
    7.0,
    8.0,
    12.0,
    16.0,
    32.0,
    64.0,
    128.0,
];

/// The lowest zoom for a document of `size` px: its shorter side 2 px on screen.
pub fn min(size: [u32; 2]) -> f32 {
    match size[0].min(size[1]) {
        // No size known yet (a view before its first frame).
        0 => LADDER[0],
        short => (2.0 / short as f32).min(MAX),
    }
}

/// `zoom` within the range for a document of `size` px (a non-finite zoom becomes 100 %).
pub fn clamp(zoom: f32, size: [u32; 2]) -> f32 {
    if zoom.is_finite() { zoom.min(MAX).max(min(size)) } else { 1.0 }
}

/// A zoom field's range in percent for `view`: the zoom range, and always the view's zoom (a
/// field narrower than the zoom would clamp it and write it back every frame, which once made
/// the wheel slide the image towards the pointer at 3200 %).
pub fn percent_range(view: &crate::state::View) -> std::ops::RangeInclusive<f32> {
    let z = if view.zoom.is_finite() { view.zoom } else { 1.0 };
    (min(view.doc_size).min(z) * 100.0)..=(MAX.max(z) * 100.0)
}

/// Zoom In (`dir` > 0) or Zoom Out (`dir` < 0) from `zoom`, for a document of `size` px.
pub fn step(zoom: f32, dir: i32, size: [u32; 2]) -> f32 {
    let z = clamp(zoom, size);
    let next = if dir > 0 {
        LADDER.iter().copied().find(|s| *s > z * 1.001).unwrap_or(MAX)
    } else {
        // Below the ladder's lowest step (reached by the wheel), Zoom Out stays put.
        LADDER.iter().rev().copied().find(|s| *s < z * 0.999).unwrap_or(z)
    };
    clamp(next, size)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PHOTO: [u32; 2] = [2000, 1500];

    fn pct(z: f32) -> f32 {
        (z * 100.0 * 1000.0).round() / 1000.0
    }

    #[test]
    fn the_ladder_is_photoshops() {
        // Photoshop 25.4, View › Zoom In from 100 %, then Zoom Out from 100 % (2000 × 1500 px).
        let mut z = 1.0;
        let mut up = Vec::new();
        while z < MAX {
            z = step(z, 1, PHOTO);
            up.push(pct(z));
        }
        assert_eq!(up, [200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 1200.0, 1600.0, 3200.0, 6400.0, 12800.0]);
        assert_eq!(step(MAX, 1, PHOTO), MAX, "Zoom In stops at 12800 %");
        let mut z = 1.0;
        let mut down = Vec::new();
        for _ in 0..20 {
            z = step(z, -1, PHOTO);
            down.push(pct(z));
        }
        assert_eq!(
            down,
            [66.667, 50.0, 33.333, 25.0, 16.667, 12.5, 8.333, 6.25, 5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.4, 0.3, 0.2, 0.133],
            "the last step is the document's minimum (1500 px tall: 2 / 1500)"
        );
        assert_eq!(step(z, -1, PHOTO), z, "Zoom Out stops at the minimum");
    }

    #[test]
    fn off_ladder_zooms_step_to_the_next_rung() {
        assert_eq!(step(8.8, 1, PHOTO), 12.0);
        assert_eq!(step(8.8, -1, PHOTO), 8.0);
        assert_eq!(step(1.331, 1, PHOTO), 2.0);
        assert_eq!(step(1.331, -1, PHOTO), 1.0);
    }

    #[test]
    fn the_minimum_follows_the_shorter_side() {
        assert_eq!(min([4, 4]), 0.5);
        assert_eq!(pct(min([300, 1000])), 0.667);
        assert_eq!(pct(min([1000, 300])), 0.667);
        // A huge document: the wheel goes below 0.1 %, Zoom Out stops there.
        let big = [12_000, 12_000];
        assert!(min(big) < 0.001);
        assert_eq!(step(0.0015, -1, big), 0.001);
        assert_eq!(step(0.001, -1, big), 0.001);
        assert_eq!(step(0.0007, -1, big), 0.0007, "below the ladder Zoom Out stays put");
        // A 1 px document can't go below 200 %.
        assert_eq!(min([1, 1]), 2.0);
        assert_eq!(step(2.0, -1, [1, 1]), 2.0);
        assert_eq!(min([0, 0]), 0.001, "unknown size");
    }

    #[test]
    fn clamp_keeps_the_range_and_survives_bad_input() {
        assert_eq!(clamp(500.0, PHOTO), MAX);
        assert_eq!(clamp(0.0, PHOTO), min(PHOTO));
        assert_eq!(clamp(f32::NAN, PHOTO), 1.0);
        assert_eq!(clamp(f32::INFINITY, PHOTO), 1.0);
        assert_eq!(clamp(3.3, PHOTO), 3.3);
    }
}
