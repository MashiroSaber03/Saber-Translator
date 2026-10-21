//! Layer Style › Blending Options › Blend If: per-channel value ranges that hide a layer's
//! pixels by their own value ("This Layer") or by the value beneath them ("Underlying Layer").
//!
//! Each slider pair is stored as Photoshop stores it in a PSD layer record's blending ranges: a
//! black point and a white point, each split into a low and a high half (Alt-drag in the dialog),
//! on the 0..=255 scale whatever the document's depth. A value below the black low point or above
//! the white high point is hidden; between the two halves of a split point it fades linearly.

use serde::{Deserialize, Serialize};

/// One Blend If slider pair: `black` = (low, high), `white` = (low, high), all on 0..=255.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlendRange {
    pub black: [u8; 2],
    pub white: [u8; 2],
}

impl Default for BlendRange {
    fn default() -> Self {
        Self::FULL
    }
}

impl BlendRange {
    /// Every value blends (the sliders at the ends).
    pub const FULL: BlendRange = BlendRange { black: [0, 0], white: [255, 255] };

    /// Whether this range hides nothing.
    pub fn is_full(&self) -> bool {
        *self == Self::FULL
    }

    /// From PSD's byte order: black low, black high, white low, white high.
    pub fn from_bytes(b: [u8; 4]) -> Self {
        BlendRange { black: [b[0], b[1]], white: [b[2], b[3]] }
    }

    /// PSD's byte order (see [`BlendRange::from_bytes`]).
    pub fn to_bytes(self) -> [u8; 4] {
        [self.black[0], self.black[1], self.white[0], self.white[1]]
    }

    /// How much of a pixel with value `v` (0..=255, fractional for deep documents) shows:
    /// 0 below the black point or above the white point, 1 between them, and a linear ramp across
    /// a split point. An unsplit black point at `b` shows `v >= b`; an unsplit white point at `w`
    /// shows `v <= w`.
    pub fn weight(&self, v: f32) -> f32 {
        let ramp = |lo: u8, hi: u8, x: f32| -> f32 {
            // 0 at or below `lo`, 1 at or above `hi`.
            let (lo, hi) = (f32::from(lo), f32::from(hi));
            if hi <= lo { if x >= lo { 1.0 } else { 0.0 } } else { ((x - lo) / (hi - lo)).clamp(0.0, 1.0) }
        };
        let lo = ramp(self.black[0], self.black[1], v);
        // Mirror for the white point: 1 at or below its low half, 0 above its high half.
        let hi = ramp(255 - self.white[1], 255 - self.white[0], 255.0 - v);
        lo * hi
    }
}

/// A layer's Blend If settings. `ranges[0]` is the "Gray" entry (composite of the colour
/// channels), then one entry per colour channel of the document's mode (R, G, B / C, M, Y, K /
/// L, a, b / Gray), each as `[this layer, underlying layer]`. Empty = every pixel blends.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlendIf {
    pub ranges: Vec<[BlendRange; 2]>,
}

impl BlendIf {
    /// Whether no range hides anything (the Photoshop default).
    pub fn is_default(&self) -> bool {
        self.ranges.iter().flatten().all(BlendRange::is_full)
    }

    /// Entry `i` (0 = Gray), full when not stored.
    pub fn get(&self, i: usize) -> [BlendRange; 2] {
        self.ranges.get(i).copied().unwrap_or([BlendRange::FULL; 2])
    }

    /// Set entry `i` (0 = Gray), growing the list with full ranges as needed. Trailing full
    /// entries are dropped so an all-default setting compares equal to `BlendIf::default()`.
    pub fn set(&mut self, i: usize, pair: [BlendRange; 2]) {
        if self.ranges.len() <= i {
            self.ranges.resize(i + 1, [BlendRange::FULL; 2]);
        }
        self.ranges[i] = pair;
        while self.ranges.last().is_some_and(|p| p.iter().all(BlendRange::is_full)) {
            self.ranges.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_range_shows_everything() {
        let r = BlendRange::FULL;
        assert!(r.is_full());
        for v in [0.0, 0.5, 127.0, 254.9, 255.0] {
            assert_eq!(r.weight(v), 1.0);
        }
    }

    #[test]
    fn unsplit_points_are_hard_thresholds() {
        let r = BlendRange { black: [50, 50], white: [200, 200] };
        assert_eq!(r.weight(49.0), 0.0);
        assert_eq!(r.weight(49.9), 0.0);
        assert_eq!(r.weight(50.0), 1.0);
        assert_eq!(r.weight(200.0), 1.0);
        assert_eq!(r.weight(200.1), 0.0);
        assert!(!r.is_full());
    }

    #[test]
    fn split_points_ramp_linearly() {
        let r = BlendRange { black: [0, 100], white: [155, 255] };
        // A split black point fades from its low half (hidden) to its high half (shown).
        assert_eq!(r.weight(0.0), 0.0);
        assert!((r.weight(50.0) - 0.5).abs() < 1e-6);
        assert_eq!(r.weight(100.0), 1.0);
        assert_eq!(r.weight(155.0), 1.0);
        assert!((r.weight(205.0) - 0.5).abs() < 1e-6);
        assert_eq!(r.weight(255.0), 0.0);
        assert!(!r.is_full());
    }

    #[test]
    fn bytes_round_trip_in_psd_order() {
        let r = BlendRange { black: [10, 20], white: [230, 240] };
        assert_eq!(r.to_bytes(), [10, 20, 230, 240]);
        assert_eq!(BlendRange::from_bytes(r.to_bytes()), r);
    }

    #[test]
    fn set_trims_trailing_full_entries() {
        let mut b = BlendIf::default();
        assert!(b.is_default());
        let r = BlendRange { black: [30, 30], white: [255, 255] };
        b.set(2, [BlendRange::FULL, r]);
        assert_eq!(b.ranges.len(), 3);
        assert_eq!(b.get(2), [BlendRange::FULL, r]);
        assert_eq!(b.get(7), [BlendRange::FULL; 2]);
        assert!(!b.is_default());
        b.set(2, [BlendRange::FULL; 2]);
        assert_eq!(b, BlendIf::default());
    }
}
