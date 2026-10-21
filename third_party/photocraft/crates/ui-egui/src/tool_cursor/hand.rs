//! The Hand tool's pointer as signed distance fields (negative inside), in logical pixels from the
//! hotspot with y pointing down. An open hand while the Hand hovers and a fist while it drags, as
//! in Photoshop. Only Windows needs it: winit maps its `Grab` and `Grabbing` to the four-arrow move
//! cursor there, while macOS and the Linux themes already draw hands.

/// Half the side of the square the hand fits in, outline excluded.
pub(super) const HALF: f32 = 12.5;

/// Distance to the segment `a`–`b`, minus `r` (a capsule; a bare line when `r` is 0).
pub(super) fn capsule(x: f32, y: f32, a: [f32; 2], b: [f32; 2], r: f32) -> f32 {
    let (px, py) = (x - a[0], y - a[1]);
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0.0 { ((px * dx + py * dy) / len2).clamp(0.0, 1.0) } else { 0.0 };
    (px - dx * t).hypot(py - dy * t) - r
}

/// A box of half-size `half` about `c` with corners rounded by `r`.
fn round_box(x: f32, y: f32, c: [f32; 2], half: [f32; 2], r: f32) -> f32 {
    let qx = (x - c[0]).abs() - (half[0] - r);
    let qy = (y - c[1]).abs() - (half[1] - r);
    qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r
}

/// Four fingers of this x, reaching up to these tip heights (index to little finger).
const FINGER_X: [f32; 4] = [-5.1, -1.7, 1.7, 5.1];
const FINGER_R: f32 = 1.7;
const TIPS: [f32; 4] = [-7.0, -9.5, -8.5, -5.5];

/// From the hotspot to the shapes' own coordinates: the hotspot is the centre of the hand's box.
fn local(x: f32, y: f32) -> (f32, f32) {
    (x - 2.0, y - 0.5)
}

/// The hand's silhouette: negative inside it.
pub(super) fn shape(closed: bool, x: f32, y: f32) -> f32 {
    let (x, y) = local(x, y);
    if closed {
        let mut d = round_box(x, y, [0.0, 3.0], [7.2, 6.5], 3.6);
        for fx in FINGER_X {
            d = d.min(capsule(x, y, [fx, -1.0], [fx, -2.2], 1.9));
        }
        d.min(capsule(x, y, [-4.5, 7.0], [-9.6, 3.6], 2.0))
    } else {
        let mut d = round_box(x, y, [0.0, 4.5], [6.8, 5.5], 3.0);
        for (fx, tip) in FINGER_X.into_iter().zip(TIPS) {
            d = d.min(capsule(x, y, [fx, 3.0], [fx, tip], FINGER_R));
        }
        d.min(capsule(x, y, [-4.5, 7.0], [-10.5, 1.5], 1.9))
    }
}

/// The creases inside the silhouette (between the fingers, the curl of the fist): distance to the
/// nearest one, so they can be drawn as thin dark lines.
pub(super) fn crease(closed: bool, x: f32, y: f32) -> f32 {
    let (x, y) = local(x, y);
    let line = |a: [f32; 2], b: [f32; 2]| capsule(x, y, a, b, 0.0);
    if closed {
        let mut d = line([-4.0, 4.4], [6.0, 4.4]);
        for fx in [-3.4, 0.0, 3.4] {
            d = d.min(line([fx, -2.6], [fx, 4.4]));
        }
        d
    } else {
        let mut d = f32::MAX;
        // Each crease starts just below the lower of the two fingertips it separates.
        for ((fx, a), b) in [-3.4, 0.0, 3.4].into_iter().zip(TIPS).zip(TIPS.into_iter().skip(1)) {
            d = d.min(line([fx, a.max(b) + 1.0], [fx, 3.5]));
        }
        d
    }
}
