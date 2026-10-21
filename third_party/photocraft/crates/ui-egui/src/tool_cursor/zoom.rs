//! The Zoom tool's pointer as signed distance fields (negative inside), in logical pixels from the
//! hotspot with y pointing down: a magnifying glass with a plus, or a minus while ⌥ zooms out, as
//! in Photoshop. The hotspot is the centre of the lens. Only Windows needs it: winit maps `ZoomIn`
//! and `ZoomOut` to the plain arrow there (#3735), while macOS and the Linux themes draw their own.

use super::hand::capsule;

/// Half the side of the square the glass fits in, outline excluded.
pub(super) const HALF: f32 = 13.0;

const LENS: f32 = 6.5;
const HANDLE: ([f32; 2], [f32; 2], f32) = ([5.2, 5.2], [10.8, 10.8], 2.2);
/// Half the length of the plus and minus arms.
const ARM: f32 = 3.4;

/// The glass's silhouette (lens and handle): negative inside it.
pub(super) fn shape(x: f32, y: f32) -> f32 {
    let (a, b, r) = HANDLE;
    (x.hypot(y) - LENS).min(capsule(x, y, a, b, r))
}

/// The plus (or the minus, `out`) inside the lens: distance to its nearest arm.
pub(super) fn glyph(out: bool, x: f32, y: f32) -> f32 {
    let across = capsule(x, y, [-ARM, 0.0], [ARM, 0.0], 0.0);
    if out { across } else { across.min(capsule(x, y, [0.0, -ARM], [0.0, ARM], 0.0)) }
}
