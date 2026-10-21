//! Pen tablet input that the windowing layer (winit 0.30) drops: pressure, tilt, barrel rotation
//! and which end of the pen is down (tip or eraser).
//!
//! - **macOS** ([`macos::Monitor`]): an AppKit local event monitor reads the tablet fields of
//!   each `NSEvent` (`pressure`, `tilt`, `rotation`, `subtype`, and `pointingDeviceType` from
//!   proximity events) before winit sees the event, so every pointer event the UI handles already
//!   has its pen sample. AppKit's block-based monitor API needs Objective-C interop, so
//!   `macos.rs` is the only module in the workspace that allows `unsafe` (craftrules
//!   never-crash: an isolated, tested helper crate).
//! - **Linux X11** ([`x11::spawn`]): a second, pure-Rust X connection (x11rb) selects XInput2
//!   raw events on the root window and maps the tablet device's valuators ("Abs Pressure",
//!   "Abs Tilt X/Y", "Abs Rotary Z") to samples. Raw events reach every client that asks, also
//!   while winit holds the pointer grab of a drag. No `unsafe`.
//! - **Wayland**: not covered. The `zwp_tablet_v2` protocol has to be bound on winit's own
//!   `wl_display` connection (the tablet events name winit's `wl_surface`), which needs
//!   `unsafe` foreign-display interop, and binding it makes compositors stop emulating the
//!   pointer for the pen, so pen motion would have to be re-injected into egui as well. Without
//!   it, compositors give the window no pen input at all, so the desktop app opens its window
//!   through Xwayland when a pen is attached, and the X11 reader applies.
//!
//! The platform glue only reads raw values and hands them to the pure mapping in [`appkit`] and
//! [`xi`], which normalise, clamp and track pen state and are tested on every platform.
//!
//! A sample is reported through a callback: `Some(sample)` while a pen is in use (hovering or
//! touching), `None` when the pointer is a mouse again.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod appkit;
pub mod xi;

#[cfg(target_os = "macos")]
pub mod macos;
#[cfg(target_os = "linux")]
pub mod x11;

/// One pen reading. Tilt is in degrees (-90..90, W3C Pointer Events convention: +x tilts to the
/// right, +y towards the user), rotation in degrees 0..360 (clockwise).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Tip (or eraser) pressure 0..1; 0 while hovering.
    pub pressure: f32,
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub rotation: f32,
    /// The eraser end of the pen is in use.
    pub eraser: bool,
}

impl Default for Sample {
    fn default() -> Self {
        Self { pressure: 1.0, tilt_x: 0.0, tilt_y: 0.0, rotation: 0.0, eraser: false }
    }
}

impl Sample {
    /// Clamp to the documented ranges and replace non-finite values (a NaN pressure reads as
    /// full pressure, like a mouse; NaN tilt and rotation as 0).
    pub fn sanitized(self) -> Self {
        let f = |v: f32, lo: f32, hi: f32, d: f32| if v.is_finite() { v.clamp(lo, hi) } else { d };
        let rotation = if self.rotation.is_finite() { self.rotation.rem_euclid(360.0) } else { 0.0 };
        Self {
            pressure: f(self.pressure, 0.0, 1.0, 1.0),
            tilt_x: f(self.tilt_x, -90.0, 90.0, 0.0),
            tilt_y: f(self.tilt_y, -90.0, 90.0, 0.0),
            // `rem_euclid` can round up to exactly 360 for tiny negative inputs.
            rotation: if rotation >= 360.0 { 0.0 } else { rotation },
            eraser: self.eraser,
        }
    }
}

/// What a raw platform event means for the current pen sample.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Update {
    /// Nothing to report (an event this module doesn't map).
    Keep,
    /// The new sample: `Some` for a pen, `None` for a mouse.
    Set(Option<Sample>),
}

/// Why tablet input could not be started.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Must be called on the main thread (AppKit).
    NotMainThread,
    /// The platform or display server doesn't offer what this needs.
    Unsupported(String),
    /// The platform call failed.
    Platform(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::NotMainThread => write!(f, "tablet input must be started on the main thread"),
            Error::Unsupported(why) => write!(f, "tablet input unsupported: {why}"),
            Error::Platform(why) => write!(f, "tablet input failed: {why}"),
        }
    }
}

impl std::error::Error for Error {}

/// Sanitize a sample and run a user callback with it, without letting a panic unwind into the platform's event loop (an unwind
/// through an Objective-C block or a detached thread would abort or silently stop input).
pub fn deliver(callback: &dyn Fn(Option<Sample>), sample: Option<Sample>) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(sample.map(Sample::sanitized))));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitized_clamps_and_replaces_non_finite() {
        let s = Sample { pressure: 1.7, tilt_x: -120.0, tilt_y: f32::NAN, rotation: -90.0, eraser: true }.sanitized();
        assert_eq!(s, Sample { pressure: 1.0, tilt_x: -90.0, tilt_y: 0.0, rotation: 270.0, eraser: true });
        let s = Sample { pressure: f32::NAN, tilt_x: f32::INFINITY, tilt_y: f32::NEG_INFINITY, rotation: f32::INFINITY, eraser: false }.sanitized();
        assert_eq!(s, Sample::default());
        let s = Sample { pressure: -0.5, rotation: -1e-9, ..Default::default() }.sanitized();
        assert_eq!((s.pressure, s.rotation), (0.0, 0.0));
        assert!(Sample { rotation: 725.0, ..Default::default() }.sanitized().rotation == 5.0);
    }

    #[test]
    fn deliver_contains_a_panicking_callback() {
        let seen = std::cell::Cell::new(None);
        deliver(&|s| seen.set(s), Some(Sample { pressure: 3.0, ..Default::default() }));
        assert_eq!(seen.get().map(|s| s.pressure), Some(1.0), "sanitized on the way out");
        deliver(&|_| panic!("boom"), None);
    }
}
