//! Cooperative cancellation and progress for long operations (background jobs, #210).
//!
//! Heavy algorithms take an [`Interrupt`] and check [`Interrupt::cancelled`] once per tile, row
//! band or iteration, returning early (typically `None`) when it is set. They may also report
//! [`Interrupt::progress`] in `0.0..=1.0`. [`Interrupt::NONE`] never cancels and ignores progress,
//! so synchronous callers pass it and get the old behaviour.
//!
//! The callbacks are `Sync`, so rayon workers can check them while filtering tiles in parallel.

/// Cancellation check plus progress sink for one long operation. Cheap to copy.
#[derive(Clone, Copy)]
pub struct Interrupt<'a> {
    cancel: &'a (dyn Fn() -> bool + Sync),
    progress: &'a (dyn Fn(f32) + Sync),
}

fn never() -> bool {
    false
}

fn ignore(_: f32) {}

impl std::fmt::Debug for Interrupt<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interrupt").field("cancelled", &self.cancelled()).finish()
    }
}

impl Default for Interrupt<'_> {
    fn default() -> Self {
        Interrupt::NONE
    }
}

impl<'a> Interrupt<'a> {
    /// Never cancelled; progress is ignored.
    pub const NONE: Interrupt<'static> = Interrupt { cancel: &never, progress: &ignore };

    /// An interrupt from a cancellation check and a progress sink (`0.0..=1.0`).
    pub fn new(cancel: &'a (dyn Fn() -> bool + Sync), progress: &'a (dyn Fn(f32) + Sync)) -> Self {
        Self { cancel, progress }
    }

    /// Only a cancellation check (progress is ignored).
    pub fn cancel_only(cancel: &'a (dyn Fn() -> bool + Sync)) -> Self {
        Self { cancel, progress: &ignore }
    }

    /// Has the operation been cancelled? Algorithms stop as soon as they see `true`.
    pub fn cancelled(&self) -> bool {
        (self.cancel)()
    }

    /// Report progress (`0.0..=1.0`; values outside are clamped, NaN is ignored).
    pub fn progress(&self, fraction: f32) {
        if fraction.is_finite() {
            (self.progress)(fraction.clamp(0.0, 1.0));
        }
    }

    /// `Err(Cancelled)` once cancelled, for `?` in loops.
    pub fn check(&self) -> Result<(), Cancelled> {
        if self.cancelled() { Err(Cancelled) } else { Ok(()) }
    }
}

/// Returned by an operation that stopped because its [`Interrupt`] was cancelled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("cancelled")
    }
}

impl std::error::Error for Cancelled {}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    use super::*;

    #[test]
    fn none_never_cancels() {
        assert!(!Interrupt::NONE.cancelled());
        Interrupt::NONE.progress(0.5);
        assert_eq!(Interrupt::NONE.check(), Ok(()));
    }

    #[test]
    fn reports_cancel_and_clamped_progress() {
        let flag = AtomicBool::new(false);
        let last = AtomicU32::new(0);
        let cancel = || flag.load(Ordering::Relaxed);
        let progress = |f: f32| last.store(f.to_bits(), Ordering::Relaxed);
        let i = Interrupt::new(&cancel, &progress);
        i.progress(2.0);
        assert_eq!(f32::from_bits(last.load(Ordering::Relaxed)), 1.0);
        i.progress(f32::NAN);
        assert_eq!(f32::from_bits(last.load(Ordering::Relaxed)), 1.0);
        assert!(!i.cancelled());
        flag.store(true, Ordering::Relaxed);
        assert!(i.cancelled());
        assert_eq!(i.check(), Err(Cancelled));
    }
}
