//! Device health (#243): notices a lost wgpu device or an uncaptured wgpu error so callers stop
//! issuing GPU work and fall back to the CPU compositor instead of panicking.
//!
//! wgpu's default uncaptured-error handler panics ("Buffer 'pc_compose_uniforms' is invalid"
//! after the OS killed a huge submission), and `Device::poll` panics on a lost device whatever
//! the handler. [`DeviceHealth::watch`] replaces the handler with one that records the fault,
//! registers the device-lost callback, and [`DeviceHealth::wait`] polls only while the device is
//! healthy (catching the poll's panic as a last resort). Every GPU entry point checks
//! [`DeviceHealth::fault`] first.
//!
//! Builds for the web too: the callbacks only touch atomics and a mutex.

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

/// Why the GPU can't be used any more.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The device was lost (driver reset, the OS killed a submission, `Device::destroy`).
    Lost(String),
    /// An uncaptured wgpu error (out of memory, validation, internal): the GPU output can't be
    /// trusted, and on most drivers the device is about to be lost anyway.
    Error(String),
}

impl Fault {
    /// Whether the device itself was lost (rather than an error reported).
    pub fn is_lost(&self) -> bool {
        matches!(self, Fault::Lost(_))
    }

    /// The driver's or wgpu's message.
    pub fn detail(&self) -> &str {
        match self {
            Fault::Lost(s) | Fault::Error(s) => s,
        }
    }
}

impl fmt::Display for Fault {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fault::Lost(s) if s.is_empty() => f.write_str("GPU device lost"),
            Fault::Lost(s) => write!(f, "GPU device lost: {s}"),
            Fault::Error(s) => write!(f, "GPU error: {s}"),
        }
    }
}

#[derive(Default)]
struct Inner {
    faulted: AtomicBool,
    fault: Mutex<Option<Fault>>,
    errors: AtomicU64,
}

/// Shared health flag of one wgpu device. Cheap to clone; every clone sees the same state.
#[derive(Clone, Default)]
pub struct DeviceHealth(Arc<Inner>);

impl fmt::Debug for DeviceHealth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DeviceHealth").field("fault", &self.fault()).field("errors", &self.errors()).finish()
    }
}

impl DeviceHealth {
    /// A healthy flag not attached to a device (tests, CPU-only callers).
    pub fn new() -> Self {
        Self::default()
    }

    /// Watch `device`: its device-lost callback and uncaptured errors mark the returned flag
    /// instead of panicking. Replaces any handler set before (wgpu keeps one of each).
    pub fn watch(device: &wgpu::Device) -> Self {
        let health = Self::new();
        let lost = health.clone();
        // Called by wgpu with its own locks held: only record, never call back into wgpu.
        device.set_device_lost_callback(move |reason, message| {
            let detail = match (reason, message.is_empty()) {
                (wgpu::DeviceLostReason::Destroyed, true) => "device destroyed".to_string(),
                (_, true) => "lost by the driver".to_string(),
                (_, false) => message,
            };
            log::error!("wgpu device lost ({reason:?}): {detail}");
            lost.mark(Fault::Lost(detail));
        });
        let errors = health.clone();
        device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| errors.uncaptured(&e)));
        health
    }

    /// Record an uncaptured wgpu error.
    fn uncaptured(&self, e: &wgpu::Error) {
        self.0.errors.fetch_add(1, Ordering::Relaxed);
        let text = e.to_string();
        log::error!("uncaptured wgpu error: {text}");
        // Errors about invalid or lost objects follow a device loss whose callback may not have
        // run yet (it fires on the next poll).
        let lower = text.to_ascii_lowercase();
        let fault = if lower.contains("lost") || lower.contains("destroyed") { Fault::Lost(first_line(&text)) } else { Fault::Error(first_line(&text)) };
        self.mark(fault);
    }

    /// Mark the device unusable. The first fault is kept; a later device loss upgrades an error.
    pub fn mark(&self, fault: Fault) {
        let mut slot = self.0.fault.lock().unwrap_or_else(PoisonError::into_inner);
        let replace = match slot.as_ref() {
            None => true,
            Some(old) => !old.is_lost() && fault.is_lost(),
        };
        if replace {
            *slot = Some(fault);
        }
        self.0.faulted.store(true, Ordering::Release);
    }

    /// Whether the device may still be used.
    pub fn is_ok(&self) -> bool {
        !self.0.faulted.load(Ordering::Acquire)
    }

    /// Why the device can't be used, if it can't.
    pub fn fault(&self) -> Option<Fault> {
        if self.is_ok() {
            return None;
        }
        self.0.fault.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Uncaptured wgpu errors seen so far.
    pub fn errors(&self) -> u64 {
        self.0.errors.load(Ordering::Relaxed)
    }

    /// Block until `index` (or all submitted work) finished, unless the device is unusable.
    /// Returns whether the device is still healthy afterwards. `Device::poll` panics on a lost
    /// device, so a panic is caught and recorded as a loss (native only; the web has no blocking
    /// wait).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait(&self, device: &wgpu::Device, index: Option<wgpu::SubmissionIndex>) -> bool {
        if !self.is_ok() {
            return false;
        }
        let poll = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| device.poll(wgpu::PollType::Wait { submission_index: index, timeout: None })));
        match poll {
            Ok(Ok(_)) => {}
            Ok(Err(e)) => log::warn!("wgpu poll: {e}"),
            Err(_) => self.mark(Fault::Lost("the device stopped responding".into())),
        }
        self.is_ok()
    }

    /// Non-blocking poll (reclaims staging memory), skipped once the device is unusable.
    pub fn poll(&self, device: &wgpu::Device) {
        if !self.is_ok() {
            return;
        }
        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| device.poll(wgpu::PollType::Poll))).is_err() {
            self.mark(Fault::Lost("the device stopped responding".into()));
        }
    }
}

fn first_line(s: &str) -> String {
    let line = s.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let mut out: String = line.chars().take(200).collect();
    if line.chars().count() > 200 {
        out.push('…');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_fault_wins_but_loss_upgrades_an_error() {
        let h = DeviceHealth::new();
        assert!(h.is_ok());
        assert_eq!(h.fault(), None);
        h.mark(Fault::Error("out of memory".into()));
        assert!(!h.is_ok());
        let c = h.clone();
        c.mark(Fault::Error("second".into()));
        assert_eq!(h.fault(), Some(Fault::Error("out of memory".into())));
        c.mark(Fault::Lost("reset".into()));
        assert_eq!(h.fault(), Some(Fault::Lost("reset".into())));
        c.mark(Fault::Lost("later".into()));
        assert_eq!(h.fault().map(|f| f.to_string()).as_deref(), Some("GPU device lost: reset"));
    }

    #[test]
    fn fault_text() {
        assert_eq!(Fault::Lost(String::new()).to_string(), "GPU device lost");
        assert_eq!(Fault::Error("x".into()).to_string(), "GPU error: x");
        assert_eq!(first_line("\n  Buffer 'pc' is invalid\nmore"), "Buffer 'pc' is invalid");
        assert_eq!(first_line(&"a".repeat(300)).chars().count(), 201);
    }
}
