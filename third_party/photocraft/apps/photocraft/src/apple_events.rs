//! macOS open-documents and quit Apple events.
//!
//! Finder double-clicks, Open With, drops on the Dock icon and `open -a PhotoCraft file.psd` don't
//! pass paths on the command line: LaunchServices sends the running (or just-launched) app a
//! `kAEOpenDocuments` ('odoc') Apple event. winit 0.30 doesn't handle it and owns the
//! `NSApplicationDelegate`, and handling it ourselves needs Objective-C class declarations, i.e.
//! `unsafe`, which this workspace forbids. The audited `fmv-macos-events` crate wraps exactly that
//! (an `NSAppleEventManager` handler registered before Finder's launch event, leaving winit's
//! delegate alone) behind a safe main-thread API; this module adapts it to [`OsEvent`]s that the UI
//! polls every frame.

use fmv_macos_events::{Event, Inbox, Registration};
use photocraft_ui_egui::OsEvent;

/// Keeps the Apple-event handlers registered; hold it until the event loop returns.
pub struct AppleEvents {
    _registration: Registration,
    inbox: Inbox,
}

impl AppleEvents {
    /// Register the handlers. Call on the main thread before the event loop starts, so the event
    /// that launched the app (Finder double-click) is caught too.
    pub fn install() -> Self {
        let (registration, inbox) = Registration::install();
        Self { _registration: registration, inbox }
    }

    /// The queue the UI drains (see `Services::os_events`); events arriving later wake `ctx`.
    pub fn connect(&self, ctx: &egui::Context) -> photocraft_ui_egui::OsEventsFn {
        let ctx = ctx.clone();
        self.inbox.set_wake(move || ctx.request_repaint());
        let inbox = self.inbox.clone();
        Box::new(move || {
            inbox
                .drain()
                .into_iter()
                .map(|e| match e {
                    Event::Open(paths) => OsEvent::Open(paths.into_iter().map(|p| p.to_string_lossy().into_owned()).collect()),
                    Event::Quit => OsEvent::Quit,
                })
                .collect()
        })
    }
}
