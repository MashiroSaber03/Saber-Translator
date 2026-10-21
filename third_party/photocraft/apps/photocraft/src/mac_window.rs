//! macOS window lifecycle: title bar dragging and quitting.
//!
//! Dragging: the window draws under a transparent title bar (`main.rs`), so AppKit would
//! otherwise drag the window from anywhere in that strip, menus included: pressing a menu title
//! and dragging down to an item moved the window instead. Turning AppKit's own dragging off leaves
//! only the app's drag region (the free gap between the menus and the controls,
//! `panels::title_bar`), which starts a drag explicitly (`ViewportCommand::StartDrag`, AppKit's
//! `performWindowDragWithEvent:`, which works on an unmovable window).
//!
//! Quitting: see [`terminate_later`]. All safe objc2 calls.

use objc2::MainThreadMarker;
use objc2_app_kit::NSApplication;

/// Stop AppKit dragging the app's windows by their title bar strip.
pub fn disable_native_title_drag() {
    let Some(mtm) = MainThreadMarker::new() else {
        log::warn!("title bar: not on the main thread; AppKit window dragging left on");
        return;
    };
    for window in NSApplication::sharedApplication(mtm).windows().iter() {
        window.setMovable(false);
    }
}

/// Quit the way a Mac app does, with `-[NSApplication terminate:]` (`Services::quit`), instead of
/// letting eframe close the window. eframe drops the window while AppKit's run loop is still
/// running, and on Macs with a Touch Bar AppKit's next display cycle then tries to stop observing
/// a view that is gone: an uncaught exception that crashes (macOS 27) or leaves a windowless app
/// that only Force Quit ends (macOS 26) (#1458, #1575). `terminate:` ends the process before
/// another display cycle runs; winit's `applicationWillTerminate:` closes the windows and has
/// eframe save the window layout first, as the old ⌘Q (winit's default menu) did.
///
/// Scheduled on the main queue: winit has to be out of its event handler, or eframe's exit
/// handling (and the save) is dropped.
pub fn terminate_later() {
    dispatch2::DispatchQueue::main().exec_async(|| match MainThreadMarker::new() {
        Some(mtm) => NSApplication::sharedApplication(mtm).terminate(None),
        None => log::warn!("quit: the main queue ran off the main thread; not terminating"),
    });
}
