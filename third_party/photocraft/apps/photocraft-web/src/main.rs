//! Photocraft in the browser.
//!
//! Runs the same [`photocraft_ui_egui::PhotocraftApp`] as the desktop app through eframe's web
//! runner (wgpu: WebGPU where available, WebGL2 otherwise). Build with `trunk build --release`
//! from this directory; see `docs/development.md` ("Web build").
//!
//! Differences from the desktop app:
//! - no TCP control server (browsers can't listen on sockets);
//! - File → Open uses the browser file picker; bytes arrive asynchronously through
//!   `Services::inbox`;
//! - saving/exporting triggers a browser download;
//! - dropped files are read asynchronously by `web::WebShell` and delivered through the inbox.
//!
//! URL query flags: `?cpu` forces the CPU canvas path (same as `PHOTOCRAFT_CPU_CANVAS=1`);
//! `?webgl` forces the WebGL2 backend instead of WebGPU.

#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(any(target_arch = "wasm32", test))]
mod preset_bridge;

#[cfg(target_arch = "wasm32")]
mod indexed_presets;

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(target_arch = "wasm32")]
mod saber_bridge;

#[cfg(target_arch = "wasm32")]
fn main() {
    web::start();
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    eprintln!("photocraft-web only runs in the browser: build it with `trunk build --release` in apps/photocraft-web");
}

#[cfg(test)]
mod tests {
    /// A WebAssembly module that never downloads (or doesn't match the page) runs no Rust, so the
    /// page itself replaces the endless "Loading" with what may have gone wrong and a way to
    /// retry, without a hand-written script.
    #[test]
    fn loading_page_offers_a_reload_when_startup_stalls() {
        let html = include_str!("../index.html");
        let (_, hint) = html.split_once("<p id=\"photocraft_stalled\">").expect("stalled-startup hint in the loading element");
        assert!(hint.contains("failed to") && hint.contains("<a href=\"\">Reload</a>"), "{hint}");
        assert!(html.contains("animation: photocraft-stalled"), "the hint appears after a delay");
        assert!(!html.contains("<script"), "trunk injects the generated loader; the page has no script of its own");
    }
}
