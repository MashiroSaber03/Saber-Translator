//! Photocraft's type engine.
//!
//! * [`fonts::FontDb`]: bundled fonts (Inter, JetBrains Mono; always available, including on the
//!   web), optional system fonts from platform directory scans plus Fontconfig's Linux file list,
//!   user font data, TrueType collections, PostScript-name lookup. Builds made with the optional
//!   craft-fonts input
//!   (`CRAFT_FONTS_DIR`, see [`craft_fonts`]) also carry its Japanese fonts as fallbacks.
//! * [`layout`]: shaping and line layout with [parley] (HarfRust shaping, bidi, line breaking),
//!   point and paragraph (box) text, per-run styles, Photoshop leading/indent/spacing rules.
//! * [`render`]: anti-aliased rasterization (exact-area accumulation, f32 coverage) into a
//!   [`photocraft_raster::Surface`] at any depth and colour model.
//! * [`psd`]: the PSD `TySh` block and its `EngineData` (text, fonts, runs, paragraphs, box).
//!
//! The usual entry point is [`TextEngine::render_layer`], which refreshes a
//! [`photocraft_doc::TextLayer`]'s cache from its style model.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod cjk;
pub mod craft_fonts;
pub mod engine_data;
pub mod fonts;
pub mod glyphs;
pub mod layout;
pub mod optical;
pub mod psd;
pub mod psd_styles;
pub mod raster;
pub mod render;
pub mod scripts;
pub mod served;
pub mod spell;
pub mod warp;

use photocraft_color::PixelFormat;
use photocraft_doc::TextLayer;

pub use craft_fonts::{CRAFT_FONTS, CraftFont};
pub use fonts::{FaceInfo, FontDb, ResolvedFont};
pub use layout::{
    ClusterInfo, LineInfo, PlacedGlyph, TextLayout, byte_index, char_index, hit_char, line_edge, line_index, line_step, text_point_inside, word_boundary,
};
pub use render::Rendered;

/// Font database + layout context. Create once and reuse (font loading and shaping caches).
pub struct TextEngine {
    pub fonts: FontDb,
    layouter: layout::Layouter,
}

impl Default for TextEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl TextEngine {
    /// Bundled fonts only: deterministic output (tests, web).
    pub fn new() -> Self {
        Self { fonts: FontDb::new(), layouter: layout::Layouter::new() }
    }

    /// Bundled plus installed system fonts (native desktop).
    pub fn with_system_fonts() -> Self {
        Self { fonts: FontDb::with_system_fonts(), layouter: layout::Layouter::new() }
    }

    /// Lays out a text layer (text space: pixels, before `layer.transform`).
    pub fn layout(&mut self, layer: &TextLayer, dpi: f32) -> TextLayout {
        self.layouter.layout(&mut self.fonts, layer, dpi)
    }

    /// Kerning (1/1000 em) between the character starting at byte `at` and the next one, as
    /// laid out: the manual kern, or the automatic (metrics / optical) kerning of the pair.
    /// `None` when there is no such pair on one line.
    pub fn pair_kerning(&mut self, layer: &TextLayer, dpi: f32, at: usize) -> Option<f32> {
        let runs = layer.char_runs();
        let ch = layer.text.get(at..)?.chars().next()?;
        let next = at + ch.len_utf8();
        if ch == '\n' || ch == '\r' || layer.text.get(next..)?.chars().next().is_none_or(|c| c == '\n' || c == '\r') {
            return None;
        }
        let gap = |l: &TextLayout| {
            let a = l.clusters.iter().find(|c| c.range.start == at)?;
            let b = l.clusters.iter().find(|c| c.range.start == next)?;
            (a.line == b.line && !a.rtl).then_some(b.x - a.x)
        };
        let with = gap(&self.layout(layer, dpi))?;
        // Em of the pair's first character (px).
        let mut acc = 0;
        let style_run = runs.iter().find(|r| {
            acc += r.len;
            at < acc
        })?;
        let em = style_run.style.size_pt * if dpi > 0.0 { dpi / 72.0 } else { 1.0 };
        if em.is_nan() || em <= 0.0 {
            return None;
        }
        // The same layer with the pair's first character unkerned.
        let mut plain = layer.clone();
        plain.runs = Vec::with_capacity(runs.len() + 2);
        let mut start = 0;
        for r in runs {
            let end = start + r.len;
            let mut push = |s: usize, e: usize, unkern: bool| {
                if e > s {
                    let mut style = r.style.clone();
                    if unkern {
                        style.kerning = photocraft_doc::text::Kerning::Off;
                        style.kern = 0.0;
                    }
                    plain.runs.push(photocraft_doc::text::TextRun { len: e - s, style });
                }
            };
            push(start, at.clamp(start, end), false);
            push(at.clamp(start, end), next.clamp(start, end), true);
            push(next.clamp(start, end), end, false);
            start = end;
        }
        let without = gap(&self.layout(&plain, dpi))?;
        let k = (with - without) / em * 1000.0;
        k.is_finite().then_some(k)
    }

    /// Lays out and rasterizes a text layer into document space.
    pub fn render(&mut self, layer: &TextLayer, dpi: f32, format: PixelFormat) -> (TextLayout, Rendered) {
        let l = self.layout(layer, dpi);
        let warp = render::layout_deform(&l, layer);
        let r = render::rasterize_warped(&l, &layer.transform, format, layer.antialias, warp.as_ref());
        (l, r)
    }

    /// Re-renders `layer.cache` from its model. Returns the document-space rectangle drawn.
    pub fn render_layer(&mut self, layer: &mut TextLayer, dpi: f32, format: PixelFormat) -> photocraft_geom::Rect {
        let (_, r) = self.render(layer, dpi, format);
        layer.cache = Some(r.surface);
        r.rect
    }
}

static SHARED: warm::Warm<std::sync::Mutex<TextEngine>> = warm::Warm::new();

/// Whether [`shared`] scans the installed fonts (native builds, outside tests, unless
/// `PHOTOCRAFT_NO_SYSTEM_FONTS` is set). Without the scan it is cheap to build.
fn uses_system_fonts() -> bool {
    cfg!(not(target_arch = "wasm32")) && !cfg!(test) && std::env::var_os("PHOTOCRAFT_NO_SYSTEM_FONTS").is_none()
}

fn new_shared() -> std::sync::Mutex<TextEngine> {
    let mut engine = if uses_system_fonts() { TextEngine::with_system_fonts() } else { TextEngine::new() };
    engine.fonts.mark_shared();
    std::sync::Mutex::new(engine)
}

/// A process-wide engine for callers without their own (bundled fonts only on the web; bundled
/// plus system fonts elsewhere). Locking is cheap relative to layout. The first call scans the
/// installed fonts and blocks until that is done (or until [`warm_shared`]'s scan finishes), so
/// code that runs every UI frame uses [`try_shared`] instead.
pub fn shared() -> &'static std::sync::Mutex<TextEngine> {
    SHARED.get_or_init(new_shared)
}

/// [`shared`] without blocking: `None` while the system font scan is still running on its
/// background thread (started here if [`warm_shared`] has not started it yet). Without system
/// fonts (web, tests) the engine is cheap and this always returns it.
pub fn try_shared() -> Option<&'static std::sync::Mutex<TextEngine>> {
    SHARED.try_get(new_shared, uses_system_fonts())
}

/// Preferences › Type › Enable Missing Glyph Protection for [`shared`]: 0 = not set yet,
/// 1 = off, 2 = on. Read under the engine's lock, so the newest setting always wins.
static WANT_GLYPH_PROTECTION: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
/// A thread is already waiting for the font scan to apply [`WANT_GLYPH_PROTECTION`].
static GLYPH_PROTECTION_WAITING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn apply_glyph_protection(engine: &std::sync::Mutex<TextEngine>) {
    let mut e = engine.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    match WANT_GLYPH_PROTECTION.load(std::sync::atomic::Ordering::SeqCst) {
        1 => e.fonts.set_missing_glyph_protection(false),
        2 => e.fonts.set_missing_glyph_protection(true),
        _ => {}
    }
}

/// Sets [`shared`]'s missing-glyph protection without waiting for the system font scan (#4519):
/// applied now when the engine is built, otherwise by a helper thread once the scan finishes.
/// Called from the UI thread at launch and on every preferences change, where [`shared`] would
/// block until the scan is done.
pub fn set_shared_missing_glyph_protection(on: bool) {
    WANT_GLYPH_PROTECTION.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::SeqCst);
    if let Some(engine) = try_shared() {
        apply_glyph_protection(engine);
        return;
    }
    if GLYPH_PROTECTION_WAITING.swap(true, std::sync::atomic::Ordering::SeqCst) {
        // The waiting thread reads the newest setting when it applies it.
        return;
    }
    let spawned = std::thread::Builder::new().name("photocraft-glyph-protection".into()).spawn(|| apply_glyph_protection(shared()));
    if spawned.is_err() {
        // No thread: apply it here, as before.
        GLYPH_PROTECTION_WAITING.store(false, std::sync::atomic::Ordering::SeqCst);
        apply_glyph_protection(shared());
    }
}

/// Starts building [`shared`] (the system font scan) on a background thread, so the Type tool's
/// font menus don't stall the UI when they first draw. Call once at startup; calling again does
/// nothing.
pub fn warm_shared() {
    let _ = try_shared();
}

mod warm {
    use std::sync::{Mutex, OnceLock, PoisonError};
    use std::thread::JoinHandle;

    /// A lazily built value whose (slow) construction can run on a background thread.
    pub(crate) struct Warm<T: 'static> {
        cell: OnceLock<T>,
        scan: Mutex<Option<JoinHandle<()>>>,
    }

    impl<T: Send + Sync + 'static> Warm<T> {
        pub(crate) const fn new() -> Self {
            Self { cell: OnceLock::new(), scan: Mutex::new(None) }
        }

        /// The value, building it on this thread (or waiting for the background build).
        pub(crate) fn get_or_init(&'static self, init: fn() -> T) -> &'static T {
            self.cell.get_or_init(init)
        }

        /// The value if it is built. With `background`, the first call starts building it on a
        /// worker thread and returns `None` until that finishes; otherwise (or when no thread can
        /// be started, or the worker died) it is built here.
        pub(crate) fn try_get(&'static self, init: fn() -> T, background: bool) -> Option<&'static T> {
            if let Some(v) = self.cell.get() {
                return Some(v);
            }
            if !background {
                return Some(self.get_or_init(init));
            }
            let mut scan = self.scan.lock().unwrap_or_else(PoisonError::into_inner);
            match scan.as_ref() {
                Some(worker) if !worker.is_finished() => None,
                // Finished: the value is set, or the worker panicked and this thread retries.
                Some(_) => {
                    drop(scan);
                    Some(self.get_or_init(init))
                }
                None => match std::thread::Builder::new().name("photocraft-font-scan".into()).spawn(move || {
                    self.get_or_init(init);
                }) {
                    Ok(worker) => {
                        *scan = Some(worker);
                        self.cell.get()
                    }
                    Err(_) => {
                        drop(scan);
                        Some(self.get_or_init(init))
                    }
                },
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::Warm;
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::time::{Duration, Instant};

        static OPEN: AtomicBool = AtomicBool::new(false);

        /// A slow build: waits until the test opens the gate (at most 2 s).
        fn slow() -> u32 {
            let start = Instant::now();
            while !OPEN.load(Ordering::SeqCst) && start.elapsed() < Duration::from_secs(2) {
                std::thread::sleep(Duration::from_millis(1));
            }
            7
        }

        #[test]
        fn try_get_does_not_block_while_the_background_build_runs() {
            static W: Warm<u32> = Warm::new();
            let start = Instant::now();
            assert_eq!(W.try_get(slow, true), None, "the first call starts the build and returns at once");
            assert_eq!(W.try_get(slow, true), None, "still building");
            assert!(start.elapsed() < Duration::from_secs(1), "try_get blocked for {:?}", start.elapsed());
            OPEN.store(true, Ordering::SeqCst);
            let deadline = Instant::now() + Duration::from_secs(10);
            let v = loop {
                if let Some(v) = W.try_get(slow, true) {
                    break *v;
                }
                assert!(Instant::now() < deadline, "the background build never finished");
                std::thread::sleep(Duration::from_millis(1));
            };
            assert_eq!(v, 7);
            assert_eq!(W.get_or_init(slow), &7);
        }
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod vertical_tests;
