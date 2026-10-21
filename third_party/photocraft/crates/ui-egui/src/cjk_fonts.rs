//! Lazy CJK and other-script fallback fonts for the UI.
//!
//! The bundled Inter / JetBrains Mono have no Japanese, Chinese or Korean glyphs, and the OS
//! fonts that do are large (Hiragino ~10 MB, Apple SD Gothic Neo 28 MB, PingFang 78 MB, Noto
//! Sans CJK ~20 MB). Instead of reading them at startup, an egui plugin scans each frame's text
//! for CJK characters no registered font covers, and registers the next system font for that
//! character's script (`ctx.add_font`, active from the next frame, which it requests). Fonts are
//! appended at the lowest priority to every family, so Latin text keeps Inter.
//!
//! Other scripts the bundled fonts lack (Thai, Arabic, Hebrew, Devanagari and the other Indic
//! scripts, ...: [`photocraft_text::scripts`]) work the same way, each with its own attempt state,
//! independent of the CJK order: the first installed candidate file whose character map really
//! has the missing character is registered, once per script. egui shapes each font run with
//! HarfRust, so Thai marks and Arabic joining come from the font; it has no bidi reordering yet.
//!
//! Script order follows the UI locale ([`photocraft_text::cjk::script_order`]): Kana prefers a
//! Japanese font, Hangul a Korean one, Bopomofo a Traditional Chinese one, and Han the
//! locale's script (Japanese forms only for a Japanese locale). At most one font is read per
//! frame, each file at most once, and once every script has been tried the scan stops.
//!
//! Builds made with the optional craft-fonts input (`CRAFT_FONTS_DIR`,
//! [`photocraft_text::craft_fonts`]) carry Japanese fonts (BIZ UDPGothic first): they are tried
//! before the system Japanese fonts, in the Japanese slot of the same locale order. Without
//! craft-fonts (and on the web, which never embeds them) nothing changes.

use egui::epaint::text::{FontInsert, FontPriority, InsertFontFamily};
use egui::{FontData, FontFamily, FontId, Shape};
use photocraft_text::cjk::{self, CjkChar, CjkScript, FontFile};
use photocraft_text::craft_fonts::{self, CraftFont};
use photocraft_text::scripts::{self, Script};
use std::path::PathBuf;

/// Skip absurdly large files (a corrupt or non-font path must not eat memory).
pub const MAX_FONT_BYTES: u64 = 128 << 20;
/// At most this many candidate files are read for one script (bounded I/O on a miss).
const MAX_SCRIPT_READS: usize = 6;
/// Name prefix of the registered fallback fonts.
pub const FONT_PREFIX: &str = "system-cjk";

/// Where fonts come from; swapped out in tests.
#[derive(Clone)]
pub struct Sources {
    pub locale: fn() -> Option<String>,
    pub files: fn(CjkScript) -> Vec<FontFile>,
    pub last_resort: fn() -> Vec<FontFile>,
    /// Candidate files for a non-CJK script, most preferred first; read only when a visible
    /// character of that script is missing.
    pub scripts: fn(Script) -> Vec<PathBuf>,
    /// The face of a font file whose character map has the character (`None`: not this file).
    pub covers: fn(&[u8], char) -> Option<u32>,
    /// Embedded fonts tried before `files` for a script (craft-fonts' Japanese fonts).
    pub embedded: fn(CjkScript) -> Vec<&'static CraftFont>,
}

/// craft-fonts' Japanese fonts for the Japanese script, UI face first (empty without craft-fonts).
pub fn craft_embedded(script: CjkScript) -> Vec<&'static CraftFont> {
    if script == CjkScript::Japanese { craft_fonts::japanese_for_ui() } else { Vec::new() }
}

/// No embedded fonts (tests that pin the system-font behaviour).
pub fn no_embedded(_: CjkScript) -> Vec<&'static CraftFont> {
    Vec::new()
}

impl Sources {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn system() -> Self {
        cjk::start_fontconfig_cjk_scan();
        Self {
            locale: || match crate::i18n::current().code() {
                code @ ("ja" | "ko" | "zh-hans" | "zh-hant") => Some(code.to_string()),
                _ => cjk::ui_locale().map(str::to_string),
            },
            files: cjk::font_files,
            last_resort: cjk::last_resort_files,
            scripts: scripts::system_candidates,
            covers: scripts::covering_face,
            embedded: craft_embedded,
        }
    }
    #[cfg(target_arch = "wasm32")]
    pub fn system() -> Self {
        Self {
            locale: || None,
            files: |_| Vec::new(),
            last_resort: Vec::new,
            scripts: |_| Vec::new(),
            covers: scripts::covering_face,
            embedded: craft_embedded,
        }
    }
}

/// Headless test frame pumps do not wait for `request_repaint_after`, so let the optional
/// Fontconfig worker finish before asserting on installed system fonts.
#[cfg(all(test, target_os = "linux"))]
pub(crate) fn wait_for_fontconfig_cjk_scan() {
    cjk::start_fontconfig_cjk_scan();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while cjk::fontconfig_cjk_scan_pending() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(!cjk::fontconfig_cjk_scan_pending(), "Fontconfig CJK lookup did not finish within 10 seconds");
}

/// The lazy loader's state: which scripts were tried and which files were read.
pub struct CjkFallback {
    sources: Sources,
    /// set_fonts takes effect next pass; do not collide with a previous loader's font names.
    defer_after_reset: bool,
    order: Option<[CjkScript; 4]>,
    tried: Vec<CjkScript>,
    last_resort_tried: bool,
    /// Non-CJK scripts whose candidates were tried (each at most once per install).
    scripts_tried: Vec<Script>,
    loaded: Vec<PathBuf>,
    /// Fonts registered so far: (name, path, face index).
    pub registered: Vec<(String, PathBuf, u32)>,
}

impl CjkFallback {
    pub fn new(sources: Sources) -> Self {
        Self {
            sources,
            defer_after_reset: false,
            order: None,
            tried: Vec::new(),
            last_resort_tried: false,
            scripts_tried: Vec::new(),
            loaded: Vec::new(),
            registered: Vec::new(),
        }
    }

    /// Script order for the UI locale (resolved on first use, not at startup).
    pub fn order(&mut self) -> [CjkScript; 4] {
        *self.order.get_or_insert_with(|| cjk::script_order((self.sources.locale)().as_deref()))
    }

    /// Every script and the last-resort fonts have been tried: nothing more to load.
    pub fn exhausted(&self) -> bool {
        self.tried.len() >= 4 && self.last_resort_tried
    }

    /// The next font to register for a character of kind `kind` that no current font covers:
    /// the first readable file of the first untried script (the character's preferred script,
    /// then the locale order), then the last-resort fonts. `None` when nothing is left.
    pub fn next_font(&mut self, kind: CjkChar) -> Option<(String, FontData)> {
        let order = self.order();
        let scripts: Vec<CjkScript> = std::iter::once(kind.preferred(&order)).chain(order).collect();
        for s in scripts {
            if self.tried.contains(&s) {
                continue;
            }
            if let Some(f) = self.load_embedded(s) {
                self.tried.push(s);
                return Some(f);
            }
            let fontconfig_pending = cjk::fontconfig_cjk_scan_pending();
            let files = (self.sources.files)(s);
            if let Some(f) = self.load_first(&files) {
                self.tried.push(s);
                return Some(f);
            }
            if fontconfig_pending {
                return None;
            }
            self.tried.push(s);
        }
        if !self.last_resort_tried {
            self.last_resort_tried = true;
            let files = (self.sources.last_resort)();
            return self.load_first(&files);
        }
        None
    }

    /// Every non-CJK script has been tried.
    pub fn scripts_exhausted(&self) -> bool {
        self.scripts_tried.len() >= Script::ALL.len()
    }

    /// The first candidate font for `c`'s script that really maps `c`; the script is tried once.
    pub fn next_script_font(&mut self, c: char) -> Option<(String, FontData)> {
        let script = scripts::classify(c)?;
        if self.scripts_tried.contains(&script) {
            return None;
        }
        self.scripts_tried.push(script);
        let mut reads = 0;
        for path in (self.sources.scripts)(script) {
            if reads >= MAX_SCRIPT_READS {
                break;
            }
            if self.loaded.contains(&path) {
                continue;
            }
            let Some(bytes) = read_font(&path) else { continue };
            reads += 1;
            let Some(index) = (self.sources.covers)(&bytes, c) else { continue };
            self.loaded.push(path.clone());
            let name = format!("{FONT_PREFIX}-{}", self.registered.len());
            log::info!("UI font fallback: registered {} (face {index}) for {script:?} as {name}", path.display());
            self.registered.push((name.clone(), path, index));
            let mut data = FontData::from_owned(bytes);
            data.index = index;
            return Some((name, data));
        }
        log::info!("UI font fallback: no installed font covers {script:?}");
        None
    }

    /// The first embedded (craft-fonts) font for `s`, registered from its static bytes.
    fn load_embedded(&mut self, s: CjkScript) -> Option<(String, FontData)> {
        let f = (self.sources.embedded)(s).into_iter().find(|f| !f.bytes.is_empty())?;
        let path = PathBuf::from(format!("craft-fonts/{} {}", f.family, f.style));
        if self.loaded.contains(&path) {
            return None;
        }
        self.loaded.push(path.clone());
        let name = format!("{FONT_PREFIX}-{}", self.registered.len());
        log::info!("UI font fallback: registered embedded {} as {name}", path.display());
        self.registered.push((name.clone(), path, 0));
        Some((name, FontData::from_static(f.bytes)))
    }

    fn load_first(&mut self, files: &[FontFile]) -> Option<(String, FontData)> {
        for f in files {
            if self.loaded.contains(&f.path) {
                continue;
            }
            let Some(bytes) = read_font(&f.path) else { continue };
            let index = cjk::face_index_for_family(&bytes, &f.family);
            self.loaded.push(f.path.clone());
            let name = format!("{FONT_PREFIX}-{}", self.registered.len());
            log::info!("UI font fallback: registered {} (face {index}) as {name}", f.path.display());
            self.registered.push((name.clone(), f.path.clone(), index));
            let mut data = FontData::from_owned(bytes);
            data.index = index;
            return Some((name, data));
        }
        None
    }
}

/// A font file's bytes, or `None` for missing, empty, oversized or unreadable files.
fn read_font(path: &std::path::Path) -> Option<Vec<u8>> {
    let meta = std::fs::metadata(path).ok()?;
    if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_FONT_BYTES {
        return None;
    }
    std::fs::read(path).ok()
}

/// First supported missing character for which a fallback has not yet been exhausted.
fn missing_char(ctx: &egui::Context, shapes: &[egui::epaint::ClippedShape], fallback: &CjkFallback) -> Option<char> {
    fn collect(shape: &Shape, out: &mut Vec<char>, fallback: &CjkFallback, cjk_pending: bool) {
        let wanted = |c: char| (cjk_pending && cjk::classify(c).is_some()) || scripts::classify(c).is_some_and(|s| !fallback.scripts_tried.contains(&s));
        match shape {
            Shape::Text(t) => {
                let text = &t.galley.job.text;
                if !text.is_ascii() {
                    for c in text.chars().filter(|c| wanted(*c)) {
                        if out.len() >= 256 {
                            return;
                        }
                        if !out.contains(&c) {
                            out.push(c);
                        }
                    }
                }
            }
            Shape::Vec(v) => v.iter().for_each(|s| collect(s, out, fallback, cjk_pending)),
            _ => {}
        }
    }
    let mut chars = Vec::new();
    for s in shapes {
        collect(&s.shape, &mut chars, fallback, !fallback.exhausted());
    }
    if chars.is_empty() {
        return None;
    }
    let font = FontId::proportional(12.0);
    ctx.fonts_mut(|f| chars.into_iter().find(|c| !f.has_glyph(&font, *c)))
}

/// Big-endian readers over a font's bytes; `None` past the end.
fn be_u16(font: &[u8], o: usize) -> Option<u16> {
    font.get(o..o.checked_add(2)?).map(|b| u16::from_be_bytes([b[0], b[1]]))
}

fn be_u32(font: &[u8], o: usize) -> Option<u32> {
    font.get(o..o.checked_add(4)?).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

/// The offset of table `tag` in face `index` of `font` (a collection or a single face).
fn table_offset(font: &[u8], index: u32, tag: &[u8; 4]) -> Option<usize> {
    let dir = if font.get(..4)? == b"ttcf" {
        // Collection: the offset table of face `index` follows a 12-byte header.
        be_u32(font, 12usize.checked_add((index as usize).checked_mul(4)?)?)? as usize
    } else {
        0
    };
    let tables = be_u16(font, dir.checked_add(4)?)? as usize;
    (0..tables).find_map(|i| {
        let rec = dir.checked_add(12)?.checked_add(i.checked_mul(16)?)?;
        (font.get(rec..rec.checked_add(4)?)? == tag).then_some(())?;
        Some(be_u32(font, rec.checked_add(8)?)? as usize)
    })
}

/// (ascent, descent, line gap) of a face in em, from its `hhea` table; `descent` is negative.
fn vertical_metrics(font: &[u8], index: u32) -> Option<(f32, f32, f32)> {
    let i16_at = |o: usize| be_u16(font, o).map(|v| v as i16 as f32);
    let (hhea, head) = (table_offset(font, index, b"hhea")?, table_offset(font, index, b"head")?);
    let upm = be_u16(font, head.checked_add(18)?)? as f32;
    if upm <= 0.0 {
        return None;
    }
    Some((i16_at(hhea.checked_add(4)?)? / upm, i16_at(hhea.checked_add(6)?)? / upm, i16_at(hhea.checked_add(8)?)? / upm))
}

/// The weight a variable face should be drawn at: Regular (400) within its `wght` axis, or
/// `None` when it has no weight axis or already defaults to Regular. Noto Sans CJK's variable
/// collection defaults to Thin (100), so the UI's CJK text came out hairline (#4820).
fn regular_weight(font: &[u8], index: u32) -> Option<f32> {
    let fvar = table_offset(font, index, b"fvar")?;
    let axes = fvar.checked_add(be_u16(font, fvar.checked_add(4)?)? as usize)?;
    let count = be_u16(font, fvar.checked_add(8)?)? as usize;
    let size = be_u16(font, fvar.checked_add(10)?)? as usize;
    let fixed = |o: usize| be_u32(font, o).map(|v| v as i32 as f32 / 65536.0);
    (0..count.min(64)).find_map(|i| {
        let rec = axes.checked_add(i.checked_mul(size)?)?;
        (font.get(rec..rec.checked_add(4)?)? == b"wght").then_some(())?;
        let (min, default, max) = (fixed(rec.checked_add(4)?)?, fixed(rec.checked_add(8)?)?, fixed(rec.checked_add(12)?)?);
        let regular = 400f32.clamp(min.min(max), max.max(min));
        (regular.is_finite() && (regular - default).abs() > 0.5).then_some(regular)
    })
}

/// `y_offset_factor` that puts `face`'s baseline where `primary`'s would be. egui centres a
/// fallback face's row in the primary font's row, so a face with a big line gap (Hiragino: 0.5em)
/// otherwise rides about 0.23em high next to Latin text.
fn baseline_offset(face: (f32, f32, f32), primary: (f32, f32, f32)) -> f32 {
    let height = |m: (f32, f32, f32)| m.0 - m.1 + m.2;
    let off = primary.0 - face.0 - 0.5 * (height(primary) - height(face));
    if off.is_finite() { off.clamp(-0.5, 0.5) } else { 0.0 }
}

/// Registers `name` at the lowest priority in every font family.
fn add_to_all_families(ctx: &egui::Context, name: String, mut data: FontData) {
    // Align to Inter, the primary of every UI family; JetBrains Mono differs by under 0.005em.
    let primary = ctx.fonts(|f| f.definitions().font_data.get("Inter").and_then(|p| vertical_metrics(&p.font, p.index)));
    if let (Some(face), Some(primary)) = (vertical_metrics(&data.font, data.index), primary) {
        data.tweak.y_offset_factor = baseline_offset(face, primary);
    }
    if let Some(weight) = regular_weight(&data.font, data.index) {
        data.tweak.coords = egui::epaint::text::VariationCoords::new([(b"wght", weight)]);
    }
    let mut families: Vec<FontFamily> = ctx.fonts(|f| f.definitions().families.keys().cloned().collect());
    for f in [FontFamily::Proportional, FontFamily::Monospace] {
        if !families.contains(&f) {
            families.push(f);
        }
    }
    let families = families.into_iter().map(|family| InsertFontFamily { family, priority: FontPriority::Lowest }).collect();
    crate::theme::size_ui_font(ctx, &name, &mut data);
    ctx.add_font(FontInsert { name, data, families });
}

/// The egui plugin that watches drawn text and loads fallback fonts on demand.
pub struct CjkFontPlugin(pub CjkFallback);

impl egui::Plugin for CjkFontPlugin {
    fn debug_name(&self) -> &'static str {
        "photocraft-cjk-fonts"
    }

    fn output_hook(&mut self, ctx: &egui::Context, output: &mut egui::FullOutput) {
        if std::mem::take(&mut self.0.defer_after_reset) {
            ctx.request_repaint();
            return;
        }
        if self.0.exhausted() && self.0.scripts_exhausted() {
            return;
        }
        let Some(c) = missing_char(ctx, &output.shapes, &self.0) else { return };
        let font = match cjk::classify(c) {
            Some(kind) => self.0.next_font(kind),
            None => self.0.next_script_font(c),
        };
        if let Some((name, data)) = font {
            add_to_all_families(ctx, name, data);
            ctx.request_repaint();
        } else if cjk::classify(c).is_some() && cjk::fontconfig_cjk_scan_pending() {
            ctx.request_repaint_after(std::time::Duration::from_millis(50));
        } else if !self.0.exhausted() || !self.0.scripts_exhausted() {
            // Nothing readable for this script; try the next one on the next frame.
            ctx.request_repaint();
        }
    }
}

/// Installs the lazy CJK and other-script fallback (system fonts; nothing on the web).
pub fn install(ctx: &egui::Context) {
    install_with(ctx, Sources::system());
}

pub fn install_with(ctx: &egui::Context, sources: Sources) {
    // egui keeps the first plugin of a type. set_fonts removes loaded fallbacks, so reset the
    // existing loader too; otherwise its tried/loaded sets prevent fonts being added again.
    let mut fallback = CjkFallback::new(sources.clone());
    fallback.defer_after_reset = true;
    if ctx
        .with_plugin::<CjkFontPlugin, _>(|plugin| {
            plugin.0 = CjkFallback::new(sources.clone());
            plugin.0.defer_after_reset = true;
        })
        .is_none()
    {
        ctx.add_plugin(CjkFontPlugin(fallback));
    }
}
#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn lazy_fallbacks_follow_ui_font_size_and_survive_size_changes() {
        use photocraft_engine::prefs::UiFontSize;
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
        crate::theme::set_ui_font_size(&ctx, UiFontSize::Large);
        let name = "test-lazy-fallback".to_string();
        add_to_all_families(&ctx, name.clone(), FontData::from_static(photocraft_text::fonts::INTER_REGULAR.as_slice()));
        for (size, scale) in [(UiFontSize::Large, 16.0 / 12.0), (UiFontSize::Tiny, 10.0 / 12.0), (UiFontSize::Small, 1.0)] {
            crate::theme::set_ui_font_size(&ctx, size);
            for _ in 0..2 {
                ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
            }
            let fonts = ctx.fonts(|f| f.definitions().clone());
            let fallback = fonts.font_data.get(&name).unwrap();
            assert_eq!(fallback.tweak.scale, scale);
            assert_eq!(fallback.tweak.y_offset_factor, 0.0, "same face as Inter keeps its baseline");
            for stack in fonts.families.values() {
                assert_eq!(stack.iter().filter(|n| *n == &name).count(), 1);
                assert_eq!(stack.last(), Some(&name), "fallback stays at the lowest priority");
            }
        }
    }

    /// A font holding only an `fvar` table with the given axes (tag, min, default, max), as a
    /// single face or as face 1 of a two-face collection.
    fn fvar_font(axes: &[(&[u8; 4], f32, f32, f32)], collection: bool) -> Vec<u8> {
        let fixed = |v: f32| ((v * 65536.0) as i32).to_be_bytes();
        let mut fvar = Vec::new();
        for x in [1u16, 0, 16, 2, axes.len() as u16, 20, 0, 0] {
            fvar.extend(x.to_be_bytes());
        }
        for (tag, min, def, max) in axes {
            fvar.extend(*tag);
            for v in [*min, *def, *max] {
                fvar.extend(fixed(v));
            }
            fvar.extend([0, 0, 1, 0]);
        }
        let face = |at: u32| {
            let mut d = vec![0, 1, 0, 0, 0, 1, 0, 16, 0, 0, 0, 0];
            d.extend(b"fvar");
            d.extend([0; 4]);
            d.extend(at.to_be_bytes());
            d.extend((fvar.len() as u32).to_be_bytes());
            d
        };
        if !collection {
            let mut f = face(28);
            f.extend(&fvar);
            return f;
        }
        // ttcf header (12 + 2×4), two 28-byte directories, then the shared table.
        let mut f = b"ttcf".to_vec();
        f.extend([0, 1, 0, 0, 0, 0, 0, 2]);
        f.extend(20u32.to_be_bytes());
        f.extend(48u32.to_be_bytes());
        f.extend(face(76));
        f.extend(face(76));
        f.extend(&fvar);
        f
    }

    /// #4820: Noto Sans CJK's variable collection defaults to Thin; the fallback asks for Regular.
    #[test]
    fn variable_fallbacks_are_drawn_at_regular_weight() {
        let thin = fvar_font(&[(b"wght", 100.0, 100.0, 900.0)], true);
        assert_eq!(regular_weight(&thin, 1), Some(400.0), "Noto Sans CJK VF, face 1 of a collection");
        assert_eq!(regular_weight(&fvar_font(&[(b"wdth", 75.0, 100.0, 100.0), (b"wght", 100.0, 100.0, 900.0)], false), 0), Some(400.0));
        assert_eq!(regular_weight(&fvar_font(&[(b"wght", 500.0, 600.0, 900.0)], false), 0), Some(500.0), "Regular clamped into the axis");
        assert_eq!(regular_weight(&fvar_font(&[(b"wght", 100.0, 400.0, 900.0)], false), 0), None, "already Regular");
        assert_eq!(regular_weight(&fvar_font(&[(b"wdth", 75.0, 100.0, 100.0)], false), 0), None, "no weight axis");
        assert_eq!(regular_weight(photocraft_text::fonts::INTER_REGULAR.as_slice(), 0), None, "static font");
        for bad in [&thin[..thin.len() - 9], &thin[..30], &b"ttcf"[..], &[][..]] {
            assert_eq!(regular_weight(bad, 1), None);
        }
        // Registered as a fallback, a static face keeps the default (no variation) coordinates.
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
        add_to_all_families(&ctx, "test-static".into(), FontData::from_static(photocraft_text::fonts::INTER_REGULAR.as_slice()));
        ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
        let fonts = ctx.fonts(|f| f.definitions().clone());
        assert_eq!(fonts.font_data.get("test-static").map(|d| d.tweak.coords.clone()), Some(Default::default()));
    }

    static DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

    #[test]
    fn changing_language_resets_the_loader_once_and_reorders_scripts() {
        crate::i18n::with_language(crate::i18n::Lang::EN, || {
            let ctx = egui::Context::default();
            crate::i18n::sync_context(&ctx, "ja");
            ctx.with_plugin::<CjkFontPlugin, _>(|plugin| {
                assert_eq!(plugin.0.order()[0], CjkScript::Japanese);
                plugin.0.tried = plugin.0.order().to_vec();
                plugin.0.last_resort_tried = true;
            })
            .expect("font plugin");
            crate::i18n::sync_context(&ctx, "ja");
            assert!(ctx.with_plugin::<CjkFontPlugin, _>(|plugin| plugin.0.exhausted()).expect("font plugin"), "idle frames must not reload fonts");
            crate::i18n::sync_context(&ctx, "ko");
            ctx.with_plugin::<CjkFontPlugin, _>(|plugin| {
                assert!(!plugin.0.exhausted());
                assert_eq!(plugin.0.order()[0], CjkScript::Korean);
            })
            .expect("font plugin");
        });
    }

    fn fake_dir() -> PathBuf {
        let mut g = DIR.lock().unwrap();
        g.get_or_insert_with(|| {
            let dir = std::env::temp_dir().join(format!("photocraft-cjk-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("font.ttf"), include_bytes!("../../../assets/fonts/Inter-Regular.ttf")).unwrap();
            std::fs::write(dir.join("empty.ttf"), b"").unwrap();
            std::fs::write(dir.join("thai.ttf"), synthetic_font(&[(0x0E01, 0x0E5B)])).unwrap();
            std::fs::write(dir.join("arabic.ttf"), synthetic_font(&[(0x0600, 0x06FF)])).unwrap();
            std::fs::write(dir.join("devanagari.ttf"), synthetic_font(&[(0x0900, 0x097F)])).unwrap();
            dir
        })
        .clone()
    }

    fn fake(name: &str) -> FontFile {
        FontFile { path: fake_dir().join(name), family: String::new() }
    }

    fn fake_sources(locale: fn() -> Option<String>) -> Sources {
        Sources {
            locale,
            // Korean has a readable "font", Japanese only broken files, Chinese none.
            files: |s| match s {
                CjkScript::Korean => vec![fake("missing.ttc"), fake("empty.ttf"), fake("font.ttf")],
                CjkScript::Japanese => vec![fake("empty.ttf")],
                _ => vec![],
            },
            last_resort: || vec![fake("font.ttf")],
            scripts: |_| Vec::new(),
            covers: scripts::covering_face,
            embedded: no_embedded,
        }
    }

    #[test]
    fn picks_preferred_script_skips_unreadable_and_exhausts() {
        let mut fb = CjkFallback::new(fake_sources(|| Some("en_US".into())));
        assert!(!fb.exhausted());
        // Hangul: Korean first; missing and empty files are skipped.
        let (name, data) = fb.next_font(CjkChar::Hangul).expect("korean font");
        assert!(name.starts_with(FONT_PREFIX));
        assert_eq!(data.index, 0);
        assert_eq!(fb.tried, vec![CjkScript::Korean]);
        // Han on an English locale: SC, TC, JA have nothing readable, and the last-resort file was
        // already loaded, so nothing new; everything is now tried.
        assert!(fb.next_font(CjkChar::Han).is_none());
        assert!(fb.exhausted());
        assert_eq!(fb.tried, vec![CjkScript::Korean, CjkScript::SimplifiedChinese, CjkScript::TraditionalChinese, CjkScript::Japanese]);
        assert_eq!(fb.registered.len(), 1, "a file is read at most once");
        assert!(fb.next_font(CjkChar::Kana).is_none());
    }

    #[test]
    fn han_follows_locale() {
        for (loc, first) in [
            ("ja_JP.UTF-8", CjkScript::Japanese),
            ("zh-Hans-CN", CjkScript::SimplifiedChinese),
            ("zh-Hant-TW", CjkScript::TraditionalChinese),
            ("ko_KR", CjkScript::Korean),
        ] {
            let mut fb = CjkFallback::new(Sources {
                locale: || None,
                files: |_| vec![],
                last_resort: Vec::new,
                scripts: |_| Vec::new(),
                covers: scripts::covering_face,
                embedded: no_embedded,
            });
            fb.order = Some(cjk::script_order(Some(loc)));
            assert!(fb.next_font(CjkChar::Han).is_none());
            assert_eq!(fb.tried[0], first, "{loc}");
        }
    }

    /// Runs frames showing `text` until fonts settle; returns whether every glyph is covered.
    fn render(ctx: &egui::Context, text: &str) -> bool {
        for _ in 0..12 {
            let mut out = ctx.run_ui(Default::default(), |ui| {
                ui.label(text);
            });
            out.textures_delta.clear();
        }
        ctx.fonts_mut(|f| f.has_glyphs(&FontId::proportional(12.0), &text.replace(' ', "")))
    }

    /// A minimal TrueType font mapping the `ranges` (inclusive) to box glyphs, so tests have a
    /// font for a script without committing font files.
    fn synthetic_font(ranges: &[(u32, u32)]) -> Vec<u8> {
        let be16 = |v: &mut Vec<u8>, x: i32| v.extend((x as u16).to_be_bytes());
        let be32 = |v: &mut Vec<u8>, x: u32| v.extend(x.to_be_bytes());
        let chars: u32 = ranges.iter().map(|(a, b)| b - a + 1).sum();
        let glyphs = chars as i32 + 1;
        // One 500x700 box (contour of four on-curve points); glyph 0 (.notdef) is empty.
        let mut boxed = Vec::new();
        for x in [1, 50, 0, 550, 700, 3, 0] {
            be16(&mut boxed, x);
        }
        boxed.extend([1u8; 4]);
        for x in [50, 0, 500, 0, 0, 700, 0, -700] {
            be16(&mut boxed, x);
        }
        boxed.extend([0, 0]);
        let glyf: Vec<u8> = boxed.repeat(chars as usize);
        let mut loca = Vec::new();
        be32(&mut loca, 0);
        for i in 0..chars {
            be32(&mut loca, i * boxed.len() as u32);
        }
        be32(&mut loca, chars * boxed.len() as u32);
        let mut head = Vec::new();
        for x in [0x0001_0000, 0x0001_0000, 0, 0x5F0F_3CF5] {
            be32(&mut head, x);
        }
        for x in [0x000B, 1000, 0, 0, 0, 0, 0, 0, 0, 0, 0, -200, 600, 800, 0, 8, 2, 1, 0] {
            be16(&mut head, x);
        }
        let mut hhea = Vec::new();
        be32(&mut hhea, 0x0001_0000);
        for x in [800, -200, 0, 600, 0, 0, 550, 1, 0, 0, 0, 0, 0, 0, 0, glyphs] {
            be16(&mut hhea, x);
        }
        let mut maxp = Vec::new();
        be32(&mut maxp, 0x0001_0000);
        for x in [glyphs, 4, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0] {
            be16(&mut maxp, x);
        }
        let mut hmtx = Vec::new();
        for _ in 0..glyphs {
            be16(&mut hmtx, 600);
            be16(&mut hmtx, 50);
        }
        let mut cmap = Vec::new();
        for x in [0, 1, 3, 10] {
            be16(&mut cmap, x);
        }
        be32(&mut cmap, 12);
        be16(&mut cmap, 12);
        be16(&mut cmap, 0);
        be32(&mut cmap, 16 + 12 * ranges.len() as u32);
        be32(&mut cmap, 0);
        be32(&mut cmap, ranges.len() as u32);
        let mut gid = 1;
        for (a, b) in ranges {
            for x in [*a, *b, gid] {
                be32(&mut cmap, x);
            }
            gid += b - a + 1;
        }
        let tables: [(&[u8; 4], Vec<u8>); 7] =
            [(b"cmap", cmap), (b"glyf", glyf), (b"head", head), (b"hhea", hhea), (b"hmtx", hmtx), (b"loca", loca), (b"maxp", maxp)];
        let mut font = Vec::new();
        be32(&mut font, 0x0001_0000);
        for x in [7, 64, 2, 48] {
            be16(&mut font, x);
        }
        let mut offset = 12 + 16 * tables.len() as u32;
        for (tag, data) in &tables {
            font.extend(*tag);
            be32(&mut font, 0);
            be32(&mut font, offset);
            be32(&mut font, data.len() as u32);
            offset += (data.len() as u32).next_multiple_of(4);
        }
        for (_, data) in &tables {
            font.extend(data);
            font.resize(font.len().next_multiple_of(4), 0);
        }
        font
    }

    /// Thai, Arabic and Devanagari each have a synthetic font; Thai's list starts with unreadable
    /// files and Inter, which doesn't map Thai. No other script has a font.
    fn script_sources() -> Sources {
        Sources {
            locale: || Some("en".into()),
            files: |_| vec![],
            last_resort: Vec::new,
            scripts: |s| {
                let f = |name: &str| fake_dir().join(name);
                match s {
                    Script::Thai => vec![f("missing.ttf"), f("empty.ttf"), f("font.ttf"), f("thai.ttf")],
                    Script::Arabic => vec![f("arabic.ttf")],
                    Script::Devanagari => vec![f("devanagari.ttf")],
                    _ => vec![],
                }
            },
            covers: scripts::covering_face,
            embedded: no_embedded,
        }
    }

    fn fallback_files(ctx: &egui::Context) -> Vec<String> {
        ctx.with_plugin::<CjkFontPlugin, _>(|p| p.0.registered.iter().map(|(_, path, _)| path.file_name().unwrap().to_string_lossy().into_owned()).collect())
            .unwrap()
    }

    #[test]
    fn layer_names_in_other_scripts_load_a_covering_font_once() {
        let ctx = egui::Context::default();
        for _ in 0..2 {
            crate::theme::install_fonts_with(&ctx, script_sources());
            assert!(render(&ctx, "Layer 1 – café"));
            ctx.with_plugin::<CjkFontPlugin, _>(|p| assert!(p.0.scripts_tried.is_empty(), "Latin text reads no font")).unwrap();
            assert!(fallback_files(&ctx).is_empty());
            // A Thai type layer's name (#2275), Arabic (#1693) and Devanagari: no missing glyphs.
            for text in ["สวัสดี", "ภาพถ่าย/วันหยุดพักผ่อน", "مرحبا بالعالم", "नमस्ते दुनिया"]
            {
                assert!(render(&ctx, text), "tofu in {text}");
            }
            assert_eq!(fallback_files(&ctx), ["thai.ttf", "arabic.ttf", "devanagari.ttf"], "the first file that maps the character, never Inter");
            // Hebrew has no font here: tried once, then left alone (no reload or repaint loop).
            assert!(!render(&ctx, "שלום"));
            assert!(!render(&ctx, "שלום"));
            ctx.with_plugin::<CjkFontPlugin, _>(|p| {
                assert!(p.0.scripts_tried.contains(&Script::Hebrew));
                assert_eq!(p.0.registered.len(), 3);
            })
            .unwrap();
            let fonts = ctx.fonts(|f| f.definitions().clone());
            assert_eq!(fonts.families[&FontFamily::Proportional].first().map(String::as_str), Some("Inter"));
            for stack in fonts.families.values() {
                assert_eq!(stack.iter().filter(|n| n.starts_with(FONT_PREFIX)).count(), 3);
                assert_eq!(stack.last().map(String::as_str), Some("system-cjk-2"), "appended last so Latin keeps Inter");
            }
        }
    }

    #[test]
    fn a_missing_cjk_font_does_not_block_other_scripts_and_vice_versa() {
        let ctx = egui::Context::default();
        crate::theme::install_fonts_with(&ctx, script_sources());
        assert!(!render(&ctx, "日本語/ภาพถ่าย/วันหยุดพักผ่อน.png"));
        assert!(render(&ctx, "ภาพถ่าย/วันหยุดพักผ่อน.png"));
        ctx.with_plugin::<CjkFontPlugin, _>(|p| assert!(p.0.exhausted())).unwrap();
        assert_eq!(fallback_files(&ctx), ["thai.ttf"]);
        // A script with no font leaves CJK available.
        let mut fallback = CjkFallback::new(fake_sources(|| Some("ko".into())));
        assert!(fallback.next_script_font('ส').is_none());
        assert!(fallback.next_script_font('ส').is_none());
        assert!(fallback.next_script_font('A').is_none(), "Latin is never a fallback script");
        assert_eq!(fallback.scripts_tried, [Script::Thai]);
        assert!(fallback.next_font(CjkChar::Hangul).is_some());
    }

    #[test]
    fn script_candidates_are_read_a_bounded_number_of_times() {
        static CHECKED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let mut fallback = CjkFallback::new(Sources {
            scripts: |_| (0..20).map(|i| fake_dir().join(if i % 2 == 0 { "font.ttf" } else { "thai.ttf" })).collect(),
            covers: |_, _| {
                CHECKED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            },
            ..script_sources()
        });
        assert!(fallback.next_script_font('ش').is_none());
        assert!(fallback.next_script_font('ش').is_none());
        assert_eq!(CHECKED.load(std::sync::atomic::Ordering::Relaxed), MAX_SCRIPT_READS, "one bounded pass per script");
        assert!(fallback.registered.is_empty());
        assert!(!fallback.scripts_exhausted());
    }

    #[test]
    fn system_fonts_cover_other_script_samples() {
        // Like the CJK system-font test, run per script when this machine has a font for it.
        // The deterministic tests above do not need installed fonts.
        let samples = [("ภาพถ่าย/วันหยุดพักผ่อน", 'ภ'), ("مرحبا", 'م'), ("שלום", 'ש'), ("नमस्ते", 'न')];
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        for (text, c) in samples {
            let Some(script) = scripts::classify(c) else { continue };
            if !scripts::system_candidates(script).iter().any(|p| read_font(p).and_then(|b| scripts::covering_face(&b, c)).is_some()) {
                eprintln!("skipping {text}: no {script:?} system font");
                continue;
            }
            assert!(render(&ctx, text), "{text} still has missing glyphs");
        }
    }

    #[test]
    fn vertical_metrics_parse_and_reject_garbage() {
        let (asc, desc, gap) = vertical_metrics(include_bytes!("../../../assets/fonts/Inter-Regular.ttf"), 0).unwrap();
        assert!((asc - 1984.0 / 2048.0).abs() < 1e-4 && (desc + 494.0 / 2048.0).abs() < 1e-4 && gap == 0.0);
        for bad in [&b""[..], b"ttcf", b"ttcf\0\0\0\0\0\0\0\0\xff\xff\xff\xff", b"\0\x01\0\0\0\xff\0\0\0\0\0\0"] {
            assert!(vertical_metrics(bad, 0).is_none());
            assert!(vertical_metrics(bad, u32::MAX).is_none());
        }
    }

    #[test]
    fn a_face_with_a_big_line_gap_is_shifted_down_onto_the_primary_baseline() {
        let inter = (1984.0 / 2048.0, -494.0 / 2048.0, 0.0);
        let hiragino = (0.88, -0.12, 0.5);
        assert!((baseline_offset(hiragino, inter) - 0.2338).abs() < 1e-3);
        assert_eq!(baseline_offset(inter, inter), 0.0);
        assert_eq!(baseline_offset((f32::NAN, 0.0, 0.0), inter), 0.0);
        let jb = (1.02, -0.3, 0.0);
        assert!((baseline_offset(hiragino, jb) - baseline_offset(hiragino, inter)).abs() < 0.005, "one offset serves the monospace family too");
    }

    #[test]
    fn lazy_registration_triggers_only_for_supported_scripts() {
        let ctx = egui::Context::default();
        install_with(&ctx, fake_sources(|| Some("ko".into())));
        let _ = render(&ctx, "Layer 1 – café");
        let n = ctx.fonts(|f| f.definitions().font_data.keys().filter(|k| k.starts_with(FONT_PREFIX)).count());
        assert_eq!(n, 0, "Latin text loads nothing");
        // Hangul isn't in the stand-in font, but the Korean candidate must have been registered.
        let _ = render(&ctx, "카드 배경");
        let fonts = ctx.fonts(|f| f.definitions().clone());
        let name = format!("{FONT_PREFIX}-0");
        assert!(fonts.font_data.contains_key(&name));
        for (fam, stack) in &fonts.families {
            assert_eq!(stack.last(), Some(&name), "{fam:?}: appended last so Latin keeps Inter");
        }
    }

    /// No system fonts at all: craft-fonts alone draws Japanese.
    fn craft_only(locale: fn() -> Option<String>) -> Sources {
        Sources { locale, files: |_| vec![], last_resort: Vec::new, scripts: |_| Vec::new(), covers: scripts::covering_face, embedded: craft_embedded }
    }

    #[test]
    fn craft_fonts_render_japanese_without_system_fonts() {
        let Some(ui_font) = craft_fonts::japanese_for_ui().first().copied() else {
            eprintln!("skipping: built without craft-fonts (set CRAFT_FONTS_DIR to a craft-fonts checkout)");
            return;
        };
        // An English locale tries Chinese first (none here), then Japanese: the craft font.
        for locale in [(|| Some("en_US".into())) as fn() -> Option<String>, || Some("ja_JP".into()), || None] {
            let ctx = egui::Context::default();
            crate::theme::install_fonts_with(&ctx, craft_only(locale));
            assert!(render(&ctx, "日本語の文字"), "tofu in 日本語の文字");
            assert!(render(&ctx, "レイヤー 1"), "tofu in kana");
            let fonts = ctx.fonts(|f| f.definitions().clone());
            let name = format!("{FONT_PREFIX}-0");
            assert!(fonts.font_data.get(&name).is_some_and(|d| std::ptr::eq(d.font.as_ref().as_ptr(), ui_font.bytes.as_ptr())), "BIZ UDPGothic Regular first");
            for (fam, stack) in &fonts.families {
                assert_eq!(stack.last(), Some(&name), "{fam:?}: appended last so Latin keeps Inter");
                assert!(stack.first().is_some_and(|f| f != &name), "{fam:?}");
            }
        }
    }

    #[test]
    fn ui_works_without_craft_fonts() {
        // The pre-craft-fonts behaviour: no embedded fonts, no system fonts. Latin renders,
        // Japanese is tried (and stays missing) without panicking, and the loader exhausts.
        let ctx = egui::Context::default();
        crate::theme::install_fonts_with(
            &ctx,
            Sources {
                locale: || None,
                files: |_| vec![],
                last_resort: Vec::new,
                scripts: |_| Vec::new(),
                covers: scripts::covering_face,
                embedded: no_embedded,
            },
        );
        assert!(render(&ctx, "Layer 1 – café"));
        assert!(!render(&ctx, "日本語の文字"));
        let n = ctx.fonts(|f| f.definitions().font_data.keys().filter(|k| k.starts_with(FONT_PREFIX)).count());
        assert_eq!(n, 0);
        if craft_fonts::CRAFT_FONTS.is_empty() {
            assert!(craft_embedded(CjkScript::Japanese).is_empty());
        }
    }

    /// With the real system fonts, every CJK sample renders (skipped per script when the
    /// machine has no font for it, e.g. CI Linux without Noto CJK).
    #[test]
    fn system_fonts_cover_cjk_samples() {
        #[cfg(target_os = "linux")]
        wait_for_fontconfig_cjk_scan();
        let available = |s: CjkScript| cjk::font_files(s).iter().any(|f| f.path.is_file());
        let samples = [
            ("图层", CjkScript::SimplifiedChinese),
            ("圖層", CjkScript::TraditionalChinese),
            ("카드 배경", CjkScript::Korean),
            ("レイヤー", CjkScript::Japanese),
        ];
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        for (text, script) in samples {
            if !available(script) {
                eprintln!("skipping {text}: no {script:?} system font");
                continue;
            }
            assert!(render(&ctx, text), "{text} still has missing glyphs");
        }
        // All at once in a fresh context too (several scripts in one frame).
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        let all: String = samples.iter().filter(|(_, s)| available(*s)).map(|(t, _)| *t).collect::<Vec<_>>().join(" ");
        assert!(render(&ctx, &all), "{all}");
    }
}
