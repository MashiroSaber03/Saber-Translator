//! Small font-menu samples, rendered by the same font database as document text.

use std::collections::VecDeque;
use std::sync::PoisonError;

use egui::{ColorImage, Rect, TextureHandle, TextureOptions, pos2, vec2};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::TextLayer;
use photocraft_engine::prefs::FontPreview;
use photocraft_geom::Affine;

const SAMPLE: &str = "AaBbCc";
/// Alef, lam, meem and ain: a face lacking any of them is not treated as an Arabic face.
const ARABIC_CORE: [char; 4] = ['\u{627}', '\u{644}', '\u{645}', '\u{639}'];
/// Short, recognisable, and joins letters, so it shows the face's shaping.
const ARABIC_SAMPLE: &str = "أبجد هوز";
/// The Latin and Arabic sample widths and the sample height at the menu's medium (18 pt).
const WIDTH: f32 = 100.0;
const ARABIC_WIDTH: f32 = 72.0;
const GAP: f32 = 6.0;
const HEIGHT: f32 = 24.0;
/// The medium point size the widths above are measured at.
const MEDIUM: f32 = 18.0;
const CACHE_LIMIT: usize = 128;
const CACHE_ID: &str = "font-menu-previews";

/// A width or height measured at 18 pt, scaled to the sample's point size.
fn scaled(v: f32, size: f32) -> f32 {
    (v * size / MEDIUM).round()
}

/// The Latin sample's `[width, height]` box for a point size: [`WIDTH`]×[`HEIGHT`] at 18 pt (the
/// menu's medium), proportional otherwise.
fn box_for(size: f32) -> [f32; 2] {
    [scaled(WIDTH, size), scaled(HEIGHT, size)]
}

/// The font menu's sample geometry for Preferences ▸ Type ▸ Font Preview Size: the sample's point
/// size and its Latin box; `None` when the preview is off.
pub(crate) fn sample(preview: FontPreview) -> Option<(f32, [f32; 2])> {
    let size = match preview {
        FontPreview::Off => return None,
        FontPreview::Small => 14.0,
        FontPreview::Medium => MEDIUM,
        FontPreview::Large => 24.0,
        FontPreview::ExtraLarge => 32.0,
        FontPreview::Huge => 40.0,
    };
    Some((size, box_for(size)))
}

/// Family, display scale, the shared font database's generation (so a family installed or
/// fetched (served fonts) after its first miss gets a sample once it is available) and the point
/// size. Not the served catalog's generation: a listed family has no sample until its fonts are
/// registered, which moves the database's generation, and keying on the catalog re-rendered
/// every sample whenever it changed.
type Key = (String, u32, u64, u32);
#[derive(Clone)]
struct Samples {
    latin: TextureHandle,
    arabic: Option<TextureHandle>,
}
#[derive(Clone, Default)]
struct Cache(VecDeque<(Key, Option<Samples>)>);

struct SampleImages {
    latin: ColorImage,
    arabic: Option<ColorImage>,
}

/// The Latin sample text, and the Arabic one for a face that has both scripts. `chars` is the
/// face's ascending charmap. Arabic-only and symbol faces show their own glyphs instead of the
/// Latin sample and get no second sample.
fn sample_texts(chars: &[char]) -> (String, Option<&'static str>) {
    let has = |c: char| chars.binary_search(&c).is_ok();
    if !SAMPLE.chars().all(has) {
        return (chars.iter().copied().filter(|c| !c.is_whitespace()).take(6).collect(), None);
    }
    let arabic = ARABIC_CORE.into_iter().chain(ARABIC_SAMPLE.chars().filter(|c| !c.is_whitespace())).all(has);
    (SAMPLE.into(), arabic.then_some(ARABIC_SAMPLE))
}

fn render(engine: &mut photocraft_text::TextEngine, family: &str, text: &str, slot: f32, scale: u32, size: f32) -> Option<ColorImage> {
    if text.is_empty() {
        return None;
    }
    let layer = TextLayer { text: text.into(), font_family: family.into(), size_pt: size * scale as f32, ..Default::default() };
    let layout = engine.layout(&layer, 72.0);
    let bounds = layout.bounds()?;
    let width = (slot as u32 * scale) as usize;
    let height = (scaled(HEIGHT, size) as u32 * scale) as usize;
    let transform = Affine::translate(f64::from(-bounds[0]), f64::from((height as f32 - (bounds[3] - bounds[1])) / 2.0 - bounds[1]));
    // Reject oversized ink before the rasterizer allocates a surface (malformed installed fonts).
    let ink = photocraft_text::render::ink_rect(&layout, &transform);
    if ink.is_empty() || ink.width() > width as u32 * 4 || ink.height() > height as u32 * 4 {
        return None;
    }
    let rendered = photocraft_text::render::rasterize(&layout, &transform, PixelFormat::new(ColorMode::Rgb, SampleType::U8, true), layer.antialias);
    let mut rgba = vec![[0u8; 4]; width * height];
    rendered.surface.read_rgba8_into(photocraft_geom::Rect::new(0, 0, width as i32, height as i32), &mut rgba);
    Some(ColorImage::from_rgba_unmultiplied([width, height], &rgba.into_iter().flat_map(|p| [255, 255, 255, p[3]]).collect::<Vec<_>>()))
}

fn images(engine: &mut photocraft_text::TextEngine, family: &str, scale: u32, size: f32) -> Option<SampleImages> {
    // Unknown/unreadable families keep their plain menu label, rather than a misleading fallback.
    if !engine.fonts.has_family(family) {
        return None;
    }
    let (latin, arabic) = sample_texts(&engine.fonts.charmap(family, 400, false));
    let latin = render(engine, family, &latin, scaled(WIDTH, size), scale, size)?;
    let arabic = arabic.and_then(|text| render(engine, family, text, scaled(ARABIC_WIDTH, size), scale, size));
    Some(SampleImages { latin, arabic })
}

fn key(ctx: &egui::Context, family: &str, size: f32) -> Key {
    let scale = ctx.pixels_per_point().ceil().clamp(1.0, 4.0) as u32;
    (family.to_owned(), scale, photocraft_text::fonts::shared_generation(), size as u32)
}

/// Width a row reserves for its samples: wider once a painted row showed the family covers
/// Arabic, so the label truncates before the samples. Never renders, so rows scrolled out of
/// view cost nothing.
pub(crate) fn slot_width(ctx: &egui::Context, family: &str, size: f32) -> f32 {
    let key = key(ctx, family, size);
    let arabic = ctx.data_mut(|d| {
        let cache = d.get_temp_mut_or_default::<Cache>(egui::Id::new(CACHE_ID));
        cache.0.iter().any(|(k, samples)| k == &key && samples.as_ref().is_some_and(|s| s.arabic.is_some()))
    });
    if arabic { scaled(WIDTH, size) + GAP + scaled(ARABIC_WIDTH, size) } else { scaled(WIDTH, size) }
}

pub(crate) fn paint(ui: &egui::Ui, family: &str, row: Rect, size: f32) {
    let ctx = ui.ctx();
    let scale = ctx.pixels_per_point().ceil().clamp(1.0, 4.0) as u32;
    let key = key(ctx, family, size);
    let id = egui::Id::new(CACHE_ID);
    let cached = ctx.data_mut(|d| {
        let cache = d.get_temp_mut_or_default::<Cache>(id);
        let index = cache.0.iter().position(|(k, _)| k == &key)?;
        let entry = cache.0.remove(index)?;
        let samples = entry.1.clone();
        cache.0.push_back(entry);
        Some(samples)
    });
    let samples = cached.unwrap_or_else(|| {
        let samples = images(&mut photocraft_text::shared().lock().unwrap_or_else(PoisonError::into_inner), family, scale, size).map(|img| {
            let load = |image| ctx.load_texture(format!("font-sample-{family}"), image, TextureOptions::LINEAR);
            Samples { latin: load(img.latin), arabic: img.arabic.map(load) }
        });
        if samples.as_ref().is_some_and(|s| s.arabic.is_some()) {
            // The row reserved only the Latin width until now.
            ctx.request_repaint();
        }
        ctx.data_mut(|d| {
            let cache = d.get_temp_mut_or_default::<Cache>(id);
            if cache.0.len() >= CACHE_LIMIT {
                cache.0.pop_front();
            }
            cache.0.push_back((key, samples.clone()));
        });
        samples
    });
    if let Some(samples) = samples {
        let tint = crate::theme::Tokens::get(ctx).text;
        let uv = Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0));
        let [w, h] = box_for(size);
        let latin = Rect::from_min_size(pos2(row.right() - w - 8.0, row.center().y - h / 2.0), vec2(w, h));
        ui.painter().image(samples.latin.id(), latin, uv, tint);
        if let Some(arabic) = samples.arabic {
            let aw = scaled(ARABIC_WIDTH, size);
            let rect = Rect::from_min_size(pos2(latin.left() - GAP - aw, latin.top()), vec2(aw, h));
            ui.painter().image(arabic.id(), rect, uv, tint);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(engine: &mut photocraft_text::TextEngine, family: &str, scale: u32) -> Option<ColorImage> {
        images(engine, family, scale, MEDIUM).map(|i| i.latin)
    }

    fn charmap(text: &str) -> Vec<char> {
        let mut chars: Vec<char> = text.chars().collect();
        chars.sort_unstable();
        chars
    }

    #[test]
    fn previews_use_the_requested_face_and_scale() {
        let mut engine = photocraft_text::TextEngine::new();
        let sans = image(&mut engine, "Inter", 1).expect("bundled sans");
        let mono = image(&mut engine, "JetBrains Mono", 1).expect("bundled mono");
        assert_eq!(sans.size, [100, 24]);
        assert!(sans.pixels.iter().any(|p| p.a() > 0));
        assert_ne!(sans.pixels, mono.pixels, "samples use different font outlines");
        assert_eq!(image(&mut engine, "Inter", 2).unwrap().size, [200, 48]);
        assert!(image(&mut engine, "Missing font", 1).is_none());
    }

    /// Preferences ▸ Type ▸ Font Preview Size: Small…Huge scale the sample, Off removes it.
    #[test]
    fn the_preview_size_preference_scales_the_sample() {
        assert_eq!(sample(FontPreview::Off), None);
        assert_eq!(sample(FontPreview::Medium), Some((18.0, [100.0, 24.0])));
        let Some((_, [w, h])) = sample(FontPreview::Huge) else { panic!("Huge previews") };
        assert!(w > 100.0 && h > 24.0, "Huge is bigger than Medium: {w}×{h}");
        assert!(sample(FontPreview::Small).unwrap().1[0] < sample(FontPreview::ExtraLarge).unwrap().1[0], "sizes are ordered");
        let mut engine = photocraft_text::TextEngine::new();
        let big = images(&mut engine, "Inter", 1, 32.0).unwrap().latin;
        assert_eq!(big.size, [178, 43], "the box follows the size");
    }

    #[test]
    fn latin_only_families_get_no_arabic_sample() {
        let mut engine = photocraft_text::TextEngine::new();
        for family in ["Inter", "JetBrains Mono"] {
            assert!(images(&mut engine, family, 1, MEDIUM).expect("bundled").arabic.is_none(), "{family}");
        }
        assert!(images(&mut engine, "", 1, MEDIUM).is_none());
        assert!(images(&mut engine, "Missing font", 1, MEDIUM).is_none());
    }

    #[test]
    fn only_faces_with_both_scripts_get_the_arabic_sample() {
        let latin = format!("{SAMPLE} ");
        let arabic = format!("{ARABIC_SAMPLE}{}", ARABIC_CORE.iter().collect::<String>());
        let (text, second) = sample_texts(&charmap(&format!("{latin}{arabic}")));
        assert_eq!((text.as_str(), second), (SAMPLE, Some(ARABIC_SAMPLE)));
        assert_eq!(sample_texts(&charmap(&latin)), (SAMPLE.into(), None));
        let (own_glyphs, none) = sample_texts(&charmap(&arabic));
        assert_eq!((own_glyphs.chars().count(), none), (6, None));
        assert!(own_glyphs.chars().all(|c| arabic.contains(c)));
        assert_eq!(sample_texts(&[]), (String::new(), None));
        for missing in ARABIC_CORE.into_iter().chain(ARABIC_SAMPLE.chars().filter(|c| !c.is_whitespace())) {
            let partial: String = format!("{latin}{arabic}").chars().filter(|&c| c != missing).collect();
            assert_eq!(sample_texts(&charmap(&partial)).1, None, "without U+{:04X}", missing as u32);
        }
    }

    #[test]
    fn a_system_arabic_face_renders_both_samples() {
        let mut engine = photocraft_text::TextEngine::new();
        engine.fonts.load_system_fonts();
        let both = engine.fonts.families().into_iter().find(|f| sample_texts(&engine.fonts.charmap(f, 400, false)).1.is_some());
        let Some(family) = both else {
            eprintln!("skipped: no installed family covers both Latin and Arabic");
            return;
        };
        let images = images(&mut engine, &family, 1, MEDIUM).expect(&family);
        let arabic = images.arabic.expect("Arabic sample");
        assert_eq!(arabic.size, [ARABIC_WIDTH as usize, HEIGHT as usize]);
        assert!(arabic.pixels.iter().any(|p| p.a() > 0), "{family}");
    }
}
