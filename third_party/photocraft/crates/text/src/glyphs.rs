//! Glyphs panel support: the characters a face maps (its `cmap`), Photoshop's "Show" categories,
//! and small glyph previews.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::TextLayer;
use photocraft_doc::text::{CharStyle, TextRun};
use photocraft_geom::Affine;
use skrifa::MetadataProvider;

use crate::FontDb;

/// The Glyphs panel's "Show" categories, in menu order.
pub const CATEGORIES: &[&str] = &[
    "Entire Font",
    "Basic Latin",
    "Latin 1",
    "Latin Extended-A",
    "Latin Extended-B",
    "Greek",
    "Cyrillic",
    "Punctuation",
    "Numbers",
    "Currency",
    "Symbols",
    "Math Symbols",
    "Arrows",
    "Ornaments",
];

/// Whether `c` belongs to a [`CATEGORIES`] entry (unknown names match everything).
pub fn in_category(c: char, category: &str) -> bool {
    let u = c as u32;
    match category {
        "Basic Latin" => (0x20..=0x7E).contains(&u),
        "Latin 1" => (0xA0..=0xFF).contains(&u),
        "Latin Extended-A" => (0x100..=0x17F).contains(&u),
        "Latin Extended-B" => (0x180..=0x24F).contains(&u),
        "Greek" => (0x370..=0x3FF).contains(&u) || (0x1F00..=0x1FFF).contains(&u),
        "Cyrillic" => (0x400..=0x52F).contains(&u),
        "Punctuation" => {
            c.is_ascii_punctuation() && !matches!(c, '$' | '+' | '<' | '=' | '>' | '^' | '`' | '|' | '~')
                || matches!(u, 0xA1 | 0xA7 | 0xAB | 0xB6 | 0xB7 | 0xBB | 0xBF)
                || (0x2010..=0x2027).contains(&u)
                || (0x2030..=0x205E).contains(&u)
                || (0x2E00..=0x2E7F).contains(&u)
                || (0x3000..=0x303F).contains(&u)
        }
        "Numbers" => c.is_numeric() || (0x2150..=0x218F).contains(&u),
        "Currency" => matches!(c, '$' | '¢' | '£' | '¤' | '¥') || (0x20A0..=0x20CF).contains(&u),
        "Symbols" => {
            matches!(c, '©' | '®' | '°' | '¦' | '¨' | '¯' | '´' | '¸' | '™' | '℠' | '№' | '℮' | '^' | '`' | '|' | '~')
                || (0x2100..=0x214F).contains(&u)
                || (0x2300..=0x23FF).contains(&u)
                || (0x2500..=0x25FF).contains(&u)
        }
        "Math Symbols" => {
            matches!(c, '+' | '<' | '=' | '>' | '±' | '×' | '÷' | '¬' | 'µ' | '¹' | '²' | '³' | '¼' | '½' | '¾')
                || (0x2070..=0x209F).contains(&u)
                || (0x2200..=0x22FF).contains(&u)
                || (0x27C0..=0x27EF).contains(&u)
                || (0x2980..=0x2AFF).contains(&u)
        }
        "Arrows" => (0x2190..=0x21FF).contains(&u) || (0x27F0..=0x27FF).contains(&u) || (0x2900..=0x297F).contains(&u) || (0x2B00..=0x2BFF).contains(&u),
        "Ornaments" => (0x2600..=0x27BF).contains(&u) || (0x1F300..=0x1FAFF).contains(&u) || (0xE000..=0xF8FF).contains(&u),
        _ => true,
    }
}

impl FontDb {
    /// Character coverage of the exact named face used by the type tool.
    pub fn charmap_for_style(&mut self, style: &CharStyle) -> Vec<char> {
        let mut style = style.clone();
        self.select_named_face(&mut style);
        self.charmap(&style.font_family, style.weight, style.italic)
    }

    /// Characters mapped by the face of `family` nearest to `weight`/`italic`, ascending
    /// (controls and whitespace other than the space excluded). Empty for an unknown family.
    pub fn charmap(&mut self, family: &str, weight: u16, italic: bool) -> Vec<char> {
        let Some(info) = self.fcx.collection.family_by_name(family) else {
            return Vec::new();
        };
        let fonts = info.fonts();
        let Some(font) = fonts.iter().min_by_key(|f| {
            let it = !matches!(f.style(), parley::fontique::FontStyle::Normal);
            (u32::from(it != italic) * 10_000) + (f.weight().value() - f32::from(weight)).abs() as u32
        }) else {
            return Vec::new();
        };
        let Some(blob) = font.load(Some(&mut self.fcx.source_cache)) else {
            return Vec::new();
        };
        let Ok(fr) = skrifa::FontRef::from_index(blob.as_ref(), font.index()) else {
            return Vec::new();
        };
        let mut out: Vec<char> = fr
            .charmap()
            .mappings()
            .filter(|(_, g)| g.to_u32() != 0)
            .filter_map(|(u, _)| char::from_u32(u))
            .filter(|c| *c == ' ' || !(c.is_control() || c.is_whitespace() || ('\u{FE00}'..='\u{FE0F}').contains(c)))
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// An 8-bit coverage preview of one character in a `px`-pixel em box, centred horizontally on
/// the baseline at 80 % of the box: (width = height = `px`, alpha bytes).
pub fn preview(engine: &mut crate::TextEngine, style: &CharStyle, c: char, px: u32) -> (u32, Vec<u8>) {
    let px = px.clamp(4, 512);
    let text = c.to_string();
    let mut st = style.clone();
    st.size_pt = px as f32 * 0.72;
    st.underline = false;
    st.strikethrough = false;
    st.tracking = 0.0;
    st.baseline_shift_pt = 0.0;
    let layer = TextLayer { runs: vec![TextRun { len: text.len(), style: st }], text, ..Default::default() };
    let layout = engine.layout(&layer, 72.0);
    let w = layout.bounds().map_or(0.0, |b| b[2] - b[0]);
    let x0 = layout.bounds().map_or(0.0, |b| b[0]);
    let tx = (px as f32 - w) / 2.0 - x0;
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
    let r = crate::render::rasterize_warped(
        &layout,
        &Affine::translate(f64::from(tx), f64::from(px as f32 * 0.78)),
        fmt,
        photocraft_doc::text::AntiAlias::Smooth,
        None,
    );
    let mut rgba = vec![[0u8; 4]; (px * px) as usize];
    r.surface.read_rgba8_into(photocraft_geom::Rect::new(0, 0, px as i32, px as i32), &mut rgba);
    (px, rgba.into_iter().map(|p| p[3]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_charmap_and_categories() {
        let mut db = FontDb::new();
        let cm = db.charmap(crate::fonts::DEFAULT_FAMILY, 400, false);
        assert!(cm.len() > 500, "{}", cm.len());
        assert!(cm.contains(&'A') && cm.contains(&'€') && cm.contains(&' '));
        assert!(!cm.contains(&'\n'));
        assert!(db.charmap("No Such Family", 400, false).is_empty());
        let latin: Vec<char> = cm.iter().copied().filter(|c| in_category(*c, "Basic Latin")).collect();
        assert_eq!(latin.len(), 95);
        assert!(in_category('€', "Currency") && in_category('—', "Punctuation") && in_category('→', "Arrows"));
        assert!(!in_category('A', "Punctuation") && in_category('A', "Entire Font"));
    }

    #[test]
    fn preview_draws_ink() {
        let mut eng = crate::TextEngine::new();
        let st = CharStyle { font_family: crate::fonts::DEFAULT_FAMILY.into(), ..Default::default() };
        let (n, a) = preview(&mut eng, &st, 'W', 32);
        assert_eq!((n, a.len()), (32, 32 * 32));
        let ink: u32 = a.iter().map(|&v| u32::from(v)).sum();
        assert!(ink > 20 * 255, "{ink}");
        let (_, sp) = preview(&mut eng, &st, ' ', 32);
        assert!(sp.iter().all(|&v| v == 0));
    }
}
