//! Default Japanese inter-character spacing (JLREQ §3.1, §3.2 and Appendix B).
//!
//! OpenType shaping supplies the glyph advance and any space already in that advance.
//! Composition fills or reduces the space at a boundary without changing either glyph.

use icu_properties::{CodePointMapData, props::{EastAsianWidth, GeneralCategory, LineBreak, Script, WordBreak}};
use parley::FontData;

use crate::cjk::{JIS_CODE_PAGE, code_pages, is_jis_symbol, regional_code_pages};

/// Use the run language when specified, otherwise use the selected face's CJK encoding.
/// A pan-CJK face's code-page coverage alone does not select Japanese composition.
pub(crate) fn uses_japanese_spacing(language: Option<&str>, face: Option<&FontData>) -> bool {
    if let Some(language) = language {
        return language.split(['-', '_']).next().is_some_and(|lang| lang.eq_ignore_ascii_case("ja"));
    }
    regional_code_pages(face.map_or(0, code_pages)) & JIS_CODE_PAGE != 0
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class { EastAsian, Western, Open, Close, OpenQuote, CloseQuote, Middle, Space, Other }

impl Class {
    fn opens(self) -> bool { matches!(self, Self::Open | Self::OpenQuote) }
    fn closes(self) -> bool { matches!(self, Self::Close | Self::CloseQuote) }
    fn punctuation(self) -> bool { self.opens() || self.closes() || self == Self::Middle }
}

/// The spacing already present on both sides of a shaped cluster, in line-space pixels.
pub(crate) struct Cell {
    class: Class,
    em: f32,
    east_asian: bool,
    japanese: bool,
    before: f32,
    after: f32,
}

impl Cell {
    pub(crate) fn ideographic(em: f32, japanese: bool) -> Self {
        Self { class: Class::EastAsian, em, east_asian: true, japanese, before: 0.0, after: 0.0 }
    }

    pub(crate) fn new(ch: char, em: f32, advance: f32, japanese: bool) -> Self {
        use Class as C;
        use GeneralCategory as Category;
        use LineBreak as Break;

        let em = if em.is_finite() { em.max(0.0) } else { 0.0 };
        let lb = CodePointMapData::<Break>::new().get(ch);
        let category = CodePointMapData::<Category>::new().get(ch);
        let width = CodePointMapData::<EastAsianWidth>::new().get(ch);
        let wide = matches!(width, EastAsianWidth::Wide | EastAsianWidth::Fullwidth);
        // Japanese punctuation cells add margins for the JIS punctuation classes.
        // Encoded vertical brackets retain the same class as ordinary brackets.
        // Extended brackets can be shaped to a half cell and retain that advance.
        let vertical_bracket = matches!(category, Category::OpenPunctuation | Category::ClosePunctuation)
            && matches!(ch, '\u{fe10}'..='\u{fe1f}' | '\u{fe30}'..='\u{fe4f}');
        let full_cell_mark = wide && matches!(lb, Break::OpenPunctuation | Break::ClosePunctuation | Break::CloseParenthesis | Break::Nonstarter)
            && (is_jis_symbol(ch) || vertical_bracket);
        let script = CodePointMapData::<Script>::new().get(ch);
        let directional_quote = lb == Break::Quotation && width == EastAsianWidth::Ambiguous
            && matches!(category, Category::InitialPunctuation | Category::FinalPunctuation);
        let apostrophe = directional_quote && CodePointMapData::<WordBreak>::new().get(ch) == WordBreak::MidNumLet;
        // Roman punctuation in the Latin-1 repertoire participates in Western spacing.
        // Other standalone marks keep their shaped advance unless they have a class below.
        let roman_symbol = ch <= '\u{ff}' && matches!(category,
            Category::OpenPunctuation | Category::ClosePunctuation | Category::InitialPunctuation
            | Category::FinalPunctuation | Category::OtherPunctuation | Category::MathSymbol);
        let class = if ch.is_whitespace() {
            C::Space
        } else if directional_quote && !apostrophe {
            if category == Category::InitialPunctuation { C::OpenQuote } else { C::CloseQuote }
        } else if (full_cell_mark && lb == Break::OpenPunctuation) || (apostrophe && category == Category::InitialPunctuation) {
            C::Open
        } else if (full_cell_mark && matches!(lb, Break::ClosePunctuation | Break::CloseParenthesis))
            || (apostrophe && category == Category::FinalPunctuation)
        {
            C::Close
        } else if full_cell_mark && lb == Break::Nonstarter && category == Category::OtherPunctuation {
            C::Middle
        } else if matches!(script, Script::Han | Script::Hiragana | Script::Katakana) || (wide && ch.is_alphanumeric() && script != Script::Hangul) {
            C::EastAsian
        } else if matches!(width, EastAsianWidth::Narrow | EastAsianWidth::Neutral | EastAsianWidth::Ambiguous)
            && matches!(script, Script::Latin | Script::Greek | Script::Cyrillic | Script::Common)
            && !matches!(lb, Break::Inseparable | Break::BreakBoth)
            && (ch.is_alphanumeric() || ch.is_ascii_graphic() || roman_symbol || category == Category::CurrencySymbol)
        {
            C::Western
        } else {
            C::Other
        };
        // Proportional Western widths are glyph advances. Only an ideographic cell or a
        // full-em rendition of an ambiguous character can carry a built-in Japanese margin.
        let embedded = if wide || advance >= em - 0.01 { (advance - em * 0.5).clamp(0.0, em * 0.5) } else { 0.0 };
        let (before, after) = match class {
            C::Open | C::OpenQuote => (embedded, 0.0),
            C::Close | C::CloseQuote => (0.0, embedded),
            C::Middle => (embedded * 0.5, embedded * 0.5),
            _ => (0.0, 0.0),
        };
        Self { class, em, east_asian: wide || class == C::EastAsian, japanese, before, after }
    }

    /// Space to add to the preceding cluster's advance. Japanese punctuation uses its
    /// outer side; a Western span receives quarter-em spacing only at the ends next to CJK.
    pub(crate) fn spacing_before(&self, prev: &Self) -> f32 {
        use Class as C;
        if !(prev.east_asian && prev.japanese || self.east_asian && self.japanese) {
            return 0.0;
        }
        let width = match (prev.class, self.class) {
            (C::EastAsian, C::Western) => prev.em * 0.25,
            (C::Western, C::EastAsian) => self.em * 0.25,
            (left, right) if (left.opens() || left == C::Space) && right.opens()
                || left.closes() && (right.closes() || right == C::Space) => 0.0,
            (C::Middle | C::OpenQuote | C::CloseQuote, _) => prev.em * 0.25,
            (_, C::Middle | C::OpenQuote | C::CloseQuote) => self.em * 0.25,
            (C::Close, C::Open) => prev.em.max(self.em) * 0.5,
            (C::Close, _) => prev.em * 0.5,
            (_, C::Open) => self.em * 0.5,
            _ => return 0.0,
        };
        width - prev.after - self.before
    }

    /// A narrow line can use the closing mark's side bearing before moving the entire word.
    pub(crate) fn closing_margin(&self) -> f32 {
        if self.japanese && self.class.closes() { self.after } else { 0.0 }
    }

    /// A tight line may reduce the preferred spacing of proportional punctuation
    /// next to CJK. Ideographic spacing keeps its prescribed value.
    pub(crate) fn spacing_reduction(&self, next: &Self, gap: f32) -> f32 {
        let proportional_mark = |cell: &Self| !cell.east_asian && cell.class.punctuation();
        if proportional_mark(self) || proportional_mark(next) { gap.max(0.0) * 0.5 } else { 0.0 }
    }
}
