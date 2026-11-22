//! Font selection and OpenType metrics for the upright portions of a vertical line.

use icu_normalizer::properties::{CanonicalDecompositionBorrowed, Decomposed};
use icu_properties::{
    CodePointMapData,
    props::{BidiMirroringGlyph, BidiPairedBracketType, EastAsianWidth, GeneralCategory, LineBreak, Script, VerticalOrientation, WordBreak},
};
use parley::FontData;
use photocraft_doc::text::CharStyle;
use skrifa::{
    FontRef, GlyphId, MetadataProvider,
    instance::{LocationRef, NormalizedCoord, Size},
    outline::DrawSettings,
    raw::{
        FontData as TableData, TableProvider,
        tables::{
            gpos::{PositionSubtables, SinglePos, ValueContext, ValueRecord},
            gsub::{SingleSubst, SubstitutionSubtables},
            layout::{FeatureList, FeatureRecord, LangSys},
        },
        types::Tag,
    },
};

use crate::cjk::{CJK_CODE_PAGES, JIS_CODE_PAGE, code_pages, is_jis_symbol, is_legacy_jis_symbol, regional_code_pages};
use crate::raster::{Bounds, Pen, Xform};

/// Shape a whole grapheme with one orientation. The returned form also selects whether to
/// request vertical substitutions when the font shaper resolves the actual face.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum VerticalForm {
    #[default]
    Sideways,
    Upright,
}

impl VerticalForm {
    pub(crate) fn of(ch: char, japanese: bool, face: Option<&FontData>, language: Option<&str>) -> Self {
        use LineBreak as Break;
        use VerticalOrientation as Vo;
        if ch.is_ascii() {
            return Self::Sideways;
        }
        let width = CodePointMapData::<EastAsianWidth>::new().get(ch);
        let vo = CodePointMapData::<Vo>::new().get(ch);
        let font = || face.and_then(|f| FontRef::from_index(f.data.as_ref(), f.index).ok());
        if matches!(width, EastAsianWidth::Wide | EastAsianWidth::Fullwidth) {
            if vo == Vo::TransformedRotated && is_fullwidth_bracket(ch) {
                // Canonical aliases outside CJK punctuation keep their horizontal shaping,
                // including palt, even when they share a glyph with the fullwidth character.
                if canonical_base(ch) != ch || font().is_some_and(|font| {
                    font.charmap().map(ch).is_some_and(|glyph| is_proportional(&font, glyph)
                        && substitute(glyph, &vertical_lookups(&font, language, true)) == glyph)
                }) {
                    return Self::Sideways;
                }
            }
            return Self::Upright;
        }
        let line_break = CodePointMapData::<Break>::new().get(ch);
        let category = CodePointMapData::<GeneralCategory>::new().get(ch);
        let pages = face.map_or(0, code_pages);
        let cjk = pages & CJK_CODE_PAGES != 0;
        let dash = category == GeneralCategory::DashPunctuation;
        // Japanese composition gives decorative bracket symbols an ideographic cell.
        if japanese && ('\u{2700}'..='\u{27bf}').contains(&ch)
            && matches!(category, GeneralCategory::OpenPunctuation | GeneralCategory::ClosePunctuation)
            && font().is_some_and(|font| font.charmap().map(ch).is_some())
        {
            return Self::Upright;
        }

        let line_mark = matches!(line_break, Break::Inseparable | Break::BreakBoth);
        let word_separator = !japanese && width == EastAsianWidth::Ambiguous && vo == Vo::Rotated
            && line_break == Break::Ambiguous && CodePointMapData::<WordBreak>::new().get(ch) == WordBreak::MidLetter;
        let hyphen = width == EastAsianWidth::Ambiguous && line_break == Break::UnambiguousHyphen;
        // Regional CJK fonts use an ideographic cell for word separators. Proportional
        // leaders keep their horizontal advance unless the glyph has a vertical form of its own.
        if word_separator && regional_code_pages(pages) != 0 {
            return Self::Upright;
        }
        if pages & JIS_CODE_PAGE != 0 {
            let standalone = width == EastAsianWidth::Ambiguous
                && matches!(line_break, Break::Ambiguous | Break::PostfixNumeric)
                && matches!(category, GeneralCategory::OtherPunctuation | GeneralCategory::MathSymbol | GeneralCategory::OtherSymbol);
            let line_symbol = hyphen
                || (line_break == Break::PrefixNumeric && category == GeneralCategory::MathSymbol)
                || (line_break == Break::Alphabetic && category == GeneralCategory::OtherPunctuation);
            if (standalone && is_jis_symbol(ch)) || (line_symbol && is_legacy_jis_symbol(ch)) {
                return Self::Upright;
            }
        }
        // A Roman dash can be a full em wide and still belong to a sideways run.
        if !cjk && face.is_some() && dash && vo == Vo::Rotated {
            return if font().and_then(|font| {
                let glyph = font.charmap().map(ch)?;
                Some(has_alternate(&font, glyph, ch, language, !hyphen))
            }) == Some(true) { Self::Upright } else { Self::Sideways };
        }
        let neutral_mark = width == EastAsianWidth::Neutral && (line_mark || category == GeneralCategory::OtherPunctuation)
            && CodePointMapData::<Script>::new().get(ch) == Script::Common;
        if neutral_mark || word_separator || hyphen || (line_mark && width == EastAsianWidth::Ambiguous) {
            let designed_form = font().and_then(|font| {
                let glyph = font.charmap().map(ch)?;
                let proportional = is_proportional(&font, glyph);
                if hyphen && proportional {
                    return Some(Self::Sideways);
                }
                if !neutral_mark && !hyphen && (!proportional
                    || (line_break == Break::BreakBoth && has_vertical_design(&font)))
                {
                    return Some(Self::Upright);
                }
                // Dashes in the horizontal punctuation class need an authored vertical
                // substitution. An encoded compatibility glyph alone does not select it.
                let alternate = has_alternate(&font, glyph, ch, language, !hyphen);
                Some(if alternate { Self::Upright } else { Self::Sideways })
            });
            return designed_form.unwrap_or(if neutral_mark || hyphen { Self::Sideways } else { Self::Upright });
        }
        // The CJK composer keeps Asian line marks in their own cell. In Japanese
        // composition, ambiguous Western word punctuation keeps its proportional advance.
        let asian_mark = line_mark
            || (line_break == Break::Ambiguous && vo == Vo::Rotated
                && matches!(category, GeneralCategory::DashPunctuation | GeneralCategory::MathSymbol));
        if asian_mark || (matches!(vo, Vo::Upright | Vo::TransformedUpright)
            && (width != EastAsianWidth::Ambiguous || category == GeneralCategory::OtherSymbol))
        {
            Self::Upright
        } else {
            Self::Sideways
        }
    }
}

pub(crate) fn has_alternate(font: &FontRef<'_>, glyph: GlyphId, ch: char, language: Option<&str>, encoded: bool) -> bool {
    let lookups = vertical_lookups(font, language, true);
    substitute(glyph, &lookups) != glyph || (encoded && lookups.is_empty() && encoded_form(font, ch).is_some())
}

fn vertical_substitution(tag: Tag) -> bool {
    matches!(&tag.to_be_bytes(), b"vert" | b"vrt2")
}

fn has_vertical_design(font: &FontRef<'_>) -> bool {
    font.vmtx().is_ok() || font.gsub().ok().and_then(|table| table.feature_list().ok()).is_some_and(|features| {
        features.feature_records().iter().any(|record| vertical_substitution(record.feature_tag()))
    })
}

fn is_proportional(font: &FontRef<'_>, glyph: GlyphId) -> bool {
    font.hmtx().ok().and_then(|metrics| metrics.advance(glyph))
        .zip(font.head().ok().map(|head| head.units_per_em()))
        .is_some_and(|(advance, em)| advance < em)
}

fn language_features<'a>(features: &FeatureList<'a>, lang: &LangSys<'a>) -> impl Iterator<Item = &'a FeatureRecord> {
    let records = features.feature_records();
    std::iter::once(lang.required_feature_index())
        .chain(lang.feature_indices().iter().map(|i| i.get()))
        .filter_map(|i| records.get(usize::from(i)))
}

fn has_default_vertical_substitution(font: &FontRef<'_>) -> bool {
    let Ok(gsub) = font.gsub() else { return false };
    let (Ok(scripts), Ok(features)) = (gsub.script_list(), gsub.feature_list()) else { return false };
    scripts.script_records().iter()
        .filter(|record| matches!(&record.script_tag().to_be_bytes(), b"DFLT" | b"hani" | b"kana" | b"hang" | b"bopo"))
        .filter_map(|record| record.script(scripts.offset_data()).ok()?.default_lang_sys()?.ok())
        .any(|lang| language_features(&features, &lang).any(|record| vertical_substitution(record.feature_tag())))
}

/// Parley shapes Latin and CJK in horizontal runs before we apply vertical metrics.
/// Its inherited language can miss a Han-specific vertical feature, including the legacy
/// CHN LangSys found in fonts that predate the registered ZHS tag.
fn vertical_lookups<'a>(font: &FontRef<'a>, language: Option<&str>, with_default: bool) -> Vec<SubstitutionSubtables<'a>> {
    use crate::cjk::CjkScript;

    let tags: &[[u8; 4]] = match language.and_then(|l| l.split(['-', '_']).next()) {
        None => &[*b"ZHS ", *b"CHN "],
        Some(l) if l.eq_ignore_ascii_case("zh") || l.eq_ignore_ascii_case("yue") => {
            if crate::cjk::script_order(language)[0] == CjkScript::TraditionalChinese {
                &[*b"ZHT ", *b"ZHH "]
            } else {
                &[*b"ZHS ", *b"CHN "]
            }
        }
        Some(l) if l.eq_ignore_ascii_case("ja") => &[*b"JAN "],
        Some(l) if l.eq_ignore_ascii_case("ko") => &[*b"KOR "],
        _ => &[],
    };
    let Ok(gsub) = font.gsub() else { return Vec::new() };
    let (Ok(scripts), Ok(features), Ok(lookups)) = (gsub.script_list(), gsub.feature_list(), gsub.lookup_list()) else {
        return Vec::new();
    };
    let feature = |lang| language_features(&features, &lang).find(|r| r.feature_tag() == Tag::new(b"vert"));
    for script_tag in [b"hani", b"DFLT", b"kana", b"hang", b"bopo"] {
        let Some(script) = scripts.script_records().iter().find(|r| r.script_tag() == Tag::new(script_tag))
            .and_then(|r| r.script(scripts.offset_data()).ok()) else { continue };
        let localized = || tags.iter().find_map(|tag| {
            let record = script.lang_sys_records().iter().find(|r| r.lang_sys_tag() == Tag::new(tag))?;
            feature(record.lang_sys(script.offset_data()).ok()?)
        });
        let default = || {
            if !with_default { return None; }
            feature(script.default_lang_sys()?.ok()?)
        };
        let selected = if language.is_none() { default().or_else(localized) } else { localized().or_else(default) };
        if let Some(table) = selected
            .and_then(|r| r.feature(features.offset_data()).ok())
        {
            return table.lookup_list_indices().iter()
                .filter_map(|i| lookups.lookups().get(usize::from(i.get())).ok()?.subtables().ok())
                .collect();
        }
    }
    Vec::new()
}

fn substitute(mut glyph: GlyphId, lookups: &[SubstitutionSubtables<'_>]) -> GlyphId {
    for lookup in lookups {
        let SubstitutionSubtables::Single(subtables) = lookup else { continue };
        if let Some(id) = subtables.iter().flatten().find_map(|st| match st {
            SingleSubst::Format1(t) => {
                t.coverage().ok()?.get(glyph)?;
                let id = u16::try_from(glyph.to_u32()).ok()?.wrapping_add_signed(t.delta_glyph_id());
                Some(GlyphId::new(u32::from(id)))
            }
            SingleSubst::Format2(t) => {
                let index = t.coverage().ok()?.get(glyph)?;
                t.substitute_glyph_ids().get(usize::from(index)).map(|id| id.get().into())
            }
        }) {
            glyph = id;
        }
    }
    glyph
}

fn is_fullwidth_bracket(ch: char) -> bool {
    const SMALL_FORM_VARIANTS: std::ops::RangeInclusive<char> = '\u{fe50}'..='\u{fe6f}';
    !SMALL_FORM_VARIANTS.contains(&ch)
        && CodePointMapData::<BidiMirroringGlyph>::new().get(ch).paired_bracket_type != BidiPairedBracketType::None
}

pub(crate) const VERTICAL_METRICS: [&str; 4] = ["vpal", "valt", "vhal", "vchw"];

fn encoded_form(font: &FontRef<'_>, ch: char) -> Option<GlyphId> {
    let presentation = presentation_form(canonical_base(ch))?;
    // A vertical-capable face already lays out its authored CJK compatibility forms.
    if ('\u{fe30}'..='\u{fe4f}').contains(&ch) && has_vertical_design(font) { return None; }
    font.charmap().map(presentation)
}

fn canonical_base(ch: char) -> char {
    match CanonicalDecompositionBorrowed::new().decompose(ch) {
        Decomposed::Singleton(base) => base,
        _ => ch,
    }
}

// The <vertical> compatibility decompositions in UnicodeData.txt, reversed.
// Underlines retain their style where two forms share the same compatibility base.
fn presentation_form(ch: char) -> Option<char> {
    let base = if ('\u{ff01}'..='\u{ff5e}').contains(&ch) {
        char::from_u32(u32::from(ch) - 0xfee0)?
    } else {
        ch
    };
    Some(match base {
        ',' => '\u{fe10}', '、' => '\u{fe11}', '。' => '\u{fe12}',
        ':' => '\u{fe13}', ';' => '\u{fe14}', '!' => '\u{fe15}', '?' => '\u{fe16}',
        '〖' => '\u{fe17}', '〗' => '\u{fe18}', '…' => '\u{fe19}',
        '‥' => '\u{fe30}', '—' => '\u{fe31}', '–' => '\u{fe32}', '_' => '\u{fe33}', '﹏' => '\u{fe34}',
        '(' => '\u{fe35}', ')' => '\u{fe36}', '{' => '\u{fe37}', '}' => '\u{fe38}',
        '〔' => '\u{fe39}', '〕' => '\u{fe3a}', '【' => '\u{fe3b}', '】' => '\u{fe3c}',
        '《' => '\u{fe3d}', '》' => '\u{fe3e}', '〈' => '\u{fe3f}', '〉' => '\u{fe40}',
        '「' => '\u{fe41}', '」' => '\u{fe42}', '『' => '\u{fe43}', '』' => '\u{fe44}',
        '[' => '\u{fe47}', ']' => '\u{fe48}',
        _ => return None,
    })
}

fn has_legacy_cjk_metrics(font: &FontRef<'_>) -> bool {
    font.vmtx().is_err() && font.os2().ok().and_then(|os2| os2.ul_code_page_range_1())
        .is_some_and(|pages| pages & CJK_CODE_PAGES != 0)
}

/// Place sideways Roman glyphs against the CJK body, separately from horizontal line metrics.
pub(crate) fn sideways_baseline(face: &FontData, size: f32, fallback: f32) -> f32 {
    let Ok(font) = FontRef::from_index(face.data.as_ref(), face.index) else { return fallback };
    let (Ok(os2), Ok(head)) = (font.os2(), font.head()) else { return fallback; };
    if os2.ul_code_page_range_1().is_none_or(|pages| pages & CJK_CODE_PAGES == 0) { return fallback; }
    let em = f32::from(head.units_per_em()).max(1.0);
    let top = f32::from(os2.s_typo_ascender());
    let bottom = f32::from(os2.s_typo_descender());
    if top - bottom == em && (0.75..=0.90).contains(&(top / em)) {
        if font.vmtx().is_err() && let Ok(hhea) = font.hhea() {
            // Horizontal-only CJK faces still use their Roman line centre.
            return (f32::from(hhea.ascender().to_i16()) + f32::from(hhea.descender().to_i16())) * size / (2.0 * em);
        }
        return (top + bottom) * size / (2.0 * em);
    }
    // With no usable one-em CJK box, the composer synthesizes the Roman baseline.
    // Measured against varying ascenders, cap heights, and units per em in Photoshop.
    size * 0.355
}

/// Metrics from the exact face and variation selected by the shaper, in unscaled text pixels.
pub(crate) struct VerticalMetrics<'a> {
    font: FontRef<'a>,
    cmap_fallback: bool,
    localized: Vec<SubstitutionSubtables<'a>>,
    coords: Vec<NormalizedCoord>,
    size: f32,
    scale: f32,
    em_origin: f32,
    // The proportional metrics feature uses single adjustments on the substituted glyph.
    lookups: Vec<PositionSubtables<'a>>,
}

#[derive(Default)]
pub(crate) struct VerticalPosition {
    pub(crate) x: f32,
    pub(crate) y: f32,
    pub(crate) advance: f32,
}

impl<'a> VerticalMetrics<'a> {
    pub(crate) fn new(fd: &'a FontData, coords: &[i16], size: f32, style: &CharStyle) -> Option<Self> {
        let font = FontRef::from_index(fd.data.as_ref(), fd.index).ok()?;
        let coords: Vec<_> = coords.iter().map(|&c| NormalizedCoord::from_bits(c)).collect();
        let m = font.metrics(Size::unscaled(), LocationRef::new(&coords));
        let em = f32::from(m.units_per_em).max(1.0);
        let total = m.ascent - m.descent;
        // A CJK em box and a horizontal line box are independent. Legacy faces with no
        // vertical metrics may describe the CJK box in OS/2, even when USE_TYPO_METRICS is off.
        let origin = if has_legacy_cjk_metrics(&font) {
            font.os2().ok().and_then(|os2| {
                let top = f32::from(os2.s_typo_ascender());
                let bottom = f32::from(os2.s_typo_descender());
                (top > 0.0 && bottom <= 0.0 && top - bottom == em).then_some(top / em)
            }).unwrap_or(0.88)
        } else if total > 0.0 { (m.ascent / total).clamp(0.0, 1.0) } else { 0.88 };
        let feature = style.features.iter().rev().find(|f| {
            f.tag == "vpal" || (f.value != 0 && VERTICAL_METRICS.contains(&f.tag.as_str()))
        });
        let tag = feature.map_or(Some(Tag::new(b"vpal")), |f| {
            if f.value == 0 { None } else { f.tag.parse().ok() }
        });
        let lookups = tag.map_or_else(Vec::new, |tag| metrics_lookups(&font, tag));
        let has_default = has_default_vertical_substitution(&font);
        let localized = if !has_default || style.language.is_some() {
            vertical_lookups(&font, style.language.as_deref(), false)
        } else {
            Vec::new()
        };
        let cmap_fallback = !has_default && localized.is_empty();
        Some(Self { font, cmap_fallback, localized, coords, size, scale: size / em, em_origin: size * origin, lookups })
    }

    /// Keep the shaper's substitutions. For an unchanged glyph, try the font's localized
    /// vertical substitution before using its optional encoded presentation form.
    pub(crate) fn glyph(&self, shaped: u32, source: Option<char>) -> u32 {
        if self.localized.is_empty() && !self.cmap_fallback { return shaped; }
        let Some(ch) = source else { return shaped };
        let cmap = self.font.charmap();
        let id = GlyphId::new(shaped);
        if cmap.map(ch) != Some(id) {
            return shaped;
        }
        let alternate = substitute(id, &self.localized);
        if alternate != id || !self.cmap_fallback {
            return alternate.to_u32();
        }
        encoded_form(&self.font, ch).map_or(shaped, |g| g.to_u32())
    }

    pub(crate) fn advance(&self, glyph: u32) -> f32 {
        let g = GlyphId::new(glyph);
        // A spacing character can share a glyph with a combining mark. Keep the
        // mark non-spacing even when the font supplies a default vertical cell.
        const MARK_CLASS: u16 = 3;
        if self.font.hmtx().ok().and_then(|h| h.advance(g)) == Some(0)
            && self.font.gdef().ok().and_then(|d| d.glyph_class_def()?.ok()).is_some_and(|classes| classes.get(g) == MARK_CLASS)
        {
            return 0.0;
        }
        let advance = self.font.vmtx().ok().and_then(|v| v.advance(g)).map_or(self.size, |h| {
            let delta = self.font.vvar().ok().and_then(|v| v.advance_height_delta(g, &self.coords).ok());
            (f32::from(h) + delta.map_or(0.0, |d| d.to_f32())) * self.scale
        });
        advance + self.position_adjustment(g).advance
    }

    pub(crate) fn origin(&self, glyph: u32) -> (f32, f32) {
        let g = GlyphId::new(glyph);
        let location = LocationRef::new(&self.coords);
        let metrics = self.font.glyph_metrics(Size::unscaled(), location);
        let width = metrics.advance_width(g).map_or(self.size, |w| w * self.scale);
        let y = if let Ok(vorg) = self.font.vorg() {
            let delta = self.font.vvar().ok().and_then(|v| v.v_org_delta(g, &self.coords).ok());
            Some((f32::from(vorg.vertical_origin_y(g)) + delta.map_or(0.0, |d| d.to_f32())) * self.scale)
        } else {
            self.font.vmtx().ok().and_then(|v| v.side_bearing(g)).and_then(|bearing| {
                let delta = self.font.vvar().ok().and_then(|v| v.tsb_delta(g, &self.coords).ok());
                let top = self.outline_top(g, location).or_else(|| metrics.bounds(g).map(|bounds| bounds.y_max * self.scale))?;
                Some(top + (f32::from(bearing) + delta.map_or(0.0, |d| d.to_f32())) * self.scale)
            })
        };
        let p = self.position_adjustment(g);
        (p.x - width * 0.5, y.unwrap_or(self.em_origin) - p.y)
    }

    pub(crate) fn horizontal_advance(&self, glyph: u32) -> Option<f32> {
        self.font.glyph_metrics(Size::unscaled(), LocationRef::new(&self.coords))
            .advance_width(GlyphId::new(glyph)).map(|width| width * self.scale)
    }

    fn outline_top(&self, glyph: GlyphId, location: LocationRef<'_>) -> Option<f32> {
        // Use the ink edge for the bearing; the TrueType box may include off-curve points.
        let outline = self.font.outline_glyphs().get(glyph)?;
        let mut bounds = Bounds::default();
        let settings = DrawSettings::unhinted(Size::new(self.size), location);
        outline.draw(settings, &mut Pen::new(&mut bounds, Xform::IDENTITY)).ok()?;
        bounds.rect.map(|rect| rect[3] as f32)
    }

    fn position_adjustment(&self, g: GlyphId) -> VerticalPosition {
        let mut p = VerticalPosition::default();
        if self.lookups.is_empty() { return p; }
        let store = self.font.gdef().ok().and_then(|t| t.item_var_store()?.ok());
        let context = ValueContext::new().with_var_store(store).with_coords(&self.coords);
        for lookup in &self.lookups {
            let PositionSubtables::Single(subtables) = lookup else { continue };
            if let Some(v) = subtables.iter().flatten().find_map(|s| {
                let (record, data) = metric_value(s, g)?;
                record.value(data, &context).ok()
            }) {
                p.x += (f32::from(v.x_placement) + v.x_placement_delta as f32) * self.scale;
                p.y += (f32::from(v.y_placement) + v.y_placement_delta as f32) * self.scale;
                p.advance += (f32::from(v.y_advance) + v.y_advance_delta as f32) * self.scale;
            }
        }
        p
    }
}

fn metrics_lookups<'a>(font: &FontRef<'a>, tag: Tag) -> Vec<PositionSubtables<'a>> {
    let Ok(gpos) = font.gpos() else { return Vec::new() };
    let Ok(features) = gpos.feature_list() else { return Vec::new() };
    let Ok(lookups) = gpos.lookup_list() else { return Vec::new() };
    let mut indices = Vec::new();
    for record in features.feature_records().iter().filter(|r| r.feature_tag() == tag) {
        if let Ok(feature) = record.feature(features.offset_data()) {
            indices.extend(feature.lookup_list_indices().iter().map(|n| n.get()));
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices.into_iter().filter_map(|i| lookups.lookups().get(usize::from(i)).ok()?.subtables().ok()).collect()
}

fn metric_value(s: SinglePos<'_>, glyph: GlyphId) -> Option<(ValueRecord, TableData<'_>)> {
    match s {
        SinglePos::Format1(t) => {
            t.coverage().ok()?.get(glyph)?;
            Some((t.value_record(), t.offset_data()))
        }
        SinglePos::Format2(t) => {
            let i = t.coverage().ok()?.get(glyph)?;
            Some((t.value_records().get(usize::from(i)).ok()?, t.offset_data()))
        }
    }
}
