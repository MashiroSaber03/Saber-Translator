//! Shaping and layout of a text layer's model into positioned glyphs (text-space pixels, y down,
//! first baseline of point text at y = 0).
//!
//! Each paragraph is shaped and line-broken by parley (HarfRust shaping, Unicode bidi and line
//! breaking); paragraphs are then stacked with Photoshop's rules: baseline-to-baseline distance =
//! the largest leading on the line (auto leading = paragraph factor × size), space before/after,
//! indents, and point-text alignment around the anchor.
//!
//! **Vertical type** (tategaki) is laid out in the same *line space* and then turned 90° clockwise
//! into text space: a line becomes a column running top to bottom, and successive columns advance
//! right to left. Within a column, CJK characters stay upright (centred on the column, placed with
//! the font's vertical metrics: `VORG`, `vhea`/`vmtx`, else the ideographic em box) and use the
//! font's vertical alternates (`vert`) for punctuation, brackets and the long-vowel mark; other
//! text (Latin, digits) runs rotated 90° clockwise, Photoshop's default. Lines, clusters, carets
//! and hit tests stay in line space; [`TextLayout::to_text`] / [`TextLayout::to_line`] convert.

use std::borrow::Cow;
use std::ops::Range;

use parley::{
    Alignment, AlignmentOptions, FontData, FontFamily, FontFeatures, FontStyle, FontVariations, FontWeight, IndentOptions, Layout, LayoutContext,
    PositionedLayoutItem, StyleProperty,
};
use photocraft_doc::TextLayer;
use photocraft_doc::text::{Caps, CharStyle, Kerning, Orientation, TextAlign, TextDirection, TextShape};
use skrifa::raw::types::Tag;

use crate::fonts::FontDb;

/// Photoshop synthesizes small caps for faces without an OpenType `smcp` table. Keep the same
/// readable hierarchy for every font instead of silently rendering lowercase text unchanged.
const SYNTHETIC_SMALL_CAPS_SCALE: f32 = 0.7;

/// Photoshop's default tab stops: every half inch, measured from the anchor of point text or the
/// left edge of a paragraph box. PSD type carries no custom tab stops, so these are the only ones.
pub const DEFAULT_TAB_STOP_PT: f32 = 36.0;

/// The rendering path for a `SmallCaps` character style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SmallCapsMode {
    None,
    OpenType,
    Synthetic,
}

/// Index of the run containing byte offset `off`, given each run's start offset (non-decreasing):
/// the last run starting at or before `off`, or 0 when none does. Binary search, so a layer with
/// many runs doesn't make per-character lookups quadratic.
fn run_index_at(run_starts: &[usize], off: usize) -> usize {
    run_starts.partition_point(|&s| s <= off).saturating_sub(1)
}

/// Whether the style's selected face has an `smcp` feature (a font load and GSUB parse).
fn probe_smcp(fonts: &mut FontDb, style: &CharStyle) -> bool {
    #[cfg(test)]
    SMCP_PROBES.with(|c| c.set(c.get() + 1));
    fonts.selected_face_has_feature(&style.font_family, style.weight, style.italic, Tag::new(b"smcp"))
}

#[cfg(test)]
thread_local! {
    static SMCP_PROBES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn small_caps_mode(caps: Caps, face_has_smcp: bool) -> SmallCapsMode {
    match caps {
        Caps::SmallCaps if face_has_smcp => SmallCapsMode::OpenType,
        Caps::SmallCaps => SmallCapsMode::Synthetic,
        _ => SmallCapsMode::None,
    }
}

/// Index of the character run whose style a glyph uses, plus the vertical-type class of its
/// characters ([`VClass`] as `u8`; always 0 in horizontal type).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RunBrush(pub u32, pub u8);

/// How a character is set in vertical type (a simplification of Unicode's
/// `Vertical_Orientation`, UAX #50).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VClass {
    /// Rotated 90° clockwise (Latin, digits, most symbols).
    Rotate = 0,
    /// Upright (ideographs, kana, Hangul, full-width forms, CJK punctuation such as 、。),
    /// using a vertical alternate when the font has one.
    Upright = 1,
    /// Upright through its vertical alternate (brackets, the long-vowel mark, dashes); rotated
    /// when the font has none.
    AlternateOrRotate = 2,
}

// Comic emphasis marks that override the usual rotation of half-width text.
fn emphasis_override(c: char) -> bool {
    matches!(c, '!' | '?' | '¡' | '¿' | '؟' | '‼' | '⁇' | '⁈' | '⁉' | '﹖' | '﹗')
}

/// The vertical-type class of a character.
pub fn vertical_class(c: char) -> VClass {
    let u = c as u32;
    let alt = matches!(
        u,
        0x2014..=0x2016
            | 0x2025
            | 0x2026
            | 0x2329
            | 0x232A
            | 0x3008..=0x3011
            | 0x3014..=0x301F
            | 0x3030
            | 0x30A0
            | 0x30FC
            | 0xFE59..=0xFE5E
            | 0xFF08
            | 0xFF09
            | 0xFF0D
            | 0xFF1A..=0xFF1E
            | 0xFF3B
            | 0xFF3D
            | 0xFF3F
            | 0xFF5B..=0xFF60
            | 0xFFE3
    );
    if alt {
        return VClass::AlternateOrRotate;
    }
    // Half-width katakana and Hangul rotate like other half-width text.
    if (0xFF61..=0xFFDC).contains(&u) {
        return VClass::Rotate;
    }
    let upright = emphasis_override(c)
        || crate::cjk::classify(c).is_some()
        || matches!(u, 0xA7 | 0xA9 | 0xAE | 0xB1 | 0xBC..=0xBE | 0xD7 | 0xF7 | 0x2E80..=0x2EFF | 0xFE10..=0xFE1F | 0xFE30..=0xFE4F | 0x1F000..=0x1FAFF);
    if upright { VClass::Upright } else { VClass::Rotate }
}

/// How a placed glyph's outline is oriented in text space.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GlyphOrient {
    /// Horizontal type: origin on the baseline, outline upright.
    #[default]
    Horizontal,
    /// Vertical type, upright: the origin is the glyph's horizontal origin (left end of its
    /// advance, on its baseline); baseline shift moves it right.
    Upright,
    /// Vertical type, rotated 90° clockwise about its origin (the baseline runs downwards).
    Rotated,
}

/// A font instance used by some glyphs.
#[derive(Clone, Debug)]
pub struct GlyphFace {
    pub font: FontData,
    /// Normalised variation coordinates (F2Dot14 bits).
    pub coords: Vec<i16>,
    pub size_px: f32,
    /// Synthetic bold requested by font matching (face lacks the weight).
    pub embolden: bool,
    /// Synthetic oblique angle in degrees (face lacks italics).
    pub skew_deg: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PlacedGlyph {
    pub face: u32,
    pub id: u32,
    /// Pen position (text space px): x along the baseline, y = baseline (before baseline shift).
    pub x: f32,
    pub y: f32,
    /// Character run index (into [`TextLayout::styles`]).
    pub style: u32,
    /// Orientation of the outline (vertical type).
    pub orient: GlyphOrient,
}

/// Underline/strikethrough rectangle (text space px).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecorationRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub style: u32,
}

/// A line (a column in vertical type) in line space: `x` runs along the line and `baseline` is the
/// cross-line position. In vertical type `baseline` is the column's centre line and
/// `ascent`/`descent` are the column's half widths on either side.
#[derive(Clone, Debug, PartialEq)]
pub struct LineInfo {
    /// Byte range in the layer text (without the paragraph break).
    pub range: Range<usize>,
    pub baseline: f32,
    /// Visual extent along the baseline.
    pub x0: f32,
    pub x1: f32,
    pub ascent: f32,
    pub descent: f32,
    pub paragraph: usize,
}

/// A grapheme cluster (caret stops, hit testing, selection), in line space.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterInfo {
    pub range: Range<usize>,
    pub x: f32,
    pub advance: f32,
    pub line: usize,
    pub rtl: bool,
}

/// Result of laying out a text layer.
#[derive(Clone, Debug, Default)]
pub struct TextLayout {
    pub faces: Vec<GlyphFace>,
    pub glyphs: Vec<PlacedGlyph>,
    pub decorations: Vec<DecorationRect>,
    pub lines: Vec<LineInfo>,
    pub clusters: Vec<ClusterInfo>,
    /// Resolved character styles (one per run of the layer).
    pub styles: Vec<CharStyle>,
    /// Pixels per point used (dpi / 72).
    pub px_per_pt: f32,
    /// Vertical type: line space is turned 90° clockwise into text space.
    pub vertical: bool,
    /// Applied block alignment in line space; export can bake this into its copy's transform.
    pub block_offset: f32,
}

impl TextLayout {
    /// Line space → text space: the identity for horizontal type; for vertical type the line
    /// direction points down and the cross-line direction (towards the next line) points left.
    pub fn to_text(&self, x: f32, y: f32) -> (f32, f32) {
        if self.vertical { (-y, x) } else { (x, y) }
    }

    /// Text space → line space (inverse of [`Self::to_text`]).
    pub fn to_line(&self, x: f32, y: f32) -> (f32, f32) {
        if self.vertical { (y, -x) } else { (x, y) }
    }

    /// Logical bounds (x0, y0, x1, y1) of all lines, text space.
    pub fn bounds(&self) -> Option<[f32; 4]> {
        let b = self.line_bounds()?;
        Some(if self.vertical { [-b[3], b[0], -b[1], b[2]] } else { b })
    }

    /// Logical bounds of all lines in line space (equal to [`Self::bounds`] for horizontal type).
    pub fn line_bounds(&self) -> Option<[f32; 4]> {
        self.lines.iter().fold(None, |acc, l| {
            let r = [l.x0, l.baseline - l.ascent, l.x1, l.baseline + l.descent];
            Some(match acc {
                None => r,
                Some(a) => [a[0].min(r[0]), a[1].min(r[1]), a[2].max(r[2]), a[3].max(r[3])],
            })
        })
    }

    /// Byte offset of the caret nearest to a text-space point.
    pub fn hit_test(&self, x: f32, y: f32) -> usize {
        let (x, y) = self.to_line(x, y);
        self.hit_test_line(x, y)
    }

    /// The visual line nearest to a line-space ordinate, including its ascent and descent.
    /// Pointer hit testing uses this even where adjacent lines share a caret byte offset.
    fn closest_line(&self, y: f32) -> Option<(usize, &LineInfo)> {
        self.lines.iter().enumerate().min_by(|a, b| {
            let d = |l: &LineInfo| {
                if y < l.baseline - l.ascent {
                    l.baseline - l.ascent - y
                } else if y > l.baseline + l.descent {
                    y - l.baseline - l.descent
                } else {
                    0.0
                }
            };
            d(a.1).total_cmp(&d(b.1))
        })
    }

    /// Byte offset of the caret nearest to a line-space point.
    pub fn hit_test_line(&self, x: f32, y: f32) -> usize {
        let Some((li, _)) = self.closest_line(y) else {
            return 0;
        };
        self.hit_test_x_on_line(li, x)
    }

    fn hit_test_x_on_line(&self, line_index: usize, x: f32) -> usize {
        let Some(line) = self.lines.get(line_index) else { return 0 };
        let mut best = (f32::MAX, line.range.end);
        for c in self.clusters.iter().filter(|c| c.line == line_index) {
            let mid = c.x + c.advance / 2.0;
            let (before, after) = if c.rtl { (c.range.end, c.range.start) } else { (c.range.start, c.range.end) };
            let (dist, off) = if x < mid { ((x - c.x).abs(), before) } else { ((x - c.x - c.advance).abs(), after) };
            if dist < best.0 {
                best = (dist, off);
            }
        }
        best.1
    }

    /// The caret for a byte offset as a text-space segment (its two end points).
    pub fn caret_segment(&self, offset: usize) -> [(f32, f32); 2] {
        let (x, top, bottom) = self.caret(offset);
        [self.to_text(x, top), self.to_text(x, bottom)]
    }

    /// Caret geometry for a byte offset: (x, top, bottom) in line space (the same as text space
    /// for horizontal type; see [`Self::caret_segment`]).
    pub fn caret(&self, offset: usize) -> (f32, f32, f32) {
        for c in &self.clusters {
            if c.range.start == offset
                && let Some(l) = self.lines.get(c.line)
            {
                let x = if c.rtl { c.x + c.advance } else { c.x };
                return (x, l.baseline - l.ascent, l.baseline + l.descent);
            }
        }
        // End of a line (or empty line): after the last cluster of the line containing it.
        let li = self.lines.iter().position(|l| offset >= l.range.start && offset <= l.range.end).unwrap_or(self.lines.len().saturating_sub(1));
        match self.lines.get(li) {
            Some(l) => {
                let x = self
                    .clusters
                    .iter()
                    .filter(|c| c.line == li && c.range.end == offset)
                    .map(|c| if c.rtl { c.x } else { c.x + c.advance })
                    .next()
                    .unwrap_or(l.x1);
                (x, l.baseline - l.ascent, l.baseline + l.descent)
            }
            None => (0.0, 0.0, 0.0),
        }
    }
}

/// Byte offset of character `index` (the text length when `index` is past the end).
pub fn byte_index(text: &str, index: usize) -> usize {
    text.char_indices().nth(index).map_or(text.len(), |(b, _)| b)
}

/// Character index of byte `byte`, floored to a char boundary so a bad offset never panics.
pub fn char_index(text: &str, byte: usize) -> usize {
    let mut byte = byte.min(text.len());
    while byte > 0 && !text.is_char_boundary(byte) {
        byte -= 1;
    }
    text[..byte].chars().count()
}

/// Line containing a byte offset. At a shared wrap boundary the caret belongs to
/// the following line, while an offset at the end of the text stays on the last.
pub fn line_index(layout: &TextLayout, byte: usize) -> usize {
    layout.lines.iter().rposition(|ln| byte >= ln.range.start && byte <= ln.range.end).unwrap_or_else(|| layout.lines.len().saturating_sub(1))
}

/// Nearest caret to a text-space point: character index and the line it sits on.
pub fn hit_char(layout: &TextLayout, text: &str, x: f32, y: f32) -> (usize, usize) {
    let byte = layout.hit_test(x, y);
    // At a soft-wrap boundary either visual line can share this byte offset. The
    // pointer's line, not the caret's preferred (following) line, is authoritative.
    let (_, line_y) = layout.to_line(x, y);
    let line = layout.closest_line(line_y).map_or_else(|| line_index(layout, byte), |(index, _)| index);
    (char_index(text, byte), line)
}

/// Text-space point inside the laid-out line boxes, expanded by `slop` px on every side.
pub fn text_point_inside(layout: &TextLayout, x: f32, y: f32, slop: f32) -> bool {
    let slop = if slop.is_finite() { slop.max(0.0) } else { 0.0 };
    layout.bounds().is_some_and(|b| x >= b[0] - slop && x <= b[2] + slop && y >= b[1] - slop && y <= b[3] + slop)
}

/// Character index of the word boundary before (`forward` is false) or after `idx`.
///
/// A word is a run of alphanumeric characters, the rule the Type tool has always used.
pub fn word_boundary(text: &str, idx: usize, forward: bool) -> usize {
    let chars: Vec<char> = text.chars().collect();
    let mut i = idx.min(chars.len());
    if forward {
        while i < chars.len() && !chars[i].is_alphanumeric() {
            i += 1;
        }
        while i < chars.len() && chars[i].is_alphanumeric() {
            i += 1;
        }
    } else {
        while i > 0 && !chars[i - 1].is_alphanumeric() {
            i -= 1;
        }
        while i > 0 && chars[i - 1].is_alphanumeric() {
            i -= 1;
        }
    }
    i
}

/// Caret on the neighbouring line (`dir` < 0 previous, otherwise next), keeping `x`
/// (line space: the position along the line). Past the first or last line the caret
/// goes to the start or end of the text. Line space is the same for both orientations,
/// so a column of vertical type steps the same way a line of horizontal type does.
pub fn line_step(layout: &TextLayout, text: &str, idx: usize, x: f32, dir: i32) -> usize {
    let n = text.chars().count();
    let idx = idx.min(n);
    if layout.lines.is_empty() {
        return idx;
    }
    let current_line = line_index(layout, byte_index(text, idx));
    let target_line = if dir < 0 {
        let Some(target) = current_line.checked_sub(1) else { return 0 };
        target
    } else {
        let Some(target) = current_line.checked_add(1).filter(|target| *target < layout.lines.len()) else { return n };
        target
    };
    char_index(text, layout.hit_test_x_on_line(target_line, x))
}

/// Line start (`end` false) or end for the line containing `idx`, as a character index.
pub fn line_edge(layout: &TextLayout, text: &str, idx: usize, end: bool) -> usize {
    let n = text.chars().count();
    let idx = idx.min(n);
    let byte = byte_index(text, idx);
    let line = layout.lines.get(line_index(layout, byte));
    line.map_or(idx, |ln| char_index(text, if end { ln.range.end } else { ln.range.start }))
}

const LRM: &str = "\u{200E}";
const RLM: &str = "\u{200F}";
/// A line break inside a paragraph (Shift+Return), stored as U+0003 in PSD type.
pub(crate) const FORCED_LINE_BREAK: char = '\u{3}';

pub(crate) struct Layouter {
    lcx: LayoutContext<RunBrush>,
    optical: crate::optical::Cache,
}

/// A cluster of a line in visual order, for kerning.
struct KernSlot {
    /// Index of the run in the paragraph layout.
    run: usize,
    /// Byte offset in the paragraph text.
    start: usize,
    first: Option<u32>,
    last: Option<u32>,
    font: FontData,
    coords: Vec<i16>,
    size: f32,
    rtl: bool,
    blank: bool,
    /// Advance along the line (px, horizontal scale included).
    advance: f32,
    /// A tab, shaped as a space; it advances to the next tab stop.
    tab: bool,
    /// Vertical type, upright glyph (its outline doesn't run along the column).
    upright: bool,
}

impl Layouter {
    pub fn new() -> Self {
        Self { lcx: LayoutContext::new(), optical: crate::optical::Cache::default() }
    }

    pub fn layout(&mut self, fonts: &mut FontDb, t: &TextLayer, dpi: f32) -> TextLayout {
        let k = if dpi > 0.0 { dpi / 72.0 } else { 1.0 };
        let runs = t.char_runs();
        let paras = t.paragraph_runs();
        let vertical = t.orientation == Orientation::Vertical;
        let mut out = TextLayout { px_per_pt: k, vertical, ..Default::default() };
        // Resolve families (PostScript names from PSDs, unknown families).
        for r in &runs {
            let mut s = r.style.clone();
            if let Some(ps) = s.postscript_name.clone()
                && !fonts.faces(&s.font_family).iter().any(|f| f.postscript_name.as_deref() == Some(&ps))
            {
                let f = fonts.resolve_postscript(&ps);
                // An exact face match wins; a guessed family only fills a missing family.
                if f.exact || (!fonts.has_family(&s.font_family) && fonts.has_family(&f.family)) {
                    s.font_family = f.family;
                    s.weight = f.weight;
                    s.italic = f.italic;
                }
            }
            // Drawn with a fallback for now; a host that serves this family fetches it (`served`).
            if !fonts.has_family(&s.font_family) {
                crate::served::request(&s.font_family);
            }
            fonts.select_named_face(&mut s);
            out.styles.push(s);
        }
        let run_starts: Vec<usize> = runs
            .iter()
            .scan(0, |a, r| {
                let s = *a;
                *a += r.len;
                Some(s)
            })
            .collect();
        let style_at = |off: usize| run_index_at(&run_starts, off);
        let para_starts: Vec<usize> = paras
            .iter()
            .scan(0, |a, r| {
                let s = *a;
                *a += r.len;
                Some(s)
            })
            .collect();
        let para_style_at = |off: usize| &paras[para_starts.iter().rposition(|&s| s <= off).unwrap_or(0)].style;

        let text = &t.text;
        let (box_rect, is_box) = match t.shape {
            TextShape::Box { x, y, width, height } => ((x, y, width, height), true),
            // Path text is laid out as one point-text line, then bent onto the path.
            TextShape::Point | TextShape::Path { .. } => ((0.0, 0.0, 0.0, 0.0), false),
        };
        let mut prev_baseline: Option<f32> = None;
        let mut pending_space = 0.0f32;
        let mut stop = false;
        // Fixed for the whole layout, and the font probe only matters for styles that use small
        // caps. A real `smcp` substitution must receive lowercase source text; only the fallback
        // path uppercases it and scales it down.
        let small_caps: Vec<SmallCapsMode> = out
            .styles
            .iter()
            .map(|style| match style.caps {
                Caps::SmallCaps => small_caps_mode(style.caps, probe_smcp(fonts, style)),
                _ => SmallCapsMode::None,
            })
            .collect();
        for (pi, prange) in split_paragraphs(text).into_iter().enumerate() {
            if stop {
                break;
            }
            let ps = para_style_at(prange.start).clone();
            let content_end = strip_break(text, &prange);
            let content = &text[prange.start..content_end];
            let prefix = match ps.direction {
                TextDirection::Auto => "",
                TextDirection::Ltr => LRM,
                TextDirection::Rtl => RLM,
            };
            let mut ptext = String::with_capacity(prefix.len() + content.len());
            ptext.push_str(prefix);
            // Ranges in `ptext` for lowercase characters rendered as synthetic small caps.
            // They have the same UTF-8 length as their uppercase form, so the layer's byte-based
            // run and caret offsets remain unchanged.
            let mut synthetic_small_caps: Vec<(Range<usize>, f32)> = Vec::new();
            // Offsets in `ptext` of the spaces standing in for tabs.
            let mut tabs: Vec<usize> = Vec::new();
            let tab_px = DEFAULT_TAB_STOP_PT * k;
            for (i, ch) in content.char_indices() {
                // A forced line break ends the line but not the paragraph. The line breaker knows
                // it as a newline, which has the same length, so text offsets don't move.
                if ch == FORCED_LINE_BREAK {
                    ptext.push('\n');
                    continue;
                }
                // Imported PSD text can contain literal tab controls. Font shaping may
                // render those as .notdef boxes; a space preserves the one-byte source
                // and style/caret offsets while supplying a real whitespace advance. The space
                // is widened to its tab stop after line breaking (`tab_px` below).
                if ch == '\t' {
                    tabs.push(ptext.len());
                    ptext.push(' ');
                    continue;
                }
                let style = &out.styles[style_at(prange.start + i)];
                let caps = style.caps;
                if caps == Caps::AllCaps || small_caps[style_at(prange.start + i)] == SmallCapsMode::Synthetic {
                    let up: String = ch.to_uppercase().collect();
                    if up.len() == ch.len_utf8() {
                        ptext.push_str(&up);
                        if caps == Caps::SmallCaps && ch.is_lowercase() {
                            let end = ptext.len();
                            synthetic_small_caps.push((end - up.len()..end, style.size_pt * k * SYNTHETIC_SMALL_CAPS_SCALE));
                        }
                        continue;
                    }
                }
                ptext.push(ch);
            }
            let first_style = &out.styles[style_at(prange.start)];
            let first_px = first_style.size_pt * k;
            // Scripts the loaded fonts may lack (Arabic, Japanese, …): a host serving a font for them
            // fetches it (`served`); it joins the fallback stack once it arrives.
            crate::served::request_for_text(&ptext);
            let fallback: Vec<String> = fonts.fallback_stack().map(str::to_string).collect();
            // Shapes the paragraph, each tab widened by its letter spacing in `tab_spacing` (px).
            let build = |lcx: &mut LayoutContext<RunBrush>,
                         fcx: &mut parley::FontContext,
                         tab_spacing: &[f32],
                         vertical_padding: &[(Range<usize>, f32)]|
             -> Layout<RunBrush> {
                let mut b = lcx.ranged_builder(fcx, &ptext, 1.0, false);
                // Paragraph-start style as the default (covers the direction mark and empty
                // paragraphs), then every run piece intersecting this paragraph.
                let si0 = style_at(prange.start);
                for p in style_props(&out.styles[si0], k, &fallback, si0 as u32, small_caps[si0]) {
                    b.push_default(p);
                }
                for (ri, st) in out.styles.iter().enumerate() {
                    let rs = run_starts[ri];
                    let (a, z) = (rs.max(prange.start), (rs + runs[ri].len).min(content_end));
                    if a >= z {
                        continue;
                    }
                    let range = (a - prange.start + prefix.len())..(z - prange.start + prefix.len());
                    for p in style_props(st, k, &fallback, ri as u32, small_caps[ri]) {
                        b.push(p, range.clone());
                    }
                    if vertical {
                        // Upright pieces get the vertical alternates and their class in the brush.
                        let piece = ptext.get(range.clone()).unwrap_or("");
                        let mut seg: Option<(usize, VClass)> = None;
                        let flush = |b: &mut parley::RangedBuilder<'_, RunBrush>, from: usize, to: usize, cls: VClass| {
                            if cls == VClass::Rotate || from >= to {
                                return;
                            }
                            let mut feats = feature_list(st, small_caps[ri]);
                            feats.push("\"vert\" 1".into());
                            let r = (range.start + from)..(range.start + to);
                            b.push(StyleProperty::FontFeatures(FontFeatures::Source(Cow::Owned(feats.join(", ")))), r.clone());
                            b.push(StyleProperty::Brush(RunBrush(ri as u32, cls as u8)), r);
                        };
                        for (i, ch) in piece.char_indices() {
                            let cls = vertical_class(ch);
                            match seg {
                                Some((_, c)) if c == cls => {}
                                Some((from, c)) => {
                                    flush(&mut b, from, i, c);
                                    seg = Some((i, cls));
                                }
                                None => seg = Some((i, cls)),
                            }
                        }
                        if let Some((from, c)) = seg {
                            flush(&mut b, from, piece.len(), c);
                        }
                    }
                }
                for (range, size) in &synthetic_small_caps {
                    b.push(StyleProperty::FontSize(*size), range.clone());
                }
                for (range, padding) in vertical_padding {
                    let st = &out.styles[style_at(prange.start + range.start.saturating_sub(prefix.len()))];
                    b.push(StyleProperty::LetterSpacing(st.tracking / 1000.0 * st.size_pt * k + padding), range.clone());
                }
                for (&at, &spacing) in tabs.iter().zip(tab_spacing) {
                    b.push(StyleProperty::LetterSpacing(spacing), at..at + 1);
                }
                b.build(&ptext)
            };
            let indent_start = ps.start_indent_pt * k;
            let indent_end = ps.end_indent_pt * k;
            let first_indent = ps.first_line_indent_pt * k;
            // Box extent along the lines: width, or height for vertical type (columns).
            let (line_origin, line_len) = if vertical { (box_rect.1, box_rect.3) } else { (box_rect.0, box_rect.2) };
            let avail = if is_box { Some((line_len - indent_start - indent_end).max(1.0)) } else { None };
            let break_lines = |layout: &mut Layout<RunBrush>| {
                if first_indent != 0.0 {
                    layout.set_text_indent(first_indent, IndentOptions::default());
                }
                layout.break_all_lines(avail);
            };
            // Paragraph text: a tab is never wider than a space plus one tab interval, so breaking
            // lines at that width keeps the expanded tabs inside the box, but can wrap early.
            // Break again with the tab widths that layout gives until they agree; if they don't
            // settle, keep the safe layout.
            let wide = if is_box { tab_px } else { 0.0 };
            let tab_spacing = vec![wide; tabs.len()];
            let mut layout = build(&mut self.lcx, &mut fonts.fcx, &tab_spacing, &[]);
            break_lines(&mut layout);
            // Upright marks such as ! need an em down the column, not their narrow horizontal
            // advance. Feed that padding back through shaping so wrapping and carets agree too.
            let mut vertical_padding = Vec::new();
            if vertical && ptext.chars().any(emphasis_override) {
                for line in layout.lines() {
                    for run in line.runs() {
                        for cluster in run.visual_clusters() {
                            let style = cluster.first_style();
                            let mark = ptext.get(cluster.text_range()).and_then(|text| text.chars().next());
                            if !mark.is_some_and(emphasis_override) || cluster.glyphs().count() != 1 {
                                continue;
                            }
                            let tracking = out.styles[style.brush.0 as usize].tracking / 1000.0 * run.font_size();
                            let padding = run.font_size() + tracking - cluster.advance();
                            if padding > 0.01 {
                                vertical_padding.push((cluster.text_range(), padding));
                            }
                        }
                    }
                }
                if !vertical_padding.is_empty() {
                    vertical_padding.sort_by_key(|(range, _)| range.start);
                    layout = build(&mut self.lcx, &mut fonts.fcx, &tab_spacing, &vertical_padding);
                    break_lines(&mut layout);
                }
            }
            if is_box && !tabs.is_empty() {
                let first = tab_widths(&layout, &tabs, &out.styles, indent_start, first_indent, tab_px);
                let mut want: Vec<f32> = first.iter().map(|t| t.width).collect();
                for _ in 0..4 {
                    let spacing: Vec<f32> = first.iter().zip(&want).map(|(t, w)| w / t.scale - (t.advance - tab_px)).collect();
                    let mut next = build(&mut self.lcx, &mut fonts.fcx, &spacing, &vertical_padding);
                    break_lines(&mut next);
                    let got: Vec<f32> = tab_widths(&next, &tabs, &out.styles, indent_start, first_indent, tab_px).iter().map(|t| t.width).collect();
                    if got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 0.01) {
                        layout = next;
                        break;
                    }
                    want = got;
                }
            }
            let alignment = if is_box {
                match ps.align {
                    TextAlign::Left => Alignment::Left,
                    TextAlign::Center => Alignment::Center,
                    TextAlign::Right => Alignment::Right,
                    _ => Alignment::Justify,
                }
            } else {
                Alignment::Left
            };
            layout.align(alignment, AlignmentOptions { align_when_overflowing: !is_box || t.overflow_visible });

            // Stack lines.
            if prev_baseline.is_some() {
                pending_space += ps.space_before_pt * k;
            }
            let nlines = layout.len();
            for (li, line) in layout.lines().enumerate() {
                let m = *line.metrics();
                // Largest leading among the glyph runs of the line.
                let mut leading = 0.0f32;
                for item in line.items() {
                    if let PositionedLayoutItem::GlyphRun(gr) = item {
                        let st = &out.styles[gr.style().brush.0 as usize];
                        let px = gr.run().font_size();
                        leading = leading.max(st.leading_pt.map_or(ps.auto_leading * px, |l| l * k));
                    }
                }
                if leading == 0.0 {
                    let st = &out.styles[style_at(prange.start)];
                    leading = st.leading_pt.map_or(ps.auto_leading * st.size_pt * k, |l| l * k);
                }
                // Vertical type: half the column width (the largest em on the column).
                let col_half = line
                    .items()
                    .filter_map(|it| match it {
                        PositionedLayoutItem::GlyphRun(gr) => Some(gr.run().font_size() * 0.5),
                        _ => None,
                    })
                    .fold(0.0f32, f32::max);
                let col_half = if col_half > 0.0 { col_half } else { first_px * 0.5 };
                let (ascent, descent) = if vertical {
                    (col_half, col_half)
                } else if m.ascent > 0.0 || m.descent > 0.0 {
                    (m.ascent, m.descent)
                } else {
                    (first_px * 0.8, first_px * 0.2)
                };
                let baseline = match prev_baseline {
                    // Photoshop offsets the first vertical baseline by half the capital height.
                    None if is_box && vertical => -(box_rect.0 + box_rect.2) + first_ascent(&line, 'H').map(|h| h * 0.5).unwrap_or(col_half),
                    // Photoshop's "first baseline: ascent": the top of the tallest ascender
                    // (height of 'd') touches the box top, not the font's hhea ascent.
                    None if is_box => box_rect.1 + first_ascent(&line, 'd').unwrap_or(ascent),
                    None => 0.0,
                    Some(b) => b + leading + pending_space,
                };
                pending_space = 0.0;
                // Photoshop keeps a horizontal line whose baseline is inside the box and lets
                // its descenders hang below the bottom edge (#1317: three 35 pt lines in a
                // 91.3 pt box, the last baseline at 88). Requiring the descent to fit dropped
                // that line.
                let overflow = if vertical { baseline + descent > -box_rect.0 + 0.5 } else { baseline > box_rect.1 + box_rect.3 + 0.5 };
                if is_box && !t.overflow_visible && overflow {
                    stop = true;
                    break;
                }
                prev_baseline = Some(baseline);
                let last_line = li + 1 == nlines;
                let adv = m.advance - m.trailing_whitespace;
                let map = |o: usize| (prange.start + o.saturating_sub(prefix.len())).min(content_end);

                // Kerning after each cluster (px), per run in visual cluster order: manual
                // kerning plus optical pair kerning. Applied after shaping (like horizontal
                // scale) in line space, so it moves the following glyphs, carets and the line
                // extent; in vertical type that is along the column. Optical kerning measures
                // horizontal outlines, so it applies to rotated (Latin) glyphs only there.
                let mut slots: Vec<KernSlot> = Vec::new();
                let mut line_runs: Vec<usize> = Vec::new();
                for item in line.items() {
                    let PositionedLayoutItem::GlyphRun(gr) = item else {
                        continue;
                    };
                    let run = gr.run();
                    if line_runs.contains(&run.index()) {
                        continue;
                    }
                    line_runs.push(run.index());
                    let hs = out.styles.get(gr.style().brush.0 as usize).map_or(1.0, |st| if st.horizontal_scale > 0.0 { st.horizontal_scale } else { 1.0 });
                    for c in run.visual_clusters() {
                        let mut gl = c.glyphs();
                        let first = gl.next().map(|g| g.id);
                        let last = gl.last().map(|g| g.id).or(first);
                        slots.push(KernSlot {
                            run: run.index(),
                            start: c.text_range().start,
                            first,
                            last,
                            font: run.font().clone(),
                            coords: run.normalized_coords().to_vec(),
                            size: run.font_size(),
                            rtl: c.is_rtl(),
                            blank: first.is_none() || c.is_space_or_nbsp() || c.text_range().end <= prefix.len(),
                            upright: vertical && c.first_style().brush.1 != VClass::Rotate as u8,
                            advance: c.advance() * hs,
                            tab: tabs.binary_search(&c.text_range().start).is_ok(),
                        });
                    }
                }
                let mut kern_px: Vec<f32> = vec![0.0; slots.len()];
                for j in 0..slots.len().saturating_sub(1) {
                    let (a, b) = (&slots[j], &slots[j + 1]);
                    if a.start < prefix.len() {
                        continue;
                    }
                    let st = &out.styles[style_at(map(a.start))];
                    let next = &out.styles[style_at(map(b.start))];
                    let mut units = if st.kern.is_finite() { st.kern } else { 0.0 };
                    // Optical pairs: both characters optical (a mode change splits shaping
                    // runs, which ends automatic kerning, as with Metrics).
                    if st.kerning == Kerning::Optical
                        && next.kerning == Kerning::Optical
                        && st.kern == 0.0
                        && next.kern == 0.0
                        && !a.rtl
                        && !b.rtl
                        && !a.blank
                        && !b.blank
                        && !a.upright
                        && !b.upright
                        && a.font.data.id() == b.font.data.id()
                        && a.font.index == b.font.index
                        && a.coords == b.coords
                        && (a.size - b.size).abs() < 1e-3
                        && let (Some(l), Some(r)) = (a.last, b.first)
                    {
                        let coords: Vec<skrifa::instance::NormalizedCoord> =
                            a.coords.iter().map(|&c| skrifa::instance::NormalizedCoord::from_bits(c)).collect();
                        if let Some(k) = self.optical.pair(a.font.data.id(), a.font.data.as_ref(), a.font.index, &coords, l, r) {
                            units += k + crate::optical::size_adjust(st.size_pt);
                        }
                    }
                    kern_px[j] = units / 1000.0 * a.size;
                }
                // Tabs: the "kerning" after a tab takes the pen to the next stop, measured from
                // the anchor or the box edge as if the line were left-aligned (alignment then
                // moves the whole line).
                let mut pos = indent_start + if li == 0 { first_indent } else { 0.0 };
                let mut trailing_tabs = 0.0f32;
                for (j, s) in slots.iter().enumerate() {
                    if s.tab {
                        kern_px[j] = next_tab_stop(pos, tab_px) - pos - s.advance;
                        // Trailing whitespace is outside the line's extent (`adv`).
                        if slots[j + 1..].iter().all(|n| n.blank) {
                            trailing_tabs += kern_px[j];
                        }
                    }
                    pos += s.advance + kern_px.get(j).copied().unwrap_or(0.0);
                }
                let line_kern: f32 = kern_px.iter().sum::<f32>() - trailing_tabs;
                // Per run on the line: (run index, glyph → slot, glyphs emitted, first slot whose
                // kerning isn't applied yet).
                let mut cursors: Vec<(usize, Vec<usize>, usize, usize)> = Vec::new();
                for item in line.items() {
                    let PositionedLayoutItem::GlyphRun(gr) = item else {
                        continue;
                    };
                    let run = gr.run();
                    if cursors.iter().any(|c| c.0 == run.index()) {
                        continue;
                    }
                    let base = slots.iter().position(|s| s.run == run.index()).unwrap_or(0);
                    let mut g2s = Vec::new();
                    for (ci, c) in run.visual_clusters().enumerate() {
                        g2s.extend(c.glyphs().map(|_| base + ci));
                    }
                    cursors.push((run.index(), g2s, 0, base));
                }
                // Parley aligned box lines without the kerning.
                let kern_align = if is_box {
                    match ps.align {
                        TextAlign::Center => line_kern / 2.0,
                        TextAlign::Right => line_kern,
                        _ => 0.0,
                    }
                } else {
                    0.0
                };
                let dx = if is_box {
                    let base = line_origin + indent_start - kern_align;
                    let slack = avail.unwrap_or(0.0) - adv - line_kern;
                    base + if last_line {
                        match ps.align {
                            TextAlign::JustifyCenter => slack * 0.5 - m.offset,
                            TextAlign::JustifyRight => slack - m.offset,
                            _ => 0.0,
                        }
                    } else {
                        0.0
                    }
                } else {
                    let target = match ps.align {
                        TextAlign::Center | TextAlign::JustifyCenter => -(adv + line_kern) / 2.0,
                        TextAlign::Right | TextAlign::JustifyRight => -(adv + line_kern),
                        _ => indent_start,
                    };
                    target - m.offset
                };
                let justify_all = is_box && last_line && ps.align == TextAlign::JustifyAll;
                let line_index = out.lines.len();
                let lr = line.text_range();
                let g0 = out.glyphs.len();
                let c0 = out.clusters.len();
                let mut vinfo: Vec<VGlyph> = Vec::new();
                let mut extra = 0.0f32; // horizontal-scale growth and kerning along the line
                let mut seen_runs: Vec<usize> = Vec::new();
                for item in line.items() {
                    let PositionedLayoutItem::GlyphRun(gr) = item else {
                        continue;
                    };
                    let run = gr.run();
                    let si = gr.style().brush.0;
                    let st = &out.styles[si as usize];
                    let hs = if st.horizontal_scale > 0.0 { st.horizontal_scale } else { 1.0 };
                    let synth = run.synthesis();
                    let face = out.faces.len() as u32;
                    out.faces.push(GlyphFace {
                        font: run.font().clone(),
                        coords: run.normalized_coords().to_vec(),
                        size_px: run.font_size(),
                        embolden: synthetic_bold(synth.embolden(), st.weight, run.font_attrs().weight.value()),
                        skew_deg: synth.skew().unwrap_or(0.0),
                    });
                    let run_x0 = dx + gr.offset() + extra;
                    if !seen_runs.contains(&run.index()) {
                        seen_runs.push(run.index());
                        let mut cx = run_x0;
                        let base = slots.iter().position(|s| s.run == run.index()).unwrap_or(0);
                        for (ci, c) in run.visual_clusters().enumerate() {
                            let a = c.advance() * hs + kern_px.get(base + ci).copied().unwrap_or(0.0);
                            let r = c.text_range();
                            if r.end > prefix.len() || prefix.is_empty() {
                                out.clusters.push(ClusterInfo { range: map(r.start)..map(r.end), x: cx, advance: a, line: line_index, rtl: c.is_rtl() });
                            }
                            cx += a;
                        }
                    }
                    let rm = run.metrics();
                    let vm = if vertical { Some(VMetrics::new(run.font(), run.normalized_coords(), run.font_size())) } else { None };
                    let class = gr.style().brush.1;
                    let padding = vertical_padding
                        .binary_search_by_key(&run.text_range().start, |(range, _)| range.start)
                        .ok()
                        .and_then(|i| vertical_padding.get(i))
                        .map_or(0.0, |(_, padding)| *padding);
                    // Glyphs of the run's own characters (not substituted by `vert`).
                    let plain: Vec<u32> = match (&vm, class) {
                        (Some(v), 2) => ptext.get(run.text_range()).unwrap_or("").chars().filter_map(|c| v.cmap(c)).collect(),
                        _ => Vec::new(),
                    };
                    let mut pen = gr.offset();
                    let mut cursor = cursors.iter_mut().find(|c| c.0 == run.index());
                    for g in gr.glyphs() {
                        // Kerning of the clusters before this glyph's cluster.
                        if let Some(c) = cursor.as_deref_mut() {
                            if let Some(&slot) = c.1.get(c.2) {
                                while c.3 < slot {
                                    extra += kern_px.get(c.3).copied().unwrap_or(0.0);
                                    c.3 += 1;
                                }
                            }
                            c.2 += 1;
                        }
                        out.glyphs.push(PlacedGlyph {
                            face,
                            id: g.id,
                            x: dx + pen + g.x + extra,
                            y: baseline + g.y,
                            style: si,
                            orient: GlyphOrient::Horizontal,
                        });
                        if let Some(v) = &vm {
                            let upright = class == 1 || (class == 2 && !plain.contains(&g.id));
                            vinfo.push(VGlyph {
                                upright,
                                // Padding separates glyphs along the column, not across it.
                                advance: (g.advance - padding).max(0.0) * hs,
                                origin: if upright { v.origin(g.id) } else { 0.0 },
                                // Rotated text is centred on the column: its baseline sits
                                // (ascent − descent) / 2 from the centre line.
                                centre: (rm.ascent - rm.descent) * 0.5,
                            });
                        }
                        extra += g.advance * (hs - 1.0);
                        pen += g.advance;
                    }
                    // The run's last glyph: its remaining clusters' kerning follows it.
                    if let Some(c) = cursor
                        && c.2 >= c.1.len()
                    {
                        let end = slots.iter().rposition(|s| s.run == c.0).map_or(c.3, |p| p + 1);
                        while c.3 < end {
                            extra += kern_px.get(c.3).copied().unwrap_or(0.0);
                            c.3 += 1;
                        }
                    }
                    let run_x1 = dx + gr.offset() + gr.advance() + extra;
                    let shift = st.baseline_shift_pt * k;
                    if vertical {
                        // Underline to the right of the column, strikethrough through its centre
                        // (stored directly in text space).
                        let half = run.font_size() * 0.5;
                        let mut push = |v0: f32, v1: f32| out.decorations.push(DecorationRect { x0: -v1, y0: run_x0, x1: -v0, y1: run_x1, style: si });
                        if gr.style().underline.is_some() {
                            let v1 = baseline - half;
                            push(v1 - rm.underline_size.max(1.0), v1);
                        }
                        if gr.style().strikethrough.is_some() {
                            let sz = rm.strikethrough_size.max(1.0);
                            push(baseline - sz * 0.5, baseline + sz * 0.5);
                        }
                    } else if gr.style().underline.is_some() {
                        let y0 = baseline - rm.underline_offset - shift;
                        out.decorations.push(DecorationRect { x0: run_x0, y0, x1: run_x1, y1: y0 + rm.underline_size.max(1.0), style: si });
                    }
                    if !vertical && gr.style().strikethrough.is_some() {
                        let y0 = baseline - rm.strikethrough_offset - shift;
                        out.decorations.push(DecorationRect { x0: run_x0, y0, x1: run_x1, y1: y0 + rm.strikethrough_size.max(1.0), style: si });
                    }
                }
                if vertical {
                    extra -= squeeze_punctuation(text, &mut out.clusters[c0..], &mut out.glyphs[g0..]);
                }
                if justify_all {
                    let slack = avail.unwrap_or(0.0) - (adv + extra);
                    let n = out.clusters.len() - c0;
                    if n > 1 && slack > 0.0 {
                        let step = slack / (n - 1) as f32;
                        let starts: Vec<f32> = out.clusters[c0..].iter().map(|c| c.x).collect();
                        for (i, c) in out.clusters[c0..].iter_mut().enumerate() {
                            c.x += step * i as f32;
                        }
                        for g in &mut out.glyphs[g0..] {
                            let i = starts.iter().rposition(|&s| s <= g.x + 1e-3).unwrap_or(0);
                            g.x += step * i as f32;
                        }
                        extra += slack;
                    }
                }
                if vertical {
                    // Line space → text space (90° clockwise); upright glyphs centred on the column.
                    for (g, v) in out.glyphs[g0..].iter_mut().zip(&vinfo) {
                        let (u, cross) = (g.x, g.y);
                        if v.upright {
                            g.x = -baseline - v.advance * 0.5;
                            g.y = u + v.origin;
                            g.orient = GlyphOrient::Upright;
                        } else {
                            g.x = -(cross + v.centre);
                            g.y = u;
                            g.orient = GlyphOrient::Rotated;
                        }
                    }
                }
                let x0 = dx + m.offset;
                out.lines.push(LineInfo {
                    range: map(lr.start)..map(lr.end).min(content_end),
                    baseline,
                    x0,
                    x1: x0 + adv + extra - trailing_tabs,
                    ascent,
                    descent,
                    paragraph: pi,
                });
            }
            pending_space += ps.space_after_pt * k;
        }
        if is_box
            && t.block_align != photocraft_doc::text::BlockAlign::Start
            && let Some(bounds) = out.line_bounds()
        {
            let factor = if t.block_align == photocraft_doc::text::BlockAlign::Center { 0.5 } else { 1.0 };
            let (origin, extent) = if vertical { (-(box_rect.0 + box_rect.2), box_rect.2) } else { (box_rect.1, box_rect.3) };
            let offset = origin + (extent - (bounds[3] - bounds[1])).max(0.0) * factor - bounds[1];
            let (dx, dy) = out.to_text(0.0, offset);
            for glyph in &mut out.glyphs {
                glyph.x += dx;
                glyph.y += dy;
            }
            for decoration in &mut out.decorations {
                decoration.x0 += dx;
                decoration.x1 += dx;
                decoration.y0 += dy;
                decoration.y1 += dy;
            }
            for line in &mut out.lines {
                line.baseline += offset;
            }
            out.block_offset = offset;
        }
        out
    }
}

/// The first tab stop after `pos` (px from the anchor or box edge); stops are `interval` apart.
fn next_tab_stop(pos: f32, interval: f32) -> f32 {
    if interval > 0.0 && pos.is_finite() { ((pos + 1e-3) / interval).floor() * interval + interval } else { pos }
}

/// A tab as line breaking saw it: its shaped advance (before horizontal scale), the horizontal
/// scale, and the width that takes it to its tab stop on its line.
struct TabWidth {
    advance: f32,
    scale: f32,
    width: f32,
}

/// The width of each tab (offsets `tabs` in the paragraph text) on the broken lines of `layout`,
/// before kerning and alignment.
fn tab_widths(layout: &Layout<RunBrush>, tabs: &[usize], styles: &[CharStyle], indent: f32, first_indent: f32, interval: f32) -> Vec<TabWidth> {
    let mut out: Vec<TabWidth> = tabs.iter().map(|_| TabWidth { advance: interval, scale: 1.0, width: interval }).collect();
    for (li, line) in layout.lines().enumerate() {
        let mut pos = indent + if li == 0 { first_indent } else { 0.0 };
        let mut seen: Vec<usize> = Vec::new();
        for item in line.items() {
            let PositionedLayoutItem::GlyphRun(gr) = item else {
                continue;
            };
            let run = gr.run();
            if seen.contains(&run.index()) {
                continue;
            }
            seen.push(run.index());
            let scale = styles.get(gr.style().brush.0 as usize).map_or(1.0, |st| if st.horizontal_scale > 0.0 { st.horizontal_scale } else { 1.0 });
            for c in run.visual_clusters() {
                let advance = c.advance();
                match tabs.binary_search(&c.text_range().start).ok().and_then(|i| out.get_mut(i)) {
                    Some(t) => {
                        let stop = next_tab_stop(pos, interval);
                        *t = TabWidth { advance, scale, width: stop - pos };
                        pos = stop;
                    }
                    None => pos += advance * scale,
                }
            }
        }
    }
    out
}

/// Byte ranges of paragraphs (each including its `\r`, `\n` or `\r\n` terminator). Text ending
/// with a break yields a final empty paragraph, like Photoshop's trailing empty line.
pub fn split_paragraphs(text: &str) -> Vec<Range<usize>> {
    let b = text.as_bytes();
    let mut v = Vec::new();
    let mut start = 0;
    let mut i = 0;
    while i < b.len() {
        match b[i] {
            b'\r' => {
                let end = if b.get(i + 1) == Some(&b'\n') { i + 2 } else { i + 1 };
                v.push(start..end);
                start = end;
                i = end;
            }
            b'\n' => {
                v.push(start..i + 1);
                start = i + 1;
                i += 1;
            }
            _ => i += 1,
        }
    }
    v.push(start..b.len());
    v
}

fn strip_break(text: &str, r: &Range<usize>) -> usize {
    let s = &text[r.clone()];
    r.start + s.trim_end_matches(['\r', '\n']).len()
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace(['\\', '"'], ""))
}

/// Placement data of one vertical-type glyph (before mapping to text space).
struct VGlyph {
    upright: bool,
    /// Advance along the column (px).
    advance: f32,
    /// Upright: distance from the top of the glyph's cell down to its baseline (px).
    origin: f32,
    /// Rotated: distance from the column centre line to the glyph's baseline (px).
    centre: f32,
}

/// Vertical metrics of a font instance: `VORG`, `vhea`/`vmtx`, else the ideographic em box.
struct VMetrics<'a> {
    font: Option<skrifa::FontRef<'a>>,
    coords: Vec<skrifa::instance::NormalizedCoord>,
    /// px per font unit.
    scale: f32,
    /// Em-box fallback origin (px below the cell top).
    em_origin: f32,
}

impl<'a> VMetrics<'a> {
    fn new(fd: &'a FontData, coords: &[i16], size: f32) -> Self {
        use skrifa::MetadataProvider;
        let font = skrifa::FontRef::from_index(fd.data.as_ref(), fd.index).ok();
        let coords: Vec<skrifa::instance::NormalizedCoord> = coords.iter().map(|&c| skrifa::instance::NormalizedCoord::from_bits(c)).collect();
        let (scale, em_origin) = match &font {
            Some(f) => {
                let m = f.metrics(skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::new(&coords));
                let upem = f32::from(m.units_per_em).max(1.0);
                let (asc, desc) = (m.ascent, -m.descent);
                // Ideographic em box: the em centred on the font's ascent/descent span.
                let r = if asc + desc > 0.0 && asc.is_finite() && desc.is_finite() { (asc / (asc + desc)).clamp(0.0, 1.0) } else { 0.88 };
                (size / upem, size * r)
            }
            None => (0.0, size * 0.88),
        };
        Self { font, coords, scale, em_origin }
    }

    fn cmap(&self, c: char) -> Option<u32> {
        use skrifa::MetadataProvider;
        self.font.as_ref()?.charmap().map(c).map(|g| g.to_u32())
    }

    /// Distance from the top of an upright glyph's cell to its baseline (px).
    fn origin(&self, gid: u32) -> f32 {
        self.vertical_origin(gid).filter(|v| v.is_finite()).unwrap_or(self.em_origin)
    }

    fn vertical_origin(&self, gid: u32) -> Option<f32> {
        use skrifa::MetadataProvider;
        use skrifa::raw::TableProvider;
        let f = self.font.as_ref()?;
        let g = skrifa::GlyphId::new(gid);
        if let Ok(vorg) = f.vorg() {
            return Some(f32::from(vorg.vertical_origin_y(g)) * self.scale);
        }
        let tsb = f.vmtx().ok()?.side_bearing(g)?;
        // Top of the outline (font units), from the outline itself (glyf and CFF alike).
        let outline = f.outline_glyphs().get(g)?;
        let mut pen = YMax(None);
        let settings = skrifa::outline::DrawSettings::unhinted(skrifa::instance::Size::unscaled(), skrifa::instance::LocationRef::new(&self.coords));
        outline.draw(settings, &mut pen).ok()?;
        Some((pen.0? + f32::from(tsb)) * self.scale)
    }
}

/// Outline pen recording the largest y.
struct YMax(Option<f32>);

impl YMax {
    fn add(&mut self, y: f32) {
        self.0 = Some(self.0.map_or(y, |m| m.max(y)));
    }
}

impl skrifa::outline::OutlinePen for YMax {
    fn move_to(&mut self, _x: f32, y: f32) {
        self.add(y);
    }
    fn line_to(&mut self, _x: f32, y: f32) {
        self.add(y);
    }
    fn quad_to(&mut self, _cx0: f32, cy0: f32, _x: f32, y: f32) {
        self.add(cy0);
        self.add(y);
    }
    fn curve_to(&mut self, _cx0: f32, cy0: f32, _cx1: f32, cy1: f32, _x: f32, y: f32) {
        self.add(cy0);
        self.add(cy1);
        self.add(y);
    }
    fn close(&mut self) {}
}

/// OpenType feature settings of a character style (CSS `font-feature-settings` items).
fn feature_list(st: &CharStyle, small_caps: SmallCapsMode) -> Vec<String> {
    let mut feats: Vec<String> = Vec::new();
    // Optical and manual kerning replace the font's kerning table; a character with a manual
    // kern is manually kerned whatever its mode (as in Photoshop).
    if st.kerning != Kerning::Metrics || st.kern != 0.0 {
        feats.push("\"kern\" 0".into());
    }
    if !st.ligatures {
        feats.push("\"liga\" 0".into());
        feats.push("\"clig\" 0".into());
    }
    if st.discretionary_ligatures {
        feats.push("\"dlig\" 1".into());
    }
    if small_caps == SmallCapsMode::OpenType {
        feats.push("\"smcp\" 1".into());
    }
    for f in &st.features {
        if f.tag.len() == 4 && f.tag.is_ascii() {
            feats.push(format!("\"{}\" {}", f.tag, f.value));
        }
    }
    feats
}

/// The font variations for a run: the explicit axis values, plus `opsz` at the type size when the
/// style doesn't set it (as CSS `font-optical-sizing: auto`). A font without an `opsz` axis
/// ignores it.
fn variation_list(st: &CharStyle) -> Vec<String> {
    let mut vars: Vec<String> =
        st.variations.iter().filter(|v| v.axis.len() == 4 && v.axis.is_ascii() && v.value.is_finite()).map(|v| format!("\"{}\" {}", v.axis, v.value)).collect();
    if st.size_pt.is_finite() && !st.variations.iter().any(|v| v.axis == "opsz") {
        vars.push(format!("\"opsz\" {}", st.size_pt.max(0.01)));
    }
    vars
}

fn style_props(st: &CharStyle, k: f32, fallback: &[String], idx: u32, small_caps: SmallCapsMode) -> Vec<StyleProperty<'static, RunBrush>> {
    let px = (st.size_pt * k).max(0.01);
    let mut fam: Vec<String> = Vec::new();
    if !st.font_family.is_empty() {
        fam.push(quote(&st.font_family));
    }
    // Serif runs fall back to a Mincho face for Japanese (craft-fonts), others to a Gothic one.
    if crate::craft_fonts::is_serif_family(&st.font_family) {
        fam.extend(crate::craft_fonts::mincho_first(fallback).iter().map(|f| quote(f)));
    } else {
        fam.extend(fallback.iter().map(|f| quote(f)));
    }
    fam.push("sans-serif".into());
    let feats = feature_list(st, small_caps);
    let vars = variation_list(st);
    vec![
        StyleProperty::FontFamily(FontFamily::Source(Cow::Owned(fam.join(", ")))),
        StyleProperty::FontSize(px),
        StyleProperty::FontWeight(FontWeight::new(st.weight.clamp(1, 1000) as f32)),
        StyleProperty::FontStyle(if st.italic { FontStyle::Italic } else { FontStyle::Normal }),
        StyleProperty::FontFeatures(FontFeatures::Source(Cow::Owned(feats.join(", ")))),
        StyleProperty::FontVariations(FontVariations::Source(Cow::Owned(vars.join(", ")))),
        StyleProperty::LetterSpacing(st.tracking / 1000.0 * px),
        StyleProperty::Underline(st.underline),
        StyleProperty::Strikethrough(st.strikethrough),
        StyleProperty::Brush(RunBrush(idx, 0)),
        StyleProperty::Locale(st.language.as_deref().and_then(|l| parley::fontique::Language::parse(l).ok())),
    ]
}

/// Closing CJK punctuation: its full-width cell has blank space after the mark.
fn is_closing(c: char) -> bool {
    matches!(c, '、' | '。' | '，' | '．' | '」' | '』' | '）' | '〕' | '】' | '〉' | '》' | '〙' | '〗' | '］' | '｝')
}

/// Opening CJK brackets: blank space before the mark.
fn is_opening(c: char) -> bool {
    matches!(c, '「' | '『' | '（' | '〔' | '【' | '〈' | '《' | '〘' | '〖' | '［' | '｛')
}

/// Basic Japanese punctuation squeeze (JIS X 4051 style) for one line in line space: between a
/// closing mark and a following opening or closing mark, the closing mark's trailing half-em
/// blank is removed; between two opening brackets, the second one's leading half-em. Clusters and
/// glyphs from the squeezed point on move back. Returns the total length removed.
pub(crate) fn squeeze_punctuation(text: &str, clusters: &mut [ClusterInfo], glyphs: &mut [PlacedGlyph]) -> f32 {
    let first = |c: &ClusterInfo| text.get(c.range.clone()).and_then(|s| s.chars().next());
    let last = |c: &ClusterInfo| text.get(c.range.clone()).and_then(|s| s.chars().next_back());
    // Shift to apply from each cluster on.
    let mut shifts = vec![0.0f32; clusters.len()];
    let mut total = 0.0f32;
    for i in 1..clusters.len() {
        let (Some(prev), Some(cur)) = (clusters.get(i - 1), clusters.get(i)) else { continue };
        if prev.rtl || cur.rtl {
            continue;
        }
        let (Some(a), Some(b)) = (last(prev), first(cur)) else { continue };
        let cut = if is_closing(a) && (is_opening(b) || is_closing(b)) {
            prev.advance * 0.5
        } else if is_opening(a) && is_opening(b) {
            cur.advance * 0.5
        } else {
            0.0
        };
        if cut.is_finite() && cut > 0.0 {
            total += cut;
        }
        if let Some(s) = shifts.get_mut(i) {
            *s = total;
        }
    }
    if total <= 0.0 {
        return 0.0;
    }
    let starts: Vec<f32> = clusters.iter().map(|c| c.x).collect();
    for (c, s) in clusters.iter_mut().zip(&shifts) {
        c.x -= s;
    }
    for g in glyphs.iter_mut() {
        let i = starts.iter().rposition(|&s| s <= g.x + 1e-3).unwrap_or(0);
        g.x -= shifts.get(i).copied().unwrap_or(0.0);
    }
    total
}

/// Whether to embolden a face synthetically. Font matching (fontique) asks for it whenever the
/// requested weight is above the face's, so Regular (400) from a family with only W3 (300) and W6
/// (600), like Hiragino Mincho ProN, came out as a smeared bold W3. Like Photoshop and CSS, only
/// a bold request (≥ 600) on a face that isn't bold (≤ 500) is emboldened; anything else uses the
/// nearest face as it is (Faux Bold stays a separate, explicit style).
pub(crate) fn synthetic_bold(matcher_says: bool, requested: u16, face_weight: f32) -> bool {
    matcher_says && requested >= 600 && face_weight <= 500.0
}

/// Tallest run's ascender: lowercase 'd' horizontally, uppercase 'H' vertically.
fn first_ascent(line: &parley::Line<'_, RunBrush>, ascender: char) -> Option<f32> {
    use skrifa::MetadataProvider;
    let mut best: Option<f32> = None;
    for run in line.runs() {
        let fd = run.font();
        let Ok(font) = skrifa::FontRef::from_index(fd.data.as_ref(), fd.index) else {
            continue;
        };
        let coords: Vec<skrifa::instance::NormalizedCoord> = run.normalized_coords().iter().map(|&c| skrifa::instance::NormalizedCoord::from_bits(c)).collect();
        let loc = skrifa::instance::LocationRef::new(&coords);
        let size = skrifa::instance::Size::new(run.font_size());
        let h = font.charmap().map(ascender).and_then(|g| font.glyph_metrics(size, loc).bounds(g)).map(|b| b.y_max).or_else(|| {
            let m = font.metrics(size, loc);
            m.cap_height.or(Some(m.ascent * 0.75))
        });
        if let Some(h) = h {
            best = Some(best.map_or(h, |b: f32| b.max(h)));
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::{SmallCapsMode, feature_list, small_caps_mode, variation_list};
    use photocraft_doc::TextLayer;
    use photocraft_doc::text::{Caps, CharStyle, FontVariation, TextRun};

    /// Reference: the lookup the layout used before it became a binary search.
    fn run_index_reference(starts: &[usize], off: usize) -> usize {
        starts.iter().rposition(|&s| s <= off).unwrap_or(0)
    }

    #[test]
    fn run_lookup_matches_the_linear_reference() {
        // Deterministic pseudo-random run layouts with empty runs, equal adjacent styles and
        // offsets past the end.
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % m) as usize
        };
        for _ in 0..300 {
            let n = next(12);
            let mut acc = 0;
            let starts: Vec<usize> = (0..n)
                .map(|_| {
                    let s = acc;
                    acc += next(4); // 0 = empty run
                    s
                })
                .collect();
            for off in 0..acc + 4 {
                assert_eq!(super::run_index_at(&starts, off), run_index_reference(&starts, off), "{starts:?} @ {off}");
            }
        }
        assert_eq!(super::run_index_at(&[], 5), 0);
    }

    /// Many runs and paragraphs: the font is probed once per small-caps style per layout, not
    /// once per style per paragraph, and styles that don't use small caps are never probed.
    #[test]
    fn small_caps_probe_runs_once_per_small_caps_style() {
        let plain = CharStyle { font_family: "Inter".into(), ..Default::default() };
        let small = CharStyle { caps: Caps::SmallCaps, ..plain.clone() };
        let text: String = "a\n".repeat(300);
        let runs = |small_every: usize| -> Vec<TextRun> {
            (0..text.len()).map(|i| TextRun { len: 1, style: if small_every > 0 && i % small_every == 0 { small.clone() } else { plain.clone() } }).collect()
        };
        let mut engine = crate::TextEngine::new();
        let mut probes = |small_every: usize| {
            super::SMCP_PROBES.with(|c| c.set(0));
            let layer = TextLayer { text: text.clone(), runs: runs(small_every), ..Default::default() };
            let _ = engine.layout(&layer, 72.0);
            super::SMCP_PROBES.with(|c| c.get())
        };
        assert_eq!(probes(0), 0);
        let styles = text.len().div_ceil(2); // every other index is a small-caps run
        assert_eq!(probes(2), styles);
    }

    /// A variable font's optical size follows the type size unless the style sets `opsz` (#2985).
    #[test]
    fn optical_size_follows_the_type_size_unless_set() {
        let style = CharStyle { size_pt: 36.0, variations: vec![FontVariation { axis: "wght".into(), value: 650.0 }], ..Default::default() };
        assert_eq!(variation_list(&style), ["\"wght\" 650", "\"opsz\" 36"]);
        let set = CharStyle { size_pt: 36.0, variations: vec![FontVariation { axis: "opsz".into(), value: 12.0 }], ..Default::default() };
        assert_eq!(variation_list(&set), ["\"opsz\" 12"]);
    }

    #[test]
    fn uses_real_small_caps_only_when_the_selected_face_supports_smcp() {
        let style = CharStyle { caps: Caps::SmallCaps, ..Default::default() };
        let real = small_caps_mode(style.caps, true);
        let synthetic = small_caps_mode(style.caps, false);

        assert_eq!(real, SmallCapsMode::OpenType);
        assert!(feature_list(&style, real).iter().any(|feature| feature == "\"smcp\" 1"));
        assert_eq!(synthetic, SmallCapsMode::Synthetic);
        assert!(!feature_list(&style, synthetic).iter().any(|feature| feature == "\"smcp\" 1"));
    }
}
