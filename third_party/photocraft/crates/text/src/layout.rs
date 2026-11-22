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
//! the font's vertical metrics: `VORG`, `vhea`/`vmtx`, else the ideographic em box).
//! Font alternates and proportional metrics are resolved before wrapping. Lines, clusters,
//! carets and hit tests stay in line space; [`TextLayout::to_text`] / [`TextLayout::to_line`] convert.

use std::borrow::Cow;
use std::collections::HashMap;
use std::ops::Range;

use icu_properties::{CodePointMapData, props::{GeneralCategory, LineBreak}};
use parley::{
    Alignment, AlignmentOptions, FontData, FontFamily, FontFeatures, FontStyle, FontVariations, FontWeight, IndentOptions, Layout, LayoutContext,
    OverflowWrap, PositionedLayoutItem, StyleProperty,
};
use photocraft_doc::TextLayer;
use photocraft_doc::text::{Caps, CharStyle, Kerning, Orientation, TextAlign, TextDirection, TextShape};
use skrifa::raw::types::Tag;

use unicode_segmentation::UnicodeSegmentation;

use crate::fonts::{DEFAULT_FAMILY, FontDb};
use crate::mojikumi::{self, Cell as SpacingCell};
use crate::vertical::{VERTICAL_METRICS, VerticalForm, VerticalMetrics, sideways_baseline};

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

/// Character style index and the orientation of the characters shaped with it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RunBrush(u32, VerticalForm);

pub(crate) fn glyph_scale(scale: f32) -> f32 {
    if scale.is_finite() && scale > 0.0 { scale } else { 1.0 }
}

fn shaping_scale(style: &CharStyle, vertical: bool) -> f32 {
    glyph_scale(if vertical { style.vertical_scale } else { style.horizontal_scale })
}

fn cross_scale(style: &CharStyle, vertical: bool) -> f32 {
    glyph_scale(if vertical { style.horizontal_scale } else { style.vertical_scale })
}

// East Asian line composition keeps inseparable marks together and treats directional
// quotes as opening and closing punctuation next to ideographs. Other boundaries use ICU.
fn cjk_line_break(cx: parley::LineBreakContext) -> Option<bool> {
    use LineBreak as Break;

    let breaks = CodePointMapData::<Break>::new();
    let categories = CodePointMapData::<GeneralCategory>::new();
    match (breaks.get(cx.before), breaks.get(cx.after)) {
        (Break::Ideographic, Break::Inseparable) => Some(true),
        (Break::Ideographic, Break::Quotation) if categories.get(cx.after) == GeneralCategory::InitialPunctuation => Some(true),
        (Break::Quotation, Break::Ideographic) if categories.get(cx.before) == GeneralCategory::FinalPunctuation => Some(true),
        _ => None,
    }
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
    /// Layout-only width adjustment for upright comic punctuation sharing a cell.
    pub(crate) scale_x: f32,
}

/// Character decoration bounds (text space px).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecorationRect {
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
    pub style: u32,
    /// Filled ellipse for a character's emphasis dot; other decorations are rectangles.
    pub round: bool,
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
    /// Unscaled type metrics used when projecting the baseline spacing into editable PSD text.
    pub(crate) nominal_size: f32,
    pub(crate) nominal_leading: f32,
    /// The distance from the preceding baseline, before paragraph spacing.
    pub(crate) baseline_step: f32,
}

/// A grapheme cluster (caret stops, hit testing, selection), in line space.
#[derive(Clone, Debug, PartialEq)]
pub struct ClusterInfo {
    pub range: Range<usize>,
    pub x: f32,
    pub advance: f32,
    pub line: usize,
    pub rtl: bool,
    /// Character hit region within a side-by-side vertical cell, in text space.
    pub combined: Option<[f32; 4]>,
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
    pub(crate) japanese_styles: Vec<bool>,
    /// Comic groups in source coordinates, also used when writing editable PSD text.
    pub(crate) comic_groups: crate::comic::Groups,
    pub(crate) comic_scales: HashMap<usize, f32>,
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
        let (tx, ty) = self.to_text(x, y);
        if let Some((cluster, bounds)) = self.clusters.iter().filter(|c| c.line == li).filter_map(|c| {
            c.combined.filter(|b| ty >= b[1] && ty < b[3]).map(|b| (c, b))
        }).min_by(|(_, a), (_, b)| {
            let distance = |r: &[f32; 4]| (r[0] - tx).max(0.0).max(tx - r[2]);
            distance(a).total_cmp(&distance(b))
        }) {
            return if tx < (bounds[0] + bounds[2]) * 0.5 { cluster.range.start } else { cluster.range.end };
        }
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
        let next = self.clusters.iter().find(|c| c.range.start == offset);
        if let Some(b) = next.and_then(|c| c.combined) {
            return [(b[0], b[1]), (b[0], b[3])];
        }
        if next.is_none()
            && let Some(b) = self.clusters.iter().find(|c| c.range.end == offset && c.combined.is_some()).and_then(|c| c.combined)
        {
            return [(b[2], b[1]), (b[2], b[3])];
        }
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
    clusters: Option<Range<usize>>,
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

struct InlineCluster {
    range: Range<usize>,
    cell: SpacingCell,
    tracking: f32,
    base_advance: f32,
    metric_delta: f32,
}

struct BoundarySpacing {
    range: Range<usize>,
    gap: f32,
    reduction: f32,
    closing_margin: f32,
}

#[derive(Clone, Copy, Default)]
struct LineCompression {
    edge: f32,
    fraction: f32,
}

#[derive(Clone)]
struct CompositionFace {
    japanese: bool,
    font: Option<FontData>,
}

impl InlineCluster {
    fn append_spacing(self, next: Option<&Self>, advances: &mut Vec<(Range<usize>, f32)>, edges: &mut Vec<BoundarySpacing>) {
        let next = next.filter(|next| next.range.start == self.range.end);
        let gap = next.map_or(0.0, |next| next.cell.spacing_before(&self.cell).max(-self.base_advance));
        let closing_margin = self.cell.closing_margin();
        if gap.abs() > 0.001 || closing_margin > 0.001 {
            let reduction = next.map_or(0.0, |next| self.cell.spacing_reduction(&next.cell, gap));
            edges.push(BoundarySpacing { range: self.range.clone(), gap, reduction, closing_margin });
        }
        let delta = self.metric_delta + gap;
        if delta.abs() > 0.001 {
            advances.push((self.range, self.tracking + delta));
        }
    }
}

/// Measure Parley's legal break segments using the final glyph advances. At a possible
/// line end, discard the gap to the next line and allow a closing mark's margin to fit.
fn break_contextual_lines(layout: &mut Layout<RunBrush>, width: f32, first_indent: f32, edges: &[BoundarySpacing], unbroken: &[Range<usize>]) -> Vec<LineCompression> {
    struct Measure {
        end: usize,
        advance: f32,
        space: bool,
        legal: bool,
        ligature_continuation: bool,
        gap: f32,
        margin: f32,
        reduction: f32,
    }
    // OverflowWrap::Normal at zero width exposes the same legal breaks (including
    // the CJK override) that Parley would consider when filling the actual box.
    layout.break_all_lines(Some(0.0));
    let mut clusters = Vec::new();
    for line in layout.lines() {
        let line_end = line.text_range().end;
        for run in line.runs() {
            for c in run.clusters() {
                let end = c.text_range().end;
                let edge = edges.binary_search_by_key(&end, |edge| edge.range.end).ok().and_then(|i| edges.get(i));
                clusters.push(Measure {
                    end, advance: c.advance(), space: c.is_space_or_nbsp(), legal: end == line_end,
                    ligature_continuation: c.is_ligature_continuation(),
                    gap: edge.map_or(0.0, |edge| edge.gap),
                    margin: edge.map_or(0.0, |edge| edge.closing_margin),
                    reduction: edge.map_or(0.0, |edge| edge.reduction),
                });
            }
        }
    }
    clusters.sort_by_key(|c| c.end);
    let max_saving = edges.iter().map(|e| e.gap + e.closing_margin).fold(0.0, f32::max);
    let mut builder = layout.break_lines();
    builder.state_mut().set_layout_max_advance(width);
    builder.state_mut().set_line_max_advance(width);
    let mut compression = Vec::new();
    let mut start = 0;
    while start < clusters.len() {
        let available = width - if start == 0 { first_indent } else { 0.0 };
        let mut width = 0.0;
        let mut whitespace = 0.0;
        let mut reduction = 0.0;
        let mut regular = None;
        let mut emergency = None;
        let mut first = None;
        for (i, c) in clusters.iter().enumerate().skip(start) {
            width += c.advance;
            whitespace = if c.space { whitespace + c.advance } else { 0.0 };
            reduction += c.reduction;
            let next = i + 1;
            if clusters.get(next).is_some_and(|n| n.ligature_continuation)
                || crate::comic::range_at(unbroken, c.end).is_some_and(|i| unbroken[i].start < c.end)
            {
                continue;
            }
            first.get_or_insert(next);
            let excess = width - whitespace - c.gap - available;
            let interior = reduction - c.reduction;
            if excess <= c.margin + interior + 0.001 {
                let edge = excess.clamp(0.0, c.margin);
                let fraction = if interior > 0.0 { ((excess - edge) / interior).clamp(0.0, 1.0) } else { 0.0 };
                let fit = (next, LineCompression { edge, fraction });
                emergency = Some(fit);
                if c.legal {
                    regular = Some(fit);
                }
            } else if width - whitespace - reduction > available + max_saving {
                break;
            }
        }
        let (end, trim) = regular.or(emergency).unwrap_or((first.unwrap_or(start + 1), LineCompression::default()));
        let count = u32::try_from(end - start).unwrap_or(u32::MAX);
        if builder.break_next_with_length(count).is_none() {
            break;
        }
        compression.push(trim);
        start = end;
    }
    builder.finish();
    compression
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
        let comic_groups = crate::comic::Groups::for_text(text, vertical);
        let (box_rect, is_box) = match t.shape {
            TextShape::Box { x, y, width, height } => ((x, y, width, height), true),
            // Path text is laid out as one point-text line, then bent onto the path.
            TextShape::Point | TextShape::Path { .. } => ((0.0, 0.0, 0.0, 0.0), false),
        };
        let mut prev_baseline: Option<f32> = None;
        let mut prev_auto_column: Option<f32> = None;
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
        let east_asian_text = !text.is_ascii();
        let composition: Vec<CompositionFace> = if !vertical && !east_asian_text {
            Vec::new()
        } else {
            let mut by_face = HashMap::new();
            out.styles.iter().map(|st| {
                let mut face = by_face.entry((st.font_family.as_str(), st.weight, st.italic)).or_insert_with(|| {
                    let data = fonts.selected_face(&st.font_family, st.weight, st.italic);
                    CompositionFace {
                        japanese: mojikumi::uses_japanese_spacing(None, data.as_ref()),
                        font: data,
                    }
                }).clone();
                if st.language.is_some() {
                    face.japanese = mojikumi::uses_japanese_spacing(st.language.as_deref(), None);
                }
                face
            }).collect()
        };
        if east_asian_text { out.japanese_styles = composition.iter().map(|face| face.japanese).collect(); }
        let compose_japanese = out.japanese_styles.iter().any(|&enabled| enabled);
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
            let mut offsets = crate::comic::SourceOffsets::new(prefix.len());
            let mut chars = content.char_indices().map(|(i, ch)| (prange.start + i, ch)).peekable();
            while let Some((at, ch)) = chars.next() {
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
                let si = style_at(at);
                let style = &out.styles[si];
                let font = composition.get(si).and_then(|face| face.font.as_ref());
                let start = ptext.len();
                if let Some(end) = crate::comic::push_display_mark(&mut ptext, (at, ch), &mut chars, &comic_groups, font, vertical, |next| style_at(next) == si) {
                    if end - at != ch.len_utf8() || ptext.len() - start != ch.len_utf8() || crate::comic::compound(ch).is_some() {
                        offsets.record(at - prange.start..end - prange.start, start..ptext.len());
                    }
                    continue;
                }
                let caps = style.caps;
                if caps == Caps::AllCaps || small_caps[si] == SmallCapsMode::Synthetic {
                    let up: String = ch.to_uppercase().collect();
                    if up.len() == ch.len_utf8() {
                        ptext.push_str(&up);
                        if caps == Caps::SmallCaps && ch.is_lowercase() {
                            let end = ptext.len();
                            synthetic_small_caps.push((end - up.len()..end, style.size_pt * k * SYNTHETIC_SMALL_CAPS_SCALE * shaping_scale(style, vertical)));
                        }
                        continue;
                    }
                }
                ptext.push(ch);
            }
            let map = |off: usize| prange.start + offsets.source(off, false);
            let map_end = |off: usize| prange.start + offsets.source(off, true);
            let first_style = &out.styles[style_at(prange.start)];
            use crate::comic::GroupKind;
            let groups = comic_groups.within(prange.start..content_end);
            let display_range = |g: &crate::comic::Group| offsets.display(g.range.start - prange.start)..offsets.display(g.range.end - prange.start);
            let cells: Vec<_> = groups.iter().filter(|g| g.kind == GroupKind::Upright).map(display_range).collect();
            let linear: Vec<_> = groups.iter().filter(|g| g.kind == GroupKind::Linear).map(display_range).collect();
            let unbroken: Vec<_> = groups.iter().map(display_range)
                .filter(|r| ptext[r.clone()].chars().nth(1).is_some()).collect();
            let cell_steps: Vec<_> = cells.iter().map(|range| {
                let members = &ptext[range.clone()];
                let size = members.char_indices().map(|(at, _)| {
                    let style = &out.styles[style_at(map(range.start + at))];
                    style.size_pt * k * glyph_scale(style.vertical_scale)
                }).fold(0.01f32, f32::max);
                let count = members.chars().count().max(1) as f32;
                (size / count, count)
            }).collect();
            let mut forms: Vec<(Range<usize>, VerticalForm)> = Vec::new();
            if vertical {
                for (start, grapheme) in ptext.grapheme_indices(true) {
                    let Some(ch) = grapheme.chars().next() else { continue };
                    let end = start + grapheme.len();
                    let si = style_at(map(start));
                    let face = composition.get(si);
                    let font = face.and_then(|face| face.font.as_ref());
                    let paired = crate::comic::range_at(&cells, start).is_some_and(|i| cell_steps[i].1 > 1.0);
                    // Shape the members of a linear group on the same horizontal baseline
                    // before rotating the line; independent vertical forms can break its stroke.
                    let rotated = start >= prefix.len() && comic_groups.at(map(start)).is_some_and(|g| g.kind != GroupKind::Upright);
                    let form = if paired || rotated { VerticalForm::Sideways } else {
                        VerticalForm::of(ch, face.is_some_and(|face| face.japanese), font, out.styles[si].language.as_deref())
                    };
                    if let Some((range, _)) = forms.last_mut().filter(|(_, prior)| *prior == form) {
                        range.end = end;
                    } else {
                        forms.push((start..end, form));
                    }
                }
            }
            let first_px = first_style.size_pt * k;
            // Scripts the loaded fonts may lack (Arabic, Japanese, …): a host serving a font for them
            // fetches it (`served`); it joins the fallback stack once it arrives.
            crate::served::request_for_text(&ptext);
            let fallback: Vec<String> = fonts.fallback_stack().map(str::to_string).collect();
            // Shapes the paragraph, each tab widened by its letter spacing in `tab_spacing` (px).
            let build = |lcx: &mut LayoutContext<RunBrush>,
                         fcx: &mut parley::FontContext,
                         tab_spacing: &[f32],
                         advances: &[(Range<usize>, f32)],
                         contextual_wrap: bool|
             -> Layout<RunBrush> {
                let mut b = lcx.ranged_builder(fcx, &ptext, 1.0, false);
                if is_box {
                    b.set_line_break_override(Some(&cjk_line_break));
                    // Fall back to grapheme boundaries only when a whole segment is too wide.
                    if !contextual_wrap {
                        b.push_default(StyleProperty::OverflowWrap(OverflowWrap::BreakWord));
                    }
                }
                // Paragraph-start style as the default (covers the direction mark and empty
                // paragraphs), then every run piece intersecting this paragraph.
                let si0 = style_at(prange.start);
                for p in style_props(&out.styles[si0], k * shaping_scale(&out.styles[si0], vertical), &fallback, si0 as u32, small_caps[si0], vertical) {
                    b.push_default(p);
                }
                for (ri, st) in out.styles.iter().enumerate() {
                    let rs = run_starts[ri];
                    let (a, z) = (rs.max(prange.start), (rs + runs[ri].len).min(content_end));
                    if a >= z {
                        continue;
                    }
                    let range = offsets.display(a - prange.start)..offsets.display(z - prange.start);
                    // A style run covering only a hidden comic tag has no shaping range.
                    if range.is_empty() {
                        continue;
                    }
                    for p in style_props(st, k * shaping_scale(st, vertical), &fallback, ri as u32, small_caps[ri], vertical) {
                        b.push(p, range.clone());
                    }
                    let mut first = forms.partition_point(|(r, _)| r.end <= range.start);
                    while let Some((span, form)) = forms.get(first).filter(|(r, _)| r.start < range.end) {
                        if *form != VerticalForm::Sideways {
                            let r = (range.start.max(span.start))..(range.end.min(span.end));
                            let feats = feature_list(st, small_caps[ri], Some(*form));
                            b.push(StyleProperty::FontFeatures(FontFeatures::Source(Cow::Owned(feats.join(", ")))), r.clone());
                            b.push(StyleProperty::Brush(RunBrush(ri as u32, *form)), r);
                        }
                        first += 1;
                    }
                    let mut first = cells.partition_point(|cell| cell.end <= range.start);
                    while let Some(cell) = cells.get(first).filter(|cell| cell.start < range.end) {
                        if cell_steps[first].1 > 1.0 {
                            // Changing letter spacing on the second shaping pass can disable
                            // kerning and ligatures. Disable them in both passes for fixed cells.
                            let mut features = feature_list(st, small_caps[ri], Some(VerticalForm::Sideways));
                            features.extend(["\"kern\" 0", "\"liga\" 0", "\"clig\" 0", "\"dlig\" 0"].map(str::to_string));
                            let span = range.start.max(cell.start)..range.end.min(cell.end);
                            b.push(StyleProperty::FontFeatures(FontFeatures::Source(Cow::Owned(features.join(", ")))), span);
                        }
                        first += 1;
                    }
                }
                for (range, size) in &synthetic_small_caps {
                    b.push(StyleProperty::FontSize(*size), range.clone());
                }
                for range in &unbroken {
                    if let Some((last, _)) = ptext[range.clone()].char_indices().next_back() && last != 0 {
                        b.push(StyleProperty::TextWrapMode(parley::TextWrapMode::NoWrap), range.start..range.start + last);
                    }
                }
                for (range, spacing) in advances {
                    b.push(StyleProperty::LetterSpacing(*spacing), range.clone());
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
            let break_lines = |layout: &mut Layout<RunBrush>, edges: &[BoundarySpacing], contextual_wrap: bool| {
                if first_indent != 0.0 {
                    layout.set_text_indent(first_indent, IndentOptions::default());
                }
                if contextual_wrap && let Some(width) = avail {
                    return break_contextual_lines(layout, width, first_indent, edges, &unbroken);
                }
                layout.break_all_lines(avail);
                Vec::new()
            };
            // Paragraph text: a tab is never wider than a space plus one tab interval, so breaking
            // lines at that width keeps the expanded tabs inside the box, but can wrap early.
            // Break again with the tab widths that layout gives until they agree; if they don't
            // settle, keep the safe layout.
            let wide = if is_box { tab_px } else { 0.0 };
            let tab_spacing = vec![wide; tabs.len()];
            let mut layout = build(&mut self.lcx, &mut fonts.fcx, &tab_spacing, &[], false);
            // Convert upright clusters to vertical font metrics, then resolve Japanese boundary
            // spacing from the same advances the line breaker and the caret will use.
            let mut advances = Vec::new();
            let mut edges = Vec::new();
            let mut contextual_wrap = false;
            if compose_japanese || !cells.is_empty() || !unbroken.is_empty() || forms.iter().any(|(_, form)| *form != VerticalForm::Sideways) {
                layout.break_all_lines(None);
                for line in layout.lines() {
                    let mut prior: Option<InlineCluster> = None;
                    for run in line.runs() {
                        let mut metrics = None;
                        let mut metrics_style = None;
                        for cluster in run.visual_clusters() {
                            let RunBrush(si, form) = cluster.first_style().brush;
                            let range = cluster.text_range();
                            let cell = crate::comic::range_at(&cells, range.start).map(|i| cell_steps[i]);
                            if range.start < prefix.len() || (!compose_japanese && form == VerticalForm::Sideways && cell.is_none()) {
                                continue;
                            }
                            let st = &out.styles[si as usize];
                            if form == VerticalForm::Upright && metrics_style != Some(si) {
                                metrics = VerticalMetrics::new(run.font(), run.normalized_coords(), run.font_size(), st);
                                metrics_style = Some(si);
                            }
                            let tracking = st.tracking / 1000.0 * run.font_size();
                            let source = ptext.get(range.clone()).and_then(|s| s.chars().next());
                            let base_advance = if let Some((step, _)) = cell { Some(step) } else if form == VerticalForm::Upright {
                                metrics.as_ref().zip(cluster.glyphs().next()).map(|(vm, glyph)| vm.advance(vm.glyph(glyph.id, source)))
                            } else { None }.unwrap_or(cluster.advance() - tracking);
                            let cell_tracking = cell.map_or(tracking, |(_, count)| tracking / count);
                            let delta = base_advance + cell_tracking - cluster.advance();
                            if compose_japanese && !cluster.is_rtl() && let Some(ch) = source
                            {
                                let japanese = composition[si as usize].japanese;
                                let cell = if cell.is_some() { SpacingCell::ideographic(run.font_size(), japanese) }
                                    else { SpacingCell::new(ch, run.font_size(), base_advance, japanese) };
                                let next = InlineCluster { range, cell, tracking, base_advance, metric_delta: delta };
                                if let Some(previous) = prior.replace(next) {
                                    previous.append_spacing(prior.as_ref(), &mut advances, &mut edges);
                                }
                            } else if delta.abs() > 0.001 {
                                advances.push((range, tracking + delta));
                            }
                        }
                    }
                    if let Some(last) = prior {
                        last.append_spacing(None, &mut advances, &mut edges);
                    }
                }
                contextual_wrap = is_box && (!edges.is_empty() || !unbroken.is_empty()) && !ptext.contains('\n');
                edges.sort_by_key(|edge| edge.range.end);
                if !advances.is_empty() || contextual_wrap {
                    advances.sort_by_key(|(range, _)| range.start);
                    layout = build(&mut self.lcx, &mut fonts.fcx, &tab_spacing, &advances, contextual_wrap);
                }
            }
            let mut compression = break_lines(&mut layout, &edges, contextual_wrap);
            if is_box && !tabs.is_empty() {
                let first = tab_widths(&layout, &tabs, indent_start, first_indent, tab_px);
                let mut want: Vec<f32> = first.iter().map(|t| t.width).collect();
                for _ in 0..4 {
                    let spacing: Vec<f32> = first.iter().zip(&want).map(|(t, w)| w - (t.advance - tab_px)).collect();
                    let mut next = build(&mut self.lcx, &mut fonts.fcx, &spacing, &advances, contextual_wrap);
                    let adjusted = break_lines(&mut next, &edges, contextual_wrap);
                    let got: Vec<f32> = tab_widths(&next, &tabs, indent_start, first_indent, tab_px).iter().map(|t| t.width).collect();
                    if got.iter().zip(&want).all(|(a, b)| (a - b).abs() < 0.01) {
                        layout = next;
                        compression = adjusted;
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
                // Automatic spacing follows the drawn cross-axis size.
                let mut leading = 0.0f32;
                let mut emphasis_leading = 0.0f32;
                let mut col_half = 0.0f32;
                let (mut nominal_size, mut nominal_leading) = (0.0f32, 0.0f32);
                let (mut line_ascent, mut line_descent) = (0.0f32, 0.0f32);
                let mut auto_column = vertical;
                for item in line.items() {
                    if let PositionedLayoutItem::GlyphRun(gr) = item {
                        let st = &out.styles[gr.style().brush.0 as usize];
                        let run = gr.run();
                        let font_scale = shaping_scale(st, vertical);
                        let px = run.font_size() / font_scale;
                        let cell = px * cross_scale(st, vertical);
                        let nominal = st.size_pt * k;
                        nominal_size = nominal_size.max(nominal);
                        nominal_leading = nominal_leading.max(st.leading_pt.map_or(ps.auto_leading * nominal, |l| l * k));
                        col_half = col_half.max(cell * 0.5);
                        let cross = glyph_scale(st.vertical_scale) / font_scale;
                        line_ascent = line_ascent.max(run.metrics().ascent * cross);
                        line_descent = line_descent.max(run.metrics().descent * cross);
                        leading = leading.max(st.leading_pt.map_or(ps.auto_leading * cell, |l| l * k));
                        auto_column &= st.leading_pt.is_none();
                        if st.emphasis {
                            let space = if vertical { cell * 1.3 } else { st.size_pt * k * 0.25 };
                            emphasis_leading = emphasis_leading.max(space);
                        }
                    }
                }
                if leading == 0.0 {
                    let st = &out.styles[style_at(prange.start)];
                    leading = st.leading_pt.map_or(ps.auto_leading * st.size_pt * k, |l| l * k);
                    auto_column &= st.leading_pt.is_none();
                }
                if nominal_size == 0.0 {
                    let st = &out.styles[style_at(prange.start)];
                    nominal_size = st.size_pt * k;
                    nominal_leading = st.leading_pt.map_or(ps.auto_leading * nominal_size, |l| l * k);
                }
                // Preserve the requested spacing whenever it already has room for the dots.
                if !vertical && emphasis_leading > 0.0 {
                    emphasis_leading += line_ascent + line_descent;
                }
                leading = leading.max(emphasis_leading);
                let col_half = if col_half > 0.0 { col_half } else { first_px * 0.5 };
                let (ascent, descent) = if vertical {
                    (col_half, col_half)
                } else if line_ascent > 0.0 || line_descent > 0.0 {
                    (line_ascent, line_descent)
                } else {
                    (first_px * 0.8, first_px * 0.2)
                };
                let step = if auto_column {
                    prev_auto_column.map_or(leading, |previous| (previous + leading) * 0.5)
                } else {
                    leading
                };
                let baseline = match prev_baseline {
                    // Photoshop offsets the first vertical baseline by half the capital height.
                    None if is_box && vertical => -(box_rect.0 + box_rect.2) + first_ascent(&line, 'H', &out.styles, k).map(|h| h * 0.5).unwrap_or(col_half),
                    // Photoshop's "first baseline: ascent": the top of the tallest ascender
                    // (height of 'd') touches the box top, not the font's hhea ascent.
                    None if is_box => box_rect.1 + first_ascent(&line, 'd', &out.styles, k).unwrap_or(ascent),
                    None => 0.0,
                    Some(b) => b + step + pending_space,
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
                prev_auto_column = auto_column.then_some(leading);
                let last_line = li + 1 == nlines;
                let adv = m.advance - m.trailing_whitespace;

                // Kerning after each cluster (px), per run in visual cluster order: manual
                // kerning plus optical pair kerning. Applied in line space after shaping,
                // so it moves the following glyphs, carets and the line
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
                    let st = &out.styles[gr.style().brush.0 as usize];
                    for c in run.visual_clusters() {
                        let mut gl = c.glyphs();
                        let first = gl.next().map(|g| g.id);
                        let last = gl.last().map(|g| g.id).or(first);
                        slots.push(KernSlot {
                            run: run.index(),
                            start: c.text_range().start,
                            clusters: None,
                            first,
                            last,
                            font: run.font().clone(),
                            coords: run.normalized_coords().to_vec(),
                            size: run.font_size() / shaping_scale(st, vertical),
                            rtl: c.is_rtl(),
                            blank: first.is_none() || c.is_space_or_nbsp() || c.text_range().end <= prefix.len(),
                            upright: vertical && (c.first_style().brush.1 == VerticalForm::Upright
                                || crate::comic::range_at(&cells, c.text_range().start).is_some()),
                            advance: c.advance(),
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
                // A soft line break separates the neighbours, so their contextual space
                // does not contribute to the visible line extent or the end caret.
                let lr = line.text_range();
                let trim = compression.get(li).copied().unwrap_or_default();
                if trim.fraction > 0.0 {
                    for (slot, adjustment) in slots.iter().zip(&mut kern_px) {
                        if let Ok(i) = edges.binary_search_by_key(&slot.start, |edge| edge.range.start) {
                            let edge = &edges[i];
                            if edge.range.end < lr.end {
                                *adjustment -= edge.reduction * trim.fraction;
                            }
                        }
                    }
                }
                if let Ok(i) = edges.binary_search_by_key(&lr.end, |edge| edge.range.end)
                    && let Some(edge) = edges.get(i)
                    && let Some(j) = slots.iter().position(|slot| slot.start == edge.range.start)
                {
                    kern_px[j] -= edge.gap + trim.edge;
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
                let g0 = out.glyphs.len();
                let c0 = out.clusters.len();
                let mut cell_members = Vec::new();
                let mut inline_positions = Vec::new();
                let mut extra = 0.0f32;
                let mut seen_runs: Vec<usize> = Vec::new();
                for item in line.items() {
                    let PositionedLayoutItem::GlyphRun(gr) = item else {
                        continue;
                    };
                    let run = gr.run();
                    let si = gr.style().brush.0;
                    let st = &out.styles[si as usize];
                    let font_scale = shaping_scale(st, vertical);
                    let metric_scale = cross_scale(st, vertical) / font_scale;
                    let synth = run.synthesis();
                    let face = out.faces.len() as u32;
                    out.faces.push(GlyphFace {
                        font: run.font().clone(),
                        coords: run.normalized_coords().to_vec(),
                        size_px: run.font_size() / font_scale,
                        embolden: synthetic_bold(synth.embolden(), st.weight, run.font_attrs().weight.value()),
                        skew_deg: synth.skew().unwrap_or(0.0),
                    });
                    let run_x0 = dx + gr.offset() + extra;
                    if !seen_runs.contains(&run.index()) {
                        seen_runs.push(run.index());
                        let mut cx = run_x0;
                        let base = slots.iter().position(|s| s.run == run.index()).unwrap_or(0);
                        for (ci, c) in run.visual_clusters().enumerate() {
                            let a = c.advance() + kern_px.get(base + ci).copied().unwrap_or(0.0);
                            let r = c.text_range();
                            if r.end > prefix.len() || prefix.is_empty() {
                                let source = map(r.start)..map_end(r.end);
                                let index = if let Some(previous) = out.clusters.last_mut().filter(|last| last.line == line_index && last.range == source) {
                                    // The two glyphs of an encoded compound are one source character.
                                    previous.advance += a;
                                    out.clusters.len() - 1
                                } else {
                                    let index = out.clusters.len();
                                    let original = text.get(source.clone()).unwrap_or_default();
                                    let count = if ptext.get(r).is_some_and(|s| s.chars().nth(1).is_none()) {
                                        original.chars().count().max(1)
                                    } else { 1 };
                                    let advance = a / count as f32;
                                    let mut chars = original.char_indices();
                                    for i in 0..count {
                                        let range = if count == 1 { source.clone() } else {
                                            let Some((at, ch)) = chars.next() else { break };
                                            source.start + at..source.start + at + ch.len_utf8()
                                        };
                                        out.clusters.push(ClusterInfo { range, x: cx + i as f32 * advance, advance, line: line_index, rtl: c.is_rtl(), combined: None });
                                    }
                                    index
                                };
                                if let Some(slot) = slots.get_mut(base + ci) { slot.clusters = Some(index..out.clusters.len()); }
                            }
                            cx += a;
                        }
                    }
                    let rm = run.metrics();
                    let form = gr.style().brush.1;
                    let vm = (vertical && form != VerticalForm::Sideways)
                        .then(|| VerticalMetrics::new(run.font(), run.normalized_coords(), run.font_size(), st)).flatten();
                    let sideways = if vertical && vm.is_none() {
                        sideways_baseline(run.font(), run.font_size(), (rm.ascent - rm.descent) * 0.5)
                    } else { 0.0 };
                    let mut pen = gr.offset();
                    let mut cursor = cursors.iter_mut().find(|c| c.0 == run.index());
                    let mut missing_face = None;
                    for g in gr.glyphs() {
                        let slot = cursor.as_deref().and_then(|c| c.1.get(c.2)).and_then(|&i| slots.get(i));
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
                        let inline = dx + pen + extra;
                        let glyph = vm.as_ref().map_or(g.id, |metrics| {
                            let source = slot.and_then(|s| ptext.get(s.start..)).and_then(|s| s.chars().next());
                            metrics.glyph(g.id, source)
                        });
                        let (x, y, orient) = if !vertical {
                            (inline + g.x, baseline + g.y * metric_scale, GlyphOrient::Horizontal)
                        } else if let Some(metrics) = vm.as_ref() {
                            let (across, down) = metrics.origin(glyph);
                            (-baseline + (across + g.x) * metric_scale, inline + down + g.y, GlyphOrient::Upright)
                        } else {
                            (-baseline - (sideways + g.y) * metric_scale, inline + g.x, GlyphOrient::Rotated)
                        };
                        if justify_all {
                            inline_positions.push(inline + g.x);
                        }
                        let face = if g.id == 0 && fonts.missing_glyph_protection() && slot.is_some_and(|s| {
                            use icu_properties::{CodePointSetData, props::DefaultIgnorableCodePoint};
                            !s.blank && ptext[s.start..].chars().next().is_some_and(|c| {
                                !c.is_control() && !CodePointSetData::new::<DefaultIgnorableCodePoint>().contains(c)
                            })
                        }) {
                            *missing_face.get_or_insert_with(|| {
                                let Some(replacement) = visible_missing_face(fonts, &out.faces[face as usize]) else { return face };
                                let index = out.faces.len() as u32;
                                out.faces.push(replacement);
                                index
                            })
                        } else {
                            face
                        };
                        let mut placed = PlacedGlyph { face, id: glyph, x, y, style: si, orient, scale_x: 1.0 };
                        if let Some(slot) = slot {
                            if let Some(group) = crate::comic::range_at(&cells, slot.start) && let Some(clusters) = &slot.clusters {
                                cell_members.push(crate::comic::Member { group, glyph: out.glyphs.len(), clusters: clusters.clone() });
                            } else if orient == GlyphOrient::Rotated && crate::comic::range_at(&linear, slot.start).is_some()
                                && let Some([left, _, right, _]) = crate::render::local_glyph_bounds(&out, &placed, orient)
                            {
                                placed.x = -baseline - (left + right) * 0.5;
                            }
                        }
                        out.glyphs.push(placed);
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
                        let half = run.font_size() * metric_scale * 0.5;
                        let mut push = |v0: f32, v1: f32| out.decorations.push(DecorationRect { x0: -v1, y0: run_x0, x1: -v0, y1: run_x1, style: si, round: false });
                        if gr.style().underline.is_some() {
                            let v1 = baseline - half;
                            push(v1 - (rm.underline_size * metric_scale).max(1.0), v1);
                        }
                        if gr.style().strikethrough.is_some() {
                            let sz = (rm.strikethrough_size * metric_scale).max(1.0);
                            push(baseline - sz * 0.5, baseline + sz * 0.5);
                        }
                    } else if gr.style().underline.is_some() {
                        let y0 = baseline - rm.underline_offset * metric_scale - shift;
                        out.decorations.push(DecorationRect { x0: run_x0, y0, x1: run_x1, y1: y0 + (rm.underline_size * metric_scale).max(1.0), style: si, round: false });
                    }
                    if !vertical && gr.style().strikethrough.is_some() {
                        let y0 = baseline - rm.strikethrough_offset * metric_scale - shift;
                        out.decorations.push(DecorationRect { x0: run_x0, y0, x1: run_x1, y1: y0 + (rm.strikethrough_size * metric_scale).max(1.0), style: si, round: false });
                    }
                }
                if justify_all {
                    let slack = avail.unwrap_or(0.0) - (adv + extra);
                    if out.clusters.len() > c0 + 1 && slack > 0.0 {
                        let mut previous = None;
                        let mut gaps = 0;
                        let starts: Vec<_> = out.clusters[c0..].iter().enumerate().map(|(i, c)| {
                            let group = crate::comic::range_at(&unbroken, offsets.display(c.range.start.saturating_sub(prange.start)));
                            if i > 0 && (group.is_none() || group != previous) { gaps += 1; }
                            previous = group;
                            (c.x, gaps)
                        }).collect();
                        let step = if gaps > 0 { slack / gaps as f32 } else { 0.0 };
                        for (c, &(_, gap)) in out.clusters[c0..].iter_mut().zip(&starts) {
                            c.x += step * gap as f32;
                        }
                        for (g, inline) in out.glyphs[g0..].iter_mut().zip(inline_positions) {
                            let gap = starts.iter().rfind(|&&(x, _)| x <= inline + 1e-3).map_or(0, |&(_, gap)| gap);
                            if vertical { g.y += step * gap as f32; } else { g.x += step * gap as f32; }
                        }
                        extra += step * gaps as f32;
                    }
                }
                crate::comic::place_cells(&mut out, &cell_members, baseline);
                for cluster in &out.clusters[c0..] {
                    let si = style_at(cluster.range.start);
                    let st = &out.styles[si];
                    if !st.emphasis { continue; }
                    let count = text.get(cluster.range.clone()).map_or(0, |s| s.chars().filter(|c| c.is_alphanumeric()).count());
                    if count == 0 {
                        continue;
                    }
                    let size = st.size_pt * k;
                    let radius = size * 0.075;
                    let gap = size * 0.1;
                    let shift = st.baseline_shift_pt * k;
                    // A ligature may contain several letters but still needs one dot per letter.
                    for i in 0..count {
                        let middle = cluster.x + cluster.advance * (i as f32 + 0.5) / count as f32;
                        let (x, y) = if vertical {
                            (-baseline + size * st.horizontal_scale.max(0.01) * 0.5 + gap + radius + shift, middle)
                        } else {
                            (middle, baseline - ascent - gap - radius - shift)
                        };
                        out.decorations.push(DecorationRect { x0: x - radius, y0: y - radius, x1: x + radius, y1: y + radius, style: si as u32, round: true });
                    }
                }
                let x0 = dx + m.offset;
                out.lines.push(LineInfo {
                    range: map(lr.start)..map_end(lr.end).min(content_end),
                    baseline,
                    x0,
                    x1: x0 + adv + extra - trailing_tabs,
                    ascent,
                    descent,
                    paragraph: pi,
                    nominal_size,
                    nominal_leading,
                    baseline_step: step,
                });
            }
            pending_space += ps.space_after_pt * k;
        }
        if is_box
            && t.block_align != photocraft_doc::text::BlockAlign::Start
            && let Some(bounds) = out.line_bounds()
        {
            let (origin, extent) = if vertical { (-(box_rect.0 + box_rect.2), box_rect.2) } else { (box_rect.1, box_rect.3) };
            let offset = if t.block_align == photocraft_doc::text::BlockAlign::Center {
                // Move the whole block by its visible centre, even when its ink exceeds the frame.
                let center = crate::render::ink_bounds(&out, &photocraft_geom::Affine::IDENTITY, None).map_or(
                    (bounds[1] + bounds[3]) * 0.5,
                    |[x0, y0, x1, y1]| if vertical { (-(x0 + x1) * 0.5) as f32 } else { ((y0 + y1) * 0.5) as f32 },
                );
                origin + extent * 0.5 - center
            } else {
                origin + (extent - (bounds[3] - bounds[1])).max(0.0) - bounds[1]
            };
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
            for cluster in &mut out.clusters {
                if let Some(b) = &mut cluster.combined {
                    b[0] += dx;
                    b[1] += dy;
                    b[2] += dx;
                    b[3] += dy;
                }
            }
            for line in &mut out.lines {
                line.baseline += offset;
            }
            out.block_offset = offset;
        }
        out.comic_groups = comic_groups;
        out
    }
}

/// The first tab stop after `pos` (px from the anchor or box edge); stops are `interval` apart.
fn next_tab_stop(pos: f32, interval: f32) -> f32 {
    if interval > 0.0 && pos.is_finite() { ((pos + 1e-3) / interval).floor() * interval + interval } else { pos }
}

/// A tab's measured advance and the width that takes it to its stop on its line.
struct TabWidth {
    advance: f32,
    width: f32,
}

/// The width of each tab (offsets `tabs` in the paragraph text) on the broken lines of `layout`,
/// before kerning and alignment.
fn tab_widths(layout: &Layout<RunBrush>, tabs: &[usize], indent: f32, first_indent: f32, interval: f32) -> Vec<TabWidth> {
    let mut out: Vec<TabWidth> = tabs.iter().map(|_| TabWidth { advance: interval, width: interval }).collect();
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
            for c in run.visual_clusters() {
                let advance = c.advance();
                match tabs.binary_search(&c.text_range().start).ok().and_then(|i| out.get_mut(i)) {
                    Some(t) => {
                        let stop = next_tab_stop(pos, interval);
                        *t = TabWidth { advance, width: stop - pos };
                        pos = stop;
                    }
                    None => pos += advance,
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

/// Use a visible .notdef when the selected face's missing glyph has no outline.
fn visible_missing_face(fonts: &mut FontDb, original: &GlyphFace) -> Option<GlyphFace> {
    use skrifa::{FontRef, GlyphId, MetadataProvider, instance::{LocationRef, Size}};
    let has_ink = FontRef::from_index(original.font.data.as_ref(), original.font.index).ok()
        .and_then(|font| font.glyph_metrics(Size::unscaled(), LocationRef::default()).bounds(GlyphId::new(0)))
        .is_some_and(|bounds| bounds.x_min < bounds.x_max && bounds.y_min < bounds.y_max);
    if has_ink { return None; }
    let font = fonts.selected_face(DEFAULT_FAMILY, 400, false)?;
    Some(GlyphFace { font, coords: Vec::new(), ..original.clone() })
}

/// OpenType feature settings of a character style (CSS `font-feature-settings` items).
fn feature_list(st: &CharStyle, small_caps: SmallCapsMode, form: Option<VerticalForm>) -> Vec<String> {
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
    let upright = form == Some(VerticalForm::Upright);
    let explicit_width = st.features.iter().any(|f| f.tag == "palt"
        || (f.value != 0 && matches!(f.tag.as_str(), "halt" | "fwid" | "hwid" | "twid" | "qwid" | "pwid")));
    if !upright && !explicit_width {
        feats.push("\"palt\" 1".into());
    }
    for f in &st.features {
        if f.tag.len() == 4 && f.tag.is_ascii()
            && (form.is_none() || !matches!(f.tag.as_str(), "vert" | "vrt2"))
            && !(form.is_some() && VERTICAL_METRICS.contains(&f.tag.as_str()))
            && !(upright && matches!(f.tag.as_str(), "palt" | "halt"))
        {
            feats.push(format!("\"{}\" {}", f.tag, f.value));
        }
    }
    if let Some(form) = form {
        feats.push(if form == VerticalForm::Sideways { "\"vert\" 0" } else { "\"vert\" 1" }.into());
        feats.push("\"vrt2\" 0".into());
    }
    feats
}

/// The font variations for a run: the explicit axis values, plus `opsz` at the type size when the
/// style doesn't set it (as CSS `font-optical-sizing: auto`). A font without an `opsz` axis
/// ignores it.
pub(crate) fn variation_axes(st: &CharStyle) -> impl Iterator<Item = (&str, f32)> {
    let optical_size = (st.size_pt.is_finite() && !st.variations.iter().any(|v| v.axis == "opsz"))
        .then_some(("opsz", st.size_pt.max(0.01)));
    st.variations.iter().filter(|v| v.axis.len() == 4 && v.axis.is_ascii() && v.value.is_finite())
        .map(|v| (v.axis.as_str(), v.value)).chain(optical_size)
}

fn variation_list(st: &CharStyle) -> Vec<String> {
    variation_axes(st).map(|(axis, value)| format!("\"{axis}\" {value}")).collect()
}

fn style_props(st: &CharStyle, k: f32, fallback: &[String], idx: u32, small_caps: SmallCapsMode, vertical: bool) -> Vec<StyleProperty<'static, RunBrush>> {
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
    let feats = feature_list(st, small_caps, vertical.then_some(VerticalForm::Sideways));
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
        StyleProperty::Brush(RunBrush(idx, VerticalForm::Sideways)),
        StyleProperty::Locale(st.language.as_deref().and_then(|l| parley::fontique::Language::parse(l).ok())),
    ]
}

/// Whether to embolden a face synthetically. Font matching (fontique) asks for it whenever the
/// requested weight is above the face's, so Regular (400) from a family with only W3 (300) and W6
/// (600), like Hiragino Mincho ProN, came out as a smeared bold W3. Like Photoshop and CSS, only
/// a bold request (≥ 600) on a face that isn't bold (≤ 500) is emboldened; anything else uses the
/// nearest face as it is (Faux Bold stays a separate, explicit style).
pub(crate) fn synthetic_bold(matcher_says: bool, requested: u16, face_weight: f32) -> bool {
    matcher_says && requested >= 600 && face_weight <= 500.0
}

/// The tallest nominal ascender: lowercase 'd' horizontally, uppercase 'H' vertically.
/// Character proportions change the ink and advances, but not the frame's first baseline.
fn first_ascent(line: &parley::Line<'_, RunBrush>, ascender: char, styles: &[CharStyle], px_per_pt: f32) -> Option<f32> {
    use skrifa::MetadataProvider;
    let mut best: Option<f32> = None;
    for item in line.items() {
        let PositionedLayoutItem::GlyphRun(glyphs) = item else { continue };
        let run = glyphs.run();
        let fd = run.font();
        let Ok(font) = skrifa::FontRef::from_index(fd.data.as_ref(), fd.index) else {
            continue;
        };
        let coords: Vec<skrifa::instance::NormalizedCoord> = run.normalized_coords().iter().map(|&c| skrifa::instance::NormalizedCoord::from_bits(c)).collect();
        let loc = skrifa::instance::LocationRef::new(&coords);
        let size = styles.get(glyphs.style().brush.0 as usize).map_or(run.font_size(), |st| st.size_pt * px_per_pt);
        let size = skrifa::instance::Size::new(size);
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
        assert!(feature_list(&style, real, None).iter().any(|feature| feature == "\"smcp\" 1"));
        assert_eq!(synthetic, SmallCapsMode::Synthetic);
        assert!(!feature_list(&style, synthetic, None).iter().any(|feature| feature == "\"smcp\" 1"));
    }
}
