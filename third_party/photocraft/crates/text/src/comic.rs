//! Vertical comic composition. The display glyphs and cells never replace the source text.

use std::{collections::BTreeSet, iter::Peekable, ops::Range};

use icu_properties::{CodePointMapData, props::EastAsianWidth};
use parley::FontData;
use skrifa::{FontRef, MetadataProvider};

use crate::layout::{GlyphOrient, TextLayout, glyph_scale};
use crate::vertical::VerticalMetrics;

fn emphasis(ch: char) -> bool {
    matches!(ch, '!' | '?' | '！' | '？' | '¡' | '¿' | '؟' | '﹖' | '﹗' | '︕' | '︖' | 'ǃ' | '❕' | '❗' | '❢' | '❣' | '‽')
}

fn joins_cell(ch: char) -> bool {
    matches!(ch, '!' | '?' | '！' | '？')
}

pub(crate) fn compound(ch: char) -> Option<&'static str> {
    match ch {
        '‼' => Some("!!"),
        '⁉' => Some("!?"),
        '⁈' => Some("?!"),
        '⁇' => Some("??"),
        _ => None,
    }
}

pub(crate) fn is_wave(ch: char) -> bool {
    matches!(ch, '~' | '～' | '〜' | '〰' | '∼')
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LinearFamily {
    Dots,
    Dashes,
    Waves,
}

pub(crate) fn linear_family(ch: char) -> Option<LinearFamily> {
    match ch {
        '.' | '…' | '⋯' => Some(LinearFamily::Dots),
        '-' | '‐' | '‑' | '‒' | '–' | '—' | '―' | '─' | '−' | '－' => Some(LinearFamily::Dashes),
        _ if is_wave(ch) => Some(LinearFamily::Waves),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GroupKind {
    Upright,
    Linear,
    Sideways,
}

#[derive(Clone, Debug)]
pub(crate) struct Group {
    pub(crate) range: Range<usize>,
    pub(crate) source: usize,
    pub(crate) kind: GroupKind,
    pub(crate) cross: bool,
    fold: bool,
    narrow: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Groups {
    pub(crate) spans: Vec<Group>,
    tags: Vec<Range<usize>>,
}

fn short_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '０'..='９' | 'Ａ'..='Ｚ' | 'ａ'..='ｚ')
}

pub(crate) fn takes_cross_baseline(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|ch| chars.next().is_some()
        || !matches!(CodePointMapData::<EastAsianWidth>::new().get(ch), EastAsianWidth::Fullwidth | EastAsianWidth::Wide))
}

impl Groups {
    pub(crate) fn is_empty(&self) -> bool {
        self.spans.is_empty() && self.tags.is_empty()
    }

    pub(crate) fn for_text(text: &str, vertical: bool) -> Self {
        let mut groups = Self::default();
        if !vertical { return groups; }
        let mut plain = 0;
        for (at, _) in text.match_indices('<') {
            if at < plain { continue; }
            let Some(tag) = text.get(at..at + 3) else { continue; };
            let horizontal = tag.eq_ignore_ascii_case("<H>");
            if !horizontal && !tag.eq_ignore_ascii_case("<E>") { continue; }
            let rest = &text[at + 3..];
            let close = if horizontal { b"</H>" } else { b"</E>" };
            let end = rest.find(['\r', '\n', '\u{3}', '<']).unwrap_or(rest.len());
            if !rest.as_bytes().get(end..end + 4).is_some_and(|s| s.eq_ignore_ascii_case(close)) { continue; }
            groups.scan(&text[plain..at], plain);
            let body = at + 3..at + 3 + end;
            groups.tags.extend([at..body.start, body.end..body.end + 4]);
            plain = body.end + 4;
            if !body.is_empty() {
                let upright = horizontal && rest[..end].chars().count() <= 2;
                let kind = if upright { GroupKind::Upright } else if horizontal { GroupKind::Sideways } else { GroupKind::Linear };
                groups.push(body, kind, upright, false, horizontal);
            }
        }
        groups.scan(&text[plain..], plain);
        groups
    }

    pub(crate) fn at(&self, at: usize) -> Option<&Group> {
        let i = self.spans.partition_point(|group| group.range.end <= at);
        self.spans.get(i).filter(|group| group.range.contains(&at))
    }

    pub(crate) fn within(&self, range: Range<usize>) -> &[Group] {
        let start = self.spans.partition_point(|group| group.range.start < range.start);
        let end = self.spans.partition_point(|group| group.range.end <= range.end);
        self.spans.get(start..end).unwrap_or_default()
    }

    pub(crate) fn insert_breaks(&mut self, breaks: &BTreeSet<usize>) {
        for range in self.spans.iter_mut().map(|group| &mut group.range).chain(self.tags.iter_mut()) {
            range.start += breaks.range(..=range.start).count();
            range.end += breaks.range(..range.end).count();
        }
    }

    fn push(&mut self, range: Range<usize>, kind: GroupKind, cross: bool, fold: bool, narrow: bool) {
        self.spans.push(Group { source: range.start, range, kind, cross, fold, narrow });
    }

    fn scan(&mut self, text: &str, base: usize) {
        let mut chars = text.char_indices().peekable();
        while let Some((at, ch)) = chars.next() {
            let mut end = at + ch.len_utf8();
            if joins_cell(ch) {
                while let Some((next, c)) = chars.next_if(|(_, c)| joins_cell(*c)) { end = next + c.len_utf8(); }
                self.emphasis_run(&text[at..end], base + at);
            } else if emphasis(ch) || compound(ch).is_some() {
                let cross = compound(ch).is_some() || takes_cross_baseline(&text[at..end]);
                self.push(base + at..base + end, GroupKind::Upright, cross, false, false);
            } else if short_word_char(ch) {
                let mut count = 1usize;
                while let Some((next, c)) = chars.next_if(|(_, c)| short_word_char(*c)) { end = next + c.len_utf8(); }
                count += text[at + ch.len_utf8()..end].chars().count();
                // The legacy recognizer treated a Western phrase as one sideways
                // item. Its short words must not turn into separate upright cells.
                if let Some(rest) = text.get(end..) {
                    let after_space = rest.trim_start_matches([' ', '\t']);
                    if after_space.len() < rest.len() && after_space.starts_with(short_word_char) {
                        let mut phrase = after_space;
                        loop {
                            let rest = phrase.trim_start_matches(short_word_char);
                            let after_space = rest.trim_start_matches([' ', '\t']);
                            if after_space.len() == rest.len() || !after_space.starts_with(short_word_char) {
                                let last = text.len() - rest.len();
                                while chars.next_if(|&(i, _)| i < last).is_some() {}
                                break;
                            }
                            phrase = after_space;
                        }
                        continue;
                    }
                }
                if count == 2 {
                    self.push(base + at..base + end, GroupKind::Upright, true, false, false);
                } else if ch == '-' && text[at..end].chars().all(|c| c == '-') {
                    while let Some((next, c)) = chars.next_if(|(_, c)| linear_family(*c) == Some(LinearFamily::Dashes)) { end = next + c.len_utf8(); }
                    if end > at + ch.len_utf8() {
                        self.push(base + at..base + end, GroupKind::Linear, false, false, false);
                    }
                }
            } else if let Some(family) = linear_family(ch) {
                while let Some((next, c)) = chars.next_if(|(_, c)| linear_family(*c) == Some(family)) {
                    end = next + c.len_utf8();
                }
                if end > at + ch.len_utf8() {
                    self.push(base + at..base + end, GroupKind::Linear, false, false, false);
                }
            }
        }
    }

    /// Legacy replacements give homogeneous triples and pairs priority over mixed pairs.
    /// A remaining mixed-width run is the old <H> fallback, with up to four characters.
    fn emphasis_run(&mut self, text: &str, base: usize) {
        let chars: Vec<_> = text.char_indices().collect();
        let mut cells = vec![0u8; chars.len()];
        for pattern in ["!!!", "!!", "！！！", "！！", "???", "??", "？？？", "？？", "!?", "?!", "！？", "？！"] {
            let len = pattern.chars().count();
            let mut i = 0;
            while let Some(window) = chars.get(i..i + len) {
                if cells[i..i + len].iter().all(|&v| v == 0) && window.iter().map(|&(_, c)| c).eq(pattern.chars()) {
                    cells[i..i + len].fill(1);
                    cells[i] = len as u8;
                    i += len;
                } else {
                    i += 1;
                }
            }
        }
        let mut i = 0;
        while i < chars.len() {
            let assigned = cells[i];
            let count = if assigned != 0 { usize::from(assigned) } else {
                cells[i..].iter().take(4).take_while(|&&v| v == 0).count()
            };
            let start = chars[i].0;
            let end = chars.get(i + count).map_or(text.len(), |&(at, _)| at);
            let kind = if assigned == 0 && count > 2 { GroupKind::Sideways } else { GroupKind::Upright };
            let cross = kind == GroupKind::Upright && (count > 1 || chars[i].1.is_ascii());
            self.push(base + start..base + end, kind, cross, assigned == 3, count > 1);
            i += count;
        }
    }
}

pub(crate) fn equivalent_marks(ch: char, vertical: bool) -> &'static str {
    match ch {
        '❕' | '❗' => "！!",
        _ if !vertical => "",
        '!' | '！' | '﹗' => "！!",
        '?' | '？' | '﹖' => "?？",
        '.' => "．", '．' => ".", ',' => "，", '，' => ",",
        ':' => "：", '：' => ":", ';' => "；", '；' => ";",
        '·' => "・", '・' => "·", '⋯' => "…", '…' => "⋯",
        '—' | '―' | '─' | '−' | '－' => "—–-",
        '～' | '〜' | '〰' => "～〜~",
        '(' => "（", '（' => "(", ')' => "）", '）' => ")",
        '[' => "［", '［' => "[", ']' => "］", '］' => "]",
        '{' => "｛", '｛' => "{", '}' => "｝", '｝' => "}",
        _ => "",
    }
}

/// Prefer an equivalent glyph in the requested face when the input mark is missing.
pub(crate) fn local_mark(face: Option<&FontData>, ch: char, vertical: bool) -> char {
    let equivalents = equivalent_marks(ch, vertical);
    if equivalents.is_empty() { return ch; }
    let Some(font) = face.and_then(|f| FontRef::from_index(f.data.as_ref(), f.index).ok()) else { return ch };
    let map = font.charmap();
    if map.map(ch).is_some() { ch } else { equivalents.chars().find(|&other| map.map(other).is_some()).unwrap_or(ch) }
}

/// Use the chosen face's ellipsis for two or three consecutive dots in the same style.
fn compact_dots(
    chars: &mut Peekable<impl Iterator<Item = (usize, char)>>, face: Option<&FontData>, same_style: impl Fn(usize) -> bool,
) -> Option<(char, usize)> {
    let is_dot = |&(at, ch): &(usize, char)| ch == '.' && same_style(at);
    if !chars.peek().is_some_and(is_dot) { return None; }
    let face = face?;
    let font = FontRef::from_index(face.data.as_ref(), face.index).ok()?;
    let mark = ['…', '⋯'].into_iter().find(|&ch| font.charmap().map(ch).is_some())?;
    let (second, _) = chars.next()?;
    let last = chars.next_if(is_dot).map_or(second, |(at, _)| at);
    Some((mark, last + 1))
}

/// Append the display spelling of a comic mark and return the consumed source end.
/// The editor and the editable PSD use this spelling with their own source offsets.
pub(crate) fn push_display_mark(
    output: &mut String, (at, ch): (usize, char), chars: &mut Peekable<impl Iterator<Item = (usize, char)>>, groups: &Groups,
    face: Option<&FontData>, vertical: bool, same_style: impl Fn(usize) -> bool,
) -> Option<usize> {
    if vertical {
        if let Some(tag) = range_at(&groups.tags, at).and_then(|i| groups.tags.get(i)) {
            while chars.next_if(|&(i, _)| i < tag.end).is_some() {}
            return Some(tag.end);
        }
        if let Some(group) = groups.at(at).filter(|g| g.narrow && joins_cell(ch)) {
            let plain = if matches!(ch, '!' | '！') { '!' } else { '?' };
            let end = if group.fold && at == group.range.start {
                chars.next().map_or(at + ch.len_utf8(), |(next, mark)| next + mark.len_utf8())
            } else { at + ch.len_utf8() };
            let mark = local_mark(face, plain, true);
            if mark != ch || end != at + ch.len_utf8() {
                output.push(mark);
                return Some(end);
            }
        }
        if let Some(marks) = compound(ch) {
            output.extend(marks.chars().map(|mark| local_mark(face, mark, true)));
            return Some(at + ch.len_utf8());
        }
        if ch == '.' && let Some((mark, end)) = compact_dots(chars, face, same_style) {
            output.push(mark);
            return Some(end);
        }
    }
    let mark = local_mark(face, ch, vertical);
    if mark == ch { return None; }
    output.push(mark);
    Some(at + ch.len_utf8())
}

/// Offsets between the source and the temporary shaped paragraph. Shaping may use
/// a single ellipsis for several source characters or several glyphs for one mark.
pub(crate) struct SourceOffsets {
    prefix: usize,
    changes: Vec<(Range<usize>, Range<usize>)>,
}

impl SourceOffsets {
    pub(crate) fn new(prefix: usize) -> Self { Self { prefix, changes: Vec::new() } }

    pub(crate) fn record(&mut self, source: Range<usize>, display: Range<usize>) {
        self.changes.push((source, display));
    }

    pub(crate) fn display(&self, at: usize) -> usize {
        self.to_display(at, false)
    }

    /// An exclusive end stays before an inserted separator assigned to the next group.
    pub(crate) fn display_end(&self, at: usize) -> usize {
        self.to_display(at, true)
    }

    fn to_display(&self, at: usize, end: bool) -> usize {
        let i = self.changes.partition_point(|(source, _)| source.end < at || (source.end == at && (!end || !source.is_empty())));
        if let Some((_, display)) = self.changes.get(i).filter(|(source, _)| source.contains(&at)) {
            display.start
        } else {
            self.changes[..i].last().map_or(at + self.prefix, |(source, display)| display.end + at - source.end)
        }
    }

    pub(crate) fn source(&self, at: usize, end: bool) -> usize {
        let i = self.changes.partition_point(|(_, display)| display.end < at || (display.end == at && (!end || !display.is_empty())));
        if let Some((source, display)) = self.changes.get(i).filter(|(_, display)| display.contains(&at)
            || (end && display.is_empty() && display.start == at))
        {
            if end && at > display.start { source.end } else { source.start }
        } else {
            self.changes[..i].last().map_or(at.saturating_sub(self.prefix), |(source, display)| source.end + at - display.end)
        }
    }
}

pub(crate) fn range_at(ranges: &[Range<usize>], at: usize) -> Option<usize> {
    let i = ranges.partition_point(|r| r.end <= at);
    ranges.get(i).filter(|r| r.contains(&at)).map(|_| i)
}

pub(crate) struct Member {
    pub(crate) group: usize,
    pub(crate) glyph: usize,
    pub(crate) clusters: Range<usize>,
}

/// Place the horizontal ink within the same cell that the line breaker measured.
pub(crate) fn place_cells(layout: &mut TextLayout, members: &[Member], baseline: f32) {
    for group in members.chunk_by(|a, b| a.group == b.group) {
        let mut top = f32::INFINITY;
        let mut bottom = f32::NEG_INFINITY;
        let mut size = 0.0f32;
        let mut origin = 0.0f32;
        let mut pen = 0.0f32;
        let mut left = f32::INFINITY;
        let mut right = f32::NEG_INFINITY;
        let mut source = usize::MAX;
        let mut ink = Vec::with_capacity(group.len());
        for member in group {
            let clusters = &layout.clusters[member.clusters.clone()];
            let glyph = &layout.glyphs[member.glyph];
            let face = &layout.faces[glyph.face as usize];
            let style = &layout.styles[glyph.style as usize];
            let em = face.size_px * glyph_scale(style.horizontal_scale);
            let vertical_em = face.size_px * glyph_scale(style.vertical_scale);
            let [l, _, r, _] = crate::render::local_glyph_bounds(layout, glyph, GlyphOrient::Upright)
                .unwrap_or([0.0, 0.0, em * 0.5, 0.0]);
            let metrics = VerticalMetrics::new(&face.font, &face.coords, vertical_em, style);
            let down = metrics.as_ref().map_or(vertical_em * 0.88, |m| m.origin(glyph.id).1);
            let advance = metrics.and_then(|m| m.horizontal_advance(glyph.id))
                .map_or((r - l).max(0.0), |a| a * em / vertical_em);
            for c in clusters {
                top = top.min(c.x);
                bottom = bottom.max(c.x + c.advance);
                source = source.min(c.range.start);
            }
            size = size.max(em);
            origin = origin.max(down);
            left = left.min(pen + l);
            right = right.max(pen + r);
            ink.push((pen, l, r));
            let spacing = style.tracking + style.kern;
            pen += advance + if spacing.is_finite() { spacing * em / 1000.0 } else { 0.0 };
        }
        let scale = (size / (right - left).max(0.01)).min(1.0);
        let offset = -baseline - (left + right) * scale * 0.5;
        if scale < 1.0 {
            layout.comic_scales.insert(source, scale);
        }
        for (member, (pen, l, r)) in group.iter().zip(ink) {
            let glyph = &mut layout.glyphs[member.glyph];
            glyph.x = offset + pen * scale;
            glyph.y = top + origin;
            glyph.orient = GlyphOrient::Upright;
            glyph.scale_x = scale;
            let left = glyph.x + l * scale;
            let clusters = &mut layout.clusters[member.clusters.clone()];
            let step = ((r - l) * scale).max(0.01) / clusters.len() as f32;
            for (i, c) in clusters.iter_mut().enumerate() {
                let bounds = [left + i as f32 * step, top, left + (i + 1) as f32 * step, bottom];
                c.combined = Some(c.combined.map_or(bounds, |old| [old[0].min(bounds[0]), old[1], old[2].max(bounds[2]), old[3]]));
            }
        }
    }
}
