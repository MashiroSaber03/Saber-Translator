//! Project the shared layout into editable Photoshop text on the export copy.

use std::{borrow::Cow, collections::BTreeSet};

use parley::FontData;
use photocraft_doc::{TextLayer, text::{CharStyle, ParagraphRun, TextRun, TextShape}};
use photocraft_geom::Affine;
use skrifa::{FontRef, MetadataProvider, instance::Size};

use crate::{
    FontDb, TextLayout,
    comic::{self, GroupKind, Groups, LinearFamily, SourceOffsets},
    layout::{glyph_scale, split_paragraphs, variation_axes},
    vertical::{VerticalForm, has_alternate, sideways_baseline},
};

struct Piece<'a> {
    len: usize,
    style: Cow<'a, CharStyle>,
    japanese: bool,
    vertical_advance: bool,
}

fn append(piece: Piece<'_>, runs: &mut Vec<TextRun>, japanese: &mut Vec<bool>) {
    if let Some(run) = runs.last_mut().filter(|r| r.style == *piece.style && japanese.last() == Some(&piece.japanese)) {
        run.len += piece.len;
    } else {
        runs.push(TextRun { len: piece.len, style: piece.style.into_owned() });
        japanese.push(piece.japanese);
    }
}

struct SidewaysSpelling {
    mark: char,
    along: f32,
    across: f32,
    advance: f32,
    leading: f32,
    baseline_shift_em: f32,
}

/// Project a centred comic stroke onto Photoshop's rotated Roman baseline. Prefer a cmap alias
/// of the same glyph. For different outlines, also preserve size, advance and leading bearing.
fn sideways_spelling(ch: char, face: Option<&FontData>, style: &CharStyle, japanese: bool) -> Option<SidewaysSpelling> {
    let family = comic::linear_family(ch)?;
    // These aliases use the Roman stream in Photoshop. A face may share their outlines
    // with the CJK entry even if the original code point would be upright on reflow.
    let candidates: &[char] = match family {
        LinearFamily::Waves => &['∼', '~'],
        LinearFamily::Dashes => &['–', '‒', '-'],
        LinearFamily::Dots => return None,
    };
    let original = comic::local_mark(face, ch, true);
    let language = style.language.as_deref();
    let face = face?;
    let font = FontRef::from_index(face.data.as_ref(), face.index).ok()?;
    let map = font.charmap();
    let id = map.map(original)?;
    let sideways = |mark| VerticalForm::of(mark, japanese, Some(face), language) == VerticalForm::Sideways;
    let alternate = has_alternate(&font, id, original, language, false);
    let already_sideways = sideways(original) && !alternate;
    if !already_sideways && (candidates.contains(&original) || (family == LinearFamily::Dashes && alternate)) {
        return None;
    }
    let aliases = candidates.iter().copied().filter(|&mark| family == LinearFamily::Waves || sideways(mark));
    let location = font.axes().location(variation_axes(style));
    let metrics = font.glyph_metrics(Size::new(1.0), location.coords());
    let a = metrics.bounds(id)?;
    let line = font.metrics(Size::new(1.0), location.coords());
    let baseline = sideways_baseline(face, 1.0, (line.ascent + line.descent) * 0.5);
    let projected = |mark| {
        let glyph = map.map(mark)?;
        let b = metrics.bounds(glyph)?;
        let baseline_shift_em = baseline - (b.y_min + b.y_max) * 0.5;
        if glyph == id {
            return Some(SidewaysSpelling { mark, along: 1.0, across: 1.0, advance: 0.0, leading: 0.0, baseline_shift_em });
        }
        let along = (a.x_max - a.x_min) / (b.x_max - b.x_min);
        let across = (a.y_max - a.y_min) / (b.y_max - b.y_min);
        if !(0.25..=4.0).contains(&along) || !(0.25..=4.0).contains(&across) { return None; }
        let scale = glyph_scale(style.vertical_scale);
        let advance = (metrics.advance_width(id)? - along * metrics.advance_width(glyph)?) * scale;
        let leading = (a.x_min - along * b.x_min) * scale;
        Some(SidewaysSpelling { mark, along, across, advance, leading, baseline_shift_em })
    };
    if already_sideways {
        projected(original)
    } else if let Some(alias) = aliases.clone().find(|&mark| map.map(mark) == Some(id)) {
        projected(alias)
    } else if family == LinearFamily::Waves {
        aliases.rev().find_map(projected)
    } else {
        // A narrow outline can still have a wide advance. Prefer the spelling that
        // fits with the smallest adjustment to spacing and its leading bearing.
        aliases.filter_map(projected).min_by(|a, b| {
            (a.advance.abs() + a.leading.abs()).total_cmp(&(b.advance.abs() + b.leading.abs()))
        })
    }
}

#[derive(Clone, Copy, Default, PartialEq)]
struct LineStyle {
    leading_pt: Option<f32>,
    baseline_shift_pt: f32,
}

fn line_style_changes(layout: &TextLayout, breaks: &BTreeSet<usize>) -> Vec<(usize, LineStyle)> {
    let mut changes = Vec::new();
    let mut breaks = breaks.iter().peekable();
    let mut inserted = 0;
    let mut previous = LineStyle::default();
    let mut baseline_shift_pt = 0.0;
    for (i, line) in layout.lines.iter().enumerate() {
        while breaks.next_if(|&&at| at <= line.range.start).is_some() {
            inserted += 1;
        }
        let desired = if layout.vertical {
            // Photoshop measures vertical leading from the preceding column's right edge.
            layout.lines.get(i + 1).map(|next| next.baseline_step - (next.nominal_size - line.nominal_size) * 0.5)
        } else {
            // Horizontal leading belongs to the destination line and follows its baseline.
            (i != 0).then_some(line.baseline_step)
        };
        // Photoshop also floors very small vertical leading based on the nominal column size.
        // Use an explicit value above that floor and carry the remainder as a baseline shift.
        let minimum = if layout.vertical { (line.nominal_size * 0.2).max(0.01) } else { 0.01 };
        let exact = desired.filter(|step| step.is_finite()
            && (*step < minimum || (step - line.nominal_leading).abs() > 0.01));
        let leading_px = exact.map(|value| value.max(minimum));
        let leading_pt = leading_px.map(|value| value / layout.px_per_pt);
        let remainder = leading_px.zip(exact).map_or(0.0, |(applied, requested)| (applied - requested) / layout.px_per_pt);
        if !layout.vertical { baseline_shift_pt += remainder; }
        let style = LineStyle { leading_pt, baseline_shift_pt };
        if style != previous {
            changes.push((line.range.start + inserted, style));
            previous = style;
        }
        if layout.vertical { baseline_shift_pt += remainder; }
    }
    changes
}

pub(super) fn apply(layer: &mut TextLayer, layout: &TextLayout, fonts: &mut FontDb, groups: &mut Groups, breaks: &BTreeSet<usize>) -> Option<Vec<bool>> {
    let source = &layer.text;
    let vertical = layout.vertical;
    let line_styles = line_style_changes(layout, breaks);
    let reserve = line_styles.last().map_or(0.0, |(_, style)| style.baseline_shift_pt * layout.px_per_pt);
    let mut line_styles = line_styles.into_iter().peekable();
    let mut line_style = LineStyle::default();
    let can_substitute = |ch| (vertical && comic::compound(ch).is_some()) || !comic::equivalent_marks(ch, vertical).is_empty();
    if line_styles.peek().is_none() && groups.is_empty() && !source.chars().any(can_substitute) { return None; }

    let original_runs = layer.char_runs();
    let mut first_indents = Vec::new();
    let mut text = String::with_capacity(source.len());
    let mut runs = Vec::new();
    let mut japanese = Vec::new();
    let mut offsets = SourceOffsets::new(0);
    let mut previous: Option<Piece<'_>> = None;
    let mut previous_cross = false;
    let mut chars = source.char_indices().peekable();
    let mut index = 0;
    let mut run_end = original_runs.first()?.len;
    let mut face = None;
    let mut face_index = None;
    let mut changed = false;
    while let Some((at, ch)) = chars.next() {
        while at >= run_end && index + 1 < original_runs.len() {
            index += 1;
            run_end += original_runs[index].len;
        }
        let group = groups.at(at);
        let original = &original_runs[index].style;
        let mut style = Cow::Borrowed(original);
        while let Some((_, style)) = line_styles.next_if(|&(start, _)| start <= at) {
            line_style = style;
        }
        if let Some(leading) = line_style.leading_pt.filter(|&value| original.leading_pt != Some(value)) {
            style.to_mut().leading_pt = Some(leading);
            changed = true;
        }
        if line_style.baseline_shift_pt != 0.0 {
            style.to_mut().baseline_shift_pt += line_style.baseline_shift_pt;
            changed = true;
        }
        let mut start = text.len();
        let mut end = at + ch.len_utf8();
        let linear = group.is_some_and(|g| g.kind == GroupKind::Linear);
        if (group.is_some() || can_substitute(ch)) && face_index != Some(index) {
            let resolved = layout.styles.get(index).unwrap_or(original);
            face = fonts.selected_face(&resolved.font_family, resolved.weight, resolved.italic);
            face_index = Some(index);
        }
        if let Some(scale) = group.and_then(|g| layout.comic_scales.get(&g.source)) {
            style.to_mut().horizontal_scale = glyph_scale(original.horizontal_scale) * scale;
            changed = true;
        }
        let japanese_style = layout.japanese_styles.get(index).copied().unwrap_or(false);
        if linear && let Some(spelling) = sideways_spelling(ch, face.as_ref(), original, japanese_style) {
            let st = style.to_mut();
            st.vertical_scale = glyph_scale(original.vertical_scale) * spelling.along;
            st.horizontal_scale = glyph_scale(original.horizontal_scale) * spelling.across;
            st.baseline_shift_pt += spelling.baseline_shift_em * st.horizontal_scale * st.size_pt;
            st.tracking += spelling.advance * 1000.0;
            let bearing = spelling.leading;
            if bearing != 0.0 {
                let bearing_pt = bearing * original.size_pt;
                // Move the origin without changing the advance of the line. A paragraph
                // opening has no preceding glyph, so the first-line indent moves its origin.
                if let Some(last) = previous.as_mut().filter(|last| last.vertical_advance && last.style.size_pt > 0.0) {
                    last.style.to_mut().tracking += bearing_pt / last.style.size_pt * 1000.0;
                    st.tracking -= bearing * 1000.0;
                } else if matches!(layer.shape, TextShape::Box { .. }) && (text.is_empty() || text.ends_with(['\r', '\n'])) {
                    first_indents.push((at, bearing_pt));
                    st.tracking -= bearing * 1000.0;
                }
            }
            text.push(spelling.mark);
            changed |= spelling.mark != ch || st != original;
        } else if let Some(after) = comic::push_display_mark(&mut text, (at, ch), &mut chars, groups, face.as_ref(), vertical, |n| n < run_end) {
            end = after;
            changed = true;
        } else {
            text.push(ch);
        }
        let cross = group.is_some_and(|g| g.kind == GroupKind::Upright
            && (g.cross || comic::takes_cross_baseline(&text[start..])));
        if previous_cross && cross && group.is_some_and(|g| g.range.start == at) {
            // A regular-baseline non-joiner prevents Adobe from fusing adjacent cells.
            let separator = '\u{200C}';
            text.insert(start, separator);
            let after = start + separator.len_utf8();
            if let Some(last) = &mut previous { last.len += separator.len_utf8(); }
            offsets.record(at..at, start..after);
            start = after;
            changed = true;
        }
        if text.len() - start != end - at { offsets.record(at..end, start..text.len()); }
        if text.len() == start { continue; }
        if let Some(last) = previous.take() { append(last, &mut runs, &mut japanese); }
        previous = Some(Piece { len: text.len() - start, style, japanese: japanese_style,
            vertical_advance: !group.is_some_and(|g| g.kind == GroupKind::Upright) && !matches!(ch, '\r' | '\n' | '\u{3}' | '\u{200C}') });
        previous_cross = cross;
    }
    if !changed { return None; }
    if let Some(last) = previous { append(last, &mut runs, &mut japanese); }
    let mut styles = layer.paragraph_runs().into_iter();
    let mut paragraph = styles.next()?;
    let mut style_end = paragraph.len;
    let mut indents = first_indents.into_iter().peekable();
    layer.paragraphs = split_paragraphs(source).into_iter().map(|range| {
        while range.start >= style_end && let Some(next) = styles.next() {
            style_end += next.len;
            paragraph = next;
        }
        let mut style = paragraph.style.clone();
        if let Some((_, indent)) = indents.next_if(|&(start, _)| start == range.start) {
            style.first_line_indent_pt += indent;
        }
        ParagraphRun { len: offsets.display(range.end) - offsets.display(range.start), style }
    }).collect();
    layer.text = text;
    layer.runs = runs;
    if reserve > 0.0 && let TextShape::Box { width, height, .. } = &mut layer.shape {
        // Keep the first edge fixed while the native composer fits the larger leading.
        if vertical {
            *width += reserve;
            layer.transform = layer.transform.mul(&Affine::translate(-f64::from(reserve), 0.0));
        } else {
            *height += reserve;
        }
    }
    groups.spans.retain_mut(|span| {
        span.range = offsets.display(span.range.start)..offsets.display_end(span.range.end);
        span.cross |= span.kind == GroupKind::Upright
            && comic::takes_cross_baseline(layer.text.get(span.range.clone()).unwrap_or_default());
        !span.range.is_empty()
    });
    Some(japanese)
}
