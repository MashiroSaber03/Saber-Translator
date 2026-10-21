//! `type.*` commands: create and edit type (text) layers.
//!
//! Text offsets in parameters and results are **character** indices (Unicode scalar values),
//! so JSON clients never split a UTF-8 sequence. Sizes, leading, indents and spacing are in
//! points (converted with the document resolution); tracking is in 1/1000 em; positions and
//! boxes are document pixels. Every edit re-renders the layer with `photocraft-text` and
//! regenerates its PSD `TySh` data, so the layer saves as editable text.

use std::sync::Arc;

use photocraft_color::Color;
use photocraft_doc::text::{
    AntiAlias, Caps, CharStyle, FontFeature, FontVariation, Kerning, Orientation, ParagraphRun, ParagraphStyle, TextAlign, TextDirection, TextRun, TextShape,
};
use photocraft_doc::{Affine, Document, Layer, LayerContent, LayerId, TextLayer};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn layer_id(s: &Session, p: &Value) -> Result<LayerId> {
    match p.get("layer").and_then(Value::as_u64) {
        Some(id) => Ok(LayerId(id)),
        None => s.active().and_then(|d| d.active_layer).ok_or(EngineError::Other("no active layer".into())),
    }
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn f32p(p: &Value, k: &str) -> Option<f32> {
    p.get(k).and_then(Value::as_f64).map(|v| v as f32)
}

fn color(v: &Value) -> Option<Color> {
    match v {
        Value::Array(a) if a.len() >= 3 => {
            let c: Vec<f32> = a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            Some(Color::rgba(c[0], c[1], c[2], c.get(3).copied().unwrap_or(1.0)))
        }
        Value::String(s) => {
            let s = s.trim_start_matches('#');
            let h = |i: usize| s.get(i..i + 2).and_then(|x| u8::from_str_radix(x, 16).ok()).map(|v| f32::from(v) / 255.0);
            match s.len() {
                6 => Some(Color::rgb(h(0)?, h(2)?, h(4)?)),
                8 => Some(Color::rgba(h(0)?, h(2)?, h(4)?, h(6)?)),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Normalises line breaks to `\n`.
pub(crate) fn norm_text(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

fn byte_at(text: &str, ci: usize) -> usize {
    text.char_indices().nth(ci).map_or(text.len(), |(b, _)| b)
}

fn char_at(text: &str, bi: usize) -> usize {
    text[..bi.min(text.len())].chars().count()
}

/// `range: [start, end]` in characters → byte range (whole text when absent).
fn range_param(text: &str, p: &Value) -> (usize, usize) {
    match p.get("range").and_then(Value::as_array) {
        Some(r) if r.len() == 2 => {
            let a = r[0].as_u64().unwrap_or(0) as usize;
            let b = r[1].as_u64().unwrap_or(a as u64) as usize;
            let (a, b) = (a.min(b), a.max(b));
            (byte_at(text, a), byte_at(text, b))
        }
        _ => (0, text.len()),
    }
}

/// Photoshop's manual kerning range (1/1000 em).
pub const KERN_MIN: f64 = -1000.0;
pub const KERN_MAX: f64 = 10_000.0;
pub const TYPE_SIZE_MIN_PT: f64 = 0.1;
pub const TYPE_SIZE_MAX_PT: f64 = 1296.0;
pub const TRACKING_MIN: f64 = -1000.0;
pub const TRACKING_MAX: f64 = 10_000.0;

/// Validates the `kerning` key of character params: a number in [`KERN_MIN`]..=[`KERN_MAX`] or
/// `"metrics"`, `"optical"`, `"off"` (`"none"`, `"0"`).
pub fn check_kerning(p: &Value) -> std::result::Result<(), String> {
    match p.get("kerning") {
        None => Ok(()),
        Some(Value::Number(n)) => match n.as_f64() {
            Some(k) if (KERN_MIN..=KERN_MAX).contains(&k) => Ok(()),
            _ => Err(format!("kerning must be between {KERN_MIN} and {KERN_MAX} (1/1000 em)")),
        },
        Some(Value::String(v)) if matches!(v.to_ascii_lowercase().as_str(), "metrics" | "optical" | "off" | "none" | "0") => Ok(()),
        Some(_) => Err("kerning must be a number (1/1000 em) or \"metrics\", \"optical\" or \"off\"".into()),
    }
}

/// Validates text metrics that can otherwise exceed the rasterizer's supported range.
pub fn check_size_tracking(p: &Value) -> std::result::Result<(), String> {
    if let Some(v) = p.get("size") {
        match v.as_f64() {
            Some(size) if size.is_finite() && (TYPE_SIZE_MIN_PT..=TYPE_SIZE_MAX_PT).contains(&size) => {}
            _ => return Err(format!("size must be between {TYPE_SIZE_MIN_PT} and {TYPE_SIZE_MAX_PT} pt")),
        }
    }
    if let Some(v) = p.get("tracking") {
        match v.as_f64() {
            Some(tracking) if tracking.is_finite() && (TRACKING_MIN..=TRACKING_MAX).contains(&tracking) => {}
            _ => return Err(format!("tracking must be between {TRACKING_MIN} and {TRACKING_MAX} (1/1000 em)")),
        }
    }
    Ok(())
}

/// The PostScript name of the installed face of `family` at exactly `weight`/`italic`. A PSD names
/// a type layer by PostScript name, and Photoshop treats a name no font has as missing (#3556).
/// Only an exact face is named: layout picks a named face before weight and italic, so naming a
/// nearby face would override the requested weight (a variable font's axis, a weight between two
/// faces) or italic. Without an exact face the style keeps no name and layout matches as before.
fn installed_postscript(family: &str, weight: u16, italic: bool) -> Option<String> {
    let mut shared = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    let faces = shared.fonts.faces(family);
    let face = faces.iter().find(|f| f.italic == italic && f.weight.round() as u16 == weight)?;
    face.postscript_name.clone()
}

/// Photoshop's leading range in points.
pub const LEADING_MIN_PT: f32 = 0.01;
pub const LEADING_MAX_PT: f32 = 5000.0;
/// Photoshop's baseline shift range in points (either direction).
pub const BASELINE_SHIFT_MAX_PT: f32 = 5000.0;
/// The relative character steps `type.setStyle` accepts, with the largest step allowed.
const STEP_KEYS: [(&str, f64); 3] = [("leadingBy", 5000.0), ("baselineShiftBy", 10_000.0), ("trackingBy", 11_000.0)];

/// Validates the relative steps (`leadingBy`, `baselineShiftBy`, `trackingBy`): finite numbers
/// no larger than the whole range they move through.
pub fn check_steps(p: &Value) -> std::result::Result<(), String> {
    for (key, max) in STEP_KEYS {
        if let Some(v) = p.get(key) {
            match v.as_f64() {
                Some(by) if by.is_finite() && by.abs() <= max => {}
                _ => return Err(format!("{key} must be a number between -{max} and {max}")),
            }
        }
    }
    Ok(())
}

/// Relative character steps (Photoshop's ⌥↑/↓ leading, ⇧⌥↑/↓ baseline shift and ⌥←/→ tracking
/// over a selection): every character moves from its own value, so mixed values keep their
/// differences. Auto leading starts from the value layout uses, the paragraph's auto-leading
/// percentage of the character's size. Results are clamped to Photoshop's ranges. Returns true
/// if any step key was present.
fn step_char_metrics(t: &mut TextLayer, a: usize, b: usize, p: &Value) -> bool {
    let leading_by = f32p(p, "leadingBy").filter(|v| v.is_finite());
    let shift_by = f32p(p, "baselineShiftBy").filter(|v| v.is_finite());
    let tracking_by = f32p(p, "trackingBy").filter(|v| v.is_finite());
    if leading_by.is_none() && shift_by.is_none() && tracking_by.is_none() {
        return false;
    }
    let paragraphs = t.paragraph_runs();
    let default_auto = ParagraphStyle::default().auto_leading;
    // Each paragraph's share of `a..b`, with the auto leading it lays out with.
    let mut pieces = Vec::new();
    let mut at = 0usize;
    for pr in &paragraphs {
        let (ps, pe) = (at, at.saturating_add(pr.len));
        at = pe;
        let (pa, pb) = (a.clamp(ps, pe), b.clamp(ps, pe));
        if pa < pb {
            pieces.push((pa, pb, pr.style.auto_leading));
        }
    }
    if at < b {
        pieces.push((a.max(at), b, default_auto));
    }
    if t.text.is_empty() {
        // Empty text: the step sets the style the next typed characters get.
        pieces = vec![(0, 0, paragraphs.first().map_or(default_auto, |r| r.style.auto_leading))];
    }
    for (pa, pb, auto) in pieces {
        style_range(t, pa, pb, &|st| {
            if let Some(by) = leading_by {
                let auto = if auto.is_finite() && auto > 0.0 { auto } else { default_auto };
                let base = st.leading_pt.unwrap_or(auto * st.size_pt);
                let base = if base.is_finite() { base } else { 0.0 };
                st.leading_pt = Some((base + by).clamp(LEADING_MIN_PT, LEADING_MAX_PT));
            }
            if let Some(by) = shift_by {
                let base = if st.baseline_shift_pt.is_finite() { st.baseline_shift_pt } else { 0.0 };
                st.baseline_shift_pt = (base + by).clamp(-BASELINE_SHIFT_MAX_PT, BASELINE_SHIFT_MAX_PT);
            }
            if let Some(by) = tracking_by {
                let base = if st.tracking.is_finite() { st.tracking } else { 0.0 };
                st.tracking = (base + by).clamp(TRACKING_MIN as f32, TRACKING_MAX as f32);
            }
        });
    }
    true
}

/// Applies character-style keys from JSON. Returns true if any key was present.
pub fn apply_char_props(s: &mut CharStyle, p: &Value) -> bool {
    let mut any = false;
    let mut hit = |b: bool| any |= b;
    let font_key = ["font", "family", "fontStyle", "weight", "italic"].iter().any(|k| p.get(k).is_some());
    if let Some(v) = p.get("font").or_else(|| p.get("family")).and_then(Value::as_str) {
        s.font_family = v.to_string();
        s.postscript_name = None;
        hit(true);
    }
    if let Some(v) = p.get("fontStyle").and_then(Value::as_str) {
        s.font_style = v.to_string();
        s.postscript_name = None;
        // The style sets the weight, so a weight slider value set earlier stops overriding it
        // (`variations` in the same call apply after this).
        s.variations.retain(|f| f.axis != "wght");
        let l = v.to_lowercase();
        s.italic = l.contains("italic") || l.contains("oblique");
        let g = photocraft_text::fonts::guess_from_postscript(&format!("X-{}", v.replace(' ', "")));
        s.weight = g.weight;
        if let Some(face) = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).fonts.named_face(&s.font_family, v) {
            s.weight = face.weight.round() as u16;
            s.italic = face.italic;
            s.postscript_name = face.postscript_name;
        }
        hit(true);
    }
    if let Some(v) = p.get("weight").and_then(Value::as_f64) {
        let weight = v.clamp(1.0, 1000.0) as u16;
        if s.weight != weight {
            s.postscript_name = None;
            s.font_style.clear();
        }
        s.variations.retain(|f| f.axis != "wght");
        s.weight = weight;
        hit(true);
    }
    if let Some(v) = p.get("italic").and_then(Value::as_bool) {
        if s.italic != v {
            s.postscript_name = None;
            s.font_style.clear();
        }
        s.italic = v;
        hit(true);
    }
    if let Some(v) = p.get("postscriptName").and_then(Value::as_str) {
        s.postscript_name = Some(v.to_string());
        hit(true);
    }
    // A font named by family (or a style that matched no face) has no PostScript name yet: fill in
    // the installed face's real name, so the PSD does not invent "<Family>-Regular" (#3556).
    if font_key && s.postscript_name.is_none() {
        s.postscript_name = installed_postscript(&s.font_family, s.weight, s.italic);
    }
    if let Some(v) = f32p(p, "size") {
        s.size_pt = v.max(0.01);
        hit(true);
    }
    if let Some(c) = p.get("color").and_then(color) {
        s.color = c;
        hit(true);
    }
    if let Some(v) = f32p(p, "tracking") {
        s.tracking = v;
        hit(true);
    }
    match p.get("leading") {
        Some(Value::Null) => {
            s.leading_pt = None;
            hit(true);
        }
        Some(Value::String(a)) if a == "auto" => {
            s.leading_pt = None;
            hit(true);
        }
        Some(v) if v.is_number() => {
            s.leading_pt = v.as_f64().map(|x| x as f32);
            hit(true);
        }
        _ => {}
    }
    for (k, f) in [("baselineShift", &mut s.baseline_shift_pt), ("horizontalScale", &mut s.horizontal_scale), ("verticalScale", &mut s.vertical_scale)] {
        if let Some(v) = f32p(p, k) {
            *f = if k == "baselineShift" { v } else { (v / 100.0).max(0.01) };
            hit(true);
        }
    }
    for (k, f) in [
        ("underline", &mut s.underline),
        ("strikethrough", &mut s.strikethrough),
        ("fauxBold", &mut s.faux_bold),
        ("fauxItalic", &mut s.faux_italic),
        ("ligatures", &mut s.ligatures),
        ("discretionaryLigatures", &mut s.discretionary_ligatures),
    ] {
        if let Some(v) = p.get(k).and_then(Value::as_bool) {
            *f = v;
            hit(true);
        }
    }
    match p.get("kerning") {
        // A number is Photoshop's manual kerning: no automatic pair kerning plus the value.
        Some(v) if v.is_number() => {
            if let Some(k) = v.as_f64().filter(|k| k.is_finite()) {
                s.kerning = Kerning::Off;
                s.kern = k.clamp(KERN_MIN, KERN_MAX) as f32;
                hit(true);
            }
        }
        Some(Value::String(v)) => {
            s.kerning = match v.to_ascii_lowercase().as_str() {
                "optical" => Kerning::Optical,
                "none" | "off" | "0" => Kerning::Off,
                _ => Kerning::Metrics,
            };
            s.kern = 0.0;
            hit(true);
        }
        _ => {}
    }
    if let Some(v) = p.get("caps").and_then(Value::as_str) {
        s.caps = match v {
            "small" | "smallCaps" => Caps::SmallCaps,
            "all" | "allCaps" => Caps::AllCaps,
            _ => Caps::Normal,
        };
        hit(true);
    }
    if let Some(Value::Object(m)) = p.get("features") {
        for (tag, v) in m {
            let value = v.as_u64().map(|x| x as u16).or_else(|| v.as_bool().map(u16::from)).unwrap_or(1);
            s.features.retain(|f| &f.tag != tag);
            s.features.push(FontFeature { tag: tag.clone(), value });
        }
        hit(true);
    }
    if let Some(Value::Object(m)) = p.get("variations") {
        for (axis, v) in m {
            s.variations.retain(|f| &f.axis != axis);
            if let Some(x) = v.as_f64() {
                s.variations.push(FontVariation { axis: axis.clone(), value: x as f32 });
            }
        }
        hit(true);
    }
    if let Some(v) = p.get("language") {
        s.language = v.as_str().map(str::to_string);
        hit(true);
    }
    any
}

/// Applies paragraph-style keys from JSON. Returns true if any key was present.
pub fn apply_para_props(s: &mut ParagraphStyle, p: &Value) -> bool {
    let mut any = false;
    if let Some(v) = p.get("align").and_then(Value::as_str) {
        s.align = match v {
            "center" => TextAlign::Center,
            "right" => TextAlign::Right,
            "justify" | "justifyLeft" => TextAlign::JustifyLeft,
            "justifyCenter" => TextAlign::JustifyCenter,
            "justifyRight" => TextAlign::JustifyRight,
            "justifyAll" => TextAlign::JustifyAll,
            _ => TextAlign::Left,
        };
        any = true;
    }
    for (k, f) in [
        ("firstLineIndent", &mut s.first_line_indent_pt),
        ("startIndent", &mut s.start_indent_pt),
        ("endIndent", &mut s.end_indent_pt),
        ("spaceBefore", &mut s.space_before_pt),
        ("spaceAfter", &mut s.space_after_pt),
        ("autoLeading", &mut s.auto_leading),
    ] {
        if let Some(v) = f32p(p, k) {
            *f = if k == "autoLeading" { v / 100.0 } else { v };
            any = true;
        }
    }
    if let Some(v) = p.get("direction").and_then(Value::as_str) {
        s.direction = match v {
            "ltr" => TextDirection::Ltr,
            "rtl" => TextDirection::Rtl,
            _ => TextDirection::Auto,
        };
        any = true;
    }
    if let Some(v) = p.get("hyphenate").and_then(Value::as_bool) {
        s.hyphenate = v;
        any = true;
    }
    any
}

fn push_merge<S: PartialEq>(out: &mut Vec<(usize, S)>, len: usize, st: S) {
    if len == 0 {
        return;
    }
    match out.last_mut() {
        Some(last) if last.1 == st => last.0 += len,
        _ => out.push((len, st)),
    }
}

/// Applies `f` to the character style of bytes `a..b` (splitting runs at the edges).
pub fn style_range(t: &mut TextLayer, a: usize, b: usize, f: &dyn Fn(&mut CharStyle)) {
    // A reversed range styles nothing; `b < a` would underflow the styled run's length (#714).
    let b = b.max(a);
    let runs = t.char_runs();
    if t.text.is_empty() {
        let mut st = runs.into_iter().next().map(|r| r.style).unwrap_or_default();
        f(&mut st);
        t.runs = vec![TextRun { len: 0, style: st }];
        t.sync_summary();
        return;
    }
    let mut out: Vec<(usize, CharStyle)> = Vec::new();
    let mut at = 0;
    for r in runs {
        let (rs, re) = (at, at + r.len);
        at = re;
        let (ca, cb) = (a.clamp(rs, re), b.clamp(rs, re));
        push_merge(&mut out, ca - rs, r.style.clone());
        let mut st = r.style.clone();
        f(&mut st);
        push_merge(&mut out, cb - ca, st);
        push_merge(&mut out, re - cb, r.style);
    }
    t.runs = out.into_iter().map(|(len, style)| TextRun { len, style }).collect();
    t.sync_summary();
}

/// Applies `f` to every paragraph touching bytes `a..b` (paragraph runs follow paragraphs).
pub fn style_paragraphs(t: &mut TextLayer, a: usize, b: usize, f: &dyn Fn(&mut ParagraphStyle)) {
    let old = t.paragraph_runs();
    let starts: Vec<usize> = old
        .iter()
        .scan(0, |acc, r| {
            let s = *acc;
            *acc += r.len;
            Some(s)
        })
        .collect();
    let style_at = |off: usize| old[starts.iter().rposition(|&s| s <= off).unwrap_or(0)].style.clone();
    let mut out: Vec<(usize, ParagraphStyle)> = Vec::new();
    let trailing_break = t.text.ends_with(['\r', '\n']);
    for pr in photocraft_text::layout::split_paragraphs(&t.text) {
        let mut st = style_at(pr.start);
        let terminal_empty = pr.start == pr.end;
        let touches =
            (pr.start < b && pr.end > a) || (a == b && a >= pr.start && (a < pr.end || terminal_empty || (pr.end == t.text.len() && !trailing_break)));
        if touches {
            f(&mut st);
        }
        if terminal_empty && trailing_break {
            // This paragraph has no bytes but must retain its own style for the
            // next typed character and for the PSD's final paragraph marker.
            out.push((0, st));
        } else {
            push_merge(&mut out, pr.len(), st.clone());
            if out.is_empty() {
                out.push((0, st));
            }
        }
    }
    t.paragraphs = out.into_iter().map(|(len, style)| ParagraphRun { len, style }).collect();
}

/// Replaces bytes `a..b` with `new`, keeping styles: inserted text takes the style of the
/// character before it (or after it at the start).
pub fn replace_text(t: &mut TextLayer, a: usize, b: usize, new: &str) {
    fn adjust<S: Clone + PartialEq>(runs: Vec<(usize, S)>, a: usize, b: usize, ins: usize, trailing_paragraph: bool) -> Vec<(usize, S)> {
        let mut out: Vec<(usize, S)> = Vec::new();
        let mut at = 0;
        let host = if trailing_paragraph && a == b && runs.last().is_some_and(|r| r.0 == 0) && a == runs.iter().map(|r| r.0).sum::<usize>() {
            // Inserted text belongs to a styled final empty paragraph, not to
            // the preceding paragraph whose break ends at this byte offset.
            runs.len() - 1
        } else {
            runs.iter()
                .scan(0, |acc, r| {
                    let s = *acc;
                    *acc += r.0;
                    Some((s, s + r.0))
                })
                .position(|(s, e)| if a == 0 { e > 0 || s == 0 } else { a > s && a <= e })
                .unwrap_or(0)
        };
        for (i, (len, st)) in runs.into_iter().enumerate() {
            let (rs, re) = (at, at + len);
            at = re;
            let keep = (a.clamp(rs, re) - rs) + (re - b.clamp(rs, re));
            let add = if i == host { ins } else { 0 };
            push_merge(&mut out, keep + add, st);
        }
        out
    }
    let (a, b) = (a.min(t.text.len()), b.min(t.text.len()).max(a.min(t.text.len())));
    let new = norm_text(new);
    let cr: Vec<(usize, CharStyle)> = t.char_runs().into_iter().map(|r| (r.len, r.style)).collect();
    let pr: Vec<(usize, ParagraphStyle)> = t.paragraph_runs().into_iter().map(|r| (r.len, r.style)).collect();
    let first_c = cr.first().map(|r| r.1.clone()).unwrap_or_default();
    let first_p = pr.first().map(|r| r.1.clone()).unwrap_or_default();
    let trailing_style = t.text.ends_with(['\r', '\n']).then(|| pr.last().filter(|r| r.0 == 0).map(|r| r.1.clone())).flatten();
    let mut cr = adjust(cr, a, b, new.len(), false);
    let mut pr = adjust(pr, a, b, new.len(), true);
    t.text.replace_range(a..b, &new);
    if t.text.ends_with(['\r', '\n'])
        && let Some(style) = trailing_style
    {
        pr.push((0, style));
    }
    if cr.is_empty() {
        cr.push((t.text.len(), first_c));
    }
    if pr.is_empty() {
        pr.push((t.text.len(), first_p));
    }
    t.runs = cr.into_iter().map(|(len, style)| TextRun { len, style }).collect();
    t.paragraphs = pr.into_iter().map(|(len, style)| ParagraphRun { len, style }).collect();
    t.sync_summary();
}

/// Re-renders the layer's pixels and regenerates its PSD `TySh` data.
pub fn refresh(doc: &Document, t: &mut TextLayer) {
    let dpi = doc.resolution_dpi;
    let mut eng = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    let (layout, r) = eng.render(t, dpi, doc.pixel_format());
    t.cache = Some(r.surface);
    t.psd_raw = Some(Arc::new(photocraft_text::psd::build_tysh(t, dpi, layout.bounds())));
}

/// Refuses to change a type layer under Lock All (its own or an enclosing group's), with
/// Photoshop's message. Every command that edits a type layer's text calls it first (#3132, #4188).
pub(crate) fn check_text_unlocked(doc: &Document, id: LayerId) -> Result<()> {
    let layer = doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    if doc.effective_locks(id).all {
        return Err(EngineError::Other(format!("Could not complete your request because the layer \"{}\" is locked", layer.name)));
    }
    Ok(())
}

/// Runs `f` on a type layer as one undo step. `f` may set an explicit layer name (`type.edit`'s
/// `name`), which is applied in the same step; otherwise an auto-named layer follows its text.
fn with_text_layer<R>(s: &mut Session, p: &Value, label: &str, f: impl FnOnce(&mut TextLayer, &Document, &mut Option<String>) -> Result<R>) -> Result<R> {
    let id = layer_id(s, p)?;
    check_text_unlocked(&s.active().ok_or(EngineError::NoDocument)?.doc, id)?;
    let (r, damage) = s.edit(label, |doc, _| {
        let snapshot = doc.clone();
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let auto_named = is_auto_named(l);
        let LayerContent::Text(t) = &mut l.content else {
            return Err(EngineError::Other(format!("layer {} is {} {} layer, not a type layer", id.0, l.content.article(), l.content.kind_name())));
        };
        let before = t.cache.as_ref().map(|c| c.tile_bounds());
        let mut rename = None;
        let r = f(t, &snapshot, &mut rename)?;
        refresh(&snapshot, t);
        // Only this layer's pixels changed: the canvas recomposites their old and new area
        // instead of the whole document (#124). Unknown old pixels mean a full refresh.
        let damage = before.zip(t.cache.as_ref().map(|c| c.tile_bounds())).map(|(a, b)| a.union(&b));
        let name = rename.or_else(|| auto_named.then(|| layer_name(&t.text)));
        if let Some(n) = name {
            l.name = n;
        }
        Ok((r, damage))
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = damage;
    }
    Ok(r)
}

/// Direct formatting follows the selected Type layers, while an explicit layer or character
/// range keeps its single-layer scope. Selected groups are not expanded.
pub(crate) fn format_targets(s: &Session, p: &Value) -> Result<Vec<LayerId>> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let single = p.get("layer").is_some() || (p.get("range").is_some() && p.get("layers").is_none());
    let ids = if single {
        vec![layer_id(s, p)?]
    } else if let Some(value) = p.get("layers") {
        value
            .as_array()
            .ok_or_else(|| EngineError::Other("layers must be an array of layer IDs".into()))?
            .iter()
            .map(|v| v.as_u64().map(LayerId).ok_or_else(|| EngineError::Other("layers must contain layer IDs".into())))
            .collect::<Result<Vec<_>>>()?
    } else {
        st.selected_layers()
    };
    let mut targets = Vec::new();
    for id in ids {
        let l = st.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
        if matches!(l.content, LayerContent::Text(_)) {
            if !targets.contains(&id) {
                targets.push(id);
            }
        } else if single {
            return Err(EngineError::Other(format!("layer {} is {} {} layer, not a type layer", id.0, l.content.article(), l.content.kind_name())));
        }
    }
    if targets.is_empty() {
        return Err(EngineError::Other("select a type layer".into()));
    }
    Ok(targets)
}

pub(crate) fn has_format_target(s: &Session) -> std::result::Result<(), String> {
    format_targets(s, &Value::Null).map(|_| ()).map_err(|e| e.to_string())
}

/// One history step for a formatting batch. Structural text edits keep their singular helper.
pub(crate) fn with_text_targets(
    s: &mut Session,
    p: &Value,
    label: &str,
    mut f: impl FnMut(&mut TextLayer, (usize, usize)) -> Result<()>,
) -> Result<Vec<LayerId>> {
    let ids = format_targets(s, p)?;
    let damage = s.edit(label, |doc, _| {
        let snapshot = doc.clone();
        let mut damage = Some(photocraft_geom::Rect::EMPTY);
        for id in &ids {
            let Some(LayerContent::Text(t)) = doc.layer_mut(*id).map(|l| &mut l.content) else { return Err(EngineError::NoLayer(*id)) };
            let before = t.cache.as_ref().map(|c| c.tile_bounds());
            let range = if p.get("layers").is_none() || p.get("layer").is_some() { range_param(&t.text, p) } else { (0, t.text.len()) };
            f(t, range)?;
            refresh(&snapshot, t);
            // An unknown old/new cache retains Session.edit's full-refresh fallback.
            damage = damage.zip(before).zip(t.cache.as_ref().map(|c| c.tile_bounds())).map(|((all, a), b)| all.union(&a).union(&b));
        }
        Ok(damage)
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = damage;
    }
    Ok(ids)
}

/// UI size/leading are measured after the layer transform; command callers otherwise use points.
fn shown_style_params<'a>(t: &TextLayer, p: &'a Value) -> Result<std::borrow::Cow<'a, Value>> {
    if p.get("metricsAsShown").and_then(Value::as_bool) != Some(true) {
        return Ok(std::borrow::Cow::Borrowed(p));
    }
    let m = t.transform.m;
    let scale = (m[0] * m[3] - m[1] * m[2]).abs().sqrt() as f32;
    let scale = if scale.is_finite() && scale > 1e-3 { f64::from(scale) } else { 1.0 };
    let mut params = p.clone();
    for key in ["size", "leading", "leadingBy"] {
        if let Some(value) = p.get(key).and_then(Value::as_f64) {
            let value = value / scale;
            if !value.is_finite() || value.abs() > f64::from(f32::MAX) {
                return Err(bad("type.setStyle", "shown size and leading must convert to finite point values"));
            }
            params[key] = json!(value);
        }
    }
    Ok(std::borrow::Cow::Owned(params))
}

/// Whether a type layer still has the name it got from its text, so the name follows edits to
/// the text. A layer renamed to anything else keeps its name (#483).
pub(crate) fn is_auto_named(l: &Layer) -> bool {
    matches!(&l.content, LayerContent::Text(t) if l.name == layer_name(&t.text))
}

/// The name a type layer gets from its text: its first non-empty line, at most 40 characters.
/// The Type tool names new layers with this too, so `is_auto_named` recognises them.
pub fn layer_name(text: &str) -> String {
    let first = text.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    let name: String = first.chars().take(40).collect();
    if name.is_empty() { "Type Layer".into() } else { name }
}

fn info(s: &Session, p: &Value) -> Result<Value> {
    let id = layer_id(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let LayerContent::Text(t) = &l.content else {
        return Err(EngineError::Other(format!("layer {} is not a type layer", id.0)));
    };
    let text = &t.text;
    let mut at = 0;
    let runs: Vec<Value> = t
        .char_runs()
        .into_iter()
        .map(|r| {
            let v = json!({ "start": char_at(text, at), "end": char_at(text, at + r.len), "style": r.style });
            at += r.len;
            v
        })
        .collect();
    let mut at = 0;
    let paragraphs: Vec<Value> = t
        .paragraph_runs()
        .into_iter()
        .map(|r| {
            let v = json!({ "start": char_at(text, at), "end": char_at(text, at + r.len), "style": r.style });
            at += r.len;
            v
        })
        .collect();
    let layout = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner()).layout(t, d.doc.resolution_dpi);
    let lines: Vec<Value> = layout
        .lines
        .iter()
        .map(|ln| json!({ "start": char_at(text, ln.range.start), "end": char_at(text, ln.range.end), "baseline": ln.baseline, "x0": ln.x0, "x1": ln.x1, "ascent": ln.ascent, "descent": ln.descent }))
        .collect();
    let bounds = t.cache.as_ref().map(|c| c.content_bounds()).map(|r| json!([r.x0, r.y0, r.x1, r.y1]));
    let warnings = crate::type_extra_cmds::missing_font_warnings_for_layer(t);
    let mut out = json!({
        "layer": id.0,
        "text": text,
        "shape": t.shape,
        "transform": t.transform.m,
        "antialias": t.antialias,
        "orientation": t.orientation,
        "warp": t.warp,
        "runs": runs,
        "paragraphs": paragraphs,
        "lines": lines,
        "bounds": bounds,
        "pxPerPt": layout.px_per_pt,
    });
    if !warnings.is_empty() {
        out["warnings"] = json!(warnings);
    }
    Ok(out)
}

/// A type layer (not yet rendered or added) from `type.create`'s params: text, point `x`/`y` or
/// paragraph `box`, `orientation`, character and paragraph keys, starting from the default type
/// styles in the foreground colour. Shared with the Type Mask tools' `type.createSelection`.
pub(crate) fn new_text_layer(s: &Session, p: &Value, cmd: &str) -> Result<TextLayer> {
    let orientation = match p.get("orientation") {
        None => Orientation::Horizontal,
        Some(Value::String(value)) if value == "horizontal" => Orientation::Horizontal,
        Some(Value::String(value)) if value == "vertical" => Orientation::Vertical,
        _ => return Err(bad(cmd, "orientation must be horizontal or vertical")),
    };
    check_kerning(p).map_err(|m| bad(cmd, m))?;
    check_size_tracking(p).map_err(|m| bad(cmd, m))?;
    let text = norm_text(p.get("text").and_then(Value::as_str).unwrap_or(""));
    // Type › Save Default Type Styles sets the starting styles; the colour is always
    // the foreground colour, as in Photoshop.
    let (mut style, mut para) = s
        .type_defaults
        .clone()
        .unwrap_or_else(|| (CharStyle { font_family: photocraft_text::fonts::DEFAULT_FAMILY.into(), ..Default::default() }, ParagraphStyle::default()));
    let fg = s.tools.foreground;
    style.color = Color::rgba(fg[0], fg[1], fg[2], fg[3]);
    apply_char_props(&mut style, p);
    apply_para_props(&mut para, p);
    let (shape, transform) = match p.get("box").and_then(Value::as_array) {
        Some(b) if b.len() == 4 => {
            let v: Vec<f32> = b.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            if v[2] <= 0.0 || v[3] <= 0.0 {
                return Err(bad(cmd, "box width and height must be positive"));
            }
            (TextShape::Box { x: 0.0, y: 0.0, width: v[2], height: v[3] }, Affine::translate(f64::from(v[0]), f64::from(v[1])))
        }
        _ => (TextShape::Point, Affine::translate(f32p(p, "x").unwrap_or(0.0).into(), f32p(p, "y").unwrap_or(0.0).into())),
    };
    let mut t = TextLayer {
        runs: vec![TextRun { len: text.len(), style }],
        paragraphs: vec![ParagraphRun { len: text.len(), style: para }],
        text,
        shape,
        transform,
        orientation,
        overflow_visible: s.type_overflow_visible,
        ..Default::default()
    };
    t.sync_summary();
    Ok(t)
}

const CHAR_PARAMS: &str = r##""font":str,"fontStyle":str,"weight":100..900,"italic":bool,"size":0.1..=1296 pt,"color":"#rrggbb"|[r,g,b,a],"tracking":-1000..=10000 (1/1000 em),"leading":pt|"auto","baselineShift":pt,"horizontalScale":%,"verticalScale":%,"underline":bool,"strikethrough":bool,"fauxBold":bool,"fauxItalic":bool,"kerning":1/1000em (manual, after each character)|"metrics"|"optical"|"off","caps":"normal|small|all","ligatures":bool,"discretionaryLigatures":bool,"features":{"ss01":1},"variations":{"wght":650},"language":str"##;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "type.create",
            label: "New Type Layer",
            menu: &["Layer", "New"],
            shortcut: None,
            params: r##"{"x":px,"y":px (baseline anchor of point text),"text":str,"box":[x,y,w,h]? (paragraph text),"name":str?,"align":"left|center|right|justify…"?,"orientation":"horizontal|vertical"="horizontal", …character keys: "font","size":pt=12,"color",…}"##,
            enabled: has_doc,
            journal: true,
            run: |s, p| {
                let mut t = new_text_layer(s, p, "type.create")?;
                let name = p.get("name").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| layer_name(&t.text));
                let id = s.edit("New Type Layer", |doc, active| {
                    refresh(doc, &mut t);
                    let id = doc.insert_above(*active, Layer::new(name, LayerContent::Text(t)));
                    *active = Some(id);
                    Ok(id)
                })?;
                let bounds = s.active().and_then(|d| d.doc.layer(id)).and_then(|l| l.surface()).map(|c| c.content_bounds());
                let mut out = json!({ "layer": id.0, "bounds": bounds.map(|r| [r.x0, r.y0, r.x1, r.y1]) });
                if let Some(LayerContent::Text(t)) = s.active().and_then(|d| d.doc.layer(id)).map(|l| &l.content) {
                    let warnings = crate::type_extra_cmds::missing_font_warnings_for_layer(t);
                    if !warnings.is_empty() {
                        out["warnings"] = json!(warnings);
                    }
                }
                Ok(out)
            },
        },
        CommandSpec {
            id: "type.edit",
            label: "Edit Type",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?,"text":str? (replace all, styles kept),"replace":{"start":char,"end":char,"text":str}?,"runs":[{"start":char,"end":char,…character keys}]?,"box":[x,y,w,h]? (to paragraph text),"point":[x,y]? (to point text),"move":[dx,dy]?,"transform":[a,b,c,d,e,f]?,"antialias":"none|sharp|crisp|strong|smooth"?,"name":str?,"kerning":1/1000em|"metrics"|"optical"|"off"? (with "range":[startChar,endChar]?, default all),"kernPair":{"at":caretChar,"by":1/1000em}? (Alt+←/→: the pair before the caret becomes manual, its current kerning + by)}"##,
            enabled: has_doc,
            journal: true,
            run: |s, p| {
                let id = layer_id(s, p)?;
                let name = p.get("name").and_then(Value::as_str).map(str::to_string);
                check_kerning(p).map_err(|m| bad("type.edit", m))?;
                check_size_tracking(p).map_err(|m| bad("type.edit", m))?;
                if let Some(Value::Array(runs)) = p.get("runs") {
                    for (i, r) in runs.iter().enumerate() {
                        check_kerning(r).map_err(|m| bad("type.edit", m))?;
                        // Checked before anything changes, so a bad run leaves the layer untouched (#714).
                        let start = r.get("start").and_then(Value::as_u64).unwrap_or(0);
                        if let Some(end) = r.get("end").and_then(Value::as_u64)
                            && end < start
                        {
                            return Err(bad("type.edit", format!("runs[{i}]: `end` ({end}) is before `start` ({start})")));
                        }
                        check_size_tracking(r).map_err(|m| bad("type.edit", m))?;
                    }
                }
                let kern_pair = match p.get("kernPair") {
                    None => None,
                    Some(k) => {
                        let at = k.get("at").and_then(Value::as_u64).ok_or_else(|| bad("type.edit", "kernPair.at must be a caret position (character index)"))?;
                        let by = k
                            .get("by")
                            .and_then(Value::as_f64)
                            .filter(|v| v.is_finite() && v.abs() <= KERN_MAX)
                            .ok_or_else(|| bad("type.edit", "kernPair.by must be a number of 1/1000 em"))?;
                        Some((at as usize, by))
                    }
                };
                let label = if kern_pair.is_some() { "Kerning" } else { "Edit Type" };
                with_text_layer(s, p, label, |t, doc, rename| {
                    if let Some((at, by)) = kern_pair {
                        // Photoshop's Alt+←/→: the pair before the caret becomes manually kerned,
                        // starting from what it shows now (its metrics/optical or manual value).
                        let n = t.text.chars().count();
                        if at == 0 || at >= n {
                            return Err(bad("type.edit", "kernPair needs a caret between two characters"));
                        }
                        let a = byte_at(&t.text, at - 1);
                        let b = byte_at(&t.text, at);
                        let now = {
                            let mut eng = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
                            eng.pair_kerning(t, doc.resolution_dpi, a)
                        }
                        .ok_or_else(|| bad("type.edit", "no kerning pair at the caret (line break or line end)"))?;
                        let value = (f64::from(now.round()) + by).clamp(KERN_MIN, KERN_MAX) as f32;
                        style_range(t, a, b, &|st| {
                            st.kerning = Kerning::Off;
                            st.kern = value;
                        });
                    }
                    if p.get("kerning").is_some() {
                        let (a, b) = range_param(&t.text, p);
                        style_range(t, a, b, &|st| {
                            apply_char_props(st, &json!({ "kerning": p.get("kerning") }));
                        });
                    }
                    if let Some(v) = p.get("text").and_then(Value::as_str) {
                        let n = t.text.len();
                        replace_text(t, 0, n, v);
                    }
                    if let Some(r) = p.get("replace") {
                        let a = byte_at(&t.text, r.get("start").and_then(Value::as_u64).unwrap_or(0) as usize);
                        let b = byte_at(&t.text, r.get("end").and_then(Value::as_u64).map_or(usize::MAX, |x| x as usize));
                        let new = r.get("text").and_then(Value::as_str).unwrap_or("");
                        replace_text(t, a, b.max(a), new);
                    }
                    if let Some(Value::Array(runs)) = p.get("runs") {
                        for r in runs {
                            let a = byte_at(&t.text, r.get("start").and_then(Value::as_u64).unwrap_or(0) as usize);
                            let b = byte_at(&t.text, r.get("end").and_then(Value::as_u64).map_or(usize::MAX, |x| x as usize));
                            style_range(t, a, b, &|st| {
                                apply_char_props(st, r);
                            });
                        }
                    }
                    let lin = t.transform.m;
                    if let Some(b) = p.get("box").and_then(Value::as_array).filter(|b| b.len() == 4) {
                        let v: Vec<f64> = b.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
                        t.shape = TextShape::Box { x: 0.0, y: 0.0, width: v[2].max(1.0) as f32, height: v[3].max(1.0) as f32 };
                        t.transform = Affine { m: [lin[0], lin[1], lin[2], lin[3], v[0], v[1]] };
                    }
                    if let Some(pt) = p.get("point").and_then(Value::as_array).filter(|b| b.len() == 2) {
                        t.shape = TextShape::Point;
                        t.transform = Affine { m: [lin[0], lin[1], lin[2], lin[3], pt[0].as_f64().unwrap_or(0.0), pt[1].as_f64().unwrap_or(0.0)] };
                    }
                    if let Some(d) = p.get("move").and_then(Value::as_array).filter(|b| b.len() == 2) {
                        t.transform.m[4] += d[0].as_f64().unwrap_or(0.0);
                        t.transform.m[5] += d[1].as_f64().unwrap_or(0.0);
                    }
                    if let Some(m) = p.get("transform").and_then(Value::as_array).filter(|b| b.len() == 6) {
                        for (i, v) in m.iter().enumerate() {
                            t.transform.m[i] = v.as_f64().unwrap_or(t.transform.m[i]);
                        }
                    }
                    if let Some(a) = p.get("antialias").and_then(Value::as_str) {
                        t.antialias = match a {
                            "none" => AntiAlias::None,
                            "sharp" => AntiAlias::Sharp,
                            "crisp" => AntiAlias::Crisp,
                            "strong" => AntiAlias::Strong,
                            _ => AntiAlias::Smooth,
                        };
                    }
                    // The rename lands in the same undo step as the edit (#497).
                    *rename = name.clone();
                    Ok(())
                })?;
                info(s, &json!({ "layer": id.0 }))
            },
        },
        CommandSpec {
            id: "type.setStyle",
            label: "Set Type Style",
            menu: &[],
            shortcut: None,
            params: Box::leak(
                format!(r##"{{"layer":id? or "layers":[id,…]? (default selected Type layers),"range":[startChar,endChar]? (single layer; default all),"metricsAsShown":bool=false (size/leading/leadingBy include each layer's transform scale), {CHAR_PARAMS}, relative steps from each character's own value: "leadingBy":pt (auto leading starts from its laid-out value; result 0.01..=5000),"baselineShiftBy":pt (result -5000..=5000),"trackingBy":1/1000 em (result -1000..=10000), paragraph keys: "align":"left|center|right|justify|justifyCenter|justifyRight|justifyAll","firstLineIndent":pt,"startIndent":pt,"endIndent":pt,"spaceBefore":pt,"spaceAfter":pt,"autoLeading":%,"direction":"auto|ltr|rtl","hyphenate":bool}}"##)
                    .into_boxed_str(),
            ),
            enabled: has_doc,
            journal: true,
            run: |s, p| {
                check_kerning(p).map_err(|m| bad("type.setStyle", m))?;
                check_steps(p).map_err(|m| bad("type.setStyle", m))?;
                let ids = with_text_targets(s, p, "Set Type Style", |t, (a, b)| {
                    let props = shown_style_params(t, p)?;
                    check_size_tracking(&props).map_err(|m| bad("type.setStyle", m))?;
                    let mut probe = CharStyle::default();
                    if apply_char_props(&mut probe, &props) {
                        style_range(t, a, b, &|st| {
                            apply_char_props(st, &props);
                        });
                    }
                    step_char_metrics(t, a, b, &props);
                    let mut pprobe = ParagraphStyle::default();
                    if apply_para_props(&mut pprobe, p) {
                        style_paragraphs(t, a, b, &|st| {
                            apply_para_props(st, p);
                        });
                    }
                    Ok(())
                })?;
                let id = ids.first().ok_or_else(|| EngineError::Other("select a type layer".into()))?;
                info(s, &json!({ "layer": id.0 }))
            },
        },
        CommandSpec {
            id: "type.rasterize",
            label: "Rasterize Type",
            menu: &["Layer", "Rasterize"],
            shortcut: None,
            params: r##"{"layer":id?}"##,
            enabled: has_doc,
            journal: true,
            run: |s, p| {
                let id = layer_id(s, p)?;
                s.edit("Rasterize Type", |doc, _| {
                    let snapshot = doc.clone();
                    let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
                    let LayerContent::Text(t) = &mut l.content else {
                        return Err(EngineError::Other(format!("layer {} is not a type layer", id.0)));
                    };
                    if t.cache.is_none() {
                        refresh(&snapshot, t);
                    }
                    let surface = t.cache.take().unwrap_or_else(|| photocraft_raster::Surface::new(snapshot.pixel_format()));
                    let surface = if surface.format() == snapshot.pixel_format() { surface } else { surface.convert(snapshot.pixel_format()) };
                    l.content = LayerContent::Raster(surface);
                    l.psd_blocks.retain(|(k, _)| k != b"TySh");
                    Ok(())
                })?;
                Ok(json!({ "layer": id.0 }))
            },
        },
        CommandSpec {
            id: "type.info",
            label: "Type Layer Info",
            menu: &[],
            shortcut: None,
            params: r##"{"layer":id?} → text, runs/paragraphs (char offsets + styles), shape, transform, laid-out lines (text space px), bounds"##,
            enabled: has_doc,
            journal: false,
            run: |s, p| info(s, p),
        },
        CommandSpec {
            id: "type.fonts",
            label: "List Fonts",
            menu: &[],
            shortcut: None,
            params: r##"{"family":str? (faces of one family)}"##,
            enabled: |_| Ok(()),
            journal: false,
            run: |_, p| {
                let mut eng = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
                match p.get("family").and_then(Value::as_str) {
                    Some(f) => {
                        let faces: Vec<Value> = eng.fonts.faces(f).into_iter().map(|x| json!({ "family": x.family, "style": x.style, "postscriptName": x.postscript_name, "weight": x.weight, "italic": x.italic, "axes": x.axes })).collect();
                        Ok(json!({ "faces": faces }))
                    }
                    None => Ok(json!({ "families": eng.fonts.families() })),
                }
            },
        },
        size_key("type.increaseSize", "Increase Type Size", "Cmd+Shift+.", |s, p| step_size(s, p, true, 2.0)),
        size_key("type.decreaseSize", "Decrease Type Size", "Cmd+Shift+,", |s, p| step_size(s, p, false, 2.0)),
        size_key("type.increaseSizeMore", "Increase Type Size by 10", "Cmd+Alt+Shift+.", |s, p| step_size(s, p, true, 10.0)),
        size_key("type.decreaseSizeMore", "Decrease Type Size by 10", "Cmd+Alt+Shift+,", |s, p| step_size(s, p, false, 10.0)),
    ]
}

fn has_type_layer(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document open")?;
    match st.active_layer.and_then(|id| st.doc.layer(id)) {
        Some(l) if matches!(l.content, LayerContent::Text(_)) => Ok(()),
        _ => Err("the active layer is not a type layer".into()),
    }
}

/// Photoshop's ⇧⌘> / ⇧⌘< (⌥ for bigger steps). On a US keyboard > and < share the . and , keys.
fn size_key(id: &'static str, label: &'static str, shortcut: &'static str, run: fn(&mut Session, &Value) -> Result<Value>) -> CommandSpec {
    CommandSpec {
        id,
        label,
        menu: &[],
        shortcut: Some(shortcut),
        params: r##"{"layer":id?,"range":[startChar,endChar]? (default all),"by":pt? (as the layer appears)}"##,
        enabled: has_type_layer,
        journal: true,
        run,
    }
}

/// Grows or shrinks every character in the range by `by` points as the layer appears, so mixed
/// sizes keep their differences and a scaled layer steps in the points its size field shows.
fn step_size(s: &mut Session, p: &Value, up: bool, default_by: f32) -> Result<Value> {
    let id = layer_id(s, p)?;
    let by = f32p(p, "by").filter(|v| v.is_finite()).unwrap_or(default_by).abs().min(TYPE_SIZE_MAX_PT as f32);
    with_text_layer(s, p, "Change Type Size", |t, _, _| {
        let m = t.transform.m;
        let scale = (m[0] * m[3] - m[1] * m[2]).abs().sqrt() as f32;
        let scale = if scale.is_finite() && scale > 1e-3 { scale } else { 1.0 };
        let delta = if up { by } else { -by } / scale;
        let (a, b) = range_param(&t.text, p);
        style_range(t, a, b, &|st| st.size_pt = (st.size_pt + delta).clamp(TYPE_SIZE_MIN_PT as f32, TYPE_SIZE_MAX_PT as f32));
        Ok(())
    })?;
    info(s, &json!({ "layer": id.0 }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 200, "height": 100, "background": "transparent"})).unwrap();
        s
    }

    fn text_layer(s: &Session, id: u64) -> TextLayer {
        match &s.active().unwrap().doc.layer(LayerId(id)).unwrap().content {
            LayerContent::Text(t) => t.clone(),
            other => panic!("{}", other.kind_name()),
        }
    }

    #[test]
    fn new_text_inherits_the_host_overflow_default() {
        let mut s = session();
        for visible in [false, true] {
            s.type_overflow_visible = visible;
            let result = s.execute("type.create", json!({"text":"wrapped text", "box":[0,0,20,10]})).unwrap();
            let id = result["layer"].as_u64().unwrap();
            let text = text_layer(&s, id);
            assert_eq!(text.overflow_visible, visible);
            assert!(matches!(text.shape, TextShape::Box { width: 20.0, height: 10.0, .. }));
        }
    }

    #[test]
    fn named_face_metadata_replaces_old_postscript_identity() {
        let mut style = CharStyle { font_family: "Inter".into(), postscript_name: Some("Inter-Regular".into()), ..Default::default() };
        apply_char_props(&mut style, &json!({"fontStyle": "SemiBold"}));
        assert_eq!(style.weight, 600);
        assert_eq!(style.postscript_name.as_deref(), Some("Inter-SemiBold"));
        apply_char_props(&mut style, &json!({"weight": 400}));
        // #3556: a style change re-resolves to the installed face instead of leaving it nameless.
        assert_eq!(style.postscript_name.as_deref(), Some("Inter-Regular"));
        assert!(style.font_style.is_empty());
        apply_char_props(&mut style, &json!({"fontStyle": "unknown-style"}));
        assert_eq!(style.postscript_name.as_deref(), Some("Inter-Regular"));
        apply_char_props(&mut style, &json!({"fontStyle": "Regular", "postscriptName": "ExplicitFace"}));
        assert_eq!(style.postscript_name.as_deref(), Some("ExplicitFace"));
    }

    /// #3556: a font named by family stores the installed face's PostScript name, so PSD export
    /// names a real font instead of inventing "<Family>-Regular" that Photoshop calls missing.
    #[test]
    fn a_family_font_gets_the_installed_postscript_name() {
        let mut style = CharStyle { font_family: "Inter".into(), ..Default::default() };
        apply_char_props(&mut style, &json!({"font": "Inter"}));
        assert_eq!(style.font_family, "Inter");
        assert_eq!(style.postscript_name.as_deref(), Some("Inter-Regular"), "resolved from the font database");
        // A weight picks that face's name.
        apply_char_props(&mut style, &json!({"font": "Inter", "weight": 600}));
        assert_eq!(style.postscript_name.as_deref(), Some("Inter-SemiBold"));
        // A weight no face has (a variable font's axis, or between two faces) names no face, so
        // layout still matches the requested weight instead of a nearby face's.
        apply_char_props(&mut style, &json!({"weight": 650}));
        assert!(style.postscript_name.is_none(), "{:?}", style.postscript_name);
        // A family PhotoCraft does not have keeps no name; the export falls back as before.
        let mut absent = CharStyle { font_family: "No Such Family 123".into(), ..Default::default() };
        apply_char_props(&mut absent, &json!({"font": "No Such Family 123"}));
        assert!(absent.postscript_name.is_none());
    }

    /// Picking a style or weight resets the weight axis to it; other axes keep their values (#2985).
    #[test]
    fn a_style_or_weight_resets_the_weight_axis() {
        let mut style = CharStyle::default();
        apply_char_props(&mut style, &json!({"variations": {"wght": 650, "wdth": 80}}));
        apply_char_props(&mut style, &json!({"fontStyle": "Bold"}));
        let axes = |s: &CharStyle| s.variations.iter().map(|v| (v.axis.clone(), v.value)).collect::<Vec<_>>();
        assert_eq!(axes(&style), [("wdth".to_string(), 80.0)]);
        apply_char_props(&mut style, &json!({"variations": {"wght": 650}}));
        apply_char_props(&mut style, &json!({"weight": 300}));
        assert_eq!(axes(&style), [("wdth".to_string(), 80.0)]);
        apply_char_props(&mut style, &json!({"fontStyle": "Light", "variations": {"wght": 350}}));
        assert_eq!(axes(&style), [("wdth".to_string(), 80.0), ("wght".to_string(), 350.0)], "same-call values apply after the style");
    }

    #[test]
    fn absurd_type_size_and_tracking_are_rejected_as_bad_params() {
        let mut s = session();
        assert!(matches!(
            s.execute("type.create", json!({"text": "abc", "size": 1e30})),
            Err(EngineError::BadParams { cmd, .. }) if cmd == "type.create"
        ));

        let id = s.execute("type.create", json!({"text": "abc", "size": 12})).unwrap()["layer"].as_u64().unwrap();
        let before = text_layer(&s, id);
        for params in [json!({"layer": id, "tracking": 1e30}), json!({"layer": id, "runs": [{"start": 0, "end": 3, "tracking": 1e30}]})] {
            assert!(matches!(
                s.execute("type.edit", params),
                Err(EngineError::BadParams { cmd, .. }) if cmd == "type.edit"
            ));
            assert_eq!(text_layer(&s, id).runs, before.runs);
        }
        assert!(matches!(
            s.execute("type.setStyle", json!({"layer": id, "tracking": 1e30})),
            Err(EngineError::BadParams { cmd, .. }) if cmd == "type.setStyle"
        ));
        assert_eq!(text_layer(&s, id).runs, before.runs);
    }

    /// A font that arrives after the layer was drawn (served fonts on the web) re-renders it with
    /// no history step, and a clean document stays clean.
    #[test]
    fn refreshing_type_layers_takes_no_history_step() {
        let mut s = session();
        let id = LayerId(s.execute("type.create", json!({"text": "Hello", "size": 24})).unwrap()["layer"].as_u64().unwrap());
        let doc_id = s.active().unwrap().doc.id;
        // Stale pixels, and a document saved in this state.
        {
            let st = s.active_mut().unwrap();
            let Some(Layer { content: LayerContent::Text(t), .. }) = Arc::make_mut(&mut st.doc).layer_mut(id) else { panic!("not a type layer") };
            t.cache = None;
            st.saved_revision = st.revision;
        }
        let rev = s.active().unwrap().revision;
        let unknown = [(doc_id, LayerId(u64::MAX)), (photocraft_doc::DocId(u64::MAX), id)];
        assert!(s.refresh_type_layers(&[&[(doc_id, id)][..], &unknown].concat()).is_empty());
        assert!(text_layer(&s, id.0).cache.is_some());
        let st = s.active().unwrap();
        assert!(st.revision > rev);
        assert_eq!(st.saved_revision, st.revision, "still clean");
        // The only history step is still the creation.
        assert!(s.undo());
        assert!(s.active().unwrap().doc.layer(id).is_none());
    }

    #[test]
    fn create_renders_and_is_undoable() {
        let mut s = session();
        let r = s.execute("type.create", json!({"x": 10, "y": 50, "text": "Hello", "size": 24, "color": "#ff0000"})).unwrap();
        let id = r["layer"].as_u64().unwrap();
        let b = r["bounds"].as_array().unwrap();
        assert!(b[0].as_i64().unwrap() >= 9 && b[3].as_i64().unwrap() <= 56, "{b:?}");
        let t = text_layer(&s, id);
        assert_eq!(t.text, "Hello");
        assert_eq!(t.runs[0].style.size_pt, 24.0);
        assert!(t.cache.is_some() && t.psd_raw.is_some());
        let l = s.active().unwrap().doc.layer(LayerId(id)).unwrap();
        assert_eq!(l.name, "Hello");
        // Red pixels exist.
        let c = t.cache.as_ref().unwrap();
        let r = c.content_bounds();
        assert!(c.read_region(r).as_chunks::<4>().0.iter().any(|p| p[3] > 0.9 && p[0] > 0.9 && p[1] < 0.1));
        // The TySh we generated parses back to the same model.
        let back = photocraft_text::psd::text_layer_from_tysh(t.psd_raw.as_ref().unwrap(), 72.0).unwrap();
        assert_eq!(back.text, "Hello");
        let strip = |mut r: Vec<TextRun>| {
            for x in &mut r {
                x.style.postscript_name = None;
            }
            r
        };
        assert_eq!(strip(back.char_runs()), strip(t.char_runs()));
        assert_eq!(back.char_runs()[0].style.postscript_name.as_deref(), Some("Inter-Regular"));
        assert!(s.undo());
        assert!(s.active().unwrap().doc.layer(LayerId(id)).is_none());
    }

    fn layer_name_of(s: &Session, id: u64) -> String {
        s.active().unwrap().doc.layer(LayerId(id)).unwrap().name.clone()
    }

    #[test]
    fn auto_name_follows_the_text() {
        // The Type tool's flow (#483): a placeholder, selected, then replaced by typing.
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Lorem Ipsum", "coalesce": "k"})).unwrap()["layer"].as_u64().unwrap();
        assert_eq!(layer_name_of(&s, id), "Lorem Ipsum");
        s.execute("type.edit", json!({"layer": id, "replace": {"start": 0, "end": 11, "text": "H"}, "coalesce": "k"})).unwrap();
        s.execute("type.edit", json!({"layer": id, "replace": {"start": 1, "end": 1, "text": "eading\nsecond line"}, "coalesce": "k"})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Heading");
        // Emptied, the name is the default one and keeps following.
        s.execute("type.edit", json!({"layer": id, "text": ""})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Type Layer");
        s.execute("type.edit", json!({"layer": id, "text": "Title"})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Title");
        assert!(s.undo());
        assert_eq!(layer_name_of(&s, id), "Type Layer");
    }

    #[test]
    fn size_keys_step_each_run_and_respect_layer_scale() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Hello world", "size": 12})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.setStyle", json!({"layer": id, "range": [6, 11], "size": 20})).unwrap();
        s.execute("type.increaseSize", json!({})).unwrap();
        let sizes = |s: &Session| text_layer(s, id).char_runs().iter().map(|r| r.style.size_pt).collect::<Vec<_>>();
        assert_eq!(sizes(&s), vec![14.0, 22.0], "every run grows by 2 pt");
        s.execute("type.decreaseSizeMore", json!({"range": [0, 5]})).unwrap();
        assert_eq!(sizes(&s)[0], 4.0, "only the range, by 10 pt");
        for _ in 0..2 {
            s.execute("type.decreaseSize", json!({"range": [0, 5]})).unwrap();
        }
        assert_eq!(sizes(&s)[0], TYPE_SIZE_MIN_PT as f32, "never below the smallest size");
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(sizes(&s)[0], 2.0, "one undo step per press");

        let scaled = s.execute("type.create", json!({"x": 0, "y": 80, "text": "Big", "size": 12})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.edit", json!({"layer": scaled, "transform": [2.0, 0.0, 0.0, 2.0, 0.0, 80.0]})).unwrap();
        s.execute("type.increaseSize", json!({"layer": scaled})).unwrap();
        assert_eq!(text_layer(&s, scaled).char_runs()[0].style.size_pt, 13.0, "a 200% layer shows 24 pt and steps to 26 pt");
    }

    #[test]
    fn size_keys_need_a_type_layer() {
        let mut s = session();
        assert!(s.execute("type.increaseSize", json!({})).is_err());
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Hi", "size": 12})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.increaseSize", json!({"by": "big"})).unwrap();
        s.execute("type.increaseSize", json!({"by": -3})).unwrap();
        s.execute("type.decreaseSize", json!({"by": 1e30, "range": [9, 0]})).unwrap();
        assert_eq!(text_layer(&s, id).char_runs()[0].style.size_pt, TYPE_SIZE_MIN_PT as f32, "bad steps fall back or clamp, never panic");
        assert_eq!(crate::commands::find("type.increaseSize").and_then(|c| c.shortcut), Some("Cmd+Shift+."));
        assert_eq!(crate::commands::find("type.decreaseSize").and_then(|c| c.shortcut), Some("Cmd+Shift+,"));
    }

    #[test]
    fn auto_name_skips_leading_blank_lines() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "\n  \nHello\nworld"})).unwrap()["layer"].as_u64().unwrap();
        assert_eq!(layer_name_of(&s, id), "Hello");
        s.execute("type.edit", json!({"layer": id, "replace": {"start": 0, "end": 0, "text": "\n"}})).unwrap();
        s.execute("type.edit", json!({"layer": id, "text": "\n\nGoodbye"})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Goodbye");
    }

    #[test]
    fn custom_name_survives_text_edits() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Hello", "name": "Logo"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.edit", json!({"layer": id, "text": "Goodbye"})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Logo");
        // Renamed after creation: kept too.
        let id = s.execute("type.create", json!({"x": 0, "y": 80, "text": "Hello"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.renameLayer", json!({"layer": id, "name": "Caption"})).unwrap();
        s.execute("type.edit", json!({"layer": id, "text": "Goodbye"})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Caption");
        // A name given with the edit wins over the text.
        s.execute("type.edit", json!({"layer": id, "text": "Again", "name": "Footer"})).unwrap();
        assert_eq!(layer_name_of(&s, id), "Footer");
    }

    #[test]
    fn edit_with_name_is_one_undo_step_for_text_name_and_bounds() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 10, "y": 50, "text": "Before", "size": 20, "name": "Original"})).unwrap()["layer"].as_u64().unwrap();
        let layer = s.active().unwrap().doc.layer(LayerId(id)).unwrap();
        let before_bounds = text_layer(&s, id).cache.unwrap().content_bounds();
        assert_eq!(layer.name, "Original");
        let steps = s.active().unwrap().history.entries().len();

        s.execute("type.edit", json!({"layer": id, "text": "A much longer edited title that wraps", "box": [10, 10, 50, 60], "name": "Renamed"})).unwrap();

        let edited = s.active().unwrap().doc.layer(LayerId(id)).unwrap();
        let edited_bounds = text_layer(&s, id).cache.unwrap().content_bounds();
        assert_eq!(edited.name, "Renamed");
        assert_eq!(text_layer(&s, id).text, "A much longer edited title that wraps");
        assert_ne!(edited_bounds, before_bounds);
        assert_eq!(s.active().unwrap().history.entries().len(), steps + 1);

        assert!(s.undo());
        let restored = s.active().unwrap().doc.layer(LayerId(id)).unwrap();
        assert_eq!(restored.name, "Original");
        assert_eq!(text_layer(&s, id).text, "Before");
        assert_eq!(text_layer(&s, id).cache.unwrap().content_bounds(), before_bounds);
    }

    #[test]
    fn edit_replace_keeps_styles() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Hello world"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.setStyle", json!({"layer": id, "range": [6, 11], "fauxBold": true, "color": [0, 0, 1]})).unwrap();
        let t = text_layer(&s, id);
        assert_eq!(t.runs.len(), 2);
        assert_eq!(t.runs[0].len, 6);
        assert!(t.runs[1].style.faux_bold);
        // Insert inside the bold word: inherits bold.
        s.execute("type.edit", json!({"layer": id, "replace": {"start": 8, "end": 8, "text": "XX"}})).unwrap();
        let t = text_layer(&s, id);
        assert_eq!(t.text, "Hello woXXrld");
        assert_eq!(t.runs[1].len, 7);
        // Delete across the style boundary.
        s.execute("type.edit", json!({"layer": id, "replace": {"start": 3, "end": 8, "text": ""}})).unwrap();
        let t = text_layer(&s, id);
        assert_eq!(t.text, "HelXXrld");
        assert_eq!(t.runs.iter().map(|r| r.len).collect::<Vec<_>>(), vec![3, 5]);
        // Whole-text replace with a multi-byte string.
        let info = s.execute("type.edit", json!({"layer": id, "text": "Größe ✓\nzwei"})).unwrap();
        assert_eq!(info["lines"].as_array().unwrap().len(), 2);
        let t = text_layer(&s, id);
        assert_eq!(t.runs.iter().map(|r| r.len).sum::<usize>(), t.text.len());
    }

    #[test]
    fn all_locked_type_edit_rejects_text_replacement_without_history() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 20, "y": 20, "text": "a", "size": 12})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.setProps", json!({"layer": id, "locked": true})).unwrap();
        let history = s.active().unwrap().history.past_len();

        let result = s.execute("type.edit", json!({"layer": id, "text": "b"}));
        assert!(matches!(&result, Err(EngineError::Other(message)) if message.contains("locked")), "{result:?}");
        assert_eq!(text_layer(&s, id).text, "a");
        assert_eq!(s.active().unwrap().history.past_len(), history);

        s.execute("layer.setProps", json!({"layer": id, "locked": false})).unwrap();
        let history = s.active().unwrap().history.past_len();
        s.execute("type.edit", json!({"layer": id, "text": "b"})).unwrap();
        assert_eq!(text_layer(&s, id).text, "b");
        assert_eq!(s.active().unwrap().history.past_len(), history + 1);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(text_layer(&s, id).text, "a");
        assert_eq!(s.active().unwrap().history.past_len(), history);
    }

    #[test]
    fn trailing_empty_paragraph_styles_without_changing_previous_lines() {
        let mut s = session();
        let id = s.execute("type.create", json!({"text":"a\nb\n","x":20,"y":20,"size":12,"font":"Inter","align":"left"})).unwrap()["layer"].as_u64().unwrap();
        let before = s.execute("type.info", json!({"layer":id})).unwrap();
        let previous_line = before["lines"][1].clone();

        let styled = s.execute("type.setStyle", json!({"layer":id,"range":[4,4],"align":"right"})).unwrap();
        assert_eq!(styled["paragraphs"].as_array().unwrap().len(), 2);
        assert_eq!(styled["paragraphs"][0]["start"], 0);
        assert_eq!(styled["paragraphs"][0]["end"], 4);
        assert_eq!(styled["paragraphs"][0]["style"]["align"], "Left");
        assert_eq!(styled["paragraphs"][1]["start"], 4);
        assert_eq!(styled["paragraphs"][1]["end"], 4);
        assert_eq!(styled["paragraphs"][1]["style"]["align"], "Right");
        assert_eq!(styled["lines"][1], previous_line, "the preceding paragraph must not move");
        let raw = text_layer(&s, id).psd_raw.clone().unwrap();
        let round_trip = photocraft_text::psd::text_layer_from_tysh(&raw, 72.0).unwrap();
        assert_eq!(round_trip.paragraph_runs().last().unwrap().style.align, TextAlign::Right);

        // Typing directly into the previously empty final paragraph retains its style.
        let edited = s.execute("type.edit", json!({"layer":id,"replace":{"start":4,"end":4,"text":"c"}})).unwrap();
        assert_eq!(edited["text"], "a\nb\nc");
        assert_eq!(edited["paragraphs"][0]["end"], 4);
        assert_eq!(edited["paragraphs"][0]["style"]["align"], "Left");
        assert_eq!(edited["paragraphs"][1]["start"], 4);
        assert_eq!(edited["paragraphs"][1]["end"], 5);
        assert_eq!(edited["paragraphs"][1]["style"]["align"], "Right");
        assert_eq!(edited["lines"][1], previous_line, "typing in the final paragraph does not move b");
    }

    #[test]
    fn set_style_paragraph_and_box() {
        let mut s = session();
        let id = s.execute("type.create", json!({"box": [10, 10, 80, 80], "text": "one two three four five six seven", "size": 12})).unwrap()["layer"]
            .as_u64()
            .unwrap();
        let info = s.execute("type.setStyle", json!({"layer": id, "align": "center", "leading": 20})).unwrap();
        let lines = info["lines"].as_array().unwrap();
        assert!(lines.len() >= 3);
        let b0 = lines[0]["baseline"].as_f64().unwrap();
        let b1 = lines[1]["baseline"].as_f64().unwrap();
        assert!((b1 - b0 - 20.0).abs() < 1e-3, "explicit leading {b0} {b1}");
        assert_eq!(info["paragraphs"][0]["style"]["align"], "Center");
        let t = text_layer(&s, id);
        assert!(matches!(t.shape, TextShape::Box { width, .. } if width == 80.0));
        // Pixels stay inside the box horizontally.
        let r = t.cache.as_ref().unwrap().content_bounds();
        assert!(r.x0 >= 9 && r.x1 <= 91, "{r:?}");
        // Convert to point text and move.
        s.execute("type.edit", json!({"layer": id, "point": [5, 30]})).unwrap();
        s.execute("type.edit", json!({"layer": id, "move": [10, 0]})).unwrap();
        let t = text_layer(&s, id);
        assert_eq!(t.shape, TextShape::Point);
        assert_eq!(t.transform.m[4], 15.0);
    }

    #[test]
    fn rasterize_and_fonts() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 5, "y": 30, "text": "Raster"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.rasterize", json!({"layer": id})).unwrap();
        let l = s.active().unwrap().doc.layer(LayerId(id)).unwrap();
        assert!(matches!(l.content, LayerContent::Raster(_)));
        assert!(l.surface().unwrap().content_bounds().width() > 10);
        assert!(s.execute("type.edit", json!({"layer": id, "text": "x"})).is_err());
        let fonts = s.execute("type.fonts", json!({})).unwrap();
        assert!(fonts["families"].as_array().unwrap().iter().any(|f| f == "Inter"));
        let faces = s.execute("type.fonts", json!({"family": "Inter"})).unwrap();
        assert!(!faces["faces"].as_array().unwrap().is_empty());
    }

    /// #124: a type edit damages only the layer's old and new pixels (the canvas recomposites
    /// that, not the document), and a drag's coalesced steps are one history step.
    #[test]
    fn type_edits_damage_only_the_layer_and_coalesce() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 3000, "height": 2000})).unwrap();
        let id = s.execute("type.create", json!({"x": 100, "y": 100, "text": "Hi", "size": 20})).unwrap()["layer"].as_u64().unwrap();
        let steps = s.active().unwrap().history.entries().len();
        for k in 0..5 {
            s.execute("type.setStyle", json!({"layer": id, "size": 30 + k * 10, "coalesce": "drag"})).unwrap();
            let d = s.active().unwrap().last_damage.expect("damage rect");
            let r = text_layer(&s, id).cache.unwrap().content_bounds();
            assert!(d.intersect(&r) == r && d.width() <= 512 && d.height() <= 512, "{d:?} {r:?}");
        }
        assert_eq!(s.active().unwrap().history.entries().len(), steps + 1);
        assert!(s.undo());
        assert_eq!(text_layer(&s, id).runs[0].style.size_pt, 20.0);
    }

    #[test]
    fn edit_rejects_a_reversed_runs_range() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Hello world", "size": 20})).unwrap()["layer"].as_u64().unwrap();
        let before = text_layer(&s, id).char_runs();
        let steps = s.active().unwrap().history.entries().len();
        for runs in [
            json!([{"start": 5, "end": 2, "fauxBold": true}]),
            json!([{"start": 0, "end": 3, "fauxBold": true}, {"start": 8, "end": 7}]),
            json!([{"end": 0, "start": 1}]),
        ] {
            let e = s.execute("type.edit", json!({"layer": id, "runs": runs})).unwrap_err();
            assert!(matches!(e, EngineError::BadParams { .. }) && e.to_string().contains("is before `start`"), "{e}");
        }
        // Nothing changed and nothing was recorded, not even the valid run before the bad one.
        assert_eq!(text_layer(&s, id).char_runs(), before);
        assert_eq!(s.active().unwrap().history.entries().len(), steps);
        // The same range the right way round still styles it.
        s.execute("type.edit", json!({"layer": id, "runs": [{"start": 2, "end": 5, "fauxBold": true}]})).unwrap();
        let runs = text_layer(&s, id).char_runs();
        assert_eq!(runs.iter().map(|r| (r.len, r.style.faux_bold)).collect::<Vec<_>>(), [(2, false), (3, true), (6, false)]);
    }

    #[test]
    fn style_range_reversed_is_a_no_op() {
        let mut t = TextLayer::default();
        replace_text(&mut t, 0, 0, "Hello world");
        let before = t.char_runs();
        style_range(&mut t, 5, 2, &|s| s.size_pt = 40.0);
        assert_eq!(t.char_runs(), before);
        assert_eq!(t.runs.iter().map(|r| r.len).sum::<usize>(), t.text.len());
    }

    #[test]
    fn style_range_on_empty_text() {
        let mut t = TextLayer::default();
        style_range(&mut t, 0, 0, &|s| s.size_pt = 40.0);
        assert_eq!(t.runs[0].style.size_pt, 40.0);
        replace_text(&mut t, 0, 0, "abc");
        assert_eq!(t.runs.len(), 1);
        assert_eq!(t.runs[0].len, 3);
        assert_eq!(t.runs[0].style.size_pt, 40.0);
    }

    fn kerning_of(s: &Session, id: u64) -> Vec<(Kerning, f32)> {
        let t = text_layer(s, id);
        let mut out = Vec::new();
        let mut at = 0;
        for r in t.char_runs() {
            out.extend(std::iter::repeat_n((r.style.kerning, r.style.kern), t.text[at..at + r.len].chars().count()));
            at += r.len;
        }
        out
    }

    /// #206: `type.edit {"kerning"}` over a range takes a number (manual, 1/1000 em) or a mode.
    #[test]
    fn edit_sets_kerning_over_a_range() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 5, "y": 50, "text": "AVAT", "size": 30})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.edit", json!({"layer": id, "range": [0, 1], "kerning": 100})).unwrap();
        s.execute("type.edit", json!({"layer": id, "range": [2, 4], "kerning": "optical"})).unwrap();
        assert_eq!(kerning_of(&s, id), vec![(Kerning::Off, 100.0), (Kerning::Metrics, 0.0), (Kerning::Optical, 0.0), (Kerning::Optical, 0.0)]);
        s.execute("type.edit", json!({"layer": id, "kerning": "metrics"})).unwrap();
        assert!(kerning_of(&s, id).iter().all(|k| *k == (Kerning::Metrics, 0.0)));
        // Runs and setStyle take the same values; type.info reports them.
        s.execute("type.edit", json!({"layer": id, "runs": [{"start": 1, "end": 2, "kerning": -50}]})).unwrap();
        s.execute("type.setStyle", json!({"layer": id, "range": [3, 4], "kerning": "off"})).unwrap();
        assert_eq!(kerning_of(&s, id)[1], (Kerning::Off, -50.0));
        assert_eq!(kerning_of(&s, id)[3], (Kerning::Off, 0.0));
        let info = s.execute("type.info", json!({"layer": id})).unwrap();
        assert_eq!(info["runs"][1]["style"]["kern"], json!(-50.0));
    }

    /// Bad kerning params are errors, never panics, and leave the layer alone.
    #[test]
    fn kerning_params_fail_gracefully() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 5, "y": 50, "text": "AV\nT", "size": 30})).unwrap()["layer"].as_u64().unwrap();
        let before = text_layer(&s, id);
        for p in [
            json!({"kerning": "tight"}),
            json!({"kerning": 1e9}),
            json!({"kerning": -5000}),
            json!({"kerning": true}),
            json!({"kerning": [1]}),
            json!({"runs": [{"start": 0, "end": 1, "kerning": "x"}]}),
            json!({"kernPair": 3}),
            json!({"kernPair": {"by": 20}}),
            json!({"kernPair": {"at": 1}}),
            json!({"kernPair": {"at": 1, "by": "20"}}),
            json!({"kernPair": {"at": 1, "by": 1e12}}),
            json!({"kernPair": {"at": 0, "by": 20}}),
            json!({"kernPair": {"at": 4, "by": 20}}),
            json!({"kernPair": {"at": 99, "by": 20}}),
            json!({"kernPair": {"at": 2, "by": 20}}),
            json!({"kernPair": {"at": 3, "by": 20}}),
            json!({"kernPair": {"at": -1, "by": 20}}),
        ] {
            let mut p = p;
            p["layer"] = json!(id);
            assert!(s.execute("type.edit", p.clone()).is_err(), "{p}");
        }
        assert!(s.execute("type.setStyle", json!({"layer": id, "kerning": "tight"})).is_err());
        assert_eq!(text_layer(&s, id).runs, before.runs);
        // type.create rejects the same values and creates no layer (#994).
        let layers = s.active().unwrap().doc.layer_count();
        for k in [json!("tight"), json!(1e9), json!(-5000), json!(true), json!([1])] {
            let p = json!({"x": 0, "y": 10, "text": "AV", "kerning": k});
            assert!(s.execute("type.create", p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.active().unwrap().doc.layer_count(), layers);
        let id = s.execute("type.create", json!({"x": 0, "y": 10, "text": "AV", "kerning": "optical"})).unwrap()["layer"].as_u64().unwrap();
        assert!(kerning_of(&s, id).iter().all(|k| *k == (Kerning::Optical, 0.0)));
        let id = s.execute("type.create", json!({"x": 0, "y": 10, "text": "AV", "kerning": -50})).unwrap()["layer"].as_u64().unwrap();
        assert!(kerning_of(&s, id).iter().all(|k| *k == (Kerning::Off, -50.0)));
    }

    /// Cost of one Alt+←/→ press (`kernPair`: two measuring layouts + the edit's re-render) on
    /// a headline and on a 2000-character paragraph. Release timings for the dev log:
    /// `cargo test --release -p photocraft-engine kern_pair_cost -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn kern_pair_cost() {
        let para = "The quick brown fox jumps over the lazy dog. ".repeat(45);
        for (name, text, size, bx, kerning) in [
            ("headline", "AVATAR Wave", 72, false, "metrics"),
            ("paragraph", para.as_str(), 12, true, "metrics"),
            ("optical paragraph", para.as_str(), 12, true, "optical"),
        ] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 2000, "height": 1500})).unwrap();
            let mut p = json!({"x": 20, "y": 100, "text": text, "size": size, "font": "Inter", "kerning": kerning});
            if bx {
                p["box"] = json!([20, 20, 900, 1400]);
            }
            let id = s.execute("type.create", p).unwrap()["layer"].as_u64().unwrap();
            let n = 20;
            let t0 = std::time::Instant::now();
            for i in 0..n {
                s.execute("type.edit", json!({"layer": id, "kernPair": {"at": 1 + i % 3, "by": 20}})).unwrap();
            }
            let per = t0.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
            let t1 = std::time::Instant::now();
            for i in 0..n {
                s.execute("type.setStyle", json!({"layer": id, "range": [1 + i % 3, 2 + i % 3], "tracking": i})).unwrap();
            }
            let base = t1.elapsed().as_secs_f64() * 1000.0 / f64::from(n);
            eprintln!("{name} ({} chars): kernPair {per:.2} ms/press; a plain style edit {base:.2} ms", text.chars().count());
        }
    }

    /// Photoshop's Alt+←/→: the pair before the caret becomes manual, starting from the kerning
    /// it shows (the font's pair kerning for "AV"), ±20 per press, one history step each.
    #[test]
    fn kern_pair_steps_from_the_shown_kerning() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 5, "y": 50, "text": "AVA", "size": 40, "font": "Inter"})).unwrap()["layer"].as_u64().unwrap();
        let metric = {
            let t = text_layer(&s, id);
            photocraft_text::shared().lock().unwrap().pair_kerning(&t, 72.0, 0).unwrap()
        };
        assert!(metric < -10.0, "Inter kerns AV: {metric}");
        let steps = s.active().unwrap().history.entries().len();
        s.execute("type.edit", json!({"layer": id, "kernPair": {"at": 1, "by": 20}})).unwrap();
        let k0 = kerning_of(&s, id)[0];
        assert_eq!(k0, (Kerning::Off, metric.round() + 20.0));
        s.execute("type.edit", json!({"layer": id, "kernPair": {"at": 1, "by": 100}})).unwrap();
        assert_eq!(kerning_of(&s, id)[0].1, metric.round() + 120.0);
        s.execute("type.edit", json!({"layer": id, "kernPair": {"at": 1, "by": -20}})).unwrap();
        assert_eq!(kerning_of(&s, id)[0].1, metric.round() + 100.0);
        // Other characters are untouched; each press is its own undoable step.
        assert_eq!(&kerning_of(&s, id)[1..], &[(Kerning::Metrics, 0.0), (Kerning::Metrics, 0.0)]);
        assert_eq!(s.active().unwrap().history.entries().len(), steps + 3);
        assert!(s.undo());
        assert_eq!(kerning_of(&s, id)[0].1, metric.round() + 120.0);
        assert!(s.undo() && s.undo());
        assert_eq!(kerning_of(&s, id)[0], (Kerning::Metrics, 0.0));
    }
    #[test]
    fn create_vertical_type_is_atomic_and_invalid_orientation_is_rejected() {
        for boxed in [false, true] {
            let mut s = session();
            let before = s.active().unwrap().history.entries().len();
            for orientation in [json!("diagonal"), json!(null), json!(true)] {
                assert!(s.execute("type.create", json!({"text": "test", "orientation": orientation})).is_err());
                assert_eq!(s.active().unwrap().history.entries().len(), before);
            }
            let mut p = json!({"text": "test", "orientation": "vertical", "x": 20, "y": 20});
            if boxed {
                p["box"] = json!([20, 20, 80, 60]);
            }
            let id = s.execute("type.create", p).unwrap()["layer"].as_u64().unwrap();
            let t = text_layer(&s, id);
            assert_eq!(t.orientation, Orientation::Vertical);
            assert_eq!(matches!(t.shape, TextShape::Box { .. }), boxed);
            s.execute("edit.undo", json!({})).unwrap();
            assert!(s.active().unwrap().doc.layer(LayerId(id)).is_none());
        }
    }

    #[test]
    fn missing_font_emits_warning_in_create_set_style_and_edit() {
        let mut s = session();
        let r = s.execute("type.create", json!({"x": 10, "y": 20, "text": "Hello", "font": "NoSuchFamily123"})).unwrap();
        assert_eq!(r["warnings"], json!(["font 'NoSuchFamily123' is not installed; text may render using a fallback face"]));
        let id = r["layer"].as_u64().unwrap();
        let styled = s.execute("type.setStyle", json!({"layer": id, "font": "StillNoSuchFamily456"})).unwrap();
        assert_eq!(styled["warnings"], json!(["font 'StillNoSuchFamily456' is not installed; text may render using a fallback face"]));
        let edited = s.execute("type.edit", json!({"layer": id, "runs": [{"start": 0, "end": 2, "font": "AnotherMissingFont789"}]})).unwrap();
        assert_eq!(
            edited["warnings"],
            json!([
                "font 'AnotherMissingFont789' is not installed; text may render using a fallback face",
                "font 'StillNoSuchFamily456' is not installed; text may render using a fallback face"
            ])
        );
    }

    /// `leadingBy`, `baselineShiftBy` and `trackingBy` step every character from its own value
    /// (auto leading from its laid-out value), clamp to Photoshop's ranges, and reject hostile
    /// steps without changing anything.
    #[test]
    fn relative_steps_move_each_character_from_its_own_value() {
        let mut s = session();
        let id = s.execute("type.create", json!({"x": 0, "y": 40, "text": "Hello\nworld", "size": 10})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.setStyle", json!({"layer": id, "range": [8, 11], "leading": 30, "tracking": 50})).unwrap();
        s.execute("type.setStyle", json!({"layer": id, "range": [6, 11], "autoLeading": 150})).unwrap();
        let steps = s.active().unwrap().history.entries().len();
        s.execute("type.setStyle", json!({"layer": id, "range": [0, 11], "leadingBy": 2, "baselineShiftBy": -2, "trackingBy": 20})).unwrap();
        assert_eq!(s.active().unwrap().history.entries().len(), steps + 1, "one history step");
        let t = text_layer(&s, id);
        let at = |c: usize| {
            let byte = byte_at(&t.text, c);
            let mut off = 0;
            t.char_runs()
                .into_iter()
                .find(|r| {
                    off += r.len;
                    off > byte
                })
                .unwrap()
                .style
        };
        // Auto leading 120% × 10 pt (first paragraph), 150% × 10 pt (second), and a set 30 pt.
        assert_eq!(at(0).leading_pt, Some(14.0));
        assert_eq!(at(6).leading_pt, Some(17.0));
        assert_eq!(at(9).leading_pt, Some(32.0));
        assert_eq!((at(0).baseline_shift_pt, at(9).baseline_shift_pt), (-2.0, -2.0));
        assert_eq!((at(0).tracking, at(9).tracking), (20.0, 70.0));
        // Clamped to the ranges.
        s.execute("type.setStyle", json!({"layer": id, "leadingBy": -5000, "baselineShiftBy": 10000, "trackingBy": -11000})).unwrap();
        let t = text_layer(&s, id);
        assert!(t.char_runs().iter().all(|r| r.style.leading_pt == Some(LEADING_MIN_PT)
            && r.style.baseline_shift_pt == BASELINE_SHIFT_MAX_PT
            && r.style.tracking == TRACKING_MIN as f32));
        // A collapsed range in non-empty text changes no character.
        let before = text_layer(&s, id).char_runs();
        s.execute("type.setStyle", json!({"layer": id, "range": [3, 3], "trackingBy": 100})).unwrap();
        assert_eq!(text_layer(&s, id).char_runs(), before);
        // Hostile steps are refused.
        let steps = s.active().unwrap().history.entries().len();
        for bad_step in [json!({"leadingBy": "up"}), json!({"trackingBy": 1e30}), json!({"baselineShiftBy": null}), json!({"leadingBy": [1]})] {
            let mut p = bad_step.clone();
            p["layer"] = json!(id);
            assert!(matches!(s.execute("type.setStyle", p), Err(EngineError::BadParams { .. })), "{bad_step}");
        }
        assert_eq!(s.active().unwrap().history.entries().len(), steps);
        // Empty text: the step sets the style the next typed characters get.
        let empty = s.execute("type.create", json!({"x": 0, "y": 80, "text": "", "size": 10})).unwrap()["layer"].as_u64().unwrap();
        s.execute("type.setStyle", json!({"layer": empty, "range": [0, 0], "leadingBy": -2})).unwrap();
        assert_eq!(text_layer(&s, empty).char_runs()[0].style.leading_pt, Some(10.0));
    }
}
