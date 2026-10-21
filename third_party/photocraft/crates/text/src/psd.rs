//! PSD type layers: the `TySh` block (transform + text descriptor with `EngineData` + warp) ⇄ the
//! typed [`TextLayer`] model.
//!
//! Units: EngineData sizes are in text-space units, which the `TySh` transform maps to document
//! pixels; Photoshop shows them as points at the document resolution, so
//! `size_pt = FontSize × 72 / dpi` (and likewise leading, tracking-independent lengths, indents).
//! Box bounds stay in text-space pixels.
//!
//! Writing starts from the layer's original `TySh` (when present) and patches only the text, runs,
//! paragraphs, fonts, shape and transform, so Photoshop-only settings survive an edit.

use photocraft_color::{Color, ColorMode};
use photocraft_doc::TextLayer;
use photocraft_doc::text::{AntiAlias, Caps, CharStyle, Kerning, Orientation, ParagraphRun, ParagraphStyle, TextAlign, TextRun, TextShape, TextWarp};
use photocraft_geom::{Affine, Point};
use photocraft_psd::descriptor::{Descriptor, Id, UnicodeString, Value as D, VersionedDescriptor};

use crate::engine_data::{self as ed, Value as E};
use crate::fonts::guess_from_postscript;

/// Parsed `TySh` contents.
#[derive(Clone, Debug, PartialEq)]
pub struct TySh {
    pub transform: Affine,
    pub text: Descriptor,
    pub warp: Option<Descriptor>,
    /// The four trailing big-endian f32 values (left, top, right, bottom) after the warp
    /// descriptor. Photoshop writes zeros for unwarped type and the text-space bounds (the
    /// `bounds` descriptor's rectangle) for warped type.
    pub bounds: [f32; 4],
}

/// Parses a `TySh` block.
pub fn parse_tysh(data: &[u8]) -> Option<TySh> {
    let f = |at: usize| data.get(at..at + 8).map(|b| f64::from_be_bytes(b.try_into().unwrap_or([0; 8])));
    let mut m = [0.0; 6];
    for (i, v) in m.iter_mut().enumerate() {
        *v = f(2 + i * 8)?;
    }
    let (text, used) = VersionedDescriptor::parse_prefix(data.get(52..)?).ok()?;
    let rest = data.get(52 + used..).unwrap_or(&[]);
    let (warp, wused) = match rest.get(2..).and_then(|r| VersionedDescriptor::parse_prefix(r).ok()) {
        Some((w, n)) => (Some(w.descriptor), n + 2),
        None => (None, 0),
    };
    let tail = rest.get(wused..).unwrap_or(&[]);
    let mut bounds = [0f32; 4];
    for (i, b) in bounds.iter_mut().enumerate() {
        if let Some(x) = tail.get(i * 4..i * 4 + 4) {
            *b = f32::from_be_bytes(x.try_into().unwrap_or([0; 4]));
        }
    }
    Some(TySh { transform: Affine { m }, text: text.descriptor, warp, bounds })
}

/// `tysh` moved by (`dx`, `dy`): only the transform's translation changes. The text, its styles,
/// the warp and the trailing bounds (which are in text space) stay byte for byte, so a moved
/// Photoshop type layer keeps everything Photoshop wrote. `None` when the block is too short.
pub fn translate_tysh(tysh: &[u8], dx: f64, dy: f64) -> Option<Vec<u8>> {
    let mut out = tysh.to_vec();
    // Version (2 bytes), then xx, xy, yx, yy, tx, ty as big-endian f64: tx at 34, ty at 42.
    for (at, d) in [(34, dx), (42, dy)] {
        let b = out.get_mut(at..at + 8)?;
        let v = f64::from_be_bytes(b.try_into().ok()?) + d;
        b.copy_from_slice(&v.to_be_bytes());
    }
    Some(out)
}

/// Serializes a `TySh` block.
pub fn write_tysh(t: &TySh) -> Vec<u8> {
    let mut v = 1u16.to_be_bytes().to_vec();
    for x in t.transform.m {
        v.extend_from_slice(&x.to_be_bytes());
    }
    v.extend_from_slice(&50u16.to_be_bytes());
    v.extend(VersionedDescriptor::new(t.text.clone()).to_bytes());
    v.extend_from_slice(&1u16.to_be_bytes());
    v.extend(VersionedDescriptor::new(t.warp.clone().unwrap_or_else(default_warp)).to_bytes());
    for b in t.bounds {
        v.extend_from_slice(&b.to_be_bytes());
    }
    v
}

fn default_warp() -> Descriptor {
    Descriptor::new("warp")
        .with("warpStyle", enumv("warpStyle", "warpNone"))
        .with("warpValue", D::Double(0.0))
        .with("warpPerspective", D::Double(0.0))
        .with("warpPerspectiveOther", D::Double(0.0))
        .with("warpRotate", enumv("Ornt", "Hrzn"))
}

fn enumv(t: &str, v: &str) -> D {
    D::Enumerated { type_id: Id::new(t), value: Id::new(v) }
}

fn enum_value(d: &Descriptor, key: &str) -> Option<String> {
    match d.get(key) {
        Some(D::Enumerated { value, .. }) => Some(String::from_utf8_lossy(value.as_bytes()).into_owned()),
        _ => None,
    }
}

fn dnum(v: Option<&D>) -> Option<f64> {
    match v {
        Some(D::Double(x)) => Some(*x),
        Some(D::UnitFloat { value, .. }) => Some(*value),
        Some(D::Integer(i)) => Some(f64::from(*i)),
        _ => None,
    }
}

/// The EngineData of a text descriptor.
pub fn engine_data(text: &Descriptor) -> Option<E> {
    match text.get("EngineData") {
        Some(D::RawData(b)) => ed::parse(b).ok(),
        _ => None,
    }
}

/// Whether the text engine of a `TySh` block has nothing to draw: its EngineData text
/// (`EngineDict/Editor/Text`, what Photoshop lays out) is present and only whitespace. Some
/// Photoshop layers carry their characters only in the descriptor's `Txt ` and render nothing.
pub fn engine_text_is_blank(tysh: &[u8]) -> bool {
    parse_tysh(tysh)
        .and_then(|t| engine_data(&t.text))
        .and_then(|e| e.path(&["EngineDict", "Editor", "Text"]).and_then(E::as_str).map(|s| s.trim().is_empty()))
        .unwrap_or(false)
}

/// Parses the document's `Txt2` block (Photoshop's text engine data for all type layers).
pub fn parse_txt2(data: &[u8]) -> Option<E> {
    // Txt2 is a bare sequence of `/key value` pairs (no enclosing `<< >>`).
    let mut wrapped = Vec::with_capacity(data.len() + 4);
    wrapped.extend_from_slice(b"<<");
    wrapped.extend_from_slice(data);
    wrapped.extend_from_slice(b">>");
    ed::parse(&wrapped).ok().filter(|v| matches!(v, E::Dict(_)))
}

/// Applies what Photoshop keeps only in `Txt2` to `layer`, read from `tysh`: the automatic
/// kerning modes (EngineData writes Optical and "0" alike as `AutoKerning true`) and the
/// OpenType settings besides (discretionary) ligatures: stylistic sets, contextual alternates,
/// swashes and the rest ([`TXT2_FEATURES`]), which Photoshop leaves out of EngineData. Only when
/// the layer's `TextIndex` entry still holds the layer's text (an edit elsewhere leaves `Txt2`
/// stale).
///
/// Observed layout: `Txt2 /1` is the list of text objects; an object's `/0` is its text model
/// with `/0` the text (`\r` breaks, trailing `\r`) and `/6 /0` its style runs, each
/// `<< /0 << /0 << … /6 <<style>> >> >> /1 utf16-length >>`; style key `/11` is the auto-kern
/// mode of the pair after each character: 0 manual, 1 metrics, 2 optical.
pub fn apply_txt2(layer: &mut TextLayer, tysh: &[u8], txt2: &E) {
    let Some(index) = parse_tysh(tysh).and_then(|t| match t.text.get("TextIndex") {
        Some(D::Integer(i)) => usize::try_from(*i).ok(),
        _ => None,
    }) else {
        return;
    };
    let Some(model) = txt2.path(&["1", "1"]).and_then(E::as_array).and_then(|a| a.get(index)).and_then(|o| o.get("0")) else {
        return;
    };
    let Some(text) = model.get("0").and_then(E::as_str) else {
        return;
    };
    if text.strip_suffix('\r').unwrap_or(text).replace('\r', "\n") != layer.text {
        return;
    }
    let Some(style_runs) = model.path(&["6", "0"]).and_then(E::as_array) else {
        return;
    };
    let styles: Vec<(f64, Option<&E>)> = style_runs.iter().map(|r| (r.get("1").and_then(E::as_f64).unwrap_or(0.0), r.path(&["0", "0", "6"]))).collect();
    let lens = utf16_to_byte_lengths(text, &styles.iter().map(|m| m.0).collect::<Vec<_>>());
    // Txt2 style per byte segment, then split the model runs at segment boundaries.
    let mut segs: Vec<(usize, Option<&E>)> = lens.into_iter().zip(styles.iter().map(|m| m.1)).collect();
    segs.retain(|s| s.0 > 0);
    let runs = layer.char_runs();
    let mut out: Vec<TextRun> = Vec::new();
    let (mut si, mut seg_end) = (0usize, segs.first().map_or(usize::MAX, |s| s.0));
    let mut at = 0usize;
    for r in runs {
        let end = at + r.len;
        while at < end {
            while at >= seg_end && si + 1 < segs.len() {
                si += 1;
                seg_end = seg_end.saturating_add(segs.get(si).map_or(0, |s| s.0));
            }
            let piece_end = if at < seg_end { end.min(seg_end) } else { end };
            let mut style = r.style.clone();
            let txt2_style = segs.get(si).and_then(|s| s.1).filter(|_| at < seg_end);
            // Txt2 has the exact mode (EngineData's reading also marks neighbours of manual
            // kerns manual, like Photoshop's legacy reader).
            match txt2_style.and_then(|st| st.get("11")).and_then(E::as_i64) {
                Some(0) => style.kerning = Kerning::Off,
                Some(1) => style.kerning = Kerning::Metrics,
                Some(2) => style.kerning = Kerning::Optical,
                _ => {}
            }
            if let Some(st) = txt2_style {
                add_txt2_features(&mut style, st);
            }
            let len = piece_end - at;
            match out.last_mut() {
                Some(last) if last.style == style => last.len += len,
                _ => out.push(TextRun { len, style }),
            }
            at = piece_end;
        }
    }
    layer.runs = out;
    layer.sync_summary();
}

/// The frame (`Txt2 /0 /8 /0` entry) of text object `index`: its rendered frame's `/10` names it.
fn txt2_frame(txt2: &E, index: usize) -> Option<&E> {
    let object = txt2.path(&["1", "1"]).and_then(E::as_array)?.get(index)?;
    let frame =
        object.path(&["1", "2"]).and_then(E::as_array)?.first()?.get("6").and_then(E::as_array)?.iter().find_map(|f| f.get("10").and_then(E::as_i64))?;
    txt2.path(&["0", "8", "0"]).and_then(E::as_array)?.get(usize::try_from(frame).ok()?)?.get("0")
}

/// Places `layer` where Photoshop draws it when the text object's frame in `Txt2` is moved
/// (#1317). Recent Photoshop keeps a type layer's position partly in its `Txt2` frame: a matrix
/// (`/2 /2`) from the frame's space to the `TySh` text space and, for point text, the anchor
/// (`/0`), while EngineData's `PointBase` stays at the origin and `BoxBounds` stays in the
/// frame's space. Ignoring them put the text far from Photoshop's pixels (off the canvas, so
/// it "disappeared" as soon as an edit started), and point text a line too high and off its
/// alignment anchor. Only when the layer's `TextIndex` entry still holds the layer's text, as
/// for [`apply_txt2`], and the frame agrees with the `TySh` `bounds`.
pub fn apply_txt2_frame(layer: &mut TextLayer, tysh: &[u8], txt2: &E) {
    let Some(t) = parse_tysh(tysh) else { return };
    let Some(index) = (match t.text.get("TextIndex") {
        Some(D::Integer(i)) => usize::try_from(*i).ok(),
        _ => None,
    }) else {
        return;
    };
    let text = txt2.path(&["1", "1"]).and_then(E::as_array).and_then(|a| a.get(index)).and_then(|o| o.path(&["0", "0"])).and_then(E::as_str);
    if text.is_none_or(|t| t.strip_suffix('\r').unwrap_or(t).replace('\r', "\n") != layer.text) {
        return;
    }
    let Some(frame) = txt2_frame(txt2, index) else { return };
    let m = arr_f(frame.path(&["2", "2"]));
    let matrix = if m.len() == 6 && m.iter().all(|v| v.is_finite()) { Affine { m: [m[0], m[1], m[2], m[3], m[4], m[5]] } } else { Affine::IDENTITY };
    let at = match layer.shape {
        TextShape::Point | TextShape::Path { .. } => {
            let p = arr_f(frame.get("0"));
            if p.len() == 2 && p.iter().all(|v| v.is_finite()) { Affine::translate(p[0], p[1]) } else { Affine::IDENTITY }
        }
        TextShape::Box { .. } => Affine::IDENTITY,
    };
    let to_text = matrix.mul(&at);
    if to_text == Affine::IDENTITY || to_text.determinant().abs() <= 1e-9 {
        return;
    }
    // The frame's origin (point text: the anchor; box text: the box's top left) must land in
    // the text-space `bounds` Photoshop wrote beside it, and not already be there without it. A `TySh` PhotoCraft rewrote (an edited
    // or moved layer whose text is unchanged keeps its old `Txt2` object and frame) already has
    // the frame folded into its transform and its `bounds` near the origin: applying the frame
    // again would move it twice.
    let origin = match layer.shape {
        TextShape::Point | TextShape::Path { .. } => Point::new(0.0, 0.0),
        TextShape::Box { .. } => {
            let b = engine_data(&t.text)
                .map(|e| {
                    arr_f(
                        e.path(&["EngineDict", "Rendered", "Shapes", "Children"])
                            .and_then(E::as_array)
                            .and_then(|c| c.first())
                            .and_then(|s| s.path(&["Cookie", "Photoshop", "BoxBounds"])),
                    )
                })
                .unwrap_or_default();
            let [x, y, ..] = b[..] else { return };
            Point::new(x, y)
        }
    };
    let Some([l, top, r, bottom]) = descriptor_rect(&t.text, "bounds") else { return };
    let inside = |p: Point| p.x >= l - 1.0 && p.x <= r + 1.0 && p.y >= top - 1.0 && p.y <= bottom + 1.0;
    if inside(origin) || !inside(to_text.apply(origin)) {
        return;
    }
    {
        // From the file's own transform: the frame's anchor is exact, so it replaces the
        // ink-box estimate `legacy_point_origin` made for such point text (#1469), which put
        // the baseline at the ink's top and the anchor at the ink's left whatever the alignment.
        layer.transform = t.transform.mul(&to_text);
    }
}

/// OpenType settings of a `Txt2` style that are on/off flags, by style key, as Photoshop
/// writes them (each turned on alone in the Character panel's OpenType settings and read back):
/// `/20` Contextual Alternates, `/23` Fractions, `/24` Ordinals, `/25` Swash, `/26` Titling
/// Alternates, `/28` Stylistic Alternates, `/29` Ornaments. (`/18` and `/19` are the standard and
/// discretionary ligatures, which EngineData carries too; `/27` is Arabic connection forms.)
pub const TXT2_FEATURES: [(&str, &str); 7] = [("20", "calt"), ("23", "frac"), ("24", "ordn"), ("25", "swsh"), ("26", "titl"), ("28", "salt"), ("29", "ornm")];

/// The `Txt2` style key holding the Stylistic Sets as a bit mask: bit n is set n + 1
/// (`ss01`..`ss20`).
pub const TXT2_STYLISTIC_SETS: &str = "85";

/// Sets in `style` the OpenType features a `Txt2` style dictionary turns on ([`TXT2_FEATURES`]
/// and the Stylistic Sets mask). A feature the style already sets keeps its value. A flag that is
/// off changes nothing, except Contextual Alternates: shapers apply `calt` by default, Photoshop
/// only when it is on (its default is off), so off is written as `calt` 0.
fn add_txt2_features(style: &mut CharStyle, txt2_style: &E) {
    let mut set = |tag: String, value: u16| {
        if !style.features.iter().any(|f| f.tag == tag) {
            style.features.push(photocraft_doc::text::FontFeature { tag, value });
        }
    };
    for (key, tag) in TXT2_FEATURES {
        match txt2_style.get(key).and_then(E::as_bool) {
            Some(true) => set(tag.to_string(), 1),
            Some(false) if tag == "calt" => set(tag.to_string(), 0),
            _ => {}
        }
    }
    // A mask beyond the 20 sets is not Photoshop's: ignore it.
    let sets = txt2_style.get(TXT2_STYLISTIC_SETS).and_then(E::as_i64).filter(|m| (0..1 << 20).contains(m)).unwrap_or(0);
    for n in 0..20 {
        if sets & (1 << n) != 0 {
            set(format!("ss{:02}", n + 1), 1);
        }
    }
}

/// Builds a [`TextLayer`] (model, text and transform; no cache, no `psd_raw`) from `TySh` data.
pub fn text_layer_from_tysh(data: &[u8], dpi: f32) -> Option<TextLayer> {
    let t = parse_tysh(data)?;
    let mut layer = TextLayer { transform: t.transform, ..Default::default() };
    let txt = match t.text.get("Txt ") {
        Some(D::Text(s)) => Some(s.to_string_lossy().trim_end_matches('\0').to_string()),
        _ => None,
    };
    layer.orientation = if enum_value(&t.text, "Ornt").as_deref() == Some("Vrtc") { Orientation::Vertical } else { Orientation::Horizontal };
    layer.antialias = match enum_value(&t.text, "AntA").as_deref() {
        Some("Anno" | "antiAliasNone") => AntiAlias::None,
        Some("antiAliasSharp" | "AnSh") => AntiAlias::Sharp,
        Some("AnCr" | "antiAliasCrisp") => AntiAlias::Crisp,
        Some("AnSt" | "antiAliasStrong") => AntiAlias::Strong,
        Some("antiAliasPlatformGray") => AntiAlias::Windows,
        Some("antiAliasPlatformLCD") => AntiAlias::WindowsLcd,
        Some(_) => AntiAlias::Smooth,
        // No `AntA`: the engine data's mode says it.
        None => {
            engine_data(&t.text).and_then(|e| e.path(&["EngineDict", "AntiAlias"]).and_then(E::as_f64)).and_then(engine_antialias).unwrap_or(AntiAlias::Smooth)
        }
    };
    if let Some(w) = &t.warp {
        let style = enum_value(w, "warpStyle").unwrap_or_else(|| "warpNone".into());
        if style != "warpNone" {
            layer.warp = Some(TextWarp {
                style,
                value: dnum(w.get("warpValue")).unwrap_or(0.0) as f32,
                horizontal_distortion: dnum(w.get("warpPerspective")).unwrap_or(0.0) as f32,
                vertical_distortion: dnum(w.get("warpPerspectiveOther")).unwrap_or(0.0) as f32,
                horizontal: enum_value(w, "warpRotate").as_deref() != Some("Vrtc"),
            });
        }
    }
    let k = 72.0 / if dpi > 0.0 { dpi } else { 72.0 };
    if let Some(e) = engine_data(&t.text) {
        // Without an `Ornt` key, the engine data's writing direction says it (2 = vertical).
        if enum_value(&t.text, "Ornt").is_none() && e.path(&["EngineDict", "Rendered", "Shapes", "WritingDirection"]).and_then(E::as_f64) == Some(2.0) {
            layer.orientation = Orientation::Vertical;
        }
        apply_engine_data(&mut layer, &e, txt.as_deref(), k);
    } else {
        layer.text = txt.unwrap_or_default().replace('\r', "\n");
    }
    // Some legacy PSDs store point text at a large local origin even though PointBase is zero.
    // Their transform maps that origin onto the canvas, but maps our zero-based layout far away.
    // Move the imported layout to the descriptor's ink origin so editing, engine commands, and
    // writing the TySh all agree on the visible position (#1469).
    if let Some([x, y]) = legacy_point_origin(&t) {
        let [a, b, c, d, e, f] = layer.transform.m;
        layer.transform.m = [a, b, c, d, a * x + c * y + e, b * x + d * y + f];
    }
    layer.sync_summary();
    Some(layer)
}

/// The local origin to fold into a legacy point-text transform, if present.
///
/// Photoshop normally keeps point text close to its PointBase. The affected legacy files instead
/// use a zero PointBase plus a very large `boundingBox` origin, with an equally remote transform
/// translation that cancels it out. Do not apply this to ordinary point text: its transform origin
/// is already the layout origin, and shifting it by the ink bounds would move the layer.
fn legacy_point_origin(t: &TySh) -> Option<[f64; 2]> {
    let e = engine_data(&t.text)?;
    let point_base = e
        .path(&["EngineDict", "Rendered", "Shapes", "Children"])
        .and_then(E::as_array)
        .and_then(|children| children.first())
        .and_then(|shape| shape.path(&["Cookie", "Photoshop", "PointBase"]))
        .and_then(E::as_array)?;
    let [px, py] = point_base else { return None };
    if px.as_f64() != Some(0.0) || py.as_f64() != Some(0.0) {
        return None;
    }
    let bounds = descriptor_rect(&t.text, "boundingBox")?;
    let [a, b, c, d, e, f] = t.transform.m;
    let values = [a, b, c, d, e, f, bounds[0], bounds[1]];
    if !values.iter().all(|v| v.is_finite()) {
        return None;
    }
    let shifted = [a * bounds[0] + c * bounds[1] + e, b * bounds[0] + d * bounds[1] + f];
    let origin_distance = e.abs().max(f.abs());
    let shifted_distance = shifted[0].abs().max(shifted[1].abs());
    // A local bounding-box origin in the thousands and a transform origin beyond a normal canvas
    // are the signature of this old writer. Requiring that the bounding-box origin is closer to
    // the document prevents applying the correction again after export/import.
    (bounds[0].abs().max(bounds[1].abs()) > 1024.0 && origin_distance > 4096.0 && shifted_distance < origin_distance).then_some([bounds[0], bounds[1]])
}

fn descriptor_rect(text: &Descriptor, key: &str) -> Option<[f64; 4]> {
    let D::Descriptor(rect) = text.get(key)? else { return None };
    let bounds = [dnum(rect.get("Left"))?, dnum(rect.get("Top "))?, dnum(rect.get("Rght"))?, dnum(rect.get("Btom"))?];
    (bounds.iter().all(|v| v.is_finite()) && bounds[2] > bounds[0] && bounds[3] > bounds[1]).then_some(bounds)
}

pub(crate) fn arr_f(v: Option<&E>) -> Vec<f64> {
    v.and_then(E::as_array).map(|a| a.iter().filter_map(E::as_f64).collect()).unwrap_or_default()
}

/// Converts UTF-16 run lengths over `text` into UTF-8 byte lengths.
pub(crate) fn utf16_to_byte_lengths(text: &str, lens: &[f64]) -> Vec<usize> {
    let mut out = Vec::with_capacity(lens.len());
    let mut chars = text.chars().peekable();
    for &l in lens {
        let mut units = l.max(0.0) as usize;
        let mut bytes = 0;
        while units > 0 {
            match chars.next() {
                Some(c) => {
                    units = units.saturating_sub(c.len_utf16());
                    bytes += c.len_utf8();
                }
                None => break,
            }
        }
        out.push(bytes);
    }
    out
}

fn apply_engine_data(layer: &mut TextLayer, e: &E, txt: Option<&str>, k: f32) {
    let ed_text = e.path(&["EngineDict", "Editor", "Text"]).and_then(E::as_str).unwrap_or("").to_string();
    // Photoshop always ends the engine text with a paragraph break; the descriptor text doesn't.
    let text = txt.map(str::to_string).unwrap_or_else(|| ed_text.strip_suffix('\r').unwrap_or(&ed_text).to_string());
    layer.text = text.replace('\r', "\n");
    let res = e.get("ResourceDict").or_else(|| e.get("DocumentResources"));
    let fonts: Vec<String> = res
        .and_then(|r| r.get("FontSet"))
        .and_then(E::as_array)
        .map(|a| a.iter().map(|f| f.get("Name").and_then(E::as_str).unwrap_or("").to_string()).collect())
        .unwrap_or_default();
    // Style resolution (#3795). A run's own `StyleSheetData` / `Properties` hold only what it
    // overrides; the rest comes from its sheets, as in Photoshop:
    // - a paragraph run: its `Properties` → the `ParagraphSheetSet` entry its `Name` names (a
    //   paragraph style; or its `Parent` index) → `TheNormalParagraphSheet`;
    // - a character run: its `StyleSheetData` → the `StyleSheetSet` entry its `Parent` or `Name`
    //   names (a character style) → the default character style of its paragraph (the paragraph run's
    //   `DefaultStyleSheet`, else its paragraph style's) → `TheNormalStyleSheet`.
    // Missing or out-of-range indices fall through to the next level, then to the defaults.
    let style_set: &[E] = res.and_then(|r| r.get("StyleSheetSet")).and_then(E::as_array).unwrap_or(&[]);
    let para_set: &[E] = res.and_then(|r| r.get("ParagraphSheetSet")).and_then(E::as_array).unwrap_or(&[]);
    let index = |key: &str| res.and_then(|r| r.get(key)).and_then(E::as_i64).and_then(|i| usize::try_from(i).ok());
    let normal_style = index("TheNormalStyleSheet").and_then(|i| style_set.get(i)).or(style_set.first());
    let normal_para = index("TheNormalParagraphSheet").and_then(|i| para_set.get(i)).or(para_set.first());
    // A run names its sheet by a `Parent` index or by `Name` (see `psd_styles::reference`).
    let named = |set: &'_ [E], sheet: &E| -> Option<usize> {
        if let Some(i) = sheet.get("Parent").and_then(E::as_i64).and_then(|i| usize::try_from(i).ok()).filter(|i| *i < set.len()) {
            return Some(i);
        }
        let name = sheet.get("Name").and_then(E::as_str)?;
        set.iter().position(|s| s.get("Name").and_then(E::as_str) == Some(name))
    };

    // Paragraph runs first: a character run's defaults depend on its paragraph.
    let prun = e.path(&["EngineDict", "ParagraphRun"]);
    let psheets: Vec<&E> =
        prun.and_then(|r| r.get("RunArray")).and_then(E::as_array).map(|a| a.iter().filter_map(|x| x.get("ParagraphSheet")).collect()).unwrap_or_default();
    let plens16 = arr_f(prun.and_then(|r| r.get("RunLengthArray")));
    let plens = utf16_to_byte_lengths(&ed_text, &plens16);
    let empty = E::Dict(Vec::new());
    // Per paragraph run: its merged base properties and its default character style.
    let para_bases: Vec<(E, Option<&E>)> = psheets
        .iter()
        .map(|sheet| {
            let style = named(para_set, sheet).and_then(|i| para_set.get(i));
            let base = merged(&[style.and_then(|s| s.get("Properties")), normal_para.and_then(|s| s.get("Properties"))]);
            let char_default = [Some(*sheet), style]
                .into_iter()
                .flatten()
                .find_map(|s| s.get("DefaultStyleSheet").and_then(E::as_i64).and_then(|i| usize::try_from(i).ok()).and_then(|i| style_set.get(i)));
            (base, char_default)
        })
        .collect();
    layer.paragraphs = psheets
        .iter()
        .zip(&para_bases)
        .zip(plens)
        .map(|((sheet, (base, _)), len)| ParagraphRun { len, style: para_style(Some(base), sheet.get("Properties").unwrap_or(&empty), k) })
        .collect();

    // Character runs.
    let srun = e.path(&["EngineDict", "StyleRun"]);
    let ssheets: Vec<&E> =
        srun.and_then(|r| r.get("RunArray")).and_then(E::as_array).map(|a| a.iter().filter_map(|x| x.get("StyleSheet")).collect()).unwrap_or_default();
    let slens16 = arr_f(srun.and_then(|r| r.get("RunLengthArray")));
    let slens = utf16_to_byte_lengths(&ed_text, &slens16);
    // The paragraph run each character run starts in (UTF-16 offsets).
    let para_of = |at: f64| -> Option<usize> {
        let mut end = 0.0;
        for (i, n) in plens16.iter().enumerate() {
            end += n;
            if at < end {
                return Some(i);
            }
        }
        plens16.len().checked_sub(1)
    };
    let mut at16 = 0.0;
    let bases: Vec<E> = ssheets
        .iter()
        .enumerate()
        .map(|(i, sheet)| {
            let own = named(style_set, sheet).and_then(|i| style_set.get(i));
            let para_default = para_of(at16).and_then(|p| para_bases.get(p)).and_then(|(_, c)| *c);
            at16 += slens16.get(i).copied().unwrap_or(0.0);
            merged(&[own, para_default, normal_style].map(|s| s.and_then(|s| s.get("StyleSheetData"))))
        })
        .collect();
    let sdata: Vec<&E> = ssheets.iter().map(|s| s.get("StyleSheetData").unwrap_or(&empty)).collect();
    let incoming: Vec<(usize, bool, f32)> = sdata
        .iter()
        .zip(&bases)
        .zip(&slens)
        .map(|((d, base), len)| {
            let auto = lookup(d, Some(base), "AutoKerning").and_then(E::as_bool) != Some(false);
            let kern = lookup(d, Some(base), "Kerning").and_then(E::as_f64).filter(|v| v.is_finite()).unwrap_or(0.0) as f32;
            (*len, auto, kern)
        })
        .collect();
    layer.runs = sdata.iter().zip(&bases).zip(slens).map(|((d, base), len)| TextRun { len, style: char_style(Some(base), d, &fonts, k) }).collect();
    layer.runs = kerning_from_pairs(&layer.text, layer.char_runs(), &incoming);

    // Shape.
    let shape = e.path(&["EngineDict", "Rendered", "Shapes", "Children"]).and_then(E::as_array).and_then(|a| a.first());
    if let Some(sh) = shape
        && sh.get("ShapeType").and_then(E::as_i64) == Some(1)
    {
        let b = arr_f(sh.path(&["Cookie", "Photoshop", "BoxBounds"]));
        if b.len() == 4 {
            layer.shape = TextShape::Box { x: b[0] as f32, y: b[1] as f32, width: (b[2] - b[0]) as f32, height: (b[3] - b[1]) as f32 };
        }
    }
}

/// The dictionaries in `levels` as one: a key comes from the first level that has it.
fn merged(levels: &[Option<&E>]) -> E {
    let mut out: Vec<(String, E)> = Vec::new();
    for level in levels.iter().flatten() {
        if let E::Dict(kv) = level {
            for (k, v) in kv {
                if !out.iter().any(|(o, _)| o == k) {
                    out.push((k.clone(), v.clone()));
                }
            }
        }
    }
    E::Dict(out)
}

fn lookup<'a>(run: &'a E, base: Option<&'a E>, key: &str) -> Option<&'a E> {
    run.get(key).or_else(|| base.and_then(|b| b.get(key)))
}

pub(crate) fn char_style(base: Option<&E>, d: &E, fonts: &[String], k: f32) -> CharStyle {
    let g = |key: &str| lookup(d, base, key);
    let num = |key: &str| g(key).and_then(E::as_f64);
    let flag = |key: &str| g(key).and_then(E::as_bool);
    let mut s = CharStyle::default();
    if let Some(ps) = g("Font").and_then(E::as_i64).and_then(|i| usize::try_from(i).ok()).and_then(|i| fonts.get(i)) {
        let r = guess_from_postscript(ps);
        s.font_family = r.family;
        s.weight = r.weight;
        s.italic = r.italic;
        s.postscript_name = Some(ps.clone());
    }
    if let Some(v) = num("FontSize") {
        s.size_pt = v as f32 * k;
    }
    if flag("AutoLeading") == Some(false)
        && let Some(v) = num("Leading")
    {
        s.leading_pt = Some(v as f32 * k);
    }
    s.tracking = num("Tracking").unwrap_or(0.0) as f32;
    s.baseline_shift_pt = num("BaselineShift").unwrap_or(0.0) as f32 * k;
    s.horizontal_scale = num("HorizontalScale").unwrap_or(1.0) as f32;
    s.vertical_scale = num("VerticalScale").unwrap_or(1.0) as f32;
    s.faux_bold = flag("FauxBold").unwrap_or(false);
    s.faux_italic = flag("FauxItalic").unwrap_or(false);
    s.underline = flag("Underline").unwrap_or(false);
    s.strikethrough = flag("Strikethrough").unwrap_or(false);
    s.ligatures = flag("Ligatures").unwrap_or(true);
    s.discretionary_ligatures = flag("DLigatures").unwrap_or(false);
    for (key, tag) in OPENTYPE_KEYS {
        if flag(key) == Some(true) {
            s.features.push(photocraft_doc::text::FontFeature { tag: tag.to_string(), value: 1 });
        }
    }
    s.kerning = if flag("AutoKerning") == Some(false) { Kerning::Off } else { Kerning::Metrics };
    s.kern = num("Kerning").filter(|v| v.is_finite()).unwrap_or(0.0) as f32;
    s.caps = match g("FontCaps").and_then(E::as_i64) {
        Some(1) => Caps::SmallCaps,
        Some(2) => Caps::AllCaps,
        _ => Caps::Normal,
    };
    if let Some(c) = g("FillColor") {
        let v = arr_f(c.get("Values"));
        let a = v.first().copied().unwrap_or(1.0) as f32;
        s.color = match v.len() {
            5 => Color { mode: ColorMode::Cmyk, c: [v[1] as f32, v[2] as f32, v[3] as f32, v[4] as f32], alpha: a },
            2 => Color { mode: ColorMode::Grayscale, c: [v[1] as f32, 0.0, 0.0, 0.0], alpha: a },
            4 => Color::rgba(v[1] as f32, v[2] as f32, v[3] as f32, a),
            _ => Color::BLACK,
        };
    }
    s
}

pub(crate) fn para_style(base: Option<&E>, d: &E, k: f32) -> ParagraphStyle {
    let num = |key: &str| lookup(d, base, key).and_then(E::as_f64);
    ParagraphStyle {
        align: match num("Justification").map(|v| v as i64) {
            Some(1) => TextAlign::Right,
            Some(2) => TextAlign::Center,
            Some(3) => TextAlign::JustifyLeft,
            Some(4) => TextAlign::JustifyRight,
            Some(5) => TextAlign::JustifyCenter,
            Some(6) => TextAlign::JustifyAll,
            _ => TextAlign::Left,
        },
        first_line_indent_pt: num("FirstLineIndent").unwrap_or(0.0) as f32 * k,
        start_indent_pt: num("StartIndent").unwrap_or(0.0) as f32 * k,
        end_indent_pt: num("EndIndent").unwrap_or(0.0) as f32 * k,
        space_before_pt: num("SpaceBefore").unwrap_or(0.0) as f32 * k,
        space_after_pt: num("SpaceAfter").unwrap_or(0.0) as f32 * k,
        auto_leading: num("AutoLeading").unwrap_or(1.2) as f32,
        hyphenate: lookup(d, base, "AutoHyphenate").and_then(E::as_bool).unwrap_or(false),
        ..Default::default()
    }
}

// ---------- writing ----------

fn real(v: f32) -> E {
    E::Real(f64::from(v))
}

/// Manual kerning as Photoshop stores it: a whole number of 1/1000 em (its UI range is ±1000;
/// anything non-finite writes 0).
pub(crate) fn kern_units(v: f32) -> i64 {
    if v.is_finite() { v.round().clamp(-10_000.0, 10_000.0) as i64 } else { 0 }
}

// Kerning in EngineData, as observed in Photoshop 2026 (writing, and reading files without
// `Txt2`): a manual kern k (1/1000 em) after character c is written on character c + 1 as
// `AutoKerning false` + `Kerning k`; everything else is `true` + 0. Photoshop's reader takes
// `Kerning k` on character j as the kern after j - 1 (making j - 1 manual), and
// `AutoKerning false` on j as "character j - 2 is manually kerned" (no automatic kerning). The
// automatic mode itself (Metrics or Optical) isn't in EngineData: Photoshop keeps it in the
// document's `Txt2` (see [`apply_txt2`]). We write and read the same way, so Photoshop reads
// our kerning as we do and Photoshop's own fields round-trip unchanged. (The mode of the last
// character has no slot and reads back as Metrics.)

fn manual(s: &CharStyle) -> bool {
    s.kerning == Kerning::Off || kern_units(s.kern) != 0
}

/// EngineData style runs for `ed_text` (the layer text with `\r` breaks and the trailing `\r`):
/// (model run index, `(AutoKerning, Kerning)`, UTF-16 length), split where those change.
fn pair_runs(ed_text: &str, runs: &[TextRun]) -> Vec<(usize, (bool, i64), usize)> {
    // Model run of each character (the trailing `\r` takes the last run).
    let mut owner: Vec<(usize, usize)> = Vec::with_capacity(ed_text.len());
    let (mut ri, mut run_end) = (0usize, runs.first().map_or(0, |r| r.len));
    for (b, ch) in ed_text.char_indices() {
        while b >= run_end && ri + 1 < runs.len() {
            ri += 1;
            run_end = run_end.saturating_add(runs.get(ri).map_or(0, |r| r.len));
        }
        owner.push((ri, ch.len_utf16()));
    }
    let style = |i: usize| owner.get(i).and_then(|o| runs.get(o.0)).map(|r| &r.style);
    let mut out: Vec<(usize, (bool, i64), usize)> = Vec::new();
    for (j, &(ri, n)) in owner.iter().enumerate() {
        let k = j.checked_sub(1).and_then(style).map_or(0, |s| kern_units(s.kern));
        // `false` marks character j - 2 manual: written when it is and its own kern (on j - 1)
        // doesn't say so already; harmless (and Photoshop's own form) next to a kern on j.
        let off = match j.checked_sub(2) {
            Some(p) => style(p).is_some_and(|s| manual(s) && (kern_units(s.kern) == 0 || k != 0)),
            None => k != 0,
        };
        let fields = (!off, k);
        match out.last_mut() {
            Some(last) if last.0 == ri && last.1 == fields => last.2 += n,
            _ => out.push((ri, fields, n)),
        }
    }
    out
}

/// Reads EngineData's kerning fields back into the model (see the notes above). `incoming` is
/// per EngineData style run: (UTF-8 length in the engine text, `AutoKerning`, `Kerning`).
fn kerning_from_pairs(text: &str, runs: Vec<TextRun>, incoming: &[(usize, bool, f32)]) -> Vec<TextRun> {
    if text.is_empty() || runs.is_empty() {
        return runs;
    }
    // Byte offset of every character, plus the trailing break.
    let starts: Vec<usize> = text.char_indices().map(|(b, _)| b).chain(std::iter::once(text.len())).collect();
    let n = starts.len().saturating_sub(1);
    let mut kern = vec![0.0f32; n];
    let mut off = vec![false; n];
    let (mut si, mut seg_end) = (0usize, incoming.first().map_or(0, |s| s.0));
    for (j, &b) in starts.iter().enumerate() {
        while b >= seg_end && si < incoming.len() {
            si += 1;
            seg_end = seg_end.saturating_add(incoming.get(si).map_or(0, |s| s.0));
        }
        let Some(&(_, auto, k)) = incoming.get(si) else { break };
        if k != 0.0
            && let Some(p) = j.checked_sub(1)
            && let (Some(kp), Some(op)) = (kern.get_mut(p), off.get_mut(p))
        {
            *kp = k;
            *op = true;
        }
        if !auto && let Some(op) = j.checked_sub(2).and_then(|p| off.get_mut(p)) {
            *op = true;
        }
    }
    let (mut ri, mut run_end) = (0usize, runs.first().map_or(0, |r| r.len));
    let mut out: Vec<TextRun> = Vec::new();
    for (c, w) in starts.windows(2).enumerate() {
        let (b, len) = (w[0], w[1] - w[0]);
        while b >= run_end && ri + 1 < runs.len() {
            ri += 1;
            run_end = run_end.saturating_add(runs.get(ri).map_or(0, |r| r.len));
        }
        let Some(r) = runs.get(ri) else { break };
        let k = kern.get(c).copied().unwrap_or(0.0);
        let kerning = if off.get(c).copied().unwrap_or(false) { Kerning::Off } else { Kerning::Metrics };
        match out.last_mut() {
            Some(last) if last.style.kern == k && last.style.kerning == kerning && same_but_kerning(&last.style, &r.style) => last.len += len,
            _ => {
                let mut style = r.style.clone();
                style.kerning = kerning;
                style.kern = k;
                out.push(TextRun { len, style });
            }
        }
    }
    out
}

/// Whether two styles differ at most in their kerning.
fn same_but_kerning(a: &CharStyle, b: &CharStyle) -> bool {
    CharStyle { kerning: b.kerning, kern: b.kern, ..a.clone() } == *b
}

/// PostScript name for a style (the PSD `FontSet` stores these).
pub(crate) fn postscript_for(s: &CharStyle) -> String {
    if let Some(ps) = &s.postscript_name {
        return ps.clone();
    }
    let fam = if s.font_family.is_empty() { crate::fonts::DEFAULT_FAMILY } else { &s.font_family };
    let w = match s.weight {
        0..=150 => "Thin",
        151..=250 => "ExtraLight",
        251..=350 => "Light",
        351..=450 => "Regular",
        451..=550 => "Medium",
        551..=650 => "SemiBold",
        651..=750 => "Bold",
        751..=850 => "ExtraBold",
        _ => "Black",
    };
    let style = match (w, s.italic) {
        ("Regular", true) => "Italic".to_string(),
        (w, true) => format!("{w}Italic"),
        (w, false) => w.to_string(),
    };
    format!("{}-{style}", fam.replace(' ', ""))
}

/// EngineData style keys for the Type › OpenType toggles and the OpenType feature each one turns
/// on (the standard and discretionary ligature keys map to dedicated [`CharStyle`] fields).
pub const OPENTYPE_KEYS: [(&str, &str); 8] = [
    ("ContextualLigatures", "calt"),
    ("Swash", "swsh"),
    ("OldStyle", "onum"),
    ("StylisticAlternates", "salt"),
    ("Titling", "titl"),
    ("Ornaments", "ornm"),
    ("Ordinals", "ordn"),
    ("Fractions", "frac"),
];

pub(crate) fn style_sheet_data(s: &CharStyle, font: usize, k: f32) -> E {
    let c = &s.color;
    let values = match c.mode {
        ColorMode::Cmyk => vec![real(c.alpha), real(c.c[0]), real(c.c[1]), real(c.c[2]), real(c.c[3])],
        ColorMode::Grayscale => vec![real(c.alpha), real(c.c[0])],
        _ => {
            let [r, g, b] = c.to_rgb();
            vec![real(c.alpha), real(r), real(g), real(b)]
        }
    };
    let color_type = match c.mode {
        ColorMode::Cmyk => 2,
        ColorMode::Grayscale => 0,
        _ => 1,
    };
    let mut opentype: Vec<(String, E)> = OPENTYPE_KEYS
        .iter()
        .filter(|(_, tag)| s.features.iter().any(|f| f.tag == *tag && f.value > 0))
        .map(|(key, _)| (key.to_string(), E::Bool(true)))
        .collect();
    let mut dict = vec![
        ("Font".into(), E::Int(font as i64)),
        ("FontSize".into(), real(s.size_pt / k)),
        ("FauxBold".into(), E::Bool(s.faux_bold)),
        ("FauxItalic".into(), E::Bool(s.faux_italic)),
        ("AutoLeading".into(), E::Bool(s.leading_pt.is_none())),
        ("Leading".into(), real(s.leading_pt.unwrap_or(0.0) / k)),
        ("HorizontalScale".into(), real(s.horizontal_scale)),
        ("VerticalScale".into(), real(s.vertical_scale)),
        ("Tracking".into(), E::Int(s.tracking.round() as i64)),
        ("AutoKerning".into(), E::Bool(s.kerning != Kerning::Off)),
        ("Kerning".into(), E::Int(kern_units(s.kern))),
        ("BaselineShift".into(), real(s.baseline_shift_pt / k)),
        (
            "FontCaps".into(),
            E::Int(match s.caps {
                Caps::Normal => 0,
                Caps::SmallCaps => 1,
                Caps::AllCaps => 2,
            }),
        ),
        ("Underline".into(), E::Bool(s.underline)),
        ("Strikethrough".into(), E::Bool(s.strikethrough)),
        ("Ligatures".into(), E::Bool(s.ligatures)),
        ("DLigatures".into(), E::Bool(s.discretionary_ligatures)),
        // Photopea lays out edited text without painting it unless its fill is enabled.
        ("FillFlag".into(), E::Bool(true)),
        ("FillColor".into(), E::Dict(vec![("Type".into(), E::Int(color_type)), ("Values".into(), E::Array(values))])),
    ];
    dict.append(&mut opentype);
    E::Dict(dict)
}

pub(crate) fn paragraph_properties(p: &ParagraphStyle, k: f32) -> E {
    let j = match p.align {
        TextAlign::Left => 0,
        TextAlign::Right => 1,
        TextAlign::Center => 2,
        TextAlign::JustifyLeft => 3,
        TextAlign::JustifyRight => 4,
        TextAlign::JustifyCenter => 5,
        TextAlign::JustifyAll => 6,
    };
    E::Dict(vec![
        ("Justification".into(), E::Int(j)),
        ("FirstLineIndent".into(), real(p.first_line_indent_pt / k)),
        ("StartIndent".into(), real(p.start_indent_pt / k)),
        ("EndIndent".into(), real(p.end_indent_pt / k)),
        ("SpaceBefore".into(), real(p.space_before_pt / k)),
        ("SpaceAfter".into(), real(p.space_after_pt / k)),
        ("AutoHyphenate".into(), E::Bool(p.hyphenate)),
        ("AutoLeading".into(), real(p.auto_leading)),
        ("LeadingType".into(), E::Int(0)),
    ])
}

fn utf16_len(s: &str) -> i64 {
    s.encode_utf16().count() as i64
}

pub(crate) fn font_entry(name: &str) -> E {
    E::Dict(vec![("Name".into(), E::String(name.into())), ("Script".into(), E::Int(0)), ("FontType".into(), E::Int(1)), ("Synthetic".into(), E::Int(0))])
}

/// The `Txt2` auto-kern mode of a style: 0 manual, 1 metrics, 2 optical — the values
/// [`apply_txt2`] maps back to [`Kerning`]. EngineData's `AutoKerning` can't say Metrics from
/// Optical, so the mode only lives in `Txt2`.
fn txt2_mode(s: &CharStyle) -> i64 {
    if manual(s) {
        0
    } else if s.kerning == Kerning::Optical {
        2
    } else {
        1
    }
}

/// `Txt2` style runs for `layer`: `(mode, UTF-16 length)` over the engine text (`\r` breaks and
/// the trailing `\r`, which takes the last run's mode), split where the auto-kern mode changes.
fn txt2_mode_runs(layer: &TextLayer) -> Vec<(i64, usize)> {
    let runs = layer.char_runs();
    let ed_text = format!("{}\r", layer.text.replace('\n', "\r"));
    let mut out: Vec<(i64, usize)> = Vec::new();
    let (mut ri, mut run_end) = (0usize, runs.first().map_or(0, |r| r.len));
    for (b, ch) in ed_text.char_indices() {
        while b >= run_end && ri + 1 < runs.len() {
            ri += 1;
            run_end = run_end.saturating_add(runs.get(ri).map_or(0, |r| r.len));
        }
        let mode = runs.get(ri).map_or(1, |r| txt2_mode(&r.style));
        match out.last_mut() {
            Some(last) if last.0 == mode => last.1 += ch.len_utf16(),
            _ => out.push((mode, ch.len_utf16())),
        }
    }
    out
}

/// The highest text-object number honoured here (`TextIndex`, and the slot array [`build_txt2`]
/// sizes). PSD text objects are numbered densely from 0; a crafted file can name object
/// `i32::MAX`, and sizing the array by it would make every save allocate gigabytes.
pub const MAX_TEXT_INDEX: i32 = 0xFFFF;

/// Whether the document `Txt2` block is needed for `objects`: only a non-Metrics auto-kern mode
/// (`Optical`) is exclusive to `Txt2` — EngineData's `AutoKerning` cannot distinguish Metrics from
/// Optical — so a fresh document of ordinary Metrics text has nothing to put in the block.
/// Writing Photoshop's empty `<< /0 << >> >>` stub anyway makes some readers (Photopea) fail
/// (#3514); the export omits the block for such a document.
pub fn needs_txt2(objects: &[(i32, &TextLayer)]) -> bool {
    objects.iter().any(|(_, l)| l.char_runs().iter().any(|r| r.style.kerning == Kerning::Optical))
}

/// The document's `Txt2` block for `objects`: `(TextIndex, layer)` pairs, one text object each
/// at the slot its `TextIndex` names (see [`set_text_index`] and [`text_index`]; unused slots are
/// empty dictionaries), carrying the automatic kerning mode of every pair (`/11`: 0 manual,
/// 1 metrics, 2 optical) — the one thing EngineData can't say (`AutoKerning true` covers Metrics
/// and Optical alike). PSD export writes this block for the type layers it writes `TySh` data
/// for; [`apply_txt2`] reads it back. Numbers outside `0..=`[`MAX_TEXT_INDEX`] are skipped: the
/// slot array is sized by real text objects, never by a file's numbers.
///
/// `previous` is the block this replaces (the file's own). A kept object — one whose text is
/// unchanged — keeps everything the file held beyond the two regenerated keys (its model's text
/// and style runs): Photoshop stores glyph pen positions under `/21 /1` there, and a save with
/// no text edit must not drop them. Its own style runs keep every style key too: only their
/// auto-kern mode is set (split where a mode changes inside a run). A changed text invalidates
/// those extras along with the old runs, so they are dropped rather than left stale. Everything
/// else the file's block held — the other keys beside the text objects, objects no layer names —
/// is kept, and a block whose content comes out unchanged is returned byte for byte (#2374:
/// Photoshop refused files whose `Txt2` an unedited save had stripped).
pub fn build_txt2(objects: &[(i32, &TextLayer)], previous: Option<&[u8]>) -> Vec<u8> {
    let prev = previous.and_then(parse_txt2);
    let prev_objects: &[E] = prev.as_ref().and_then(|p| p.path(&["1", "1"]).and_then(E::as_array)).unwrap_or(&[]);
    let mut slots: Vec<E> = prev_objects.to_vec();
    for (index, layer) in objects {
        if !(0..=MAX_TEXT_INDEX).contains(index) {
            continue;
        }
        let text = format!("{}\r", layer.text.replace('\n', "\r"));
        let modes = txt2_mode_runs(layer);
        let at = *index as usize;
        let kept = prev_objects.get(at).filter(|o| o.get("0").and_then(|m| m.get("0")).and_then(E::as_str) == Some(text.as_str()));
        let mut object = kept.cloned().unwrap_or_else(E::dict);
        let mut model = object.get("0").cloned().filter(|m| matches!(m, E::Dict(_))).unwrap_or_else(E::dict);
        let file_runs = kept.and_then(|_| model.path(&["6", "0"]).and_then(E::as_array)).and_then(|runs| with_modes(runs, &modes));
        let runs = file_runs.unwrap_or_else(|| {
            modes
                .iter()
                .map(|&(mode, len)| {
                    // `<< /0 << /0 << /0 () /6 << /0 0 /11 mode >> >> >> /1 len >>`.
                    let style = E::Dict(vec![("0".into(), E::Int(0)), ("11".into(), E::Int(mode))]);
                    let data = E::Dict(vec![("0".into(), E::String(String::new())), ("6".into(), style)]);
                    E::Dict(vec![("0".into(), E::Dict(vec![("0".into(), data)])), ("1".into(), E::Int(len as i64))])
                })
                .collect()
        });
        model.set("0", E::String(text));
        match model.get_mut("6") {
            Some(style_run @ E::Dict(_)) => style_run.set("0", E::Array(runs)),
            _ => model.set("6", E::Dict(vec![("0".into(), E::Array(runs))])),
        }
        object.set("0", model);
        if slots.len() <= at {
            slots.resize(at + 1, E::dict());
        }
        if let Some(slot) = slots.get_mut(at) {
            *slot = object;
        }
    }
    // The outer dictionary keeps whatever the block held (`/98` version, block extras, …), and
    // `/1` whatever it holds beside the text objects.
    let mut outer = prev.clone().unwrap_or_else(|| E::Dict(vec![("98".into(), E::Dict(vec![("0".into(), E::Int(14))])), ("0".into(), E::dict())]));
    match outer.get_mut("1") {
        Some(objs @ E::Dict(_)) => objs.set("1", E::Array(slots)),
        _ => outer.set("1", E::Dict(vec![("1".into(), E::Array(slots))])),
    }
    // Nothing changed: the file's own bytes (writing them back would re-encode strings and numbers).
    if let (Some(bytes), Some(p)) = (previous, &prev)
        && *p == outer
    {
        return bytes.to_vec();
    }
    match outer {
        E::Dict(items) => ed::write_bare(&items),
        // Unreachable: `parse_txt2` and the fallback above both yield a dictionary.
        _ => ed::write_bare(&[]),
    }
}

/// The file's own `Txt2` style runs `runs` with the auto-kern modes `modes` (`(mode, UTF-16
/// length)`, see [`txt2_mode_runs`]) set at `/0 /0 /6 /11`, keeping every other style key. A
/// run a mode change falls inside is split there. `None` when the runs don't cover the text or
/// lack a style dictionary: the caller writes its own runs instead.
fn with_modes(runs: &[E], modes: &[(i64, usize)]) -> Option<Vec<E>> {
    let lens: Vec<usize> = runs.iter().map(|r| r.get("1").and_then(E::as_i64).and_then(|l| usize::try_from(l).ok())).collect::<Option<_>>()?;
    let total = modes.iter().try_fold(0usize, |a, m| a.checked_add(m.1))?;
    if lens.iter().try_fold(0usize, |a, l| a.checked_add(*l))? != total {
        return None;
    }
    let mut out = Vec::with_capacity(runs.len());
    let (mut mi, mut mode_end) = (0usize, modes.first()?.1);
    let mut at = 0usize;
    for (run, len) in runs.iter().zip(lens) {
        if len == 0 {
            out.push(run.clone());
            continue;
        }
        let end = at + len;
        while at < end {
            while at >= mode_end && mi + 1 < modes.len() {
                mi += 1;
                mode_end = mode_end.saturating_add(modes.get(mi)?.1);
            }
            let piece_end = end.min(mode_end);
            if piece_end <= at {
                return None;
            }
            let mut piece = run.clone();
            if piece_end - at != len {
                piece.set("1", E::Int(i64::try_from(piece_end - at).ok()?));
            }
            let style = piece.get_mut("0").and_then(|v| v.get_mut("0")).and_then(|v| v.get_mut("6")).filter(|v| matches!(v, E::Dict(_)))?;
            let mode = modes.get(mi)?.0;
            // Keep the file's value (and its number type) when the mode is the same.
            if style.get("11").and_then(E::as_i64) != Some(mode) {
                style.set("11", E::Int(mode));
            }
            out.push(piece);
            at = piece_end;
        }
    }
    Some(out)
}

/// The layer's text-object number in the document's `Txt2` block, when its `TySh` names one
/// within `0..=`[`MAX_TEXT_INDEX`]. PSD export keeps a preserved number — its regenerated `Txt2`
/// object sits at that slot, and [`apply_txt2`] looks the slot up — and numbers layers without
/// one (or with a file-controlled out-of-range one) afresh.
pub fn text_index(tysh: &[u8]) -> Option<i32> {
    let t = parse_tysh(tysh)?;
    match t.text.get("TextIndex") {
        Some(D::Integer(i)) if (0..=MAX_TEXT_INDEX).contains(i) => Some(*i),
        _ => None,
    }
}

/// Rewrites a `TySh`'s `TextIndex`: the layer's text-object number in the document's `Txt2`
/// block ([`build_txt2`] writes them in this order). `None` when `tysh` doesn't parse.
pub fn set_text_index(tysh: &[u8], index: i32) -> Option<Vec<u8>> {
    let mut t = parse_tysh(tysh)?;
    match t.text.items.iter_mut().find(|(k, _)| k.is("TextIndex")) {
        Some(e) => e.1 = D::Integer(index),
        None => t.text.items.push((Id::new("TextIndex"), D::Integer(index))),
    }
    Some(write_tysh(&t))
}

/// PSD paragraph type hides overset text. Fit a copy to the complete layout and freeze its
/// automatic wraps as explicit paragraphs; the source frame and character styles stay untouched.
/// Cached pixels are retained, and Photoshop can recompose every character on the first edit.
pub fn visible_overflow_for_psd(layer: &TextLayer, layout: &crate::TextLayout) -> Option<TextLayer> {
    let TextShape::Box { x, y, width, height } = layer.shape else { return None };
    if !layer.overflow_visible && layer.block_align == photocraft_doc::text::BlockAlign::Start {
        return None;
    }
    let mut bounds = layout.bounds()?;
    let (dx, dy) = layout.to_text(0.0, layout.block_offset);
    bounds[0] -= dx;
    bounds[2] -= dx;
    bounds[1] -= dy;
    bounds[3] -= dy;
    let paras = layer.paragraph_runs();
    let starts: Vec<_> = paras
        .iter()
        .scan(0usize, |at, para| {
            let start = *at;
            *at += para.len;
            Some(start)
        })
        .collect();
    let paragraph_at = |offset| &paras[starts.partition_point(|&start| start <= offset).saturating_sub(1)].style;
    let k = layout.px_per_pt.max(0.001);
    let inline = if layout.vertical { height } else { width };
    let mut ends: Vec<_> = layout.lines.iter().map(|line| line.x1).collect();
    for cluster in &layout.clusters {
        if let Some(end) = ends.get_mut(cluster.line) {
            *end = end.max(cluster.x + cluster.advance);
        }
    }
    // Photoshop uses the font's vertical advance; some CJK faces use more than one em.
    let vertical_advance = if layout.vertical {
        use skrifa::raw::TableProvider;
        layout
            .faces
            .iter()
            .filter_map(|face| {
                let font = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index).ok()?;
                Some(
                    f32::from(font.vmtx().ok()?.v_metrics().iter().map(|metric| metric.advance()).max()?)
                        / f32::from(font.head().ok()?.units_per_em()).max(1.0),
                )
            })
            .fold(1.0f32, f32::max)
    } else {
        1.0
    };
    // Leave one em for Photoshop's first-glyph ascent/descent in vertical paragraph frames.
    let vertical_inset = if layout.vertical { layout.faces.iter().map(|f| f.size_px).fold(0.0f32, f32::max) } else { 0.0 };
    let needed = layout.lines.iter().enumerate().fold(0.0f32, |acc, (index, line)| {
        let indents = (paragraph_at(line.range.start).start_indent_pt + paragraph_at(line.range.start).end_indent_pt) * k;
        // Trailing spaces before a frozen wrap still need room in Photoshop.
        let end = ends.get(index).copied().unwrap_or(line.x1);
        acc.max((end - line.x0) * vertical_advance + indents + vertical_inset)
    });
    let delta = if needed > inline + 0.001 { needed - inline + 1.0 } else { 0.0 };
    let block_overflow = layout.lines.iter().any(|line| {
        if layout.vertical { line.baseline - layout.block_offset + line.descent > -x + 0.5 } else { line.baseline - layout.block_offset > y + height + 0.5 }
    });
    let mut copy = layer.clone();
    copy.block_align = photocraft_doc::text::BlockAlign::Start;
    // Photoshop has no block alignment field. Bake only the export copy's layout
    // translation into TySh, preserving the page frame and its repair geometry.
    let m = &mut copy.transform.m;
    m[4] += m[0] * f64::from(dx) + m[2] * f64::from(dy);
    m[5] += m[1] * f64::from(dx) + m[3] * f64::from(dy);
    copy.psd_raw = None;
    copy.overflow_visible = false;
    if !block_overflow && delta < 0.01 {
        return (layout.block_offset != 0.0).then_some(copy);
    }
    let breaks: std::collections::BTreeSet<_> = layout
        .lines
        .iter()
        .skip(1)
        .map(|line| line.range.start)
        .filter(|&at| {
            at > 0
                && at < layer.text.len()
                && layer.text.is_char_boundary(at)
                && layer.text.get(..at).and_then(|s| s.chars().next_back()).is_some_and(|c| !matches!(c, '\r' | '\n' | '\u{3}'))
        })
        .collect();
    copy.text = layer.text.char_indices().fold(String::new(), |mut text, (at, ch)| {
        if breaks.contains(&at) {
            text.push('\n');
        }
        text.push(if ch == '\u{3}' { '\n' } else { ch });
        text
    });
    let mut at = 0;
    copy.runs = layer.char_runs();
    for run in &mut copy.runs {
        let end = at + run.len;
        // An inserted break inherits the preceding character, otherwise a following large font
        // changes the previous line's ascent and moves every subsequent line.
        run.len += breaks.range((std::ops::Bound::Excluded(at), std::ops::Bound::Included(end))).count();
        at = end;
    }
    copy.paragraphs = layout
        .lines
        .iter()
        .enumerate()
        .map(|(index, line)| {
            let mut style = paragraph_at(line.range.start).clone();
            if index > 0 && layout.lines[index - 1].paragraph == line.paragraph {
                style.space_before_pt = 0.0;
                style.first_line_indent_pt = 0.0;
            }
            let end = layout.lines.get(index + 1).map_or(layer.text.len(), |next| {
                if next.paragraph == line.paragraph {
                    style.space_after_pt = 0.0;
                }
                next.range.start
            });
            let factor = match style.align {
                TextAlign::Center | TextAlign::JustifyCenter => 0.5,
                TextAlign::Right | TextAlign::JustifyRight => 1.0,
                _ => 0.0,
            };
            style.start_indent_pt -= factor * delta / k;
            style.end_indent_pt += factor * delta / k;
            let added = breaks.range((std::ops::Bound::Excluded(line.range.start), std::ops::Bound::Included(end))).count();
            ParagraphRun { len: end - line.range.start + added, style }
        })
        .collect();
    copy.shape = if layout.vertical {
        let fitted = width.max(x + width - bounds[0] + 1.0 + vertical_inset);
        TextShape::Box { x: x + width - fitted, y, width: fitted, height: height + delta }
    } else {
        TextShape::Box { x, y, width: width + delta, height: height.max(bounds[3] - y + 1.0) }
    };
    Some(copy)
}

/// Updates (or creates) an EngineData tree for `layer`.
pub fn build_engine_data(layer: &TextLayer, template: Option<E>, dpi: f32) -> E {
    let k = 72.0 / if dpi > 0.0 { dpi } else { 72.0 };
    let fresh = || E::Dict(vec![("EngineDict".into(), E::dict()), ("ResourceDict".into(), E::dict())]);
    // The template comes from the PSD: it may not be a dictionary or may lack `EngineDict`.
    let mut e = match template {
        Some(t @ E::Dict(_)) => t,
        _ => fresh(),
    };
    if !matches!(e.get("EngineDict"), Some(E::Dict(_))) {
        e.set("EngineDict", E::dict());
    }
    let runs = layer.char_runs();
    let paras = layer.paragraph_runs();
    let text = layer.text.replace('\n', "\r");
    let ed_text = format!("{text}\r");

    // Fonts: keep the template's font list and append new PostScript names.
    let mut fonts: Vec<String> = e
        .path(&["ResourceDict", "FontSet"])
        .and_then(E::as_array)
        .map(|a| a.iter().map(|f| f.get("Name").and_then(E::as_str).unwrap_or("").to_string()).collect())
        .unwrap_or_default();
    let mut font_index = |ps: String| match fonts.iter().position(|f| *f == ps) {
        Some(i) => i,
        None => {
            fonts.push(ps);
            fonts.len() - 1
        }
    };
    let mut sruns = Vec::new();
    let mut slens = Vec::new();
    for (ri, pair, n) in pair_runs(&ed_text, &runs) {
        let Some(r) = runs.get(ri) else { continue };
        let fi = font_index(postscript_for(&r.style));
        let mut data = style_sheet_data(&r.style, fi, k);
        data.set("AutoKerning", E::Bool(pair.0));
        data.set("Kerning", E::Int(pair.1));
        sruns.push(E::Dict(vec![("StyleSheet".into(), E::Dict(vec![("StyleSheetData".into(), data)]))]));
        slens.push(E::Int(n as i64));
    }
    let mut pruns = Vec::new();
    let mut plens = Vec::new();
    let mut at = 0usize;
    let mut pi = 0;
    let mut end = paras[0].len;
    // Photoshop needs one record per hard paragraph, including the terminal
    // marker. Native style runs may span paragraphs; use the style at each start.
    for paragraph in ed_text.split_inclusive('\r') {
        while at >= end && pi + 1 < paras.len() {
            pi += 1;
            end += paras[pi].len;
        }
        pruns.push(E::Dict(vec![(
            "ParagraphSheet".into(),
            E::Dict(vec![("DefaultStyleSheet".into(), E::Int(0)), ("Properties".into(), paragraph_properties(&paras[pi].style, k))]),
        )]));
        plens.push(E::Int(utf16_len(paragraph)));
        at += paragraph.len();
    }
    let fontset = E::Array(fonts.iter().map(|f| font_entry(f)).collect());

    let Some(dict) = e.get_mut("EngineDict") else {
        return fresh();
    };
    let mut editor = dict.get("Editor").cloned().unwrap_or_else(E::dict);
    editor.set("Text", E::String(ed_text));
    dict.set("Editor", editor);
    let mut pr = dict.get("ParagraphRun").cloned().unwrap_or_else(E::dict);
    pr.set("RunArray", E::Array(pruns));
    pr.set("RunLengthArray", E::Array(plens));
    if pr.get("IsJoinable").is_none() {
        pr.set("IsJoinable", E::Int(1));
    }
    dict.set("ParagraphRun", pr);
    let mut sr = dict.get("StyleRun").cloned().unwrap_or_else(E::dict);
    sr.set("RunArray", E::Array(sruns));
    sr.set("RunLengthArray", E::Array(slens));
    if sr.get("IsJoinable").is_none() {
        sr.set("IsJoinable", E::Int(2));
    }
    dict.set("StyleRun", sr);
    dict.set(
        "AntiAlias",
        E::Int(match layer.antialias {
            AntiAlias::None => 0,
            AntiAlias::Crisp => 1,
            AntiAlias::Strong => 2,
            AntiAlias::Smooth => 3,
            AntiAlias::Sharp => 4,
            // No Photoshop file confirms the platform modes' numbers here; the `AntA` descriptor
            // carries the exact mode, and Smooth is the closest.
            AntiAlias::Windows | AntiAlias::WindowsLcd => 3,
        }),
    );
    if dict.get("UseFractionalGlyphWidths").is_none() {
        dict.set("UseFractionalGlyphWidths", E::Bool(true));
    }
    let (shape_type, box_bounds) = match layer.shape {
        // PSD path type is not written yet: the layer saves as point text.
        TextShape::Point | TextShape::Path { .. } => (0, None),
        TextShape::Box { x, y, width, height } => (1, Some([x, y, x + width, y + height])),
    };
    let mut photoshop = E::Dict(vec![("ShapeType".into(), E::Int(shape_type))]);
    match box_bounds {
        None => photoshop.set("PointBase", E::Array(vec![real(0.0), real(0.0)])),
        Some(b) => photoshop.set("BoxBounds", E::Array(b.iter().map(|v| real(*v)).collect())),
    }
    photoshop.set(
        "Base",
        E::Dict(vec![
            ("ShapeType".into(), E::Int(shape_type)),
            ("TransformPoint0".into(), E::Array(vec![real(1.0), real(0.0)])),
            ("TransformPoint1".into(), E::Array(vec![real(0.0), real(1.0)])),
            ("TransformPoint2".into(), E::Array(vec![real(0.0), real(0.0)])),
        ]),
    );
    // EngineData writing direction and procession, as Photoshop writes vertical type: 2 and 1
    // (horizontal: 0 and 0). The descriptor's `Ornt` agrees.
    let writing = if layer.orientation == Orientation::Vertical { 2 } else { 0 };
    let child = E::Dict(vec![
        ("ShapeType".into(), E::Int(shape_type)),
        ("Procession".into(), E::Int(i64::from(writing == 2))),
        ("Lines".into(), E::Dict(vec![("WritingDirection".into(), E::Int(writing)), ("Children".into(), E::Array(vec![]))])),
        ("Cookie".into(), E::Dict(vec![("Photoshop".into(), photoshop)])),
    ]);
    dict.set(
        "Rendered",
        E::Dict(vec![
            ("Version".into(), E::Int(1)),
            ("Shapes".into(), E::Dict(vec![("WritingDirection".into(), E::Int(writing)), ("Children".into(), E::Array(vec![child]))])),
        ]),
    );

    for key in ["ResourceDict", "DocumentResources"] {
        if key == "DocumentResources" && e.get(key).is_none() {
            let rd = e.get("ResourceDict").cloned().unwrap_or_else(E::dict);
            e.set(key, rd);
            continue;
        }
        let Some(rd) = e.get_mut(key) else { continue };
        rd.set("FontSet", fontset.clone());
        if rd.get("StyleSheetSet").is_none() {
            let base = runs.first().map(|r| r.style.clone()).unwrap_or_default();
            rd.set(
                "StyleSheetSet",
                E::Array(vec![E::Dict(vec![("Name".into(), E::String("Normal RGB".into())), ("StyleSheetData".into(), style_sheet_data(&base, 0, k))])]),
            );
        }
        if rd.get("ParagraphSheetSet").is_none() {
            rd.set(
                "ParagraphSheetSet",
                E::Array(vec![E::Dict(vec![
                    ("Name".into(), E::String("Normal RGB".into())),
                    ("DefaultStyleSheet".into(), E::Int(0)),
                    ("Properties".into(), paragraph_properties(&ParagraphStyle::default(), k)),
                ])]),
            );
        }
        for (k2, v) in [("TheNormalStyleSheet", 0), ("TheNormalParagraphSheet", 0)] {
            if rd.get(k2).is_none() {
                rd.set(k2, E::Int(v));
            }
        }
    }
    e
}

/// The anti-aliasing mode for EngineData's `AntiAlias` number, as Photoshop writes it: 0 None,
/// 1 Crisp, 2 Strong, 3 Smooth, 4 Sharp. Other numbers (platform modes) are left to `AntA`.
fn engine_antialias(n: f64) -> Option<AntiAlias> {
    Some(match n as i64 {
        0 => AntiAlias::None,
        1 => AntiAlias::Crisp,
        2 => AntiAlias::Strong,
        3 => AntiAlias::Smooth,
        4 => AntiAlias::Sharp,
        _ => return None,
    })
}

/// Builds the `TySh` block for `layer`, patching its current `psd_raw` when present.
/// `ink` is the layer's rendered bounds in text space (for `bounds`/`boundingBox`).
pub fn build_tysh(layer: &TextLayer, dpi: f32, ink: Option<[f32; 4]>) -> Vec<u8> {
    let old = layer.psd_raw.as_deref().and_then(|d| parse_tysh(d));
    let template = old.as_ref().and_then(|t| engine_data(&t.text));
    let e = build_engine_data(layer, template, dpi);
    let mut text = old.as_ref().map(|t| t.text.clone()).unwrap_or_else(|| Descriptor::new("TxLr"));
    let set = |d: &mut Descriptor, key: &str, v: D| match d.items.iter_mut().find(|(k, _)| k.is(key)) {
        Some(e) => e.1 = v,
        None => d.items.push((Id::new(key), v)),
    };
    set(&mut text, "Txt ", D::Text(UnicodeString::new_nul(&layer.text.replace('\n', "\r"))));
    if text.get("textGridding").is_none() {
        set(&mut text, "textGridding", enumv("textGridding", "None"));
    }
    set(&mut text, "Ornt", enumv("Ornt", if layer.orientation == Orientation::Vertical { "Vrtc" } else { "Hrzn" }));
    set(
        &mut text,
        "AntA",
        enumv(
            "Annt",
            match layer.antialias {
                AntiAlias::None => "Anno",
                AntiAlias::Sharp => "antiAliasSharp",
                AntiAlias::Crisp => "AnCr",
                AntiAlias::Strong => "AnSt",
                AntiAlias::Smooth => "AnSm",
                AntiAlias::Windows => "antiAliasPlatformGray",
                AntiAlias::WindowsLcd => "antiAliasPlatformLCD",
            },
        ),
    );
    let [l, t, r, b] = match (&layer.shape, ink) {
        (&TextShape::Box { x, y, width, height }, _) => [x, y, x + width, y + height],
        (_, Some(i)) => i,
        _ => [0.0; 4],
    };
    let rect = |cls: &str| {
        D::Descriptor(
            Descriptor::new(cls)
                .with("Left", D::UnitFloat { unit: *b"#Pnt", value: f64::from(l) })
                .with("Top ", D::UnitFloat { unit: *b"#Pnt", value: f64::from(t) })
                .with("Rght", D::UnitFloat { unit: *b"#Pnt", value: f64::from(r) })
                .with("Btom", D::UnitFloat { unit: *b"#Pnt", value: f64::from(b) }),
        )
    };
    set(&mut text, "bounds", rect("bounds"));
    set(&mut text, "boundingBox", rect("boundingBox"));
    if text.get("TextIndex").is_none() {
        set(&mut text, "TextIndex", D::Integer(0));
    }
    set(&mut text, "EngineData", D::RawData(ed::write(&e)));
    let warp = match (&layer.warp, old.as_ref().and_then(|o| o.warp.clone())) {
        (Some(w), _) => Some(
            Descriptor::new("warp")
                .with("warpStyle", enumv("warpStyle", &w.style))
                .with("warpValue", D::Double(f64::from(w.value)))
                .with("warpPerspective", D::Double(f64::from(w.horizontal_distortion)))
                .with("warpPerspectiveOther", D::Double(f64::from(w.vertical_distortion)))
                .with("warpRotate", enumv("Ornt", if w.horizontal { "Hrzn" } else { "Vrtc" })),
        ),
        (None, Some(old)) if enum_value(&old, "warpStyle").as_deref() == Some("warpNone") => Some(old),
        _ => None,
    };
    // Photoshop's trailing bounds: zeros for unwarped type, the text-space bounds when warped.
    let warped = warp.as_ref().is_some_and(|w| enum_value(w, "warpStyle").is_some_and(|s| s != "warpNone"));
    let bounds = if warped { [l, t, r, b] } else { [0.0; 4] };
    write_tysh(&TySh { transform: layer.transform, text, warp, bounds })
}
