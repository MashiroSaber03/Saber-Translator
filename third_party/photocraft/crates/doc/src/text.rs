//! Typed style model for type (text) layers: character runs, paragraph runs, point/box shape and
//! warp. Pure data; layout and rendering live in `photocraft-text`.
//!
//! Units follow Photoshop's Character/Paragraph panels: sizes, leading, indents and spacing are in
//! **points** (converted to pixels with the document resolution: `px = pt * dpi / 72`), tracking
//! is in 1/1000 em, and box bounds are in text-space **pixels** (before the layer transform).
//!
//! Runs cover the text by **UTF-8 byte length**, in order. A run list whose lengths don't add up
//! to `text.len()` is normalised by [`crate::TextLayer::char_runs`] (the last run is stretched or
//! truncated), so edits never produce unstyled text.

use photocraft_color::Color;
use serde::{Deserialize, Serialize};

/// Position the wrapped text block inside its frame, without moving the frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockAlign {
    #[default]
    Start,
    Center,
    End,
}

/// Kerning mode (Photoshop: Metrics, Optical, 0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kerning {
    /// Font kerning (`kern` feature).
    #[default]
    Metrics,
    /// Photoshop's optical kerning: pair spacing computed from the glyph outlines.
    Optical,
    /// No automatic kerning (Photoshop's "0" / manual kerning).
    Off,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Caps {
    #[default]
    Normal,
    SmallCaps,
    AllCaps,
}

/// An OpenType feature setting, e.g. `{ tag: "ss01", value: 1 }`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FontFeature {
    pub tag: String,
    pub value: u16,
}

/// A variable-font axis setting, e.g. `{ axis: "wght", value: 650.0 }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FontVariation {
    pub axis: String,
    pub value: f32,
}

/// Character style (Photoshop's Character panel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CharStyle {
    /// Family name ("Inter"). Empty = the default family.
    pub font_family: String,
    /// Face style name as shown in the UI ("Bold Italic"); informational, `weight`/`italic` rule.
    pub font_style: String,
    /// PostScript name from a PSD (`ArialMT`), used to find the exact face when present.
    pub postscript_name: Option<String>,
    /// CSS-style weight, 100–900.
    pub weight: u16,
    pub italic: bool,
    pub size_pt: f32,
    pub color: Color,
    /// Letter spacing in 1/1000 em.
    pub tracking: f32,
    /// Baseline-to-baseline distance in points; `None` = auto (paragraph `auto_leading` × size).
    pub leading_pt: Option<f32>,
    /// Positive raises the text.
    pub baseline_shift_pt: f32,
    pub horizontal_scale: f32,
    pub vertical_scale: f32,
    pub underline: bool,
    pub strikethrough: bool,
    pub faux_bold: bool,
    pub faux_italic: bool,
    /// Automatic (pair) kerning mode.
    pub kerning: Kerning,
    /// Manual kerning in 1/1000 em between this character and the next one (Photoshop stores it
    /// per character run as `Kerning`). Added to the automatic kerning of the pair.
    pub kern: f32,
    pub caps: Caps,
    /// Standard ligatures (`liga`/`clig`).
    pub ligatures: bool,
    /// Discretionary ligatures (`dlig`).
    pub discretionary_ligatures: bool,
    /// Extra OpenType features (`ss01`, `onum`, `tnum`, …).
    pub features: Vec<FontFeature>,
    /// Variable-font axis values.
    pub variations: Vec<FontVariation>,
    /// BCP-47 language tag (affects shaping and line breaking).
    pub language: Option<String>,
    /// Applied character style (id in [`crate::TextStyles`]); `None` = "None". Attributes above
    /// stay fully resolved; differences from the style are local overrides.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style_sheet: Option<u32>,
}

impl Default for CharStyle {
    fn default() -> Self {
        Self {
            font_family: String::new(),
            font_style: String::new(),
            postscript_name: None,
            weight: 400,
            italic: false,
            size_pt: 12.0,
            color: Color::BLACK,
            tracking: 0.0,
            leading_pt: None,
            baseline_shift_pt: 0.0,
            horizontal_scale: 1.0,
            vertical_scale: 1.0,
            underline: false,
            strikethrough: false,
            faux_bold: false,
            faux_italic: false,
            kerning: Kerning::Metrics,
            kern: 0.0,
            caps: Caps::Normal,
            ligatures: true,
            discretionary_ligatures: false,
            features: Vec::new(),
            variations: Vec::new(),
            language: None,
            style_sheet: None,
        }
    }
}

/// A character style applied to `len` UTF-8 bytes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TextRun {
    pub len: usize,
    pub style: CharStyle,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextAlign {
    #[default]
    Left,
    Center,
    Right,
    /// Justified, last line left.
    JustifyLeft,
    /// Justified, last line centered.
    JustifyCenter,
    /// Justified, last line right.
    JustifyRight,
    /// Justified including the last line.
    JustifyAll,
}

impl TextAlign {
    pub fn is_justified(self) -> bool {
        matches!(self, Self::JustifyLeft | Self::JustifyCenter | Self::JustifyRight | Self::JustifyAll)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TextDirection {
    /// From the first strong character (Unicode bidi rules).
    #[default]
    Auto,
    Ltr,
    Rtl,
}

/// Paragraph style (Photoshop's Paragraph panel).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParagraphStyle {
    pub align: TextAlign,
    pub first_line_indent_pt: f32,
    pub start_indent_pt: f32,
    pub end_indent_pt: f32,
    pub space_before_pt: f32,
    pub space_after_pt: f32,
    /// Auto leading as a multiple of the font size (Photoshop default 1.2).
    pub auto_leading: f32,
    pub direction: TextDirection,
    pub hyphenate: bool,
    /// Applied paragraph style (id in [`crate::TextStyles`]); `None` = Basic Paragraph.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style_sheet: Option<u32>,
}

impl Default for ParagraphStyle {
    fn default() -> Self {
        Self {
            align: TextAlign::Left,
            first_line_indent_pt: 0.0,
            start_indent_pt: 0.0,
            end_indent_pt: 0.0,
            space_before_pt: 0.0,
            space_after_pt: 0.0,
            auto_leading: 1.2,
            direction: TextDirection::Auto,
            hyphenate: false,
            style_sheet: None,
        }
    }
}

/// A paragraph style applied to `len` UTF-8 bytes (whole paragraphs, including the break).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParagraphRun {
    pub len: usize,
    pub style: ParagraphStyle,
}

/// Point text grows from its anchor (text-space origin = first baseline at the anchor);
/// paragraph (box) text wraps inside a rectangle; path text runs along a curve.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub enum TextShape {
    #[default]
    Point,
    /// Text-space pixels, before the layer transform.
    Box { x: f32, y: f32, width: f32, height: f32 },
    /// Type on a path: one line laid out like point text, then each glyph is placed on `path`
    /// (text-space pixels, before the layer transform) by arc length and turned to follow the
    /// curve. `start` is the arc-length offset (px) of the text origin along the path; `flip`
    /// puts the text on the other side of the path.
    Path { path: super::vector::Subpath, start: f32, flip: bool },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Orientation {
    #[default]
    Horizontal,
    /// Vertical type (tategaki): columns top to bottom, advancing right to left.
    Vertical,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum AntiAlias {
    None,
    Sharp,
    Crisp,
    Strong,
    #[default]
    Smooth,
    /// Photoshop's "Windows" (platform grayscale) anti-aliasing; rendered like `Sharp`.
    Windows,
    /// Photoshop's "Windows LCD" (platform subpixel) anti-aliasing; rendered like `Sharp` (we
    /// never produce colour-fringed subpixel text in document pixels).
    WindowsLcd,
}

/// Warp text settings (Photoshop `warp` descriptor), applied to the glyph outlines by
/// `photocraft-text` (see its `warp` module) and round-tripped through PSD.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TextWarp {
    /// `warpNone`, `warpArc`, `warpFlag`, …
    pub style: String,
    /// Bend in percent (-100..100).
    pub value: f32,
    pub horizontal_distortion: f32,
    pub vertical_distortion: f32,
    /// Horizontal (true) or vertical warp axis.
    pub horizontal: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextLayer;

    fn run(len: usize, size: f32) -> TextRun {
        TextRun { len, style: CharStyle { size_pt: size, ..Default::default() } }
    }

    #[test]
    fn runs_are_normalized_to_the_text() {
        let base = TextLayer { text: "héllo".into(), size_pt: 9.0, ..Default::default() };
        // No runs: one run from the summary fields.
        let r = base.char_runs();
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].len, r[0].style.size_pt), (6, 9.0));
        // Too short: last run stretched. Split inside 'é' snaps to the char boundary.
        let t = TextLayer { runs: vec![run(2, 10.0), run(1, 20.0)], ..base.clone() };
        let r = t.char_runs();
        assert_eq!(r.iter().map(|r| r.len).collect::<Vec<_>>(), vec![3, 3]);
        // Too long: truncated; empty runs dropped.
        let t = TextLayer { runs: vec![run(0, 1.0), run(100, 10.0), run(5, 20.0)], ..base.clone() };
        let r = t.char_runs();
        assert_eq!(r.len(), 1);
        assert_eq!((r[0].len, r[0].style.size_pt), (6, 10.0));
        // Empty text keeps the first run's style.
        let t = TextLayer { text: String::new(), runs: vec![run(0, 33.0)], ..base.clone() };
        assert_eq!(t.char_runs()[0].style.size_pt, 33.0);
        assert_eq!(t.paragraph_runs().len(), 1);
    }

    #[test]
    fn summary_follows_first_run() {
        let mut t = TextLayer {
            text: "ab".into(),
            runs: vec![TextRun { len: 2, style: CharStyle { font_family: "X".into(), size_pt: 7.0, ..Default::default() } }],
            ..Default::default()
        };
        t.sync_summary();
        assert_eq!((t.font_family.as_str(), t.size_pt), ("X", 7.0));
    }

    #[test]
    fn styles_serialize() {
        let s = CharStyle { features: vec![FontFeature { tag: "ss01".into(), value: 1 }], ..Default::default() };
        let v = serde_json::to_value(&s).unwrap();
        let back: CharStyle = serde_json::from_value(v).unwrap();
        assert_eq!(back, s);
        // Missing keys take defaults.
        let p: ParagraphStyle = serde_json::from_str(r#"{"align":"Center"}"#).unwrap();
        assert_eq!((p.align, p.auto_leading), (TextAlign::Center, 1.2));
    }
}
