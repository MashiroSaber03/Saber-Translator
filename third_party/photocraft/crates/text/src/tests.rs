use photocraft_color::{Color, PixelFormat, SampleType};
use photocraft_doc::TextLayer;
use photocraft_doc::text::{Caps, CharStyle, FontFeature, Orientation, ParagraphRun, ParagraphStyle, TextAlign, TextDirection, TextRun, TextShape};
use photocraft_geom::Affine;

use crate::{TextEngine, fonts};

fn point(text: &str, size_pt: f32) -> TextLayer {
    TextLayer { text: text.into(), font_family: "Inter".into(), size_pt, ..Default::default() }
}

#[test]
fn block_alignment_moves_glyphs_and_carets_without_moving_the_frame() {
    use photocraft_doc::text::BlockAlign;
    let mut engine = TextEngine::new();
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        let shape = TextShape::Box { x: 11.0, y: 17.0, width: 240.0, height: 240.0 };
        for size in [18.0, 48.0] {
            let mut text = TextLayer { shape: shape.clone(), orientation, overflow_visible: true, ..point("Ab\nCD", size) };
            let start = engine.layout(&text, 72.0);
            for align in [BlockAlign::Center, BlockAlign::End] {
                text.block_align = align;
                let positioned = engine.layout(&text, 72.0);
                let bounds = positioned.line_bounds().unwrap();
                let origin = if orientation == Orientation::Vertical { -251.0 } else { 17.0 };
                let factor = if align == BlockAlign::Center { 0.5 } else { 1.0 };
                assert!((bounds[1] - origin - (240.0 - (bounds[3] - bounds[1])) * factor).abs() < 0.001);
                assert_eq!(start.lines.len(), positioned.lines.len());
                let (dx, dy) = positioned.to_text(0.0, positioned.block_offset);
                for (before, after) in start.glyphs.iter().zip(&positioned.glyphs) {
                    assert_eq!(before.id, after.id);
                    assert!((after.x - before.x - dx).abs() < 0.001 && (after.y - before.y - dy).abs() < 0.001);
                }
                for (before, after) in start.caret_segment(0).iter().zip(positioned.caret_segment(0)) {
                    assert!((after.0 - before.0 - dx).abs() < 0.001 && (after.1 - before.1 - dy).abs() < 0.001);
                }
                assert_eq!(text.shape, shape);
                assert!(text.paragraphs.is_empty(), "alignment is not paragraph spacing");
            }
        }
    }
}

#[test]
fn visible_overflow_keeps_wrapping_and_the_original_frame() {
    let mut engine = TextEngine::new();
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        let shape = if orientation == Orientation::Vertical {
            TextShape::Box { x: 0.0, y: 0.0, width: 18.0, height: 100.0 }
        } else {
            TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 18.0 }
        };
        let mut text = TextLayer { shape: shape.clone(), orientation, ..point("one two three four five six seven", 36.0) };
        let clipped = engine.layout(&text, 72.0);
        text.overflow_visible = true;
        let visible = engine.layout(&text, 72.0);
        assert!(visible.lines.len() > 2);
        assert!(visible.glyphs.len() > clipped.glyphs.len());
        assert_eq!(visible.lines.last().unwrap().range.end, text.text.len());
        assert_eq!(text.shape, shape, "visible overflow must never resize the repair frame");
        let mut roomy = text.clone();
        if let TextShape::Box { width, height, .. } = &mut roomy.shape {
            if orientation == Orientation::Vertical { *width = 2000.0 } else { *height = 2000.0 }
        }
        let expected = engine.layout(&roomy, 72.0);
        assert_eq!(visible.lines.iter().map(|l| l.range.clone()).collect::<Vec<_>>(), expected.lines.iter().map(|l| l.range.clone()).collect::<Vec<_>>());
    }
}

#[test]
fn psd_visible_overflow_copy_preserves_all_glyphs_styles_and_positions() {
    let mut engine = TextEngine::new();
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        for align in [TextAlign::Left, TextAlign::Center, TextAlign::Right] {
            for inline in [100.0, 12.0] {
                let shape = if orientation == Orientation::Vertical {
                    TextShape::Box { x: 0.0, y: 0.0, width: 18.0, height: inline }
                } else {
                    TextShape::Box { x: 0.0, y: 0.0, width: inline, height: 18.0 }
                };
                let text = "Ab中😀 CD EF\nGH IJ KL";
                let t = TextLayer {
                    shape: shape.clone(),
                    orientation,
                    overflow_visible: true,
                    paragraphs: vec![ParagraphRun { len: text.len(), style: ParagraphStyle { align, ..Default::default() } }],
                    runs: vec![
                        TextRun { len: 2, style: CharStyle { size_pt: 24.0, color: Color::rgb(1.0, 0.0, 0.0), ..Default::default() } },
                        TextRun { len: text.len() - 2, style: CharStyle { size_pt: 40.0, ..Default::default() } },
                    ],
                    ..point(text, 40.0)
                };
                let layout = engine.layout(&t, 72.0);
                let exported = crate::psd::visible_overflow_for_psd(&t, &layout).unwrap();
                let psd = engine.layout(&exported, 72.0);
                assert_eq!(t.text, text);
                assert_eq!(t.shape, shape);
                assert_eq!(exported.text.replace('\n', ""), text.replace('\n', ""));
                assert!(!exported.overflow_visible);
                assert_eq!(exported.runs.iter().map(|r| r.len).sum::<usize>(), exported.text.len());
                assert_eq!(exported.paragraphs.iter().map(|r| r.len).sum::<usize>(), exported.text.len());
                assert_eq!(
                    psd.glyphs.len(),
                    layout.glyphs.len(),
                    "{orientation:?} {align:?} {inline}: {:?} {:?}\nBefore {:?}\nAfter {:?}",
                    exported.shape,
                    exported.text,
                    layout.lines,
                    psd.lines
                );
                for (a, b) in layout.glyphs.iter().zip(&psd.glyphs) {
                    assert_eq!((a.id, a.style), (b.id, b.style));
                    assert!((a.x - b.x).abs() < 0.1 && (a.y - b.y).abs() < 0.1, "{orientation:?} {align:?} {inline}: {a:?} -> {b:?}");
                }
            }
        }
    }
}

fn styled(text: &str, style: CharStyle) -> TextLayer {
    TextLayer { text: text.into(), runs: vec![TextRun { len: text.len(), style }], ..Default::default() }
}

#[test]
fn raster_preserves_hdr_and_signed_float_text_colors() {
    let mut e = TextEngine::new();
    for alpha in [1.0, 0.4] {
        let color = Color::rgba(2.5, -0.25, 0.5, alpha);
        let style = CharStyle { size_pt: 48.0, color, ..Default::default() };
        let (_, rendered) = e.render(&styled("HDR", style), 72.0, PixelFormat::RGBA32F);
        let pixels = rendered.surface.read_region(rendered.rect);
        let mut covered = 0;
        for pixel in pixels.as_chunks::<4>().0.iter().filter(|p| p[3] > 0.01) {
            covered += 1;
            for channel in 0..3 {
                assert!((pixel[channel] - color.c[channel]).abs() < 1e-5, "alpha={alpha}, pixel={pixel:?}");
            }
            assert!(pixel[3] <= alpha + 1e-6);
        }
        assert!(covered > 100);
    }
}

#[test]
fn raster_integer_text_colors_stay_bounded() {
    let mut e = TextEngine::new();
    for format in [PixelFormat::RGBA8, PixelFormat::RGBA16] {
        let style = CharStyle { size_pt: 48.0, color: Color::rgba(2.5, -0.25, 0.5, 0.4), ..Default::default() };
        let (_, rendered) = e.render(&styled("HDR", style), 72.0, format);
        let pixels = rendered.surface.read_region(rendered.rect);
        let mut covered = 0;
        for pixel in pixels.as_chunks::<4>().0.iter().filter(|p| p[3] > 0.01) {
            covered += 1;
            assert_eq!(&pixel[..2], &[1.0, 0.0]);
            assert!((pixel[2] - 0.5).abs() < 1.0 / 255.0);
            assert!(pixel[3] <= 0.4 + 1.0 / 255.0);
        }
        assert!(covered > 100);
    }
}

fn with_para(mut t: TextLayer, p: ParagraphStyle) -> TextLayer {
    t.paragraphs = vec![ParagraphRun { len: t.text.len(), style: p }];
    t
}

fn width(l: &crate::TextLayout) -> f32 {
    l.lines.iter().map(|l| l.x1 - l.x0).fold(0.0, f32::max)
}

fn alpha_sum(s: &photocraft_raster::Surface, r: photocraft_geom::Rect) -> f64 {
    let n = s.channels();
    s.read_region(r).chunks_exact(n).map(|p| f64::from(p[n - 1])).sum()
}

#[test]
fn bundled_fonts_cover_latin() {
    let mut e = TextEngine::new();
    let fams = e.fonts.families();
    assert!(fams.iter().any(|f| f == "Inter"), "{fams:?}");
    assert!(fams.iter().any(|f| f == "JetBrains Mono"), "{fams:?}");
    let l = e.layout(&point("Hello, Wörld! 0123", 12.0), 72.0);
    assert_eq!(l.glyphs.len(), "Hello, Wörld! 0123".chars().count());
    assert!(l.glyphs.iter().all(|g| g.id != 0), "no .notdef");
    assert_eq!(e.fonts.faces("Inter").len(), 3);
}

#[test]
fn registering_fonts_moves_the_generation() {
    let mut e = TextEngine::new();
    let g = fonts::generation();
    e.fonts.register_font_data(fonts::INTER_REGULAR.to_vec());
    assert!(fonts::generation() > g);
}

/// Font menus cache on the shared engine's generation: other engines registering fonts (a test's
/// throwaway one, in parallel) must not invalidate them, or the menus re-render every frame.
#[test]
fn only_the_shared_database_moves_the_shared_generation() {
    let _ = crate::shared();
    let g = fonts::shared_generation();
    TextEngine::new().fonts.register_font_data(fonts::INTER_REGULAR.to_vec());
    assert_eq!(fonts::shared_generation(), g, "an unrelated engine leaves it alone");
    let mut e = TextEngine::new();
    e.fonts.mark_shared();
    let g = fonts::shared_generation();
    e.fonts.register_font_data(fonts::INTER_REGULAR.to_vec());
    assert!(fonts::shared_generation() > g);
}

#[test]
fn literal_psd_tabs_shape_as_whitespace_without_shifting_text_offsets() {
    let mut e = TextEngine::new();
    let src = "A\tB";
    let l = e.layout(&point(src, 20.0), 72.0);
    let plain = e.layout(&point("AB", 20.0), 72.0);
    assert_eq!(l.lines.len(), 1);
    assert!(l.glyphs.iter().all(|g| g.id != 0), "a tab must not render a tofu glyph");
    assert!(width(&l) > width(&plain), "the tab reserves whitespace");
    assert!(l.clusters.iter().all(|c| c.range.end <= src.len()), "the source byte offsets stay valid");
    assert!(l.caret(2).0 > l.caret(1).0, "caret moves across the tab");
}

/// Pen position of the cluster starting at byte `at`.
fn cluster_x(l: &crate::TextLayout, at: usize) -> f32 {
    l.clusters.iter().find(|c| c.range.start == at).map(|c| c.x).unwrap()
}

/// #1881: a tab advances to Photoshop's next default tab stop (every 36 pt, from the anchor or
/// the box's left edge), not one space width.
#[test]
fn tabs_advance_to_the_next_default_tab_stop() {
    let mut e = TextEngine::new();
    let near = |a: f32, b: f32| (a - b).abs() < 0.05;
    let l = e.layout(&point("a\tb", 20.0), 72.0);
    assert!(l.glyphs.iter().all(|g| g.id != 0), "no .notdef");
    assert!(near(cluster_x(&l, 2), 36.0), "{}", cluster_x(&l, 2));
    assert!(near(l.caret(2).0, 36.0), "the caret after the tab sits on the stop");
    // Stops are in points: twice the pixels at 144 dpi.
    let l = e.layout(&point("a\tb", 20.0), 144.0);
    assert!(near(cluster_x(&l, 2), 72.0), "{}", cluster_x(&l, 2));
    // Each tab moves to the following stop; text past a stop skips to the next one.
    let l = e.layout(&point("a\t\tb", 20.0), 72.0);
    assert!(near(cluster_x(&l, 3), 72.0), "{}", cluster_x(&l, 3));
    let l = e.layout(&point("MMM\tb", 20.0), 72.0);
    assert!(cluster_x(&l, 3) > 36.0);
    assert!(near(cluster_x(&l, 4), 72.0), "{}", cluster_x(&l, 4));
    // A tab at the start of a line goes to the first stop; the line extent includes it.
    let l = e.layout(&point("\tb", 20.0), 72.0);
    assert!(near(cluster_x(&l, 1), 36.0), "{}", cluster_x(&l, 1));
    assert!(l.lines[0].x1 > 36.0);
    // Paragraph text measures from the box's left edge.
    let mut t = point("a\tb", 20.0);
    t.shape = TextShape::Box { x: 10.0, y: 0.0, width: 300.0, height: 100.0 };
    let l = e.layout(&t, 72.0);
    assert!(near(cluster_x(&l, 2), 46.0), "{}", cluster_x(&l, 2));
}

/// The tab stays a tab in the document: PSD type round-trips it unchanged.
#[test]
fn psd_round_trips_tabs() {
    for shape in [TextShape::Point, TextShape::Box { x: 0.0, y: 0.0, width: 300.0, height: 100.0 }] {
        let t = TextLayer { shape: shape.clone(), ..point("Name\tValue\n\tIndented", 20.0) };
        let back = crate::psd::text_layer_from_tysh(&crate::psd::build_tysh(&t, 72.0, None), 72.0).unwrap();
        assert_eq!(back.text, t.text, "{shape:?}");
    }
}

/// Expanded tabs never push paragraph text past the box's right edge.
#[test]
fn tabs_wrap_inside_the_box() {
    let mut e = TextEngine::new();
    let mut t = point("a\tb\tc\td\te\tf\tg", 20.0);
    t.shape = TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 1000.0 };
    let l = e.layout(&t, 72.0);
    assert!(l.lines.len() > 1, "{}", l.lines.len());
    // Lines break at the real tab widths: "c" (stop 72 px) still fits, "d" (stop 108 px) wraps.
    let line_of = |at: usize| l.clusters.iter().find(|c| c.range.start == at).map(|c| c.line).unwrap();
    assert_eq!((line_of(4), line_of(6)), (0, 1));
    assert!((cluster_x(&l, 6) - 0.0).abs() < 0.05, "a wrapped line starts at the box edge");
    for c in &l.clusters {
        let ch = &t.text[c.range.clone()];
        if ch != "\t" {
            assert!(c.x + c.advance <= 100.0 + 1e-3, "{ch:?} at {} overflows", c.x);
        }
    }
}

/// Right-aligned and centred point text keep their alignment around the expanded tab.
#[test]
fn tabs_keep_point_text_alignment() {
    let mut e = TextEngine::new();
    let r = e.layout(&with_para(point("a\tb", 20.0), ParagraphStyle { align: TextAlign::Right, ..Default::default() }), 72.0);
    assert!(r.lines[0].x1.abs() < 0.05, "{:?}", r.lines[0]);
    let b = r.clusters.iter().find(|c| c.range.start == 2).unwrap();
    assert!((b.x + b.advance).abs() < 0.05, "b ends at the anchor: {b:?}");
    assert!(r.lines[0].x0 < -36.0);
    // Paragraph text: the line still ends at the box's right edge; a trailing tab is
    // whitespace outside the line's extent.
    for text in ["a\tb", "a\tb\t"] {
        let t = with_para(
            TextLayer { shape: TextShape::Box { x: 0.0, y: 0.0, width: 200.0, height: 100.0 }, ..point(text, 20.0) },
            ParagraphStyle { align: TextAlign::Right, ..Default::default() },
        );
        let l = e.layout(&t, 72.0);
        let b = l.clusters.iter().find(|c| c.range.start == 2).unwrap();
        assert!((b.x + b.advance - 200.0).abs() < 0.5, "{text:?}: {b:?}");
        assert!((l.lines[0].x1 - 200.0).abs() < 0.5, "{text:?}: {:?}", l.lines[0]);
    }
}

#[test]
fn metrics_are_stable_and_scale_with_dpi() {
    let mut e = TextEngine::new();
    let a = e.layout(&point("Hamburgefonstiv", 12.0), 72.0);
    let w = width(&a);
    // Inter 12 px: stable to a tenth of a pixel across runs.
    let again = width(&e.layout(&point("Hamburgefonstiv", 12.0), 72.0));
    assert_eq!(w, again);
    assert!((80.0..110.0).contains(&w), "{w}");
    let b = e.layout(&point("Hamburgefonstiv", 12.0), 144.0);
    assert!((width(&b) - 2.0 * w).abs() < 0.05, "{} vs {}", width(&b), 2.0 * w);
    // Point text: first baseline at the anchor.
    assert_eq!(a.lines[0].baseline, 0.0);
    assert!(a.lines[0].ascent > 8.0 && a.lines[0].ascent < 13.0);
}

#[test]
fn small_caps_are_synthesized_when_the_font_has_no_small_caps_feature() {
    let style = CharStyle { font_family: "Inter".into(), size_pt: 40.0, caps: Caps::SmallCaps, ..Default::default() };
    let mut engine = TextEngine::new();
    let small = engine.layout(&styled("aA", style), 72.0);
    let upper = engine.layout(&styled("AA", CharStyle { font_family: "Inter".into(), size_pt: 40.0, ..Default::default() }), 72.0);
    assert_eq!(small.glyphs.len(), 2);
    assert_eq!(small.glyphs[0].id, upper.glyphs[0].id, "lowercase uses the uppercase glyph");
    assert_eq!(small.glyphs[1].id, upper.glyphs[1].id, "uppercase remains uppercase");
    let size = |layout: &crate::TextLayout, glyph: usize| layout.faces[layout.glyphs[glyph].face as usize].size_px;
    assert!((size(&small, 0) - 28.0).abs() < 0.01, "{}", size(&small, 0));
    assert!((size(&small, 1) - 40.0).abs() < 0.01, "{}", size(&small, 1));
}

#[test]
fn auto_and_explicit_leading() {
    let mut e = TextEngine::new();
    let l = e.layout(&point("one\ntwo\rthree", 10.0), 72.0);
    assert_eq!(l.lines.len(), 3);
    assert!((l.lines[1].baseline - 12.0).abs() < 1e-3, "auto leading = 1.2 × size");
    assert!((l.lines[2].baseline - 24.0).abs() < 1e-3);
    assert_eq!(&"one\ntwo\rthree"[l.lines[2].range.clone()], "three");
    let t = styled("a\nb", CharStyle { size_pt: 10.0, leading_pt: Some(30.0), ..Default::default() });
    let l = e.layout(&t, 144.0);
    assert!((l.lines[1].baseline - 60.0).abs() < 1e-3, "{}", l.lines[1].baseline);
    // Space before/after in points.
    let t = with_para(point("a\nb", 10.0), ParagraphStyle { space_after_pt: 5.0, ..Default::default() });
    let l = e.layout(&t, 72.0);
    assert!((l.lines[1].baseline - 17.0).abs() < 1e-3, "{}", l.lines[1].baseline);
}

#[test]
fn box_text_wraps_inside_width() {
    let mut e = TextEngine::new();
    let mut t = point("The quick brown fox jumps over the lazy dog again and again", 12.0);
    t.shape = TextShape::Box { x: 10.0, y: 20.0, width: 100.0, height: 1000.0 };
    let l = e.layout(&t, 72.0);
    assert!(l.lines.len() >= 3, "{}", l.lines.len());
    for line in &l.lines {
        assert!(line.x0 >= 10.0 - 1e-3 && line.x1 <= 110.0 + 1e-3, "{line:?}");
    }
    // First baseline = top + ascender height ('d'), a bit less than the hhea ascent.
    let drop = l.lines[0].baseline - 20.0;
    assert!(drop > 0.65 * 12.0 && drop < l.lines[0].ascent, "{drop}");
    // Lines cover the text in order.
    assert_eq!(l.lines[0].range.start, 0);
    for w in l.lines.windows(2) {
        assert!(w[1].range.start >= w[0].range.end);
    }
    // Overflowing lines are hidden, like Photoshop.
    t.shape = TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 30.0 };
    let l2 = e.layout(&t, 72.0);
    assert_eq!(l2.lines.len(), 2);
}

#[test]
fn alignment_point_and_box() {
    let mut e = TextEngine::new();
    let c = e.layout(&with_para(point("Centered", 20.0), ParagraphStyle { align: TextAlign::Center, ..Default::default() }), 72.0);
    let ln = &c.lines[0];
    assert!((ln.x0 + ln.x1).abs() < 0.01, "centered on anchor: {ln:?}");
    let r = e.layout(&with_para(point("Right", 20.0), ParagraphStyle { align: TextAlign::Right, ..Default::default() }), 72.0);
    assert!(r.lines[0].x1.abs() < 0.01 && r.lines[0].x0 < -10.0);

    let mut t = point("aaa bbb ccc ddd eee fff ggg hhh iii jjj kkk", 12.0);
    t.shape = TextShape::Box { x: 0.0, y: 0.0, width: 120.0, height: 500.0 };
    let t = with_para(t, ParagraphStyle { align: TextAlign::JustifyLeft, ..Default::default() });
    let l = e.layout(&t, 72.0);
    assert!(l.lines.len() > 1);
    let first = &l.lines[0];
    // Justified line: last glyph ends at the right edge (within a pixel).
    let last_glyph_x = l.glyphs.iter().filter(|g| (g.y - first.baseline).abs() < 1e-3).map(|g| g.x).fold(f32::MIN, f32::max);
    assert!(last_glyph_x > 110.0, "{last_glyph_x}");
    let right = with_para(
        TextLayer { shape: TextShape::Box { x: 0.0, y: 0.0, width: 200.0, height: 100.0 }, ..point("end", 12.0) },
        ParagraphStyle { align: TextAlign::Right, ..Default::default() },
    );
    let l = e.layout(&right, 72.0);
    assert!((l.lines[0].x1 - 200.0).abs() < 0.5, "{:?}", l.lines[0]);
}

#[test]
fn rtl_text_is_reordered() {
    let mut e = TextEngine::new();
    // Hebrew isn't in the bundled fonts, but bidi still resolves (glyphs may be .notdef).
    let text = "abc שלום def";
    let l = e.layout(&point(text, 12.0), 72.0);
    let heb: Vec<_> = l.clusters.iter().filter(|c| text[c.range.clone()].chars().all(|ch| ('\u{0590}'..='\u{05FF}').contains(&ch))).collect();
    assert_eq!(heb.len(), 4);
    assert!(heb.iter().all(|c| c.rtl));
    // Visual order: the first logical Hebrew letter is rightmost among them.
    let first = heb.iter().min_by_key(|c| c.range.start).unwrap();
    assert!(heb.iter().all(|c| c.x <= first.x + 1e-3));
    // Forced RTL paragraph: the Latin run ends up to the right of the Hebrew.
    let t = with_para(point("שלום abc", 12.0), ParagraphStyle { direction: TextDirection::Rtl, ..Default::default() });
    let l = e.layout(&t, 72.0);
    let a = l.clusters.iter().find(|c| &"שלום abc"[c.range.clone()] == "a").unwrap();
    let shin = l.clusters.iter().find(|c| c.range.start == 0).unwrap();
    assert!(a.x < shin.x, "abc left of the Hebrew in an RTL paragraph");
    assert!(l.clusters.iter().all(|c| c.range.end <= "שלום abc".len()));
}

/// Builds a TrueType collection from standalone fonts (table offsets rebased).
fn make_ttc(fonts: &[&[u8]]) -> Vec<u8> {
    let header_len = 12 + 4 * fonts.len();
    let mut out = Vec::new();
    out.extend_from_slice(b"ttcf");
    out.extend_from_slice(&0x0001_0000u32.to_be_bytes());
    out.extend_from_slice(&(fonts.len() as u32).to_be_bytes());
    let mut offsets = Vec::new();
    let mut body: Vec<u8> = Vec::new();
    for f in fonts {
        while !(header_len + body.len()).is_multiple_of(4) {
            body.push(0);
        }
        let base = (header_len + body.len()) as u32;
        offsets.push(base);
        let mut copy = f.to_vec();
        let n = u16::from_be_bytes([copy[4], copy[5]]) as usize;
        for i in 0..n {
            let at = 12 + i * 16 + 8;
            let off = u32::from_be_bytes(copy[at..at + 4].try_into().unwrap());
            copy[at..at + 4].copy_from_slice(&(off + base).to_be_bytes());
        }
        body.extend(copy);
    }
    for o in offsets {
        out.extend_from_slice(&o.to_be_bytes());
    }
    out.extend(body);
    out
}

#[test]
fn truetype_collections_load() {
    let ttc = make_ttc(&[fonts::BUNDLED[0].1.as_slice(), fonts::BUNDLED[3].1.as_slice()]);
    assert_eq!(fonts::face_count(&ttc), 2);
    let mut db = fonts::FontDb::new();
    // A fresh DB without the bundled mono font would miss it; register the collection anyway
    // and check both faces are indexed (families already present gain faces).
    let before = db.faces("JetBrains Mono").len();
    let names = db.register_font_data(ttc);
    assert!(names.iter().any(|n| n == "Inter") && names.iter().any(|n| n == "JetBrains Mono"), "{names:?}");
    assert_eq!(db.faces("JetBrains Mono").len(), before + 1);
    // Layout with the TTC-backed face works.
    let mut e = TextEngine { fonts: db, layouter: crate::layout::Layouter::new() };
    let t = styled("mono", CharStyle { font_family: "JetBrains Mono".into(), ..Default::default() });
    let l = e.layout(&t, 72.0);
    assert!(l.glyphs.iter().all(|g| g.id != 0));
}

#[test]
fn opentype_features_and_tracking() {
    let mut e = TextEngine::new();
    let ids = |e: &mut TextEngine, feats: Vec<FontFeature>, lig: bool| {
        let t = styled("a0", CharStyle { features: feats, ligatures: lig, ..Default::default() });
        e.layout(&t, 72.0).glyphs.iter().map(|g| g.id).collect::<Vec<_>>()
    };
    let plain = ids(&mut e, vec![], true);
    let alt = ids(&mut e, vec![FontFeature { tag: "zero".into(), value: 1 }], true);
    assert_ne!(plain, alt, "Inter's `zero` feature selects the slashed zero");
    let a = width(&e.layout(&styled("tracking", CharStyle::default()), 72.0));
    let b = width(&e.layout(&styled("tracking", CharStyle { tracking: 100.0, ..Default::default() }), 72.0));
    // 100/1000 em at 12 px per letter (8 letters; trailing spacing counts too).
    assert!((b - a - 8.0 * 1.2).abs() < 1.3, "{a} → {b}");
}

#[test]
fn raster_coverage_scales_and_depths_agree() {
    let mut e = TextEngine::new();
    let t = |size: f32| TextLayer { transform: Affine::translate(10.0, 50.0), ..point("Ink", size) };
    let (_, r12) = e.render(&t(12.0), 72.0, PixelFormat::RGBA8);
    let (_, r24) = e.render(&t(24.0), 72.0, PixelFormat::RGBA8);
    let s12 = alpha_sum(&r12.surface, r12.rect);
    let s24 = alpha_sum(&r24.surface, r24.rect);
    assert!(s12 > 20.0, "{s12}");
    let ratio = s24 / s12;
    assert!((3.6..4.4).contains(&ratio), "ink ∝ size²: {ratio}");
    // Placement: anchored at (10, 50) baseline.
    assert!(r12.rect.x0 >= 8 && r12.rect.x0 <= 11 && r12.rect.y1 <= 53 && r12.rect.y1 >= 50, "{:?}", r12.rect);
    for fmt in [PixelFormat::RGBA16, PixelFormat::RGBA32F, PixelFormat::GRAYA8, PixelFormat::CMYKA8] {
        let (_, r) = e.render(&t(12.0), 72.0, fmt);
        assert_eq!(r.rect, r12.rect);
        let s = alpha_sum(&r.surface, r.rect);
        assert!((s - s12).abs() / s12 < 0.01, "{fmt:?}: {s} vs {s12}");
    }
    // Deterministic.
    let (_, again) = e.render(&t(12.0), 72.0, PixelFormat::RGBA8);
    assert_eq!(again.surface, r12.surface);
    // Golden-ish total ink for "Ink" in Inter 12 px (±3%).
    assert!((s12 - GOLDEN_INK_12).abs() / GOLDEN_INK_12 < 0.03, "ink sum {s12}");
}

/// Sum of alpha for "Ink", Inter Regular 12 px (measured once; guards rasterizer regressions).
const GOLDEN_INK_12: f64 = 44.4;

#[test]
fn styles_change_pixels() {
    let mut e = TextEngine::new();
    let base = CharStyle { size_pt: 30.0, ..Default::default() };
    let ink = |e: &mut TextEngine, s: CharStyle| {
        let (_, r) = e.render(&TextLayer { transform: Affine::translate(5.0, 40.0), ..styled("Hi", s) }, 72.0, PixelFormat::RGBA8);
        (alpha_sum(&r.surface, r.rect), r.rect)
    };
    let (plain, prect) = ink(&mut e, base.clone());
    let (bold, _) = ink(&mut e, CharStyle { faux_bold: true, ..base.clone() });
    assert!(bold > plain * 1.1, "{plain} → {bold}");
    let (under, urect) = ink(&mut e, CharStyle { underline: true, ..base.clone() });
    assert!(under > plain && urect.y1 > prect.y1);
    let (_, irect) = ink(&mut e, CharStyle { faux_italic: true, ..base.clone() });
    assert!(irect.x1 > prect.x1, "slanted top extends right");
    let (_, srect) = ink(&mut e, CharStyle { baseline_shift_pt: 10.0, ..base.clone() });
    assert_eq!(srect.y0, prect.y0 - 10);
    let (_, hrect) = ink(&mut e, CharStyle { horizontal_scale: 2.0, ..base.clone() });
    assert!(hrect.width() as f32 > prect.width() as f32 * 1.7);
}

/// #4130: Faux bold at 72 dpi looked pixelated because the offset copies were summed into one
/// coverage buffer, saturating every antialiased edge pixel. Edges must stay antialiased and the
/// ink must grow only by the dilation (about perimeter x radius), not by edge saturation.
#[test]
fn faux_bold_keeps_antialiased_edges_at_small_sizes() {
    let mut e = TextEngine::new();
    let base = CharStyle { size_pt: 12.0, ..Default::default() };
    let stats = |e: &mut TextEngine, s: CharStyle| {
        let (_, r) = e.render(&TextLayer { transform: Affine::translate(5.0, 20.0), ..styled("Bold Test", s) }, 72.0, PixelFormat::RGBA8);
        let (w, h) = (r.rect.width() as usize, r.rect.height() as usize);
        let mut partial = 0usize;
        for y in 0..h {
            for x in 0..w {
                let a = alpha_sum(
                    &r.surface,
                    photocraft_geom::Rect::new(r.rect.x0 + x as i32, r.rect.y0 + y as i32, r.rect.x0 + x as i32 + 1, r.rect.y0 + y as i32 + 1),
                );
                if a > 0.05 && a < 0.95 {
                    partial += 1;
                }
            }
        }
        (alpha_sum(&r.surface, r.rect), partial)
    };
    let (plain_ink, plain_partial) = stats(&mut e, base.clone());
    let (bold_ink, bold_partial) = stats(&mut e, CharStyle { faux_bold: true, ..base });
    assert!(bold_ink > plain_ink, "bold must add ink");
    // Weight: the old summed-coverage faux bold (before #4130) measured 282.7 total ink for this
    // text at 12 px (plain 138.0). The antialiased version must stay close to that weight.
    const MAIN_BOLD_INK: f64 = 282.7;
    assert!(bold_ink >= MAIN_BOLD_INK * 0.85, "stems too thin: {bold_ink} vs old {MAIN_BOLD_INK}");
    assert!(bold_ink <= MAIN_BOLD_INK * 1.15, "stems too heavy: {bold_ink} vs old {MAIN_BOLD_INK}");
    assert!(bold_partial * 10 >= plain_partial * 8, "edges stay antialiased: {plain_partial} -> {bold_partial}");
}

/// #3705: Faux bold dilation must scale with horizontal and vertical glyph scaling.
#[test]
fn faux_bold_scales_with_glyph_scaling() {
    let mut e = TextEngine::new();
    let base = CharStyle { size_pt: 40.0, ..Default::default() };
    let ink = |e: &mut TextEngine, s: CharStyle| {
        let (_, r) = e.render(&TextLayer { transform: Affine::translate(5.0, 50.0), ..styled("Hi", s) }, 72.0, PixelFormat::RGBA8);
        (alpha_sum(&r.surface, r.rect), r.rect)
    };
    let (_plain, prect) = ink(&mut e, base.clone());
    let (_bold, brect) = ink(&mut e, CharStyle { faux_bold: true, ..base.clone() });
    let unscaled_extra_w = brect.width() - prect.width();
    let unscaled_extra_h = brect.height() - prect.height();

    // With 2x horizontal scale, the horizontal dilation scales with the glyph:
    let hscaled = CharStyle { size_pt: 40.0, horizontal_scale: 2.0, ..Default::default() };
    let (_, phrect) = ink(&mut e, hscaled.clone());
    let (_, bhrect) = ink(&mut e, CharStyle { faux_bold: true, ..hscaled });
    let hscaled_extra_w = bhrect.width() - phrect.width();
    assert!(
        hscaled_extra_w >= unscaled_extra_w * 2 - 1 && hscaled_extra_w <= unscaled_extra_w * 2 + 1,
        "hscaled extra width {hscaled_extra_w} vs unscaled {unscaled_extra_w}"
    );

    // With 2x vertical scale, the vertical dilation scales with the glyph:
    let vscaled = CharStyle { size_pt: 40.0, vertical_scale: 2.0, ..Default::default() };
    let (_, pvrect) = ink(&mut e, vscaled.clone());
    let (_, bvrect) = ink(&mut e, CharStyle { faux_bold: true, ..vscaled });
    let vscaled_extra_h = bvrect.height() - pvrect.height();
    assert!(
        vscaled_extra_h >= unscaled_extra_h * 2 - 1 && vscaled_extra_h <= unscaled_extra_h * 2 + 1,
        "vscaled extra height {vscaled_extra_h} vs unscaled {unscaled_extra_h}"
    );
}

#[test]
fn multicolor_runs() {
    let mut e = TextEngine::new();
    let red = CharStyle { size_pt: 40.0, color: Color::rgb(1.0, 0.0, 0.0), ..Default::default() };
    let blue = CharStyle { color: Color::rgb(0.0, 0.0, 1.0), ..red.clone() };
    let t = TextLayer {
        text: "HH".into(),
        runs: vec![TextRun { len: 1, style: red }, TextRun { len: 1, style: blue }],
        transform: Affine::translate(0.0, 40.0),
        ..Default::default()
    };
    let (l, r) = e.render(&t, 72.0, PixelFormat::RGBA8);
    assert_eq!(l.glyphs.len(), 2);
    let px = r.surface.read_region(r.rect);
    let opaque: Vec<&[f32; 4]> = px.as_chunks::<4>().0.iter().filter(|p| p[3] > 0.99).collect();
    assert!(opaque.iter().any(|p| p[0] > 0.99 && p[2] < 0.01));
    assert!(opaque.iter().any(|p| p[2] > 0.99 && p[0] < 0.01));
    // CMYK target keeps colour in the document model.
    let (_, rc) = e.render(&t, 72.0, PixelFormat::CMYKA8);
    assert!(rc.surface.read_region(rc.rect).as_chunks::<5>().0.iter().any(|p| p[4] > 0.99 && p[1] > 0.9 && p[2] > 0.9));
}

#[test]
fn hit_test_and_caret() {
    let mut e = TextEngine::new();
    let l = e.layout(&point("abc\ndef", 20.0), 72.0);
    assert_eq!(l.hit_test(-5.0, -5.0), 0);
    assert_eq!(l.hit_test(1000.0, -5.0), 3);
    assert_eq!(l.hit_test(-5.0, 24.0), 4);
    let (x, top, bottom) = l.caret(1);
    assert!(x > 0.0 && top < 0.0 && bottom > 0.0);
    let (x2, ..) = l.caret(2);
    assert!(x2 > x);
    assert_eq!(l.caret(3).0, l.lines[0].x1);
}

#[test]
fn wrapped_and_forced_line_starts_use_the_following_line() {
    use crate::layout::{byte_index, line_edge, line_index};

    let mut engine = TextEngine::new();
    let mut wrapped = point("ab cd", 12.0);
    wrapped.shape = TextShape::Box { x: 0.0, y: 0.0, width: 20.0, height: 70.0 };
    let layout = engine.layout(&wrapped, 72.0);
    assert!(layout.lines.len() >= 2, "paragraph must wrap: {:?}", layout.lines);
    let boundary = layout.lines[1].range.start;
    assert_eq!(layout.lines[0].range.end, boundary);
    let index = wrapped.text[..boundary].chars().count();
    assert_eq!(line_index(&layout, byte_index(&wrapped.text, index)), 1);
    assert_eq!(line_edge(&layout, &wrapped.text, index, false), index);
    assert_eq!(line_edge(&layout, &wrapped.text, index, true), wrapped.text[..layout.lines[1].range.end].chars().count());
    // The previous line's right edge shares the caret index, but a click there
    // must still report the previous visual line.
    let first = &layout.lines[0];
    let (x, y) = layout.to_text(first.x1, first.baseline);
    assert_eq!(crate::layout::hit_char(&layout, &wrapped.text, x, y).1, 0);

    let text = "a\u{3}b";
    let forced = engine.layout(&point(text, 12.0), 72.0);
    assert_eq!(forced.lines.len(), 2);
    let start = forced.lines[1].range.start;
    let index = text[..start].chars().count();
    assert_eq!(line_index(&forced, start), 1);
    assert_eq!(line_edge(&forced, text, index, false), index);
    assert_eq!(line_edge(&forced, text, index, true), text.chars().count());

    // A normal paragraph break has distinct boundaries and retains its line.
    let text = "a\nb";
    let paragraph = engine.layout(&point(text, 12.0), 72.0);
    assert_eq!(line_index(&paragraph, paragraph.lines[0].range.end), 0);
    assert_eq!(line_index(&paragraph, paragraph.lines[1].range.start), 1);
    assert_eq!(line_index(&paragraph, text.len()), 1);
}

#[test]
fn empty_text_and_empty_lines() {
    let mut e = TextEngine::new();
    let l = e.layout(&point("", 12.0), 72.0);
    assert_eq!(l.lines.len(), 1);
    assert!(l.glyphs.is_empty());
    let l = e.layout(&point("a\n\nb", 12.0), 72.0);
    assert_eq!(l.lines.len(), 3);
    assert!((l.lines[2].baseline - 2.0 * 14.4).abs() < 1e-3);
    let (_, r) = e.render(&point("", 12.0), 72.0, PixelFormat::RGBA8);
    assert_eq!(r.rect.width(), 0);
}

/// U+0003 is a forced line break inside a paragraph (as PSD type stores Shift+Return): a new
/// line with the paragraph's leading and no paragraph spacing, never a missing-glyph box.
#[test]
fn forced_line_break_starts_a_line_in_the_same_paragraph() {
    let mut e = TextEngine::new();
    let text = "one\u{3}two";
    let spaced = ParagraphStyle { space_before_pt: 5.0, space_after_pt: 5.0, ..Default::default() };
    let l = e.layout(&with_para(point(text, 10.0), spaced.clone()), 72.0);
    assert_eq!(l.lines.len(), 2, "{:?}", l.lines);
    assert_eq!(text[l.lines[0].range.clone()].trim_end_matches('\u{3}'), "one");
    assert_eq!(&text[l.lines[1].range.clone()], "two");
    assert_eq!((l.lines[0].paragraph, l.lines[1].paragraph), (0, 0));
    assert!((l.lines[1].baseline - 12.0).abs() < 1e-3, "auto leading, no paragraph spacing: {}", l.lines[1].baseline);
    assert!(l.glyphs.iter().all(|g| g.id != 0), "no .notdef for the break");
    // A paragraph break in the same place adds the spacing.
    let p = e.layout(&with_para(point("one\ntwo", 10.0), spaced), 72.0);
    assert!((p.lines[1].baseline - 22.0).abs() < 1e-3, "{}", p.lines[1].baseline);
    // Centred lines are centred on their own, and box text breaks there too.
    let c = e.layout(&with_para(point("a\u{3}wide line", 20.0), ParagraphStyle { align: TextAlign::Center, ..Default::default() }), 72.0);
    assert_eq!(c.lines.len(), 2);
    assert!((c.lines[0].x0 + c.lines[0].x1).abs() < 0.01 && (c.lines[1].x0 + c.lines[1].x1).abs() < 0.01, "{:?}", c.lines);
    let mut b = point("short\u{3}next", 12.0);
    b.shape = TextShape::Box { x: 0.0, y: 0.0, width: 500.0, height: 500.0 };
    assert_eq!(e.layout(&b, 72.0).lines.len(), 2);
    // Rendered at every depth: two rows of ink, and nothing after the first line's text.
    for fmt in [PixelFormat::RGBA8, PixelFormat::RGBA16, PixelFormat::RGBA32F] {
        let (l, r) = e.render(&TextLayer { transform: Affine::translate(0.0, 40.0), ..point(text, 20.0) }, 72.0, fmt);
        let row = |i: usize| {
            let ln = &l.lines[i];
            photocraft_geom::Rect::new(-2, (ln.baseline - ln.ascent + 40.0) as i32, 200, (ln.baseline + 40.0) as i32)
        };
        assert!(alpha_sum(&r.surface, row(0)) > 1.0 && alpha_sum(&r.surface, row(1)) > 1.0, "{fmt:?}");
        let one_end = l.lines[0].x1.ceil() as i32 + 2;
        assert!(alpha_sum(&r.surface, photocraft_geom::Rect::new(one_end, row(0).y0, 200, row(0).y1)) < 1e-3, "{fmt:?}: ink after the first line's text");
    }
}

#[test]
fn hyphenated_postscript_names_preserve_family_prefixes() {
    for (ps, family, weight, italic) in
        [("FOT-NewRodinPro-Regular", "FOT-New Rodin Pro", 400, false), ("Vendor-FamilyPro-BoldIt", "Vendor-Family Pro", 700, true)]
    {
        let guessed = fonts::guess_from_postscript(ps);
        assert_eq!((guessed.family.as_str(), guessed.weight, guessed.italic), (family, weight, italic));
    }
}

#[test]
fn hyphenated_postscript_style_suffix_keeps_weight_and_slant() {
    for (ps, family, weight, italic) in [
        ("MyriadPro-Bold-It", "Myriad Pro", 700, true),
        ("FOT-NewRodinPro-Bold-It", "FOT-New Rodin Pro", 700, true),
        ("Vendor-Bold-Regular", "Vendor-Bold", 400, false),
    ] {
        let guessed = fonts::guess_from_postscript(ps);
        assert_eq!((guessed.family.as_str(), guessed.weight, guessed.italic), (family, weight, italic));
    }
}

#[test]
fn postscript_names() {
    let g = fonts::guess_from_postscript("MyriadPro-BoldIt");
    assert_eq!((g.family.as_str(), g.weight, g.italic), ("Myriad Pro", 700, true));
    let g = fonts::guess_from_postscript("TimesNewRomanPSMT");
    assert_eq!((g.family.as_str(), g.weight, g.italic), ("Times New Roman", 400, false));
    let g = fonts::guess_from_postscript("Arial-BoldMT");
    assert_eq!((g.family.as_str(), g.weight), ("Arial", 700));
    let mut db = fonts::FontDb::new();
    let r = db.resolve_postscript("Inter-SemiBold");
    assert_eq!((r.family.as_str(), r.weight, r.exact), ("Inter", 600, true));
    // Unknown fonts fall back gracefully in layout.
    let mut e = TextEngine::new();
    let t = styled("x", CharStyle { font_family: "Nonexistent Sans".into(), postscript_name: Some("Nope-Bold".into()), ..Default::default() });
    assert!(e.layout(&t, 72.0).glyphs[0].id != 0);
}

#[test]
fn depth_is_respected() {
    let mut e = TextEngine::new();
    let (_, r) = e.render(&TextLayer { transform: Affine::translate(2.0, 20.0), ..point("a", 20.0) }, 72.0, PixelFormat::RGBA16);
    assert_eq!(r.surface.format().sample, SampleType::U16);
    assert!(r.surface.format().alpha);
}

/// Visual check: `PHOTOCRAFT_TEXT_DUMP=/tmp/t.ppm cargo test -p photocraft-text dump -- --ignored`.
#[test]
#[ignore]
fn dump_sample() {
    use photocraft_doc::text::{Caps, TextAlign};
    let Some(path) = std::env::var_os("PHOTOCRAFT_TEXT_DUMP") else {
        return;
    };
    let mut e = if std::env::var_os("PHOTOCRAFT_TEXT_SYSTEM").is_some() { TextEngine::with_system_fonts() } else { TextEngine::new() };
    let s = CharStyle { size_pt: 28.0, ..Default::default() };
    let text =
        "Photocraft Type\nBold faux, italic faux, underline\nשלום عربي mixed ✓\nJustified paragraph text wraps inside the box nicely and evenly across lines.";
    let runs = vec![
        TextRun { len: 16, style: CharStyle { size_pt: 40.0, color: Color::rgb(0.1, 0.3, 0.9), caps: Caps::Normal, ..s.clone() } },
        TextRun { len: 5, style: CharStyle { faux_bold: true, ..s.clone() } },
        TextRun { len: 13, style: CharStyle { faux_italic: true, color: Color::rgb(0.8, 0.1, 0.1), ..s.clone() } },
        TextRun { len: 11, style: CharStyle { underline: true, strikethrough: false, ..s.clone() } },
        TextRun { len: 1000, style: CharStyle { size_pt: 20.0, tracking: 20.0, ..s.clone() } },
    ];
    let t = TextLayer {
        text: text.into(),
        runs,
        paragraphs: vec![
            ParagraphRun { len: 45, style: ParagraphStyle { align: TextAlign::Left, ..Default::default() } },
            ParagraphRun { len: 1000, style: ParagraphStyle { align: TextAlign::JustifyLeft, space_before_pt: 6.0, ..Default::default() } },
        ],
        shape: TextShape::Box { x: 0.0, y: 0.0, width: 560.0, height: 400.0 },
        transform: Affine::translate(20.0, 20.0),
        ..Default::default()
    };
    let (_, r) = e.render(&t, 72.0, PixelFormat::RGBA8);
    let (w, h) = (600u32, 360u32);
    let mut out = format!("P6 {w} {h} 255\n").into_bytes();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let p = r.surface.pixel(x, y);
            let a = p[3];
            for c in &p[..3] {
                out.push(((c * a + (1.0 - a)) * 255.0).round() as u8);
            }
        }
    }
    std::fs::write(path, out).unwrap();
}

/// Variable-font axes reach the outlines (uses a system variable font when one is present).
#[test]
fn variable_font_axes() {
    let candidates =
        ["/System/Library/Fonts/SFNS.ttf", "/usr/share/fonts/truetype/noto/NotoSans-VariableFont_wdth,wght.ttf", "C:\\Windows\\Fonts\\bahnschrift.ttf"];
    let Some(bytes) = candidates.iter().find_map(|p| std::fs::read(p).ok()) else {
        return;
    };
    let mut e = TextEngine::new();
    let fams = e.fonts.register_font_data(bytes);
    let Some(fam) = fams.first().cloned() else {
        return;
    };
    if !e.fonts.faces(&fam).iter().any(|f| f.axes.iter().any(|a| a.0 == "wght")) {
        return;
    }
    let ink = |e: &mut TextEngine, w: f32| {
        let s = CharStyle {
            font_family: fam.clone(),
            size_pt: 40.0,
            variations: vec![photocraft_doc::text::FontVariation { axis: "wght".into(), value: w }],
            ..Default::default()
        };
        let (_, r) = e.render(&TextLayer { transform: Affine::translate(0.0, 50.0), ..styled("Weight", s) }, 72.0, PixelFormat::RGBA8);
        alpha_sum(&r.surface, r.rect)
    };
    let light = ink(&mut e, 200.0);
    let heavy = ink(&mut e, 800.0);
    assert!(heavy > light * 1.5, "{light} → {heavy}");
}

#[test]
fn warp_bends_rendered_text_and_outlines() {
    use photocraft_doc::text::TextWarp;
    let mut e = TextEngine::new();
    let mut t = point("WARPED TEXT", 24.0);
    t.transform = Affine::translate(20.0, 60.0);
    let fmt = PixelFormat::RGBA8;
    let (_, flat) = e.render(&t, 72.0, fmt);
    t.warp = Some(TextWarp { style: "warpArc".into(), value: 60.0, horizontal: true, ..Default::default() });
    let (layout, arced) = e.render(&t, 72.0, fmt);
    // An arc pushes the ends down: the warped ink is taller and pixels differ.
    assert!(arced.rect.height() > flat.rect.height() + 10, "{:?} vs {:?}", arced.rect, flat.rect);
    assert!(alpha_sum(&arced.surface, arced.rect) > 0.0);
    // Outlines follow the same warp: the first glyph's outline sits lower than the middle one's.
    let warp = crate::render::layout_warp(&layout, t.warp.as_ref());
    let outs = crate::render::outlines(&layout, &t.transform, warp.as_ref());
    assert_eq!(outs.len(), layout.glyphs.iter().filter(|g| g.id != 0).count() - 1, "space has no outline");
    let low_y = |els: &Vec<crate::render::PathEl>| {
        els.iter()
            .filter_map(|e| match e {
                crate::render::PathEl::MoveTo(p) | crate::render::PathEl::LineTo(p) => Some(p[1]),
                _ => None,
            })
            .fold(f64::MIN, f64::max)
    };
    assert!(low_y(&outs[0]) > low_y(&outs[outs.len() / 2]) + 5.0);
    // Unwarped outlines stay within the flat raster's bounds.
    let flat_outs = crate::render::outlines(&layout, &t.transform, None);
    let r = flat.rect;
    for els in &flat_outs {
        for el in els {
            if let crate::render::PathEl::MoveTo(p) | crate::render::PathEl::LineTo(p) = el {
                assert!(p[0] >= f64::from(r.x0) && p[0] <= f64::from(r.x1) && p[1] >= f64::from(r.y0) && p[1] <= f64::from(r.y1));
            }
        }
    }
}

/// `Txt2` as Photoshop writes it (a bare `/key value` sequence; text objects at `/1 /1`, style
/// runs at `/0 /6 /0`, auto-kern mode at `/11`), for the layer text `text`.
fn txt2_with_modes(text: &str, modes: &[(usize, i64)]) -> Vec<u8> {
    txt2_with_styles(text, &modes.iter().map(|(len, m)| (*len, format!("/11 {m}"))).collect::<Vec<_>>())
}

/// [`txt2_with_modes`] with each style run's keys given as EngineData text (`"/11 1 /85 2"`).
fn txt2_with_styles(text: &str, styles: &[(usize, String)]) -> Vec<u8> {
    let mut v = b"\n\n/98 << /0 14 >> /0 << >> /1 << /1 [ << /0 << /0 (\xfe\xff".to_vec();
    for u in format!("{text}\r").encode_utf16() {
        let b = u.to_be_bytes();
        for x in b {
            if matches!(x, b'(' | b')' | b'\\') {
                v.push(b'\\');
            }
            v.push(x);
        }
    }
    v.extend_from_slice(b") /6 << /0 [ ");
    for (len, keys) in styles {
        v.extend_from_slice(format!("<< /0 << /0 << /0 (\u{fe}\u{ff}) /6 << /0 0 {keys} >> >> >> /1 {len} >> ").as_bytes());
    }
    v.extend_from_slice(b"] >> >> >> ] >>");
    v
}

fn with_text_index(tysh: &[u8], index: i32) -> Vec<u8> {
    let mut t = crate::psd::parse_tysh(tysh).unwrap();
    t.text.items.retain(|(k, _)| !k.is("TextIndex"));
    t.text.items.push((photocraft_psd::descriptor::Id::new("TextIndex"), photocraft_psd::descriptor::Value::Integer(index)));
    crate::psd::write_tysh(&t)
}

/// Optical (and "0") kerning only lives in `Txt2`: import applies it when the layer's
/// `TextIndex` object still holds the same text.
#[test]
fn txt2_carries_optical_kerning() {
    use photocraft_doc::text::Kerning;
    let t = styled("AVA", CharStyle::default());
    let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, None), 0);
    let mut back = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    assert!(back.char_runs().iter().all(|r| r.style.kerning == Kerning::Metrics));
    // "AVA\r": two optical characters, then manual ("0") for the last and the break.
    let txt2 = crate::psd::parse_txt2(&txt2_with_modes("AVA", &[(2, 2), (2, 0)])).unwrap();
    crate::psd::apply_txt2(&mut back, &tysh, &txt2);
    let modes: Vec<(usize, Kerning)> = back.char_runs().iter().map(|r| (r.len, r.style.kerning)).collect();
    assert_eq!(modes, vec![(2, Kerning::Optical), (1, Kerning::Off)]);
    // Stale Txt2 (other text), another index or garbage: nothing changes, nothing panics.
    for (data, index) in [(txt2_with_modes("AVX", &[(4, 2)]), 0), (txt2_with_modes("AVA", &[(4, 2)]), 3), (b"<< /1 [ (x".to_vec(), 0), (Vec::new(), 0)] {
        let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, None), index);
        let mut l = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
        if let Some(txt2) = crate::psd::parse_txt2(&data) {
            crate::psd::apply_txt2(&mut l, &tysh, &txt2);
        }
        assert!(
            l.char_runs().iter().all(|r| r.style.kerning == Kerning::Metrics),
            "case {index}: {:?}",
            l.char_runs().iter().map(|r| r.style.kerning).collect::<Vec<_>>()
        );
    }
    // Run lengths longer than the text, zero lengths and odd modes are tolerated.
    let txt2 = crate::psd::parse_txt2(&txt2_with_modes("AVA", &[(0, 2), (100, 2), (5, 9)])).unwrap();
    let mut l = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    crate::psd::apply_txt2(&mut l, &tysh, &txt2);
    assert!(l.char_runs().iter().all(|r| r.style.kerning == Kerning::Optical));
}

/// #3617: Photoshop keeps Stylistic Sets, Contextual Alternates, Swash and the other OpenType
/// settings besides (discretionary) ligatures only in `Txt2`. Import reads them, so the layout
/// our engine draws when the type is edited keeps Photoshop's glyphs instead of falling back to
/// the default ones.
#[test]
fn txt2_carries_opentype_features() {
    let feats = |l: &TextLayer| l.char_runs().iter().map(|r| r.style.features.iter().map(|f| (f.tag.clone(), f.value)).collect::<Vec<_>>()).collect::<Vec<_>>();
    let f = |tag: &str, value: u16| (tag.to_string(), value);
    // "Il->36" plus the engine's closing break: Stylistic Set 2 (mask 2), Contextual Alternates
    // and Swash on "Il"; Contextual Alternates off (Photoshop's default), Stylistic Set 1 and
    // Fractions on the rest.
    let text = "Il->36";
    let plain = styled(text, CharStyle { font_family: "Inter".into(), ..Default::default() });
    let tysh = with_text_index(&crate::psd::build_tysh(&plain, 72.0, None), 0);
    let styles = [(2, "/11 1 /18 true /19 false /20 true /25 true /85 2".to_string()), (5, "/11 1 /20 false /23 true /85 1".to_string())];
    let txt2 = crate::psd::parse_txt2(&txt2_with_styles(text, &styles)).unwrap();
    let mut back = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    assert_eq!(feats(&back), vec![Vec::<(String, u16)>::new()], "EngineData alone doesn't have them");
    crate::psd::apply_txt2(&mut back, &tysh, &txt2);
    assert_eq!(feats(&back), vec![vec![f("calt", 1), f("swsh", 1), f("ss02", 1)], vec![f("calt", 0), f("frac", 1), f("ss01", 1)]]);
    assert_eq!(back.char_runs().iter().map(|r| r.len).collect::<Vec<_>>(), vec![2, 4]);
    // What editing shows (our layout) has Photoshop's glyphs: Inter's `ss02` draws another I and
    // l, `ss01` another 3 and 6, and `->` stays two glyphs without `calt` (with it, an arrow).
    let mut e = TextEngine::new();
    let ids = |e: &mut TextEngine, t: &TextLayer| e.layout(t, 72.0).glyphs.iter().map(|g| g.id).collect::<Vec<_>>();
    let with = |len: usize, tags: &[(&str, u16)]| TextRun {
        len,
        style: CharStyle {
            features: tags.iter().map(|(tag, value)| FontFeature { tag: (*tag).into(), value: *value }).collect(),
            ..back.char_runs()[0].style.clone()
        },
    };
    let mut expected = back.clone();
    expected.runs = vec![with(2, &[("ss02", 1)]), with(4, &[("calt", 0), ("ss01", 1)])];
    let shown = ids(&mut e, &back);
    assert_eq!(shown, ids(&mut e, &expected));
    let defaults = ids(&mut e, &crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap());
    assert_eq!((shown.len(), defaults.len()), (6, 5), "`->` is two glyphs, as in Photoshop, not an arrow");
    assert!(shown.first() != defaults.first() && shown.last() != defaults.last(), "{shown:?} vs {defaults:?}");
    // A feature the style already sets (from EngineData, read first) keeps its value; masks and
    // flags of the wrong type, masks beyond the 20 sets and unknown keys change nothing.
    let mut t = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    t.runs = vec![TextRun { len: text.len(), style: CharStyle { features: vec![FontFeature { tag: "ss01".into(), value: 0 }], ..Default::default() } }];
    let odd =
        [(1, "/85 1".to_string()), (1, "/85 (x) /20 (x) /25 (true)".to_string()), (1, "/85 -1".to_string()), (4, "/85 99999999999 /999 true".to_string())];
    crate::psd::apply_txt2(&mut t, &tysh, &crate::psd::parse_txt2(&txt2_with_styles(text, &odd)).unwrap());
    assert_eq!(feats(&t), vec![vec![f("ss01", 0)]]);
}

/// A `Txt2` written by [`crate::psd::build_txt2`] (what PSD export writes for the type layers)
/// restores every auto-kern mode on import (#1348: a new Optical layer used to reopen as
/// Metrics, because EngineData's `AutoKerning true` covers both).
#[test]
fn build_txt2_round_trips_kerning_modes() {
    use photocraft_doc::text::Kerning;
    let style = |kerning: Kerning, kern: f32| CharStyle { kerning, kern, ..Default::default() };
    let t = runs_of("AVAT", &[(2, style(Kerning::Optical, 0.0)), (1, style(Kerning::Metrics, 0.0)), (1, style(Kerning::Off, 0.0))]);
    let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, None), 0);
    let mut back = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    assert!(
        back.char_runs().iter().all(|r| r.style.kerning != Kerning::Optical),
        "EngineData alone can't say Optical: {:?}",
        back.char_runs().iter().map(|r| r.style.kerning).collect::<Vec<_>>()
    );
    let txt2 = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(0, &t)], None)).unwrap();
    crate::psd::apply_txt2(&mut back, &tysh, &txt2);
    let modes: Vec<(usize, Kerning)> = back.char_runs().iter().map(|r| (r.len, r.style.kerning)).collect();
    assert_eq!(modes, vec![(2, Kerning::Optical), (1, Kerning::Metrics), (1, Kerning::Off)]);
    // A manual kern (a nonzero pair value) is the manual mode, as in the EngineData pair fields.
    let manual = runs_of("AB", &[(1, style(Kerning::Metrics, 50.0)), (1, style(Kerning::Metrics, 0.0))]);
    let tysh = with_text_index(&crate::psd::build_tysh(&manual, 72.0, None), 0);
    let mut back = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    let txt2 = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(0, &manual)], None)).unwrap();
    crate::psd::apply_txt2(&mut back, &tysh, &txt2);
    let modes: Vec<(usize, Kerning)> = back.char_runs().iter().map(|r| (r.len, r.style.kerning)).collect();
    assert_eq!(modes, vec![(1, Kerning::Off), (1, Kerning::Metrics)]);
}

/// #3514: the document `Txt2` is only needed for a non-Metrics (`Optical`) auto-kern mode. A fresh
/// document of ordinary Metrics (or EngineData-carried Off) text must not write Photoshop's empty
/// stub, which some readers (Photopea) reject.
#[test]
fn needs_txt2_only_for_optical() {
    use photocraft_doc::text::Kerning;
    let style = |kerning: Kerning| CharStyle { kerning, ..Default::default() };
    let metrics = runs_of("AV", &[(2, style(Kerning::Metrics))]);
    assert!(!crate::psd::needs_txt2(&[(0, &metrics)]));
    let off = runs_of("AV", &[(2, style(Kerning::Off))]);
    assert!(!crate::psd::needs_txt2(&[(0, &off)]), "Off lives in EngineData, not Txt2");
    let optical = runs_of("AV", &[(1, style(Kerning::Optical)), (1, CharStyle::default())]);
    assert!(crate::psd::needs_txt2(&[(0, &optical)]));
    assert!(!crate::psd::needs_txt2(&[]));
}

/// A kept `Txt2` object — same text, so its extras are still true — keeps everything the file
/// held beyond the regenerated keys (Photoshop stores glyph pen positions under `/21 /1`), and a
/// changed text drops those extras with the stale runs. The block's own extras always survive.
#[test]
fn txt2_keeps_a_kept_objects_extras_and_drops_them_with_its_text() {
    use crate::engine_data::Value as E;
    use photocraft_doc::text::Kerning;
    let style = |kerning: Kerning, kern: f32| CharStyle { kerning, kern, ..Default::default() };
    // The file's block: object 0 with `/21 /1` pen positions, a model extra, and a block extra.
    let obj = E::Dict(vec![
        (
            "0".into(),
            E::Dict(vec![("0".into(), E::String("AB\r".into())), ("6".into(), E::Dict(vec![("0".into(), E::Array(vec![]))])), ("keep".into(), E::Int(7))]),
        ),
        ("21".into(), E::Dict(vec![("1".into(), E::Array(vec![E::Real(1.5), E::Real(2.5)]))])),
    ]);
    let prev = crate::engine_data::write_bare(&[
        ("98".into(), E::Dict(vec![("0".into(), E::Int(14))])),
        ("0".into(), E::dict()),
        ("1".into(), E::Dict(vec![("1".into(), E::Array(vec![obj]))])),
        ("extra".into(), E::Int(3)),
    ]);
    // Same text: the extras survive, the style runs are ours.
    let t = runs_of("AB", &[(1, style(Kerning::Optical, 0.0)), (1, style(Kerning::Metrics, 0.0))]);
    let out = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(0, &t)], Some(&prev))).unwrap();
    assert_eq!(out.get("extra").and_then(E::as_i64), Some(3), "block extras survive");
    let object = out.path(&["1", "1"]).and_then(E::as_array).unwrap()[0].clone();
    assert_eq!(object.path(&["21", "1"]).and_then(E::as_array).map(|items| items.len()), Some(2), "pen positions survive an unchanged text");
    assert_eq!(object.path(&["0", "keep"]).and_then(E::as_i64), Some(7), "model extras survive");
    assert_eq!(object.path(&["0", "6", "0"]).and_then(E::as_array).map(|items| items.len()), Some(2), "the style runs are regenerated");
    // Changed text: the extras go stale with the old runs and are dropped.
    let t = runs_of("XY", &[(1, style(Kerning::Optical, 0.0)), (1, style(Kerning::Metrics, 0.0))]);
    let out = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(0, &t)], Some(&prev))).unwrap();
    assert_eq!(out.get("extra").and_then(E::as_i64), Some(3), "block extras survive a text change too");
    let object = out.path(&["1", "1"]).and_then(E::as_array).unwrap()[0].clone();
    assert!(object.get("21").is_none(), "pen positions of another text are dropped: {:?}", object.get("21"));
    assert!(object.path(&["0", "keep"]).is_none(), "model extras of another text are dropped");
}

/// A `Txt2` shaped like Photoshop's: document resources and settings beside the text objects
/// (`/0`, `/1 /0`, `/1 /2`), style runs carrying a full style (font, size, leading) besides the
/// auto-kern mode, a Latin-1 string without a byte-order mark, reals, and a second object no
/// layer names. `modes` are the object's `(UTF-16 length, auto-kern mode)` runs for "AVA\r".
fn photoshop_txt2(modes: &[(usize, i64)]) -> Vec<u8> {
    let mut v = b"\n\n/98 << /0 14 >> /0 << /1 << /0 [ << /0 << /0 (MyriadPro-Regular) /2 0 >> >> ] >> >> /1 << /0 << /0 1 /1 .5 >> /1 [ << /0 << /0 (\xfe\xff\0A\0V\0A\0\r) /6 << /0 [ ".to_vec();
    for (len, m) in modes {
        v.extend_from_slice(format!("<< /0 << /0 << /0 (\u{fe}\u{ff}) /6 << /0 0 /1 12.0 /2 14.5 /11 {m} >> >> >> /1 {len} >> ").as_bytes());
    }
    v.extend_from_slice(
        b"] >> /7 << /0 1 >> >> /21 << /1 [ 1.25 2.5 ] >> >> << /0 << /0 (\xfe\xff\0B\0\r) >> /21 << /1 [ 3.0 ] >> >> ] /2 << /0 7 >> >> /3 << /0 .25 >>",
    );
    v
}

/// #2374: an unedited save hands Photoshop back its own `Txt2`, byte for byte. Rebuilding it
/// dropped `/1 /0` and `/1 /2`, emptied unnamed objects and cut every style run down to its
/// auto-kern mode, and Photoshop refused to open the file.
#[test]
fn unchanged_txt2_is_written_back_byte_for_byte() {
    use photocraft_doc::text::Kerning;
    let prev = photoshop_txt2(&[(2, 1), (2, 1)]);
    let tysh = with_text_index(&crate::psd::build_tysh(&styled("AVA", CharStyle::default()), 72.0, None), 0);
    let mut t = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    crate::psd::apply_txt2(&mut t, &tysh, &crate::psd::parse_txt2(&prev).unwrap());
    assert!(t.char_runs().iter().all(|r| r.style.kerning == Kerning::Metrics));
    assert_eq!(crate::psd::build_txt2(&[(0, &t)], Some(&prev)), prev);
    // An optical run reads back and writes back the same way.
    let prev = photoshop_txt2(&[(1, 2), (3, 1)]);
    crate::psd::apply_txt2(&mut t, &tysh, &crate::psd::parse_txt2(&prev).unwrap());
    assert_eq!(t.char_runs().first().map(|r| r.style.kerning), Some(Kerning::Optical));
    assert_eq!(crate::psd::build_txt2(&[(0, &t)], Some(&prev)), prev);
}

/// An edited auto-kern mode changes only `/11` (splitting the run it falls inside); the file's
/// style keys, the other `/1` keys and the objects no layer names all survive.
#[test]
fn edited_txt2_keeps_the_files_styles_and_objects() {
    use crate::engine_data::Value as E;
    use photocraft_doc::text::Kerning;
    let prev = photoshop_txt2(&[(4, 1)]);
    let style = |kerning: Kerning| CharStyle { kerning, ..Default::default() };
    let t = runs_of("AVA", &[(1, style(Kerning::Optical)), (2, style(Kerning::Metrics))]);
    let out = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(0, &t)], Some(&prev))).unwrap();
    let before = crate::psd::parse_txt2(&prev).unwrap();
    for key in [&["0"][..], &["1", "0"], &["1", "2"], &["3"]] {
        assert_eq!(out.path(key), before.path(key), "/{} survives", key.join(" /"));
    }
    let objects = out.path(&["1", "1"]).and_then(E::as_array).unwrap();
    assert_eq!(objects.len(), 2, "the object no layer names is kept");
    assert_eq!(objects.get(1), before.path(&["1", "1"]).and_then(E::as_array).unwrap().get(1));
    let object = &objects[0];
    assert_eq!(object.path(&["0", "7", "0"]).and_then(E::as_i64), Some(1), "model extras survive");
    assert!(object.path(&["21", "1"]).is_some(), "pen positions survive an unchanged text");
    let runs = object.path(&["0", "6", "0"]).and_then(E::as_array).unwrap();
    let got: Vec<(i64, i64)> =
        runs.iter().map(|r| (r.get("1").and_then(E::as_i64).unwrap(), r.path(&["0", "0", "6", "11"]).and_then(E::as_i64).unwrap())).collect();
    assert_eq!(got, vec![(1, 2), (3, 1)], "the run is split where the mode changes");
    for r in runs {
        assert_eq!(r.path(&["0", "0", "6", "1"]).and_then(E::as_f64), Some(12.0), "the font size survives");
        assert_eq!(r.path(&["0", "0", "6", "2"]).and_then(E::as_f64), Some(14.5), "the leading survives");
    }
    // And it reads back with the edited modes.
    let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, None), 0);
    let mut back = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    crate::psd::apply_txt2(&mut back, &tysh, &out);
    let modes: Vec<(usize, Kerning)> = back.char_runs().iter().map(|r| (r.len, r.style.kerning)).collect();
    assert_eq!(modes, vec![(1, Kerning::Optical), (2, Kerning::Metrics)]);
}

/// The file's style runs are only reused when they cover the text: runs of another length (or
/// hostile lengths) fall back to generated runs, without panicking.
#[test]
fn txt2_runs_that_dont_cover_the_text_are_regenerated() {
    use crate::engine_data::Value as E;
    let t = runs_of("AVA", &[(3, CharStyle::default())]);
    for modes in [&[(3, 1)][..], &[(9, 1)], &[(i64::MAX as usize, 1), (2, 1)], &[]] {
        let prev = photoshop_txt2(modes);
        let out = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(0, &t)], Some(&prev))).unwrap();
        let runs = out.path(&["1", "1"]).and_then(E::as_array).unwrap()[0].path(&["0", "6", "0"]).and_then(E::as_array).unwrap();
        let total: i64 = runs.iter().filter_map(|r| r.get("1").and_then(E::as_i64)).sum();
        assert_eq!(total, 4, "{modes:?}");
        assert!(out.path(&["1", "2"]).is_some());
    }
}

/// A file-controlled `TextIndex` must not size the save: out-of-range numbers are ignored (the
/// slot array is sized by real text objects, never by a file's numbers) and reading one back is
/// a no-op, so a hostile file degrades instead of allocating gigabytes on export.
#[test]
fn hostile_text_index_is_ignored_not_sized() {
    use crate::engine_data::Value as E;
    let t = runs_of("AB", &[(2, CharStyle::default())]);
    let tysh = crate::psd::build_tysh(&t, 72.0, None);
    assert_eq!(crate::psd::text_index(&crate::psd::set_text_index(&tysh, 0).unwrap()), Some(0));
    assert_eq!(crate::psd::text_index(&crate::psd::set_text_index(&tysh, crate::psd::MAX_TEXT_INDEX).unwrap()), Some(crate::psd::MAX_TEXT_INDEX));
    for hostile in [crate::psd::MAX_TEXT_INDEX + 1, i32::MAX] {
        assert_eq!(crate::psd::text_index(&crate::psd::set_text_index(&tysh, hostile).unwrap()), None, "{hostile}");
        let out = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(hostile, &t)], None)).unwrap();
        let slots = out.path(&["1", "1"]).and_then(E::as_array).unwrap();
        assert!(slots.is_empty(), "{hostile} sized {} slots", slots.len());
    }
    // The bound is real but generous: the last honoured number lands at its slot.
    let out = crate::psd::parse_txt2(&crate::psd::build_txt2(&[(crate::psd::MAX_TEXT_INDEX, &t)], None)).unwrap();
    let slots = out.path(&["1", "1"]).and_then(E::as_array).unwrap();
    assert_eq!(slots.len(), crate::psd::MAX_TEXT_INDEX as usize + 1);
}

/// Photopea needs an enabled fill, not just FillColor, to paint text after an edit. Inspect the
/// serialized TySh rather than our renderer, which does not consume the PSD fill flag.
#[test]
fn psd_text_fill_is_enabled_in_runs_and_new_default_styles() {
    let mut t = point("AB", 100.0);
    t.runs = vec![
        TextRun { len: 1, style: CharStyle { color: Color::rgb(0.2, 0.4, 0.6), ..Default::default() } },
        TextRun { len: 1, style: CharStyle { faux_bold: true, size_pt: 100.0, ..Default::default() } },
    ];
    for shape in [TextShape::Point, TextShape::Box { x: 0.0, y: 0.0, width: 300.0, height: 200.0 }] {
        t.shape = shape;
        let tysh = crate::psd::parse_tysh(&crate::psd::build_tysh(&t, 72.0, None)).unwrap();
        let data = crate::psd::engine_data(&tysh.text).unwrap();
        let runs = data.path(&["EngineDict", "StyleRun", "RunArray"]).unwrap().as_array().unwrap();
        assert_eq!(runs.len(), 2);
        for run in runs {
            assert_eq!(run.path(&["StyleSheet", "StyleSheetData", "FillFlag"]).and_then(crate::engine_data::Value::as_bool), Some(true));
        }
        for key in ["ResourceDict", "DocumentResources"] {
            let sheets = data.path(&[key, "StyleSheetSet"]).unwrap().as_array().unwrap();
            assert_eq!(sheets[0].path(&["StyleSheetData", "FillFlag"]).and_then(crate::engine_data::Value::as_bool), Some(true));
        }
        let back = crate::psd::text_layer_from_tysh(&crate::psd::write_tysh(&tysh), 72.0).unwrap();
        assert_eq!(back.text, t.text);
        assert_eq!(back.runs.len(), t.runs.len());
        for (actual, expected) in back.runs.iter().zip(&t.runs) {
            assert_eq!(actual.style.color, expected.style.color);
            assert_eq!(actual.style.size_pt, expected.style.size_pt);
            assert_eq!(actual.style.faux_bold, expected.style.faux_bold);
        }
    }
}

#[test]
fn psd_text_fill_is_enabled_after_editing_a_legacy_export() {
    use crate::engine_data::Value as E;
    // A legacy PhotoCraft export has a normal style and character runs without FillFlag.
    let mut t = point("Before", 40.0);
    let mut legacy = crate::psd::parse_tysh(&crate::psd::build_tysh(&t, 72.0, None)).unwrap();
    fn remove_fill_flags(value: &mut E) {
        match value {
            E::Dict(items) => {
                items.retain(|(key, _)| key != "FillFlag");
                for (_, value) in items {
                    remove_fill_flags(value);
                }
            }
            E::Array(items) => items.iter_mut().for_each(remove_fill_flags),
            _ => {}
        }
    }
    let mut data = crate::psd::engine_data(&legacy.text).unwrap();
    remove_fill_flags(&mut data);
    for (key, value) in &mut legacy.text.items {
        if key.as_bytes() == b"EngineData" {
            *value = photocraft_psd::descriptor::Value::RawData(crate::engine_data::write(&data));
        }
    }
    t.psd_raw = Some(crate::psd::write_tysh(&legacy).into());
    t.text = "After".into();
    let tysh = crate::psd::parse_tysh(&crate::psd::build_tysh(&t, 72.0, None)).unwrap();
    let data = crate::psd::engine_data(&tysh.text).unwrap();
    let runs = data.path(&["EngineDict", "StyleRun", "RunArray"]).unwrap().as_array().unwrap();
    assert!(!runs.is_empty());
    for run in runs {
        assert_eq!(run.path(&["StyleSheet", "StyleSheetData", "FillFlag"]).and_then(E::as_bool), Some(true));
    }
}

/// EngineData pair fields: manual kerning and "no automatic kerning" round-trip exactly, in
/// Photoshop's form (see `psd::pair_runs`).
#[test]
fn psd_round_trips_manual_kerning() {
    use photocraft_doc::text::Kerning::{Metrics as M, Off as O};
    let s = CharStyle::default();
    let cases: Vec<Vec<(usize, photocraft_doc::text::Kerning, f32)>> = vec![
        vec![(1, O, 100.0), (1, O, -50.0), (2, M, 0.0)],
        vec![(1, M, 0.0), (1, O, 0.0), (2, M, 0.0)],
        vec![(4, O, 0.0)],
        vec![(2, M, 0.0), (1, O, 25.0), (1, M, 0.0)],
        vec![(3, M, 0.0), (1, O, 300.0)],
    ];
    for runs in cases {
        let t = runs_of("AVAT", &runs.iter().map(|&(len, kerning, kern)| (len, CharStyle { kerning, kern, ..s.clone() })).collect::<Vec<_>>());
        let back = crate::psd::text_layer_from_tysh(&crate::psd::build_tysh(&t, 72.0, None), 72.0).unwrap();
        let per = |t: &TextLayer| t.char_runs().iter().flat_map(|r| std::iter::repeat_n((r.style.kerning, r.style.kern), r.len)).collect::<Vec<_>>();
        let mut want = per(&t);
        // The last character's mode has no EngineData slot (it reads back as Metrics unless
        // it has a manual kern).
        if let Some(last) = want.last_mut()
            && last.1 == 0.0
        {
            last.0 = M;
        }
        assert_eq!(per(&back), want, "{runs:?}");
    }
}

#[test]
fn psd_round_trips_antialias_opentype_and_warp() {
    use photocraft_doc::text::{AntiAlias, TextWarp};
    let style = CharStyle {
        font_family: "Inter".into(),
        size_pt: 20.0,
        features: vec![FontFeature { tag: "swsh".into(), value: 1 }, FontFeature { tag: "frac".into(), value: 1 }],
        discretionary_ligatures: true,
        ..Default::default()
    };
    for aa in [AntiAlias::Windows, AntiAlias::WindowsLcd, AntiAlias::Crisp, AntiAlias::None] {
        let mut t = styled("1/2 Swash", style.clone());
        t.antialias = aa;
        t.warp = Some(TextWarp { style: "warpFlag".into(), value: -35.0, horizontal_distortion: 10.0, vertical_distortion: 0.0, horizontal: false });
        let bytes = crate::psd::build_tysh(&t, 72.0, None);
        let back = crate::psd::text_layer_from_tysh(&bytes, 72.0).unwrap();
        assert_eq!(back.antialias, aa);
        assert_eq!(back.warp, t.warp);
        let st = &back.char_runs()[0].style;
        assert!(st.discretionary_ligatures);
        let mut tags: Vec<&str> = st.features.iter().map(|f| f.tag.as_str()).collect();
        tags.sort_unstable();
        assert_eq!(tags, ["frac", "swsh"]);
    }
}

/// #1469: Some older PSD writers combine a zero PointBase with an enormous local ink origin.
/// The TySh transform cancels that origin, so importing it as a zero-based layout puts text far
/// off-canvas. Fold the descriptor origin into the imported transform and preserve it on a TySh
/// round trip so every editing and export path uses the same position.
#[test]
fn legacy_tysh_point_origin_imports_and_round_trips_on_canvas() {
    use photocraft_psd::descriptor::{Descriptor, Id, Value as D};

    let source = styled("Name", CharStyle { font_family: "Inter".into(), size_pt: 120.0, ..Default::default() });
    let mut tysh = crate::psd::parse_tysh(&crate::psd::build_tysh(&source, 72.0, None)).unwrap();
    tysh.transform = Affine { m: [4.1667, 0.0, 0.0, 4.1667, -32375.0, -32887.5] };
    let ink = [8050.0, 8160.0, 8220.0, 8300.0];
    let rect = |class| {
        D::Descriptor(
            Descriptor::new(class)
                .with("Left", D::UnitFloat { unit: *b"#Pnt", value: ink[0] })
                .with("Top ", D::UnitFloat { unit: *b"#Pnt", value: ink[1] })
                .with("Rght", D::UnitFloat { unit: *b"#Pnt", value: ink[2] })
                .with("Btom", D::UnitFloat { unit: *b"#Pnt", value: ink[3] }),
        )
    };
    tysh.text.items.retain(|(key, _)| !key.is("bounds") && !key.is("boundingBox"));
    tysh.text.items.push((Id::new("bounds"), rect("bounds")));
    tysh.text.items.push((Id::new("boundingBox"), rect("boundingBox")));
    let data = crate::psd::write_tysh(&tysh);

    let imported = crate::psd::text_layer_from_tysh(&data, 72.0).unwrap();
    let expected = [4.1667, 0.0, 0.0, 4.1667, 1166.935, 1112.772];
    for (got, want) in imported.transform.m.iter().zip(expected) {
        assert!((got - want).abs() < 0.001, "{:?}", imported.transform.m);
    }
    let (_, rendered) = TextEngine::new().render(&imported, 72.0, PixelFormat::RGBA8);
    let placed = rendered.surface.content_bounds();
    assert!(!placed.is_empty(), "the normalized layer renders");
    assert!((1000..3000).contains(&placed.x0) && (500..2000).contains(&placed.y0), "{placed:?}");

    let round_trip = crate::psd::text_layer_from_tysh(&crate::psd::build_tysh(&imported, 72.0, Some(ink.map(|v| v as f32))), 72.0).unwrap();
    for (got, want) in round_trip.transform.m.iter().zip(expected) {
        assert!((got - want).abs() < 0.001, "{:?}", round_trip.transform.m);
    }
}

/// Regression: a PSD whose text engine data has no `EngineDict` (or isn't a dictionary)
/// panicked with `expect("EngineDict")` when the layer was written back (PSD export).
#[test]
fn engine_data_template_without_engine_dict() {
    use crate::engine_data::{self as ed, Value as E};
    let t = styled("Hi", CharStyle::default());
    let no_engine_dict = ed::parse(b"<< /ResourceDict << >> >>").unwrap();
    for template in [no_engine_dict, E::Dict(vec![]), E::Int(3)] {
        let e = crate::psd::build_engine_data(&t, Some(template), 72.0);
        let text = e.path(&["EngineDict", "Editor", "Text"]);
        assert!(matches!(text, Some(E::String(s)) if s == "Hi\r"), "{text:?}");
    }
}

/// Photoshop needs one paragraph record per hard break even when styles match.
#[test]
fn psd_paragraph_records_keep_explicit_breaks_and_empty_paragraphs() {
    use crate::engine_data::Value as E;
    for (text, lengths) in [("A\nB", vec![2, 2]), ("A\n\nB", vec![2, 1, 2]), ("A\n", vec![2, 1]), ("", vec![1]), ("甲\n😀", vec![2, 3])] {
        let layer = TextLayer { text: text.into(), ..Default::default() };
        let before = layer.clone();
        let engine = crate::psd::build_engine_data(&layer, None, 72.0);
        let actual = engine.path(&["EngineDict", "ParagraphRun", "RunLengthArray"]).unwrap().as_array().unwrap();
        assert_eq!(*actual, lengths.into_iter().map(E::Int).collect::<Vec<_>>(), "{text:?}");
        assert_eq!(layer, before, "export does not mutate native text");
    }
    // A legacy style boundary before the break must not create another paragraph.
    let layer = TextLayer {
        text: "Größe\nzwei\n".into(),
        paragraphs: vec![
            ParagraphRun { len: 7, style: ParagraphStyle { align: TextAlign::Center, ..Default::default() } },
            ParagraphRun { len: 6, style: ParagraphStyle { align: TextAlign::Right, ..Default::default() } },
            ParagraphRun { len: 0, style: ParagraphStyle { align: TextAlign::Left, ..Default::default() } },
        ],
        ..Default::default()
    };
    let engine = crate::psd::build_engine_data(&layer, None, 72.0);
    let paragraphs = engine.path(&["EngineDict", "ParagraphRun"]).unwrap();
    assert_eq!(*paragraphs.get("RunLengthArray").unwrap().as_array().unwrap(), vec![E::Int(6), E::Int(5), E::Int(1)]);
    let alignments: Vec<_> = paragraphs
        .get("RunArray")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|run| run.path(&["ParagraphSheet", "Properties", "Justification"]).unwrap().as_i64().unwrap())
        .collect();
    assert_eq!(alignments, vec![2, 1, 0]);
}

/// #123: the caret geometry (clusters, lines) sits on the rendered glyphs, so a click on a glyph
/// lands next to it: its left part before it, its right part after it. Plain text, tracking,
/// horizontal scale, mixed sizes, and centred, wrapped paragraph text.
#[test]
fn clusters_sit_on_rendered_glyphs() {
    let mut e = TextEngine::new();
    let big = CharStyle { size_pt: 40.0, ..Default::default() };
    let small = CharStyle { size_pt: 18.0, ..Default::default() };
    let mixed =
        TextLayer { text: "HOHOHO".into(), runs: vec![TextRun { len: 3, style: big.clone() }, TextRun { len: 3, style: small.clone() }], ..Default::default() };
    let boxed = with_para(
        TextLayer { shape: TextShape::Box { x: 0.0, y: 0.0, width: 150.0, height: 200.0 }, ..styled("HOH HOH HOH HOH", small.clone()) },
        ParagraphStyle { align: TextAlign::Center, ..Default::default() },
    );
    let cases = [
        ("plain", styled("HOHOH", big.clone())),
        ("tracking", styled("HOHOH", CharStyle { tracking: 300.0, ..big.clone() })),
        ("hscale", styled("HOHOH", CharStyle { horizontal_scale: 1.6, ..big.clone() })),
        ("kerned", styled("HOHOH", CharStyle { kern: 250.0, kerning: photocraft_doc::text::Kerning::Optical, ..big.clone() })),
        ("mixed", mixed),
        ("box", boxed),
    ];
    for (name, t) in cases {
        let t = TextLayer { transform: Affine::translate(7.0, 60.0), ..t };
        let (l, r) = e.render(&t, 72.0, PixelFormat::RGBA8);
        assert!(l.lines.len() >= if name == "box" { 2 } else { 1 }, "{name}");
        for c in l.clusters.iter().filter(|c| !t.text[c.range.clone()].trim().is_empty()) {
            let ln = &l.lines[c.line];
            // Ink under the cluster's middle, between the line's ascent and descent.
            let col = photocraft_geom::Rect::new(
                (c.x + c.advance * 0.3 + 7.0).floor() as i32,
                (ln.baseline - ln.ascent + 60.0).floor() as i32,
                (c.x + c.advance * 0.55 + 7.0).ceil() as i32,
                (ln.baseline + ln.descent + 60.0).ceil() as i32,
            );
            assert!(alpha_sum(&r.surface, col) > 1.0, "{name}: no ink under cluster {:?} ({col:?})", c.range);
            let y = ln.baseline - ln.ascent * 0.4;
            assert_eq!(l.hit_test(c.x + c.advance * 0.2, y), c.range.start, "{name}: left part of {:?}", c.range);
            assert_eq!(l.hit_test(c.x + c.advance * 0.8, y), c.range.end, "{name}: right part of {:?}", c.range);
            let (cx, top, bottom) = l.caret(c.range.start);
            assert!((cx - c.x).abs() < 1e-3 && top < y && bottom > y, "{name}: caret at {:?}", c.range);
        }
    }
}

fn runs_of(text: &str, styles: &[(usize, CharStyle)]) -> TextLayer {
    TextLayer { text: text.into(), runs: styles.iter().map(|(len, style)| TextRun { len: *len, style: style.clone() }).collect(), ..Default::default() }
}

/// Manual kerning (1/1000 em) after a character moves everything after it by kern × size.
#[test]
fn manual_kerning_moves_the_next_glyph() {
    use photocraft_doc::text::Kerning;
    let mut e = TextEngine::new();
    let s = CharStyle { size_pt: 100.0, ..Default::default() };
    let plain = e.layout(&styled("HOH", s.clone()), 72.0);
    let kerned = e.layout(&runs_of("HOH", &[(1, CharStyle { kern: 100.0, ..s.clone() }), (2, s.clone())]), 72.0);
    let x = |l: &crate::TextLayout, i: usize| l.glyphs[i].x;
    assert_eq!(x(&kerned, 0), x(&plain, 0));
    // 100/1000 em at 100 px = 10 px, for the next glyph and everything after it.
    assert!((x(&kerned, 1) - x(&plain, 1) - 10.0).abs() < 1e-3, "{} vs {}", x(&kerned, 1), x(&plain, 1));
    assert!((x(&kerned, 2) - x(&plain, 2) - 10.0).abs() < 1e-3);
    assert!((width(&kerned) - width(&plain) - 10.0).abs() < 1e-3);
    // Carets follow: the cluster after the kerned pair starts 10 px later.
    assert!((kerned.caret(1).0 - plain.caret(1).0 - 10.0).abs() < 1e-3);
    // Negative kerning tightens; kerning on the last character doesn't move anything.
    let tight = e.layout(&runs_of("HOH", &[(1, CharStyle { kern: -50.0, ..s.clone() }), (2, s.clone())]), 72.0);
    assert!((x(&tight, 1) - x(&plain, 1) + 5.0).abs() < 1e-3);
    let last = e.layout(&runs_of("HOH", &[(2, s.clone()), (1, CharStyle { kern: 500.0, ..s.clone() })]), 72.0);
    assert!((width(&last) - width(&plain)).abs() < 1e-3);
    // Off replaces the font's pair kerning: "AV" with Off is wider than with Metrics.
    let mut av = |st: CharStyle| e.layout(&styled("AV", st), 72.0).glyphs[1].x;
    let metric = av(s.clone());
    let off = av(CharStyle { kerning: Kerning::Off, ..s.clone() });
    assert!(off > metric + 1.0, "Inter kerns AV: {off} vs {metric}");
    // Centred point text stays centred around the anchor with kerning.
    let centred = with_para(
        runs_of("HOH", &[(1, CharStyle { kern: 300.0, ..s.clone() }), (2, s.clone())]),
        ParagraphStyle { align: TextAlign::Center, ..Default::default() },
    );
    let l = e.layout(&centred, 72.0);
    assert!((l.lines[0].x0 + l.lines[0].x1).abs() < 0.5, "{:?}", l.lines[0]);
}

/// Optical kerning computes pair spacing from the outlines: tighter for open pairs ("AV", "To")
/// than the unkerned advance, about neutral for straight stems, and never absurd.
#[test]
fn optical_kerning_tightens_open_pairs() {
    use photocraft_doc::text::Kerning;
    let mut e = TextEngine::new();
    let s = CharStyle { size_pt: 100.0, ..Default::default() };
    let gap = |e: &mut TextEngine, text: &str, k: Kerning| {
        let l = e.layout(&styled(text, CharStyle { kerning: k, ..s.clone() }), 72.0);
        l.glyphs[1].x - l.glyphs[0].x
    };
    for pair in ["AV", "To", "LT", "Ty"] {
        let off = gap(&mut e, pair, Kerning::Off);
        let optical = gap(&mut e, pair, Kerning::Optical);
        assert!(optical < off - 3.0, "{pair}: optical {optical} vs unkerned {off}");
    }
    for pair in ["HH", "nn", "oo", "HO"] {
        let off = gap(&mut e, pair, Kerning::Off);
        let optical = gap(&mut e, pair, Kerning::Optical);
        assert!((optical - off).abs() < 6.0, "{pair}: optical {optical} vs unkerned {off}");
    }
    // A space breaks the pair; a manual kern replaces the automatic one (as in Photoshop).
    assert_eq!(gap(&mut e, "A V", Kerning::Optical), gap(&mut e, "A V", Kerning::Off));
    for mode in [Kerning::Optical, Kerning::Metrics] {
        let l = e.layout(&styled("AV", CharStyle { kerning: mode, kern: 100.0, ..s.clone() }), 72.0);
        let off = gap(&mut e, "AV", Kerning::Off);
        assert!((l.glyphs[1].x - l.glyphs[0].x - off - 10.0).abs() < 1e-3, "{mode:?}");
    }
}

/// A mode change splits shaping runs: the pairs on both sides of a manually kerned character
/// lose their automatic kerning (Photoshop renders it the same way).
#[test]
fn kerning_modes_split_pairs() {
    use photocraft_doc::text::Kerning;
    let mut e = TextEngine::new();
    let s = CharStyle { size_pt: 100.0, ..Default::default() };
    let off = CharStyle { kerning: Kerning::Off, ..s.clone() };
    let xs = |e: &mut TextEngine, t: &TextLayer| e.layout(t, 72.0).glyphs.iter().map(|g| g.x).collect::<Vec<_>>();
    let metric = xs(&mut e, &styled("AVAV", s.clone()));
    let plain = xs(&mut e, &styled("AVAV", off.clone()));
    let mixed = xs(&mut e, &runs_of("AVAV", &[(2, s.clone()), (1, off.clone()), (1, s.clone())]));
    let adv = |v: &[f32], i: usize| v[i + 1] - v[i];
    assert!((adv(&mixed, 0) - adv(&metric, 0)).abs() < 1e-3, "AV before stays kerned");
    assert!((adv(&mixed, 1) - adv(&plain, 1)).abs() < 1e-3, "VA into the manual character");
    assert!((adv(&mixed, 2) - adv(&plain, 2)).abs() < 1e-3, "AV out of it");
}

#[test]
fn japanese_dictionary_word_boundaries() {
    // Use only the existing bundled Latin fonts: segmentation must not require a CJK font.
    let text = "私は学生です";
    let mut fonts = fonts::FontDb::new();
    let mut context = parley::LayoutContext::<[u8; 4]>::new();
    let mut layout = parley::Layout::new();
    let mut builder = context.ranged_builder(&mut fonts.fcx, text, 1.0, false);
    builder.push_default(parley::StyleProperty::FontFamily(parley::FontFamily::named("Inter")));
    // Suppress CJK line-break opportunities so these flags expose word boundaries alone.
    builder.push_default(parley::StyleProperty::WordBreak(parley::WordBreak::KeepAll));
    builder.build_into(&mut layout, text);
    layout.break_all_lines(None);
    let mut boundaries = Vec::new();
    for line in layout.lines() {
        for run in line.runs() {
            for cluster in run.clusters() {
                if cluster.is_word_boundary() {
                    boundaries.push(cluster.text_range().start);
                }
            }
        }
    }
    assert_eq!(boundaries, vec![0, 3, 6, 12]);
}

#[test]
fn japanese_box_text_preserves_wrap_boundaries() {
    let mut engine = TextEngine::new();
    let mut text = point("日本語の文章を折り返します。", 12.0);
    text.shape = TextShape::Box { x: 0.0, y: 0.0, width: 40.0, height: 1000.0 };
    let layout = engine.layout(&text, 72.0);
    assert!(layout.lines.len() > 1);
    let mut end = 0;
    for line in &layout.lines {
        assert_eq!(line.range.start, end);
        let content = text.text.get(line.range.clone()).expect("UTF-8 line boundaries");
        assert!(!content.starts_with('。'), "closing punctuation must stay with its preceding text");
        end = line.range.end;
    }
    assert_eq!(end, text.text.len());
}

/// Which craft-fonts family drew the glyphs of `l` (matched by the embedded bytes).
fn craft_family_of(l: &crate::TextLayout) -> Vec<&'static str> {
    let mut v = Vec::new();
    for g in &l.glyphs {
        let Some(face) = l.faces.get(g.face as usize) else { continue };
        let data: &[u8] = face.font.data.as_ref();
        if let Some(f) = crate::CRAFT_FONTS.iter().find(|f| std::ptr::eq(f.bytes.as_ptr(), data.as_ptr()) && f.bytes.len() == data.len())
            && !v.contains(&f.family)
        {
            v.push(f.family);
        }
    }
    v
}

#[test]
fn craft_fonts_cover_japanese_without_system_fonts() {
    if !crate::CRAFT_FONTS.iter().any(|f| f.is_japanese()) {
        eprintln!("skipping: built without craft-fonts (set CRAFT_FONTS_DIR to a craft-fonts checkout)");
        return;
    }
    let mut e = TextEngine::new();
    let text = "日本語の文字、カタカナ。";
    let l = e.layout(&point(text, 24.0), 72.0);
    assert_eq!(l.glyphs.len(), text.chars().count());
    assert!(l.glyphs.iter().all(|g| g.id != 0), "no .notdef with craft-fonts");
    // Sans (Inter) runs fall back to the Gothic UI family.
    assert_eq!(craft_family_of(&l), vec![crate::craft_fonts::UI_JAPANESE_FAMILY]);
    // Serif runs fall back to a Mincho face when the build has one (not the web build).
    if crate::CRAFT_FONTS.iter().any(|f| f.is_mincho()) {
        let mut t = point(text, 24.0);
        t.font_family = "Times New Roman".into();
        let l = e.layout(&t, 72.0);
        assert!(l.glyphs.iter().all(|g| g.id != 0));
        let fams = craft_family_of(&l);
        assert!(fams.len() == 1 && fams[0].contains("Mincho"), "{fams:?}");
    }
    // Latin keeps Inter.
    let l = e.layout(&point("Layer 1", 24.0), 72.0);
    assert!(craft_family_of(&l).is_empty());
}

#[test]
fn works_without_craft_fonts() {
    // Whatever the build: the bundled fonts load, Latin lays out, and Japanese never panics
    // (without craft-fonts or system fonts it may be .notdef).
    let mut e = TextEngine::new();
    assert!(e.fonts.has_family("Inter"));
    let l = e.layout(&point("日本語 Latin", 12.0), 72.0);
    assert_eq!(l.glyphs.len(), "日本語 Latin".chars().count());
    for f in crate::CRAFT_FONTS {
        assert!(e.fonts.has_family(f.family), "{} registered under its manifest name", f.family);
    }
    if crate::CRAFT_FONTS.is_empty() {
        assert!(crate::craft_fonts::japanese_families().is_empty());
        let fb = fonts::fallback_candidates(&crate::cjk::script_order(Some("ja")));
        assert!(!fb.iter().any(|f| f.contains("BIZ UD")));
    }
}

#[test]
fn word_and_line_navigation() {
    use crate::layout::{byte_index, char_index, hit_char, line_edge, line_index, line_step, word_boundary};
    assert_eq!(word_boundary("hello big world", 0, true), 5);
    assert_eq!(word_boundary("hello big world", 7, false), 6);
    assert_eq!(word_boundary("hello big world", 15, false), 10);
    assert_eq!(word_boundary("hello", 0, false), 0);
    assert_eq!(word_boundary("hello", 5, true), 5);
    assert_eq!(word_boundary("", 4, true), 0);
    assert_eq!(word_boundary("ab", 100, true), 2);
    assert_eq!(word_boundary("ab, cd", 0, true), 2);
    assert_eq!(word_boundary("ab, cd", 2, true), 6);
    assert_eq!(word_boundary("Größe", 0, true), 5);

    let mut e = TextEngine::new();
    let text = "AäB\ncd";
    for vertical in [false, true] {
        let mut t = point(text, 20.0);
        if vertical {
            t.orientation = Orientation::Vertical;
        }
        let l = e.layout(&t, 72.0);
        let n = text.chars().count();
        for i in 0..=n {
            let b = byte_index(text, i);
            let [(x0, y0), (x1, y1)] = l.caret_segment(b);
            let (idx, line) = hit_char(&l, text, (x0 + x1) / 2.0, (y0 + y1) / 2.0);
            assert_eq!(idx, i, "vertical {vertical} index {i}");
            assert_eq!(line, line_index(&l, b), "vertical {vertical} index {i}");
        }
        assert!(l.lines.len() >= 2, "vertical {vertical}");
        let end0 = char_index(text, l.lines[0].range.end);
        let start1 = char_index(text, l.lines[1].range.start);
        assert_eq!(line_edge(&l, text, 1, true), end0, "vertical {vertical}");
        assert_eq!(line_edge(&l, text, 1, false), 0, "vertical {vertical}");
        assert_eq!(line_edge(&l, text, start1, false), start1, "vertical {vertical}");
        assert_eq!(line_edge(&l, text, start1, true), n, "vertical {vertical}");
        let x = l.caret(byte_index(text, 0)).0;
        let next = line_step(&l, text, 0, x, 1);
        assert!((start1..=n).contains(&next), "vertical {vertical} line_step -> {next}");
        assert_eq!(line_step(&l, text, 0, x, -1), 0, "vertical {vertical}");
        assert_eq!(line_step(&l, text, n, x, 1), n, "vertical {vertical}");
    }

    // A remembered column stays on the short line's start; the caret's own x falls off its end.
    let text = "WWWWWW\nI";
    let l = e.layout(&point(text, 30.0), 72.0);
    let end0 = line_edge(&l, text, 0, true);
    // end0 is the first line's end, which is not the second line. Step from there.
    let kept = line_step(&l, text, end0, l.caret(0).0, 1);
    let jumped = line_step(&l, text, end0, l.caret(byte_index(text, end0)).0, 1);
    assert_eq!(kept, char_index(text, l.lines[1].range.start), "kept column");
    assert_eq!(jumped, char_index(text, l.lines[1].range.end), "own column");
    assert!(jumped > kept);
}

#[test]
fn line_navigation_reaches_adjacent_line_with_large_leading() {
    use crate::layout::{byte_index, line_step};

    let mut e = TextEngine::new();
    let text = "ab\ncd";
    let mut t = point(text, 12.0);
    t.runs = vec![TextRun { len: text.len(), style: CharStyle { font_family: "Inter".into(), size_pt: 12.0, leading_pt: Some(40.0), ..Default::default() } }];
    let l = e.layout(&t, 72.0);

    assert_eq!(l.lines.len(), 2);
    assert!((l.lines[1].baseline - l.lines[0].baseline - 40.0).abs() < 0.01);
    let next = line_step(&l, text, 1, l.caret(byte_index(text, 1)).0, 1);
    assert_eq!(next, 4, "lineNext must reach the second line");
    let prev = line_step(&l, text, 4, l.caret(byte_index(text, 4)).0, -1);
    assert_eq!(prev, 1, "linePrev must reach the first line");
}

/// Thai text sample with above/below marks (sara i, mai ek, mai tho, mai han-akat, sara u).
const THAI_SAMPLE: &str = "ภาษาไทย สวัสดีครับ ผู้ที่น้ำ";

#[test]
fn thai_in_latin_font_falls_back_to_installed_thai_font() {
    // #1909: Thai typed in a Latin-only font (a newly chosen font has no PostScript name to find
    // the original Thai face) must fall back to an installed Thai-capable font, not .notdef.
    let mut e = TextEngine::with_system_fonts();
    let Some(thai) = fonts::THAI_FAMILIES.iter().find(|f| e.fonts.has_family(f)) else {
        eprintln!("skipped: no Thai-capable font installed");
        return;
    };
    let l = e.layout(&point(THAI_SAMPLE, 24.0), 72.0);
    assert!(!l.glyphs.is_empty());
    assert!(l.glyphs.iter().all(|g| g.id != 0), "Thai drawn with .notdef although {thai} is installed");
    // Latin next to Thai keeps the chosen font; only the Thai clusters fall back.
    let l = e.layout(&point("Thai ไทย", 24.0), 72.0);
    assert!(l.glyphs.iter().all(|g| g.id != 0));
    let (first, last) = (l.glyphs.first().map(|g| g.face), l.glyphs.last().map(|g| g.face));
    assert_ne!(first, last, "Latin and Thai drawn with the same face");
}

#[test]
fn thai_fallback_candidates_cover_every_platform() {
    // Logic-level half of #1909 (runs without Thai fonts): Windows, macOS and Linux each have a
    // Thai-capable family in the fallback candidates, ahead of the broad last-resort fonts.
    for order in [crate::cjk::script_order(None), crate::cjk::script_order(Some("ja"))] {
        let fb = fonts::fallback_candidates(&order);
        let last = fb.iter().position(|f| *f == "Arial Unicode MS").unwrap();
        for fam in ["Leelawadee UI", "Tahoma", "Thonburi", "Noto Sans Thai"] {
            let i = fb.iter().position(|f| *f == fam).unwrap_or_else(|| panic!("{fam} missing from {fb:?}"));
            assert!(i < last, "{fam} after the last-resort fonts");
        }
    }
}

/// The Southeast Asian scripts with their hard cases: Khmer coeng subscripts, the ah-sat mark and
/// numerals; Lao vowel and tone signs; Myanmar medials, stacks and dependent vowels (#218).
const SOUTHEAST_ASIAN: &[(&str, &str, &[&str])] = &[
    ("Khmer", "ភាសាខ្មែរ កម្ពុជា សួស្ដី រដ្ឋធម្ណុសាស្តៗ ១២៣", fonts::KHMER_FAMILIES),
    ("Lao", "ພາສາລາວ ສະບາຍດີ ຄ ໂລກ ຂອບໃຈ", fonts::LAO_FAMILIES),
    ("Myanmar", "မြန်မာဘာသာ မင်္ဂလာပါ ကျေးဇူးတင်ပါတယ်", fonts::MYANMAR_FAMILIES),
];

#[test]
fn southeast_asian_scripts_fall_back_to_installed_fonts() {
    // Detection and selection are tested apart, so a failure names its layer.
    for (name, sample, families) in SOUTHEAST_ASIAN {
        let mut e = TextEngine::with_system_fonts();
        let Some(hit) = families.iter().find(|f| e.fonts.has_family(f)) else {
            eprintln!("skipped: no {name}-capable font installed");
            continue;
        };
        // Layer 1, detection + loading + shaping: the installed family, asked for by name.
        let direct = TextLayer { text: (*sample).into(), font_family: hit.to_string(), size_pt: 24.0, ..Default::default() };
        let l = e.layout(&direct, 72.0);
        assert!(!l.glyphs.is_empty() && l.glyphs.iter().all(|g| g.id != 0), "{hit} is installed but shapes .notdef for {name}");
        // Layer 2, selection: the same text in a Latin-only font falls back instead of drawing
        // tofu, and the Latin around it keeps the chosen face.
        let mixed = format!("Latin {sample}");
        let l = e.layout(&point(&mixed, 24.0), 72.0);
        assert!(l.glyphs.iter().all(|g| g.id != 0), "{name} in a Latin font drawn with .notdef although {hit} is installed");
        let latin = l.glyphs.first().map(|g| g.face);
        assert!(l.glyphs.iter().any(|g| Some(g.face) != latin), "Latin and {name} drawn with the same face");
    }
}

#[test]
fn southeast_asian_fallback_candidates_cover_every_platform() {
    // Logic-level half of #218 (runs without these fonts): each script has a family for Windows,
    // macOS and Linux in the fallback candidates, ahead of the broad last-resort fonts.
    for order in [crate::cjk::script_order(None), crate::cjk::script_order(Some("ja"))] {
        let fb = fonts::fallback_candidates(&order);
        let last = fb.iter().position(|f| *f == "Arial Unicode MS").unwrap();
        for fam in [
            "Khmer UI",
            "Khmer Sangam MN",
            "Noto Sans Khmer",
            "Lao UI",
            "Lao Sangam MN",
            "Noto Sans Lao",
            "Myanmar Text",
            "Myanmar Sangam MN",
            "Noto Sans Myanmar",
        ] {
            let i = fb.iter().position(|f| *f == fam).unwrap_or_else(|| panic!("{fam} missing from {fb:?}"));
            assert!(i < last, "{fam} after the last-resort fonts");
        }
    }
}

/// The Indic scripts of document text, with conjuncts, matras and viramas (#218).
const INDIC: &[(&str, &str, &[&str])] = &[
    ("Devanagari", "हिन्दी भाषा क्षत्रिय श्री", fonts::DEVANAGARI_FAMILIES),
    ("Bengali", "বাংলা ভাষা ক্ষত্রিয় শ্রী", fonts::BENGALI_FAMILIES),
    ("Gurmukhi", "ਪੰਜਾਬੀ ਭਾਸ਼ਾ ਸਤਿ ਸ੍ਰੀ", fonts::GURMUKHI_FAMILIES),
    ("Gujarati", "ગુજરાતી ભાષા ક્ષત્રિય શ્રી", fonts::GUJARATI_FAMILIES),
    ("Tamil", "தமிழ் மொழி ஸ்ரீ கொடை", fonts::TAMIL_FAMILIES),
    ("Telugu", "తెలుగు భాష క్షత్రియ శ్రీ", fonts::TELUGU_FAMILIES),
    ("Kannada", "ಕನ್ನಡ ಭಾಷೆ ಕ್ಷತ್ರಿಯ ಶ್ರೀ", fonts::KANNADA_FAMILIES),
    ("Malayalam", "മലയാളം ഭാഷ ക്ഷത്രിയ ശ്രീ", fonts::MALAYALAM_FAMILIES),
    ("Sinhala", "සිංහල භාෂාව ක්‍ෂත්‍රිය ශ්‍රී", fonts::SINHALA_FAMILIES),
];

#[test]
fn indic_scripts_fall_back_to_installed_fonts() {
    for (name, sample, families) in INDIC {
        let mut e = TextEngine::with_system_fonts();
        let Some(hit) = families.iter().find(|f| e.fonts.has_family(f)) else {
            eprintln!("skipped: no {name}-capable font installed");
            continue;
        };
        let direct = TextLayer { text: (*sample).into(), font_family: hit.to_string(), size_pt: 24.0, ..Default::default() };
        let l = e.layout(&direct, 72.0);
        assert!(!l.glyphs.is_empty() && l.glyphs.iter().all(|g| g.id != 0), "{hit} is installed but shapes .notdef for {name}");
        let mixed = format!("Latin {sample}");
        let l = e.layout(&point(&mixed, 24.0), 72.0);
        assert!(l.glyphs.iter().all(|g| g.id != 0), "{name} in a Latin font drawn with .notdef although {hit} is installed");
        let latin = l.glyphs.first().map(|g| g.face);
        assert!(l.glyphs.iter().any(|g| Some(g.face) != latin), "Latin and {name} drawn with the same face");
    }
}

#[test]
fn indic_fallback_candidates_cover_every_platform() {
    // Logic-level half (runs without these fonts): every script has a Windows, macOS and Linux
    // family in the candidates, ahead of the broad last-resort fonts; the shared Nirmala UI is
    // probed once.
    for order in [crate::cjk::script_order(None), crate::cjk::script_order(Some("ja"))] {
        let fb = fonts::fallback_candidates(&order);
        let last = fb.iter().position(|f| *f == "Arial Unicode MS").unwrap();
        assert_eq!(fb.iter().filter(|f| **f == "Nirmala UI").count(), 1, "{fb:?}");
        for (name, _, families) in INDIC {
            assert!(families.contains(&"Nirmala UI"), "{name}");
            assert!(families.iter().any(|f| f.ends_with("Sangam MN") || f.starts_with("Kohinoor")), "{name}: no macOS family");
            assert!(families.iter().any(|f| f.starts_with("Noto Sans ")), "{name}: no Linux family");
            for fam in *families {
                let i = fb.iter().position(|f| f == fam).unwrap_or_else(|| panic!("{fam} missing from {fb:?}"));
                assert!(i < last, "{fam} after the last-resort fonts");
            }
        }
    }
}

#[test]
fn a_served_script_fallback_joins_the_fallback_stack_once_it_arrives() {
    let Some(cairo) = crate::CRAFT_FONTS.iter().find(|f| f.family == "Cairo") else {
        eprintln!("skipping: built without craft-fonts (set CRAFT_FONTS_DIR to a craft-fonts checkout)");
        return;
    };
    crate::served::add_fonts(&crate::served::parse_manifest("Cairo | Regular | fonts/cairo/Cairo.ttf | Arab,Latn\n").0);
    let mut db = fonts::FontDb::new();
    assert!(!db.fallback_stack().any(|f| f == "Cairo"), "not before it arrives");
    db.register_font_data(cairo.bytes.to_vec());
    assert!(db.fallback_stack().any(|f| f == "Cairo"), "{:?}", db.fallback_stack().collect::<Vec<_>>());
}

/// #1317: Photoshop keeps the last line of a text box whose baseline is inside the box even
/// when its descenders hang below the bottom edge (three 35 pt lines of 25 pt type in a
/// 91.3 pt box). A line whose baseline falls outside the box is still dropped.
#[test]
fn box_lines_whose_descent_hangs_below_the_box_are_kept() {
    let mut e = TextEngine::new();
    let mut t = styled("one\ntwo\nthree", CharStyle { size_pt: 25.0, leading_pt: Some(35.0), ..Default::default() });
    t.shape = TextShape::Box { x: 0.0, y: 0.0, width: 300.0, height: 91.27 };
    let l = e.layout(&t, 72.0);
    assert_eq!(l.lines.len(), 3, "{:?}", l.lines.iter().map(|l| l.baseline).collect::<Vec<_>>());
    assert!(l.lines[2].baseline <= 91.27 + 0.5);
    t.shape = TextShape::Box { x: 0.0, y: 0.0, width: 300.0, height: 80.0 };
    assert_eq!(e.layout(&t, 72.0).lines.len(), 2, "the third baseline (about 88) is below an 80 pt box");
}

/// `Txt2` with one text object for `text` whose rendered frame is frame 0, and that frame:
/// for point text an anchor `point`, and a frame-to-text matrix `matrix` (recent Photoshop,
/// #1317).
fn txt2_with_frame(text: &str, point: Option<[f64; 2]>, matrix: [f64; 6]) -> Vec<u8> {
    let mut v = txt2_with_modes(text, &[(text.encode_utf16().count() + 1, 1)]);
    // Close the object's model early: `… >> >> ] >>` → `… >> /1 << /2 [ frame ] >> >> ] >>`.
    let tail = b"] >> >> >> ] >>";
    assert!(v.ends_with(tail));
    v.truncate(v.len() - tail.len());
    v.extend_from_slice(b"] >> >> /1 << /2 [ << /6 [ << /10 0 >> ] >> ] >> >> ] >>");
    let pt = point.map(|[x, y]| format!("/0 [ {x:.5} {y:.5} ] ")).unwrap_or_default();
    let m = matrix.map(|x| format!("{x:.5}")).join(" ");
    let frames = format!("/0 << /8 << /0 [ << /0 << {pt}/2 << /2 [ {m} ] >> >> >> ] >> >>");
    let at = v.windows(9).position(|w| w == b"/0 << >> ").unwrap();
    v.splice(at..at + 9, frames.into_bytes().into_iter().chain(*b" "));
    v
}

/// #1317: point text whose anchor Photoshop keeps only in its `Txt2` frame (EngineData's
/// `PointBase` at the origin, the transform far away) is placed at that anchor, not at the
/// ink's top left. A rewritten `TySh` (bounds near the origin) is left alone.
#[test]
fn txt2_frame_places_point_text_at_its_anchor() {
    let mut t = point("2", 100.0);
    t.transform = Affine::translate(-1000.0, -2000.0);
    // Photoshop's text space: anchor at (1100, 2050), ink above its baseline.
    let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, Some([1100.0, 1980.0, 1160.0, 2050.0])), 0);
    let txt2 = crate::psd::parse_txt2(&txt2_with_frame("2", Some([1000.0, 2100.0]), [1.0, 0.0, 0.0, 1.0, 100.0, -50.0])).unwrap();
    let mut l = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    crate::psd::apply_txt2_frame(&mut l, &tysh, &txt2);
    let o = l.transform.apply(photocraft_geom::Point::new(0.0, 0.0));
    assert!((o.x - 100.0).abs() < 1e-3 && (o.y - 50.0).abs() < 1e-3, "anchor at {o:?}");
    // Already folded in (bounds near the origin): unchanged; another text: unchanged.
    let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, Some([0.0, -70.0, 60.0, 0.0])), 0);
    let mut l = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    let before = l.transform;
    crate::psd::apply_txt2_frame(&mut l, &tysh, &txt2);
    assert_eq!(l.transform, before);
    let tysh = with_text_index(&crate::psd::build_tysh(&t, 72.0, Some([1100.0, 1980.0, 1160.0, 2050.0])), 0);
    let other = crate::psd::parse_txt2(&txt2_with_frame("3", Some([1000.0, 2100.0]), [1.0, 0.0, 0.0, 1.0, 100.0, -50.0])).unwrap();
    let mut l = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    let before = l.transform;
    crate::psd::apply_txt2_frame(&mut l, &tysh, &other);
    assert_eq!(l.transform, before);
}

/// `tysh` with its text-space `bounds` set to `r` (Photoshop's, where it drew the text).
fn with_bounds(tysh: &[u8], r: [f64; 4]) -> Vec<u8> {
    use photocraft_psd::descriptor::{Descriptor, Id, Value};
    let mut t = crate::psd::parse_tysh(tysh).unwrap();
    let mut d = Descriptor::default();
    for (k, v) in ["Left", "Top ", "Rght", "Btom"].iter().zip(r) {
        d.items.push((Id::new(k), Value::Double(v)));
    }
    t.text.items.retain(|(k, _)| !k.is("bounds"));
    t.text.items.push((Id::new("bounds"), Value::Descriptor(d)));
    crate::psd::write_tysh(&t)
}

/// #1317: a text box whose `BoxBounds` are in its `Txt2` frame's space is moved by the frame's
/// matrix onto the `bounds` Photoshop drew it at (it used to land off the canvas, so its text
/// "disappeared" when an edit started).
#[test]
fn txt2_frame_moves_box_text_onto_its_bounds() {
    let mut t = styled("Casemiro 45", CharStyle { size_pt: 25.0, ..Default::default() });
    t.shape = TextShape::Box { x: 500.0, y: 600.0, width: 300.0, height: 90.0 };
    t.transform = Affine::translate(-700.0, -300.0);
    let built = with_text_index(&crate::psd::build_tysh(&t, 72.0, None), 0);
    let tysh = with_bounds(&built, [960.0, 300.0, 1260.0, 390.0]);
    let txt2 = crate::psd::parse_txt2(&txt2_with_frame("Casemiro 45", None, [1.0, 0.0, 0.0, 1.0, 460.0, -300.0])).unwrap();
    let mut l = crate::psd::text_layer_from_tysh(&tysh, 72.0).unwrap();
    assert!(matches!(l.shape, TextShape::Box { .. }));
    crate::psd::apply_txt2_frame(&mut l, &tysh, &txt2);
    let TextShape::Box { x, y, .. } = l.shape else { panic!("not a box") };
    let o = l.transform.apply(photocraft_geom::Point::new(f64::from(x), f64::from(y)));
    assert!((o.x - 260.0).abs() < 1e-3 && (o.y - 0.0).abs() < 1e-3, "box top left at {o:?}");
    // PhotoCraft's own `TySh` (bounds = the box, frame already folded in): unchanged.
    let mut l = crate::psd::text_layer_from_tysh(&built, 72.0).unwrap();
    let before = l.transform;
    crate::psd::apply_txt2_frame(&mut l, &built, &txt2);
    assert_eq!(l.transform, before);
}

/// Preferences ▸ Type ▸ Missing Glyph Protection: off, only the default family is in the
/// fallback stack, so a face without the glyph shows the missing-glyph box instead of a
/// fallback's glyph (#1909).
#[test]
fn missing_glyph_protection_can_turn_the_fallback_stack_off() {
    let Some(cairo) = crate::CRAFT_FONTS.iter().find(|f| f.family == "Cairo") else {
        eprintln!("skipping: built without craft-fonts (set CRAFT_FONTS_DIR to a craft-fonts checkout)");
        return;
    };
    let mut db = fonts::FontDb::new();
    db.register_font_data(cairo.bytes.to_vec());
    assert!(db.missing_glyph_protection(), "on by default, as in Photoshop");
    assert!(db.fallback_stack().any(|f| f == "Cairo"));
    db.set_missing_glyph_protection(false);
    assert_eq!(db.fallback_stack().collect::<Vec<_>>(), [fonts::DEFAULT_FAMILY], "only the default family remains");
    db.set_missing_glyph_protection(true);
    assert!(db.fallback_stack().any(|f| f == "Cairo"), "turning it back on restores the coverage fonts");
}

/// The shared engine's missing-glyph protection follows the setting without waiting for the
/// system font scan (#4519). Under test the shared engine has no scan, so it applies at once;
/// the newest setting wins.
#[test]
fn shared_missing_glyph_protection_follows_the_setting() {
    let read = || crate::shared().lock().unwrap_or_else(std::sync::PoisonError::into_inner).fonts.missing_glyph_protection();
    crate::set_shared_missing_glyph_protection(false);
    assert!(!read());
    crate::set_shared_missing_glyph_protection(true);
    crate::set_shared_missing_glyph_protection(false);
    assert!(!read(), "the newest setting wins");
    crate::set_shared_missing_glyph_protection(true);
    assert!(read());
}

#[test]
fn translate_tysh_moves_only_the_translation() {
    let mut t = point("Hello!", 26.0);
    t.transform = Affine { m: [1.5, 0.0, 0.0, 1.5, 80.0, 340.0] };
    let tysh = crate::psd::build_tysh(&t, 72.0, Some([0.0, -20.0, 300.0, 4.0]));
    let moved = crate::psd::translate_tysh(&tysh, -8.0, -279.0).unwrap();
    let back = crate::psd::parse_tysh(&moved).unwrap();
    assert_eq!(back.transform.m, [1.5, 0.0, 0.0, 1.5, 72.0, 61.0]);
    // Everything but the 16 translation bytes is unchanged.
    assert_eq!(moved.len(), tysh.len());
    assert_eq!((&moved[..34], &moved[50..]), (&tysh[..34], &tysh[50..]));
    assert_eq!(crate::psd::parse_tysh(&tysh).unwrap().bounds, back.bounds, "bounds are in text space");
    assert!(crate::psd::translate_tysh(&tysh[..40], 1.0, 1.0).is_none(), "a truncated block is refused");
}

/// The four values after the `TySh` warp descriptor are big-endian f32, as Photoshop writes them:
/// zeros for unwarped type, the text-space bounds for warped type.
#[test]
fn tysh_trailing_bounds_are_f32() {
    use photocraft_doc::text::TextWarp;
    let tail = |b: &[u8]| -> [f32; 4] {
        let t = b.get(b.len() - 16..).unwrap();
        std::array::from_fn(|i| f32::from_be_bytes(t[i * 4..i * 4 + 4].try_into().unwrap()))
    };
    let ink = [0.0, -34.316406, 288.90625, 15.058594];
    let plain = crate::psd::build_tysh(&point("Type", 20.0), 72.0, Some(ink));
    assert_eq!(tail(&plain), [0.0; 4]);
    let warped = TextLayer {
        warp: Some(TextWarp { style: "warpArc".into(), value: 50.0, horizontal_distortion: 0.0, vertical_distortion: 0.0, horizontal: true }),
        ..point("Type", 20.0)
    };
    let bytes = crate::psd::build_tysh(&warped, 72.0, Some(ink));
    assert_eq!(tail(&bytes), ink);
    let parsed = crate::psd::parse_tysh(&bytes).unwrap();
    assert_eq!(parsed.bounds, ink);
    assert_eq!(crate::psd::write_tysh(&parsed), bytes);
}

/// EngineData's `AntiAlias` numbers as Photoshop writes them, beside the `AntA` descriptor:
/// 0 None, 1 Crisp, 2 Strong, 3 Smooth, 4 Sharp. Read back from EngineData alone (no `AntA`),
/// every mode survives.
#[test]
fn engine_data_antialias_numbers_match_photoshop() {
    use photocraft_doc::text::AntiAlias;
    for (mode, n) in [(AntiAlias::None, 0), (AntiAlias::Crisp, 1), (AntiAlias::Strong, 2), (AntiAlias::Smooth, 3), (AntiAlias::Sharp, 4)] {
        let t = TextLayer { antialias: mode, ..point("Type", 20.0) };
        let mut tysh = crate::psd::parse_tysh(&crate::psd::build_tysh(&t, 72.0, None)).unwrap();
        let e = crate::psd::engine_data(&tysh.text).unwrap();
        assert_eq!(e.path(&["EngineDict", "AntiAlias"]).and_then(crate::engine_data::Value::as_f64), Some(f64::from(n)), "{mode:?}");
        tysh.text.items.retain(|(k, _)| !k.is("AntA"));
        let back = crate::psd::text_layer_from_tysh(&crate::psd::write_tysh(&tysh), 72.0).unwrap();
        assert_eq!(back.antialias, mode, "{n}");
    }
}

/// Type on a path (#218): along a vertical line the type runs top to bottom with its glyphs
/// turned, along a curve it follows the curve, and glyphs beyond the end are not drawn.
#[test]
fn type_on_a_path_follows_the_curve() {
    use photocraft_doc::vector::{Knot, Subpath};
    let fmt = PixelFormat::RGBA8;
    let mut eng = TextEngine::new();
    let ink = |eng: &mut TextEngine, t: &TextLayer| {
        let (_, r) = eng.render(t, 72.0, fmt);
        r.rect
    };
    let straight = point("Path text", 24.0);
    let flat = ink(&mut eng, &straight);
    assert!(flat.width() > flat.height() * 3, "{flat:?}");

    let mut down = straight.clone();
    down.shape = TextShape::Path { path: Subpath::polyline(&[(0.0, 0.0), (0.0, 400.0)]), start: 0.0, flip: false };
    let r = ink(&mut eng, &down);
    assert!(r.height() > r.width() * 3, "vertical path gives tall ink: {r:?}");
    assert!(r.height() >= flat.width() - 8, "arc length is preserved: {r:?} vs {flat:?}");

    // A half circle bends the text through a larger area than the straight line.
    let k = 0.5523 * 60.0;
    let arc = Subpath {
        closed: false,
        knots: vec![
            Knot {
                anchor: photocraft_geom::Point::new(-60.0, 0.0),
                in_ctrl: photocraft_geom::Point::new(-60.0, 0.0),
                out_ctrl: photocraft_geom::Point::new(-60.0, -k),
                smooth: true,
            },
            Knot {
                anchor: photocraft_geom::Point::new(0.0, -60.0),
                in_ctrl: photocraft_geom::Point::new(-k, -60.0),
                out_ctrl: photocraft_geom::Point::new(k, -60.0),
                smooth: true,
            },
            Knot {
                anchor: photocraft_geom::Point::new(60.0, 0.0),
                in_ctrl: photocraft_geom::Point::new(60.0, -k),
                out_ctrl: photocraft_geom::Point::new(60.0, 0.0),
                smooth: true,
            },
        ],
        ..Default::default()
    };
    let mut curved = straight.clone();
    curved.shape = TextShape::Path { path: arc.clone(), start: 0.0, flip: false };
    let c = ink(&mut eng, &curved);
    assert!(c.height() > flat.height() * 2, "curved ink is taller: {c:?} vs {flat:?}");

    // A path too short for the text hides the glyphs past its end.
    let mut short = straight.clone();
    short.shape = TextShape::Path { path: Subpath::polyline(&[(0.0, 0.0), (30.0, 0.0)]), start: 0.0, flip: false };
    assert!(ink(&mut eng, &short).width() < flat.width() / 2);
    // A degenerate path falls back to straight point text instead of failing.
    let mut dot = straight.clone();
    dot.shape = TextShape::Path { path: Subpath::polyline(&[(5.0, 5.0), (5.0, 5.0)]), start: 0.0, flip: false };
    assert_eq!(ink(&mut eng, &dot), flat);
}

#[test]
fn hyphenless_postscript_names_split_trailing_styles() {
    for (ps, family, weight, italic) in [
        ("GmarketSansBold", "Gmarket Sans", 700, false),
        ("NanumGothicBold", "Nanum Gothic", 700, false),
        ("GmarketSansTTFBold", "Gmarket Sans TTF", 700, false),
        ("ExampleSansSemiBoldItalic", "Example Sans", 600, true),
        ("ExampleSansExtraLight", "Example Sans", 200, false),
        // A style word after a vendor tag belongs to a legacy family name; lowercase is no style.
        ("ArialRoundedMTBold", "Arial Rounded MT Bold", 400, false),
        ("Daylight", "Daylight", 400, false),
    ] {
        let r = fonts::guess_from_postscript(ps);
        assert_eq!((r.family.as_str(), r.weight, r.italic), (family, weight, italic), "{ps}");
    }
}
