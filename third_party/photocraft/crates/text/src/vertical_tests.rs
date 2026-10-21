//! Vertical type (tategaki): column flow, upright CJK, vertical alternates, rotated Latin,
//! box wrapping, carets and hit tests. Tests needing a CJK font use an installed system font
//! and skip (with a note) when none is found.

use photocraft_color::PixelFormat;
use photocraft_doc::TextLayer;
use photocraft_doc::text::{CharStyle, Orientation, TextRun, TextShape};
use photocraft_geom::Affine;

use crate::TextEngine;
use crate::layout::{GlyphOrient, VClass, vertical_class};

fn vertical(text: &str, family: &str, size_pt: f32) -> TextLayer {
    TextLayer {
        text: text.into(),
        runs: vec![TextRun { len: text.len(), style: CharStyle { font_family: family.into(), size_pt, ..Default::default() } }],
        orientation: Orientation::Vertical,
        ..Default::default()
    }
}

/// An engine with an installed Japanese font registered, and that font's family.
fn cjk_engine() -> Option<(TextEngine, String)> {
    #[cfg(not(target_arch = "wasm32"))]
    for script in [crate::cjk::CjkScript::Japanese, crate::cjk::CjkScript::SimplifiedChinese] {
        for f in crate::cjk::font_files(script) {
            let Ok(bytes) = std::fs::read(&f.path) else { continue };
            let mut e = TextEngine::new();
            let fams = e.fonts.register_font_data(bytes);
            let want = if f.family.is_empty() { None } else { fams.iter().find(|n| n.as_str() == f.family) };
            if let Some(fam) = want.or(fams.first()).cloned() {
                return Some((e, fam));
            }
        }
    }
    eprintln!("no CJK system font found: skipping");
    None
}

#[test]
fn classes_follow_vertical_orientation() {
    for c in ['縦', 'か', 'カ', '한', '、', '。', 'Ａ', '１'] {
        assert_eq!(vertical_class(c), VClass::Upright, "{c}");
    }
    for c in ['ー', '「', '」', '（', '）', '〜', '…', '—'] {
        assert_eq!(vertical_class(c), VClass::AlternateOrRotate, "{c}");
    }
    for c in ['A', 'z', '1', ' ', '-', 'ｶ', 'é'] {
        assert_eq!(vertical_class(c), VClass::Rotate, "{c}");
    }
}

#[test]
fn latin_is_rotated_and_runs_down_columns_right_to_left() {
    let mut e = TextEngine::new();
    let l = e.layout(&vertical("abc\nde", "Inter", 20.0), 72.0);
    assert!(l.vertical);
    assert_eq!(l.lines.len(), 2);
    assert!(l.glyphs.iter().all(|g| g.orient == GlyphOrient::Rotated));
    let (a, b, c, d) = (l.glyphs[0], l.glyphs[1], l.glyphs[2], l.glyphs[3]);
    // Down the first column, on one vertical baseline.
    assert!(a.y < b.y && b.y < c.y, "{a:?} {b:?} {c:?}");
    assert!((a.x - b.x).abs() < 1e-3 && (b.x - c.x).abs() < 1e-3);
    // The second column is to the left (by the leading) and starts at the top again.
    assert!((a.x - d.x - 24.0).abs() < 1e-3, "{a:?} {d:?}");
    assert!((d.y - a.y).abs() < 1e-3);
    // Rotated text is centred on the column through the anchor (x = 0): its baseline sits left.
    assert!(a.x < 0.0 && a.x > -10.0, "{a:?}");
    // Bounds: one column for "abc" is taller than wide.
    let one = e.layout(&vertical("abcdef", "Inter", 20.0), 72.0);
    let [x0, y0, x1, y1] = one.bounds().unwrap();
    assert!((y1 - y0) > 2.0 * (x1 - x0), "{:?}", one.bounds());
    assert!((x0 + 10.0).abs() < 1e-3 && (x1 - 10.0).abs() < 1e-3 && y0.abs() < 1e-3);
}

#[test]
fn rotated_render_ink_is_taller_than_wide() {
    let mut e = TextEngine::new();
    let horizontal = TextLayer { orientation: Orientation::Horizontal, ..vertical("Vertical", "Inter", 30.0) };
    let (_, h) = e.render(&TextLayer { transform: Affine::translate(100.0, 50.0), ..horizontal.clone() }, 72.0, PixelFormat::RGBA8);
    let (_, v) = e.render(&TextLayer { transform: Affine::translate(100.0, 50.0), orientation: Orientation::Vertical, ..horizontal }, 72.0, PixelFormat::RGBA8);
    assert!(h.rect.width() > h.rect.height());
    assert!(v.rect.height() > 2 * v.rect.width(), "{:?}", v.rect);
    // The rotated word is the horizontal word turned: same ink length.
    assert!(v.rect.height().abs_diff(h.rect.width()) <= 3, "{:?} {:?}", v.rect, h.rect);
    // It runs down from the anchor, centred on x = 100.
    assert!(v.rect.y0 >= 49 && v.rect.x0 < 100 && v.rect.x1 > 100, "{:?}", v.rect);
}

#[test]
fn caret_hit_test_and_arrow_geometry_follow_columns() {
    let mut e = TextEngine::new();
    let l = e.layout(&vertical("abc\ndef", "Inter", 20.0), 72.0);
    // Above the first column's top → offset 0; below it → end of the first line.
    assert_eq!(l.hit_test(0.0, -5.0), 0);
    assert_eq!(l.hit_test(0.0, 1000.0), 3);
    // Left of the anchor is the second column.
    assert_eq!(l.hit_test(-24.0, -5.0), 4);
    // The caret is a horizontal segment across the column, lower for later offsets.
    let [(x0, y0), (x1, y1)] = l.caret_segment(1);
    assert!((y0 - y1).abs() < 1e-4 && x0 > x1 && (x0 - x1 - 20.0).abs() < 1e-3, "{x0} {y0} {x1} {y1}");
    let [(_, y2), _] = l.caret_segment(2);
    assert!(y2 > y0);
    // Round trip through line space.
    let (u, v) = l.to_line(-7.0, 33.0);
    assert_eq!(l.to_text(u, v), (-7.0, 33.0));
}

#[test]
fn box_text_wraps_columns_within_the_box_height() {
    let mut e = TextEngine::new();
    let t = TextLayer { shape: TextShape::Box { x: 10.0, y: 20.0, width: 200.0, height: 60.0 }, ..vertical("ab cd ef gh ij kl mn op", "Inter", 20.0) };
    let l = e.layout(&t, 72.0);
    assert!(l.lines.len() >= 3, "{}", l.lines.len());
    let [x0, y0, x1, y1] = l.bounds().unwrap();
    // The capital-height baseline can put part of the em/caret cell beyond the right edge.
    // Wrapping still follows the box height, rather than clipping that cell.
    assert!((210.0..220.0).contains(&x1), "{x1}");
    assert!(x0 >= 10.0 - 1e-3 && y0 >= 20.0 - 1e-3 && y1 <= 80.0 + 1e-3, "{:?}", l.bounds());
    for g in &l.glyphs {
        assert!(g.y >= 20.0 - 1e-3 && g.y <= 80.0 + 1e-3, "{g:?}");
    }
    // A narrow box drops the columns that don't fit (like overflowing lines).
    let narrow = TextLayer { shape: TextShape::Box { x: 0.0, y: 0.0, width: 30.0, height: 60.0 }, ..t };
    assert_eq!(e.layout(&narrow, 72.0).lines.len(), 1);
}

#[test]
fn horizontal_layout_is_unchanged_by_the_vertical_path() {
    let mut e = TextEngine::new();
    let t = TextLayer { orientation: Orientation::Horizontal, ..vertical("abc", "Inter", 20.0) };
    let l = e.layout(&t, 72.0);
    assert!(!l.vertical && l.glyphs.iter().all(|g| g.orient == GlyphOrient::Horizontal));
    assert_eq!(l.bounds(), l.line_bounds());
    assert_eq!(l.to_text(3.0, 4.0), (3.0, 4.0));
}

/// Photoshop's vertical paragraph baseline uses half the capital height, not half an em.
/// Verified in actual Photoshop at 21/42/84 px and several fonts/leading values.
#[test]
fn paragraph_first_column_uses_capital_height_without_changing_column_spacing() {
    use skrifa::{
        MetadataProvider,
        instance::{LocationRef, Size},
    };
    let mut engine = TextEngine::new();
    for size in [21.0, 42.0, 84.0] {
        for leading in [0.75, 1.0, 1.25, 1.5] {
            let mut text = vertical("Abc\nDEF", "Inter", size);
            text.shape = TextShape::Box { x: 10.0, y: 20.0, width: 340.0, height: 400.0 };
            text.runs[0].style.leading_pt = Some(size * leading);
            let before = text.clone();
            let layout = engine.layout(&text, 72.0);
            let face = &layout.faces[0];
            let font = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index).unwrap();
            let h = font.charmap().map('H').unwrap();
            let cap = font.glyph_metrics(Size::new(size), LocationRef::default()).bounds(h).unwrap().y_max;
            assert!((layout.lines[0].baseline - (-350.0 + cap * 0.5)).abs() < 0.001);
            assert!((layout.lines[1].baseline - layout.lines[0].baseline - size * leading).abs() < 0.001);
            assert_eq!(text, before, "layout must not move the stored frame or styles");
        }
    }
}

#[test]
fn cjk_is_upright_centred_with_vertical_alternates() {
    let Some((mut e, fam)) = cjk_engine() else { return };
    let text = "縦書き、テスト。ーA";
    let l = e.layout(&vertical(text, &fam, 40.0), 72.0);
    let h = e.layout(&TextLayer { orientation: Orientation::Horizontal, ..vertical(text, &fam, 40.0) }, 72.0);
    let g = &l.glyphs;
    assert_eq!(g.len(), text.chars().count(), "{fam}");
    assert!(g.iter().all(|g| g.id != 0), "{fam}: no .notdef");
    // Ideographs and kana: upright, centred on the column (full-width advance ≈ 40 px).
    for (i, gl) in g.iter().take(8).enumerate() {
        assert_eq!(gl.orient, GlyphOrient::Upright, "{fam}: glyph {i}");
        assert!((gl.x + 20.0).abs() < 4.0, "{fam}: glyph {i} at {gl:?}");
    }
    // Positions go down the column.
    assert!(g.windows(2).all(|w| w[1].y > w[0].y), "{fam}: {g:?}");
    // The first glyph's baseline is inside its em cell (vertical origin within the em).
    assert!(g[0].y > 20.0 && g[0].y < 40.0, "{fam}: {:?}", g[0]);
    // 、 and 。 use their vertical alternates (different glyphs than in horizontal type).
    for i in [3usize, 7] {
        assert_ne!(g[i].id, h.glyphs[i].id, "{fam}: char {i} should use its vertical alternate");
    }
    // ー: upright through its alternate when the font has one, else rotated.
    let dash = g[8];
    if dash.id != h.glyphs[8].id {
        assert_eq!(dash.orient, GlyphOrient::Upright);
    } else {
        assert_eq!(dash.orient, GlyphOrient::Rotated);
    }
    // Latin inside vertical CJK runs rotated.
    assert_eq!(g[9].orient, GlyphOrient::Rotated);
    // Punctuation squeeze: 、「 takes 1.5 em down the column, ト、 keeps full cells.
    let s = e.layout(&vertical("ト、「東", &fam, 40.0), 72.0);
    let cx: Vec<f32> = s.clusters.iter().map(|c| c.x).collect();
    assert!((cx[1] - cx[0] - 40.0).abs() < 1.0 && (cx[2] - cx[1] - 20.0).abs() < 1.0, "{fam}: {cx:?}");
    assert!((s.glyphs[2].y - s.glyphs[1].y - 20.0).abs() < 1.0, "{fam}: glyphs follow the clusters");
    // Rendered: the column is tall and narrow.
    let (_, r) = e.render(&TextLayer { transform: Affine::translate(900.0, 30.0), ..vertical("縦書きテスト", &fam, 40.0) }, 72.0, PixelFormat::RGBA8);
    assert!(r.rect.height() > 4 * r.rect.width(), "{fam}: {:?}", r.rect);
    assert!(r.rect.x0 > 870 && r.rect.x1 < 930 && r.rect.y0 >= 28 && r.rect.y1 < 30 + 6 * 40 + 4, "{fam}: {:?}", r.rect);
}

#[test]
fn psd_tysh_round_trips_orientation_and_writing_direction() {
    use crate::engine_data::Value as E;
    for o in [Orientation::Vertical, Orientation::Horizontal] {
        let t = TextLayer { orientation: o, ..vertical("縦書きAB", "Inter", 20.0) };
        let bytes = crate::psd::build_tysh(&t, 72.0, None);
        let back = crate::psd::text_layer_from_tysh(&bytes, 72.0).unwrap();
        assert_eq!(back.orientation, o);
        let parsed = crate::psd::parse_tysh(&bytes).unwrap();
        let ed = crate::psd::engine_data(&parsed.text).unwrap();
        let wd = ed.path(&["EngineDict", "Rendered", "Shapes", "WritingDirection"]).and_then(E::as_f64);
        assert_eq!(wd, Some(if o == Orientation::Vertical { 2.0 } else { 0.0 }));
    }
    // A file without `Ornt` but with a vertical writing direction imports as vertical.
    let t = TextLayer { orientation: Orientation::Vertical, ..vertical("ab", "Inter", 20.0) };
    let mut parsed = crate::psd::parse_tysh(&crate::psd::build_tysh(&t, 72.0, None)).unwrap();
    parsed.text.items.retain(|(k, _)| !k.is("Ornt"));
    let back = crate::psd::text_layer_from_tysh(&crate::psd::write_tysh(&parsed), 72.0).unwrap();
    assert_eq!(back.orientation, Orientation::Vertical);
}

/// Regular (400) from a family with only W3/W6 (Hiragino Mincho ProN) used W3 *and* asked for
/// synthetic bold, rendering a smeared bold serif. Only bold requests on non-bold faces embolden.
#[test]
fn nearest_face_is_not_synthetically_emboldened() {
    use crate::layout::synthetic_bold;
    assert!(!synthetic_bold(true, 400, 300.0), "regular from W3: use W3 as is");
    assert!(!synthetic_bold(true, 500, 300.0));
    assert!(synthetic_bold(true, 700, 400.0), "bold from a regular-only family");
    assert!(!synthetic_bold(true, 700, 600.0), "W6 is bold enough");
    assert!(!synthetic_bold(false, 900, 100.0), "matcher didn't ask");
    // With an installed W3/W6 family (macOS Hiragino Mincho), the regular request uses W3, unemboldened.
    #[cfg(target_os = "macos")]
    {
        let path = "/System/Library/Fonts/ヒラギノ明朝 ProN.ttc";
        let Ok(bytes) = std::fs::read(path) else { return };
        let mut e = TextEngine::new();
        let fams = e.fonts.register_font_data(bytes);
        let Some(fam) = fams.iter().find(|f| f.contains("Mincho")).cloned() else { return };
        let l = e.layout(&vertical("縦書き", &fam, 30.0), 72.0);
        assert!(!l.faces.is_empty());
        assert!(l.faces.iter().all(|f| !f.embolden), "{fam}");
        let bold = TextLayer {
            runs: vec![TextRun { len: "縦".len(), style: CharStyle { font_family: fam.clone(), size_pt: 30.0, weight: 700, ..Default::default() } }],
            ..vertical("縦", &fam, 30.0)
        };
        // W6 serves the bold request: still no synthesis.
        assert!(e.layout(&bold, 72.0).faces.iter().all(|f| !f.embolden));
    }
}

#[test]
fn punctuation_squeeze_removes_half_em_between_marks() {
    use crate::layout::{ClusterInfo, squeeze_punctuation};
    // Line-space clusters of 40 px full-width cells: ト、「東」。
    let text = "ト、「東」。";
    let mut clusters = Vec::new();
    let mut glyphs = Vec::new();
    let mut x = 0.0;
    for (i, (b, c)) in text.char_indices().enumerate() {
        clusters.push(ClusterInfo { range: b..b + c.len_utf8(), x, advance: 40.0, line: 0, rtl: false });
        glyphs.push(crate::PlacedGlyph { face: 0, id: i as u32 + 1, x, y: 0.0, style: 0, orient: GlyphOrient::Horizontal });
        x += 40.0;
    }
    let cut = squeeze_punctuation(text, &mut clusters, &mut glyphs);
    // 、「 and 」。 each lose half an em.
    assert_eq!(cut, 40.0);
    let xs: Vec<f32> = clusters.iter().map(|c| c.x).collect();
    assert_eq!(xs, vec![0.0, 40.0, 60.0, 100.0, 140.0, 160.0]);
    assert_eq!(glyphs.iter().map(|g| g.x).collect::<Vec<_>>(), xs);
    // Opening + opening: the second bracket's leading half goes.
    let text = "「「a";
    let mut cl: Vec<ClusterInfo> = text
        .char_indices()
        .enumerate()
        .map(|(i, (b, c))| ClusterInfo { range: b..b + c.len_utf8(), x: 40.0 * i as f32, advance: 40.0, line: 0, rtl: false })
        .collect();
    assert_eq!(squeeze_punctuation(text, &mut cl, &mut []), 20.0);
    assert_eq!(cl.iter().map(|c| c.x).collect::<Vec<_>>(), vec![0.0, 20.0, 60.0]);
    // Nothing to squeeze, and no panic on empty input or ranges outside the text.
    let mut none = vec![ClusterInfo { range: 0..3, x: 0.0, advance: 40.0, line: 0, rtl: false }];
    assert_eq!(squeeze_punctuation("東京", &mut none, &mut []), 0.0);
    let mut bad = vec![ClusterInfo { range: 50..60, x: 0.0, advance: 1.0, line: 0, rtl: false }; 3];
    assert_eq!(squeeze_punctuation("", &mut bad, &mut []), 0.0);
    assert_eq!(squeeze_punctuation("", &mut [], &mut []), 0.0);
}

#[test]
fn hostile_vertical_input_does_not_panic() {
    let mut e = TextEngine::new();
    for text in ["", "\n\n", "\u{200F}א\u{0301}", "「」ー。、\r\n\u{FFFF}", "a\u{0000}b"] {
        for shape in
            [TextShape::Point, TextShape::Box { x: 0.0, y: 0.0, width: 0.0, height: 0.0 }, TextShape::Box { x: f32::NAN, y: 1e30, width: -5.0, height: 1.0 }]
        {
            let t = TextLayer { shape, ..vertical(text, "Inter", 12.0) };
            let (l, _) = e.render(&t, 72.0, PixelFormat::RGBA8);
            let _ = (l.bounds(), l.hit_test(1.0, 2.0), l.caret_segment(1));
        }
    }
}

/// Manual kerning in vertical type moves the following glyphs down the column (#206); with a
/// CJK font, upright characters too, and the punctuation squeeze still applies.
#[test]
fn manual_kerning_runs_along_the_column() {
    use photocraft_doc::text::Kerning;
    let kerned = |t: &TextLayer, kern: f32| {
        let mut k = t.clone();
        let first = k.text.chars().next().map_or(0, char::len_utf8);
        let style = k.runs[0].style.clone();
        k.runs = vec![TextRun { len: first, style: CharStyle { kerning: Kerning::Off, kern, ..style.clone() } }, TextRun { len: k.text.len() - first, style }];
        k
    };
    let mut e = TextEngine::new();
    // Rotated Latin: 100/1000 em at 40 px = 4 px further down for the second glyph on.
    let t = vertical("HOH", "Inter", 40.0);
    let plain = e.layout(&t, 72.0);
    let k = e.layout(&kerned(&t, 100.0), 72.0);
    assert!(k.vertical);
    assert_eq!(k.glyphs[0].y, plain.glyphs[0].y);
    for i in 1..3 {
        assert!((k.glyphs[i].y - plain.glyphs[i].y - 4.0).abs() < 1e-3, "{i}: {} vs {}", k.glyphs[i].y, plain.glyphs[i].y);
        assert!((k.glyphs[i].x - plain.glyphs[i].x).abs() < 1e-3, "stays on the column");
    }
    let (b0, b1) = (plain.bounds().unwrap(), k.bounds().unwrap());
    assert!((b1[3] - b0[3] - 4.0).abs() < 1e-3, "the column grows by the kern");
    // Optical kerning in vertical type (rotated Latin) lays out without trouble.
    let mut opt = vertical("AVATAR", "Inter", 40.0);
    opt.runs[0].style.kerning = Kerning::Optical;
    let o = e.layout(&opt, 72.0);
    assert_eq!(o.glyphs.len(), 6);

    let Some((mut e, fam)) = cjk_engine() else { return };
    let t = vertical("東京、「都」", &fam, 40.0);
    let plain = e.layout(&t, 72.0);
    let k = e.layout(&kerned(&t, -250.0), 72.0);
    assert!((k.glyphs[1].y - plain.glyphs[1].y + 10.0).abs() < 1e-3, "upright CJK moves up by 10 px");
    // The squeeze between 、 and 「 is unchanged (same gap as without kerning).
    let gap = |l: &crate::TextLayout| l.glyphs[3].y - l.glyphs[2].y;
    assert!((gap(&k) - gap(&plain)).abs() < 1e-3);
}
