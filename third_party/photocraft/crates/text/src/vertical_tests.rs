//! Vertical type (tategaki): column flow, upright CJK, vertical alternates, rotated Latin,
//! box wrapping, carets and hit tests. Tests needing a CJK font use an installed system font
//! and skip (with a note) when none is found.

use photocraft_color::PixelFormat;
use photocraft_doc::TextLayer;
use photocraft_doc::text::{CharStyle, Orientation, TextRun, TextShape};
use photocraft_geom::Affine;

use crate::TextEngine;
use crate::layout::GlyphOrient;

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
fn orientation_uses_cjk_composer_rules_and_unicode_graphemes() {
    use crate::vertical::VerticalForm;
    let form = |c, japanese| VerticalForm::of(c, japanese, None, None);
    for c in ['縦', 'か', '한', '１', '、', '。', '！', '…', '—', '·', 'ー', '（', '：', '；', '～',
              '「', '」', '『', '』', '〝', '〞', '★', '☆', '♪', '᙮'] {
        assert_eq!(form(c, false), VerticalForm::Upright, "{c}");
    }
    for c in ['a', 'ｶ', '0', '!', '?', '.', ',', '“', '”', '‘', '’', '§', '†', '‰', '‼', '⁈', '⁇', '⁂', '⋯'] {
        assert_eq!(form(c, false), VerticalForm::Sideways, "{c}");
    }
    assert_eq!(form('·', true), VerticalForm::Sideways);
    assert_eq!(form('・', true), VerticalForm::Upright);
}

#[test]
fn sideways_runs_keep_font_advances_and_source_carets() {
    let mut engine = TextEngine::new();
    let text = vertical("A019-B /", "Inter", 40.0);
    let before = text.clone();
    let v = engine.layout(&text, 72.0);
    let h = engine.layout(&TextLayer { orientation: Orientation::Horizontal, ..text.clone() }, 72.0);
    assert_eq!(text, before);
    assert_eq!(v.glyphs.len(), text.text.chars().count());
    assert_eq!(v.glyphs.len(), h.glyphs.len());
    for (v, h) in v.glyphs.iter().zip(&h.glyphs) {
        assert_eq!(v.orient, GlyphOrient::Rotated);
        assert_eq!(v.id, h.id);
        assert!((v.y - h.x).abs() < 0.001);
    }
    for offset in 0..=text.text.len() {
        let [a, b] = v.caret_segment(offset);
        assert_eq!(v.hit_test((a.0 + b.0) * 0.5, a.1), offset);
    }
}

#[test]
fn font_punctuation_metrics_drive_wrap_and_carets_in_both_directions() {
    use photocraft_doc::text::FontFeature;

    let mut engine = TextEngine::new();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/[toolbox]宋体-简繁-Medium(v2.4).ttf");
    let Ok(bytes) = std::fs::read(path) else { return };
    let families = engine.fonts.register_font_data(bytes);
    let fam = &families[0];
    for direction in [Orientation::Horizontal, Orientation::Vertical] {
        let (width, height, feature) = if direction == Orientation::Horizontal {
            (105.0, 220.0, "palt")
        } else {
            (220.0, 105.0, "vpal")
        };
        let mut text = TextLayer {
            shape: TextShape::Box { x: 0.0, y: 0.0, width, height },
            orientation: direction,
            ..vertical("田、田", fam, 40.0)
        };
        let compact = engine.layout(&text, 72.0);
        assert_eq!(compact.lines.len(), 1);
        assert!((compact.clusters[1].advance - 20.0).abs() < 0.02);
        assert!((compact.clusters[2].x - compact.clusters[0].x - 60.0).abs() < 0.02);
        text.runs[0].style.features.push(FontFeature { tag: feature.into(), value: 0 });
        let full = engine.layout(&text, 72.0);
        assert_eq!(full.lines.len(), 2);
        assert!((full.clusters[1].advance - 40.0).abs() < 0.02);

        for (sample, available, expected) in [
            ("“田”", 75.0, vec!["“田", "”"]),
            ("田“田”", 90.0, vec!["田", "“田”"]),
            ("田“田”田", 105.0, vec!["田", "“田”", "田"]),
            ("田‘田’", 90.0, vec!["田", "‘田’"]),
            ("田……田", 105.0, vec!["田", "……", "田"]),
        ] {
            let shape = if direction == Orientation::Horizontal {
                TextShape::Box { x: 0.0, y: 0.0, width: available, height: 220.0 }
            } else {
                TextShape::Box { x: 0.0, y: 0.0, width: 220.0, height: available }
            };
            let text = TextLayer { shape, orientation: direction, ..vertical(sample, fam, 40.0) };
            let layout = engine.layout(&text, 72.0);
            let lines: Vec<_> = layout.lines.iter().map(|line| &sample[line.range.clone()]).collect();
            assert_eq!(lines, expected);
        }
    }
}

#[test]
fn japanese_spacing_uses_the_shaped_edges_before_wrapping_in_both_directions() {
    let mut engine = TextEngine::new();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/思源黑体SourceHanSansK-Bold.TTF");
    let Ok(bytes) = std::fs::read(path) else { return };
    let families = engine.fonts.register_font_data(bytes);
    let fam = &families[0];
    let brackets = engine.layout(&vertical("田「田」田", fam, 40.0), 72.0);
    let encoded = engine.layout(&vertical("田﹁田﹂田", fam, 40.0), 72.0);
    for index in [1, 3] {
        assert_eq!(brackets.glyphs[index].orient, GlyphOrient::Upright);
        assert_eq!(brackets.glyphs[index].id, encoded.glyphs[index].id);
    }
    for direction in [Orientation::Horizontal, Orientation::Vertical] {
        for (sample, expected) in [
            ("田ABC田", 20.0),
            ("田(田", 20.0),
            ("田“田", 20.0),
            ("田‘田", 20.0),
            ("田×田", 20.0),
            ("田†田", 0.0),
            ("田′田", 0.0),
            ("田‼田", 0.0),
            ("田｢田｣田", 0.0),
            ("田ｰ田", 0.0),
            ("田･田", 0.0),
            ("田、。田", -20.0),
            ("田「田」田", 0.0),
            ("田…田", 0.0),
            ("田！田", 0.0),
        ] {
            let text = TextLayer { orientation: direction, ..vertical(sample, fam, 40.0) };
            let implicit = engine.layout(&text, 72.0);
            let mut no_japanese = text.clone();
            no_japanese.runs[0].style.language = Some("zh-Hans".into());
            let plain = engine.layout(&no_japanese, 72.0);
            let width = |layout: &crate::TextLayout| layout.lines[0].x1 - layout.lines[0].x0;
            let delta = width(&implicit) - width(&plain);
            assert!((delta - expected).abs() < 0.05, "{direction:?} {sample}: expected {expected}, got {delta}");
            assert_eq!(implicit.clusters.len(), plain.clusters.len());
        }

        let text = TextLayer { orientation: direction, ..vertical("田A田", fam, 40.0) };
        let normal = engine.layout(&text, 72.0);
        let extent = normal.lines[0].x1 - 15.0;
        let (width, height) = if direction == Orientation::Horizontal { (extent, 250.0) } else { (250.0, extent) };
        let text = TextLayer { shape: TextShape::Box { x: 0.0, y: 0.0, width, height }, ..text };
        let wrapped = engine.layout(&text, 72.0);
        assert_eq!(wrapped.lines[0].range, 0.."田A".len());
        let a = &wrapped.clusters[1];
        let plain_a = normal.clusters[1].advance - 10.0;
        assert!((a.advance - plain_a).abs() < 0.05, "a wrapped edge discards the contextual gap");

        let available = 136.0;
        let (width, height) = if direction == Orientation::Horizontal { (available, 250.0) } else { (250.0, available) };
        let text = TextLayer {
            shape: TextShape::Box { x: 0.0, y: 0.0, width, height },
            orientation: direction,
            ..vertical("田「A」田", fam, 36.0)
        };
        let wrapped = engine.layout(&text, 72.0);
        assert_eq!(wrapped.lines[0].range, 0.."田「A」".len(), "{direction:?}");
    }
}

#[test]
fn localized_vertical_alternates_precede_cmap_fallback() {
    use skrifa::MetadataProvider;

    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../src/backend_v2/resources/fonts");
    for file in ["simhei.ttf", "simkai.ttf", "simfang.ttf"] {
        let Ok(bytes) = std::fs::read(directory.join(file)) else { continue };
        let mut engine = TextEngine::new();
        let families = engine.fonts.register_font_data(bytes);
        for language in [None, Some("zh-CN")] {
            for (ch, expected) in [('，', Some(1064)), ('。', Some(1045)), ('！', Some(1061)), ('？', Some(1069)), ('‥', None)] {
                let mut text = vertical(&format!("田{ch}田"), &families[0], 28.0);
                text.runs[0].style.language = language.map(str::to_string);
                let v = engine.layout(&text, 72.0);
                let h = engine.layout(&TextLayer { orientation: Orientation::Horizontal, ..text }, 72.0);
                let (v, normal) = (&v.glyphs[1], &h.glyphs[1]);
                let face = &h.faces[normal.face as usize];
                let font = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index).unwrap();
                assert_eq!(Some(normal.id), font.charmap().map(ch).map(|g| g.to_u32()));
                // The selected Han LangSys leaves the two-dot leader in its base form.
                assert_eq!((v.id, v.orient), (expected.unwrap_or(normal.id), GlyphOrient::Upright), "{file} {language:?} {ch}");
            }
        }
    }
}

#[test]
fn proportional_marks_keep_their_advance_in_a_vertical_box() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../src/backend_v2/resources/fonts");
    for (file, sample, advance) in [
        ("msyh.ttc", "田․田", 28.0 * 493.0 / 2048.0),
        ("msyhbd.ttc", "田․田", 28.0 * 586.0 / 2048.0),
        ("Arial_Unicode.ttf", "田·田", 28.0 * 569.0 / 2048.0),
    ] {
        let Ok(bytes) = std::fs::read(directory.join(file)) else { continue };
        let mut engine = TextEngine::new();
        let families = engine.fonts.register_font_data(bytes);
        let layer = TextLayer {
            shape: TextShape::Box { x: 0.0, y: 0.0, width: 110.0, height: 66.0 },
            ..vertical(sample, &families[0], 28.0)
        };
        let layout = engine.layout(&layer, 72.0);
        assert_eq!(layout.lines.len(), 1, "{file}");
        assert_eq!(layout.glyphs[1].orient, GlyphOrient::Rotated, "{file}");
        assert!((layout.clusters[1].advance - advance).abs() < 0.02, "{file}");
        assert!((layout.clusters[2].x - layout.clusters[0].x - (28.0 + advance)).abs() < 0.02, "{file}");
    }
}

#[test]
fn vertical_fallback_uses_the_ideographic_cell_and_authored_bearings() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../src/backend_v2/resources/fonts");
    for (file, sample, index, expected) in [
        ("[toolbox]淡古印-简繁(v2.4).ttf", "田~田", 0, 28.0 * 0.88),
        ("方正趣圆简体 Bold.TTF", "田（田", 1, 28.0 * 0.725),
    ] {
        let Ok(bytes) = std::fs::read(directory.join(file)) else { continue };
        let mut engine = TextEngine::new();
        let families = engine.fonts.register_font_data(bytes);
        let layer = vertical(sample, &families[0], 28.0);
        let layout = engine.layout(&layer, 72.0);
        let origin = layout.glyphs[index].y - layout.clusters[index].x;
        assert!((origin - expected).abs() < 0.1, "{file} {sample}: {origin}");
    }
}

#[test]
fn punctuation_orientation_uses_the_selected_faces_design() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../src/backend_v2/resources/fonts");
    let mut engine = TextEngine::new();
    for (file, sample, expected) in [
        ("[toolbox]宋体-简繁-Medium(v2.4).ttf", "田⋯田", GlyphOrient::Upright),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田⋯田", GlyphOrient::Rotated),
        ("NotoSans-Medium.ttf", "H…—―H", GlyphOrient::Rotated),
        ("msyh.ttc", "田·田", GlyphOrient::Upright),
        ("Arial_Unicode.ttf", "田–田", GlyphOrient::Rotated),
        ("Arial_Unicode.ttf", "田‐‾−田", GlyphOrient::Upright),
        ("SIMLI.TTF", "田–田", GlyphOrient::Rotated),
        ("msgothic.ttc", "田‐田", GlyphOrient::Upright),
        ("msgothic.ttc", "田‾−田", GlyphOrient::Upright),
        ("msgothic.ttc", "田—田", GlyphOrient::Upright),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田§†°″÷田", GlyphOrient::Upright),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田❲❳‾−田", GlyphOrient::Upright),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田‖‵田", GlyphOrient::Rotated),
        ("[toolbox]宋体-简繁-Medium(v2.4).ttf", "田§°″÷田", GlyphOrient::Rotated),
    ] {
        let Ok(bytes) = std::fs::read(directory.join(file)) else { continue };
        let families = engine.fonts.register_font_data(bytes);
        let text = vertical(sample, &families[0], 28.0);
        let layout = engine.layout(&text, 72.0);
        let marks = &layout.glyphs[1..layout.glyphs.len() - 1];
        assert!(marks.iter().all(|glyph| glyph.orient == expected), "{file} {sample}");
        assert_eq!(layout.clusters.last().unwrap().range.start, sample.char_indices().last().unwrap().0);
    }
}

#[test]
fn vertical_symbols_keep_the_authored_advance_and_presentation() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../../src/backend_v2/resources/fonts");
    for (file, sample, language, orient, vertical_advance, horizontal_advance) in [
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田❲田", None, GlyphOrient::Upright, 28.0, 14.0),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田❲田", Some("zh-Hans"), GlyphOrient::Rotated, 14.0, 14.0),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田❳田", Some("zh-Hant"), GlyphOrient::Rotated, 14.0, 14.0),
        ("ChisugaShotai_Regular0.2.12.1.ttf", "田‾田", None, GlyphOrient::Upright, 0.0, 0.0),
        ("Arial_Unicode.ttf", "田‐田", None, GlyphOrient::Upright, 28.0, 28.0 * 682.0 / 2048.0),
    ] {
        let Ok(bytes) = std::fs::read(directory.join(file)) else { continue };
        let mut engine = TextEngine::new();
        let families = engine.fonts.register_font_data(bytes);
        let mut layer = vertical(sample, &families[0], 28.0);
        layer.runs[0].style.language = language.map(str::to_string);
        let v = engine.layout(&layer, 72.0);
        let h = engine.layout(&TextLayer { orientation: Orientation::Horizontal, ..layer }, 72.0);
        assert_eq!(v.glyphs[1].orient, orient, "{file} {language:?} {sample}");
        assert!((v.clusters[1].advance - vertical_advance).abs() < 0.02, "{file} {sample}");
        assert!((h.clusters[1].advance - horizontal_advance).abs() < 0.02, "{file} {sample}");
    }
    for (file, form) in [("SIMLI.TTF", "田︴田"), ("Arial_Unicode.ttf", "田﹏田")] {
        let Ok(bytes) = std::fs::read(directory.join(file)) else { continue };
        let mut engine = TextEngine::new();
        let families = engine.fonts.register_font_data(bytes);
        let v = engine.layout(&vertical("田﹏田", &families[0], 28.0), 72.0);
        let h = engine.layout(&TextLayer { orientation: Orientation::Horizontal, ..vertical(form, &families[0], 28.0) }, 72.0);
        assert_eq!((v.glyphs[1].id, v.glyphs[1].orient), (h.glyphs[1].id, GlyphOrient::Upright), "{file}");
    }
}

#[test]
fn extended_punctuation_does_not_restore_space_removed_by_font_metrics() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/ChisugaShotai_Regular0.2.12.1.ttf");
    let Ok(bytes) = std::fs::read(path) else { return };
    let mut engine = TextEngine::new();
    let families = engine.fonts.register_font_data(bytes);
    for direction in [Orientation::Horizontal, Orientation::Vertical] {
        for sample in ["田〈田〉田", "田〘田〙田", "田〝田〟田"] {
            let mut text = TextLayer { orientation: direction, ..vertical(sample, &families[0], 28.0) };
            let default = engine.layout(&text, 72.0);
            text.runs[0].style.language = Some("zh-Hans".into());
            let no_japanese = engine.layout(&text, 72.0);
            for (a, b) in default.clusters.iter().zip(&no_japanese.clusters) {
                assert!((a.advance - b.advance).abs() <= 0.02, "{direction:?} {sample}");
            }
        }
    }
    let encoded = engine.layout(&vertical("田︵田︶田", &families[0], 28.0), 72.0);
    assert!((encoded.clusters[4].x - encoded.clusters[0].x - 112.0).abs() < 0.05);
}

#[test]
fn encoded_vertical_forms_use_the_same_face_without_rewriting_source_text() {
    use skrifa::MetadataProvider;

    let mut engine = TextEngine::new();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/思源黑体SourceHanSansK-Bold.TTF");
    let Ok(bytes) = std::fs::read(&path) else { return };
    let families = engine.fonts.register_font_data(bytes);
    for (source, form) in [('，', '\u{fe10}'), ('、', '\u{fe11}'), ('。', '\u{fe12}'), ('！', '\u{fe15}'), ('？', '\u{fe16}'), ('…', '\u{fe19}')] {
        let text = vertical(&source.to_string(), &families[0], 36.0);
        let layout = engine.layout(&text, 72.0);
        let glyph = &layout.glyphs[0];
        let face = &layout.faces[glyph.face as usize];
        let font = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index).unwrap();
        assert_eq!(Some(glyph.id), font.charmap().map(form).map(|id| id.to_u32()), "{form}");
        assert_eq!(layout.clusters[0].range, 0..source.len_utf8());
    }

    let authored = path.with_file_name("ChisugaShotai_Regular0.2.12.1.ttf");
    let Ok(bytes) = std::fs::read(authored) else { return };
    let family = engine.fonts.register_font_data(bytes);
    let layout = engine.layout(&vertical("田！田", &family[0], 36.0), 72.0);
    let glyph = &layout.glyphs[1];
    let face = &layout.faces[glyph.face as usize];
    let font = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index).unwrap();
    assert_eq!(Some(glyph.id), font.charmap().map('！').map(|id| id.to_u32()));
}

#[test]
fn narrow_japanese_text_distributes_needed_compression_across_punctuation_boundaries() {
    let mut engine = TextEngine::new();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/ChisugaShotai_Regular0.2.12.1.ttf");
    let Ok(bytes) = std::fs::read(path) else { return };
    let families = engine.fonts.register_font_data(bytes);
    for direction in [Orientation::Horizontal, Orientation::Vertical] {
        let mut text = TextLayer { orientation: direction, ..vertical("田“田”田", &families[0], 40.0) };
        let unwrapped = engine.layout(&text, 72.0);
        let (width, height) = if direction == Orientation::Horizontal { (125.0, 210.0) } else { (210.0, 125.0) };
        text.shape = TextShape::Box { x: 0.0, y: 0.0, width, height };
        text.overflow_visible = true;
        let wrapped = engine.layout(&text, 72.0);
        assert_eq!(wrapped.lines[0].range, 0.."田“田”".len(), "{direction:?}");
        assert_eq!(wrapped.lines.len(), 2);
        for (plain, fitted) in unwrapped.clusters[..3].iter().zip(&wrapped.clusters) {
            assert!((plain.advance - fitted.advance - 13.0 / 3.0).abs() < 0.06);
        }
        let exported = crate::psd::prepare_text_for_psd(&text, 72.0, &wrapped, &mut engine.fonts, true).unwrap();
        assert_eq!(exported.shape, text.shape, "the narrow frame already fits");
        assert_eq!(exported.text, text.text, "the existing wraps stay editable");
    }
}

#[test]
fn vertical_scaling_wraps_at_the_final_advance_and_centres_overflow() {
    use photocraft_doc::text::BlockAlign;

    let mut engine = TextEngine::new();
    let mut text = TextLayer {
        shape: TextShape::Box { x: 55.0, y: 40.0, width: 30.0, height: 115.0 },
        block_align: BlockAlign::Center,
        overflow_visible: true,
        ..vertical("1 2 3 4", "Inter", 40.0)
    };
    let reference = engine.layout(&TextLayer { orientation: Orientation::Horizontal, shape: TextShape::Point, ..text.clone() }, 72.0);
    let mut counts = Vec::new();
    for (width, height) in [(1.0, 0.7), (1.0, 1.0), (0.5, 1.6), (1.5, 1.6)] {
        text.runs[0].style.horizontal_scale = width;
        text.runs[0].style.vertical_scale = height;
        let layout = engine.layout(&text, 72.0);
        let columns = layout.lines.len();
        counts.push(columns);
        for c in layout.clusters.iter().filter(|c| c.range.start % 2 == 0) {
            let unscaled = reference.clusters.iter().find(|r| r.range == c.range).unwrap().advance;
            assert!((c.advance - unscaled * height).abs() < 0.02, "{c:?}");
        }
        assert!(layout.lines.windows(2).all(|pair| {
            let [a, b] = [&pair[0], &pair[1]];
            b.baseline - a.baseline >= a.descent + b.ascent - 0.01
        }), "automatic spacing must leave room for wider characters");
        let [left, _, right, _] = crate::render::ink_bounds(&layout, &Affine::IDENTITY, None).unwrap();
        assert!((left + right - 140.0).abs() < 0.001);
        if columns > 1 && width > 1.0 {
            assert!(left < 55.0 && right > 85.0);
        }
    }
    assert!(counts[0] < counts[2]);
    assert_eq!(counts[2], counts[3]);

    let mut glyph = vertical("A", "Inter", 40.0);
    let (_, normal) = engine.render(&glyph, 72.0, PixelFormat::RGBA8);
    glyph.runs[0].style.horizontal_scale = 0.65;
    glyph.runs[0].style.vertical_scale = 1.35;
    let (_, scaled) = engine.render(&glyph, 72.0, PixelFormat::RGBA8);
    assert!(scaled.rect.width() < normal.rect.width());
    assert!(scaled.rect.height() > normal.rect.height());
}

#[test]
fn comic_punctuation_preserves_mixed_character_styles_and_block_alignment() {
    let mut engine = TextEngine::new();
    let text = TextLayer {
        shape: TextShape::Box { x: 0.0, y: 0.0, width: 200.0, height: 100.0 },
        block_align: photocraft_doc::text::BlockAlign::Center,
        runs: vec![
            TextRun { len: 1, style: CharStyle { font_family: "Inter".into(), size_pt: 40.0, underline: true, ..Default::default() } },
            TextRun { len: 1, style: CharStyle { font_family: "Inter".into(), size_pt: 60.0, faux_bold: true, ..Default::default() } },
        ],
        ..vertical("!?", "Inter", 40.0)
    };
    let before = text.clone();
    let layout = engine.layout(&text, 72.0);
    assert_ne!(layout.glyphs[0].style, layout.glyphs[1].style);
    assert!((layout.glyphs[1].y - layout.glyphs[0].y).abs() < 0.01);
    assert!(layout.glyphs.iter().all(|g| g.orient == GlyphOrient::Upright));
    for (byte, _) in text.text.char_indices() {
        let [a, b] = layout.caret_segment(byte);
        assert_eq!(layout.hit_test(a.0, (a.1 + b.1) * 0.5), byte);
    }
    assert_eq!(text, before);
}

#[test]
fn comic_groups_are_upright_atomic_and_keep_source_characters() {
    let mut engine = TextEngine::new();
    for sample in ["!!", "!?", "?!", "??", "!!!", "！！！", "???", "？？？", "！！", "！？", "？！", "？？", "！?", "‼", "⁉", "⁈", "⁇", "12", "９９", "OK", "A1", "--"] {
        let mut text = vertical(sample, "Inter", 40.0);
        text.shape = TextShape::Box { x: 0.0, y: 0.0, width: 220.0, height: 35.0 };
        let layout = engine.layout(&text, 72.0);
        assert_eq!(layout.lines.len(), 1, "{sample}");
        assert!((layout.lines[0].x1 - layout.lines[0].x0 - 40.0).abs() < 0.02, "{sample}: {:?}", layout.lines);
        assert_eq!(layout.clusters.len(), sample.chars().count(), "{sample}");
        assert_eq!(layout.glyphs.len(), 2, "{sample}");
        assert!(layout.glyphs.iter().all(|g| g.orient == GlyphOrient::Upright));
        for (byte, _) in sample.char_indices() {
            let [a, b] = layout.caret_segment(byte);
            assert_eq!(layout.hit_test(a.0, (a.1 + b.1) / 2.0), byte, "{sample}");
        }
        text.orientation = Orientation::Horizontal;
        assert!(engine.layout(&text, 72.0).clusters.iter().all(|c| c.combined.is_none()));
    }

    for (sample, glyphs) in [("..", 1), ("...", 1), ("......", 2), ("……", 2), ("——", 2), ("～～", 2)] {
        let text = TextLayer { shape: TextShape::Box { x: 0.0, y: 0.0, width: 220.0, height: 30.0 }, ..vertical(sample, "Inter", 40.0) };
        let layout = engine.layout(&text, 72.0);
        assert_eq!(layout.glyphs.len(), glyphs, "{sample}");
        assert!(layout.glyphs.iter().all(|glyph| glyph.orient == GlyphOrient::Rotated), "{sample}");
        assert_eq!(layout.clusters.len(), sample.chars().count(), "{sample}");
        assert_eq!(layout.lines[0].range, 0..sample.len(), "{sample}");
    }
}

#[test]
fn legacy_comic_grouping_projects_from_source_once_for_layout_and_psd() {
    let mut engine = TextEngine::new();
    let layer = vertical("!!!田！？！田!?!?田<H>12</H>田<h>OK</h>田<E>—</E>田<H>LONG</H>", "Inter", 36.0);
    let layout = engine.layout(&layer, 72.0);
    let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
    let tysh = export.psd_raw.as_deref().unwrap();
    assert_eq!(export.text, "!!田!?\u{200c}!田!?\u{200c}!?田12田OK田—田LONG");
    assert_eq!(crate::psd::text_layer_from_tysh(tysh, 72.0).unwrap().text, export.text);
    assert!(!layout.clusters.iter().any(|c| layer.text[c.range.clone()].contains('<')));
    assert_eq!(layout.glyphs.iter().rev().take(4).filter(|g| g.orient == GlyphOrient::Rotated).count(), 4);

    let sample = "AB CD 9。09:30";
    let codes = crate::comic::Groups::for_text(sample, true);
    let spans: Vec<_> = codes.spans.iter().map(|g| &sample[g.range.clone()]).collect();
    assert_eq!(spans, ["09", "30"]);

    let text = vertical("<H>WW</H>", "Inter", 36.0);
    let layout = engine.layout(&text, 72.0);
    assert_eq!(layout.comic_scales.len(), 1);
    let expected = layout.comic_scales[&3];
    assert!(expected < 1.0);
    let export = crate::psd::prepare_text_for_psd(&text, 72.0, &layout, &mut engine.fonts, true).unwrap();
    let parsed = crate::psd::text_layer_from_tysh(export.psd_raw.as_deref().unwrap(), 72.0).unwrap();
    assert_eq!(parsed.text, "WW");
    assert!((parsed.runs[0].style.horizontal_scale - expected).abs() < 0.001);
}

#[test]
fn hidden_comic_tags_do_not_supply_glyph_styles() {
    let mut engine = TextEngine::new();
    let source = "A<E>—</E>B";
    let layer = TextLayer {
        runs: ["A", "<E>", "—", "</E>", "B"].into_iter().zip([36.0, 72.0, 45.0, 90.0, 36.0])
            .map(|(text, size_pt)| TextRun {
                len: text.len(),
                style: CharStyle { font_family: "Inter".into(), size_pt, ..Default::default() },
            }).collect(),
        ..vertical(source, "Inter", 36.0)
    };
    let layout = engine.layout(&layer, 72.0);
    let sizes: Vec<_> = layout.glyphs.iter().map(|glyph| layout.styles[glyph.style as usize].size_pt).collect();
    assert_eq!(sizes, [36.0, 45.0, 36.0]);
    let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
    assert_eq!(export.text, "A—B");
    assert_eq!(layer.text, source);
}

#[test]
fn frozen_psd_wraps_keep_the_source_western_phrase_sideways() {
    use crate::engine_data::Value as E;
    let mut engine = TextEngine::new();
    let layer = TextLayer {
        shape: TextShape::Box { x: 0.0, y: 0.0, width: 34.0, height: 54.0 },
        overflow_visible: true,
        ..vertical("AB CD", "Inter", 36.0)
    };
    let layout = engine.layout(&layer, 72.0);
    assert!(layout.lines.len() > 1 && layout.glyphs.iter().all(|glyph| glyph.orient == GlyphOrient::Rotated));
    let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
    assert!(export.text.contains('\n'));
    let parsed = crate::psd::parse_tysh(export.psd_raw.as_deref().unwrap()).unwrap();
    let data = crate::psd::engine_data(&parsed.text).unwrap();
    let runs = data.path(&["EngineDict", "StyleRun", "RunArray"]).and_then(E::as_array).unwrap();
    assert!(runs.iter().all(|run| run.path(&["StyleSheet", "StyleSheetData", "BaselineDirection"]).is_none()));
}

#[test]
fn psd_linear_fallback_prefers_a_proportional_advance() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/[toolbox]宋体-简繁-Medium(v2.4).ttf");
    let Ok(bytes) = std::fs::read(path) else { return };
    let mut engine = TextEngine::new();
    let families = engine.fonts.register_font_data(bytes);
    let layer = vertical("田――田", &families[0], 28.0);
    let layout = engine.layout(&layer, 72.0);
    let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
    assert_eq!(export.text, "田--田");
    assert_eq!(layer.text, "田――田");
}

#[test]
fn psd_centres_projected_comic_strokes_on_the_columns_ink_axis() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../../src/backend_v2/resources/fonts/ChisugaShotai_Regular0.2.12.1.ttf");
    let mut engine = TextEngine::new();
    let families = engine.fonts.register_font_data(std::fs::read(path).unwrap());
    for (source, displayed, size, width, projection) in [
        ("田<E>—</E>田", "田–田", 28.0, 1.0, 58.0 / 63.0),
        ("田<E>–</E>田", "田–田", 28.0, 1.0, 1.0),
        ("田——田", "田––田", 56.0, 0.8, 58.0 / 63.0),
        ("田<E>—</E>田", "田–田", 72.0, 1.25, 58.0 / 63.0),
    ] {
        let mut layer = vertical(source, &families[0], size);
        layer.runs[0].style.horizontal_scale = width;
        layer.runs[0].style.baseline_shift_pt = 1.5;
        let layout = engine.layout(&layer, 72.0);
        let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
        let parsed = crate::psd::text_layer_from_tysh(export.psd_raw.as_deref().unwrap(), 72.0).unwrap();
        assert_eq!(parsed.text, displayed);
        assert_eq!(parsed.runs.first().unwrap().style.baseline_shift_pt, 1.5);
        assert_eq!(parsed.runs.last().unwrap().style.baseline_shift_pt, 1.5);
        // The rotated en dash's ink midpoint is .2595 em and the Roman baseline is .38 em.
        let expected = 1.5 + (0.38 - 0.2595) * projection * size * width;
        assert!((parsed.runs[1].style.baseline_shift_pt - expected).abs() < 0.01);
        assert_eq!(layer.text, source);
        assert_eq!(layer.runs[0].style.baseline_shift_pt, 1.5);
    }
}

#[test]
fn psd_leading_preserves_scaled_and_mixed_line_positions() {
    let mut engine = TextEngine::new();
    for orientation in [Orientation::Horizontal, Orientation::Vertical] {
        for fixed in [false, true] {
            let layer = TextLayer {
                orientation,
                shape: TextShape::Box { x: 0.0, y: 0.0, width: 300.0, height: 300.0 },
                overflow_visible: true,
                runs: [32.0, 68.0, 44.0].into_iter().zip([0.8, 1.3, 0.84]).enumerate()
                    .map(|(i, (size_pt, scale))| TextRun {
                        len: if i == 2 { 1 } else { 2 },
                        style: CharStyle {
                            font_family: "Inter".into(), size_pt,
                            horizontal_scale: scale, vertical_scale: scale,
                            leading_pt: fixed.then_some([90.0, 62.0, 80.0][i]),
                            ..Default::default()
                        },
                    }).collect(),
                ..vertical("A\nB\nC", "Inter", 32.0)
            };
            let layout = engine.layout(&layer, 72.0);
            let copy = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
            let decoded = crate::psd::text_layer_from_tysh(copy.psd_raw.as_deref().unwrap(), 72.0).unwrap();
            assert_eq!(decoded.text, layer.text);
            for (i, pair) in layout.lines.windows(2).enumerate() {
                let vertical = orientation == Orientation::Vertical;
                let reference = if vertical { &pair[0] } else { &pair[1] };
                let mut runs = decoded.runs.iter().scan(0, |at, run| {
                    *at += run.len;
                    Some((*at, &run.style))
                });
                let (_, style) = runs.find(|(end, _)| *end > reference.range.start).unwrap();
                let leading = style.leading_pt.unwrap();
                let body_change = if vertical { [18.0, -12.0][i] } else { 0.0 };
                let real = pair[1].baseline - pair[0].baseline;
                assert!((real - (leading + body_change)).abs() < 0.01, "{orientation:?} {fixed} {leading} {real}");
            }
            assert_eq!(layer.runs[0].style.leading_pt, fixed.then_some(90.0));
        }
    }
}

#[test]
fn psd_tight_columns_preserve_spacing_with_native_baseline_shift() {
    let mut engine = TextEngine::new();
    let mut layer = vertical("A\nB", "Inter", 32.0);
    layer.shape = TextShape::Box { x: 0.0, y: 0.0, width: 120.0, height: 200.0 };
    layer.overflow_visible = true;
    layer.runs[0].len = 2;
    layer.runs[0].style.leading_pt = Some(12.0);
    layer.runs.push(TextRun {
        len: 1,
        style: CharStyle { size_pt: 68.0, leading_pt: Some(10.0), baseline_shift_pt: 2.0, ..layer.runs[0].style.clone() },
    });
    let layout = engine.layout(&layer, 72.0);
    let copy = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
    let parsed = crate::psd::text_layer_from_tysh(copy.psd_raw.as_deref().unwrap(), 72.0).unwrap();
    assert_eq!(parsed.runs[0].style.leading_pt, Some(6.4));
    assert!((parsed.runs[1].style.baseline_shift_pt - 16.4).abs() < 0.001);
    let TextShape::Box { width, .. } = parsed.shape else { panic!("expected a paragraph frame") };
    assert!((width - 134.4).abs() < 0.001);
    assert_eq!(layer.runs[1].style.baseline_shift_pt, 2.0);
}

#[test]
fn comic_psd_preserves_source_text_and_native_vertical_styles() {
    let mut engine = TextEngine::new();
    let layer = vertical("甲！？乙——丙...丁", "Inter", 36.0);
    let layout = engine.layout(&layer, 72.0);
    let tysh = crate::psd::build_layout_tysh(&layer, 72.0, &layout);
    let parsed = crate::psd::parse_tysh(&tysh).unwrap();
    let data = crate::psd::engine_data(&parsed.text).unwrap();
    assert_eq!(data.path(&["EngineDict", "Editor", "Text"]).and_then(crate::engine_data::Value::as_str), Some("甲！？乙——丙...丁\r"));
    let runs = data.path(&["EngineDict", "StyleRun"]).unwrap();
    let lengths = runs.get("RunLengthArray").and_then(crate::engine_data::Value::as_array).unwrap();
    let styles = runs.get("RunArray").and_then(crate::engine_data::Value::as_array).unwrap();
    let mut flags = Vec::new();
    for (run, length) in styles.iter().zip(lengths) {
        let style = run.path(&["StyleSheet", "StyleSheetData"]).unwrap();
        let cross = style.get("BaselineDirection").and_then(crate::engine_data::Value::as_i64) == Some(3);
        let no_break = style.get("NoBreak").and_then(crate::engine_data::Value::as_bool) == Some(true);
        flags.extend(std::iter::repeat_n((cross, no_break), length.as_i64().unwrap() as usize));
    }
    assert_eq!(flags, [
        (false, false), (true, false), (true, false), (false, false), (false, true), (false, true),
        (false, false), (false, true), (false, true), (false, true), (false, false), (false, false),
    ]);
}

#[test]
fn comic_psd_writes_visible_members_for_each_literal_cell() {
    let mut engine = TextEngine::new();
    let layer = vertical("甲！‼⁉乙\n丙⁈⁇...丁", "Inter", 36.0);
    let layout = engine.layout(&layer, 72.0);
    let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
    let tysh = export.psd_raw.as_ref().unwrap().as_slice();
    let parsed = crate::psd::text_layer_from_tysh(tysh, 72.0).unwrap();
    assert_eq!(parsed.text, "甲!\u{200C}!!\u{200C}!?乙\n丙?!\u{200C}??…丁");
    let data = crate::psd::engine_data(&crate::psd::parse_tysh(tysh).unwrap().text).unwrap();
    let runs = data.path(&["EngineDict", "StyleRun"]).unwrap();
    let lengths = runs.get("RunLengthArray").and_then(crate::engine_data::Value::as_array).unwrap();
    let styles = runs.get("RunArray").and_then(crate::engine_data::Value::as_array).unwrap();
    let spans: Vec<_> = styles.iter().zip(lengths).filter_map(|(run, length)| {
        let style = run.path(&["StyleSheet", "StyleSheetData"])?;
        (style.get("BaselineDirection")?.as_i64()? == 3).then(|| length.as_i64().unwrap())
    }).collect();
    assert_eq!(spans, [1, 2, 2, 2, 2]);
    assert!(crate::psd::build_composition_txt2(std::iter::once(tysh), &[(0, &export)]).is_some());
    assert_eq!(layer.text, "甲！‼⁉乙\n丙⁈⁇...丁");
}

#[test]
fn comic_psd_keeps_same_face_fallback_from_the_layouter() {
    let mut engine = TextEngine::new();
    for (direction, expected) in [(Orientation::Vertical, "甲!\u{200c}!乙"), (Orientation::Horizontal, "甲!!乙")] {
        let layer = TextLayer { orientation: direction, ..vertical("甲❕❗乙", "Inter", 36.0) };
        let layout = engine.layout(&layer, 72.0);
        let native = crate::psd::build_layout_tysh(&layer, 72.0, &layout);
        let export = crate::psd::prepare_text_for_psd(&layer, 72.0, &layout, &mut engine.fonts, true).unwrap();
        assert_eq!(crate::psd::text_layer_from_tysh(&native, 72.0).unwrap().text, "甲❕❗乙");
        assert_eq!(crate::psd::text_layer_from_tysh(export.psd_raw.as_deref().unwrap(), 72.0).unwrap().text, expected);
    }
}

#[test]
fn latin_is_rotated_and_runs_down_columns_right_to_left() {
    let mut e = TextEngine::new();
    let l = e.layout(&vertical("abc\ndef", "Inter", 20.0), 72.0);
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

#[test]
fn paragraph_first_baseline_uses_nominal_ascenders_at_any_character_scale() {
    use skrifa::{
        MetadataProvider,
        instance::{LocationRef, Size},
    };
    let mut engine = TextEngine::new();
    for (orientation, glyph, origin, factor) in [
        (Orientation::Horizontal, 'd', 20.0, 1.0),
        (Orientation::Vertical, 'H', -350.0, 0.5),
    ] {
        for size in [21.0, 42.0, 84.0] {
            for (width, height, leading) in [(1.0, 1.0, 1.0), (0.75, 1.25, 0.75), (1.5, 0.8, 1.5)] {
                let mut text = vertical("Abc\nDEF", "Inter", size);
                text.orientation = orientation;
                text.shape = TextShape::Box { x: 10.0, y: 20.0, width: 340.0, height: 400.0 };
                let style = &mut text.runs[0].style;
                style.leading_pt = Some(size * leading);
                style.horizontal_scale = width;
                style.vertical_scale = height;
                let before = text.clone();
                let layout = engine.layout(&text, 72.0);
                let face = &layout.faces[0];
                let font = skrifa::FontRef::from_index(face.font.data.as_ref(), face.font.index).unwrap();
                let id = font.charmap().map(glyph).unwrap();
                let ascent = font.glyph_metrics(Size::new(size), LocationRef::default()).bounds(id).unwrap().y_max;
                assert!((layout.lines[0].baseline - (origin + ascent * factor)).abs() < 0.001);
                assert!((layout.lines[1].baseline - layout.lines[0].baseline - size * leading).abs() < 0.001);
                assert_eq!(text, before, "layout must not move the stored frame or styles");
            }
        }
    }
}

#[test]
fn cjk_is_upright_centred_with_vertical_alternates() {
    let Some((mut e, fam)) = cjk_engine() else { return };
    let text = "縦書き、テスト。ーA";
    let l = e.layout(&vertical(text, &fam, 40.0), 72.0);
    let g = &l.glyphs;
    assert_eq!(g.len(), text.chars().count(), "{fam}");
    assert!(g.iter().all(|g| g.id != 0), "{fam}: no .notdef");
    // Ideographs and kana centre their full-width advance. Sentence marks can
    // have their own glyph metrics and optical placement within that column.
    for (i, gl) in g.iter().take(8).enumerate() {
        assert_eq!(gl.orient, GlyphOrient::Upright, "{fam}: glyph {i}");
        if ![3, 7].contains(&i) {
            assert!((gl.x + 20.0).abs() < 4.0, "{fam}: glyph {i} at {gl:?}");
        }
    }
    // Positions go down the column.
    assert!(g.windows(2).all(|w| w[1].y > w[0].y), "{fam}: {g:?}");
    // The first glyph's baseline is inside its em cell (vertical origin within the em).
    assert!(g[0].y > 20.0 && g[0].y < 40.0, "{fam}: {:?}", g[0]);
    // Compare with explicitly enabling the font's vertical substitutions. A
    // face can supply a dedicated form or keep the regular glyph for either mark.
    let mut alternate = TextLayer { orientation: Orientation::Horizontal, ..vertical(text, &fam, 40.0) };
    alternate.runs[0].style.features.push(photocraft_doc::text::FontFeature { tag: "vert".into(), value: 1 });
    let alternate = e.layout(&alternate, 72.0);
    for i in [3usize, 7] {
        assert_eq!(g[i].id, alternate.glyphs[i].id, "{fam}: char {i} should use the font's vertical form");
    }
    assert_eq!(g[8].orient, GlyphOrient::Upright);
    // Latin inside vertical CJK runs rotated.
    assert_eq!(g[9].orient, GlyphOrient::Rotated);
    let s = e.layout(&vertical("ト、「東", &fam, 40.0), 72.0);
    let bracket = e.layout(&vertical("「", &fam, 40.0), 72.0);
    assert!((s.glyphs[2].y - s.clusters[2].x - bracket.glyphs[0].y).abs() < 1.0, "{fam}: glyphs follow the clusters");
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

#[test]
fn psd_tysh_persists_the_layouters_paragraph_composition() {
    use crate::engine_data::Value as E;

    let mut text = vertical("田A、田\n田A、田\nWestern paragraph", "Inter", 24.0);
    text.shape = TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 110.0 };
    let first = text.text.find('\n').unwrap() + 1;
    let style = text.runs[0].style.clone();
    text.runs = vec![
        TextRun { len: first, style: CharStyle { language: Some("ja".into()), ..style.clone() } },
        TextRun { len: text.text.len() - first, style: CharStyle { language: Some("zh-Hans".into()), ..style } },
    ];
    let mut engine = TextEngine::new();
    for direction in [Orientation::Horizontal, Orientation::Vertical] {
        text.orientation = direction;
        let layout = engine.layout(&text, 72.0);
        let encoded = crate::psd::build_layout_tysh(&text, 72.0, &layout);
        let parsed = crate::psd::parse_tysh(&encoded).unwrap();
        let data = crate::psd::engine_data(&parsed.text).unwrap();
        let runs = data.path(&["EngineDict", "ParagraphRun", "RunArray"]).and_then(E::as_array).unwrap();
        for (index, run) in runs.iter().enumerate() {
            let properties = run.path(&["ParagraphSheet", "Properties"]).unwrap();
            assert_eq!(properties.get("MojiKumi").and_then(E::as_i64), (index == 0).then_some(1));
            assert_eq!(properties.get("EveryLineComposer").and_then(E::as_bool), Some(false));
            assert_eq!(properties.get("Kinsoku").and_then(E::as_i64), (index < 2).then_some(0));
            assert_eq!(properties.get("KinsokuOrder").and_then(E::as_i64), (index < 2).then_some(0));
        }
        let resources = data.path(&["ResourceDict", "KinsokuSet"]).and_then(E::as_array).unwrap();
        let strict = &resources[0];
        assert_eq!(strict.get("Name").and_then(E::as_str), Some("PhotoshopKinsokuHard"));
        let no_start = strict.get("NoStart").and_then(E::as_str).unwrap();
        assert!(['」', 'ぁ'].into_iter().all(|mark| no_start.contains(mark)));
        assert!(strict.get("NoEnd").and_then(E::as_str).unwrap().contains('「'));
        assert_eq!(data.path(&["ResourceDict", "KinsokuSet"]), data.path(&["DocumentResources", "KinsokuSet"]));
        let round_trip = crate::psd::text_layer_from_tysh(&encoded, 72.0).unwrap();
        assert_eq!(round_trip.text, text.text);
    }
}

#[test]
fn psd_tysh_preserves_an_imported_composer_and_kinsoku_choice() {
    use crate::engine_data::{self as ed, Value as E};
    use photocraft_psd::descriptor::Value as D;

    let text = TextLayer {
        shape: TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 110.0 },
        ..vertical("田、田", "Inter", 24.0)
    };
    let mut tysh = crate::psd::parse_tysh(&crate::psd::build_tysh(&text, 72.0, None)).unwrap();
    let template = ed::parse(br"<<
        /EngineDict << /ParagraphRun << /RunArray [
            << /ParagraphSheet << /Properties << /EveryLineComposer true /Kinsoku 1 /KinsokuOrder 2 >> >> >>
        ] >> >>
        /ResourceDict << /KinsokuSet [ << /Name /Other >> << /Name /Custom >> ] >>
    >>").unwrap();
    let (_, raw) = tysh.text.items.iter_mut().find(|(id, _)| id.is("EngineData")).unwrap();
    *raw = D::RawData(ed::write(&template));
    let text = TextLayer { psd_raw: Some(std::sync::Arc::new(crate::psd::write_tysh(&tysh))), ..text };
    let layout = TextEngine::new().layout(&text, 72.0);
    let encoded = crate::psd::build_layout_tysh(&text, 72.0, &layout);
    let parsed = crate::psd::parse_tysh(&encoded).unwrap();
    let data = crate::psd::engine_data(&parsed.text).unwrap();
    let runs = data.path(&["EngineDict", "ParagraphRun", "RunArray"]).and_then(E::as_array).unwrap();
    let settings = runs[0].path(&["ParagraphSheet", "Properties"]).unwrap();
    assert_eq!(settings.get("EveryLineComposer").and_then(E::as_bool), Some(true));
    assert_eq!(settings.get("Kinsoku").and_then(E::as_i64), Some(1));
    assert_eq!(settings.get("KinsokuOrder").and_then(E::as_i64), Some(2));
    assert_eq!(data.path(&["ResourceDict", "KinsokuSet"]), template.path(&["ResourceDict", "KinsokuSet"]));
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
/// CJK font, upright characters too.
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
    // The default punctuation spacing is unchanged by kerning an earlier pair.
    let gap = |l: &crate::TextLayout| l.glyphs[3].y - l.glyphs[2].y;
    assert!((gap(&k) - gap(&plain)).abs() < 1e-3);
}
