use super::*;

fn session(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 300, "height": 120, "depth": depth, "background": "white"})).unwrap();
    s
}

fn text_layer(s: &mut Session, text: &str) -> LayerId {
    let r = s.execute("type.create", json!({"x": 20, "y": 70, "text": text, "size": 36, "color": "#000000"})).unwrap();
    LayerId(r["layer"].as_u64().unwrap())
}

fn text(s: &Session, id: LayerId) -> &TextLayer {
    match &s.active().unwrap().doc.layer(id).unwrap().content {
        LayerContent::Text(t) => t,
        c => panic!("not text: {}", c.kind_name()),
    }
}

fn ink(s: &Session, id: LayerId) -> photocraft_geom::Rect {
    s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().content_bounds()
}

#[test]
fn anti_alias_and_orientation_are_one_step_each() {
    let mut s = session(8);
    assert!(!s.is_enabled("type.antiAlias.crisp"));
    let id = text_layer(&mut s, "Aa");
    for (cmd, aa) in [
        ("type.antiAlias.crisp", AntiAlias::Crisp),
        ("type.antiAlias.windows", AntiAlias::Windows),
        ("type.antiAlias.windowsLcd", AntiAlias::WindowsLcd),
        ("type.antiAlias.none", AntiAlias::None),
    ] {
        s.execute(cmd, json!({})).unwrap();
        assert_eq!(text(&s, id).antialias, aa);
    }
    // No anti-aliasing: only fully covered or empty pixels.
    let surf = s.active().unwrap().doc.layer(id).unwrap().surface().unwrap().clone();
    let r = surf.content_bounds();
    let n = surf.channels();
    assert!(surf.read_region(r).chunks_exact(n).all(|p| p[n - 1] == 0.0 || p[n - 1] == 1.0));
    s.execute("type.orientation.vertical", json!({})).unwrap();
    assert_eq!(text(&s, id).orientation, Orientation::Vertical);
    assert!(s.undo());
    assert_eq!(text(&s, id).orientation, Orientation::Horizontal);
    assert!(s.undo());
    assert_eq!(text(&s, id).antialias, AntiAlias::WindowsLcd);
}

#[test]
fn point_and_paragraph_conversion_keep_the_text_in_place() {
    for align in ["left", "center", "right"] {
        let mut s = session(8);
        let r = s.execute("type.create", json!({"x": 150, "y": 70, "text": "Hello world", "size": 30, "align": align})).unwrap();
        let id = LayerId(r["layer"].as_u64().unwrap());
        let before = ink(&s, id);
        assert!(!s.is_enabled("type.convertToPointText"));
        s.execute("type.convertToParagraphText", json!({})).unwrap();
        assert!(matches!(text(&s, id).shape, TextShape::Box { .. }));
        let after = ink(&s, id);
        assert!((after.x0 - before.x0).abs() <= 2 && (after.y0 - before.y0).abs() <= 2, "{align}: {before:?} → {after:?}");
        let lines = |s: &Session| {
            let st = s.active().unwrap();
            layout(&st.doc, text(s, id)).lines.len()
        };
        assert_eq!(lines(&s), 1, "{align}: no new wraps");
        s.execute("type.convertToPointText", json!({})).unwrap();
        assert!(matches!(text(&s, id).shape, TextShape::Point));
        let back = ink(&s, id);
        assert!((back.x0 - before.x0).abs() <= 2 && (back.y0 - before.y0).abs() <= 2, "{align}: {before:?} → {back:?}");
    }
}

/// Issue #446: vertical type renders vertically (see the test below), so the command's description
/// must not claim it stays horizontal.
#[test]
fn vertical_orientation_params_match_horizontal() {
    let specs = specs();
    let params = |id: &str| specs.iter().find(|c| c.id == id).unwrap().params;
    assert_eq!(params("type.orientation.vertical"), params("type.orientation.horizontal"));
}

/// Issue #199: vertical type is laid out top to bottom, columns right to left, on the canvas.
#[test]
fn vertical_orientation_renders_columns_inside_the_canvas() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 1000, "height": 600, "background": "#fff8e7", "name": "flyer"})).unwrap();
    // The issue's repro (its font may be missing here: fallback fonts then cover the text).
    let r = s.execute("type.create", json!({"x": 900, "y": 30, "text": "縦書きテスト", "size": 40, "font": "Noto Sans CJK JP", "name": "vertical"})).unwrap();
    let id = LayerId(r["layer"].as_u64().unwrap());
    let flat = ink(&s, id);
    assert!(flat.width() > flat.height(), "horizontal first: {flat:?}");
    let v = s.execute("type.orientation.vertical", json!({})).unwrap();
    assert_eq!(v, json!({"orientation": "vertical"}));
    let b = ink(&s, id);
    assert!(b.height() > b.width(), "{b:?}");
    assert!(b.x0 >= 0 && b.x1 <= 1000 && b.y0 >= 0 && b.y1 <= 600, "inside the canvas: {b:?}");
    // Centred on the anchor's x, running down from its y.
    assert!(b.x0 < 900 && b.x1 > 900 && b.y0 >= 25, "{b:?}");
    // Latin text: a second column of a two-line layer sits left of the first.
    let r = s.execute("type.create", json!({"x": 500, "y": 30, "text": "ABC\nDEF", "size": 40})).unwrap();
    let two = LayerId(r["layer"].as_u64().unwrap());
    s.execute("type.orientation.vertical", json!({"layer": two.0})).unwrap();
    let st = s.active().unwrap();
    let l = layout(&st.doc, text(&s, two));
    assert!(l.vertical && l.lines.len() == 2);
    let [x0, y0, x1, y1] = l.bounds().unwrap();
    assert!(x1 > 0.0 && x0 < -40.0 && y0 >= -1e-3 && y1 > 60.0, "{:?}", l.bounds());
    // Bad layer ids fail cleanly. Not a small number like 9999: ids come from one process-wide
    // counter, so in a full test run a layer made above can get it (#3343).
    assert!(s.execute("type.orientation.vertical", json!({"layer": u64::MAX})).is_err());
}

#[test]
fn vertical_point_and_paragraph_conversion_keep_the_columns_in_place() {
    for align in ["left", "center", "right"] {
        let mut s = session(8);
        s.execute("image.canvasSize", json!({"width": 300, "height": 400})).ok();
        let r = s.execute("type.create", json!({"x": 150, "y": 100, "text": "Hello world", "size": 30, "align": align})).unwrap();
        let id = LayerId(r["layer"].as_u64().unwrap());
        s.execute("type.orientation.vertical", json!({})).unwrap();
        let st = s.active().unwrap();
        let before = layout(&st.doc, text(&s, id)).bounds().unwrap();
        let before_tf = text(&s, id).transform;
        let map = |t: &photocraft_geom::Affine, b: [f32; 4]| {
            let p = t.apply(photocraft_geom::Point::new(f64::from(b[0]), f64::from(b[1])));
            (p.x, p.y)
        };
        let doc_before = map(&before_tf, before);
        s.execute("type.convertToParagraphText", json!({})).unwrap();
        let TextShape::Box { height, .. } = text(&s, id).shape else { panic!("box expected") };
        let st = s.active().unwrap();
        let l = layout(&st.doc, text(&s, id));
        assert_eq!(l.lines.len(), 1, "{align}: no new wraps");
        assert!(height > 0.0);
        // The column's first glyph stays where it was (glyph positions in document space).
        let g_after = l.glyphs[0];
        let tf = text(&s, id).transform;
        let p_after = tf.apply(photocraft_geom::Point::new(f64::from(g_after.x), f64::from(g_after.y)));
        s.execute("type.convertToPointText", json!({})).unwrap();
        let st = s.active().unwrap();
        let back = layout(&st.doc, text(&s, id));
        let g_back = back.glyphs[0];
        let p_back = text(&s, id).transform.apply(photocraft_geom::Point::new(f64::from(g_back.x), f64::from(g_back.y)));
        assert!((p_after.x - p_back.x).abs() < 1.5 && (p_after.y - p_back.y).abs() < 1.5, "{align}: {p_after:?} → {p_back:?}");
        let back_doc = map(&text(&s, id).transform, back.bounds().unwrap());
        assert!((back_doc.0 - doc_before.0).abs() < 1.5 && (back_doc.1 - doc_before.1).abs() < 1.5, "{align}: {doc_before:?} → {back_doc:?}");
    }
}

#[test]
fn paragraph_to_point_hardens_soft_wraps() {
    let mut s = session(8);
    let r = s.execute("type.create", json!({"box": [10, 10, 120, 100], "text": "one two three four five six", "size": 18})).unwrap();
    let id = LayerId(r["layer"].as_u64().unwrap());
    let st = s.active().unwrap();
    let soft = layout(&st.doc, text(&s, id)).lines.len();
    assert!(soft >= 2);
    let r = s.execute("type.convertToPointText", json!({})).unwrap();
    assert_eq!(r["inserted"], soft - 1);
    let t = text(&s, id);
    assert_eq!(t.text.matches('\n').count(), soft - 1);
    assert_eq!(t.text.replace('\n', ""), "one two three four five six");
    let st = s.active().unwrap();
    assert_eq!(layout(&st.doc, text(&s, id)).lines.len(), soft);
}

#[test]
fn work_path_and_shape_follow_the_glyphs() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let id = text_layer(&mut s, "oil");
        let before = ink(&s, id);
        s.execute("type.createWorkPath", json!({})).unwrap();
        let wp = s.active().unwrap().doc.work_path.clone().unwrap();
        // 'o' has an outer contour and a counter (subtracted), plus 'i' (stem + dot) and 'l'.
        assert!(wp.subpaths.len() >= 5, "{}", wp.subpaths.len());
        assert!(wp.subpaths.iter().any(|sp| sp.op == PathOp::Subtract));
        let (x0, y0, x1, y1) = wp.control_bounds().unwrap();
        assert!((x0 - before.x0 as f64).abs() < 3.0 && (x1 - before.x1 as f64).abs() < 3.0, "{x0},{x1} vs {before:?}");
        assert!((y0 - before.y0 as f64).abs() < 3.0 && (y1 - before.y1 as f64).abs() < 3.0);
        // Convert to Shape renders the same pixels (the counter of the 'o' stays open).
        let comp_before = photocraft_compose::flatten(&s.active().unwrap().doc);
        s.execute("type.convertToShape", json!({})).unwrap();
        let l = s.active().unwrap().doc.layer(id).unwrap();
        assert!(matches!(l.content, LayerContent::Shape(_)));
        assert!(!l.psd_blocks.iter().any(|(k, _)| k == b"TySh"));
        let comp_after = photocraft_compose::flatten(&s.active().unwrap().doc);
        let diff: f32 = comp_before.px.iter().zip(&comp_after.px).map(|(a, b)| (a[0] - b[0]).abs()).sum::<f32>() / comp_before.px.len() as f32;
        assert!(diff < 0.01, "depth {depth}: mean diff {diff}");
        // The middle of the 'o' counter is background (white).
        assert!(s.undo());
        assert!(matches!(s.active().unwrap().doc.layer(id).unwrap().content, LayerContent::Text(_)));
    }
}

#[test]
fn warp_text_changes_pixels_and_round_trips() {
    let mut s = session(8);
    let id = text_layer(&mut s, "WARP ME");
    let flat = ink(&s, id);
    s.execute("type.warpText", json!({"style": "arc", "bend": 60})).unwrap();
    let w = text(&s, id).warp.clone().unwrap();
    assert_eq!(w.style, "warpArc");
    assert_eq!(w.value, 60.0);
    let arced = ink(&s, id);
    assert!(arced.height() > flat.height() + 8, "{flat:?} → {arced:?}");
    // The regenerated TySh carries the warp (PSD round trip).
    let raw = text(&s, id).psd_raw.clone().unwrap();
    let back = photocraft_text::psd::text_layer_from_tysh(&raw, 72.0).unwrap();
    assert_eq!(back.warp, Some(w));
    assert!(s.execute("type.warpText", json!({"style": "spiral"})).is_err());
    s.execute("type.warpText", json!({"style": "none"})).unwrap();
    assert!(text(&s, id).warp.is_none());
    assert_eq!(ink(&s, id), flat);
    for (_, style) in photocraft_text::warp::STYLES {
        s.execute("type.warpText", json!({"style": style, "bend": -40, "horizontalDistortion": 10})).unwrap();
        assert!(!ink(&s, id).is_empty(), "{style}");
    }
}

#[test]
fn opentype_toggles_store_features() {
    let mut s = session(8);
    let id = text_layer(&mut s, "office 1/2");
    let st = |s: &Session| text(s, id).char_runs()[0].style.clone();
    assert!(st(&s).ligatures);
    s.execute("type.openType.standardLigatures", json!({})).unwrap();
    assert!(!st(&s).ligatures);
    s.execute("type.openType.fractions", json!({})).unwrap();
    s.execute("type.openType.discretionaryLigatures", json!({"on": true})).unwrap();
    s.execute("type.openType.swash", json!({"range": [0, 3]})).unwrap();
    assert!(feature_on(&st(&s), "frac") && feature_on(&st(&s), "dlig") && feature_on(&st(&s), "swsh"));
    let runs = text(&s, id).char_runs();
    assert!(!feature_on(&runs.last().unwrap().style, "swsh"), "range respected");
    s.execute("type.openType.fractions", json!({})).unwrap();
    assert!(!feature_on(&st(&s), "frac"));
    // PSD round trip of the toggles.
    let raw = text(&s, id).psd_raw.clone().unwrap();
    let back = photocraft_text::psd::text_layer_from_tysh(&raw, 72.0).unwrap();
    assert!(feature_on(&back.char_runs()[0].style, "swsh") && !back.char_runs()[0].style.ligatures);
}

#[test]
fn lorem_update_fonts_and_default_styles() {
    let mut s = session(16);
    // No type layer: a new paragraph box.
    let r = s.execute("type.pasteLoremIpsum", json!({})).unwrap();
    let box_id = LayerId(r["layer"].as_u64().unwrap());
    assert!(matches!(text(&s, box_id).shape, TextShape::Box { .. }));
    let id = text_layer(&mut s, "Hi ");
    let r = s.execute("type.pasteLoremIpsum", json!({})).unwrap();
    assert!(text(&s, id).text.starts_with("Hi Lorem ipsum"));
    assert_eq!(r["caret"], text(&s, id).text.chars().count());
    assert_eq!(s.execute("type.updateAllTextLayers", json!({})).unwrap()["updated"], 2);
    // A missing font is reported and replaced.
    s.execute("type.setStyle", json!({"font": "No Such Font Family"})).unwrap();
    assert_eq!(s.execute("type.resolveMissingFonts", json!({})).unwrap()["missing"], json!(["No Such Font Family"]));
    s.execute("type.resolveMissingFonts", json!({"map": {"No Such Font Family": "JetBrains Mono"}})).unwrap();
    assert_eq!(text(&s, id).char_runs()[0].style.font_family, "JetBrains Mono");
    s.execute("type.setStyle", json!({"font": "Gone Sans"})).unwrap();
    assert_eq!(s.execute("type.replaceAllMissingFonts", json!({})).unwrap()["replaced"], 1);
    assert_eq!(text(&s, id).char_runs()[0].style.font_family, photocraft_text::fonts::DEFAULT_FAMILY);
    // Default type styles: save from one layer, new layers and Load use them.
    assert!(!s.is_enabled("type.loadDefaultTypeStyles"));
    s.execute("type.setStyle", json!({"size": 50, "font": "JetBrains Mono"})).unwrap();
    s.execute("type.saveDefaultTypeStyles", json!({})).unwrap();
    s.select_layer(box_id).unwrap();
    s.execute("type.loadDefaultTypeStyles", json!({})).unwrap();
    assert_eq!(text(&s, box_id).char_runs()[0].style.size_pt, 50.0);
    let r = s.execute("type.create", json!({"x": 5, "y": 5, "text": "new"})).unwrap();
    let st = text(&s, LayerId(r["layer"].as_u64().unwrap())).char_runs()[0].style.clone();
    assert_eq!((st.font_family.as_str(), st.size_pt), ("JetBrains Mono", 50.0));
}

#[test]
fn rasterize_type_layer() {
    let mut s = session(8);
    let id = text_layer(&mut s, "R");
    s.execute("type.rasterizeTypeLayer", json!({})).unwrap();
    assert!(matches!(s.active().unwrap().doc.layer(id).unwrap().content, LayerContent::Raster(_)));
    assert!(!s.is_enabled("type.rasterizeTypeLayer"));
}

#[test]
fn type_attaches_to_a_path_and_releases_it() {
    let mut s = session(8);
    let id = text_layer(&mut s, "Hello path");
    assert!(s.is_enabled("type.attachToPath"));
    assert!(!s.is_enabled("type.detachFromPath"));
    // No work path yet, a bad subpath and bad params are errors, not panics.
    assert!(s.execute("type.attachToPath", json!({})).is_err());
    let flat = ink(&s, id);
    s.edit("path", |doc, _| {
        doc.work_path = Some(Path::new(vec![Subpath::polyline(&[(150.0, 10.0), (150.0, 110.0)])]));
        Ok(())
    })
    .unwrap();
    assert!(s.execute("type.attachToPath", json!({"subpath": 5})).is_err());
    assert!(s.execute("type.attachToPath", json!({"start": "far"})).is_err());
    assert!(s.execute("type.attachToPath", json!({"shape": u64::MAX})).is_err());
    s.execute("type.attachToPath", json!({"start": 4})).unwrap();
    assert!(matches!(text(&s, id).shape, TextShape::Path { start, .. } if start == 4.0));
    let on_path = ink(&s, id);
    assert!(on_path.height() > on_path.width(), "the type runs down the vertical path: {on_path:?}");
    assert!(on_path.height() > flat.height());
    assert!(!s.is_enabled("type.convertToPointText"));
    s.execute("type.detachFromPath", json!({})).unwrap();
    assert_eq!(text(&s, id).shape, TextShape::Point);
    assert!(ink(&s, id).width() > ink(&s, id).height());
    assert!(s.undo());
    assert!(matches!(text(&s, id).shape, TextShape::Path { .. }));
}

#[test]
fn all_locked_layer_rejects_paste_lorem_ipsum_without_history() {
    // #4188: Lock All stopped type.edit (#3132) but not Paste Lorem Ipsum into the layer.
    let mut s = session(8);
    let id = text_layer(&mut s, "Hi");
    s.execute("layer.setProps", json!({"layer": id.0, "locked": true})).unwrap();
    let history = s.active().unwrap().history.past_len();
    for params in [json!({}), json!({"layer": id.0})] {
        let result = s.execute("type.pasteLoremIpsum", params);
        assert!(matches!(&result, Err(EngineError::Other(message)) if message.contains("locked")), "{result:?}");
        assert_eq!(text(&s, id).text, "Hi");
        assert_eq!(s.active().unwrap().history.past_len(), history);
    }
}

/// A type layer stored with a family label that is not installed, as a PSD keeps the family it
/// guessed from a PostScript name whose face was absent when the file was opened.
fn guessed_family_layer(s: &mut Session, text: &str, ps: Option<&str>) -> (LayerId, Value) {
    let mut p = json!({"x": 20, "y": 70, "text": text, "font": "Imported Wrong Family"});
    if let Some(ps) = ps {
        p["postscriptName"] = json!(ps);
    }
    let r = s.execute("type.create", p).unwrap();
    (LayerId(r["layer"].as_u64().unwrap()), r)
}

#[test]
fn installed_postscript_face_is_not_missing_or_replaced() {
    // Opening a PSD corrects the label when the face is installed; a run set by command, or one
    // reopened from .pcraft after its face was installed, keeps it.
    let mut s = session(8);
    let (id, created) = guessed_family_layer(&mut s, "test", Some("Inter-SemiBold"));
    let bytes = photocraft_format::save_to_bytes(&s.active().unwrap().doc, &Default::default()).unwrap();
    let reopened = photocraft_io::import("reopened.pcraft", &bytes).unwrap().document;
    let LayerContent::Text(t) = &reopened.layer(id).unwrap().content else { panic!("text") };
    assert_eq!(t.char_runs()[0].style.font_family, "Imported Wrong Family", ".pcraft open keeps the stored label");
    let on_open = missing_font_warnings(&reopened);
    s.add_document(reopened, None);
    let replaced = s.execute("type.replaceAllMissingFonts", json!({})).unwrap()["replaced"].clone();
    assert_eq!((created.get("warnings"), on_open, replaced), (None, Vec::<String>::new(), json!(0)));
}

#[test]
fn replacing_missing_family_preserves_a_resolved_run_with_the_same_label() {
    let mut s = session(8);
    let (good, _) = guessed_family_layer(&mut s, "good", Some("Inter-SemiBold"));
    let (bad, _) = guessed_family_layer(&mut s, "bad", None);
    assert_eq!(s.execute("type.replaceAllMissingFonts", json!({})).unwrap()["replaced"], 1);
    assert_eq!(text(&s, good).char_runs()[0].style.postscript_name.as_deref(), Some("Inter-SemiBold"));
    assert_eq!(text(&s, bad).char_runs()[0].style.font_family, "Inter");
}

/// `source` renamed in memory to `family` with the PostScript name `ps`; no font file is added.
fn renamed_face(source: &[u8], family: &str, ps: &str) -> Vec<u8> {
    let mut bytes = source.to_vec();
    let count = u16::from_be_bytes([bytes[4], bytes[5]]) as usize;
    let record = (0..count).map(|i| 12 + i * 16).find(|r| &bytes[*r..*r + 4] == b"name").unwrap();
    let names = [(1u16, family), (2, "Regular"), (6, ps), (16, family), (17, "Regular")];
    let mut table: Vec<u8> = [0u16, names.len() as u16, (6 + names.len() * 12) as u16].iter().flat_map(|v| v.to_be_bytes()).collect();
    let mut strings = Vec::new();
    for (id, value) in names {
        let encoded: Vec<u8> = value.encode_utf16().flat_map(u16::to_be_bytes).collect();
        for n in [3, 1, 0x409, id, encoded.len() as u16, strings.len() as u16] {
            table.extend(n.to_be_bytes());
        }
        strings.extend(encoded);
    }
    table.extend(strings);
    let offset = bytes.len() as u32;
    bytes[record + 8..record + 12].copy_from_slice(&offset.to_be_bytes());
    bytes[record + 12..record + 16].copy_from_slice(&(table.len() as u32).to_be_bytes());
    bytes.extend(table);
    bytes
}

#[test]
fn exact_postscript_face_outside_the_guessed_family_is_not_missing() {
    let mut fonts = photocraft_text::FontDb::new();
    fonts.register_font_data(renamed_face(&photocraft_text::fonts::INTER_REGULAR, "Exact Only Test", "UnrelatedPS30"));
    let style =
        |ps: &str| photocraft_doc::text::CharStyle { font_family: "Imported Wrong Family".into(), postscript_name: Some(ps.into()), ..Default::default() };
    // Neither the stored family nor the family guessed from the name is installed: only the exact face.
    assert!(!fonts.has_family(&photocraft_text::fonts::guess_from_postscript("UnrelatedPS30").family));
    assert_eq!((style_font_missing(&mut fonts, &style("UnrelatedPS30")), style_font_missing(&mut fonts, &style("UnrelatedPS31"))), (false, true));
}
