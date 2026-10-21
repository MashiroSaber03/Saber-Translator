use super::*;

#[test]
fn color_replacement_preserves_pixel_membership_and_transparency() {
    let px = vec![[1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.6, 0.0, 0.4, 1.0], [0.0, 0.0, 0.0, 0.0]];
    for format in ["gif", "png8"] {
        for dither in ["none", "diffusion", "pattern", "noise"] {
            let p = json!({"format": format, "colorTable": ["#ff0000", "#0000ff"], "dither": dither, "matte": "none"});
            let original = WebSettings::from_params(&p, "test").unwrap();
            let before = optimize(&px, 4, Rect::new(0, 0, 4, 1), &original, None, None, 72.0, true).unwrap();
            let mut edited = p.clone();
            // Replace red with blue: re-quantization would now send all opaque pixels to blue.
            edited["colorTableEdits"] = json!({"0": "#0000ff", "1": "#00ff00"});
            let settings = WebSettings::from_params(&edited, "test").unwrap();
            assert_eq!(WebSettings::from_params(&settings.to_params(), "test").unwrap(), settings);
            let after = optimize(&px, 4, Rect::new(0, 0, 4, 1), &settings, None, None, 72.0, true).unwrap();
            for (old, new) in before.preview.as_chunks::<4>().0.iter().zip(after.preview.as_chunks::<4>().0) {
                if old[3] == 0 {
                    assert_eq!(new[3], 0);
                } else if old[..3] == [255, 0, 0] {
                    assert_eq!(*new, [0, 0, 255, 255]);
                } else {
                    assert_eq!(*new, [0, 255, 0, 255]);
                }
            }
            assert_eq!(before.indices, after.indices, "palette edits must not remap pixels");
            assert_eq!(after.transparent_index, Some(2));
            assert_eq!(after.color_table, vec![[255, 0, 0], [0, 0, 255]]);
            let decoded = photocraft_codecs::decode(&after.bytes).unwrap().convert(photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8);
            // Transparent RGB has no significance; opaque preview pixels must equal the file.
            for (file, preview) in decoded.data().as_chunks::<4>().0.iter().zip(after.preview.as_chunks::<4>().0) {
                assert_eq!(file[3], preview[3]);
                if preview[3] != 0 {
                    assert_eq!(file, preview);
                }
            }
        }
    }
}

#[test]
fn exported_palette_edits_work_at_every_document_depth() {
    let dir = tmp("table-edits");
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let mut p = json!({"format": "gif", "colors": 16, "dither": "none"});
        let estimate = s.execute("file.export.saveForWebLegacy", p.clone()).unwrap();
        let table = estimate["colorTable"].as_array().unwrap();
        let red = table.iter().position(|c| c == "#ff0000").unwrap();
        p["colorTable"] = estimate["colorTable"].clone();
        p["colorTableEdits"] = json!({red.to_string(): "#00ff00"});
        let path = format!("{dir}/{depth}.gif");
        p["path"] = json!(path);
        let original_doc = s.active().unwrap().doc.clone();
        s.execute("file.export.saveForWebLegacy", p).unwrap();
        assert_eq!(s.active().unwrap().doc, original_doc, "export must not edit the document");
        let img = decode(&path).convert(photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8);
        assert_eq!(&img.data()[(16 * 64 + 16) * 4..(16 * 64 + 16) * 4 + 4], &[0, 255, 0, 255]);
        assert_eq!(img.data()[3], 0);
    }
}

#[test]
fn malformed_tables_and_out_of_bounds_slices_return_errors() {
    for p in [
        json!({"colorTable": []}),
        json!({"colorTable": "bad"}),
        json!({"colorTable": ["#ff00é"]}),
        json!({"colorTable": vec!["#000000"; 257]}),
        json!({"colorTable": [123]}),
        json!({"colorTableEdits": {"0": "#ffffff"}}),
        json!({"colorTable": ["#ff0000"], "colorTableEdits": {"-1": "#ffffff"}}),
        json!({"colorTable": ["#ff0000"], "colorTableEdits": {"1": "#ffffff"}}),
        json!({"colorTable": ["#ff0000"], "colorTableEdits": []}),
    ] {
        assert!(WebSettings::from_params(&p, "test").is_err(), "{p}");
        let mut s = session(8);
        let doc = s.active().unwrap().doc.clone();
        assert!(s.execute("file.export.saveForWebLegacy", p).is_err());
        assert_eq!(s.active().unwrap().doc, doc);
    }
    let settings = WebSettings::default();
    for (bw, rect) in [(0, Rect::new(0, 0, 1, 1)), (1, Rect::new(-1, 0, 1, 1)), (1, Rect::new(0, 0, 2, 1)), (1, Rect::new(0, 0, 1, i32::MAX))] {
        assert!(optimize(&[[0.0; 4]], bw, rect, &settings, None, None, 72.0, false).is_err());
    }
    let settings = WebSettings::from_params(&json!({"format": "gif", "colorTable": vec!["#000000"; 256]}), "test").unwrap();
    assert!(optimize(&[[0.0; 4]], 1, Rect::new(0, 0, 1, 1), &settings, None, None, 72.0, true).is_err());
}
