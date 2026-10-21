use super::*;
use photocraft_color::SampleType;

fn session(mode: &str, depth: u64) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 12, "height": 8, "depth": depth, "mode": mode})).unwrap();
    s.edit("setup", |doc, active| {
        let b = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let ch = surf.channels();
        let mut data = Vec::new();
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                for c in 0..ch - 1 {
                    data.push(((x * 3 + y * 5 + c as i32 * 7) % 16) as f32 / 15.0);
                }
                data.push(1.0);
            }
        }
        surf.write_region(b, &data);
        Ok(())
    })
    .unwrap();
    s
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn pixels(s: &Session) -> Vec<f32> {
    let d = doc(s);
    d.layers[0].surface().unwrap().read_region(d.bounds())
}

fn tol(depth: u64) -> f32 {
    if depth == 8 { 0.5 / 255.0 + 1e-6 } else { 1e-4 }
}

#[test]
fn rgb_becomes_cmy_inks_and_back_exactly() {
    for depth in [8, 16, 32] {
        let mut s = session("rgb", depth);
        let before = pixels(&s);
        assert!(s.is_enabled("image.mode.multichannel"));
        let r = s.execute("image.mode.multichannel", json!({})).unwrap();
        assert_eq!(r["channels"], json!(["Cyan", "Magenta", "Yellow"]));
        let d = doc(&s);
        assert_eq!(d.mode, ColorMode::Multichannel);
        assert!(d.layers.is_empty());
        assert_eq!(ink_names(d), ["Cyan", "Magenta", "Yellow"]);
        // Cyan ink = the red data read as ink.
        let cyan = d.channels[0].surface.pixel(4, 3)[0];
        let red = before[(3 * 12 + 4) * 4];
        assert!((cyan - (1.0 - red)).abs() <= tol(depth), "{depth}: {cyan} vs {red}");
        assert!(!s.is_enabled("image.mode.multichannel"), "already multichannel");
        // The display prints the inks: a white RGB pixel shows paper.
        let shown = photocraft_compose::flatten(doc(&s));
        assert!(shown.px.iter().all(|p| p[3] == 1.0));
        let r = s.execute("image.mode.rgb", json!({})).unwrap();
        assert_eq!(r["converted"], "channels");
        assert_eq!(doc(&s).mode, ColorMode::Rgb);
        assert!(doc(&s).channels.is_empty());
        let after = pixels(&s);
        assert_eq!(after.len(), before.len());
        for (a, b) in after.iter().zip(&before) {
            assert!((a - b).abs() <= 2.0 * tol(depth), "{depth}: {a} vs {b}");
        }
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(doc(&s).mode, ColorMode::Multichannel);
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(doc(&s).mode, ColorMode::Rgb);
        assert_eq!(pixels(&s), before);
    }
}

#[test]
fn cmyk_lab_and_gray_channel_sets() {
    for (mode, back, names) in [
        ("cmyk", "image.mode.cmyk", vec!["Cyan", "Magenta", "Yellow", "Black"]),
        ("lab", "image.mode.lab", vec!["Alpha 1", "Alpha 2", "Alpha 3"]),
        ("grayscale", "image.mode.grayscale", vec!["Black"]),
    ] {
        for depth in [8, 16] {
            let mut s = session(mode, depth);
            let before = pixels(&s);
            let r = s.execute("image.mode.multichannel", json!({})).unwrap();
            assert_eq!(r["channels"], json!(names), "{mode}");
            assert_eq!(doc(&s).depth, if depth == 8 { SampleType::U8 } else { SampleType::U16 });
            s.execute(back, json!({})).unwrap();
            for (a, b) in pixels(&s).iter().zip(&before) {
                assert!((a - b).abs() <= 2.0 * tol(depth), "{mode}/{depth}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn flattens_layers_and_keeps_alpha_channels() {
    let mut s = session("rgb", 8);
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 4, "height": 4})).unwrap();
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    s.execute("select.saveSelection", json!({})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    let alpha = doc(&s).channels.len();
    assert_eq!(alpha, 1);
    s.execute("image.mode.multichannel", json!({})).unwrap();
    let d = doc(&s);
    assert_eq!(d.channels.len(), 4);
    assert!(d.channels[3].spot.is_none(), "the saved selection stays an alpha channel");
    // The black fill on the upper layer is solid ink in all three plates.
    for k in 0..3 {
        assert!(d.channels[k].surface.pixel(1, 1)[0] > 0.99);
    }
    // Back to RGB: the alpha channel survives, the inks become colour channels.
    s.execute("image.mode.rgb", json!({})).unwrap();
    assert_eq!(doc(&s).channels.len(), 1);
    assert!(doc(&s).layers[0].surface().unwrap().pixel(1, 1)[0] < 0.01);
}

#[test]
fn mismatched_channel_count_converts_the_printed_look() {
    let mut s = session("rgb", 8);
    s.execute("image.mode.multichannel", json!({})).unwrap();
    // Drop the Yellow plate: two inks can't be RGB channels.
    s.edit("drop", |doc, _| {
        doc.channels.truncate(2);
        Ok(())
    })
    .unwrap();
    let shown = photocraft_compose::flatten(doc(&s));
    let r = s.execute("image.mode.rgb", json!({})).unwrap();
    assert_eq!(r["converted"], "appearance");
    let px = doc(&s).layers[0].surface().unwrap().pixel(5, 2);
    let want = shown.get(5, 2);
    for c in 0..3 {
        assert!((px[c] - want[c]).abs() < 1.0 / 255.0 + 1e-4);
    }
}

#[test]
fn duotone_inks_become_channels() {
    let mut s = session("grayscale", 8);
    s.execute("image.mode.duotone", json!({"type": "duotone"})).unwrap();
    let inks = doc(&s).duotone.as_ref().unwrap().inks.len();
    let r = s.execute("image.mode.multichannel", json!({})).unwrap();
    assert_eq!(r["channels"].as_array().unwrap().len(), inks);
}

#[test]
fn psd_round_trip_keeps_names_inks_and_values() {
    for depth in [8, 16] {
        let mut s = session("cmyk", depth);
        s.execute("image.mode.multichannel", json!({})).unwrap();
        let d = doc(&s).clone();
        let out = photocraft_io::export(&d, "x.psd", &photocraft_io::ExportOptions::default()).unwrap();
        let file = photocraft_io::document_to_psd(&d);
        assert_eq!(file.header.color_mode.as_u16(), 7);
        assert_eq!(file.header.channels, 4);
        // Photoshop stores ink as dark: a solid-ink sample is 0.
        let back = photocraft_io::import("x.psd", &out.bytes).unwrap().document;
        assert_eq!(back.mode, ColorMode::Multichannel);
        assert_eq!(back.depth, d.depth);
        assert!(back.layers.is_empty());
        assert_eq!(ink_names(&back), ink_names(&d));
        for (a, b) in back.channels.iter().zip(&d.channels) {
            assert_eq!(a.spot.map(|(c, _)| c.to_rgba8()), b.spot.map(|(c, _)| c.to_rgba8()));
            let (va, vb) = (a.surface.read_region(d.bounds()), b.surface.read_region(d.bounds()));
            for (x, y) in va.iter().zip(&vb) {
                assert!((x - y).abs() <= tol(depth), "{depth}");
            }
        }
    }
}

#[test]
fn psd_stores_ink_as_dark() {
    let mut s = session("grayscale", 8);
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    s.execute("image.mode.multichannel", json!({})).unwrap();
    let file = photocraft_io::document_to_psd(doc(&s));
    let merged = file.decode_merged().unwrap();
    assert!(merged.iter().all(|&b| b == 0), "solid black ink is stored as 0");
}

#[test]
fn pcraft_round_trip() {
    let mut s = session("rgb", 16);
    s.execute("image.mode.multichannel", json!({})).unwrap();
    let d = doc(&s).clone();
    let bytes = photocraft_format::save_to_bytes(&d, &Default::default()).unwrap();
    let back = photocraft_format::load_from_bytes(&bytes).unwrap();
    assert_eq!(back.mode, ColorMode::Multichannel);
    assert_eq!(back.channels, d.channels);
    assert_eq!(photocraft_compose::flatten(&back), photocraft_compose::flatten(&d));
}

#[test]
fn flat_export_prints_the_inks() {
    let mut s = session("grayscale", 8);
    s.execute("edit.fill", json!({"color": "#000000"})).unwrap();
    s.execute("image.mode.multichannel", json!({})).unwrap();
    let out = photocraft_io::export(doc(&s), "x.png", &photocraft_io::ExportOptions::default()).unwrap();
    let back = photocraft_io::import("x.png", &out.bytes).unwrap().document;
    let px = photocraft_compose::flatten(&back).get(3, 3);
    // Solid process black ink, not paper.
    assert!(px[0] < 0.3 && px[3] > 0.99, "{px:?}");
}
