use super::*;

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("photocraft-file-cmds-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().into_owned()
}

fn session(w: u32, h: u32, depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
    s
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

/// Writes a solid-colour PNG of `w × h` and returns its path.
fn png(dir: &str, name: &str, w: u32, h: u32, color: &str) -> String {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": w, "height": h, "background": color})).unwrap();
    let path = join(dir, name);
    save_doc(doc(&s), &path, None).unwrap();
    path
}

fn composite(s: &Session, x: i32, y: i32) -> [f32; 4] {
    let b = photocraft_compose::render(doc(s), Rect::new(x, y, x + 1, y + 1));
    b.px[0]
}

#[test]
fn close_all_and_close_others() {
    let mut s = session(8, 8, 8);
    s.execute("file.new", json!({"width": 9, "height": 9})).unwrap();
    s.execute("file.new", json!({"width": 10, "height": 10})).unwrap();
    s.set_active(1);
    assert_eq!(s.execute("file.closeOthers", json!({})).unwrap()["closed"], 2);
    assert_eq!(s.documents().len(), 1);
    assert_eq!(doc(&s).size.width, 9);
    assert_eq!(s.active_index(), Some(0));
    assert!(s.execute("file.closeOthers", json!({"document": 5})).is_err());
    s.execute("file.new", json!({})).unwrap();
    assert_eq!(s.execute("file.closeAll", json!({})).unwrap()["closed"], 2);
    assert!(s.documents().is_empty());
    assert!(!s.is_enabled("file.closeAll"));
}

#[test]
fn camera_raw_opens_as_16_bit_with_notes() {
    use photocraft_raw::testgen::{DngSpec, mosaic, scene};
    let (w, h) = (24, 16);
    let mut spec = DngSpec::cfa(w, h, mosaic(&scene(w, h), w, [0, 1, 1, 2], 0, 65535));
    spec.as_shot_neutral = Some([0.5, 1.0, 0.7]);
    let mut s = Session::new();
    let r = open_bytes_as(&mut s, "IMG_0001.dng", &spec.build(), None, None).unwrap();
    assert!(r["warnings"].as_array().is_some_and(|w| w.iter().any(|m| m.as_str().is_some_and(|m| m.contains("DNG")))), "{r}");
    assert_eq!((doc(&s).size.width, doc(&s).size.height), (24, 16));
    assert_eq!(doc(&s).depth, photocraft_color::SampleType::U16);
    // Damaged raw data is an error, not a crash.
    let mut bad = spec.build();
    bad.truncate(bad.len() / 2);
    assert!(open_bytes_as(&mut s, "bad.dng", &bad, None, None).is_err());
}

#[test]
fn revert_reloads_as_one_undoable_step() {
    let dir = tmp("revert");
    let path = png(&dir, "a.png", 16, 12, "#ff0000");
    let mut s = Session::new();
    assert!(!s.is_enabled("file.revert"));
    let bytes = std::fs::read(&path).unwrap();
    open_bytes_as(&mut s, &path, &bytes, None, Some(path.clone())).unwrap();
    let id = doc(&s).id;
    assert!(s.is_enabled("file.revert"));
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("edit.fill", json!({"color": "#0000ff"})).unwrap();
    assert!(s.active().unwrap().is_dirty());
    s.execute("file.revert", json!({})).unwrap();
    assert_eq!(doc(&s).layers.len(), 1);
    assert_eq!(doc(&s).id, id, "identity kept");
    assert!(!s.active().unwrap().is_dirty());
    assert!(composite(&s, 3, 3)[0] > 0.99);
    // Revert is a history state: undo brings the edits back.
    assert!(s.undo());
    assert_eq!(doc(&s).layers.len(), 2);
    // An unsaved document cannot revert.
    s.execute("file.new", json!({})).unwrap();
    assert!(s.execute("file.revert", json!({})).is_err());
}

#[test]
fn injected_import_failure_does_not_open_a_document() {
    let mut s = session(8, 8, 8);
    let before = s.documents().len();
    crate::allocation::fail_next_for_test();
    let result = open_bytes_as(&mut s, "image.png", b"not decoded", None, None);
    assert!(result.as_ref().is_err_and(|e| e.to_string().contains("not enough memory for importing document")));
    assert_eq!(s.documents().len(), before);
}

#[test]
fn injected_save_failure_does_not_modify_the_destination() {
    let dir = tmp("allocation-save");
    let path = join(&dir, "existing.psd");
    std::fs::write(&path, b"previous file").unwrap();
    let s = session(8, 8, 8);
    crate::allocation::fail_next_for_test();
    let result = save_doc(doc(&s), &path, None);
    assert!(result.as_ref().is_err_and(|e| e.to_string().contains("not enough memory for saving document")));
    assert_eq!(std::fs::read(path).unwrap(), b"previous file");
}

#[test]
fn save_a_copy_keeps_path_and_dirty_state() {
    let dir = tmp("copy");
    let mut s = session(20, 10, 16);
    s.execute("layer.new.layer", json!({})).unwrap();
    let rev = s.active().unwrap().revision;
    let out = join(&dir, "copy.psd");
    s.execute("file.saveACopy", json!({"path": out})).unwrap();
    assert!(std::fs::metadata(&out).unwrap().len() > 0);
    assert_eq!(s.active().unwrap().path, None);
    assert_eq!(s.active().unwrap().revision, rev);
    assert!(s.active().unwrap().is_dirty());
    let back = photocraft_io::import("copy.psd", &std::fs::read(&out).unwrap()).unwrap().document;
    assert_eq!(back.layers.len(), 2);
    assert_eq!(back.depth, photocraft_color::SampleType::U16);
    // Flattened copy.
    let flat = join(&dir, "flat.psd");
    s.execute("file.saveACopy", json!({"path": flat, "layers": false})).unwrap();
    assert_eq!(photocraft_io::import("flat.psd", &std::fs::read(&flat).unwrap()).unwrap().document.layers.len(), 1);
    assert!(s.execute("file.saveACopy", json!({})).is_err());
}

/// `tgaBits` picks the Targa's bits per pixel (byte 16 of the header); unset, a document with an
/// alpha channel is written as 32-bit, with the channel as the alpha.
#[test]
fn save_a_copy_writes_the_targa_bits_asked_for() {
    let dir = tmp("tga-bits");
    let mut s = session(20, 10, 8);
    s.execute("select.all", json!({})).unwrap();
    s.execute("select.saveSelection", json!({})).unwrap();
    for (params, bpp) in [(json!({}), 32), (json!({"tgaBits": 24}), 24), (json!({"tgaBits": 32}), 32), (json!({"tgaBits": null}), 32)] {
        let out = join(&dir, "a.tga");
        let mut p = params.clone();
        p["path"] = json!(out);
        s.execute("file.saveACopy", p).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(bytes[16], bpp, "{params}");
        let back = photocraft_io::import("a.tga", &bytes).unwrap().document;
        assert_eq!(back.channels.len(), usize::from(bpp == 32), "{params}: the selection reopens as Alpha 1");
    }
}

/// `dpxBits` picks a DPX's bits per sample (byte 803 of the header); unset, it follows the
/// document's depth. Anything but 8, 10, 12 or 16 is refused before anything is written.
#[test]
fn save_a_copy_writes_the_dpx_bits_asked_for() {
    let dir = tmp("dpx-bits");
    for (depth, params, bits) in
        [(8, json!({}), 8), (16, json!({}), 16), (16, json!({"dpxBits": 10}), 10), (8, json!({"dpxBits": 12}), 12), (16, json!({"dpxBits": null}), 16)]
    {
        let mut s = session(12, 6, depth);
        let out = join(&dir, "a.dpx");
        let mut p = params.clone();
        p["path"] = json!(out);
        s.execute("file.saveACopy", p).unwrap();
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!((&bytes[..4], bytes[803]), (&b"SDPX"[..], bits), "{depth}-bit {params}");
        assert_eq!(photocraft_io::import("a.dpx", &bytes).unwrap().document.size, photocraft_geom::Size::new(12, 6));
    }
    let mut s = session(8, 8, 8);
    for bad in [json!(24), json!(0), json!("10"), json!(10.5)] {
        let out = join(&dir, "bad.dpx");
        let r = s.execute("file.saveACopy", json!({"path": out, "dpxBits": bad}));
        assert!(matches!(r, Err(EngineError::BadParams { .. })), "{bad}: {r:?}");
        assert!(!std::path::Path::new(&out).exists(), "{bad}: nothing written");
    }
}

/// A `tgaBits` other than 24 or 32 is a bad parameter, reported before anything is written.
#[test]
fn bad_targa_bits_are_refused_before_writing() {
    let dir = tmp("tga-bad");
    let mut s = session(8, 8, 8);
    for bad in [json!(16), json!(0), json!(-32), json!(32.5), json!("32"), json!(true), json!([32])] {
        let out = join(&dir, "a.tga");
        let r = s.execute("file.saveACopy", json!({"path": out, "tgaBits": bad}));
        assert!(matches!(r, Err(EngineError::BadParams { .. })), "{bad}: {r:?}");
        assert!(!std::path::Path::new(&out).exists(), "{bad}: nothing written");
        let r = s.execute("file.export.layersToFiles", json!({"dir": dir, "format": "tga", "tgaBits": bad}));
        assert!(matches!(r, Err(EngineError::BadParams { .. })), "{bad}: {r:?}");
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "{bad}: no layer written");
    }
}

/// Preferences › File Handling › Image Previews decides whether a PSD gets Photoshop's
/// thumbnail resource (1036): Always and Ask When Saving write it, Never doesn't.
#[test]
fn image_previews_preference_controls_the_psd_thumbnail() {
    use crate::prefs::Ask;
    let dir = tmp("previews");
    let mut s = session(300, 200, 16);
    let has_thumbnail = |path: &str| {
        let file = photocraft_psd::PsdFile::from_bytes(&std::fs::read(path).unwrap()).unwrap();
        file.resources.iter().any(|r| r.id == photocraft_psd::resources::ids::THUMBNAIL)
    };
    for (ask, expect) in [(Ask::Always, true), (Ask::Never, false), (Ask::Ask, true)] {
        s.edit_prefs(|p| p.file_handling.image_previews = ask);
        let out = join(&dir, &format!("{ask:?}.psd"));
        s.execute("file.saveACopy", json!({"path": out})).unwrap();
        assert_eq!(has_thumbnail(&out), expect, "{ask:?}");
        let png = join(&dir, &format!("{ask:?}.png"));
        s.execute("file.saveACopy", json!({"path": png})).unwrap();
    }
}

/// Preferences › File Handling › Maximize PSD and PSB File Compatibility decides whether a
/// layered PSD keeps its real merged image: Always and Ask When Saving write it, Never writes
/// Photoshop's placeholder and says so in resource 1057.
#[test]
fn maximize_compatibility_preference_controls_the_psd_merged_image() {
    use crate::prefs::Ask;
    let dir = tmp("maximize");
    let mut s = session(300, 200, 16);
    s.execute("layer.new.layer", json!({})).unwrap();
    let real_merged = |path: &str| photocraft_psd::PsdFile::from_bytes(&std::fs::read(path).unwrap()).unwrap().has_real_merged_data();
    for (ask, expect) in [(Ask::Always, true), (Ask::Never, false), (Ask::Ask, true)] {
        s.edit_prefs(|p| p.file_handling.maximize_psd_compatibility = ask);
        for ext in ["psd", "psb"] {
            let out = join(&dir, &format!("{ask:?}.{ext}"));
            s.execute("file.saveACopy", json!({"path": out})).unwrap();
            assert_eq!(real_merged(&out), Some(expect), "{ask:?} {ext}");
        }
    }
}

#[test]
fn open_as_forces_the_decoder() {
    let dir = tmp("openas");
    let path = png(&dir, "a.png", 6, 5, "#00ff00");
    let renamed = join(&dir, "mystery.bin");
    std::fs::copy(&path, &renamed).unwrap();
    let mut s = Session::new();
    s.execute("file.openAs", json!({"path": renamed, "as": "png"})).unwrap();
    assert_eq!(doc(&s).size.width, 6);
    assert_eq!(doc(&s).name, "mystery.bin");
    assert_eq!(s.active().unwrap().path.as_deref(), Some(renamed.as_str()));
    assert!(s.execute("file.openAs", json!({"path": join(&dir, "missing.png")})).is_err());
}

#[test]
fn place_embedded_centres_fits_and_embeds() {
    let dir = tmp("place");
    let big = png(&dir, "big.png", 200, 100, "#ff0000");
    let small = png(&dir, "small.png", 10, 10, "#0000ff");
    for depth in [8, 16, 32] {
        let mut s = session(100, 100, depth);
        let r = s.execute("file.placeEmbedded", json!({"path": big})).unwrap();
        assert!((r["scale"].as_f64().unwrap() - 50.0).abs() < 1e-9);
        let d = s.active().unwrap();
        let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
        assert_eq!(l.name, "big");
        let LayerContent::Smart(so) = &l.content else { panic!("smart object expected") };
        assert!(matches!(&so.source, SmartSource::Embedded { file_name, bytes } if file_name == "big.png" && !bytes.is_empty()));
        assert_eq!(so.transform.m[0], 0.5);
        // Fitted to the canvas width, centred vertically: rows 25..75 red.
        assert!(composite(&s, 50, 50)[0] > 0.99 && composite(&s, 50, 50)[2] < 0.01);
        assert!(composite(&s, 50, 10)[1] > 0.99, "white background above");
        // Small images keep their size, centred.
        s.execute("file.placeLinked", json!({"path": small})).unwrap();
        assert!(composite(&s, 50, 50)[2] > 0.99);
        assert!(composite(&s, 40, 40)[2] < 0.01);
        let d = s.active().unwrap();
        let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
        assert!(matches!(&l.content, LayerContent::Smart(so) if matches!(&so.source, SmartSource::Linked { path } if path == &small)));
        // One step each.
        assert!(s.undo() && s.undo());
        assert_eq!(doc(&s).layers.len(), 1);
    }
}

/// Preferences ▸ General ▸ Always Create Smart Objects When Placing: unchecked, an embedded place
/// lands as a plain pixel layer; a linked place is always a smart object and `"smartObject"` wins.
#[test]
fn always_create_smart_objects_preference_places_a_pixel_layer() {
    let dir = tmp("place-smart-pref");
    let big = png(&dir, "big.png", 200, 100, "#ff0000");
    let mut s = session(100, 100, 8);
    s.execute("prefs.set", json!({"path": "general.alwaysCreateSmartObjectsWhenPlacing", "value": false})).unwrap();
    let r = s.execute("file.placeEmbedded", json!({"path": big})).unwrap();
    assert_eq!(r["smartObject"], json!(false));
    let d = s.active().unwrap();
    let l = d.doc.layer(d.active_layer.unwrap()).unwrap();
    assert!(matches!(l.content, LayerContent::Raster(_)), "a pixel layer");
    // Fitted to the canvas width, centred vertically, exactly like the smart-object place.
    assert!(composite(&s, 50, 50)[0] > 0.99 && composite(&s, 50, 50)[2] < 0.01);
    assert!(composite(&s, 50, 10)[1] > 0.99, "white background above");
    // An explicit "smartObject": true overrides, and Place Linked with the preference off still
    // creates a linked smart object.
    assert_eq!(s.execute("file.placeEmbedded", json!({"path": big, "smartObject": true})).unwrap()["smartObject"], json!(true));
    assert_eq!(s.execute("file.placeLinked", json!({"path": big})).unwrap()["smartObject"], json!(true));
}

/// Preferences ▸ General ▸ Resize Image During Place: off, a larger image keeps its natural size
/// instead of being fitted to the canvas; an explicit `"fit"` still wins.
#[test]
fn resize_image_during_place_preference_controls_the_fit() {
    let dir = tmp("place-fit-pref");
    let big = png(&dir, "big.png", 200, 100, "#ff0000");
    let mut s = session(100, 100, 8);
    s.execute("prefs.set", json!({"path": "general.resizeImageDuringPlace", "value": false})).unwrap();
    let r = s.execute("file.placeEmbedded", json!({"path": big})).unwrap();
    assert!((r["scale"].as_f64().unwrap() - 100.0).abs() < 1e-9, "natural size: {r}");
    let b = r["bounds"].as_array().unwrap();
    assert!(b[2].as_f64().unwrap() > 100.0, "wider than the canvas: {b:?}");
    // An explicit "fit": true overrides the preference.
    let r = s.execute("file.placeEmbedded", json!({"path": big, "fit": true})).unwrap();
    assert!((r["scale"].as_f64().unwrap() - 50.0).abs() < 1e-9, "fitted on request: {r}");
}

/// A 40×20 JPEG (left half red, right half blue) tagged EXIF Orientation = 6: shown upright it
/// is 20×40, red on top.
fn rotated_jpeg(dir: &str) -> String {
    let px: Vec<u8> = (0..20).flat_map(|_| (0..40).flat_map(|x| if x < 20 { [230, 20, 20] } else { [20, 20, 230] })).collect();
    let img = photocraft_codecs::Image::from_u8(40, 20, photocraft_codecs::ChannelLayout::Rgb, px).unwrap();
    let jpeg = photocraft_codecs::encode(&img, photocraft_codecs::Format::Jpeg, &Default::default()).unwrap();
    let mut seg = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0\x06\0\0\0\0\0\0\0".to_vec();
    let mut file = vec![0xFF, 0xD8, 0xFF, 0xE1];
    file.extend_from_slice(&((seg.len() + 2) as u16).to_be_bytes());
    file.append(&mut seg);
    file.extend_from_slice(&jpeg[2..]);
    let path = join(dir, "IMG_0001.jpg");
    std::fs::write(&path, file).unwrap();
    path
}

#[test]
fn rotated_jpegs_open_and_place_upright() {
    let dir = tmp("orientation");
    let path = rotated_jpeg(&dir);
    let mut s = Session::new();
    s.execute("file.openAs", json!({"path": path, "as": "jpg"})).unwrap();
    assert_eq!((doc(&s).size.width, doc(&s).size.height), (20, 40));
    assert!(composite(&s, 10, 5)[0] > 0.8 && composite(&s, 10, 35)[2] > 0.8, "red on top, blue below");
    // Placed as a smart object: upright too, and it renders upright from its embedded bytes.
    let mut s = session(100, 100, 8);
    let r = s.execute("file.placeEmbedded", json!({"path": path})).unwrap();
    assert_eq!(r["bounds"], json!([40.0, 30.0, 60.0, 70.0]));
    assert!(composite(&s, 50, 35)[0] > 0.8 && composite(&s, 50, 65)[2] > 0.8);
}

#[test]
fn file_info_round_trips_through_xmp_and_psd() {
    let mut s = session(8, 8, 8);
    let empty = s.execute("file.fileInfo", json!({})).unwrap();
    assert_eq!(empty["title"], "");
    assert!(!s.undo(), "reading is not a history step");
    s.execute("file.fileInfo", json!({"title": "Starry <Night>", "author": "V. van Gogh", "keywords": ["stars", "night"], "copyright": "Public domain", "copyrightStatus": "publicDomain", "description": "Oil & canvas"})).unwrap();
    let info = s.execute("file.fileInfo", json!({})).unwrap();
    assert_eq!(info["title"], "Starry <Night>");
    assert_eq!(info["author"], "V. van Gogh");
    assert_eq!(info["keywords"], json!(["stars", "night"]));
    assert_eq!(info["description"], "Oil & canvas");
    assert_eq!(info["copyrightStatus"], "publicDomain");
    // Editing one field keeps the others and replaces, not duplicates.
    s.execute("file.fileInfo", json!({"title": "The Starry Night"})).unwrap();
    let xmp = doc(&s).metadata.xmp.clone().unwrap();
    assert_eq!(xmp.matches("<dc:title>").count(), 1);
    assert_eq!(read_file_info(Some(&xmp))["author"], "V. van Gogh");
    // PSD round trip.
    let (bytes, _) = encode(doc(&s), "x.psd", None).unwrap();
    let back = photocraft_io::import("x.psd", &bytes).unwrap().document;
    let info = read_file_info(back.metadata.xmp.as_deref());
    assert_eq!(info["title"], "The Starry Night");
    assert_eq!(info["keywords"], json!(["stars", "night"]));
    // Foreign XMP (attribute form, other namespaces) keeps its other properties.
    let foreign = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:CreatorTool="Other App" photoshop:AuthorsPosition="Painter"/></rdf:RDF></x:xmpmeta>"#;
    assert_eq!(read_file_info(Some(foreign))["authorTitle"], "Painter");
    let out = write_file_info(Some(foreign), &json!({"authorTitle": "Artist"}));
    assert!(out.contains("xmp:CreatorTool=\"Other App\""));
    assert_eq!(read_file_info(Some(&out))["authorTitle"], "Artist");
}

fn camera_exif() -> Vec<u8> {
    photocraft_algo::exif::build(&photocraft_algo::exif::CameraInfo {
        make: Some("SONY".into()),
        model: Some("ILCE-7RM5".into()),
        lens: Some("24-70mm F2.8".into()),
        exposure_time: Some(1.0 / 250.0),
        f_number: Some(8.0),
        iso: Some(400.0),
        focal_length: Some(33.1),
        focal_length_35mm: Some(50.0),
        date_taken: Some("2024:02:17 12:40:19".into()),
    })
}

#[test]
fn file_info_reads_camera_data_from_jpeg_and_psd_exif() {
    let mut s = session(8, 8, 8);
    assert_eq!(s.execute("file.fileInfo", json!({})).unwrap()["camera"], json!({}));
    s.edit("exif", |doc, _| {
        doc.metadata.exif = Some(Arc::new(camera_exif()));
        Ok(())
    })
    .unwrap();
    let want = json!({
        "make": "SONY", "model": "ILCE-7RM5", "dateTaken": "2024-02-17 12:40:19", "exposure": "1/250 s", "fNumber": "f/8",
        "iso": "ISO 400", "focalLength": "33.1 mm", "focalLength35mm": "50 mm", "lens": "24-70mm F2.8"
    });
    for name in ["x.jpg", "x.psd"] {
        let (bytes, _) = encode(doc(&s), name, None).unwrap();
        let back = photocraft_io::import(name, &bytes).unwrap().document;
        let mut t = Session::new();
        t.add_document(back, None);
        assert_eq!(t.execute("file.fileInfo", json!({})).unwrap()["camera"], want, "{name}");
        // Editing the description fields leaves the read-only camera data alone.
        let edited = t.execute("file.fileInfo", json!({"title": "Pier", "camera": {"make": "ignored"}})).unwrap();
        assert_eq!(edited["camera"], want, "{name}");
    }
}

#[test]
fn camera_data_formats_like_photoshop_and_falls_back_to_xmp() {
    let info = |t: Option<f64>, n: Option<f64>, f: Option<f64>| {
        let exif = photocraft_algo::exif::build(&photocraft_algo::exif::CameraInfo { exposure_time: t, f_number: n, focal_length: f, ..Default::default() });
        camera_data(Some(&exif), None)
    };
    assert_eq!(info(Some(2.0), Some(2.8), Some(35.0)), json!({"exposure": "2 s", "fNumber": "f/2.8", "focalLength": "35 mm"}));
    assert_eq!(info(Some(0.5), Some(5.6), None), json!({"exposure": "1/2 s", "fNumber": "f/5.6"}));
    assert_eq!(info(Some(0.4), None, None), json!({"exposure": "0.4 s"}));
    assert_eq!(info(Some(1.0 / 2000.0), None, None), json!({"exposure": "1/2000 s"}));
    assert_eq!(info(Some(0.0), Some(0.0), Some(0.0)), json!({}));
    // No EXIF block (a raw import, XMP-only files): Photoshop's exif:/tiff:/aux: properties.
    let xmp = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:tiff="http://ns.adobe.com/tiff/1.0/" xmlns:exif="http://ns.adobe.com/exif/1.0/" xmlns:aux="http://ns.adobe.com/exif/1.0/aux/"
 tiff:Make="SONY" tiff:Model="ILCE-7RM5" exif:ExposureTime="1/250" exif:FNumber="8/1" exif:FocalLength="331/10"
 exif:FocalLengthIn35mmFilm="50" exif:DateTimeOriginal="2024-02-17T12:40:19.00+01:00" aux:Lens="Lens &amp; Co">
<exif:ISOSpeedRatings><rdf:Seq><rdf:li>400</rdf:li></rdf:Seq></exif:ISOSpeedRatings>
</rdf:Description></rdf:RDF></x:xmpmeta>"#;
    let want = json!({
        "make": "SONY", "model": "ILCE-7RM5", "dateTaken": "2024-02-17 12:40:19", "exposure": "1/250 s", "fNumber": "f/8",
        "iso": "ISO 400", "focalLength": "33.1 mm", "focalLength35mm": "50 mm", "lens": "Lens & Co"
    });
    assert_eq!(camera_data(None, Some(xmp)), want);
    // EXIF wins; XMP only fills what EXIF lacks.
    let exif = photocraft_algo::exif::build(&photocraft_algo::exif::CameraInfo { make: Some("Other".into()), ..Default::default() });
    let both = camera_data(Some(&exif), Some(xmp));
    assert_eq!(both["make"], "Other");
    assert_eq!(both["exposure"], "1/250 s");
    // Garbage in either block reads as no data, never a panic.
    assert_eq!(camera_data(Some(b"II*\0\xff\xff\xff\xff"), Some("<exif:FNumber>0/0</exif:FNumber> exif:ExposureTime=\"-1/0\"")), json!({}));
}

const MULTILINGUAL_XMP: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="" xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmlns:xmpRights="http://ns.adobe.com/xap/1.0/rights/"
 xmp:CreatorTool="Other App" xmpRights:Marked="True" xmpRights:WebStatement="https://example.com/license">
<dc:title><rdf:Alt><rdf:li xml:lang="x-default">Sun</rdf:li><rdf:li xml:lang="fr">Soleil</rdf:li><rdf:li xml:lang="ar">الشمس</rdf:li></rdf:Alt></dc:title>
<dc:description><rdf:Alt><rdf:li xml:lang="x-default">Sunrise</rdf:li><rdf:li xml:lang="fr">Lever du soleil</rdf:li></rdf:Alt></dc:description>
<dc:rights><rdf:Alt><rdf:li xml:lang="x-default">All rights reserved</rdf:li><rdf:li xml:lang="fr">Tous droits réservés</rdf:li></rdf:Alt></dc:rights>
<dc:subject><rdf:Bag><rdf:li>sky</rdf:li></rdf:Bag></dc:subject>
</rdf:Description></rdf:RDF></x:xmpmeta>"#;

fn assert_translations_kept(xmp: &str) {
    for prop in ["dc:title", "dc:description", "dc:rights"] {
        assert_eq!(find_element(xmp, prop).unwrap().2, find_element(MULTILINGUAL_XMP, prop).unwrap().2, "{prop}");
        assert_eq!(xmp.matches(&format!("<{prop}>")).count(), 1);
    }
    assert!(xmp.contains("xmp:CreatorTool=\"Other App\""));
    assert!(xmp.contains("xmpRights:Marked=\"True\""));
    assert!(xmp.contains("xmpRights:WebStatement=\"https://example.com/license\""));
}

#[test]
fn file_info_keeps_translations_when_editing_keywords() {
    let partial = json!({"keywords": ["sky", "sunrise"]});
    let mut dialog = read_file_info(Some(MULTILINGUAL_XMP));
    dialog["keywords"] = json!("sky; sunrise");
    for params in [partial, dialog] {
        let out = write_file_info(Some(MULTILINGUAL_XMP), &params);
        assert_translations_kept(&out);
        assert_eq!(read_file_info(Some(&out))["keywords"], json!(["sky", "sunrise"]));
    }
    let unchanged = read_file_info(Some(MULTILINGUAL_XMP));
    assert_eq!(write_file_info(Some(MULTILINGUAL_XMP), &unchanged), MULTILINGUAL_XMP);
    assert_eq!(write_file_info(Some(MULTILINGUAL_XMP), &json!({})), MULTILINGUAL_XMP);
}

#[test]
fn file_info_clears_only_the_requested_field() {
    let out = write_file_info(Some(MULTILINGUAL_XMP), &json!({"keywords": []}));
    assert_translations_kept(&out);
    assert!(find_element(&out, "dc:subject").is_none());
    assert_eq!(read_file_info(Some(&out))["keywords"], json!([]));
}

#[test]
fn file_info_translations_survive_undo_and_psd_round_trip() {
    for depth in [8, 16, 32] {
        let mut s = session(8, 8, depth);
        s.edit("Set metadata", |doc, _| {
            doc.metadata.xmp = Some(MULTILINGUAL_XMP.to_string());
            Ok(())
        })
        .unwrap();
        s.execute("file.fileInfo", json!({"keywords": ["sunrise"]})).unwrap();
        let edited = doc(&s).metadata.xmp.clone().unwrap();
        assert_translations_kept(&edited);
        assert!(s.undo());
        assert_eq!(doc(&s).metadata.xmp.as_deref(), Some(MULTILINGUAL_XMP));
        assert!(s.redo());
        assert_eq!(doc(&s).metadata.xmp.as_deref(), Some(edited.as_str()));
        let (bytes, _) = encode(doc(&s), "x.psd", None).unwrap();
        let back = photocraft_io::import("x.psd", &bytes).unwrap().document;
        assert_translations_kept(back.metadata.xmp.as_deref().unwrap());
        assert_eq!(read_file_info(back.metadata.xmp.as_deref())["keywords"], json!(["sunrise"]));
    }
}

#[test]
fn fit_image_and_conditional_mode_change() {
    let mut s = session(400, 200, 8);
    s.execute("file.automate.fitImage", json!({"width": 100, "height": 100})).unwrap();
    assert_eq!((doc(&s).size.width, doc(&s).size.height), (100, 50));
    let r = s.execute("file.automate.fitImage", json!({"width": 1000, "height": 1000, "dontEnlarge": true})).unwrap();
    assert_eq!(r["changed"], false);
    s.execute("file.automate.fitImage", json!({"width": 1000, "height": 1000})).unwrap();
    assert_eq!(doc(&s).size.width, 1000);
    let r = s.execute("file.automate.conditionalModeChange", json!({"from": ["cmyk"], "to": "grayscale"})).unwrap();
    assert_eq!(r["changed"], false);
    s.execute("file.automate.conditionalModeChange", json!({"from": ["rgb", "lab"], "to": "grayscale"})).unwrap();
    assert_eq!(doc(&s).mode, ColorMode::Grayscale);
    s.execute("file.automate.conditionalModeChange", json!({"to": "rgb"})).unwrap();
    assert_eq!(doc(&s).mode, ColorMode::Rgb);
    assert!(s.execute("file.automate.conditionalModeChange", json!({"to": "duotone"})).is_err());
}

#[test]
fn conditional_mode_change_cannot_run_a_mode_change_the_gate_denies() {
    fn deny_mode(id: &str, _: &Value) -> Result<()> {
        if id.starts_with("image.mode.") { Err(EngineError::Other(format!("automation command `{id}` is disabled"))) } else { Ok(()) }
    }
    let mut s = session(4, 4, 8);
    s.authorize = Some(deny_mode);
    let e = s.execute("file.automate.conditionalModeChange", json!({"to": "grayscale"})).unwrap_err();
    assert!(e.to_string().contains("image.mode.grayscale"), "{e}");
    assert_eq!(doc(&s).mode, ColorMode::Rgb, "the refused nested step leaves the document alone");
    // Commands that compose only allowed steps still run.
    s.execute("file.automate.fitImage", json!({"width": 2, "height": 2})).unwrap();
    assert_eq!(doc(&s).size.width, 2);
}

#[test]
fn flatten_all_layer_effects_and_masks() {
    for depth in [8, 16, 32] {
        let mut s = session(40, 40, depth);
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let id = s.active().unwrap().active_layer.unwrap();
        s.edit("fx", |doc, _| {
            let l = doc.layer_mut(id).unwrap();
            l.effects.enabled = true;
            l.effects.items.push(photocraft_doc::Effect::ColorOverlay {
                common: photocraft_doc::effects::FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
                color: photocraft_color::Color::rgb(0.0, 1.0, 0.0),
            });
            Ok(())
        })
        .unwrap();
        let before = composite(&s, 15, 15);
        assert!(before[1] > 0.99, "overlay renders green");
        assert_eq!(s.execute("file.scripts.flattenAllLayerEffects", json!({})).unwrap()["flattened"], 1);
        let l = doc(&s).layer(id).unwrap();
        assert!(l.effects.items.is_empty() && matches!(l.content, LayerContent::Raster(_)));
        let after = composite(&s, 15, 15);
        assert!((after[1] - before[1]).abs() < 0.01 && after[0] < 0.01, "{after:?}");
        // Masks: hide the left half, flatten, mask gone, pixels gone.
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 40})).unwrap();
        s.execute("select.inverse", json!({})).unwrap();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        assert!(composite(&s, 15, 15)[1] > 0.99 && composite(&s, 15, 15)[0] > 0.99, "white shows through the mask");
        let r = s.execute("file.scripts.flattenAllMasks", json!({})).unwrap();
        assert_eq!(r["flattened"], 1);
        let l = doc(&s).layer(id).unwrap();
        assert!(l.mask.is_none());
        assert!(l.surface().unwrap().content_bounds().x0 >= 20);
        assert!(composite(&s, 25, 15)[1] > 0.99 && composite(&s, 25, 15)[0] < 0.01);
        // Each script is a single history step.
        assert!(s.undo());
        assert!(doc(&s).layer(id).unwrap().mask.is_some());
    }
}

/// Flatten All Layer Effects must bake the effect visuals without destroying the layer's masks:
/// an enabled raster mask and a vector mask on one layer, a disabled raster mask on another
/// (#3804). The command is a visual no-op; the masks (pixels and state) survive, including a
/// `.pcraft` round trip.
#[test]
fn flatten_all_layer_effects_keeps_masks() {
    for depth in [8, 16, 32] {
        let mut s = session(40, 40, depth);
        // Layer 1: a red square with a green colour overlay, an enabled raster mask revealing
        // x < 20 and a vector mask revealing x 15..30.
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 10, "y": 10, "width": 20, "height": 20})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let id1 = s.active().unwrap().active_layer.unwrap();
        s.edit("fx", |doc, _| {
            let l = doc.layer_mut(id1).unwrap();
            l.effects.enabled = true;
            l.effects.items.push(photocraft_doc::Effect::ColorOverlay {
                common: photocraft_doc::effects::FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
                color: photocraft_color::Color::rgb(0.0, 1.0, 0.0),
            });
            l.vector_mask = Some(photocraft_doc::VectorMask::new(photocraft_doc::Path::new(vec![photocraft_doc::Subpath::polygon(&[
                (15.0, 15.0),
                (30.0, 15.0),
                (30.0, 30.0),
                (15.0, 30.0),
            ])])));
            Ok(())
        })
        .unwrap();
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 20, "height": 40})).unwrap();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        // Layer 2: a red square at the bottom-right with the same overlay and a *disabled* mask.
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("select.rect", json!({"x": 30, "y": 30, "width": 8, "height": 8})).unwrap();
        s.execute("edit.fill", json!({"color": "#ff0000"})).unwrap();
        s.execute("select.deselect", json!({})).unwrap();
        let id2 = s.active().unwrap().active_layer.unwrap();
        s.edit("fx", |doc, _| {
            let l = doc.layer_mut(id2).unwrap();
            l.effects.enabled = true;
            l.effects.items.push(photocraft_doc::Effect::ColorOverlay {
                common: photocraft_doc::effects::FxCommon::new(photocraft_color::BlendMode::Normal, 1.0),
                color: photocraft_color::Color::rgb(0.0, 1.0, 0.0),
            });
            Ok(())
        })
        .unwrap();
        s.execute("select.rect", json!({"x": 34, "y": 34, "width": 2, "height": 2})).unwrap();
        s.execute("layer.layerMask.revealSelection", json!({})).unwrap();
        s.edit("disable", |doc, _| {
            if let Some(m) = doc.layer_mut(id2).and_then(|l| l.mask.as_mut()) {
                m.enabled = false;
            }
            Ok(())
        })
        .unwrap();
        let m1 = doc(&s).layer(id1).unwrap().mask.clone();
        let v1 = doc(&s).layer(id1).unwrap().vector_mask.clone();
        let m2 = doc(&s).layer(id2).unwrap().mask.clone();
        assert_eq!(m1.as_ref().map(|m| m.enabled), Some(true), "layer 1 raster mask is enabled");
        assert!(v1.as_ref().is_some_and(|v| v.enabled), "layer 1 vector mask is enabled");
        assert_eq!(m2.as_ref().map(|m| m.enabled), Some(false), "layer 2 raster mask is disabled");
        let before = [
            composite(&s, 17, 15), // layer 1: inside both masks
            composite(&s, 22, 15), // layer 1: raster mask hides
            composite(&s, 12, 15), // layer 1: vector mask hides
            composite(&s, 34, 34), // layer 2: the disabled mask does not hide
        ];
        assert!(before[0][1] > 0.99 && before[0][0] < 0.01, "overlay renders green: {:?}", before[0]);
        // The white background shows where a mask hides the layer.
        assert!(before[1][0] > 0.99 && before[1][1] > 0.99, "the raster mask hides: {:?}", before[1]);
        assert!(before[2][0] > 0.99 && before[2][1] > 0.99, "the vector mask hides: {:?}", before[2]);
        assert!(before[3][1] > 0.99, "the disabled mask does not hide");
        let r = s.execute("file.scripts.flattenAllLayerEffects", json!({})).unwrap();
        assert_eq!(r["flattened"], 2);
        {
            let d = doc(&s);
            assert_eq!(d.layer(id1).unwrap().mask, m1, "layer 1 raster mask survives");
            assert_eq!(d.layer(id1).unwrap().vector_mask, v1, "layer 1 vector mask survives");
            assert_eq!(d.layer(id2).unwrap().mask, m2, "the disabled mask survives");
        }
        let after = [composite(&s, 17, 15), composite(&s, 22, 15), composite(&s, 12, 15), composite(&s, 34, 34)];
        for (a, b) in after.iter().zip(&before) {
            for k in 0..4 {
                assert!((a[k] - b[k]).abs() < 0.01, "flattening is a visual no-op: {after:?} vs {before:?}");
            }
        }
        // One history step: undo restores the effects, redo re-flattens, and the masks stay.
        assert!(s.undo());
        assert!(!doc(&s).layer(id1).unwrap().effects.items.is_empty());
        assert_eq!(doc(&s).layer(id1).unwrap().mask, m1);
        assert!(s.redo());
        assert_eq!(doc(&s).layer(id1).unwrap().mask, m1);
        assert_eq!(doc(&s).layer(id1).unwrap().vector_mask, v1);
        assert_eq!(doc(&s).layer(id2).unwrap().mask, m2);
        // The masks survive a `.pcraft` save and reopen.
        let dir = tmp("flatten-effects-masks");
        let path = join(&dir, "masks.pcraft");
        save_doc(doc(&s), &path, None).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let mut s2 = Session::new();
        open_bytes_as(&mut s2, "masks.pcraft", &bytes, None, None).unwrap();
        let layers: Vec<_> = doc(&s2).walk().into_iter().map(|(_, _, l)| l).collect();
        assert!(layers.iter().any(|l| l.mask == m1 && l.vector_mask == v1), "the raster and vector masks survive .pcraft");
        assert!(layers.iter().any(|l| l.mask == m2), "the disabled mask survives .pcraft");
    }
}

#[test]
fn layers_to_files_writes_each_visible_layer() {
    let dir = tmp("layers");
    let mut s = session(12, 12, 8);
    s.execute("layer.new.layer", json!({"name": "Ink/1"})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Hidden"})).unwrap();
    s.execute("layer.setProps", json!({"visible": false})).unwrap();
    let r = s.execute("file.export.layersToFiles", json!({"dir": dir, "prefix": "doc", "format": "png"})).unwrap();
    let files: Vec<String> = r["files"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(files.len(), 2, "{files:?}");
    assert!(files[0].ends_with("doc_0000_Ink_1.png"), "{files:?}");
    assert!(files.iter().all(|f| std::path::Path::new(f).is_file()));
    let r = s.execute("file.export.layersToFiles", json!({"dir": dir, "visibleOnly": false, "format": "psd"})).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), 3);
}

#[test]
fn load_files_into_stack_and_image_processor_and_batch() {
    let dir = tmp("batch");
    let input = join(&dir, "in");
    std::fs::create_dir_all(&input).unwrap();
    let a = png(&input, "a.png", 40, 20, "#ff0000");
    let b = png(&input, "b.png", 30, 30, "#0000ff");
    std::fs::write(join(&input, "notes.txt"), "not an image").unwrap();
    let mut s = Session::new();
    let r = s.execute("file.scripts.loadFilesIntoStack", json!({"paths": [a, b]})).unwrap();
    assert_eq!(r["layers"], 2);
    assert_eq!((doc(&s).size.width, doc(&s).size.height), (40, 30));
    assert_eq!(doc(&s).layers[0].name, "a.png");
    // Image Processor: fit into 20×20 JPEGs.
    let out = join(&dir, "jpg");
    let r = s.execute("file.scripts.imageProcessor", json!({"input": input, "output": out, "format": "jpg", "width": 20, "height": 20})).unwrap();
    let files = r["files"].as_array().unwrap();
    assert_eq!(files.len(), 2, "{r}");
    let back = photocraft_io::import("a.jpg", &std::fs::read(files[0].as_str().unwrap()).unwrap()).unwrap().document;
    assert_eq!((back.size.width, back.size.height), (20, 10));
    // Batch: run an action (invert) over the folder, save PNGs.
    let out = join(&dir, "png");
    let r = s.execute("file.automate.batch", json!({"steps": [["image.adjustments.invert", {}]], "input": input, "output": out, "format": "png"})).unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), 2, "{r}");
    assert!(r["errors"].as_array().unwrap().is_empty());
    let inv = photocraft_io::import("a.png", &std::fs::read(join(&out, "a.png")).unwrap()).unwrap().document;
    let px = photocraft_compose::render(&inv, Rect::new(1, 1, 2, 2)).px[0];
    assert!(px[0] < 0.01 && px[1] > 0.99 && px[2] > 0.99, "red inverted to cyan: {px:?}");
    // A recorded Quick Export writes to the current document path, but Batch owns
    // the destination. Replaying it would fail for scratch documents with no save path.
    let quick_out = join(&dir, "batch-with-quick-export");
    let r = s
        .execute(
            "file.automate.batch",
            json!({"steps": [["image.adjustments.invert", {}], ["file.export.quickExport", {}]], "input": input, "output": quick_out, "format": "png"}),
        )
        .unwrap();
    assert_eq!(r["files"].as_array().unwrap().len(), 2, "{r}");
    assert!(r["errors"].as_array().unwrap().is_empty(), "{r}");
    assert!(std::path::Path::new(&join(&quick_out, "a.png")).is_file());
    assert!(s.execute("file.automate.batch", json!({"steps": [["no.such.command", {}]], "input": input, "output": out})).is_err());
}

#[test]
fn color_lookup_table_bakes_the_adjustment_stack() {
    let mut s = session(8, 8, 8);
    assert!(!s.is_enabled("file.export.colorLookupTables"));
    s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    let r = s.execute("file.export.colorLookupTables", json!({"size": 3})).unwrap();
    let cube = r["cube"].as_str().unwrap();
    assert!(cube.contains("LUT_3D_SIZE 3"));
    let rows: Vec<Vec<f32>> =
        cube.lines().filter(|l| l.chars().next().is_some_and(|c| c.is_ascii_digit())).map(|l| l.split(' ').map(|v| v.parse().unwrap()).collect()).collect();
    assert_eq!(rows.len(), 27);
    // First entry (0,0,0) → white; second (r = 0.5) → 0.5; last (1,1,1) → black.
    assert!(rows[0].iter().all(|v| (v - 1.0).abs() < 1e-3), "{:?}", rows[0]);
    assert!((rows[1][0] - 0.5).abs() < 0.01 && (rows[1][1] - 1.0).abs() < 1e-3);
    assert!(rows[2][0].abs() < 1e-3 && (rows[2][2] - 1.0).abs() < 1e-3, "red fastest");
    assert!(rows[26].iter().all(|v| v.abs() < 1e-3));
    let dir = tmp("lut");
    let path = join(&dir, "invert.cube");
    s.execute("file.export.colorLookupTables", json!({"path": path, "size": 17})).unwrap();
    assert!(std::fs::read_to_string(&path).unwrap().contains("LUT_3D_SIZE 17"));
}

#[test]
fn cube_export_uses_only_selected_adjustments_in_document_order() {
    let mut s = session(8, 8, 8);
    let first = s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap();
    let second = s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap();
    let both = s.execute("file.export.colorLookupTables", json!({"size": 3})).unwrap();
    let all = photocraft_cms::lutfile::parse_cube(both["cube"].as_str().unwrap()).unwrap();
    assert_eq!(both["layerCount"], 2);
    assert!(all.data[0] < 0.001, "two Invert adjustments produce identity");

    // Selection comes from the Layers panel rather than the active layer alone.
    s.execute("layer.select", json!({"layer": first})).unwrap();
    let selected = s.execute("file.export.colorLookupTables", json!({"size": 3, "scope": "selected"})).unwrap();
    let one = photocraft_cms::lutfile::parse_cube(selected["cube"].as_str().unwrap()).unwrap();
    assert_eq!(selected["layerCount"], 1);
    assert!(one.data[0] > 0.999, "one invert maps black to white");
    assert_eq!(all.size, one.size);
    assert_eq!(s.execute("file.export.colorLookupTables", json!({"size": 3, "layers": [first]})).unwrap()["cube"], selected["cube"]);

    // Reversing the caller's ID list must not reverse document stack order.
    let normal = s.execute("file.export.colorLookupTables", json!({"size": 3, "layers": [first, second]})).unwrap();
    let reversed = s.execute("file.export.colorLookupTables", json!({"size": 3, "layers": [second, first]})).unwrap();
    assert_eq!(normal["cube"], reversed["cube"]);
    assert_eq!(normal["cube"], both["cube"]);

    // Additional non-adjustment layers must not affect the LUT export.
    s.execute("layer.new.layer", json!({"name": "Raster"})).unwrap();
    assert_eq!(s.execute("file.export.colorLookupTables", json!({"size": 3})).unwrap()["cube"], both["cube"]);
}

#[test]
fn selected_cube_validation_is_explicit_and_never_mutates_the_document() {
    let mut s = session(8, 8, 8);
    let a = s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap();
    let raster = s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap();
    let rev = s.active().unwrap().revision;
    let history = s.active().unwrap().history.past_len();
    let valid = s.execute("file.export.colorLookupTables", json!({"size": 3, "layers": [a]})).unwrap();
    assert_eq!(valid["layerCount"], 1);
    for params in [
        json!({"size": 1}),
        json!({"size": 130}),
        json!({"size": 1.5}),
        json!({"size": -3}),
        json!({"scope": "unknown"}),
        json!({"scope": 42}),
        json!({"scope": "selected", "layers": [a]}),
        json!({"layers": []}),
        json!({"layers": 42}),
        json!({"layers": [a, a]}),
        json!({"layers": [a, "oops"]}),
        json!({"layers": [u64::MAX]}),
        json!({"layers": [raster]}),
    ] {
        assert!(s.execute("file.export.colorLookupTables", params.clone()).is_err(), "{params}");
        assert_eq!(s.active().unwrap().revision, rev, "{params}: export must be read-only");
        assert_eq!(s.active().unwrap().history.past_len(), history);
    }
    s.execute("layer.select", json!({"layer": raster})).unwrap();
    assert!(s.execute("file.export.colorLookupTables", json!({"size": 3, "scope": "selected"})).is_err());
}

#[test]
fn exported_selected_cube_round_trips_through_lookup_importer() {
    let dir = tmp("selected_cube");
    let mut s = session(8, 8, 8);
    let a = s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap()["layer"].as_u64().unwrap();
    let path = join(&dir, "chosen.cube");
    let r = s.execute("file.export.colorLookupTables", json!({"size": 17, "title": "Chosen", "layers": [a], "path": path})).unwrap();
    assert_eq!(r["size"], 17);
    assert_eq!(r["layerCount"], 1);
    let text = std::fs::read_to_string(&path).unwrap();
    let parsed = photocraft_cms::lutfile::parse_cube(&text).unwrap();
    assert_eq!(parsed.size, 17);
    assert_eq!(parsed.data.len(), 17 * 17 * 17 * 3);
    assert!(parsed.data[..3].iter().all(|v| *v > 0.999));
    assert!(parsed.data[parsed.data.len() - 3..].iter().all(|v| *v < 0.001));
    s.execute("layer.newAdjustmentLayer.colorLookup", json!({"file": path})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), 3);
}

#[test]
fn guide_layouts() {
    let mut s = session(100, 50, 8);
    let r = s.execute("view.newGuideLayout", json!({"columns": 2, "gutter": 10, "margin": [5, 10, 5, 10]})).unwrap();
    assert_eq!(r["vertical"], 4);
    let g = &doc(&s).guides;
    assert_eq!(g.vertical, vec![10.0, 90.0, 45.0, 55.0]);
    assert_eq!(g.horizontal, vec![5.0, 45.0]);
    s.execute("view.newGuideLayout", json!({"rows": 2, "clearExisting": true})).unwrap();
    let g = &doc(&s).guides;
    assert!(g.vertical.is_empty());
    assert_eq!(g.horizontal, vec![0.0, 25.0, 50.0]);
    // Fixed-width centred columns.
    s.execute("view.newGuideLayout", json!({"columns": 2, "width": 20, "gutter": 10, "centerColumns": true, "clearExisting": true})).unwrap();
    assert_eq!(doc(&s).guides.vertical, vec![25.0, 45.0, 55.0, 75.0]);
    assert!(s.execute("view.newGuideLayout", json!({})).is_err());
    assert_eq!(s.execute("view.clearCanvasGuides", json!({})).unwrap()["cleared"], 4);
    assert!(doc(&s).guides.vertical.is_empty());
    assert!(!s.is_enabled("view.newGuidesFromShape"));
    s.execute("shape.create", json!({"kind": "rect", "rect": [10, 10, 40, 20]})).unwrap();
    s.execute("view.newGuidesFromShape", json!({})).unwrap();
    let g = &doc(&s).guides;
    assert_eq!(g.vertical, vec![10.0, 30.0, 50.0]);
    assert_eq!(g.horizontal, vec![10.0, 20.0, 30.0]);
}

#[test]
fn only_layered_files_save_in_place() {
    for path in ["a.psd", "dir/a.PSB", r"C:\w\a.pcraft", "my.dir/a.psd", "a.ora", "dir/A.ORA"] {
        assert!(saves_in_place(path), "{path}");
    }
    // Flat formats, no extension, a dotted folder with an extensionless file, a dot file.
    for path in ["a.png", "a.jpg", "a.pdn", "a.PDN", "a", "my.psd/a", ".psd", "a.", ""] {
        assert!(!saves_in_place(path), "{path}");
    }
    assert_eq!(extension("dir/Photo.JPEG").as_deref(), Some("jpeg"));
    assert_eq!(extension("my.dir/name"), None);
}

/// #3806: a layered format holds a document only if it keeps everything in it.
#[test]
fn openraster_holds_a_document_only_when_it_drops_nothing() {
    let mut s = session(16, 16, 8);
    s.execute("layer.new.layer", json!({})).unwrap();
    let plain = doc(&s).clone();
    for p in ["a.psd", "a.psb", "a.pcraft", "a.ora", "A.ORA"] {
        assert!(holds_at(&plain, p), "{p}");
    }
    for p in ["a.png", "a.tif", "a", ""] {
        assert!(!holds_at(&plain, p), "{p}");
    }
    assert!(losses_at(&plain, "a.ora").is_empty());
    s.execute("layer.newAdjustmentLayer.invert", json!({})).unwrap();
    let adjusted = doc(&s).clone();
    assert!(holds_at(&adjusted, "a.psd") && !holds_at(&adjusted, "a.ora"));
    assert!(losses_at(&adjusted, "a.ora").iter().any(|l| l.contains("adjustment layers")), "{:?}", losses_at(&adjusted, "a.ora"));
    assert!(losses_at(&adjusted, "a.psd").is_empty());
}

#[test]
fn templates_open_untitled_without_their_path() {
    assert!(is_template("dir/card.PSDT") && !is_template("card.psd") && !is_template("psdt"));
    let dir = tmp("template");
    let path = format!("{dir}/card.psdt");
    let mut s = session(8, 8, 8);
    std::fs::write(&path, encode(doc(&s), "x.psd", None).unwrap().0).unwrap();
    s.execute("file.openAs", json!({"path": path})).unwrap();
    s.execute("file.openAs", json!({"path": path, "as": "psd"})).unwrap();
    let opened: Vec<_> = s.documents()[1..].iter().map(|d| (d.doc.name.as_str(), d.path.as_deref())).collect();
    assert_eq!(opened, [("Untitled-1", None), ("Untitled-2", None)]);
}

#[test]
fn batch_naming_serial_keeps_same_named_inputs_and_their_pixels() {
    let dir = tmp("naming-serial");
    let first = join(&dir, "first");
    let second = join(&dir, "second");
    std::fs::create_dir_all(&first).unwrap();
    std::fs::create_dir_all(&second).unwrap();
    let inputs = [png(&first, "photo.png", 8, 8, "#ff0000"), png(&second, "photo.png", 8, 8, "#0000ff")];
    let out = join(&dir, "out");
    let mut s = Session::new();
    let result = s
        .execute(
            "file.automate.batch",
            json!({
                "input": inputs, "output": out, "steps": [], "format": "png",
                "naming": "{name}_{serial}", "startingSerial": 7
            }),
        )
        .unwrap();
    assert_eq!(result["errors"], json!([]), "{result}");
    assert_eq!(result["files"], json!([join(&out, "photo_007.png"), join(&out, "photo_008.png")]));
    for (name, expected) in [("photo_007.png", [1.0, 0.0, 0.0, 1.0]), ("photo_008.png", [0.0, 0.0, 1.0, 1.0])] {
        let saved = photocraft_io::import(name, &std::fs::read(join(&out, name)).unwrap()).unwrap().document;
        let pixel = photocraft_compose::render(&saved, Rect::new(1, 1, 2, 2)).px[0];
        assert_eq!(pixel, expected, "{name}");
    }
}

/// Two inputs with one output name: the first result is kept and the second is an error, never
/// a silent overwrite listed as written (#422).
#[test]
fn batch_reports_inputs_that_share_an_output_name() {
    let dir = tmp("collide");
    let input = join(&dir, "in");
    std::fs::create_dir_all(&input).unwrap();
    png(&input, "a.png", 8, 8, "#ff0000");
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8, "background": "#0000ff"})).unwrap();
    save_doc(doc(&s), &join(&input, "a.jpg"), None).unwrap();
    png(&input, "B.png", 8, 8, "#00ff00");
    png(&input, "b.tif", 8, 8, "#00ff00");
    for (cmd, extra) in [
        ("file.automate.batch", json!({"steps": [], "format": "jpg"})),
        ("file.scripts.imageProcessor", json!({})),
        ("file.automate.lensCorrection", json!({"format": "jpg"})),
    ] {
        let out = join(&dir, cmd);
        let mut p = extra;
        p["input"] = json!(input);
        p["output"] = json!(out);
        let r = s.execute(cmd, p).unwrap();
        let files = r["files"].as_array().unwrap();
        let errors = r["errors"].as_array().unwrap();
        // `a.*` and `B.png`/`b.tif` (names that differ only in case) each collide once.
        assert_eq!((files.len(), errors.len()), (2, 2), "{cmd}: {r}");
        assert!(errors.iter().all(|e| e["error"].as_str().unwrap().contains("not written")), "{r}");
        let mut written: Vec<_> = files.iter().map(|f| f.as_str().unwrap().to_lowercase()).collect();
        written.dedup();
        assert_eq!(written.len(), 2, "each listed file is distinct: {r}");
    }
    // Keeping each file's format, nothing collides.
    let out = join(&dir, "same");
    let r = s.execute("file.automate.batch", json!({"steps": [], "input": input, "output": out})).unwrap();
    assert_eq!((r["files"].as_array().unwrap().len(), r["errors"].as_array().unwrap().len()), (4, 0), "{r}");
    // A droplet over two folders holding the same name.
    let other = join(&dir, "other");
    std::fs::create_dir_all(&other).unwrap();
    png(&other, "a.png", 8, 8, "#ffffff");
    let droplet = join(&dir, "d.pcdroplet");
    std::fs::write(&droplet, json!({"photocraftDroplet": 1, "name": "d", "action": {"steps": []}, "options": {"format": "png"}}).to_string()).unwrap();
    let out = join(&dir, "droplet");
    let r = s.execute("file.automate.runDroplet", json!({"droplet": droplet, "input": [join(&input, "a.png"), other], "output": out})).unwrap();
    assert_eq!((r["files"].as_array().unwrap().len(), r["errors"].as_array().unwrap().len()), (1, 1), "{r}");
    let kept = photocraft_io::import("a.png", &std::fs::read(join(&out, "a.png")).unwrap()).unwrap().document;
    let px = photocraft_compose::render(&kept, Rect::new(1, 1, 2, 2)).px[0];
    assert!(px[0] > 0.99 && px[1] < 0.01, "the first input's result is kept: {px:?}");
}

/// `file.revealInFinder` (the document tab's Reveal, UI-217-6) reveals the clicked document's
/// saved path — the tab menu passes `document`, not just the active one.
#[test]
fn reveal_in_finder_reveals_the_documents_saved_path() {
    let mut s = session(8, 8, 8);
    assert!(s.is_enabled("file.revealInFinder"));
    // A new document has no file yet.
    let err = s.execute("file.revealInFinder", json!({"dryRun": true})).unwrap_err();
    assert!(err.to_string().contains("no saved file"), "{err}");
    let path = "/tmp/pics/a.psd";
    s.active_mut().unwrap().path = Some(path.into());
    let r = s.execute("file.revealInFinder", json!({"dryRun": true})).unwrap();
    let (program, args) = crate::layer_menu_cmds::reveal_command(path);
    assert_eq!(r["program"], json!(program));
    assert_eq!(r["args"], json!(args));
    // The tab menu names a background tab: the active document is not the one revealed.
    s.execute("file.new", json!({"width": 8, "height": 8, "name": "other"})).unwrap();
    let err = s.execute("file.revealInFinder", json!({"document": 1, "dryRun": true})).unwrap_err();
    assert!(err.to_string().contains("no saved file"), "{err}");
    let r = s.execute("file.revealInFinder", json!({"document": 0, "dryRun": true})).unwrap();
    assert_eq!(r["program"], json!(program));
    assert_eq!(r["args"], json!(args));
}
#[test]
fn batch_naming_validates_the_whole_plan_before_writing() {
    let dir = tmp("naming-invalid");
    let input = png(&dir, "source.png", 8, 8, "#ff0000");
    let mut s = Session::new();
    for (index, extra) in [
        json!({"naming": "../{name}"}),
        json!({"naming": "sub\\{name}"}),
        json!({"naming": "C:{name}"}),
        json!({"naming": "{unknown}"}),
        json!({"naming": "{name"}),
        json!({"naming": "name}"}),
        json!({"naming": 42}),
        json!({"naming": "{serial}", "startingSerial": 0}),
        json!({"naming": "{serial}", "startingSerial": 1.5}),
        json!({"naming": "{serial}", "startingSerial": "1"}),
        json!({"naming": "{serial}", "startingSerial": 4294967295_u64}),
        json!({"naming": "x".repeat(256)}),
    ]
    .into_iter()
    .enumerate()
    {
        let out = join(&dir, &format!("out-{index}"));
        let mut params = json!({"input": [input, input], "output": out, "steps": []});
        params.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        let error = s.execute("file.automate.batch", params).unwrap_err();
        assert!(matches!(error, EngineError::BadParams { .. }), "{error}");
        assert!(!std::path::Path::new(&out).exists(), "invalid naming wrote outputs: {extra}");
    }
}

#[test]
fn batch_naming_keeps_serial_positions_after_errors_and_captures_the_utc_date() {
    let dir = tmp("naming-errors");
    let input = png(&dir, "good.png", 8, 8, "#0000ff");
    let out = join(&dir, "out");
    let before = crate::analysis_cmds::now_iso();
    let mut s = Session::new();
    let result = s
        .execute(
            "file.automate.batch",
            json!({
                "input": [join(&dir, "missing.png"), input], "output": out, "steps": [],
                "naming": "{date}_{serial}_{name}", "startingSerial": 9
            }),
        )
        .unwrap();
    let after = crate::analysis_cmds::now_iso();
    assert_eq!(result["errors"].as_array().unwrap().len(), 1, "{result}");
    let files = result["files"].as_array().unwrap();
    assert_eq!(files.len(), 1, "{result}");
    let name = file_name(files[0].as_str().unwrap());
    assert!(
        name == format!("{}_010_good.png", &before[..10]) || name == format!("{}_010_good.png", &after[..10]),
        "the missing input still consumes its serial: {name}"
    );
}

#[test]
fn batch_naming_still_reports_collisions_instead_of_overwriting() {
    let dir = tmp("naming-collision");
    let red = png(&dir, "red.png", 8, 8, "#ff0000");
    let blue = png(&dir, "blue.png", 8, 8, "#0000ff");
    let out = join(&dir, "out");
    let mut s = Session::new();
    let result = s
        .execute(
            "file.automate.batch",
            json!({
                "input": [red, blue], "output": out, "steps": [], "naming": "constant"
            }),
        )
        .unwrap();
    assert_eq!(result["files"], json!([join(&out, "constant.png")]));
    assert_eq!(result["errors"].as_array().unwrap().len(), 1, "{result}");
    assert!(result["errors"][0]["error"].as_str().unwrap().contains("not written"));
    let saved = photocraft_io::import("constant.png", &std::fs::read(join(&out, "constant.png")).unwrap()).unwrap().document;
    assert_eq!(photocraft_compose::render(&saved, Rect::new(1, 1, 2, 2)).px[0], [1.0, 0.0, 0.0, 1.0]);
}
#[test]
fn batch_naming_rejects_outputs_that_replace_inputs_before_processing() {
    let dir = tmp("naming-input-alias");
    let first = png(&dir, "002.png", 8, 8, "#ff0000");
    let second = png(&dir, "001.png", 8, 8, "#0000ff");
    let before = [std::fs::read(&first).unwrap(), std::fs::read(&second).unwrap()];
    let mut s = Session::new();
    // Without preflight, processing 002 first writes 001, so the next input's original pixels
    // are lost before it is read. The full plan must be rejected without writing either file.
    let error = s
        .execute(
            "file.automate.batch",
            json!({
                "input": [first, second], "output": join(&dir, "."), "steps": [], "naming": "{serial}"
            }),
        )
        .unwrap_err();
    assert!(error.to_string().contains("input"), "{error}");
    assert_eq!(std::fs::read(&first).unwrap(), before[0]);
    assert_eq!(std::fs::read(&second).unwrap(), before[1]);
}
