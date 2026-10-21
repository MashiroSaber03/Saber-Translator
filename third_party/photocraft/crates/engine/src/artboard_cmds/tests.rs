use super::*;
use photocraft_geom::Rect;

fn session(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 100, "height": 80, "depth": depth})).unwrap();
    s
}

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

fn add_square(s: &mut Session, r: Rect, rgba: [f32; 4]) -> LayerId {
    s.edit("sq", |doc, active| {
        let fmt = doc.pixel_format();
        let mut l = Layer::raster(doc.next_layer_name("Layer"), fmt);
        l.surface_mut().unwrap().fill_rect(r, &photocraft_raster::from_rgba(&fmt, rgba));
        let id = doc.insert_above(None, l);
        *active = Some(id);
        Ok(id)
    })
    .unwrap()
}

#[test]
fn new_artboards_place_and_grow_canvas() {
    let mut s = session(8);
    let r = s.execute("layer.new.artboard", json!({"rect": [0, 0, 100, 80]})).unwrap();
    let a1 = LayerId(r["layer"].as_u64().unwrap());
    assert_eq!(doc(&s).layer(a1).unwrap().name, "Artboard 1");
    // Default: canvas-sized, right of the last board with a gap; the canvas grows.
    let r = s.execute("layer.new.artboard", json!({"background": "black"})).unwrap();
    assert_eq!(r["rect"], json!([200, 0, 100, 80]));
    assert_eq!(doc(&s).size.width, 300);
    let r = s.execute("layer.new.artboard", json!({"preset": "iPhone 14", "x": 0, "y": 100})).unwrap();
    assert_eq!(r["rect"], json!([0, 100, 390, 844]));
    assert_eq!(doc(&s).artboards().len(), 3);
    assert!(s.execute("layer.new.artboard", json!({"preset": "Nope"})).is_err());
    assert!(s.execute("layer.new.artboard", json!({"rect": [0, 0, 0, 5]})).is_err());
    assert!(s.execute("layer.new.artboard", json!({"background": "plaid"})).is_err());
    // Undo removes the board (and the canvas growth).
    s.undo();
    assert_eq!(doc(&s).artboards().len(), 2);
    assert_eq!(doc(&s).size.height, 80);
}

#[test]
fn from_layers_and_group_then_compose_clips() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let red = add_square(&mut s, Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 1.0]);
        let r = s.execute("layer.new.artboardFromLayers", json!({"name": "Board"})).unwrap();
        assert_eq!(r["rect"], json!([10, 10, 20, 20]));
        let gid = LayerId(r["layer"].as_u64().unwrap());
        assert_eq!(doc(&s).artboard_of(red), Some(gid));
        // Grow the board; the white background shows around the square, nothing outside it.
        s.execute("layer.artboard.set", json!({"width": 40, "height": 40})).unwrap();
        let px = |s: &Session, x, y| photocraft_compose::render(doc(s), Rect::new(x, y, x + 1, y + 1)).px[0];
        assert!(px(&s, 40, 40)[0] > 0.99 && px(&s, 40, 40)[1] > 0.99, "{depth}");
        assert!(px(&s, 15, 15)[1] < 0.01);
        // Moving the board moves its contents.
        s.execute("layer.artboard.set", json!({"x": 50, "y": 20})).unwrap();
        let a = doc(&s).layer(gid).unwrap().artboard().unwrap().clone();
        assert_eq!(a.rect, Rect::from_xywh(50, 20, 40, 40));
        let surf = doc(&s).layer(red).unwrap().surface().unwrap();
        assert_eq!(surf.content_bounds(), Rect::new(50, 20, 70, 40));
        // layer.translate on the board does the same.
        s.execute("layer.translate", json!({"layer": gid.0, "dx": -5, "dy": 0})).unwrap();
        assert_eq!(doc(&s).layer(gid).unwrap().artboard().unwrap().rect.x0, 45);
        assert_eq!(doc(&s).layer(red).unwrap().surface().unwrap().content_bounds().x0, 45);
        // Background and name.
        s.execute("layer.artboard.set", json!({"background": "custom", "color": "#00ff00", "name": "Green"})).unwrap();
        let l = doc(&s).layer(gid).unwrap();
        assert_eq!(l.name, "Green");
        assert!(matches!(l.artboard().unwrap().background, ArtboardBackground::Custom(_)));
        // A plain group becomes an artboard sized to its contents.
        add_square(&mut s, Rect::new(0, 60, 8, 70), [0.0, 0.0, 1.0, 1.0]);
        let g = s.execute("layer.groupLayers", json!({})).unwrap();
        let gid2 = g["layer"].as_u64().unwrap();
        assert!(s.is_enabled("layer.new.artboardFromGroup"));
        let r = s.execute("layer.new.artboardFromGroup", json!({"layer": gid2})).unwrap();
        assert_eq!(r["rect"], json!([0, 60, 8, 10]));
        assert!(!s.is_enabled("layer.new.artboardFromGroup"));
    }
}

#[test]
fn clear_artboard_guides_only_inside() {
    let mut s = session(8);
    s.execute("layer.new.artboard", json!({"rect": [0, 0, 50, 50]})).unwrap();
    s.edit("guides", |d, _| {
        d.guides.vertical = vec![10.0, 70.0];
        d.guides.horizontal = vec![20.0];
        Ok(())
    })
    .unwrap();
    assert_eq!(s.execute("view.clearSelectedArtboardGuides", json!({})).unwrap()["cleared"], 2);
    assert_eq!(doc(&s).guides.vertical, vec![70.0]);
    assert!(doc(&s).guides.horizontal.is_empty());
    // Clear Canvas Guides leaves the artboard's guides.
    s.edit("guides", |d, _| {
        d.guides.vertical = vec![10.0, 70.0];
        Ok(())
    })
    .unwrap();
    assert_eq!(s.execute("view.clearCanvasGuides", json!({})).unwrap()["cleared"], 1);
    assert_eq!(doc(&s).guides.vertical, vec![10.0]);
}

/// Clearing guides drops their colours, so a guide added later at the same place doesn't inherit one.
#[test]
fn clearing_guides_drops_their_colours() {
    let mut s = session(8);
    let coloured = |s: &mut Session, x: f64| s.execute("view.newGuide", json!({"orientation": "vertical", "position": x, "color": "magenta"})).unwrap();
    let colour = |s: &Session, x: f64| doc(s).guides.color(true, x);
    s.execute("layer.new.artboard", json!({"rect": [0, 0, 50, 50]})).unwrap();
    coloured(&mut s, 10.0);
    coloured(&mut s, 70.0);
    s.execute("view.clearSelectedArtboardGuides", json!({})).unwrap();
    s.execute("view.clearCanvasGuides", json!({})).unwrap();
    assert!(doc(&s).guides.colors.is_empty());
    s.execute("view.newGuide", json!({"orientation": "vertical", "position": 10})).unwrap();
    s.execute("view.newGuide", json!({"orientation": "vertical", "position": 70})).unwrap();
    assert_eq!((colour(&s, 10.0), colour(&s, 70.0)), (None, None));
    // New Guide Layout's Clear Existing Guides likewise; a guide it keeps keeps its colour.
    s.execute("view.clearGuides", json!({})).unwrap();
    coloured(&mut s, 0.0);
    s.execute("view.newGuideLayout", json!({"columns": 1})).unwrap();
    assert_eq!(colour(&s, 0.0), Some([0xff, 0x00, 0xff]));
    s.execute("view.newGuideLayout", json!({"columns": 1, "clearExisting": true})).unwrap();
    assert!(doc(&s).guides.vertical.contains(&0.0));
    assert_eq!(colour(&s, 0.0), None);
}

#[test]
fn export_to_files_and_pdf() {
    let mut s = session(16);
    add_square(&mut s, Rect::new(0, 0, 10, 10), [1.0, 0.0, 0.0, 1.0]);
    s.execute("layer.new.artboardFromLayers", json!({"name": "Red Board"})).unwrap();
    s.execute("layer.new.artboard", json!({"rect": [20, 0, 30, 15], "name": "Empty/One", "background": "transparent"})).unwrap();
    let dir = std::env::temp_dir().join(format!("photocraft-artboards-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let d = dir.to_string_lossy().into_owned();
    let r = s.execute("file.export.artboardsToFiles", json!({"dir": d, "prefix": "site"})).unwrap();
    let files: Vec<String> = r["files"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(files.len(), 2);
    assert!(files[0].ends_with("site_Empty_One.png"), "{files:?}");
    assert!(files[1].ends_with("site_Red Board.png"), "{files:?}");
    let img = photocraft_io::import("x.png", &std::fs::read(&files[1]).unwrap()).unwrap().document;
    assert_eq!((img.size.width, img.size.height), (10, 10));
    let img2 = photocraft_io::import("x.png", &std::fs::read(&files[0]).unwrap()).unwrap().document;
    assert_eq!((img2.size.width, img2.size.height), (30, 15));
    // Layered PSD keeps the board, moved to the origin.
    let r = s.execute("file.export.artboardsToFiles", json!({"dir": d, "prefix": "", "format": "psd", "artboards": [doc(&s).artboards()[0].0.0]})).unwrap();
    let psd = photocraft_io::import("x.psd", &std::fs::read(r["files"][0].as_str().unwrap()).unwrap()).unwrap().document;
    assert_eq!(psd.artboards()[0].2.rect, Rect::new(0, 0, 10, 10));
    let pdf = join(&d, "boards.pdf");
    let r = s.execute("file.export.artboardsToPdf", json!({"path": pdf})).unwrap();
    assert_eq!(r["pages"], 2);
    let bytes = std::fs::read(&pdf).unwrap();
    assert!(bytes.starts_with(b"%PDF-1.4"));
    assert!(bytes.ends_with(b"%%EOF\n"));
    let text = String::from_utf8_lossy(&bytes);
    assert_eq!(text.matches("/Type /Page ").count(), 2);
    assert_eq!(text.matches("/DCTDecode").count(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn raster_pdf_xref_offsets_point_at_objects() {
    let pdf = raster_pdf(&[(2, 3, 72.0, vec![0xff, 0xd8, 0xff, 0xd9])]);
    let text = String::from_utf8_lossy(&pdf).into_owned();
    let xref = text.rfind("xref\n").unwrap();
    let offsets: Vec<usize> = text[xref..].lines().skip(3).take(5).map(|l| l[..10].parse().unwrap()).collect();
    for (i, o) in offsets.iter().enumerate() {
        assert!(text[*o..].starts_with(&format!("{} 0 obj", i + 1)));
    }
    assert!(text.contains("/MediaBox [0 0 2.000 3.000]"));
}

#[test]
fn disabled_without_artboards() {
    let mut s = session(8);
    for id in
        ["layer.artboard.set", "view.clearSelectedArtboardGuides", "file.export.artboardsToFiles", "file.export.artboardsToPdf", "layer.new.artboardFromGroup"]
    {
        assert!(!s.is_enabled(id), "{id}");
    }
    assert!(s.is_enabled("layer.new.artboard"));
    assert!(s.is_enabled("layer.new.artboardFromLayers"));
    let _ = s.execute("layer.new.artboard", json!({}));
    assert!(!s.is_enabled("layer.new.artboardFromLayers"));
}

/// Runs `cmd` and asserts it fails with a bad-params error that leaves the document, its
/// revision, its active layer and its history exactly as they were (#931).
fn assert_rejected(s: &mut Session, cmd: &str, p: Value) -> String {
    let d = s.active().unwrap();
    let (before, rev, active, past) = (d.doc.clone(), d.revision, d.active_layer, d.history.past_len());
    let err = s.execute(cmd, p.clone()).expect_err(&format!("{cmd} {p} should fail"));
    assert!(matches!(err, EngineError::BadParams { .. }), "{cmd} {p}: {err}");
    let d = s.active().unwrap();
    assert!(d.doc == before, "{cmd} {p} changed the document");
    assert_eq!(d.revision, rev, "{cmd} {p}");
    assert_eq!(d.active_layer, active, "{cmd} {p}");
    assert_eq!(d.history.past_len(), past, "{cmd} {p} recorded a history step");
    err.to_string()
}

fn board_rect(s: &Session, id: u64) -> Rect {
    doc(s).layer(LayerId(id)).unwrap().artboard().unwrap().rect
}

#[test]
fn new_artboard_rejects_coordinates_past_the_32_bit_range() {
    let mut s = session(8);
    // A board flush with the far right edge: x1 = i32::MAX.
    let r = s.execute("layer.new.artboard", json!({"x": i32::MAX - 50, "y": 0, "width": 50, "height": 50})).unwrap();
    assert_eq!(r["rect"], json!([i32::MAX - 50, 0, 50, 50]));
    // Default placement right of it has no room: an error naming the edge, not a wrapped x.
    let msg = assert_rejected(&mut s, "layer.new.artboard", json!({}));
    assert!(msg.contains(&i32::MAX.to_string()), "{msg}");
    assert_rejected(&mut s, "layer.new.artboard", json!({"preset": "iPhone 14"}));
    // An explicit position still works beside it (the canvas grew to its 300000 px cap).
    let r = s.execute("layer.new.artboard", json!({"x": 0})).unwrap();
    assert_eq!(r["rect"], json!([0, 0, 300_000, 80]));
    let r = s.execute("layer.new.artboard", json!({"preset": "A4", "x": 0, "y": 100})).unwrap();
    assert_eq!(r["rect"], json!([0, 100, 595, 842]));
    assert_eq!(doc(&s).artboards().len(), 3);

    // Default placement whose x fits but whose canvas-sized width runs past the edge.
    let mut s = session(8);
    s.execute("layer.new.artboard", json!({"x": i32::MAX - 250, "width": 50, "height": 50})).unwrap();
    assert_rejected(&mut s, "layer.new.artboard", json!({}));

    // Explicit rects that would end past i32::MAX (or start outside it) are rejected, not clamped.
    let mut s = session(8);
    for p in [
        json!({"x": i32::MAX, "width": 50, "height": 50}),
        json!({"x": i32::MAX - 10, "width": 50, "height": 50}),
        json!({"y": i32::MAX - 10, "width": 50, "height": 50}),
        json!({"rect": [i32::MAX - 10, 0, 50, 50]}),
        json!({"rect": [0, i32::MAX, 50, 50]}),
        json!({"rect": [3e9, 0, 50, 50]}),
        json!({"rect": [-3e9, 0, 50, 50]}),
        json!({"rect": [0, 0, 3e9, 50]}),
        json!({"x": 0, "width": 3_000_000_000_i64, "height": 50}),
    ] {
        assert_rejected(&mut s, "layer.new.artboard", p);
    }
    // Representable extremes are fine: flush with either edge.
    let r = s.execute("layer.new.artboard", json!({"x": i32::MIN, "y": i32::MIN, "width": 50, "height": 50})).unwrap();
    assert_eq!(r["rect"], json!([i32::MIN, i32::MIN, 50, 50]));
    let r = s.execute("layer.new.artboard", json!({"rect": [i32::MAX - 50, i32::MAX - 50, 50, 50]})).unwrap();
    assert_eq!(r["rect"], json!([i32::MAX - 50, i32::MAX - 50, 50, 50]));
}

#[test]
fn edit_artboard_rejects_moves_and_sizes_past_the_32_bit_range() {
    let mut s = session(8);
    let r = s.execute("layer.new.artboard", json!({"x": i32::MIN, "y": i32::MIN, "width": 50, "height": 50})).unwrap();
    let id = r["layer"].as_u64().unwrap();
    // Moving the board with its contents from one edge to the other needs an offset of about
    // 2^32, which no layer can be translated by.
    let msg = assert_rejected(&mut s, "layer.artboard.set", json!({"x": i32::MAX - 50}));
    assert!(msg.contains("4294967245"), "{msg}");
    assert_rejected(&mut s, "layer.artboard.set", json!({"y": i32::MAX - 50}));
    // A rect that would end past the edge is rejected with or without moving the contents.
    assert_rejected(&mut s, "layer.artboard.set", json!({"x": i32::MAX, "moveContents": false}));
    assert_rejected(&mut s, "layer.artboard.set", json!({"x": i32::MAX - 10}));
    assert_rejected(&mut s, "layer.artboard.set", json!({"rect": [i32::MAX - 10, 0, 50, 50], "moveContents": false}));
    // Without the contents only the board's own rect changes, so the far move is fine.
    let r = s.execute("layer.artboard.set", json!({"x": i32::MAX - 50, "y": i32::MAX - 50, "moveContents": false})).unwrap();
    assert_eq!(r["rect"], json!([i32::MAX - 50, i32::MAX - 50, 50, 50]));
    assert_eq!(board_rect(&s, id), Rect::new(i32::MAX - 50, i32::MAX - 50, i32::MAX, i32::MAX));
    // Resizing past the edge is rejected instead of shrinking the board to fit.
    assert_rejected(&mut s, "layer.artboard.set", json!({"width": 51}));
    assert_rejected(&mut s, "layer.artboard.set", json!({"height": 3_000_000_000_i64}));
    assert_rejected(&mut s, "layer.artboard.set", json!({"preset": "iPhone 14"}));
    // A board at the edge can still be shrunk or moved back in.
    let r = s.execute("layer.artboard.set", json!({"width": 20})).unwrap();
    assert_eq!(r["rect"], json!([i32::MAX - 50, i32::MAX - 50, 20, 50]));
    let r = s.execute("layer.artboard.set", json!({"x": 0, "y": 0, "moveContents": false})).unwrap();
    assert_eq!(r["rect"], json!([0, 0, 20, 50]));
}

#[test]
fn artboards_at_ordinary_coordinates_still_place_and_edit() {
    // Default placement leaves a 100 px gap right of the rightmost board.
    let mut s = session(8);
    let r = s.execute("layer.new.artboard", json!({"x": 10, "width": 50, "height": 50})).unwrap();
    let first = r["layer"].as_u64().unwrap();
    let r = s.execute("layer.new.artboard", json!({"width": 50})).unwrap();
    assert_eq!(r["rect"], json!([160, 0, 50, 80]));
    // Editing moves the board (and its contents) and resizes it.
    let red = add_square(&mut s, Rect::new(20, 10, 30, 20), [1.0, 0.0, 0.0, 1.0]);
    s.edit("into board", |doc, _| {
        let l = doc.remove(red).unwrap();
        let LayerContent::Group(g) = &mut doc.layer_mut(LayerId(first)).unwrap().content else { panic!("not a group") };
        g.children.push(l);
        Ok(())
    })
    .unwrap();
    let r = s.execute("layer.artboard.set", json!({"layer": first, "x": 200})).unwrap();
    assert_eq!(r["rect"], json!([200, 0, 50, 50]));
    assert_eq!(doc(&s).layer(red).unwrap().surface().unwrap().content_bounds(), Rect::new(210, 10, 220, 20));
    let r = s.execute("layer.artboard.set", json!({"layer": first, "width": 70, "height": 40})).unwrap();
    assert_eq!(r["rect"], json!([200, 0, 70, 40]));
}

#[test]
fn export_rejects_a_board_too_far_from_the_origin() {
    let mut s = session(8);
    s.execute("layer.new.artboard", json!({"x": i32::MIN, "width": 50, "height": 50, "name": "Far"})).unwrap();
    let dir = std::env::temp_dir().join(format!("photocraft-artboards-931-{}", std::process::id()));
    let d = dir.to_string_lossy().into_owned();
    // Moving the board to the origin needs dx = 2^31, one past i32::MAX: an error, not a panic.
    let err = s.execute("file.export.artboardsToFiles", json!({"dir": d})).unwrap_err().to_string();
    assert!(err.contains(&i32::MIN.to_string()), "{err}");
    let pdf = join(&d, "boards.pdf");
    assert!(s.execute("file.export.artboardsToPdf", json!({"path": pdf})).is_err());
    assert!(!dir.exists(), "nothing was written");
    // One pixel in from the edge is representable and still exports.
    let one = artboard_document(doc(&s), doc(&s).artboards()[0].0);
    assert!(one.is_err());
    s.execute("layer.artboard.set", json!({"x": i32::MIN + 1, "moveContents": false})).unwrap();
    let one = artboard_document(doc(&s), doc(&s).artboards()[0].0).unwrap().unwrap();
    assert_eq!((one.size.width, one.size.height), (50, 50));
    assert_eq!(one.artboards()[0].2.rect, Rect::new(0, 0, 50, 50));
}

/// #1531: a board 40×40 at (10, 10) holding a red square at (10..30)², the artboard active.
fn board_with_square(s: &mut Session) -> LayerId {
    add_square(s, Rect::new(10, 10, 30, 30), [1.0, 0.0, 0.0, 1.0]);
    let gid = LayerId(s.execute("layer.new.artboardFromLayers", json!({"name": "Board"})).unwrap()["layer"].as_u64().unwrap());
    s.execute("layer.artboard.set", json!({"width": 40, "height": 40})).unwrap();
    s.execute("layer.select", json!({"layer": gid.0})).unwrap();
    gid
}

fn red_at(s: &Session, x: i32, y: i32) -> bool {
    let px = photocraft_compose::render(doc(s), Rect::new(x, y, x + 1, y + 1)).px[0];
    px[0] > 0.99 && px[1] < 0.01 && px[3] > 0.99
}

#[test]
fn duplicate_artboard_lands_beside_the_original_with_its_contents() {
    // #1531: the copy sat exactly on the original, so nothing visible happened.
    let mut s = session(8);
    let gid = board_with_square(&mut s);
    let size = doc(&s).size;
    let r = s.execute("layer.duplicate", json!({})).unwrap();
    let copy = r["layer"].as_u64().unwrap();
    assert_eq!(board_rect(&s, gid.0), Rect::new(10, 10, 50, 50), "the original stays");
    // Photoshop: right of the original with the artboard gap, the canvas growing to show it.
    assert_eq!(board_rect(&s, copy), Rect::new(150, 10, 190, 50));
    assert!(doc(&s).size.width >= 190, "{:?}", doc(&s).size);
    assert!(red_at(&s, 155, 15), "the copy's contents moved with it");
    assert!(red_at(&s, 15, 15), "the original's contents stay");
    assert_eq!(doc(&s).layer(LayerId(copy)).unwrap().name, "Board copy");
    // Again from the original: the spot beside it is taken, so the next free one.
    s.execute("layer.select", json!({"layer": gid.0})).unwrap();
    let copy2 = s.execute("layer.duplicate", json!({})).unwrap()["layer"].as_u64().unwrap();
    assert_eq!(board_rect(&s, copy2), Rect::new(290, 10, 330, 50));
    // One undo step each: the copy and the canvas growth go together.
    s.undo();
    s.undo();
    assert_eq!(doc(&s).artboards().len(), 1);
    assert_eq!(doc(&s).size, size);
    // `inPlace` (the Move tool's ⌥-drag) keeps the copy on the original.
    s.execute("layer.select", json!({"layer": gid.0})).unwrap();
    let copy3 = s.execute("layer.duplicate", json!({"inPlace": true})).unwrap()["layer"].as_u64().unwrap();
    assert_eq!(board_rect(&s, copy3), Rect::new(10, 10, 50, 50));
}

#[test]
fn duplicating_several_artboards_places_each_copy_in_a_free_spot() {
    let mut s = session(8);
    let a = board_with_square(&mut s);
    let b = s.execute("layer.new.artboard", json!({"rect": [10, 100, 40, 40]})).unwrap()["layer"].as_u64().unwrap();
    s.execute("layer.select", json!({"layer": a.0})).unwrap();
    s.execute("layer.select", json!({"layer": b, "mode": "add"})).unwrap();
    let r = s.execute("layer.duplicate", json!({})).unwrap();
    assert_eq!(r["layers"].as_array().unwrap().len(), 2);
    let boards: Vec<Rect> = doc(&s).artboards().iter().map(|b| b.2.rect).collect();
    assert_eq!(boards.len(), 4);
    for (i, x) in boards.iter().enumerate() {
        for y in &boards[i + 1..] {
            assert!(x.intersect(y).is_empty(), "{x:?} overlaps {y:?}");
        }
    }
    let canvas = doc(&s).bounds();
    assert!(boards.iter().all(|r| r.intersect(&canvas) == *r), "{boards:?} in {canvas:?}");
    // One undo step.
    s.undo();
    assert_eq!(doc(&s).artboards().len(), 2);
}

#[test]
fn moving_an_artboard_past_the_canvas_grows_it() {
    // #1531: a board dragged right of the canvas was cut off at the old edge.
    let mut s = session(8);
    let gid = board_with_square(&mut s);
    s.execute("layer.translate", json!({"dx": 200, "dy": 60})).unwrap();
    assert_eq!(board_rect(&s, gid.0), Rect::new(210, 70, 250, 110));
    assert_eq!((doc(&s).size.width, doc(&s).size.height), (250, 110));
    assert!(red_at(&s, 215, 75));
    s.undo();
    assert_eq!((doc(&s).size.width, doc(&s).size.height), (100, 80));
}

#[test]
fn moving_an_artboard_past_the_canvas_keeps_its_shapes_whole() {
    // #2389: a shape's pixels are cut at the canvas; the board moved before the canvas grew,
    // so its shape rendered against the old edge and stayed missing or clipped.
    let shape_board = |s: &mut Session| {
        s.execute("file.new", json!({"width": 100, "height": 100, "background": "transparent"})).unwrap();
        let sh = s.execute("shape.create", json!({"kind": "rect", "rect": [10, 20, 30, 30], "fill": "#ff0000"})).unwrap()["layer"].as_u64().unwrap();
        s.execute("layer.new.artboardFromLayers", json!({})).unwrap();
        sh
    };
    for (cmd, p, size, red, not_red) in [
        ("layer.artboard.set", json!({"x": 120}), (150, 100), [(121, 21), (130, 30), (149, 49)], None),
        ("layer.translate", json!({"dx": 110, "dy": 0}), (150, 100), [(121, 21), (130, 30), (149, 49)], None),
        ("layer.artboard.set", json!({"y": 120}), (100, 150), [(11, 121), (20, 130), (39, 149)], None),
        // Partly past the edge: the whole shape, not just the part inside the old canvas.
        ("layer.artboard.set", json!({"x": 90}), (120, 100), [(91, 21), (95, 30), (110, 30)], Some((15, 30))),
    ] {
        let mut s = Session::new();
        let sh = shape_board(&mut s);
        s.execute(cmd, p.clone()).unwrap();
        assert_eq!((doc(&s).size.width, doc(&s).size.height), size, "{cmd} {p}");
        for (x, y) in red {
            assert!(red_at(&s, x, y), "{cmd} {p}: ({x}, {y})");
        }
        if let Some((x, y)) = not_red {
            assert!(!red_at(&s, x, y), "{cmd} {p}: ({x}, {y}) left behind");
        }
        let info = s.execute("shape.info", json!({"layer": sh})).unwrap();
        assert_eq!(info["bounds"], json!([red[0].0 - 1, red[0].1 - 1, 30, 30]), "{cmd} {p}");
        // One undo step back to the original board; redo brings the whole shape back.
        s.undo();
        assert_eq!((doc(&s).size.width, doc(&s).size.height), (100, 100));
        assert!(red_at(&s, 20, 30), "{cmd} {p}: undo");
        s.redo();
        assert_eq!((doc(&s).size.width, doc(&s).size.height), size);
        for (x, y) in red {
            assert!(red_at(&s, x, y), "{cmd} {p}: redo ({x}, {y})");
        }
    }
}

/// A type layer as a PSD import leaves it: Photoshop's rendering as the cache (deliberately not
/// what PhotoCraft would draw, a solid block, so a re-render shows) and Photoshop's `TySh`.
fn add_imported_type(s: &mut Session, block: Rect) -> LayerId {
    s.edit("type", |doc, active| {
        let mut t = photocraft_doc::TextLayer { text: "Hello!".into(), font_family: "No Such Font".into(), size_pt: 26.0, ..Default::default() };
        t.transform = photocraft_geom::Affine::translate(f64::from(block.x0), f64::from(block.y1));
        t.psd_raw = Some(std::sync::Arc::new(photocraft_text::psd::build_tysh(&t, 72.0, None)));
        let fmt = doc.pixel_format();
        let mut cache = photocraft_raster::Surface::new(fmt);
        cache.fill_rect(block, &photocraft_raster::from_rgba(&fmt, [0.4, 0.2, 0.6, 1.0]));
        t.cache = Some(cache);
        let id = doc.insert_above(None, Layer::new("Hello!", LayerContent::Text(t)));
        *active = Some(id);
        Ok(id)
    })
    .unwrap()
}

fn type_layer(doc: &Document, id: LayerId) -> photocraft_doc::TextLayer {
    match &doc.layer(id).unwrap().content {
        LayerContent::Text(t) => t.clone(),
        other => panic!("not a type layer: {}", other.kind_name()),
    }
}

#[test]
fn artboard_export_keeps_imported_type_as_rendered() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let text = add_imported_type(&mut s, Rect::new(30, 20, 60, 30));
        s.execute("layer.new.artboard", json!({"rect": [20, 10, 50, 40]})).unwrap();
        let board = doc(&s).artboards()[0].0;
        s.edit("into board", |doc, _| {
            let l = doc.remove(text).unwrap();
            let Some(LayerContent::Group(g)) = doc.layer_mut(board).map(|b| &mut b.content) else { panic!("artboard group") };
            g.children.push(l);
            Ok(())
        })
        .unwrap();
        let one = artboard_document(doc(&s), board).unwrap().unwrap();
        let t = type_layer(&one, text);
        // The same pixels, moved to the board's origin: not re-rendered.
        let c = t.cache.as_ref().unwrap();
        assert_eq!(c.content_bounds(), Rect::new(10, 10, 40, 20), "depth {depth}");
        let px = photocraft_raster::to_rgba(&c.format(), &c.pixel(15, 15));
        assert!((px[0] - 0.4).abs() < 0.01 && (px[2] - 0.6).abs() < 0.01, "depth {depth}: {px:?}");
        let tysh = photocraft_text::psd::parse_tysh(t.psd_raw.as_deref().unwrap()).unwrap();
        assert_eq!((tysh.transform.m[4], tysh.transform.m[5]), (10.0, 20.0), "the PSD position moves with it");
    }
}

#[test]
fn moving_imported_type_shifts_it_without_re_rendering() {
    let mut s = session(8);
    let text = add_imported_type(&mut s, Rect::new(30, 20, 60, 30));
    let before = type_layer(doc(&s), text);
    s.execute("layer.translate", json!({"layer": text.0, "dx": 5, "dy": -3})).unwrap();
    let after = type_layer(doc(&s), text);
    assert_eq!(after.cache.as_ref().unwrap().content_bounds(), Rect::new(35, 17, 65, 27));
    assert_eq!(after.transform.m[4..], [35.0, 27.0]);
    let (old, new) = (before.psd_raw.unwrap(), after.psd_raw.unwrap());
    assert_eq!((&new[..34], &new[50..]), (&old[..34], &old[50..]), "only the TySh translation changes");
    s.undo();
    assert_eq!(type_layer(doc(&s), text).cache, before.cache);
    // Type with nothing rendered yet is rendered after the move, as before.
    s.edit("drop cache", |doc, _| {
        if let Some(LayerContent::Text(t)) = doc.layer_mut(text).map(|l| &mut l.content) {
            t.cache = None;
        }
        Ok(())
    })
    .unwrap();
    s.execute("layer.translate", json!({"layer": text.0, "dx": 1, "dy": 0})).unwrap();
    assert!(type_layer(doc(&s), text).cache.is_some());
}

#[test]
fn moving_an_artboard_left_of_or_above_the_origin_grows_the_canvas_there() {
    // Photoshop grows an artboard document in every direction; the canvas only grew right and
    // down, so a board moved past the origin stayed partly off-canvas. Now everything shifts
    // right/down (boards, contents, guides) and the canvas grows, in the move's undo step.
    for (cmd, p) in [("layer.translate", json!({"dx": -60, "dy": -30})), ("layer.artboard.set", json!({"x": -50, "y": -20}))] {
        let mut s = session(8);
        let gid = board_with_square(&mut s);
        let other = s.execute("layer.new.artboard", json!({"rect": [60, 10, 30, 30]})).unwrap()["layer"].as_u64().unwrap();
        s.edit("guide", |doc, _| {
            doc.guides.vertical.push(70.0);
            Ok(())
        })
        .unwrap();
        s.execute("layer.select", json!({"layer": gid.0})).unwrap();
        s.execute(cmd, p.clone()).unwrap();
        assert_eq!(board_rect(&s, gid.0), Rect::new(0, 0, 40, 40), "{cmd}");
        assert_eq!(board_rect(&s, other), Rect::new(110, 30, 140, 60), "{cmd}: the other board shifts too");
        assert_eq!((doc(&s).size.width, doc(&s).size.height), (150, 100), "{cmd}");
        assert_eq!(doc(&s).guides.vertical, vec![120.0], "{cmd}");
        assert!(red_at(&s, 0, 0) && red_at(&s, 19, 19) && !red_at(&s, 20, 20), "{cmd}: the contents moved with the board");
        // One undo step back.
        s.undo();
        assert_eq!(board_rect(&s, gid.0), Rect::new(10, 10, 50, 50), "{cmd}");
        assert_eq!(board_rect(&s, other), Rect::new(60, 10, 90, 40), "{cmd}");
        assert_eq!((doc(&s).size.width, doc(&s).size.height), (100, 80), "{cmd}");
        assert_eq!(doc(&s).guides.vertical, vec![70.0], "{cmd}");
    }
}

#[test]
fn add_beside_places_a_board_on_each_side_and_alt_duplicates_contents() {
    // #3033: the Artboard tool's "+" buttons. Board 40×40 at (10, 10) holding a red square.
    let mut s = session(8);
    let gid = board_with_square(&mut s);
    let r = s.execute("layer.artboard.addBeside", json!({"side": "right"})).unwrap();
    assert_eq!(r["rect"], json!([150, 10, 40, 40]), "same size, a gap to the right");
    assert!(!red_at(&s, 155, 15), "a plain \"+\" adds an empty board");
    let r = s.execute("layer.artboard.addBeside", json!({"layer": gid.0, "side": "bottom", "duplicate": true})).unwrap();
    assert_eq!(r["rect"], json!([10, 150, 40, 40]));
    assert!(red_at(&s, 15, 155), "⌥: the copy brings the contents");
    // Again on the right: that spot is taken, so the next free one.
    let r = s.execute("layer.artboard.addBeside", json!({"layer": gid.0})).unwrap();
    assert_eq!(r["rect"], json!([290, 10, 40, 40]));
    // Left of x = 10 is past the canvas origin: everything shifts right by 130 to make room.
    let before = doc(&s).size;
    let r = s.execute("layer.artboard.addBeside", json!({"layer": gid.0, "side": "left"})).unwrap();
    assert_eq!(r["rect"], json!([0, 10, 40, 40]));
    assert_eq!(board_rect(&s, gid.0), Rect::new(140, 10, 180, 50));
    assert!(red_at(&s, 145, 15), "the original's contents moved with the shift");
    assert_eq!(doc(&s).size.width, before.width + 130);
    // One undo step.
    s.undo();
    assert_eq!(board_rect(&s, gid.0), Rect::new(10, 10, 50, 50));
    assert_eq!(doc(&s).size, before);
    assert!(s.execute("layer.artboard.addBeside", json!({"side": "diagonal"})).is_err());
}
