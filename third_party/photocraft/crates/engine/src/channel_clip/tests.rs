use super::*;
use photocraft_color::ColorMode;
use photocraft_geom::Rect;

const W: i32 = 40;
const H: i32 = 20;

/// A white document whose left half is #336699: red 0.2, green 0.4, blue 0.6 there.
fn session(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": W, "height": H, "depth": depth})).unwrap();
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": W / 2, "height": H})).unwrap();
    s.execute("edit.fill", json!({"color": "#336699"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s
}

fn px(s: &Session, x: i32, y: i32) -> Vec<f32> {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().pixel(x, y)
}

fn layers(s: &Session) -> usize {
    s.active().unwrap().doc.walk().len()
}

fn near(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
}

#[test]
fn copy_red_paste_into_blue() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let (left, right) = (px(&s, 5, 5), px(&s, 30, 5));
        s.execute("channel.target", json!({"channel": "red"})).unwrap();
        s.execute("edit.copy", json!({})).unwrap();
        let clip = s.clipboard.as_ref().unwrap();
        assert_eq!(clip.surface.format().mode, ColorMode::Grayscale, "depth {depth}: a channel copies as grayscale");
        assert_eq!(clip.bounds, Rect::new(0, 0, W, H), "no selection: the whole channel");
        let n = layers(&s);
        s.execute("channel.target", json!({"channel": "blue"})).unwrap();
        s.execute("edit.paste", json!({})).unwrap();
        assert_eq!(layers(&s), n, "depth {depth}: no new layer");
        assert!(near(&px(&s, 5, 5), &[left[0], left[1], left[0], 1.0]), "depth {depth}: blue = old red, {:?}", px(&s, 5, 5));
        assert!(near(&px(&s, 30, 5), &right));
        assert!(s.active().unwrap().doc.selection.is_none());
        s.undo();
        assert!(near(&px(&s, 5, 5), &left), "depth {depth}: one undoable step");
    }
}

#[test]
fn alpha_channels_copy_and_paste_into_colour_channels() {
    let mut s = session(8);
    // Alpha 1: white on the top half.
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": W, "height": H / 2})).unwrap();
    s.execute("channel.new", json!({"fill": "selection"})).unwrap();
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let clip = &s.clipboard.as_ref().unwrap().surface;
    assert_eq!((clip.pixel(30, 2), clip.pixel(30, 15)), (vec![1.0, 1.0], vec![0.0, 1.0]), "the alpha channel's values");
    s.execute("channel.target", json!({"channel": "green"})).unwrap();
    s.execute("edit.paste", json!({})).unwrap();
    assert!(near(&px(&s, 30, 2), &[1.0, 1.0, 1.0, 1.0]));
    assert!(near(&px(&s, 30, 15), &[1.0, 0.0, 1.0, 1.0]), "green = Alpha 1, {:?}", px(&s, 30, 15));
    // With the composite targeted, Paste still makes a layer (grey, from the grayscale clip).
    s.execute("channel.target", json!({"channel": "composite"})).unwrap();
    let n = layers(&s);
    s.execute("edit.paste", json!({})).unwrap();
    assert_eq!(layers(&s), n + 1);
    assert!(near(&px(&s, 30, 15), &[0.0, 0.0, 0.0, 1.0]));
}

#[test]
fn copy_an_alpha_channel_without_a_pixel_layer() {
    let mut s = session(8);
    s.execute("channel.new", json!({"fill": "white"})).unwrap();
    s.edit("no pixel layer", |_, active| {
        *active = None;
        Ok(())
    })
    .unwrap();
    assert!(s.is_enabled("edit.copy"), "an alpha channel needs no pixel layer");
    s.execute("edit.copy", json!({})).unwrap();
    assert_eq!(s.clipboard.as_ref().unwrap().surface.pixel(3, 3), vec![1.0, 1.0]);
}

#[test]
fn paste_into_a_colour_channel_keeps_to_the_selection() {
    let mut s = session(16);
    // Paste Into centres the paste on the selection: copy the same area so it lands in place.
    s.execute("select.rect", json!({"x": 0, "y": 0, "width": 10, "height": H})).unwrap();
    s.execute("channel.target", json!({"channel": "red"})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    s.execute("channel.target", json!({"channel": "green"})).unwrap();
    let n = layers(&s);
    s.execute("edit.pasteSpecial.pasteInto", json!({})).unwrap();
    assert_eq!(layers(&s), n, "no new layer or mask");
    let (inside, outside) = (px(&s, 5, 5), px(&s, 15, 5));
    assert!(near(&inside, &[0.2, 0.2, 0.6, 1.0]), "green = red inside, {inside:?}");
    assert!(near(&outside, &[0.2, 0.4, 0.6, 1.0]), "untouched outside, {outside:?}");
}

#[test]
fn copying_and_pasting_channels_fail_cleanly() {
    let mut s = session(8);
    s.execute("channel.target", json!({"channel": "red"})).unwrap();
    // An empty selection copies nothing and leaves the clipboard alone.
    s.edit("empty selection", |doc, _| {
        doc.selection = Some(Surface::new(photocraft_color::PixelFormat::GRAY8));
        Ok(())
    })
    .unwrap();
    assert!(s.execute("edit.copy", json!({})).is_err());
    assert!(s.clipboard.is_none());
    // A pixel-locked layer refuses the paste and keeps its pixels.
    s.execute("select.deselect", json!({})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    let id = s.active().unwrap().active_layer.unwrap();
    s.edit("lock", |doc, _| {
        doc.layer_mut(id).unwrap().locks.pixels = true;
        Ok(())
    })
    .unwrap();
    let before = px(&s, 5, 5);
    s.execute("channel.target", json!({"channel": "blue"})).unwrap();
    assert!(s.execute("edit.paste", json!({})).is_err());
    assert_eq!(px(&s, 5, 5), before);
}

#[test]
fn cut_a_colour_channel_clears_only_that_channel() {
    let mut s = session(8);
    let (left, right) = (px(&s, 5, 5), px(&s, 30, 5));
    s.tools.background = [0.8, 0.1, 0.9, 1.0];
    s.execute("channel.target", json!({"channel": "red"})).unwrap();
    s.execute("select.rect", json!({"x": 2, "y": 2, "width": 10, "height": 10})).unwrap();
    s.execute("edit.cut", json!({})).unwrap();
    let clip = s.clipboard.as_ref().unwrap();
    assert_eq!(clip.surface.format().mode, ColorMode::Grayscale, "the red channel copies as grayscale");
    assert!(near(&clip.surface.pixel(5, 5), &[left[0], 1.0]), "the clip holds the old red, {:?}", clip.surface.pixel(5, 5));
    // Red takes the background colour's red; green, blue and transparency stay.
    assert!(near(&px(&s, 5, 5), &[0.8, left[1], left[2], 1.0]), "{:?}", px(&s, 5, 5));
    assert!(near(&px(&s, 15, 5), &left), "outside the selection");
    assert!(near(&px(&s, 30, 5), &right));
    s.undo();
    assert!(near(&px(&s, 5, 5), &left), "one undoable step");
}

#[test]
fn clearing_a_channel_reports_allocation_failure_without_editing() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        let id = s.active().unwrap().active_layer.unwrap();
        let before = s.active().unwrap().doc.clone();
        let history = s.active().unwrap().history.entries();
        crate::allocation::fail_next_for_test();
        let result = s.edit("Clear", |doc, _| crate::channel_cmds::clear_color_channel(doc, id, 0, [1.0; 4]));
        let _ = crate::allocation::capacity::<u8>(0, "reset failpoint");
        assert!(result.is_err());
        assert_eq!(*s.active().unwrap().doc, *before);
        assert_eq!(s.active().unwrap().history.entries(), history);
    }
}

#[test]
fn channel_copy_allocation_failure_keeps_clipboard_and_document() {
    for depth in [8, 16, 32] {
        let mut s = session(depth);
        s.execute("edit.copy", json!({})).unwrap();
        s.execute("channel.target", json!({"channel": "red"})).unwrap();
        let before = s.active().unwrap().doc.clone();
        let clip = s.clipboard.as_ref().unwrap().surface.clone();
        let history = s.active().unwrap().history.entries();
        crate::allocation::fail_next_for_test();
        let result = copy(&mut s);
        let _ = crate::allocation::capacity::<u8>(0, "reset failpoint");
        assert!(result.is_err());
        assert_eq!(*s.active().unwrap().doc, *before);
        assert_eq!(s.active().unwrap().history.entries(), history);
        assert_eq!(s.clipboard.as_ref().unwrap().surface, clip);
    }
}
