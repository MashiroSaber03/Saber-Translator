use super::*;
use crate::Session;
use photocraft_doc::vector::LiveShape;
use serde_json::json;

const W: u32 = 64;
const H: u32 = 40;

fn doc(s: &Session) -> &Document {
    &s.active().unwrap().doc
}

/// Paints the active raster layer with `f(x, y)` (straight RGBA).
fn paint(s: &mut Session, f: impl Fn(i32, i32) -> [f32; 4]) {
    s.edit("setup", |doc, active| {
        let b = doc.bounds();
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        let fmt = surf.format();
        let mut data = Vec::new();
        for y in b.y0..b.y1 {
            for x in b.x0..b.x1 {
                data.extend(photocraft_raster::from_rgba(&fmt, f(x, y)));
            }
        }
        surf.write_region(b, &data);
        surf.prune();
        Ok(())
    })
    .unwrap();
}

/// An asymmetric document: a gradient Background, a type layer, a shape layer, a raster layer
/// with a vector mask and a smart object, plus a guide, a saved path and a slice.
fn mixed(depth: u64, mode: &str) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": W, "height": H, "depth": depth, "mode": mode})).unwrap();
    paint(&mut s, |x, y| [x as f32 / W as f32, y as f32 / H as f32, 0.3, 1.0]);
    s.execute("type.create", json!({"x": 4, "y": 16, "text": "Fj", "size": 14, "color": "#102080"})).unwrap();
    s.execute("shape.create", json!({"kind": "rect", "rect": [40, 4, 18, 9], "fill": "#ff8800"})).unwrap();
    s.execute("shape.create", json!({"kind": "ellipse", "rect": [6, 24, 12, 10], "fill": "#00aa44"})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s, |x, y| if (28..60).contains(&x) && (18..38).contains(&y) { [0.9, 0.1, 0.6, 1.0] } else { [0.0; 4] });
    s.execute("path.set", json!({"path": {"subpaths": [{"knots": [[30, 20], [58, 22], [36, 37]]}]}})).unwrap();
    s.execute("layer.vectorMask.currentPath", json!({})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    paint(&mut s, |x, y| if (20..27).contains(&x) && (2..12).contains(&y) { [0.1, 0.8, 0.9, 1.0] } else { [0.0; 4] });
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    s.execute("view.newGuide", json!({"orientation": "vertical", "position": 10})).unwrap();
    s.execute("view.newGuide", json!({"orientation": "horizontal", "position": 5})).unwrap();
    s.execute("path.set", json!({"name": "Saved", "path": {"subpaths": [{"knots": [[2, 2], [12, 2], [12, 8]]}]}})).unwrap();
    s.execute("slice.new", json!({"rect": [1, 2, 10, 6]})).unwrap();
    s
}

fn command(t: Turn) -> &'static str {
    match t {
        Turn::FlipHorizontal => "image.imageRotation.flipCanvasHorizontal",
        Turn::FlipVertical => "image.imageRotation.flipCanvasVertical",
        Turn::Rotate180 => "image.imageRotation.180",
        Turn::Cw90 => "image.imageRotation.90cw",
        Turn::Ccw90 => "image.imageRotation.90ccw",
    }
}

const TURNS: [Turn; 5] = [Turn::FlipHorizontal, Turn::FlipVertical, Turn::Rotate180, Turn::Cw90, Turn::Ccw90];

/// The flattened original, remapped by the turn's pixel map.
fn expected(before: &photocraft_compose::Buffer, t: Turn, nw: u32) -> Vec<[f32; 4]> {
    let map = t.pixel_map(W as i32, H as i32);
    let mut out = vec![[0.0f32; 4]; before.px.len()];
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let (nx, ny) = map(x, y);
            out[ny as usize * nw as usize + nx as usize] = before.px[y as usize * W as usize + x as usize];
        }
    }
    out
}

/// (mean absolute difference, fraction of pixels differing by more than 0.2).
fn diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    let mut big = 0usize;
    for (p, q) in a.iter().zip(b) {
        let d = (0..4).map(|c| (p[c] * p[3] - q[c] * q[3]).abs()).fold(0.0f32, f32::max);
        sum += d;
        big += usize::from(d > 0.2);
    }
    (sum / a.len() as f32, big as f32 / a.len() as f32)
}

fn check_turn(depth: u64, mode: &str, t: Turn) {
    let mut s = mixed(depth, mode);
    let before = photocraft_compose::flatten(doc(&s));
    s.execute(command(t), json!({})).unwrap();
    let nw = doc(&s).size.width;
    let want = expected(&before, t, nw);
    // Caches are remapped exactly; vector masks re-rasterize from their mapped path.
    let got = photocraft_compose::flatten(doc(&s));
    let (mean, big) = diff(&got.px, &want);
    assert!(mean < 1.0 / 255.0 && big < 0.01, "{t:?} {depth} {mode}: mean {mean} big {big}");
    // Re-rendering every type, shape and smart object from its *geometry* gives the same image,
    // so the transforms (not only the cached pixels) moved.
    let mut d = doc(&s).clone();
    refresh(&mut d, Refresh::All);
    let (mean, big) = diff(&photocraft_compose::flatten(&d).px, &want);
    assert!(mean < 2.0 / 255.0 && big < 0.01, "{t:?} {depth} {mode} re-rendered: mean {mean} big {big}");
    // Undo restores the original, bit-exactly.
    s.execute("edit.undo", json!({})).unwrap();
    let undone = photocraft_compose::flatten(doc(&s));
    assert_eq!(undone.px.len(), before.px.len(), "{t:?} {depth} {mode}: undo changed the canvas size");
    let (mut n, mut max, mut first) = (0usize, 0.0f32, None);
    for (i, (p, q)) in undone.px.iter().zip(&before.px).enumerate() {
        if p != q {
            n += 1;
            max = (0..4).map(|c| (p[c] - q[c]).abs()).fold(max, f32::max);
            first.get_or_insert((i % W as usize, i / W as usize, *p, *q));
        }
    }
    assert!(n == 0, "{t:?} {depth} {mode}: undo left {n} pixels different (max {max}); first (x, y, undone, before): {first:?}");
}

#[test]
fn right_angle_turns_move_vectors_type_and_smart_objects() {
    for t in TURNS {
        check_turn(8, "rgb", t);
    }
}

#[test]
fn turns_at_every_depth_and_mode() {
    for (depth, mode) in [(16, "rgb"), (32, "rgb"), (8, "cmyk"), (16, "cmyk"), (8, "gray"), (16, "gray")] {
        check_turn(depth, mode, Turn::Cw90);
    }
    check_turn(32, "rgb", Turn::FlipHorizontal);
    check_turn(8, "cmyk", Turn::Ccw90);
}

#[test]
fn turns_map_guides_paths_slices_and_live_shapes() {
    let mut s = mixed(8, "rgb");
    s.execute("image.imageRotation.90cw", json!({})).unwrap();
    let d = doc(&s);
    assert_eq!((d.size.width, d.size.height), (H, W));
    // A vertical guide at x=10 becomes a horizontal one at y=10; a horizontal one at y=5 becomes
    // a vertical one at x = H − 5.
    assert_eq!(d.guides.horizontal, vec![10.0]);
    assert_eq!(d.guides.vertical, vec![f64::from(H - 5)]);
    // Slice [1,2 10×6] → x' = H − y: [H−8, 1, 6×10].
    let sl = &d.slices.list[0];
    assert_eq!(sl.rect, Rect::new(H as i32 - 8, 1, H as i32 - 2, 11));
    let saved = d.paths.iter().find(|p| p.name == "Saved").unwrap();
    let k = saved.path.subpaths[0].knots[1].anchor;
    assert!((k.x - (f64::from(H) - 2.0)).abs() < 1e-9 && (k.y - 12.0).abs() < 1e-9, "{k:?}");
    // The rectangle stays a live rectangle with the rotated box.
    let live: Vec<&LiveShape> = d
        .walk()
        .into_iter()
        .filter_map(|(_, _, l)| match &l.content {
            LayerContent::Shape(sh) => sh.live.as_ref(),
            _ => None,
        })
        .collect();
    assert!(
        live.iter()
            .any(|l| matches!(l, LiveShape::Rect { rect, .. } if rect.iter().zip([f64::from(H) - 13.0, 40.0, 9.0, 18.0]).all(|(a, b)| (a - b).abs() < 1e-9))),
        "{live:?}"
    );
    assert!(live.iter().any(|l| matches!(l, LiveShape::Ellipse { .. })));
    // Flip twice is the identity for the geometry.
    let before = doc(&s).clone();
    s.execute("image.imageRotation.flipCanvasVertical", json!({})).unwrap();
    s.execute("image.imageRotation.flipCanvasVertical", json!({})).unwrap();
    let after = doc(&s);
    assert_eq!(after.guides, before.guides);
    assert_eq!(after.slices.list[0].rect, before.slices.list[0].rect);
    assert_eq!(photocraft_compose::flatten(after).px, photocraft_compose::flatten(&before).px);
}

#[test]
fn gradient_fill_angle_follows_the_canvas() {
    let a = Turn::Cw90.affine(10.0, 20.0);
    // Up (90°) turns clockwise to the right (0°); flips mirror.
    assert_eq!(map_angle(&a, 90.0), 0.0);
    assert_eq!(map_angle(&Turn::FlipHorizontal.affine(10.0, 20.0), 0.0), 180.0);
    assert_eq!(map_angle(&Turn::FlipVertical.affine(10.0, 20.0), 30.0), -30.0);
    assert_eq!(map_angle(&Turn::Ccw90.affine(10.0, 20.0), 0.0), 90.0);
}

fn type_and_shape_origin(d: &Document) -> ([f64; 6], [f64; 2]) {
    let mut t = None;
    let mut k = None;
    for (_, _, l) in d.walk() {
        match &l.content {
            LayerContent::Text(x) => t = Some(x.transform.m),
            LayerContent::Shape(sh) if k.is_none() => k = Some([sh.path.subpaths[0].knots[0].anchor.x, sh.path.subpaths[0].knots[0].anchor.y]),
            _ => {}
        }
    }
    (t.unwrap(), k.unwrap())
}

#[test]
fn canvas_size_and_crop_move_vector_geometry() {
    let mut s = mixed(8, "rgb");
    let before = photocraft_compose::flatten(doc(&s));
    let (t0, k0) = type_and_shape_origin(doc(&s));
    s.execute("image.canvasSize", json!({"width": 20, "height": 10, "relative": true, "anchor": "bottomRight", "extensionColor": "transparent"})).unwrap();
    let d = doc(&s);
    let (t1, k1) = type_and_shape_origin(d);
    assert_eq!((t1[4] - t0[4], t1[5] - t0[5]), (20.0, 10.0));
    assert_eq!((k1[0] - k0[0], k1[1] - k0[1]), (20.0, 10.0));
    assert_eq!(d.guides.vertical, vec![30.0]);
    assert_eq!(d.slices.list[0].rect, Rect::new(21, 12, 31, 18));
    // The old canvas shows unchanged at its new offset, also after re-rendering from geometry.
    let mut r = d.clone();
    refresh(&mut r, Refresh::All);
    for img in [photocraft_compose::flatten(d), photocraft_compose::flatten(&r)] {
        let nw = img.rect.width() as usize;
        let shifted: Vec<[f32; 4]> = (0..H as usize).flat_map(|y| img.px[(y + 10) * nw + 20..(y + 10) * nw + 20 + W as usize].to_vec()).collect();
        let (mean, big) = diff(&shifted, &before.px);
        assert!(mean < 2.0 / 255.0 && big < 0.01, "mean {mean} big {big}");
    }
    // Crop back: everything returns.
    s.execute("image.crop", json!({"x": 20, "y": 10, "width": W, "height": H})).unwrap();
    let (t2, k2) = type_and_shape_origin(doc(&s));
    assert_eq!((t2, k2), (t0, k0));
    let (mean, _) = diff(&photocraft_compose::flatten(doc(&s)).px, &before.px);
    assert!(mean < 1.0 / 255.0, "{mean}");
}

#[test]
fn image_size_scales_vector_geometry() {
    let mut s = mixed(8, "rgb");
    let (t0, k0) = type_and_shape_origin(doc(&s));
    s.execute("image.imageSize", json!({"width": W * 2, "height": H * 2})).unwrap();
    let d = doc(&s);
    let (t1, k1) = type_and_shape_origin(d);
    assert!((t1[0] - 2.0 * t0[0]).abs() < 1e-9 && (t1[4] - 2.0 * t0[4]).abs() < 1e-9, "{t0:?} {t1:?}");
    assert_eq!((k1[0], k1[1]), (2.0 * k0[0], 2.0 * k0[1]));
    assert_eq!(d.guides.vertical, vec![20.0]);
}

#[test]
fn arbitrary_rotation_moves_marks_and_unlinked_vector_masks() {
    let mut s = mixed(8, "rgb");
    // Unlink the vector mask: a canvas rotation still moves it.
    s.edit("unlink", |doc, _| {
        for l in doc.layers.iter_mut() {
            if let Some(vm) = &mut l.vector_mask {
                vm.linked = false;
            }
        }
        Ok(())
    })
    .unwrap();
    let before = doc(&s).clone();
    s.execute("image.rotation.arbitrary", json!({"angle": 90, "direction": "cw"})).unwrap();
    let d = doc(&s);
    let a = Turn::Cw90.affine(f64::from(W), f64::from(H));
    let vm = |d: &Document| d.walk().into_iter().find_map(|(_, _, l)| l.vector_mask.as_ref().map(|v| v.path.subpaths[0].knots[0].anchor)).unwrap();
    let (p, q) = (a.apply(vm(&before)), vm(d));
    assert!((p.x - q.x).abs() < 1e-6 && (p.y - q.y).abs() < 1e-6, "{p:?} {q:?}");
    assert_eq!(d.slices.list[0].rect, map_rect(&a, before.slices.list[0].rect));
    // 90° arbitrary == 90° clockwise for the composite.
    let mut s2 = mixed(8, "rgb");
    s2.execute("image.imageRotation.90cw", json!({})).unwrap();
    let (mean, big) = diff(&photocraft_compose::flatten(d).px, &photocraft_compose::flatten(doc(&s2)).px);
    assert!(mean < 3.0 / 255.0 && big < 0.02, "mean {mean} big {big}");
}

#[test]
fn turns_on_empty_and_tiny_documents_never_panic() {
    for (w, h) in [(1, 1), (1, 7), (5, 1)] {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h})).unwrap();
        for t in TURNS {
            s.execute(command(t), json!({"junk": [1, 2]})).unwrap();
        }
        s.execute("image.rotation.arbitrary", json!({"angle": 33})).unwrap();
    }
    let mut s = Session::new();
    for t in TURNS {
        assert!(s.execute(command(t), json!({})).is_err(), "no document");
    }
}
