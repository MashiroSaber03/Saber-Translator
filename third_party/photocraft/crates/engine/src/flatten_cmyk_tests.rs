//! #3778: Flatten Image, Merge Visible and Merge Down keep a CMYK document's ink values instead
//! of round-tripping them through RGB.

use photocraft_cms::synth::{CmykParams, cmyk_profile};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerContent, Size};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::json;

use crate::Session;

const W: i32 = 16;
const H: i32 = 8;

fn canvas() -> Rect {
    Rect::new(0, 0, W, H)
}

/// Ink values that vary per pixel and per channel, away from the 8-bit grid.
fn ink(x: i32, y: i32) -> [f32; 4] {
    let f = |k: i32| ((x * 7 + y * 13 + k * 29) % 97) as f32 / 96.0 * 0.9 + 0.0371;
    [f(0), f(1), f(2), f(3)]
}

/// A 16-bit CMYK document tagged with a synthetic profile (not the built-in coated CMYK), holding
/// just an opaque Background of varied ink.
fn cmyk_doc() -> Document {
    let mut d = Document::new("cmyk", Size::new(W as u32, H as u32), ColorMode::Cmyk, SampleType::U16);
    let profile =
        cmyk_profile(&CmykParams { description: "Flatten Test CMYK".into(), tvi: [0.26, 0.26, 0.26, 0.3], grid_a2b: 5, grid_b2a: 9, ..Default::default() });
    d.icc_profile = Some(profile.to_bytes());
    let mut bg = Surface::new(PixelFormat::new(ColorMode::Cmyk, SampleType::U16, false));
    for y in 0..H {
        for x in 0..W {
            bg.write_pixel(x, y, &ink(x, y));
        }
    }
    let mut layer = Layer::new("Background", LayerContent::Raster(bg));
    layer.locks.transparency = true;
    layer.locks.position = true;
    d.layers.push(layer);
    d
}

/// A layer above the Background: solid ink on the left half, half-transparent ink on a strip,
/// nothing on the rest.
fn upper_layer() -> Layer {
    let mut s = Surface::new(PixelFormat::new(ColorMode::Cmyk, SampleType::U16, true));
    s.fill_rect(Rect::new(0, 0, W / 2, H), &[0.8123, 0.1234, 0.4567, 0.0789, 1.0]);
    s.fill_rect(Rect::new(W / 2, 0, W / 2 + 2, H), &[0.25, 0.5, 0.75, 0.1, 0.5]);
    Layer::new("Layer 1", LayerContent::Raster(s))
}

fn session(doc: Document) -> Session {
    let mut s = Session::new();
    s.add_document(doc, None);
    s
}

/// The bottom layer's ink values (C, M, Y, K per pixel, without any alpha channel).
fn pixels(s: &Session) -> Vec<f32> {
    let surf = s.active().unwrap().doc.layers[0].surface().unwrap();
    let n = surf.channels();
    surf.read_region(canvas()).chunks_exact(n).flat_map(|p| p[..4].to_vec()).collect()
}

fn bottom_format(s: &Session) -> PixelFormat {
    s.active().unwrap().doc.layers[0].surface().unwrap().format()
}

#[test]
fn flattening_a_lone_cmyk_background_keeps_its_pixels() {
    let doc = cmyk_doc();
    let before = doc.layers[0].surface().unwrap().read_region(canvas());
    let mut s = session(doc);
    s.execute("layer.flattenImage", json!({})).unwrap();
    let fmt = bottom_format(&s);
    assert_eq!((fmt.mode, fmt.sample), (ColorMode::Cmyk, SampleType::U16), "16-bit CMYK stays 16-bit CMYK");
    let moved = pixels(&s).iter().zip(&before).filter(|(a, b)| a != b).count();
    assert_eq!(moved, 0, "a single Background must flatten to the same ink values");
}

/// What a Normal 100 % layer over the Background gives, worked out in ink values: the solid part
/// replaces the ink, the half-transparent strip mixes half and half, the rest shows the Background.
fn expected_two_layers() -> Vec<f32> {
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let below = ink(x, y);
            let px = if x < W / 2 {
                [0.8123, 0.1234, 0.4567, 0.0789]
            } else if x < W / 2 + 2 {
                let top = [0.25, 0.5, 0.75, 0.1];
                [0, 1, 2, 3].map(|c| top[c] * 0.5 + below[c] * 0.5)
            } else {
                below
            };
            out.extend_from_slice(&px);
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: pixel count");
    // One 16-bit code (the stored value is quantised to 1/65535).
    let worst = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst <= 1.0 / 65535.0 + 1e-6, "{what}: ink values moved by up to {worst}");
}

#[test]
fn flattening_normal_cmyk_layers_composites_in_ink_values() {
    let mut doc = cmyk_doc();
    doc.layers.push(upper_layer());
    let mut s = session(doc);
    s.execute("layer.flattenImage", json!({})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), 1);
    assert_eq!(bottom_format(&s).sample, SampleType::U16);
    assert_close(&pixels(&s), &expected_two_layers(), "Flatten Image");
}

#[test]
fn merge_visible_and_merge_down_keep_cmyk_ink_values() {
    for cmd in ["layer.mergeVisible", "layer.mergeDown"] {
        let mut doc = cmyk_doc();
        let upper = upper_layer();
        let id = upper.id;
        doc.layers.push(upper);
        let mut s = session(doc);
        s.execute(cmd, json!({"layer": id.0})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.len(), 1, "{cmd}");
        assert_eq!(bottom_format(&s).sample, SampleType::U16, "{cmd}");
        assert_close(&pixels(&s), &expected_two_layers(), cmd);
    }
}

/// A stack the ink-value path can't take (a Multiply layer) still goes through the compositor,
/// but separates back through the document's profile, the one it read the ink through, so the
/// round trip lands near the original ink instead of on the built-in profile's separation.
#[test]
fn compositor_flatten_separates_through_the_documents_profile() {
    let mut doc = cmyk_doc();
    // Light inks, inside the printable gamut, so a separation can give them back.
    let bg = doc.layers[0].surface_mut().unwrap();
    for y in 0..H {
        for x in 0..W {
            bg.write_pixel(x, y, &ink(x, y).map(|v| v * 0.4));
        }
    }
    let before = doc.layers[0].surface().unwrap().read_region(canvas());
    let mut empty = Layer::new("Layer 1", LayerContent::Raster(Surface::new(PixelFormat::new(ColorMode::Cmyk, SampleType::U16, true))));
    empty.blend = photocraft_color::blend::BlendMode::Multiply;
    doc.layers.push(empty);
    let mut s = session(doc);
    s.execute("layer.flattenImage", json!({})).unwrap();
    // Compare what the inks print as (a CMYK colour has many separations): RGB through the profile.
    let d = &s.active().unwrap().doc;
    let space = photocraft_compose::cmyk_space(d);
    let shown = |v: &[f32]| -> Vec<[f32; 3]> {
        photocraft_color::convert::with_cmyk_space(space.as_ref(), || v.as_chunks::<4>().0.iter().map(|p| photocraft_color::convert::cmyk_to_rgb(*p)).collect())
    };
    let (a, b) = (shown(&pixels(&s)), shown(&before));
    let diffs: Vec<f32> = a.iter().zip(&b).flat_map(|(p, q)| (0..3).map(move |c| (p[c] - q[c]).abs())).collect();
    let mean = diffs.iter().sum::<f32>() / diffs.len() as f32;
    // The synthetic profile's small tables don't round-trip exactly; the built-in profile's
    // separation lands about 0.08 off on average.
    assert!(mean < 0.03, "flattening shifted the printed colour by {mean} on average");
}
