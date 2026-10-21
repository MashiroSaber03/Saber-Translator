//! Public Affinity documents (`cargo xtask corpus --affinity`, pinned and sha256-verified) open
//! natively and look like Affinity's own picture of them: older corpus files embed a thumbnail,
//! while the #1606 samples include exported PNG references. Each file's mean difference (0–255 per
//! channel, flattened on white and compared at the smaller size) must stay under its ceiling, set
//! just above the difference measured when the importer landed.
#![cfg(feature = "corpus")]
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::PathBuf;

use photocraft_io::{ExportOptions, export, import};

/// (file, ceiling). Measured differences when the importer landed are in the comments.
const FILES: &[(&str, f64)] = &[
    ("vector-art/mexican-guy.af", 3e+00),                  // 1.92: brush strokes as plain strokes
    ("vector-art/mexican-man.af", 4e+00),                  // 2.74
    ("vector-art/mexican-woman.af", 4e+00),                // 2.24
    ("vector-art/playing-cards.af", 4e+00),                // 3.04: layer effects
    ("vector-art/beer.afdesign", 4e+00),                   // 2.91
    ("vector-art/cactus.afdesign", 2e+00),                 // 0.57
    ("vector-art/car.afdesign", 2e+00),                    // 0.39
    ("vector-art/flowers.afdesign", 3e+00),                // 2.09
    ("vector-art/lion-track.afdesign", 6e+00),             // 4.41
    ("afdesignload/color.afdesign", 1e+00),                // 0.00
    ("afdesignload/layer_mode.afdesign", 1e+00),           // 0.00
    ("afdesignload/layer_test.afdesign", 1e+00),           // 0.00
    ("afdesignload/margins.afdesign", 1e+00),              // 0.00
    ("afdesignload/raster_test.afdesign", 2e+00),          // 0.82: pixel layer
    ("afdesignload/revision_test.afdesign", 1e+00),        // 0.01
    ("afdesignload/shape_test.afdesign", 6e+00),           // 5.44 after importing the public cloud/heart geometry
    ("afdesignload/slice_test.afdesign", 1e+00),           // 0.02
    ("afdesignload/test_path.afdesign", 2e+00),            // 1.37
    ("jac21/SimpleLogo.afdesign", 2e+00),                  // 0.78
    ("jac21/SimpleLogoBanner.afdesign", 4e+00),            // 2.72: placed images in a Display P3 document
    ("asset-store-template/AssetStore.aftemplate", 1e+00), // 0.00: artboards
    ("affinity-samples/01-layer-effects.af", 11.0),        // 8.86–9.64 measured (fonts differ by machine): effects and live filters not imported
    ("affinity-samples/02-adjustment-layers.af", 38.0),    // 36.19–36.69 measured (fonts differ by machine): adjustment layers not imported
    ("affinity-samples/03-brush-strokes.af", 3.0),         // 1.39–1.93 measured (fonts differ by machine): pressure strokes approximated
    ("affinity-samples/04-master-page.af", 3.0),           // 1.49–1.51 measured (fonts differ by machine): master content not placed on pages
    ("affinity-samples/05-symbols.af", 21.5),              // 19.49–20.47 measured (fonts differ by machine): cached symbol and embedded pictures
    ("affinity-samples/06-special-shapes.af", 7.5),        // 5.68–6.42 measured (fonts differ by machine): special-shape geometry
    ("affinity-samples/07-transparency.af", 13.5),         // 11.61–12.40 measured (fonts differ by machine): transparency and bitmap fills
    ("affinity-samples/08-gradients.af", 2.0),             // 0.44–0.94 measured (fonts differ by machine): conical gradient currently maps to Angle
    ("affinity-samples/09-corners-stars.af", 2.5),         // 0.80–1.48 measured (fonts differ by machine): non-round corners and rounded stars
    ("affinity-samples/10-multiple-fills.af", 11.5),       // 9.50–10.29 measured (fonts differ by machine): only one fill/stroke per object
    ("affinity-samples/11-text.af", 4.5),                  // 2.66–3.41 measured (fonts differ by machine): outlined/scaled text and text fields
    ("affinity-samples/12-cmyk8.af", 16.0),                // 14.04–14.67 measured (fonts differ by machine): document profile not applied
    ("affinity-samples/12-grey8.af", 26.5),                // 24.85–25.47 measured (fonts differ by machine): grey document opened as RGB
    ("affinity-samples/12-lab16.af", 7.5),                 // 5.55–6.17 measured (fonts differ by machine): Lab profile and depth not preserved
    ("affinity-samples/12-rgb16.af", 2.5),                 // 0.47–1.09 measured (fonts differ by machine): image content remains close after conversion
    ("affinity-samples/12-rgb32.af", 87.5),                // 86.05–86.19 measured (fonts differ by machine): 32-bit RGB is reduced to 8-bit
    ("affinity-samples/12-rgb8.af", 2.5),                  // 0.47–1.10 measured (fonts differ by machine): RGB/8 reference
    ("affinity-samples/13-blend-modes.af", 12.0),          // 9.67–10.57 measured (fonts differ by machine): Affinity-only blend modes map to Normal
];

fn corpus(rel: &str) -> Vec<u8> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../corpus/affinity").join(rel);
    std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}: run `cargo xtask corpus --affinity`", p.display()))
}

/// Straight RGBA8 pixels flattened on white.
fn on_white(px: &[u8]) -> Vec<f64> {
    px.as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| {
            let a = f64::from(p[3]) / 255.0;
            [0, 1, 2].map(|i| f64::from(p[i]) * a + 255.0 * (1.0 - a))
        })
        .collect()
}

/// Resample `src` (`sw`×`sh`, 3 channels) to `dw`×`dh`: each target pixel averages the source
/// pixels it covers (at least one, so small documents scale up too).
fn shrink(src: &[f64], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f64> {
    let span = |d: usize, s: usize, n: usize| {
        let a = (d * s / n).min(s - 1);
        (a, ((d + 1) * s).div_ceil(n).clamp(a + 1, s))
    };
    let mut out = Vec::with_capacity(dw * dh * 3);
    for ty in 0..dh {
        let (y0, y1) = span(ty, sh, dh);
        for tx in 0..dw {
            let (x0, x1) = span(tx, sw, dw);
            let mut sum = [0.0f64; 3];
            for y in y0..y1 {
                for x in x0..x1 {
                    for (c, s) in sum.iter_mut().enumerate() {
                        *s += src[(y * sw + x) * 3 + c];
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as f64;
            out.extend(sum.map(|s| s / n));
        }
    }
    out
}

#[test]
fn public_affinity_documents_match_their_render_reference() {
    let mut failures = Vec::new();
    for (name, ceiling) in FILES {
        let bytes = corpus(name);
        let r = import(name, &bytes).unwrap();
        assert!(!r.preview_only, "{name} opened as its preview: {:?}", r.warnings);
        let reference = if name.starts_with("affinity-samples/") {
            let path = if *name == "affinity-samples/04-master-page.af" {
                "affinity-samples/04-master-page_1.png".to_string()
            } else {
                format!("{}.png", &name[..name.len() - 3])
            };
            photocraft_codecs::decode(&corpus(&path)).unwrap()
        } else {
            photocraft_codecs::decode(photocraft_affinity::preview(&bytes).unwrap().png).unwrap()
        };
        let (tw, th) = reference.dimensions();
        let png = export(&r.document, "x.png", &ExportOptions::default()).unwrap().bytes;
        let flat = photocraft_codecs::decode(&png).unwrap();
        let (fw, fh) = flat.dimensions();
        // Several pages sit side by side; the thumbnail shows the first.
        let fw_first = ((f64::from(fh) * f64::from(tw) / f64::from(th)).round() as u32).min(fw);
        let rgba = flat.to_rgba8();
        let cropped: Vec<u8> = rgba.chunks_exact(fw as usize * 4).flat_map(|row| row[..fw_first as usize * 4].to_vec()).collect();
        // Compare at the smaller of the two sizes, so neither picture is enlarged.
        let (cw, ch) = if fh < th { (fw_first, fh) } else { (tw, th) };
        let got = shrink(&on_white(&cropped), fw_first as usize, fh as usize, cw as usize, ch as usize);
        let want = shrink(&on_white(&reference.to_rgba8()), tw as usize, th as usize, cw as usize, ch as usize);
        let diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum::<f64>() / want.len() as f64;
        eprintln!("{name}: {diff:.2} (ceiling {ceiling})");
        if diff > *ceiling {
            failures.push(format!("{name}: {diff:.2} > {ceiling}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn issue_1606_special_shapes_match_the_exported_sample() {
    let name = "affinity-samples/06-special-shapes.af";
    let bytes = corpus(name);
    let imported = import(name, &bytes).unwrap();
    assert!(
        !imported.warnings.iter().any(|w| w.contains("unrecognized parametric shape")),
        "supported special-shape sample still reports a bounding-ellipse fallback: {:?}",
        imported.warnings
    );
    let rendered = photocraft_codecs::decode(&export(&imported.document, "x.png", &ExportOptions::default()).unwrap().bytes).unwrap();
    let reference = photocraft_codecs::decode(&corpus("affinity-samples/06-special-shapes.png")).unwrap();
    let (rw, rh) = reference.dimensions();
    let (ow, oh) = rendered.dimensions();
    let (w, h) = (rw.min(ow), rh.min(oh));
    let got = shrink(&on_white(&rendered.to_rgba8()), ow as usize, oh as usize, w as usize, h as usize);
    let want = shrink(&on_white(&reference.to_rgba8()), rw as usize, rh as usize, w as usize, h as usize);
    let diff = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum::<f64>() / want.len() as f64;
    eprintln!("{name} vs exported PNG: {diff:.2} ({}×{} vs {}×{})", ow, oh, rw, rh);
    assert!(diff <= 7.5, "special-shape sample exceeded its corpus ceiling: {diff:.2} > 7.5");
}
