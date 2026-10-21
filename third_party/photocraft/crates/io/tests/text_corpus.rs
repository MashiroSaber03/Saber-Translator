//! Type layers in the real-file corpus (`corpus/psd`, gitignored; feature `corpus`, fetched by
//! `cargo xtask corpus --all`; a missing corpus fails).
//!
//! For every text layer: the TySh/EngineData model must parse (text, fonts, sizes, colours,
//! runs), the `TySh` must survive a PSD round trip byte-for-byte, and our engine's re-render is
//! compared with Photoshop's cached pixels. The fonts usually differ, so the comparison checks
//! geometry (ink bounds and alpha overlap) and is reported; the assertions are loose.

#[cfg(feature = "corpus")]
use std::path::{Path, PathBuf};

#[cfg(feature = "corpus")]
use photocraft_doc::Layer;
use photocraft_doc::LayerContent;
#[cfg(feature = "corpus")]
use photocraft_geom::Rect;
#[cfg(feature = "corpus")]
use photocraft_raster::Surface;

#[cfg(feature = "corpus")]
fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            collect(&p, out);
        } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb")) {
            out.push(p);
        }
    }
}

#[cfg(feature = "corpus")]
fn walk<'a>(layers: &'a [Layer], out: &mut Vec<&'a Layer>) {
    for l in layers {
        out.push(l);
        if let Some(ch) = l.children() {
            walk(ch, out);
        }
    }
}

#[cfg(feature = "corpus")]
fn alpha(s: &Surface, r: Rect) -> Vec<f32> {
    let n = s.channels();
    s.read_region(r).chunks_exact(n).map(|p| p[n - 1]).collect()
}

#[cfg(feature = "corpus")]
#[test]
fn corpus_text_layers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd");
    assert!(root.is_dir(), "{} is missing: run `cargo xtask corpus --all`", root.display());
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let mut engine = photocraft_text::TextEngine::with_system_fonts();
    let (mut n, mut good_geometry) = (0, 0);
    for f in files {
        let bytes = std::fs::read(&f).unwrap();
        if !bytes.windows(4).any(|w| w == b"TySh") {
            continue;
        }
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let Ok(imp) = photocraft_io::import(&name, &bytes) else {
            continue;
        };
        let doc = imp.document;
        // The document's Txt2 (kept verbatim on export) carries what EngineData can't, e.g.
        // optical kerning; re-reads apply it like an import does.
        let txt2 = doc.metadata.psd_global_blocks.iter().find(|b| &b.1 == b"Txt2").and_then(|b| photocraft_text::psd::parse_txt2(&b.2));
        let mut all = Vec::new();
        walk(&doc.layers, &mut all);
        for l in all {
            let LayerContent::Text(t) = &l.content else {
                continue;
            };
            n += 1;
            assert!(t.psd_raw.is_some());
            assert!(!t.runs.is_empty(), "{name}/{}: no style runs parsed", l.name);
            assert_eq!(t.runs.iter().map(|r| r.len).sum::<usize>().min(t.text.len()), t.text.len(), "{name}/{}: runs cover text", l.name);
            let st = &t.runs[0].style;
            assert!(st.size_pt > 0.0 && st.postscript_name.is_some(), "{name}/{}: {st:?}", l.name);
            // Round trip of the model through our own TySh writer.
            let rebuilt = photocraft_text::psd::build_tysh(t, doc.resolution_dpi, None);
            let mut back = photocraft_text::psd::text_layer_from_tysh(&rebuilt, doc.resolution_dpi).unwrap();
            if let Some(txt2) = &txt2 {
                photocraft_text::psd::apply_txt2(&mut back, &rebuilt, txt2);
            }
            assert_eq!(back.text, t.text);
            assert_eq!(back.char_runs(), t.char_runs(), "{name}/{}", l.name);
            assert_eq!(back.paragraph_runs(), t.paragraph_runs(), "{name}/{}", l.name);
            assert_eq!(back.shape, t.shape);

            // Geometry vs Photoshop's pixels.
            let Some(ps) = &t.cache else { continue };
            let ps_rect = ps.content_bounds();
            let (_, ours) = engine.render(t, doc.resolution_dpi, doc.pixel_format());
            let our_rect = ours.surface.content_bounds();
            let u = Rect::new(ps_rect.x0.min(our_rect.x0), ps_rect.y0.min(our_rect.y0), ps_rect.x1.max(our_rect.x1), ps_rect.y1.max(our_rect.y1));
            let (a, b) = (alpha(ps, u), alpha(&ours.surface, u));
            let inter: f32 = a.iter().zip(&b).map(|(x, y)| x.min(*y)).sum();
            let union: f32 = a.iter().zip(&b).map(|(x, y)| x.max(*y)).sum();
            let iou = if union > 0.0 { inter / union } else { 1.0 };
            let h = ps_rect.height().max(1) as f32;
            let dy = ((ps_rect.y0 + ps_rect.y1) - (our_rect.y0 + our_rect.y1)) as f32 / 2.0;
            let dx0 = (ps_rect.x0 - our_rect.x0) as f32;
            let dh = our_rect.height() as f32 / h;
            let ok = dy.abs() <= 0.25 * h && dx0.abs() <= 0.5 * h && (0.6..1.6).contains(&dh);
            good_geometry += usize::from(ok);
            println!(
                "{name:40} {:12} {:24} {:>6.1}pt {:?} ps={:?} ours={:?} iou={iou:.2} dy={dy:+.1} dx0={dx0:+.1} h×{dh:.2} {}",
                l.name,
                st.postscript_name.as_deref().unwrap_or(""),
                st.size_pt,
                t.shape,
                (ps_rect.x0, ps_rect.y0, ps_rect.width(), ps_rect.height()),
                (our_rect.x0, our_rect.y0, our_rect.width(), our_rect.height()),
                if ok { "ok" } else { "GEOMETRY OFF" }
            );
        }
    }
    println!("text layers: {n}, geometry within tolerance: {good_geometry}");
    if n > 0 {
        assert!(good_geometry * 2 >= n, "most text layers should land where Photoshop drew them");
    }
}

/// Unedited text layers are written back byte-for-byte.
#[cfg(feature = "corpus")]
#[test]
fn corpus_tysh_lossless() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/psd/ag-psd/read-write/text/src.psd");
    let bytes = std::fs::read(&root).unwrap_or_else(|e| panic!("{}: {e}: run `cargo xtask corpus --all`", root.display()));
    let doc = photocraft_io::import("src.psd", &bytes).unwrap().document;
    let out = photocraft_io::export(&doc, "out.psd", &Default::default()).unwrap().bytes;
    let find = |b: &[u8]| {
        let i = b.windows(8).position(|w| w == b"8BIMTySh").unwrap();
        let len = u32::from_be_bytes(b[i + 8..i + 12].try_into().unwrap()) as usize;
        b[i + 12..i + 12 + len].to_vec()
    };
    assert_eq!(find(&bytes), find(&out));
}

/// An unedited save of a Photoshop-authored type file writes its `Txt2` back byte-for-byte
/// (#2374: a regenerated `Txt2` made Photoshop refuse the re-saved file).
#[cfg(feature = "corpus")]
#[test]
fn corpus_photoshop_txt2_survives_an_unedited_save() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/photoshop/text");
    assert!(root.is_dir(), "{} is missing: run `cargo xtask corpus --photoshop`", root.display());
    let mut files = Vec::new();
    collect(&root, &mut files);
    files.sort();
    let txt2 = |doc: &photocraft_doc::Document| doc.metadata.psd_global_blocks.iter().find(|b| &b.1 == b"Txt2").map(|b| b.2.clone());
    let mut checked = 0;
    for f in &files {
        let name = f.file_name().unwrap().to_string_lossy().to_string();
        let doc = photocraft_io::import(&name, &std::fs::read(f).unwrap()).unwrap().document;
        let Some(before) = txt2(&doc) else { continue };
        let out = photocraft_io::export(&doc, &name, &Default::default()).unwrap().bytes;
        let back = photocraft_io::import(&name, &out).unwrap().document;
        assert!(txt2(&back) == Some(before), "{name}: Txt2 changed on an unedited save");
        checked += 1;
    }
    assert!(checked > 0, "no Photoshop type files with Txt2 in {}", root.display());
    println!("Txt2 kept byte-for-byte in {checked} Photoshop type files");
}

/// A text layer created in Photocraft (no PSD data) exports as an editable type layer and
/// imports back with the same model.
#[test]
fn created_text_layer_roundtrips_through_psd() {
    use photocraft_color::{Color, ColorMode, SampleType};
    use photocraft_doc::text::{CharStyle, ParagraphRun, ParagraphStyle, TextAlign, TextRun, TextShape};
    use photocraft_doc::{Document, Size, TextLayer};
    let mut doc = Document::new("t", Size::new(120, 80), ColorMode::Rgb, SampleType::U8);
    doc.resolution_dpi = 144.0;
    let a = CharStyle { font_family: "Inter".into(), size_pt: 10.0, color: Color::rgb(0.2, 0.4, 0.6), ..Default::default() };
    let b = CharStyle { faux_bold: true, underline: true, tracking: 50.0, leading_pt: Some(14.0), ..a.clone() };
    let mut t = TextLayer {
        text: "Größe\nzwei".into(),
        runs: vec![TextRun { len: 3, style: a }, TextRun { len: 9, style: b }],
        paragraphs: vec![
            ParagraphRun { len: 8, style: ParagraphStyle { align: TextAlign::Center, space_after_pt: 3.0, ..Default::default() } },
            ParagraphRun { len: 4, style: ParagraphStyle { align: TextAlign::JustifyAll, ..Default::default() } },
        ],
        shape: TextShape::Box { x: 0.0, y: 0.0, width: 100.0, height: 60.0 },
        transform: photocraft_geom::Affine::translate(8.0, 6.0),
        ..Default::default()
    };
    t.sync_summary();
    photocraft_text::TextEngine::new().render_layer(&mut t, doc.resolution_dpi, doc.pixel_format());
    doc.layers.push(photocraft_doc::Layer::new("t", LayerContent::Text(t.clone())));
    let out = photocraft_io::export(&doc, "t.psd", &Default::default()).unwrap();
    assert!(!out.warnings.iter().any(|w| w.contains("text layer")), "{:?}", out.warnings);
    let back = photocraft_io::import("t.psd", &out.bytes).unwrap().document;
    let LayerContent::Text(bt) = &back.layers[0].content else { panic!("not text") };
    let strip = |v: Vec<photocraft_doc::text::TextRun>| {
        v.into_iter()
            .map(|mut r| {
                r.style.postscript_name = None;
                r.style.color.c = r.style.color.c.map(|c| (c * 1000.0).round() / 1000.0);
                r
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(bt.text, t.text);
    assert_eq!(strip(bt.char_runs()), strip(t.char_runs()));
    assert_eq!(bt.paragraph_runs(), t.paragraph_runs());
    assert_eq!(bt.shape, t.shape);
    assert_eq!(bt.transform, t.transform);
    assert!(bt.cache.as_ref().is_some_and(|c| c.content_bounds().width() > 10));
}
