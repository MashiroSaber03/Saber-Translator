//! doc → .pcraft → doc equality, incremental saves, ids, previews.

mod common;
use common::*;
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::text::{CharStyle, ParagraphStyle, TextRun};
use photocraft_doc::text_styles::{CharacterStyleDef, ParagraphStyleDef};
use photocraft_doc::{Document, Layer, LayerContent, LayerId, Slice, TextLayer};
use photocraft_format::*;
use photocraft_raster::Rgba8Image;

const MODES: [ColorMode; 4] = [ColorMode::Rgb, ColorMode::Grayscale, ColorMode::Cmyk, ColorMode::Lab];

#[test]
fn custom_shape_stroke_roundtrip_keeps_every_style_field() {
    for mode in MODES {
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            let mut d = rich_doc(mode, depth);
            let stroke = photocraft_doc::ShapeStroke {
                width: 5.5,
                opacity: 0.35,
                align: photocraft_doc::StrokeAlign::Outside,
                cap: photocraft_doc::LineCap::Square,
                join: photocraft_doc::LineJoin::Bevel,
                miter_limit: 2.0,
                dashes: vec![0.0, 2.0, 4.0, 1.0, 1.5, 3.0],
                dash_offset: -0.5,
                ..Default::default()
            };
            d.layers.push(Layer::new(
                "Custom stroke",
                LayerContent::Shape(photocraft_doc::ShapeLayer {
                    path: photocraft_doc::Path::new(vec![photocraft_doc::Subpath::polygon(&[(10.0, 10.0), (40.0, 10.0), (25.0, 40.0)])]),
                    stroke: Some(stroke),
                    ..Default::default()
                }),
            ));
            let bytes = save_to_bytes(&d, &SaveOptions::default()).unwrap();
            assert_eq!(load_from_bytes(&bytes).unwrap(), d);
        }
    }
}

fn check_zip(mode: ColorMode, depth: SampleType) {
    let doc = rich_doc(mode, depth);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    assert!(is_pcraft(&bytes));
    let back = load_from_bytes(&bytes).unwrap();
    assert_eq!(back, doc, "{mode:?} {depth:?}");
}

macro_rules! zip_cases {
    ($($name:ident: $m:ident, $d:ident;)*) => {$(
        #[test]
        fn $name() {
            check_zip(ColorMode::$m, SampleType::$d);
        }
    )*};
}

zip_cases! {
    rgb_u8: Rgb, U8; rgb_u16: Rgb, U16; rgb_f32: Rgb, F32;
    gray_u8: Grayscale, U8; gray_u16: Grayscale, U16; gray_f32: Grayscale, F32;
    cmyk_u8: Cmyk, U8; cmyk_u16: Cmyk, U16; cmyk_f32: Cmyk, F32;
    lab_u8: Lab, U8; lab_u16: Lab, U16; lab_f32: Lab, F32;
}

#[test]
fn directory_bundle_roundtrip_all_modes() {
    for mode in MODES {
        for depth in SampleType::ALL {
            let doc = rich_doc(mode, depth);
            let dir = temp_dir("dir");
            PcraftWriter::new().save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
            assert!(dir.join("manifest.json").is_file());
            assert_eq!(load_path(&dir).unwrap(), doc, "{mode:?} {depth:?}");
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn save_path_zip_file() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("file");
    let path = dir.join("a.pcraft");
    PcraftWriter::new().save_path(&doc, &path, &SaveOptions::default()).unwrap();
    assert!(path.is_file());
    assert_eq!(load_path(&path).unwrap(), doc);
    std::fs::remove_dir_all(dir).unwrap();
}

/// The public [`ByteSource`] entry loads a saved directory like `load_path` does.
#[test]
fn load_bytes_source_matches_the_path_load() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("bytes-source");
    let bundle = dir.join("a.pcraft");
    PcraftWriter::new().save_dir(&doc, &bundle, &SaveOptions::default()).unwrap();
    let loaded = load_bytes_source(&FsDir { root: bundle.clone() }, &LoadOptions::default()).unwrap();
    assert_eq!(loaded, doc);
    assert!(load_bytes_source(&FsDir { root: bundle.join("no-such-dir") }, &LoadOptions::default()).is_err(), "a missing bundle errors");
    std::fs::remove_dir_all(dir).unwrap();
}

/// `read_entry` bounds the read itself, not the stat hint: a stream that ignores the hint
/// and yields more than `max` errors instead of filling memory.
#[test]
fn read_entry_bounds_the_read_not_the_hint() {
    struct Endless;
    impl std::io::Read for Endless {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            buf.fill(b'x');
            Ok(buf.len())
        }
    }
    let mut endless = Endless;
    let error = read_entry(&mut endless, 0, 100, "manifest.json").unwrap_err().to_string();
    assert!(error.contains("exceeds"), "{error}");
    let mut small = &b"hi"[..];
    assert_eq!(read_entry(&mut small, 2, 100, "manifest.json").unwrap(), b"hi");
}

/// Reads the on-disk bundle shape for [`load_bytes_source_matches_the_path_load`].
struct FsDir {
    root: std::path::PathBuf,
}

impl ByteSource for FsDir {
    fn get(&self, path: &str, max: usize) -> Result<Vec<u8>> {
        let p = self.root.join(path);
        let metadata = std::fs::metadata(&p).map_err(|_| FormatError::Corrupt(format!("missing `{path}`")))?;
        if !metadata.is_file() {
            return Err(FormatError::Corrupt(format!("`{path}` is not a regular file")));
        }
        let mut file = std::fs::File::open(&p).map_err(|_| FormatError::Corrupt(format!("missing `{path}`")))?;
        read_entry(&mut file, metadata.len(), max, path)
    }
}

/// Deepest group nesting in `layers` (0 when there are no groups).
fn group_depth(layers: &[Layer]) -> usize {
    layers.iter().map(|l| if let photocraft_doc::LayerContent::Group(g) = &l.content { 1 + group_depth(&g.children) } else { 0 }).max().unwrap_or(0)
}

/// [`rich_doc`] with its layers wrapped in groups until they are nested `levels` deep.
fn nested_doc(levels: usize, depth: SampleType) -> Document {
    let mut doc = rich_doc(ColorMode::Rgb, depth);
    let mut layers = std::mem::take(&mut doc.layers);
    for i in group_depth(&layers)..levels {
        layers = vec![Layer::group(format!("Level {i}"), layers)];
    }
    doc.layers = layers;
    assert_eq!(group_depth(&doc.layers), levels);
    doc
}

/// serde_json's default limit (128 levels) used to make bundles with 40+ nested groups unreadable.
/// Runs on a 1 MiB stack, the smallest main-thread stack we ship on (Windows, wasm).
#[test]
fn deeply_nested_groups_roundtrip() {
    let run = || {
        for (levels, depth) in [(40, SampleType::U8), (MAX_GROUP_DEPTH, SampleType::U16), (MAX_GROUP_DEPTH, SampleType::F32)] {
            let doc = nested_doc(levels, depth);
            let back = load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
            assert_eq!(back, doc, "{levels} levels at {depth:?}");
        }
    };
    std::thread::Builder::new().stack_size(1 << 20).spawn(run).unwrap().join().unwrap();
}

/// A save never writes a bundle the loader would reject for its nesting.
#[test]
fn nesting_beyond_the_load_limit_is_refused_at_save() {
    let e = save_to_bytes(&nested_doc(MAX_GROUP_DEPTH + 1, SampleType::U8), &SaveOptions::default()).unwrap_err();
    assert!(matches!(e, FormatError::LimitExceeded(_)), "{e}");
}

#[test]
fn empty_document_roundtrips() {
    let doc = Document::new("empty", photocraft_doc::Size::new(1, 1), ColorMode::Rgb, SampleType::U8);
    assert_eq!(load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap(), doc);
}

#[test]
fn exhausted_document_ids_and_maximal_text_runs_roundtrip() {
    let mut doc = Document::new("exhausted", photocraft_doc::Size::new(2, 1), ColorMode::Rgb, SampleType::U8);
    doc.slices.list.push(Slice { id: u32::MAX, ..Default::default() });
    doc.text_styles.character.push(CharacterStyleDef { id: u32::MAX, ..Default::default() });
    doc.text_styles.paragraph.push(ParagraphStyleDef { id: u32::MAX, ..Default::default() });
    doc.layers.push(Layer::new(
        "Text",
        LayerContent::Text(TextLayer {
            text: "ab".into(),
            runs: vec![TextRun { len: 1, style: CharStyle::default() }, TextRun { len: usize::MAX, style: CharStyle { size_pt: 24.0, ..Default::default() } }],
            paragraphs: vec![
                photocraft_doc::text::ParagraphRun { len: 1, style: ParagraphStyle::default() },
                photocraft_doc::text::ParagraphRun { len: usize::MAX, style: ParagraphStyle::default() },
            ],
            ..Default::default()
        }),
    ));

    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let loaded = load_from_bytes(&bytes).unwrap();
    assert_eq!(loaded.slices.next_id(), None);
    assert_eq!(loaded.text_styles.next_char_id(), None);
    assert_eq!(loaded.text_styles.next_para_id(), None);
    let LayerContent::Text(text) = &loaded.layers.last().unwrap().content else { panic!("text layer must roundtrip") };
    assert_eq!(text.char_runs().iter().map(|r| r.len).sum::<usize>(), text.text.len());
    assert_eq!(text.char_runs()[1].len, 1);
    assert_eq!(text.char_runs()[1].style.size_pt, 24.0);
    assert_eq!(text.paragraph_runs().iter().map(|r| r.len).sum::<usize>(), text.text.len());
}

#[test]
fn zip_incremental_only_new_tiles() {
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U16);
    let mut w = PcraftWriter::new();
    let (_, s1) = w.save_zip(&doc, &SaveOptions::default()).unwrap();
    assert!(s1.tiles_total > 5);
    assert_eq!(s1.tiles_written, s1.tiles_total);
    let (_, s2) = w.save_zip(&doc, &SaveOptions::default()).unwrap();
    assert_eq!(s2.tiles_written, 0);
    assert_eq!(s2.tiles_reused, s2.tiles_total);
    assert_eq!(s2.blobs_written, 0);
    // Touch one pixel of one layer: exactly one tile is new.
    let id = doc.layers[1].id;
    doc.layer_mut(id).unwrap().surface_mut().unwrap().write_pixel(3, 3, &[0.5, 0.5, 0.5, 1.0]);
    let (bytes, s3) = w.save_zip(&doc, &SaveOptions::default()).unwrap();
    assert_eq!(s3.tiles_written, 1);
    assert_eq!(load_from_bytes(&bytes).unwrap(), doc);
}

#[test]
fn directory_incremental_and_gc() {
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("gc");
    let mut w = PcraftWriter::new();
    let s1 = w.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    assert_eq!(s1.tiles_written, s1.tiles_total);
    let s2_same_writer = w.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    assert_eq!(s2_same_writer.tiles_written, 0);
    assert_eq!(s2_same_writer.tiles_reused, s2_same_writer.tiles_total);
    // A fresh writer still skips files already on disk.
    let s2 = PcraftWriter::new().save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    assert_eq!(s2.tiles_written, 0);
    assert_eq!(s2.objects_removed, 0);
    // Delete a layer: its unique tiles get garbage-collected.
    let paint = doc.layers[1].id;
    doc.remove(paint);
    let s3 = w.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    assert!(s3.objects_removed > 0, "{s3:?}");
    assert_eq!(load_path(&dir).unwrap(), doc);
    let files = std::fs::read_dir(dir.join("tiles")).unwrap().count();
    assert_eq!(files, s3.tiles_total);
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(unix)]
#[test]
fn directory_save_rejects_object_directory_symlink_without_deleting_external_objects() {
    use std::os::unix::fs::symlink;

    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("symlink-object-dir");
    let outside = temp_dir("symlink-object-dir-outside");
    let external_object = outside.join("unrelated.zst");
    std::fs::write(&external_object, b"keep this object").unwrap();
    symlink(&outside, dir.join("tiles")).unwrap();

    assert!(PcraftWriter::new().save_dir(&doc, &dir, &SaveOptions::default()).is_err());
    assert_eq!(std::fs::read(&external_object).unwrap(), b"keep this object");
    assert!(!dir.join("manifest.json").exists());

    std::fs::remove_file(dir.join("tiles")).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}

#[cfg(unix)]
#[test]
fn directory_save_rejects_object_symlink_without_overwriting_target() {
    use std::os::unix::fs::symlink;

    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("symlink-object");
    let outside = temp_dir("symlink-object-outside");
    let mut writer = PcraftWriter::new();
    writer.save_dir(&doc, &dir, &SaveOptions::default()).unwrap();
    let object = std::fs::read_dir(dir.join("tiles")).unwrap().next().unwrap().unwrap().path();
    std::fs::remove_file(&object).unwrap();
    let external_object = outside.join("external.zst");
    std::fs::write(&external_object, b"do not overwrite").unwrap();
    symlink(&external_object, &object).unwrap();

    assert!(writer.save_dir(&doc, &dir, &SaveOptions::default()).is_err());
    assert_eq!(std::fs::read(&external_object).unwrap(), b"do not overwrite");
    assert!(std::fs::symlink_metadata(&object).unwrap().file_type().is_symlink());

    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}

#[cfg(unix)]
#[test]
fn directory_save_accepts_chosen_root_symlink_but_rejects_preview_symlink() {
    use std::os::unix::fs::symlink;

    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let outside = temp_dir("symlink-preview-outside");
    let root_link_parent = temp_dir("symlink-root-parent");
    let root_link = root_link_parent.join("linked.pcraft");
    symlink(&outside, &root_link).unwrap();
    PcraftWriter::new().save_dir(&doc, &root_link, &SaveOptions::default()).unwrap();
    assert_eq!(load_path(&root_link).unwrap(), doc);
    let trailing_separator = std::path::PathBuf::from(format!("{}/", root_link.display()));
    PcraftWriter::new().save_dir(&doc, &trailing_separator, &SaveOptions::default()).unwrap();
    assert!(outside.join("manifest.json").is_file());

    let dir = temp_dir("symlink-preview");
    let external_preview = outside.join("preview.png");
    std::fs::write(&external_preview, b"keep preview").unwrap();
    std::fs::create_dir(dir.join("composite")).unwrap();
    symlink(&external_preview, dir.join("composite/preview.png")).unwrap();
    let opts = SaveOptions { composite: Some(Rgba8Image::new(2, 2)), ..Default::default() };
    assert!(PcraftWriter::new().save_dir(&doc, &dir, &opts).is_err());
    assert_eq!(std::fs::read(&external_preview).unwrap(), b"keep preview");
    assert!(!dir.join("manifest.json").exists());

    std::fs::remove_file(root_link).unwrap();
    std::fs::remove_dir_all(root_link_parent).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}

#[cfg(unix)]
#[test]
fn directory_save_rejects_manifest_symlink_without_overwriting_target() {
    use std::os::unix::fs::symlink;

    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let dir = temp_dir("symlink-manifest");
    let outside = temp_dir("symlink-manifest-outside");
    let external_manifest = outside.join("external.json");
    std::fs::write(&external_manifest, b"keep manifest").unwrap();
    symlink(&external_manifest, dir.join("manifest.json")).unwrap();

    assert!(PcraftWriter::new().save_dir(&doc, &dir, &SaveOptions::default()).is_err());
    assert_eq!(std::fs::read(&external_manifest).unwrap(), b"keep manifest");
    assert_eq!(std::fs::read_dir(dir.join("tiles")).unwrap().count(), 0);

    std::fs::remove_dir_all(dir).unwrap();
    std::fs::remove_dir_all(outside).unwrap();
}

#[test]
fn identical_tiles_are_stored_once() {
    let mut doc = Document::new("d", photocraft_doc::Size::new(10, 10), ColorMode::Rgb, SampleType::U8);
    let mut a = Layer::raster("a", doc.pixel_format());
    scribble(a.surface_mut().unwrap(), 5, false);
    let b = a.duplicate();
    doc.layers.push(a);
    doc.layers.push(b);
    let (_, s) = PcraftWriter::new().save_zip(&doc, &SaveOptions::default()).unwrap();
    let per_layer = doc.layers[0].surface().unwrap().tile_count();
    assert_eq!(s.tiles_total, per_layer);
}

/// A solid fill saves one tile however many coordinates share it; reopening must decode it once,
/// charge the byte budget once and keep the sharing (#3315). The budget here holds one tile,
/// standing in for a canvas too large to expand tile by tile.
#[test]
fn solid_fill_reopens_with_one_shared_tile() {
    for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let size = photocraft_doc::Size::new(2048, 2048);
        let mut doc = Document::new("d", size, ColorMode::Rgb, depth);
        let mut l = Layer::raster("solid", doc.pixel_format());
        l.surface_mut().unwrap().fill_rect(photocraft_geom::Rect::from_size(size), &[1.0, 0.5, 0.25, 1.0]);
        doc.layers.push(l);
        let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
        let one_tile = (256 * 256 * doc.pixel_format().bytes_per_pixel()) as u64;
        let back = load_from_bytes_with(&bytes, &LoadOptions { max_total_bytes: one_tile, ..Default::default() }).unwrap();
        assert_eq!(back, doc, "{depth:?}");
        let s = back.layers[0].surface().unwrap();
        assert_eq!(s.tile_count(), 64, "{depth:?}");
        let first = s.tiles().next().unwrap().1;
        assert!(s.tiles().all(|(_, t)| std::sync::Arc::ptr_eq(t, first)), "{depth:?}: tiles not shared after reopening");
    }
}

#[test]
fn ids_preserved_and_counter_advanced() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    // Simulate a file written with much larger ids.
    let mut m = read_manifest(&bytes).unwrap();
    m.document.layers[0].id = LayerId::fresh().0 + 5000;
    let big = m.document.layers[0].id;
    let text = serde_json::to_vec(&m).unwrap();
    let mut z = zip::ZipWriter::new();
    z.add("manifest.json", &text).unwrap();
    let r = zip::ZipReader::new(&bytes).unwrap();
    for e in r.entries.iter().filter(|e| e.name != "manifest.json") {
        z.add(&e.name, &r.read(e, usize::MAX).unwrap()).unwrap();
    }
    let back = load_from_bytes(&z.finish().unwrap()).unwrap();
    assert_eq!(back.layers[0].id.0, big);
    assert!(LayerId::fresh().0 > big, "fresh ids must not collide with loaded ids");
}

#[test]
fn fresh_ids_option() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let back = load_from_bytes_with(&bytes, &LoadOptions { preserve_ids: false, ..Default::default() }).unwrap();
    assert_ne!(back.id, doc.id);
    assert_ne!(back.layers[0].id, doc.layers[0].id);
    assert_eq!(back.layers.len(), doc.layers.len());
    assert_eq!(back.layers[1].surface(), doc.layers[1].surface());
}

#[test]
fn previews_embedded_and_readable() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let mut thumb = Rgba8Image::new(4, 3);
    thumb.pixels.iter_mut().enumerate().for_each(|(i, p)| *p = i as u8);
    let opts = SaveOptions { thumbnail: Some(thumb.clone()), composite: Some(thumb.clone()) };
    let bytes = save_to_bytes(&doc, &opts).unwrap();
    let png = read_thumbnail(&bytes).unwrap().unwrap();
    let img = photocraft_codecs::decode(&png).unwrap();
    assert_eq!(img.dimensions(), (4, 3));
    assert_eq!(img.to_rgba8(), thumb.pixels);
    let m = read_manifest(&bytes).unwrap();
    assert_eq!(m.composite.as_deref(), Some("composite/preview.png"));
    assert_eq!(m.format_version, FORMAT_VERSION);
    assert!(read_thumbnail(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap().is_none());
}

#[test]
fn manifest_is_human_readable_json() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let r = zip::ZipReader::new(&bytes).unwrap();
    assert_eq!(r.entries[0].name, "manifest.json");
    let text = String::from_utf8(r.read_by_name("manifest.json", usize::MAX).unwrap()).unwrap();
    assert!(text.contains("\"format_version\": 1"));
    assert!(text.contains("\"Hue/Sat\""));
    assert!(text.contains("filter.blur.gaussian"));
    assert!(r.entries.iter().any(|e| e.name.starts_with("tiles/") && e.name.ends_with(".zst")));
    assert!(r.entries.iter().any(|e| e.name.starts_with("blobs/")));
}

#[test]
fn save_is_deterministic() {
    let doc = rich_doc(ColorMode::Cmyk, SampleType::U16);
    let a = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let b = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    assert_eq!(a, b);
}

#[test]
fn not_pcraft() {
    assert!(!is_pcraft(b""));
    assert!(!is_pcraft(b"8BPS...."));
    let mut z = zip::ZipWriter::new();
    z.add("other.txt", b"x").unwrap();
    assert!(!is_pcraft(&z.finish().unwrap()));
}

#[test]
fn vector_fields_roundtrip_and_default_when_absent() {
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let back = load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
    assert_eq!(back.paths, doc.paths);
    assert_eq!(back.work_path, doc.work_path);
    assert_eq!(back.clipping_path, doc.clipping_path);
    let vm = |d: &Document| d.walk().into_iter().find_map(|(_, _, l)| l.vector_mask.clone());
    assert!(vm(&back).is_some());
    assert_eq!(vm(&back), vm(&doc));
    // Manifests written before the vector fields existed still load (serde defaults).
    let m = read_manifest(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
    let mut v = serde_json::to_value(&m.document).unwrap();
    for k in ["paths", "work_path", "clipping_path"] {
        v.as_object_mut().unwrap().remove(k);
    }
    fn strip(layers: &mut serde_json::Value) {
        for l in layers.as_array_mut().unwrap() {
            l.as_object_mut().unwrap().remove("vector_mask");
            let c = l["content"].as_object_mut().unwrap();
            for k in ["path", "stroke", "live"] {
                c.remove(k);
            }
            if let Some(ch) = c.get_mut("children") {
                strip(ch);
            }
        }
    }
    strip(&mut v["layers"]);
    let old: photocraft_format::manifest::DocM = serde_json::from_value(v).unwrap();
    assert!(old.paths.is_empty() && old.work_path.is_none());
    assert!(old.layers.iter().all(|l| l.vector_mask.is_none()));
}

/// Type kerning (#206): the auto mode and manual kerning survive .pcraft; manifests written
/// before `kern` existed load with no manual kerning.
#[test]
fn text_kerning_roundtrips_and_defaults() {
    use photocraft_doc::LayerContent;
    use photocraft_doc::text::Kerning;
    let doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let back = load_from_bytes(&bytes).unwrap();
    let runs = |d: &Document| {
        d.walk()
            .into_iter()
            .find_map(|(_, _, l)| match &l.content {
                LayerContent::Text(t) => Some(t.runs.iter().map(|r| (r.len, r.style.kerning, r.style.kern)).collect::<Vec<_>>()),
                _ => None,
            })
            .unwrap()
    };
    assert_eq!(runs(&back), vec![(1, Kerning::Off, 120.0), (5, Kerning::Metrics, 0.0), (9, Kerning::Optical, 0.0)]);
    assert_eq!(runs(&back), runs(&doc));
    let m = read_manifest(&bytes).unwrap();
    let mut v = serde_json::to_value(&m.document).unwrap();
    fn strip(v: &mut serde_json::Value) {
        match v {
            serde_json::Value::Object(o) => {
                o.remove("kern");
                o.values_mut().for_each(strip);
            }
            serde_json::Value::Array(a) => a.iter_mut().for_each(strip),
            _ => {}
        }
    }
    strip(&mut v);
    assert!(!v.to_string().contains("\"kern\""));
    let old: photocraft_format::manifest::DocM = serde_json::from_value(v).unwrap();
    let s = serde_json::to_string(&old).unwrap();
    assert!(s.contains("\"kern\":0.0"), "kern defaults to 0");
}

/// Type on a path (#218): the path, start offset and flip survive .pcraft.
#[test]
fn path_text_shape_roundtrips() {
    use photocraft_doc::text::TextShape;
    use photocraft_doc::vector::Subpath;
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    let path = Subpath::polyline(&[(0.0, 0.0), (50.0, 20.0), (120.0, 5.5)]);
    let ids: Vec<LayerId> = doc.walk().into_iter().filter(|(_, _, l)| matches!(l.content, LayerContent::Text(_))).map(|(_, _, l)| l.id).collect();
    for id in ids {
        if let Some(LayerContent::Text(t)) = doc.layer_mut(id).map(|l| &mut l.content) {
            t.shape = TextShape::Path { path: path.clone(), start: 12.5, flip: true };
        }
    }
    let back = load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
    let shape = back.walk().into_iter().find_map(|(_, _, l)| match &l.content {
        LayerContent::Text(t) => Some(t.shape.clone()),
        _ => None,
    });
    assert_eq!(shape, Some(TextShape::Path { path, start: 12.5, flip: true }));
}

#[test]
fn channel_restrictions_and_bevel_elements_roundtrip() {
    use photocraft_doc::{Bevel, BevelContour, BevelTexture, Contour, Effect};
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U8);
    doc.layers[0].excluded_channels = 0b101;
    let mut b = Bevel {
        enabled: true,
        style: photocraft_doc::BevelStyle::Emboss,
        technique: photocraft_doc::BevelTechnique::ChiselHard,
        depth: 1.2,
        up: false,
        size: 9.0,
        soften: 1.0,
        angle: 30.0,
        altitude: 40.0,
        use_global_light: false,
        gloss_contour: Contour::Linear,
        highlight: photocraft_doc::FxCommon::new(photocraft_color::BlendMode::Screen, 0.7),
        highlight_color: photocraft_color::Color::WHITE,
        shadow: photocraft_doc::FxCommon::new(photocraft_color::BlendMode::Multiply, 0.6),
        shadow_color: photocraft_color::Color::BLACK,
        contour: None,
        texture: None,
    };
    b.contour = Some(BevelContour { contour: Contour::Linear, range: 0.3, anti_alias: true });
    b.texture = Some(BevelTexture { name: "p".into(), id: "i".into(), scale: 0.5, depth: -1.5, invert: true, link: false, phase: (1.0, 2.0) });
    doc.layers[0].effects.items.push(Effect::BevelEmboss(Bevel { ..b }));
    let back = load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
    assert_eq!(back, doc);
    assert_eq!(back.layers[0].excluded_channels, 0b101);
}

#[test]
fn blend_if_roundtrips() {
    use photocraft_doc::{BlendIf, BlendRange};
    let mut doc = rich_doc(ColorMode::Rgb, SampleType::U16);
    let mut bi = BlendIf::default();
    bi.set(0, [BlendRange { black: [20, 60], white: [255, 255] }, BlendRange::FULL]);
    bi.set(3, [BlendRange::FULL, BlendRange { black: [0, 0], white: [180, 220] }]);
    doc.layers[0].blend_if = bi.clone();
    let back = load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
    assert_eq!(back, doc);
    assert_eq!(back.layers[0].blend_if, bi);
    // Layers without Blend If keep the default (the field is omitted from the manifest).
    assert!(back.layers.iter().skip(1).all(|l| l.blend_if.is_default()));
}

/// Advanced Blending (knockout, as-group switches, transparency shapes, masks hide effects)
/// survives .pcraft at every depth; manifests written before the field existed load with
/// Photoshop's defaults, and default layers don't write it.
#[test]
fn advanced_blending_roundtrips_and_defaults_when_absent() {
    use photocraft_doc::{AdvancedBlending, Knockout};
    for depth in SampleType::ALL {
        let mut doc = rich_doc(ColorMode::Rgb, depth);
        let a = AdvancedBlending {
            knockout: Knockout::Deep,
            blend_interior: true,
            blend_clipped: false,
            transparency_shapes: false,
            layer_mask_hides_effects: true,
            vector_mask_hides_effects: true,
        };
        doc.layers[0].advanced = a;
        if let Some(l) = doc.layers.get_mut(1) {
            l.advanced.knockout = Knockout::Shallow;
        }
        let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
        let back = load_from_bytes(&bytes).unwrap();
        assert_eq!(back, doc, "{depth:?}");
        assert_eq!(back.layers[0].advanced, a);
        // Only the non-default layers carry the field.
        let m = read_manifest(&bytes).unwrap();
        let v = serde_json::to_value(&m.document).unwrap();
        let with = v["layers"].as_array().unwrap().iter().filter(|l| l.get("advanced").is_some()).count();
        assert_eq!(with, if doc.layers.len() > 1 { 2 } else { 1 });
        // An older manifest (no `advanced`) loads with the defaults.
        let mut old = v.clone();
        for l in old["layers"].as_array_mut().unwrap() {
            l.as_object_mut().unwrap().remove("advanced");
        }
        let old: photocraft_format::manifest::DocM = serde_json::from_value(old).unwrap();
        assert!(old.layers.iter().all(|l| l.advanced.is_default()));
    }
}

#[test]
fn video_layer_frames_survive_roundtrip() {
    use photocraft_doc::{Timeline, VideoData, VideoSource};
    use photocraft_geom::Rect;
    let mut doc = Document::new("Vid", photocraft_geom::Size { width: 8, height: 6 }, ColorMode::Rgb, SampleType::U8);
    doc.timeline = Some(Timeline::new(3, 24.0));
    let fmt = doc.pixel_format();
    let mut l = Layer::raster("Clip", fmt);
    let mut frames = Vec::new();
    for i in 0..3u8 {
        let mut s = photocraft_raster::Surface::new(fmt);
        let v = i as f32 / 2.0;
        s.fill_rect(Rect::new(0, 0, 8, 6), &photocraft_raster::from_rgba(&fmt, [v, 0.0, 1.0 - v, 1.0]));
        frames.push(s);
    }
    let mut vd = VideoData::new(frames, 24.0);
    vd.original_frames = Some(vd.frames.clone());
    vd.source = VideoSource::File { path: "/tmp/seq".into() };
    vd.show_altered = false;
    l.video = Some(vd);
    doc.layers.push(l);

    let bytes = save_to_bytes(&doc, &SaveOptions::default()).unwrap();
    let back = load_from_bytes(&bytes).unwrap();
    let v = back.layers.last().unwrap().video.as_ref().expect("video survived");
    assert_eq!(v.frames.len(), 3);
    assert_eq!(v.original_frames.as_ref().map(Vec::len), Some(3));
    assert_eq!(v.original_frames.as_ref().unwrap()[2].rgba(1, 1), v.frames[2].rgba(1, 1));
    assert_eq!(v.fps, 24.0);
    assert!(!v.show_altered);
    assert_eq!(v.source, VideoSource::File { path: "/tmp/seq".into() });
    assert_eq!(back.timeline.as_ref().unwrap().duration, 3);
    // A middle frame's pixel survived.
    let mut px = [[0.0f32; 4]; 1];
    v.frames[2].read_rgba_into(Rect::from_xywh(1, 1, 1, 1), &mut px);
    assert!(px[0][0] > 0.9 && px[0][2] < 0.1, "frame 2 is reddish: {:?}", px[0]);
}

/// Issue #1101: serde_json writes NaN and the infinities as `null`, which the manifest's float
/// fields refuse on load, so such a document saved "successfully" and never opened again. The
/// save is refused instead, naming the value; finite documents are unaffected.
#[test]
fn a_non_finite_float_is_refused_at_save_rather_than_breaking_every_later_load() {
    use photocraft_doc::Adjustment;
    type Mutate = Box<dyn Fn(&mut Document)>;
    let base = rich_doc(ColorMode::Rgb, SampleType::U8);
    let cases: Vec<(&str, Mutate)> = vec![
        (
            "exposure",
            Box::new(|d| {
                d.insert_above(None, Layer::new("adj", LayerContent::Adjustment(Adjustment::Exposure { exposure: f32::NAN, offset: 0.0, gamma: 1.0 })));
            }),
        ),
        (
            "gamma",
            Box::new(|d| {
                d.insert_above(
                    None,
                    Layer::new("adj", LayerContent::Adjustment(Adjustment::Exposure { exposure: 0.0, offset: 0.0, gamma: f32::NEG_INFINITY })),
                );
            }),
        ),
        ("resolution_dpi", Box::new(|d| d.resolution_dpi = f32::INFINITY)),
    ];
    for (field, mutate) in cases {
        let mut doc = base.clone();
        mutate(&mut doc);
        let err = save_to_bytes(&doc, &SaveOptions::default()).expect_err(field);
        assert!(matches!(err, FormatError::NonFinite { .. }), "{field}: {err:?}");
        let msg = err.to_string();
        assert!(msg.contains(field) && msg.contains("NaN or infinite"), "{field}: {msg}");
        // The incremental writer shares the check, for autosave and directory bundles.
        let mut w = PcraftWriter::new();
        assert!(matches!(w.save_zip(&doc, &SaveOptions::default()), Err(FormatError::NonFinite { .. })), "{field}: PcraftWriter");
    }
    // The same document with finite values round-trips as before.
    let mut doc = base.clone();
    doc.insert_above(None, Layer::new("adj", LayerContent::Adjustment(Adjustment::Exposure { exposure: 0.5, offset: 0.0, gamma: 1.0 })));
    let back = load_from_bytes(&save_to_bytes(&doc, &SaveOptions::default()).unwrap()).unwrap();
    assert_eq!(back, doc);
}
