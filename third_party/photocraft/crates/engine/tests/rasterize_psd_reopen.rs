//! Layer › Rasterize › Fill Content / Smart Object stays rasterized after a PSD save and reopen.
//! A fill or smart layer opened from a PSD keeps its file blocks (`SoCo`/`GdFl`/`PtFl`,
//! `SoLd`/`PlLd`); left on the rasterized layer, they made the reopened file a fill layer or
//! smart object again and threw the pixels away. Photoshop writes none of them for a pixel layer.

use photocraft_doc::{Document, LayerContent, LayerId};
use photocraft_engine::Session;
use serde_json::{Value, json};

fn id_of(r: Value) -> LayerId {
    LayerId(r["layer"].as_u64().unwrap())
}

fn kind(l: &photocraft_doc::Layer) -> &'static str {
    match &l.content {
        LayerContent::Fill(_) => "fill",
        LayerContent::Smart(_) => "smart",
        LayerContent::Raster(_) => "raster",
        _ => "other",
    }
}

fn kind_named(doc: &Document, name: &str) -> &'static str {
    let l = doc.walk().into_iter().map(|(_, _, l)| l).find(|l| l.name == name).unwrap_or_else(|| panic!("no layer {name}"));
    kind(l)
}

fn reopen(doc: &Document) -> Document {
    let out = photocraft_io::export(doc, "psd", &Default::default()).unwrap();
    photocraft_io::import("x.psd", &out.bytes).unwrap().document
}

/// Whether the PSD written for `doc` gives the layer `name` any of `keys`.
fn has_block(doc: &Document, name: &str, keys: &[&[u8; 4]]) -> bool {
    let out = photocraft_io::export(doc, "psd", &Default::default()).unwrap();
    let f = photocraft_psd::PsdFile::from_bytes(&out.bytes).unwrap();
    let rec = f.layers().iter().find(|r| r.name() == name).unwrap();
    keys.iter().any(|k| rec.block(k).is_some())
}

/// Rasterizes every layer named in `names` of a session opened on `doc`, then saves and reopens.
fn rasterize_and_reopen(doc: Document, names: &[&str]) -> Document {
    let mut s = Session::new();
    s.add_document(doc, None);
    for name in names {
        let id = s.active().unwrap().doc.walk().into_iter().map(|(_, _, l)| l).find(|l| l.name == *name).unwrap().id;
        s.execute("layer.rasterize.layer", json!({"layer": id.0})).unwrap();
        assert_eq!(kind_named(&s.active().unwrap().doc, name), "raster");
    }
    reopen(&s.active().unwrap().doc)
}

#[test]
fn rasterized_fill_and_smart_object_reopen_as_pixels() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 48, "height": 32})).unwrap();
    s.execute("layer.new.layer", json!({"name": "Source"})).unwrap();
    s.execute("paint.stroke", json!({"points": [[20, 16]], "size": 12})).unwrap();
    let smart = id_of(s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap());
    let fill = id_of(s.execute("layer.newFillLayer.solidColor", json!({"color": "#00ff00"})).unwrap());
    let doc = s.active().unwrap().doc.clone();
    let smart_name = doc.layer(smart).unwrap().name.clone();
    let fill_name = doc.layer(fill).unwrap().name.clone();
    assert_ne!(smart_name, fill_name);

    // Opened from a PSD, both layers carry their file blocks.
    let opened = reopen(&doc);
    assert_eq!(kind_named(&opened, &fill_name), "fill");
    assert_eq!(kind_named(&opened, &smart_name), "smart");

    let back = rasterize_and_reopen(opened, &[&fill_name, &smart_name]);
    assert_eq!(kind_named(&back, &fill_name), "raster");
    assert_eq!(kind_named(&back, &smart_name), "raster");
    assert!(!has_block(&back, &fill_name, &[b"SoCo", b"GdFl", b"PtFl"]));
    assert!(!has_block(&back, &smart_name, &[b"SoLd", b"PlLd", b"SoLE"]));
}

/// The exporter never writes content-defining blocks for a pixel layer, whichever command made it
/// one (a backstop for paths that swap the content without pruning `psd_blocks`).
#[test]
fn raster_layer_with_stale_blocks_writes_none() {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
    s.execute("layer.newFillLayer.solidColor", json!({"color": "#ff0000"})).unwrap();
    let mut doc = reopen(&s.active().unwrap().doc);
    let (id, name) = doc.walk().into_iter().map(|(_, _, l)| l).find(|l| kind(l) == "fill").map(|l| (l.id, l.name.clone())).unwrap();
    let fmt = doc.pixel_format();
    let l = doc.layer_mut(id).unwrap();
    assert!(l.psd_blocks.iter().any(|(k, _)| k == b"SoCo"));
    l.content = LayerContent::Raster(photocraft_raster::Surface::new(fmt));
    l.psd_blocks.push((*b"TySh", std::sync::Arc::new(vec![0; 8])));
    l.psd_blocks.push((*b"vscg", std::sync::Arc::new(vec![0; 8])));
    l.psd_blocks.push((*b"levl", std::sync::Arc::new(vec![0; 8])));
    assert!(!has_block(&doc, &name, &[b"SoCo", b"TySh", b"vstk", b"vscg", b"SoLd", b"PlLd", b"levl"]));
    assert_eq!(kind_named(&reopen(&doc), &name), "raster");
}

/// The same through files Photoshop wrote: a gradient fill layer and a placed smart object.
#[cfg(feature = "corpus")]
#[test]
fn photoshop_fill_and_smart_layers_stay_rasterized() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    for (file, want) in [("psd-tools/gradient-fill.psd", "fill"), ("photoshop/smart-filters/sf-none-plain.psd", "smart")] {
        let path = root.join(file);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e} (run `cargo xtask corpus --all`)", path.display()));
        let doc = photocraft_io::import(file, &bytes).unwrap().document;
        let names: Vec<String> = doc.walk().into_iter().map(|(_, _, l)| l).filter(|l| kind(l) == want).map(|l| l.name.clone()).collect();
        assert!(!names.is_empty(), "{file}: no {want} layer");
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let back = rasterize_and_reopen(doc, &refs);
        for n in &names {
            assert_eq!(kind_named(&back, n), "raster", "{file}: {n}");
        }
    }
}
