//! Krita documents through the agent surface (`doc.open`, `engine.execute`, `doc.save`, as MCP
//! and the CLI use it): open a .kra with its layers, edit it, and save it as OpenRaster.

use photocraft_automation::Headless;
use serde_json::json;

/// A tiled pixel file with one raw 64x64 tile of `bgra` at (0, 0).
fn tiles(bgra: [u8; 4]) -> Vec<u8> {
    let n = 64 * 64;
    let mut out = b"VERSION 2\nTILEWIDTH 64\nTILEHEIGHT 64\nPIXELSIZE 4\nDATA 1\n".to_vec();
    out.extend(format!("0,0,LZF,{}\n", 1 + n * 4).into_bytes());
    out.push(0);
    for b in bgra {
        out.extend(std::iter::repeat_n(b, n));
    }
    out
}

fn kra() -> Vec<u8> {
    let maindoc = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE DOC PUBLIC '-//KDE//DTD krita 2.0//EN' 'http://www.calligra.org/DTD/krita-2.0.dtd'>
<DOC xmlns="http://www.calligra.org/DTD/krita" syntaxVersion="2.0"><IMAGE name="art" width="32" height="24" colorspacename="RGBA" x-res="72" y-res="72"><layers>
<layer nodetype="grouplayer" name="Inks" filename="layer1" compositeop="normal" opacity="255" visible="1"><layers>
  <layer nodetype="paintlayer" name="Line" colorspacename="RGBA" filename="layer2" compositeop="multiply" opacity="204" visible="1"/>
</layers></layer>
<layer nodetype="shapelayer" name="Speech bubble" filename="layer3"/>
<layer nodetype="paintlayer" name="Paper" colorspacename="RGBA" filename="layer4" compositeop="normal" opacity="255" visible="1"/>
</layers></IMAGE></DOC>"#;
    let mut z = photocraft_format::zip::ZipWriter::new();
    z.add("mimetype", b"application/x-krita").unwrap();
    z.add("maindoc.xml", maindoc.as_bytes()).unwrap();
    z.add("art/layers/layer2", &tiles([0, 0, 255, 255])).unwrap();
    z.add("art/layers/layer4", &tiles([255, 255, 255, 255])).unwrap();
    z.finish().unwrap()
}

fn exec(h: &mut Headless, command: &str, params: serde_json::Value) -> serde_json::Value {
    h.handle("engine.execute", json!({"command": command, "params": params})).unwrap()
}

#[test]
fn krita_documents_open_with_layers_and_save_as_openraster() {
    let dir = std::env::temp_dir().join(format!("pc-krita-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("art.kra");
    std::fs::write(&path, kra()).unwrap();
    let mut h = Headless::trusted_local();
    let opened = h.handle("doc.open", json!({"path": path.to_str().unwrap()})).unwrap();
    let warnings = opened["warnings"].to_string();
    assert!(warnings.contains("vector layers") && warnings.contains("Speech bubble"), "{opened}");
    let doc = &h.session.active().unwrap().doc;
    let names: Vec<&str> = doc.layers.iter().map(|l| l.name.as_str()).collect();
    assert_eq!(names, ["Paper", "Inks"]);
    let line = doc.layers[1].children().unwrap()[0].clone();
    assert_eq!((line.name.as_str(), line.blend), ("Line", photocraft_doc::BlendMode::Multiply));
    let flat = photocraft_compose::flatten(doc);
    assert!((flat.px[0][1] - 0.2).abs() < 0.01, "80% red multiplied over white: {:?}", flat.px[0]);

    // Read-only source: the file itself is never overwritten as Krita.
    assert!(h.handle("doc.save", json!({"path": path.to_str().unwrap()})).is_err());
    exec(&mut h, "layer.setProps", json!({"layer": line.id.0, "opacity": 0.5}));
    let ora = dir.join("art.ora");
    h.handle("doc.save", json!({"path": ora.to_str().unwrap()})).unwrap();
    let back = photocraft_io::import("art.ora", &std::fs::read(&ora).unwrap()).unwrap().document;
    assert_eq!(back.layers.len(), 2);
    assert_eq!(back.layers[1].children().unwrap()[0].opacity, 0.5);
    let _ = std::fs::remove_dir_all(&dir);
}
