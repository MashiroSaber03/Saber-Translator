//! EXR parts: what a multi-part or channel-grouped EXR file holds, and opening the chosen
//! parts/groups as the layers of one document (a Maya/Arnold render writes one part per
//! AOV). The query command feeds the File › Open EXR Parts… dialog, the CLI, MCP and the
//! control channel alike.

use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::file_cmds::{native, read_file, stem};
use crate::{EngineError, Result, Session};

fn entries_json(path: &str) -> Result<Value> {
    let bytes = read_file(path)?;
    let entries = photocraft_io::exr_parts::entries(&bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    Ok(json!(entries
        .iter()
        .map(|e| json!({
            "kind": match e.kind { photocraft_io::exr_parts::EntryKind::Part => "part", photocraft_io::exr_parts::EntryKind::Group => "group", photocraft_io::exr_parts::EntryKind::Cryptomatte => "cryptomatte" },
            "name": e.name,
            "view": e.view,
            "width": e.width,
            "height": e.height,
            "channels": e.channels,
            "objects": e.objects,
        }))
        .collect::<Vec<_>>()))
}

fn exr_parts(s: &mut Session, p: &Value) -> Result<Value> {
    let _ = s;
    let path = crate::file_cmds::str_param(p, "path", "file.exrParts")?.to_string();
    entries_json(&path)
}

/// Collects `part/<name>` and `group/<name>` boolean params into the selection names.
fn selected(p: &Value, prefix: &str) -> Vec<String> {
    p.as_object()
        .map(|m| m.iter().filter(|(k, v)| k.starts_with(prefix) && v.as_bool() == Some(true)).map(|(k, _)| k[prefix.len()..].to_string()).collect())
        .unwrap_or_default()
}

fn open_exr_parts(s: &mut Session, p: &Value) -> Result<Value> {
    let path = crate::file_cmds::str_param(p, "path", "file.openExrParts")?.to_string();
    let view = p.get("view").and_then(Value::as_str).unwrap_or("both");
    if !matches!(view, "both" | "left" | "right") {
        return Err(EngineError::BadParams { cmd: "file.openExrParts".into(), msg: format!("view is \"{view}\"; it is \"both\", \"left\" or \"right\"") });
    }
    let sel = photocraft_io::exr_parts::Selection {
        parts: selected(p, "part/"),
        groups: selected(p, "group/"),
        view: view.to_string(),
        precomp: p.get("precomp").and_then(Value::as_bool).unwrap_or(false),
    };
    let bytes = read_file(&path)?;
    let r = photocraft_io::exr_parts::import(&stem(&path), &bytes, &sel).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    let (i, color) = s.open_document(r.document, Some(path.clone()));
    Ok(json!({"document": i, "color": color, "warnings": r.warnings}))
}

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $sc:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: $sc, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    vec![
        spec!("file.exrParts", "EXR Parts Info", &[], None, r##"{"path":str (the EXR file)}"##, native, exr_parts),
        spec!(
            "file.openExrParts",
            "Open EXR Parts…",
            &["File"],
            None,
            r##"{"path":str (the EXR file),"view":"both|left|right"="both","part/<name>":bool (import this part as a layer),"group/<name>":bool (import this channel group as a layer),"precomp":bool=false (add the additive beauty precomp from the AOVs)}"##,
            native,
            open_exr_parts
        ),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::ColorMode;

    fn session() -> Session {
        Session::new()
    }

    // The fixture writer mirrors the codecs tests: a small multi-part EXR via the `exr`
    // crate, one part per AOV plus a single-part file with prefixed channel groups.

    fn write_multipart(path: &std::path::Path) {
        use exr::prelude::*;
        let (w, h) = (4usize, 4usize);
        let mk = |n: String, v: f32| AnyChannel::new(n.as_str(), FlatSamples::F32(vec![v; w * h]));
        let beauty = AnyChannels::sort(vec![mk("R".into(), 0.8), mk("G".into(), 0.4), mk("B".into(), 0.2), mk("A".into(), 1.0)].into_iter().collect());
        let z = AnyChannels::sort(vec![mk("Z".into(), 5.0)].into_iter().collect());
        let layer = |name: &str, chans: AnyChannels<FlatSamples>| {
            Layer::new(
                (w, h),
                LayerAttributes::named(name),
                Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
                chans,
            )
        };
        let image = Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer("Z", z), layer("beauty", beauty)]);
        image.write().to_buffered(&mut std::io::BufWriter::new(std::fs::File::create(path).expect("create fixture"))).expect("write fixture");
    }

    fn write_grouped(path: &std::path::Path) {
        use exr::prelude::*;
        let (w, h) = (4usize, 4usize);
        let mk = |n: String, v: f32| AnyChannel::new(n.as_str(), FlatSamples::F32(vec![v; w * h]));
        // Two AOV groups (each with alpha) plus the beauty RGBA, in ONE part.
        let list = vec![
            mk("diffuse.R".into(), 0.3),
            mk("diffuse.G".into(), 0.3),
            mk("diffuse.B".into(), 0.3),
            mk("diffuse.A".into(), 0.5),
            mk("specular.R".into(), 0.6),
            mk("specular.G".into(), 0.6),
            mk("specular.B".into(), 0.6),
            mk("specular.A".into(), 0.25),
            mk("R".into(), 0.9),
            mk("G".into(), 0.9),
            mk("B".into(), 0.9),
            mk("A".into(), 1.0),
        ];
        let layer = Layer::new(
            (w, h),
            LayerAttributes::named("main"),
            Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
            AnyChannels::sort(list.into_iter().collect()),
        );
        let image = Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
        image.write().to_buffered(&mut std::io::BufWriter::new(std::fs::File::create(path).expect("create fixture"))).expect("write fixture");
    }

    #[test]
    fn exr_parts_lists_parts_groups_and_views() {
        let dir = std::env::temp_dir().join(format!("pcraft-exr-parts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aov.exr");
        write_multipart(&path);
        let info = entries_json(path.to_str().unwrap()).unwrap();
        let kinds: Vec<&str> = info.as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap()).collect();
        assert_eq!(kinds, ["part", "part"], "{info}");
        assert_eq!(info[0]["name"], "Z");
        assert_eq!(info[1]["name"], "beauty");
        assert_eq!(info[1]["channels"], 4);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn open_exr_parts_imports_the_chosen_part_as_a_layer() {
        let dir = std::env::temp_dir().join(format!("pcraft-exr-open-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aov.exr");
        write_multipart(&path);
        let mut s = session();
        let r = s.execute("file.openExrParts", json!({"path": path.to_str().unwrap(), "part/beauty": true, "part/Z": false})).unwrap();
        assert_eq!(r["warnings"].as_array().unwrap().len(), 0, "{r}");
        let doc = &s.active().unwrap().doc;
        assert_eq!(doc.layers.len(), 1);
        assert_eq!(doc.layers[0].name, "beauty");
        // The beauty pixels survived (R 0.8 at pixel 0).
        if let photocraft_doc::LayerContent::Raster(surface) = &doc.layers[0].content {
            let px = surface.pixel(0, 0);
            assert!((px[0] - 0.8).abs() < 1e-4, "{:?}", px);
        } else {
            panic!("expected a raster layer");
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn open_exr_parts_refuses_an_empty_selection_and_unknown_names() {
        let dir = std::env::temp_dir().join(format!("pcraft-exr-empty-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("aov.exr");
        write_multipart(&path);
        let mut s = session();
        let e = s.execute("file.openExrParts", json!({"path": path.to_str().unwrap()})).unwrap_err();
        assert!(e.to_string().contains("no parts or groups selected"), "{e}");
        let e = s.execute("file.openExrParts", json!({"path": path.to_str().unwrap(), "part/nope": true})).unwrap_err();
        assert!(e.to_string().contains("no flat part named nope"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn channel_groups_import_from_their_own_channels() {
        let dir = std::env::temp_dir().join(format!("pcraft-exr-groups-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("grouped.exr");
        write_grouped(&path);
        let info = entries_json(path.to_str().unwrap()).unwrap();
        let names: Vec<&str> = info.as_array().unwrap().iter().map(|e| e["name"].as_str().unwrap()).collect();
        assert_eq!(names, ["main", "diffuse", "specular"], "{info}");
        let mut s = session();
        s.execute("file.openExrParts", json!({"path": path.to_str().unwrap(), "group/diffuse": true})).unwrap();
        let doc = &s.active().unwrap().doc;
        assert_eq!(doc.layers.len(), 1);
        assert_eq!(doc.layers[0].name, "diffuse");
        if let photocraft_doc::LayerContent::Raster(surface) = &doc.layers[0].content {
            let px = surface.pixel(0, 0);
            // The channels import as stored (straight here): values are untouched.
            assert!((px[0] - 0.3).abs() < 1e-4, "{:?}", px);
            assert!((px[3] - 0.5).abs() < 1e-4, "{:?}", px);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// A single-part EXR with no part name (the usual render output) plus an AOV group.
    fn write_unnamed_grouped(path: &std::path::Path) {
        use exr::prelude::*;
        let (w, h) = (4usize, 4usize);
        let mk = |n: &str, v: f32| AnyChannel::new(n, FlatSamples::F32(vec![v; w * h]));
        let list = vec![mk("R", 0.7), mk("G", 0.7), mk("B", 0.7), mk("diffuse.R", 0.3), mk("diffuse.G", 0.3), mk("diffuse.B", 0.3)];
        let layer = Layer::new(
            (w, h),
            LayerAttributes::default(),
            Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
            AnyChannels::sort(list.into_iter().collect()),
        );
        let image = Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
        image.write().to_buffered(&mut std::io::BufWriter::new(std::fs::File::create(path).expect("create fixture"))).expect("write fixture");
    }

    #[test]
    fn an_unnamed_part_opens_by_the_name_the_listing_gives_it() {
        // #4210: the listing names an unnamed part "part 0"; feeding that name back must open it.
        let dir = std::env::temp_dir().join(format!("pcraft-exr-unnamed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("unnamed.exr");
        write_unnamed_grouped(&path);
        let info = entries_json(path.to_str().unwrap()).unwrap();
        let part = info.as_array().unwrap().iter().find(|e| e["kind"] == "part").expect("a part entry");
        let name = part["name"].as_str().unwrap();
        assert_eq!(name, "part 0", "{info}");
        let key = format!("part/{name}");
        let mut s = session();
        s.execute("file.openExrParts", json!({"path": path.to_str().unwrap(), key: true})).unwrap();
        let doc = &s.active().unwrap().doc;
        assert_eq!(doc.layers.len(), 1);
        assert_eq!(doc.layers[0].name, "part 0");
        if let photocraft_doc::LayerContent::Raster(surface) = &doc.layers[0].content {
            let px = surface.pixel(0, 0);
            assert!((px[0] - 0.7).abs() < 1e-4, "{:?}", px);
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn precomp_sums_the_light_path_set_and_refuses_an_incomplete_one() {
        let dir = std::env::temp_dir().join(format!("pcraft-exr-precomp-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("grouped.exr");
        write_grouped(&path);
        let mut s = session();
        // diffuse + specular make the set incomplete: emission is missing.
        let e = s.execute("file.openExrParts", json!({"path": path.to_str().unwrap(), "precomp": true, "group/diffuse": true})).unwrap_err();
        assert!(e.to_string().contains("missing") && e.to_string().contains("emission"), "{e}");

        // A complete set: write a file holding direct/indirect diffuse+specular and emission.
        let full = dir.join("full.exr");
        use exr::prelude::*;
        let (w, h) = (4usize, 4usize);
        let mk = |n: String, v: f32| AnyChannel::new(n.as_str(), FlatSamples::F32(vec![v; w * h]));
        let aovs = [
            ("direct_diffuse", 0.10, 1.0),
            ("indirect_diffuse", 0.20, 1.0),
            ("direct_specular", 0.05, 1.0),
            ("indirect_specular", 0.15, 1.0),
            ("emission", 0.30, 1.0),
        ];
        let mut list: Vec<AnyChannel<FlatSamples>> = Vec::new();
        let mut expect = 0.0f32;
        for (name, v, a) in aovs {
            list.push(mk(format!("{name}.R"), v));
            list.push(mk(format!("{name}.G"), v));
            list.push(mk(format!("{name}.B"), v));
            list.push(mk(format!("{name}.A"), a));
            expect += v * a;
        }
        let layer = Layer::new(
            (w, h),
            LayerAttributes::named("main"),
            Encoding { compression: Compression::Uncompressed, blocks: Blocks::ScanLines, line_order: LineOrder::Increasing },
            AnyChannels::sort(list.into_iter().collect()),
        );
        let image = Image::from_layers(ImageAttributes::new(IntegerBounds::new((0, 0), (w, h))), vec![layer]);
        image.write().to_buffered(&mut std::io::BufWriter::new(std::fs::File::create(&full).expect("create fixture"))).expect("write fixture");
        let mut s = session();
        let r = s.execute("file.openExrParts", json!({"path": full.to_str().unwrap(), "precomp": true})).unwrap();
        assert!(r["warnings"].as_array().unwrap().iter().any(|w| w.as_str().unwrap().contains("precomp")), "{r}");
        let doc = &s.active().unwrap().doc;
        let precomp = doc.layers.iter().find(|l| l.name == "beauty (precomp)").expect("the precomp layer");
        if let photocraft_doc::LayerContent::Raster(surface) = &precomp.content {
            let px = surface.pixel(0, 0);
            assert!((px[0] - expect).abs() < 1e-3, "{:?} vs {expect}", px);
        }
        assert_eq!(doc.mode, ColorMode::Rgb);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
