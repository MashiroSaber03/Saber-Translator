//! An Affinity document is never a save target for its source; a native one can be placed, a
//! preview fallback cannot.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use photocraft_engine::Session;
use photocraft_engine::file_cmds::open_bytes_as;
use photocraft_engine::jobs::{OpenSource, Started};
use serde_json::json;

fn fixture() -> Vec<u8> {
    let mut src = Session::new();
    src.execute("file.new", json!({"width": 2, "height": 1, "background": "#dc283c"})).unwrap();
    let png = photocraft_io::export(&src.active().unwrap().doc, "preview.png", &Default::default()).unwrap().bytes;
    // Observed v12 preview envelope, without a native object graph.
    let mut bytes = vec![0; 72];
    bytes[..4].copy_from_slice(b"\x00\xffKA");
    bytes[4..6].copy_from_slice(&12u16.to_le_bytes());
    bytes[8..12].copy_from_slice(b"nsrP");
    bytes[12..16].copy_from_slice(b"#Inf");
    bytes[24..32].copy_from_slice(&72u64.to_le_bytes());
    bytes[64..68].copy_from_slice(b"Prot");
    bytes.extend(b"\xff\xff\xff\xffThmb");
    bytes.extend(1u32.to_le_bytes());
    bytes.extend((png.len() as u32 + 13).to_le_bytes());
    bytes.extend(29u32.to_le_bytes());
    bytes.extend(0u32.to_le_bytes());
    bytes.extend((png.len() as u32).to_le_bytes());
    bytes.push(1);
    bytes.extend(png);
    bytes
}

#[test]
fn preview_never_saves_back_to_its_native_or_renamed_source() {
    let dir = std::env::temp_dir().join(format!("photocraft-affinity-source-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = fixture();
    for name in ["source.af", "renamed.psd"] {
        let path = dir.join(name).to_string_lossy().into_owned();
        std::fs::write(&path, &bytes).unwrap();
        let mut s = Session::new();
        let r = open_bytes_as(&mut s, &path, &bytes, None, Some(path.clone())).unwrap();
        assert!(r["warnings"].to_string().contains("embedded"));
        assert!(s.active().unwrap().path.is_none(), "{name}");
        assert!(s.active().unwrap().source_read_only);
        assert!(!s.is_enabled("file.revert"));
        assert_eq!(std::fs::read(&path).unwrap(), bytes);
        // Explicit native export is refused before any file is written.
        let native = dir.join("refused.af");
        assert!(s.execute("file.saveACopy", json!({"path": native})).is_err());
        assert!(!native.exists());
        let copy = dir.join(format!("{name}.pcraft"));
        s.execute("file.saveACopy", json!({"path": copy})).unwrap();
        assert!(photocraft_io::import("copy.pcraft", &std::fs::read(copy).unwrap()).is_ok());
        assert_eq!(std::fs::read(path).unwrap(), bytes);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn background_preview_retains_the_source_protection_and_warning() {
    let mut s = Session::new();
    let started = s.start_open("source.af", OpenSource::Bytes(Arc::new(fixture()))).unwrap();
    let result = match started {
        Started::Job(id) => s.wait_job(id).unwrap(),
        Started::Done(v) => v,
    };
    assert!(result["warnings"].to_string().contains("no layers"));
    assert!(s.active().unwrap().source_read_only);
    assert!(s.active().unwrap().path.is_none());
}

#[test]
fn warningless_auxiliary_paths_cannot_silently_use_a_thumbnail() {
    let bytes = fixture();
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
    let layers = s.active().unwrap().doc.layers.len();
    let error = photocraft_engine::file_cmds::place_bytes(&mut s, "source.af", bytes.clone(), None, &json!({})).unwrap_err().to_string();
    assert!(error.contains("embedded preview could be read"), "{error}");
    assert_eq!(s.active().unwrap().doc.layers.len(), layers);
    assert!(photocraft_engine::smart_cmds::decode_source("source.af", &bytes).unwrap_err().to_string().contains("export PSD or PNG"));
}

/// A minimal native document: one 10×10 px red square at (5, 5) on a 40×20 page.
fn native() -> Vec<u8> {
    use photocraft_affinity::synth::{self, F, Method, tag};
    let node = |x: f64, y: f64| {
        let mut r = x.to_le_bytes().to_vec();
        r.extend(y.to_le_bytes());
        r.extend([1, 0]);
        r
    };
    let red = F::Struct([1.0f32, 0.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect());
    let square = F::Def(
        3,
        vec![tag(b"PCrv")],
        vec![
            (
                tag(b"Crvs"),
                F::Obj(
                    tag(b"PCvD"),
                    vec![(
                        tag(b"Data"),
                        F::Pos(vec![
                            F::U8(0),
                            F::U32(1),
                            F::Bool(true),
                            F::Records(18, vec![node(5.0, 5.0), node(15.0, 5.0), node(15.0, 15.0), node(5.0, 15.0), node(5.0, 5.0)]),
                        ]),
                    )],
                ),
            ),
            (
                tag(b"BFFl"),
                F::Shared(vec![F::Def(
                    4,
                    vec![tag(b"FDsc")],
                    vec![(tag(b"FDeF"), F::Def(5, vec![tag(b"FilS")], vec![(tag(b"Colr"), F::Def(6, vec![tag(b"RGBA")], vec![(tag(b"_col"), red)]))]))],
                )]),
            ),
        ],
    );
    let spread = F::Def(
        2,
        vec![tag(b"Sprd")],
        vec![(tag(b"SprB"), F::F64s(vec![0.0, 0.0, 40.0, 20.0])), (tag(b"SprT"), F::Bool(true)), (tag(b"Chld"), F::Shared(vec![square]))],
    );
    let doc = synth::stream(&[(tag(b"DocR"), F::Def(1, vec![tag(b"DocN")], vec![(tag(b"Chld"), F::Shared(vec![spread]))]))]);
    synth::container(&[("doc.dat", &doc, Method::Zstd)], None)
}

#[test]
fn native_documents_open_read_only_and_can_be_placed() {
    let bytes = native();
    let mut s = Session::new();
    let r = open_bytes_as(&mut s, "art.af", &bytes, None, Some("/source/art.af".into())).unwrap();
    assert!(!r["warnings"].to_string().contains("preview"), "{r}");
    let st = s.active().unwrap();
    assert!(st.path.is_none() && st.source_read_only);
    assert_eq!((st.doc.size.width, st.doc.size.height), (40, 20));
    s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
    let before = s.active().unwrap().doc.layers.len();
    photocraft_engine::file_cmds::place_bytes(&mut s, "art.af", bytes, None, &json!({})).unwrap();
    assert_eq!(s.active().unwrap().doc.layers.len(), before + 1);
}
