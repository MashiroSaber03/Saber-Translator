use super::*;
use photocraft_geom::Rect;

/// The example Invert plug-in, built from `examples/plugins/invert-rs`.
const INVERT: &[u8] = include_bytes!("../../../plugins/tests/fixtures/invert.wasm");
const INVERT_ID: &str = "org.photocraft.example.invert";

fn b64(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in b.chunks(3) {
        let n = (u32::from(c[0]) << 16) | (u32::from(*c.get(1).unwrap_or(&0)) << 8) | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            out.push(if i <= c.len() { T[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    out
}

/// A WAT plug-in with its own `id` whose `pc_filter` body is `filter`.
fn wat_plugin(id: &str, params: &str, filter: &str) -> Vec<u8> {
    let manifest = format!(r#"{{"id":"{id}","name":"Test {id}","version":"1","kind":"filter","params":{params}}}"#);
    let src = format!(
        r#"(module
  (memory (export "memory") 1)
  (data (i32.const 16) "{data}")
  (func (export "pc_abi_version") (result i32) (i32.const 1))
  (func (export "pc_manifest") (result i64) (i64.or (i64.shl (i64.const {len}) (i64.const 32)) (i64.const 16)))
  (func (export "pc_alloc") (param $n i32) (result i32) (local $old i32)
    (local.set $old (memory.grow (i32.shr_u (i32.add (local.get $n) (i32.const 65535)) (i32.const 16))))
    (if (result i32) (i32.eq (local.get $old) (i32.const -1)) (then (i32.const 0)) (else (i32.mul (local.get $old) (i32.const 65536)))))
  (func (export "pc_filter") (param $buf i32) (param $len i32) (param $w i32) (param $h i32)
    (param $ch i32) (param $fmt i32) (param $pp i32) (param $pl i32) (result i32) {filter}))"#,
        data = manifest.replace('"', "\\\""),
        len = manifest.len()
    );
    wat::parse_str(src).unwrap()
}

/// Sets every sample to 0.5.
const GREY: &str = r#"(local $i i32)
  (block $done (loop $l
    (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
    (f32.store (i32.add (local.get $buf) (local.get $i)) (f32.const 0.5))
    (local.set $i (i32.add (local.get $i) (i32.const 4)))
    (br $l)))
  (i32.const 0)"#;

fn doc(depth: u32) -> Session {
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 40, "height": 28, "depth": depth})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.edit("pattern", |doc, active| {
        let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
        for y in 0..28 {
            for x in 0..40 {
                let v = ((x * 5 + y * 3) % 17) as f32 / 16.0;
                let a = if (x * y) % 11 == 3 { 0.0 } else { 0.3 + (x % 3) as f32 * 0.35 };
                surf.write_pixel(x, y, &[v, 1.0 - v, (x % 4) as f32 / 3.0, a]);
            }
        }
        Ok(())
    })
    .unwrap();
    s
}

fn pixels(s: &Session) -> Vec<f32> {
    let d = s.active().unwrap();
    d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 40, 28))
}

fn select_partial(s: &mut Session) {
    s.edit("sel", |doc, _| {
        let mut sel = Surface::new(photocraft_color::PixelFormat::GRAY8);
        sel.fill_rect(Rect::new(4, 4, 30, 20), &[1.0]);
        sel.fill_rect(Rect::new(4, 4, 30, 9), &[0.4]);
        doc.selection = Some(sel);
        Ok(())
    })
    .unwrap();
}

#[test]
fn example_invert_matches_builtin_invert_and_round_trips_the_registry() {
    // One test owns the example plug-in's id (the registry is process-wide).
    let r = Session::new().execute("plugin.install", json!({"data": b64(INVERT)})).unwrap();
    assert_eq!(r["id"], INVERT_ID);
    assert_eq!(r["name"], "Invert (WebAssembly)");
    let list = Session::new().execute("plugin.list", json!({})).unwrap();
    assert!(list["plugins"].as_array().unwrap().iter().any(|p| p["id"] == INVERT_ID));

    for depth in [8, 16, 32] {
        for with_selection in [false, true] {
            let mut a = doc(depth);
            let mut b = doc(depth);
            if with_selection {
                select_partial(&mut a);
                select_partial(&mut b);
            }
            let before = pixels(&a);
            a.execute("image.adjustments.invert", json!({})).unwrap();
            let r = b.execute("plugin.run", json!({"id": INVERT_ID})).unwrap();
            assert_eq!(r["plugin"], INVERT_ID);
            assert_eq!(pixels(&a), pixels(&b), "depth {depth}, selection {with_selection}");
            assert_ne!(pixels(&b), before);
            // One undo step.
            b.execute("edit.undo", json!({})).unwrap();
            assert_eq!(pixels(&b), before, "depth {depth} undo");
        }
    }

    // Smart objects record a smart filter.
    let mut s = doc(8);
    s.execute("layer.smartObjects.convertToSmartObject", json!({})).unwrap();
    s.execute("plugin.run", json!({"id": INVERT_ID})).unwrap();
    let d = s.active().unwrap();
    let LayerContent::Smart(sm) = &d.doc.layer(d.active_layer.unwrap()).unwrap().content else { panic!("not a smart object") };
    assert_eq!(sm.smart_filters.len(), 1);
    assert_eq!(sm.smart_filters[0].command, RUN);
    assert_eq!(sm.smart_filters[0].params["id"], INVERT_ID);

    // Install from a path (native), then remove.
    let dir = std::env::temp_dir().join(format!("pc-plugin-cmds-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("invert.wasm");
    std::fs::write(&path, INVERT).unwrap();
    let r = Session::new().execute("plugin.install", json!({"path": path.to_str().unwrap()})).unwrap();
    assert_eq!(r["source"], path.to_str().unwrap());
    assert!(Session::new().execute("plugin.install", json!({"path": path.to_str().unwrap(), "replace": false})).is_err());
    Session::new().execute("plugin.remove", json!({"id": INVERT_ID})).unwrap();
    assert!(doc(8).execute("plugin.run", json!({"id": INVERT_ID})).is_err(), "removed");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn params_are_validated_and_passed() {
    let bytes = wat_plugin("test.cmds.grey", r#"{"amount":{"type":"number","min":0,"max":1,"default":1}}"#, GREY);
    Session::new().execute("plugin.install", json!({"data": b64(&bytes)})).unwrap();
    let list = Session::new().execute("plugin.list", json!({})).unwrap();
    let me = list["plugins"].as_array().unwrap().iter().find(|p| p["id"] == "test.cmds.grey").unwrap().clone();
    assert_eq!(me["params"], r#"{"amount":0..1=1}"#);
    let mut s = doc(16);
    let before = pixels(&s);
    let undo_before = s.active().unwrap().history.can_undo();
    // Wrong type: refused before any edit (no undo step).
    assert!(matches!(s.execute("plugin.run", json!({"id": "test.cmds.grey", "amount": "x"})), Err(EngineError::BadParams { .. })));
    assert!(s.execute("plugin.run", json!({"id": "test.cmds.grey", "params": {"amount": true}})).is_err());
    assert_eq!(pixels(&s), before);
    assert_eq!(s.active().unwrap().history.can_undo(), undo_before);
    // Flat and nested params both work.
    s.execute("plugin.run", json!({"id": "test.cmds.grey", "amount": 0.5})).unwrap();
    s.execute("plugin.run", json!({"id": "test.cmds.grey", "params": {"amount": 0.25}})).unwrap();
    assert!((pixels(&s)[0] - 0.5).abs() < 1e-4);
    Session::new().execute("plugin.remove", json!({"id": "test.cmds.grey"})).unwrap();
}

#[test]
fn hostile_plugins_fail_without_touching_the_document() {
    for (id, body) in [
        ("test.cmds.trap", "unreachable"),
        ("test.cmds.spin", "(loop $l (br $l)) (i32.const 0)"),
        ("test.cmds.oob", "(i32.store (i32.const -4) (i32.const 1)) (i32.const 0)"),
    ] {
        Session::new().execute("plugin.install", json!({"data": b64(&wat_plugin(id, "{}", body))})).unwrap();
        let mut s = doc(8);
        let before = pixels(&s);
        let undo_before = s.active().unwrap().history.can_undo();
        let t = std::time::Instant::now();
        assert!(s.execute("plugin.run", json!({"id": id})).is_err(), "{id}");
        assert!(t.elapsed() < std::time::Duration::from_secs(10), "{id} took {:?}", t.elapsed());
        assert_eq!(pixels(&s), before, "{id}");
        assert_eq!(s.active().unwrap().history.can_undo(), undo_before, "{id}: no undo step");
        Session::new().execute("plugin.remove", json!({"id": id})).unwrap();
    }
}

#[test]
fn bad_params_are_errors_not_panics() {
    let mut s = doc(8);
    for (cmd, p) in [
        ("plugin.install", json!({})),
        ("plugin.install", json!({"data": 5})),
        ("plugin.install", json!({"data": "!!!not base64"})),
        ("plugin.install", json!({"data": "QUJD"})),
        ("plugin.install", json!({"data": b64(b"\0asm\x01\0\0\0junk")})),
        ("plugin.install", json!({"path": ""})),
        ("plugin.install", json!({"path": 3})),
        ("plugin.install", json!({"path": "/definitely/not/here.wasm"})),
        ("plugin.install", json!({"path": "/"})),
        ("plugin.remove", json!({})),
        ("plugin.remove", json!({"id": "no.such.plugin"})),
        ("plugin.run", json!({})),
        ("plugin.run", json!({"id": 7})),
        ("plugin.run", json!({"id": "no.such.plugin"})),
        ("plugin.reload", json!({"path": 1})),
        ("plugin.reload", json!({"path": ""})),
        ("plugin.reload", json!({"path": "/definitely/not/here"})),
    ] {
        assert!(s.execute(cmd, p.clone()).is_err(), "{cmd} {p}");
    }
    // No document: running is disabled.
    assert!(Session::new().execute("plugin.run", json!({"id": "x"})).is_err());
    assert_eq!(b64_decode("aGk="), Some(b"hi".to_vec()));
    assert_eq!(b64_decode("aGk"), Some(b"hi".to_vec()));
    assert_eq!(b64_decode("a"), None);
}

#[test]
fn preference_folder_is_loaded() {
    let dir = std::env::temp_dir().join(format!("pc-plugin-prefs-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("grey.wasm"), wat_plugin("test.cmds.folder", "{}", GREY)).unwrap();
    std::fs::write(dir.join("broken.wasm"), b"nope").unwrap();
    let mut s = Session::new();
    // Off by default: setting the folder alone loads nothing.
    s.execute("prefs.set", json!({"path": "plugIns.additionalPluginsFolder", "value": dir.to_str().unwrap()})).unwrap();
    assert!(registry::get("test.cmds.folder").is_none());
    s.execute("prefs.set", json!({"path": "plugIns.useAdditionalPluginsFolder", "value": true})).unwrap();
    assert!(registry::get("test.cmds.folder").is_some());
    let list = s.execute("plugin.list", json!({})).unwrap();
    assert_eq!(list["folder"]["failed"].as_array().map(Vec::len), Some(1));
    registry::remove("test.cmds.folder");
    let r = s.execute("plugin.reload", json!({})).unwrap();
    assert_eq!(r["loaded"], json!(["test.cmds.folder"]));
    registry::remove("test.cmds.folder");
    std::fs::remove_dir_all(&dir).ok();
}
