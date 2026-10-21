//! The plug-in ABI end to end: the example Invert plug-in, and hostile modules (infinite loops,
//! memory bombs, out-of-bounds access, bad manifests…), which must all fail with an error, quickly.

use std::time::{Duration, Instant};

use photocraft_color::{PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_plugins::{Error, Limits, Plugin};
use photocraft_raster::Surface;
use serde_json::json;

const INVERT: &[u8] = include_bytes!("fixtures/invert.wasm");

/// A module implementing the ABI whose pieces can be swapped: `manifest` JSON, `version`,
/// `filter` body (params `$buf $len $w $h $ch $fmt $pp $pl`, result i32) and extra items.
fn module(manifest: &str, version: i32, filter: &str, extra: &str) -> Vec<u8> {
    let escaped = manifest.replace('\\', "\\\\").replace('"', "\\\"");
    let src = format!(
        r#"(module
  {extra}
  (memory (export "memory") 1)
  (data (i32.const 16) "{escaped}")
  (func (export "pc_abi_version") (result i32) (i32.const {version}))
  (func (export "pc_manifest") (result i64)
    (i64.or (i64.shl (i64.const {len}) (i64.const 32)) (i64.const 16)))
  (func (export "pc_alloc") (param $n i32) (result i32) (local $old i32)
    (local.set $old (memory.grow (i32.shr_u (i32.add (local.get $n) (i32.const 65535)) (i32.const 16))))
    (if (result i32) (i32.eq (local.get $old) (i32.const -1))
      (then (i32.const 0)) (else (i32.mul (local.get $old) (i32.const 65536)))))
  (func (export "pc_filter") (param $buf i32) (param $len i32) (param $w i32) (param $h i32)
    (param $ch i32) (param $fmt i32) (param $pp i32) (param $pl i32) (result i32)
    {filter})
)"#,
        len = manifest.len()
    );
    wat::parse_str(&src).unwrap_or_else(|e| panic!("bad test WAT: {e}\n{src}"))
}

const MANIFEST: &str = r#"{"id":"test.plugin","name":"Test","version":"1","kind":"filter"}"#;

fn ok_module(filter: &str) -> Vec<u8> {
    module(MANIFEST, 1, filter, "")
}

fn surface(fmt: PixelFormat, w: i32, h: i32) -> Surface {
    let mut s = Surface::new(fmt);
    for y in 0..h {
        for x in 0..w {
            let v = ((x * 7 + y * 13) % 31) as f32 / 30.0;
            let a = if (x + y) % 9 == 0 { 0.0 } else { 0.25 + 0.75 * ((x % 4) as f32 / 3.0) };
            s.write_pixel(x, y, &[v, 1.0 - v, (x % 5) as f32 / 4.0, a]);
        }
    }
    s
}

/// Runs `bytes` as a filter on a small image; returns the error and how long it took.
fn run_err(bytes: &[u8], limits: Limits) -> (Error, Duration) {
    let t = Instant::now();
    let r = Plugin::load(bytes, limits).and_then(|p| {
        let s = surface(PixelFormat::RGBA8, 32, 24);
        p.apply(&s, Rect::new(0, 0, 32, 24), None, &json!({}))
    });
    (r.expect_err("hostile module must fail"), t.elapsed())
}

#[test]
fn example_invert_loads_and_reports_its_manifest() {
    let p = Plugin::load(INVERT, Limits::default()).unwrap();
    let m = p.manifest();
    assert_eq!(m.id, "org.photocraft.example.invert");
    assert_eq!(m.name, "Invert (WebAssembly)");
    assert_eq!(m.kind, photocraft_plugins::Kind::Filter);
    assert!(m.params.is_empty());
    assert!(p.size() < 4096, "the example stays tiny: {} bytes", p.size());
}

#[test]
fn example_invert_inverts_colour_at_every_depth() {
    let p = Plugin::load(INVERT, Limits::default()).unwrap();
    for sample in SampleType::ALL {
        let fmt = PixelFormat { sample, ..PixelFormat::RGBA8 };
        let s = surface(fmt, 40, 30);
        let out = p.apply(&s, Rect::new(0, 0, 40, 30), None, &json!({})).unwrap();
        for (x, y) in [(1, 1), (0, 0), (17, 3), (39, 29)] {
            let (a, b) = (s.pixel(x, y), out.pixel(x, y));
            if a[3] <= 0.0 {
                assert_eq!(a, b, "transparent pixels are left alone");
                continue;
            }
            for c in 0..3 {
                assert!((b[c] - (1.0 - a[c])).abs() < 1e-6, "{sample:?} ({x},{y}) c{c}: {a:?} -> {b:?}");
            }
            assert_eq!(a[3], b[3], "alpha is kept");
        }
    }
}

#[test]
fn bands_are_seamless_and_selection_blends() {
    // Tiny bands: many instances, same result as one band.
    let small = Limits { band_bytes: 40 * 4 * 4 * 3, ..Limits::default() };
    let p1 = Plugin::load(INVERT, Limits::default()).unwrap();
    let p2 = Plugin::load(INVERT, small).unwrap();
    let s = surface(PixelFormat::RGBA16, 40, 30);
    let canvas = Rect::new(0, 0, 40, 30);
    assert_eq!(p1.apply(&s, canvas, None, &json!({})).unwrap(), p2.apply(&s, canvas, None, &json!({})).unwrap());

    // A half-strength selection over the left half.
    let mut sel = Surface::new(PixelFormat::GRAY8);
    sel.fill_rect(Rect::new(0, 0, 20, 30), &[1.0]);
    sel.fill_rect(Rect::new(0, 0, 20, 10), &[0.5]);
    let out = p1.apply(&s, canvas, Some(&sel), &json!({})).unwrap();
    assert_eq!(out.pixel(30, 20), s.pixel(30, 20), "outside the selection");
    let (a, b) = (s.pixel(4, 20), out.pixel(4, 20));
    assert!((b[0] - (1.0 - a[0])).abs() < 1e-4, "fully selected");
    let (a, b) = (s.pixel(5, 5), out.pixel(5, 5));
    let k = sel.pixel(5, 5)[0];
    let half = a[0] + (1.0 - a[0] - a[0]) * k;
    assert!((b[0] - half).abs() < 2.0 / 65535.0, "half selected: {a:?} {b:?}");
}

#[test]
fn params_and_image_context_reach_the_plugin() {
    // Writes params_len / 1000 into every sample, so the host must pass the JSON through.
    let fill = r#"(local $i i32)
      (block $done (loop $l
        (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
        (f32.store (i32.add (local.get $buf) (local.get $i))
          (f32.div (f32.convert_i32_u (local.get $pl)) (f32.const 1000)))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $l)))
      (i32.const 0)"#;
    let man = r#"{"id":"test.params","name":"P","kind":"filter","params":{"amount":{"type":"number","min":0,"max":10,"default":5}}}"#;
    let p = Plugin::load(&module(man, 1, fill, ""), Limits::default()).unwrap();
    assert_eq!(p.manifest().params_notation(), r#"{"amount":0..10=5}"#);
    let s = surface(PixelFormat::RGBA32F, 8, 8);
    let out = p.apply(&s, Rect::new(0, 0, 8, 8), None, &json!({"amount": 3})).unwrap();
    let v = out.pixel(1, 1)[0] * 1000.0;
    assert!(v > 100.0 && v < 600.0, "params JSON length {v}");
    // Wrong-typed params are refused before the plug-in runs.
    assert!(matches!(p.apply(&s, Rect::new(0, 0, 8, 8), None, &json!({"amount": "x"})), Err(Error::Params(_))));
}

#[test]
fn nan_and_out_of_range_output_is_sanitised() {
    // Fills the buffer with NaN: every pixel keeps its value.
    let nan = r#"(local $i i32)
      (block $done (loop $l
        (br_if $done (i32.ge_u (local.get $i) (local.get $len)))
        (f32.store (i32.add (local.get $buf) (local.get $i)) (f32.const nan))
        (local.set $i (i32.add (local.get $i) (i32.const 4)))
        (br $l)))
      (i32.const 0)"#;
    let p = Plugin::load(&ok_module(nan), Limits::default()).unwrap();
    let s = surface(PixelFormat::RGBA32F, 8, 8);
    assert_eq!(p.apply(&s, Rect::new(0, 0, 8, 8), None, &json!({})).unwrap(), s);
}

#[test]
fn hostile_modules_fail_quickly() {
    let quick = Duration::from_secs(5);
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("infinite loop", ok_module("(loop $l (br $l)) (i32.const 0)")),
        ("memory.grow bomb, then trap", ok_module("(loop $l (br_if $l (i32.ne (memory.grow (i32.const 512)) (i32.const -1)))) unreachable")),
        ("memory.grow forever", ok_module("(loop $l (drop (memory.grow (i32.const 1))) (br $l)) (i32.const 0)")),
        ("out-of-bounds write", ok_module("(i32.store (i32.const -4) (i32.const 0)) (i32.const 0)")),
        ("trap", ok_module("unreachable")),
        ("division by zero", ok_module("(i32.div_u (i32.const 1) (local.get $ch))  (drop) (i32.div_u (i32.const 1) (i32.const 0))")),
        ("deep recursion", module(MANIFEST, 1, "(call $r) (i32.const 0)", "(func $r (call $r))")),
        ("error code", ok_module("(i32.const 7)")),
        ("wrong ABI version", module(MANIFEST, 2, "(i32.const 0)", "")),
        ("bad manifest", module("not json", 1, "(i32.const 0)", "")),
        ("manifest with a bad id", module(r#"{"id":"a b","name":"x","kind":"filter"}"#, 1, "(i32.const 0)", "")),
        ("manifest of the wrong kind", module(r#"{"id":"a","name":"x","kind":"panel"}"#, 1, "(i32.const 0)", "")),
        ("start function loops", module(MANIFEST, 1, "(i32.const 0)", "(func $spin (loop $l (br $l))) (start $spin)")),
        ("WASI import", module(MANIFEST, 1, "(i32.const 0)", r#"(import "wasi_snapshot_preview1" "fd_write" (func (param i32 i32 i32 i32) (result i32)))"#)),
    ];
    for (name, bytes) in cases {
        let (e, t) = run_err(&bytes, Limits::default());
        assert!(t < quick, "{name} took {t:?}");
        eprintln!("{name}: {e} ({t:?})");
    }
}

#[test]
fn malformed_and_oversized_modules_are_refused() {
    let half = &INVERT[..INVERT.len() / 2];
    for (name, bytes) in [("empty", &b""[..]), ("garbage", &b"\0asm\x01\0\0\0\xff\xff\xff"[..]), ("not wasm", &b"MZ\x90\0"[..]), ("truncated", half)] {
        assert!(matches!(Plugin::load(bytes, Limits::default()), Err(Error::Module(_))), "{name}");
    }
    let tiny = Limits { max_module_bytes: 100, ..Limits::default() };
    assert!(matches!(Plugin::load(INVERT, tiny), Err(Error::Module(_))));
    // Huge initial memory, manifest outside memory, missing exports, bogus allocations.
    let huge = wat::parse_str(r#"(module (memory (export "memory") 60000))"#).unwrap();
    assert!(Plugin::load(&huge, Limits::default()).is_err());
    let no_filter = wat::parse_str(
        r#"(module (memory (export "memory") 1) (func (export "pc_abi_version") (result i32) (i32.const 1))
           (func (export "pc_manifest") (result i64) (i64.const 0)))"#,
    )
    .unwrap();
    assert!(Plugin::load(&no_filter, Limits::default()).is_err());
    let src = r#"(module (memory (export "memory") 1) (func (export "pc_abi_version") (result i32) (i32.const 1))
           (func (export "pc_manifest") (result i64) (i64.const 0x00000064ffffff00))
           (func (export "pc_alloc") (param i32) (result i32) (i32.const 16))
           (func (export "pc_filter") (param i32 i32 i32 i32 i32 i32 i32 i32) (result i32) (i32.const 0)))"#;
    assert!(matches!(Plugin::load(&wat::parse_str(src).unwrap(), Limits::default()), Err(Error::Abi(_))));
    let bogus_alloc = wat::parse_str(format!(
        r#"(module (memory (export "memory") 1) (data (i32.const 16) "{}")
           (func (export "pc_abi_version") (result i32) (i32.const 1))
           (func (export "pc_manifest") (result i64) (i64.const {}))
           (func (export "pc_alloc") (param i32) (result i32) (i32.const -16))
           (func (export "pc_filter") (param i32 i32 i32 i32 i32 i32 i32 i32) (result i32) (i32.const 0)))"#,
        MANIFEST.replace('"', "\\\""),
        ((MANIFEST.len() as i64) << 32) | 16
    ))
    .unwrap();
    let (e, _) = run_err(&bogus_alloc, Limits::default());
    assert!(matches!(e, Error::Abi(_)), "{e}");
}

#[test]
fn budgets_stop_runaway_plugins() {
    let spin = ok_module("(loop $l (br $l)) (i32.const 0)");
    // Instruction budget.
    let (e, _) = run_err(&spin, Limits { fuel_base: 1_000_000, fuel_per_sample: 10, ..Limits::default() });
    assert_eq!(e, Error::Limit("instruction budget".into()));
}

#[test]
fn parallel_bands_preserve_the_originating_error() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build().unwrap();
    let mut s = Surface::new(PixelFormat::RGBA8);
    s.fill_rect(Rect::new(0, 0, 256, 256), &[0.0, 0.0, 0.0, 1.0]);
    s.fill_rect(Rect::new(0, 256, 256, 512), &[1.0, 0.0, 0.0, 1.0]);
    for (body, expected) in [
        ("(i32.const 7)", Error::Failed("pc_filter returned error code 7".into())),
        ("unreachable", Error::Trap("wasm `unreachable` instruction executed".into())),
    ] {
        // The later band fails immediately while the earlier one is still running (or hasn't
        // started). Collecting in band order must not turn that failure into sibling cancellation.
        let filter = format!(
            "(if (result i32) (f32.gt (f32.load (local.get $buf)) (f32.const 0.5)) \
             (then {body}) (else (loop $spin (br $spin)) (i32.const 0)))"
        );
        let limits = Limits { band_bytes: 1, fuel_base: 100_000_000, fuel_per_sample: 0, ..Limits::default() };
        let p = Plugin::load(&ok_module(&filter), limits).unwrap();
        let before = s.clone();
        let e = pool.install(|| p.apply(&s, Rect::new(0, 0, 256, 512), None, &json!({}))).unwrap_err();
        assert_eq!(e, expected);
        assert_eq!(s, before, "a failed run must not change the input");
    }
}

#[test]
fn parallel_bands_preserve_instruction_budget() {
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let p = Plugin::load(
        &ok_module("(loop $spin (br $spin)) (i32.const 0)"),
        Limits { band_bytes: 1, fuel_base: 1_000_000, fuel_per_sample: 0, ..Limits::default() },
    )
    .unwrap();
    let mut s = Surface::new(PixelFormat::RGBA8);
    s.fill_rect(Rect::new(0, 0, 768, 512), &[0.25, 0.5, 0.75, 1.0]);
    for _ in 0..8 {
        let e = pool.install(|| p.apply(&s, Rect::new(0, 0, 768, 512), None, &json!({}))).unwrap_err();
        assert_eq!(e, Error::Limit("instruction budget".into()));
    }
}

#[test]
fn wall_clock_budget_stops_runaway_plugins() {
    let spin = ok_module("(loop $l (br $l)) (i32.const 0)");
    // Wall-clock budget, with an effectively unlimited instruction budget.
    let (e, t) = run_err(&spin, Limits { fuel_base: u64::MAX / 4, wall_time_ms: 300, ..Limits::default() });
    assert_eq!(e, Error::Limit("time budget".into()));
    assert!(t < Duration::from_secs(3), "{t:?}");
}

#[test]
fn memory_cap_limits_bands() {
    // Memory cap: a band that can't fit is refused up front.
    let s = surface(PixelFormat::RGBA8, 64, 8);
    let p = Plugin::load(INVERT, Limits { max_memory_bytes: 4 << 20, band_bytes: 64 << 20, ..Limits::default() }).unwrap();
    assert!(p.apply(&s, Rect::new(0, 0, 64, 8), None, &json!({})).is_ok(), "small bands still fit");
    let p = Plugin::load(INVERT, Limits { max_memory_bytes: 1024, ..Limits::default() });
    assert!(p.is_err() || p.unwrap().apply(&s, Rect::new(0, 0, 64, 8), None, &json!({})).is_err());
}

#[test]
fn registry_install_list_remove() {
    // Its own id: the registry is process-wide and tests run in parallel.
    let id = "test.registry";
    let p = photocraft_plugins::registry::install_bytes(&module(r#"{"id":"test.registry","name":"R","kind":"filter"}"#, 1, "(i32.const 0)", "")).unwrap();
    assert_eq!(p.id(), id);
    let rev = photocraft_plugins::registry::revision();
    assert!(photocraft_plugins::registry::list().iter().any(|p| p.id() == id));
    assert!(photocraft_plugins::registry::get(id).is_some());
    assert!(photocraft_plugins::registry::remove(id));
    assert!(!photocraft_plugins::registry::remove(id));
    assert!(photocraft_plugins::registry::get(id).is_none());
    assert!(photocraft_plugins::registry::revision() > rev);
}

#[test]
fn folder_loading_skips_bad_files() {
    let dir = std::env::temp_dir().join(format!("photocraft-plugins-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("a-invert.wasm"), INVERT).unwrap();
    std::fs::write(dir.join("b-broken.wasm"), b"\0asm junk").unwrap();
    std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();
    let r = photocraft_plugins::registry::load_folder(&dir).unwrap();
    assert_eq!(r.loaded, ["org.photocraft.example.invert"]);
    assert_eq!(r.failed.len(), 1);
    assert!(r.failed[0].0.ends_with("b-broken.wasm"));
    assert!(photocraft_plugins::registry::load_folder(&dir.join("missing")).is_err());
    assert!(photocraft_plugins::registry::load_file(&dir, Limits::default()).is_err(), "a directory is not a module");
    std::fs::remove_dir_all(&dir).ok();
}
