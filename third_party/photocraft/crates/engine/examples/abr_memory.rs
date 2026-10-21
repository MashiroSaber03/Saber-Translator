//! Memory of a large imported brush library (#1843): import `.abr` files into a session with a
//! preset store on disk, or reopen that store (a restart), then use a few brushes, printing the
//! process memory after each step. Run each phase as its own process so freed memory doesn't hide
//! what a fresh start costs. Generate a library with `photocraft-io`'s `gen_abr_library` example.
//!
//! ```text
//! cargo run --release -p photocraft-engine --example abr_memory -- import <abr dir> <store dir>
//! cargo run --release -p photocraft-engine --example abr_memory -- reopen <store dir>
//! ```
//!
//! Memory comes from PowerShell `Get-Process` on Windows and `/proc/self/status` elsewhere.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::time::Instant;

use photocraft_engine::Session;
use photocraft_engine::paint::{Pattern, TipShape};
use serde_json::json;

fn memory() -> String {
    #[cfg(windows)]
    {
        let pid = std::process::id();
        let cmd = format!("$p = Get-Process -Id {pid}; '{{0:N0}} MB working set, {{1:N0}} MB private' -f ($p.WorkingSet64/1MB), ($p.PrivateMemorySize64/1MB)");
        std::process::Command::new("powershell")
            .args(["-NoProfile", "-Command", &cmd])
            .output()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|e| e.to_string())
    }
    #[cfg(not(windows))]
    {
        std::fs::read_to_string("/proc/self/status")
            .unwrap_or_default()
            .lines()
            .filter(|l| l.starts_with("VmRSS") || l.starts_with("VmHWM"))
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Full-size tip bitmaps held by the session's preset library (count, bytes), and the bytes of
/// the previews of stored tips that aren't loaded.
fn library_tip_bytes(s: &Session) -> (usize, u64, u64) {
    let (mut n, mut bytes, mut previews) = (0, 0u64, 0u64);
    let mut add = |full: Option<&photocraft_engine::paint::GrayTile>, preview: Option<&photocraft_engine::paint::GrayTile>| {
        if let Some(g) = full {
            n += 1;
            bytes += g.data.len() as u64 * 2;
        }
        if let Some(g) = preview {
            previews += g.data.len() as u64 * 2;
        }
    };
    for p in &s.tools.presets {
        let b = &p.brush;
        for t in [&b.tip, &b.dual_brush.tip] {
            match t {
                TipShape::Sampled(g) => add(Some(g), None),
                TipShape::Stored(r) => add(r.full.as_deref(), Some(&r.preview)),
                TipShape::Round => {}
            }
        }
        match &b.texture.pattern {
            Pattern::Tile(g) => add(Some(g), None),
            Pattern::Stored(r) => add(r.full.as_deref(), Some(&r.preview)),
            Pattern::Procedural { .. } => {}
        }
    }
    (n, bytes, previews)
}

fn report(s: &Session, step: &str) {
    let (n, b, p) = library_tip_bytes(s);
    let cache = s.preset_store.as_ref().map(|st| st.tip_cache()).unwrap_or_default();
    println!(
        "{step}: {} | {} presets, {n} full tips in the library ({:.0} MB), previews {:.1} MB, tip cache {} tips {:.0} MB ({} decodes)",
        memory(),
        s.tools.presets.len(),
        b as f64 / 1048576.0,
        p as f64 / 1048576.0,
        cache.tips,
        cache.bytes as f64 / 1048576.0,
        cache.decodes
    );
}

fn use_brushes(s: &mut Session, count: usize) {
    let names: Vec<String> = s.tools.presets.iter().filter(|p| !p.builtin).map(|p| p.name.clone()).take(count).collect();
    s.execute("file.new", json!({"width": 800, "height": 600})).unwrap();
    let t = Instant::now();
    for (i, n) in names.iter().enumerate() {
        s.execute("tools.setBrush", json!({"preset": n})).unwrap();
        let y = 50.0 + i as f64 * 40.0;
        s.execute("paint.stroke", json!({"points": [[50.0, y], [400.0, y], [700.0, y + 20.0]], "brush": {"size": 60.0}})).unwrap();
    }
    println!("used {} brushes in {:.2} s", names.len(), t.elapsed().as_secs_f64());
    report(s, &format!("after using {} brushes", names.len()));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    println!("start: {}", memory());
    match args.get(1).map(String::as_str) {
        Some("import") => {
            let (abr, store) = (args.get(2).expect("abr dir"), args.get(3).expect("store dir"));
            let _ = std::fs::remove_dir_all(store);
            let mut s = Session::new();
            s.attach_preset_store(photocraft_engine::preset_store::open_dir(store));
            let mut files: Vec<_> = std::fs::read_dir(abr).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "abr")).collect();
            files.sort();
            let t = Instant::now();
            for f in &files {
                let r = s.execute("brush.presets.importAbr", json!({"path": f.to_string_lossy()})).unwrap();
                println!("imported {} ({} presets)", f.display(), r["count"]);
            }
            println!("import + store sync: {:.2} s", t.elapsed().as_secs_f64());
            for w in s.preset_store.as_mut().map(|st| st.take_warnings()).unwrap_or_default() {
                println!("warning: {w}");
            }
            report(&s, "after import");
            use_brushes(&mut s, 5);
        }
        Some("reopen") => {
            let store = args.get(2).expect("store dir");
            let t = Instant::now();
            let opened = photocraft_engine::preset_store::open_dir(store);
            let open_s = t.elapsed().as_secs_f64();
            let mut s = Session::new();
            let w = s.attach_preset_store(opened);
            println!("store open {open_s:.2} s, attach {:.2} s, {} warnings", t.elapsed().as_secs_f64() - open_s, w.len());
            for w in w.iter().take(5) {
                println!("warning: {w}");
            }
            report(&s, "after reopen");
            use_brushes(&mut s, 5);
        }
        _ => eprintln!("usage: abr_memory import <abr dir> <store dir> | reopen <store dir>"),
    }
}
