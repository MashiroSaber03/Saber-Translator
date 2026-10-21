//! Font-size drag latency (#124): a large document with many type layers, one layer's size
//! changed step by step the way the options bar / Properties drag does it (one `type.setStyle`
//! per frame, all sharing one coalesce key), each followed by the CPU canvas refresh the app does
//! for it (`canvas::ensure_texture`, the path without a GPU or when the GPU compositor falls back).
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example type_bench -- [--size 6000x4000] [--layers 50] [--steps 30] [--effects] [--typing]
//! ```

use std::time::Instant;

use photocraft_ui_egui::{PhotocraftApp, Services};
use serde_json::json;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

/// `--typing`: per-keystroke cost of typing into a horizontal and a vertical (#199) Japanese type
/// layer: the edit command (re-layout and re-render), the Type tool's overlay layout and the CPU
/// canvas refresh, as the Type tool does it for each character.
fn typing(app: &mut PhotocraftApp, steps: usize) {
    let text: Vec<char> = "縦書きのテキストは、「右から左へ」進みます。PhotoCraft 2026年ー".chars().collect();
    for orient in ["horizontal", "vertical"] {
        let r = app.run("type.create", json!({"x": 3000, "y": 200, "text": "", "size": 48, "font": "Hiragino Sans"})).expect("type.create");
        let id = r["layer"].as_u64().unwrap_or(0);
        app.run(&format!("type.orientation.{orient}"), json!({"layer": id})).expect("orientation");
        app.sync_views();
        let ctx = egui::Context::default();
        let _ = photocraft_ui_egui::canvas::ensure_texture(app, &ctx, 0, None);
        let (mut cmd, mut lay, mut canvas, mut total) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for i in 0..steps {
            let c = text.get(i % text.len()).copied().unwrap_or('あ');
            let t0 = Instant::now();
            app.run("type.edit", json!({"layer": id, "replace": {"start": i, "end": i, "text": c.to_string()}, "coalesce": format!("typing-{orient}")}))
                .expect("type.edit");
            let t1 = Instant::now();
            let _ = photocraft_ui_egui::type_tool::layout(app, photocraft_doc::LayerId(id));
            let t2 = Instant::now();
            let _ = photocraft_ui_egui::canvas::ensure_texture(app, &ctx, 0, None);
            let t3 = Instant::now();
            cmd.push((t1 - t0).as_secs_f64() * 1e3);
            lay.push((t2 - t1).as_secs_f64() * 1e3);
            canvas.push((t3 - t2).as_secs_f64() * 1e3);
            total.push((t3 - t0).as_secs_f64() * 1e3);
        }
        let worst = total.iter().copied().fold(0.0, f64::max);
        println!(
            "{orient:>10} typing (median of {steps} keystrokes): command {:.2} ms, overlay layout {:.2} ms, canvas refresh {:.2} ms, total {:.2} ms (worst {worst:.2} ms)",
            median(cmd),
            median(lay),
            median(canvas),
            median(total),
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) =
        arg(&args, "--size").and_then(|s| s.split_once('x').and_then(|(a, b)| Some((a.parse::<u32>().ok()?, b.parse::<u32>().ok()?)))).unwrap_or((6000, 4000));
    let n: usize = arg(&args, "--layers").and_then(|s| s.parse().ok()).unwrap_or(50);
    let steps: usize = arg(&args, "--steps").and_then(|s| s.parse().ok()).unwrap_or(30);
    let effects = args.iter().any(|a| a == "--effects");
    // `--no-coalesce`: one history step per step, as the options bar did before #124.
    let coalesce = !args.iter().any(|a| a == "--no-coalesce");

    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
    app.run("file.new", json!({"width": w, "height": h})).expect("new document");
    if args.iter().any(|a| a == "--typing") {
        typing(&mut app, steps);
        return;
    }
    let t0 = Instant::now();
    let mut ids = Vec::new();
    for i in 0..n {
        let (x, y) = (40 + (i % 5) as u32 * (w / 5), 120 + (i / 5) as u32 * (h / (n as u32 / 5 + 1)).max(60));
        let r = app
            .run("type.create", json!({"x": x, "y": y, "text": format!("Layer {i}: the quick brown fox jumps"), "size": 24 + (i % 5) * 6}))
            .expect("type.create");
        ids.push(r["layer"].as_u64().unwrap_or(0));
        if effects {
            let _ = app.run("layer.layerStyle.dropShadow", json!({"layer": r["layer"], "distance": 8, "size": 10}));
        }
    }
    eprintln!("{w}×{h}, {n} type layers created in {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
    app.sync_views();
    let ctx = egui::Context::default();
    let t0 = Instant::now();
    let _ = photocraft_ui_egui::canvas::ensure_texture(&mut app, &ctx, 0, None);
    eprintln!("first canvas composite: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);

    let id = ids.get(n / 2).copied().unwrap_or(0);
    let (mut cmd, mut canvas, mut total) = (Vec::new(), Vec::new(), Vec::new());
    for s in 0..steps {
        let t0 = Instant::now();
        app.run("type.setStyle", {
            let mut p = json!({"layer": id, "size": 30.0 + s as f32 * 0.5});
            if coalesce {
                p["coalesce"] = json!("bench-size-drag");
            }
            p
        })
        .expect("type.setStyle");
        let t1 = Instant::now();
        let _ = photocraft_ui_egui::canvas::ensure_texture(&mut app, &ctx, 0, None);
        let t2 = Instant::now();
        cmd.push((t1 - t0).as_secs_f64() * 1e3);
        canvas.push((t2 - t1).as_secs_f64() * 1e3);
        total.push((t2 - t0).as_secs_f64() * 1e3);
    }
    // Undo steps back to the size before the drag.
    let size = |app: &PhotocraftApp| match app.session.active().and_then(|s| s.doc.layer(photocraft_doc::LayerId(id))).map(|l| &l.content) {
        Some(photocraft_doc::LayerContent::Text(t)) => t.size_pt,
        _ => 0.0,
    };
    let start = 24.0 + ((n / 2) % 5) as f32 * 6.0;
    let mut undos = 0;
    while size(&app) != start && undos < steps + 1 && app.session.undo() {
        undos += 1;
    }
    // `--json out.json` (`cargo xtask perf`): the per-step samples.
    if let Some(out) = arg(&args, "--json") {
        use photocraft_testkit::perf::{process_peak_rss_bytes, report, row, write_report};
        let rss = photocraft_testkit::perf::current_rss_bytes().max(process_peak_rss_bytes());
        let rows = vec![
            row("font size step: command", &cmd, rss, None),
            row("font size step: canvas refresh", &canvas, rss, None),
            row("font size step: total", &total, rss, None),
        ];
        let rep = report("type_bench", json!({"width": w, "height": h, "layers": n, "steps": steps, "effects": effects}), rows, None);
        if let Err(e) = write_report(&out, &rep) {
            eprintln!("{e}");
        }
    }
    println!(
        "size step (median of {steps}): command {:.2} ms, canvas refresh {:.2} ms, total {:.2} ms; undo steps for the drag: {undos}",
        median(cmd),
        median(canvas),
        median(total),
    );
}
