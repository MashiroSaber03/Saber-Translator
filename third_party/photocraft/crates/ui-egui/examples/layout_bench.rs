//! Frame times of the whole app on a designer's layout document (#125, #128): ~175 layers in
//! nested groups (pass-through and isolated), 60 text layers in several fonts and sizes, stroked
//! shapes, smart objects, drop shadows / strokes / gradient overlays, adjustment layers and masks,
//! 4000×3000 8-bit RGB (`tests/support/layout_doc.rs`), saved as PSD and reopened through the real
//! import path.
//!
//! ```sh
//! cargo run --release -p photocraft-ui-egui --example layout_bench -- [--reps 9] [--json out.json] [--direct] [--navigator] [--size 4000x3000]
//! ```
//!
//! The real `PhotocraftApp` runs in an offscreen egui_kittest harness on wgpu (the GPU canvas
//! path, with the device the app requests). Interactions go through real input where the harness
//! can reach it (key shortcuts, pointer drags and clicks on the canvas) and otherwise through the
//! command a panel runs (`layer.select`, `layer.setProps`), then the frames that follow. Each row
//! is the median / p90 / max of a frame's time over the repetitions; `PHOTOCRAFT_GPU_SYNC` makes
//! the canvas wait for the GPU, so a frame's time includes its composite. Move drags report the
//! press, the pointer-move frames and the release (the commit) separately. `--direct` skips the
//! PSD round trip; `--navigator` also shows the Navigator panel; `--spin idle|select|tool|move-
//! pixel|move-text|move-shape|move-group|move-smart|move-headline` repeats one interaction for 20 s
//! for a sampling profiler.

#[path = "../tests/support/layout_doc.rs"]
mod layout_doc;

use std::time::Instant;

use egui::{Event, Key, Modifiers, PointerButton, Pos2};
use egui_kittest::kittest::Queryable;
use photocraft_doc::{LayerContent, LayerId};
use photocraft_ui_egui::canvas::ViewXform;
use photocraft_ui_egui::state::Tool;
use photocraft_ui_egui::{PhotocraftApp, Services};
use serde_json::{Value, json};

type H = egui_kittest::Harness<'static, PhotocraftApp>;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1).cloned())
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// One frame, timed.
fn frame(h: &mut H) -> f64 {
    let t = Instant::now();
    h.step();
    ms(t)
}

struct Rows {
    reps: usize,
    rows: Vec<(String, f64, f64, f64)>,
}

impl Rows {
    /// A row from frame times already taken.
    fn add(&mut self, name: &str, v: Vec<f64>) {
        let mut v: Vec<f64> = v.into_iter().filter(|t| t.is_finite()).collect();
        if v.is_empty() {
            return;
        }
        v.sort_by(f64::total_cmp);
        let (med, p90, max) = (v[v.len() / 2], v[(v.len() * 9 / 10).min(v.len() - 1)], v[v.len() - 1]);
        let flag = if med > 16.0 { "  <-- over 16 ms" } else { "" };
        println!("{name:<46} median {med:>8.2} ms   p90 {p90:>8.2}   max {max:>8.2}{flag}");
        self.rows.push((name.to_string(), med, p90, max));
    }

    /// `f` performs one interaction and returns the worst frame it took (ms).
    fn time(&mut self, name: &str, mut f: impl FnMut() -> f64) {
        let v = (0..self.reps).map(|_| f()).collect();
        self.add(name, v);
    }
}

/// `n` frames (returns the worst).
fn settle(h: &mut H, n: usize) -> f64 {
    (0..n).map(|_| frame(h)).fold(0.0, f64::max)
}

/// Screen position of document point (x, y) in the main canvas.
fn screen(h: &H, x: f64, y: f64) -> Pos2 {
    let app = h.state();
    let idx = app.session.active_index().unwrap_or(0);
    let v = &app.ui.views[idx];
    let rect = photocraft_ui_egui::rulers::content_rect(app, app.last_canvas_rect);
    ViewXform { rect, zoom: v.zoom, center: v.center, flip: app.ui.view.flip_horizontal, rotation: v.rotation, aspect: 1.0 }.to_screen(x as f32, y as f32)
}

fn select(h: &mut H, id: LayerId) {
    let _ = h.state_mut().run("layer.select", json!({"layer": id.0}));
}

/// A drag on the canvas from document point `from` by `d`, `steps` pointer moves one frame apart.
/// Returns the frame times: hover, press, the moves, release, one more.
fn drag(h: &mut H, from: [f64; 2], d: [f64; 2], steps: usize) -> Vec<f64> {
    let p0 = screen(h, from[0], from[1]);
    let mut times = Vec::new();
    h.event(Event::PointerMoved(p0));
    times.push(frame(h));
    h.event(Event::PointerButton { pos: p0, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    times.push(frame(h));
    for i in 1..=steps {
        let f = i as f64 / steps as f64;
        let p = screen(h, from[0] + d[0] * f, from[1] + d[1] * f);
        h.event(Event::PointerMoved(p));
        times.push(frame(h));
    }
    let p1 = screen(h, from[0] + d[0], from[1] + d[1]);
    h.event(Event::PointerButton { pos: p1, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    times.push(frame(h));
    times.push(frame(h));
    times
}

/// Rows for Move drags (frame times from [`drag`]): the press, the pointer moves, the release.
fn move_rows(rows: &mut Rows, label: &str, drags: &[Vec<f64>]) {
    let pick = |f: &dyn Fn(&[f64]) -> Vec<f64>| drags.iter().flat_map(|d| f(d)).collect::<Vec<f64>>();
    rows.add(&format!("Move {label}: press"), pick(&|d| d.get(1).copied().into_iter().collect()));
    rows.add(&format!("Move {label}: drag frames"), pick(&|d| d.get(2..d.len().saturating_sub(2)).unwrap_or_default().to_vec()));
    rows.add(&format!("Move {label}: release (commit)"), pick(&|d| d.len().checked_sub(2).and_then(|i| d.get(i)).copied().into_iter().collect()));
}

fn contains(l: &photocraft_doc::Layer, t: LayerId) -> bool {
    l.id == t || l.children().is_some_and(|c| c.iter().any(|k| contains(k, t)))
}

/// A point where `id` (or one of its layers, for a group) is the topmost layer with pixels.
fn point_on(h: &H, id: LayerId) -> Option<[f64; 2]> {
    let st = h.state().session.active()?;
    let l = st.doc.layer(id)?;
    let b = photocraft_engine::layer_multi_cmds::layer_bounds(l)?;
    let (cx, cy) = ((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2);
    for r in 0..60 {
        for (dx, dy) in [(0, 0), (r, 0), (-r, 0), (0, r), (0, -r)] {
            let (x, y) = (cx + dx * 3, cy + dy * 3);
            if photocraft_engine::pick_cmds::layers_at(&st.doc, x, y).first().is_some_and(|&t| contains(l, t)) {
                return Some([x as f64 + 0.5, y as f64 + 0.5]);
            }
        }
    }
    None
}

fn main() {
    reexec_with_gpu_sync();
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = arg(&args, "--reps").and_then(|v| v.parse().ok()).unwrap_or(9).max(3);
    let mut spec = layout_doc::Spec::full();
    if let Some((w, hh)) = arg(&args, "--size").and_then(|s| s.split_once('x').and_then(|(a, b)| Some((a.parse().ok()?, b.parse().ok()?)))) {
        (spec.width, spec.height) = (w, hh);
    }
    let t = Instant::now();
    let (doc, _) = layout_doc::build(spec);
    let (n, texts) = layout_doc::count(&doc);
    println!("built {}×{} layout: {n} layers, {texts} text layers in {:.0} ms", spec.width, spec.height, ms(t));
    let doc = if args.iter().any(|a| a == "--direct") {
        doc
    } else {
        let t = Instant::now();
        let psd = photocraft_io::export(&doc, "layout.psd", &Default::default()).expect("psd export").bytes;
        let t_save = ms(t);
        let t = Instant::now();
        let r = photocraft_io::import("layout.psd", &psd).expect("psd import");
        println!("PSD: {:.1} MB, save {t_save:.0} ms, open {:.0} ms ({} warnings)", psd.len() as f64 / 1e6, ms(t), r.warnings.len());
        // PSD export writes engine-made smart objects as pixels (no SoLd yet): convert them back,
        // as a Photoshop-made file would have them.
        let mut s = photocraft_engine::Session::new();
        s.open_document(r.document, None);
        let smart = |l: &photocraft_doc::Layer| l.name == "Logo" || (l.name.starts_with("Product ") && !l.name.contains('—') && !l.is_group());
        let ids: Vec<u64> = s.active().map(|st| st.doc.walk().iter().filter(|(_, _, l)| smart(l)).map(|(_, _, l)| l.id.0).collect()).unwrap_or_default();
        for id in ids {
            s.execute("layer.smartObjects.convertToSmartObject", json!({"layer": id})).expect("convert");
        }
        (*s.active().expect("doc").doc).clone()
    };
    // Layer ids change through the PSD round trip: find the layers by name and kind.
    let find = |name: &str, pred: &dyn Fn(&LayerContent) -> bool| -> LayerId {
        doc.walk().iter().find(|(_, _, l)| l.name == name && pred(&l.content)).map(|(_, _, l)| l.id).unwrap_or_else(|| panic!("no layer {name}"))
    };
    let pixel = find("Tile 1", &|c| matches!(c, LayerContent::Raster(_)));
    let text = find("Product 1 — Oak", &|c| matches!(c, LayerContent::Text(_)));
    let shape = find("Badge", &|c| matches!(c, LayerContent::Shape(_)));
    let group = find("Card 1", &|c| matches!(c, LayerContent::Group(_)));
    let smart = find("Product 1", &|c| matches!(c, LayerContent::Smart(_)));
    let hero = find("Hero", &|c| matches!(c, LayerContent::Group(_)));
    let headline = find("Make things that last", &|c| matches!(c, LayerContent::Text(_)));
    let products = find("Products", &|c| matches!(c, LayerContent::Group(_)));

    let navigator = args.iter().any(|a| a == "--navigator");
    let ppp: f32 = arg(&args, "--ppp").and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let t = Instant::now();
    let builder = egui_kittest::Harness::builder().with_size(egui::vec2(1600.0, 1000.0)).with_pixels_per_point(ppp);
    let mut h: H = builder.wgpu_setup(photocraft_ui_egui::gpu_canvas::wgpu_setup()).build_eframe(move |cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Services::default());
        if let Some(rs) = cc.wgpu_render_state.as_ref() {
            rs.device.set_device_lost_callback(|r, m| eprintln!("GPU device lost ({r:?}): {m}"));
            app.set_wgpu(rs.clone());
        }
        app.session.open_document(doc, Some("layout.psd".into()));
        app.sync_views();
        app.ui.panels.navigator = navigator;
        app
    });
    settle(&mut h, 4);
    {
        let p = &h.state().perf;
        println!(
            "open + first frames: {:.0} ms (canvas: gpu {}, fallback {:?}, first composite {} {:.0} ms)",
            ms(t),
            p.gpu,
            p.gpu_fallback,
            p.last_refresh,
            p.composite_ms
        );
    }
    settle(&mut h, 4);

    if let Some(dir) = arg(&args, "--shots") {
        shots(&mut h, &dir, ppp, [group, headline]);
        return;
    }
    if let Some(what) = arg(&args, "--spin") {
        spin(&mut h, &what, [pixel, text, shape, group, smart, headline]);
        return;
    }
    let mut rows = Rows { reps, rows: Vec::new() };
    rows.time("idle frame", || frame(&mut h));
    rows.time("hover over the canvas", || {
        let p = screen(&h, 1000.0, 1000.0);
        h.event(Event::PointerMoved(p));
        let a = frame(&mut h);
        h.event(Event::PointerMoved(p + egui::vec2(7.0, 3.0)));
        a.max(frame(&mut h))
    });

    // Selecting layers: the command a Layers panel row click runs, then the frames after (the
    // Layers and Properties panels follow the new layer).
    let picks = [pixel, text, shape, group, smart, hero, headline];
    let mut k = 0;
    rows.time("select a layer (Layers panel)", || {
        k += 1;
        let t = Instant::now();
        select(&mut h, picks[k % picks.len()]);
        (ms(t) + frame(&mut h)).max(frame(&mut h))
    });
    // A real click on a Layers panel row (rows carry their layer's name for accessibility).
    let names = ["Gallery", "Products", "Hero", "Header", "Footer"];
    rows.time("click a Layers panel row", || {
        k += 1;
        // The row is the widest node with the name (its disclosure triangle has it too).
        let row = h.query_all_by_label(names[k % names.len()]).map(|n| n.rect()).max_by(|a, b| a.width().total_cmp(&b.width()));
        let Some(p) = row.map(|r| r.center()) else { return f64::NAN };
        h.event(Event::PointerMoved(p));
        let a = frame(&mut h);
        h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
        let b = frame(&mut h);
        h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
        a.max(b).max(frame(&mut h)).max(frame(&mut h))
    });
    rows.time("open / close a group (Layers panel)", || {
        let t = Instant::now();
        let _ = h.state_mut().run("layer.setExpanded", json!({"layer": products.0}));
        (ms(t) + frame(&mut h)).max(frame(&mut h))
    });
    let tools = [Key::V, Key::B, Key::M, Key::T, Key::U, Key::E];
    rows.time("switch tool (shortcut)", || {
        k += 1;
        h.key_press(tools[k % tools.len()]);
        frame(&mut h).max(frame(&mut h))
    });
    let mut vis = false;
    for (label, id) in [("text layer", text), ("group", group)] {
        rows.time(&format!("toggle visibility ({label})"), || {
            vis = !vis;
            let t = Instant::now();
            let _ = h.state_mut().run("layer.setProps", json!({"layer": id.0, "visible": vis}));
            (ms(t) + frame(&mut h)).max(frame(&mut h))
        });
        let _ = h.state_mut().run("layer.setProps", json!({"layer": id.0, "visible": true}));
        settle(&mut h, 3);
    }
    let mut op = 1.0;
    select(&mut h, shape);
    settle(&mut h, 2);
    rows.time("opacity change (Layers / Properties)", || {
        op = if op > 0.6 { 0.5 } else { 0.9 };
        let t = Instant::now();
        let _ = h.state_mut().run("layer.setProps", json!({"layer": shape.0, "opacity": op}));
        (ms(t) + frame(&mut h)).max(frame(&mut h))
    });

    // Move tool drags. Auto-Select off: the selected layer moves.
    h.state_mut().ui.tool = Tool::Move;
    h.state_mut().ui.tool_options.move_auto_select = false;
    settle(&mut h, 3);
    for (label, id) in [("pixel", pixel), ("text", text), ("shape", shape), ("group", group), ("smart object", smart)] {
        select(&mut h, id);
        settle(&mut h, 3);
        let Some(from) = point_on(&h, id) else {
            println!("move {label}: no visible pixel found");
            continue;
        };
        // Back and forth, so every other drag returns the layer where it started.
        let mut commit = Vec::new();
        let drags: Vec<Vec<f64>> = (0..reps)
            .map(|i| {
                let d = drag(&mut h, from, if i.is_multiple_of(2) { [37.0, 23.0] } else { [-37.0, -23.0] }, 8);
                commit.push(h.state().perf.command_ms);
                d
            })
            .collect();
        move_rows(&mut rows, label, &drags);
        rows.add(&format!("  of which layer.translate ({label})"), commit);
    }
    // Auto-Select on: a click on the canvas selects the layer under the pointer; a press on a
    // type layer that isn't selected drags it.
    h.state_mut().ui.tool_options.move_auto_select = true;
    if let (Some(a), Some(b)) = (point_on(&h, text), point_on(&h, pixel)) {
        let mut i = 0u32;
        rows.time("click the canvas (auto-select a layer)", || {
            i += 1;
            let p = screen(&h, if i.is_multiple_of(2) { a[0] } else { b[0] }, if i.is_multiple_of(2) { a[1] } else { b[1] });
            h.event(Event::PointerMoved(p));
            let f0 = frame(&mut h);
            h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
            let f1 = frame(&mut h);
            h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
            let f2 = frame(&mut h);
            let f3 = frame(&mut h);
            f0.max(f1).max(f2).max(f3)
        });
    }
    if let Some(from) = point_on(&h, headline) {
        let drags: Vec<Vec<f64>> = (0..reps)
            .map(|i| {
                select(&mut h, pixel);
                settle(&mut h, 2);
                drag(&mut h, from, if i.is_multiple_of(2) { [25.0, 0.0] } else { [-25.0, 0.0] }, 8)
            })
            .collect();
        move_rows(&mut rows, "auto-select type", &drags);
    }

    println!("perf spans: {:?}", h.state().perf.spans);
    if let Some(out) = arg(&args, "--json") {
        let r: Vec<Value> = rows.rows.iter().map(|(n, med, p90, max)| json!({"name": n, "median_ms": med, "p90_ms": p90, "max_ms": max})).collect();
        let report = json!({"width": spec.width, "height": spec.height, "layers": n, "text_layers": texts, "rows": r});
        std::fs::write(&out, serde_json::to_string_pretty(&report).expect("json")).expect("write json");
    }
}

/// `--shots DIR`: screenshots of the whole app while a Move drag is under way and after the
/// release, fitted on screen and at 100 %, for a card group and the hero headline (`{DIR}/
/// move-{what}-{view}-{phase}@{ppp}x.png`).
fn shots(h: &mut H, dir: &str, ppp: f32, [group, headline]: [LayerId; 2]) {
    let _ = std::fs::create_dir_all(dir);
    let save = |h: &mut H, name: &str| match h.render() {
        Ok(img) => {
            let p = format!("{dir}/{name}@{ppp}x.png");
            match img.save(&p) {
                Ok(()) => println!("wrote {p}"),
                Err(e) => eprintln!("{p}: {e}"),
            }
        }
        Err(e) => eprintln!("{name}: render failed: {e}"),
    };
    h.state_mut().ui.tool = Tool::Move;
    h.state_mut().ui.tool_options.move_auto_select = false;
    for (what, id) in [("card", group), ("headline", headline)] {
        for view in ["fit", "100"] {
            select(h, id);
            if let Some(b) = h.state().session.active().and_then(|s| s.doc.layer(id)).and_then(photocraft_engine::layer_multi_cmds::layer_bounds) {
                let app = h.state_mut();
                let v = &mut app.ui.views[0];
                if view == "100" {
                    v.zoom = 1.0;
                    v.center = [(b.x0 + b.x1) as f32 / 2.0, (b.y0 + b.y1) as f32 / 2.0];
                    v.fit_pending = false;
                } else {
                    v.fit_pending = true;
                }
            }
            settle(h, 6);
            let Some(from) = point_on(h, id) else { continue };
            // Press and drag most of the way, render mid-drag, then release and render again.
            let d = [140.0, 60.0];
            let p0 = screen(h, from[0], from[1]);
            h.event(Event::PointerMoved(p0));
            frame(h);
            h.event(Event::PointerButton { pos: p0, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
            frame(h);
            for i in 1..=10 {
                let f = f64::from(i) / 10.0;
                h.event(Event::PointerMoved(screen(h, from[0] + d[0] * f, from[1] + d[1] * f)));
                frame(h);
            }
            save(h, &format!("move-{what}-{view}-dragging"));
            let p1 = screen(h, from[0] + d[0], from[1] + d[1]);
            h.event(Event::PointerButton { pos: p1, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
            settle(h, 4);
            save(h, &format!("move-{what}-{view}-released"));
            let _ = h.state_mut().run("edit.undo", json!({}));
            settle(h, 3);
        }
    }
}

/// `--spin`: one interaction over and over for 20 s, for a sampling profiler.
fn spin(h: &mut H, what: &str, [pixel, text, shape, group, smart, headline]: [LayerId; 6]) {
    let t = Instant::now();
    let mut k = 0usize;
    if let Some(target) = what.strip_prefix("move-") {
        let id = match target {
            "group" => group,
            "text" => text,
            "shape" => shape,
            "smart" => smart,
            "headline" => headline,
            _ => pixel,
        };
        h.state_mut().ui.tool = Tool::Move;
        select(h, id);
        settle(h, 3);
        let from = point_on(h, id).unwrap_or([1000.0, 1000.0]);
        while t.elapsed().as_secs() < 20 {
            k += 1;
            let s = if k.is_multiple_of(2) { 1.0 } else { -1.0 };
            drag(h, from, [30.0 * s, 20.0 * s], 30);
        }
        return;
    }
    if what == "click" {
        h.state_mut().ui.tool = Tool::Move;
        h.state_mut().ui.tool_options.move_auto_select = true;
        let pts: Vec<[f64; 2]> = [text, pixel].iter().filter_map(|&id| point_on(h, id)).collect();
        while t.elapsed().as_secs() < 20 && !pts.is_empty() {
            k += 1;
            let p = screen(h, pts[k % pts.len()][0], pts[k % pts.len()][1]);
            h.event(Event::PointerMoved(p));
            frame(h);
            h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
            frame(h);
            h.event(Event::PointerButton { pos: p, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
            settle(h, 2);
        }
        return;
    }
    while t.elapsed().as_secs() < 20 {
        k += 1;
        match what {
            "select" => select(h, [pixel, text, shape, group][k % 4]),
            "tool" => h.key_press([Key::V, Key::B, Key::M, Key::T][k % 4]),
            _ => {}
        }
        frame(h);
    }
}

/// Run again with `PHOTOCRAFT_GPU_SYNC=1` unless it is set (the workspace forbids the `unsafe`
/// that setting it in-process needs).
fn reexec_with_gpu_sync() {
    if std::env::var_os("PHOTOCRAFT_GPU_SYNC").is_some() {
        return;
    }
    let exe = std::env::current_exe().expect("exe");
    let status = std::process::Command::new(exe).args(std::env::args().skip(1)).env("PHOTOCRAFT_GPU_SYNC", "1").status().expect("re-exec");
    std::process::exit(status.code().unwrap_or(1));
}
