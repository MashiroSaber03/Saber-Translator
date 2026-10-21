//! Content-Aware family: timings of the perf scenarios P21, P22 and P49 (the same documents and
//! steps as `photocraft-ui-egui`'s `perf_scenarios`, without the canvas refresh), and a hash of
//! every tool's output, to check that a speed-up leaves the result bit for bit as it was.
//!
//! ```text
//! cargo run --release -p photocraft-engine --example content_aware_bench -- bench caf|heal|spot|cancel|cancelscan|full24 [reps]
//! cargo run --release -p photocraft-engine --example content_aware_bench -- hash|hash-big
//! ```
//!
//! `bench` prints one line per sample (`<scenario> <ms>`), so runs of two builds can be
//! interleaved and pooled. `hash-big` hashes the benchmarks' own fills (10 and 24 MP). `hash` prints `<case> <blake3 of the output>` for Content-Aware Fill
//! (every sampling and adaptation option), Edit › Fill Content-Aware, Delete and Fill, Spot
//! Healing, Patch and Content-Aware Move at 8, 16 and 32 bits, and for the algorithms called
//! directly on synthetic buffers (1, 3 and 4 channels).
use std::time::{Duration, Instant};

use photocraft_algo::content_aware::{self, FillOptions};
use photocraft_algo::inpaint::{self, Adapt, CompleteParams};
use photocraft_engine::Session;
use photocraft_engine::jobs::{JobId, Started};
use photocraft_geom::Rect;
use serde_json::{Value, json};

type Res<T> = Result<T, String>;

fn exec(s: &mut Session, id: &str, p: Value) -> Res<Value> {
    s.execute(id, p).map_err(|e| format!("{id}: {e}"))
}

/// Wall-clock milliseconds (mod 100 s), to line samples up with a trace.
fn epoch_ms() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0.0, |d| d.as_secs_f64() * 1000.0) % 100000.0
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// `perf_scenarios`' photo: gradients, a soft blob, fine noise and a high-contrast disc, through
/// JPEG so it opens like a real photo.
fn open_photo(w: u32, h: u32) -> Res<Session> {
    use rayon::prelude::*;
    let mut px = vec![0u8; w as usize * h as usize * 3];
    let (cx, cy, r) = (w as f32 * 0.55, h as f32 * 0.5, w.min(h) as f32 / 3.0);
    px.par_chunks_mut(w as usize * 3).enumerate().for_each(|(y, row)| {
        for x in 0..w as usize {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % 997) as f32 / 997.0 - 0.5;
            let blob = (-(((fx - 0.4) * 3.0).powi(2) + ((fy - 0.55) * 4.0).powi(2))).exp();
            let inside = ((x as f32 - cx).powi(2) + (y as f32 - cy).powi(2)).sqrt() < r;
            let (r0, g0, b0) = if inside { (0.85, 0.3 + 0.2 * (fx * 40.0).sin(), 0.15) } else { (0.35, 0.45, 0.55) };
            let rgb = [r0 + 0.1 * blob + n * 0.08, g0 + 0.1 * blob + n * 0.08, b0 + 0.2 * fy + n * 0.08];
            for (c, v) in rgb.into_iter().enumerate() {
                row[x * 3 + c] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    });
    let img =
        photocraft_codecs::Image::from_raw(w, h, photocraft_codecs::ChannelLayout::Rgb, photocraft_codecs::SampleType::U8, px).map_err(|e| e.to_string())?;
    let jpeg = photocraft_codecs::encode(&img, photocraft_codecs::Format::Jpeg, &Default::default()).map_err(|e| e.to_string())?;
    let doc = photocraft_io::import("photo.jpg", &jpeg).map_err(|e| e.to_string())?.document;
    let mut s = Session::new();
    s.open_document(doc, Some("photo.jpg".into()));
    Ok(s)
}

/// Minor page faults of this process so far (Linux; 0 elsewhere).
fn minor_faults() -> u64 {
    std::fs::read_to_string("/proc/self/stat")
        .ok()
        .and_then(|st| st.rsplit(')').next().and_then(|rest| rest.split_whitespace().nth(7)?.parse().ok()))
        .unwrap_or(0)
}

fn start_job(s: &mut Session, id: &str, p: Value) -> Res<JobId> {
    match s.start(id, p).map_err(|e| format!("{id}: {e}"))? {
        Started::Job(j) => Ok(j),
        Started::Done(_) => Err(format!("{id} did not run as a background job")),
    }
}

fn wait_job(s: &mut Session, id: JobId) -> Res<()> {
    let t = Instant::now();
    loop {
        if s.poll_jobs().iter().any(|e| e.id == id) {
            return Ok(());
        }
        if t.elapsed().as_secs() > 120 {
            return Err("job did not finish in 120 s".into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn bench(which: &str, reps: usize) -> Res<()> {
    match which {
        // P21: Content-Aware Fill 10 MP (200 px hole).
        "caf" => {
            let (w, h) = (3872u32, 2592u32);
            let mut s = open_photo(w, h)?;
            for i in 1..=reps {
                exec(&mut s, "select.rect", json!({"x": w / 3 + i as u32 * 40, "y": h / 3, "width": 200, "height": 200, "ellipse": true}))?;
                let t = Instant::now();
                exec(&mut s, "edit.contentAwareFill", json!({"seed": i}))?;
                println!("caf {:.3}", ms(t));
                exec(&mut s, "edit.undo", json!({}))?;
                exec(&mut s, "select.deselect", json!({}))?;
            }
        }
        // P22: Spot Healing stroke 10 MP.
        "heal" => {
            let (w, h) = (3872u32, 2592u32);
            let mut s = open_photo(w, h)?;
            for i in 1..=reps {
                let (x, y) = (w / 4 + i as u32 * 60, h / 4);
                let pts: Vec<Value> = (0..10).map(|k| json!([x + k * 12, y + k * 4])).collect();
                let t = Instant::now();
                exec(&mut s, "paint.spotHealing", json!({"points": pts, "size": 60}))?;
                println!("heal {:.3}", ms(t));
            }
        }
        // P49: cancel a Content-Aware Fill job on 24 MP, early, midway and late.
        "cancel" => {
            let (w, h) = (6000u32, 4000u32);
            let mut s = open_photo(w, h)?;
            exec(&mut s, "select.rect", json!({"x": w / 3, "y": h / 3, "width": w / 5, "height": h / 4}))?;
            for i in 1..=reps {
                let job = start_job(&mut s, "edit.contentAwareFill", json!({}))?;
                let after = 20 + (i as u64 * 97) % 700;
                std::thread::sleep(Duration::from_millis(after));
                let t = Instant::now();
                s.cancel_job(job);
                s.join_cancelled_jobs();
                println!("cancel {:.3} (after {after} ms)", ms(t));
                wait_job(&mut s, job)?;
            }
        }
        // P49's job cancelled every `reps` ms through its run: where cancelling waits longest.
        "cancelscan" => {
            let (w, h) = (6000u32, 4000u32);
            let mut s = open_photo(w, h)?;
            exec(&mut s, "select.rect", json!({"x": w / 3, "y": h / 3, "width": w / 5, "height": h / 4}))?;
            let env = |k: &str, d: u64| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
            let (mut after, until) = (env("CAF_SCAN_FROM", 10), env("CAF_SCAN_TO", 2500));
            while after < until {
                let job = start_job(&mut s, "edit.contentAwareFill", json!({}))?;
                std::thread::sleep(Duration::from_millis(after));
                let t = Instant::now();
                if std::env::var_os("CAF_TRACE").is_some() {
                    eprintln!("CANCEL @{:.1}", epoch_ms());
                }
                s.cancel_job(job);
                s.join_cancelled_jobs();
                println!("cancelscan {:.3} (after {after} ms)", ms(t));
                if std::env::var_os("CAF_TRACE").is_some() {
                    eprintln!("JOINED @{:.1}", epoch_ms());
                }
                wait_job(&mut s, job)?;
                after += reps.max(1) as u64;
            }
        }
        // `retouch_bench`'s Spot Healing dabs (sizes 50, 100, 200, 300; Content-Aware, or the type
        // in `CAF_SPOT_TYPE`) on its 6016² document, `reps` rounds.
        "spot" => {
            let side = 6016u32;
            let mut s = Session::new();
            exec(&mut s, "file.new", json!({"width": side, "height": side, "depth": 8}))?;
            let area = Rect::new(2000, 2000, 4048, 4048);
            s.edit("texture", |doc, active| {
                let Some(surf) = active.and_then(|id| doc.layer_mut(id)).and_then(|l| l.surface_mut()) else { return Ok(()) };
                let fmt = surf.format();
                let mut data = Vec::with_capacity(area.width() as usize * area.height() as usize * 4);
                for y in area.y0..area.y1 {
                    for x in area.x0..area.x1 {
                        let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) as u32 % 1000) as f32 / 1000.0 * 0.08 - 0.04;
                        let blemish = (x % 400 - 200).abs() < 8 && (y % 400 - 200).abs() < 8;
                        let v = if blemish {
                            [0.05, 0.05, 0.05, 1.0]
                        } else {
                            [0.3 + 0.0002 * x as f32 + n, 0.4 + ((x + y) as f32 * 0.01).sin() * 0.1 + n, 0.5 + n, 1.0]
                        };
                        data.extend(photocraft_raster::from_rgba(&fmt, v));
                    }
                }
                surf.write_region(area, &data);
                Ok(())
            })
            .map_err(|e| e.to_string())?;
            // Comma-separated types, run in turn (`CAF_SPOT_TYPE`, default Content-Aware).
            let kinds = std::env::var("CAF_SPOT_TYPE").unwrap_or_else(|_| "contentAware".into());
            let mut k = 4u32;
            for _ in 0..reps {
                for kind in kinds.split(',') {
                    for size in [50.0, 100.0, 200.0, 300.0] {
                        k += 1;
                        let (x, y) = (2200 + (k % 4) * 400, 2200 + (k / 4 % 4) * 400);
                        let (t, f) = (Instant::now(), minor_faults());
                        exec(&mut s, "paint.spotHealing", json!({"points": [[x, y]], "size": size, "hardness": 50, "type": kind}))?;
                        println!("{}{size} {:.3} faults {}", &kind[..4], ms(t), minor_faults().saturating_sub(f));
                    }
                }
            }
        }
        // The whole 24 MP fill, uncancelled (how long P49's job runs).
        "full24" => {
            let (w, h) = (6000u32, 4000u32);
            let mut s = open_photo(w, h)?;
            exec(&mut s, "select.rect", json!({"x": w / 3, "y": h / 3, "width": w / 5, "height": h / 4}))?;
            for _ in 1..=reps {
                let t = Instant::now();
                exec(&mut s, "edit.contentAwareFill", json!({}))?;
                println!("full24 {:.3}", ms(t));
                exec(&mut s, "edit.undo", json!({}))?;
            }
        }
        o => return Err(format!("unknown scenario `{o}` (caf|heal|spot|cancel|cancelscan|full24)")),
    }
    Ok(())
}

fn hash_f32(v: &[f32]) -> String {
    let mut hs = blake3::Hasher::new();
    for x in v {
        hs.update(&x.to_bits().to_le_bytes());
    }
    hs.finalize().to_hex()[..16].to_string()
}

/// The active layer's pixels over the canvas.
fn layer_hash(s: &Session) -> Res<String> {
    let d = s.active().ok_or("no document")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let surf = d.doc.layer(id).and_then(|l| l.surface()).ok_or("no surface")?;
    Ok(hash_f32(&surf.read_region(d.doc.bounds())))
}

/// A textured document: gradients, stripes, a bright band, blobs and fine noise.
fn textured(w: u32, h: u32, depth: u64) -> Res<Session> {
    let mut s = Session::new();
    exec(&mut s, "file.new", json!({"width": w, "height": h, "depth": depth}))?;
    s.edit("texture", |doc, active| {
        let Some(surf) = active.and_then(|id| doc.layer_mut(id)).and_then(|l| l.surface_mut()) else { return Ok(()) };
        let fmt = surf.format();
        let area = Rect::new(0, 0, w as i32, h as i32);
        let mut data = Vec::with_capacity(w as usize * h as usize * 4);
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                let n = ((x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) as u32 % 1000) as f32 / 1000.0 * 0.08 - 0.04;
                let stripe = if (x / 9) % 2 == 0 { 0.1 } else { 0.0 };
                let band = if (x - y / 2).rem_euclid(400) < 40 { 0.25 } else { 0.0 };
                let blob = (-(((x as f32 / w as f32 - 0.6) * 4.0).powi(2) + ((y as f32 / h as f32 - 0.4) * 5.0).powi(2))).exp() * 0.2;
                let v = [0.25 + 0.0002 * x as f32 + stripe + band + n, 0.4 + ((x + y) as f32 * 0.02).sin() * 0.1 + blob + n, 0.5 + band * 0.5 - blob + n, 1.0];
                data.extend(photocraft_raster::from_rgba(&fmt, v.map(|c| c.clamp(0.0, 1.0))));
            }
        }
        surf.write_region(area, &data);
        Ok(())
    })
    .map_err(|e| e.to_string())?;
    Ok(s)
}

/// The tools on an 8-bit document converted to another colour mode.
fn mode_cases(mode: &str, out: &mut Vec<(String, String)>) -> Res<()> {
    let mut s = textured(600, 450, 8)?;
    exec(&mut s, &format!("image.mode.{mode}"), json!({}))?;
    let mut step = |s: &mut Session, name: &str, cmds: &[(&str, Value)]| -> Res<()> {
        for (id, p) in cmds {
            exec(s, id, p.clone())?;
        }
        out.push((format!("{mode}/{name}"), layer_hash(s)?));
        Ok(())
    };
    let ellipse = |x: u32, y: u32, sw: u32, sh: u32| ("select.rect", json!({"x": x, "y": y, "width": sw, "height": sh, "ellipse": true}));
    step(&mut s, "caf-auto", &[ellipse(250, 200, 70, 60), ("edit.contentAwareFill", json!({}))])?;
    step(&mut s, "caf-none", &[ellipse(400, 120, 50, 50), ("edit.contentAwareFill", json!({"colorAdaptation": "none", "sampling": "rectangular"}))])?;
    step(&mut s, "patch-ca", &[ellipse(120, 300, 50, 40), ("paint.patch", json!({"offset": [150, -80], "contentAware": true, "color": 4}))])?;
    step(&mut s, "move", &[ellipse(450, 320, 60, 50), ("paint.contentAwareMove", json!({"offset": [-90, -60]}))])?;
    exec(&mut s, "select.deselect", json!({}))?;
    step(&mut s, "spot", &[("paint.spotHealing", json!({"points": [[100, 100], [130, 110], [160, 125]], "size": 30}))])?;
    Ok(())
}

fn doc_cases(depth: u64, out: &mut Vec<(String, String)>) -> Res<()> {
    let (w, h) = (900u32, 700u32);
    let mut s = textured(w, h, depth)?;
    let mut step = |s: &mut Session, name: &str, cmds: &[(&str, Value)]| -> Res<()> {
        for (id, p) in cmds {
            exec(s, id, p.clone())?;
        }
        let hash = layer_hash(s)?;
        // A case that changes nothing checks nothing.
        let same = out.last().is_some_and(|(_, prev): &(String, String)| *prev == hash);
        out.push((format!("doc{depth}/{name}"), if same { format!("{hash} UNCHANGED") } else { hash }));
        Ok(())
    };
    let ellipse = |x: u32, y: u32, sw: u32, sh: u32| ("select.rect", json!({"x": x, "y": y, "width": sw, "height": sh, "ellipse": true}));
    step(&mut s, "caf-auto", &[ellipse(400, 300, 90, 70), ("edit.contentAwareFill", json!({}))])?;
    step(
        &mut s,
        "caf-rect-none",
        &[ellipse(150, 120, 60, 80), ("edit.contentAwareFill", json!({"sampling": "rectangular", "colorAdaptation": "none", "seed": 3}))],
    )?;
    step(&mut s, "caf-high-rot", &[ellipse(600, 450, 40, 40), ("edit.contentAwareFill", json!({"colorAdaptation": "high", "rotationAdaptation": "low"}))])?;
    step(
        &mut s,
        "caf-veryhigh-mirror-scale",
        &[
            ellipse(700, 150, 36, 30),
            ("edit.contentAwareFill", json!({"colorAdaptation": "veryHigh", "mirror": true, "scale": true, "rotationAdaptation": "full"})),
        ],
    )?;
    step(&mut s, "caf-custom", &[ellipse(300, 500, 50, 50), ("edit.contentAwareFill", json!({"sampling": "custom", "area": [100, 400, 200, 250]}))])?;
    // A wide hole: the finest levels only refine (the hole is wider than `refine_extent`).
    step(&mut s, "caf-big", &[("select.rect", json!({"x": 300, "y": 200, "width": 330, "height": 250})), ("edit.contentAwareFill", json!({"seed": 9}))])?;
    step(&mut s, "fill-ca", &[ellipse(200, 300, 50, 60), ("edit.fill", json!({"contents": "contentAware"}))])?;
    step(&mut s, "fill-ca-noadapt", &[ellipse(650, 550, 50, 40), ("edit.fill", json!({"contents": "contentAware", "colorAdaptation": false}))])?;
    step(&mut s, "delete-and-fill", &[ellipse(100, 600, 45, 45), ("edit.deleteAndFillSelection", json!({}))])?;
    exec(&mut s, "select.deselect", json!({}))?;
    for (k, kind) in ["contentAware", "proximityMatch", "createTexture"].into_iter().enumerate() {
        let (x, y) = (150 + 220 * k as u32, 250);
        let pts: Vec<Value> = (0..8).map(|j| json!([x + j * 10, y + j * 5])).collect();
        step(&mut s, &format!("spot-{kind}"), &[("paint.spotHealing", json!({"points": pts, "size": 40, "type": kind}))])?;
    }
    step(&mut s, "spot-dab", &[("paint.spotHealing", json!({"points": [[820, 640]], "size": 25}))])?;
    step(&mut s, "patch-normal", &[ellipse(350, 100, 60, 50), ("paint.patch", json!({"offset": [120, 60]}))])?;
    step(&mut s, "patch-ca", &[ellipse(500, 450, 70, 60), ("paint.patch", json!({"offset": [-150, -40], "contentAware": true, "color": 3}))])?;
    step(&mut s, "patch-ca-s1", &[ellipse(200, 150, 50, 50), ("paint.patch", json!({"offset": [100, 90], "contentAware": true, "structure": 1}))])?;
    step(&mut s, "move", &[ellipse(420, 380, 80, 60), ("paint.contentAwareMove", json!({"offset": [-120, 80]}))])?;
    step(&mut s, "move-color", &[ellipse(600, 200, 60, 60), ("paint.contentAwareMove", json!({"offset": [90, 110], "color": 5, "structure": 2}))])?;
    step(
        &mut s,
        "extend",
        &[ellipse(150, 450, 70, 50), ("paint.contentAwareMove", json!({"offset": [80, -60], "mode": "extend", "color": 10, "structure": 7}))],
    )?;
    // A feathered selection (partial coverage) and the fill onto a duplicate.
    step(&mut s, "caf-feather", &[ellipse(640, 330, 70, 50), ("select.modify.feather", json!({"radius": 6})), ("edit.contentAwareFill", json!({"seed": 4}))])?;
    step(&mut s, "caf-duplicate", &[ellipse(260, 380, 50, 40), ("edit.contentAwareFill", json!({"output": "duplicate"}))])?;
    // Last: the fill lands on a new layer, which becomes the active one.
    step(&mut s, "caf-margin-new", &[ellipse(500, 200, 70, 40), ("edit.contentAwareFill", json!({"sampling": "rectangular", "margin": 60, "output": "new"}))])?;
    Ok(())
}

/// Smooth gradient, stripes and deterministic noise, `ch` channels.
fn synth(w: usize, h: usize, ch: usize, seed: u64) -> Vec<f32> {
    let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..w * h)
        .flat_map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let n = (rng % 1000) as f32 / 1000.0 * 0.06 - 0.03;
            let stripe = if (i % w / 5).is_multiple_of(2) { 0.15 } else { 0.0 };
            (0..ch).map(move |c| (0.2 + 0.003 * x * (c as f32 + 1.0) / ch as f32 + 0.002 * y + stripe * (c % 2) as f32 + n).clamp(0.0, 1.0))
        })
        .collect()
}

fn algo_cases(out: &mut Vec<(String, String)>) {
    let disc = |w: usize, h: usize, cx: f32, cy: f32, rx: f32, ry: f32| -> Vec<bool> {
        (0..w * h).map(|i| (((i % w) as f32 - cx) / rx).hypot(((i / w) as f32 - cy) / ry) < 1.0).collect()
    };
    for &(w, h, ch) in &[(96usize, 80usize, 1usize), (160, 140, 3), (300, 260, 4), (520, 400, 3)] {
        let img = synth(w, h, ch, (w * h) as u64);
        let hole = disc(w, h, w as f32 * 0.45, h as f32 * 0.55, w as f32 * 0.12, h as f32 * 0.1);
        for (name, adapt) in [("none", Adapt::NONE), ("default", content_aware::color_level("default")), ("veryHigh", content_aware::color_level("veryHigh"))] {
            let p = CompleteParams { adapt, seed: 5, ..Default::default() };
            let r = inpaint::complete(w, h, ch, &img, &hole, &p);
            out.push((format!("complete/{w}x{h}x{ch}/{name}"), r.map_or("none".into(), |v| hash_f32(&v))));
        }
        let src: Vec<bool> = (0..w * h).map(|i| (i % w) < w * 3 / 4).collect();
        let opts = FillOptions { rotations: content_aware::rotation_level("medium"), mirror: true, scale: ch == 3, seed: 2, ..Default::default() };
        out.push((format!("fill/{w}x{h}x{ch}/variants"), hash_f32(&content_aware::fill(w, h, ch, &img, &hole, &src, &opts))));
        let auto = content_aware::auto_sampling(w, h, ch, &img, &hole);
        out.push((format!("auto/{w}x{h}x{ch}"), hash_f32(&auto.iter().map(|b| f32::from(u8::from(*b))).collect::<Vec<_>>())));
        out.push((format!("offset/{w}x{h}x{ch}"), format!("{:?}", inpaint::best_offset(w, h, ch, &img, &hole, 4, (w / 3) as i32))));
        out.push((format!("synth/{w}x{h}x{ch}"), hash_f32(&inpaint::synthesize(w, h, ch, &img, &hole, 12, 0x5eed))));
    }
    // A hole wider than `refine_extent` (refine-only levels) and a long thin one.
    let (w, h, ch) = (900usize, 640usize, 3usize);
    let img = synth(w, h, ch, 77);
    let big = disc(w, h, 450.0, 320.0, 160.0, 120.0);
    let p = CompleteParams { adapt: content_aware::color_level("default"), ..Default::default() };
    out.push(("complete/big/default".into(), inpaint::complete(w, h, ch, &img, &big, &p).map_or("none".into(), |v| hash_f32(&v))));
    let thin: Vec<bool> = (0..w * h).map(|i| (100..800).contains(&(i % w)) && (300..312).contains(&(i / w))).collect();
    out.push(("complete/thin/none".into(), inpaint::complete(w, h, ch, &img, &thin, &CompleteParams::default()).map_or("none".into(), |v| hash_f32(&v))));
    // Roles: a void gap and a known-but-not-source area.
    let (w, h, ch) = (200usize, 160usize, 3usize);
    let img = synth(w, h, ch, 3);
    let px: Vec<inpaint::Px> = (0..w * h)
        .map(|i| {
            let (x, y) = (i % w, i / w);
            if (80..120).contains(&x) && (60..100).contains(&y) {
                inpaint::Px::Hole
            } else if x < 12 || y > 150 {
                inpaint::Px::Void
            } else if x > 170 {
                inpaint::Px::Known
            } else {
                inpaint::Px::Source
            }
        })
        .collect();
    let p = CompleteParams { adapt: content_aware::color_level("high"), seed: 11, ..Default::default() };
    let r = inpaint::complete_masked(w, h, ch, &img, &px, &p, &photocraft_raster::Interrupt::NONE);
    out.push((
        "masked/roles/high".into(),
        match r {
            Ok(Some(v)) => hash_f32(&v),
            Ok(None) => "none".into(),
            Err(_) => "cancelled".into(),
        },
    ));
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("bench") => {
            let which = args.get(1).map_or("caf", String::as_str);
            let reps = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(8);
            bench(which, reps)
        }
        // The benchmarks' own documents (10 and 24 MP): the fills P21, P22 and P49 time.
        Some("hash-big") => {
            let (w, h) = (3872u32, 2592u32);
            let mut s = open_photo(w, h)?;
            for i in 1..=2u32 {
                exec(&mut s, "select.rect", json!({"x": w / 3 + i * 40, "y": h / 3, "width": 200, "height": 200, "ellipse": true}))?;
                exec(&mut s, "edit.contentAwareFill", json!({"seed": i}))?;
                println!("P21/{i} {}", layer_hash(&s)?);
                exec(&mut s, "select.deselect", json!({}))?;
                let (x, y) = (w / 4 + i * 60, h / 4);
                let pts: Vec<Value> = (0..10).map(|k| json!([x + k * 12, y + k * 4])).collect();
                exec(&mut s, "paint.spotHealing", json!({"points": pts, "size": 60}))?;
                println!("P22/{i} {}", layer_hash(&s)?);
            }
            let (w, h) = (6000u32, 4000u32);
            let mut s = open_photo(w, h)?;
            exec(&mut s, "select.rect", json!({"x": w / 3, "y": h / 3, "width": w / 5, "height": h / 4}))?;
            exec(&mut s, "edit.contentAwareFill", json!({}))?;
            println!("P49 {}", layer_hash(&s)?);
            Ok(())
        }
        Some("hash") => {
            let mut out = Vec::new();
            algo_cases(&mut out);
            for depth in [8, 16, 32] {
                doc_cases(depth, &mut out)?;
            }
            for mode in ["grayscale", "cmyk", "lab"] {
                mode_cases(mode, &mut out)?;
            }
            for (k, v) in out {
                println!("{k} {v}");
            }
            Ok(())
        }
        _ => Err("usage: content_aware_bench bench caf|heal|spot|cancel|cancelscan|full24 [reps] | hash | hash-big".into()),
    }
}
