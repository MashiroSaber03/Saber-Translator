//! Compare our composite with Photoshop's merged image for one PSD (rendering-fidelity work).
//!
//! ```sh
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd [N]          # layers + first N pixels
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 png out.png # ours | Photoshop | diff heatmap
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 col [x]     # column samples
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 row y x0 x1 # row samples
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 worst [n]   # n worst pixels
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 grid x0 y0 x1 y1 [ch] # value grids
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 layerpx x y  # each layer's pixel
//! cargo run --release -p photocraft-io --example oracle_diff -- corpus/psd             # every file: max err, bad %, PASS/DIFF
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 dump prefix # raw f32 planes for offline fitting
//! cargo run --release -p photocraft-io --example oracle_diff -- file.psd 0 bylayer    # max error around each layer
//! SCALE=4 …                                                                         # upscale the png
//! HIDE_ADJ=1 …                                                                        # adjustment layers hidden
//! ONLY="name,name" …                                                                  # only these top-level layers (+ the bottom one)
//! DUMP_FX=1 …                                                                          # raw effects descriptors
//! ```
fn main() {
    let path = std::env::args().nth(1).expect("path");
    if std::path::Path::new(&path).is_dir() {
        corpus_summary(std::path::Path::new(&path));
        return;
    }
    let bytes = std::fs::read(&path).unwrap();
    let file = photocraft_psd::PsdFile::from_bytes(&bytes).unwrap();
    let mut imp = photocraft_io::import(&path, &bytes).unwrap();
    if std::env::var_os("HIDE_ADJ").is_some() {
        // Composite without adjustment layers (their input, for fitting transfer curves).
        fn hide(ls: &mut [photocraft_doc::Layer]) {
            for l in ls {
                if matches!(l.content, photocraft_doc::LayerContent::Adjustment(_)) {
                    l.visible = false;
                }
                if let photocraft_doc::LayerContent::Group(g) = &mut l.content {
                    hide(&mut g.children);
                }
            }
        }
        hide(&mut imp.document.layers);
    }
    if let Some(keep) = std::env::var_os("ONLY") {
        // Hide every top-level layer but the bottom one and those named in ONLY (comma-separated).
        let keep = keep.to_string_lossy().to_string();
        let names: Vec<&str> = keep.split(',').collect();
        for l in imp.document.layers.iter_mut().skip(1) {
            l.visible = names.contains(&l.name.as_str());
        }
    }
    let doc = &imp.document;
    println!("mode {:?} depth {:?} layers {} warnings {:?} light {:?}", doc.mode, doc.depth, doc.layer_count(), imp.warnings, doc.global_light);
    for l in doc.walk() {
        let l = l.2;
        println!(
            "  layer {:?} {:?} blend {:?} op {} fill {} fill_cache {} visible {}",
            l.name,
            l.content.kind_name(),
            l.blend,
            l.opacity,
            l.fill_opacity,
            l.fill_cache.is_some(),
            l.visible
        );
        if let photocraft_doc::LayerContent::Adjustment(a) = &l.content {
            println!("    {}", format!("{a:?}").chars().take(800).collect::<String>());
        }
        if let photocraft_doc::LayerContent::Fill(f) = &l.content {
            println!("    {f:?}");
        }
        if let Some(m) = &l.mask {
            println!("    mask default {:?} bounds {:?} enabled {}", m.surface.default_pixel(), m.surface.content_bounds(), m.enabled);
            let b = m.surface.content_bounds();
            let cx = (b.x0 + b.x1) / 2;
            let col: Vec<String> = (b.y0..b.y0 + 6).chain(b.y1 - 6..b.y1).map(|y| format!("{y}:{:.2}", m.surface.pixel(cx, y)[0])).collect();
            println!("    mask column x={cx}: {}", col.join(" "));
            let row: Vec<String> = (b.x0..b.x0 + 4).chain(b.x1 - 4..b.x1).map(|x| format!("{x}:{:.2}", m.surface.pixel(x, (b.y0 + b.y1) / 2)[0])).collect();
            println!("    mask row: {}", row.join(" "));
        }
        if let Some(sf) = l.surface() {
            let b = sf.content_bounds();
            let (mut x0, mut y0, mut x1, mut y1) = (i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for y in b.y0..b.y1 {
                for x in b.x0..b.x1 {
                    let p = sf.pixel(x, y);
                    if p[p.len() - 1] > 0.0 {
                        (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x + 1), y1.max(y + 1));
                    }
                }
            }
            println!("    alpha bounds ({x0},{y0})-({x1},{y1})");
        }
        if let photocraft_doc::LayerContent::Shape(sh) = &l.content {
            println!(
                "    shape path bounds {:?} subpaths {} rule {:?} inverted {} fill {} stroke {} outline {}",
                sh.path.control_bounds(),
                sh.path.subpaths.len(),
                sh.path.fill_rule,
                sh.path.inverted,
                sh.fill.is_some(),
                sh.stroke.is_some(),
                photocraft_compose::effect_outline(l).is_some()
            );
            if std::env::var_os("PATHS").is_some() {
                for s in &sh.path.subpaths {
                    println!("      subpath {:?} closed {} knots {:?}", s.op, s.closed, s.knots.iter().map(|k| (k.anchor.x, k.anchor.y)).collect::<Vec<_>>());
                }
            }
        }
        if let Some(vm) = &l.vector_mask {
            println!("    vector mask bounds {:?}", vm.path.control_bounds());
        }
        println!("    clipped {} frame {:?} bounds {:?}", l.clipped, photocraft_compose::fill_frame(l, doc.bounds()), l.surface().map(|s| s.content_bounds()));
        for e in &l.effects.items {
            if let photocraft_doc::Effect::GradientOverlay { common, gradient, .. } = e
                && common.enabled
            {
                let g = format!("{gradient:?}");
                println!("    gradient overlay: {g}");
            } else if format!("{e:?}").contains("enabled: true") {
                println!("    fx {}", format!("{e:?}").chars().take(if std::env::var_os("FULL").is_some() { 100_000 } else { 200 }).collect::<String>());
            }
        }
        println!("    blocks {:?}", l.psd_blocks.iter().map(|(k, v)| format!("{}({})", String::from_utf8_lossy(k), v.len())).collect::<Vec<_>>());
        for (k, v) in &l.psd_blocks {
            if k == b"lfx2" || k == b"lmfx" {
                // Recursively list descriptor keys (effects), skipping colour stop lists.
                fn walk(d: &photocraft_psd::descriptor::Descriptor, depth: usize, out: &mut Vec<String>) {
                    for (k, v) in &d.items {
                        let key = match k {
                            photocraft_psd::descriptor::Id::Code(c) => String::from_utf8_lossy(c).to_string(),
                            other => String::from_utf8_lossy(other.as_bytes()).to_string(),
                        };
                        let vs = format!("{v:?}");
                        match v {
                            photocraft_psd::descriptor::Value::Descriptor(sub) => {
                                out.push(format!("{}{key}:", "  ".repeat(depth)));
                                walk(sub, depth + 1, out);
                            }
                            _ => out.push(format!("{}{key} = {}", "  ".repeat(depth), vs.chars().take(90).collect::<String>())),
                        }
                    }
                }
                if let Ok((vd, _)) = photocraft_psd::descriptor::VersionedDescriptor::parse_prefix(&v[4..]) {
                    let mut out = Vec::new();
                    walk(&vd.descriptor, 3, &mut out);
                    for l in out.iter().filter(|l| !l.trim_start().starts_with("Clrs") && !l.trim_start().starts_with("Trns")) {
                        println!("{l}");
                    }
                }
            }
            if k == b"TySh"
                && let Some(Ok((vd, _))) = v.get(52..).map(photocraft_psd::descriptor::VersionedDescriptor::parse_prefix)
            {
                // Text bounds (text space) and the transform.
                let t: Vec<f64> = (0..6).map(|i| f64::from_be_bytes(v[2 + i * 8..10 + i * 8].try_into().unwrap())).collect();
                println!("    TySh transform {t:?}");
                for key in ["bounds", "boundingBox"] {
                    if let Some(photocraft_psd::descriptor::Value::Descriptor(d)) = vd.descriptor.get(key) {
                        let n: Vec<String> = d.items.iter().map(|(k, v)| format!("{}={v:?}", String::from_utf8_lossy(k.as_bytes()))).collect();
                        println!("    TySh {key}: {}", n.join(" "));
                    }
                }
            }
            if k == b"GdFl" {
                // Skip the 4-byte version before the descriptor.
                if let Ok((vd, _)) = photocraft_psd::descriptor::VersionedDescriptor::parse_prefix(v) {
                    println!(
                        "    GdFl keys: {:?}",
                        vd.descriptor
                            .items
                            .iter()
                            .filter(|(k, _)| !format!("{k:?}").contains("71, 114, 97, 100"))
                            .map(|(k, v)| format!(
                                "{:?}={}",
                                match k {
                                    photocraft_psd::descriptor::Id::Code(c) => String::from_utf8_lossy(c).to_string(),
                                    other => format!("{other:?}"),
                                },
                                format!("{v:?}").chars().take(120).collect::<String>()
                            ))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
    if std::env::var_os("DUMP_FX").is_some() {
        fn walk(d: &photocraft_psd::descriptor::Descriptor, depth: usize) {
            for (k, v) in &d.items {
                let key = String::from_utf8_lossy(k.as_bytes()).to_string();
                if key == "Clrs" || key == "Trns" {
                    continue;
                }
                match v {
                    photocraft_psd::descriptor::Value::Descriptor(sub) => {
                        println!("{}{key}:", "  ".repeat(depth));
                        walk(sub, depth + 1);
                    }
                    photocraft_psd::descriptor::Value::Text(t) => println!("{}{key} = {:?}", "  ".repeat(depth), t.to_string_lossy()),
                    photocraft_psd::descriptor::Value::List(items) if items.iter().all(|i| matches!(i, photocraft_psd::descriptor::Value::Descriptor(_))) => {
                        println!("{}{key} = [{} items]", "  ".repeat(depth), items.len());
                        for (i, it) in items.iter().enumerate() {
                            if let photocraft_psd::descriptor::Value::Descriptor(sub) = it {
                                println!("{}[{i}]:", "  ".repeat(depth + 1));
                                walk(sub, depth + 2);
                            }
                        }
                    }
                    _ => println!("{}{key} = {}", "  ".repeat(depth), format!("{v:?}").chars().take(600).collect::<String>()),
                }
            }
        }
        for rec in file.layers() {
            println!("-- layer {:?} rect {:?}", String::from_utf8_lossy(&rec.name), rec.rect);
            if let Some(b) = rec.block(b"lmfx").or(rec.block(b"lfx2")).or(rec.block(b"lfxs"))
                && let Ok((vd, _)) = photocraft_psd::descriptor::VersionedDescriptor::parse_prefix(&b.data[4..])
            {
                walk(&vd.descriptor, 1);
            }
        }
    }
    let ours = photocraft_compose::flatten(doc).px;
    let merged = photocraft_io::merged_composite(&file).unwrap();
    if std::env::args().nth(3).as_deref() == Some("png") {
        // ours | photoshop | diff heatmap, side by side (over white).
        let out = std::env::args().nth(4).expect("out.png");
        let (w, h) = (doc.size.width as usize, doc.size.height as usize);
        let mut img = vec![0u8; w * 3 * h * 4];
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (ours[y * w + x], merged[y * w + x]);
                let over = |p: [f32; 4]| [p[0] * p[3] + 1.0 - p[3], p[1] * p[3] + 1.0 - p[3], p[2] * p[3] + 1.0 - p[3]];
                let (oa, ob) = (over(a), over(b));
                let d = (0..3).map(|c| (oa[c] - ob[c]).abs()).fold(0.0f32, f32::max);
                let heat = [q((d * 4.0).min(1.0)), q((1.0 - d * 4.0).max(0.0) * 0.3), 0];
                for (k, px) in [[q(oa[0]), q(oa[1]), q(oa[2])], [q(ob[0]), q(ob[1]), q(ob[2])], heat].iter().enumerate() {
                    let o = (y * w * 3 + k * w + x) * 4;
                    img[o..o + 3].copy_from_slice(px);
                    img[o + 3] = 255;
                }
            }
        }
        // SCALE=n: nearest-neighbour upscale (small files).
        let k: usize = std::env::var("SCALE").ok().and_then(|s| s.parse().ok()).unwrap_or(1).clamp(1, 16);
        let (img, w, h) = if k > 1 {
            let mut big = vec![0u8; w * 3 * k * h * k * 4];
            for y in 0..h * k {
                for x in 0..w * 3 * k {
                    let (s, d) = (((y / k) * w * 3 + x / k) * 4, (y * w * 3 * k + x) * 4);
                    big[d..d + 4].copy_from_slice(&img[s..s + 4]);
                }
            }
            (big, w * k, h * k)
        } else {
            (img, w, h)
        };
        let image =
            photocraft_codecs::Image::from_raw((w * 3) as u32, h as u32, photocraft_codecs::ChannelLayout::Rgba, photocraft_codecs::SampleType::U8, img)
                .unwrap();
        std::fs::write(&out, photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).unwrap()).unwrap();
        println!("wrote {out}");
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("dump") {
        // dump prefix: raw little-endian f32 RGBA (straight) planes for offline fitting:
        // prefix_ours.f32, prefix_ps.f32, prefix_L<i>.f32 (each top-level layer's own pixels,
        // masks not applied) and prefix.txt (width, height, layer names).
        let out = std::env::args().nth(4).expect("prefix");
        let wr = |name: String, px: &[[f32; 4]]| {
            let bytes: Vec<u8> = px.iter().flat_map(|p| p.iter().flat_map(|v| v.to_le_bytes())).collect();
            std::fs::write(name, bytes).unwrap();
        };
        wr(format!("{out}_ours.f32"), &ours);
        wr(format!("{out}_ps.f32"), &merged);
        let mut meta = format!("{} {}\n", doc.size.width, doc.size.height);
        for (i, l) in doc.layers.iter().enumerate() {
            let b = photocraft_compose::surface_to_buffer(
                l.surface().unwrap_or(&photocraft_raster::Surface::new(photocraft_color::PixelFormat::RGBA8)),
                doc.bounds(),
            );
            wr(format!("{out}_L{i}.f32"), &b.px);
            meta += &format!("{i} {}\n", l.name);
        }
        std::fs::write(format!("{out}.txt"), meta).unwrap();
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("layerpx") {
        // layerpx x y: each layer's own pixel (0-255).
        let x: i32 = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let y: i32 = std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(0);
        for l in doc.walk() {
            if let Some(s) = l.2.surface() {
                println!("{:?}: {:?}", l.2.name, s.pixel(x, y).iter().map(|v| (v * 255.0 * 100.0).round() / 100.0).collect::<Vec<_>>());
            }
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("grid") {
        // grid x0 y0 x1 y1 [channel]: ours / photoshop values (0-255, premultiplied colour).
        let w = doc.size.width as usize;
        let a: Vec<usize> = (4..8).map(|i| std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(0)).collect();
        let ch: usize = std::env::args().nth(8).and_then(|s| s.parse().ok()).unwrap_or(3);
        for (label, img) in [("ours", &ours), ("ps", &merged)] {
            println!("{label} (channel {ch}):");
            for y in a[1]..a[3] {
                let row: Vec<String> = (a[0]..a[2])
                    .map(|x| {
                        let p = img[y * w + x];
                        let v = if ch == 3 { p[3] } else { p[ch] * p[3] };
                        format!("{:4}", (v * 255.0).round())
                    })
                    .collect();
                println!("  y={y:4} {}", row.join(""));
            }
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("bylayer") {
        // Max error and bad-pixel count inside each layer's bounds grown by its effects' reach
        // (regions overlap; a quick way to tell which layer or effect is off).
        let (w, h) = (doc.size.width as i32, doc.size.height as i32);
        for (_, _, l) in doc.walk() {
            let Some(sf) = l.surface() else { continue };
            let cb = sf.content_bounds();
            let mut b = photocraft_geom::Rect::new(i32::MAX, i32::MAX, i32::MIN, i32::MIN);
            for y in cb.y0..cb.y1 {
                for x in cb.x0..cb.x1 {
                    let p = sf.pixel(x, y);
                    if p[p.len() - 1] > 0.0 {
                        b = photocraft_geom::Rect::new(b.x0.min(x), b.y0.min(y), b.x1.max(x + 1), b.y1.max(y + 1));
                    }
                }
            }
            if b.x0 > b.x1 {
                continue;
            }
            let m = photocraft_compose::effects::margin(l);
            let r = photocraft_geom::Rect::new((b.x0 - m).max(0), (b.y0 - m).max(0), (b.x1 + m).min(w), (b.y1 + m).min(h));
            let (mut worst, mut at, mut bad) = (0.0f32, (0, 0), 0usize);
            for y in r.y0..r.y1 {
                for x in r.x0..r.x1 {
                    let i = (y * w + x) as usize;
                    let (a, b) = (ours[i], merged[i]);
                    let d = (0..3).map(|c| (a[c] * a[3] - b[c] * b[3]).abs()).fold((a[3] - b[3]).abs(), f32::max);
                    if d > 2.0 / 255.0 {
                        bad += 1;
                    }
                    if d > worst {
                        (worst, at) = (d, (x, y));
                    }
                }
            }
            let fx: Vec<&str> = l.effects.items.iter().filter(|e| e.enabled()).map(|e| e.label()).collect();
            println!("{:<32} {:>6.1} at {:?} bad {:>6}  {:?} {:?}", l.name, worst * 255.0, at, bad, l.blend, fx);
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("worst") {
        // The N worst pixels (premultiplied max channel error), with coordinates.
        let w = doc.size.width as usize;
        let n: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(20);
        let mut v: Vec<(f32, usize)> = ours
            .iter()
            .zip(&merged)
            .enumerate()
            .map(|(i, (a, b))| ((0..4).map(|c| (a[c] * a[3] - b[c] * b[3]).abs()).fold((a[3] - b[3]).abs(), f32::max), i))
            .collect();
        v.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (d, i) in v.into_iter().take(n) {
            let (a, b) = (ours[i], merged[i]);
            println!("({:4},{:4}) d={:5.1} ours {:?} ps {:?}", i % w, i / w, d * 255.0, a.map(|v| (v * 255.0).round()), b.map(|v| (v * 255.0).round()));
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("row") {
        let w = doc.size.width as usize;
        let y: usize = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(0);
        let (x0, x1): (usize, usize) =
            (std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(0), std::env::args().nth(6).and_then(|s| s.parse().ok()).unwrap_or(w));
        for x in (x0..x1).step_by(((x1 - x0) / 20).max(1)) {
            let (a, b) = (ours[y * w + x], merged[y * w + x]);
            println!("x={x:5} ours {:?} ps {:?}", a.map(|v| (v * 1000.0).round() / 1000.0), b.map(|v| (v * 1000.0).round() / 1000.0));
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("solve") {
        // solve layer-name x0 y0 x1 y1: per pixel the layer's content alpha `l`, its shape's path
        // coverage `cov`, Photoshop's alpha `A` and the alpha `s` an effect beneath the layer
        // would need (A = s + l (1 - s)) over a transparent backdrop.
        let name = std::env::args().nth(4).expect("layer name");
        let a: Vec<i32> = (5..9).map(|i| std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(0)).collect();
        let w = doc.size.width as usize;
        let l = doc.walk().into_iter().map(|t| t.2).find(|l| l.name == name).expect("layer");
        let r = photocraft_geom::Rect::new(a[0], a[1], a[2], a[3]);
        let cov = match &l.content {
            photocraft_doc::LayerContent::Shape(sh) => photocraft_vector::path_coverage(&sh.path, r),
            _ => vec![0.0; r.width() as usize * r.height() as usize],
        };
        for y in r.y0..r.y1 {
            for x in r.x0..r.x1 {
                let la = l.surface().map_or(0.0, |s| s.pixel(x, y)[3]);
                let c = cov[((y - r.y0) * r.width() as i32 + x - r.x0) as usize];
                let p = merged[y as usize * w + x as usize];
                let o = ours[y as usize * w + x as usize];
                let s = if la < 1.0 { (p[3] - la) / (1.0 - la) } else { f32::NAN };
                println!("({x:3},{y:3}) l {la:.3} cov {c:.3} A {:.3} ours {:.3} s {s:.3} ps {:?}", p[3], o[3], p.map(|v| (v * 255.0).round()));
            }
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("line") {
        // line x0 y0 x1 y1: every pixel on the segment, straight RGBA 0-255, ours | ps.
        let w = doc.size.width as usize;
        let a: Vec<i64> = (4..8).map(|i| std::env::args().nth(i).and_then(|s| s.parse().ok()).unwrap_or(0)).collect();
        let n = (a[2] - a[0]).abs().max((a[3] - a[1]).abs()).max(1);
        for k in 0..=n {
            let (x, y) = ((a[0] + (a[2] - a[0]) * k / n) as usize, (a[1] + (a[3] - a[1]) * k / n) as usize);
            let (o, p) = (ours[y * w + x], merged[y * w + x]);
            println!("({x:4},{y:4}) ours {:?} ps {:?}", o.map(|v| (v * 255.0).round()), p.map(|v| (v * 255.0).round()));
        }
        return;
    }
    if std::env::args().nth(3).as_deref() == Some("col") {
        let (w, h) = (doc.size.width as usize, doc.size.height as usize);
        let x = std::env::args().nth(4).and_then(|s| s.parse().ok()).unwrap_or(w / 2);
        for k in 0..=16 {
            let y = ((h - 1) * k / 16).min(h - 1);
            let (a, b) = (ours[y * w + x], merged[y * w + x]);
            println!("y={y:5} ours {:?} ps {:?}", a.map(|v| (v * 1000.0).round() / 1000.0), b.map(|v| (v * 1000.0).round() / 1000.0));
        }
        return;
    }
    let n = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8usize);
    for (i, (a, b)) in ours.iter().zip(&merged).enumerate().take(n) {
        println!("{i:4}: ours {:?}\n      ps   {:?}", a.map(|v| (v * 1000.0).round() / 1000.0), b.map(|v| (v * 1000.0).round() / 1000.0));
    }
}

/// The corpus oracle table (as `tests/corpus.rs`), files in parallel, without a test build.
fn corpus_summary(root: &std::path::Path) {
    fn collect(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(&p, out);
            } else if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("psd") || e.eq_ignore_ascii_case("psb")) {
                out.push(p);
            }
        }
    }
    let mut files = Vec::new();
    collect(root, &mut files);
    files.sort();
    const TOL: f32 = 2.0 / 255.0;
    type Row = (String, Option<(f32, f32)>);
    let row = |p: &std::path::PathBuf| -> Row {
        let name = p.strip_prefix(root).unwrap_or(p).display().to_string();
        let Ok(bytes) = std::fs::read(p) else { return (name, None) };
        let Ok(file) = photocraft_psd::PsdFile::from_bytes(&bytes) else { return (name, None) };
        let Ok(imp) = photocraft_io::import(&name, &bytes) else { return (name, None) };
        if file.has_real_merged_data() == Some(false) || file.layers().is_empty() {
            return (name, None);
        }
        let Ok(merged) = photocraft_io::merged_composite(&file) else { return (name, None) };
        let ours = photocraft_compose::flatten(&imp.document).px;
        let mut m = 0.0f32;
        let mut bad = 0usize;
        for (a, b) in ours.iter().zip(&merged) {
            let d = (0..4).map(|c| (a[c] * a[3] - b[c] * b[3]).abs()).fold((a[3] - b[3]).abs(), f32::max);
            m = m.max(d);
            if (0..4).any(|c| (a[c] * a[3] - b[c] * b[3]).abs() > TOL) {
                bad += 1;
            }
        }
        (name, Some((m, 100.0 * bad as f32 / ours.len().max(1) as f32)))
    };
    // Files in parallel (a few threads; flatten itself is tile-parallel).
    let next = std::sync::atomic::AtomicUsize::new(0);
    let mut rows: Vec<(usize, Row)> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..4)
            .map(|_| {
                sc.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some(p) = files.get(i) else { break };
                        out.push((i, row(p)));
                    }
                    out
                })
            })
            .collect();
        hs.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });
    rows.sort_by_key(|r| r.0);
    let rows: Vec<Row> = rows.into_iter().map(|r| r.1).collect();
    let (mut pass, mut diff, mut skip) = (0, 0, 0);
    for (name, r) in &rows {
        match r {
            Some((m, pct)) if *m <= TOL => {
                pass += 1;
                println!("{name:<60} {m:>9.4} {pct:>7.2}%  PASS");
            }
            Some((m, pct)) => {
                diff += 1;
                println!("{name:<60} {m:>9.4} {pct:>7.2}%  DIFF");
            }
            None => skip += 1,
        }
    }
    println!("{} files: {pass} pass, {diff} differ, {skip} skipped/errors", rows.len());
}
