//! Filter › Render › Flame…, Picture Frame… and Tree…: procedural renderers
//! (`photocraft_algo::render2`). They draw onto the active pixel layer (or a
//! new layer with `newLayer: true`, or when the active layer has no pixels),
//! inside the selection, in any colour mode and depth, as one undo step.
//! Results report what was drawn (`layer`, `bounds`, `primitives`) so agents
//! can verify them without screenshots.

use photocraft_algo::render2::{self, FlameSpec, FrameSpec, Prim, TreeSpec};
use photocraft_doc::{Layer, LayerContent, LayerId};
use photocraft_geom::Rect;
use photocraft_raster::Surface;
use serde_json::{Map, Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// `[r, g, b(, a)]` (0–1) or `#rrggbb` → `[r, g, b, a]`.
fn parse_colour(v: &Value) -> Option<[f32; 4]> {
    match v {
        Value::Array(a) if a.len() >= 3 => {
            let c = |j: usize, d: f64| a.get(j).and_then(Value::as_f64).unwrap_or(d) as f32;
            Some([c(0, 0.0), c(1, 0.0), c(2, 0.0), c(3, 1.0)])
        }
        Value::String(h) if h.len() == 7 && h.starts_with('#') => {
            let ch = |j: usize| h.get(j..j + 2).and_then(|s| u8::from_str_radix(s, 16).ok()).map_or(0.0, |v| v as f32 / 255.0);
            Some([ch(1), ch(3), ch(5), 1.0])
        }
        _ => None,
    }
}

/// Params as an object with the named colour keys normalized to arrays (invalid ones dropped).
fn normalized(p: &Value, colours: &[&str]) -> Map<String, Value> {
    let mut m = p.as_object().cloned().unwrap_or_default();
    m.remove("layer");
    m.remove("newLayer");
    m.remove("path");
    for k in colours {
        if let Some(v) = m.remove(*k)
            && let Some(c) = parse_colour(&v)
        {
            m.insert((*k).to_string(), json!(c));
        }
    }
    m
}

/// Integer fields of a spec (`u32` in [`FlameSpec`], [`TreeSpec`], [`FrameSpec`]) given as JSON
/// floats: dialogs send every number as a float (`1.0`) and agents may too, which serde refuses
/// for a `u32` ("invalid type: floating point `1.0`, expected u32", #2501). Finite floats round to
/// the nearest integer, like [`crate::commands::int`]; a negative or too-large result is then
/// still refused by `spec_from`. Strings (names) and other types are left alone.
fn whole_numbers(m: &mut Map<String, Value>, keys: &[&str]) {
    for k in keys {
        if let Some(Value::Number(n)) = m.get(*k)
            && !(n.is_u64() || n.is_i64())
            && let Some(f) = n.as_f64().filter(|f| f.is_finite())
        {
            m.insert((*k).to_string(), json!(f.round() as i64));
        }
    }
}

fn spec_from<T: serde::de::DeserializeOwned>(m: Map<String, Value>) -> Result<T> {
    serde_json::from_value(Value::Object(m)).map_err(|e| EngineError::Other(format!("invalid parameters: {e}")))
}

/// Accepts a 1-based number or a name (camelCase) for an enumerated option.
fn index_param(m: &mut Map<String, Value>, k: &str, names: &[&str]) {
    if let Some(Value::String(n)) = m.get(k) {
        let lower = n.to_ascii_lowercase().replace([' ', '-', '_'], "");
        let i = names.iter().position(|s| s.to_ascii_lowercase().replace([' ', '-', '_'], "") == lower).map_or(1, |i| i + 1);
        m.insert(k.to_string(), json!(i));
    }
}

/// Flattened polylines of the path Flame follows: `path` (a saved path name, `"work"`), else the
/// work path, else the first saved path. `None` when the document has no path.
fn flame_paths(s: &Session, p: &Value) -> Option<Vec<Vec<(f32, f32)>>> {
    let d = s.active()?;
    let doc = &d.doc;
    let path = match p.get("path").and_then(Value::as_str) {
        Some(n) if !n.is_empty() && !n.eq_ignore_ascii_case("work") => doc.paths.iter().find(|sp| sp.name == n).map(|sp| &sp.path),
        _ => doc.work_path.as_ref().or_else(|| doc.paths.first().map(|sp| &sp.path)),
    }?;
    let polys: Vec<Vec<(f32, f32)>> = photocraft_vector::flatten_path(path, 0.5)
        .into_iter()
        .map(|pl| {
            let mut v: Vec<(f32, f32)> = pl.pts.iter().map(|&(x, y)| (x as f32, y as f32)).collect();
            if pl.closed
                && let Some(&f) = v.first()
            {
                v.push(f);
            }
            v
        })
        .filter(|v| !v.is_empty())
        .collect();
    (!polys.is_empty()).then_some(polys)
}

/// Draws `prims` onto the target layer as one undo step.
fn draw(s: &mut Session, label: &str, p: &Value, prims: Vec<Prim>, extra: Value) -> Result<Value> {
    let want_new = p.get("newLayer").and_then(Value::as_bool).unwrap_or(false);
    let target = p.get("layer").and_then(Value::as_u64).map(LayerId);
    let count = prims.len();
    let (layer, created, used) = s.edit(label, |doc, active| {
        let id = target.or(*active);
        let raster = id.and_then(|i| doc.layer(i)).is_some_and(|l| matches!(l.content, LayerContent::Raster(_)));
        let (id, created) = if want_new || !raster {
            let l = Layer::new(label.trim_end_matches('…'), LayerContent::Raster(Surface::new(doc.pixel_format())));
            let nid = doc.insert_above(*active, l);
            *active = Some(nid);
            (nid, true)
        } else {
            (id.ok_or(EngineError::Other("no active layer".into()))?, false)
        };
        let selection = doc.selection.clone();
        let mut clip = doc.bounds();
        if let Some(sel) = &selection {
            clip = clip.intersect(&sel.content_bounds());
        }
        let surf = crate::commands::paint_surface(doc, id, &Value::Null)?;
        let used = render2::composite(surf, &prims, clip, selection.as_ref());
        surf.prune();
        Ok((id, created, used))
    })?;
    let mut r = json!({
        "layer": layer.0,
        "newLayer": created,
        "bounds": if used.is_empty() { Value::Null } else { json!([used.x0, used.y0, used.x1, used.y1]) },
        "primitives": count,
    });
    if let (Value::Object(m), Value::Object(e)) = (&mut r, extra) {
        m.extend(e);
    }
    Ok(r)
}

fn canvas(s: &Session) -> Result<Rect> {
    Ok(s.active().ok_or(EngineError::NoDocument)?.doc.bounds())
}

fn run_flame(s: &mut Session, p: &Value) -> Result<Value> {
    let mut m = normalized(p, &["color"]);
    if !m.get("useCustomColor").and_then(Value::as_bool).unwrap_or(m.contains_key("color")) {
        m.remove("color");
    }
    m.remove("useCustomColor");
    if let Some(Value::Number(q)) = m.get("quality").cloned() {
        m.insert("quality".into(), json!(q.as_f64().unwrap_or(2.0).clamp(0.0, 4.0) as u32));
    } else if let Some(Value::String(q)) = m.get("quality").cloned() {
        let i = ["draft", "low", "medium", "high", "fine"].iter().position(|n| *n == q).unwrap_or(2);
        m.insert("quality".into(), json!(i));
    }
    whole_numbers(&mut m, &["seed"]);
    let spec: FlameSpec = spec_from(m)?;
    let found = flame_paths(s, p);
    let used_path = found.is_some();
    let paths = found.unwrap_or_else(|| {
        let c = canvas(s).unwrap_or(Rect::new(0, 0, 1, 1));
        if spec.flame_type == render2::FlameType::CandleLight {
            vec![vec![(c.x0 as f32 + c.width() as f32 * 0.5, c.y0 as f32 + c.height() as f32 * 0.75)]]
        } else {
            vec![render2::default_flame_path(c)]
        }
    });
    let prims = render2::flame(&spec, &paths);
    draw(s, "Flame…", p, prims, json!({ "usedPath": used_path }))
}

fn run_tree(s: &mut Session, p: &Value) -> Result<Value> {
    let mut m = normalized(p, &["leavesColor", "branchesColor"]);
    index_param(&mut m, "baseTreeType", &render2::tree_type_names());
    if m.contains_key("leavesColor") && !m.contains_key("defaultLeaves") {
        m.insert("defaultLeaves".into(), json!(false));
    }
    if m.contains_key("branchesColor") && !m.contains_key("customBranchColor") {
        m.insert("customBranchColor".into(), json!(true));
    }
    whole_numbers(&mut m, &["baseTreeType", "lightDirection", "seed"]);
    let spec: TreeSpec = spec_from(m)?;
    let prims = render2::tree(&spec, canvas(s)?);
    let name = render2::tree_type_names()[(spec.base_tree_type.clamp(1, 34) - 1) as usize];
    draw(s, "Tree…", p, prims, json!({ "treeType": name }))
}

fn run_frame(s: &mut Session, p: &Value) -> Result<Value> {
    let mut m = normalized(p, &["vineColor", "flowerColor", "leafColor"]);
    index_param(&mut m, "frame", &render2::FRAME_STYLES);
    whole_numbers(&mut m, &["frame", "lines", "seed"]);
    let spec: FrameSpec = spec_from(m)?;
    let prims = render2::picture_frame(&spec, canvas(s)?);
    let name = render2::FRAME_STYLES[(spec.frame.clamp(1, render2::FRAME_STYLES.len() as u32) - 1) as usize];
    draw(s, "Picture Frame…", p, prims, json!({ "frame": name }))
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// The Render commands of this module.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "filter.render.flame",
            label: "Flame…",
            menu: &["Filter", "Render"],
            shortcut: None,
            params: r##"{"flameType":"oneFlameAlongPath|multipleFlamesAlongPath|multipleFlamesPathDirections|multipleFlamesVariousLength|candleLight|multipleFlamesOneDirection","length":1..1000=150,"randomizeLength":bool,"width":1..1000=40,"angle":-180..180=0,"interval":1..1000=60,"adjustIntervalForLoops":bool=true,"useCustomColor":bool,"color":color,"turbulent":0..100=25,"jag":0..100=25,"opacity":0..100=75,"flameLines":1..100=20,"flameBottomAlignment":0..100=20,"flameStyle":"normal|violent|flat","flameShape":"parallel|toCenter|spread|oval|pointed","randomizeShapes":bool,"seed":u32=0,"quality":"draft|low|medium|high|fine","path":text,"newLayer":bool} → {layer,newLayer,bounds,primitives,usedPath}"##,
            enabled: has_doc,
            run: run_flame,
            journal: true,
        },
        CommandSpec {
            id: "filter.render.pictureFrame",
            label: "Picture Frame…",
            menu: &["Filter", "Render"],
            shortcut: None,
            params: r##"{"frame":"vineWithFlowers|vineWithLeaves|ivy|roses|daisies|berries|bamboo|waves|zigzag|dots|rope|scallops|doubleLine|snowflakes|stars|hearts","margin":0..30=4,"size":1..100=50,"arrangement":1..100=50,"lines":1..5=1,"thickness":1..100=30,"fade":0..100=0,"vineColor":color,"flowerColor":color,"leafColor":color,"seed":u32=0,"newLayer":bool} → {layer,newLayer,bounds,primitives,frame}"##,
            enabled: has_doc,
            run: run_frame,
            journal: true,
        },
        CommandSpec {
            id: "filter.render.tree",
            label: "Tree…",
            menu: &["Filter", "Render"],
            shortcut: None,
            params: r##"{"baseTreeType":1..34=1,"lightDirection":1..5=3,"leavesAmount":0..100=50,"leavesSize":0..200=100,"branchesHeight":50..300=100,"branchesThickness":50..200=100,"defaultLeaves":bool=true,"leavesColor":color,"customBranchColor":bool,"branchesColor":color,"flatShading":bool,"seed":u32=0,"x":0..1=0.5,"y":0..1=0.95,"size":0.05..2=0.8,"newLayer":bool} → {layer,newLayer,bounds,primitives,treeType}"##,
            enabled: has_doc,
            run: run_tree,
            journal: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(depth: u64) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 160, "height": 120, "depth": depth})).unwrap();
        assert_eq!(Some(s.active().unwrap().doc.depth), crate::document_preset_cmds::sample_type(depth));
        s.execute("layer.new.layer", json!({})).unwrap();
        s
    }

    fn pixels(s: &Session) -> Vec<f32> {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().read_region(Rect::new(0, 0, 160, 120))
    }

    fn opaque(s: &Session) -> usize {
        pixels(s).as_chunks::<4>().0.iter().filter(|p| p[3] > 0.05).count()
    }

    #[test]
    fn each_renderer_draws_undoes_and_is_seeded_at_every_depth() {
        for depth in [8, 16, 32] {
            for (id, p) in [
                ("filter.render.flame", json!({})),
                ("filter.render.tree", json!({"baseTreeType": 7})),
                ("filter.render.pictureFrame", json!({"frame": "roses"})),
            ] {
                let mut s = session(depth);
                let r = s.execute(id, p.clone()).unwrap_or_else(|e| panic!("{id}@{depth}: {e}"));
                assert!(r["primitives"].as_u64().unwrap() > 0, "{id}");
                assert!(r["bounds"].is_array(), "{id}@{depth}: {r}");
                let n = opaque(&s);
                assert!(n > 50, "{id}@{depth} drew {n} px");
                let first = pixels(&s);
                s.execute("edit.undo", json!({})).unwrap();
                assert_eq!(opaque(&s), 0, "{id} undo");
                s.execute(id, p.clone()).unwrap();
                assert_eq!(pixels(&s), first, "{id} deterministic");
                s.execute("edit.undo", json!({})).unwrap();
                let mut q = p.clone();
                q["seed"] = json!(5);
                s.execute(id, q).unwrap();
                assert_ne!(pixels(&s), first, "{id} seed changes the result");
            }
        }
    }

    #[test]
    fn flame_follows_the_work_path() {
        let mut s = session(8);
        let r = s.execute("filter.render.flame", json!({})).unwrap();
        assert_eq!(r["usedPath"], json!(false));
        s.execute("edit.undo", json!({})).unwrap();
        // A vertical line path on the left edge: the flame stays near it.
        s.edit("path", |doc, _| {
            let mut sub = photocraft_doc::Subpath::default();
            for (x, y) in [(20.0, 110.0), (20.0, 20.0)] {
                sub.knots.push(photocraft_doc::Knot::corner(x, y));
            }
            doc.work_path = Some(photocraft_doc::Path { subpaths: vec![sub], ..Default::default() });
            Ok(())
        })
        .unwrap();
        let r = s.execute("filter.render.flame", json!({"width": 16, "length": 60})).unwrap();
        assert_eq!(r["usedPath"], json!(true));
        let b: Vec<i64> = r["bounds"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        assert!(b[0] < 20 && b[2] < 60, "flame bounds {b:?} hug the path at x=20");
    }

    #[test]
    fn selection_limits_and_new_layer() {
        let mut s = session(8);
        s.execute("select.rect", json!({"x": 0, "y": 0, "width": 80, "height": 120})).unwrap();
        let r = s.execute("filter.render.tree", json!({"newLayer": true, "baseTreeType": 1, "leavesAmount": 100})).unwrap();
        assert_eq!(r["newLayer"], json!(true));
        let px = pixels(&s);
        let right = px.as_chunks::<4>().0.iter().enumerate().filter(|(i, p)| i % 160 >= 81 && p[3] > 0.0).count();
        assert_eq!(right, 0, "nothing drawn outside the selection");
        assert!(opaque(&s) > 50);
    }

    #[test]
    fn works_in_cmyk_and_gray() {
        for mode in ["cmyk", "grayscale", "lab"] {
            let mut s = Session::new();
            s.execute("file.new", json!({"width": 96, "height": 96, "mode": mode})).unwrap();
            let r = s.execute("filter.render.pictureFrame", json!({"frame": 13, "newLayer": true})).unwrap_or_else(|e| panic!("{mode}: {e}"));
            assert!(r["bounds"].is_array(), "{mode}");
        }
    }

    /// #2501: the Tree and Picture Frame dialogs send every number as a JSON float, and their
    /// `u32` fields (`baseTreeType`, `lightDirection`, `frame`, `lines`, `seed`) refused `1.0`
    /// ("invalid type: floating point `1.0`, expected u32"). Whole floats now draw exactly what
    /// the integers draw, fractional ones (a dragged 1–5 slider) round, and hostile values fail.
    #[test]
    fn integer_params_accept_whole_floats_and_round_fractions() {
        let tree = json!({"baseTreeType": 1.0, "lightDirection": 3.0, "leavesAmount": 50.0, "leavesSize": 100.0, "branchesHeight": 100.0,
            "branchesThickness": 100.0, "defaultLeaves": true, "customBranchColor": false, "flatShading": false, "seed": 0, "x": 0.5, "y": 0.95, "size": 0.8});
        let frame =
            json!({"frame": "vineWithFlowers", "margin": 4.0, "size": 50.0, "arrangement": 50.0, "lines": 1.0, "thickness": 30.0, "fade": 0.0, "seed": 0});
        for (id, floats, ints) in [
            ("filter.render.tree", tree.clone(), json!({"baseTreeType": 1, "lightDirection": 3})),
            (
                "filter.render.tree",
                json!({"baseTreeType": 7.4, "lightDirection": 2.6, "seed": 3.0}),
                json!({"baseTreeType": 7, "lightDirection": 3, "seed": 3}),
            ),
            ("filter.render.pictureFrame", frame.clone(), json!({"frame": 1, "lines": 1})),
            ("filter.render.pictureFrame", json!({"frame": 13.0, "lines": 3.0, "seed": 2.0}), json!({"frame": 13, "lines": 3, "seed": 2})),
            ("filter.render.flame", json!({"seed": 4.0}), json!({"seed": 4})),
        ] {
            let mut s = session(8);
            let a = s.execute(id, floats.clone()).unwrap_or_else(|e| panic!("{id} {floats}: {e}"));
            let drawn = pixels(&s);
            s.execute("edit.undo", json!({})).unwrap();
            let b = s.execute(id, ints.clone()).unwrap();
            assert_eq!(a, b, "{id}: {floats} reports what {ints} does");
            assert_eq!(pixels(&s), drawn, "{id}: {floats} draws what {ints} draws");
        }
        // Negative, too large or non-numeric integer fields are refused, never a panic.
        let mut s = session(8);
        for (id, p) in [
            ("filter.render.tree", json!({"baseTreeType": -1.0})),
            ("filter.render.tree", json!({"lightDirection": 1e300})),
            ("filter.render.tree", json!({"seed": -0.6})),
            ("filter.render.pictureFrame", json!({"lines": -3.5})),
            ("filter.render.pictureFrame", json!({"seed": 5e9})),
            ("filter.render.pictureFrame", json!({"lines": "two"})),
            ("filter.render.flame", json!({"seed": f64::MAX})),
        ] {
            let e = s.execute(id, p.clone()).expect_err(&format!("{id} {p}"));
            assert!(e.to_string().contains("invalid parameters"), "{id} {p}: {e}");
        }
    }

    /// `cargo test --release -p photocraft-engine render_cmds::tests::bench_24mp -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn bench_24mp() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 6000, "height": 4000})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        for (id, p) in [
            ("filter.render.flame", json!({"flameType": "multipleFlamesAlongPath", "length": 600, "width": 160, "interval": 120})),
            ("filter.render.flame", json!({"length": 1500, "width": 400, "flameLines": 100, "quality": "fine"})),
            ("filter.render.tree", json!({"baseTreeType": 1, "leavesAmount": 100})),
            ("filter.render.tree", json!({"baseTreeType": 6, "leavesAmount": 100})),
            ("filter.render.pictureFrame", json!({"frame": 1})),
        ] {
            let t = std::time::Instant::now();
            let r = s.execute(id, p).unwrap();
            println!("{id}: {:?} ({} prims)", t.elapsed(), r["primitives"]);
        }
    }
}
