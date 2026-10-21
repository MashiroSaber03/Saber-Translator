//! Window › Shapes and the Custom Shape tool: vector shape presets in groups, placed as shape
//! layers through `shape.create`.
//!
//! Built-in shapes were drawn from scratch for PhotoCraft in a 100 × 100 box with a small path
//! language ([`parse`]): `M x y`, `L x y…`, `C x1 y1 x2 y2 x y…`, `Q x1 y1 x y…`, `Z`, plus `O cx cy r`
//! (a circle) and `!` before `M`/`O` to subtract that subpath (holes). Shapes defined with Edit ›
//! Define Custom Shape appear in a trailing "Custom Shapes" group.

use photocraft_doc::vector::{Knot, Path, PathOp, Subpath};
use photocraft_geom::{Affine, Point, Rect};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{Group, Named, always, bad, edit_groups, find, group_index, has_doc, req_str, str_param, unique_name};
use crate::commands::CommandSpec;
use crate::vector_cmds::path_json;
use crate::{EngineError, Result, Session};

/// Group listing Edit › Define Custom Shape results (`EditState::custom_shapes`).
pub const CUSTOM_GROUP: &str = "Custom Shapes";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ShapePreset {
    pub name: String,
    pub path: Path,
}

impl Named for ShapePreset {
    fn name(&self) -> &str {
        &self.name
    }
    fn set_name(&mut self, n: String) {
        self.name = n;
    }
}

fn circle(cx: f64, cy: f64, r: f64) -> Subpath {
    // Four cubic quarter arcs (κ = 0.5523).
    let k = r * 0.552_284_75;
    let knot = |x: f64, y: f64, ix: f64, iy: f64, ox: f64, oy: f64| Knot::smooth(Point::new(x, y), Point::new(ix, iy), Point::new(ox, oy));
    Subpath {
        closed: true,
        op: PathOp::Combine,
        knots: vec![
            knot(cx, cy - r, cx - k, cy - r, cx + k, cy - r),
            knot(cx + r, cy, cx + r, cy - k, cx + r, cy + k),
            knot(cx, cy + r, cx + k, cy + r, cx - k, cy + r),
            knot(cx - r, cy, cx - r, cy + k, cx - r, cy - k),
        ],
    }
}

/// Parse the path language described in the module docs.
pub fn parse(src: &str) -> std::result::Result<Path, String> {
    let toks: Vec<String> = src.replace(',', " ").replace('!', " ! ").split_whitespace().map(str::to_string).collect();
    let mut i = 0;
    let num = |i: &mut usize| -> std::result::Result<f64, String> {
        let v = toks.get(*i).ok_or("unexpected end")?.parse::<f64>().map_err(|_| format!("expected a number at `{}`", toks[*i]))?;
        *i += 1;
        Ok(v)
    };
    let is_num = |i: usize| toks.get(i).is_some_and(|t| t.parse::<f64>().is_ok());
    let mut subs: Vec<Subpath> = Vec::new();
    let mut cur: Option<Subpath> = None;
    let mut op = PathOp::Combine;
    let finish = |cur: &mut Option<Subpath>, subs: &mut Vec<Subpath>, closed: bool| {
        if let Some(mut s) = cur.take() {
            s.closed = closed;
            if closed && s.knots.len() > 1 {
                let (f, l) = (s.knots[0].anchor, s.knots[s.knots.len() - 1].anchor);
                if (f.x - l.x).abs() < 1e-9
                    && (f.y - l.y).abs() < 1e-9
                    && let Some(last) = s.knots.pop()
                {
                    s.knots[0].in_ctrl = last.in_ctrl;
                }
            }
            if !s.knots.is_empty() {
                subs.push(s);
            }
        }
    };
    while i < toks.len() {
        let t = toks[i].clone();
        i += 1;
        match t.as_str() {
            "!" => op = PathOp::Subtract,
            "M" => {
                finish(&mut cur, &mut subs, true);
                let (x, y) = (num(&mut i)?, num(&mut i)?);
                cur = Some(Subpath { closed: true, knots: vec![Knot::corner(x, y)], op });
                op = PathOp::Combine;
            }
            "L" => {
                let s = cur.as_mut().ok_or("L before M")?;
                while is_num(i) {
                    let (x, y) = (num(&mut i)?, num(&mut i)?);
                    s.knots.push(Knot::corner(x, y));
                }
            }
            "C" | "Q" => {
                let s = cur.as_mut().ok_or("curve before M")?;
                while is_num(i) {
                    let last = s.knots.last().ok_or("curve before M")?.anchor;
                    let (c1, c2, p) = if t == "C" {
                        let c1 = Point::new(num(&mut i)?, num(&mut i)?);
                        let c2 = Point::new(num(&mut i)?, num(&mut i)?);
                        (c1, c2, Point::new(num(&mut i)?, num(&mut i)?))
                    } else {
                        let q = Point::new(num(&mut i)?, num(&mut i)?);
                        let p = Point::new(num(&mut i)?, num(&mut i)?);
                        (
                            Point::new(last.x + (q.x - last.x) * 2.0 / 3.0, last.y + (q.y - last.y) * 2.0 / 3.0),
                            Point::new(p.x + (q.x - p.x) * 2.0 / 3.0, p.y + (q.y - p.y) * 2.0 / 3.0),
                            p,
                        )
                    };
                    if let Some(k) = s.knots.last_mut() {
                        k.out_ctrl = c1;
                    }
                    s.knots.push(Knot { anchor: p, in_ctrl: c2, out_ctrl: p, smooth: false });
                }
            }
            "Z" => finish(&mut cur, &mut subs, true),
            "O" => {
                finish(&mut cur, &mut subs, true);
                let mut c = circle(num(&mut i)?, num(&mut i)?, num(&mut i)?);
                c.op = op;
                op = PathOp::Combine;
                subs.push(c);
            }
            o => return Err(format!("unknown command `{o}`")),
        }
    }
    finish(&mut cur, &mut subs, true);
    if subs.is_empty() {
        return Err("empty path".into());
    }
    Ok(Path::new(subs))
}

/// Star / burst polygon with `n` points in the 100 box (`ratio` = inner / outer radius).
fn star(n: usize, ratio: f64, rot: f64) -> String {
    let mut s = String::new();
    for k in 0..2 * n {
        let r = if k % 2 == 0 { 50.0 } else { 50.0 * ratio };
        let a = rot + std::f64::consts::PI * k as f64 / n as f64 - std::f64::consts::FRAC_PI_2;
        s += &format!("{} {:.3} {:.3} ", if k == 0 { "M" } else { "L" }, 50.0 + r * a.cos(), 50.0 + r * a.sin());
    }
    s + "Z"
}

fn sun() -> String {
    let mut s = "O 50 50 22 ".to_string();
    for k in 0..12 {
        let a = std::f64::consts::TAU * k as f64 / 12.0;
        let p = |r: f64, da: f64| (50.0 + r * (a + da).cos(), 50.0 + r * (a + da).sin());
        let (b1, b2, tip) = (p(28.0, -0.12), p(28.0, 0.12), p(50.0, 0.0));
        s += &format!("M {:.3} {:.3} L {:.3} {:.3} L {:.3} {:.3} Z ", b1.0, b1.1, tip.0, tip.1, b2.0, b2.1);
    }
    s
}

fn flower() -> String {
    let mut s = String::new();
    for k in 0..6 {
        let a = std::f64::consts::TAU * k as f64 / 6.0 - std::f64::consts::FRAC_PI_2;
        s += &format!("O {:.3} {:.3} 18 ", 50.0 + 30.0 * a.cos(), 50.0 + 30.0 * a.sin());
    }
    s + "O 50 50 20 ! O 50 50 9"
}

fn shout() -> String {
    // A jagged burst stretched into a bubble, with a tail.
    let mut s = String::new();
    let n = 14;
    for k in 0..2 * n {
        let r = if k % 2 == 0 { 1.0 } else { 0.78 };
        let a = std::f64::consts::PI * k as f64 / n as f64;
        s += &format!("{} {:.3} {:.3} ", if k == 0 { "M" } else { "L" }, 50.0 + 50.0 * r * a.cos(), 40.0 + 40.0 * r * a.sin());
    }
    s + "Z M 30 70 L 14 100 L 46 74 Z"
}

/// Built-in groups (our own drawings).
pub fn builtin() -> Vec<Group<ShapePreset>> {
    let mk = |list: Vec<(&str, String)>| -> Vec<ShapePreset> {
        list.into_iter().filter_map(|(n, src)| parse(&src).ok().map(|path| ShapePreset { name: n.into(), path })).collect()
    };
    let s = |x: &str| x.to_string();
    vec![
        Group::new(
            "Symbols",
            mk(vec![
                ("Heart", s("M 50 92 C 20 70 0 52 0 30 C 0 12 14 0 28 0 C 40 0 47 8 50 16 C 53 8 60 0 72 0 C 86 0 100 12 100 30 C 100 52 80 70 50 92 Z")),
                ("Star", star(5, 0.4, 0.0)),
                ("Six-Point Star", star(6, 0.55, 0.0)),
                ("Burst", star(16, 0.72, 0.0)),
                ("Check Mark", s("M 5 55 L 20 40 L 38 58 L 82 12 L 97 27 L 38 88 Z")),
                ("Cross", s("M 20 5 L 50 35 L 80 5 L 95 20 L 65 50 L 95 80 L 80 95 L 50 65 L 20 95 L 5 80 L 35 50 L 5 20 Z")),
                ("Plus", s("M 38 0 L 62 0 L 62 38 L 100 38 L 100 62 L 62 62 L 62 100 L 38 100 L 38 62 L 0 62 L 0 38 L 38 38 Z")),
                ("Ring", s("O 50 50 50 ! O 50 50 32")),
                ("Lightning", s("M 58 0 L 18 56 L 44 56 L 34 100 L 82 38 L 55 38 L 70 0 Z")),
                ("Diamond", s("M 50 0 L 90 50 L 50 100 L 10 50 Z")),
            ]),
        ),
        Group::new(
            "Arrows",
            mk(vec![
                ("Arrow Right", s("M 0 35 L 55 35 L 55 15 L 100 50 L 55 85 L 55 65 L 0 65 Z")),
                ("Arrow Left", s("M 100 35 L 45 35 L 45 15 L 0 50 L 45 85 L 45 65 L 100 65 Z")),
                ("Arrow Up", s("M 35 100 L 35 45 L 15 45 L 50 0 L 85 45 L 65 45 L 65 100 Z")),
                ("Arrow Down", s("M 35 0 L 35 55 L 15 55 L 50 100 L 85 55 L 65 55 L 65 0 Z")),
                ("Double Arrow", s("M 0 50 L 25 20 L 25 38 L 75 38 L 75 20 L 100 50 L 75 80 L 75 62 L 25 62 L 25 80 Z")),
                ("Chevron", s("M 10 0 L 45 0 L 95 50 L 45 100 L 10 100 L 60 50 Z")),
                ("Curved Arrow", s("M 8 92 C 8 45 35 22 68 22 L 68 4 L 98 34 L 68 64 L 68 44 C 48 44 30 58 30 92 Z")),
            ]),
        ),
        Group::new(
            "Speech Bubbles",
            mk(vec![
                (
                    "Speech Bubble",
                    s(
                        "M 15 5 L 85 5 C 93 5 100 12 100 20 L 100 60 C 100 68 93 75 85 75 L 45 75 L 20 98 L 25 75 L 15 75 C 7 75 0 68 0 60 L 0 20 C 0 12 7 5 15 5 Z",
                    ),
                ),
                ("Round Bubble", s("M 50 5 C 78 5 100 22 100 42 C 100 62 78 79 50 79 C 44 79 38 78 33 77 L 12 95 L 18 72 C 7 65 0 54 0 42 C 0 22 22 5 50 5 Z")),
                ("Thought Bubble", s("M 50 5 C 78 5 100 20 100 38 C 100 56 78 71 50 71 C 22 71 0 56 0 38 C 0 20 22 5 50 5 Z O 22 82 8 O 9 95 5")),
                ("Shout Bubble", shout()),
            ]),
        ),
        Group::new(
            "Nature",
            mk(vec![
                (
                    "Leaf",
                    s(
                        "M 50 0 C 80 20 95 50 80 75 C 72 88 60 92 53 92 L 53 100 L 47 100 L 47 92 C 40 92 28 88 20 75 C 5 50 20 20 50 0 Z ! M 48.5 22 L 51.5 22 L 51.5 88 L 48.5 88 Z",
                    ),
                ),
                ("Sun", sun()),
                ("Crescent Moon", s("O 45 50 45 ! O 66 40 38")),
                (
                    "Cloud",
                    s("M 25 80 C 10 80 0 70 0 58 C 0 46 10 37 22 37 C 24 20 38 10 54 10 C 70 10 82 20 85 34 C 95 35 100 45 100 57 C 100 70 90 80 78 80 Z"),
                ),
                ("Raindrop", s("M 50 0 C 60 25 85 45 85 68 C 85 87 69 100 50 100 C 31 100 15 87 15 68 C 15 45 40 25 50 0 Z")),
                ("Flower", flower()),
                ("Tree", s("M 50 0 L 85 45 L 68 45 L 92 75 L 56 75 L 56 100 L 44 100 L 44 75 L 8 75 L 32 45 L 15 45 Z")),
            ]),
        ),
        Group::new(
            "Animals",
            mk(vec![
                ("Fish", s("M 0 50 C 15 25 45 15 70 32 L 96 12 L 88 50 L 96 88 L 70 68 C 45 85 15 75 0 50 Z ! O 20 45 4")),
                ("Cat", s("M 15 10 L 35 30 C 45 27 55 27 65 30 L 85 10 L 88 50 C 92 75 72 95 50 95 C 28 95 8 75 12 50 Z ! O 36 58 6 ! O 64 58 6")),
                ("Bird", s("M 0 40 C 15 30 35 32 50 50 C 65 32 85 30 100 40 C 85 40 65 48 50 70 C 35 48 15 40 0 40 Z")),
                (
                    "Butterfly",
                    s(
                        "M 50 30 C 40 5 5 0 5 25 C 5 45 30 50 45 50 C 25 55 10 75 25 90 C 38 100 48 80 50 65 C 52 80 62 100 75 90 C 90 75 75 55 55 50 C 70 50 95 45 95 25 C 95 0 60 5 50 30 Z",
                    ),
                ),
                (
                    "Paw Print",
                    s(
                        "M 50 50 C 68 50 82 66 82 80 C 82 92 70 96 60 94 C 55 93 52 91 50 91 C 48 91 45 93 40 94 C 30 96 18 92 18 80 C 18 66 32 50 50 50 Z O 18 42 10 O 37 22 11 O 63 22 11 O 82 42 10",
                    ),
                ),
                ("Rabbit", s("O 50 72 26 M 34 52 C 22 30 24 2 34 2 C 44 2 46 30 44 50 Z M 66 52 C 78 30 76 2 66 2 C 56 2 54 30 56 50 Z")),
            ]),
        ),
    ]
}

/// Every shape the panel shows: the preset groups plus Define Custom Shape results.
pub fn all_groups(s: &Session) -> Vec<Group<ShapePreset>> {
    let mut g = s.presets.shapes.clone();
    let custom: Vec<ShapePreset> = s.edit_state.custom_shapes.iter().map(|c| ShapePreset { name: c.name.clone(), path: c.path.clone() }).collect();
    if !custom.is_empty() {
        g.push(Group::new(CUSTOM_GROUP, custom));
    }
    g
}

fn lookup(s: &Session, name: &str, group: Option<&str>, cmd: &str) -> Result<ShapePreset> {
    let groups = all_groups(s);
    let (gi, ii) = find(&groups, name, group).ok_or_else(|| bad(cmd, format!("no shape \"{name}\" (see shape.presets.list)")))?;
    Ok(groups[gi].items[ii].clone())
}

/// Coverage (0..1) of `path` fitted into a `size`² swatch with a small margin.
pub fn thumbnail(path: &Path, size: u32) -> Vec<f32> {
    let size = size.clamp(4, 512);
    let m = (size as f64 * 0.08).max(1.0);
    let fitted = fit(path, [m, m, size as f64 - 2.0 * m, size as f64 - 2.0 * m], true);
    photocraft_vector::path_coverage(&fitted, Rect::new(0, 0, size as i32, size as i32))
}

/// Map `path` (its control bounds) onto `rect` = [x, y, w, h]; `keep` preserves the aspect ratio
/// (centred in the rect).
pub fn fit(path: &Path, rect: [f64; 4], keep: bool) -> Path {
    let Some((x0, y0, x1, y1)) = path.control_bounds() else { return path.clone() };
    let (bw, bh) = ((x1 - x0).max(1e-9), (y1 - y0).max(1e-9));
    let (mut sx, mut sy) = (rect[2] / bw, rect[3] / bh);
    let (mut ox, mut oy) = (rect[0], rect[1]);
    if keep {
        let k = sx.min(sy);
        ox += (rect[2] - bw * k) / 2.0;
        oy += (rect[3] - bh * k) / 2.0;
        (sx, sy) = (k, k);
    }
    path.transform(&Affine { m: [sx, 0.0, 0.0, sy, ox - x0 * sx, oy - y0 * sy] })
}

fn list(s: &mut Session, _: &Value) -> Result<Value> {
    let groups: Vec<Value> = all_groups(s)
        .iter()
        .map(|g| {
            let items: Vec<Value> = g.items.iter().map(|i| json!({"name": i.name, "subpaths": i.path.subpaths.len()})).collect();
            json!({"name": g.name, "shapes": items})
        })
        .collect();
    Ok(json!({"groups": groups}))
}

fn place(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "shape.presets.place";
    let name = req_str(p, "preset", CMD)?;
    let shape = lookup(s, name, str_param(p, "group"), CMD)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (w, h) = (d.doc.size.width as f64, d.doc.size.height as f64);
    let keep = p.get("keepAspect").and_then(Value::as_bool).unwrap_or(true);
    let rect = match p.get("rect").and_then(Value::as_array) {
        Some(a) if a.len() == 4 => {
            let v: Vec<f64> = a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect();
            // Negative sizes (dragging up/left) flip into a positive rect.
            let (x, wd) = if v[2] < 0.0 { (v[0] + v[2], -v[2]) } else { (v[0], v[2]) };
            let (y, hd) = if v[3] < 0.0 { (v[1] + v[3], -v[3]) } else { (v[1], v[3]) };
            if wd < 1.0 || hd < 1.0 {
                return Err(bad(CMD, "the shape rect is empty"));
            }
            [x, y, wd, hd]
        }
        _ => {
            let side = w.min(h) * 0.5;
            [(w - side) / 2.0, (h - side) / 2.0, side, side]
        }
    };
    let path = fit(&shape.path, rect, keep);
    let mut q = json!({"kind": "path", "path": path_json(&path), "name": str_param(p, "name").unwrap_or(&shape.name)});
    for k in ["fill", "stroke", "addTo", "op"] {
        if let Some(v) = p.get(k) {
            q[k] = v.clone();
        }
    }
    let mut r = super::call(s, "shape.create", q)?;
    r["preset"] = json!(shape.name);
    Ok(r)
}

fn new_preset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "shape.presets.new";
    let path = match p.get("path") {
        Some(v @ Value::Object(_)) => crate::vector_cmds::parse_path(v).map_err(|e| bad(CMD, e))?,
        Some(Value::String(src)) if src.trim_start().starts_with(['M', 'O', '!']) => parse(src).map_err(|e| bad(CMD, e))?,
        other => {
            let d = s.active().ok_or_else(|| bad(CMD, "no document (pass `path`)"))?;
            crate::edit_menu_cmds::current_path(&d.doc, d.active_layer, other.and_then(Value::as_str))
                .ok_or_else(|| bad(CMD, "select a path or shape layer first"))?
        }
    };
    if path.is_empty() {
        return Err(bad(CMD, "the path is empty"));
    }
    // Normalise into the 100 box so thumbnails and placing don't depend on where it was drawn.
    let path = fit(&path, [0.0, 0.0, 100.0, 100.0], true);
    let name = unique_name(&s.presets.shapes, str_param(p, "name").unwrap_or("Shape"));
    let gi = group_index(&mut s.presets.shapes, str_param(p, "group"));
    let group = s.presets.shapes[gi].name.clone();
    s.presets.shapes[gi].items.push(ShapePreset { name: name.clone(), path });
    s.presets_changed();
    Ok(json!({"name": name, "group": group}))
}

fn edit(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "shape.presets.edit";
    let action = req_str(p, "action", CMD)?.to_string();
    // Define Custom Shape results live in the edit state; rename/delete them there.
    let custom = |s: &Session, n: &str| {
        str_param(p, "group").is_none_or(|g| g == CUSTOM_GROUP)
            && find(&s.presets.shapes, n, None).is_none()
            && s.edit_state.custom_shapes.iter().any(|c| c.name == n)
    };
    let r = match action.as_str() {
        "rename" if custom(s, req_str(p, "preset", CMD)?) => {
            let (n, to) = (req_str(p, "preset", CMD)?, req_str(p, "name", CMD)?);
            if let Some(c) = s.edit_state.custom_shapes.iter_mut().find(|c| c.name == n) {
                c.name = to.to_string();
            }
            json!({"name": to})
        }
        "delete" if p.get("preset").and_then(Value::as_str).is_some_and(|n| custom(s, n)) => {
            let n = req_str(p, "preset", CMD)?;
            s.edit_state.custom_shapes.retain(|c| c.name != n);
            json!({"deleted": 1})
        }
        a => edit_groups(&mut s.presets.shapes, a, p, CMD)?,
    };
    s.presets_changed();
    Ok(r)
}

fn reset(s: &mut Session, p: &Value) -> Result<Value> {
    let mut b = builtin();
    if p.get("append").and_then(Value::as_bool).unwrap_or(false) {
        for g in &mut b {
            g.name = super::gradients::unique_group(&s.presets.shapes, &g.name);
        }
        s.presets.shapes.extend(b);
    } else {
        s.presets.shapes = b;
    }
    s.presets_changed();
    Ok(json!({"groups": s.presets.shapes.len()}))
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "shape.presets.list",
            label: "Shape Presets",
            menu: &[],
            shortcut: None,
            params: "{} → {groups:[{name,shapes:[{name,subpaths}]}]}",
            enabled: always,
            run: list,
            journal: false,
        },
        CommandSpec {
            id: "shape.presets.place",
            label: "Place Custom Shape",
            menu: &[],
            shortcut: None,
            params: r##"{"preset":name,"group":name?,"rect":[x,y,w,h]? (default: centred, half the canvas),"keepAspect":bool=true,"fill":…?=foreground,"stroke":…?,"name":str?,"addTo":layerId?,"op":"combine|subtract|intersect|exclude"?} → shape.info (Custom Shape tool / Shapes panel; fill and stroke as in shape.create)"##,
            enabled: has_doc,
            run: place,
            journal: true,
        },
        CommandSpec {
            id: "shape.presets.new",
            label: "New Shape Preset",
            menu: &[],
            shortcut: None,
            params: r##"{"name":str="Shape","group":name?,"path":{…path}|"M x y L …" (preset path language)|"work"|saved path name? (default: work path, active shape layer or vector mask)}"##,
            enabled: always,
            run: new_preset,
            journal: true,
        },
        CommandSpec {
            id: "shape.presets.edit",
            label: "Edit Shape Presets",
            menu: &[],
            shortcut: None,
            params: super::GROUP_EDIT_PARAMS,
            enabled: always,
            run: edit,
            journal: true,
        },
        CommandSpec {
            id: "shape.presets.reset",
            label: "Restore Default Shapes",
            menu: &[],
            shortcut: None,
            params: r##"{"append":bool=false}"##,
            enabled: always,
            run: reset,
            journal: true,
        },
    ]
}
