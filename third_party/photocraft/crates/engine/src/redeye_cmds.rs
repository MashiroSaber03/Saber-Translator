//! Red Eye tool: click a pupil to neutralize the flash reflection.
//!
//! Toolbar-only (`paint.redEye`, empty menu path). The kernel in `photocraft-algo::redeye`
//! runs on a window around each click plus the mask-filter halo.

use photocraft_algo::redeye::{self, MAX_POINTS, MAX_WINDOW};
use photocraft_doc::LayerId;
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

const CMD: &str = "paint.redEye";
const MISS: &str = "no red-eye pixels near the click";

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

fn num(p: &Value, key: &str, default: f32) -> Result<f32> {
    match p.get(key) {
        None => Ok(default),
        Some(v) => v.as_f64().filter(|x| x.is_finite()).map(|x| x as f32).ok_or_else(|| bad(format!("`{key}` must be a finite number (got {v})"))),
    }
}

fn coord(v: &Value) -> Option<f64> {
    v.as_f64().filter(|x| x.is_finite())
}

fn pixel(x: f64, y: f64) -> Result<(i32, i32)> {
    if !x.is_finite() || !y.is_finite() {
        return Err(bad("click coordinates must be finite"));
    }
    let clamp = |v: f64| v.clamp(i32::MIN as f64, i32::MAX as f64).floor() as i32;
    Ok((clamp(x), clamp(y)))
}

/// One click, or a batch of clicks, as document pixels.
fn clicks(p: &Value) -> Result<Vec<(i32, i32)>> {
    if let Some(arr) = p.get("points").and_then(Value::as_array) {
        if arr.is_empty() {
            return Err(bad("`points` is empty"));
        }
        if arr.len() > MAX_POINTS {
            return Err(bad(format!("at most {MAX_POINTS} points")));
        }
        return arr
            .iter()
            .map(|q| match q.as_array().map(Vec::as_slice) {
                Some([x, y]) => match (coord(x), coord(y)) {
                    (Some(x), Some(y)) => pixel(x, y),
                    _ => Err(bad(format!("bad point {q} (want [x, y])"))),
                },
                _ => Err(bad(format!("bad point {q} (want [x, y])"))),
            })
            .collect();
    }
    match (p.get("x").and_then(coord), p.get("y").and_then(coord)) {
        (Some(x), Some(y)) => Ok(vec![pixel(x, y)?]),
        (None, None) if p.get("x").is_none() && p.get("y").is_none() => Err(bad("missing `x`,`y` or `points`")),
        _ => Err(bad("`x` and `y` must be finite numbers")),
    }
}

fn read_rect(cx: i32, cy: i32, radius: f32) -> Result<Rect> {
    let r = if radius.is_finite() { radius.ceil().clamp(0.0, 256.0) as i32 } else { 64 };
    let halo = redeye::halo_radius();
    let ext = r.saturating_add(halo);
    let area = Rect::new(cx.saturating_sub(ext), cy.saturating_sub(ext), cx.saturating_add(ext).saturating_add(1), cy.saturating_add(ext).saturating_add(1));
    let w = area.width() as u64;
    let h = area.height() as u64;
    let cap = MAX_WINDOW as u64;
    if area.is_empty() || w.saturating_mul(h) > cap.saturating_mul(cap) {
        return Err(bad(format!("pupil window is larger than {MAX_WINDOW}² px")));
    }
    Ok(area)
}

fn rgba_region(surf: &Surface, area: Rect) -> Vec<[f32; 4]> {
    let n = (area.width() as usize).saturating_mul(area.height() as usize);
    let mut v = vec![[0.0f32; 4]; n];
    if !area.is_empty() && v.len() == n {
        surf.read_rgba_into(area, &mut v);
    }
    v
}

/// Write back only the pixels whose RGBA changed. Untouched pixels keep their original channel
/// data, so CMYK (K) and Lab values outside the corrected pupil are never re-encoded.
fn write_rgba(surf: &mut Surface, area: Rect, before: &[[f32; 4]], px: &[[f32; 4]]) {
    if area.is_empty() {
        return;
    }
    let fmt = surf.format();
    let n = fmt.channels();
    let w = area.width() as usize;
    let h = area.height() as usize;
    let need = w.saturating_mul(h);
    let expect = need.saturating_mul(n);
    if px.len() < need || before.len() < need || n == 0 || expect / n != need {
        return;
    }
    let mut data = surf.read_region(area);
    if data.len() != expect {
        return;
    }
    for i in 0..need {
        let (Some(q), Some(o)) = (px.get(i), before.get(i)) else { continue };
        if !rgb_changed(o, q) {
            continue;
        }
        let Some(out) = data.get_mut(i.saturating_mul(n)..i.saturating_mul(n).saturating_add(n)) else { continue };
        let alpha = out.last().copied();
        from_rgba_into(&fmt, *q, out);
        if fmt.alpha
            && let (Some(dst), Some(src)) = (out.last_mut(), alpha)
        {
            *dst = src;
        }
    }
    surf.write_region(area, &data);
}

fn rgb_changed(a: &[f32; 4], b: &[f32; 4]) -> bool {
    (a[0] - b[0]).abs() > 1e-5 || (a[1] - b[1]).abs() > 1e-5 || (a[2] - b[2]).abs() > 1e-5
}

fn mix_selection(orig: &[[f32; 4]], px: &mut [[f32; 4]], area: Rect, sel: &Surface) {
    let w = area.width() as usize;
    if w == 0 {
        return;
    }
    let n = px.len().min(orig.len());
    for i in 0..n {
        let y = i / w;
        let x = i % w;
        let k = sel.sample_channel(area.x0.saturating_add(x as i32), area.y0.saturating_add(y as i32), 0).clamp(0.0, 1.0);
        if k >= 1.0 {
            continue;
        }
        let Some(o) = orig.get(i) else { continue };
        let Some(p) = px.get_mut(i) else { continue };
        if k <= 0.0 {
            *p = *o;
            continue;
        }
        p[0] = o[0] + (p[0] - o[0]) * k;
        p[1] = o[1] + (p[1] - o[1]) * k;
        p[2] = o[2] + (p[2] - o[2]) * k;
        p[3] = o[3];
    }
}

fn restore_locked(orig: &[[f32; 4]], px: &mut [[f32; 4]]) {
    let n = px.len().min(orig.len());
    for i in 0..n {
        let Some(o) = orig.get(i) else { continue };
        if o[3] > 1.0e-5 {
            continue;
        }
        if let Some(p) = px.get_mut(i) {
            *p = *o;
        }
    }
}

fn count_changed(orig: &[[f32; 4]], px: &[[f32; 4]]) -> u32 {
    let n = orig.iter().zip(px.iter()).filter(|(a, b)| rgb_changed(a, b)).count();
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn apply_click(surf: &mut Surface, sel: Option<&Surface>, lock: bool, click: (i32, i32), pupil: f32, darken: f32, clip: Rect) -> Result<(Rect, u32)> {
    let area = read_rect(click.0, click.1, redeye::search_radius(pupil))?.intersect(&clip);
    if area.is_empty() {
        return Ok((Rect::EMPTY, 0));
    }
    let w = area.width() as usize;
    let h = area.height() as usize;
    let mut px = rgba_region(surf, area);
    let orig = px.clone();
    let n = redeye::apply(&mut px, w, h, click, (area.x0, area.y0), pupil, darken);
    if n == 0 {
        return Ok((area, 0));
    }
    if let Some(sel) = sel {
        mix_selection(&orig, &mut px, area, sel);
    }
    if lock {
        restore_locked(&orig, &mut px);
    }
    let changed = count_changed(&orig, &px);
    if changed == 0 {
        return Ok((area, 0));
    }
    write_rgba(surf, area, &orig, &px);
    Ok((area, changed))
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    crate::commands::has_paintable(s)?;
    if crate::channel_cmds::edits_channel(s) {
        return Ok(());
    }
    let d = s.active().ok_or("no document open")?;
    let id = d.active_layer.ok_or("no active layer")?;
    let locks = d.doc.effective_locks(id);
    if locks.pixels || locks.all {
        let name = d.doc.layer(id).map(|l| l.name.as_str()).unwrap_or("layer");
        return Err(format!("the layer \"{name}\" is locked"));
    }
    Ok(())
}

fn red_eye(s: &mut Session, p: &Value) -> Result<Value> {
    let pts = clicks(p)?;
    let pupil_raw = num(p, "pupilSize", 50.0)?;
    let darken = num(p, "darken", 50.0)?;
    if redeye::unclamped_window_side(pupil_raw).is_none() {
        return Err(bad(format!("pupil window is larger than {MAX_WINDOW}² px")));
    }
    let pupil = pupil_raw.clamp(1.0, 100.0);
    let darken = darken.clamp(0.0, 100.0);
    let id = match p.get("layer").and_then(Value::as_u64) {
        Some(id) => Some(LayerId(id)),
        None => s.active().and_then(|d| d.active_layer),
    };
    let (damage, total) = s.edit("Red Eye", |doc, active| {
        let canvas = doc.bounds();
        let sel = doc.selection.clone();
        let (surf, lock) = crate::channel_cmds::target_surface(doc, id.or(*active), p)?;
        let mut total = 0u32;
        let mut damage = Rect::EMPTY;
        for click in &pts {
            let (area, n) = apply_click(surf, sel.as_ref(), lock, *click, pupil, darken, canvas)?;
            total = total.saturating_add(n);
            damage = if damage.is_empty() { area } else { damage.union(&area) };
        }
        if total == 0 {
            return Err(EngineError::Other(MISS.into()));
        }
        Ok((damage, total))
    })?;
    if let Some(st) = s.active_mut() {
        st.last_damage = Some(damage);
    }
    Ok(json!({ "damage": [damage.x0, damage.y0, damage.width(), damage.height()], "pixels": total }))
}

/// Red Eye command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Red Eye",
        menu: &[],
        shortcut: None,
        params: r#"{"x":px, "y":px} | {"points":[[x,y],…] (at most 64)},"pupilSize":1..100=50,"darken":0..100=50,"layer":id?=active,"target":"pixels"|"mask"|"quickMask"|{"channel":i} → {"damage":[x,y,w,h], "pixels":n}"#,
        enabled,
        run: red_eye,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::ColorMode;
    use serde_json::json;

    const DEPTHS: [u64; 3] = [8, 16, 32];

    fn session(w: u32, h: u32, depth: u64, mode: &str) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "depth": depth, "mode": mode})).unwrap();
        s
    }

    fn rgba(s: &Session, x: i32, y: i32) -> [f32; 4] {
        let d = s.active().unwrap();
        d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap().rgba(x, y)
    }

    fn paint_layer(s: &mut Session, f: impl Fn(i32, i32) -> [f32; 4]) {
        s.edit("setup", |doc, active| {
            let b = doc.bounds();
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            let fmt = surf.format();
            let mut data = Vec::new();
            for y in b.y0..b.y1 {
                for x in b.x0..b.x1 {
                    data.extend(photocraft_raster::from_rgba(&fmt, f(x, y)));
                }
            }
            surf.write_region(b, &data);
            Ok(())
        })
        .unwrap();
    }

    fn red_disc(cx: i32, cy: i32, r: i32) -> impl Fn(i32, i32) -> [f32; 4] {
        move |x, y| {
            let dx = (x - cx) as f32;
            let dy = (y - cy) as f32;
            if dx.hypot(dy) <= 1.2 {
                [1.0, 1.0, 1.0, 1.0]
            } else if dx.hypot(dy) <= r as f32 {
                [0.95, 0.10, 0.10, 1.0]
            } else {
                [0.38, 0.28, 0.16, 1.0]
            }
        }
    }

    #[test]
    fn red_eye_corrects_a_disc_at_every_depth_and_undoes() {
        for depth in DEPTHS {
            let mut s = session(64, 64, depth, "rgb");
            paint_layer(&mut s, red_disc(32, 32, 8));
            let before = rgba(&s, 32, 26);
            assert!(before[0] > 0.8, "setup {before:?}");
            let r = s.execute("paint.redEye", json!({"x": 32, "y": 32})).unwrap();
            assert!(r["pixels"].as_u64().unwrap() > 0, "{r}");
            let after = rgba(&s, 32, 26);
            assert!(after[0] < before[0] - 0.2, "depth {depth}: {after:?} vs {before:?}");
            let catch = rgba(&s, 32, 32);
            assert!(catch[0] >= 0.9 && catch[1] >= 0.9, "catchlight {catch:?}");
            assert!(s.undo());
            let restored = rgba(&s, 32, 26);
            for c in 0..3 {
                assert!((restored[c] - before[c]).abs() < 0.02, "undo depth {depth}");
            }
            assert!(s.redo());
            let redone = rgba(&s, 32, 26);
            for c in 0..3 {
                assert!((redone[c] - after[c]).abs() < 0.02, "redo depth {depth}");
            }
        }
    }

    #[test]
    fn miss_and_bad_params_are_errors() {
        let mut s = session(32, 32, 8, "rgb");
        paint_layer(&mut s, |_, _| [0.1, 0.2, 0.8, 1.0]);
        let err = s.execute("paint.redEye", json!({"x": 16, "y": 16})).unwrap_err().to_string();
        assert!(err.contains(MISS), "{err}");
        assert!(s.execute("paint.redEye", json!({})).is_err());
        assert!(s.execute("paint.redEye", json!({"x": null, "y": 1})).is_err());
        let pts: Vec<[i32; 2]> = (0..65).map(|i| [i, 0]).collect();
        assert!(s.execute("paint.redEye", json!({"points": pts})).is_err());
        assert!(s.execute("paint.redEye", json!({"x": "nope", "y": 1})).is_err());
        assert!(s.execute("paint.redEye", json!({"points": []})).is_err());
        let huge = s.execute("paint.redEye", json!({"x": 8, "y": 8, "pupilSize": 10_000.0}));
        assert!(huge.unwrap_err().to_string().contains("512"), "window cap");
        s.execute("paint.redEye", json!({"x": 8, "y": 8, "pupilSize": 0})).ok();
        s.execute("paint.redEye", json!({"x": 8, "y": 8, "pupilSize": 101})).ok();
        let mut s = session(64, 64, 8, "rgb");
        paint_layer(&mut s, red_disc(32, 32, 8));
        assert!(s.execute("paint.redEye", json!({"x": 32, "y": 32, "pupilSize": 0})).is_ok(), "pupilSize 0 clamps");
        assert!(s.undo());
        assert!(s.execute("paint.redEye", json!({"x": 32, "y": 32, "pupilSize": 101})).is_ok(), "pupilSize 101 clamps");
    }

    #[test]
    fn disabled_without_a_paintable_target() {
        let s = Session::new();
        assert!(!s.is_enabled("paint.redEye"));
        let mut s = session(32, 32, 8, "rgb");
        assert!(s.is_enabled("paint.redEye"));
        s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
        assert!(!s.is_enabled("paint.redEye"), "locked pixels disable the tool");
        let err = s.execute("paint.redEye", json!({"x": 8, "y": 8})).unwrap_err().to_string();
        assert!(err.to_ascii_lowercase().contains("locked"), "{err}");
        let mut s = session(64, 64, 8, "rgb");
        s.execute("type.create", json!({"x": 8, "y": 24, "text": "Hi", "size": 12})).unwrap();
        assert!(!s.is_enabled("paint.redEye"));
        assert!(s.execute("paint.redEye", json!({"x": 8, "y": 8})).is_err());
    }

    #[test]
    fn several_points_are_one_history_step() {
        let mut s = session(80, 40, 8, "rgb");
        paint_layer(&mut s, |x, y| {
            if ((x - 20).abs() <= 6 && (y - 20).abs() <= 6) || ((x - 60).abs() <= 6 && (y - 20).abs() <= 6) {
                [0.95, 0.1, 0.1, 1.0]
            } else {
                [0.2, 0.2, 0.2, 1.0]
            }
        });
        let past = s.active().unwrap().history.past_len();
        s.execute("paint.redEye", json!({"points": [[20, 20], [60, 20]]})).unwrap();
        assert_eq!(s.active().unwrap().history.past_len(), past + 1);
        assert!(rgba(&s, 20, 20)[0] < 0.7);
        assert!(rgba(&s, 60, 20)[0] < 0.7);
    }

    #[test]
    fn cmyk_document_keeps_untouched_pixels_bit_exact() {
        let mut s = session(64, 64, 8, "cmyk");
        assert_eq!(s.active().unwrap().doc.pixel_format().mode, ColorMode::Cmyk);
        // Rich black (all four inks) around the eye: a naive RGBA round trip regenerates K.
        let disc = red_disc(32, 32, 8);
        s.edit("setup", |doc, active| {
            let b = doc.bounds();
            let surf = doc.layer_mut(active.unwrap()).unwrap().surface_mut().unwrap();
            let fmt = surf.format();
            let n = fmt.channels();
            let mut data = Vec::new();
            for y in b.y0..b.y1 {
                for x in b.x0..b.x1 {
                    if (x - 32).abs() <= 10 && (y - 32).abs() <= 10 {
                        data.extend(photocraft_raster::from_rgba(&fmt, disc(x, y)));
                    } else {
                        let mut px = vec![0.6f32, 0.5, 0.5, 0.7];
                        px.resize(n, 1.0);
                        data.extend(px);
                    }
                }
            }
            surf.write_region(b, &data);
            Ok(())
        })
        .unwrap();
        let raw = |s: &Session, x: i32, y: i32| {
            let d = s.active().unwrap();
            let surf = d.doc.layer(d.active_layer.unwrap()).unwrap().surface().unwrap();
            surf.read_region(Rect::new(x, y, x + 1, y + 1))
        };
        let probes = [(32, 18), (2, 2), (50, 40)];
        let before: Vec<Vec<f32>> = probes.iter().map(|&(x, y)| raw(&s, x, y)).collect();
        let r = s.execute("paint.redEye", json!({"x": 32, "y": 32})).unwrap();
        assert!(r["pixels"].as_u64().unwrap() > 0, "{r}");
        for (&(x, y), b) in probes.iter().zip(&before) {
            assert_eq!(&raw(&s, x, y), b, "untouched CMYK pixel ({x},{y}) inside the window changed");
        }
        assert!(rgba(&s, 32, 26)[0] < 0.7, "pupil corrected");
    }

    #[test]
    fn gray_document_misses() {
        let mut s = session(24, 24, 16, "gray");
        paint_layer(&mut s, |_, _| [0.5, 0.5, 0.5, 1.0]);
        assert_eq!(s.active().unwrap().doc.pixel_format().mode, ColorMode::Grayscale);
        assert!(s.execute("paint.redEye", json!({"x": 12, "y": 12})).is_err());
    }
}
