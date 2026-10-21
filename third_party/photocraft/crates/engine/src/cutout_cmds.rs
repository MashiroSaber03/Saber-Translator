//! Remove Background, Photoshop's Quick Action for a pixel layer (Properties › Quick Actions):
//! Select Subject with a light edge refinement, turned into a layer mask, in one history step.
//! Like Photoshop it is non-destructive (the pixels stay; the mask hides the background), turns
//! the Background layer into a normal layer first and has no menu item.

use photocraft_algo::matting::{self, RefineParams};
use photocraft_algo::segment::subject;
use photocraft_doc::{Document, Layer, LayerContent, LayerMask};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::smartselect_cmds::with_doc_sampler;
use crate::{EngineError, Result, Session};

const CMD: &str = "layer.removeBackground";

/// Edge refinement after Select Subject: a narrow smart radius and a little smoothing, so soft
/// edges and hair get partial coverage instead of a hard cut.
const REFINE: RefineParams = RefineParams { radius: 2.0, smart_radius: true, smooth: 10.0, feather: 0.5, contrast: 10.0, shift_edge: 0.0 };

/// Remove Background needs an unlocked pixel layer.
fn check(doc: &Document, l: &Layer) -> std::result::Result<(), String> {
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("the layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    let locks = doc.effective_locks(l.id);
    if locks.pixels || locks.all {
        return Err(format!("Could not complete your request because the layer \"{}\" is locked", l.name));
    }
    Ok(())
}

fn enabled(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    check(&d.doc, crate::active_layer_of(s)?)
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let doc = &s.active().ok_or(EngineError::NoDocument)?.doc;
    check(doc, doc.layer(id).ok_or(EngineError::NoLayer(id))?).map_err(EngineError::Other)?;
    let sample_all = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let refine = p.get("refine").and_then(Value::as_bool).unwrap_or(true);
    crate::jobs::edit_job(
        s,
        "Remove Background",
        move |doc, _, ctx| {
            // Cancel reaches the min cuts inside Select Subject, so Esc doesn't wait for them.
            let found = ctx
                .stage(0.1, 0.6, "Finding subject", |ctl| {
                    with_doc_sampler(doc, Some(id), sample_all, |smp, d| subject::select_subject_with(smp, d.bounds(), ctl))
                })
                .map_err(|_| EngineError::Cancelled)?;
            let mut region = found.ok_or_else(|| EngineError::Other("no subject found".into()))?;
            if ctx.cancelled() {
                return Err(EngineError::Cancelled);
            }
            if refine {
                ctx.progress(0.6, "Refining edge");
                let refined = with_doc_sampler(doc, Some(id), sample_all, |smp, d| {
                    matting::refine_mask(smp, &matting::region_reader(&region), region.bbox, d.bounds(), &REFINE)
                });
                region = refined.unwrap_or(region);
                if ctx.cancelled() {
                    return Err(EngineError::Cancelled);
                }
            }
            crate::extra_cmds::background_to_layer_for_mask(doc, id);
            doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(LayerMask { surface: matting::region_surface(&region), ..LayerMask::reveal_all() });
            doc.selection = None;
            let b = region.bbox;
            Ok([b.x0, b.y0, b.width() as i32, b.height() as i32])
        },
        move |bounds| json!({ "layer": id.0, "bounds": bounds }),
    )
}

/// Remove Background command spec (no menu path: a Properties Quick Action in Photoshop).
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Remove Background",
        menu: &[],
        shortcut: None,
        params: r##"{"layer":id?,"sampleAllLayers":bool=false,"refine":bool=true} → {layer,bounds} (adds a layer mask from Select Subject; the Background becomes a normal layer)"##,
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_algo::segment::Rng;
    use photocraft_doc::LayerId;
    use photocraft_geom::Rect;

    fn session_with(w: u32, h: u32, depth: u32, px: impl Fn(u32, u32) -> [f32; 3]) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h, "depth": depth})).unwrap();
        s.edit("paint", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let mut data = Vec::with_capacity((w * h) as usize * n);
            for y in 0..h {
                for x in 0..w {
                    let c = px(x, y);
                    data.extend_from_slice(&c[..n.min(3)]);
                    if n == 4 {
                        data.push(1.0);
                    }
                }
            }
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        s
    }

    /// Textured disc (centre (100, 80), radius 45) on a noisy background.
    fn disc(depth: u32) -> (Session, impl Fn(i32, i32) -> bool) {
        let mut rng = Rng::new(2);
        let nz: Vec<f32> = (0..200 * 160).map(|_| rng.normal()).collect();
        let inside = |x: i32, y: i32| {
            let (dx, dy) = (x as f32 + 0.5 - 100.0, y as f32 + 0.5 - 80.0);
            dx * dx + dy * dy <= 45.0 * 45.0
        };
        let s = session_with(200, 160, depth, move |x, y| {
            let n = 0.05 * nz[(y * 200 + x) as usize];
            if inside(x as i32, y as i32) {
                let t = ((x + y) as f32 * 0.35).sin() * 0.5 + 0.5;
                [0.85 + 0.05 * t + n, 0.2 + 0.3 * t + n, 0.15 + n]
            } else {
                [0.25 + n, 0.45 + n, 0.7 + n]
            }
            .map(|v| v.clamp(0.0, 1.0))
        });
        (s, inside)
    }

    fn active_id(s: &Session) -> LayerId {
        s.active().unwrap().active_layer.unwrap()
    }

    fn layer(s: &Session, id: LayerId) -> &Layer {
        s.active().unwrap().doc.layer(id).unwrap()
    }

    fn mask_cov(s: &Session, x: i32, y: i32) -> f32 {
        layer(s, active_id(s)).mask.as_ref().map_or(0.0, |m| m.value(x, y))
    }

    fn mask_iou(s: &Session, w: i32, h: i32, truth: impl Fn(i32, i32) -> bool) -> f32 {
        let (mut i, mut u) = (0, 0);
        for y in 0..h {
            for x in 0..w {
                let (a, b) = (mask_cov(s, x, y) >= 0.5, truth(x, y));
                i += (a && b) as i32;
                u += (a || b) as i32;
            }
        }
        i as f32 / u.max(1) as f32
    }

    #[test]
    fn remove_background_masks_the_subject_at_every_depth_and_undoes() {
        for depth in [8, 16, 32] {
            let (mut s, inside) = disc(depth);
            assert!(s.is_enabled(CMD));
            let id = active_id(&s);
            let pixels = layer(&s, id).surface().unwrap().read_region(Rect::new(0, 0, 200, 160));
            let steps = s.active().unwrap().history.past_len();
            let r = s.execute(CMD, json!({})).unwrap();
            assert_eq!(r["layer"], id.0);
            assert_eq!(s.active().unwrap().history.past_len(), steps + 1, "depth {depth}: one history step");
            assert!(layer(&s, id).mask.is_some(), "depth {depth}: expected a layer mask");
            let v = mask_iou(&s, 200, 160, &inside);
            assert!(v >= 0.75, "depth {depth}: IoU {v}");
            assert!(mask_cov(&s, 0, 0) < 0.2, "depth {depth}: background corner still revealed");
            assert!(mask_cov(&s, 100, 80) >= 0.8, "depth {depth}: disc centre hidden");
            // Non-destructive: the pixels are untouched, only masked.
            assert_eq!(layer(&s, id).surface().unwrap().read_region(Rect::new(0, 0, 200, 160)), pixels, "depth {depth}");
            let corner = photocraft_compose::render(&s.active().unwrap().doc, Rect::new(0, 0, 1, 1)).px[0][3];
            assert!(corner < 0.2, "depth {depth}: composite corner alpha {corner}");
            assert!(s.active().unwrap().doc.selection.is_none());
            s.execute("edit.undo", json!({})).unwrap();
            assert!(layer(&s, id).mask.is_none(), "depth {depth}: mask remained after undo");
            s.execute("edit.redo", json!({})).unwrap();
            assert!(layer(&s, id).mask.is_some(), "depth {depth}: mask missing after redo");
        }
    }

    #[test]
    fn remove_background_turns_the_background_into_a_layer() {
        let (mut s, _) = disc(8);
        let id = active_id(&s);
        assert_eq!(layer(&s, id).name, "Background");
        s.execute(CMD, json!({})).unwrap();
        let l = layer(&s, id);
        assert_eq!(l.name, "Layer 0");
        assert!(!l.locks.transparency && !l.locks.position);
        assert!(l.mask.is_some());
    }

    #[test]
    fn remove_background_cancelled_while_finding_the_subject_changes_nothing() {
        let (mut s, _) = disc(8);
        let id = active_id(&s);
        let before = s.active().unwrap().revision;
        // The job's cancel reaches Select Subject, which stops at its first check.
        let ctx = crate::jobs::JobCtx::new();
        ctx.cancel();
        s.set_inline_job_ctx(Some(ctx.clone()));
        assert!(matches!(s.execute(CMD, json!({})), Err(EngineError::Cancelled)));
        assert_eq!(ctx.message(), "Finding subject");
        assert_eq!(s.active().unwrap().revision, before);
        assert!(layer(&s, id).mask.is_none());
        s.set_inline_job_ctx(None);
        s.execute(CMD, json!({})).unwrap();
        assert!(layer(&s, id).mask.is_some());
    }

    #[test]
    fn remove_background_without_a_subject_leaves_the_document() {
        let mut s = session_with(32, 32, 8, |_, _| [0.5, 0.5, 0.5]);
        let before = s.active().unwrap().revision;
        let id = active_id(&s);
        assert!(s.execute(CMD, json!({})).is_err());
        assert_eq!(s.active().unwrap().revision, before);
        assert!(layer(&s, id).mask.is_none());
    }

    #[test]
    fn remove_background_disabled_states_and_bad_params() {
        assert!(!Session::new().is_enabled(CMD));
        assert!(Session::new().execute(CMD, json!({})).is_err());

        let (mut s, _) = disc(8);
        s.execute("type.create", json!({"text": "Hi", "size": 24, "x": 10, "y": 20})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());

        let (mut s, _) = disc(8);
        s.execute("shape.create", json!({"kind": "rect", "rect": [10, 10, 40, 20], "fill": "#ff0000"})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());

        let (mut s, _) = disc(8);
        s.execute("layer.setProps", json!({"locks": {"pixels": true}})).unwrap();
        assert!(!s.is_enabled(CMD));
        assert!(s.execute(CMD, json!({})).is_err());

        let (mut s, _) = disc(8);
        let before = s.active().unwrap().revision;
        assert!(s.execute(CMD, json!({"layer": u64::MAX})).is_err());
        // A `layer` string names a layer; one that isn't there is an error, not the active layer.
        assert!(s.execute(CMD, json!({"layer": "x"})).is_err());
        assert_eq!(s.active().unwrap().revision, before);
        // Wrong types for the options fall back to the defaults (layer-only sampling, refined).
        assert!(s.execute(CMD, json!({"sampleAllLayers": 3, "refine": "no"})).is_ok());
        assert!(s.active().unwrap().revision > before);
    }

    #[test]
    fn remove_background_looks_at_the_layer_alone_by_default() {
        let (mut s, _) = disc(8);
        let bottom = active_id(&s);
        s.execute("layer.new.layer", json!({"name": "Empty"})).unwrap();
        let top = active_id(&s);
        let before = s.active().unwrap().revision;
        // The empty layer has no subject of its own, though the photo shows through it.
        assert!(s.execute(CMD, json!({})).is_err());
        assert_eq!(s.active().unwrap().revision, before);
        assert!(layer(&s, top).mask.is_none() && layer(&s, bottom).mask.is_none());
        // Sampling all layers finds the subject in the composite.
        s.execute(CMD, json!({"sampleAllLayers": true})).unwrap();
        assert!(layer(&s, top).mask.is_some() && layer(&s, bottom).mask.is_none());
    }

    #[test]
    fn remove_background_twice_replaces_the_mask() {
        let (mut s, _) = disc(8);
        s.execute(CMD, json!({})).unwrap();
        s.execute(CMD, json!({"refine": false})).unwrap();
        let masks = s.active().unwrap().doc.walk().into_iter().filter(|(_, _, l)| l.mask.is_some()).count();
        assert_eq!(masks, 1);
    }
}
