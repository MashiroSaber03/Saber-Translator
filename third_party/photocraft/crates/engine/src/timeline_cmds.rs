//! Window › Timeline: create and drive the document timeline (playhead over N frames at a frame
//! rate). Navigation doesn't add history steps. Video layers (Layer › Video Layers, added later)
//! read `current` to show their frame. Headless and scriptable so agents can render frame ranges.

use std::sync::Arc;

use photocraft_doc::Timeline;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Longest timeline, in frames: every frame becomes a surface in a new video layer and a file in
/// Render Video, so `duration` is capped well before it can exhaust memory or overflow (#1009).
const MAX_DURATION: usize = 1_000_000;
/// Highest frame rate; larger values only come from mistakes and become `inf` as `f32` (#1009).
const MAX_FPS: f32 = 1000.0;

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document".into())
}

fn has_timeline(s: &Session) -> std::result::Result<(), String> {
    match s.active() {
        Some(d) if d.doc.timeline.is_some() => Ok(()),
        Some(_) => Err("no timeline (create one first)".into()),
        None => Err("no document".into()),
    }
}

/// Mutate the document timeline without a history step (navigation / metadata).
fn with_timeline(s: &mut Session, f: impl FnOnce(&mut Option<Timeline>)) -> Result<Value> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let mut doc = (*st.doc).clone();
    crate::video_cmds::store(&mut doc);
    f(&mut doc.timeline);
    if let Some(t) = &mut doc.timeline {
        t.clamp();
    }
    crate::video_cmds::sync(&mut doc);
    st.doc = Arc::new(doc);
    st.revision += 1;
    info(s)
}

fn bad(cmd: &str, msg: String) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg }
}

/// The optional `duration` param, rejected unless it's a whole number in `1..=MAX_DURATION`.
/// Before #1009 any `u64` was kept and stepping panicked on it.
fn duration_param(cmd: &str, p: &Value) -> Result<Option<usize>> {
    let Some(v) = p.get("duration").filter(|v| !v.is_null()) else { return Ok(None) };
    v.as_u64()
        .and_then(|d| usize::try_from(d).ok())
        .filter(|d| (1..=MAX_DURATION).contains(d))
        .map(Some)
        .ok_or_else(|| bad(cmd, format!("`duration` must be a whole number of frames from 1 to {MAX_DURATION}, got {v}")))
}

/// The optional `fps` param, rejected unless it's a finite number in `(0, MAX_FPS]` (#1009).
fn fps_param(cmd: &str, p: &Value) -> Result<Option<f32>> {
    let Some(v) = p.get("fps").filter(|v| !v.is_null()) else { return Ok(None) };
    v.as_f64()
        .filter(|f| f.is_finite() && *f > 0.0 && *f <= f64::from(MAX_FPS))
        .map(|f| Some(f as f32))
        .ok_or_else(|| bad(cmd, format!("`fps` must be a number above 0 and at most {MAX_FPS}, got {v}")))
}

fn create(s: &mut Session, p: &Value) -> Result<Value> {
    let duration = duration_param("timeline.create", p)?.unwrap_or(30);
    let fps = fps_param("timeline.create", p)?.unwrap_or(30.0);
    with_timeline(s, |t| *t = Some(Timeline::new(duration, fps)))
}

fn delete(s: &mut Session) -> Result<Value> {
    with_timeline(s, |t| *t = None)
}

fn set_frame(s: &mut Session, p: &Value) -> Result<Value> {
    let f =
        p.get("frame").and_then(Value::as_u64).ok_or_else(|| EngineError::BadParams { cmd: "timeline.setFrame".into(), msg: "need `frame`".into() })? as usize;
    with_timeline(s, |t| {
        if let Some(t) = t {
            t.current = f;
        }
    })
}

fn step(s: &mut Session, delta: i64) -> Result<Value> {
    with_timeline(s, |t| {
        if let Some(t) = t {
            t.step(delta);
        }
    })
}

fn set_props(s: &mut Session, p: &Value) -> Result<Value> {
    let fps = fps_param("timeline.setProps", p)?;
    let duration = duration_param("timeline.setProps", p)?;
    with_timeline(s, |t| {
        if let Some(t) = t {
            if let Some(fps) = fps {
                t.fps = fps;
            }
            if let Some(d) = duration {
                t.duration = d;
            }
            if let Some(ws) = p.get("workStart").and_then(Value::as_u64) {
                t.work_start = ws as usize;
            }
            if let Some(we) = p.get("workEnd").and_then(Value::as_u64) {
                t.work_end = we as usize;
            }
        }
    })
}

fn info(s: &mut Session) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    Ok(match &st.doc.timeline {
        Some(t) => {
            json!({"timeline": {"fps": t.fps, "duration": t.duration, "current": t.current, "workStart": t.work_start, "workEnd": t.work_end, "time": t.time()}})
        }
        None => json!({"timeline": null}),
    })
}

macro_rules! spec {
    ($id:expr, $label:expr, $params:expr, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[], shortcut: None, params: $params, enabled: $en, journal: false, run: $run }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!("timeline.create", "Create Video Timeline", r#"{duration?:30, fps?:30} → {timeline}"#, has_doc, |s, p| create(s, p)),
        spec!("timeline.delete", "Delete Timeline", "{} → {timeline:null}", has_timeline, |s, _| delete(s)),
        spec!("timeline.setFrame", "Go to Frame", r#"{frame} → {timeline}"#, has_timeline, |s, p| set_frame(s, p)),
        spec!("timeline.nextFrame", "Next Frame", "{} → {timeline}", has_timeline, |s, _| step(s, 1)),
        spec!("timeline.previousFrame", "Previous Frame", "{} → {timeline}", has_timeline, |s, _| step(s, -1)),
        spec!("timeline.setProps", "Timeline Settings", r#"{fps?, duration?, workStart?, workEnd?} → {timeline}"#, has_timeline, |s, p| set_props(s, p)),
        spec!("timeline.info", "Timeline Info", "{} → {timeline}", has_doc, |s, _| info(s)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_navigate_delete() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 16, "height": 16})).unwrap();
        assert_eq!(s.execute("timeline.info", json!({})).unwrap()["timeline"], Value::Null);
        let r = s.execute("timeline.create", json!({"duration": 24, "fps": 24.0})).unwrap();
        assert_eq!(r["timeline"]["duration"], 24);
        assert_eq!(r["timeline"]["current"], 0);
        // Navigate.
        s.execute("timeline.setFrame", json!({"frame": 10})).unwrap();
        s.execute("timeline.nextFrame", json!({})).unwrap();
        assert_eq!(s.execute("timeline.info", json!({})).unwrap()["timeline"]["current"], 11);
        s.execute("timeline.previousFrame", json!({})).unwrap();
        assert_eq!(s.execute("timeline.info", json!({})).unwrap()["timeline"]["current"], 10);
        // Clamp beyond the end.
        let r = s.execute("timeline.setFrame", json!({"frame": 999})).unwrap();
        assert_eq!(r["timeline"]["current"], 23);
        // Props.
        s.execute("timeline.setProps", json!({"fps": 30.0})).unwrap();
        assert_eq!(s.execute("timeline.info", json!({})).unwrap()["timeline"]["fps"], 30.0);
        // Delete.
        s.execute("timeline.delete", json!({})).unwrap();
        assert_eq!(s.execute("timeline.info", json!({})).unwrap()["timeline"], Value::Null);
    }

    /// The document's timeline, revision and history depth, to show a rejected command changed nothing.
    fn snapshot(s: &Session) -> (Option<Timeline>, u64, usize) {
        let st = s.active().unwrap();
        (st.doc.timeline.clone(), st.revision, st.history.past_len())
    }

    fn current(s: &mut Session) -> Value {
        s.execute("timeline.info", json!({})).unwrap()["timeline"]["current"].clone()
    }

    /// #1009: `timeline.create` kept any `duration` (even `u64::MAX`) and `fps`, and stepping then
    /// panicked in `i64::clamp`. Bad values are now rejected before anything changes.
    #[test]
    fn create_rejects_out_of_range_params() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("timeline.create", json!({"duration": 24, "fps": 24.0})).unwrap();
        s.execute("timeline.setFrame", json!({"frame": 5})).unwrap();
        let before = snapshot(&s);
        let too_long = MAX_DURATION as u64 + 1;
        for p in [
            json!({"duration": 0}),
            json!({"duration": u64::MAX}),
            json!({"duration": i64::MAX as u64 + 1}),
            json!({"duration": too_long}),
            json!({"duration": -1}),
            json!({"duration": 2.5}),
            json!({"duration": "24"}),
            json!({"fps": 0.0}),
            json!({"fps": -24.0}),
            json!({"fps": MAX_FPS as f64 + 1.0}),
            json!({"fps": 1e300}),
            json!({"fps": "fast"}),
        ] {
            let e = s.execute("timeline.create", p.clone()).expect_err(&format!("{p} must be rejected"));
            assert!(matches!(e, EngineError::BadParams { .. }), "{p}: {e}");
            assert_eq!(snapshot(&s), before, "{p}: nothing changes");
        }
        // The issue's repro: stepping still works on the surviving timeline.
        s.execute("timeline.nextFrame", json!({})).unwrap();
        assert_eq!(current(&mut s), 6);
        s.execute("timeline.previousFrame", json!({})).unwrap();
        assert_eq!(current(&mut s), 5);
        // Control: the largest accepted values still create a timeline that steps at both ends.
        let r = s.execute("timeline.create", json!({"duration": MAX_DURATION, "fps": MAX_FPS as f64})).unwrap();
        assert_eq!(r["timeline"]["duration"], MAX_DURATION);
        s.execute("timeline.previousFrame", json!({})).unwrap();
        assert_eq!(current(&mut s), 0);
        s.execute("timeline.setFrame", json!({"frame": MAX_DURATION})).unwrap();
        s.execute("timeline.nextFrame", json!({})).unwrap();
        assert_eq!(current(&mut s), MAX_DURATION - 1);
        // Control: `null` means "use the default", as before.
        let r = s.execute("timeline.create", json!({"duration": null, "fps": null})).unwrap();
        assert_eq!((r["timeline"]["duration"].as_u64(), r["timeline"]["fps"].as_f64()), (Some(30), Some(30.0)));
    }

    /// #1009: `timeline.setProps` could store the same out-of-range `duration` and `fps`.
    #[test]
    fn set_props_rejects_out_of_range_params() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("timeline.create", json!({"duration": 24, "fps": 24.0})).unwrap();
        let before = snapshot(&s);
        for p in [json!({"duration": 0}), json!({"duration": u64::MAX}), json!({"fps": 0.0}), json!({"fps": -1.0}), json!({"fps": 30.0, "duration": 0})] {
            let e = s.execute("timeline.setProps", p.clone()).expect_err(&format!("{p} must be rejected"));
            assert!(matches!(e, EngineError::BadParams { .. }), "{p}: {e}");
            assert_eq!(snapshot(&s), before, "{p}: nothing changes");
        }
        s.execute("timeline.nextFrame", json!({})).unwrap();
        assert_eq!(current(&mut s), 1);
    }

    /// #1009: a timeline stored with `duration` 0 or `u64::MAX` (an old or hand-edited file) steps
    /// without panicking and lands on a valid frame.
    #[test]
    fn step_is_safe_on_any_stored_timeline() {
        for duration in [0, 1, usize::MAX] {
            for current in [0, usize::MAX] {
                for (cmd, delta) in [("timeline.nextFrame", 1i64), ("timeline.previousFrame", -1)] {
                    let mut s = Session::new();
                    s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
                    let st = s.active_mut().unwrap();
                    let mut doc = (*st.doc).clone();
                    doc.timeline = Some(Timeline { fps: f32::NAN, duration, current, work_start: usize::MAX, work_end: 0 });
                    st.doc = Arc::new(doc);
                    let r = s.execute(cmd, json!({})).unwrap_or_else(|e| panic!("{cmd} d={duration} c={current}: {e}"));
                    let t = &r["timeline"];
                    let d = t["duration"].as_u64().unwrap();
                    let c = t["current"].as_u64().unwrap();
                    assert!(d >= 1 && c < d, "{cmd} d={duration} c={current}: {t}");
                    if duration == usize::MAX && current == 0 {
                        assert_eq!(c, if delta > 0 { 1 } else { 0 });
                    }
                }
            }
        }
    }

    #[test]
    fn navigation_without_timeline_errors() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        assert!(s.execute("timeline.setFrame", json!({"frame": 1})).is_err());
    }
}
