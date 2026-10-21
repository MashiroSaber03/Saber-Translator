//! Magnetic Lasso: a selection border that snaps to edges between fastening points, traced by
//! `photocraft_algo::magnetic`.
//!
//! The tool is interactive (ui-egui `magnetic_lasso_ui`): it traces live through an
//! [`EdgeSource`] and commits the outline it showed with `select.magneticLasso` and
//! `"trace": false`, so an action replays exactly that outline. Given fastening points alone,
//! `select.magneticLasso` traces the whole border itself, for scripts and agents.

use std::sync::Arc;

use photocraft_algo::magnetic::{self, Settings, Tracer};
use photocraft_algo::selection as sel;
use photocraft_doc::{DocId, Document, LayerId};
use photocraft_geom::Rect;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{DocState, EngineError, Result, Session};

const CMD: &str = "select.magneticLasso";
/// Most points one command takes.
const MAX_POINTS: usize = 100_000;
/// Longest border one command traces (px, measured along the fastening points).
const MAX_LENGTH: f64 = 200_000.0;

/// The pixels the Magnetic Lasso follows: the active layer's when it is a pixel layer (Photoshop's
/// tool sees only the current layer), else the composite. A snapshot of one document revision,
/// read lazily and cached while a border is drawn.
pub struct EdgeSource {
    doc: Arc<Document>,
    layer: Option<LayerId>,
    revision: u64,
    tracer: Tracer,
}

/// The active layer when it has pixels of its own.
fn pixel_layer(d: &DocState) -> Option<LayerId> {
    d.active_layer.filter(|id| d.doc.layer(*id).is_some_and(|l| l.surface().is_some()))
}

impl EdgeSource {
    /// For the session's active document; `None` without one.
    pub fn new(s: &Session) -> Option<EdgeSource> {
        let d = s.active()?;
        Some(EdgeSource { doc: d.doc.clone(), layer: pixel_layer(d), revision: d.revision, tracer: Tracer::new(d.doc.bounds()) })
    }

    /// Whether this still shows the session's active document, layer and revision.
    pub fn is_current(&self, s: &Session) -> bool {
        s.active().is_some_and(|d| d.doc.id == self.doc.id && d.revision == self.revision && pixel_layer(d) == self.layer)
    }

    /// The document this reads.
    pub fn doc_id(&self) -> DocId {
        self.doc.id
    }

    /// Where a fastening point near `p` goes: on the most prominent edge within the detection
    /// width, else at `p` (see [`Tracer::snap`]).
    pub fn snap(&mut self, p: [f64; 2], s: Settings) -> Option<[f64; 2]> {
        let (doc, layer) = (&self.doc, self.layer);
        self.tracer.snap(&mut |r| read(doc, layer, r), p, s)
    }

    /// The border from `from` to `to` along the edges near `guide`, the pointer's path between them
    /// (see [`Tracer::trace`]).
    pub fn trace(&mut self, from: [f64; 2], to: [f64; 2], guide: &[[f64; 2]], s: Settings) -> Vec<[f64; 2]> {
        let (doc, layer) = (&self.doc, self.layer);
        self.tracer.trace(&mut |r| read(doc, layer, r), from, to, guide, s)
    }
}

/// Straight RGBA8 pixels of `r` from the layer, or the composite without one.
fn read(doc: &Document, layer: Option<LayerId>, r: Rect) -> Vec<[u8; 4]> {
    match layer.and_then(|id| doc.layer(id)).and_then(|l| l.surface()) {
        Some(surface) => sel::rgba8_image(surface, r),
        None => {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            photocraft_compose::render(doc, r).px.iter().map(|p| p.map(q)).collect()
        }
    }
}

fn bad(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: CMD.into(), msg: msg.into() }
}

/// `points`: `[[x, y], …]` of finite numbers.
fn points(p: &Value) -> Result<Vec<[f64; 2]>> {
    let a = p.get("points").and_then(Value::as_array).ok_or_else(|| bad("`points` must be [[x, y], …]"))?;
    if a.len() > MAX_POINTS {
        return Err(bad(format!("at most {MAX_POINTS} points")));
    }
    a.iter()
        .map(|q| match q.as_array().map(Vec::as_slice) {
            Some([x, y]) => match (x.as_f64(), y.as_f64()) {
                (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Ok([x, y]),
                _ => Err(bad(format!("bad point {q} (want [x, y])"))),
            },
            _ => Err(bad(format!("bad point {q} (want [x, y])"))),
        })
        .collect()
}

/// A number in `range`, or `default` when absent.
fn number(p: &Value, key: &str, default: f64, range: std::ops::RangeInclusive<f64>) -> Result<f64> {
    match p.get(key) {
        None => Ok(default),
        Some(v) => {
            v.as_f64().filter(|x| range.contains(x)).ok_or_else(|| bad(format!("`{key}` must be a number in {}..{} (got {v})", range.start(), range.end())))
        }
    }
}

fn magnetic_lasso(s: &mut Session, p: &Value) -> Result<Value> {
    let pts = points(p)?;
    if pts.len() < 3 {
        return Err(bad("needs at least 3 points"));
    }
    let trace = match p.get("trace") {
        None => true,
        Some(v) => v.as_bool().ok_or_else(|| bad("`trace` must be true or false"))?,
    };
    let outline = if trace {
        let width = number(p, "width", 10.0, magnetic::WIDTH_RANGE.0..=magnetic::WIDTH_RANGE.1)?;
        let contrast = number(p, "contrast", 10.0, 1.0..=100.0)?;
        let straight = match p.get("close").map(Value::as_str) {
            None | Some(Some("magnetic")) => false,
            Some(Some("straight")) => true,
            Some(_) => return Err(bad("`close` must be \"magnetic\" or \"straight\"")),
        };
        let settings = Settings::new(width, (contrast / 100.0) as f32);
        let mut src = EdgeSource::new(s).ok_or(EngineError::NoDocument)?;
        trace_border(&mut src, &pts, settings, straight)?
    } else {
        pts
    };
    let polygon: Vec<(f32, f32)> = outline.iter().map(|q| (q[0] as f32, q[1] as f32)).collect();
    let mut r = crate::selection_cmds::polygon_selection(s, CMD, "Magnetic Lasso", &polygon, p)?;
    r["points"] = json!(outline.len());
    Ok(r)
}

/// The closed border through the fastening points `pts`, each snapped to the nearest prominent
/// edge first, as the tool places them. The last point joins the first along the edges, or
/// straight with `straight`.
fn trace_border(src: &mut EdgeSource, pts: &[[f64; 2]], settings: Settings, straight: bool) -> Result<Vec<[f64; 2]>> {
    let b = src.tracer.bounds();
    let inside = |q: &[f64; 2]| [q[0].clamp(b.x0 as f64, b.x1 as f64), q[1].clamp(b.y0 as f64, b.y1 as f64)];
    let mut ring: Vec<[f64; 2]> = pts.iter().map(inside).collect();
    if let Some(&first) = ring.first() {
        ring.push(first);
    }
    if magnetic::length(&ring) > MAX_LENGTH {
        return Err(bad(format!("the border is longer than {MAX_LENGTH} px")));
    }
    ring.pop();
    let anchors: Vec<[f64; 2]> = ring.iter().filter_map(|q| src.snap(*q, settings)).collect();
    let Some(&first) = anchors.first() else { return Err(EngineError::Other("the document has no pixels".into())) };
    let mut out = vec![first];
    for w in anchors.windows(2) {
        out.extend(src.trace(w[0], w[1], &[], settings).into_iter().skip(1));
    }
    if !straight && let Some(&last) = anchors.last() {
        out.extend(src.trace(last, first, &[], settings).into_iter().skip(1));
        // The ring ends where it started.
        out.pop();
    }
    Ok(out)
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// Magnetic Lasso command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Magnetic Lasso",
        menu: &[],
        shortcut: None,
        params: r##"{"points":[[x,y],…] (≥3 fastening points in order; with trace=false, the finished outline),"width":1..256=10 (px: follows edges this close to the points),"contrast":1..100=10 (%: weaker edges are ignored),"close":"magnetic|straight"="magnetic" (how the last point joins the first),"trace":bool=true,"mode":"replace|add|subtract|intersect"="replace","antiAlias":bool=true,"feather":0..1000=0 (px)}"##,
        enabled: has_doc,
        run: magnetic_lasso,
        journal: true,
    }]
}

#[cfg(test)]
mod tests;
