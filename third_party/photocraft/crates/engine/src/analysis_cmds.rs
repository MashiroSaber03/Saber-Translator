//! Image › Analysis: measurement scale, data points, Record Measurements and the Measurement
//! Log, the Ruler tool (with protractor and Straighten Layer), the Count tool and Place Scale
//! Marker.
//!
//! The scale, count groups and ruler line are document data ([`photocraft_doc::Measurement`],
//! saved in `.pcraft`; the scale also as PSD resource 1074). The Measurement Log and the data
//! point choices are session-wide, as in Photoshop.
//!
//! Selection measurements follow Photoshop's definitions: the selection (coverage ≥ 50 %) is
//! split into 8-connected features, each recorded with a summary row first. Area is the pixel
//! count, perimeter the length of the marching-squares contour through the pixel-centre grid
//! (exact along straight edges, √2 per diagonal step), circularity 4π·area/perimeter², gray
//! values the luminosity (0.30 R + 0.59 G + 0.11 B) of the composite in the document's value
//! range (0–255 at 8 bits, 0–32768 at 16 bits, 0–1 at 32 bits), integrated density area × mean.
//! Lengths and areas are scaled by the measurement scale.

use std::sync::Arc;

use photocraft_color::{Color, SampleType};
use photocraft_doc::{CountGroup, Document, Layer, MeasurementScale, Ruler};
use photocraft_geom::Rect;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

// ------------------------------------------------------------------ state

/// Data point columns: (key, Photoshop column header).
pub const COLUMNS: &[(&str, &str)] = &[
    ("label", "Label"),
    ("dateTime", "Date and Time"),
    ("document", "Document"),
    ("source", "Source"),
    ("scale", "Scale"),
    ("scaleUnits", "Scale Units"),
    ("scaleFactor", "Scale Factor"),
    ("count", "Count"),
    ("area", "Area"),
    ("perimeter", "Perimeter"),
    ("circularity", "Circularity"),
    ("height", "Height"),
    ("width", "Width"),
    ("grayMin", "Gray Value (Minimum)"),
    ("grayMax", "Gray Value (Maximum)"),
    ("grayMean", "Gray Value (Mean)"),
    ("grayMedian", "Gray Value (Median)"),
    ("integratedDensity", "Integrated Density"),
    ("histogram", "Histogram"),
    ("length", "Length"),
    ("angle", "Angle"),
];

const COMMON: &[&str] = &["label", "dateTime", "document", "source", "scale", "scaleUnits", "scaleFactor"];
const SELECTION_ONLY: &[&str] =
    &["count", "area", "perimeter", "circularity", "height", "width", "grayMin", "grayMax", "grayMean", "grayMedian", "integratedDensity", "histogram"];
const RULER_ONLY: &[&str] = &["length", "angle"];
const COUNT_ONLY: &[&str] = &["count"];

/// Data points a measurement source can record.
pub fn available(source: &str) -> Vec<&'static str> {
    let extra = match source {
        "selection" => SELECTION_ONLY,
        "ruler" => RULER_ONLY,
        _ => COUNT_ONLY,
    };
    COMMON.iter().chain(extra).copied().collect()
}

/// Image › Analysis › Select Data Points: what Record Measurements writes per source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DataPoints {
    pub selection: Vec<String>,
    pub ruler: Vec<String>,
    pub count: Vec<String>,
}

impl Default for DataPoints {
    fn default() -> Self {
        let all = |s| available(s).into_iter().map(String::from).collect();
        Self { selection: all("selection"), ruler: all("ruler"), count: all("count") }
    }
}

impl DataPoints {
    fn of(&self, source: &str) -> &[String] {
        match source {
            "selection" => &self.selection,
            "ruler" => &self.ruler,
            _ => &self.count,
        }
    }
    fn of_mut(&mut self, source: &str) -> &mut Vec<String> {
        match source {
            "selection" => &mut self.selection,
            "ruler" => &mut self.ruler,
            _ => &mut self.count,
        }
    }
}

/// One Measurement Log row.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LogRow {
    pub id: u64,
    /// Data point values by key (see [`COLUMNS`]).
    pub values: Map<String, Value>,
}

/// Session-wide analysis state (`Session::analysis`).
#[derive(Clone, Debug, Default)]
pub struct AnalysisState {
    pub log: Vec<LogRow>,
    next_row: u64,
    /// Counter for "Measurement N" labels.
    next_measurement: u64,
    pub data_points: DataPoints,
}

// ------------------------------------------------------------------ helpers

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

fn has_ruler(s: &Session) -> std::result::Result<(), String> {
    has_doc(s)?;
    s.active().and_then(|d| d.doc.measurement.ruler).map(|_| ()).ok_or_else(|| "no ruler line".into())
}

fn has_log(s: &Session) -> std::result::Result<(), String> {
    if s.analysis.log.is_empty() { Err("the Measurement Log is empty".into()) } else { Ok(()) }
}

fn point(v: Option<&Value>) -> Option<[f64; 2]> {
    let a = v?.as_array()?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
}

fn xy(p: &Value) -> Option<[f64; 2]> {
    Some([p.get("x")?.as_f64()?, p.get("y")?.as_f64()?])
}

fn doc(s: &Session) -> Result<Arc<Document>> {
    Ok(s.active().ok_or(EngineError::NoDocument)?.doc.clone())
}

/// Change the document without a history step (ruler line: view-like state that is still saved
/// with the document). A clean document stays clean.
fn set_quiet(s: &mut Session, f: impl FnOnce(&mut Document)) -> Result<()> {
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    let mut d = (*st.doc).clone();
    f(&mut d);
    st.doc = Arc::new(d);
    let clean = st.saved_revision == st.revision;
    st.revision += 1;
    if clean {
        st.saved_revision = st.revision;
    }
    st.last_damage = Some(Rect::EMPTY);
    Ok(())
}

/// Run several commands as one history step labelled `label`. On failure the document goes back
/// as it was and the journal forgets the commands `f` ran, so an action being recorded does not
/// replay an edit that was taken back.
pub(crate) fn compound<R>(s: &mut Session, label: &str, f: impl FnOnce(&mut Session) -> Result<R>) -> Result<R> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let (before, history, prior) = (st.doc.clone(), st.history.clone(), st.layer_target());
    let journal = s.journal.len();
    // Color Sampler points are view state a step may move (crop, rotation): roll them back too, so
    // a failed second step can't leave them pinned to pixels the restored document no longer has.
    let samplers = st.color_samplers.clone();
    let r = f(s);
    if r.is_err() {
        s.journal.truncate(journal);
    }
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    st.history = history;
    match r {
        Ok(v) => {
            // As in `Session::edit`: undo targets what was selected just before the step (#1356).
            st.history.set_current_layers(prior);
            st.history.record(label, before, st.layer_target());
            st.history.trim(&st.doc);
            st.coalesce = None;
            Ok(v)
        }
        Err(e) => {
            st.doc = before;
            st.color_samplers = samplers;
            st.revision += 1;
            st.last_damage = None;
            Err(e)
        }
    }
}

/// Current time as ISO 8601 UTC (`2026-10-01T12:34:56Z`); empty on the web (no clock).
pub fn now_iso() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0);
        let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
        // Civil date from days since 1970-01-01 (Howard Hinnant's algorithm).
        let z = days + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        let y = yoe + era * 400 + i64::from(m <= 2);
        format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
    }
    #[cfg(target_arch = "wasm32")]
    {
        String::new()
    }
}

fn round4(v: f64) -> f64 {
    (v * 10_000.0).round() / 10_000.0
}

// ------------------------------------------------------------------ scale

fn set_scale(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.analysis.setMeasurementScale";
    let cur = doc(s)?.measurement.scale.clone();
    let preset = p.get("preset").and_then(Value::as_str).unwrap_or(if p.as_object().is_none_or(|o| o.is_empty()) { "query" } else { "custom" });
    let next = match preset {
        "query" => return Ok(scale_json(&cur)),
        "default" => MeasurementScale::default(),
        "custom" => {
            let pl = p.get("pixelLength").and_then(Value::as_f64).unwrap_or(cur.pixel_length);
            let ll = p.get("logicalLength").and_then(Value::as_f64).unwrap_or(cur.logical_length);
            let units = p.get("units").or_else(|| p.get("logicalUnits")).and_then(Value::as_str).map(str::to_string).unwrap_or(cur.units.clone());
            if !(pl > 0.0 && pl.is_finite() && ll > 0.0 && ll.is_finite()) {
                return Err(bad(CMD, "pixelLength and logicalLength must be positive"));
            }
            if units.trim().is_empty() {
                return Err(bad(CMD, "units must not be empty"));
            }
            MeasurementScale { pixel_length: pl, logical_length: ll, units }
        }
        other => return Err(bad(CMD, format!("unknown preset `{other}` (default|custom)"))),
    };
    if next != cur {
        let n = next.clone();
        s.edit("Set Measurement Scale", move |d, _| {
            d.measurement.scale = n;
            Ok(())
        })?;
    }
    Ok(scale_json(&next))
}

fn scale_json(sc: &MeasurementScale) -> Value {
    json!({"pixelLength": sc.pixel_length, "logicalLength": sc.logical_length, "units": sc.units, "factor": sc.factor(), "text": sc.describe(), "default": sc.is_default()})
}

// ------------------------------------------------------------------ data points

fn select_data_points(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.analysis.selectDataPoints";
    let mut next = if p.get("reset").and_then(Value::as_bool) == Some(true) { DataPoints::default() } else { s.analysis.data_points.clone() };
    for source in ["selection", "ruler", "count"] {
        let avail = available(source);
        let Some(v) = p.get(source) else { continue };
        let list = next.of_mut(source);
        match v {
            // A full list of keys…
            Value::Array(a) => {
                let mut keys = Vec::new();
                for k in a {
                    let k = k.as_str().ok_or_else(|| bad(CMD, format!("`{source}` entries must be strings")))?;
                    if !avail.contains(&k) {
                        return Err(bad(CMD, format!("`{k}` is not a {source} data point ({})", avail.join("|"))));
                    }
                    keys.push(k.to_string());
                }
                *list = keys;
            }
            // …or {key: on/off} changes.
            Value::Object(o) => {
                for (k, on) in o {
                    if !avail.contains(&k.as_str()) {
                        return Err(bad(CMD, format!("`{k}` is not a {source} data point ({})", avail.join("|"))));
                    }
                    let on = on.as_bool().ok_or_else(|| bad(CMD, format!("`{source}.{k}` must be a bool")))?;
                    list.retain(|x| x != k);
                    if on {
                        list.push(k.clone());
                    }
                }
            }
            _ => return Err(bad(CMD, format!("`{source}` must be a list of keys or an object of bools"))),
        }
        // Keep the column order.
        list.sort_by_key(|k| COLUMNS.iter().position(|c| c.0 == k));
    }
    s.analysis.data_points = next;
    Ok(
        json!({"dataPoints": s.analysis.data_points, "available": {"selection": available("selection"), "ruler": available("ruler"), "count": available("count")}, "columns": COLUMNS.iter().map(|c| json!({"key": c.0, "name": c.1})).collect::<Vec<_>>()}),
    )
}

// ------------------------------------------------------------------ selection measurements

/// Measurements of one selection feature (or the summary of all).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Feature {
    pub area: f64,
    pub perimeter: f64,
    pub bounds: Rect,
    pub gray_min: f64,
    pub gray_max: f64,
    pub gray_mean: f64,
    pub gray_median: f64,
    pub histogram: Vec<u64>,
}

impl Feature {
    pub fn circularity(&self) -> f64 {
        if self.perimeter > 0.0 { (4.0 * std::f64::consts::PI * self.area / (self.perimeter * self.perimeter)).min(1.0) } else { 0.0 }
    }
}

/// Union-find root with path halving.
fn root(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        parent[i as usize] = parent[parent[i as usize] as usize];
        i = parent[i as usize];
    }
    i
}

/// Gray value range of a depth: what a "gray value" of 1.0 reads as.
fn gray_range(depth: SampleType) -> f64 {
    match depth {
        SampleType::U8 => 255.0,
        SampleType::U16 => 32768.0,
        _ => 1.0,
    }
}

/// Split `mask` (w×h, true = selected) into 8-connected features and measure each against the
/// gray image `gray` (0–1). Returns features in scan order of their first pixel; `origin` offsets
/// the reported bounds.
pub fn measure_features(mask: &[bool], gray: &[f32], w: usize, h: usize, origin: (i32, i32), range: f64) -> Vec<Feature> {
    // Two-pass connected components.
    let mut label = vec![u32::MAX; w * h];
    let mut parent: Vec<u32> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            if !mask[i] {
                continue;
            }
            let mut best = u32::MAX;
            let mut nbrs = [u32::MAX; 4];
            if x > 0 {
                nbrs[0] = label[i - 1];
            }
            if y > 0 {
                nbrs[1] = label[i - w];
                if x > 0 {
                    nbrs[2] = label[i - w - 1];
                }
                if x + 1 < w {
                    nbrs[3] = label[i - w + 1];
                }
            }
            for &n in &nbrs {
                if n != u32::MAX {
                    let r = root(&mut parent, n);
                    best = best.min(r);
                }
            }
            if best == u32::MAX {
                best = parent.len() as u32;
                parent.push(best);
            } else {
                for &n in &nbrs {
                    if n != u32::MAX {
                        let r = root(&mut parent, n);
                        if r != best {
                            let (a, b) = (r.min(best), r.max(best));
                            parent[b as usize] = a;
                            best = a;
                        }
                    }
                }
            }
            label[i] = best;
        }
    }
    // Compact roots → feature indices in order of first appearance.
    let mut index = vec![u32::MAX; parent.len()];
    let mut n = 0u32;
    for l in label.iter_mut() {
        if *l == u32::MAX {
            continue;
        }
        let r = root(&mut parent, *l) as usize;
        if index[r] == u32::MAX {
            index[r] = n;
            n += 1;
        }
        *l = index[r];
    }
    let n = n as usize;
    // Perimeter: marching squares over the pixel-centre grid (padded with unselected pixels),
    // per row band in parallel, each cell's length attributed to the feature of a pixel in it.
    let at = |x: isize, y: isize| -> u32 { if x < 0 || y < 0 || x >= w as isize || y >= h as isize { u32::MAX } else { label[y as usize * w + x as usize] } };
    const HALF_DIAG: f64 = std::f64::consts::FRAC_1_SQRT_2;
    let per: Vec<f64> = (-1..h as isize)
        .into_par_iter()
        .fold(
            || vec![0.0f64; n],
            |mut acc, y| {
                for x in -1..w as isize {
                    let c = [at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1)];
                    let inside = c.iter().filter(|v| **v != u32::MAX).count();
                    let len = match inside {
                        1 | 3 => HALF_DIAG,
                        2 if (c[0] != u32::MAX) == (c[2] != u32::MAX) => 2.0 * HALF_DIAG,
                        2 => 1.0,
                        _ => 0.0,
                    };
                    if len > 0.0
                        && let Some(f) = c.iter().find(|v| **v != u32::MAX)
                    {
                        acc[*f as usize] += len;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0.0f64; n],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    // Area, bounds and gray values.
    let mut feats: Vec<Feature> =
        (0..n).map(|_| Feature { bounds: Rect::EMPTY, gray_min: f64::MAX, gray_max: f64::MIN, histogram: vec![0; 256], ..Default::default() }).collect();
    let mut sums = vec![0.0f64; n];
    let mut values: Vec<Vec<u16>> = vec![Vec::new(); n];
    for y in 0..h {
        for x in 0..w {
            let l = label[y * w + x];
            if l == u32::MAX {
                continue;
            }
            let f = &mut feats[l as usize];
            f.area += 1.0;
            let (px, py) = (x as i32 + origin.0, y as i32 + origin.1);
            let r = Rect::new(px, py, px + 1, py + 1);
            f.bounds = if f.bounds.is_empty() { r } else { f.bounds.union(&r) };
            let g = f64::from(gray[y * w + x]);
            f.gray_min = f.gray_min.min(g);
            f.gray_max = f.gray_max.max(g);
            sums[l as usize] += g;
            f.histogram[(g.clamp(0.0, 1.0) * 255.0).round() as usize] += 1;
            values[l as usize].push((g.clamp(0.0, 1.0) * 65535.0).round() as u16);
        }
    }
    for (i, f) in feats.iter_mut().enumerate() {
        f.perimeter = per[i];
        f.gray_mean = sums[i] / f.area.max(1.0) * range;
        f.gray_min *= range;
        f.gray_max *= range;
        f.gray_median = median(&mut values[i]) * range;
    }
    feats
}

fn median(v: &mut [u16]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mid = v.len() / 2;
    let (_, m, _) = v.select_nth_unstable(mid);
    let m = f64::from(*m);
    let m = if v.len().is_multiple_of(2) { (m + f64::from(*v[..mid].iter().max().unwrap_or(&0))) / 2.0 } else { m };
    m / 65535.0
}

/// Summary over features (area-weighted mean, pooled median).
fn summary(feats: &[Feature], all_values_median: f64) -> Feature {
    let area: f64 = feats.iter().map(|f| f.area).sum();
    let mut hist = vec![0u64; 256];
    let mut b = Rect::EMPTY;
    for f in feats {
        hist.iter_mut().zip(&f.histogram).for_each(|(a, v)| *a += v);
        b = if b.is_empty() { f.bounds } else { b.union(&f.bounds) };
    }
    Feature {
        area,
        perimeter: feats.iter().map(|f| f.perimeter).sum(),
        bounds: b,
        gray_min: feats.iter().map(|f| f.gray_min).fold(f64::MAX, f64::min),
        gray_max: feats.iter().map(|f| f.gray_max).fold(f64::MIN, f64::max),
        gray_mean: feats.iter().map(|f| f.gray_mean * f.area).sum::<f64>() / area.max(1.0),
        gray_median: all_values_median,
        histogram: hist,
    }
}

/// Largest area Record Measurements handles in one pass: the gray render alone is 16 bytes per
/// pixel, and the label buffer and mask come on top (#933).
pub const MAX_MEASURE_PIXELS: u64 = 400_000_000;

/// Selection mask (coverage ≥ 0.5) over `r` (`n` pixels), or every pixel when there is no selection.
fn selection_mask(d: &Document, r: Rect, n: usize) -> Vec<bool> {
    match &d.selection {
        None => vec![true; n],
        Some(sel) => {
            let c = sel.channels().max(1);
            let v = sel.read_region(r);
            v.chunks_exact(c).map(|p| p[0] >= 0.5).collect()
        }
    }
}

fn gray_image(d: &Document, r: Rect) -> Vec<f32> {
    let buf = photocraft_compose::render(d, r);
    buf.px.par_iter().map(|p| 0.30 * p[0] + 0.59 * p[1] + 0.11 * p[2]).collect()
}

fn selection_rows(d: &Document) -> Result<(Feature, Vec<Feature>)> {
    let r = match &d.selection {
        Some(sel) => sel.content_bounds().intersect(&d.bounds()),
        None => d.bounds(),
    };
    if r.is_empty() {
        return Err(EngineError::Other("the selection is empty".into()));
    }
    // A legal 65536×65536 document is 2^32 pixels: size everything from a checked count,
    // and refuse what cannot be measured in one pass instead of overflowing (#933).
    let n = u64::from(r.width()) * u64::from(r.height());
    let Some(n) = (n <= MAX_MEASURE_PIXELS).then(|| usize::try_from(n).ok()).flatten() else {
        return Err(EngineError::Other(format!(
            "the area is too large to measure ({}×{} px; max {} megapixels): select a smaller region",
            r.width(),
            r.height(),
            MAX_MEASURE_PIXELS / 1_000_000
        )));
    };
    let (w, h) = (r.width() as usize, r.height() as usize);
    let mask = selection_mask(d, r, n);
    let gray = gray_image(d, r);
    let range = gray_range(d.depth);
    let feats = measure_features(&mask, &gray, w, h, (r.x0, r.y0), range);
    if feats.is_empty() {
        return Err(EngineError::Other("the selection is empty".into()));
    }
    let mut all: Vec<u16> = mask.iter().zip(&gray).filter(|(m, _)| **m).map(|(_, g)| (g.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
    let med = median(&mut all) * range;
    Ok((summary(&feats, med), feats))
}

// ------------------------------------------------------------------ ruler

/// Ruler readout: X/Y (start), W/H, angle A and lengths L1/L2, as in the options bar.
pub fn ruler_info(r: &Ruler, sc: &MeasurementScale) -> Value {
    let (dx, dy) = (r.end[0] - r.start[0], r.end[1] - r.start[1]);
    let l1 = dx.hypot(dy);
    let f = sc.factor();
    match r.protractor {
        None => {
            // Counter-clockwise from the +x axis with y up, −180…180.
            let a = (-dy).atan2(dx).to_degrees();
            json!({"x": r.start[0], "y": r.start[1], "w": dx, "h": dy, "angle": round4(a), "l1": round4(l1), "l2": Value::Null, "length": round4(l1 * f), "units": sc.units, "protractor": false})
        }
        Some(q) => {
            let (ex, ey) = (q[0] - r.start[0], q[1] - r.start[1]);
            let l2 = ex.hypot(ey);
            let a1 = (-dy).atan2(dx);
            let a2 = (-ey).atan2(ex);
            let mut a = (a1 - a2).to_degrees().abs();
            if a > 180.0 {
                a = 360.0 - a;
            }
            json!({"x": r.start[0], "y": r.start[1], "w": dx, "h": dy, "angle": round4(a), "l1": round4(l1), "l2": round4(l2), "length": round4(l1 * f), "units": sc.units, "protractor": true})
        }
    }
}

fn ruler_tool(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.analysis.rulerTool";
    let d = doc(s)?;
    if p.get("clear").and_then(Value::as_bool) == Some(true) {
        set_quiet(s, |d| d.measurement.ruler = None)?;
        return Ok(json!({"ruler": Value::Null}));
    }
    let start = point(p.get("start"));
    let end = point(p.get("end"));
    let mut r = d.measurement.ruler;
    match (start, end) {
        (Some(a), Some(b)) => {
            let prot = match p.get("protractor") {
                Some(Value::Null) | None => None,
                Some(v) => Some(point(Some(v)).ok_or_else(|| bad(CMD, "`protractor` must be [x, y] or null"))?),
            };
            r = Some(Ruler { start: a, end: b, protractor: prot });
        }
        (None, None) => {
            if let (Some(cur), Some(v)) = (r.as_mut(), p.get("protractor")) {
                cur.protractor = if v.is_null() { None } else { Some(point(Some(v)).ok_or_else(|| bad(CMD, "`protractor` must be [x, y] or null"))?) };
            }
        }
        _ => return Err(bad(CMD, "pass both `start` and `end`")),
    }
    for v in r.iter().flat_map(|r| [Some(r.start), Some(r.end), r.protractor]).flatten() {
        if !v[0].is_finite() || !v[1].is_finite() {
            return Err(bad(CMD, "coordinates must be finite"));
        }
    }
    if r != d.measurement.ruler {
        set_quiet(s, |d| d.measurement.ruler = r)?;
    }
    Ok(match r {
        Some(r) => json!({"ruler": {"start": r.start, "end": r.end, "protractor": r.protractor}, "info": ruler_info(&r, &d.measurement.scale)}),
        None => json!({"ruler": Value::Null}),
    })
}

/// Largest axis-aligned rectangle of a `w`×`h` rectangle rotated by `a` radians.
fn inscribed(w: f64, h: f64, a: f64) -> (f64, f64) {
    let (sa, ca) = (a.sin().abs(), a.cos().abs());
    let long_w = w >= h;
    let (long, short) = if long_w { (w, h) } else { (h, w) };
    if short <= 2.0 * sa * ca * long || (sa - ca).abs() < 1e-10 {
        let x = 0.5 * short;
        if long_w { (x / sa, x / ca) } else { (x / ca, x / sa) }
    } else {
        let c2 = ca * ca - sa * sa;
        ((w * ca - h * sa) / c2, (h * ca - w * sa) / c2)
    }
}

/// Clockwise rotation in degrees that turns the ruler line onto its nearest axis.
pub fn straighten_angle(r: &Ruler) -> f64 {
    let (dx, dy) = (r.end[0] - r.start[0], r.end[1] - r.start[1]);
    let a = (-dy).atan2(dx).to_degrees();
    a - 90.0 * (a / 90.0).round()
}

fn straighten(s: &mut Session, p: &Value) -> Result<Value> {
    let d = doc(s)?;
    let r = d.measurement.ruler.ok_or_else(|| EngineError::Other("no ruler line".into()))?;
    let rot = straighten_angle(&r);
    if rot.abs() < 1e-9 {
        return Ok(json!({"angle": 0.0}));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let active_bg = st.active_layer.and_then(|id| d.layer(id)).is_some_and(|l| d.is_background(l.id));
    let crop = p.get("crop").and_then(Value::as_bool).unwrap_or(active_bg || st.active_layer.is_none());
    let (w0, h0) = (f64::from(d.size.width), f64::from(d.size.height));
    compound(s, "Straighten", |s| {
        if crop {
            s.execute("image.rotation.arbitrary", json!({"angle": rot, "direction": "cw"}))?;
            let nd = doc(s)?;
            let (cw, ch) = inscribed(w0, h0, rot.to_radians());
            let (nw, nh) = (f64::from(nd.size.width), f64::from(nd.size.height));
            let (cw, ch) = (cw.floor().clamp(1.0, nw), ch.floor().clamp(1.0, nh));
            s.execute("image.crop", json!({"x": ((nw - cw) / 2.0).round(), "y": ((nh - ch) / 2.0).round(), "width": cw, "height": ch}))?;
        } else {
            let (cx, cy) = (w0 / 2.0, h0 / 2.0);
            let (sn, cs) = rot.to_radians().sin_cos();
            // Clockwise on screen (y down) about the canvas centre.
            let m = [cs, sn, -sn, cs, cx - (cs * cx - sn * cy), cy - (sn * cx + cs * cy)];
            s.execute("edit.transform", json!({"matrix": m}))?;
        }
        // The ruler is cleared after straightening, as in Photoshop.
        set_quiet(s, |d| d.measurement.ruler = None)
    })?;
    let nd = doc(s)?;
    Ok(json!({"angle": round4(rot), "cropped": crop, "width": nd.size.width, "height": nd.size.height}))
}

// ------------------------------------------------------------------ count

fn group_index(d: &Document, p: &Value) -> Option<usize> {
    let i = p.get("group").and_then(Value::as_u64).map(|v| v as usize).unwrap_or(d.measurement.active_count_group);
    (i < d.measurement.count_groups.len()).then_some(i)
}

fn color_of(p: &Value, key: &str) -> Option<Color> {
    p.get(key)?;
    let c = crate::commands::color_param(p, key, [1.0, 0.0, 0.0, 1.0]);
    Some(Color::rgb(c[0], c[1], c[2]))
}

/// Photoshop's colours for new count groups, in order.
const GROUP_COLORS: [[f32; 3]; 6] = [[1.0, 0.0, 0.0], [0.0, 0.6, 1.0], [0.0, 0.8, 0.0], [1.0, 0.6, 0.0], [0.8, 0.0, 0.8], [1.0, 1.0, 0.0]];

fn new_group(d: &Document, name: Option<String>, color: Option<Color>) -> CountGroup {
    let n = d.measurement.count_groups.len();
    let c = GROUP_COLORS[n % GROUP_COLORS.len()];
    CountGroup { name: name.unwrap_or_else(|| format!("Count Group {}", n + 1)), color: color.unwrap_or(Color::rgb(c[0], c[1], c[2])), ..Default::default() }
}

fn count_json(d: &Document) -> Value {
    let m = &d.measurement;
    json!({
        "total": m.count_total(),
        "activeGroup": m.active_count_group,
        "groups": m.count_groups.iter().map(|g| json!({"name": g.name, "color": g.color.to_rgb(), "markerSize": g.marker_size, "labelSize": g.label_size, "visible": g.visible, "count": g.points.len(), "points": g.points})).collect::<Vec<_>>(),
    })
}

fn count_tool(s: &mut Session, _p: &Value) -> Result<Value> {
    Ok(count_json(doc(s)?.as_ref()))
}

fn count_add(s: &mut Session, p: &Value) -> Result<Value> {
    let at = xy(p).or_else(|| point(p.get("point"))).ok_or_else(|| bad("count.add", "pass `x` and `y`"))?;
    if !at[0].is_finite() || !at[1].is_finite() {
        return Err(bad("count.add", "coordinates must be finite"));
    }
    let gi = p.get("group").and_then(Value::as_u64).map(|v| v as usize);
    let n = s.edit("Count", |d, _| {
        if d.measurement.count_groups.is_empty() {
            let g = new_group(d, None, None);
            d.measurement.count_groups.push(g);
            d.measurement.active_count_group = 0;
        }
        let i = gi.unwrap_or(d.measurement.active_count_group).min(d.measurement.count_groups.len() - 1);
        let g = &mut d.measurement.count_groups[i];
        g.points.push(at);
        Ok(g.points.len())
    })?;
    let mut r = count_json(doc(s)?.as_ref());
    r["number"] = json!(n);
    Ok(r)
}

/// Marker nearest `at` within `radius` px in group `g` (or any visible group).
fn nearest(d: &Document, at: [f64; 2], radius: f64, g: Option<usize>) -> Option<(usize, usize)> {
    let mut best: Option<(f64, usize, usize)> = None;
    for (gi, grp) in d.measurement.count_groups.iter().enumerate() {
        if g.is_some_and(|x| x != gi) || !grp.visible {
            continue;
        }
        for (pi, q) in grp.points.iter().enumerate() {
            let dd = (q[0] - at[0]).hypot(q[1] - at[1]);
            if dd <= radius && best.is_none_or(|b| dd < b.0) {
                best = Some((dd, gi, pi));
            }
        }
    }
    best.map(|b| (b.1, b.2))
}

fn locate(s: &Session, p: &Value, cmd: &str) -> Result<(usize, usize)> {
    let d = s.active().ok_or(EngineError::NoDocument)?.doc.clone();
    if let Some(i) = p.get("index").and_then(Value::as_u64) {
        let g = group_index(&d, p).ok_or_else(|| bad(cmd, "no such count group"))?;
        if (i as usize) < d.measurement.count_groups[g].points.len() {
            return Ok((g, i as usize));
        }
        return Err(bad(cmd, format!("no marker {i}")));
    }
    let at = xy(p).ok_or_else(|| bad(cmd, "pass `index` or `x`/`y`"))?;
    let radius = p.get("radius").and_then(Value::as_f64).unwrap_or(6.0);
    let g = p.get("group").and_then(Value::as_u64).map(|v| v as usize);
    nearest(&d, at, radius, g).ok_or_else(|| EngineError::Other("no count marker there".into()))
}

fn count_remove(s: &mut Session, p: &Value) -> Result<Value> {
    let (g, i) = locate(s, p, "count.remove")?;
    s.edit("Count", |d, _| {
        d.measurement.count_groups[g].points.remove(i);
        Ok(())
    })?;
    Ok(count_json(doc(s)?.as_ref()))
}

fn count_move(s: &mut Session, p: &Value) -> Result<Value> {
    let to = point(p.get("to")).ok_or_else(|| bad("count.move", "pass `to`: [x, y]"))?;
    let (g, i) = locate(s, p, "count.move")?;
    s.edit("Count", |d, _| {
        d.measurement.count_groups[g].points[i] = to;
        Ok(())
    })?;
    Ok(count_json(doc(s)?.as_ref()))
}

fn count_clear(s: &mut Session, p: &Value) -> Result<Value> {
    let d = doc(s)?;
    let all = p.get("group").and_then(Value::as_str) == Some("all");
    let g = if all { None } else { Some(group_index(&d, p).ok_or_else(|| bad("count.clear", "no such count group"))?) };
    s.edit("Clear Count", |d, _| {
        for (i, grp) in d.measurement.count_groups.iter_mut().enumerate() {
            if g.is_none_or(|g| g == i) {
                grp.points.clear();
            }
        }
        Ok(())
    })?;
    Ok(count_json(doc(s)?.as_ref()))
}

fn count_new_group(s: &mut Session, p: &Value) -> Result<Value> {
    let name = p.get("name").and_then(Value::as_str).map(str::to_string);
    let color = color_of(p, "color");
    s.edit("New Count Group", |d, _| {
        let g = new_group(d, name, color);
        d.measurement.count_groups.push(g);
        d.measurement.active_count_group = d.measurement.count_groups.len() - 1;
        Ok(())
    })?;
    Ok(count_json(doc(s)?.as_ref()))
}

fn count_delete_group(s: &mut Session, p: &Value) -> Result<Value> {
    let d = doc(s)?;
    let g = group_index(&d, p).ok_or_else(|| bad("count.deleteGroup", "no such count group"))?;
    s.edit("Delete Count Group", |d, _| {
        d.measurement.count_groups.remove(g);
        let n = d.measurement.count_groups.len();
        d.measurement.active_count_group = d.measurement.active_count_group.min(n.saturating_sub(1));
        Ok(())
    })?;
    Ok(count_json(doc(s)?.as_ref()))
}

/// A size parameter (`markerSize`, `labelSize`) clamped to `lo..=hi`: `None` when absent or null.
/// Whole numbers sent as JSON floats (`8.0`, as UIs and recorded actions do) count; fractions,
/// negatives and non-numbers are bad params rather than silently ignored. Clamping happens in
/// `u64`/`f64`, so a huge value can't wrap through `as u32`.
fn whole_size(cmd: &str, p: &Value, key: &str, lo: u32, hi: u32) -> Result<Option<u32>> {
    let v = match p.get(key) {
        None | Some(Value::Null) => return Ok(None),
        Some(v) => v,
    };
    let n = match (v.as_u64(), v.as_f64()) {
        (Some(n), _) => n.clamp(u64::from(lo), u64::from(hi)),
        (None, Some(f)) if f.is_finite() && f >= 0.0 && f.fract() == 0.0 => f.clamp(f64::from(lo), f64::from(hi)) as u64,
        _ => return Err(bad(cmd, format!("`{key}` = {v} must be a whole number ({lo}..={hi})"))),
    };
    Ok(Some(u32::try_from(n).unwrap_or(hi)))
}

fn count_set_group(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "count.setGroup";
    let d = doc(s)?;
    let g = group_index(&d, p).ok_or_else(|| bad(CMD, "no such count group"))?;
    let mut grp = d.measurement.count_groups[g].clone();
    if let Some(n) = p.get("name").and_then(Value::as_str) {
        grp.name = n.to_string();
    }
    if let Some(c) = color_of(p, "color") {
        grp.color = c;
    }
    if let Some(v) = whole_size(CMD, p, "markerSize", 1, 10)? {
        grp.marker_size = v;
    }
    if let Some(v) = whole_size(CMD, p, "labelSize", 8, 72)? {
        grp.label_size = v;
    }
    if let Some(v) = p.get("visible").and_then(Value::as_bool) {
        grp.visible = v;
    }
    let activate = p.get("active").and_then(Value::as_bool) == Some(true) || p.as_object().is_some_and(|o| o.len() == 1 && o.contains_key("group"));
    let changed = grp != d.measurement.count_groups[g];
    if changed || (activate && d.measurement.active_count_group != g) {
        s.edit("Count Group Options", |d, _| {
            d.measurement.count_groups[g] = grp;
            if activate {
                d.measurement.active_count_group = g;
            }
            Ok(())
        })?;
    }
    Ok(count_json(doc(s)?.as_ref()))
}

// ------------------------------------------------------------------ record + log

fn base_row(s: &mut Session, d: &Document, source: &str, label: &str, now: &str) -> Map<String, Value> {
    let sc = &d.measurement.scale;
    let mut m = Map::new();
    m.insert("label".into(), json!(label));
    m.insert("dateTime".into(), json!(now));
    m.insert("document".into(), json!(d.name));
    m.insert(
        "source".into(),
        json!(match source {
            "selection" => "Selection",
            "ruler" => "Ruler Tool",
            _ => "Count Tool",
        }),
    );
    m.insert("scale".into(), json!(sc.describe()));
    m.insert("scaleUnits".into(), json!(sc.units));
    m.insert("scaleFactor".into(), json!(sc.factor()));
    let _ = s;
    m
}

fn feature_values(m: &mut Map<String, Value>, f: &Feature, count: usize, factor: f64) {
    m.insert("count".into(), json!(count));
    m.insert("area".into(), json!(round4(f.area * factor * factor)));
    m.insert("perimeter".into(), json!(round4(f.perimeter * factor)));
    m.insert("circularity".into(), json!(round4(f.circularity())));
    m.insert("height".into(), json!(round4(f64::from(f.bounds.height()) * factor)));
    m.insert("width".into(), json!(round4(f64::from(f.bounds.width()) * factor)));
    m.insert("grayMin".into(), json!(round4(f.gray_min)));
    m.insert("grayMax".into(), json!(round4(f.gray_max)));
    m.insert("grayMean".into(), json!(round4(f.gray_mean)));
    m.insert("grayMedian".into(), json!(round4(f.gray_median)));
    m.insert("integratedDensity".into(), json!(round4(f.area * f.gray_mean)));
    m.insert("histogram".into(), json!(f.histogram));
}

fn record(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.analysis.recordMeasurements";
    let d = doc(s)?;
    let source = match p.get("source").and_then(Value::as_str).unwrap_or("auto") {
        "auto" => {
            if d.selection.is_some() {
                "selection"
            } else if d.measurement.ruler.is_some() {
                "ruler"
            } else if d.measurement.count_total() > 0 {
                "count"
            } else {
                "selection"
            }
        }
        x @ ("selection" | "ruler" | "count") => x,
        other => return Err(bad(CMD, format!("unknown source `{other}` (auto|selection|ruler|count)"))),
    };
    // Numbered only once the rows exist, so a failed measurement does not use up a label.
    let label = format!("Measurement {}", s.analysis.next_measurement + 1);
    let now = p.get("dateTime").and_then(Value::as_str).map(str::to_string).unwrap_or_else(now_iso);
    let factor = d.measurement.scale.factor();
    let mut rows: Vec<Map<String, Value>> = Vec::new();
    match source {
        "selection" => {
            let (sum, feats) = selection_rows(&d)?;
            let mut m = base_row(s, &d, source, &label, &now);
            feature_values(&mut m, &sum, feats.len(), factor);
            rows.push(m);
            if feats.len() > 1 {
                for (i, f) in feats.iter().enumerate() {
                    let mut m = base_row(s, &d, source, &format!("Feature {}", i + 1), &now);
                    feature_values(&mut m, f, 1, factor);
                    rows.push(m);
                }
            }
        }
        "ruler" => {
            let r = d.measurement.ruler.ok_or_else(|| EngineError::Other("no ruler line to measure".into()))?;
            let info = ruler_info(&r, &d.measurement.scale);
            let mut m = base_row(s, &d, source, &label, &now);
            m.insert("length".into(), info["length"].clone());
            m.insert("angle".into(), info["angle"].clone());
            rows.push(m);
        }
        _ => {
            let mut m = base_row(s, &d, source, &label, &now);
            m.insert("count".into(), json!(d.measurement.count_total()));
            rows.push(m);
        }
    }
    s.analysis.next_measurement += 1;
    let keep = s.analysis.data_points.of(source).to_vec();
    let mut ids = Vec::new();
    for mut m in rows {
        m.retain(|k, _| keep.iter().any(|x| x == k));
        s.analysis.next_row += 1;
        ids.push(s.analysis.next_row);
        s.analysis.log.push(LogRow { id: s.analysis.next_row, values: m });
    }
    let new: Vec<&LogRow> = s.analysis.log.iter().filter(|r| ids.contains(&r.id)).collect();
    Ok(json!({"source": source, "rows": new}))
}

fn log_list(s: &mut Session, _p: &Value) -> Result<Value> {
    Ok(json!({"rows": s.analysis.log, "columns": log_columns(&s.analysis.log)}))
}

/// Columns any row has, in Photoshop's order.
pub fn log_columns(rows: &[LogRow]) -> Vec<&'static str> {
    COLUMNS.iter().map(|c| c.0).filter(|k| rows.iter().any(|r| r.values.contains_key(*k))).collect()
}

fn ids_param(p: &Value) -> Option<Vec<u64>> {
    p.get("rows").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_u64).collect())
}

fn log_delete(s: &mut Session, p: &Value) -> Result<Value> {
    let before = s.analysis.log.len();
    match ids_param(p) {
        Some(ids) => s.analysis.log.retain(|r| !ids.contains(&r.id)),
        None if p.get("all").and_then(Value::as_bool) == Some(true) => s.analysis.log.clear(),
        None => return Err(bad("measurementLog.delete", "pass `rows`: [ids] or `all`: true")),
    }
    Ok(json!({"deleted": before - s.analysis.log.len(), "remaining": s.analysis.log.len()}))
}

fn csv_field(v: &Value) -> String {
    let s = match v {
        Value::String(s) => s.clone(),
        Value::Array(a) => a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    if s.contains([',', '"', '\n', '\r']) { format!("\"{}\"", s.replace('"', "\"\"")) } else { s }
}

/// The log (or the given rows) as CSV with Photoshop's column headers.
pub fn log_csv(rows: &[LogRow]) -> String {
    let cols = log_columns(rows);
    let mut out = String::new();
    out.push_str(&cols.iter().map(|k| COLUMNS.iter().find(|c| c.0 == *k).map_or(*k, |c| c.1)).collect::<Vec<_>>().join(","));
    out.push('\n');
    for r in rows {
        out.push_str(&cols.iter().map(|k| r.values.get(*k).map(csv_field).unwrap_or_default()).collect::<Vec<_>>().join(","));
        out.push('\n');
    }
    out
}

fn log_export(s: &mut Session, p: &Value) -> Result<Value> {
    let rows: Vec<LogRow> = match ids_param(p) {
        Some(ids) => s.analysis.log.iter().filter(|r| ids.contains(&r.id)).cloned().collect(),
        None => s.analysis.log.clone(),
    };
    let csv = log_csv(&rows);
    match p.get("path").and_then(Value::as_str) {
        Some(path) => {
            crate::file_cmds::write_file(path, csv.as_bytes())?;
            Ok(json!({"path": path, "rows": rows.len()}))
        }
        None => Ok(json!({"csv": csv, "rows": rows.len()})),
    }
}

// ------------------------------------------------------------------ scale marker

/// A "nice" bar length (1, 2 or 5 × 10ⁿ) near a fifth of the image width, in logical units.
fn nice_length(d: &Document) -> f64 {
    let target = f64::from(d.size.width) / 5.0 * d.measurement.scale.factor();
    let e = 10f64.powf(target.log10().floor());
    [5.0, 2.0, 1.0].into_iter().map(|m| m * e).find(|v| *v <= target).unwrap_or(e)
}

/// The shortest image a scale bar fits: four times its 2 px minimum height (#932).
const MIN_SCALE_MARKER_HEIGHT: f64 = 8.0;

fn place_scale_marker(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "image.analysis.placeScaleMarker";
    let d = doc(s)?;
    let sc = d.measurement.scale.clone();
    let length = p.get("length").and_then(Value::as_f64).unwrap_or_else(|| nice_length(&d));
    if !(length > 0.0 && length.is_finite()) {
        return Err(bad(CMD, "length must be positive"));
    }
    let px = length / sc.factor();
    let (w, h) = (f64::from(d.size.width), f64::from(d.size.height));
    if px < 1.0 || px > w {
        return Err(bad(CMD, format!("a {length} {} bar is {px:.1} px; it must fit the {w} px wide image", sc.units)));
    }
    // The bar is at least 2 px tall and at most a quarter of the image, so a shorter image has no
    // room for it (and would invert the height clamp below, which panics) (#932).
    if h < MIN_SCALE_MARKER_HEIGHT {
        return Err(bad(CMD, format!("the image is {h} px tall; a scale marker needs at least {MIN_SCALE_MARKER_HEIGHT} px")));
    }
    let font_size = p.get("fontSize").and_then(Value::as_f64).unwrap_or(12.0).clamp(1.0, 1000.0);
    let font = p.get("font").and_then(Value::as_str).map(str::to_string);
    let show_text = p.get("displayText").and_then(Value::as_bool).unwrap_or(true);
    let text_top = match p.get("textPosition").and_then(Value::as_str).unwrap_or("bottom") {
        "top" => true,
        "bottom" => false,
        o => return Err(bad(CMD, format!("textPosition `{o}` (top|bottom)"))),
    };
    let rgb: [f32; 3] = match p.get("color").and_then(Value::as_str).unwrap_or("black") {
        "black" => [0.0, 0.0, 0.0],
        "white" => [1.0, 1.0, 1.0],
        o => return Err(bad(CMD, format!("color `{o}` (black|white)"))),
    };
    // Bottom-left with a margin; the text sits under (or over) the bar.
    let margin = (w.min(h) * 0.03).round().max(2.0);
    let bar_h = (font_size / 4.0).round().clamp(2.0, h / 4.0);
    let text_h = if show_text { font_size * 1.25 } else { 0.0 };
    // Keep the bar on the canvas when the text under it doesn't fit; the label is clipped instead (#932).
    let bar_y = if text_top { h - margin - bar_h } else { h - margin - bar_h - text_h }.max(0.0);
    // Likewise pull a bar nearly as wide as the image in from the right edge (#932).
    let bar_x = margin.min(w - px);
    let bar = Rect::new(bar_x.round() as i32, bar_y.round() as i32, (bar_x + px).round() as i32, (bar_y + bar_h).round() as i32);
    let label = format!("{} {}", trim_num(length), sc.units);
    compound(s, "Place Scale Marker", |s| {
        let fmt = doc(s)?.pixel_format();
        let bar_id = s.edit("Place Scale Marker", |doc, active| {
            let mut l = Layer::raster("Scale Bar", fmt);
            if let Some(surf) = l.surface_mut() {
                surf.fill_rect(bar, &photocraft_raster::from_rgba(&fmt, [rgb[0], rgb[1], rgb[2], 1.0]));
            }
            let id = l.id;
            doc.layers.push(Layer::group("Measurement Scale Marker", vec![l]));
            *active = Some(id);
            Ok(id)
        })?;
        let mut text_layer = Value::Null;
        if show_text {
            let baseline = if text_top { f64::from(bar.y0) - font_size * 0.3 } else { f64::from(bar.y1) + font_size };
            let mut tp = json!({"x": bar.x0, "y": baseline, "text": label, "size": font_size, "color": [rgb[0], rgb[1], rgb[2], 1.0], "name": label});
            if let Some(f) = &font {
                tp["font"] = json!(f);
            }
            text_layer = s.execute("type.create", tp)?["layer"].clone();
        }
        let group = doc(s)?.layers.last().map(|l| l.id.0);
        Ok(json!({"group": group, "bar": bar_id.0, "text": text_layer, "barRect": [bar.x0, bar.y0, bar.x1, bar.y1], "pixels": px, "label": label}))
    })
}

fn trim_num(v: f64) -> String {
    if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") }
}

// ------------------------------------------------------------------ info

fn analysis_info(s: &mut Session, _p: &Value) -> Result<Value> {
    let d = doc(s)?;
    let m = &d.measurement;
    Ok(json!({
        "scale": scale_json(&m.scale),
        "ruler": m.ruler.map(|r| json!({"start": r.start, "end": r.end, "protractor": r.protractor, "info": ruler_info(&r, &m.scale)})),
        "count": count_json(&d),
        "notes": d.notes.len(),
        "logRows": s.analysis.log.len(),
    }))
}

macro_rules! spec {
    ($id:literal, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr, $journal:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: $journal }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            "image.analysis.setMeasurementScale",
            "Set Measurement Scale…",
            ["Image", "Analysis"],
            r##"{"preset":"default|custom"="custom","pixelLength":px,"logicalLength":number,"units":str (e.g. "mm")} (no params: read the scale)"##,
            has_doc,
            set_scale,
            true
        ),
        spec!(
            "image.analysis.selectDataPoints",
            "Select Data Points…",
            ["Image", "Analysis"],
            r##"{"selection":[keys]|{key:bool}?,"ruler":[keys]|{key:bool}?,"count":[keys]|{key:bool}?,"reset":bool=false} (keys: label,dateTime,document,source,scale,scaleUnits,scaleFactor,count,area,perimeter,circularity,height,width,grayMin,grayMax,grayMean,grayMedian,integratedDensity,histogram,length,angle)"##,
            always,
            select_data_points,
            true
        ),
        spec!(
            "image.analysis.recordMeasurements",
            "Record Measurements",
            ["Image", "Analysis"],
            r##"{"source":"auto|selection|ruler|count"="auto"} → appended Measurement Log rows (selection: summary + one row per feature)"##,
            has_doc,
            record,
            true
        ),
        spec!(
            "image.analysis.rulerTool",
            "Ruler Tool",
            ["Image", "Analysis"],
            r##"{"start":[x,y],"end":[x,y],"protractor":[x,y]|null?,"clear":bool=false} (no params: read) → X/Y/W/H/angle/L1/L2"##,
            has_doc,
            ruler_tool,
            true
        ),
        spec!(
            "image.analysis.countTool",
            "Count Tool",
            ["Image", "Analysis"],
            r##"{} → count groups and markers (edit with count.*)"##,
            has_doc,
            count_tool,
            false
        ),
        spec!(
            "image.analysis.placeScaleMarker",
            "Place Scale Marker…",
            ["Image", "Analysis"],
            r##"{"length":logical units=nice ≈ width/5,"font":str?,"fontSize":pt=12,"displayText":bool=true,"textPosition":"top|bottom"="bottom","color":"black|white"="black"}"##,
            has_doc,
            place_scale_marker,
            true
        ),
        spec!(
            "image.analysis.straightenLayer",
            "Straighten Layer",
            [],
            r##"{"crop":bool=(active layer is the Background)} (rotates so the ruler line is level; crop = rotate the canvas and crop to the image)"##,
            has_ruler,
            straighten,
            true
        ),
        spec!("image.analysis.info", "Analysis Info", [], r##"{} → scale, ruler readout, count groups, note and log counts"##, has_doc, analysis_info, false),
        spec!("count.add", "Add Count", [], r##"{"x":px,"y":px,"group":index?}"##, has_doc, count_add, true),
        spec!("count.remove", "Remove Count", [], r##"{"index":n,"group":index?} | {"x":px,"y":px,"radius":px=6}"##, has_doc, count_remove, true),
        spec!("count.move", "Move Count", [], r##"{"index":n,"group":index?,"to":[x,y]} | {"x":px,"y":px,"to":[x,y]}"##, has_doc, count_move, true),
        spec!("count.clear", "Clear Count", [], r##"{"group":index|"all"=active}"##, has_doc, count_clear, true),
        spec!("count.newGroup", "New Count Group", [], r##"{"name":str?,"color":"#rrggbb"|[r,g,b]?}"##, has_doc, count_new_group, true),
        spec!("count.deleteGroup", "Delete Count Group", [], r##"{"group":index=active}"##, has_doc, count_delete_group, true),
        spec!(
            "count.setGroup",
            "Count Group Options",
            [],
            r##"{"group":index=active,"name":str?,"color":"#rrggbb"|[r,g,b]?,"markerSize":1..10?,"labelSize":8..72?,"visible":bool?,"active":bool?}"##,
            has_doc,
            count_set_group,
            true
        ),
        spec!("measurementLog.list", "Measurement Log", [], r##"{} → rows and columns"##, always, log_list, false),
        spec!("measurementLog.delete", "Delete Measurements", [], r##"{"rows":[ids]}|{"all":true}"##, has_log, log_delete, true),
        spec!(
            "measurementLog.export",
            "Export Measurements…",
            [],
            r##"{"path":str? (CSV file; omitted → returns the CSV text),"rows":[ids]?}"##,
            always,
            log_export,
            true
        ),
    ]
}

#[cfg(test)]
mod tests;
