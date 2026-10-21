//! Stroke latency trace (#1343): where each frame's time goes while a brush stroke is drawn, so
//! a tester on any platform can report a per-stage breakdown instead of one input-to-pixel number.
//!
//! Set `PHOTOCRAFT_STROKE_TRACE` before launching: `1` prints a summary to stderr when each stroke
//! ends; a file path also appends every frame of every stroke to that CSV. The last summary is in
//! the control channel's `perf` output (`strokeTrace`).
//!
//! Per frame of a stroke it records the interval since the previous frame started, the pointer
//! moves the frame delivered, the live stroke's render time, the canvas refresh (composite and
//! upload), the UI build, and eframe's CPU time for the frame before (UI, tessellation and GPU
//! submission, without the vsync wait). What the app can't see (the OS input queue before a frame
//! starts, frames queued for the display, the compositor) is the rest of the input-to-photon time;
//! the summary names the GPU backend because GL ignores the frame-latency setting.

use std::fmt::Write as _;

/// One frame of a stroke. Times in ms.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Row {
    /// Frame start, from [`crate::gpu_canvas::now_ms`].
    pub t: f64,
    /// Since the previous frame started (0 for the first frame the trace saw).
    pub interval: f64,
    /// Pointer moves the frame delivered.
    pub moves: usize,
    /// Rendering the stroke's new points (the live stroke, or the commit on release).
    pub stroke: f64,
    /// Canvas refresh: composite and upload.
    pub canvas: f64,
    /// Building the UI (`App::ui`), the two above included.
    pub ui: f64,
    /// eframe's CPU time for the previous frame: UI, tessellation and GPU submission.
    pub cpu_prev: Option<f64>,
}

/// Percentiles of a stroke's frames, for the summary line and the control channel.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Summary {
    pub frames: usize,
    /// GPU backend and adapter ("vulkan NVIDIA GeForce …").
    pub backend: String,
    /// `[p50, p95]` of each stage, in ms.
    pub interval: [f64; 2],
    pub moves: [f64; 2],
    pub stroke: [f64; 2],
    pub canvas: [f64; 2],
    pub ui: [f64; 2],
    /// eframe's CPU time per frame (UI, tessellation, submission).
    pub cpu: [f64; 2],
    /// Frame interval minus that CPU time: waiting on vsync / the swap chain, or idle.
    pub wait: [f64; 2],
}

impl Summary {
    /// One line for stderr.
    pub fn line(&self) -> String {
        let f = |name: &str, [p50, p95]: [f64; 2]| format!("{name} {p50:.2}/{p95:.2}");
        [
            format!("stroke trace: {} frames on {} (p50/p95 ms):", self.frames, self.backend),
            f("interval", self.interval),
            f("moves/frame", self.moves),
            f("stroke", self.stroke),
            f("canvas", self.canvas),
            f("ui", self.ui),
            f("cpu", self.cpu),
            f("wait", self.wait),
        ]
        .join(" ")
    }
}

/// Where the trace goes ([`StrokeTrace::from_env`]).
// The web build has no clock, so it never traces.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) enum Output {
    #[default]
    Off,
    /// A summary per stroke on stderr.
    Stderr,
    /// The summary on stderr, and every frame appended to this CSV.
    Csv(std::path::PathBuf),
}

#[derive(Debug, Default)]
pub struct StrokeTrace {
    pub(crate) output: Output,
    /// The frame being built (started by `begin_frame`).
    frame: Option<Row>,
    last_start: Option<f64>,
    /// The current stroke's frames.
    rows: Vec<Row>,
    /// Strokes finished so far (numbers the CSV rows).
    strokes: usize,
    /// The last finished stroke's summary.
    pub last: Option<Summary>,
}

/// `[p50, p95]` of `v` (nearest rank; zeros when empty).
fn percentiles(mut v: Vec<f64>) -> [f64; 2] {
    v.retain(|x| x.is_finite());
    if v.is_empty() {
        return [0.0, 0.0];
    }
    v.sort_by(f64::total_cmp);
    let at = |p: f64| v.get(((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1)).copied().unwrap_or(0.0);
    let r = |x: f64| (x * 100.0).round() / 100.0;
    [r(at(0.5)), r(at(0.95))]
}

impl StrokeTrace {
    /// Reads `PHOTOCRAFT_STROKE_TRACE` (off on the web, where there is no clock).
    pub fn from_env() -> Self {
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(v) = std::env::var_os("PHOTOCRAFT_STROKE_TRACE").filter(|v| !v.is_empty()) {
            let output = if v == "1" || v == "stderr" { Output::Stderr } else { Output::Csv(v.into()) };
            return Self { output, ..Default::default() };
        }
        Self::default()
    }

    pub fn enabled(&self) -> bool {
        self.output != Output::Off
    }

    /// A frame starts (the raw input hook) with `moves` pointer moves.
    pub fn begin_frame(&mut self, now: f64, moves: usize) {
        if !self.enabled() {
            return;
        }
        let interval = self.last_start.map_or(0.0, |t| now - t);
        self.last_start = Some(now);
        self.frame = Some(Row { t: now, interval, moves, ..Default::default() });
    }

    /// eframe's CPU time for the previous frame, in seconds (`Frame::info().cpu_usage`).
    pub fn cpu_prev(&mut self, secs: Option<f32>) {
        if let Some(f) = self.frame.as_mut() {
            f.cpu_prev = secs.map(|s| f64::from(s) * 1000.0);
        }
    }

    /// Time spent rendering stroke points this frame.
    pub fn stroke(&mut self, ms: f64) {
        if let Some(f) = self.frame.as_mut() {
            f.stroke += ms;
        }
    }

    /// Time spent refreshing the canvas this frame.
    pub fn canvas(&mut self, ms: f64) {
        if let Some(f) = self.frame.as_mut() {
            f.canvas += ms;
        }
    }

    /// The frame's UI is built (`ui` ms). `stroking`: a stroke is being drawn (a frame that
    /// rendered stroke points counts too, such as the release's commit). A stroke's frames are
    /// kept until the first frame without one, which ends it.
    pub fn end_frame(&mut self, ui: f64, stroking: bool, backend: impl FnOnce() -> String) {
        let Some(mut f) = self.frame.take() else { return };
        if stroking || f.stroke > 0.0 {
            f.ui = ui;
            // The first frame of a stroke doesn't follow a stroke frame: its interval is idle time.
            if self.rows.is_empty() {
                f.interval = 0.0;
            }
            self.rows.push(f);
        } else if !self.rows.is_empty() {
            self.finish(backend());
        }
    }

    fn finish(&mut self, backend: String) {
        let rows = std::mem::take(&mut self.rows);
        self.strokes += 1;
        let col = |g: fn(&Row) -> Option<f64>| percentiles(rows.iter().filter_map(g).collect());
        let summary = Summary {
            frames: rows.len(),
            backend,
            interval: col(|r| (r.interval > 0.0).then_some(r.interval)),
            moves: col(|r| Some(r.moves as f64)),
            stroke: col(|r| Some(r.stroke)),
            canvas: col(|r| Some(r.canvas)),
            ui: col(|r| Some(r.ui)),
            // A frame's cpu_prev belongs to the frame before it, so pair it with that interval.
            cpu: percentiles(rows.iter().skip(1).filter_map(|r| r.cpu_prev).collect()),
            wait: percentiles(rows.windows(2).filter_map(|w| Some((w[1].interval - w[1].cpu_prev?).max(0.0))).collect()),
        };
        if self.enabled() {
            eprintln!("{}", summary.line());
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Output::Csv(path) = &self.output
            && let Err(e) = append_csv(path, self.strokes, &rows)
        {
            eprintln!("stroke trace: can't write {}: {e}", path.display());
        }
        self.last = Some(summary);
    }
}

/// The CSV text for `rows` of stroke `stroke` (with the header when `header`).
// The web build has no clock, so it never traces.
#[cfg_attr(target_arch = "wasm32", allow(dead_code))]
pub(crate) fn csv(stroke: usize, rows: &[Row], header: bool) -> String {
    let mut s = String::new();
    if header {
        s.push_str("stroke,t_ms,interval_ms,moves,stroke_ms,canvas_ms,ui_ms,cpu_prev_ms\n");
    }
    for r in rows {
        let cpu = r.cpu_prev.map_or(String::new(), |c| format!("{c:.3}"));
        let _ = writeln!(s, "{stroke},{:.3},{:.3},{},{:.3},{:.3},{:.3},{cpu}", r.t, r.interval, r.moves, r.stroke, r.canvas, r.ui);
    }
    s
}

#[cfg(not(target_arch = "wasm32"))]
fn append_csv(path: &std::path::Path, stroke: usize, rows: &[Row]) -> std::io::Result<()> {
    use std::io::Write as _;
    let header = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
    let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    f.write_all(csv(stroke, rows, header).as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn on() -> StrokeTrace {
        StrokeTrace { output: Output::Stderr, ..Default::default() }
    }

    fn frame(t: &mut StrokeTrace, now: f64, cpu: f32, stroking: bool) {
        t.begin_frame(now, 3);
        t.cpu_prev(Some(cpu / 1000.0));
        if stroking {
            t.stroke(1.0);
            t.canvas(2.0);
        }
        t.end_frame(4.0, stroking, || "vulkan Test GPU".into());
    }

    #[test]
    fn a_stroke_is_its_frames_from_the_first_stroke_frame_to_the_first_without() {
        let mut t = on();
        frame(&mut t, 0.0, 5.0, false);
        frame(&mut t, 100.0, 5.0, true);
        for k in 1..=10 {
            frame(&mut t, 100.0 + 16.0 * f64::from(k), 6.0, true);
        }
        assert!(t.last.is_none(), "still drawing");
        frame(&mut t, 276.0, 6.0, false);
        let s = t.last.clone().expect("the stroke ended");
        assert_eq!(s.frames, 11);
        assert_eq!(s.interval, [16.0, 16.0], "the idle gap before the stroke isn't a frame interval");
        assert_eq!((s.stroke, s.canvas, s.ui, s.moves), ([1.0, 1.0], [2.0, 2.0], [4.0, 4.0], [3.0, 3.0]));
        assert_eq!((s.cpu, s.wait), ([6.0, 6.0], [10.0, 10.0]));
        assert!(s.line().contains("11 frames on vulkan Test GPU") && s.line().contains("wait 10.00/10.00"), "{}", s.line());
        // Off: nothing is kept.
        let mut off = StrokeTrace::default();
        frame(&mut off, 0.0, 1.0, true);
        frame(&mut off, 16.0, 1.0, false);
        assert!(off.last.is_none() && off.rows.is_empty());
    }

    #[test]
    fn csv_rows_and_odd_numbers() {
        let rows = [Row { t: 1.0, interval: 16.5, moves: 2, stroke: 0.25, canvas: 1.0, ui: 3.0, cpu_prev: None }];
        assert_eq!(csv(1, &rows, true), "stroke,t_ms,interval_ms,moves,stroke_ms,canvas_ms,ui_ms,cpu_prev_ms\n1,1.000,16.500,2,0.250,1.000,3.000,\n");
        assert_eq!(percentiles(vec![]), [0.0, 0.0]);
        assert_eq!(percentiles(vec![f64::NAN, 2.0, f64::INFINITY]), [2.0, 2.0]);
        assert_eq!(percentiles((1..=100).map(f64::from).collect()), [51.0, 95.0]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn the_csv_gets_one_header_and_every_strokes_frames() {
        let dir = std::env::temp_dir().join(format!("photocraft-stroke-trace-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("trace.csv");
        let _ = std::fs::remove_file(&path);
        let mut t = StrokeTrace { output: Output::Csv(path.clone()), ..Default::default() };
        for start in [0.0, 1000.0] {
            for k in 0..3 {
                frame(&mut t, start + 16.0 * f64::from(k), 5.0, true);
            }
            frame(&mut t, start + 48.0, 5.0, false);
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 7, "{text}");
        assert!(lines[0].starts_with("stroke,") && lines[1].starts_with("1,") && lines[6].starts_with("2,"));
    }
}
