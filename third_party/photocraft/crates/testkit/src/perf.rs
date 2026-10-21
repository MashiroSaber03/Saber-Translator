//! Benchmark helpers shared by the `--json` benches that `cargo xtask perf` runs (#221).
//!
//! - [`RssSampler`]: peak resident memory measured in-process. A background thread polls the
//!   process's resident set size every few milliseconds; [`RssSampler::reset`] starts a new
//!   window, so each scenario gets its own peak. Polling can miss a spike shorter than the
//!   interval; on Linux [`process_peak_rss_bytes`] also reads the kernel's exact high-water mark.
//! - [`Summary`]: p50 / p95 / max of a scenario's samples (nearest-rank percentiles).
//! - [`row`] / [`report`]: the JSON shape `xtask perf` reads.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use serde_json::{Value, json};

/// Current resident set size of this process, in bytes.
pub fn current_rss_bytes() -> Option<u64> {
    memory_stats::memory_stats().map(|m| m.physical_mem as u64)
}

/// The process's peak resident set size so far: the kernel's high-water mark on Linux
/// (`VmHWM`), `None` elsewhere (use an [`RssSampler`]).
pub fn process_peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_vm_hwm(&status)
}

fn parse_vm_hwm(status: &str) -> Option<u64> {
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kb.checked_mul(1024)
}

/// Polls the resident set size on a background thread and keeps the maximum.
pub struct RssSampler {
    peak: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl RssSampler {
    /// Starts sampling every `interval` (2–5 ms is cheap: one syscall per sample).
    pub fn start(interval: Duration) -> Self {
        let peak = Arc::new(AtomicU64::new(current_rss_bytes().unwrap_or(0)));
        let stop = Arc::new(AtomicBool::new(false));
        let (p, s) = (Arc::clone(&peak), Arc::clone(&stop));
        let thread = std::thread::Builder::new()
            .name("rss-sampler".into())
            .spawn(move || {
                while !s.load(Ordering::Relaxed) {
                    if let Some(v) = current_rss_bytes() {
                        p.fetch_max(v, Ordering::Relaxed);
                    }
                    std::thread::sleep(interval);
                }
            })
            .ok();
        Self { peak, stop, thread }
    }

    /// Starts a new window: the peak becomes the current resident size.
    pub fn reset(&self) {
        self.peak.store(current_rss_bytes().unwrap_or(0), Ordering::Relaxed);
    }

    /// The peak resident size since the last [`reset`](Self::reset) (or start), including now.
    pub fn peak(&self) -> Option<u64> {
        if let Some(v) = current_rss_bytes() {
            self.peak.fetch_max(v, Ordering::Relaxed);
        }
        Some(self.peak.load(Ordering::Relaxed)).filter(|v| *v > 0)
    }
}

impl Drop for RssSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(t) = self.thread.take() {
            // A sampler thread that panicked has nothing to report; don't cascade.
            let _ = t.join();
        }
    }
}

/// p50 / p95 / max of a set of timings (ms).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Summary {
    pub n: usize,
    pub p50: f64,
    pub p95: f64,
    pub max: f64,
}

/// Nearest-rank percentile `q` (0..=1) of sorted, non-empty `v`.
fn rank(v: &[f64], q: f64) -> Option<f64> {
    let n = v.len();
    let k = ((q * n as f64).ceil() as usize).clamp(1, n.max(1));
    v.get(k - 1).copied()
}

impl Summary {
    /// `None` when there are no finite samples.
    pub fn of(samples: &[f64]) -> Option<Self> {
        let mut v: Vec<f64> = samples.iter().copied().filter(|x| x.is_finite()).collect();
        v.sort_by(f64::total_cmp);
        Some(Self { n: v.len(), p50: rank(&v, 0.5)?, p95: rank(&v, 0.95)?, max: *v.last()? })
    }
}

/// One scenario row: `name`, sample count, p50/p95/max, the raw samples, peak RSS and GPU bytes.
pub fn row(name: &str, samples: &[f64], peak_rss: Option<u64>, gpu_bytes: Option<u64>) -> Value {
    let s = Summary::of(samples);
    json!({
        "name": name,
        "n": s.map_or(0, |s| s.n),
        "p50_ms": s.map(|s| s.p50),
        "p95_ms": s.map(|s| s.p95),
        "max_ms": s.map(|s| s.max),
        "samples_ms": samples,
        "peak_rss_bytes": peak_rss,
        "gpu_bytes": gpu_bytes,
    })
}

/// A bench's whole report: its name, free-form context (document size, adapter…), the rows and
/// the process's peak RSS (the kernel's high-water mark where there is one, else the highest
/// sampled peak).
pub fn report(bench: &str, context: Value, rows: Vec<Value>, sampler: Option<&RssSampler>) -> Value {
    let sampled = rows.iter().filter_map(|r| r["peak_rss_bytes"].as_u64()).chain(sampler.and_then(RssSampler::peak)).max();
    let process_peak = process_peak_rss_bytes().or(sampled);
    json!({
        "bench": bench,
        "schema": 1,
        "context": context,
        "rows": rows,
        "process_peak_rss_bytes": process_peak,
    })
}

/// Writes `report` to `path` (pretty JSON). Errors are returned, not panicked.
pub fn write_report(path: &str, report: &Value) -> Result<(), String> {
    let text = serde_json::to_string_pretty(report).map_err(|e| format!("encode {path}: {e}"))?;
    std::fs::write(path, text).map_err(|e| format!("write {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summary_nearest_rank() {
        let v: Vec<f64> = (1..=20).map(f64::from).collect();
        let s = Summary::of(&v).unwrap();
        assert_eq!((s.n, s.p50, s.p95, s.max), (20, 10.0, 19.0, 20.0));
        let s = Summary::of(&[3.0]).unwrap();
        assert_eq!((s.p50, s.p95, s.max), (3.0, 3.0, 3.0));
        assert!(Summary::of(&[]).is_none());
        assert!(Summary::of(&[f64::NAN]).is_none());
    }

    #[test]
    fn vm_hwm_parses() {
        assert_eq!(parse_vm_hwm("Name:\tx\nVmHWM:\t  2048 kB\nVmRSS: 1 kB\n"), Some(2048 * 1024));
        assert_eq!(parse_vm_hwm("VmHWM: lots\n"), None);
        assert_eq!(parse_vm_hwm(""), None);
    }

    #[test]
    fn sampler_reports_a_peak() {
        let s = RssSampler::start(Duration::from_millis(1));
        s.reset();
        let big = vec![1u8; 32 << 20];
        std::hint::black_box(&big);
        let p = s.peak();
        drop(big);
        // Platforms without a memory API report nothing rather than a made-up number.
        if current_rss_bytes().is_some() {
            assert!(p.unwrap() > 0);
        }
    }

    #[test]
    fn row_shape() {
        let r = row("x", &[2.0, 1.0, 3.0], Some(10), None);
        assert_eq!(r["p50_ms"], 2.0);
        assert_eq!(r["max_ms"], 3.0);
        assert_eq!(r["n"], 3);
        assert!(r["gpu_bytes"].is_null());
        let empty = row("y", &[], None, None);
        assert!(empty["p95_ms"].is_null());
    }
}
