//! `cargo xtask perf`: run the release benches, merge their `--json` reports into one result keyed
//! by scenario id, and check it against `perf/budgets.toml` and `perf/baseline.json` (#221).
//!
//! ```text
//! cargo xtask perf [--quick] [--update-baseline] [--threshold PCT] [--bench NAME]... [--skip-build] [--reuse]
//! ```
//!
//! - `--quick`: only the benches with `quick_args` (small synthetic documents; minutes, not tens
//!   of minutes). Quick numbers are compared only with a quick baseline, and budgets don't apply
//!   (they are set for the full-size documents).
//! - `--update-baseline`: write this run into `perf/baseline.json` (the only way it changes).
//! - `--threshold PCT`: the regression threshold (default `settings.regression_pct`, 15 %).
//! - `--bench NAME`: run only these benches (repeatable).
//! - `--skip-build`: don't run `cargo build` first. `--reuse`: don't run the benches either;
//!   re-evaluate the reports already in `target/perf/`.
//!
//! Output: `target/perf/results.json` (everything), `target/perf/summary.md` (the table the
//! nightly job posts as its summary) and one `target/perf/<bench>.json` per bench.
//!
//! Exit status: non-zero when an enforced budget breaks (full mode), a scenario's p50 regresses
//! more than the threshold against a baseline from the same machine class and mode, or a bench
//! fails. Runs from different machine classes are never compared.
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use serde::Deserialize;
use serde_json::{Value, json};

pub const BUDGETS: &str = "perf/budgets.toml";
pub const BASELINE: &str = "perf/baseline.json";
const MISSING_PREFIX: &str = "**missing**: ";

// ---- budgets.toml ----------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Budgets {
    #[serde(default)]
    pub settings: Settings,
    #[serde(default)]
    pub bench: Vec<BenchSpec>,
    #[serde(default)]
    pub scenario: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Settings {
    /// A scenario regresses when its p50 grows by more than this share of the baseline…
    pub regression_pct: f64,
    /// …and by more than this many milliseconds (timer noise on sub-millisecond rows).
    pub regression_min_ms: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Self { regression_pct: 15.0, regression_min_ms: 1.0 }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSpec {
    pub name: String,
    pub package: String,
    /// Example name; defaults to `name`.
    pub example: Option<String>,
    /// Arguments of a full run (`--json <file>` is appended).
    #[serde(default)]
    pub args: Vec<String>,
    /// Arguments of a `--quick` run; absent = the bench is skipped in quick mode.
    pub quick_args: Option<Vec<String>>,
    /// Source file that must exist for the bench to run (a bench that lives on another branch):
    /// skipped, not failed, while it is missing.
    pub requires: Option<String>,
    #[serde(default)]
    pub note: String,
}

impl BenchSpec {
    pub fn example(&self) -> &str {
        self.example.as_deref().unwrap_or(&self.name)
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Metric {
    P50,
    #[default]
    P95,
    Max,
}

impl Metric {
    fn label(self) -> &'static str {
        match self {
            Metric::P50 => "p50",
            Metric::P95 => "p95",
            Metric::Max => "max",
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Scenario {
    pub id: String,
    pub title: String,
    pub issue: Option<u32>,
    /// Bench and row name that measure it; both absent when `not_measurable` is set.
    pub bench: Option<String>,
    pub row: Option<String>,
    #[serde(default)]
    pub metric: Metric,
    pub budget_ms: Option<f64>,
    /// Free-form target when it isn't a plain time budget.
    pub target: Option<String>,
    /// A broken budget fails the run. Off for targets the code doesn't meet yet: they are
    /// reported ("over budget") but only regressions fail.
    #[serde(default = "yes")]
    pub enforce: bool,
    /// Why this can't be measured yet (the code doesn't support it).
    pub not_measurable: Option<String>,
}

fn yes() -> bool {
    true
}

impl Scenario {
    pub fn target_text(&self) -> String {
        if let Some(t) = &self.target {
            return t.clone();
        }
        match self.budget_ms {
            Some(b) if b.fract() == 0.0 => format!("≤ {b:.0} ms {}", self.metric.label()),
            Some(b) => format!("≤ {b} ms {}", self.metric.label()),
            None => "track only".into(),
        }
    }
}

pub fn parse_budgets(text: &str) -> Result<Budgets, String> {
    let b: Budgets = toml::from_str(text).map_err(|e| format!("{BUDGETS}: {e}"))?;
    let mut ids = std::collections::BTreeSet::new();
    for s in &b.scenario {
        if !ids.insert(s.id.as_str()) {
            return Err(format!("{BUDGETS}: duplicate scenario id {}", s.id));
        }
        match (&s.bench, &s.row, &s.not_measurable) {
            (Some(bench), Some(_), None) => {
                if !b.bench.iter().any(|x| &x.name == bench) {
                    return Err(format!("{BUDGETS}: scenario {} uses undefined bench `{bench}`", s.id));
                }
            }
            (None, None, Some(_)) => {}
            _ => return Err(format!("{BUDGETS}: scenario {} needs either bench + row, or not_measurable", s.id)),
        }
        if s.budget_ms.is_some_and(|v| !(v.is_finite() && v > 0.0)) {
            return Err(format!("{BUDGETS}: scenario {} has an invalid budget_ms", s.id));
        }
    }
    if !(b.settings.regression_pct.is_finite() && b.settings.regression_pct > 0.0) {
        return Err(format!("{BUDGETS}: settings.regression_pct must be > 0"));
    }
    Ok(b)
}

// ---- measurements ----------------------------------------------------------------------------

/// One scenario's numbers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    pub n: usize,
    pub p50: Option<f64>,
    pub p95: Option<f64>,
    pub max: Option<f64>,
    pub peak_rss: Option<u64>,
    pub gpu_bytes: Option<u64>,
}

impl Stats {
    pub fn metric(&self, m: Metric) -> Option<f64> {
        match m {
            Metric::P50 => self.p50,
            Metric::P95 => self.p95,
            Metric::Max => self.max,
        }
    }

    pub fn to_json(&self) -> Value {
        json!({"n": self.n, "p50_ms": self.p50, "p95_ms": self.p95, "max_ms": self.max, "peak_rss_bytes": self.peak_rss, "gpu_bytes": self.gpu_bytes})
    }

    pub fn from_json(v: &Value) -> Self {
        Self {
            n: v.get("n").and_then(Value::as_u64).unwrap_or(0) as usize,
            p50: v.get("p50_ms").and_then(Value::as_f64),
            p95: v.get("p95_ms").and_then(Value::as_f64),
            max: v.get("max_ms").and_then(Value::as_f64),
            peak_rss: v.get("peak_rss_bytes").and_then(Value::as_u64),
            gpu_bytes: v.get("gpu_bytes").and_then(Value::as_u64),
        }
    }
}

/// Nearest-rank percentile of sorted, non-empty `v`.
fn rank(v: &[f64], q: f64) -> Option<f64> {
    let n = v.len();
    if n == 0 {
        return None;
    }
    let k = ((q * n as f64).ceil() as usize).clamp(1, n);
    v.get(k - 1).copied()
}

/// A bench row → stats. Rows carry `samples_ms` (preferred: percentiles are recomputed here),
/// or `p50_ms`/`p95_ms`/`max_ms`, or the older `median_ms`/`max_ms`. A p95 is never invented:
/// with neither samples nor a reported p95 it stays empty.
pub fn row_stats(row: &Value) -> Stats {
    let mut samples: Vec<f64> =
        row.get("samples_ms").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_f64).filter(|x| x.is_finite()).collect()).unwrap_or_default();
    samples.sort_by(f64::total_cmp);
    let f = |k: &str| row.get(k).and_then(Value::as_f64);
    let mut s = Stats {
        n: samples.len(),
        p50: f("p50_ms").or_else(|| f("median_ms")),
        p95: f("p95_ms"),
        max: f("max_ms"),
        peak_rss: row.get("peak_rss_bytes").and_then(Value::as_u64),
        gpu_bytes: row.get("gpu_bytes").and_then(Value::as_u64),
    };
    if !samples.is_empty() {
        s.p50 = rank(&samples, 0.5);
        s.p95 = rank(&samples, 0.95);
        s.max = samples.last().copied();
    } else if let Some(n) = row.get("n").and_then(Value::as_u64) {
        s.n = n as usize;
    }
    s
}

/// The row named `name` in a bench report (`{"rows": [{"name": …}, …]}`).
pub fn find_row<'a>(report: &'a Value, name: &str) -> Option<&'a Value> {
    report.get("rows")?.as_array()?.iter().find(|r| r.get("name").and_then(Value::as_str) == Some(name))
}

/// `cur` regressed against `base` by more than `pct` % and more than `min_ms`.
pub fn regressed(cur: f64, base: f64, pct: f64, min_ms: f64) -> bool {
    base > 0.0 && cur.is_finite() && cur > base * (1.0 + pct / 100.0) && cur - base > min_ms
}

pub fn change_pct(cur: f64, base: f64) -> Option<f64> {
    (base > 0.0 && cur.is_finite()).then(|| (cur / base - 1.0) * 100.0)
}

#[derive(Clone, Debug, PartialEq)]
pub enum Status {
    Pass,
    /// Over budget, not enforced (target not met yet).
    Over,
    /// Over an enforced budget.
    Broken,
    Regressed,
    /// Measured, no budget, no regression.
    Tracked,
    /// The bench ran but the row is missing or errored.
    Missing(String),
    NotRun,
    NotMeasurable(String),
}

impl Status {
    pub fn label(&self) -> String {
        match self {
            Status::Pass => "pass".into(),
            Status::Over => "over budget (target not met yet)".into(),
            Status::Broken => "**BUDGET BROKEN**".into(),
            Status::Regressed => "**REGRESSED**".into(),
            Status::Tracked => "tracked".into(),
            Status::Missing(e) => format!("{MISSING_PREFIX}{e}"),
            Status::NotRun => "not run".into(),
            Status::NotMeasurable(_) => "not measurable yet".into(),
        }
    }
    pub fn fails(&self) -> bool {
        matches!(self, Status::Broken | Status::Regressed | Status::Missing(_))
    }
}

/// One evaluated scenario.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub id: String,
    pub title: String,
    pub issue: Option<u32>,
    pub target: String,
    pub stats: Option<Stats>,
    pub baseline_p50: Option<f64>,
    pub change_pct: Option<f64>,
    pub status: Status,
}

/// Budget and regression verdict for one scenario. `budgets_apply` is false in quick mode.
pub fn verdict(sc: &Scenario, stats: &Stats, base: Option<&Stats>, settings: &Settings, threshold: f64, budgets_apply: bool) -> Status {
    if let (Some(cur), Some(b)) = (stats.p50, base.and_then(|b| b.p50))
        && regressed(cur, b, threshold, settings.regression_min_ms)
    {
        return Status::Regressed;
    }
    match (budgets_apply, sc.budget_ms, stats.metric(sc.metric)) {
        (true, Some(budget), Some(v)) if v > budget => {
            if sc.enforce {
                Status::Broken
            } else {
                Status::Over
            }
        }
        (true, Some(_), Some(_)) => Status::Pass,
        (true, Some(_), None) => Status::Missing(format!("no {} in the bench row", sc.metric.label())),
        _ => Status::Tracked,
    }
}

/// Evaluates every scenario against the bench reports (`bench name → report`, `None` when the
/// bench didn't run in this mode) and the baseline's scenarios for this mode.
pub fn evaluate(
    budgets: &Budgets,
    reports: &BTreeMap<String, Result<Value, String>>,
    baseline: Option<&BTreeMap<String, Stats>>,
    threshold: f64,
    budgets_apply: bool,
) -> Vec<Outcome> {
    budgets
        .scenario
        .iter()
        .map(|sc| {
            let base = baseline.and_then(|b| b.get(&sc.id));
            let mut o = Outcome {
                id: sc.id.clone(),
                title: sc.title.clone(),
                issue: sc.issue,
                target: sc.target_text(),
                stats: None,
                baseline_p50: base.and_then(|b| b.p50),
                change_pct: None,
                status: Status::NotRun,
            };
            if let Some(why) = &sc.not_measurable {
                o.status = Status::NotMeasurable(why.clone());
                return o;
            }
            let (Some(bench), Some(row)) = (&sc.bench, &sc.row) else { return o };
            match reports.get(bench) {
                None => {}
                Some(Err(e)) => o.status = Status::Missing(format!("bench `{bench}` failed: {e}")),
                Some(Ok(rep)) => match find_row(rep, row) {
                    None => {
                        let err = rep
                            .get("context")
                            .and_then(|c| c.get("errors"))
                            .and_then(Value::as_array)
                            .and_then(|a| a.iter().find(|e| e.get("name").and_then(Value::as_str) == Some(row.as_str())))
                            .and_then(|e| e.get("error"))
                            .and_then(Value::as_str)
                            .map_or_else(|| format!("row `{row}` not in `{bench}`"), str::to_string);
                        o.status = Status::Missing(err);
                    }
                    Some(r) => {
                        let st = row_stats(r);
                        o.change_pct = st.p50.zip(o.baseline_p50).and_then(|(c, b)| change_pct(c, b));
                        o.status = verdict(sc, &st, base, &budgets.settings, threshold, budgets_apply);
                        o.stats = Some(st);
                    }
                },
            }
            o
        })
        .collect()
}

// ---- formatting ------------------------------------------------------------------------------

pub fn fmt_num(v: f64) -> String {
    if v >= 100.0 {
        format!("{v:.0}")
    } else if v >= 10.0 {
        format!("{v:.1}")
    } else {
        format!("{v:.2}")
    }
}

fn fmt_ms(v: Option<f64>) -> String {
    v.map_or_else(|| "–".into(), fmt_num)
}

pub fn fmt_bytes(v: Option<u64>) -> String {
    match v {
        None | Some(0) => "–".into(),
        Some(b) if b >= 1 << 30 => format!("{:.1} GB", b as f64 / (1u64 << 30) as f64),
        Some(b) => format!("{} MB", b >> 20),
    }
}

pub fn issue_link(n: Option<u32>) -> String {
    n.map_or_else(String::new, |n| format!("[#{n}](https://github.com/storytold/photocraft/issues/{n})"))
}

/// The scenario table (Markdown); `with_baseline` adds the change against the baseline.
/// Not-measurable rows go to a separate list after it.
pub fn render_table(outcomes: &[Outcome], with_baseline: bool) -> String {
    let mut s = if with_baseline {
        String::from(
            "| Id | Scenario | Target | p50 ms | p95 ms | max ms | Peak RSS | GPU | vs baseline | Status | Issue |\n|---|---|---|---:|---:|---:|---:|---:|---:|---|---|\n",
        )
    } else {
        String::from(
            "| Id | Scenario | Target | p50 ms | p95 ms | max ms | Peak RSS | GPU | Status | Issue |\n|---|---|---|---:|---:|---:|---:|---:|---|---|\n",
        )
    };
    for o in outcomes.iter().filter(|o| !matches!(o.status, Status::NotMeasurable(_))) {
        let st = o.stats.clone().unwrap_or_default();
        let vs = if with_baseline { format!(" {} |", o.change_pct.map_or_else(|| "–".into(), |c| format!("{c:+.0} %"))) } else { String::new() };
        s.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |{} {} | {} |\n",
            o.id,
            o.title,
            o.target,
            fmt_ms(st.p50),
            fmt_ms(st.p95),
            fmt_ms(st.max),
            fmt_bytes(st.peak_rss),
            fmt_bytes(st.gpu_bytes),
            vs,
            o.status.label(),
            issue_link(o.issue),
        ));
    }
    let nm: Vec<&Outcome> = outcomes.iter().filter(|o| matches!(o.status, Status::NotMeasurable(_))).collect();
    if !nm.is_empty() {
        s.push_str(
            "\n**Not measurable yet** (the code can't do it yet; no number is recorded):\n\n| Id | Scenario | Target | Why | Issue |\n|---|---|---|---|---|\n",
        );
        for o in nm {
            let why = match &o.status {
                Status::NotMeasurable(w) => w.as_str(),
                _ => "",
            };
            s.push_str(&format!("| {} | {} | {} | {} | {} |\n", o.id, o.title, o.target, why, issue_link(o.issue)));
        }
    }
    s
}

/// Counts for the scorecard summary: (within budget, over budget or failing, tracked only, not measurable).
/// Not-run scenarios are in none of them; the scorecard counts those separately.
pub fn counts(outcomes: &[Outcome]) -> (usize, usize, usize, usize) {
    let mut c = (0, 0, 0, 0);
    for o in outcomes {
        match o.status {
            Status::Pass => c.0 += 1,
            Status::Over | Status::Broken | Status::Regressed | Status::Missing(_) => c.1 += 1,
            Status::Tracked => c.2 += 1,
            Status::NotMeasurable(_) => c.3 += 1,
            Status::NotRun => {}
        }
    }
    c
}

// ---- machine ---------------------------------------------------------------------------------

/// Lower-case, `-`-separated, alphanumeric only.
pub fn slug(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
    }
    out.trim_end_matches('-').to_string()
}

/// Runs are compared only within one class: OS, architecture, CPU model and GPU adapter.
pub fn machine_class(os: &str, arch: &str, cpu: &str, gpu: &str) -> String {
    [os, arch, cpu, gpu].iter().map(|p| slug(p)).filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-")
}

/// The first three numbers in `/proc/loadavg` or `sysctl -n vm.loadavg` output.
pub fn parse_loadavg(s: &str) -> Option<[f64; 3]> {
    let v: Vec<f64> = s.split_whitespace().filter_map(|t| t.trim_matches(|c| c == '{' || c == '}').parse().ok()).take(3).collect();
    Some([*v.first()?, *v.get(1)?, *v.get(2)?])
}

fn output(cmd: &str, args: &[&str]) -> Option<String> {
    let o = Command::new(cmd).args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string()).filter(|s| !s.is_empty())
}

pub fn load_avg() -> Option<[f64; 3]> {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        return parse_loadavg(&s);
    }
    output("sysctl", &["-n", "vm.loadavg"]).and_then(|s| parse_loadavg(&s))
}

fn proc_field(file: &str, key: &str) -> Option<String> {
    let s = std::fs::read_to_string(file).ok()?;
    s.lines().find(|l| l.starts_with(key)).and_then(|l| l.split_once(':').or_else(|| l.split_once('='))).map(|(_, v)| v.trim().trim_matches('"').to_string())
}

pub fn machine() -> Value {
    let (os, arch) = (std::env::consts::OS, std::env::consts::ARCH);
    let cores = std::thread::available_parallelism().map(|n| n.get()).ok();
    let (cpu, ram, os_version) = match os {
        "macos" => (
            output("sysctl", &["-n", "machdep.cpu.brand_string"]),
            output("sysctl", &["-n", "hw.memsize"]).and_then(|s| s.parse::<u64>().ok()),
            output("sw_vers", &["-productVersion"]).map(|v| format!("macOS {v}")),
        ),
        "linux" => (
            proc_field("/proc/cpuinfo", "model name").or_else(|| proc_field("/proc/cpuinfo", "CPU part").map(|p| format!("ARM part {p}"))),
            proc_field("/proc/meminfo", "MemTotal")
                .and_then(|v| v.split_whitespace().next().and_then(|k| k.parse::<u64>().ok()))
                .and_then(|k| k.checked_mul(1024)),
            proc_field("/etc/os-release", "PRETTY_NAME"),
        ),
        _ => (std::env::var("PROCESSOR_IDENTIFIER").ok(), None, output("cmd", &["/c", "ver"])),
    };
    json!({"os": os, "os_version": os_version, "arch": arch, "cpu": cpu, "cores": cores, "ram_bytes": ram})
}

/// `YYYY-MM-DDTHH:MM:SSZ` for a Unix time (civil-from-days, proleptic Gregorian).
pub fn iso_utc(secs: u64) -> String {
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, rem % 3600 / 60, rem % 60)
}

fn now_iso() -> String {
    iso_utc(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0))
}

fn git(root: &Path, args: &[&str]) -> Option<String> {
    let o = Command::new("git").current_dir(root).args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).trim().to_string())
}

// ---- baseline --------------------------------------------------------------------------------

/// The baseline's scenarios for `mode`, if it was recorded on `class`. `Err` explains a skip.
pub fn baseline_for(baseline: &Value, class: &str, mode: &str) -> Result<BTreeMap<String, Stats>, String> {
    let bc = baseline.get("machine_class").and_then(Value::as_str).unwrap_or("");
    if bc != class {
        return Err(format!("baseline is from machine class `{bc}`, this is `{class}`: regression check skipped"));
    }
    let sc = baseline
        .get("modes")
        .and_then(|m| m.get(mode))
        .and_then(|m| m.get("scenarios"))
        .and_then(Value::as_object)
        .ok_or_else(|| format!("baseline has no `{mode}` run: regression check skipped"))?;
    Ok(sc.iter().map(|(k, v)| (k.clone(), Stats::from_json(v))).collect())
}

/// `old` with this run's mode replaced (or a fresh baseline when the machine class changed).
pub fn updated_baseline(old: Option<&Value>, results: &Value) -> Value {
    let class = results.get("machine").and_then(|m| m.get("class")).cloned().unwrap_or(Value::Null);
    let mode = results.get("mode").and_then(Value::as_str).unwrap_or("full").to_string();
    let mut modes = old.filter(|o| o.get("machine_class") == Some(&class)).and_then(|o| o.get("modes")).and_then(Value::as_object).cloned().unwrap_or_default();
    let scenarios: serde_json::Map<String, Value> = results
        .get("scenarios")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter(|(_, v)| v.get("p50_ms").is_some_and(|x| !x.is_null())).map(|(k, v)| (k.clone(), Stats::from_json(v).to_json())).collect())
        .unwrap_or_default();
    // Scenarios whose bench row failed (a crash the bench found): kept so the scorecard says so.
    let failed: serde_json::Map<String, Value> = results
        .get("scenarios")
        .and_then(Value::as_object)
        .map(|m| m.iter().filter_map(|(k, v)| v.get("status")?.as_str()?.strip_prefix(MISSING_PREFIX).map(|e| (k.clone(), json!(e)))).collect())
        .unwrap_or_default();
    modes.insert(
        mode,
        json!({
            "date": results.get("date"),
            "commit": results.get("commit"),
            "load_avg_start": results.get("load_avg_start"),
            "load_avg_end": results.get("load_avg_end"),
            "scenarios": scenarios,
            "failed": failed,
        }),
    );
    json!({"schema": 1, "machine_class": class, "machine": results.get("machine"), "modes": modes})
}

// ---- running ---------------------------------------------------------------------------------

struct Opts {
    quick: bool,
    update_baseline: bool,
    threshold: Option<f64>,
    benches: Vec<String>,
    skip_build: bool,
    reuse: bool,
}

fn parse_opts(rest: &[&str]) -> Result<Opts, String> {
    let mut o = Opts { quick: false, update_baseline: false, threshold: None, benches: Vec::new(), skip_build: false, reuse: false };
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match *a {
            "--quick" => o.quick = true,
            "--update-baseline" => o.update_baseline = true,
            "--skip-build" => o.skip_build = true,
            "--reuse" => {
                o.reuse = true;
                o.skip_build = true;
            }
            "--threshold" => {
                let v = it.next().and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0).ok_or("--threshold needs a percentage > 0")?;
                o.threshold = Some(v);
            }
            "--bench" => o.benches.push(it.next().ok_or("--bench needs a name")?.to_string()),
            other => return Err(format!("perf: unknown option `{other}`")),
        }
    }
    Ok(o)
}

fn target_dir() -> Result<PathBuf, String> {
    let m = crate::metadata()?;
    m.get("target_directory").and_then(Value::as_str).map(PathBuf::from).ok_or_else(|| "cargo metadata: no target_directory".into())
}

fn write(path: &Path, text: &str) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

pub fn run(root: &Path, rest: &[&str]) -> Result<(), String> {
    let o = parse_opts(rest)?;
    let text = std::fs::read_to_string(root.join(BUDGETS)).map_err(|e| format!("read {BUDGETS}: {e}"))?;
    let budgets = parse_budgets(&text)?;
    let mode = if o.quick { "quick" } else { "full" };
    let threshold = o.threshold.unwrap_or(budgets.settings.regression_pct);
    let target = target_dir()?;
    let out_dir = target.join("perf");

    // Which benches run in this mode.
    let mut plan: Vec<(&BenchSpec, Vec<String>)> = Vec::new();
    let mut skipped: Vec<(String, String)> = Vec::new();
    for b in &budgets.bench {
        if !o.benches.is_empty() && !o.benches.contains(&b.name) {
            continue;
        }
        let args = if o.quick {
            match &b.quick_args {
                Some(a) => a.clone(),
                None => continue,
            }
        } else {
            b.args.clone()
        };
        if let Some(req) = &b.requires
            && !root.join(req).exists()
        {
            skipped.push((b.name.clone(), format!("{req} not on this branch yet")));
            continue;
        }
        plan.push((b, args));
    }

    if !o.skip_build {
        let mut by_pkg: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (b, _) in &plan {
            by_pkg.entry(b.package.as_str()).or_default().push(b.example());
        }
        for (pkg, examples) in by_pkg {
            let mut c = crate::cargo();
            c.args(["build", "--release", "-p", pkg]);
            for e in &examples {
                c.args(["--example", e]);
            }
            crate::run(c, &format!("cargo build --release -p {pkg} --example {}", examples.join(" --example ")))?;
        }
    }

    let load_start = load_avg();
    let mut reports: BTreeMap<String, Result<Value, String>> = BTreeMap::new();
    let mut bench_meta = serde_json::Map::new();
    for (b, args) in &plan {
        let json_path = out_dir.join(format!("{}.json", b.name));
        let before = load_avg();
        let t = Instant::now();
        let result = if o.reuse {
            std::fs::read_to_string(&json_path).map_err(|e| format!("--reuse: {}: {e}", json_path.display()))
        } else {
            let _ = std::fs::remove_file(&json_path);
            if let Some(dir) = json_path.parent() {
                std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
            }
            let exe = target.join("release").join("examples").join(format!("{}{}", b.example(), std::env::consts::EXE_SUFFIX));
            let mut c = Command::new(&exe);
            c.current_dir(root).args(args).arg("--json").arg(&json_path);
            let shown = format!("{} {} --json {}", b.example(), args.join(" "), json_path.display());
            crate::run(c, &shown).and_then(|()| std::fs::read_to_string(&json_path).map_err(|e| format!("{}: no JSON report ({e})", b.name)))
        };
        let parsed = result.and_then(|t| serde_json::from_str::<Value>(&t).map_err(|e| format!("{}: bad JSON: {e}", b.name)));
        let after = load_avg();
        bench_meta.insert(
            b.name.clone(),
            json!({
                "status": if parsed.is_ok() { "ok" } else { "error" },
                "note": b.note,
                "error": parsed.as_ref().err(),
                "seconds": t.elapsed().as_secs_f64(),
                "args": args,
                "load_avg_before": before,
                "load_avg_after": after,
                "context": parsed.as_ref().ok().and_then(|r| r.get("context")).cloned(),
                "process_peak_rss_bytes": parsed.as_ref().ok().and_then(|r| r.get("process_peak_rss_bytes")).cloned(),
                // Every row, mapped to a scenario or not (new benches show up here first).
                "rows": parsed.as_ref().ok().and_then(|r| r.get("rows")).and_then(Value::as_array).map(|rows| {
                    rows.iter()
                        .map(|r| {
                            let mut v = row_stats(r).to_json();
                            if let Some(o) = v.as_object_mut() {
                                o.insert("name".into(), r.get("name").cloned().unwrap_or(Value::Null));
                            }
                            v
                        })
                        .collect::<Vec<_>>()
                }),
            }),
        );
        reports.insert(b.name.clone(), parsed);
    }
    for (name, why) in &skipped {
        bench_meta.insert(name.clone(), json!({"status": "skipped", "error": why}));
    }
    let load_end = load_avg();

    let mut m = machine();
    let gpu = reports
        .values()
        .filter_map(|r| r.as_ref().ok())
        .find_map(|r| r.get("context").and_then(|c| c.get("gpu_adapter")).and_then(Value::as_str).map(str::to_string));
    let gpu_name = gpu.clone().unwrap_or_else(|| "cpu-canvas".into());
    let class = machine_class(
        m.get("os").and_then(Value::as_str).unwrap_or(""),
        m.get("arch").and_then(Value::as_str).unwrap_or(""),
        m.get("cpu").and_then(Value::as_str).unwrap_or(""),
        &gpu_name,
    );
    if let Some(obj) = m.as_object_mut() {
        obj.insert("gpu_adapter".into(), json!(gpu));
        obj.insert("class".into(), json!(class));
    }

    let baseline_value: Option<Value> = std::fs::read_to_string(root.join(BASELINE)).ok().and_then(|t| serde_json::from_str(&t).ok());
    let (baseline, baseline_note) = match baseline_value.as_ref().map(|b| baseline_for(b, &class, mode)) {
        Some(Ok(b)) => (Some(b), None),
        Some(Err(e)) => (None, Some(e)),
        None => (None, Some(format!("no {BASELINE}: regression check skipped"))),
    };
    let outcomes = evaluate(&budgets, &reports, baseline.as_ref(), threshold, !o.quick);

    let scenarios: serde_json::Map<String, Value> = budgets
        .scenario
        .iter()
        .zip(&outcomes)
        .map(|(sc, oc)| {
            let mut v = oc.stats.as_ref().map(Stats::to_json).unwrap_or_else(|| json!({}));
            if let Some(obj) = v.as_object_mut() {
                obj.insert("title".into(), json!(sc.title));
                obj.insert("issue".into(), json!(sc.issue));
                obj.insert("bench".into(), json!(sc.bench));
                obj.insert("row".into(), json!(sc.row));
                obj.insert("metric".into(), json!(sc.metric.label()));
                obj.insert("budget_ms".into(), json!(sc.budget_ms));
                obj.insert("target".into(), json!(sc.target_text()));
                obj.insert("enforce".into(), json!(sc.enforce));
                obj.insert("baseline_p50_ms".into(), json!(oc.baseline_p50));
                obj.insert("change_pct".into(), json!(oc.change_pct));
                obj.insert("status".into(), json!(oc.status.label()));
                if let Status::NotMeasurable(w) = &oc.status {
                    obj.insert("not_measurable".into(), json!(w));
                }
            }
            (sc.id.clone(), v)
        })
        .collect();
    let failures: Vec<String> = outcomes.iter().filter(|o| o.status.fails()).map(|o| format!("{} {}: {}", o.id, o.title, o.status.label())).collect();
    let bench_failures: Vec<String> = reports.iter().filter_map(|(n, r)| r.as_ref().err().map(|e| format!("bench {n}: {e}"))).collect();
    let commit = git(root, &["rev-parse", "HEAD"]);
    let dirty = git(root, &["status", "--porcelain"]).is_some_and(|s| !s.is_empty());
    let results = json!({
        "schema": 1,
        "mode": mode,
        "date": now_iso(),
        "commit": commit,
        "dirty": dirty,
        "machine": m,
        "load_avg_start": load_start,
        "load_avg_end": load_end,
        "threshold_pct": threshold,
        "baseline_note": baseline_note,
        "benches": bench_meta,
        "scenarios": scenarios,
        "failures": failures.iter().chain(&bench_failures).collect::<Vec<_>>(),
    });
    let text = serde_json::to_string_pretty(&results).map_err(|e| format!("encode results: {e}"))?;
    write(&out_dir.join("results.json"), &text)?;

    let summary = render_summary(&results, &outcomes, baseline_note.as_deref());
    write(&out_dir.join("summary.md"), &summary)?;
    println!("\n{summary}");
    println!("wrote {}", out_dir.join("results.json").display());

    if o.update_baseline {
        let nb = updated_baseline(baseline_value.as_ref(), &results);
        let text = serde_json::to_string_pretty(&nb).map_err(|e| format!("encode baseline: {e}"))?;
        write(&root.join(BASELINE), &format!("{text}\n"))?;
        println!("updated {BASELINE} ({mode}, machine class {class})");
    }
    let all: Vec<&String> = failures.iter().chain(&bench_failures).collect();
    if all.is_empty() { Ok(()) } else { Err(format!("{} perf failure(s):\n  {}", all.len(), all.iter().map(|s| s.as_str()).collect::<Vec<_>>().join("\n  "))) }
}

/// The committed baseline's full-mode numbers judged against the budgets (what the scorecard
/// shows: deterministic, no run needed). Scenarios without a baseline number are "not run".
pub fn outcomes_from_baseline(budgets: &Budgets, baseline: Option<&Value>) -> Vec<Outcome> {
    let mode = baseline.and_then(|b| b.get("modes")?.get("full"));
    let full: Option<BTreeMap<String, Stats>> =
        mode.and_then(|m| m.get("scenarios")?.as_object().cloned()).map(|m| m.iter().map(|(k, v)| (k.clone(), Stats::from_json(v))).collect());
    let failed = |id: &str| mode.and_then(|m| m.get("failed")?.get(id)?.as_str()).map(str::to_string);
    budgets
        .scenario
        .iter()
        .map(|sc| {
            let stats = full.as_ref().and_then(|f| f.get(&sc.id)).cloned();
            let status = match (&sc.not_measurable, &stats) {
                (Some(w), _) => Status::NotMeasurable(w.clone()),
                (None, Some(st)) => verdict(sc, st, None, &budgets.settings, budgets.settings.regression_pct, true),
                (None, None) => failed(&sc.id).map_or(Status::NotRun, Status::Missing),
            };
            Outcome {
                id: sc.id.clone(),
                title: sc.title.clone(),
                issue: sc.issue,
                target: sc.target_text(),
                stats,
                baseline_p50: None,
                change_pct: None,
                status,
            }
        })
        .collect()
}

pub fn fmt_load(v: Option<&Value>) -> String {
    v.and_then(Value::as_array).map_or_else(|| "n/a".into(), |a| a.iter().filter_map(Value::as_f64).map(|x| format!("{x:.2}")).collect::<Vec<_>>().join(" / "))
}

/// The Markdown summary of one run: context lines and the scenario table.
pub fn render_summary(results: &Value, outcomes: &[Outcome], baseline_note: Option<&str>) -> String {
    let m = results.get("machine").cloned().unwrap_or(Value::Null);
    let s = |v: Option<&Value>| v.and_then(Value::as_str).unwrap_or("unknown").to_string();
    let ram = m.get("ram_bytes").and_then(Value::as_u64).map_or_else(|| "?".into(), |b| format!("{} GB", b >> 30));
    let mut out = format!(
        "## Performance ({} run)\n\n- Machine: {} ({} cores, {ram}), GPU {}, {} {}; class `{}`\n- Commit {}{}, {}\n- Load average (1/5/15 min): {} at start, {} at end\n",
        s(results.get("mode")),
        s(m.get("cpu")),
        m.get("cores").and_then(Value::as_u64).unwrap_or(0),
        s(m.get("gpu_adapter")),
        s(m.get("os_version")),
        s(m.get("arch")),
        s(m.get("class")),
        s(results.get("commit")),
        if results.get("dirty").and_then(Value::as_bool) == Some(true) { " (dirty)" } else { "" },
        s(results.get("date")),
        fmt_load(results.get("load_avg_start")),
        fmt_load(results.get("load_avg_end")),
    );
    if let Some(n) = baseline_note {
        out.push_str(&format!("- {n}\n"));
    }
    if results.get("mode").and_then(Value::as_str) == Some("quick") {
        out.push_str("- Quick mode: small documents, budgets not applied (they are set for full-size documents).\n");
    }
    if let Some(b) = results.get("benches").and_then(Value::as_object) {
        for (name, v) in b {
            let st = v.get("status").and_then(Value::as_str).unwrap_or("?");
            let secs = v.get("seconds").and_then(Value::as_f64).map_or_else(String::new, |x| format!(", {x:.0} s"));
            let err = v.get("error").and_then(Value::as_str).map_or_else(String::new, |e| format!(": {e}"));
            out.push_str(&format!("- Bench `{name}`: {st}{secs}, load {}{err}\n", fmt_load(v.get("load_avg_before"))));
        }
    }
    out.push('\n');
    out.push_str(&render_table(outcomes, true));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOML: &str = r#"
[settings]
regression_pct = 15.0
regression_min_ms = 0.5

[[bench]]
name = "b"
package = "p"
quick_args = ["--quick"]

[[scenario]]
id = "P1"
title = "move"
issue = 209
bench = "b"
row = "move row"
budget_ms = 25

[[scenario]]
id = "P2"
title = "opacity"
bench = "b"
row = "opacity row"
budget_ms = 20
enforce = false

[[scenario]]
id = "P3"
title = "zoom"
target = "≤ 16 ms per frame"
not_measurable = "needs a frame harness"
"#;

    fn report(rows: Value) -> Value {
        json!({"bench": "b", "rows": rows})
    }

    #[test]
    fn budgets_parse_and_validate() {
        let b = parse_budgets(TOML).unwrap();
        assert_eq!(b.scenario.len(), 3);
        assert_eq!(b.scenario[0].metric, Metric::P95);
        assert!(b.scenario[0].enforce);
        assert!(!b.scenario[1].enforce);
        assert_eq!(b.scenario[0].target_text(), "≤ 25 ms p95");
        assert_eq!(b.scenario[2].target_text(), "≤ 16 ms per frame");
        assert_eq!(b.settings.regression_min_ms, 0.5);
        // Duplicate ids, undefined benches, half-specified rows and bad budgets are rejected.
        assert!(parse_budgets(&format!("{TOML}\n[[scenario]]\nid = \"P1\"\ntitle = \"x\"\nnot_measurable = \"y\"\n")).is_err());
        assert!(parse_budgets("[[scenario]]\nid = \"P1\"\ntitle = \"x\"\nbench = \"nope\"\nrow = \"r\"\n").is_err());
        assert!(parse_budgets("[[bench]]\nname = \"b\"\npackage = \"p\"\n[[scenario]]\nid = \"P1\"\ntitle = \"x\"\nbench = \"b\"\n").is_err());
        assert!(
            parse_budgets("[[bench]]\nname = \"b\"\npackage = \"p\"\n[[scenario]]\nid = \"P1\"\ntitle = \"x\"\nbench = \"b\"\nrow = \"r\"\nbudget_ms = -1\n")
                .is_err()
        );
        assert!(parse_budgets("[settings]\nregression_pct = 0\n").is_err());
        assert!(parse_budgets("bogus = 1\n").is_err());
        assert!(parse_budgets("not toml [").is_err());
    }

    #[test]
    fn row_stats_prefers_samples_and_never_invents_p95() {
        let s = row_stats(&json!({"name": "x", "samples_ms": [5.0, 1.0, 3.0, 2.0, 4.0], "p50_ms": 99.0}));
        assert_eq!((s.n, s.p50, s.p95, s.max), (5, Some(3.0), Some(5.0), Some(5.0)));
        // Older rows: median + max only. p95 stays empty.
        let s = row_stats(&json!({"name": "x", "median_ms": 4.0, "min_ms": 1.0, "max_ms": 9.0}));
        assert_eq!((s.p50, s.p95, s.max), (Some(4.0), None, Some(9.0)));
        let s = row_stats(&json!({"name": "x", "samples_ms": ["bad", null]}));
        assert_eq!(s.p50, None);
    }

    #[test]
    fn regression_detection() {
        assert!(!regressed(11.4, 10.0, 15.0, 0.5));
        assert!(regressed(11.6, 10.0, 15.0, 0.5));
        // Below the absolute noise floor: no regression.
        assert!(!regressed(0.30, 0.20, 15.0, 0.5));
        assert!(!regressed(5.0, 0.0, 15.0, 0.5));
        assert!(!regressed(f64::NAN, 1.0, 15.0, 0.0));
        assert_eq!(change_pct(12.0, 10.0).map(|v| v.round()), Some(20.0));
        assert_eq!(change_pct(1.0, 0.0), None);
    }

    #[test]
    fn evaluate_budgets_regressions_and_missing_rows() {
        let b = parse_budgets(TOML).unwrap();
        let mut reports = BTreeMap::new();
        reports.insert(
            "b".to_string(),
            Ok(report(json!([{"name": "move row", "samples_ms": [10.0, 12.0, 30.0]}, {"name": "opacity row", "samples_ms": [40.0]}]))),
        );
        let out = evaluate(&b, &reports, None, 15.0, true);
        assert_eq!(out[0].status, Status::Broken, "p95 30 > 25 and enforced");
        assert_eq!(out[1].status, Status::Over, "not enforced: reported, doesn't fail");
        assert!(matches!(out[2].status, Status::NotMeasurable(_)));
        assert!(out[0].status.fails() && !out[1].status.fails() && !out[2].status.fails());

        // Quick mode: no budgets, only regressions.
        let mut base = BTreeMap::new();
        base.insert("P1".to_string(), Stats { p50: Some(10.0), ..Default::default() });
        base.insert("P2".to_string(), Stats { p50: Some(30.0), ..Default::default() });
        let out = evaluate(&b, &reports, Some(&base), 15.0, false);
        assert_eq!(out[0].status, Status::Regressed, "p50 12 vs 10 is +20 %");
        assert_eq!(out[1].status, Status::Regressed, "p50 40 vs 30");
        let out = evaluate(&b, &reports, Some(&base), 50.0, false);
        assert_eq!(out[0].status, Status::Tracked, "threshold is configurable");

        // A row the bench didn't produce fails, with the bench's own error when it has one.
        let mut reports = BTreeMap::new();
        reports.insert("b".to_string(), Ok(json!({"rows": [], "context": {"errors": [{"name": "move row", "error": "boom"}]}})));
        let out = evaluate(&b, &reports, None, 15.0, true);
        assert_eq!(out[0].status, Status::Missing("boom".into()));
        assert!(matches!(&out[1].status, Status::Missing(e) if e.contains("not in")));
        // A bench that didn't run in this mode: not run, not a failure.
        let out = evaluate(&b, &BTreeMap::new(), None, 15.0, true);
        assert_eq!(out[0].status, Status::NotRun);
        assert!(!out[0].status.fails());
        assert_eq!(counts(&out), (0, 0, 0, 1), "not-run scenarios are not tracked; the scorecard counts them separately");
    }

    #[test]
    fn baselines_never_cross_machine_classes() {
        let results = json!({"mode": "full", "date": "d", "commit": "c", "machine": {"class": "macos-aarch64-m4"}, "scenarios": {"P1": {"p50_ms": 3.0, "p95_ms": 4.0, "max_ms": 5.0, "n": 3}, "P3": {"status": "not measurable yet"}}});
        let b = updated_baseline(None, &results);
        let got = baseline_for(&b, "macos-aarch64-m4", "full").unwrap();
        assert_eq!(got.get("P1").and_then(|s| s.p50), Some(3.0));
        assert!(!got.contains_key("P3"), "rows without numbers are not baselined");
        assert!(baseline_for(&b, "linux-x86-64-epyc", "full").is_err());
        assert!(baseline_for(&b, "macos-aarch64-m4", "quick").is_err());
        // Same class: the other mode is kept. New class: replaced.
        let quick = json!({"mode": "quick", "machine": {"class": "macos-aarch64-m4"}, "scenarios": {"P1": {"p50_ms": 1.0}}});
        let b2 = updated_baseline(Some(&b), &quick);
        assert!(baseline_for(&b2, "macos-aarch64-m4", "full").is_ok());
        assert!(baseline_for(&b2, "macos-aarch64-m4", "quick").is_ok());
        // A crashed row is kept as a failure, and the scorecard shows it.
        let crashed = json!({"mode": "full", "machine": {"class": "c"}, "scenarios": {"P1": {"status": "**missing**: panicked: boom"}}});
        let bc = updated_baseline(None, &crashed);
        let b = parse_budgets(TOML).unwrap();
        let o = outcomes_from_baseline(&b, Some(&bc));
        assert_eq!(o[0].status, Status::Missing("panicked: boom".into()));
        assert_eq!(o[1].status, Status::NotRun);
        let other = json!({"mode": "quick", "machine": {"class": "other"}, "scenarios": {}});
        let b3 = updated_baseline(Some(&b2), &other);
        assert!(baseline_for(&b3, "other", "full").is_err());
    }

    #[test]
    fn machine_helpers() {
        assert_eq!(slug("Apple M4 Pro (Metal)"), "apple-m4-pro-metal");
        assert_eq!(machine_class("macos", "aarch64", "Apple M4 Pro", "Apple M4 Pro"), "macos-aarch64-apple-m4-pro-apple-m4-pro");
        assert_eq!(parse_loadavg("{ 2.10 1.50 0.75 }"), Some([2.1, 1.5, 0.75]));
        assert_eq!(parse_loadavg("0.52 0.58 0.59 1/467 12345\n"), Some([0.52, 0.58, 0.59]));
        assert_eq!(parse_loadavg("garbage"), None);
        assert_eq!(iso_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_utc(1_791_158_400), "2026-10-05T00:00:00Z");
        assert_eq!(iso_utc(951_782_400), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn table_lists_not_measurable_rows_separately() {
        let b = parse_budgets(TOML).unwrap();
        let mut reports = BTreeMap::new();
        reports.insert("b".to_string(), Ok(report(json!([{"name": "move row", "samples_ms": [10.0], "peak_rss_bytes": 3u64 << 30}]))));
        let t = render_table(&evaluate(&b, &reports, None, 15.0, true), true);
        assert!(t.contains("| P1 | move | ≤ 25 ms p95 | 10.0 | 10.0 | 10.0 | 3.0 GB |"));
        assert!(t.contains("**Not measurable yet**"));
        assert!(t.contains("| P3 | zoom | ≤ 16 ms per frame | needs a frame harness |  |"));
        assert_eq!(counts(&evaluate(&b, &reports, None, 15.0, true)), (1, 1, 0, 1), "the row the bench lacks counts as failing");
    }

    #[test]
    fn options() {
        assert!(parse_opts(&["--bogus"]).is_err());
        assert!(parse_opts(&["--threshold", "x"]).is_err());
        assert!(parse_opts(&["--threshold", "-3"]).is_err());
        let o = parse_opts(&["--quick", "--threshold", "20", "--bench", "a", "--reuse"]).unwrap();
        assert!(o.quick && o.reuse && o.skip_build);
        assert_eq!(o.threshold, Some(20.0));
        assert_eq!(o.benches, ["a"]);
    }
}
