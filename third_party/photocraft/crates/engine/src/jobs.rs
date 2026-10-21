//! Background jobs: long commands run on a worker thread, report progress and can be cancelled
//! (#210).
//!
//! A command opts in by doing its heavy work through [`run`] (or [`edit_job`] for a plain document
//! edit). The work closure gets a [`JobCtx`] (progress plus a cancellation flag) and runs against
//! a snapshot: documents are copy-on-write, so cloning one is O(layers). Its result is applied on
//! the UI thread as one undo step, through the same path as [`Session::execute`], so the journal,
//! history, damage and post-command hooks all behave as for a synchronous command.
//!
//! - [`Session::execute`] runs everything inline, as before: tests, the CLI and scripts see no
//!   difference.
//! - [`Session::start`] runs a job-capable command on a worker thread and returns its
//!   [`JobId`]; [`Session::poll_jobs`] (called every frame by the UI) applies finished jobs.
//!   On wasm `start` runs inline (same API, written once).
//! - [`Session::cancel_job`] sets the flag; the algorithms check it per tile, row band or
//!   iteration, and the result is discarded. The document is never touched by a worker.
//! - A panic in the worker becomes an error; the document is unchanged.
//! - While a job runs on a document, commands that would edit that document are disabled with a
//!   message naming the job (queries and document-independent commands still run).

use std::any::Any;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use photocraft_doc::{DocId, Document, LayerId};
use photocraft_raster::Interrupt;
use serde::Serialize;
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

/// Identifies one background job for the session's lifetime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct JobId(pub u64);

/// What a job's work closure sees: report progress, check for cancellation. Cheap to clone; all
/// clones share the same state.
#[derive(Clone, Default)]
pub struct JobCtx {
    shared: Arc<Shared>,
}

#[derive(Default)]
struct Shared {
    cancel: AtomicBool,
    /// `f32` bits of the progress fraction. Non-negative floats order like their bits, so
    /// `fetch_max` keeps progress monotonic even with several reporting threads.
    progress: AtomicU32,
    message: Mutex<String>,
}

impl std::fmt::Debug for JobCtx {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JobCtx").field("progress", &self.fraction()).field("cancelled", &self.cancelled()).finish()
    }
}

impl JobCtx {
    pub fn new() -> Self {
        Self::default()
    }

    /// Report progress in `0.0..=1.0` with a short status (empty keeps the previous one).
    /// Progress never goes backwards: a lower value than already reported is ignored.
    pub fn progress(&self, fraction: f32, message: &str) {
        if fraction.is_finite() {
            self.shared.progress.fetch_max(fraction.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        }
        if !message.is_empty() {
            let mut m = self.shared.message.lock().unwrap_or_else(PoisonError::into_inner);
            if *m != message {
                message.clone_into(&mut m);
            }
        }
    }

    /// Has the job been cancelled? Long loops check this and stop early.
    pub fn cancelled(&self) -> bool {
        self.shared.cancel.load(Ordering::Relaxed)
    }

    /// Ask the job to stop (idempotent).
    pub fn cancel(&self) {
        self.shared.cancel.store(true, Ordering::Relaxed);
    }

    /// `Err(Cancelled)` once cancelled, for `?` between stages.
    pub fn check(&self) -> Result<()> {
        if self.cancelled() { Err(EngineError::Cancelled) } else { Ok(()) }
    }

    /// Progress reported so far (`0.0..=1.0`).
    pub fn fraction(&self) -> f32 {
        f32::from_bits(self.shared.progress.load(Ordering::Relaxed))
    }

    /// The latest status message.
    pub fn message(&self) -> String {
        self.shared.message.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// Run `f` with an algorithm-level [`Interrupt`] whose progress maps onto `lo..hi` of this
    /// job (with `message`) and whose cancellation is this job's.
    pub fn stage<R>(&self, lo: f32, hi: f32, message: &str, f: impl FnOnce(&Interrupt) -> R) -> R {
        self.progress(lo, message);
        let cancel = || self.cancelled();
        let progress = |p: f32| self.progress(lo + (hi - lo) * p, "");
        f(&Interrupt::new(&cancel, &progress))
    }
}

/// Final state of a job.
#[derive(Clone, Debug, PartialEq)]
pub enum JobOutcome {
    /// Applied; the command's result.
    Done(Value),
    /// Failed (including a panic in the worker); the document is unchanged.
    Failed(String),
    /// Cancelled; the result (if any) was discarded and the document is unchanged.
    Cancelled,
}

/// A job that has ended, as reported by [`Session::poll_jobs`].
#[derive(Clone, Debug, PartialEq)]
pub struct JobEvent {
    pub id: JobId,
    pub command: String,
    pub label: String,
    pub document: Option<DocId>,
    pub outcome: JobOutcome,
}

/// A job's state for the UI, the control channel and MCP (`jobs.list`).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobInfo {
    pub id: JobId,
    pub command: String,
    /// Shown in the progress UI ("Gaussian Blur").
    pub label: String,
    /// The document the job edits (and locks), if any.
    pub document: Option<DocId>,
    /// `running`, `done`, `failed` or `cancelled`.
    pub state: &'static str,
    pub progress: f32,
    pub message: String,
    pub elapsed_ms: f64,
    /// The command's result (done) or the error (failed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

type Payload = Box<dyn Any + Send>;
type Apply = Box<dyn FnOnce(&mut Session, Payload) -> Result<Value> + Send>;

/// Wall-clock milliseconds since some fixed point (0 on wasm, where `Instant` is unavailable and
/// jobs run inline anyway).
fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    {
        0.0
    }
}

struct Running {
    id: JobId,
    command: String,
    label: String,
    params: Value,
    journal: bool,
    doc: Option<DocId>,
    /// The document as the job saw it: applying fails if it was edited meanwhile.
    snapshot: Option<Weak<Document>>,
    ctx: JobCtx,
    started_ms: f64,
    coalesce: Option<String>,
    color_restrict: Option<usize>,
    rx: std::sync::mpsc::Receiver<Result<Payload>>,
    apply: Option<Apply>,
    #[cfg(not(target_arch = "wasm32"))]
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Running {
    fn info(&self) -> JobInfo {
        JobInfo {
            id: self.id,
            command: self.command.clone(),
            label: self.label.clone(),
            document: self.doc,
            state: "running",
            progress: self.ctx.fraction(),
            message: self.ctx.message(),
            elapsed_ms: now_ms() - self.started_ms,
            result: None,
            error: None,
        }
    }
}

/// The session's job table (see the module docs).
#[derive(Default)]
pub struct Jobs {
    next: u64,
    /// Set while [`Session::start`] runs a command: [`run`] spawns instead of running inline.
    spawn: bool,
    /// The job a command registered while `spawn` was set (picked up by `start`).
    pending: Option<Running>,
    running: Vec<Running>,
    /// Ended jobs not yet collected by [`Session::poll_jobs`].
    events: Vec<JobEvent>,
    /// The last few ended jobs, for `jobs.list`.
    recent: Vec<JobInfo>,
    /// Workers of cancelled jobs that may still be unwinding (native).
    #[cfg(not(target_arch = "wasm32"))]
    draining: Vec<std::thread::JoinHandle<()>>,
    /// The context inline job work runs under (see [`Session::set_inline_job_ctx`]).
    inline: Option<JobCtx>,
}

const RECENT: usize = 16;

impl Drop for Jobs {
    fn drop(&mut self) {
        // Workers own snapshots only; stop them so they don't burn CPU for a closed session.
        for r in self.running.iter().chain(self.pending.iter()) {
            r.ctx.cancel();
        }
    }
}

impl std::fmt::Debug for Jobs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jobs").field("running", &self.running.len()).finish()
    }
}

/// How [`Session::start`] ended.
#[derive(Clone, Debug, PartialEq)]
pub enum Started {
    /// The command ran to completion inline (it isn't a job, or this is wasm).
    Done(Value),
    /// The command is running in the background.
    Job(JobId),
}

/// Commands that never edit the active document, allowed while a job runs on it.
fn independent_of_document(id: &str) -> bool {
    const PREFIXES: &[&str] = &["jobs.", "tools.", "view.", "window.", "help.", "brush.presets.", "gradient.presets."];
    const IDS: &[&str] = &["file.new", "file.close", "file.closeAll", "edit.preferences", "edit.colorSettings", "edit.keyboardShortcuts", "edit.menus"];
    PREFIXES.iter().any(|p| id.starts_with(p)) || IDS.contains(&id)
}

#[cfg(not(target_arch = "wasm32"))]
fn read_path(path: &str) -> Result<Vec<u8>> {
    // Bounded reads, and a clear error for a file larger than memory (#375).
    photocraft_format::read_file(std::path::Path::new(path)).map_err(|e| EngineError::Other(format!("{path}: {e}")))
}

#[cfg(target_arch = "wasm32")]
fn read_path(path: &str) -> Result<Vec<u8>> {
    Err(EngineError::Other(format!("cannot open paths on the web: {path}")))
}

/// Run a job-capable command's heavy part. `work` runs on a worker thread when the command was
/// started with [`Session::start`] (inline otherwise, and always on wasm); `apply` then runs on
/// the calling thread with its output and returns the command's result. With `lock_document`,
/// the active document is locked against other edits while the job runs, and applying fails if
/// it was edited anyway.
///
/// `work` must only read what it captured (clone the document `Arc` first): it never sees the
/// session.
pub fn run<T: Send + 'static>(
    s: &mut Session,
    label: &str,
    lock_document: bool,
    work: impl FnOnce(&JobCtx) -> Result<T> + Send + 'static,
    apply: impl FnOnce(&mut Session, T) -> Result<Value> + Send + 'static,
) -> Result<Value> {
    if !s.jobs.spawn || cfg!(target_arch = "wasm32") {
        let t = work(&s.jobs.inline.clone().unwrap_or_default())?;
        return apply(s, t);
    }
    if s.jobs.pending.is_some() {
        return Err(EngineError::Other(format!("`{label}` started a second background job in one command")));
    }
    let (doc, snapshot) = match (lock_document, s.active()) {
        (true, Some(st)) => (Some(st.doc.id), Some(Arc::downgrade(&st.doc))),
        (true, None) => return Err(EngineError::NoDocument),
        (false, _) => (None, None),
    };
    let ctx = JobCtx::new();
    let (tx, rx) = std::sync::mpsc::channel::<Result<Payload>>();
    let erased: Apply = Box::new(move |s, any| {
        let t = any.downcast::<T>().map_err(|_| EngineError::Other("internal error: job result type mismatch".into()))?;
        apply(s, *t)
    });
    s.jobs.next += 1;
    let id = JobId(s.jobs.next);
    #[cfg(not(target_arch = "wasm32"))]
    let handle = {
        let wctx = ctx.clone();
        let label_w = label.to_string();
        // The work closure is moved into the thread. If spawning fails (thread limits), there is
        // no way to get it back, so the error is reported instead.
        let spawned = std::thread::Builder::new().name(format!("photocraft-job-{}", id.0)).spawn(move || {
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| work(&wctx).map(|t| Box::new(t) as Payload)))
                .unwrap_or_else(|_| Err(EngineError::Other(format!("`{label_w}` failed with an internal error (logged); the document is unchanged"))));
            // The receiver is gone when the job was cancelled: nothing to report.
            let _ = tx.send(r);
        });
        Some(spawned.map_err(|e| EngineError::Other(format!("could not start a background job: {e}")))?)
    };
    #[cfg(target_arch = "wasm32")]
    {
        // Unreachable (wasm runs inline above); keep the types used.
        drop((work, tx));
    }
    s.jobs.pending = Some(Running {
        id,
        command: String::new(),
        label: label.to_string(),
        params: Value::Null,
        journal: true,
        doc,
        snapshot,
        ctx,
        started_ms: now_ms(),
        coalesce: s.coalesce_request.clone(),
        color_restrict: s.color_restrict,
        rx,
        apply: Some(erased),
        #[cfg(not(target_arch = "wasm32"))]
        handle,
    });
    Ok(json!({"job": id.0}))
}

/// A document edit whose body is the job: like [`Session::edit`] (one undo step named `label`),
/// but `f` runs on a worker against a copy of the active document when started in the
/// background. `finish` turns `f`'s output into the command's result.
pub fn edit_job<R: Send + 'static>(
    s: &mut Session,
    label: &str,
    f: impl FnOnce(&mut Document, &mut Option<LayerId>, &JobCtx) -> Result<R> + Send + 'static,
    finish: impl FnOnce(R) -> Value + Send + 'static,
) -> Result<Value> {
    if !s.jobs.spawn || cfg!(target_arch = "wasm32") {
        // Inline: exactly `Session::edit`, as before jobs existed.
        let ctx = s.jobs.inline.clone().unwrap_or_default();
        let r = s.edit(label, |doc, active| f(doc, active, &ctx))?;
        return Ok(finish(r));
    }
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let (base, active) = (st.doc.clone(), st.active_layer);
    let label_owned = label.to_string();
    run(
        s,
        label,
        true,
        move |ctx| {
            let mut doc = (*base).clone();
            let mut active = active;
            let r = f(&mut doc, &mut active, ctx)?;
            Ok((doc, active, r))
        },
        move |s, (doc, new_active, r)| {
            s.edit(&label_owned, move |d, active| {
                *d = doc;
                *active = new_active;
                Ok(())
            })?;
            Ok(finish(r))
        },
    )
}

/// Where a background open reads its file from.
#[derive(Clone, Debug)]
pub enum OpenSource {
    /// File contents already in memory (web pickers, drops, automation).
    Bytes(Arc<Vec<u8>>),
    /// A path the worker reads itself (native), so even the read stays off the UI thread.
    Path(String),
}

/// The command id open jobs report (there is no engine command: opening is the shell's).
pub const OPEN_JOB: &str = "file.open";

impl Session {
    /// Run the heavy part of inline (not background) job commands under `ctx`, so another thread
    /// can cancel them: a live preview whose parameters changed stops its stale computation
    /// (`Err(Cancelled)`, the document unchanged) instead of finishing it. `None` restores the
    /// default, a context nothing can cancel.
    pub fn set_inline_job_ctx(&mut self, ctx: Option<JobCtx>) {
        self.jobs.inline = ctx;
    }

    /// Run a command, in the background when it supports it (see [`jobs`](crate::jobs)). Errors
    /// like [`Session::execute`] for unknown, disabled or failing commands.
    pub fn start(&mut self, id: &str, params: Value) -> Result<Started> {
        self.dispatch(id, params, true)
    }

    /// Open a file in the background: read (for a path) and decode on a worker, with progress
    /// (per layer for PSD/PSB) and cancellation, then add the document like
    /// [`Session::open_document`] (Color Settings policies apply). `name` names the document.
    /// The job's result is `{document, name, warnings, color}`; on wasm it runs inline. When
    /// Color Settings ask about the file's profile, the document is held back instead and the
    /// result is `{held, name, warnings, color}` (`color` the question): take it with
    /// [`Session::take_held_open`] once answered.
    pub fn start_open(&mut self, name: &str, source: OpenSource) -> Result<Started> {
        let label = format!("Opening {name}");
        let name_w = name.to_string();
        let name_a = name.to_string();
        let max_svg_group_depth = self.prefs().file_handling.rasterize_svg_groups_deeper_than as usize;
        self.start_job(
            OPEN_JOB,
            json!({"name": name}),
            &label,
            false,
            move |ctx| {
                ctx.progress(0.0, "Reading");
                let bytes = match source {
                    OpenSource::Bytes(b) => b,
                    OpenSource::Path(p) => Arc::new(read_path(&p)?),
                };
                ctx.check()?;
                ctx.progress(0.02, "Decoding");
                ctx.stage(0.02, 1.0, "Decoding", |ctl| photocraft_io::import_with_svg_group_depth_and_interrupt(&name_w, &bytes, max_svg_group_depth, ctl))
                    .map_err(|e| match e {
                        photocraft_io::IoError::Cancelled => EngineError::Cancelled,
                        e => EngineError::Other(e.to_string()),
                    })
            },
            move |s, r: photocraft_io::ImportResult| {
                // Ask When Opening: held back until the prompt is answered (`take_held_open`). A
                // developed raw has no embedded profile to ask about.
                let raw = r.warnings.iter().any(|w| w.contains(photocraft_io::raw::DEVELOPED_NOTE));
                if let Some(question) = s.color.open_question(&r.document).filter(|_| !raw) {
                    let held = s.hold_open(r.document, r.source_read_only);
                    return Ok(json!({"held": held, "name": name_a, "warnings": r.warnings, "color": question}));
                }
                let (index, color) = s.open_document(r.document, None);
                if let Some(st) = s.active_mut() {
                    st.source_read_only = r.source_read_only;
                }
                Ok(json!({"document": index, "name": name_a, "warnings": r.warnings, "color": color}))
            },
        )
    }

    /// Start a job that isn't an engine command (opening a file, a shell operation, a test's fake
    /// job): `work` runs on a worker, `apply` on this thread when [`Session::poll_jobs`] or
    /// [`Session::wait_job`] picks the result up. `command` and `params` identify it in
    /// [`JobInfo`]; it is not journaled. With `lock_document` the active document is locked as
    /// for a command. Runs inline on wasm.
    pub fn start_job<T: Send + 'static>(
        &mut self,
        command: &str,
        params: Value,
        label: &str,
        lock_document: bool,
        work: impl FnOnce(&JobCtx) -> Result<T> + Send + 'static,
        apply: impl FnOnce(&mut Session, T) -> Result<Value> + Send + 'static,
    ) -> Result<Started> {
        self.jobs.spawn = true;
        let r = run(self, label, lock_document, work, apply);
        self.jobs.spawn = false;
        let pending = self.jobs.pending.take();
        match (r, pending) {
            (Err(e), Some(job)) => {
                job.ctx.cancel();
                Err(e)
            }
            (Err(e), None) => Err(e),
            (Ok(_), Some(mut job)) => {
                job.command = command.to_string();
                job.params = params;
                job.journal = false;
                let id = job.id;
                self.jobs.running.push(job);
                Ok(Started::Job(id))
            }
            (Ok(v), None) => Ok(Started::Done(v)),
        }
    }

    /// Apply finished background jobs (on this thread) and return every job that ended since
    /// the last call. Cheap when nothing runs; the UI calls it every frame.
    pub fn poll_jobs(&mut self) -> Vec<JobEvent> {
        let mut i = 0;
        while i < self.jobs.running.len() {
            match self.jobs.running.get(i).map(|r| r.rx.try_recv()) {
                Some(Ok(r)) => {
                    let job = self.jobs.running.remove(i);
                    self.finish_job(job, r);
                }
                Some(Err(std::sync::mpsc::TryRecvError::Empty)) => i += 1,
                Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => {
                    let job = self.jobs.running.remove(i);
                    let label = job.label.clone();
                    self.finish_job(job, Err(EngineError::Other(format!("`{label}` stopped unexpectedly; the document is unchanged"))));
                }
                None => break,
            }
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.jobs.draining.retain(|h| !h.is_finished());
        std::mem::take(&mut self.jobs.events)
    }

    /// Block until job `id` ends, apply it, and return its result (the control channel's and
    /// MCP's default `wait`, and the CLI). `Err(Cancelled)` if it was cancelled.
    pub fn wait_job(&mut self, id: JobId) -> Result<Value> {
        if let Some(i) = self.jobs.running.iter().position(|r| r.id == id) {
            let job = self.jobs.running.remove(i);
            let r = job.rx.recv().unwrap_or_else(|_| Err(EngineError::Other(format!("`{}` stopped unexpectedly", job.label))));
            self.finish_job(job, r);
        }
        match self.jobs.recent.iter().rev().find(|j| j.id == id) {
            Some(j) if j.state == "done" => Ok(j.result.clone().unwrap_or(Value::Null)),
            Some(j) if j.state == "cancelled" => Err(EngineError::Cancelled),
            Some(j) => Err(EngineError::Other(j.error.clone().unwrap_or_else(|| "the job failed".into()))),
            None => Err(EngineError::Other(format!("no job {}", id.0))),
        }
    }

    /// Cancel job `id`: it stops at its next check, its result is discarded and the document
    /// stays unchanged. The document is unlocked at once. False if no such job is running.
    pub fn cancel_job(&mut self, id: JobId) -> bool {
        let Some(i) = self.jobs.running.iter().position(|r| r.id == id) else { return false };
        // A print worker can already have handed the job to the OS. Discarding its result
        // would falsely report cancellation; removal belongs to the system print queue.
        if matches!(self.jobs.running[i].command.as_str(), "file.print" | "file.printOneCopy") {
            return false;
        }
        let job = self.jobs.running.remove(i);
        job.ctx.cancel();
        self.end_job(&job, JobOutcome::Cancelled);
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(h) = job.handle {
            self.jobs.draining.push(h);
        }
        true
    }

    /// Cancel every running job (e.g. before quitting).
    pub fn cancel_all_jobs(&mut self) {
        let ids: Vec<JobId> = self.jobs.running.iter().map(|r| r.id).collect();
        for id in ids {
            self.cancel_job(id);
        }
    }

    /// Is any job running? (Cheap: the UI asks every frame.)
    pub fn has_jobs(&self) -> bool {
        !self.jobs.running.is_empty()
    }

    /// Running job `id`.
    pub fn job(&self, id: JobId) -> Option<JobInfo> {
        self.jobs.running.iter().find(|r| r.id == id).map(Running::info)
    }

    /// Running jobs.
    pub fn jobs(&self) -> Vec<JobInfo> {
        self.jobs.running.iter().map(Running::info).collect()
    }

    /// Running jobs followed by the last few that ended (newest last), for `jobs.list`.
    pub fn jobs_with_recent(&self) -> Vec<JobInfo> {
        self.jobs().into_iter().chain(self.jobs.recent.iter().cloned()).collect()
    }

    /// The running job that locks document `doc`, if any.
    pub fn job_on(&self, doc: DocId) -> Option<JobInfo> {
        self.jobs.running.iter().find(|r| r.doc == Some(doc)).map(Running::info)
    }

    /// The running job that locks the active document, if any.
    pub fn active_job(&self) -> Option<JobInfo> {
        self.active().and_then(|st| self.job_on(st.doc.id))
    }

    /// Wait for background workers of cancelled jobs to exit (tests and benchmarks measure
    /// cancel latency with it). Native only; returns at once on wasm.
    pub fn join_cancelled_jobs(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        for h in self.jobs.draining.drain(..) {
            let _ = h.join();
        }
    }

    /// Why command `id` can't run now because of a background job (None = no conflict).
    pub(crate) fn job_conflict(&self, id: &str, journal: bool) -> Option<String> {
        if !journal || independent_of_document(id) {
            return None;
        }
        let job = self.active_job()?;
        Some(format!("“{}” is still running on this document; wait for it to finish or cancel it (Esc)", job.label))
    }

    /// Cancel the jobs editing document `doc` (it is closing).
    pub(crate) fn cancel_jobs_on(&mut self, doc: DocId) {
        let ids: Vec<JobId> = self.jobs.running.iter().filter(|r| r.doc == Some(doc)).map(|r| r.id).collect();
        for id in ids {
            self.cancel_job(id);
        }
    }

    /// Run the command `id`: inline, or (with `background`) as a job when it is one.
    pub(crate) fn dispatch(&mut self, id: &str, params: Value, background: bool) -> Result<Started> {
        let spec = crate::commands::find(id).ok_or_else(|| EngineError::UnknownCommand(id.to_string()))?;
        // Params are named fields: anything but an object (or null, meaning none) is a caller
        // mistake, not "use every default".
        if !(params.is_object() || params.is_null()) {
            return Err(EngineError::BadParams { cmd: id.to_string(), msg: "params must be a JSON object".into() });
        }
        // Untrusted sessions gate every command, including the ones a command runs on its own
        // behalf (`file.automate.conditionalModeChange` runs `image.mode.*`), before any side
        // effect.
        if let Some(gate) = self.authorize {
            gate(id, &params)?;
        }
        // Last Filter runs as the filter it repeats: journaled, recorded in an action, faded and
        // run as a job like that filter, so a recorded step replays the same filter anywhere
        // (#3089), as Transform Again records its transform (#1440). With no filter yet it falls
        // through to its own precondition.
        if id == "filter.lastFilter"
            && let Some((filter, p)) = crate::filters::last_filter(self).filter(|(f, _)| f != id)
        {
            return self.dispatch(&filter, p, background);
        }
        if id == "file.new" {
            crate::document_preset_cmds::validate_new_params(self, &params)?;
        }
        crate::param_check::check(spec, &params)?;
        crate::filters::validate_params(id, &params)?;
        // A floating selection drops before any other command (Undo puts it back instead).
        if let Some(v) = crate::float_cmds::before_command(self, id)? {
            return Ok(Started::Done(v));
        }
        // Pixel commands follow the Channels panel target unless the caller names one.
        let run_params = crate::channel_cmds::inject_target(self, id, crate::commands::inject_kind(id, params.clone()));
        if let Err(why) = self.explicit_layer_enabled(spec, &params).unwrap_or_else(|| self.precondition(spec, &run_params)) {
            return Err(EngineError::Disabled(id.to_string(), why));
        }
        if let Some(why) = self.job_conflict(id, spec.journal) {
            return Err(EngineError::Disabled(id.to_string(), why));
        }
        // Painting on or moving a hidden layer is refused, as in Photoshop (#571).
        if let Some(why) = crate::hidden_target::refusal(self, id, &params) {
            return Err(EngineError::Other(why.into()));
        }
        self.coalesce_request = params.get("coalesce").and_then(Value::as_str).map(str::to_string);
        self.color_restrict = crate::channel_cmds::color_restriction(self, id, &run_params);
        self.jobs.spawn = background;
        // Commands this one runs on its own behalf journal after this mark (#3067).
        let nested_from = self.journal.len();
        // A replayable command is recorded as one step, so recording pauses while it runs: an
        // action it plays (a script step) must not flush the steps it ran so far into the action
        // being recorded, nor move the recording mark past the entries it is about to replace.
        let recording = if spec.journal && crate::actions_cmds::replayable(id) { self.actions.recording.take() } else { None };
        // Last-resort guard (AGENTS.md, Never crash): a command that panics anyway fails with an
        // error instead of taking the app down. `edit` only commits a document after its closure
        // returns, so the documents are unchanged.
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (spec.run)(self, &run_params)))
            .unwrap_or_else(|_| Err(EngineError::Other(format!("`{id}` failed with an internal error (logged); the document is unchanged"))));
        if self.actions.recording.is_none() {
            self.actions.recording = recording;
        }
        self.jobs.spawn = false;
        self.coalesce_request = None;
        self.color_restrict = None;
        let pending = self.jobs.pending.take();
        match (r, pending) {
            (Err(e), Some(job)) => {
                job.ctx.cancel();
                Err(e)
            }
            (Err(e), None) => Err(e),
            (Ok(_), Some(mut job)) => {
                job.command = id.to_string();
                job.params = params;
                job.journal = spec.journal;
                let jid = job.id;
                self.jobs.running.push(job);
                Ok(Started::Job(jid))
            }
            (Ok(v), None) => {
                self.after_command(id, params, spec.journal, Some(nested_from));
                Ok(Started::Done(v))
            }
        }
    }

    /// Bookkeeping after a command (or a job's apply) succeeded.
    ///
    /// `nested_from` is the journal length when the command started. A replayable command stands
    /// for the commands it ran itself (Straighten runs Image Rotation and Crop): their entries are
    /// dropped, so a recorded action replays the command once, not its steps as well (#3067).
    /// Edit › Transform › Again (and Again on a Copy) is not replayable and keeps the concrete
    /// steps it ran instead.
    fn after_command(&mut self, id: &str, params: Value, journal: bool, nested_from: Option<usize>) {
        // A layer-mask view ends when another layer becomes active (#196).
        if let Some(st) = self.active_mut() {
            crate::mask_view_cmds::fix(st);
        }
        crate::edit_menu_cmds::after_command(self, id);
        crate::automate_cmds::after_command(self, id);
        self.sync_preset_store();
        if !journal {
            return;
        }
        if let Some(from) = nested_from.filter(|_| crate::actions_cmds::replayable(id)) {
            self.journal.truncate(from);
        }
        if !crate::brush_cmds::coalesce_journal(self, id, &params) {
            self.journal.push((id.to_string(), params));
        }
    }

    /// Apply (or report the failure of) a job whose worker has delivered `r`.
    fn finish_job(&mut self, mut job: Running, r: Result<Payload>) {
        let apply = job.apply.take();
        let outcome = match r {
            Err(EngineError::Cancelled) => JobOutcome::Cancelled,
            Err(e) => JobOutcome::Failed(e.to_string()),
            Ok(_) if job.ctx.cancelled() => JobOutcome::Cancelled,
            Ok(payload) => match self.apply_job(&job, apply, payload) {
                Ok(v) => JobOutcome::Done(v),
                Err(EngineError::Cancelled) => JobOutcome::Cancelled,
                Err(e) => JobOutcome::Failed(e.to_string()),
            },
        };
        self.end_job(&job, outcome);
    }

    fn apply_job(&mut self, job: &Running, apply: Option<Apply>, payload: Payload) -> Result<Value> {
        let Running { label, doc, snapshot, coalesce, color_restrict, command, params, journal, .. } = job;
        let prev_active = self.active;
        let target = match doc {
            Some(d) => {
                let i =
                    self.documents().iter().position(|st| st.doc.id == *d).ok_or_else(|| EngineError::Other(format!("`{label}`: the document was closed")))?;
                let unchanged = snapshot.as_ref().is_some_and(|w| self.documents().get(i).is_some_and(|st| Weak::ptr_eq(w, &Arc::downgrade(&st.doc))));
                if !unchanged {
                    return Err(EngineError::Other(format!("`{label}`: the document changed while it ran; the result was discarded")));
                }
                self.active = Some(i);
                Some(i)
            }
            None => None,
        };
        self.coalesce_request = coalesce.clone();
        self.color_restrict = *color_restrict;
        // Apply-time hooks can dispatch commands too. Keep the same single-step recording
        // contract as synchronous dispatch without truncating work done while the worker ran.
        let nested_from = self.journal.len();
        let recording = if *journal && crate::actions_cmds::replayable(command) { self.actions.recording.take() } else { None };
        let r = match apply {
            Some(f) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(self, payload)))
                .unwrap_or_else(|_| Err(EngineError::Other(format!("`{label}` failed with an internal error (logged); the document is unchanged")))),
            None => Err(EngineError::Other("internal error: job applied twice".into())),
        };
        if self.actions.recording.is_none() {
            self.actions.recording = recording;
        }
        self.coalesce_request = None;
        self.color_restrict = None;
        // Finish successful command bookkeeping on the document the job edited.
        // Restoring the viewed document first would attribute Fade and slice updates
        // to that document instead, when the user switched tabs mid-job.
        if r.is_ok() {
            self.after_command(command, params.clone(), *journal, Some(nested_from));
        }
        // Keep the user's active document unless the job opened a new one.
        // This also restores the selection when the apply step returned an error.
        if target.is_some() && self.active == target {
            self.active = prev_active.filter(|i| *i < self.docs.len()).or(self.active);
        }
        r
    }

    fn end_job(&mut self, job: &Running, outcome: JobOutcome) {
        let (state, result, error) = match &outcome {
            JobOutcome::Done(v) => ("done", Some(v.clone()), None),
            JobOutcome::Failed(e) => ("failed", None, Some(e.clone())),
            JobOutcome::Cancelled => ("cancelled", None, None),
        };
        let mut info = job.info();
        info.state = state;
        info.result = result;
        info.error = error;
        if state == "done" {
            info.progress = 1.0;
        }
        self.jobs.recent.push(info);
        if self.jobs.recent.len() > RECENT {
            let extra = self.jobs.recent.len() - RECENT;
            self.jobs.recent.drain(..extra);
        }
        self.jobs.events.push(JobEvent { id: job.id, command: job.command.clone(), label: job.label.clone(), document: job.doc, outcome });
    }
}

/// Engine commands for the job table (the control channel and MCP use them too).
pub fn specs() -> Vec<crate::commands::CommandSpec> {
    use crate::commands::CommandSpec;
    vec![
        CommandSpec {
            id: "jobs.list",
            label: "List Background Jobs",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: |_| Ok(()),
            run: |s, _| Ok(json!({"jobs": s.jobs_with_recent()})),
            journal: false,
        },
        CommandSpec {
            id: "jobs.cancel",
            label: "Cancel Background Job",
            menu: &[],
            shortcut: None,
            params: r##"{"job":u64? (default: every running job)}"##,
            enabled: |_| Ok(()),
            run: |s, p| match p.get("job") {
                None | Some(Value::Null) => {
                    let n = s.jobs.running.len();
                    s.cancel_all_jobs();
                    Ok(json!({"cancelled": n.saturating_sub(s.jobs.running.len())}))
                }
                Some(v) => {
                    let id = v.as_u64().ok_or_else(|| EngineError::BadParams { cmd: "jobs.cancel".into(), msg: "`job` must be a job id".into() })?;
                    if s.cancel_job(JobId(id)) {
                        Ok(json!({"cancelled": 1}))
                    } else if s.job(JobId(id)).is_some() {
                        Err(EngineError::Other("printing cannot be cancelled here; use the system print queue".into()))
                    } else {
                        Err(EngineError::Other(format!("no running job {id}")))
                    }
                }
            },
            journal: false,
        },
    ]
}

#[cfg(test)]
#[path = "jobs_tests.rs"]
mod tests;
