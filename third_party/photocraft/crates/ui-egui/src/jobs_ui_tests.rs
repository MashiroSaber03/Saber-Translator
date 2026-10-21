//! UI tests for background jobs (#210): the whole app (eframe harness, CPU canvas) keeps
//! rendering frames while a slow job runs, shows the status-bar progress and the modal progress
//! dialog, and Esc / Cancel leave the document unchanged.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use egui::vec2;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use photocraft_engine::jobs::{JobId, Started};
use serde_json::{Value, json};

use crate::PhotocraftApp;

fn app_harness() -> Harness<'static, PhotocraftApp> {
    app_harness_with(crate::Services::default())
}

/// [`app_harness`] over the given services (the beep test records the hook's calls).
fn app_harness_with(services: crate::Services) -> Harness<'static, PhotocraftApp> {
    let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).with_max_steps(8).build_eframe(|cc| {
        PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 320, "height": 240})).unwrap();
        s.execute("layer.new.layer", json!({})).unwrap();
        s.execute("edit.fill", json!({"color": "#3366cc"})).unwrap();
        let mut app = PhotocraftApp::new(s, services);
        app.background_jobs = true;
        app
    });
    h.run_steps(4);
    h
}

/// A fake document job that runs until `gate` opens (or it is cancelled), reporting 40 %.
fn slow_job(app: &mut PhotocraftApp, gate: &Arc<AtomicBool>) -> JobId {
    let gate = gate.clone();
    let r = app
        .session
        .start_job(
            "test.slow",
            json!({}),
            "Slow Filter",
            true,
            move |ctx| {
                ctx.progress(0.4, "Filtering tiles");
                while !gate.load(Ordering::Relaxed) {
                    ctx.check()?;
                    std::thread::sleep(Duration::from_millis(1));
                }
                Ok(())
            },
            |s, ()| {
                s.edit("Slow Filter", |_, _| Ok(()))?;
                Ok(json!({"slow": true}))
            },
        )
        .unwrap();
    match r {
        Started::Job(id) => id,
        Started::Done(v) => panic!("ran inline: {v}"),
    }
}

/// Step frames for `ms` of wall time; returns (frames, slowest frame in ms).
fn frames_for(h: &mut Harness<'_, PhotocraftApp>, ms: u64) -> (u64, f64) {
    frames_for_at_least(h, ms, 0)
}

/// Prove the frame loop advances while a gated worker stays blocked, without assuming a
/// minimum frame rate on software-rendered CI machines. Also wait for the progress modal.
fn frames_for_at_least(h: &mut Harness<'_, PhotocraftApp>, ms: u64, min_frames: u64) -> (u64, f64) {
    let t = Instant::now();
    let (mut n, mut worst) = (0, 0.0f64);
    while t.elapsed() < Duration::from_millis(ms) || n < min_frames {
        assert!(t.elapsed() < Duration::from_secs(30), "frame observation timed out after {n} frames");
        let f = Instant::now();
        h.step();
        worst = worst.max(f.elapsed().as_secs_f64() * 1000.0);
        n += 1;
    }
    (n, worst)
}

#[test]
fn the_frame_loop_keeps_running_during_a_slow_job_and_esc_cancels_it() {
    let mut h = app_harness();
    let gate = Arc::new(AtomicBool::new(false));
    let doc_before = h.state().session.active().unwrap().doc.clone();
    let job = slow_job(h.state_mut(), &gate);
    let frame0 = h.state().frame;
    let (n, worst) = frames_for_at_least(&mut h, 400, 10);
    assert!(h.state().frame - frame0 >= 10, "the UI kept rendering ({n} frames)");
    eprintln!("{n} frames during the job, slowest {worst:.1} ms");
    // The status bar shows the job, and after the delay the modal progress dialog.
    assert!(h.query_all_by_label("Slow Filter").count() >= 2, "job label in the status bar and the dialog title");
    assert!(h.query_by_label("Cancel").is_some(), "the progress dialog's Cancel");
    assert!(h.query_by_label("Filtering tiles").is_some(), "the job's status message");
    // Commands that would edit the busy document are disabled with a clear reason.
    assert!(!h.state().session.is_enabled("layer.new.layer"));
    let err = h.state_mut().run("layer.new.layer", json!({})).unwrap_err();
    assert!(err.contains("Slow Filter"), "{err}");
    // Esc cancels; the document is untouched.
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert!(h.state().session.job(job).is_none());
    assert!(Arc::ptr_eq(&h.state().session.active().unwrap().doc, &doc_before));
    assert!(h.state().ui.status.contains("Cancelled"), "{}", h.state().ui.status);
    assert!(h.query_by_label("Esc to cancel").is_none(), "the dialog closed");
}

#[test]
fn a_job_started_while_drawing_requests_the_follow_up_poll() {
    let mut h = app_harness();
    let gate = Arc::new(AtomicBool::new(false));
    let job = slow_job(h.state_mut(), &gate);
    let repaints = Arc::new(AtomicUsize::new(0));
    let seen = repaints.clone();
    h.ctx.set_request_repaint_callback(move |_| {
        seen.fetch_add(1, Ordering::Relaxed);
    });

    // A job may start from a menu or dialog after the frame's initial `jobs_ui::tick`.
    // The UI must still schedule a poll, or an idle Wayland event loop leaves the document
    // locked after the worker finishes and blocks Edit > Undo / Redo.
    assert!(crate::jobs_ui::keep_polling(h.state(), &h.ctx));
    assert_eq!(repaints.load(Ordering::Relaxed), 1);

    crate::jobs_ui::cancel(h.state_mut(), job);
    h.run_steps(2);
}

#[test]
fn a_finished_job_lands_as_one_undo_step_and_the_dialog_cancel_button_works() {
    let mut h = app_harness();
    let gate = Arc::new(AtomicBool::new(false));
    let steps = h.state().session.active().unwrap().history.past_len();
    slow_job(h.state_mut(), &gate);
    frames_for(&mut h, 320);
    gate.store(true, Ordering::Relaxed);
    let t = Instant::now();
    while h.state().session.has_jobs() && t.elapsed() < Duration::from_secs(10) {
        h.step();
    }
    h.run_steps(2);
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);

    // A second job, cancelled from the dialog's button.
    let gate = Arc::new(AtomicBool::new(false));
    let job = slow_job(h.state_mut(), &gate);
    frames_for(&mut h, 320);
    h.get_by_label("Cancel").click();
    h.run_steps(3);
    assert!(h.state().session.job(job).is_none());
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
}

#[test]
fn filters_from_the_ui_run_in_the_background() {
    let mut h = app_harness();
    let steps = h.state().session.active().unwrap().history.past_len();
    let r = h.state_mut().run("filter.blur.gaussianBlur", json!({"radius": 3})).unwrap();
    assert_eq!(r["pending"], true, "{r}");
    let t = Instant::now();
    while h.state().session.has_jobs() && t.elapsed() < Duration::from_secs(30) {
        h.step();
    }
    h.run_steps(2);
    let st = h.state().session.active().unwrap();
    assert_eq!(st.history.past_len(), steps + 1, "one step for the blur");
    assert_eq!(h.state().ui.status, "Gaussian Blur");
}

#[test]
fn control_requests_wait_for_jobs_unless_asked_not_to() {
    let mut h = app_harness();
    let ctx = h.ctx.clone();
    let (req, _rx) = crate::ControlRequest::new("engine.execute", json!({"command": "filter.blur.gaussianBlur", "params": {"radius": 2}}));
    let job = match crate::control::handle(h.state_mut(), &ctx, &req) {
        crate::control::Outcome::AfterJob(job) => job,
        _ => panic!("expected a deferred reply"),
    };
    assert!(h.state().session.jobs().iter().any(|j| j.id == job) || h.state().session.jobs_with_recent().iter().any(|j| j.id == job));
    // The app's frame loop answers the request when the job lands.
    let (req, rx) = crate::ControlRequest::new("engine.execute", json!({"command": "jobs.list"}));
    let _ = crate::control::handle(h.state_mut(), &ctx, &req);
    drop(rx);
    let reply = job_reply(&mut h, job);
    assert_eq!(reply["ok"], true, "{reply}");
    assert!(reply["result"]["filter"].is_object(), "{reply}");

    // `wait: false` replies with the job id at once.
    let (req, _rx) = crate::ControlRequest::new("engine.execute", json!({"command": "filter.blur.gaussianBlur", "params": {"radius": 2}, "wait": false}));
    let crate::control::Outcome::Done(v) = crate::control::handle(h.state_mut(), &ctx, &req) else { panic!("expected an immediate reply") };
    let id = v["result"]["job"].as_u64().unwrap();
    let (req, _rx) = crate::ControlRequest::new("jobs.cancel", json!({"job": id}));
    let crate::control::Outcome::Done(v) = crate::control::handle(h.state_mut(), &ctx, &req) else { panic!() };
    assert_eq!(v["ok"], true, "{v}");
    let (req, _rx) = crate::ControlRequest::new("jobs.cancel", json!({"job": 99999}));
    let crate::control::Outcome::Done(v) = crate::control::handle(h.state_mut(), &ctx, &req) else { panic!() };
    assert_eq!(v["ok"], false);
}

#[test]
fn accepted_control_job_applies_after_its_dispatch_deadline() {
    let mut h = app_harness();
    let ctx = h.ctx.clone();
    let steps = h.state().session.active().unwrap().history.past_len();
    let deadline = Instant::now() + Duration::from_secs(1);
    let (mut req, reply) = crate::ControlRequest::new("engine.execute", json!({"command": "filter.blur.gaussianBlur", "params": {"radius": 2}}));
    req.deadline = Some(deadline);
    let (tx, rx) = std::sync::mpsc::channel();
    h.state_mut().control_rx = Some(rx);
    tx.send(req).unwrap();
    h.state_mut().drain_control(&ctx);
    let job = h.state().jobs.waiters.first().expect("the accepted request waits for its job").0;
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps, "the job has not been applied yet");
    assert!(reply.try_recv().is_err());
    // No frames poll the worker until the transport's deadline has elapsed.
    std::thread::sleep(deadline.saturating_duration_since(Instant::now()));
    assert!(Instant::now() >= deadline);
    drop(reply);
    let result = job_reply(&mut h, job);
    assert_eq!(result["ok"], true, "{result}");
    assert!(result["result"]["filter"].is_object(), "{result}");
    assert_eq!(h.state().session.active().unwrap().history.past_len(), steps + 1);
}

/// The control reply for `job`, once the app's frame loop has applied it.
fn job_reply(h: &mut Harness<'static, PhotocraftApp>, job: JobId) -> Value {
    let (tx, rx) = std::sync::mpsc::channel();
    h.state_mut().jobs.waiters.push((job, tx));
    let t = Instant::now();
    loop {
        h.step();
        if let Ok(v) = rx.try_recv() {
            return v;
        }
        assert!(t.elapsed() < Duration::from_secs(30));
    }
}

/// #511: a filter started by a menu item (Blur More has no dialog) or by a filter dialog's OK
/// replies with its result once applied, like `engine.execute`; `wait: false` replies at once.
#[test]
fn menu_and_dialog_requests_wait_for_jobs_unless_asked_not_to() {
    let mut h = app_harness();
    let ctx = h.ctx.clone();
    let control = |h: &mut Harness<'static, PhotocraftApp>, method: &str, params: Value| {
        let (req, _rx) = crate::ControlRequest::new(method, params);
        crate::control::handle(h.state_mut(), &ctx, &req)
    };
    // The job a request started: deferred while waiting, `{job, pending}` at once otherwise.
    // Lets it land and checks the filter's result.
    let land = |h: &mut Harness<'static, PhotocraftApp>, what: &str, outcome: crate::control::Outcome, wait: bool| {
        let job = match (outcome, wait) {
            (crate::control::Outcome::AfterJob(job), true) => job,
            (crate::control::Outcome::Done(v), false) => {
                assert_eq!(v["result"]["pending"], true, "{what}: {v}");
                JobId(v["result"]["job"].as_u64().unwrap())
            }
            _ => panic!("{what}, wait {wait}: wrong reply kind"),
        };
        let reply = job_reply(h, job);
        assert!(reply["result"]["filter"].is_object(), "{what}: {reply}");
    };
    let steps = |h: &Harness<'static, PhotocraftApp>| h.state().session.active().unwrap().history.past_len();
    for wait in [true, false] {
        let before = steps(&h);
        let outcome = control(&mut h, "ui.menu.invoke", json!({"id": "filter.blur.blurMore", "wait": wait}));
        land(&mut h, "Blur More", outcome, wait);
        let crate::control::Outcome::Done(v) = control(&mut h, "ui.menu.invoke", json!({"id": "filter.blur.gaussianBlur"})) else { panic!("dialog") };
        let dialog = v["result"]["dialog"].as_u64().expect("Gaussian Blur opens its dialog");
        control(&mut h, "ui.dialog.set", json!({"dialog": dialog, "field": "radius", "value": 3}));
        let outcome = control(&mut h, "ui.dialog.confirm", json!({"dialog": dialog, "wait": wait}));
        land(&mut h, "Gaussian Blur", outcome, wait);
        assert_eq!(steps(&h), before + 2, "both filters applied");
    }
}

fn psd_bytes() -> Vec<u8> {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    let doc = (*s.active().unwrap().doc).clone();
    photocraft_io::export(&doc, "x.psd", &Default::default()).unwrap().bytes
}

#[test]
fn files_open_in_the_background_with_a_tab_and_cancel_closes_it() {
    let mut h = app_harness();
    let bytes = psd_bytes();
    h.state_mut().open_file("/tmp/photocraft-test/bg.psd", &bytes).unwrap();
    // The tab is there before the document.
    assert_eq!(h.state().jobs.opens.len(), 1);
    assert_eq!(h.state().session.documents().len(), 1);
    let t = Instant::now();
    while !h.state().jobs.opens.is_empty() && t.elapsed() < Duration::from_secs(30) {
        h.step();
    }
    h.run_steps(2);
    let st = h.state().session.active().unwrap();
    assert_eq!(st.doc.size.width, 64);
    assert_eq!(st.path.as_deref(), Some("/tmp/photocraft-test/bg.psd"));
    assert_eq!(h.state().ui.recent_files.first().map(String::as_str), Some("/tmp/photocraft-test/bg.psd"));

    // Cancel: the half-open tab closes, no document is added.
    h.state_mut().open_file("/tmp/photocraft-test/bg2.psd", &bytes).unwrap();
    let job = h.state().jobs.opens[0].job;
    crate::jobs_ui::cancel(h.state_mut(), job);
    h.run_steps(3);
    assert!(h.state().jobs.opens.is_empty());
    assert_eq!(h.state().session.documents().len(), 2);
    assert!(h.state().jobs.focus.is_none());
}

/// #2423: a long file name used to wrap the card's title and push the percentage and Cancel
/// below the card. The name stays on one line and every row is inside the card.
#[test]
fn the_opening_card_keeps_a_long_name_and_cancel_inside() {
    let mut h = app_harness();
    let gate = Arc::new(AtomicBool::new(false));
    let job = slow_job(h.state_mut(), &gate);
    let long = "a7rv-test_2024_02_17_shot_30_3705_with_an_even_longer_suffix_that_never_ends.psb";
    for name in ["short.psd", long] {
        h.state_mut().jobs.opens.clear();
        h.state_mut().jobs.opens.push(crate::jobs_ui::OpenTab { job, name: name.to_string(), path: None, slot: None, order: None });
        h.state_mut().jobs.focus = Some(job);
        h.run_steps(4);
        let r = crate::jobs_ui::open_card_rects(&h.ctx).expect("the opening card was drawn");
        assert!(r.title.height() < 22.0, "{name}: the title is one line, {:?}", r.title);
        assert!(r.card.contains_rect(r.title), "{name}: title {:?} inside card {:?}", r.title, r.card);
        assert!(r.card.contains_rect(r.cancel), "{name}: Cancel {:?} inside card {:?}", r.cancel, r.card);
        assert!(r.title.right() <= r.card.right() - 10.0, "{name}: the title keeps the card's padding");
        assert!(r.cancel.width() > 0.0 && r.cancel.height() > 0.0, "{name}: Cancel was drawn");
    }
    gate.store(true, Ordering::Relaxed);
    h.state_mut().jobs.focus = None;
    h.state_mut().jobs.opens.clear();
    h.run_steps(4);
}

#[test]
fn templates_open_untitled_in_the_background() {
    let mut h = app_harness();
    h.state_mut().open_file("/tmp/photocraft-test/card.psdt", &psd_bytes()).unwrap();
    assert_eq!(h.state().jobs.opens[0].name, "Untitled-1");
    let t = Instant::now();
    while !h.state().jobs.opens.is_empty() && t.elapsed() < Duration::from_secs(30) {
        h.step();
    }
    h.run_steps(2);
    let st = h.state().session.active().unwrap();
    assert_eq!((st.doc.name.as_str(), st.path.as_deref()), ("Untitled-1", None));
    assert_eq!(h.state().ui.recent_files.first().map(String::as_str), Some("/tmp/photocraft-test/card.psdt"));
}

/// Gates for a fake save: it encodes until `encode` opens and writes until `write` opens
/// (polling for cancellation only while encoding, as the real one does).
#[derive(Clone, Default)]
struct Gates {
    encode: Arc<AtomicBool>,
    write: Arc<AtomicBool>,
    written: Arc<std::sync::Mutex<Vec<String>>>,
}

impl Gates {
    fn open() -> Self {
        let g = Gates::default();
        g.encode.store(true, Ordering::Relaxed);
        g.write.store(true, Ordering::Relaxed);
        g
    }

    /// A `save_file` service held by these gates that records the path written, or fails with
    /// `fail` once encoded.
    fn save(&self, fail: Option<&'static str>) -> crate::SaveFileFn {
        let g = self.clone();
        Arc::new(move |_, path, _, ctl| {
            ctl.progress(0.0, "Encoding");
            while !g.encode.load(Ordering::Relaxed) {
                ctl.check()?;
                std::thread::sleep(Duration::from_millis(1));
            }
            if let Some(e) = fail {
                return Err(e.into());
            }
            ctl.commit()?;
            ctl.progress(0.0, "Writing");
            while !g.write.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            g.written.lock().unwrap().push(path.to_string());
            Ok(vec!["a warning".into()])
        })
    }

    fn written(&self) -> Vec<String> {
        self.written.lock().unwrap().clone()
    }
}

fn noticed(app: &PhotocraftApp, text: &str) -> bool {
    app.ui.notices.iter().any(|n| n.title.contains(text) || n.lines.iter().any(|l| l.contains(text)))
}

fn step_until(h: &mut Harness<'_, PhotocraftApp>, done: impl Fn(&PhotocraftApp) -> bool) {
    let t = Instant::now();
    while !done(h.state()) && t.elapsed() < Duration::from_secs(30) {
        h.step();
    }
    h.run_steps(2);
}

#[test]
fn saves_run_in_the_background_and_are_recorded_when_they_end() {
    let mut h = app_harness();
    let gates = Gates { write: Arc::new(AtomicBool::new(true)), ..Gates::default() };
    h.state_mut().services.save_file = Some(gates.save(None));
    let r = h.state_mut().save_as(Some("/tmp/photocraft-test/big.psd".into())).unwrap();
    assert_eq!(r["pending"], true);
    assert_eq!(r["job"].as_u64(), h.state().jobs.saves.first().map(|s| s.job.0), "control clients get the job to wait on");
    assert!(h.state().saving());

    // The window keeps drawing; the document stays unsaved and can't be edited meanwhile.
    let (frames, _) = frames_for_at_least(&mut h, 400, 4);
    assert!(frames >= 4, "only {frames} frames while saving");
    let st = h.state().session.active().unwrap();
    assert!(st.is_dirty() && st.path.is_none());
    assert!(h.state_mut().run("layer.new.layer", json!({})).is_err(), "the saving document is locked");
    // The progress dialog names the save and shows its stage; the status bar reads "Saving 0%"
    // until the encoder reports progress (#3389).
    assert_eq!(h.query_all_by_label("Saving big.psd").count(), 1);
    h.get_by_label("Encoding");
    h.get_by_label("Saving 0%");

    gates.encode.store(true, Ordering::Relaxed);
    step_until(&mut h, |a| !a.saving());
    assert_eq!(gates.written(), ["/tmp/photocraft-test/big.psd"]);
    let st = h.state().session.active().unwrap();
    assert!(!st.is_dirty());
    assert_eq!(st.path.as_deref(), Some("/tmp/photocraft-test/big.psd"));
    assert_eq!(st.doc.name, "big.psd", "the document takes the saved file's name, as a synchronous Save As does");
    assert_eq!(h.state().ui.status, "Saved big.psd: a warning");
    assert!(noticed(h.state(), "a warning"), "the export warnings are shown");
    assert!(h.state_mut().run("layer.new.layer", json!({})).is_ok(), "unlocked again");
}

#[test]
fn a_failed_save_or_one_cancelled_while_encoding_leaves_the_document_unsaved() {
    let mut h = app_harness();
    let gates = Gates::open();
    h.state_mut().services.save_file = Some(gates.save(Some("disk full")));
    h.state_mut().save_as(Some("/tmp/photocraft-test/a.psd".into())).unwrap();
    step_until(&mut h, |a| !a.saving());
    let st = h.state().session.active().unwrap();
    assert!(st.is_dirty() && st.path.is_none());
    assert!(noticed(h.state(), "disk full"), "the failure is reported");

    gates.encode.store(false, Ordering::Relaxed);
    h.state_mut().services.save_file = Some(gates.save(None));
    h.state_mut().save_as(Some("/tmp/photocraft-test/b.psd".into())).unwrap();
    h.run_steps(2);
    h.key_press(egui::Key::Escape);
    step_until(&mut h, |a| !a.saving());
    assert!(gates.written().is_empty(), "nothing written");
    let st = h.state().session.active().unwrap();
    assert!(st.is_dirty() && st.path.is_none());
    assert_eq!(h.state().ui.status, "Cancelled Saving b.psd");
}

/// Step until the save in `h` has started writing.
fn until_writing(h: &mut Harness<'_, PhotocraftApp>) {
    let t = Instant::now();
    while h.state().session.jobs().first().is_none_or(|j| j.message != "Writing") && t.elapsed() < Duration::from_secs(30) {
        h.step();
    }
}

#[test]
fn a_save_that_has_started_writing_cannot_be_cancelled() {
    let mut h = app_harness();
    let gates = Gates { encode: Arc::new(AtomicBool::new(true)), ..Gates::default() };
    h.state_mut().services.save_file = Some(gates.save(None));
    h.state_mut().save_as(Some("/tmp/photocraft-test/w.psd".into())).unwrap();
    until_writing(&mut h);
    let job = h.state().jobs.saves[0].job;
    crate::jobs_ui::cancel(h.state_mut(), job);
    h.key_press(egui::Key::Escape);
    h.run_steps(2);
    assert_eq!(h.state().ui.status, "Saving w.psd is writing the file and can no longer be cancelled");
    assert!(h.state().session.job(job).is_some(), "still running, the document still locked");

    gates.write.store(true, Ordering::Relaxed);
    step_until(&mut h, |a| !a.saving());
    assert_eq!(gates.written(), ["/tmp/photocraft-test/w.psd"]);
    assert!(!h.state().session.active().unwrap().is_dirty(), "the written save is recorded");
}

/// #3389: like Photoshop, a running save reads "Saving 45%" in the status bar and appends
/// " - Saving 45%" to its document's tab title, in both tab strips; both go when it ends.
#[test]
fn a_running_save_shows_its_percentage_in_the_status_bar_and_tab_title() {
    for theme in crate::theme::ThemeKind::ALL {
        let mut h = app_harness();
        h.state_mut().run("prefs.set", json!({"values": {"interface.theme": theme.id()}})).unwrap();
        h.run_steps(2);
        let gate = Arc::new(AtomicBool::new(false));
        let g = gate.clone();
        let save: crate::SaveFileFn = Arc::new(move |_, _, _, ctl| {
            ctl.progress(0.4567, "Encoding");
            while !g.load(Ordering::Relaxed) {
                ctl.check()?;
                std::thread::sleep(Duration::from_millis(1));
            }
            ctl.commit()?;
            ctl.progress(1.0, "Writing");
            Ok(Vec::new())
        });
        h.state_mut().services.save_file = Some(save);
        h.state_mut().save_as(Some("/tmp/photocraft-test/pct.psd".into())).unwrap();
        let t = Instant::now();
        while h.query_by_label("Saving 45%").is_none() && t.elapsed() < Duration::from_secs(30) {
            h.step();
        }
        assert!(h.query_by_label("Saving 45%").is_some(), "{theme:?}: the status bar reads Saving 45%");
        assert!(h.query_by_label_contains(" - Saving 45%").is_some(), "{theme:?}: the tab title carries the save's progress");
        gate.store(true, Ordering::Relaxed);
        step_until(&mut h, |a| !a.saving());
        assert!(h.query_by_label_contains("Saving").is_none(), "{theme:?}: the readout and the tab suffix are gone once saved");
        assert_eq!(h.state().session.active().unwrap().path.as_deref(), Some("/tmp/photocraft-test/pct.psd"));
    }
}

/// One frame with the window's close button pressed; whether the guard cancelled the close.
fn press_window_close(app: &mut PhotocraftApp) -> bool {
    let mut info = egui::ViewportInfo::default();
    info.events.push(egui::ViewportEvent::Close);
    let mut input = egui::RawInput::default();
    input.viewports.insert(egui::ViewportId::ROOT, info);
    let mut out = egui::Context::default().run_ui(input, |ui| crate::discard_ui::guard_window_close(app, ui.ctx()));
    out.textures_delta.clear();
    out.viewport_output.remove(&egui::ViewportId::ROOT).is_some_and(|o| o.commands.iter().any(|c| matches!(c, egui::ViewportCommand::CancelClose)))
}

/// A clean document (so the close prompt has nothing to ask) saved as a copy in the background.
fn save_a_copy(h: &mut Harness<'static, PhotocraftApp>, gates: &Gates, path: &str) {
    h.state_mut().services.save_file = Some(gates.save(None));
    let st = h.state_mut().session.active_mut().unwrap();
    st.saved_revision = st.revision;
    assert_eq!(h.state_mut().write_document(path.into(), &crate::ExportSettings::default(), true), Ok(None));
    h.run_steps(2);
}

#[test]
fn quitting_waits_for_a_background_save_to_finish() {
    let mut h = app_harness();
    let gates = Gates { write: Arc::new(AtomicBool::new(true)), ..Gates::default() };
    save_a_copy(&mut h, &gates, "/tmp/photocraft-test/copy.psd");
    assert!(press_window_close(h.state_mut()), "the close waits for the save");
    assert!(h.state().jobs.close_after_saves);

    gates.encode.store(true, Ordering::Relaxed);
    step_until(&mut h, |a| !a.saving());
    assert_eq!(gates.written(), ["/tmp/photocraft-test/copy.psd"]);
    assert!(!h.state().jobs.close_after_saves, "the close is repeated once the save ends");
}

#[test]
fn a_failed_save_does_not_close_the_window_over_its_error() {
    let mut h = app_harness();
    let gates = Gates::default();
    h.state_mut().services.save_file = Some(gates.save(Some("disk full")));
    let st = h.state_mut().session.active_mut().unwrap();
    st.saved_revision = st.revision;
    h.state_mut().write_document("/tmp/photocraft-test/copy.psd".into(), &crate::ExportSettings::default(), true).unwrap();
    assert!(press_window_close(h.state_mut()));
    gates.encode.store(true, Ordering::Relaxed);
    step_until(&mut h, |a| !a.saving());
    assert!(noticed(h.state(), "disk full"));
    assert!(!h.state().jobs.close_after_saves, "the window stays open to show the error");
}

#[test]
fn a_cancelled_save_still_counts_until_its_worker_returns() {
    let mut h = app_harness();
    // A worker that keeps encoding after the cancel, as the real encoder does (it checks only
    // before writing).
    let (encode, release) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let (e, r) = (encode.clone(), release.clone());
    h.state_mut().services.save_file = Some(Arc::new(move |_, _, _, ctl| {
        while !r.load(Ordering::Relaxed) {
            e.store(true, Ordering::Relaxed);
            std::thread::sleep(Duration::from_millis(1));
        }
        ctl.commit()?;
        Ok(Vec::new())
    }));
    let st = h.state_mut().session.active_mut().unwrap();
    st.saved_revision = st.revision;
    h.state_mut().write_document("/tmp/photocraft-test/c.psd".into(), &crate::ExportSettings::default(), true).unwrap();
    while !encode.load(Ordering::Relaxed) {
        std::thread::sleep(Duration::from_millis(1));
    }
    let job = h.state().jobs.saves[0].job;
    crate::jobs_ui::cancel(h.state_mut(), job);
    h.run_steps(3);
    assert!(h.state().session.job(job).is_none(), "the job ended");
    assert!(h.state().saving(), "but its worker is still running");
    assert!(press_window_close(h.state_mut()), "quitting waits for the worker");
    assert!(
        h.state_mut().save_as(Some("/tmp/photocraft-test/c.psd".into())).unwrap_err().contains("still running"),
        "a second save to the same file waits for it too"
    );

    release.store(true, Ordering::Relaxed);
    step_until(&mut h, |a| !a.saving());
    assert!(!h.state().jobs.close_after_saves, "the held close is repeated once the worker returns");
}

/// Preferences ▸ General ▸ Beep When Done (off by default): a finished background job beeps
/// once through the services hook; with the preference off it stays silent.
#[test]
fn a_finished_job_beeps_when_the_preference_is_on() {
    static BEEPS: AtomicUsize = AtomicUsize::new(0);
    fn count_beep() {
        BEEPS.fetch_add(1, Ordering::Relaxed);
    }
    BEEPS.store(0, Ordering::Relaxed);
    let mut h = app_harness_with(crate::Services { beep: Some(count_beep), ..Default::default() });
    // Off by default: the first job finishes without a beep.
    let gate = Arc::new(AtomicBool::new(true));
    let _ = slow_job(h.state_mut(), &gate);
    frames_for(&mut h, 800);
    assert_eq!(BEEPS.load(Ordering::Relaxed), 0, "silent by default");
    // On: the next finished job beeps once.
    h.state_mut().run("prefs.set", json!({"path": "general.beepWhenDone", "value": true})).unwrap();
    let gate = Arc::new(AtomicBool::new(true));
    let _ = slow_job(h.state_mut(), &gate);
    frames_for(&mut h, 800);
    assert_eq!(BEEPS.load(Ordering::Relaxed), 1, "one beep for the finished job");
}

#[test]
fn printing_keeps_drawing_and_reports_only_the_final_spooler_result() {
    for success in [true, false] {
        let mut h = app_harness();
        let gate = Arc::new(AtomicBool::new(false));
        let worker_gate = gate.clone();
        let Started::Job(id) = h
            .state_mut()
            .session
            .start_job(
                "file.print",
                json!({}),
                "Print",
                false,
                move |_| {
                    while !worker_gate.load(Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    if success {
                        Ok(json!({"sent": true, "spooler": "request id is Office-42", "pdf": null}))
                    } else {
                        Err(photocraft_engine::EngineError::Other("queue refused".into()))
                    }
                },
                |_, result| Ok(result),
            )
            .unwrap()
        else {
            panic!("must run on a worker")
        };
        let frame = h.state().frame;
        h.run_steps(4);
        assert!(h.state().frame > frame);
        assert!(!h.state().ui.status.contains("Sent to the printer"));
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert!(h.state().session.job(id).is_some(), "Esc cannot retract a submitted job");
        let (request, _) = crate::control::ControlRequest::new("jobs.cancel", json!({"job": id.0}));
        let ctx = h.ctx.clone();
        let crate::control::Outcome::Done(reply) = crate::control::handle(h.state_mut(), &ctx, &request) else { panic!("cancel replies immediately") };
        assert_eq!(reply["result"]["cancelled"], 0, "control must not falsely report cancellation: {reply}");
        let st = h.state_mut().session.active_mut().unwrap();
        st.saved_revision = st.revision;
        assert!(press_window_close(h.state_mut()), "quit waits for print submission");
        assert!(h.state().jobs.close_after_saves);
        gate.store(true, Ordering::Relaxed);
        let deadline = Instant::now() + Duration::from_secs(10);
        while h.state().session.has_jobs() && Instant::now() < deadline {
            h.step();
        }
        assert!(!h.state().session.has_jobs());
        assert!(!h.state().jobs.close_after_saves, "close repeats on success or is cleared for a visible error");
        if success {
            assert!(h.state().ui.status.contains("Office-42"), "{}", h.state().ui.status);
            assert!(!h.state().ui.status_error);
        } else {
            assert!(h.state().ui.status.contains("queue refused"), "{}", h.state().ui.status);
            assert!(h.state().ui.status_error);
        }
    }
}

fn layer_count(app: &PhotocraftApp) -> usize {
    app.session.active().unwrap().doc.layers.len()
}

/// Free Transform the active layer to the right half of the canvas, then ↩.
fn transform_and_commit(h: &mut Harness<'static, PhotocraftApp>, copy: bool) {
    let ctx = h.ctx.clone();
    let app = h.state_mut();
    if copy { crate::transform_tool::begin_copy(app, &ctx) } else { crate::transform_tool::begin(app, &ctx) }.unwrap();
    app.ui.transform.as_mut().unwrap().quad = [[160.0, 0.0], [320.0, 0.0], [320.0, 120.0], [160.0, 120.0]];
    crate::transform_tool::commit(app);
}

#[test]
fn a_free_transform_commit_runs_in_the_background_and_keeps_its_preview() {
    let mut h = app_harness();
    let (layers, steps) = (layer_count(h.state()), h.state().session.active().unwrap().history.past_len());
    transform_and_commit(&mut h, true);
    let app = h.state();
    // The warp runs as a job; the moving pixels stay on screen until it lands, and the box's
    // history is closed meanwhile.
    let job = crate::transform_tool::committing(app).expect("the commit runs as a job");
    assert!(app.session.job(job).is_some_and(|j| j.label == "Free Transform"));
    assert!(app.ui.transform.is_some() && app.transform_preview.is_some());
    assert_eq!(crate::transform_tool::is_enabled(app, "edit.undo"), Some(false));
    assert!(h.state_mut().run("edit.undo", json!({})).is_err());
    crate::transform_tool::commit(h.state_mut());
    let t = Instant::now();
    while h.state().session.has_jobs() && t.elapsed() < Duration::from_secs(10) {
        h.step();
    }
    h.run_steps(2);
    let app = h.state();
    assert!(app.ui.transform.is_none() && app.transform_preview.is_none(), "the session ends with the job");
    // The copy and its transform are one step, as when the commit ran inline.
    let st = app.session.active().unwrap();
    assert_eq!((layer_count(app), st.history.past_len()), (layers + 1, steps + 1));
    assert_eq!(st.history.undo_label(), Some("Free Transform"));
    let moved = st.doc.layer(st.active_layer.unwrap()).unwrap().surface().unwrap().content_bounds();
    assert_eq!((moved.x0, moved.y0, moved.x1, moved.y1), (160, 0, 320, 120));
}

#[test]
fn esc_cancels_a_background_free_transform_commit() {
    let mut h = app_harness();
    let (layers, steps) = (layer_count(h.state()), h.state().session.active().unwrap().history.past_len());
    let before = h.state().session.active().unwrap().doc.clone();
    transform_and_commit(&mut h, true);
    let job = crate::transform_tool::committing(h.state()).expect("the commit runs as a job");
    // Esc reaches the transform tool first (shortcuts), which cancels the job.
    crate::transform_tool::cancel(h.state_mut());
    h.run_steps(3);
    let app = h.state();
    assert!(app.session.job(job).is_none());
    assert!(app.ui.transform.is_none() && app.transform_preview.is_none());
    // The copy is taken back with the transform, leaving nothing to redo.
    let st = app.session.active().unwrap();
    assert_eq!((layer_count(app), st.history.past_len()), (layers, steps));
    assert!(!st.history.can_redo());
    assert_eq!(st.doc.layers.len(), before.layers.len());
}

/// #4559: a background open (the desktop app's) also warns, when its document arrives, about
/// type in a font that isn't installed.
#[test]
fn background_open_warns_about_missing_fonts() {
    let mut s = photocraft_engine::Session::new();
    s.execute("file.new", json!({"width": 40, "height": 20})).unwrap();
    s.execute("type.create", json!({"text": "Hi", "x": 2, "y": 12, "size": 10, "font": "NoSuchFamily4559"})).unwrap();
    let doc = (*s.active().unwrap().doc).clone();
    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
    // What the open job leaves behind: the decoded document in the session, then its result.
    let (index, _) = app.session.open_document(doc, None);
    super::finish_open(&mut app, "poster.psd", None, &json!({"document": index, "warnings": []}), None).unwrap();
    let want = vec!["font 'NoSuchFamily4559' is not installed; text may render using a fallback face".to_string()];
    assert!(app.ui.notices.last().is_some_and(|n| n.title.contains("poster.psd") && n.lines == want), "{:?}", app.ui.notices);
}
