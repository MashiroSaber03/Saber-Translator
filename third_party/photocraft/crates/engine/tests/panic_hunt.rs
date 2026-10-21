//! Fuzz every command with adversarial params; a command must return `Err`, never panic or hang.
//! (Rule 9 in AGENTS.md.) Each command runs on its own thread with a timeout, so one bad command
//! can't wedge the run — panics are caught and hangs are reported. Opt-in (slow):
//! `cargo test -p photocraft-engine --test panic_hunt -- --ignored`.
use photocraft_engine::{Session, command_specs};
use serde_json::{Value, json};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, Once, PoisonError, mpsc};
use std::time::Duration;

fn fresh() -> Session {
    // Belt and braces with the EXTERNAL skip below: nothing in this process may reach a printer.
    photocraft_engine::print_cmds::disable_printing();
    let mut s = Session::new();
    s.execute("file.new", json!({"width": 24, "height": 16})).unwrap();
    s.execute("layer.new.layer", json!({})).unwrap();
    s.execute("select.rect", json!({"x": 1, "y": 1, "width": 6, "height": 5})).unwrap();
    s
}

/// Wall-clock budget for one command on the 24×16 test document. Every command needs well under
/// 0.3 s even on a heavily loaded machine, so this only catches real hangs.
const DEADLINE: Duration = Duration::from_secs(4);

/// Panics anywhere in the process, counted by a hook. `Session::execute` contains a command's
/// panic and returns an ordinary `Err`, so the unwind never reaches `catch_unwind` here; the hook
/// still sees it, on whichever thread (rayon workers included) it happened (#4982).
static PANICS: AtomicUsize = AtomicUsize::new(0);
/// One run at a time, so a panic is charged to the run that caused it.
static RUN: Mutex<()> = Mutex::new(());

/// Run `f` on a worker thread. `None` if it timed out, else whether it finished without
/// panicking, contained or not (returning Ok or a graceful Err both count).
fn run_watched(f: impl FnOnce() + Send + 'static) -> Option<bool> {
    static HOOK: Once = Once::new();
    HOOK.call_once(|| {
        let default = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            PANICS.fetch_add(1, Ordering::SeqCst);
            default(info);
        }));
    });
    let _one = RUN.lock().unwrap_or_else(PoisonError::into_inner);
    let before = PANICS.load(Ordering::SeqCst);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let r = catch_unwind(AssertUnwindSafe(f));
        let _ = tx.send(r.is_ok());
    });
    rx.recv_timeout(DEADLINE).ok().map(|ok| ok && PANICS.load(Ordering::SeqCst) == before)
}

/// Run `id` with `p` on a fresh session; see [`run_watched`].
fn run_once(id: &str, p: &Value) -> Option<bool> {
    let (id, p) = (id.to_string(), p.clone());
    run_watched(move || {
        let mut s = fresh();
        let _ = s.execute(&id, p);
    })
}

/// `Err` with a reason on panic or hang, `Ok` otherwise.
///
/// The first command to touch process-wide lazy state pays for it: the first `type.create`
/// starts the shared text engine, which scans every system font (3.5–5 s under load; #2994). A
/// real hang reproduces and one-time setup doesn't, so a timeout counts only if a rerun on a
/// fresh session times out too.
fn run_guarded(id: &str, p: &Value) -> Result<(), String> {
    match run_once(id, p).or_else(|| run_once(id, p)) {
        Some(true) => Ok(()),
        Some(false) => Err("panicked".into()),
        None => Err(format!("hung (>{}s, twice)", DEADLINE.as_secs())),
    }
}

#[test]
#[ignore = "slow full-registry fuzz; run with --ignored"]
fn no_command_panics_or_hangs_on_adversarial_params() {
    let params: Vec<Value> = vec![
        json!({}),
        json!({"x": -9, "y": -9, "radius": 0, "amount": 0, "angle": 0, "opacity": 0, "scale": 0, "levels": 0, "tolerance": 0, "gamma": 0, "layer": 999999, "channel": 999, "index": 999999, "count": 0}),
        json!({"x": 1e5, "y": -1e5, "radius": 300, "amount": 300, "angle": 1e5, "opacity": -100, "scale": -5, "gamma": -1}),
        json!({"points": [], "from": [0, 0], "to": [0, 0], "stops": [], "matrix": [0, 0, 0, 0, 0, 0], "colors": [], "name": "", "mode": "", "style": ""}),
    ];
    let mut bad: Vec<String> = Vec::new();
    for spec in command_specs() {
        for p in &params {
            // Print and Print One Copy reach real printers (lp, or the Windows driver): fuzz them
            // as dry runs so the hunt never prints a page.
            let mut p = p.clone();
            if spec.id.starts_with("file.print")
                && let Some(o) = p.as_object_mut()
            {
                o.insert("dryRun".into(), json!(true));
            }
            let p = &p;
            if let Err(why) = run_guarded(spec.id, p) {
                bad.push(format!("{} [{why}] <- {p}", spec.id));
                break;
            }
        }
    }
    assert!(bad.is_empty(), "{} commands panicked or hung:\n{}", bad.len(), bad.join("\n"));
}

/// #4982: a panic the code under test contains still fails the run; a graceful `Err` does not.
#[test]
fn contained_panics_count_and_graceful_errors_do_not() {
    let contained = run_watched(|| {
        let _ = catch_unwind(|| panic!("contained, as Session::execute does"));
    });
    assert_eq!(contained, Some(false));
    let graceful = run_watched(|| {
        let mut s = fresh();
        assert!(s.execute("layer.delete", json!({"layer": 999999})).is_err());
    });
    assert_eq!(graceful, Some(true));
}
