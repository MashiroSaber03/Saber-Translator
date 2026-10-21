//! Edit › Preferences, Keyboard Shortcuts and Menus, Color Settings and the rest of the Edit
//! menu's dialogs, plus the shell side of preferences: loading and saving them through
//! [`crate::Services`], applying the interface theme, autosave and crash recovery, and the
//! history log.
//!
//! The values live in the engine ([`photocraft_engine::prefs::Preferences`], so agents read and
//! change them with `prefs.get` / `prefs.set`); this module only edits a working copy in a dialog
//! and commits it with those commands on Apply or OK.

use std::collections::{BTreeMap, HashMap, VecDeque};

use egui::{Color32, RichText, Sense, vec2};
use photocraft_doc::DocId;
use photocraft_engine::jobs::{JobEvent, JobId, JobOutcome, Started};
#[cfg(test)]
use photocraft_engine::prefs::Theme;
use photocraft_engine::prefs::{self, AppearanceMode, DarkTheme, LightTheme, SECTIONS, Theme as SavedTheme};
use photocraft_engine::snap::{SnapLine, SnapTargets};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;
use crate::theme::{ThemeKind, Tokens};

/// Shell runtime state for preferences, autosave and snapping (not serialised).
#[derive(Default)]
pub struct Runtime {
    loaded: bool,
    /// Preferences revision last written to storage (or loaded from it).
    saved_rev: u64,
    /// This instance's preferences as of `saved_rev`: the base of the three-way merge on save.
    saved_value: Option<Value>,
    save_retry: SaveRetry,
    next_autosave_ms: f64,
    /// Last revision whose recovery file was actually written successfully.
    autosaved: HashMap<DocId, u64>,
    /// Queued writes must not masquerade as durable snapshots.
    autosave_pending: HashMap<DocId, u64>,
    recovery_started: bool,
    recovery_queue: VecDeque<crate::Recoverable>,
    recovery_pending: Option<(JobId, String)>,
    log_len: usize,
    /// Snapping state of the drag in progress (see `snap_ui`).
    pub(crate) snap: Option<crate::snap_ui::ActiveSnap>,
    pub(crate) snap_lines: Vec<SnapLine>,
    pub(crate) guide_targets: Option<((DocId, u64), SnapTargets)>,
    /// Checkerboard colours the cached CPU checker texture was built with.
    pub(crate) checker_key: Option<[[u8; 3]; 2]>,
    /// Style last sent to the GPU canvas.
    pub(crate) gpu_style: Option<crate::gpu_canvas::CanvasStyle>,
    /// `unitsAndRulers.showRulers` as last applied to View › Rulers.
    show_rulers: Option<bool>,
    /// Tool options as last remembered in the preferences (see `tool_memory`).
    pub(crate) tool_options: Option<crate::state::ToolOptions>,
}

/// The GPU canvas colours from Preferences › Transparency & Gamut.
pub fn canvas_style(app: &PhotocraftApp) -> crate::gpu_canvas::CanvasStyle {
    let t = &app.session.prefs().transparency_and_gamut;
    let [l, d] = t.colors();
    let f = |c: [u8; 3]| c.map(|v| v as f32 / 255.0);
    let g = prefs::parse_hex(&t.gamut_warning_color).unwrap_or([128; 3]);
    crate::gpu_canvas::CanvasStyle {
        checker_square: t.square().unwrap_or(0.0),
        checker_light: f(l),
        checker_dark: f(d),
        gamut_color: f(g),
        gamut_opacity: t.gamut_warning_opacity as f32 / 100.0,
    }
}

/// Pasteboard colour from Preferences › Interface (`None` = the theme's default canvas).
pub fn pasteboard_color(app: &PhotocraftApp) -> Option<Color32> {
    use prefs::CanvasColor;
    let i = &app.session.prefs().interface;
    Some(match i.canvas_color {
        CanvasColor::Default => return None,
        CanvasColor::Black => Color32::BLACK,
        CanvasColor::DarkGray => Color32::from_gray(40),
        CanvasColor::MediumGray => Color32::from_gray(83),
        CanvasColor::LightGray => Color32::from_gray(184),
        CanvasColor::Custom => color_of(&i.canvas_custom_color),
    })
}

fn selected_theme(interface: &prefs::Interface, system: Option<egui::Theme>) -> ThemeKind {
    let light = match interface.appearance_mode {
        AppearanceMode::Light => true,
        AppearanceMode::Dark => false,
        AppearanceMode::Auto => system == Some(egui::Theme::Light),
    };
    if light {
        match interface.light_theme {
            LightTheme::StudioLight => ThemeKind::StudioLight,
            LightTheme::ProLightGray => ThemeKind::ProLightGray,
            LightTheme::ProLight => ThemeKind::ProLight,
            LightTheme::Classic => ThemeKind::Classic,
            LightTheme::Adwaita => ThemeKind::Adwaita,
            LightTheme::BreezeLight => ThemeKind::BreezeLight,
            LightTheme::SoftLight => ThemeKind::SoftLight,
        }
    } else {
        match interface.dark_theme {
            DarkTheme::Pro => ThemeKind::Pro,
            DarkTheme::ProMedium => ThemeKind::ProMedium,
            DarkTheme::Studio => ThemeKind::Studio,
            DarkTheme::SolarizedDark => ThemeKind::SolarizedDark,
            DarkTheme::AdwaitaDark => ThemeKind::AdwaitaDark,
            DarkTheme::BreezeDark => ThemeKind::BreezeDark,
            DarkTheme::SoftDark => ThemeKind::SoftDark,
        }
    }
}

pub(crate) fn cycle_appearance(app: &mut PhotocraftApp, ctx: &egui::Context) {
    let next = match app.session.prefs().interface.appearance_mode {
        AppearanceMode::Auto => AppearanceMode::Light,
        AppearanceMode::Light => AppearanceMode::Dark,
        AppearanceMode::Dark => AppearanceMode::Auto,
    };
    if let Err(e) = app.run("prefs.set", json!({"path": "interface.appearanceMode", "value": next.name()})) {
        app.ui.status = e;
        return;
    }
    let selected = selected_theme(&app.session.prefs().interface, system_theme(app, ctx));
    if app.ui.theme != selected {
        app.apply_theme(ctx, selected);
    }
}

const LIGHT_THEME_OPTIONS: &[(&str, ThemeKind)] = &[
    ("proLight", ThemeKind::ProLight),
    ("proLightGray", ThemeKind::ProLightGray),
    ("studioLight", ThemeKind::StudioLight),
    ("classic", ThemeKind::Classic),
    ("adwaita", ThemeKind::Adwaita),
    ("breezeLight", ThemeKind::BreezeLight),
    ("softLight", ThemeKind::SoftLight),
];
const DARK_THEME_OPTIONS: &[(&str, ThemeKind)] = &[
    ("proMedium", ThemeKind::ProMedium),
    ("pro", ThemeKind::Pro),
    ("studio", ThemeKind::Studio),
    ("solarizedDark", ThemeKind::SolarizedDark),
    ("adwaitaDark", ThemeKind::AdwaitaDark),
    ("breezeDark", ThemeKind::BreezeDark),
    ("softDark", ThemeKind::SoftDark),
];

fn favorite_theme_groups(favorites: &[SavedTheme]) -> (Vec<ThemeKind>, Vec<ThemeKind>) {
    let group = |options: &[(&str, ThemeKind)]| {
        options.iter().filter_map(|(id, kind)| SavedTheme::parse(id).filter(|theme| favorites.contains(theme)).map(|_| *kind)).collect()
    };
    (group(LIGHT_THEME_OPTIONS), group(DARK_THEME_OPTIONS))
}

/// Right-click menu on the top-right appearance button. A plain click still cycles modes.
pub(crate) fn appearance_context_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    if ui.button(tl!("Appearance Settings…")).clicked() {
        open_preferences(app, "interface");
        ui.close();
    }
    let (light, dark) = favorite_theme_groups(&app.session.prefs().interface.favorite_themes);
    let has_light = !light.is_empty();
    if !light.is_empty() || !dark.is_empty() {
        ui.separator();
    }
    for kind in light {
        if ui.selectable_label(app.ui.theme == kind, tl!(kind.label())).clicked() {
            app.set_theme(ui.ctx(), kind);
            ui.close();
        }
    }
    if has_light && !dark.is_empty() {
        ui.separator();
    }
    for kind in dark {
        if ui.selectable_label(app.ui.theme == kind, tl!(kind.label())).clicked() {
            app.set_theme(ui.ctx(), kind);
            ui.close();
        }
    }
}

fn system_theme(app: &PhotocraftApp, ctx: &egui::Context) -> Option<egui::Theme> {
    app.services.system_theme.as_ref().and_then(|read| read(ctx)).or_else(|| ctx.system_theme())
}

/// View › Rulers follows `unitsAndRulers.showRulers` when the preference changes (at launch, from
/// `prefs.set` or a reset), so documents open with the rulers as last set.
fn sync_rulers(app: &mut PhotocraftApp) {
    let want = app.session.prefs().units_and_rulers.show_rulers;
    if app.prefs_rt.show_rulers != Some(want) {
        app.ui.extras.rulers = want;
        app.prefs_rt.show_rulers = Some(want);
    }
}

/// View › Rulers was toggled: remember it for the next launch (Photoshop keeps the last state).
pub(crate) fn remember_rulers(app: &mut PhotocraftApp, on: bool) {
    if app.session.prefs().units_and_rulers.show_rulers != on {
        app.session.prefs.edit(|p| p.units_and_rulers.show_rulers = on);
    }
    app.prefs_rt.show_rulers = Some(on);
}

// ------------------------------------------------------------------ lifecycle

/// Load saved preferences once, when the app is created. Recovery starts during frame upkeep.
pub fn load(app: &mut PhotocraftApp) {
    if app.prefs_rt.loaded {
        return;
    }
    app.prefs_rt.loaded = true;
    if let Some(text) = app.services.load_prefs.as_mut().and_then(|f| f())
        && let Err(e) = app.session.load_prefs_json(&text)
    {
        let template = "Preferences were reset: {error}";
        app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
    }
    crate::dock::restore(app);
    crate::brush_picker::restore(app);
    crate::last_brush::restore(app);
    sync_rulers(app);
    crate::tool_memory::restore(app);
    crate::view_cmds::restore(app);
    app.sync_recent();
    app.prefs_rt.saved_rev = app.session.prefs.rev();
    app.prefs_rt.saved_value = Some(app.session.prefs_value());
}

/// Discover recovery entries after construction, then decode one at a time without blocking
/// the window. Only the completed job's apply step touches the session.
fn recovery(app: &mut PhotocraftApp) {
    if !app.prefs_rt.recovery_started {
        app.prefs_rt.recovery_started = true;
        if app.session.prefs().file_handling.recover_on_launch
            && let Some(recover) = app.services.recover.as_mut()
        {
            app.prefs_rt.recovery_queue.extend(recover());
        }
    }
    start_next_recovery(app);
}

fn start_next_recovery(app: &mut PhotocraftApp) {
    while app.prefs_rt.recovery_pending.is_none() {
        let Some(r) = app.prefs_rt.recovery_queue.pop_front() else { return };
        let label = format!("Recovering {}", r.name);
        let started = app.session.start_job(
            "file.recover",
            json!({}),
            &label,
            false,
            move |ctx| {
                ctx.check()?;
                ctx.progress(0.0, "Loading recovery data");
                let doc = (r.load)().map_err(photocraft_engine::EngineError::Other)?;
                ctx.check()?;
                Ok(doc)
            },
            move |s, doc| {
                let active = s.active().map(|st| st.doc.id);
                let first = s.documents().is_empty();
                s.add_document(doc, r.path);
                let st = s.active_mut().ok_or(photocraft_engine::EngineError::NoDocument)?;
                st.saved_revision = 0;
                let (id, revision) = (st.doc.id, st.revision);
                if let Some(index) = active.and_then(|id| s.documents().iter().position(|st| st.doc.id == id)) {
                    s.set_active(index);
                }
                Ok(json!({"documentId": id.0, "revision": revision, "firstDocument": first}))
            },
        );
        match started {
            Ok(Started::Job(job)) => app.prefs_rt.recovery_pending = Some((job, r.key)),
            Ok(Started::Done(v)) => finish_recovery(app, &r.key, &v),
            Err(e) => crate::notices::error(app, format!("{label}: {e}")),
        }
    }
}

fn finish_recovery(app: &mut PhotocraftApp, key: &str, v: &Value) {
    let recovered = v
        .get("documentId")
        .and_then(Value::as_u64)
        .zip(v.get("revision").and_then(Value::as_u64))
        .and_then(|(id, revision)| app.session.documents().iter().find(|st| st.doc.id.0 == id).map(|st| (st.doc.id, revision, st.doc.name.clone())));
    let Some((id, revision, name)) = recovered else {
        crate::notices::error(app, "The recovered document is no longer open".into());
        return;
    };
    // Track the admitted snapshot's revision, even if another job has already edited it.
    app.prefs_rt.autosaved.insert(id, revision);
    if let Some(adopt) = app.services.adopt_autosave.as_mut() {
        adopt(id.0, key);
    }
    if v.get("firstDocument").and_then(Value::as_bool) == Some(true) {
        app.ui.chrome.home = None;
    }
    app.sync_views();
    let template = "Recovered {name}";
    app.ui.status = crate::i18n::fmt(tl!(template), &[("name", &name)]);
    app.ui.status_error = false;
}

/// Recovery uses the existing job polling/progress/cancel path. Its successful result adopts
/// the original entry on the UI thread, using identity rather than a mutable tab position.
pub(crate) fn on_recovery_event(app: &mut PhotocraftApp, e: &JobEvent) -> bool {
    if !app.prefs_rt.recovery_pending.as_ref().is_some_and(|(job, _)| *job == e.id) {
        return false;
    }
    let Some((_, key)) = app.prefs_rt.recovery_pending.take() else { return false };
    match &e.outcome {
        JobOutcome::Done(v) => finish_recovery(app, &key, v),
        JobOutcome::Failed(err) => crate::notices::error(app, format!("{}: {err}", e.label)),
        JobOutcome::Cancelled => {
            // Native bundle decoding cannot stop midway. Cancel admission and the rest of the
            // batch so another large decoder isn't started while this worker finishes.
            app.prefs_rt.recovery_queue.clear();
            app.ui.status = crate::i18n::fmt(tl!("Cancelled {label}"), &[("label", tl!(&e.label))]);
            app.ui.status_error = false;
        }
    }
    start_next_recovery(app);
    true
}

/// Attach the brush preset store once its background load finishes, and surface write
/// failures in the status bar.
fn presets_store(app: &mut PhotocraftApp) {
    if let Some(rx) = &app.services.preset_store {
        match rx.try_recv() {
            Ok(opened) => {
                app.services.preset_store = None;
                let warnings = app.session.attach_preset_store(opened);
                if let Some(w) = warnings.first() {
                    app.ui.status = if warnings.len() == 1 {
                        w.clone()
                    } else {
                        crate::i18n::fmt(
                            &crate::i18n::trn(
                                crate::i18n::current(),
                                (warnings.len() - 1) as u64,
                                "{warning} (+{n} more brush preset warning)",
                                "{warning} (+{n} more brush preset warnings)",
                            ),
                            &[("warning", w)],
                        )
                    };
                }
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => app.services.preset_store = None,
        }
    }
    if let Some(w) = app.session.preset_store.as_mut().map(|s| s.take_warnings()).and_then(|w| w.into_iter().next()) {
        app.ui.status = w;
    }
}

/// Resolve an absolute pixels-per-point preference without multiplying the OS DPI twice.
/// Monitor dimensions are physical pixels, independent of window size and UI zoom.
fn display_scale(pref: prefs::UiScale, native: Option<f32>) -> f32 {
    let native = native.filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0);
    match pref {
        // The system's scale, fractional ones included (125%, 150%, 175%: #1072), and 100% on a
        // large 4K display run unscaled (#1278); 200% there is a fixed choice.
        prefs::UiScale::Auto => native,
        fixed => fixed.name().parse::<f32>().map_or(1.0, |pct| pct / 100.0),
    }
}

fn sync_display_scale(app: &PhotocraftApp, ctx: &egui::Context) {
    let native = ctx.native_pixels_per_point();
    let native_scale = native.filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0);
    let scale = display_scale(app.session.prefs().interface.ui_scale, native);
    ctx.set_zoom_factor(scale / native_scale);
}

/// Interface › Show Tooltips and Tools › Show Tooltips (either one off hides them): egui never
/// shows a tooltip whose delay is infinite. Re-checked every frame because a theme change
/// rebuilds the style; that's one style read, and a write only when it differs.
fn sync_tooltips(app: &PhotocraftApp, ctx: &egui::Context) {
    let p = app.session.prefs();
    let delay = if p.interface.show_tooltips && p.tools.show_tooltips { crate::theme::TOOLTIP_DELAY } else { f32::INFINITY };
    if ctx.global_style().interaction.tooltip_delay != delay {
        ctx.global_style_mut(|s| s.interaction.tooltip_delay = delay);
    }
}

/// Interface › Enable Animations: off, egui's fades and slides take no time and scroll areas jump
/// straight to their target (#3273). Re-checked every frame because a theme change rebuilds the
/// style; both the dark and light styles are set, so a system appearance switch keeps it.
fn sync_animations(app: &PhotocraftApp, ctx: &egui::Context) {
    let on = app.session.prefs().interface.enable_animations;
    let time = if on { egui::Style::default().animation_time } else { 0.0 };
    if ctx.global_style().animation_time != time {
        let scroll = if on { egui::style::ScrollAnimation::default() } else { egui::style::ScrollAnimation::none() };
        ctx.all_styles_mut(|s| {
            s.animation_time = time;
            s.scroll_animation = scroll;
        });
    }
}

/// Resolve the saved appearance mode without changing it or either theme slot.
pub(crate) fn sync_appearance(app: &mut PhotocraftApp, ctx: &egui::Context) {
    // A restored or embedding-supplied pin must not suppress native appearance events.
    ctx.set_theme(egui::ThemePreference::System);
    // The native service also covers Wayland sessions where winit has no system theme.
    let selected = selected_theme(&app.session.prefs().interface, system_theme(app, ctx));
    if app.ui.theme != selected {
        app.apply_theme(ctx, selected);
    }
}

/// Per-frame upkeep: theme sync, persistence, autosave and the history log.
pub fn tick(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if !app.prefs_rt.loaded {
        load(app);
    }
    recovery(app);
    sync_rulers(app);
    sync_display_scale(app, ctx);
    sync_appearance(app, ctx);
    crate::theme::set_ui_font_size(ctx, app.session.prefs().interface.ui_font_size);
    presets_store(app);
    sync_tooltips(app, ctx);
    sync_animations(app, ctx);
    app.sync_recent();
    if let Some(wait) = persist(app, ctx.input(|i| i.time)) {
        ctx.request_repaint_after(std::time::Duration::try_from_secs_f64(wait).unwrap_or_default());
    }
    let style = canvas_style(app);
    if let Some(gpu) = app.gpu.as_ref()
        && app.prefs_rt.gpu_style != Some(style)
    {
        gpu.set_style(style);
        app.prefs_rt.gpu_style = Some(style);
    }
    autosave(app);
    #[cfg(not(target_arch = "wasm32"))]
    link_updates(app, ctx);
    history_log(app);
}

/// First retry delay after a failed preferences write, in seconds; it doubles with every further
/// failure up to [`SAVE_RETRY_MAX_S`]. A new change retries sooner, but no sooner than this.
const SAVE_RETRY_S: f64 = 2.0;
const SAVE_RETRY_MAX_S: f64 = 30.0;
/// Failed writes in a row before a notice says preferences aren't being saved.
const SAVE_NOTICE_AFTER: u32 = 3;

/// Retry state of a failed preferences write (times are egui input time, in seconds).
#[derive(Default)]
struct SaveRetry {
    failures: u32,
    /// When the next retry is due.
    at: f64,
    /// When the last write was tried, and the revision it carried.
    tried_at: f64,
    tried_rev: u64,
    /// The "not saved" notice, while it's shown.
    notice: Option<u64>,
}

/// Save changed preferences. A failed write stays pending and is retried with a backoff (a newer
/// change retries sooner); the status bar reports the first failure and a notice a persistent
/// one. Returns the seconds until the next retry while one is pending.
fn persist(app: &mut PhotocraftApp, now: f64) -> Option<f64> {
    let rev = app.session.prefs.rev();
    if rev == app.prefs_rt.saved_rev {
        return None;
    }
    let r = &app.prefs_rt.save_retry;
    if r.failures > 0 {
        let due = if rev == r.tried_rev { r.at } else { r.at.min(r.tried_at + SAVE_RETRY_S) };
        if now < due {
            return Some(due - now);
        }
    }
    let ours = app.session.prefs_value();
    // A change that was undone (or set a value to what it already was) leaves nothing to write.
    let written = if app.prefs_rt.saved_value.as_ref() == Some(&ours) { Ok(()) } else { write_prefs(app, &ours) };
    match written {
        Ok(()) => {
            app.prefs_rt.saved_rev = rev;
            app.prefs_rt.saved_value = Some(ours);
            let r = std::mem::take(&mut app.prefs_rt.save_retry);
            if r.failures > 0 {
                let message = "Preferences saved";
                app.ui.status = tl!(message).into();
                app.ui.status_error = false;
                if let Some(id) = r.notice {
                    app.ui.notices.retain(|n| n.id != id);
                }
            }
            None
        }
        Err(e) => {
            let r = &mut app.prefs_rt.save_retry;
            r.failures = r.failures.saturating_add(1);
            (r.tried_at, r.tried_rev) = (now, rev);
            let delay = (SAVE_RETRY_S * 2f64.powi(r.failures.min(16) as i32 - 1)).min(SAVE_RETRY_MAX_S);
            r.at = now + delay;
            let failures = r.failures;
            if failures == 1 {
                let template = "Couldn't save preferences: {error}. Retrying…";
                app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
                app.ui.status_error = true;
            } else if failures == SAVE_NOTICE_AFTER {
                let lines = vec![e, "PhotoCraft keeps retrying; until a save succeeds, preference changes are lost when it closes.".into()];
                let id = crate::notices::post(app, "Preferences can't be saved", lines, true, None);
                app.prefs_rt.save_retry.notice = Some(id);
            }
            Some(delay)
        }
    }
}

/// Write `ours` over the stored preferences, keeping what another instance (a second browser
/// tab, another app window) changed there since this one last loaded or saved them.
fn write_prefs(app: &mut PhotocraftApp, ours: &Value) -> Result<(), String> {
    let Some(save) = app.services.save_prefs.as_mut() else { return Ok(()) };
    let stored = app.services.load_prefs.as_mut().and_then(|load| load()).and_then(|t| serde_json::from_str::<Value>(&t).ok());
    let text = match (stored, &app.prefs_rt.saved_value) {
        (Some(mut theirs @ Value::Object(_)), Some(base)) => {
            merge_changes(base, ours, &mut theirs);
            serde_json::to_string_pretty(&theirs)
        }
        _ => serde_json::to_string_pretty(ours),
    }
    .map_err(|e| e.to_string())?;
    save(&text)
}

/// Three-way merge of preference trees: whatever `ours` changed since `base` goes over `theirs`
/// (what storage holds now) and the rest of `theirs` is kept. Objects merge key by key; any
/// other value, arrays included, is replaced whole.
fn merge_changes(base: &Value, ours: &Value, theirs: &mut Value) {
    if ours == base {
        return;
    }
    match (base, ours, theirs) {
        (Value::Object(b), Value::Object(o), Value::Object(t)) => {
            for (k, ov) in o {
                match (b.get(k), t.get_mut(k)) {
                    (Some(bv), Some(tv)) => merge_changes(bv, ov, tv),
                    // Removed there, unchanged here: it stays removed.
                    (Some(bv), None) if bv == ov => {}
                    _ => {
                        t.insert(k.clone(), ov.clone());
                    }
                }
            }
            for k in b.keys().filter(|k| !o.contains_key(*k)) {
                t.remove(k);
            }
        }
        (_, _, theirs) => *theirs = ours.clone(),
    }
}

/// Write the preferences now, through the same merge as [`persist`], for recovery choices that
/// need the result immediately. The pending revision stays unsaved when the write fails.
pub(crate) fn save_preferences(app: &mut PhotocraftApp) -> Result<(), String> {
    let rev = app.session.prefs.rev();
    let ours = app.session.prefs_value();
    write_prefs(app, &ours)?;
    app.prefs_rt.saved_rev = rev;
    app.prefs_rt.saved_value = Some(ours);
    if let Some(id) = std::mem::take(&mut app.prefs_rt.save_retry).notice {
        app.ui.notices.retain(|n| n.id != id);
    }
    Ok(())
}

/// Linked smart objects follow their files while the document is open, as in Photoshop: every two
/// seconds, and as soon as the window comes back to the front (after saving in another app), stamp
/// the active document's linked files on a worker thread and re-render the layers of the ones that
/// changed ([`photocraft_engine::link_watch`]).
#[cfg(not(target_arch = "wasm32"))]
fn link_updates(app: &mut PhotocraftApp, ctx: &egui::Context) {
    match app.session.poll_link_scan() {
        Some(Ok(r)) => {
            if let Some(n) = r["updated"].as_array().map(Vec::len).filter(|n| *n > 0) {
                app.ui.status = crate::i18n::trn(
                    crate::i18n::current(),
                    n as u64,
                    "Updated {n} linked smart object changed on disk",
                    "Updated {n} linked smart objects changed on disk",
                );
            }
        }
        Some(Err(e)) => app.ui.status = e.to_string(),
        None => {}
    }
    let (last_key, focus_key) = (egui::Id::new("linkWatch.lastScan"), egui::Id::new("linkWatch.focused"));
    let now = ctx.input(|i| i.time);
    let focused = ctx.input(|i| i.focused);
    let was_focused: bool = ctx.data(|d| d.get_temp(focus_key)).unwrap_or(focused);
    ctx.data_mut(|d| d.insert_temp(focus_key, focused));
    let has_links = app.session.active().is_some_and(|d| !photocraft_engine::link_watch::linked_files(&d.doc).is_empty());
    if !has_links {
        return;
    }
    let last: f64 = ctx.data(|d| d.get_temp(last_key)).unwrap_or(f64::NEG_INFINITY);
    if now - last >= 2.0 || (focused && !was_focused) {
        ctx.data_mut(|d| d.insert_temp(last_key, now));
        app.session.start_link_scan();
    }
    let wait = if app.session.link_scan.is_some() { 100 } else { 2000 };
    ctx.request_repaint_after(std::time::Duration::from_millis(wait));
}

/// Background autosave of documents with unsaved changes every N minutes (File Handling).
fn autosave(app: &mut PhotocraftApp) {
    let fh = &app.session.prefs().file_handling;
    let (on, minutes) = (fh.autosave, fh.autosave_minutes.max(1));
    if app.services.autosave.is_none() {
        return;
    }
    // A successful queue operation is not a successful disk write. Poll each frame,
    // including frames before the next autosave interval, so failures are visible.
    if let Some(results) = app.services.autosave_results.as_mut() {
        for (raw_id, revision, result) in results() {
            let id = DocId(raw_id);
            if app.prefs_rt.autosave_pending.get(&id) != Some(&revision) {
                continue;
            }
            app.prefs_rt.autosave_pending.remove(&id);
            match result {
                Ok(()) => {
                    app.prefs_rt.autosaved.insert(id, revision);
                }
                Err(e) => {
                    let template = "Autosave failed: {error}";
                    app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
                    app.ui.status_error = true;
                }
            }
        }
    }
    let now = crate::gpu_canvas::now_ms();
    // Saved or closed documents drop their recovery data.
    let live: HashMap<DocId, bool> = app.session.documents().iter().map(|d| (d.doc.id, d.is_dirty())).collect();
    let mut stale: Vec<DocId> = app.prefs_rt.autosaved.keys().filter(|id| live.get(id) != Some(&true)).copied().collect();
    stale.extend(app.prefs_rt.autosave_pending.keys().filter(|id| live.get(*id) != Some(&true) && !app.prefs_rt.autosaved.contains_key(*id)).copied());
    for id in stale {
        app.prefs_rt.autosaved.remove(&id);
        app.prefs_rt.autosave_pending.remove(&id);
        if let Some(d) = app.services.discard_autosave.as_mut() {
            d(id.0);
        }
    }
    if !on {
        return;
    }
    let interval = minutes as f64 * 60_000.0;
    if app.prefs_rt.next_autosave_ms == 0.0 || app.prefs_rt.next_autosave_ms > now + interval {
        app.prefs_rt.next_autosave_ms = now + interval;
        return;
    }
    if now < app.prefs_rt.next_autosave_ms {
        return;
    }
    app.prefs_rt.next_autosave_ms = now + interval;
    let jobs: Vec<_> = app
        .session
        .documents()
        .iter()
        .filter(|d| d.is_dirty() && app.prefs_rt.autosaved.get(&d.doc.id) != Some(&d.revision) && !app.prefs_rt.autosave_pending.contains_key(&d.doc.id))
        .map(|d| (d.doc.clone(), d.revision, d.path.clone()))
        .collect();
    for (doc, rev, path) in jobs {
        if let Some(save) = app.services.autosave.as_mut() {
            match save(&doc, rev, path.as_deref()) {
                Ok(()) => {
                    if app.services.autosave_results.is_some() {
                        app.prefs_rt.autosave_pending.insert(doc.id, rev);
                    } else {
                        // Synchronous or mocked services report completion on return.
                        app.prefs_rt.autosaved.insert(doc.id, rev);
                    }
                }
                Err(e) => {
                    let template = "Autosave failed: {error}";
                    app.ui.status = crate::i18n::fmt(tl!(template), &[("error", &e)]);
                    app.ui.status_error = true;
                }
            }
        }
    }
}

/// Force the autosave timer to fire on the next tick (tests, `prefs` changes).
pub fn autosave_now(app: &mut PhotocraftApp) {
    app.prefs_rt.next_autosave_ms = f64::MIN_POSITIVE;
}

/// History Log preference: append executed commands to a text file.
fn history_log(app: &mut PhotocraftApp) {
    let n = app.session.journal.len();
    if n <= app.prefs_rt.log_len {
        app.prefs_rt.log_len = n;
        return;
    }
    let hl = app.session.prefs().history_log.clone();
    let start = app.prefs_rt.log_len;
    app.prefs_rt.log_len = n;
    if !hl.enabled || hl.file_path.is_empty() || hl.destination == prefs::LogDestination::Metadata {
        return;
    }
    let Some(append) = app.services.append_text.as_mut() else { return };
    let mut text = String::new();
    for (id, params) in &app.session.journal[start..] {
        let label = photocraft_engine::commands::find(id).map_or(id.as_str(), |c| c.label).trim_end_matches('…');
        match hl.detail {
            prefs::LogDetail::SessionsOnly => {
                if matches!(id.as_str(), "file.new" | "file.open" | "file.close" | "file.openAs") {
                    text.push_str(&format!("{label}\n"));
                }
            }
            prefs::LogDetail::Concise => text.push_str(&format!("{label}\n")),
            prefs::LogDetail::Detailed => text.push_str(&format!("{label}  {id} {params}\n")),
        }
    }
    if !text.is_empty() {
        let _ = append(&hl.file_path, &text);
    }
}

// ------------------------------------------------------------------ shortcuts

pub use crate::shortcuts::{default_shortcut, effective_shortcut};

/// Every shortcut-bearing command (engine registry plus shell commands): (id, label, menu path,
/// default shortcut), menu items first in Photoshop order.
pub fn shortcut_items(app: &PhotocraftApp) -> Vec<(String, String, Vec<String>, Option<String>)> {
    let mut out: Vec<(String, String, Vec<String>, Option<String>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for it in crate::menus::menu_items(app) {
        if it.label == "---" || !seen.insert(it.id.clone()) {
            continue;
        }
        let default = default_shortcut(&it.id).or(it.shortcut.clone().filter(|_| !app.session.prefs().shortcuts.contains_key(&it.id)));
        out.push((it.id, it.label, it.path, default));
    }
    // Commands without a menu item: the colour and fill keys (D, X, ⌥⌫…) sit under Tools, as
    // Photoshop lists its colour keys in the Tools shortcut set, after the rest; then the held
    // temporary tools (#249).
    let tools_key = |id: &str| id.starts_with("tools.") || photocraft_engine::fill_key_cmds::IDS.contains(&id);
    let mut tools = Vec::new();
    let mut layer = Vec::new();
    for c in photocraft_engine::command_specs() {
        if seen.insert(c.id.to_string()) && c.shortcut.is_some() {
            let item = (c.id.to_string(), c.label.to_string(), c.menu.iter().map(|s| s.to_string()).collect::<Vec<_>>(), c.shortcut.map(Into::into));
            if c.menu.is_empty() && tools_key(c.id) {
                tools.push((item.0, item.1, vec!["Tools".to_string()], item.3));
            } else if c.menu.is_empty() && c.id.starts_with("layer.") {
                // Stamp Visible / Stamp Down (#217): no menu item in Photoshop; listed at the end
                // of the Layer section.
                layer.push((item.0, item.1, vec!["Layer".to_string()], item.3));
            } else {
                out.push(item);
            }
        }
    }
    let at = out.iter().rposition(|i| i.2.first().map(String::as_str) == Some("Layer")).map_or(out.len(), |i| i + 1);
    out.splice(at..at, layer);
    out.extend(tools);
    out.extend(
        crate::state::Tool::ALL
            .into_iter()
            .map(|tool| (crate::tool_shortcuts::id(tool), tool.label().to_string(), vec!["Tools".into()], crate::tool_shortcuts::default(tool))),
    );
    for (id, label, def) in prefs::TEMPORARY_TOOLS {
        if seen.insert(id.to_string()) {
            out.push((id.to_string(), label.to_string(), vec!["Tools".into(), "Temporary".into()], Some(def.to_string())));
        }
    }
    out
}

/// Text of a key press for the shortcut editor (`Cmd+Shift+K`), `None` for bare modifiers.
pub fn shortcut_text(key: egui::Key, m: egui::Modifiers) -> Option<String> {
    if is_modifier_key(key) {
        return None;
    }
    let mut parts = Vec::new();
    if m.command || m.mac_cmd {
        parts.push(tl!("Cmd").to_string());
    }
    if m.ctrl && !m.command {
        parts.push(tl!("Ctrl").into());
    }
    if m.alt {
        parts.push(tl!("Alt").into());
    }
    if m.shift {
        parts.push(tl!("Shift").into());
    }
    let name = match key {
        egui::Key::Equals => "=",
        egui::Key::Minus => "-",
        egui::Key::OpenBracket => "[",
        egui::Key::CloseBracket => "]",
        egui::Key::Semicolon => ";",
        egui::Key::Quote => "'",
        egui::Key::Plus => "",
        k => k.name(),
    };
    parts.push(name.to_string());
    prefs::normalize_shortcut(&parts.join("+"))
}

/// A modifier key on its own. egui reports Ctrl, Shift, Alt and ⌘ presses as key events too
/// (`ControlLeft`…), before the key pressed with them: they are never a shortcut's key, so
/// shortcut capture waits for the real key (#292: Ctrl+F was recorded as "Ctrl+ControlLeft").
pub fn is_modifier_key(key: egui::Key) -> bool {
    use egui::Key::*;
    matches!(key, ShiftLeft | ShiftRight | ControlLeft | ControlRight | AltLeft | AltRight | SuperLeft | SuperRight)
}

// ------------------------------------------------------------------ menu routing

/// Shell front ends for Edit-menu commands invoked without parameters (dialogs, pickers).
/// `None` when `id` isn't handled here.
pub fn invoke(app: &mut PhotocraftApp, ctx: &egui::Context, id: &str, params: &Value) -> Option<Result<Value, String>> {
    let empty = params.as_object().is_none_or(|o| o.is_empty());
    if !empty {
        return None;
    }
    if let Some(section) = id.strip_prefix("edit.preferences.") {
        return Some(Ok(json!({"dialog": open_preferences(app, section)})));
    }
    let dialog = |d: Option<u64>| Some(d.map(|d| json!({"dialog": d})).ok_or_else(|| "not available".to_string()));
    match id {
        "edit.keyboardShortcuts" => Some(Ok(json!({"dialog": open_shortcuts(app, 0)}))),
        "edit.menus" => Some(Ok(json!({"dialog": open_shortcuts(app, 1)}))),
        "edit.toolbar" => Some(Ok(json!({"dialog": open_shortcuts(app, 2)}))),
        "edit.colorSettings" => {
            let d = crate::filter_dialog::open(app, "edit.colorSettings");
            let cur = serde_json::to_value(&app.session.color.settings).unwrap_or_default();
            // What Monitor Profile resolves to, so a fallback to sRGB is visible (#569): read the
            // displays again now, and the dialog refreshes this line as it draws.
            crate::monitor_status::read_now(app);
            let note = crate::monitor_status::note(app);
            if let Some(dm) = d.and_then(|d| app.ui.dialog_mut(d)) {
                for (k, v) in cur.as_object().into_iter().flatten() {
                    if dm.fields.contains_key(k) {
                        dm.fields.insert(k.clone(), v.clone());
                    }
                }
                dm.fields.insert("__note".into(), json!(note));
            }
            dialog(d)
        }
        "edit.fade" | "edit.findAndReplaceText" | "edit.defineBrushPreset" | "edit.defineCustomShape" | "edit.autoAlignLayers" | "edit.autoBlendLayers" => {
            if !app.session.is_enabled(id) {
                return Some(Err(photocraft_engine::commands::find(id)
                    .map_or("not available".into(), |c| format!("{} is not available right now", c.label.trim_end_matches('…')))));
            }
            let d = crate::filter_dialog::open(app, id);
            if let Some(dm) = d.and_then(|d| app.ui.dialog_mut(d)) {
                // Optional parameters the generic dialog can't express keep their defaults.
                for k in ["reference", "path"] {
                    dm.fields.remove(k);
                }
            }
            dialog(d)
        }
        "edit.contentAwareFill" => {
            if !app.session.is_enabled(id) {
                return Some(Err("Content-Aware Fill needs a selection on a pixel layer".into()));
            }
            let d = crate::filter_dialog::open(app, id);
            if let Some(dm) = d.and_then(|d| app.ui.dialog_mut(d)) {
                dm.fields.insert("__preview".into(), json!(true));
                for k in ["margin", "seed", "channel"] {
                    dm.fields.remove(k);
                }
            }
            dialog(d)
        }
        // Photoshop has no dialog: the menu opens the scale box (with params: the engine).
        crate::cas_transform::COMMAND => Some(crate::cas_transform::begin(app, ctx)),
        "edit.presets.presetManager" => {
            Some(Ok(json!({"dialog": open_kind(app, "presets", "Preset Manager", json!({"kind": "brushes", "selected": 0, "newName": ""}))})))
        }
        "edit.presets.exportImportPresets" => Some(Ok(
            json!({"dialog": open_kind(app, "presetsIO", "Export/Import Presets", json!({"action": "export", "brushes": true, "customShapes": true}))}),
        )),
        _ => None,
    }
}

fn open_kind(app: &mut PhotocraftApp, kind: &str, label: &str, fields: Value) -> u64 {
    let mut f = Map::new();
    f.insert("__prefsui".into(), json!(kind));
    f.insert("__label".into(), json!(label));
    if let Value::Object(m) = fields {
        f.extend(m);
    }
    app.ui.open_dialog(DialogKind::Command, f)
}

/// Is this a dialog rendered by this module?
pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key("__prefsui")
}

/// Only Preferences supports applying changes without closing the dialog.
pub fn is_preferences(fields: &Map<String, Value>) -> bool {
    fields.get("__prefsui").and_then(Value::as_str) == Some("prefs")
}

fn preference_values(p: &prefs::Preferences) -> Map<String, Value> {
    let values = p.to_json();
    SECTIONS.iter().filter_map(|(id, _)| Some((id.to_string(), values.get(id)?.clone()))).collect()
}

/// Compare the editable sections, excluding settings managed outside Preferences.
pub fn preferences_changed(app: &PhotocraftApp, fields: &Map<String, Value>) -> bool {
    is_preferences(fields) && fields.get("values").and_then(Value::as_object).is_some_and(|values| *values != preference_values(app.session.prefs()))
}

/// Commit the working copy and keep the current section open. Failure leaves the draft intact.
pub fn apply(app: &mut PhotocraftApp, id: u64) -> Result<Value, String> {
    let d = app.ui.dialogs.iter().find(|d| d.id == id).ok_or_else(|| format!("no dialog {id}"))?;
    if d.kind != DialogKind::Command || !is_preferences(&d.fields) {
        return Err("Apply is only available for Preferences".into());
    }
    let fields = d.fields.clone();
    let result = confirm(app, &fields)?;
    let values = Value::Object(preference_values(app.session.prefs()));
    if let Some(d) = app.ui.dialog_mut(id) {
        // Use the validated values as the next draft, including any normalisation by prefs.set.
        d.fields.insert("values".into(), values);
    }
    Ok(result)
}

/// Max dialog width for our dialogs.
pub fn width(fields: &Map<String, Value>, available: f32) -> Option<f32> {
    match fields.get("__prefsui").and_then(Value::as_str)? {
        "prefs" => Some(available.clamp(380.0, 1040.0)),
        "shortcuts" => Some(720.0),
        _ => Some(460.0),
    }
}

/// Open Edit › Preferences on `section`.
pub fn open_preferences(app: &mut PhotocraftApp, section: &str) -> u64 {
    let section = if SECTIONS.iter().any(|(id, _)| *id == section) { section } else { "general" };
    let working = preference_values(app.session.prefs());
    let order = field_order(app.session.prefs(), &working);
    open_kind(app, "prefs", "Preferences", json!({"section": section, "values": working, "__order": order}))
}

/// Each section's keys in declaration order (JSON objects sort their keys; the serialised text
/// keeps the struct's order, which groups related settings like Photoshop's dialog).
fn field_order(p: &prefs::Preferences, sections: &Map<String, Value>) -> Value {
    let text = serde_json::to_string(p).unwrap_or_default();
    let mut out = Map::new();
    for (id, v) in sections {
        let Some(start) = text.find(&format!("\"{id}\":{{")) else { continue };
        let body = &text[start..];
        let end = body.find('}').unwrap_or(body.len());
        let region = &body[..end];
        let mut keys: Vec<(usize, String)> =
            v.as_object().into_iter().flatten().map(|(k, _)| (region.find(&format!("\"{k}\":")).unwrap_or(usize::MAX), k.clone())).collect();
        keys.sort();
        out.insert(id.clone(), json!(keys.into_iter().map(|(_, k)| k).collect::<Vec<_>>()));
    }
    Value::Object(out)
}

/// Open Keyboard Shortcuts and Menus on a tab (0 shortcuts, 1 menus, 2 toolbar).
pub fn open_shortcuts(app: &mut PhotocraftApp, tab: u64) -> u64 {
    let p = app.session.prefs();
    let fields = json!({
        "tab": tab,
        "filter": "",
        "overrides": p.shortcuts,
        "hidden": p.menus.hidden,
        "colors": p.menus.colors,
        "toolbarHidden": p.toolbar.hidden,
        "selected": "",
        "capture": false,
        "message": "",
    });
    open_kind(app, "shortcuts", tl!("Keyboard Shortcuts and Menus"), fields)
}

/// Open the Embedded Profile Mismatch prompt for a file waiting to open (`open_prompt`): its
/// `message`, the answer selected first (`action`), and `marker`, the field that routes OK and
/// Cancel back to `open_prompt`.
pub fn open_mismatch(app: &mut PhotocraftApp, message: &str, action: &str, missing: bool, marker: &str) -> u64 {
    let mut fields = json!({"message": message, "action": action, "missing": missing});
    fields[marker] = json!(true);
    open_kind(app, "mismatch", tl!("Embedded Profile Mismatch"), fields)
}

// ------------------------------------------------------------------ bodies

/// Render one of our dialogs' bodies.
pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    match f.get("__prefsui").and_then(Value::as_str).unwrap_or("") {
        "prefs" => {
            f.insert("__gpuInfo".into(), json!(app.perf.gpu_info.lines()));
            let system = system_theme(app, ui.ctx());
            prefs_body(ui, f, system);
        }
        "shortcuts" => shortcuts_body(app, ui, f),
        "presets" => presets_body(app, ui, f),
        "presetsIO" => presets_io_body(ui, f),
        "mismatch" => mismatch_body(ui, f),
        _ => {}
    }
}

fn humanize(key: &str) -> String {
    match key {
        "appearanceMode" => return "Appearance Mode".into(),
        "darkTheme" => return "Dark Theme".into(),
        "lightTheme" => return "Light Theme".into(),
        _ => {}
    }
    // These controls appear only for WebP, so reuse the existing translated labels.
    if key == "webpLossless" {
        return "Lossless".into();
    }
    if key == "webpQuality" {
        return "Quality".into();
    }
    // The auto-hide notice settings read best with their own labels (Interface › Notification, #2022).
    if key == "notificationAutoHide" {
        return "Auto Hide Notifications".into();
    }
    if key == "notificationDurationSeconds" {
        return "Notification Duration (seconds)".into();
    }
    let mut s = String::new();
    for (i, ch) in key.chars().enumerate() {
        if i == 0 {
            s.extend(ch.to_uppercase());
        } else if ch.is_uppercase() {
            s.push(' ');
            s.extend(ch.to_lowercase());
        } else {
            s.push(ch);
        }
    }
    s.replace("Psd", "PSD").replace("Gpu", "GPU").replace("Ui ", "UI ").replace("Mb", "(MB)").replace("Exif", "EXIF").replace("Hud", "HUD")
}

fn choice_label(v: &str) -> String {
    match v {
        "cm" => "Centimeters".into(),
        "mm" => "Millimeters".into(),
        "75" | "80" | "85" | "90" | "95" | "100" | "125" | "150" | "175" | "200" | "250" | "300" => format!("{v}%"),
        "8" => "8 Bits/Channel".into(),
        "16" => "16 Bits/Channel".into(),
        "postScript" => "PostScript (72 points/inch)".into(),
        "traditional" => "Traditional (72.27 points/inch)".into(),
        "vulkan" => "Vulkan".into(),
        "dx12" => "DirectX 12".into(),
        "metal" => "Metal".into(),
        "gl" => "OpenGL".into(),
        "cpu" => "CPU (no GPU acceleration)".into(),
        v => humanize(v),
    }
}

/// An editable `#rrggbb` field beside a colour swatch (#668): typing a valid value sets the
/// colour; while it's being typed a partial value is kept, and it shows the colour otherwise.
fn hex_field(ui: &mut egui::Ui, key: &str, rgb: &mut [u8; 3]) {
    let id = egui::Id::new(("pref-hex", key));
    let shown = format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2]);
    let mut text = ui.data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| shown.clone());
    let r = crate::widgets::text_field_ex(ui, id, &mut text, "", 72.0, true, Some(7));
    if r.changed()
        && let Some(c) = prefs::parse_hex(&text)
    {
        *rgb = c;
    }
    if r.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, text));
    } else {
        ui.data_mut(|d| d.remove::<String>(id));
    }
}

fn color_of(s: &str) -> Color32 {
    prefs::parse_hex(s).map_or(Color32::GRAY, |c| Color32::from_rgb(c[0], c[1], c[2]))
}

/// Search the same visible, editable preference labels used in the dialog.
fn preference_matches(section: &str, key: &str, query: &str) -> bool {
    let label = humanize(key);
    format!("{section}.{key}").to_lowercase().contains(query) || label.to_lowercase().contains(query) || tl!(&label).to_lowercase().contains(query)
}

fn preference_sections(values: &Value, search: &str) -> Vec<(&'static str, &'static str)> {
    let query = search.trim().to_lowercase();
    SECTIONS
        .iter()
        .copied()
        .filter(|(id, title)| {
            query.is_empty()
                || title.to_lowercase().contains(&query)
                || tl!(title).to_lowercase().contains(&query)
                || values
                    .get(*id)
                    .and_then(Value::as_object)
                    .is_some_and(|fields| fields.keys().any(|key| !prefs::is_hidden(&format!("{id}.{key}")) && preference_matches(id, key, &query)))
        })
        .collect()
}

/// Preferences: section list on the left, the section's settings on the right.
fn prefs_body(ui: &mut egui::Ui, f: &mut Map<String, Value>, system: Option<egui::Theme>) {
    let t = Tokens::get(ui.ctx());
    let mut section = f.get("section").and_then(Value::as_str).unwrap_or("general").to_string();
    let mut values = f.get("values").cloned().unwrap_or(Value::Null);
    // The dialog follows the language being edited, so a change shows before OK.
    let lang = crate::i18n::Lang::from_pref(values.pointer("/interface/language").and_then(Value::as_str).unwrap_or("auto"));
    let mut search = f.get("__search").and_then(Value::as_str).unwrap_or("").to_string();
    crate::widgets::text_field(ui, ui.id().with("prefs-search"), &mut search, tl!("Search settings…"), ui.available_width(), false);
    let found = preference_sections(&values, &search);
    if !found.iter().any(|(id, _)| *id == section)
        && let Some((first, _)) = found.first()
    {
        section = (*first).to_string();
    }
    ui.add_space(6.0);
    ui.horizontal_top(|ui| {
        // Section list.
        ui.vertical(|ui| {
            ui.set_width(170.0);
            if found.is_empty() {
                ui.label(RichText::new(tl!("No matching settings")).color(t.text_dim));
            }
            for &(id, title) in &found {
                let sel = section == id;
                let empty = !has_visible_fields(&values, id);
                let (rect, resp) = ui.allocate_exact_size(vec2(170.0, 22.0), Sense::click());
                if sel {
                    ui.painter().rect_filled(rect, t.radius_sm, t.row_selected);
                } else if resp.hovered() {
                    ui.painter().rect_filled(rect, t.radius_sm, t.hover);
                }
                let color = if sel {
                    t.text
                } else if empty {
                    t.text_faint
                } else {
                    t.text_dim
                };
                // Translated section names can be longer than the column: elide them (the full
                // name is the tooltip) instead of drawing over the settings.
                let mut job = egui::text::LayoutJob::simple_singleline(tl!(title).to_string(), crate::theme::medium(12.5), color);
                job.wrap = egui::text::TextWrapping::truncate_at_width(rect.width() - 12.0);
                let galley = ui.painter().layout_job(job);
                let elided = galley.elided;
                ui.painter().galley(rect.left_center() + vec2(8.0, -galley.size().y / 2.0), galley, color);
                let resp = if elided { resp.on_hover_text(tl!(title)) } else { resp };
                if resp.clicked() {
                    section = id.to_string();
                }
            }
        });
        let content_height = (ui.ctx().content_rect().height() - 240.0).clamp(240.0, 680.0);
        crate::widgets::vline(ui, content_height);
        ui.vertical(|ui| {
            ui.set_width((ui.available_width() - 12.0).max(340.0));
            let title = SECTIONS.iter().find(|(id, _)| *id == section).map_or("General", |(_, t)| *t);
            ui.label(RichText::new(tl!(&title)).font(crate::theme::semibold(14.0)).color(t.text));
            let show_all = search.trim().is_empty()
                || title.to_lowercase().contains(&search.trim().to_lowercase())
                || tl!(&title).to_lowercase().contains(&search.trim().to_lowercase());
            let filter = if show_all { "" } else { search.trim() };
            ui.add_space(6.0);
            egui::ScrollArea::vertical().max_height(content_height - 30.0).id_salt("prefs-scroll").show(ui, |ui| {
                let order: Vec<String> =
                    f.get("__order").and_then(|o| o.get(&section)).and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
                if !has_visible_fields(&values, &section) {
                    ui.add_space(4.0);
                    ui.label(RichText::new(tl!("These settings aren't available in PhotoCraft yet.")).color(t.text_faint));
                } else if let Some(obj) = values.get_mut(&section).and_then(Value::as_object_mut) {
                    section_fields(ui, &section, obj, &order, lang, system, filter);
                    if section == "fileHandling" {
                        ui.label(
                            RichText::new(tl!("0 turns this threshold off; SVG groups nested deeper than 100 levels are always rasterized."))
                                .color(t.text_faint),
                        );
                    }
                    if section == "performance" {
                        gpu_status_rows(ui, f.get("__gpuInfo"), obj);
                    }
                    ui.add_space(8.0);
                }
                if search.trim().is_empty()
                    && has_visible_fields(&values, &section)
                    && crate::widgets::secondary_button(ui, tl!("Reset Section"), 110.0).clicked()
                    && let Some(def) = prefs::Preferences::default().get(&section)
                {
                    values[&section] = def;
                }
            });
        });
    });
    f.insert("section".into(), json!(section));
    f.insert("__search".into(), json!(search));
    f.insert("values".into(), values);
    // A colour swatch was clicked: open the Color Picker on that value (#2144).
    if let Some((path, name)) = ctx_take_pick(ui) {
        crate::color_picker_ui::request_field(f, &format!("/values/{}", path.replace('.', "/")), &crate::color_picker_ui::title_for(&name));
    }
}

/// Preferences › Performance: what the app renders with now, and a reset of the GPU backend
/// (a crashed start may have moved it to a safer choice).
fn gpu_status_rows(ui: &mut egui::Ui, info: Option<&Value>, obj: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(10.0);
    ui.label(RichText::new(tl!("Graphics")).font(crate::theme::semibold(12.5)).color(t.text));
    for line in info.and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str) {
        ui.label(RichText::new(line).color(t.text_dim));
    }
    ui.add_space(4.0);
    ui.label(RichText::new(tl!("Applies at next launch.")).color(t.text_faint));
    ui.collapsing(tl!("Advanced"), |ui| {
        ui.horizontal(|ui| {
            ui.label(tl!("GPU Backend"));
            let mut current = obj.get("gpuBackend").and_then(Value::as_str).unwrap_or("auto").to_string();
            let options = prefs::choices("performance.gpuBackend").unwrap_or(&[]);
            let labels: Vec<String> = options.iter().map(|o| choice_label(o)).collect();
            let pairs: Vec<(String, &str)> = options.iter().map(|o| o.to_string()).zip(labels.iter().map(String::as_str)).collect();
            crate::widgets::dropdown(ui, "graphics-backend", &mut current, &pairs, 220.0);
            obj.insert("gpuBackend".into(), json!(current));
        });
    });
}

/// One user-facing choice; legacy flags remain compatible with older settings.
fn rendering_mode_row(ui: &mut egui::Ui, obj: &mut Map<String, Value>) {
    let mut current = rendering_mode_value(obj);
    ui.horizontal(|ui| {
        ui.label(tl!("Rendering Mode"));
        let pairs =
            vec![("auto".to_string(), tl!("Automatic (recommended)")), ("gpu".to_string(), tl!("GPU")), ("cpu".to_string(), tl!("CPU / Compatibility"))];
        let previous = current.clone();
        crate::widgets::dropdown(ui, "rendering-mode", &mut current, &pairs, 240.0);
        if current != previous {
            obj.insert("renderingMode".into(), json!(current));
            obj.insert("useGpu".into(), json!(current != "cpu"));
        }
    });
    ui.label(tl!("Automatic uses GPU acceleration when available and falls back to CPU rendering on errors."));
    ui.add_space(8.0);
}

fn rendering_mode_value(obj: &Map<String, Value>) -> String {
    obj.get("renderingMode").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| {
        if obj.get("useGpu").and_then(Value::as_bool) == Some(false) || obj.get("gpuBackend").and_then(Value::as_str) == Some("cpu") {
            "cpu".into()
        } else {
            "auto".into()
        }
    })
}

/// Does `section` have any setting the dialog shows (see [`prefs::HIDDEN_UNTIL_IMPLEMENTED`])?
fn has_visible_fields(values: &Value, section: &str) -> bool {
    values.get(section).and_then(Value::as_object).is_some_and(|o| o.keys().any(|k| !prefs::is_hidden(&format!("{section}.{k}"))))
}

/// JPEG and WebP have independent settings; unrelated format controls stay out of view.
fn export_field_visible(obj: &Map<String, Value>, key: &str) -> bool {
    let format = obj.get("quickExportFormat").and_then(Value::as_str).unwrap_or("png");
    match key {
        "jpegQuality" => format == "jpg",
        "webpLossless" => format == "webp",
        "webpQuality" => format == "webp" && obj.get("webpLossless").and_then(Value::as_bool) != Some(true),
        _ => true,
    }
}

/// One theme as a Color Theme swatch: its panel colour under a chrome strip with an accent dot,
/// outlined when chosen. The name is the tooltip and the accessible label. A star badge on its
/// corner adds the theme to (or removes it from) the favourites listed by the appearance
/// button's menu; it shows while the swatch is hovered, and stays for a favourite.
fn theme_swatch(ui: &mut egui::Ui, kind: ThemeKind, id: &str, selected: &mut String, favorites: &mut Vec<SavedTheme>) {
    const BADGE: f32 = 7.0;
    let t = Tokens::get(ui.ctx());
    let p = Tokens::for_kind(kind);
    let label = tl!(&choice_label(id)).to_string();
    let on = selected.as_str() == id;
    let (rect, resp) = ui.allocate_exact_size(vec2(30.0, 26.0), Sense::click());
    let painter = ui.painter();
    let inner = rect.shrink(3.0);
    painter.rect_filled(inner, 2.0, p.dock);
    painter.rect_filled(egui::Rect::from_min_size(inner.min, vec2(inner.width(), 5.0)), 0.0, p.chrome);
    painter.circle_filled(inner.right_bottom() - vec2(6.0, 6.0), 2.5, p.accent);
    let stroke = if on {
        egui::Stroke::new(2.0, t.accent)
    } else if resp.hovered() {
        egui::Stroke::new(1.0, t.text_dim)
    } else {
        egui::Stroke::new(1.0, t.card_border)
    };
    painter.rect_stroke(rect.shrink(1.0), 3.0, stroke, egui::StrokeKind::Inside);
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::RadioButton, true, on, &label));
    if resp.on_hover_text(&label).clicked() {
        *selected = id.to_string();
    }
    let Some(theme) = SavedTheme::parse(id) else { return };
    let favorite = favorites.contains(&theme);
    let center = rect.right_top() + vec2(-3.0, 3.0);
    let badge = egui::Rect::from_center_size(center, vec2(BADGE * 2.0, BADGE * 2.0));
    if !(favorite || ui.rect_contains_pointer(rect.union(badge))) {
        return;
    }
    ui.painter().circle(center, BADGE, t.card, egui::Stroke::new(1.0, t.card_border));
    let (icon, tint) = if favorite { ("star-filled", t.accent) } else { ("star", t.text_dim) };
    let action = if favorite { tl!("Remove from favorites") } else { tl!("Add to favorites") };
    let star = egui::Button::image(crate::icons::image(icon, 10.0, tint)).frame(false);
    let response = ui.put(badge, star).on_hover_text(action);
    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, format!("{action}: {label}")));
    if response.clicked() {
        if favorite {
            favorites.retain(|saved| *saved != theme);
        } else {
            favorites.push(theme);
        }
    }
}

/// A grid row of theme swatches, marked Active when the appearance mode shows this family.
fn theme_row(ui: &mut egui::Ui, title: &str, active: bool, selected: &mut String, favorites: &mut Vec<SavedTheme>, options: &[(&str, ThemeKind)]) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(tl!(title)).color(t.text_dim));
    ui.horizontal(|ui| {
        // Room for the favourite star badges on the swatch corners.
        ui.spacing_mut().item_spacing.x = 6.0;
        for (id, kind) in options {
            theme_swatch(ui, *kind, id, selected, favorites);
        }
        if active {
            ui.add_space(6.0);
            ui.label(RichText::new(tl!("Active")).color(t.accent_text).small());
        }
    });
    ui.end_row();
}

/// Which of the Appearance rows drawn by `appearance_rows` a search shows.
#[derive(Clone, Copy)]
struct AppearanceRows {
    mode: bool,
    light: bool,
    dark: bool,
}

impl AppearanceRows {
    /// `query` is lowercased; empty shows every row.
    fn matching(query: &str) -> Self {
        let shown = |key: &str| query.is_empty() || preference_matches("interface", key, query);
        Self { mode: shown("appearanceMode"), light: shown("lightTheme"), dark: shown("darkTheme") }
    }

    fn any(self) -> bool {
        self.mode || self.light || self.dark
    }
}

/// Appearance Mode and the light and dark Color Theme swatches, as rows of the Appearance grid.
fn appearance_rows(ui: &mut egui::Ui, obj: &mut Map<String, Value>, system: Option<egui::Theme>, rows: AppearanceRows) {
    let t = Tokens::get(ui.ctx());
    let mut mode = obj.get("appearanceMode").and_then(Value::as_str).unwrap_or("auto").to_string();
    if rows.mode {
        ui.label(RichText::new(tl!("Appearance Mode")).color(t.text_dim));
        let choices: [(String, &str); 3] = [("auto".into(), tl!("Sync with system")), ("dark".into(), tl!("Dark")), ("light".into(), tl!("Light"))];
        let pairs: Vec<(String, &str)> = choices.iter().map(|(id, label)| (id.clone(), *label)).collect();
        crate::widgets::dropdown(ui, "appearance-mode", &mut mode, &pairs, 220.0);
        ui.end_row();
    }
    let light_active = mode == "light" || (mode == "auto" && system == Some(egui::Theme::Light));
    obj.insert("appearanceMode".into(), json!(mode));
    let mut light = obj.get("lightTheme").and_then(Value::as_str).unwrap_or("studioLight").to_string();
    let mut dark = obj.get("darkTheme").and_then(Value::as_str).unwrap_or("proMedium").to_string();
    let mut favorites: Vec<SavedTheme> = obj.get("favoriteThemes").cloned().and_then(|v| serde_json::from_value(v).ok()).unwrap_or_default();
    if rows.light {
        theme_row(ui, "Light Theme", light_active, &mut light, &mut favorites, LIGHT_THEME_OPTIONS);
    }
    if rows.dark {
        theme_row(ui, "Dark Theme", !light_active, &mut dark, &mut favorites, DARK_THEME_OPTIONS);
    }
    obj.insert("lightTheme".into(), json!(light));
    obj.insert("darkTheme".into(), json!(dark));
    obj.insert("favoriteThemes".into(), json!(favorites));
}

/// A titled, outlined group of settings (Photoshop's Interface page has Appearance,
/// Presentation and Options).
fn pref_group(ui: &mut egui::Ui, title: &str, add: impl FnOnce(&mut egui::Ui)) {
    let t = Tokens::get(ui.ctx());
    egui::Frame::new().stroke(egui::Stroke::new(1.0, t.card_border)).corner_radius(t.radius).inner_margin(10.0).show(ui, |ui| {
        ui.set_width(ui.available_width());
        ui.label(RichText::new(tl!(title)).strong().color(t.text));
        ui.add_space(4.0);
        add(ui);
    });
    ui.add_space(8.0);
}

/// Preferences › Interface laid out like Photoshop's: the theme and canvas settings, then UI
/// language, font size and scaling, then everything else. Compact enough that the language
/// shows without scrolling in a small window (#2532). A search (`query`, lowercased) hides the
/// rows that don't match and skips a group left empty.
fn interface_groups(ui: &mut egui::Ui, obj: &mut Map<String, Value>, keys: &[String], lang: crate::i18n::Lang, system: Option<egui::Theme>, query: &str) {
    const APPEARANCE: [&str; 3] = ["canvasColor", "canvasCustomColor", "canvasBorder"];
    const PRESENTATION: [&str; 3] = ["language", "uiFontSize", "uiScale"];
    let shown = |k: &str| keys.iter().any(|key| key == k) && field_visible("interface", obj, k, query);
    // In Photoshop's order, not the stored one.
    let pick = |names: &[&str]| names.iter().filter(|n| shown(n)).map(|n| n.to_string()).collect::<Vec<_>>();
    let appearance = pick(&APPEARANCE);
    let presentation = pick(&PRESENTATION);
    let rest = keys.iter().filter(|k| !APPEARANCE.contains(&k.as_str()) && !PRESENTATION.contains(&k.as_str()) && shown(k));
    // Checkboxes line up at the left edge, as in Photoshop; the other options keep a label column.
    let (checks, other): (Vec<String>, Vec<String>) = rest.cloned().partition(|k| obj.get(k).is_some_and(Value::is_boolean));
    let rows = AppearanceRows::matching(query);
    let grid = |id: &str| egui::Grid::new(("prefs-grid", id.to_string())).num_columns(2).spacing([14.0, 7.0]).min_col_width(150.0);
    if rows.any() || !appearance.is_empty() {
        pref_group(ui, "Appearance", |ui| {
            grid("interface-appearance").show(ui, |ui| {
                appearance_rows(ui, obj, system, rows);
                field_rows(ui, "interface", obj, appearance, lang);
            });
        });
    }
    if !presentation.is_empty() {
        pref_group(ui, "Presentation", |ui| {
            grid("interface-presentation").show(ui, |ui| field_rows(ui, "interface", obj, presentation, lang));
        });
    }
    if !checks.is_empty() || !other.is_empty() {
        pref_group(ui, "Options", |ui| {
            egui::Grid::new("prefs-grid-interface-checks").num_columns(2).spacing([0.0, 7.0]).show(ui, |ui| field_rows(ui, "interface", obj, checks, lang));
            grid("interface-options").show(ui, |ui| field_rows(ui, "interface", obj, other, lang));
        });
    }
}

/// Where a clicked colour swatch leaves its preference path and label for the dialog body, which
/// holds the fields the Color Picker request goes in.
const PICK_ID: &str = "prefs-pick-color";

/// The colour swatch clicked this frame, if any: (preference path, label).
fn ctx_take_pick(ui: &egui::Ui) -> Option<(String, String)> {
    ui.ctx().data_mut(|d| d.remove_temp::<(String, String)>(egui::Id::new(PICK_ID)))
}

/// Whether `section.key` gets a generic row for this search (`query`, lowercased; empty matches
/// everything). Settings nothing reads yet stay out of the dialog (issue #204), as do those drawn
/// by their own widgets; their stored values pass through untouched.
fn field_visible(section: &str, obj: &Map<String, Value>, k: &str, query: &str) -> bool {
    let path = format!("{section}.{k}");
    let drawn_elsewhere = (section == "interface" && matches!(k, "theme" | "appearanceMode" | "darkTheme" | "lightTheme" | "favoriteThemes"))
        // View › Rulers (⌘R) sets it, as in Photoshop.
        || path == "unitsAndRulers.showRulers"
        || (section == "performance" && matches!(k, "useGpu" | "gpuBackend" | "renderingMode"))
        || (section == "export" && !export_field_visible(obj, k));
    (query.is_empty() || preference_matches(section, k, query))
        && !drawn_elsewhere
        && !prefs::is_hidden(&path)
        && obj.get(k).is_some_and(|v| !matches!(v, Value::Null | Value::Object(_)))
}

/// Generic editor for a section's fields: checkboxes, dropdowns for choices, colour swatches,
/// number fields with the preference's range, text fields.
fn section_fields(
    ui: &mut egui::Ui,
    section: &str,
    obj: &mut Map<String, Value>,
    order: &[String],
    lang: crate::i18n::Lang,
    system: Option<egui::Theme>,
    search: &str,
) {
    let query = search.to_lowercase();
    let mut keys: Vec<String> = order.iter().filter(|k| obj.contains_key(*k)).cloned().collect();
    keys.extend(obj.keys().filter(|k| !order.contains(k)).cloned());
    if section == "interface" {
        interface_groups(ui, obj, &keys, lang, system, &query);
        return;
    }
    if section == "performance" && (query.is_empty() || preference_matches(section, "renderingMode", &query)) {
        rendering_mode_row(ui, obj);
    }
    keys.retain(|k| field_visible(section, obj, k, &query));
    egui::Grid::new(("prefs-grid", section)).num_columns(2).spacing([14.0, 7.0]).show(ui, |ui| field_rows(ui, section, obj, keys, lang));
}

/// One grid row per setting in `keys` (already filtered by `field_visible`).
fn field_rows(ui: &mut egui::Ui, section: &str, obj: &mut Map<String, Value>, keys: Vec<String>, lang: crate::i18n::Lang) {
    let t = Tokens::get(ui.ctx());
    for k in keys {
        let path = format!("{section}.{k}");
        let v = obj.get(&k).cloned().unwrap_or(Value::Null);
        let human = humanize(&k);
        let label = tl!(&human).to_string();
        match &v {
            Value::Bool(b) => {
                ui.label("");
                let mut b = *b;
                if path == "interface.systemTitleBar" && cfg!(all(not(target_arch = "wasm32"), not(target_os = "macos"))) {
                    ui.vertical(|ui| {
                        crate::widgets::checkbox(ui, &mut b, &label);
                        ui.label(RichText::new(tl!("Applies at next launch.")).color(t.text_dim));
                    });
                } else {
                    crate::widgets::checkbox(ui, &mut b, &label);
                }
                obj.insert(k, json!(b));
            }
            Value::String(s) if path == "interface.language" => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                let mut pairs: Vec<(String, &str)> = vec![("auto".into(), tl!("Auto"))];
                pairs.extend(crate::i18n::Lang::all().map(|l| (l.code().to_string(), l.name())));
                let mut cur = s.clone();
                crate::widgets::dropdown(ui, &format!("pref-{path}"), &mut cur, &pairs, 220.0);
                obj.insert(k, json!(cur));
            }
            Value::String(s) if prefs::choices(&path).is_some() => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                let opts = prefs::choices(&path).unwrap_or(&[]);
                let labels: Vec<String> = opts.iter().map(|o| choice_label(o)).collect();
                let pairs: Vec<(String, &str)> = opts.iter().map(|o| o.to_string()).zip(labels.iter().map(String::as_str)).collect();
                let mut cur = s.clone();
                crate::widgets::dropdown(ui, &format!("pref-{path}"), &mut cur, &pairs, 220.0);
                obj.insert(k, json!(cur));
            }
            Value::String(s) if prefs::is_color(&path) => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                let c = prefs::parse_hex(s).unwrap_or([128, 128, 128]);
                let mut rgb = c;
                ui.horizontal(|ui| {
                    // PhotoCraft's Color Picker, as everywhere else (#2144).
                    if crate::widgets::color_swatch_button(ui, egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]), &label).clicked() {
                        ui.ctx().data_mut(|d| d.insert_temp(egui::Id::new(PICK_ID), (path.clone(), label.clone())));
                    }
                    hex_field(ui, &path, &mut rgb);
                });
                obj.insert(k, json!(format!("#{:02x}{:02x}{:02x}", rgb[0], rgb[1], rgb[2])));
                let _ = color_of(s);
            }
            Value::String(s) => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                let mut s = s.clone();
                ui.add(egui::TextEdit::singleline(&mut s).desired_width(260.0));
                obj.insert(k, json!(s));
            }
            Value::Number(n) => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                let (lo, hi) = prefs::range(&path).unwrap_or((-1e9, 1e9));
                if n.is_u64() || n.is_i64() {
                    let mut x = n.as_i64().unwrap_or(0);
                    ui.add(egui::DragValue::new(&mut x).range(lo as i64..=hi as i64).custom_parser(crate::widgets::parse_num));
                    obj.insert(k, json!(x));
                } else {
                    let mut x = n.as_f64().unwrap_or(0.0);
                    ui.add(egui::DragValue::new(&mut x).range(lo..=hi).speed(0.1).max_decimals(3).custom_parser(crate::widgets::parse_num));
                    obj.insert(k, json!(x));
                }
            }
            Value::Array(_) if path == "tools.pressureCurve" => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                if let Some(next) = crate::pressure_curve_ui::editor(ui, &v) {
                    obj.insert(k, next);
                }
            }
            Value::Array(items) if k == "disks" => {
                ui.label(RichText::new(tl!("Scratch disks")).color(t.text_dim));
                let mut items = items.clone();
                ui.vertical(|ui| {
                    for d in &mut items {
                        ui.horizontal(|ui| {
                            let mut on = d.get("enabled").and_then(Value::as_bool).unwrap_or(false);
                            let mut path = d.get("path").and_then(Value::as_str).unwrap_or("").to_string();
                            crate::widgets::checkbox(ui, &mut on, "");
                            ui.add(egui::TextEdit::singleline(&mut path).desired_width(240.0));
                            *d = json!({"enabled": on, "path": path});
                        });
                    }
                    if ui.small_button(tl!("Add disk")).clicked() {
                        items.push(json!({"enabled": true, "path": ""}));
                    }
                });
                obj.insert(k, Value::Array(items));
            }
            Value::Array(items) => {
                ui.label(RichText::new(tl!(&label)).color(t.text_dim));
                ui.label(RichText::new(crate::i18n::trn(lang, items.len() as u64, "{n} item", "{n} items")).color(t.text_faint));
                if !items.is_empty() && ui.small_button(tl!("Clear")).clicked() {
                    obj.insert(k, json!([]));
                }
            }
            _ => continue,
        }
        ui.end_row();
    }
}

/// Keyboard Shortcuts and Menus: a searchable list of commands by menu with editable shortcuts
/// (click a shortcut, press keys; ⌫ removes it), menu visibility and colours, toolbar tools.
fn shortcuts_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let mut tab = f.get("tab").and_then(Value::as_u64).unwrap_or(0);
    ui.horizontal(|ui| {
        for (i, name) in [tl!("Keyboard Shortcuts"), tl!("Menus"), tl!("Toolbar")].iter().enumerate() {
            if crate::widgets::pill_tab(ui, name, tab == i as u64).clicked() {
                tab = i as u64;
                f.insert("capture".into(), json!(false));
            }
        }
    });
    f.insert("tab".into(), json!(tab));
    ui.add_space(6.0);
    if tab == 2 {
        toolbar_tab(ui, f);
        return;
    }
    let mut filter = f.get("filter").and_then(Value::as_str).unwrap_or("").to_string();
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Search")).color(t.text_dim));
        ui.add(egui::TextEdit::singleline(&mut filter).desired_width(260.0).hint_text(tl!("command or shortcut")));
    });
    f.insert("filter".into(), json!(filter));
    // A malformed map (set over automation) is left as it is, so OK rejects it instead of
    // saving the empty map drawn in its place.
    let parsed = f.get("overrides").map(|v| serde_json::from_value::<BTreeMap<String, String>>(v.clone()));
    let overrides_ok = parsed.as_ref().is_none_or(Result::is_ok);
    let mut overrides = parsed.and_then(Result::ok).unwrap_or_default();
    let mut hidden: Vec<String> = f.get("hidden").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let mut colors: BTreeMap<String, String> = f.get("colors").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    let mut selected = f.get("selected").and_then(Value::as_str).unwrap_or("").to_string();
    let mut capture = f.get("capture").and_then(Value::as_bool).unwrap_or(false);
    let mut message = f.get("message").and_then(Value::as_str).unwrap_or("").to_string();
    let mut message_warning = f.get("messageWarning").and_then(Value::as_bool).unwrap_or(false);
    let items = shortcut_items(app);
    // Menu path and command label as shown in the menus, in the UI language.
    let shown = |id: &str, label: &str, path: &[String]| -> (Vec<String>, String) {
        let lang = crate::i18n::current();
        (path.iter().map(|p| crate::i18n::tr(lang, p).to_string()).collect(), crate::i18n::tr_id(lang, id, label).to_string())
    };
    let eff = |overrides: &BTreeMap<String, String>, id: &str, def: &Option<String>| -> Option<String> {
        match overrides.get(id) {
            Some(s) if s.is_empty() => None,
            Some(s) => Some(s.clone()),
            None => def.clone(),
        }
    };
    // Key capture for the selected command.
    if capture && tab == 0 && !selected.is_empty() {
        let pressed = ui.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Key { key, pressed: true, modifiers, .. } if !is_modifier_key(*key) => Some((*key, *modifiers)),
                _ => None,
            })
        });
        if let Some((key, m)) = pressed {
            ui.input_mut(|i| {
                i.consume_key(m, key);
            });
            if key == egui::Key::Escape {
                capture = false;
            } else if matches!(key, egui::Key::Backspace | egui::Key::Delete) && m.is_none() {
                overrides.insert(selected.clone(), String::new());
                capture = false;
                message = tl!("Shortcut removed.").into();
                message_warning = false;
            } else if let Some(sc) = shortcut_text(key, m) {
                let clash: Vec<String> = items
                    .iter()
                    .filter(|(id, _, _, def)| *id != selected && eff(&overrides, id, def).as_deref().and_then(prefs::normalize_shortcut) == Some(sc.clone()))
                    .map(|(id, label, path, _)| {
                        let (path, label) = shown(id, label, path);
                        format!("{} › {}", path.join(" › "), label.trim_end_matches('…'))
                    })
                    .collect();
                overrides.insert(selected.clone(), sc.clone());
                message_warning = !clash.is_empty();
                message = if clash.is_empty() {
                    let template = "{shortcut} assigned.";
                    crate::i18n::fmt(tl!(template), &[("shortcut", &crate::shortcuts::pretty(&sc))])
                } else {
                    let template = "{shortcut} is already in use by {commands} and will be removed from it when you click OK.";
                    crate::i18n::fmt(tl!(template), &[("shortcut", &crate::shortcuts::pretty(&sc)), ("commands", &clash.join(", "))])
                };
                capture = false;
            }
        }
    }
    let needle = filter.to_ascii_lowercase();
    egui::ScrollArea::vertical().max_height(420.0).id_salt("shortcuts-scroll").show(ui, |ui| {
        let mut last_top = String::new();
        egui::Grid::new("shortcut-grid").num_columns(3).spacing([12.0, 3.0]).striped(true).show(ui, |ui| {
            for (id, label, path, def) in &items {
                if tab == 1 && crate::tool_shortcuts::tool(id).is_some() {
                    continue;
                }
                let cur = eff(&overrides, id, def);
                let (shown_path, shown_label) = shown(id, label, path);
                // Match what is shown and the English name (commands are documented in English).
                let hay = format!("{} {} {} {} {}", shown_path.join(" "), shown_label, path.join(" "), label, cur.clone().unwrap_or_default()).to_lowercase();
                if !needle.is_empty() && !hay.contains(&needle) && !id.to_ascii_lowercase().contains(&needle) {
                    continue;
                }
                if tab == 0 && path.is_empty() && def.is_none() && cur.is_none() {
                    continue;
                }
                let top = shown_path.first().cloned().unwrap_or_else(|| tl!("Other").into());
                if top != last_top {
                    ui.label(RichText::new(&top).font(crate::theme::semibold(12.5)).color(t.text));
                    ui.label("");
                    ui.label("");
                    ui.end_row();
                    last_top = top;
                }
                let name = match shown_path.get(1..) {
                    Some(rest) if !rest.is_empty() => format!("{} › {}", rest.join(" › "), shown_label),
                    _ => shown_label,
                };
                let sel = selected == *id;
                if ui.selectable_label(sel, RichText::new(format!("   {name}")).color(t.text_dim)).clicked() {
                    selected = id.clone();
                }
                if tab == 0 {
                    let text = if sel && capture { tl!("Press keys…").to_string() } else { cur.as_deref().map(crate::shortcuts::pretty).unwrap_or_default() };
                    let changed = overrides.contains_key(id);
                    let r = ui.add(
                        egui::Button::new(RichText::new(tl!(&text)).size(12.0).color(if changed { t.accent_text } else { t.text })).min_size(vec2(120.0, 18.0)),
                    );
                    if r.clicked() {
                        selected = id.clone();
                        capture = true;
                        message.clear();
                        message_warning = false;
                    }
                    ui.label(if changed { RichText::new(tl!("modified")).color(t.text_faint).size(10.5) } else { RichText::new("") });
                } else {
                    let mut visible = !hidden.contains(id);
                    crate::widgets::checkbox(ui, &mut visible, tl!("Visible"));
                    if visible {
                        hidden.retain(|h| h != id);
                    } else if !hidden.contains(id) {
                        hidden.push(id.clone());
                    }
                    let mut col = colors.get(id).cloned().unwrap_or_else(|| "none".into());
                    let opts: Vec<(String, &str)> =
                        ["none", "red", "orange", "yellow", "green", "blue", "violet", "gray"].iter().map(|c| (c.to_string(), *c)).collect();
                    crate::widgets::dropdown(ui, &format!("menu-color-{id}"), &mut col, &opts, 90.0);
                    if col == "none" {
                        colors.remove(id);
                    } else {
                        colors.insert(id.clone(), col);
                    }
                }
                ui.end_row();
            }
        });
    });
    ui.add_space(6.0);
    if tab == 0 {
        ui.horizontal(|ui| {
            let has_sel = !selected.is_empty();
            if ui.add_enabled(has_sel, egui::Button::new(tl!("Use Default"))).clicked() {
                overrides.remove(&selected);
                message = tl!("Default restored.").into();
                message_warning = false;
            }
            if ui.add_enabled(has_sel, egui::Button::new(tl!("Delete Shortcut"))).clicked() {
                overrides.insert(selected.clone(), String::new());
                message = tl!("Shortcut removed.").into();
                message_warning = false;
            }
            if ui.button(tl!("Reset All to Defaults")).clicked() {
                overrides.clear();
                message = tl!("All shortcuts reset to PhotoCraft defaults.").into();
                message_warning = false;
            }
            // A Photoshop set (`.kys`) fills the overrides; the file arrives after this frame
            // and `kys_import` writes the dialog's fields then (#1163).
            if ui.button(tl!("Import Shortcuts…")).on_hover_text(tl!("Load a keyboard shortcut set saved as .kys")).clicked()
                && let Err(e) = crate::kys_import::import_dialog(app)
            {
                message = e;
                message_warning = false;
            }
        });
    } else if ui.button(tl!("Show All Menu Items")).clicked() {
        hidden.clear();
        colors.clear();
    }
    if !message.is_empty() {
        ui.label(RichText::new(&message).color(if message_warning { t.warning } else { t.text_dim }));
    }
    if overrides_ok {
        f.insert("overrides".into(), json!(overrides));
    }
    f.insert("hidden".into(), json!(hidden));
    f.insert("colors".into(), json!(colors));
    f.insert("selected".into(), json!(selected));
    f.insert("capture".into(), json!(capture));
    f.insert("message".into(), json!(message));
    f.insert("messageWarning".into(), json!(message_warning));
}

fn toolbar_tab(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let mut hidden: Vec<String> = f.get("toolbarHidden").and_then(|v| serde_json::from_value(v.clone()).ok()).unwrap_or_default();
    egui::ScrollArea::vertical().max_height(380.0).id_salt("toolbar-scroll").show(ui, |ui| {
        egui::Grid::new("toolbar-grid").num_columns(2).spacing([16.0, 4.0]).show(ui, |ui| {
            for tool in crate::state::Tool::ALL {
                let name = format!("{tool:?}");
                let mut on = !hidden.contains(&name);
                crate::widgets::checkbox(ui, &mut on, tl!(tool.label()));
                if on {
                    hidden.retain(|h| *h != name);
                } else if !hidden.contains(&name) {
                    hidden.push(name);
                }
                let k = tool.key();
                ui.label(if k == '\0' { String::new() } else { k.to_string() });
                ui.end_row();
            }
        });
    });
    if ui.button(tl!("Restore Defaults")).clicked() {
        hidden.clear();
    }
    f.insert("toolbarHidden".into(), json!(hidden));
}

fn presets_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let mut kind = f.get("kind").and_then(Value::as_str).unwrap_or("brushes").to_string();
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Preset Type")).color(t.text_dim));
        let opts = [("brushes".to_string(), tl!("Brushes")), ("customShapes".to_string(), tl!("Custom Shapes")), ("patterns".to_string(), tl!("Patterns"))];
        crate::widgets::dropdown(ui, "preset-kind", &mut kind, &opts, 180.0);
    });
    let list = app
        .session
        .execute("edit.presets.presetManager", json!({"kind": kind}))
        .ok()
        .and_then(|v| v.get(&kind).cloned())
        .and_then(|v| serde_json::from_value::<Vec<String>>(v).ok())
        .unwrap_or_default();
    let mut selected = f.get("selected").and_then(Value::as_u64).unwrap_or(0) as usize;
    egui::ScrollArea::vertical().max_height(260.0).id_salt("preset-list").show(ui, |ui| {
        for (i, name) in list.iter().enumerate() {
            if ui.selectable_label(i == selected, name).clicked() {
                selected = i;
                f.insert("newName".into(), json!(name));
            }
        }
        if list.is_empty() {
            ui.label(RichText::new(tl!("No presets of this type.")).color(t.text_faint));
        }
    });
    let mut new_name = f.get("newName").and_then(Value::as_str).unwrap_or("").to_string();
    ui.horizontal(|ui| {
        ui.add(egui::TextEdit::singleline(&mut new_name).desired_width(200.0));
        if ui.add_enabled(selected < list.len() && !new_name.trim().is_empty(), egui::Button::new(tl!("Rename"))).clicked() {
            let _ = app.run("edit.presets.presetManager", json!({"action": "rename", "kind": kind, "index": selected, "newName": new_name}));
        }
        if ui.add_enabled(selected < list.len(), egui::Button::new(tl!("Delete"))).clicked() {
            let _ = app.run("edit.presets.presetManager", json!({"action": "delete", "kind": kind, "index": selected}));
        }
        // Load Photoshop brushes (.abr) into the library.
        if kind == "brushes" && ui.button(tl!("Load…")).on_hover_text(tl!("Import brushes (.abr)")).clicked() {
            let _ = app.open_dialog_file();
        }
    });
    f.insert("kind".into(), json!(kind));
    f.insert("selected".into(), json!(selected));
    f.insert("newName".into(), json!(new_name));
}

fn presets_io_body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    let mut action = f.get("action").and_then(Value::as_str).unwrap_or("export").to_string();
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Action")).color(t.text_dim));
        crate::widgets::dropdown(
            ui,
            "presets-io",
            &mut action,
            &[("export".to_string(), tl!("Export Presets")), ("import".to_string(), tl!("Import Presets"))],
            180.0,
        );
    });
    for (k, label) in [("brushes", tl!("Brushes")), ("customShapes", tl!("Custom Shapes"))] {
        let mut on = f.get(k).and_then(Value::as_bool).unwrap_or(true);
        crate::widgets::checkbox(ui, &mut on, label);
        f.insert(k.into(), json!(on));
    }
    f.insert("action".into(), json!(action));
}

fn mismatch_body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(f.get("message").and_then(Value::as_str).unwrap_or("")).color(t.text));
    ui.add_space(6.0);
    let mut action = f.get("action").and_then(Value::as_str).unwrap_or("preserve").to_string();
    let missing = f.get("missing").and_then(Value::as_bool).unwrap_or(false);
    let opts: &[(&str, &str)] = if missing {
        &[("preserve", tl!("Leave as is (don't color manage)")), ("assignWorking", tl!("Assign working space"))]
    } else {
        &[
            ("preserve", tl!("Use the embedded profile (instead of the working space)")),
            ("convert", tl!("Convert document's colors to the working space")),
            ("discard", tl!("Discard the embedded profile (don't color manage)")),
        ]
    };
    for (v, label) in opts {
        ui.radio_value(&mut action, v.to_string(), *label);
    }
    f.insert("action".into(), json!(action));
}

// ------------------------------------------------------------------ confirm

/// OK on one of our dialogs.
pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    match f.get("__prefsui").and_then(Value::as_str).unwrap_or("") {
        "prefs" => {
            let values = f.get("values").cloned().unwrap_or(Value::Null);
            app.run("prefs.set", json!({"path": "", "value": values}))
        }
        "shortcuts" => {
            let overrides = f.get("overrides").cloned().unwrap_or(json!({}));
            // Shortcuts moved to another command are taken from their old owner (shell
            // commands included, which the engine doesn't know).
            // The map replaces every stored override, so a malformed one is an error, not "none".
            let mut ov: BTreeMap<String, String> =
                serde_json::from_value(overrides).map_err(|e| format!("overrides must map command IDs to shortcut strings: {e}"))?;
            let items = shortcut_items(app);
            // Newly assigned shortcuts are checked first and take theirs from any holder, custom
            // overrides included; unchanged overrides only take theirs from defaults.
            let before = &app.session.prefs().shortcuts;
            let mut todo: Vec<(String, String, bool)> =
                ov.iter().filter(|(_, s)| !s.is_empty()).map(|(k, s)| (k.clone(), s.clone(), before.get(k) != Some(s))).collect();
            todo.sort_by_key(|(_, _, new)| !new);
            for (id, sc, new) in todo {
                if ov.get(&id).is_some_and(String::is_empty) {
                    continue; // taken by a newer assignment
                }
                let norm = prefs::normalize_shortcut(&sc);
                for (other, _, _, def) in &items {
                    let held = match ov.get(other) {
                        Some(s) if new => Some(s.as_str()),
                        Some(_) => None,
                        None => def.as_deref(),
                    };
                    if *other != id && held.and_then(prefs::normalize_shortcut) == norm {
                        ov.insert(other.clone(), String::new());
                    }
                }
            }
            let hidden = f.get("hidden").cloned().unwrap_or(json!([]));
            let colors = f.get("colors").cloned().unwrap_or(json!({}));
            let toolbar = f.get("toolbarHidden").cloned().unwrap_or(json!([]));
            app.run("edit.keyboardShortcuts", json!({"reset": true, "set": ov, "allowUnknown": true, "removeConflicts": false}))?;
            app.run("prefs.set", json!({"values": {"menus": {"hidden": hidden, "colors": colors}, "toolbar": {"hidden": toolbar}}}))
        }
        "presets" => Ok(Value::Null),
        "presetsIO" => {
            let kinds: Vec<&str> = ["brushes", "customShapes"].into_iter().filter(|k| f.get(*k).and_then(Value::as_bool).unwrap_or(true)).collect();
            if f.get("action").and_then(Value::as_str) == Some("import") {
                app.pick_file_bytes(move |app, name, bytes| {
                    let text = String::from_utf8(bytes).map_err(|_| format!("{name} is not a preset file"))?;
                    app.run("edit.presets.exportImportPresets", json!({"action": "import", "kinds": kinds, "data": text}))
                })
            } else {
                let out = app.run("edit.presets.exportImportPresets", json!({"action": "export", "kinds": kinds}))?;
                let text = serde_json::to_string_pretty(&out["data"]).map_err(|e| e.to_string())?;
                app.pick_save("Presets.pcpresets", move |app, path| {
                    let write = app.services.write.as_mut().ok_or("no writer configured")?;
                    write(&path, text.as_bytes())?;
                    let template = "Exported presets to {path}";
                    app.ui.status = crate::i18n::fmt(tl!(template), &[("path", &path)]);
                    Ok(json!({"path": path}))
                })
            }
        }
        // The file it asks about opens with this answer (`open_prompt`).
        "mismatch" => {
            let action = f.get("action").and_then(Value::as_str).unwrap_or("preserve");
            crate::open_prompt::answer(app, Some(action))
        }
        _ => Err("unknown dialog".into()),
    }
}

// Test modules stay after the production code: the prefs_usage test reads each file up to
// its first `#[cfg(test)] mod`.
#[cfg(test)]
mod recovery_tests;

#[cfg(test)]
#[path = "shortcut_capture_tests.rs"]
mod shortcut_capture_tests;

#[cfg(test)]
mod tests {
    #[test]
    fn colour_preferences_take_a_typed_hex_value() {
        // #668: the canvas colour (and every colour preference) showed its hex but couldn't take one.
        use egui::accesskit::Role;
        use egui_kittest::kittest::Queryable;
        let mut h = egui_kittest::Harness::new_ui_state(|ui, rgb: &mut [u8; 3]| super::hex_field(ui, "interface.canvasCustomColor", rgb), [0x28, 0x28, 0x28]);
        h.run();
        assert_eq!(h.get_by_role(Role::TextInput).value().as_deref(), Some("#282828"));
        h.get_by_role(Role::TextInput).click();
        h.run_steps(1);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        h.run_steps(1);
        for ch in "#80".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        // A partial value stays as typed and leaves the colour alone.
        assert_eq!(h.get_by_role(Role::TextInput).value().as_deref(), Some("#80"));
        assert_eq!(*h.state(), [0x28, 0x28, 0x28]);
        for ch in "8080".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        assert_eq!(*h.state(), [0x80, 0x80, 0x80]);
    }

    #[test]
    fn quick_export_controls_match_selected_format() {
        let mut obj = serde_json::to_value(photocraft_engine::prefs::Export::default()).unwrap().as_object().unwrap().clone();
        assert!(!export_field_visible(&obj, "jpegQuality"));
        assert!(!export_field_visible(&obj, "webpLossless"));
        assert!(!export_field_visible(&obj, "webpQuality"));

        obj.insert("quickExportFormat".into(), json!("jpg"));
        assert!(export_field_visible(&obj, "jpegQuality"));
        assert!(!export_field_visible(&obj, "webpLossless"));
        assert!(!export_field_visible(&obj, "webpQuality"));

        obj.insert("quickExportFormat".into(), json!("webp"));
        assert!(!export_field_visible(&obj, "jpegQuality"));
        assert!(export_field_visible(&obj, "webpLossless"));
        assert!(!export_field_visible(&obj, "webpQuality"), "lossless WebP does not have a quality setting");

        obj.insert("webpLossless".into(), json!(false));
        assert!(export_field_visible(&obj, "webpQuality"));
        assert_eq!(humanize("webpLossless"), "Lossless");
        assert_eq!(humanize("webpQuality"), "Quality");
    }

    #[test]
    fn rendering_mode_display_respects_explicit_mode_and_legacy_disable() {
        let legacy = serde_json::json!({"useGpu": false, "gpuBackend": "auto"});
        assert_eq!(super::rendering_mode_value(legacy.as_object().unwrap()), "cpu");
        let explicit = serde_json::json!({"renderingMode": "gpu", "useGpu": false, "gpuBackend": "cpu"});
        assert_eq!(super::rendering_mode_value(explicit.as_object().unwrap()), "gpu");
        let automatic = serde_json::json!({"renderingMode": null, "useGpu": true, "gpuBackend": "auto"});
        assert_eq!(super::rendering_mode_value(automatic.as_object().unwrap()), "auto");
    }

    #[test]
    fn preferences_search_matches_editable_fields_across_sections() {
        let values = prefs::Preferences::default().to_json();
        let contains = |needle: &str, section: &str| preference_sections(&values, needle).iter().any(|(id, _)| *id == section);
        assert!(contains("UI Scale", "interface"));
        assert!(contains("placeholder", "type"));
        assert!(contains("performance", "performance"));
        assert!(contains("interface.uiScale", "interface"));
        assert!(preference_sections(&values, "this setting does not exist").is_empty());
        assert_eq!(preference_sections(&values, "").len(), SECTIONS.len());
    }

    #[test]
    fn preferences_dialog_width_follows_the_window() {
        let prefs = serde_json::json!({"__prefsui": "prefs"});
        let prefs = prefs.as_object().unwrap();
        assert_eq!(super::width(prefs, 200.0), Some(380.0));
        assert_eq!(super::width(prefs, 800.0), Some(800.0));
        assert_eq!(super::width(prefs, 3000.0), Some(1040.0));
        let shortcuts = serde_json::json!({"__prefsui": "shortcuts"});
        assert_eq!(super::width(shortcuts.as_object().unwrap(), 3000.0), Some(720.0));
    }
    use super::*;
    use std::sync::{Arc, Mutex};

    #[test]
    fn appearance_resolves_both_saved_themes_and_follows_system() {
        let mut interface = prefs::Interface { appearance_mode: AppearanceMode::Auto, ..prefs::Interface::default() };
        assert_eq!(selected_theme(&interface, Some(egui::Theme::Dark)), ThemeKind::ProMedium);
        assert_eq!(selected_theme(&interface, Some(egui::Theme::Light)), ThemeKind::ProLight);
        interface.dark_theme = DarkTheme::Studio;
        interface.light_theme = LightTheme::Classic;
        assert_eq!(selected_theme(&interface, Some(egui::Theme::Dark)), ThemeKind::Studio);
        assert_eq!(selected_theme(&interface, Some(egui::Theme::Light)), ThemeKind::Classic);
        interface.appearance_mode = AppearanceMode::Dark;
        assert_eq!(selected_theme(&interface, Some(egui::Theme::Light)), ThemeKind::Studio);
        interface.appearance_mode = AppearanceMode::Light;
        assert_eq!(selected_theme(&interface, Some(egui::Theme::Dark)), ThemeKind::Classic);
    }

    #[test]
    fn breeze_themes_are_chosen_by_preference_and_fix_the_matching_mode() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        for (id, kind, mode) in [
            ("breezeLight", ThemeKind::BreezeLight, AppearanceMode::Light),
            ("breezeDark", ThemeKind::BreezeDark, AppearanceMode::Dark),
            ("softLight", ThemeKind::SoftLight, AppearanceMode::Light),
            ("softDark", ThemeKind::SoftDark, AppearanceMode::Dark),
        ] {
            app.run("prefs.set", json!({"path": "interface.theme", "value": id})).unwrap();
            tick(&mut app, &ctx);
            let interface = app.session.prefs().interface.clone();
            assert_eq!(interface.appearance_mode, mode, "{id}");
            assert_eq!(selected_theme(&interface, None), kind, "{id}");
            assert_eq!(app.ui.theme, kind, "{id}");
        }
        // The two slots are independent: choosing the dark one left the light choice alone.
        assert_eq!(app.session.prefs().interface.light_theme, LightTheme::SoftLight);
        assert_eq!(app.session.prefs().interface.dark_theme, DarkTheme::SoftDark);
    }

    #[test]
    fn auto_uses_native_system_theme_when_egui_does_not_report_one() {
        let (mut app, _) = app_with_store();
        app.services.system_theme = Some(Box::new(|_| Some(egui::Theme::Light)));
        app.run("prefs.set", json!({"path": "interface.appearanceMode", "value": "auto"})).unwrap();
        app.run("prefs.set", json!({"path": "interface.appearanceMode", "value": "auto"})).unwrap();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        assert_eq!(app.ui.theme, ThemeKind::ProLight);
    }

    #[test]
    fn appearance_button_cycles_modes_without_losing_theme_choices() {
        let ctx = egui::Context::default();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("prefs.set", json!({"values": {"interface.darkTheme": "studio", "interface.lightTheme": "classic"}})).unwrap();
        for (mode, visible) in
            [(AppearanceMode::Auto, ThemeKind::Studio), (AppearanceMode::Light, ThemeKind::Classic), (AppearanceMode::Dark, ThemeKind::Studio)]
        {
            cycle_appearance(&mut app, &ctx);
            assert_eq!(app.session.prefs().interface.appearance_mode, mode);
            assert_eq!(app.ui.theme, visible);
            assert_eq!(app.session.prefs().interface.dark_theme, DarkTheme::Studio);
            assert_eq!(app.session.prefs().interface.light_theme, LightTheme::Classic);
        }
    }

    fn appearance_frame(app: &mut PhotocraftApp, ctx: &egui::Context, appearance: Option<egui::Theme>) {
        ctx.run_ui(egui::RawInput { system_theme: appearance, ..Default::default() }, |ui| tick(app, ui.ctx())).textures_delta.clear();
    }

    fn assert_palette(ctx: &egui::Context, kind: ThemeKind) {
        let t = Tokens::get(ctx);
        assert_eq!(t.kind, kind);
        for style in [ctx.global_style(), ctx.style_of(egui::Theme::Light), ctx.style_of(egui::Theme::Dark)] {
            assert_eq!(style.visuals.panel_fill, t.chrome);
            assert_eq!(style.visuals.window_fill, t.card);
            assert_eq!(style.visuals.override_text_color, Some(t.text));
            assert_eq!(style.visuals.widgets.inactive.bg_fill, t.field);
            assert_eq!(style.visuals.dark_mode, t.dark());
            assert_eq!(style.text_styles[&egui::TextStyle::Body].size, if t.pro { 12.0 } else { 12.5 });
        }
        assert_eq!(ctx.options(|o| o.theme_preference), egui::ThemePreference::System);
    }

    #[test]
    fn auto_appearance_follows_live_changes_without_saving_resolved_palettes() {
        for (dark, light) in [(ThemeKind::ProMedium, ThemeKind::StudioLight), (ThemeKind::SolarizedDark, ThemeKind::Adwaita)] {
            let (mut app, store) = app_with_store();
            let ctx = egui::Context::default();
            PhotocraftApp::setup_context(&ctx, Default::default());
            app.run("prefs.set", json!({"values": {"interface.appearanceMode": "auto", "interface.darkTheme": dark.id(), "interface.lightTheme": light.id()}}))
                .unwrap();
            appearance_frame(&mut app, &ctx, Some(egui::Theme::Light));
            let revision = app.session.prefs.rev();
            let interface = app.session.prefs().interface.clone();
            let saved = store.lock().unwrap().clone().unwrap();
            assert_eq!(stored(&store)["interface"]["appearanceMode"], "auto");
            let gpu_style = canvas_style(&app);
            for (appearance, expected) in [(Some(egui::Theme::Light), light), (Some(egui::Theme::Dark), dark), (Some(egui::Theme::Light), light), (None, dark)]
            {
                appearance_frame(&mut app, &ctx, appearance);
                assert_palette(&ctx, expected);
                assert_eq!(app.ui.theme, expected);
                assert_eq!(app.session.prefs().interface, interface);
                assert_eq!(app.session.prefs.rev(), revision);
                assert_eq!(store.lock().unwrap().as_ref(), Some(&saved));
                assert_eq!(canvas_style(&app), gpu_style);
                assert_eq!(pasteboard_color(&app), None);
            }
            // A fresh launch resolves its saved Auto choice against the new OS input.
            for (appearance, expected) in [(Some(egui::Theme::Light), light), (Some(egui::Theme::Dark), dark), (None, dark)] {
                let (mut restarted, store) = app_with_saved(Some(saved.clone()));
                let fresh = egui::Context::default();
                PhotocraftApp::setup_context(&fresh, Default::default());
                appearance_frame(&mut restarted, &fresh, appearance);
                assert_palette(&fresh, expected);
                assert_eq!(restarted.ui.theme, expected);
                assert_eq!(restarted.session.prefs().interface, interface);
                assert_eq!(store.lock().unwrap().as_ref(), Some(&saved));
            }
            app.run("prefs.set", json!({"values": {"interface.canvasColor": "custom", "interface.canvasCustomColor": "#123456"}})).unwrap();
            for appearance in [Some(egui::Theme::Light), Some(egui::Theme::Dark)] {
                appearance_frame(&mut app, &ctx, appearance);
                assert_eq!(pasteboard_color(&app), Some(Color32::from_rgb(0x12, 0x34, 0x56)));
                assert_eq!(canvas_style(&app), gpu_style);
            }
        }
    }

    #[test]
    fn manual_theme_tokens_and_global_styles_stay_fixed_across_os_changes() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, Default::default());
        for kind in ThemeKind::ALL {
            app.run("prefs.set", json!({"path": "interface.theme", "value": kind.id()})).unwrap();
            appearance_frame(&mut app, &ctx, Some(egui::Theme::Dark));
            let interface = app.session.prefs().interface.clone();
            let revision = app.session.prefs.rev();
            let saved = store.lock().unwrap().clone().unwrap();
            assert_eq!(interface.appearance_mode, if Tokens::for_kind(kind).dark() { AppearanceMode::Dark } else { AppearanceMode::Light });
            for appearance in [Some(egui::Theme::Dark), Some(egui::Theme::Light), Some(egui::Theme::Dark), None] {
                appearance_frame(&mut app, &ctx, appearance);
                assert_palette(&ctx, kind);
                assert_eq!(Tokens::get(&ctx), Tokens::for_kind(kind));
                assert_eq!(app.ui.theme, kind);
                assert_eq!(app.session.prefs().interface, interface);
                assert_eq!(app.session.prefs.rev(), revision);
                assert_eq!(store.lock().unwrap().as_ref(), Some(&saved));
            }
        }
    }

    #[test]
    fn auto_appearance_replaces_a_restored_egui_dark_pin() {
        let (mut app, _) = app_with_saved(Some(json!({"interface": {"appearanceMode": "auto"}}).to_string()));
        let ctx = egui::Context::default();
        ctx.set_theme(egui::ThemePreference::Dark);
        assert_eq!(ctx.options(|o| o.theme_preference), egui::ThemePreference::Dark);
        appearance_frame(&mut app, &ctx, Some(egui::Theme::Light));
        assert_palette(&ctx, ThemeKind::ProLight);
        assert_eq!(app.session.prefs().interface.appearance_mode, AppearanceMode::Auto);
        // A restored pin must also clear when the concrete PhotoCraft palette stays the same.
        ctx.set_theme(egui::ThemePreference::Dark);
        appearance_frame(&mut app, &ctx, Some(egui::Theme::Light));
        assert_eq!(ctx.options(|o| o.theme_preference), egui::ThemePreference::System);
        assert_palette(&ctx, ThemeKind::ProLight);
        appearance_frame(&mut app, &ctx, Some(egui::Theme::Dark));
        assert_palette(&ctx, ThemeKind::ProMedium);
        assert_eq!(app.session.prefs().interface.appearance_mode, AppearanceMode::Auto);
    }

    #[test]
    fn system_theme_menu_choice_stays_checked_as_the_appearance_changes() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, Default::default());
        app.run("prefs.set", json!({"values": {"interface.darkTheme": "solarizedDark", "interface.lightTheme": "adwaita"}})).unwrap();
        appearance_frame(&mut app, &ctx, Some(egui::Theme::Light));
        crate::menus::invoke(&mut app, &ctx, "window.theme.system", json!({})).unwrap();
        appearance_frame(&mut app, &ctx, Some(egui::Theme::Light));
        let interface = app.session.prefs().interface.clone();
        let saved = store.lock().unwrap().clone().unwrap();
        let revision = app.session.prefs.rev();
        for appearance in [Some(egui::Theme::Light), Some(egui::Theme::Dark), None] {
            appearance_frame(&mut app, &ctx, appearance);
            let items = crate::menus::menu_items(&app);
            let item = |id: &str| items.iter().find(|i| i.id == id).unwrap();
            assert_eq!(item("window.theme.system").checked, Some(true));
            assert!(item("window.theme.system").enabled);
            assert_eq!(item("window.theme.system").path, ["Window", "Theme"]);
            for kind in ThemeKind::ALL {
                assert_eq!(item(&format!("window.theme.{}", kind.id())).checked, Some(false));
            }
            assert_eq!(app.session.prefs().interface.appearance_mode, AppearanceMode::Auto);
            assert_eq!(app.session.prefs().interface, interface);
            assert_eq!(app.session.prefs.rev(), revision);
            assert_eq!(store.lock().unwrap().as_ref(), Some(&saved));
        }
        crate::menus::invoke(&mut app, &ctx, "window.theme.solarizedDark", json!({})).unwrap();
        for appearance in [Some(egui::Theme::Light), Some(egui::Theme::Dark)] {
            appearance_frame(&mut app, &ctx, appearance);
            assert_palette(&ctx, ThemeKind::SolarizedDark);
            let items = crate::menus::menu_items(&app);
            let item = |id: &str| items.iter().find(|i| i.id == id).unwrap();
            assert_eq!(item("window.theme.solarizedDark").checked, Some(true));
            assert_eq!(item("window.theme.system").checked, Some(false));
            assert_eq!(stored(&store)["interface"]["appearanceMode"], "dark");
            assert_eq!(app.session.prefs().interface.light_theme, LightTheme::Adwaita);
        }
    }

    #[test]
    fn preferences_interface_offers_and_applies_sync_with_system() {
        use egui_kittest::{Harness, kittest::Queryable};
        let (mut app, store) = app_with_store();
        app.run("prefs.set", json!({"values": {"interface.language": "en", "interface.darkTheme": "solarizedDark", "interface.lightTheme": "adwaita"}}))
            .unwrap();
        let interface = app.session.prefs().interface.clone();
        open_preferences(&mut app, "interface");
        let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.input_mut().system_theme = Some(egui::Theme::Light);
        h.run_steps(4);
        for label in ["Adwaita", "Adwaita dark", "Solarized dark"] {
            assert!(h.query_by_label(label).is_some(), "missing theme choice {label}");
        }
        h.get_by_value("Dark").click();
        h.run_steps(2);
        h.get_by_label("Sync with system").click();
        h.run_steps(2);
        h.get_by_label("Apply").click();
        h.run_steps(4);
        assert_eq!(h.state().session.prefs().interface.appearance_mode, AppearanceMode::Auto);
        assert_eq!(h.state().session.prefs().interface.dark_theme, interface.dark_theme);
        assert_eq!(h.state().session.prefs().interface.light_theme, interface.light_theme);
        assert_eq!(h.state().session.prefs().interface.theme, interface.theme);
        assert_palette(&h.ctx, ThemeKind::Adwaita);
        assert_eq!(stored(&store)["interface"]["appearanceMode"], "auto");
        let saved = store.lock().unwrap().clone().unwrap();
        let (mut restarted, reloaded) = app_with_saved(Some(saved.clone()));
        let fresh = egui::Context::default();
        PhotocraftApp::setup_context(&fresh, Default::default());
        appearance_frame(&mut restarted, &fresh, Some(egui::Theme::Dark));
        assert_palette(&fresh, ThemeKind::SolarizedDark);
        assert_eq!(restarted.session.prefs().interface.appearance_mode, AppearanceMode::Auto);
        assert_eq!(restarted.session.prefs().interface.dark_theme, interface.dark_theme);
        assert_eq!(restarted.session.prefs().interface.light_theme, interface.light_theme);
        assert_eq!(restarted.session.prefs().interface.theme, interface.theme);
        assert_eq!(reloaded.lock().unwrap().as_ref(), Some(&saved));
    }

    #[test]
    fn theme_star_saves_favorite_and_menu_groups_it() {
        use egui_kittest::{Harness, kittest::Queryable};
        let (mut app, _) = app_with_store();
        open_preferences(&mut app, "interface");
        let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        h.get_by_label("Classic").hover();
        h.run_steps(2);
        h.get_by_label("Add to favorites: Classic").click();
        h.run_steps(2);
        h.get_by_label("Apply").click();
        h.run_steps(2);
        assert_eq!(h.state().session.prefs().interface.favorite_themes, vec![SavedTheme::Classic]);
        assert!(h.query_by_label("Remove from favorites: Classic").is_some());
        let (light, dark) = favorite_theme_groups(&[SavedTheme::Studio, SavedTheme::Classic, SavedTheme::Adwaita]);
        assert_eq!(light, vec![ThemeKind::Classic, ThemeKind::Adwaita]);
        assert_eq!(dark, vec![ThemeKind::Studio]);
    }

    #[test]
    fn preferences_interface_shows_language_without_scrolling() {
        // #2532: in a small window the theme cards filled the page and Language sat out of view
        // below them. Like Photoshop, Presentation follows Appearance and both fit.
        use egui_kittest::{Harness, kittest::Queryable};
        let (mut app, _) = app_with_store();
        app.run("prefs.set", json!({"values": {"interface.language": "en"}})).unwrap();
        open_preferences(&mut app, "interface");
        let mut h = Harness::builder().with_size(vec2(1000.0, 750.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        let language = h.get_by_label("Language").rect();
        let dark = h.get_by_label("Dark Theme").rect();
        let ok = h.get_by_label("OK").rect();
        assert!(dark.bottom() < language.top(), "Presentation {language:?} follows the theme swatches {dark:?}");
        assert!(language.bottom() < ok.top() - 20.0, "Language {language:?} must show above the buttons {ok:?} without scrolling");
        // Every theme in the option lists gets a swatch, Breeze included.
        for (id, _) in LIGHT_THEME_OPTIONS.iter().chain(DARK_THEME_OPTIONS) {
            let label = choice_label(id);
            assert!(h.query_by_label(&label).is_some(), "missing theme swatch {label}");
        }
        assert!(h.query_by_label("Breeze light").is_some() && h.query_by_label("Breeze dark").is_some());
    }

    #[test]
    fn preferences_interface_search_hides_rows_and_skips_empty_groups() {
        use egui_kittest::{Harness, kittest::Queryable};
        let render = |search: &str| {
            let (mut app, _) = app_with_store();
            app.run("prefs.set", json!({"values": {"interface.language": "en"}})).unwrap();
            let id = open_preferences(&mut app, "interface");
            app.ui.dialog_mut(id).unwrap().fields.insert("__search".into(), json!(search));
            let mut h = Harness::builder().with_size(vec2(1000.0, 750.0)).build_eframe(move |cc| {
                PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
                app
            });
            h.run_steps(4);
            h
        };
        let shown = |h: &Harness<'_, PhotocraftApp>, label: &str| h.query_all_by_label(label).next().is_some();

        let h = render("language");
        assert!(shown(&h, "Language") && shown(&h, "Presentation"));
        for gone in ["Appearance", "Appearance Mode", "Dark Theme", "Options", "UI scale"] {
            assert!(!shown(&h, gone), "\"language\" must not show {gone}");
        }

        let h = render("theme");
        assert!(shown(&h, "Appearance") && shown(&h, "Light Theme") && shown(&h, "Dark Theme") && shown(&h, "Breeze dark"));
        for gone in ["Appearance Mode", "Presentation", "Language", "Options"] {
            assert!(!shown(&h, gone), "\"theme\" must not show {gone}");
        }

        let h = render("");
        for group in ["Appearance", "Appearance Mode", "Presentation", "Options", "UI scale"] {
            assert!(shown(&h, group), "an empty search shows {group}");
        }
    }

    /// Every generated preference label, section title and choice label has an entry in each
    /// language that claims complete menus (Japanese, Traditional Chinese, ...).
    #[test]
    fn preference_labels_are_translated() {
        let session = photocraft_engine::Session::new();
        let v: Value = serde_json::from_str(&session.prefs_to_json()).expect("prefs json");
        for lang in crate::i18n::Lang::all().filter(|l| l.complete_menus()) {
            let mut missing = Vec::new();
            for (sec, title) in SECTIONS {
                if !crate::i18n::has(lang, title) {
                    missing.push(title.to_string());
                }
                let Some(obj) = v.get(sec).and_then(Value::as_object) else { continue };
                for k in obj.keys() {
                    if sec == "interface" && k == "favoriteThemes" {
                        continue; // Rendered as star controls beside the theme choices.
                    }
                    let mut labels = vec![humanize(k)];
                    labels.extend(prefs::choices(&format!("{sec}.{k}")).into_iter().flatten().map(|c| choice_label(c)));
                    missing.extend(labels.into_iter().filter(|l| !crate::i18n::has(lang, l)));
                }
            }
            missing.sort();
            missing.dedup();
            assert!(missing.is_empty(), "{}: untranslated preference labels: {missing:#?}", lang.code());
        }
    }

    fn app_with_store() -> (PhotocraftApp, Arc<Mutex<Option<String>>>) {
        app_with_saved(None)
    }

    fn app_with_saved(text: Option<String>) -> (PhotocraftApp, Arc<Mutex<Option<String>>>) {
        let store: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(text));
        (app_on(&store, Arc::default()), store)
    }

    /// An app whose preferences live in `store`; writes fail while `fail` is set.
    fn app_on(store: &Arc<Mutex<Option<String>>>, fail: Arc<std::sync::atomic::AtomicBool>) -> PhotocraftApp {
        let (a, b) = (store.clone(), store.clone());
        let services = crate::Services {
            load_prefs: Some(Box::new(move || a.lock().unwrap().clone())),
            save_prefs: Some(Box::new(move |s: &str| {
                if fail.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err("storage is full".into());
                }
                *b.lock().unwrap() = Some(s.to_string());
                Ok(())
            })),
            ..Default::default()
        };
        PhotocraftApp::new(photocraft_engine::Session::new(), services)
    }

    fn stored(store: &Arc<Mutex<Option<String>>>) -> Value {
        serde_json::from_str(store.lock().unwrap().as_deref().unwrap_or("null")).unwrap()
    }

    #[test]
    fn merge_keeps_changes_made_elsewhere() {
        let base = json!({"a": {"x": 1, "y": 1}, "list": [1], "gone": 1, "kept": 1});
        // Here: a.x changed, `gone` removed, `new` added.
        let ours = json!({"a": {"x": 2, "y": 1}, "list": [1], "kept": 1, "new": 1});
        // There: a.y, `list` and `kept` changed, and a key this version doesn't know.
        let mut theirs = json!({"a": {"x": 1, "y": 3}, "list": [1, 2], "gone": 1, "kept": 5, "future": true});
        merge_changes(&base, &ours, &mut theirs);
        assert_eq!(theirs, json!({"a": {"x": 2, "y": 3}, "list": [1, 2], "kept": 5, "new": 1, "future": true}));
        // Arrays are leaves: a list changed here replaces the stored one whole.
        merge_changes(&json!({"list": [1]}), &json!({"list": [1, 3]}), &mut theirs);
        assert_eq!(theirs["list"], json!([1, 3]));
        // Nothing changed here: storage is left as it is, whatever it holds.
        let mut odd = json!({"x": [7]});
        merge_changes(&ours, &ours, &mut odd);
        assert_eq!(odd, json!({"x": [7]}));
        // A value that changed type here replaces the stored one.
        merge_changes(&json!({"a": 1}), &json!({"a": {"b": 2}}), &mut theirs);
        assert_eq!(theirs["a"], json!({"b": 2}));
    }

    #[test]
    fn saving_keeps_preferences_another_instance_changed() {
        let store = Arc::new(Mutex::new(None));
        let ctx = egui::Context::default();
        // Two browser tabs (or app windows) on the same storage.
        let (mut a, mut b) = (app_on(&store, Arc::default()), app_on(&store, Arc::default()));
        tick(&mut a, &ctx);
        tick(&mut b, &ctx);
        a.run("prefs.set", json!({"values": {"interface.theme": "studio"}})).unwrap();
        tick(&mut a, &ctx);
        assert_eq!(stored(&store)["interface"]["theme"], "studio");
        // The other one, never reloaded, saves an unrelated change: the theme stays.
        b.run("prefs.set", json!({"values": {"performance.historyStates": 12}})).unwrap();
        tick(&mut b, &ctx);
        let v = stored(&store);
        assert_eq!((v["interface"]["theme"].as_str(), v["performance"]["historyStates"].as_u64()), (Some("studio"), Some(12)));
        // And the first one's next save keeps the second one's change.
        a.run("prefs.set", json!({"values": {"interface.showTooltips": false}})).unwrap();
        tick(&mut a, &ctx);
        let v = stored(&store);
        assert_eq!(v["performance"]["historyStates"], 12);
        assert_eq!(v["interface"]["showTooltips"], false);
        // A value changed in both: the latest save wins.
        b.run("prefs.set", json!({"values": {"interface.theme": "classic"}})).unwrap();
        tick(&mut b, &ctx);
        assert_eq!(stored(&store)["interface"]["theme"], "classic");
        // Unreadable storage is overwritten with this instance's preferences.
        *store.lock().unwrap() = Some("not json".into());
        a.run("prefs.set", json!({"values": {"performance.historyStates": 30}})).unwrap();
        tick(&mut a, &ctx);
        let (mut c, _) = app_with_saved(store.lock().unwrap().clone());
        tick(&mut c, &ctx);
        assert_eq!(c.session.prefs().performance.history_states, 30);
        assert_eq!(c.session.prefs().interface.theme, Theme::Studio);
    }

    #[test]
    fn failed_preference_saves_are_retried() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let store = Arc::new(Mutex::new(None));
        let fail = Arc::new(AtomicBool::new(true));
        let mut app = app_on(&store, fail.clone());
        tick(&mut app, &egui::Context::default());
        app.run("prefs.set", json!({"values": {"interface.theme": "studio"}})).unwrap();
        // The write fails: reported, and retried after a backoff rather than marked saved.
        assert_eq!(persist(&mut app, 100.0), Some(SAVE_RETRY_S));
        assert!(app.ui.status.starts_with("Couldn't save preferences: storage is full") && app.ui.status_error);
        assert!(store.lock().unwrap().is_none());
        assert_eq!(persist(&mut app, 101.0), Some(1.0), "not due yet");
        // Storage works again: the pending change is written without another edit.
        fail.store(false, Ordering::Relaxed);
        assert_eq!(persist(&mut app, 102.0), None);
        assert_eq!(stored(&store)["interface"]["theme"], "studio");
        assert_eq!(app.ui.status, "Preferences saved");
        assert_eq!(persist(&mut app, 103.0), None, "nothing left to save");

        // A persistent failure backs off up to a cap and raises a notice once.
        fail.store(true, Ordering::Relaxed);
        app.run("prefs.set", json!({"values": {"interface.theme": "classic"}})).unwrap();
        let mut now = 200.0;
        let mut delays = Vec::new();
        for _ in 0..6 {
            let wait = persist(&mut app, now).unwrap();
            delays.push(wait);
            now += wait;
        }
        assert_eq!(delays, [2.0, 4.0, 8.0, 16.0, SAVE_RETRY_MAX_S, SAVE_RETRY_MAX_S]);
        assert_eq!(app.ui.notices.iter().filter(|n| n.error && n.title == "Preferences can't be saved").count(), 1);
        // A newer change retries sooner than the backoff, but not on every frame.
        app.run("prefs.set", json!({"values": {"interface.theme": "pro"}})).unwrap();
        assert_eq!(persist(&mut app, now - 29.0), Some(1.0));
        fail.store(false, Ordering::Relaxed);
        assert_eq!(persist(&mut app, now - 28.0), None);
        assert_eq!(stored(&store)["interface"]["theme"], "pro");
        assert!(app.ui.notices.is_empty(), "the notice goes once preferences are saved");
    }

    #[test]
    fn auto_scale_preserves_system_scale() {
        use prefs::UiScale::Auto;
        // Whatever the monitor, 4K included: an unscaled desktop stays at 100% (#1278), and a
        // fractional or below-100% system scale is the user's choice (#1072).
        for dpi in [0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0] {
            assert_eq!(display_scale(Auto, Some(dpi)), dpi);
        }
        assert_eq!(display_scale(Auto, None), 1.0);
        assert_eq!(display_scale(Auto, Some(f32::NAN)), 1.0);
        assert_eq!(display_scale(Auto, Some(-2.0)), 1.0);
    }

    #[test]
    fn scale_preferences_and_monitor_changes_apply_live() {
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        {
            let mut step = |physical: egui::Vec2, native: f32, expected: f32| {
                let mut input = egui::RawInput::default();
                let viewport = input.viewports.get_mut(&egui::ViewportId::ROOT).unwrap();
                viewport.native_pixels_per_point = Some(native);
                viewport.monitor_size = Some(physical / (native * ctx.zoom_factor()));
                ctx.run_ui(input, |ui| tick(&mut app, ui.ctx())).textures_delta.clear();
                // Scale changes take effect on the following pass.
                let mut input = egui::RawInput::default();
                let viewport = input.viewports.get_mut(&egui::ViewportId::ROOT).unwrap();
                viewport.native_pixels_per_point = Some(native);
                viewport.monitor_size = Some(physical / (native * ctx.zoom_factor()));
                ctx.run_ui(input, |ui| tick(&mut app, ui.ctx())).textures_delta.clear();
                assert!((ctx.pixels_per_point() - expected).abs() < 1e-4);
            };
            for _ in 0..4 {
                step(vec2(3840.0, 2160.0), 1.0, 1.0);
            }
            step(vec2(1920.0, 1080.0), 1.0, 1.0);
            step(vec2(3840.0, 2160.0), 1.5, 1.5);
            step(vec2(3840.0, 2160.0), 1.25, 1.25);
            step(vec2(2560.0, 1440.0), 1.75, 1.75);
            step(vec2(3840.0, 2160.0), 1.0, 1.0);
        }
        for (pref, expected) in [("200", 2.0), ("125", 1.25), ("150", 1.5), ("100", 1.0), ("95", 0.95), ("90", 0.9), ("85", 0.85), ("80", 0.8), ("auto", 1.5)] {
            app.run("prefs.set", json!({"values": {"interface.uiScale": pref}})).unwrap();
            let mut input = egui::RawInput::default();
            input.viewports.get_mut(&egui::ViewportId::ROOT).unwrap().native_pixels_per_point = Some(1.5);
            ctx.run_ui(input.clone(), |ui| tick(&mut app, ui.ctx())).textures_delta.clear();
            ctx.run_ui(input, |ui| tick(&mut app, ui.ctx())).textures_delta.clear();
            assert!((ctx.pixels_per_point() - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn ui_font_size_commands_follow_themes_and_dpi_and_reset_live() {
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, ThemeKind::Pro);
        let mut input = egui::RawInput::default();
        input.viewports.get_mut(&egui::ViewportId::ROOT).unwrap().native_pixels_per_point = Some(1.5);
        for (theme, scale, expected) in
            [("pro", "100", 1.0), ("studio", "auto", 1.5), ("studioLight", "200", 2.0), ("proMedium", "100", 1.0), ("classic", "auto", 1.5)]
        {
            app.run("prefs.set", json!({"values": {"interface.uiFontSize": "large", "interface.theme": theme, "interface.uiScale": scale}})).unwrap();
            for _ in 0..3 {
                ctx.run_ui(input.clone(), |ui| tick(&mut app, ui.ctx())).textures_delta.clear();
            }
            assert!((ctx.pixels_per_point() - expected).abs() < 1e-4);
            assert_eq!(ctx.fonts(|f| f.definitions().font_data["Inter"].tweak.scale), 16.0 / 12.0);
        }
        app.run("prefs.reset", json!({"path": "interface.uiFontSize"})).unwrap();
        for _ in 0..2 {
            ctx.run_ui(input.clone(), |ui| tick(&mut app, ui.ctx())).textures_delta.clear();
        }
        assert_eq!(ctx.fonts(|f| f.definitions().font_data["Inter"].tweak.scale), 1.0);
        assert!((ctx.pixels_per_point() - 1.5).abs() < 1e-4, "resetting font size keeps UI Scale");
    }

    #[test]
    fn recent_files_survive_a_restart_and_honour_the_count() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        app.push_recent("/work/a.psd");
        app.push_recent("/work/b.png");
        tick(&mut app, &ctx);
        let saved = store.lock().unwrap().clone().unwrap();
        // A new app instance (a restart) gets the list back, newest first.
        let (mut app2, _) = app_with_saved(Some(saved));
        tick(&mut app2, &ctx);
        assert_eq!(app2.ui.recent_files, vec!["/work/b.png".to_string(), "/work/a.psd".to_string()]);
        // Lowering "Recent File List Contains" shortens the menu at once; 0 turns it off.
        app2.run("prefs.set", json!({"values": {"fileHandling.recentFileCount": 1}})).unwrap();
        tick(&mut app2, &ctx);
        assert_eq!(app2.ui.recent_files, vec!["/work/b.png".to_string()]);
        app2.push_recent("/work/c.tif");
        assert_eq!(app2.session.prefs().file_handling.recent_files, vec!["/work/c.tif".to_string()]);
        app2.run("prefs.set", json!({"values": {"fileHandling.recentFileCount": 0}})).unwrap();
        tick(&mut app2, &ctx);
        app2.push_recent("/work/d.tif");
        assert!(app2.ui.recent_files.is_empty());
        // Clearing the list in the Preferences dialog (or by an agent) reaches the menu.
        app2.run("prefs.set", json!({"values": {"fileHandling.recentFileCount": 20, "fileHandling.recentFiles": ["/x.psd"]}})).unwrap();
        tick(&mut app2, &ctx);
        assert_eq!(app2.ui.recent_files, vec!["/x.psd".to_string()]);
        // A hostile count from a hand-edited preferences file is capped, not trusted.
        app2.run("prefs.set", json!({"values": {"fileHandling.recentFileCount": 100}})).unwrap();
        app2.session.prefs.edit(|p| p.file_handling.recent_file_count = u32::MAX);
        assert_eq!(app2.recent_cap(), 100);
    }

    #[test]
    fn brush_picker_view_survives_a_restart() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        app.ui.brush_picker_list.show_stroke = false;
        app.ui.brush_picker_list.scale = 0.4;
        crate::brush_picker::persist(&mut app, &ctx);
        tick(&mut app, &ctx);
        assert_eq!(stored(&store)["brushPicker"]["showStroke"], json!(false));
        // A restart gets the view back at startup, before the picker is ever opened.
        let saved = store.lock().unwrap().clone().unwrap();
        let (mut app2, _) = app_with_saved(Some(saved));
        tick(&mut app2, &ctx);
        assert!(!app2.ui.brush_picker_list.show_stroke);
        assert!(app2.ui.brush_picker_list.show_name && app2.ui.brush_picker_list.show_tip);
        assert!((app2.ui.brush_picker_list.scale - 0.4).abs() < 1e-6);
        // A hand-edited file: the scale is clamped to the slider's range and the last part comes
        // back on — an empty card would show no brush at all.
        let bad = json!({"brushPicker": {"showName": false, "showStroke": false, "showTip": false, "scale": 9.0}}).to_string();
        let (mut app3, _) = app_with_saved(Some(bad));
        tick(&mut app3, &ctx);
        assert_eq!(app3.ui.brush_picker_list.scale, *crate::brush_picker::SCALE_RANGE.end());
        assert!(app3.ui.brush_picker_list.show_name && app3.ui.brush_picker_list.show_stroke && app3.ui.brush_picker_list.show_tip);
    }

    #[test]
    fn tool_options_survive_a_restart() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        app.ui.tool_options.move_auto_select = false;
        app.ui.tool_options.feather = 12.0;
        crate::tool_memory::persist(&mut app, &ctx);
        tick(&mut app, &ctx);
        // Only what differs from the defaults is stored.
        assert_eq!(stored(&store)["toolOptions"], json!({"move_auto_select": false, "feather": 12.0}));
        let saved = store.lock().unwrap().clone().unwrap();
        let (mut app2, _) = app_with_saved(Some(saved));
        tick(&mut app2, &ctx);
        assert!(!app2.ui.tool_options.move_auto_select);
        assert_eq!(app2.ui.tool_options.feather, 12.0);
        assert_eq!(app2.ui.tool_options.tolerance, crate::state::ToolOptions::default().tolerance);
        // A hand-edited file: unreadable fields keep their defaults, the good ones still load.
        for bad in [json!("junk"), json!({"move_auto_select": "no", "feather": 1e300, "tolerance": 5.0, "unknown": 1})] {
            let (mut app3, _) = app_with_saved(Some(json!({"toolOptions": bad}).to_string()));
            tick(&mut app3, &ctx);
            let o = &app3.ui.tool_options;
            assert!(o.move_auto_select && o.feather == 0.0, "{bad}");
            assert_eq!(o.tolerance, if bad.is_object() { 5.0 } else { 32.0 });
        }
    }

    #[test]
    fn view_show_items_survive_a_restart() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        // Untouched: nothing is stored, so later default changes still reach the user.
        crate::view_cmds::persist(&mut app, &ctx);
        tick(&mut app, &ctx);
        assert!(stored(&store).get("viewExtras").is_none_or(Value::is_null));
        // The menu commands, as a user toggles them (#3762: Pixel Grid off).
        for id in ["view.show.pixelGrid", "view.show.grid", "view.snapTo.layers", "view.extras"] {
            crate::menus::invoke(&mut app, &ctx, id, json!({})).unwrap();
        }
        crate::view_cmds::persist(&mut app, &ctx);
        tick(&mut app, &ctx);
        assert_eq!(stored(&store)["viewExtras"], json!({"extras": false, "grid": true, "show": {"pixelGrid": false}, "snapTo": {"layers": false}}));
        // A restart gets them back.
        let saved = store.lock().unwrap().clone().unwrap();
        let (mut app2, _) = app_with_saved(Some(saved));
        tick(&mut app2, &ctx);
        let v = &app2.ui.view;
        assert!(!v.extras && !v.show.pixel_grid && !v.snap_to.layers && app2.ui.extras.grid);
        assert!(v.show.selection_edges && v.snap_to.guides && app2.ui.extras.guides, "the rest keep their defaults");
        // A hand-edited file: unknown keys and non-booleans keep the defaults, good ones still load.
        for bad in [json!("junk"), json!({"extras": 0, "show": {"pixelGrid": "no", "mesh": true, "bogus": true}, "snapTo": [1]})] {
            let (mut app3, _) = app_with_saved(Some(json!({"viewExtras": bad}).to_string()));
            tick(&mut app3, &ctx);
            let v = &app3.ui.view;
            assert!(v.extras && v.show.pixel_grid && v.snap_to.layers, "{bad}");
            assert_eq!(v.show.mesh, bad.is_object());
        }
    }

    #[test]
    fn show_tooltips_preferences_turn_tooltips_off() {
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        assert_eq!(ctx.global_style().interaction.tooltip_delay, crate::theme::TOOLTIP_DELAY);
        for path in ["interface.showTooltips", "tools.showTooltips"] {
            app.run("prefs.set", json!({"values": {path: false}})).unwrap();
            tick(&mut app, &ctx);
            assert!(ctx.global_style().interaction.tooltip_delay.is_infinite(), "{path} off hides tooltips");
            app.run("prefs.set", json!({"values": {path: true}})).unwrap();
            tick(&mut app, &ctx);
            assert_eq!(ctx.global_style().interaction.tooltip_delay, crate::theme::TOOLTIP_DELAY);
        }
    }

    #[test]
    fn enable_animations_off_stops_fades_and_scroll_glides_and_is_remembered() {
        let on = egui::Style::default().animation_time;
        let both = |ctx: &egui::Context| [egui::Theme::Dark, egui::Theme::Light].map(|t| ctx.style_of(t).animation_time);
        // Saved off: the first frame after a restart applies it to both appearance styles.
        let (mut app, _) = app_with_saved(Some(json!({"interface": {"enableAnimations": false}}).to_string()));
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        assert_eq!(both(&ctx), [0.0, 0.0]);
        assert_eq!(ctx.global_style().scroll_animation.duration.max, 0.0);
        // A theme change rebuilds the style; the next frame turns animations off again.
        app.apply_theme(&ctx, crate::theme::ThemeKind::StudioLight);
        tick(&mut app, &ctx);
        assert_eq!(both(&ctx), [0.0, 0.0]);
        app.run("prefs.set", json!({"values": {"interface.enableAnimations": true}})).unwrap();
        tick(&mut app, &ctx);
        assert_eq!(both(&ctx), [on, on]);
        assert!(on > 0.0 && ctx.global_style().scroll_animation.duration.max > 0.0);
        // A hand-edited value of the wrong type doesn't crash: the defaults (animations on) stay.
        let (mut bad, _) = app_with_saved(Some(json!({"interface": {"enableAnimations": "nope"}}).to_string()));
        let ctx2 = egui::Context::default();
        tick(&mut bad, &ctx2);
        assert_eq!(both(&ctx2), [on, on]);
    }

    #[test]
    fn unimplemented_preferences_are_hidden_from_the_dialog() {
        let values = prefs::Preferences::default().to_json();
        assert!(has_visible_fields(&values, "general"));
        assert!(has_visible_fields(&values, "fileHandling"));
        // Every setting of these sections is still unimplemented.
        for section in ["integrations", "scratchDisks"] {
            assert!(!has_visible_fields(&values, section), "{section}");
        }
        // "Fill new type layers with placeholder text" and "Use Escape to Commit" are live; other Type rows stay hidden.
        assert!(has_visible_fields(&values, "type"));
        assert!(!prefs::is_hidden("type.fillNewTypeLayersWithPlaceholder"));
        assert!(!prefs::is_hidden("type.useEscToCommit"));
        assert!(!prefs::is_hidden("type.smartQuotes"));
        assert!(!prefs::is_hidden("type.showFontNamesInEnglish"));
        // Rotate View with Trackpad and the pinch zoom are live; the other Enhanced Controls rows
        // stay hidden.
        assert!(has_visible_fields(&values, "enhancedControls"));
        assert!(!prefs::is_hidden("enhancedControls.rotateViewWithTrackpad"));
        assert!(!prefs::is_hidden("enhancedControls.zoomWithTrackpadPinch"));
        // Camera Raw Defaults shows only "Open in Camera Raw" so far.
        assert!(has_visible_fields(&values, "rawDefaults"));
        assert!(!prefs::is_hidden("rawDefaults.openInCameraRaw"));
        assert!(prefs::is_hidden("rawDefaults.applyAutoTone"));
        assert!(!prefs::is_hidden("general.autoShowHomeScreen"));
        assert!(!prefs::is_hidden("interface.uiScale"));
        assert!(!prefs::is_hidden("interface.uiFontSize"));
        // Hidden values still round-trip through the dialog untouched.
        let (mut app, _) = app_with_store();
        app.run("prefs.set", json!({"values": {"type.smartQuotes": false}})).unwrap();
        let id = open_preferences(&mut app, "type");
        let fields = app.ui.dialogs.iter().find(|d| d.id == id).map(|d| d.fields.clone()).unwrap();
        confirm(&mut app, &fields).unwrap();
        assert!(!app.session.prefs().type_.smart_quotes);
    }

    #[test]
    fn preferences_persist_through_services_and_theme_follows() {
        let (mut app, store) = app_with_store();
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        app.run("prefs.set", json!({"values": {"interface.theme": "studioLight", "performance.historyStates": 12}})).unwrap();
        tick(&mut app, &ctx);
        assert_eq!(app.ui.theme, ThemeKind::StudioLight);
        let saved = store.lock().unwrap().clone().unwrap();
        assert!(saved.contains("\"historyStates\": 12"));
        // A new app instance loads them.
        let (mut app2, store2) = app_with_saved(Some(saved));
        tick(&mut app2, &ctx);
        assert_eq!(app2.session.prefs().performance.history_states, 12);
        assert_eq!(app2.ui.theme, ThemeKind::StudioLight);
        // Picking a theme from the Window menu becomes the preference.
        crate::menus::invoke(&mut app2, &ctx, "window.theme.classic", json!({})).unwrap();
        tick(&mut app2, &ctx);
        assert_eq!(app2.session.prefs().interface.theme, Theme::Classic);
        assert!(store2.lock().unwrap().as_ref().unwrap().contains("classic"));
    }

    #[test]
    fn brush_preset_store_attaches_when_loaded_and_persists() {
        use photocraft_engine::preset_store::{MemBackend, open};
        let mem = MemBackend::default();
        let ctx = egui::Context::default();
        let (tx, rx) = std::sync::mpsc::channel();
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services { preset_store: Some(rx), ..Default::default() });
        // Still loading: presets made now are kept and written once the store arrives.
        tick(&mut app, &ctx);
        app.run("brush.presets.save", json!({"name": "Early"})).unwrap();
        assert!(mem.files.lock().unwrap().is_empty());
        tx.send(open(Box::new(mem.clone()))).unwrap();
        tick(&mut app, &ctx);
        assert!(app.services.preset_store.is_none() && app.session.preset_store.is_some());
        app.run("brush.presets.save", json!({"name": "Late"})).unwrap();
        let mut s2 = photocraft_engine::Session::new();
        assert!(s2.attach_preset_store(open(Box::new(mem))).is_empty());
        for n in ["Early", "Late"] {
            assert!(s2.tools.presets.iter().any(|p| p.name == n), "{n}");
        }
    }

    #[test]
    fn preferences_dialog_round_trip() {
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        let r = crate::menus::invoke(&mut app, &ctx, "edit.preferences.cursors", json!({})).unwrap();
        let id = r["dialog"].as_u64().unwrap();
        let d = app.ui.dialogs.iter().find(|d| d.id == id).unwrap().clone();
        assert_eq!(d.fields["section"], "cursors");
        let mut values = d.fields["values"].clone();
        values["cursors"]["painting"] = json!("fullSizeTip");
        values["unitsAndRulers"]["rulers"] = json!("inches");
        app.ui.dialog_mut(id).unwrap().fields.insert("values".into(), values);
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(app.session.prefs().cursors.painting, prefs::PaintingCursor::FullSizeTip);
        assert_eq!(app.session.prefs().units_and_rulers.rulers, prefs::Unit::Inches);
        // Invalid values are rejected and nothing changes.
        let id = open_preferences(&mut app, "performance");
        let mut values = app.ui.dialogs.iter().find(|d| d.id == id).unwrap().fields["values"].clone();
        values["performance"]["historyStates"] = json!(0);
        app.ui.dialog_mut(id).unwrap().fields.insert("values".into(), values);
        assert!(crate::dialogs::confirm(&mut app, id).is_err());
        assert_eq!(app.session.prefs().performance.history_states, 50);
    }

    #[test]
    fn preferences_apply_button_saves_without_closing_and_cancel_keeps_applied_values() {
        use egui_kittest::{
            Harness,
            kittest::{NodeT, Queryable},
        };

        let (mut app, store) = app_with_store();
        app.run("prefs.set", json!({"values": {"interface.language": "en", "type.smartQuotes": false}})).unwrap();
        let id = open_preferences(&mut app, "interface");
        let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        assert!(h.get_by_label("Apply").accesskit_node().is_disabled());

        let values = h.state_mut().ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap();
        values["interface"]["appearanceMode"] = json!("light");
        values["interface"]["lightTheme"] = json!("studioLight");
        values["interface"]["uiFontSize"] = json!("large");
        values["performance"]["historyStates"] = json!(12);
        h.run_steps(2);
        assert!(!h.get_by_label("Apply").accesskit_node().is_disabled());
        h.get_by_label("Apply").click();
        h.run_steps(4);

        assert_eq!(h.state().session.prefs().performance.history_states, 12);
        assert_eq!(h.state().ui.theme, ThemeKind::StudioLight);
        assert_eq!(h.ctx.fonts(|f| f.definitions().font_data["Inter"].tweak.scale), 16.0 / 12.0);
        assert!(!h.state().session.prefs().type_.smart_quotes, "hidden settings round-trip unchanged");
        let d = h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap();
        assert_eq!(d.fields["section"], "interface");
        assert_eq!(d.fields["values"]["performance"]["historyStates"], 12);
        assert!(h.get_by_label("Apply").accesskit_node().is_disabled());
        let saved: Value = serde_json::from_str(store.lock().unwrap().as_ref().unwrap()).unwrap();
        assert_eq!(saved["performance"]["historyStates"], 12);
        assert_eq!(saved["interface"]["appearanceMode"], "light");
        assert_eq!(saved["interface"]["lightTheme"], "studioLight");
        assert_eq!(saved["interface"]["uiFontSize"], "large");

        // Repeated Apply starts from the validated values, not the dialog's original snapshot.
        h.state_mut().ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["performance"]["historyStates"] = json!(22);
        h.run_steps(2);
        h.get_by_label("Apply").click();
        h.run_steps(4);
        assert_eq!(h.state().session.prefs().performance.history_states, 22);
        h.state_mut().ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["performance"]["historyStates"] = json!(33);
        h.state_mut().ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["interface"]["uiFontSize"] = json!("tiny");
        h.run_steps(2);
        h.get_by_label("Cancel").click();
        h.run_steps(4);
        assert!(h.state().ui.dialogs.iter().all(|d| d.id != id));
        assert_eq!(h.state().session.prefs().performance.history_states, 22);
        assert_eq!(h.ctx.fonts(|f| f.definitions().font_data["Inter"].tweak.scale), 16.0 / 12.0);
        let saved = store.lock().unwrap().clone().unwrap();
        let (mut restarted, _) = app_with_saved(Some(saved));
        tick(&mut restarted, &egui::Context::default());
        assert_eq!(restarted.session.prefs().performance.history_states, 22);
        assert_eq!(restarted.ui.theme, ThemeKind::StudioLight);
        assert_eq!(restarted.session.prefs().interface.ui_font_size, prefs::UiFontSize::Large);
    }

    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "macos")))]
    #[test]
    fn system_title_bar_explains_next_launch_and_preserves_apply_and_cancel() {
        use egui_kittest::{Harness, kittest::Queryable};

        let (mut app, store) = app_with_store();
        app.run("prefs.set", json!({"values": {"interface.language": "en"}})).unwrap();
        app.custom_titlebar = true;
        let id = open_preferences(&mut app, "interface");
        let mut h = Harness::builder().with_size(vec2(1280.0, 800.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        h.get_by_label("System title bar").scroll_to_me();
        h.run_steps(4);
        h.get_by_label("Applies at next launch.");
        for desired in [true, false] {
            h.get_by_label("System title bar").click();
            h.run_steps(4);
            assert_eq!(h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields["values"]["interface"]["systemTitleBar"], desired);
            assert_eq!(h.state().session.prefs().interface.system_title_bar, !desired, "the draft is not applied yet");
            h.get_by_label("Applies at next launch.");
            h.get_by_label("Apply").click();
            h.run_steps(4);
            assert_eq!(h.state().session.prefs().interface.system_title_bar, desired);
            assert_eq!(stored(&store)["interface"]["systemTitleBar"], desired);
            assert!(h.state().custom_titlebar, "do not change the current window or restart it");
            let (reloaded, _) = app_with_saved(store.lock().unwrap().clone());
            assert_eq!(reloaded.session.prefs().interface.system_title_bar, desired);
            h.get_by_label("Applies at next launch.");
        }
        h.get_by_label("System title bar").click();
        h.run_steps(4);
        h.get_by_label("Cancel").click();
        h.run_steps(4);
        assert!(h.state().ui.dialogs.iter().all(|d| d.id != id));
        assert!(!h.state().session.prefs().interface.system_title_bar);
        assert_eq!(stored(&store)["interface"]["systemTitleBar"], false);
        assert!(h.state().custom_titlebar);
    }

    #[test]
    fn next_launch_notice_does_not_mark_unrelated_interface_preferences() {
        use egui_kittest::{Harness, kittest::Queryable};
        let obj = json!({"showTooltips": true}).as_object().unwrap().clone();
        let mut h = Harness::new_ui_state(
            |ui, obj: &mut Map<String, Value>| section_fields(ui, "interface", obj, &[], crate::i18n::Lang::from_pref("en"), None, ""),
            obj,
        );
        h.run_steps(4);
        assert!(h.query_by_label("Applies at next launch.").is_none());
        h.get_by_label("Show tooltips").click();
        h.run_steps(4);
        assert_eq!(h.state()["showTooltips"], false);
        assert!(h.query_by_label("Applies at next launch.").is_none());
    }

    #[test]
    fn preferences_confirm_after_apply_commits_later_edits() {
        let (mut app, _) = app_with_store();
        let id = open_preferences(&mut app, "cursors");
        app.ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["cursors"]["painting"] = json!("fullSizeTip");
        apply(&mut app, id).unwrap();
        app.ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["unitsAndRulers"]["rulers"] = json!("inches");
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert!(app.ui.dialogs.iter().all(|d| d.id != id));
        assert_eq!(app.session.prefs().cursors.painting, prefs::PaintingCursor::FullSizeTip);
        assert_eq!(app.session.prefs().units_and_rulers.rulers, prefs::Unit::Inches);
    }

    #[test]
    fn preferences_apply_rejects_invalid_drafts_and_keeps_them_open() {
        let (mut app, _) = app_with_store();
        let before = app.session.prefs().to_json();
        for invalid in [json!(0), json!("invalid"), Value::Null] {
            let id = open_preferences(&mut app, "performance");
            app.ui.dialog_mut(id).unwrap().fields.get_mut("values").unwrap()["performance"]["historyStates"] = invalid;
            let draft = app.ui.dialog_mut(id).unwrap().fields.clone();
            assert!(apply(&mut app, id).is_err());
            assert_eq!(app.session.prefs().to_json(), before);
            assert_eq!(app.ui.dialog_mut(id).unwrap().fields, draft);
            app.ui.close_dialog(id);
        }
        let id = open_preferences(&mut app, "interface");
        app.ui.dialog_mut(id).unwrap().fields.remove("values");
        assert!(apply(&mut app, id).is_err());
        assert!(app.ui.dialog_mut(id).is_some());
        assert_eq!(app.session.prefs().to_json(), before);
    }

    #[test]
    fn apply_rejects_missing_and_non_preferences_dialogs() {
        let (mut app, _) = app_with_store();
        let before = app.session.prefs().to_json();
        assert!(apply(&mut app, u64::MAX).is_err());
        let id = open_shortcuts(&mut app, 0);
        assert!(apply(&mut app, id).is_err());
        assert!(app.ui.dialog_mut(id).is_some());
        let fields = json!({"__prefsui": "prefs", "values": preference_values(app.session.prefs())}).as_object().unwrap().clone();
        let id = app.ui.open_dialog(DialogKind::About, fields);
        assert!(apply(&mut app, id).is_err());
        assert!(app.ui.dialog_mut(id).is_some());
        assert_eq!(app.session.prefs().to_json(), before);
    }

    #[test]
    fn shortcut_dialog_moves_shortcuts_and_shell_dispatch_follows() {
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        let id = crate::menus::invoke(&mut app, &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
        // ⌘O (File › Open, a shell command) given to Layer › New › Layer.
        app.ui.dialog_mut(id).unwrap().fields.insert("overrides".into(), json!({"layer.new.layer": "Cmd+O"}));
        crate::dialogs::confirm(&mut app, id).unwrap();
        assert_eq!(effective_shortcut(&app, "layer.new.layer", None).as_deref(), Some("Cmd+O"));
        assert_eq!(effective_shortcut(&app, "file.open", Some("Cmd+O")), None, "taken from File › Open");
        // Menu items show the effective shortcut.
        let items = crate::menus::menu_items(&app);
        assert!(items.iter().any(|i| i.id == "layer.new.layer" && i.shortcut.as_deref() == Some("Cmd+O")));
        assert!(items.iter().any(|i| i.id == "file.open" && i.shortcut.is_none()));
        assert_eq!(shortcut_text(egui::Key::K, egui::Modifiers { command: true, shift: true, ..Default::default() }).as_deref(), Some("Cmd+Shift+K"));

        // Re-assigning a shortcut already held by a custom override takes it from the override.
        let id2 = crate::menus::invoke(&mut app, &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
        let mut cur_overrides = app.session.prefs().shortcuts.clone();
        cur_overrides.insert("layer.new.group".into(), "Cmd+O".into());
        app.ui.dialog_mut(id2).unwrap().fields.insert("overrides".into(), json!(cur_overrides));
        crate::dialogs::confirm(&mut app, id2).unwrap();
        assert_eq!(effective_shortcut(&app, "layer.new.group", None).as_deref(), Some("Cmd+O"));
        assert_eq!(effective_shortcut(&app, "layer.new.layer", None), None, "taken from layer.new.layer override");

        // OK with nothing changed keeps every shortcut where it is.
        let id3 = crate::menus::invoke(&mut app, &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
        let unchanged = app.session.prefs().shortcuts.clone();
        app.ui.dialog_mut(id3).unwrap().fields.insert("overrides".into(), json!(unchanged));
        crate::dialogs::confirm(&mut app, id3).unwrap();
        assert_eq!(app.session.prefs().shortcuts, unchanged);

        // Giving the shortcut back to Layer › New › Layer takes it from the group override; a
        // cleared override (empty) holds nothing and is left alone.
        let id4 = crate::menus::invoke(&mut app, &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
        let mut ov = app.session.prefs().shortcuts.clone();
        ov.insert("layer.new.layer".into(), "Cmd+O".into());
        app.ui.dialog_mut(id4).unwrap().fields.insert("overrides".into(), json!(ov));
        crate::dialogs::confirm(&mut app, id4).unwrap();
        assert_eq!(effective_shortcut(&app, "layer.new.layer", None).as_deref(), Some("Cmd+O"));
        assert_eq!(effective_shortcut(&app, "layer.new.group", None), None, "taken back from the group");
        assert_eq!(effective_shortcut(&app, "file.open", Some("Cmd+O")), None, "File › Open stays cleared");
    }

    #[test]
    fn shortcut_dialog_rejects_malformed_overrides() {
        // #956: a non-map `overrides` was read as "no overrides" and wiped every custom shortcut.
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        app.run("edit.keyboardShortcuts", json!({"set": {"file.new": "Ctrl+Shift+N"}})).unwrap();
        let before = app.session.prefs().shortcuts.clone();
        assert!(!before.is_empty());
        for bad in [json!("bogus"), json!(["file.new"]), json!({"file.new": 5})] {
            let id = crate::menus::invoke(&mut app, &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
            app.ui.dialog_mut(id).unwrap().fields.insert("overrides".into(), bad.clone());
            assert!(crate::dialogs::confirm(&mut app, id).is_err(), "{bad} accepted");
            assert_eq!(app.session.prefs().shortcuts, before, "{bad} changed the shortcuts");
        }

        // Drawing the open dialog keeps the malformed value for OK to reject, instead of
        // replacing it with the empty map it shows.
        let id = crate::menus::invoke(&mut app, &ctx, "edit.keyboardShortcuts", json!({})).unwrap()["dialog"].as_u64().unwrap();
        app.ui.dialog_mut(id).unwrap().fields.insert("overrides".into(), json!("bogus"));
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1280.0, 800.0)).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            app
        });
        h.run_steps(4);
        assert_eq!(h.state().ui.dialogs.iter().find(|d| d.id == id).unwrap().fields["overrides"], json!("bogus"));
        assert!(crate::dialogs::confirm(h.state_mut(), id).is_err());
        assert_eq!(h.state().session.prefs().shortcuts, before);
    }

    #[test]
    fn hidden_menu_items_disappear() {
        let (mut app, _) = app_with_store();
        app.run("edit.menus", json!({"hide": ["edit.fade"]})).unwrap();
        assert!(!crate::menus::menu_items(&app).iter().any(|i| i.id == "edit.fade"));
    }

    #[test]
    fn autosave_runs_for_dirty_documents() {
        type Saved = Arc<Mutex<Vec<(u64, u64)>>>;
        let saved: Saved = Arc::default();
        let s2 = saved.clone();
        let services = crate::Services {
            autosave: Some(Box::new(move |doc: &Arc<photocraft_doc::Document>, rev: u64, _path: Option<&str>| {
                s2.lock().unwrap().push((doc.id.0, rev));
                Ok(())
            })),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 8, "height": 8})).unwrap();
        tick(&mut app, &ctx);
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        assert!(saved.lock().unwrap().is_empty(), "clean documents are not autosaved");
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        assert_eq!(saved.lock().unwrap().len(), 1);
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        assert_eq!(saved.lock().unwrap().len(), 1, "unchanged since the last autosave");

        // Loaded copies can have the same persisted identity and dirty revision.
        let original = app.session.active().unwrap().doc.clone();
        let original_id = original.id;
        app.session.add_document(original.as_ref().clone(), None);
        app.run("edit.fill", json!({"color": "#0000ff"})).unwrap();
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        {
            let saved = saved.lock().unwrap();
            assert_eq!(saved.len(), 2, "both dirty copies need independent autosaves");
            assert_eq!(saved[0].0, original_id.0);
            assert_ne!(saved[0].0, saved[1].0, "recovery ownership must be distinct");
            assert_eq!(saved[0].1, saved[1].1, "the revisions must match to reproduce suppression");
        }
        app.run("prefs.set", json!({"path": "fileHandling.autosave", "value": false})).unwrap();
        app.run("edit.fill", json!({"color": "#00ff00"})).unwrap();
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        assert_eq!(saved.lock().unwrap().len(), 2, "autosave off");
    }

    #[test]
    fn recovered_documents_adopt_their_entries_until_saved_or_closed() {
        type Log = Arc<Mutex<Vec<String>>>;
        let log: Log = Arc::default();
        let (l1, l2, l3) = (log.clone(), log.clone(), log.clone());
        use photocraft_doc::{Color, ColorMode, Document, SampleType, Size};
        let doc = || Document::with_background("R", Size::new(4, 4), ColorMode::Rgb, SampleType::U8, Color::WHITE);
        let services = crate::Services {
            autosave: Some(Box::new(move |d: &Arc<Document>, _: u64, _: Option<&str>| {
                l1.lock().unwrap().push(format!("save {}", d.id.0));
                Ok(())
            })),
            discard_autosave: Some(Box::new(move |id: u64| l2.lock().unwrap().push(format!("discard {id}")))),
            recover: Some(Box::new(move || {
                ["a", "b"].map(|key| crate::Recoverable { key: key.into(), name: "R".into(), path: None, load: Box::new(move || Ok(doc())) }).into()
            })),
            adopt_autosave: Some(Box::new(move |id: u64, key: &str| l3.lock().unwrap().push(format!("adopt {id} {key}")))),
            ..Default::default()
        };
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), services);
        let ctx = egui::Context::default();
        tick(&mut app, &ctx);
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while app.session.has_jobs() && std::time::Instant::now() < deadline {
            crate::jobs_ui::tick(&mut app, &ctx);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(!app.session.has_jobs(), "recovery finished");
        let ids: Vec<u64> = app.session.documents().iter().map(|d| d.doc.id.0).collect();
        assert!(app.session.documents().iter().all(|d| d.is_dirty()), "recovered documents are unsaved");
        let take = || std::mem::take(&mut *log.lock().unwrap());
        assert_eq!(take(), [format!("adopt {} a", ids[0]), format!("adopt {} b", ids[1])]);
        // Their entries are current: nothing to autosave until they change.
        tick(&mut app, &ctx);
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        assert!(take().is_empty());
        // Closing one drops its entry; editing the other autosaves over its own.
        app.run("file.close", json!({"document": 0})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        autosave_now(&mut app);
        tick(&mut app, &ctx);
        assert_eq!(take(), [format!("discard {}", ids[0]), format!("save {}", ids[1])]);
    }

    #[test]
    fn edit_dialogs_open() {
        let (mut app, _) = app_with_store();
        let ctx = egui::Context::default();
        app.run("file.new", json!({"width": 32, "height": 32})).unwrap();
        app.sync_views();
        let d = crate::menus::invoke(&mut app, &ctx, "edit.colorSettings", json!({})).unwrap()["dialog"].as_u64().unwrap();
        assert_eq!(app.ui.dialogs.iter().find(|x| x.id == d).unwrap().fields["workingRgb"], "srgb");
        let note = app.ui.dialogs.iter().find(|x| x.id == d).unwrap().fields["__note"].as_str().unwrap().to_string();
        assert!(note.starts_with("Monitor profile in use: sRGB") && note.contains("fallback"), "{note}");
        assert!(crate::menus::invoke(&mut app, &ctx, "edit.fade", json!({})).is_err());
        app.run("select.rect", json!({"x": 4, "y": 4, "width": 8, "height": 8})).unwrap();
        let d = crate::menus::invoke(&mut app, &ctx, "edit.contentAwareFill", json!({})).unwrap()["dialog"].as_u64().unwrap();
        assert_eq!(app.ui.dialogs.iter().find(|x| x.id == d).unwrap().fields["__preview"], true);
        crate::dialogs::confirm(&mut app, d).unwrap();
        let r = crate::menus::invoke(&mut app, &ctx, "edit.contentAwareScale", json!({})).unwrap();
        assert_eq!(r["transform"]["rect"], json!([4.0, 4.0, 12.0, 12.0]));
        crate::transform_tool::cancel(&mut app);
        let d = crate::menus::invoke(&mut app, &ctx, "edit.presets.presetManager", json!({})).unwrap()["dialog"].as_u64().unwrap();
        assert!(crate::dialogs::confirm(&mut app, d).is_ok());
    }
}
