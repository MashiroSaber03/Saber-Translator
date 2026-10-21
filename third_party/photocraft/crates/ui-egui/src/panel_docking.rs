//! Application-owned panel IDs and commands around the shared docking layout.

use crate::dock::Group;
use craft_ui::docking::{Action, Floating, Layout, Node, Placement, Zone};
use craft_ui::layout::{SplitAxis, SplitSize};
use serde_json::{Value, json};

use crate::PhotocraftApp;

const PANELS: &[(&str, &str, Group)] = &[
    ("color", "Color", Group::Color),
    ("swatches", "Swatches", Group::Color),
    ("gradients", "Gradients", Group::Color),
    ("patterns", "Patterns", Group::Color),
    ("properties", "Properties", Group::Properties),
    ("adjustments", "Adjustments", Group::Properties),
    ("character", "Character", Group::Character),
    ("paragraph", "Paragraph", Group::Character),
    ("navigator", "Navigator", Group::Navigator),
    ("histogram", "Histogram", Group::Navigator),
    ("info", "Info", Group::Navigator),
    ("history", "History", Group::History),
    ("actions", "Actions", Group::History),
    ("layerComps", "Layer Comps", Group::History),
    ("layers", "Layers", Group::Layers),
    ("channels", "Channels", Group::Layers),
    ("paths", "Paths", Group::Layers),
];
#[derive(Clone, Default)]
struct LegacyDrag {
    generation: u64,
    layout: Option<Layout<String>>,
    tabs: crate::state::DockTabs,
    hidden: std::collections::BTreeMap<String, craft_ui::docking::Location<String>>,
}

fn cancel_legacy_drag(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    let key = egui::Id::new("photocraft-panel-docking").with("legacy-origin");
    let previous = ctx.data_mut(|data| data.remove_temp::<LegacyDrag>(key));
    if let Some(previous) = previous {
        craft_ui::docking::cancel_drag::<String>(ctx, egui::Id::new("photocraft-panel-docking"));
        if previous.generation != app.ui.docking_generation {
            return false;
        }
        app.ui.dock_tabs = previous.tabs;
        app.ui.docking = previous.layout;
        app.ui.docking_hidden = previous.hidden;
        true
    } else {
        false
    }
}

pub(crate) fn normalize(raw: &str) -> Option<&'static str> {
    PANELS.iter().find(|(id, label, _)| id.eq_ignore_ascii_case(raw.trim()) || label.eq_ignore_ascii_case(raw.trim())).map(|p| p.0)
}

pub(crate) fn valid(layout: &Layout<String>) -> bool {
    layout.validate().is_ok() && layout.panels().iter().all(|id| normalize(id) == Some(id.as_str()))
}

fn legacy(app: &PhotocraftApp) -> Layout<String> {
    legacy_layout(app, false)
}

fn legacy_layout(app: &PhotocraftApp, all: bool) -> Layout<String> {
    let pro = matches!(app.ui.theme, crate::theme::ThemeKind::Pro | crate::theme::ThemeKind::ProMedium);
    let mut layout = Layout::default();
    for group in app.ui.dock.order().into_iter().rev().filter(|g| all || g.shown(&app.ui.panels)) {
        let labels = group.tabs(pro);
        let visible = if all { group.tabs(pro).iter().copied().enumerate().collect() } else { app.ui.dock.visible_tabs(group, pro) };
        let panels: Vec<String> = visible.iter().filter_map(|(_, label)| normalize(label).map(str::to_owned)).collect();
        if panels.is_empty() {
            continue;
        }
        let selected = match group {
            Group::Color => app.ui.dock_tabs.color,
            Group::Properties => app.ui.dock_tabs.properties,
            Group::Character => app.ui.dock_tabs.character,
            Group::Navigator => app.ui.dock_tabs.navigator,
            Group::History => app.ui.dock_tabs.history,
            Group::Layers => app.ui.dock_tabs.layers,
        };
        let active = labels.get(selected).and_then(|label| normalize(label)).and_then(|id| panels.iter().position(|p| p == id)).unwrap_or(0);
        if group == Group::Properties && !pro {
            layout.floating.push(Floating { panels, active, rect: [120.0, 160.0, 320.0, 440.0] });
            continue;
        }
        let node = Node::Tabs { panels, active };
        layout.root = Some(match layout.root.take() {
            Some(second) => Node::Split {
                axis: SplitAxis::Vertical,
                size: SplitSize::FixedFirst(app.ui.dock.height(group)),
                first: Box::new(node),
                second: Box::new(second),
            },
            None => node,
        });
    }
    layout
}

// Native collapse flags remain app-owned. An intact native tab group keeps its collapsed
// presentation when the workspace first becomes customizable; mixed custom groups are expanded.
fn collapsed_members(app: &PhotocraftApp, layout: &Layout<String>) -> std::collections::HashSet<String> {
    let mut collapsed: std::collections::HashSet<String> = app.ui.docking_collapsed.iter().cloned().collect();
    let mut pending: Vec<_> = layout.root.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Split { first, second, .. } => pending.extend([first.as_ref(), second.as_ref()]),
            Node::Tabs { panels, .. } => {
                let group = panels.first().and_then(|id| PANELS.iter().find(|p| p.0 == id)).map(|p| p.2);
                if let Some(group) = group.filter(|group| app.ui.dock.is_collapsed(*group)) {
                    let pro = matches!(app.ui.theme, crate::theme::ThemeKind::Pro | crate::theme::ThemeKind::ProMedium);
                    let native: Vec<_> = app.ui.dock.visible_tabs(group, pro).iter().filter_map(|(_, name)| normalize(name)).collect();
                    if panels.len() == native.len() && panels.iter().all(|panel| native.contains(&panel.as_str())) {
                        collapsed.extend(panels.iter().cloned());
                    }
                }
            }
            _ => {}
        }
    }
    collapsed
}

fn first_root(layout: &Layout<String>, except: &str) -> Option<String> {
    let root = Layout { root: layout.root.clone(), floating: Vec::new() };
    root.panels().into_iter().find(|p| p.as_str() != except).cloned()
}

fn ensure(layout: &mut Layout<String>, panel: &str) -> Result<(), String> {
    let anchor = first_root(layout, panel);
    layout.apply(Action::Open { panel: panel.into(), anchor }).map_err(|e| e.to_string())
}

fn number(params: &Value, key: &str, default: f32) -> Result<f32, String> {
    match params.get(key) {
        None => Ok(default),
        Some(v) => v.as_f64().filter(|v| v.is_finite() && v.abs() <= 1_000_000.0).map(|v| v as f32).ok_or_else(|| format!("{key} must be a finite coordinate")),
    }
}

fn placement(params: &Value) -> Result<Placement<String>, String> {
    if let Some(before) = params.get("before") {
        let id = before.as_str().and_then(normalize).ok_or("before must name a panel")?;
        return Ok(Placement::Tab { before: Some(id.into()) });
    }
    let zone = match params.get("zone") {
        None => "tab",
        Some(value) => value.as_str().ok_or("zone must be a string")?,
    };
    Ok(match zone {
        "tab" => Placement::Tab { before: None },
        "center" => Placement::Split(Zone::Center),
        "left" => Placement::Split(Zone::Left),
        "right" => Placement::Split(Zone::Right),
        "top" => Placement::Split(Zone::Top),
        "bottom" => Placement::Split(Zone::Bottom),
        _ => return Err("zone must be tab, center, left, right, top or bottom".into()),
    })
}

pub(crate) fn command(app: &mut PhotocraftApp, id: &str, params: &Value) -> Option<Result<Value, String>> {
    if id == "window.panel.group" {
        return Some(group_command(app, params));
    }
    if id == "window.panel.layout" {
        return Some(
            params
                .get("action")
                .ok_or_else(|| "action is required".to_string())
                .and_then(|value| serde_json::from_value::<Action<String>>(value.clone()).map_err(|error| error.to_string()))
                .and_then(|action| apply_action(app, action)),
        );
    }
    if let Some(layout) = &app.ui.docking
        && let Some(panel) = id.strip_prefix("window.panel.").or_else(|| id.strip_prefix("window.toggle.")).and_then(normalize)
    {
        let showing = layout.contains(&panel.to_string());
        let show = match params.get("show").or_else(|| params.get("on")) {
            None => !showing || !panel_active(layout, panel) || collapsed_members(app, layout).contains(panel),
            Some(value) => match value.as_bool() {
                Some(show) => show,
                None => return Some(Err("show must be a boolean".into())),
            },
        };
        return Some(change(app, if show { "activate" } else { "close" }, &json!({"panel":panel})));
    }
    let operation = match id {
        "window.panel.move" => "move",
        "window.panel.float" => "float",
        "window.panel.dock" => "dock",
        "window.panel.activate" => "activate",
        "window.panel.close" => "close",
        "window.panel" if app.ui.docking.is_some() => "activate",
        _ => return None,
    };
    Some(change(app, operation, params))
}

pub(crate) fn checked(app: &PhotocraftApp, id: &str) -> Option<bool> {
    let layout = app.ui.docking.as_ref()?;
    let panel = id.strip_prefix("window.panel.").or_else(|| id.strip_prefix("window.toggle.")).and_then(normalize)?;
    Some(layout.contains(&panel.to_string()))
}

// UI output and automation share this atomic dispatcher. The application owns the registry
// and saved return locations; craft-ui owns tree validation and structural changes.
fn apply_action(app: &mut PhotocraftApp, action: Action<String>) -> Result<Value, String> {
    if app.session.prefs().workspace_locked && !matches!(action, Action::Activate { .. }) {
        return Err("workspace is locked".into());
    }
    let registered = |id: &String| normalize(id) == Some(id.as_str());
    let known = match &action {
        Action::Move { panel, anchor, placement } | Action::OpenAt { panel, anchor, placement } => {
            registered(panel)
                && registered(anchor)
                && match placement {
                    Placement::Tab { before: Some(id) } => registered(id),
                    _ => true,
                }
        }
        Action::Open { panel, anchor } => registered(panel) && anchor.as_ref().is_none_or(registered),
        Action::Float { panel, .. }
        | Action::Close { panel }
        | Action::Activate { panel }
        | Action::MoveFloating { panel, .. }
        | Action::SetStackOpen { panel, .. }
        | Action::ResizeStack { panel, .. } => registered(panel),
        Action::ResizeSplit { path, .. } => path.len() <= craft_ui::docking::MAX_DEPTH,
    };
    if !known {
        return Err("docking action contains an unknown panel or an excessive split path".into());
    }
    if app.ui.docking.as_ref().is_some_and(|layout| !valid(layout)) {
        return Err("saved panel layout is invalid; reset the workspace first".into());
    }
    let mut layout = app.ui.docking.clone().unwrap_or_else(|| legacy(app));
    let returning = match &action {
        Action::Close { panel } => layout.location(panel).ok().map(|location| (panel.clone(), location)),
        Action::Float { panel, .. } => layout.location(panel).ok().filter(|location| location.floating.is_none()).map(|location| (panel.clone(), location)),
        _ => None,
    };
    let reveal = match &action {
        Action::Activate { panel } => Some(panel.clone()),
        _ => None,
    };
    let activate = match &action {
        Action::Activate { panel } => PANELS.iter().find(|p| p.0 == panel).map(|p| p.2),
        _ => None,
    };
    layout.apply(action).map_err(|error| error.to_string())?;
    if let Some(group) = activate {
        app.ui.dock.set_collapsed(group, false);
    }
    if let Some((panel, location)) = returning {
        app.ui.docking_hidden.insert(panel, location);
    }
    if app.ui.docking.is_none() {
        app.ui.docking_hidden.extend(migration_hidden(app, &layout));
    }
    app.ui.docking = Some(layout);
    app.ui.panels.dock = true;
    sync_visibility(app);
    if let Some(panel) = reveal {
        app.ui.docking_reveal = Some(panel.clone());
        reveal_group(app, &panel);
    }
    Ok(json!({"layout": app.ui.docking}))
}

fn change(app: &mut PhotocraftApp, operation: &str, params: &Value) -> Result<Value, String> {
    if app.session.prefs().workspace_locked && operation != "activate" {
        return Err("workspace is locked".into());
    }
    let panel = params.get("panel").and_then(Value::as_str).and_then(normalize).ok_or("panel must name a registered panel")?;
    if app.ui.docking.as_ref().is_some_and(|layout| !valid(layout)) {
        return Err("saved panel layout is invalid; reset the workspace first".into());
    }
    let mut layout = app.ui.docking.clone().unwrap_or_else(|| legacy(app));
    let migrated = if app.ui.docking.is_none() { migration_hidden(app, &layout) } else { app.ui.docking_hidden.clone() };
    let mut hidden: std::collections::BTreeMap<_, _> =
        migrated.iter().filter(|(id, _)| normalize(id).is_some()).take(64).map(|(id, at)| (id.clone(), at.clone())).collect();
    let mut restored = false;
    if operation == "dock"
        && params.get("anchor").or_else(|| params.get("onto")).is_none()
        && let Some(location) = hidden.get(panel).filter(|location| {
            location.floating.is_none()
                && location.anchor.as_ref().is_none_or(|anchor| Layout { root: layout.root.clone(), floating: Vec::new() }.contains(anchor))
        })
    {
        let mut candidate = layout.clone();
        if candidate.contains(&panel.to_string()) {
            candidate.apply(Action::Close { panel: panel.into() }).map_err(|e| e.to_string())?;
        }
        if candidate.restore(panel.into(), location).is_ok() {
            layout = candidate;
            restored = true;
            hidden.remove(panel);
        }
    }
    match operation {
        "dock" if restored => {}
        "float" => {
            let rect = [number(params, "x", 400.0)?, number(params, "y", 140.0)?, number(params, "width", 300.0)?, number(params, "height", 400.0)?];
            ensure(&mut layout, panel)?;
            if let Ok(location) = layout.location(&panel.to_string())
                && location.floating.is_none()
            {
                hidden.insert(panel.into(), location);
            }
            layout.apply(Action::Float { panel: panel.into(), rect }).map_err(|e| e.to_string())?;
        }
        "move" | "dock" => {
            if operation == "move" && params.get("anchor").or_else(|| params.get("onto")).is_none() {
                return Err("move requires a destination panel".into());
            }
            let anchor = match params.get("anchor").or_else(|| params.get("onto")) {
                Some(v) => Some(v.as_str().and_then(normalize).ok_or("anchor must name a registered panel")?.to_string()),
                None => first_root(&layout, panel),
            };
            let placement = placement(params)?;
            if let Some(anchor) = anchor {
                ensure(&mut layout, panel)?;
                layout.apply(Action::Move { panel: panel.into(), anchor, placement }).map_err(|e| e.to_string())?;
            } else if operation == "dock" {
                if layout.contains(&panel.to_string()) {
                    layout.apply(Action::Close { panel: panel.into() }).map_err(|e| e.to_string())?;
                }
                layout.apply(Action::Open { panel: panel.into(), anchor: None }).map_err(|e| e.to_string())?;
            } else {
                return Err("move requires an existing destination panel".into());
            }
        }
        "close" => {
            hidden.insert(panel.into(), layout.location(&panel.to_string()).map_err(|e| e.to_string())?);
            layout.apply(Action::Close { panel: panel.into() }).map_err(|e| e.to_string())?;
        }
        _ => {
            if !layout.contains(&panel.to_string()) && hidden.contains_key(panel) {
                // A missing neighbor is allowed: the normal visible dock becomes the fallback.
                restore_hidden(&mut layout, &mut hidden, panel);
            }
            ensure(&mut layout, panel)?;
            layout.apply(Action::Activate { panel: panel.into() }).map_err(|e| e.to_string())?;
            hidden.remove(panel);
        }
    }
    app.ui.docking = Some(layout);
    app.ui.docking_hidden = hidden;
    app.ui.panels.dock = true;
    sync_visibility(app);
    if operation == "activate"
        && let Some((_, _, group)) = PANELS.iter().find(|p| p.0 == panel)
    {
        app.ui.dock.set_collapsed(*group, false);
        app.ui.docking_reveal = Some(panel.to_owned());
        reveal_group(app, panel);
    }
    Ok(json!({"panel": panel, "layout": app.ui.docking}))
}

pub(crate) fn show(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> bool {
    let area = egui::Id::new("photocraft-panel-docking");
    let stale_origin =
        ui.ctx().data_mut(|data| data.get_temp::<LegacyDrag>(area.with("legacy-origin")).is_some_and(|origin| origin.generation != app.ui.docking_generation));
    let changed_workspace = ui.ctx().data_mut(|data| {
        let previous = data.get_temp::<u64>(area.with("generation"));
        data.insert_temp(area.with("generation"), app.ui.docking_generation);
        previous.is_some_and(|generation| generation != app.ui.docking_generation)
    });
    if changed_workspace || stale_origin {
        ui.ctx().data_mut(|data| data.insert_temp(area.with("legacy-suppressed"), true));
        craft_ui::docking::cancel_drag::<String>(ui.ctx(), area);
        ui.ctx().data_mut(|data| data.remove::<LegacyDrag>(area.with("legacy-origin")));
    }

    let cancelled =
        ui.input(|input| !input.focused || input.key_pressed(egui::Key::Escape) || (!input.pointer.primary_down() && !input.pointer.any_released()));
    if cancelled && cancel_legacy_drag(app, ui.ctx()) {
        ui.ctx().data_mut(|data| data.insert_temp(area.with("legacy-suppressed"), true));
        if app.ui.docking.is_none() {
            return false;
        }
    }
    let Some(layout) = app.ui.docking.as_ref() else { return false };
    if !valid(layout) {
        app.ui.docking = None;
        let message = "Invalid panel layout; restored the default dock";
        app.ui.status = tl!(message).into();
        return false;
    }
    let layout = layout.clone();
    let reveal = app.ui.docking_reveal.take();
    let t = crate::theme::Tokens::get(ui.ctx());
    let locked = app.session.prefs().workspace_locked;
    let collapsed = collapsed_members(app, &layout);
    let requested_width = ui.ctx().data_mut(|data| data.remove_temp::<f32>(crate::panels::dock_width_id()));
    let mut commands = Vec::new();
    let mut targets = Vec::new();
    let mut draw = |ui: &mut egui::Ui| {
        let mut style = craft_ui::docking::DockStyle::from_ui(ui);
        style.tab_height = if t.pro { 28.0 } else { 40.0 };
        style.min_pane = 0.0;
        style.background = t.card;
        // Studio themes leave the strip transparent over the dock's card; a floating group has no
        // card behind it, so the canvas showed through its titles (#4509).
        style.tab_background = if t.tab_strip.a() == 0 { t.card } else { t.tab_strip };
        style.active_background = t.card;
        style.text = t.text;
        style.inactive_text = t.text_dim;
        style.border = egui::Stroke::new(1.0, t.card_border);
        style.accent = t.accent;
        style.float_label = tl!("Float Panel").to_string();
        style.close_label = tl!("Close Panel").to_string();
        style.move_label = tl!("Group with").to_string();
        style.panels_label = tl!("Panels").to_string();
        style.resize_label = tl!("Resize panels").to_string();
        style.resize_window_label = tl!("Resize panel window").to_string();
        let mut area = craft_ui::docking::DockArea::new(egui::Id::new("photocraft-panel-docking"));
        if let Some(panel) = &reveal {
            area = area.raise_panel(panel.clone());
        }
        area.show_customized(
            ui,
            &layout,
            &style,
            |id| PANELS.iter().find(|p| p.0 == id).map(|p| tl!(p.1).to_string()).unwrap_or_else(|| id.clone()),
            |_| craft_ui::docking::Permissions { movable: !locked, floatable: !locked, closable: !locked, accepts_tabs: !locked },
            |id| {
                if collapsed.contains(id) {
                    craft_ui::docking::PanelLimits { min: egui::vec2(220.0, 0.0), max: egui::vec2(1_000_000.0, 0.0) }
                } else {
                    // Layers owns its row scroll and footer; leave room for its fixed controls
                    // plus a scrolling row area instead of clipping the footer after a split.
                    let height = if id == "layers" { 220.0 } else { 80.0 };
                    craft_ui::docking::PanelLimits { min: egui::vec2(220.0, height), ..Default::default() }
                }
            },
            &mut PanelContent { app, collapsed: &collapsed, commands: &mut commands, targets: &mut targets },
        )
    };
    let output = if layout.root.is_some() {
        let mut panel = egui::Panel::right("shared-panel-dock")
            .default_size(if t.pro { 290.0 } else { 300.0 })
            .size_range(crate::panels::DOCK_WIDTH)
            .resizable(!locked)
            .frame(egui::Frame::NONE.fill(t.card));
        if let Some(width) = requested_width {
            panel = panel.exact_size(width);
        }
        panel.show(ui, &mut draw).inner
    } else {
        draw(ui)
    };
    if let Some(error) = output.error {
        app.ui.status = error.to_string();
    }
    if ui.input(|input| input.pointer.any_released()) {
        if !output.actions.iter().any(|action| matches!(action, Action::Move { .. } | Action::Float { .. })) && cancel_legacy_drag(app, ui.ctx()) {
            return app.ui.docking.is_some();
        }
        ui.ctx().data_mut(|data| data.remove::<LegacyDrag>(egui::Id::new("photocraft-panel-docking").with("legacy-origin")));
    }
    if let Some(panel) = output.group_drag.as_ref()
        && let Some(pointer) = ui.ctx().pointer_interact_pos()
    {
        let members = group_members(&layout, panel);
        let eligible: Vec<_> = targets.iter().filter(|target| !members.contains(&target.anchor)).cloned().collect();
        if let Some(proposal) = craft_ui::docking::resolve_drop(panel, pointer, &eligible) {
            ui.painter().rect_stroke(proposal.preview, 0.0, egui::Stroke::new(2.0, t.accent), egui::StrokeKind::Inside);
        }
    }
    if let Some((panel, pointer)) = output.group_drop {
        let members = group_members(&layout, &panel);
        let eligible: Vec<_> = targets.iter().filter(|target| !members.contains(&target.anchor)).cloned().collect();
        if let Some(proposal) = craft_ui::docking::resolve_drop(&panel, pointer, &eligible) {
            commands.push(("window.panel.group", json!({"panel":panel,"operation":"move","anchor":proposal.anchor,"placement":proposal.placement})));
        }
    }
    for action in output.actions {
        if let Err(error) = app.run("window.panel.layout", json!({"action": action})) {
            app.ui.status = error;
        }
    }
    for (id, params) in commands {
        if let Err(error) = app.run(id, params) {
            app.ui.status = error;
        }
    }
    crate::dock::persist(app, ui.ctx());
    true
}

pub(crate) fn legacy_tab(app: &mut PhotocraftApp, ui: &mut egui::Ui, panel: &str, response: &egui::Response) {
    let suppressed = egui::Id::new("photocraft-panel-docking").with("legacy-suppressed");
    if ui.input(|input| input.pointer.button_pressed(egui::PointerButton::Primary)) {
        ui.ctx().data_mut(|data| data.remove::<bool>(suppressed));
    }
    if ui.ctx().data(|data| data.get_temp::<bool>(suppressed)).unwrap_or(false) || app.session.prefs().workspace_locked {
        return;
    }
    let Some(panel) = normalize(panel) else { return };
    // Native tab reordering and vertical whole-group moves stay in the dock.
    // Crossing the dock column horizontally detaches through the shared layout.
    let strip = egui::Rect::from_min_max(egui::pos2(ui.max_rect().left(), response.rect.top()), egui::pos2(ui.max_rect().right(), response.rect.bottom()));
    if response.dragged()
        && ui.input(|input| input.focused && input.pointer.primary_down() && !input.key_pressed(egui::Key::Escape))
        && ui.ctx().pointer_interact_pos().is_some_and(|position| position.x < strip.left() - 24.0 || position.x > strip.right() + 24.0)
        && let Some(id) = normalize(panel)
    {
        let mut layout = app.ui.docking.clone().unwrap_or_else(|| legacy(app));
        let bounds = ui.max_rect();
        let source = egui::Rect::from_min_size(
            egui::pos2(bounds.left(), response.rect.top()),
            egui::vec2(bounds.width().clamp(240.0, 600.0), bounds.height().clamp(200.0, 600.0)),
        );
        if ensure(&mut layout, id).is_ok() && craft_ui::docking::begin_drag(ui.ctx(), egui::Id::new("photocraft-panel-docking"), id.to_string(), source) {
            ui.ctx().data_mut(|data| {
                data.insert_temp(
                    egui::Id::new("photocraft-panel-docking").with("legacy-origin"),
                    LegacyDrag {
                        generation: app.ui.docking_generation,
                        layout: app.ui.docking.clone(),
                        tabs: app.ui.dock_tabs,
                        hidden: app.ui.docking_hidden.clone(),
                    },
                )
            });
            if app.ui.docking.is_none() {
                app.ui.docking_hidden = migration_hidden(app, &layout);
            }
            app.ui.docking = Some(layout);
        }
    }
}

fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, id: &str) {
    let Some((_, label, group)) = PANELS.iter().find(|p| p.0 == id) else { return };
    let pro = crate::theme::Tokens::get(ui.ctx()).pro;
    let Some(tab) = group.tabs(pro).iter().position(|name| name == label) else { return };
    *group.tab_mut(&mut app.ui.dock_tabs) = tab;
    if group.scrolls_itself(tab) {
        crate::panels::dock_body(app, ui, *group, tab);
    } else {
        egui::ScrollArea::vertical()
            .id_salt(("shared-panel-body", id))
            .auto_shrink([false, false])
            .show(ui, |ui| crate::panels::dock_body(app, ui, *group, tab));
    }
    let selected = *group.tab_mut(&mut app.ui.dock_tabs);
    if selected != tab
        && let Some(label) = group.tabs(pro).get(selected)
    {
        let _ = app.run("window.panel.activate", json!({"panel":label}));
    }
}

#[cfg(test)]
#[path = "panel_docking_tests.rs"]
mod tests;

/// Return positions for panels already hidden in a legacy workspace.
fn migration_hidden(app: &PhotocraftApp, visible: &Layout<String>) -> std::collections::BTreeMap<String, craft_ui::docking::Location<String>> {
    let full = legacy_layout(app, true);
    PANELS
        .iter()
        .filter(|(id, _, _)| !visible.contains(&id.to_string()))
        .filter_map(|(id, _, _)| full.location(&id.to_string()).ok().map(|at| ((*id).to_owned(), at)))
        .collect()
}

fn group_members(layout: &Layout<String>, panel: &str) -> Vec<String> {
    let mut pending: Vec<_> = layout.root.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Split { first, second, .. } => pending.extend([second.as_ref(), first.as_ref()]),
            Node::Tabs { panels, .. } if panels.iter().any(|id| id == panel) => return panels.clone(),
            Node::Stack { entries } if entries.iter().any(|entry| entry.panel == panel) => return vec![panel.to_owned()],
            _ => {}
        }
    }
    layout.floating.iter().find(|group| group.panels.iter().any(|id| id == panel)).map(|group| group.panels.clone()).unwrap_or_default()
}

fn reveal_group(app: &mut PhotocraftApp, panel: &str) {
    if let Some(layout) = &app.ui.docking {
        let members = group_members(layout, panel);
        app.ui.docking_collapsed.retain(|id| !members.contains(id));
        for id in members {
            if let Some((_, _, group)) = PANELS.iter().find(|p| p.0 == id) {
                app.ui.dock.set_collapsed(*group, false);
            }
        }
    }
}

pub(crate) fn rail_on(app: &PhotocraftApp, group: Group) -> bool {
    app.ui.docking.as_ref().is_some_and(|layout| {
        let collapsed = collapsed_members(app, layout);
        PANELS.iter().filter(|p| p.2 == group).any(|p| layout.contains(&p.0.to_owned()) && !collapsed.contains(p.0))
    })
}

pub(crate) fn rail_click(app: &mut PhotocraftApp, group: Group) {
    let Some(layout) = app.ui.docking.as_ref() else { return };
    let existing = PANELS.iter().find(|p| p.2 == group && layout.contains(&p.0.to_owned())).map(|p| p.0);
    let panel = existing.or_else(|| {
        group
            .tabs(matches!(app.ui.theme, crate::theme::ThemeKind::Pro | crate::theme::ThemeKind::ProMedium))
            .get(*group.tab_mut(&mut app.ui.dock_tabs))
            .and_then(|label| normalize(label))
    });
    if let Some(panel) = panel {
        let result = if existing.is_some() && rail_on(app, group) {
            group_command(app, &json!({"panel": panel, "operation":"collapse"}))
        } else {
            change(app, "activate", &json!({"panel":panel}))
        };
        if let Err(error) = result {
            app.ui.status = error;
        }
    }
}

fn root_groups(layout: &Layout<String>) -> Vec<Vec<String>> {
    let mut out = Vec::new();
    let mut pending: Vec<_> = layout.root.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Split { first, second, .. } => pending.extend([second.as_ref(), first.as_ref()]),
            Node::Tabs { panels, .. } => out.push(panels.clone()),
            Node::Stack { entries } => out.extend(entries.iter().map(|e| vec![e.panel.clone()])),
        }
    }
    out
}

fn group_command(app: &mut PhotocraftApp, params: &Value) -> Result<Value, String> {
    let panel = params.get("panel").and_then(Value::as_str).and_then(normalize).ok_or("panel must name a registered panel")?;
    let operation = params.get("operation").and_then(Value::as_str).ok_or("operation is required")?;
    if app.session.prefs().workspace_locked {
        return Err("workspace is locked".into());
    }
    let mut layout = app.ui.docking.clone().unwrap_or_else(|| legacy(app));
    if !valid(&layout) {
        return Err("invalid panel layout".into());
    }
    let members = group_members(&layout, panel);
    if members.is_empty() {
        return Err("panel is hidden".into());
    }
    let mut hidden = if app.ui.docking.is_none() { migration_hidden(app, &layout) } else { app.ui.docking_hidden.clone() };
    let mut collapsed = collapsed_members(app, &layout);
    match operation {
        "close" => {
            for id in &members {
                hidden.insert(id.clone(), layout.location(id).map_err(|e| e.to_string())?);
                layout.apply(Action::Close { panel: id.clone() }).map_err(|e| e.to_string())?;
            }
            collapsed.retain(|id| !members.contains(id));
        }
        "collapse" => {
            collapsed.extend(members.iter().cloned());
        }
        "expand" => {
            collapsed.retain(|id| !members.contains(id));
        }
        "move" => {
            let anchor = params.get("anchor").and_then(Value::as_str).and_then(normalize).ok_or("anchor must name a registered panel")?;
            let placement = params
                .get("placement")
                .map(|v| serde_json::from_value::<Placement<String>>(v.clone()).map_err(|e| e.to_string()))
                .unwrap_or_else(|| placement(params))?;
            if members.iter().any(|id| id == anchor) {
                return Err("group cannot be moved into itself".into());
            }
            let first = members.first().ok_or("group is empty")?;
            layout.apply(Action::Move { panel: first.clone(), anchor: anchor.to_owned(), placement }).map_err(|e| e.to_string())?;
            let destination = group_members(&layout, first);
            let before = destination.iter().position(|id| id == first).and_then(|index| destination.get(index + 1)).cloned();
            for id in members.iter().skip(1) {
                layout
                    .apply(Action::Move { panel: id.clone(), anchor: first.clone(), placement: Placement::Tab { before: before.clone() } })
                    .map_err(|e| e.to_string())?;
            }
            layout.apply(Action::Activate { panel: panel.to_owned() }).map_err(|e| e.to_string())?;
        }
        "up" | "down" => {
            let groups = root_groups(&layout);
            let index = groups.iter().position(|g| g.contains(&panel.to_owned())).ok_or("floating groups cannot move within the dock")?;
            let neighbor = if operation == "up" { index.checked_sub(1) } else { index.checked_add(1).filter(|i| *i < groups.len()) };
            let anchor = neighbor.and_then(|i| groups.get(i)).and_then(|g| g.first()).ok_or("group is already at the edge")?;
            let first = members.first().ok_or("group is empty")?;
            layout
                .apply(Action::Move {
                    panel: first.clone(),
                    anchor: anchor.clone(),
                    placement: Placement::Split(if operation == "up" { Zone::Top } else { Zone::Bottom }),
                })
                .map_err(|e| e.to_string())?;
            for id in members.iter().skip(1) {
                layout
                    .apply(Action::Move { panel: id.clone(), anchor: first.clone(), placement: Placement::Tab { before: None } })
                    .map_err(|e| e.to_string())?;
            }
            layout.apply(Action::Activate { panel: panel.to_owned() }).map_err(|e| e.to_string())?;
        }
        _ => return Err("operation must be close, collapse, expand, move, up or down".into()),
    }
    app.ui.docking = Some(layout);
    app.ui.docking_hidden = hidden;
    app.ui.docking_collapsed = collapsed.into_iter().collect();
    sync_visibility(app);
    // Native collapse flags must not re-collapse a custom group after explicit expansion.
    if operation == "expand" || operation == "move" {
        reveal_group(app, panel);
    }
    Ok(json!({"layout":app.ui.docking}))
}

pub(crate) struct LegacyUpdate {
    layout: Layout<String>,
    hidden: std::collections::BTreeMap<String, craft_ui::docking::Location<String>>,
    clear_collapse: Vec<Group>,
}

/// Validate and stage the legacy controls before ui.set commits any fields.
pub(crate) fn prepare_legacy_update(
    app: &PhotocraftApp,
    request: &Value,
    panels: Option<&crate::state::Panels>,
    tabs: Option<&crate::state::DockTabs>,
    dock: Option<&crate::dock::DockLayout>,
    pro: bool,
) -> Result<Option<LegacyUpdate>, String> {
    let Some(current) = &app.ui.docking else { return Ok(None) };
    if panels.is_none() && tabs.is_none() && dock.is_none() {
        return Ok(None);
    }
    if !valid(current) {
        return Err("invalid panel layout; reset the workspace first".into());
    }
    // A legacy flat-column replacement cannot represent arbitrary custom splits. Reject it
    // explicitly instead of accepting a command whose requested order/height is ignored.
    if let Some(dock) = dock
        && (dock.order != app.ui.dock.order || dock.heights != app.ui.dock.heights)
    {
        return Err("dock order/heights cannot replace a customized layout; use window.panel.layout".into());
    }
    if app.session.prefs().workspace_locked {
        return Err("workspace is locked".into());
    }
    let clear_collapse: Vec<_> =
        dock.map(|d| Group::ALL.into_iter().filter(|g| d.is_collapsed(*g) != app.ui.dock.is_collapsed(*g)).collect()).unwrap_or_default();
    if !clear_collapse.is_empty()
        && root_groups(current).iter().any(|ids| {
            let groups: std::collections::HashSet<_> = ids.iter().filter_map(|id| PANELS.iter().find(|p| p.0 == id).map(|p| p.2)).collect();
            groups.len() > 1
        })
    {
        return Err("legacy collapse settings cannot replace mixed custom groups; use window.panel.group".into());
    }
    let mut layout = current.clone();
    let mut hidden = app.ui.docking_hidden.clone();
    for group in Group::ALL {
        let visibility = request.get("panels").and_then(|p| p.get(group.key())).and_then(Value::as_bool);
        let selected = request.get("dockTabs").and_then(|p| p.get(group.key())).is_some();
        let group_panels: Vec<_> = PANELS.iter().filter(|p| p.2 == group).collect();
        for &&(id, label, _) in &group_panels {
            let explicitly_hidden = dock.is_some_and(|d| d.hidden_tabs.get(&group).is_some_and(|names| names.iter().any(|name| name == label)));
            let explicitly_shown = dock.is_some() && !explicitly_hidden;
            if visibility == Some(false) || explicitly_hidden {
                if layout.contains(&id.to_owned()) {
                    hidden.insert(id.to_owned(), layout.location(&id.to_owned()).map_err(|e| e.to_string())?);
                    layout.apply(Action::Close { panel: id.to_owned() }).map_err(|e| e.to_string())?;
                }
            } else if (visibility == Some(true) || explicitly_shown) && !layout.contains(&id.to_owned()) {
                if hidden.contains_key(id) {
                    restore_hidden(&mut layout, &mut hidden, id);
                }
                ensure(&mut layout, id)?;
            }
        }
        if selected {
            let tabs = tabs.ok_or("dockTabs is missing")?;
            let index = match group {
                Group::Color => tabs.color,
                Group::Properties => tabs.properties,
                Group::Character => tabs.character,
                Group::Navigator => tabs.navigator,
                Group::History => tabs.history,
                Group::Layers => tabs.layers,
            };
            let id = group.tabs(pro).get(index).and_then(|label| normalize(label)).ok_or("dockTabs index is outside the panel group")?;
            if visibility == Some(false) {
                return Err("cannot select a tab in a hidden group".into());
            }
            if !layout.contains(&id.to_owned()) {
                if hidden.contains_key(id) {
                    restore_hidden(&mut layout, &mut hidden, id);
                }
                ensure(&mut layout, id)?;
            }
            layout.apply(Action::Activate { panel: id.to_owned() }).map_err(|e| e.to_string())?;
        }
    }
    Ok(Some(LegacyUpdate { layout, hidden, clear_collapse }))
}

pub(crate) fn commit_legacy_update(app: &mut PhotocraftApp, update: LegacyUpdate, request: &Value) {
    for group in update.clear_collapse {
        let ids: Vec<_> = PANELS.iter().filter(|p| p.2 == group).map(|p| p.0).collect();
        app.ui.docking_collapsed.retain(|id| !ids.contains(&id.as_str()));
    }
    app.ui.docking = Some(update.layout);
    app.ui.docking_hidden = update.hidden;
    sync_visibility(app);
    for group in Group::ALL {
        if request.get("dockTabs").and_then(|p| p.get(group.key())).is_some()
            || request.get("panels").and_then(|p| p.get(group.key())).and_then(Value::as_bool) == Some(true)
        {
            let ids: Vec<_> = PANELS.iter().filter(|p| p.2 == group).map(|p| p.0).collect();
            app.ui.docking_collapsed.retain(|id| !ids.contains(&id.as_str()));
            app.ui.dock.set_collapsed(group, false);
        }
    }
}

struct PanelContent<'a> {
    app: &'a mut PhotocraftApp,
    collapsed: &'a std::collections::HashSet<String>,
    commands: &'a mut Vec<(&'static str, Value)>,
    targets: &'a mut Vec<craft_ui::docking::DropTarget<String>>,
}

fn panel_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui, panel: &str) {
    match panel {
        "layers" => {
            crate::layer_row_ui::panel_menu(app, ui);
            ui.separator();
        }
        "swatches" => {
            crate::swatches_ui::panel_menu(app, ui);
            ui.separator();
        }
        "brushSettings" | "brush_settings" => {
            crate::brush_panel::panel_menu(app, ui);
            ui.separator();
        }
        "actions" => {
            crate::actions::panel_menu(app, ui);
            ui.separator();
        }
        _ => {}
    }
}

impl craft_ui::docking::DockContent<String> for PanelContent<'_> {
    fn body(&mut self, ui: &mut egui::Ui, id: &String) {
        if let Some(target) = self.targets.iter_mut().find(|target| target.anchor == *id) {
            target.rect = target.rect.union(ui.max_rect());
            if let Some(strip) = target.tab_strip.as_mut() {
                strip.min.x = target.rect.left();
                strip.max.x = target.rect.right();
            }
            let members = self.app.ui.docking.as_ref().map(|layout| group_members(layout, id)).unwrap_or_default();
            target.tabs = members
                .into_iter()
                .filter_map(|panel| {
                    ui.ctx().read_response(egui::Id::new("photocraft-panel-docking").with(("tab", &panel))).map(|response| (panel, response.rect))
                })
                .collect();
        }
        if !self.collapsed.contains(id) {
            egui::Frame::NONE.inner_margin(egui::Margin::same(8)).show(ui, |ui| body(self.app, ui, id));
        }
    }
    fn panel_menu(&mut self, ui: &mut egui::Ui, id: &String) {
        panel_menu(self.app, ui, id);
    }
    fn group_header_width(&self, _panels: &[String], _active: &String, _floating: bool) -> f32 {
        28.0
    }
    fn group_header(&mut self, ui: &mut egui::Ui, panels: &[String], active: &String, floating: bool) {
        if !floating {
            let rect = ui.max_rect();
            self.targets.push(craft_ui::docking::DropTarget {
                anchor: active.clone(),
                rect,
                tab_strip: Some(rect),
                tabs: Vec::new(),
                allow_splits: true,
                accepts_tabs: !self.app.session.prefs().workspace_locked,
            });
        }
        let collapsed = panels.iter().all(|id| self.collapsed.contains(id));
        let locked = self.app.session.prefs().workspace_locked;
        let resp = crate::icons::button(ui, "ellipsis", ui.available_height().min(28.0), false, tl!("Panel menu"));
        egui::Popup::menu(&resp).show(|ui| {
            ui.set_min_width(180.0);
            panel_menu(self.app, ui, active);
            if ui.add_enabled(!locked, egui::Button::new(if collapsed { tl!("Expand Panel Group") } else { tl!("Collapse Panel Group") })).clicked() {
                self.commands.push(("window.panel.group", json!({"panel":active,"operation":if collapsed {"expand"} else {"collapse"}})));
                ui.close();
            }
            if !floating {
                let groups = self.app.ui.docking.as_ref().map(root_groups).unwrap_or_default();
                let index = groups.iter().position(|g| g.contains(active));
                for (operation, label, enabled) in
                    [("up", tl!("Move Group Up"), index.is_some_and(|i| i > 0)), ("down", tl!("Move Group Down"), index.is_some_and(|i| i + 1 < groups.len()))]
                {
                    if ui.add_enabled(!locked && enabled, egui::Button::new(label)).clicked() {
                        self.commands.push(("window.panel.group", json!({"panel":active,"operation":operation})));
                        ui.close();
                    }
                }
            }
            ui.separator();
            if ui.add_enabled(!locked, egui::Button::new(tl!("Close Tab Group"))).clicked() {
                self.commands.push(("window.panel.group", json!({"panel":active,"operation":"close"})));
                ui.close();
            }
        });
    }
}

pub(crate) fn sync_visibility(app: &mut PhotocraftApp) {
    if let Some(layout) = &app.ui.docking {
        if !valid(layout) {
            return;
        }
        let pro = matches!(app.ui.theme, crate::theme::ThemeKind::Pro | crate::theme::ThemeKind::ProMedium);
        for group in Group::ALL {
            *group.shown_mut(&mut app.ui.panels) = PANELS.iter().filter(|p| p.2 == group).any(|p| layout.contains(&p.0.to_owned()));
            for (index, label) in group.tabs(pro).iter().enumerate() {
                if let Some(id) = normalize(label) {
                    if layout.contains(&id.to_owned()) {
                        app.ui.dock.show_tab(group, index, pro);
                    } else {
                        app.ui.dock.hide_tab(group, index, pro);
                    }
                }
            }
            let previous = *group.tab_mut(&mut app.ui.dock_tabs);
            if !group.tabs(pro).get(previous).and_then(|name| normalize(name)).is_some_and(|id| panel_active(layout, id))
                && let Some(selected) = group.tabs(pro).iter().position(|name| normalize(name).is_some_and(|id| panel_active(layout, id)))
            {
                *group.tab_mut(&mut app.ui.dock_tabs) = selected;
            }
        }
    }
}

fn panel_active(layout: &Layout<String>, panel: &str) -> bool {
    let mut pending: Vec<_> = layout.root.iter().collect();
    while let Some(node) = pending.pop() {
        match node {
            Node::Split { first, second, .. } => pending.extend([first.as_ref(), second.as_ref()]),
            Node::Tabs { panels, active } if panels.iter().any(|p| p == panel) => return panels.get(*active).is_some_and(|p| p == panel),
            Node::Stack { entries } if entries.iter().any(|e| e.panel == panel) => return true,
            _ => {}
        }
    }
    layout.floating.iter().any(|g| g.panels.get(g.active).is_some_and(|p| p == panel))
}

/// Extra legacy menu choices use the same commands as the customized renderer.
pub(crate) fn legacy_menu(app: &mut PhotocraftApp, ui: &mut egui::Ui, label: &str) {
    let Some(panel) = normalize(label) else { return };
    let locked = app.session.prefs().workspace_locked;
    ui.add_enabled_ui(!locked, |ui| {
        if ui.button(tl!("Float Panel")).clicked() {
            if let Err(error) = app.run("window.panel.float", json!({"panel":panel})) {
                app.ui.status = error;
            }
            ui.close();
        }
        ui.menu_button(tl!("Group with"), |ui| {
            for &(anchor, label, _) in PANELS {
                if anchor != panel && ui.button(tl!(label)).clicked() {
                    let result = app
                        .run("window.panel.activate", json!({"panel":anchor}))
                        .and_then(|_| app.run("window.panel.move", json!({"panel":panel,"anchor":anchor})));
                    if let Err(error) = result {
                        app.ui.status = error;
                    }
                    ui.close();
                }
            }
        });
    });
}

// A closed group forms a bounded chain of hidden tab neighbors. Restore just the requested
// panel at the first surviving return position, then let the remaining tabs rejoin it.
fn restore_hidden(layout: &mut Layout<String>, hidden: &mut std::collections::BTreeMap<String, craft_ui::docking::Location<String>>, panel: &str) {
    let Some(mut at) = hidden.get(panel).cloned() else { return };
    let mut terminal = panel.to_owned();
    let mut seen = vec![terminal.clone()];
    while let Some(anchor) = at.anchor.as_ref().filter(|id| !layout.contains(id)).cloned() {
        if seen.len() >= PANELS.len() || seen.contains(&anchor) {
            return;
        }
        let Some(next) = hidden.get(&anchor).cloned() else { return };
        seen.push(anchor.clone());
        terminal = anchor;
        at = next;
    }
    if layout.restore(panel.to_owned(), &at).is_ok()
        && terminal != panel
        && let Some(last) = hidden.get_mut(&terminal)
    {
        last.anchor = Some(panel.to_owned());
        last.placement = Placement::Tab { before: None };
        last.size = None;
        last.sibling.clear();
    }
}
