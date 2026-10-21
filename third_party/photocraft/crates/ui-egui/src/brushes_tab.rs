//! Window › Brushes: the brush presets in collapsible groups, as a list (tip, size, stroke
//! preview, name) or a grid of tips, with a size slider and a search field.
//!
//! Presets and groups are organised like Photoshop's panel: drag a preset within its group or
//! into another group or folder (onto a preset, or onto a header to append), drag a group's or
//! folder's header onto another header to put it inside that folder (or onto the header's upper
//! edge to place it before, which reorders siblings and moves a folder out to the top level), and
//! right-click for Rename and Delete. Groups hold nested folders ([`BrushPreset::folder`],
//! e.g. an imported `.abr` file's own folders), drawn indented under their group and opened or
//! closed like groups. Every change is a `brush.presets.*` command, so it is journaled, drivable,
//! and persisted by the preset store.
//!
//! The list itself ([`preset_list`]) is shared with the Brush Preset picker (`brush_picker`), so
//! both show, pick and organise presets the same way; each keeps its own view state.

use egui::{Color32, RichText, Sense, Stroke, pos2, vec2};
use photocraft_engine::paint::{BrushPreset, MAX_BRUSH_SIZE};
use serde_json::json;

use crate::brush_panel::{BrushesPanelState, BrushesView, Renaming, UNGROUPED, WIDTH, commit_gesture, full_uv, grouped_presets, run_or_status};
use crate::brush_preview;
use crate::theme::{self, Tokens};
use crate::{PhotocraftApp, icons, widgets};

/// What a drag in the Brushes tab carries.
#[derive(Clone, Debug, PartialEq)]
pub enum BrushDrag {
    Preset(String),
    Group(String),
    /// A nested folder of a group.
    Folder {
        group: String,
        folder: Vec<String>,
    },
}

/// What a preset list does after drawing (the presets are borrowed while it draws).
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// A click: make the preset current.
    Select(String),
    /// A double-click: make the preset current and dismiss the picker showing it (Photoshop).
    Choose(String),
    /// View only: open or close a group or folder (by its [`folder_view_key`]; [`preset_list`]
    /// applies it to its view state).
    ToggleGroup(String),
    Move {
        name: String,
        group: String,
        /// The nested folder in `group` (`None`: the command's default).
        folder: Option<Vec<String>>,
        index: Option<usize>,
    },
    /// Move a group (`folder` empty) among the groups, or a nested folder among its siblings;
    /// `before` is a group, or a sibling folder's name.
    MoveGroup {
        group: String,
        folder: Vec<String>,
        before: Option<String>,
    },
    /// Move a group (`folder` empty) or nested folder into another folder: `into` is (group,
    /// folder path, `[]` = directly in the group), or `None` for the top level (it becomes a
    /// group). `before` is a folder (or top-level group) there to land before.
    MoveInto {
        group: String,
        folder: Vec<String>,
        into: Option<(String, Vec<String>)>,
        before: Option<String>,
    },
    /// View only: show the rename bar for a preset or group ([`preset_list`] applies it).
    BeginRename(Renaming),
    /// Commit a rename typed in the rename bar.
    Rename(Renaming),
    Delete(String),
    /// Delete a group (`folder` empty) or one of its nested folders, with their presets.
    DeleteGroup {
        group: String,
        folder: Vec<String>,
    },
}

/// How a preset list is laid out: the Brushes tab's, or the Brush Preset picker's (narrower,
/// denser).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ListLayout {
    /// The scroll area's id and its height limit.
    pub id: &'static str,
    pub max_height: f32,
    /// Width of a grid cell (its height is 6 points more) and the grid's left indent.
    pub cell: f32,
    pub indent: f32,
    /// The Brush Preset picker's cards: a tip cell and a stroke cell on top and the name across
    /// the bottom (or, with only the tip and the name on, in the stroke's cell). Replaces the
    /// list/grid choice in the picker.
    pub cards: bool,
}

/// The Brushes tab's list.
pub const PANEL_LIST: ListLayout = ListLayout { id: "brush-presets", max_height: 400.0, cell: 52.0, indent: 20.0, cards: false };

/// The group key (`BrushPreset::group`) behind a panel label: ungrouped presets show as
/// [`UNGROUPED`].
pub fn group_key(presets: &[BrushPreset], label: &str) -> String {
    if label == UNGROUPED && !presets.iter().any(|p| p.group == UNGROUPED) { String::new() } else { label.to_string() }
}

/// Deepest nested folder the panel draws; deeper folders show at this depth.
pub const MAX_VIEW_DEPTH: usize = photocraft_engine::brush_preset_cmds::MAX_FOLDER_DEPTH;

/// The collapse key of a nested folder of the group shown as `label` (`label` itself for the
/// group).
pub fn folder_view_key(label: &str, folder: &[String]) -> String {
    if folder.is_empty() { label.to_string() } else { format!("{label}/{}", folder.join("/")) }
}

/// A group's content in panel order: its presets and nested folders.
#[derive(Debug)]
pub enum Node<'a> {
    Preset(&'a BrushPreset),
    Folder(FolderNode<'a>),
}

/// A nested folder and what it holds.
#[derive(Debug)]
pub struct FolderNode<'a> {
    /// The folder's path in its group.
    pub path: Vec<String>,
    pub children: Vec<Node<'a>>,
    /// Presets in it and below it.
    pub count: usize,
}

/// The tree of a group's presets (`items`, in library order): each folder appears where its first
/// preset is, presets keep their order. Paths deeper than [`MAX_VIEW_DEPTH`] are cut there.
pub fn folder_tree<'a>(items: &[&'a BrushPreset]) -> Vec<Node<'a>> {
    level(items.iter().map(|p| (p.folder.get(..p.folder.len().min(MAX_VIEW_DEPTH)).unwrap_or_default(), *p)).collect(), &[])
}

/// A preset with its folder path below the level being built.
type Entry<'a> = (&'a [String], &'a BrushPreset);

/// One level of [`folder_tree`]: `items` pairs each preset with its path below `prefix`. The
/// recursion is as deep as the longest path (at most [`MAX_VIEW_DEPTH`]).
fn level<'a>(items: Vec<Entry<'a>>, prefix: &[String]) -> Vec<Node<'a>> {
    enum Slot {
        Preset(usize),
        Folder(usize),
    }
    let mut slots = Vec::new();
    let mut folders: Vec<(&String, Vec<Entry<'a>>)> = Vec::new();
    for (i, (path, p)) in items.iter().enumerate() {
        match path.split_first() {
            None => slots.push(Slot::Preset(i)),
            Some((name, rest)) => match folders.iter_mut().position(|(n, _)| *n == name) {
                Some(k) => {
                    if let Some((_, v)) = folders.get_mut(k) {
                        v.push((rest, *p));
                    }
                }
                None => {
                    slots.push(Slot::Folder(folders.len()));
                    folders.push((name, vec![(rest, *p)]));
                }
            },
        }
    }
    let mut folders: Vec<Option<(&String, Vec<Entry<'a>>)>> = folders.into_iter().map(Some).collect();
    slots
        .into_iter()
        .filter_map(|s| match s {
            Slot::Preset(i) => items.get(i).map(|(_, p)| Node::Preset(p)),
            Slot::Folder(k) => {
                let (name, sub) = folders.get_mut(k)?.take()?;
                let path: Vec<String> = prefix.iter().cloned().chain(std::iter::once(name.clone())).collect();
                let count = sub.len();
                let children = level(sub, &path);
                Some(Node::Folder(FolderNode { path, children, count }))
            }
        })
        .collect()
}

/// Where a preset dropped onto `target` lands: (group, folder, index in that group after the
/// dragged preset is taken out). `after` = dropped on the target's lower (list) or right (grid)
/// half.
pub fn drop_target(presets: &[BrushPreset], dragged: &str, target: &str, after: bool) -> Option<(String, Vec<String>, usize)> {
    let t = presets.iter().find(|p| p.name == target)?;
    let members: Vec<&str> = presets.iter().filter(|p| p.group == t.group).map(|p| p.name.as_str()).collect();
    let tp = members.iter().position(|n| *n == target)?;
    let mut idx = tp + usize::from(after);
    if let Some(dp) = members.iter().position(|n| *n == dragged)
        && dp < idx
    {
        idx -= 1;
    }
    Some((t.group.clone(), t.folder.clone(), idx))
}

/// Apply the view-only actions to `st`; returns the rest, the commands for [`apply`].
fn view_actions(st: &mut BrushesPanelState, acts: Vec<Action>) -> Vec<Action> {
    acts.into_iter()
        .filter_map(|a| match a {
            Action::ToggleGroup(g) => {
                let c = &mut st.collapsed;
                match c.iter().position(|x| *x == g) {
                    Some(i) => drop(c.remove(i)),
                    None => c.push(g),
                }
                None
            }
            Action::BeginRename(r) => {
                st.renaming = Some(r);
                None
            }
            a => Some(a),
        })
        .collect()
}

/// Turn a preset list's actions into commands.
pub fn apply(app: &mut PhotocraftApp, acts: Vec<Action>) {
    for a in acts {
        match a {
            Action::Select(name) => run_or_status(app, "tools.setBrush", json!({ "preset": name })),
            Action::Choose(name) => {
                // The double-click's first click already picked it.
                if app.session.tools.current_preset.as_deref() != Some(name.as_str()) {
                    run_or_status(app, "tools.setBrush", json!({ "preset": name }));
                }
            }
            Action::ToggleGroup(_) | Action::BeginRename(_) => {}
            Action::Move { name, group, folder, index } => {
                let mut p = json!({ "name": name, "group": group });
                if let Some(f) = folder {
                    p["folder"] = json!(f);
                }
                if let Some(i) = index {
                    p["index"] = json!(i);
                }
                run_or_status(app, "brush.presets.move", p);
            }
            Action::MoveGroup { group, folder, before } => {
                let mut p = json!({ "group": group, "before": before });
                if !folder.is_empty() {
                    p["folder"] = json!(folder);
                }
                run_or_status(app, "brush.presets.moveGroup", p);
            }
            Action::MoveInto { group, folder, into, before } => {
                let into = match into {
                    Some((g, f)) => json!({ "group": g, "folder": f }),
                    None => json!({}),
                };
                let mut p = json!({ "group": group, "into": into });
                if !folder.is_empty() {
                    p["folder"] = json!(folder);
                }
                if let Some(b) = before {
                    p["before"] = json!(b);
                }
                run_or_status(app, "brush.presets.moveGroup", p);
            }
            Action::Rename(r) => {
                let text = r.text.trim().to_string();
                let old = r.folder.last().unwrap_or(&r.name);
                if !text.is_empty() && text != *old {
                    if r.group {
                        let mut p = json!({ "group": r.name, "newName": text });
                        if !r.folder.is_empty() {
                            p["folder"] = json!(r.folder);
                        }
                        run_or_status(app, "brush.presets.renameGroup", p);
                    } else {
                        run_or_status(app, "brush.presets.rename", json!({ "name": r.name, "newName": text }));
                    }
                }
            }
            Action::Delete(name) => run_or_status(app, "brush.presets.delete", json!({ "name": name })),
            Action::DeleteGroup { group, folder } => {
                let mut p = json!({ "group": group });
                if !folder.is_empty() {
                    p["folder"] = json!(folder);
                }
                run_or_status(app, "brush.presets.deleteGroup", p);
            }
        }
    }
}

/// Is the pointer on the second half of `r` (lower in a list, right in a grid)?
fn second_half(ui: &egui::Ui, r: egui::Rect, grid: bool) -> bool {
    ui.ctx().pointer_interact_pos().is_some_and(|p| if grid { p.x > r.center().x } else { p.y > r.center().y })
}

/// Drag, drop and context menu of one preset cell or row.
fn preset_interactions(ui: &egui::Ui, resp: &egui::Response, r: egui::Rect, p: &BrushPreset, presets: &[BrushPreset], grid: bool, acts: &mut Vec<Action>) {
    let t = Tokens::get(ui.ctx());
    resp.dnd_set_drag_payload(BrushDrag::Preset(p.name.clone()));
    if let Some(d) = resp.dnd_hover_payload::<BrushDrag>()
        && matches!(&*d, BrushDrag::Preset(n) if *n != p.name)
    {
        widgets::drop_line(ui, r, second_half(ui, r, grid), grid, &t);
    }
    if let Some(d) = resp.dnd_release_payload::<BrushDrag>()
        && let BrushDrag::Preset(n) = &*d
        && *n != p.name
        && let Some((group, folder, index)) = drop_target(presets, n, &p.name, second_half(ui, r, grid))
    {
        acts.push(Action::Move { name: n.clone(), group, folder: Some(folder), index: Some(index) });
    }
    if resp.double_clicked() {
        acts.push(Action::Choose(p.name.clone()));
    } else if resp.clicked() {
        acts.push(Action::Select(p.name.clone()));
    }
    resp.context_menu(|ui| {
        if ui.button(tl!("Rename Brush…")).clicked() {
            acts.push(Action::BeginRename(Renaming { group: false, name: p.name.clone(), folder: Vec::new(), text: String::new() }));
            ui.close();
        }
        if ui.button(tl!("Delete Brush")).clicked() {
            acts.push(Action::Delete(p.name.clone()));
            ui.close();
        }
    });
}

/// Width a list row keeps for the gaps around its stroke preview and the preset name.
const STROKE_NAME_ROOM: f32 = 140.0;

fn list_row(ui: &mut egui::Ui, p: &BrushPreset, current: bool, presets: &[BrushPreset], indent: f32, acts: &mut Vec<Action>) {
    let t = Tokens::get(ui.ctx());
    let (full, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 44.0), Sense::click_and_drag());
    // Rows inside nested folders are indented; the whole width stays the drop target.
    let r = egui::Rect::from_min_max(full.min + vec2(indent.min(full.width() / 2.0), 0.0), full.max);
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, current, &p.name));
    if !ui.is_rect_visible(r) {
        return;
    }
    if current {
        ui.painter().rect_filled(r, t.radius_sm, t.accent_soft);
    } else if resp.hovered() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    let pb = &p.brush;
    let tip = brush_preview::tip_texture(ui.ctx(), &format!("brushes-tip:{}", p.name), &pb.tip, (pb.hardness, pb.angle, pb.roundness), 36, t.text);
    let cell = egui::Rect::from_min_size(r.left_top() + vec2(22.0, 2.0), vec2(36.0, 30.0));
    let tr = egui::Rect::from_center_size(cell.center(), vec2(30.0, 30.0) * crate::brush_sections::thumb_scale(pb.size));
    ui.painter().image(tip.id(), tr, full_uv(), Color32::WHITE);
    ui.painter().text(
        pos2(cell.center().x, r.bottom() - 1.0),
        egui::Align2::CENTER_BOTTOM,
        format!("{}", pb.size.round() as i64),
        egui::FontId::proportional(9.0),
        t.text_faint,
    );
    // The stroke preview gives up width so a narrow list (the picker) still shows the name; its
    // texture slot carries the width, so lists of different widths don't re-render each other's.
    let w = (r.width() - (cell.right() - r.left()) - STROKE_NAME_ROOM).clamp(60.0, 170.0).round();
    let stroke = brush_preview::stroke_texture(ui.ctx(), &format!("brushes-stroke-{w}:{}", p.name), pb, w as u32, 36, t.text);
    let sr = egui::Rect::from_min_size(pos2(cell.right() + 8.0, r.top() + 4.0), vec2(w, 36.0));
    ui.painter().image(stroke.id(), sr, full_uv(), Color32::WHITE);
    crate::layer_row_ui::label(ui.painter(), sr.right() + 12.0, r.center().y, r.right() - 4.0, &p.name, egui::FontId::proportional(12.0), t.text_dim);
    preset_interactions(ui, &resp, r, p, presets, false, acts);
}

/// The gap between grid cells, across and down.
const GRID_GAP: f32 = 3.0;
/// The picker's cards pack tighter than the grid: their horizontal gap (the vertical one stays
/// [`GRID_GAP`]).
const CARD_GAP: f32 = 1.0;

/// The grid's columns in a list `width` points wide: how many `cell`-wide cells separated by
/// `gap` fit past the `indent`, and the left margin that centres them in the room past it. At
/// least one column.
pub(crate) fn grid_columns(width: f32, cell: f32, indent: f32, gap: f32) -> (f32, usize) {
    let room = if width.is_finite() { (width - indent).max(0.0) } else { 0.0 };
    let cols = if cell.is_finite() && cell > 0.0 { ((room + gap) / (cell + gap)).floor() } else { 1.0 };
    // `as` saturates (and NaN becomes 0), so this is a whole number from 1 to a sane bound.
    let cols = (cols as usize).clamp(1, 256);
    let used = cols as f32 * cell + (cols - 1) as f32 * gap;
    // Whole points keep the tiles' edges crisp.
    let left = indent + ((room - used) / 2.0).max(0.0).floor();
    (if left.is_finite() { left } else { 0.0 }, cols)
}

fn grid_cell(ui: &mut egui::Ui, p: &BrushPreset, current: bool, presets: &[BrushPreset], cell: f32, acts: &mut Vec<Action>) {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(cell, cell + 6.0), Sense::click_and_drag());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, current, &p.name));
    if !ui.is_rect_visible(r) {
        return;
    }
    if current {
        ui.painter().rect_filled(r, 3.0, t.accent_soft);
        ui.painter().rect_stroke(r, 3.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
    } else if resp.hovered() {
        ui.painter().rect_filled(r, 3.0, t.hover);
    }
    let pb = &p.brush;
    let tip = brush_preview::tip_texture(ui.ctx(), &format!("brushes-tip:{}", p.name), &pb.tip, (pb.hardness, pb.angle, pb.roundness), 36, t.text);
    let side = cell - 14.0;
    let ir = egui::Rect::from_center_size(pos2(r.center().x, r.top() + side / 2.0 + 4.0), vec2(side, side) * crate::brush_sections::thumb_scale(pb.size));
    ui.painter().image(tip.id(), ir, full_uv(), Color32::WHITE);
    ui.painter().text(
        pos2(r.center().x, r.bottom() - 7.0),
        egui::Align2::CENTER_CENTER,
        format!("{}", pb.size.round() as i64),
        egui::FontId::proportional(9.5),
        t.text_dim,
    );
    let resp = resp.on_hover_text(&p.name);
    preset_interactions(ui, &resp, r, p, presets, true, acts);
}

/// Padding, tip cell width and name row height, at scale 1; the footer slider multiplies them
/// (heights down to [`CARD_MIN_H_SCALE`]).
const CARD_PAD: f32 = 4.0;
const CARD_TIP_W: f32 = 46.0;
/// The tip thumbnail's base size at scale 1, before the footer slider and the per-preset
/// [`crate::brush_sections::thumb_scale`]: the tips read small at 28 px, so the base carries a
/// 60 % bump.
const CARD_TIP_BASE: f32 = 28.0 * 1.6;
const CARD_NAME_H: f32 = 20.0;
/// A full card's height at scale 1: the tip/stroke row takes what the name row leaves. The
/// footer slider multiplies it down to [`CARD_MIN_H_SCALE`].
const CARD_MAX_H: f32 = 70.0;
/// The widest a lone card grows at scale 1; past it the room is shared between columns.
const CARD_MAX_W: f32 = 300.0;
/// Tip alone: a grid of small [`CARD_TIP_ONLY_W`] × [`CARD_TIP_ONLY_H`] cells (scale 1).
const CARD_TIP_ONLY_W: f32 = 60.0;
const CARD_TIP_ONLY_H: f32 = 70.0;
/// Tip + name, no stroke: a grid of [`CARD_TIP_NAME_W`] × [`CARD_TIP_NAME_H`] cells (scale 1).
const CARD_TIP_NAME_W: f32 = 150.0;
const CARD_TIP_NAME_H: f32 = 70.0;
/// The footer scale's floor for heights and padding (the widths keep shrinking).
const CARD_MIN_H_SCALE: f32 = 0.60;
/// At or below this scale the tips drop their size numbers and fill their cell whole.
const CARD_COMPACT_SCALE: f32 = 0.70;
/// stay readable from the smallest card to the largest.
const CARD_NAME_FONT: f32 = 12.0;
const CARD_NUM_FONT: f32 = 11.0;
/// The cards' own fill: [`Tokens::card`] (what the picker's popup is filled with) this much
/// darker, so the cards read as tiles against it.
const CARD_BG_SHADE: f32 = 0.90;

/// The picker's cards in a list `width` points wide, `indent` past its left edge: how wide each
/// card is and how many sit in a row. A lone card fills the room up to [`CARD_MAX_W`] × `scale`
/// (wider lists leave the rest empty); from two columns on, the cards share the room, filling it
/// fully. `scale` is the picker's footer slider ([`crate::brush_picker::body`]), already floored
/// so the widths never reach zero.
fn card_columns(width: f32, indent: f32, scale: f32) -> (f32, usize) {
    let max_w = CARD_MAX_W * scale;
    let room = if width.is_finite() { (width - indent).max(0.0) } else { CARD_MAX_W };
    let cols = ((room / max_w).floor() as usize).clamp(1, 256);
    let share = (room - CARD_GAP * (cols - 1) as f32) / cols as f32;
    let card_w = if cols == 1 { room.min(max_w) } else { share };
    (if card_w.is_finite() { card_w.max(1.0) } else { CARD_MAX_W }, cols)
}

/// Is the card showing the tip alone (no name, no stroke)? Then it's a small tip cell in a grid.
fn tip_only(show: (bool, bool, bool)) -> bool {
    show == (false, false, true)
}

/// Is the card showing the tip and the name with the stroke off? Then it's a tip-and-name cell.
fn tip_and_name(show: (bool, bool, bool)) -> bool {
    show == (true, false, true)
}

/// `c` darker by `f` in gamma space, kept opaque — [`Color32::gamma_multiply`] fades the alpha
/// along with the channels, so it can't darken a fill over its own color.
fn darker(c: Color32, f: f32) -> Color32 {
    let [r, g, b, a] = c.to_srgba_unmultiplied();
    Color32::from_rgba_unmultiplied((r as f32 * f).round() as u8, (g as f32 * f).round() as u8, (b as f32 * f).round() as u8, a)
}

/// Where a card's tip thumbnail goes in its `cell`. `number_h` is the height of the size number
/// under it, `None` on compact cards (no number: the tip takes the cell whole). Otherwise the
/// number keeps a band at the bottom of the cell and the tip is centred in what's left above it,
/// at [`CARD_TIP_BASE`] × `scale` or that room's side, whichever is smaller, times the preset's
/// [`crate::brush_sections::thumb_scale`]: big brushes still read bigger than small ones, and
/// none of them reaches down over the number.
fn card_tip_rect(cell: egui::Rect, number_h: Option<f32>, size: f32, scale: f32) -> egui::Rect {
    let Some(number_h) = number_h else {
        return egui::Rect::from_center_size(cell.center(), egui::Vec2::splat((cell.width().min(cell.height()) - 2.0).max(1.0)));
    };
    // The number sits 1 pt off the cell's bottom, with a 1 pt gap above it.
    let room = egui::Rect::from_min_max(cell.min, pos2(cell.right(), (cell.bottom() - number_h - 2.0).max(cell.top())));
    let side = (CARD_TIP_BASE * scale).min(room.width().min(room.height())).max(0.0) * crate::brush_sections::thumb_scale(size);
    egui::Rect::from_center_size(room.center(), egui::Vec2::splat(if side.is_finite() { side.max(1.0) } else { 1.0 }))
}

/// One preset in the Brush Preset picker: a 2 × 2 grid whose bottom row (the name) spans both
/// columns — tip at the top left, stroke preview at the top right. A part that's off loses its
/// cell (the other takes the room): with only the tip and the name on, the name moves into the
/// stroke's cell and the bottom row is gone. At least one part is always on. `show` is the three
/// gear boxes (name, stroke, tip); `width` is the row slot the card fills (a [`card_columns`]
/// column or a fixed cell, [`CARD_TIP_ONLY_W`]/[`CARD_TIP_NAME_W`]), and `scale` the footer
/// slider: heights and padding floor at [`CARD_MIN_H_SCALE`], the text holds its 1.0 size
/// ([`CARD_NAME_FONT`], [`CARD_NUM_FONT`]), and at or below [`CARD_COMPACT_SCALE`] the tips
/// drop their size numbers and take their cell whole.
#[allow(clippy::too_many_arguments)]
fn preset_card(
    ui: &mut egui::Ui,
    p: &BrushPreset,
    current: bool,
    presets: &[BrushPreset],
    show: (bool, bool, bool),
    width: f32,
    scale: f32,
    acts: &mut Vec<Action>,
) {
    let (show_name, show_stroke, show_tip) = show;
    let tip_only = tip_only(show);
    let compact = scale <= CARD_COMPACT_SCALE;
    let t = Tokens::get(ui.ctx());
    let name_in_top = tip_and_name(show);
    // Widths (which arrive scaled) keep narrowing with the slider and heights stop at the floor;
    // the text holds its 1.0 size — the fonts don't take the scale.
    let scale_h = scale.max(CARD_MIN_H_SCALE);
    let pad = CARD_PAD * scale;
    let pad_y = CARD_PAD * scale_h;
    let name_h = if show_name && !name_in_top { CARD_NAME_H * scale_h } else { 0.0 };
    // The tip/stroke row: the tip-only or tip-and-name card's own height less the padding, else
    // what [`CARD_MAX_H`] leaves past the padding and the name row. A card with no top part at
    // all (the name alone) has no such row.
    let top_h = if tip_only {
        (CARD_TIP_ONLY_H - CARD_PAD * 2.0) * scale_h
    } else if name_in_top {
        (CARD_TIP_NAME_H - CARD_PAD * 2.0) * scale_h
    } else if show_tip || show_stroke {
        (CARD_MAX_H - CARD_PAD * 2.0 - CARD_NAME_H) * scale_h
    } else {
        0.0
    };
    let (full, resp) = ui.allocate_exact_size(vec2(width, pad_y * 2.0 + top_h + name_h), Sense::click_and_drag());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::Button, true, current, &p.name));
    if !ui.is_rect_visible(full) {
        return;
    }
    // The card is inset a little in its column; the whole column stays the drop target.
    let card = full.shrink2(vec2((4.0 * scale).min(full.width() * 0.5), 0.0));
    // Every card carries its own background, a touch darker than the popup behind it, so the
    // cards read as tiles; the current and hover fills paint over it.
    ui.painter().rect_filled(card, 3.0, darker(t.card, CARD_BG_SHADE));
    if current {
        ui.painter().rect_filled(card, 3.0, t.accent_soft);
        ui.painter().rect_stroke(card, 3.0, Stroke::new(1.0, t.accent), egui::StrokeKind::Inside);
    } else if resp.hovered() {
        ui.painter().rect_filled(card, 3.0, t.hover);
    }
    let pb = &p.brush;
    let inner = card.shrink2(vec2(pad, pad_y));
    let top = egui::Rect::from_min_size(inner.left_top(), vec2(inner.width(), top_h));
    let mut x = top.left();
    if show_tip {
        // The tip thumbnail with the size under it (like the list row). With the tip alone the
        // cell takes the whole card, so the thumbnail is centred in it (the tip-only grid).
        // Compact cards drop the number — it wouldn't fit — and the tip takes the cell whole.
        let tip_w = if tip_only { top.width() } else { (CARD_TIP_W * scale).min(top.width() * 0.5) };
        let cell = egui::Rect::from_min_size(top.left_top(), vec2(tip_w, top_h));
        let tip = brush_preview::tip_texture(ui.ctx(), &format!("brushes-tip:{}", p.name), &pb.tip, (pb.hardness, pb.angle, pb.roundness), 36, t.text);
        let number =
            (!compact).then(|| ui.painter().layout_no_wrap(format!("{}", pb.size.round() as i64), egui::FontId::proportional(CARD_NUM_FONT), t.text_faint));
        let tr = card_tip_rect(cell, number.as_ref().map(|g| g.size().y), pb.size, scale);
        ui.painter().image(tip.id(), tr, full_uv(), Color32::WHITE);
        if let Some(g) = number {
            let at = pos2(cell.center().x - g.size().x / 2.0, cell.bottom() - 1.0 - g.size().y);
            ui.painter().galley(at, g, t.text_faint);
        }
        x = cell.right();
    }
    if show_stroke || name_in_top {
        // The stroke preview, or the name where it goes when the stroke is hidden.
        let cell = egui::Rect::from_min_max(pos2(x, top.top()), top.right_bottom());
        if show_stroke {
            let cw = if cell.width().is_finite() { cell.width().max(1.0) } else { 1.0 };
            let w = ((cw - 16.0 * scale).clamp(24.0 * scale, 260.0 * scale)).min(cw).max(1.0).round();
            let h = (36.0 * scale_h).round().max(1.0) as u32;
            let stroke = brush_preview::stroke_texture(ui.ctx(), &format!("brushes-stroke-{w}:{}", p.name), pb, w as u32, h, t.text);
            let sr = egui::Rect::from_min_size(pos2(cell.left() + 8.0 * scale, cell.center().y - 18.0 * scale_h), vec2(w, 36.0 * scale_h));
            ui.painter().image(stroke.id(), sr, full_uv(), Color32::WHITE);
        } else {
            crate::layer_row_ui::label(
                ui.painter(),
                cell.left() + 8.0 * scale,
                cell.center().y,
                cell.right() - 6.0 * scale,
                &p.name,
                egui::FontId::proportional(CARD_NAME_FONT),
                t.text_dim,
            );
        }
    }
    if name_h > 0.0 {
        // The merged bottom row: the name across the whole card.
        let cell = egui::Rect::from_min_size(pos2(inner.left(), top.bottom()), vec2(inner.width(), name_h));
        crate::layer_row_ui::label(
            ui.painter(),
            cell.left() + 6.0 * scale,
            cell.center().y,
            cell.right() - 6.0 * scale,
            &p.name,
            egui::FontId::proportional(CARD_NAME_FONT),
            t.text_dim,
        );
    }
    let resp = resp.on_hover_text(&p.name);
    // A tip-only card's grid drops between columns, like the Brushes tab's: the rest drop between
    // rows (there is a row under the tip there).
    preset_interactions(ui, &resp, card, p, presets, tip_only, acts);
}

/// Indent of one nested folder level.
const FOLDER_INDENT: f32 = 16.0;

/// What dropping the group or folder (`src_group`, `src_folder`; a group when `src_folder` is
/// empty) on the header of (`group`, `folder`) does, as in Photoshop's Brushes panel: on the
/// header's upper edge (`edge`) it lands before that folder, next to it (reordering when they are
/// siblings); anywhere else it goes into that folder. `None` when the drop does nothing: onto
/// itself, into one of its own folders, or into the folder it is already in.
pub fn folder_drop(src_group: &str, src_folder: &[String], group: &str, folder: &[String], edge: bool) -> Option<Action> {
    if src_group == group && folder.starts_with(src_folder) {
        return None;
    }
    let src = (src_group.to_string(), src_folder.to_vec());
    if !edge {
        let parent = src_folder.split_last().map(|(_, p)| p);
        if src_group == group && parent == Some(folder) {
            return None;
        }
        return Some(Action::MoveInto { group: src.0, folder: src.1, into: Some((group.to_string(), folder.to_vec())), before: None });
    }
    match folder.split_last() {
        // Before a top-level group: groups reorder; a nested folder moves out to the top level.
        None if src_folder.is_empty() => Some(Action::MoveGroup { group: src.0, folder: src.1, before: Some(group.to_string()) }),
        None => Some(Action::MoveInto { group: src.0, folder: src.1, into: None, before: Some(group.to_string()) }),
        Some((leaf, parent)) => {
            let sibling = src_group == group && src_folder.split_last().map(|(_, p)| p) == Some(parent);
            if sibling {
                Some(Action::MoveGroup { group: src.0, folder: src.1, before: Some(leaf.clone()) })
            } else {
                Some(Action::MoveInto { group: src.0, folder: src.1, into: Some((group.to_string(), parent.to_vec())), before: Some(leaf.clone()) })
            }
        }
    }
}

/// A group's or nested folder's header: chevron, folder, name, count. Click toggles; a group or
/// folder dragged here lands before it (its upper edge) or goes into it ([`folder_drop`]); a
/// preset dropped here moves to the end of the group or folder. `key` is the group key, `folder` the nested folder's path (empty for the
/// group), `label` the group's panel label.
fn group_header(ui: &mut egui::Ui, label: &str, key: &str, folder: &[String], open: bool, count: usize, acts: &mut Vec<Action>) {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click_and_drag());
    if resp.hovered() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    let shown = folder.last().map_or(label, String::as_str);
    let x = r.left() + 4.0 + FOLDER_INDENT * folder.len().min(MAX_VIEW_DEPTH) as f32;
    icons::paint(
        ui,
        egui::Rect::from_min_size(pos2(x, r.center().y - 7.0), vec2(14.0, 14.0)),
        if open { "chevron-down" } else { "chevron-right" },
        12.0,
        t.text_dim,
    );
    icons::paint(
        ui,
        egui::Rect::from_min_size(pos2(x + 18.0, r.center().y - 8.0), vec2(16.0, 16.0)),
        if open { "folder-open" } else { "folder" },
        14.0,
        t.icon,
    );
    let font = if folder.is_empty() { theme::semibold(12.0) } else { theme::medium(12.0) };
    ui.painter().text(pos2(x + 40.0, r.center().y), egui::Align2::LEFT_CENTER, shown, font, t.text);
    ui.painter().text(pos2(r.right() - 8.0, r.center().y), egui::Align2::RIGHT_CENTER, count.to_string(), theme::medium(11.0), t.text_faint);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::CollapsingHeader, true, shown));
    let me = match folder {
        [] => BrushDrag::Group(key.to_string()),
        f => BrushDrag::Folder { group: key.to_string(), folder: f.to_vec() },
    };
    resp.dnd_set_drag_payload(me);
    // The upper third of the header places a dragged folder before this one; the rest drops it in.
    let edge = ui.ctx().pointer_interact_pos().is_some_and(|p| p.y < r.top() + r.height() / 3.0);
    let folder_action = |d: &BrushDrag| match d {
        BrushDrag::Group(g) => folder_drop(g, &[], key, folder, edge),
        BrushDrag::Folder { group, folder: f } => folder_drop(group, f, key, folder, edge),
        BrushDrag::Preset(_) => None,
    };
    if let Some(d) = resp.dnd_hover_payload::<BrushDrag>() {
        match (&*d, folder_action(&d)) {
            (BrushDrag::Preset(_), _) | (_, Some(Action::MoveInto { before: None, .. })) => {
                ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.5, t.accent), egui::StrokeKind::Inside);
            }
            (_, Some(_)) => widgets::drop_line(ui, r, false, false, &t),
            _ => {}
        }
    }
    if let Some(d) = resp.dnd_release_payload::<BrushDrag>() {
        match &*d {
            BrushDrag::Preset(n) => acts.push(Action::Move { name: n.clone(), group: key.to_string(), folder: Some(folder.to_vec()), index: None }),
            other => acts.extend(folder_action(other)),
        }
    }
    if resp.clicked() {
        acts.push(Action::ToggleGroup(folder_view_key(label, folder)));
    }
    resp.context_menu(|ui| {
        if ui.button(tl!("Rename Group…")).clicked() {
            acts.push(Action::BeginRename(Renaming { group: true, name: key.to_string(), folder: folder.to_vec(), text: String::new() }));
            ui.close();
        }
        if ui.button(tl!("Delete Group")).clicked() {
            acts.push(Action::DeleteGroup { group: key.to_string(), folder: folder.to_vec() });
            ui.close();
        }
    });
}

/// How a group's content is drawn.
struct Draw<'a> {
    label: &'a str,
    key: &'a str,
    presets: &'a [BrushPreset],
    /// The selected preset's name (by identity, so look-alike duplicates stay distinct).
    current: Option<&'a str>,
    collapsed: &'a [String],
    filtering: bool,
    grid: bool,
    layout: ListLayout,
    /// The picker cards' parts — name, stroke, tip — from the gear's boxes.
    show: (bool, bool, bool),
    /// The picker cards' size scale (its footer slider, 1 = standard): multiplies [`CARD_MAX_W`]
    /// and [`CARD_MAX_H`], or [`CARD_TIP_ONLY_W`]/[`CARD_TIP_ONLY_H`] for tip-only cards and
    /// [`CARD_TIP_NAME_W`]/[`CARD_TIP_NAME_H`] for tip-and-name ones; the text holds its 1.0
    /// size ([`CARD_NAME_FONT`], [`CARD_NUM_FONT`]); at or below [`CARD_COMPACT_SCALE`] the tips
    /// drop their size numbers.
    scale: f32,
}

/// Draw `nodes` (a group's or folder's content) at nesting `depth`: runs of presets as rows or a
/// grid, nested folders as headers with their content below when open. `depth` is bounded by
/// [`folder_tree`] (paths are cut at [`MAX_VIEW_DEPTH`]).
fn draw_nodes(ui: &mut egui::Ui, d: &Draw, nodes: &[Node], depth: usize, acts: &mut Vec<Action>) {
    let indent = FOLDER_INDENT * depth as f32;
    let mut run: Vec<&BrushPreset> = Vec::new();
    let flush = |ui: &mut egui::Ui, run: &mut Vec<&BrushPreset>, acts: &mut Vec<Action>| {
        if run.is_empty() {
            return;
        }
        if d.layout.cards {
            // The picker's cards: [`card_columns`] cards sharing each row, a centred grid of
            // small [`CARD_TIP_ONLY_W`] cells, or card-wide [`CARD_TIP_NAME_W`] cells snapped
            // to the indent like the full-card rows. The scale floors at the slider's own
            // bottom end.
            let scale = d.scale.max(*crate::brush_picker::SCALE_RANGE.start());
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = vec2(CARD_GAP, GRID_GAP);
                let fixed = if tip_only(d.show) {
                    Some(CARD_TIP_ONLY_W * scale)
                } else if tip_and_name(d.show) {
                    Some(CARD_TIP_NAME_W * scale)
                } else {
                    None
                };
                if let Some(cell) = fixed {
                    // Tip-only cells centre their whole columns like a grid; the card-sized
                    // tip-and-name cells start at the indent.
                    let (mut left, cols) = grid_columns(ui.available_width(), cell, indent, CARD_GAP);
                    if tip_and_name(d.show) {
                        left = indent;
                    }
                    for row in run.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.add_space(left);
                            for p in row {
                                preset_card(ui, p, d.current.is_some_and(|n| p.name.eq_ignore_ascii_case(n)), d.presets, d.show, cell, scale, acts);
                            }
                        });
                    }
                } else {
                    let (card_w, cols) = card_columns(ui.available_width(), indent, scale);
                    for row in run.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.add_space(indent);
                            for p in row {
                                preset_card(ui, p, d.current.is_some_and(|n| p.name.eq_ignore_ascii_case(n)), d.presets, d.show, card_w, scale, acts);
                            }
                        });
                    }
                }
            });
        } else if d.grid {
            // Rows of whole columns, each with the same margin: a wrapped row indented only
            // its first line and left the spare width on the right.
            let (left, cols) = grid_columns(ui.available_width(), d.layout.cell, d.layout.indent + indent, GRID_GAP);
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing = vec2(GRID_GAP, GRID_GAP);
                for row in run.chunks(cols) {
                    ui.horizontal(|ui| {
                        ui.add_space(left);
                        for p in row {
                            grid_cell(ui, p, d.current.is_some_and(|n| p.name.eq_ignore_ascii_case(n)), d.presets, d.layout.cell, acts);
                        }
                    });
                }
            });
        } else {
            for p in run.iter() {
                list_row(ui, p, d.current.is_some_and(|n| p.name.eq_ignore_ascii_case(n)), d.presets, indent, acts);
            }
        }
        run.clear();
    };
    for n in nodes {
        match n {
            Node::Preset(p) => run.push(p),
            Node::Folder(f) => {
                flush(ui, &mut run, acts);
                let open = d.filtering || !d.collapsed.contains(&folder_view_key(d.label, &f.path));
                group_header(ui, d.label, d.key, &f.path, open, f.count, acts);
                if open && depth < MAX_VIEW_DEPTH {
                    draw_nodes(ui, d, &f.children, depth + 1, acts);
                }
            }
        }
    }
    flush(ui, &mut run, acts);
}

/// The rename bar shown while a preset or group is being renamed. Enter or OK renames, Escape or
/// Cancel stops.
fn rename_bar(ui: &mut egui::Ui, st: &mut BrushesPanelState, acts: &mut Vec<Action>) {
    let Some(r) = st.renaming.as_mut() else { return };
    let t = Tokens::get(ui.ctx());
    if r.text.is_empty() {
        r.text = r.folder.last().unwrap_or(&r.name).clone();
    }
    let mut done = false;
    ui.horizontal(|ui| {
        ui.label(RichText::new(if r.group { tl!("Group name") } else { tl!("Brush name") }).color(t.text_dim));
        let resp = ui.add(egui::TextEdit::singleline(&mut r.text).desired_width((ui.available_width() - 140.0).max(60.0)).id_salt("brush-rename"));
        if !resp.has_focus() && !resp.lost_focus() {
            resp.request_focus();
        }
        let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
        let clicked = widgets::dialog_buttons(
            ui,
            &[
                widgets::DialogButton::new(widgets::ButtonRole::Default, tl!("OK"), 52.0),
                widgets::DialogButton::new(widgets::ButtonRole::Cancel, tl!("Cancel"), 60.0),
            ],
        );
        if clicked == Some(widgets::ButtonRole::Default) || enter {
            acts.push(Action::Rename(r.clone()));
            done = true;
        }
        if clicked == Some(widgets::ButtonRole::Cancel) || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            done = true;
        }
    });
    if done {
        st.renaming = None;
    }
    ui.add_space(4.0);
}

/// The presets in their groups, filtered by `st.filter`, as rows or tip cells (`st.view`) or as
/// the picker's cards (`layout.cards`, drawn from the `st.show_*` parts), with the rename bar
/// above them while a rename is open. `current` is the selected preset's name: selection is by
/// identity, so it survives edits and tells look-alike duplicates apart. Group toggles and renames
/// update `st`; the returned actions are commands for [`apply`].
pub fn preset_list(ui: &mut egui::Ui, presets: &[BrushPreset], current: Option<&str>, st: &mut BrushesPanelState, layout: ListLayout) -> Vec<Action> {
    let mut acts = Vec::new();
    rename_bar(ui, st, &mut acts);
    let filter = st.filter.trim().to_lowercase();
    let grid = st.view == BrushesView::Grid;
    let show = (st.show_name, st.show_stroke, st.show_tip);
    let collapsed = &st.collapsed;
    egui::ScrollArea::vertical().id_salt(layout.id).max_height(layout.max_height).auto_shrink([false, true]).show(ui, |ui| {
        ui.spacing_mut().item_spacing.y = 1.0;
        for (label, items) in grouped_presets(presets) {
            let items: Vec<&BrushPreset> =
                items.into_iter().filter_map(|i| presets.get(i)).filter(|p| filter.is_empty() || p.name.to_lowercase().contains(&filter)).collect();
            if items.is_empty() {
                continue;
            }
            let key = group_key(presets, &label);
            let open = !filter.is_empty() || !collapsed.contains(&label);
            group_header(ui, &label, &key, &[], open, items.len(), &mut acts);
            if !open {
                continue;
            }
            let d = Draw { label: &label, key: &key, presets, current, collapsed, filtering: !filter.is_empty(), grid, layout, show, scale: st.scale };
            draw_nodes(ui, &d, &folder_tree(&items), 0, &mut acts);
            ui.add_space(2.0);
        }
        if presets.is_empty() {
            ui.label(RichText::new(tl!("No brush presets")).color(Tokens::get(ui.ctx()).text_faint));
        }
    });
    // The dragged preset or group follows the pointer.
    if let Some(d) = egui::DragAndDrop::payload::<BrushDrag>(ui.ctx())
        && let Some(p) = ui.ctx().pointer_interact_pos()
    {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
        let name = match &*d {
            BrushDrag::Preset(n) => n.clone(),
            BrushDrag::Group(g) => {
                if g.is_empty() {
                    UNGROUPED.to_string()
                } else {
                    g.clone()
                }
            }
            BrushDrag::Folder { folder, .. } => folder.last().cloned().unwrap_or_default(),
        };
        egui::Area::new(egui::Id::new(("brush-drag-label", layout.id))).order(egui::Order::Tooltip).fixed_pos(p + vec2(12.0, 8.0)).interactable(false).show(
            ui.ctx(),
            |ui| {
                egui::Frame::popup(ui.style()).show(ui, |ui| ui.label(name));
            },
        );
    }
    view_actions(st, acts)
}

/// The Brushes tab.
pub fn show(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    // Size of the current brush (Photoshop's Brushes panel slider).
    let before = app.session.tools.brush.clone();
    let mut b = before.clone();
    ui.horizontal(|ui| {
        ui.label(RichText::new(tl!("Size")).color(t.text_dim));
        let mut lv = b.size.max(1.0).ln();
        ui.add_sized(vec2(WIDTH - 140.0, 18.0), |ui: &mut egui::Ui| {
            let r = widgets::slider(ui, &mut lv, 0.0..=MAX_BRUSH_SIZE.ln(), None);
            if r.changed() {
                b.size = lv.exp().round().clamp(1.0, MAX_BRUSH_SIZE);
            }
            r
        });
        let mut s = b.size;
        if widgets::value_field(ui, &mut s, 1.0..=MAX_BRUSH_SIZE, "px", 74.0).changed() {
            b.size = s.round().clamp(1.0, MAX_BRUSH_SIZE);
        }
    });
    commit_gesture(app, ui.ctx(), &before, &b);
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        icons::paint(ui, egui::Rect::from_min_size(ui.cursor().min + vec2(0.0, 3.0), vec2(16.0, 16.0)), "search", 14.0, t.text_faint);
        ui.add_space(20.0);
        widgets::text_field(ui, ui.id().with("brush-search"), &mut app.ui.brushes_panel.filter, tl!("Search Brushes"), WIDTH - 120.0, false);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            let view = &mut app.ui.brushes_panel.view;
            let grid_tip = "Grid view";
            let list_tip = "List view";
            if icons::button(ui, "grid-2x2", 24.0, *view == BrushesView::Grid, tl!(grid_tip)).clicked() {
                *view = BrushesView::Grid;
            }
            if icons::button(ui, "align-justify", 24.0, *view == BrushesView::List, tl!(list_tip)).clicked() {
                *view = BrushesView::List;
            }
        });
    });
    ui.add_space(6.0);
    let acts = preset_list(ui, &app.session.tools.presets, app.session.tools.current_preset.as_deref(), &mut app.ui.brushes_panel, PANEL_LIST);
    apply(app, acts);
    ui.add_space(4.0);
    widgets::hairline(ui);
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(crate::i18n::trn(crate::i18n::current(), app.session.tools.presets.len() as u64, "{n} preset", "{n} presets")).color(t.text_faint),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // The current preset by identity: editing the brush after picking must not deselect it.
            let current = app.session.tools.current_preset.clone().filter(|n| app.session.tools.presets.iter().any(|p| p.name.eq_ignore_ascii_case(n)));
            if ui.add_enabled_ui(current.is_some(), |ui| icons::button(ui, "trash", 24.0, false, "Delete brush")).inner.clicked()
                && let Some(name) = current
            {
                run_or_status(app, "brush.presets.delete", json!({ "name": name }));
            }
            let new = icons::button(ui, "square-plus", 24.0, false, "Create new brush from the current settings");
            if crate::brush_picker::named(new, "Create new brush from the current settings").clicked() {
                crate::brush_preset_dialog::open(app, None);
            }
            if icons::button(ui, "folder-open", 24.0, false, "Import Brushes… (.abr)").clicked() {
                let _ = app.open_dialog_file();
            }
        });
    });
    // Forget previews of presets that no longer exist.
    let names: std::collections::HashSet<&str> = app.session.tools.presets.iter().map(|p| p.name.as_str()).collect();
    brush_preview::with_cache(ui.ctx(), |c| {
        c.retain(|slot| slot.split_once(':').is_none_or(|(_, n)| names.contains(n)));
    });
}

#[cfg(test)]
#[path = "brushes_tab_tests.rs"]
mod tests;
