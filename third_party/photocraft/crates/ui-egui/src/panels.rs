//! Chrome around the canvas: title bar, options bar, toolbar, status bar, dock cards, Properties.

use egui::{Align2, Color32, CornerRadius, Pos2, Rect, RichText, Sense, Stroke, StrokeKind, pos2, vec2};
use photocraft_color::BlendMode;
use photocraft_doc::{Layer, LayerContent, LayerId};
use serde_json::{Value, json};

use crate::mask_thumbs_ui::MaskKind;
use crate::state::Tool;
use crate::theme::{self, Tokens};
use crate::{PhotocraftApp, icons, widgets};

// ----------------------------------------------------------------------------- toolbar

/// Tool groups separated by hairlines, Photoshop order.
/// Photoshop toolbar: divider-separated sections of slots; each slot is a flyout group whose button
/// shows the group's last-used tool.
const TOOL_SECTIONS: &[&[&[Tool]]] = &[
    &[&[Tool::Move, Tool::Artboard]],
    &[
        &[Tool::RectMarquee, Tool::EllipseMarquee, Tool::SingleRowMarquee, Tool::SingleColumnMarquee],
        &[Tool::Lasso, Tool::PolygonLasso, Tool::MagneticLasso],
        &[Tool::ObjectSelection, Tool::QuickSelection, Tool::MagicWand],
        &[Tool::Crop, Tool::PerspectiveCrop, Tool::Slice, Tool::SliceSelect],
        &[Tool::Frame],
        &[Tool::Eyedropper, Tool::ColorSampler, Tool::Ruler, Tool::Note, Tool::Count],
    ],
    &[
        &[Tool::Remove, Tool::SpotHealing, Tool::Healing, Tool::Patch, Tool::ContentAwareMove, Tool::RedEye],
        &[Tool::Brush, Tool::Pencil, Tool::MixerBrush, Tool::ColorReplacement],
        &[Tool::CloneStamp, Tool::PatternStamp],
        &[Tool::HistoryBrush],
        &[Tool::Eraser, Tool::BackgroundEraser, Tool::MagicEraser],
        &[Tool::Gradient, Tool::PaintBucket],
        &[Tool::Blur, Tool::Sharpen, Tool::Smudge],
        &[Tool::Dodge, Tool::Burn, Tool::Sponge],
    ],
    &[
        &[Tool::Pen, Tool::FreeformPen, Tool::CurvaturePen, Tool::AddAnchorPoint, Tool::DeleteAnchorPoint, Tool::ConvertPoint],
        &[Tool::Type, Tool::VerticalType, Tool::VerticalTypeMask, Tool::HorizontalTypeMask],
        &[Tool::PathSelection, Tool::DirectSelection],
        &[Tool::Rectangle, Tool::EllipseShape, Tool::Triangle, Tool::Polygon, Tool::Line, Tool::CustomShape],
    ],
    &[&[Tool::Hand, Tool::RotateView], &[Tool::Zoom]],
];

/// The tool a slot shows: the current tool if it belongs to the slot, else the last one used.
fn slot_tool(ui: &egui::Ui, current: Tool, slot: &[Tool], key: egui::Id) -> Tool {
    if slot.contains(&current) {
        ui.data_mut(|d| d.insert_temp(key, current));
        return current;
    }
    ui.data(|d| d.get_temp::<Tool>(key)).filter(|t| slot.contains(t)).unwrap_or(slot[0])
}

/// [`TOOL_SECTIONS`] without the tools hidden in Edit › Toolbar (`hidden` holds tool names: the
/// dialog writes `Tool` debug names, `edit.toolbar` takes any name `Tool::from_name` reads). A
/// slot keeps only its visible tools, so its flyout lists just those; a slot with none left is
/// dropped, and so is a section with no slots. Each slot carries its index in `TOOL_SECTIONS`,
/// so its remembered tool and flyout stay put when other slots are hidden.
fn visible_sections(hidden: &[String], page: bool) -> Vec<Vec<(usize, Vec<Tool>)>> {
    let hidden: Vec<Tool> = hidden.iter().map(String::as_str).filter_map(Tool::from_name).collect();
    let mut index = 0usize;
    let mut sections = Vec::new();
    for section in TOOL_SECTIONS {
        let mut slots = Vec::new();
        for slot in section.iter() {
            let tools: Vec<Tool> = slot.iter().copied().filter(|tool| !hidden.contains(tool) && (!page || crate::tool_shortcuts::page_tool(*tool))).collect();
            if !tools.is_empty() {
                slots.push((index, tools));
            }
            index += 1;
        }
        if !slots.is_empty() {
            sections.push(slots);
        }
    }
    sections
}

pub fn toolbar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let (w1, bx, m) = if t.pro { (40.0, 30.0, 5i8) } else { (50.0, 36.0, 7i8) };
    let sections = visible_sections(&app.session.prefs().toolbar.hidden, app.services.host_save.is_some());
    // Two columns when the header chevron asks for them; a column that doesn't fit scrolls.
    let double = app.ui.panels.toolbar_double;
    let w = if double { w1 + bx + 2.0 } else { w1 };
    crate::direction::start_panel("toolbar")
        .resizable(false)
        .exact_size(w)
        .frame(egui::Frame::NONE.fill(t.chrome).inner_margin(egui::Margin::symmetric(m, 8)))
        .show(ui, |ui| {
            if t.pro {
                // Photoshop's toolbar header chevrons switch between one and two columns.
                let (cr, resp) = ui.allocate_exact_size(vec2(bx, 14.0), Sense::click());
                // The chevrons point the way the toolbar grows or shrinks: towards the canvas.
                let (shrink, grow) = if crate::direction::is_rtl() { ("chevrons-right", "chevrons-left") } else { ("chevrons-left", "chevrons-right") };
                icons::paint(ui, cr, if double { shrink } else { grow }, 11.0, if resp.hovered() { t.text } else { t.text_faint });
                resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Toolbar")));
                if resp.clicked() {
                    app.ui.panels.toolbar_double = !app.ui.panels.toolbar_double;
                }
                ui.add_space(4.0);
            }
            // Subtle violet wash at the bottom of the toolbar.
            let full = ui.max_rect();
            if !t.bevel && !t.pro && t.dark() {
                let mut mesh = egui::Mesh::default();
                let r = Rect::from_min_max(pos2(full.left() - 7.0, full.bottom() - 260.0), pos2(full.right() + 7.0, full.bottom() + 8.0));
                let top = Color32::TRANSPARENT;
                let bottom = Color32::from_rgba_unmultiplied(90, 70, 190, 34);
                mesh.colored_vertex(r.left_top(), top);
                mesh.colored_vertex(r.right_top(), top);
                mesh.colored_vertex(r.right_bottom(), bottom);
                mesh.colored_vertex(r.left_bottom(), bottom);
                mesh.add_triangle(0, 1, 2);
                mesh.add_triangle(0, 2, 3);
                ui.painter().add(mesh);
            }
            ui.spacing_mut().item_spacing = vec2(2.0, 3.0);
            let flyout_id = egui::Id::new("tool-flyout");
            let held_id = flyout_id.with("held");
            if ui.input(|i| i.pointer.any_pressed()) {
                ui.data_mut(|d| d.remove::<egui::Id>(held_id));
            }
            // No tool tooltips over an open flyout, or while a button is held to open one (#4450):
            // Photoshop shows none there. (The flyout is our own Area, so egui's rule for popups
            // doesn't hide them.)
            let tool_tips = ui.data(|d| d.get_temp::<(egui::Id, Rect)>(flyout_id)).is_none() && !ui.input(|i| i.pointer.primary_down());
            let ctx = ui.ctx().clone();
            let footer = (if t.pro { 8.0 } else { 14.0 })
                + chips_height(t.pro, t.round_chips, t.round_chips && double)
                + if t.pro { 8.0 + if double { bx } else { bx * 2.0 + 3.0 } } else { 0.0 }
                + 6.0;
            // As tall as the tools, so the colour chips sit right under them (Photoshop); a column
            // taller than the window scrolls and keeps the chips at the bottom.
            egui::ScrollArea::vertical()
                .id_salt("toolbar-columns")
                .auto_shrink([false, true])
                .max_height((ui.available_height() - footer).max(0.0))
                .min_scrolled_height(0.0)
                .show(ui, |ui| {
                    // Pro has no group dividers, so its two columns fill every row across groups.
                    let sections: Vec<Vec<(usize, Vec<Tool>)>> = if t.pro { vec![sections.into_iter().flatten().collect()] } else { sections };
                    for (si, section) in sections.iter().enumerate() {
                        // Photoshop 2026 draws one uninterrupted column (no group dividers).
                        if si > 0 && !t.pro {
                            ui.add_space(4.0);
                            let (r, _) = ui.allocate_exact_size(vec2(if double { bx * 2.0 + 2.0 } else { bx }, 1.0), Sense::hover());
                            ui.painter().line_segment([r.left_center() + vec2(8.0, 0.0), r.right_center() - vec2(8.0, 0.0)], Stroke::new(1.0, t.separator));
                            ui.add_space(4.0);
                        }
                        let rows: Vec<&[(usize, Vec<Tool>)]> = if double { section.chunks(2).collect() } else { section.chunks(1).collect() };
                        for row in rows {
                            ui.horizontal(|ui| {
                                for (slot_index, slot) in row.iter() {
                                    let key = egui::Id::new(("tool-slot", *slot_index));
                                    let tool = slot_tool(ui, app.ui.tool, slot, key);
                                    let sel = slot.contains(&app.ui.tool);
                                    let tip = match crate::tool_shortcuts::shortcut(app, tool) {
                                        Some(sc) => format!("{}  ({})", tl!(tool.label()), crate::shortcuts::pretty(&sc)),
                                        None => tl!(tool.label()).to_string(),
                                    };
                                    let resp =
                                        icons::button_with_icon_size_tip(ui, icons::tool_icon(tool), bx, toolbar_icon_size(bx, t.pro), sel, &tip, tool_tips);
                                    if slot.len() > 1 {
                                        paint_toolbar_flyout_indicator(ui, &t, &resp);
                                    }
                                    if resp.clicked() && ui.data(|d| d.get_temp::<egui::Id>(held_id)) != Some(key) {
                                        app.ui.tool = tool;
                                    }
                                    // Double-clicking the Hand tool fits the image on screen (Photoshop).
                                    if resp.double_clicked() && tool == Tool::Hand {
                                        app.ui.tool = tool;
                                        // With no document open there is nothing to fit.
                                        let _ = app.run("view.fitOnScreen", json!({}));
                                    }
                                    if resp.double_clicked() && tool == Tool::Zoom {
                                        app.ui.tool = tool;
                                        let _ = app.run("view.actualPixels", json!({}));
                                    }
                                    // Right-click or long-press opens the flyout (Photoshop).
                                    let held_for =
                                        resp.is_pointer_button_down_on().then(|| ui.input(|i| i.pointer.press_start_time().map(|t0| i.time - t0))).flatten();
                                    if slot.len() > 1
                                        && let Some(seconds) = held_for
                                        && seconds < 0.35
                                    {
                                        ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(0.35 - seconds));
                                    }
                                    let long_press = held_for.is_some_and(|seconds| seconds >= 0.35);
                                    // Clicking the button of its open flyout again, with either button, closes
                                    // it (#4932); the release that ends the long press that opened it doesn't.
                                    let open_here = ui.data(|d| d.get_temp::<(egui::Id, Rect)>(flyout_id)).is_some_and(|(k, _)| k == key);
                                    let reclicked = open_here
                                        && (resp.secondary_clicked() || (resp.clicked() && ui.data(|d| d.get_temp::<egui::Id>(held_id)) != Some(key)));
                                    if reclicked {
                                        ui.data_mut(|d| d.remove::<(egui::Id, Rect)>(flyout_id));
                                    } else if slot.len() > 1 && (resp.secondary_clicked() || long_press) {
                                        ui.data_mut(|d| {
                                            d.insert_temp(flyout_id, (key, resp.rect));
                                            if long_press {
                                                d.insert_temp(held_id, key);
                                            }
                                        });
                                    }
                                    if slot.len() > 1 && long_press {
                                        // Holding a tool button selects the tool it shows, as a click would (#4453).
                                        app.ui.tool = tool;
                                        ui.ctx().request_repaint();
                                    }
                                    if let Some((open_key, anchor)) = ui.data(|d| d.get_temp::<(egui::Id, Rect)>(flyout_id))
                                        && open_key == key
                                    {
                                        // Beside the tool, towards the canvas.
                                        let (pivot, pos) = if crate::direction::is_rtl() {
                                            (egui::Align2::RIGHT_TOP, anchor.left_top() - vec2(6.0, 0.0))
                                        } else {
                                            (egui::Align2::LEFT_TOP, anchor.right_top() + vec2(6.0, 0.0))
                                        };
                                        let area = egui::Area::new(key.with("flyout")).order(egui::Order::Foreground).pivot(pivot).fixed_pos(pos).show(
                                            ui.ctx(),
                                            |ui| {
                                                egui::Frame::popup(ui.style()).show(ui, |ui| {
                                                    let label_w = slot
                                                        .iter()
                                                        .map(|it| {
                                                            ui.painter()
                                                                .layout_no_wrap(tl!(it.label()).to_string(), egui::FontId::proportional(12.5), t.text)
                                                                .size()
                                                                .x
                                                        })
                                                        .fold(0.0f32, f32::max);
                                                    let fw = (label_w + 42.0 + 40.0).max(180.0);
                                                    for &item in slot.iter() {
                                                        let (r, ir) = ui.allocate_exact_size(vec2(fw, 26.0), Sense::click());
                                                        if ir.hovered() {
                                                            ui.painter().rect_filled(r, 3.0, t.hover);
                                                        }
                                                        if item == tool {
                                                            ui.painter().rect_filled(
                                                                Rect::from_center_size(pos2(r.left() + 8.0, r.center().y), vec2(4.0, 4.0)),
                                                                0.0,
                                                                t.text,
                                                            );
                                                        }
                                                        icons::paint(
                                                            ui,
                                                            Rect::from_center_size(pos2(r.left() + 26.0, r.center().y), vec2(18.0, 18.0)),
                                                            icons::tool_icon(item),
                                                            14.0,
                                                            t.icon,
                                                        );
                                                        ui.painter().text(
                                                            pos2(r.left() + 42.0, r.center().y),
                                                            Align2::LEFT_CENTER,
                                                            tl!(item.label()),
                                                            egui::FontId::proportional(12.5),
                                                            t.text,
                                                        );
                                                        if let Some(sc) = crate::tool_shortcuts::shortcut(app, item) {
                                                            ui.painter().text(
                                                                pos2(r.right() - 8.0, r.center().y),
                                                                Align2::RIGHT_CENTER,
                                                                crate::shortcuts::pretty(&sc),
                                                                egui::FontId::proportional(12.0),
                                                                t.text_dim,
                                                            );
                                                        }
                                                        if ir.clicked() {
                                                            app.ui.tool = item;
                                                            ui.data_mut(|d| d.remove::<(egui::Id, Rect)>(flyout_id));
                                                        }
                                                    }
                                                });
                                            },
                                        );
                                        // Any press outside the flyout and its button closes it, as in Photoshop: a
                                        // drag on the canvas (cropping, painting) isn't a click (#4460).
                                        let pressed_outside = ui.input(|i| {
                                            i.pointer.any_pressed()
                                                && i.pointer.press_origin().is_some_and(|p| !area.response.rect.contains(p) && !resp.rect.contains(p))
                                        });
                                        if pressed_outside || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                            ui.data_mut(|d| d.remove::<(egui::Id, Rect)>(flyout_id));
                                        }
                                    }
                                }
                            });
                        }
                    }
                    if t.pro && app.services.host_save.is_none() && icons::button(ui, "ellipsis", bx, false, tl!("Edit Toolbar…")).clicked() {
                        let _ = crate::menus::invoke(app, &ctx, "edit.toolbar", json!({}));
                    }
                    reveal_focused_control(ui);
                });
            ui.add_space(if t.pro { 8.0 } else { 14.0 });
            color_chips(app, ui, double);
            if t.pro {
                ui.add_space(8.0);

                let layout = if double { egui::Layout::left_to_right(egui::Align::Min) } else { egui::Layout::top_down(egui::Align::Min) };

                ui.with_layout(layout, |ui| {
                    let quick_mask = app.session.active().is_some_and(|s| s.doc.quick_mask.is_some());
                    if icons::button_with_icon_size_tip(
                        ui,
                        "square-dashed",
                        bx,
                        (bx * 0.52).round(),
                        quick_mask,
                        if quick_mask { tl!("Edit in Standard Mode  (Q)") } else { tl!("Edit in Quick Mask Mode  (Q)") },
                        tool_tips,
                    )
                    .clicked()
                    {
                        let _ = crate::menus::invoke(app, &ctx, "select.editInQuickMaskMode", json!({}));
                    }
                    if app.services.host_save.is_some() {
                        return;
                    }
                    let sm = icons::button_with_icon_size_tip(ui, "app-window", bx, (bx * 0.52).round(), false, tl!("Change Screen Mode  (F)"), tool_tips);
                    paint_toolbar_flyout_indicator(ui, &t, &sm);

                    // Right-click, or click and hold like the tool flyouts (#4444), lists the modes,
                    // like Photoshop's flyout. The release that ends a hold doesn't cycle the mode.
                    let held_id = sm.id.with("held");
                    let was_held = ui.data(|d| d.get_temp::<bool>(held_id)).unwrap_or(false);
                    let held_for = sm.is_pointer_button_down_on().then(|| ui.input(|i| i.pointer.press_start_time().map(|t0| i.time - t0))).flatten();
                    if let Some(seconds) = held_for
                        && seconds < 0.35
                    {
                        ui.ctx().request_repaint_after(std::time::Duration::from_secs_f64(0.35 - seconds));
                    }
                    let long_press = !was_held && held_for.is_some_and(|seconds| seconds >= 0.35);
                    if long_press {
                        ui.data_mut(|d| d.insert_temp(held_id, true));
                    } else if was_held && !sm.is_pointer_button_down_on() {
                        ui.data_mut(|d| d.remove::<bool>(held_id));
                    }
                    let cycle = sm.clicked() && !was_held;
                    if cycle {
                        let _ = crate::menus::invoke(app, &ctx, "view.screenMode.cycle", json!({}));
                    }
                    let open = if sm.secondary_clicked() || long_press {
                        Some(egui::SetOpenCommand::Bool(true))
                    } else if cycle {
                        Some(egui::SetOpenCommand::Bool(false))
                    } else {
                        None
                    };
                    // The release ending the hold counts as a click; it mustn't close the list it opened.
                    let close = if was_held || long_press { egui::PopupCloseBehavior::IgnoreClicks } else { egui::PopupCloseBehavior::CloseOnClick };
                    egui::Popup::menu(&sm).open_memory(open).close_behavior(close).at_pointer_fixed().show(|ui| {
                        ui.set_min_width(220.0);
                        for (id, label) in [
                            ("view.screenMode.standard", tl!("Standard Screen Mode")),
                            ("view.screenMode.fullScreenWithMenuBar", tl!("Full Screen Mode With Menu Bar")),
                            ("view.screenMode.fullScreen", tl!("Full Screen Mode")),
                        ] {
                            let on = crate::view_cmds::checked(app, id).unwrap_or(false);
                            if ui.add(egui::Button::selectable(on, label)).clicked() {
                                let _ = crate::menus::invoke(app, &ctx, id, json!({}));
                                ui.close();
                            }
                        }
                    });
                });
            }
        });
}

/// Glyph size of a toolbar tool button. Pro fills two thirds of the button, close to Photoshop's
/// toolbar glyphs (#2665); the other themes keep the general icon-button proportion.
fn toolbar_icon_size(box_size: f32, pro: bool) -> f32 {
    (box_size * if pro { 2.0 / 3.0 } else { 0.52 }).round()
}

/// Paints a triangle on the bottom corner of an icon button that faces the canvas (bottom right,
/// bottom left in a right-to-left UI) to indicate that a toolbar item has a submenu.
pub fn paint_toolbar_flyout_indicator(ui: &mut egui::Ui, tokens: &Tokens, resp: &egui::Response) {
    let r = resp.rect;
    let tri = if crate::direction::is_rtl() {
        vec![r.left_bottom() + vec2(2.0, -2.0), r.left_bottom() + vec2(6.0, -2.0), r.left_bottom() + vec2(2.0, -6.0)]
    } else {
        vec![r.right_bottom() + vec2(-2.0, -2.0), r.right_bottom() + vec2(-6.0, -2.0), r.right_bottom() + vec2(-2.0, -6.0)]
    };
    ui.painter().add(egui::Shape::convex_polygon(tri, tokens.text_faint, Stroke::NONE));
}

fn c32(c: [f32; 4]) -> Color32 {
    Color32::from_rgba_unmultiplied((c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8, (c[3] * 255.0) as u8)
}

/// Colour chips: chip size and the background chip's offset (points).
fn chip_metrics(pro: bool) -> (f32, f32) {
    if pro { (18.0, 10.0) } else { (21.0, 12.0) }
}

/// Height of the default-colours and swap icons over the chips.
const CHIP_ICON: f32 = 13.0;

/// Studio themes: round chip diameter and the background chip's vertical offset (points).
const STUDIO_CHIP: (f32, f32) = (30.0, 19.0);
/// Room under the round chips for the Default Colors and Switch Colors buttons.
const STUDIO_ROW: f32 = 15.0;
/// Room above the stacked round chips so Default Colors, at their top-right, clears them.
const STUDIO_TOP: f32 = 6.0;

/// Height of the colour chips block.
fn chips_height(pro: bool, studio: bool, wide: bool) -> f32 {
    if studio && wide {
        return STUDIO_CHIP.0 + STUDIO_ROW;
    }
    if studio {
        return STUDIO_TOP + STUDIO_CHIP.0 + STUDIO_CHIP.1 + STUDIO_ROW;
    }
    let (chip, step) = chip_metrics(pro);
    CHIP_ICON + 3.0 + chip + step
}

/// Photoshop's foreground / background colour chips: the Default Colors (D) and Switch Colors (X)
/// icons above them, the foreground chip over the background one, all centred in the toolbar
/// column and kept inside it.
fn color_chips(app: &mut PhotocraftApp, ui: &mut egui::Ui, two_columns: bool) {
    let t = Tokens::get(ui.ctx());
    // Studio themes: round chips stacked vertically, the controls below them.
    let studio = t.round_chips;
    let (chip, step) = if studio { STUDIO_CHIP } else { chip_metrics(t.pro) };
    // With two tool columns (a short window) the round chips lie side by side, which saves height.
    let wide = studio && two_columns;
    let group = if studio && !wide { chip } else { chip + step };
    let w = ui.available_width().max(group);
    let (rect, _) = ui.allocate_exact_size(vec2(w, chips_height(t.pro, studio, wide)), Sense::hover());
    let left = (rect.left() + (w - group) / 2.0).round();
    let (fg, bg) = if wide {
        let fg = Rect::from_min_size(pos2(left, rect.top()), vec2(chip, chip));
        (fg, fg.translate(vec2(step, 0.0)))
    } else if studio {
        let fg = Rect::from_min_size(pos2(left, rect.top() + STUDIO_TOP), vec2(chip, chip));
        (fg, fg.translate(vec2(0.0, step)))
    } else {
        let fg = Rect::from_min_size(pos2(left, rect.top() + CHIP_ICON + 3.0), vec2(chip, chip));
        (fg, fg.translate(vec2(step, step)))
    };
    let radius = t.radius_sm.min(3.0);
    let p = ui.painter();
    if studio {
        let dot = |r: Rect, color: Color32| {
            p.circle_filled(r.center(), r.width() / 2.0, color);
            // A light line inside a dark one, so any colour reads against the toolbar.
            p.circle_stroke(r.center(), r.width() / 2.0 - 0.5, Stroke::new(1.0, t.text.gamma_multiply(0.9)));
            p.circle_stroke(r.center(), r.width() / 2.0 + 0.5, Stroke::new(1.0, t.field_border));
        };
        dot(bg, c32(app.session.tools.background));
        // The foreground chip sits on the background one, a toolbar-coloured gap between them.
        p.circle_filled(fg.center(), fg.width() / 2.0 + 2.0, t.chrome);
        dot(fg, c32(app.session.tools.foreground));
    } else {
        let frame = |r: Rect| {
            // A light line inside a dark one, so any colour reads against the toolbar.
            p.rect_stroke(r, radius, Stroke::new(1.0, t.text.gamma_multiply(0.9)), StrokeKind::Inside);
            p.rect_stroke(r, radius, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
        };
        p.rect_filled(bg, radius, c32(app.session.tools.background));
        frame(bg);
        p.rect_filled(fg.expand(2.0), radius + 2.0, t.chrome);
        p.rect_filled(fg, radius, c32(app.session.tools.foreground));
        frame(fg);
    }
    // Photoshop: clicking a chip opens the Color Picker for that colour.
    let bg_resp = ui.interact(bg, ui.id().with("bgchip"), Sense::click());
    let fg_resp = ui.interact(fg, ui.id().with("fgchip"), Sense::click());
    if fg_resp.on_hover_text(tl!("Set foreground color")).clicked() {
        crate::color_picker_ui::open(app, "foreground");
    } else if bg_resp.on_hover_text(tl!("Set background color")).clicked() {
        crate::color_picker_ui::open(app, "background");
    }
    // Default Colors at the left, Switch Colors at the right, over the chips.
    let (dr, sr) = if studio {
        // Switch Colors is a curve right under the chips, centred on them so its heads touch
        // them. Default Colors never moves it: at the left edge beside it when the chips lie side
        // by side, at the block's top-right corner when they are stacked.
        let (cx, row) = ((fg.center().x + bg.center().x) / 2.0, fg.bottom().max(bg.bottom()) + 3.5);
        let default = if wide {
            Rect::from_center_size(pos2(rect.left() + CHIP_ICON / 2.0, row), vec2(CHIP_ICON, CHIP_ICON))
        } else {
            // Just high enough that its bottom-left corner clears the foreground chip's outline.
            let x = rect.right() - CHIP_ICON;
            let (dx, r) = ((x - fg.center().x).max(0.0), fg.width() / 2.0 + 1.0);
            let clear = if dx < r { (r * r - dx * dx).sqrt() } else { 0.0 };
            Rect::from_min_size(pos2(x, fg.center().y - clear - CHIP_ICON), vec2(CHIP_ICON, CHIP_ICON))
        };
        (Some(default), Rect::from_center_size(pos2(cx, row), vec2(20.0, 14.0)))
    } else {
        let icon = |x: f32| Rect::from_min_size(pos2(x, rect.top()), vec2(CHIP_ICON, CHIP_ICON));
        (Some(icon(left)), icon(left + group - CHIP_ICON))
    };
    let d = dr.map(|dr| ui.interact(dr, ui.id().with("default-colors"), Sense::click()));
    let s = ui.interact(sr, ui.id().with("swap-colors"), Sense::click());
    let p = ui.painter();
    for (r, resp) in dr.into_iter().zip(d.iter()).chain([(sr, &s)]) {
        if resp.hovered() && !studio {
            p.rect_filled(r.expand(2.0), radius, t.hover);
        }
    }
    let ink = |resp: &egui::Response| if resp.hovered() { t.text } else { t.icon };
    if studio {
        // Switch Colors: a shallow curve hugging the bottom chip, with a filled head at each end.
        // It glows on hover instead of getting a button box.
        let hot = s.hovered();
        let stroke = Stroke::new(1.5, ink(&s));
        let (c, rad) = (pos2(sr.center().x, sr.center().y - 5.0), 9.0);
        let (from, to) = (0.45, std::f32::consts::PI - 0.45);
        let arc: Vec<_> = (0..=12)
            .map(|i| {
                let a = from + (to - from) * i as f32 / 12.0;
                c + vec2(rad * a.cos(), rad * a.sin())
            })
            .collect();
        if hot {
            p.add(egui::Shape::line(arc.clone(), Stroke::new(5.0, t.text.gamma_multiply(0.18))));
        }
        let ends = [(arc[0], from, -1.0_f32), (arc[arc.len() - 1], to, 1.0)];
        p.add(egui::Shape::line(arc, stroke));
        // Each head points along the curve, away from its middle.
        for (end, a, sign) in ends {
            let dir = vec2(-a.sin(), a.cos()) * sign;
            let side = vec2(-dir.y, dir.x);
            let tip = end + dir * 2.5;
            let base = tip - dir * 5.0;
            p.add(egui::Shape::convex_polygon(vec![tip, base + side * 3.0, base - side * 3.0], ink(&s), Stroke::NONE));
        }
    } else {
        // Switch Colors: the compact clockwise bend has left- and down-pointing heads.
        // swap_arrow.rs owns its geometry and the pixel-level render regression test.
        crate::swap_arrow::paint(p, sr.min, ink(&s));
    }
    // Default Colors: a small black chip over a small white one.
    if let (Some(dr), Some(d)) = (dr, d.as_ref()) {
        let small = 7.0;
        let (b, w) = (Rect::from_min_size(dr.min + vec2(1.0, 1.0), vec2(small, small)), Rect::from_min_size(dr.min + vec2(5.0, 5.0), vec2(small, small)));
        p.rect_filled(w, 1.0, Color32::WHITE);
        p.rect_stroke(w, 1.0, Stroke::new(1.0, ink(d)), StrokeKind::Inside);
        p.rect_filled(b.expand(1.0), 1.5, t.chrome);
        p.rect_filled(b, 1.0, Color32::BLACK);
        p.rect_stroke(b, 1.0, Stroke::new(1.0, ink(d)), StrokeKind::Inside);
    }
    if let Some(d) = &d {
        d.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Default colours (D)")));
    }
    s.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Swap colours (X)")));
    if s.on_hover_text(tl!("Swap colours (X)")).clicked() {
        let _ = app.run("tools.swapColors", json!({}));
    }
    if d.is_some_and(|d| d.on_hover_text(tl!("Default colours (D)")).clicked()) {
        let _ = app.run("tools.defaultColors", json!({}));
    }
}

// ----------------------------------------------------------------------------- title bar

/// The app's top bar: brand mark, menus, the document title and the workspace controls. With
/// [`PhotocraftApp::custom_titlebar`] (Windows and Linux) it is also the window's title bar, as in
/// Photoshop on Windows: the caption buttons take its ends (`titlebar`). With the system
/// title bar (Windows and Linux, `system_title_bar`) the OS draws its own icon and title, so the
/// in-app bar shows menus and workspace controls only: no brand mark, no centred title.
pub fn title_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    if app.services.host_save.is_some() {
        egui::Panel::top("title_bar").exact_size(32.0).frame(egui::Frame::NONE.fill(t.chrome).inner_margin(egui::Margin::symmetric(12, 0))).show(ui, |ui| {
            crate::direction::horizontal_centered(ui, |ui| {
                crate::menus::menu_bar(app, ui);
            });
        });
        return;
    }
    let custom = app.custom_titlebar;
    let system = system_title_bar(app);
    // The caption buttons' room at each end (`titlebar`); a side with buttons has no margin.
    let (caption_left, caption_right) = crate::titlebar::reserved(app, ui.ctx());
    let left = if cfg!(target_os = "macos") && app.integrated_titlebar {
        78
    } else if caption_left > 0.0 {
        0
    } else {
        10
    };
    let right = if caption_right > 0.0 { 0 } else { 10 };
    let bar = egui::Panel::top("title_bar")
        .exact_size(crate::titlebar::bar_height(app, ui.ctx(), if t.pro { 32.0 } else { 38.0 }))
        .frame(egui::Frame::NONE.fill(t.chrome).inner_margin(egui::Margin { left, right, top: 0, bottom: 0 }))
        .show(ui, |ui| {
            let full = ui.max_rect();
            // Only the free gap between the menus and the far-end controls drags the window: a
            // press on a menu title must open the menu, never move the window. The gap is last
            // frame's, as the menus are laid out after this.
            let span_id = ui.id().with("titledrag-span");
            let (gap_l, gap_r) = ui.ctx().data(|d| d.get_temp::<(f32, f32)>(span_id)).unwrap_or((full.right(), full.right()));
            let gap = egui::Rect::from_x_y_ranges(gap_l.max(full.left())..=gap_r.min(full.right()).max(gap_l), full.y_range());
            ui.ctx().data_mut(|d| d.insert_temp(crate::titlebar::modal_drag_area_id(), gap));
            let drag = ui.interact(gap, ui.id().with("titledrag"), Sense::click_and_drag());
            if drag.drag_started() {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::StartDrag);
            }
            if drag.double_clicked() {
                let max = ui.ctx().input(|i| i.viewport().maximized.unwrap_or(false));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Maximized(!max));
            }
            let title = app.session.active().map(|d| format!("{}{}", d.doc.name, if d.is_dirty() { "  •" } else { "" })).unwrap_or_else(|| "PhotoCraft".into());
            // The menus and the far-end controls are laid out first; the title is centred in
            // whatever room is left between them (`free`, left to right: menus then controls, or
            // controls then menus in a right-to-left UI), shortened or dropped rather than drawn
            // over them.
            let rtl = crate::direction::is_rtl();
            let mut free = (full.left(), full.right());
            crate::direction::horizontal_centered(ui, |ui| {
                // The menus start beside the caption buttons at their end, if it has any.
                let lead = if rtl { caption_right } else { caption_left };
                if lead > 0.0 {
                    ui.add_space(lead + 4.0);
                }
                if !system {
                    let side = if t.pro { 18.0 } else { 20.0 };
                    let (mark, _) = ui.allocate_exact_size(vec2(side, side), Sense::hover());
                    crate::brand::paint_mark(ui, mark);
                    ui.add_space(6.0);
                }
                // With the macOS menu bar the menus are at the top of the screen instead. Either way
                // this is the menus' inner edge, facing the title.
                let menus_end = if app.services.native_menu.is_some() {
                    if rtl { ui.cursor().right() } else { ui.cursor().left() }
                } else {
                    crate::menus::menu_bar(app, ui)
                };
                // The menu bar takes the whole row, so the far-end group gets its own rect: from
                // the menus to the bar's end, or to the caption buttons.
                let group = if rtl {
                    let left_edge = if caption_left > 0.0 { full.left() + caption_left + 4.0 } else { full.left() };
                    egui::Rect::from_min_max(egui::pos2(left_edge, full.top()), egui::pos2((menus_end - TITLE_GAP).max(left_edge), full.bottom()))
                } else {
                    let right_edge = if caption_right > 0.0 { full.right() - caption_right - 4.0 } else { full.right() };
                    egui::Rect::from_min_max(egui::pos2((menus_end + TITLE_GAP).min(right_edge), full.top()), egui::pos2(right_edge, full.bottom()))
                };
                let mut group_ui = ui.new_child(egui::UiBuilder::new().max_rect(group).layout(crate::direction::trailing()));
                let controls_end = {
                    let ui = &mut group_ui;
                    ui.spacing_mut().item_spacing.x = 6.0;
                    let mut ws = app.ui.workspace.clone();
                    let opts = [
                        // Keys stay English (`apply_workspace` matches them); the dropdown
                        // shows the labels in the UI language.
                        ("Essentials".to_string(), tl!("Essentials")),
                        ("Photography".to_string(), tl!("Photography")),
                        ("Painting".to_string(), tl!("Painting")),
                        ("Graphic and Web".to_string(), tl!("Graphic and Web")),
                        ("Pixel Art".to_string(), tl!("Pixel Art")),
                        ("Motion".to_string(), tl!("Motion")),
                    ];
                    // A narrow bar drops what is also in a menu, Discord first (below), then
                    // the theme toggle (Preferences), then search (Edit › Search), and narrows
                    // the switcher (Window › Workspace) before anything runs over.
                    let room = ui.available_width();
                    let icons_shown: u8 = if room >= WORKSPACE_MIN + 2.0 * ICON_SLOT {
                        2
                    } else if room >= WORKSPACE_MIN + ICON_SLOT {
                        1
                    } else {
                        0
                    };
                    let ws_width = (room - f32::from(icons_shown) * ICON_SLOT - 6.0).clamp(WORKSPACE_MIN, 130.0);
                    if room >= WORKSPACE_MIN && widgets::dropdown(ui, "workspace", &mut ws, &opts, ws_width) {
                        app.ui.workspace = ws;
                        crate::menus::apply_workspace(app);
                    }
                    if icons_shown >= 1
                        && icons::button(ui, "search", 28.0, app.ui.palette_open, &crate::shortcuts::tip_label(app, "Search commands", "edit.search")).clicked()
                    {
                        app.ui.palette_open = !app.ui.palette_open;
                    }
                    let (theme_icon, appearance_label) = match app.session.prefs().interface.appearance_mode {
                        photocraft_engine::prefs::AppearanceMode::Auto => ("monitor", tl!("Auto")),
                        photocraft_engine::prefs::AppearanceMode::Dark => ("moon", tl!("Dark")),
                        photocraft_engine::prefs::AppearanceMode::Light => ("sun", tl!("Light")),
                    };
                    let appearance_tip = format!("{}: {}", tl!("Appearance Mode"), appearance_label);
                    if icons_shown >= 2 {
                        let button = icons::button(ui, theme_icon, 28.0, false, &appearance_tip);
                        if button.clicked() {
                            crate::prefs_ui::cycle_appearance(app, ui.ctx());
                        }
                        button.context_menu(|ui| crate::prefs_ui::appearance_context_menu(app, ui));
                    }
                    // The community Discord, one click away while the bar has room for it
                    // (narrow windows drop it first; it is also Help › Discord).
                    if ui.available_width() >= DISCORD_ROOM {
                        let discord = egui::Button::image_and_text(
                            icons::image("message-square", 14.0, t.text_dim),
                            egui::RichText::new(tl!("Discord")).color(t.text_dim).size(12.0),
                        )
                        .frame(false);
                        if ui
                            .add(discord)
                            .on_hover_text({
                                let template = "Join the ArtCraft Discord ({url})";
                                crate::i18n::fmt(tl!(template), &[("url", crate::links::DISCORD)])
                            })
                            .clicked()
                        {
                            crate::links::open(app, ui.ctx(), crate::links::DISCORD);
                        }
                    }
                    if rtl { ui.min_rect().right() } else { ui.min_rect().left() }
                };
                free = if rtl { (controls_end, menus_end) } else { (menus_end, controls_end) };
            });
            ui.ctx().data_mut(|d| d.insert_temp(span_id, free));
            if system {
                return;
            }
            let font = theme::medium(13.0);
            let galley = ui.painter().layout_no_wrap(title.clone(), font.clone(), t.text_dim);
            if let Some(x) = title_x(full.center().x, free.0, free.1, galley.size().x) {
                ui.painter().galley(egui::pos2(x, full.center().y - galley.size().y / 2.0), galley, t.text_dim);
            } else if free.1 - free.0 > 80.0 {
                // Too narrow for the whole name: show the start of it, elided, in the free gap
                // next to the menus.
                let avail = free.1 - free.0 - 2.0 * TITLE_GAP;
                let mut job = egui::text::LayoutJob::simple_singleline(title, font, t.text_dim);
                job.wrap = egui::text::TextWrapping::truncate_at_width(avail);
                let g = ui.painter().layout_job(job);
                let x = if rtl { free.1 - TITLE_GAP - g.size().x } else { free.0 + TITLE_GAP };
                ui.painter().galley(egui::pos2(x, full.center().y - g.size().y / 2.0), g, t.text_dim);
            }
        });
    if custom {
        crate::titlebar::caption_buttons(app, ui, bar.response.rect);
    }
}

/// Whether the window shows the system's title bar instead of the app drawing its own: Windows
/// and Linux with Preferences › Interface › System Title Bar. macOS always has system decorations,
/// but keeps the in-app brand mark and centred title, so it never counts as system mode here; nor
/// does the web build (the browser tab has no document title), nor an embedder or the snapshot
/// example that never turned the preference on.
pub fn system_title_bar(app: &PhotocraftApp) -> bool {
    !app.custom_titlebar && app.session.prefs().interface.system_title_bar && !cfg!(any(target_os = "macos", target_arch = "wasm32"))
}

/// The OS window title: the active document's name suffixed with the app name (a `*` prefix
/// marks unsaved changes, like the `•` in the app-drawn title), or just the app name with no
/// document open.
pub fn window_title(app: &PhotocraftApp) -> String {
    app.session.active().map(|d| format!("{}{} \u{2014} PhotoCraft", if d.is_dirty() { "*" } else { "" }, d.doc.name)).unwrap_or_else(|| "PhotoCraft".into())
}

/// Keep the OS window title (and the taskbar / Alt-Tab entry) on the active file. Sends
/// `ViewportCommand::Title` only when the title changed since the last frame.
pub fn sync_window_title(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.services.host_save.is_some() {
        return;
    }
    let want = window_title(app);
    if app.last_window_title != want {
        app.last_window_title = want.clone();
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(want));
    }
}

/// Minimum space kept between the window title and the menus or controls beside it.
const TITLE_GAP: f32 = 16.0;
/// Free room the title bar needs after its other controls to show the Discord button.
const DISCORD_ROOM: f32 = 120.0;
/// The workspace switcher's narrowest width, and one 28 pt title-bar icon with its spacing.
const WORKSPACE_MIN: f32 = 90.0;
const ICON_SLOT: f32 = 34.0;

/// Left edge for a title `width` wide: centred on `center` when it fits in the free gap from
/// `left` to `right` (the menus' and the controls' facing edges, whichever side each is on),
/// else slid into that gap; None when the gap is too small for the whole title.
fn title_x(center: f32, left: f32, right: f32, width: f32) -> Option<f32> {
    let (lo, hi) = (left + TITLE_GAP, right - TITLE_GAP - width);
    if !(lo.is_finite() && hi.is_finite() && width.is_finite()) || hi < lo {
        return None;
    }
    if !center.is_finite() {
        return Some(lo);
    }
    Some((center - width / 2.0).clamp(lo, hi))
}

#[cfg(test)]
mod title_tests {
    use super::{system_title_bar, title_x, window_title};

    #[test]
    fn title_never_overlaps_menus_or_controls() {
        // Wide window: centred.
        assert_eq!(title_x(800.0, 420.0, 1300.0, 60.0), Some(770.0));
        // Centre would hit the menus: slid right into the gap.
        assert_eq!(title_x(400.0, 420.0, 1300.0, 60.0), Some(436.0));
        // Centre would hit the controls: slid left.
        assert_eq!(title_x(1280.0, 420.0, 1300.0, 60.0), Some(1224.0));
        // No room for the whole title: the caller elides or drops it.
        assert_eq!(title_x(400.0, 420.0, 480.0, 60.0), None);
        assert_eq!(title_x(f32::NAN, 420.0, 1300.0, 60.0), Some(436.0));
        assert_eq!(title_x(400.0, f32::INFINITY, 1300.0, 60.0), None);
    }

    fn app(custom_titlebar: bool) -> crate::PhotocraftApp {
        let mut app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.custom_titlebar = custom_titlebar;
        // The preference that makes the shell launch without its own title bar.
        app.session.edit_prefs(|p| p.interface.system_title_bar = !custom_titlebar);
        app
    }

    #[test]
    fn without_the_preference_the_mark_stays() {
        // The default (e.g. the web build or the snapshot example): no custom bar, no preference.
        let app = crate::PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert!(!system_title_bar(&app));
    }

    fn open_named(app: &mut crate::PhotocraftApp, name: &str) {
        app.session.add_document(
            photocraft_doc::Document::new(name, photocraft_doc::Size::new(4, 4), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8),
            None,
        );
    }

    #[test]
    fn window_title_with_no_document_is_the_app_name() {
        assert_eq!(window_title(&app(true)), "PhotoCraft");
    }

    #[test]
    fn window_title_follows_the_active_file() {
        let mut app = app(true);
        open_named(&mut app, "foo.psd");
        assert_eq!(window_title(&app), "foo.psd \u{2014} PhotoCraft");
        // Unsaved changes gain a `*` prefix, like the `•` in the app-drawn title.
        if let Some(st) = app.session.active_mut() {
            st.revision += 1;
        }
        assert_eq!(window_title(&app), "*foo.psd \u{2014} PhotoCraft");
    }

    #[test]
    fn system_mode_hides_the_mark_and_the_centred_title() {
        for custom in [true, false] {
            let mut app = app(custom);
            open_named(&mut app, "foo.psd");
            let ctx = egui::Context::default();
            crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
            let out = ctx.run_ui(egui::RawInput::default(), |ui| super::title_bar(&mut app, ui));
            let mut out = out;
            out.textures_delta.clear();
            let mut texts = Vec::new();
            for shape in &out.shapes {
                collect_text(&shape.shape, &mut texts);
            }
            let system = system_title_bar(&app);
            assert_eq!(system, !custom && !cfg!(target_os = "macos"));
            assert_eq!(crate::brand::mark_rect(&ctx).is_some(), !system, "mark in system mode (custom={custom})");
            assert_eq!(texts.iter().any(|t| t.contains("foo.psd")), !system, "centred title in system mode (custom={custom}): {texts:?}");
        }
    }

    fn collect_text(shape: &egui::Shape, out: &mut Vec<String>) {
        match shape {
            egui::Shape::Text(t) => out.push(t.galley.text().to_string()),
            egui::Shape::Vec(shapes) => shapes.iter().for_each(|s| collect_text(s, out)),
            _ => {}
        }
    }

    #[test]
    fn embedded_menu_keeps_commands_without_the_standalone_app_chrome() {
        let mut app = app(false);
        app.services.host_save = Some(Box::new(|| {}));
        open_named(&mut app, "embedded-page.pcraft");
        let ctx = egui::Context::default();
        crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ProLight);
        super::sync_window_title(&mut app, &ctx);
        assert!(app.last_window_title.is_empty());
        let mut out = ctx.run_ui(egui::RawInput::default(), |ui| super::title_bar(&mut app, ui));
        out.textures_delta.clear();
        let mut texts = Vec::new();
        for shape in &out.shapes {
            collect_text(&shape.shape, &mut texts);
        }
        assert!(!texts.is_empty(), "the command menus remain available");
        assert!(!texts.iter().any(|t| t.contains("embedded-page") || t.contains("Discord") || t.contains("Essentials")));
        assert!(crate::brand::mark_rect(&ctx).is_none());
    }

    #[test]
    fn sync_window_title_sends_only_on_change() {
        let mut app = app(true);
        let ctx = egui::Context::default();
        crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        super::sync_window_title(&mut app, &ctx);
        assert_eq!(app.last_window_title, "PhotoCraft");
        let mut out = ctx.run_ui(egui::RawInput::default(), |_| {});
        out.textures_delta.clear();
        let cmds = out.viewport_output.remove(&egui::ViewportId::ROOT).map(|v| v.commands).unwrap_or_default();
        assert!(cmds.iter().any(|c| matches!(c, egui::ViewportCommand::Title(t) if t == "PhotoCraft")), "{cmds:?}");
        // Same document state again: the cached title stops a repeat command.
        super::sync_window_title(&mut app, &ctx);
        let mut out = ctx.run_ui(egui::RawInput::default(), |_| {});
        out.textures_delta.clear();
        let cmds = out.viewport_output.remove(&egui::ViewportId::ROOT).map(|v| v.commands).unwrap_or_default();
        assert!(!cmds.iter().any(|c| matches!(c, egui::ViewportCommand::Title(_))), "repeat Title: {cmds:?}");
        open_named(&mut app, "foo.psd");
        super::sync_window_title(&mut app, &ctx);
        assert_eq!(app.last_window_title, "foo.psd \u{2014} PhotoCraft");
    }
}

// Custom controls also need to bring keyboard focus into a scroll viewport.
fn reveal_focused_control(ui: &egui::Ui) {
    // read_response reprocesses pointer events. Reading a popup's newly focused field
    // during its opening click can undo the focus that the popup just requested.
    if ui.input(|input| input.pointer.any_pressed() || input.pointer.any_released()) {
        return;
    }
    let focused = ui.memory(|memory| memory.focused());
    let key = ui.id().with("scroll-focused-control");
    let changed = ui.data_mut(|data| {
        let previous = data.get_temp::<Option<egui::Id>>(key).flatten();
        data.insert_temp(key, focused);
        previous != focused
    });
    if changed
        && let Some(response) = focused.and_then(|id| ui.ctx().read_response(id))
        && response.layer_id == ui.layer_id()
        && ui.min_rect().contains_rect(response.rect)
    {
        ui.scroll_to_rect(response.rect, Some(egui::Align::Center));
    }
}

// ----------------------------------------------------------------------------- options bar

/// Options bar height: Preferences ▸ Workspace ▸ Enable Narrow Options Bar trims it, as in
/// Photoshop, leaving the document more room.
fn options_bar_height(pro: bool, narrow: bool) -> f32 {
    match (pro, narrow) {
        (true, false) => 36.0,
        (true, true) => 30.0,
        (false, false) => 42.0,
        (false, true) => 34.0,
    }
}

pub fn options_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let narrow = app.session.prefs().workspace.narrow_options_bar();
    egui::Panel::top("options_bar")
        .exact_size(options_bar_height(t.pro, narrow))
        .frame(egui::Frame::NONE.fill(t.chrome).inner_margin(egui::Margin::symmetric(10, 0)))
        .show(ui, |ui| {
            ui.style_mut().spacing.scroll = egui::style::ScrollStyle { bar_width: 4.0, ..egui::style::ScrollStyle::floating() };
            egui::ScrollArea::horizontal().id_salt("tool-options").auto_shrink([false, false]).show(ui, |ui| {
            // Font ink can extend beyond its advance width at fractional positions.
            // Include its gutter in the container layout, including right-aligned actions.
            egui::Frame::NONE.inner_margin(egui::Margin { right: 2, ..Default::default() }).show(ui, |ui| {
            crate::direction::horizontal_centered(ui, |ui| {
                if t.pro && app.services.host_save.is_none() {
                    crate::chrome_ui::home_button(app, ui);
                    widgets::vline(ui, 22.0);
                }
                let tool_icon = icons::button(ui, icons::tool_icon(app.ui.tool), if t.pro { 26.0 } else { 28.0 }, !t.pro, tl!(app.ui.tool.label()));
                // Photoshop: right-click the tool icon for Reset Tool / Reset All Tools (#2825).
                let mut reset = None;
                tool_icon.context_menu(|ui| {
                    if ui.button(tl!("Reset Tool")).clicked() {
                        reset = Some("tool.reset");
                        ui.close();
                    }
                    if ui.button(tl!("Reset All Tools")).clicked() {
                        reset = Some("tool.resetAll");
                        ui.close();
                    }
                });
                if let Some(id) = reset
                    && let Err(e) = crate::menus::invoke(app, ui.ctx(), id, json!({}))
                {
                    app.ui.status = e;
                }
                if t.pro {
                    let (r, _) = ui.allocate_exact_size(vec2(10.0, 20.0), Sense::hover());
                    icons::paint(ui, r, "chevron-down", 10.0, t.text_faint);
                }
                widgets::vline(ui, 22.0);
                if app.ui.transform.is_some() {
                    crate::transform_tool::options_bar(app, ui);
                    return;
                }
                if crate::symmetry_ui::options_bar(app, ui) || crate::distort_ui::options_bar(app, ui) {
                    return;
                }
                let tool = app.ui.tool;
                // The options bar shows the active tool's brush: switch it in before drawing the
                // chip, so a tool change swaps the brush with it (Photoshop keeps one per tool).
                crate::paint_mouse::sync_tool_brush(app);
                // Brush edits here go through `tools.setBrush`, one journal entry per gesture (Rule 1).
                if (tool.is_brushlike() && !matches!(tool, Tool::Brush | Tool::Pencil | Tool::MixerBrush | Tool::Eraser)) || tool == Tool::QuickSelection {
                    brush_preset_chip(ui, &app.session.tools.brush, &mut app.ui);
                    crate::brush_picker::settings_toggle(app, ui);
                    widgets::vline(ui, 22.0);
                }
                if tool == Tool::CloneStamp {
                    let before = app.session.tools.brush.clone();
                    let mut brush = before.clone();
                    let width = if t.pro { 62.0 } else { 66.0 };
                    percent_field(ui, tl!("Opacity"), &mut brush.opacity, 1.0..=100.0, width);
                    percent_field(ui, tl!("Flow"), &mut brush.flow, 1.0..=100.0, width);
                    crate::brush_panel::commit_gesture(app, ui.ctx(), &before, &brush);
                    widgets::vline(ui, 22.0);
                }
                if crate::eraser_ui::options_bar(app, ui, tool)
                    || crate::retouch_ui::options_bar(app, ui, tool)
                    || crate::vector_ui::options_bar(app, ui, tool)
                    || crate::analysis_ui::options_bar(app, ui, tool)
                    || crate::slice_ui::options_bar(app, ui, tool)
                    || crate::frame_tool::options_bar(app, ui, tool)
                    || crate::artboard_tool::options_bar(app, ui, tool)
                {
                    return;
                }
                let brush_before = app.session.tools.brush.clone();
                let mut brush = brush_before.clone();
                let b = &mut brush;
                match app.ui.tool {
                    Tool::Brush | Tool::Eraser if t.pro => {
                        // A Block-mode Eraser has a fixed size, opacity and flow (Photoshop, #2770).
                        let block = crate::eraser_ui::block_mode(app, app.ui.tool);
                        ui.add_enabled_ui(!block, |ui| {
                            brush_preset_chip(ui, b, &mut app.ui);
                            crate::brush_picker::settings_toggle(app, ui);
                        });
                        widgets::vline(ui, 22.0);
                        // The Eraser has no blend mode: its Mode is Brush, Pencil or Block (#2662).
                        if app.ui.tool == Tool::Eraser {
                            crate::eraser_ui::mode_dropdown(ui, &mut app.ui.tool_options.eraser_mode);
                        } else {
                            opt_label(ui, tl!("Mode"));
                            let held = app.held_erase;
                            paint_mode_dropdown(ui, "brush-mode", b, &mut app.ui.tool_options.brush_clear, held);
                        }
                        ui.add_enabled_ui(!block, |ui| {
                            percent_field(ui, tl!("Opacity"), &mut b.opacity, 0.0..=100.0, 62.0);
                            if icons::button(ui, "circle-dot", 24.0, b.pressure_opacity, tl!("Always use pressure for opacity")).clicked() {
                                b.pressure_opacity = !b.pressure_opacity;
                            }
                        });
                        // A Pencil- or Block-mode Eraser is always full flow, without build-up (Photoshop).
                        ui.add_enabled_ui(!block && !crate::eraser_ui::pencil_mode(app, app.ui.tool), |ui| {
                            percent_field(ui, tl!("Flow"), &mut b.flow, 1.0..=100.0, 62.0);
                            let airbrush = icons::button(ui, "sparkles", 24.0, b.build_up, tl!("Enable airbrush-style build-up effects"));
                            if crate::brush_picker::named(airbrush, "Enable airbrush-style build-up effects").clicked() {
                                b.build_up = !b.build_up;
                            }
                        });
                        opt_label(ui, tl!("Smoothing"));
                        smoothing_field(ui, b, 58.0);
                        smoothing_options(ui, b);
                        widgets::vline(ui, 22.0);
                        if icons::button(ui, "circle-dot", 24.0, b.pressure_size, tl!("Always use pressure for size")).clicked() {
                            b.pressure_size = !b.pressure_size;
                        }
                    }
                    // Pencil: Photoshop's options (no hardness or flow: the pencil is always hard).
                    Tool::Pencil => {
                        brush_preset_chip(ui, b, &mut app.ui);
                        crate::brush_picker::settings_toggle(app, ui);
                        widgets::vline(ui, 22.0);
                        if !t.pro {
                            opt_label(ui, tl!("Size"));
                            widgets::value_field(ui, &mut b.size, 1.0..=photocraft_engine::paint::MAX_BRUSH_SIZE, "px", 76.0);
                            widgets::vline(ui, 22.0);
                        }
                        opt_label(ui, tl!("Mode"));
                        let held = app.held_erase;
                        paint_mode_dropdown(ui, "pencil-mode", b, &mut app.ui.tool_options.pencil_clear, held);
                        percent_field(ui, tl!("Opacity"), &mut b.opacity, 0.0..=100.0, if t.pro { 62.0 } else { 66.0 });
                        opt_label(ui, tl!("Smoothing"));
                        smoothing_field(ui, b, if t.pro { 58.0 } else { 66.0 });
                        widgets::vline(ui, 22.0);
                        widgets::checkbox(ui, &mut app.ui.tool_options.pencil_auto_erase, tl!("Auto Erase"));
                    }
                    Tool::Brush | Tool::Eraser => {
                        brush_preset_chip(ui, b, &mut app.ui);
                        crate::brush_picker::settings_toggle(app, ui);
                        widgets::vline(ui, 22.0);
                        // A Block-mode Eraser has a fixed size, opacity and flow (Photoshop, #2770).
                        let block = crate::eraser_ui::block_mode(app, app.ui.tool);
                        ui.add_enabled_ui(!block, |ui| {
                            opt_label(ui, tl!("Size"));
                            widgets::value_field(ui, &mut b.size, 1.0..=2500.0, "px", 76.0);
                        });
                        widgets::vline(ui, 22.0);
                        if app.ui.tool == Tool::Eraser {
                            crate::eraser_ui::mode_dropdown(ui, &mut app.ui.tool_options.eraser_mode);
                        } else {
                            opt_label(ui, tl!("Mode"));
                            let held = app.held_erase;
                            paint_mode_dropdown(ui, "brush-mode", b, &mut app.ui.tool_options.brush_clear, held);
                        }
                        // A Pencil-mode Eraser is always hard and at full flow (Photoshop).
                        let fixed = block || crate::eraser_ui::pencil_mode(app, app.ui.tool);
                        ui.add_enabled_ui(!fixed, |ui| percent_field(ui, tl!("Hardness"), &mut b.hardness, 0.0..=100.0, 66.0));
                        ui.add_enabled_ui(!block, |ui| percent_field(ui, tl!("Opacity"), &mut b.opacity, 0.0..=100.0, 66.0));
                        ui.add_enabled_ui(!fixed, |ui| percent_field(ui, tl!("Flow"), &mut b.flow, 1.0..=100.0, 66.0));
                        opt_label(ui, tl!("Smoothing"));
                        smoothing_field(ui, b, 66.0);
                        widgets::vline(ui, 22.0);
                        widgets::toggle(ui, &mut b.pressure_size, tl!("Pressure for Size"));
                        widgets::toggle(ui, &mut b.pressure_opacity, tl!("Pressure for Opacity"));
                    }
                    Tool::MixerBrush => {
                        brush_preset_chip(ui, b, &mut app.ui);
                        crate::brush_picker::settings_toggle(app, ui);
                        widgets::vline(ui, 22.0);
                        for (label, value) in [
                            (tl!("Wet"), &mut b.mixer.wet),
                            (tl!("Load"), &mut b.mixer.load),
                            (tl!("Mix"), &mut b.mixer.mix),
                            (tl!("Flow"), &mut b.mixer.flow),
                        ] {
                            percent_field(ui, label, value, 0.0..=100.0, 62.0);
                        }
                        widgets::vline(ui, 22.0);
                        widgets::checkbox(ui, &mut b.mixer.sample_all_layers, tl!("Sample All Layers"));
                    }
                    Tool::RectMarquee | Tool::EllipseMarquee | Tool::SingleRowMarquee | Tool::SingleColumnMarquee if t.pro => {
                        ui.spacing_mut().item_spacing.x = 2.0;
                        for (i, (icon, tip)) in [
                            ("square", tl!("New selection").to_string()),
                            ("plus", crate::i18n::fmt(tl!("Add to selection  ({key})"), &[("key", &crate::shortcuts::pretty("Shift"))])),
                            ("minus", crate::i18n::fmt(tl!("Subtract from selection  ({key})"), &[("key", &crate::shortcuts::pretty("Alt"))])),
                            (
                                "squares-subtract",
                                crate::i18n::fmt(tl!("Intersect with selection  ({key})"), &[("key", &crate::shortcuts::pretty("Shift+Alt"))]),
                            ),
                        ]
                        .iter()
                        .enumerate()
                        {
                            if icons::button(ui, icon, 24.0, app.ui.selection_mode == i as u8, tip).clicked() {
                                app.ui.selection_mode = i as u8;
                            }
                        }
                        ui.spacing_mut().item_spacing.x = 8.0;
                        widgets::vline(ui, 22.0);
                        let o = &mut app.ui.tool_options;
                        opt_label(ui, tl!("Feather"));
                        widgets::value_field(ui, &mut o.feather, 0.0..=1000.0, "px", 62.0);
                        // Photoshop greys Anti-alias for the Rectangular Marquee (its edges are always hard).
                        ui.add_enabled_ui(app.ui.tool == Tool::EllipseMarquee, |ui| widgets::checkbox(ui, &mut o.anti_alias, tl!("Anti-alias")));
                        widgets::vline(ui, 22.0);
                        // A single row or column has no style or size: Photoshop greys them.
                        let sized = matches!(app.ui.tool, Tool::RectMarquee | Tool::EllipseMarquee);
                        ui.add_enabled_ui(sized, |ui| opt_label(ui, tl!("Style")));
                        let styles = [
                            ("normal".to_string(), tl!("Normal")),
                            ("fixedRatio".to_string(), tl!("Fixed Ratio")),
                            ("fixedSize".to_string(), tl!("Fixed Size")),
                        ];
                        if ui.add_enabled_ui(sized, |ui| widgets::dropdown(ui, "marquee-style", &mut o.marquee_style, &styles, 96.0)).inner {
                            (o.marquee_width, o.marquee_height) = if o.marquee_style == "fixedSize" { (64.0, 64.0) } else { (1.0, 1.0) };
                        }
                        let fixed = sized && o.marquee_style != "normal";
                        let (unit, range) = if o.marquee_style == "fixedSize" { ("px", 1.0..=300_000.0) } else { ("", 0.001..=999.0) };
                        ui.add_enabled_ui(fixed, |ui| {
                            opt_label(ui, tl!("Width"));
                            widgets::value_field(ui, &mut o.marquee_width, range.clone(), unit, 62.0);
                            if icons::button(ui, "arrow-left-right", 22.0, false, tl!("Swaps height and width")).clicked() {
                                std::mem::swap(&mut o.marquee_width, &mut o.marquee_height);
                            }
                            opt_label(ui, tl!("Height"));
                            widgets::value_field(ui, &mut o.marquee_height, range, unit, 62.0);
                        });
                        widgets::vline(ui, 22.0);
                        if widgets::secondary_button(ui, tl!("Select and Mask…"), 0.0).clicked() {
                            let _ = crate::menus::invoke(app, ui.ctx(), "select.selectAndMask", json!({}));
                        }
                    }
                    Tool::Lasso | Tool::PolygonLasso | Tool::MagneticLasso if t.pro => {
                        selection_mode_buttons(app, ui);
                        widgets::vline(ui, 22.0);
                        opt_label(ui, tl!("Feather"));
                        widgets::value_field(ui, &mut app.ui.tool_options.feather, 0.0..=1000.0, "px", 62.0);
                        widgets::checkbox(ui, &mut app.ui.tool_options.anti_alias, tl!("Anti-alias"));
                        widgets::vline(ui, 22.0);
                        if app.ui.tool == Tool::MagneticLasso {
                            let o = &mut app.ui.tool_options;
                            opt_label(ui, tl!("Width"));
                            widgets::value_field(ui, &mut o.magnetic_width, 1.0..=256.0, "px", 58.0)
                                .on_hover_text(tl!("Follows only edges this close to the pointer ([ and ] change it)"));
                            opt_label(ui, tl!("Contrast"));
                            widgets::value_field(ui, &mut o.magnetic_contrast, 1.0..=100.0, "%", 58.0)
                                .on_hover_text(tl!("Higher values follow only edges that contrast sharply with their surroundings"));
                            opt_label(ui, tl!("Frequency"));
                            widgets::value_field(ui, &mut o.magnetic_frequency, 0.0..=100.0, "", 50.0)
                                .on_hover_text(tl!("How often fastening points are placed by themselves"));
                            if icons::button(ui, "circle-dot", 24.0, o.magnetic_pressure, tl!("Use tablet pressure to change pen width")).clicked() {
                                o.magnetic_pressure = !o.magnetic_pressure;
                            }
                            widgets::vline(ui, 22.0);
                        }
                        if widgets::secondary_button(ui, tl!("Select and Mask…"), 0.0).clicked() {
                            let _ = crate::menus::invoke(app, ui.ctx(), "select.selectAndMask", json!({}));
                        }
                        if crate::lasso_ui::active(app) || (app.ui.tool == Tool::PolygonLasso && !app.ui.polygon.is_empty()) {
                            hint(
                                ui,
                                &crate::i18n::fmt(
                                    tl!("Click the first point or press {key} to close · Esc cancels"),
                                    &[("key", &crate::shortcuts::pretty("Enter"))],
                                ),
                            );
                        } else if app.ui.tool == Tool::Lasso {
                            hint(ui, tl!("Hold Alt while drawing for straight segments"));
                        }
                        if app.ui.tool == Tool::MagneticLasso && app.ui.magnetic.active() {
                            hint(
                                ui,
                                &crate::i18n::fmt(
                                    tl!("Click the first point or press {key} to close · {del} removes a point · Esc cancels"),
                                    &[("key", &crate::shortcuts::pretty("Enter")), ("del", &crate::shortcuts::pretty("Backspace"))],
                                ),
                            );
                        }
                    }
                    Tool::MagicWand => {
                        selection_mode_buttons(app, ui);
                        widgets::vline(ui, 22.0);
                        opt_label(ui, tl!("Tolerance"));
                        widgets::value_field(ui, &mut app.ui.tool_options.tolerance, 0.0..=255.0, "", 58.0);
                        widgets::checkbox(ui, &mut app.ui.tool_options.anti_alias, tl!("Anti-alias"));
                        widgets::checkbox(ui, &mut app.ui.tool_options.contiguous, tl!("Contiguous"));
                        widgets::checkbox(ui, &mut app.ui.tool_options.sample_all_layers, tl!("Sample All Layers"));
                        widgets::vline(ui, 22.0);
                        if widgets::secondary_button(ui, tl!("Select Subject"), 0.0).clicked() {
                            let r = app.run("select.subject", json!({}));
                            crate::retouch_ui::report_smart_result(app, r, tl!("No subject found — try Object Selection or the Lasso"));
                        }
                        if widgets::secondary_button(ui, tl!("Select and Mask…"), 0.0).clicked() {
                            let _ = crate::menus::invoke(app, ui.ctx(), "select.selectAndMask", json!({}));
                        }
                    }
                    Tool::PaintBucket if t.pro => {
                        opt_label(ui, tl!("Fill"));
                        let mut src = u8::from(app.ui.tool_options.bucket_fill_pattern);
                        if widgets::dropdown(ui, "bucket-src", &mut src, &[(0u8, tl!("Foreground")), (1, tl!("Pattern"))], 100.0) {
                            app.ui.tool_options.bucket_fill_pattern = src == 1;
                        }
                        opt_label(ui, tl!("Opacity"));
                        widgets::value_field(ui, &mut app.ui.tool_options.fill_opacity, 1.0..=100.0, "%", 62.0);
                        opt_label(ui, tl!("Tolerance"));
                        widgets::value_field(ui, &mut app.ui.tool_options.tolerance, 0.0..=255.0, "", 58.0);
                        widgets::checkbox(ui, &mut app.ui.tool_options.anti_alias, tl!("Anti-alias"));
                        widgets::checkbox(ui, &mut app.ui.tool_options.contiguous, tl!("Contiguous"));
                        widgets::checkbox(ui, &mut app.ui.tool_options.sample_all_layers, tl!("All Layers"));
                    }
                    Tool::Gradient => {
                        // Live "Gradient" (a Gradient Fill layer) or "Classic gradient" (pixels).
                        let mut classic = app.ui.tool_options.gradient_classic;
                        if widgets::dropdown(ui, "gradient-mode", &mut classic, &[(false, "Gradient"), (true, "Classic gradient")], 118.0) {
                            app.ui.tool_options.gradient_classic = classic;
                        }
                        // Live "Gradient" (a Gradient Fill layer) or "Classic gradient" (pixels):
                        // both paint with the same engine-owned preset (#1651), and either swatch
                        // click opens the Gradient Editor window, as in Photoshop.
                        if crate::gradient_ui::preset_swatch(app, ui) {
                            app.ui.panels.gradient_editor = true;
                        }
                        widgets::vline(ui, 22.0);
                        ui.spacing_mut().item_spacing.x = 2.0;
                        let before = app.ui.tool_options.clone();
                        for (style, icon, tip) in [
                            ("linear", "gradient", tl!("Linear Gradient")),
                            ("radial", "circle", tl!("Radial Gradient")),
                            ("angle", "rotate-cw", tl!("Angle Gradient")),
                            ("reflected", "arrow-left-right", tl!("Reflected Gradient")),
                            ("diamond", "diamond", tl!("Diamond Gradient")),
                        ] {
                            if icons::button(ui, icon, 24.0, app.ui.tool_options.gradient_style == style, tip).clicked() {
                                app.ui.tool_options.gradient_style = style.into();
                            }
                        }
                        ui.spacing_mut().item_spacing.x = 8.0;
                        widgets::vline(ui, 22.0);
                        opt_label(ui, tl!("Mode"));
                        let opts: Vec<(BlendMode, &str)> = BlendMode::LAYER_MODES.iter().map(|m| (*m, m.label())).collect();
                        // The wheel over it steps the mode, as over the Layers panel's (#2061).
                        widgets::dropdown_wheel_hovered(ui, "gradient-blend-mode", &mut app.ui.tool_options.gradient_blend_mode, &opts, 96.0);
                        opt_label(ui, tl!("Opacity"));
                        let r = widgets::value_field(ui, &mut app.ui.tool_options.fill_opacity, 1.0..=100.0, "%", 62.0);
                        // Live while scrubbed, one undo step per scrub (#5012).
                        let drag_key = r.id.with("gradient-opacity-drag");
                        if r.drag_started() {
                            let n = ui.ctx().cumulative_pass_nr();
                            ui.data_mut(|d| d.insert_temp(drag_key, n));
                        }
                        if r.changed() {
                            let drag = if r.dragged() { ui.data(|d| d.get_temp::<u64>(drag_key)) } else { None };
                            crate::gradient_ui::opacity_changed(app, drag);
                        }
                        widgets::checkbox(ui, &mut app.ui.tool_options.gradient_reverse, tl!("Reverse"));
                        widgets::checkbox(ui, &mut app.ui.tool_options.gradient_dither, tl!("Dither"));
                        // Live mode: the options also change a selected gradient fill layer.
                        crate::gradient_ui::options_changed(app, &before);
                    }
                    Tool::PerspectiveCrop if t.pro => crate::perspective_crop_ui::options_bar(app, ui),
                    Tool::Crop if t.pro => {
                        let before_ratio = crate::crop_ui::preset_ratio(app);
                        let doc_size = app.session.active().map_or((0.0, 0.0), |s| (f64::from(s.doc.size.width), f64::from(s.doc.size.height)));
                        // Ratios, sizes, saved presets, New / Delete Crop Preset… (#2443, #1919).
                        crate::crop_presets::dropdown(app, ui);
                        let o = &mut app.ui.tool_options;
                        if o.crop_ratio == crate::crop_size::WHR {
                            crate::crop_size::fields(o, ui);
                        } else {
                            // Keep partial edits in ToolOptions so the first side survives a frame;
                            // Original Ratio shows the document's numbers, as Photoshop does.
                            let (mut rw, mut rh) = crate::crop_size::custom_ratio_fields(o, doc_size);
                            let cw = widgets::value_field(ui, &mut rw, 0.0..=99_999.0, "", 54.0).changed();
                            let swap = icons::button(ui, "arrow-left-right", 22.0, false, tl!("Swaps height and width")).clicked();
                            let ch = widgets::value_field(ui, &mut rh, 0.0..=99_999.0, "", 54.0).changed();
                            if swap {
                                std::mem::swap(&mut rw, &mut rh);
                            }
                            if cw || ch || swap {
                                crate::crop_size::set_custom_ratio_fields(o, rw, rh);
                            }
                        }
                        widgets::vline(ui, 22.0);
                        if widgets::secondary_button(ui, tl!("Clear"), 0.0).clicked() {
                            if o.crop_ratio == crate::crop_size::WHR {
                                crate::crop_size::clear(o);
                            } else {
                                o.crop_ratio.clear();
                            }
                        }
                        crate::crop_straighten::options_button(&mut app.crop.straighten, ui);
                        crate::crop_overlay::options_button(o, ui);
                        let pick_shield_color = crate::crop_shield::options_button(&mut o.crop_shield, ui);
                        widgets::checkbox(ui, &mut o.crop_delete, tl!("Delete Cropped Pixels"));
                        if crate::crop_ui::preset_ratio(app) != before_ratio {
                            crate::crop_ui::apply_ratio(app);
                        }
                        if pick_shield_color {
                            crate::crop_shield::pick_custom_color(app);
                        }
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if icons::button(
                                ui,
                                "check",
                                26.0,
                                false,
                                &crate::i18n::fmt(tl!("Commit current crop operation  ({key})"), &[("key", &crate::shortcuts::pretty("Enter"))]),
                            )
                            .clicked()
                            {
                                crate::canvas::commit_crop(app);
                            }
                            if icons::button(ui, "ban", 26.0, false, tl!("Cancel current crop operation  (Esc)")).clicked() {
                                crate::crop_ui::cancel(app);
                            }
                        });
                    }
                    // Every theme gets the font, style and size controls (#1387), as with Auto-Select (#1275).
                    Tool::Type | Tool::VerticalType | Tool::HorizontalTypeMask | Tool::VerticalTypeMask => crate::type_tool::options_bar(app, ui),
                    Tool::Move if t.pro => {
                        let o = &mut app.ui.tool_options;
                        widgets::checkbox(ui, &mut o.move_auto_select, tl!("Auto-Select:"));
                        widgets::dropdown(
                            ui,
                            "move-target",
                            &mut o.move_target,
                            &[("layer".to_string(), tl!("Layer")), ("group".to_string(), tl!("Group"))],
                            76.0,
                        );
                        widgets::vline(ui, 22.0);
                        widgets::checkbox(ui, &mut o.move_show_transform, tl!("Show Transform Controls"));
                        widgets::vline(ui, 22.0);
                        ui.spacing_mut().item_spacing.x = 0.0;
                        for (icon, tip, cmd) in [
                            ("panels-top-left", tl!("Align left edges"), "layer.align.leftEdges"),
                            ("app-window", tl!("Align horizontal centers"), "layer.align.horizontalCenters"),
                            ("panel-right", tl!("Align right edges"), "layer.align.rightEdges"),
                        ] {
                            let on = app.session.is_enabled(cmd);
                            if ui.add_enabled_ui(on, |ui| icons::button(ui, icon, 24.0, false, tip)).inner.clicked() {
                                let _ = app.run(cmd, json!({}));
                            }
                        }
                        ui.add_space(6.0);
                        let more = icons::button(ui, "ellipsis", 24.0, false, tl!("Align and distribute"));
                        egui::Popup::menu(&more).show(|ui| {
                            ui.set_min_width(200.0);
                            let mut section = |ui: &mut egui::Ui, title: &str, prefix: &str| {
                                ui.label(egui::RichText::new(tl!(&title)).small().color(t.text_dim));
                                for c in photocraft_engine::command_specs().iter().filter(|c| c.id.starts_with(prefix)) {
                                    if ui.add_enabled(app.session.is_enabled(c.id), egui::Button::new(tl!(c.label))).clicked() {
                                        let _ = app.run(c.id, json!({}));
                                        ui.close();
                                    }
                                }
                            };
                            section(ui, tl!("Align"), "layer.align.");
                            ui.separator();
                            section(ui, tl!("Distribute"), "layer.distribute.");
                        });
                    }
                    Tool::RectMarquee | Tool::EllipseMarquee => hint(
                        ui,
                        &format!(
                            "Drag to select  ·  {} add  ·  {} subtract  ·  {} intersect  ·  click to deselect",
                            crate::shortcuts::pretty("Shift"),
                            crate::shortcuts::pretty("Alt"),
                            crate::shortcuts::pretty("Shift+Alt")
                        ),
                    ),
                    Tool::SingleRowMarquee | Tool::SingleColumnMarquee => hint(
                        ui,
                        &format!(
                            "Click to select a one-pixel line  ·  {} add  ·  {} subtract  ·  {} intersect",
                            crate::shortcuts::pretty("Shift"),
                            crate::shortcuts::pretty("Alt"),
                            crate::shortcuts::pretty("Shift+Alt")
                        ),
                    ),
                    Tool::Move => {
                        // The Studio themes keep Photoshop's Auto-Select toggle too (#1275).
                        widgets::checkbox(ui, &mut app.ui.tool_options.move_auto_select, tl!("Auto-Select:"));
                        hint(ui, tl!("Drag to move the active layer"));
                    }
                    Tool::Eyedropper => crate::eyedropper_ui::options(app, ui),
                    Tool::Zoom => {
                        widgets::checkbox(ui, &mut app.ui.tool_options.zoom_scrubby, tl!("Scrubby Zoom"));
                        hint(
                            ui,
                            &crate::i18n::fmt(
                                tl!("Click to zoom in  ·  {key}-click to zoom out  ·  drag right/left to zoom in/out"),
                                &[("key", &crate::shortcuts::pretty("Alt"))],
                            ),
                        );
                        if widgets::secondary_button(ui, tl!("Fit Screen"), 0.0).clicked()
                            && let Some(i) = app.session.active_index()
                        {
                            app.ui.views[i].fit_pending = true;
                        }
                        if widgets::secondary_button(ui, "100%", 0.0).clicked()
                            && let Some(i) = app.session.active_index()
                        {
                            app.ui.views[i].zoom = 1.0;
                        }
                    }
                    Tool::RotateView => {
                        hint(ui, tl!("Drag around the centre to rotate the view  ·  Shift constrains to 15°"));
                        if let Some(i) = app.session.active_index() {
                            ui.label(tl!("Angle"));
                            let mut angle = app.ui.views.get(i).map(|v| v.rotation).unwrap_or(0.0);
                            if widgets::value_field(ui, &mut angle, -180.0..=180.0, "°", 56.0).changed()
                                && let Some(v) = app.ui.views.get_mut(i)
                            {
                                v.rotation = crate::rotate_view::wrap_deg(angle);
                            }
                            if widgets::secondary_button(ui, tl!("Reset View"), 0.0).clicked()
                                && let Some(v) = app.ui.views.get_mut(i)
                            {
                                v.rotation = 0.0;
                            }
                        }
                    }
                    Tool::Hand => {
                        hint(ui, tl!("Drag to pan  ·  hold Space with any tool"));
                        if widgets::secondary_button(ui, "100%", 0.0).clicked()
                            && let Some(i) = app.session.active_index()
                        {
                            app.ui.views[i].zoom = 1.0;
                            app.ui.views[i].fit_pending = false;
                            app.ui.views[i].fill_pending = false;
                        }
                        if widgets::secondary_button(ui, tl!("Fit Screen"), 0.0).clicked()
                            && let Some(i) = app.session.active_index()
                        {
                            app.ui.views[i].fit_pending = true;
                            app.ui.views[i].fill_pending = false;
                        }
                        if widgets::secondary_button(ui, tl!("Fill Screen"), 0.0).clicked()
                            && let Some(i) = app.session.active_index()
                        {
                            app.ui.views[i].fill_pending = true;
                            app.ui.views[i].fit_pending = false;
                        }
                    }
                    Tool::Lasso | Tool::PolygonLasso => hint(
                        ui,
                        &crate::i18n::fmt(
                            tl!("Drag or click polygon points · hold Alt during lasso for straight segments · {add} add · {sub} before drawing subtracts"),
                            &[("add", &crate::shortcuts::pretty("Shift")), ("sub", &crate::shortcuts::pretty("Alt"))],
                        ),
                    ),
                    Tool::MagneticLasso => hint(
                        ui,
                        &crate::i18n::fmt(
                            tl!("Click, then move along an edge · click to fasten a point · {alt}-click for a straight segment · double-click to close"),
                            &[("alt", &crate::shortcuts::pretty("Alt"))],
                        ),
                    ),
                    Tool::Crop => hint(
                        ui,
                        &crate::i18n::fmt(
                            tl!("Drag a crop box · drag inside to move · edges resize ({ratio} ratio, {centre} centre) · Space moves while drawing · arrows nudge · X swaps orientation · {commit} commits · Esc cancels"),
                            &[
                                ("ratio", &crate::shortcuts::pretty("Shift")),
                                ("centre", &crate::shortcuts::pretty("Alt")),
                                ("commit", &crate::shortcuts::pretty("Enter")),
                            ],
                        ),
                    ),
                    Tool::PerspectiveCrop => hint(
                        ui,
                        &crate::i18n::fmt(
                            tl!("Drag a box or click four corners · drag the corners onto the edges of a skewed shape · {commit} commits · Esc cancels"),
                            &[("commit", &crate::shortcuts::pretty("Enter"))],
                        ),
                    ),
                    Tool::PaintBucket => hint(ui, tl!("Click to fill similar colours")),
                    // Retouching and smart-selection tools draw their bar in `retouch_ui::options_bar`.
                    _ => {}
                }
                // Painting symmetry is independent of visual theme. Keep the options-bar
                // action available for Brush, Pencil and Eraser in every palette (#2659).
                if matches!(tool, Tool::Brush | Tool::Pencil | Tool::Eraser) {
                    crate::symmetry_ui::menu(app, ui);
                }
                crate::brush_panel::commit_gesture(app, ui.ctx(), &brush_before, &brush);
            });
            reveal_focused_control(ui);
            });
            });
        });
}

/// Width of the Layers panel's Opacity and Fill fields, with room for their pop-up slider's ▾.
const LAYER_PCT_W: f32 = 66.0 + widgets::POPUP_ARROW_W;

/// The `layer.setProps` edit of a Layers panel Opacity or Fill (`key`) field at `pct` percent. A
/// drag of the pop-up slider (`drag`) coalesces into one history step, as in Photoshop.
fn pct_action(l: &Layer, key: &str, pct: f32, drag: Option<u64>) -> (String, Value) {
    let mut p = json!({"layer": l.id.0, key: pct / 100.0});
    if let Some(n) = drag {
        p["coalesce"] = json!(format!("layer-{key}-slider:{n}"));
    }
    ("layer.setProps".into(), p)
}

fn label(ui: &mut egui::Ui, s: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(s).color(t.text_dim));
}

/// Photoshop's scrubby Opacity/Fill labels: drag horizontally to change the percentage without
/// hitting the small number field. The unique drag ID coalesces every frame into one undo step.
fn scrub_pct_label(ui: &mut egui::Ui, text: &str, percent: &mut f32, max_w: f32) -> widgets::PopupFieldResponse {
    // Elide to the room left in the row (full text on hover): a longer translation shrinks here
    // instead of running over the fixed controls beside it (#3580).
    let resp = widgets::elided_label_sensed(ui, text, max_w, Sense::click_and_drag()).on_hover_cursor(egui::CursorIcon::ResizeHorizontal);
    let key = resp.id.with("percent-label-drag");
    if resp.drag_started() {
        let stamp = ui.ctx().cumulative_pass_nr();
        ui.data_mut(|d| d.insert_temp(key, (*percent, stamp)));
    }
    let mut out = widgets::PopupFieldResponse::default();
    if resp.dragged()
        && let Some((start, stamp)) = ui.data(|d| d.get_temp::<(f32, u64)>(key))
        && let (Some(origin), Some(pos)) = (ui.input(|i| i.pointer.press_origin()), resp.interact_pointer_pos())
    {
        // Match DragValue's 0.5 percentage-point per screen point; do not jump on press.
        let next = (start + (pos.x - origin.x) * 0.5).clamp(0.0, 100.0);
        if next.is_finite() && next != *percent {
            *percent = next;
            out.changed = true;
            out.drag = Some(stamp);
        }
    }
    if resp.drag_stopped() {
        ui.data_mut(|d| d.remove::<(f32, u64)>(key));
    }
    out
}

/// Width of `s` as [`label`] draws it (body text).
fn body_text_width(ui: &egui::Ui, s: &str) -> f32 {
    let font = egui::TextStyle::Body.resolve(ui.style());
    ui.painter().layout_no_wrap(s.to_string(), font, Color32::WHITE).size().x
}

/// Options-bar label: Photoshop writes "Size:" with a colon.
/// An options-bar label; one right before a value field scrubs it when dragged (Photoshop).
fn opt_label(ui: &mut egui::Ui, s: &str) {
    let t = Tokens::get(ui.ctx());
    let text = if t.pro && !s.ends_with(':') { format!("{s}:") } else { s.to_string() };
    let label = ui.label(RichText::new(tl!(&text)).color(t.text_dim));
    widgets::scrub_label(ui, &label);
}

fn percent_field(ui: &mut egui::Ui, label: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, width: f32) {
    opt_label(ui, label);
    let mut percent = *value * 100.0;
    if widgets::value_field(ui, &mut percent, range, "%", width).changed() {
        *value = percent / 100.0;
    }
}

fn hint(ui: &mut egui::Ui, s: &str) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(s).color(t.text_faint));
}

// ----------------------------------------------------------------------------- status bar

/// The status bar's height and the height of the controls in it (zoom field, Fit), which leaves
/// them an even margin above and below inside the bar (#2823).
pub fn status_bar_heights(t: &Tokens) -> (f32, f32) {
    if t.pro { (24.0, 18.0) } else { (30.0, 22.0) }
}

pub fn status_bar(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    egui::Panel::bottom("status_bar")
        .exact_size(status_bar_heights(&t).0)
        .frame(egui::Frame::NONE.fill(t.chrome).inner_margin(egui::Margin::symmetric(10, 0)))
        .show(ui, |ui| {
            let control_h = status_bar_heights(&t).1;
            // Read from the right in a right-to-left UI, with the trailing group at the left.
            crate::direction::horizontal_centered(ui, |ui| {
                if t.pro {
                    crate::chrome_ui::status_bar_pro(app, ui);
                    // Background job progress with Cancel, at the far end (#210).
                    ui.with_layout(crate::direction::trailing(), |ui| {
                        crate::jobs_ui::status_progress(app, ui);
                        crate::background_redraw::status(app, ui);
                    });
                    return;
                }
                if let (Some(st), Some(i)) = (app.session.active(), app.session.active_index()) {
                    let (w, h, mode, bits, layers) =
                        (st.doc.size.width, st.doc.size.height, crate::canvas::mode_label(&st.doc), st.doc.depth.bits(), st.doc.layer_count());
                    let mut pct = app.ui.views[i].zoom * 100.0;
                    if widgets::compact_wrapping_value_field(ui, &mut pct, crate::zoom_levels::percent_range(&app.ui.views[i]), "%", vec2(78.0, control_h))
                        .changed()
                    {
                        app.ui.views[i].zoom = crate::zoom_levels::clamp(pct / 100.0, app.ui.views[i].doc_size);
                        app.ui.views[i].fit_pending = false;
                    }
                    widgets::vline(ui, 16.0);
                    label(ui, &crate::i18n::fmt(tl!("{mode} Color · {bits} bit"), &[("mode", mode), ("bits", &bits.to_string())]));
                    widgets::vline(ui, 16.0);
                    label(ui, &format!("{w} × {h} px"));
                    widgets::vline(ui, 16.0);
                    label(ui, &crate::i18n::trn(crate::i18n::current(), layers as u64, "{n} layer", "{n} layers"));
                } else {
                    label(ui, tl!("No document"));
                }
                if !app.ui.status.is_empty() {
                    widgets::vline(ui, 16.0);
                    let is_err = app.ui.status_error || app.ui.status.starts_with("Couldn");
                    ui.label(RichText::new(&app.ui.status).color(if is_err { t.warning } else { t.text_faint }));
                }
                ui.with_layout(crate::direction::trailing(), |ui| {
                    if app.session.active().is_some()
                        && !t.pro
                        && widgets::compact_button(ui, tl!("Fit"), control_h).clicked()
                        && let Some(i) = app.session.active_index()
                    {
                        app.ui.views[i].fit_pending = true;
                    }
                    if app.session.has_jobs() {
                        ui.add_space(6.0);
                        crate::jobs_ui::status_progress(app, ui);
                    }
                    crate::background_redraw::status(app, ui);
                });
            });
        });
}

// ----------------------------------------------------------------------------- dock

pub fn right_dock(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    // Even legacy destinations must discard gestures captured by an earlier workspace.
    let customized = crate::panel_docking::show(app, ui);
    let t = Tokens::get(ui.ctx());
    let p = app.ui.panels.clone();
    if t.pro && !customized {
        dock_panels(app, ui, &p, &t);
    }
    // Narrow icon rail, unless an embedding app hides it: shows, expands or collapses panel groups.
    if p.rail {
        let (rw, rb) = if t.pro { (36.0, 28.0) } else { (44.0, 32.0) };
        crate::direction::end_panel("rail")
            .resizable(false)
            .exact_size(rw)
            .frame(egui::Frame::NONE.fill(t.chrome).inner_margin(egui::Margin::symmetric(4, 8)))
            .show(ui, |ui| {
                // The separator is on the canvas side.
                let r = ui.max_rect();
                let x = if crate::direction::is_rtl() { r.right() + 6.0 } else { r.left() - 6.0 };
                ui.painter().line_segment([pos2(x, r.top() - 8.0), pos2(x, r.bottom() + 8.0)], Stroke::new(1.0, t.separator));
                ui.spacing_mut().item_spacing.y = 4.0;
                use crate::dock::Group;
                let entries: [(&str, &str, Group); 5] = [
                    ("sliders-horizontal", tl!("Properties"), Group::Properties),
                    ("navigation", tl!("Navigator"), Group::Navigator),
                    ("palette", tl!("Color & Swatches"), Group::Color),
                    ("layers", tl!("Layers"), Group::Layers),
                    ("clock", tl!("History"), Group::History),
                ];
                for (icon, name, g) in entries {
                    // Studio floats Properties outside the dock.
                    let docked = t.pro || g != Group::Properties;
                    let on = if customized { crate::panel_docking::rail_on(app, g) } else { g.shown(&p) && !(docked && app.ui.dock.is_collapsed(g)) };
                    if icons::rail_button(ui, icon, rb, on, name).clicked() {
                        if customized {
                            crate::panel_docking::rail_click(app, g);
                        } else {
                            crate::dock::rail_click(app, g, docked);
                        }
                    }
                }
            });
    }
    if !t.pro && !customized {
        dock_panels(app, ui, &p, &t);
    }
    crate::dock::persist(app, ui.ctx());
}

/// The dock's width range (points). It is the right dock, on the left in a right-to-left UI.
pub(crate) const DOCK_WIDTH: std::ops::RangeInclusive<f32> = 250.0..=520.0;

pub(crate) fn dock_width_id() -> egui::Id {
    egui::Id::new("dock-width-request")
}

/// Set the right dock's width on the next frame (`ui.set {dockWidth}`); clamped to its range.
pub fn request_dock_width(ctx: &egui::Context, w: f32) {
    let w = if w.is_finite() { w.clamp(*DOCK_WIDTH.start(), *DOCK_WIDTH.end()) } else { *DOCK_WIDTH.start() };
    ctx.data_mut(|d| d.insert_temp(dock_width_id(), w));
}

fn dock_panels(app: &mut PhotocraftApp, ui: &mut egui::Ui, p: &crate::state::Panels, t: &Tokens) {
    use crate::dock::Group;
    // Floating in Studio, Properties docks only in Pro (Photoshop).
    let shown: Vec<Group> = [
        (Group::Color, p.color),
        (Group::Properties, t.pro && p.properties),
        (Group::Character, p.character),
        (Group::Navigator, p.navigator),
        (Group::History, p.history),
        (Group::Layers, p.layers),
    ]
    .into_iter()
    .filter_map(|(g, on)| on.then_some(g))
    .collect();
    if shown.is_empty() {
        return;
    }
    let margin = if t.pro { 2 } else { 8 };
    let mut panel = crate::direction::end_panel("dock").resizable(true).default_size(if t.pro { 290.0 } else { 300.0 }).size_range(DOCK_WIDTH);
    if let Some(w) = ui.ctx().data_mut(|d| d.remove_temp::<f32>(dock_width_id())) {
        panel = panel.exact_size(w);
    }
    panel.frame(egui::Frame::NONE.fill(t.dock).inner_margin(egui::Margin::same(margin))).show(ui, |ui| {
        // Groups keep their heights whatever they show (#88): see `dock`.
        crate::dock::show(app, ui, &shown, dock_body);
    });
}

/// One dock group's tab content; `dock` bounds it and scrolls it when it's taller.
pub(crate) fn dock_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, group: crate::dock::Group, tab: usize) {
    use crate::dock::Group;
    let pro = Tokens::get(ui.ctx()).pro;
    match (group, tab) {
        (Group::Color, 2) => crate::preset_panels::gradients_panel(app, ui),
        (Group::Color, 3) => crate::preset_panels::patterns_panel(app, ui),
        (Group::Color, 0) if pro => color_field(app, ui),
        (Group::Color, _) if pro => crate::swatches_ui::panel(app, ui),
        (Group::Color, 0) => crate::swatches_ui::panel(app, ui),
        (Group::Color, _) => color_picker(app, ui),
        (Group::Properties, 0) => properties_body(app, ui),
        (Group::Properties, _) => adjustments_grid(app, ui),
        (Group::Character, tab) => crate::type_tool::character_panel(app, ui, tab == 1),
        (Group::Navigator, 0) => navigator(app, ui),
        (Group::Navigator, 1) => crate::tone::histogram_panel(app, ui),
        (Group::Navigator, _) => info_panel(app, ui),
        (Group::History, 0) => history(app, ui),
        (Group::History, 1) => crate::actions::panel(app, ui),
        (Group::History, _) => crate::comps_ui::panel(app, ui),
        (Group::Layers, 0) => layers(app, ui),
        (Group::Layers, 1) => channels(app, ui),
        (Group::Layers, _) => crate::vector_ui::paths_panel(app, ui),
    }
}

/// Photoshop's Info panel: colour under the pointer (RGB and CMYK), position, selection size.
fn info_panel(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else {
        empty(ui, tl!("No document"));
        return;
    };
    let (rev, size, origin) = (st.revision, st.doc.size, st.doc.ruler_origin);
    // Positions and sizes in the ruler unit (Preferences › Units & Rulers).
    let units = app.session.prefs().units_and_rulers.clone();
    let dpi = st.doc.resolution_dpi as f64;
    let fx = move |px: f64| units.format(px, dpi, size.width as f64);
    let units_y = app.session.prefs().units_and_rulers.clone();
    let fy = move |px: f64| units_y.format(px, dpi, size.height as f64);
    let sel = st.doc.selection.as_ref().map(photocraft_compose::bounds::content_bounds);
    let pos = app
        .hover_doc
        .map(|p| (p[0].floor() as i32, p[1].floor() as i32))
        .filter(|(x, y)| *x >= 0 && *y >= 0 && *x < size.width as i32 && *y < size.height as i32);
    // The readout averages the Eyedropper's Sample Size, as in Photoshop (#1649).
    let sample_size = app.ui.tool_options.eyedropper_size;
    let rgba = pos.and_then(|(x, y)| {
        if let Some(((cx, cy, cr, cs), v)) = app.info_sample
            && (cx, cy, cr, cs) == (x, y, rev, sample_size)
        {
            return Some(v);
        }
        let params = json!({"x": f64::from(x) + 0.5, "y": f64::from(y) + 0.5, "size": sample_size});
        let v: Vec<f32> = serde_json::from_value(app.session.execute("document.sampleColor", params).ok()?).ok()?;
        let v = [*v.first()?, *v.get(1)?, *v.get(2)?, *v.get(3)?];
        app.info_sample = Some(((x, y, rev, sample_size), v));
        Some(v)
    });
    let mono = theme::mono(11.5);
    let row = |ui: &mut egui::Ui, k: &str, v: String| {
        ui.horizontal(|ui| {
            ui.add_sized([18.0, 16.0], egui::Label::new(RichText::new(k).color(t.text_dim).size(11.5)));
            ui.label(RichText::new(v).font(mono.clone()).color(t.text));
        });
    };
    ui.columns(2, |cols| {
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as i32;
        let dash = || "—".to_string();
        row(&mut cols[0], "R:", rgba.map_or_else(dash, |c| q(c[0]).to_string()));
        row(&mut cols[0], "G:", rgba.map_or_else(dash, |c| q(c[1]).to_string()));
        row(&mut cols[0], "B:", rgba.map_or_else(dash, |c| q(c[2]).to_string()));
        // Naive device CMYK (no ICC profile yet), in percent like Photoshop's readout. The Color
        // Sampler readout uses the same conversion (`sampler_cmds::device_cmyk`).
        let cmyk = rgba.map(|c| photocraft_engine::sampler_cmds::device_cmyk(c[0], c[1], c[2]));
        for (i, k) in ["C:", "M:", "Y:", "K:"].iter().enumerate() {
            row(&mut cols[1], k, cmyk.map_or_else(dash, |c| format!("{}%", c[i])));
        }
    });
    widgets::hairline(ui);
    ui.columns(2, |cols| {
        let dash = || "—".to_string();
        // Measured from the rulers' zero point (#2955).
        row(&mut cols[0], "X:", pos.map_or_else(dash, |p| fx(p.0 as f64 - origin[0])));
        row(&mut cols[0], "Y:", pos.map_or_else(dash, |p| fy(p.1 as f64 - origin[1])));
        row(&mut cols[1], "W:", sel.map_or_else(dash, |r| fx(r.width() as f64)));
        row(&mut cols[1], "H:", sel.map_or_else(dash, |r| fy(r.height() as f64)));
    });
    widgets::hairline(ui);
    ui.label(
        RichText::new(crate::i18n::fmt(tl!("Doc: {w} × {h} px"), &[("w", &size.width.to_string()), ("h", &size.height.to_string())]))
            .color(t.text_dim)
            .size(11.5),
    );
    // Color Sampler points (#1046): each under its number, with its position in the current ruler
    // unit and the same RGB/CMYK readout as the pointer (one composite pixel each).
    let samplers = app.session.execute("view.colorSamplers.list", json!({})).ok();
    let samplers = samplers.as_ref().and_then(|v| v["samplers"].as_array()).cloned().unwrap_or_default();
    if !samplers.is_empty() {
        widgets::hairline(ui);
        let cell = |ui: &mut egui::Ui, s: String| {
            ui.label(RichText::new(s).font(mono.clone()).color(t.text));
        };
        for s in &samplers {
            let n = s["number"].as_i64().unwrap_or(0);
            let at = s["position"]
                .as_array()
                .map_or((0.0, 0.0), |a| (a.first().and_then(Value::as_f64).unwrap_or(0.0), a.get(1).and_then(Value::as_f64).unwrap_or(0.0)));
            let num = |v: &Value, i: usize| v.get(i).and_then(Value::as_i64).map_or_else(|| "—".to_string(), |n| n.to_string());
            let (rgb, cmyk) = (&s["rgb"], &s["cmyk"]);
            ui.horizontal(|ui| {
                ui.add_sized([24.0, 16.0], egui::Label::new(RichText::new(format!("#{n}")).color(t.text_dim).size(11.5)));
                cell(ui, format!("X: {}  Y: {}", fx(at.0), fy(at.1)));
            });
            ui.horizontal(|ui| {
                ui.add_sized([24.0, 16.0], egui::Label::new(RichText::new("").size(11.5)));
                cell(ui, format!("R:{} G:{} B:{}", num(rgb, 0), num(rgb, 1), num(rgb, 2)));
            });
            ui.horizontal(|ui| {
                ui.add_sized([24.0, 16.0], egui::Label::new(RichText::new("").size(11.5)));
                cell(ui, format!("C:{}% M:{}% Y:{}% K:{}%", num(cmyk, 0), num(cmyk, 1), num(cmyk, 2), num(cmyk, 3)));
            });
        }
    }
}

fn navigator(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(idx) = app.session.active_index() else {
        empty(ui, tl!("No document"));
        return;
    };
    let ctx = ui.ctx().clone();
    let size = app.session.documents()[idx].doc.size;
    let avail = ui.available_width();
    // The preview fills the group's height above the zoom controls (their height as measured
    // last frame), so it grows when the group is made taller, as in Photoshop (#2726).
    let controls_h = ui.data(|d| d.get_temp::<f32>(navigator_controls_id())).unwrap_or(48.0);
    let left = ui.available_height() - controls_h;
    let box_h = if left.is_finite() { left.max(40.0) } else { 150.0 };
    let (frame, resp) = ui.allocate_exact_size(vec2(avail, box_h), Sense::click_and_drag());
    ui.painter().rect_filled(frame, t.radius_sm, t.canvas);
    let rect = navigator_fit(frame, size.width, size.height);
    ui.data_mut(|d| d.insert_temp(navigator_preview_id(), rect));
    // On the GPU canvas, its own texture at this size (#4518); otherwise the CPU thumbnail.
    if let Some(params) = crate::canvas::navigator_gpu_view(app, idx, rect, &ctx) {
        crate::gpu_canvas::GpuCanvas::paint(ui.painter(), rect, params);
    } else if crate::canvas::navigator_waits_for_gpu(app, idx, &ctx) {
        widgets::checker(ui.painter(), rect, 6.0);
    } else {
        let Some(tex) = crate::canvas::navigator_texture(app, &ctx, idx) else { return };
        widgets::checker(ui.painter(), rect, 6.0);
        ui.painter().image(tex, rect, Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
    }
    let s = rect.width() / size.width.max(1) as f32;
    if (resp.clicked() || resp.dragged())
        && let Some(pp) = resp.interact_pointer_pos()
    {
        let d = (pp - rect.min) / s;
        app.ui.views[idx].center = [d.x.clamp(0.0, size.width as f32), d.y.clamp(0.0, size.height as f32)];
        app.ui.views[idx].fit_pending = false;
    }
    // The Navigator has its own zoom target: an unmodified wheel over the preview
    // should zoom even when the document canvas is configured to scroll on wheel.
    // Use the canvas wheel classifier to honour one ×1.1 step per notch without
    // applying egui's smoothed wheel tail a second time.
    let pinch_zooms = app.session.prefs().enhanced_controls.zoom_with_trackpad_pinch;
    if resp.hovered()
        && let Some(crate::wheel_nav::Wheel::Zoom(factor)) = crate::wheel_nav::read(&ctx, true, pinch_zooms)
    {
        let view = &mut app.ui.views[idx];
        let next = crate::zoom_levels::clamp(view.zoom * factor, view.doc_size);
        if next != view.zoom {
            view.zoom = next;
            view.fit_pending = false;
        }
    }
    // Visible-area rectangle.
    let v = app.ui.views[idx].clone();
    let canvas = app.last_canvas_rect;
    let point_zoom = (v.zoom / app.canvas_ppp()).max(1e-6);
    // Pixel aspect correction shows fewer document columns across the same canvas width.
    let vw = canvas.width() / point_zoom / app.ui.view.display_aspect() * s;
    let vh = canvas.height() / point_zoom * s;
    let c = pos2(rect.min.x + v.center[0] * s, rect.min.y + v.center[1] * s);
    let vr = Rect::from_center_size(c, vec2(vw, vh)).intersect(frame.shrink(1.0));
    ui.painter().rect_stroke(vr, 2.0, Stroke::new(1.5, Color32::from_rgb(255, 84, 84)), StrokeKind::Middle);
    // Photoshop's one row under the preview (#4840): the zoom, typed in percent, then Zoom Out,
    // the slider and Zoom In.
    let controls = ui.vertical(|ui| {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            let set = |app: &mut PhotocraftApp, zoom: f32| {
                app.ui.views[idx].zoom = crate::zoom_levels::clamp(zoom, v.doc_size);
                app.ui.views[idx].fit_pending = false;
            };
            let mut pct = v.zoom * 100.0;
            let field = widgets::compact_wrapping_value_field(ui, &mut pct, crate::zoom_levels::percent_range(&app.ui.views[idx]), "%", vec2(64.0, NAV_ROW_H));
            if field.changed() {
                set(app, pct / 100.0);
            }
            let out = crate::brush_picker::named(icons::button(ui, "minus", NAV_ROW_H, false, tl!("Zoom Out")), "Zoom Out");
            if out.clicked() {
                set(app, crate::zoom_levels::step(v.zoom, -1, v.doc_size));
            }
            // The whole zoom range, and always the current zoom: a narrower slider would pull it back.
            let (lo, hi) = (crate::zoom_levels::min(v.doc_size).min(v.zoom).log2(), crate::zoom_levels::MAX.max(v.zoom).log2());
            let mut lz = v.zoom.log2();
            let slider_w = (ui.available_width() - NAV_ROW_H - 4.0).max(60.0);
            let slid = ui.allocate_ui(vec2(slider_w, NAV_ROW_H), |ui| lz.is_finite() && widgets::slider(ui, &mut lz, lo..=hi, None).changed()).inner;
            if slid {
                set(app, 2f32.powf(lz));
            }
            let zin = crate::brush_picker::named(icons::button(ui, "plus", NAV_ROW_H, false, tl!("Zoom In")), "Zoom In");
            if zin.clicked() {
                set(app, crate::zoom_levels::step(v.zoom, 1, v.doc_size));
            }
        });
    });
    // From the preview's bottom, so the spacing above the controls counts too.
    let controls = Rect::from_min_max(pos2(frame.left(), frame.bottom()), pos2(frame.right(), controls.response.rect.bottom()));
    ui.data_mut(|d| {
        d.insert_temp(navigator_controls_id(), controls.height());
        d.insert_temp(navigator_controls_rect_id(), controls);
    });
}

/// Height of the Navigator's zoom row: its field, buttons and slider.
const NAV_ROW_H: f32 = 22.0;

/// The document's `w` × `h` fitted inside the Navigator's `frame` with an 8 pt inset, centred and
/// keeping its aspect.
fn navigator_fit(frame: Rect, w: u32, h: u32) -> Rect {
    let inner = (frame.size() - vec2(16.0, 16.0)).max(vec2(1.0, 1.0));
    let (w, h) = (w.max(1) as f32, h.max(1) as f32);
    let s = (inner.x / w).min(inner.y / h);
    Rect::from_center_size(frame.center(), vec2(w * s, h * s))
}

fn navigator_preview_id() -> egui::Id {
    egui::Id::new("navigator-preview-rect")
}

fn navigator_controls_id() -> egui::Id {
    egui::Id::new("navigator-controls-height")
}

fn navigator_controls_rect_id() -> egui::Id {
    egui::Id::new("navigator-controls-rect")
}

/// Where the Navigator drew its document preview last frame.
#[cfg(test)]
pub(crate) fn last_navigator_preview(ctx: &egui::Context) -> Option<Rect> {
    ctx.data(|d| d.get_temp(navigator_preview_id()))
}

/// The Navigator's zoom controls (below the preview) last frame.
#[cfg(test)]
pub(crate) fn last_navigator_controls(ctx: &egui::Context) -> Option<Rect> {
    ctx.data(|d| d.get_temp(navigator_controls_rect_id()))
}

#[cfg(test)]
mod navigator_wheel_tests {
    use super::*;
    use egui::{Event, Modifiers, MouseWheelUnit, RawInput, TouchPhase};

    fn frame(app: &mut PhotocraftApp, ctx: &egui::Context, pointer: Pos2, wheel: f32) {
        let mut events = vec![Event::PointerMoved(pointer)];
        if wheel != 0.0 {
            events.push(Event::MouseWheel { unit: MouseWheelUnit::Line, delta: vec2(0.0, wheel), phase: TouchPhase::Move, modifiers: Modifiers::NONE });
        }
        let mut out = ctx
            .run_ui(RawInput { screen_rect: Some(Rect::from_min_size(Pos2::ZERO, vec2(430.0, 460.0))), events, ..Default::default() }, |ui| navigator(app, ui));
        out.textures_delta.clear();
    }

    #[test]
    fn navigator_wheel_changes_zoom_only_when_over_preview() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
        app.sync_views();
        app.ui.views[0].zoom = 1.0;
        app.ui.views[0].fit_pending = false;
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        crate::wheel_nav::configure(&ctx);
        frame(&mut app, &ctx, pos2(50.0, 55.0), 0.0);
        let center = app.ui.views[0].center;
        frame(&mut app, &ctx, pos2(50.0, 55.0), 1.0);
        assert!((app.ui.views[0].zoom - 1.1).abs() < 1e-4, "one wheel notch zooms once");
        assert_eq!(app.ui.views[0].center, center, "Navigator wheel must not pan");
        frame(&mut app, &ctx, pos2(50.0, 55.0), -1.0);
        assert!((app.ui.views[0].zoom - 1.0).abs() < 1e-4);
        // Below the preview (it now fills the group, #2726): over the zoom controls.
        let below = last_navigator_controls(&ctx).map_or(pos2(50.0, 455.0), |r| pos2(r.left() + 4.0, r.bottom() - 2.0));
        frame(&mut app, &ctx, below, 1.0);
        assert!((app.ui.views[0].zoom - 1.0).abs() < 1e-4, "wheel outside preview leaves zoom unchanged");
    }

    /// #4840: under the preview, one row as in Photoshop: the zoom in percent, Zoom Out, the
    /// slider and Zoom In. The buttons step the zoom like View › Zoom In / Out.
    #[test]
    fn navigator_zoom_row_is_one_line_and_its_buttons_step_the_zoom() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 100})).unwrap();
        app.sync_views();
        app.ui.views[0].zoom = 1.0;
        app.ui.views[0].fit_pending = false;
        let mut h = egui_kittest::Harness::builder().with_size(vec2(360.0, 300.0)).build_ui_state(|ui, app: &mut PhotocraftApp| navigator(app, ui), app);
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ALL[0]);
        h.run_steps(4);
        let row = last_navigator_controls(&h.ctx).expect("the zoom row");
        assert!(row.height() <= NAV_ROW_H + 14.0, "one row (with the gap above it): {row:?}");
        let zoom = |h: &egui_kittest::Harness<'_, PhotocraftApp>| h.state().ui.views[0].zoom;
        h.get_by_label("Zoom In").click();
        h.run_steps(2);
        assert!((zoom(&h) - crate::zoom_levels::step(1.0, 1, [200, 100])).abs() < 1e-4, "{}", zoom(&h));
        h.get_by_label("Zoom Out").click();
        h.run_steps(2);
        h.get_by_label("Zoom Out").click();
        h.run_steps(2);
        assert!((zoom(&h) - crate::zoom_levels::step(1.0, -1, [200, 100])).abs() < 1e-4, "{}", zoom(&h));
        assert!(!h.state().ui.views[0].fit_pending);
    }
}

fn empty(ui: &mut egui::Ui, s: &str) {
    let t = Tokens::get(ui.ctx());
    ui.add_space(6.0);
    ui.label(RichText::new(s).color(t.text_faint));
    ui.add_space(6.0);
}

/// The Color tab of the themes other than Photoshop's: the same spectrum field (chips, saturation
/// and brightness field, hue strip and readout) with HSB sliders under it (#4367).
fn color_picker(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    color_field(app, ui);
    ui.add_space(8.0);
    hsb_sliders(app, ui);
}

/// Hue, Saturation and Brightness sliders for the foreground colour.
fn hsb_sliders(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let fg = app.session.tools.foreground;
    // Keep the last edited HSB while it still gives the foreground: black and greys have no hue or
    // saturation of their own, so recomputing them from RGB would reset what was just typed to 0.
    let key = egui::Id::new("color-picker-hsva");
    let stored: Option<egui::ecolor::Hsva> = ui.data(|d| d.get_temp(key));
    let hsva0 = stored.filter(|h| srgb_bytes(hsva_srgb(*h)) == srgb_bytes(fg)).unwrap_or_else(|| srgb_hsva(fg));
    let mut h = hsva0.h * 360.0;
    let mut s = hsva0.s * 100.0;
    let mut v = hsva0.v * 100.0;
    let hue = widgets::hue_stops();
    // Preferences ▸ Interface ▸ Dynamic Color Sliders: off, the tracks are the plain ones.
    let dynamic = app.session.prefs().interface.dynamic_color_sliders;
    let mut changed = widgets::slider_row(ui, tl!("Hue"), &mut h, 0.0..=360.0, "°", dynamic.then_some(&hue)).changed();
    let sat_stops =
        [egui::ecolor::Hsva::new(hsva0.h, 0.0, hsva0.v.max(0.2), 1.0), egui::ecolor::Hsva::new(hsva0.h, 1.0, hsva0.v.max(0.2), 1.0)].map(hsva_color32);
    changed |= widgets::slider_row(ui, tl!("Saturation"), &mut s, 0.0..=100.0, "%", dynamic.then_some(&sat_stops)).changed();
    let val_stops = [Color32::BLACK, hsva_color32(egui::ecolor::Hsva::new(hsva0.h, hsva0.s, 1.0, 1.0))];
    changed |= widgets::slider_row(ui, tl!("Brightness"), &mut v, 0.0..=100.0, "%", dynamic.then_some(&val_stops)).changed();
    let hsva = egui::ecolor::Hsva::new(h / 360.0, s / 100.0, v / 100.0, 1.0);
    // Only an edit counts: the h/s/v round trip isn't exact, so comparing values would rewrite
    // the foreground (and recolour selected type) every frame.
    if changed {
        app.session.tools.foreground = hsva_srgb(hsva);
        ui.data_mut(|d| d.insert_temp(key, hsva));
        crate::type_tool::foreground_changed(app);
    }
}

// ----------------------------------------------------------------------------- layers

/// The simple themes' single lock button. On the Background it converts the layer to a normal
/// one, as clicking the Background's lock does in Photoshop. Elsewhere it clears every lock when
/// any is set, and locks all otherwise. Toggling only "lock all" would leave the Background's
/// transparency lock (still showing as locked), and a second click would lock its pixels,
/// shutting out the Eraser and every other paint tool (#76).
fn simple_lock_toggle(background: bool, l: &Layer) -> (String, Value) {
    if background {
        return ("layer.new.layerFromBackground".into(), json!({}));
    }
    let locks =
        if l.locks.any() { json!({"transparency": false, "pixels": false, "position": false, "artboard": false, "all": false}) } else { json!({"all": true}) };
    ("layer.setProps".into(), json!({"layer": l.id.0, "locks": locks}))
}

/// The Layers panel's blend modes: Photoshop's, plus the layer's own Paint.NET mode when it has one
/// (an imported `.pdn`), so the menu matches Photoshop's for every other layer.
fn blend_options(groups: bool, current: BlendMode) -> Vec<(BlendMode, &'static str)> {
    let own = Some(current).filter(|m| !m.has_psd_equivalent());
    std::iter::once(BlendMode::PassThrough).filter(|_| groups).chain(BlendMode::LAYER_MODES).chain(own).map(|m| (m, m.label())).collect()
}

/// Scroll the Layers panel while holding a layer drag over its top/bottom edge or past them.
///
/// Returns the *content* displacement in points for this frame, so positive moves the
/// list downward (reveals rows above) and negative upward (reveals rows below).
/// The speed ramps with proximity to the edge and clamps to the maximum speed when
/// dragged outside, using elapsed time instead of assuming a particular refresh rate.
fn layer_drag_edge_scroll(pointer: Option<Pos2>, viewport: Rect, dragging: bool, dt: f32) -> f32 {
    if !dragging || viewport.width() <= 0.0 || viewport.height() <= 0.0 || dt.is_nan() || dt <= 0.0 {
        return 0.0;
    }
    let Some(pointer) = pointer.filter(|p| p.x.is_finite() && p.y.is_finite() && p.x >= viewport.left() && p.x <= viewport.right()) else {
        return 0.0;
    };
    let edge = 32.0_f32.min(viewport.height() * 0.25);
    let top = ((edge - (pointer.y - viewport.top())) / edge).clamp(0.0, 1.0);
    let bottom = ((edge - (viewport.bottom() - pointer.y)) / edge).clamp(0.0, 1.0);
    let direction = top - bottom;
    if direction == 0.0 {
        return 0.0;
    }
    let velocity = 80.0 + 520.0 * direction.abs();
    direction.signum() * velocity * dt.clamp(0.0, 0.05)
}

/// Fit blend mode + Opacity into the available Layers-panel width (#2309).
/// Preserve the normal translated label when there is room; on a narrow dock,
/// keep the field interactive and give the dropdown the remaining width.
fn opacity_row_layout(available: f32, label_width: f32, gap: f32) -> (f32, bool) {
    let full_right = label_width + LAYER_PCT_W + 2.0 * gap + 16.0;
    // The label shows while the blend dropdown keeps a usable width (at the default dock width
    // with an English label it does; a long translated label on a narrow dock gives way).
    let show_label = available >= full_right + 80.0;
    let reserved = LAYER_PCT_W + gap + if show_label { label_width + gap + 16.0 } else { 0.0 };
    ((available - reserved).max(40.0), show_label)
}

fn layers(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    // A new active layer opens its parent groups and is scrolled into view (#152).
    let reveal = crate::layer_reveal::track(app, ui.ctx());
    let Some(st) = app.session.active() else {
        empty(ui, tl!("No document"));
        return;
    };
    let doc = st.doc.clone();
    let active = st.active_layer;
    let active_layer = active.and_then(|id| doc.layer(id));
    let selection = st.selected_layers();
    let isolated = st.isolated_layers.clone();
    let mut actions: Vec<(String, Value)> = Vec::new();

    let t = Tokens::get(ui.ctx());
    if t.pro {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 2.0;
            let (r, _) = ui.allocate_exact_size(vec2(64.0, 22.0), Sense::hover());
            ui.painter().rect_filled(r, t.radius_sm, t.field);
            ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
            icons::paint(ui, Rect::from_center_size(r.left_center() + vec2(11.0, 0.0), vec2(14.0, 14.0)), "search", 11.0, t.text_faint);
            ui.painter().text(r.left_center() + vec2(22.0, 0.0), Align2::LEFT_CENTER, tl!("Kind"), egui::FontId::proportional(11.5), t.text_dim);
            ui.add_space(4.0);
            for (kind, icon, tip) in [
                ("pixel", "image", tl!("Filter for pixel layers")),
                ("adjustment", "adjustment-layer", tl!("Filter for adjustment layers")),
                ("type", "type", tl!("Filter for type layers")),
                ("shape", "square", tl!("Filter for shape layers")),
                ("smart", "app-window", tl!("Filter for smart objects")),
            ] {
                let on = app.ui.layer_filter.iter().any(|k| k == kind);
                if icons::button(ui, icon, 22.0, on, tip).clicked() {
                    if on {
                        app.ui.layer_filter.retain(|k| k != kind);
                    } else {
                        app.ui.layer_filter.push(kind.to_string());
                    }
                }
            }
        });
        ui.add_space(4.0);
    }
    if let Some(l) = active_layer {
        // Photoshop greys blend mode, Opacity and Fill for the Background layer.
        let bg = crate::doc_props_ui::is_background(&doc, l);
        ui.horizontal(|ui| {
            ui.add_enabled_ui(!bg, |ui| {
                let mut m = l.blend;
                // Reserve space for the numeric field first. On a narrow dock, hide the
                // redundant visible Opacity label rather than force the dropdown beyond
                // the panel's right edge (#2309). The numeric field keeps its name.
                let opacity_label = if t.pro { tl!("Opacity:") } else { tl!("Opacity") };
                let (blend_width, show_opacity_label) =
                    opacity_row_layout(ui.available_width(), body_text_width(ui, opacity_label), ui.spacing().item_spacing.x);
                let (chosen, hovered) = widgets::dropdown_wheel_hovered(ui, "blend", &mut m, &blend_options(l.is_group(), l.blend), blend_width);
                // One step per choice: a click, an arrow key or each wheel notch (#1747).
                for m in &chosen {
                    actions.push(("layer.setProps".into(), json!({"layer": l.id.0, "blend": m.label()})));
                }
                // Hovering a mode previews it on the canvas (#970).
                crate::blend_preview::hover(app, l.id, hovered.filter(|_| chosen.is_empty()));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut o = l.opacity * 100.0;
                    let field = widgets::popup_value_field(ui, opacity_label, &mut o, 0.0..=100.0, "%", LAYER_PCT_W);
                    // On a narrow dock the (scrubby) label yields its room to the controls (#2309).
                    let scrub = if show_opacity_label {
                        scrub_pct_label(ui, opacity_label, &mut o, ui.available_width())
                    } else {
                        widgets::PopupFieldResponse::default()
                    };
                    if field.changed || scrub.changed {
                        actions.push(pct_action(l, "opacity", o, scrub.drag.or(field.drag)));
                    }
                });
            });
        });
        // Docks zero the item spacing: keep the Opacity and Fill fields apart (#155).
        ui.add_space((theme::ROW_GAP - ui.spacing().item_spacing.y).max(0.0));
        ui.horizontal(|ui| {
            let lock_label = if t.pro { tl!("Lock:") } else { tl!("Lock") };
            // Layer fill opacity (≠ Edit › Fill / path fill → Tô).
            let lang = crate::i18n::current();
            let fill_label = if t.pro { crate::i18n::tr_ctx(lang, "layers", "Fill:") } else { crate::i18n::tr_ctx(lang, "layers", "Fill") };
            // In a narrow panel with long translated labels, drop the "Lock:" text (the icons keep
            // their tooltips) rather than let the Fill label run over the lock icons.
            let gap = ui.spacing().item_spacing.x;
            let icons_w = if t.pro { 5.0 * 20.0 + gap } else { 22.0 + gap };
            let fits = body_text_width(ui, lock_label) + icons_w + body_text_width(ui, fill_label) + LAYER_PCT_W + 2.0 * gap <= ui.available_width();
            if fits {
                label(ui, lock_label);
            }
            if t.pro {
                // The lock icons touch, but only they: the Fill label keeps the same gap before its
                // field as Opacity's, so both stack in one right-aligned column (#2085).
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 0.0;
                    let lk = l.locks;
                    for (key, icon, tip, on) in [
                        ("transparency", "grid-3x3", tl!("Lock transparent pixels"), lk.transparency),
                        ("pixels", "brush", tl!("Lock image pixels"), lk.pixels),
                        ("position", "move", tl!("Lock position"), lk.position),
                        ("artboard", "scan", tl!("Prevent auto-nesting in and out of Artboards and Frames"), lk.artboard),
                        ("all", "lock", tl!("Lock all"), lk.all),
                    ] {
                        // Lock Image Pixels includes Lock Transparent Pixels, which waits for it (#4892).
                        let enabled = key != "transparency" || !lk.pixels;
                        if ui.add_enabled_ui(enabled, |ui| icons::button(ui, icon, 20.0, on, tip)).inner.clicked() {
                            actions.push(("layer.setProps".into(), json!({"layer": l.id.0, "locks": { key: !on }})));
                        }
                    }
                });
            }
            if !t.pro {
                let locked = l.locks.any();
                if icons::button(ui, if locked { "lock" } else { "lock-open" }, 22.0, locked, tl!("Lock layer")).clicked() {
                    actions.push(simple_lock_toggle(bg, l));
                }
            }
            ui.add_enabled_ui(!bg, |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let mut f = l.fill_opacity * 100.0;
                    let field = widgets::popup_value_field(ui, fill_label, &mut f, 0.0..=100.0, "%", LAYER_PCT_W);
                    let scrub = scrub_pct_label(ui, fill_label, &mut f, ui.available_width());
                    if field.changed || scrub.changed {
                        actions.push(pct_action(l, "fill", f, scrub.drag.or(field.drag)));
                    }
                })
            });
        });
        ui.add_space(2.0);
    }

    // Top of the stack first, groups above their contents, closed groups' contents hidden (#126).
    let rows = crate::layer_tree_ui::panel_rows(app, &doc);
    let ctx = ui.ctx().clone();
    let footer = widgets::footer_height(ui) + ui.spacing().item_spacing.y;
    let fill = ui.available_height() > footer + 60.0;
    let rows_h = if fill { ui.available_height() - footer } else { f32::INFINITY };
    egui::ScrollArea::vertical()
        .id_salt("layer-rows")
        .max_height(rows_h)
        .min_scrolled_height(if fill { rows_h } else { 0.0 })
        .auto_shrink([false, !fill])
        .show(ui, |ui| {
            // The drag keys are set only by actual layer-row and fx drags, not clicks or
            // ordinary scrolling. The ScrollArea applies this to its own content.
            // An fx drag that never saw its release (the panel was hidden) must not outlive the button.
            if !ctx.input(|i| i.pointer.primary_down() || i.pointer.any_released()) {
                ctx.data_mut(|d| d.remove::<FxDrag>(fx_drag_key()));
                ctx.data_mut(|d| d.remove::<MaskDrag>(mask_drag_key()));
                ctx.data_mut(|d| d.remove::<crate::smart_ui::FilterDrag>(crate::smart_ui::drag_key()));
            }
            let held = ctx.data(|d| {
                d.get_temp::<u64>(egui::Id::new("layer-drag")).is_some()
                    || d.get_temp::<FxDrag>(fx_drag_key()).is_some()
                    || d.get_temp::<crate::smart_ui::FilterDrag>(crate::smart_ui::drag_key()).is_some()
            });
            let dragging = held && ctx.input(|i| i.pointer.primary_down());
            let pointer = ctx.input(|i| i.pointer.interact_pos());
            let delta = layer_drag_edge_scroll(pointer, ui.clip_rect(), dragging, ctx.input(|i| i.stable_dt));
            if delta != 0.0 {
                ui.scroll_with_delta(vec2(0.0, delta));
                ctx.request_repaint();
            }
            let filter = app.ui.layer_filter.clone();
            let fx_collapsed = app.session.active().map(|d| d.fx_collapsed.clone()).unwrap_or_default();
            crate::layer_row_ui::begin(ui.ctx());
            // The drawn rows, top to bottom (#1721): the space between them and below the last
            // one is a drop zone too.
            let mut drawn: Vec<(LayerId, Rect)> = Vec::new();
            for &(depth, l) in &rows {
                // Select › Isolate Layers.
                if !photocraft_engine::select_extra_cmds::isolation_shows(&doc, &isolated, l.id) {
                    continue;
                }
                if !filter.is_empty() {
                    let kind = match &l.content {
                        LayerContent::Raster(_) => "pixel",
                        LayerContent::Adjustment(_) | LayerContent::Fill(_) => "adjustment",
                        LayerContent::Text(_) => "type",
                        LayerContent::Shape(_) => "shape",
                        LayerContent::Smart(_) => "smart",
                        LayerContent::Group(_) => "",
                    };
                    if !filter.iter().any(|k| k == kind) {
                        continue;
                    }
                }
                let row = RowSel { selected: selection.contains(&l.id), primary: active == Some(l.id), multi: selection.len() > 1 };
                let top = ui.cursor().top();
                drawn.push((l.id, layer_row(app, &ctx, ui, &doc, l, depth, row, &mut actions)));
                if reveal == Some(l.id) {
                    crate::layer_reveal::scroll_to_row(ui, top);
                }
                if !l.effects.items.is_empty() && fx_collapsed.iter().all(|id| *id != l.id) {
                    effect_rows(app, ui, l, depth, &mut actions);
                }
                crate::smart_ui::filter_rows(app, ui, l, depth, &mut actions);
            }
            // #1721: the list takes drops everywhere, not only on a row (Photoshop): the space
            // between two rows lands above the lower one, the empty space below the bottom-most
            // row below it. Both copy with ⌥ held, move without.
            for w in drawn.windows(2) {
                let gap = Rect::from_min_max(pos2(w[0].1.left(), w[0].1.bottom()), pos2(w[0].1.right(), w[1].1.top()));
                layer_drop_zone(app, &ctx, ui, gap, (w[1].0, w[1].1), "above", &mut actions);
            }
            if let Some(&(last, last_rect)) = drawn.last() {
                let below = Rect::from_min_max(pos2(last_rect.left(), last_rect.bottom()), pos2(last_rect.right(), ui.clip_rect().bottom()));
                layer_drop_zone(app, &ctx, ui, below, (last, last_rect), "below", &mut actions);
            }
            // Tab / ⇧Tab in a rename: rename the layer below / above next, as in Photoshop. The
            // Background isn't renamed in place (double-clicking it converts it), so it ends there.
            if let Some((from, up)) = crate::layer_row_ui::take_next_rename(ui.ctx())
                && let Some(at) = drawn.iter().position(|(id, _)| id.0 == from)
                && let Some(&(next, _)) = if up { at.checked_sub(1).and_then(|i| drawn.get(i)) } else { drawn.get(at + 1) }
                && let Some(l) = doc.layer(next).filter(|l| !crate::doc_props_ui::is_background(&doc, l))
                && let Some(done) = crate::layer_row_ui::start_rename(ui.ctx(), next.0, &l.name)
            {
                actions.push(done);
            }
            // ⌥-click the line between two layers: clip / release the upper one (#967).
            crate::clip_line_ui::show(ui, &doc, &mut actions);
            // A rename whose row is gone (deleted, filtered out, inside a closed group) ends,
            // committed: nothing else could commit or cancel it.
            if let Some(layer) = crate::layer_row_ui::renaming(ui.ctx())
                && !crate::layer_row_ui::recorded(ui.ctx()).iter().any(|r| r.layer == layer)
                && let Some(done) = crate::layer_row_ui::end_rename(ui.ctx(), true)
            {
                actions.push(done);
            }
        });
    // A layer being dragged can also be dropped on the footer's Delete, New Layer and Group
    // buttons (#736); read the drag before it ends.
    let footer_drag = ctx.data(|d| d.get_temp::<u64>(egui::Id::new("layer-drag"))).map(|id| {
        let in_selection = selection.iter().any(|s| s.0 == id);
        (id, in_selection)
    });
    // An effect row can be dropped on Delete too (#3102), and a mask's thumbnail (#3113, #3119).
    let fx_footer_drag = ctx.data(|d| d.get_temp::<FxDrag>(fx_drag_key()));
    fx_drag_feedback(&ctx, &doc);
    // And a smart filter's row (#4271).
    let filter_footer_drag = ctx.data(|d| d.get_temp::<crate::smart_ui::FilterDrag>(crate::smart_ui::drag_key()));
    crate::smart_ui::drag_feedback(&ctx, &doc);
    let mask_footer_drag = ctx.data(|d| d.get_temp::<MaskDrag>(mask_drag_key()));
    if let Some((_, kind)) = mask_footer_drag
        && let Some(p) = ctx.input(|i| i.pointer.interact_pos())
    {
        let label = if kind == MaskKind::Vector { tl!("Vector Mask") } else { tl!("Layer Mask") };
        crate::layer_transfer::ghost(&ctx, p, label);
    }
    // End any layer drag after every row has had a chance to accept the drop.
    if ctx.input(|i| i.pointer.any_released()) {
        ctx.data_mut(|d| d.remove::<u64>(egui::Id::new("layer-drag")));
        ctx.data_mut(|d| d.remove::<FxDrag>(fx_drag_key()));
        ctx.data_mut(|d| d.remove::<MaskDrag>(mask_drag_key()));
        ctx.data_mut(|d| d.remove::<crate::smart_ui::FilterDrag>(crate::smart_ui::drag_key()));
    }
    widgets::panel_footer(ui, |ui| {
        let active_layer = active.and_then(|id| doc.layer(id));
        let (delete, label) = if app.ui.vector_mask_target && active_layer.is_some_and(|l| l.vector_mask.is_some()) {
            ("layer.vectorMask.delete", "Delete Vector Mask")
        } else if app.ui.mask_target && active_layer.is_some_and(|l| l.mask.is_some()) {
            ("layer.layerMask.delete", "Delete Layer Mask")
        } else {
            ("layer.delete", "Delete layer")
        };
        let trash = icons::button(ui, "trash", 26.0, false, tl!(label));
        // A selected smart filter row is what Delete removes, as in Photoshop (#4751).
        let picked_filter = crate::smart_ui::selected(&ctx, &doc, active);
        if trash.clicked()
            && let Some(picked) = picked_filter
        {
            actions.push(crate::smart_ui::trash_action(picked));
            crate::smart_ui::deselect(&ctx, doc.id);
        } else if trash.clicked() {
            // Like Photoshop, ⌥/Alt-click deletes without asking.
            let confirm = delete == "layer.delete" && !ui.input(|i| i.modifiers.alt);
            let params = if confirm { json!({"__trash": true}) } else { json!({}) };
            actions.push((delete.into(), params));
        }
        actions.extend(footer_drop(ui, &trash, footer_drag, "layer.delete"));
        actions.extend(fx_trash_drop(ui, &trash, fx_footer_drag));
        if let Some(drag) = filter_footer_drag
            && dropped_on(ui, &trash)
        {
            actions.push(crate::smart_ui::trash_action(drag));
        }
        // Photoshop asks "Apply mask to layer before removing?" or "Delete the vector mask?"
        // (`mask_trash_prompt`).
        if let Some((layer, kind)) = mask_footer_drag
            && dropped_on(ui, &trash)
        {
            actions.push(crate::mask_trash_prompt::drop_action(layer, kind));
        }
        // A click follows the selected thumbnail; name that action for screen readers even
        // though dropping a whole layer row onto the same button always deletes the layer.
        trash.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!(label)));
        let new_layer = icons::button(ui, "square-plus", 26.0, false, &crate::shortcuts::tip_label(app, "Create a new layer", "layer.new.layer"));
        if new_layer.clicked() {
            // A click makes the layer above the selection; ⌥-click asks first (the New Layer
            // dialog) and ⌘/Ctrl-click adds it below, as in Photoshop.
            let (alt, below) = ui.input(|i| (i.modifiers.alt, i.modifiers.command));
            let params = if alt {
                Value::Null
            } else if below {
                json!({"position": "below"})
            } else {
                json!({})
            };
            actions.push(("layer.new.layer".into(), params));
        }
        actions.extend(footer_drop(ui, &new_layer, footer_drag, "layer.duplicate"));
        let group = icons::button(ui, "folder", 26.0, false, tl!("Create a new group"));
        if group.clicked() {
            // Like Photoshop a plain click adds an empty group; Shift-click groups the selected
            // layers, the same as Layer › Group Layers (#2561).
            let shift = ui.input(|i| i.modifiers.shift);
            let cmd = if shift && active_layer.is_some() { "layer.groupLayers" } else { "layer.new.group" };
            actions.push((cmd.into(), json!({})));
        }
        actions.extend(footer_drop(ui, &group, footer_drag, "layer.groupLayers"));
        let adj = footer_menu_button(ui, "adjustment-layer", 26.0, tl!("Create new fill or adjustment layer"));
        // Opens on the press like Photoshop's: drag onto an item and release to choose it (#2071).
        crate::press_menu::show(&adj, |ui, menu| {
            ui.set_min_width(190.0);
            for c in photocraft_engine::command_specs().iter().filter(|c| c.id.starts_with("layer.newAdjustmentLayer.")) {
                if menu.item(ui, tl!(c.label).trim_end_matches('…')) {
                    actions.push((c.id.into(), json!({})));
                }
            }
            ui.separator();
            if menu.item(ui, tl!("Solid Color…")) {
                actions.push(("layer.newFillLayer.solidColor".into(), json!({})));
            }
            if menu.item(ui, tl!("Gradient…")) {
                actions.push(("layer.newFillLayer.gradient".into(), json!({})));
            }
        });
        // Like Photoshop the button never replaces a mask (#2075): with a layer mask it adds a
        // vector mask, and with both it is greyed.
        let alt = ui.input(|i| i.modifiers.alt);
        let mask_cmd = crate::layer_menu_ui::mask_button_command(active_layer, doc.selection.is_some(), alt);
        let mask_tip = if active_layer.is_some_and(|l| l.mask.is_some()) {
            tl!("Add vector mask").to_string()
        } else {
            crate::i18n::fmt(tl!("Add a mask  (from the selection; {key} inverts)"), &[("key", &crate::shortcuts::pretty("Alt"))])
        };
        let mask_btn = ui.add_enabled_ui(mask_cmd.is_some(), |ui| icons::button(ui, "layer-mask", 26.0, false, &mask_tip)).inner;
        mask_btn.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &mask_tip));
        if mask_btn.clicked()
            && let Some(cmd) = mask_cmd
        {
            // A path selected in the Paths panel becomes the vector mask, as in Photoshop (#4728);
            // Alt still inverts it.
            match crate::vector_ui::selected_document_path(app).filter(|_| cmd.starts_with("layer.vectorMask.")) {
                Some(name) => actions.push(("layer.vectorMask.currentPath".into(), json!({"name": name, "hide": alt}))),
                None => actions.push((cmd.into(), json!({}))),
            }
        }
        let fx = fx_button(ui, 26.0, tl!("Add a layer style"));
        crate::press_menu::show(&fx, |ui, menu| {
            ui.set_min_width(180.0);
            if menu.item(ui, tl!("Blending Options…")) {
                crate::layer_style::open(app, None);
            }
            ui.separator();
            for &(kind, label) in crate::layer_style::KINDS {
                if menu.item(ui, format!("{}…", tl!(label))) {
                    crate::layer_style::open(app, Some(kind));
                }
            }
        });
        // Photoshop's footer starts with Link Layers (enabled with two or more layers selected).
        let can_link = app.session.is_enabled("layer.linkLayers");
        if ui.add_enabled_ui(can_link, |ui| icons::button(ui, "link-2", 26.0, false, tl!("Link layers"))).inner.clicked() {
            actions.push(("layer.linkLayers".into(), json!({})));
        }
    });
    for (id, p) in actions {
        if id == "ui.maskTarget" {
            app.ui.mask_target = p.as_bool().unwrap_or(false);
            app.ui.vector_mask_target = false;
            app.sync_mask_targets();
            continue;
        }
        if id == "ui.vectorMaskTarget" {
            app.ui.vector_mask_target = p.as_bool().unwrap_or(false);
            app.ui.mask_target = false;
            app.sync_mask_targets();
            continue;
        }
        // Double-clicking a mask thumbnail opens its settings: show Properties.
        if id == "ui.properties" {
            app.ui.panels.properties = true;
            app.ui.dock_tabs.properties = 0;
            app.ui.dock.collapsed.retain(|g| *g != crate::dock::Group::Properties);
            continue;
        }
        if p.is_null() {
            // Context-menu items behave like their menu-bar twins (dialogs included).
            let ctx = ui.ctx().clone();
            if let Err(e) = crate::menus::invoke(app, &ctx, &id, json!({})) {
                app.ui.status = e;
                app.ui.status_error = true;
            }
            continue;
        }
        let _ = app.run(&id, p);
    }
}

/// A layer row dragged onto a Layers panel footer button, as in Photoshop: onto Delete deletes it,
/// onto New Layer duplicates it, onto New Group groups it. A row that is part of the selection
/// carries the whole selection; any other row goes alone. Highlights the button while over it and
/// returns the command on release.
fn footer_drop(ui: &egui::Ui, button: &egui::Response, drag: Option<(u64, bool)>, command: &str) -> Option<(String, Value)> {
    // Named for screen readers (and tests) after the command a drop runs.
    button.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, footer_label(command)));
    let (layer, in_selection) = drag?;
    dropped_on(ui, button).then(|| (command.into(), if in_selection { json!({}) } else { json!({"layer": layer}) }))
}

/// While something is dragged over a footer `button`: highlights it, and is true on the release.
pub(crate) fn dropped_on(ui: &egui::Ui, button: &egui::Response) -> bool {
    let Some(p) = ui.ctx().input(|i| i.pointer.interact_pos()) else { return false };
    if !button.rect.contains(p) {
        return false;
    }
    let t = Tokens::get(ui.ctx());
    ui.painter().rect_stroke(button.rect, t.radius_sm, Stroke::new(2.0, t.accent), StrokeKind::Inside);
    ui.ctx().input(|i| i.pointer.any_released())
}

/// Effects dragged onto the footer's Delete button, as in Photoshop (#3102): the "Effects" row
/// clears the layer style, an effect's row deletes just that effect.
fn fx_trash_drop(ui: &egui::Ui, button: &egui::Response, drag: Option<FxDrag>) -> Option<(String, Value)> {
    let (layer, effect) = drag?;
    if !dropped_on(ui, button) {
        return None;
    }
    let params = match effect {
        Some(i) => json!({"layer": layer, "effect": i}),
        None => json!({"layer": layer}),
    };
    Some(("layer.layerStyle.clear".into(), params))
}

fn footer_label(command: &str) -> &'static str {
    match command {
        "layer.delete" => tl!("Delete layer"),
        "layer.duplicate" => tl!("Create a new layer"),
        _ => tl!("Create a new group"),
    }
}

/// The small down-arrow makes it clear that the footer icon expands into a menu.
fn footer_menu_arrow(ui: &egui::Ui, rect: Rect) {
    let t = Tokens::get(ui.ctx());
    let arrow = Rect::from_min_size(rect.right_bottom() - vec2(11.0, 10.0), vec2(9.0, 9.0));
    icons::paint(ui, arrow, "chevron-down", 8.0, t.text_dim);
}

fn footer_menu_button(ui: &mut egui::Ui, icon: &str, size: f32, tip: &str) -> egui::Response {
    let response = icons::button(ui, icon, size, false, tip);
    // Named for screen readers (and tests) after its tooltip.
    response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tip));
    footer_menu_arrow(ui, response.rect);
    response
}

/// Photoshop's italic "fx" footer button (no icon-font equivalent).
fn fx_button(ui: &mut egui::Ui, size: f32, tip: &str) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let (r, resp) = ui.allocate_exact_size(vec2(size, size), Sense::click());
    if resp.hovered() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    let mut job = egui::text::LayoutJob::default();
    job.append("fx", 0.0, egui::TextFormat { font_id: egui::FontId::proportional(15.0), color: t.icon, italics: true, ..Default::default() });
    let g = ui.painter().layout_job(job);
    ui.painter().galley(r.center() - g.size() / 2.0, g, t.icon);
    footer_menu_arrow(ui, r);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tip));
    resp.on_hover_text(tip)
}

/// How a Layers panel row is selected: in the (multi-)selection, the primary/active layer, and
/// whether several layers are selected.
#[derive(Clone, Copy)]
struct RowSel {
    selected: bool,
    primary: bool,
    multi: bool,
}

/// `layer.select` mode for a click with these modifiers (Photoshop: ⌘-click toggles, ⇧-click
/// selects a range, a plain click selects one layer).
fn select_mode(m: egui::Modifiers) -> &'static str {
    if m.command {
        "toggle"
    } else if m.shift {
        "range"
    } else {
        "replace"
    }
}

/// A drag down the Layers panel's eye column (Photoshop): the first eye toggles and every row
/// swept over gets the same visibility, once per row, all in one history step.
#[derive(Clone)]
struct EyeSweep {
    visible: bool,
    /// The `coalesce` key shared by the drag's `layer.setProps` calls.
    key: u64,
    swept: Vec<u64>,
}

/// Starts an eye-column sweep from `eye`, or applies a running one to the row `row` of `l`.
fn eye_sweep(ctx: &egui::Context, l: &Layer, row: Rect, eye: &egui::Response, actions: &mut Vec<(String, Value)>) {
    let id = egui::Id::new("layer-eye-sweep");
    let set =
        |visible: bool, key: u64| ("layer.setProps".to_string(), json!({"layer": l.id.0, "visible": visible, "coalesce": format!("layer-eye-sweep:{key}")}));
    if eye.drag_started() {
        let sweep = EyeSweep { visible: !l.visible, key: ctx.cumulative_pass_nr(), swept: vec![l.id.0] };
        actions.push(set(sweep.visible, sweep.key));
        ctx.data_mut(|d| d.insert_temp(id, sweep));
        return;
    }
    let Some(mut sweep) = ctx.data(|d| d.get_temp::<EyeSweep>(id)) else { return };
    let (down, pos, delta) = ctx.input(|i| (i.pointer.primary_down(), i.pointer.interact_pos(), i.pointer.delta()));
    if !down {
        ctx.data_mut(|d| d.remove::<EyeSweep>(id));
        return;
    }
    // Everything the pointer passed since the last frame, so a fast drag skips no row.
    let Some(p) = pos else { return };
    let (y0, y1) = ((p.y - delta.y).min(p.y), (p.y - delta.y).max(p.y));
    if y1 < row.top() || y0 >= row.bottom() || sweep.swept.contains(&l.id.0) {
        return;
    }
    if l.visible != sweep.visible {
        actions.push(set(sweep.visible, sweep.key));
    }
    sweep.swept.push(l.id.0);
    ctx.data_mut(|d| d.insert_temp(id, sweep));
}

/// Marker in the clipped row's existing 12-point slot, beside its thumbnail.
fn paint_clipping_indicator(painter: &egui::Painter, at: Pos2, color: Color32) {
    let tip = at + vec2(5.0, 3.0);
    let stroke = Stroke::new(1.0, color);
    painter.add(egui::Shape::line(vec![at + vec2(9.0, -4.0), at + vec2(5.0, -4.0), tip], stroke));
    painter.add(egui::Shape::line(vec![at + vec2(2.0, 0.0), tip, at + vec2(8.0, 0.0)], stroke));
}

#[allow(clippy::too_many_arguments)]
fn layer_row(
    app: &mut PhotocraftApp,
    ctx: &egui::Context,
    ui: &mut egui::Ui,
    doc: &photocraft_doc::Document,
    l: &Layer,
    depth: usize,
    row: RowSel,
    actions: &mut Vec<(String, Value)>,
) -> Rect {
    let selected = row.selected;
    let t = Tokens::get(ctx);
    // Photoshop's default (medium) thumbnails: 32 pt rows.
    let row_h = if t.pro { 32.0 } else { 46.0 };
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), row_h), Sense::click_and_drag());
    // A drag from a mask's thumbnail drags that mask, not the layer (#3113, #3119).
    let mask_drag = starts_mask_drag(ctx, l, &resp);
    if let Some(kind) = mask_drag {
        ctx.data_mut(|d| d.insert_temp::<MaskDrag>(mask_drag_key(), (l.id.0, kind)));
    }
    let mask_drag = mask_drag.is_some();
    layer_drag_and_drop(app, ctx, ui, l, rect, &resp, actions);
    fx_drop(ctx, ui, l, rect, actions);
    crate::smart_ui::drop_on_row(ctx, ui, l, rect, actions);
    if resp.drag_started() && !mask_drag {
        crate::layer_transfer::begin_from_panel(app, ctx, l.id);
    }
    // Rows are painted: name them for screen readers and UI tests.
    resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, selected, &l.name));
    // A row scrolled out of view only keeps its place (#125): a layout's hundreds of rows would
    // otherwise lay out names, icons and thumbnails every frame. Group rows stay whole: their
    // disclosure triangle is a widget (accessibility, scroll-to).
    if !ui.is_rect_visible(rect) && !l.is_group() && !resp.context_menu_opened() && crate::layer_row_ui::renaming(ctx) != Some(l.id.0) {
        return rect;
    }
    let painter = ui.painter_at(rect.expand(1.0));
    let is_hovered = resp.hovered() || app.canvas_hover_layer == Some(l.id);
    if resp.hovered() {
        app.hover_layer = Some(l.id);
    }
    if t.pro {
        if selected {
            painter.rect_filled(rect, 0.0, t.row_selected);
        } else if is_hovered {
            painter.rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
            painter.rect_filled(rect, 0.0, t.accent.gamma_multiply(0.12));
        }
    } else if selected {
        painter.rect_filled(rect, t.radius, t.hover);
        painter.rect_stroke(rect, t.radius, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    } else if is_hovered {
        painter.rect_filled(rect, t.radius, t.hover.gamma_multiply(0.5));
        painter.rect_filled(rect, t.radius, t.accent.gamma_multiply(0.12));
    }
    let eye_cell = Rect::from_min_max(rect.left_top(), pos2((rect.left() + 30.0).min(rect.right()), rect.bottom()));
    if let Some((_, background)) = t.layer_label_colors(l.label) {
        painter.rect_filled(eye_cell, 0.0, background);
    }
    if t.pro {
        // Paint dividers over the label so the eye column stays distinct.
        painter.line_segment([pos2(rect.left() + 30.0, rect.top()), pos2(rect.left() + 30.0, rect.bottom())], Stroke::new(1.0, t.separator));
        painter.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, t.separator));
    }
    let mut x = rect.left() + 6.0;
    let eye = Rect::from_min_size(pos2(x, rect.center().y - 11.0), vec2(22.0, 22.0));
    // The eye takes drags too (so a drag starting on it never reorders the row): dragging down
    // the eyes gives every row swept over the visibility the first eye toggled to.
    let eye_resp = ui.interact(eye, ui.id().with(("eye", l.id.0)), Sense::click_and_drag());
    // A hidden layer's eye box is left empty (still clickable).
    if l.visible {
        // Centre the drawing in the column while preserving the visibility hit area.
        icons::paint(ui, eye_cell, "eye", 15.0, t.layer_label_icon(l.label));
    }
    eye_sweep(ctx, l, rect, &eye_resp, actions);
    if eye_resp.clicked() {
        // ⌥-click shows only this layer; ⌥-click it again to restore the others.
        if ui.input(|i| i.modifiers.alt) {
            actions.push(("layer.showOnly".into(), json!({"layer": l.id.0})));
        } else {
            actions.push(("layer.setProps".into(), json!({"layer": l.id.0, "visible": !l.visible})));
        }
    }
    eye_resp.context_menu(|ui| {
        crate::widgets::style_spectrum_popup_menu(ui);
        crate::layer_menu_ui::color_choices(app, ui, l, row.multi && selected, actions);
    });
    // Everything but the thumbnails, the indentation and the name, so a narrow panel squeezes
    // the indentation first, then the thumbnails (a layer with two masks has three).
    let others = 6.0
        + 28.0
        + if l.is_group() { crate::layer_tree_ui::TRIANGLE_W } else { 0.0 }
        + if l.clipped { 12.0 } else { 0.0 }
        + crate::layer_row_ui::reserved_width(l);
    let ts = crate::mask_thumbs_ui::thumb_size(l, if t.pro { 24.0 } else { 34.0 }, rect.width() - others);
    let fixed = others + ts + 6.0 + crate::mask_thumbs_ui::width(l, ts);
    x += 28.0 + crate::layer_row_ui::indent(depth, rect.width(), fixed);
    let toggled = crate::layer_tree_ui::disclosure(ui, rect, &mut x, l, actions);
    if l.clipped {
        paint_clipping_indicator(&painter, pos2(x, rect.center().y), t.text_faint);
        x += 12.0;
    }
    let thumb = Rect::from_min_size(pos2(x, rect.center().y - ts / 2.0), vec2(ts, ts));
    draw_layer_thumb(app, ctx, ui, doc, l, thumb, row.primary);
    x += ts + 6.0;
    // Link chains and pixel / vector mask thumbnails (#153).
    let masks = crate::mask_thumbs_ui::paint(app, ctx, ui, &painter, doc, l, &mut x, rect.center().y, ts, actions);
    let mask_rect = masks.thumb(crate::mask_thumbs_ui::MaskKind::Pixel);
    let vector_rect = masks.thumb(crate::mask_thumbs_ui::MaskKind::Vector);
    // Photoshop frames the targeted thumbnail (pixels, mask or vector mask) of the active layer
    // with corner brackets.
    if row.primary {
        let target = if app.ui.vector_mask_target && vector_rect.is_some() {
            vector_rect
        } else if app.ui.mask_target {
            mask_rect.or(Some(thumb))
        } else {
            Some(thumb)
        };
        if let Some(r) = target {
            crate::mask_thumbs_ui::paint_brackets(&painter, r, t.text);
        }
    }
    // Right-hand indicators first; the name gets what is left and ends in "…" (#144).
    let fx_open = app.session.active().is_none_or(|d| !d.fx_collapsed.contains(&l.id));
    let (name_right, indicators, fx_toggled) = crate::layer_row_ui::indicators(ui, &painter, rect, x, l, fx_open, actions);
    // Dragging the fx badge drags all the layer's effects, not the layer (Photoshop). Like the eye,
    // it takes the drag from the row; clicks still reach the row.
    if let Some(&(_, badge)) = indicators.iter().find(|(k, _)| *k == crate::layer_row_ui::Indicator::Fx)
        && ui.interact(badge, ui.id().with(("fx-badge", l.id.0)), Sense::drag()).drag_started()
    {
        start_fx_drag(ctx, l.id, None);
    }
    // Clicking the lock icon unlocks the layer, and makes the Background a normal one (Photoshop,
    // #4723). It takes the click from the row, so the row isn't selected.
    if let Some(&(_, lock)) = indicators.iter().find(|(k, _)| *k == crate::layer_row_ui::Indicator::Lock)
        && ui.interact(lock, ui.id().with(("row-lock", l.id.0)), Sense::click()).clicked()
    {
        actions.push(simple_lock_toggle(doc.is_background(l.id), l));
    }
    let name_color = if l.visible { t.text } else { t.text_faint };
    let font = if selected && !t.pro { theme::medium(13.0) } else { egui::FontId::proportional(if t.pro { 12.0 } else { 13.0 }) };
    // Photoshop before 2026 set the Background layer's name in italics; 2026 sets it upright.
    let italic = !t.pro && doc.is_background(l.id);
    // Photoshop rows show only the name; the kind sub-label is a Studio-theme addition.
    let is_pixel = t.pro || matches!(l.content, LayerContent::Raster(_));
    let name_rect = crate::layer_row_ui::truncated(&painter, &l.name, font, name_color, italic, name_right - x).map(|galley| {
        let text_pos = pos2(x, rect.center().y - galley.size().y / 2.0 - if is_pixel { 0.0 } else { 7.0 });
        let r = Rect::from_min_size(text_pos, galley.size());
        painter.galley(text_pos, galley, name_color);
        r
    });
    if !is_pixel {
        let sub = match &l.content {
            LayerContent::Adjustment(a) => tl!(a.label()).to_string(),
            LayerContent::Group(g) => crate::i18n::trn(crate::i18n::current(), g.children.len() as u64, "Group · {n} layer", "Group · {n} layers"),
            other => tl!(other.kind_name()).to_string(),
        };
        crate::layer_row_ui::label(&painter, x, rect.center().y + 8.0, name_right, &sub, egui::FontId::proportional(11.0), t.text_faint);
    }
    crate::layer_row_ui::record(ctx, crate::layer_row_ui::RowRects { layer: l.id.0, row: rect, name: name_rect, indicators });
    // ⌘-click a layer, mask or vector-mask thumbnail loads its transparency / mask / path as a
    // selection (⇧ add, ⌥ subtract, ⇧⌥ intersect) instead of changing the layer selection.
    let thumb_load = resp.clicked().then(|| (ui.input(|i| i.modifiers), resp.interact_pointer_pos())).and_then(|(m, pos)| {
        let p = pos.filter(|_| m.command)?;
        if let Some(kind) = masks.hit(p) {
            return Some(crate::mask_thumbs_ui::load_params(l, kind, m));
        }
        thumb.expand(2.0).contains(p).then(|| json!({"channel": "transparency", "layer": l.id.0, "operation": crate::channels_panel::load_operation(m)}))
    });
    // ⇧-click a mask thumbnail: disable / enable that mask; ⌥-click a layer mask: view it.
    let mask_toggle = resp.clicked().then(|| (ui.input(|i| i.modifiers), resp.interact_pointer_pos())).and_then(|(m, pos)| {
        let kind = masks.hit(pos?)?;
        crate::mask_thumbs_ui::click_command(l, kind, m)
    });
    if let Some(p) = thumb_load {
        actions.push(("select.loadSelection".into(), p));
    } else if let Some(cmd) = mask_toggle {
        actions.push(cmd);
    } else if resp.clicked() && !eye_resp.clicked() && !toggled && !fx_toggled && !masks.clicked {
        let mode = select_mode(ui.input(|i| i.modifiers));
        actions.push(("layer.select".into(), json!({"layer": l.id.0, "mode": mode})));
        crate::smart_ui::deselect(ctx, doc.id);
        // A thumbnail explicitly selects content or a mask. Selecting an adjustment/fill row
        // defaults to its mask, but clicking its content thumbnail must leave the mask target.
        let pos = resp.interact_pointer_pos();
        let on_mask = mask_rect.zip(pos).is_some_and(|(r, p)| r.expand(2.0).contains(p));
        let on_thumb = pos.is_some_and(|p| thumb.expand(2.0).contains(p));
        let on_vector = pos.and_then(|p| masks.hit(p)) == Some(crate::mask_thumbs_ui::MaskKind::Vector);
        // Clicking the layer thumbnail leaves mask view (#196).
        let viewing = app.session.active().and_then(photocraft_engine::mask_view_cmds::current).is_some_and(|v| v.layer == l.id);
        if on_thumb && viewing {
            actions.push((photocraft_engine::mask_view_cmds::ID.into(), json!({"layer": l.id.0, "mode": "off"})));
        }
        let content_less = matches!(l.content, LayerContent::Adjustment(_) | LayerContent::Fill(_));
        if on_vector {
            // The vector mask thumbnail targets the vector mask: the path tools edit it and the
            // Paths panel selects the layer's path.
            app.ui.selected_path = Some("layer".into());
            actions.push(("ui.vectorMaskTarget".into(), json!(true)));
        } else if on_mask || (content_less && l.mask.is_some() && !on_thumb) {
            if on_mask {
                // The mask thumbnail also switches Properties to the mask (#3369).
                app.ui.properties_layer = None;
            }
            actions.push(("ui.maskTarget".into(), json!(true)));
        } else if on_thumb || !row.primary {
            actions.push(("ui.maskTarget".into(), json!(false)));
        }
        // ⌥-click the layer thumbnail: fit the layer on screen, as Photoshop does.
        let m = ui.input(|i| i.modifiers);
        if on_thumb && m.alt && !m.shift && !m.command {
            actions.push(("view.fitLayersOnScreen".into(), Value::Null));
        }
    }
    // Double-click: the name renames in place (over the row's full height, not just the glyphs:
    // #651); the Background, which can't be renamed while it's locked, becomes a normal layer; an
    // adjustment, fill or shape thumbnail opens its settings (a solid colour: the Color Picker,
    // #2812) and a Smart Object thumbnail its contents, and a type thumbnail edits its text;
    // anywhere else on the row opens Layer Style (#350, #537).
    // The first click already made this the active layer.
    if resp.double_clicked() {
        let pos = resp.interact_pointer_pos();
        let on = |r: Rect| pos.is_some_and(|p| r.expand(2.0).contains(p));
        if crate::doc_props_ui::is_background(doc, l) {
            actions.push(("layer.new.layerFromBackground".into(), json!({})));
        } else if name_rect.is_some_and(|n| on(Rect::from_x_y_ranges(n.x_range(), rect.y_range()))) {
            if let Some(done) = crate::layer_row_ui::start_rename(ctx, l.id.0, &l.name) {
                // One rename at a time: starting this one commits any other (#314).
                actions.push(done);
            }
        } else if let Some(kind) = pos.and_then(|p| masks.hit(p)) {
            match kind {
                crate::mask_thumbs_ui::MaskKind::Pixel => actions.push(("ui.maskTarget".into(), json!(true))),
                crate::mask_thumbs_ui::MaskKind::Vector => actions.push(("ui.vectorMaskTarget".into(), json!(true))),
            }
            if let Some((cmd, val)) = crate::mask_thumbs_ui::double_click_command(app, kind) {
                // The preference launches Select and Mask on a pixel mask.
                actions.push((cmd.into(), val));
            } else {
                // Otherwise double-click opens the mask's settings (Density/Feather).
                actions.push(("ui.properties".into(), json!(true)));
            }
        } else if pos.and_then(|p| masks.hit(p)).is_none() {
            let id = match &l.content {
                LayerContent::Adjustment(_) | LayerContent::Fill(_) | LayerContent::Shape(_) if on(thumb) => "layer.layerContentOptions",
                LayerContent::Smart(_) if on(thumb) => "layer.smartObjects.editContents",
                LayerContent::Text(_) if on(thumb) => "type.editText",
                _ => "layer.layerStyle.blendingOptions",
            };
            if crate::menus::is_enabled(app, id) {
                // Null params: run like the menu item, dialog included.
                actions.push((id.into(), Value::Null));
            }
        }
    }
    let edit_rect = Rect::from_min_max(pos2(x - 3.0, rect.center().y - 11.0), pos2(name_right.max(x + 40.0), rect.center().y + 11.0));
    if let Some(done) = crate::layer_row_ui::rename_field(ui, l.id.0, edit_rect) {
        actions.push(done);
    }
    // The visibility icon has its own color menu, and the layer-mask thumbnail its mask menu
    // (#4575); elsewhere use the full layer menu.
    let on_mask_id = ui.id().with(("mask-menu", l.id.0));
    if resp.secondary_clicked() {
        let on_mask = resp.interact_pointer_pos().and_then(|p| masks.hit(p)) == Some(crate::mask_thumbs_ui::MaskKind::Pixel);
        ctx.data_mut(|d| d.insert_temp(on_mask_id, on_mask));
    }
    if !eye_resp.secondary_clicked() && !eye_resp.context_menu_opened() {
        resp.context_menu(|ui| {
            crate::widgets::style_spectrum_popup_menu(ui);
            if ui.ctx().data(|d| d.get_temp::<bool>(on_mask_id)).unwrap_or(false) {
                crate::layer_menu_ui::show_mask(app, ui, l, actions);
                return;
            }
            // Right-clicking inside a multi-selection keeps it and acts on every selected layer.
            let on_set = row.multi && selected;
            if crate::layer_menu_ui::show(app, ui, l, on_set, actions)
                && let Some(done) = crate::layer_row_ui::start_rename(ui.ctx(), l.id.0, &l.name)
            {
                actions.push(done);
            }
        });
    }
    rect
}

/// Screen rect and UVs for a layer thumbnail: the document-shaped part of the square cell and of
/// the letterboxed square texture, so tall and wide documents don't show empty bars.
fn layer_thumb_fit(cell: Rect, w: u32, h: u32) -> (Rect, Rect) {
    crate::channels_panel::fit_thumb(cell, w, h)
}

fn draw_layer_thumb(app: &mut PhotocraftApp, ctx: &egui::Context, ui: &egui::Ui, doc: &photocraft_doc::Document, l: &Layer, rect: Rect, selected: bool) {
    let t = Tokens::get(ctx);
    let p = ui.painter();
    match &l.content {
        LayerContent::Adjustment(_) | LayerContent::Group(_) => {
            p.rect_filled(rect, 6.0, t.field);
            let icon = if l.is_group() { "folder" } else { "contrast" };
            icons::paint(ui, rect, icon, 18.0, t.icon);
        }
        LayerContent::Text(_) => {
            // Photoshop: type layers show a "T" tile rather than a pixel thumbnail.
            p.rect_filled(rect, if t.pro { 0.0 } else { 6.0 }, Color32::from_gray(if t.pro { 222 } else { 236 }));
            icons::paint(ui, rect, "type", 20.0, Color32::from_gray(40));
        }
        // Gradient fills show the gradient itself (Photoshop); solid ones their colour.
        LayerContent::Fill(f) if crate::gradient_ui::paint_thumbnail(ui, l.id, f, rect) => {}
        LayerContent::Fill(photocraft_doc::Fill::Solid(color)) => {
            crate::solid_fill_ui::paint_thumbnail(ui, crate::solid_fill_ui::thumbnail_color(app, doc.id, l.id, *color), rect);
        }
        LayerContent::Fill(_) => {
            let c = [128, 128, 128, 255];
            p.rect_filled(rect, 6.0, Color32::from_rgba_unmultiplied(c[0], c[1], c[2], c[3]));
        }
        _ => {
            // Keep row layout and outside thumbnail decorations when the image is clipped.
            if ui.is_rect_visible(rect) {
                // The texture is a letterboxed square: draw only the part the document fills.
                let (fitted, uv) = layer_thumb_fit(rect, doc.size.width, doc.size.height);
                widgets::checker(p, fitted, 5.0);
                let tex = app.layer_thumb(ctx, doc, l);
                p.image(tex, fitted, uv, Color32::WHITE);
            }
        }
    }
    let stroke = if t.pro {
        Stroke::new(1.0, Color32::from_gray(20))
    } else if selected {
        Stroke::new(1.5, t.text)
    } else {
        Stroke::new(1.0, t.field_border)
    };
    p.rect_stroke(rect, CornerRadius::same(if t.pro { 0 } else { 5 }), stroke, StrokeKind::Outside);
    crate::smart_ui::thumb_badge(ui, l, rect);
}

fn channels(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    crate::channels_panel::show(app, ui);
}

fn history(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else {
        empty(ui, tl!("No document"));
        return;
    };
    let doc = st.doc.clone();
    let entries = st.history.entries();
    let current = entries.len() - 1;
    let redo: Vec<String> = st.history.redo_labels().map(str::to_string).collect();
    // Snapshot row (Photoshop shows the document's opening state with a thumbnail).
    {
        let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 38.0), Sense::hover());
        let thumb = Rect::from_min_size(pos2(rect.left() + 28.0, rect.center().y - 14.0), vec2(28.0, 28.0));
        let ctx = ui.ctx().clone();
        if let Some(layer) = doc.layers.first() {
            widgets::checker(ui.painter(), thumb, 4.0);
            let tex = app.layer_thumb(&ctx, &doc, layer);
            ui.painter().image(tex, thumb, Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        }
        ui.painter().rect_stroke(thumb, 0.0, Stroke::new(1.0, t.separator), StrokeKind::Outside);
        ui.painter().text(pos2(thumb.right() + 8.0, rect.center().y), Align2::LEFT_CENTER, &doc.name, egui::FontId::proportional(12.0), t.text);
        ui.painter().line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0, t.separator));
    }
    let mut snap_cmd = None;
    let mut target: Option<usize> = None;
    let all = entries.iter().map(|e| (e.clone(), false)).chain(redo.iter().map(|e| (e.clone(), true)));
    // The dock gives History a fixed height: the rows scroll above the footer.
    let footer = if t.pro { widgets::footer_height(ui) + ui.spacing().item_spacing.y } else { 0.0 };
    let max_h = (ui.available_height() - footer).max(40.0);
    egui::ScrollArea::vertical().id_salt("history-rows").max_height(max_h).auto_shrink([false, false]).show(ui, |ui| {
        // Snapshots head the list and scroll with the states, so many of them never push the
        // footer out of a short dock.
        snap_cmd = snapshot_rows(app, ui);
        for (i, (e, is_redo)) in all.enumerate() {
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
            resp.widget_info(|| egui::WidgetInfo::selected(egui::WidgetType::SelectableLabel, true, i == current, &e));
            if i == current {
                ui.painter().rect_filled(rect, if t.pro { 0.0 } else { t.radius_sm }, if t.pro { t.row_selected } else { t.accent_soft });
            } else if resp.hovered() {
                ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
            }
            let icon = match e.as_str() {
                s if s.contains("Brush") || s.contains("Eraser") => "brush",
                s if s.contains("Layer") => "layers",
                s if s.contains("Select") || s.contains("Marquee") || s.contains("Deselect") || s.contains("Lasso") => "square-dashed",
                s if s.contains("Fill") => "paint-bucket",
                "Open" => "image",
                _ => "sliders-horizontal",
            };
            let color = if is_redo { t.text_faint } else { t.text };
            icons::paint(
                ui,
                Rect::from_center_size(pos2(rect.left() + 42.0, rect.center().y), vec2(16.0, 16.0)),
                icon,
                12.0,
                if is_redo { t.text_faint } else { t.icon },
            );
            ui.painter().text(pos2(rect.left() + 58.0, rect.center().y), Align2::LEFT_CENTER, &e, egui::FontId::proportional(12.0), color);
            if resp.clicked() && i != current {
                target = Some(i);
            }
        }
    });
    let mut footer_cmd: Option<&str> = None;
    if t.pro {
        let can_delete = app.session.is_enabled("history.deleteState");
        let can_new = app.session.is_enabled("history.newDocument");
        widgets::panel_footer(ui, |ui| {
            if ui.add_enabled_ui(can_delete, |ui| icons::button(ui, "trash", 26.0, false, tl!("Delete current state"))).inner.clicked() {
                footer_cmd = Some("history.deleteState");
            }
            let can_snap = app.session.is_enabled("history.newSnapshot");
            if ui.add_enabled_ui(can_snap, |ui| icons::button(ui, "scan", 26.0, false, tl!("Create new snapshot"))).inner.clicked() {
                footer_cmd = Some("history.newSnapshot");
            }
            if ui.add_enabled_ui(can_new, |ui| icons::button(ui, "file-plus", 26.0, false, tl!("Create new document from current state"))).inner.clicked() {
                footer_cmd = Some("history.newDocument");
            }
        });
    }
    // An open Free Transform owns Undo (transform_tool::intercept): stepping the document's history under
    // its box would leave it transforming pixels that changed.
    // Selecting a snapshot changes the document, so it waits for an open transform like the rows.
    let snap_cmd = snap_cmd.filter(|(cmd, _)| *cmd != "history.selectSnapshot" || app.ui.transform.is_none());
    if let Some((cmd, p)) = footer_cmd.map(|c| (c, json!({}))).filter(|_| app.ui.transform.is_none()).or(snap_cmd)
        && let Err(e) = app.run(cmd, p)
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
    // One command per click (#2966): stepping with Undo/Redo once per state made the canvas
    // recomposite the whole document.
    if let Some(state) = target.filter(|_| app.ui.transform.is_none())
        && let Err(e) = app.run("history.selectState", json!({ "state": state }))
    {
        app.ui.status = e;
        app.ui.status_error = true;
    }
}

/// The History panel's snapshot rows (#2966), under the opening row, each with its thumbnail and
/// name: a click returns to the snapshot (a new state), a double-click or right-click › Rename…
/// edits the name in place (Enter or a click elsewhere keeps it, Escape cancels), and
/// right-click › Delete removes it. Returns the command a row asks for.
fn snapshot_rows(app: &mut PhotocraftApp, ui: &mut egui::Ui) -> Option<(&'static str, Value)> {
    let t = Tokens::get(ui.ctx());
    let st = app.session.active()?;
    let (doc_id, snaps) = (st.doc.id, st.snapshots.clone());
    let live: Vec<u64> = snaps.iter().map(|s| s.id).collect();
    // Per document: snapshot ids restart in each, and a tab switch must not carry the edit over.
    let rename_id = egui::Id::new(("history-snapshot-rename", doc_id));
    let mut renaming: Option<(u64, String)> = ui.data(|d| d.get_temp(rename_id)).flatten();
    let mut cmd = None;
    let ctx = ui.ctx().clone();
    for snap in &snaps {
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 38.0), Sense::click());
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, &snap.name));
        if resp.hovered() {
            ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.5));
        }
        let thumb = Rect::from_min_size(pos2(rect.left() + 28.0, rect.center().y - 14.0), vec2(28.0, 28.0));
        widgets::checker(ui.painter(), thumb, 4.0);
        let tex = app.snapshot_thumb(&ctx, doc_id, snap, &live);
        ui.painter().image(tex, thumb, Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        ui.painter().rect_stroke(thumb, 0.0, Stroke::new(1.0, t.separator), StrokeKind::Outside);
        match renaming.as_mut().filter(|(id, _)| *id == snap.id) {
            Some((_, text)) => {
                let field = Rect::from_min_max(pos2(thumb.right() + 6.0, rect.center().y - 10.0), pos2(rect.right() - 6.0, rect.center().y + 10.0));
                let r = ui.put(field, egui::TextEdit::singleline(text));
                if r.lost_focus() {
                    if !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                        cmd = Some(("history.renameSnapshot", json!({"snapshot": snap.id, "name": text.clone()})));
                    }
                    renaming = None;
                } else if !r.has_focus() {
                    // Starting to rename: focus the field with the whole name selected, so typing
                    // replaces it (Photoshop).
                    r.request_focus();
                    if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), r.id) {
                        let all = egui::text::CCursorRange::two(egui::text::CCursor::new(0), egui::text::CCursor::new(text.chars().count()));
                        state.cursor.set_char_range(Some(all));
                        state.store(ui.ctx(), r.id);
                    }
                }
            }
            None => {
                ui.painter().text(pos2(thumb.right() + 8.0, rect.center().y), Align2::LEFT_CENTER, &snap.name, egui::FontId::proportional(12.0), t.text);
            }
        }
        if resp.double_clicked() {
            renaming = Some((snap.id, snap.name.clone()));
        } else if resp.clicked() {
            cmd = Some(("history.selectSnapshot", json!({"snapshot": snap.id})));
        }
        resp.context_menu(|ui| {
            if ui.button(tl!("Rename…")).clicked() {
                renaming = Some((snap.id, snap.name.clone()));
                ui.close();
            }
            if ui.button(tl!("Delete")).clicked() {
                cmd = Some(("history.deleteSnapshot", json!({"snapshot": snap.id})));
                ui.close();
            }
        });
    }
    if !snaps.is_empty() {
        let y = ui.cursor().top();
        ui.painter().hline(ui.max_rect().x_range(), y, Stroke::new(1.0, t.separator));
    }
    ui.data_mut(|d| d.insert_temp(rename_id, renaming));
    cmd
}

// ----------------------------------------------------------------------------- properties

/// Floating Properties card anchored to the canvas' top-right corner. It shows the same content as
/// the docked panel (`properties_body`): document, layer, adjustment and mask properties.
pub fn properties_window(app: &mut PhotocraftApp, ctx: &egui::Context) {
    if app.ui.docking.is_some() {
        return;
    }
    // Pro (Photoshop) docks Properties; Studio floats it over the canvas.
    if !app.ui.panels.properties || Tokens::get(ctx).pro {
        return;
    }
    if app.session.active().is_none() {
        return;
    }
    let t = Tokens::get(ctx);
    let canvas = app.last_canvas_rect;
    let width = 320.0;
    let pos = pos2(canvas.right() - width - 12.0, canvas.top() + 44.0);
    let frame = egui::Frame::NONE
        .fill(t.card)
        .stroke(Stroke::new(1.0, t.card_border))
        .corner_radius(CornerRadius::same(t.radius_lg as u8))
        .shadow(egui::Shadow { offset: [0, 12], blur: 36, spread: 0, color: t.shadow })
        .inner_margin(egui::Margin::same(14));
    // A floating panel (Order::Middle like other panels, so menus, popups and dialogs stay above
    // it), anchored to the canvas corner and dragged by its title off the part being worked on.
    let card_id = egui::Id::new("properties-card");
    let offset: egui::Vec2 = ctx.data(|m| m.get_temp(card_id)).unwrap_or_default();
    let mut drag = egui::Vec2::ZERO;
    let shown = egui::Area::new(card_id).order(egui::Order::Middle).fixed_pos(pos + offset).show(ctx, |ui| {
        frame.show(ui, |ui| {
            ui.set_width(width - 28.0);
            let title = ui.horizontal(|ui| {
                ui.label(RichText::new(tl!("Properties")).font(theme::semibold(13.5)).color(t.text));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if icons::button(ui, "minus", 22.0, false, tl!("Hide Properties")).clicked() {
                        app.ui.panels.properties = false;
                    }
                });
            });
            let bar = title.response.rect.with_max_x(title.response.rect.right() - 28.0);
            let grip = ui.interact(bar, card_id.with("title"), Sense::drag());
            if grip.hovered() || grip.dragged() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
            }
            drag = grip.drag_delta();
            ui.add_space(6.0);
            widgets::hairline(ui);
            ui.add_space(8.0);
            properties_body(app, ui);
        });
    });
    if drag != egui::Vec2::ZERO && canvas.is_positive() {
        // Keep the title bar inside the canvas.
        let r = shown.response.rect.translate(drag);
        let dx = (canvas.left() - r.left()).max(0.0) - (r.right() - canvas.right()).max(0.0);
        let dy = (canvas.top() - r.top()).max(0.0) - (r.top() + 40.0 - canvas.bottom()).max(0.0);
        ctx.data_mut(|m| m.insert_temp(card_id, offset + drag + egui::vec2(dx, dy)));
    }
}

fn adjustment_controls(app: &mut PhotocraftApp, ui: &mut egui::Ui, id: LayerId, adj: &photocraft_doc::Adjustment) {
    crate::adjust_editors::layer_editor(app, ui, id, adj);
    ui.add_space(6.0);
    let layer = app.session.active().and_then(|s| s.doc.layer(id).map(|l| (l.visible, l.clipped)));
    if let Some((visible, clipped)) = layer {
        ui.horizontal(|ui| {
            let mut vis = visible;
            if widgets::toggle(ui, &mut vis, tl!("Adjustment visible")).changed() {
                let _ = app.run("layer.setProps", json!({"layer": id.0, "visible": vis}));
            }
            let mut clip = clipped;
            if widgets::toggle(ui, &mut clip, tl!("Clip to layer below")).changed() {
                let _ = app.run("layer.setProps", json!({"layer": id.0, "clipped": clip}));
            }
        });
    }
    ui.add_space(6.0);
    if widgets::secondary_button(ui, tl!("Reset to defaults"), ui.available_width()).clicked() {
        let doc = app.session.active().map_or(0, |s| s.doc.id.0);
        // Photoshop's two-step Reset where measured (Color Balance), else the defaults.
        let params = crate::adjust_editors::properties_reset(ui.ctx(), doc, id, adj).unwrap_or_else(|| json!({"layer": id.0}));
        let _ = app.run("layer.setAdjustment", params);
        app.live_adjust = None;
    }
}

fn layer_controls(app: &mut PhotocraftApp, ui: &mut egui::Ui, layer: &Layer) {
    let t = Tokens::get(ui.ctx());
    if let Some(s) = layer.surface() {
        let b = app.cached_bounds(layer.id.0, s);
        egui::Grid::new("props-grid").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            for (k, v) in [("X", b.x0.to_string()), ("Y", b.y0.to_string()), ("W", b.width().to_string()), ("H", b.height().to_string())] {
                ui.label(RichText::new(k).color(t.text_dim));
                ui.label(RichText::new(format!("{v} px")).font(theme::mono(12.0)).color(t.text));
                ui.end_row();
            }
        });
        ui.add_space(6.0);
    }
    let mut o = layer.opacity * 100.0;
    let r = widgets::slider_row(ui, tl!("Opacity"), &mut o, 0.0..=100.0, "%", None);
    if r.drag_stopped() || (r.changed() && !r.dragged()) {
        let _ = app.run("layer.setProps", json!({"layer": layer.id.0, "opacity": o / 100.0}));
    }
    let mut f = layer.fill_opacity * 100.0;
    let r = widgets::slider_row(ui, crate::i18n::tr_ctx(crate::i18n::current(), "layers", "Fill"), &mut f, 0.0..=100.0, "%", None);
    if r.drag_stopped() || (r.changed() && !r.dragged()) {
        let _ = app.run("layer.setProps", json!({"layer": layer.id.0, "fill": f / 100.0}));
    }
}

/// Docked Properties body (Pro theme).
fn properties_body(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let identity = app.session.active().map(|st| (st.doc.id, st.active_layer, app.ui.mask_target, app.ui.vector_mask_target, app.ui.properties_layer));
    ui.push_id(("properties-target", identity), |ui| properties_body_inner(app, ui));
}

fn properties_body_inner(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let Some(st) = app.session.active() else {
        empty(ui, tl!("No properties"));
        return;
    };
    let doc = st.doc.clone();
    if let Some(layer) = st.active_layer.and_then(|id| doc.layer(id))
        && crate::mask_props_ui::targeted(app, layer)
    {
        crate::props_layout::header(ui, layer);
        crate::mask_props_ui::properties(app, ui, layer);
        return;
    }
    // The vector mask's own controls, when its thumbnail is targeted.
    if let Some(layer) = st.active_layer.and_then(|id| doc.layer(id))
        && crate::mask_props_ui::vector_targeted(app, layer)
    {
        crate::props_layout::header(ui, layer);
        crate::mask_props_ui::vector_properties(app, ui, layer);
        return;
    }
    // Photoshop shows the Document properties when nothing or the Background layer is selected,
    // with Align and Distribute above them when the Background is one of several (#4608).
    if crate::doc_props_ui::shows_document(&st.doc, st.active_layer) {
        if t.pro && st.selected_layers().len() > 1 {
            crate::layer_props_ui::selection_align(app, ui);
        }
        crate::doc_props_ui::properties(app, ui);
        return;
    }
    let Some(id) = st.active_layer else { return };
    // Borrowed from the document snapshot: cloning the layer every frame copied whole groups.
    let Some(layer) = doc.layer(id) else { return };
    // Header: kind icon, layer name and kind (#155); sections below draw their own separators.
    crate::props_layout::header(ui, layer);
    let is_adjustment = matches!(layer.content, LayerContent::Adjustment(_));
    if is_adjustment || layer.artboard().is_some() || !t.pro {
        ui.add_space(4.0);
        widgets::hairline(ui);
        ui.add_space(6.0);
    }
    if let LayerContent::Adjustment(adj) = &layer.content {
        adjustment_controls(app, ui, id, adj);
    } else if layer.artboard().is_some() {
        crate::artboard_ui::properties(app, ui, layer);
    } else if t.pro {
        // Transform, Align, the kind's sections and Quick Actions.
        crate::layer_props_ui::properties(app, ui, layer);
    } else {
        if matches!(layer.content, LayerContent::Fill(photocraft_doc::Fill::Gradient { .. })) {
            crate::gradient_ui::properties(app, ui, layer);
            ui.add_space(6.0);
        }
        layer_controls(app, ui, layer);
        if matches!(layer.content, LayerContent::Text(_)) {
            crate::type_tool::type_properties(app, ui);
        }
        if matches!(layer.content, LayerContent::Shape(_)) {
            crate::vector_ui::shape_properties(app, ui, id);
        }
    }
}

/// Photoshop's Adjustments panel: a grid of one-click adjustment layers.
fn adjustments_grid(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    ui.label(RichText::new(tl!("Add an adjustment")).color(t.text_dim));
    ui.add_space(4.0);
    let items: [(&str, &str, &str); 16] = [
        ("brightnessContrast", "sun", tl!("Brightness/Contrast")),
        ("levels", "gauge", tl!("Levels")),
        ("curves", "pen-tool", tl!("Curves")),
        ("exposure", "scan", tl!("Exposure")),
        ("vibrance", "sparkles", tl!("Vibrance")),
        ("hueSaturation", "droplet", tl!("Hue/Saturation")),
        ("colorBalance", "blend", tl!("Color Balance")),
        ("blackWhite", "contrast", tl!("Black & White")),
        ("photoFilter", "circle-dot", tl!("Photo Filter")),
        ("channelMixer", "sliders-horizontal", tl!("Channel Mixer")),
        ("invert", "squares-subtract", tl!("Invert")),
        ("posterize", "layers", tl!("Posterize")),
        ("threshold", "square", tl!("Threshold")),
        ("gradientMap", "palette", tl!("Gradient Map")),
        ("selectiveColor", "swatch-book", tl!("Selective Color")),
        ("colorLookup", "grid-3x3", tl!("Color Lookup")),
    ];
    let mut run = None;
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(2.0, 2.0);
        for (kind, icon, tip) in items {
            if icons::button(ui, icon, 30.0, false, tip).clicked() {
                run = Some(format!("layer.newAdjustmentLayer.{kind}"));
            }
        }
    });
    if let Some(id) = run {
        let _ = app.run(&id, json!({}));
        app.ui.dock_tabs.properties = 0;
    }
}

/// The Color panel's foreground and background chips. A click picks the colour the field edits,
/// framed; a double-click opens the Color Picker on it.
fn field_chips(app: &mut PhotocraftApp, ui: &mut egui::Ui, chips: Rect) {
    let t = Tokens::get(ui.ctx());
    let bgr = Rect::from_min_size(chips.min + vec2(13.0, 13.0), vec2(22.0, 22.0));
    let fgr = Rect::from_min_size(chips.min + vec2(3.0, 3.0), vec2(22.0, 22.0));
    let frame = Stroke::new(1.0, t.text_dim);
    let bg_active = app.ui.color_panel.background;
    let p = ui.painter();
    p.rect_filled(bgr, 2.0, c32(app.session.tools.background));
    p.rect_stroke(bgr, 2.0, Stroke::new(1.0, t.field_border), StrokeKind::Outside);
    if bg_active {
        p.rect_stroke(bgr.expand(2.0), 2.0, frame, StrokeKind::Outside);
    }
    p.rect_filled(fgr, 2.0, c32(app.session.tools.foreground));
    p.rect_stroke(fgr, 2.0, Stroke::new(1.0, Color32::from_gray(210)), StrokeKind::Outside);
    if !bg_active {
        p.rect_stroke(fgr.expand(2.0), 2.0, frame, StrokeKind::Outside);
    }
    // The foreground is on top, so it takes the clicks where the two overlap.
    let bg_resp = ui.interact(bgr, ui.id().with("field-bg"), Sense::click());
    let fg_resp = ui.interact(fgr, ui.id().with("field-fg"), Sense::click());
    let picked = if fg_resp.clicked() { false } else { bg_resp.clicked() || bg_active };
    app.ui.color_panel.background = picked;
    if fg_resp.double_clicked() || bg_resp.double_clicked() {
        crate::color_picker_ui::open(app, if picked { "background" } else { "foreground" });
    }
}

/// Where the dock tells the Color field how tall its group body is.
pub(crate) fn color_field_fill_id() -> egui::Id {
    egui::Id::new("color-field-fill-height")
}

/// Height of the Color field: the group body's height (when docked) less the hex/RGB readout,
/// which folds onto a second line in a narrow dock. 120 pt when nothing sets a height.
fn color_field_height(avail: Option<f32>, width: f32) -> f32 {
    let reserve = 32.0 + if width < 280.0 { 26.0 } else { 0.0 };
    avail.filter(|a| a.is_finite()).map_or(120.0, |a| (a - reserve).clamp(120.0, 2000.0))
}

/// The colour the SV field draws for `hsva`, with what the field showed last (`hue`, `saturation`)
/// filling in what the colour itself cannot carry: a grey has no hue of its own, and black has
/// neither hue nor saturation — so a drag that bottoms out into black would otherwise round-trip
/// through RGB as `s = 0` and drop the knob from the bottom-right corner to the bottom-left one
/// (#3228).
fn field_hsva(mut hsva: egui::ecolor::Hsva, remembered: (f32, f32)) -> egui::ecolor::Hsva {
    if hsva.v < 0.01 {
        hsva.h = remembered.0;
        hsva.s = remembered.1;
    } else if hsva.s < 0.01 {
        // A grey: the saturation really is zero (that is where the knob belongs), only the hue
        // is undefined.
        hsva.h = remembered.0;
    }
    hsva
}

/// Photoshop Color panel: saturation/brightness field + hue strip, drawn as shaded meshes.
fn color_field(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    let t = Tokens::get(ui.ctx());
    let bg_active = app.ui.color_panel.background;
    let key = egui::Id::new(("color-field-hue", bg_active));
    let mut hsva = srgb_hsva(if bg_active { app.session.tools.background } else { app.session.tools.foreground });
    // What the field showed last: a grey carries no hue, black neither hue nor saturation (#3228).
    let remembered: (f32, f32) = ui.data(|d| d.get_temp(key)).unwrap_or((hsva.h, hsva.s));
    hsva = field_hsva(hsva, remembered);
    let w = ui.available_width();
    let strip_w = 14.0;
    // Photoshop's Color panel field fills the panel: take the height the dock offers, less the
    // hex/RGB readout (which folds onto a second line in a narrow dock). Without a dock, 120 pt.
    // Only Photoshop's themes stretch it over the group: the other themes put sliders under it.
    let fill = if t.pro { ui.data(|d| d.get_temp::<f32>(color_field_fill_id())) } else { None };
    let h = color_field_height(fill, w);
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 8.0;
        let (chips, _) = ui.allocate_exact_size(vec2(38.0, h), Sense::hover());
        field_chips(app, ui, chips);
        // SV field.
        let field_w = w - 38.0 - strip_w - 16.0;
        let (field, fresp) = ui.allocate_exact_size(vec2(field_w, h), Sense::click_and_drag());
        let mut mesh = egui::Mesh::default();
        let n = 16;
        for j in 0..=n {
            for i in 0..=n {
                let (sx, vy) = (i as f32 / n as f32, j as f32 / n as f32);
                let c = hsva_color32(egui::ecolor::Hsva::new(hsva.h, sx, 1.0 - vy, 1.0));
                mesh.colored_vertex(pos2(field.left() + sx * field.width(), field.top() + vy * field.height()), c);
            }
        }
        for j in 0..n {
            for i in 0..n {
                let a = (j * (n + 1) + i) as u32;
                let b = a + 1;
                let c = a + (n + 1) as u32;
                let d = c + 1;
                mesh.add_triangle(a, b, d);
                mesh.add_triangle(a, d, c);
            }
        }
        ui.painter().add(mesh);
        ui.painter().rect_stroke(field, 0.0, Stroke::new(1.0, t.separator), StrokeKind::Outside);
        if (fresp.dragged() || fresp.clicked())
            && let Some(p) = fresp.interact_pointer_pos()
        {
            hsva.s = ((p.x - field.left()) / field.width()).clamp(0.0, 1.0);
            hsva.v = 1.0 - ((p.y - field.top()) / field.height()).clamp(0.0, 1.0);
        }
        let knob = pos2(field.left() + hsva.s * field.width(), field.top() + (1.0 - hsva.v) * field.height());
        ui.painter().circle_stroke(knob, 5.0, Stroke::new(1.5, Color32::WHITE));
        ui.painter().circle_stroke(knob, 6.5, Stroke::new(1.0, Color32::from_black_alpha(160)));
        // Hue strip.
        let (strip, sresp) = ui.allocate_exact_size(vec2(strip_w, h), Sense::click_and_drag());
        let mut m2 = egui::Mesh::default();
        let steps = 24;
        for k in 0..=steps {
            let f = k as f32 / steps as f32;
            let c = hsva_color32(egui::ecolor::Hsva::new(1.0 - f, 1.0, 1.0, 1.0));
            m2.colored_vertex(pos2(strip.left(), strip.top() + f * strip.height()), c);
            m2.colored_vertex(pos2(strip.right(), strip.top() + f * strip.height()), c);
        }
        for k in 0..steps {
            let a = (k * 2) as u32;
            m2.add_triangle(a, a + 1, a + 3);
            m2.add_triangle(a, a + 3, a + 2);
        }
        ui.painter().add(m2);
        if (sresp.dragged() || sresp.clicked())
            && let Some(p) = sresp.interact_pointer_pos()
        {
            hsva.h = 1.0 - ((p.y - strip.top()) / strip.height()).clamp(0.0, 0.9999);
        }
        let y = strip.top() + (1.0 - hsva.h) * strip.height();
        let tri = vec![pos2(strip.right() + 1.0, y), pos2(strip.right() + 6.0, y - 4.0), pos2(strip.right() + 6.0, y + 4.0)];
        ui.painter().add(egui::Shape::convex_polygon(tri, t.text, Stroke::NONE));
        if fresp.dragged() || fresp.clicked() || sresp.dragged() || sresp.clicked() {
            if bg_active {
                app.session.tools.background = hsva_srgb(hsva);
            } else {
                app.session.tools.foreground = hsva_srgb(hsva);
                crate::type_tool::foreground_changed(app);
            }
            ui.data_mut(|d| d.insert_temp(key, (hsva.h, hsva.s)));
        }
    });
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        let mut c = if bg_active { app.session.tools.background } else { app.session.tools.foreground };
        if color_readout(ui, ui.id().with(("color-field", bg_active)), &mut c) {
            if bg_active {
                app.session.tools.background = c;
            } else {
                app.session.tools.foreground = c;
                crate::type_tool::foreground_changed(app);
            }
            // Keep the hue for greys, so the field marker doesn't jump to red; the saturation with
            // it, so a later black lands on the corner it was last dragged to (#3228).
            let h = srgb_hsva(c);
            if h.s >= 0.01 && h.v >= 0.01 {
                ui.data_mut(|d| d.insert_temp(key, (h.h, h.s)));
            }
        }
    });
}

/// Tool colours are sRGB-encoded floats; egui's Hsva works on sRGB bytes via these helpers
/// (its `from_rgba_unmultiplied` expects *linear* RGB, which gave wrong readouts).
fn srgb_bytes(c: [f32; 4]) -> [u8; 3] {
    let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    [q(c[0]), q(c[1]), q(c[2])]
}

/// The Color panel's HSB, as Photoshop's and the Color Picker's: on the sRGB-encoded values,
/// so a 50% grey has Brightness 50%. egui's `Hsva` conversions work in linear light, which put
/// that grey at 22% and drew the field's middle as 188 (#3323); `Hsva` only holds the numbers.
fn srgb_hsva(c: [f32; 4]) -> egui::ecolor::Hsva {
    let [h, s, v] = crate::color_picker_ui::rgb_to_hsv([c[0], c[1], c[2]].map(|v| v.clamp(0.0, 1.0)));
    egui::ecolor::Hsva::new(h / 360.0, s, v, 1.0)
}

fn hsva_srgb(h: egui::ecolor::Hsva) -> [f32; 4] {
    let [r, g, b] = crate::color_picker_ui::hsv_to_rgb(h.h * 360.0, h.s.clamp(0.0, 1.0), h.v.clamp(0.0, 1.0));
    [r, g, b, 1.0]
}

/// The colour to draw for Color panel HSB `h` (see [`srgb_hsva`]).
fn hsva_color32(h: egui::ecolor::Hsva) -> Color32 {
    let [r, g, b] = srgb_bytes(hsva_srgb(h));
    Color32::from_rgb(r, g, b)
}

/// The Color panel's editable colour readout: a hex field and R, G, B fields, all drawn like the
/// dock's [`widgets::value_field`] so they sit in the theme. Editing any of them sets `color`
/// (sRGB-encoded floats) and returns `true`.
fn color_readout(ui: &mut egui::Ui, id: egui::Id, color: &mut [f32; 4]) -> bool {
    let t = Tokens::get(ui.ctx());
    let mut changed = false;
    // Wrapped so the fields fold onto a second line in a narrow dock instead of being clipped.
    ui.horizontal_wrapped(|ui| {
        // Docks zero the item spacing; keep the fields apart.
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.label(RichText::new("#").font(theme::mono(12.0)).color(t.text_faint));
        changed = hex_edit(ui, id.with("hex"), color);
        let [mut r, mut g, mut b] = srgb_bytes(*color).map(f32::from);
        let mut rgb = false;
        for (label, v) in [("R", &mut r), ("G", &mut g), ("B", &mut b)] {
            ui.label(RichText::new(label).font(theme::mono(11.0)).color(t.text_faint));
            rgb |= widgets::value_field(ui, v, 0.0..=255.0, "", 38.0).changed();
        }
        if rgb {
            *color = [r / 255.0, g / 255.0, b / 255.0, 1.0];
            changed = true;
        }
    });
    changed
}

/// The hex field of [`color_readout`]: type a colour with or without the leading `#`. Valid input
/// sets `color` and returns `true`. While it has focus it shows exactly what is typed, so partial
/// or invalid input isn't overwritten by the colour; an invalid entry snaps back when focus leaves.
fn hex_edit(ui: &mut egui::Ui, id: egui::Id, color: &mut [f32; 4]) -> bool {
    let t = Tokens::get(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(vec2(64.0, 24.0), Sense::hover());
    widgets::surface(ui, rect, t.field, false);
    if !t.bevel {
        ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    }
    let [r, g, b] = srgb_bytes(*color);
    let typing = if ui.memory(|m| m.has_focus(id)) { ui.data(|d| d.get_temp::<String>(id)) } else { None };
    let mut text = typing.unwrap_or_else(|| format!("{r:02X}{g:02X}{b:02X}"));
    let field = rect.shrink2(vec2(5.0, 2.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(field).layout(egui::Layout::left_to_right(egui::Align::Center)));
    // Room for a pasted "#rrggbb"; the parse trims it, and the field shows plain digits otherwise.
    // `Frame::NONE`: the themed surface above is this field's frame.
    let resp =
        child.add(egui::TextEdit::singleline(&mut text).id(id).char_limit(7).desired_width(field.width()).frame(egui::Frame::NONE).font(theme::mono(12.0)));
    if resp.has_focus() {
        ui.data_mut(|d| d.insert_temp(id, text.clone()));
    } else {
        ui.data_mut(|d| d.remove::<String>(id));
    }
    if resp.changed()
        && let Some(c) = crate::color_picker_ui::parse_hex(&text)
    {
        *color = [c[0], c[1], c[2], 1.0];
        return true;
    }
    false
}

/// Photoshop's brush preset picker chip: a soft/hard round tip preview with the size underneath.
/// Draw a round brush tip preview (hard core fading to a soft edge).
fn brush_tip(p: &egui::Painter, c: egui::Pos2, rad: f32, hardness: f32, color: Color32) {
    let inner = rad * hardness.clamp(0.05, 1.0);
    let steps = 8;
    for k in (0..=steps).rev() {
        let f = k as f32 / steps as f32;
        let rr = inner + (rad - inner) * f;
        let a = if hardness >= 0.99 { 1.0 } else { (1.0 - f).powf(1.5) };
        p.circle_filled(c, rr, color.gamma_multiply(a));
    }
}

/// The Brush's or Pencil's Mode dropdown: the blend modes, with Photoshop's painting-only Clear
/// after Dissolve (#3439). Clear erases with the brush (`clear`); it shows while chosen, or while
/// `~` is held (`held`). The wheel over it steps the mode, as over the Layers panel's (#2061).
fn paint_mode_dropdown(ui: &mut egui::Ui, id: &str, b: &mut photocraft_engine::BrushSettings, clear: &mut bool, held: bool) {
    let mut opts: Vec<(Option<BlendMode>, &str)> = BlendMode::LAYER_MODES.iter().map(|m| (Some(*m), m.label())).collect();
    let at = opts.iter().position(|o| o.0 == Some(BlendMode::Dissolve)).map_or(opts.len(), |i| i + 1);
    opts.insert(at, (None, "Clear"));
    let mut mode = if *clear || held { None } else { Some(b.mode) };
    if !widgets::dropdown_wheel_hovered(ui, id, &mut mode, &opts, 96.0).0.is_empty() {
        *clear = mode.is_none();
        if let Some(m) = mode {
            b.mode = m;
        }
    }
}

/// Options-bar Smoothing % (the brush's stroke smoothing; the live stroke and the commit use it).
fn smoothing_field(ui: &mut egui::Ui, b: &mut photocraft_engine::BrushSettings, width: f32) {
    let mut sm = (b.smoothing.amount * 100.0).round();
    if widgets::value_field(ui, &mut sm, 0.0..=100.0, "%", width).changed() {
        b.smoothing.enabled = true;
        b.smoothing.amount = (sm / 100.0).clamp(0.0, 1.0);
    }
}

/// Options-bar gear beside Smoothing: Photoshop's smoothing options popup, edited on the brush
/// like the Smoothing % (so they stay per tool and reach the live stroke and the commit).
fn smoothing_options(ui: &mut egui::Ui, b: &mut photocraft_engine::BrushSettings) {
    let resp = icons::button(ui, "settings", 24.0, false, tl!("Set additional smoothing options"));
    let resp = crate::brush_picker::named(resp, "Set additional smoothing options");
    egui::Popup::from_toggle_button_response(&resp).close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside).show(|ui| {
        let s = &mut b.smoothing;
        widgets::checkbox(ui, &mut s.pulled_string, tl!("Pulled String Mode"));
        widgets::checkbox(ui, &mut s.catch_up, tl!("Stroke Catch-up"));
        widgets::checkbox(ui, &mut s.catch_up_on_end, tl!("Catch-up on Stroke End"));
        widgets::checkbox(ui, &mut s.adjust_for_zoom, tl!("Adjust for Zoom"));
    });
}

/// Options-bar brush chip: a click shows or hides Photoshop's Brush Preset picker below it (size,
/// hardness, the preset library), the one a right-click on the canvas opens.
fn brush_preset_chip(ui: &mut egui::Ui, b: &photocraft_engine::BrushSettings, state: &mut crate::state::UiState) {
    let t = Tokens::get(ui.ctx());
    let (r, _) = ui.allocate_exact_size(vec2(44.0, 30.0), Sense::hover());
    let resp = ui.interact(r, crate::brush_picker::chip_id(), Sense::click());
    if resp.hovered() || state.brush_picker.is_some() {
        ui.painter().rect_filled(r, t.radius_sm, t.hover);
    }
    let c = pos2(r.left() + 14.0, r.top() + 11.0);
    // Photoshop's white tip vanishes on light chrome; light themes use their icon colour.
    let tip = if t.dark() { Color32::WHITE } else { t.icon };
    brush_tip(ui.painter(), c, 7.0, b.hardness, tip);
    ui.painter().text(pos2(c.x, r.bottom() - 5.0), Align2::CENTER_CENTER, format!("{}", b.size.round() as i64), egui::FontId::proportional(9.5), t.text_dim);
    icons::paint(ui, Rect::from_center_size(pos2(r.right() - 9.0, c.y), vec2(10.0, 10.0)), "chevron-down", 9.0, t.text_faint);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, tl!("Brush Preset picker")));
    if resp.on_hover_text(tl!("Brush Preset picker")).clicked() {
        if state.brush_picker.is_some() {
            crate::brush_picker::close(state);
        } else {
            crate::brush_picker::open(state, ui.ctx(), [r.left(), r.bottom() + 2.0]);
        }
    }
}

/// A drag from within a multi-selection moves the whole selection, not only the grabbed
/// row. Dragging an unselected row keeps the existing single-layer behavior.
fn layer_drop_payload(dragged: u64, target: LayerId, position: &str, selected: &[LayerId]) -> Value {
    if selected.len() > 1 && selected.contains(&LayerId(dragged)) {
        json!({"layers": selected.iter().map(|id| id.0).collect::<Vec<_>>(), "target": target.0, "position": position})
    } else {
        json!({"layer": dragged, "target": target.0, "position": position})
    }
}

/// Would this drop move the Background off the bottom, or put a layer below it? Photoshop
/// refuses both (no drop indicator); ⌥-dragging the Background still copies it above.
fn background_drop_refused(doc: &photocraft_doc::Document, payload: &Value, target: LayerId, position: &str, copy: bool) -> bool {
    let Some(bg) = doc.layers.first().filter(|b| crate::doc_props_ui::is_background(doc, b)).map(|b| b.id) else { return false };
    let moves_bg =
        !copy && (payload["layer"].as_u64() == Some(bg.0) || payload["layers"].as_array().is_some_and(|a| a.iter().any(|v| v.as_u64() == Some(bg.0))));
    moves_bg || (target == bg && position == "below")
}

/// Drag a layer row to reorder: drop above, below, or inside an existing group. With ⌥ held on
/// release the layers stay put and copies land there instead (Photoshop).
/// Multi-layer moves are atomic (one undo step), using the engine's stable document order.
fn layer_drag_and_drop(
    app: &PhotocraftApp,
    ctx: &egui::Context,
    ui: &egui::Ui,
    l: &Layer,
    rect: Rect,
    resp: &egui::Response,
    actions: &mut Vec<(String, Value)>,
) {
    let t = Tokens::get(ctx);
    let key = egui::Id::new("layer-drag");
    // A drag from the mask thumbnail (`starts_mask_drag`) moves the mask, not the layer.
    let mask_drag = ctx.data(|d| d.get_temp::<MaskDrag>(mask_drag_key())).is_some_and(|(id, _)| id == l.id.0);
    if resp.drag_started() && !mask_drag {
        ctx.data_mut(|d| d.insert_temp(key, l.id.0));
    }
    let Some(dragged) = ctx.data(|d| d.get_temp::<u64>(key)) else { return };
    let pointer = ctx.input(|i| i.pointer.interact_pos());
    let released = ctx.input(|i| i.pointer.any_released());
    let copy = ctx.input(|i| i.modifiers.alt);
    if dragged == l.id.0 {
        // Ghost label following the pointer, and the copy cursor while ⌥ is held.
        if let Some(p) = pointer {
            crate::layer_transfer::ghost(ctx, p, &l.name);
        }
        if copy {
            ctx.set_cursor_icon(egui::CursorIcon::Copy);
        }
        // #1721: with ⌥ held the row also takes the drop onto itself — the only drop target a
        // single-layer document has. The copy lands above or below it (never into a group's own
        // centre). A multi-selection is left alone: the engine copies it beside each of its own
        // layers, not onto one of them.
        let selected = app.session.active().map(|st| st.selected_layers()).unwrap_or_default();
        let in_multi = selected.len() > 1 && selected.contains(&l.id);
        if copy
            && !in_multi
            && let Some(p) = pointer.filter(|p| rect.contains(*p))
        {
            let position = if (p.y - rect.top()) / rect.height() < 0.5 { "above" } else { "below" };
            let painter = ui.painter();
            if position == "above" {
                painter.line_segment([rect.left_top(), rect.right_top()], Stroke::new(2.0, t.accent));
            } else {
                painter.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(2.0, t.accent));
            }
            if released {
                let mut payload = layer_drop_payload(dragged, l.id, position, &selected);
                if let Some(o) = payload.as_object_mut() {
                    o.insert("copy".into(), json!(true));
                }
                actions.push(("layer.moveTo".into(), payload));
            }
        }
        return;
    }
    let Some(p) = pointer else { return };
    if !rect.contains(p) {
        return;
    }
    let f = (p.y - rect.top()) / rect.height();
    let position = if l.is_group() && (0.3..0.7).contains(&f) {
        "into"
    } else if f < 0.5 {
        "above"
    } else {
        "below"
    };
    let Some(st) = app.session.active() else { return };
    let mut payload = layer_drop_payload(dragged, l.id, position, &st.selected_layers());
    if background_drop_refused(&st.doc, &payload, l.id, position, copy) {
        return;
    }
    let painter = ui.painter();
    match position {
        "into" => {
            painter.rect_stroke(rect.shrink(1.0), t.radius_sm, Stroke::new(2.0, t.accent), StrokeKind::Inside);
        }
        "above" => {
            painter.line_segment([rect.left_top(), rect.right_top()], Stroke::new(2.0, t.accent));
        }
        _ => {
            painter.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(2.0, t.accent));
        }
    }
    if released {
        if copy && let Some(o) = payload.as_object_mut() {
            o.insert("copy".into(), json!(true));
        }
        actions.push(("layer.moveTo".into(), payload));
    }
}

/// #1721: the drop zones no row owns — the space between two rows and the empty space below the
/// bottom-most one. The drop lands on `target` at `position` (`above` or `below` its row): a copy
/// with ⌥ held on release (Photoshop), a move without. A multi-selection dropped on the row it
/// was grabbed from is left alone: the engine can only copy it beside each of its own layers.
fn layer_drop_zone(
    app: &PhotocraftApp,
    ctx: &egui::Context,
    ui: &egui::Ui,
    zone: Rect,
    target: (LayerId, Rect),
    position: &'static str,
    actions: &mut Vec<(String, Value)>,
) {
    let Some(dragged) = ctx.data(|d| d.get_temp::<u64>(egui::Id::new("layer-drag"))) else { return };
    let selected = app.session.active().map(|st| st.selected_layers()).unwrap_or_default();
    if target.0 == LayerId(dragged) && selected.len() > 1 && selected.contains(&target.0) {
        return;
    }
    let Some(p) = ctx.input(|i| i.pointer.interact_pos()) else { return };
    if !zone.contains(p) {
        return;
    }
    let t = Tokens::get(ctx);
    let line = if position == "above" { [target.1.left_top(), target.1.right_top()] } else { [target.1.left_bottom(), target.1.right_bottom()] };
    ui.painter().line_segment(line, Stroke::new(2.0, t.accent));
    if ctx.input(|i| i.pointer.any_released()) {
        let mut payload = layer_drop_payload(dragged, target.0, position, &selected);
        if ctx.input(|i| i.modifiers.alt)
            && let Some(o) = payload.as_object_mut()
        {
            o.insert("copy".into(), json!(true));
        }
        actions.push(("layer.moveTo".into(), payload));
    }
}

/// An fx row being dragged: the layer it belongs to and the effect index (`None` = the "Effects"
/// row, all of them).
type FxDrag = (u64, Option<usize>);

/// A mask thumbnail being dragged: the layer and which of its masks (#3113, #3119).
type MaskDrag = (u64, MaskKind);

fn mask_drag_key() -> egui::Id {
    egui::Id::new("mask-drag")
}

/// Which mask's thumbnail (where it was drawn last frame) the row's drag starts on, if any.
fn starts_mask_drag(ctx: &egui::Context, l: &Layer, resp: &egui::Response) -> Option<MaskKind> {
    if !resp.drag_started() {
        return None;
    }
    let p = ctx.input(|i| i.pointer.press_origin())?;
    let (thumbs, _) = crate::mask_thumbs_ui::recorded(ctx, l.id.0)?;
    let kind = thumbs.iter().find(|(_, r)| r.contains(p))?.0;
    let present = match kind {
        MaskKind::Pixel => l.mask.is_some(),
        MaskKind::Vector => l.vector_mask.is_some(),
    };
    present.then_some(kind)
}

fn fx_drag_key() -> egui::Id {
    egui::Id::new("fx-drag")
}

/// Start dragging the effects of `layer`: one (`effect`) or all of them.
fn start_fx_drag(ctx: &egui::Context, layer: LayerId, effect: Option<usize>) {
    ctx.data_mut(|d| d.insert_temp::<FxDrag>(fx_drag_key(), (layer.0, effect)));
}

/// While effects are dragged: their name by the pointer, and the copy cursor while ⌥ is held.
fn fx_drag_feedback(ctx: &egui::Context, doc: &photocraft_doc::Document) {
    let Some((from, effect)) = ctx.data(|d| d.get_temp::<FxDrag>(fx_drag_key())) else { return };
    let Some(p) = ctx.input(|i| i.pointer.interact_pos()) else { return };
    let label = match effect {
        Some(i) => doc.layer(LayerId(from)).and_then(|l| l.effects.items.get(i)).map_or("", |e| e.label()),
        None => "Effects",
    };
    crate::layer_transfer::ghost(ctx, p, label);
    if ctx.input(|i| i.modifiers.alt) {
        ctx.set_cursor_icon(egui::CursorIcon::Copy);
    }
}

/// The command for dropping effects of layer `from` on `target`: they move there, or with ⌥ held
/// on release are copied (Photoshop). `effect` is one effect row, or all effects when `None`.
fn fx_drop_action(from: u64, effect: Option<usize>, target: LayerId, copy: bool) -> (String, Value) {
    let mut payload = json!({"from": from, "to": target.0, "copy": copy});
    if let Some(o) = payload.as_object_mut()
        && let Some(i) = effect
    {
        o.insert("effect".into(), json!(i));
    }
    ("layer.layerStyle.transferEffects".into(), payload)
}

/// A layer row accepts the fx row being dragged: outlined while the pointer is over it, dropped
/// on release.
fn fx_drop(ctx: &egui::Context, ui: &egui::Ui, l: &Layer, rect: Rect, actions: &mut Vec<(String, Value)>) {
    let Some((from, effect)) = ctx.data(|d| d.get_temp::<FxDrag>(fx_drag_key())) else { return };
    let Some(p) = ctx.input(|i| i.pointer.interact_pos()) else { return };
    if from == l.id.0 || !rect.contains(p) {
        return;
    }
    let t = Tokens::get(ctx);
    ui.painter().rect_stroke(rect.shrink(1.0), t.radius_sm, Stroke::new(2.0, t.accent), StrokeKind::Inside);
    if ctx.input(|i| i.pointer.any_released()) {
        actions.push(fx_drop_action(from, effect, l.id, ctx.input(|i| i.modifiers.alt)));
    }
}

/// Photoshop shows a layer's effects as indented sub-rows ("Effects", then each effect). The eye
/// on "Effects" shows or hides them all, the eye on an effect's row just that one (#1622).
fn effect_rows(app: &mut PhotocraftApp, ui: &mut egui::Ui, l: &Layer, depth: usize, actions: &mut Vec<(String, Value)>) {
    let t = Tokens::get(ui.ctx());
    let indent = 30.0 + depth as f32 * 14.0 + 34.0;
    let mut rows: Vec<(String, bool, Option<&'static str>)> = vec![("Effects".into(), l.effects.enabled, None)];
    for e in &l.effects.items {
        let kind = crate::layer_style::KINDS.iter().find(|k| k.1 == e.label()).map(|k| k.0);
        rows.push((e.label().to_string(), e.enabled(), kind));
    }
    for (i, (name, on, kind)) in rows.into_iter().enumerate() {
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::click_and_drag());
        // Drag the row onto another layer to move its effects there, ⌥-drag to copy them.
        if resp.drag_started() {
            start_fx_drag(ui.ctx(), l.id, i.checked_sub(1));
        }
        if !ui.is_rect_visible(rect) {
            continue;
        }
        if resp.hovered() {
            ui.painter().rect_filled(rect, 0.0, t.hover.gamma_multiply(0.35));
        }
        if t.pro {
            ui.painter().line_segment([pos2(rect.left() + 30.0, rect.top()), pos2(rect.left() + 30.0, rect.bottom())], Stroke::new(1.0, t.separator));
        }
        // Nested under the layer's eye column so an effect does not read as its own layer (#4572).
        let eye = Rect::from_min_size(pos2(rect.left() + 36.0, rect.center().y - 9.0), vec2(18.0, 18.0));
        // A hidden effect's eye box is left empty (still clickable), like a hidden layer's.
        if on {
            icons::paint(ui, eye, "eye", 12.0, t.icon);
        }
        // The whole eye column of the row takes the click, so it never opens the Layer Style dialog.
        let eye_cell = Rect::from_min_max(pos2(rect.left() + 30.0, rect.top()), pos2((rect.left() + 58.0).min(rect.right()), rect.bottom()));
        if ui.interact(eye_cell, ui.id().with(("fx-eye", l.id.0, i)), Sense::click()).clicked() {
            let mut params = json!({"layer": l.id.0, "visible": !on});
            if let (Some(index), Some(o)) = (i.checked_sub(1), params.as_object_mut()) {
                o.insert("index".into(), json!(index));
            }
            actions.push(("layer.setEffectsVisible".into(), params));
        }
        let x = rect.left() + indent + if i == 0 { 8.0 } else { 24.0 };
        if i == 0 {
            icons::paint(ui, Rect::from_center_size(pos2(x - 12.0, rect.center().y), vec2(14.0, 14.0)), "sparkles", 11.0, t.text_dim);
        }
        let right = rect.right() - crate::layer_row_ui::RIGHT_PAD;
        crate::layer_row_ui::label(
            ui.painter(),
            x,
            rect.center().y,
            right,
            &name,
            egui::FontId::proportional(11.5),
            if on { t.text_dim } else { t.text_faint },
        );
        resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, true, name.clone()));
        if resp.double_clicked() {
            crate::layer_style::open_for_layer(app, l.id, kind);
        }
        // #2846: right-click opens the layer style menu (Create Layer, Copy Layer Style…), as in Photoshop.
        resp.context_menu(|ui| {
            crate::widgets::style_spectrum_popup_menu(ui);
            crate::layer_menu_ui::show_effects(app, ui, l, actions);
        });
    }
}

/// New / add / subtract / intersect selection buttons (shared by selection tools).
fn selection_mode_buttons(app: &mut PhotocraftApp, ui: &mut egui::Ui) {
    ui.spacing_mut().item_spacing.x = 2.0;
    for (i, (icon, tip)) in [
        ("square", tl!("New selection").to_string()),
        ("plus", crate::i18n::fmt(tl!("Add to selection  ({key})"), &[("key", &crate::shortcuts::pretty("Shift"))])),
        ("minus", crate::i18n::fmt(tl!("Subtract from selection  ({key})"), &[("key", &crate::shortcuts::pretty("Alt"))])),
        ("squares-subtract", crate::i18n::fmt(tl!("Intersect with selection  ({key})"), &[("key", &crate::shortcuts::pretty("Shift+Alt"))])),
    ]
    .iter()
    .enumerate()
    {
        if icons::button(ui, icon, 24.0, app.ui.selection_mode == i as u8, tip).clicked() {
            app.ui.selection_mode = i as u8;
        }
    }
    ui.spacing_mut().item_spacing.x = 8.0;
}

#[cfg(test)]
mod symmetry_theme_tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    #[test]
    fn brush_pencil_and_eraser_expose_symmetry_menu_in_every_theme() {
        for kind in crate::theme::ThemeKind::ALL {
            for tool in [Tool::Brush, Tool::Pencil, Tool::Eraser] {
                let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
                app.run("file.new", json!({"width": 80, "height": 80})).unwrap();
                app.ui.tool = tool;
                let mut h = Harness::builder().with_size(vec2(1600.0, 300.0)).build_ui_state(|ui, app| options_bar(app, ui), app);
                PhotocraftApp::setup_context(&h.ctx, kind);
                h.run_steps(4);
                assert!(h.query_by_label("Set painting symmetry options").is_some(), "missing symmetry menu in {:?} for {:?}", kind, tool);
            }
        }
    }
}

#[cfg(test)]
mod color_tests {
    use super::*;

    #[test]
    /// #3323: the Color panel's HSB is Photoshop's (and the Color Picker's), on sRGB values, not
    /// egui's linear-light HSV: a 50% grey has Brightness 50%, and the field's left edge halfway
    /// down is that grey (128), not 188.
    fn color_panel_hsb_is_srgb_like_photoshop() {
        let grey = srgb_hsva([128.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0, 1.0]);
        assert!((grey.v - 128.0 / 255.0).abs() < 1e-3 && grey.s < 1e-3, "{grey:?}");
        let mid = hsva_srgb(egui::ecolor::Hsva::new(0.3, 0.0, 0.5, 1.0));
        assert!(mid[..3].iter().all(|v| (v - 0.5).abs() <= 1.0 / 255.0), "{mid:?}");
        // What the field paints there.
        assert_eq!(hsva_color32(egui::ecolor::Hsva::new(0.3, 0.0, 0.5, 1.0)), Color32::from_gray(128));
        // The same numbers as the Color Picker dialog for a saturated colour.
        let c = [0.847, 0.271, 0.180];
        let (ours, dialog) = (srgb_hsva([c[0], c[1], c[2], 1.0]), crate::color_picker_ui::rgb_to_hsv(c));
        assert!((ours.h * 360.0 - dialog[0]).abs() < 0.5 && (ours.s - dialog[1]).abs() < 1e-3 && (ours.v - dialog[2]).abs() < 1e-3, "{ours:?} vs {dialog:?}");
    }

    #[test]
    fn srgb_hsva_roundtrip() {
        for c in [[0.847, 0.271, 0.180, 1.0], [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.2, 0.6, 0.4, 1.0]] {
            let back = hsva_srgb(srgb_hsva(c));
            for i in 0..3 {
                assert!((back[i] - c[i]).abs() <= 1.0 / 255.0, "{c:?} -> {back:?}");
            }
        }
    }

    /// Click the hex field, select its text and type `text` one key per frame.
    fn type_hex(h: &mut egui_kittest::Harness<'static, PhotocraftApp>, text: &str) {
        use egui_kittest::kittest::Queryable;
        h.get_by_role(egui::accesskit::Role::TextInput).click();
        h.run_steps(1);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        h.run_steps(1);
        for ch in text.chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
    }

    /// The Color panel's hex readout is editable, with or without the leading `#`.
    #[test]
    fn color_panel_hex_field_takes_a_typed_hex() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [1.0, 1.0, 1.0, 1.0];
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 300.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    color_picker(app, ui);
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        type_hex(&mut h, "003300");
        assert_eq!(srgb_bytes(h.state().session.tools.foreground), [0x00, 0x33, 0x00]);
        // A leading '#' is accepted too.
        type_hex(&mut h, "#ff8000");
        assert_eq!(srgb_bytes(h.state().session.tools.foreground), [0xff, 0x80, 0x00]);
    }

    /// Themes with `round_chips` (Studio) draw round chips, stacked or side by side in two tool
    /// columns, with Switch Colors right under them and Default Colors at the top-right corner
    /// (stacked) or the left edge (side by side); both buttons work in either layout.
    #[test]
    fn studio_color_chips_keep_default_and_swap_buttons_below() {
        use egui_kittest::kittest::Queryable;
        for (kind, two_columns) in [
            (crate::theme::ThemeKind::Studio, false),
            (crate::theme::ThemeKind::StudioLight, false),
            (crate::theme::ThemeKind::Studio, true),
            (crate::theme::ThemeKind::StudioLight, true),
        ] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.session.tools.foreground = [1.0, 0.0, 0.0, 1.0];
            app.session.tools.background = [0.0, 0.0, 1.0, 1.0];
            let mut h = egui_kittest::Harness::builder()
                .with_size(vec2(80.0, 120.0))
                .build_ui_state(|ui, app: &mut PhotocraftApp| color_chips(app, ui, two_columns), app);
            PhotocraftApp::setup_context(&h.ctx, kind);
            h.run_steps(2);
            h.get_by_label("Swap colours (X)").click();
            h.run_steps(2);
            assert_eq!(h.state().session.tools.foreground, [0.0, 0.0, 1.0, 1.0], "{kind:?}");
            assert_eq!(h.state().session.tools.background, [1.0, 0.0, 0.0, 1.0], "{kind:?}");
            h.get_by_label("Default colours (D)").click();
            h.run_steps(2);
            assert_eq!(h.state().session.tools.foreground, [0.0, 0.0, 0.0, 1.0], "{kind:?}");
            assert_eq!(h.state().session.tools.background, [1.0, 1.0, 1.0, 1.0], "{kind:?}");
        }
    }

    /// The Studio layout (chips stacked, icons below) fits the height the toolbar reserves for
    /// the chips, so the two-column check does not change with the theme.
    #[test]
    fn studio_color_chips_fit_the_reserved_height() {
        // Chips, then the row holding the Switch Colors curve.
        assert!(STUDIO_CHIP.0 + STUDIO_CHIP.1 + 8.0 + 4.0 <= chips_height(false, true, false));
        // Side by side: the chips plus the same row, and shorter than stacked.
        assert!(STUDIO_CHIP.0 + 8.0 + 4.0 <= chips_height(false, true, true));
        assert!(chips_height(false, true, true) < chips_height(false, true, false));
        // The Studio block is taller than Photoshop's, and the two-column check knows it.
        assert!(chips_height(false, true, false) > chips_height(false, false, false));
    }

    /// An incomplete entry never changes the colour and the field snaps back when focus leaves.
    #[test]
    fn color_panel_hex_field_ignores_partial_input() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.2, 0.4, 0.6, 1.0];
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 300.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    color_picker(app, ui);
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        let fg = h.state().session.tools.foreground;
        h.get_by_role(egui::accesskit::Role::TextInput).click();
        h.run_steps(1);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        h.run_steps(1);
        for ch in "12zz".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        assert_eq!(h.state().session.tools.foreground, fg, "invalid input changes nothing");
        h.key_press(egui::Key::Tab);
        h.run_steps(2);
        assert_eq!(h.get_by_role(egui::accesskit::Role::TextInput).value().as_deref(), Some("336699"), "snaps back to the colour");
    }

    /// The R, G and B fields next to the hex are editable, like the hex field.
    #[test]
    fn color_readout_rgb_fields_take_values() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.0, 0.0, 0.0, 1.0];
        let mut h = field_harness(app);
        // The only spin buttons are R, G and B, in that order.
        h.query_all_by_role(egui::accesskit::Role::SpinButton).nth(1).unwrap().click();
        h.run_steps(1);
        h.key_press_modifiers(egui::Modifiers::COMMAND, egui::Key::A);
        h.run_steps(1);
        for ch in "128".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        h.key_press(egui::Key::Tab);
        h.run_steps(2);
        assert_eq!(srgb_bytes(h.state().session.tools.foreground), [0x00, 0x80, 0x00]);
    }

    /// The pro Color panel's hex readout edits the foreground too.
    #[test]
    fn color_field_hex_field_takes_a_typed_hex() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [1.0, 1.0, 1.0, 1.0];
        let mut h = field_harness(app);
        type_hex(&mut h, "003300");
        assert_eq!(srgb_bytes(h.state().session.tools.foreground), [0x00, 0x33, 0x00]);
    }

    #[test]
    fn hue_and_saturation_fields_take_values_on_black() {
        use egui_kittest::kittest::Queryable;
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.0, 0.0, 0.0, 1.0];
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 300.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    hsb_sliders(app, ui);
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        // Hue, Saturation, Brightness, typed one key per frame. Black has no hue or saturation of its
        // own, so the first two used to reset to 0 and this ended on white.
        for (field, typed) in [(0, "120"), (1, "100"), (2, "100")] {
            h.query_all_by_role(egui::accesskit::Role::SpinButton).nth(field).unwrap().click();
            h.run_steps(1);
            for ch in typed.chars() {
                h.event(egui::Event::Text(ch.to_string()));
                h.run_steps(1);
            }
            h.key_press(egui::Key::Tab);
            h.run_steps(2);
        }
        assert_eq!(srgb_bytes(h.state().session.tools.foreground), [0, 255, 0]);
    }

    /// The Color field fills the dock group, never shrinks below 120 pt, and ignores a bad height.
    #[test]
    fn color_field_height_fills_the_group() {
        assert_eq!(color_field_height(None, 300.0), 120.0);
        assert_eq!(color_field_height(Some(500.0), 300.0), 468.0);
        assert_eq!(color_field_height(Some(500.0), 200.0), 442.0);
        assert_eq!(color_field_height(Some(60.0), 300.0), 120.0);
        assert_eq!(color_field_height(Some(f32::NAN), 300.0), 120.0);
        assert_eq!(color_field_height(Some(1.0e9), 300.0), 2000.0);
    }

    fn field_harness(app: PhotocraftApp) -> egui_kittest::Harness<'static, PhotocraftApp> {
        // 60 fps steps, so two clicks a frame apart are a double-click.
        let mut h = egui_kittest::Harness::builder().with_size(vec2(300.0, 200.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    color_field(app, ui);
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        h
    }

    fn click(h: &mut egui_kittest::Harness<'_, PhotocraftApp>, p: egui::Pos2) {
        h.hover_at(p);
        h.run_steps(1);
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
            h.run_steps(1);
        }
    }

    /// A click on the background chip makes the field edit the background, not the foreground.
    #[test]
    fn background_chip_retargets_the_color_field() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.2, 0.4, 0.6, 1.0];
        app.session.tools.background = [0.0, 0.0, 0.0, 1.0];
        let mut h = field_harness(app);
        let min = h.ctx.input(|i| i.viewport_rect()).min;
        // The background chip's corner that the foreground chip doesn't cover.
        click(&mut h, min + vec2(40.0, 40.0));
        // The field's top-left: no saturation, full brightness.
        click(&mut h, min + vec2(60.0, 10.0));
        assert!(h.state().ui.color_panel.background);
        let tools = &h.state().session.tools;
        assert_eq!(tools.foreground, [0.2, 0.4, 0.6, 1.0]);
        assert!(tools.background[..3].iter().all(|&v| v > 0.9), "{:?}", tools.background);
    }

    #[test]
    fn double_clicking_a_chip_opens_its_color_picker() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut h = field_harness(app);
        let min = h.ctx.input(|i| i.viewport_rect()).min;
        // The foreground point is where the chips overlap: the foreground is on top there.
        for (p, target) in [(vec2(40.0, 40.0), "background"), (vec2(30.0, 30.0), "foreground")] {
            h.state_mut().ui.dialogs.clear();
            // Past the last double-click, so this one isn't counted as a triple-click.
            h.run_steps(40);
            click(&mut h, min + p);
            assert!(h.state().ui.dialogs.is_empty(), "a single click only picks the chip");
            click(&mut h, min + p);
            let [d] = h.state().ui.dialogs.as_slice() else { panic!("one Color Picker") };
            assert_eq!(d.fields.get("__colorPicker").and_then(Value::as_str), Some(target));
        }
    }

    /// Press at `from`, move to `to` in six steps and release, as egui sees a drag.
    fn drag(h: &mut egui_kittest::Harness<'_, PhotocraftApp>, from: egui::Pos2, to: egui::Pos2) {
        h.hover_at(from);
        h.run_steps(1);
        h.drag_at(from);
        h.run_steps(1);
        for i in 1..=6 {
            h.hover_at(from + (to - from) * (i as f32 / 6.0));
            h.run_steps(1);
        }
        h.drop_at(to);
        h.run_steps(3);
    }

    /// Black carries neither hue nor saturation, a grey only the saturation it really has, and any
    /// other colour nothing at all: the knob keeps where it was dragged instead of round-tripping
    /// through RGB to the left edge (#3228).
    #[test]
    fn field_hsva_keeps_black_where_it_was_and_leaves_a_real_colour_alone() {
        // Read back from black, as every frame does: nothing of the drag survives the RGB trip.
        let black = field_hsva(egui::ecolor::Hsva::new(0.0, 0.0, 0.0, 1.0), (0.6, 1.0));
        assert_eq!((black.h, black.s), (0.6, 1.0), "the corner the drag ended in");
        // A grey: the saturation is genuinely zero — that is where the knob belongs — only the
        // hue was undefined.
        let grey = field_hsva(egui::ecolor::Hsva::new(0.0, 0.0, 0.5, 1.0), (0.6, 1.0));
        assert_eq!((grey.h, grey.s), (0.6, 0.0), "the grey stays on the left edge");
        // An ordinary colour is left as it is.
        let red = field_hsva(egui::ecolor::Hsva::new(0.1, 0.8, 0.9, 1.0), (0.6, 1.0));
        assert_eq!((red.h, red.s), (0.1, 0.8));
    }

    /// Dragging the selector into the bottom-right corner picks black there and leaves the knob
    /// in that corner: black has no saturation of its own, so it used to be read back as `s = 0`
    /// and drawn at the bottom-left instead, one frame after the drag that put it on the right
    /// (#3228).
    #[test]
    fn dragging_to_the_bottom_right_keeps_the_knob_there() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [1.0, 1.0, 1.0, 1.0];
        let mut h = field_harness(app);
        let min = h.ctx.input(|i| i.viewport_rect()).min;
        // Into the far bottom-right corner of the SV field: full saturation, no value. The end
        // point lies past the corner on purpose — a real drag goes there too, and the widget
        // clamps it to the field.
        drag(&mut h, min + vec2(200.0, 60.0), min + vec2(280.0, 170.0));
        assert_eq!(srgb_bytes(h.state().session.tools.foreground), [0, 0, 0], "both corners are black");
        // What the knob is drawn from on the very next frame: the field kept the saturation the
        // drag ended at (not the zero black reads back as), together with the hue.
        let remembered: (f32, f32) = h.ctx.data(|d| d.get_temp(egui::Id::new(("color-field-hue", false)))).expect("the field remembers what it showed");
        assert!(remembered.1 > 0.9, "the saturation the corner was dragged to: {remembered:?}");
        let shown = field_hsva(egui::ecolor::Hsva::new(0.0, 0.0, 0.0, 1.0), remembered);
        assert!(shown.s > 0.9, "the knob stays at the right edge, not the left: {remembered:?}");
        assert!(shown.v < 0.01, "still black: {shown:?}");
    }

    /// Black has no hue of its own, so each chip remembers the hue last picked for it.
    #[test]
    fn each_chip_keeps_its_own_hue_on_black() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.session.tools.foreground = [0.0, 0.0, 0.0, 1.0];
        app.session.tools.background = [0.0, 0.0, 0.0, 1.0];
        let mut h = field_harness(app);
        let min = h.ctx.input(|i| i.viewport_rect()).min;
        // The hue strip runs from red at the top through blue (a third down) and green (two thirds).
        let (strip_x, green, blue) = (285.0, 88.0, 48.0);
        click(&mut h, min + vec2(strip_x, green));
        click(&mut h, min + vec2(40.0, 40.0));
        click(&mut h, min + vec2(strip_x, blue));
        click(&mut h, min + vec2(16.0, 16.0));
        // Near the field's top-right: high saturation and brightness.
        click(&mut h, min + vec2(265.0, 10.0));
        let [r, g, b] = srgb_bytes(h.state().session.tools.foreground);
        assert!(g > 200 && r < 64 && b < 64, "the foreground's green, not the background's blue: {:?}", [r, g, b]);
    }
}

#[cfg(test)]
mod info_panel_sampler_tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    /// #1046: the Info panel lists each Color Sampler point under its number, with its position and
    /// the composite RGB/CMYK readout, one composite pixel each.
    #[test]
    fn info_panel_lists_the_sample_points() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 80, "height": 60, "background": "white"})).unwrap();
        app.run("view.colorSamplers.add", json!({"x": 5, "y": 6})).unwrap();
        app.run("view.colorSamplers.add", json!({"x": 40, "y": 30})).unwrap();
        let mut h = Harness::builder().with_size(vec2(340.0, 400.0)).build_ui_state(|ui, app: &mut PhotocraftApp| info_panel(app, ui), app);
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.run_steps(3);
        // Both points are listed, numbered, and read as white (RGB 255 / CMYK 0%).
        assert!(h.query_by_label("#1").is_some(), "point 1 is listed");
        assert!(h.query_by_label("#2").is_some(), "point 2 is listed");
        assert_eq!(h.query_all_by_label_contains("R:255 G:255 B:255").count(), 2, "each point shows its RGB");
        assert_eq!(h.query_all_by_label_contains("C:0% M:0% Y:0% K:0%").count(), 2, "each point shows its CMYK");
        // Clearing them takes the list away.
        h.state_mut().run("view.colorSamplers.clear", json!({})).unwrap();
        h.run_steps(2);
        assert!(h.query_by_label("#1").is_none(), "no points listed once cleared");
    }
}

#[cfg(test)]
mod history_transform_tests {
    use super::*;
    use egui_kittest::Harness;

    fn click(h: &mut Harness<'static, PhotocraftApp>, p: egui::Pos2) {
        h.hover_at(p);
        h.run_steps(1);
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
            h.run_steps(1);
        }
    }

    /// While Free Transform is open, clicking a History state leaves the document alone: the
    /// transform owns Undo. Without a transform the same click steps back as usual.
    #[test]
    fn history_rows_wait_for_an_open_transform() {
        for transforming in [true, false] {
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
            app.run("layer.new.layer", json!({})).unwrap();
            app.run("select.rect", json!({"x": 20, "y": 20, "width": 60, "height": 40})).unwrap();
            app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
            app.run("select.deselect", json!({})).unwrap();
            let mut h = Harness::builder().with_size(vec2(300.0, 400.0)).build_ui_state(|ui, app: &mut PhotocraftApp| history(app, ui), app);
            h.run_steps(2);
            if transforming {
                let ctx = h.ctx.clone();
                crate::transform_tool::begin(h.state_mut(), &ctx).unwrap();
            }
            let steps = h.state().session.active().unwrap().history.entries().len();
            // The first state's row, below the 38 pt snapshot row.
            let row = h.ctx.input(|i| i.viewport_rect()).min + vec2(60.0, 60.0);
            click(&mut h, row);
            let after = h.state().session.active().unwrap().history.entries().len();
            if transforming {
                assert_eq!(after, steps, "the document's history is untouched");
                assert!(h.state().ui.transform.is_some());
            } else {
                assert!(after < steps, "the click steps back: {steps} -> {after}");
            }
        }
    }

    /// The Photoshop-theme footer (#1117): trash deletes the current state, the camera takes a
    /// snapshot (#2966), and file-plus makes a new document from the current state.
    #[test]
    fn history_footer_buttons_run_their_commands() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        let mut h = Harness::builder().with_size(vec2(300.0, 400.0)).build_ui_state(|ui, app: &mut PhotocraftApp| history(app, ui), app);
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ProMedium);
        h.state_mut().ui.theme = crate::theme::ThemeKind::ProMedium;
        h.run_steps(3);
        // Footer buttons are the only 26 pt squares; the bar lays them out right to left.
        let buttons = |h: &Harness<'static, PhotocraftApp>| {
            let mut b: Vec<(Rect, bool)> = h.ctx.viewport(|v| {
                v.prev_pass.widgets.layers().flat_map(|(_, w)| w.iter()).filter(|w| w.rect.size() == vec2(26.0, 26.0)).map(|w| (w.rect, w.enabled)).collect()
            });
            b.sort_by(|a, b| b.0.center().x.total_cmp(&a.0.center().x));
            b
        };
        let b = buttons(&h);
        assert_eq!(b.len(), 3, "trash, snapshot, new document: {b:?}");
        assert!(b.iter().all(|b| b.1), "all enabled: {b:?}");
        let steps = h.state().session.active().unwrap().history.entries().len();
        click(&mut h, b[1].0.center());
        let st = h.state().session.active().unwrap();
        assert_eq!(st.snapshots.len(), 1, "the camera takes a snapshot");
        assert_eq!(st.history.entries().len(), steps, "which is not a history state");
        let docs = h.state().session.documents().len();
        click(&mut h, b[2].0.center());
        assert_eq!(h.state().session.documents().len(), docs + 1, "file-plus makes a new document");
        h.state_mut().session.set_active(0);
        h.run_steps(2);
        let steps = h.state().session.active().unwrap().history.entries().len();
        let b = buttons(&h);
        click(&mut h, b[0].0.center());
        let st = h.state().session.active().unwrap();
        assert_eq!(st.history.entries().len(), steps - 1, "trash deletes the current state");
        assert!(!st.history.can_redo(), "and it can't be redone");
    }
}

#[cfg(test)]
mod snapshot_row_tests {
    use egui::vec2;
    use egui_kittest::Harness;
    use egui_kittest::kittest::Queryable;
    use serde_json::json;

    use super::*;

    fn top_pixel(h: &Harness<'static, PhotocraftApp>) -> [f32; 4] {
        h.state().session.active().unwrap().doc.layers.last().unwrap().surface().unwrap().rgba(1, 1)
    }

    fn snapshot_names(h: &Harness<'static, PhotocraftApp>) -> Vec<String> {
        h.state().session.active().unwrap().snapshots.iter().map(|s| s.name.clone()).collect()
    }

    /// Snapshot rows (#2966) sit under the opening row: clicking one returns to it as a new
    /// state, and its right-click menu renames or deletes it.
    #[test]
    fn snapshot_rows_select_rename_and_delete() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
        app.run("layer.new.layer", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#ff0000"})).unwrap();
        app.run("history.newSnapshot", json!({})).unwrap();
        app.run("edit.fill", json!({"color": "#0000ff"})).unwrap();
        // 60 fps steps, so two clicks a frame apart count as a double-click.
        let mut h =
            Harness::builder().with_size(vec2(300.0, 400.0)).with_step_dt(1.0 / 60.0).build_ui_state(|ui, app: &mut PhotocraftApp| history(app, ui), app);
        h.run_steps(3);
        let steps = h.state().session.active().unwrap().history.entries().len();
        h.get_by_label("Snapshot 1").click();
        h.run_steps(2);
        let st = h.state().session.active().unwrap();
        assert_eq!(st.history.entries().len(), steps + 1);
        assert_eq!(st.history.entries().last().map(String::as_str), Some("Snapshot 1"));
        assert_eq!(top_pixel(&h), [1.0, 0.0, 0.0, 1.0], "back to red");
        // Right-click › Rename…: type a name, Enter. The new "Snapshot 1" state is listed under the
        // snapshot rows, so the first match is the snapshot's row.
        h.get_all_by_label("Snapshot 1").next().unwrap().click_secondary();
        h.run_steps(2);
        h.get_by_label("Rename…").click();
        h.run_steps(2);
        h.get_by_role(egui::accesskit::Role::TextInput).type_text("Lesson start");
        h.key_press(egui::Key::Enter);
        h.run_steps(2);
        assert_eq!(snapshot_names(&h), ["Lesson start"]);
        // A double-click renames too; Escape keeps the old name.
        let steps_now = h.state().session.active().unwrap().history.entries().len();
        // A real double-click (accessibility clicks are single clicks).
        let p = h.get_by_label("Lesson start").rect().center();
        let p = egui::pos2(p.x as f32, p.y as f32);
        h.event(egui::Event::PointerMoved(p));
        // Wait out the double-click window of the earlier clicks.
        h.run_steps(30);
        for pressed in [true, false, true, false] {
            h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
            h.step();
        }
        h.run_steps(3);
        h.get_by_role(egui::accesskit::Role::TextInput).type_text("Oops");
        h.key_press(egui::Key::Escape);
        h.run_steps(2);
        assert_eq!(snapshot_names(&h), ["Lesson start"], "Escape cancels");
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps_now + 1, "the click selected it once");
        // Right-click › Delete (the snapshot's row, above the state the click made).
        h.get_all_by_label("Lesson start").next().unwrap().click_secondary();
        h.run_steps(2);
        h.get_by_label("Delete").click();
        h.run_steps(2);
        assert!(snapshot_names(&h).is_empty());
        assert_eq!(h.state().session.active().unwrap().history.entries().len(), steps + 2, "deleting leaves the history alone");
    }

    /// Snapshot rows scroll with the states: many snapshots in a short dock leave the footer and
    /// its camera button in view.
    #[test]
    fn many_snapshots_keep_the_footer_in_a_short_panel() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
        for _ in 0..6 {
            app.run("history.newSnapshot", json!({})).unwrap();
        }
        let mut h = Harness::builder().with_size(vec2(300.0, 250.0)).build_ui_state(|ui, app: &mut PhotocraftApp| history(app, ui), app);
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::ProMedium);
        h.state_mut().ui.theme = crate::theme::ThemeKind::ProMedium;
        h.run_steps(3);
        let view = h.ctx.input(|i| i.viewport_rect());
        let buttons: Vec<Rect> = h
            .ctx
            .viewport(|v| v.prev_pass.widgets.layers().flat_map(|(_, w)| w.iter()).filter(|w| w.rect.size() == vec2(26.0, 26.0)).map(|w| w.rect).collect());
        assert_eq!(buttons.len(), 3, "{buttons:?}");
        assert!(buttons.iter().all(|b| view.contains_rect(*b)), "footer in view: {buttons:?} in {view:?}");
    }

    /// Renaming belongs to one document: snapshot ids restart in each, so switching tabs must not
    /// carry the edit over to the other document's snapshot with the same id.
    #[test]
    fn a_rename_in_progress_stays_with_its_document() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        for _ in 0..2 {
            app.run("file.new", json!({"width": 200, "height": 150})).unwrap();
            app.run("history.newSnapshot", json!({})).unwrap();
        }
        let mut h = Harness::builder().with_size(vec2(300.0, 400.0)).build_ui_state(|ui, app: &mut PhotocraftApp| history(app, ui), app);
        h.run_steps(3);
        h.get_by_label("Snapshot 1").click_secondary();
        h.run_steps(2);
        h.get_by_label("Rename…").click();
        h.run_steps(2);
        assert!(h.query_by_role(egui::accesskit::Role::TextInput).is_some(), "renaming in document 2");
        h.state_mut().session.set_active(0);
        h.run_steps(2);
        assert!(h.query_by_role(egui::accesskit::Role::TextInput).is_none(), "not in document 1");
    }
}

#[cfg(test)]
#[path = "layer_pct_slider_tests.rs"]
mod layer_pct_slider_tests;

#[cfg(test)]
#[path = "fx_drag_tests.rs"]
mod fx_drag_tests;

#[cfg(test)]
mod lock_tests {
    use super::*;
    use crate::canvas::{ToolEvent, tool_event};
    use egui::Modifiers;

    fn click_lock(app: &mut PhotocraftApp) {
        let st = app.session.active().unwrap();
        let l = st.active_layer.and_then(|id| st.doc.layer(id)).unwrap();
        let (cmd, p) = simple_lock_toggle(crate::doc_props_ui::is_background(&st.doc, l), l);
        app.run(&cmd, p).unwrap();
    }

    fn erase_line(app: &mut PhotocraftApp) {
        app.ui.tool = Tool::Eraser;
        let m = Modifiers::NONE;
        tool_event(app, ToolEvent::Down { x: 10.0, y: 40.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Move { x: 100.0, y: 40.0, pressure: 1.0 }, m);
        tool_event(app, ToolEvent::Up { x: 100.0, y: 40.0 }, m);
    }

    #[test]
    fn simple_lock_button_never_locks_the_eraser_out_of_the_background() {
        // #76: the simple themes' lock button used to toggle "lock all" on the Background, which
        // kept showing as locked and then refused every paint tool.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 120, "height": 80})).unwrap();
        app.run("tools.setColors", json!({"background": "#ff0000"})).unwrap();
        app.run("tools.setBrush", json!({"brush": {"size": 12, "hardness": 1.0}})).unwrap();
        // Clicking the Background's lock converts it to a normal layer (Photoshop).
        click_lock(&mut app);
        let l = &app.session.active().unwrap().doc.layers[0];
        assert_eq!(l.name, "Layer 0");
        assert!(!l.locks.any());
        erase_line(&mut app);
        assert!(!app.ui.status_error, "{}", app.ui.status);
        assert_eq!(app.session.active().unwrap().doc.layers[0].surface().unwrap().rgba(50, 40)[3], 0.0, "erased to transparency");
        // On a normal layer the button locks all, then unlocks everything.
        click_lock(&mut app);
        assert!(app.session.active().unwrap().doc.layers[0].locks.all);
        click_lock(&mut app);
        assert!(!app.session.active().unwrap().doc.layers[0].locks.any());
        // A layer with only transparency locked unlocks in one click.
        app.run("layer.setProps", json!({"locks": {"transparency": true}})).unwrap();
        click_lock(&mut app);
        assert!(!app.session.active().unwrap().doc.layers[0].locks.transparency);
    }
}

#[cfg(test)]
mod swatch_type_tests {
    use super::*;
    use egui_kittest::Harness;

    /// The HSB sliders only act on an edit. Their h/s/v round trip isn't exact for every colour
    /// (#D8452E isn't), and comparing values used to rewrite the foreground every frame, which
    /// would recolour selected type the moment it was selected.
    #[test]
    fn idle_hsb_sliders_leave_the_foreground_and_selected_type_alone() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 400, "height": 200})).unwrap();
        let id = app.run("type.create", json!({"text": "Hello world", "size": 40, "x": 20, "y": 100, "color": "#ffffff"})).unwrap()["layer"].as_u64().unwrap();
        app.run("tools.setColors", json!({"foreground": "#d8452e"})).unwrap();
        let fg = app.session.tools.foreground;
        app.ui.tool = Tool::Type;
        app.ui.text_edit = Some(crate::state::TextEdit {
            layer: id,
            caret: 0,
            anchor: 5,
            session: "s".into(),
            created: false,
            mask: false,
            dragging: false,
            resize: None,
            preedit: None,
        });
        let mut h = Harness::builder().with_size(vec2(300.0, 300.0)).build_ui_state(|ui, app: &mut PhotocraftApp| color_picker(app, ui), app);
        h.run_steps(4);
        assert_eq!(h.state().session.tools.foreground, fg);
        let st = h.state().session.active().unwrap();
        let Some(photocraft_doc::LayerContent::Text(t)) = st.doc.layer(photocraft_doc::LayerId(id)).map(|l| &l.content) else { panic!("type layer") };
        assert_eq!(t.char_runs().len(), 1, "still one white run");
    }
}

#[cfg(test)]
mod type_flyout_tests {
    use super::*;

    fn frame(app: &mut PhotocraftApp, ctx: &egui::Context, time: f64, events: Vec<egui::Event>) {
        let mut out = ctx.run_ui(
            egui::RawInput { time: Some(time), events, screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(1200.0, 1800.0))), ..Default::default() },
            |ui| toolbar(app, ui),
        );
        out.textures_delta.clear();
    }

    #[test]
    fn long_press_type_button_selects_type_then_vertical_from_the_flyout() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        assert_ne!(app.ui.tool, Tool::Type);
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        frame(&mut app, &ctx, 0.0, vec![]);
        frame(&mut app, &ctx, 0.1, vec![]);
        let index = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&Tool::Type)).unwrap();
        let bx = if Tokens::get(&ctx).pro { 30.0 } else { 36.0 };
        let mut buttons: Vec<Rect> = ctx.viewport(|v| {
            v.prev_pass
                .widgets
                .layers()
                .flat_map(|(_, w)| w.iter())
                .filter(|w| w.rect.size() == egui::Vec2::splat(bx) && w.sense.senses_click())
                .map(|w| w.rect)
                .collect()
        });
        buttons.sort_by(|a, b| a.top().total_cmp(&b.top()));
        let at = buttons[index].center();
        let pointer = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        frame(&mut app, &ctx, 1.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
        frame(&mut app, &ctx, 1.36, vec![]);
        assert_eq!(ctx.data(|d| d.get_temp::<(egui::Id, Rect)>(egui::Id::new("tool-flyout"))).map(|(id, _)| id), Some(egui::Id::new(("tool-slot", index))));
        frame(&mut app, &ctx, 1.4, vec![pointer(at, false)]);
        frame(&mut app, &ctx, 1.45, vec![]);
        // The held button's tool is selected, as Photoshop does (#4453).
        assert_eq!(app.ui.tool, Tool::Type);
        let key = egui::Id::new(("tool-slot", index));
        let menu = ctx.memory(|m| m.area_rect(key.with("flyout"))).unwrap();
        let row = egui::pos2(menu.left() + 65.0, menu.top() + 39.0 + 8.0);
        frame(&mut app, &ctx, 2.0, vec![egui::Event::PointerMoved(row), pointer(row, true)]);
        frame(&mut app, &ctx, 2.05, vec![pointer(row, false)]);
        assert_eq!(app.ui.tool, Tool::VerticalType);
        assert!(ctx.data(|d| d.get_temp::<(egui::Id, Rect)>(egui::Id::new("tool-flyout"))).is_none());
    }

    /// #4444: click and hold on Change Screen Mode opens its list like the tool flyouts, without
    /// cycling the mode on release; a plain click still cycles, and right-click still opens it.
    #[test]
    fn long_press_screen_mode_opens_the_modes_without_cycling() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        frame(&mut app, &ctx, 0.0, vec![]);
        frame(&mut app, &ctx, 0.1, vec![]);
        let bx = if Tokens::get(&ctx).pro { 30.0 } else { 36.0 };
        // The toolbar's last button: Change Screen Mode, after Quick Mask.
        let at = ctx
            .viewport(|v| {
                v.prev_pass
                    .widgets
                    .layers()
                    .flat_map(|(_, w)| w.iter())
                    .filter(|w| w.rect.size() == egui::Vec2::splat(bx) && w.sense.senses_click())
                    .map(|w| w.rect)
                    .max_by(|a, b| a.top().total_cmp(&b.top()).then(a.left().total_cmp(&b.left())))
            })
            .unwrap()
            .center();
        let standard = |app: &PhotocraftApp| crate::view_cmds::checked(app, "view.screenMode.standard") == Some(true);
        let menu_open = |ctx: &egui::Context| egui::Popup::is_any_open(ctx);
        let pointer = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        assert!(standard(&app));
        // Hold, then release: the list opens and stays open, and the mode is unchanged.
        frame(&mut app, &ctx, 1.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
        frame(&mut app, &ctx, 1.2, vec![]);
        assert!(!menu_open(&ctx), "not before the hold time");
        frame(&mut app, &ctx, 1.4, vec![]);
        assert!(menu_open(&ctx), "open while still held");
        frame(&mut app, &ctx, 1.5, vec![pointer(at, false)]);
        frame(&mut app, &ctx, 1.55, vec![]);
        assert!(menu_open(&ctx), "the release keeps it open");
        assert!(standard(&app), "the release doesn't cycle the mode");
        // Escape closes it; a plain click cycles the mode and opens nothing.
        frame(
            &mut app,
            &ctx,
            2.0,
            vec![egui::Event::Key { key: egui::Key::Escape, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() }],
        );
        frame(&mut app, &ctx, 2.05, vec![]);
        assert!(!menu_open(&ctx));
        frame(&mut app, &ctx, 3.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
        frame(&mut app, &ctx, 3.1, vec![pointer(at, false)]);
        frame(&mut app, &ctx, 3.15, vec![]);
        assert!(!standard(&app), "a click cycles");
        assert!(!menu_open(&ctx));
        // Right-click opens it too.
        let secondary = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Secondary, pressed, modifiers: Default::default() };
        frame(&mut app, &ctx, 4.0, vec![secondary(true)]);
        frame(&mut app, &ctx, 4.05, vec![secondary(false)]);
        frame(&mut app, &ctx, 4.1, vec![]);
        assert!(menu_open(&ctx));
    }

    /// Every text drawn in one toolbar frame.
    fn frame_texts(app: &mut PhotocraftApp, ctx: &egui::Context, time: f64, events: Vec<egui::Event>) -> Vec<String> {
        fn collect(shape: &egui::Shape, out: &mut Vec<String>) {
            match shape {
                egui::Shape::Text(t) => out.push(t.galley.text().to_string()),
                egui::Shape::Vec(v) => v.iter().for_each(|s| collect(s, out)),
                _ => {}
            }
        }
        let mut out = ctx.run_ui(
            egui::RawInput { time: Some(time), events, screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(1200.0, 1800.0))), ..Default::default() },
            |ui| toolbar(app, ui),
        );
        out.textures_delta.clear();
        let mut texts = Vec::new();
        out.shapes.iter().for_each(|c| collect(&c.shape, &mut texts));
        texts
    }

    /// #4450: no tool tooltip while a flyout is open or a tool button is held to open one, for the
    /// held tool or any other; a plain hover still shows its tooltip.
    #[test]
    fn no_tooltips_over_an_open_tool_flyout() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        frame(&mut app, &ctx, 0.0, vec![]);
        frame(&mut app, &ctx, 0.1, vec![]);
        let bx = if Tokens::get(&ctx).pro { 30.0 } else { 36.0 };
        let mut buttons: Vec<Rect> = ctx.viewport(|v| {
            v.prev_pass
                .widgets
                .layers()
                .flat_map(|(_, w)| w.iter())
                .filter(|w| w.rect.size() == egui::Vec2::splat(bx) && w.sense.senses_click())
                .map(|w| w.rect)
                .collect()
        });
        buttons.sort_by(|a, b| a.top().total_cmp(&b.top()));
        let slot_of = |tool: Tool| TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&tool)).unwrap();
        let (eyedropper, brush) = (buttons[slot_of(Tool::Eyedropper)].center(), buttons[slot_of(Tool::Brush)].center());
        let tip = |tool: Tool| format!("{}  ({})", tl!(tool.label()), tool.key());
        let pointer = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        // A plain hover shows the tooltip.
        frame(&mut app, &ctx, 1.0, vec![egui::Event::PointerMoved(brush)]);
        let mut texts = Vec::new();
        for k in 1..=10 {
            texts = frame_texts(&mut app, &ctx, 1.0 + 0.1 * f64::from(k), vec![]);
        }
        assert!(texts.contains(&tip(Tool::Brush)), "hovered long enough: {texts:?}");
        // Held on the Eyedropper: no tooltip before or after its flyout opens.
        frame(&mut app, &ctx, 3.0, vec![egui::Event::PointerMoved(eyedropper), pointer(eyedropper, true)]);
        for k in 1..=10 {
            let texts = frame_texts(&mut app, &ctx, 3.0 + 0.1 * f64::from(k), vec![]);
            assert!(!texts.contains(&tip(Tool::Eyedropper)), "held {k}: {texts:?}");
        }
        assert!(ctx.data(|d| d.get_temp::<(egui::Id, Rect)>(egui::Id::new("tool-flyout"))).is_some(), "the flyout is open");
        // Released, with the flyout still open: none for the held tool or another one.
        frame(&mut app, &ctx, 4.1, vec![pointer(eyedropper, false)]);
        for (start, at, tool) in [(4.5, eyedropper, Tool::Eyedropper), (6.0, brush, Tool::Brush)] {
            frame(&mut app, &ctx, start, vec![egui::Event::PointerMoved(at)]);
            for k in 1..=10 {
                let texts = frame_texts(&mut app, &ctx, start + 0.1 * f64::from(k), vec![]);
                assert!(!texts.contains(&tip(tool)), "{tool:?} under the flyout: {texts:?}");
            }
        }
    }

    /// #4460: a press outside an open flyout closes it, even when it becomes a drag (cropping or
    /// painting on the canvas) rather than a click; a press on the flyout keeps it for its pick.
    /// #4932: so does a click on its own tool button.
    #[test]
    fn a_drag_outside_closes_the_tool_flyout() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        frame(&mut app, &ctx, 0.0, vec![]);
        frame(&mut app, &ctx, 0.1, vec![]);
        let index = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&Tool::Brush)).unwrap();
        let bx = if Tokens::get(&ctx).pro { 30.0 } else { 36.0 };
        let mut buttons: Vec<Rect> = ctx.viewport(|v| {
            v.prev_pass
                .widgets
                .layers()
                .flat_map(|(_, w)| w.iter())
                .filter(|w| w.rect.size() == egui::Vec2::splat(bx) && w.sense.senses_click())
                .map(|w| w.rect)
                .collect()
        });
        buttons.sort_by(|a, b| a.top().total_cmp(&b.top()));
        let at = buttons[index].center();
        let flyout = |ctx: &egui::Context| ctx.data(|d| d.get_temp::<(egui::Id, Rect)>(egui::Id::new("tool-flyout"))).is_some();
        let secondary = |pressed| egui::Event::PointerButton { pos: at, button: egui::PointerButton::Secondary, pressed, modifiers: Default::default() };
        let primary = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        // A right-click on the button of an open flyout closes it (#4932), so open only a closed one.
        let open = |app: &mut PhotocraftApp, t: f64| {
            if !flyout(&ctx) {
                frame(app, &ctx, t, vec![egui::Event::PointerMoved(at), secondary(true)]);
                frame(app, &ctx, t + 0.05, vec![secondary(false)]);
                frame(app, &ctx, t + 0.1, vec![]);
            }
            assert!(flyout(&ctx), "right-click opens it");
        };
        // A press on the flyout itself keeps it open (the row picks on release).
        open(&mut app, 1.0);
        let menu = ctx.memory(|m| m.area_rect(egui::Id::new(("tool-slot", index)).with("flyout"))).unwrap();
        frame(&mut app, &ctx, 1.5, vec![egui::Event::PointerMoved(menu.center()), primary(menu.center(), true)]);
        assert!(flyout(&ctx), "a press on the flyout keeps it");
        frame(&mut app, &ctx, 1.55, vec![primary(menu.center(), false)]);
        // A drag far from the toolbar (the canvas) closes it as soon as it starts.
        open(&mut app, 2.0);
        let canvas = egui::pos2(900.0, 900.0);
        frame(&mut app, &ctx, 2.5, vec![egui::Event::PointerMoved(canvas), primary(canvas, true)]);
        frame(&mut app, &ctx, 2.55, vec![egui::Event::PointerMoved(canvas + vec2(80.0, 60.0))]);
        assert!(!flyout(&ctx), "a press outside closes it, even one that becomes a drag");
        frame(&mut app, &ctx, 2.6, vec![primary(canvas + vec2(80.0, 60.0), false)]);
        assert!(!flyout(&ctx));
        // #4932: clicking its own tool button again closes it, with either button.
        open(&mut app, 3.0);
        frame(&mut app, &ctx, 3.5, vec![egui::Event::PointerMoved(at), secondary(true)]);
        frame(&mut app, &ctx, 3.55, vec![secondary(false)]);
        frame(&mut app, &ctx, 3.6, vec![]);
        assert!(!flyout(&ctx), "a second right-click on the button closes it");
        open(&mut app, 4.0);
        app.ui.tool = Tool::Move;
        frame(&mut app, &ctx, 4.5, vec![egui::Event::PointerMoved(at), primary(at, true)]);
        frame(&mut app, &ctx, 4.55, vec![primary(at, false)]);
        frame(&mut app, &ctx, 4.6, vec![]);
        assert!(!flyout(&ctx), "a click on the button closes it");
        assert_eq!(app.ui.tool, Tool::Brush, "and picks the tool, as any click on it does");
        // The release that ends a long press keeps the flyout it opened.
        frame(&mut app, &ctx, 5.0, vec![egui::Event::PointerMoved(at), primary(at, true)]);
        frame(&mut app, &ctx, 5.2, vec![]);
        frame(&mut app, &ctx, 5.5, vec![]);
        assert!(flyout(&ctx), "a long press opens it");
        frame(&mut app, &ctx, 5.55, vec![primary(at, false)]);
        frame(&mut app, &ctx, 5.6, vec![]);
        assert!(flyout(&ctx), "releasing the long press keeps it open");
    }

    #[test]
    fn red_eye_is_in_the_j_flyout() {
        let j = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).find(|slot| slot.contains(&Tool::SpotHealing)).expect("J group");
        assert!(j.contains(&Tool::RedEye), "{j:?}");
        assert_eq!(j.last(), Some(&Tool::RedEye));
        // Remove leads the flyout, so the slot shows it until another J tool is picked.
        assert_eq!(j.first(), Some(&Tool::Remove));
    }

    #[test]
    fn long_press_clone_stamp_selects_pattern_stamp() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        frame(&mut app, &ctx, 0.0, vec![]);
        frame(&mut app, &ctx, 0.1, vec![]);
        let index = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&Tool::CloneStamp)).unwrap();
        let slot = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).nth(index).copied().unwrap();
        assert_eq!(slot, &[Tool::CloneStamp, Tool::PatternStamp]);
        let bx = if Tokens::get(&ctx).pro { 30.0 } else { 36.0 };
        let mut buttons: Vec<Rect> = ctx.viewport(|v| {
            v.prev_pass
                .widgets
                .layers()
                .flat_map(|(_, w)| w.iter())
                .filter(|w| w.rect.size() == egui::Vec2::splat(bx) && w.sense.senses_click())
                .map(|w| w.rect)
                .collect()
        });
        buttons.sort_by(|a, b| a.top().total_cmp(&b.top()));
        let at = buttons[index].center();
        let pointer = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        frame(&mut app, &ctx, 1.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
        frame(&mut app, &ctx, 1.36, vec![]);
        assert_eq!(ctx.data(|d| d.get_temp::<(egui::Id, Rect)>(egui::Id::new("tool-flyout"))).map(|(id, _)| id), Some(egui::Id::new(("tool-slot", index))));
        frame(&mut app, &ctx, 1.4, vec![pointer(at, false)]);
        frame(&mut app, &ctx, 1.45, vec![]);
        let key = egui::Id::new(("tool-slot", index));
        let menu = ctx.memory(|m| m.area_rect(key.with("flyout"))).unwrap();
        let row = egui::pos2(menu.left() + 65.0, menu.top() + 39.0 + 8.0);
        frame(&mut app, &ctx, 2.0, vec![egui::Event::PointerMoved(row), pointer(row, true)]);
        frame(&mut app, &ctx, 2.05, vec![pointer(row, false)]);
        assert_eq!(app.ui.tool, Tool::PatternStamp);
    }

    #[test]
    fn long_press_brush_selects_color_replacement() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        frame(&mut app, &ctx, 0.0, vec![]);
        frame(&mut app, &ctx, 0.1, vec![]);
        let index = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&Tool::Brush)).unwrap();
        let slot = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).nth(index).copied().unwrap();
        // Photoshop's Brush group, Color Replacement after the Mixer Brush.
        assert_eq!(slot, &[Tool::Brush, Tool::Pencil, Tool::MixerBrush, Tool::ColorReplacement]);
        let bx = if Tokens::get(&ctx).pro { 30.0 } else { 36.0 };
        let mut buttons: Vec<Rect> = ctx.viewport(|v| {
            v.prev_pass
                .widgets
                .layers()
                .flat_map(|(_, w)| w.iter())
                .filter(|w| w.rect.size() == egui::Vec2::splat(bx) && w.sense.senses_click())
                .map(|w| w.rect)
                .collect()
        });
        buttons.sort_by(|a, b| a.top().total_cmp(&b.top()));
        let at = buttons[index].center();
        let pointer = |pos, pressed| egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed, modifiers: Default::default() };
        frame(&mut app, &ctx, 1.0, vec![egui::Event::PointerMoved(at), pointer(at, true)]);
        frame(&mut app, &ctx, 1.36, vec![]);
        frame(&mut app, &ctx, 1.4, vec![pointer(at, false)]);
        frame(&mut app, &ctx, 1.45, vec![]);
        let key = egui::Id::new(("tool-slot", index));
        let menu = ctx.memory(|m| m.area_rect(key.with("flyout"))).unwrap();
        // The last of the four 26-point rows.
        let row = egui::pos2(menu.left() + 65.0, menu.bottom() - 20.0);
        frame(&mut app, &ctx, 2.0, vec![egui::Event::PointerMoved(row), pointer(row, true)]);
        frame(&mut app, &ctx, 2.05, vec![pointer(row, false)]);
        assert_eq!(app.ui.tool, Tool::ColorReplacement);
        assert_eq!(crate::icons::tool_icon(Tool::ColorReplacement), "brush-replace");
    }

    #[test]
    fn rotate_view_is_in_the_hand_flyout() {
        let hand = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).find(|slot| slot.contains(&Tool::Hand)).expect("Hand group");
        assert_eq!(*hand, [Tool::Hand, Tool::RotateView]);
        assert_eq!(Tool::RotateView.key(), 'R');
    }

    #[test]
    fn pro_toolbar_glyphs_fill_two_thirds_of_the_button() {
        // #2665: Photoshop-style themes draw tool glyphs larger than other icon buttons.
        assert_eq!(toolbar_icon_size(30.0, true), 20.0);
        assert_eq!(toolbar_icon_size(36.0, false), (36.0f32 * 0.52).round(), "other themes keep the icon-button size");
        // The glyph stays inside the button with room for the flyout triangle's corner.
        assert!(toolbar_icon_size(30.0, true) <= 30.0 - 2.0 * 4.0);
    }
}

#[cfg(test)]
mod toolbar_hidden_tests {
    use super::*;

    /// Clickable tool-sized buttons the toolbar drew in its last pass.
    fn tool_buttons(app: &mut PhotocraftApp) -> usize {
        let ctx = egui::Context::default();
        PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::ALL[0]);
        for time in [0.0, 0.1] {
            let mut out = ctx.run_ui(
                egui::RawInput { time: Some(time), screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(1200.0, 1800.0))), ..Default::default() },
                |ui| toolbar(app, ui),
            );
            out.textures_delta.clear();
        }
        let size = egui::Vec2::splat(if Tokens::get(&ctx).pro { 30.0 } else { 36.0 });
        ctx.viewport(|v| v.prev_pass.widgets.layers().flat_map(|(_, w)| w.iter()).filter(|w| w.rect.size() == size && w.sense.senses_click()).count())
    }

    fn hide(app: &mut PhotocraftApp, tools: &[Tool]) {
        let names: Vec<String> = tools.iter().map(|tool| format!("{tool:?}")).collect();
        app.session.edit_prefs(|p| p.toolbar.hidden = names);
    }

    #[test]
    fn hidden_tools_leave_the_toolbar() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let all = tool_buttons(&mut app);
        let slots: usize = TOOL_SECTIONS.iter().map(|section| section.len()).sum();
        hide(&mut app, &Tool::ALL);
        assert_eq!(tool_buttons(&mut app), all - slots, "every tool slot is gone when every tool is hidden");
        hide(&mut app, &[Tool::Brush, Tool::Pencil, Tool::MixerBrush, Tool::ColorReplacement]);
        assert_eq!(tool_buttons(&mut app), all - 1, "a slot whose tools are all hidden is gone");
        hide(&mut app, &[Tool::Sponge]);
        assert_eq!(tool_buttons(&mut app), all, "a slot with a visible tool stays");
    }

    #[test]
    fn a_slot_keeps_only_its_visible_tools_and_its_index() {
        let sections = visible_sections(&["Sponge".to_string(), "Move".to_string(), "Artboard".to_string()], false);
        assert_eq!(sections.len(), TOOL_SECTIONS.len() - 1, "the Move section has no tools left");
        let index = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&Tool::Sponge)).unwrap();
        let slot = sections.iter().flatten().find(|(i, _)| *i == index).map(|(_, tools)| tools.clone());
        assert_eq!(slot, Some(vec![Tool::Dodge, Tool::Burn]));
        assert!(sections.iter().flatten().all(|(_, tools)| !tools.contains(&Tool::Move) && !tools.contains(&Tool::Sponge)));
        assert_eq!(visible_sections(&[], false).iter().map(Vec::len).sum::<usize>(), TOOL_SECTIONS.iter().map(|section| section.len()).sum::<usize>());
        let by_command = visible_sections(&["sponge".to_string(), "move".to_string(), "artboard".to_string()], false);
        assert_eq!(by_command, sections, "edit.toolbar names are read like `ui.set {{tool}}` names");
    }

    #[test]
    fn embedded_toolbar_only_offers_page_tools() {
        let sections = visible_sections(&[], true);
        assert!(sections.iter().flatten().all(|(_, tools)| tools.iter().copied().all(crate::tool_shortcuts::page_tool)));
        for tool in [Tool::Move, Tool::RectMarquee, Tool::Brush, Tool::CloneStamp, Tool::Type, Tool::Hand, Tool::Zoom] {
            assert!(sections.iter().flatten().any(|(_, tools)| tools.contains(&tool)));
        }
    }
}

#[cfg(test)]
mod properties_card_tests {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    /// Arithmetic left uncommitted in one fill layer's card never lands on the layer selected next.
    #[test]
    fn uncommitted_arithmetic_stays_with_its_layer() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        let mut ids = Vec::new();
        for color in ["#00ff00", "#0000ff"] {
            s.execute("layer.newFillLayer.solidColor", json!({"color": color})).unwrap();
            ids.push(s.active().unwrap().active_layer.unwrap());
        }
        // The layer selected next already shows 25, the value the edited one has when `25*` stops it.
        s.execute("layer.setProps", json!({"layer": ids[0].0, "opacity": 0.25})).unwrap();
        let mut app = PhotocraftApp::new(s, crate::Services::default());
        app.ui.panels.properties = true;
        let mut h = Harness::builder().with_size(vec2(800.0, 600.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("semibold".into()))) {
                    properties_window(app, ui.ctx());
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        h.query_all_by_role(egui::accesskit::Role::SpinButton).next().unwrap().click();
        h.run_steps(1);
        for ch in "25*2".chars() {
            h.event(egui::Event::Text(ch.to_string()));
            h.run_steps(1);
        }
        // The Layers panel selects the other layer in the frame the click leaves the field.
        h.state_mut().session.execute("layer.select", json!({"layer": ids[0].0})).unwrap();
        h.ctx.memory_mut(|m| m.stop_text_input());
        h.run_steps(3);
        let doc = &h.state().session.active().unwrap().doc;
        assert_eq!(ids.iter().map(|&id| doc.layer(id).unwrap().opacity).collect::<Vec<_>>(), [0.25, 0.25]);
    }
}

#[cfg(test)]
mod toolbar_tests {
    use super::*;
    use egui_kittest::kittest::Queryable;

    /// #1197: the header chevron switches the toolbar between one and two columns.
    #[test]
    fn header_chevron_toggles_two_columns() {
        let app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        let mut h = egui_kittest::Harness::builder().with_size(vec2(800.0, 1400.0)).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    toolbar(app, ui);
                    let left = ui.available_rect_before_wrap().left();
                    ui.data_mut(|d| d.insert_temp(egui::Id::new("toolbar-test-left"), left));
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Pro);
        h.run_steps(2);
        let left = |h: &egui_kittest::Harness<'_, PhotocraftApp>| h.ctx.data(|d| d.get_temp::<f32>(egui::Id::new("toolbar-test-left"))).unwrap_or(0.0);
        let single = left(&h);
        assert!(!h.state().ui.panels.toolbar_double);

        h.get_by_label("Toolbar").click();
        h.run_steps(2);
        assert!(h.state().ui.panels.toolbar_double);
        assert!(left(&h) > single + 20.0, "the toolbar did not widen: {single} -> {}", left(&h));

        h.get_by_label("Toolbar").click();
        h.run_steps(2);
        assert!(!h.state().ui.panels.toolbar_double);
        assert_eq!(left(&h), single);
    }

    fn click(h: &mut egui_kittest::Harness<'_, PhotocraftApp>, p: egui::Pos2) {
        h.hover_at(p);
        h.run_steps(1);
        for pressed in [true, false] {
            h.event(egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
            h.run_steps(1);
        }
    }

    /// A tool's toolbar button: tool buttons have no label, so find it by slot order.
    fn tool_button(h: &egui_kittest::Harness<'_, PhotocraftApp>, tool: Tool) -> egui::Pos2 {
        let size = egui::Vec2::splat(if Tokens::get(&h.ctx).pro { 30.0 } else { 36.0 });
        let buttons: Vec<Rect> = h.ctx.viewport(|v| {
            v.prev_pass.widgets.layers().flat_map(|(_, w)| w.iter()).filter(|w| w.rect.size() == size && w.sense.senses_click()).map(|w| w.rect).collect()
        });
        let index = TOOL_SECTIONS.iter().flat_map(|section| section.iter()).position(|slot| slot.contains(&tool)).unwrap();
        buttons[index].center()
    }

    fn hand_button(h: &egui_kittest::Harness<'_, PhotocraftApp>) -> egui::Pos2 {
        tool_button(h, Tool::Hand)
    }

    fn toolbar_harness(app: PhotocraftApp) -> egui_kittest::Harness<'static, PhotocraftApp> {
        // 60 fps steps, so two clicks a few frames apart are a double-click.
        let mut h = egui_kittest::Harness::builder().with_size(vec2(800.0, 1400.0)).with_step_dt(1.0 / 60.0).build_ui_state(
            |ui, app: &mut PhotocraftApp| {
                if ui.ctx().fonts(|f| f.families().contains(&egui::FontFamily::Name("medium".into()))) {
                    toolbar(app, ui);
                }
            },
            app,
        );
        PhotocraftApp::setup_context(&h.ctx, crate::theme::ThemeKind::Studio);
        h.run_steps(2);
        h
    }

    /// Double-clicking the Hand tool fits the image on screen (Photoshop); a single click only
    /// picks the tool.
    #[test]
    fn double_clicking_the_hand_tool_fits_on_screen() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.ui.tool = Tool::Move;
        app.ui.views[0].fit_pending = false;
        let mut h = toolbar_harness(app);
        let p = hand_button(&h);
        click(&mut h, p);
        assert_eq!(h.state().ui.tool, Tool::Hand);
        assert!(!h.state().ui.views[0].fit_pending, "a single click doesn't fit");
        // Past the double-click window, so the next two clicks are a fresh double-click.
        h.run_steps(40);
        click(&mut h, p);
        click(&mut h, p);
        assert_eq!(h.state().ui.tool, Tool::Hand);
        assert!(h.state().ui.views[0].fit_pending, "a double-click fits on screen");
    }

    /// With no document open the double-click still picks the tool and doesn't panic.
    #[test]
    fn double_clicking_the_hand_tool_without_a_document_is_harmless() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.tool = Tool::Move;
        let mut h = toolbar_harness(app);
        let p = hand_button(&h);
        click(&mut h, p);
        click(&mut h, p);
        assert_eq!(h.state().ui.tool, Tool::Hand);
        assert!(h.state().session.active().is_none());
    }

    /// Double-clicking the Zoom tool zooms to 100% (Photoshop); a single click only picks the tool.
    #[test]
    fn double_clicking_the_zoom_tool_zooms_to_actual_pixels() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", json!({"width": 64, "height": 64})).unwrap();
        app.ui.tool = Tool::Move;
        app.ui.views[0].fit_pending = true;
        app.ui.views[0].zoom = 0.25;
        let mut h = toolbar_harness(app);
        let p = tool_button(&h, Tool::Zoom);
        click(&mut h, p);
        assert_eq!(h.state().ui.tool, Tool::Zoom);
        assert_eq!(h.state().ui.views[0].zoom, 0.25, "a single click doesn't zoom");
        // Past the double-click window, so the next two clicks are a fresh double-click.
        h.run_steps(40);
        click(&mut h, p);
        click(&mut h, p);
        assert_eq!(h.state().ui.tool, Tool::Zoom);
        assert_eq!(h.state().ui.views[0].zoom, 1.0, "a double-click zooms to 100%");
        assert!(!h.state().ui.views[0].fit_pending, "100% replaces a pending fit");
    }

    /// With no document open the double-click still picks the tool and doesn't panic.
    #[test]
    fn double_clicking_the_zoom_tool_without_a_document_is_harmless() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.tool = Tool::Move;
        let mut h = toolbar_harness(app);
        let p = tool_button(&h, Tool::Zoom);
        click(&mut h, p);
        click(&mut h, p);
        assert_eq!(h.state().ui.tool, Tool::Zoom);
        assert!(h.state().session.active().is_none());
    }
}

#[cfg(test)]
mod embedding_tests {
    use super::*;
    use egui_kittest::kittest::Queryable;

    fn whole_app(rail: bool, menu_bar: bool) -> egui_kittest::Harness<'static, PhotocraftApp> {
        let mut h = egui_kittest::Harness::builder().with_size(vec2(1280.0, 800.0)).with_max_steps(64).build_eframe(move |cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
            app.ui.panels.rail = rail;
            app.ui.panels.menu_bar = menu_bar;
            app
        });
        h.run_steps(4);
        h
    }

    fn drawn(h: &egui_kittest::Harness<'_, PhotocraftApp>, panel: &str) -> bool {
        egui::containers::panel::PanelState::load(&h.ctx, egui::Id::new(panel)).is_some()
    }

    /// An app that embeds the UI can hide the icon rail and the menu bar; both are on by default.
    #[test]
    fn rail_and_menu_bar_follow_their_panel_flags() {
        let panels = crate::state::Panels::default();
        assert!(panels.rail && panels.menu_bar);

        let h = whole_app(true, true);
        assert!(drawn(&h, "rail") && drawn(&h, "title_bar"));
        assert!(h.query_by_label("File").is_some());

        let h = whole_app(false, false);
        assert!(!drawn(&h, "rail"), "the icon rail is drawn");
        assert!(!drawn(&h, "title_bar"), "the title bar is drawn");
        assert!(h.query_by_label("File").is_none(), "the menus are drawn");
        assert!(drawn(&h, "dock"), "the dock itself stays");
    }

    /// Settings saved before the flags existed load with both on.
    #[test]
    fn panels_without_the_flags_deserialize_with_both_on() {
        let mut v = serde_json::to_value(crate::state::Panels::default()).unwrap();
        let m = v.as_object_mut().unwrap();
        m.remove("rail");
        m.remove("menu_bar");
        let p: crate::state::Panels = serde_json::from_value(v).unwrap();
        assert!(p.rail && p.menu_bar);
    }
}

#[cfg(test)]
mod layer_drag_edge_scroll_tests {
    use super::*;

    #[test]
    fn inside_edge_scrolls_proportional_to_proximity() {
        let viewport = Rect::from_min_max(pos2(10.0, 30.0), pos2(310.0, 330.0));
        let point = |y| Some(pos2(100.0, y));
        let near_top = layer_drag_edge_scroll(point(33.0), viewport, true, 1.0 / 60.0);
        let far_top = layer_drag_edge_scroll(point(53.0), viewport, true, 1.0 / 60.0);
        let near_bottom = layer_drag_edge_scroll(point(327.0), viewport, true, 1.0 / 60.0);
        let far_bottom = layer_drag_edge_scroll(point(307.0), viewport, true, 1.0 / 60.0);
        assert!(near_top > far_top && far_top > 0.0, "approaching the top reveals earlier rows");
        assert!(near_bottom < far_bottom && far_bottom < 0.0, "approaching the bottom reveals later rows");
        assert!((near_top + near_bottom).abs() < 1e-5, "symmetric edge behavior");
        assert_eq!(layer_drag_edge_scroll(point(160.0), viewport, true, 1.0 / 60.0), 0.0);
    }

    #[test]
    fn outside_edges_scroll_in_drag_direction() {
        let viewport = Rect::from_min_max(pos2(10.0, 30.0), pos2(310.0, 330.0));
        let above = layer_drag_edge_scroll(Some(pos2(100.0, 20.0)), viewport, true, 1.0 / 60.0);
        let below = layer_drag_edge_scroll(Some(pos2(100.0, 335.0)), viewport, true, 1.0 / 60.0);
        assert!(above > 0.0, "dragging past the top continues scrolling upward");
        assert!(below < 0.0, "dragging past the bottom continues scrolling downward");
    }

    #[test]
    fn clamp_bounds_speed_outside_edges_and_across_frame_times() {
        let viewport = Rect::from_min_max(pos2(10.0, 30.0), pos2(310.0, 330.0));
        let at_top = layer_drag_edge_scroll(Some(pos2(100.0, 30.0)), viewport, true, 1.0 / 60.0);
        let past_top = layer_drag_edge_scroll(Some(pos2(100.0, 15.0)), viewport, true, 1.0 / 60.0);
        let far_past_top = layer_drag_edge_scroll(Some(pos2(100.0, -50.0)), viewport, true, 1.0 / 60.0);
        assert_eq!(past_top, at_top, "speed past top edge is clamped to maximum edge speed");
        assert_eq!(far_past_top, at_top, "speed far past top edge remains clamped");

        let at_bottom = layer_drag_edge_scroll(Some(pos2(100.0, 330.0)), viewport, true, 1.0 / 60.0);
        let past_bottom = layer_drag_edge_scroll(Some(pos2(100.0, 345.0)), viewport, true, 1.0 / 60.0);
        let far_past_bottom = layer_drag_edge_scroll(Some(pos2(100.0, 500.0)), viewport, true, 1.0 / 60.0);
        assert_eq!(past_bottom, at_bottom, "speed past bottom edge is clamped to maximum edge speed");
        assert_eq!(far_past_bottom, at_bottom, "speed far past bottom edge remains clamped");

        let active = Some(pos2(100.0, 325.0));
        let step = layer_drag_edge_scroll(active, viewport, true, 1.0 / 60.0);
        let twice = layer_drag_edge_scroll(active, viewport, true, 2.0 / 60.0);
        assert!((twice - step * 2.0).abs() < 1e-4, "time-based scrolling scales across refresh rates");
        assert!(layer_drag_edge_scroll(active, viewport, true, 0.5).abs() <= 30.0, "long frames have a bounded step");
    }

    #[test]
    fn zero_when_no_drag_or_outside_horizontal_bounds() {
        let viewport = Rect::from_min_max(pos2(10.0, 30.0), pos2(310.0, 330.0));
        let active = Some(pos2(100.0, 325.0));
        assert_eq!(layer_drag_edge_scroll(active, viewport, false, 1.0 / 60.0), 0.0);
        assert_eq!(layer_drag_edge_scroll(None, viewport, true, 1.0 / 60.0), 0.0);
        assert_eq!(layer_drag_edge_scroll(Some(pos2(9.0, 325.0)), viewport, true, 1.0 / 60.0), 0.0);
        assert_eq!(layer_drag_edge_scroll(Some(pos2(311.0, 325.0)), viewport, true, 1.0 / 60.0), 0.0);
        assert_eq!(layer_drag_edge_scroll(active, viewport, true, 0.0), 0.0);
    }

    #[test]
    fn short_viewports_keep_the_edge_zones_disjoint() {
        let viewport = Rect::from_min_max(pos2(0.0, 0.0), pos2(150.0, 40.0));
        assert!(layer_drag_edge_scroll(Some(pos2(10.0, 2.0)), viewport, true, 0.016) > 0.0);
        assert!(layer_drag_edge_scroll(Some(pos2(10.0, 38.0)), viewport, true, 0.016) < 0.0);
        assert_eq!(layer_drag_edge_scroll(Some(pos2(10.0, 20.0)), viewport, true, 0.016), 0.0);
    }
}

#[cfg(test)]
mod group_drag_selection_tests {
    use super::*;

    #[test]
    fn dragging_a_selected_layer_moves_the_complete_selection_into_a_group() {
        let a = LayerId(10);
        let b = LayerId(11);
        let group = LayerId(20);
        let payload = layer_drop_payload(a.0, group, "into", &[a, b]);
        assert_eq!(payload, json!({"layers": [10, 11], "target": 20, "position": "into"}));
        assert_eq!(payload.get("layer"), None, "batch drops must not also send a single layer");

        // Above/below use the same batch route; engine preserves the document stack order.
        assert_eq!(layer_drop_payload(b.0, group, "above", &[a, b])["position"], "above");
        assert_eq!(layer_drop_payload(b.0, group, "below", &[a, b])["position"], "below");
    }

    #[test]
    fn the_background_row_neither_moves_nor_takes_a_layer_below_it() {
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        let bg = s.active().unwrap().doc.layers[0].id;
        let a = LayerId(s.execute("layer.new.layer", json!({})).unwrap()["layer"].as_u64().unwrap());
        let doc = &s.active().unwrap().doc;
        let refused = |dragged: LayerId, target: LayerId, pos: &str, sel: &[LayerId], copy: bool| {
            background_drop_refused(doc, &layer_drop_payload(dragged.0, target, pos, sel), target, pos, copy)
        };
        assert!(refused(bg, a, "above", &[bg], false));
        assert!(refused(a, bg, "below", &[a], false));
        assert!(refused(a, bg, "below", &[a], true));
        assert!(refused(a, a, "above", &[bg, a], false), "the Background is among the dragged layers");
        assert!(!refused(a, bg, "above", &[a], false));
        assert!(!refused(bg, a, "above", &[bg], true), "⌥-dragging copies the Background above");
        // Without a Background every drop is fine.
        let mut s = photocraft_engine::Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8})).unwrap();
        s.execute("layer.new.layerFromBackground", json!({})).unwrap();
        let l0 = s.active().unwrap().doc.layers[0].id;
        let doc = &s.active().unwrap().doc;
        assert!(!background_drop_refused(doc, &layer_drop_payload(l0.0, a, "above", &[l0]), a, "above", false));
    }

    #[test]
    fn dragging_unselected_or_singular_row_remains_a_single_layer_move() {
        let a = LayerId(10);
        let b = LayerId(11);
        let group = LayerId(20);
        assert_eq!(layer_drop_payload(9, group, "into", &[a, b]), json!({"layer": 9, "target": 20, "position": "into"}));
        assert_eq!(layer_drop_payload(a.0, group, "above", &[a]), json!({"layer": 10, "target": 20, "position": "above"}));
        assert_eq!(layer_drop_payload(a.0, group, "below", &[]), json!({"layer": 10, "target": 20, "position": "below"}));
    }

    #[test]
    fn layer_thumb_of_a_tall_document_fills_the_height() {
        let cell = Rect::from_min_size(pos2(10.0, 20.0), vec2(30.0, 30.0));
        let (r, uv) = layer_thumb_fit(cell, 100, 400);
        assert_eq!((r.top(), r.bottom()), (cell.top(), cell.bottom()));
        assert!((r.width() - 7.5).abs() < 1e-3, "{r:?}");
        assert!((uv.width() - 0.25).abs() < 1e-3 && (uv.height() - 1.0).abs() < 1e-3, "{uv:?}");
    }
}

#[cfg(test)]
mod opacity_row_layout_tests {
    use super::*;

    #[test]
    fn opacity_label_yields_to_controls_in_narrow_layers_panel() {
        let gap = 2.0;
        let label = 62.0;
        let (wide, visible) = opacity_row_layout(360.0, label, gap);
        assert!(visible);
        assert!(wide >= 100.0);
        // The default dock width keeps the English label (the Layers dock at its 250 px minimum).
        assert!(opacity_row_layout(255.0, 46.0, 8.0).1);
        let (medium, visible) = opacity_row_layout(180.0, label, gap);
        assert!(!visible, "the long label must not overlap the blend field");
        assert!(medium + LAYER_PCT_W + gap <= 180.0);
        let (narrow, visible) = opacity_row_layout(135.0, label, gap);
        assert!(!visible);
        assert!(narrow >= 40.0);
        assert!(narrow + LAYER_PCT_W + gap <= 135.0);
    }
}

#[cfg(test)]
#[path = "clipping_indicator_tests.rs"]
mod clipping_indicator_tests;

#[cfg(test)]
#[path = "mask_props_ui_tests.rs"]
mod mask_properties_tests;

#[cfg(test)]
mod compact_visual_regressions {
    use super::*;
    use egui_kittest::kittest::Queryable;

    fn input_at_scale(screen: Rect, scale: f32) -> egui::RawInput {
        let mut input = egui::RawInput { screen_rect: Some(screen), ..Default::default() };
        input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(scale);
        input
    }

    fn toolbar_harness(kind: theme::ThemeKind, scale: f32) -> egui_kittest::Harness<'static, PhotocraftApp> {
        toolbar_harness_sized(kind, scale, 360.0)
    }

    fn toolbar_harness_sized(kind: theme::ThemeKind, scale: f32, height: f32) -> egui_kittest::Harness<'static, PhotocraftApp> {
        let mut ready = false;
        let mut harness = egui_kittest::Harness::builder().with_size(vec2(400.0, height)).with_pixels_per_point(scale).build_ui_state(
            move |ui, app: &mut PhotocraftApp| {
                if !ready {
                    egui_extras::install_image_loaders(ui.ctx());
                    theme::install_fonts(ui.ctx());
                    theme::apply(ui.ctx(), kind);
                    app.session.prefs.edit(|prefs| prefs.shortcuts.insert("window.togglePanels".into(), String::new()));
                    ready = true;
                    return;
                }
                crate::shortcuts::handle(app, ui.ctx());
                toolbar(app, ui);
            },
            PhotocraftApp::new(photocraft_engine::Session::new(), Default::default()),
        );
        harness.run_steps(4);
        harness
    }

    #[test]
    fn compact_toolbar_scrolls_to_and_activates_the_last_tool() {
        for kind in theme::ThemeKind::ALL {
            for scale in [1.0, 1.5, 2.0] {
                let mut harness = toolbar_harness(kind, scale);
                harness.event(egui::Event::PointerMoved(pos2(25.0, 100.0)));
                harness.event(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: vec2(0.0, -1500.0),
                    modifiers: egui::Modifiers::NONE,
                    phase: egui::TouchPhase::Move,
                });
                harness.run_steps(30);
                let node = harness.get_by_label("Zoom Tool  (Z)");
                let rect = node.rect();
                assert!(rect.top() >= 0.0 && rect.bottom() <= 360.0, "{kind:?} at {scale}x: last tool remains outside the toolbar: {rect:?}");
                node.click();
                harness.run_steps(3);
                assert_eq!(harness.state().ui.tool, Tool::Zoom, "the scrolled button must select its real tool");
            }
        }
    }

    #[test]
    fn compact_toolbar_tab_reveals_and_activates_the_last_tool() {
        for kind in theme::ThemeKind::ALL {
            for scale in [1.0, 1.5, 2.0] {
                let mut harness = toolbar_harness(kind, scale);
                // The supported shortcut preference clears the canvas Tab binding. From no
                // focus, every step below is a real Tab processed by the app's dispatcher.
                let mut reached = false;
                for _ in 0..80 {
                    harness.key_press(egui::Key::Tab);
                    harness.run_steps(3);
                    reached = harness.get_by_label("Zoom Tool  (Z)").is_focused();
                    if reached {
                        break;
                    }
                }
                assert!(reached, "{kind:?} at {scale}x: Tab must reach the last tool");
                harness.run_steps(20);
                let rect = harness.get_by_label("Zoom Tool  (Z)").rect();
                assert!(rect.top() >= 0.0 && rect.bottom() <= 360.0, "{kind:?} at {scale}x: focused tool remains clipped: {rect:?}");
                harness.key_press(egui::Key::Space);
                harness.run_steps(3);
                assert_eq!(harness.state().ui.tool, Tool::Zoom, "Space must select the revealed tool");
            }
        }
    }

    /// A window taller than the tools keeps the colour chips right under the last tool, as
    /// Photoshop does, instead of at the bottom of the toolbar.
    #[test]
    fn tall_toolbar_keeps_the_colour_chips_under_the_tools() {
        for kind in theme::ThemeKind::ALL {
            let harness = toolbar_harness_sized(kind, 1.0, 1600.0);
            let last = harness.get_by_label("Zoom Tool  (Z)").rect();
            let swap = harness.get_by_label("Swap colours (X)").rect();
            assert!(swap.top() - last.bottom() < 80.0, "{kind:?}: the chips {swap:?} sit far below the last tool {last:?}");
        }
    }

    #[test]
    fn capture_compact_visual_fixtures() {
        let Some(directory) = std::env::var_os("CRAFT_UI_VISUAL_FIXTURES").map(std::path::PathBuf::from) else { return };
        std::fs::create_dir_all(&directory).unwrap();
        for kind in crate::theme::ThemeKind::ALL {
            for (width, height) in [(800.0, 480.0), (1280.0, 720.0)] {
                for scale in [1.0, 1.5, 2.0] {
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                    app.session.tools.foreground = [1.0, 0.0, 1.0, 1.0];
                    app.session.tools.background = [0.0, 1.0, 0.0, 1.0];
                    let mut ready = false;
                    let mut harness = egui_kittest::Harness::builder().with_size(vec2(width, height)).with_pixels_per_point(scale).wgpu().build_ui_state(
                        move |ui, app: &mut PhotocraftApp| {
                            if !ready {
                                egui_extras::install_image_loaders(ui.ctx());
                                crate::theme::install_fonts(ui.ctx());
                                crate::theme::apply(ui.ctx(), kind);
                                ready = true;
                                return;
                            }
                            app.ui.tool = if Tokens::get(ui.ctx()).pro { Tool::Pencil } else { Tool::Brush };
                            options_bar(app, ui);
                            toolbar(app, ui);
                        },
                        app,
                    );
                    harness.input_mut().max_texture_side = Some(8192);
                    harness.run_steps(4);
                    let stem = format!("photocraft-compact-{}-{width}-{height}-{scale}x", kind.id());
                    harness.render().unwrap().save(directory.join(format!("{stem}.png"))).unwrap();
                    let mut regions = serde_json::Map::new();
                    for id in ["options_bar", "toolbar"] {
                        let r = egui::containers::panel::PanelState::load(&harness.ctx, egui::Id::new(id)).unwrap().outer_rect;
                        regions.insert(id.into(), json!([r.left(), r.top(), r.width(), r.height()]));
                    }
                    let metadata = json!({"screen": [width, height], "scale": scale, "regions": regions});
                    std::fs::write(directory.join(format!("{stem}.json")), serde_json::to_vec_pretty(&metadata).unwrap()).unwrap();
                    for _ in 0..6 {
                        harness.input_mut().events.extend([
                            egui::Event::PointerMoved(pos2(width * 0.5, 20.0)),
                            egui::Event::MouseWheel {
                                unit: egui::MouseWheelUnit::Point,
                                delta: vec2(-1500.0, 0.0),
                                modifiers: egui::Modifiers::NONE,
                                phase: egui::TouchPhase::Move,
                            },
                        ]);
                        harness.run_steps(1);
                    }
                    harness.render().unwrap().save(directory.join(format!("{stem}-scrolled.png"))).unwrap();
                    std::fs::write(directory.join(format!("{stem}-scrolled.json")), serde_json::to_vec_pretty(&metadata).unwrap()).unwrap();
                }
            }
        }
    }

    #[test]
    fn compact_toolbar_keeps_both_color_chips_visible() {
        for kind in crate::theme::ThemeKind::ALL {
            for height in [480.0, 720.0] {
                for scale in [1.0, 1.5, 2.0] {
                    let ctx = egui::Context::default();
                    egui_extras::install_image_loaders(&ctx);
                    crate::theme::install_fonts(&ctx);
                    crate::theme::apply(&ctx, kind);
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                    app.session.tools.foreground = [1.0, 0.0, 1.0, 1.0];
                    app.session.tools.background = [0.0, 1.0, 0.0, 1.0];
                    let mut output = ctx.run_ui(input_at_scale(Rect::from_min_size(egui::Pos2::ZERO, vec2(800.0, height)), scale), |ui| toolbar(&mut app, ui));
                    output.textures_delta.clear();
                    for color in [Color32::from_rgb(255, 0, 255), Color32::GREEN] {
                        let (clip, chip) = output
                            .shapes
                            .iter()
                            .find_map(|shape| match &shape.shape {
                                egui::Shape::Rect(rect) if rect.fill == color => Some((shape.clip_rect, rect.rect)),
                                egui::Shape::Circle(circle) if circle.fill == color => {
                                    Some((shape.clip_rect, Rect::from_center_size(circle.center, vec2(circle.radius * 2.0, circle.radius * 2.0))))
                                }
                                _ => None,
                            })
                            .unwrap_or_else(|| panic!("{kind:?} at {height}/{scale}: both color chips are painted"));
                        assert!(clip.contains_rect(chip) && chip.bottom() <= height, "{kind:?} at {height}: color chip {chip:?} clipped by {clip:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn tab_focus_reveals_the_final_narrow_tool_option() {
        for kind in theme::ThemeKind::ALL {
            for scale in [1.0, 1.5, 2.0] {
                let mut ready = false;
                let mut harness = egui_kittest::Harness::builder().with_size(vec2(360.0, 120.0)).with_pixels_per_point(scale).build_ui_state(
                    move |ui, app: &mut PhotocraftApp| {
                        if !ready {
                            egui_extras::install_image_loaders(ui.ctx());
                            theme::install_fonts(ui.ctx());
                            theme::apply(ui.ctx(), kind);
                            app.ui.tool = if Tokens::get(ui.ctx()).pro { Tool::Pencil } else { Tool::Brush };
                            app.session.prefs.edit(|prefs| prefs.shortcuts.insert("window.togglePanels".into(), String::new()));
                            ready = true;
                            return;
                        }
                        crate::shortcuts::handle(app, ui.ctx());
                        options_bar(app, ui);
                    },
                    PhotocraftApp::new(photocraft_engine::Session::new(), Default::default()),
                );
                harness.run_steps(4);
                let pro = Tokens::get(&harness.ctx).pro;
                let last_label = if pro { "Auto Erase" } else { "Pressure for Opacity" };
                // Only the canvas Tab shortcut is cleared, using its public preference;
                // no response is assigned focus programmatically.
                let mut reached = false;
                for _ in 0..80 {
                    harness.key_press(egui::Key::Tab);
                    harness.run_steps(3);
                    let label = harness.output().shapes.iter().find_map(|shape| match &shape.shape {
                        egui::Shape::Text(text) if text.galley.job.text.contains(last_label) => Some(text.visual_bounding_rect()),
                        _ => None,
                    });
                    reached = label.is_some_and(|label| {
                        harness.ctx.memory(|memory| memory.focused()).and_then(|id| harness.ctx.read_response(id)).is_some_and(|response| {
                            response.rect.intersects(label)
                                || ((label.left() - response.rect.right()).abs() <= 12.0
                                    && response.rect.top() < label.bottom()
                                    && response.rect.bottom() > label.top())
                        })
                    });
                    if reached {
                        break;
                    }
                }
                assert!(reached, "{kind:?} at {scale}x: Tab must reach the final tool option");
                harness.run_steps(20);
                let (clip, label) = harness
                    .output()
                    .shapes
                    .iter()
                    .find_map(|shape| match &shape.shape {
                        egui::Shape::Text(text) if text.galley.job.text.contains(last_label) => Some((shape.clip_rect, text.visual_bounding_rect())),
                        _ => None,
                    })
                    .expect("the final option remains painted");
                assert!(clip.contains_rect(label), "{kind:?} at {scale}x: Tab leaves the focused final option outside {clip:?}: {label:?}");
                let previous = if pro { harness.state().ui.tool_options.pencil_auto_erase } else { harness.state().session.tools.brush.pressure_opacity };
                harness.key_press(egui::Key::Space);
                harness.run_steps(3);
                let current = if pro { harness.state().ui.tool_options.pencil_auto_erase } else { harness.state().session.tools.brush.pressure_opacity };
                assert_ne!(current, previous, "{kind:?} at {scale}x: Space must activate the revealed option");
                assert!(harness.state().ui.panels.options_bar, "widget traversal must preserve the options bar");
            }
        }
    }

    #[test]
    fn a_wide_crop_options_bar_does_not_scroll() {
        for kind in crate::theme::ThemeKind::ALL {
            for scale in [1.0, 1.5, 2.0] {
                for mode in ["", crate::crop_size::WHR] {
                    let ctx = egui::Context::default();
                    crate::theme::install_fonts(&ctx);
                    crate::theme::apply(&ctx, kind);
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                    if !Tokens::get(&ctx).pro {
                        continue;
                    }
                    app.ui.tool = Tool::Crop;
                    app.ui.tool_options.crop_ratio = mode.into();
                    let screen = Rect::from_min_size(egui::Pos2::ZERO, vec2(1600.0, 120.0));
                    let clear_x = |output: &egui::FullOutput| {
                        output
                            .shapes
                            .iter()
                            .find_map(|s| match &s.shape {
                                egui::Shape::Text(t) if t.galley.job.text == "Clear" => Some(t.pos.x),
                                _ => None,
                            })
                            .unwrap()
                    };
                    let mut before = ctx.run_ui(input_at_scale(screen, scale), |ui| options_bar(&mut app, ui));
                    before.textures_delta.clear();
                    for _ in 0..3 {
                        before = ctx.run_ui(input_at_scale(screen, scale), |ui| options_bar(&mut app, ui));
                        before.textures_delta.clear();
                    }
                    let mut after = before.clone();
                    for _ in 0..6 {
                        after = ctx.run_ui(
                            egui::RawInput {
                                events: vec![
                                    egui::Event::PointerMoved(pos2(800.0, 20.0)),
                                    egui::Event::MouseWheel {
                                        unit: egui::MouseWheelUnit::Point,
                                        delta: vec2(-100.0, 0.0),
                                        modifiers: egui::Modifiers::NONE,
                                        phase: egui::TouchPhase::Move,
                                    },
                                ],
                                ..input_at_scale(screen, scale)
                            },
                            |ui| options_bar(&mut app, ui),
                        );
                        after.textures_delta.clear();
                    }
                    assert_eq!(clear_x(&before), clear_x(&after), "a roomy bar must not have horizontal overflow");
                }
            }
        }
    }

    #[test]
    fn the_narrow_options_bar_preference_shortens_the_bar() {
        assert_eq!(options_bar_height(true, false), 36.0);
        assert_eq!(options_bar_height(true, true), 30.0);
        assert_eq!(options_bar_height(false, false), 42.0);
        assert_eq!(options_bar_height(false, true), 34.0);
        let ctx = egui::Context::default();
        crate::theme::install_fonts(&ctx);
        crate::theme::apply(&ctx, crate::theme::ThemeKind::ALL[0]);
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        let screen = Rect::from_min_size(egui::Pos2::ZERO, vec2(1600.0, 900.0));
        let run = |app: &mut PhotocraftApp| {
            let mut top = 0.0;
            let mut out = ctx.run_ui(input_at_scale(screen, 1.0), |ui| {
                options_bar(app, ui);
                top = ui.available_rect_before_wrap().top();
            });
            out.textures_delta.clear();
            top
        };
        let normal = run(&mut app);
        assert_eq!(normal, options_bar_height(Tokens::get(&ctx).pro, false));
        app.session.execute("prefs.set", json!({"path": "workspace.enableNarrowOptionsBar", "value": true})).unwrap();
        let narrow = run(&mut app);
        assert_eq!(narrow, options_bar_height(Tokens::get(&ctx).pro, true));
        assert!(narrow < normal, "narrow {narrow} must trim the bar under {normal}");
    }

    #[test]
    fn narrow_tool_options_can_scroll_to_the_last_control() {
        for kind in crate::theme::ThemeKind::ALL {
            for width in [360.0, 800.0, 1280.0] {
                for scale in [1.0, 1.5, 2.0] {
                    let ctx = egui::Context::default();
                    egui_extras::install_image_loaders(&ctx);
                    crate::theme::install_fonts(&ctx);
                    crate::theme::apply(&ctx, kind);
                    let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
                    let (tool, last_label) = if Tokens::get(&ctx).pro { (Tool::Pencil, "Auto Erase") } else { (Tool::Brush, "Pressure for Opacity") };
                    app.ui.tool = tool;
                    let screen = Rect::from_min_size(egui::Pos2::ZERO, vec2(width, 120.0));
                    let mut output = ctx.run_ui(input_at_scale(screen, scale), |ui| options_bar(&mut app, ui));
                    output.textures_delta.clear();
                    for frame in 1..=6 {
                        output = ctx.run_ui(
                            egui::RawInput {
                                time: Some(f64::from(frame) * 0.1),
                                events: vec![
                                    // Studio: above the controls, clear of the Brush's Mode dropdown, which takes the wheel
                                    // to step through blend modes (#4512).
                                    egui::Event::PointerMoved(if Tokens::get(&ctx).pro { pos2(width * 0.5, 20.0) } else { pos2(width * 0.5, 2.0) }),
                                    egui::Event::MouseWheel {
                                        unit: egui::MouseWheelUnit::Point,
                                        delta: vec2(-1500.0, 0.0),
                                        modifiers: egui::Modifiers::NONE,
                                        phase: egui::TouchPhase::Move,
                                    },
                                ],
                                ..input_at_scale(screen, scale)
                            },
                            |ui| options_bar(&mut app, ui),
                        );
                        output.textures_delta.clear();
                    }
                    let (clip, label) = output
                        .shapes
                        .iter()
                        .find_map(|shape| match &shape.shape {
                            egui::Shape::Text(text) if text.galley.job.text.contains(last_label) => Some((shape.clip_rect, text)),
                            _ => None,
                        })
                        .expect("scrolling reaches the final tool control");
                    assert!(
                        clip.contains_rect(label.visual_bounding_rect()),
                        "{kind:?} at {width}: final control is clipped: {:?} / {clip:?}",
                        label.visual_bounding_rect()
                    );
                    assert!(label.visual_bounding_rect().right() <= width, "the final option stays inside the window");
                }
            }
        }
    }
}

#[cfg(test)]
mod properties_target_regressions {
    use super::*;
    use egui_kittest::{Harness, kittest::Queryable};

    #[test]
    fn pending_transform_arithmetic_does_not_apply_to_another_document_with_the_same_layer_id() {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
        app.run("file.new", json!({"width":200,"height":200,"background":"transparent"})).unwrap();
        app.run("select.rect", json!({"x":10,"y":20,"width":100,"height":50})).unwrap();
        app.run("edit.fill", json!({"color":"#ff0000"})).unwrap();
        app.run("select.deselect", json!({})).unwrap();
        let layer = app.session.active().unwrap().active_layer.unwrap();
        let duplicate = (*app.session.active().unwrap().doc).clone();
        let second = app.session.add_document(duplicate, None);
        app.session.active_mut().unwrap().active_layer = Some(layer);
        assert!(app.session.set_active(0));
        let mut ready = false;
        let mut h = Harness::builder().with_size(vec2(360.0, 650.0)).build_ui_state(
            move |ui, app: &mut PhotocraftApp| {
                if !ready {
                    PhotocraftApp::setup_context(ui.ctx(), theme::ThemeKind::ProMedium);
                    ready = true;
                    return;
                }
                properties_body(app, ui);
            },
            app,
        );
        h.run_steps(3);
        h.get_all_by_role(egui::accesskit::Role::SpinButton).next().expect("transform width").click();
        h.run_steps(1);
        for c in "50/2".chars() {
            h.event(egui::Event::Text(c.to_string()));
            h.run_steps(1);
        }
        let before = h.state().session.documents()[second].doc.layer(layer).unwrap().surface().unwrap().content_bounds();
        assert!(h.state_mut().session.set_active(second));
        h.run_steps(2);
        h.key_press(egui::Key::Enter);
        h.run_steps(2);
        let after = h.state().session.active().unwrap().doc.layer(layer).unwrap().surface().unwrap().content_bounds();
        assert_eq!(after, before, "pending width belongs to the old document/layer");
    }
}

#[cfg(test)]
#[path = "history_panel_select_tests.rs"]
mod history_panel_select_tests;
