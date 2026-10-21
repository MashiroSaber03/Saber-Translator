//! A live, locally owned popup: egui's memory popup slot belongs to its child ComboBoxes.
use egui::{Rect, Sense, Stroke, pos2, vec2};
use photocraft_doc::{LineCap, LineJoin, StrokeAlign};
use serde::{Deserialize, Serialize};

use super::StrokeOptions;
use crate::theme::{self, Tokens};

/// Inspection state. Valid edits are applied immediately; only invalid numeric input is pending.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrokeEditor {
    pub key: String,
    pub draft: StrokeOptions,
    pub original: StrokeOptions,
    pub target: Option<(photocraft_doc::DocId, photocraft_doc::LayerId)>,
}

pub struct StrokeChange {
    pub options: StrokeOptions,
    pub coalesce: String,
}

#[derive(Default)]
struct Children {
    shown: bool,
    clicked: bool,
    ids: Vec<egui::Id>,
}

pub(super) fn close_children(ctx: &egui::Context, key: &str) {
    let ids = ctx.data_mut(|d| d.remove_temp::<Vec<egui::Id>>(egui::Id::new(("stroke-children", key)))).unwrap_or_default();
    for id in ids {
        egui::Popup::close_id(ctx, id);
    }
}

/// A line sample and the entire style row share one accessible, keyboard-operable hit target.
fn style_button(ui: &mut egui::Ui, options: &StrokeOptions, label: &str, selected: bool, width: f32, opener: bool, enabled: bool) -> egui::Response {
    let t = Tokens::get(ui.ctx());
    let font = egui::TextStyle::Button.resolve(ui.style());
    let color = if selected && t.bevel { t.accent_text } else { t.text };
    let galley = ui.painter().layout_no_wrap(tl!(label).to_owned(), font, color);
    let height = (galley.size().y + if opener { 8.0 } else { 14.0 }).max(if opener { 24.0 } else { 32.0 });
    let (rect, response) = ui.allocate_exact_size(vec2(width.max(galley.size().x + 92.0), height), Sense::click());
    response.widget_info(|| {
        if opener {
            egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Stroke Options"))
        } else {
            egui::WidgetInfo::selected(egui::WidgetType::RadioButton, ui.is_enabled(), selected, tl!(label))
        }
    });
    let active = selected || response.has_focus();
    let fill = if selected {
        t.accent_soft
    } else if response.hovered() {
        t.hover
    } else {
        t.field
    };
    ui.painter().rect_filled(rect, t.radius_sm, fill);
    ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, if active { t.accent_border } else { t.field_border }), egui::StrokeKind::Inside);
    ui.painter().galley(pos2(rect.left() + 8.0, rect.center().y - galley.size().y / 2.0), galley.clone(), color);
    let sample = Rect::from_min_max(
        pos2(rect.left() + galley.size().x + 18.0, rect.top() + 3.0),
        pos2(rect.right() - if opener { 20.0 } else { 8.0 }, rect.bottom() - 3.0),
    );
    if enabled {
        paint_sample(ui, sample, response.id.with("sample"), options, false, color);
    } else {
        ui.painter().line_segment([sample.left_center(), sample.right_center()], Stroke::new(1.0, t.text_faint));
        ui.painter().line_segment([sample.center() + vec2(-4.0, 6.0), sample.center() + vec2(4.0, -6.0)], Stroke::new(1.5, t.danger));
    }
    if opener {
        crate::icons::paint(ui, Rect::from_center_size(rect.right_center() - vec2(10.0, 0.0), vec2(12.0, 12.0)), "chevron-down", 12.0, t.text_dim);
    }
    response
}

/// Emit only deliberate, valid changes. No confirmation footer or document preview snapshot.
#[allow(clippy::too_many_arguments)]
pub fn button(
    ui: &mut egui::Ui,
    key: &str,
    current: &StrokeOptions,
    editor: &mut Option<StrokeEditor>,
    closed: bool,
    corners: bool,
    target: Option<(photocraft_doc::DocId, photocraft_doc::LayerId)>,
    stroke_enabled: bool,
) -> Option<StrokeChange> {
    let label = if stroke_enabled { current.label() } else { "No Color" };
    let response = style_button(ui, current, label, false, 138.0, true, stroke_enabled).on_hover_text(tl!("Stroke Options"));
    let own = editor.as_ref().is_some_and(|e| e.key == key);
    if response.clicked() {
        if let Some(e) = editor.as_ref() {
            close_children(ui.ctx(), &e.key);
        }
        if own {
            *editor = None;
        } else {
            let draft = if current.valid() { current.clone() } else { StrokeOptions::default() };
            *editor = Some(StrokeEditor { key: key.into(), draft, original: current.clone(), target });
        }
    }
    if editor.as_ref().is_some_and(|e| e.key == key && (e.original != *current || e.target != target)) || !ui.is_enabled() && own {
        close_children(ui.ctx(), key);
        *editor = None;
    }
    let e = editor.as_mut().filter(|e| e.key == key)?;
    super::mark_rendered(ui.ctx(), key);
    let before = e.draft.clone();
    let mut selected_preset = false;
    let mut children = Children::default();
    let gesture_id = response.id.with("gesture");
    let pass = ui.ctx().cumulative_pass_nr();
    // Track presses even when a drag has not changed a value yet, like the mask controls.
    let pressed = ui.input(|i| i.pointer.any_pressed());
    let gesture = ui.data_mut(|d| {
        let g = d.get_temp_mut_or_default::<(u64, u64)>(gesture_id);
        if pressed && g.1 != pass {
            g.0 = g.0.wrapping_add(1);
            g.1 = pass;
        }
        g.0
    });
    // A memory-backed parent is replaced as soon as a ComboBox opens. Explicit ownership
    // keeps the parent alive; dismissal below considers both parent and child hit regions.
    let popup = egui::Popup::from_response(&response).open(true).close_behavior(egui::PopupCloseBehavior::IgnoreClicks).show(|ui| {
        ui.spacing_mut().item_spacing = vec2(8.0, theme::ROW_GAP);
        let width = 272.0_f32.min((ui.ctx().content_rect().width() - 24.0).max(160.0));
        ui.set_width(width);
        let height = (ui.ctx().content_rect().height() - 40.0).max(80.0);
        // An Area remembers its previous size. Let a newly selected dashed style grow the
        // body before the ScrollArea measures it, then shrink back to the actual content.
        ui.set_max_height(height);
        egui::ScrollArea::vertical().max_height(height).auto_shrink([false, true]).show(ui, |ui| {
            ui.label(egui::RichText::new(tl!("Stroke Options")).strong());
            let s = &mut e.draft;
            for name in ["Solid", "Dashed", "Dotted"] {
                let mut sample = s.clone();
                sample.preset(name);
                // Presets illustrate their pattern even when current opacity is zero.
                sample.opacity = 1.0;
                if style_button(ui, &sample, name, s.label() == name, width, false, true).clicked() {
                    s.preset(name);
                    selected_preset = true;
                }
            }
            ui.add_space(theme::ROW_GAP);
            ui.separator();
            egui::Grid::new("stroke-geometry").num_columns(2).spacing(vec2(12.0, theme::ROW_GAP)).show(ui, |ui| {
                ui.add_enabled_ui(closed, |ui| {
                    ui.label(tl!("Alignment"));
                });
                ui.add_enabled_ui(closed, |ui| {
                    choice(
                        ui,
                        "Alignment",
                        &mut s.align,
                        &[(StrokeAlign::Inside, "Inside"), (StrokeAlign::Center, "Center"), (StrokeAlign::Outside, "Outside")],
                        &mut children,
                    )
                });
                ui.end_row();
                ui.add_enabled_ui(!closed || !s.dashes.is_empty(), |ui| {
                    ui.label(tl!("Caps"));
                });
                ui.add_enabled_ui(!closed || !s.dashes.is_empty(), |ui| {
                    choice(ui, "Caps", &mut s.cap, &[(LineCap::Butt, "Butt / Flat"), (LineCap::Round, "Round"), (LineCap::Square, "Square")], &mut children)
                });
                ui.end_row();
                ui.add_enabled_ui(corners, |ui| {
                    ui.label(tl!("Corners"));
                });
                ui.add_enabled_ui(corners, |ui| {
                    choice(ui, "Corners", &mut s.join, &[(LineJoin::Miter, "Miter"), (LineJoin::Round, "Round"), (LineJoin::Bevel, "Bevel")], &mut children)
                });
                ui.end_row();
                ui.add_enabled_ui(corners && s.join == LineJoin::Miter, |ui| {
                    ui.label(tl!("Miter limit"));
                });
                ui.add_enabled_ui(corners && s.join == LineJoin::Miter, |ui| number(ui, "Miter limit", &mut s.miter_limit, 1.0..=500.0, "×"));
                ui.end_row();
                ui.label(tl!("Opacity"));
                let mut opacity = s.opacity * 100.0;
                if number(ui, "Opacity", &mut opacity, 0.0..=100.0, "%") {
                    s.opacity = opacity / 100.0;
                }
                ui.end_row();
            });
            if !closed {
                ui.label(egui::RichText::new(tl!("Open paths use centered strokes")).small());
            }
            ui.separator();
            let mut dashed = !s.dashes.is_empty();
            if crate::widgets::checkbox(ui, &mut dashed, "Dashed line").changed() {
                s.preset(if dashed { "Dashed" } else { "Solid" });
            }
            if dashed {
                ui.label(egui::RichText::new(tl!("Dash / gap lengths × stroke width")).small());
                for (i, pair) in s.dashes.chunks_mut(2).enumerate() {
                    ui.push_id(i, |ui| {
                        ui.horizontal(|ui| {
                            for (j, value) in pair.iter_mut().enumerate() {
                                let name = if j == 0 { "Dash" } else { "Gap" };
                                ui.label(tl!(name));
                                number(ui, name, value, 0.0..=10000.0, "×");
                            }
                        });
                    });
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(s.dashes.len() < 32, egui::Button::new(tl!("Add pair"))).clicked() {
                        if s.dashes.len() % 2 == 1 {
                            s.dashes.push(2.0);
                        } else {
                            s.dashes.extend([4.0, 2.0]);
                        }
                    }
                    if ui.add_enabled(s.dashes.len() > 2, egui::Button::new(tl!("Remove pair"))).clicked() {
                        s.dashes.truncate(s.dashes.len().saturating_sub(2));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label(tl!("Dash offset"));
                    number(ui, "Dash offset", &mut s.dash_offset, -1_000_000.0..=1_000_000.0, "×");
                });
            }
            let t = Tokens::get(ui.ctx());
            let (rect, sample_response) = ui.allocate_exact_size(vec2(width, 58.0), Sense::hover());
            ui.painter().rect_filled(rect, t.radius_sm, t.field);
            paint_sample(ui, rect, sample_response.id, s, true, t.text);
            ui.painter().rect_stroke(rect, t.radius_sm, Stroke::new(1.0, t.field_border), egui::StrokeKind::Inside);
            if !s.valid() {
                ui.colored_label(t.danger, tl!("Use finite lengths and a positive pattern total"));
            }
        });
    });
    let changed = e.draft != before || selected_preset && !stroke_enabled;
    let result = if changed && e.draft.valid() {
        e.original = e.draft.clone();
        if !ui.input(|i| i.pointer.any_down()) {
            ui.data_mut(|d| d.get_temp_mut_or_default::<(u64, u64)>(gesture_id).0 = gesture.wrapping_add(1));
        }
        Some(StrokeChange { options: e.draft.clone(), coalesce: format!("shape-stroke:{key}:{}:{gesture}", gesture_id.value()) })
    } else {
        None
    };
    let escape = ui.input(|i| i.key_pressed(egui::Key::Escape));
    let outside = popup.is_some_and(|p| p.response.clicked_elsewhere()) && !children.clicked && !response.clicked();
    ui.data_mut(|d| d.insert_temp(egui::Id::new(("stroke-children", key)), children.ids));
    if outside || escape && !children.shown {
        close_children(ui.ctx(), key);
        *editor = None;
        if escape {
            response.request_focus();
        }
    }
    result
}

fn choice<T: Copy + PartialEq>(ui: &mut egui::Ui, label: &str, value: &mut T, choices: &[(T, &str)], children: &mut Children) {
    let parent_layer = ui.layer_id();
    let (press, close) = crate::press_menu::PressCombo::before(ui, label);
    let response = egui::ComboBox::from_id_salt(label)
        .width(152.0)
        .selected_text(choices.iter().find(|(v, _)| v == value).map_or("", |(_, n)| tl!(*n)))
        .close_behavior(close)
        .show_ui(ui, |ui| {
            // Restored Area order can place a previously used child behind its parent.
            // Both request the top on opening, so explicitly preserve their relationship.
            ui.ctx().set_sublayer(parent_layer, ui.layer_id());
            for (v, name) in choices {
                let item = ui.selectable_label(*value == *v, tl!(*name));
                if press.chosen(ui, &item) {
                    *value = *v;
                }
            }
        });
    press.after(&response.response);
    response.response.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::ComboBox, ui.is_enabled(), tl!(label)));
    let id = egui::Popup::default_response_id(&response.response);
    children.ids.push(id);
    if response.inner.is_some() {
        children.shown = true;
        children.clicked |= ui.ctx().read_response(id).is_some_and(|r| !r.clicked_elsewhere());
    }
}

fn number(ui: &mut egui::Ui, name: &str, value: &mut f32, range: std::ops::RangeInclusive<f32>, unit: &str) -> bool {
    let response = crate::widgets::value_field(ui, value, range, unit, 74.0);
    response.widget_info(|| egui::WidgetInfo::slider(ui.is_enabled(), f64::from(*value), tl!(name)));
    response.changed()
}

#[derive(Clone)]
struct SampleCache {
    options: StrokeOptions,
    size: egui::Vec2,
    scale: f32,
    corners: bool,
    color: egui::Color32,
    texture: egui::TextureHandle,
}

/// Bounded, DPI-aware samples of the actual vector renderer, cached independently per control.
fn paint_sample(ui: &egui::Ui, rect: Rect, id: egui::Id, options: &StrokeOptions, corners: bool, color: egui::Color32) {
    if !rect.is_finite() || rect.width() < 2.0 || rect.height() < 2.0 {
        return;
    }
    let size = rect.size();
    let scale = ui.ctx().pixels_per_point().clamp(0.5, 4.0).min(1024.0 / size.x.max(size.y));
    let mut sample = options.clone();
    if !corners {
        sample.align = StrokeAlign::Center;
    }
    let cached = ui.data(|d| d.get_temp::<SampleCache>(id));
    let cache = cached.filter(|c| c.options == sample && c.size == size && c.scale == scale && c.corners == corners && c.color == color).unwrap_or_else(|| {
        let width = (size.x * scale).ceil().max(1.0) as u32;
        let height = (size.y * scale).ceil().max(1.0) as u32;
        let path = if corners {
            photocraft_vector::shapes::rounded_rect(12.0, 12.0, f64::from(size.x - 24.0), f64::from(size.y - 24.0), [0.0; 4])
        } else {
            let mut sp = photocraft_doc::Subpath::polygon(&[(4.0, f64::from(size.y / 2.0)), (f64::from(size.x - 4.0), f64::from(size.y / 2.0))]);
            sp.closed = false;
            photocraft_doc::Path::new(vec![sp])
        };
        let path = path.transform(&photocraft_geom::Affine { m: [f64::from(scale), 0.0, 0.0, f64::from(scale), 0.0, 0.0] });
        let shape = photocraft_doc::ShapeLayer {
            path,
            stroke: Some(sample.stroke(
                2.5 * scale,
                photocraft_doc::Fill::Solid(photocraft_doc::Color::rgba(
                    f32::from(color.r()) / 255.0,
                    f32::from(color.g()) / 255.0,
                    f32::from(color.b()) / 255.0,
                    1.0,
                )),
            )),
            ..Default::default()
        };
        let bounds = photocraft_geom::Rect::from_size(photocraft_geom::Size::new(width, height));
        let pixels = photocraft_vector::CompiledShape::new(&shape, photocraft_vector::DEFAULT_TOLERANCE, bounds).render_rgba(bounds);
        let bytes: Vec<u8> = pixels.iter().flat_map(|p| p.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)).collect();
        let texture = ui.ctx().load_texture(
            "stroke-sample",
            egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &bytes),
            egui::TextureOptions::LINEAR,
        );
        SampleCache { options: sample, size, scale, corners, color, texture }
    });
    ui.painter().image(cache.texture.id(), rect, Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0)), egui::Color32::WHITE);
    ui.data_mut(|d| d.insert_temp(id, cache));
}
