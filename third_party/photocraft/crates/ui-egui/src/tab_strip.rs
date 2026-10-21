//! Dock tab strips that fit any width (#151), like Photoshop's: when the tabs don't fit beside
//! the panel menu button they shrink, eliding their labels with '…', and once they are at their
//! minimum width the ones that still don't fit move into a » chevron menu at the end of the
//! strip. The selected tab always stays on the strip, and no tab ever runs under the menu button.
//! On a reorderable strip a tab dragged along the strip moves to where it is dropped (#2272).

use egui::{CornerRadius, Rect, Response, Sense, Stroke, StrokeKind, Ui, pos2, vec2};

use crate::theme::{self, Tokens};

pub use craft_ui::tabs::TabFit;

/// Fit tabs to a strip using the shared, bounded width policy. Local wrappers retain the
/// application's themes, translated labels and panel/document overflow menus.
pub fn fit(natural: &[f32], selected: usize, avail: f32, min_w: f32, chevron_w: f32) -> TabFit {
    craft_ui::tabs::fit(natural, selected, avail, min_w, chevron_w)
}

/// A label laid out on one row no wider than `max_w`, cut with '…' when it doesn't fit.
pub fn elided(ui: &Ui, text: &str, font: egui::FontId, color: egui::Color32, max_w: f32) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text.to_owned(), font, color);
    job.wrap = egui::text::TextWrapping { max_width: max_w.max(1.0), max_rows: 1, break_anywhere: true, overflow_character: Some('…') };
    ui.painter().layout_job(job)
}

/// Context-menu action chosen on a dock tab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TabContextAction {
    Close(usize),
    CloseGroup,
}

/// What a strip reported this frame.
pub struct StripOut {
    /// A tab was activated, including a choice from the overflow menu.
    pub clicked: bool,
    pub double_clicked: bool,
    pub context: Option<TabContextAction>,
    /// Rects of the tabs on the strip, `(tab index, rect)`.
    pub tabs: Vec<(usize, Rect)>,
    pub responses: Vec<(usize, Response)>,
    /// The » overflow button, when some tabs didn't fit.
    pub chevron: Option<Rect>,
    /// A tab was dragged along the strip and dropped: `(tab, before)`, draw `tab` just before
    /// tab `before` (`None`: last).
    pub reorder: Option<(usize, Option<usize>)>,
    /// The index of a tab being dragged away from this strip.
    pub dragging_out: Option<usize>,
    /// The index of a tab whose drag ended away from this strip.
    pub dropped_out: Option<usize>,
}

/// Width of the » overflow button.
pub const CHEVRON_W: f32 = 18.0;

/// Draw the » overflow button at `r`: hover tip `tip`, listing the tabs named by `labels` at the
/// indices in `overflow`. A picked index is left in `picked`.
pub fn overflow_button(ui: &mut Ui, id: egui::Id, r: Rect, tip: &str, labels: &[&str], overflow: &[usize], picked: &mut Option<usize>) {
    let t = Tokens::get(ui.ctx());
    let resp = ui.interact(r, id, Sense::click());
    let mut paint_ui = ui.new_child(egui::UiBuilder::new().max_rect(r));
    paint_ui.set_clip_rect(r.intersect(ui.clip_rect()));
    if resp.hovered() {
        paint_ui.painter().rect_filled(r.shrink2(vec2(1.0, 3.0)), t.radius_sm, t.hover.gamma_multiply(0.6));
    }
    crate::icons::paint(&paint_ui, r, "chevrons-right", 12.0, if resp.hovered() { t.text } else { t.text_dim });
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tip));
    let resp = resp.on_hover_text(tip);
    egui::Popup::menu(&resp).show(|ui| {
        ui.set_min_width(140.0);
        for &i in overflow {
            if let Some(name) = labels.get(i)
                && ui.button(*name).clicked()
            {
                *picked = Some(i);
                ui.close();
            }
        }
    });
}

/// Draw the tabs of a strip in `area` (the strip minus the menu button). `paint_tab` draws one
/// tab: (ui, rect, index, label galley, response, active).
#[allow(clippy::too_many_arguments)]
fn tabs_in(
    ui: &mut Ui,
    id: egui::Id,
    area: Rect,
    tabs: &[&str],
    selected: &mut usize,
    font: egui::FontId,
    pad: f32,
    min_w: f32,
    reorderable: bool,
    active: impl Fn(usize) -> bool,
    mut paint_tab: impl FnMut(&Ui, Rect, usize, std::sync::Arc<egui::Galley>, &Response, bool),
) -> StripOut {
    let t = Tokens::get(ui.ctx());
    // Panel names are English keys; draw them in the UI language.
    let names: Vec<&str> = tabs.iter().map(|n| tl!(n)).collect();
    let tabs = names.as_slice();
    let natural: Vec<f32> = tabs.iter().map(|n| ui.painter().layout_no_wrap((*n).to_owned(), font.clone(), t.text).size().x + pad).collect();
    let f = fit(&natural, *selected, area.width(), min_w, CHEVRON_W);
    let mut x = area.left();
    let mut out = StripOut {
        clicked: false,
        double_clicked: false,
        context: None,
        tabs: Vec::with_capacity(f.shown.len()),
        responses: Vec::with_capacity(f.shown.len()),
        chevron: None,
        reorder: None,
        dragging_out: None,
        dropped_out: None,
    };
    // Without reordering, drags on a tab fall through to the strip behind it.
    let sense = if reorderable { Sense::click_and_drag() } else { Sense::click() };
    let mut drag: Option<(usize, bool)> = None;
    for &(i, w) in &f.shown {
        let Some(name) = tabs.get(i) else { continue };
        let r = Rect::from_min_size(pos2(x, area.top()), vec2(w, area.height()));
        // The padding gives way (down to a third) before the label is cut.
        let galley = elided(ui, name, font.clone(), t.text, (w - pad / 3.0).max(1.0));
        let cut = galley.size().x + pad + 0.5 < natural.get(i).copied().unwrap_or(0.0) && galley.size().x + pad / 3.0 >= w - 0.5;
        let resp = craft_ui::tabs::Tab::new(id.with(("tab", i)), name, i == *selected).sense(sense).focus_stroke(Stroke::new(1.0, t.accent)).show_at(
            ui,
            r,
            |ui, resp| paint_tab(ui, r, i, galley, resp, active(i)),
        );
        let resp = if cut { resp.on_hover_text(*name) } else { resp };
        if resp.drag_started() {
            *selected = i;
        }
        if resp.dragged() {
            drag = Some((i, false));
        } else if resp.drag_stopped() {
            drag = Some((i, true));
        }
        out.double_clicked |= resp.double_clicked();
        if resp.clicked() {
            out.clicked = true;
            *selected = i;
        }
        // The context menu belongs to the actual tab response, so its normal clicks
        // and double-click-to-collapse behavior are not intercepted by an overlay.
        resp.context_menu(|ui| {
            if ui.button(tl!("Close")).clicked() {
                out.context = Some(TabContextAction::Close(i));
                ui.close();
            }
            if ui.button(tl!("Close Tab Group")).clicked() {
                out.context = Some(TabContextAction::CloseGroup);
                ui.close();
            }
        });
        out.tabs.push((i, r));
        out.responses.push((i, resp));
        x = r.right();
    }
    if !f.overflow.is_empty() && f.overflow_width > 0.0 {
        let r = Rect::from_min_size(pos2(x, area.top()), vec2(f.overflow_width, area.height()));
        let mut picked = None;
        overflow_button(ui, id.with("tab-overflow"), r, tl!("More panels"), tabs, &f.overflow, &mut picked);
        if let Some(i) = picked {
            out.clicked = true;
            *selected = i;
        }
        out.chevron = Some(r);
    }
    if let Some((tab, released)) = drag
        && let Some(p) = ui.ctx().pointer_interact_pos().or_else(|| ui.ctx().pointer_latest_pos())
    {
        match drop_slot(&out.tabs, area, p) {
            Some(slot) => {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                let before = match out.tabs.get(slot) {
                    Some((b, _)) => Some(*b),
                    // After the last tab on the strip: before the first overflowing one, if any.
                    None => out.tabs.last().map(|(l, _)| l + 1).filter(|b| *b < tabs.len()),
                };
                if released {
                    out.reorder = Some((tab, before));
                } else if let Some((_, r)) = out.tabs.get(slot.min(out.tabs.len().saturating_sub(1))) {
                    crate::widgets::drop_line(ui, *r, slot >= out.tabs.len(), true, &t);
                }
            }
            None if released => out.dropped_out = Some(tab),
            None => out.dragging_out = Some(tab),
        }
    }
    out
}

/// Where a tab dragged to `p` lands: before the `n`th tab on the strip (`tabs.len()`: after the
/// last), or `None` once the pointer has left the strip.
pub fn drop_slot(tabs: &[(usize, Rect)], area: Rect, p: egui::Pos2) -> Option<usize> {
    if !area.expand2(vec2(24.0, area.height())).contains(p) {
        return None;
    }
    Some(tabs.iter().filter(|(_, r)| r.center().x < p.x).count())
}

/// Photoshop-grammar strip (Pro themes): flat tabs on the dark strip; `menu_left` is where the
/// panel menu button starts.
#[allow(clippy::too_many_arguments)]
pub fn pro_tabs(ui: &mut Ui, id: egui::Id, strip: Rect, menu_left: f32, tabs: &[&str], selected: &mut usize, collapsed: bool, reorderable: bool) -> StripOut {
    let t = Tokens::get(ui.ctx());
    let area = Rect::from_min_max(strip.min, pos2(menu_left.max(strip.left()), strip.bottom()));
    let sel = *selected;
    tabs_in(
        ui,
        id,
        area,
        tabs,
        selected,
        egui::FontId::proportional(11.5),
        22.0,
        40.0,
        reorderable,
        |i| i == sel && !collapsed,
        |ui, r, i, galley, resp, active| {
            if active {
                ui.painter().rect_filled(r, CornerRadius { nw: if i == 0 { 3 } else { 0 }, ne: 0, sw: 0, se: 0 }, t.card);
            } else if resp.hovered() {
                ui.painter().rect_filled(r, 0.0, t.hover.gamma_multiply(0.4));
            }
            let color = if active {
                t.text
            } else if resp.hovered() {
                t.text_dim
            } else {
                t.text_faint
            };
            ui.painter().with_clip_rect(r.intersect(ui.clip_rect())).galley_with_override_text_color(r.center() - galley.size() / 2.0, galley, color);
        },
    )
}

/// Studio strip: pill tabs in `area` (the row minus the menu button).
pub fn pill_tabs(ui: &mut Ui, id: egui::Id, area: Rect, tabs: &[&str], selected: &mut usize, reorderable: bool) -> StripOut {
    let t = Tokens::get(ui.ctx());
    let sel = *selected;
    tabs_in(
        ui,
        id,
        area,
        tabs,
        selected,
        theme::medium(12.5),
        20.0,
        44.0,
        reorderable,
        |i| i == sel,
        |ui, r, _, galley, resp, active| {
            // 2 pt between pills, as the old horizontal layout had.
            let r = Rect::from_min_max(r.min, pos2((r.right() - 2.0).max(r.left()), r.bottom()));
            if active {
                crate::widgets::surface(ui, r, t.hover, true);
                if !t.bevel {
                    ui.painter().rect_stroke(r, t.radius_sm, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
                }
            } else if resp.hovered() {
                ui.painter().rect_filled(r, t.radius_sm, t.hover.gamma_multiply(0.6));
            }
            let color = if active { t.text } else { t.text_dim };
            ui.painter().with_clip_rect(r.intersect(ui.clip_rect())).galley_with_override_text_color(r.center() - galley.size() / 2.0, galley, color);
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn total(f: &TabFit) -> f32 {
        f.shown.iter().map(|(_, w)| w).sum()
    }

    #[test]
    fn tabs_that_fit_keep_their_widths() {
        let f = fit(&[60.0, 70.0, 80.0], 0, 400.0, 40.0, 18.0);
        assert_eq!(f.shown, vec![(0, 60.0), (1, 70.0), (2, 80.0)]);
        assert!(f.overflow.is_empty());
    }

    #[test]
    fn narrow_strips_shrink_the_widest_tabs_first() {
        let f = fit(&[50.0, 70.0, 120.0], 0, 200.0, 40.0, 18.0);
        assert!(f.overflow.is_empty());
        assert!((total(&f) - 200.0).abs() < 0.1, "{f:?}");
        assert_eq!(f.shown[0], (0, 50.0), "a tab narrower than the cap keeps its width");
        assert_eq!(f.shown[1], (1, 70.0));
        assert!((f.shown[2].1 - 80.0).abs() < 0.1, "only the widest shrank: {f:?}");
    }

    #[test]
    fn overflow_keeps_the_selected_tab_and_leaves_room_for_the_chevron() {
        let natural = [80.0, 90.0, 85.0, 95.0];
        for sel in 0..4 {
            let f = fit(&natural, sel, 120.0, 40.0, 18.0);
            assert!(f.shown.iter().any(|(i, _)| *i == sel), "{sel}: {f:?}");
            assert!(!f.overflow.is_empty() && !f.overflow.contains(&sel));
            assert!(total(&f) <= 120.0 - 18.0 + 1e-3, "{f:?}");
            assert_eq!(f.shown.len() + f.overflow.len(), 4);
            assert!(f.shown.windows(2).all(|w| w[0].0 < w[1].0), "tab order kept");
        }
    }

    #[test]
    fn hostile_inputs_never_panic_or_go_negative() {
        for avail in [0.0, -10.0, 5.0, f32::NAN, f32::INFINITY] {
            for natural in [&[][..], &[f32::NAN, 50.0][..], &[1e9, -5.0, 30.0][..]] {
                for sel in [0, 1, 99] {
                    let f = fit(natural, sel, avail, 40.0, 18.0);
                    assert!(f.shown.iter().all(|(_, w)| w.is_finite() && *w >= 0.0), "{avail} {natural:?} {f:?}");
                    assert_eq!(f.shown.len() + f.overflow.len(), natural.len());
                }
            }
        }
    }

    #[test]
    fn tiny_panel_strips_keep_tabs_and_overflow_inside_the_available_area() {
        let ctx = egui::Context::default();
        crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::Pro);
        for width in [0.0, 1.0, 5.0, 18.0, 19.0, 80.0, 180.0, 300.0] {
            let area = Rect::from_min_size(pos2(20.0, 20.0), vec2(width, 26.0));
            let mut selected = 3;
            let mut output = ctx.run_ui(egui::RawInput::default(), |ui| {
                let out =
                    pro_tabs(ui, ui.id().with("tiny-tabs"), area, area.right(), &["Color", "Swatches", "Gradients", "Patterns"], &mut selected, false, false);
                assert!(out.tabs.iter().any(|(i, _)| *i == 3));
                for (_, rect) in &out.tabs {
                    assert!(rect.left() >= area.left() && rect.right() <= area.right() + 0.001, "{width}: {rect:?}");
                }
                if let Some(chevron) = out.chevron {
                    assert!(chevron.left() >= area.left() && chevron.right() <= area.right() + 0.001, "{width}: {chevron:?}");
                }
            });
            output.textures_delta.clear();
            assert_eq!(selected, 3);
            for shape in output.shapes.iter().filter(|s| matches!(s.shape, egui::Shape::Text(_))) {
                assert!(shape.clip_rect.right() <= area.right() + 0.001, "{width}: text is clipped to its tab");
            }
        }
    }

    #[test]
    fn overflow_icon_and_hover_paint_stay_inside_a_tiny_button() {
        for scale in [1.0, 2.0] {
            let ctx = egui::Context::default();
            crate::PhotocraftApp::setup_context(&ctx, crate::theme::ThemeKind::Pro);
            ctx.set_pixels_per_point(scale);
            for width in [1.0, 5.0, 11.0, 12.0, 18.0] {
                let rect = Rect::from_min_size(pos2(20.25, 20.25), vec2(width, 26.0));
                for _ in 0..3 {
                    let mut picked = None;
                    let input = egui::RawInput { events: vec![egui::Event::PointerMoved(rect.center())], ..Default::default() };
                    let mut output = ctx.run_ui(input, |ui| {
                        overflow_button(ui, ui.id().with("tiny-overflow"), rect, "More panels", &["Layers"], &[0], &mut picked);
                    });
                    output.textures_delta.clear();
                    let mut icons = 0;
                    for shape in &output.shapes {
                        // egui paints an unrotated image as a textured rectangle, not a mesh.
                        if matches!(&shape.shape, egui::Shape::Rect(rect) if rect.brush.is_some()) {
                            icons += 1;
                        }
                        let painted = shape.shape.visual_bounding_rect().intersect(shape.clip_rect);
                        if painted.is_positive() {
                            assert!(rect.contains_rect(painted), "{scale}x, width {width}: {painted:?}");
                        }
                    }
                    assert!(icons > 0, "the actual SVG icon was painted");
                    assert!(picked.is_none());
                }
            }
        }
    }
}
