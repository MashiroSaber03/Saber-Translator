//! Camera Raw navigation in source-pixel coordinates. No document edits.
use egui::{Key, Modifiers, PointerButton, Pos2, Rect, Response, Sense, Ui, Vec2, vec2};
use serde_json::Value;

use crate::{state::CameraRawPreviewState, widgets};

const MIN_ZOOM: f32 = 0.0001;
const MAX_ZOOM: f32 = 16.0;

impl CameraRawPreviewState {
    pub(crate) fn scale(&self, viewport: Rect, size: Vec2) -> f32 {
        self.zoom
            .filter(|z| z.is_finite() && *z > 0.0)
            .map(|z| z.clamp(MIN_ZOOM, MAX_ZOOM))
            .unwrap_or_else(|| (viewport.width().max(1.0) / size.x.max(f32::EPSILON)).min(viewport.height().max(1.0) / size.y.max(f32::EPSILON)).min(MAX_ZOOM))
    }

    pub(crate) fn image_rect(&mut self, viewport: Rect, size: Vec2) -> Rect {
        let extent = size.max(Vec2::splat(f32::EPSILON)) * self.scale(viewport, size);
        for (center, (image, visible)) in self.center.iter_mut().zip([(extent.x, viewport.width()), (extent.y, viewport.height())]) {
            let margin = (visible.max(1.0) / image.max(1.0) * 0.5).min(0.5);
            *center = if center.is_finite() { center.clamp(margin, 1.0 - margin) } else { 0.5 };
        }
        Rect::from_min_size(viewport.center() - vec2(self.center[0] * extent.x, self.center[1] * extent.y), extent)
    }

    pub(crate) fn zoom_at(&mut self, zoom: f32, at: Pos2, viewport: Rect, size: Vec2) {
        if !zoom.is_finite() || !at.x.is_finite() || !at.y.is_finite() {
            return;
        }
        let old = self.image_rect(viewport, size);
        let point = (at - old.min) / old.size();
        self.zoom = Some(zoom.clamp(MIN_ZOOM, MAX_ZOOM));
        let extent = size.max(Vec2::splat(f32::EPSILON)) * self.scale(viewport, size);
        let center = point - (at - viewport.center()) / extent;
        self.center = [center.x, center.y];
        self.image_rect(viewport, size);
    }

    pub(crate) fn pan(&mut self, delta: Vec2, viewport: Rect, size: Vec2) {
        if !delta.x.is_finite() || !delta.y.is_finite() {
            return;
        }
        let rect = self.image_rect(viewport, size);
        // Leaving fit mode on a hand drag would make later window resizes stop fitting.
        self.center[0] -= delta.x / rect.width();
        self.center[1] -= delta.y / rect.height();
        self.image_rect(viewport, size);
    }

    pub(crate) fn fit(&mut self) {
        self.zoom = None;
        self.center = [0.5, 0.5];
    }
}

/// Validate with the rest of the Camera Raw request before changing anything.
pub(crate) fn prepare(current: CameraRawPreviewState, value: &Value) -> Result<CameraRawPreviewState, String> {
    let obj = value.as_object().ok_or("Camera Raw view must be an object")?;
    let mut next = current;
    for (key, value) in obj {
        match key.as_str() {
            "zoom" if value.as_str() == Some("fit") => next.fit(),
            "zoom" => {
                let z = value
                    .as_f64()
                    .filter(|z| z.is_finite() && (MIN_ZOOM as f64..=MAX_ZOOM as f64).contains(z))
                    .ok_or("Camera Raw zoom must be 'fit' or 0.0001–16 (1 = 100%)")?;
                next.zoom = Some(z as f32);
            }
            "center" => {
                let a = value.as_array().filter(|a| a.len() == 2).ok_or("Camera Raw center must be [u,v]")?;
                let coordinate = |i| {
                    a.get(i)
                        .and_then(Value::as_f64)
                        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
                        .map(|v| v as f32)
                        .ok_or("Camera Raw center must be finite coordinates in 0–1")
                };
                next.center = [coordinate(0)?, coordinate(1)?];
            }
            "hand" => next.hand = value.as_bool().ok_or("Camera Raw hand must be a boolean")?,
            _ => return Err(format!("unknown Camera Raw view property {key}")),
        }
    }
    if obj.get("zoom").and_then(Value::as_str) == Some("fit") {
        next.fit();
    }
    Ok(next)
}

#[derive(Clone, Copy)]
struct ZoomCapture {
    start: Pos2,
    image: Rect,
    box_zoom: bool,
}

/// One capture shared with the colour sampler. Space/middle drag always pans; S takes
/// precedence over the persistent Zoom/Hand tool, so a sampler click never zooms.
pub(crate) fn interact(ui: &mut Ui, state: &mut CameraRawPreviewState, viewport: Rect, size: Vec2, sampler: &mut bool) -> (Rect, Response, bool, Option<Rect>) {
    let sampling = *sampler;
    let response = ui.interact(viewport, ui.id().with("cr-sample-preview"), Sense::click_and_drag());
    let modifiers = ui.input(|i| i.modifiers);
    let temporary_zoom = modifiers.command && (state.hand || sampling);
    let hand = !temporary_zoom && (state.hand && !sampling || ui.input(|i| i.key_down(Key::Space)));
    let zoom_tool = temporary_zoom || !hand && !sampling;
    let panning = hand || response.dragged_by(PointerButton::Middle);
    let at = response.hover_pos().unwrap_or(viewport.center());
    let rect = state.image_rect(viewport, size);
    // Reuse the canvas wheel policy: retain the modifiers of the wheel event while egui
    // smooths it, even if Alt is released before the final part of the notch arrives.
    let wheel = crate::wheel_nav::read(ui.ctx(), false, true);
    if response.hovered() {
        ui.input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
        match wheel {
            Some(crate::wheel_nav::Wheel::Zoom(factor)) => state.zoom_at(state.scale(viewport, size) * factor, at, viewport, size),
            Some(crate::wheel_nav::Wheel::Pan(delta)) => state.pan(delta, viewport, size),
            None => {}
        }
        response.clone().on_hover_cursor(if hand {
            crate::tool_cursor::hand(ui.ctx(), false)
        } else if !zoom_tool {
            egui::CursorIcon::Crosshair
        } else {
            crate::tool_cursor::zoom(ui.ctx(), modifiers.alt)
        });
    }
    let capture_id = response.id.with("zoom-capture");
    if response.drag_started_by(PointerButton::Primary) && zoom_tool {
        let start = ui.input(|i| i.pointer.press_origin()).unwrap_or(at);
        if rect.contains(start) {
            ui.ctx().data_mut(|data| data.insert_temp(capture_id, ZoomCapture { start, image: rect, box_zoom: modifiers.command && !temporary_zoom }));
        }
    }
    let capture = ui.ctx().data(|data| data.get_temp::<ZoomCapture>(capture_id));
    let mut zoom_box = None;
    if panning && (response.dragged_by(PointerButton::Primary) || response.dragged_by(PointerButton::Middle)) {
        state.pan(response.drag_delta(), viewport, size);
        ui.ctx().set_cursor_icon(crate::tool_cursor::hand(ui.ctx(), true));
    } else if response.dragged_by(PointerButton::Primary)
        && let Some(capture) = capture
    {
        if capture.box_zoom {
            if let Some(end) = response.interact_pointer_pos() {
                let region = Rect::from_two_pos(capture.start, end).intersect(viewport);
                zoom_box = Some(region);
            }
        } else {
            // The anchor is the initial press, not the moving pointer; reversing a scrub
            // follows the same detail instead of drifting across the image.
            state.zoom_at(state.scale(viewport, size) * (response.drag_delta().x * 0.01).clamp(-1.0, 1.0).exp(), capture.start, viewport, size);
        }
    } else if response.double_clicked() && hand {
        state.fit();
    } else if response.clicked() && zoom_tool && rect.contains(at) {
        if modifiers.command && modifiers.shift {
            state.zoom_at(1.0, at, viewport, size);
        } else if modifiers.alt {
            state.zoom_at(state.scale(viewport, size) / 1.25, at, viewport, size);
        } else if state.zoom.is_none() {
            state.zoom_at(1.0, at, viewport, size);
        } else {
            state.fit();
        }
    }
    if response.drag_stopped_by(PointerButton::Primary) {
        if let Some(capture) = capture.filter(|c| c.box_zoom)
            && let Some(end) = response.interact_pointer_pos()
        {
            let region = Rect::from_two_pos(capture.start, end).intersect(viewport);
            if region.width() >= 4.0 && region.height() >= 4.0 {
                let point = (region.center() - capture.image.min) / capture.image.size();
                state.zoom = Some(
                    (capture.image.width() / size.x * (viewport.width() / region.width()).min(viewport.height() / region.height())).clamp(MIN_ZOOM, MAX_ZOOM),
                );
                state.center = [point.x, point.y];
            }
        }
        ui.ctx().data_mut(|data| data.remove::<ZoomCapture>(capture_id));
    }
    response.context_menu(|ui| {
        if ui.button(tl!("Fit in View")).clicked() {
            state.fit();
            ui.close();
        }
        if ui.button(tl!("100%")).clicked() {
            state.zoom_at(1.0, at, viewport, size);
            ui.close();
        }
        ui.menu_button(tl!("Zoom Tool"), |ui| {
            for percent in [25.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0] {
                if ui.button(format!("{percent:.0}%")).clicked() {
                    state.zoom_at(percent / 100.0, at, viewport, size);
                    ui.close();
                }
            }
        });
    });
    if !ui.ctx().text_edit_focused() {
        ui.input_mut(|i| {
            if i.consume_key(Modifiers::COMMAND | Modifiers::ALT, Key::Num0) {
                state.zoom_at(1.0, viewport.center(), viewport, size);
            } else if i.consume_key(Modifiers::COMMAND, Key::Num0) {
                state.fit();
            }
            if i.consume_key(Modifiers::COMMAND, Key::Plus) || i.consume_key(Modifiers::COMMAND, Key::Equals) {
                state.zoom_at(state.scale(viewport, size) * 1.25, viewport.center(), viewport, size);
            }
            if i.consume_key(Modifiers::COMMAND, Key::Minus) {
                state.zoom_at(state.scale(viewport, size) / 1.25, viewport.center(), viewport, size);
            }
            if i.consume_key(Modifiers::NONE, Key::H) {
                state.hand = true;
                *sampler = false;
            }
            if i.consume_key(Modifiers::NONE, Key::Z) {
                state.hand = false;
                *sampler = false;
            }
        });
    }
    (state.image_rect(viewport, size), response, panning || temporary_zoom, zoom_box)
}

pub(crate) fn toolbar(ui: &mut Ui, state: &mut CameraRawPreviewState, viewport: Rect, size: Vec2, sampler: &mut bool) {
    let zoom_tool = crate::icons::button(ui, "zoom-in", 28.0, !state.hand && !*sampler, tl!("Zoom Tool"));
    zoom_tool.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Zoom Tool")));
    if zoom_tool.double_clicked() {
        state.fit();
    }
    if zoom_tool.clicked() {
        state.hand = false;
        *sampler = false;
    }
    let hand_tool = crate::icons::button(ui, "hand", 28.0, state.hand && !*sampler, tl!("Hand Tool"));
    hand_tool.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), tl!("Hand Tool")));
    if hand_tool.double_clicked() {
        state.fit();
    }
    if hand_tool.clicked() {
        state.hand = true;
        *sampler = false;
    }
    if widgets::secondary_button(ui, tl!("Fit in View"), 84.0).clicked() {
        state.fit();
    }
    if widgets::secondary_button(ui, tl!("100%"), 52.0).clicked() {
        state.zoom_at(1.0, viewport.center(), viewport, size);
    }
    let mut zoom = state.zoom;
    let mut options = vec![(None, tl!("Fit in View"))];
    let labels = ["25%", "50%", "100%", "200%", "400%", "800%", "1600%"];
    for (z, label) in [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0].into_iter().zip(labels) {
        options.push((Some(z), label));
    }
    let current_label = format!("{:.1}%", state.scale(viewport, size) * 100.0);
    if !options.iter().any(|(z, _)| *z == zoom) {
        options.push((zoom, &current_label));
    }
    if widgets::dropdown(ui, "cr-preview-zoom", &mut zoom, &options, 80.0) {
        match zoom {
            Some(z) => state.zoom_at(z, viewport.center(), viewport, size),
            None => state.fit(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::pos2;
    #[test]
    fn zoom_preserves_the_point_under_the_pointer_and_pan_clamps() {
        let viewport = Rect::from_min_size(pos2(10.0, 20.0), vec2(800.0, 600.0));
        let size = vec2(1600.0, 1200.0);
        let mut view = CameraRawPreviewState::default();
        let at = pos2(310.0, 270.0);
        let initial = view.image_rect(viewport, size);
        let point = (at - initial.min) / initial.size();
        view.zoom_at(1.0, at, viewport, size);
        let enlarged = view.image_rect(viewport, size);
        assert!(((at - enlarged.min) / enlarged.size() - point).length() < 1e-5);
        view.pan(vec2(1e9, -1e9), viewport, size);
        let panned = view.image_rect(viewport, size);
        assert!(panned.contains_rect(viewport));
        view.fit();
        assert_eq!(view.image_rect(viewport, size), initial);
    }
    #[test]
    fn tiny_images_nonfinite_state_and_resize_stay_centered_and_finite() {
        let viewport = Rect::from_min_size(Pos2::ZERO, vec2(600.0, 400.0));
        let size = vec2(40.0, 20.0);
        let mut view = CameraRawPreviewState { zoom: Some(f32::NAN), center: [f32::INFINITY, f32::NAN], hand: true };
        assert!(view.image_rect(viewport, size).is_finite());
        view.zoom_at(1.0, pos2(100.0, 100.0), viewport, size);
        assert_eq!(view.image_rect(viewport, size).center(), viewport.center());
        let before = view;
        view.zoom_at(f32::INFINITY, viewport.center(), viewport, size);
        view.pan(vec2(f32::NAN, 1.0), viewport, size);
        assert_eq!(view, before);
    }
}
