//! Explicit, one-shot desktop sampling. Snapshots remain in memory until selection/cancellation.
use crate::PhotocraftApp;
use egui::{Color32, Context, Id, Pos2, Rect, Sense, Stroke, StrokeKind, TextureOptions, vec2};
use std::sync::{
    Arc, Mutex,
    mpsc::{Receiver, TryRecvError},
};

/// Display-encoded RGBA8 pixels, native window points (before UI zoom), and optional source ICC.
#[derive(Clone)]
pub struct ScreenImage {
    /// Native monitor index (winit order); places the picker on that exact output.
    pub monitor: Option<usize>,
    pub position: Pos2,
    pub size: egui::Vec2,
    pub width: usize,
    pub height: usize,
    pub rgba: Vec<u8>,
    pub profile: Option<Vec<u8>>,
}
pub enum Capture {
    Images(Vec<ScreenImage>),
    /// Native portal/browser pickers return sRGB; None means cancellation.
    Color(Option<[f32; 3]>),
}
/// Dropping a pending request also cancels portal/browser work.
pub struct Pending {
    pub receiver: Receiver<Result<Capture, String>>,
    pub cancelled: Arc<std::sync::atomic::AtomicBool>,
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}
pub type Service = Box<dyn FnMut(&Context) -> Pending>;
pub const MAX_PIXELS: usize = 64 * 1024 * 1024;
impl ScreenImage {
    pub fn validate(&self) -> Result<(), String> {
        let count = self.width.checked_mul(self.height).filter(|&n| n > 0 && n <= MAX_PIXELS).ok_or("Screen image is too large or empty")?;
        if count.checked_mul(4) != Some(self.rgba.len())
            || !self.position.is_finite()
            || !self.size.is_finite()
            || self.size.min_elem() <= 0.0
            || self.size.max_elem() > 65535.0
            || self.position.to_vec2().abs().max_elem() > 1_000_000.0
            || self.profile.as_ref().is_some_and(|p| p.len() > 1024 * 1024)
        {
            return Err("Invalid screen image geometry".into());
        }
        Ok(())
    }
    fn pixel(&self, x: usize, y: usize) -> Option<Color32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let start = y.checked_mul(self.width)?.checked_add(x)?.checked_mul(4)?;
        let p = self.rgba.get(start..start.checked_add(4)?)?;
        Some(Color32::from_rgb(*p.first()?, *p.get(1)?, *p.get(2)?))
    }
    fn at(&self, rect: Rect, pos: Pos2) -> Option<(usize, usize)> {
        if !rect.contains(pos) || !pos.is_finite() || rect.size().min_elem() <= 0.0 {
            return None;
        }
        let p = (pos - rect.min) / rect.size();
        Some((((p.x * self.width as f32) as usize).min(self.width.saturating_sub(1)), ((p.y * self.height as f32) as usize).min(self.height.saturating_sub(1))))
    }
    fn conversion(&self) -> Result<Option<Arc<photocraft_cms::Transform>>, String> {
        let Some(bytes) = &self.profile else { return Ok(None) };
        let source = photocraft_cms::Profile::parse(bytes).map_err(|e| e.to_string())?;
        if source.color_space != photocraft_cms::ColorSpace::Rgb {
            return Err("Screen profile must describe RGB pixels".into());
        }
        photocraft_cms::Transform::new(&source, photocraft_cms::Builtin::Srgb.profile(), photocraft_cms::Intent::RelativeColorimetric, true)
            .map(|t| Some(Arc::new(t)))
            .map_err(|e| e.to_string())
    }
    fn sample(&self, x: usize, y: usize, conversion: Option<&photocraft_cms::Transform>) -> Result<[f32; 3], String> {
        let p = self.pixel(x, y).ok_or("Screen pixel is out of bounds")?;
        let mut rgb = [f32::from(p.r()) / 255.0, f32::from(p.g()) / 255.0, f32::from(p.b()) / 255.0];
        if let Some(t) = conversion {
            t.apply(&mut rgb, 3);
        }
        Ok(rgb.map(|v| v.clamp(0.0, 1.0)))
    }
}
#[derive(Clone)]
struct Snapshot {
    image: Arc<ScreenImage>,
    texture: egui::TextureHandle,
    conversion: Option<Arc<photocraft_cms::Transform>>,
}
type Target = (Option<photocraft_doc::DocId>, Option<photocraft_doc::LayerId>, Option<u64>);
#[derive(Default)]
struct Runtime {
    request: Option<Id>,
    owner: Option<Id>,
    pending: Option<Pending>,
    images: Vec<Snapshot>,
    answer: Option<Result<Option<[f32; 3]>, String>>,
    completed: Option<(Id, [f32; 3])>,
    zoom: usize,
    generation: u64,
    target: Option<Target>,
}
type Shared = Arc<Mutex<Runtime>>;
fn runtime(ctx: &Context) -> Shared {
    let id = Id::new("screen-color-picker-runtime");
    ctx.data_mut(|d| {
        if let Some(r) = d.get_temp::<Shared>(id) {
            return r;
        }
        let r = Arc::new(Mutex::new(Runtime::default()));
        d.insert_temp(id, r.clone());
        r
    })
}
fn lock(r: &Shared) -> std::sync::MutexGuard<'_, Runtime> {
    r.lock().unwrap_or_else(|e| e.into_inner())
}
/// Consume a result only at its initiating widget, including when its compact popup closed.
pub fn take(ctx: &Context, id: Id) -> Option<[f32; 3]> {
    let r = runtime(ctx);
    let mut s = lock(&r);
    if s.completed.as_ref().is_some_and(|(owner, _)| *owner == id) { s.completed.take().map(|(_, rgb)| rgb) } else { None }
}
/// Same action for full dialogs and compact swatches. Cancellation returns no change.
pub fn button(ui: &mut egui::Ui, id: Id) -> Option<[f32; 3]> {
    let r = runtime(ui.ctx());
    let available = ui.ctx().data(|d| d.get_temp::<bool>(Id::new("screen-color-picker-available"))).unwrap_or(false);
    let mut state = lock(&r);
    let result = if state.completed.as_ref().is_some_and(|(owner, _)| *owner == id) { state.completed.take().map(|(_, rgb)| rgb) } else { None };
    let busy = state.owner.is_some() || state.request.is_some();
    if ui.add_enabled(available && !busy, egui::Button::new(tl!("Pick screen color"))).on_hover_text(tl!("Sample a pixel anywhere on the screen")).clicked() {
        state.request = Some(id);
        ui.ctx().request_repaint();
    }
    result
}
fn target(app: &PhotocraftApp) -> Target {
    (app.session.active().map(|s| s.doc.id), app.session.active().and_then(|s| s.active_layer), app.ui.dialogs.last().map(|d| d.id))
}
/// Poll before shortcuts. A changed editing target invalidates a pending pick.
pub fn tick(app: &mut PhotocraftApp, ctx: &Context) -> bool {
    ctx.data_mut(|d| d.insert_temp(Id::new("screen-color-picker-available"), app.services.screen_pick.is_some()));
    let r = runtime(ctx);
    let mut s = lock(&r);
    if s.completed.is_some() && s.target != Some(target(app)) {
        s.completed = None;
    }
    if let Some(id) = s.request.take() {
        s.completed = None;
        if let Some(service) = &mut app.services.screen_pick {
            s.pending = Some(service(ctx));
            s.owner = Some(id);
            s.target = Some(target(app));
            s.zoom = 12;
            s.generation = s.generation.wrapping_add(1);
        }
    }
    if s.owner.is_some() {
        if s.target != Some(target(app))
            || ctx.input(|i| i.viewport().close_requested())
            || ctx.input_mut(|i| i.consume_key(egui::Modifiers::NONE, egui::Key::Escape))
        {
            s.answer = Some(Ok(None));
        }
        if let Some(pending) = s.pending.as_ref().filter(|_| s.answer.is_none()) {
            match pending.receiver.try_recv() {
                Ok(Ok(Capture::Color(c))) => {
                    s.pending = None;
                    s.answer = Some(Ok(c));
                }
                Ok(Ok(Capture::Images(images))) => {
                    s.pending = None;
                    let total = images.iter().try_fold(0usize, |n, i| n.checked_add(i.width.checked_mul(i.height)?));
                    let valid =
                        !images.is_empty() && images.len() <= 16 && total.is_some_and(|n| n <= MAX_PIXELS) && images.iter().all(|i| i.validate().is_ok());
                    if valid {
                        let max = ctx.input(|i| i.max_texture_side);
                        if images.iter().any(|i| i.width > max || i.height > max) {
                            s.answer = Some(Err("This display exceeds the GPU's screen picker texture limit".into()));
                        } else {
                            s.images = images
                                .into_iter()
                                .map(|i| {
                                    let conversion = i.conversion()?;
                                    let texture = ctx.load_texture(
                                        "screen-color-snapshot",
                                        egui::ColorImage::from_rgba_unmultiplied([i.width, i.height], &i.rgba),
                                        TextureOptions::NEAREST,
                                    );
                                    Ok(Snapshot { image: Arc::new(i), texture, conversion })
                                })
                                .collect::<Result<Vec<_>, String>>()
                                .unwrap_or_else(|e| {
                                    s.answer = Some(Err(e));
                                    Vec::new()
                                });
                        }
                    } else {
                        s.answer = Some(Err("Invalid or oversized screen capture".into()));
                    }
                }
                Ok(Err(e)) => {
                    s.pending = None;
                    s.answer = Some(Err(e));
                }
                Err(TryRecvError::Disconnected) => {
                    s.pending = None;
                    s.answer = Some(Err("Screen color picker disconnected".into()));
                }
                Err(TryRecvError::Empty) => {}
            }
        }
        if let Some(answer) = s.answer.take() {
            let owner = s.owner.take();
            s.pending = None;
            for n in 0..s.images.len() {
                ctx.send_viewport_cmd_to(egui::ViewportId::from_hash_of(("screen-color", s.generation, n)), egui::ViewportCommand::Close);
            }
            s.images.clear();
            match answer {
                Ok(Some(rgb)) if rgb.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) => s.completed = owner.map(|id| (id, rgb)),
                Err(e) => {
                    app.ui.status = e;
                    app.ui.status_error = true;
                }
                _ => {}
            }
        }
    }
    let busy = s.owner.is_some();
    if busy {
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
    busy
}
pub fn showing(ctx: &Context) -> bool {
    let r = runtime(ctx);
    !lock(&r).images.is_empty()
}
pub fn busy(ctx: &Context) -> bool {
    let r = runtime(ctx);
    let s = lock(&r);
    s.owner.is_some() || s.request.is_some()
}
pub fn cancel(ctx: &Context) {
    let r = runtime(ctx);
    lock(&r).answer = Some(Ok(None));
    ctx.request_repaint();
}

/// The picker window: hidden until its snapshot has been painted (#3210).
fn viewport_builder(ctx: &Context, image: &ScreenImage) -> egui::ViewportBuilder {
    let mut builder = egui::ViewportBuilder::default()
        .with_title(tl!("PhotoCraft — Screen Color Picker"))
        .with_decorations(false)
        .with_resizable(false)
        .with_position(image.position / ctx.zoom_factor())
        .with_inner_size(image.size / ctx.zoom_factor())
        // Not `AlwaysOnTop` here: a monitor-sized window created topmost makes the DX12 backend
        // fail its swapchain creation (0x887A0001) and the app panics. The level is raised once
        // the first frame has been painted, in `show`.
        // Do not expose the native window's empty first frame over the desktop.
        // The viewport is made visible after its snapshot has been painted below.
        .with_visible(false)
        .with_clamp_size_to_monitor_size(false);
    if let Some(monitor) = image.monitor {
        builder = builder.with_monitor(monitor);
    }
    builder
}
pub fn show(ctx: &Context) {
    let r = runtime(ctx);
    let s = lock(&r);
    let images = s.images.clone();
    let generation = s.generation;
    drop(s);
    for (n, snapshot) in images.into_iter().enumerate() {
        let Snapshot { image, texture, conversion } = snapshot;
        let shared = r.clone();
        let vid = egui::ViewportId::from_hash_of(("screen-color", generation, n));
        let builder = viewport_builder(ctx, &image);
        // Immediate, so it paints inside the main window's own frame: as a deferred viewport the
        // main window stopped being redrawn on Windows (OpenGL) once the picker was up, so a click
        // or Esc was never turned into an answer and the overlay could not be left.
        ctx.show_viewport_immediate(vid, builder, move |ui, _| {
            {
                let s = lock(&shared);
                if s.generation != generation || s.owner.is_none() {
                    ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    return;
                }
            }
            frame(ui, &shared, &image, &texture, conversion.as_deref());
            // Showing a new native viewport before this frame is painted can briefly cover
            // the screen with the window's default white clear color.
            ui.ctx().send_viewport_cmd(egui::ViewportCommand::Visible(true));
            // Once per picker window (`show` runs every frame, so a local flag would resend it):
            // raise it, and take the keyboard so Esc reaches the picker: a window shown later
            // isn't always given focus by the OS, and the picker covers the screen (#4291).
            let raised = Id::new(("screen-picker-raised", ui.ctx().viewport_id()));
            if !ui.ctx().data_mut(|d| std::mem::replace(d.get_temp_mut_or_default::<bool>(raised), true)) {
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::WindowLevel(egui::WindowLevel::AlwaysOnTop));
                ui.ctx().send_viewport_cmd(egui::ViewportCommand::Focus);
            }
        });
    }
}

/// What the picker resolves to: the colour from a pick, or `Ok(None)` to cancel.
type Answer = Result<Option<[f32; 3]>, String>;

/// One frame of a picker window over its snapshot `image`. The picker covers the whole screen
/// and stays on top, so every way out must work (#4291): Esc, closing the window, a right-click,
/// and a left click (or a press and release that moved) anywhere, which picks the pixel under
/// it, or cancels when there's none to sample. Nothing it gets leaves the user stuck in it.
fn frame(ui: &mut egui::Ui, shared: &Shared, image: &ScreenImage, texture: &egui::TextureHandle, conversion: Option<&photocraft_cms::Transform>) {
    let ctx = ui.ctx().clone();
    let answer = |a: Answer| {
        lock(shared).answer = Some(a);
        ctx.request_repaint_of(egui::ViewportId::ROOT);
    };
    if ctx.input(|i| i.viewport().close_requested() || i.key_pressed(egui::Key::Escape) || i.pointer.secondary_released()) {
        answer(Ok(None));
        return;
    }
    egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
        let rect = ui.max_rect();
        ui.painter().image(texture.id(), rect, Rect::from_min_max(Pos2::ZERO, egui::pos2(1.0, 1.0)), Color32::WHITE);
        let response = ui.interact(rect, ui.id().with("screen-sample"), Sense::click_and_drag());
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
        let released = response.clicked() || response.drag_stopped();
        let pixel = ui.ctx().pointer_latest_pos().and_then(|pos| Some((pos, image.at(rect, pos)?)));
        let Some((pos, (x, y))) = pixel else {
            // No pixel under the pointer to sample: a click still ends the picker.
            if released {
                answer(Ok(None));
            }
            return;
        };
        let zoom = {
            let delta = ui.input(|i| {
                i.raw
                    .events
                    .iter()
                    .filter_map(|e| match e {
                        egui::Event::MouseWheel { delta, .. } => Some(delta.y),
                        _ => None,
                    })
                    .sum::<f32>()
            });
            let mut s = lock(shared);
            if delta > 0.0 {
                s.zoom = (s.zoom + 2).min(24);
            }
            if delta < 0.0 {
                s.zoom = s.zoom.saturating_sub(2).max(4);
            }
            s.zoom
        };
        let picked = image.sample(x, y, conversion);
        loupe(ui, image, rect, pos, (x, y), zoom, picked.as_ref().ok().copied());
        if released {
            answer(picked.map(Some));
        }
    });
}
fn loupe(ui: &egui::Ui, image: &ScreenImage, screen: Rect, pos: Pos2, pixel: (usize, usize), zoom: usize, rgb: Option<[f32; 3]>) {
    let (x, y) = pixel;
    let t = crate::theme::Tokens::get(ui.ctx());
    // The label describes physical pixel magnification, including Retina and UI zoom.
    let cell = zoom as f32 / ui.ctx().pixels_per_point();
    let p = ui.painter();
    let pixel = image.pixel(x, y).unwrap_or(Color32::BLACK);
    let color = rgb.map(crate::color_picker_ui::hex).unwrap_or_else(|| format!("#{:02x}{:02x}{:02x}", pixel.r(), pixel.g(), pixel.b()));
    let label = p.layout_no_wrap(format!("{}   {}×", color.to_uppercase(), zoom), egui::FontId::monospace(12.0), t.text);
    let hint = p.layout_no_wrap(tl!("Click to pick · Esc to cancel").to_owned(), egui::FontId::proportional(11.0), t.text);
    let wheel = p.layout_no_wrap(tl!("Scroll to zoom").to_owned(), egui::FontId::proportional(11.0), t.text);
    let width = (11.0 * cell + 12.0).max(label.size().x + 16.0).max(hint.size().x + 16.0).max(wheel.size().x + 16.0);
    let size = vec2(width, 11.0 * cell + 76.0);
    let desired = pos + vec2(24.0, 24.0);
    let at = egui::pos2(desired.x.min(screen.right() - size.x).max(screen.left()), desired.y.min(screen.bottom() - size.y).max(screen.top()));
    let area = Rect::from_min_size(at, size);
    p.rect_filled(area, t.radius, t.card);
    p.rect_stroke(area, t.radius, Stroke::new(1.0, t.field_border), StrokeKind::Inside);
    let origin = at + vec2((width - 11.0 * cell) / 2.0, 6.0);
    for row in 0..11usize {
        for col in 0..11usize {
            let px = x.checked_add(col).and_then(|v| v.checked_sub(5));
            let py = y.checked_add(row).and_then(|v| v.checked_sub(5));
            let color = px.zip(py).and_then(|(x, y)| image.pixel(x, y)).unwrap_or(t.canvas);
            p.rect_filled(Rect::from_min_size(origin + vec2(col as f32 * cell, row as f32 * cell), vec2(cell, cell)), 0.0, color);
        }
    }
    let center = Rect::from_min_size(origin + vec2(5.0 * cell, 5.0 * cell), vec2(cell, cell));
    p.rect_stroke(center.expand(1.0), 0.0, Stroke::new(1.0, Color32::BLACK), StrokeKind::Inside);
    p.rect_stroke(center, 0.0, Stroke::new(1.0, Color32::WHITE), StrokeKind::Inside);
    let footer = at + vec2(8.0, 11.0 * cell + 14.0);
    p.galley(footer, label, t.text);
    p.galley(footer + vec2(0.0, 20.0), hint, t.text);
    p.galley(footer + vec2(0.0, 36.0), wheel, t.text);
}

#[cfg(test)]
mod tests {
    use super::*;
    fn image() -> ScreenImage {
        ScreenImage {
            monitor: None,
            position: egui::pos2(-100.0, 0.0),
            size: vec2(2.0, 1.0),
            width: 2,
            height: 1,
            rgba: vec![255, 0, 0, 255, 0, 128, 255, 255],
            profile: None,
        }
    }
    #[test]
    fn screen_picker_viewport_starts_hidden_until_its_snapshot_is_painted() {
        let builder = viewport_builder(&Context::default(), &image());
        assert_eq!(builder.visible, Some(false));
    }
    #[test]
    fn screen_picker_viewport_is_not_created_always_on_top() {
        // Creating the monitor-sized window topmost fails DX12 swapchain creation (Windows).
        let builder = viewport_builder(&Context::default(), &image());
        assert_eq!(builder.window_level, None);
    }
    #[test]
    fn pixel_mapping_handles_display_edges_negative_origins_and_hidpi() {
        let i = image();
        let rect = Rect::from_min_size(i.position, i.size);
        assert_eq!(i.at(rect, egui::pos2(-100.0, 0.0)), Some((0, 0)));
        assert_eq!(i.at(rect, egui::pos2(-98.0, 1.0)), Some((1, 0)));
        assert_eq!(i.at(rect, egui::pos2(-97.0, 0.0)), None);
        assert_eq!(i.at(rect, egui::pos2(f32::NAN, 0.0)), None);
        let scaled = Rect::from_min_size(Pos2::ZERO, vec2(4.0, 2.0));
        assert_eq!(i.at(scaled, egui::pos2(2.5, 1.0)), Some((1, 0)));
        assert_eq!(i.pixel(2, 0), None);
        assert_eq!(i.sample(1, 0, i.conversion().unwrap().as_deref()).unwrap(), [0.0, 128.0 / 255.0, 1.0]);
    }
    #[test]
    fn malformed_images_and_non_rgb_profiles_fail_before_sampling() {
        let mut i = image();
        assert!(i.validate().is_ok());
        i.rgba.pop();
        assert!(i.validate().is_err());
        i.width = usize::MAX;
        i.height = 2;
        assert!(i.validate().is_err());
        let mut i = image();
        i.size.x = f32::NAN;
        assert!(i.validate().is_err());
        let mut i = image();
        i.profile = Some(photocraft_cms::Builtin::LabD50.profile().to_bytes().as_ref().clone());
        assert!(i.conversion().is_err());
    }
    /// The picker window's frame over [`image`] (red | azure) filling a 200 × 100 window, with its
    /// own runtime: what the frame answered after `act`.
    fn picker_answer(act: impl FnOnce(&mut egui_kittest::Harness<'_>)) -> Option<Answer> {
        let shared: Shared = Arc::default();
        let inner = shared.clone();
        let texture: std::cell::OnceCell<egui::TextureHandle> = std::cell::OnceCell::new();
        let image = image();
        let mut h = egui_kittest::Harness::builder().with_size(vec2(200.0, 100.0)).build_ui(move |ui| {
            let tex = texture.get_or_init(|| {
                ui.ctx().load_texture("test-snapshot", egui::ColorImage::new([2, 1], vec![Color32::RED, Color32::BLUE]), TextureOptions::NEAREST)
            });
            frame(ui, &inner, &image, tex, None);
        });
        h.run();
        act(&mut h);
        h.run();
        lock(&shared).answer.take()
    }
    /// #4291: the picker covers the screen and stays on top, so every input ends it: Esc, a
    /// right-click, a click (the pixel under it), a press dragged before its release (the pixel
    /// where it's released).
    #[test]
    fn every_way_out_of_the_picker_ends_it() {
        let left = egui::pos2(40.0, 50.0);
        let right = egui::pos2(160.0, 50.0);
        assert_eq!(picker_answer(|h| h.hover_at(left)), None, "hovering alone keeps it open");
        assert_eq!(
            picker_answer(|h| {
                h.hover_at(left);
                h.run();
                h.drag_at(left);
                h.run();
                h.drop_at(left);
            }),
            Some(Ok(Some([1.0, 0.0, 0.0]))),
            "a click picks the pixel under it"
        );
        assert_eq!(
            picker_answer(|h| {
                h.drag_at(left);
                h.run();
                h.hover_at(right);
                h.run();
                h.drop_at(right);
            }),
            Some(Ok(Some([0.0, 128.0 / 255.0, 1.0]))),
            "a press dragged across picks where it's released, instead of being ignored"
        );
        assert_eq!(
            picker_answer(|h| {
                h.hover_at(left);
                h.run();
                h.event(egui::Event::PointerButton { pos: left, button: egui::PointerButton::Secondary, pressed: true, modifiers: Default::default() });
                h.run();
                h.event(egui::Event::PointerButton { pos: left, button: egui::PointerButton::Secondary, pressed: false, modifiers: Default::default() });
            }),
            Some(Ok(None)),
            "a right-click cancels"
        );
        assert_eq!(picker_answer(|h| h.key_press(egui::Key::Escape)), Some(Ok(None)), "Esc cancels");
    }
    fn app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", serde_json::json!({"width":8,"height":8})).unwrap();
        app
    }
    fn queued(app: &mut PhotocraftApp, ctx: &Context) -> (Id, std::sync::mpsc::Sender<Result<Capture, String>>, Arc<std::sync::atomic::AtomicBool>) {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let stop = cancelled.clone();
        let mut receiver = Some(rx);
        app.services.screen_pick = Some(Box::new(move |_| Pending { receiver: receiver.take().unwrap(), cancelled: stop.clone() }));
        let owner = Id::new("test-destination");
        lock(&runtime(ctx)).request = Some(owner);
        assert!(tick(app, ctx));
        (owner, tx, cancelled)
    }
    #[test]
    fn cancellation_wins_over_a_simultaneously_completed_pick() {
        let mut app = app();
        let ctx = Context::default();
        let (id, tx, cancelled) = queued(&mut app, &ctx);
        tx.send(Ok(Capture::Color(Some([1.0, 0.0, 0.0])))).unwrap();
        cancel(&ctx);
        assert!(!tick(&mut app, &ctx));
        assert_eq!(take(&ctx, id), None);
        assert!(cancelled.load(std::sync::atomic::Ordering::Relaxed));
    }
    #[test]
    fn changed_document_discards_the_pending_result_and_replies_route_once() {
        let mut app = app();
        let ctx = Context::default();
        let (id, tx, _) = queued(&mut app, &ctx);
        tx.send(Ok(Capture::Color(Some([1.0, 0.0, 0.0])))).unwrap();
        app.run("file.new", serde_json::json!({"width":8,"height":8})).unwrap();
        assert!(!tick(&mut app, &ctx));
        assert_eq!(take(&ctx, id), None);
        let (id, tx, _) = queued(&mut app, &ctx);
        tx.send(Ok(Capture::Color(Some([0.25, 0.5, 1.0])))).unwrap();
        assert!(!tick(&mut app, &ctx));
        assert_eq!(take(&ctx, Id::new("different-widget")), None);
        assert_eq!(take(&ctx, id), Some([0.25, 0.5, 1.0]));
        assert_eq!(take(&ctx, id), None);
    }
    #[test]
    fn disconnected_service_and_invalid_portal_colors_leave_the_document_unchanged() {
        let mut app = app();
        let ctx = Context::default();
        let foreground = app.session.tools.foreground;
        let (id, tx, _) = queued(&mut app, &ctx);
        drop(tx);
        assert!(!tick(&mut app, &ctx));
        assert!(app.ui.status_error);
        assert_eq!(take(&ctx, id), None);
        let (id, tx, _) = queued(&mut app, &ctx);
        tx.send(Ok(Capture::Color(Some([f32::NAN, 0.0, 1.0])))).unwrap();
        assert!(!tick(&mut app, &ctx));
        assert_eq!(take(&ctx, id), None);
        assert_eq!(app.session.tools.foreground, foreground);
    }
}
