//! Canvas cursors must move with the OS pointer, independently of the canvas frame rate: the brush
//! outline and crosshairs are OS cursor bitmaps on every desktop platform (Windows, macOS, X11 and
//! Wayland), so they track the hardware pointer instead of trailing it by the app's render,
//! present and compositor latency. A painted outline lagged the pointer by several refreshes on
//! Linux and macOS and "jumped to catch up" on slow frames (#1343). Windows' stock crosshair
//! inverts grey into grey, so the crosshairs are our black/white bitmap there too.

use std::sync::Arc;

use egui::{CursorIcon, CustomCursorImage, Painter, Pos2, Stroke, Vec2, vec2};

mod hand;
mod zoom;

/// The largest cursor bitmap (physical pixels; points on macOS, see [`native_scale`]); a larger tip
/// paints its outline instead, with a small OS crosshair at the hotspot, and that outline trails
/// the pointer by a frame or two (26 ms on a 120 Hz Mac, #1343). Windows takes winit's limit, and
/// so does macOS: AppKit sets no size limit on an `NSCursor` image. On Linux, X servers and
/// hardware cursor planes commonly cap cursors at 256 px and fall back to slow software cursors (or
/// reject them) past that, so stay within it there (#968 used 128). Either way it keeps allocations
/// bounded for hostile brush sizes or zooms.
const MAX_SIDE: u16 = if cfg!(any(target_os = "windows", target_os = "macos")) { 2048 } else { 255 };

/// What a bitmap cursor shows. `Hand` is the Hand tool's open hand, or its fist while it drags;
/// `Zoom` is the Zoom tool's magnifier, with a minus when it zooms out. `Outline` is a brush tip
/// outline: closed rings of logical-point offsets from the hotspot.
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    Circle { radius: f32, centre: bool },
    Crosshair { length: f32, gap: f32 },
    Hand { closed: bool },
    Zoom { out: bool },
    Outline { rings: Arc<[Vec<Vec2>]>, centre: bool },
}

#[derive(Clone)]
struct Cached {
    shape: Shape,
    scale: f32,
    image: CustomCursorImage,
}

/// The image is sticky in egui's platform output. Clear it every pass, including when a
/// document closes or the pointer leaves the canvas; a later widget's cursor takes precedence.
pub(crate) struct CursorLifecycle;

impl egui::plugin::Plugin for CursorLifecycle {
    fn debug_name(&self) -> &'static str {
        "PhotoCraft tool cursor"
    }

    fn on_begin_pass(&mut self, ui: &mut egui::Ui) {
        ui.ctx().set_cursor_image(None);
    }

    fn on_end_pass(&mut self, ui: &mut egui::Ui) {
        ui.ctx().output_mut(|o| {
            if o.cursor_image.is_some() {
                if o.cursor_icon == CursorIcon::None {
                    // None marks our request until all widgets have had their say. A backend
                    // that cannot upload the bitmap must still leave a visible pointer.
                    o.cursor_icon = CursorIcon::Crosshair;
                } else {
                    o.cursor_image = None;
                }
            }
        });
    }
}

pub(crate) fn circle(painter: &Painter, at: Pos2, radius: f32, centre: bool) -> CursorIcon {
    show(painter, at, Shape::Circle { radius, centre })
}

pub(crate) fn crosshair(painter: &Painter, at: Pos2, length: f32, gap: f32) -> CursorIcon {
    show(painter, at, Shape::Crosshair { length, gap })
}

pub(crate) fn outline(painter: &Painter, at: Pos2, rings: Vec<Vec<Vec2>>, centre: bool) -> CursorIcon {
    show(painter, at, Shape::Outline { rings: rings.into(), centre })
}

/// The Pen's pointer over the anchor a click would close the path on (#4596): the centre cross
/// with a small ring below right, as Photoshop's pen shows a small circle there.
pub(crate) fn close_path(painter: &Painter, at: Pos2) -> CursorIcon {
    outline(painter, at, vec![close_ring()], true)
}

fn close_ring() -> Vec<Vec2> {
    (0..16).map(|i| (i as f32 / 16.0 * std::f32::consts::TAU, vec2(9.0, 9.0))).map(|(a, c)| c + vec2(a.cos(), a.sin()) * 3.5).collect()
}

/// Can this pass show an OS cursor bitmap? eframe's immediate viewports and its web integration
/// don't upload cursor images; every desktop root viewport does (winit custom cursors).
fn native(ctx: &egui::Context) -> bool {
    cfg!(not(target_arch = "wasm32")) && ctx.viewport_id() == egui::ViewportId::ROOT
}

/// Bitmap pixels per egui point. winit takes an `NSCursor` image's dimensions as logical points on
/// macOS, so there the bitmap scales only with the UI zoom (the monitor's density would enlarge it
/// a second time on Retina displays); Windows, X11 and Wayland take physical pixels. (The macOS
/// sizing is from @jerrykimo's #968.)
pub(crate) fn native_scale(ctx: &egui::Context) -> f32 {
    if cfg!(target_os = "macos") { ctx.zoom_factor() } else { ctx.pixels_per_point() }
}

/// The Hand tool's pointer: an open hand, a fist while it drags (`closed`). winit maps `Grab` and
/// `Grabbing` to the four-arrow move cursor on Windows, so there it is a bitmap; macOS and the
/// Linux themes already draw hands, and the OS moves those without waiting for a frame.
pub(crate) fn hand(ctx: &egui::Context, closed: bool) -> CursorIcon {
    hand_with(ctx, closed, cfg!(target_os = "windows") && ctx.viewport_id() == egui::ViewportId::ROOT)
}

fn hand_with(ctx: &egui::Context, closed: bool, bitmap: bool) -> CursorIcon {
    if bitmap && let Some(image) = cached_image(ctx, &Shape::Hand { closed }, native_scale(ctx)) {
        ctx.set_cursor_image(Some(image));
        return CursorIcon::None;
    }
    if closed { CursorIcon::Grabbing } else { CursorIcon::Grab }
}

/// The Zoom tool's pointer: a magnifier with a plus, or a minus when it zooms `out`. winit maps
/// `ZoomIn` and `ZoomOut` to the plain arrow on Windows (#3735), so there it is a bitmap; macOS and
/// the Linux themes already draw magnifiers.
pub(crate) fn zoom(ctx: &egui::Context, out: bool) -> CursorIcon {
    zoom_with(ctx, out, cfg!(target_os = "windows") && ctx.viewport_id() == egui::ViewportId::ROOT)
}

fn zoom_with(ctx: &egui::Context, out: bool, bitmap: bool) -> CursorIcon {
    if bitmap && let Some(image) = cached_image(ctx, &Shape::Zoom { out }, native_scale(ctx)) {
        ctx.set_cursor_image(Some(image));
        return CursorIcon::None;
    }
    if out { CursorIcon::ZoomOut } else { CursorIcon::ZoomIn }
}

fn show(painter: &Painter, at: Pos2, shape: Shape) -> CursorIcon {
    let ctx = painter.ctx();
    // Where cursor images can't be uploaded (immediate viewports, the web), paint the shape.
    if native(ctx) {
        let scale = native_scale(ctx);
        if let Some(image) = cached_image(ctx, &shape, scale) {
            ctx.set_cursor_image(Some(image));
            return CursorIcon::None;
        }
        // An enormous tip cannot fit an OS cursor. Preserve its full-size outline, with an OS
        // crosshair at the hotspot so the user still gets immediate position feedback.
        paint(painter, at, &shape);
        if let Some(image) = cached_image(ctx, &Shape::Crosshair { length: 6.0, gap: 2.0 }, scale) {
            ctx.set_cursor_image(Some(image));
        }
        return CursorIcon::None;
    }
    paint(painter, at, &shape);
    CursorIcon::None
}

fn cached_image(ctx: &egui::Context, shape: &Shape, scale: f32) -> Option<CustomCursorImage> {
    let id = egui::Id::new("photocraft-tool-cursor");
    if let Some(cached) = ctx.data(|d| d.get_temp::<Cached>(id))
        && cached.shape == *shape
        && cached.scale == scale
    {
        return Some(cached.image);
    }
    let image = rasterize(shape, scale)?;
    ctx.data_mut(|d| d.insert_temp(id, Cached { shape: shape.clone(), scale, image: image.clone() }));
    Some(image)
}

fn paint(painter: &Painter, at: Pos2, shape: &Shape) {
    if extent(shape).is_none() {
        return;
    }
    let [dark, light] = crate::theme::Tokens::cursor_outline();
    for (width, color) in [(3.0, dark), (1.0, light)] {
        let stroke = Stroke::new(width, color);
        match shape {
            Shape::Circle { radius, centre } => {
                painter.circle_stroke(at, *radius, stroke);
                if *centre {
                    paint_crosshair(painter, at, 3.0, 0.0, stroke);
                }
            }
            Shape::Crosshair { length, gap } => paint_crosshair(painter, at, *length, *gap, stroke),
            Shape::Outline { rings, centre } => {
                for ring in rings.iter() {
                    if ring.len() >= 2 {
                        painter.add(egui::Shape::closed_line(ring.iter().map(|v| at + *v).collect(), stroke));
                    }
                }
                if *centre {
                    paint_crosshair(painter, at, 3.0, 0.0, stroke);
                }
            }
            // Only ever a bitmap: without one the OS draws its own grab cursors.
            Shape::Hand { .. } | Shape::Zoom { .. } => {}
        }
    }
}

fn paint_crosshair(painter: &Painter, at: Pos2, length: f32, gap: f32, stroke: Stroke) {
    for d in [vec2(1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0, 1.0), vec2(0.0, -1.0)] {
        painter.line_segment([at + d * gap, at + d * length], stroke);
    }
}

/// Distance to the nearest arm (round caps), in physical pixels from the hotspot.
fn cross_distance(x: f32, y: f32, length: f32, gap: f32) -> f32 {
    let horizontal = (x - x.clamp(gap, length)).hypot(y);
    let vertical = x.hypot(y - y.clamp(gap, length));
    horizontal.min(vertical)
}

fn extent(shape: &Shape) -> Option<f32> {
    Some(match shape {
        Shape::Circle { radius, centre } => {
            if !radius.is_finite() || *radius <= 0.0 {
                return None;
            }
            if *centre { radius.max(3.0) } else { *radius }
        }
        Shape::Crosshair { length, gap } => {
            if !length.is_finite() || !gap.is_finite() || *gap < 0.0 || length < gap {
                return None;
            }
            *length
        }
        Shape::Outline { rings, centre } => {
            let mut e = 0.0f32;
            for ring in rings.iter() {
                for p in ring {
                    if !p.x.is_finite() || !p.y.is_finite() {
                        return None;
                    }
                    e = e.max(p.x.hypot(p.y));
                }
            }
            if e <= 0.0 {
                return None;
            }
            if *centre { e.max(3.0) } else { e }
        }
        Shape::Hand { .. } => hand::HALF,
        Shape::Zoom { .. } => zoom::HALF,
    })
}

/// Distance from (`x`, `y`) to the segment `a`-`b`, all in the same units.
fn seg_distance(x: f32, y: f32, a: Vec2, b: Vec2) -> f32 {
    let (abx, aby) = (b.x - a.x, b.y - a.y);
    let t = (((x - a.x) * abx + (y - a.y) * aby) / (abx * abx + aby * aby).max(1e-12)).clamp(0.0, 1.0);
    ((x - a.x - abx * t).powi(2) + (y - a.y - aby * t).powi(2)).sqrt()
}

/// How much of the pixel at `(dx, dy)` physical pixels from the hotspot the light fill covers and
/// how much the dark outline around it does, each 0 to 1. A dark crease inside the hand takes
/// from the fill, so the fill never covers it.
fn coverage(shape: &Shape, dx: f32, dy: f32, scale: f32) -> (f32, f32) {
    let (x, y) = (dx.abs(), dy.abs());
    match shape {
        Shape::Circle { radius, centre } => {
            let ring = (x.hypot(y) - radius * scale).abs();
            let distance = if *centre { ring.min(cross_distance(x, y, 3.0 * scale, 0.0)) } else { ring };
            stroke_coverage(distance, scale)
        }
        Shape::Crosshair { length, gap } => stroke_coverage(cross_distance(x, y, length * scale, gap * scale), scale),
        Shape::Outline { rings, centre } => {
            let (lx, ly) = (dx / scale, dy / scale);
            let mut distance = f32::INFINITY;
            for ring in rings.iter() {
                let n = ring.len();
                if n < 2 {
                    continue;
                }
                for (i, &a) in ring.iter().enumerate() {
                    distance = distance.min(seg_distance(lx, ly, a, ring[(i + 1) % n]));
                }
            }
            if *centre {
                distance = distance.min(cross_distance(dx, dy, 3.0 * scale, 0.0));
            }
            stroke_coverage(distance * scale, scale)
        }
        Shape::Hand { closed } => {
            let (lx, ly) = (dx / scale, dy / scale);
            silhouette(hand::shape(*closed, lx, ly), hand::crease(*closed, lx, ly), 0.45, scale)
        }
        Shape::Zoom { out } => {
            let (lx, ly) = (dx / scale, dy / scale);
            silhouette(zoom::shape(lx, ly), zoom::glyph(*out, lx, ly), 0.75, scale)
        }
    }
}

/// A light filled shape with a 1 px dark outline and dark lines `half_width` logical pixels either
/// side of its creases, from their signed distances in logical pixels.
fn silhouette(edge: f32, crease: f32, half_width: f32, scale: f32) -> (f32, f32) {
    let edge = edge * scale;
    let inside = (0.5 - edge).clamp(0.0, 1.0);
    let outline = (scale + 0.5 - edge).clamp(0.0, 1.0);
    let crease = (half_width * scale + 0.5 - crease * scale).clamp(0.0, 1.0);
    (inside * (1.0 - crease), outline.max(inside * crease))
}

/// A 1 px light line with a 1 px dark one on each side, `distance` physical pixels from its centre.
fn stroke_coverage(distance: f32, scale: f32) -> (f32, f32) {
    ((0.5 * scale + 0.5 - distance).clamp(0.0, 1.0), (1.5 * scale + 0.5 - distance).clamp(0.0, 1.0))
}

/// Splat the distance to one ring segment into an outline's side × side distance buffer.
fn splat_segment(dist: &mut [f32], side: usize, hot: f32, a: Vec2, b: Vec2, reach: f32) {
    let n = side as f32;
    let (x0, x1) = (a.x.min(b.x) + hot - reach, a.x.max(b.x) + hot + reach);
    let (y0, y1) = (a.y.min(b.y) + hot - reach, a.y.max(b.y) + hot + reach);
    if x1 < 0.0 || y1 < 0.0 || x0 > n - 1.0 || y0 > n - 1.0 {
        return;
    }
    let (c0, c1) = (x0.max(0.0).floor() as i32, x1.min(n - 1.0).ceil() as i32);
    let (r0, r1) = (y0.max(0.0).floor() as i32, y1.min(n - 1.0).ceil() as i32);
    for row in r0..=r1 {
        let dy = row as f32 - hot;
        for col in c0..=c1 {
            let d = seg_distance(col as f32 - hot, dy, a, b);
            let i = row as usize * side + col as usize;
            if d < dist[i] {
                dist[i] = d;
            }
        }
    }
}

/// Fold the centre crosshair (the circle cursor's) into the outline distances.
fn splat_cross(dist: &mut [f32], side: usize, hot: f32, arm: f32) {
    let reach = arm + 2.0;
    let lo = (hot - reach).max(0.0).floor() as i32;
    let hi = ((hot + reach).min(side as f32 - 1.0)).ceil() as i32;
    for row in lo..=hi {
        let dy = row as f32 - hot;
        for col in lo..=hi {
            let dx = col as f32 - hot;
            let d = cross_distance(dx, dy, arm, 0.0);
            let i = row as usize * side + col as usize;
            if d < dist[i] {
                dist[i] = d;
            }
        }
    }
}

/// One bitmap pixel from its distance to the stroke centre, in straight RGBA (winit form).
/// A pixel with light `fill` over a dark `outline` (coverages 0 to 1), in straight RGBA as winit
/// requires (not egui's premultiplied). No coverage is `[0; 4]`.
fn stroke_pixel((fill, outline): (f32, f32), dark: egui::Color32, light: egui::Color32) -> [u8; 4] {
    let a_dark = outline * f32::from(dark.a()) / 255.0;
    let a_light = fill * f32::from(light.a()) / 255.0;
    let alpha = a_light + a_dark * (1.0 - a_light);
    let white = if alpha > 0.0 { (255.0 * a_light / alpha).round() as u8 } else { 0 };
    [white, white, white, (255.0 * alpha).round() as u8]
}

/// The columns of row offset `dy` (from the hotspot, in pixels) that can be painted by a ring of
/// radius `r` drawn `reach` wide either side: the left and right arcs of the annulus, padded by
/// a pixel. Every other pixel of the row is transparent.
fn ring_spans(dy: f32, r: f32, reach: f32) -> Option<[(f32, f32); 2]> {
    let (ro, ri) = (r + reach + 1.0, (r - reach - 1.0).max(0.0));
    if dy.abs() > ro {
        return None;
    }
    let xo = (ro * ro - dy * dy).max(0.0).sqrt();
    let xi = if dy.abs() < ri { (ri * ri - dy * dy).sqrt() } else { 0.0 };
    Some([(-xo, -xi), (xi, xo)])
}

fn rasterize(shape: &Shape, scale: f32) -> Option<CustomCursorImage> {
    if !scale.is_finite() || scale <= 0.0 {
        return None;
    }
    let extent = extent(shape)?;
    // Odd dimensions put the hotspot on the centre pixel at every DPI. Include the outline
    // and its antialiasing fringe, then check before converting or allocating.
    let half = ((extent + 1.5) * scale + 1.0).ceil();
    if !half.is_finite() || half > f32::from((MAX_SIDE - 1) / 2) {
        return None;
    }
    let hot = half as u16;
    let side = hot * 2 + 1;
    let bytes = usize::from(side).checked_mul(usize::from(side))?.checked_mul(4)?;
    // Transparent to start; only pixels a shape can reach are evaluated below (a big tip's
    // outline is a thin ring in a large square, #1343).
    let mut rgba = vec![0u8; bytes];
    let [dark, light] = crate::theme::Tokens::cursor_outline();
    let n = usize::from(side);
    let hotf = f32::from(hot);
    let mut put = |col: usize, row: usize, cov: (f32, f32)| {
        if let Some(px) = row.checked_mul(n).and_then(|i| i.checked_add(col)).and_then(|i| rgba.get_mut(i * 4..i * 4 + 4)) {
            px.copy_from_slice(&stroke_pixel(cov, dark, light));
        }
    };
    if let Shape::Outline { rings, centre } = shape {
        // Splat each segment into its bounding box instead of measuring every ring per pixel:
        // tip outlines may carry hundreds of points.
        let reach = 1.5 * scale + 2.0;
        let mut dist = vec![f32::INFINITY; n * n];
        for ring in rings.iter() {
            let len = ring.len();
            if len < 2 {
                continue;
            }
            for (i, &a) in ring.iter().enumerate() {
                splat_segment(&mut dist, n, hotf, a * scale, ring[(i + 1) % len] * scale, reach);
            }
        }
        if *centre {
            splat_cross(&mut dist, n, hotf, 3.0 * scale);
        }
        for (i, &d) in dist.iter().enumerate() {
            if d.is_finite() {
                put(i % n, i / n, stroke_coverage(d, scale));
            }
        }
        return Some(CustomCursorImage { rgba: Arc::from(rgba), size: [side, side], hotspot: [hot, hot] });
    }
    // Columns to evaluate in each row: the ring's two arcs (and the centre mark) for a circle,
    // the whole row otherwise.
    let last = f32::from(side - 1);
    for row in 0..n {
        let dy = row as f32 - hotf;
        let mut spans: [Option<(f32, f32)>; 3] = [None; 3];
        if let Shape::Circle { radius, centre } = shape {
            let reach = 1.5 * scale + 0.5;
            if let Some([l, r]) = ring_spans(dy, radius * scale, reach) {
                spans = [Some(l), Some(r), None];
            }
            let cross = 3.0 * scale + reach + 1.0;
            if *centre && dy.abs() <= cross {
                spans[2] = Some((-cross, cross));
            }
        } else {
            spans[0] = Some((-hotf, hotf));
        }
        for (a, b) in spans.into_iter().flatten() {
            let (c0, c1) = ((hotf + a).floor().clamp(0.0, last) as usize, (hotf + b).ceil().clamp(0.0, last) as usize);
            for col in c0..=c1 {
                put(col, row, coverage(shape, col as f32 - hotf, dy, scale));
            }
        }
    }
    Some(CustomCursorImage { rgba: Arc::from(rgba), size: [side, side], hotspot: [hot, hot] })
}

#[cfg(test)]
mod tests;
