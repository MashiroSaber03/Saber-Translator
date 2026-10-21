//! The Crop tool's composition overlay (Photoshop's options-bar overlay menu, #1919): which guide
//! the crop box shows (Rule of Thirds, Grid, Diagonal, Triangle, Golden Ratio, Golden Spiral),
//! when it shows it (Auto: only while the box is dragged; Always; Never), and its orientation.
//! With the Crop tool showing a box, O cycles the overlay and ⇧O its orientation (Triangle and
//! Golden Spiral); otherwise O stays the Dodge/Burn/Sponge key.
//!
//! The guides are polylines in the frame's unit square (0..1 along its top edge, 0..1 down its
//! left edge), so the canvas maps the same geometry onto an upright rectangle or a turned frame.
//! The state lives in `ToolOptions` (`crop_overlay`, `crop_overlay_show`,
//! `crop_overlay_orientation`), which `ui.inspect` reports and `ui.set` drives.

use egui::{Key, Modifiers};
use serde::{Deserialize, Serialize};

use crate::PhotocraftApp;
use crate::state::{Tool, ToolOptions};

/// The guide drawn inside the crop box.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CropOverlay {
    #[default]
    Thirds,
    Grid,
    Diagonal,
    Triangle,
    GoldenRatio,
    GoldenSpiral,
}

impl CropOverlay {
    /// In Photoshop's menu order (the order O cycles through).
    pub const ALL: [CropOverlay; 6] = [Self::Thirds, Self::Grid, Self::Diagonal, Self::Triangle, Self::GoldenRatio, Self::GoldenSpiral];

    pub fn label(self) -> &'static str {
        match self {
            Self::Thirds => "Rule of Thirds",
            Self::Grid => "Grid",
            Self::Diagonal => "Diagonal",
            Self::Triangle => "Triangle",
            Self::GoldenRatio => "Golden Ratio",
            Self::GoldenSpiral => "Golden Spiral",
        }
    }

    /// The `ui.set` / serde name.
    pub fn id(self) -> &'static str {
        match self {
            Self::Thirds => "thirds",
            Self::Grid => "grid",
            Self::Diagonal => "diagonal",
            Self::Triangle => "triangle",
            Self::GoldenRatio => "goldenRatio",
            Self::GoldenSpiral => "goldenSpiral",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.id() == id)
    }

    /// How many distinct orientations ⇧O steps through: the Triangle's diagonal runs either way,
    /// the Golden Spiral winds into any of the four corners; the others are symmetric.
    pub fn orientations(self) -> u8 {
        match self {
            Self::Triangle => 2,
            Self::GoldenSpiral => 4,
            _ => 1,
        }
    }

    /// The next overlay (O).
    pub fn next(self) -> Self {
        let i = Self::ALL.iter().position(|k| *k == self).unwrap_or(0);
        Self::ALL.get((i + 1) % Self::ALL.len()).copied().unwrap_or_default()
    }
}

/// When the overlay shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OverlayShow {
    /// Only while the crop box is being dragged (drawn, moved, resized or turned).
    #[default]
    Auto,
    Always,
    Never,
}

impl OverlayShow {
    pub const ALL: [OverlayShow; 3] = [Self::Auto, Self::Always, Self::Never];

    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto Show Overlay",
            Self::Always => "Always Show Overlay",
            Self::Never => "Never Show Overlay",
        }
    }

    pub fn id(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Always => "always",
            Self::Never => "never",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.id() == id)
    }

    /// Whether the overlay shows, `gesture` being a crop drag in progress.
    pub fn shows(self, gesture: bool) -> bool {
        match self {
            Self::Auto => gesture,
            Self::Always => true,
            Self::Never => false,
        }
    }
}

/// A guide polyline in the frame's unit square.
pub type Polyline = Vec<[f32; 2]>;

/// Golden section: 1/φ.
const INV_PHI: f32 = 0.618_034;
/// Target spacing (screen points) of the Grid overlay's lines.
const GRID_PX: f32 = 32.0;
/// Most lines the Grid draws per axis, however large the frame is on screen.
const GRID_MAX: usize = 96;

/// The overlay's guides for a frame `w` × `h` screen points (along its own axes), in its unit
/// square. Orientation `orient` is taken modulo the overlay's [`CropOverlay::orientations`].
/// Degenerate or non-finite sizes give no guides.
pub fn guides(kind: CropOverlay, orient: u8, w: f32, h: f32) -> Vec<Polyline> {
    if !(w.is_finite() && h.is_finite()) || w < 1.0 || h < 1.0 {
        return Vec::new();
    }
    let seg = |a: [f32; 2], b: [f32; 2]| vec![a, b];
    let cross = |fs: &[f32]| -> Vec<Polyline> { fs.iter().flat_map(|&f| [seg([f, 0.0], [f, 1.0]), seg([0.0, f], [1.0, f])]).collect() };
    let lines = match kind {
        CropOverlay::Thirds => cross(&[1.0 / 3.0, 2.0 / 3.0]),
        CropOverlay::GoldenRatio => cross(&[1.0 - INV_PHI, INV_PHI]),
        CropOverlay::Grid => {
            let axis = |len: f32, vertical: bool| -> Vec<Polyline> {
                // Square cells of about GRID_PX on screen, at most GRID_MAX lines per axis.
                let n = ((len / GRID_PX) as usize).clamp(2, GRID_MAX);
                (1..n)
                    .map(|i| {
                        let f = i as f32 / n as f32;
                        if vertical { seg([f, 0.0], [f, 1.0]) } else { seg([0.0, f], [1.0, f]) }
                    })
                    .collect()
            };
            let mut v = axis(w, true);
            v.extend(axis(h, false));
            v
        }
        CropOverlay::Diagonal => {
            // 45° on screen from each corner until the line meets the far edge.
            let m = w.min(h);
            let (u, v) = (m / w, m / h);
            vec![seg([0.0, 0.0], [u, v]), seg([1.0, 0.0], [1.0 - u, v]), seg([0.0, 1.0], [u, 1.0 - v]), seg([1.0, 1.0], [1.0 - u, 1.0 - v])]
        }
        CropOverlay::Triangle => {
            // The top-left to bottom-right diagonal, and the perpendiculars to it (on screen) from
            // the other two corners; their feet sit at `t` and `s` along it.
            // t = w² / (w² + h²) and s = 1 - t, from the smaller ratio so huge sizes can't overflow.
            let (t, s) = if w >= h {
                let t = 1.0 / (1.0 + (h / w).powi(2));
                (t, 1.0 - t)
            } else {
                let s = 1.0 / (1.0 + (w / h).powi(2));
                (1.0 - s, s)
            };
            vec![seg([0.0, 0.0], [1.0, 1.0]), seg([1.0, 0.0], [t, t]), seg([0.0, 1.0], [s, s])]
        }
        CropOverlay::GoldenSpiral => vec![golden_spiral()],
    };
    let o = orient % kind.orientations().max(1);
    let (fx, fy) = (o & 1 != 0, o & 2 != 0);
    lines.into_iter().map(|l| l.into_iter().map(|[x, y]| [if fx { 1.0 - x } else { x }, if fy { 1.0 - y } else { y }]).collect()).collect()
}

/// The golden spiral: quarter arcs through nested golden rectangles, squares cut off the left,
/// top, right and bottom in turn, winding in towards the lower right. Stretched to the unit square.
fn golden_spiral() -> Polyline {
    const PHI: f32 = 1.0 + INV_PHI;
    const STEPS: usize = 12;
    // The remaining rectangle [x0, y0, x1, y1] in a φ × 1 space, and the spiral's current point.
    let mut r = [0.0f32, 0.0, PHI, 1.0];
    let mut p = [0.0f32, 1.0];
    let mut pts = vec![p];
    for i in 0..12 {
        let (w, h) = (r[2] - r[0], r[3] - r[1]);
        let s = w.min(h);
        if s < 1e-4 {
            break;
        }
        // The square cut off, the arc's centre (a corner of it) and the end point; the rest stays.
        let (centre, end) = match i % 4 {
            0 => {
                r[0] += s;
                ([r[0], r[3]], [r[0], r[1]])
            }
            1 => {
                r[1] += s;
                ([r[0], r[1]], [r[2], r[1]])
            }
            2 => {
                r[2] -= s;
                ([r[2], r[1]], [r[2], r[3]])
            }
            _ => {
                r[3] -= s;
                ([r[2], r[3]], [r[0], r[3]])
            }
        };
        let a0 = (p[1] - centre[1]).atan2(p[0] - centre[0]);
        let a1 = (end[1] - centre[1]).atan2(end[0] - centre[0]);
        let mut da = a1 - a0;
        if da > std::f32::consts::PI {
            da -= std::f32::consts::TAU;
        } else if da < -std::f32::consts::PI {
            da += std::f32::consts::TAU;
        }
        for k in 1..=STEPS {
            let a = a0 + da * k as f32 / STEPS as f32;
            pts.push([centre[0] + s * a.cos(), centre[1] + s * a.sin()]);
        }
        p = end;
    }
    pts.into_iter().map(|[x, y]| [(x / PHI).clamp(0.0, 1.0), y.clamp(0.0, 1.0)]).collect()
}

/// The guides the pending crop box shows now: none when the options say not to.
pub fn current_guides(app: &PhotocraftApp, w: f32, h: f32) -> Vec<Polyline> {
    let o = &app.ui.tool_options;
    if app.ui.tool != Tool::Crop || !o.crop_overlay_show.shows(app.crop.drag.is_some()) {
        return Vec::new();
    }
    guides(o.crop_overlay, o.crop_overlay_orientation, w, h)
}

/// O: the next overlay (the orientation starts over).
pub fn cycle(o: &mut ToolOptions) {
    o.crop_overlay = o.crop_overlay.next();
    o.crop_overlay_orientation = 0;
}

/// ⇧O: the overlay's next orientation (nothing for a symmetric one).
pub fn cycle_orientation(o: &mut ToolOptions) {
    let n = o.crop_overlay.orientations().max(1);
    o.crop_overlay_orientation = (o.crop_overlay_orientation % n + 1) % n;
}

/// O / ⇧O with the Crop tool showing a box and no drag in progress. Returns true when a key was
/// used; otherwise O stays the Dodge/Burn/Sponge tool key.
pub fn keys(app: &mut PhotocraftApp, ctx: &egui::Context) -> bool {
    if app.ui.tool != Tool::Crop || app.ui.crop_rect.is_none() || app.crop.drag.is_some() {
        return false;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::SHIFT, Key::O)) {
        cycle_orientation(&mut app.ui.tool_options);
        return true;
    }
    if ctx.input_mut(|i| i.consume_key(Modifiers::NONE, Key::O)) {
        cycle(&mut app.ui.tool_options);
        return true;
    }
    false
}

/// The options bar's overlay button and its menu (Photoshop's "Set the overlay options for the
/// Crop tool").
pub fn options_button(o: &mut ToolOptions, ui: &mut egui::Ui) {
    let r =
        crate::icons::button(ui, "grid-3x3", 24.0, false, &format!("{}  ({})", tl!("Set the overlay options for the Crop tool"), tl!(o.crop_overlay.label())));
    egui::Popup::menu(&r).show(|ui| {
        ui.set_min_width(220.0);
        let item = |ui: &mut egui::Ui, on: bool, label: &str, key: Option<&str>| {
            let mut b = egui::Button::new(format!("{} {}", if on { "✔" } else { "  " }, label));
            if let Some(k) = key {
                b = b.shortcut_text(crate::shortcuts::pretty(k));
            }
            ui.add(b).clicked()
        };
        for k in CropOverlay::ALL {
            if item(ui, o.crop_overlay == k, tl!(k.label()), None) {
                if o.crop_overlay != k {
                    o.crop_overlay_orientation = 0;
                }
                o.crop_overlay = k;
                ui.close();
            }
        }
        ui.separator();
        for s in OverlayShow::ALL {
            if item(ui, o.crop_overlay_show == s, tl!(s.label()), None) {
                o.crop_overlay_show = s;
                ui.close();
            }
        }
        ui.separator();
        if item(ui, false, tl!("Cycle Overlay"), Some("O")) {
            cycle(o);
            ui.close();
        }
        let turns = o.crop_overlay.orientations() > 1;
        let b = egui::Button::new(format!("   {}", tl!("Cycle Orientation"))).shortcut_text(crate::shortcuts::pretty("Shift+O"));
        if ui.add_enabled(turns, b).clicked() {
            cycle_orientation(o);
            ui.close();
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_points(g: &[Polyline]) -> impl Iterator<Item = [f32; 2]> + '_ {
        g.iter().flatten().copied()
    }

    #[test]
    fn every_guide_stays_in_the_unit_square() {
        for kind in CropOverlay::ALL {
            for o in 0..8 {
                for (w, h) in [(1.0, 1.0), (300.0, 200.0), (200.0, 300.0), (5000.0, 3.0), (3.0, 5000.0), (1e7, 1e7), (1e30, 2.0)] {
                    let g = guides(kind, o, w, h);
                    assert!(!g.is_empty(), "{kind:?} {w}x{h}");
                    for p in all_points(&g) {
                        assert!(p.iter().all(|v| v.is_finite() && (-1e-5..=1.0 + 1e-5).contains(v)), "{kind:?} o{o} {w}x{h}: {p:?}");
                    }
                    assert!(g.iter().all(|l| l.len() >= 2));
                }
            }
        }
    }

    #[test]
    fn degenerate_frames_have_no_guides() {
        for kind in CropOverlay::ALL {
            for (w, h) in [(0.0, 10.0), (10.0, 0.0), (f32::NAN, 10.0), (f32::INFINITY, 10.0), (-5.0, -5.0), (0.5, 100.0)] {
                assert!(guides(kind, 0, w, h).is_empty(), "{kind:?} {w}x{h}");
            }
        }
    }

    #[test]
    fn thirds_and_golden_ratio_positions() {
        let xs = |g: Vec<Polyline>| -> Vec<f32> { g.iter().filter(|l| l[0][1] == 0.0 && l[1][1] == 1.0).map(|l| l[0][0]).collect() };
        let t = xs(guides(CropOverlay::Thirds, 0, 300.0, 200.0));
        assert_eq!(t.len(), 2);
        assert!((t[0] - 1.0 / 3.0).abs() < 1e-6 && (t[1] - 2.0 / 3.0).abs() < 1e-6);
        let g = xs(guides(CropOverlay::GoldenRatio, 0, 300.0, 200.0));
        assert!((g[0] - 0.381_966).abs() < 1e-5 && (g[1] - 0.618_034).abs() < 1e-5, "{g:?}");
        assert_eq!(guides(CropOverlay::GoldenRatio, 0, 300.0, 200.0).len(), 4);
    }

    #[test]
    fn grid_spacing_is_bounded() {
        for (w, h) in [(10.0, 10.0), (64.0, 64.0), (640.0, 320.0), (1e6, 1e6)] {
            let g = guides(CropOverlay::Grid, 0, w, h);
            let v = g.iter().filter(|l| l[0][0] == l[1][0]).count();
            let hz = g.len() - v;
            assert!((1..GRID_MAX).contains(&v) && (1..GRID_MAX).contains(&hz), "{w}x{h}: {v} {hz}");
        }
        // About one line per GRID_PX on screen.
        let g = guides(CropOverlay::Grid, 0, 640.0, 320.0);
        assert_eq!(g.iter().filter(|l| l[0][0] == l[1][0]).count(), 19);
        assert_eq!(g.iter().filter(|l| l[0][1] == l[1][1]).count(), 9);
    }

    #[test]
    fn diagonals_run_at_45_degrees_on_screen() {
        let (w, h) = (300.0f32, 200.0f32);
        for l in guides(CropOverlay::Diagonal, 0, w, h) {
            let (dx, dy) = ((l[1][0] - l[0][0]) * w, (l[1][1] - l[0][1]) * h);
            assert!((dx.abs() - dy.abs()).abs() < 1e-3 && (dx.abs() - 200.0).abs() < 1e-3, "{l:?}");
        }
        // A square frame: corner to corner.
        let g = guides(CropOverlay::Diagonal, 0, 100.0, 100.0);
        assert_eq!(g[0], vec![[0.0, 0.0], [1.0, 1.0]]);
    }

    #[test]
    fn triangle_perpendiculars_meet_the_diagonal_at_right_angles() {
        let (w, h) = (300.0f32, 200.0f32);
        let g = guides(CropOverlay::Triangle, 0, w, h);
        let diag = [w, h];
        for l in &g[1..] {
            let d = [(l[1][0] - l[0][0]) * w, (l[1][1] - l[0][1]) * h];
            assert!((d[0] * diag[0] + d[1] * diag[1]).abs() < 1e-2, "{l:?}");
            assert!((l[1][0] - l[1][1]).abs() < 1e-6, "foot on the diagonal: {l:?}");
        }
        // The other orientation mirrors it: the diagonal runs top-right to bottom-left.
        let m = guides(CropOverlay::Triangle, 1, w, h);
        assert_eq!(m[0], vec![[1.0, 0.0], [0.0, 1.0]]);
        for (a, b) in g.iter().zip(&m) {
            for (p, q) in a.iter().zip(b) {
                assert!((p[0] - (1.0 - q[0])).abs() < 1e-6 && (p[1] - q[1]).abs() < 1e-6);
            }
        }
        // Two orientations: 2 is 0 again.
        assert_eq!(guides(CropOverlay::Triangle, 2, w, h), g);
    }

    #[test]
    fn golden_spiral_orientations_mirror_it_into_each_corner() {
        let base = guides(CropOverlay::GoldenSpiral, 0, 300.0, 200.0);
        let pts = &base[0];
        assert!(pts.len() > 40);
        // It starts at the bottom-left corner and winds in towards the golden point (lower right).
        assert_eq!(pts[0], [0.0, 1.0]);
        let end = pts[pts.len() - 1];
        assert!((end[0] - 0.7236).abs() < 0.02 && (end[1] - 0.7236).abs() < 0.02, "{end:?}");
        for o in 1..4u8 {
            let g = guides(CropOverlay::GoldenSpiral, o, 300.0, 200.0);
            for (p, q) in pts.iter().zip(&g[0]) {
                let x = if o & 1 != 0 { 1.0 - p[0] } else { p[0] };
                let y = if o & 2 != 0 { 1.0 - p[1] } else { p[1] };
                assert!((x - q[0]).abs() < 1e-6 && (y - q[1]).abs() < 1e-6);
            }
        }
        assert_eq!(guides(CropOverlay::GoldenSpiral, 4, 300.0, 200.0), base);
        // Symmetric overlays ignore the orientation.
        assert_eq!(guides(CropOverlay::Thirds, 3, 300.0, 200.0), guides(CropOverlay::Thirds, 0, 300.0, 200.0));
    }

    #[test]
    fn cycling_overlays_and_orientations() {
        let mut o = ToolOptions::default();
        assert_eq!((o.crop_overlay, o.crop_overlay_show, o.crop_overlay_orientation), (CropOverlay::Thirds, OverlayShow::Auto, 0));
        let mut seen = vec![o.crop_overlay];
        for _ in 0..6 {
            cycle(&mut o);
            seen.push(o.crop_overlay);
        }
        assert_eq!(&seen[..6], &CropOverlay::ALL);
        assert_eq!(seen[6], CropOverlay::Thirds, "wraps around");
        cycle_orientation(&mut o);
        assert_eq!(o.crop_overlay_orientation, 0, "thirds has one orientation");
        o.crop_overlay = CropOverlay::GoldenSpiral;
        let turns: Vec<u8> = (0..5)
            .map(|_| {
                cycle_orientation(&mut o);
                o.crop_overlay_orientation
            })
            .collect();
        assert_eq!(turns, [1, 2, 3, 0, 1]);
        cycle(&mut o);
        assert_eq!((o.crop_overlay, o.crop_overlay_orientation), (CropOverlay::Thirds, 0));
        // A stray orientation (an old or hand-edited state) still cycles in range.
        o.crop_overlay = CropOverlay::Triangle;
        o.crop_overlay_orientation = 200;
        cycle_orientation(&mut o);
        assert!(o.crop_overlay_orientation < 2);
    }

    #[test]
    fn visibility_modes() {
        assert!(OverlayShow::Auto.shows(true) && !OverlayShow::Auto.shows(false));
        assert!(OverlayShow::Always.shows(true) && OverlayShow::Always.shows(false));
        assert!(!OverlayShow::Never.shows(true) && !OverlayShow::Never.shows(false));
        for k in CropOverlay::ALL {
            assert_eq!(CropOverlay::from_id(k.id()), Some(k));
        }
        for s in OverlayShow::ALL {
            assert_eq!(OverlayShow::from_id(s.id()), Some(s));
        }
        assert_eq!(CropOverlay::from_id("spiral"), None);
    }

    /// One key press through the shortcut handler, as egui delivers it.
    fn press(app: &mut PhotocraftApp, mods: Modifiers) {
        let ctx = egui::Context::default();
        ctx.input_mut(|i| {
            i.modifiers = mods;
            i.events.push(egui::Event::Key { key: Key::O, physical_key: None, pressed: true, repeat: false, modifiers: mods });
        });
        crate::shortcuts::handle(app, &ctx);
    }

    fn crop_app() -> PhotocraftApp {
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.run("file.new", serde_json::json!({"width": 200, "height": 120})).unwrap();
        app.ui.tool = Tool::Crop;
        crate::crop_ui::ensure_frame(&mut app);
        assert!(app.ui.crop_rect.is_some());
        app
    }

    #[test]
    fn o_cycles_the_overlay_while_the_crop_tool_shows_a_box() {
        let mut app = crop_app();
        press(&mut app, Modifiers::NONE);
        assert_eq!((app.ui.tool, app.ui.tool_options.crop_overlay), (Tool::Crop, CropOverlay::Grid));
        // ⇧O on a symmetric overlay changes nothing, and doesn't switch tools either.
        press(&mut app, Modifiers::SHIFT);
        assert_eq!((app.ui.tool, app.ui.tool_options.crop_overlay, app.ui.tool_options.crop_overlay_orientation), (Tool::Crop, CropOverlay::Grid, 0));
        press(&mut app, Modifiers::NONE);
        press(&mut app, Modifiers::NONE);
        assert_eq!(app.ui.tool_options.crop_overlay, CropOverlay::Triangle);
        press(&mut app, Modifiers::SHIFT);
        assert_eq!(app.ui.tool_options.crop_overlay_orientation, 1);
        assert_eq!(app.ui.tool, Tool::Crop);
    }

    #[test]
    fn o_stays_the_dodge_key_without_a_crop_box() {
        // No document: the Crop tool shows no box, so O picks the Dodge tool.
        let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), crate::Services::default());
        app.ui.tool = Tool::Crop;
        press(&mut app, Modifiers::NONE);
        assert_eq!(app.ui.tool, Tool::Dodge);
        assert_eq!(app.ui.tool_options.crop_overlay, CropOverlay::Thirds);
        // Another tool: O is its tool key as before.
        let mut app = crop_app();
        app.ui.tool = Tool::Brush;
        press(&mut app, Modifiers::NONE);
        assert_eq!((app.ui.tool, app.ui.tool_options.crop_overlay), (Tool::Dodge, CropOverlay::Thirds));
        // A crop drag in progress leaves the overlay alone.
        let mut app = crop_app();
        app.crop.drag = Some(crate::crop_ui::CropDrag::Move { start: [0.0, 0.0], rect: [0.0, 0.0, 10.0, 10.0] });
        let ctx = egui::Context::default();
        ctx.input_mut(|i| i.events.push(egui::Event::Key { key: Key::O, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE }));
        assert!(!keys(&mut app, &ctx));
        assert_eq!(app.ui.tool_options.crop_overlay, CropOverlay::Thirds);
    }

    #[test]
    fn the_overlay_shows_by_mode_and_gesture() {
        let mut app = crop_app();
        assert!(current_guides(&app, 300.0, 200.0).is_empty(), "Auto: no guides without a drag");
        app.crop.drag = Some(crate::crop_ui::CropDrag::Move { start: [0.0, 0.0], rect: [0.0, 0.0, 10.0, 10.0] });
        assert_eq!(current_guides(&app, 300.0, 200.0).len(), 4, "Auto: the thirds while dragging");
        app.ui.tool_options.crop_overlay_show = OverlayShow::Never;
        assert!(current_guides(&app, 300.0, 200.0).is_empty());
        app.crop.drag = None;
        app.ui.tool_options.crop_overlay_show = OverlayShow::Always;
        app.ui.tool_options.crop_overlay = CropOverlay::Triangle;
        assert_eq!(current_guides(&app, 300.0, 200.0).len(), 3);
        app.ui.tool = Tool::Brush;
        assert!(current_guides(&app, 300.0, 200.0).is_empty(), "a pending frame under another tool shows no guide");
    }

    #[test]
    fn old_state_files_get_photoshops_defaults() {
        let o: ToolOptions = serde_json::from_str(r#"{"crop_ratio": "1:1"}"#).unwrap();
        assert_eq!((o.crop_overlay, o.crop_overlay_show, o.crop_overlay_orientation), (CropOverlay::Thirds, OverlayShow::Auto, 0));
        let o: ToolOptions = serde_json::from_str(r#"{"crop_overlay": "goldenSpiral", "crop_overlay_show": "always", "crop_overlay_orientation": 3}"#).unwrap();
        assert_eq!((o.crop_overlay, o.crop_overlay_show, o.crop_overlay_orientation), (CropOverlay::GoldenSpiral, OverlayShow::Always, 3));
    }
}
