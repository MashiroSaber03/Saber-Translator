//! The transparency checker is made of 8 screen pixel cells (the Medium preference) at every zoom,
//! anchored to the document's top-left corner so the pattern moves with the image. The pixel grid shows above 500% zoom,
//! over pixels with content only, never over empty checker.
//!
//! These tests render the real app offscreen and read pixels back. They skip when no GPU adapter
//! is available. The CPU canvas (no wgpu handed to the app) has its own pixel grid tests at the end.

use photocraft_ui_egui::PhotocraftApp;
use photocraft_ui_egui::control::{ControlRequest, handle};
use serde_json::{Value, json};

type Harness = egui_kittest::Harness<'static, PhotocraftApp>;

const LIGHT: u8 = 255;
const DARK: u8 = 204;

fn harness() -> Option<Harness> {
    harness_with(true)
}

/// The app with the GPU canvas (`gpu`), or without it: the CPU fallback path.
fn harness_with(gpu: bool) -> Option<Harness> {
    let built = std::panic::catch_unwind(|| {
        egui_kittest::Harness::builder().with_size(egui::vec2(1400.0, 800.0)).with_pixels_per_point(1.0).with_max_steps(64).wgpu().build_eframe(|cc| {
            PhotocraftApp::setup_context(&cc.egui_ctx, Default::default());
            let mut app = PhotocraftApp::new(photocraft_engine::Session::new(), Default::default());
            if let Some(rs) = cc.wgpu_render_state.as_ref().filter(|_| gpu) {
                app.set_wgpu(rs.clone());
            }
            app
        })
    });
    match built {
        Ok(h) => Some(h),
        Err(_) => {
            eprintln!("skipping: no GPU adapter");
            None
        }
    }
}

fn control(h: &mut Harness, method: &str, params: Value) {
    let ctx = h.ctx.clone();
    let (req, _rx) = ControlRequest::new(method, params);
    handle(h.state_mut(), &ctx, &req);
    h.run_steps(4);
}

/// True for the neutral light greys the default checker (and its pixel-grid darkening) paints.
fn is_checker_grey(p: [u8; 4]) -> bool {
    p[0] == p[1] && p[1] == p[2] && p[0] >= 160
}

/// Longest run of `true` in `flags`, as `(start, len)`.
fn longest_run(flags: impl Iterator<Item = bool>) -> (usize, usize) {
    let (mut best, mut start, mut len) = ((0, 0), 0, 0);
    for (i, f) in flags.enumerate() {
        if f {
            if len == 0 {
                start = i;
            }
            len += 1;
            if len > best.1 {
                best = (start, len);
            }
        } else {
            len = 0;
        }
    }
    best
}

/// A rendered screen.
struct Shot {
    w: u32,
    px: Vec<u8>,
}

impl Shot {
    fn at(&self, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * self.w + x) * 4) as usize;
        [self.px[i], self.px[i + 1], self.px[i + 2], self.px[i + 3]]
    }
}

/// Opens a `doc` px square document at `zoom` (transparent, or filled black when `black`), renders
/// it with the pixel grid on or off, and returns the screenshot with the document's left and top
/// screen edge and its width in screen pixels.
fn render(h: &mut Harness, zoom: f32, doc: u32, black: bool, grid: bool) -> (Shot, u32, u32, u32) {
    render_on(h, true, zoom, doc, black, grid)
}

/// [`render`] on the GPU canvas (`gpu`) or the CPU canvas.
fn render_on(h: &mut Harness, gpu: bool, zoom: f32, doc: u32, black: bool, grid: bool) -> (Shot, u32, u32, u32) {
    {
        let app = h.state_mut();
        while app.session.active().is_some() {
            app.run("file.close", json!({"discard": true})).expect("close");
        }
        let background = if black { "white" } else { "transparent" };
        app.run("file.new", json!({"width": doc, "height": doc, "background": background})).expect("new");
        if black {
            app.run("edit.fill", json!({"color": "#000000"})).expect("fill");
        }
        app.sync_views();
        app.ui.view.show.pixel_grid = grid;
    }
    control(h, "ui.set", json!({"zoom": zoom, "center": [doc as f32 / 2.0, doc as f32 / 2.0]}));
    h.run_steps(3);
    assert_eq!(h.state().perf.gpu, gpu, "zoom {zoom}: expected the {} canvas", if gpu { "GPU" } else { "CPU" });
    let rendered = h.render().expect("render");
    let (width, height) = (rendered.width(), rendered.height());
    let img = Shot { w: width, px: rendered.into_raw() };
    let r = h.state().last_canvas_rect;
    let (cx, cy) = (r.center().x as u32, r.center().y as u32);
    // Black fill is 0 (64 on a grid line), the pasteboard is 39.
    // (Two CPU grid lines crossing blend twice: about 112.)
    let on_doc = |p: [u8; 4]| if black { p[0] < 12 || (58..=70).contains(&p[0]) || (105..=120).contains(&p[0]) } else { is_checker_grey(p) };
    let (x0, w) = longest_run((0..width).map(|x| on_doc(img.at(x, cy))));
    let (y0, hgt) = longest_run((0..height).map(|y| on_doc(img.at(cx, y))));
    let doc_px = (doc as f32 * zoom).round() as i64;
    assert!((w as i64 - doc_px).abs() <= 1 && (hgt as i64 - doc_px).abs() <= 1, "zoom {zoom}: found a {w}x{hgt} document, want {doc_px}");
    (img, x0 as u32, y0 as u32, w as u32)
}

#[test]
fn checker_cells_are_8_screen_pixels_at_every_zoom() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    // The 8 px cell never changes with the zoom.
    for (zoom, doc) in [(0.5, 64), (1.0, 64), (1.1, 64), (2.0, 32), (3.3, 32), (5.5, 32), (8.0, 32), (13.0, 32), (32.0, 12)] {
        let (img, x0, y0, _) = render(&mut h, zoom, doc, false, false);
        let n = ((doc as f32 * zoom) / 8.0) as u32;
        assert!(n >= 2, "zoom {zoom}: test needs at least two cells");
        for row in 0..n.min(4) {
            for col in 0..n.min(6) {
                let (sx, sy) = (x0 + col * 8 + 4, y0 + row * 8 + 4);
                let got = img.at(sx, sy);
                let want = if (col + row) % 2 == 0 { LIGHT } else { DARK };
                assert!(
                    got[0] == want && got[1] == want && got[2] == want,
                    "zoom {zoom}, cell ({col}, {row}) centre ({sx}, {sy}): got {got:?}, want grey {want}"
                );
            }
        }
        // Crisp: every pixel of the row is exactly one of the two greys, no blended edges.
        let y = y0 + 4;
        for x in x0..x0 + (n * 8).min(48) {
            let v = img.at(x, y)[0];
            assert!(v == LIGHT || v == DARK, "zoom {zoom}: blended pixel {v} at ({x}, {y})");
        }
    }
}

#[test]
fn checker_phase_moves_with_the_image() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    // The document origin moves four screen pixels between these two views, and the checker
    // phase moves with it: the same screen position flips from a light to a dark cell.
    let mut sample_at = None;
    for (zoom, expected) in [(1.0, LIGHT), (1.25, DARK)] {
        let (img, x0, y0, _) = render(&mut h, zoom, 32, false, false);
        let (sx, sy) = *sample_at.get_or_insert((x0 + 6, y0 + 2));
        let px = img.at(sx, sy);
        assert!(px[0] == expected, "zoom {zoom}: phase at fixed screen position ({sx}, {sy}) is {px:?}, want {expected}");
    }
}

/// Screen columns inside the document (from `x0`) whose pixel is lighter than the black fill.
fn lit_columns(img: &Shot, x0: u32, y: u32, len: u32) -> Vec<u32> {
    (0..len).filter(|i| img.at(x0 + i, y)[0] > 25).collect()
}

#[test]
fn pixel_grid_shows_above_500_percent_and_lightens_a_dark_pixel_by_about_a_quarter() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    let (flat, x0, y0, _) = render(&mut h, 5.0, 48, true, true);
    assert!(lit_columns(&flat, x0, y0 + 20, 100).is_empty(), "no grid at 500%");
    for zoom in [6.0f32, 8.0, 12.0] {
        let (img, x0, y0, _) = render(&mut h, zoom, 48, true, true);
        let lit = lit_columns(&img, x0, y0 + (zoom * 3.5) as u32, (zoom * 10.0) as u32);
        assert_eq!(lit.len(), 10, "zoom {zoom}: grid columns {lit:?}");
        assert!(lit.windows(2).all(|p| p[1] - p[0] == zoom as u32), "zoom {zoom}: spacing of {lit:?}");
        let v = img.at(x0 + lit[2], y0 + (zoom * 3.5) as u32)[0];
        assert!((60..=68).contains(&v), "zoom {zoom}: a grid line over black is {v}, want about 64 (a quarter of the way to white)");
    }
}

#[test]
fn pixel_grid_is_not_drawn_over_empty_checker() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    for zoom in [8.0f32, 16.0] {
        let (img, x0, y0, _) = render(&mut h, zoom, 24, false, true);
        for y in y0..y0 + 64 {
            for x in x0..x0 + 64 {
                let v = img.at(x, y)[0];
                assert!(v == LIGHT || v == DARK, "zoom {zoom}: grid or blend {v} over empty checker at ({x}, {y})");
            }
        }
    }
}

#[test]
fn cpu_pixel_grid_shows_above_500_percent_over_content() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness_with(false) else { return };
    h.run_steps(4);
    let (flat, x0, y0, _) = render_on(&mut h, false, 5.0, 48, true, true);
    assert!(lit_columns(&flat, x0, y0 + 20, 100).is_empty(), "no grid at 500%");
    for zoom in [6.0f32, 8.0, 12.0] {
        let (img, x0, y0, _) = render_on(&mut h, false, zoom, 48, true, true);
        let y = y0 + (zoom * 3.5) as u32;
        let lit = lit_columns(&img, x0, y, (zoom * 10.0) as u32);
        assert_eq!(lit.len(), 10, "zoom {zoom}: grid columns {lit:?}");
        assert!(lit.windows(2).all(|p| p[1] - p[0] == zoom as u32), "zoom {zoom}: spacing of {lit:?}");
        let v = img.at(x0 + lit[2], y)[0];
        assert!((60..=68).contains(&v), "zoom {zoom}: a grid line over black is {v}, want about 64");
    }
}

#[test]
fn cpu_pixel_grid_is_not_drawn_over_empty_checker() {
    let _gpu = photocraft_testkit::gpu_lock();
    let Some(mut h) = harness_with(false) else { return };
    h.run_steps(4);
    // The CPU checker has soft cell edges of its own, so compare with and without the grid.
    for zoom in [8.0f32, 16.0] {
        let (without, x0, y0, _) = render_on(&mut h, false, zoom, 24, false, false);
        let (with, x1, y1, _) = render_on(&mut h, false, zoom, 24, false, true);
        assert_eq!((x0, y0), (x1, y1), "zoom {zoom}: the document moved");
        for y in y0..y0 + 64 {
            for x in x0..x0 + 64 {
                assert_eq!(with.at(x, y), without.at(x, y), "zoom {zoom}: the grid changed empty checker at ({x}, {y})");
            }
        }
    }
}

/// The pixel grid over a flat `grey` document at 800%: the largest difference, in 8-bit levels,
/// between a grid line and the pixel interior beside it.
fn grid_line_contrast_over_grey(h: &mut Harness, grey: u8) -> i32 {
    {
        let app = h.state_mut();
        while app.session.active().is_some() {
            app.run("file.close", json!({"discard": true})).expect("close");
        }
        app.run("file.new", json!({"width": 48, "height": 48, "background": "white"})).expect("new");
        app.run("edit.fill", json!({"color": format!("#{grey:02x}{grey:02x}{grey:02x}")})).expect("fill");
        app.sync_views();
        app.ui.view.show.pixel_grid = true;
    }
    control(h, "ui.set", json!({"zoom": 8.0, "center": [24.0, 24.0]}));
    h.run_steps(3);
    assert!(h.state().perf.gpu, "expected the GPU canvas");
    let rendered = h.render().expect("render");
    let img = Shot { w: rendered.width(), px: rendered.into_raw() };
    let r = h.state().last_canvas_rect;
    let (cx, cy) = (r.center().x as u32, r.center().y as u32);
    // The median of an 8 px block is a pixel interior whichever line the block straddles.
    let mut block: Vec<i32> = (0..64).map(|i| img.at(cx + i % 8, cy + i / 8)[0] as i32).collect();
    block.sort_unstable();
    let interior = block[32];
    // Walk out to the document edge: the pasteboard is far darker than any grid line.
    let near = |x: u32, y: u32| (img.at(x, y)[0] as i32 - interior).abs() < 50;
    let x0 = (0..=cx).rev().take_while(|x| near(*x, cy)).last().unwrap_or(cx);
    let y0 = (0..=cy).rev().take_while(|y| near(cx, *y)).last().unwrap_or(cy);
    // Pixel (col, row) owns the grid line on its left and top edge, 8 screen px apart.
    let y = y0 + 8 * 3 + 4;
    (2..6).map(|col| (img.at(x0 + 8 * col, y)[0] as i32 - img.at(x0 + 8 * col + 4, y)[0] as i32).abs()).max().unwrap_or(0)
}

#[test]
fn pixel_grid_does_not_flip_colour_across_mid_tones() {
    let Some(mut h) = harness() else { return };
    h.run_steps(4);
    // 0x87 and 0x90 sit either side of luma 0.55, where the line used to switch from a full white
    // line to a full black one. Both now give a faint line, so neighbouring cells match.
    for grey in [0x87u8, 0x90] {
        let c = grid_line_contrast_over_grey(&mut h, grey);
        assert!(c <= 14, "grey {grey:#04x}: grid line differs from its pixel by {c} levels, want a faint line");
    }
    // The fade must not remove the grid where it is needed: dark and light pixels keep a clear
    // line (white over 0x60, black over 0xd0 and white; black itself is in the unit tests, as the
    // dark pasteboard defeats this walk to the document edge).
    for grey in [0x60u8, 0xd0, 0xff] {
        let c = grid_line_contrast_over_grey(&mut h, grey);
        assert!(c >= 20, "grey {grey:#04x}: grid line differs from its pixel by only {c} levels");
    }
}
