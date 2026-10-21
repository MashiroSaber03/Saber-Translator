//! Gradient Fill layers: their colour ramp (colour stops with midpoints, opacity stops, dither)
//! and the on-canvas geometry of the Gradient tool's live gradients, where a start and an end
//! point stand for the fill's angle, scale and centre offset.
//!
//! The mapping matches the Gradient tool's destructive drag (`photocraft_algo::paint`): a
//! Linear gradient runs from the start (t = 0) to the end point (t = 1); Radial, Angle,
//! Reflected and Diamond gradients are centred on the start point and reach t = 1 at the end
//! point's distance. Fill and painted gradients share this geometry (see the tests).

use photocraft_doc::{Fill, GradientStyle};
use photocraft_geom::Rect;

use crate::effects::gradient_units;

/// Smallest gradient length a handle drag produces (pixels), so the fill stays invertible.
const MIN_LEN: f32 = 0.5;

/// A gradient fill's colour ramp, prepared for one render (colours converted once, while the
/// document's CMYK profile is active).
#[derive(Clone, Debug)]
pub struct Ramp<const N: usize = 4> {
    color: Vec<(f32, [f32; N])>,
    mids: Vec<f32>,
    opacity: Vec<(f32, f32)>,
}

impl Ramp {
    /// The ramp of a gradient fill (`None` for other fills).
    pub fn new(f: &Fill) -> Option<Ramp> {
        Self::with_colors(f, |c| {
            let rgb = c.to_rgb();
            [rgb[0], rgb[1], rgb[2], c.alpha]
        })
    }
}

impl<const N: usize> Ramp<N> {
    fn with_colors(f: &Fill, convert: impl Fn(photocraft_doc::Color) -> [f32; N]) -> Option<Self> {
        let Fill::Gradient { stops, opacity_stops, midpoints, .. } = f else { return None };
        let mut color: Vec<_> = stops.iter().map(|(p, c)| (*p, convert(*c))).collect();
        color.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut opacity = opacity_stops.clone();
        opacity.sort_by(|a, b| a.0.total_cmp(&b.0));
        Some(Ramp { color, mids: midpoints.clone(), opacity })
    }

    /// Colour and alpha at `t` (`0..=1`).
    pub fn sample(&self, t: f32) -> [f32; N] {
        let mut c = sample_color(&self.color, &self.mids, t);
        if let Some(alpha) = c.last_mut() {
            *alpha *= sample_opacity(&self.opacity, t);
        }
        c
    }
}

/// Where a segment of relative position `u` lands once its midpoint is `m` (0.5 = unchanged):
/// the colour is half-way at `m`, linear on either side.
pub fn midpoint_remap(u: f32, m: f32) -> f32 {
    let m = m.clamp(0.05, 0.95);
    if (m - 0.5).abs() < 1e-6 {
        u
    } else if u <= m {
        0.5 * u / m
    } else {
        0.5 + 0.5 * (u - m) / (1.0 - m)
    }
}

fn sample_color<const N: usize>(stops: &[(f32, [f32; N])], mids: &[f32], t: f32) -> [f32; N] {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else { return [0.0; N] };
    if t <= first.0 {
        return first.1;
    }
    for (i, w) in stops.windows(2).enumerate() {
        let (a, b) = (&w[0], &w[1]);
        if t <= b.0 {
            let u = if b.0 > a.0 { (t - a.0) / (b.0 - a.0) } else { 0.0 };
            let k = midpoint_remap(u, mids.get(i).copied().unwrap_or(0.5));
            return std::array::from_fn(|c| a.1[c] + (b.1[c] - a.1[c]) * k);
        }
    }
    last.1
}

fn sample_opacity(stops: &[(f32, f32)], t: f32) -> f32 {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else { return 1.0 };
    if t <= first.0 {
        return first.1;
    }
    for w in stops.windows(2) {
        if t <= w[1].0 {
            let u = if w[1].0 > w[0].0 { (t - w[0].0) / (w[1].0 - w[0].0) } else { 0.0 };
            return w[0].1 + (w[1].1 - w[0].1) * u;
        }
    }
    last.1
}

/// Adds one level of the shared dither noise to a colour (as the Gradient tool does).
pub fn dither(c: &mut [f32; 4], x: i32, y: i32) {
    dither_channels(&mut c[..3], x, y);
}

fn dither_channels(c: &mut [f32], x: i32, y: i32) {
    dither_channels_to(c, x, y, None);
}

/// [`dither_channels`], then rounded to `quantum` steps when given. The noise is a fraction of a
/// level that only means something once the value lands on a level, as it does in a painted
/// gradient: left in float, a dark channel lifted 0.002 above black is still 0.002 to the next
/// blend, and Color Dodge under a bright layer turns that into visible speckle (#2755).
fn dither_channels_to(c: &mut [f32], x: i32, y: i32, quantum: Option<f32>) {
    let n = (photocraft_color::dither_noise(x, y) - 0.5) / 255.0;
    for ch in c {
        *ch = (*ch + n).clamp(0.0, 1.0);
        if let Some(q) = quantum {
            *ch = (*ch * q + 0.5).floor() / q;
        }
    }
}

/// Renders a gradient fill over `rect`, laid out in `frame` (row-major RGBA).
pub fn render(f: &Fill, rect: Rect, frame: Rect) -> Vec<[f32; 4]> {
    render_quantized(f, rect, frame, None)
}

/// [`render`] with dithered pixels rounded to `quantum` steps (the document depth's levels, see
/// [`crate::adjustment_quantum`]), as a gradient painted into a layer of that depth stores them.
pub fn render_quantized(f: &Fill, rect: Rect, frame: Rect, quantum: Option<f32>) -> Vec<[f32; 4]> {
    render_ramp(f, rect, frame, Ramp::new(f), quantum)
}

/// Native CMYK solid/gradient pixels at document depth; patterns return `None`.
/// Use the same samples for rendering and PSD caches to avoid a lossy RGB separation.
pub fn render_cmyk_fill(f: &Fill, rect: Rect, frame: Rect, depth: photocraft_color::SampleType) -> Option<photocraft_raster::Surface> {
    let native = |c: photocraft_doc::Color| {
        let c = c.in_mode(photocraft_color::ColorMode::Cmyk);
        [c.c[0], c.c[1], c.c[2], c.c[3], c.alpha]
    };
    let mut surface = photocraft_raster::Surface::new(photocraft_color::PixelFormat::new(photocraft_color::ColorMode::Cmyk, depth, true));
    match f {
        Fill::Solid(c) => surface.fill_rect(rect, &native(*c)),
        Fill::Gradient { .. } => {
            let px = render_ramp(f, rect, frame, Ramp::with_colors(f, native), None);
            surface.write_region(rect, px.as_flattened());
        }
        Fill::Pattern { .. } => return None,
    }
    surface.prune();
    Some(surface)
}

fn render_ramp<const N: usize>(f: &Fill, rect: Rect, frame: Rect, ramp: Option<Ramp<N>>, quantum: Option<f32>) -> Vec<[f32; N]> {
    let mut px = vec![[0.0f32; N]; rect.width() as usize * rect.height() as usize];
    let (Some(ramp), Fill::Gradient { angle, scale, style, reverse, offset, dither: dith, .. }) = (ramp, f) else { return px };
    let w = rect.width() as usize;
    if w == 0 {
        return px;
    }
    // Linear and Reflected end points on whole pixels, as Photoshop (crate::fill_layout).
    let (angle, scale, offset) = &crate::fill_layout::gradient_layout(*style, *angle, *scale, *offset, frame);
    let row = |y: i32, out: &mut [[f32; N]]| {
        for (i, p) in out.iter_mut().enumerate() {
            let x = rect.x0 + i as i32;
            let t = crate::effects::gradient_t(*style, *angle, *scale, *reverse, *offset, frame, x as f32 + 0.5, y as f32 + 0.5);
            *p = ramp.sample(t);
            if *dith {
                dither_channels_to(p.get_mut(..N.saturating_sub(1)).unwrap_or_default(), x, y, quantum);
            }
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        px.par_chunks_mut(w).enumerate().for_each(|(j, out)| row(rect.y0 + j as i32, out));
    }
    #[cfg(target_arch = "wasm32")]
    for (j, out) in px.chunks_mut(w).enumerate() {
        row(rect.y0 + j as i32, out);
    }
    px
}

fn frame_dims(frame: Rect) -> (f32, f32) {
    (frame.width().max(1) as f32, frame.height().max(1) as f32)
}

/// The centre a gradient is laid out around: the frame's centre moved by `offset`.
fn centre(offset: (f32, f32), frame: Rect) -> [f32; 2] {
    let (w, h) = frame_dims(frame);
    [frame.x0 as f32 + w / 2.0 + offset.0 * w, frame.y0 as f32 + h / 2.0 + offset.1 * h]
}

/// The gradient's reach from its centre at scale 1: the full chord for Linear, half the chord
/// for Reflected, the radius for the others.
fn unit_reach(style: GradientStyle, angle: f32, frame: Rect) -> f32 {
    let (w, h) = frame_dims(frame);
    let (chord, len) = gradient_units(angle, w, h);
    match style {
        GradientStyle::Linear => chord,
        GradientStyle::Reflected => chord / 2.0,
        GradientStyle::Radial | GradientStyle::Angle | GradientStyle::Diamond => len / 2.0,
    }
}

/// The start and end points (document pixels) of a gradient laid out in `frame`.
pub fn handles(style: GradientStyle, angle: f32, scale: f32, offset: (f32, f32), frame: Rect) -> ([f32; 2], [f32; 2]) {
    let c = centre(offset, frame);
    let (s, co) = angle.to_radians().sin_cos();
    let dir = [co, -s];
    let reach = unit_reach(style, angle, frame) * scale.max(1e-3);
    match style {
        GradientStyle::Linear => {
            let h = reach / 2.0;
            ([c[0] - dir[0] * h, c[1] - dir[1] * h], [c[0] + dir[0] * h, c[1] + dir[1] * h])
        }
        _ => (c, [c[0] + dir[0] * reach, c[1] + dir[1] * reach]),
    }
}

/// The handles of a gradient fill in `frame` (`None` for other fills).
pub fn fill_handles(f: &Fill, frame: Rect) -> Option<([f32; 2], [f32; 2])> {
    match f {
        Fill::Gradient { style, angle, scale, offset, .. } => Some(handles(*style, *angle, *scale, *offset, frame)),
        _ => None,
    }
}

/// Angle (degrees), scale and centre offset that put a gradient's start and end points at
/// `from` and `to` in `frame` (inverse of [`handles`]). A zero-length drag keeps the direction
/// of `fallback_angle`.
pub fn from_handles(style: GradientStyle, from: [f32; 2], to: [f32; 2], frame: Rect, fallback_angle: f32) -> (f32, f32, (f32, f32)) {
    let (dx, dy) = (to[0] - from[0], to[1] - from[1]);
    let len = dx.hypot(dy);
    let angle = if len > 1e-4 { (-dy).atan2(dx).to_degrees() } else { fallback_angle };
    let len = len.max(MIN_LEN);
    let c = match style {
        GradientStyle::Linear => [(from[0] + to[0]) / 2.0, (from[1] + to[1]) / 2.0],
        _ => from,
    };
    let (w, h) = frame_dims(frame);
    let offset = ((c[0] - frame.x0 as f32 - w / 2.0) / w, (c[1] - frame.y0 as f32 - h / 2.0) / h);
    let scale = len / unit_reach(style, angle, frame);
    (angle, scale, offset)
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_doc::Color;

    const STYLES: [GradientStyle; 5] = [GradientStyle::Linear, GradientStyle::Radial, GradientStyle::Angle, GradientStyle::Reflected, GradientStyle::Diamond];

    #[test]
    fn native_cmyk_ramp_interpolates_black_and_alpha_separately() {
        let a = Color { mode: photocraft_color::ColorMode::Cmyk, c: [0.0, 0.2, 0.4, 0.6], alpha: 0.5 };
        let b = Color { c: [0.4, 0.6, 0.8, 1.0], alpha: 1.0, ..a };
        let mut f = Fill::gradient(vec![(1.0, b), (0.0, a)], 0.0, 1.0, GradientStyle::Linear, false);
        if let Fill::Gradient { midpoints, opacity_stops, .. } = &mut f {
            *midpoints = vec![0.25];
            *opacity_stops = vec![(1.0, 0.0), (0.0, 1.0)];
        }
        let ramp = Ramp::with_colors(&f, |c| [c.c[0], c.c[1], c.c[2], c.c[3], c.alpha]).unwrap();
        let expected = [0.2, 0.4, 0.6, 0.8, 0.5625];
        for (v, expected) in ramp.sample(0.25).into_iter().zip(expected) {
            assert!((v - expected).abs() < 1e-6);
        }
        let mut dithered = ramp.sample(0.25);
        dither_channels(&mut dithered[..4], 3, 7);
        assert_eq!(dithered[4], expected[4]);
        let noise = (photocraft_color::dither_noise(3, 7) - 0.5) / 255.0;
        for i in 0..4 {
            assert!((dithered[i] - expected[i] - noise).abs() < 1e-6);
        }
    }

    #[test]
    fn handles_round_trip() {
        let frame = Rect::new(10, -4, 210, 96);
        for style in STYLES {
            for (from, to) in [([30.0, 40.0], [180.0, 10.0]), ([100.0, 50.0], [100.0, 90.0]), ([5.0, 5.0], [-40.0, 70.0])] {
                let (a, s, o) = from_handles(style, from, to, frame, 0.0);
                let (f2, t2) = handles(style, a, s, o, frame);
                for (p, q) in [(from, f2), (to, t2)] {
                    assert!((p[0] - q[0]).abs() < 1e-2 && (p[1] - q[1]).abs() < 1e-2, "{style:?}: {p:?} vs {q:?}");
                }
            }
        }
    }

    #[test]
    fn handles_map_to_the_ends_of_the_ramp() {
        let frame = Rect::new(0, 0, 100, 60);
        let (a, s, o) = from_handles(GradientStyle::Linear, [20.0, 30.0], [80.0, 30.0], frame, 0.0);
        // Linear gradients sample each pixel's top-left corner (effects::gradient_t): the pixel
        // whose corner is the handle, centred half a pixel further, is where t is 0 or 1.
        let t = |x: f32| crate::effects::gradient_t(GradientStyle::Linear, a, s, false, o, frame, x + 0.5, 30.5);
        assert!(t(20.0).abs() < 1e-4 && (t(80.0) - 1.0).abs() < 1e-4 && (t(50.0) - 0.5).abs() < 1e-4);
        let (a, s, o) = from_handles(GradientStyle::Radial, [40.0, 20.0], [40.0, 50.0], frame, 0.0);
        let r = |x: f32, y: f32| crate::effects::gradient_t(GradientStyle::Radial, a, s, false, o, frame, x, y);
        assert!(r(40.0, 20.0) < 1e-4 && (r(70.0, 20.0) - 1.0).abs() < 1e-3 && (r(55.0, 20.0) - 0.5).abs() < 1e-3);
    }

    #[test]
    fn ramp_midpoints_and_opacity() {
        let f = Fill::Gradient {
            stops: vec![(1.0, Color::WHITE), (0.0, Color::BLACK)],
            angle: 0.0,
            scale: 1.0,
            style: GradientStyle::Linear,
            reverse: false,
            opacity_stops: vec![(0.0, 1.0), (1.0, 0.0)],
            midpoints: vec![0.25],
            offset: (0.0, 0.0),
            dither: false,
            align: true,
        };
        let r = Ramp::new(&f).unwrap();
        // Unsorted stops are sorted; the midpoint puts 50 % gray at t = 0.25.
        assert!((r.sample(0.25)[0] - 0.5).abs() < 1e-5);
        assert!((r.sample(0.25)[3] - 0.75).abs() < 1e-5);
        assert_eq!(r.sample(1.0), [1.0, 1.0, 1.0, 0.0]);
        assert_eq!(midpoint_remap(0.3, 0.5), 0.3);
        assert!(Ramp::new(&Fill::Solid(Color::BLACK)).is_none());
    }

    #[test]
    fn render_handles_empty_and_degenerate_rects() {
        let f = Fill::gradient(vec![], 0.0, 0.0, GradientStyle::Angle, true);
        assert!(render(&f, Rect::new(0, 0, 0, 5), Rect::new(0, 0, 0, 0)).is_empty());
        let px = render(&f, Rect::new(0, 0, 3, 2), Rect::new(0, 0, 0, 0));
        assert_eq!(px.len(), 6);
    }
}
