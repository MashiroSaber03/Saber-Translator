//! Camera Raw-style continuous RGB ribbons. Raw bins remain exact; display filtering is local.
use crate::theme::Tokens;
use egui::{Mesh, Rect, pos2};
use photocraft_algo::histogram::{BINS, RgbHistogram};

/// A five-bin binomial display kernel softens quantization combs. It cannot bridge
/// a wide empty range. Clipping bins are pinned and excluded from filtering adjacent bins.
fn curves(h: &RgbHistogram) -> [[f32; BINS]; 3] {
    let mut curves = [[0.0; BINS]; 3];
    for (out, bins) in curves.iter_mut().zip(&h.channels) {
        for (i, value) in out.iter_mut().enumerate() {
            let center = bins.get(i).copied().unwrap_or(0) as f32;
            if i == 0 || i == BINS - 1 {
                *value = center;
            } else {
                let sample = |j: usize| bins.get(j.clamp(1, BINS - 2)).copied().unwrap_or(0) as f32;
                *value = (sample(i.saturating_sub(2)) + 4.0 * sample(i - 1) + 6.0 * center + 4.0 * sample(i + 1) + sample(i + 2)) / 16.0;
            }
        }
    }
    // One scale for RGB. Edge clipping spikes must not flatten all interior tonal detail;
    // preserve them at the edges, capped at the top. A flat black/white image uses its edge peak.
    let interior = curves.iter().flat_map(|c| c.iter().skip(1).take(BINS - 2)).copied().fold(0.0f32, f32::max);
    let peak = if interior > 0.0 { interior } else { curves.iter().flatten().copied().fold(0.0f32, f32::max) };
    let scale = if peak > 0.0 { peak * 1.1 } else { 1.0 };
    for value in curves.iter_mut().flatten() {
        *value = (*value / scale).min(1.0);
    }
    curves
}

fn quad(mesh: &mut Mesh, rect: Rect, x: [f32; 2], bottom: [f32; 2], top: [f32; 2], color: egui::Color32) {
    let vertex = mesh.vertices.len() as u32; // Bounded: at most 257 spans × 4 cuts × 3 bands × 4 vertices.
    for (x, height) in [(x[0], bottom[0]), (x[0], top[0]), (x[1], top[1]), (x[1], bottom[1])] {
        mesh.colored_vertex(pos2(x, rect.bottom() - height * rect.height()), color);
    }
    mesh.add_triangle(vertex, vertex + 1, vertex + 2);
    mesh.add_triangle(vertex, vertex + 2, vertex + 3);
}

/// Split where two RGB curves cross, so every subspan has a stable channel order. This gives
/// exact overlap colours without draw-order-dependent alpha stacking or cracks between bars.
fn span(mesh: &mut Mesh, rect: Rect, x: [f32; 2], a: [f32; 3], b: [f32; 3], t: &Tokens) {
    let mut cuts = [0.0f32, 1.0, 1.0, 1.0, 1.0];
    let mut n = 2;
    for (i, (ai, bi)) in a.iter().zip(&b).enumerate() {
        for (aj, bj) in a.iter().zip(&b).skip(i + 1) {
            let denominator = (bi - ai) - (bj - aj);
            if denominator.abs() > f32::EPSILON {
                let crossing = (aj - ai) / denominator;
                if crossing > 0.0
                    && crossing < 1.0
                    && let Some(cut) = cuts.get_mut(n)
                {
                    *cut = crossing;
                    n += 1;
                }
            }
        }
    }
    let Some(cuts) = cuts.get_mut(..n) else { return };
    cuts.sort_unstable_by(f32::total_cmp);
    for cut in cuts.windows(2) {
        let (start, end) = (cut[0], cut[1]);
        if end <= start {
            continue;
        }
        let left = std::array::from_fn::<_, 3, _>(|c| a[c] + (b[c] - a[c]) * start);
        let right = std::array::from_fn::<_, 3, _>(|c| a[c] + (b[c] - a[c]) * end);
        let mut levels = [(0usize, 1u8), (1, 2), (2, 4)];
        levels.sort_unstable_by(|(i, _), (j, _)| (left[*i] + right[*i]).total_cmp(&(left[*j] + right[*j])));
        let x = [x[0] + (x[1] - x[0]) * start, x[0] + (x[1] - x[0]) * end];
        let mut floor = [0.0; 2];
        let mut mask = 7;
        for (channel, bit) in levels {
            let top = [left[channel], right[channel]];
            if top[0] > floor[0] || top[1] > floor[1] {
                quad(mesh, rect, x, floor, top, t.histogram_fill(mask));
            }
            floor = top;
            mask &= !bit;
        }
    }
}

fn fill(rect: Rect, curves: &[[f32; BINS]; 3], t: &Tokens) -> Mesh {
    let mut mesh = Mesh::default();
    if !rect.is_finite() || !rect.is_positive() || !rect.size().is_finite() {
        return mesh;
    }
    let mut previous_x = rect.left();
    let mut previous = curves.map(|channel| channel[0]);
    for (i, ((r, g), b)) in curves[0].iter().zip(&curves[1]).zip(&curves[2]).enumerate() {
        let x = rect.left() + (i as f32 + 0.5) / BINS as f32 * rect.width();
        let next = [*r, *g, *b];
        span(&mut mesh, rect, [previous_x, x], previous, next, t);
        previous_x = x;
        previous = next;
    }
    span(&mut mesh, rect, [previous_x, rect.right()], previous, previous, t);
    mesh
}

pub(crate) fn paint(painter: &egui::Painter, rect: Rect, h: &RgbHistogram, t: &Tokens) {
    if !rect.is_finite() || !rect.is_positive() || !rect.size().is_finite() {
        return;
    }
    let curves = curves(h);
    painter.add(fill(rect, &curves, t));
    for (channel, values) in curves.iter().enumerate() {
        if !values.iter().any(|&v| v > 0.0) {
            continue;
        }
        let mut points = Vec::with_capacity(BINS + 2);
        points.push(pos2(rect.left(), rect.bottom() - values[0] * rect.height()));
        for (i, v) in values.iter().enumerate() {
            points.push(pos2(rect.left() + (i as f32 + 0.5) / BINS as f32 * rect.width(), rect.bottom() - v * rect.height()));
        }
        points.push(pos2(rect.right(), rect.bottom() - values[BINS - 1] * rect.height()));
        painter.add(egui::Shape::line(points, egui::Stroke::new(1.0, t.histogram_color(1 << channel))));
    }
}

/// One neutral tone marker, using the same RGB weights as Camera Raw's tone controls.
/// This is a constant-size overlay, independent of bin counting and normalization.
pub(crate) fn paint_sample(painter: &egui::Painter, rect: Rect, rgb: [f32; 3], t: &Tokens) {
    if !rect.is_finite() || !rect.is_positive() || !rgb.iter().all(|x| x.is_finite()) {
        return;
    }
    let painter = painter.with_clip_rect(rect);
    let [r, g, b] = rgb.map(|v| v.clamp(0.0, 1.0));
    let tone = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    let x = rect.left() + tone * rect.width();
    let color = t.histogram_color(7);
    let left = (x - 1.0).round();
    let band = Rect::from_min_max(pos2(left, rect.top()), pos2(left + 2.0, rect.bottom()));
    painter.rect_filled(band, 0.0, color.gamma_multiply(0.40));
}

#[cfg(test)]
fn mesh(rect: Rect, h: &RgbHistogram, t: &Tokens) -> Mesh {
    fill(rect, &curves(h), t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::ThemeKind;

    fn color_at(mesh: &Mesh, point: egui::Pos2) -> Option<egui::Color32> {
        mesh.indices.as_chunks::<3>().0.iter().find_map(|triangle| {
            let a = mesh.vertices[triangle[0] as usize].pos;
            let b = mesh.vertices[triangle[1] as usize].pos;
            let c = mesh.vertices[triangle[2] as usize].pos;
            let cross = |u: egui::Vec2, v: egui::Vec2| u.x * v.y - u.y * v.x;
            if cross(b - a, c - a).abs() < 0.00001 {
                return None;
            }
            let sides = [cross(b - a, point - a), cross(c - b, point - b), cross(a - c, point - c)];
            (sides.iter().all(|&s| s >= -0.00001) || sides.iter().all(|&s| s <= 0.00001)).then_some(mesh.vertices[triangle[0] as usize].color)
        })
    }

    #[test]
    fn shared_scale_and_overlaps_are_exact_for_every_theme() {
        let h = RgbHistogram { channels: [[3; BINS], [2; BINS], [1; BINS]], ..Default::default() };
        let rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 110.0));
        for kind in ThemeKind::ALL {
            let t = Tokens::for_kind(kind);
            let m = mesh(rect, &h, &t);
            for (y, mask) in [(100.0, 7), (60.0, 3), (20.0, 1)] {
                assert_eq!(color_at(&m, pos2(128.7, y)), Some(t.histogram_fill(mask)));
            }
            assert!(m.vertices.iter().all(|v| rect.contains(v.pos)));
        }
    }

    #[test]
    fn all_overlaps_and_curve_crossings_have_order_independent_colours() {
        let rect = Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 100.0));
        let mut h = RgbHistogram::default();
        for mask in 1..=7 {
            for (c, bins) in h.channels.iter_mut().enumerate() {
                bins[mask * 30..mask * 30 + 10].fill(u64::from(mask & (1 << c) != 0));
            }
        }
        for kind in ThemeKind::ALL {
            let t = Tokens::for_kind(kind);
            let m = mesh(rect, &h, &t);
            for mask in 1..=7 {
                assert_eq!(color_at(&m, pos2((mask * 30 + 5) as f32 + 0.5, 95.0)), Some(t.histogram_fill(mask as u8)));
            }
            let mut m = Mesh::default();
            span(&mut m, rect, [0.0, 256.0], [1.0, 0.25, 0.75], [0.25, 1.0, 0.75], &t);
            for (x, y, mask) in [(25.6, 90.0, 7), (25.6, 50.0, 5), (25.6, 10.0, 1), (230.4, 50.0, 6), (230.4, 10.0, 2)] {
                assert_eq!(color_at(&m, pos2(x, y)), Some(t.histogram_fill(mask)));
            }
            // Triangle interiors never stack: alpha fill must be independent of the channel order.
            let area: f32 = m
                .indices
                .as_chunks::<3>()
                .0
                .iter()
                .map(|indices| {
                    let a = m.vertices[indices[0] as usize].pos;
                    let b = m.vertices[indices[1] as usize].pos;
                    let c = m.vertices[indices[2] as usize].pos;
                    ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() * 0.5
                })
                .sum();
            assert!((area - 256.0 * 100.0 * (5.0 / 6.0)).abs() < 0.01);
        }
    }

    #[test]
    fn edge_spikes_do_not_flatten_or_bleed_into_interior_detail() {
        let mut h = RgbHistogram { channels: [[10; BINS]; 3], ..Default::default() };
        h.channels[0][0] = 1_000_000;
        h.channels[1][255] = 1_000_000;
        let p = curves(&h);
        assert_eq!(p[0][0], 1.0);
        assert_eq!(p[1][255], 1.0);
        assert!((p[0][1] - 1.0 / 1.1).abs() < 0.0001);
        assert!((p[1][254] - 1.0 / 1.1).abs() < 0.0001);
        let mut h = RgbHistogram::default();
        h.channels[0][255] = 100;
        assert!(curves(&h)[0][255] > 0.9);
        assert_eq!(curves(&h)[0][254], 0.0);
    }

    #[test]
    fn empty_and_bad_rectangles_draw_no_ribbons() {
        let t = Tokens::for_kind(ThemeKind::Pro);
        let h = RgbHistogram::default();
        assert!(mesh(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 100.0)), &h, &t).is_empty());
        assert!(mesh(Rect::NOTHING, &h, &t).is_empty());
    }
    fn covers(mesh: &Mesh, point: egui::Pos2) -> bool {
        color_at(mesh, point).is_some()
    }

    #[test]
    fn periodic_quantization_gaps_do_not_make_a_sawtooth_contour() {
        let mut h = RgbHistogram::default();
        for channel in &mut h.channels {
            for i in (30..220).step_by(3) {
                channel[i] = 100;
            }
        }
        let p = curves(&h);
        let low = p[0][40..210].iter().copied().fold(f32::INFINITY, f32::min);
        let high = p[0][40..210].iter().copied().fold(0.0f32, f32::max);
        assert!(high - low < 0.2, "quantization comb still dominates the contour: {}", high - low);
        assert_eq!(p[0][240], 0.0, "wide empty tonal ranges must stay empty");
    }

    #[test]
    fn isolated_quantization_gaps_do_not_cut_the_ribbon_into_bars() {
        let mut h = RgbHistogram::default();
        for channel in &mut h.channels {
            channel[119] = 100;
            channel[121] = 100;
        }
        let original = h.clone();
        let m = mesh(Rect::from_min_max(pos2(0.0, 0.0), pos2(256.0, 100.0)), &h, &Tokens::for_kind(ThemeKind::Pro));
        assert!(covers(&m, pos2(120.5, 85.0)), "isolated empty bin creates a full-height black stripe");
        assert!(!covers(&m, pos2(180.5, 85.0)), "a real empty tonal range must stay empty");
        assert_eq!(h, original, "display filtering must never change raw statistics");
    }
}
