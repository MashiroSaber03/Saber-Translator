//! App-owned print-mark composition in points from the sheet's top-left.
use super::Marks;
use craft_print::rgb::{Primitive, Shape};

pub(super) fn primitives(marks: Marks, (x, y, w, h): (f64, f64, f64, f64)) -> Vec<Primitive> {
    let mut out = Vec::with_capacity(24);
    let line = |out: &mut Vec<Primitive>, x0, y0, x1, y1| {
        out.push(Primitive { shape: Shape::Line { from: (x0, y0), to: (x1, y1) }, width: 0.3, color: [0; 3] });
    };
    let (gap, len) = (6.0, 18.0);
    if marks.corner_crop {
        for (cx, cy, dx, dy) in [(x, y, -1.0, -1.0), (x + w, y, 1.0, -1.0), (x, y + h, -1.0, 1.0), (x + w, y + h, 1.0, 1.0)] {
            line(&mut out, cx + dx * gap, cy, cx + dx * (gap + len), cy);
            line(&mut out, cx, cy + dy * gap, cx, cy + dy * (gap + len));
        }
    }
    if marks.center_crop {
        let (mx, my) = (x + w / 2.0, y + h / 2.0);
        line(&mut out, mx, y - gap, mx, y - gap - len);
        line(&mut out, mx, y + h + gap, mx, y + h + gap + len);
        line(&mut out, x - gap, my, x - gap - len, my);
        line(&mut out, x + w + gap, my, x + w + gap + len, my);
    }
    if marks.registration {
        let (r, off) = (5.0, gap + len / 2.0);
        for (cx, cy) in [(x + w / 2.0, y - off), (x + w / 2.0, y + h + off), (x - off, y + h / 2.0), (x + w + off, y + h / 2.0)] {
            out.push(Primitive { shape: Shape::Ellipse { bounds: (cx - r, cy - r, 2.0 * r, 2.0 * r) }, width: 0.3, color: [0; 3] });
            line(&mut out, cx - r * 1.6, cy, cx + r * 1.6, cy);
            line(&mut out, cx, cy - r * 1.6, cx, cy + r * 1.6);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn marks_surround_the_final_image_box_in_physical_points() {
        let out = primitives(Marks { corner_crop: true, center_crop: true, registration: true }, (100.0, 200.0, 144.0, 72.0));
        assert_eq!(out.len(), 24);
        assert_eq!(out[0].shape, Shape::Line { from: (94.0, 200.0), to: (76.0, 200.0) });
        assert_eq!(out[12].shape, Shape::Ellipse { bounds: (167.0, 180.0, 10.0, 10.0) });
        assert!(out.iter().all(|p| p.width == 0.3 && p.color == [0; 3]));
        assert!(primitives(Marks::default(), (0.0, 0.0, 10.0, 10.0)).is_empty());
    }
}
