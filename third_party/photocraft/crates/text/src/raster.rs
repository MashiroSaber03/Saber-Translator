//! Anti-aliased path rasterizer: exact signed-area accumulation (the "font-rs" technique), f32
//! coverage, nonzero-ish fill (|winding| clamped to 1). Curves are flattened to lines.

/// A 2D affine map `[a c e; b d f]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Xform(pub [f64; 6]);

impl Xform {
    pub const IDENTITY: Xform = Xform([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let [a, b, c, d, e, f] = self.0;
        (a * x + c * y + e, b * x + d * y + f)
    }
    /// `self ∘ o` (apply `o` first).
    pub fn mul(&self, o: &Xform) -> Xform {
        let [a, b, c, d, e, f] = self.0;
        let [oa, ob, oc, od, oe, of] = o.0;
        Xform([a * oa + c * ob, b * oa + d * ob, a * oc + c * od, b * oc + d * od, a * oe + c * of + e, b * oe + d * of + f])
    }
    /// Largest scale factor (for flattening tolerance).
    pub fn max_scale(&self) -> f64 {
        let [a, b, c, d, ..] = self.0;
        (a * a + b * b).sqrt().max((c * c + d * d).sqrt())
    }
}

/// Receives flattened line segments.
pub trait LineSink {
    fn line(&mut self, p0: (f64, f64), p1: (f64, f64));
}

/// Collects the bounding box of everything drawn.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bounds {
    pub rect: Option<[f64; 4]>,
}

impl Bounds {
    pub fn add(&mut self, p: (f64, f64)) {
        if !(p.0.is_finite() && p.1.is_finite()) {
            return;
        }
        self.rect = Some(match self.rect {
            None => [p.0, p.1, p.0, p.1],
            Some(r) => [r[0].min(p.0), r[1].min(p.1), r[2].max(p.0), r[3].max(p.1)],
        });
    }
}

impl LineSink for Bounds {
    fn line(&mut self, p0: (f64, f64), p1: (f64, f64)) {
        self.add(p0);
        self.add(p1);
    }
}

impl LineSink for Coverage {
    fn line(&mut self, p0: (f64, f64), p1: (f64, f64)) {
        Coverage::line(self, p0, p1);
    }
}

/// Adds a transformed axis-aligned rectangle.
pub fn rect_to(sink: &mut impl LineSink, xf: &Xform, x0: f64, y0: f64, x1: f64, y1: f64) {
    let p = [xf.apply(x0, y0), xf.apply(x1, y0), xf.apply(x1, y1), xf.apply(x0, y1)];
    for i in 0..4 {
        sink.line(p[i], p[(i + 1) % 4]);
    }
}

/// Coverage accumulation buffer of `width × height` pixels.
pub struct Coverage {
    pub width: usize,
    pub height: usize,
    acc: Vec<f32>,
}

impl Coverage {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height, acc: vec![0.0; width * height + 2] }
    }

    /// Adds a line segment (pixel coordinates, y down).
    pub fn line(&mut self, p0: (f64, f64), p1: (f64, f64)) {
        let (w, h) = (self.width as f32, self.height as f32);
        if p0.1 == p1.1 || !(p0.0.is_finite() && p0.1.is_finite() && p1.0.is_finite() && p1.1.is_finite()) {
            return;
        }
        let (dir, (x0, y0), (x1, y1)) = if p0.1 < p1.1 { (1.0f32, p0, p1) } else { (-1.0f32, p1, p0) };
        let (x0, y0, x1, y1) = (x0 as f32, y0 as f32, x1 as f32, y1 as f32);
        if y1 <= 0.0 || y0 >= h {
            return;
        }
        let dxdy = (x1 - x0) / (y1 - y0);
        let mut x = x0;
        let ys = y0.max(0.0);
        if y0 < 0.0 {
            x -= y0 * dxdy;
        }
        let ye = y1.min(h);
        let stride = self.width;
        let mut y = ys;
        let mut row = ys.floor() as usize;
        while y < ye {
            let ynext = ((row + 1) as f32).min(ye);
            let dy = ynext - y;
            let xnext = x + dxdy * dy;
            let d = dy * dir;
            let (xa, xb) = if x < xnext { (x, xnext) } else { (xnext, x) };
            let (xa, xb) = (xa.clamp(0.0, w), xb.clamp(0.0, w));
            let ls = row * stride;
            let x0f = xa.floor();
            let x0i = x0f as usize;
            let x1c = xb.ceil();
            let x1i = x1c as usize;
            if x1i <= x0i + 1 {
                let xmf = 0.5 * (xa + xb) - x0f;
                self.acc[ls + x0i] += d - d * xmf;
                self.acc[ls + x0i + 1] += d * xmf;
            } else {
                let s = 1.0 / (xb - xa);
                let x0frac = xa - x0f;
                let a0 = 0.5 * s * (1.0 - x0frac) * (1.0 - x0frac);
                let x1frac = xb - x1c + 1.0;
                let am = 0.5 * s * x1frac * x1frac;
                self.acc[ls + x0i] += d * a0;
                if x1i == x0i + 2 {
                    self.acc[ls + x0i + 1] += d * (1.0 - a0 - am);
                } else {
                    let a1 = s * (1.5 - x0frac);
                    self.acc[ls + x0i + 1] += d * (a1 - a0);
                    for xi in x0i + 2..x1i - 1 {
                        self.acc[ls + xi] += d * s;
                    }
                    let a2 = a1 + (x1i - x0i - 3) as f32 * s;
                    self.acc[ls + x1i - 1] += d * (1.0 - a2 - am);
                }
                self.acc[ls + x1i] += d * am;
            }
            x = xnext;
            y = ynext;
            row += 1;
        }
    }

    /// Resolves the accumulation into per-pixel coverage in 0..=1.
    pub fn finish(self) -> Vec<f32> {
        let n = self.width * self.height;
        let mut out = Vec::with_capacity(n);
        let mut sum = 0.0f32;
        for v in &self.acc[..n] {
            sum += v;
            out.push(sum.abs().min(1.0));
        }
        out
    }
}

/// Outline pen that transforms font-space points (y up) and flattens curves into a [`Coverage`].
pub struct Pen<'a, S: LineSink> {
    pub cov: &'a mut S,
    pub xf: Xform,
    start: (f64, f64),
    last: (f64, f64),
    /// Flattening tolerance in output pixels.
    tol: f64,
}

impl<'a, S: LineSink> Pen<'a, S> {
    pub fn new(cov: &'a mut S, xf: Xform) -> Self {
        Pen { cov, xf, start: (0.0, 0.0), last: (0.0, 0.0), tol: 0.03 }
    }
    fn to(&mut self, p: (f64, f64)) {
        self.cov.line(self.last, p);
        self.last = p;
    }
    fn segments(&self, a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> usize {
        // Deviation of the control polygon from the chord bounds the flattening error.
        let dd = ((a.0 - 2.0 * b.0 + c.0).powi(2) + (a.1 - 2.0 * b.1 + c.1).powi(2)).sqrt();
        ((0.75 * dd / self.tol).sqrt().ceil() as usize).clamp(1, 100)
    }
}

impl<S: LineSink> skrifa::outline::OutlinePen for Pen<'_, S> {
    fn move_to(&mut self, x: f32, y: f32) {
        self.close();
        let p = self.xf.apply(x as f64, y as f64);
        self.start = p;
        self.last = p;
    }
    fn line_to(&mut self, x: f32, y: f32) {
        let p = self.xf.apply(x as f64, y as f64);
        self.to(p);
    }
    fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let a = self.last;
        let b = self.xf.apply(cx0 as f64, cy0 as f64);
        let c = self.xf.apply(x as f64, y as f64);
        let n = self.segments(a, b, c);
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let mt = 1.0 - t;
            let p = (mt * mt * a.0 + 2.0 * mt * t * b.0 + t * t * c.0, mt * mt * a.1 + 2.0 * mt * t * b.1 + t * t * c.1);
            self.to(p);
        }
    }
    fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let a = self.last;
        let b = self.xf.apply(cx0 as f64, cy0 as f64);
        let c = self.xf.apply(cx1 as f64, cy1 as f64);
        let d = self.xf.apply(x as f64, y as f64);
        let n = self.segments(a, b, c).max(self.segments(b, c, d));
        for i in 1..=n {
            let t = i as f64 / n as f64;
            let mt = 1.0 - t;
            let (k0, k1, k2, k3) = (mt * mt * mt, 3.0 * mt * mt * t, 3.0 * mt * t * t, t * t * t);
            let p = (k0 * a.0 + k1 * b.0 + k2 * c.0 + k3 * d.0, k0 * a.1 + k1 * b.1 + k2 * c.1 + k3 * d.1);
            self.to(p);
        }
    }
    fn close(&mut self) {
        if self.last != self.start {
            let s = self.start;
            self.to(s);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(v: &[f32]) -> f32 {
        v.iter().sum()
    }

    #[test]
    fn square_area_exact() {
        let mut c = Coverage::new(10, 10);
        rect_to(&mut c, &Xform::IDENTITY, 2.25, 3.5, 6.75, 8.0);
        let v = c.finish();
        assert!((sum(&v) - 4.5 * 4.5).abs() < 1e-3, "{}", sum(&v));
        assert_eq!(v[5 * 10 + 4], 1.0);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn triangle_area_and_clipping() {
        let mut c = Coverage::new(8, 8);
        // Right triangle partly outside on the left and bottom.
        let pts = [(-4.0, 2.0), (6.0, 2.0), (-4.0, 12.0)];
        for i in 0..3 {
            c.line(pts[i], pts[(i + 1) % 3]);
        }
        let v = c.finish();
        // Visible part: region x in [0,6], y in [2,8] under the hypotenuse x + y <= 8.
        // Area = ∫_{y=2}^{8} (8 - y) dy = 18.
        assert!((sum(&v) - 18.0).abs() < 1e-3, "{}", sum(&v));
        assert!(v.iter().all(|x| (0.0..=1.0).contains(x)));
    }

    #[test]
    fn circle_area_via_pen() {
        use skrifa::outline::OutlinePen;
        let mut c = Coverage::new(40, 40);
        {
            let mut p = Pen::new(&mut c, Xform([1.0, 0.0, 0.0, -1.0, 20.0, 20.0]));
            // Cubic circle approximation, r = 15.
            let (r, k) = (15.0f32, 0.552_284_8f32 * 15.0);
            p.move_to(r, 0.0);
            p.curve_to(r, k, k, r, 0.0, r);
            p.curve_to(-k, r, -r, k, -r, 0.0);
            p.curve_to(-r, -k, -k, -r, 0.0, -r);
            p.curve_to(k, -r, r, -k, r, 0.0);
            p.close();
        }
        let a = sum(&c.finish());
        let exact = std::f32::consts::PI * 225.0;
        assert!((a - exact).abs() / exact < 0.005, "{a} vs {exact}");
    }
}
