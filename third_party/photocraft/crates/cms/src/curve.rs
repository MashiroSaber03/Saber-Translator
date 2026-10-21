//! One-dimensional tone curves (ICC `curv` and `para` tag types) and their inverses.

/// A tone reproduction curve.
///
/// Gamma and parametric curves are evaluated analytically, including outside `[0, 1]` (negative
/// inputs mirror, values above 1 extend the formula), so unbounded float transforms keep HDR
/// values. Sampled tables clamp their input to `[0, 1]`.
#[derive(Clone, Debug, PartialEq)]
pub enum Curve {
    Identity,
    /// `y = x^g`.
    Gamma(f64),
    /// ICC `parametricCurveType` function `kind` (0–4) with parameters `[g, a, b, c, d, e, f]`
    /// (unused ones are 0).
    Parametric {
        kind: u16,
        p: [f64; 7],
    },
    /// Samples at evenly spaced inputs over `[0, 1]`, normalised outputs.
    Table(Vec<f32>),
}

/// The sRGB transfer function (IEC 61966-2.1) as ICC parametric type 3.
pub fn srgb_trc() -> Curve {
    Curve::Parametric { kind: 3, p: [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045, 0.0, 0.0] }
}

impl Curve {
    /// Number of parameters stored for a parametric function type.
    pub fn param_count(kind: u16) -> usize {
        match kind {
            0 => 1,
            1 => 3,
            2 => 4,
            3 => 5,
            _ => 7,
        }
    }

    pub fn is_identity(&self) -> bool {
        match self {
            Curve::Identity => true,
            Curve::Gamma(g) => (*g - 1.0).abs() < 1e-9,
            Curve::Parametric { kind: 0, p } => (p[0] - 1.0).abs() < 1e-9,
            Curve::Table(t) => {
                let n = t.len();
                n < 2 || t.iter().enumerate().all(|(i, v)| (*v as f64 - i as f64 / (n - 1) as f64).abs() < 1.0 / 65535.0)
            }
            _ => false,
        }
    }

    /// Evaluates the curve at `x` (f64 reference path).
    pub fn eval64(&self, x: f64) -> f64 {
        match self {
            Curve::Identity => x,
            Curve::Gamma(g) => mirror(x, |v| v.powf(*g)),
            Curve::Parametric { kind, p } => para(*kind, p, x),
            Curve::Table(t) => table(t, x as f32) as f64,
        }
    }

    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        match self {
            Curve::Identity => x,
            Curve::Table(t) => table(t, x),
            _ => self.eval64(x as f64) as f32,
        }
    }

    /// Evaluates the inverse curve at `y`.
    pub fn eval_inverse64(&self, y: f64) -> f64 {
        match self {
            Curve::Identity => y,
            Curve::Gamma(g) => {
                if g.abs() < 1e-12 {
                    y
                } else {
                    mirror(y, |v| v.powf(1.0 / g))
                }
            }
            Curve::Parametric { kind, p } => para_inverse(*kind, p, y).unwrap_or_else(|| numeric_inverse(|x| para(*kind, p, x), y)),
            Curve::Table(t) => table_inverse(t, y as f32) as f64,
        }
    }

    #[inline]
    pub fn eval_inverse(&self, y: f32) -> f32 {
        match self {
            Curve::Identity => y,
            Curve::Table(t) => table_inverse(t, y),
            _ => self.eval_inverse64(y as f64) as f32,
        }
    }

    /// Samples `n` evenly spaced points over `[0, 1]`.
    pub fn sample(&self, n: usize) -> Vec<f32> {
        (0..n).map(|i| self.eval64(i as f64 / (n - 1) as f64) as f32).collect()
    }

    /// `true` when the curve never decreases over `[0, 1]`.
    pub fn is_monotonic(&self) -> bool {
        let s = self.sample(1024);
        s.windows(2).all(|w| w[1] >= w[0] - 1e-6) || s.windows(2).all(|w| w[1] <= w[0] + 1e-6)
    }
}

#[inline]
fn mirror(x: f64, f: impl Fn(f64) -> f64) -> f64 {
    if x < 0.0 { -f(-x) } else { f(x) }
}

fn pw(base: f64, g: f64) -> f64 {
    if base <= 0.0 { 0.0 } else { base.powf(g) }
}

fn para(kind: u16, p: &[f64; 7], x: f64) -> f64 {
    let [g, a, b, c, d, e, f] = *p;
    match kind {
        0 => mirror(x, |v| v.powf(g)),
        1 => {
            if a != 0.0 && x >= -b / a {
                pw(a * x + b, g)
            } else {
                0.0
            }
        }
        2 => {
            if a != 0.0 && x >= -b / a {
                pw(a * x + b, g) + c
            } else {
                c
            }
        }
        3 => {
            if x < 0.0 && b == 0.0 && d == 0.0 {
                // Pure gamma written as type 3: mirror negatives.
                -para(kind, p, -x)
            } else if x >= d {
                pw(a * x + b, g)
            } else {
                c * x
            }
        }
        _ => {
            if x >= d {
                pw(a * x + b, g) + e
            } else {
                c * x + f
            }
        }
    }
}

fn para_inverse(kind: u16, p: &[f64; 7], y: f64) -> Option<f64> {
    let [g, a, b, c, d, e, f] = *p;
    if g.abs() < 1e-12 {
        return None;
    }
    let root = |v: f64| if v <= 0.0 { 0.0 } else { v.powf(1.0 / g) };
    Some(match kind {
        0 => mirror(y, |v| v.powf(1.0 / g)),
        1 => {
            if a == 0.0 {
                return None;
            }
            if y <= 0.0 { -b / a } else { (root(y) - b) / a }
        }
        2 => {
            if a == 0.0 {
                return None;
            }
            if y <= c { -b / a } else { (root(y - c) - b) / a }
        }
        3 => {
            if a == 0.0 {
                return None;
            }
            if y < 0.0 && b == 0.0 && d == 0.0 {
                return para_inverse(kind, p, -y).map(|v| -v);
            }
            let yd = pw(a * d + b, g);
            if y >= yd {
                (root(y) - b) / a
            } else if c != 0.0 {
                y / c
            } else {
                0.0
            }
        }
        _ => {
            if a == 0.0 {
                return None;
            }
            let yd = pw(a * d + b, g) + e;
            if y >= yd {
                (root(y - e) - b) / a
            } else if c != 0.0 {
                (y - f) / c
            } else {
                0.0
            }
        }
    })
}

/// Bisection inverse of an increasing function on `[0, 1]`.
fn numeric_inverse(f: impl Fn(f64) -> f64, y: f64) -> f64 {
    let (f0, f1) = (f(0.0), f(1.0));
    let increasing = f1 >= f0;
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        let v = f(mid);
        if (v < y) == increasing {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

#[inline]
fn table(t: &[f32], x: f32) -> f32 {
    let n = t.len();
    match n {
        0 => x,
        1 => t[0],
        _ => {
            let pos = x.clamp(0.0, 1.0) * (n - 1) as f32;
            let i = (pos as usize).min(n - 2);
            let f = pos - i as f32;
            t[i] + (t[i + 1] - t[i]) * f
        }
    }
}

/// Exact inverse of a piecewise-linear table (binary search), for increasing or decreasing tables.
fn table_inverse(t: &[f32], y: f32) -> f32 {
    let n = t.len();
    if n < 2 {
        return y;
    }
    let increasing = t[n - 1] >= t[0];
    let (lo_v, hi_v) = if increasing { (t[0], t[n - 1]) } else { (t[n - 1], t[0]) };
    if y <= lo_v {
        // Flat start: choose the last index still at the lowest value.
        return if increasing { edge_index(t, true) } else { edge_index(t, false) };
    }
    if y >= hi_v {
        return if increasing { edge_index(t, false) } else { edge_index(t, true) };
    }
    let (mut lo, mut hi) = (0usize, n - 1);
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if (t[mid] < y) == increasing {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (a, b) = (t[lo], t[hi]);
    let f = if (b - a).abs() > 1e-12 { (y - a) / (b - a) } else { 0.5 };
    (lo as f32 + f) / (n - 1) as f32
}

/// For a flat run at the table start (`start = true`) the last index of the run, or for a flat
/// run at the end the first index of it, as a normalised input.
fn edge_index(t: &[f32], start: bool) -> f32 {
    let n = t.len();
    if start {
        let v = t[0];
        let i = t.iter().position(|x| (*x - v).abs() > 1e-7).map_or(n - 1, |i| i - 1);
        i as f32 / (n - 1) as f32
    } else {
        let v = t[n - 1];
        let i = t.iter().rposition(|x| (*x - v).abs() > 1e-7).map_or(0, |i| i + 1);
        i as f32 / (n - 1) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_matches_formula() {
        let c = srgb_trc();
        for i in 0..=255 {
            let v = i as f64 / 255.0;
            let want = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
            assert!((c.eval64(v) - want).abs() < 1e-12);
            assert!((c.eval_inverse64(want) - v).abs() < 1e-9);
        }
    }

    #[test]
    fn parametric_inverses() {
        let curves = [
            Curve::Parametric { kind: 0, p: [2.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] },
            Curve::Parametric { kind: 1, p: [2.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0] },
            Curve::Parametric { kind: 2, p: [2.0, 0.9, 0.1, 0.05, 0.0, 0.0, 0.0] },
            srgb_trc(),
            Curve::Parametric { kind: 4, p: [2.4, 0.9, 0.05, 0.08, 0.04, 0.01, 0.002] },
            Curve::Gamma(1.8),
        ];
        for c in &curves {
            for i in 1..100 {
                let x = i as f64 / 100.0;
                let y = c.eval64(x);
                assert!((c.eval_inverse64(y) - x).abs() < 1e-6, "{c:?} at {x}");
            }
        }
    }

    #[test]
    fn table_inverse_exact_on_knots_and_between() {
        let t = Curve::Gamma(2.2).sample(1024);
        let c = Curve::Table(t);
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            let back = c.eval_inverse(c.eval(x));
            assert!((back - x).abs() < 1e-4, "{x} -> {back}");
        }
        let dec = Curve::Table(vec![1.0, 0.5, 0.0]);
        assert!((dec.eval_inverse(0.25) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn hdr_and_negative_extension() {
        let c = srgb_trc();
        assert!(c.eval64(2.0) > 1.0);
        assert!((c.eval_inverse64(c.eval64(2.0)) - 2.0).abs() < 1e-9);
        let g = Curve::Gamma(2.2);
        assert!((g.eval64(-0.5) + 0.5f64.powf(2.2)).abs() < 1e-12);
    }
}
