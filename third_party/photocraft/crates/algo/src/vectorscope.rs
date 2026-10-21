//! Bounded hue/saturation distribution. Input colours must already be in the chosen RGB space.
//! Display orientation and colour management belong to callers, not to this analysis.
pub const HUES: usize = 128;
pub const SATURATIONS: usize = 64;

#[derive(Clone, Debug, PartialEq)]
pub struct Vectorscope {
    pub counts: Vec<u32>,
    pub samples: u64,
}

impl Default for Vectorscope {
    fn default() -> Self {
        Self { counts: vec![0; HUES * SATURATIONS], samples: 0 }
    }
}

impl Vectorscope {
    pub fn from_rgba(pixels: &[[f32; 4]]) -> Self {
        let mut out = Self::default();
        for p in pixels {
            if p[3] <= 0.0 || !p.iter().all(|v| v.is_finite()) {
                continue;
            }
            let rgb = [p[0].clamp(0.0, 1.0), p[1].clamp(0.0, 1.0), p[2].clamp(0.0, 1.0)];
            let (h, s) = hue_saturation(rgb);
            let hue = ((h * HUES as f32) as usize).min(HUES - 1);
            let saturation = ((s * (SATURATIONS - 1) as f32).round() as usize).min(SATURATIONS - 1);
            if let Some(n) = out.counts.get_mut(hue * SATURATIONS + saturation) {
                *n = n.saturating_add(1);
            }
            out.samples = out.samples.saturating_add(1);
        }
        out
    }
}

/// Hue as turns and saturation in 0..1; achromatic pixels have zero saturation.
pub fn hue_saturation([r, g, b]: [f32; 3]) -> (f32, f32) {
    let high = r.max(g).max(b);
    let low = r.min(g).min(b);
    let delta = high - low;
    if delta <= f32::EPSILON || high <= 0.0 {
        return (0.0, 0.0);
    }
    let h = if high == r {
        (g - b) / delta
    } else if high == g {
        2.0 + (b - r) / delta
    } else {
        4.0 + (r - g) / delta
    };
    ((h / 6.0).rem_euclid(1.0), delta / high)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rgb_primary_positions_neutrals_and_transparency() {
        for (rgb, expected) in [([1.0, 0.0, 0.0], 0.0), ([1.0, 1.0, 0.0], 1.0 / 6.0), ([0.0, 1.0, 0.0], 1.0 / 3.0), ([0.0, 0.0, 1.0], 2.0 / 3.0)] {
            let (h, s) = hue_saturation(rgb);
            assert!((h - expected).abs() < 1e-6);
            assert_eq!(s, 1.0);
        }
        let v = Vectorscope::from_rgba(&[[0.5, 0.5, 0.5, 1.0], [1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [f32::NAN, 0.0, 0.0, 1.0]]);
        assert_eq!(v.samples, 2);
        assert_eq!(v.counts[0], 1);
        assert_eq!(v.counts[SATURATIONS - 1], 1);
        assert_eq!(v.counts.iter().map(|&c| u64::from(c)).sum::<u64>(), v.samples);
    }
}
