//! Image stack statistics (Layer › Smart Objects › Stack Mode): a per-pixel, per-channel
//! statistic across N aligned images, e.g. median to remove passers-by or mean to reduce noise.
//!
//! Inputs are straight RGBA buffers of equal size. Only samples with alpha > 0 contribute; the
//! output alpha is the largest input alpha. Moments use the population definitions; skewness and
//! kurtosis are standardised (kurtosis is excess-free, a normal sample gives ≈ 3), scaled into
//! 0..=1 for display; entropy is the Shannon entropy (bits) of the 8-bit values over log2(N).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stat {
    Entropy,
    Kurtosis,
    Maximum,
    Mean,
    Median,
    Minimum,
    Range,
    Skewness,
    StandardDeviation,
    Summation,
    Variance,
}

fn stat(v: &mut [f32], s: Stat) -> f32 {
    let n = v.len() as f32;
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().sum::<f32>() / n;
    let m = |k: i32| v.iter().map(|x| (x - mean).powi(k)).sum::<f32>() / n;
    match s {
        Stat::Mean => mean,
        Stat::Summation => v.iter().sum::<f32>(),
        Stat::Minimum => v.iter().copied().fold(f32::MAX, f32::min),
        Stat::Maximum => v.iter().copied().fold(f32::MIN, f32::max),
        Stat::Range => v.iter().copied().fold(f32::MIN, f32::max) - v.iter().copied().fold(f32::MAX, f32::min),
        Stat::Median => {
            v.sort_by(f32::total_cmp);
            let k = v.len();
            if k % 2 == 1 { v[k / 2] } else { (v[k / 2 - 1] + v[k / 2]) / 2.0 }
        }
        Stat::Variance => m(2),
        Stat::StandardDeviation => m(2).sqrt(),
        Stat::Skewness => {
            let sd = m(2).sqrt();
            // Map −2..2 to 0..1 (0.5 = symmetric).
            if sd < 1e-6 { 0.5 } else { (m(3) / sd.powi(3) / 4.0 + 0.5).clamp(0.0, 1.0) }
        }
        Stat::Kurtosis => {
            let var = m(2);
            // 0..10 → 0..1.
            if var < 1e-12 { 0.0 } else { (m(4) / (var * var) / 10.0).clamp(0.0, 1.0) }
        }
        Stat::Entropy => {
            if v.len() < 2 {
                return 0.0;
            }
            let mut counts: std::collections::HashMap<u8, u32> = std::collections::HashMap::new();
            for x in v.iter() {
                *counts.entry((x.clamp(0.0, 1.0) * 255.0).round() as u8).or_default() += 1;
            }
            let h: f32 = counts
                .values()
                .map(|&c| {
                    let p = c as f32 / n;
                    -p * p.log2()
                })
                .sum();
            h / n.log2()
        }
    }
}

/// Combines `layers` (each `len` pixels) with statistic `s`. Colour results retain their float
/// range; quantisation/clipping belongs to the destination surface, not the statistic.
pub fn combine(layers: &[Vec<[f32; 4]>], s: Stat) -> Vec<[f32; 4]> {
    let len = layers.iter().map(Vec::len).max().unwrap_or(0);
    let mut out = vec![[0.0f32; 4]; len];
    let mut samples: [Vec<f32>; 3] = Default::default();
    for (i, o) in out.iter_mut().enumerate() {
        for v in samples.iter_mut() {
            v.clear();
        }
        let mut alpha = 0.0f32;
        for l in layers {
            if let Some(p) = l.get(i)
                && p[3] > 0.0
            {
                for c in 0..3 {
                    samples[c].push(p[c]);
                }
                alpha = alpha.max(p[3]);
            }
        }
        if alpha <= 0.0 {
            continue;
        }
        for c in 0..3 {
            o[c] = stat(&mut samples[c], s);
        }
        o[3] = alpha;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn px(v: f32) -> Vec<[f32; 4]> {
        vec![[v, v, v, 1.0]]
    }

    #[test]
    fn basic_statistics() {
        let layers = vec![px(0.1), px(0.9), px(0.2), px(0.3), px(0.1)];
        let at = |s| combine(&layers, s)[0][0];
        assert!((at(Stat::Mean) - 0.32).abs() < 1e-6);
        assert!((at(Stat::Median) - 0.2).abs() < 1e-6);
        assert_eq!(at(Stat::Minimum), 0.1);
        assert_eq!(at(Stat::Maximum), 0.9);
        assert!((at(Stat::Range) - 0.8).abs() < 1e-6);
        assert!((at(Stat::Summation) - 1.6).abs() < 1e-6);
        let var = at(Stat::Variance);
        assert!((at(Stat::StandardDeviation) - var.sqrt()).abs() < 1e-6);
        assert!(at(Stat::Skewness) > 0.5, "right-skewed");
        assert!(at(Stat::Entropy) > 0.0 && at(Stat::Entropy) <= 1.0);
        assert!(at(Stat::Kurtosis) > 0.0);
    }

    #[test]
    fn statistics_preserve_hdr_and_signed_samples() {
        let layers = vec![px(-2.0), px(4.0)];
        for (mode, expected) in [
            (Stat::Mean, 1.0),
            (Stat::Median, 1.0),
            (Stat::Minimum, -2.0),
            (Stat::Maximum, 4.0),
            (Stat::Range, 6.0),
            (Stat::Summation, 2.0),
            (Stat::Variance, 9.0),
            (Stat::StandardDeviation, 3.0),
        ] {
            assert_eq!(combine(&layers, mode)[0], [expected, expected, expected, 1.0], "{mode:?}");
        }
        for mode in [Stat::Mean, Stat::Median] {
            assert_eq!(combine(&[px(2.0), px(4.0)], mode)[0], [3.0, 3.0, 3.0, 1.0]);
            assert_eq!(combine(&[px(-2.0), px(-4.0)], mode)[0], [-3.0, -3.0, -3.0, 1.0]);
        }
        // These display-oriented statistics still apply their own normalisation.
        for mode in [Stat::Entropy, Stat::Skewness, Stat::Kurtosis] {
            assert!((0.0..=1.0).contains(&combine(&layers, mode)[0][0]), "{mode:?}");
        }
    }

    #[test]
    fn transparent_samples_are_ignored() {
        let layers = vec![vec![[0.5, 0.5, 0.5, 1.0]], vec![[1.0, 1.0, 1.0, 0.0]]];
        assert_eq!(combine(&layers, Stat::Mean)[0], [0.5, 0.5, 0.5, 1.0]);
        let empty = vec![vec![[1.0, 1.0, 1.0, 0.0]]];
        assert_eq!(combine(&empty, Stat::Mean)[0][3], 0.0);
    }

    #[test]
    fn median_removes_outlier() {
        let layers = vec![px(0.5), px(0.5), px(1.0)];
        assert_eq!(combine(&layers, Stat::Median)[0][0], 0.5);
    }
}
