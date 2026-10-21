//! Bounded RGB histograms of straight, normalized float RGBA samples.
//!
//! No colour conversion is performed: bins describe the supplied sample domain, not an
//! assumed sRGB/ICC display space. 256 bins are a display resolution, not an input bit depth.
pub const BINS: usize = 256;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RgbHistogram {
    pub channels: [[u64; BINS]; 3],
    pub samples: u64,
    pub transparent: u64,
    pub invalid: u64,
    /// Finite samples outside 0..=1 are included in the edge bins and counted separately.
    pub underflow: [u64; 3],
    pub overflow: [u64; 3],
    /// Exact endpoints, before display-bin rounding. A near-white value is not clipped.
    pub shadows: [u64; 3],
    pub highlights: [u64; 3],
}

impl Default for RgbHistogram {
    fn default() -> Self {
        Self { channels: [[0; BINS]; 3], samples: 0, transparent: 0, invalid: 0, underflow: [0; 3], overflow: [0; 3], shadows: [0; 3], highlights: [0; 3] }
    }
}

impl RgbHistogram {
    /// Each visible pixel counts once, including partially transparent pixels. Non-finite
    /// pixels are rejected as a whole; transparent RGB cannot contribute phantom peaks.
    pub fn from_rgba(pixels: &[[f32; 4]]) -> Self {
        let mut h = Self::default();
        for pixel in pixels {
            if !pixel[3].is_finite() {
                h.invalid = h.invalid.saturating_add(1);
                continue;
            }
            if pixel[3] <= 0.0 {
                h.transparent = h.transparent.saturating_add(1);
                continue;
            }
            if !pixel.iter().all(|v| v.is_finite()) {
                h.invalid = h.invalid.saturating_add(1);
                continue;
            }
            h.samples = h.samples.saturating_add(1);
            let (low, high) = clipping(*pixel);
            for (c, ((((bins, below), above), shadows), highlights)) in
                h.channels.iter_mut().zip(&mut h.underflow).zip(&mut h.overflow).zip(&mut h.shadows).zip(&mut h.highlights).enumerate()
            {
                let Some(value) = pixel.get(c) else { continue };
                if low & (1 << c) != 0 {
                    *shadows = shadows.saturating_add(1);
                }
                if high & (1 << c) != 0 {
                    *highlights = highlights.saturating_add(1);
                }
                if *value < 0.0 {
                    *below = below.saturating_add(1);
                } else if *value > 1.0 {
                    *above = above.saturating_add(1);
                }
                // Match the preview encoder's nearest-bin rule, before it loses float precision.
                let bin = (value.clamp(0.0, 1.0) * (BINS - 1) as f32 + 0.5) as usize;
                if let Some(count) = bins.get_mut(bin) {
                    *count = count.saturating_add(1);
                }
            }
        }
        h
    }
}

/// Clipped channels of one visible straight RGBA pixel as bit masks (bit 0=R, 1=G, 2=B):
/// `(shadows, highlights)` at exact endpoints ≤0 / ≥1, not rounded edge bins.
/// Transparent and non-finite pixels clip nothing.
pub fn clipping(p: [f32; 4]) -> (u8, u8) {
    if p[3] <= 0.0 || !p.iter().all(|v| v.is_finite()) {
        return (0, 0);
    }
    let mut shadows = 0;
    let mut highlights = 0;
    for (c, v) in p.iter().take(3).enumerate() {
        if *v <= 0.0 {
            shadows |= 1 << c;
        }
        if *v >= 1.0 {
            highlights |= 1 << c;
        }
    }
    (shadows, highlights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_bins_and_channel_totals() {
        let pixels: Vec<_> = (0..BINS).map(|i| [i as f32 / 255.0, (255 - i) as f32 / 255.0, 0.5, 0.25]).collect();
        let h = RgbHistogram::from_rgba(&pixels);
        assert_eq!(h.samples, 256);
        assert!(h.channels[0].iter().all(|&v| v == 1));
        assert!(h.channels[1].iter().all(|&v| v == 1));
        assert_eq!(h.channels[2][128], 256);
        for channel in h.channels {
            assert_eq!(channel.iter().sum::<u64>(), h.samples);
        }
    }

    #[test]
    fn empty_transparent_invalid_and_hdr_are_explicit() {
        assert_eq!(RgbHistogram::from_rgba(&[]), RgbHistogram::default());
        let pixels = [[0.7, 0.9, 0.4, 0.0], [f32::NAN, 0.0, 0.0, 1.0], [0.0, f32::INFINITY, 0.0, 1.0], [0.0, 0.0, 0.0, f32::NAN], [-2.0, 4.0, 0.5, 1.0]];
        let original = pixels;
        let h = RgbHistogram::from_rgba(&pixels);
        assert_eq!((h.samples, h.transparent, h.invalid), (1, 1, 3));
        assert_eq!(h.underflow, [1, 0, 0]);
        assert_eq!(h.overflow, [0, 1, 0]);
        assert_eq!((h.channels[0][0], h.channels[1][255], h.channels[2][128]), (1, 1, 1));
        assert_eq!(pixels[4], original[4]);
        assert!(pixels[1][0].is_nan());
    }

    #[test]
    fn clipping_indicators_use_exact_samples_not_edge_bin_rounding() {
        let h = RgbHistogram::from_rgba(&[[0.0001, 0.9999, 0.5, 1.0], [-0.2, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]);
        assert_eq!(h.shadows, [1, 0, 1]);
        assert_eq!(h.highlights, [0, 1, 0]);
        assert_eq!(h.channels[0][0], 2);
        assert_eq!(h.channels[1][255], 2);
    }

    #[test]
    fn rgb_and_gray_at_all_depths_use_the_same_float_contract() {
        use photocraft_color::{ColorMode, PixelFormat, SampleType};
        use photocraft_geom::Rect;
        use photocraft_raster::Surface;
        for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
            for sample in SampleType::ALL {
                let mut surface = Surface::new(PixelFormat::new(mode, sample, true));
                for (x, v) in [0.0, 0.25, 0.75, 1.0].into_iter().enumerate() {
                    let px = if mode == ColorMode::Rgb { vec![v, v, v, 1.0] } else { vec![v, 1.0] };
                    surface.write_pixel(x as i32, 0, &px);
                }
                let mut pixels = [[0.0; 4]; 4];
                surface.read_rgba_into(Rect::new(0, 0, 4, 1), &mut pixels);
                let h = RgbHistogram::from_rgba(&pixels);
                assert_eq!(h.samples, 4, "{mode:?}/{sample:?}");
                for c in &h.channels {
                    for bin in [0, 64, 191, 255] {
                        assert_eq!(c[bin], 1, "{mode:?}/{sample:?}, bin {bin}");
                    }
                }
            }
        }
    }

    #[test]
    fn exact_clipping_endpoints_not_quantized_neighbours() {
        assert_eq!(clipping([0.0001, 0.9999, 0.5, 1.0]), (0, 0));
        assert_eq!(clipping([-0.1, 1.0, 0.0, 1.0]), (5, 2));
        assert_eq!(clipping([0.0, 1.0, 0.0, 0.0]), (0, 0));
        assert_eq!(clipping([0.0, f32::INFINITY, 0.0, 1.0]), (0, 0));
    }

    #[test]
    fn exposure_of_an_eight_bit_gradient_reproduces_the_quantization_comb() {
        let mut pixels: Vec<_> = (0..256).map(|i| [i as f32 / 255.0, i as f32 / 255.0, i as f32 / 255.0, 1.0]).collect();
        assert!(RgbHistogram::from_rgba(&pixels).channels[0].iter().all(|&v| v == 1));
        crate::camera_raw::develop(&mut pixels, 256, 1, &crate::camera_raw::CameraRaw { exposure: 0.5, ..Default::default() }, true);
        let h = RgbHistogram::from_rgba(&pixels);
        assert!(h.channels[0][10..230].iter().filter(|&&v| v == 0).count() > 10);
        assert_eq!(h.samples, 256);
        assert!(h.overflow[0] > 0);
    }
}
