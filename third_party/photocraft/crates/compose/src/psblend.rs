//! Photoshop-exact blend functions where they differ from the generic
//! reference in `photocraft_color::blend`, plus the compositing formula.
//!
//! Derived from Photoshop's rendered composites in the corpus oracle
//! (psd-tools `blend-modes/*.psd`):
//!
//! * **Vivid Light**: the source extremes win: `cs = 0` gives 0 even over a
//!   white backdrop, `cs = 1` gives 1 even over black (the generic Color
//!   Burn/Dodge let the backdrop extremes win instead).
//! * **Hard Mix**: the thresholded *generic* vivid light (backdrop extremes
//!   win): 1 where `VL(cb, cs) >= 0.5`, else 0. For interior values this is
//!   the familiar `cb + cs >= 1` rule; the asymmetric edge cases (black
//!   source over white → white, white over black → black) match Photoshop.

use photocraft_color::blend::{self as generic, BlendMode};

fn vivid_light_ps(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        if cs <= 0.0 { 0.0 } else { 1.0 - ((1.0 - cb) / (2.0 * cs)).min(1.0) }
    } else if cs >= 1.0 {
        1.0
    } else {
        (cb / (2.0 * (1.0 - cs))).min(1.0)
    }
}

/// Backdrop values within `EDGE` of 0 / 1 count as exact in the modes whose result jumps there
/// (Color Burn / Dodge, Hard Mix): a composited value that should be 1 can come out a rounding
/// step below it (and differently on the GPU), flipping the corner case by up to 255 levels.
pub const EDGE: f32 = 1e-4;

fn color_burn(cb: f32, cs: f32) -> f32 {
    if cb >= 1.0 - EDGE {
        1.0
    } else if cs <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - cb) / cs).min(1.0)
    }
}

fn color_dodge(cb: f32, cs: f32) -> f32 {
    if cb <= EDGE {
        0.0
    } else if cs >= 1.0 {
        1.0
    } else {
        (cb / (1.0 - cs)).min(1.0)
    }
}

fn vivid_light_generic(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        if cb >= 1.0 - EDGE {
            1.0
        } else if cs <= 0.0 {
            0.0
        } else {
            1.0 - ((1.0 - cb) / (2.0 * cs)).min(1.0)
        }
    } else if cb <= EDGE {
        0.0
    } else if cs >= 1.0 {
        1.0
    } else {
        (cb / (2.0 * (1.0 - cs))).min(1.0)
    }
}

fn hard_mix_ps(cb: f32, cs: f32) -> f32 {
    if vivid_light_generic(cb, cs) >= 0.5 - 1e-6 { 1.0 } else { 0.0 }
}

thread_local! {
    /// Set while rendering a Lab document: Normal blending mixes backdrop and source in CIELAB
    /// (Photoshop composites Lab documents in Lab; an anti-aliased edge between two colours
    /// differs by up to 14 / 255 from an sRGB mix: psd-tools stroke-color-descriptors-lab).
    pub static LAB_MIX: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Set while rendering a 32-bit float document: Linear Dodge (Add) and Divide don't clip at 1,
    /// and Hue, Saturation, Color and Luminosity keep values above 1 ([`generic::blend_rgb_hdr`]),
    /// as in Photoshop's 32-bit mode; integer depths clip.
    pub static HDR: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` with [`LAB_MIX`] set to `lab`, then restores the value it had (also on unwind). A
/// nested rayon job can run another tile on this thread mid-composite; clearing the flag there
/// made the rest of the outer tile mix in sRGB (#1112).
pub fn with_lab_mix<R>(lab: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            LAB_MIX.with(|l| l.set(self.0));
        }
    }
    let _restore = Restore(LAB_MIX.with(|l| l.replace(lab)));
    f()
}

/// Runs `f` with [`HDR`] set to `hdr`, then restores the value it had (also on unwind), like
/// [`with_lab_mix`].
pub fn with_hdr<R>(hdr: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            HDR.with(|h| h.set(self.0));
        }
    }
    let _restore = Restore(HDR.with(|h| h.replace(hdr)));
    f()
}

/// `B(Cb, Cs)` with Photoshop's variants.
pub fn blend_rgb(mode: BlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    match mode {
        BlendMode::VividLight => std::array::from_fn(|i| vivid_light_ps(cb[i], cs[i])),
        BlendMode::HardMix => std::array::from_fn(|i| hard_mix_ps(cb[i], cs[i])),
        BlendMode::ColorBurn => std::array::from_fn(|i| color_burn(cb[i], cs[i])),
        BlendMode::ColorDodge => std::array::from_fn(|i| color_dodge(cb[i], cs[i])),
        m if HDR.with(|h| h.get()) => generic::blend_rgb_hdr(m, cb, cs),
        m => generic::blend_rgb(m, cb, cs),
    }
}

/// Photoshop's "special eight": the modes for which a layer's Fill isn't just less coverage.
/// Below 100% Fill, Photoshop blends at full strength a source pulled toward the mode's neutral
/// colour, the one that leaves the backdrop unchanged (Hard Mix has its own formula); Opacity
/// still mixes the result in. Measured on Photoshop 27.11 (Fill 50%, eight modes, ±1 level).
pub fn fill_is_special(mode: BlendMode) -> bool {
    matches!(
        mode,
        BlendMode::ColorBurn
            | BlendMode::LinearBurn
            | BlendMode::ColorDodge
            | BlendMode::LinearDodge
            | BlendMode::VividLight
            | BlendMode::LinearLight
            | BlendMode::HardMix
            | BlendMode::Difference
    )
}

/// [`blend_rgb`] for a layer at `fill` (0..1): the special eight ([`fill_is_special`]) change
/// their blend below 100%; every other mode, or `fill >= 1`, is plain [`blend_rgb`].
pub fn blend_rgb_fill(mode: BlendMode, cb: [f32; 3], cs: [f32; 3], fill: f32) -> [f32; 3] {
    let f = fill.clamp(0.0, 1.0);
    if f >= 1.0 || !fill_is_special(mode) {
        return blend_rgb(mode, cb, cs);
    }
    let toward = |neutral: f32| -> [f32; 3] { std::array::from_fn(|i| neutral + (cs[i] - neutral) * f) };
    match mode {
        // Hard Mix at Fill f: (cb − f·(1 − cs)) / (1 − f), clamped. At f = 1 it is the hard
        // threshold (cb + cs ≥ 1), at f = 0 the backdrop.
        BlendMode::HardMix => std::array::from_fn(|i| ((cb[i] - f * (1.0 - cs[i])) / (1.0 - f)).clamp(0.0, 1.0)),
        BlendMode::ColorBurn | BlendMode::LinearBurn => blend_rgb(mode, cb, toward(1.0)),
        BlendMode::VividLight | BlendMode::LinearLight => blend_rgb(mode, cb, toward(0.5)),
        _ => blend_rgb(mode, cb, toward(0.0)),
    }
}

/// [`composite_gamma`] for a layer with separate `opacity` and `fill`. Normal modes take both
/// as coverage, as before. For the special eight below 100% Fill, coverage over the backdrop is
/// `opacity` and the blend is [`blend_rgb_fill`]; over transparency, Fill stays coverage, so a
/// layer over nothing looks as it did.
pub fn composite_fill(mode: BlendMode, backdrop: [f32; 4], source: [f32; 4], opacity: f32, fill: f32, gamma: f32) -> [f32; 4] {
    if fill >= 1.0 || !fill_is_special(mode) {
        return composite_gamma(mode, backdrop, source, opacity * fill.clamp(0.0, 1.0), gamma);
    }
    let ab = backdrop[3];
    let over = source[3] * opacity;
    let alone = over * fill.clamp(0.0, 1.0);
    if over <= 0.0 {
        return backdrop;
    }
    let ao = ab + (1.0 - ab) * alone;
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let cb = [backdrop[0], backdrop[1], backdrop[2]];
    let cs = [source[0], source[1], source[2]];
    let b = blend_rgb_fill(mode, cb, cs, fill);
    let (wb, wbs, ws) = (ab * (1.0 - over) / ao, ab * over / ao, (1.0 - ab) * alone / ao);
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        out[i] = if gamma == 1.0 {
            wb * cb[i] + wbs * b[i] + ws * cs[i]
        } else {
            text_decode(wb * text_encode(cb[i], gamma) + wbs * text_encode(b[i], gamma) + ws * text_encode(cs[i], gamma), gamma)
        };
    }
    out[3] = ao;
    out
}

/// Straight-alpha source over straight-alpha backdrop (W3C/PDF general
/// formula) using [`blend_rgb`].
pub fn composite(mode: BlendMode, backdrop: [f32; 4], source: [f32; 4], opacity: f32) -> [f32; 4] {
    let ab = backdrop[3];
    let as_ = source[3] * opacity;
    if as_ <= 0.0 {
        return backdrop;
    }
    let ao = as_ + ab * (1.0 - as_);
    if mode == BlendMode::Normal && LAB_MIX.with(|l| l.get()) && ab > 0.0 && ao > 0.0 {
        let lb = photocraft_color::convert::srgb_to_lab([backdrop[0], backdrop[1], backdrop[2]]);
        let ls = photocraft_color::convert::srgb_to_lab([source[0], source[1], source[2]]);
        let kb = ab * (1.0 - as_) / ao;
        let k = as_ / ao;
        let m: [f32; 3] = std::array::from_fn(|i| lb[i] * kb + ls[i] * k);
        let r = photocraft_color::convert::lab_to_srgb(m);
        return [r[0], r[1], r[2], ao];
    }
    if mode == BlendMode::Normal {
        return normal_rgb(backdrop, source, as_, ao);
    }
    let cb = [backdrop[0], backdrop[1], backdrop[2]];
    let cs = [source[0], source[1], source[2]];
    let b = blend_rgb(mode, cb, cs);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        out[i] = ((1.0 - as_) * ab * cb[i] + (1.0 - ab) * as_ * cs[i] + as_ * ab * b[i]) / ao;
    }
    out[3] = ao;
    out
}

/// Normal mixing outside Lab, for callers that select the blend space once per buffer.
#[inline]
pub(crate) fn composite_normal_rgb(backdrop: [f32; 4], source: [f32; 4], opacity: f32) -> [f32; 4] {
    let ab = backdrop[3];
    let as_ = source[3] * opacity;
    if as_ <= 0.0 {
        return backdrop;
    }
    let ao = as_ + ab * (1.0 - as_);
    normal_rgb(backdrop, source, as_, ao)
}

#[inline]
fn normal_rgb(backdrop: [f32; 4], source: [f32; 4], as_: f32, ao: f32) -> [f32; 4] {
    // B(Cb, Cs) = Cs. Keep the scalar composite's operation order and straight alpha.
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let kb = backdrop[3] * (1.0 - as_);
    let inv = 1.0 / ao;
    [(kb * backdrop[0] + as_ * source[0]) * inv, (kb * backdrop[1] + as_ * source[1]) * inv, (kb * backdrop[2] + as_ * source[2]) * inv, ao]
}

/// Photoshop's "Blend Text Colors Using Gamma" (Color Settings › Advanced, on at 1.45 by
/// default): type layers mix their colour with the backdrop in linear light raised to 1 / 1.45
/// ([`text_encode`]), so anti-aliased glyph edges come out lighter over dark backdrops than a
/// mix of the encoded values (psd-tools layer_effects.psd: a 69 % edge pixel of a blue overlay
/// over grey is 57/255 in red, an encoded mix gives 40).
pub const TEXT_GAMMA: f32 = 1.45;

static TEXT_GAMMA_SETTING: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0x3fb9_999a); // 1.45f32

/// Sets Color Settings › "Blend Text Colors Using Gamma" (1 = off). It is an application
/// setting, not stored in documents (1 here means switched off: a plain mix of the encoded
/// values). Values are clamped to Photoshop's 1.00–2.20.
///
/// The value is process-wide: a test that changes it races every concurrently running test
/// that composites a type layer, so such tests belong in their own test binary (see
/// `crates/engine/tests/text_gamma.rs`).
pub fn set_text_gamma(gamma: f32) {
    let g = if gamma.is_finite() { gamma.clamp(1.0, 2.2) } else { TEXT_GAMMA };
    TEXT_GAMMA_SETTING.store(g.to_bits(), std::sync::atomic::Ordering::Relaxed);
}

/// The current text blending gamma (see [`set_text_gamma`]; 1 = off).
pub fn text_gamma() -> f32 {
    f32::from_bits(TEXT_GAMMA_SETTING.load(std::sync::atomic::Ordering::Relaxed))
}

/// Into the text blending space: linear light raised to `1 / gamma`. Gamma 2.2 is close to
/// mixing the encoded values, lower settings approach linear light. Fitted on psd-tools
/// layer_effects.psd (a blue colour overlay's glyph edges over grey, every coverage level within
/// 0.6/255; a plain `v^1.45` power mix is 6/255 too light at 94 % coverage) and ag-psd
/// float-color.
#[inline]
pub fn text_encode(v: f32, gamma: f32) -> f32 {
    photocraft_color::convert::srgb_to_linear(v.max(0.0)).powf(1.0 / gamma)
}

/// Inverse of [`text_encode`].
#[inline]
pub fn text_decode(v: f32, gamma: f32) -> f32 {
    photocraft_color::convert::linear_to_srgb(v.max(0.0).powf(gamma))
}

/// [`composite`] with the coverage mix done in the text blending space of `gamma`
/// ([`text_encode`]; `gamma == 1` is [`composite`], the setting switched off).
pub fn composite_gamma(mode: BlendMode, backdrop: [f32; 4], source: [f32; 4], opacity: f32, gamma: f32) -> [f32; 4] {
    if gamma == 1.0 {
        return composite(mode, backdrop, source, opacity);
    }
    let ab = backdrop[3];
    let as_ = source[3] * opacity;
    if as_ <= 0.0 {
        return backdrop;
    }
    let ao = as_ + ab * (1.0 - as_);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let cb = [backdrop[0], backdrop[1], backdrop[2]];
    let cs = [source[0], source[1], source[2]];
    let b = blend_rgb(mode, cb, cs);
    let (wb, ws, wbs) = ((1.0 - as_) * ab / ao, (1.0 - ab) * as_ / ao, as_ * ab / ao);
    let g = |v: f32| text_encode(v, gamma);
    let ginv = |v: f32| text_decode(v, gamma);
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        out[i] = ginv(wb * g(cb[i]) + ws * g(cs[i]) + wbs * g(b[i]));
    }
    out[3] = ao;
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn text_gamma_lightens_dark_edges() {
        // 69 % blue (0, 51, 153) over grey 129: Photoshop shows (57, 80, 146).
        let r = composite_gamma(BlendMode::Normal, [129.0 / 255.0, 129.0 / 255.0, 129.0 / 255.0, 1.0], [0.0, 0.2, 0.6, 0.686], 1.0, TEXT_GAMMA);
        assert!((r[0] * 255.0 - 57.0).abs() < 1.0 && (r[1] * 255.0 - 80.0).abs() < 1.0 && (r[2] * 255.0 - 146.0).abs() < 1.0, "{r:?}");
        // Near-opaque edges stay dark: 94 % coverage shows (13, 57, 152) (a v^1.45 mix gives 19 red).
        let r = composite_gamma(BlendMode::Normal, [129.0 / 255.0, 129.0 / 255.0, 129.0 / 255.0, 1.0], [0.0, 0.2, 0.6, 0.937], 1.0, TEXT_GAMMA);
        assert!((r[0] * 255.0 - 13.0).abs() < 1.0 && (r[1] * 255.0 - 57.0).abs() < 1.0 && (r[2] * 255.0 - 152.0).abs() < 1.0, "{r:?}");
        assert_eq!(
            composite_gamma(BlendMode::Multiply, [0.2, 0.4, 0.6, 0.7], [0.5, 0.1, 0.9, 0.5], 0.8, 1.0),
            composite(BlendMode::Multiply, [0.2, 0.4, 0.6, 0.7], [0.5, 0.1, 0.9, 0.5], 0.8)
        );
    }

    use super::*;

    /// Photoshop 27.11, measured: Background #c89664, a layer #64b4dc at Fill 50% (128/255),
    /// Opacity 100%, in each of the special eight. Values are the composite at 8 bits.
    #[test]
    fn special_eight_fill_matches_photoshop() {
        let measured: [(BlendMode, [u8; 3]); 8] = [
            (BlendMode::ColorBurn, [176, 132, 88]),
            (BlendMode::LinearBurn, [122, 112, 82]),
            (BlendMode::ColorDodge, [249, 232, 176]),
            (BlendMode::LinearDodge, [250, 240, 210]),
            (BlendMode::VividLight, [193, 188, 157]),
            (BlendMode::LinearLight, [171, 202, 192]),
            (BlendMode::HardMix, [245, 225, 165]),
            (BlendMode::Difference, [150, 60, 10]),
        ];
        let px = |c: [u8; 3]| [f32::from(c[0]) / 255.0, f32::from(c[1]) / 255.0, f32::from(c[2]) / 255.0, 1.0];
        let (b, t) = (px([200, 150, 100]), px([100, 180, 220]));
        for (mode, want) in measured {
            let got = composite_fill(mode, b, t, 1.0, 128.0 / 255.0, 1.0);
            for i in 0..3 {
                assert!((got[i] * 255.0 - f32::from(want[i])).abs() <= 1.5, "{mode:?} channel {i}: {} vs Photoshop {}", got[i] * 255.0, want[i]);
            }
            // At 100% Fill it is the plain blend; at 0% the backdrop.
            assert_eq!(composite_fill(mode, b, t, 1.0, 1.0, 1.0), composite(mode, b, t, 1.0), "{mode:?}");
            let none = composite_fill(mode, b, t, 1.0, 0.0, 1.0);
            assert!((0..3).all(|i| (none[i] - b[i]).abs() < 1e-5), "{mode:?} at Fill 0: {none:?}");
        }
        // A second measurement: Background #3c78b4, layer #d25a1e at Fill 25% (64/255) and 75%
        // (191/255). Hard Mix at 75% is 3 levels off in red: dividing by 1 − f = 0.25 magnifies
        // Photoshop's 8-bit rounding.
        let confirm: [(BlendMode, [u8; 3], [u8; 3]); 8] = [
            (BlendMode::ColorBurn, [51, 94, 159], [30, 0, 33]),
            (BlendMode::LinearBurn, [49, 79, 124], [26, 0, 11]),
            (BlendMode::ColorDodge, [76, 132, 186], [156, 163, 197]),
            (BlendMode::LinearDodge, [113, 143, 188], [217, 187, 202]),
            (BlendMode::VividLight, [71, 109, 162], [117, 81, 80]),
            (BlendMode::LinearLight, [100, 100, 130], [183, 63, 33]),
            (BlendMode::HardMix, [65, 105, 166], [102, 0, 47]),
            (BlendMode::Difference, [7, 97, 172], [97, 53, 158]),
        ];
        let (b2, t2) = (px([60, 120, 180]), px([210, 90, 30]));
        for (mode, q, tq) in confirm {
            for (fill, want) in [(64.0 / 255.0, q), (191.0 / 255.0, tq)] {
                let got = composite_fill(mode, b2, t2, 1.0, fill, 1.0);
                let tol = if mode == BlendMode::HardMix { 3.5 } else { 1.5 };
                for i in 0..3 {
                    assert!(
                        (got[i] * 255.0 - f32::from(want[i])).abs() <= tol,
                        "{mode:?} fill {fill} channel {i}: {} vs Photoshop {}",
                        got[i] * 255.0,
                        want[i]
                    );
                }
            }
        }
        // Other modes keep Fill as coverage, like Opacity.
        assert_eq!(composite_fill(BlendMode::Multiply, b, t, 0.8, 0.5, 1.0), composite(BlendMode::Multiply, b, t, 0.4));
        // Over transparency Fill is still coverage for the special eight.
        let r = composite_fill(BlendMode::LinearDodge, [0.0; 4], t, 1.0, 0.5, 1.0);
        assert!((r[3] - 0.5).abs() < 1e-6 && (r[0] - t[0]).abs() < 1e-6, "{r:?}");
    }

    #[test]
    fn vivid_light_source_extremes_win() {
        assert_eq!(vivid_light_ps(0.0, 1.0), 1.0);
        assert_eq!(vivid_light_ps(1.0, 0.0), 0.0);
        assert!((vivid_light_ps(0.5, 0.75) - 1.0).abs() < 1e-6);
        assert!((vivid_light_ps(0.5, 0.25) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn hard_mix_edges_match_photoshop() {
        assert_eq!(hard_mix_ps(1.0, 0.0), 1.0);
        assert_eq!(hard_mix_ps(0.0, 1.0), 0.0);
        assert_eq!(hard_mix_ps(0.6, 0.6), 1.0);
        assert_eq!(hard_mix_ps(0.3, 0.3), 0.0);
    }

    #[test]
    fn burn_and_dodge_treat_near_extremes_as_exact() {
        assert_eq!(color_burn(1.0 - 1e-6, 0.0), 1.0);
        assert_eq!(color_dodge(1e-6, 1.0), 0.0);
        assert!((color_burn(0.5, 0.5) - 0.0).abs() < 1e-6);
        assert!((color_dodge(0.25, 0.5) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn hdr_nests_and_restores() {
        let bright = |m| blend_rgb(m, [0.5, 0.5, 0.5], [1.0, 0.0, 0.0])[0];
        with_hdr(true, || {
            // A 32-bit document keeps Color's red above 1.
            assert!(bright(BlendMode::Color) > 1.1);
            // Another tile's (or document's) scope run inside this one doesn't reset it.
            with_hdr(false, || assert!(bright(BlendMode::Color) <= 1.0));
            assert!(HDR.get());
        });
        assert!(!HDR.get());
        assert!(bright(BlendMode::Color) <= 1.0);
    }

    #[test]
    fn other_modes_delegate() {
        for m in BlendMode::LAYER_MODES {
            if matches!(m, BlendMode::VividLight | BlendMode::HardMix | BlendMode::ColorBurn | BlendMode::ColorDodge) {
                continue;
            }
            let (cb, cs) = ([0.2, 0.5, 0.9], [0.7, 0.1, 0.4]);
            assert_eq!(blend_rgb(m, cb, cs), generic::blend_rgb(m, cb, cs));
            let (a, b) =
                (composite(m, [0.2, 0.5, 0.9, 0.7], [0.7, 0.1, 0.4, 0.6], 0.8), generic::composite(m, [0.2, 0.5, 0.9, 0.7], [0.7, 0.1, 0.4, 0.6], 0.8));
            for c in 0..4 {
                assert!((a[c] - b[c]).abs() < 1e-6, "{m:?} {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn generic_vivid_light_and_hard_mix_composites_match_photoshop() {
        let cases = [
            (BlendMode::HardMix, 0.0, 1.0),
            (BlendMode::HardMix, 1.0, 0.0),
            (BlendMode::HardMix, 0.6, 0.4),
            (BlendMode::HardMix, 0.6, 0.3),
            (BlendMode::VividLight, 1.0, 0.0),
            (BlendMode::VividLight, 0.0, 1.0),
            (BlendMode::VividLight, 0.75, 0.25),
            (BlendMode::VividLight, 0.25, 0.75),
        ];

        for (mode, cb, cs) in cases {
            let backdrop = [cb, cb, cb, 1.0];
            let source = [cs, cs, cs, 1.0];
            let generic = generic::composite(mode, backdrop, source, 1.0);
            let photoshop = composite(mode, backdrop, source, 1.0);
            for channel in 0..4 {
                assert!((generic[channel] - photoshop[channel]).abs() < 1e-6, "{mode:?} Cb={cb} Cs={cs}: {generic:?} != {photoshop:?}");
            }
        }
    }
}
