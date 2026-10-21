//! Photoshop and Paint.NET blend modes (scalar reference implementation).
//!
//! Formulas follow the public PDF / ISO 32000-2 and W3C Compositing definitions,
//! with Photoshop's variants where they differ (Soft Light, Hard Mix, Darker/Lighter Color,
//! Linear Burn/Dodge, Vivid/Linear/Pin Light, Subtract, Divide). This is the CPU reference
//! that the GPU kernels must match.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BlendMode {
    /// Groups only: children blend directly into the group's backdrop.
    PassThrough,
    Normal,
    Dissolve,
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    DarkerColor,
    Lighten,
    Screen,
    ColorDodge,
    LinearDodge,
    LighterColor,
    Overlay,
    SoftLight,
    HardLight,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
    Difference,
    Exclusion,
    Subtract,
    Divide,
    Hue,
    Saturation,
    Color,
    Luminosity,
    Reflect,
    Glow,
    Negation,
    Xor,
    PaintNetColorBurn,
    PaintNetColorDodge,
}

impl BlendMode {
    /// Paint.NET modes without an exact Photoshop equivalent.
    pub const PAINT_NET_MODES: [BlendMode; 6] = [Self::Reflect, Self::Glow, Self::Negation, Self::Xor, Self::PaintNetColorBurn, Self::PaintNetColorDodge];

    /// All editable layer modes, with Paint.NET modes after Photoshop's menu order.
    pub fn layer_modes() -> impl Iterator<Item = Self> {
        Self::LAYER_MODES.into_iter().chain(Self::PAINT_NET_MODES)
    }

    pub fn has_psd_equivalent(self) -> bool {
        !Self::PAINT_NET_MODES.contains(&self)
    }
    /// Photoshop modes except PassThrough, in Photoshop menu order.
    pub const LAYER_MODES: [BlendMode; 27] = [
        BlendMode::Normal,
        BlendMode::Dissolve,
        BlendMode::Darken,
        BlendMode::Multiply,
        BlendMode::ColorBurn,
        BlendMode::LinearBurn,
        BlendMode::DarkerColor,
        BlendMode::Lighten,
        BlendMode::Screen,
        BlendMode::ColorDodge,
        BlendMode::LinearDodge,
        BlendMode::LighterColor,
        BlendMode::Overlay,
        BlendMode::SoftLight,
        BlendMode::HardLight,
        BlendMode::VividLight,
        BlendMode::LinearLight,
        BlendMode::PinLight,
        BlendMode::HardMix,
        BlendMode::Difference,
        BlendMode::Exclusion,
        BlendMode::Subtract,
        BlendMode::Divide,
        BlendMode::Hue,
        BlendMode::Saturation,
        BlendMode::Color,
        BlendMode::Luminosity,
    ];

    pub fn label(self) -> &'static str {
        match self {
            BlendMode::PassThrough => "Pass Through",
            BlendMode::Normal => "Normal",
            BlendMode::Dissolve => "Dissolve",
            BlendMode::Darken => "Darken",
            BlendMode::Multiply => "Multiply",
            BlendMode::ColorBurn => "Color Burn",
            BlendMode::LinearBurn => "Linear Burn",
            BlendMode::DarkerColor => "Darker Color",
            BlendMode::Lighten => "Lighten",
            BlendMode::Screen => "Screen",
            BlendMode::ColorDodge => "Color Dodge",
            BlendMode::LinearDodge => "Linear Dodge (Add)",
            BlendMode::LighterColor => "Lighter Color",
            BlendMode::Overlay => "Overlay",
            BlendMode::SoftLight => "Soft Light",
            BlendMode::HardLight => "Hard Light",
            BlendMode::VividLight => "Vivid Light",
            BlendMode::LinearLight => "Linear Light",
            BlendMode::PinLight => "Pin Light",
            BlendMode::HardMix => "Hard Mix",
            BlendMode::Difference => "Difference",
            BlendMode::Exclusion => "Exclusion",
            BlendMode::Subtract => "Subtract",
            BlendMode::Divide => "Divide",
            BlendMode::Hue => "Hue",
            BlendMode::Saturation => "Saturation",
            BlendMode::Color => "Color",
            BlendMode::Luminosity => "Luminosity",
            BlendMode::Reflect => "Reflect",
            BlendMode::Glow => "Glow",
            BlendMode::Negation => "Negation",
            BlendMode::Xor => "XOR",
            BlendMode::PaintNetColorBurn => "Color Burn (Paint.NET)",
            BlendMode::PaintNetColorDodge => "Color Dodge (Paint.NET)",
        }
    }

    /// PSD 4-character blend key. Modes without an equivalent fall back to Normal;
    /// exporters must report that loss using [`Self::has_psd_equivalent`].
    pub fn psd_key(self) -> [u8; 4] {
        *match self {
            BlendMode::PassThrough => b"pass",
            BlendMode::Normal => b"norm",
            BlendMode::Dissolve => b"diss",
            BlendMode::Darken => b"dark",
            BlendMode::Multiply => b"mul ",
            BlendMode::ColorBurn => b"idiv",
            BlendMode::LinearBurn => b"lbrn",
            BlendMode::DarkerColor => b"dkCl",
            BlendMode::Lighten => b"lite",
            BlendMode::Screen => b"scrn",
            BlendMode::ColorDodge => b"div ",
            BlendMode::LinearDodge => b"lddg",
            BlendMode::LighterColor => b"lgCl",
            BlendMode::Overlay => b"over",
            BlendMode::SoftLight => b"sLit",
            BlendMode::HardLight => b"hLit",
            BlendMode::VividLight => b"vLit",
            BlendMode::LinearLight => b"lLit",
            BlendMode::PinLight => b"pLit",
            BlendMode::HardMix => b"hMix",
            BlendMode::Difference => b"diff",
            BlendMode::Exclusion => b"smud",
            BlendMode::Subtract => b"fsub",
            BlendMode::Divide => b"fdiv",
            BlendMode::Hue => b"hue ",
            BlendMode::Saturation => b"sat ",
            BlendMode::Color => b"colr",
            BlendMode::Luminosity => b"lum ",
            BlendMode::Reflect | BlendMode::Glow | BlendMode::Negation | BlendMode::Xor | BlendMode::PaintNetColorBurn | BlendMode::PaintNetColorDodge => {
                b"norm"
            }
        }
    }

    pub fn from_psd_key(key: [u8; 4]) -> Option<BlendMode> {
        std::iter::once(BlendMode::PassThrough).chain(Self::LAYER_MODES).find(|m| m.psd_key() == key)
    }

    pub fn is_separable(self) -> bool {
        !matches!(self, BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity | BlendMode::DarkerColor | BlendMode::LighterColor)
    }
}

/// Separable per-channel blend function `B(cb, cs)`.
#[inline]
pub fn blend_channel(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    channel(mode, cb, cs, 1.0)
}

/// [`blend_channel`] for 32-bit float documents, whose values may exceed 1: Linear Dodge (Add)
/// and Divide don't clip at 1 there (Photoshop's 32-bit Add is how light is added in HDR), only
/// at `f32::MAX`, so they stay finite. Every other mode is [`blend_channel`].
#[inline]
pub fn blend_channel_hdr(mode: BlendMode, cb: f32, cs: f32) -> f32 {
    channel(mode, cb, cs, f32::MAX)
}

/// `B(cb, cs)` with Linear Dodge and Divide clipped at `hi` (1 at integer depths).
#[inline]
fn channel(mode: BlendMode, cb: f32, cs: f32, hi: f32) -> f32 {
    match mode {
        BlendMode::Normal | BlendMode::Dissolve | BlendMode::PassThrough => cs,
        BlendMode::Darken => cb.min(cs),
        BlendMode::Multiply => cb * cs,
        BlendMode::ColorBurn => color_burn(cb, cs),
        BlendMode::LinearBurn => (cb + cs - 1.0).max(0.0),
        BlendMode::Lighten => cb.max(cs),
        BlendMode::Screen => cb + cs - cb * cs,
        BlendMode::ColorDodge => color_dodge(cb, cs),
        BlendMode::LinearDodge => (cb + cs).min(hi),
        BlendMode::Overlay => hard_light(cs, cb),
        BlendMode::SoftLight => soft_light_ps(cb, cs),
        BlendMode::HardLight => hard_light(cb, cs),
        BlendMode::VividLight if cs <= 0.0 => 0.0,
        BlendMode::VividLight if cs >= 1.0 => 1.0,
        BlendMode::VividLight => {
            if cs <= 0.5 {
                color_burn(cb, 2.0 * cs)
            } else {
                color_dodge(cb, 2.0 * (cs - 0.5))
            }
        }
        BlendMode::LinearLight => (cb + 2.0 * cs - 1.0).clamp(0.0, 1.0),
        BlendMode::PinLight => {
            if cs <= 0.5 {
                cb.min(2.0 * cs)
            } else {
                cb.max(2.0 * cs - 1.0)
            }
        }
        BlendMode::HardMix => {
            if vivid_light_hard_mix(cb, cs) >= 0.5 - 1e-6 {
                1.0
            } else {
                0.0
            }
        }
        BlendMode::Difference => (cb - cs).abs(),
        BlendMode::Exclusion => cb + cs - 2.0 * cb * cs,
        BlendMode::Subtract => (cb - cs).max(0.0),
        BlendMode::Divide => {
            if cs <= 0.0 {
                if cb <= 0.0 { 0.0 } else { 1.0 }
            } else {
                (cb / cs).min(hi)
            }
        }
        BlendMode::Reflect => reflect(cb, cs),
        BlendMode::Glow => reflect(cs, cb),
        BlendMode::Negation => 1.0 - (1.0 - cb - cs).abs(),
        BlendMode::Xor => {
            // Paint.NET's XOR is defined on 8-bit channels even in a floating-point compositor.
            let b = (cb.clamp(0.0, 1.0) * 255.0).round() as u8;
            let s = (cs.clamp(0.0, 1.0) * 255.0).round() as u8;
            f32::from(b ^ s) / 255.0
        }
        BlendMode::PaintNetColorBurn => {
            if cs <= 0.0 {
                0.0
            } else {
                (1.0 - (1.0 - cb) / cs).max(0.0)
            }
        }
        BlendMode::PaintNetColorDodge => {
            if cs >= 1.0 {
                1.0
            } else {
                (cb / (1.0 - cs)).min(1.0)
            }
        }
        // Non-separable modes are handled by `blend_rgb`; fall back to source.
        BlendMode::DarkerColor | BlendMode::LighterColor | BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => cs,
    }
}

fn reflect(cb: f32, cs: f32) -> f32 {
    if cs >= 1.0 { 1.0 } else { (cb * cb / (1.0 - cs)).min(1.0) }
}

#[inline]
fn vivid_light_hard_mix(cb: f32, cs: f32) -> f32 {
    const EDGE: f32 = 1e-4;
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

#[inline]
fn color_burn(cb: f32, cs: f32) -> f32 {
    if cb >= 1.0 {
        1.0
    } else if cs <= 0.0 {
        0.0
    } else {
        1.0 - ((1.0 - cb) / cs).min(1.0)
    }
}

#[inline]
fn color_dodge(cb: f32, cs: f32) -> f32 {
    if cb <= 0.0 {
        0.0
    } else if cs >= 1.0 {
        1.0
    } else {
        (cb / (1.0 - cs)).min(1.0)
    }
}

#[inline]
fn hard_light(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 {
        cb * 2.0 * cs
    } else {
        let s = 2.0 * cs - 1.0;
        cb + s - cb * s
    }
}

/// Photoshop's Soft Light (differs from the W3C/PDF definition in the upper half).
#[inline]
fn soft_light_ps(cb: f32, cs: f32) -> f32 {
    if cs <= 0.5 { 2.0 * cb * cs + cb * cb * (1.0 - 2.0 * cs) } else { 2.0 * cb * (1.0 - cs) + cb.max(0.0).sqrt() * (2.0 * cs - 1.0) }
}

// ---- non-separable helpers (W3C Compositing Level 1) ----

#[inline]
pub fn lum(c: [f32; 3]) -> f32 {
    0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2]
}

/// ClipColor: moves `c` towards its grey ([`lum`]) until every channel is within `0..=top`,
/// keeping the luminosity. `top` is the brightest in-gamut value: 1, or `f32::MAX` in 32-bit
/// documents (values above 1 are valid light there, so only negatives are pulled in).
///
/// The scale factors `l / (l − min)` and `(top − l) / (max − l)` lie in 0..1 only while the
/// luminosity itself is in range. Outside it no colour of that luminosity is in gamut: the plain
/// formula turned colours inside out (a grey of 2 as the Luminosity of (0.5, 0.25, 0.125) gave
/// (1.0, 2.32, 2.99)) and, for greys, divided by the rounding noise of `max − l`, so results were
/// arbitrary and differed between the CPU and the GPU. Such colours become their grey instead
/// (the formula's value as the luminosity reaches the bound).
fn clip_color(c: [f32; 3], top: f32) -> [f32; 3] {
    let l = lum(c);
    let n = c[0].min(c[1]).min(c[2]);
    let x = c[0].max(c[1]).max(c[2]);
    let mut out = c;
    if n < 0.0 {
        let d = l - n;
        for v in &mut out {
            *v = if l > 0.0 && d > 0.0 { l + (*v - l) * l / d } else { l };
        }
    }
    if x > top {
        let d = x - l;
        for v in &mut out {
            *v = if l < top && d > 0.0 { l + (*v - l) * (top - l) / d } else { l };
        }
    }
    out
}

/// `c` with its luminosity ([`lum`]) moved to `l`, clipped back into gamut towards grey (the
/// Luminosity blend mode's SetLum + ClipColor).
pub fn set_lum(c: [f32; 3], l: f32) -> [f32; 3] {
    set_lum_within(c, l, 1.0)
}

/// [`set_lum`] clipping at `top` instead of 1 (see [`blend_rgb_within`]).
fn set_lum_within(c: [f32; 3], l: f32, top: f32) -> [f32; 3] {
    let d = l - lum(c);
    clip_color([c[0] + d, c[1] + d, c[2] + d], top)
}

fn sat(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2])
}

fn set_sat(c: [f32; 3], s: f32) -> [f32; 3] {
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&a, &b| c[a].partial_cmp(&c[b]).unwrap_or(std::cmp::Ordering::Equal));
    let (imin, imid, imax) = (idx[0], idx[1], idx[2]);
    let mut out = [0.0f32; 3];
    if c[imax] > c[imin] {
        out[imid] = (c[imid] - c[imin]) * s / (c[imax] - c[imin]);
        out[imax] = s;
    }
    out[imin] = 0.0;
    out
}

/// Blend a whole RGB triple: `B(Cb, Cs)` including non-separable modes.
pub fn blend_rgb(mode: BlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    blend_rgb_within(mode, cb, cs, 1.0)
}

/// [`blend_rgb`] with Hue / Saturation / Color / Luminosity clipped at `top` instead of 1:
/// `f32::MAX` keeps the values above 1 of 32-bit documents (their gamut has no upper bound).
pub fn blend_rgb_within(mode: BlendMode, cb: [f32; 3], cs: [f32; 3], top: f32) -> [f32; 3] {
    match mode {
        BlendMode::Hue => set_lum_within(set_sat(cs, sat(cb)), lum(cb), top),
        BlendMode::Saturation => set_lum_within(set_sat(cb, sat(cs)), lum(cb), top),
        BlendMode::Color => set_lum_within(cs, lum(cb), top),
        BlendMode::Luminosity => set_lum_within(cb, lum(cs), top),
        BlendMode::DarkerColor => {
            if lum(cs) < lum(cb) {
                cs
            } else {
                cb
            }
        }
        BlendMode::LighterColor => {
            if lum(cs) > lum(cb) {
                cs
            } else {
                cb
            }
        }
        m => [blend_channel(m, cb[0], cs[0]), blend_channel(m, cb[1], cs[1]), blend_channel(m, cb[2], cs[2])],
    }
}

/// [`blend_rgb`] for 32-bit float documents: separable modes use [`blend_channel_hdr`], and the
/// non-separable ones keep values above 1 ([`blend_rgb_within`] at `f32::MAX`).
pub fn blend_rgb_hdr(mode: BlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
    if mode.is_separable() { std::array::from_fn(|i| blend_channel_hdr(mode, cb[i], cs[i])) } else { blend_rgb_within(mode, cb, cs, f32::MAX) }
}

/// Composite a straight-alpha source over a straight-alpha backdrop with blend mode and opacity.
///
/// Uses the W3C/PDF general formula:
/// `αo = αs + αb(1−αs)`,
/// `αo·Co = (1−αs)·αb·Cb + (1−αb)·αs·Cs + αs·αb·B(Cb,Cs)`.
/// Dissolve must be resolved by the caller (per-pixel coverage decision) before calling.
pub fn composite(mode: BlendMode, backdrop: [f32; 4], source: [f32; 4], opacity: f32) -> [f32; 4] {
    let ab = backdrop[3];
    let as_ = source[3] * opacity;
    if as_ <= 0.0 {
        return backdrop;
    }
    let cb = [backdrop[0], backdrop[1], backdrop[2]];
    let cs = [source[0], source[1], source[2]];
    let b = blend_rgb(mode, cb, cs);
    let ao = as_ + ab * (1.0 - as_);
    if ao <= 0.0 {
        return [0.0; 4];
    }
    let mut out = [0.0f32; 4];
    for i in 0..3 {
        let premul = (1.0 - as_) * ab * cb[i] + (1.0 - ab) * as_ * cs[i] + as_ * ab * b[i];
        out[i] = premul / ao;
    }
    out[3] = ao;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const E: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < E
    }

    #[test]
    fn psd_keys_roundtrip_and_unique() {
        let mut seen = std::collections::HashSet::new();
        for m in std::iter::once(BlendMode::PassThrough).chain(BlendMode::LAYER_MODES) {
            assert!(seen.insert(m.psd_key()), "duplicate key {m:?}");
            assert_eq!(BlendMode::from_psd_key(m.psd_key()), Some(m));
        }
        assert_eq!(BlendMode::from_psd_key(*b"zzzz"), None);
        assert_eq!(seen.len(), 28);
    }

    #[test]
    fn paint_net_modes_keep_their_math_and_distinct_psd_status() {
        assert!(close(blend_channel(BlendMode::Reflect, 0.5, 0.5), 0.5));
        assert!(close(blend_channel(BlendMode::Reflect, 0.2, 0.0), 0.04));
        assert!(close(blend_channel(BlendMode::Glow, 0.5, 0.2), 0.08));
        assert!(close(blend_channel(BlendMode::Negation, 0.2, 0.7), 0.9));
        assert_eq!(blend_channel(BlendMode::Reflect, 0.0, 1.0), 1.0);
        assert_eq!(blend_channel(BlendMode::Glow, 1.0, 0.0), 1.0);
        assert_eq!(blend_channel(BlendMode::PaintNetColorBurn, 1.0, 0.0), 0.0);
        assert_eq!(blend_channel(BlendMode::PaintNetColorDodge, 0.0, 1.0), 1.0);
        for b in 0..=255u8 {
            for s in 0..=255u8 {
                assert!(close(blend_channel(BlendMode::Xor, f32::from(b) / 255.0, f32::from(s) / 255.0), f32::from(b ^ s) / 255.0));
            }
        }
        for mode in BlendMode::PAINT_NET_MODES {
            assert!(!mode.has_psd_equivalent());
            assert_eq!(mode.psd_key(), *b"norm");
            for b in [0.0, 0.1, 0.5, 1.0] {
                for s in [0.0, 0.1, 0.5, 1.0] {
                    let v = blend_channel(mode, b, s);
                    assert!(v.is_finite() && (0.0..=1.0).contains(&v), "{mode:?}");
                }
            }
        }
        assert_eq!(BlendMode::layer_modes().count(), 33);
    }

    #[test]
    fn normal_is_source() {
        for &(cb, cs) in &[(0.2, 0.7), (1.0, 0.0), (0.0, 1.0)] {
            assert!(close(blend_channel(BlendMode::Normal, cb, cs), cs));
        }
    }

    #[test]
    fn identities_with_white_and_black() {
        // Multiply by white / screen with black / add black / darken with white leave backdrop unchanged.
        for cb in [0.0f32, 0.1, 0.5, 0.9, 1.0] {
            assert!(close(blend_channel(BlendMode::Multiply, cb, 1.0), cb));
            assert!(close(blend_channel(BlendMode::Screen, cb, 0.0), cb));
            assert!(close(blend_channel(BlendMode::LinearDodge, cb, 0.0), cb));
            assert!(close(blend_channel(BlendMode::Darken, cb, 1.0), cb));
            assert!(close(blend_channel(BlendMode::Lighten, cb, 0.0), cb));
            assert!(close(blend_channel(BlendMode::Difference, cb, 0.0), cb));
            assert!(close(blend_channel(BlendMode::Subtract, cb, 0.0), cb));
            assert!(close(blend_channel(BlendMode::LinearBurn, cb, 1.0), cb));
            assert!(close(blend_channel(BlendMode::Exclusion, cb, 0.0), cb));
            // 50% gray is neutral for the "light" family
            assert!(close(blend_channel(BlendMode::HardLight, cb, 0.5), cb));
            assert!(close(blend_channel(BlendMode::SoftLight, cb, 0.5), cb));
            assert!(close(blend_channel(BlendMode::LinearLight, cb, 0.5), cb));
        }
    }

    #[test]
    fn known_values() {
        assert!(close(blend_channel(BlendMode::Multiply, 0.5, 0.5), 0.25));
        assert!(close(blend_channel(BlendMode::Screen, 0.5, 0.5), 0.75));
        assert!(close(blend_channel(BlendMode::Overlay, 0.25, 0.5), 0.25));
        assert!(close(blend_channel(BlendMode::Overlay, 0.75, 1.0), 1.0));
        assert!(close(blend_channel(BlendMode::ColorDodge, 0.5, 0.5), 1.0));
        assert!(close(blend_channel(BlendMode::ColorBurn, 0.5, 0.5), 0.0));
        assert!(close(blend_channel(BlendMode::ColorBurn, 1.0, 0.0), 1.0));
        assert!(close(blend_channel(BlendMode::ColorDodge, 0.0, 1.0), 0.0));
        assert!(close(blend_channel(BlendMode::Divide, 0.25, 0.5), 0.5));
        assert!(close(blend_channel(BlendMode::Divide, 0.0, 0.0), 0.0));
        assert!(close(blend_channel(BlendMode::Divide, 0.3, 0.0), 1.0));
        assert!(close(blend_channel(BlendMode::Exclusion, 0.5, 0.5), 0.5));
        assert!(close(blend_channel(BlendMode::PinLight, 0.8, 0.2), 0.4));
        assert!(close(blend_channel(BlendMode::PinLight, 0.1, 0.9), 0.8));
        assert!(close(blend_channel(BlendMode::HardMix, 0.6, 0.4), 1.0));
        assert!(close(blend_channel(BlendMode::HardMix, 0.6, 0.3), 0.0));
    }

    #[test]
    fn vivid_light_source_extremes_and_interior_values() {
        for cb in [0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(blend_channel(BlendMode::VividLight, cb, 0.0), 0.0);
            assert_eq!(blend_channel(BlendMode::VividLight, cb, 1.0), 1.0);
        }
        assert!(close(blend_channel(BlendMode::VividLight, 0.75, 0.25), 0.5));
        assert!(close(blend_channel(BlendMode::VividLight, 0.25, 0.75), 0.5));
    }

    #[test]
    fn hard_mix_matches_photoshop_extremes_and_interior_values() {
        assert_eq!(blend_channel(BlendMode::HardMix, 1.0, 0.0), 1.0);
        assert_eq!(blend_channel(BlendMode::HardMix, 0.0, 1.0), 0.0);
        assert_eq!(blend_channel(BlendMode::HardMix, 1.0 - 1e-5, 0.0), 1.0);
        assert_eq!(blend_channel(BlendMode::HardMix, 1e-5, 1.0), 0.0);
        assert_eq!(blend_channel(BlendMode::HardMix, 0.6, 0.4), 1.0);
        assert_eq!(blend_channel(BlendMode::HardMix, 0.6, 0.3), 0.0);
    }

    #[test]
    fn vivid_light_and_hard_mix_composite_extremes() {
        assert_eq!(composite(BlendMode::VividLight, [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], 1.0), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(composite(BlendMode::VividLight, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], 1.0), [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(composite(BlendMode::HardMix, [0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], 1.0), [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(composite(BlendMode::HardMix, [1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0], 1.0), [1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn separable_outputs_stay_in_unit_range() {
        let steps: Vec<f32> = (0..=20).map(|i| i as f32 / 20.0).collect();
        for m in BlendMode::LAYER_MODES {
            for &cb in &steps {
                for &cs in &steps {
                    let v = blend_rgb(m, [cb, cs, 0.5], [cs, cb, 0.25]);
                    for c in v {
                        assert!((-E..=1.0 + E).contains(&c), "{m:?} cb={cb} cs={cs} -> {c}");
                    }
                }
            }
        }
    }

    #[test]
    fn add_and_divide_clip_at_integer_depths_but_not_in_float() {
        // Integer depths: Photoshop clips Linear Dodge (Add) and Divide at 1.
        assert_eq!(blend_channel(BlendMode::LinearDodge, 2.0, 0.5), 1.0);
        assert_eq!(blend_channel(BlendMode::LinearDodge, 0.75, 0.5), 1.0);
        assert_eq!(blend_channel(BlendMode::Divide, 2.0, 0.5), 1.0);
        assert_eq!(blend_channel(BlendMode::Divide, 0.5, 0.25), 1.0);
        // 32-bit float: light adds up past 1 (adding light never darkens).
        assert!(close(blend_channel_hdr(BlendMode::LinearDodge, 2.0, 0.5), 2.5));
        assert!(close(blend_channel_hdr(BlendMode::LinearDodge, 0.75, 0.5), 1.25));
        assert!(close(blend_channel_hdr(BlendMode::Divide, 2.0, 0.5), 4.0));
        assert!(close(blend_channel_hdr(BlendMode::Divide, 0.5, 0.25), 2.0));
        assert_eq!(blend_rgb_hdr(BlendMode::LinearDodge, [2.0, 0.5, 4.0], [0.5, 0.25, 0.5]), [2.5, 0.75, 4.5]);
        assert_eq!(blend_rgb_hdr(BlendMode::Divide, [2.0, 0.5, 4.0], [0.5, 0.25, 0.5]), [4.0, 2.0, 8.0]);
        // Divide by zero keeps its guard; overflow stays finite.
        assert_eq!(blend_channel_hdr(BlendMode::Divide, 0.0, 0.0), 0.0);
        assert_eq!(blend_channel_hdr(BlendMode::Divide, 3.0, 0.0), 1.0);
        assert_eq!(blend_channel_hdr(BlendMode::Divide, 3.0, -1.0), 1.0);
        assert_eq!(blend_channel_hdr(BlendMode::Divide, f32::MAX, 1e-30), f32::MAX);
        assert_eq!(blend_channel_hdr(BlendMode::LinearDodge, f32::MAX, f32::MAX), f32::MAX);
        assert!(blend_channel_hdr(BlendMode::LinearDodge, f32::INFINITY, 1.0).is_finite());
        // Every other separable mode is unchanged (the non-separable ones keep values above 1:
        // blend_rgb_within), and the in-range results agree.
        for m in BlendMode::LAYER_MODES.into_iter().filter(|m| m.is_separable() && !matches!(m, BlendMode::LinearDodge | BlendMode::Divide)) {
            for (cb, cs) in [(0.2, 0.3), (0.9, 0.05), (2.0, 0.5), (0.4, 1.5)] {
                assert_eq!(blend_rgb(m, [cb, cs, 0.5], [cs, cb, 0.25]), blend_rgb_hdr(m, [cb, cs, 0.5], [cs, cb, 0.25]), "{m:?} cb={cb} cs={cs}");
            }
        }
        let (cb, cs) = ([0.2, 0.3, 0.1], [0.3, 0.4, 0.4]);
        assert_eq!(blend_rgb_hdr(BlendMode::LinearDodge, cb, cs), blend_rgb(BlendMode::LinearDodge, cb, cs));
        assert_eq!(blend_rgb_hdr(BlendMode::Divide, cb, cs), blend_rgb(BlendMode::Divide, cb, cs));
    }

    #[test]
    fn luminosity_and_color_swap_components() {
        let cb = [0.8, 0.2, 0.2];
        let cs = [0.1, 0.1, 0.9];
        let l = blend_rgb(BlendMode::Luminosity, cb, cs);
        assert!(close(lum(l), lum(cs)));
        let c = blend_rgb(BlendMode::Color, cb, cs);
        assert!(close(lum(c), lum(cb)));
        let h = blend_rgb(BlendMode::Hue, cb, cs);
        assert!(close(lum(h), lum(cb)));
        let s = blend_rgb(BlendMode::Saturation, cb, [0.5, 0.5, 0.5]);
        // zero-saturation source desaturates the backdrop
        assert!(close(s[0], s[1]) && close(s[1], s[2]));
    }

    /// ClipColor exactly as ISO 32000 / W3C write it (what `clip_color` was before it guarded
    /// out-of-range luminosities).
    fn spec_clip_color(c: [f32; 3]) -> [f32; 3] {
        let l = lum(c);
        let n = c[0].min(c[1]).min(c[2]);
        let x = c[0].max(c[1]).max(c[2]);
        let mut out = c;
        if n < 0.0 {
            let d = l - n;
            for v in &mut out {
                *v = if d.abs() < 1e-12 { l } else { l + (*v - l) * l / d };
            }
        }
        if x > 1.0 {
            let d = x - l;
            for v in &mut out {
                *v = if d.abs() < 1e-12 { l } else { l + (*v - l) * (1.0 - l) / d };
            }
        }
        out
    }

    fn spec_blend_rgb(mode: BlendMode, cb: [f32; 3], cs: [f32; 3]) -> [f32; 3] {
        let set_lum = |c: [f32; 3], l: f32| {
            let d = l - lum(c);
            spec_clip_color([c[0] + d, c[1] + d, c[2] + d])
        };
        match mode {
            BlendMode::Hue => set_lum(set_sat(cs, sat(cb)), lum(cb)),
            BlendMode::Saturation => set_lum(set_sat(cb, sat(cs)), lum(cb)),
            BlendMode::Color => set_lum(cs, lum(cb)),
            BlendMode::Luminosity => set_lum(cb, lum(cs)),
            m => blend_rgb(m, cb, cs),
        }
    }

    const NON_SEPARABLE: [BlendMode; 4] = [BlendMode::Hue, BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity];

    #[test]
    fn non_separable_modes_unchanged_on_integer_inputs() {
        // 8/16-bit documents only hold 0..1, where the luminosity stays in range (up to a
        // rounding step at white) and ClipColor is the plain ISO 32000 formula.
        let mut edges: Vec<f32> = vec![0.0, 1.0 / 255.0, 0.5, 254.0 / 255.0, 65534.0 / 65535.0, 1.0];
        edges.extend((0..=8).map(|i| i as f32 / 8.0));
        let mut colors = Vec::new();
        for &r in &edges {
            for &g in &edges {
                for &b in &edges {
                    colors.push([r, g, b]);
                }
            }
        }
        let mut seed = 0x1234_5678u32;
        let mut next = |levels: u32| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 8) % (levels + 1)) as f32 / levels as f32
        };
        for _ in 0..2000 {
            let levels = if next(1) > 0.5 { 255 } else { 65535 };
            colors.push([next(levels), next(levels), next(levels)]);
        }
        let mut worst = 0.0f32;
        for (i, &cb) in colors.iter().enumerate() {
            for &cs in colors.iter().skip(i % 7).step_by(7) {
                for m in NON_SEPARABLE {
                    let (got, want) = (blend_rgb(m, cb, cs), spec_blend_rgb(m, cb, cs));
                    for k in 0..3 {
                        worst = worst.max((got[k] - want[k]).abs());
                    }
                }
            }
        }
        // Only white-ish colours whose luminosity rounds a step above 1 can differ, by a rounding step.
        assert!(worst <= 1e-6, "{worst}");
    }

    #[test]
    fn non_separable_modes_stable_when_luminosity_is_out_of_range() {
        // The spec formula divided by `max − lum`, which is rounding noise for a grey above 1 and
        // turns colours inside out when the luminosity is above 1: Luminosity of a grey of 2 over
        // (0.5, 0.25, 0.125) gave (1.0, 2.32, 2.99).
        let spec = spec_blend_rgb(BlendMode::Luminosity, [0.5, 0.25, 0.125], [2.0; 3]);
        assert!(spec[0] < spec[2], "{spec:?}");
        let colors: [[f32; 3]; 10] = [
            [1.0, 1.0, 1.0],
            [1.5, 1.5, 1.5],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
            [1000.0, 1000.0, 1000.0],
            [2.0, 2.0, 1.999_999],
            [4.0, 3.999_9, 4.0],
            [3.0, 0.25, 0.125],
            [1.25, 0.875, 3.75],
            [-0.25, 0.5, 1.375],
        ];
        for top in [1.0, f32::MAX] {
            for m in NON_SEPARABLE {
                for cb in colors {
                    for cs in colors {
                        let r = blend_rgb_within(m, cb, cs, top);
                        assert!(r.iter().all(|v| v.is_finite()), "{m:?} {cb:?} {cs:?} {top}: {r:?}");
                        // The target luminosity is kept, at any depth.
                        let want = if m == BlendMode::Luminosity { lum(cs) } else { lum(cb) };
                        if want >= 0.0 {
                            assert!((lum(r) - want).abs() <= 1e-4 * want.max(1.0), "{m:?} {cb:?} {cs:?} {top}: {r:?}");
                        }
                        let (lo, hi) = (r[0].min(r[1]).min(r[2]), r[0].max(r[1]).max(r[2]));
                        if top == 1.0 {
                            // In gamut, or (a luminosity above 1 has no in-gamut colour) its grey.
                            assert!((lo >= 0.0 && hi <= 1.0 + E) || hi - lo <= 1e-6 * hi, "{m:?} {cb:?} {cs:?}: {r:?}");
                        } else {
                            assert!(lo >= -1e-6 * hi.max(1.0), "{m:?} {cb:?} {cs:?}: {r:?}");
                        }
                    }
                }
            }
        }
        // 32-bit: a grey backdrop gives a grey of its own luminosity in Hue and Saturation, a
        // grey source a grey in Color, and bright colours keep their values above 1.
        for g in [1.5f32, 2.0, 4.0, 1000.0] {
            let grey = [g; 3];
            for cs in colors {
                for m in [BlendMode::Hue, BlendMode::Saturation] {
                    let r = blend_rgb_within(m, grey, cs, f32::MAX);
                    assert!(r.iter().all(|&v| (v - lum(grey)).abs() <= 1e-5 * g), "{m:?} {g} {cs:?}: {r:?}");
                }
            }
            let r = blend_rgb_within(BlendMode::Luminosity, [0.5, 0.25, 0.125], grey, f32::MAX);
            assert!(r[0] > 1.0 && r[0] > r[1] && r[1] > r[2], "{g}: {r:?}");
            let r = blend_rgb_within(BlendMode::Color, grey, [3.0, 0.25, 0.125], f32::MAX);
            assert!(r[0] > 1.0 && r[0] > r[1] && r[1] > r[2], "{g}: {r:?}");
            // Integer documents: no colour of that luminosity fits in 0..1, so it is its grey.
            let r = blend_rgb_within(BlendMode::Luminosity, [0.5, 0.25, 0.125], grey, 1.0);
            assert!(r.iter().all(|&v| (v - lum(grey)).abs() <= 1e-5 * g), "{g}: {r:?}");
        }
        // Continuous across near-greys: the result barely moves from the grey's.
        for m in NON_SEPARABLE {
            let a = blend_rgb_within(m, [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], f32::MAX);
            let b = blend_rgb_within(m, [2.0, 2.0, 1.999_999], [4.0, 3.999_999, 4.0], f32::MAX);
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() <= 1e-4), "{m:?}: {a:?} {b:?}");
        }
    }

    #[test]
    fn darker_lighter_color_pick_whole_pixel() {
        let a = [0.9, 0.1, 0.1];
        let b = [0.2, 0.6, 0.2];
        assert_eq!(blend_rgb(BlendMode::DarkerColor, a, b), if lum(b) < lum(a) { b } else { a });
        assert_eq!(blend_rgb(BlendMode::LighterColor, a, b), if lum(b) > lum(a) { b } else { a });
    }

    #[test]
    fn composite_alpha_rules() {
        let bg = [0.2, 0.4, 0.6, 1.0];
        // Transparent source leaves backdrop.
        assert_eq!(composite(BlendMode::Multiply, bg, [1.0, 0.0, 0.0, 0.0], 1.0), bg);
        // Opaque normal source replaces.
        let r = composite(BlendMode::Normal, bg, [1.0, 0.0, 0.0, 1.0], 1.0);
        assert!(close(r[0], 1.0) && close(r[3], 1.0));
        // Half opacity normal = lerp.
        let r = composite(BlendMode::Normal, bg, [1.0, 1.0, 1.0, 1.0], 0.5);
        assert!(close(r[0], 0.6) && close(r[1], 0.7) && close(r[2], 0.8));
        // Blend over transparent backdrop yields source regardless of mode.
        let r = composite(BlendMode::Multiply, [0.0; 4], [0.3, 0.5, 0.7, 1.0], 1.0);
        assert!(close(r[0], 0.3) && close(r[1], 0.5) && close(r[2], 0.7) && close(r[3], 1.0));
        // Alpha accumulates.
        let r = composite(BlendMode::Screen, [0.0, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.5], 1.0);
        assert!(close(r[3], 0.75));
    }
}
