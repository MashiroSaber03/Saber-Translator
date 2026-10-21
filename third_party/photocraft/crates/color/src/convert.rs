//! Colour conversions between the document colour models.
//!
//! CMYK ↔ RGB goes through the ICC colour management module (`photocraft-cms`) with the
//! built-in coated CMYK profile and sRGB (relative colorimetric with black point
//! compensation, Photoshop's default), so CMYK pixels display and accept painted colours the
//! way a colour-managed editor does. A document's embedded CMYK profile replaces the built-in
//! one while its [`CmykSpace`] is active on the thread ([`with_cmyk_space`]: the compositor,
//! the GPU upload and composite exports enter it); otherwise the built-in profiles apply.
//! Lab uses the exact D50 formulas; gray is treated as sGray (gray `v` = sRGB `(v, v, v)`).

use std::cell::RefCell;
use std::sync::{Arc, Mutex, OnceLock, Weak};

use photocraft_cms::{Builtin, ColorSpace, Intent, Profile, Transform};

/// A document's own CMYK profile, linked to sRGB both ways (relative colorimetric + BPC, like
/// the built-in defaults). While a space is active on a thread ([`with_cmyk_space`]),
/// [`cmyk_to_rgb`] and [`rgb_to_cmyk`] (and so `Surface::rgba`, the compositor's reads of CMYK
/// layers and CMYK fill colours) use it instead of the built-in coated CMYK.
#[derive(Debug)]
pub struct CmykSpace {
    to_srgb: Transform,
    from_srgb: Transform,
    /// Content hash of the profile (cache keys of converted pixels).
    pub id: u64,
}

impl CmykSpace {
    /// The space of a document's embedded profile bytes: `None` when there are none, they do
    /// not parse as a CMYK profile, or they are the built-in coated CMYK (the default path).
    /// Cached by allocation, so calling this per render is cheap.
    pub fn for_profile(icc: Option<&Arc<Vec<u8>>>) -> Option<Arc<CmykSpace>> {
        type Cache = Mutex<Vec<(Weak<Vec<u8>>, Option<Arc<CmykSpace>>)>>;
        static CACHE: OnceLock<Cache> = OnceLock::new();
        let bytes = icc?;
        let cache = CACHE.get_or_init(Default::default);
        {
            let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
            c.retain(|(w, _)| w.strong_count() > 0);
            if let Some((_, s)) = c.iter().find(|(w, _)| w.as_ptr() == Arc::as_ptr(bytes)) {
                return s.clone();
            }
        }
        let space = Self::build(bytes);
        cache.lock().unwrap_or_else(|e| e.into_inner()).push((Arc::downgrade(bytes), space.clone()));
        space
    }

    fn build(bytes: &[u8]) -> Option<Arc<CmykSpace>> {
        let p = Profile::parse(bytes).ok().filter(|p| p.color_space == ColorSpace::Cmyk)?;
        let coated = Builtin::CoatedCmyk.profile();
        if p.content_hash() == coated.content_hash() {
            return None;
        }
        let srgb = Builtin::Srgb.profile();
        let to_srgb = Transform::new(&p, srgb, Intent::RelativeColorimetric, true).ok()?;
        let from_srgb = Transform::new(srgb, &p, Intent::RelativeColorimetric, true).ok()?;
        Some(Arc::new(CmykSpace { to_srgb, from_srgb, id: p.content_hash() }))
    }
}

thread_local! {
    static ACTIVE_CMYK: RefCell<Option<Arc<CmykSpace>>> = const { RefCell::new(None) };
}

/// Restores the previously active CMYK space when dropped (also on unwind).
struct Restore(Option<Arc<CmykSpace>>);

impl Drop for Restore {
    fn drop(&mut self) {
        let prev = self.0.take();
        ACTIVE_CMYK.with(|a| *a.borrow_mut() = prev);
    }
}

/// Runs `f` with `space` as this thread's CMYK profile (`None`: the built-in coated CMYK).
/// Thread-local: parallel workers must enter the scope themselves.
pub fn with_cmyk_space<R>(space: Option<&Arc<CmykSpace>>, f: impl FnOnce() -> R) -> R {
    if space.is_none() && ACTIVE_CMYK.with(|a| a.borrow().is_none()) {
        return f();
    }
    let prev = ACTIVE_CMYK.with(|a| std::mem::replace(&mut *a.borrow_mut(), space.cloned()));
    let _restore = Restore(prev);
    f()
}

/// The CMYK space active on this thread (see [`with_cmyk_space`]).
pub fn active_cmyk_space() -> Option<Arc<CmykSpace>> {
    ACTIVE_CMYK.with(|a| a.borrow().clone())
}

// The built-in profiles always link (the tests use these transforms); `None` only guards
// against a cms regression, and then the conversions fall back to the naive formulas.
fn cmyk_to_srgb_transform() -> Option<&'static Transform> {
    static T: OnceLock<Option<Transform>> = OnceLock::new();
    T.get_or_init(|| Transform::new(Builtin::CoatedCmyk.profile(), Builtin::Srgb.profile(), Intent::RelativeColorimetric, true).ok()).as_ref()
}

fn srgb_to_cmyk_transform() -> Option<&'static Transform> {
    static T: OnceLock<Option<Transform>> = OnceLock::new();
    T.get_or_init(|| Transform::new(Builtin::Srgb.profile(), Builtin::CoatedCmyk.profile(), Intent::RelativeColorimetric, true).ok()).as_ref()
}

/// Colour-managed CMYK → sRGB (the active document CMYK profile, else the built-in coated
/// CMYK; relative colorimetric + BPC).
#[inline]
pub fn cmyk_to_rgb(c: [f32; 4]) -> [f32; 3] {
    let mut o = [0.0f32; 3];
    // The active document's embedded CMYK profile, if one is in scope.
    let active = ACTIVE_CMYK.with(|a| match &*a.borrow() {
        Some(s) => {
            s.to_srgb.eval_fast(&c, &mut o);
            true
        }
        None => false,
    });
    if !active {
        let Some(t) = cmyk_to_srgb_transform() else {
            return cmyk_to_rgb_naive(c);
        };
        t.eval_fast(&c, &mut o);
    }
    o
}

/// Colour-managed sRGB → CMYK (the active document CMYK profile, else the built-in coated
/// CMYK; relative colorimetric + BPC).
#[inline]
pub fn rgb_to_cmyk(rgb: [f32; 3]) -> [f32; 4] {
    let v = [rgb[0].clamp(0.0, 1.0), rgb[1].clamp(0.0, 1.0), rgb[2].clamp(0.0, 1.0)];
    let mut o = [0.0f32; 4];
    let active = ACTIVE_CMYK.with(|a| match &*a.borrow() {
        Some(s) => {
            s.from_srgb.eval_fast(&v, &mut o);
            true
        }
        None => false,
    });
    if !active {
        let Some(t) = srgb_to_cmyk_transform() else {
            return rgb_to_cmyk_naive(rgb);
        };
        t.eval_fast(&v, &mut o);
    }
    o
}

/// sRGB transfer function: encoded → linear.
#[inline]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// sRGB transfer function: linear → encoded.
#[inline]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Naive CMYK → RGB (no profile): `(1 − c)(1 − k)`. Kept for reference and tests.
pub fn cmyk_to_rgb_naive(c: [f32; 4]) -> [f32; 3] {
    let k = 1.0 - c[3];
    [(1.0 - c[0]) * k, (1.0 - c[1]) * k, (1.0 - c[2]) * k]
}

/// Naive RGB → CMYK with full GCR.
pub fn rgb_to_cmyk_naive(rgb: [f32; 3]) -> [f32; 4] {
    let k = 1.0 - rgb[0].max(rgb[1]).max(rgb[2]);
    if k >= 1.0 - 1e-6 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let d = 1.0 - k;
    [(1.0 - rgb[0] - k) / d, (1.0 - rgb[1] - k) / d, (1.0 - rgb[2] - k) / d, k]
}

/// The D50 white point (ICC PCS).
pub const D50: [f32; 3] = [0.964_22, 1.0, 0.825_21];

/// Linear sRGB → D50 XYZ (Bradford-adapted, as in the ICC sRGB profile). Rows sum to [`D50`].
pub const SRGB_TO_XYZ_D50: [[f32; 3]; 3] = [[0.436_074, 0.385_064, 0.143_080], [0.222_504, 0.716_878, 0.060_618], [0.013_932, 0.097_104, 0.714_173]];

/// D50 XYZ → linear sRGB (the inverse of [`SRGB_TO_XYZ_D50`]).
pub const XYZ_D50_TO_SRGB: [[f32; 3]; 3] = [[3.133_856, -1.616_867, -0.490_615], [-0.978_768, 1.916_142, 0.033_454], [0.071_945, -0.228_991, 1.405_243]];

/// `m · v` for a row-major 3 × 3 matrix.
#[inline]
pub fn mat3_mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    m.map(|r| r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
}

/// CIE L*a*b* (D50) → encoded sRGB (Bradford-adapted to D65).
pub fn lab_to_srgb(lab: [f32; 3]) -> [f32; 3] {
    let [l, a, b] = lab;
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let finv = |t: f32| if t > 6.0 / 29.0 { t * t * t } else { 3.0 * (6.0f32 / 29.0).powi(2) * (t - 4.0 / 29.0) };
    let xyz = [finv(fx) * D50[0], finv(fy) * D50[1], finv(fz) * D50[2]];
    mat3_mul(&XYZ_D50_TO_SRGB, xyz).map(|lin| linear_to_srgb(lin.clamp(0.0, 1.0)))
}

/// Encoded sRGB → CIE L*a*b* (D50).
pub fn srgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let lin = rgb.map(srgb_to_linear);
    let xyz = mat3_mul(&SRGB_TO_XYZ_D50, lin);
    let f = |t: f32| if t > (6.0f32 / 29.0).powi(3) { t.cbrt() } else { t / (3.0 * (6.0f32 / 29.0).powi(2)) + 4.0 / 29.0 };
    let (fx, fy, fz) = (f(xyz[0] / D50[0]), f(xyz[1] / D50[1]), f(xyz[2] / D50[2]));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// Rec.601 luma used by Photoshop's Grayscale conversion default.
pub fn rgb_to_gray(rgb: [f32; 3]) -> f32 {
    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_roundtrip() {
        for i in 0..=255 {
            let v = i as f32 / 255.0;
            assert!((linear_to_srgb(srgb_to_linear(v)) - v).abs() < 1e-5);
        }
    }

    #[test]
    fn cmyk_roundtrip() {
        for rgb in [[1.0, 0.0, 0.0], [0.2, 0.4, 0.6], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]] {
            let back = cmyk_to_rgb_naive(rgb_to_cmyk_naive(rgb));
            for i in 0..3 {
                assert!((back[i] - rgb[i]).abs() < 1e-5, "{rgb:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn managed_cmyk() {
        let white = cmyk_to_rgb([0.0; 4]);
        assert!(white.iter().all(|v| *v > 0.99), "{white:?}");
        // 100 % K alone is a dark neutral, not pure black (as in any real CMYK profile).
        let k = cmyk_to_rgb([0.0, 0.0, 0.0, 1.0]);
        assert!(k[0] < 0.3 && (k[0] - k[2]).abs() < 0.05, "{k:?}");
        // Round trip of in-gamut colours.
        for rgb in [[0.5, 0.5, 0.5], [0.6, 0.4, 0.3], [0.3, 0.5, 0.6]] {
            let back = cmyk_to_rgb(rgb_to_cmyk(rgb));
            for i in 0..3 {
                assert!((back[i] - rgb[i]).abs() < 0.03, "{rgb:?} -> {back:?}");
            }
        }
    }

    #[test]
    fn document_cmyk_space_scope() {
        // A synthetic uncoated-like profile (heavier dot gain than the built-in coated one).
        let params = photocraft_cms::synth::CmykParams {
            description: "Test Uncoated".into(),
            tvi: [0.26, 0.26, 0.26, 0.3],
            grid_a2b: 5,
            grid_b2a: 9,
            ..Default::default()
        };
        let p = photocraft_cms::synth::cmyk_profile(&params);
        let bytes = p.to_bytes();
        let space = CmykSpace::for_profile(Some(&bytes)).expect("a CMYK profile other than the default");
        // Cached by allocation.
        assert!(Arc::ptr_eq(&space, &CmykSpace::for_profile(Some(&bytes)).unwrap()));
        // The built-in profile, absent or non-CMYK bytes keep the default path.
        assert!(CmykSpace::for_profile(Some(&Builtin::CoatedCmyk.profile().to_bytes())).is_none());
        assert!(CmykSpace::for_profile(Some(&Builtin::Srgb.profile().to_bytes())).is_none());
        assert!(CmykSpace::for_profile(Some(&Arc::new(vec![1, 2, 3]))).is_none());
        assert!(CmykSpace::for_profile(None).is_none());
        let ink = [0.0, 0.5, 0.5, 0.0];
        let default = cmyk_to_rgb(ink);
        let mut expect = [0.0f32; 3];
        Transform::new(&p, Builtin::Srgb.profile(), Intent::RelativeColorimetric, true).unwrap().eval(&ink, &mut expect);
        let inside = with_cmyk_space(Some(&space), || {
            assert!(active_cmyk_space().is_some());
            cmyk_to_rgb(ink)
        });
        for i in 0..3 {
            assert!((inside[i] - expect[i]).abs() < 0.01, "{inside:?} vs {expect:?}");
        }
        assert!((0..3).any(|i| (inside[i] - default[i]).abs() > 0.02), "the document profile changes the colour: {inside:?} {default:?}");
        // Restored afterwards (and nested scopes restore the outer one).
        assert!(active_cmyk_space().is_none());
        assert_eq!(cmyk_to_rgb(ink), default);
        with_cmyk_space(Some(&space), || {
            with_cmyk_space(None, || assert!(active_cmyk_space().is_none()));
            assert!(active_cmyk_space().is_some());
            let back = cmyk_to_rgb(rgb_to_cmyk([0.6, 0.4, 0.3]));
            assert!((back[0] - 0.6).abs() < 0.04 && (back[2] - 0.3).abs() < 0.04, "{back:?}");
        });
    }

    #[test]
    fn lab_roundtrip_and_anchors() {
        let white = srgb_to_lab([1.0, 1.0, 1.0]);
        assert!((white[0] - 100.0).abs() < 0.1 && white[1].abs() < 0.5 && white[2].abs() < 0.5, "{white:?}");
        let black = srgb_to_lab([0.0, 0.0, 0.0]);
        assert!(black[0].abs() < 0.1);
        for rgb in [[0.8, 0.3, 0.1], [0.1, 0.5, 0.9], [0.5, 0.5, 0.5]] {
            let back = lab_to_srgb(srgb_to_lab(rgb));
            for i in 0..3 {
                assert!((back[i] - rgb[i]).abs() < 2e-3, "{rgb:?} -> {back:?}");
            }
        }
    }
}
