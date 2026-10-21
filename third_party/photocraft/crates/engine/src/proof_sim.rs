//! View › Proof Setup simulations beyond a plain profile proof, and View › 32-bit Preview
//! Options. Both become the canvas display LUT (`ColorState::display_lut`).
//!
//! * Working Cyan/Magenta/Yellow/Black Plate: the composite is separated into the working CMYK
//!   (proof profile, its intent and BPC) and only that plate is printed, shown as gray ink (the
//!   plate's value printed with black ink) like Photoshop's grayscale plate preview. Working CMY
//!   Plate prints cyan, magenta and yellow with the black plate left out.
//! * Legacy Macintosh RGB: the document's RGB numbers shown on a gamma-1.8 display (numbers
//!   preserved, sRGB primaries).
//! * Color Blindness (protanopia / deuteranopia): Viénot, Brettel & Mollon (1999), "Digital
//!   video colourmaps for checking the legibility of displays by dichromats", Color Research &
//!   Application 24(4): linear RGB → LMS, the missing cone response replaced by its projection
//!   onto the dichromat's plane, back to RGB (gamut first reduced as in the paper).
//! * 32-bit Preview Options: exposure (stops) and gamma applied in linear light before the
//!   display transform, for 32-bit documents. Highlight Compression maps the canvas range to the
//!   display unchanged: the canvas texture is clamped to 0–1, so there is nothing above white
//!   to compress.

use photocraft_cms::{Builtin, Intent, Lut3d, Transform};
use photocraft_color::SampleType;
use photocraft_doc::Document;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::color_cmds::ColorState;
use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// What a proof simulates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ProofKind {
    /// The proof profile (Working CMYK, Custom…).
    #[default]
    Profile,
    /// One plate of the proof profile (0 cyan, 1 magenta, 2 yellow, 3 black).
    Plate(u8),
    /// Cyan, magenta and yellow plates (no black).
    CmyPlate,
    LegacyMacintoshRgb,
    Protanopia,
    Deuteranopia,
}

impl ProofKind {
    pub fn id(self) -> &'static str {
        match self {
            ProofKind::Profile => "profile",
            ProofKind::Plate(0) => "workingCyanPlate",
            ProofKind::Plate(1) => "workingMagentaPlate",
            ProofKind::Plate(2) => "workingYellowPlate",
            ProofKind::Plate(_) => "workingBlackPlate",
            ProofKind::CmyPlate => "workingCmyPlate",
            ProofKind::LegacyMacintoshRgb => "legacyMacintoshRgb",
            ProofKind::Protanopia => "colorBlindnessProtanopia",
            ProofKind::Deuteranopia => "colorBlindnessDeuteranopia",
        }
    }
    pub fn from_id(s: &str) -> Option<ProofKind> {
        Some(match s {
            "profile" => ProofKind::Profile,
            "workingCyanPlate" => ProofKind::Plate(0),
            "workingMagentaPlate" => ProofKind::Plate(1),
            "workingYellowPlate" => ProofKind::Plate(2),
            "workingBlackPlate" => ProofKind::Plate(3),
            "workingCmyPlate" => ProofKind::CmyPlate,
            "legacyMacintoshRgb" => ProofKind::LegacyMacintoshRgb,
            "colorBlindnessProtanopia" => ProofKind::Protanopia,
            "colorBlindnessDeuteranopia" => ProofKind::Deuteranopia,
            _ => return None,
        })
    }
}

/// View › 32-bit Preview Options.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct HdrPreview {
    /// "exposureGamma" | "highlightCompression".
    pub highlight_compression: bool,
    /// Stops, −20…20.
    pub exposure: f32,
    /// 0.1…9.99.
    pub gamma: f32,
}

impl Default for HdrPreview {
    fn default() -> Self {
        Self { highlight_compression: false, exposure: 0.0, gamma: 1.0 }
    }
}

impl HdrPreview {
    pub fn is_identity(&self) -> bool {
        self.highlight_compression || (self.exposure == 0.0 && self.gamma == 1.0)
    }
    fn apply(&self, v: f32) -> f32 {
        if self.is_identity() {
            return v;
        }
        let lin = srgb_decode(v) * 2f32.powf(self.exposure);
        srgb_encode(lin.max(0.0).powf(1.0 / self.gamma).min(1.0))
    }
}

/// Is a 32-bit preview adjustment active for `doc` (a 32-bit document with non-default options)?
pub fn hdr_active(c: &ColorState, doc: &Document) -> bool {
    doc.depth == SampleType::F32 && c.hdr.get(&doc.id).is_some_and(|h| !h.is_identity())
}

/// Changes whenever the display LUT of `doc` changes because of this module (for UI caches).
pub fn signature(c: &ColorState, doc: &Document) -> String {
    let pv = c.proof(doc.id);
    let h = if hdr_active(c, doc) { c.hdr.get(&doc.id).copied() } else { None };
    format!("{:?} {:?}", pv.setup.kind, h)
}

fn srgb_decode(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn srgb_encode(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Viénot et al. (1999): linear RGB → LMS.
const RGB_LMS: [[f32; 3]; 3] = [[17.8824, 43.5161, 4.11935], [3.45565, 27.1554, 3.86714], [0.0299566, 0.184309, 1.46709]];
/// LMS → linear RGB (the paper's values, kept verbatim).
#[allow(clippy::excessive_precision)]
const LMS_RGB: [[f32; 3]; 3] =
    [[0.080_944_45, -0.130_504_41, 0.116_721_07], [-0.010_248_533, 0.054_019_327, -0.113_614_71], [-0.000_365_297, -0.004_121_614_7, 0.693_511_4]];

fn mul(m: &[[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// Simulate a dichromat's view of a linear-RGB colour.
pub fn dichromat(rgb_lin: [f32; 3], protan: bool) -> [f32; 3] {
    // Gamut reduction so the simulated colours stay displayable.
    let v = rgb_lin.map(|c| 0.992_052 * c + 0.003_974);
    let [l, m, s] = mul(&RGB_LMS, v);
    let lms = if protan { [2.02344 * m - 2.52581 * s, m, s] } else { [l, 0.494_207 * l + 1.24827 * s, s] };
    mul(&LMS_RGB, lms).map(|c| c.clamp(0.0, 1.0))
}

/// The display LUT when a simulation or 32-bit preview applies (None = plain profile proof).
pub fn display_lut(c: &ColorState, doc: &Document, size: usize) -> Result<Option<Lut3d>> {
    display_lut_with(c, doc, size, true, c.main_display)
}

/// [`display_lut`], leaving the 32-bit preview out when `include_hdr` is false (the GPU canvas
/// applies it in its shader).
pub fn display_lut_with(c: &ColorState, doc: &Document, size: usize, include_hdr: bool, display: Option<u32>) -> Result<Option<Lut3d>> {
    let pv = c.proof(doc.id);
    let kind = if pv.enabled { pv.setup.kind } else { ProofKind::Profile };
    let hdr = if include_hdr { c.hdr_preview(doc) } else { None };
    if kind == ProofKind::Profile && hdr.is_none() {
        return Ok(None);
    }
    let err = |e: photocraft_cms::CmsError| EngineError::Other(format!("colour management: {e}"));
    let src = c.canvas_display_for(doc, display)?.source.clone();
    let mon = c.monitor_for(display);
    let srgb = Builtin::Srgb.profile();
    let setup = &pv.setup;
    // composite → display for the plain case (with the profile proof when on).
    let base: Box<dyn Fn([f32; 3]) -> [f32; 3]> = match kind {
        ProofKind::Profile => {
            let t = if pv.enabled {
                Transform::proof(&src, &setup.profile, &mon, setup.intent, setup.bpc, setup.simulate_paper)
            } else {
                Transform::new(&src, &mon, crate::display_color::DISPLAY_INTENT, crate::display_color::DISPLAY_BPC)
            }
            .map_err(err)?;
            Box::new(move |v| eval3(&t, &v))
        }
        ProofKind::Plate(_) | ProofKind::CmyPlate => {
            if setup.profile.channels() != 4 {
                return Err(EngineError::Other("plate previews need a CMYK proof profile".into()));
            }
            let sep = Transform::new(&src, &setup.profile, setup.intent, setup.bpc).map_err(err)?;
            let print = Transform::new(&setup.profile, &mon, Intent::RelativeColorimetric, true).map_err(err)?;
            Box::new(move |v| {
                let mut ink = [0.0f32; 16];
                sep.eval(&v, &mut ink);
                let shown = match kind {
                    ProofKind::Plate(p) => [0.0, 0.0, 0.0, ink[usize::from(p.min(3))]],
                    _ => [ink[0], ink[1], ink[2], 0.0],
                };
                eval3(&print, &shown)
            })
        }
        ProofKind::LegacyMacintoshRgb => {
            let t = Transform::new(srgb, &mon, Intent::RelativeColorimetric, true).map_err(err)?;
            // Numbers preserved, shown with a 1.8 gamma instead of the sRGB curve.
            Box::new(move |v| eval3(&t, &v.map(|x| srgb_encode(x.max(0.0).powf(1.8)))))
        }
        ProofKind::Protanopia | ProofKind::Deuteranopia => {
            let to_srgb = Transform::new(&src, srgb, Intent::RelativeColorimetric, true).map_err(err)?;
            let to_mon = Transform::new(srgb, &mon, Intent::RelativeColorimetric, true).map_err(err)?;
            let protan = kind == ProofKind::Protanopia;
            Box::new(move |v| {
                let s = eval3(&to_srgb, &v);
                let sim = dichromat(s.map(srgb_decode), protan).map(srgb_encode);
                eval3(&to_mon, &sim)
            })
        }
    };
    let n = size.max(2);
    let sc = (n - 1) as f32;
    let mut data = vec![[0.0f32, 0.0, 0.0, 1.0]; n * n * n];
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let mut v = [r as f32 / sc, g as f32 / sc, b as f32 / sc];
                if let Some(h) = hdr {
                    v = v.map(|x| h.apply(x));
                }
                let o = base(v);
                data[r + g * n + b * n * n] = [o[0], o[1], o[2], 1.0];
            }
        }
    }
    Ok(Some(Lut3d { size: n, data }))
}

fn eval3(t: &Transform, v: &[f32]) -> [f32; 3] {
    let mut out = [0.0f32; 16];
    t.eval(v, &mut out);
    if t.outputs() == 1 { [out[0]; 3] } else { [out[0], out[1], out[2]] }
}

// ------------------------------------------------------------------ commands

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn is_32(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.depth == SampleType::F32 { Ok(()) } else { Err("32-bit Preview Options apply to 32-bit documents".into()) }
}

/// Select a simulated proof (`kind`) and turn Proof Colors on. Plates use the working CMYK.
pub fn set_kind(s: &mut Session, kind: ProofKind) -> Result<Value> {
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let (id, doc) = (st.doc.id, st.doc.clone());
    let plates = matches!(kind, ProofKind::Plate(_) | ProofKind::CmyPlate);
    let working = if plates { Some(s.color.resolve("working-cmyk", Some(&doc), Some(photocraft_color::ColorMode::Cmyk))?) } else { None };
    let pv = s.color.proof_mut(id);
    if let Some(w) = working {
        pv.setup.profile = w;
        pv.setup.name = "working-cmyk".into();
    }
    pv.setup.kind = kind;
    pv.enabled = true;
    Ok(json!({"proof": kind.id(), "profile": pv.setup.profile.description, "proofColors": true}))
}

fn run_kind(s: &mut Session, id: &str) -> Result<Value> {
    let kind = id.strip_prefix("view.proofSetup.").and_then(ProofKind::from_id).ok_or_else(|| EngineError::UnknownCommand(id.into()))?;
    set_kind(s, kind)
}

fn preview_32(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "view.thirtyTwoBitPreviewOptions";
    let id = s.active().ok_or(EngineError::NoDocument)?.doc.id;
    let mut h = s.color.hdr.get(&id).copied().unwrap_or_default();
    if let Some(m) = p.get("method").and_then(Value::as_str) {
        h.highlight_compression = match m {
            "exposureGamma" | "exposureAndGamma" => false,
            "highlightCompression" => true,
            o => return Err(EngineError::BadParams { cmd: CMD.into(), msg: format!("method `{o}` (exposureGamma|highlightCompression)") }),
        };
    }
    if let Some(e) = p.get("exposure").and_then(Value::as_f64) {
        h.exposure = (e as f32).clamp(-20.0, 20.0);
    }
    if let Some(g) = p.get("gamma").and_then(Value::as_f64) {
        h.gamma = (g as f32).clamp(0.1, 9.99);
    }
    s.color.hdr.insert(id, h);
    Ok(json!({"method": if h.highlight_compression { "highlightCompression" } else { "exposureGamma" }, "exposure": h.exposure, "gamma": h.gamma}))
}

macro_rules! kind_spec {
    ($id:literal, $label:literal) => {
        CommandSpec {
            id: $id,
            label: $label,
            menu: &["View", "Proof Setup"],
            shortcut: None,
            params: "{} (sets the proof and turns Proof Colors on)",
            enabled: has_doc,
            run: |s, _| run_kind(s, $id),
            journal: false,
        }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        kind_spec!("view.proofSetup.workingCyanPlate", "Working Cyan Plate"),
        kind_spec!("view.proofSetup.workingMagentaPlate", "Working Magenta Plate"),
        kind_spec!("view.proofSetup.workingYellowPlate", "Working Yellow Plate"),
        kind_spec!("view.proofSetup.workingBlackPlate", "Working Black Plate"),
        kind_spec!("view.proofSetup.workingCmyPlate", "Working CMY Plate"),
        kind_spec!("view.proofSetup.legacyMacintoshRgb", "Legacy Macintosh RGB"),
        kind_spec!("view.proofSetup.colorBlindnessProtanopia", "Color Blindness — Protanopia-type"),
        kind_spec!("view.proofSetup.colorBlindnessDeuteranopia", "Color Blindness — Deuteranopia-type"),
        CommandSpec {
            id: "view.thirtyTwoBitPreviewOptions",
            label: "32-bit Preview Options…",
            menu: &["View"],
            shortcut: None,
            params: r##"{"method":"exposureGamma|highlightCompression"="exposureGamma","exposure":-20..20=0,"gamma":0.1..9.99=1}"##,
            enabled: is_32,
            run: preview_32,
            journal: false,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session(mode: &str, depth: u32) -> Session {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 8, "height": 8, "mode": mode, "depth": depth})).unwrap();
        s
    }

    fn sample(lut: &Lut3d, idx: [usize; 3]) -> [f32; 3] {
        let n = lut.size;
        let v = lut.data[idx[0] + idx[1] * n + idx[2] * n * n];
        [v[0], v[1], v[2]]
    }

    #[test]
    fn dichromat_keeps_neutrals_and_merges_red_green() {
        for protan in [true, false] {
            let w = dichromat([1.0, 1.0, 1.0], protan);
            assert!(w.iter().all(|c| (c - 1.0).abs() < 0.02), "{w:?}");
            let g = dichromat([0.2, 0.2, 0.2], protan);
            assert!(g.iter().all(|c| (c - 0.2).abs() < 0.02), "{g:?}");
            // Red and green become (nearly) indistinguishable in hue: both yellowish/brown.
            let r = dichromat([1.0, 0.0, 0.0], protan);
            let gr = dichromat([0.0, 1.0, 0.0], protan);
            assert!((r[0] - r[1]).abs() < 0.15 && (gr[0] - gr[1]).abs() < 0.15, "{r:?} {gr:?}");
            // Blue stays blue.
            let b = dichromat([0.0, 0.0, 1.0], protan);
            assert!(b[2] > 0.8);
        }
        // Protanopes see red much darker than deuteranopes do.
        assert!(dichromat([1.0, 0.0, 0.0], true)[0] < dichromat([1.0, 0.0, 0.0], false)[0]);
    }

    #[test]
    fn plates_and_simulations_build_luts() {
        for depth in [8, 16, 32] {
            let mut s = session("rgb", depth);
            for id in [
                "view.proofSetup.workingCyanPlate",
                "view.proofSetup.workingBlackPlate",
                "view.proofSetup.workingCmyPlate",
                "view.proofSetup.legacyMacintoshRgb",
                "view.proofSetup.colorBlindnessProtanopia",
                "view.proofSetup.colorBlindnessDeuteranopia",
            ] {
                let r = s.execute(id, json!({})).unwrap();
                assert_eq!(r["proofColors"], true);
                let d = s.active().unwrap().doc.clone();
                let pv = s.color.proof(d.id);
                assert_eq!(pv.setup.kind.id(), id.trim_start_matches("view.proofSetup."));
                let lut = s.color.display_lut(&d, 5).unwrap();
                assert_eq!(lut.data.len(), 125);
                let bytes = s.color.canvas_lut(&d, 5).unwrap().unwrap();
                assert_eq!(bytes.len(), 125 * 4);
            }
        }
    }

    #[test]
    fn plate_previews_are_gray_and_track_ink() {
        let mut s = session("rgb", 8);
        s.execute("view.proofSetup.workingCyanPlate", json!({})).unwrap();
        let d = s.active().unwrap().doc.clone();
        let lut = s.color.display_lut(&d, 5).unwrap();
        // Pure cyan: heavy cyan plate → dark gray; white: no ink → white; red: little cyan.
        let cyan = sample(&lut, [0, 4, 4]);
        let white = sample(&lut, [4, 4, 4]);
        let red = sample(&lut, [4, 0, 0]);
        for c in [cyan, white, red] {
            assert!((c[0] - c[1]).abs() < 0.03 && (c[1] - c[2]).abs() < 0.03, "plates are gray: {c:?}");
        }
        assert!(white[0] > 0.9, "{white:?}");
        assert!(cyan[0] < red[0] && cyan[0] < 0.6, "cyan {cyan:?} red {red:?}");
        // Back to the plain working CMYK proof via Proof Setup.
        s.execute("view.proofSetup", json!({"profile": "working-cmyk"})).unwrap();
        assert_eq!(s.color.proof(d.id).setup.kind, ProofKind::Profile);
    }

    #[test]
    fn legacy_mac_is_darker_in_midtones() {
        let mut s = session("rgb", 8);
        s.execute("view.proofSetup.legacyMacintoshRgb", json!({})).unwrap();
        let d = s.active().unwrap().doc.clone();
        let lut = s.color.display_lut(&d, 5).unwrap();
        let mid = sample(&lut, [2, 2, 2]);
        let white = sample(&lut, [4, 4, 4]);
        // 0.5^1.8 = 0.287 linear vs 0.214 for sRGB 0.5: the gamma-1.8 display is brighter.
        assert!(mid[0] > 0.52 && mid[0] < 0.62, "{mid:?}");
        assert!(white[0] > 0.99);
    }

    #[test]
    fn thirty_two_bit_preview_options() {
        let mut s = session("rgb", 8);
        assert!(s.execute("view.thirtyTwoBitPreviewOptions", json!({"exposure": 1})).is_err(), "8-bit: disabled");
        let mut s = session("rgb", 32);
        let d = s.active().unwrap().doc.clone();
        assert!(s.color.canvas_lut(&d, 5).unwrap().is_none());
        let r = s.execute("view.thirtyTwoBitPreviewOptions", json!({"exposure": 1.0, "gamma": 1.0})).unwrap();
        assert_eq!(r["exposure"], 1.0);
        assert!(hdr_active(&s.color, &d));
        let lut = s.color.display_lut(&d, 5).unwrap();
        // +1 stop doubles linear light: sRGB 0.25 (0.0508 linear) → 0.1016 linear ≈ 0.354 sRGB.
        let q = sample(&lut, [1, 1, 1]);
        assert!((q[0] - 0.354).abs() < 0.02, "{q:?}");
        assert!(s.color.canvas_lut(&d, 5).unwrap().is_some());
        // The GPU canvas applies it in its shader: its LUT leaves it out (none for sRGB).
        assert_eq!(s.color.hdr_preview(&d), Some(HdrPreview { highlight_compression: false, exposure: 1.0, gamma: 1.0 }));
        assert!(s.color.gpu_canvas_lut(&d, 5).unwrap().is_none());
        // Highlight Compression: identity mapping, so no LUT.
        s.execute("view.thirtyTwoBitPreviewOptions", json!({"method": "highlightCompression"})).unwrap();
        assert!(!hdr_active(&s.color, &d));
        assert!(s.execute("view.thirtyTwoBitPreviewOptions", json!({"method": "nope"})).is_err());
        // Clamped ranges.
        let r = s.execute("view.thirtyTwoBitPreviewOptions", json!({"method": "exposureGamma", "exposure": 99, "gamma": 0})).unwrap();
        assert_eq!(r["exposure"], 20.0);
        assert!((r["gamma"].as_f64().unwrap() - 0.1).abs() < 1e-6);
    }
}
