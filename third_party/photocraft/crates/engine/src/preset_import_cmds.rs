//! Importing Photoshop preset files.
//!
//! - `brush.presets.importAbr` adds the presets of an `.abr` file (v1, v2, v6+) to the brush
//!   library as one group named after the file, with the file's own folders nested inside it
//!   as Photoshop does (Brushes panel › Import Brushes…, Preset Manager). Parsing lives in
//!   `photocraft-psd` (`abr`), the mapping onto [`BrushSettings`] in `photocraft-io` (`abr_map`).
//! - `gradient.presets.importGrd` adds the gradients of a `.grd` file (version 5) to the
//!   Gradients panel as one group.
//!
//! [`BrushSettings`]: photocraft_paint::BrushSettings

use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn always(_: &Session) -> std::result::Result<(), String> {
    Ok(())
}

/// Largest accepted file (Photoshop's biggest bundled sets are well under this).
pub const MAX_FILE_BYTES: u64 = 512 << 20;

/// File bytes from `data` (base64) or `path`, plus a default group name.
pub(crate) fn file_bytes(p: &Value, cmd: &str, what: &str) -> Result<(Vec<u8>, String)> {
    if let Some(d) = p.get("data") {
        let s = d.as_str().ok_or_else(|| bad(cmd, "`data` must be a base64 string"))?;
        if s.len() as u64 > MAX_FILE_BYTES / 3 * 4 + 4 {
            return Err(bad(cmd, "`data` is too large"));
        }
        let bytes = photocraft_paint::tile::b64_decode(s).ok_or_else(|| bad(cmd, "`data` is not valid base64"))?;
        return Ok((bytes, String::new()));
    }
    let path =
        p.get("path").and_then(Value::as_str).filter(|s| !s.trim().is_empty()).ok_or_else(|| bad(cmd, format!("give `path` ({what}) or `data` (base64)")))?;
    let len = std::fs::metadata(path).map_err(|e| EngineError::Other(format!("{path}: {e}")))?.len();
    if len > MAX_FILE_BYTES {
        return Err(bad(cmd, format!("{path}: file too large ({len} bytes)")));
    }
    let bytes = std::fs::read(path).map_err(|e| EngineError::Other(format!("{path}: {e}")))?;
    let stem = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    Ok((bytes, stem))
}

/// A name not used by any preset in `taken` (case-insensitive): `name`, `name 2`, `name 3`…
fn unique(name: &str, taken: &[String]) -> String {
    let free = |n: &str| !taken.iter().any(|t| t.eq_ignore_ascii_case(n));
    if free(name) {
        return name.to_string();
    }
    (2..).map(|i| format!("{name} {i}")).find(|n| free(n)).unwrap_or_else(|| name.to_string())
}

fn import_abr(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "brush.presets.importAbr";
    let label = "Import Brushes";
    // A background job when started with `Session::start` (#210): reading and decoding the
    // file run on a worker, cancellable before each brush; the library changes on apply.
    let params = p.clone();
    let params_apply = p.clone();
    crate::jobs::run(
        s,
        label,
        false,
        move |ctx| {
            ctx.progress(0.0, "Importing brushes");
            let p = &params;
            let (bytes, stem) = file_bytes(p, cmd, ".abr file")?;
            let group = match p.get("group") {
                Some(v) => v.as_str().map(str::trim).filter(|g| !g.is_empty()).ok_or_else(|| bad(cmd, "`group` must be a non-empty string"))?.to_string(),
                None if !stem.is_empty() => stem,
                None => "Imported Brushes".to_string(),
            };
            ctx.check()?;
            let imp = ctx
                .stage(0.05, 1.0, "Importing brushes", |ctl| photocraft_io::abr_map::read_abr_with(&bytes, &group, ctl))
                .map_err(|e| if ctx.cancelled() { EngineError::Cancelled } else { bad(cmd, e) })?;
            Ok((group, imp))
        },
        move |s, (group, imp)| add_abr_presets(s, &params_apply, group, imp),
    )
}

fn add_abr_presets(s: &mut Session, p: &Value, group: String, imp: photocraft_io::abr_map::AbrImport) -> Result<Value> {
    // Re-importing a file replaces its group instead of duplicating it.
    if p.get("replace").and_then(Value::as_bool).unwrap_or(true) {
        s.tools.presets.retain(|x| x.builtin || x.group != group);
    }
    let mut taken: Vec<String> = s.tools.presets.iter().map(|x| x.name.clone()).collect();
    let mut names = Vec::with_capacity(imp.presets.len());
    for mut preset in imp.presets {
        preset.name = unique(&preset.name, &taken);
        taken.push(preset.name.clone());
        names.push(preset.name.clone());
        s.tools.presets.push(preset);
    }
    s.brush_presets_changed();
    if p.get("select").and_then(Value::as_bool).unwrap_or(false)
        && let Some(first) = names.first()
        && let Some(pr) = photocraft_paint::presets::find(&s.tools.presets, first)
    {
        let mut b = pr.brush.clone().picked_over(&s.tools.brush);
        s.load_brush_tips(&mut b).map_err(|e| bad("brush.presets.importAbr", e))?;
        s.tools.brush = b;
        s.tools.brush_changed();
    }
    // Distinct folder paths the file brought (its Photoshop folders, nested inside `group`).
    let folders: std::collections::BTreeSet<&[String]> =
        s.tools.presets.iter().filter(|x| !x.builtin && x.group == group).flat_map(|x| (1..=x.folder.len()).filter_map(|d| x.folder.get(..d))).collect();
    let folders = folders.len();
    Ok(json!({ "group": group, "imported": names, "count": names.len(), "folders": folders, "version": imp.version, "warnings": imp.warnings }))
}

/// Brush-file import command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "brush.presets.importAbr",
            label: "Import Brushes…",
            menu: &[],
            shortcut: None,
            params: r##"{"path":".abr file"?,"data":base64 bytes?,"group":string?=file name,"replace":bool=true (replace a group of the same name),"select":bool=false (make the first imported preset current)} → {group, imported, count, folders (nested folders kept from the file), version, warnings}"##,
            enabled: always,
            run: import_abr,
            journal: true,
        },
        CommandSpec {
            id: "gradient.presets.importGrd",
            label: "Import Gradients…",
            menu: &[],
            shortcut: None,
            params: r##"{"path":".grd file"?,"data":base64 bytes?,"group":string?=file name,"replace":bool=true (replace a group of the same name)}"##,
            enabled: always,
            run: import_grd,
            journal: true,
        },
    ]
}

// ------------------------------------------------------------------ gradients (.grd)

fn hsb_to_rgb([h, s, v]: [f32; 3]) -> [f32; 3] {
    let h = h.rem_euclid(360.0) / 60.0;
    let c = v * s;
    let x = c * (1.0 - (h % 2.0 - 1.0).abs());
    let (r, g, b) = match h as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

/// CIE Lab (D50, as Photoshop stores it) to sRGB through XYZ with Bradford adaptation to D65.
fn lab_to_rgb([l, a, b]: [f32; 3]) -> [f32; 3] {
    let fy = (l + 16.0) / 116.0;
    let (fx, fz) = (fy + a / 500.0, fy - b / 200.0);
    let inv = |t: f32| if t > 6.0 / 29.0 { t * t * t } else { 3.0 * (6.0f32 / 29.0).powi(2) * (t - 4.0 / 29.0) };
    let (x, y, z) = (0.9642 * inv(fx), inv(fy), 0.8249 * inv(fz));
    // XYZ (D50) → linear sRGB (D65), Bradford-adapted matrix.
    let r = 3.1339 * x - 1.6169 * y - 0.4906 * z;
    let g = -0.9788 * x + 1.9161 * y + 0.0335 * z;
    let bb = 0.0719 * x - 0.2290 * y + 1.4052 * z;
    let gamma = |c: f32| {
        let c = c.clamp(0.0, 1.0);
        if c <= 0.003_130_8 { 12.92 * c } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 }
    };
    [gamma(r), gamma(g), gamma(bb)]
}

fn stop_color(c: photocraft_psd::grd::GrdColor) -> crate::presets::gradients::StopColor {
    use crate::presets::gradients::StopColor;
    use photocraft_psd::grd::GrdColor;
    let clamp = |c: [f32; 3]| StopColor::Rgb(c.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 }));
    match c {
        GrdColor::Foreground => StopColor::Foreground,
        GrdColor::Background => StopColor::Background,
        GrdColor::Rgb(c) => clamp(c),
        GrdColor::Hsb(c) => clamp(hsb_to_rgb(c)),
        // Naive (profile-free) ink → RGB, like an uncalibrated preview.
        GrdColor::Cmyk([c, m, y, k]) => clamp([(1.0 - c) * (1.0 - k), (1.0 - m) * (1.0 - k), (1.0 - y) * (1.0 - k)]),
        GrdColor::Lab(c) => clamp(lab_to_rgb(c)),
        GrdColor::Gray(g) => clamp([1.0 - g; 3]),
    }
}

fn import_grd(s: &mut Session, p: &Value) -> Result<Value> {
    use crate::presets::{Group, gradients::GradientPreset};
    let cmd = "gradient.presets.importGrd";
    let (bytes, stem) = file_bytes(p, cmd, ".grd file")?;
    let group = match p.get("group") {
        Some(v) => v.as_str().map(str::trim).filter(|g| !g.is_empty()).ok_or_else(|| bad(cmd, "`group` must be a non-empty string"))?.to_string(),
        None if !stem.is_empty() => stem,
        None => "Imported Gradients".to_string(),
    };
    let grads = photocraft_psd::grd::parse(&bytes).map_err(|e| bad(cmd, format!("not a readable gradient file (.grd): {e}")))?;
    let mut warnings = Vec::new();
    let mut items = Vec::new();
    let (mut noise, mut midpoints) = (0, false);
    for (i, g) in grads.iter().enumerate() {
        if g.noise || g.stops.len() < 2 {
            noise += 1;
            continue;
        }
        midpoints |= g.stops.iter().any(|st| (st.midpoint - 0.5).abs() > 0.01) || g.opacity.iter().any(|o| (o.2 - 0.5).abs() > 0.01);
        let mut stops: Vec<(f32, _)> = g.stops.iter().map(|st| (st.location, stop_color(st.color))).collect();
        stops.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut opacity: Vec<(f32, f32)> = g.opacity.iter().map(|o| (o.0, o.1)).collect();
        opacity.sort_by(|a, b| a.0.total_cmp(&b.0));
        if opacity.iter().all(|o| o.1 >= 1.0) {
            opacity.clear();
        }
        let name = if g.name.trim().is_empty() { format!("Gradient {}", i + 1) } else { g.name.trim().to_string() };
        items.push(GradientPreset { name, stops, opacity });
    }
    if noise > 0 {
        warnings.push(format!("{noise} noise gradients are not supported and were skipped"));
    }
    if midpoints {
        warnings.push("colour and opacity midpoints other than 50 % are drawn at 50 %".to_string());
    }
    if items.is_empty() {
        return Err(bad(cmd, "the file holds no gradients PhotoCraft can use (noise gradients only)"));
    }
    let names: Vec<String> = items.iter().map(|g| g.name.clone()).collect();
    let st = &mut s.presets;
    if p.get("replace").and_then(Value::as_bool).unwrap_or(true) {
        st.gradients.retain(|g| g.name != group);
    }
    st.gradients.push(Group::new(&group, items));
    s.presets_changed();
    Ok(json!({"group": group, "imported": names, "count": names.len(), "warnings": warnings}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_psd::abr::{AbrSample, LegacyBrush, LegacyTip, write_v6, write_v12};
    use photocraft_psd::descriptor::{Descriptor, UnicodeString, Value as DV};

    fn abr_v2() -> Vec<u8> {
        let tip = AbrSample { id: String::new(), width: 6, height: 4, depth: 8, data: vec![255; 24] };
        write_v12(
            2,
            &[
                LegacyBrush { name: "Hard Round".into(), spacing: 25, anti_alias: true, tip: LegacyTip::Sampled(tip) },
                LegacyBrush {
                    name: String::new(),
                    spacing: 10,
                    anti_alias: true,
                    tip: LegacyTip::Computed { diameter: 9, hardness: 100, angle: 0, roundness: 100 },
                },
            ],
            true,
        )
        .unwrap()
    }

    #[test]
    fn import_adds_a_group_and_paints() {
        let mut s = Session::new();
        let before = s.tools.presets.len();
        let data = photocraft_paint::tile::b64_encode(&abr_v2());
        let r = s.execute("brush.presets.importAbr", json!({"data": data, "group": "Legacy", "select": true})).unwrap();
        assert_eq!(r["count"], 2);
        // The name clashing with the built-in Hard Round is made unique.
        assert_eq!(r["imported"], json!(["Hard Round 2", "9 px Round"]));
        assert_eq!(s.tools.presets.len(), before + 2);
        assert!(s.tools.presets.iter().filter(|p| p.group == "Legacy").all(|p| !p.builtin));
        assert!(matches!(s.tools.brush.tip, photocraft_paint::TipShape::Sampled(_)));
        // Re-import replaces the group.
        s.execute("brush.presets.importAbr", json!({"data": data, "group": "Legacy"})).unwrap();
        assert_eq!(s.tools.presets.len(), before + 2);
        // Imported presets paint.
        s.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        s.execute("paint.stroke", json!({"points": [[5, 30], [60, 30]], "preset": "9 px Round"})).unwrap();
        // From a file on disk, the group defaults to the file name.
        let dir = std::env::temp_dir().join(format!("pc-abr-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("My Set.abr");
        let preset = Descriptor::new("brushPreset")
            .with("Nm  ", DV::Text(UnicodeString::new_nul("Speckle")))
            .with("Brsh", DV::Descriptor(Descriptor::new("sampledBrush").with("sampledData", DV::Text(UnicodeString::new_nul("$s")))));
        let tip = AbrSample { id: "$s".into(), width: 5, height: 5, depth: 16, data: vec![0x80; 50] };
        std::fs::write(&path, write_v6(2, &[tip], &[], &[preset], true).unwrap()).unwrap();
        let r = s.execute("brush.presets.importAbr", json!({"path": path.to_string_lossy()})).unwrap();
        assert_eq!((r["group"].as_str(), r["imported"][0].as_str(), r["version"].as_u64()), (Some("My Set"), Some("Speckle"), Some(6)));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn bad_params_fail_gracefully() {
        let mut s = Session::new();
        let n = s.tools.presets.len();
        for p in [
            json!({}),
            json!({"path": ""}),
            json!({"path": "/nonexistent/brushes.abr"}),
            json!({"data": 42}),
            json!({"data": "%%%not base64"}),
            json!({"data": ""}),
            json!({"data": "AAAA"}),
            json!({"data": photocraft_paint::tile::b64_encode(b"8BPS\0\x01junk")}),
            json!({"data": photocraft_paint::tile::b64_encode(&abr_v2()), "group": 5}),
            json!({"data": photocraft_paint::tile::b64_encode(&abr_v2()), "group": "  "}),
            json!({"data": photocraft_paint::tile::b64_encode(&abr_v2()[..20])}),
        ] {
            assert!(s.execute("brush.presets.importAbr", p.clone()).is_err(), "{p}");
        }
        assert_eq!(s.tools.presets.len(), n);
    }

    #[test]
    fn grd_import_adds_a_gradient_group() {
        use crate::presets::gradients::StopColor;
        use photocraft_psd::grd::{GrdColor, GrdGradient, GrdStop, write};
        let grads = vec![
            GrdGradient {
                name: "Ember".into(),
                noise: false,
                stops: vec![
                    GrdStop { location: 1.0, midpoint: 0.5, color: GrdColor::Rgb([1.0, 0.0, 0.0]) },
                    GrdStop { location: 0.0, midpoint: 0.5, color: GrdColor::Foreground },
                ],
                opacity: vec![(0.0, 1.0, 0.5), (1.0, 0.0, 0.5)],
            },
            GrdGradient { name: "Static".into(), noise: true, ..Default::default() },
        ];
        let mut s = Session::new();
        let data = photocraft_paint::tile::b64_encode(&write(&grads));
        let r = s.execute("gradient.presets.importGrd", json!({"data": data, "group": "Fire"})).unwrap();
        assert_eq!(r["imported"], json!(["Ember"]));
        assert_eq!(r["warnings"].as_array().map(Vec::len), Some(1));
        let g = s.presets.gradients.iter().find(|g| g.name == "Fire").unwrap();
        assert_eq!(g.items[0].stops, vec![(0.0, StopColor::Foreground), (1.0, StopColor::Rgb([1.0, 0.0, 0.0]))]);
        assert_eq!(g.items[0].opacity, vec![(0.0, 1.0), (1.0, 0.0)]);
        // Re-import replaces the group.
        let n = s.presets.gradients.len();
        s.execute("gradient.presets.importGrd", json!({"data": data, "group": "Fire"})).unwrap();
        assert_eq!(s.presets.gradients.len(), n);
        for p in [json!({}), json!({"data": "@@"}), json!({"data": photocraft_paint::tile::b64_encode(b"8BGR\0\x03")}), json!({"path": "/nope.grd"})] {
            assert!(s.execute("gradient.presets.importGrd", p.clone()).is_err(), "{p}");
        }
        let only_noise = photocraft_paint::tile::b64_encode(&write(&grads[1..]));
        assert!(s.execute("gradient.presets.importGrd", json!({"data": only_noise})).is_err());
    }

    #[test]
    fn colour_conversions() {
        assert_eq!(hsb_to_rgb([120.0, 1.0, 1.0]), [0.0, 1.0, 0.0]);
        let w = lab_to_rgb([100.0, 0.0, 0.0]);
        assert!(w.iter().all(|v| (v - 1.0).abs() < 0.01), "{w:?}");
        let k = lab_to_rgb([0.0, 0.0, 0.0]);
        assert!(k.iter().all(|v| *v < 0.01), "{k:?}");
    }

    #[test]
    fn unique_names() {
        let taken = vec!["A".to_string(), "a 2".to_string()];
        assert_eq!(unique("a", &taken), "a 3");
        assert_eq!(unique("B", &taken), "B");
    }
}
