use super::*;
use photocraft_psd::abr::{LegacyBrush, write_v6, write_v6_folders, write_v12};
use photocraft_psd::descriptor::{Id, UnicodeString};

fn sample(id: &str, w: u32, h: u32, depth: u16) -> AbrSample {
    let bpp = usize::from(depth / 8);
    let data = (0..w as usize * h as usize).flat_map(|i| std::iter::repeat_n(if i % 2 == 0 { 255u8 } else { 0 }, bpp)).collect();
    AbrSample { id: id.into(), width: w, height: h, depth, data }
}

fn t(s: &str) -> Value {
    Value::Text(UnicodeString::new_nul(s))
}
fn u(unit: &[u8; 4], v: f64) -> Value {
    Value::UnitFloat { unit: *unit, value: v }
}
fn prc(v: f64) -> Value {
    u(b"#Prc", v)
}
fn en(ty: &str, v: &str) -> Value {
    Value::Enumerated { type_id: Id::new(ty), value: Id::new(v) }
}
fn var(control: i32, jitter: f64, minimum: f64) -> Value {
    Value::Descriptor(
        Descriptor::new("brVr").with("bVTy", Value::Integer(control)).with("fStp", Value::Integer(40)).with("jitter", prc(jitter)).with("Mnm ", prc(minimum)),
    )
}

fn sampled_tip(id: &str) -> Descriptor {
    Descriptor::new("sampledBrush")
        .with("Dmtr", u(b"#Pxl", 64.0))
        .with("Angl", u(b"#Ang", 30.0))
        .with("Rndn", prc(80.0))
        .with("Spcn", prc(40.0))
        .with("Intr", Value::Boolean(true))
        .with("flipX", Value::Boolean(true))
        .with("flipY", Value::Boolean(false))
        .with("sampledData", t(id))
}

/// A preset touching every section.
fn full_preset() -> Descriptor {
    Descriptor::new("brushPreset")
        .with("Nm  ", t("Kitchen Sink"))
        .with("Brsh", Value::Descriptor(sampled_tip("$tip-1")))
        .with("useTipDynamics", Value::Boolean(true))
        .with("flipX", Value::Boolean(true))
        .with("flipY", Value::Boolean(false))
        .with("minimumDiameter", prc(25.0))
        .with("minimumRoundness", prc(10.0))
        .with("tiltScale", prc(0.0))
        .with("szVr", var(2, 20.0, 0.0))
        .with("angleDynamics", var(6, 10.0, 0.0))
        .with("roundnessDynamics", var(3, 5.0, 0.0))
        .with("useScatter", Value::Boolean(true))
        .with("Cnt ", Value::Integer(3))
        .with("bothAxes", Value::Boolean(true))
        .with("countDynamics", var(0, 50.0, 0.0))
        .with("scatterDynamics", var(1, 250.0, 0.0))
        .with("useTexture", Value::Boolean(true))
        .with("TxtC", Value::Boolean(true))
        .with("InvT", Value::Boolean(true))
        .with("textureScale", prc(150.0))
        .with("textureBrightness", Value::Integer(-75))
        .with("textureContrast", Value::Integer(25))
        .with("textureBlendMode", en("BlnM", "CBrn"))
        .with("textureDepth", prc(60.0))
        .with("minimumDepth", prc(20.0))
        .with("textureDepthDynamics", var(2, 30.0, 0.0))
        .with("Txtr", Value::Descriptor(Descriptor::new("Ptrn").with("Nm  ", t("Grit")).with("Idnt", t("pat-uuid"))))
        .with(
            "dualBrush",
            Value::Descriptor(
                Descriptor::new("dualBrush")
                    .with("useDualBrush", Value::Boolean(true))
                    .with("Flip", Value::Boolean(true))
                    .with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 12.0)).with("Hrdn", prc(50.0))))
                    .with("BlnM", en("BlnM", "hardMix"))
                    .with("Spcn", prc(30.0))
                    .with("Cnt ", Value::Integer(2))
                    .with("bothAxes", Value::Boolean(true))
                    .with("scatterDynamics", var(0, 120.0, 0.0)),
            ),
        )
        .with("useColorDynamics", Value::Boolean(true))
        .with("clVr", var(2, 40.0, 0.0))
        .with("H   ", prc(10.0))
        .with("Strt", prc(20.0))
        .with("Brgh", prc(30.0))
        .with("purity", prc(-50.0))
        .with("colorDynamicsPerTip", Value::Boolean(false))
        .with("usePaintDynamics", Value::Boolean(true))
        .with("opVr", var(2, 0.0, 10.0))
        .with("prVr", var(1, 15.0, 5.0))
        .with("wtVr", var(0, 0.0, 0.0))
        .with("useBrushPose", Value::Boolean(true))
        .with("brushPoseTiltX", Value::Integer(50))
        .with("brushPoseAngle", Value::Integer(90))
        .with("overridePoseAngle", Value::Boolean(true))
        .with("Wtdg", Value::Boolean(true))
        .with("Nose", Value::Boolean(true))
        .with("Rpt ", Value::Boolean(true))
        .with("someFutureKey", Value::Boolean(true))
        .with(
            "toolOptions",
            Value::Descriptor(
                Descriptor::new("currentToolOptions")
                    .with("Opct", Value::Integer(80))
                    .with("flow", Value::Integer(60))
                    .with("smoothing", Value::Boolean(true))
                    .with("smoothingValue", Value::Integer(40))
                    .with("usePressureOverridesSize", Value::Boolean(true)),
            ),
        )
}

fn grit_pattern() -> PsdPattern {
    PsdPattern {
        mode: 1,
        width: 4,
        height: 2,
        name: "Grit".into(),
        id: "pat-uuid".into(),
        palette: None,
        depth: 8,
        channels: vec![vec![0, 64, 128, 255, 255, 128, 64, 0]],
        alpha: None,
    }
}

#[test]
fn v6_settings_map_onto_brush_sections() {
    for sub in [1, 2] {
        let computed = Descriptor::new("brushPreset")
            .with("Nm  ", t("Round 19"))
            .with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 19.0)).with("Hrdn", prc(0.0)).with("Spcn", prc(10.0))));
        let bytes = write_v6(sub, &[sample("$tip-1", 16, 8, 8)], &[grit_pattern()], &[full_preset(), computed], true).unwrap();
        let imp = read_abr(&bytes, "Test").unwrap();
        assert_eq!(imp.presets.len(), 2);
        let p = &imp.presets[0];
        assert_eq!((p.name.as_str(), p.group.as_str(), p.builtin), ("Kitchen Sink", "Test", false));
        let b = &p.brush;
        let TipShape::Sampled(tile) = &b.tip else { panic!("sampled tip expected") };
        assert_eq!((tile.width, tile.height), (16, 8));
        assert_eq!((tile.data[0], tile.data[1]), (65535, 0));
        assert_eq!((b.size, b.angle, b.roundness, b.spacing, b.flip_x, b.flip_y), (64.0, 30.0, 0.8, 0.4, true, false));
        let sd = &b.shape_dynamics;
        assert!(sd.enabled && sd.flip_x_jitter && !sd.flip_y_jitter);
        assert_eq!((sd.size.control, sd.size.jitter, sd.size.minimum), (Control::PenPressure, 0.2, 0.25));
        assert_eq!((sd.angle.control, sd.roundness.control, sd.roundness.minimum), (Control::Direction, Control::PenTilt, 0.1));
        let sc = &b.scattering;
        assert!(sc.enabled && sc.both_axes);
        assert_eq!((sc.count, sc.scatter.jitter, sc.scatter.control, sc.scatter.fade_steps, sc.count_jitter.jitter), (3, 2.5, Control::Fade, 40, 0.5));
        let tx = &b.texture;
        assert!(tx.enabled && tx.each_tip && tx.invert);
        assert_eq!((tx.scale, tx.brightness, tx.contrast, tx.mode, tx.depth), (1.5, -0.5, 0.25, MaskMode::ColorBurn, 0.6));
        assert_eq!((tx.depth_jitter.jitter, tx.depth_jitter.minimum), (0.3, 0.2));
        let Pattern::Tile(pt) = &tx.pattern else { panic!("embedded pattern expected, got {:?}", tx.pattern) };
        assert_eq!((pt.width, pt.height, pt.data[3]), (4, 2, 65535));
        let db = &b.dual_brush;
        assert!(db.enabled && db.flip && db.both_axes && db.tip == TipShape::Round);
        assert_eq!((db.size, db.hardness, db.mode, db.spacing, db.count, db.scatter), (12.0, 0.5, MaskMode::HardMix, 0.3, 2, 1.2));
        let cd = &b.color_dynamics;
        assert!(cd.enabled && !cd.per_tip);
        assert_eq!((cd.fg_bg.jitter, cd.hue_jitter, cd.saturation_jitter, cd.brightness_jitter, cd.purity), (0.4, 0.1, 0.2, 0.3, -0.5));
        assert!(b.transfer.enabled);
        assert_eq!((b.transfer.opacity.control, b.transfer.opacity.minimum), (Control::PenPressure, 0.1));
        assert_eq!((b.transfer.flow.control, b.transfer.flow.jitter), (Control::Fade, 0.15));
        assert!(b.pose.enabled && b.pose.override_rotation && !b.pose.override_tilt);
        assert_eq!((b.pose.tilt_x, b.pose.rotation), (45.0, 90.0));
        assert!(b.wet_edges && b.noise && b.build_up);
        assert_eq!((b.opacity, b.flow, b.smoothing.amount, b.pressure_size), (0.8, 0.6, 0.4, true));
        assert!(imp.warnings.iter().any(|w| w.contains("someFutureKey")), "{:?}", imp.warnings);
        // The computed preset.
        let r = &imp.presets[1].brush;
        assert_eq!((r.tip.clone(), r.size, r.hardness, r.spacing, r.pressure_size), (TipShape::Round, 19.0, 0.0, 0.1, false));
    }
}

#[test]
fn missing_pattern_and_tip_are_reported() {
    let orphan = Descriptor::new("brushPreset").with("Nm  ", t("Orphan")).with("Brsh", Value::Descriptor(sampled_tip("$nowhere")));
    let bytes = write_v6(2, &[sample("$tip-1", 4, 4, 16), sample("$tip-2", 4, 4, 8)], &[], &[full_preset(), orphan], false).unwrap();
    let imp = read_abr(&bytes, "x").unwrap();
    assert_eq!(imp.presets.len(), 1);
    assert!(matches!(imp.presets[0].brush.texture.pattern, Pattern::Procedural { .. }));
    assert!(imp.warnings.iter().any(|w| w.contains("Grit")), "{:?}", imp.warnings);
    assert!(imp.warnings.iter().any(|w| w.contains("Orphan")), "{:?}", imp.warnings);
}

#[test]
fn disabled_smoothing_tool_options_keep_the_amount_and_mode() {
    let preset = Descriptor::new("brushPreset")
        .with("Nm  ", t("Smoothing off"))
        .with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 19.0))))
        .with(
            "toolOptions",
            Value::Descriptor(
                Descriptor::new("currentToolOptions")
                    .with("smoothing", Value::Boolean(false))
                    .with("smoothingValue", Value::Integer(40))
                    .with("smoothingRadiusMode", Value::Boolean(true)),
            ),
        );
    for sub in [1, 2] {
        let bytes = write_v6(sub, &[], &[], std::slice::from_ref(&preset), true).unwrap();
        let imported = read_abr(&bytes, "Test").unwrap();
        let smoothing = &imported.presets[0].brush.smoothing;
        assert!(!smoothing.enabled && smoothing.amount == 0.4 && smoothing.pulled_string);
    }
}

#[test]
fn file_folders_nest_inside_the_import_group() {
    let named = |n: &str| Descriptor::new("brushPreset").with("Nm  ", t(n));
    let orphan = Descriptor::new("brushPreset").with("Nm  ", t("Orphan")).with("Brsh", Value::Descriptor(sampled_tip("$nowhere")));
    let presets = [named("Top"), named("Pen"), orphan, named("Nib"), named("Chalk")];
    let path = |p: &[&str]| p.iter().map(|s| s.to_string()).collect::<Vec<String>>();
    let folders = [path(&[]), path(&["Inks"]), path(&["Inks"]), path(&["Inks", "Fine"]), path(&["Dry"])];
    let bytes = write_v6_folders(2, &[sample("$a", 4, 4, 8), sample("$b", 4, 4, 8)], &[], &presets, &folders, false).unwrap();
    let imp = read_abr(&bytes, "My Set").unwrap();
    // The skipped preset (missing tip) doesn't shift the others' folders.
    let got: Vec<(&str, &str, String)> = imp.presets.iter().map(|p| (p.name.as_str(), p.group.as_str(), p.folder.join("/"))).collect();
    assert_eq!(
        got,
        [("Top", "My Set", String::new()), ("Pen", "My Set", "Inks".into()), ("Nib", "My Set", "Inks/Fine".into()), ("Chalk", "My Set", "Dry".into())]
    );
    // Without a hierarchy every preset is directly in the group.
    let flat = read_abr(&write_v6(2, &[], &[], &[named("A")], false).unwrap(), "Flat").unwrap();
    assert!(flat.presets.iter().all(|p| p.group == "Flat" && p.folder.is_empty()));
}

#[test]
fn v1_v2_brushes_become_presets() {
    let brushes = vec![
        LegacyBrush { name: String::new(), spacing: 25, anti_alias: true, tip: LegacyTip::Computed { diameter: 13, hardness: 50, angle: 20, roundness: 60 } },
        LegacyBrush { name: "Dune Grass".into(), spacing: 0, anti_alias: false, tip: LegacyTip::Sampled(sample("", 30, 10, 8)) },
    ];
    for version in [1, 2] {
        let imp = read_abr(&write_v12(version, &brushes, true).unwrap(), "Old").unwrap();
        assert_eq!(imp.version, version);
        assert_eq!(imp.presets.len(), 2);
        let c = &imp.presets[0];
        assert_eq!(c.name, "13 px Round");
        assert_eq!((c.brush.size, c.brush.hardness, c.brush.angle, c.brush.roundness, c.brush.spacing), (13.0, 0.5, 20.0, 0.6, 0.25));
        let s = &imp.presets[1];
        assert_eq!(s.name, if version == 2 { "Dune Grass" } else { "Sampled Brush 2" });
        assert_eq!((s.brush.size, s.brush.spacing, s.brush.aliased), (30.0, 0.25, true));
    }
}

#[test]
fn v6_without_settings_lists_tips() {
    let bytes = write_v6(1, &[sample("$a", 8, 8, 8), sample("$b", 3, 9, 16)], &[], &[], true).unwrap();
    // An empty `Brsh` list: presets come from the tips.
    let imp = read_abr(&bytes, "g").unwrap();
    assert_eq!(imp.presets.iter().map(|p| p.brush.size).collect::<Vec<_>>(), vec![8.0, 9.0]);
}

#[test]
fn big_tips_are_downsampled() {
    let s = AbrSample { id: "$big".into(), width: 3000, height: 1500, depth: 8, data: vec![200; 3000 * 1500] };
    let bytes = write_v6(2, &[s], &[], &[], true).unwrap();
    let imp = read_abr(&bytes, "g").unwrap();
    let TipShape::Sampled(t) = &imp.presets[0].brush.tip else { panic!() };
    assert_eq!((t.width, t.height), (MAX_TIP_EDGE, 1250));
    assert_eq!(imp.presets[0].brush.size, 3000.0);
    assert!(imp.warnings.iter().any(|w| w.contains("downsampled")));
}

#[test]
fn tilt_scale_projection_spacing_and_mixer_fields_map() {
    let tip = Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 30.0)).with("Spcn", prc(35.0)).with("Intr", Value::Boolean(false));
    let preset = Descriptor::new("brushPreset")
        .with("Nm  ", t("Wet Pen"))
        .with("Brsh", Value::Descriptor(tip))
        .with("useTipDynamics", Value::Boolean(true))
        .with("szVr", var(3, 0.0, 0.0))
        .with("tiltScale", prc(150.0))
        .with("brushProjection", Value::Boolean(true))
        .with("usePaintDynamics", Value::Boolean(true))
        .with("wtVr", var(2, 30.0, 10.0))
        .with("mxVr", var(1, 45.0, 0.0))
        .with(
            "toolOptions",
            Value::Descriptor(
                Descriptor::new("currentToolOptions")
                    .with("wetness", prc(80.0))
                    .with("dryness", prc(20.0))
                    .with("mix", prc(65.0))
                    .with("flow", prc(70.0))
                    .with("sampleMerged", Value::Boolean(true)),
            ),
        );
    let plain =
        Descriptor::new("brushPreset").with("Nm  ", t("Plain")).with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 9.0))));
    let bytes = write_v6(2, &[], &[], &[preset, plain], true).unwrap();
    let imp = read_abr(&bytes, "Mixer").unwrap();
    let b = &imp.presets[0].brush;
    assert!(!b.spacing_enabled && (b.spacing - 0.35).abs() < 1e-6, "Intr off = Spacing unchecked, the value kept");
    let sd = &b.shape_dynamics;
    assert!(sd.brush_projection && (sd.tilt_scale - 1.5).abs() < 1e-6 && sd.size.control == Control::PenTilt);
    let tr = &b.transfer;
    assert_eq!((tr.wetness.control, tr.wetness.jitter, tr.wetness.minimum), (Control::PenPressure, 0.3, 0.1));
    assert_eq!((tr.mix.control, tr.mix.jitter), (Control::Fade, 0.45));
    let m = &b.mixer;
    assert_eq!((m.wet, m.load, m.mix, m.flow, m.sample_all_layers), (0.8, 0.2, 0.65, 0.7, true));
    assert!(!imp.warnings.iter().any(|w| w.contains("Tilt Scale") || w.contains("Projection") || w.contains("Mixer")), "{:?}", imp.warnings);
    // Defaults when the file says nothing.
    let p = &imp.presets[1].brush;
    assert!(p.spacing_enabled && !p.shape_dynamics.brush_projection && p.shape_dynamics.tilt_scale == 0.0);
    assert_eq!(p.mixer, photocraft_paint::MixerSettings::default());
    assert_eq!((p.transfer.wetness, p.transfer.mix), (Dynamic::default(), Dynamic::default()));
}

#[test]
fn garbage_is_an_error() {
    assert!(read_abr(b"", "g").is_err());
    assert!(read_abr(b"8BPS not a brush", "g").is_err());
    assert!(read_abr(&[0, 6, 0, 2], "g").is_err());
}

#[test]
fn sampled_tip_paints_the_brush_colour_never_black() {
    // An imported tip is coverage: its pixels decide where paint lands, never what colour.
    // A black tip pixel must mean "no paint", and a painted pixel must carry the brush colour,
    // so a regression that treats the tip bitmap as RGB shows up here as black output.
    use photocraft_color::PixelFormat;
    use photocraft_paint::StrokePoint;
    use photocraft_paint::render::render_stroke;
    use photocraft_raster::Surface;

    let preset = Descriptor::new("brushPreset").with("Nm  ", t("Tip")).with("Brsh", Value::Descriptor(sampled_tip("$tip-1")));
    let bytes = write_v6(2, &[sample("$tip-1", 8, 8, 8)], &[], &[preset], false).unwrap();
    let imp = read_abr(&bytes, "Test").unwrap();
    let mut b = imp.presets[0].brush.clone();
    b.pressure_size = false;
    b.color = [0.8, 0.1, 0.1, 1.0];
    let mut s = Surface::new(PixelFormat::RGBA32F);
    render_stroke(&mut s, &b, &[StrokePoint::new(30.0, 32.5, 1.0), StrokePoint::new(34.0, 32.5, 1.0)], None, false, 1.0);
    let mut painted = 0;
    for y in 0..64 {
        for x in 0..64 {
            let p = s.rgba(x, y);
            if p[3] > 0.5 {
                painted += 1;
                assert!((p[0] - 0.8).abs() < 0.05 && (p[1] - 0.1).abs() < 0.05, "not the brush colour at ({x},{y}): {p:?}");
            }
        }
    }
    // The helper writes an alternating 255/0 tip, so both painted and unpainted pixels exist.
    assert!(painted > 0, "the imported tip painted nothing");
    assert!(painted < 64 * 64, "the tip covered everything - coverage was treated as solid colour");
}

/// #3340: Photoshop's texture Contrast runs -50..100; the model maps -50..0 and 0..100 onto
/// -1..0 and 0..1 (as Brush Settings shows it), so +24 must not read back as +48.
#[test]
fn texture_contrast_maps_both_halves() {
    for (c, want) in [(24, 0.24), (100, 1.0), (-30, -0.6), (-50, -1.0), (0, 0.0)] {
        let preset = Descriptor::new("brushPreset")
            .with("Nm  ", t("C"))
            .with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 20.0))))
            .with("useTexture", Value::Boolean(true))
            .with("textureContrast", Value::Integer(c));
        let bytes = write_v6(2, &[], &[], &[preset], false).unwrap();
        let imp = read_abr(&bytes, "Test").unwrap();
        let got = imp.presets[0].brush.texture.contrast;
        assert!((got - want).abs() < 1e-6, "contrast {c}: {got}");
    }
}

/// #3340: a Height-mode brush at a low depth with a darkened pattern (how Photoshop's own
/// "Pen" presets in the reported file are set up) imported but painted nothing.
#[test]
fn low_depth_height_texture_brush_paints() {
    use photocraft_color::PixelFormat;
    use photocraft_paint::StrokePoint;
    use photocraft_paint::render::render_stroke;
    use photocraft_raster::Surface;

    for mode in ["Hght", "linearHeight"] {
        let preset = Descriptor::new("brushPreset")
            .with("Nm  ", t("Pen"))
            .with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 20.0)).with("Hrdn", prc(100.0)).with("Spcn", prc(5.0))))
            .with("useTexture", Value::Boolean(true))
            .with("textureBlendMode", en("BlnM", mode))
            .with("textureDepth", prc(13.0))
            .with("textureBrightness", Value::Integer(-87))
            .with("textureContrast", Value::Integer(5))
            .with("Txtr", Value::Descriptor(Descriptor::new("Ptrn").with("Nm  ", t("Grit")).with("Idnt", t("pat-uuid"))));
        let bytes = write_v6(2, &[], &[grit_pattern()], &[preset], false).unwrap();
        let imp = read_abr(&bytes, "Test").unwrap();
        let b = imp.presets[0].brush.clone();
        let mut s = Surface::new(PixelFormat::RGBA32F);
        render_stroke(&mut s, &b, &[StrokePoint::new(10.0, 20.0, 1.0), StrokePoint::new(50.0, 20.0, 1.0)], None, false, 1.0);
        let covered = (12..48).filter(|&x| s.rgba(x, 20)[3] > 0.5).count();
        // 13 % depth reaches 1.55 pattern ranges: above the whole darkened Grit pattern.
        assert_eq!(covered, 36, "{mode}: the stroke painted {covered} of 36 pixels");
    }
}

/// #4809: Photoshop CC presets store the dual tip's spacing in `dualBrush › Brsh › Spcn` and a
/// `Spcn` of 100 % on the `dualBrush` object beside it; the tip's value is the panel's. A tip
/// without one still takes the outer value.
#[test]
fn dual_brush_spacing_comes_from_its_tip() {
    let dual = |tip: Descriptor| {
        Descriptor::new("brushPreset")
            .with("Nm  ", t("SPACING 10%"))
            .with("Brsh", Value::Descriptor(Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 30.0))))
            .with(
                "dualBrush",
                Value::Descriptor(
                    Descriptor::new("dualBrush")
                        .with("useDualBrush", Value::Boolean(true))
                        .with("Brsh", Value::Descriptor(tip))
                        .with("BlnM", en("BlnM", "Ovrl"))
                        .with("Spcn", prc(100.0)),
                ),
            )
    };
    let spacing = |preset: Descriptor| {
        let bytes = write_v6(2, &[], &[], &[preset], false).unwrap();
        read_abr(&bytes, "Test").unwrap().presets[0].brush.dual_brush.spacing
    };
    let tip = Descriptor::new("computedBrush").with("Dmtr", u(b"#Pxl", 40.0));
    assert!((spacing(dual(tip.clone().with("Spcn", prc(10.0)))) - 0.10).abs() < 1e-6, "the tip's 10 %, not the outer 100 %");
    assert!((spacing(dual(tip)) - 1.0).abs() < 1e-6, "no tip spacing: the outer value");
}
