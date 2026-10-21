//! Working RGB must support reading and editing for the configured intent.

use photocraft_cms::{
    Builtin, Curve, Intent, Profile, ProfileClass,
    pipeline::Stage,
    profile::{Lut, LutKind},
};
use photocraft_color::{Color, ColorMode, SampleType};
use photocraft_doc::{Document, Layer};
use photocraft_engine::Session;
use photocraft_geom::Size;
use serde_json::json;
use std::sync::atomic::{AtomicU64, Ordering};

/// Repository-built, serialized profiles test capabilities without proprietary ICC fixtures
/// or an invented inverse. The synthetic BToA table is not an appearance oracle.
fn directional_profile(source: bool, destination: bool) -> Profile {
    let mut p = Builtin::DisplayP3.profile().clone();
    p.class = ProfileClass::Input;
    p.description = "Synthetic directional RGB".into();
    let curves = p.trc.take().unwrap().to_vec();
    let matrix = p.matrix.take().unwrap();
    let matrix = Stage::Matrix { rows: 3, cols: 3, m: matrix.iter().flatten().map(|v| v * 32768.0 / 65535.0).collect(), offset: vec![0.0; 3] };
    let a2b = Lut { kind: LutKind::Ab, inputs: 3, outputs: 3, stages: vec![Stage::Curves(curves), matrix, Stage::Curves(vec![Curve::Identity; 3])] };
    let b2a = Lut { kind: LutKind::Ab, inputs: 3, outputs: 3, stages: vec![Stage::Curves(vec![Curve::Identity; 3])] };
    p.a2b = if source { [Some(a2b.clone()), Some(a2b.clone()), Some(a2b)] } else { Default::default() };
    p.b2a = if destination { [Some(b2a.clone()), Some(b2a.clone()), Some(b2a)] } else { Default::default() };
    p
}

struct ProfileFile {
    path: std::path::PathBuf,
    profile: Profile,
}

impl ProfileFile {
    fn new(profile: Profile) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!("photocraft-working-rgb-{}-{}.icc", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
        let bytes = profile.with_encoded_bytes().to_bytes();
        let profile = Profile::parse(&bytes).unwrap();
        std::fs::write(&path, &*bytes).unwrap();
        Self { path, profile }
    }
    fn spec(&self) -> &str {
        self.path.to_str().unwrap()
    }
}

impl Drop for ProfileFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn reject_one_way_setting(source: bool, destination: bool, missing_direction: &str) {
    let file = ProfileFile::new(directional_profile(source, destination));
    for intent in Intent::ALL {
        assert_eq!(file.profile.device_to_pcs(intent).is_ok(), source);
        assert_eq!(file.profile.pcs_to_device(intent).is_ok(), destination);
        let mut s = Session::new();
        let before = s.color.settings.clone();
        let error = s.execute("edit.colorSettings", json!({"workingRgb": file.spec(), "intent": intent.id(), "bpc": false})).unwrap_err();
        assert!(error.to_string().contains(missing_direction), "{error}");
        assert_eq!(s.color.settings, before, "reject the full settings update before any mutation");
        assert!(s.execute("prefs.set", json!({"path": "colorSettings.workingRgb", "value": file.spec()})).is_err());
        assert_eq!(s.color.settings, before, "individual preference setter also rejects atomically");
    }
}

#[test]
fn a_source_only_rgb_working_space_is_rejected_atomically() {
    reject_one_way_setting(true, false, "destination");
}

#[test]
fn a_destination_only_rgb_working_space_is_rejected_atomically() {
    reject_one_way_setting(false, true, "source");
}

fn legacy_session(file: &ProfileFile, intent: Intent) -> Session {
    let mut s = Session::new();
    s.load_prefs_json(&json!({"colorSettings": {"workingRgb": file.spec(), "intent": intent.id()}}).to_string()).unwrap();
    assert_eq!(s.color.settings.working_rgb, file.spec(), "legacy preferences still load");
    s
}

fn display_and_copy(s: &mut Session, depth: SampleType) {
    let target = s.active().unwrap().doc.clone();
    let display = s.color.display_transform(&target).expect("the destination document must also be a readable display source");
    let mut displayed = [0.0; 3];
    display.eval(&[0.35, 0.65, 0.45], &mut displayed);
    assert!(displayed.iter().all(|v| v.is_finite()));
    let mut source = Document::new("Copy source", Size::new(8, 6), ColorMode::Rgb, SampleType::F32);
    source.icc_profile = Some(Builtin::AdobeRgbCompat.profile().to_bytes());
    let mut layer = Layer::raster("Incoming", source.pixel_format());
    layer.surface_mut().unwrap().fill_rect(source.bounds(), &[0.35, 0.65, 0.45, 0.75]);
    source.layers.push(layer);
    s.add_document(source, None);
    let source_before = s.documents()[1].doc.clone();
    s.execute("layer.copyToDocument", json!({"document": 0})).expect("an untagged or new document must accept a converted layer");
    assert_eq!(s.documents()[0].doc.depth, depth);
    assert_eq!(s.documents()[0].doc.layers.len(), target.layers.len() + 1);
    assert!(s.documents()[1].doc.as_ref() == source_before.as_ref());
    assert!(s.undo());
    assert!(s.active().unwrap().doc.as_ref() == target.as_ref());
}

fn legacy_untagged_workflow(source: bool, destination: bool) {
    let file = ProfileFile::new(directional_profile(source, destination));
    for intent in Intent::ALL {
        for depth in SampleType::ALL {
            let mut s = legacy_session(&file, intent);
            let doc = Document::with_background("Untagged image", Size::new(8, 6), ColorMode::Rgb, depth, Color::rgba(0.35, 0.65, 0.45, 0.75));
            assert!(doc.icc_profile.is_none());
            s.open_document(doc, None);
            display_and_copy(&mut s, depth);
            assert_eq!(s.color.working(ColorMode::Rgb).content_hash(), Builtin::Srgb.profile().content_hash());
            assert!(s.documents()[0].doc.icc_profile.is_none(), "the default working fallback keeps untagged documents untagged");
        }
    }
}

#[test]
fn legacy_source_only_working_rgb_keeps_untagged_display_and_copy_usable() {
    legacy_untagged_workflow(true, false);
}

#[test]
fn legacy_destination_only_working_rgb_keeps_untagged_display_and_copy_usable() {
    legacy_untagged_workflow(false, true);
}

#[test]
fn legacy_one_way_working_spaces_keep_new_document_controls_usable() {
    for (source, destination) in [(true, false), (false, true)] {
        let file = ProfileFile::new(directional_profile(source, destination));
        for intent in Intent::ALL {
            for depth in SampleType::ALL {
                let mut s = legacy_session(&file, intent);
                let bits = match depth {
                    SampleType::U8 => 8,
                    SampleType::U16 => 16,
                    SampleType::F32 => 32,
                };
                s.execute("file.new", json!({"width": 8, "height": 6, "depth": bits, "background": "transparent"})).unwrap();
                display_and_copy(&mut s, depth);
            }
        }
    }
}

#[test]
fn valid_input_matrix_and_lut_working_profiles_use_the_configured_intent() {
    let mut matrix = Builtin::DisplayP3.profile().clone();
    matrix.class = ProfileClass::Input;
    let mut colorimetric_lut = directional_profile(true, true);
    colorimetric_lut.b2a[0] = None;
    colorimetric_lut.b2a[2] = None;
    for (profile, colorimetric_only) in [(matrix, false), (directional_profile(true, true), false), (colorimetric_lut, true)] {
        let file = ProfileFile::new(profile);
        for intent in Intent::ALL {
            let supported = !colorimetric_only || matches!(intent, Intent::RelativeColorimetric | Intent::AbsoluteColorimetric);
            let mut s = Session::new();
            let before = s.color.settings.clone();
            let result = s.execute("edit.colorSettings", json!({"workingRgb": file.spec(), "intent": intent.id()}));
            assert_eq!(result.is_ok(), supported, "{intent:?}/colorimetric_only={colorimetric_only}");
            if !supported {
                assert_eq!(s.color.settings, before);
            }
            let s = legacy_session(&file, intent);
            let expected = if supported { &file.profile } else { Builtin::Srgb.profile() };
            assert_eq!(s.color.working(ColorMode::Rgb).content_hash(), expected.content_hash());
        }
    }
}

fn intent_lut(intent: Intent) -> ProfileFile {
    let mut profile = directional_profile(true, true);
    let i = match intent {
        Intent::Perceptual => 0,
        Intent::RelativeColorimetric | Intent::AbsoluteColorimetric => 1,
        Intent::Saturation => 2,
    };
    for (index, lut) in profile.b2a.iter_mut().enumerate() {
        if index != i {
            *lut = None;
        }
    }
    ProfileFile::new(profile)
}

fn colorimetric_session(file: &ProfileFile) -> Session {
    let mut s = Session::new();
    s.execute("edit.colorSettings", json!({"workingRgb": file.spec(), "intent": "relative"})).unwrap();
    s
}

#[test]
fn valid_cross_field_profile_and_intent_changes_validate_the_final_batch() {
    let file = intent_lut(Intent::RelativeColorimetric);
    let cases = [
        ("edit.colorSettings", json!({"workingRgb": "display-p3", "intent": "perceptual"})),
        ("prefs.set", json!({"values": {"colorSettings.intent": "perceptual", "colorSettings.workingRgb": "display-p3"}})),
        (
            "prefs.set",
            json!({"path": "colorSettings", "value": {"workingRgb": file.spec(), "intent": "perceptual"}, "values": {"colorSettings.workingRgb": "display-p3"}}),
        ),
        (
            "prefs.set",
            json!({"path": "", "value": {"colorSettings": {"workingRgb": file.spec(), "intent": "perceptual"}}, "values": {"colorSettings.workingRgb": "display-p3"}}),
        ),
    ];
    for (command, params) in cases {
        let mut s = colorimetric_session(&file);
        s.execute(command, params.clone()).unwrap_or_else(|e| panic!("final valid batch must succeed: {command}/{params}: {e}"));
        assert_eq!(s.color.settings.working_rgb, "display-p3");
        assert_eq!(s.color.settings.intent, "perceptual");
    }
}

#[test]
fn invalid_batches_restore_colour_and_unrelated_preferences_for_both_apis() {
    let working = intent_lut(Intent::RelativeColorimetric);
    let invalid = ProfileFile::new(directional_profile(true, false));
    let cases = [
        ("edit.colorSettings", json!({"workingRgb": invalid.spec(), "intent": "perceptual", "bpc": false})),
        (
            "prefs.set",
            json!({"values": {"colorSettings.bpc": false, "colorSettings.intent": "perceptual", "colorSettings.workingRgb": invalid.spec(), "performance.historyStates": 17}}),
        ),
        (
            "prefs.set",
            json!({"values": {"colorSettings.bpc": false, "colorSettings.intent": "perceptual", "colorSettings.workingRgb": "display-p3", "performance.historyStates": 0}}),
        ),
    ];
    for (command, params) in cases {
        let mut s = colorimetric_session(&working);
        let before = s.prefs_to_json();
        assert!(s.execute(command, params).is_err());
        assert!(s.prefs_to_json() == before, "failed batch must preserve both preference stores");
    }
}

#[test]
fn incompatible_single_intent_reset_preserves_the_original_colour_settings() {
    let file = intent_lut(Intent::Saturation);
    let mut s = Session::new();
    s.execute("edit.colorSettings", json!({"workingRgb": file.spec(), "intent": "saturation"})).unwrap();
    let before = s.prefs_to_json();
    assert!(s.execute("prefs.reset", json!({"path": "colorSettings.intent"})).is_err());
    assert!(s.prefs_to_json() == before, "single-setting reset must validate before assignment");
}

#[test]
fn unrelated_preference_batches_still_work_with_legacy_invalid_colour_settings() {
    let invalid = ProfileFile::new(directional_profile(true, false));
    let mut s = legacy_session(&invalid, Intent::RelativeColorimetric);
    let before = s.color.settings.clone();
    s.execute("prefs.set", json!({"values": {"cursors.painting": "precise", "performance.historyStates": 17}})).unwrap();
    assert_eq!(s.color.settings, before);
    assert_eq!(s.execute("prefs.get", json!({"path": "performance.historyStates"})).unwrap(), json!(17));
}
