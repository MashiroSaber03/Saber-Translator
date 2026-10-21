//! Rendering policy migration and command validation, without a graphics device.
use photocraft_engine::prefs::{GpuBackend, Preferences, RenderingMode};
use serde_json::json;

#[test]
fn old_preferences_preserve_cpu_choices() {
    for old in [json!({"useGpu": false}), json!({"gpuBackend": "cpu"})] {
        let p: Preferences = serde_json::from_value(json!({"performance": old})).unwrap();
        assert_eq!(p.performance.effective_rendering_mode(), RenderingMode::Cpu);
    }
    assert_eq!(Preferences::default().performance.effective_rendering_mode(), RenderingMode::Auto);
}

#[test]
fn explicit_choices_override_legacy_flags_and_round_trip() {
    let mut p = Preferences::default();
    p.performance.use_gpu = false;
    p.performance.gpu_backend = GpuBackend::Cpu;
    for mode in [RenderingMode::Auto, RenderingMode::Gpu, RenderingMode::Cpu] {
        p.set("performance.renderingMode", json!(mode.name())).unwrap();
        assert_eq!(p.performance.effective_rendering_mode(), mode);
        let restored: Preferences = serde_json::from_value(p.to_json()).unwrap();
        assert_eq!(restored.performance.effective_rendering_mode(), mode);
    }
}

#[test]
fn malformed_policy_is_rejected_without_changing_preferences() {
    let mut p = Preferences::default();
    let before = p.to_json();
    for invalid in [json!("quantum"), json!(7), json!({}), json!(true)] {
        assert!(p.set("performance.renderingMode", invalid).is_err());
        assert_eq!(p.to_json(), before);
    }
    p.reset(Some("performance.renderingMode")).unwrap();
    assert_eq!(p.performance.rendering_mode, None);
}
