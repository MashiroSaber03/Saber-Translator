//! A failed Linux desktop launch must not poison the next launch's graphics policy.
#![cfg(target_os = "linux")]

#[test]
fn missing_display_does_not_leave_a_gpu_crash_marker() {
    let dir = std::env::temp_dir().join(format!("photocraft-no-display-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prefs = br#"{"performance":{"gpuBackend":"vulkan","useGpu":true}}"#;
    std::fs::write(dir.join("preferences.json"), prefs).unwrap();

    let error = launch_without_display(&dir);
    assert!(!dir.join("gpu-starting.json").exists(), "a display error is not a GPU crash: {error}");
    assert_eq!(std::fs::read(dir.join("preferences.json")).unwrap(), prefs);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn missing_display_preserves_existing_gpu_crash_evidence() {
    let dir = std::env::temp_dir().join(format!("photocraft-no-display-prior-crash-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let marker = serde_json::json!({
        "backend": "vulkan", "adapter": "Test GPU", "adapterBackend": "vulkan",
        "driver": "test", "version": "test"
    });
    std::fs::write(dir.join("gpu-starting.json"), serde_json::to_vec(&marker).unwrap()).unwrap();
    launch_without_display(&dir);
    let restored: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("gpu-starting.json")).unwrap()).unwrap();
    assert_eq!(restored, marker, "a windowing error must not erase or advance prior GPU crash evidence");
    std::fs::remove_dir_all(&dir).unwrap();
}

fn launch_without_display(dir: &std::path::Path) -> String {
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_photocraft"))
        .env("PHOTOCRAFT_CONFIG_DIR", dir)
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .env_remove("WAYLAND_SOCKET")
        .env_remove("WGPU_BACKEND")
        .env_remove("PHOTOCRAFT_GPU_STARTUP_FAILURE")
        .env_remove("PHOTOCRAFT_CONTROL_PORT")
        .output()
        .unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success(), "a desktop cannot open without a display");
    assert!(error.contains("WAYLAND_DISPLAY") && error.contains("DISPLAY"), "expected a display error, got: {error}");
    error.into_owned()
}
