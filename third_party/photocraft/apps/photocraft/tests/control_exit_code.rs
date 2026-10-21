//! `--control` setup failures must exit non-zero (#1816): a launcher that sees
//! status 0 believes the control server is running when it is not.
//!
//! These tests run the real binary as a subprocess and assert the exit code,
//! covering both failure paths in `main.rs`: control authentication and the
//! automation workspace.

/// A syntactically valid 64-hex-char token, so the workspace test reaches the
/// workspace check instead of failing on token validation first.
const VALID_TOKEN: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

/// Run `photocraft` with the given arguments in an isolated config dir and
/// return the exit status and stderr. Each call gets its own config dir (tests run in
/// parallel), and no display variables, so a regression fails fast instead of opening a window.
fn run_photocraft(args: &[&str]) -> (std::process::ExitStatus, String) {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("photocraft-control-exit-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_photocraft"))
        .args(args)
        .env("PHOTOCRAFT_CONFIG_DIR", &dir)
        // Isolate from the host environment so the test is reproducible.
        .env_remove("PHOTOCRAFT_CONTROL_PORT")
        .env_remove("PHOTOCRAFT_CONTROL_TOKEN")
        .env_remove("PHOTOCRAFT_CONTROL_TOKEN_FILE")
        .env_remove("PHOTOCRAFT_AUTOMATION_READ_ROOT")
        .env_remove("PHOTOCRAFT_AUTOMATION_WRITE_ROOT")
        .env_remove("DISPLAY")
        .env_remove("WAYLAND_DISPLAY")
        .env_remove("WAYLAND_SOCKET")
        .output()
        .expect("failed to run photocraft binary");
    std::fs::remove_dir_all(&dir).ok();
    (output.status, String::from_utf8_lossy(&output.stderr).into_owned())
}

#[test]
fn invalid_control_token_exits_with_status_2() {
    // `validate_token` rejects a token that is not exactly 64 hex characters.
    let (status, stderr) = run_photocraft(&["--control", "0", "--control-token", "invalid"]);
    assert_eq!(status.code(), Some(2), "invalid control token must exit 2, stderr: {stderr}");
    assert!(stderr.contains("cannot configure control authentication"), "stderr: {stderr}");
}

#[test]
fn both_token_and_token_file_exits_with_status_2() {
    // `server_token` rejects the combination of a supplied token and a token file.
    let dir = std::env::temp_dir().join(format!("photocraft-token-file-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let token_file = dir.join("token.txt");
    std::fs::write(&token_file, VALID_TOKEN).unwrap();
    let (status, stderr) = run_photocraft(&["--control", "0", "--control-token", VALID_TOKEN, "--control-token-file", token_file.to_str().unwrap()]);
    std::fs::remove_dir_all(&dir).ok();
    assert_eq!(status.code(), Some(2), "token + token file must exit 2, stderr: {stderr}");
    assert!(stderr.contains("cannot configure control authentication"), "stderr: {stderr}");
}

#[test]
fn missing_automation_read_root_exits_with_status_2() {
    // `AuthorizedWorkspace::new` fails when the read root does not exist.
    let (status, stderr) = run_photocraft(&["--control", "0", "--control-token", VALID_TOKEN, "--automation-read-root", "/nonexistent/photocraft-test-path"]);
    assert_eq!(status.code(), Some(2), "missing read root must exit 2, stderr: {stderr}");
    assert!(stderr.contains("cannot configure automation workspace"), "stderr: {stderr}");
}

#[test]
fn missing_automation_write_root_exits_with_status_2() {
    // `AuthorizedWorkspace::new` fails when the write root does not exist.
    let (status, stderr) = run_photocraft(&["--control", "0", "--control-token", VALID_TOKEN, "--automation-write-root", "/nonexistent/photocraft-test-path"]);
    assert_eq!(status.code(), Some(2), "missing write root must exit 2, stderr: {stderr}");
    assert!(stderr.contains("cannot configure automation workspace"), "stderr: {stderr}");
}
