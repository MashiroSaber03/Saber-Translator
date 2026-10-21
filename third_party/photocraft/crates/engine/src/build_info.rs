//! Version and build provenance shared by every frontend (About dialog, `--version`).
//!
//! The version is the workspace version (`[workspace.package] version` in the root
//! `Cargo.toml`, managed by `cargo xtask version`). The commit and build date are optional and
//! come from environment variables at compile time, set by CI and the packaging scripts:
//!
//! - `PHOTOCRAFT_BUILD_SHA`: the git commit (any length; shown shortened to 9 characters)
//! - `PHOTOCRAFT_BUILD_DATE`: the build date, conventionally `YYYY-MM-DD`
//!
//! Without them (a plain `cargo build`) the build reports itself as a dev build. Nothing here
//! shells out to git, so builds work from a source tarball and for wasm.

/// The workspace version, such as `0.1.0` or `0.2.0-rc.1`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The full git commit this build was made from, when the build environment recorded it.
pub fn git_sha() -> Option<&'static str> {
    option_env!("PHOTOCRAFT_BUILD_SHA").map(str::trim).filter(|s| !s.is_empty())
}

/// The build date (`YYYY-MM-DD`), when the build environment recorded it.
pub fn build_date() -> Option<&'static str> {
    option_env!("PHOTOCRAFT_BUILD_DATE").map(str::trim).filter(|s| !s.is_empty())
}

/// The commit shortened for display.
pub fn short_sha() -> Option<&'static str> {
    git_sha().map(|s| &s[..s.len().min(9)])
}

/// `0.1.0 (3f2a9c1d0, 2026-10-01)`, or `0.1.0 (dev build)` without provenance.
pub fn long_version() -> String {
    format_long(VERSION, short_sha(), build_date())
}

fn format_long(version: &str, sha: Option<&str>, date: Option<&str>) -> String {
    match (sha, date) {
        (Some(s), Some(d)) => format!("{version} ({s}, {d})"),
        (Some(s), None) => format!("{version} ({s})"),
        (None, Some(d)) => format!("{version} ({d})"),
        (None, None) => format!("{version} (dev build)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_version_formats() {
        assert_eq!(format_long("1.2.3", Some("abc"), Some("2026-10-01")), "1.2.3 (abc, 2026-10-01)");
        assert_eq!(format_long("1.2.3", Some("abc"), None), "1.2.3 (abc)");
        assert_eq!(format_long("1.2.3", None, Some("2026-10-01")), "1.2.3 (2026-10-01)");
        assert_eq!(format_long("1.2.3-rc.1", None, None), "1.2.3-rc.1 (dev build)");
    }

    #[test]
    fn version_is_the_workspace_version() {
        assert!(!VERSION.is_empty());
        assert!(long_version().starts_with(VERSION));
        if let Some(s) = short_sha() {
            assert!(s.len() <= 9);
        }
    }
}
