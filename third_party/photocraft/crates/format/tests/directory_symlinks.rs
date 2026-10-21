//! Directory cleanup only owns content-addressed objects; unrelated files survive.

mod common;

use common::temp_dir;
use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::Document;
use photocraft_format::{PcraftWriter, SaveOptions};
use photocraft_geom::Size;

fn empty_doc() -> Document {
    Document::new("clean", Size::new(1, 1), ColorMode::Rgb, SampleType::U8)
}

#[test]
fn directory_cleanup_preserves_unrelated_zst_files() {
    let root = temp_dir("unrelated-object");
    let bundle = root.join("project.pcraft");
    let mut writer = PcraftWriter::new();
    writer.save_dir(&empty_doc(), &bundle, &SaveOptions::default()).unwrap();

    let unrelated = bundle.join("tiles/personal-notes.zst");
    let orphan = bundle.join("blobs").join(format!("{}.zst", "b".repeat(64)));
    std::fs::write(&unrelated, b"unrelated file").unwrap();
    std::fs::write(&orphan, b"unused content-addressed object").unwrap();

    let stats = writer.save_dir(&empty_doc(), &bundle, &SaveOptions::default()).unwrap();
    assert_eq!(std::fs::read(&unrelated).unwrap(), b"unrelated file");
    assert!(!orphan.exists(), "a valid but unreferenced object is removed");
    assert_eq!(stats.objects_removed, 1);
    std::fs::remove_dir_all(root).unwrap();
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
mod permissions {
    use super::*;
    use photocraft_format::load_path;
    use photocraft_raster::Rgba8Image;
    use std::os::unix::fs::{PermissionsExt, symlink};
    use std::path::{Path, PathBuf};

    struct RestoreMode {
        path: PathBuf,
        permissions: std::fs::Permissions,
    }

    impl RestoreMode {
        fn set(path: &Path, mode: u32) -> Self {
            let permissions = std::fs::metadata(path).unwrap().permissions();
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
            Self { path: path.to_path_buf(), permissions }
        }
    }

    impl Drop for RestoreMode {
        fn drop(&mut self) {
            let _ = std::fs::set_permissions(&self.path, self.permissions.clone());
        }
    }

    #[test]
    fn directory_save_without_previews_does_not_require_composite_read_permission() {
        let root = temp_dir("composite-search-only");
        let bundle = root.join("project.pcraft");
        let mut writer = PcraftWriter::new();
        writer.save_dir(&empty_doc(), &bundle, &SaveOptions::default()).unwrap();
        let restricted = RestoreMode::set(&bundle.join("composite"), 0o300);
        let mut changed = empty_doc();
        changed.name = "saved without previews".into();
        let result = writer.save_dir(&changed, &bundle, &SaveOptions::default());
        drop(restricted);
        result.expect("saving without previews does not read composite");
        assert_eq!(load_path(&bundle).unwrap(), changed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_save_cleans_stale_previews_through_search_only_root_and_composite() {
        let root = temp_dir("bundle-search-only");
        let bundle = root.join("project.pcraft");
        let link = root.join("linked.pcraft");
        let mut writer = PcraftWriter::new();
        writer.save_dir(&empty_doc(), &bundle, &SaveOptions::default()).unwrap();
        symlink(&bundle, &link).unwrap();
        std::fs::write(bundle.join("thumb.png"), b"stale thumbnail").unwrap();
        std::fs::write(bundle.join("composite/preview.png"), b"stale composite").unwrap();
        let orphan = bundle.join("blobs").join(format!("{}.zst", "a".repeat(64)));
        std::fs::write(&orphan, b"unreferenced object").unwrap();
        let composite_mode = RestoreMode::set(&bundle.join("composite"), 0o300);
        let root_mode = RestoreMode::set(&bundle, 0o300);
        let mut changed = empty_doc();
        changed.name = "saved through linked root".into();
        let result = writer.save_dir(&changed, &link, &SaveOptions::default());
        drop(root_mode);
        drop(composite_mode);
        let stats = result.expect("cleanup needs directory traversal and write permission, not read");
        assert_eq!(stats.objects_removed, 1);
        assert!(!orphan.exists());
        assert!(!bundle.join("thumb.png").exists());
        assert!(!bundle.join("composite/preview.png").exists());
        assert_eq!(load_path(&bundle).unwrap(), changed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn directory_save_writes_current_preview_without_composite_read_permission() {
        let root = temp_dir("preview-search-only");
        let bundle = root.join("project.pcraft");
        let mut writer = PcraftWriter::new();
        writer.save_dir(&empty_doc(), &bundle, &SaveOptions::default()).unwrap();
        let restricted = RestoreMode::set(&bundle.join("composite"), 0o300);
        let mut changed = empty_doc();
        changed.name = "saved with preview".into();
        let opts = SaveOptions { composite: Some(Rgba8Image::new(2, 2)), ..Default::default() };
        let result = writer.save_dir(&changed, &bundle, &opts);
        drop(restricted);
        result.expect("writing a preview does not require listing its directory");
        assert!(std::fs::read(bundle.join("composite/preview.png")).unwrap().starts_with(b"\x89PNG\r\n\x1a\n"));
        assert_eq!(load_path(&bundle).unwrap(), changed);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn denied_stale_preview_cleanup_preserves_the_previous_manifest() {
        let root = temp_dir("denied-preview-cleanup");
        let bundle = root.join("project.pcraft");
        let mut writer = PcraftWriter::new();
        let original = empty_doc();
        writer.save_dir(&original, &bundle, &SaveOptions::default()).unwrap();
        let preview = bundle.join("composite/preview.png");
        std::fs::write(&preview, b"stale composite").unwrap();
        let restricted = RestoreMode::set(&bundle.join("composite"), 0o500);
        let permission_probe = bundle.join("composite/permission-probe");
        if std::fs::write(&permission_probe, b"probe").is_ok() {
            // An elevated test process may bypass directory permissions.
            std::fs::remove_file(permission_probe).unwrap();
            drop(restricted);
            std::fs::remove_dir_all(root).unwrap();
            return;
        }
        let mut changed = empty_doc();
        changed.name = "must not commit".into();
        let result = writer.save_dir(&changed, &bundle, &SaveOptions::default());
        drop(restricted);
        assert!(result.is_err(), "a required pre-commit cleanup failure is reported");
        assert_eq!(std::fs::read(preview).unwrap(), b"stale composite");
        assert_eq!(load_path(&bundle).unwrap(), original);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn denied_object_cleanup_remains_best_effort_after_manifest_commit() {
        let root = temp_dir("denied-object-cleanup");
        let bundle = root.join("project.pcraft");
        let mut writer = PcraftWriter::new();
        writer.save_dir(&empty_doc(), &bundle, &SaveOptions::default()).unwrap();
        let orphan = bundle.join("tiles").join(format!("{}.zst", "d".repeat(64)));
        std::fs::write(&orphan, b"unreferenced object").unwrap();
        let restricted = RestoreMode::set(&bundle.join("tiles"), 0o500);
        let permission_probe = bundle.join("tiles/permission-probe");
        if std::fs::write(&permission_probe, b"probe").is_ok() {
            std::fs::remove_file(permission_probe).unwrap();
            drop(restricted);
            std::fs::remove_dir_all(root).unwrap();
            return;
        }
        let mut changed = empty_doc();
        changed.name = "committed despite housekeeping failure".into();
        let result = writer.save_dir(&changed, &bundle, &SaveOptions::default());
        drop(restricted);
        assert_eq!(result.unwrap().objects_removed, 0);
        assert!(orphan.exists());
        assert_eq!(load_path(&bundle).unwrap(), changed);
        std::fs::remove_dir_all(root).unwrap();
    }
}
