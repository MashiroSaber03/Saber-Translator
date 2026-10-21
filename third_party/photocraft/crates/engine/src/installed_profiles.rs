//! ICC profiles installed in the operating system, as Photoshop lists them in Print › Printer
//! Profile, View › Proof Setup › Custom and Edit › Convert to Profile: printer and paper profiles
//! a user installs (Windows › Install Profile, macOS ColorSync, Linux `~/.local/share/icc`) appear
//! by their description, and the commands take the file's path.
//!
//! `edit.installedProfiles` lists them for agents; the dialogs call [`installed`].

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use photocraft_cms::{ColorSpace, Profile, ProfileClass};
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// One installed profile.
#[derive(Clone, Debug, PartialEq)]
pub struct InstalledProfile {
    /// The file, as the profile commands take it.
    pub path: String,
    pub description: String,
    pub class: ProfileClass,
    pub color_space: ColorSpace,
}

impl InstalledProfile {
    /// Whether a document can be assigned, converted or proofed to it (device and colour space
    /// profiles in RGB, CMYK or Gray; device links, abstract and named colour profiles are not).
    pub fn is_destination(&self) -> bool {
        matches!(self.class, ProfileClass::Input | ProfileClass::Display | ProfileClass::Output | ProfileClass::ColorSpace)
            && matches!(self.color_space, ColorSpace::Rgb | ColorSpace::Cmyk | ColorSpace::Gray)
    }
}

/// Folders scanned at most this deep (macOS keeps profiles in vendor subfolders).
const MAX_DEPTH: usize = 3;
/// At most this many files are parsed per scan.
const MAX_FILES: usize = 1000;
/// Larger files are not profiles anyone prints with.
const MAX_BYTES: u64 = 32 << 20;

/// The operating system's colour profile folders.
pub fn profile_dirs() -> Vec<PathBuf> {
    let mut v = Vec::new();
    let home = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from);
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").map_or_else(|| PathBuf::from(r"C:\Windows"), PathBuf::from);
        v.push(root.join(r"System32\spool\drivers\color"));
    } else if cfg!(target_os = "macos") {
        v.push(PathBuf::from("/System/Library/ColorSync/Profiles"));
        v.push(PathBuf::from("/Library/ColorSync/Profiles"));
        if let Some(h) = &home {
            v.push(h.join("Library/ColorSync/Profiles"));
        }
    } else {
        v.push(PathBuf::from("/usr/share/color/icc"));
        v.push(PathBuf::from("/usr/local/share/color/icc"));
        if let Some(h) = &home {
            v.push(h.join(".local/share/icc"));
            v.push(h.join(".color/icc"));
        }
    }
    v
}

/// Every profile under `dirs` that parses, sorted by description; unreadable or malformed files
/// are skipped.
pub fn scan(dirs: &[PathBuf]) -> Vec<InstalledProfile> {
    let mut files = Vec::new();
    for d in dirs {
        collect(d, 0, &mut files);
    }
    let mut out: Vec<InstalledProfile> = files.iter().filter_map(|f| read_one(f)).collect();
    out.sort_by(|a, b| a.description.to_lowercase().cmp(&b.description.to_lowercase()).then_with(|| a.path.cmp(&b.path)));
    out.dedup_by(|a, b| a.path == b.path);
    out
}

fn collect(dir: &Path, depth: usize, files: &mut Vec<PathBuf>) {
    if depth > MAX_DEPTH || files.len() >= MAX_FILES {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        if files.len() >= MAX_FILES {
            return;
        }
        let path = e.path();
        let Ok(ft) = e.file_type() else { continue };
        if ft.is_dir() {
            collect(&path, depth + 1, files);
        } else if path.extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case("icc") || x.eq_ignore_ascii_case("icm")) {
            files.push(path);
        }
    }
}

fn read_one(path: &Path) -> Option<InstalledProfile> {
    if std::fs::metadata(path).ok()?.len() > MAX_BYTES {
        return None;
    }
    let bytes = std::fs::read(path).ok()?;
    let p = Profile::parse(&bytes).ok()?;
    let path = path.to_string_lossy().into_owned();
    let description = if p.description.trim().is_empty() {
        Path::new(&path).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.clone())
    } else {
        p.description.trim().to_string()
    };
    Some(InstalledProfile { path, description, class: p.class, color_space: p.color_space })
}

/// How long a scan is reused: dialogs ask every frame, and a newly installed profile shows up
/// within this long.
const FRESH: std::time::Duration = std::time::Duration::from_secs(10);

/// The installed profiles (cached; see [`FRESH`]). Empty on the web.
pub fn installed() -> Vec<InstalledProfile> {
    if cfg!(target_arch = "wasm32") {
        return Vec::new();
    }
    type Cache = Mutex<Option<(std::time::Instant, Vec<InstalledProfile>)>>;
    static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    if let Some((at, list)) = cache.lock().unwrap_or_else(|e| e.into_inner()).as_ref()
        && at.elapsed() < FRESH
    {
        return list.clone();
    }
    let list = scan(&profile_dirs());
    *cache.lock().unwrap_or_else(|e| e.into_inner()) = Some((std::time::Instant::now(), list.clone()));
    list
}

fn class_id(c: ProfileClass) -> &'static str {
    match c {
        ProfileClass::Input => "input",
        ProfileClass::Display => "display",
        ProfileClass::Output => "output",
        ProfileClass::DeviceLink => "deviceLink",
        ProfileClass::ColorSpace => "colorSpace",
        ProfileClass::Abstract => "abstract",
        ProfileClass::NamedColor => "namedColor",
        ProfileClass::Unknown(_) => "unknown",
    }
}

fn space_id(c: ColorSpace) -> String {
    format!("{c:?}").to_lowercase()
}

fn list(_: &mut Session, p: &Value) -> Result<Value> {
    let bad = |msg: String| EngineError::BadParams { cmd: "edit.installedProfiles".into(), msg };
    let class = match p.get("class") {
        None | Some(Value::Null) => "any",
        Some(Value::String(s)) if ["any", "input", "display", "output", "deviceLink", "colorSpace", "abstract", "namedColor"].contains(&s.as_str()) => s,
        Some(v) => return Err(bad(format!("class is any|input|display|output|deviceLink|colorSpace|abstract|namedColor, not {v}"))),
    };
    let space = match p.get("colorSpace") {
        None | Some(Value::Null) => "any",
        Some(Value::String(s)) if ["any", "rgb", "cmyk", "gray"].contains(&s.as_str()) => s,
        Some(v) => return Err(bad(format!("colorSpace is any|rgb|cmyk|gray, not {v}"))),
    };
    let profiles: Vec<Value> = installed()
        .into_iter()
        .filter(|i| class == "any" || class_id(i.class) == class)
        .filter(|i| space == "any" || space_id(i.color_space) == space)
        .map(|i| json!({"path": i.path, "description": i.description, "class": class_id(i.class), "colorSpace": space_id(i.color_space)}))
        .collect();
    let dirs: Vec<String> = profile_dirs().iter().map(|d| d.to_string_lossy().into_owned()).collect();
    Ok(json!({"profiles": profiles, "dirs": dirs}))
}

/// `edit.installedProfiles`.
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "edit.installedProfiles",
        label: "Installed Profiles",
        menu: &[],
        shortcut: None,
        params: r##"{"class":"any|input|display|output|deviceLink|colorSpace|abstract|namedColor"="any","colorSpace":"any|rgb|cmyk|gray"="any"} → {profiles: [{path, description, class, colorSpace}], dirs} (the OS profile folders; pass a path as any command's "profile")"##,
        enabled: |_| Ok(()),
        run: list,
        journal: false,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_cms::Builtin;

    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("pc-installed-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("vendor")).unwrap();
        d
    }

    #[test]
    fn scan_lists_parsable_profiles_by_description_and_skips_junk() {
        let d = temp_dir("scan");
        std::fs::write(d.join("b.icc"), Builtin::Srgb.profile().to_bytes().as_slice()).unwrap();
        std::fs::write(d.join("vendor").join("a.ICM"), Builtin::CoatedCmyk.profile().to_bytes().as_slice()).unwrap();
        std::fs::write(d.join("broken.icc"), b"not a profile").unwrap();
        std::fs::write(d.join("empty.icc"), b"").unwrap();
        std::fs::write(d.join("notes.txt"), b"x").unwrap();
        let found = scan(&[d.clone(), d.join("missing")]);
        let names: Vec<&str> = found.iter().map(|p| p.description.as_str()).collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(names[0].to_lowercase() <= names[1].to_lowercase());
        let cmyk = found.iter().find(|p| p.color_space == ColorSpace::Cmyk).unwrap();
        assert!(cmyk.path.ends_with("a.ICM"));
        assert!(cmyk.is_destination());
        // The listed path is what the profile commands accept.
        assert!(crate::color_cmds::resolve_profile(&cmyk.path, None, None).is_ok());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn scan_of_nothing_is_empty() {
        assert!(scan(&[]).is_empty());
        assert!(scan(&[PathBuf::from("/definitely/not/a/folder")]).is_empty());
    }

    #[test]
    fn command_filters_and_rejects_bad_params() {
        let mut s = Session::new();
        let r = s.execute("edit.installedProfiles", json!({})).unwrap();
        assert!(r["profiles"].is_array() && r["dirs"].is_array());
        let r = s.execute("edit.installedProfiles", json!({"class": "output", "colorSpace": "cmyk"})).unwrap();
        for p in r["profiles"].as_array().unwrap() {
            assert_eq!((p["class"].as_str(), p["colorSpace"].as_str()), (Some("output"), Some("cmyk")));
        }
        for bad in [json!({"class": 3}), json!({"class": "printer"}), json!({"colorSpace": "lab"}), json!({"colorSpace": []})] {
            assert!(s.execute("edit.installedProfiles", bad).is_err());
        }
    }
}
