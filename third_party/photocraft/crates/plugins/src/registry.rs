//! The process-wide set of installed plug-ins.
//!
//! Plug-ins are installed per process (as in Photoshop), not per document: smart filters and
//! filter previews (which run on scratch sessions) must find the same plug-ins.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

#[cfg(not(target_arch = "wasm32"))]
use crate::Error;
use crate::{Limits, Plugin, Result};

static PLUGINS: RwLock<BTreeMap<String, Arc<Plugin>>> = RwLock::new(BTreeMap::new());
static REVISION: AtomicU64 = AtomicU64::new(0);

fn bump() {
    REVISION.fetch_add(1, Ordering::Relaxed);
}

/// Installs `plugin`, replacing one with the same id.
pub fn install(plugin: Plugin) -> Arc<Plugin> {
    let p = Arc::new(plugin);
    PLUGINS.write().unwrap_or_else(|e| e.into_inner()).insert(p.id().to_string(), p.clone());
    bump();
    p
}

/// Loads `bytes` with the default limits and installs the plug-in.
pub fn install_bytes(bytes: &[u8]) -> Result<Arc<Plugin>> {
    Ok(install(Plugin::load(bytes, Limits::default())?))
}

pub fn get(id: &str) -> Option<Arc<Plugin>> {
    PLUGINS.read().unwrap_or_else(|e| e.into_inner()).get(id).cloned()
}

/// Installed plug-ins, sorted by id.
pub fn list() -> Vec<Arc<Plugin>> {
    PLUGINS.read().unwrap_or_else(|e| e.into_inner()).values().cloned().collect()
}

/// Uninstalls `id`; false if it wasn't installed.
pub fn remove(id: &str) -> bool {
    let gone = PLUGINS.write().unwrap_or_else(|e| e.into_inner()).remove(id).is_some();
    if gone {
        bump();
    }
    gone
}

/// Changes whenever a plug-in is installed or removed (for caches such as menus).
pub fn revision() -> u64 {
    REVISION.load(Ordering::Relaxed)
}

/// Reads a `.wasm` file (refusing anything that isn't a regular file or exceeds the module limit)
/// and loads it.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_file(path: &std::path::Path, limits: Limits) -> Result<Plugin> {
    use std::io::Read;
    let meta = std::fs::metadata(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(Error::Io(format!("{} is not a file", path.display())));
    }
    if meta.len() > limits.max_module_bytes as u64 {
        return Err(Error::Module(format!("{} is {} bytes (limit {})", path.display(), meta.len(), limits.max_module_bytes)));
    }
    let f = std::fs::File::open(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let mut bytes = Vec::new();
    f.take(limits.max_module_bytes as u64 + 1).read_to_end(&mut bytes).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    Ok(Plugin::load(&bytes, limits)?.with_source(path.display().to_string()))
}

/// What [`load_folder`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FolderReport {
    /// Ids of the plug-ins installed.
    pub loaded: Vec<String>,
    /// `(file, error)` for each `.wasm` that failed to load.
    pub failed: Vec<(String, String)>,
}

/// Installs every `*.wasm` directly in `dir` (not recursive), in file-name order. A bad module is
/// reported and skipped; the others still load.
#[cfg(not(target_arch = "wasm32"))]
pub fn load_folder(dir: &std::path::Path) -> Result<FolderReport> {
    let rd = std::fs::read_dir(dir).map_err(|e| Error::Io(format!("{}: {e}", dir.display())))?;
    let mut files: Vec<std::path::PathBuf> =
        rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wasm"))).collect();
    files.sort();
    files.truncate(1000);
    let mut report = FolderReport::default();
    for f in files {
        match load_file(&f, Limits::default()) {
            Ok(p) => report.loaded.push(install(p).id().to_string()),
            Err(e) => report.failed.push((f.display().to_string(), e.to_string())),
        }
    }
    Ok(report)
}
