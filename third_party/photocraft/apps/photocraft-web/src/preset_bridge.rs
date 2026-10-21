//! Synchronous engine boundary for asynchronous browser storage. No browser handles cross the
//! engine's `Send` boundary. Byte buffers are shared with the single in-flight transaction.
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard};

use photocraft_engine::{
    Session,
    preset_store::{self, PresetBackend, *},
};

pub type Batch = BTreeMap<String, Option<Arc<[u8]>>>;
/// Three sampled images per preset (tip, dual tip, texture), plus group/index/action records.
pub const MAX_RECORDS: usize = MAX_GROUPS + MAX_PRESETS_PER_GROUP * 3 + 2;

#[derive(Default)]
struct State {
    files: BTreeMap<String, Arc<[u8]>>,
    bytes: u64,
    dirty: BTreeSet<String>,
    writing: bool,
    error: Option<String>,
    rejected: BTreeMap<String, String>,
    preserve_tips: bool,
}

#[derive(Clone, Default)]
pub struct Bridge(Arc<Mutex<State>>);

pub fn session(backend: Bridge) -> (Session, Vec<String>) {
    let mut session = Session::new();
    let opened = preset_store::open(Box::new(backend.clone()));
    // An unreadable group may reference any tip, including one shared with a good group.
    // Keep all tips until the damaged library can be recovered; unrelated edits still save.
    backend.state().preserve_tips = !opened.warnings.is_empty();
    let warnings = session.attach_preset_store(opened);
    (session, warnings)
}

/// Check metadata before copying a browser record into Rust memory.
pub fn record_limit(name: &str) -> Result<u64, String> {
    let leaf = name.strip_prefix("tips/").unwrap_or(name);
    if leaf.is_empty() || leaf.len() > 128 || leaf.starts_with('.') || !leaf.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.')) {
        return Err("invalid preset record name".into());
    }
    match name {
        INDEX_FILE => Ok(MAX_INDEX_BYTES),
        ACTIONS_FILE => Ok(MAX_ACTIONS_BYTES),
        _ if name.starts_with("tips/") && leaf.ends_with(".pctip") => Ok(MAX_TIP_BYTES),
        _ if !name.contains('/') && name.ends_with(".pcbrushes") => Ok(MAX_GROUP_BYTES),
        _ => Err("unknown preset record type".into()),
    }
}

impl Bridge {
    fn state(&self) -> MutexGuard<'_, State> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn insert(s: &mut State, name: &str, bytes: &[u8]) -> Result<(), String> {
        if bytes.len() as u64 > record_limit(name)? {
            return Err(format!("{name}: preset record too large"));
        }
        let old = s.files.get(name).map_or(0, |b| b.len() as u64);
        let size = s.bytes.saturating_sub(old).saturating_add(bytes.len() as u64);
        if size > MAX_STORE_BYTES || (!s.files.contains_key(name) && s.files.len() >= MAX_RECORDS) {
            return Err("browser preset store is full".into());
        }
        s.files.insert(name.to_string(), Arc::from(bytes));
        s.bytes = size;
        Ok(())
    }

    pub fn load(&self, name: &str, bytes: Vec<u8>) -> Result<(), String> {
        Self::insert(&mut self.state(), name, &bytes)
    }

    pub fn begin(&self) -> Option<Batch> {
        let mut s = self.state();
        if s.writing || s.error.is_some() || s.dirty.is_empty() {
            return None;
        }
        s.writing = true;
        Some(
            std::mem::take(&mut s.dirty)
                .into_iter()
                .map(|name| {
                    let bytes = s.files.get(&name).cloned();
                    (name, bytes)
                })
                .collect(),
        )
    }

    pub fn finish(&self, batch: Batch, result: Result<(), String>) {
        let mut s = self.state();
        s.writing = false;
        if let Err(e) = result {
            // Requeue names, never old bytes: edits/deletes made during the failed write win.
            s.dirty.extend(batch.into_keys());
            s.error = Some(e);
        }
    }

    pub fn unavailable(&self, error: String) {
        self.state().error = Some(error);
    }

    pub fn retry(&self) {
        self.state().error = None;
    }

    pub fn error(&self) -> Option<String> {
        let s = self.state();
        s.rejected.values().next().cloned().or_else(|| s.error.clone())
    }

    pub fn unsaved(&self) -> bool {
        let s = self.state();
        s.writing || !s.dirty.is_empty() || !s.rejected.is_empty()
    }
}

impl PresetBackend for Bridge {
    fn list(&self) -> Result<Vec<(String, u64)>, String> {
        Ok(self.state().files.iter().map(|(k, b)| (k.clone(), b.len() as u64)).collect())
    }

    fn read(&self, name: &str, max: u64) -> Result<Vec<u8>, String> {
        let s = self.state();
        let bytes = s.files.get(name).ok_or_else(|| format!("{name}: not found"))?;
        if bytes.len() as u64 > max {
            return Err(format!("{name}: preset record too large"));
        }
        Ok(bytes.to_vec())
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let mut s = self.state();
        if s.files.get(name).is_some_and(|old| old.as_ref() == bytes) {
            s.rejected.remove(name);
            return Ok(());
        }
        if !s.dirty.contains(name) && s.dirty.len() >= MAX_RECORDS {
            let e = "too many pending preset records".to_string();
            s.rejected.insert(name.to_string(), e.clone());
            return Err(e);
        }
        if let Err(e) = Self::insert(&mut s, name, bytes) {
            s.rejected.insert(name.to_string(), e.clone());
            return Err(e);
        }
        s.dirty.insert(name.to_string());
        // This current replacement resolves only its own omitted edit. Pending transactions
        // still keep unsaved() true; Retry or an older successful batch cannot clear it.
        s.rejected.remove(name);
        Ok(())
    }

    fn remove(&self, name: &str) -> Result<(), String> {
        record_limit(name)?;
        let mut s = self.state();
        if s.preserve_tips && name.starts_with("tips/") {
            s.rejected.remove(name);
            return Ok(());
        }
        if !s.dirty.contains(name) && s.dirty.len() >= MAX_RECORDS {
            let e = "too many pending preset records".to_string();
            s.rejected.insert(name.to_string(), e.clone());
            return Err(e);
        }
        if let Some(bytes) = s.files.remove(name) {
            s.bytes = s.bytes.saturating_sub(bytes.len() as u64);
            s.dirty.insert(name.to_string());
        }
        s.rejected.remove(name);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_psd::abr::{AbrSample, LegacyBrush, LegacyTip, write_v12};
    use serde_json::json;

    fn import(s: &mut Session) {
        let abr = write_v12(
            2,
            &[LegacyBrush {
                name: "Browser seam brush".into(),
                spacing: 25,
                anti_alias: true,
                tip: LegacyTip::Sampled(AbrSample { id: String::new(), width: 6, height: 4, depth: 8, data: vec![255; 24] }),
            }],
            false,
        )
        .unwrap();
        s.execute("brush.presets.importAbr", json!({"data": photocraft_paint::tile::b64_encode(&abr), "group": "Browser samples"})).unwrap();
    }

    #[test]
    fn browser_session_reloads_imported_tip_settings_group_and_order() {
        let backend = Bridge::default();
        let (mut first, _) = session(backend.clone());
        import(&mut first);
        let (second, warnings) = session(backend);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(first.tools.presets, second.tools.presets);
    }
    #[test]
    fn pending_writes_coalesce_and_only_one_transaction_runs() {
        let b = Bridge::default();
        b.write("index.json", b"one").unwrap();
        b.write("index.json", b"two").unwrap();
        let batch = b.begin().unwrap();
        assert_eq!(batch.get("index.json").unwrap().as_deref(), Some(&b"two"[..]));
        b.remove("index.json").unwrap();
        assert!(b.begin().is_none());
        assert!(b.unsaved());
        b.finish(batch, Ok(()));
        let batch = b.begin().unwrap();
        assert!(batch.get("index.json").unwrap().is_none());
        b.finish(batch, Ok(()));
        assert!(!b.unsaved());
    }

    #[test]
    fn failed_transaction_retains_latest_writes_until_explicit_retry() {
        let b = Bridge::default();
        b.write("index.json", b"old").unwrap();
        let batch = b.begin().unwrap();
        b.write("index.json", b"new").unwrap();
        b.finish(batch, Err("QuotaExceededError".into()));
        assert!(b.unsaved());
        assert!(b.error().unwrap().contains("QuotaExceeded"));
        assert!(b.begin().is_none());
        b.retry();
        let batch = b.begin().unwrap();
        assert_eq!(batch.get("index.json").unwrap().as_deref(), Some(&b"new"[..]));
        b.finish(batch, Ok(()));
        assert!(!b.unsaved());
        assert!(b.error().is_none());
    }

    #[test]
    fn browser_deletion_and_reordering_survive_reload() {
        let b = Bridge::default();
        let (mut s, _) = session(b.clone());
        import(&mut s);
        let deleted = s.tools.presets[0].name.clone();
        s.execute("brush.presets.delete", json!({"name": deleted})).unwrap();
        s.execute("brush.presets.moveGroup", json!({"group":"Browser samples", "index":0})).unwrap();
        let persisted = b.begin().unwrap();
        let restored = Bridge::default();
        for (key, value) in &persisted {
            if let Some(bytes) = value {
                restored.load(key, bytes.to_vec()).unwrap();
            }
        }
        b.finish(persisted, Ok(()));
        let (loaded, warnings) = session(restored);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(s.tools.presets, loaded.tools.presets);
        s.execute("brush.presets.delete", json!({"name":"Browser seam brush"})).unwrap();
        let (loaded, _) = session(b);
        assert_eq!(s.tools.presets, loaded.tools.presets);
    }

    #[test]
    fn corrupt_record_is_skipped_with_warning_and_kept_for_recovery() {
        let b = Bridge::default();
        b.load("bad.pcbrushes", b"corrupt".to_vec()).unwrap();
        let (s, warnings) = session(b.clone());
        assert_eq!(s.tools.presets, Session::new().tools.presets);
        assert!(warnings.iter().any(|w| w.contains("bad.pcbrushes")));
        assert_eq!(b.read("bad.pcbrushes", 100).unwrap(), b"corrupt");
    }

    #[test]
    fn rejects_invalid_and_oversized_records_before_copying() {
        assert!(record_limit("../index.json").is_err());
        assert!(record_limit("unknown.bin").is_err());
        assert_eq!(record_limit("index.json").unwrap(), MAX_INDEX_BYTES);
        assert_eq!(record_limit("tips/a.pctip").unwrap(), MAX_TIP_BYTES);
        let b = Bridge::default();
        b.load("index.json", b"initial".to_vec()).unwrap();
        assert!(!b.unsaved(), "hydration is not a user edit");
        assert!(b.read("index.json", 2).is_err());
        assert!(b.write("index.json", &vec![0; MAX_INDEX_BYTES as usize + 1]).is_err());
        assert!(b.error().is_some());
        b.retry();
        assert!(b.error().is_some(), "staging errors cannot be cleared by retrying older writes");
        assert_eq!(b.read("index.json", 100).unwrap(), b"initial");
    }

    #[test]
    fn accepted_replacement_resolves_only_its_own_staging_rejection() {
        let b = Bridge::default();
        b.load("index.json", b"initial".to_vec()).unwrap();
        let oversized = vec![0; MAX_INDEX_BYTES as usize + 1];
        assert!(b.write("index.json", &oversized).is_err());
        b.retry();
        assert!(b.error().is_some(), "retrying older bytes cannot save the rejected edit");
        b.write("index.json", b"repaired").unwrap();
        assert!(b.unsaved(), "accepted replacement still needs its transaction");
        b.finish(b.begin().unwrap(), Ok(()));
        assert!(b.error().is_none());
        assert!(!b.unsaved());
        assert_eq!(b.read("index.json", 100).unwrap(), b"repaired");

        assert!(b.write("index.json", &oversized).is_err());
        assert!(b.write("other.pcbrushes", &vec![0; MAX_GROUP_BYTES as usize + 1]).is_err());
        b.write("index.json", b"repaired").unwrap();
        assert!(b.error().is_some(), "an unchanged accepted record cannot clear another record's rejection");
        assert!(b.unsaved());
        b.remove("other.pcbrushes").unwrap();
        assert!(b.error().is_none());
        assert!(!b.unsaved(), "discarding the rejected absent record resolves its pending edit");
    }

    #[test]
    fn older_successful_transaction_does_not_resolve_an_omitted_edit() {
        let b = Bridge::default();
        b.write("index.json", b"older").unwrap();
        let older = b.begin().unwrap();
        assert!(b.write("index.json", &vec![0; MAX_INDEX_BYTES as usize + 1]).is_err());
        b.finish(older, Ok(()));
        assert!(b.error().is_some());
        assert!(b.unsaved());
        b.write("index.json", b"older").unwrap();
        assert!(b.error().is_none(), "explicitly restoring the already saved bytes resolves the omitted edit");
        assert!(!b.unsaved());
    }

    #[test]
    fn denied_storage_keeps_imports_in_memory_and_marks_them_unsaved() {
        let b = Bridge::default();
        b.unavailable("SecurityError".into());
        let (mut s, _) = session(b.clone());
        assert!(!b.unsaved(), "nothing is lost before the first change, so the web shell shows no banner");
        import(&mut s);
        assert!(s.tools.presets.iter().any(|p| p.name == "Browser seam brush"));
        assert!(b.unsaved());
        assert!(b.error().unwrap().contains("SecurityError"));
        assert!(b.begin().is_none());
    }
    #[test]
    fn corrupt_group_and_shared_tip_survive_unrelated_save_delete_and_reload() {
        let b = Bridge::default();
        let (mut s, _) = session(b.clone());
        import(&mut s);
        let tip = b.list().unwrap().into_iter().find(|(n, _)| n.starts_with("tips/")).unwrap().0;
        b.load("damaged.pcbrushes", b"recoverable broken group".to_vec()).unwrap();
        let (mut s, warnings) = session(b.clone());
        assert!(!warnings.is_empty());
        s.execute("brush.presets.save", json!({"name":"Unrelated brush"})).unwrap();
        s.execute("brush.presets.delete", json!({"name":"Browser seam brush"})).unwrap();
        assert_eq!(b.read("damaged.pcbrushes", 100).unwrap(), b"recoverable broken group");
        assert!(b.read(&tip, MAX_TIP_BYTES).is_ok());
        let (reloaded, warnings) = session(b.clone());
        assert!(!warnings.is_empty());
        assert!(reloaded.tools.presets.iter().any(|p| p.name == "Unrelated brush"));
        assert!(!reloaded.tools.presets.iter().any(|p| p.name == "Browser seam brush"));
        assert_eq!(b.read("damaged.pcbrushes", 100).unwrap(), b"recoverable broken group");
        assert!(b.read(&tip, MAX_TIP_BYTES).is_ok());
    }
    #[test]
    fn indexed_brush_store_coexists_with_imported_gradient_preferences() {
        use photocraft_psd::grd::{GrdColor, GrdGradient, GrdStop, write};
        let b = Bridge::default();
        let (mut s, _) = session(b.clone());
        import(&mut s);
        let data = write(&[GrdGradient {
            name: "Browser gradient".into(),
            noise: false,
            stops: vec![
                GrdStop { location: 0.0, midpoint: 0.5, color: GrdColor::Foreground },
                GrdStop { location: 1.0, midpoint: 0.5, color: GrdColor::Rgb([1.0, 0.0, 0.0]) },
            ],
            opacity: vec![(0.0, 1.0, 0.5), (1.0, 0.5, 0.5)],
        }]);
        s.execute("gradient.presets.importGrd", json!({"data":photocraft_paint::tile::b64_encode(&data),"group":"Browser gradients"})).unwrap();
        let prefs = s.prefs_to_json();
        let (mut loaded, _) = session(b);
        loaded.load_prefs_json(&prefs).unwrap();
        assert_eq!(loaded.tools.presets, s.tools.presets);
        assert_eq!(loaded.presets.gradients, s.presets.gradients);
        assert!(loaded.presets.gradients.iter().any(|g| g.name == "Browser gradients"));
    }
}
