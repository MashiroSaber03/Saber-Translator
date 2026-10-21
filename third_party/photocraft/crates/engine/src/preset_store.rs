//! Persistent brush preset store: user and imported (`.abr`) brush presets survive a restart.
//!
//! Sampled tips would bloat the preferences document, so brush presets live in their own store,
//! a directory next to the preferences (desktop: `<config dir>/Presets`):
//!
//! - `<group>-<hash>.pcbrushes`: one JSON file per preset group (folder) holding the presets'
//!   settings with every bitmap (sampled tip, Dual Brush tip, texture tile) replaced by a
//!   reference to a tip file, plus each tip's size and the tip file of its small preview. A
//!   preset in a nested folder of its group records the folder path (`folder`, #1851);
//! - `tips/<hash>.pctip`: content-addressed bitmaps, deflated, stored at 8 bits per sample when
//!   the tip came from 8-bit data and 16 bits otherwise. Renaming or re-grouping a preset only
//!   rewrites a small JSON file; a tip no group references any more is deleted;
//! - `index.json`: group order, the order of every preset (built-ins included, so drag and drop
//!   in the Brushes panel persists) and the built-in presets the user deleted.
//!
//! Full tips load on demand (#1843): a stored preset's bitmaps are [`StoredTile`]s holding only a
//! preview ([`GrayTile::preview`], at most [`PREVIEW_SIDE`] px), which the Brushes panel and the
//! picker draw. Opening the store reads previews, never a full tip, and once a group is written
//! its presets drop their bitmaps the same way. A tip is decoded when its brush is used
//! ([`Session::load_brush_tips`]: picking a preset, a stroke naming one, exporting), and kept in a
//! byte-capped least-recently-used cache ([`TIP_CACHE_BYTES`]), so memory grows with the brushes
//! in use, not with the size of the library. Stores written before #1843 have no previews: their
//! tips are decoded once at the next start, a few at a time, and the groups rewritten.
//!
//! Built-in presets are never written (they are regenerated on start). The store syncs after
//! every command that changes the brush presets ([`Session::brush_presets_changed`]); only groups
//! whose content changed are rewritten. Every write is atomic (temp file + rename) and bounded
//! ([`MAX_GROUP_BYTES`], [`MAX_TIP_BYTES`], [`MAX_STORE_BYTES`]); corrupt or oversized files are
//! skipped with a warning and left on disk, never a panic.
//!
//! A [`Session`] has no store by default, so headless CLI/MCP sessions and tests never write.
//! The desktop app opens the store on a background thread ([`open`]) and attaches it with
//! [`Session::attach_preset_store`] once loaded, so the first frame never waits for it. The web
//! app hydrates a bounded memory backend from IndexedDB before accepting edits and commits
//! changes asynchronously. Gradient presets, including imported `.grd` groups, are small and persist with
//! the preferences document (`presets.gradients`).
//!
//! The Actions list (`actions.json`, see [`crate::actions_cmds`]) lives in the same directory.
//! A missing file is an empty list. A corrupt or oversized file is skipped with a warning.
//! Headless sessions have no store, so their actions stay in memory.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::Read;
use std::sync::{Arc, Mutex, PoisonError};

use photocraft_paint::tile::PREVIEW_SIDE;
use photocraft_paint::{BrushPreset, BrushSettings, GrayTile, Pattern, PresetCaptures, StoredTile, TipShape};
use serde::{Deserialize, Serialize};

use crate::Session;

/// Group file extension.
pub const GROUP_EXT: &str = "pcbrushes";
/// Tip file extension (inside [`TIPS_DIR`]).
pub const TIP_EXT: &str = "pctip";
/// Subdirectory holding the tip bitmaps.
pub const TIPS_DIR: &str = "tips";
/// Group order and deleted built-ins.
pub const INDEX_FILE: &str = "index.json";
/// Largest group file read or written.
pub const MAX_GROUP_BYTES: u64 = 16 << 20;
/// Largest tip file read or written.
pub const MAX_TIP_BYTES: u64 = 64 << 20;
/// Largest index file read.
pub const MAX_INDEX_BYTES: u64 = 4 << 20;
/// The Actions list (`actions.json`).
pub const ACTIONS_FILE: &str = "actions.json";
/// Largest Actions file read or written.
pub const MAX_ACTIONS_BYTES: u64 = 16 << 20;
/// The whole store never grows beyond this; further groups are skipped with a warning.
pub const MAX_STORE_BYTES: u64 = 2 << 30;
/// Largest tip side accepted from disk.
pub const MAX_TIP_SIDE: u32 = 16384;
/// Largest tip sample count accepted from disk (64 Mi samples).
pub const MAX_TIP_SAMPLES: u64 = 1 << 26;
/// Most group files loaded.
pub const MAX_GROUPS: usize = 4096;
/// Most presets loaded from one group file.
pub const MAX_PRESETS_PER_GROUP: usize = 50_000;

const GROUP_FORMAT: &str = "photocraft-brush-group";
const TIP_MAGIC: &[u8; 6] = b"PCTIP1";
const TIP_HEADER: usize = 6 + 1 + 4 + 4;

/// Where the store's files live. File names are relative (`name.pcbrushes`, `tips/x.pctip`).
pub trait PresetBackend: Send {
    /// Every file in the store with its size in bytes (the top level plus [`TIPS_DIR`]).
    /// A missing store is empty, not an error.
    fn list(&self) -> Result<Vec<(String, u64)>, String>;
    /// Read a file, failing when it is larger than `max` bytes.
    fn read(&self, name: &str, max: u64) -> Result<Vec<u8>, String>;
    /// Replace a file atomically (a crash leaves the old or the new file, never half of one).
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), String>;
    /// Delete a file; deleting a missing file succeeds.
    fn remove(&self, name: &str) -> Result<(), String>;
}

/// Store file names are generated (`[A-Za-z0-9 _.-]`, optionally under `tips/`); anything else
/// (path separators, `..`) is refused so a backend never leaves its directory.
fn valid_name(name: &str) -> bool {
    let leaf = name.strip_prefix("tips/").unwrap_or(name);
    !leaf.is_empty() && leaf.len() <= 128 && !leaf.starts_with('.') && leaf.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-' | '.'))
}

/// An in-memory store (tests, and frontends without a file system). Clones share the files.
#[derive(Clone, Default)]
pub struct MemBackend {
    pub files: std::sync::Arc<std::sync::Mutex<std::collections::BTreeMap<String, Vec<u8>>>>,
}

impl MemBackend {
    fn lock(&self) -> Result<std::sync::MutexGuard<'_, std::collections::BTreeMap<String, Vec<u8>>>, String> {
        self.files.lock().map_err(|_| "preset store lock poisoned".to_string())
    }
}

impl PresetBackend for MemBackend {
    fn list(&self) -> Result<Vec<(String, u64)>, String> {
        Ok(self.lock()?.iter().map(|(k, v)| (k.clone(), v.len() as u64)).collect())
    }
    fn read(&self, name: &str, max: u64) -> Result<Vec<u8>, String> {
        let files = self.lock()?;
        let b = files.get(name).ok_or_else(|| format!("{name}: not found"))?;
        if b.len() as u64 > max {
            return Err(format!("{name}: too large ({} bytes)", b.len()));
        }
        Ok(b.clone())
    }
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        if !valid_name(name) {
            return Err(format!("invalid store file name `{name}`"));
        }
        self.lock()?.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }
    fn remove(&self, name: &str) -> Result<(), String> {
        self.lock()?.remove(name);
        Ok(())
    }
}

/// A store directory on disk (desktop).
#[cfg(not(target_arch = "wasm32"))]
pub struct DirBackend {
    root: std::path::PathBuf,
}

#[cfg(not(target_arch = "wasm32"))]
impl DirBackend {
    pub fn new(root: impl Into<std::path::PathBuf>) -> Self {
        DirBackend { root: root.into() }
    }
    pub fn root(&self) -> &std::path::Path {
        &self.root
    }
    fn path(&self, name: &str) -> Result<std::path::PathBuf, String> {
        if !valid_name(name) {
            return Err(format!("invalid store file name `{name}`"));
        }
        Ok(self.root.join(name))
    }
    fn list_dir(&self, sub: &str, out: &mut Vec<(String, u64)>) -> Result<(), String> {
        let dir = if sub.is_empty() { self.root.clone() } else { self.root.join(sub) };
        let rd = match std::fs::read_dir(&dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(format!("{}: {e}", dir.display())),
        };
        for entry in rd.flatten() {
            let Ok(meta) = entry.metadata() else { continue };
            if !meta.is_file() {
                continue;
            }
            let Some(leaf) = entry.file_name().to_str().map(str::to_string) else { continue };
            let name = if sub.is_empty() { leaf } else { format!("{sub}/{leaf}") };
            if valid_name(&name) {
                out.push((name, meta.len()));
            }
        }
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl PresetBackend for DirBackend {
    fn list(&self) -> Result<Vec<(String, u64)>, String> {
        let mut out = Vec::new();
        self.list_dir("", &mut out)?;
        self.list_dir(TIPS_DIR, &mut out)?;
        Ok(out)
    }
    fn read(&self, name: &str, max: u64) -> Result<Vec<u8>, String> {
        let path = self.path(name)?;
        // A missing file, or a store directory that doesn't exist yet (Windows reports that as
        // "path not found", os error 3), reads as "not found", the wording `missing_file` and the
        // in-memory backend share, so it is an empty file rather than a warning.
        let f = std::fs::File::open(&path)
            .map_err(|e| if e.kind() == std::io::ErrorKind::NotFound { format!("{name}: not found") } else { format!("{name}: {e}") })?;
        let mut out = Vec::new();
        f.take(max.saturating_add(1)).read_to_end(&mut out).map_err(|e| format!("{name}: {e}"))?;
        if out.len() as u64 > max {
            return Err(format!("{name}: larger than {max} bytes"));
        }
        Ok(out)
    }
    fn write(&self, name: &str, bytes: &[u8]) -> Result<(), String> {
        let path = self.path(name)?;
        let dir = path.parent().ok_or_else(|| format!("{name}: no parent directory"))?;
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        photocraft_format::atomic_write(&path, bytes).map_err(|e| format!("{name}: {e}"))
    }
    fn remove(&self, name: &str) -> Result<(), String> {
        match std::fs::remove_file(self.path(name)?) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(format!("{name}: {e}")),
        }
    }
}

// ------------------------------------------------------------------ tips

/// Content hash of a tip (32 hex digits); the tip file's name.
fn tip_hash(t: &GrayTile) -> String {
    let mut h = blake3::Hasher::new();
    h.update(&t.width.to_le_bytes());
    h.update(&t.height.to_le_bytes());
    let mut buf = Vec::with_capacity(8192);
    for chunk in t.data.chunks(4096) {
        buf.clear();
        for v in chunk {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        h.update(&buf);
    }
    h.finalize().to_hex()[..32].to_string()
}

/// `PCTIP1`, bits (8|16), width, height (u32 LE), then the zlib-deflated samples.
pub fn encode_tip(t: &GrayTile) -> Vec<u8> {
    use std::io::Write;
    let eight = t.data.iter().all(|v| v % 257 == 0);
    let mut out = Vec::with_capacity(TIP_HEADER + t.data.len() / 4);
    out.extend_from_slice(TIP_MAGIC);
    out.push(if eight { 8 } else { 16 });
    out.extend_from_slice(&t.width.to_le_bytes());
    out.extend_from_slice(&t.height.to_le_bytes());
    let mut z = flate2::write::ZlibEncoder::new(out, flate2::Compression::fast());
    let mut buf = Vec::with_capacity(8192);
    let mut ok = true;
    for chunk in t.data.chunks(4096) {
        buf.clear();
        if eight {
            buf.extend(chunk.iter().map(|v| (v / 257) as u8));
        } else {
            for v in chunk {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        ok &= z.write_all(&buf).is_ok();
    }
    match z.finish() {
        Ok(v) if ok => v,
        _ => Vec::new(),
    }
}

/// Inverse of [`encode_tip`]; any malformed, truncated or oversized input is an error.
pub fn decode_tip(bytes: &[u8]) -> Result<GrayTile, String> {
    decode_tip_max(bytes, MAX_TIP_SIDE)
}

/// [`decode_tip`] refusing (before inflating anything) a bitmap with a side over `max_side`.
fn decode_tip_max(bytes: &[u8], max_side: u32) -> Result<GrayTile, String> {
    let head = bytes.get(..TIP_HEADER).ok_or("truncated tip header")?;
    if &head[..6] != TIP_MAGIC {
        return Err("not a PhotoCraft tip".into());
    }
    let bits = head[6];
    let u32_at = |i: usize| u32::from_le_bytes([head[i], head[i + 1], head[i + 2], head[i + 3]]);
    let (w, h) = (u32_at(7), u32_at(11));
    if w == 0 || h == 0 || w > MAX_TIP_SIDE.min(max_side) || h > MAX_TIP_SIDE.min(max_side) {
        return Err(format!("bad tip size {w}×{h}"));
    }
    let n = u64::from(w) * u64::from(h);
    if n > MAX_TIP_SAMPLES {
        return Err(format!("tip too large ({w}×{h})"));
    }
    let bytes_per = match bits {
        8 => 1u64,
        16 => 2,
        b => return Err(format!("unsupported tip depth {b}")),
    };
    let want = n * bytes_per;
    let mut raw = Vec::with_capacity(want as usize);
    flate2::read::ZlibDecoder::new(&bytes[TIP_HEADER..]).take(want + 1).read_to_end(&mut raw).map_err(|e| format!("tip data: {e}"))?;
    if raw.len() as u64 != want {
        return Err(format!("tip data has {} bytes, expected {want}", raw.len()));
    }
    let data: Vec<u16> =
        if bits == 8 { raw.iter().map(|&v| u16::from(v) * 257).collect() } else { raw.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes(*c)).collect() };
    Ok(GrayTile { width: w, height: h, data: data.into() })
}

// ------------------------------------------------------------------ group files

/// References from a stored preset to its tip files.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct TipRefs {
    #[serde(skip_serializing_if = "Option::is_none")]
    tip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dual_tip: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    texture: Option<String>,
}

impl TipRefs {
    fn hashes(&self) -> impl Iterator<Item = &String> {
        [&self.tip, &self.dual_tip, &self.texture].into_iter().flatten()
    }
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct StoredPreset {
    name: String,
    brush: BrushSettings,
    #[serde(default)]
    tips: TipRefs,
    /// Nested folders inside the group ([`BrushPreset::folder`]); missing in stores written
    /// before #1851, which load with every preset directly in its group.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    folder: Vec<String>,
    /// [`BrushPreset::captures`]; missing for presets saved without the New Brush Preset dialog.
    #[serde(default, skip_serializing_if = "PresetCaptures::is_default")]
    captures: PresetCaptures,
}

/// What a group file records about each tip it references, so loading needs only the preview.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TipInfo {
    width: u32,
    height: u32,
    /// Hash of the preview's tip file ([`GrayTile::preview`], at most [`PREVIEW_SIDE`] px).
    preview: String,
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GroupFile {
    format: String,
    version: u32,
    group: String,
    presets: Vec<StoredPreset>,
    /// Tip hash → size and preview. Missing in stores written before #1843: those tips are
    /// decoded once at the next start to make their previews, then the group is rewritten.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    tips: BTreeMap<String, TipInfo>,
}

#[derive(Default, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct IndexFile {
    version: u32,
    /// Group files in display order.
    groups: Vec<String>,
    /// Built-in presets the user deleted (by name).
    hidden_builtins: Vec<String>,
    /// Every preset's name in library (panel) order; empty when it is the default order.
    order: Vec<String>,
}

/// The group's file name: a readable, sanitised prefix plus a hash of the exact name, so
/// different names never share a file.
pub fn group_file_name(group: &str) -> String {
    let mut stem: String = group.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, ' ' | '_' | '-') { c } else { '_' }).take(48).collect();
    stem = stem.trim().to_string();
    if stem.is_empty() {
        stem = "Ungrouped".into();
    }
    let h = blake3::hash(group.as_bytes()).to_hex();
    format!("{stem}-{}.{GROUP_EXT}", &h[..8])
}

/// A preset bitmap as the settings hold it: embedded, or already a store reference.
enum Bitmap<'a> {
    Full(&'a GrayTile),
    Stored(&'a StoredTile),
}

/// The bitmaps of a preset's settings: sampled tip, Dual Brush tip, texture tile.
fn bitmaps(b: &BrushSettings) -> [Option<Bitmap<'_>>; 3] {
    fn tip(t: &TipShape) -> Option<Bitmap<'_>> {
        match t {
            TipShape::Round => None,
            TipShape::Sampled(g) => Some(Bitmap::Full(g)),
            TipShape::Stored(r) => Some(Bitmap::Stored(r)),
        }
    }
    let texture = match &b.texture.pattern {
        Pattern::Procedural { .. } => None,
        Pattern::Tile(g) => Some(Bitmap::Full(g)),
        Pattern::Stored(r) => Some(Bitmap::Stored(r)),
    };
    [tip(&b.tip), tip(&b.dual_brush.tip), texture]
}

/// A copy of `b` without its bitmaps (stored as tip files): no bitmap is cloned.
fn settings_only(b: &mut BrushSettings) -> BrushSettings {
    let tip = std::mem::take(&mut b.tip);
    let dual = std::mem::take(&mut b.dual_brush.tip);
    let texture = match b.texture.pattern {
        Pattern::Procedural { .. } => None,
        _ => Some(std::mem::take(&mut b.texture.pattern)),
    };
    let out = b.clone();
    b.tip = tip;
    b.dual_brush.tip = dual;
    if let Some(t) = texture {
        b.texture.pattern = t;
    }
    out
}

/// Point `b`'s bitmaps at stored tiles (`refs` in [`bitmaps`] order); a `None` leaves a slot.
fn set_stored(b: &mut BrushSettings, refs: [Option<StoredTile>; 3]) {
    let [tip, dual, texture] = refs;
    if let Some(r) = tip {
        b.tip = TipShape::Stored(r);
    }
    if let Some(r) = dual {
        b.dual_brush.tip = TipShape::Stored(r);
    }
    if let Some(r) = texture {
        b.texture.pattern = Pattern::Stored(r);
    }
}

/// Settings read from a group file with their tip references resolved: `None` when a referenced
/// tip is missing.
fn join_tips(mut b: BrushSettings, refs: &TipRefs, tiles: &HashMap<String, StoredTile>) -> Option<BrushSettings> {
    let get = |h: &Option<String>| -> Option<Option<StoredTile>> {
        match h {
            None => Some(None),
            Some(h) => tiles.get(h).cloned().map(Some),
        }
    };
    let r = [get(&refs.tip)?, get(&refs.dual_tip)?, get(&refs.texture)?];
    set_stored(&mut b, r);
    Some(b)
}

/// A folder path read from disk, bounded like the commands' (at most
/// [`crate::brush_preset_cmds::MAX_FOLDER_DEPTH`] non-empty names): a hand-edited file can't
/// build an absurd tree.
fn sanitize_folder(folder: Vec<String>) -> Vec<String> {
    folder.into_iter().map(|f| f.trim().to_string()).filter(|f| !f.is_empty()).take(crate::brush_preset_cmds::MAX_FOLDER_DEPTH).collect()
}

fn tip_file(hash: &str) -> String {
    format!("{TIPS_DIR}/{hash}.{TIP_EXT}")
}

/// What a group looked like when it was last written (or loaded).
struct Synced {
    presets: Vec<BrushPreset>,
    /// Tip files the group uses (full tips and previews).
    tips: Vec<String>,
    bytes: u64,
    /// Loaded from a file that lacks tip previews: rewrite it at the next sync.
    stale: bool,
}

/// The default byte budget of decoded full-size tips kept for reuse ([`PresetStore::load_tile`]).
pub const TIP_CACHE_BYTES: u64 = 256 << 20;

/// Full-size tips decoded on demand, least recently used evicted first once over budget.
#[derive(Default)]
struct TipCache {
    limit: u64,
    tick: u64,
    bytes: u64,
    decodes: u64,
    entries: HashMap<String, (Arc<GrayTile>, u64)>,
}

impl TipCache {
    fn get(&mut self, key: &str) -> Option<Arc<GrayTile>> {
        self.tick += 1;
        let tick = self.tick;
        self.entries.get_mut(key).map(|(t, used)| {
            *used = tick;
            t.clone()
        })
    }
    fn insert(&mut self, key: &str, t: Arc<GrayTile>) {
        self.tick += 1;
        self.bytes += tile_bytes(&t);
        if let Some((old, _)) = self.entries.insert(key.to_string(), (t, self.tick)) {
            self.bytes = self.bytes.saturating_sub(tile_bytes(&old));
        }
        self.evict(key);
    }
    /// Drop least recently used tips until within the limit (never `keep`, the one just used).
    fn evict(&mut self, keep: &str) {
        while self.bytes > self.limit {
            let Some(oldest) = self.entries.iter().filter(|(k, _)| k.as_str() != keep).min_by_key(|(_, (_, used))| *used).map(|(k, _)| k.clone()) else {
                break;
            };
            if let Some((t, _)) = self.entries.remove(&oldest) {
                self.bytes = self.bytes.saturating_sub(tile_bytes(&t));
            }
        }
    }
}

fn tile_bytes(t: &GrayTile) -> u64 {
    t.data.len() as u64 * 2
}

/// The decoded-tip cache's state ([`PresetStore::tip_cache`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TipCacheStats {
    /// Tips held.
    pub tips: usize,
    /// Bytes of the tips held.
    pub bytes: u64,
    /// Full tips decoded since the store opened (cache misses).
    pub decodes: u64,
    /// The byte budget.
    pub limit: u64,
}

/// The persistent brush preset store attached to a session (see the module docs).
pub struct PresetStore {
    backend: Box<dyn PresetBackend>,
    /// `tools.presets_rev` at the last sync.
    synced_rev: Option<u64>,
    /// Group file name → its last written content.
    groups: HashMap<String, Synced>,
    /// Group files that failed to load: left on disk untouched, so a sync never writes them.
    skipped: HashSet<String>,
    /// Tip hash → file size, for every tip file the store owns.
    tips: HashMap<String, u64>,
    index: Option<IndexFile>,
    /// [`crate::actions_cmds::ActionState::rev`] last written (or loaded).
    actions_rev: u64,
    /// `actions.json` exists but failed to load: left on disk untouched, so it is never written.
    actions_skipped: bool,
    warnings: Vec<String>,
    /// Full tips loaded for brushes in use (behind a lock so `&Session` paths can load).
    cache: Mutex<TipCache>,
}

/// A loaded store: the presets on disk plus what the session should hide.
pub struct Opened {
    pub store: PresetStore,
    /// User and imported presets, in group order.
    pub presets: Vec<BrushPreset>,
    /// Built-in presets the user deleted.
    pub hidden_builtins: Vec<String>,
    /// The saved library order (preset names), empty when none was saved.
    pub order: Vec<String>,
    /// Files that were skipped (corrupt, oversized, missing tips).
    pub warnings: Vec<String>,
    /// Recorded actions (`actions.json`). Empty when the file is missing or unreadable.
    pub actions: Vec<crate::actions_cmds::Action>,
}

/// Load a store (any thread; the desktop app does this in the background at start).
pub fn open(backend: Box<dyn PresetBackend>) -> Opened {
    let mut warnings = Vec::new();
    let files = backend.list().unwrap_or_else(|e| {
        warnings.push(format!("brush presets: {e}"));
        Vec::new()
    });
    let index: Option<IndexFile> = if files.iter().any(|(n, _)| n == INDEX_FILE) {
        match backend.read(INDEX_FILE, MAX_INDEX_BYTES).and_then(|b| serde_json::from_slice(&b).map_err(|e| e.to_string())) {
            Ok(i) => Some(i),
            Err(e) => {
                warnings.push(format!("brush presets: {INDEX_FILE} skipped: {e}"));
                None
            }
        }
    } else {
        None
    };
    // Group files in index order, then the rest by name.
    let mut group_files: Vec<(String, u64)> = files.iter().filter(|(n, _)| !n.contains('/') && n.ends_with(&format!(".{GROUP_EXT}"))).cloned().collect();
    group_files.sort_by(|a, b| a.0.cmp(&b.0));
    if let Some(ix) = &index {
        let pos = |n: &str| ix.groups.iter().position(|g| g == n).unwrap_or(usize::MAX);
        group_files.sort_by_key(|(n, _)| pos(n));
    }
    if group_files.len() > MAX_GROUPS {
        warnings.push(format!("brush presets: only the first {MAX_GROUPS} of {} groups were loaded", group_files.len()));
        group_files.truncate(MAX_GROUPS);
    }
    let tip_sizes: HashMap<String, u64> =
        files.iter().filter_map(|(n, sz)| Some((n.strip_prefix(&format!("{TIPS_DIR}/"))?.strip_suffix(&format!(".{TIP_EXT}"))?.to_string(), *sz))).collect();

    // Parse the group files.
    let mut parsed: Vec<(String, u64, GroupFile)> = Vec::new();
    let mut skipped = HashSet::new();
    let mut failed = false;
    for (name, size) in group_files {
        let r = backend.read(&name, MAX_GROUP_BYTES).and_then(|b| serde_json::from_slice::<GroupFile>(&b).map_err(|e| e.to_string())).and_then(|g| {
            if g.format != GROUP_FORMAT {
                Err(format!("unknown format `{}`", g.format))
            } else if g.version != 1 {
                Err(format!("unsupported version {}", g.version))
            } else if g.presets.len() > MAX_PRESETS_PER_GROUP {
                Err(format!("{} presets (limit {MAX_PRESETS_PER_GROUP})", g.presets.len()))
            } else {
                Ok(g)
            }
        });
        match r {
            Ok(g) => parsed.push((name, size, g)),
            Err(e) => {
                failed = true;
                warnings.push(format!("brush presets: {name} skipped: {e}"));
                skipped.insert(name);
            }
        }
    }
    // Every referenced tip as a stored tile: its size and preview come from the group files, so
    // no full tip is decoded here (they load when a brush is used, `PresetStore::load_tile`).
    let wanted: Vec<String> = {
        let mut seen = HashSet::new();
        parsed.iter().flat_map(|(_, _, g)| g.presets.iter().flat_map(|p| p.tips.hashes())).filter(|h| seen.insert(h.as_str())).cloned().collect()
    };
    let mut info: HashMap<String, TipInfo> = HashMap::new();
    for (_, _, g) in &parsed {
        for (h, i) in &g.tips {
            info.entry(h.clone()).or_insert_with(|| i.clone());
        }
    }
    let (tiles, regenerated) = load_tiles(backend.as_ref(), &wanted, &tip_sizes, &info, &mut warnings);

    let loaded_actions = load_actions(backend.as_ref(), &mut warnings);
    let actions_skipped = loaded_actions.is_none();
    let actions = loaded_actions.unwrap_or_default();
    let mut store = PresetStore {
        backend,
        synced_rev: None,
        groups: HashMap::new(),
        skipped: HashSet::new(),
        tips: HashMap::new(),
        index: None,
        actions_rev: 0,
        actions_skipped,
        warnings: Vec::new(),
        cache: Mutex::new(TipCache { limit: TIP_CACHE_BYTES, ..Default::default() }),
    };
    let mut presets = Vec::new();
    for (file, size, g) in parsed {
        let mut items = Vec::with_capacity(g.presets.len());
        let mut hashes = Vec::new();
        let mut missing = 0;
        let mut stale = false;
        for sp in g.presets {
            match join_tips(sp.brush, &sp.tips, &tiles) {
                Some(brush) => {
                    for h in sp.tips.hashes() {
                        stale |= regenerated.contains(h) || !g.tips.contains_key(h);
                        hashes.push(h.clone());
                        if let Some(i) = info.get(h).filter(|i| tip_sizes.contains_key(&i.preview)) {
                            hashes.push(i.preview.clone());
                        }
                    }
                    items.push(BrushPreset {
                        name: sp.name,
                        brush,
                        builtin: false,
                        group: g.group.clone(),
                        folder: sanitize_folder(sp.folder),
                        captures: sp.captures,
                    });
                }
                None => missing += 1,
            }
        }
        if missing > 0 {
            // Left on disk untouched (a sync never deletes a file it did not load).
            failed = true;
            warnings.push(format!("brush presets: {file} skipped: {missing} preset(s) have missing or unreadable tips"));
            skipped.insert(file);
            continue;
        }
        for h in &hashes {
            if let Some(sz) = tip_sizes.get(h) {
                store.tips.insert(h.clone(), *sz);
            }
        }
        presets.extend(items.iter().cloned());
        store.groups.insert(file, Synced { presets: items, tips: hashes, bytes: size, stale });
    }
    store.skipped = skipped;
    // Tip files nothing references are garbage (e.g. left by a crash), unless a group failed to
    // load: its tips must survive so a fixed file still works.
    if !failed {
        for (h, sz) in &tip_sizes {
            store.tips.entry(h.clone()).or_insert(*sz);
        }
    }
    let hidden_builtins = index.as_ref().map(|i| i.hidden_builtins.clone()).unwrap_or_default();
    let order = index.as_ref().map(|i| i.order.clone()).unwrap_or_default();
    store.index = index;
    Opened { store, presets, hidden_builtins, order, warnings, actions }
}

/// `actions.json`: a missing file is an empty list. Anything else unreadable is a warning.
/// The saved actions; `None` when `actions.json` exists but could not be loaded.
fn load_actions(backend: &dyn PresetBackend, warnings: &mut Vec<String>) -> Option<Vec<crate::actions_cmds::Action>> {
    let bytes = match backend.read(ACTIONS_FILE, MAX_ACTIONS_BYTES) {
        Ok(b) => b,
        Err(e) if missing_file(&e) => return Some(Vec::new()),
        Err(e) => {
            warnings.push(format!("actions: {ACTIONS_FILE} skipped: {e}"));
            return None;
        }
    };
    match serde_json::from_slice::<crate::actions_cmds::ActionsFile>(&bytes) {
        Ok(file) if file.version == 1 => Some(file.actions),
        Ok(file) => {
            warnings.push(format!("actions: {ACTIONS_FILE} skipped: unsupported version {}", file.version));
            None
        }
        Err(e) => {
            warnings.push(format!("actions: {ACTIONS_FILE} skipped: {e}"));
            None
        }
    }
}

fn missing_file(err: &str) -> bool {
    let lower = err.to_ascii_lowercase();
    lower.contains("not found") || lower.contains("no such file") || lower.contains("os error 2")
}

/// Largest preview tip file read.
const MAX_PREVIEW_BYTES: u64 = 1 << 20;
/// Tips decoded at a time when making missing previews (bounds the memory of that one-off pass).
const PREVIEW_BATCH: usize = 8;

/// The stored tiles for `hashes` (tip files that are missing or unreadable become warnings), and
/// the hashes whose preview had to be made from the full tip: stores written before #1843 have
/// none, so their tips are decoded once, a few at a time, and the groups rewritten at the next sync.
fn load_tiles(
    backend: &dyn PresetBackend,
    hashes: &[String],
    sizes: &HashMap<String, u64>,
    info: &HashMap<String, TipInfo>,
    warnings: &mut Vec<String>,
) -> (HashMap<String, StoredTile>, HashSet<String>) {
    let mut tiles = HashMap::with_capacity(hashes.len());
    let mut todo = Vec::new();
    let sane = |w: u32, h: u32| w > 0 && h > 0 && w <= MAX_TIP_SIDE && h <= MAX_TIP_SIDE && u64::from(w) * u64::from(h) <= MAX_TIP_SAMPLES;
    for h in hashes {
        if !sizes.contains_key(h) {
            warnings.push(format!("brush presets: tip {h} skipped: file missing"));
            continue;
        }
        let preview = info.get(h).filter(|i| sane(i.width, i.height) && sizes.contains_key(&i.preview)).and_then(|i| {
            let bytes = backend.read(&tip_file(&i.preview), MAX_PREVIEW_BYTES).ok()?;
            Some((i, decode_tip_max(&bytes, PREVIEW_SIDE).ok()?))
        });
        match preview {
            Some((i, p)) => {
                tiles.insert(h.clone(), StoredTile { key: h.clone(), width: i.width, height: i.height, preview: Arc::new(p), full: None });
            }
            None => todo.push(h.clone()),
        }
    }
    let mut regenerated = HashSet::new();
    for batch in todo.chunks(PREVIEW_BATCH) {
        // Reads stay sequential (the backend need not be Sync); decoding is the expensive part.
        let raw: Vec<Result<Vec<u8>, String>> = batch.iter().map(|h| backend.read(&tip_file(h), MAX_TIP_BYTES)).collect();
        let make = |r: &Result<Vec<u8>, String>| -> Result<(u32, u32, GrayTile), String> {
            let t = decode_tip(r.as_ref().map_err(Clone::clone)?)?;
            Ok((t.width, t.height, t.preview(PREVIEW_SIDE)))
        };
        #[cfg(not(target_arch = "wasm32"))]
        let made: Vec<_> = {
            use rayon::prelude::*;
            raw.par_iter().map(make).collect()
        };
        #[cfg(target_arch = "wasm32")]
        let made: Vec<_> = raw.iter().map(make).collect();
        for (h, r) in batch.iter().zip(made) {
            match r {
                Ok((w, ht, p)) => {
                    tiles.insert(h.clone(), StoredTile { key: h.clone(), width: w, height: ht, preview: Arc::new(p), full: None });
                    regenerated.insert(h.clone());
                }
                Err(e) => warnings.push(format!("brush presets: tip {h} skipped: {e}")),
            }
        }
    }
    (tiles, regenerated)
}

/// Open a store directory (desktop). Never fails: problems become warnings.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_dir(root: impl Into<std::path::PathBuf>) -> Opened {
    open(Box::new(DirBackend::new(root)))
}

/// Open a store directory on a background thread; the receiver yields it once loaded.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_dir_async(root: std::path::PathBuf) -> std::sync::mpsc::Receiver<Opened> {
    let (tx, rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().name("brush-presets".into()).spawn(move || {
        let _ = tx.send(open_dir(root));
    });
    if let Err(e) = spawned {
        // No thread: the caller simply never gets a store (presets stay session-only).
        eprintln!("brush presets: {e}");
    }
    rx
}

impl PresetStore {
    /// Warnings from writes since the last call (the shell shows them in the status bar).
    pub fn take_warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }

    /// Total bytes of the files the store owns.
    pub fn bytes(&self) -> u64 {
        self.groups.values().map(|g| g.bytes).sum::<u64>() + self.tips.values().sum::<u64>()
    }

    /// The decoded-tip cache: tips held, their bytes, decodes so far and the budget.
    pub fn tip_cache(&self) -> TipCacheStats {
        let c = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        TipCacheStats { tips: c.entries.len(), bytes: c.bytes, decodes: c.decodes, limit: c.limit }
    }

    /// Change the decoded-tip cache's byte budget (evicting down to it at once).
    pub fn set_tip_cache_limit(&self, bytes: u64) {
        let mut c = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        c.limit = bytes;
        c.evict("");
    }

    /// The full bitmap of a stored tile: the tile's own when loaded, else from the cache, else
    /// read and decoded from its tip file (then cached; least recently used tips are dropped once
    /// the cache is over budget). A missing, corrupt or mismatched tip file is an error.
    pub fn load_tile(&self, t: &StoredTile) -> Result<Arc<GrayTile>, String> {
        if let Some(f) = &t.full {
            return Ok(f.clone());
        }
        let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(f) = cache.get(&t.key) {
            return Ok(f);
        }
        if !self.tips.contains_key(&t.key) {
            return Err(format!("brush tip {} is not in the preset store", t.key));
        }
        let bytes = self.backend.read(&tip_file(&t.key), MAX_TIP_BYTES)?;
        let full = decode_tip(&bytes).map_err(|e| format!("brush tip {}: {e}", t.key))?;
        if (full.width, full.height) != (t.width, t.height) {
            return Err(format!("brush tip {} is {}×{} px, expected {}×{}", t.key, full.width, full.height, t.width, t.height));
        }
        let full = Arc::new(full);
        cache.decodes += 1;
        cache.insert(&t.key, full.clone());
        Ok(full)
    }

    /// Bring the files in line with `presets` (built-ins skipped): write changed groups, delete
    /// removed ones and their unreferenced tips, update the index. The bitmaps of every group
    /// written become stored tiles (#1843): the files hold them, so the presets keep previews only.
    pub fn sync(&mut self, presets: &mut [BrushPreset]) {
        // User presets by group, in first-appearance order.
        let mut order: Vec<(String, Vec<usize>)> = Vec::new();
        for (i, p) in presets.iter().enumerate().filter(|(_, p)| !p.builtin) {
            match order.iter_mut().find(|(g, _)| *g == p.group) {
                Some((_, v)) => v.push(i),
                None => order.push((p.group.clone(), vec![i])),
            }
        }
        let mut files = Vec::with_capacity(order.len());
        for (group, idx) in &order {
            let file = group_file_name(group);
            files.push(file.clone());
            let unchanged = self
                .groups
                .get(&file)
                .is_some_and(|s| !s.stale && s.presets.len() == idx.len() && s.presets.iter().zip(idx).all(|(a, &i)| presets.get(i) == Some(a)));
            if unchanged {
                continue;
            }
            if self.skipped.contains(&file) {
                self.warnings.push(format!(
                    "Couldn't save brush presets `{}`: {file} could not be read, so it was left unchanged",
                    if group.is_empty() { "(ungrouped)" } else { group }
                ));
                continue;
            }
            let mut items: Vec<&mut BrushPreset> = presets.iter_mut().filter(|p| !p.builtin && p.group == *group).collect();
            if let Err(e) = self.write_group(&file, group, &mut items) {
                self.warnings.push(format!("Couldn't save brush presets `{}`: {e}", if group.is_empty() { "(ungrouped)" } else { group }));
            }
        }
        // Groups that are gone.
        let gone: Vec<String> = self.groups.keys().filter(|f| !files.contains(f)).cloned().collect();
        for f in gone {
            match self.backend.remove(&f) {
                Ok(()) => drop(self.groups.remove(&f)),
                Err(e) => self.warnings.push(format!("Couldn't delete brush presets file: {e}")),
            }
        }
        // Tips nothing references any more.
        let used: HashSet<&String> = self.groups.values().flat_map(|g| g.tips.iter()).collect();
        let unused: Vec<String> = self.tips.keys().filter(|h| !used.contains(h)).cloned().collect();
        for h in unused {
            match self.backend.remove(&tip_file(&h)) {
                Ok(()) => drop(self.tips.remove(&h)),
                Err(e) => self.warnings.push(format!("Couldn't delete brush tip: {e}")),
            }
        }
        // Index: group order and deleted built-ins.
        let hidden: Vec<String> = builtin_names().iter().filter(|n| !presets.iter().any(|p| p.name.eq_ignore_ascii_case(n))).map(|n| n.to_string()).collect();
        let groups: Vec<String> = files.into_iter().filter(|f| self.groups.contains_key(f)).collect();
        // The full order is only saved once it differs from the default (built-ins, then the rest).
        let order: Vec<String> = presets.iter().map(|p| p.name.clone()).collect();
        let default_order: Vec<String> = presets.iter().filter(|p| p.builtin).chain(presets.iter().filter(|p| !p.builtin)).map(|p| p.name.clone()).collect();
        let builtins_in_order = {
            let names = builtin_names();
            let mut it = presets.iter().filter(|p| p.builtin).map(|p| names.iter().position(|n| *n == p.name));
            let first = it.next().flatten();
            it.try_fold(first, |prev, cur| match (prev, cur) {
                (Some(a), Some(b)) if b > a => Some(Some(b)),
                _ => None,
            })
            .is_some()
        };
        let order = if builtins_in_order && order == default_order { Vec::new() } else { order };
        let want = IndexFile { version: 1, groups, hidden_builtins: hidden, order };
        let same = self.index.as_ref().is_some_and(|i| i.groups == want.groups && i.hidden_builtins == want.hidden_builtins && i.order == want.order);
        if !same {
            let empty = want.groups.is_empty() && want.hidden_builtins.is_empty() && want.order.is_empty();
            let r = if empty && self.index.is_none() {
                Ok(())
            } else {
                serde_json::to_vec_pretty(&want).map_err(|e| e.to_string()).and_then(|b| self.backend.write(INDEX_FILE, &b))
            };
            match r {
                Ok(()) => self.index = Some(want),
                Err(e) => self.warnings.push(format!("Couldn't save the brush preset index: {e}")),
            }
        }
    }

    /// Write one tip file (`others` bytes are already used by the rest of the store).
    fn put_tip(&mut self, h: &str, t: &GrayTile, others: u64, new_bytes: &mut u64, name: &str) -> Result<(), String> {
        let bytes = encode_tip(t);
        if bytes.is_empty() || bytes.len() as u64 > MAX_TIP_BYTES {
            return Err(format!("the tip of `{name}` is too large to save"));
        }
        *new_bytes += bytes.len() as u64;
        if others + *new_bytes > MAX_STORE_BYTES {
            return Err(format!("the preset store is full ({} MB limit)", MAX_STORE_BYTES >> 20));
        }
        self.backend.write(&tip_file(h), &bytes)?;
        self.tips.insert(h.to_string(), bytes.len() as u64);
        Ok(())
    }

    /// Write a group file (and any tip or preview file it needs that isn't on disk yet). Once it
    /// is written, the presets' bitmaps become stored tiles without their full bitmaps; on an
    /// error the presets are left unchanged.
    fn write_group(&mut self, file: &str, group: &str, items: &mut [&mut BrushPreset]) -> Result<(), String> {
        let others: u64 = self.groups.iter().filter(|(f, _)| f.as_str() != file).map(|(_, g)| g.bytes).sum::<u64>() + self.tips.values().sum::<u64>();
        let mut stored = Vec::with_capacity(items.len());
        let mut tiles = Vec::with_capacity(items.len());
        let mut hashes = Vec::new();
        let mut infos = BTreeMap::new();
        let mut new_bytes = 0u64;
        for p in items.iter_mut() {
            let mut refs: [Option<StoredTile>; 3] = [None, None, None];
            for (slot, bm) in refs.iter_mut().zip(bitmaps(&p.brush)) {
                let tile = match bm {
                    None => continue,
                    Some(Bitmap::Full(t)) => {
                        if !t.is_valid() {
                            return Err(format!("preset `{}` has an invalid tip", p.name));
                        }
                        let h = tip_hash(t);
                        if !self.tips.contains_key(&h) {
                            self.put_tip(&h, t, others, &mut new_bytes, &p.name)?;
                        }
                        StoredTile { key: h, width: t.width, height: t.height, preview: Arc::new(t.preview(PREVIEW_SIDE)), full: None }
                    }
                    Some(Bitmap::Stored(r)) => {
                        if !self.tips.contains_key(&r.key) {
                            return Err(format!("the tip of `{}` is missing from the preset store", p.name));
                        }
                        StoredTile { full: None, ..r.clone() }
                    }
                };
                let ph = tip_hash(&tile.preview);
                if !self.tips.contains_key(&ph) {
                    self.put_tip(&ph, &tile.preview, others, &mut new_bytes, &p.name)?;
                }
                hashes.push(tile.key.clone());
                hashes.push(ph.clone());
                infos.insert(tile.key.clone(), TipInfo { width: tile.width, height: tile.height, preview: ph });
                *slot = Some(tile);
            }
            let key = |r: &Option<StoredTile>| r.as_ref().map(|t| t.key.clone());
            let tips = TipRefs { tip: key(&refs[0]), dual_tip: key(&refs[1]), texture: key(&refs[2]) };
            stored.push(StoredPreset {
                name: p.name.clone(),
                brush: settings_only(&mut p.brush),
                tips,
                folder: p.folder.clone(),
                captures: p.captures.clone(),
            });
            tiles.push(refs);
        }
        let gf = GroupFile { format: GROUP_FORMAT.into(), version: 1, group: group.to_string(), presets: stored, tips: infos };
        let bytes = serde_json::to_vec(&gf).map_err(|e| e.to_string())?;
        if bytes.len() as u64 > MAX_GROUP_BYTES {
            return Err(format!("the group is too large to save ({} MB limit)", MAX_GROUP_BYTES >> 20));
        }
        if others + new_bytes + bytes.len() as u64 > MAX_STORE_BYTES {
            return Err(format!("the preset store is full ({} MB limit)", MAX_STORE_BYTES >> 20));
        }
        self.backend.write(file, &bytes)?;
        for (p, refs) in items.iter_mut().zip(tiles) {
            set_stored(&mut p.brush, refs);
        }
        let presets = items.iter().map(|p| (**p).clone()).collect();
        self.groups.insert(file.to_string(), Synced { presets, tips: hashes, bytes: bytes.len() as u64, stale: false });
        Ok(())
    }

    /// Write the Actions list when `rev` differs from the last load or write.
    /// An oversized list is warned once and not retried. An I/O error is warned and retried
    /// on the next command.
    pub fn sync_actions(&mut self, list: &[crate::actions_cmds::Action], rev: u64) {
        if self.actions_rev == rev {
            return;
        }
        if self.actions_skipped {
            self.warnings.push(format!("Couldn't save actions: {ACTIONS_FILE} could not be read, so it was left unchanged"));
            self.actions_rev = rev;
            return;
        }
        let file = crate::actions_cmds::ActionsFile { version: 1, actions: list.to_vec() };
        let bytes = match serde_json::to_vec_pretty(&file) {
            Ok(b) => b,
            Err(e) => {
                self.warnings.push(format!("Couldn't save actions: {e}"));
                self.actions_rev = rev;
                return;
            }
        };
        if bytes.len() as u64 > MAX_ACTIONS_BYTES {
            self.warnings.push(format!("actions: the action list is too large to save ({} MB limit)", MAX_ACTIONS_BYTES >> 20));
            self.actions_rev = rev;
            return;
        }
        match self.backend.write(ACTIONS_FILE, &bytes) {
            Ok(()) => self.actions_rev = rev,
            Err(e) => self.warnings.push(format!("Couldn't save actions: {e}")),
        }
    }
}

fn builtin_names() -> &'static [String] {
    static NAMES: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    NAMES.get_or_init(|| photocraft_paint::presets::builtin().into_iter().map(|p| p.name).collect())
}

impl Session {
    /// Record a brush preset change (create, delete, rename, import…): the attached store, if
    /// any, syncs after the running command.
    pub fn brush_presets_changed(&mut self) {
        self.tools.presets_rev += 1;
    }

    /// Attach a loaded store: merge its presets into the library (a stored preset replaces the
    /// built-in of the same name; presets created before the store finished loading win), hide
    /// deleted built-ins, then sync. Returns the load warnings.
    pub fn attach_preset_store(&mut self, opened: Opened) -> Vec<String> {
        let Opened { store, presets, hidden_builtins, order, mut warnings, actions: disk } = opened;
        let lib = &mut self.tools.presets;
        lib.retain(|p| !(p.builtin && hidden_builtins.iter().any(|h| h.eq_ignore_ascii_case(&p.name))));
        for p in presets {
            match lib.iter_mut().find(|x| x.name.eq_ignore_ascii_case(&p.name)) {
                Some(x) if x.builtin => *x = p,
                Some(_) => {}
                None => lib.push(p),
            }
        }
        // The saved panel order (stable: presets it doesn't name keep their order, at the end).
        if !order.is_empty() {
            let pos: HashMap<String, usize> = order.iter().enumerate().map(|(i, n)| (n.to_lowercase(), i)).collect();
            lib.sort_by_key(|p| pos.get(&p.name.to_lowercase()).copied().unwrap_or(usize::MAX));
        }
        self.preset_store = Some(store);
        // In-memory actions (created before the store finished loading) win on name. Disk-only
        // names are appended. An empty session takes the disk list and does not rewrite it.
        if self.actions.list.is_empty() {
            self.actions.list = disk;
            self.actions.rev = 1;
            if let Some(st) = self.preset_store.as_mut() {
                st.actions_rev = 1;
            }
        } else {
            for action in disk {
                if !self.actions.list.iter().any(|have| have.name == action.name) {
                    self.actions.list.push(action);
                }
            }
            self.actions.rev = self.actions.rev.saturating_add(1).max(1);
        }
        self.brush_presets_changed();
        self.sync_preset_store();
        // A brush remembered from the previous run (or picked before the store loaded) may be
        // painting with previews; load the full tips now (a missing one keeps its preview).
        let mut brush = std::mem::take(&mut self.tools.brush);
        let _ = self.load_brush_tips(&mut brush);
        self.tools.brush = brush;
        if let Some(st) = self.preset_store.as_mut() {
            warnings.extend(st.take_warnings());
        }
        warnings
    }

    /// Write pending brush preset and Actions changes to the attached store (no-op without one).
    pub fn sync_preset_store(&mut self) {
        let rev = self.tools.presets_rev;
        let actions_rev = self.actions.rev;
        if let Some(st) = self.preset_store.as_mut() {
            if st.synced_rev != Some(rev) {
                st.sync(&mut self.tools.presets);
                st.synced_rev = Some(rev);
                if link_stored_tips(st, &mut self.tools.brush) {
                    // An embedded tip became a store reference: what the last-brush
                    // persistence would write changed without a command touching the brush.
                    self.tools.brush_changed();
                }
            }
            st.sync_actions(&self.actions.list, actions_rev);
        }
    }

    /// Load the stored tips of `b` (a preset's settings picked as the tool's brush, or exported) so
    /// it paints with its full bitmaps. Tips already loaded or embedded are left alone. Fails when
    /// a tip can't be loaded: its file is missing or corrupt, or there is no preset store.
    pub fn load_brush_tips(&self, b: &mut BrushSettings) -> Result<(), String> {
        let load = |r: &mut StoredTile| -> Result<(), String> {
            if r.full.is_none() {
                let st = self.preset_store.as_ref().ok_or_else(|| format!("brush tip {} is not available: no preset store is attached", r.key))?;
                r.full = Some(st.load_tile(r)?);
            }
            Ok(())
        };
        if let TipShape::Stored(r) = &mut b.tip {
            load(r)?;
        }
        if let TipShape::Stored(r) = &mut b.dual_brush.tip {
            load(r)?;
        }
        if let Pattern::Stored(r) = &mut b.texture.pattern {
            load(r)?;
        }
        Ok(())
    }
}

/// After a sync, an embedded bitmap of the tool's brush that the store now holds (the brush was
/// just saved as a preset) becomes the stored tile, keeping its bitmap: the brush then matches its
/// preset (Brushes panel highlight) like a preset picked from the library.
fn link_stored_tips(st: &PresetStore, b: &mut BrushSettings) -> bool {
    let link = |g: &GrayTile| -> Option<StoredTile> {
        if !g.is_valid() {
            return None;
        }
        let h = tip_hash(g);
        st.tips.contains_key(&h).then(|| StoredTile { key: h, width: g.width, height: g.height, preview: Arc::new(g.preview(PREVIEW_SIDE)), full: None })
    };
    let mut linked = false;
    if let TipShape::Sampled(g) = &b.tip
        && let Some(r) = link(g)
        && let TipShape::Sampled(g) = std::mem::take(&mut b.tip)
    {
        b.tip = TipShape::Stored(StoredTile { full: Some(Arc::new(g)), ..r });
        linked = true;
    }
    if let TipShape::Sampled(g) = &b.dual_brush.tip
        && let Some(r) = link(g)
        && let TipShape::Sampled(g) = std::mem::take(&mut b.dual_brush.tip)
    {
        b.dual_brush.tip = TipShape::Stored(StoredTile { full: Some(Arc::new(g)), ..r });
        linked = true;
    }
    if let Pattern::Tile(g) = &b.texture.pattern
        && let Some(r) = link(g)
        && let Pattern::Tile(g) = std::mem::take(&mut b.texture.pattern)
    {
        b.texture.pattern = Pattern::Stored(StoredTile { full: Some(Arc::new(g)), ..r });
        linked = true;
    }
    linked
}

#[cfg(test)]
mod tests;
