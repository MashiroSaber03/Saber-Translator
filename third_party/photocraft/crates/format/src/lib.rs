//! # photocraft-format
//!
//! The native, lossless `.pcraft` document bundle (architecture §9).
//!
//! ```text
//! manifest.json            versioned document tree (see [`manifest`])
//! tiles/<blake3>.zst       content-addressed 256×256 tiles (zstd, little-endian samples)
//! blobs/<blake3>.zst       content-addressed binary data (ICC, EXIF, PSD blocks, smart objects)
//! thumb.png                optional thumbnail (caller-rendered)
//! composite/preview.png    optional flattened preview (caller-rendered)
//! ```
//!
//! A bundle is either a ZIP archive (STORE entries) or a directory with the
//! same layout. Saving is incremental: [`PcraftWriter`] remembers what it
//! already compressed/wrote, so only new tiles are encoded. Directory saves
//! verify existing content-addressed objects once per writer and folder,
//! rechecking files whose size or modification time changes; they replace
//! damaged or missing objects and garbage-collect unreferenced ones.
//!
//! This crate sits at L3 next to the compositor, so it does not render.
//! Callers pass previews in [`SaveOptions`]; `photocraft-io` does that.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod atomic;
pub mod autosave;
mod convert;
mod finite;
pub mod manifest;
mod migrate;
pub mod read;
mod store;
pub mod zip;

use std::path::Path;

use photocraft_doc::Document;
use photocraft_raster::Rgba8Image;

pub use atomic::{atomic_write, atomic_write_progress};
#[cfg(not(target_arch = "wasm32"))]
pub use autosave::RecoveryStore;
pub use autosave::{Autosaver, RecoveryEntry, discard_recovery, list_recovery, recover};
pub use convert::MAX_GROUP_DEPTH;
pub use manifest::{FORMAT_VERSION, Manifest};
pub use read::read_file;
pub use store::{ByteSource, DEFAULT_REVERIFY_BUDGET, PcraftWriter, SaveStats, read_entry};

/// File extension of the native format.
pub const EXTENSION: &str = "pcraft";

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("corrupt .pcraft bundle: {0}")]
    Corrupt(String),
    #[error("unsupported .pcraft feature: {0}")]
    Unsupported(String),
    #[error("bundle written by a newer version (format {found}, this build reads up to {supported})")]
    TooNew { found: u32, supported: u32 },
    #[error("limit exceeded: {0}")]
    LimitExceeded(String),
    #[error("`{path}` is NaN or infinite, which a .pcraft manifest cannot hold; reset that value and save again")]
    NonFinite { path: String },
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("manifest JSON: {0}")]
    Json(#[from] serde_json::Error),
}

impl FormatError {
    pub(crate) fn corrupt(msg: impl Into<String>) -> Self {
        FormatError::Corrupt(msg.into())
    }
}

pub type Result<T> = std::result::Result<T, FormatError>;

/// Previews to embed. The format crate cannot render (it sits beside the
/// compositor), so callers provide them.
#[derive(Debug, Clone, Default)]
pub struct SaveOptions {
    pub thumbnail: Option<Rgba8Image>,
    pub composite: Option<Rgba8Image>,
}

/// Limits applied while loading untrusted bundles.
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Maximum size of `manifest.json`.
    pub max_manifest_bytes: usize,
    /// Maximum size of one decompressed blob.
    pub max_blob_bytes: usize,
    /// Maximum total decompressed tile + blob bytes.
    pub max_total_bytes: u64,
    /// Keep document and layer ids from the file (the id counter is advanced
    /// past them). When `false`, fresh ids are assigned.
    pub preserve_ids: bool,
    /// Longest canvas side the manifest may declare (Photoshop's PSB limit, as for PSD and
    /// `file.new`).
    pub max_canvas_side: u32,
    /// Most canvas pixels the manifest may declare. A canvas is a few bytes of JSON, but the
    /// compositor, thumbnails and display textures work in proportion to its area, so a caller
    /// handling untrusted files (a previewer, a batch job) may lower it. The default admits every
    /// canvas the app lets a user create.
    pub max_canvas_pixels: u64,
}

/// Default [`LoadOptions::max_canvas_side`].
pub const MAX_CANVAS_SIDE: u32 = 300_000;
/// Default [`LoadOptions::max_canvas_pixels`]: the full 300000 × 300000 PSB canvas. `file.new`,
/// Canvas Size, Image Size (Resample off) and PSB import all reach it, and a document that can be
/// created and saved must reopen. Loading itself stays proportional to the file: layers, masks
/// and channels are sparse tiles read from the bundle under the byte budgets, so a huge declared
/// canvas with no content allocates nothing for its area.
pub const MAX_CANVAS_PIXELS: u64 = MAX_CANVAS_SIDE as u64 * MAX_CANVAS_SIDE as u64;

impl Default for LoadOptions {
    fn default() -> Self {
        LoadOptions {
            max_manifest_bytes: 256 << 20,
            max_blob_bytes: 1 << 30,
            max_total_bytes: 16 << 30,
            preserve_ids: true,
            max_canvas_side: MAX_CANVAS_SIDE,
            max_canvas_pixels: MAX_CANVAS_PIXELS,
        }
    }
}

/// `true` when ZIP bytes contain a `manifest.json` entry.
///
/// Bundles written here place the manifest first; re-zipped bundles may reorder entries.
pub fn is_pcraft(bytes: &[u8]) -> bool {
    if !bytes.starts_with(b"PK\x03\x04") {
        return false;
    }
    if bytes.len() >= 30 {
        let name_len = u16::from_le_bytes([bytes[26], bytes[27]]) as usize;
        if bytes.get(30..30 + name_len) == Some(b"manifest.json") {
            return true;
        }
    }
    zip::ZipReader::new(bytes).is_ok_and(|archive| archive.find("manifest.json").is_some())
}

/// Save to an in-memory ZIP bundle (one-shot; use [`PcraftWriter`] for
/// incremental saves).
pub fn save_to_bytes(doc: &Document, opts: &SaveOptions) -> Result<Vec<u8>> {
    PcraftWriter::new().save_zip(doc, opts).map(|(b, _)| b)
}

/// Load a ZIP bundle from memory.
pub fn load_from_bytes(bytes: &[u8]) -> Result<Document> {
    load_from_bytes_with(bytes, &LoadOptions::default())
}

pub fn load_from_bytes_with(bytes: &[u8], opts: &LoadOptions) -> Result<Document> {
    let src = store::ZipSource::new(bytes)?;
    store::load(&src, opts)
}

/// Load a bundle from a path: a directory bundle or a ZIP file.
pub fn load_path(path: &Path) -> Result<Document> {
    load_path_with(path, &LoadOptions::default())
}

/// Load from any byte source — a directory behind a capability handle, an archive, a test
/// fixture — the same migration and limits as the path loads.
pub fn load_bytes_source(src: &dyn ByteSource, opts: &LoadOptions) -> Result<Document> {
    store::load(src, opts)
}

pub fn load_path_with(path: &Path, opts: &LoadOptions) -> Result<Document> {
    if path.is_dir() {
        store::load(&store::DirSource { root: path.to_path_buf() }, opts)
    } else {
        let bytes = read::read_file(path)?;
        load_from_bytes_with(&bytes, opts)
    }
}

/// Read and migrate only the manifest (fast: no tiles are decoded).
pub fn read_manifest(bytes: &[u8]) -> Result<Manifest> {
    let src = store::ZipSource::new(bytes)?;
    store::read_manifest(&src, &LoadOptions::default())
}

/// The embedded thumbnail, if present (PNG bytes).
pub fn read_thumbnail(bytes: &[u8]) -> Result<Option<Vec<u8>>> {
    let src = store::ZipSource::new(bytes)?;
    let m = store::read_manifest(&src, &LoadOptions::default())?;
    match m.thumbnail {
        Some(p) => Ok(Some(src.get(&p, 64 << 20)?)),
        None => Ok(None),
    }
}
