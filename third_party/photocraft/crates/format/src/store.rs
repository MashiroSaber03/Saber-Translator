//! Object storage (ZIP or directory), incremental writer and loader.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};
use std::time::SystemTime;

use photocraft_color::{PixelFormat, SampleType};
use photocraft_doc::Document;
use photocraft_raster::{Rgba8Image, Tile};

use crate::convert::{self, Fetch, Loader, Sink, is_valid_hash, swap_to_le};
use crate::manifest::{ContentM, FORMAT_VERSION, Hash, LayerM, Manifest};
use crate::zip::{ZipReader, ZipWriter};
use crate::{FormatError, LoadOptions, Result, SaveOptions};

pub(crate) const MANIFEST: &str = "manifest.json";
pub(crate) const THUMB: &str = "thumb.png";
pub(crate) const COMPOSITE: &str = "composite/preview.png";

fn tile_path(h: &str) -> String {
    format!("tiles/{h}.zst")
}
fn blob_path(h: &str) -> String {
    format!("blobs/{h}.zst")
}

fn compress(data: &[u8]) -> Vec<u8> {
    ruzstd::encoding::compress_to_vec(data, ruzstd::encoding::CompressionLevel::Fastest)
}

/// Decodes a zstd frame of at most `max` bytes (`LimitExceeded` beyond). With `exact`, the
/// decoded length must be exactly that (`Corrupt` otherwise).
fn decompress_bounded(data: &[u8], max: usize, exact: Option<usize>, what: &str) -> Result<Vec<u8>> {
    let zstd = |e: &dyn std::fmt::Display| FormatError::corrupt(format!("{what}: zstd: {e}"));
    let mut dec = ruzstd::decoding::StreamingDecoder::new(data).map_err(|e| zstd(&e))?;
    let mut out = Vec::with_capacity(exact.unwrap_or(0).min(max).min(64 << 20));
    // `max` may be `usize::MAX`: one byte past it must not overflow.
    let limit = u64::try_from(max).unwrap_or(u64::MAX).saturating_add(1);
    (&mut dec).take(limit).read_to_end(&mut out).map_err(|e| zstd(&e))?;
    match exact {
        Some(n) if out.len() != n => Err(FormatError::corrupt(format!("{what}: expected {n} bytes, got {}", out.len()))),
        None if out.len() > max => Err(FormatError::LimitExceeded(format!("{what} exceeds {max} bytes"))),
        _ => Ok(out),
    }
}

fn decompress(data: &[u8], expected: usize, what: &str) -> Result<Vec<u8>> {
    decompress_bounded(data, expected, Some(expected), what)
}

fn hash_bytes(b: &[u8]) -> Hash {
    blake3::hash(b).to_hex().to_string()
}

fn png(img: &Rgba8Image) -> Result<Vec<u8>> {
    let image = photocraft_codecs::Image::from_u8(img.width, img.height, photocraft_codecs::ChannelLayout::Rgba, img.pixels.clone())
        .map_err(|e| FormatError::Unsupported(format!("preview: {e}")))?;
    photocraft_codecs::encode(&image, photocraft_codecs::Format::Png, &Default::default()).map_err(|e| FormatError::Unsupported(format!("preview: {e}")))
}

// ---------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------

pub trait ByteSource {
    fn get(&self, path: &str, max: usize) -> Result<Vec<u8>>;
}

pub(crate) struct ZipSource<'a> {
    zip: ZipReader<'a>,
}

impl<'a> ZipSource<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Ok(ZipSource { zip: ZipReader::new(bytes)? })
    }
}

impl ByteSource for ZipSource<'_> {
    fn get(&self, path: &str, max: usize) -> Result<Vec<u8>> {
        self.zip.read_by_name(path, max)
    }
}

pub(crate) struct DirSource {
    pub root: PathBuf,
}

impl ByteSource for DirSource {
    fn get(&self, path: &str, max: usize) -> Result<Vec<u8>> {
        let p = self.root.join(path);
        let metadata = std::fs::metadata(&p).map_err(|_| FormatError::corrupt(format!("missing `{path}`")))?;
        if !metadata.is_file() {
            return Err(FormatError::corrupt(format!("`{path}` is not a regular file")));
        }
        let mut file = std::fs::File::open(&p).map_err(|_| FormatError::corrupt(format!("missing `{path}`")))?;
        read_entry(&mut file, metadata.len(), max, path)
    }
}

/// Read one opened bundle entry under its `max` budget. The stat `len` is a buffer hint only
/// (files can change while open, and a FIFO reports 0), so the bound is enforced on the read
/// itself, never on the hint.
pub fn read_entry(file: &mut impl std::io::Read, len: u64, max: usize, path: &str) -> Result<Vec<u8>> {
    let mut limited = file.take((max as u64).saturating_add(1));
    let hint = len.min((max as u64).saturating_add(1));
    let bytes = crate::read::read_all(&mut limited, hint)?;
    if bytes.len() > max {
        return Err(FormatError::LimitExceeded(format!("`{path}` exceeds {max} bytes")));
    }
    Ok(bytes)
}

// ---------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------

/// Deepest JSON nesting a manifest may have: each group level adds three (the layer object, its
/// content object and the `children` array), with headroom for the document and leaf layer fields.
/// The saver checks the same bound, so every bundle it writes loads again.
const MAX_MANIFEST_DEPTH: usize = 3 * (convert::MAX_GROUP_DEPTH + 1) + 64;

/// Rejects JSON nested deeper than [`MAX_MANIFEST_DEPTH`] with a linear scan (no recursion), so
/// parsing with serde_json's recursion limit disabled can't exhaust the stack on hostile input.
fn check_manifest_depth(json: &[u8]) -> Result<()> {
    let (mut depth, mut in_str, mut escaped) = (0usize, false, false);
    for &b in json {
        if in_str {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => in_str = false,
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_str = true,
            b'[' | b'{' => {
                depth += 1;
                if depth > MAX_MANIFEST_DEPTH {
                    return Err(FormatError::LimitExceeded(format!("manifest JSON nested deeper than {MAX_MANIFEST_DEPTH} levels")));
                }
            }
            b']' | b'}' => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    Ok(())
}

pub(crate) fn read_manifest(src: &dyn ByteSource, opts: &LoadOptions) -> Result<Manifest> {
    let raw = src.get(MANIFEST, opts.max_manifest_bytes)?;
    // serde_json stops at 128 levels (about 40 nested groups); the bounded check replaces that limit.
    check_manifest_depth(&raw)?;
    let mut de = serde_json::Deserializer::from_slice(&raw);
    de.disable_recursion_limit();
    let mut v = <serde_json::Value as serde::Deserialize>::deserialize(&mut de)?;
    de.end()?;
    crate::migrate::migrate(&mut v)?;
    let layers = take_array(&mut v, "/document/layers");
    let mut m: Manifest = serde_json::from_value(v)?;
    m.document.layers = layers_from_json(layers)?;
    Ok(m)
}

/// Decodes layers one at a time, group children first. Decoding the whole tree in one go costs
/// kilobytes of stack per nesting level; here each level only holds a few pointers.
fn layers_from_json(values: Vec<serde_json::Value>) -> Result<Vec<LayerM>> {
    let mut out = Vec::with_capacity(values.len());
    for mut v in values {
        let children = layers_from_json(take_array(&mut v, "/content/children"))?;
        out.push(layer_from_json(v, children)?);
    }
    Ok(out)
}

/// One layer without its children (kept out of line so its large locals stay off the recursion).
#[inline(never)]
fn layer_from_json(v: serde_json::Value, children: Vec<LayerM>) -> Result<LayerM> {
    let mut layer: LayerM = serde_json::from_value(v)?;
    if let ContentM::Group { children: c, .. } = &mut layer.content {
        *c = children;
    }
    Ok(layer)
}

/// Takes the array at JSON `pointer` out of `v`, leaving an empty one (nothing if absent).
fn take_array(v: &mut serde_json::Value, pointer: &str) -> Vec<serde_json::Value> {
    match v.pointer_mut(pointer) {
        Some(serde_json::Value::Array(a)) => std::mem::take(a),
        _ => Vec::new(),
    }
}

struct LoadFetch<'a> {
    src: &'a dyn ByteSource,
    opts: LoadOptions,
    total: u64,
    blobs: HashMap<String, Arc<Vec<u8>>>,
}

impl LoadFetch<'_> {
    fn account(&mut self, n: usize) -> Result<()> {
        self.total += n as u64;
        if self.total > self.opts.max_total_bytes {
            return Err(FormatError::LimitExceeded(format!("bundle expands beyond {} bytes", self.opts.max_total_bytes)));
        }
        Ok(())
    }
}

impl Fetch for LoadFetch<'_> {
    fn tile(&mut self, hash: &str, len: usize) -> Result<Vec<u8>> {
        if !is_valid_hash(hash) {
            return Err(FormatError::corrupt(format!("invalid tile hash `{hash}`")));
        }
        self.account(len)?;
        let path = tile_path(hash);
        let z = self.src.get(&path, len + len / 8 + 4096)?;
        let data = decompress(&z, len, &path)?;
        if hash_bytes(&data) != hash {
            return Err(FormatError::corrupt(format!("{path}: content does not match its hash")));
        }
        Ok(data)
    }

    fn blob(&mut self, hash: &str) -> Result<Arc<Vec<u8>>> {
        if let Some(b) = self.blobs.get(hash) {
            return Ok(b.clone());
        }
        if !is_valid_hash(hash) {
            return Err(FormatError::corrupt(format!("invalid blob hash `{hash}`")));
        }
        let path = blob_path(hash);
        let z = self.src.get(&path, self.opts.max_blob_bytes)?;
        // Blob length is not in the manifest; the frame header bounds it and
        // we cap at max_blob_bytes.
        let data = decompress_bounded(&z, self.opts.max_blob_bytes, None, &path)?;
        self.account(data.len())?;
        if hash_bytes(&data) != hash {
            return Err(FormatError::corrupt(format!("{path}: content does not match its hash")));
        }
        let data = Arc::new(data);
        self.blobs.insert(hash.to_owned(), data.clone());
        Ok(data)
    }
}

/// Refuses a declared canvas outside `opts`' budget before anything is decoded for it (#2935).
fn admit_canvas(size: photocraft_geom::Size, opts: &LoadOptions) -> Result<()> {
    let (w, h) = (size.width, size.height);
    if w == 0 || h == 0 {
        return Err(FormatError::corrupt(format!("canvas size {w}x{h} is empty")));
    }
    if w > opts.max_canvas_side || h > opts.max_canvas_side {
        return Err(FormatError::LimitExceeded(format!("canvas {w}x{h} exceeds {} pixels per side", opts.max_canvas_side)));
    }
    if size.area() > opts.max_canvas_pixels {
        return Err(FormatError::LimitExceeded(format!("canvas {w}x{h} exceeds {} pixels", opts.max_canvas_pixels)));
    }
    Ok(())
}

pub(crate) fn load(src: &dyn ByteSource, opts: &LoadOptions) -> Result<Document> {
    let m = read_manifest(src, opts)?;
    admit_canvas(m.document.size, opts)?;
    let mut fetch = LoadFetch { src, opts: *opts, total: 0, blobs: HashMap::new() };
    let mut loader = Loader {
        fetch: &mut fetch,
        preserve_ids: opts.preserve_ids,
        id_map: HashMap::new(),
        contents_id_map: HashMap::new(),
        legacy_linked_contents: HashMap::new(),
        tiles: HashMap::new(),
    };
    loader.document(&m.document)
}

// ---------------------------------------------------------------------------
// Saving
// ---------------------------------------------------------------------------

/// What a save did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SaveStats {
    /// Distinct tiles referenced by the document.
    pub tiles_total: usize,
    /// Tiles that had to be compressed (ZIP) or written (directory).
    pub tiles_written: usize,
    /// Tiles reused from the previous save.
    pub tiles_reused: usize,
    pub blobs_total: usize,
    pub blobs_written: usize,
    /// Unreferenced objects removed (directory bundles).
    pub objects_removed: usize,
    /// Bytes of the manifest.
    pub manifest_bytes: usize,
}

#[derive(Default)]
struct Collect<'c> {
    tiles: BTreeMap<Hash, (Arc<Tile>, SampleType)>,
    blobs: BTreeMap<Hash, Arc<Vec<u8>>>,
    hash_cache: Option<&'c mut HashMap<usize, (Weak<Tile>, Hash)>>,
}

impl Sink for Collect<'_> {
    fn tile(&mut self, format: PixelFormat, tile: &Arc<Tile>) -> Hash {
        let key = Arc::as_ptr(tile) as usize;
        if let Some(cache) = self.hash_cache.as_deref_mut()
            && let Some((w, h)) = cache.get(&key)
            && w.upgrade().is_some_and(|t| Arc::ptr_eq(&t, tile))
        {
            let h = h.clone();
            self.tiles.entry(h.clone()).or_insert_with(|| (tile.clone(), format.sample));
            return h;
        }
        let h = hash_bytes(&tile_le(tile, format.sample));
        if let Some(cache) = self.hash_cache.as_deref_mut() {
            cache.insert(key, (Arc::downgrade(tile), h.clone()));
        }
        self.tiles.entry(h.clone()).or_insert_with(|| (tile.clone(), format.sample));
        h
    }

    fn blob(&mut self, data: &Arc<Vec<u8>>) -> Hash {
        let h = hash_bytes(data);
        self.blobs.entry(h.clone()).or_insert_with(|| data.clone());
        h
    }
}

fn tile_le(t: &Tile, sample: SampleType) -> std::borrow::Cow<'_, [u8]> {
    if cfg!(target_endian = "big") {
        let mut b = t.bytes().to_vec();
        swap_to_le(&mut b, sample);
        std::borrow::Cow::Owned(b)
    } else {
        std::borrow::Cow::Borrowed(t.bytes())
    }
}

/// Default for [`PcraftWriter::set_reverify_budget`]: compressed bytes of already-verified
/// objects re-read per directory save.
pub const DEFAULT_REVERIFY_BUDGET: u64 = 16 * 1024 * 1024;

/// Incremental `.pcraft` writer. Keep one per open document: it remembers
/// tile hashes (by `Arc` identity), compressed objects, and objects it has
/// verified in the current directory, so repeated saves stay cheap.
///
/// **What the directory cache trusts.** A directory save reuses an existing object file only
/// after decompressing it and checking its content hash. After that, the writer remembers the
/// file's signature (length and modification time, plus inode, device and change time on Unix)
/// and skips the full check on later saves while the signature is unchanged. A file damaged in
/// place without changing its signature (bit rot, or a tool that restores timestamps where
/// there is no change time) would be trusted, so each save also re-verifies a rolling share of
/// the cached objects, oldest-checked first, up to [`DEFAULT_REVERIFY_BUDGET`] bytes (see
/// [`Self::set_reverify_budget`]). Over enough saves every object is read again, and a damaged
/// one is rewritten.
pub struct PcraftWriter {
    hash_cache: HashMap<usize, (Weak<Tile>, Hash)>,
    /// Compressed objects by bundle path (ZIP mode).
    compressed: HashMap<String, Arc<Vec<u8>>>,
    verified_directory: Option<DirectoryVerificationCache>,
    reverify_budget: u64,
}

impl Default for PcraftWriter {
    fn default() -> Self {
        Self { hash_cache: HashMap::new(), compressed: HashMap::new(), verified_directory: None, reverify_budget: DEFAULT_REVERIFY_BUDGET }
    }
}

#[derive(Default)]
struct DirectoryVerificationCache {
    directory: PathBuf,
    /// Directory saves so far; an object's `checked_at` is the save that last read it fully.
    generation: u64,
    objects: HashMap<String, Verified>,
}

#[derive(Clone, Copy)]
struct Verified {
    signature: FileSignature,
    checked_at: u64,
}

/// What a file looked like when it was verified. Any change means "verify again".
#[derive(Clone, Copy, PartialEq, Eq)]
struct FileSignature {
    len: u64,
    modified: SystemTime,
    /// Inode, device and status-change time (s, ns): an in-place rewrite changes the change
    /// time even when a tool restores the modification time.
    #[cfg(unix)]
    unix: (u64, u64, i64, i64),
}

struct Prepared {
    manifest: Vec<u8>,
    objects: BTreeMap<String, Object>,
    previews: Vec<(&'static str, Vec<u8>)>,
    stats: SaveStats,
}

enum Object {
    Tile(Arc<Tile>, SampleType),
    Blob(Arc<Vec<u8>>),
}

impl Object {
    fn compressed(&self) -> Vec<u8> {
        compress(&self.content())
    }

    /// The bytes this object stores (tiles little-endian).
    fn content(&self) -> std::borrow::Cow<'_, [u8]> {
        match self {
            Object::Tile(tile, sample) => tile_le(tile, *sample),
            Object::Blob(blob) => std::borrow::Cow::Borrowed(blob.as_slice()),
        }
    }
}

/// Whether `compressed` decodes to exactly `expected`.
fn content_matches(expected: &[u8], compressed: &[u8], what: &str) -> bool {
    decompress(compressed, expected.len(), what).is_ok_and(|actual| hash_bytes(&actual) == hash_bytes(expected))
}

fn file_signature(metadata: &std::fs::Metadata) -> Option<FileSignature> {
    #[cfg(unix)]
    let unix = {
        use std::os::unix::fs::MetadataExt;
        (metadata.ino(), metadata.dev(), metadata.ctime(), metadata.ctime_nsec())
    };
    Some(FileSignature {
        len: metadata.len(),
        modified: metadata.modified().ok()?,
        #[cfg(unix)]
        unix,
    })
}

fn path_signature(path: &Path) -> Result<Option<FileSignature>> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(file_signature(&metadata)),
        Ok(_) => Err(unsafe_bundle_entry(path)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn unsafe_bundle_entry(path: &Path) -> FormatError {
    FormatError::Io(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("refusing directory bundle save: {} is a symlink or not a regular file", path.display()),
    ))
}

/// The generic atomic writer intentionally follows a destination symlink. Directory bundles
/// must reject one instead: all of their managed entries stay below the bundle root.
fn plain_file_or_missing(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_file() => Ok(()),
        Ok(_) => Err(unsafe_bundle_entry(path)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn plain_directory_or_missing(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if metadata.is_dir() => Ok(()),
        Ok(_) => Err(FormatError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("refusing directory bundle save: {} is a symlink or not a directory", path.display()),
        ))),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e.into()),
    }
}

/// Use the canonical root after creation so a later replacement of the caller's path cannot
/// redirect subsequent bundle operations. The caller may choose a symlinked root, but managed
/// subdirectories and files below its resolved target must not be symlinks.
fn bundle_root(dir: &Path) -> Result<PathBuf> {
    let dir: PathBuf = dir.components().collect();
    std::fs::create_dir_all(&dir)?;
    let root = std::fs::canonicalize(&dir)?;
    plain_directory_or_missing(&root)?;
    for sub in ["tiles", "blobs", "composite"] {
        let path = root.join(sub);
        plain_directory_or_missing(&path)?;
        if !path.exists() {
            match std::fs::create_dir(&path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(e) => return Err(e.into()),
            }
        }
        plain_directory_or_missing(&path)?;
        if std::fs::canonicalize(&path)? != path {
            return Err(unsafe_bundle_entry(&path));
        }
    }
    Ok(root)
}

fn validate_bundle_file(root: &Path, path: &Path) -> Result<()> {
    let parent = path.parent().ok_or_else(|| unsafe_bundle_entry(path))?;
    if parent != root && parent != root.join("tiles") && parent != root.join("blobs") && parent != root.join("composite") {
        return Err(unsafe_bundle_entry(path));
    }
    plain_directory_or_missing(parent)?;
    if std::fs::canonicalize(parent)? != parent {
        return Err(unsafe_bundle_entry(parent));
    }
    plain_file_or_missing(path)
}

fn write_bundle_file(root: &Path, path: &Path, data: &[u8]) -> Result<()> {
    validate_bundle_file(root, path)?;
    let parent = path.parent().ok_or_else(|| unsafe_bundle_entry(path))?;
    let name = path.file_name().ok_or_else(|| unsafe_bundle_entry(path))?;
    let mut temporary = None;
    for _ in 0..8 {
        let candidate = parent.join(crate::atomic::temp_name(&name.to_string_lossy()));
        match std::fs::OpenOptions::new().write(true).create_new(true).open(&candidate) {
            Ok(file) => {
                temporary = Some((candidate, file));
                break;
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e.into()),
        }
    }
    let (tmp, mut file) = temporary.ok_or_else(|| {
        FormatError::Io(std::io::Error::new(std::io::ErrorKind::AlreadyExists, format!("could not create a temporary file beside {}", path.display())))
    })?;
    let result = (|| -> Result<()> {
        // The parent may have changed between validation and create_new. Check the new file's
        // resolved location before putting document bytes into it.
        if std::fs::canonicalize(&tmp)? != tmp {
            return Err(unsafe_bundle_entry(&tmp));
        }
        validate_bundle_file(root, path)?;
        file.write_all(data)?;
        file.sync_all()?;
        drop(file);
        // Preserve the existing mode where possible, as the generic atomic writer does.
        if let Ok(permissions) = std::fs::metadata(path).map(|m| m.permissions()) {
            let _ = std::fs::set_permissions(&tmp, permissions);
        }
        // Recheck the parent and target just before publishing. rename replaces a symlink
        // itself, never its referent, even if one appears after this check.
        validate_bundle_file(root, path)?;
        crate::atomic::retry_rename(crate::atomic::RenameRetry::platform(), || std::fs::rename(&tmp, path))?;
        #[cfg(unix)]
        {
            let _ = std::fs::File::open(parent).and_then(|d| d.sync_all());
        }
        Ok(())
    })();
    if result.is_err() && validate_bundle_file(root, &tmp).is_ok() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// Keep the existing leaf/path validation and pin the actual unlink to opened
/// directory handles on Unix. This closes the check-then-remove path-swap gap.
fn remove_bundle_file(root: &Path, path: &Path, gc: &BundleGc) -> Result<()> {
    validate_bundle_file(root, path)?;
    let relative = path.strip_prefix(root).map_err(|_| unsafe_bundle_entry(path))?;
    gc.remove(relative)
}

#[cfg(unix)]
struct BundleGc {
    root: rustix::fd::OwnedFd,
    tiles: rustix::fd::OwnedFd,
    blobs: rustix::fd::OwnedFd,
    composite: rustix::fd::OwnedFd,
}

#[cfg(unix)]
fn gc_directory_flags() -> rustix::fs::OFlags {
    use rustix::fs::OFlags;

    // GC needs directory traversal, not permission to read a directory listing.
    // Both modes retain a descriptor for the directory opened here, so a later
    // pathname replacement cannot redirect descriptor-relative cleanup.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    let access = OFlags::PATH;
    #[cfg(target_vendor = "apple")]
    let access = OFlags::from_bits_retain(libc::O_SEARCH.cast_unsigned());
    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    let access = OFlags::RDONLY;
    access | OFlags::DIRECTORY | OFlags::NOFOLLOW | OFlags::CLOEXEC
}

#[cfg(unix)]
impl BundleGc {
    fn new(root: &Path) -> Result<Self> {
        use rustix::fs::{Mode, open, openat};

        let flags = gc_directory_flags();
        let root = open(root, flags, Mode::empty()).map_err(std::io::Error::from)?;
        let tiles = openat(&root, "tiles", flags, Mode::empty()).map_err(std::io::Error::from)?;
        let blobs = openat(&root, "blobs", flags, Mode::empty()).map_err(std::io::Error::from)?;
        let composite = openat(&root, "composite", flags, Mode::empty()).map_err(std::io::Error::from)?;
        Ok(Self { root, tiles, blobs, composite })
    }

    fn remove(&self, relative: &Path) -> Result<()> {
        let relative = relative.to_str().ok_or_else(|| unsafe_bundle_entry(relative))?;
        let (handle, name) = match relative {
            THUMB => (&self.root, THUMB),
            COMPOSITE => (&self.composite, "preview.png"),
            _ => match relative.split_once('/') {
                Some(("tiles", name)) if name.strip_suffix(".zst").is_some_and(is_valid_hash) => (&self.tiles, name),
                Some(("blobs", name)) if name.strip_suffix(".zst").is_some_and(is_valid_hash) => (&self.blobs, name),
                _ => return Err(FormatError::Unsupported(format!("invalid bundle cleanup target: {relative}"))),
            },
        };
        rustix::fs::unlinkat(handle, name, rustix::fs::AtFlags::empty()).map_err(std::io::Error::from)?;
        Ok(())
    }
}

#[cfg(not(unix))]
struct BundleGc {
    root: PathBuf,
}

#[cfg(not(unix))]
impl BundleGc {
    fn new(root: &Path) -> Result<Self> {
        Ok(Self { root: root.to_path_buf() })
    }

    fn remove(&self, relative: &Path) -> Result<()> {
        std::fs::remove_file(self.root.join(relative))?;
        Ok(())
    }
}

fn existing_object_is_valid(path: &Path, object: &Object, what: &str) -> Result<(bool, Option<FileSignature>)> {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((false, None)),
        Err(e) => return Err(e.into()),
    };
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Ok((false, None));
    }
    let signature = file_signature(&metadata);

    let expected = object.content();
    let expected_len = expected.len();
    // This is the same bound used by the tile loader. It comfortably includes
    // zstd framing overhead while keeping damaged files bounded when inspected.
    let max_compressed_len = expected_len.saturating_add(expected_len / 8).saturating_add(4096);
    let max_compressed_len_u64 = u64::try_from(max_compressed_len).unwrap_or(u64::MAX);
    if metadata.len() > max_compressed_len_u64 {
        return Ok((false, None));
    }

    let mut compressed = Vec::new();
    (&mut file).take(max_compressed_len_u64.saturating_add(1)).read_to_end(&mut compressed)?;
    if compressed.len() > max_compressed_len {
        return Ok((false, None));
    }
    Ok((content_matches(&expected, &compressed, what), signature))
}

/// A bundle path and its compressed bytes.
type Compressed<'a> = (&'a String, Arc<Vec<u8>>);

/// Compresses `objects` on scoped worker threads (sequentially on wasm).
fn par_compress<'a>(objects: &[(&'a String, &'a Object)]) -> Result<Vec<Compressed<'a>>> {
    let threads = if cfg!(target_arch = "wasm32") { 1 } else { std::thread::available_parallelism().map_or(1, |n| n.get()).clamp(1, 32) };
    if threads < 2 || objects.len() < 2 {
        return Ok(objects.iter().map(|(p, o)| (*p, Arc::new(o.compressed()))).collect());
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads.min(objects.len()))
            .map(|_| {
                sc.spawn(|| {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((p, o)) = objects.get(i) else { break };
                        out.push((*p, Arc::new(o.compressed())));
                    }
                    out
                })
            })
            .collect();
        // Join every worker before looking at the results, so one failure can't leave
        // another worker unjoined (the scope would then panic).
        let joined: Vec<_> = hs.into_iter().map(|h| h.join()).collect();
        let mut out = Vec::with_capacity(objects.len());
        for r in joined {
            out.extend(r.map_err(|_| FormatError::Io(std::io::Error::other("compression worker panicked")))?);
        }
        Ok(out)
    })
}

impl PcraftWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many compressed bytes of already-verified objects each directory save re-reads and
    /// re-checks (oldest-checked first; at least one object when the budget is non-zero). `0`
    /// trusts unchanged signatures entirely; `u64::MAX` re-verifies everything on every save.
    pub fn set_reverify_budget(&mut self, bytes: u64) {
        self.reverify_budget = bytes;
    }

    fn prepare(&mut self, doc: &Document, opts: &SaveOptions) -> Result<Prepared> {
        self.hash_cache.retain(|_, (w, _)| w.strong_count() > 0);
        convert::check_nesting(&doc.layers)?;
        let mut c = Collect { hash_cache: Some(&mut self.hash_cache), ..Default::default() };
        let document = convert::doc_m(doc, &mut c);
        let mut previews = Vec::new();
        if let Some(t) = &opts.thumbnail {
            previews.push((THUMB, png(t)?));
        }
        if let Some(t) = &opts.composite {
            previews.push((COMPOSITE, png(t)?));
        }
        let manifest = Manifest {
            format_version: FORMAT_VERSION,
            generator: format!("photocraft-format {}", env!("CARGO_PKG_VERSION")),
            document,
            thumbnail: opts.thumbnail.as_ref().map(|_| THUMB.to_owned()),
            composite: opts.composite.as_ref().map(|_| COMPOSITE.to_owned()),
        };
        // serde_json writes NaN and the infinities as `null`, which the manifest's float fields
        // refuse on load (issue #1101): refuse the save instead, naming the value.
        crate::finite::check(&manifest).map_err(|e| match e {
            crate::finite::Problem::NonFinite { path } => FormatError::NonFinite { path },
            crate::finite::Problem::Other(m) => FormatError::Unsupported(format!("manifest: {m}")),
        })?;
        let manifest = serde_json::to_vec_pretty(&manifest)?;
        check_manifest_depth(&manifest)?;
        let stats = SaveStats { tiles_total: c.tiles.len(), blobs_total: c.blobs.len(), manifest_bytes: manifest.len(), ..Default::default() };
        let mut objects = BTreeMap::new();
        for (h, (t, s)) in c.tiles {
            objects.insert(tile_path(&h), Object::Tile(t, s));
        }
        for (h, b) in c.blobs {
            objects.insert(blob_path(&h), Object::Blob(b));
        }
        Ok(Prepared { manifest, objects, previews, stats })
    }

    /// Save as a ZIP bundle in memory.
    pub fn save_zip(&mut self, doc: &Document, opts: &SaveOptions) -> Result<(Vec<u8>, SaveStats)> {
        let p = self.prepare(doc, opts)?;
        let mut stats = p.stats;
        let mut z = ZipWriter::new();
        z.add(MANIFEST, &p.manifest)?;
        for (name, data) in &p.previews {
            z.add(name, data)?;
        }
        let mut next = HashMap::with_capacity(p.objects.len());
        // New objects are compressed on all cores first (zstd dominates a full save).
        let todo: Vec<(&String, &Object)> = p.objects.iter().filter(|(path, _)| !self.compressed.contains_key(*path)).collect();
        let mut fresh: HashMap<&String, Arc<Vec<u8>>> = par_compress(&todo)?.into_iter().collect();
        for (path, obj) in &p.objects {
            let data = match self.compressed.get(path) {
                Some(d) => {
                    if matches!(obj, Object::Tile(..)) {
                        stats.tiles_reused += 1;
                    }
                    d.clone()
                }
                None => {
                    match obj {
                        Object::Tile(..) => stats.tiles_written += 1,
                        Object::Blob(_) => stats.blobs_written += 1,
                    }
                    fresh.remove(path).unwrap_or_else(|| Arc::new(obj.compressed()))
                }
            };
            z.add(path, &data)?;
            next.insert(path.clone(), data);
        }
        self.compressed = next;
        Ok((z.finish()?, stats))
    }

    /// Save into a directory bundle: reuses only valid existing objects,
    /// rewrites missing or damaged objects, then writes the manifest
    /// atomically and removes unreferenced objects.
    pub fn save_dir(&mut self, doc: &Document, dir: &Path, opts: &SaveOptions) -> Result<SaveStats> {
        let canonical_dir = bundle_root(dir)?;
        let dir = canonical_dir.as_path();
        let existing = list_objects(dir)?;
        // Reject a hostile existing entry before writing any object or manifest. This also
        // covers entries that this save would otherwise leave untouched or garbage-collect.
        for path in [MANIFEST, THUMB, COMPOSITE] {
            validate_bundle_file(dir, &dir.join(path))?;
        }
        let p = self.prepare(doc, opts)?;
        let mut stats = p.stats;
        if self.verified_directory.as_ref().is_none_or(|cache| cache.directory != canonical_dir) {
            self.verified_directory = Some(DirectoryVerificationCache { directory: canonical_dir.clone(), ..Default::default() });
        }
        let generation = match &mut self.verified_directory {
            Some(cache) => {
                cache.generation = cache.generation.saturating_add(1);
                cache.generation
            }
            None => 0,
        };
        let recheck = self.rolling_recheck(&p.objects);
        for (path, obj) in &p.objects {
            let object_path = dir.join(path);
            // bundle_root and list_objects validated the fixed parent directories once. Check
            // each leaf here; write_bundle_file rechecks its parent around every mutation.
            plain_file_or_missing(&object_path)?;
            let signature = path_signature(&object_path)?;
            let already_verified = !recheck.contains(path)
                && signature.is_some_and(|signature| {
                    self.verified_directory.as_ref().is_some_and(|cache| cache.objects.get(path).is_some_and(|v| v.signature == signature))
                });
            if existing.contains(path) && already_verified {
                if matches!(obj, Object::Tile(..)) {
                    stats.tiles_reused += 1;
                }
                continue;
            }
            if existing.contains(path) {
                let (valid, signature) = existing_object_is_valid(&object_path, obj, path)?;
                if valid {
                    if let Some(signature) = signature
                        && let Some(cache) = &mut self.verified_directory
                    {
                        cache.objects.insert(path.clone(), Verified { signature, checked_at: generation });
                    }
                    if matches!(obj, Object::Tile(..)) {
                        stats.tiles_reused += 1;
                    }
                    continue;
                }
            }
            match obj {
                Object::Tile(..) => stats.tiles_written += 1,
                Object::Blob(_) => stats.blobs_written += 1,
            }
            write_bundle_file(dir, &object_path, &obj.compressed())?;
            let written = path_signature(&object_path)?;
            if let Some(cache) = &mut self.verified_directory {
                match written {
                    Some(signature) => {
                        cache.objects.insert(path.clone(), Verified { signature, checked_at: generation });
                    }
                    None => {
                        cache.objects.remove(path);
                    }
                }
            }
        }
        for (name, data) in &p.previews {
            write_bundle_file(dir, &dir.join(name), data)?;
        }
        // Pin all GC directories before removing stale previews or objects.
        // Unix cleanup then stays within these directories even if a pathname is swapped.
        let gc = BundleGc::new(dir)?;
        for stale in [THUMB, COMPOSITE] {
            if !p.previews.iter().any(|(n, _)| *n == stale) {
                let path = dir.join(stale);
                validate_bundle_file(dir, &path)?;
                match remove_bundle_file(dir, &path, &gc) {
                    Err(FormatError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {}
                    other => other?,
                }
            }
        }
        write_bundle_file(dir, &dir.join(MANIFEST), &p.manifest)?;
        // The manifest is committed: the save has succeeded. Removing objects it no longer
        // references is best-effort housekeeping (a vanished or locked file must not turn a
        // complete save into a reported failure); a leftover is retried on the next save.
        for path in existing {
            if !p.objects.contains_key(&path) {
                if remove_bundle_file(dir, &dir.join(&path), &gc).is_ok() {
                    stats.objects_removed += 1;
                }
                if let Some(cache) = &mut self.verified_directory {
                    cache.objects.remove(&path);
                }
            }
        }
        Ok(stats)
    }

    /// The cached objects this save re-verifies even though their signature is unchanged:
    /// oldest-checked first, until their recorded sizes reach the re-verify budget (always at
    /// least one, so an object larger than the budget is still reached in turn).
    fn rolling_recheck(&self, objects: &BTreeMap<String, Object>) -> HashSet<String> {
        let mut out = HashSet::new();
        let Some(cache) = &self.verified_directory else { return out };
        if self.reverify_budget == 0 {
            return out;
        }
        let mut candidates: Vec<(&String, &Verified)> = cache.objects.iter().filter(|(path, _)| objects.contains_key(*path)).collect();
        candidates.sort_by(|a, b| a.1.checked_at.cmp(&b.1.checked_at).then_with(|| a.0.cmp(b.0)));
        let mut spent = 0u64;
        for (path, verified) in candidates {
            let next = spent.saturating_add(verified.signature.len);
            if !out.is_empty() && next > self.reverify_budget {
                break;
            }
            spent = next;
            out.insert(path.clone());
        }
        out
    }

    /// Save to `path`: a directory bundle if `path` is an existing directory
    /// or ends with a path separator, otherwise a ZIP file (written atomically).
    pub fn save_path(&mut self, doc: &Document, path: &Path, opts: &SaveOptions) -> Result<SaveStats> {
        let s = path.to_string_lossy();
        if path.is_dir() || s.ends_with('/') || s.ends_with('\\') {
            self.save_dir(doc, path, opts)
        } else {
            let (bytes, stats) = self.save_zip(doc, opts)?;
            write_atomic(path, &bytes)?;
            Ok(stats)
        }
    }
}

#[cfg(all(test, unix))]
#[test]
fn gc_pinned_directory_handles_resist_subdirectory_swap() {
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT: AtomicU64 = AtomicU64::new(0);
    let root = std::env::temp_dir().join(format!("pcraft-gc-swap-{}-{}", std::process::id(), NEXT.fetch_add(1, Ordering::Relaxed)));
    let bundle = root.join("project.pcraft");
    let outside = root.join("outside");
    for sub in ["tiles", "blobs", "composite"] {
        std::fs::create_dir_all(bundle.join(sub)).unwrap();
    }
    std::fs::create_dir(&outside).unwrap();
    let filename = format!("{}.zst", "c".repeat(64));
    std::fs::write(bundle.join("tiles").join(&filename), b"stale bundle file").unwrap();
    std::fs::write(outside.join(&filename), b"outside file must survive").unwrap();

    let gc = BundleGc::new(&bundle).unwrap();
    std::fs::rename(bundle.join("tiles"), bundle.join("old-tiles")).unwrap();
    symlink(&outside, bundle.join("tiles")).unwrap();

    // Exercise the descriptor-relative unlink directly: the public save path
    // also validates the leaf, but that check itself cannot prevent a later swap.
    gc.remove(Path::new(&format!("tiles/{filename}"))).unwrap();
    assert!(!bundle.join("old-tiles").join(&filename).exists());
    assert_eq!(std::fs::read(outside.join(&filename)).unwrap(), b"outside file must survive");
    std::fs::remove_dir_all(root).unwrap();
}

fn list_objects(dir: &Path) -> Result<HashSet<String>> {
    let mut out = HashSet::new();
    for sub in ["tiles", "blobs"] {
        let object_dir = dir.join(sub);
        plain_directory_or_missing(&object_dir)?;
        if std::fs::canonicalize(&object_dir)? != object_dir {
            return Err(unsafe_bundle_entry(&object_dir));
        }
        let rd = std::fs::read_dir(&object_dir)?;
        for e in rd {
            let e = e?;
            let name = e.file_name().to_string_lossy().into_owned();
            if name.strip_suffix(".zst").is_some_and(is_valid_hash) {
                let path = e.path();
                match std::fs::symlink_metadata(&path) {
                    Ok(metadata) if metadata.file_type().is_symlink() => return Err(unsafe_bundle_entry(&path)),
                    Ok(metadata) if metadata.is_file() => {
                        out.insert(format!("{sub}/{name}"));
                    }
                    // Stale directories cannot be removed with remove_file; leave them for later
                    // cleanup without failing a save of otherwise valid content.
                    Ok(_) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(e.into()),
                }
            }
        }
    }
    Ok(out)
}

/// Crash-safe replace of one file (see [`crate::atomic`]).
pub(crate) fn write_atomic(path: &Path, data: &[u8]) -> Result<()> {
    crate::atomic::atomic_write(path, data)?;
    Ok(())
}
