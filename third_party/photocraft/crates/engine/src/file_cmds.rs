//! File-menu commands beyond New/Close: Close All / Others, Revert, Save a Copy, Open As, Place
//! Embedded / Linked, File Info, the Automate and Scripts items (Fit Image, Conditional Mode
//! Change, Batch, Image Processor, Load Files into Stack, Flatten All Layer Effects / Masks),
//! Export › Layers to Files and Color Lookup Tables, plus the document-level guide layouts of the
//! View menu (New Guide Layout, New Guides From Shape, Clear Canvas Guides).
//!
//! Commands that touch the file system take explicit paths (the UI asks for them with its own
//! pickers) and are unavailable on the web, where there is no file system; their byte-level
//! cores ([`place_bytes`], [`open_bytes_as`]) work everywhere.

use std::sync::Arc;

use photocraft_algo::resample::{Resample, resize_surface, translate_surface};
use photocraft_color::{ColorMode, PixelFormat};
use photocraft_doc::{Affine, Document, Layer, LayerContent, LayerId, SmartObject, SmartSource};
use photocraft_geom::Rect;
use photocraft_raster::{Surface, from_rgba_into};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, layer_param};
use crate::{EngineError, Result, Session};

mod batch_naming;

// ---------- predicates ----------

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

/// The file system is reachable (not on the web).
pub(crate) fn native(_: &Session) -> std::result::Result<(), String> {
    if cfg!(target_arch = "wasm32") { Err("not available on the web (no file system)".into()) } else { Ok(()) }
}

pub(crate) fn native_doc(s: &Session) -> std::result::Result<(), String> {
    native(s)?;
    has_doc(s)
}

fn can_revert(s: &Session) -> std::result::Result<(), String> {
    native(s)?;
    let d = s.active().ok_or("no document open")?;
    let path = d.path.as_deref().ok_or("the document has never been saved")?;
    if !is_file(path) {
        return Err(format!("{path} is not a file on disk"));
    }
    Ok(())
}

fn has_adjustments(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.layers.iter().any(|l| l.visible && matches!(l.content, LayerContent::Adjustment(_))) {
        Ok(())
    } else {
        Err("the document has no visible adjustment layers to export".into())
    }
}

fn has_shape(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    match d.active_layer.and_then(|id| d.doc.layer(id)).map(|l| &l.content) {
        Some(LayerContent::Shape(_)) => Ok(()),
        _ => Err("the active layer is not a shape layer".into()),
    }
}

// ---------- file system (native only) ----------

#[cfg(not(target_arch = "wasm32"))]
fn is_file(path: &str) -> bool {
    std::path::Path::new(path).is_file()
}
#[cfg(target_arch = "wasm32")]
fn is_file(_path: &str) -> bool {
    false
}

#[cfg(not(target_arch = "wasm32"))]
/// Every file the engine opens is read here: in bounded reads, with a clear error when it does
/// not fit in memory (see [`photocraft_format::read`]).
pub(crate) fn read_file(path: &str) -> Result<Vec<u8>> {
    photocraft_format::read_file(std::path::Path::new(path)).map_err(|e| EngineError::Other(format!("{path}: {e}")))
}
#[cfg(target_arch = "wasm32")]
pub(crate) fn read_file(path: &str) -> Result<Vec<u8>> {
    Err(EngineError::Other(format!("cannot read {path}: no file system on the web")))
}

#[cfg(not(target_arch = "wasm32"))]
/// Every file the engine writes goes through here: crash-safe (temp file + fsync + rename, see
/// [`photocraft_format::atomic`]), so a failed save never destroys the previous file.
pub(crate) fn write_file(path: &str, bytes: &[u8]) -> Result<()> {
    photocraft_format::atomic_write(std::path::Path::new(path), bytes).map_err(|e| EngineError::Other(e.to_string()))
}
#[cfg(target_arch = "wasm32")]
pub(crate) fn write_file(path: &str, _bytes: &[u8]) -> Result<()> {
    Err(EngineError::Other(format!("cannot write {path}: no file system on the web")))
}

/// Files of a folder that look like images we can open, sorted by name.
#[cfg(not(target_arch = "wasm32"))]
pub fn list_images(dir: &str) -> Result<Vec<String>> {
    let rd = std::fs::read_dir(dir).map_err(|e| EngineError::Other(format!("{dir}: {e}")))?;
    let mut out: Vec<String> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file() && p.extension().and_then(|x| x.to_str()).is_some_and(|x| OPENABLE.contains(&x.to_ascii_lowercase().as_str())))
        .map(|p| p.to_string_lossy().into_owned())
        .collect();
    out.sort();
    Ok(out)
}
#[cfg(target_arch = "wasm32")]
pub fn list_images(dir: &str) -> Result<Vec<String>> {
    Err(EngineError::Other(format!("cannot list {dir}: no file system on the web")))
}

/// Extensions the batch commands pick up from a folder.
#[cfg(not(target_arch = "wasm32"))]
const OPENABLE: &[&str] = &[
    "pdn", "ora", "kra", "krz", "psd", "psb", "pcraft", "png", "jpg", "jpeg", "tif", "tiff", "webp", "gif", "bmp", "tga", "exr", "hdr", "qoi", "ico", "pnm",
    "ppm", "pgm", "fits", "fit", "fts", "dpx", "heic", "heif", "hif", "dng", "cr2", "nef", "nrw", "arw", "pef", "svg", "svgz", "af", "afdesign", "afphoto",
    "afpub",
];

/// Whether saving `doc` as a TIFF writes Photoshop layer data (anything beyond a lone
/// Background layer). The UI asks before such a save when the "ask before saving layered TIFF"
/// preference is on; see `photocraft_io::tiff_layers`.
pub fn tiff_would_write_layers(doc: &Document) -> bool {
    photocraft_io::tiff_layers::would_write_layers(doc)
}

pub(crate) fn file_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

impl crate::DocState {
    /// Complete a successful document save. File identity is not a history step; callers must
    /// finish the write first and must not call this for a copy or export.
    pub fn saved_to(&mut self, path: String) {
        let name = file_name(&path);
        if self.doc.name != name {
            Arc::make_mut(&mut self.doc).name = name;
        }
        self.path = Some(path);
        self.saved_revision = self.revision;
    }
}

/// The lower-case extension of the file name in `path` (none for `.hidden` or `name`).
pub fn extension(path: &str) -> Option<String> {
    file_name(path).rsplit_once('.').filter(|(base, ext)| !base.is_empty() && !ext.is_empty()).map(|(_, ext)| ext.to_ascii_lowercase())
}

/// Whether a save without a new path may write back to `path` as far as its format goes: only
/// layered files (PSD, PSB, OpenRaster, .pcraft). A flat file goes through Save As instead, so it
/// is never flattened over the original. Whether this document fits is [`holds_at`].
pub fn saves_in_place(path: &str) -> bool {
    extension(path).is_some_and(|ext| layered_extension(&ext))
}

/// Whether the lower-case extension `ext` names a layered document format (PSD, PSB, OpenRaster,
/// .pcraft). A save to one becomes the document's file; any other format is a copy for automation
/// saves, desktop and headless alike (#1547, #2579).
pub fn layered_extension(ext: &str) -> bool {
    matches!(ext, "psd" | "psb" | "pcraft" | "ora")
}

/// Whether a file in format `ext` (lower-case) holds the whole of `doc`, so it can be the
/// document's own file: PSD, PSB and .pcraft always; OpenRaster only when it drops nothing (no
/// adjustment layers, styles, masks…, [`photocraft_io::ora_losses`]). A save to a format that
/// doesn't hold the document is a copy, and File › Save goes through Save As, so an editable
/// original is never overwritten with less than it had (#3806).
pub fn holds(doc: &photocraft_doc::Document, ext: &str) -> bool {
    match ext {
        "psd" | "psb" | "pcraft" => true,
        "ora" => photocraft_io::ora_losses(doc).is_empty(),
        _ => false,
    }
}

/// [`holds`] for the file at `path`.
pub fn holds_at(doc: &photocraft_doc::Document, path: &str) -> bool {
    extension(path).is_some_and(|ext| holds(doc, &ext))
}

/// What a save of `doc` to `path` would leave out when its format can't hold it: OpenRaster's
/// [`photocraft_io::ora_losses`]; empty for other formats.
pub fn losses_at(doc: &photocraft_doc::Document, path: &str) -> Vec<String> {
    match extension(path).as_deref() {
        Some("ora") => photocraft_io::ora_losses(doc),
        _ => Vec::new(),
    }
}

/// Whether `path` names a document template (.psdt). A template opens as a new untitled document
/// without its path, so a save never writes over the template.
pub fn is_template(path: &str) -> bool {
    extension(path).as_deref() == Some("psdt")
}

/// The first "`base`-N" that isn't `taken`.
pub fn untitled_name(base: &str, taken: impl Fn(&str) -> bool) -> String {
    (1..).map(|i| format!("{base}-{i}")).find(|n| !taken(n)).unwrap_or_default()
}

/// The name a document opened from `path` gets when `path` is a template: the first "Untitled-N"
/// no open document has.
pub fn template_name(s: &Session, path: &str) -> Option<String> {
    is_template(path).then(|| untitled_name("Untitled", |n| s.documents().iter().any(|d| d.doc.name == n)))
}

pub(crate) fn stem(path: &str) -> String {
    let n = file_name(path);
    match n.rfind('.') {
        Some(i) if i > 0 => n[..i].to_string(),
        _ => n,
    }
}

pub(crate) fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_string()
    } else if dir.ends_with('/') || dir.ends_with('\\') {
        format!("{dir}{name}")
    } else {
        format!("{dir}/{name}")
    }
}

/// A file-name-safe version of a layer name.
pub(crate) fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_alphanumeric() || matches!(c, '-' | '_' | ' ' | '.') { c } else { '_' }).collect();
    let s = s.trim().trim_matches('.').to_string();
    if s.is_empty() { "layer".into() } else { s }
}

pub(crate) fn import(name: &str, bytes: &[u8]) -> Result<Document> {
    crate::allocation::checkpoint("importing document")?;
    let r = photocraft_io::import(name, bytes).map_err(|e| EngineError::Other(format!("{name}: {e}")))?;
    // Auxiliary imports return only a document and cannot surface the preview's fidelity warning.
    // Open has its own warning-preserving path; never silently place or process a thumbnail.
    if r.preview_only {
        return Err(EngineError::Other(format!(
            "{name}: only this Affinity file's embedded preview could be read; open it with File › Open to see the warning, or export PSD or PNG from Affinity before using it here"
        )));
    }
    Ok(r.document)
}

/// What a headless save writes beyond the format: JPEG quality, TIFF layers, Targa bits, PSD
/// previews and the PSD merged image.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SaveOpts {
    /// Photoshop's 0–12 JPEG scale.
    pub quality: Option<f64>,
    /// TIFF: keep the layers. Off unless a command's params ask (`"tiffLayers": true`).
    pub tiff_layers: bool,
    /// Targa: 24 or 32 bits per pixel (`"tgaBits"`); unset, 32 with an alpha channel or
    /// transparency, else 24.
    pub tga_bits: Option<photocraft_io::TgaBits>,
    /// DPX: 8, 10, 12 or 16 bits per sample (`"dpxBits"`); unset, 8 for an 8-bit document, else 16.
    pub dpx_bits: Option<photocraft_io::DpxBits>,
    /// PSD/PSB: embed the thumbnail resource (Preferences › File Handling › Image Previews).
    pub image_previews: bool,
    /// PSD/PSB: write the real merged image (Preferences › File Handling › Maximize PSD and PSB
    /// File Compatibility).
    pub psd_maximize_compatibility: bool,
}

impl Default for SaveOpts {
    fn default() -> Self {
        SaveOpts { quality: None, tiff_layers: false, tga_bits: None, dpx_bits: None, image_previews: true, psd_maximize_compatibility: true }
    }
}

impl SaveOpts {
    /// `quality`, `tiffLayers` and `tgaBits` from command `cmd`'s params (a `tgaBits` other than
    /// 24 or 32 is an error); the image previews and the PSD merged image from the session's File
    /// Handling preferences (Ask When Saving has no dialog here, so it saves them, as Photoshop's
    /// Save As checkbox and its Maximize Compatibility checkbox both start on).
    pub(crate) fn from_params(s: &Session, p: &Value, cmd: &str) -> Result<Self> {
        let file_handling = &s.prefs().file_handling;
        Ok(SaveOpts {
            quality: f64_param(p, "quality"),
            tiff_layers: p.get("tiffLayers").and_then(Value::as_bool).unwrap_or(false),
            tga_bits: tga_bits_param(p, cmd)?,
            dpx_bits: dpx_bits_param(p, cmd)?,
            image_previews: file_handling.image_previews != crate::prefs::Ask::Never,
            psd_maximize_compatibility: file_handling.maximize_psd_compatibility != crate::prefs::Ask::Never,
        })
    }

    pub(crate) fn or_quality(mut self, q: f64) -> Self {
        self.quality = self.quality.or(Some(q));
        self
    }
}

impl From<Option<f64>> for SaveOpts {
    fn from(quality: Option<f64>) -> Self {
        SaveOpts { quality, ..Default::default() }
    }
}

/// The Targa bits per pixel in `p` (`"tgaBits": 24 | 32`; absent or null: the automatic choice).
/// Anything else is a bad parameter of command `cmd`. Shared by every save that takes it (the
/// engine's commands, the app's Save As and control channel, the headless server).
pub fn tga_bits_param(p: &Value, cmd: &str) -> Result<Option<photocraft_io::TgaBits>> {
    match p.get("tgaBits") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(photocraft_io::TgaBits::from_bits)
            .map(Some)
            .ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: format!("`tgaBits` must be 24 or 32, got {v}") }),
    }
}

/// The DPX bits per sample in `p` (`"dpxBits": 8 | 10 | 12 | 16`; absent or null: the document's
/// depth). Anything else is a bad parameter of command `cmd`.
pub fn dpx_bits_param(p: &Value, cmd: &str) -> Result<Option<photocraft_io::DpxBits>> {
    match p.get("dpxBits") {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .and_then(photocraft_io::DpxBits::from_bits)
            .map(Some)
            .ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: format!("`dpxBits` must be 8, 10, 12 or 16, got {v}") }),
    }
}

/// Encodes `doc` for `path`'s extension.
pub(crate) fn encode(doc: &Document, path: &str, save: impl Into<SaveOpts>) -> Result<(Vec<u8>, Vec<String>)> {
    crate::allocation::checkpoint("saving document")?;
    let save = save.into();
    let mut opts = photocraft_io::ExportOptions {
        tiff_layers: save.tiff_layers,
        tga_bits: save.tga_bits,
        image_previews: save.image_previews,
        psd_maximize_compatibility: save.psd_maximize_compatibility,
        ..Default::default()
    };
    // No quality chosen: write the file the way it was opened (a JPEG at its own quality, a WebP
    // lossy or lossless as it was), as Photoshop's Save does.
    photocraft_io::apply_file_encoding(doc, &mut opts.encode);
    opts.encode.dpx_bits = save.dpx_bits;
    if let Some(q) = save.quality {
        let q = (q.clamp(0.0, 12.0) / 12.0 * 99.0 + 1.0).round() as u8;
        opts.encode.jpeg_quality = q;
        // A quality on a WebP save asks for the lossy encoder; the default WebP stays lossless.
        opts.encode.webp_quality = q;
        opts.encode.webp_lossless = false;
    }
    photocraft_io::export(doc, path, &opts).map(|r| (r.bytes, r.warnings)).map_err(|e| EngineError::Other(format!("{path}: {e}")))
}

pub(crate) fn save_doc(doc: &Document, path: &str, save: impl Into<SaveOpts>) -> Result<Vec<String>> {
    let (bytes, warnings) = encode(doc, path, save)?;
    write_file(path, &bytes)?;
    Ok(warnings)
}

pub(crate) fn str_param<'a>(p: &'a Value, key: &str, cmd: &str) -> Result<&'a str> {
    p.get(key).and_then(Value::as_str).filter(|v| !v.is_empty()).ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: format!("missing \"{key}\"") })
}

pub(crate) fn f64_param(p: &Value, key: &str) -> Option<f64> {
    p.get(key).and_then(Value::as_f64).filter(|v| v.is_finite())
}

/// A size parameter; zero or negative means "not set" (dialogs send 0 for empty fields).
fn size_param(p: &Value, key: &str) -> Option<f64> {
    f64_param(p, key).filter(|v| *v > 0.0)
}

// ---------- pixels ----------

/// A composite buffer as a surface in `fmt` (straight RGBA → the format's model and depth).
pub(crate) fn buffer_surface(buf: &photocraft_compose::Buffer, fmt: PixelFormat) -> Surface {
    let n = fmt.channels();
    let mut data = vec![0.0f32; buf.px.len() * n];
    for (p, out) in buf.px.iter().zip(data.chunks_exact_mut(n)) {
        from_rgba_into(&fmt, *p, out);
    }
    let mut s = Surface::new(fmt);
    if !buf.rect.is_empty() {
        s.write_region(buf.rect, &data);
    }
    s.prune();
    s
}

/// The flattened image of `doc` as a surface in `fmt`, placed at the origin.
pub(crate) fn flattened(doc: &Document, fmt: PixelFormat) -> Surface {
    photocraft_compose::flatten_to_surface(doc, fmt, None)
}

// ---------- close / revert / save a copy / open as ----------

fn close_all(s: &mut Session) -> Result<Value> {
    let n = s.documents().len();
    while !s.documents().is_empty() {
        s.close(s.documents().len() - 1);
    }
    Ok(json!({"closed": n}))
}

fn close_others(s: &mut Session, p: &Value) -> Result<Value> {
    let keep = p.get("document").and_then(Value::as_u64).map(|v| v as usize).or(s.active_index()).ok_or(EngineError::NoDocument)?;
    if keep >= s.documents().len() {
        return Err(EngineError::BadParams { cmd: "file.closeOthers".into(), msg: format!("no document {keep}") });
    }
    let n = s.documents().len() - 1;
    for i in (0..s.documents().len()).rev() {
        if i != keep {
            s.close(i);
        }
    }
    s.set_active(0);
    Ok(json!({"closed": n}))
}

/// File › Revert: reload the saved file as one history step (undoable, like Photoshop's
/// "Revert" history state); the document is clean afterwards.
fn revert(s: &mut Session) -> Result<Value> {
    let path = s.active().and_then(|d| d.path.clone()).ok_or(EngineError::Other("the document has never been saved".into()))?;
    let bytes = read_file(&path)?;
    let fresh = import(&file_name(&path), &bytes)?;
    s.edit("Revert", |doc, active| {
        let (id, name) = (doc.id, doc.name.clone());
        *doc = fresh;
        // Keep the identity (colour state, views and caches are keyed by it) and the tab name.
        doc.id = id;
        doc.name = name;
        *active = doc.top_layer();
        Ok(())
    })?;
    let st = s.active_mut().ok_or(EngineError::NoDocument)?;
    st.saved_revision = st.revision;
    Ok(json!({"path": path, "layers": st.doc.layers.len()}))
}

fn save_a_copy(s: &mut Session, p: &Value) -> Result<Value> {
    let path = str_param(p, "path", "file.saveACopy")?.to_string();
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let mut doc = (*d.doc).clone();
    if p.get("layers").and_then(Value::as_bool) == Some(false) {
        // "Layers" unchecked in Save a Copy: write the flattened image.
        let fmt = doc.pixel_format();
        let px = flattened(&doc, fmt);
        doc.layers = vec![Layer::new("Background", LayerContent::Raster(px))];
    }
    let warnings = save_doc(&doc, &path, SaveOpts::from_params(s, p, "file.saveACopy")?)?;
    Ok(json!({"path": path, "warnings": warnings}))
}

/// Open a file's bytes, decoding them as the format `as_ext` (Open As) when given.
pub fn open_bytes_as(s: &mut Session, name: &str, bytes: &[u8], as_ext: Option<&str>, path: Option<String>) -> Result<Value> {
    crate::allocation::checkpoint("importing document")?;
    let decode_name = match as_ext {
        Some(ext) => format!("{}.{}", stem(name), ext.trim_start_matches('.')),
        None => name.to_string(),
    };
    let r = photocraft_io::import(&decode_name, bytes).map_err(|e| EngineError::Other(format!("{decode_name}: {e}")))?;
    let mut doc = r.document;
    let path = match template_name(s, name) {
        Some(untitled) => {
            doc.name = untitled;
            None
        }
        None => {
            doc.name = file_name(name);
            path
        }
    };
    // Color Settings policies (preserve / convert / discard the embedded profile).
    let (i, color) = s.open_document(doc, path.filter(|_| !r.source_read_only));
    if let Some(st) = s.active_mut() {
        st.source_read_only = r.source_read_only;
    }
    // Import notes (e.g. how a camera raw was developed, or that only its preview opened).
    Ok(json!({"document": i, "color": color, "warnings": r.warnings}))
}

fn open_as(s: &mut Session, p: &Value) -> Result<Value> {
    let path = str_param(p, "path", "file.openAs")?.to_string();
    let bytes = read_file(&path)?;
    let as_ext = p.get("as").or_else(|| p.get("format")).and_then(Value::as_str);
    open_bytes_as(s, &path, &bytes, as_ext, Some(path.clone()))
}

// ---------- place ----------

/// History label of an embedded place.
pub const PLACE_EMBEDDED: &str = "Place Embedded";

/// Place a file's bytes as a smart object layer, centred, and (with Preferences ▸ General ▸
/// Resize Image During Place, on by default) scaled down to fit when larger than the canvas.
/// `linked` makes it a linked smart object that refers to that path instead of embedding the bytes.
/// With Preferences ▸ General ▸ Always Create Smart Objects When Placing off, an embedded place
/// lands as a plain pixel layer instead (an explicit `"smartObject": false` asks for one).
pub fn place_bytes(s: &mut Session, name: &str, bytes: Vec<u8>, linked: Option<String>, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (cw, ch) = (d.doc.size.width as f64, d.doc.size.height as f64);
    let fmt = d.doc.pixel_format();
    let vector = photocraft_io::svg::is_svg(&bytes);
    let max_svg_group_depth = s.prefs().file_handling.rasterize_svg_groups_deeper_than as usize;
    let imported = photocraft_io::import_with_svg_group_depth(name, &bytes, max_svg_group_depth).map_err(|e| EngineError::Other(format!("{name}: {e}")))?;
    if imported.preview_only {
        return Err(EngineError::Other(format!(
            "{name}: only this Affinity file's embedded preview could be read; open it with File › Open to see the warning, or export PSD or PNG from Affinity before using it here"
        )));
    }
    let src = imported.document;
    let (w, h) = (src.size.width as f64, src.size.height as f64);
    // The implicit `fit` is Preferences ▸ General ▸ Resize Image During Place (on by default).
    let fit = p.get("fit").and_then(Value::as_bool).unwrap_or(s.prefs().general.resize_image_during_place);
    let scale = match f64_param(p, "scale") {
        Some(k) => (k / 100.0).max(1e-4),
        None if fit && (w > cw || h > ch) => (cw / w).min(ch / h),
        None => 1.0,
    };
    let center = match p.get("center").and_then(Value::as_array) {
        Some(c) if c.len() == 2 => (c[0].as_f64().unwrap_or(cw / 2.0), c[1].as_f64().unwrap_or(ch / 2.0)),
        _ => (cw / 2.0, ch / 2.0),
    };
    let (dx, dy) = ((center.0 - w * scale / 2.0).round(), (center.1 - h * scale / 2.0).round());
    let mut px = flattened(&src, fmt);
    if (scale - 1.0).abs() > 1e-9 {
        px = resize_surface(&px, scale, scale, Resample::Bicubic);
    }
    let px = translate_surface(&px, dx as i32, dy as i32);
    // Preferences ▸ General ▸ Always Create Smart Objects When Placing (on by default): off, an
    // embedded place lands as a plain pixel layer. A linked place is always a smart object.
    let smart = linked.is_some() || p.get("smartObject").and_then(Value::as_bool).unwrap_or(s.prefs().general.always_create_smart_objects_when_placing);
    let source = match linked {
        Some(path) => SmartSource::Linked { path },
        None => SmartSource::Embedded { file_name: file_name(name), bytes: Arc::new(bytes) },
    };
    let layer_name = stem(name);
    let label = if matches!(&source, SmartSource::Linked { .. }) { "Place Linked" } else { PLACE_EMBEDDED };
    let id = s.edit(label, |doc, active| {
        let layer = if smart {
            let mut so = SmartObject::new(source, Affine { m: [scale, 0.0, 0.0, scale, dx, dy] }, Some(px));
            // A vector source (SVG) renders at its placement scale instead of as a resampled raster.
            if vector && let Some(sharp) = crate::smart_cmds::render(doc, &so)? {
                so.cache = Some(sharp);
            }
            Layer::new(layer_name, LayerContent::Smart(so))
        } else {
            Layer::new(layer_name, LayerContent::Raster(px))
        };
        let id = doc.insert_above(*active, layer);
        *active = Some(id);
        Ok(id)
    })?;
    Ok(json!({"layer": id.0, "scale": scale * 100.0, "bounds": [dx, dy, dx + w * scale, dy + h * scale], "smartObject": smart}))
}

fn place(s: &mut Session, p: &Value, linked: bool) -> Result<Value> {
    let cmd = if linked { "file.placeLinked" } else { "file.placeEmbedded" };
    let path = str_param(p, "path", cmd)?.to_string();
    let bytes = read_file(&path)?;
    place_bytes(s, &path, bytes, linked.then(|| path.clone()), p)
}

// ---------- File Info (XMP) ----------

/// The File Info fields we edit, as (JSON key, XMP property, container kind).
const INFO_FIELDS: [(&str, &str, &str); 6] = [
    ("title", "dc:title", "Alt"),
    ("author", "dc:creator", "Seq"),
    ("description", "dc:description", "Alt"),
    ("keywords", "dc:subject", "Bag"),
    ("copyright", "dc:rights", "Alt"),
    ("authorTitle", "photoshop:AuthorsPosition", ""),
];

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&apos;", "'").replace("&amp;", "&")
}

/// The element `<tag …>…</tag>` (first occurrence) as (start, end, inner).
fn find_element<'a>(xmp: &'a str, tag: &str) -> Option<(usize, usize, &'a str)> {
    let open = format!("<{tag}");
    let mut from = 0;
    while let Some(i) = xmp[from..].find(&open).map(|i| i + from) {
        let after = xmp[i + open.len()..].chars().next();
        if matches!(after, Some('>' | ' ' | '/' | '\n' | '\r' | '\t')) {
            let gt = xmp[i..].find('>')? + i;
            if xmp[..gt].ends_with('/') {
                return Some((i, gt + 1, ""));
            }
            let close = format!("</{tag}>");
            let c = xmp[gt..].find(&close)? + gt;
            return Some((i, c + close.len(), &xmp[gt + 1..c]));
        }
        from = i + open.len();
    }
    None
}

/// `rdf:li` values of a container (or the element text for simple properties).
fn li_values(inner: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = inner;
    while let Some((a, b, v)) = find_element(rest, "rdf:li") {
        out.push(xml_unescape(v.trim()));
        rest = &rest[b..];
        let _ = a;
    }
    if out.is_empty() && !inner.contains('<') && !inner.trim().is_empty() {
        out.push(xml_unescape(inner.trim()));
    }
    out
}

/// Attribute form `prop="value"` on an `rdf:Description`.
fn find_attr<'a>(xmp: &'a str, prop: &str) -> Option<(usize, usize, &'a str)> {
    let key = format!(" {prop}=\"");
    let i = xmp.find(&key)?;
    let v0 = i + key.len();
    let v1 = xmp[v0..].find('"')? + v0;
    Some((i, v1 + 1, &xmp[v0..v1]))
}

/// File Info fields read from an XMP packet.
pub fn read_file_info(xmp: Option<&str>) -> Value {
    let mut m = serde_json::Map::new();
    let x = xmp.unwrap_or("");
    for (key, prop, kind) in INFO_FIELDS {
        let vals = match find_element(x, prop) {
            Some((_, _, inner)) => li_values(inner),
            None => find_attr(x, prop).map(|(_, _, v)| vec![xml_unescape(v)]).unwrap_or_default(),
        };
        let v = if kind == "Bag" {
            json!(vals)
        } else if kind == "Seq" {
            json!(vals.join("; "))
        } else {
            json!(vals.into_iter().next().unwrap_or_default())
        };
        m.insert(key.into(), v);
    }
    let marked =
        find_element(x, "xmpRights:Marked").map(|(_, _, v)| v.trim().to_string()).or_else(|| find_attr(x, "xmpRights:Marked").map(|(_, _, v)| v.to_string()));
    m.insert(
        "copyrightStatus".into(),
        json!(match marked.as_deref() {
            Some("True") => "copyrighted",
            Some("False") => "publicDomain",
            _ => "unknown",
        }),
    );
    let url = find_element(x, "xmpRights:WebStatement")
        .map(|(_, _, v)| xml_unescape(v.trim()))
        .or_else(|| find_attr(x, "xmpRights:WebStatement").map(|(_, _, v)| xml_unescape(v)));
    m.insert("copyrightUrl".into(), json!(url.unwrap_or_default()));
    Value::Object(m)
}

/// File Info's read-only Camera Data, as (result key, English label) in display order.
pub const CAMERA_FIELDS: [(&str, &str); 9] = [
    ("make", "Make"),
    ("model", "Model"),
    ("dateTaken", "Date Taken"),
    ("exposure", "Exposure"),
    ("fNumber", "F-Stop"),
    ("iso", "ISO Speed"),
    ("focalLength", "Focal Length"),
    ("focalLength35mm", "Focal Length (35 mm)"),
    ("lens", "Lens"),
];

/// `v` with at most `decimals` decimals and no trailing zeros (33.10 → "33.1", 8.0 → "8").
fn short_num(v: f64, decimals: usize) -> String {
    let s = format!("{v:.decimals$}");
    if s.contains('.') { s.trim_end_matches('0').trim_end_matches('.').to_string() } else { s }
}

/// Exposure time as cameras print it: "1/250 s" for fractions of a second, "0.4 s" or "2 s"
/// when the reciprocal isn't a whole number or the exposure is a second or longer.
fn exposure_text(t: f64) -> Option<String> {
    if !(t.is_finite() && t > 0.0) {
        return None;
    }
    let r = 1.0 / t;
    if t < 1.0 && (r - r.round()).abs() <= r * 0.01 {
        return Some(format!("1/{} s", r.round()));
    }
    Some(format!("{} s", short_num(t, if t < 1.0 { 2 } else { 1 })))
}

/// EXIF (`2024:02:17 12:40:19`) or XMP (`2024-02-17T12:40:19.00+01:00`) dates as
/// `2024-02-17 12:40:19`; anything else is shown as stored.
fn date_text(s: &str) -> String {
    let b = s.trim().as_bytes();
    let digits = |r: &[usize]| r.iter().all(|&i| b.get(i).is_some_and(u8::is_ascii_digit));
    let sep = |i: usize, set: &[u8]| b.get(i).is_some_and(|c| set.contains(c));
    let ch = |i: usize| b.get(i).map_or('0', |&c| c as char);
    let part = |r: std::ops::Range<usize>| r.map(ch).collect::<String>();
    if !(digits(&[0, 1, 2, 3, 5, 6, 8, 9]) && sep(4, b":-") && sep(7, b":-")) {
        return s.trim().to_string();
    }
    let date = format!("{}-{}-{}", part(0..4), part(5..7), part(8..10));
    if digits(&[11, 12, 14, 15, 17, 18]) && sep(10, b" T") && sep(13, b":") && sep(16, b":") { format!("{date} {}", part(11..19)) } else { date }
}

/// A single XMP value: the first `rdf:li` of a container, element text or attribute.
fn xmp_value(x: &str, prop: &str) -> Option<String> {
    let v = match find_element(x, prop) {
        Some((_, _, inner)) => li_values(inner).into_iter().next(),
        None => find_attr(x, prop).map(|(_, _, v)| xml_unescape(v)),
    }?;
    let v = v.trim().to_string();
    (!v.is_empty()).then_some(v)
}

/// An XMP number: a rational "331/10" or a decimal.
fn xmp_number(x: &str, prop: &str) -> Option<f64> {
    let v = xmp_value(x, prop)?;
    let n = match v.split_once('/') {
        Some((a, b)) => {
            let (a, b) = (a.trim().parse::<f64>().ok()?, b.trim().parse::<f64>().ok()?);
            (b != 0.0).then(|| a / b)?
        }
        None => v.parse::<f64>().ok()?,
    };
    n.is_finite().then_some(n)
}

/// File Info's read-only Camera Data (keys of [`CAMERA_FIELDS`], present ones only), formatted
/// for display. Read from the EXIF block (JPEG APP1, PSD resource 1058, …); properties it lacks
/// fall back to the XMP `tiff:`/`exif:`/`aux:` copies (files without an EXIF block).
pub fn camera_data(exif: Option<&[u8]>, xmp: Option<&str>) -> Value {
    let e = exif.map(photocraft_algo::exif::read).unwrap_or_default();
    let x = xmp.unwrap_or("");
    let text = |v: Option<String>, props: &[&str]| v.or_else(|| props.iter().find_map(|p| xmp_value(x, p)));
    let num =
        |v: Option<f64>, props: &[&str]| v.filter(|v| v.is_finite() && *v > 0.0).or_else(|| props.iter().find_map(|p| xmp_number(x, p))).filter(|v| *v > 0.0);
    let values = [
        text(e.make, &["tiff:Make"]),
        text(e.model, &["tiff:Model"]),
        text(e.date_taken, &["exif:DateTimeOriginal"]).map(|d| date_text(&d)),
        num(e.exposure_time, &["exif:ExposureTime"]).and_then(exposure_text),
        num(e.f_number, &["exif:FNumber"]).map(|n| format!("f/{}", short_num(n, 1))),
        num(e.iso, &["exif:ISOSpeedRatings", "exifEX:PhotographicSensitivity"]).map(|n| format!("ISO {}", n.round())),
        num(e.focal_length, &["exif:FocalLength"]).map(|f| format!("{} mm", short_num(f, 1))),
        num(e.focal_length_35mm, &["exif:FocalLengthIn35mmFilm"]).map(|f| format!("{} mm", short_num(f, 1))),
        text(e.lens, &["exifEX:LensModel", "aux:Lens"]),
    ];
    let mut m = serde_json::Map::new();
    for ((key, _), v) in CAMERA_FIELDS.into_iter().zip(values) {
        // Hostile files can carry megabytes of "lens name"; the dialog shows a line.
        if let Some(v) = v.map(|v| v.chars().take(200).collect::<String>()).filter(|v| !v.is_empty()) {
            m.insert(key.into(), json!(v));
        }
    }
    Value::Object(m)
}

/// Rewrites only changed File Info fields of an XMP packet (creating one when there is none),
/// keeping the other properties verbatim. Edited properties move into their own
/// `rdf:Description` (several descriptions per packet are valid XMP).
pub fn write_file_info(xmp: Option<&str>, info: &Value) -> String {
    let cur = read_file_info(xmp);
    // The dialog submits all fields, including unchanged language alternatives.
    let changed = |key: &str| info.get(key).is_some_and(|v| cur.get(key) != Some(v));
    let mut x = xmp.filter(|s| s.contains("</rdf:RDF>")).map(str::to_string).unwrap_or_else(|| {
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n<x:xmpmeta xmlns:x=\"adobe:ns:meta/\">\n <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n </rdf:RDF>\n</x:xmpmeta>\n<?xpacket end=\"w\"?>".to_string()
    });
    for (key, prop, _) in INFO_FIELDS.into_iter().chain([("copyrightStatus", "xmpRights:Marked", ""), ("copyrightUrl", "xmpRights:WebStatement", "")]) {
        if !changed(key) {
            continue;
        }
        while let Some((a, b, _)) = find_element(&x, prop) {
            x.replace_range(a..b, "");
        }
        while let Some((a, b, _)) = find_attr(&x, prop) {
            x.replace_range(a..b, "");
        }
    }
    let get = |k: &str| info.get(k).filter(|_| changed(k)).cloned().unwrap_or(Value::Null);
    let mut body = String::new();
    for (key, prop, kind) in INFO_FIELDS {
        let vals: Vec<String> = match get(key) {
            Value::Array(a) => a.iter().filter_map(|v| v.as_str()).map(str::to_string).filter(|v| !v.is_empty()).collect(),
            Value::String(s) if kind == "Bag" || kind == "Seq" => s.split([';', ',']).map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect(),
            Value::String(s) if !s.is_empty() => vec![s],
            _ => Vec::new(),
        };
        if vals.is_empty() {
            continue;
        }
        let items: String = vals
            .iter()
            .map(|v| {
                if kind == "Alt" { format!("<rdf:li xml:lang=\"x-default\">{}</rdf:li>", xml_escape(v)) } else { format!("<rdf:li>{}</rdf:li>", xml_escape(v)) }
            })
            .collect();
        if kind.is_empty() {
            body.push_str(&format!("   <{prop}>{}</{prop}>\n", xml_escape(&vals[0])));
        } else {
            body.push_str(&format!("   <{prop}><rdf:{kind}>{items}</rdf:{kind}></{prop}>\n"));
        }
    }
    match get("copyrightStatus").as_str() {
        Some("copyrighted") => body.push_str("   <xmpRights:Marked>True</xmpRights:Marked>\n"),
        Some("publicDomain") => body.push_str("   <xmpRights:Marked>False</xmpRights:Marked>\n"),
        _ => {}
    }
    if let Some(u) = get("copyrightUrl").as_str().filter(|u| !u.is_empty()) {
        body.push_str(&format!("   <xmpRights:WebStatement>{}</xmpRights:WebStatement>\n", xml_escape(u)));
    }
    if !body.is_empty() {
        let desc = format!(
            "  <rdf:Description rdf:about=\"\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\" xmlns:xmpRights=\"http://ns.adobe.com/xap/1.0/rights/\">\n{body}  </rdf:Description>\n "
        );
        let at = x.rfind("</rdf:RDF>").unwrap_or(x.len());
        x.insert_str(at, &desc);
    }
    x
}

fn file_info(s: &mut Session, p: &Value) -> Result<Value> {
    let keys = ["title", "author", "authorTitle", "description", "keywords", "copyright", "copyrightStatus", "copyrightUrl"];
    let edits = p.as_object().is_some_and(|m| m.keys().any(|k| keys.contains(&k.as_str())));
    if edits {
        let xmp = s.active().ok_or(EngineError::NoDocument)?.doc.metadata.xmp.clone();
        let new = write_file_info(xmp.as_deref(), p);
        s.edit("File Info", |doc, _| {
            doc.metadata.xmp = Some(new);
            Ok(())
        })?;
    }
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let mut info = read_file_info(d.doc.metadata.xmp.as_deref());
    if let Value::Object(m) = &mut info {
        m.insert("camera".into(), camera_data(d.doc.metadata.exif.as_deref().map(Vec::as_slice), d.doc.metadata.xmp.as_deref()));
    }
    Ok(info)
}

// ---------- Automate ----------

fn fit_image(s: &mut Session, p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let (w, h) = (d.doc.size.width as f64, d.doc.size.height as f64);
    let bw = f64_param(p, "width").unwrap_or(w).max(1.0);
    let bh = f64_param(p, "height").unwrap_or(h).max(1.0);
    let k = (bw / w).min(bh / h);
    if (k > 1.0 && p.get("dontEnlarge").and_then(Value::as_bool).unwrap_or(false)) || ((w * k).round() == w && (h * k).round() == h) {
        return Ok(json!({"width": w, "height": h, "changed": false}));
    }
    let (nw, nh) = ((w * k).round().max(1.0), (h * k).round().max(1.0));
    let resample = p.get("resample").and_then(Value::as_str).unwrap_or("bicubic");
    s.execute("image.imageSize", json!({"width": nw, "height": nh, "resample": resample}))?;
    Ok(json!({"width": nw, "height": nh, "changed": true}))
}

fn mode_name(m: ColorMode) -> &'static str {
    match m {
        ColorMode::Bitmap => "bitmap",
        ColorMode::Grayscale => "grayscale",
        ColorMode::Duotone => "duotone",
        ColorMode::Indexed => "indexed",
        ColorMode::Rgb => "rgb",
        ColorMode::Cmyk => "cmyk",
        ColorMode::Lab => "lab",
        ColorMode::Multichannel => "multichannel",
    }
}

fn conditional_mode_change(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.automate.conditionalModeChange";
    let to = str_param(p, "to", cmd)?.to_ascii_lowercase();
    let to = match to.as_str() {
        "gray" | "grayscale" => "grayscale",
        "rgb" => "rgb",
        "cmyk" => "cmyk",
        "lab" => "lab",
        other => return Err(EngineError::BadParams { cmd: cmd.into(), msg: format!("unsupported target mode \"{other}\" (rgb, grayscale, cmyk, lab)") }),
    };
    let cur = mode_name(s.active().ok_or(EngineError::NoDocument)?.doc.mode);
    let from: Vec<String> = match p.get("from") {
        Some(Value::Array(a)) => {
            a.iter().filter_map(Value::as_str).map(|v| v.to_ascii_lowercase().replace("gray", "grayscale").replace("grayscalescale", "grayscale")).collect()
        }
        Some(Value::String(v)) if v != "any" => vec![v.to_ascii_lowercase()],
        _ => vec!["any".into()],
    };
    if cur == to || !(from.iter().any(|f| f == "any" || f == cur)) {
        return Ok(json!({"changed": false, "mode": cur}));
    }
    s.execute(&format!("image.mode.{to}"), json!({}))?;
    Ok(json!({"changed": true, "mode": to}))
}

/// One `[id, params]` / `{command, params}` / `{id, params}` step of an action.
fn parse_steps(v: &Value) -> Result<Vec<(String, Value)>> {
    let bad = |m: &str| EngineError::BadParams { cmd: "file.automate.batch".into(), msg: m.into() };
    let arr = v.as_array().ok_or_else(|| bad("\"steps\" must be an array"))?;
    arr.iter()
        .map(|st| match st {
            Value::Array(a) if !a.is_empty() => {
                Ok((a[0].as_str().ok_or_else(|| bad("step id must be a string"))?.to_string(), a.get(1).cloned().unwrap_or(json!({}))))
            }
            Value::Object(o) => {
                let id = o.get("command").or_else(|| o.get("id")).and_then(Value::as_str).ok_or_else(|| bad("step needs \"command\""))?;
                Ok((id.to_string(), o.get("params").cloned().unwrap_or(json!({}))))
            }
            Value::String(id) => Ok((id.clone(), json!({}))),
            _ => Err(bad("each step is [id, params] or {\"command\", \"params\"}")),
        })
        .collect()
}

/// Inputs of a batch command: `"input"` is a folder or an array of files.
pub(crate) fn batch_inputs(p: &Value, cmd: &str) -> Result<Vec<String>> {
    match p.get("input").or_else(|| p.get("files")) {
        Some(Value::Array(a)) => Ok(a.iter().filter_map(Value::as_str).map(str::to_string).collect()),
        Some(Value::String(dir)) => list_images(dir),
        _ => Err(EngineError::BadParams { cmd: cmd.into(), msg: "missing \"input\" (folder or array of files)".into() }),
    }
}

/// Opens each input in a scratch session, runs `f` on it and saves it to `output` as `format`
/// (`"same"` keeps the input's extension). Errors per file are collected, not fatal.
pub(crate) fn process_files(inputs: &[String], output: &str, format: &str, save: SaveOpts, suffix: &str, f: &dyn Fn(&mut Session) -> Result<()>) -> Value {
    let outputs = inputs.iter().map(|path| {
        let ext =
            if format == "same" { path.rsplit('.').next().unwrap_or("png").to_ascii_lowercase() } else { format.trim_start_matches('.').to_ascii_lowercase() };
        (path.as_str(), join(output, &format!("{}{suffix}.{ext}", stem(path))))
    });
    process_outputs(outputs, save, f)
}

fn process_outputs<'a>(outputs: impl IntoIterator<Item = (&'a str, String)>, save: SaveOpts, f: &dyn Fn(&mut Session) -> Result<()>) -> Value {
    let mut files = Vec::new();
    let mut errors = Vec::new();
    let mut written = OutputClaims::default();
    for (path, out) in outputs {
        let r = (|| -> Result<()> {
            written.check(&out).map_err(EngineError::Other)?;
            let bytes = read_file(path)?;
            let mut scratch = Session::new();
            let doc = import(&file_name(path), &bytes)?;
            scratch.add_document(doc, Some(path.to_string()));
            f(&mut scratch)?;
            let d = scratch.active().ok_or(EngineError::NoDocument)?;
            save_doc(&d.doc, &out, save)?;
            Ok(())
        })();
        match r {
            Ok(()) => {
                written.record(&out, path);
                files.push(out);
            }
            Err(e) => errors.push(json!({"file": path, "error": e.to_string()})),
        }
    }
    json!({"files": files, "errors": errors})
}

/// The output paths a batch run has written, so a later input whose output name matches an
/// earlier one's (`a.png` and `a.jpg` saved as JPEG, or the same name in two input folders) is
/// reported instead of silently replacing that result (#420, #422). Names are compared ignoring
/// case, because macOS and Windows file systems do.
#[derive(Default)]
pub struct OutputClaims(std::collections::HashMap<String, String>);

impl OutputClaims {
    /// `Err` (naming the earlier input) when `out` was already written in this run.
    pub fn check(&self, out: &str) -> std::result::Result<(), String> {
        match self.0.get(&out.to_lowercase()) {
            Some(first) => Err(format!("not written: {out} already holds the result of {first} from this run (same output name)")),
            None => Ok(()),
        }
    }
    /// Remember that `input`'s result was written to `out`.
    pub fn record(&mut self, out: &str, input: &str) {
        self.0.insert(out.to_lowercase(), input.to_string());
    }
}

fn batch(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.automate.batch";
    let mut steps =
        parse_steps(p.get("steps").or_else(|| p.get("action")).ok_or_else(|| EngineError::BadParams { cmd: cmd.into(), msg: "missing \"steps\"".into() })?)?;
    // Each result goes to the destination folder, which overrides the action's recorded Save /
    // Save As steps (Photoshop's "Override Action 'Save As' Commands"). Recorded view steps
    // (zoom, fit) don't change the output, so they are skipped (#2752).
    steps.retain(|(id, _)| !crate::actions_cmds::shell_save_command(id) && !crate::actions_cmds::shell_view_command(id) && id != "file.export.quickExport");
    if let Some((id, _)) = steps.iter().find(|(id, _)| crate::commands::find(id).is_none()) {
        return Err(EngineError::BadParams { cmd: cmd.into(), msg: format!("unknown command `{id}` in the action") });
    }
    let inputs = batch_inputs(p, cmd)?;
    let output = str_param(p, "output", cmd)?.to_string();
    let format = p.get("format").and_then(Value::as_str).unwrap_or("same").to_string();
    // Steps that call another action play it from the caller's actions; the playback stack comes
    // along so an action that batches itself is caught as a recursive call.
    let list = match p.get("calledActions") {
        None | Some(Value::Null) => s.actions.list.clone(),
        // A droplet brings the actions it calls (#2794) and runs with those alone.
        Some(v) => {
            let list: Vec<crate::actions_cmds::Action> =
                serde_json::from_value(v.clone()).map_err(|e| EngineError::BadParams { cmd: cmd.into(), msg: format!("\"calledActions\": {e}") })?;
            if list.len() > 4096 {
                return Err(EngineError::BadParams { cmd: cmd.into(), msg: "\"calledActions\" holds more than 4096 actions".into() });
            }
            list
        }
    };
    let actions = crate::actions_cmds::ActionState { list, playing: s.actions.playing, playback_stack: s.actions.playback_stack.clone(), ..Default::default() };
    let names = batch_naming::plan(&inputs, &output, &format, p)?;
    let save = SaveOpts::from_params(s, p, cmd)?;
    let run = |scratch: &mut Session| {
        scratch.actions = actions.clone();
        for (id, params) in &steps {
            let r = scratch.execute(id, params.clone())?;
            // A failed step inside a called action makes this file an error, not a saved result.
            if let Some(error) = crate::actions_cmds::nested_failure(id, &r) {
                return Err(EngineError::Other(error));
            }
        }
        Ok(())
    };
    let r = match names {
        Some(names) => process_outputs(inputs.iter().map(String::as_str).zip(names), save, &run),
        None => process_files(&inputs, &output, &format, save, "", &run),
    };
    Ok(r)
}

fn image_processor(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.scripts.imageProcessor";
    let inputs = batch_inputs(p, cmd)?;
    let output = str_param(p, "output", cmd)?.to_string();
    let format = p.get("format").and_then(Value::as_str).unwrap_or("jpg").to_string();
    let fit = match (size_param(p, "width"), size_param(p, "height")) {
        (None, None) => None,
        (w, h) => Some(json!({"width": w.unwrap_or(1e9), "height": h.unwrap_or(1e9), "dontEnlarge": true})),
    };
    let to_srgb = p.get("convertToSrgb").and_then(Value::as_bool).unwrap_or(false);
    let r = process_files(&inputs, &output, &format, SaveOpts::from_params(s, p, cmd)?.or_quality(8.0), "", &|scratch| {
        if to_srgb && scratch.active().is_some_and(|d| d.doc.mode != ColorMode::Rgb) {
            scratch.execute("image.mode.rgb", json!({}))?;
        }
        if let Some(fp) = &fit {
            fit_image(scratch, fp)?;
        }
        Ok(())
    });
    Ok(r)
}

fn load_files_into_stack(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.scripts.loadFilesIntoStack";
    let paths: Vec<String> = match p.get("paths").or_else(|| p.get("input")) {
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(str::to_string).collect(),
        Some(Value::String(dir)) => list_images(dir)?,
        _ => return Err(EngineError::BadParams { cmd: cmd.into(), msg: "missing \"paths\"".into() }),
    };
    if paths.is_empty() {
        return Err(EngineError::BadParams { cmd: cmd.into(), msg: "no files to load".into() });
    }
    let mut docs = Vec::new();
    for path in &paths {
        let bytes = read_file(path)?;
        docs.push((file_name(path), import(&file_name(path), &bytes)?));
    }
    let first = &docs[0].1;
    let w = docs.iter().map(|(_, d)| d.size.width).max().unwrap_or(1);
    let h = docs.iter().map(|(_, d)| d.size.height).max().unwrap_or(1);
    let mut stack = Document::new(stem(&paths[0]), photocraft_doc::Size::new(w, h), first.mode, first.depth);
    stack.resolution_dpi = first.resolution_dpi;
    stack.icc_profile = first.icc_profile.clone();
    let fmt = stack.pixel_format();
    // First file at the bottom, like Photoshop's script.
    for (name, d) in &docs {
        stack.layers.push(Layer::new(name.clone(), LayerContent::Raster(flattened(d, fmt))));
    }
    let n = stack.layers.len();
    let smart = p.get("createSmartObject").and_then(Value::as_bool).unwrap_or(false);
    // "Attempt to Automatically Align Source Images": Auto-Align Layers on the loaded layers, before
    // they go into the smart object.
    let align = p.get("align").and_then(Value::as_bool).unwrap_or(false) && n > 1;
    if smart && !align {
        // "Create Smart Object after Loading Layers": the layers go inside one smart object, ready
        // for Layer › Smart Objects › Stack Mode. Like Photoshop, they sit at the top level of the
        // contents, with no wrapper group.
        let layers = std::mem::take(&mut stack.layers);
        let smart = crate::smart_cmds::layers_to_smart(&stack, &stem(&paths[0]), layers)?;
        stack.layers = vec![smart];
    }
    let i = s.add_document(stack, None);
    if align {
        s.execute("select.allLayers", json!({}))?;
        s.execute("edit.autoAlignLayers", json!({"projection": "auto"}))?;
        if smart {
            let name = s.active().map(|d| d.doc.name.clone()).unwrap_or_default();
            s.edit("Convert to Smart Object", |doc, active| {
                let layers = std::mem::take(&mut doc.layers);
                let smart = crate::smart_cmds::layers_to_smart(doc, &name, layers)?;
                *active = Some(smart.id);
                doc.layers = vec![smart];
                Ok(())
            })?;
            if let Some(st) = s.active_mut() {
                st.selected_layers = st.active_layer.into_iter().collect();
            }
        }
    }
    Ok(json!({"document": i, "layers": n}))
}

// ---------- Scripts: flatten effects / masks ----------

fn has_live_effects(l: &Layer) -> bool {
    l.effects.enabled && l.effects.items.iter().any(|e| e.enabled())
}

/// The layer rendered alone (keeping what `keep` says), as a surface in `fmt`. With `apply_mask`
/// off the layer's pixel and vector masks are left out of the render, so a caller that keeps them
/// live on the baked raster does not mask the pixels twice.
fn bake(l: &Layer, canvas: Rect, fmt: PixelFormat, keep_effects: bool, apply_mask: bool) -> Surface {
    let mut tmp = l.clone();
    tmp.opacity = 1.0;
    tmp.blend = photocraft_color::BlendMode::Normal;
    tmp.visible = true;
    tmp.clipped = false;
    if !keep_effects {
        tmp.effects.items.clear();
        tmp.fill_opacity = 1.0;
    }
    if !apply_mask {
        tmp.mask = None;
        tmp.vector_mask = None;
    }
    // Render over the union of the canvas and the content so off-canvas pixels survive.
    let area = l.surface().map_or(canvas, |s| s.content_bounds().union(&canvas));
    let area = if keep_effects { area.inflate(256) } else { area };
    buffer_surface(&photocraft_compose::render_layer(&tmp, area), fmt)
}

fn walk_ids(doc: &Document) -> Vec<LayerId> {
    doc.walk().into_iter().map(|(_, _, l)| l.id).collect()
}

fn flatten_all_effects(s: &mut Session) -> Result<Value> {
    let mut n = 0;
    s.edit("Flatten All Layer Effects", |doc, _| {
        let canvas = doc.bounds();
        let fmt = doc.pixel_format();
        for id in walk_ids(doc) {
            let Some(l) = doc.layer(id) else { continue };
            if !has_live_effects(l) || matches!(l.content, LayerContent::Group(_) | LayerContent::Adjustment(_)) {
                continue;
            }
            // The pixel and vector masks stay on the layer (baking must not destroy them), so the
            // pixels are baked without them and each mask then applies once to the baked raster.
            let px = bake(l, canvas, fmt, true, false);
            let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            l.content = LayerContent::Raster(px);
            l.effects = Default::default();
            l.fill_opacity = 1.0;
            l.fill_cache = None;
            l.psd_blocks.retain(|(k, _)| !matches!(k, b"TySh" | b"SoLd" | b"PlLd" | b"SoLE" | b"vmsk" | b"vsms" | b"vogk" | b"vscg" | b"vstk"));
            n += 1;
        }
        Ok(())
    })?;
    Ok(json!({"flattened": n}))
}

fn flatten_all_masks(s: &mut Session) -> Result<Value> {
    let (mut n, mut skipped) = (0, 0);
    s.edit("Flatten All Masks", |doc, _| {
        let canvas = doc.bounds();
        let fmt = doc.pixel_format();
        for id in walk_ids(doc) {
            let Some(l) = doc.layer(id) else { continue };
            let masked = l.mask.as_ref().is_some_and(|m| m.enabled) || l.vector_mask.is_some();
            if !masked {
                continue;
            }
            if !matches!(l.content, LayerContent::Raster(_)) {
                // Masks on groups, adjustment, type, shape and smart layers have no pixels to
                // fold into; Photoshop's script leaves them too.
                skipped += 1;
                continue;
            }
            let px = bake(l, canvas, fmt, false, true);
            let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
            l.content = LayerContent::Raster(px);
            l.mask = None;
            l.vector_mask = None;
            l.psd_blocks.retain(|(k, _)| !matches!(k, b"vmsk" | b"vsms"));
            n += 1;
        }
        Ok(())
    })?;
    Ok(json!({"flattened": n, "skipped": skipped}))
}

// ---------- Export ----------

fn layers_to_files(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "file.export.layersToFiles";
    let dir = str_param(p, "dir", cmd).or_else(|_| str_param(p, "output", cmd))?.to_string();
    let names = export_layers_with(s, p, |name, bytes| write_file(&join(&dir, name), bytes))?;
    let files: Vec<String> = names.iter().map(|name| join(&dir, name)).collect();
    Ok(json!({"files": files}))
}

/// The bodies of `file.export.layersToFiles`, streamed to `write` as a single-component filename
/// plus its bytes: one file per exported layer (top to bottom, like Photoshop's numbering),
/// encoded but never touched as a path. Automation writes exactly these bytes beneath its write
/// root (#3216), so the two paths cannot drift. Returns the filenames written, in order.
pub fn export_layers_with(s: &Session, p: &Value, mut write: impl FnMut(&str, &[u8]) -> Result<()>) -> Result<Vec<String>> {
    let cmd = "file.export.layersToFiles";
    let format = p.get("format").and_then(Value::as_str).unwrap_or("png").trim_start_matches('.').to_ascii_lowercase();
    let visible_only = p.get("visibleOnly").and_then(Value::as_bool).unwrap_or(true);
    let save = SaveOpts::from_params(s, p, cmd)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let prefix = p.get("prefix").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stem(&d.doc.name));
    let doc = d.doc.clone();
    let mut written = Vec::new();
    // Top-level layers (groups export merged), top to bottom like Photoshop's numbering.
    let layers: Vec<&Layer> = doc.layers.iter().rev().filter(|l| l.visible || !visible_only).collect();
    for (i, l) in layers.iter().enumerate() {
        if matches!(l.content, LayerContent::Adjustment(_)) {
            continue;
        }
        let mut one = (*doc).clone();
        let mut only = (*l).clone();
        only.visible = true;
        only.clipped = false;
        one.layers = vec![only];
        let name = format!("{}_{:04}_{}.{format}", sanitize(&prefix), i, sanitize(&l.name));
        let (bytes, _warnings) = encode(&one, &name, save)?;
        write(&name, &bytes)?;
        written.push(name);
    }
    Ok(written)
}

/// Adjustment layers that can be represented as an RGB-only LUT, in document stack order.
/// A LUT has no spatial coordinates, so masks and clipping geometry are deliberately stripped
/// at bake time. Group adjustments need their enclosing group's compositing semantics and are
/// not silently flattened as independent layers.
fn lut_layers<'a>(doc: &'a Document, ids: Option<&[LayerId]>) -> Result<Vec<&'a Layer>> {
    const CMD: &str = "file.export.colorLookupTables";
    let mut selected = Vec::new();
    if let Some(ids) = ids {
        if ids.is_empty() {
            return Err(EngineError::BadParams { cmd: CMD.into(), msg: "select at least one adjustment layer".into() });
        }
        for id in ids {
            if selected.contains(id) {
                return Err(EngineError::BadParams { cmd: CMD.into(), msg: format!("duplicate layer id {}", id.0) });
            }
            let Some(layer) = doc.layer(*id) else { return Err(EngineError::NoLayer(*id)) };
            if !matches!(layer.content, LayerContent::Adjustment(_)) {
                return Err(EngineError::BadParams { cmd: CMD.into(), msg: format!("layer {} is not an adjustment layer", id.0) });
            }
            if !layer.visible {
                return Err(EngineError::BadParams { cmd: CMD.into(), msg: format!("layer {} is hidden", id.0) });
            }
            if !doc.layers.iter().any(|root| root.id == *id) {
                return Err(EngineError::BadParams {
                    cmd: CMD.into(),
                    msg: format!("layer {} is nested in a group; export top-level adjustment layers", id.0),
                });
            }
            selected.push(*id);
        }
    }
    let mut layers = Vec::new();
    for l in &doc.layers {
        if l.visible && matches!(l.content, LayerContent::Adjustment(_)) && (ids.is_none() || selected.contains(&l.id)) {
            layers.push(l);
        }
    }
    if layers.is_empty() {
        return Err(EngineError::BadParams { cmd: CMD.into(), msg: "no visible adjustment layers to export".into() });
    }
    Ok(layers)
}

/// Bake the chosen adjustment stack against an identity RGB lattice.
/// Source order always follows the document stack, even if an explicit list is reversed.
fn bake_cube_layers(size: usize, title: &str, layers: &[&Layer]) -> String {
    let n = size;
    let (w, h) = ((n * n) as u32, n as u32);
    let mut lattice = Document::new("lut", photocraft_doc::Size::new(w, h), ColorMode::Rgb, photocraft_color::SampleType::F32);
    let fmt = lattice.pixel_format();
    let step = 1.0 / (n - 1) as f32;
    let mut data = Vec::with_capacity((w * h) as usize * fmt.channels());
    for g in 0..n {
        for b in 0..n {
            for r in 0..n {
                let mut px = [0.0f32; 8];
                let k = from_rgba_into(&fmt, [r as f32 * step, g as f32 * step, b as f32 * step, 1.0], &mut px);
                data.extend_from_slice(&px[..k]);
            }
        }
    }
    let mut surf = Surface::new(fmt);
    surf.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
    lattice.layers.push(Layer::new("Lattice", LayerContent::Raster(surf)));
    for l in layers {
        let mut a = (*l).clone();
        // Masks and clipping are spatial; a LUT is the adjustment stack's colour mapping.
        a.mask = None;
        a.vector_mask = None;
        a.clipped = false;
        lattice.layers.push(a);
    }
    let out = photocraft_compose::flatten(&lattice);
    let mut s = format!("TITLE \"{}\"\n# Created by Photocraft\nLUT_3D_SIZE {n}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n", title.replace('"', "'"));
    // .cube order: red changes fastest, then green, then blue.
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let (x, y) = (b * n + r, g);
                let px = out.px[y * w as usize + x];
                s.push_str(&format!("{:.6} {:.6} {:.6}\n", px[0].clamp(0.0, 1.0), px[1].clamp(0.0, 1.0), px[2].clamp(0.0, 1.0)));
            }
        }
    }
    s
}

/// Backwards-compatible helper for exporting all visible top-level adjustments.
pub fn bake_cube(doc: &Document, size: usize, title: &str) -> String {
    let layers: Vec<&Layer> = doc.layers.iter().filter(|l| l.visible && matches!(l.content, LayerContent::Adjustment(_))).collect();
    bake_cube_layers(size.clamp(2, photocraft_cms::lutfile::MAX_SIZE), title, &layers)
}

/// Export either the whole visible stack (backward-compatible default), the Layers panel's
/// current selection, or an explicit list supplied by scripts/CLI/automation.
fn color_lookup_tables(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "file.export.colorLookupTables";
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let size = match p.get("size") {
        None | Some(Value::Null) => 33,
        Some(v) => v.as_u64().ok_or_else(|| EngineError::BadParams { cmd: CMD.into(), msg: "size must be an integer".into() })?,
    };
    if !(2..=photocraft_cms::lutfile::MAX_SIZE as u64).contains(&size) {
        return Err(EngineError::BadParams { cmd: CMD.into(), msg: format!("size must be 2..={}", photocraft_cms::lutfile::MAX_SIZE) });
    }
    let scope = match p.get("scope") {
        None | Some(Value::Null) => "all",
        Some(Value::String(s)) => s.as_str(),
        Some(_) => return Err(EngineError::BadParams { cmd: CMD.into(), msg: "scope must be all or selected".into() }),
    };
    if !matches!(scope, "all" | "selected") {
        return Err(EngineError::BadParams { cmd: CMD.into(), msg: "scope must be all or selected".into() });
    }
    if p.get("layers").is_some() && scope == "selected" {
        return Err(EngineError::BadParams { cmd: CMD.into(), msg: "use either layers or scope=selected, not both".into() });
    }
    let explicit: Option<Vec<LayerId>> = if let Some(v) = p.get("layers") {
        let Some(a) = v.as_array() else { return Err(EngineError::BadParams { cmd: CMD.into(), msg: "layers must be an array of layer IDs".into() }) };
        let mut out = Vec::with_capacity(a.len());
        for id in a {
            let Some(id) = id.as_u64() else { return Err(EngineError::BadParams { cmd: CMD.into(), msg: "layer IDs must be unsigned integers".into() }) };
            out.push(LayerId(id));
        }
        Some(out)
    } else if scope == "selected" {
        Some(d.selected_layers())
    } else {
        None
    };
    let layers = lut_layers(&d.doc, explicit.as_deref())?;
    let count = layers.len();
    let title = p.get("title").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| stem(&d.doc.name));
    let cube = bake_cube_layers(size as usize, &title, &layers);
    match p.get("path").and_then(Value::as_str) {
        Some(path) => {
            write_file(path, cube.as_bytes())?;
            Ok(json!({"path": path, "size": size, "layerCount": count}))
        }
        None => Ok(json!({"cube": cube, "size": size, "layerCount": count})),
    }
}

// ---------- guides ----------

fn push_unique(v: &mut Vec<f64>, x: f64) {
    if !v.iter().any(|g| (g - x).abs() < 0.01) {
        v.push(x);
    }
}

fn new_guide_layout(s: &mut Session, p: &Value) -> Result<Value> {
    crate::guide_layout::apply(s, p)
}

fn guides_from_shape(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, p)?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let l = d.doc.layer(id).ok_or(EngineError::NoLayer(id))?;
    let LayerContent::Shape(sh) = &l.content else { return Err(EngineError::Other("the layer is not a shape layer".into())) };
    let (x0, y0, x1, y1) = sh.path.control_bounds().ok_or(EngineError::Other("the shape has no path".into()))?;
    s.edit("New Guides From Shape", |doc, _| {
        for x in [x0, (x0 + x1) / 2.0, x1] {
            push_unique(&mut doc.guides.vertical, x);
        }
        for y in [y0, (y0 + y1) / 2.0, y1] {
            push_unique(&mut doc.guides.horizontal, y);
        }
        Ok(())
    })?;
    Ok(json!({"bounds": [x0, y0, x1, y1]}))
}

fn clear_canvas_guides(s: &mut Session) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    // Guides inside an artboard belong to it (View › Clear Selected Artboard Guides); the rest
    // are canvas guides. Without artboards every guide is a canvas guide.
    let boards: Vec<Rect> = d.doc.artboards().iter().map(|b| b.2.rect).collect();
    let in_x = move |x: f64| boards.iter().any(|r| x > f64::from(r.x0) && x < f64::from(r.x1));
    let boards_y: Vec<Rect> = d.doc.artboards().iter().map(|b| b.2.rect).collect();
    let in_y = move |y: f64| boards_y.iter().any(|r| y > f64::from(r.y0) && y < f64::from(r.y1));
    let n = d.doc.guides.vertical.iter().filter(|x| !in_x(**x)).count() + d.doc.guides.horizontal.iter().filter(|y| !in_y(**y)).count();
    if n == 0 {
        return Ok(json!({"cleared": 0}));
    }
    s.edit("Clear Canvas Guides", |doc, _| {
        doc.guides.vertical.retain(|x| in_x(*x));
        doc.guides.horizontal.retain(|y| in_y(*y));
        doc.guides.prune_colors();
        Ok(())
    })?;
    Ok(json!({"cleared": n}))
}

// ---------- registry ----------

pub fn specs() -> Vec<CommandSpec> {
    macro_rules! spec {
        ($id:expr, $label:expr, $menu:expr, $sc:expr, $params:expr, $enabled:expr, $run:expr) => {
            CommandSpec { id: $id, label: $label, menu: $menu, shortcut: $sc, params: $params, enabled: $enabled, journal: true, run: $run }
        };
    }
    vec![
        spec!("file.closeAll", "Close All", &["File"], Some("Cmd+Alt+W"), "{}", has_doc, |s, _| close_all(s)),
        spec!(
            "file.closeOthers",
            "Close Others",
            &["File"],
            Some("Cmd+Alt+P"),
            r##"{"document":index? (the one to keep; default active)}"##,
            has_doc,
            close_others
        ),
        // The document tab's menu (UI-217-6): the tab's own file in the platform file manager.
        // Not a Photoshop menu-bar item; enabled when the target document has a saved path.
        spec!("file.revealInFinder", "Reveal in Finder", &[], None, r##"{"document":index? (default active),"dryRun":bool=false}"##, native_doc, |s, p| {
            let i = match p.get("document") {
                Some(v) => v
                    .as_u64()
                    .and_then(|v| usize::try_from(v).ok())
                    .ok_or_else(|| EngineError::BadParams { cmd: "file.revealInFinder".into(), msg: "`document` must be an index".into() })?,
                None => s.active_index().ok_or(EngineError::NoDocument)?,
            };
            let path = s
                .documents()
                .get(i)
                .and_then(|d| d.path.clone())
                .ok_or_else(|| EngineError::BadParams { cmd: "file.revealInFinder".into(), msg: format!("document {i} has no saved file") })?;
            crate::layer_menu_cmds::reveal(&path, p.get("dryRun").and_then(Value::as_bool).unwrap_or(false))
        }),
        spec!("file.revert", "Revert", &["File"], Some("F12"), "{} (reloads the saved file as one undoable step)", can_revert, |s, _| revert(s)),
        spec!(
            "file.saveACopy",
            "Save a Copy…",
            &["File"],
            Some("Cmd+Alt+S"),
            r##"{"path":str (format from the extension),"quality":0..12? (JPEG),"layers":bool=true,"tiffLayers":bool=false (TIFF: keep the layers; flat by default),"tgaBits":24|32? (TGA: 32 writes the alpha channel, or else the transparency, as the alpha; unset, 32 with an alpha channel or transparency, else 24),"dpxBits":8|10|12|16? (DPX: bits per sample; unset, 8 for an 8-bit document, else 16)}"##,
            native_doc,
            save_a_copy
        ),
        spec!(
            "file.openAs",
            "Open As…",
            &["File"],
            Some("Cmd+Alt+Shift+O"),
            r##"{"path":str,"as":"psd|png|jpg|tiff|…"? (decode as this format)}"##,
            native,
            open_as
        ),
        spec!(
            "file.placeEmbedded",
            "Place Embedded…",
            &["File"],
            None,
            r##"{"path":str,"scale":%? (default: fit when larger than the canvas),"fit":bool? (default: Preferences > General > Resize Image During Place),"smartObject":bool? (default: Preferences > General > Always Create Smart Objects When Placing),"center":[x,y]?}"##,
            native_doc,
            |s, p| place(s, p, false)
        ),
        spec!(
            "file.placeLinked",
            "Place Linked…",
            &["File"],
            None,
            r##"{"path":str,"scale":%?,"fit":bool? (default: Preferences > General > Resize Image During Place),"center":[x,y]?}"##,
            native_doc,
            |s, p| place(s, p, true)
        ),
        spec!(
            "file.fileInfo",
            "File Info…",
            &["File"],
            Some("Cmd+Alt+Shift+I"),
            r##"{"title":str?,"author":str?,"authorTitle":str?,"description":str?,"keywords":[str]|"a; b"?,"copyright":str?,"copyrightStatus":"unknown|copyrighted|publicDomain"?,"copyrightUrl":str?} (no keys: read; the result also has read-only "camera":{"make","model","dateTaken","exposure","fNumber","iso","focalLength","focalLength35mm","lens"}, present keys only)"##,
            has_doc,
            file_info
        ),
        spec!(
            "file.automate.fitImage",
            "Fit Image…",
            &["File", "Automate"],
            None,
            r##"{"width":px,"height":px,"dontEnlarge":bool=false,"resample":"automatic|preserveDetails|bicubicSmoother|bicubicSharper|bicubicAutomatic|bicubic|nearest|bilinear|lanczos"="bicubic"}"##,
            has_doc,
            fit_image
        ),
        spec!(
            "file.automate.conditionalModeChange",
            "Conditional Mode Change…",
            &["File", "Automate"],
            None,
            r##"{"from":["rgb","grayscale","cmyk","lab","indexed","bitmap",…]|"any"="any","to":"rgb|grayscale|cmyk|lab"}"##,
            has_doc,
            conditional_mode_change
        ),
        spec!(
            "file.automate.batch",
            "Batch…",
            &["File", "Automate"],
            None,
            r##"{"steps":[[commandId,params]|{"command":id,"params":{}}…] (a recorded action),"input":folder|[paths],"output":folder,"format":"same|png|jpg|psd|tiff|…"="same","quality":0..12?,"naming":string? (blank keeps source names; {name}, {serial} padded to 3 digits, {date} as YYYY-MM-DD UTC; extension added automatically),"startingSerial":1..4294967295=1 (input position, including failed inputs),"tiffLayers":bool=false,"tgaBits":24|32?,"dpxBits":8|10|12|16? (DPX: bits per sample; unset, 8 for an 8-bit document, else 16),"calledActions":[{"name","steps"}…]? (the actions Play Action steps call; default: the session's)} → {files, errors} (an input whose output name was already written in the run goes to errors)"##,
            native,
            batch
        ),
        spec!(
            "file.scripts.imageProcessor",
            "Image Processor…",
            &["File", "Scripts"],
            None,
            r##"{"input":folder|[paths],"output":folder,"format":"jpg|png|psd|tiff|…"="jpg","quality":0..12=8,"tiffLayers":bool=false,"tgaBits":24|32?,"dpxBits":8|10|12|16? (DPX: bits per sample; unset, 8 for an 8-bit document, else 16),"width":px?,"height":px? (fit, never enlarge),"convertToSrgb":bool=false} → {files, errors} (an input whose output name was already written in the run goes to errors)"##,
            native,
            image_processor
        ),
        spec!(
            "file.scripts.loadFilesIntoStack",
            "Load Files into Stack…",
            &["File", "Scripts"],
            None,
            r##"{"paths":[str]|folder,"align":bool=false (Auto-Align Layers on them first),"createSmartObject":bool=false} → new document with one layer per file (inside one smart object with createSmartObject)"##,
            native,
            load_files_into_stack
        ),
        spec!("file.scripts.flattenAllLayerEffects", "Flatten All Layer Effects", &["File", "Scripts"], None, "{}", has_doc, |s, _| flatten_all_effects(s)),
        spec!(
            "file.scripts.flattenAllMasks",
            "Flatten All Masks",
            &["File", "Scripts"],
            None,
            "{} (pixel layers; masks on other layer kinds are left)",
            has_doc,
            |s, _| flatten_all_masks(s)
        ),
        spec!(
            "file.export.layersToFiles",
            "Layers to Files…",
            &["File", "Export"],
            None,
            r##"{"dir":folder,"format":"png|jpg|psd|tiff|…"="png","prefix":str=document name,"visibleOnly":bool=true,"quality":0..12?,"tiffLayers":bool=false,"tgaBits":24|32?,"dpxBits":8|10|12|16? (DPX: bits per sample; unset, 8 for an 8-bit document, else 16)} → {files}"##,
            native_doc,
            layers_to_files
        ),
        spec!(
            "file.export.colorLookupTables",
            "Color Lookup Tables…",
            &["File", "Export"],
            None,
            r##"{"path":str? (.cube; omit to return the text),"size":2..129=33,"title":str?,"scope":"all|selected"="all","layers":[id,…]?} (selected layer IDs use original document stacking; masks are spatial and omitted)"##,
            has_adjustments,
            color_lookup_tables
        ),
        spec!(
            "view.newGuideLayout",
            "New Guide Layout…",
            &["View"],
            None,
            r##"{"columns":n=0,"width":px?,"gutter":px=0,"rows":n=0,"height":px?,"rowGutter":px=gutter,"margin":px|[top,left,bottom,right]=0,"centerColumns":bool=false,"clearExisting":bool=false,"target":"document|selectedArtboards"="document"}"##,
            has_doc,
            new_guide_layout
        ),
        spec!("view.newGuidesFromShape", "New Guides From Shape", &["View"], None, r##"{"layer":id?}"##, has_shape, guides_from_shape),
        spec!("view.clearCanvasGuides", "Clear Canvas Guides", &["View"], None, "{}", has_doc, |s, _| clear_canvas_guides(s)),
    ]
}

#[cfg(test)]
#[path = "file_cmds/tests.rs"]
mod tests;
