//! Layer › Video Layers (and Rasterize › Video). A video layer holds a stack of frames; the
//! layer's displayed raster content is the frame at the timeline playhead, kept in sync by [`sync`]
//! after any edit or navigation. Paint, fills and filters edit that displayed content, so [`store`]
//! writes it back into the playhead's frame before anything re-syncs. Headless and scriptable. Frame persistence in `.pcraft` is a
//! follow-up; a saved document keeps the current frame as the layer's content.

use photocraft_doc::{Document, Layer, LayerContent, Timeline, VideoData};
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

/// Set every video layer's displayed content to its frame at the playhead.
pub fn sync(doc: &mut Document) {
    let cur = doc.timeline.as_ref().map_or(0, |t| t.current);
    sync_layers(&mut doc.layers, cur);
}

fn sync_layers(layers: &mut [Layer], cur: usize) {
    for l in layers.iter_mut() {
        let frame = l.video.as_ref().and_then(|v| {
            let visible = if v.show_altered { &v.frames } else { v.original_frames.as_ref().unwrap_or(&v.frames) };
            visible.get(cur.min(visible.len().saturating_sub(1))).cloned()
        });
        if let Some(f) = frame
            && let LayerContent::Raster(s) = &mut l.content
        {
            *s = f;
        }
        if let Some(ch) = l.children_mut() {
            sync_layers(ch, cur);
        }
    }
}

/// Keep edits made to every video layer's displayed content (paint, fills, filters) by writing it
/// back into its frame at the playhead. Call before moving the playhead or re-syncing.
pub fn store(doc: &mut Document) {
    let cur = doc.timeline.as_ref().map_or(0, |t| t.current);
    store_layers(&mut doc.layers, cur);
}

fn store_layers(layers: &mut [Layer], cur: usize) {
    for l in layers.iter_mut() {
        if let (Some(v), LayerContent::Raster(s)) = (&mut l.video, &l.content) {
            let at = cur.min(v.frames.len().saturating_sub(1));
            // When showing the original, the layer's displayed surface is a preview,
            // not an edit to write over the altered frame (#3060).
            if (v.show_altered || v.original_frames.is_none())
                && let Some(f) = v.frames.get_mut(at)
            {
                *f = s.clone();
            }
        }
        if let Some(ch) = l.children_mut() {
            store_layers(ch, cur);
        }
    }
}

/// The active layer, if it's a video layer.
fn active_video(s: &Session) -> std::result::Result<(), String> {
    let st = s.active().ok_or("no document")?;
    match st.active_layer.and_then(|id| st.doc.layer(id)) {
        Some(l) if l.video.is_some() => Ok(()),
        _ => Err("select a video layer".into()),
    }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document".into())
}

/// Grow the timeline to at least `frames` frames (never shrink it). Only a work area that reached
/// the old end follows the new end; a narrowed one stays where the user put it (#4170).
fn grow_duration(t: &mut Timeline, frames: usize) {
    let old = t.duration;
    t.duration = t.duration.max(frames.max(1));
    if t.work_end >= old {
        t.work_end = t.duration;
    }
    t.clamp();
}

/// Edit the active video layer's [`VideoData`] at the playhead, then re-sync the displayed frame.
fn edit_video(s: &mut Session, label: &str, f: impl FnOnce(&mut VideoData, usize) -> Result<()>) -> Result<Value> {
    let count = s.edit(label, |doc, active| {
        store(doc);
        let cur = doc.timeline.as_ref().map_or(0, |t| t.current);
        let id = active.ok_or(EngineError::NoDocument)?;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        let v = l.video.as_mut().ok_or_else(|| EngineError::BadParams { cmd: label.into(), msg: "the active layer is not a video layer".into() })?;
        f(v, cur)?;
        let n = v.frames.len();
        if let Some(t) = &mut doc.timeline {
            grow_duration(t, n);
        }
        sync(doc);
        Ok(n)
    })?;
    Ok(json!({"frames": count}))
}

fn blank_of(fmt: photocraft_color::PixelFormat) -> Surface {
    Surface::new(fmt)
}

fn new_blank(s: &mut Session, _p: &Value) -> Result<Value> {
    let id = s.edit("New Blank Video Layer", |doc, active| {
        store(doc);
        let fmt = doc.pixel_format();
        if doc.timeline.is_none() {
            doc.timeline = Some(Timeline::new(1, 30.0));
        }
        let fps = doc.timeline.as_ref().map_or(30.0, |t| t.fps);
        let n = doc.timeline.as_ref().map_or(1, |t| t.duration).max(1);
        let frames: Vec<Surface> = (0..n).map(|_| Surface::new(fmt)).collect();
        let name = doc.next_layer_name("Video Layer");
        let mut l = Layer::raster(name, fmt);
        l.video = Some(VideoData::new(frames, fps));
        let id = doc.insert_above(*active, l);
        *active = Some(id);
        sync(doc);
        Ok(id)
    })?;
    Ok(json!({"layer": id.0}))
}

fn insert_blank(s: &mut Session, _p: &Value) -> Result<Value> {
    let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
    edit_video(s, "Insert Blank Frame", move |v, cur| {
        let at = (cur + 1).min(v.frames.len());
        v.frames.insert(at, blank_of(fmt));
        if let Some(original) = &mut v.original_frames {
            // The stacks stay aligned, but never index past a shorter one.
            original.insert(at.min(original.len()), blank_of(fmt));
        }
        Ok(())
    })
}

fn duplicate_frame(s: &mut Session, _p: &Value) -> Result<Value> {
    edit_video(s, "Duplicate Frame", |v, cur| {
        if let Some(f) = v.frames.get(cur).cloned() {
            if let Some(original) = &mut v.original_frames {
                let orig = original.get(cur).cloned().unwrap_or_else(|| blank_of(f.format()));
                original.insert((cur + 1).min(original.len()), orig);
            }
            v.frames.insert(cur + 1, f);
        }
        Ok(())
    })
}

fn delete_frame(s: &mut Session, _p: &Value) -> Result<Value> {
    edit_video(s, "Delete Frame", |v, cur| {
        if v.frames.len() > 1 && cur < v.frames.len() {
            v.frames.remove(cur);
            if let Some(original) = &mut v.original_frames
                && cur < original.len()
            {
                original.remove(cur);
            }
        }
        Ok(())
    })
}

fn restore_frame(s: &mut Session, _p: &Value) -> Result<Value> {
    let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
    edit_video(s, "Restore Frame", move |v, cur| {
        let original = v.original_frames.as_ref().and_then(|frames| frames.get(cur)).cloned();
        if let Some(f) = v.frames.get_mut(cur) {
            *f = original.unwrap_or_else(|| blank_of(fmt));
        }
        Ok(())
    })
}

fn restore_all(s: &mut Session, _p: &Value) -> Result<Value> {
    let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
    edit_video(s, "Restore All Frames", move |v, _| {
        for (i, f) in v.frames.iter_mut().enumerate() {
            *f = v.original_frames.as_ref().and_then(|frames| frames.get(i)).cloned().unwrap_or_else(|| blank_of(fmt));
        }
        Ok(())
    })
}

fn show_altered(s: &mut Session, _p: &Value) -> Result<Value> {
    edit_video(s, "Show Altered Video", |v, _| {
        v.show_altered = !v.show_altered;
        Ok(())
    })
}

/// Rasterize: drop the frame stack, keeping the current frame as a normal pixel layer.
fn rasterize(s: &mut Session, _p: &Value) -> Result<Value> {
    s.edit("Rasterize Video", |doc, active| {
        let id = active.ok_or(EngineError::NoDocument)?;
        let l = doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?;
        // The content already shows the current frame (kept in sync); just drop the stack.
        l.video = None;
        Ok(())
    })?;
    Ok(json!({"rasterized": true}))
}

// ------------------------------------------------------------------ file-based footage

/// Frames loaded from a file or an image sequence.
struct Footage {
    frames: Vec<Surface>,
    /// The frame rate the footage itself implies (a single animated GIF's delays); `None` for
    /// stills and sequences, which carry none.
    fps: Option<f32>,
}

/// The frame rate for a GIF's frame delays (milliseconds): that of the most common delay, the
/// first one seen on a tie. GIF has a delay per frame but the timeline one rate. Delays under
/// 20 ms, which includes the 0 and 10 ms many encoders write, are shown at 100 ms, as browsers
/// do, since they would otherwise play at an unintended 50-100 fps.
fn fps_from_delays(delays: &[u32]) -> Option<f32> {
    let clean = |d: u32| if d < 20 { 100 } else { d };
    let mut counts: Vec<(u32, usize)> = Vec::new();
    for d in delays.iter().copied().map(clean) {
        match counts.iter_mut().find(|(k, _)| *k == d) {
            Some((_, n)) => *n += 1,
            None => counts.push((d, 1)),
        }
    }
    let best = counts.iter().fold(None::<(u32, usize)>, |best, &(d, n)| match best {
        Some((_, bn)) if bn >= n => best,
        _ => Some((d, n)),
    });
    best.map(|(d, _)| 1000.0 / d as f32)
}

/// Load frames from `path`: a single image (1 frame, or every frame of an animated GIF), or a
/// directory of images (sorted = an image sequence). Each becomes a full raster frame in the
/// document's format.
fn load_frames(path: &str, fmt: photocraft_color::PixelFormat) -> Result<Footage> {
    let p = std::path::Path::new(path);
    let is_img = |f: &std::path::Path| {
        matches!(
            f.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(),
            Some("png" | "jpg" | "jpeg" | "tif" | "tiff" | "webp" | "bmp" | "gif" | "tga" | "exr" | "pcraft" | "psd")
        )
    };
    let files: Vec<std::path::PathBuf> = if p.is_dir() {
        let mut v: Vec<_> = std::fs::read_dir(p)
            .map_err(|e| EngineError::Other(format!("read dir `{path}`: {e}")))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|f| is_img(f))
            .collect();
        v.sort();
        v
    } else {
        vec![p.to_path_buf()]
    };
    if files.is_empty() {
        return Err(EngineError::Other(format!("no images in `{path}`")));
    }
    let mut frames = Vec::with_capacity(files.len());
    let mut delays = Vec::new();
    for f in &files {
        let bytes = std::fs::read(f).map_err(|e| EngineError::Other(format!("read `{}`: {e}", f.display())))?;
        let name = f.to_string_lossy();
        if photocraft_codecs::detect(&bytes) == Some(photocraft_codecs::Format::Gif) {
            photocraft_io::import_gif_frames(&name, &bytes, &mut |doc, delay_ms| {
                frames.push(crate::file_cmds::flattened(&doc, fmt));
                delays.push(delay_ms);
                Ok(())
            })
            .map_err(|e| EngineError::Other(format!("`{}`: {e}", f.display())))?;
            continue;
        }
        let mut doc = crate::file_cmds::import(&name, &bytes)?;
        // A Targa's alpha opens as a channel (as in Photoshop); a frame's alpha is transparency.
        if f.extension().is_some_and(|e| e.eq_ignore_ascii_case("tga")) {
            photocraft_io::tga_alpha_channel_to_transparency(&mut doc);
        }
        frames.push(crate::file_cmds::flattened(&doc, fmt));
    }
    // Only a lone GIF says how fast it plays; a sequence mixing files has no single rate.
    let fps = if files.len() == 1 { fps_from_delays(&delays) } else { None };
    Ok(Footage { frames, fps })
}

fn new_from_file(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::BadParams { cmd: "layer.videoLayers.newVideoLayerFromFile".into(), msg: "need `path`".into() })?
        .to_string();
    let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
    let Footage { frames, fps: footage_fps } = load_frames(&path, fmt)?;
    // An explicit `fps` wins, then the rate the footage implies, then 30.
    let fps = p.get("fps").and_then(Value::as_f64).map(|f| f as f32).or(footage_fps).unwrap_or(30.0);
    let n = frames.len();
    let id = s.edit("New Video Layer from File", |doc, active| {
        store(doc);
        if doc.timeline.is_none() {
            doc.timeline = Some(Timeline::new(n, fps));
        } else if let Some(t) = &mut doc.timeline {
            grow_duration(t, n);
        }
        let name = doc.next_layer_name(std::path::Path::new(&path).file_stem().and_then(|s| s.to_str()).unwrap_or("Video"));
        let mut l = Layer::raster(name, fmt);
        let mut v = VideoData::new(frames, fps);
        v.original_frames = Some(v.frames.clone());
        v.source = photocraft_doc::VideoSource::File { path: path.clone() };
        l.video = Some(v);
        let id = doc.insert_above(*active, l);
        *active = Some(id);
        sync(doc);
        Ok(id)
    })?;
    Ok(json!({"layer": id.0, "frames": n}))
}

fn replace_footage(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::BadParams { cmd: "layer.videoLayers.replaceFootage".into(), msg: "need `path`".into() })?
        .to_string();
    let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
    let frames = load_frames(&path, fmt)?.frames;
    edit_video(s, "Replace Footage", move |v, _| {
        v.original_frames = Some(frames.clone());
        v.frames = frames;
        v.source = photocraft_doc::VideoSource::File { path: path.clone() };
        Ok(())
    })
}

fn interpret_footage(s: &mut Session, p: &Value) -> Result<Value> {
    let fps = p.get("fps").and_then(Value::as_f64);
    edit_video(s, "Interpret Footage", move |v, _| {
        if let Some(f) = fps {
            v.fps = f as f32;
        }
        Ok(())
    })
}

fn reload_frame(s: &mut Session, _p: &Value) -> Result<Value> {
    // Reload the whole stack from its file source (what a frame reload needs for a sequence).
    let (src, fmt) = {
        let st = s.active().ok_or(EngineError::NoDocument)?;
        let id = st.active_layer.ok_or(EngineError::NoDocument)?;
        let v = st
            .doc
            .layer(id)
            .and_then(|l| l.video.as_ref())
            .ok_or_else(|| EngineError::BadParams { cmd: "layer.videoLayers.reloadFrame".into(), msg: "not a video layer".into() })?;
        (v.source.clone(), st.doc.pixel_format())
    };
    let photocraft_doc::VideoSource::File { path } = src else {
        return Err(EngineError::BadParams { cmd: "layer.videoLayers.reloadFrame".into(), msg: "this layer has no file source to reload".into() });
    };
    let frames = load_frames(&path, fmt)?.frames;
    edit_video(s, "Reload Frame", move |v, _| {
        v.original_frames = Some(frames.clone());
        v.frames = frames;
        Ok(())
    })
}

/// Import › Video Frames to Layers: load frames from a file/sequence and add each as its own layer.
fn frames_to_layers(s: &mut Session, p: &Value) -> Result<Value> {
    let path = p
        .get("path")
        .and_then(Value::as_str)
        .ok_or_else(|| EngineError::BadParams { cmd: "file.import.videoFramesToLayers".into(), msg: "need `path`".into() })?
        .to_string();
    let fmt = s.active().ok_or(EngineError::NoDocument)?.doc.pixel_format();
    let frames = load_frames(&path, fmt)?.frames;
    let n = frames.len();
    s.edit("Video Frames to Layers", |doc, active| {
        let mut above = *active;
        for (i, f) in frames.into_iter().enumerate() {
            let mut l = Layer::raster(format!("Frame {}", i + 1), fmt);
            if let LayerContent::Raster(surf) = &mut l.content {
                *surf = f;
            }
            above = Some(doc.insert_above(above, l));
        }
        *active = above;
        Ok(())
    })?;
    Ok(json!({"layers": n}))
}

/// Export › Render Video: composite the whole document at each timeline frame and write an image
/// sequence (`{dir}/{stem}_NNNN.{format}`). Clean-room; no proprietary codec.
fn render_video(s: &mut Session, p: &Value) -> Result<Value> {
    let dir = p.get("dir").and_then(Value::as_str).ok_or_else(|| EngineError::BadParams { cmd: "file.export.renderVideo".into(), msg: "need `dir`".into() })?;
    std::fs::create_dir_all(dir).map_err(|e| EngineError::Other(format!("mkdir `{dir}`: {e}")))?;
    render_video_with(s, p, |name, bytes| crate::file_cmds::write_file(&format!("{dir}/{name}"), bytes), |_, _| true)
}

/// Render through a caller-owned writer. The writer receives one filename, never an ambient
/// directory. Returning false from progress stops at a frame boundary; the caller owns cleanup.
pub fn render_video_with(
    s: &Session,
    p: &Value,
    mut write: impl FnMut(&str, &[u8]) -> Result<()>,
    mut progress: impl FnMut(usize, usize) -> bool,
) -> Result<Value> {
    let dir = p.get("dir").and_then(Value::as_str).unwrap_or("");
    let format = p.get("format").and_then(Value::as_str).unwrap_or("png");
    let st = s.active().ok_or(EngineError::NoDocument)?;
    let mut base = (*st.doc).clone();
    store(&mut base);
    let frames = base.timeline.as_ref().map_or(1, |t| t.duration).max(1);
    let fps = base.timeline.as_ref().map_or(30.0, |t| t.fps).max(1.0);
    let stem = base.name.rsplit_once('.').map_or(base.name.as_str(), |(a, _)| a).to_string();
    let fmt = base.pixel_format();
    let bounds = base.bounds();
    let (w, h) = (bounds.width() as usize, bounds.height() as usize);

    let cancelled = || EngineError::Other("render cancelled".into());
    if !progress(0, frames) {
        return Err(cancelled());
    }

    // Animated GIF: one file, every frame quantised to its own palette.
    if format.eq_ignore_ascii_case("gif") {
        let delay = (100.0 / fps).round().clamp(1.0, 65535.0) as u16;
        let mut gframes = Vec::with_capacity(frames);
        for f in 0..frames {
            let mut doc = base.clone();
            if let Some(t) = &mut doc.timeline {
                t.current = f;
            }
            sync(&mut doc);
            let surf = crate::file_cmds::flattened(&doc, fmt);
            let mut px = vec![[0.0f32; 4]; w * h];
            surf.read_rgba_into(bounds, &mut px);
            gframes.push(gif_frame(px, w, delay)?);
            // The final report follows encoding/writing, so total means the output exists.
            if !progress((f + 1).min(frames.saturating_sub(1)), frames) {
                return Err(cancelled());
            }
        }
        let gif = photocraft_codecs::web::encode_gif_animated(w as u32, h as u32, &gframes, true).map_err(|e| EngineError::Other(e.to_string()))?;
        let name = format!("{stem}.gif");
        write(&name, &gif)?;
        if !progress(frames, frames) {
            return Err(cancelled());
        }
        let path = format!("{dir}/{name}");
        return Ok(json!({"frames": frames, "file": path}));
    }

    // Image sequence.
    let opts = photocraft_io::ExportOptions::default();
    let mut files = Vec::new();
    for f in 0..frames {
        let mut doc = base.clone();
        if let Some(t) = &mut doc.timeline {
            t.current = f;
        }
        sync(&mut doc);
        let path = format!("{stem}_{f:04}.{format}");
        let bytes = photocraft_io::export(&doc, format, &opts).map(|r| r.bytes).map_err(|e| EngineError::Other(format!("render frame {f}: {e}")))?;
        write(&path, &bytes)?;
        files.push(path);
        if !progress(f + 1, frames) {
            return Err(cancelled());
        }
    }
    Ok(json!({"frames": files.len(), "dir": dir}))
}

/// One animated GIF frame from straight RGBA pixels, as Save for Web's GIF does (#3061): pixels
/// under 50 % opacity get the frame's transparent index, the rest are drawn opaque, and the
/// palette is built from the visible pixels with one entry kept for transparency.
fn gif_frame(mut px: Vec<[f32; 4]>, w: usize, delay_cs: u16) -> Result<photocraft_codecs::web::GifFrame> {
    use photocraft_algo::quantize::{self, Dither, Forced, PaletteKind};
    let transparent = px.iter().any(|p| p[3] < 0.5);
    let visible: Vec<[f32; 4]> = px.iter().filter(|p| p[3] >= 0.5).map(|p| [p[0], p[1], p[2], 1.0]).collect();
    // A fully transparent frame still needs a palette.
    let src = if visible.is_empty() { &px } else { &visible };
    let mut pal = quantize::build_palette(src, PaletteKind::Adaptive, if transparent { 255 } else { 256 }, Forced::None).map_err(EngineError::Other)?;
    let t = transparent.then(|| {
        pal.truncate(255);
        pal.push(pal.first().copied().unwrap_or([0, 0, 0]));
        pal.len() - 1
    });
    let indices = quantize::quantize(&mut px, w, &pal, Dither::None, 1.0, t);
    Ok(photocraft_codecs::web::GifFrame { indices, palette: pal, transparent: t.map(|i| i as u8), delay_cs })
}

macro_rules! spec {
    ($id:expr, $label:expr, $menu:expr, $en:expr, $run:expr) => {
        CommandSpec { id: $id, label: $label, menu: $menu, shortcut: None, params: "{} → {}", enabled: $en, journal: true, run: $run }
    };
}

pub fn specs() -> Vec<CommandSpec> {
    let vl = &["Layer", "Video Layers"] as &[&str];
    vec![
        spec!("layer.videoLayers.newBlankVideoLayer", "New Blank Video Layer", vl, has_doc, |s, p| new_blank(s, p)),
        spec!("layer.videoLayers.insertBlankFrame", "Insert Blank Frame", vl, active_video, |s, p| insert_blank(s, p)),
        spec!("layer.videoLayers.duplicateFrame", "Duplicate Frame", vl, active_video, |s, p| duplicate_frame(s, p)),
        spec!("layer.videoLayers.deleteFrame", "Delete Frame", vl, active_video, |s, p| delete_frame(s, p)),
        spec!("layer.videoLayers.restoreFrame", "Restore Frame", vl, active_video, |s, p| restore_frame(s, p)),
        spec!("layer.videoLayers.restoreAllFrames", "Restore All Frames", vl, active_video, |s, p| restore_all(s, p)),
        spec!("layer.videoLayers.showAlteredVideo", "Show Altered Video", vl, active_video, |s, p| show_altered(s, p)),
        spec!("layer.videoLayers.rasterize", "Rasterize", vl, active_video, |s, p| rasterize(s, p)),
        spec!("layer.rasterize.video", "Video", &["Layer", "Rasterize"], active_video, |s, p| rasterize(s, p)),
        spec!("layer.videoLayers.newVideoLayerFromFile", "New Video Layer from File…", vl, has_doc, |s, p| new_from_file(s, p)),
        spec!("layer.videoLayers.replaceFootage", "Replace Footage…", vl, active_video, |s, p| replace_footage(s, p)),
        spec!("layer.videoLayers.interpretFootage", "Interpret Footage…", vl, active_video, |s, p| interpret_footage(s, p)),
        spec!("layer.videoLayers.reloadFrame", "Reload Frame", vl, active_video, |s, p| reload_frame(s, p)),
        spec!("file.import.videoFramesToLayers", "Video Frames to Layers…", &["File", "Import"], has_doc, |s, p| frames_to_layers(s, p)),
        spec!("file.export.renderVideo", "Render Video…", &["File", "Export"], has_doc, |s, p| render_video(s, p)),
    ]
}

#[cfg(test)]
mod tests;
