//! Full-resolution refinement of Camera Raw's interactive proxy. Native work is off the
//! UI thread; only one revision runs at a time, and obsolete results never replace new edits.
use egui::{Color32, ColorImage, Context, Pos2, Rect, TextureHandle, TextureOptions, pos2, vec2};
use photocraft_algo::camera_raw::CameraRaw;
use photocraft_geom::Rect as PixelRect;
use photocraft_raster::Surface;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
    mpsc::{Receiver, TryRecvError},
};

#[derive(Clone)]
pub(crate) enum Coverage {
    Selection(Surface),
    Mask(photocraft_doc::LayerMask),
    None,
}

type DetailResult = Result<(u64, Surface), String>;
type OverlayKey = (u64, bool, PixelRect, crate::camera_raw_scope_ui::ClippingMode, crate::theme::ThemeKind);

pub(crate) struct DetailPreview {
    source: Surface,
    area: PixelRect,
    coverage: Coverage,
    result: Option<(u64, Surface)>,
    pending: Option<(u64, Receiver<DetailResult>)>,
    cancel: Arc<AtomicBool>,
    pub(crate) error: Option<String>,
    failed_revision: Option<u64>,
    crop: Option<(u64, bool, PixelRect, u64)>,
    texture: Option<TextureHandle>,
    overlay_texture: Option<TextureHandle>,
    overlay_key: Option<OverlayKey>,
    pub(crate) ready: bool,
}

impl Drop for DetailPreview {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl DetailPreview {
    pub(crate) fn new(source: Surface, area: PixelRect, coverage: Coverage) -> Self {
        Self {
            source,
            area,
            coverage,
            result: None,
            pending: None,
            cancel: Arc::new(AtomicBool::new(false)),
            error: None,
            failed_revision: None,
            crop: None,
            texture: None,
            overlay_texture: None,
            overlay_key: None,
            ready: false,
        }
    }

    pub(crate) fn pending(&self) -> bool {
        self.pending.is_some()
    }

    fn poll(&mut self, revision: u64) {
        let Some((requested_revision, rx)) = &self.pending else { return };
        let requested_revision = *requested_revision;
        let result = match rx.try_recv() {
            Ok(result) => result,
            Err(TryRecvError::Empty) => return,
            Err(TryRecvError::Disconnected) => Err("Camera Raw detail preview worker stopped".into()),
        };
        self.pending = None;
        match result {
            Ok((r, surface)) if r == revision => {
                self.result = Some((r, surface));
                self.crop = None;
                self.error = None;
            }
            Ok(_) => {} // A parameter changed while the worker was developing this revision.
            Err(e) => {
                if requested_revision == revision {
                    self.error = Some(e);
                }
                self.failed_revision = Some(requested_revision);
            }
        }
    }

    fn request(&mut self, ctx: &Context, revision: u64, params: &CameraRaw) {
        if self.pending.is_some() || self.result.as_ref().is_some_and(|(r, _)| *r == revision) || self.failed_revision == Some(revision) {
            return;
        }
        self.result = None;
        self.crop = None;
        self.error = None;
        // The engine's full-image pipeline needs multiple temporary float buffers. Never start
        // an unbounded allocation just because a huge sparse document was zoomed into.
        if self.area.width() as u64 * self.area.height() as u64 > 64_000_000 {
            self.error = Some("Full-resolution Camera Raw preview is limited to 64 megapixels; the proxy remains available".into());
            self.failed_revision = Some(revision);
            return;
        }
        let source = self.source.clone();
        let area = self.area;
        let coverage = self.coverage.clone();
        let mut params = params.clone();
        params.pixel_scale = 1.0;
        let cancel = self.cancel.clone();
        let ctx = ctx.clone();
        let (tx, rx) = std::sync::mpsc::channel::<DetailResult>();
        let work = move || {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| develop(&source, area, &params, &coverage)))
                .map(|surface| (revision, surface))
                .map_err(|_| "Camera Raw detail preview failed; the proxy remains available".into());
            if !cancel.load(Ordering::Relaxed) {
                let _ = tx.send(result);
                ctx.request_repaint();
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            // A closed dialog can leave one non-interruptible engine stage finishing. Serialize
            // that work with the next dialog instead of multiplying its peak memory.
            static WORKER: std::sync::Mutex<()> = std::sync::Mutex::new(());
            match std::thread::Builder::new().name("camera-raw-preview".into()).spawn(move || {
                let _guard = WORKER.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                work();
            }) {
                Ok(_) => self.pending = Some((revision, rx)),
                Err(e) => {
                    self.error = Some(format!("Cannot start Camera Raw preview: {e}"));
                    self.failed_revision = Some(revision);
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // No browser worker/executor is available here. Preserve the interactive proxy
            // instead of blocking its UI with the full-image engine pipeline.
            drop(work);
            drop(rx);
            self.error = Some("Full-resolution filtered preview requires a native Camera Raw worker; the web preview keeps the proxy".into());
            self.failed_revision = Some(revision);
        }
    }

    /// Draw only the visible pixel rectangle. No full-image GPU texture and no GPU-size limit
    /// on the source image; uploads are capped to the current viewport plus a one-pixel border.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn paint(
        &mut self,
        ui: &egui::Ui,
        image: Rect,
        revision: u64,
        before: bool,
        params: &CameraRaw,
        needed: bool,
        display: &photocraft_engine::display_color::CanvasDisplay,
    ) -> bool {
        self.ready = false;
        self.poll(revision);
        if !needed {
            return false;
        }
        let Some(crop) = visible_pixels(image, ui.clip_rect(), self.area, ui.input(|i| i.max_texture_side)) else { return false };
        let original = before || params.is_identity();
        if !original {
            self.request(ui.ctx(), revision, params);
            self.poll(revision);
        }
        let surface = if original {
            &self.source
        } else if let Some((r, surface)) = &self.result
            && *r == revision
        {
            surface
        } else {
            return false;
        };
        let key = (if original { 0 } else { revision }, original, crop, display.key);
        if self.crop != Some(key) {
            let color = crop_image(surface, crop, display);
            let options = TextureOptions { magnification: egui::TextureFilter::Nearest, ..TextureOptions::LINEAR };
            match &mut self.texture {
                Some(t) => t.set(color, options),
                None => self.texture = Some(ui.ctx().load_texture("cr-detail-preview", color, options)),
            }
            self.crop = Some(key);
        }
        if let Some(texture) = &self.texture {
            let scale = image.size() / vec2(self.area.width() as f32, self.area.height() as f32);
            let screen = Rect::from_min_size(
                image.min + vec2((crop.x0 - self.area.x0) as f32, (crop.y0 - self.area.y0) as f32) * scale,
                vec2(crop.width() as f32, crop.height() as f32) * scale,
            );
            ui.painter().image(texture.id(), screen, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
            self.ready = true;
        }
        true
    }
    pub(crate) fn sample(&self, p: [f32; 2], revision: u64, before: bool, params: &CameraRaw) -> Option<[f32; 4]> {
        if !self.ready || !p.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            return None;
        }
        let source = if before || params.is_identity() {
            &self.source
        } else if let Some((r, source)) = &self.result
            && *r == revision
        {
            source
        } else {
            return None;
        };
        let x = self.area.x0 + ((p[0] * self.area.width() as f32) as i32).min((self.area.width() - 1) as i32);
        let y = self.area.y0 + ((p[1] * self.area.height() as f32) as i32).min((self.area.height() - 1) as i32);
        Some(source.rgba(x, y))
    }

    pub(crate) fn overlay(&mut self, ui: &egui::Ui, image: Rect, revision: u64, mode: crate::camera_raw_scope_ui::ClippingMode) -> bool {
        use crate::camera_raw_scope_ui::ClippingMode;
        if !self.ready {
            return false;
        }
        let Some((r, original, crop, _display_key)) = self.crop else { return false };
        let t = crate::theme::Tokens::get(ui.ctx());
        let key = (r, original, crop, mode, t.kind);
        if self.overlay_key != Some(key) {
            let surface = if original {
                &self.source
            } else if let Some((r, surface)) = &self.result
                && *r == revision
            {
                surface
            } else {
                return false;
            };
            let mut row = vec![[0.0; 4]; crop.width() as usize];
            let mut pixels = Vec::with_capacity(crop.width() as usize * crop.height() as usize);
            for y in crop.y0..crop.y1 {
                surface.read_rgba_into(PixelRect::new(crop.x0, y, crop.x1, y + 1), &mut row);
                pixels.extend(row.iter().map(|p| {
                    if p[3] <= 0.0 || !p.iter().all(|v| v.is_finite()) {
                        return Color32::TRANSPARENT;
                    }
                    let (lo, hi) = photocraft_algo::histogram::clipping(*p);
                    match mode {
                        ClippingMode::ShadowChannels => t.histogram_color(7 ^ lo),
                        ClippingMode::HighlightChannels => {
                            if hi == 0 {
                                t.histogram_background()
                            } else {
                                t.histogram_color(hi)
                            }
                        }
                        ClippingMode::Warnings(w) => {
                            if w & 2 != 0 && hi != 0 {
                                t.histogram_color(1)
                            } else if w & 1 != 0 && lo != 0 {
                                t.histogram_color(4)
                            } else {
                                Color32::TRANSPARENT
                            }
                        }
                        ClippingMode::None => Color32::TRANSPARENT,
                    }
                }));
            }
            let color = ColorImage::new([crop.width() as usize, crop.height() as usize], pixels);
            match &mut self.overlay_texture {
                Some(t) => t.set(color, TextureOptions::NEAREST),
                None => self.overlay_texture = Some(ui.ctx().load_texture("cr-detail-clipping", color, TextureOptions::NEAREST)),
            }
            self.overlay_key = Some(key);
        }
        if let Some(texture) = &self.overlay_texture {
            let scale = image.size() / vec2(self.area.width() as f32, self.area.height() as f32);
            let screen = Rect::from_min_size(
                image.min + vec2((crop.x0 - self.area.x0) as f32, (crop.y0 - self.area.y0) as f32) * scale,
                vec2(crop.width() as f32, crop.height() as f32) * scale,
            );
            ui.painter().image(texture.id(), screen, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
        }
        true
    }
}

fn develop(source: &Surface, area: PixelRect, params: &CameraRaw, coverage: &Coverage) -> Surface {
    let mut result = photocraft_engine::lens_cmds::camera_raw_surface(source, area, params);
    if matches!(coverage, Coverage::None) {
        return result;
    }
    let width = area.width() as usize;
    let channels = source.channels();
    let mut original = Vec::new();
    let mut row = Vec::new();
    for y in area.y0..area.y1 {
        let rect = PixelRect::new(area.x0, y, area.x1, y + 1);
        source.read_region_into(rect, &mut original);
        result.read_region_into(rect, &mut row);
        for (x, (out, original)) in row.chunks_exact_mut(channels).zip(original.chunks_exact(channels)).take(width).enumerate() {
            let x = area.x0 + x as i32;
            let alpha = match coverage {
                Coverage::Selection(mask) => mask.sample_channel(x, y, 0),
                Coverage::Mask(mask) => mask.value(x, y),
                Coverage::None => 1.0,
            }
            .clamp(0.0, 1.0);
            for (out, original) in out.iter_mut().zip(original) {
                *out = *original + (*out - *original) * alpha;
            }
        }
        result.write_region(rect, &row);
    }
    result
}

fn visible_pixels(image: Rect, viewport: Rect, area: PixelRect, max_side: usize) -> Option<PixelRect> {
    let visible = image.intersect(viewport);
    if !image.is_finite() || image.width() <= 0.0 || image.height() <= 0.0 || !visible.is_positive() || area.is_empty() {
        return None;
    }
    let scale = vec2(area.width() as f32, area.height() as f32) / image.size();
    let lo = (visible.min - image.min) * scale;
    let hi = (visible.max - image.min) * scale;
    let crop = PixelRect::new(
        area.x0.saturating_add((lo.x.floor() as i32).saturating_sub(1)),
        area.y0.saturating_add((lo.y.floor() as i32).saturating_sub(1)),
        area.x0.saturating_add((hi.x.ceil() as i32).saturating_add(1)),
        area.y0.saturating_add((hi.y.ceil() as i32).saturating_add(1)),
    )
    .intersect(&area);
    // Proxy is sufficient below its native scale; a larger intermediate should never be
    // allocated if an unusually large viewport exceeds the device's texture limit.
    (crop.width() as usize <= max_side && crop.height() as usize <= max_side && crop.width() as u64 * crop.height() as u64 <= 16_777_216).then_some(crop)
}

/// Pixels per band [`display_image`] reads and colour-converts at once: at least `photocraft-cms`'s
/// parallel threshold (16K pixels), and small enough that a band's float scratch is 1 MiB.
const CROP_BAND_PIXELS: usize = 64 * 1024;

/// The visible pixels of `surface`, as the canvas shows them.
fn crop_image(surface: &Surface, area: PixelRect, display: &photocraft_engine::display_color::CanvasDisplay) -> ColorImage {
    display_image(area, display, |rect, out| surface.read_rgba_into(rect, out))
}

/// `area` as the canvas shows it, with `read` filling each band's float pixels. Converted in
/// bands of rows, in parallel off wasm, not one row or one whole copy at a time: a row is below
/// `photocraft-cms`'s parallel threshold, so a per-row conversion ran the display transform (any
/// document profile other than the monitor's, such as an Adobe RGB photo on sRGB) on one core of
/// the UI thread for every edit (#4682).
pub(crate) fn display_image(
    area: PixelRect,
    display: &photocraft_engine::display_color::CanvasDisplay,
    read: impl Fn(PixelRect, &mut [[f32; 4]]) + Sync,
) -> ColorImage {
    let (width, height) = (area.width() as usize, area.height() as usize);
    let mut pixels = vec![Color32::TRANSPARENT; width * height];
    if width == 0 {
        return ColorImage::new([width, height], pixels);
    }
    let band_rows = (CROP_BAND_PIXELS / width).max(1);
    let band = |index: usize, out: &mut [Color32]| {
        let y0 = area.y0.saturating_add((index * band_rows) as i32);
        let rect = PixelRect::new(area.x0, y0, area.x1, y0.saturating_add((out.len() / width) as i32));
        let mut buf = photocraft_compose::Buffer { rect, px: vec![[0.0; 4]; out.len()] };
        read(rect, &mut buf.px);
        let img = display.to_rgba8(&buf);
        for (o, &[r, g, b, a]) in out.iter_mut().zip(img.pixels.as_chunks::<4>().0) {
            *o = Color32::from_rgba_unmultiplied(r, g, b, a);
        }
    };
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        pixels.par_chunks_mut(band_rows * width).enumerate().for_each(|(i, out)| band(i, out));
    }
    #[cfg(target_arch = "wasm32")]
    for (i, out) in pixels.chunks_mut(band_rows * width).enumerate() {
        band(i, out);
    }
    ColorImage::new([width, height], pixels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use photocraft_color::PixelFormat;
    #[test]
    fn refinement_uses_the_engine_at_source_resolution_and_respects_selection() {
        let area = PixelRect::new(10, -5, 42, 19);
        let mut source = Surface::new(PixelFormat::RGBA8);
        let mut values = vec![0.0; 32 * 24 * 4];
        for (i, p) in values.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            p.copy_from_slice(&[if i % 2 == 0 { 0.1 } else { 0.8 }, 0.3, 0.6, 1.0]);
        }
        source.write_region(area, &values);
        let params = CameraRaw { exposure: 0.5, texture: 15.0, grain_amount: 30.0, ..Default::default() };
        let expected = photocraft_engine::lens_cmds::camera_raw_surface(&source, area, &params);
        assert_eq!(develop(&source, area, &params, &Coverage::None), expected);
        let mask = Surface::with_default(PixelFormat::GRAY8, &[0.0]);
        assert_eq!(develop(&source, area, &params, &Coverage::Selection(mask)), source);
    }
    #[test]
    fn stale_refinements_and_failed_old_revisions_do_not_replace_new_settings() {
        let source = Surface::with_default(PixelFormat::RGBA8, &[0.2, 0.4, 0.6, 1.0]);
        let mut preview = DetailPreview::new(source.clone(), PixelRect::new(0, 0, 16, 16), Coverage::None);
        let (tx, rx) = std::sync::mpsc::channel();
        preview.pending = Some((1, rx));
        tx.send(Ok((1, source.clone()))).unwrap();
        preview.poll(2);
        assert!(preview.result.is_none());
        assert!(!preview.pending());
        let (tx, rx) = std::sync::mpsc::channel();
        preview.pending = Some((1, rx));
        tx.send(Err("old failure".into())).unwrap();
        preview.poll(2);
        assert_eq!(preview.failed_revision, Some(1), "a failure must not suppress the new revision");
        let (tx, rx) = std::sync::mpsc::channel();
        preview.pending = Some((2, rx));
        tx.send(Ok((2, source))).unwrap();
        preview.poll(2);
        assert_eq!(preview.result.as_ref().map(|(r, _)| *r), Some(2));
        assert!(preview.error.is_none());
    }

    #[test]
    fn full_resolution_sampling_and_crops_keep_fine_detail_at_every_supported_depth() {
        use photocraft_color::{ColorMode, PixelFormat, SampleType};
        for depth in [SampleType::U8, SampleType::U16, SampleType::F32] {
            for mode in [ColorMode::Rgb, ColorMode::Grayscale] {
                let format = PixelFormat { mode, sample: depth, alpha: true };
                let area = PixelRect::new(-12, 8, 28, 38);
                let mut source = Surface::new(format);
                let channels = format.channels();
                let mut pixels = vec![0.0; 40 * 30 * channels];
                for (i, pixel) in pixels.chunks_exact_mut(channels).enumerate() {
                    pixel.fill(if i % 2 == 0 { 0.1 } else { 0.9 });
                    pixel[channels - 1] = 1.0;
                }
                source.write_region(area, &pixels);
                let params = CameraRaw { exposure: 0.5, sharpen_amount: 20.0, grain_amount: 30.0, ..Default::default() };
                let full = develop(&source, area, &params, &Coverage::None);
                let expected = photocraft_engine::lens_cmds::camera_raw_surface(&source, area, &params);
                assert_eq!(full, expected);
                let mut preview = DetailPreview::new(source, area, Coverage::None);
                preview.result = Some((1, full));
                preview.ready = true;
                let dark = preview.sample([0.0, 0.0], 1, false, &params).unwrap();
                let light = preview.sample([1.0 / 40.0, 0.0], 1, false, &params).unwrap();
                assert!(light[0] - dark[0] > 0.5, "native alternating pixels survive: {mode:?}/{depth:?}");
                let mut doc =
                    photocraft_doc::Document::new("test", photocraft_geom::Size::new(40, 30), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
                doc.icc_profile = Some(photocraft_cms::Builtin::Srgb.profile().to_bytes());
                let display = photocraft_engine::color_cmds::ColorState::default().canvas_display(&doc).unwrap();
                assert_eq!(crop_image(&expected, PixelRect::new(-12, 8, -10, 9), &display).size, [2, 1]);
                assert!(preview.sample([f32::NAN, 0.0], 1, false, &params).is_none());
                assert!(preview.sample([0.5, 0.5], 2, false, &params).is_none());
            }
        }
    }

    #[test]
    fn refined_crop_uses_linear_document_display_encoding() {
        let area = PixelRect::new(0, 0, 1, 1);
        let mut source =
            Surface::new(photocraft_color::PixelFormat { mode: photocraft_color::ColorMode::Rgb, sample: photocraft_color::SampleType::F32, alpha: true });
        source.write_region(area, &[0.25, 0.25, 0.25, 1.0]);
        let mut doc = photocraft_doc::Document::new("test", photocraft_geom::Size::new(1, 1), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::F32);
        doc.icc_profile = Some(photocraft_cms::Builtin::LinearSrgb.profile().to_bytes());
        let mut color = photocraft_engine::color_cmds::ColorState::default();
        color.settings.monitor_profile = "srgb".into();
        let display = color.canvas_display(&doc).unwrap();
        let image = crop_image(&source, area, &display);
        assert!(image.pixels[0].r() > 130, "linear 0.25 should be shown as encoded sRGB");
    }

    #[test]
    fn huge_sparse_images_keep_the_proxy_without_starting_an_unbounded_worker() {
        let source = Surface::new(PixelFormat::RGBA8);
        let mut preview = DetailPreview::new(source, PixelRect::new(0, 0, 8001, 8000), Coverage::None);
        preview.request(&Context::default(), 1, &CameraRaw { exposure: 1.0, ..Default::default() });
        assert!(!preview.pending());
        assert!(preview.error.as_deref().unwrap().contains("64 megapixels"));
        assert_eq!(preview.failed_revision, Some(1));
    }

    #[test]
    fn zoom_uploads_only_visible_pixels_even_on_a_huge_offset_image() {
        let area = PixelRect::new(-200, 10, 15800, 10010);
        let image = Rect::from_min_size(pos2(-7800.0, -4800.0), vec2(16000.0, 10000.0));
        let visible = Rect::from_min_size(Pos2::ZERO, vec2(800.0, 600.0));
        let crop = visible_pixels(image, visible, area, 4096).unwrap();
        assert_eq!((crop.width(), crop.height()), (802, 602));
        assert!(visible_pixels(image, Rect::from_min_size(Pos2::ZERO, vec2(8000.0, 6000.0)), area, 4096).is_none());
        assert!(visible_pixels(Rect::NOTHING, visible, area, 4096).is_none());
    }

    #[test]
    fn banded_crop_matches_one_conversion_of_the_whole_area() {
        // Several bands, a ragged last one and an area off the image origin, through a real
        // transform (an Adobe RGB document on an sRGB monitor) and without one.
        let area = PixelRect::new(-3, 5, 1021, 5 + 2 * (CROP_BAND_PIXELS / 1024) as i32 + 7);
        let mut source = Surface::new(PixelFormat { mode: photocraft_color::ColorMode::Rgb, sample: photocraft_color::SampleType::U8, alpha: true });
        let n = (area.width() * area.height()) as usize;
        let px: Vec<f32> = (0..n).flat_map(|i| [(i % 251) as f32 / 255.0, (i % 241) as f32 / 255.0, (i % 239) as f32 / 255.0, (i % 7) as f32 / 6.0]).collect();
        source.write_region(area, &px);
        for profile in [photocraft_cms::Builtin::AdobeRgbCompat, photocraft_cms::Builtin::Srgb] {
            let mut doc =
                photocraft_doc::Document::new("test", photocraft_geom::Size::new(1024, 1024), photocraft_doc::ColorMode::Rgb, photocraft_doc::SampleType::U8);
            doc.icc_profile = Some(profile.profile().to_bytes());
            let mut color = photocraft_engine::color_cmds::ColorState::default();
            color.settings.monitor_profile = "srgb".into();
            let display = color.canvas_display(&doc).unwrap();
            assert_eq!(display.transform.is_some(), profile == photocraft_cms::Builtin::AdobeRgbCompat);
            let mut whole = photocraft_compose::Buffer { rect: area, px: vec![[0.0; 4]; n] };
            source.read_rgba_into(area, &mut whole.px);
            let expected = display.to_rgba8(&whole);
            let image = crop_image(&source, area, &display);
            assert_eq!(image.size, [area.width() as usize, area.height() as usize]);
            let expected: Vec<Color32> = expected.pixels.as_chunks::<4>().0.iter().map(|&[r, g, b, a]| Color32::from_rgba_unmultiplied(r, g, b, a)).collect();
            assert_eq!(image.pixels, expected, "{profile:?}");
        }
        assert_eq!(
            crop_image(
                &source,
                PixelRect::new(0, 0, 0, 4),
                &photocraft_engine::color_cmds::ColorState::default()
                    .canvas_display(&photocraft_doc::Document::new(
                        "t",
                        photocraft_geom::Size::new(1, 1),
                        photocraft_doc::ColorMode::Rgb,
                        photocraft_doc::SampleType::U8
                    ))
                    .unwrap()
            )
            .size,
            [0, 4]
        );
    }
}
