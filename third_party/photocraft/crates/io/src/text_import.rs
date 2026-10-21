//! Type layers read from a file, made ready for the canvas and the panels.

use photocraft_doc::{Document, Layer, LayerContent, TextLayer};

/// Largest type raster the import draws on its own (pixels). The model comes from an untrusted
/// file: an absurd type size, scale or box must not make opening it allocate gigabytes (the
/// renderer's own guard is far larger). A layer over it keeps empty pixels, as before, until
/// it is edited.
pub(crate) const MAX_RENDER_PIXELS: u64 = 64 * 1024 * 1024;

/// Renders `t` from its model unless its ink would exceed [`MAX_RENDER_PIXELS`].
fn render_bounded(eng: &mut photocraft_text::TextEngine, t: &mut TextLayer, dpi: f32, format: photocraft_color::PixelFormat) {
    let layout = eng.layout(t, dpi);
    let warp = photocraft_text::render::layout_deform(&layout, t);
    let ink = photocraft_text::render::ink_rect_warped(&layout, &t.transform, warp.as_ref());
    if u64::from(ink.width()).saturating_mul(u64::from(ink.height())) > MAX_RENDER_PIXELS {
        t.cache = Some(photocraft_raster::Surface::new(photocraft_color::PixelFormat { alpha: true, ..format }));
        return;
    }
    let r = photocraft_text::render::rasterize_warped(&layout, &t.transform, format, t.antialias, warp.as_ref());
    t.cache = Some(r.surface);
}

/// Prepares the type layers of a document that was just imported:
/// - a run whose font came as a PostScript name (PSD files) gets the installed family that name
///   resolves to, as the layout does, so Missing Fonts and the Character panel see the real
///   family (`CCWildWords-Regular` is the family "CCWildWords", not the guessed "CC Wild Words");
/// - a type layer without pixels (ag-psd, GIMP and other writers leave the image data out and
///   let Photoshop re-render on open) is rendered from its model, instead of staying empty
///   until it is edited.
pub(crate) fn prepare(doc: &mut Document) {
    fn visit(layers: &mut [Layer], eng: &mut photocraft_text::TextEngine, dpi: f32, format: photocraft_color::PixelFormat) {
        for l in layers {
            match &mut l.content {
                LayerContent::Text(t) => {
                    let mut changed = false;
                    for r in &mut t.runs {
                        if let Some(ps) = r.style.postscript_name.clone()
                            && !eng.fonts.faces(&r.style.font_family).iter().any(|f| f.postscript_name.as_deref() == Some(&ps))
                        {
                            let f = eng.fonts.resolve_postscript(&ps);
                            // Keep installed families, including legacy names whose faces are
                            // indexed under a typographic family. An exact face may recover a
                            // truncated hyphenated family; a heuristic only fills a missing one.
                            let installed = eng.fonts.has_family(&r.style.font_family);
                            let exact_recovery =
                                f.exact && (!installed || f.family.strip_prefix(&r.style.font_family).is_some_and(|suffix| suffix.starts_with('-')));
                            if exact_recovery || (!installed && eng.fonts.has_family(&f.family)) {
                                r.style.font_family = f.family;
                                r.style.weight = f.weight;
                                r.style.italic = f.italic;
                                changed = true;
                            }
                        }
                    }
                    if changed {
                        t.sync_summary();
                    }
                    if t.cache.is_none() {
                        render_bounded(eng, t, dpi, format);
                    }
                }
                LayerContent::Group(g) => visit(&mut g.children, eng, dpi, format),
                _ => {}
            }
        }
    }
    let dpi = doc.resolution_dpi;
    let format = doc.pixel_format();
    let mut eng = photocraft_text::shared().lock().unwrap_or_else(|e| e.into_inner());
    visit(&mut doc.layers, &mut eng, dpi, format);
}
