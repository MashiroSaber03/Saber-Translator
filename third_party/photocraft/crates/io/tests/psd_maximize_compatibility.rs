//! Preferences › File Handling › Maximize PSD and PSB File Compatibility: turned off, a layered
//! PSD/PSB stores Photoshop's white placeholder instead of the merged image and says so in
//! resource 1057 (`has_real_merged_data = false`). Layers, alpha channels and the Quick Mask are
//! written as usual, so the file reopens the same.

use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{AlphaChannel, Document, Layer};
use photocraft_geom::{Rect, Size};
use photocraft_io::{ExportOptions, export, import};
use photocraft_psd::PsdFile;
use photocraft_raster::Surface;

fn compat_off() -> ExportOptions {
    ExportOptions { psd_maximize_compatibility: false, ..Default::default() }
}

fn parse(bytes: &[u8]) -> PsdFile {
    PsdFile::from_bytes(bytes).expect("parse")
}

/// A `w`×`h` document of `mode`/`depth` holding one layer of `rgba` over the whole canvas.
fn filled(mode: ColorMode, depth: SampleType, w: u32, h: u32, rgba: [f32; 4]) -> Document {
    let mut d = Document::new("compat", Size::new(w, h), mode, depth);
    let fmt = d.pixel_format();
    let mut px = vec![0.0; fmt.channels()];
    photocraft_raster::from_rgba_into(&fmt, rgba, &mut px);
    let mut layer = Layer::raster("pixels", fmt);
    layer.surface_mut().unwrap().fill_rect(Rect::new(0, 0, w as i32, h as i32), &px);
    d.layers.push(layer);
    d
}

/// An 8-bit RGB document whose layer is seeded noise, alpha included: its merged image barely
/// compresses, unlike the placeholder.
fn noisy(w: u32, h: u32) -> Document {
    let mut d = Document::new("noise", Size::new(w, h), ColorMode::Rgb, SampleType::U8);
    let fmt = d.pixel_format();
    let mut state = 0x2545_f491_u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state & 0xff) as f32 / 255.0
    };
    let vals: Vec<f32> = (0..(w * h) as usize * fmt.channels()).map(|_| next()).collect();
    let mut layer = Layer::raster("noise", fmt);
    layer.surface_mut().unwrap().write_region(Rect::new(0, 0, w as i32, h as i32), &vals);
    d.layers.push(layer);
    d
}

#[test]
fn off_writes_the_white_placeholder_and_flags_it() {
    let d = noisy(96, 64);
    for name in ["x.psd", "x.psb"] {
        let no_previews = |o: ExportOptions| ExportOptions { image_previews: false, ..o };
        let on = export(&d, name, &no_previews(ExportOptions::default())).unwrap();
        let off = export(&d, name, &no_previews(compat_off())).unwrap();
        let (real, placeholder) = (parse(&on.bytes), parse(&off.bytes));
        assert_eq!(real.has_real_merged_data(), Some(true), "{name}");
        assert_eq!(placeholder.has_real_merged_data(), Some(false), "{name}");
        assert!(real.merged_has_alpha() && !placeholder.merged_has_alpha(), "{name}: the placeholder has no alpha");
        assert!(placeholder.composite_rgba8().unwrap().data.iter().all(|&b| b == 255), "{name}: opaque white");
        assert_eq!(real.layers(), placeholder.layers(), "{name}: the layers are written the same");
        assert!(off.bytes.len() * 3 < on.bytes.len() * 2, "{name}: {} bytes without the merged image, {} with it", off.bytes.len(), on.bytes.len());
    }
}

/// The placeholder is exactly the white a real merged image would store, in every colour model
/// and depth PSD has: CMYK white is no ink, Lab white is L 100 with neutral a*/b*.
#[test]
fn the_placeholder_is_the_white_a_real_merged_image_stores() {
    use ColorMode::{Cmyk, Grayscale, Lab, Rgb};
    use SampleType::{F32, U8, U16};
    for (mode, depth) in
        [(Rgb, U8), (Rgb, U16), (Rgb, F32), (Grayscale, U8), (Grayscale, U16), (Grayscale, F32), (Cmyk, U8), (Cmyk, U16), (Lab, U8), (Lab, U16)]
    {
        let d = filled(mode, depth, 9, 5, [1.0; 4]);
        let real = parse(&export(&d, "x.psd", &ExportOptions::default()).unwrap().bytes);
        let placeholder = parse(&export(&d, "x.psd", &compat_off()).unwrap().bytes);
        assert_eq!(placeholder.has_real_merged_data(), Some(false), "{mode:?} {depth:?}");
        assert!(!real.merged_has_alpha(), "{mode:?} {depth:?}: an opaque document");
        assert_eq!(placeholder.header, real.header, "{mode:?} {depth:?}");
        assert_eq!(placeholder.decode_merged().unwrap(), real.decode_merged().unwrap(), "{mode:?} {depth:?}");
    }
}

#[test]
fn layers_and_alpha_channels_reopen_unchanged() {
    let mut d = noisy(32, 16);
    let mut selection = Surface::new(PixelFormat::new(ColorMode::Grayscale, SampleType::U8, false));
    selection.fill_rect(Rect::new(0, 0, 10, 16), &[1.0]);
    d.channels.push(AlphaChannel::new("Alpha 1", selection.clone()));
    d.quick_mask = Some(AlphaChannel::new("Quick Mask", selection));
    let reopen = |opts: &ExportOptions| {
        let r = import("x.psd", &export(&d, "x.psd", opts).unwrap().bytes).unwrap();
        assert!(r.warnings.is_empty(), "{:?}", r.warnings);
        r.document
    };
    let (on, off) = (reopen(&ExportOptions::default()), reopen(&compat_off()));
    assert_eq!(off.layers.len(), 1);
    assert_eq!(off.channels, on.channels, "alpha channels follow the placeholder");
    assert_eq!(off.quick_mask, on.quick_mask);
    let canvas = d.bounds();
    assert_eq!(photocraft_compose::render(&off, canvas).px, photocraft_compose::render(&on, canvas).px, "the same image");
}

/// Without layer records the merged image is the only copy of the pixels, so it stays real.
#[test]
fn a_file_without_layers_keeps_its_real_merged_image() {
    let d = Document::new("empty", Size::new(8, 8), ColorMode::Rgb, SampleType::U8);
    let file = parse(&export(&d, "x.psd", &compat_off()).unwrap().bytes);
    assert!(file.layers().is_empty());
    assert_eq!(file.has_real_merged_data(), Some(true));
}
