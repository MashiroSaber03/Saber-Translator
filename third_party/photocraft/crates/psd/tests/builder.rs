//! Builder and pixel extraction tests.

use photocraft_psd::*;

fn rgba_pattern(w: usize, h: usize, seed: u8) -> Vec<u8> {
    let mut v = Vec::with_capacity(w * h * 4);
    for y in 0..h {
        for x in 0..w {
            v.extend_from_slice(&[(x * 10) as u8 ^ seed, (y * 20) as u8, (x + y) as u8 ^ seed, if (x + y) % 3 == 0 { 0 } else { 200 + (x % 50) as u8 }]);
        }
    }
    v
}

fn built(b: &PsdBuilder) -> PsdFile {
    let bytes = b.to_bytes().unwrap();
    let f = PsdFile::from_bytes(&bytes).unwrap();
    assert_eq!(f.to_bytes().unwrap(), bytes, "byte stable");
    assert_eq!(f, b.build().unwrap(), "model stable");
    f
}

macro_rules! rgba8_case {
    ($name:ident, $ver:expr, $comp:expr) => {
        #[test]
        fn $name() {
            let px = rgba_pattern(7, 5, 3);
            let mut b = PsdBuilder::new(10, 8).version($ver).compression($comp);
            b.push_layer(LayerSpec::new("L", -2, 1, 7, 5, PixelData::Rgba8(px.clone())));
            let f = built(&b);
            let img = f.layer(0).unwrap().rgba8().unwrap();
            assert_eq!((img.left, img.top, img.width, img.height), (-2, 1, 7, 5));
            assert_eq!(img.data, px);
        }
    };
}

rgba8_case!(rgba8_psd_raw, Version::Psd, Compression::Raw);
rgba8_case!(rgba8_psd_rle, Version::Psd, Compression::Rle);
rgba8_case!(rgba8_psd_zip, Version::Psd, Compression::Zip);
rgba8_case!(rgba8_psd_zipp, Version::Psd, Compression::ZipPrediction);
rgba8_case!(rgba8_psb_raw, Version::Psb, Compression::Raw);
rgba8_case!(rgba8_psb_rle, Version::Psb, Compression::Rle);
rgba8_case!(rgba8_psb_zip, Version::Psb, Compression::Zip);
rgba8_case!(rgba8_psb_zipp, Version::Psb, Compression::ZipPrediction);

#[test]
fn rgba16_layers_scale_to_8bit() {
    let n = 4 * 3;
    let px: Vec<u16> = (0..n * 4).map(|i| (i as u16).wrapping_mul(5003)).collect();
    for comp in Compression::ALL {
        let mut b = PsdBuilder::new(4, 3).depth(16).compression(comp);
        b.push_layer(LayerSpec::new("x", 0, 0, 4, 3, PixelData::Rgba16(px.clone())));
        let f = built(&b);
        assert!(matches!(f.layer_info_placement, LayerInfoPlacement::GlobalBlock { key: [b'L', b'r', b'1', b'6'], .. }));
        let l = f.layer(0).unwrap();
        let img = l.rgba8().unwrap();
        let expect: Vec<u8> = px.iter().map(|&v| ((u32::from(v) * 255 + 32767) / 65535) as u8).collect();
        assert_eq!(img.data, expect);
        // Raw 16-bit samples are exact.
        let red = pixels::samples_u16(&l.channel_bytes(0).unwrap());
        assert_eq!(red, px.iter().step_by(4).copied().collect::<Vec<_>>());
    }
}

#[test]
fn grayscale_builder() {
    let px: Vec<u8> = (0..6).flat_map(|i| [i * 40, 255 - i]).collect();
    let mut b = PsdBuilder::new(3, 2).color_mode(ColorMode::Grayscale);
    b.push_layer(LayerSpec::new("g", 0, 0, 3, 2, PixelData::GrayA8(px.clone())));
    b.composite(PixelData::GrayA8(px.clone()));
    let f = built(&b);
    assert_eq!(f.header.color_mode, ColorMode::Grayscale);
    let img = f.layer(0).unwrap().rgba8().unwrap();
    for i in 0..6 {
        assert_eq!(&img.data[i * 4..i * 4 + 4], &[px[i * 2], px[i * 2], px[i * 2], px[i * 2 + 1]]);
    }
    let c = f.composite_rgba8().unwrap();
    assert_eq!(c.data, img.data);
    assert_eq!(f.header.channels, 2);
}

#[test]
fn grayscale16_builder() {
    let px: Vec<u16> = vec![0, 65535, 32768, 65535];
    let mut b = PsdBuilder::new(2, 1).color_mode(ColorMode::Grayscale).depth(16);
    b.push_layer(LayerSpec::new("g", 0, 0, 2, 1, PixelData::GrayA16(px)));
    let f = built(&b);
    assert_eq!(f.layer(0).unwrap().rgba8().unwrap().data, vec![0, 0, 0, 255, 128, 128, 128, 255]);
}

#[test]
fn cmyk_builder_inverts_ink() {
    // One pixel: C=0 M=0 Y=0 K=0 ink (white), one: K=255 (black).
    let px = vec![0, 0, 0, 0, 255, 0, 0, 0, 255, 255];
    let mut b = PsdBuilder::new(2, 1).color_mode(ColorMode::Cmyk);
    b.push_layer(LayerSpec::new("c", 0, 0, 2, 1, PixelData::Cmyka8(px.clone())));
    b.composite(PixelData::Cmyka8(px));
    let f = built(&b);
    assert_eq!(f.header.channels, 4);
    let l = f.layer(0).unwrap();
    assert_eq!(l.channel_bytes(0).unwrap(), vec![255, 255]); // stored inverted
    assert_eq!(l.channel_bytes(3).unwrap(), vec![255, 0]);
    assert_eq!(l.rgba8().unwrap().data, vec![255, 255, 255, 255, 0, 0, 0, 255]);
    assert_eq!(f.composite_rgba8().unwrap().data, vec![255, 255, 255, 255, 0, 0, 0, 255]);
}

#[test]
fn cmyk16_builder() {
    let px = vec![0u16, 0, 0, 0, 65535];
    let mut b = PsdBuilder::new(1, 1).color_mode(ColorMode::Cmyk).depth(16);
    b.push_layer(LayerSpec::new("c", 0, 0, 1, 1, PixelData::Cmyka16(px)));
    let f = built(&b);
    assert_eq!(f.layer(0).unwrap().rgba8().unwrap().data, vec![255, 255, 255, 255]);
}

#[test]
fn layer_properties_roundtrip() {
    let mut b = PsdBuilder::new(4, 4);
    let mut spec = LayerSpec::new("Props \u{1F3A8}", 1, 1, 2, 2, PixelData::Rgba8(vec![9; 16]));
    spec.blend_mode = BlendMode::SoftLight;
    spec.opacity = 128;
    spec.fill_opacity = Some(64);
    spec.visible = false;
    spec.clipping = true;
    spec.extra_blocks.push(TaggedBlock::sheet_color(4));
    b.push_layer(spec);
    let f = built(&b);
    let r = &f.layers()[0];
    assert_eq!(r.name(), "Props \u{1F3A8}");
    assert_eq!(r.name, b"Props ?");
    assert_eq!(r.blend_mode, BlendMode::SoftLight);
    assert_eq!(r.opacity, 128);
    assert_eq!(r.fill_opacity(), 64);
    assert!(!r.is_visible());
    assert_eq!(r.clipping, 1);
    assert_eq!(r.layer_id(), Some(1));
    assert_eq!(r.block(b"lclr").unwrap().parsed(), Some(Ok(BlockData::SheetColor(4))));
}

#[test]
fn all_blend_modes_via_builder() {
    let mut b = PsdBuilder::new(1, 1);
    for m in BlendMode::ALL {
        let mut s = LayerSpec::new(format!("{m:?}"), 0, 0, 1, 1, PixelData::Rgba8(vec![1, 2, 3, 4]));
        s.blend_mode = m;
        b.push_layer(s);
    }
    let f = built(&b);
    let modes: Vec<_> = f.layers().iter().map(|l| l.blend_mode).collect();
    assert_eq!(modes, BlendMode::ALL.to_vec());
}

#[test]
fn masks_roundtrip() {
    let mask_px: Vec<u8> = (0..12).map(|i| i * 20).collect();
    let mut s = LayerSpec::new("m", 0, 0, 3, 3, PixelData::Rgba8(vec![100; 36]));
    s.mask = Some(MaskSpec { rect: Rect::from_xywh(-1, 0, 4, 3), data: mask_px.clone(), default_color: 255, disabled: true });
    for depth in [8u16, 16] {
        let mut b = PsdBuilder::new(3, 3).depth(depth);
        let mut s = s.clone();
        if depth == 16 {
            s.pixels = PixelData::Rgba16(vec![25700; 36]);
        }
        b.push_layer(s);
        let f = built(&b);
        let l = f.layer(0).unwrap();
        let m = l.user_mask().unwrap().unwrap();
        assert_eq!((m.left, m.top, m.width, m.height), (-1, 0, 4, 3));
        assert_eq!(m.data, mask_px);
        assert_eq!(m.default_value, 255);
        assert!(l.record.layer_mask().unwrap().disabled());
        // Mask is not applied to rgba8.
        assert!(l.rgba8().unwrap().data.chunks(4).all(|p| p == [100, 100, 100, 100]));
    }
}

#[test]
fn no_mask_returns_none() {
    let mut b = PsdBuilder::new(1, 1);
    b.push_layer(LayerSpec::new("x", 0, 0, 1, 1, PixelData::Rgba8(vec![0; 4])));
    let f = built(&b);
    assert!(f.layer(0).unwrap().user_mask().is_none());
}

#[test]
fn empty_layer() {
    let mut b = PsdBuilder::new(2, 2);
    b.push_layer(LayerSpec::new("empty", 0, 0, 0, 0, PixelData::Rgba8(vec![])));
    let f = built(&b);
    let img = f.layer(0).unwrap().rgba8().unwrap();
    assert_eq!((img.width, img.height), (0, 0));
    assert!(img.data.is_empty());
}

#[test]
fn composite_supplied_with_alpha() {
    let px = vec![255, 0, 0, 128, 17, 89, 203, 0];
    let mut b = PsdBuilder::new(2, 1);
    b.push_layer(LayerSpec::new("x", 0, 0, 2, 1, PixelData::Rgba8(px.clone())));
    b.composite(PixelData::Rgba8(px.clone()));
    let f = built(&b);
    assert_eq!(f.header.channels, 4);
    assert!(f.layer_info.as_ref().unwrap().merged_alpha);
    assert!(f.merged_has_alpha());
    assert_eq!(f.has_real_merged_data(), Some(true));
    assert_eq!(f.decode_merged().unwrap(), [255, 17, 127, 89, 127, 203, 128, 0]);
    assert_eq!(f.composite_rgba8().unwrap().data, px);
}

#[test]
fn composite_opaque_drops_alpha() {
    let px: Vec<u8> = (0..6).flat_map(|i| [i, i, i, 255]).collect();
    let mut b = PsdBuilder::new(3, 2);
    b.composite(PixelData::Rgba8(px.clone()));
    let f = built(&b);
    assert_eq!(f.header.channels, 3);
    assert!(f.layer_info.is_none());
    assert_eq!(f.composite_rgba8().unwrap().data, px);
}

#[test]
fn composite_16bit() {
    let px: Vec<u16> = vec![65535, 0, 0, 65535, 0, 65535, 0, 32768];
    let mut b = PsdBuilder::new(2, 1).depth(16);
    b.composite(PixelData::Rgba16(px));
    let f = built(&b);
    assert_eq!(f.composite_rgba8().unwrap().data, vec![255, 0, 0, 255, 0, 255, 0, 128]);
}

#[test]
fn placeholder_composite_is_flagged() {
    let mut b = PsdBuilder::new(2, 2);
    b.push_layer(LayerSpec::new("x", 0, 0, 1, 1, PixelData::Rgba8(vec![0; 4])));
    let f = built(&b);
    assert_eq!(f.has_real_merged_data(), Some(false));
    assert!(f.composite_rgba8().unwrap().data.iter().all(|&v| v == 255));
}

#[test]
fn builder_resources() {
    let b = PsdBuilder::new(1, 1).resolution(300.0).icc_profile(vec![1, 2, 3]);
    let f = built(&b);
    assert!((f.resolution().unwrap().v_res() - 300.0).abs() < 1e-9);
    assert_eq!(f.icc_profile(), Some(&[1u8, 2, 3][..]));
}

#[test]
fn builder_errors() {
    let mut b = PsdBuilder::new(2, 2);
    b.push_layer(LayerSpec::new("bad", 0, 0, 2, 2, PixelData::Rgba8(vec![0; 3])));
    assert!(b.build().is_err());
    let mut b = PsdBuilder::new(2, 2);
    b.push_layer(LayerSpec::new("wrong fmt", 0, 0, 1, 1, PixelData::GrayA8(vec![0; 2])));
    assert!(b.build().is_err());
    let mut b = PsdBuilder::new(2, 2);
    b.begin_group(GroupSpec::new("open"));
    assert!(b.build().is_err());
    let mut b = PsdBuilder::new(2, 2);
    assert!(b.end_group().is_err());
    assert!(PsdBuilder::new(1, 1).depth(32).build().is_err());
    assert!(PsdBuilder::new(1, 1).color_mode(ColorMode::Lab).build().is_err());
    let mut b = PsdBuilder::new(2, 2);
    b.composite(PixelData::Rgba8(vec![0; 4]));
    assert!(b.build().is_err());
    let mut b = PsdBuilder::new(2, 2);
    let mut s = LayerSpec::new("m", 0, 0, 1, 1, PixelData::Rgba8(vec![0; 4]));
    s.mask = Some(MaskSpec { rect: Rect::from_xywh(0, 0, 2, 2), data: vec![0; 3], default_color: 0, disabled: false });
    b.push_layer(s);
    assert!(b.build().is_err());
}

#[test]
fn group_properties() {
    let mut b = PsdBuilder::new(1, 1);
    b.begin_group(GroupSpec { blend_mode: BlendMode::Multiply, opacity: 77, visible: false, ..GroupSpec::new("G") });
    b.push_layer(LayerSpec::new("x", 0, 0, 1, 1, PixelData::Rgba8(vec![0; 4])));
    b.end_group().unwrap();
    let f = built(&b);
    let g = &f.layers()[2];
    assert_eq!(g.name(), "G");
    assert_eq!(g.opacity, 77);
    assert!(!g.is_visible());
    let sd = g.section_divider().unwrap();
    assert_eq!(sd.kind, SectionType::OpenFolder);
    assert_eq!(sd.blend_mode, Some(BlendMode::Multiply));
    assert_eq!(f.layers()[0].section_type(), SectionType::BoundingDivider);
}

#[test]
fn psb_builder_large_canvas_header() {
    let b = PsdBuilder::new(40_000, 1).version(Version::Psb).compression(Compression::Rle);
    let f = built(&b);
    assert_eq!(f.header.width, 40_000);
    assert!(f.validate().is_ok());
}

#[test]
fn iter_layers_indices() {
    let mut b = PsdBuilder::new(1, 1);
    for i in 0..3 {
        b.push_layer(LayerSpec::new(format!("l{i}"), 0, 0, 1, 1, PixelData::Rgba8(vec![0; 4])));
    }
    let f = built(&b);
    let v: Vec<_> = f.iter_layers().map(|l| (l.index, l.name())).collect();
    assert_eq!(v, vec![(0, "l0".into()), (1, "l1".into()), (2, "l2".into())]);
    assert!(f.layer(3).is_none());
    assert_eq!(f.layer(0).unwrap().version(), Version::Psd);
}
