//! ICC / EXIF / XMP / DPI / text preservation, per format, per caps.

mod common;
use common::*;
use photocraft_codecs::*;

fn rich_image(layout: ChannelLayout) -> Image {
    let mut img = test_image(layout, SampleType::U8);
    img.icc = Some(sample_icc(3000));
    img.meta = Metadata {
        exif: Some(sample_exif()),
        xmp: Some(SAMPLE_XMP.to_owned()),
        dpi: Some((300.0, 150.0)),
        text: vec![("Description".into(), "synthetic test image".into()), ("Software".into(), "photocraft".into())],
        ..Default::default()
    };
    img
}

fn check_format(format: Format) {
    let c = caps(format);
    if !(c.read && c.write) {
        return;
    }
    let layout = if c.layouts.contains(&ChannelLayout::Rgb) { ChannelLayout::Rgb } else { c.layouts[0] };
    let img = rich_image(layout);
    let bytes = encode(&img, format, &EncodeOptions::default()).unwrap();
    let back = decode(&bytes).unwrap();
    let warnings = fidelity_warnings(&img, format);

    assert_eq!(back.icc.is_some(), c.icc, "{format:?} icc presence");
    if c.icc {
        assert_eq!(back.icc, img.icc, "{format:?} icc bytes");
    } else {
        assert!(warnings.contains(&FidelityWarning::IccDropped), "{format:?}");
    }
    if c.exif && format == Format::Tiff {
        // TIFF writes the block's directories as its own and rebuilds a block from them on
        // reading (tests/tiff_exif.rs); IFD0's orientation here is the image's.
        assert!(!warnings.contains(&FidelityWarning::ExifDropped), "{format:?}");
    } else if c.exif {
        assert_eq!(back.meta.exif, img.meta.exif, "{format:?} exif");
    } else {
        assert!(warnings.contains(&FidelityWarning::ExifDropped), "{format:?}");
    }
    if c.xmp {
        assert_eq!(back.meta.xmp, img.meta.xmp, "{format:?} xmp");
    } else {
        assert!(back.meta.xmp.is_none());
        assert!(warnings.contains(&FidelityWarning::XmpDropped), "{format:?}");
    }
    if c.dpi {
        let (x, y) = back.meta.dpi.unwrap_or_else(|| panic!("{format:?} lost dpi"));
        assert!((x - 300.0).abs() < 0.05 && (y - 150.0).abs() < 0.05, "{format:?} dpi {x},{y}");
    } else {
        assert!(warnings.contains(&FidelityWarning::DpiDropped), "{format:?}");
    }
    if c.text {
        for kv in &img.meta.text {
            assert!(back.meta.text.contains(kv), "{format:?} lost text {kv:?}: {:?}", back.meta.text);
        }
    } else {
        assert!(back.meta.text.is_empty());
        assert!(warnings.contains(&FidelityWarning::TextDropped), "{format:?}");
    }
}

#[test]
fn png_metadata() {
    check_format(Format::Png);
}
#[test]
fn jpeg_metadata() {
    check_format(Format::Jpeg);
}
#[test]
fn tiff_metadata() {
    check_format(Format::Tiff);
}
#[test]
fn webp_metadata() {
    check_format(Format::WebP);
}
#[test]
fn gif_metadata() {
    check_format(Format::Gif);
}
#[test]
fn bmp_metadata() {
    check_format(Format::Bmp);
}
#[test]
fn tga_metadata() {
    check_format(Format::Tga);
}
#[test]
fn ico_metadata() {
    check_format(Format::Ico);
}
#[test]
fn fits_metadata() {
    check_format(Format::Fits);
}
#[test]
fn pnm_metadata() {
    check_format(Format::Pnm);
}
#[test]
fn qoi_metadata() {
    check_format(Format::Qoi);
}
#[test]
fn exr_metadata() {
    check_format(Format::OpenExr);
}
#[test]
fn hdr_metadata() {
    check_format(Format::Hdr);
}

#[test]
fn metadata_on_16bit_png_and_tiff() {
    for f in [Format::Png, Format::Tiff] {
        let mut img = test_image(ChannelLayout::Rgba, SampleType::U16);
        img.icc = Some(sample_icc(512));
        img.meta.dpi = Some((72.0, 72.0));
        let back = decode(&encode(&img, f, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(back.icc, img.icc);
        assert_eq!(back.sample_type(), SampleType::U16);
    }
}

#[test]
fn icc_on_cmyk_tiff_and_jpeg() {
    for f in [Format::Tiff, Format::Jpeg] {
        let img = test_image(ChannelLayout::Cmyk, SampleType::U8).with_icc(Some(sample_icc(900)));
        let back = decode(&encode(&img, f, &EncodeOptions::default()).unwrap()).unwrap();
        assert_eq!(back.layout(), ChannelLayout::Cmyk, "{f:?}");
        assert_eq!(back.icc, img.icc, "{f:?}");
    }
}

#[test]
fn cmyk_icc_not_written_to_rgb_format() {
    let img = test_image(ChannelLayout::Cmyk, SampleType::U8).with_icc(Some(sample_icc(900)));
    let back = decode(&encode(&img, Format::Png, &EncodeOptions::default()).unwrap()).unwrap();
    assert!(back.icc.is_none(), "a CMYK profile must not be attached to converted RGB data");
    assert!(fidelity_warnings(&img, Format::Png).contains(&FidelityWarning::IccDropped));
}

#[test]
fn jpeg_multi_segment_icc() {
    // > 65519 bytes forces several APP2 segments.
    for len in [65519, 65520, 150_000, 300_001] {
        let img = test_image(ChannelLayout::Rgb, SampleType::U8).with_icc(Some(sample_icc(len)));
        let bytes = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
        let app2 = bytes.windows(14).filter(|w| w[0] == 0xFF && w[1] == 0xE2 && &w[4..14] == b"ICC_PROFIL").count();
        assert_eq!(app2, len.div_ceil(65519), "segments for {len}");
        assert_eq!(decode(&bytes).unwrap().icc.map(|v| v.len()), Some(len));
        // Oracle: the image crate reads the same profile.
        use image::ImageDecoder;
        let mut d = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
        assert_eq!(d.icc_profile().unwrap(), img.icc);
    }
}

/// A layered document's XMP can list every document ever placed in it and the text of every type
/// layer: far more than the 64 KB one JPEG segment holds. The JPEG is still written, with that
/// bookkeeping left out of the XMP; metadata too large even then is dropped with a warning.
#[test]
fn jpeg_with_oversized_metadata_still_encodes() {
    let ancestors: String = (0..3000).map(|i| format!("<rdf:li>xmp.did:{i:032x}</rdf:li>")).collect();
    let texts: String = (0..400)
        .map(|i| format!("<rdf:li rdf:parseType=\"Resource\"><ps:LayerName>Layer {i}</ps:LayerName><ps:LayerText>Words on layer {i}</ps:LayerText></rdf:li>"))
        .collect();
    let xmp = format!(
        "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?><x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\" xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\" xmlns:ps=\"http://example.com/ps/1.0/\" xmp:CreatorTool=\"synthetic\"><ps:DocumentAncestors>\n<rdf:Bag>{ancestors}</rdf:Bag></ps:DocumentAncestors><ps:TextLayers><rdf:Bag>{texts}</rdf:Bag></ps:TextLayers><ps:DocumentAncestorsNote/></rdf:Description></rdf:RDF></x:xmpmeta><?xpacket end=\"w\"?>"
    );
    assert!(xmp.len() > 120_000, "{}", xmp.len());
    for sample in [SampleType::U8, SampleType::U16, SampleType::F32] {
        let mut img = test_image(ChannelLayout::Rgb, sample);
        img.meta.xmp = Some(xmp.clone());
        let bytes = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
        let back = decode(&bytes).unwrap().meta.xmp.unwrap();
        assert!(back.contains("xmp:CreatorTool=\"synthetic\"") && back.ends_with("<?xpacket end=\"w\"?>"), "{sample:?}");
        assert!(!back.contains("DocumentAncestors>") && !back.contains("TextLayers"), "{sample:?}");
        assert!(back.contains("<ps:DocumentAncestorsNote/>"), "{sample:?}: a longer name is not the element");
        assert!(!fidelity_warnings(&img, Format::Jpeg).iter().any(|w| matches!(w, FidelityWarning::MetadataTooLarge { .. })));
        // Formats without a segment limit keep the packet whole.
        assert_eq!(decode(&encode(&img, Format::Png, &EncodeOptions::default()).unwrap()).unwrap().meta.xmp.as_deref(), Some(xmp.as_str()));
    }
    // Still too large: the JPEG is written without it.
    let mut img = test_image(ChannelLayout::Rgb, SampleType::U8);
    let big = format!("<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><!-- {} --></x:xmpmeta>", "x".repeat(70_000));
    let mut exif = sample_exif();
    exif.resize(70_000, 0);
    img.meta.xmp = Some(big.clone());
    img.meta.exif = Some(exif);
    let back = decode(&encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap()).unwrap();
    assert!(back.meta.xmp.is_none() && back.meta.exif.is_none());
    let w = fidelity_warnings(&img, Format::Jpeg);
    assert!(w.contains(&FidelityWarning::MetadataTooLarge { what: "XMP", bytes: big.len() }), "{w:?}");
    assert!(w.contains(&FidelityWarning::MetadataTooLarge { what: "EXIF", bytes: 70_000 }), "{w:?}");
}

#[test]
fn embed_flags_disable_metadata() {
    let img = rich_image(ChannelLayout::Rgb);
    let opts = EncodeOptions { embed_icc: false, embed_metadata: false, ..Default::default() };
    for f in [Format::Png, Format::Jpeg, Format::Tiff, Format::WebP] {
        let back = decode(&encode(&img, f, &opts).unwrap()).unwrap();
        assert!(back.icc.is_none(), "{f:?}");
        assert!(back.meta.exif.is_none() && back.meta.xmp.is_none(), "{f:?}");
        assert!(back.meta.text.is_empty(), "{f:?}");
        assert!(fidelity_warnings_with(&img, Format::Bmp, &opts).iter().all(|w| !matches!(w, FidelityWarning::IccDropped | FidelityWarning::ExifDropped)));
    }
}

#[test]
fn png_unicode_text_uses_itxt() {
    let mut img = test_image(ChannelLayout::Rgb, SampleType::U8);
    img.meta.text = vec![("Title".into(), "Ünïcödé ☃ 雪".into()), ("Author".into(), "latin1 ok é".into())];
    let bytes = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    assert!(bytes.windows(4).any(|w| w == b"iTXt"));
    assert!(bytes.windows(4).any(|w| w == b"tEXt"));
    let mut back = decode(&bytes).unwrap().meta.text;
    let mut want = img.meta.text.clone();
    back.sort();
    want.sort();
    assert_eq!(back, want);
}

#[test]
fn png_xmp_in_itxt_keyword() {
    let mut img = test_image(ChannelLayout::Gray, SampleType::U8);
    img.meta.xmp = Some(SAMPLE_XMP.into());
    let bytes = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    assert!(bytes.windows(17).any(|w| w == b"XML:com.adobe.xmp"));
    let back = decode(&bytes).unwrap();
    assert_eq!(back.meta.xmp.as_deref(), Some(SAMPLE_XMP));
    assert!(back.meta.text.is_empty(), "XMP must not leak into text");
}

#[test]
fn png_exif_chunk_written() {
    let mut img = test_image(ChannelLayout::Rgb, SampleType::U8);
    img.meta.exif = Some(sample_exif());
    let bytes = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    assert!(bytes.windows(4).any(|w| w == b"eXIf"));
    assert!(!bytes.windows(4).any(|w| w == b"iCCP"));
}

#[test]
fn jpeg_exif_readable_by_oracle() {
    let mut img = test_image(ChannelLayout::Rgb, SampleType::U8);
    img.meta.exif = Some(sample_exif());
    let bytes = encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap();
    use image::ImageDecoder;
    let mut d = image::codecs::jpeg::JpegDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
    let exif = d.exif_metadata().unwrap().unwrap();
    assert!(exif.ends_with(&sample_exif()));
}

#[test]
fn jpeg_dpi_in_jfif() {
    let mut img = test_image(ChannelLayout::Gray, SampleType::U8);
    img.meta.dpi = Some((96.0, 200.0));
    let back = decode(&encode(&img, Format::Jpeg, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(back.meta.dpi, Some((96.0, 200.0)));
}

#[test]
fn tiff_icc_readable_by_oracle() {
    let img = test_image(ChannelLayout::Rgb, SampleType::U16).with_icc(Some(sample_icc(1234)));
    let bytes = encode(&img, Format::Tiff, &EncodeOptions::default()).unwrap();
    use image::ImageDecoder;
    let mut d = image::codecs::tiff::TiffDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
    assert_eq!(d.icc_profile().unwrap(), img.icc);
}

#[test]
fn png_icc_readable_by_oracle() {
    let img = test_image(ChannelLayout::Rgba, SampleType::U8).with_icc(Some(sample_icc(777)));
    let bytes = encode(&img, Format::Png, &EncodeOptions::default()).unwrap();
    use image::ImageDecoder;
    let mut d = image::codecs::png::PngDecoder::new(std::io::Cursor::new(&bytes)).unwrap();
    assert_eq!(d.icc_profile().unwrap(), img.icc);
}

#[test]
fn metadata_survives_depth_conversion() {
    let mut img = test_image(ChannelLayout::Rgb, SampleType::F32).with_icc(Some(sample_icc(300)));
    img.meta.xmp = Some(SAMPLE_XMP.into());
    let back = decode(&encode(&img, Format::Png, &EncodeOptions::default()).unwrap()).unwrap();
    assert_eq!(back.icc, img.icc);
    assert_eq!(back.meta.xmp, img.meta.xmp);
}
