//! #647: Export As's Metadata: None (`XmpEmbed::None`) embeds none of the document's XMP (the
//! packet lists the text of every type layer and one id per placed document); the default,
//! `XmpEmbed::All`, keeps it as Save As does, and layered saves (PSD, `.pcraft`) always keep it.

use photocraft_color::{ColorMode, SampleType};
use photocraft_doc::{Document, Layer, LayerContent, Size};
use photocraft_io::{ExportOptions, XmpEmbed, export};
use photocraft_raster::Surface;

const XMP: &str = "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF>\
<photoshop:DocumentAncestors><rdf:Bag><rdf:li>xmp.did:11111111-2222-3333-4444-555555555555</rdf:li></rdf:Bag></photoshop:DocumentAncestors>\
<photoshop:TextLayers><rdf:Bag><rdf:li>Secret campaign copy</rdf:li></rdf:Bag></photoshop:TextLayers>\
</rdf:RDF></x:xmpmeta>";

fn doc_with_xmp() -> Document {
    let mut d = Document::new("x", Size::new(8, 6), ColorMode::Rgb, SampleType::U8);
    let mut s = Surface::new(d.pixel_format());
    s.fill_rect(photocraft_geom::Rect::new(0, 0, 8, 6), &[0.5, 0.4, 0.3, 1.0]);
    d.layers.push(Layer::new("flat", LayerContent::Raster(s)));
    d.metadata.xmp = Some(XMP.to_string());
    d
}

fn leaks_xmp(bytes: &[u8]) -> bool {
    let s = String::from_utf8_lossy(bytes);
    s.contains("x:xmpmeta") || s.contains("DocumentAncestors") || s.contains("Secret campaign copy")
}

#[test]
fn metadata_none_embeds_no_xmp() {
    let opts = ExportOptions { xmp: XmpEmbed::None, ..ExportOptions::default() };
    for ext in ["png", "jpg", "webp", "tif"] {
        let r = export(&doc_with_xmp(), &format!("out.{ext}"), &opts).expect(ext);
        assert!(!leaks_xmp(&r.bytes), "{ext}: the document XMP leaked into the export");
    }
}

#[test]
fn all_embeds_the_whole_packet_and_is_the_default() {
    assert_eq!(ExportOptions::default().xmp, XmpEmbed::All);
    let opts = ExportOptions { xmp: XmpEmbed::All, ..ExportOptions::default() };
    for ext in ["png", "jpg", "webp"] {
        let r = export(&doc_with_xmp(), &format!("out.{ext}"), &opts).expect(ext);
        assert!(leaks_xmp(&r.bytes), "{ext}: XmpEmbed::All must embed the packet");
    }
}

#[test]
fn layered_saves_keep_the_xmp_whatever_the_option() {
    for opts in [ExportOptions::default(), ExportOptions { xmp: XmpEmbed::None, ..ExportOptions::default() }] {
        let psd = export(&doc_with_xmp(), "out.psd", &opts).expect("psd");
        assert!(leaks_xmp(&psd.bytes), "PSD always keeps the document XMP");
        let pcraft = export(&doc_with_xmp(), "out.pcraft", &opts).expect("pcraft");
        assert!(leaks_xmp(&pcraft.bytes), ".pcraft always keeps the document XMP");
    }
}
