//! A PSD's Background is recognised by its structure (the bottom record, transparency-protected,
//! stored without a transparency channel), whatever its name in the Photoshop language that
//! saved the file.

use photocraft_io::psd_to_document;
use photocraft_psd::{ColorMode, Compression, PsdFile, TaggedBlock, Version, testgen};

/// `testgen::layered` with its bottom layer turned into a Photoshop Background named `name`.
fn file(name: &str, transparency_channel: bool) -> PsdFile {
    let mut f = testgen::layered(Version::Psd, ColorMode::Rgb, 8, Compression::Rle);
    let bottom = f.layers_mut().first_mut().unwrap();
    bottom.name = name.bytes().filter(u8::is_ascii).collect();
    bottom.blocks.retain(|b| &b.key != b"luni" && &b.key != b"lspf");
    bottom.blocks.push(TaggedBlock::unicode_name(name));
    bottom.flags.set_transparency_protected(true);
    if !transparency_channel {
        bottom.channels.retain(|c| c.id != -1);
    }
    f
}

#[test]
fn a_localized_background_is_the_background() {
    for name in ["Hintergrund", "背景", "Arrière-plan"] {
        let (d, _) = psd_to_document(&file(name, false));
        let bg = d.background().unwrap_or_else(|| panic!("{name}: no Background"));
        assert_eq!(bg.name, name);
        assert!(bg.locks.transparency && bg.locks.position, "{name}");
        assert_eq!(bg.id, d.layers[0].id);
    }
}

#[test]
fn a_protected_layer_with_transparency_is_not_the_background() {
    // Lock Transparent Pixels on an ordinary bottom layer: it keeps its transparency channel.
    let (d, _) = psd_to_document(&file("Background", true));
    assert!(d.layers[0].locks.transparency);
    assert!(d.background().is_none());
}
