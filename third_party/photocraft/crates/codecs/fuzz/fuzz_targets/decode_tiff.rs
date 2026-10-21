#![no_main]

use libfuzzer_sys::fuzz_target;
use photocraft_codecs::{decode_as_with, decode_tiff_page, exif_orientation, tiff_info, tiff_orientation, DecodeOptions, Format, Limits};

fuzz_target!(|data: &[u8]| {
    let opts = DecodeOptions {
        limits: Limits { max_width: 4096, max_height: 4096, max_pixels: 1 << 22, max_alloc: 256 << 20 },
        ..Default::default()
    };
    let _ = decode_as_with(Format::Tiff, data, &opts);
    let _ = exif_orientation(data);
    let _ = tiff_orientation(data);
    // Every directory of the chain and its SubIFDs (BigTIFF too), not only the default page.
    if let Ok(info) = tiff_info(data) {
        for page in 0..info.pages.len().min(16) {
            let _ = decode_tiff_page(data, page, &opts);
        }
    }
});
