//! Multi-page TIFF and BigTIFF open through `photocraft-io`: the default page is the first
//! full-resolution one (like Photoshop), any page can be opened by index, and a BigTIFF's
//! orientation is applied (#693).

/// A TIFF (classic or BigTIFF, little-endian) with one 8-bit gray page per `(w, h, level,
/// NewSubfileType, orientation)`, chained in order.
fn tiff(big: bool, pages: &[(u32, u32, u8, u32, u16)]) -> Vec<u8> {
    let mut f = Vec::new();
    f.extend_from_slice(b"II");
    if big {
        f.extend_from_slice(&43u16.to_le_bytes());
        f.extend_from_slice(&8u16.to_le_bytes());
        f.extend_from_slice(&0u16.to_le_bytes());
        f.extend_from_slice(&0u64.to_le_bytes());
    } else {
        f.extend_from_slice(&42u16.to_le_bytes());
        f.extend_from_slice(&0u32.to_le_bytes());
    }
    let mut next_field = if big { 8 } else { 4 };
    for &(w, h, level, kind, orientation) in pages {
        let pixels_at = f.len() as u64;
        f.extend(std::iter::repeat_n(level, (w * h) as usize));
        if f.len() % 2 == 1 {
            f.push(0);
        }
        let ifd_at = f.len() as u64;
        let width = if big { 8 } else { 4 };
        let put = |f: &mut Vec<u8>, at: usize, v: u64| f[at..at + width].copy_from_slice(&v.to_le_bytes()[..width]);
        put(&mut f, next_field, ifd_at);
        let entries: [(u16, u16, u64); 11] = [
            (254, 4, u64::from(kind)),
            (256, 4, u64::from(w)),
            (257, 4, u64::from(h)),
            (258, 3, 8),
            (259, 3, 1),
            (262, 3, 1),
            (273, 4, pixels_at),
            (274, 3, u64::from(orientation)),
            (277, 3, 1),
            (278, 4, u64::from(h)),
            (279, 4, u64::from(w * h)),
        ];
        if big {
            f.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        } else {
            f.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        }
        for (tag, ty, v) in entries {
            f.extend_from_slice(&tag.to_le_bytes());
            f.extend_from_slice(&ty.to_le_bytes());
            if big {
                f.extend_from_slice(&1u64.to_le_bytes());
                f.extend_from_slice(&v.to_le_bytes());
            } else {
                f.extend_from_slice(&1u32.to_le_bytes());
                f.extend_from_slice(&(v as u32).to_le_bytes());
            }
        }
        next_field = f.len();
        f.extend(std::iter::repeat_n(0, width));
    }
    f
}

#[test]
fn the_first_full_resolution_page_opens_and_any_page_on_request() {
    for big in [false, true] {
        // A thumbnail, then two pages.
        let bytes = tiff(big, &[(2, 2, 9, 1, 1), (6, 4, 50, 0, 1), (3, 5, 90, 2, 1)]);
        let r = photocraft_io::import("pages.tif", &bytes).unwrap();
        assert_eq!((r.document.size.width, r.document.size.height), (6, 4), "big {big}");
        assert!(r.warnings.iter().any(|w| w == "only the first of 2 pages was imported"), "{:?}", r.warnings);
        let info = photocraft_codecs::tiff_info(&bytes).unwrap();
        assert_eq!((info.pages.len(), info.default_page(), info.big_tiff), (3, Some(1), big));
        for (page, size) in [(0, (2, 2)), (1, (6, 4)), (2, (3, 5))] {
            let r = photocraft_io::import_tiff_page("pages.tif", &bytes, Some(page)).unwrap();
            assert_eq!((r.document.size.width, r.document.size.height), size, "page {page} big {big}");
        }
        assert!(photocraft_io::import_tiff_page("pages.tif", &bytes, Some(3)).is_err());
    }
}

#[test]
fn bigtiff_orientation_is_applied_on_open() {
    // #693: Orientation 6 turns the stored 6×4 landscape into a 4×6 portrait, in both containers.
    for big in [false, true] {
        let bytes = tiff(big, &[(6, 4, 50, 0, 6)]);
        let r = photocraft_io::import("turned.tif", &bytes).unwrap();
        assert_eq!((r.document.size.width, r.document.size.height), (4, 6), "big {big}");
        let r = photocraft_io::import_tiff_page("turned.tif", &bytes, Some(0)).unwrap();
        assert_eq!((r.document.size.width, r.document.size.height), (4, 6), "big {big}");
    }
}
