//! Fuzz the typed image-resource views: `ImageResource::parsed` for an arbitrary resource id, plus
//! the standalone `ResolutionInfo` and `VersionInfo` parsers. All must return `Ok` / `Err` / `None`
//! (and `xmp_str` an `Option`), never panic. A parsed `ResolutionInfo` (16 fixed bytes) must be
//! stable across a byte round trip. The first two bytes pick the id so every typed branch is
//! reachable; the rest is the resource data.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::resources::{ImageResource, ResolutionInfo, VersionInfo};
fuzz_target!(|data: &[u8]| {
    let (id, body) = match data.split_first_chunk::<2>() {
        Some((p, rest)) => (u16::from_be_bytes(*p), rest.to_vec()),
        None => (0u16, data.to_vec()),
    };
    let r = ImageResource::new(id, body.clone());
    let _ = r.parsed();
    let _ = r.xmp_str();
    if let Ok(ri) = ResolutionInfo::from_bytes(&body) {
        let round = ResolutionInfo::from_bytes(&ri.to_bytes()).expect("resolution info re-parses");
        assert_eq!(round, ri, "ResolutionInfo is not stable across a round trip");
    }
    if let Ok(vi) = VersionInfo::from_bytes(&body) {
        let _ = vi.to_bytes();
    }
});
