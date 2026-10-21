//! Fuzz the `FEid` / `FXid` smart-filter cache parser: any input must return `Ok` or `Err`, never
//! panic. Decoding every cached plane and the filter mask must also stay panic-free (a size the
//! data cannot hold is refused before allocation, for RLE and zlib alike), and a parsed block must
//! re-parse to the same value and re-serialise byte for byte.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::filter_effects::FilterEffects;
fuzz_target!(|data: &[u8]| {
    let Ok(fx) = FilterEffects::parse(data) else { return };
    for it in &fx.items {
        let _ = it.decoded_mask();
        if let (Ok(depth), Ok((w, h))) = (u16::try_from(it.depth), it.rect.size()) {
            for p in it.slots.iter().flatten() {
                let _ = p.decode(w, h, depth);
            }
        }
    }
    let out = fx.to_bytes();
    let back = FilterEffects::parse(&out).expect("a written FEid block must re-parse");
    assert_eq!(back, fx, "a written FEid block does not re-parse to the same value");
    assert_eq!(back.to_bytes(), out, "a second round trip is not byte-stable");
});
