//! Fuzz the per-block views of a tagged block: `parsed` (typed, well-known keys) and
//! `check_structure` (strict re-parse of blocks that carry their own inner sizes: placed layers,
//! linked files, object effects, filter caches). Both must return `Ok` / `Err` / `None`, never
//! panic, for arbitrary data.
//!
//! The first byte selects the key from a table of every key with a typed or strict branch, so the
//! fuzzer reaches each branch deterministically instead of having to guess a 4-byte key from raw
//! bytes (which it would almost never do); the rest is the block data.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::tagged::TaggedBlock;

const KEYS: [&[u8; 4]; 20] = [
    // typed (parsed)
    b"luni", b"lsct", b"lsdk", b"lyid", b"lnsr", b"clbl", b"infx", b"knko", b"tsly", b"lmgm", b"vmgm", b"lspf", b"lclr", b"iOpa", b"shmd",
    // strict (check_structure)
    b"PlLd", b"SoLd", b"lnk2", b"lfx2", b"FEid",
];

fuzz_target!(|data: &[u8]| {
    let (key, body) = match data.split_first() {
        Some((&sel, rest)) => (*KEYS[sel as usize % KEYS.len()], rest.to_vec()),
        None => (*b"luni", Vec::new()),
    };
    let b = TaggedBlock::new(key, body);
    let _ = b.parsed();
    let _ = b.check_structure();
});
