//! Fuzz the pattern parsers: the `Patt` / `Pat2` / `Pat3` tagged blocks and standalone `.pat`
//! files. Any input must return `Ok` or `Err`, never panic. The channel offsets and padding a
//! hostile file can carry are wire artifacts, not fields of `PsdPattern`, so a parsed pattern must
//! survive a write/parse round trip unchanged -- a stronger invariant than byte stability, which
//! would also hold if the writer silently normalised a field away.
#![no_main]
use libfuzzer_sys::fuzz_target;
use photocraft_psd::patterns::{parse_pat_file, parse_pattern_block, write_pat_file, write_pattern_block};
fuzz_target!(|data: &[u8]| {
    if let Ok(ps) = parse_pattern_block(data) {
        let out = write_pattern_block(&ps).expect("a parsed pattern block must be writable");
        let back = parse_pattern_block(&out).expect("a written pattern block must re-parse");
        assert_eq!(back, ps, "a pattern block does not survive a write/parse round trip");
    }
    if let Ok(ps) = parse_pat_file(data) {
        let out = write_pat_file(&ps).expect("a parsed .pat must be writable");
        let back = parse_pat_file(&out).expect("a written .pat must re-parse");
        assert_eq!(back, ps, "a .pat file does not survive a write/parse round trip");
    }
});
