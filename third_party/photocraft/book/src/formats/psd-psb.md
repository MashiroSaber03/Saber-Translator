# PSD and PSB

`photocraft-psd` is a standalone clean-room reader and writer based on Adobe's public PSD specification. It supports PSD version 1 and PSB version 2, preserves unknown blocks for round-trip fidelity, and decodes channel data lazily.

The parser accepts all four PSD compression methods handled by the crate and represents format-level structures before `photocraft-io` maps them to `photocraft-doc`. The current public API and fidelity behavior are documented in [`crates/psd/README.md`](https://github.com/storytold/photocraft/blob/main/crates/psd/README.md).

## Implemented parser protections

Current source enforces several explicit limits:

- document width and height: at most 300,000 pixels each;
- document channels: at most 56;
- layer channels: at most 64;
- decoded channel data: at most 2 GiB for one decode layout;
- ActionDescriptor nesting: at most 64 levels;
- pattern edge: at most 30,000 pixels;
- arithmetic for row counts and decoded lengths uses checked operations.

Malformed-input tests, property tests, and `cargo-fuzz` targets cover the main parser and ActionDescriptor parser. These controls reduce risk but do not prove the parser safe for every hostile file or bound total work across every embedded object.

## Remaining hardening

Priorities include broader PSB and embedded-object fuzz corpora, measuring aggregate allocations and parse time, ensuring every nested length is budgeted, and retaining minimized crashing inputs as regression fixtures. See [Fuzzing and regression testing](../security/fuzzing.md).
