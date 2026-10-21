# Fuzzing and regression testing

PhotoCraft already has `cargo-fuzz` workspaces next to its primary untrusted-format crates:

| Crate | Targets |
|---|---|
| `crates/psd` | `parse`, `descriptor` |
| `crates/codecs` | `decode_any`, `decode_png`, `decode_jpeg`, `decode_tiff`, `decode_exr` |
| `crates/format` | `load` |

The main CI workflow can run these targets through a manually dispatched fuzz job. It is not currently scheduled continuous fuzzing.

Run a target from its crate directory with nightly Rust and `cargo-fuzz` installed:

```sh
cd crates/psd
cargo fuzz run parse
```

The normal test suites complement libFuzzer with property tests, structured malformed cases, truncation sweeps, bit mutation, bomb-like headers, archive corruption, and command `panic_hunt` coverage.

## Security properties

A useful fuzz target should detect more than memory-safety crashes. Relevant failures include:

- panic or abort;
- integer overflow or out-of-bounds access;
- unreasonable allocation or decompression expansion;
- timeout, infinite loop, or algorithmic complexity spike;
- recursion/stack exhaustion;
- malformed input accepted into an invalid internal state;
- inconsistent results between detection and explicit-format decode;
- corrupted `.pcraft` content accepted despite hash/CRC mismatch.

## Corpus policy

Every confirmed defect should produce a minimized, permanent regression test when disclosure and licensing allow it. Store only synthetic or redistributable inputs. Do not commit personal documents, customer data, credentials, proprietary assets, or an undisclosed weaponized sample.

A regression entry should record the affected parser, expected error class, resource budget, and issue/advisory reference after disclosure. Fuzz crashes must be reproduced under a normal test before the fix is considered durable.

## Future program

Priority additions are PSB-specific structures, WebP, native directory bundles, ICC/LUT/pattern parsers, and embedded smart objects. Continuous scheduled fuzzing and artifact triage remain future work.
