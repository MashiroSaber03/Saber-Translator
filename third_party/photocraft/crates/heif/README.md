# photocraft-heif

The optional HEIF/HEIC decoder of PhotoCraft: iPhone and Mac photos. A thin, panic-guarded wrapper
around [`heic-rs`](https://github.com/tbraun96/heic-rs), a pure-Rust HEVC still-picture decoder
(no `unsafe`, MIT OR Apache-2.0): single pictures and grid-tiled photos, 8- and 10-bit (10-bit
decodes to 16-bit), alpha auxiliary images, the container's rotation/mirror/crop, ICC, EXIF and XMP.
Read-only: writing would need an HEVC encoder.

```rust
let info = photocraft_heif::probe(&bytes)?;             // size, depth, alpha: check limits first
let img = photocraft_heif::decode(&bytes, &photocraft_heif::Options::default())?;
// img.width, img.height, img.has_alpha, img.sixteen_bit, img.data (RGB/RGBA), img.icc, img.exif, img.xmp
```

Errors are `Error::Unsupported` (image sequences, overlays, …), `Error::Limit` or
`Error::Malformed`. It never panics: every call into heic-rs runs under `catch_unwind`, and a panic
inside it (fuzzing found two in 0.1.1) becomes `Error::Malformed`.

## Why it is a separate, optional crate

- **A young decoder.** heic-rs is new and has a single maintainer. It is pinned exactly
  (`=0.1.1`); moving the pin is a reviewed change (re-run `cargo fuzz run decode_heif` in
  `crates/codecs` and `cargo xtask test-corpus`).
- **HEVC patents are a distributor's call.** HEVC is patent-encumbered in some jurisdictions, so
  whether a build includes an HEVC decoder is a build-time choice. `photocraft-codecs` uses this
  crate only behind its non-default `heif` feature; the apps forward it
  (`cargo build -p photocraft --features heif`). Official PhotoCraft builds and CI enable it.
  Without it, HEIC files are still recognised and opening one is a clear "HEIC/HEIF support isn't
  included in this build" error.

The crate depends on no other PhotoCraft crate and knows nothing of `photocraft-codecs`' types;
`codecs` → `heif` is the one allowed dependency between standalone crates (`cargo xtask layers`).
Tests on real files live in `photocraft-codecs` (`tests/heif.rs`, features `corpus,heif`, files
fetched by `cargo xtask corpus --heif`).
