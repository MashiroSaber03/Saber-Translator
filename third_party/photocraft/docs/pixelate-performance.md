# Facet and Color Halftone performance

2026-10-10. Two changes remove repeated work from the existing Pixelate filters:

- **Facet:** calculate luma once per source pixel and calculate each 3x3 quadrant once. A three-row ring shares that quadrant with its four output pixels. The nine-sample addition order, variance calculation, quadrant order and strict tie comparison stay the same.
- **Color Halftone:** memoize the five-tap ink calculation and dot radius in the rotated screen grid, separately for each channel. Pixels still use the same candidate dots, distances and anti-alias coverage. Lab conversion, CMYK ink direction and alpha handling stay the same.

Facet uses about 291 KiB of additional scratch for a 256x256 output tile. The luma plane is capped at 1,048,576 samples. Halftone tables are capped at 2 MiB per channel. Scratch allocations are fallible; an incomplete halo, oversized table or unsuitable screen coordinates uses the original calculation. The default tiled filter path stays parallel on native targets and single-threaded on wasm.

## Full filter timings

EC2 m7i.4xlarge, 16 vCPUs (Intel Xeon Platinum 8488C), Ubuntu 24.04, rustc 1.99.0, default release profile, `RAYON_NUM_THREADS=16`. Base: `7722172585a01cbdb93c06f0f5ff2634fcb17999`. The changed algorithms are in `1bc3a9f1`. Before and after were built in separate target directories; no other builds or benchmarks ran during measurement.

Each entry is the median of three alternating pairs: before/after, after/before, before/after. The timer covers `apply_in`, including tile reads, filtering, assembly, depth conversion, writeback and pruning. Fixture construction, the small warmup and output serialization are outside the timer. The fixture is a dense deterministic native-color image with varying alpha, negative document origin and transparent pixels. These are CPU filter application times; window presentation and undo bookkeeping are outside this measurement.

| Filter | Size | Model + alpha | Depth | Before (ms) | After (ms) | Speedup |
|---|---:|---|---|---:|---:|---:|
| Facet | 24 MP | RGB | u8 | 684.4 | 311.5 | 2.20x |
| Halftone r8 | 24 MP | RGB | u8 | 1137.3 | 603.8 | 1.88x |
| Facet | 36 MP | RGB | u8 | 1027.5 | 454.5 | 2.26x |
| Halftone r8 | 36 MP | RGB | u8 | 1707.8 | 898.0 | 1.90x |
| Facet | 24 MP | RGB | u16 | 756.9 | 370.1 | 2.04x |
| Halftone r8 | 24 MP | RGB | u16 | 1198.9 | 664.7 | 1.80x |
| Facet | 24 MP | RGB | f32 | 806.9 | 435.7 | 1.85x |
| Halftone r8 | 24 MP | RGB | f32 | 1272.0 | 725.6 | 1.75x |
| Halftone r2 | 24 MP | RGB | u8 | 1417.8 | 716.4 | 1.98x |
| Halftone r64 | 24 MP | RGB | u8 | 1138.9 | 587.5 | 1.94x |
| Facet | 24 MP | CMYK | u8 | 7888.4 | 574.2 | 13.74x |
| Halftone r8 | 24 MP | LAB | u8 | 3588.7 | 718.3 | 5.00x |

Every pair was compared with `cmp` over the complete output, serialized as little-endian f32 sample bits in tile order. All 36 comparisons matched exactly, including the 16-bit and float cases. [Raw three-run timings and output SHA-256 hashes](pixelate-performance.json) are committed alongside this table.

## Reproduce

Build the same example against the base and the changed tree. Keep their Cargo target directories separate:

```sh
git worktree add --detach ../photocraft-pixelate-base 7722172585a01cbdb93c06f0f5ff2634fcb17999
cp crates/algo/examples/bench_pixelate.rs ../photocraft-pixelate-base/crates/algo/examples/
(cd ../photocraft-pixelate-base && CARGO_TARGET_DIR=target cargo build --release -p photocraft-algo --example bench_pixelate)
CARGO_TARGET_DIR=target cargo build --release -p photocraft-algo --example bench_pixelate

RAYON_NUM_THREADS=16 ../photocraft-pixelate-base/target/release/examples/bench_pixelate facet 6000 4000 u8 1 before.bin rgb
RAYON_NUM_THREADS=16 target/release/examples/bench_pixelate facet 6000 4000 u8 1 after.bin rgb
cmp before.bin after.bin
sha256sum before.bin after.bin
```

Repeat each pair three times, reversing the order for the middle pair. Use `7200 5000` for 36 MP; `u16` or `f32` for the other depths; `halftone`, `halftone-small` or `halftone-large` for the screen cases; and `cmyk` or `lab` for the other color models. The example prints the timed operation separately from file writing.

## Correctness and repository checks

The independent scalar oracles retain the original kernels. Tests compare every sample bit across all eight color models, alpha on/off, constant colors and edges, random samples, HDR/negative values, non-finite samples, thin regions, partial halos, extreme screen offsets, extended channel counts and scratch rejection. Tiled surface tests add 8/16/32-bit storage, soft selections, tile sizes 1/7/17/256 and both transparent and repeated edges.

Passed: 401 algorithm tests; clippy with warnings denied; formatting; dependency layering; the full `cargo xtask wasm` set; `cargo xtask perf --quick` on llvmpipe; and scorecard generation. No command, UI or format contracts changed.
