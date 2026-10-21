# Five imaging algorithms: measured CPU performance

Measured on 2026-10-10. These changes remove repeated work from Extrude Blocks,
Wind/Blast, Spin Blur, Lens Correction and prepress Trap. They preserve the existing
output and parameter behavior. Measurements compare the full CPU operation against
`7722172585a01cbdb93c06f0f5ff2634fcb17999`, not just the inner kernel.

## What changed

- **Extrude Blocks:** visit each cell once, calculate its color and height once, then
  rasterize its swept bounds. Keep the original candidate-range checks, intersection
  tests and cell tie order. The default 30 px blocks benefit too, not only dense blocks.
- **Wind/Blast:** scatter each actual source streak once. Reuse source luminance and
  hash work, retain the nearest-source tie rule, then blend once per output pixel.
- **Spin Blur:** prepare all 2..64 sample-count rotation tables once per pin. Fixed
  channel samplers keep bilinear accumulators in registers without changing tap or
  premultiplied-alpha arithmetic order.
- **Lens Correction:** share Catmull-Rom coordinates and weights across channels
  sampling the same point, using fixed channel accumulators. RGB chromatic-aberration
  offsets still use independent samples.
- **Trap:** replace wide-window ink scans with separable prefix/suffix block maxima,
  independent of radius. Transpose the intermediate output for contiguous column
  writes. Widths 1 and 2 retain the original path.

## Method

AWS m7i.4xlarge, 16 vCPUs, Intel Xeon Platinum 8488C, 61 GiB RAM, Ubuntu 24.04.5.
Rust `1.99.0 (b940084d7 2026-09-28)`, release profile, `RAYON_NUM_THREADS=16`.
Builds used independent target directories. No build or test ran during timing.
The final rebase adds upstream canvas and automation changes; the timed algorithm
source, fixture, dependencies and Cargo configuration are unchanged. The raw results
record both the measured revision and its equivalent rebased revision.

The identical committed `bench_algorithms` example runs at both revisions. It creates
native RGB or CMYK tiles at 8-bit, 16-bit or 32-bit float depth, with deterministic
65536-level busy texture, variable alpha and roughly 20% zero-alpha pixels. Bounds begin at
(-17, -29). A 128 x 128 operation warms each process before timing.

Each case has three independent before/after pairs, alternating process order:
before/after, after/before, before/after. The table reports the median of each set
of three, with speedup computed from unrounded medians. Fixture construction,
warmup, output serialization and result destruction are outside the timer.

For filters, the timer includes `apply_in`, tile reads, processing, writeback and
pruning. Lens times the full `lens::correct` call. Trap includes surface reads,
ink dilation, clone, writeback and pruning. Document flattening, undo bookkeeping
and UI rendering are outside these CPU measurements.

Every pair serializes every output sample as its normalized f32 bits in tile order.
All 72 before/after SHA-256 comparisons match exactly, including alpha. The
[raw results](five-algorithm-performance.json) contain all times, medians and hashes.

Extrude uses random-depth Blocks, seed 17, size/depth 30/30 px, or 8/128 px for
`extrude-dense`; solid-front, level-based and incomplete-cell masking are off.
Wind/Blast use seed 17; `blast-right` changes direction. `spin` uses one full-frame
elliptical pin, 30 degree blur and 17 degree orientation; `spin-default` uses the
unchanged default pin. Lens uses distortion 12, rotation 2.5 degrees and vignette 20;
`lens-ca` also enables red/cyan 12 and blue/yellow -8. Trap widths are 10, 50
(`trap-wide`) and 1 (`trap-default`) pixels.

## Full-operation results

| Case | Dimensions | Mode/depth | Before (s) | After (s) | Speedup |
|---|---|---|---:|---:|---:|
| `extrude` | 6000 x 4000 | RGB/u8 | 2.0641 | 0.3296 | 6.26x |
| `extrude-dense` | 6000 x 4000 | RGB/u8 | 160.3010 | 0.3866 | 414.69x |
| `extrude` | 7200 x 5000 | RGB/u8 | 3.1539 | 0.4954 | 6.37x |
| `extrude` | 6000 x 4000 | RGB/u16 | 2.1515 | 0.4004 | 5.37x |
| `extrude` | 6000 x 4000 | RGB/f32 | 2.2541 | 0.4774 | 4.72x |
| `blast` | 6000 x 4000 | RGB/u8 | 1.4413 | 0.4114 | 3.50x |
| `blast` | 7200 x 5000 | RGB/u8 | 2.1646 | 0.6327 | 3.42x |
| `blast` | 6000 x 4000 | CMYK/u8 | 6.2310 | 0.7092 | 8.79x |
| `blast-right` | 6000 x 4000 | RGB/u16 | 1.5043 | 0.4821 | 3.12x |
| `wind` | 6000 x 4000 | RGB/f32 | 0.7678 | 0.4866 | 1.58x |
| `wind` | 6000 x 4000 | CMYK/u8 | 2.5829 | 0.6568 | 3.93x |
| `spin` | 6000 x 4000 | RGB/u8 | 6.3307 | 3.9093 | 1.62x |
| `spin-default` | 6000 x 4000 | RGB/u8 | 1.5961 | 1.1203 | 1.42x |
| `spin` | 7200 x 5000 | RGB/u8 | 9.4388 | 5.8694 | 1.61x |
| `spin` | 6000 x 4000 | RGB/u16 | 6.3066 | 3.9617 | 1.59x |
| `spin` | 6000 x 4000 | RGB/f32 | 6.3749 | 4.0619 | 1.57x |
| `lens` | 6000 x 4000 | RGB/u8 | 0.6208 | 0.4448 | 1.40x |
| `lens` | 6000 x 4000 | CMYK/u16 | 0.8080 | 0.5649 | 1.43x |
| `lens` | 6000 x 4000 | RGB/f32 | 0.6847 | 0.5103 | 1.34x |
| `lens-ca` | 6000 x 4000 | RGB/u8 | 0.6214 | 0.6312 | 0.98x |
| `trap` | 6000 x 4000 | CMYK/u8 | 0.9710 | 0.8082 | 1.20x |
| `trap-wide` | 7200 x 5000 | CMYK/u16 | 4.6265 | 1.3331 | 3.47x |
| `trap` | 6000 x 4000 | CMYK/f32 | 1.1924 | 1.0273 | 1.16x |
| `trap-default` | 6000 x 4000 | CMYK/u8 | 0.6363 | 0.6506 | 0.98x |

## Correctness and bounded work

407 algorithm tests passed, with 14 existing ignored tests. New direct-reference
comparisons cover all eight color models, alpha on/off, all three depths, edge
modes, unusual tile sizes and soft selections where applicable. They also exercise
ties, partial cells, extreme coordinates, nonfinite samples and fallback paths.
Spin rotation tables match the original trigonometric results bit for bit.
Trap tests include clipped windows, paper preservation, extra channels and signed zero.

Extrude caps its additional winner plane at 4,194,304 pixels (16 MiB), cell visits
at 262,144 and coordinates at +/-2^22; normal tiled execution fits these bounds.
Wind caps winner scratch at 1,048,576 entries per row (8 MiB). Spin tables are
under 2 MiB for up to 120 pins. These scratch allocations are fallible and use the
original direct implementation if preparation fails or bounds are exceeded.
Extrude Pyramids and Wind Stagger keep their original implementations. Trap's new
per-line suffix buffer is fallible, with a direct fallback; it adds no full-image
plane beyond the existing dilation workspace. Lines containing negative zero use
the original scan to retain float bit patterns.

`cargo fmt --all -- --check`, algorithm clippy with all targets and denied warnings,
`cargo xtask layers`, full `cargo xtask wasm`, `cargo xtask perf --quick`, and
scorecard generation/check passed. The quick performance run used llvmpipe and
still reports existing over-budget UI scenarios. These operation measurements do
not change those budget claims or establish Photoshop pixel parity.

## Reproduce

From the PR checkout, create a base worktree and copy only the benchmark fixture:

```sh
base=7722172585a01cbdb93c06f0f5ff2634fcb17999
git worktree add --detach ../photocraft-five-base "$base"
cp crates/algo/examples/bench_algorithms.rs ../photocraft-five-base/crates/algo/examples/
CARGO_TARGET_DIR="$PWD/target/five-after" cargo build --release -p photocraft-algo --example bench_algorithms
(cd ../photocraft-five-base && CARGO_TARGET_DIR="$PWD/target/five-before" cargo build --release -p photocraft-algo --example bench_algorithms)
before="$PWD/../photocraft-five-base/target/five-before/release/examples/bench_algorithms"
after="$PWD/target/five-after/release/examples/bench_algorithms"
export RAYON_NUM_THREADS=16

# Each invocation prints its full-operation time. Keep the machine otherwise idle.
# Use the case names, dimensions, depth and mode from the table.
case=extrude-dense
w=6000
h=4000
depth=u8
mode=rgb
for pair in 1 2 3; do
    if [ "$pair" = 2 ]; then
        "$after" "$case" "$w" "$h" "$depth" 1 after.bin "$mode"
        "$before" "$case" "$w" "$h" "$depth" 1 before.bin "$mode"
    else
        "$before" "$case" "$w" "$h" "$depth" 1 before.bin "$mode"
        "$after" "$case" "$w" "$h" "$depth" 1 after.bin "$mode"
    fi
    cmp before.bin after.bin || exit 1
    sha256sum before.bin after.bin
done
```

Take the median of the three times for each binary. Output files can reach 720 MB
per side for 36 MP CMYK+alpha; reuse the two filenames between cases. Hardware,
worker count and image content affect the times, so preserve the fixture when
comparing revisions.
