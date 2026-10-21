# Selection distance transforms

The distance transform used by Select > Modify > Expand, Contract and Border now runs
as two native parallel passes. It also serves matting and mask feathering callers in the
engine. The compositor has a separate implementation and is outside this change.

The first pass finds the nearest selected pixel above and below each pixel. Since its
input is binary, two sweeps replace the general parabola envelope for that axis. Vertical
distances are stored as integers in column order. The second pass copies 32-row bands
into contiguous rows and evaluates the lower envelope of squared-distance parabolas.
Both passes are linear in the number of pixels. WebAssembly uses the same scalar kernels.

The row envelope follows [Felzenszwalb and Huttenlocher, 2012](https://toc.cs.uchicago.edu/articles/v008a019/).
Its intersections use `(fq - fp) / (2 * (q - p)) + (q + p) / 2` in f64. This avoids the old
subtraction of f32 coordinate squares, which lost half-pixel boundary precision on large
canvases. Distances are rounded to f32 only at the output.

## Correctness

The previous transform can return a positive distance at a selected pixel past coordinate
4096. A 6000 x 1 fully selected mask reproduces it. The regression tests also compare sparse,
competing sites in 65537 x 2 and 2 x 65537 masks against direct Euclidean distances.

Tests enumerate every mask up to 4 x 3 against a brute-force oracle, check rectangular and
partial row bands against the previous implementation, preserve fractional Expand/Contract/
Border coverage on smaller inputs, and compare native worker counts 1, 2 and 4. Empty or
inconsistent dimensions return an empty buffer. No-seed masks retain the finite 1e10 result.
Empty and fully selected masks skip the transform passes.

Large benchmark inputs check every selected pixel has zero distance and check independent
nearest-site distances at corners and the centre. The sparse-grid cases match the old
output exactly. Ellipse and dense cases intentionally correct old distance errors, so this
is not a claim of identical output everywhere or of measured Photoshop parity.

## Measurements

Recorded 2026-10-10 on an AWS c7i.4xlarge in ap-southeast-2: 16 vCPUs, 32 GiB RAM,
Rust 1.99.0, the workspace's normal release profile, 16 Rayon workers. Synthetic masks are
an ellipse, a sparse regular grid, and a deterministic dense pattern. Input generation,
reference validation and destruction of the output happen outside the timed interval.
Each implementation gets a warmup, then three measured runs with alternating order.

| Size | Mask | Before p50 | After p50 | Speedup | Changed distances | Largest correction |
|---|---|---:|---:|---:|---:|---:|
| 24 MP | ellipse | 630.74 ms | 83.11 ms | 7.59x | 827,500 | 1 px |
| 24 MP | sparse | 621.81 ms | 71.03 ms | 8.75x | 0 | 0 px |
| 24 MP | dense | 1048.21 ms | 99.73 ms | 10.51x | 2,034,343 | 1.4142135 px |
| 36 MP | ellipse | 994.14 ms | 124.48 ms | 7.99x | 2,431,817 | 1 px |
| 36 MP | sparse | 984.52 ms | 109.94 ms | 8.96x | 0 | 0 px |
| 36 MP | dense | 1588.23 ms | 150.28 ms | 10.57x | 4,771,985 | 1.4142135 px |

These are kernel timings. Mask conversion, selection history, document compositing and
canvas upload are excluded. They do not establish an end-to-end interaction budget.

## Memory and limits

The old transform used one f32 plane. The new transform needs one u32 plane plus the f32
result, or eight bytes per pixel, excluding the caller's input mask. It also uses up to
`32 * width * 4 + width * (size_of::<usize>() + 8)` scratch bytes per active row task.
At 24 MP the two planes total 192 MB; at 36 MP they total 288 MB. Fully empty and fully
selected masks need only the result plane. Small mixed masks below 65536 pixels run serially.

The extra plane buys contiguous writes and independent parallel bands. This remains a dense
transform, so it does not solve streaming selections on very large PSB documents. Dimension
multiplication, element-size limits and input length are checked before allocation.

## Reproduce

```sh
cargo test --release -p photocraft-algo distance_transform_release_comparison -- --ignored --nocapture
EDT_WORKERS=1 cargo test --release -p photocraft-algo distance_transform_release_comparison -- --ignored --nocapture
```

The previous scalar implementation is retained only in the test module as the reference
and benchmark baseline.
