# Exact octagonal mask shifts

Select and Mask applies Shift Edge as flat grayscale dilation or erosion. Previously,
`matting::morph` ran one full 3x3 pass for each radius step, alternating a square and a
cross. A 64 px shift needed 64 image passes. The filter supports fractional coverage.

## Decomposition

For radius `r`, let `a = ceil(r / 2)` and `b = floor(r / 2)`. The original footprint is
exactly a square of radius `a` followed by a Manhattan ball of radius `b`:

```text
max(abs(dx) - a, 0) + max(abs(dy) - a, 0) <= b
```

Equivalently, `max(abs(dx), abs(dy)) <= r` and `abs(dx) + abs(dy) <= r + a`.
Reordering the unit squares and crosses preserves the footprint on a rectangular
canvas: a path to any included pixel can stay between its endpoints, inside the canvas.
No outside pixel participates in the old clipped passes or in the new result.

The square uses the existing van Herk / Gil-Werman prefix/suffix window extrema, with
constant comparisons per sample. Native dense buffers of at least 262,144 pixels run
the horizontal windows in parallel, transpose in 32x32 cache blocks, filter contiguous columns, then transpose
back. The two full-size float work planes are reused. Small inputs and wasm keep the
serial square pass. Each axis caps the window radius at that axis's extent.

The Manhattan ball is built from the radius's binary digits. A radius-`2d` ball is the
union of five radius-`d` balls, centered at the output and at its four axial offsets of
`d` pixels. For any point in the larger ball, choose its larger absolute coordinate.
Moving the center `d` pixels along that axis leaves Manhattan distance at most `d`.
Clamping that center to the canvas retains this property and adds no out-of-footprint
pixels. An odd radius adds one unit cross after doubling.

For masks without NaNs or negative zero, this changes the work from **O(width * height * radius)** to
**O(width * height * log(radius + 1))**. Large native cross passes also run in parallel.
Small radii 0-2 retain the original path. NaNs and negative zero retain the original
comparison order, including NaN payloads and signed-zero behavior. Other values,
including infinities and HDR mask values, use exact max/min comparisons with no
quantization or floating-point sums.

## Measurements

Recorded 2026-10-10 on an AWS c7i.4xlarge, Intel Xeon Platinum 8488C, 16 vCPUs,
32 GiB RAM, Rust 1.99.0, standard workspace release profile. Native Rayon uses 16
workers. The 384x384 case is serial in both paths; it represents the size of a 256 px
core with a 64 px halo. Large masks use 256x128 output cores with a radius halo,
scheduled across the native workers. Constant halo windows fill their cores directly.

Input is a circular grayscale mask with a 7 px fractional edge. Input generation is
outside timing. Each case warms both paths, then measures three runs each in alternating
old/new order. The table gives the median. Every run compares every output pixel exactly
against the original repeated-pass implementation.

| Mask | Radius | Operation | Before p50 | After p50 | Speedup |
|---|---:|---|---:|---:|---:|
| 384 x 384 | 4 px | Dilate | 5.559 ms | 3.301 ms | 1.68x |
| 384 x 384 | 4 px | Erode | 4.765 ms | 2.941 ms | 1.62x |
| 384 x 384 | 16 px | Dilate | 21.254 ms | 4.666 ms | 4.55x |
| 384 x 384 | 16 px | Erode | 18.393 ms | 4.080 ms | 4.51x |
| 384 x 384 | 32 px | Dilate | 42.498 ms | 5.469 ms | 7.77x |
| 384 x 384 | 32 px | Erode | 37.368 ms | 4.874 ms | 7.67x |
| 384 x 384 | 64 px | Dilate | 84.287 ms | 6.601 ms | 12.77x |
| 384 x 384 | 64 px | Erode | 72.744 ms | 5.603 ms | 12.98x |
| 6000 x 4000 | 4 px | Dilate | 986.909 ms | 41.678 ms | 23.68x |
| 6000 x 4000 | 4 px | Erode | 847.584 ms | 40.829 ms | 20.76x |
| 6000 x 4000 | 16 px | Dilate | 3630.533 ms | 50.315 ms | 72.16x |
| 6000 x 4000 | 16 px | Erode | 3069.906 ms | 47.808 ms | 64.21x |
| 6000 x 4000 | 32 px | Dilate | 7153.449 ms | 60.465 ms | 118.31x |
| 6000 x 4000 | 32 px | Erode | 6024.692 ms | 57.223 ms | 105.28x |
| 6000 x 4000 | 64 px | Dilate | 14414.481 ms | 99.749 ms | 144.51x |
| 6000 x 4000 | 64 px | Erode | 11942.596 ms | 83.131 ms | 143.66x |
| 7200 x 5000 | 4 px | Dilate | 1483.109 ms | 59.551 ms | 24.91x |
| 7200 x 5000 | 4 px | Erode | 1263.256 ms | 57.847 ms | 21.84x |
| 7200 x 5000 | 16 px | Dilate | 5463.782 ms | 67.911 ms | 80.46x |
| 7200 x 5000 | 16 px | Erode | 4593.086 ms | 65.145 ms | 70.51x |
| 7200 x 5000 | 32 px | Dilate | 10766.328 ms | 81.414 ms | 132.24x |
| 7200 x 5000 | 32 px | Erode | 9048.115 ms | 78.240 ms | 115.65x |
| 7200 x 5000 | 64 px | Dilate | 21348.501 ms | 124.007 ms | 172.16x |
| 7200 x 5000 | 64 px | Erode | 17968.390 ms | 106.289 ms | 169.05x |

## Full CPU refinement

The second comparison runs the actual tiled `refine_mask` pipeline, swapping only the
morphology callback between the previous loop and the new implementation. It includes
guide sampling, guided filtering, boundary distances, quantization, tile assembly and
output trimming. Both paths use the same synthetic RGB guide and a trimmed circular
selection, radius 64 and Shift Edge +100%. Each result's bounding box and every mask
byte must match. Warmup and three alternating paired runs use the same method as above.
Document sizes are 24 and 36 MP; the selection is cropped by the normal workflow.

| Document | Before p50 | After p50 | Speedup |
|---|---:|---:|---:|
| 6000 x 4000 | 3019.176 ms | 761.687 ms | 3.96x |
| 7200 x 5000 | 3705.934 ms | 920.891 ms | 4.02x |

A third kernel comparison uses a dense fractional gradient, `0.6*x/w + 0.4*y/h`,
so no tile can take the constant-window shortcut:

| Document | Before p50 | After p50 | Speedup |
|---|---:|---:|---:|
| 6000 x 4000 | 14207.853 ms | 394.622 ms | 36.00x |

Kernel tables include filter allocations. The CPU refinement table includes the rest
of the refinement function. Document history and UI rendering are outside that timing.

## Bounded workspace

For image-shaped mask buffers without NaNs or negative zero, at least 1,048,576 pixels, width at least 256,
height at least 128 and radii 3-64, the returned f32 output is the only full-image allocation. Each active worker uses at most a 384x256 halo window, with three f32
planes and less than 6 KiB of line scratch. At 16 workers the derived allocation bound,
including output, is about **115 MB at 24 MP** or **163 MB at 36 MP**, versus the old
**192 MB / 288 MB** two-plane workspace. This is about **40% / 43% less**. These are
allocation bounds, not measured process RSS; the source and caller-retained copies are
separate. For N pixels and P workers, the bound is `4*N + P*(12*384*256 + 6144)`
bytes. Constant windows need no filter scratch. Smaller calls and larger radii use
the dense decomposition, with two f32 work planes plus line scratch.

## Correctness and reproduction

Coverage includes all binary masks on small rectangular canvases up to 4x3; every radius
0-65 on fractional 8-bit, 16-bit and HDR float values; impulse footprints at corners,
edges and interior positions; thin images; partial transpose blocks; worker-count
identity; infinities, NaN payloads and signed zero. Shifted tiled masks are compared
with whole-buffer refinement across canvas and tile edges. Checked dimensions reject
short input and multiplication overflow. Radii larger than the canvas saturate to
reachable extents on the decomposition path, including very thin canvases.

```sh
cargo test --release -p photocraft-algo octagonal_morphology_release_comparison -- --ignored --nocapture --test-threads=1
cargo test --release -p photocraft-algo tiled_refinement_release_comparison -- --ignored --nocapture --test-threads=1
cargo test --release -p photocraft-algo dense_matte_release_comparison -- --ignored --nocapture --test-threads=1
```

The benchmark's old path is also the production fallback for exceptional values and
small radii. It uses the previous loop without a changed footprint or tie order.
