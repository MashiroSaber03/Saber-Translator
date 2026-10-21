# Quick Selection stroke performance

Measured 2026-10-10 on an AWS c7i.4xlarge, Intel Xeon Platinum 8488C,
16 vCPUs, 32 GiB RAM, Ubuntu 24.04, Rust 1.99.0. Release builds and the
default native worker count. Builds and other tests were idle during timing.
Algorithm commit: `78dca424510bbdecb01228a4a98a971fd070b658`.

## What changed

The fitted Quick Selection energy, brush footprint, costs, weights, working
resolution, boundary bands and window expansion rules stay the same.

Large finite graphs with both terminal directions use FIFO push/relabel,
implemented from [Goldberg and Tarjan (JACM 1988)](https://doi.org/10.1145/48014.61051).
Local excess pushes replace repeated augmenting paths. Periodic global
relabelling computes distances to the sink and, for the remaining vertices,
back to the source. Returning unused source flow is necessary to retain a
feasible residual graph and support another solve after adding capacities.
The final source-reachability pass returns the smallest source-side cut.
Graphs below 4096 nodes, one-terminal graphs, and graphs with nonfinite or
negative edge capacities keep the existing Boykov-Kolmogorov implementation.
Both solvers use the existing f32 residual capacities and saturation epsilon.
The new solver stores f32 reverse source capacities and f64 excess, 12 bytes
per free graph node, and reuses the graph's height, current-arc and active arrays.

Graph construction skips weights between two fixed pixels. With fewer than
one free pixel in eight, a five-way merge enumerates just the canonical edge
sources incident to free pixels, in their original raster order. This makes
edge enumeration and weight evaluation O(free pixels), while the existing
indexing and terminal setup remain linear in the working image. Preserving
edge order also preserves f32 accumulation into fixed-neighbour terminals.

## Raster surface strokes

The timed call is the full `quick_select` CPU pipeline through `SurfaceSampler`,
as used by `select.quick`: raster reading and depth conversion, RGB sampling,
window expansion, footprint and distance costs, coarse-to-fine cuts, mask
upsampling and region trimming. Document construction is outside the timer.
The stroke paints three rows across a 2400 x 1200 textured object with a
180 px brush. The fixture is deterministic synthetic RGB with a contrasting
background and a small texture component.

One warm pair, then five alternating before/after pairs per depth and document.
The table reports medians. Every run asserts identical bounds and every mask byte.

| Document | Depth | Before (ms) | After (ms) | Speedup |
| --- | --- | --- | --- | --- |
| 6000x4000 | 8-bit | 747.60 | 201.96 | 3.70x |
| 6000x4000 | 16-bit | 747.36 | 208.29 | 3.59x |
| 6000x4000 | 32f | 738.22 | 211.35 | 3.49x |
| 7200x5000 | 8-bit | 721.61 | 207.24 | 3.48x |
| 7200x5000 | 16-bit | 719.26 | 207.96 | 3.46x |
| 7200x5000 | 32f | 732.08 | 205.96 | 3.55x |

## Other complete CPU strokes

These use `ImageSampler` to isolate the same complete stroke pipeline from
raster storage conversion. One warm pair and three alternating measured pairs.
The click uses a 30 px brush on a 220 x 220 object; the long stroke uses a
256 px brush across 2000 px on a 2400 x 1000 object; paint-object is the
three-row stroke above. All 24 MP and 36 MP fixtures retain normal working
resolution and automatic window expansion.

| Document | Stroke | Before (ms) | After (ms) | Speedup |
| --- | --- | --- | --- | --- |
| 6000x4000 | click | 113.60 | 39.90 | 2.85x |
| 6000x4000 | long-stroke | 1199.74 | 360.31 | 3.33x |
| 6000x4000 | paint-object | 736.28 | 205.92 | 3.58x |
| 7200x5000 | click | 121.53 | 33.09 | 3.67x |
| 7200x5000 | long-stroke | 1168.25 | 353.19 | 3.31x |
| 7200x5000 | paint-object | 725.02 | 201.60 | 3.60x |

## Correctness and regression checks

- 1170 integer-capacity graphs compare flow and all labels against BK, then solve again.
- 600 arbitrary f32-capacity graphs compare flows and all labels against BK.
- 240 fractional graphs enumerate every possible cut, checking optimal energy
  and the intersection of all optimal source sets independently of BK.
- A 5000-node disconnected chain checks return-to-source progress, a repeated
  solve, and adding a link before another solve.
- Exhaustive small constraint masks and edge-source masks cover borders,
  diagonals, degenerate dimensions and tied cuts.
- 36 full Quick Selection comparisons cover texture, low contrast, gradients,
  noise, thin bars, discs, uniform areas and canvas-border strokes, with direct
  and coarse-to-fine cuts.
- Every measured workflow compares full region bounds and mask bytes.

## Reproduction

```sh
cargo test --release -p photocraft-algo quick_selection_surface_release_comparison -- --ignored --nocapture --test-threads=1
cargo test --release -p photocraft-algo quick_selection_workflow_release_comparison -- --ignored --nocapture --test-threads=1
cargo test --release -p photocraft-algo solver_workload_release_comparison -- --ignored --nocapture --test-threads=1
```

The private const test path keeps the previous dense weight scan and the
unchanged BK solver, sharing the rest of the full stroke pipeline. Solver-only
fixtures additionally cover chains, one-terminal grids and dense terminal grids
at 1024, 4096, 16384 and 65536 nodes. Their numbers measure just the solve,
with graph cloning outside the timer.

## Raw samples

```text
SURFACE 6000x4000 format=PixelFormat { mode: Rgb, sample: U8, alpha: true } old_ms=[745.840809, 746.246292, 747.604875, 747.82731, 748.878614] new_ms=[201.579251, 201.723446, 201.956769, 202.38253, 202.948418] speedup=3.70
SURFACE 6000x4000 format=PixelFormat { mode: Rgb, sample: U16, alpha: true } old_ms=[745.974128, 746.216757, 747.364691, 748.216512, 748.33632] new_ms=[206.63495600000002, 206.932412, 208.287491, 209.847055, 210.480741] speedup=3.59
SURFACE 6000x4000 format=PixelFormat { mode: Rgb, sample: F32, alpha: true } old_ms=[737.480891, 737.6333950000001, 738.216963, 738.513631, 739.73055] new_ms=[209.130586, 209.490552, 211.347315, 211.520884, 213.142088] speedup=3.49
SURFACE 7200x5000 format=PixelFormat { mode: Rgb, sample: U8, alpha: true } old_ms=[720.72951, 721.2813970000001, 721.6096229999999, 722.4278710000001, 723.3628600000001] new_ms=[206.039241, 207.077233, 207.241128, 207.246913, 210.42549] speedup=3.48
SURFACE 7200x5000 format=PixelFormat { mode: Rgb, sample: U16, alpha: true } old_ms=[718.551252, 718.5636020000001, 719.25546, 719.810975, 719.868964] new_ms=[203.704922, 205.042543, 207.958082, 208.070805, 208.25623900000002] speedup=3.46
SURFACE 7200x5000 format=PixelFormat { mode: Rgb, sample: F32, alpha: true } old_ms=[730.028451, 731.879452, 732.082273, 733.348433, 734.600343] new_ms=[205.59655800000002, 205.75180600000002, 205.957811, 206.02760800000001, 207.490222] speedup=3.55
QUICK 6000x4000 case=click brush=30 old_ms=[113.313339, 113.600531, 114.41872000000001] new_ms=[39.298109, 39.903393, 39.940391] speedup=2.85 bbox=Rect { x0: 2891, y0: 1891, x1: 3111, y1: 2111 }
QUICK 6000x4000 case=long-stroke brush=256 old_ms=[1184.445199, 1199.7410909999999, 1199.926849] new_ms=[352.47967, 360.312007, 367.333399] speedup=3.33 bbox=Rect { x0: 1801, y0: 1501, x1: 4201, y1: 2501 }
QUICK 6000x4000 case=paint-object brush=180 old_ms=[732.409731, 736.284987, 736.734234] new_ms=[204.055566, 205.92469899999998, 208.78787100000002] speedup=3.58 bbox=Rect { x0: 1801, y0: 1401, x1: 4201, y1: 2601 }
QUICK 7200x5000 case=click brush=30 old_ms=[121.405737, 121.530837, 122.215665] new_ms=[32.693926, 33.087069, 33.834208] speedup=3.67 bbox=Rect { x0: 3491, y0: 2391, x1: 3711, y1: 2611 }
QUICK 7200x5000 case=long-stroke brush=256 old_ms=[1166.053819, 1168.25045, 1169.190979] new_ms=[352.533248, 353.18776299999996, 355.84682899999996] speedup=3.31 bbox=Rect { x0: 2401, y0: 2001, x1: 4801, y1: 3001 }
QUICK 7200x5000 case=paint-object brush=180 old_ms=[723.5930350000001, 725.024707, 726.1529770000001] new_ms=[200.803006, 201.595578, 202.331318] speedup=3.60 bbox=Rect { x0: 2401, y0: 1901, x1: 4801, y1: 3101 }
SOLVER width=32 n=1024 case=chain old_ms=[0.026896, 0.027003, 0.02743] new_ms=[0.02681, 0.027076999999999997, 0.033721] speedup=1.00
SOLVER width=32 n=1024 case=all-source old_ms=[0.016474, 0.01651, 0.024678] new_ms=[0.016201999999999998, 0.016773999999999997, 0.017098000000000002] speedup=0.98
SOLVER width=32 n=1024 case=dense-grid old_ms=[0.5345420000000001, 0.541424, 0.546977] new_ms=[0.519929, 0.543284, 0.566567] speedup=1.00
SOLVER width=64 n=4096 case=chain old_ms=[0.118043, 0.118185, 0.120604] new_ms=[0.106363, 0.106517, 0.114974] speedup=1.11
SOLVER width=64 n=4096 case=all-source old_ms=[0.065567, 0.06612499999999999, 0.067655] new_ms=[0.071212, 0.07142100000000001, 0.073924] speedup=0.93
SOLVER width=64 n=4096 case=dense-grid old_ms=[3.866533, 3.8864490000000003, 3.922701] new_ms=[1.256155, 1.319511, 1.3287730000000002] speedup=2.95
SOLVER width=128 n=16384 case=chain old_ms=[0.439333, 0.43940500000000005, 0.44905] new_ms=[0.38673599999999997, 0.395393, 0.475274] speedup=1.11
SOLVER width=128 n=16384 case=all-source old_ms=[0.26877999999999996, 0.28474099999999997, 0.295766] new_ms=[0.285945, 0.301348, 0.305836] speedup=0.94
SOLVER width=128 n=16384 case=dense-grid old_ms=[13.136574999999999, 13.181371, 13.289321999999999] new_ms=[1.110547, 1.204107, 1.2151900000000002] speedup=10.95
SOLVER width=256 n=65536 case=chain old_ms=[1.83223, 1.855307, 1.887005] new_ms=[1.5852430000000002, 1.624603, 1.795397] speedup=1.14
SOLVER width=256 n=65536 case=all-source old_ms=[1.075863, 1.0908529999999999, 1.178408] new_ms=[1.16852, 1.184938, 1.224871] speedup=0.92
SOLVER width=256 n=65536 case=dense-grid old_ms=[33.334272999999996, 33.502357, 33.531151] new_ms=[4.31308, 4.609208, 4.618033] speedup=7.27
```
