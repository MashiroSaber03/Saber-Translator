# Proximity Match search and memory

Spot Healing's Proximity Match chooses a source displacement by comparing the ring around
the healing hole. Two changes reduce that search's work:

- Source-validity counts use a summed-area table over the hole's nonzero bounding box.
  Every excluded cell outside that box is zero, so translating and clipping queries gives
  the same counts as a full-image table. This also covers separated holes and canvas edges.
- Each candidate's SSD is compared with the best score after every 32 ring samples. All
  SSD terms are nonnegative, so the partial sum divided by the final sample count and
  multiplied by the same distance penalty is a lower bound on the final score. A candidate
  that reaches the best score cannot win the existing strict comparison.

The sample order, accumulation order, distance penalty, candidate visit order, tie handling,
search radius and source-pixel copying stay the same. Nonfinite best scores disable pruning.
The cropped table checks allocation, arithmetic and source slices and returns None on failure.
The other summed-area tables used by content-aware completion and texture synthesis are unchanged.

## Measurements

Recorded 2026-10-10 on an AWS c7i.4xlarge, Intel Xeon Platinum 8488C, 16 vCPUs, 32 GiB RAM,
Rust 1.99.0, normal workspace release profile. This kernel is serial. The input is a synthetic
periodic RGB float texture and a circular hole; values are generated before timing. Each path
gets a warmup and five measured runs, alternating old/new order. Every run checks identical
source displacement against the old implementation.

The four small regions follow the existing retouch benchmark's stroke-domain sizing:
`width = height = diameter + 2 * (diameter + 16)`, search radius `diameter + 16`.
Ring width is `clamp(diameter / 8, 3, 16)`. Large-image cases use search radius 512.
These are kernel calls, excluding brush rasterization, surface reads, Poisson blending,
document history and compositing.

| Input region | Hole diameter | Before p50 | After p50 | Speedup | Old table | New table |
|---|---:|---:|---:|---:|---:|---:|
| 182 x 182 | 50 px | 2.924 ms | 0.765 ms | 3.82x | 133,956 B | 10,000 B |
| 332 x 332 | 100 px | 0.828 ms | 0.409 ms | 2.02x | 443,556 B | 40,000 B |
| 632 x 632 | 200 px | 1.959 ms | 1.272 ms | 1.54x | 1,602,756 B | 160,000 B |
| 932 x 932 | 300 px | 4.816 ms | 1.372 ms | 3.51x | 3,481,956 B | 360,000 B |
| 6000 x 4000 | 32 px | 67.862 ms | 14.672 ms | 4.63x | 96,040,004 B | 4,096 B |
| 6000 x 4000 | 256 px | 96.509 ms | 15.469 ms | 6.24x | 96,040,004 B | 262,144 B |
| 7200 x 5000 | 32 px | 99.693 ms | 21.658 ms | 4.60x | 144,048,804 B | 4,096 B |
| 7200 x 5000 | 256 px | 132.300 ms | 22.397 ms | 5.91x | 144,048,804 B | 262,144 B |

Table storage alone falls by roughly 90–93 percent in these stroke regions. On a full
24–36 MP input it falls from 96–144 MB to 4 KiB for the 32 px hole, or 256 KiB for the
256 px hole. Those are table sizes, not total process memory: the input image, hole mask
and sampled ring still exist. The initial hole-bounds scan remains linear in image size.

The engine already crops a normal healing stroke to a local region. The large-image cases
show the public kernel's scaling, not an additional 4.6–6.2x UI speedup. Source textures,
hole shapes, candidate validity and the quality of the current best match affect pruning;
these measurements do not establish a worst-case speedup or Photoshop image-quality parity.
A hole whose bounds cover the image still needs a full-size table.

## Correctness and reproduction

Tests compare cropped counts with full-table counts for rectangles wholly inside, outside,
crossing or clipped against the hole's bounds. They include separated holes, canvas edges,
very large query coordinates and checked table allocation/bounds failures.

Search tests compare exact source displacements with the previous implementation over
1, 3, 4 and 5 channels; 8-bit, 16-bit and float working values; several ring/search radii;
empty, one-pixel and separated holes; uniform ties and nonfinite samples. A completely
unknown image still has no source candidate.

```sh
cargo test --release -p photocraft-algo proximity_integral_release_comparison -- --ignored --nocapture
```

The old search is retained only in the test module. The selected source displacement is
unchanged, so downstream source-patch copying and gradient-domain blending get the same input.
