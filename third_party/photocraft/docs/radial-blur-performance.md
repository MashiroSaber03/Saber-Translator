# Radial Blur sampler optimization

The tiled filter prepares Spin rotations or Zoom scales once per operation and
reuses them across output tiles. Bilinear taps load all channels together, with
specialized loops for one through five channels and a generic fallback. Rayon
continues to parallelize output tiles on native builds.

Draft, Good and Best keep upstream's 64, 256 and 4096 interval ceilings, sample
positions, transparent edges and alpha arithmetic. The positive-alpha
unpremultiply/premultiply sequence is retained to preserve floating-point rounding.
Tests compare against the unchanged upstream sampler at large radii across
Gray, RGB, CMYK and Lab, with and without alpha, at U8, U16 and F32 depths.
Selection blending and unusual float alpha are covered as well.

Only interval counts reachable within the output area are prepared; small Best
images avoid allocating/preparing the full roughly 64 MiB transform table.
Preparation and output allocations use checked sizes and fallible reservation.
If the optional optimization cannot prepare its tables, the existing sampler
runs instead. Cancellation is checked during preparation and within output tiles.

This revision adds no source-size rejection: it retains upstream's shared
normalized source read and therefore still requires memory proportional to source
size. It does not claim bounded source memory or unlimited physical capacity.
The polar approximation and background preview scheduler are separate work.

## Local comparison (2026-10-09)

| Method | Upstream | Optimized | Speedup | Output |
|---|---:|---:|---:|---|
| Spin | 66.10 s | 16.43 s | 4.02× | Exact match |
| Zoom | 19.19 s | 5.35 s | 3.59× | Exact match |


Baseline: current upstream sampler at main `578bc905`, through `kernel`, using
the same shared source read, 256px output tiles, 256 MiB result batches, surface
writeback and pruning as the tiled filter. Optimized: `apply_in`.
The benchmark's baseline wrapper covers full-canvas, unselected filtering only;
selection behavior is checked separately in tests.

Machine: Intel Core i7-9750H (6 cores / 12 logical workers), Linux, Rust 1.98.1,
release profile with opt-level 3, thin LTO and one codegen unit. Source: deterministic
textured RGBA8 with transparent pixels, 6000×4000 (24 MP). Good quality, amount 1,
center (0.5, 0.5), 12 Rayon workers. One paired run per method, upstream first,
after a small untimed warmup. Entire filtering calls are timed; fixture generation
and exact comparison of every output tile are outside the timer.

These are single local measurements, not a cross-machine baseline or an
amount-100 performance claim. The lower amount keeps this review benchmark
bounded on the local laptop. Quality ceilings are verified separately at large
radii. The example defaults to three alternating-order runs and amount 100 for
longer measurements. Historical handoff polar/default timings do not describe
this output-preserving implementation.

```sh
cargo run --release -j 12 -p photocraft-algo --example bench_radial -- 6000 4000 1 1
```
