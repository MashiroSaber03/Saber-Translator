# Iris Blur computation (2026-10-10)

Base: `7722172585a01cbdb93c06f0f5ff2634fcb17999` (freshly fetched
`origin/main` and `upstream/main`). Branch: `codex/iris-blur-invariants-20261010`.

## Upstream investigation

Open, closed and merged issues/PRs were searched for Iris, Blur Gallery, blur,
sampling, portable and SIMD. No overlapping Iris Blur change was found.
Gallery history includes [PR #1010](https://github.com/storytold/photocraft/pull/1010),
which normalizes nonfinite pin angles and maps nonfinite distance to infinity
(part of #916). Its behaviour is retained in prepared Iris geometry and tested.
[PR #1524](https://github.com/storytold/photocraft/pull/1524), merged as
`d651be1393b709d638dd9d0e4824a420f795d3ae`, is the merged SIMD-related work:
the [review](https://github.com/storytold/photocraft/pull/1524#issuecomment-6072959394)
requested removal of the AVX2/pulp portion because it did not improve complete
filter timings. The author removed that portion before merge. Main has portable
contiguous Mosaic row reads, with exact-output tests, rather than a SIMD subsystem.

[PR #1602](https://github.com/storytold/photocraft/pull/1602) carries fixed-channel
Gaussian/resize accumulators from #1511, allowing compiler vectorization without
unsafe code or a new dependency. Those box accumulators use f32; Gallery uses f64
running sums for tile independence, so they are not an exact drop-in replacement.
No nightly `std::simd`, architecture-specific instructions, new dependency or
`target-cpu=native` is introduced here. Transitive codec SIMD dependencies do not
provide a Gallery kernel.

[PR #1899](https://github.com/storytold/photocraft/pull/1899) already runs filter
previews on a cancellable worker, with a 150 ms settle delay for stale requests.
[PR #2207](https://github.com/storytold/photocraft/pull/2207) addresses preview scale.
This change leaves the worker, proxy size, engine command and UI interactions intact.

## Algorithm and numerical contract

Iris Blur interpolates seven Gaussian levels (zero plus six geometrically spaced
sigmas). This is not a ray/disk sampling kernel. Every level, box radius, weight,
premultiplication, unpremultiplication and pin combination remains the same.

- Prepare each pin's centre, rotation, normalized radii, exponent, feather and
  blur once per tile. Allocation failure uses on-demand preparation instead.
- A point outside a pin's normalized bounding square is beyond its superellipse.
  Its falloff is saturated; skip the square root/powers there. Inside the square,
  use the original expression, divisions and floating-point order.
- Copy interleaved source samples together and premultiply in place, following
  the portable row-read principle in #1524.
- For each nonzero level, retain the source's top/left and trim the right/bottom
  to the output plus that level's actual three-box reach. Removing leading rows
  or columns would change running-sum cancellation, especially for HDR. Keeping
  the start preserves exact results; the trailing clamp cannot reach output.
  Tilt-Shift and Field retain their full windows.

There is no additional image-sized cache. The premultiplied window, sigma map,
output accumulator and reused level buffer already existed. The level buffer
reserves the original full-window capacity once when needed, preventing row
appends from doubling capacity; only its populated length and blur scratch shrink.
Prepared geometry uses 36 bytes per pin per active tile. Source reads, tile sizes,
Rayon dispatch, selection mixing and cancellation services remain unchanged.

`tests_iris_reference.rs` freezes the base kernel. Differential tests require
exact finite f32 bits (including signed zero), with matching NaN classifications.
They exercise 1–6 channels, RGB/Gray/CMYK/Lab, alpha/no alpha, U8/U16/F32, negative
and HDR samples, extreme HDR cancellation, rotated/translated/anisotropic pins,
roundness and blur thresholds, feather 0.998, image edges, cropped output, full
focus/full blur, empty/multiple/4096 pins, degenerate images, nonfinite/extreme pin
fields, selections and multiple tile sizes. Cancellation before work and after
progress preserves the source. Invalid optimized source lengths/zero channels
return an empty internal result. Public API/command behaviour is unchanged.

## Measurement method

Windows x86_64, AMD Ryzen 7 9700X (8 cores / 16 logical), 64 GiB RAM, stable Rust
1.99.0 / LLVM 23.1.1. Normal workspace release profile (opt-level 3, thin LTO, one
codegen unit), eight Rayon workers. No concurrent Cargo builds or other task
benchmarks were scheduled during the measurements; ordinary system activity remains.

`bench_iris` constructs deterministic textured RGB layers with varied alpha and
HDR F32 values. Fixture construction and output destruction are excluded; source
reads, pin preparation, filtering, selection-free surface writeback and pruning
are included. Each case has one untimed warmup, then repeated complete calls.
The unchanged base executable is retained locally for repeat comparisons. Use
the same harness on the base revision to reproduce a comparison.

```powershell
$env:CARGO_TARGET_DIR = 'target/agent-iris-blur'
$env:RAYON_NUM_THREADS = '8'
cargo run --release -p photocraft-algo --example bench_iris -- 6000 4000 3
cargo run --release -p photocraft-algo --example bench_iris -- 6000 6000 3
cargo run --release -p photocraft-algo --example bench_iris -- 1500 1000 5 4
cargo run --release -p photocraft-algo --example bench_iris -- 2000 1334 5
cargo run --release -p photocraft-algo --example bench_iris -- 128 96 5
cargo test --release -p photocraft-algo iris_profile_baseline_stages -- --ignored --nocapture
```

The 2000×1334 run is a preview-sized stress workload at fixed blur parameters.
PhotoCraft actually reduces 24 MP by factor four to 1500×1000 (its factor is the
ceiling of sqrt(pixels/2,500,000)). The separate 1500×1000 run models that existing
filtered-proxy workload directly: top-level document blur 15/80/300 becomes
3.75/20/75; the nested multiple-pin JSON amounts retain their existing values.
Neither run measures proxy construction, engine history, preview-worker queueing,
compositing, GPU upload, frame presentation or native slider response. Baseline and
candidate always use identical resolution, blur values and sample counts.

Before-change stage profiling of a 512×512 output tile, one rotated rounded pin:
geometry 4.3–5.8 ms; premultiplication 1.5/3.5/11.7 ms at blur 15/80/300;
each active Gaussian level about 3–9/18–19/87–100 ms. Whole kernels were
30.8/52.1/208.8 ms (unused levels are skipped). These diagnostic single-tile
figures isolate the computation; end-to-end results below use repeated runs.

## Repeated release results

Values are median milliseconds (minimum–maximum), not frame times. Run order:
base 24 MP / stress / small, candidate 24 MP / stress / small, base 24 MP again;
then base 36 MP, candidate 36 MP, base 36 MP again, base default proxy and candidate
default proxy. Each full-size base column pools six samples from two three-repeat
rounds; each candidate has three. Proxy, stress and small-image columns each have
five samples.
Both base rounds have slower medians than the candidate in every 24 MP case.
Variation is visible, particularly in the larger proxy runs; these are local
measurements, not cross-machine performance guarantees.

Small = blur 15; medium = blur 80, rotation 27°; large = blur 300.
Multiple = two pins at blur 80/40, one with roundness 65 and one rotated 73°.
Focused = radii 5×5, blur 80; blurred = pin centre x = −5, blur 80.

### 6000×4000 (24 MP)

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 1395.7 (1191.4–1550.9) | 1092.7 (1070.0–1113.7) | 1.28× |
| U8 small | 626.8 (589.8–655.9) | 426.9 (425.8–437.3) | 1.47× |
| U8 large | 4760.6 (4369.7–5154.0) | 3148.4 (3112.9–3175.3) | 1.51× |
| U8 multiple | 1789.9 (1695.9–1885.7) | 1016.2 (1015.6–1018.6) | 1.76× |
| U8 focused | 489.8 (472.8–527.3) | 405.3 (400.6–423.1) | 1.21× |
| U8 blurred | 1161.4 (1102.5–1195.4) | 981.3 (924.7–1007.1) | 1.18× |
| U16 medium | 1522.5 (1410.5–1577.9) | 1115.9 (1108.9–1119.3) | 1.36× |
| F32 medium | 1570.8 (1453.0–1656.7) | 1131.1 (1117.1–1134.3) | 1.39× |

### 6000×6000 (36 MP)

| Depth / case | Base ms (range) | Candidate ms (range) | Median ratio |
|---|---:|---:|---:|
| U8 medium | 1976.4 (1842.1–2360.1) | 1923.9 (1745.4–1993.3) | 1.03× |
| U8 small | 854.1 (773.1–947.7) | 806.0 (784.6–818.5) | 1.06× |
| U8 large | 6162.4 (5767.0–6578.0) | 5423.8 (5245.8–5995.2) | 1.14× |
| U8 multiple | 2595.8 (2544.7–2731.7) | 1822.8 (1781.5–1826.7) | 1.42× |
| U8 focused | 708.7 (684.9–740.7) | 628.6 (618.0–630.7) | 1.13× |
| U8 blurred | 1705.5 (1585.5–1821.2) | 1609.4 (1561.4–1613.6) | 1.06× |
| U16 medium | 2217.2 (2034.6–2438.8) | 2065.5 (1961.2–2127.9) | 1.07× |
| F32 medium | 2346.8 (2318.4–2411.9) | 1702.1 (1647.2–1795.3) | 1.38× |

The second base round was faster than the candidate for U8 small and medium,
despite better pooled medians: neither demonstrates a consistent improvement at
36 MP. U16 medium and fully blurred ranges also overlap; their gains are
inconclusive. Large blur improves both round medians but ranges overlap.
Multiple pins, full focus and F32 medium have separated observed ranges.

### 1500×1000 default filtered proxy for a 24 MP document

Single-pin blur values below are the document values divided by four; multiple
pins retain the supplied JSON blur values, matching the existing preview path.

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 51.8 (47.9–54.9) | 42.6 (39.3–45.5) | 1.22× |
| U8 small | 28.7 (27.4–31.0) | 24.9 (23.5–25.9) | 1.15× |
| U8 large | 144.0 (132.5–153.0) | 110.6 (104.0–111.9) | 1.30× |
| U8 multiple | 178.6 (174.8–205.9) | 111.6 (109.3–114.7) | 1.60× |
| U8 focused | 18.0 (17.1–19.0) | 15.0 (14.8–15.9) | 1.20× |
| U8 blurred | 33.1 (31.9–36.9) | 28.2 (27.8–29.9) | 1.17× |
| U16 medium | 52.1 (48.1–55.3) | 42.8 (40.7–49.4) | 1.22× |
| F32 medium | 60.4 (56.6–63.5) | 45.0 (43.9–50.0) | 1.34× |

Every default-proxy median improves. U16 medium ranges overlap, so that case's
gain is less certain than the other seven separated-range cases.

### 2000×1334 preview-sized stress workload

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 306.0 (280.6–465.1) | 163.8 (150.1–203.0) | 1.87× |
| U8 small | 82.5 (75.0–87.4) | 59.0 (58.2–63.4) | 1.40× |
| U8 large | 2051.0 (1724.3–2087.2) | 1324.2 (1106.4–1533.7) | 1.55× |
| U8 multiple | 303.6 (289.2–330.7) | 209.1 (200.5–231.5) | 1.45× |
| U8 focused | 61.5 (60.6–73.2) | 55.4 (51.6–78.7) | 1.11× |
| U8 blurred | 151.0 (146.5–155.1) | 143.3 (136.3–145.8) | 1.05× |
| U16 medium | 269.7 (262.7–289.8) | 216.1 (207.1–249.0) | 1.25× |
| F32 medium | 252.5 (237.9–295.5) | 228.0 (207.6–254.8) | 1.11× |

The already fully blurred stress case has little distance/level work to eliminate;
its improvement is modest. Focused and F32 stress ranges overlap, so their small
median gains are inconclusive. These cases do not justify additional machinery.

### 128×96 overhead

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 30.8 (30.3–31.3) | 17.7 (17.7–17.8) | 1.74× |
| U8 small | 4.1 (4.0–4.1) | 3.2 (3.1–3.3) | 1.26× |
| U8 large | 280.4 (276.1–294.6) | 149.1 (145.2–151.2) | 1.88× |
| U8 multiple | 26.0 (25.7–26.6) | 13.0 (13.0–13.1) | 1.99× |
| U8 focused | 1.7 (1.6–1.7) | 1.2 (1.2–1.3) | 1.36× |
| U8 blurred | 6.9 (6.9–7.0) | 6.3 (6.3–6.4) | 1.09× |
| U16 medium | 30.4 (30.2–31.0) | 18.3 (18.3–18.3) | 1.67× |
| F32 medium | 30.1 (30.0–30.5) | 18.3 (18.2–18.7) | 1.64× |

Windows process peak working set for one entire 24 MP suite (including fixture
creation and all depths/cases): base repeat 2,068,135,936 bytes, candidate
1,990,852,608 bytes, about 3.7% lower. This is an allocator-dependent process
high-water measurement sampled via `PeakWorkingSet64`, not per-filter live memory
or a new allocation bound. Existing full-window buffer capacities remain the same.

Maximum observed finite pixel error in the differential tests is **zero**: exact
bits, rather than an average-error bound. Matching every finite pixel covers
focus-transition structure, sampling aliasing, opacity and tile boundaries without
permitting a localized artifact to hide in a mean error. NaN classifications match.

## Validation and limits

Algorithm tests: 399 passed, 15 existing/diagnostic tests ignored. The five Iris
differential/cancellation tests also pass in release; their profiling test is
opt-in. All-target
algorithm Clippy passes with warnings denied. Layering passes (29 crates). Wasm
checks pass for all required crates and HEIF; unrelated existing dead-code warnings
in CMS/engine remain. Scorecard regeneration produces identical content; no budget,
corpus floor, menu parity or checklist status changed. `test-corpus --changed`
correctly skips this algorithm-only change. Commands and UI code are unchanged,
so command panic-hunt and UI screenshot checks are not triggered.

`cargo xtask perf --quick` completed successfully, running its configured quick
native CPU/GPU examples. Its regression comparison was skipped because the
checked-in baseline is for an Apple M4 Pro, while this machine is Windows/Ryzen
with a Radeon RX 9060 XT. Quick mode does not enforce the full-size budgets.
Results are retained locally in `target/agent-iris-blur/perf/results.json`;
the Iris base/candidate measurements above provide this change's comparison.
Final complete wasm and algorithm formatting checks also pass.

Final diagnostic single-tile reference/candidate calls (same profile workload,
one pair each, not a statistical benchmark): blur 15, 31.6 → 25.3 ms; blur 80,
52.9 → 43.1 ms; blur 300, 224.1 → 174.7 ms. The repeated complete-filter results
remain the performance evidence used to retain the change.

Native Iris GUI frame/slider latency, ARM/macOS/Linux performance and wasm runtime
speed were not measured. Compilation confirms wasm compatibility only. Photoshop
equivalence beyond the base implementation's semantics is not asserted. The private
shared craftrules checkout was unavailable at the supplied locations; repository
AGENTS.md and architecture/development/contributing/control/performance rules were
followed. Existing source allocation and per-tile cancellation limits are retained.
