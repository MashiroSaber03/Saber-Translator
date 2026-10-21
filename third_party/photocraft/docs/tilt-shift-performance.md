# Tilt-Shift computation (2026-10-10)

Starting base: `a28d786638ffbb84c5fb7d5d30ea1c3d594cd3bc`, freshly fetched
`origin/main`. Branch: `codex/tilt-shift-performance-20261010`. Worktree:
`C:\Users\Colum\.codex\worktrees\6c32\photocraft`. At investigation time,
`upstream/main` was `5963e5b7f0aefa0d94b5cf67ef3c1c9b6d3399dd`, two commits ahead:
[Recent entries](https://github.com/storytold/photocraft/commit/b773a39efa4a65888fcce4a5d8dffcf4fa45700a)
and [temporary Move](https://github.com/storytold/photocraft/commit/5963e5b7f0aefa0d94b5cf67ef3c1c9b6d3399dd).
Neither changes filtering. The original checkout, local main and other worktrees
were not edited. Cargo uses `target/agent-tilt-shift` throughout.

## Upstream evidence and choice

GitHub open/closed/merged PRs and open/closed issues were searched for blur,
Tilt-Shift, Field Blur, sampling, SIMD, portable, std::simd and rustfft. Relevant
bodies, discussions, final diffs, merge commits, Gallery history and current
source were inspected before the implementation was selected.

| Work | Status and evidence | Consequence for Tilt-Shift |
|---|---|---|
| [#1524: portable Mosaic reads](https://github.com/storytold/photocraft/pull/1524), [merge d651be13](https://github.com/storytold/photocraft/commit/d651be1393b709d638dd9d0e4824a420f795d3ae) | The SIMD-related proposal the human referred to. Its AVX2/pulp portion was experimental and removed before merge: the [review](https://github.com/storytold/photocraft/pull/1524#issuecomment-6072959394) found no complete-filter gain to justify dependencies/features/CI. Portable contiguous row reads and exact-output tests landed. | Reuse contiguous reads. Main has no general portable-SIMD Gallery backend or nightly std::simd path to reuse. |
| [#1471: adaptive rustfft Motion Blur](https://github.com/storytold/photocraft/pull/1471), [merge c7dbb37b](https://github.com/storytold/photocraft/commit/c7dbb37b3c61c18223e56c10d06efc71e48c7677) | Merged. Builds on [#902](https://github.com/storytold/photocraft/pull/902)'s merged bilinear taps/row convolution. Wide uniform streaks benefit from FFT; short/unsupported cases retain rows. Current code bounds FFT allocations and rejects unsuitable coordinates/samples from that path. | Tilt-Shift is spatially varying, not one uniform convolution. A single FFT would change the band; one FFT per existing Gaussian level would preserve spatial blending but replace linear-time box cascades with transforms, padding, complex scratch and numerical differences. No demonstrated crossover warrants that machinery here. |
| [#867: Gaussian running sums and bands](https://github.com/storytold/photocraft/pull/867), [#628: Box Blur](https://github.com/storytold/photocraft/pull/628), [#1602: fixed-channel accumulators](https://github.com/storytold/photocraft/pull/1602) carrying closed [#1511](https://github.com/storytold/photocraft/pull/1511) | Merged, with radius-independent separable passes, traversal/allocation improvements and compiler-friendly accumulators. | Gallery already uses separable boxes, with f64 running sums. Replacing them with the f32 Gaussian accumulators would change HDR cancellation/rounding. Avoid a broad kernel/traversal rewrite. |
| [#1499: Radial sampling](https://github.com/storytold/photocraft/pull/1499) | Merged after [review](https://github.com/storytold/photocraft/pull/1499#issuecomment-6071866387) removed a changed-output polar approximation and a separate preview scheduler. Exact transform preparation and interleaved taps remained. | Preserve the kernel and reuse current services. Tilt-Shift already prepares its band geometry once per tile and has no radial/disk samples to precompute. |
| [#2578: Iris Blur](https://github.com/storytold/photocraft/pull/2578), [merge fa33712d](https://github.com/storytold/photocraft/commit/fa33712dd3bd87c4502c01dbeae3d235383e1b46) | Already merged in the starting base. Prepares Iris geometry and adds exact-output contiguous premultiplication and per-level trailing-halo trimming to the shared helper; only Iris opts in. | Opt Tilt-Shift into that established helper. No new algorithm, dependency, cache or scheduler is needed. Field remains unchanged. |
| [#1899: preview worker/cancellation](https://github.com/storytold/photocraft/pull/1899), [#1825: keep preview during commit](https://github.com/storytold/photocraft/pull/1825), [#2207: preview scale](https://github.com/storytold/photocraft/pull/2207) | Merged. Latest-request worker; cancel stale work after 150 ms without parameter changes, preserving updates during dragging. Pixel parameters scale with the existing proxy. | Leave the UI and worker intact; faster tile computation also shortens the existing worker's filtering work. |
| Active [#2685](https://github.com/storytold/photocraft/pull/2685), [#2580](https://github.com/storytold/photocraft/pull/2580), [#2739](https://github.com/storytold/photocraft/pull/2739) | Respectively Spin/Lens/Extrude/Wind/Trap sampling, general compression/IO/compositing allocation work, and smart-filter stage caching/background jobs. #2685 touches Gallery's Spin path, not Tilt-Shift's Gaussian field. | No existing Tilt-Shift fix or overlapping implementation found. Keep this change confined to Tilt-Shift's opt-in and its tests. |

[Issue #464](https://github.com/storytold/photocraft/issues/464) records Gaussian
preview lag and profiling with copying and waits prominent; [#211](https://github.com/storytold/photocraft/issues/211)
tracks broader slow algorithms. They inform the need for complete-filter and
preview-workload measurements, rather than claiming a microbenchmark fixes GUI latency.

[RustFFT's primary documentation](https://github.com/ejmahler/RustFFT/blob/master/README.md)
describes O(N log N) transforms and automatic native AVX/SSE/NEON selection, with
different wasm SIMD requirements. That is useful existing infrastructure for
uniform expensive kernels, but does not remove the spatially varying blend or
improve the asymptotic cost of Gallery's existing box passes. No FFT or explicit
SIMD candidate is benchmarked here; the comparison above is an algorithmic and
maintenance assessment, not a measured claim that they can never be faster.

## Numerical and memory contract

Tilt-Shift has one band, described by `centerX`, `centerY`, `angle`, `focus`,
`transition` and `blur`; its engine/model has no pins parameter. Six nonzero
Gaussian levels plus the zero level retain their existing geometric spacing,
three-box radii, smoothstep sigma field and hat weights. Sharp/blurred and
transition pixels still blend their original levels. Pixel centres, image
coordinates, transparent premultiplication and edge policy are unchanged.

The optimized helper clones the already interleaved source and premultiplies it
with the same per-pixel operations. For each Gaussian level it keeps the
original top/left, trimming only right/bottom to the output plus that level's
summed box radii. Its artificial trailing edge cannot influence output through
the three finite-support passes. Keeping the leading edge preserves the
running-sum initialization, update order and f64 cancellation, including HDR.
No samples or levels are dropped. Cropped-output containment retains the
helper's existing fallback to a full window.

The existing source, premultiplied buffer, sigma map, accumulator and reusable
level buffer remain. Full-window capacity is still reserved once; populated
length and blur scratch shrink. There is no retained cache, feature dispatch,
threshold or planning cost. Source halo reads, auto tile sizes, worker count,
selection mixing, output writeback, undo/redo and cancellation services remain
the same. No change requires a native CPU feature or unsafe code.

One pre-existing crash found by the synthetic tests is fixed: a NaN low-level
blur amount previously became f32::clamp's invalid upper bound. It now acts as
identity. The regression asserts the frozen starting implementation panics and
the corrected kernel returns the source. Valid finite parameters keep exact bits.

`tests_tilt_shift_reference.rs` freezes the starting scalar Gallery/Tilt-Shift
expressions, using the unchanged Gaussian helpers. Differential tests compare
every finite f32 bit (including signed zero), with matching NaN classifications:
1–6 channels, RGB/Gray/CMYK/Lab and the expanded Bitmap/Indexed/Duotone/Multichannel
storage, U8/U16/F32, alpha/no alpha, narrow and soft band
boundaries, translated/rotated bands, tiny/empty/one-row images, blur thresholds,
full focus/full blur, extreme and nonfinite parameters, HDR/negative/high-contrast
samples, nonfinite pixels, selections, transparent/repeated edges and tile sizes
16/47/128. Cancellation before work and after progress preserves the source.
Malformed internal source lengths and zero channels are also checked.

## Measurement method

Windows x86_64, Ryzen 7 9700X (8 cores/16 logical), 64 GiB installed RAM;
stable Rust 1.99.0 / LLVM 23.1.1. Normal workspace release profile: opt-level 3,
thin LTO, one codegen unit. Eight Rayon workers. Builds/tests are stopped during
measurements, and process inspection checks for concurrent Cargo/rustc/benchmark
work. Ordinary system activity and thermal/allocator variation remain.

`bench_tilt_shift` constructs deterministic textured RGBA surfaces with varying
alpha and HDR F32 values. Two untimed warmups precede each case. Timing covers
complete `apply_in`: source/halo reads, tiled field and level computation, Rayon,
writeback and pruning. Fixture construction and output destruction are excluded.
It uses the unchanged pipeline, not an isolated hot loop. Baseline executables
are built before editing the algorithm and retained in local `plan/`.
The `blurred` case puts the band outside the image through the low-level Rust
API; it is a saturated-field stress case, outside the engine dialog's centre
range. All other cases use the supported dialog parameters. No pin list is
invented for Tilt-Shift. Model performance is measured for RGB; other models are
covered by differential correctness tests.

The preview workload models the existing 24 MP document's 1500×1000 filtered
proxy (factor 4), with blur 15/80/300 becoming 3.75/20/75. Position, angle, focus
and transition are relative and unchanged. This times filtering a proxy after a
parameter update. It excludes proxy construction, Session/history, queue/settle
delay, compositing, GPU upload and presentation; it is not actual GUI input/frame
latency. The implementation does not alter the preview factor or kernel.

```powershell
$env:CARGO_TARGET_DIR = 'target/agent-tilt-shift'
$env:RAYON_NUM_THREADS = '8'
cargo run --release -p photocraft-algo --example bench_tilt_shift -- 6000 4000 5 1
cargo run --release -p photocraft-algo --example bench_tilt_shift -- 6000 6000 5 1
cargo run --release -p photocraft-algo --example bench_tilt_shift -- 1500 1000 7 4
cargo run --release -p photocraft-algo --example bench_tilt_shift -- 128 96 7 1
```

## Results

Order: base A, candidate, base B, with all four sizes completed per implementation
round. Full-size cases have five timed samples per round; proxy and small cases
have seven. Base p50 pools the two unchanged rounds (10/14 samples); candidate
p50 uses 5/7 samples. Ranges include every timed sample, not only best runs.
Local raw logs are `plan/tilt-{baseline,candidate,baseline-repeat}-*.jsonl`.

The parameters are fixed by the example: `small` is blur 15, centred, angle 0,
focus 0.1, transition 0.15; `medium` is blur 80 at angle 27; `large` is blur 300,
angle -71, focus 0.03, transition 0.4. `wide_focus` is blur 80, angle 0, focus 0.4,
transition 0.05; `sharp_transition` is centred at (0.2, 0.8), blur 80, angle 63,
focus 0.05, transition 0.01. `focused` uses focus 1; `blurred` uses the low-level
outside-image centre (-5, -5). The latter two use blur 80. Proxy blur alone is
divided by four. U16/F32 time the rotated medium case.

### 24 MP (6000×4000)

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 1729.3 (1530.2–1996.6) | 1241.4 (1227.0–1256.5) | 1.39× |
| U8 small | 609.5 (594.2–616.1) | 451.7 (443.1–479.9) | 1.35× |
| U8 large | 5340.1 (5042.7–5861.9) | 3698.9 (3498.1–3722.3) | 1.44× |
| U8 wide_focus | 1204.9 (1177.8–1273.9) | 828.7 (792.5–874.5) | 1.45× |
| U8 sharp_transition | 1615.6 (1576.2–1636.4) | 1148.4 (1135.1–1153.2) | 1.41× |
| U8 focused | 433.5 (426.1–438.2) | 406.1 (399.6–410.5) | 1.07× |
| U8 blurred | 1105.7 (1051.6–1146.0) | 963.6 (938.0–1045.8) | 1.15× |
| U16 medium | 1935.8 (1877.7–1978.8) | 1306.8 (1268.9–1403.0) | 1.48× |
| F32 medium | 1961.7 (1858.5–2010.5) | 1316.8 (1270.2–1396.5) | 1.49× |

### 36 MP (6000×6000)

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 2458.7 (2066.6–2541.9) | 1668.0 (1656.9–1682.4) | 1.47× |
| U8 small | 806.6 (733.7–864.2) | 655.1 (627.0–678.9) | 1.23× |
| U8 large | 8856.0 (6766.5–9301.8) | 6176.2 (5780.6–6622.0) | 1.43× |
| U8 wide_focus | 1612.8 (1562.4–1672.7) | 1066.4 (1057.8–1084.3) | 1.51× |
| U8 sharp_transition | 2529.7 (2448.6–2590.1) | 1714.6 (1705.0–1890.4) | 1.48× |
| U8 focused | 646.3 (627.5–680.0) | 614.1 (613.8–618.5) | 1.05× |
| U8 blurred | 1633.2 (1595.3–1687.9) | 1577.6 (1566.4–1623.2) | 1.04× |
| U16 medium | 2618.4 (2498.4–2687.4) | 2020.7 (1994.9–2027.0) | 1.30× |
| F32 medium | 2669.3 (2561.2–2741.6) | 2020.5 (1734.9–2173.3) | 1.32× |

### 1500×1000 filtered preview proxy, factor 4

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 66.9 (63.1–82.0) | 55.9 (54.9–59.0) | 1.20× |
| U8 small | 29.4 (27.2–32.9) | 28.0 (26.4–29.9) | 1.05× |
| U8 large | 179.4 (157.0–205.6) | 132.8 (116.7–149.6) | 1.35× |
| U8 wide_focus | 52.1 (46.7–72.0) | 40.4 (38.9–44.0) | 1.29× |
| U8 sharp_transition | 64.3 (54.6–75.4) | 44.2 (41.2–45.7) | 1.45× |
| U8 focused | 17.9 (16.7–18.7) | 15.0 (14.6–15.1) | 1.20× |
| U8 blurred | 37.4 (36.1–48.0) | 29.5 (28.2–33.0) | 1.27× |
| U16 medium | 89.2 (81.3–95.8) | 58.1 (56.0–59.0) | 1.54× |
| F32 medium | 93.7 (83.1–102.4) | 60.3 (57.9–65.8) | 1.55× |

### 128×96 small-image overhead

| Depth / case | Base ms (range) | Candidate ms (range) | Speedup |
|---|---:|---:|---:|
| U8 medium | 30.25 (30.09–31.83) | 18.14 (17.69–19.34) | 1.67× |
| U8 small | 3.93 (3.85–3.99) | 3.15 (3.13–3.22) | 1.25× |
| U8 large | 278.90 (273.62–313.48) | 145.97 (142.65–148.51) | 1.91× |
| U8 wide_focus | 26.08 (25.29–27.79) | 15.48 (15.38–15.69) | 1.69× |
| U8 sharp_transition | 31.12 (30.48–46.89) | 17.40 (17.35–17.65) | 1.79× |
| U8 focused | 1.56 (1.50–1.64) | 1.07 (1.06–1.09) | 1.46× |
| U8 blurred | 7.57 (6.82–11.06) | 6.19 (6.05–7.02) | 1.22× |
| U16 medium | 30.99 (30.58–32.48) | 17.94 (17.61–18.07) | 1.73× |
| F32 medium | 31.53 (30.28–48.73) | 17.90 (17.82–18.74) | 1.76× |

All nine 24 MP cases have separated observed ranges. Gains are modest for the
focused and saturated stress fields. At 36 MP the saturated field's ranges
overlap; no convincing gain is established there. Default blur 15 on the preview
proxy also has overlapping ranges: its 1.05× pooled median change is inconclusive.
Medium/large proxy cases improve 1.20/1.35×, and U16/F32 medium proxy cases improve
1.54/1.55×. The small-image saturated case is also inconclusive; the remaining
small-image cases improve 1.25–1.91× with separated ranges.

There is meaningful drift, so the pooled p50 must not imply precise repeatability:
24 MP U8 medium base A/B medians are 1909.1/1564.3 ms (candidate 1241.4), giving
1.54/1.26× against the individual rounds. The 36 MP large case's base A/B medians
are 7234.7/9170.0 ms (candidate 6176.2), giving 1.17/1.48×. Process CPU sampling
during the repeat found the benchmark using about 7.6 cores and no other heavy
CPU consumer. The report retains the full variation rather than attributing it
to a proven cause.

For one complete 24 MP suite (fixture construction, warmups and all depths/cases
included), sampled Windows `PeakWorkingSet64` is 2,162,429,952 bytes for the base
and 2,044,227,584 for the candidate, about 5.5% lower (2.01 → 1.90 GiB).
This is an allocator-dependent process high-water measurement, not a per-filter
live-memory bound. Allocation counts are not instrumented; the source-copy,
sigma-map, level-metadata, output and level-buffer allocations remain, and full-window buffer
capacity is retained. Smaller populated levels and scratch account for the
structural reduction in work. Neither SIMD nor FFT crossover performance is
claimed; no adaptive threshold is introduced.

Maximum observed finite pixel error is **zero** in the differential tests. Every
pixel must match, so localized band-edge, seam, opacity or aliasing errors cannot
hide in an average-error metric. NaN classifications match. The NaN blur crash
regression intentionally differs from the starting kernel by returning identity.

## Validation and remaining limits

The full `photocraft-algo` suite passes: 425 tests, 18 ignored, no failures.
All six Tilt-Shift differential/regression tests also pass in release. Strict
all-targets Clippy passes for the algorithm crate. The existing engine filter
integration group passes all 16 tests, including selection, repeat, undo and
16/32-bit filtering. `cargo xtask layers` reports 29 crates with no violations;
`cargo xtask wasm` passes the required workspace packages and the HEIF-enabled
codec check. Regenerating the scorecard produces identical content.

`cargo xtask perf --quick` exits successfully; all five configured quick benches
(`perf_scenarios`, `type_bench`, `layout_bench`, `camera_raw_bench`, `bevel_bench`)
complete. Reports are retained locally in
`target/agent-tilt-shift/perf/{results.json,summary.md}`. The regression comparison
is skipped because the checked-in baseline is from an Apple M4 Pro/macOS machine
class. Quick mode does not enforce full-size budgets. The layout harness still
reports unrelated frame overruns; completing this gate does not establish
broader responsiveness or budget compliance. Formatting and Git whitespace
checks pass.

Algorithm checks also pass for aarch64/x86_64 macOS, aarch64 Windows and
aarch64/x86_64 Linux. These are compilation checks from Windows, not executions
or performance measurements on those platforms. Native timing evidence is
limited to this Windows x86_64 machine. `cargo xtask test-corpus --changed`
correctly skips its suites because no format, IO, compositor, GPU or text crate
changed. Command registrations and UI code are unchanged.

The measurements establish equivalence to the starting PhotoCraft kernel, not a
new Photoshop oracle or broader parity increase. Cold startup, actual GUI
input/frame latency, allocation counts and other platforms' native performance
remain unmeasured. No performance budget, scorecard checklist or parity floor is
raised from these local results. The shared craftrules checkout was unavailable
at the documented paths; the repository's own contributor and validation rules
were followed.
