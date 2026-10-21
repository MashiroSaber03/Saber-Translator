# Unsharp Mask performance

Large-radius native Unsharp Mask uses separable FFT convolution with the original
discrete Gaussian coefficients and the existing `rustfft` dependency. A streamed
separable path preserves the original tap order for exact processing.

FFT is eligible for radii 64–1000 with at least 65,536 output pixels, a nonzero
amount and a zero or negative threshold. Positive thresholds, unstable samples,
unsupported geometry and insufficient scratch use the exact path. WebAssembly
retains exact processing. Alpha, selections, edges and cancellation are preserved.
Smart Sharpen, High Pass and shared blur math are unchanged.

FFT changes summation order and permits small rounding differences. The measured
maximum was one 8-bit storage level, approximately one 16-bit level, and 8.762e-6
for the HDR fixture. Exact-path comparisons were byte-identical. Transform axes
are capped at 32,768; FFT scratch grouping has a 1 GiB budget, excluding document
tiles. This is not a whole-process memory limit.

Before/after numbers, benchmark methodology, numerical tolerances and preview
measurement limits are recorded in [PR #2987](https://github.com/storytold/photocraft/pull/2987).
The comparisons use synthetic images and measure CPU computation, not slider
event-to-screen latency. No published performance budget or rendering-parity
floor is raised by these measurements.
