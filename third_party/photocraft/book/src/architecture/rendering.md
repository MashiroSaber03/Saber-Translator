# Rendering

PhotoCraft has two compositors with distinct roles:

- `photocraft-compose` is the CPU compositor and correctness oracle used by export and tests.
- `photocraft-gpu` plans and executes supported canvas composition through wgpu.

The interactive canvas uses the GPU path when the planner can represent the document. Unsupported cases fall back to CPU composition. Current documentation identifies Multichannel documents and regions above the GPU texture limit as fallback cases.

Pixel storage in `photocraft-raster` supports runtime sample depth rather than a public 8-bit-only path. Color mode and profiles are document data; color conversion belongs in `photocraft-cms` rather than being treated as implicit sRGB.

## Verification

GPU changes require parity checks against the CPU oracle. UI changes require an actual rendered image, using either the offscreen snapshot example or a control-channel screenshot:

```sh
cargo run -p photocraft-ui-egui --example snapshot
cargo test -p photocraft-gpu
```

Performance-sensitive work should avoid full-surface work per frame, operate per tile, skip empty tiles, and cache by revision. The current commands and benchmark entry points are listed in [`docs/development.md`](https://github.com/storytold/photocraft/blob/main/docs/development.md).

GPU processing is not a security sandbox. Dimensions, decoded buffers, shaders, and command parameters must be validated before resource creation or dispatch.
