# Crates and dependency layers

The layer numbers below describe allowed dependency direction, not runtime privilege. A crate may depend only on lower layers. Standalone format crates remain independent of other workspace crates where specified.

| Layer | Crates | Responsibility |
|---|---|---|
| L0 | `geom`, `cms`, `color`, `raster` | Geometry, ICC processing, color/blend math, pixel formats, and copy-on-write tiles |
| L0 standalone | `psd`, `codecs`, `affinity` | PSD/PSB and flat raster format parsing/encoding; bounded Affinity preview extraction |
| L1 | `doc` | Pure-data documents, layers, masks, effects, adjustments, text, vectors, and smart objects |
| L2 | `ops`, `paint`, `algo`, `text`, `vector` | History operations, painting, imaging algorithms, type, paths, and shapes |
| L3 | `compose`, `gpu`, `format` | CPU composition, wgpu composition, and `.pcraft` persistence |
| L4 | `io` | Import/export between the document model and external formats |
| L5 | `engine` | `Session`, command registry, validation, history integration, and inspection |
| L6 | `ui-egui`, `automation` | egui shell, headless automation, MCP, and the live bridge |
| Test support | `testkit` | Shared test helpers |

Workspace applications are:

| Application | Path | Purpose |
|---|---|---|
| PhotoCraft desktop | `apps/photocraft` | eframe/wgpu application and loopback control server |
| PhotoCraft CLI | `apps/photocraft-cli` | Convert, inspect, run, batch, JSON-lines serve, and MCP |
| PhotoCraft web | `apps/photocraft-web` | The UI and engine compiled for WebAssembly |

New crates must be registered in `xtask/src/layers.rs`. Nothing below `ui-egui` may depend on egui, eframe, winit, or rfd. The `psd`, `codecs`, and `cms` crates have additional independence requirements documented in [`AGENTS.md`](https://github.com/storytold/photocraft/blob/main/AGENTS.md).
