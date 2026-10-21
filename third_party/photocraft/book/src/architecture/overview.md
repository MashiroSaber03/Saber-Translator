# Architecture overview

PhotoCraft separates the document engine from presentation and transport. The desktop UI, CLI, JSON control channel, and MCP server reach editing behavior through the command registry rather than maintaining separate implementations.

```text
desktop UI       CLI       JSON control       MCP
    |             |              |              |
    +-------------+--------------+--------------+
                          |
                   command registry
                          |
                    engine Session
                          |
       document model + operations + compositors
                          |
             PSD / raster / .pcraft I/O
```

The enforced dependency direction runs from format-independent foundation crates through the document and operation layers to composition, I/O, the engine, and finally UI/automation. `cargo xtask layers` checks this rule.

The complete design narrative and dependency diagram remain in [`docs/architecture.md`](https://github.com/storytold/photocraft/blob/main/docs/architecture.md). That document contains historical design material as well as implemented architecture; confirm behavior in current source when security or compatibility depends on it.

## Core properties

- `photocraft-doc` holds the pure-data document model.
- `photocraft-engine::Session` owns open documents, active state, history, and command dispatch.
- `photocraft-compose` is the CPU reference compositor; `photocraft-gpu` accelerates supported interactive paths.
- `photocraft-io` maps between documents and PSD or flat formats.
- `photocraft-format` reads and writes the native `.pcraft` bundle.
- `photocraft-ui-egui` is a thin shell over engine commands.
- `photocraft-automation` exposes the headless engine and the live-GUI bridge.

See [Crates and dependency layers](crates.md) for the complete workspace map.
