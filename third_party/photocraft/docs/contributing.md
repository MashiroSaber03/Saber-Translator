# Contributing

- **Language:** Rust only. No JavaScript or TypeScript. On the web, `wasm-bindgen` generates a small loader; never hand-write JS.
- **Licence:** contributions are MIT OR Apache-2.0 (`LICENSE-MIT`, `LICENSE-APACHE`, `NOTICE`). The ArtCraft logos in `docs/brand/` are not open source (`docs/brand/LICENSE-brand.txt`).
- **Clean-room:** do not copy code, shaders, icons, ICC profiles or other assets from proprietary software (such as Photoshop). Match behaviour and look by observation and public specs. Only use assets with permissive licences, keep their license next to the asset (e.g. `assets/fonts/OFL-*.txt`, `assets/icons/LICENSE-lucide.txt`), and list every asset in `ATTRIBUTION.md`. **Fonts** live in [storytold/craft-fonts](https://github.com/storytold/craft-fonts), never in this repo: add new fonts there, and build with them through the optional `CRAFT_FONTS_DIR` build input (`docs/development.md` › Fonts; rules in `../craftrules/standards/fonts.md`, a sibling checkout — `storytold/craftrules` is not public, so ask a maintainer for the rules relevant to your change).
- **Never crash:** non-test code must not panic: no `unwrap`/`expect`/`panic!`/`unreachable!`/`todo!`/`unimplemented!` and no `unsafe`. Errors go through `Result` and `?`, input-derived indices and sizes are checked, and every crash fix ships with a regression test. See *Never crash* in `AGENTS.md`.
- **Commands, not handlers:** new features are engine commands with tests, and the UI calls them (checklist below).
- **Layering:** `cargo xtask layers` must pass. Register new crates in `xtask/src/layers.rs`.
- **Tests:** required for every change. Format code needs round-trip and malformed-input tests. Pixel code is tested at 8, 16 and 32-bit. If you touch psd, io, codecs, compose, gpu, text or format, also run the real-file corpus tests: `cargo xtask test-corpus` (fetches the pinned corpora, including our Photoshop oracles from https://github.com/storytold/photocraft-corpus, then runs the `corpus`-feature tests; CI always runs them). Never commit corpus files; see `docs/development.md` › Test corpora.
- **Style:** `cargo fmt`, and `cargo clippy -- -D warnings`. Match surrounding code. Comments explain *why*.
- **UI:** use `theme::Tokens` and `widgets::*`. Verify visually (offscreen `snapshot` example or the control channel) before submitting, and attach before/after screenshots to PRs.
- **Commits:** small, focused, with a clear subject line.

## Adding a command

1. **Find the id.** Search `crates/ui-egui/src/menu_catalog.rs` for the Photoshop menu item. Using its id makes the menu item live with no UI work. Commands without a Photoshop menu entry use a descriptive id in the same style (`layer.smartFilter.delete`) and an empty menu path.
2. **Algorithm** goes in the lowest crate that fits (`algo` for imaging, `paint`, `vector`, `text`, `cms`), with unit tests. It takes depth-agnostic surfaces (`photocraft-raster`), works per tile, and is deterministic (seeded randomness).
3. **Command** goes in an engine module (`crates/engine/src/<area>_cmds.rs`) exposing `specs()`, registered with `v.extend(...)` in `commands.rs`. Fill in:
   - `id`, `label`, `menu` path and Photoshop's default `shortcut`,
   - a params doc string such as `{"radius":px=4,"mode":"a|b"="a"}` (this is what agents read through `commands` / `command_list`),
   - an `enabled` predicate (greys the menu item out),
   - `run`, which returns a JSON result and records exactly **one** history step (use `coalesce` for drags and typing sessions).
4. **Respect the context:** active selection (feathered), layer vs mask target, locks, the colour model and depth.
5. **Tests** in the module: behaviour, undo/redo, disabled states, bad params, several depths.
6. **Dialog** (if it has parameters): filters and adjustments get schema-driven dialogs with live preview (`ui-egui/src/filter_dialog.rs`, `dialogs.rs`).
7. Run `cargo xtask parity` and commit the updated `docs/parity-checklist.md`.
