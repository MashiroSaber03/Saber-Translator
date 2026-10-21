# PhotoCraft vs Adobe Photoshop: target-app parity

> **Last reviewed:** 2026-10-12 · **Last updated:** 2026-10-13 · **Change:** minor (`.atn` import maps 112 events; Free Transform H/V skew fields; stale rows corrected against `main` and the scorecard, estimates unchanged; Photoshop Actions `.atn` load and save; 2026-10-10: Tilt-Shift measurements added; earlier the same day: major, full re-measure against Photoshop 2026 27.11.0; merged `parity-estimate.md` and the roadmap's honest assessment; stale missing-tool rows corrected: Color Replacement and Red Eye are live toolbar tools, Frame landed; percentages unchanged) · **Target:** Adobe Photoshop 2026

The authoritative answer to "how close is PhotoCraft to Photoshop, and how much work is left". The
one-page summary is [`ROADMAP.md`](../ROADMAP.md); the ranked work list is [`gaps.md`](gaps.md);
the measured checklists are [`scorecard.md`](scorecard.md) (`cargo xtask scorecard`) and
[`parity-checklist.md`](parity-checklist.md) (`cargo xtask parity`). Detail by dimension:
[UI](ui-parity.md), [brushes and painting](brush-parity.md), [context menus](context-menu-parity.md),
[file formats](file-format-parity.md), [hardware](hardware-parity.md),
[localization](localization-parity.md).

## Headline (2026-10-10, PhotoCraft 0.6.0, `main` at 3e000bc1)

| Number | Value | Kind |
|---|---|---|
| **Feature breadth** (does each Photoshop feature exist?) | **~76%** | partly measured (menus, tools, panels, formats), weights below |
| **Ready for real work** (could a professional replace Photoshop with it?) | **~45%** (range 40–50%) | estimated, weights and evidence below |
| **Mainstream practitioner** (typical pro, weekly areas only) | **~43%** (38–48%) | estimated, [method](#mainstream-practitioner-43) |
| **Essentials user** (core features only) | **~61%** (55–67%) | estimated, [method](#essentials-user-61) |
| **Stage** | **alpha** | ready-for-real-work is inside the ~40–75% band and the [alpha gate](roadmap.md#alpha-gate) passes (six core workflows end to end); PSD exchange with Photoshop is not yet reliable enough for beta |
| Remaining to **beta** (~75% ready, PSD reliable) | **~1,100–1,800 Opus 5.5 agent-hours** | estimated, calibrated below |
| Remaining to **full parity** (~95–100%) | **~2,300–4,100 Opus 5.5 agent-hours** | estimated; includes AI and ecosystem, which need owner decisions |

These numbers replace the 2026-10-03 estimate (82% "of the way", 140–220 agent-hours to 95%,
kept below for the record). That estimate counted feature surface and was far too optimistic:
about 1,230 issues have been filed since the first release (579 open on 2026-10-10), most of them
against features already counted as present. The 2026-10-05 honest assessment (feature surface 60–70%,
ready for real work 25–35%) was closer; ~700 merged PRs since then moved both numbers up.

### Readiness by audience

| Audience | Ready % | Opus 5.5 agent-hours to ~95% | Work that dominates |
|---|---:|---:|---|
| Full target (ready for real work) | ~45% (additive weighted sum: 46%) | 2,300–4,100 | Everything below, plus AI tier, plug-in/scripting ecosystem, video, localization, hardware |
| Mainstream practitioner | ~43% | 1,700–2,800 | Depth in the 11 weekly areas (1,180–2,000), performance budgets, stability on real machines, PSD exchange with Photoshop |
| Essentials user | ~61% | 400–700 | Startup and GPU robustness, brush and transform responsiveness, Type tool, discoverability (flyouts, commit buttons), AVIF/PDF open |

Hours calibrated as in [Remaining effort](#remaining-effort-and-calibration) (2–9 h per checklist
item, 8–20 h per format or subsystem, from the repo's 1,004 merged PRs); each tier is a subset of
the one above. About 80% parallelizes (independent crates and command modules); performance
architecture and the PSD writer are the serial parts.

## Target measured

- **Adobe Photoshop 2026, version 27.11.0** (build 20260929.r.26, macOS universal), installed at
  `/Applications/Adobe Photoshop 2026`, with Camera Raw 18.7. Inspected without launching it:
  `Info.plist` document types, the file-format plug-ins' PiPL read/write flags, `.lproj` and
  `Locales/` language packs, `Default Keyboard Shortcuts.kys` (67 toolbox tools, 107 menu commands
  with default shortcuts), `Required/` plug-ins (85 filter entries, 12 format plug-ins), presets,
  and strings in the main binary (81 "… Tool" titles; HDR/EDR, Metal, Touch Bar, pen features).
- PhotoCraft's own Photoshop menu tree (`crates/ui-egui/src/menu_catalog.rs`, 628 items on 2026-10-10, 631 on 2026-10-12) was
  written from Photoshop's menus; it is the denominator of the menu number. Photoshop's binary
  `Default Menus.mnu` was not parsed, so items added in 27.x that the catalog lacks would not show
  (Generative and cloud items are the likely ones).
- User reports on GitHub (`storytold/photocraft`, 579 open issues on 2026-10-10) as evidence of
  depth and feel. Adobe's documentation and release notes for behaviour not visible in the bundle.

## Feature breadth: ~76% (weighted)

| Component | Weight | Ours / Photoshop | % | Kind |
|---|---:|---|---:|---|
| Menu commands | 30% | 631 / 631 catalog items dispatch a live command (628 / 628 on 2026-10-10) | 100% | measured (`cargo xtask parity`) |
| Toolbox tools | 15% | 59 / 68 (Photoshop's 67 `.kys` tools plus Remove, Triangle, Curvature Pen; minus legacy Rounded Rectangle and Targeted Adjustment) | 87% | measured (`Tool` enum in `ui-egui/src/state.rs`, `TOOL_SECTIONS` in `panels.rs`) |
| Panels | 10% | 30 / 35 Window-menu panels | 86% | measured (`view_cmds.rs`, panel modules) |
| Filters and adjustments | 10% | every Filter and Image › Adjustments item live; Filter Gallery 47 looks | 100% | measured (menu catalog) |
| File formats (usage-weighted) | 15% | 13 of 31 Photoshop formats supported in Photoshop's directions, 4 partial; the common ones (PSD/PSB, TIFF, JPEG, PNG, WebP) all present | ~80% (48% by count) | measured list, estimated weights ([file-format-parity.md](file-format-parity.md)) |
| AI / generative | 10% | classical Select Subject/Sky, Remove tool, Content-Aware; no Generative Fill/Expand, Neural Filters, Sky Replacement, Super Zoom | ~15% | estimated |
| Ecosystem (plug-ins, scripting, libraries) | 10% | WebAssembly plug-ins, CLI, MCP, actions, `.atn` load/save (112 events); no .8BF, UXP/ExtendScript, Libraries | ~20% | estimated |
| **Weighted** | 100% | | **~76%** | |

## Ready for real work: ~45% (weighted)

Weights reflect what a working Photoshop user depends on (photographers, retouchers, designers).

| Dimension | Weight | % ready | Remaining (agent-h) | Evidence | Detail |
|---|---:|---:|---:|---|---|
| Features in depth (tools, layers, selections, type, filters, transforms) | 25% | 55% | 600–1,000 | Scorecard checklists: Tools 20 done / 15 partial / 6 missing, Type 0 / 3 / 6, Automation 3 / 2 / 6. 14 tools missing; Select and Mask, Vanishing Point and Content-Aware Fill are parameter dialogs, not workspaces; type on a path, manual kerning, RTL paragraphs missing | [gaps.md](gaps.md) |
| UI / UX fidelity and feel | 15% | 40% | 350–600 | ~150 open UI issues; Workspace checklist 13 / 5 / 5 (2026-10-12; 5 / 7 / 11 on 2026-10-10); 27 of 156 preferences do nothing (was 46 of 149); no floating panel groups, thin Color panel, brush lag and pressure jumps reported | [ui-parity.md](ui-parity.md), [brush-parity.md](brush-parity.md) |
| File formats and fidelity | 15% | 60% | 300–500 | PSD oracle on Photoshop-authored files 133/256 (52%); io corpus 146/170 (86%); psd-tools 236/309 (76%); round trips ~100% of corpus; but users report Photoshop refusing our files (#1281, #2469; a text-layer case, #2374, was fixed 2026-10-10); no Photoshop PDF, EPS, JPEG 2000, JPEG XL, AVIF read, CR3 | [file-format-parity.md](file-format-parity.md) |
| Performance | 10% | 25% | 200–350 | 3 of 25 budgeted scenarios meet budget (baseline 2026-10-05); 150-layer nudge crashes the GPU path (P12); layer move on 15000×10000 16-bit 2.7 s p50; users report brush and cursor lag (#2619, #2566, #1994) | [scorecard.md](scorecard.md#performance) |
| Stability | 10% | 45% | 150–250 | Never-crash lints on 28/28 crates, `panic_hunt` in the gate, atomic saves; but field reports of freezes (#2518 Type tool 60 s on macOS), GPU panics (#2020, #1215), no-GPU start failures (#2519, #2412) | [gaps.md](gaps.md) |
| Hardware | 5% | 50% | 80–150 | GPU compositor (wgpu), pen pressure on all desktops but tilt/rotation not on Windows, Wayland pen broken (#2622), no HDR/EDR output, monitor profile auto-detected on macOS only | [hardware-parity.md](hardware-parity.md) |
| Localization | 5% | 55% | 80–140 | 7 of the 12 key languages at 100% of measured UI keys; Hindi and Arabic absent, Vietnamese catalog added 2026-10-12 (#2636, coverage not re-measured); no RTL UI; engine messages English | [localization-parity.md](localization-parity.md) |
| Platforms | 5% | 80% | 30–60 | macOS (notarized), Windows (unsigned), Linux (AppImage/deb/rpm/Flatpak), web, FreeBSD partial; Windows ARM64 is packaged and install-checked by `.github/workflows/windows-arm64.yml` (DIST-220-2); no in-app updates | [scorecard.md](scorecard.md#distribution) |
| Ecosystem (plug-ins, scripting, presets, libraries) | 5% | 15% | 250–450 | No .8BF hosting (#2647), no UXP/ExtendScript; .atn load/save (2026-10-12, 112 events, the rest kept verbatim), .abr/.grd/.pat/.aco/.ase/.cube supported | [gaps.md](gaps.md) |
| AI features | 5% | 10% | 300–600 | No ML runtime at all (scorecard AUTO-219-7); classical substitutes only. Needs an owner decision and openly licensed models (#41) | [gaps.md](gaps.md) |
| **Weighted** | 100% | **~46%** | **2,340–4,100** | | |

## Mainstream practitioner: ~43%

The same question for the typical professional Photoshop user (photographer, retoucher,
designer), not the whole product. Method: craftrules `standards/progress-docs.md`. Areas a
professional uses weekly (the [alpha-gate](roadmap.md#alpha-gate) workflows plus their everyday
tools), depth from the [feature areas](#feature-areas) table. Left out: AI, plug-ins and
scripting, Libraries/cloud, video, specialist hardware (tilt, Touch Bar), languages.

| Area | Weight | Depth |
|---|---:|---:|
| Layers, masks, blend modes, smart objects, styles | 20% | 65% |
| Adjustments and filters | 15% | 70% |
| Selections and masking | 15% | 55% |
| Retouching | 12% | 50% |
| Transform and warp | 8% | 55% |
| Type | 8% | 40% |
| Painting and brushes | 7% | 55% |
| Vector shapes and paths | 5% | 55% |
| Camera Raw | 4% | 45% |
| Colour management | 3% | 70% |
| Export, Save for Web | 3% | 50% |
| **Weighted depth** | 100% | **57%** |

Discounts for what still stops real work (estimated, each with evidence):

| Discount | Factor | Evidence |
|---|---:|---|
| Interaction fidelity | ×0.92 | ~150 open UI issues; spring-loaded tools, panel tear-off missing ([ui-parity.md](ui-parity.md)); brush lag and pressure reports ([brush-parity.md](brush-parity.md)) |
| Stability and speed on real machines | ×0.88 | 3 of 25 performance budgets met; 150-layer GPU crash (P12); Type tool freeze (#2518); GPU start failures (#2519, #2412, #2020); 41 open performance issues |
| Exchanging files with Photoshop users | ×0.92 | Photoshop-authored oracle 133/256; exports refused by Photoshop (#1281, #2469); smart-filter re-render 5/30 |

**57% × 0.92 × 0.88 × 0.92 ≈ 43%** (range 38–48), estimated.

It is not higher than ready-for-real-work (~45%) because the areas it leaves out carry little
weight in the full number (AI, ecosystem, localization and hardware together 20%, averaging
~33%), while the discounts apply to every workflow. The two numbers agree within their error;
the full number stays where its written weights put it.

**User evidence** (GitHub, 2026-10-10; search hits include maintainer replies, so they are upper
bounds): 0 issues mention switching from or replacing Photoshop; praise words appear in ~19
("love"), 10 ("amazing"), 6 ("awesome"), 4 ("great work") issues. Of 579 open issues, 136 are
feature requests and 443 bugs or reports, of which ~244 are on core paths (layers, selection,
brush, transform, type, save/open, performance) by title. Users are trying it for real work,
and most of what they report is core-path friction, not niche requests.

## Essentials user: ~61%

An occasional user who touches only the essentials: open or create a file, the most used tools
and commands, undo, save and export, default settings. Excluded: advanced options, pro
workflows, PSD exchange edge cases, and everything excluded above.

| Core feature | Weight | Depth | Notes |
|---|---:|---:|---|
| Save, Save As, Export JPEG/PNG | 12% | 85% | |
| Layers (add, reorder, hide, opacity) | 12% | 80% | |
| New / open a photo | 10% | 90% | |
| Select (marquee, lasso, magic wand, object) | 10% | 75% | |
| Move and Free Transform | 10% | 70% | commit and cancel buttons in the options bar (#2442, closed) |
| Brush and eraser | 10% | 70% | lag with big brushes (#2619) |
| Undo, History | 8% | 90% | |
| Basic adjustments (brightness, levels, curves, hue/saturation) | 8% | 80% | |
| Crop and straighten | 6% | 85% | |
| Text | 6% | 55% | freeze report (#2518) |
| Spot healing, Remove | 5% | 65% | |
| Blur and sharpen filters | 3% | 85% | |
| **Weighted depth** | 100% | **78%** | |

Discounts: launch and stability ×0.90 (start failures without a usable GPU, #2519, #2412,
#1859; Type tool freeze, #2518); discoverability and UI clarity ×0.92 (language option not
found, #2532; flyouts only on right-click, #2301; toolbar hover issue, #2603); opening files
people send ×0.95 (JPEG, PNG, HEIC, WebP, PSD open; AVIF and PDF don't, #463, #2541).
**78% × 0.90 × 0.92 × 0.95 ≈ 61%** (range 55–67), estimated.

## Feature areas

Percent is ready-for-real-work for that area (depth and fidelity, not presence). Hours are to full
parity for the area and are already contained in the dimension rows above.

| Area | % | Hours | Notes |
|---|---:|---:|---|
| Layers, groups, masks, blend modes, smart objects, layer styles | 65% | 150–250 | Broad and GPU-accelerated. Open: layer clipboard between documents (UI-217-23: `layer.copyToDocument` exists, no copy/paste), smart-object perspective (FILE-215-6), a user contour library (FILE-215-15). Multi-layer Smart Object conversion (#2052, #2651), Stamp Visible (UI-217-22) and mask targeting by command (#2458) are done |
| Selections, Select and Mask, channels | 55% | 120–200 | All selection tools, single row/column marquee included (#2837). Select and Mask is a dialog with Photoshop's view modes, a Refine Edge Brush and six outputs, not a workspace (TOOL-214-1), no Refine Hair (#2671); Select Subject/Sky classical, 559 ms p50 on 24 MP vs a 100 ms budget |
| Painting and brush engine | 55% | 150–250 | Every Brush Settings section exists; feel is the gap: lag on big brushes, pressure jumps, smoothing and edge quality reports ([brush-parity.md](brush-parity.md)) |
| Retouching (healing, clone, patch, remove, content-aware) | 50% | 100–180 | Remove tool non-generative; Content-Aware Fill has no workspace (TOOL-214-15); live stroke preview only for Brush/Eraser (TOOL-213-7) |
| Transform, warp, puppet, perspective, liquify | 55% | 100–180 | Free Transform modifiers right, with a right-click transform menu (TOOL-214-5) and H/V skew fields; Puppet Warp mesh follows the object outline (TOOL-214-6); Liquify called "useless" by a user (#2606) |
| Adjustments and filters | 70% | 120–200 | All present; smart-filter re-render matches Photoshop on 5 of 30 files; Filter Gallery looks approximate; no Neural Filters |
| Type | 40% | 120–200 | Engine works; no type on a path, manual kerning, RTL paragraphs, OpenType panel UI, live Warp Text preview; Type tool freeze on macOS (#2518) |
| Vector shapes and paths | 55% | 80–140 | Pen, Direct/Path Selection, shapes, shape strokes. Freeform and Curvature Pen (#1054). Add/Delete Anchor Point and Convert Point tools exist (`Tool` enum); missing on-canvas corner radius (TOOL-214-12), path-point nudge; shape parity tracking issue #2496 |
| Colour management and modes | 70% | 60–100 | Own ICC engine, soft proof, CMYK/Lab/Indexed/Duotone/Multichannel; no Adobe CMYK profiles, no spot plates (FILE-215-17), monitor profile only at launch |
| Camera Raw and computational photo (Photomerge, HDR, lens) | 45% | 120–200 | Raw for DNG/CR2/NEF/ARW/RW2/ORF/RAF(uncompressed)/PEF; no CR3; ACR dialog partial (no masking, calibration); generic lens profiles only |
| Automation, actions, batch, scripting | 45% | 120–200 | CLI/MCP/control channel ahead of Photoshop; Actions panel and recording not document-independent (AUTO-219-1/2); `.atn` load/save since 2026-10-12 (112 events mapped, the rest kept verbatim); no ExtendScript/UXP |
| File I/O | 60% | 300–500 | See [file-format-parity.md](file-format-parity.md) |
| Print, export, Save for Web | 50% | 60–100 | JPEG Options dialog missing (FILE-215-19), no Photoshop PDF, no separations, Save for Web settings not remembered |
| Video / timeline, data-driven graphics | 30% | 100–200 | Menu items live; frame model persists in `.pcraft`; no PSD video-layer mapping |
| Workspace, panels, preferences | 40% | 150–250 | 46 of 149 prefs do nothing; no floating/OS-window panel groups; Libraries, Comments, Contextual Task Bar missing |
| AI / generative | 10% | 300–600 | See the dimension row |

Batch supports [output naming templates](batch-naming.md); AUTO-219-5 remains partial.

## Remaining effort and calibration

**Calibration.** The repository's history from the first commit (2026-09-30) to 2026-10-10: 1,152
commits on `main` and 1,004 merged PRs in 10.5 days. Median PR open-to-merge time is 3.4 h
(p75 6.4 h, p90 14.3 h; that includes CI and review queueing). Comparable features:

| Feature | PRs | Lines added | PR hours |
|---|---|---:|---:|
| Crop tool parity set (#1919: nudge/swap/readout, overlays, Straighten, shield) | #1958, #2026, #2127, #2128 | ~2,050 | 6.5 |
| Crop rotation (#1792) | #1879 | 710 | 2.8 |
| Remove tool with non-local patch completion | #2193 | 2,005 | 9.0 |
| OpenRaster open and save | #2257 | 1,331 | 8.9 |
| Paint.NET import | #1998 | 1,461 | 7.0 |
| Shape stroke controls | #2218 | 1,506 | 9.0 |
| Red Eye tool | #1084 | 1,232 | 17.9 |
| Korean catalog and live switching | #582, #604 | ~3,800 | 13.0 |

So one checklist item of tool or UI depth costs about 2–9 agent-hours including tests, and a new
format or subsystem 8–20. With roughly 2–4 agent-hours per merged PR, the ~1,000 PRs so far
represent about **2,000–4,000 agent-hours**, which took ready-for-real-work from 0 to ~45%. The
remaining points are harder per point (fidelity against Photoshop's unpublished maths, feel,
performance on large documents, user-reported long tail), so the remaining estimate is not lower
than the work already done.

**Totals.**

| Milestone | Scope | Opus 5.5 agent-hours |
|---|---|---:|
| **Beta** (~75% ready, PSD reliable) | PSD fidelity and Photoshop re-open check, the performance budgets, the user bug backlog, the 15 missing tools, Select and Mask / Content-Aware Fill workspaces, type depth, UI feel (brush, transform, panels), JPEG/PDF/AVIF/JXL formats, Hindi/Arabic/Vietnamese and RTL | **1,100–1,800** |
| **Full parity** | the above plus the AI tier, plug-in/scripting ecosystem, video depth, long-tail fidelity | **2,300–4,100** |

**Parallelism.** About 80% parallelizes: areas are independent crates and command modules, and
the repository has sustained 10–20 concurrent agents. At ~12 agents, beta is roughly 100–150
wall-clock hours. Serial parts: GPU compositor and performance architecture, the PSD writer.

**Needs a human.** Owner decisions on the AI backend (#41: which openly licensed models, local
vs remote) and plug-in hosting (.8BF needs native code under the isolated-`unsafe` rule; UXP is a
JavaScript runtime, which AGENTS.md forbids in the UI); Windows code-signing material; tablet
hardware checks (Wacom/XP-Pen on Windows, Wayland, macOS); native-speaker review of every
catalog; checking exports by opening them in Photoshop (the owner's machine has it installed).

## How to re-measure

1. `cargo xtask parity` and `cargo xtask scorecard` for the measured rows.
2. Tools: diff `Tool` in `crates/ui-egui/src/state.rs` against the `<tool>` entries of
   Photoshop's `Locales/en_US/Support Files/Shortcuts/Mac/Default Keyboard Shortcuts.kys`.
3. Formats: Photoshop's `Info.plist` `CFBundleDocumentTypes` and the PiPL flags of
   `Contents/PlugIns/Required/File Formats`, against `crates/codecs/src/format.rs` and
   `crates/io/src/lib.rs`.
4. Languages: `cargo xtask i18n-coverage`.
5. Re-grade the estimated rows from the open issues by area and the corpus floors. Move a number
   only with new evidence, and add a revision-history row.

---

# Earlier assessments (kept for the record)

The two sections below are superseded by the tables above. They are kept because they hold dated
evidence that later numbers build on.

## Feature-parity estimate of 2026-10-03 (from the former `docs/parity-estimate.md`)

How close PhotoCraft is to Adobe Photoshop 2026, and how much Opus 5.5 work remains. Generated by
assessment, not a tool; revisit when a major arc lands. Menu coverage is from `cargo xtask parity`;
fidelity is from the PSD oracle (`cargo xtask corpus` / `oracle_diff`).

### Headline

> **Update 2026-10-03:** menu/command breadth reached **100% (625/625)** — every Photoshop
> menu item now dispatches a live command (incl. Variables/Data Sets, Trap, the Timeline panel and
> Layer › Video Layers). Remaining parity work is depth and fidelity, not breadth: PSD oracle
> (~66%), Camera Raw/video-layer depth, and the AI/Neural tier. See the table below.


- **Menu / command breadth:** 625 / 625 Photoshop menu items live = **100%** (every menu item has a live command).
- **PSD rendering fidelity (oracle):** ~111 / 170 corpus files pixel-match = **~65%**.
- **Overall functional parity (weighted, mainstream workflows):** **~86%** (menu breadth complete; depth + fidelity are the remaining gap).
- **Overall including Adobe's AI / Firefly features:** **~75%** (these are mostly cloud/ML and the
  hardest to reach clean-room).

Menu coverage (96%) overstates true parity because many live commands are faithful-but-approximate
(Camera Raw without masking/calibration, Vanishing Point without its UI, heuristic Select Subject/Sky
vs Adobe's ML, Content-Aware Fill basic). The weighted figure accounts for depth and fidelity, not
just presence.

### Where we stand, by area

| Area | Parity | Notes |
|---|---:|---|
| Layers, masks, groups, blend modes, smart objects, layer styles | ~92% | Broad + GPU-accelerated; bevel/satin shapes and a few blend edge cases remain |
| Selections, paths, vector shapes, pen | ~90% | Select menu 100% of items; Select Subject/Sky are heuristic, not ML |
| Paint / brush engine, retouch (clone, healing) | ~85% | Solid; no Mixer Brush depth, no Content-Aware healing |
| Adjustments + adjustment layers | ~90% | All kinds present; Camera Raw depth and a few toe-curve models approximate |
| Filters (incl. Filter Gallery ×47, Liquify, blurs, render) | ~88% | Gallery looks are approximations; no Neural Filters |
| Type / text engine, character & paragraph styles, glyphs | ~90% | Vertical type laid out and rendered (#199; no mojikumi or upright-Roman option yet); OpenType alternates thin |
| Color management (ICC v2/v4, intents, soft-proof) | ~90% | Own pure-Rust CMS; no Adobe CMYK profiles (synthetic) |
| File I/O: PSD, native .pcraft, flats (png/jpg/tiff/webp/exr/…) | ~85% | PSD oracle ~65% pixel-exact; most structure round-trips |
| Transform, warp, puppet, perspective, liquify | ~88% | No Face-Aware Liquify (needs a landmark model) |
| Automation: CLI, MCP, headless server, control channel | ~95% | Ahead of Photoshop here (see below) |
| Image modes (RGB/Gray/CMYK/Lab/Indexed/Duotone/Multichannel) | ~85% | Multichannel display-only; no ink-plate model |
| Print / export / Save for Web / slices / Generator | ~85% | No Separations/bleed; droplets basic |
| Computational photo (Photomerge, HDR, Lens, Camera Raw, Wide Angle) | ~70% | Classical implementations; generic lens profiles only |
| Data-driven graphics (Variables, Data Sets) | ~0% | Not started (5 menu items) |
| Video / Timeline / animation | ~0% | Not started (13 menu items) — a whole subsystem |
| AI / Neural / Generative (Firefly, Sensei, Neural Filters) | ~5% | Mostly absent; needs licensed/trained models |
| Prepress niche (Trap, measurement scale pro) | ~30% | Trap, ink spread not modelled |

### The remaining 25 menu items, grouped

- **Video & Timeline (13):** Video Layers (new/insert/duplicate/delete/replace/interpret/restore/
  reload/rasterize), Rasterize › Video, Timeline panel, Render Video, Import Video Frames to Layers.
  One cohesive subsystem: a frame/time model, a video decoder/encoder (clean-room Rust, no ffmpeg),
  onion-skinning, and a timeline UI.
- **Data-driven graphics (5):** Variables › Define / Data Sets, Apply Data Set, Export Data Sets as
  Files, Import Variable Data Sets. A template-binding model over layers + a CSV/data-set engine.
- **Niche / platform (7):** Trap (prepress ink spread), Frame from Layers, Migrate Presets, WIA
  Support (Windows scanners), and two already-adjacent items.

### Effort to parity (Opus 5.5)

Two clocks: **agent-hours** (total model work) and **wall-clock** (with this session's observed
~6–8× parallelism across independent arcs). This session landed ~70 menu items + major subsystems
in a handful of wall-clock hours using 8 parallel agents, so wall-clock ≈ agent-hours ÷ 6.

| Milestone | Scope | Agent-hours | Wall-clock |
|---|---|---:|---:|
| **To 90% functional** | remaining non-AI/non-video menu items (Variables, Trap, Frame, Migrate, WIA), Camera Raw depth, Vanishing Point UI, selection quality, polish | 60–90 | 10–15 h |
| **To 95% functional** | the above + PSD oracle 65% → ~90% (bevel/satin shapes, Lab blending, noise-gradient RNG, toe curves), broad UI-parity pass, perf pass | +80–130 | +15–22 h |
| **Video / Timeline subsystem** | frame model, clean-room decode/encode, timeline, animation, Render Video | 50–80 | 10–14 h |
| **Credible AI tier** | ML Select Subject/Sky, Content-Aware Fill (PatchMatch++), basic generative fill/expand — needs model integration; some features may be infeasible fully clean-room without licensed models | 120–250+ | 25–45 h |

**Totals:**
- To **~95% functional parity** (everything except video and AI): **~140–220 agent-hours ≈ 25–40
  wall-clock hours** of Opus 5.5 with parallel agents.
- To **~98%** incl. the video subsystem: **+50–80 agent-hours ≈ +10–14 wall-clock hours**.
- To **~99%** incl. a credible AI tier: **+120–250 agent-hours ≈ +25–45 wall-clock hours**, with the
  caveat that pixel-for-pixel Firefly/Sensei parity is not achievable clean-room without trained
  models; we target functional equivalents.

**Best single number:** PhotoCraft is **~82% of the way** to full Photoshop parity, and reaching a
**95%, ships-for-most-users** state is roughly **25–40 wall-clock hours** of parallel Opus 5.5 work.
Closing the last 5% (video + AI) is the long, uncertain tail.

### Assumptions & caveats

- "Parity" = functional + visual equivalence of what a user can do, not byte-identical output. Clean-
  room means no Adobe code, profiles, shaders, or trained models; AI features target equivalents.
- Wall-clock assumes 6–8 independent arcs in parallel with their own target dirs, as this session ran.
  Serial work is ~6× longer.
- Adobe removed 3D/Substance from Photoshop, so 3D is intentionally out of scope.
- Fidelity hours are the least certain: the remaining oracle failures are proprietary RNGs and
  unpublished curve models fitted by observation.

## Honest parity assessment of 2026-10-05, with dated evidence to 2026-10-10 (from the former section of `docs/roadmap.md`)

This is the reference answer to "how close are we to Photoshop parity, really". Agents: read it
before picking work. Update it (with dated measurements) when the numbers move; don't restate the
menu-parity number in its place.

**Measured numbers live in the [scorecard](scorecard.md)** (`cargo xtask scorecard`, kept current
by CI): performance scenarios against their budgets, corpus floors, per-area checklists with
done / partial / missing counts, and the number of settings that do nothing. Where the scorecard
and the estimates below disagree, the scorecard's numbers supersede them. As of 2026-10-05 it
counts 67 of 129 preferences that nothing reads; 3 of the 25 budgeted performance scenarios meet
their #209–#211 targets (first baseline, taken on a heavily loaded machine), and the 150-layer
nudge scenario crashes the GPU compositor; the open area issues' checklists (#203–#220) are
almost entirely missing or partial.

2026-10-07: UI Font Size now applies to interface text, including CJK fallbacks, independently
of display and canvas zoom (#532). The current preference audit drops from 59 to 58 unread
settings out of 135; see the regenerated scorecard.

2026-10-09: Output-preserving Radial Blur direct sampling measured on a local
i7-9750H, 12 workers, 24 MP RGBA8, Good quality, amount 1: Spin 66.10 → 16.43 s
(4.02×), Zoom 19.19 → 5.35 s (3.59×), one paired run each, exact output equality.
These are local measurements at amount 1; see [method and limits](radial-blur-performance.md).

2026-10-10: native selection distance transforms measured 7.59–10.57x faster on six
24–36 MP synthetic masks on an AWS c7i.4xlarge (16 workers, three paired release runs).
The transform also corrects f32 envelope errors past coordinate 4096, including nonzero
distances at selected pixels. The parallel path needs an additional four bytes per pixel;
see [measurements, correctness and limits](selection-distance-performance.md).

**Bottom line.** Two days after 0.2.0 we had merged ~96 PRs and closed ~48 issues, but **real
Photoshop parity is still well below 50%**. The biggest gaps are AI, missing tools, professional
workflow depth and the plug-in ecosystem. Most fixes since 0.2.0 have passed our tests but have
**not yet been validated by users**, and this week proved our tests miss what users hit:
on first public use they found broken basics (text selection offset, 214 shortcut failures, panels
resizing themselves, an immovable crop frame, folders that wouldn't collapse, lag on layout PSDs)
that all counted as "live" in `parity-checklist.md`.

**Overall.** Feature surface: roughly **60–70%** of what a typical Photoshop user touches exists in
some form. "A professional could switch for daily work": roughly **25–35%**. True 1:1 parity is
**many months** of focused work, and some areas need product decisions rather than effort.
Confidence: moderate — the next users of 0.2.x will move these numbers either way.

### By dimension

2026-10-10: Tilt-Shift reuses the established contiguous premultiplication and
per-level trailing-halo trimming. Nine local 24 MP RGB U8/U16/F32 cases improve
1.07–1.49× with separated observed ranges (Ryzen 7 9700X, Windows, eight workers):
default blur 15, 609.5 → 451.7 ms; rotated blur 80, 1729.3 → 1241.4 ms. At 36 MP,
rotated blur 80: 2458.7 → 1668.0 ms; the saturated-field gain is inconclusive.
The existing filtered preview proxy at document blur 80/300 improves from
66.9/179.4 → 55.9/132.8 ms; its default blur 15 gain is inconclusive.
Finite pixel bits match the frozen starting kernel exactly. These are computation
timings with two baseline rounds, not GUI frame latency or a Photoshop parity
increase; see [method, upstream SIMD/rustfft investigation and full ranges](tilt-shift-performance.md).

2026-10-10: Iris Blur pin preparation, saturated-distance evaluation and trailing
Gaussian-level halo trimming measure 1.18–1.76× faster across eight local 24 MP
RGB U8/U16/F32 cases (Ryzen 7 9700X, Windows, eight workers). Default blur 15:
626.8 → 426.9 ms; two pins: 1789.9 → 1016.2 ms. The default 1500×1000 filtered
proxy for 24 MP, document blur 15/80: 28.7/51.8 → 24.9/42.6 ms. At 36 MP,
two pins: 2595.8 → 1822.8 ms; small/medium U8 gains are inconclusive.
Differential tests preserve exact finite pixel bits.
These are computation timings, not GUI frame latency or new Photoshop parity;
see [measurements, upstream SIMD findings and limits](iris-blur-performance.md).

2026-10-10: Quick Selection object strokes through raster surfaces measured
3.46-3.70x faster at 8-bit, 16-bit and 32f on 24-36 MP documents (AWS
c7i.4xlarge, five paired release runs), with identical region bounds and mask bytes.
Hybrid FIFO push/relabel and sparse edge enumeration preserve the fitted energy.
See [stroke measurements and reproduction](quick-selection-performance.md).

2026-10-10: Pixelate's Facet shares exact 3x3 quadrant statistics and Color Halftone
shares dot radii within each rotated screen grid. Full tiled CPU application at 24 MP
on a 16-vCPU m7i.4xlarge improves Facet from 0.684 s to 0.312 s in RGB8
(2.20x), and from 7.888 s to 0.574 s in CMYK8 (13.74x).
Color Halftone improves from 1.137 s to 0.604 s in RGB8 (1.88x)
and from 3.589 s to 0.718 s in Lab8 (5.00x).
Three alternating paired release runs also cover 36 MP RGB, 16/32-bit storage,
and screen radii 2/8/64. All 36 complete output comparisons match exactly.
See [method, raw timings and checks](pixelate-performance.md).

2026-10-10: Native CMYK solid and gradient fills share depth-quantized samples between
rendering and PSD export. The uncached 16-bit CMYK gradient now round-trips with zero
rendered error; psd-tools round trips rise from 307 to 308. Mixed and Photoshop round
trips remain 169 and 258; rendering oracle counts remain 146, 237 and 133.
Exact 8-bit CMYK row reuse reduces 1 MP conversion-read time by 77% for smooth ramps,
33% with dithering and 10% for random colours (M4 Max, release optimization level 3).

2026-10-10: [Shape stroke controls](shape-strokes.md) expose existing Solid/Dashed/Dotted,
custom dash/gap, offset, caps, joins, alignment, miter-limit and opacity capabilities in
Shape/Pen options and Properties. Properties stages a cached canvas preview before one
undoable edit. Gradient fill controls from #1053 remain open; the Line tool still draws a
filled bar. This extends UI reachability and does not claim Photoshop-authored stroke parity.

| Dimension | Measured / evidence (2026-10-05) | Grade | Notes |
|---|---|---|---|
| Menu wiring | 626/626 menu items dispatch a command (`parity-checklist.md`) | high but shallow | Says nothing about behaviour. |
| PSD fidelity (rendering) | Corpus oracle 115/170 (68%): 30 differ, 26 have no usable reference, 1 import error. 2026-10-07: io corpus 146/170, psd-tools corpus 236/309 (was 229: Advanced Blending knockouts), Photoshop oracles 132/258 | medium | Push to 170/170 under way (effects/strokes, multi-instance effects, 16/32-bit and colour modes, references for skipped files). |
| PSD round trip | 169/169 re-import identically; every adjustment layer and blend mode round-trips | high (within corpus) | Floors in `crates/io/tests/corpus.rs`; raise, never lower. Corpora: `cargo xtask corpus --all` (ours: https://github.com/storytold/photocraft-corpus). |
| OpenRaster read/write | 2026-10-09: 34 Krita 5.2.9-authored `.ora` files (27 blend modes, groups, pass-through, offsets, opacity, visibility, masks, 16-bit, gray) render within 3/255 of Krita's merged image (30) or differ for known reasons (4: Krita's lighter/darker-colour tie-break, Hard Mix at exactly 1, a linear-light 16-bit document); Krita re-opens our exports and renders them identically to its originals (33/34; Soft Light is written as `svg:soft-light`) | medium (one authoring app) | MyPaint and GIMP files untested. Masks are applied to pixels and layer styles dropped on save (reported). See [OpenRaster](ora.md). |
| Paint.NET import | 2026-10-09: 14/14 local PDN3 textures preserve editable layers exactly through `.pcraft` and render within 2.142/255 of full-size previews; a supplied 5.x document (1280×720, 11 layers) passes editing/history/native saves and matches its thumbnail within 0.418/255 mean; all 14 blend modes exercised synthetically | medium (limited corpus) | Import only; metadata omitted. More current-version files and full-size exports needed. See [PDN support](pdn.md). |
| Smart filters / text / effect shapes in PSDs | Measured on our Photoshop-authored set (https://github.com/storytold/photocraft-corpus, `corpus/photoshop`, 258 files): see the per-group floors in `crates/io/tests/corpus.rs` and `crates/engine/tests/photoshop_oracles.rs`. Smart objects and smart filters now survive PSD save and open (41 corpus files, 206 smart objects, round trip strict; Photoshop opens our exports with live filters); re-rendering Photoshop's smart filters with ours matches 5/30 (was 1/30) | low–medium | Remaining re-render gaps are filter maths (Gaussian/Motion Blur, Unsharp Mask, Emboss, Add Noise RNG) and bicubic placement. |
| Core editing (layers, masks, selections, adjustments, filters, transforms) | Broad engine coverage; many interaction bugs fixed after 0.2.0 (adjustment dialogs, Curves, crop, Move/Transform modifiers, gesture origin) | medium | Fixes not yet user-validated. |
| UI / UX polish | Shortcut audit 214 → 0 failures; dock, Layers rows and menus reworked; first visual-QA sweep found 14 defects (#147–#157). 2026-10-07: the keyboard-only shortcuts with no menu item (⌥[ ⌥] ⌥, ⌥. layer navigation, ⇧⌥[ ⇧⌥] to extend the selection, 1–0 for opacity and ⇧ for flow or fill, ⇧[ ⇧] hardness, ⌥⌘T to transform a copy and ⌥⇧⌘T to step and repeat), ⌥-click colour sampling with painting tools, double-click a Layers row for Layer Style, File › New from Clipboard, a centred main window and remembered Liquify settings (#352, #417, #350, #368, #419, #418). 2026-10-09: canvas zoom is physical at any display scale — 100% is one document pixel per physical display pixel, and every screen-space overlay, cursor, scrollbar, navigator and fit follows it (#1943) | low–medium | Needs recurring visual QA with realistic documents. |
| Tools | 2026-10-10: Pencil, Mixer Brush, Color Replacement, Patch, Content-Aware Move, Vertical Type, Pattern Stamp and Rotate View are toolbar tools. The only tool still missing is Art History Brush: Frame landed (#2914), Red Eye is a live toolbar tool (`paint.redEye`, TOOL-213-17) and Perspective Crop landed (2026-10-10, #1043, `image.perspectiveCrop`) | low–medium | Patch has a live healing preview. Content-Aware Move is partial: Move/Extend, Structure/Color and Sample All Layers exist; Transform On Drop and a live result preview remain missing (`TOOL-213-4` in the scorecard). Magic/Background Eraser added; live gradients in progress (#180). Magnetic Lasso added 2026-10-08 (live-wire edge tracing, Width/Contrast/Frequency, `select.magneticLasso`). Pattern Stamp added 2026-10-08 (S flyout, `paint.patternStamp`, Aligned / Impressionist). Rotate View (2026-10-09) turns the canvas camera around its centre without rewriting pixels; Reset View and Match Rotation copy the angle. Crop rotation (2026-10-09, #1792): dragging outside the crop frame turns it (⇧ 15° steps, angle readout), and ↵ rotates the document and crops in one undo step (`image.crop` `angle`); the frame turns over the image as in Photoshop's Classic Mode, and the options-bar Straighten is still missing. |
| Painting | Brush model and Brush Settings panel near Photoshop; .abr/.grd import; persistent presets; pen pressure/tilt on Windows, web, macOS and X11 | medium | Native Wayland pen input open (#79; a pen opens the window through Xwayland meanwhile); X11 pressure confirmed by a user, macOS not yet verified on tablet hardware. |
| Text / typography | Engine works; caret placement and size editing fixed; OpenType features, text-on-path editing, composer parity partial | medium-low | Measure with the Photoshop-authored set. |
| Colour management | Colour-managed canvas (document → monitor), embedded CMYK profiles, linear EXR/HDR, 16-bit float canvas | medium-high | Monitor profile follows only at launch. |
| Performance | 14k+ px on the GPU at ~⅓ the memory; adjustment preview 285 ms → 4–9 ms; font-size edits 297 ms → 4.6 ms; 2026-10-07: 30 MP TIFF open (banded, parallel strip/tile decode) Deflate 345 → 32 ms, LZW 428 → 43 ms, BigTIFF and every IFD readable; 2026-10-09: 4.2 MP Indexed Color, 256 colours, full-resolution CPU preview p50 3018 → 1061 ms (Ryzen AI 7 350, three measured runs; GPU upload excluded); 2026-10-10: Liquify 3500×2200, 800 px brush, visible 1:1 crop 3.20 → 3.63 ms p50 (5 runs) while refreshing ~9× as many pixels | medium-high on rasters | Complex layout documents still laggy (#125/#128); >16384 px GPU tiling in progress (#49). Liquify now displays source-resolution detail above 100% zoom; its ≤1 ms update target is not met (3.73 ms p95 on a Windows run; GPU upload excluded). Native Indexed Color previews run on one background worker with stale-result rejection; large palettes can still take about a second to compute. Web previews remain synchronous. |
| Stability | Never-crash lint series, crash guard, `panic_hunt` fuzzing in the gate | medium-high | No field crash data yet. |
| Camera RAW | DNG, CR2, Sony ARW (lossless + compressed), Nikon NEF (lossless + lossy compressed), RW2, uncompressed ORF; 2026-10-09: uncompressed Fujifilm RAF, Bayer and X-Trans (edge-directed X-Trans demosaic; 26 MP X-Trans develop 306 ms on 12 cores) | medium | CR3, compressed RAF and Nikon "lossy after split" still open (#50). Non-DNG cameras have no calibration: their colour (matrix + tone curve) is fitted per file to the camera's embedded JPEG, with the neutral sRGB-primaries fallback when the fit doesn't clearly help (2026-10-08: 8 D3200 NEFs, mean ΔE vs the camera JPEG 11.6–26.0 → 1.0–7.5); measured per-model calibration still open. |
| AI / generative | none | ~0% | Deferred by decision (#41). |
| Ecosystem | Sandboxed WebAssembly plug-ins instead of .8BF; `.atn` actions load and save (2026-10-12); no ExtendScript/UXP; no Adobe Fonts/Libraries/cloud docs | low | By design for 8BF; scripting compatibility open. |
| Platforms | macOS (notarized), Windows, Linux (AppImage/deb/rpm/Flatpak bundle), web | medium-high | Flathub later (#173); Windows signing material pending. |
| Localisation | 2026-10-07: 10 UI languages; menu, `tl!`, blend mode, preference and brush-section coverage enforced by tests; live switching and scoped Preferences previews | medium | Engine errors/status messages still partly English; CJK web fonts, browser-locale detection, and RTL remain open. |

2026-10-08: Motion Blur adds adaptive FFT convolution for wide streaks while retaining the
row kernel merged in [#902](https://github.com/storytold/photocraft/pull/902) for shorter
streaks and unsupported cases. Native release medians of three paired runs on an Intel
i7-9750H with 12 Rayon workers, synthetic RGBA8 at 30° / distance 1000: 24 MP
22.594 → 9.785 s (2.31×), and 1.5 MP
2.455 → 0.592 s (4.15×).
Full-image comparisons differ by at most one U8 code level; distance 64/256 retain the
byte-identical row output. These measure decode/halo/filter/write/prune, excluding UI proxy
creation, compositing and upload; the UI remains synchronous. FFT work is scheduled within
a conservative 512 MiB working-set estimate, excluding stored Surface tiles and row fallback
memory. The crossover is a heuristic; images dominated by near-cutoff alpha can need extra
scalar work. These measurements do not establish Photoshop filter parity.

2026-10-10: Proximity Match search medians on an AWS c7i.4xlarge improved 1.54–3.82x
for four synthetic stroke regions (50–300 px holes), with identical source displacements.
Bounding the mask table to the hole and pruning losing SSD candidates reduced table storage
by about 90–93 percent there. Full 24–36 MP kernel inputs improved 4.60–6.24x, but normal
engine strokes already use a cropped region. See [method and limits](proximity-match-performance.md).

2026-10-09: the Remove tool is in the J flyout (`paint.remove`, a background job). A stroke
around an object also removes what it encloses. The fill is non-local patch completion with
texture features (A. Newson et al., IPOL 2017: `crates/algo/src/nonlocal.rs`), finished by a
best-patch copy and gradient-domain seam hiding. On 20 holes in public-domain photos
(`remove_quality`, `docs/development.md` › Remove Tool quality) the fill keeps 0.89 of the
original's texture (Wexler/PatchMatch completion as Content-Aware Fill uses it: 0.54). On a 24 MP
document a 100 px ring takes 120 ms and a 1000 px ring 2.5 s (`remove_bench`). It is not
generative: large objects over complex structure fill less convincingly than Photoshop's AI mode
(#41). 2026-10-12: the options bar has Photoshop's three options: Sample All Layers, Create a new
layer (the fill lands on a new layer above the target, `paint.remove` `newLayer`) and Remove after
each stroke; off, the strokes collect as a tint and ↩ (or ✓) applies them together as one job and
one undo step (`strokes`), Esc (or ⊘) drops them (#1750).

2026-10-08: the tools update above is checked against the current toolbar groups in
[`panels.rs`](../crates/ui-egui/src/panels.rs), the Pencil and tool-cycle tests in
[`pencil_tests.rs`](../crates/ui-egui/src/pencil_tests.rs), Patch's live-preview test in
[`patch_preview.rs`](../crates/ui-egui/src/patch_preview.rs), and Vertical Type's point/paragraph
creation and undo test in [`type_tool_tests.rs`](../crates/ui-egui/src/type_tool_tests.rs).
These implementations remove the tools from the missing list; they do not establish Photoshop
behavioural parity. The historical estimates and corpus measurements in this assessment are
unchanged; current measured floors remain in the scorecard.

2026-10-08: GPU lower-stack checkpoints reuse unchanged composites after property edits.
On a dense 24 MP / 12-layer stack, warm top-layer opacity edits at 8/16/32-bit depth measured
p50 229/228/186 → 31/25/26 ms (AMD Radeon 860M, Windows, Vulkan; synchronized compositor only,
`bench_stack_reuse`). Gains in the existing 20 MP P2–P5 canvas fixture were small and
run-order sensitive: a reversed-order repeat measured p50
13.2/13.2/12.0/8.2 → 12.5/12.2/12.0/8.0 ms; P5 p95 remains 13.7 ms, above its 10 ms budget.
One checkpoint is capped at 512 MiB within the existing total budget;
the 24 MP run held 384 MB. Planning and output/mipmap work remain. These local comparisons
do not replace the published baseline or enable budget enforcement on other hardware.

2026-10-07: Camera Raw PSD mapping covers relative custom white balance, Light/Presence,
parametric and four point curves, HSL, Color Grading, sharpening/noise detail, grain and numeric
post-crop vignette controls. Two revisions of one supplied Photoshop ACR 18.4 PSD preserve their
filter descriptors without edits. The updated revision stays opaque because of active manual
Optics and unverified vignette style; a descriptor projection verifies its supported controls.
Synthetic 8/16/32-bit PSD → edit → `.pcraft` → PSD tests preserve the filter stack, mask and
undo/redo. Unmapped Camera Raw fields/versions remain opaque; Photoshop
acceptance of generated exports and pixel parity are still unverified. Corpus floors above are
unchanged.

2026-10-10: Select and Mask's Shift Edge now computes the exact octagonal footprint
with a square window extreme plus a dyadically built Manhattan ball. On a c7i.4xlarge
(16 vCPUs), full tiled CPU refinement at radius 64 / Shift Edge +100% takes
3.019 s -> 0.762 s at 24 MP (3.96x),
3.706 s -> 0.921 s at 36 MP (4.02x).
Three alternating paired release runs include guide sampling, guided filtering,
boundary distances, quantization and assembly. Every output mask matches exactly.
The dense fractional 24 MP radius-64 dilation kernel improves 36.00x;
circular soft-mask kernels improve 20.76-172.16x at 24-36 MP, radii 4-64.
Bounded halo tiles reduce the large-mask workspace allocation bound from 192/288 MB
to about 115/163 MB, including output and 16 workers' scratch, excluding the source.
See [method and results](octagonal-mask-performance.md).

2026-10-10: five CPU imaging algorithms now reuse source work and interpolation.
On an m7i.4xlarge (16 vCPUs), three alternating paired release runs measured
dense Extrude Blocks (8/128 px, 24 MP RGB8) 160.301 s -> 0.387 s (414.69x);
Blast (24 MP CMYK8) 6.231 s -> 0.709 s (8.79x);
Spin Blur (36 MP RGB8) 9.439 s -> 5.869 s (1.61x);
Lens Correction (24 MP CMYK16) 0.808 s -> 0.565 s (1.43x);
Trap (50 px, 36 MP CMYK16) 4.627 s -> 1.333 s (3.47x).
Default 30/30 px Extrude is 6.26x faster at 24 MP. All 72 full-operation output
comparisons match exactly, including alpha, across 24 cases and 8/16/32-bit depths.
Timers include surface assembly but exclude UI, undo and Trap document flattening.
See [method, raw measurements and reproduction](five-algorithm-performance.md).

### Where we're going (priority order)

1. **Ship the fixes:** cut 0.2.1 once the current batch lands, so users validate them.
2. **PSD fidelity to 170/170** with round trips, plus the Photoshop-authored reference set for smart
   filters, the text engine and effect shapes (https://github.com/storytold/photocraft-corpus, fetched by
   `cargo xtask corpus --all`); enforce floors in `corpus.rs`.
3. **Workflow acceptance tests:** 30–50 real tasks (retouch a portrait, social post with text and
   effects, composite with masks and adjustment layers, CMYK print prep…) scripted end to end and
   checked against Photoshop's output on every build. Their pass rate becomes the headline parity
   number.
4. **Tool coverage and workflow depth:** Pen variants (Direct Selection landed, #790; Add/Delete Anchor Point, #1044);
   finish Content-Aware Move's Transform On Drop and live result preview.
   Pencil, Mixer Brush, Patch, Vertical Type, Pattern Stamp and Rotate View are already available on the toolbar.
5. **Complex-document performance** (#125/#128) and GPU tiling beyond the texture limit (#49).
6. **Recurring visual QA** (`cargo run -p photocraft-engine --example designer_psd`) and fast
   turnaround on user reports (OS, document size, layer count, screenshot).
7. Later / needs decisions: generative AI backend (#41), scripting compatibility (ExtendScript /
   UXP / .atn), Flathub (#173), Wayland pen pressure (#79).

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-13 | minor | Stale "missing tool" rows corrected against main: Color Replacement (`paint.colorReplacement`), Red Eye (`paint.redEye`) and Add/Delete Anchor Point (the Pen flyout) are live toolbar tools and Frame landed, so Art History Brush is the only tool still missing; percentages unchanged |
| 2026-10-12 | minor | Free Transform options bar H/V skew fields; percentages unchanged |
| 2026-10-12 | minor | `.atn` import: 112 events mapped (TrueGrit Texture Supply sets play end to end); percentages unchanged |
| 2026-10-12 | minor | Photoshop Actions `.atn` load and save (90 events mapped both ways); percentages unchanged |
| 2026-10-12 | minor | Rows checked against `main` and the regenerated scorecard: menu count 631, Workspace checklist and preference count, transform context menu, Puppet Warp mesh, Select and Mask, anchor-point tools, Smart Object conversion, Stamp Visible, mask targeting, Free Transform commit/cancel, Vietnamese catalog, Windows ARM64; percentages, weights and hours not re-measured |
| 2026-10-10 | minor | Added measured Tilt-Shift computation/filtered-proxy gains and inconclusive cases; parity percentages and budgets unchanged |
| 2026-10-10 | minor | Readiness-by-audience table: percent and hours for full, mainstream and essentials; full number confirmed as the additive weighted sum over dimensions (46%, reported ~45%) |
| 2026-10-10 | minor | Added mainstream-practitioner (~43%) and essentials-user (~61%) numbers with written weights and discounts; full ready-for-real-work rechecked against its written weights (46%, reported ~45%), unchanged |
| 2026-10-10 | minor | Stage checked against the core-workflow alpha gate in `roadmap.md`: passes |
| 2026-10-10 | major | Full re-measure against Photoshop 2026 27.11.0 (installed bundle inspected); two numbers, weights, per-dimension and per-area hours calibrated from git history; merged `parity-estimate.md` and the roadmap's honest assessment into this file |
| 2026-10-05 | major | Honest parity assessment in `docs/roadmap.md`: surface 60–70%, ready for real work 25–35% |
| 2026-10-03 | major | `docs/parity-estimate.md`: menu breadth 625/625, weighted ~82–86% |
