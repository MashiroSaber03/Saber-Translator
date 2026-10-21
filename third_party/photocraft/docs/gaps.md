# Where PhotoCraft falls short of Photoshop

> **Last reviewed:** 2026-10-10 · **Last updated:** 2026-10-13 · **Change:** minor (G7: Field Blur's on-canvas pins, #3373; G4: Hue Jitter amount track, #3825 partial; G4: minimum field interaction, #3825 partial; G1: CMYK Invert, Posterize and Threshold on the inks, 16-bit Lab Invert; Photoshop-oracle floor 145/258; G11: `.atn` import maps 112 events: Select and Mask, Filter Gallery Spatter, Ripple, Crystallize, Pattern Fill layers, mask targeting, Step Backward, nudges in points, Bitmap mode; G5: Free Transform H/V skew fields; G9: toolbar tool shortcut editing; refreshed against main: G2, G5, G7, G9, G10, G13, G14, G17 items that landed; G10: DPX read and write, #1049; G4: checkbox-only Brush Settings rows; G11: Load/Save Actions `.atn` landed, AUTO-219-2 partial; G5: hold-a-key spring-loaded tools landed, #2672; G1: #2465 fixed; stale missing-tool rows corrected: Color Replacement, Red Eye and Add/Delete Anchor Point are live tools, so only Art History Brush is missing; no estimate changed) · **Target:** Adobe Photoshop 2026

Every known shortfall against Photoshop 2026 (27.11.0), one entry each, ranked by user impact
on the way to beta. This is the work list agents pick from: take the highest-ranked entry that
nobody is working on, and when you close part of one, update its evidence and estimate here and
the matching parity doc. Numbers come from [`target-app-parity.md`](target-app-parity.md);
checklist ids (`TOOL-…`, `FILE-…`, `UI-…`) are rows in [`scorecard.md`](scorecard.md); `#n` are
GitHub issues.

No gap blocks alpha: the [alpha gate](roadmap.md#alpha-gate) passes. "Blocks beta" marks what
stands between alpha and beta.

Hours are Opus 5.5 agent-hours, one agent working sequentially, including tests. **Human** marks
what an agent can't finish alone.

## Summary

| # | Gap | Kind | Impact | Estimate | Parity doc |
|---|---|---|---|---:|---|
| G1 | PSD fidelity and Photoshop acceptance | file format | blocks beta | 150–250 h | [file-format-parity](file-format-parity.md) |
| G2 | Interactive performance | performance | blocks beta | 200–350 h | [scorecard](scorecard.md#performance) |
| G3 | User-reported bug backlog | stability, UI | blocks beta | 250–400 h | [target-app-parity](target-app-parity.md) |
| G4 | Brush feel | UI / painting | high | 100–180 h | [brush-parity](brush-parity.md) |
| G5 | Transform, handles and modifiers | UI | high | 60–110 h | [ui-parity](ui-parity.md) |
| G6 | Missing tools (12) | features | high | 60–100 h | [ui-parity](ui-parity.md#tools) |
| G7 | Workspaces shown as parameter dialogs | features | high | 80–140 h | [target-app-parity](target-app-parity.md#feature-areas) |
| G8 | Type depth | features | high | 120–200 h | [target-app-parity](target-app-parity.md#feature-areas) |
| G9 | Panels, preferences and workspace | UI | medium-high | 150–250 h | [ui-parity](ui-parity.md#panels-and-workspace) |
| G10 | Missing and partial file formats | file format | medium | 120–220 h | [file-format-parity](file-format-parity.md) |
| G11 | Actions and scripting | ecosystem | medium | 120–200 h | [target-app-parity](target-app-parity.md) |
| G12 | Pen tablets and display hardware | hardware | medium | 80–150 h | [hardware-parity](hardware-parity.md) |
| G13 | Languages: Hindi, Arabic, Vietnamese, RTL, engine messages | localization | medium | 80–140 h + human | [localization-parity](localization-parity.md) |
| G14 | Camera Raw depth | features, formats | medium | 120–200 h | [file-format-parity](file-format-parity.md#camera-raw) |
| G15 | AI and generative features | AI | medium (high for some users) | 300–600 h + human | [target-app-parity](target-app-parity.md) |
| G16 | Plug-in compatibility (.8BF, UXP) | ecosystem | medium | 250–450 h + human | [target-app-parity](target-app-parity.md) |
| G17 | Platforms and distribution | platforms | low-medium | 30–60 h + human | [scorecard](scorecard.md#distribution) |

## Feature gaps

### G1. PSD fidelity and Photoshop acceptance
- **Missing:** PSDs we write are not always accepted by Photoshop, and Photoshop-authored files
  don't always render as Photoshop renders them.
- **Evidence:** corpus floors (`crates/io/tests/corpus.rs`, scorecard): Photoshop-authored
  oracles 145/258 (56%; CMYK Invert, Posterize and Threshold on the inks 2026-10-12), io corpus 146/170 (86%), psd-tools 236/309 (76%); round trips ~100%
  within the corpus. Smart-filter re-render matches 5/30. User reports: a PSD with text layers
  re-saved unmodified would not reopen (#2374, closed 2026-10-10), Photoshop 26.6 cannot reopen our files (#1281), "PSD not
  opening in Photoshop" (#2469), a CS2-era smart object unavailable (#2465, fixed 2026-10-12), PSB smart object
  blank (#1764), perspective placed layers flattened (#579, fixed by #426), PSB > 4 GB fails on Windows (#375).
  No test opens our exports in a third-party editor (FILE-216-4).
- **Impact:** PSD is the format professionals exchange. Beta requires it to be reliable.
- **Estimate:** 150–250 h. **Human:** opening exports in Photoshop on the owner's Mac (or a
  scripted check there) is the only real oracle.

### G6. Missing tools
- **Missing (1 of 68):** Art History Brush. Convert Point landed (#3204). Freeform
  Pen and Curvature Pen landed (#1054); Color Sampler landed (#1046, `view.colorSamplers.*`); Horizontal and Vertical Type Mask landed (#1055,
  `type.createSelection`); Perspective Crop landed (#1043, `image.perspectiveCrop`); Frame landed
  (#2914).
- **Evidence:** `Tool` enum (`crates/ui-egui/src/state.rs`) against the 67 `<tool>` entries of
  Photoshop's `Default Keyboard Shortcuts.kys` plus Remove, Triangle and Curvature Pen.
- **Impact:** users look for them by name and shortcut; several are daily tools for designers.
- **Estimate:** 60–100 h (2–9 h each, per the calibration).

### G7. Workspaces shown as parameter dialogs
- **Missing:** Select and Mask workspace layout
  (TOOL-214-1, Refine Hair #2671; its view modes preview live since #1754, Refine Edge Brush #2972); Content-Aware Fill workspace with a paintable sampling area
  (TOOL-214-15); Vanishing Point plane editor (TOOL-214-14); Content-Aware Move's live preview
  of the result (TOOL-213-4; Transform On Drop and Content-Aware Scale through the transform box,
  TOOL-214-7, are done).
  Blur Gallery: Field Blur has its on-canvas pins (2026-10-11, #3373); Iris, Tilt-Shift, Spin and
  Path Blur still edit their pins as parameter dialogs.
- **Evidence:** scorecard rows above, all partial or missing.
- **Impact:** these are the tools retouchers and compositors use daily; a slider dialog is not a
  substitute.
- **Estimate:** 80–140 h.

### G8. Type depth
- **Missing:** type on a path (TYPE-218-1), paragraph direction UI (TYPE-218-2), complex-script
  fallback fonts and golden tests (TYPE-218-3), OpenType Numerals and Stylistic Sets submenus
  (TYPE-218-5; the Character panel buttons landed 2026-10-11, #3752), live Warp Text preview
  (TYPE-218-7, #2689), "rasterize and continue" for distort/perspective on type
  (TYPE-218-9), text free transform (#2630). Done since the re-measure: manual and optical kerning
  with PSD round trip (TYPE-218-6, #206), variable-font axes (TYPE-218-4, #2985).
- **Evidence:** scorecard Type: 3 done, 2 partial, 4 missing. 2026-10-11: 2/3/4/5 clicks select
  word, line, paragraph and all text, with tests (TYPE-218-8, #3493). The Type tool froze the app for
  60 s on one macOS machine (#2518).
- **Estimate:** 120–200 h.

### G11. Actions and scripting
- **2026-10-13 evidence:** Batch supports [output naming templates](batch-naming.md) using
  `{name}`, `{serial}` and a captured UTC `{date}`, with a starting serial number. Colliding
  names remain errors; custom output names cannot replace inputs. AUTO-219-5 stays partial.
- **Missing:** document-independent action recording (AUTO-219-1), Batch recursion, open-document inputs, progress/cancel and an error-log file
  (AUTO-219-5), ExtendScript/UXP compatibility. `.atn` steps outside the
  mapped events (paths, scripts, Generative Fill, Filter Gallery effects other than Spatter,
  Duplicate Layers into a new document, Bitmap halftone screens…) load as kept Photoshop steps
  that fail at playback by name.
- **Done 2026-10-12:** Load/Save Actions `.atn` in the panel menu, `batch --actions set.atn`
  and `actions.load`/`actions.save`; 112 Photoshop events map to commands both ways (Transform,
  Apply Image, Save for Web, smart-object edits, Gradient Map, Quick Mask, polygon selections,
  Place, Open, Image Size in percent, the Paint Bucket and Insert Conditional since the third
  and fourth PRs; Select and Mask, Filter Gallery Spatter, Ripple, Crystallize, Pattern Fill
  layers, targeting the layer mask or composite channel, loading the targeted channel, Step
  Backward/Forward, removing a layer from the selection, nudges in points, Layer From Background,
  Grayscale and Bitmap mode objects and Image Size with Resample keeping the print size since the
  fifth), Play, Stop and conditional steps included; layer steps replay by layer name in any
  document. Four TrueGrit Texture Supply sets (Debaser, Distress Press, Halftone Zine Machine,
  Transparency) import with 3 of 533 steps unmapped and play end to end. Filter Gallery Spatter's Smoothness
  now smooths the spray's displacement, so high values tear an edge without detaching specks, as
  Photoshop's does (calibrated by eye on the Distress Press guide's screenshots). Crystallize
  fills each cell with the colour at its point instead of the cell's average, so a black-and-white
  mask keeps hard edges (not yet checked against a Photoshop render). The panel groups actions
  in set folders, has the dialog toggle column (playback pauses on the step's dialog and resumes
  on OK) and Insert Stop…; `actions.insert`, `actions.dialog`, `actions.assignSet` for agents
  (AUTO-219-2 done).
- **Ahead of Photoshop:** CLI, MCP and the JSON control channel drive every command.
- **Estimate:** 120–200 h. **Human:** whether to support a JavaScript scripting dialect at all
  (AGENTS.md forbids JS in the app).

### G14. Camera Raw depth
- **Missing:** CR3 (detected, unsupported), compressed RAF, lossy DNG (FILE-215-11), .xmp
  sidecars, masking and calibration in the Camera Raw dialog, per-camera colour for non-DNG
  cameras, lens profiles.
- **Evidence:** `crates/raw/src/lib.rs:228` (CR3), scorecard FILE-215-11/12, #50.
- **Progress (2026-10-11):** on the embedded-JPEG rendering work in #2135, Camera Raw
  tuning retains the fitted look and applies user gains afterwards. Three D5100 NEFs
  establish a guarded 4928 × 3264 image area. This is an estimated rendering, not a
  measured D5100 colour calibration; [validation](nikon-nef-development.md).
- **Estimate:** 120–200 h.

### G15. AI and generative features
- **Missing:** Generative Fill and Expand, Generative Upscale, Neural Filters, Sky Replacement,
  Super Zoom, ML Select Subject/Sky. There is no ML runtime (AUTO-219-7). Classical substitutes
  exist (GrabCut/max-flow selection, Remove tool by non-local patch completion).
- **Evidence:** Photoshop's `Required/UXP` ships the Neural Filters gallery and its `.kys` has
  a Neural Filters taskspace; users ask for local models (#2313, #1996, #2682).
- **Estimate:** 300–600 h. **Human:** owner decision #41 (model licences, local vs remote).

### G16. Plug-in compatibility
- **Missing:** hosting Photoshop .8BF filters (#2647), UXP panels, CEP. PhotoCraft has its own
  sandboxed WebAssembly plug-in ABI v1 (`docs/plugins.md`), but ABI v2 (panels, progress, other
  layers) is missing (AUTO-219-6).
- **Estimate:** 250–450 h. **Human:** .8BF needs native code under the isolated-`unsafe` rule.

## UI / UX gaps

### G3. User-reported bug backlog
- **Missing:** fixes for 579 open issues (2026-10-10), filed mostly by 0.5/0.6 users. By title:
  ~150 UI/panels/menus/shortcuts, 94 file/format, 87 layers, 54 painting, 54 selection,
  46 type, 41 performance, 34 transform/move, 25 Linux/Wayland, 24 Windows, 20 macOS.
- **Impact:** each is a place where a Photoshop user hits something different from what they
  expect; together they are why ready-for-real-work is ~45% while breadth is ~76%.
- **Estimate:** 250–400 h (many overlap with other gaps; count each fix once).

### G4. Brush feel
- **Progress (2026-10-13, #3825 partial):** Hue Jitter shows a standard themed
  percentage-amount track rather than a hue spectrum. Other Brush Settings layout
  and brush-behaviour gaps remain open.
- **Progress (2026-10-13, #3825 partial):** Shape Dynamics' Minimum Diameter / Minimum
  Roundness and Transfer's four Minimum rows disable editing when their Control is Off,
  retaining the configured values. The other layout and control gaps in #3825 remain open.
- **Progress (2026-10-12, #4283):** the five checkbox-only Brush Settings rows keep the open
  controls selected; Smoothing can be disabled without discarding its amount or mode. This
  closes that panel interaction gap, not the stroke-quality or performance gaps below.
- **Missing:** lag with big brushes (#2619, #1994) and under the cursor (#2566, #1343); jumpy
  pressure on Windows (#2046) and tap sensitivity (#1933); stair-stepped line weights (#1776),
  interpolation and smoothing problems (#2138), edge falloff (#2149), pixelated cursor vs
  Photoshop (#1433), brush rotation not driven by pen rotation (#1301); live preview only for
  Brush and Eraser (TOOL-213-7); no image file as a brush tip (TOOL-213-12); per-control pressure
  curves (TOOL-213-11); 4 of Photoshop's symmetry modes.
- **Detail:** [`brush-parity.md`](brush-parity.md).
- **Estimate:** 100–180 h. **Human:** checking feel on real tablets.

### G5. Transform, handles and modifiers
- **Missing:** the full interpolation list; blend mode/opacity edits during
  a transform (#1384); path-point nudge; autoscroll at the edge while selecting (#2231);
  wheel and pinch behaviour (#2485, #2463, #1989);
  toolbar press-and-hold flyouts (#2301).
- **Done since the re-measure (2026-10-10 to 12):** right-click menu during Free Transform
  (TOOL-214-5); commit/cancel buttons in the options bar (#2442); transform keeps the layer's
  place in the stack (#2439, #2480); Hand tool no longer interrupts a transform (#2302, #2327);
  locked layers refused before the drag (#2585, #2609); ⌘-drag temporary Move with box and
  snapping (#2681, #2684); Auto-Select box follows the picked layer (#2666, #2675); ⌘-arrow
  layer nudge (#2474, #2556); polygonal lasso Shift angles (#2352, #2396); double-click Zoom tool
  to 100% (#2508, #2527); tool-key memory in a group (#2608, #2625); hold-a-key spring-loaded
  tools (#2672); H/V skew fields in the Free Transform options bar.
- **Detail:** [`ui-parity.md`](ui-parity.md).
- **Estimate:** 60–110 h.

### G9. Panels, preferences and workspace
- **Progress (2026-10-12):** toolbar tool shortcuts can be reassigned, removed and reset (#4086),
  with saved bindings, matching toolbar hints and regression coverage (`ui-egui/src/tool_shortcuts.rs`).
- **Missing:** 27 of 156 preferences do nothing (scorecard prefs audit); panel groups can't
  float or tear off into OS windows (UI-217-5, #1828, #1213, #2305); Color panel lacks wheel and
  slider modes (UI-217-17, #2569); more themes (UI-217-11, #2661); layer clipboard
  (UI-217-23, #2670); tool-specific context menus beyond brushes, selections and Pen
  ([context-menu-parity.md](context-menu-parity.md)); Libraries, Comments, Contextual Task Bar
  panels.
- **Estimate:** 150–250 h.

### G2. Interactive performance
- **Measured progress (2026-10-10):** Tilt-Shift's existing per-level halo/copy
  optimization measures 1.07–1.49× faster across nine local 24 MP cases, with exact
  finite output bits. The filtered proxy improves for medium/large blur; its
  default small blur and the 36 MP saturated-field gain are inconclusive.
  [Method and full ranges](tilt-shift-performance.md). Native GUI latency remains
  unmeasured; no scorecard budget or parity percentage is changed by this work.
- **Missing:** 22 of 25 budgeted scenarios are over budget (baseline 2026-10-05, a heavily
  loaded machine): layer move with styles 321 ms p50 against 25 ms, opacity change 276 ms
  against 20 ms, paste 12 MP 1.2 s against 100 ms, layer move on 15000×10000 16-bit 2.7 s,
  Content-Aware Scale 24 MP 954 s against 3 s, Select Subject 559 ms against 100 ms; the
  150-layer nudge crashes the GPU compositor (P12); menus slow with thousands of layers (#2460); old machines lag (#2687).
- **Estimate:** 200–350 h. Serial in places (compositor architecture).

## File-format gaps

### G10. Missing and partial file formats
- **Missing:** JPEG Options dialog (FILE-215-19), Photoshop PDF read and write and multi-page PDF
  (FILE-215-10, #2541), AVIF read (#463) and default-build AVIF write, JPEG XL, JPEG 2000 (#1047),
  Cineon (DPX landed, #1049), DICOM, EPS/DCS, PCX, Pixar, IFF, WBMP, MPO/JPS, PICT, Scitex CT, Photoshop
  Raw; layered TIFF extra alpha channels (FILE-215-7); spot plates (FILE-215-17); .atn, .asl,
  .shc, .csh presets (FILE-215-15); C2PA (#1050).
- **Detail:** [`file-format-parity.md`](file-format-parity.md).
- **Estimate:** 120–220 h.

## Hardware gaps

### G12. Pen tablets and display hardware
- **Missing:** tilt and rotation on Windows (pressure only, via WM_POINTER; no WinTab); native
  Wayland tablet-v2 (#79; the pen does not click on GNOME 50, #2622, KDE #2297); eraser end on
  flipped Wacom pens (#2285); HDR/EDR output (Photoshop has "Precise Color Management for HDR
  Display"); monitor profile detection off macOS and live profile changes; Touch Bar (macOS
  Intel only, low value); 3D mice.
- **Detail:** [`hardware-parity.md`](hardware-parity.md).
- **Estimate:** 80–150 h. **Human:** tablets and HDR displays to test on.

## Localization gaps

### G13. Hindi, Arabic, Vietnamese, RTL and engine messages
- **Missing:** catalogs for Hindi, Arabic and Vietnamese (3 of the 12 key languages); right-to-
  left UI layout; Devanagari fallback fonts (Thai #1909 and Khmer, Lao and Myanmar #3259 landed); engine error and status messages in
  English in every language; browser-locale detection and CJK fonts on the web; Japanese IME on
  Windows (#590); installed Japanese fonts missing from the font list (#2330); Arabic layer names
  as tofu (#1408, #1693).
- **Evidence (2026-10-11):** Czech dynamic UI wrappers, hardware labels and composed messages
  translate at display sites; engine errors remain English. This does not close G13.
- **Detail:** [`localization-parity.md`](localization-parity.md).
- **Estimate:** 80–140 h. **Human:** native-speaker review of every catalog (none is recorded).

## Platform gaps

### G17. Platforms and distribution
- **Missing:** Windows code signing (material not obtained), in-app update check (DIST-220-1,
  #2650, #2375), in-app bug report (DIST-220-3), Flathub (#173),
  starting without a GPU adapter (#2519, #2412), Windows 7 (#2310, out of scope), Intel Macs with
  OpenCore (#1859).
- **Estimate:** 30–60 h. **Human:** signing certificates.

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-13 | minor | G6: stale "missing tool" list corrected against `main` — Color Replacement (`paint.colorReplacement`), Red Eye (`paint.redEye`) and Add/Delete Anchor Point (P flyout, #4144) are live tools, so the toolbox is 67 of 68 and Art History Brush is the only one missing; no percentage or estimate changed |
| 2026-10-13 | minor | G4: Hue Jitter amount-track appearance (#3825, partial); no readiness percentage or estimate changed |
| 2026-10-13 | minor | G4: minimum inputs in Shape Dynamics and Transfer follow their Control (#3825, partial); no readiness percentage or estimate changed |
| 2026-10-12 | minor | G5: Free Transform options bar H/V skew fields done |
| 2026-10-12 | minor | G11: `.atn` import maps 112 events (Select and Mask, Spatter, Ripple, Crystallize, Pattern Fill, mask targeting, Step Backward, points, Bitmap) |
| 2026-10-12 | minor | G9: toolbar tool shortcut editing (#4086); the remaining workspace gaps stay open |
| 2026-10-12 | minor | G1: #2465 (CS2 `PlLd`-only smart objects) marked fixed |
| 2026-10-12 | minor | G1: CMYK Invert, Posterize and Threshold work on the inks, 16-bit Lab Invert keeps greys neutral; Photoshop-oracle floor 133 → 145 of 258 (145 already passed on main) |
| 2026-10-12 | minor | G5: spring-loaded tool letters done (#2672) |
| 2026-10-12 | minor | G10: DPX read and write (#1049); Cineon remains open |
| 2026-10-12 | minor | G4: checkbox-only Brush Settings rows preserve the open controls and Smoothing settings (#4283); no readiness percentage or estimate changed |
| 2026-10-12 | minor | G11: Load/Save Actions `.atn` landed (AUTO-219-2 partial); remaining panel work listed |
| 2026-10-12 | minor | Refreshed against main: G5 items fixed by #2480, #2327, #2609, #2684, #2675, #2556, #2396, #2527, #2625 and TOOL-214-5/#2442; G7 TOOL-214-7 and Transform On Drop done; G2 #2377 fixed (#2417); G9 UI scale (UI-217-9) and Stamp Visible (UI-217-22) done, prefs 27 of 156; G10 #2225 done (#2482); G13 Khmer, Lao, Myanmar fallback (#3259); G14 CR3 line; G17 Windows ARM64 done (DIST-220-2) |
| 2026-10-11 | minor | G7: Field Blur pins are placed, dragged and tuned on the image (#3373); the other Blur Gallery filters remain dialogs |
| 2026-10-11 | minor | D5100 image area and Camera Raw tuning validation; G14 remains open |
| 2026-10-11 | minor | G13: Czech dynamic UI text localized; engine messages and native review remain open |
| 2026-10-11 | minor | G8: triple/quadruple/five-click type selection done (TYPE-218-8, #3493); Type evidence recounted |
| 2026-10-11 | minor | G8: Character panel OpenType buttons landed (TYPE-218-5, #3752); Numerals and Stylistic Sets submenus still missing |
| 2026-10-10 | minor | G2: recorded Tilt-Shift computation/filtered-proxy measurements and limits; gap and estimate remain open |
| 2026-10-10 | minor | Checked against the alpha gate (no alpha blockers); #2374 marked closed |
| 2026-10-10 | major | First version, from the full re-measure against Photoshop 2026 27.11.0 and the open issues |
