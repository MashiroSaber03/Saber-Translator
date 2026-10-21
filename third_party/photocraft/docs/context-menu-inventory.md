# Context-menu inventory and parity contract

> **Last reviewed:** 2026-10-10 · **Last updated:** 2026-10-10 · **Change:** trivial (status line added; the inventory is the 2026-10-06 branch-point snapshot, current status lives in context-menu-parity.md) · **Target:** Adobe Photoshop 2026

This is the source-code inventory of PhotoCraft's right-click behavior at the branch point, captured 2026-10-06. It distinguishes a menu from another secondary-button gesture. The initial selection, Paths row, and document-tab menus added by this PR are called out separately. Photoshop comparison should be checked against an identified Photoshop version and platform before claiming exact parity; the current Layers menu explicitly targets Photoshop 2026 order.

## Canvas: every tool

The shared rule is **Command/Ctrl + right-click** on opaque document pixels → a topmost-first list of visible layers at that point; choosing a row dispatches `layer.select`. The Move tool gets that menu on plain right-click too. The list comes from `layer.pickAt` and is omitted over fully transparent pixels. On the remaining tools, plain right-click depends on the tool group below. The canvas behavior lives in `canvas.rs`, `paint_mouse.rs`, `layer_pick_ui.rs`, and `brush_resize.rs`.

| Tools | Plain right-click today | Other secondary gesture | Parity work |
|---|---|---|---|
| Move | Layer-under-pointer list (`layer.pickAt`, `layer.select`) | Command/Ctrl gives the same list | Verify Photoshop modifier and auto-select behavior; test overlaps, masks, groups, transformed pixels, empty areas. |
| Brush, Pencil, Eraser, Background Eraser | Brush Preset picker (size, hardness, presets) | Alt/Option + right-drag changes size/hardness via `tools.setBrush`; Brush/Eraser can right-drag erase via `tools.rightClickWithPaintingTools=erase` | Verify whether context should expose brush presets, settings, or tool-specific eraser controls. Keep erase preference and modifier priority deterministic. |
| Spot Healing, Healing, Clone Stamp, History Brush, Blur, Sharpen, Smudge, Dodge, Burn, Sponge | Brush Preset picker | Alt/Option + right-drag changes size/hardness | Compare each tool's Photoshop picker and source/retouch options; use existing brush command route. |
| Quick Selection | Brush Preset picker | Alt/Option + right-drag changes size/hardness | Check selection-tool brush options and whether mode controls belong in picker. |
| Rectangular/Elliptical Marquee, Lasso, Polygonal Lasso, Magnetic Lasso, Magic Wand, Object Selection | Selection context menu: the full Photoshop list with a selection, Select All / Reselect / Color Range / Load Selection without (`canvas_tool_menu.rs`, rows in `docs/context-menu-parity.md`) | Command/Ctrl opens layer list | Confirm the no-selection variant against a dated Photoshop capture. |
| Crop, Slice, Slice Select | No plain secondary action | Command/Ctrl opens layer list | Define crop/straighten commit and cancel, aspect/preset, slice selection/edit behavior separately. |
| Eyedropper, Ruler, Note, Count | No plain secondary action | Command/Ctrl opens layer list | Check tool-specific sampling, measurement, annotation, and count menus. |
| Gradient, Paint Bucket | No plain secondary action | Command/Ctrl opens layer list | Expose gradient preset and fill settings through commands if Photoshop's context UI provides them; avoid painting on right-click. |
| Type | No plain secondary action | Command/Ctrl opens layer list | Test text edit versus non-edit context, spelling, type options, and selection behavior. |
| Pen, Path Selection | No plain secondary action | Command/Ctrl opens layer list | High-priority path-point context contract: add/delete/convert point, close path, make selection, fill/stroke path, path operations when applicable. |
| Rectangle, Ellipse Shape, Triangle, Polygon, Line, Custom Shape | No plain secondary action | Command/Ctrl opens layer list | Define shape versus path context, point edit, fill/stroke, and transform entries. |
| Hand, Zoom | View menu: Zoom In, Zoom Out, Fit on Screen, Fill Screen, 100%, 200%, Print Size (`canvas_tool_menu.rs`, #3242) | Command/Ctrl opens layer list | Confirm the row order against a dated Photoshop capture. |

Canvas invariant: secondary click must not accidentally start a paint stroke, commit a crop, or edit the document; every mutating menu action dispatches a command. Escape/outside click closes a popup. Existing `ui.pointer` supports a secondary button; UI tests should exercise both native and control-driven gestures.

## Other right-click surfaces

| Surface | Current entries or behavior | Command route / gap |
|---|---|---|
| Layers panel row | Blending Options; Duplicate/Delete; Group from Layers; Quick Export PNG / Export As; Artboard/Frame from Layers; Convert to Smart Object; content-specific Rasterize; Add/Disable/Apply/Delete Layer Mask; Create/Release Clipping Mask; Link/Select Linked; Copy/Paste/Clear Layer Style; Merge Layers/Down, Merge Visible, Flatten Image; Rename Layer. Multi-select changes duplicate/delete/merge labels and target. | `layer_menu_ui::entries` supplies IDs for all except inline rename, then `menus::invoke` or engine dispatch. Missing or disabled commands are filtered/greyed. Check menu visibility and enabled state for groups, backgrounds, masks, type, adjustment/fill, smart objects, and multi-selection. |
| Channels panel row | Alpha: Duplicate/Delete/Rename, spot conversion or merge, selection-display and overlay controls. Quick Mask: Exit, display and overlay. Composite/color: Duplicate. Layer mask: Enable/Disable, Delete. Common: New Channel, New Spot Channel, Split, Merge. | `channel.*`, `layer.layerMask.*`, `select.editInQuickMaskMode`, `ui.renameChannel`. Check selection loading/saving, channel options, and per-row enablement against Photoshop. |
| Brushes panel preset | Rename Brush, Delete Brush. | Routed through `brush.presets.*` panel actions. Check duplicate/export/import and library organization. |
| Brushes panel group | Rename Group, Delete Group. | Routed through preset panel actions; confirm group deletion consequences. |
| Generic presets group (gradients, patterns, swatches, etc.) | Rename Group, Delete Group. | `preset_panels.rs`; verify each collection's appropriate create, export/import, and delete semantics. |
| Generic preset item | Rename, Delete, Move to another group. | `preset_panels.rs`; behavior is generic across preset collections. |
| Tool Presets item | Rename Tool Preset, Delete Tool Preset. | `tool.presets.edit`; compare current-tool filtering and preset application. |
| Toolbar grouped tool slot | Right-click or long-press opens its tool flyout. | Sets active `Tool` UI state. Single-tool slots have no flyout. |
| Toolbar Screen Mode | Standard, Full Screen With Menu Bar, Full Screen. | `view.screenMode.*` via `menus::invoke`. |
| Color chip in color panel | Right-click sets background color (left-click sets foreground). | Direct UI state behavior; verify foreground/background target commands. |
| Wide Angle constraint | Right-click deletes that constraint. | Tool-local action; no menu. |
| Camera Raw point curve | Right-click deletes a curve point. | Tool-local action; no menu. |
| Document tab / loading tab | No context menu at branch point. This PR adds Close, Close Others, Close All for document tabs. Left click activates; close icon closes/cancels. | Existing guarded File commands handle unsaved documents; reveal file and tab layout remain open. |
| Paths panel row | Make Selection, Fill/Stroke Path, Save Work Path or Duplicate Path, Delete Path or Vector Mask as applicable (added in this PR). | `path.*` and `layer.vectorMask.delete`; direct action on the clicked path. |
| History, Properties and empty panel areas | No `context_menu` handler found in UI source at this audit. | Audit each panel's Photoshop menu and prioritize its useful commands; do not invent dead entries. |

## Contract for implementation

1. Record a Photoshop reference matrix by version, operating system, surface, tool and state (empty/selection/path/text edit/multi-layer). Capture exact item labels, order, separators, submenus, disabled conditions, and modifier variants. Mark observations versus inference.
2. Encode menu descriptors as data with predicate tests. Route document edits through existing or new engine command IDs; view-only actions can stay in UI. A menu on the canvas must receive the document coordinate and relevant target from hit testing, so actions operate on what was clicked rather than silently on an unrelated active object.
3. Prioritize shared canvas menus, then high-value tool-specific interactions (path, selection, type, crop/zoom), then panel and tab gaps. Reuse Layers/Channels implementations where their command contract already exists. Preserve the existing Move/layer pick, brush picker, erase preference, Alt/Option brush resize, and toolbar flyout gestures.
4. Test each descriptor's item order and enablement, dispatch with expected params, empty/invalid states, modifiers, and action result. Add UI gesture tests for right-click opening, target selection, Escape/outside dismissal, and the absence of accidental edits. Validate native and web builds.
5. Capture screenshots from the running app for representative tool groups and panel states, with test fixture documents rather than personal photos. A screenshot proves visible menu composition; command tests prove its behavior.

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-10 | trivial | Status line, revision history; marked as the branch-point snapshot |
| 2026-10-06 | major | Source inventory of right-click behaviour |
