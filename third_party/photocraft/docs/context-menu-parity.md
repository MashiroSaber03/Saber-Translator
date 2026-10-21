# Context-menu parity contract

> **Last reviewed:** 2026-10-13 · **Last updated:** 2026-10-13 · **Change:** minor (layer-mask thumbnail menu, #4575) · **Target:** Adobe Photoshop 2026

Status: implementation contract, updated 2026-10-09. Scope: canvas, document tabs, and panels. “Right-click” means a secondary pointer click; macOS Control-click should also open the same menu where the platform sends a context-menu gesture. Photoshop changes a context menu with the active tool, selection, and panel item. Adobe documents that principle but does not publish a complete current menu catalog. **Documented** means an Adobe source explicitly names the gesture or action. **Inferred** means a PhotoCraft design target or an item known from a main menu whose presence in a Photoshop context menu still needs a current-product capture. Do not present inferred rows as exact Photoshop menus. [Adobe: panels and menus](https://helpx.adobe.com/photoshop/using/panels-menus.html), [Adobe: shortcuts and context menus](https://helpx.adobe.com/sa_en/photoshop/using/customizing-keyboard-shortcuts.html).

## Dispatch contract

1. Hit-test the target before opening: modal operation/handle, text selection, canvas content, panel row or icon, tab, then empty workspace. A generic canvas menu must not swallow a more specific target.
2. Snapshot the target and selection when the menu opens. A row in an existing multiselection acts on that set; a row outside it becomes the target before its command runs. Apply the same rule to layers, channels, paths, and presets.
3. Resolve items from a pure menu model keyed by target kind, active tool, document state, platform, and active operation. The model supplies label, command id, params, enabled state, and reason. Every mutating item dispatches an engine command; view-only items may update shell state. An unavailable feature is disabled with a reason or omitted, never a clickable no-op.
4. Preserve primary-button tool behavior. Secondary click only opens a menu unless a right-button drag gesture or the user's explicit right-erase preference owns the gesture. A drag must not briefly open a menu, and opening a menu must never paint, transform, sample, or commit.
5. Anchor at the pointer and clamp to the viewport; support keyboard context-menu key/Shift+F10 where available, Escape/outside click, separators only between nonempty groups, checked/radio state, keyboard navigation, and screen-reader labels.
6. Use the same command id and enablement as the menu bar/options bar. Provide UI inspection data and control-channel operations to select a menu item without pixel coordinates, while retaining real pointer tests and screenshots.
7. Test each menu model independently for conditional rows and wrong-state safety; test actual secondary-click routing, menu selection, undo/redo, and no accidental primary action.

## Canvas tool matrix

The “PhotoCraft at branch point” column records behavior before this PR. Picker means the Brush Preset quick picker in paint_mouse.rs; Layers means the hit-tested layer list in layer_pick_ui.rs. Command/Ctrl + secondary-click opens the layer list with every tool, and Move opens it without a modifier. Exact Photoshop secondary-click items beyond Adobe-documented rows are **proposed** until captured from current Photoshop.

| Tool(s) in PhotoCraft | Photoshop / desired contextual action | PhotoCraft at branch point | Priority, evidence |
|---|---|---|---|
| Move | Choose a layer at the point; an auto-select choice also exists in the options bar | Layers | P0; [Adobe layer movement](https://helpx.adobe.com/photoshop/using/moving-stacking-locking-layers.html); exact pop-up layout requires capture |
| Rectangular/Elliptical Marquee, Lasso, Polygonal Lasso, Magnetic Lasso, Magic Wand, Quick Selection, Object Selection | Active selection: the 2026 selection menu (see *Selection canvas reference* below); no selection: making a selection | Quick Selection: Picker (a brush tool, as in Photoshop); the other six: the selection menu below (#614, #738) | P1; no-selection variant still needs a dated capture |
| Crop | During crop: ratio/preset, overlay, clear/reset, commit/cancel **(inferred context placement)** | No tool menu | P1; pending crop target |
| Eyedropper | Sample-size/source choices **(inferred)** | No tool menu | P2; [Adobe illustration uses Eyedropper](https://helpx.adobe.com/photoshop/using/panels-menus.html), but does not publish item text |
| Ruler, Note, Count | Unit / measurement, note edit/delete, count-group operations **(inferred)** | No tool menu | P2 |
| Brush, Pencil, Eraser, Background Eraser, Spot Healing, Healing, Clone Stamp, History Brush, Blur, Sharpen, Smudge, Dodge, Burn, Sponge | Brush preset, size and hardness where applicable | Picker | P0; [Adobe Brush Settings](https://helpx.adobe.com/photoshop/desktop/apply-painting-techniques/brushes-presets/create-brush-set-painting-options.html). Alt+secondary-drag on Windows and Control+Option-drag on macOS adjust size/hardness ([Adobe cursor shortcuts](https://helpx.adobe.com/in/photoshop/desktop/get-started/settings-and-preferences/change-tool-pointers.html)) |
| Gradient, Paint Bucket | Gradient preset/type/mode or fill source/tolerance **(inferred context placement)** | No tool menu | P1 |
| Type | In text: cut/copy/paste, spelling suggestions, text formatting; outside text: type-layer action **(inferred context placement)** | While editing text (canvas_tool_menu.rs `TYPE_EDIT_MENU`): Undo, Cut, Copy, Paste, Select All (on the text), Check Spelling…, Find and Replace Text…, Faux Bold/Italic (checked), Rasterize Type, Create Work Path, Convert to Shape, Horizontal/Vertical and the anti-alias modes (checked), Convert to Paragraph/Point Text, Warp Text…; Esc closes only the menu. Over a misspelled word the menu starts with up to 3 spelling suggestions (replace the word) and Add to Dictionary (#4370); Ignore / Ignore All and the non-editing type-layer menu remain open | P1; separate editing and nonediting states; [Adobe type editing](https://helpx.adobe.com/photoshop/desktop/text-typography/get-started-with-text/edit-text.html) |
| Hand, Zoom | Fit on Screen, 100%, 200%, Print Size; zoom step choices **(inferred context placement)** | View menu (#3242): Zoom In, Zoom Out, Fit on Screen, Fill Screen, 100%, 200%, Print Size. Zoom In/Out step about the right-clicked point (Preferences › Tools › Zoom Clicked Point to Center centres it). Outside the image the pasteboard menu still wins. | P1; view actions stay out of document history; row order still needs a dated Photoshop capture |
| Pen, Path Selection | Path/point actions, Make Selection, Fill/Stroke Path, Free Transform Path **(inferred context placement)** | Pen canvas right-click follows the user-provided 20-row Photoshop capture below. Paths rows offer Make Selection, Fill and Stroke. Path Selection's own canvas menu remains open. | P1; distinguish node, segment, path, and blank canvas |
| Rectangle, Ellipse Shape, Triangle, Polygon, Line, Custom Shape | Shape/path operations, fill/stroke settings, transform **(inferred context placement)** | No tool menu | P2 |
| Slice, Slice Select | Slice properties, duplicate/delete, divide, guides **(inferred context placement)** | No tool menu | P2 |
| Any tool while Free Transform is active | Scale, Rotate, Skew, Distort, Perspective, Warp, rotate/flip variants, commit/cancel **(inferred exact context placement)** | Missing per docs/scorecard.md TOOL-214-5 | P0; transform operations [documented by Adobe](https://helpx.adobe.com/photoshop/desktop/crop-resize-transform/transform-manipulate-reshape/adjust-scale-rotation-and-perspective.html); type restrictions [documented here](https://helpx.adobe.com/photoshop/using/creating-type.html) |
| Any tool with Command/Ctrl + secondary click | List visible layers with pixels under pointer | Layers | P0; preserve existing route |

The layer-list menu on an empty pixel or all-hidden hit should be empty/closed; it must not select a transparent or hidden layer. The right-erase preference is an alternate PhotoCraft behavior, and its selected state takes precedence over the Brush picker only for Brush/Eraser.

## Pen canvas reference capture (2026-10-07)

The user supplied a Photoshop screenshot of a closed path in Path mode. The table preserves every row in its order; separator boundaries appear between the groups. Gray rows in that capture remain visible but disabled for a plain work path. Enablement changes with the active shape layer, saved path, copied style, and symmetry state. Ellipsis rows open a parameter dialog before changing the document.

| Group | Rows in screenshot order | PhotoCraft action and applicability |
|---|---|---|
| Path | Create Vector Mask; Delete Path | `layer.vectorMask.fromPath` on an eligible layer; `path.delete` on a work/saved path |
| Preset | Define Custom Shape… | `edit.defineCustomShape` with a name dialog |
| Selection/paint | Make Selection…; New Guides From Shape; Fill Path…; Stroke Path… | `path.toSelection` dialog; `view.newGuidesFromShape` for shape layers; `path.fill` and `path.stroke` dialogs on pixel layers |
| Export | Clipping Path… | `path.clippingPath.set` on a saved path, with flatness parameter |
| Geometry | Free Transform Path | `path.transform` with translation, scale and rotation parameters |
| Shape operation | Unite Shapes; Subtract Front Shape; Unite Shapes at Overlap; Subtract Shapes at Overlap | The four `layer.combineShapes.*` operations on shape layers with multiple components |
| Copy style | Copy Fill; Copy Complete Stroke | `path.style.copyFill` / `copyStroke` when the shape has that style |
| Paste style | Paste Fill; Paste Complete Stroke | `path.style.pasteFill` / `pasteStroke` when a matching style has been copied |
| Layer view | Isolate Layers | `select.isolateLayers` for an active layer |
| Symmetry | Make Symmetry Path; Disable Symmetry Path | `paint.symmetryFromPath` on a path; `paint.symmetryDisable` when symmetry is active |

The reference does not establish Photoshop's behavior for right-clicks on a single anchor, a segment, or empty canvas; these still need separate captures. PhotoCraft's disabled rows do not call a command.

## Selection canvas reference (2026-10-08)

Right-click with a Marquee, Lasso, Polygonal Lasso, Magic Wand or Object Selection tool. Rows keep the command id, dialog and enablement of their menu-bar twins (`canvas_tool_menu::SELECTION_MENU`); a greyed row never runs.

| Group | Rows with an active selection (Photoshop 2026 order) | PhotoCraft command |
|---|---|---|
| Selection | Deselect; Select Inverse; Feather…; Select and Mask… | `select.deselect`, `select.inverse`, `select.modify.feather`, `select.selectAndMask` |
| Store | Save Selection…; Make Work Path… | `select.saveSelection`; `select.toWorkPath` after a Tolerance dialog |
| Layer | Layer via Copy; Layer via Cut; New Layer… | `layer.new.layerViaCopy`, `layer.new.layerViaCut`, `layer.new.layer` |
| Transform | Free Transform; Transform Selection; Distort; Perspective | `edit.freeTransform`, `select.transformSelection`, `edit.transform.distort`, `edit.transform.perspective` |
| Fill | Fill…; Stroke…; Content-Aware Fill… | `edit.fill`, `edit.stroke`, `edit.contentAwareFill` |
| Filter | Last Filter | `filter.lastFilter` |
| Fade | Fade… | `edit.fade` |

Photoshop's Generative Fill and Delete and Fill Selection rows are left out: PhotoCraft has no such commands. With nothing selected the menu offers Select All, Reselect, Color Range… and Load Selection… (`NO_SELECTION_MENU`). Adobe-adjacent sources document Color Range there; the other rows are **inferred** until a dated capture confirms them.

## Panel and workspace matrix

| Surface / target | Desired options and state | PhotoCraft now | Source and confidence |
|---|---|---|---|
| Layers panel row | Blending Options, Duplicate/Delete, Group, export, convert/rasterize, mask, clipping, link, style copy/paste/clear, merge/flatten, Rename; filter by layer kind and selection | Implemented in layer_menu_ui.rs; item enablement uses menu commands | PhotoCraft inventory. Adobe documents layer rename/delete, color tagging, rasterization, and flattening [here](https://helpx.adobe.com/photoshop/using/layers.html); exact combined ordering **inferred** without a live capture |
| Layers panel color label / thumbnail / mask thumbnail / effect row | Color label; context actions specific to hit icon | Layer-mask thumbnail: Disable/Delete/Apply Layer Mask, Add/Subtract/Intersect Mask with Selection, Select and Mask… (layer_menu_ui.rs `show_mask`, #4575; Mask Options… missing). Generic row on the other targets; inspect before extending | Adobe explicitly documents right-click on layer/group to assign a color ([layers](https://helpx.adobe.com/photoshop/using/layers.html)); other icon menus need capture |
| Channels panel row | Duplicate/Delete/Rename, channel display and overlay color, spot conversion; new/split/merge | Implemented in channels_panel.rs | PhotoCraft inventory; exact Photoshop order **inferred** |
| Brushes and preset panels | Rename/Delete brush or group, export selected brushes, view/reset preset choices | Rename/Delete, and preset group/item menus exist | Adobe explicitly documents Export Selected Brushes from brush/group right-click ([brushes](https://helpx.adobe.com/photoshop/desktop/apply-painting-techniques/brushes-presets/create-brush-set-painting-options.html)); Photoshop pop-up-panel rename/delete/reset/view options are [documented](https://helpx.adobe.com/photoshop/using/panels-menus.html), though some are panel-button menus |
| Paths, History, Actions, Swatches, Gradients, Patterns, Styles, Libraries, Properties | Target-specific duplicate/delete/rename and panel operations where a panel exists | Paths row menu added in this PR; other panels need audit | **Inferred** catalog; record a Photoshop capture before claiming parity |
| Document tab | Close, Close Others, Close All, Reveal in file manager, duplicate/move to window where supported | Close, Close Others, Close All, Reveal in Finder, and drag to reorder implemented; duplicate/move to window where supported remains open | **Inferred** exact list; P1 |
| Empty pasteboard / workspace | Gray, Black, Custom canvas color choices; verify any additional entries in a current Photoshop capture | No menu | Adobe [documents right-clicking outside the image for canvas color](https://helpx.adobe.com/ph_en/photoshop/using/filling-stroking-selections-layers-paths.html); exact current layout still needs a capture |
| Text field / selected text | Native edit menu and spelling suggestions without routing to active image tool | Edit menu implemented in text_field_menu.rs for every text field (Undo, Cut, Copy, Paste, Delete, Select All; a right-click inside the selection keeps it; never reaches the canvas or tool menu). Spelling suggestions live in the Type tool's edit menu (canvas_tool_menu.rs), not in generic fields, which hold no type layer text (UI-217-8) | P1; **inferred** exact spelling list |
| Tool icon in Options bar | Reset Tool, Reset All Tools | Audit | Explicitly documented in [Adobe Photoshop reference](https://helpx.adobe.com/archive/en/photoshop/cc/2015/photoshop_reference.pdf); historical source, verify modern label |
| Adaptive Wide Angle constraint line | Auto, Horizontal, Vertical orientation | Current right-click deletes constraint | P1 behavior mismatch: [Adobe documents orientation choice](https://helpx.adobe.com/photoshop/using/adaptive-wide-angle-filter.html) |

## Delivery slices and acceptance

1. Core routing: target hit-test, menu model, keyboard accessibility, control inspection, conflict rules. Existing Move layer list and painting picker remain functional.
2. Canvas high frequency: selection, Free Transform, Type, Hand/Zoom, Gradient/Paint Bucket. Add rows only where the underlying command works. Record exact Photoshop label/order/disabled-state captures before declaring 1:1 parity.
3. Panels: Paths row actions, document tabs and distinct layer eye/thumbnail/mask/effect targets; finish Brushes export and observed channels/preset gaps. Adobe [documents the eye-icon visibility gesture](https://helpx.adobe.com/content/dam/help/en/photoshop/using/default-keyboard-shortcuts/photoshop-keyboard-shortcuts.pdf).
4. Long tail: shape/path, crop, metadata and specialist tools. Keep a per-tool fixture with no selection, active selection, masked/locked layer, and active operation where applicable.

Each slice needs pure model tests (including every item dispatching a known command), UI pointer tests for secondary click and modifier precedence, command/undo tests, a disabled-state test, and screenshots from PhotoCraft with the menu actually open. Compare labels, order, and state against dated Photoshop screenshots; a screenshot of a different tool or menu bar is insufficient evidence. Maintain a checklist beside each captured reference: Photoshop version/platform, target description, tool, selection/document state, modifiers, and exact items observed.

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-13 | minor | Layer-mask thumbnail menu (#4575) |
| 2026-10-10 | trivial | Status line, revision history |
| 2026-10-09 | minor | Selection canvas reference and tool matrix updates |
| 2026-10-07 | minor | Pen canvas reference capture |
