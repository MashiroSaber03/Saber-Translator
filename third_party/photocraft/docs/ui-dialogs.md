# Dialogs

Modal and modeless parameter windows in the egui shell. They mirror Photoshop: no dimmed backdrop,
canvas pan/zoom under an open dialog, Tab cycles text and number fields only, Esc cancels, Enter
confirms when OK is enabled. Automation drives the same `fields` map as the UI (`ui.dialog.*` in
[`control-protocol.md`](control-protocol.md)).

## Architecture

| Piece | Location | Role |
|---|---|---|
| Frame, routing, OK/Cancel/Esc, stacking | `crates/ui-egui/src/dialogs.rs` | Single hub for every `UiState::dialogs` entry |
| Alt+Cancel → Reset (opt-in) | `crates/ui-egui/src/dialog_alt_reset.rs` | Relabels Cancel while Alt is held; dispatches per-marker `reset` |
| Tab-through-fields | `crates/ui-egui/src/field_tab.rs` | Photoshop-style Tab / ⇧Tab (tests in same file) |
| State | `crates/ui-egui/src/state.rs` | `Dialog`, `DialogKind`, `open_dialog` |
| Button row | `crates/ui-egui/src/widgets.rs` | `dialog_buttons`, `focus_first_field`; `number_edit` registers for Tab |
| Menu / command entry | `dialogs::open_command_dialog`, `menus.rs` | Picks a dedicated module, schema dialog, or bare confirm |
| Filters | `filter_dialog.rs` | Schema from engine params, live preview |
| Adjustments | `adjust_dialog.rs` | Shared editors with Properties; preview via `adjust_preview` |
| Generic forms | `view_cmds.rs` | `__form` grids for simple command params |
| Dedicated bodies | See **Command routing** below | `body` + usually `open` / `confirm` |

**Routing.** Non-command kinds use `DialogKind` alone. `DialogKind::Command` bodies are chosen by
marker keys on `fields` (`owns()` or `contains_key("__export")`, etc.); **first match in
`dialogs::show` wins**, so new markers must not collide with an earlier arm. `show`, `confirm`, and
`cancel` share the same dispatch table—wire all three when adding a dialog.

**Field contract.** `Dialog.fields` is a JSON map: command parameters agents read and write, plus
private keys prefixed with `__`. Never use `__` for user-facing or engine params.

### `DialogKind` (top-level stack)

| Kind | Body module | Typical entry |
|---|---|---|
| `NewDocument` | `new_doc_ui.rs` | File › New, `file.new` with no size params |
| `About` | `dialogs.rs` (About / Contributors / Models tabs) | Help › About PhotoCraft; `systemInfo: true` → System Info variant |
| `Command` | Routed—see below | Most menu commands with parameters |
| `LayerStyle` | `layer_style.rs` | Layer › Layer Style… |
| `Error` | inline in `dialogs.rs` | `message` field only (e.g. move lock, locked transform) |

### Command routing (`DialogKind::Command`)

Dispatch order matches `dialogs::show` in `dialogs.rs` (check earlier rows before adding a marker).

| Match | Module | Role |
|---|---|---|
| `__guideLayout` | `guide_layout_ui.rs` | View › New Guide Layout… |
| `__fill` | `fill_ui.rs` | Edit › Fill… |
| `__patternFill` | `pattern_fill_ui.rs` | Pattern Fill layer / New Fill Layer › Pattern… |
| `__stroke` | `stroke_ui.rs` | Edit › Stroke… |
| `__createShape` | `shape_dialog.rs` | Click-with-shape-tool creation (rectangle, ellipse, …) |
| `__deleteLayer` | `delete_layer_prompt.rs` | Layers panel trash confirmation |
| `__newLayer` | `new_layer_ui.rs` | Layer › New › Layer… (⇧⌘N; ⌥⇧⌘N makes the layer without it) |
| `__rasterize` | `rasterize_prompt.rs` | Rasterize-before-paint prompt |
| `__resetAllTools` | `tool_reset.rs` | Reset All Tools confirmation |
| `__cropPresetNew` / `__cropPresetDelete` | `crop_presets.rs` | New / delete crop preset |
| `__variables` | `variables_ui.rs` | Image › Variables (Define, Data Sets, Apply) |
| `__web` / `__print` | `file_ui.rs` | Save for Web, Print |
| `__colorPicker` | `color_picker_ui.rs` | Foreground/background, nested field/command pickers (see nested keys below) |
| `__colorRange` | `color_range_ui.rs` | Select › Color Range… |
| `__prefsui` | `prefs_ui.rs` | Edit › Preferences… and related preset/import subdialogs |
| `__export` | `export_dialog.rs` | Export As (document or layer) |
| `__sizing` | `sizing.rs` | Image › Image Size… / Canvas Size… |
| `__adjust` | `adjust_dialog.rs` | Image › Adjustments… (also sets `__filter` + `__preview` for live preview) |
| `__filter` | `filter_dialog.rs` (+ `select_mask_view.rs` when `__command` is `select.refineEdge`) | Filter Gallery commands, Edit › Color Settings…, Select and Mask, and other schema dialogs |
| `__form` | `view_cmds.rs` | Simple param grids (e.g. slice options, many Image › Apply Image-style commands) |
| (none of the above) | — | **Bare confirm**: title + OK/Cancel only; OK runs `__command` with fields via generic `confirm` |

**Menu shortcut.** `dialogs::open_command_dialog` opens guide layout, stroke, and color range
directly; then tries `adjust_dialog::open`, then `filter_dialog::open`; otherwise a bare confirm
with `__command` + `__label`. Other dedicated modules are opened from `menus.rs`, tools, or panel
actions with their marker already set.

### Shared private keys

| Key | Meaning |
|---|---|
| `__command` | Engine command id run on OK (bare confirm, filters, sizing, …) |
| `__label` | Menu/catalog label for the title |
| `__document` | Target document id (validate staleness) |
| `__dialog` / `__field` | Nested color picker: parent dialog id and field name to write on OK |
| `__params` | Extra command params for command-targeted color pickers |
| `__preview` | Live canvas preview enabled (filters and adjustments) |
| `__choices` | Dropdown options for `__form` dialogs |
| `__filter` | Filter-schema body (`filter_dialog`); also set on adjustments for preview plumbing |
| `__adjust` | Image › Adjustments editor kind (`adjust_dialog`) |
| `__form` | Generic `view_cmds::form_body` |
| `__formOrig` | Snapshot of non-`__` params at open; Alt+Reset restores it for `__form` dialogs |
| `__variablesOrig` | Snapshot of `__variables` at open; Alt+Reset restores it |
| `__altReset` | Set to `false` to opt a dialog out of Alt+Cancel → Reset |
| `__export` | Export As |
| `__sizing` | Image Size / Canvas Size |

### Routing markers (dedicated `Command` bodies)

| Key | Module |
|---|---|
| `__guideLayout` | `guide_layout_ui` |
| `__fill` | `fill_ui` |
| `__patternFill` | `pattern_fill_ui` |
| `__stroke` | `stroke_ui` |
| `__createShape` | `shape_dialog` |
| `__deleteLayer` | `delete_layer_prompt` |
| `__newLayer` | `new_layer_ui` |
| `__rasterize` | `rasterize_prompt` |
| `__resetAllTools` | `tool_reset` |
| `__cropPresetNew`, `__cropPresetDelete` | `crop_presets` |
| `__variables` | `variables_ui` |
| `__web`, `__print` | `file_ui` |
| `__colorPicker` | `color_picker_ui` |
| `__colorRange` | `color_range_ui` |
| `__prefsui` | `prefs_ui` (string sub-kind: `prefs`, `presetsIO`, …) |

### Nested color pickers (`__colorPicker` on the stack)

Same hub entry as standalone pickers; `color_picker_ui::confirm` branches on extra keys:

| Extra key | Behaviour |
|---|---|
| `__layerStyleColor` | Layer Style child; OK writes effect colour on parent `LayerStyle` dialog |
| `__textStyleColor` | Character/paragraph style colour (type panels) |
| `__shapeFill` | Shape layer solid fill via `shape.edit` |
| `__command` = `layer.fill.solidColor.set` | Solid Color fill layer (no `__shapeFill`) |
| `__dialog` + `__field` | OK updates one field on a parent command dialog (e.g. Fill › Color…) |

## Parallel modal systems (not the main stack)

| System | Location | When |
|---|---|---|
| Unsaved changes | `discard_ui.rs` | Close, Revert, Close Others/All, Exit—Save / Don't Save / Cancel |
| Shell dialog | `control.rs`, `ui.inspect` id `"shell"` | Workspace, custom pixel aspect, show-extras options |
| Camera Raw | `camera_raw_ui.rs` | Separate OS window on desktop; see control-protocol § Camera Raw |
| Error message | `DialogKind::Error` | e.g. `move_lock`—message + OK |
| Nested pickers | `color_picker_ui` (+ markers in **Nested color pickers**) | Stack on parent; parent confirm refused while child open |

## Rules

### Architecture and commands

1. **Thin UI, engine command on OK.** Run `app.run` or a dedicated `*_ui::confirm`; one history step unless the command itself batches.
2. **Register in the hub.** Add `owns()` (or a stable marker), `body()`, `open()` if needed, and matching arms in `dialogs::show`, `confirm`, and `cancel` (and `open_command_dialog` when menu-driven).
3. **Extend existing paths first.** Filters → `filter_dialog`; adjustments → `adjust_dialog`; simple param grids → `__form`; avoid a second modal stack unless product needs Camera Raw–level UX.
4. **`fields` is the automation contract.** Anything agents must drive lives in named fields, not only in local egui state.
5. **Remember last-used values** after successful OK where Photoshop does (`filter_dialog::remember`, prefs `dialogs["…"]`, module-specific helpers).

### Chrome and layout

6. **Use `dialog_buttons`** for OK / Cancel / Apply—platform order via `ButtonRole` (macOS default action in the corner).
7. **Respect width rules** in `dialogs::show` unless `file_ui::dialog_width`, color picker, or kind-specific mins apply (typical 380–440; Layer Style / export / picker up to 600; New Document 800).
8. **Scroll long bodies** inside the content (`DIALOG_CHROME` for filters); title and button row stay visible.
9. **Transparent backdrop**—no dimming; previews are judged at true contrast.
10. **Position:** first open centered; then persist offset (`place_id` per command or kind); title bar drag; clamp on screen.
11. **Stacking:** only the **top** dialog is interactive; non-top instances render disabled; raise the live dialog if an older modal steals focus.

### Keyboard and focus

12. **Tab / ⇧Tab:** text and number fields only. Frame calls `field_tab::begin` / `end`; each field calls `field_tab::register` (`number_edit` does this; custom `TextEdit` must register manually).
13. **Do not rely on egui default Tab** through sliders, dropdowns, checkboxes, or preset cards.
14. **Open on first value:** `first_field_focused` + `widgets::focus_first_field` for value-first dialogs (most filters; Image/Canvas Size; not Select and Mask view keys).
15. **Select-all on focus** when reached via Tab or field arrows (`field_tab::select_all`; see `value_field`, `new_doc_ui`).
16. **Enter** confirms when OK is enabled (`can_ok`); exceptions must be explicit (`guide_layout_ui`, color picker).
17. **Esc** on topmost dialog with no popup open → Cancel. **Click outside does not dismiss** (canvas may still pan/zoom via `free_pointer_over`, `pan_delta`, `free_press`). Esc ignores Alt (it does not become Reset).
18. **Alt** turns Cancel into **Reset** when the dialog opts in via `dialog_alt_reset::supported` (dialog stays open). Opt-in markers: `__guideLayout`, `__fill`, `__variables`, `__colorPicker`, `__sizing`, `__adjust` (all kinds), `__filter`, `__form`. Each module implements its own `reset`; set `"__altReset": false` to disable.

### Canvas and preview

19. **Live preview** for filters/adjustments uses `filter_preview` / `adjust_preview`; `cancel()` clears preview and related shell state.
20. **Under-dialog canvas** respects dialog rects and open popups; Hand / Space / middle-button pan on free canvas only.

### Labels and i18n

21. **Titles:** command dialogs use `__label` (catalogued; trailing `…` stripped in chrome); fixed kinds use `dialogs::title` + `tl!`.
22. **User-visible strings:** `tl!("…")`; command labels from the menu catalog where possible.

### Confirm / cancel safety

23. **Child dialogs first**—refuse parent confirm if a nested color picker is open; cancel parent closes children where implemented.
24. **Validate before close** when params can be inconsistent (`guide_layout_ui::validate` in `confirm`).
25. **Disable OK** when required input is missing (`crop_presets::can_confirm`, Preferences Apply when unchanged).
26. **Preferences over control:** whole-dialog OK/Apply may be refused; safe single prefs via `prefs.set` (see control-protocol).

### Testing

27. **egui_kittest:** open dialog, `run_steps`, key presses, assert `fields` or accessibility roles (`field_tab` tests are the reference harness).
28. **Tab-order regressions:** add tests when a dialog gains multiple text/number fields.
29. **Filter preview:** `dialogs.rs` tests for preview persistence across background jobs.

### Anti-patterns

30. No panics in `body` / `confirm` on bad input—errors to status or validation.
31. No hard-coded colours or radii—`theme::Tokens` and `widgets.rs`.
32. Do not skip `confirm` dispatch—Enter and `ui.dialog.confirm` use it.
33. Do not expect click-outside-to-close on `egui::Modal`.
34. Do not register sliders or dropdowns for `field_tab` unless product explicitly changes Photoshop parity.

## Adding a command dialog (checklist)

1. Choose path: see **Command routing** (`adjust_dialog`, `filter_dialog`, `__form`, `__sizing`, …, or new `*_ui` with a unique marker).
2. `open`: build `fields` (`__command`, `__label`, params, marker); `app.ui.open_dialog(DialogKind::Command, f)`.
3. `body(app, ui, &mut fields)`—mutate `fields` only; use shared widgets.
4. `confirm` / `cancel` arms in `dialogs.rs` (or delegate to module `confirm`).
5. Tab: register every text/number field; use `value_field` / `number_edit` where possible.
6. **Alt+Reset (optional):** implement `reset` on the module and add an arm in `dialog_alt_reset` (same dispatch order as `dialogs::show`). Snapshot open values under a `__*Orig` key when Reset should restore “as opened” rather than factory defaults.
7. Tests: at least open + confirm or Tab sequence if non-trivial; if Alt+Reset is supported, assert Reset restores fields and leaves the dialog open.
8. If menu-visible: command already in engine; menu id opens dialog via existing menu wiring.

## Known gaps

- **Tab registration** for plain `TextEdit` is manual; generic `__form` string fields may not join `field_tab` unless wired.
- **“Dialog module”** is any UI routed from `dialogs::show`, not only files named `*dialog*`.
- **Numeric field parity** (units in expressions, scrubby labels) is tracked in [`ui-parity.md`](ui-parity.md) (Navigation / numeric fields), not duplicated here.

## Revision history

| Date | Change |
|---|---|
| 2026-10-11 | Initial doc from dialog codebase audit |
| 2026-10-11 | Document all `DialogKind` values and `Command` routing markers |
| 2026-10-11 | Alt+Cancel → Reset hub (`dialog_alt_reset`); opt-in markers |
