# Brush and painting parity

> **Last reviewed:** 2026-10-10 · **Last updated:** 2026-10-13 · **Change:** minor (Hue Jitter amount track; Brush Settings minimum fields follow their Control; checkbox-only rows and non-destructive Smoothing switch; shaped-tip cursor follows the view; Color Replacement is a live toolbar tool; no brush-feel estimate changed) · **Target:** Adobe Photoshop 2026

Painting is where Photoshop users notice differences first, so it gets its own checklist: the
brush engine (`crates/paint/src/brush.rs`), the Brush Settings UI
(`crates/ui-egui/src/brush_sections.rs`), the live stroke renderer, pen input
([`hardware-parity.md`](hardware-parity.md)), and the brush-based tools. Status words as in
[`ui-parity.md`](ui-parity.md).

**Summary (estimated):** breadth ~90% (every Brush Settings section and option exists), ready for
real work ~55%. The engine models what Photoshop models; the gaps are feel (latency, pressure
response, edge quality, smoothing), a few missing options, and live previews for retouch tools.
Remaining: 100–180 h (gap G4 in [`gaps.md`](gaps.md)). Human: feel must be checked on real
tablets against Photoshop side by side.

## Brush Settings sections

| Section | Status | Evidence |
|---|---|---|
| Brush Tip Shape: size, angle, roundness, hardness, spacing, flip X/Y | done | `brush.rs` |
| Shape Dynamics (size/angle/roundness jitter, pen pressure/tilt/rotation/wheel controls, minimums, flip jitter, brush projection) | done | #174 |
| Scattering (both axes, count, count jitter) | done | |
| Texture (pattern, scale, brightness, contrast, mode, depth, each tip) | done | |
| Dual Brush | done | |
| Color Dynamics (FG/BG, hue/sat/brightness jitter, purity, per tip) | done | |
| Hue Jitter uses a standard percentage-amount track instead of a hue spectrum | done | #3825 (Hue Jitter appearance only); rendered track checked in all 14 themes; numeric and pointer edits retain percentage units and engine command routing. |
| Transfer (opacity/flow/wetness/mix jitter and controls) | done | |
| Shape Dynamics and Transfer minimum fields disabled when their Control is Off, retaining their values | done | #3825 (minimum-field interaction only); `brush_sections.rs` UI regressions cover all six fields, reactivation and disabled input |
| Brush Pose (tilt, rotation, pressure overrides) | done | |
| Noise, Wet Edges, Build-up, Protect Texture | done | |
| Smoothing: amount, Pulled String, Stroke Catch-up, Catch-up on End, Adjust for Zoom | done | `brush.rs:428-438` |
| Section locks | done | #174 |
| Noise, Wet Edges, Build-up, Smoothing and Protect Texture: checkbox-only rows keep the selected controls open | done | `brush_panel_tests.rs`, #4283; Smoothing enable/disable preserves its amount and options, including imported ABR tool settings |
| Brush tip from an image file | missing | TOOL-213-12 |
| Erodible, Airbrush and Bristle tips (Photoshop's special tips) | partial | not audited in detail; "3D-based brushes don't work" (#1660) |
| Anti-aliasing off for brushes | missing | #1666 (Pencil is the aliased path) |

## Pressure and pen

| Behaviour | Status | Evidence |
|---|---|---|
| Pressure drives size/opacity/flow per sample (no shared pressure within a frame) | done | #2112 |
| Global pressure curve in Preferences | done | #2195 |
| Per-control pressure curves | missing | TOOL-213-11 |
| Quick taps keep their pressure | partial | #1967 fixed one case; tap paints a full blob (#1798), tap sensitivity (#1933) |
| Smooth pressure on Windows | partial | "jumpy" (#2046) |
| Tilt and rotation drive angle/roundness | partial | macOS, X11, web; not Windows; rotation not applied (#1301) |
| Eraser end of the pen switches to Eraser | partial | macOS; flipped Wacom on Windows not (#2285) |

## Stroke quality and feel

| Behaviour | Status | Evidence |
|---|---|---|
| No seams, hard edges stay hard, zero-scatter defaults, ABR tips never black | done | TOOL-213-13 |
| Smooth line weight along a pressure ramp | partial | stair-steps (#1776) |
| Interpolation and smoothing match Photoshop | partial | #2138, #2057 |
| Soft-edge falloff matches Photoshop's curve | partial | #2149, #1619 |
| Brush preview and cursor crisp, not pixelated | partial | #1433 |
| Cursor outline equals the painted footprint at every zoom | partial | #1761, #2364; shaped tips trace their outline (#3467), turned and flipped with a rotated or flipped view |
| Latency: dab to screen within a frame | partial | budgets P13/P16/P33 over (scorecard); big-brush lag #2619, #1994; cursor lag #2566, #1343 |
| No hitch at mouse-up on a long stroke | not measurable | P32 needs a frame harness |
| Live stroke for Brush and Eraser | done | |
| Live stroke for every brush-based retouch tool | missing | TOOL-213-7 |
| No OS arrow cursor over the canvas with a pen | partial | TOOL-213-14 |

## Gestures and cursors

| Behaviour | Status | Evidence |
|---|---|---|
| ⌃⌥-drag (macOS) / Alt+right-drag (Windows) resizes and changes hardness with a readout | done | TOOL-213-10 |
| `[` `]` size, `⇧[` `⇧]` hardness, number keys opacity, ⇧+numbers flow | done | 2026-10-07 |
| ⌥-click samples a colour with painting tools; Eyedropper comparison ring | done | TOOL-213-9 |
| ⇧-click draws a straight line from the last point | done | |
| Hold `~` with the Brush or Pencil to erase with its brush: Clear mode while held (rebindable under Tools › Temporary) | done | `hold_keys.rs`, #3439 |
| Clear painting mode in the Brush and Pencil Mode menus | done | `paint_mode_dropdown` in `panels.rs`, `"mode": "clear"`, #3439 |
| Right-click opens the Brush Preset picker | done | [`context-menu-parity.md`](context-menu-parity.md); XP-Pen right-click not (#1980), macOS issue #2360 |
| Precise / normal / full-size tip cursors, crosshair in the tip | partial | Caps Lock precise cursor not audited; round Pencil cursor requested (#2662, #1057) |
| Brush choice kept per tool (Brush, Eraser, Blur each their own) | partial | #2542 |

## Symmetry

| Mode | Status |
|---|---|
| Vertical, Horizontal, Dual axis, Diagonal | done |
| From a path (Make Symmetry Path) | done |
| Wavy, Circle, Spiral, Parallel Lines, Radial, Mandala | missing (#2659) |

## Brush-based tools

| Tool | Status | Notes |
|---|---|---|
| Brush, Pencil (aliased, Auto Erase), Eraser (Brush/Pencil/Block) | done | Pencil 1 px quality #2139 |
| Mixer Brush (Wet, Load, Mix, Flow, Sample All Layers) | partial | "incomplete" (#2247) |
| Clone Stamp (opacity, flow, aligned, sample, overlay) | partial | Clone Source W/H link and Alt target cursor (TOOL-213-8) |
| Pattern Stamp (aligned, impressionist) | done | |
| History Brush | done | Art History Brush missing |
| Spot Healing, Healing | partial | grain not matched on flat colour (#2104), "horrible in general" (#1092), stops responding after switching (#1751) |
| Blur, Sharpen, Smudge (premultiplied, Sample All Layers, spacing) | done | BUG-207-2/3, #2171 |
| Dodge, Burn, Sponge | done | |
| Background Eraser, Magic Eraser | done | |
| Color Replacement | done | |
| Quick Selection (brush-driven) | done | |

## Presets

| Behaviour | Status | Evidence |
|---|---|---|
| .abr import (v1/v2/v6) and export, persistent presets, folders | done | `psd/src/abr.rs` |
| Brushes panel: drag a folder into a folder | missing | #2014 |
| Tool presets (.tpl) | partial | Tool Presets panel exists; .tpl import not found |
| Import 500 brushes without blocking the UI | not run | P24 |

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-13 | minor | Stale "missing tool" row corrected against main: Color Replacement (`paint.colorReplacement`) is a live toolbar tool; no brush-feel estimate changed |
| 2026-10-10 | major | First version, from a source audit and the open issues |
| 2026-10-11 | minor | `~` held with the Brush or Pencil erases with that tool's brush, and Clear joins their Mode menus (#3439) |
| 2026-10-12 | minor | Five Brush Settings switch rows toggle without selecting a properties pane; Smoothing has a default-on switch and retains its settings while disabled (#4283). No brush-feel estimate changed. |
| 2026-10-12 | minor | A shaped tip's cursor outline turns and flips with a rotated or flipped view, and follows pixel aspect correction |
| 2026-10-13 | minor | Shape Dynamics and Transfer minimum inputs follow their Control and retain their values while disabled (#3825, partial). No brush-feel estimate changed. |
| 2026-10-13 | minor | Hue Jitter uses the same themed amount slider as other jitter controls (#3825, partial). No brush-feel estimate changed. |
