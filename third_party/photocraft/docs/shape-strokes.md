# Shape stroke options

Updated 2026-10-11: direct editing, compact layout, and clickable colour/style controls.

PhotoCraft's Shape tool options bar has a Stroke Options button beside the existing width
and colour controls. It sets defaults for new Rectangle, Ellipse, Triangle, Polygon, Line
and Custom Shape layers. Pen in Shape mode uses the same style defaults; open Pen shapes retain
their existing foreground colour and minimum 1 px width. To edit an existing shape,
use Properties → Appearance → Stroke Options.

Fill and Stroke swatches are clickable and open the existing compact colour picker. Choosing
a colour preserves its alpha; No Color removes the corresponding paint. Tool-bar fill uses
the foreground colour and stroke uses the background colour, as before. Choosing a stroke
colour or style deliberately enables a 3 px stroke when no stroke is present. The style button
remains available with no stroke, but opening or dismissing it alone does not enable one.
Properties edits the selected shape; the options bar continues to set defaults for new shapes.

The editor offers Solid, Dashed (4 widths on, 2 off), Dotted (zero-length dashes with round
caps, spaced every 2 widths), and custom repeating dash/gap sequences. Dash lengths and
offsets are multiples of the stroke width; width itself is in document pixels. Odd-length
imported sequences repeat twice, following the existing renderer convention. Up to 16
dash/gap pairs can be edited, with Add pair / Remove pair. Zero-length dashes draw dots with
Round caps and squares with Square caps; Butt caps draw nothing at a zero-length dash.
Each preset's full row, including its line sample, is one clickable target with a visible
selected state and an accessible label.

Caps are Butt / Flat, Round, and Square. They affect open endpoints and individual dashed
segments. Closed solid shapes have no endpoints, so their cap control is disabled. Corners
are Miter, Round, or Bevel; Miter limit is a ratio, 1–500 (the existing default remains 100).
Sharp miters exceeding the limit become bevels. Ellipse has no sharp corners, so its corner
controls are disabled. The Line tool retains its existing filled-bar geometry: strokes follow
the bar's closed outline. Use an open Pen shape for a centerline with endpoint caps.

Inside / Center / Outside alignment applies to paths whose subpaths are all closed. Inside
and Outside clip a double-width stroke against the shape's fill area and respect the existing
fill rule and component operations. Open and mixed open/closed paths use Center; their
alignment control is disabled. Dash lengths continue to use the specified width, including
for Inside and Outside. Opacity is independent of width and paint.

Stroke Options applies valid changes immediately and has no Apply or Cancel buttons.
Properties sends changes through `shape.edit`; one continuous numeric drag is one undo step.
Escape or a click outside closes the popup and retains already applied changes. Opening or
closing it without changing a value creates no history entry. Invalid pending numeric values
do not replace the last valid style. Child dropdowns can open and select an option without
dismissing Stroke Options; Escape closes the child before the parent. The popup fits its
content and scrolls only when the available height requires it.

The compact line and rectangle previews use the real vector renderer. Tool-default changes
do not edit the document. Shape drag previews use the vector renderer, with a cached screen
proxy capped at 2048 pixels on the longest axis; like the prior drag overlay, their colours
bypass monitor-profile conversion. Committed shapes, including live Properties edits, use
the normal colour-managed compositor.

Finite values are required: width 0–1,000,000 px, opacity 0–100%, dash/gap lengths
0–10,000 widths, offset ±1,000,000 widths, at most 32 lengths, and a nonempty pattern total
of at least 0.01 widths. Interactive commands reject geometry beyond ±1,000,000 px,
over 4096 knots/subpaths, more than 100,000 tessellation points, or a dash workload over
100,000 steps. File-supplied invalid/dense patterns are bounded during rendering. There are
no custom saved preset libraries in this change. Style defaults use existing serde UI state,
but the app does not save and restore these defaults across restarts.

All fields already exist in `ShapeStroke` and the native manifest/conversion. Old `.pcraft`
files retain their serde defaults; saves preserve all style fields. PSD import/export maps
the same fields through `vstk` (line width, dash set/offset, cap, join, alignment, miter limit,
opacity, and paint); existing unknown layer blocks and original compatible vector records
remain handled by the existing I/O path. Stroke overprint, non-Normal PSD vector-stroke
blend modes, corner-fitting dash optimization, and Photoshop's saved stroke preset libraries
are not implemented. No Photoshop-authored oracle establishes exact stroke UI parity here.

Automation: `shape.create`, `shape.edit`, and `shape.info` expose document strokes. The
control channel also supports `ui.set {shapeStroke: {width, align, cap, join, miterLimit,
dashes, dashOffset, opacity}}` for tool defaults, merging partial fields; null sets width to
zero. Set tool colours with `tools.setColors`. `ui.inspect` reports `toolOptions.shape_stroke`,
`toolOptions.stroke_width`, and the transient `strokeEditor`. Keyboard focus, standard number
fields, colour pickers, and ComboBoxes remain native egui controls.

## Upstream and public references

- [PR #2218](https://github.com/storytold/photocraft/pull/2218) introduced these controls with
  staged previews. The direct-edit interaction above replaces that Apply/Cancel workflow.
- [Issue #2496](https://github.com/storytold/photocraft/issues/2496) tracks broader Shape tool
  parity, including selected-shape editing from the options bar and additional fill types;
  those remain separate work.
- [Issue #1053](https://github.com/storytold/photocraft/issues/1053) documents the missing
  stroke controls and existing engine/model support. Gradient fill UI remains separate work.
- [Issue #1931](https://github.com/storytold/photocraft/issues/1931) requests sharp stroke options.
- [PR #233](https://github.com/storytold/photocraft/pull/233),
  [#397](https://github.com/storytold/photocraft/pull/397), and
  [#1495](https://github.com/storytold/photocraft/pull/1495) already address stroke compositing,
  hairlines, and depth preservation; their implementations are reused.
- [PR #2031](https://github.com/storytold/photocraft/pull/2031) supplies the click-to-create
  dialogs. Both dialogs and drag creation use its shared `create_shape` helper, which receives
  the complete stroke defaults. A gesture regression checks both paths and their undo steps.
- Adobe's [Photoshop shape fill/stroke guide](https://helpx.adobe.com/photoshop/desktop/draw-shapes-paths/create-shapes/fill-and-stroke-shapes.html)
  describes the tool options and Properties entry points, and its
  [cap/join reference](https://helpx.adobe.com/ca/illustrator/desktop/paint-and-fill/apply-and-edit-strokes/change-the-caps-or-joins-of-a-line.html)
  defines endpoint and corner behavior and the 1–500 miter range. Dash units reuse
  PhotoCraft's existing documented `ShapeStroke`/`vstk` convention.
