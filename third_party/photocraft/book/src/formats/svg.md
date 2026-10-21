# SVG

An SVG opens as a document of shape layers, and places (File › Place Embedded / Place Linked) as a vector smart object. Both go through `photocraft-io`'s `svg` module; there is no SVG export.

## Opening

`usvg` parses the file (it is the same crate the UI's icons go through): CSS, `use`, units, nested transforms and `viewBox` are resolved, and text is shaped into outlines with the bundled craft fonts plus, on the desktop, the system's fonts. The resulting tree becomes layers in paint order:

| SVG | Layer |
|---|---|
| `path`, `rect`, `circle`, `ellipse`, `line`, `polyline`, `polygon` | A shape layer: cubic Bézier knots with the element's transform baked in, fill (colour and opacity, fill rule), stroke (width, colour, caps, joins, miter limit, dashes). |
| `g` with an `id`, opacity or blend mode | A layer group with that opacity and blend mode. Groups that only carry a transform are not layers. |
| `text` | A group of outline shapes named after the text. |
| Linear and radial gradient paints | The layer's own gradient fill with the stops and direction; the exact gradient geometry is not kept (a warning says so). |
| `clip-path`, `mask`, `filter`, pattern paints, `image` | Rasterised by `resvg` into a pixel layer over the element's bounding box, so the picture looks right while those parts are not editable as shapes (a warning names each kind). |

The canvas is the drawing's size in CSS px (from `width`/`height`, else the `viewBox`), RGB 8-bit, transparent where nothing is drawn. A drawing larger than 16 384 px on a side is scaled down to fit. Past 2 000 elements the drawing opens as a single pixel layer rather than thousands of shapes. Malformed SVG is an error, never a panic; `.svgz` is accepted by its extension and plain SVG is also recognised by content when the extension is missing.

## Placing

A placed SVG keeps its bytes as the smart object's source. Whenever the smart object is rendered (at placement, after Free Transform, when a smart filter changes) the drawing is rasterised at the placement's scale and then positioned, so it stays sharp at 1 000 % where a resampled raster would blur. Edit Contents opens the source as the shape-layer document above; saving it back stores the edited document, as for any smart object.

Limits: a single render is capped at 24 million pixels; warped smart objects and stacks render from the drawing's own size.
