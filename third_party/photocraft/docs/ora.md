# OpenRaster documents

PhotoCraft opens and saves OpenRaster (`.ora`), the layered format Krita, MyPaint, GIMP, Pinta
and others share. File › Open, recent files, drops, Save and Save As, the CLI (`convert`, `info`,
`batch`) and automation (`doc.open`, `doc.save`, MCP) all use the same reader and writer in
`photocraft-io`. The browser build accepts `.ora` too. A file is recognised by its `mimetype`
entry, so an OpenRaster file with another extension still opens as one.

## What is kept

| | Open | Save |
|---|---|---|
| Pixel layers (any size and position, also off-canvas) | yes | yes, cropped to their pixels |
| Groups, nested | yes | yes |
| Pass-through groups | `isolation="auto"` without a blend mode | `isolation="auto"` |
| Names, visibility, opacity | yes | yes |
| Blend modes | the `svg:*` names, `svg:plus`, and Krita's `krita:*` names | `svg:*` where the spec has the mode, else `krita:*` (read back by Krita, Normal elsewhere) |
| Locks | `edit-locked`, `alpha-preserve` | the same |
| 8 and 16-bit | yes (the deepest layer sets the depth) | 8-bit documents as 8-bit PNGs, 16 and 32-bit as 16-bit PNGs |
| Resolution | `xres` | `xres` and `yres` |
| Merged image and thumbnail | not read (we render our own) | written (`mergedimage.png`, `Thumbnails/thumbnail.png`) |

Saving reports what OpenRaster can't hold, layer by layer: pixel masks are applied to the
layer's pixels, fill opacity is folded into the pixels, text, shape, fill and smart object layers
are written as their rendered pixels, and adjustment layers, layer styles, vector masks,
clipping, alpha channels, metadata and a colour profile other than sRGB are left out. Xor and Paint.NET's
Color Burn and Color Dodge variants have no OpenRaster name and are saved as Normal. CMYK, Lab
and other modes are refused with a message to convert first; Grayscale saves as RGB.

An OpenRaster file is the document's own file only while it holds everything in it (#3806).
`photocraft_io::ora_losses` lists what a save would leave out, from the same checks the writer
warns with, without encoding anything. When that list isn't empty, File › Save doesn't overwrite
the `.ora`: Save As opens on a PSD of the same name, with a notice listing what OpenRaster would
drop. A save to an `.ora` path then writes a copy, as Photoshop does for formats that can't keep
everything: the document keeps its file and its unsaved changes. Automation follows the same rule
(`doc.save` and `app.save` without a path refuse, saying why).

OpenRaster elements other than `stack` and `layer` (MyPaint's text, for example), missing
layer images and unknown composite operations open with a warning instead of failing the file.
Krita's W3C Soft Light (`svg:soft-light`, `krita:soft_light_svg`) and its own Hard Mix open as
Photoshop's Soft Light and Hard Mix, which differ slightly, also with a warning.

## Limits and safety

The reader takes the archive with the bounded ZIP reader of `photocraft-format` (no ZIP64,
CRC-checked), parses `stack.xml` with `roxmltree` (16 MiB cap), and decodes each layer image
through `photocraft-codecs` with size limits. Canvases are at most 300,000 px a side, at most
8,192 layers and groups, nesting at most `MAX_GROUP_DEPTH` deep, and 1 GiB of decoded pixels in
total. Malformed archives, XML, numbers and images return errors; truncated and corrupted files
are covered by tests.

## Verification

Unit tests (`crates/io/src/ora/tests.rs`) round-trip layers, groups, pass-through, offsets,
opacity, visibility, locks and every blend mode at 8 and 16 bit, check masks, rendered layer
kinds, the archive layout and malformed input. `crates/automation/tests/openraster.rs` saves,
reopens, edits and saves in place through the agent surface, and checks that a document with an
adjustment layer never overwrites its `.ora`; the CLI test converts both ways.

Real-file checks use Krita-authored oracles from
[photocraft-corpus](https://github.com/storytold/photocraft-corpus) (`krita/`, made by
`tools/krita-oracles/generate.py`). Each Krita file carries Krita's rendering of itself in
`mergedimage.png`:

```sh
KRITA_TEST_DIR=/path/to/photocraft-corpus/krita cargo test -p photocraft-io --test krita_corpus -- --ignored --nocapture
```

2026-10-09, Krita 5.2.9: 32 of 35 `.ora` files render within 3/255 of Krita's merged image; the
other three differ for known reasons listed in `crates/io/tests/krita_corpus.rs` (Krita's
tie-break in Darker and Lighter Color, and Hard Mix at a channel sum of exactly 1). In the other
direction, Krita opened every PhotoCraft re-export of the first 34 of those files and rendered 33
of them identically to its original; the exception is Photoshop's Soft Light, which is written
as `svg:soft-light`. MyPaint and GIMP files have not been checked yet.

Format reference: the [OpenRaster specification](https://www.openraster.org/). The reader and
writer are written from the specification and from files Krita writes; no application code
was used.
