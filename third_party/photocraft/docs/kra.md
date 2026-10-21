# Krita documents

PhotoCraft opens Krita's native documents (`.kra`, and the `.krz` archive variant) with their
layers. File › Open, recent files, drops, the CLI (`convert`, `info`, `batch`) and automation
(`doc.open`, MCP) all use the reader in `photocraft-io`; the browser build accepts them too. A
file is recognised by its `mimetype` entry (`application/x-krita`). Krita files are import-only:
Save suggests OpenRaster (`.ora`), which Krita opens with its layers, and `.pcraft` keeps
everything.

## What is read

| Krita | PhotoCraft |
|---|---|
| Paint layers, 8 and 16-bit RGBA, 8 and 16-bit gray | pixel layers, at the document's depth (a gray layer in an RGB document is converted) |
| Layer position (`x`, `y`), tiles anywhere, also off-canvas | kept |
| Groups, nested; pass-through groups; collapsed state | groups; pass-through; expanded state |
| Names, visibility, opacity, lock | kept |
| Blend modes | Krita's ids, mapped as for OpenRaster (`docs/ora.md`); a few are approximations, reported |
| Transparency masks | the layer mask (the first one; more are reported and left out) |
| Selection masks | ignored (a layer's local selection, not part of its look) |
| The image's ICC profile, resolution | kept |

Left out with a warning per layer: vector, filter, fill, file and clone layers, filter, transform
and colorize masks, layers in another colour space than those above, Inherit Alpha, per-channel
blending flags (channel locks), and animation beyond the frame saved in the layer. Drawing
assistants and reference images are reported as left out. Krita's merged image still shows
the original look, so the result can differ where those were used.

Documents whose image colour space we don't read yet (CMYK, Lab, XYZ, YCbCr, 16 and 32-bit
float) open as Krita's merged image (`mergedimage.png`), flagged as a preview, with a warning.

## The format, as observed

A ZIP whose first, stored entry is `mimetype`; `maindoc.xml` holds the node tree (topmost first)
and points each paint layer and mask at a file under `<image name>/layers/`. Those files are a
text header (`VERSION 2`, `TILEWIDTH`, `TILEHEIGHT`, `PIXELSIZE`, `DATA <tiles>`) and per tile a
`x,y,LZF,<bytes>` line followed by a compression flag and the tile's pixels in byte planes, LZF
compressed. RGB pixels are B, G, R, A; 16-bit samples are little-endian. A `.defaultpixel` file
next to each holds the colour outside its tiles. A group's own `x`/`y` does not move its
children (checked against Krita's merged image).

## Limits and safety

The reader uses the bounded ZIP reader of `photocraft-format` (no ZIP64, CRC-checked), parses
`maindoc.xml` with `roxmltree` (32 MiB cap; the DTD is declared by public id and not fetched),
and decodes tiles with a bounds-checked LZF decoder that must produce exactly one tile. Tiles
are at most 1024 px a side; canvases at most 300,000 px a side and 1 GiB of decoded pixels;
8,192 nodes; nesting at most `MAX_GROUP_DEPTH`. Version 1 tile files (Krita 1.x) are refused.
Malformed archives, XML, numbers, tile headers and tile data return errors; truncated and
corrupted files are covered by tests.

## Verification

Unit tests (`crates/io/src/kra/tests.rs`) build Krita archives in code (raw and LZF tiles) and
check layers, groups, pass-through, offsets, opacity, visibility, locks, masks, default pixels,
16-bit and gray conversion, the merged-image fallback, LZF edge cases and malformed input.
`crates/automation/tests/krita.rs` opens a `.kra` through the agent surface, edits it and saves
it as OpenRaster. `crates/engine/tests/kra.rs` (opt-in) opens a Krita-authored 16-bit file in a
session, checks it is read-only, edits with undo and saves as `.ora` and `.pcraft`.

Real-file checks use the Krita-authored oracles in
[photocraft-corpus](https://github.com/storytold/photocraft-corpus) (`krita/`, made by
`tools/krita-oracles/generate.py`, which removes Krita's embedded third-party ICC profiles so
the files are sRGB):

```sh
KRITA_TEST_DIR=/path/to/photocraft-corpus/krita cargo test -p photocraft-io --test krita_corpus -- --ignored --nocapture
KRITA_TEST_DIR=/path/to/photocraft-corpus/krita cargo test -p photocraft-engine --test kra -- --ignored
```

2026-10-09, Krita 5.2.9: 31 of 35 `.kra` files render within 3/255 of Krita's merged image; the
others differ for known reasons (the three blend-mode tie-breaks as for OpenRaster, and the
file with a vector layer, which is left out). Before the profiles were stripped, a linear-light
16-bit document with its embedded profile matched Krita's merged image within 0.5/255 after
flattening and converting to sRGB.

The reader is written from files Krita writes; no Krita code was used.
