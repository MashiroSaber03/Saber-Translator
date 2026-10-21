# Raster formats

`photocraft-codecs` provides format detection, capability declarations, decode limits, metadata handling, and encoding for flat images. The default build declares read/write support for PNG, JPEG, TIFF, WebP, GIF, BMP, TGA, ICO, Netpbm/PFM, QOI, OpenEXR, and Radiance HDR. HEIF/HEIC (iPhone and Mac photos) is read-only, decoded in pure Rust by `heic-rs` through the optional `photocraft-heif` crate. It is a build-time choice behind the `heif` cargo feature (`cargo build -p photocraft --features heif`): official builds include it; without it HEIC files are still recognised and opening one reports that HEIC support isn't included in that build. AVIF encoding is optional; AVIF decoding is not implemented in the current crate.

## Layered TIFF

A TIFF saved by Photoshop with its layers carries them in two private tags: 37724 (`ImageSourceData`, the same layer section a PSD has, preceded by the string `Adobe Photoshop Document Data Block`) and 34377 (the PSD image resources). `photocraft-codecs` passes both through as opaque bytes on `Metadata`; `photocraft-io` rebuilds a PSD model from them plus the TIFF's pixels and hands it to the PSD importer, so groups, masks, layer styles, type, shapes, smart objects and adjustment layers open from a TIFF exactly as from a PSD, and are written back the same way. In Intel-order (`II`) files Photoshop stores the whole block structure byte-swapped (reversed keys such as `MIB8` and `ryaL`, little-endian integers and doubles) while pixel samples stay big-endian; `photocraft-psd`'s `tiff` module converts between the two orders field by field, and drops, with a warning, any block whose layout it does not know rather than pass byte-swapped data to other readers. Limits: a TIFF holds one transparency sample, so extra alpha channels are not written; Lab documents are saved flat. Save As asks "Including layers will increase file size" (with "Discard Layers and Save a Copy") while *Ask Before Saving Layered TIFF Files* is on in Preferences › File Handling. Scripted and agent saves write a flat TIFF unless they ask for layers: `--tiff-layers` for `photocraft-cli convert`, `run` and `batch`, and `"tiffLayers": true` in the save, batch and export commands and MCP `doc_save`.

## WebP

Lossless WebP (VP8L) is encoded and decoded by `image-webp`. Lossy WebP is written by PhotoCraft's own VP8 key-frame encoder in `crates/codecs/src/codecs/vp8`, a clean-room implementation of RFC 6386: BT.601 4:2:0 conversion, 16×16 and 4×4 intra prediction chosen per macroblock by reconstruction error, the forward DCT and Walsh-Hadamard transforms paired with the bit-exact inverse transforms, quantisation on the RFC's `dc_qlookup`/`ac_qlookup` tables, token coding with per-frame probability updates, and skipped macroblocks for flat areas. `EncodeOptions::webp_lossless` picks the encoder and `webp_quality` (0–100, the libwebp scale) sets the quantiser; alpha travels losslessly in an `ALPH` chunk and ICC/EXIF/XMP in their own chunks of the extended (`VP8X`) container. The decoder applies the frame's loop filter; no C library is involved on either side.

Capability details such as sample depth, channel layout, alpha, ICC, EXIF, XMP, DPI, and animation behavior are defined by `crates/codecs/src/format.rs`. Do not infer fidelity from the filename extension alone.

## Exporting documents

Every route that writes a document to a flat format (Save As, the CLI, MCP `doc_export`, Layers to Files, Layer › Export As) flattens it the same way:

- Formats without transparency (JPEG, Radiance HDR) get the image composited over white, as Export As and Quick Export do; in CMYK, white is no ink.
- Formats that can't embed an ICC profile (GIF, BMP, TGA, QOI, ICO, Netpbm/PFM) get RGB documents in another profile (linear light, Display P3, …) converted to sRGB through the colour engine (perceptual intent), since an untagged file is read as sRGB. OpenEXR and Radiance HDR get linear sRGB instead (linear gray for a grayscale OpenEXR), which is how they are tagged when opened.
- TGA, as Photoshop's Targa: an RGB Targa is 24 or 32 bits per pixel. Save As and Save a Copy ask (Targa Options, starting at 32 with an alpha channel, else 24); Export As has a Resolution choice (Auto, 24 or 32 bits/pixel); scripts pass `tgaBits` (`--tga-bits` on the CLI). At 32 bits the document's alpha channel is the file's alpha when it has exactly one (spot channels don't count; with several, none is used), and the colours are written exactly, never premultiplied or cleared where the alpha is 0, so a game texture's mask and its colours both survive. Without a usable alpha channel the layers' transparency is the alpha (opaque without any). 24 bits write no alpha, compositing transparency over white. Without a choice: 32 with an alpha channel or transparency, else 24. A 32-bit Targa opens as an opaque Background with its alpha as "Alpha 1", so it round-trips. Grayscale Targas are 8-bit.

Flat TIFF and JPEG exports of composited CMYK documents convert the RGB composite back through
the document's CMYK ICC profile (relative colorimetric with black point compensation) and embed
that exact profile. Untagged composites use and embed PhotoCraft's synthetic coated CMYK profile;
invalid, non-CMYK or unusable embedded profiles are rejected before compositing. TIFF retains
8/16/32f depth and straight alpha; JPEG reduces to 8-bit with a warning and mattes transparency
over zero-ink white. A plain single raster layer retains its native samples. RGB-only formats use
the RGB composite directly. This RGB composite roundtrip does **not** preserve original ink
separations, black generation or native CMYK blending, and is not a claim of Photoshop parity.

## Implemented decode limits

`DecodeOptions::default()` applies `Limits` before decoded pixel allocation:

| Limit | Default |
|---|---:|
| Maximum width | 262,144 pixels |
| Maximum height | 262,144 pixels |
| Maximum pixel count | 1,073,741,824 pixels |
| Maximum decoded allocation | 8 GiB on 64-bit targets; 2 GiB on 32-bit targets (including wasm32) |

Format adapters pass compatible allocation limits into PNG, TIFF, WebP, and `image`-based decoders, while other implementations perform explicit checked dimension/buffer validation. Tests cover bomb-like headers, truncation, bit flips, header overwrites, and random data.

`Limits::none()` exists for callers with a deliberate reason to relax policy. It remains bounded by representable memory sizes, but it removes the default resource policy and should not be used for untrusted input.

Animated containers and multi-page TIFFs are represented as a single decoded frame or page by the current codec API; opening one adds an import warning such as "only the first of 3 frames was imported". A JPEG cut off inside its image data opens with a warning that its data ends early (an error when it holds no image data at all). Refer to [`crates/codecs/README.md`](https://github.com/storytold/photocraft/blob/main/crates/codecs/README.md) and source capability declarations for current behavior.
