# Camera Raw preview navigation

The Camera Raw dialog's preview has Zoom, Hand, Fit in View, 100%, and zoom preset
controls at the bottom left.

Choose **Fit in View**, or press **Ctrl+0** on Windows/Linux (**Command+0** on macOS),
to fit the whole image inside the preview.

At **100%**, one source pixel maps to one physical display pixel, including HiDPI.
The native preview refines the processed source in the background so texture,
sharpening, and grain can be inspected beyond the reduced proxy's resolution.

## Controls

| Action | Windows / Linux | macOS |
|---|---|---|
| Select Zoom | Z | Z |
| Scrub zoom in / out | Left-drag right / left with Zoom | Same |
| Toggle Fit / 100% | Click with Zoom | Same |
| Zoom a selected region | Ctrl+drag | Command+drag |
| Select Hand | H | H |
| Temporary pan | Space+drag, or middle-button drag | Same |
| Pointer-anchored zoom | Alt+wheel, or pinch | Option+wheel, or pinch |
| Zoom in / out | Ctrl+plus / minus | Command+plus / minus |
| Fit in View | Ctrl+0 | Command+0 |
| Actual size | Ctrl+Alt+0 | Command+Option+0 |
| Zoom presets | Bottom-left dropdown or right-click menu | Same |

Zoom and pan are view state: they do not change filter parameters or document
history. Before/After keeps the same position; colour probes and clipping overlays
follow the image coordinates. Cancel discards the filter; OK applies it through
the existing Camera Raw command.

## Preview limits

The fast proxy remains available while native refinement is pending. Processed
full-resolution refinement is limited to 64 megapixels because the existing
engine allocates full-image temporary buffers. Larger sources and failed
refinement retain the proxy with a notification. Only the visible crop is uploaded
to the GPU (at most 16,777,216 pixels and the device texture-size limit).
The initial preview domain is bounded to 512 megapixels, including sparse content
outside the canvas. Histogram and vectorscope statistics remain proxy-based.
WebAssembly retains the processed proxy until a browser worker path is available.

Gesture references: [Adobe Camera Raw introduction](https://helpx.adobe.com/camera-raw/desktop/get-started/overview-and-setup/introduction-camera-raw.html)
and [Julieanne Kost's Camera Raw shortcuts](https://jkost.com/blog/2022/10/225-shortcuts-tips-and-tricks-for-adobe-camera-raw.html).
