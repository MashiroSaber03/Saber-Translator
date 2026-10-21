# Color selection and screen sampling

Click the color swatch in the Type options bar or Character panel to open the full Color Picker. It offers the color field, component strip, HSB/RGB/Lab/CMYK and hex fields, and current/new swatches. Selecting characters targets that range; with no character selection, the active text layer is targeted. **OK** applies through `type.setStyle`; **Cancel** leaves the text and history unchanged. The foreground color is independent of this dialog.

The existing canvas eyedropper continues to sample the document while the Color Picker is open, including canvas zoom and pan. This works without operating-system screen-capture permission.

## Pick a color outside PhotoCraft

**Pick screen color** is available in full Color Pickers and compact color-selection popups, including fills, shapes, gradients, layer effects, guides and preferences.

On X11, Windows and macOS, clicking it takes one snapshot of the connected screens and opens a sampling viewport on each monitor. Move the crosshair over a pixel, use the wheel to change the 11 × 11 pixel loupe from 4× to 24×, then click to select. **Esc** cancels. The center pixel has a black/white outline and its sampled hex value is displayed below the loupe.

A full dialog receives the new color for inspection; **OK** is still required to apply it. Compact popups retain their normal immediate-edit behavior. Existing alpha and additive blending are preserved; float RGB widgets retain their linear RGB storage.

Screen snapshots are held in memory, are never written or transmitted by the picker, and are released after selection, cancellation, or a change of document, layer or dialog. The combined capture budget is 64 × 1024 × 1024 pixels (about 67 MP) across at most 16 displays, subject to the GPU's texture-size limit. These limits affect screen sampling only, not document dimensions.

## Platform behavior

- **Linux/X11:** a native Rust X11 client captures the composed desktop. An advertised primary-display ICC profile is converted to sRGB through PhotoCraft's color-management library. Outputs with no published profile use an sRGB fallback.
- **Linux/Wayland:** the desktop's XDG `PickColor` portal owns selection and permission UI and returns sRGB. Magnification and zoom controls depend on the portal implementation; PhotoCraft cannot capture other applications directly under Wayland.
- **Windows:** native screen capture with monitor-specific fullscreen viewports. Captured pixels currently use an sRGB fallback; HDR and arbitrary monitor-profile accuracy are not guaranteed.
- **macOS:** safe CoreGraphics bindings capture the composed screen and use the snapshot's own ICC profile when available. Screen Recording permission may be required. The existing macOS 11 deployment target is preserved. The window-list capture API is deprecated by Apple, so current macOS releases require explicit manual validation; migration to modern native sampling must also preserve older-app compatibility.
- **Web:** browsers exposing the standard EyeDropper API use the browser's native sampling/magnification UI. Browser support and zoom controls vary. Unsupported browsers retain document sampling and color fields; external screen sampling is unavailable.

The implementation follows publicly documented Photoshop behavior for the full Type Color Picker and outside-image sampling. PhotoCraft uses an explicit screen-pick action and frozen snapshots rather than Photoshop's press-and-drag gesture.

## Verification

Unit and UI integration tests cover selected text versus whole-layer color changes, Cancel/OK and undo, routing asynchronous picks to their initiating popup, alpha preservation, malformed capture bounds, negative display origins, HiDPI mapping and cancellation races. The native capture tests also cover padded pixel layouts and memory budgets.

Generate screenshots using synthetic artwork without exposing your desktop:

```sh
cargo run -p photocraft-ui-egui --example color_picker_demo -- /tmp/photocraft-color-picker
```

For reproducible native X11 capture verification, the explicitly ignored test displays an 8 × 8 synthetic swatch in an isolated Xvfb display, captures it in memory and checks the known pixel:

```sh
xvfb-run -a cargo test -p photocraft screen_capture_reads_a_known_x11_fixture_pixel -- --ignored --test-threads=1
```

Windows, macOS, Wayland portals, browser activation and mixed-DPI monitor arrangements additionally need manual validation on those environments; mock UI tests do not substitute for it.

References: [Adobe color selection](https://helpx.adobe.com/photoshop/desktop/adjust-color/choose-colors/set-foreground-and-background-colors.html), [XDG color portal](https://flatpak.github.io/xdg-desktop-portal/docs/doc-org.freedesktop.portal.Screenshot.html), [Apple image color space](https://developer.apple.com/documentation/coregraphics/cgimage/colorspace), [EyeDropper specification](https://wicg.github.io/eyedropper-api/).
