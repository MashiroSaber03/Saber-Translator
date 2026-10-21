# Hardware parity

> **Last reviewed:** 2026-10-10 · **Last updated:** 2026-10-10 · **Change:** major (first version: GPU, pen tablets, displays, gestures per platform against Photoshop 2026) · **Target:** Adobe Photoshop 2026

Each hardware feature Photoshop 2026 (27.11.0) uses, per platform, against PhotoCraft. Photoshop
side: its bundle (Metal and OpenCL paths, "Precise Color Management for HDR Display" and EDR
strings, Touch Bar resources, pen pressure/tilt strings) and Adobe's GPU FAQ. PhotoCraft side:
`crates/tablet`, `crates/ui-egui/src/stylus.rs`, `apps/photocraft/src/gpu_startup.rs`,
`engine/src/display_color.rs`, `apps/photocraft/src/monitor_profile.rs`.

**Summary (estimated):** ~50% ready. GPU compositing and colour-managed display are solid; pen
input is uneven by platform (Windows lacks tilt/rotation, Wayland is broken) and there is no HDR
output. Remaining: 80–150 h (gap G12). Human: tablets (Wacom, XP-Pen, Huion) on each OS and an
HDR display to verify.

## GPU

| Feature | Photoshop | PhotoCraft | Status |
|---|---|---|---|
| GPU canvas, zoom/pan, rotate view | Metal (macOS), DirectX/OpenCL (Windows) | wgpu 30: Metal, Vulkan, DX12, GL; backend selectable | done |
| GPU compositing of layers, blend modes, effects | partial (Photoshop composites mostly on CPU) | wgpu compositor, ≤ 1/255 vs CPU oracle; Multichannel and > texture-limit regions fall back to CPU | done (ahead in places) |
| GPU filters (Blur Gallery, Liquify, Smart Sharpen, Camera Raw) | yes | CPU (rayon) for filters; GPU for compose and effects | partial |
| Documents larger than the GPU texture limit | yes (tiled) | CPU fallback; tiling in progress (#49) | partial |
| CPU-only fallback when no GPU | yes, reduced | `cpu` canvas preference; fails to start without an adapter on some Linux systems (#2519, #2412) | partial |
| Old / integrated GPUs | yes | black line on Intel HD (#2426), OpenCore Macs fail (#1859), Rgba32Float panic (#2020) | partial |

## Pen tablets

| Feature | Photoshop | macOS | Windows | Linux X11 | Linux Wayland | Web |
|---|---|---|---|---|---|---|
| Pressure | yes | yes (AppKit monitor) | yes (WM_POINTER), jumpy (#2046) | yes (XInput2) | via Xwayland only; GNOME 50 pen doesn't click (#2622), KDE (#2297) | yes |
| Tilt | yes | yes | **no** | yes | no | yes |
| Barrel rotation | yes | yes (not applied to tip angle, #1301) | no | partial | no | yes (twist) |
| Eraser end | yes | yes | no (#2285) | partial | no | partial |
| WinTab driver mode | yes | – | no | – | – | – |
| Pressure curve | Windows Ink / driver | global curve in Preferences (#2195) | same | same | same | same |

## Displays and colour

| Feature | Photoshop | PhotoCraft | Status |
|---|---|---|---|
| Colour-managed canvas (document → monitor profile) | yes | yes, folded into the GPU 3D LUT | done |
| Monitor profile detection | per display, live | macOS per display at launch; others sRGB or a chosen .icc | partial |
| HDR / EDR output for 32-bit and HDR images | yes ("Precise Color Management for HDR Display") | no; 8-bit sRGB-encoded canvas texture | missing |
| High-DPI, physical 100% | yes | yes (#1943) | done |
| Multiple monitors: documents in separate OS windows | yes | yes (`extra_windows`) | done |
| Panels in separate OS windows | yes | in-app floating only (UI-217-5) | missing |

## Input devices

| Feature | Photoshop | PhotoCraft | Status |
|---|---|---|---|
| Trackpad pinch zoom, two-finger pan | yes | yes; pinch scrolls instead in 0.6.0 (#2463) | partial |
| Trackpad rotate (Rotate View) | yes | yes (`rotate_view.rs`) | done |
| Touch Bar | yes (Intel Macs) | no | missing (low value) |
| Touch screens (Windows) | yes | basic pointer only | partial |
| 3D mouse, MIDI, Surface Dial | Surface Dial on Windows | no | missing (low value) |
| Scanners (WIA / TWAIN) | WIA menu (Windows) | menu item live; no driver integration | partial |

## Revision history

| Date | Change | Summary |
|---|---|---|
| 2026-10-10 | major | First version |
