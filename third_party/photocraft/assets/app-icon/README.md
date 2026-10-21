# PhotoCraft app icon

A **nine-tailed kitsune with a brow diamond**: an engraving portrait, seated, looking back at the
viewer, in the Crafting Apps' owl-template framing (full-bleed field, head and body filling the
tile, tails running off the edges).

## Palette

Exactly three colours:

| Colour | Hex | Used for |
|---|---|---|
| Ink | `#0b0b0c` | line work and contours |
| Paper | `#efe9dc` | the figure |
| PhotoCraft blue (app colour) | `#2f7bf5` | the full-bleed field |

## Geometry

A 512-unit tile (`viewBox="0 0 512 512"`), rounded square with `rx=112`, no border. The macOS
renders pad it onto Apple's 824/1024 icon grid; the Windows and Linux renders crop 22 units off
each side so the figure reads at 16–48 px.

## Provenance

The owner's original drawing, made in ArtCraft (2880 px, engraving style, keyed to the palette),
vectorised with craftrules `assets/logo-options/_tools/vectorize_tile.py`. The source PNG stays in
craftrules at `assets/app-icons/photocraft/source.png`. License: see `LICENSE.txt`.

## Files

- `photocraft.svg`: canonical master (traced at 2048 px).
- `photocraft-small.svg`: lighter trace (traced at 1024 px); also the hicolor scalable icon.
- `photocraft-1024.png`: 1024 px render on the macOS grid.
- `photocraft.icns`: macOS bundle icon (`CFBundleIconFile`).
- `photocraft.ico`: Windows icon, 16–256 px, embedded in the `.exe` by `apps/photocraft/build.rs`.
- `hicolor/<size>/apps/ai.storyteller.photocraft.png` (16–512) and `hicolor/scalable/...svg`:
  Linux icon theme.

The app also sets the window icon and Wayland app ID at runtime (`apps/photocraft/src/app_icon.rs`,
`main.rs`). On Windows the taskbar shows the icon of the Start Menu shortcut that launches the
exe, so the MSI's `<Icon Id>` keeps the `.exe` extension (ICE50); `app_icon.rs` documents the
details and tests the `.ico` sizes and the WiX icon references.

## Regenerate

Replace `photocraft.svg` (and `photocraft-small.svg`), then run `packaging/icons.sh`. It needs
`resvg`, plus `iconutil` on macOS; the `.ico` is packed by `cargo xtask ico`.
