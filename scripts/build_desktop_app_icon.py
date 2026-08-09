#!/usr/bin/env python3
"""Build rounded desktop application icons from the canonical project logo."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_LOGO = PROJECT_ROOT / "pic" / "logo.png"
ASSET_ROOT = PROJECT_ROOT / "src" / "backend_v2" / "desktop" / "assets"
PNG_OUTPUT = ASSET_ROOT / "app-icon.png"
ICO_OUTPUT = ASSET_ROOT / "app-icon.ico"
ICON_SIZE = 1024
CORNER_RADIUS = 224
SUPERSAMPLE = 4
ICO_SIZES = ((16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256))


def build_icon() -> None:
    with Image.open(SOURCE_LOGO) as source:
        artwork = ImageOps.fit(
            source.convert("RGB"),
            (ICON_SIZE, ICON_SIZE),
            method=Image.Resampling.LANCZOS,
        ).convert("RGBA")

    mask_size = ICON_SIZE * SUPERSAMPLE
    mask = Image.new("L", (mask_size, mask_size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, mask_size - 1, mask_size - 1),
        radius=CORNER_RADIUS * SUPERSAMPLE,
        fill=255,
    )
    mask = mask.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
    artwork.putalpha(mask)

    alpha = artwork.getchannel("A")
    if alpha.getpixel((0, 0)) != 0 or alpha.getpixel((ICON_SIZE // 2, ICON_SIZE // 2)) != 255:
        raise ValueError("rounded icon alpha validation failed")

    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    artwork.save(PNG_OUTPUT, format="PNG", optimize=True)
    artwork.save(ICO_OUTPUT, format="ICO", sizes=ICO_SIZES)
    print(f"PNG: {PNG_OUTPUT}")
    print(f"ICO: {ICO_OUTPUT}")


if __name__ == "__main__":
    build_icon()
