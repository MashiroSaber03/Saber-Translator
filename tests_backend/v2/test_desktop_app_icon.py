from __future__ import annotations

from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_LOGO = PROJECT_ROOT / "pic" / "logo.png"
ASSET_ROOT = PROJECT_ROOT / "src" / "backend_v2" / "desktop" / "assets"
PNG_ICON = ASSET_ROOT / "app-icon.png"
ICO_ICON = ASSET_ROOT / "app-icon.ico"
EXPECTED_ICO_SIZES = {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)}


def test_runtime_icon_preserves_logo_and_has_transparent_rounded_corners() -> None:
    with Image.open(SOURCE_LOGO) as source, Image.open(PNG_ICON) as icon:
        source_rgb = source.convert("RGB")
        rendered = icon.convert("RGBA")

    assert rendered.size == (1024, 1024)
    assert rendered.getpixel((512, 512))[:3] == source_rgb.getpixel((512, 512))
    assert rendered.getpixel((512, 512))[3] == 255
    assert all(
        rendered.getpixel(point)[3] == 0
        for point in ((0, 0), (1023, 0), (0, 1023), (1023, 1023))
    )
    assert all(
        rendered.getpixel(point)[3] == 255
        for point in ((512, 0), (0, 512), (1023, 512), (512, 1023))
    )


def test_windows_icon_contains_all_required_sizes_and_alpha() -> None:
    with Image.open(ICO_ICON) as icon:
        assert icon.ico.sizes() == EXPECTED_ICO_SIZES
        largest = icon.ico.getimage((256, 256)).convert("RGBA")

    assert largest.getpixel((0, 0))[3] == 0
    assert largest.getpixel((128, 128))[3] == 255
