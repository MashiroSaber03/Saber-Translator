"""Image encoding shared by single-page and packaged downloads."""

from contextlib import ExitStack
from pathlib import Path
import shutil
from typing import BinaryIO

from PIL import Image, ImageChops, ImageOps


IMAGE_FORMATS = {"jpeg": (".jpg", "image/jpeg"), "png": (".png", "image/png")}


def validate_image_format(value: object) -> str:
    if not isinstance(value, str) or value not in IMAGE_FORMATS:
        raise ValueError("imageFormat must be jpeg or png")
    return value


def write_export_image(source: Path, output: BinaryIO, image_format: str) -> None:
    """Keep dimensions; only the download is encoded, never the stored asset."""
    validate_image_format(image_format)
    with Image.open(source) as opened, ExitStack() as stack:
        # Already-compressed originals need no additional lossy generation.
        if image_format == "jpeg" and opened.format == "JPEG":
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
            return
        orientation = opened.getexif().get(274, 1)
        image = opened if orientation == 1 else ImageOps.exif_transpose(opened)
        if image is not opened:
            stack.callback(image.close)
        if image_format == "jpeg" and ("A" in image.getbands() or "transparency" in image.info):
            rgba = image.convert("RGBA")
            stack.callback(rgba.close)
            image = Image.new("RGB", rgba.size, "white")
            stack.callback(image.close)
            alpha = rgba.getchannel("A")
            stack.callback(alpha.close)
            image.paste(rgba, mask=alpha)
        elif image.mode not in ("1", "L", "LA", "P", "RGB", "RGBA", "I;16") or (image_format == "jpeg" and image.mode not in ("L", "RGB")):
            image = image.convert("RGB")
            stack.callback(image.close)

        # Equality, rather than a grayscale conversion, keeps coloured text intact.
        # Keep RGB for an embedded colour profile that describes RGB channels.
        if image.mode == "RGB" and not opened.info.get("icc_profile"):
            red, green, blue = image.split()
            for channel in (red, green, blue):
                stack.callback(channel.close)
            with ImageChops.difference(red, green) as rg, ImageChops.difference(red, blue) as rb:
                if rg.getbbox() is None and rb.getbbox() is None:
                    image = red

        # PNGs already encoded in a suitable mode gain nothing from another
        # encoder pass, which can even make a well-compressed file larger.
        if image_format == "png" and opened.format == "PNG" and image.mode == opened.mode and orientation == 1:
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, output)
            return

        options = {"icc_profile": opened.info["icc_profile"]} if opened.info.get("icc_profile") else {}
        dpi = opened.info.get("dpi")
        if isinstance(dpi, tuple) and len(dpi) == 2 and all(isinstance(v, (int, float)) and v > 0 for v in dpi):
            options["dpi"] = dpi
        if image_format == "jpeg":
            image.save(output, format="JPEG", quality=95, subsampling=0, optimize=True, **options)
        else:
            image.save(output, format="PNG", optimize=True, **options)
