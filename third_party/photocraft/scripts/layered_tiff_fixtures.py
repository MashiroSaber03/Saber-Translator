#!/usr/bin/env python3
"""Generates the layered-TIFF fixtures under corpus/layered-tiff (gitignored) with an independent
implementation (psdtags + tifffile, BSD-3), so PhotoCraft's reader is checked against a second
writer rather than only against itself:

    layered-{le,be}-{8,16}bit.tif   2 layers (one offset, one with transparency), RLE and ZIP
                                    channels, an Intel-order and a Motorola-order variant.

Run with:  python scripts/layered_tiff_fixtures.py
Needs:     pip install psdtags tifffile numpy
"""

from pathlib import Path

import numpy
import tifffile
from psdtags import (
    PsdBlendMode,
    PsdChannel,
    PsdChannelId,
    PsdClippingType,
    PsdColorSpaceType,
    PsdCompressionType,
    PsdFormat,
    PsdKey,
    PsdLayer,
    PsdLayerFlag,
    PsdLayerMask,
    PsdLayers,
    PsdRectangle,
    PsdString,
    PsdUserMask,
    TiffImageSourceData,
)

OUT = Path(__file__).resolve().parent.parent / "corpus" / "layered-tiff"
W, H = 6, 4


def gradient(dtype, seed):
    """A deterministic W×H×4 RGBA ramp on the 8-bit grid (exact at 16 bits too)."""
    maxv = 255 if dtype == numpy.uint8 else 65535
    img = numpy.zeros((H, W, 4), dtype=numpy.uint32)
    for y in range(H):
        for x in range(W):
            img[y, x] = [(x * 40 + seed) % 256, (y * 60 + seed * 3) % 256, (x * y * 17 + seed) % 256, 255]
    img = img * (maxv // 255)
    return img.astype(dtype)


def layer(name, data, left, top, blend, opacity, compression):
    h, w = data.shape[:2]
    rect = PsdRectangle(top, left, top + h, left + w)
    channels = []
    for cid, idx in [(PsdChannelId.TRANSPARENCY_MASK, 3), (PsdChannelId.CHANNEL0, 0), (PsdChannelId.CHANNEL1, 1), (PsdChannelId.CHANNEL2, 2)]:
        channels.append(PsdChannel(channelid=cid, compression=compression, data=numpy.ascontiguousarray(data[:, :, idx])))
    return PsdLayer(
        name=name,
        rectangle=rect,
        channels=channels,
        mask=PsdLayerMask(),
        opacity=opacity,
        blendmode=blend,
        blending_ranges=(),
        clipping=PsdClippingType.BASE,
        flags=PsdLayerFlag.PHOTOSHOP5,
        info=[PsdString(PsdKey.UNICODE_LAYER_NAME, name)],
    )


def build(dtype, psdformat):
    lower = gradient(dtype, 1)
    upper = gradient(dtype, 7)[:3, :4].copy()
    # A soft transparency ramp on the upper layer.
    maxv = 255 if dtype == numpy.uint8 else 65535
    for y in range(3):
        for x in range(4):
            upper[y, x, 3] = (x * 80 % 256) * (maxv // 255)
    layers = PsdLayers(
        key=PsdKey.LAYER if dtype == numpy.uint8 else PsdKey.LAYER_16,
        has_transparency=False,
        layers=[
            layer("Background", lower, 0, 0, PsdBlendMode.NORMAL, 255, PsdCompressionType.RLE),
            layer("Upper é", upper, 1, 1, PsdBlendMode.MULTIPLY, 200, PsdCompressionType.ZIP),
        ],
    )
    usermask = PsdUserMask(colorspace=PsdColorSpaceType.RGB, components=(65535, 0, 0, 0), opacity=50)
    return TiffImageSourceData(name="Layered", psdformat=psdformat, layers=layers, usermask=usermask)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for dtype, bits in [(numpy.uint8, 8), (numpy.uint16, 16)]:
        for order, psdformat in [("le", PsdFormat.LE32BIT), ("be", PsdFormat.BE32BIT)]:
            isd = build(dtype, psdformat)
            composite = gradient(dtype, 1)
            path = OUT / f"layered-{order}-{bits}bit.tif"
            byteorder = "<" if order == "le" else ">"
            tifffile.imwrite(
                path,
                composite,
                byteorder=byteorder,
                photometric="rgb",
                extrasamples=["unassalpha"],
                resolution=(300, 300),
                resolutionunit="inch",
                extratags=[isd.tifftag()],
                metadata=None,
            )
            back = TiffImageSourceData.fromtiff(path)
            names = [l.name for l in back.layers]
            assert names == ["Background", "Upper é"], names
            print(path.name, path.stat().st_size, "bytes", names, back.psdformat)


if __name__ == "__main__":
    main()
