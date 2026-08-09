from __future__ import annotations

from unittest import mock

from PIL import Image
import pytest

from src.core import inpainting


def test_lama_failure_uses_the_documented_solid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inpainting, "is_lama_available", lambda: True)
    monkeypatch.setattr(
        inpainting,
        "clean_image_with_lama",
        mock.Mock(side_effect=RuntimeError("lama failed")),
    )
    source = Image.new("RGB", (8, 8), "white")
    repaired, clean = inpainting.inpaint_bubbles(
        source,
        [(1, 1, 6, 6)],
        method="lama",
        fill_color="#112233",
    )
    try:
        assert repaired.getpixel((3, 3)) == (17, 34, 51)
        assert getattr(repaired, "_lama_inpainted", False) is False
        assert clean is not None
    finally:
        repaired.close()
        if clean is not None:
            clean.close()
        source.close()


def test_lama_allocation_failure_does_not_use_solid_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(inpainting, "is_lama_available", lambda: True)
    monkeypatch.setattr(
        inpainting,
        "clean_image_with_lama",
        mock.Mock(side_effect=RuntimeError("CUDA out of memory")),
    )
    source = Image.new("RGB", (8, 8), "white")
    try:
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            inpainting.inpaint_bubbles(
                source,
                [(1, 1, 6, 6)],
                method="lama",
                fill_color="#112233",
            )
    finally:
        source.close()


def test_solid_fill_failure_is_not_reported_as_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        inpainting.ImageDraw,
        "Draw",
        mock.Mock(side_effect=RuntimeError("draw failed")),
    )
    source = Image.new("RGB", (8, 8), "white")
    try:
        with pytest.raises(RuntimeError, match="draw failed"):
            inpainting.inpaint_bubbles(
                source,
                [(1, 1, 6, 6)],
                method="solid",
                fill_color="#FFFFFF",
            )
    finally:
        source.close()


def test_unknown_inpaint_method_is_rejected() -> None:
    source = Image.new("RGB", (8, 8), "white")
    try:
        with pytest.raises(ValueError, match="不支持的修复方法"):
            inpainting.inpaint_bubbles(
                source,
                [(1, 1, 6, 6)],
                method="unknown",
            )
    finally:
        source.close()
