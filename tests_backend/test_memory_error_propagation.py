from __future__ import annotations

from unittest import mock

import numpy as np
from PIL import Image
import pytest

from src.core import ocr_hybrid_manga_48
from src.interfaces import manga_ocr_interface
from src.interfaces.ocr_48px.interface import Model48pxOCR
from src.shared.memory_errors import is_memory_allocation_error


@pytest.mark.parametrize(
    "error",
    [
        MemoryError(),
        RuntimeError("CUDA out of memory"),
        RuntimeError("std::bad_alloc: bad allocation"),
        RuntimeError("OpenCV: Insufficient memory: Failed to allocate 4096 bytes"),
    ],
)
def test_memory_allocation_error_classifier(error: BaseException) -> None:
    assert is_memory_allocation_error(error)


def test_non_memory_model_error_is_not_classified_as_allocation_failure() -> None:
    assert not is_memory_allocation_error(RuntimeError("invalid model output"))


def test_manga_ocr_allocation_failure_is_not_converted_to_empty_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_ocr = mock.Mock(side_effect=RuntimeError("bad allocation"))
    monkeypatch.setattr(
        manga_ocr_interface,
        "get_manga_ocr_instance",
        lambda: failing_ocr,
    )
    image = Image.new("RGB", (8, 8), "white")
    try:
        with pytest.raises(RuntimeError, match="bad allocation"):
            manga_ocr_interface.recognize_japanese_text(image)
    finally:
        image.close()


def test_hybrid_ocr_does_not_reswallow_manga_memory_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ocr_hybrid_manga_48,
        "_get_mangaocr_region",
        lambda _image, _line: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        ocr_hybrid_manga_48,
        "recognize_japanese_text",
        mock.Mock(side_effect=MemoryError()),
    )
    image = Image.new("RGB", (8, 8), "white")
    try:
        with pytest.raises(MemoryError):
            ocr_hybrid_manga_48._recognize_manga_textlines(
                image,
                [
                    {
                        "polygon": [[0, 0], [7, 0], [7, 7], [0, 7]],
                        "direction": "h",
                    }
                ],
                primary_engine="manga_ocr",
                fallback_used=False,
            )
    finally:
        image.close()


def test_48px_color_allocation_failure_is_not_converted_to_empty_colors() -> None:
    handler = Model48pxOCR()
    handler.initialized = True
    handler.device = "cpu"
    handler.model = mock.Mock()
    handler.model.infer_beam_batch_tensor.side_effect = RuntimeError(
        "DefaultCPUAllocator: can't allocate memory"
    )
    image = Image.new("RGB", (16, 16), "white")
    try:
        with pytest.raises(RuntimeError, match="can't allocate memory"):
            handler.extract_colors_for_bubbles(
                image,
                [(0, 0, 16, 16)],
                None,
            )
    finally:
        image.close()
