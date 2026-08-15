from __future__ import annotations

from unittest import mock

import numpy as np
from PIL import Image
import pytest
import torch

from src.interfaces import lama_interface, lama_mpe_interface


def test_lama_mpe_preserves_unmasked_pixels_and_original_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class WhiteModel:
        def __call__(self, image, _mask):
            return torch.ones_like(image)

    monkeypatch.setattr(lama_mpe_interface.LamaMPEInpainter, "_loaded", True)
    monkeypatch.setattr(
        lama_mpe_interface.LamaMPEInpainter,
        "_model",
        WhiteModel(),
    )
    monkeypatch.setattr(lama_mpe_interface.LamaMPEInpainter, "_device", "cpu")
    image = np.zeros((5, 7, 3), dtype=np.uint8)
    mask = np.zeros((5, 7), dtype=np.uint8)
    mask[2, 3] = 255

    result = lama_mpe_interface.LamaMPEInpainter().inpaint(
        image,
        mask,
        disable_resize=True,
    )

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert result[2, 3].tolist() == [255, 255, 255]
    assert np.count_nonzero(result) == 3


def test_lama_mpe_allocation_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        def __call__(self, _image, _mask):
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(lama_mpe_interface.LamaMPEInpainter, "_loaded", True)
    monkeypatch.setattr(
        lama_mpe_interface.LamaMPEInpainter,
        "_model",
        FailingModel(),
    )
    monkeypatch.setattr(lama_mpe_interface.LamaMPEInpainter, "_device", "cpu")

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        lama_mpe_interface.LamaMPEInpainter().inpaint(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 255, dtype=np.uint8),
        )


def test_lama_mpe_rejects_non_finite_model_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class NonFiniteModel:
        def __call__(self, image, _mask):
            return torch.full_like(image, float("nan"))

    monkeypatch.setattr(lama_mpe_interface.LamaMPEInpainter, "_loaded", True)
    monkeypatch.setattr(
        lama_mpe_interface.LamaMPEInpainter,
        "_model",
        NonFiniteModel(),
    )
    monkeypatch.setattr(lama_mpe_interface.LamaMPEInpainter, "_device", "cpu")

    with pytest.raises(RuntimeError, match="无效张量"):
        lama_mpe_interface.LamaMPEInpainter().inpaint(
            np.zeros((8, 8, 3), dtype=np.uint8),
            np.full((8, 8), 255, dtype=np.uint8),
        )


def test_lama_mpe_rejects_malformed_inputs() -> None:
    inpainter = lama_mpe_interface.LamaMPEInpainter()
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="同尺寸"):
        inpainter.inpaint(image, np.zeros((7, 8), dtype=np.uint8))
    with pytest.raises(ValueError, match="正整数"):
        inpainter.inpaint(image, np.zeros((8, 8), dtype=np.uint8), 0)
    with pytest.raises(ValueError, match="布尔值"):
        inpainter.inpaint(
            image,
            np.zeros((8, 8), dtype=np.uint8),
            disable_resize=1,
        )


def test_selected_lama_model_failure_does_not_switch_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(lama_interface, "LAMA_MPE_AVAILABLE", True)
    selected = mock.Mock(side_effect=RuntimeError("selected model failed"))
    other = mock.Mock()
    monkeypatch.setattr(lama_interface, "_clean_with_lama_mpe", selected)
    monkeypatch.setattr(lama_interface, "_clean_with_litelama", other)

    with Image.new("RGB", (8, 8), "white") as image, Image.new(
        "L",
        (8, 8),
        255,
    ) as mask:
        with pytest.raises(RuntimeError, match="selected model failed"):
            lama_interface.lama_clean_object(
                image,
                mask,
                lama_model="lama_mpe",
            )

    selected.assert_called_once()
    other.assert_not_called()


def test_litelama_preserves_pixels_outside_the_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RedModel:
        @staticmethod
        def predict(image, _mask):
            return Image.new("RGB", image.size, "red")

    monkeypatch.setattr(lama_interface.LiteLamaInpainter, "_loaded", True)
    monkeypatch.setattr(lama_interface.LiteLamaInpainter, "_model", RedModel())
    monkeypatch.setattr(lama_interface.LiteLamaInpainter, "_device", "cpu")
    with Image.new("RGB", (8, 8), "white") as image, Image.new(
        "L",
        (8, 8),
        0,
    ) as mask:
        mask.putpixel((3, 4), 255)
        result = lama_interface.LiteLamaInpainter().inpaint(image, mask)
    try:
        assert result.getpixel((3, 4)) == (255, 0, 0)
        assert result.getpixel((0, 0)) == (255, 255, 255)
    finally:
        result.close()
