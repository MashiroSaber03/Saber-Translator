import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

from src.interfaces.lama_interface import LiteLamaInpainter


def _ceil_modulo(value: int, modulo: int = 8) -> int:
    return value if value % modulo == 0 else ((value // modulo) + 1) * modulo


class _FakeLiteLamaModel:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def predict(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        self.calls.append((image.size, mask.size))
        width, height = image.size
        padded_width = _ceil_modulo(width)
        padded_height = _ceil_modulo(height)
        result = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
        result[:, :] = [255, 0, 0]
        return Image.fromarray(result)


class LiteLamaInpainterRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_loaded = LiteLamaInpainter._loaded
        self.original_model = LiteLamaInpainter._model
        self.original_device = LiteLamaInpainter._device
        self.fake_model = _FakeLiteLamaModel()
        LiteLamaInpainter._loaded = True
        LiteLamaInpainter._model = self.fake_model
        LiteLamaInpainter._device = "cpu"
        self.cleanup_patcher = mock.patch.object(LiteLamaInpainter, "_cleanup_memory", lambda self: None)
        self.cleanup_patcher.start()
        self.inpainter = LiteLamaInpainter()

    def tearDown(self) -> None:
        self.cleanup_patcher.stop()
        LiteLamaInpainter._loaded = self.original_loaded
        LiteLamaInpainter._model = self.original_model
        LiteLamaInpainter._device = self.original_device

    def _make_image_and_mask(self, width: int, height: int) -> tuple[Image.Image, Image.Image]:
        image = Image.new("RGB", (width, height), "white")
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle((width // 4, height // 4, width // 2, height // 2), fill=255)
        return image, mask

    def test_inpaint_restores_original_size_when_non_multiple_of_eight_without_resize(self) -> None:
        image, mask = self._make_image_and_mask(11, 13)

        result = self.inpainter.inpaint(image, mask, disable_resize=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.size, image.size)
        self.assertEqual(self.fake_model.calls, [((11, 13), (11, 13))])
        self.assertEqual(result.getpixel((1, 1)), (255, 255, 255))
        self.assertEqual(result.getpixel((4, 4)), (255, 0, 0))

    def test_inpaint_resizes_large_images_without_pre_padding_before_predict(self) -> None:
        image, mask = self._make_image_and_mask(1200, 800)

        result = self.inpainter.inpaint(image, mask, disable_resize=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.size, image.size)
        self.assertEqual(self.fake_model.calls, [((1024, 682), (1024, 682))])
        self.assertEqual(result.getpixel((10, 10)), (255, 255, 255))
        self.assertEqual(result.getpixel((400, 300)), (255, 0, 0))

    def test_inpaint_restores_original_size_when_disable_resize_is_true(self) -> None:
        image, mask = self._make_image_and_mask(788, 563)

        result = self.inpainter.inpaint(image, mask, disable_resize=True)

        self.assertIsNotNone(result)
        self.assertEqual(result.size, image.size)
        self.assertEqual(self.fake_model.calls, [((788, 563), (788, 563))])
        self.assertEqual(result.getpixel((10, 10)), (255, 255, 255))
        self.assertEqual(result.getpixel((300, 200)), (255, 0, 0))
