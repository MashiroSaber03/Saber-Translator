import tempfile
import unittest
from unittest import mock

from flask import Flask
from PIL import Image

from src.app.api.system import system_bp
from src.app.api.system import tests as system_tests


class LamaApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = Flask(__name__)
        self.app.register_blueprint(system_bp)
        self.client = self.app.test_client()

    def test_test_lama_repair_accepts_post_and_uses_black_repair_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            captured = {}

            def fake_clean_image_with_lama(image, mask, lama_model='lama_mpe', disable_resize=False):
                captured["image_size"] = image.size
                captured["mask_mode"] = mask.mode
                captured["corner_pixel"] = mask.getpixel((0, 0))
                captured["center_pixel"] = mask.getpixel((image.size[0] // 2, image.size[1] // 2))
                return Image.new("RGB", image.size, color=(255, 255, 255))

            with mock.patch.object(system_tests, "get_debug_dir", return_value=temp_dir), \
                 mock.patch.object(system_tests, "LAMA_AVAILABLE", True), \
                 mock.patch.object(system_tests, "clean_image_with_lama", side_effect=fake_clean_image_with_lama):
                response = self.client.post("/api/test_lama_repair")

            self.assertEqual(response.status_code, 200)
            payload = response.get_json()
            self.assertTrue(payload["success"])
            self.assertEqual(captured["mask_mode"], "L")
            self.assertEqual(captured["corner_pixel"], 255)
            self.assertEqual(captured["center_pixel"], 0)
