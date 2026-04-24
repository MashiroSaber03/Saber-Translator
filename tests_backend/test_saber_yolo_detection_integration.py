import base64
import io
import os
import sys
import unittest
from unittest import mock

import numpy as np
from flask import Flask
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.core import detection as detection_module
from src.core.detector.data_types import DetectionResult, TextBlock, TextLine
from src.app.api.translation.parallel_routes import parallel_bp


def make_line(x1: int, y1: int, x2: int, y2: int) -> TextLine:
    return TextLine(
        pts=np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        ),
        confidence=0.95,
    )


class SaberYoloDetectionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (160, 120), "white")
        self.app = Flask(__name__)
        self.app.register_blueprint(parallel_bp)
        self.client = self.app.test_client()

    def test_get_bubble_detection_result_applies_saber_yolo_refinement(self) -> None:
        left_line = make_line(10, 20, 45, 60)
        right_line = make_line(85, 20, 120, 60)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line, right_line])],
            raw_lines=[left_line, right_line],
        )
        refined_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line]), TextBlock(lines=[right_line])],
            raw_lines=[left_line, right_line],
        )

        with mock.patch.object(detection_module, "detect", return_value=merged_result) as detect_mock, \
             mock.patch.object(
                 detection_module,
                 "apply_saber_yolo_refinement",
                 return_value=refined_result,
                 create=True,
             ) as refine_mock:
            result = detection_module.get_bubble_detection_result(self.image, detector_type="default")

        self.assertEqual(detect_mock.call_count, 1)
        self.assertEqual(refine_mock.call_count, 1)
        self.assertEqual(len(result["coords"]), 2)

    def test_auto_direction_detection_uses_refined_blocks(self) -> None:
        horizontal_line = make_line(10, 20, 60, 40)
        vertical_line = make_line(100, 10, 120, 70)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[horizontal_line, vertical_line])],
            raw_lines=[horizontal_line, vertical_line],
        )
        refined_result = DetectionResult(
            blocks=[TextBlock(lines=[horizontal_line]), TextBlock(lines=[vertical_line])],
            raw_lines=[horizontal_line, vertical_line],
        )

        with mock.patch.object(detection_module, "detect", return_value=merged_result) as detect_mock, \
             mock.patch.object(
                 detection_module,
                 "apply_saber_yolo_refinement",
                 return_value=refined_result,
                 create=True,
             ) as refine_mock:
            result = detection_module.get_bubble_detection_result_with_auto_directions(
                self.image,
                detector_type="default",
            )

        self.assertEqual(detect_mock.call_count, 1)
        self.assertEqual(refine_mock.call_count, 1)
        self.assertEqual(result["auto_directions"], ["h", "v"])

    def test_parallel_detect_response_shape_is_unchanged(self) -> None:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mocked_result = {
            "coords": [(10, 20, 30, 40)],
            "angles": [0.0],
            "polygons": [[[10, 20], [30, 20], [30, 40], [10, 40]]],
            "auto_directions": ["h"],
            "raw_mask": None,
            "textlines_per_bubble": [[{"polygon": [[10, 20], [30, 20], [30, 40], [10, 40]], "direction": "h"}]],
        }

        with mock.patch(
            "src.app.api.translation.parallel_routes.get_bubble_detection_result_with_auto_directions",
            return_value=mocked_result,
        ):
            response = self.client.post("/api/parallel/detect", json={"image": image_base64})

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(
            set(data.keys()),
            {"success", "bubble_coords", "bubble_angles", "bubble_polygons", "auto_directions", "raw_mask", "textlines_per_bubble"},
        )
        self.assertEqual(data["bubble_coords"], [[10, 20, 30, 40]])

    def test_parallel_detect_forwards_saber_refine_toggle(self) -> None:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mocked_result = {
            "coords": [],
            "angles": [],
            "polygons": [],
            "auto_directions": [],
            "raw_mask": None,
            "textlines_per_bubble": [],
        }

        with mock.patch(
            "src.app.api.translation.parallel_routes.get_bubble_detection_result_with_auto_directions",
            return_value=mocked_result,
        ) as detect_result_mock:
            response = self.client.post(
                "/api/parallel/detect",
                json={"image": image_base64, "enable_saber_yolo_refine": False},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(detect_result_mock.call_args.kwargs["enable_saber_yolo_refine"], False)

    def test_parallel_detect_forwards_overlap_threshold(self) -> None:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mocked_result = {
            "coords": [],
            "angles": [],
            "polygons": [],
            "auto_directions": [],
            "raw_mask": None,
            "textlines_per_bubble": [],
        }

        with mock.patch(
            "src.app.api.translation.parallel_routes.get_bubble_detection_result_with_auto_directions",
            return_value=mocked_result,
        ) as detect_result_mock:
            response = self.client.post(
                "/api/parallel/detect",
                json={"image": image_base64, "saber_yolo_refine_overlap_threshold": 35},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(detect_result_mock.call_args.kwargs["saber_yolo_refine_overlap_threshold"], 35)

    def test_parallel_detect_forwards_aux_yolo_settings(self) -> None:
        buffer = io.BytesIO()
        self.image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        mocked_result = {
            "coords": [],
            "angles": [],
            "polygons": [],
            "auto_directions": [],
            "raw_mask": None,
            "textlines_per_bubble": [],
        }

        with mock.patch(
            "src.app.api.translation.parallel_routes.get_bubble_detection_result_with_auto_directions",
            return_value=mocked_result,
        ) as detect_result_mock:
            response = self.client.post(
                "/api/parallel/detect",
                json={
                    "image": image_base64,
                    "enable_aux_yolo_detection": True,
                    "aux_yolo_conf_threshold": 0.55,
                    "aux_yolo_overlap_threshold": 0.2,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(detect_result_mock.call_args.kwargs["enable_aux_yolo_detection"], True)
        self.assertEqual(detect_result_mock.call_args.kwargs["aux_yolo_conf_threshold"], 0.55)
        self.assertEqual(detect_result_mock.call_args.kwargs["aux_yolo_overlap_threshold"], 0.2)

    def test_get_bubble_detection_result_does_not_force_merge_lines(self) -> None:
        fake_result = DetectionResult(blocks=[], raw_lines=[])

        with mock.patch.object(detection_module, "detect", return_value=fake_result) as detect_mock, \
             mock.patch.object(
                 detection_module,
                 "apply_saber_yolo_refinement",
                 return_value=fake_result,
                 create=True,
             ):
            detection_module.get_bubble_detection_result(self.image, detector_type="default")

        self.assertIsNone(detect_mock.call_args.kwargs.get("merge_lines"))


if __name__ == "__main__":
    unittest.main()
