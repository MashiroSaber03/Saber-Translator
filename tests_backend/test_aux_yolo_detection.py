import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.core import detection as detection_module
from src.core.detector.aux_yolo import merge_aux_yolo_lines, maybe_merge_with_aux_yolo
from src.core.detector.data_types import DetectionResult, TextBlock, TextLine
from src.core.large_image_detection import LargeImageDetectorWrapper


def make_line(x1: int, y1: int, x2: int, y2: int) -> TextLine:
    return TextLine(
        pts=np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        ),
        confidence=0.95,
    )


class AuxYoloDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (240, 160), "white")

    def test_merge_aux_detection_adds_non_overlapping_box(self) -> None:
        main_lines = [make_line(10, 20, 40, 60)]
        aux_lines = [make_line(120, 20, 160, 60)]

        merged = merge_aux_yolo_lines(
            main_lines,
            aux_lines,
            overlap_threshold=0.1,
        )

        self.assertEqual(len(merged), 2)
        self.assertEqual(sorted(line.xyxy for line in merged), [(10, 20, 40, 60), (120, 20, 160, 60)])

    def test_merge_aux_detection_drops_overlapping_box_without_replacement(self) -> None:
        main_line = make_line(10, 20, 70, 80)
        main_lines = [main_line]
        aux_lines = [make_line(20, 30, 60, 70)]

        merged = merge_aux_yolo_lines(
            main_lines,
            aux_lines,
            overlap_threshold=0.1,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].xyxy, main_line.xyxy)

    def test_merge_aux_detection_replaces_fully_contained_smaller_main_boxes(self) -> None:
        left_line = make_line(20, 20, 40, 60)
        right_line = make_line(45, 20, 65, 60)
        main_lines = [left_line, right_line]
        aux_lines = [make_line(10, 10, 90, 90)]

        merged = merge_aux_yolo_lines(
            main_lines,
            aux_lines,
            overlap_threshold=0.1,
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].xyxy, (10, 10, 90, 90))

    def test_aux_detection_skips_when_main_detector_is_yolo(self) -> None:
        main_line = make_line(10, 20, 40, 60)

        with mock.patch("src.core.detector.aux_yolo.detect_aux_yolo_lines", side_effect=AssertionError("should not call aux detect")):
            merged = maybe_merge_with_aux_yolo(
                np.zeros((20, 20, 3), dtype=np.uint8),
                [main_line],
                detector_type="yolo",
                enabled=True,
                conf_threshold=0.4,
                overlap_threshold=0.1,
            )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].xyxy, main_line.xyxy)

    def test_aux_detection_falls_back_when_aux_detect_raises(self) -> None:
        main_line = make_line(10, 20, 40, 60)

        with mock.patch("src.core.detector.aux_yolo.detect_aux_yolo_lines", side_effect=RuntimeError("aux failed")):
            merged = maybe_merge_with_aux_yolo(
                np.zeros((20, 20, 3), dtype=np.uint8),
                [main_line],
                detector_type="default",
                enabled=True,
                conf_threshold=0.4,
                overlap_threshold=0.1,
            )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].xyxy, main_line.xyxy)

    def test_aux_detection_can_recover_from_empty_main_result(self) -> None:
        aux_line = make_line(80, 20, 120, 60)

        merged = maybe_merge_with_aux_yolo(
            np.zeros((80, 160, 3), dtype=np.uint8),
            [],
            detector_type="default",
            enabled=True,
            aux_detector=mock.Mock(_detect_raw=mock.Mock(return_value=([aux_line], None))),
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].xyxy, aux_line.xyxy)

    def test_detect_with_optional_saber_refinement_runs_aux_before_saber(self) -> None:
        main_result = DetectionResult(
            blocks=[TextBlock(lines=[make_line(10, 20, 40, 60)])],
            raw_lines=[make_line(10, 20, 40, 60)],
        )

        with mock.patch.object(detection_module, "detect", return_value=main_result) as detect_mock, \
             mock.patch.object(detection_module, "apply_saber_yolo_refinement", return_value=main_result) as saber_mock:
            result = detection_module._detect_with_optional_saber_refinement(
                self.image,
                detector_type="default",
                edge_ratio_threshold=0.0,
                merge_lines=None,
                enable_aux_yolo_detection=True,
                aux_yolo_conf_threshold=0.55,
                aux_yolo_overlap_threshold=0.2,
                enable_saber_yolo_refine=True,
                saber_yolo_refine_overlap_threshold=35,
            )

        self.assertIs(result, main_result)
        self.assertEqual(detect_mock.call_count, 1)
        self.assertEqual(saber_mock.call_count, 1)
        self.assertIs(saber_mock.call_args.args[1], main_result)
        self.assertEqual(detect_mock.call_args.kwargs["enable_aux_yolo_detection"], True)
        self.assertEqual(detect_mock.call_args.kwargs["aux_yolo_conf_threshold"], 0.55)
        self.assertEqual(detect_mock.call_args.kwargs["aux_yolo_overlap_threshold"], 0.2)

    def test_large_image_detection_runs_aux_on_empty_patch_result(self) -> None:
        line = make_line(10, 10, 40, 40)
        fake_detector = mock.Mock()
        fake_detector.requires_merge = False
        fake_detector.detector_id = "default"
        fake_detector._detect_raw.return_value = ([], None)
        wrapper = LargeImageDetectorWrapper(detector=fake_detector, target_size=1536)

        context = mock.Mock()
        context.is_rearranged = True

        with mock.patch("src.core.large_image_detection.slice_image_for_detection", return_value=([np.zeros((64, 64, 3), dtype=np.uint8)], context)), \
             mock.patch("src.core.large_image_detection.transform_textlines_to_original", return_value=[line]), \
             mock.patch("src.core.large_image_detection.maybe_merge_with_aux_yolo", return_value=[line]):
            result = wrapper._detect_with_slicing(
                np.zeros((120, 120, 3), dtype=np.uint8),
                120,
                120,
                merge_lines=False,
                edge_ratio_threshold=0.0,
                expand_ratio=0.0,
                expand_top=0.0,
                expand_bottom=0.0,
                expand_left=0.0,
                expand_right=0.0,
                enable_aux_yolo_detection=True,
            )

        self.assertEqual(len(result.raw_lines), 1)
        self.assertEqual(result.raw_lines[0].xyxy, line.xyxy)


if __name__ == "__main__":
    unittest.main()
