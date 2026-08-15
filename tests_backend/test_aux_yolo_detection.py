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
from src.core.detector import registry as detector_registry
from src.core.detector.aux_yolo import (
    merge_aux_yolo_lines,
    maybe_merge_with_aux_yolo,
    normalize_aux_overlap_threshold,
)
from src.core.detector.data_types import DetectionResult, TextBlock, TextLine
from src.core.detector.base import BaseTextDetector
from src.core.detector.backends.default_backend import DefaultBackend
from src.core.detector.textline_merge import build_text_block_from_lines
from src.core.large_image_detection import LargeImageDetectorWrapper
from src.utils.image_rearrange import (
    PatchInfo,
    RearrangeContext,
    merge_masks_from_patches,
)


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

    def tearDown(self) -> None:
        self.image.close()

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

    def test_aux_detection_propagates_aux_detector_failure(self) -> None:
        main_line = make_line(10, 20, 40, 60)

        with mock.patch(
            "src.core.detector.aux_yolo.detect_aux_yolo_lines",
            side_effect=RuntimeError("aux failed"),
        ), self.assertRaisesRegex(RuntimeError, "aux failed"):
            maybe_merge_with_aux_yolo(
                np.zeros((20, 20, 3), dtype=np.uint8),
                [main_line],
                detector_type="default",
                enabled=True,
                conf_threshold=0.4,
                overlap_threshold=0.1,
            )

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

    def test_aux_threshold_rejects_invalid_values_instead_of_clamping(self) -> None:
        for value in (-0.1, 1.1, float("nan"), "0.5", True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_aux_overlap_threshold(value)

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
                saber_yolo_refine_overlap_threshold=0.35,
            )

        self.assertIs(result, main_result)
        self.assertEqual(detect_mock.call_count, 1)
        self.assertEqual(saber_mock.call_count, 1)
        self.assertIs(saber_mock.call_args.args[1], main_result)
        self.assertEqual(detect_mock.call_args.kwargs["enable_aux_yolo_detection"], True)
        self.assertEqual(detect_mock.call_args.kwargs["aux_yolo_conf_threshold"], 0.55)
        self.assertEqual(detect_mock.call_args.kwargs["aux_yolo_overlap_threshold"], 0.2)

    def test_auto_direction_detection_propagates_main_detector_failure(self) -> None:
        with mock.patch.object(
            detection_module,
            "_detect_with_optional_saber_refinement",
            side_effect=RuntimeError("detector failed"),
        ), self.assertRaisesRegex(RuntimeError, "detector failed"):
            detection_module.get_bubble_detection_result_with_auto_directions(
                self.image
            )

    def test_large_image_detection_runs_aux_on_empty_patch_result(self) -> None:
        line = make_line(10, 10, 40, 40)
        fake_detector = mock.Mock()
        fake_detector.requires_merge = False
        fake_detector.detector_id = "default"
        fake_detector._detect_raw.return_value = ([], None)
        wrapper = LargeImageDetectorWrapper(detector=fake_detector, target_size=1536)

        context = RearrangeContext(
            is_rearranged=True,
            original_height=120,
            original_width=120,
            patches_info=[
                PatchInfo(
                    top=0,
                    bottom=120,
                    down_scale_ratio=1.0,
                    pad_height=0,
                    pad_width=0,
                )
            ],
        )

        with mock.patch("src.core.large_image_detection.slice_image_for_detection", return_value=([np.zeros((64, 64, 3), dtype=np.uint8)], context)), \
             mock.patch("src.core.large_image_detection.transform_textlines_to_original", return_value=[line]), \
             mock.patch("src.core.large_image_detection.maybe_merge_with_aux_yolo", return_value=[line]):
            result = wrapper._detect_with_slicing(
                np.zeros((120, 120, 3), dtype=np.uint8),
                120,
                120,
                merge_lines=False,
                edge_ratio_threshold=0.0,
                enable_aux_yolo_detection=True,
            )

        self.assertEqual(len(result.raw_lines), 1)
        self.assertEqual(result.raw_lines[0].xyxy, line.xyxy)

    def test_large_image_detection_propagates_mask_merge_failure(self) -> None:
        line = make_line(10, 10, 40, 40)
        fake_detector = mock.Mock()
        fake_detector.requires_merge = False
        fake_detector.detector_id = "default"
        fake_detector._detect_raw.return_value = (
            [line],
            np.zeros((64, 64), dtype=np.uint8),
        )
        wrapper = LargeImageDetectorWrapper(detector=fake_detector, target_size=1536)
        context = RearrangeContext(
            is_rearranged=True,
            original_height=120,
            original_width=120,
            patches_info=[
                PatchInfo(
                    top=0,
                    bottom=120,
                    down_scale_ratio=1.0,
                    pad_height=0,
                    pad_width=0,
                )
            ],
        )

        with mock.patch(
            "src.core.large_image_detection.slice_image_for_detection",
            return_value=([np.zeros((64, 64, 3), dtype=np.uint8)], context),
        ), mock.patch(
            "src.core.large_image_detection.transform_textlines_to_original",
            return_value=[line],
        ), mock.patch(
            "src.core.large_image_detection.merge_masks_from_patches",
            side_effect=RuntimeError("mask merge failed"),
        ), self.assertRaisesRegex(RuntimeError, "mask merge failed"):
            wrapper._detect_with_slicing(
                np.zeros((120, 120, 3), dtype=np.uint8),
                120,
                120,
                merge_lines=False,
                edge_ratio_threshold=0.0,
            )

    def test_large_image_registry_propagates_wrapper_failure(self) -> None:
        fake_detector = mock.Mock()
        with mock.patch.object(
            detector_registry,
            "get_detector",
            return_value=fake_detector,
        ), mock.patch(
            "src.core.large_image_detection.LargeImageDetectorWrapper.detect",
            side_effect=RuntimeError("slice failed"),
        ), self.assertRaisesRegex(RuntimeError, "slice failed"):
            detector_registry.detect(
                self.image,
                enable_large_image=True,
            )

    def test_mask_merge_preserves_patch_index_and_uint8_range(self) -> None:
        context = RearrangeContext(
            is_rearranged=True,
            original_height=6,
            original_width=2,
            patches_info=[
                PatchInfo(0, 2, 1.0, 0, 0),
                PatchInfo(2, 4, 1.0, 0, 0),
                PatchInfo(4, 6, 1.0, 0, 0),
            ],
        )

        merged = merge_masks_from_patches(
            [
                np.full((2, 2), 64, dtype=np.uint8),
                None,
                np.full((2, 2), 128, dtype=np.uint8),
            ],
            context,
        )

        self.assertIsNotNone(merged)
        np.testing.assert_array_equal(merged[:2], np.full((2, 2), 64, dtype=np.uint8))
        np.testing.assert_array_equal(merged[2:4], np.zeros((2, 2), dtype=np.uint8))
        np.testing.assert_array_equal(merged[4:], np.full((2, 2), 128, dtype=np.uint8))

    def test_raw_detector_contract_rejects_misaligned_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "掩码尺寸"):
            BaseTextDetector._validate_raw_result(
                ([make_line(0, 0, 4, 4)], np.zeros((3, 4), dtype=np.uint8)),
                4,
                4,
            )

    def test_textline_clip_refreshes_cached_geometry(self) -> None:
        line = make_line(-100, 0, 100, 10)
        self.assertEqual(line.direction, "h")
        _ = line.xyxy

        line.clip(5, 10)

        self.assertEqual(line.xyxy, (0, 0, 5, 10))
        self.assertEqual(line.direction, "v")

    def test_zero_confidence_line_keeps_zero_block_confidence(self) -> None:
        line = make_line(0, 0, 10, 10)
        line.confidence = 0.0

        block = build_text_block_from_lines([line])

        self.assertIsNotNone(block)
        self.assertEqual(block.prob, 0.0)

    def test_default_detector_applies_box_confidence_threshold(self) -> None:
        class FakeTensor:
            def __init__(self, value: np.ndarray) -> None:
                self.value = value

            def sigmoid(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        backend = object.__new__(DefaultBackend)
        backend.model = mock.Mock(
            return_value=(
                FakeTensor(np.zeros((1, 2, 2, 2), dtype=np.float32)),
                FakeTensor(np.zeros((1, 1, 4, 4), dtype=np.float32)),
            )
        )
        backend.seg_rep = mock.Mock(
            return_value=(
                [
                    np.array(
                        [
                            [[0, 0], [2, 0], [2, 2], [0, 2]],
                            [[1, 1], [3, 1], [3, 3], [1, 3]],
                        ],
                        dtype=np.int64,
                    )
                ],
                [np.array([0.69, 0.71], dtype=np.float32)],
            )
        )
        backend.box_threshold = 0.7
        backend.device = "cpu"
        backend._preprocess_image = mock.Mock(
            return_value=(np.zeros((4, 4, 3), dtype=np.uint8), 1.0, 0, 0)
        )

        lines, _mask = backend._detect_raw(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(len(lines), 1)
        self.assertAlmostEqual(lines[0].confidence, 0.71, places=2)


if __name__ == "__main__":
    unittest.main()
