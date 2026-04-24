import os
import sys
import unittest
from unittest import mock

import numpy as np
from PIL import Image


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from src.core.detector.data_types import DetectionResult, TextBlock, TextLine
from src.core.detector.refinement import (
    apply_saber_yolo_refinement,
    refine_detection_result_with_reference_blocks,
)


def make_line(x1: int, y1: int, x2: int, y2: int) -> TextLine:
    return TextLine(
        pts=np.array(
            [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            dtype=np.int32,
        ),
        confidence=0.95,
    )


class SaberYoloRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = Image.new("RGB", (180, 120), "white")

    def test_refinement_splits_merged_block_with_two_reference_blocks(self) -> None:
        left_line = make_line(10, 20, 50, 60)
        right_line = make_line(90, 20, 130, 60)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line, right_line])],
            raw_lines=[left_line, right_line],
        )
        reference_result = DetectionResult(
            blocks=[TextBlock(lines=[make_line(8, 18, 52, 62)]), TextBlock(lines=[make_line(88, 18, 132, 62)])],
            raw_lines=[],
        )

        refined = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
        )

        self.assertEqual(len(refined.blocks), 2)
        self.assertEqual(sorted(block.xyxy for block in refined.blocks), [(10, 20, 50, 60), (90, 20, 130, 60)])

    def test_refinement_keeps_block_when_only_one_reference_block_matches(self) -> None:
        left_line = make_line(10, 20, 50, 60)
        right_line = make_line(90, 20, 130, 60)
        merged_block = TextBlock(lines=[left_line, right_line])
        merged_result = DetectionResult(blocks=[merged_block], raw_lines=[left_line, right_line])
        reference_result = DetectionResult(
            blocks=[TextBlock(lines=[make_line(5, 15, 140, 65)])],
            raw_lines=[],
        )

        refined = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
        )

        self.assertEqual(len(refined.blocks), 1)
        self.assertEqual(refined.blocks[0].xyxy, merged_block.xyxy)

    def test_refinement_rolls_back_when_a_reference_block_gets_no_lines(self) -> None:
        left_line = make_line(10, 20, 50, 30)
        right_line = make_line(90, 20, 130, 30)
        merged_block = TextBlock(lines=[left_line, right_line])
        merged_result = DetectionResult(blocks=[merged_block], raw_lines=[left_line, right_line])
        reference_result = DetectionResult(
            blocks=[TextBlock(lines=[make_line(8, 18, 52, 32)]), TextBlock(lines=[make_line(66, 18, 70, 22)])],
            raw_lines=[],
        )

        refined = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
        )

        self.assertEqual(len(refined.blocks), 1)
        self.assertEqual(refined.blocks[0].xyxy, merged_block.xyxy)

    def test_refinement_uses_raw_lines_not_reference_boxes_for_final_boxes(self) -> None:
        left_line = make_line(20, 20, 45, 50)
        right_line = make_line(90, 20, 115, 50)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line, right_line])],
            raw_lines=[left_line, right_line],
        )
        reference_result = DetectionResult(
            blocks=[TextBlock(lines=[make_line(0, 0, 70, 70)]), TextBlock(lines=[make_line(70, 0, 150, 70)])],
            raw_lines=[],
        )

        refined = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
        )

        self.assertEqual(sorted(block.xyxy for block in refined.blocks), [(20, 20, 45, 50), (90, 20, 115, 50)])

    def test_refinement_does_not_remerge_split_blocks(self) -> None:
        left_line = make_line(10, 20, 110, 50)
        right_line = make_line(20, 20, 120, 50)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line, right_line])],
            raw_lines=[left_line, right_line],
        )
        reference_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line]), TextBlock(lines=[right_line])],
            raw_lines=[],
        )

        refined = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
        )

        self.assertEqual(len(refined.blocks), 2)

    def test_refinement_respects_configurable_overlap_threshold(self) -> None:
        left_line = make_line(10, 20, 40, 60)
        right_line = make_line(70, 20, 100, 60)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line, right_line])],
            raw_lines=[left_line, right_line],
        )
        reference_result = DetectionResult(
            blocks=[
                TextBlock(lines=[make_line(8, 18, 42, 62)]),
                TextBlock(lines=[make_line(60, 18, 160, 62)]),
            ],
            raw_lines=[],
        )

        conservative = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
            reference_overlap_threshold=0.5,
        )
        aggressive = refine_detection_result_with_reference_blocks(
            merged_result,
            reference_result,
            self.image,
            right_to_left=False,
            reference_overlap_threshold=0.3,
        )

        self.assertEqual(len(conservative.blocks), 1)
        self.assertEqual(len(aggressive.blocks), 2)

    def test_apply_saber_yolo_refinement_skips_reference_detector_when_disabled(self) -> None:
        left_line = make_line(10, 20, 50, 60)
        right_line = make_line(90, 20, 130, 60)
        merged_result = DetectionResult(
            blocks=[TextBlock(lines=[left_line, right_line])],
            raw_lines=[left_line, right_line],
        )

        with mock.patch('src.core.detector.refinement.detect') as detect_mock:
            refined = apply_saber_yolo_refinement(
                self.image,
                merged_result,
                detector_type='default',
                enabled=False,
            )

        detect_mock.assert_not_called()
        self.assertIs(refined, merged_result)


if __name__ == "__main__":
    unittest.main()
