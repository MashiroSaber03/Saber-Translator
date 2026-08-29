from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from src.core.bubble_geometry import rotated_box_polygon
from src.core.inpainting import create_bubble_mask, inpaint_bubbles


def test_rotated_box_polygon_uses_coords_center_and_clockwise_image_angle() -> None:
    assert rotated_box_polygon([10, 20, 30, 60], 0) == [
        [10, 20],
        [30, 20],
        [30, 60],
        [10, 60],
    ]
    assert rotated_box_polygon([10, 20, 30, 60], 90) == [
        [40, 30],
        [40, 50],
        [0, 50],
        [0, 30],
    ]


def test_precise_mask_is_clipped_by_rotated_bubble_geometry() -> None:
    coords = [35, 50, 85, 70]
    polygon = rotated_box_polygon(coords, 45)
    expected_region = np.zeros((120, 120), dtype=np.uint8)
    cv2.fillPoly(expected_region, [np.asarray(polygon, dtype=np.int32)], 255)
    precise_mask = expected_region.copy()
    precise_mask[2:8, 2:8] = 255

    source = Image.new("RGB", (120, 120), "white")
    try:
        result = inpaint_bubbles(
            source,
            [coords],
            method="solid",
            fill_color="#ff0000",
            bubble_polygons=[polygon],
            precise_mask=precise_mask,
            mask_dilate_size=0,
            mask_box_expand_ratio=0,
        )
        try:
            pixels = np.asarray(result)
            repaired = np.all(pixels == [255, 0, 0], axis=2)
            assert np.array_equal(repaired, expected_region > 0)
        finally:
            result.close()
    finally:
        source.close()


def test_polygon_mask_does_not_add_an_axis_aligned_border() -> None:
    coords = [35, 50, 85, 70]
    polygon = rotated_box_polygon(coords, 45)
    expected = np.full((120, 120), 255, dtype=np.uint8)
    cv2.fillPoly(expected, [np.asarray(polygon, dtype=np.int32)], 0)

    actual = create_bubble_mask(
        (120, 120, 3),
        [coords],
        [polygon],
    )

    assert np.array_equal(actual, expected)


def test_rotated_precise_mask_is_clipped_at_image_edge_without_shape_distortion() -> None:
    coords = [0, 0, 30, 10]
    polygon = rotated_box_polygon(coords, 15)
    expected_region = np.zeros((100, 100), dtype=np.uint8)
    cv2.fillPoly(expected_region, [np.asarray(polygon, dtype=np.int32)], 255)
    precise_mask = np.full((100, 100), 255, dtype=np.uint8)

    source = Image.new("RGB", (100, 100), "white")
    try:
        result = inpaint_bubbles(
            source,
            [coords],
            method="solid",
            fill_color="#ff0000",
            bubble_polygons=[polygon],
            precise_mask=precise_mask,
            mask_dilate_size=0,
            mask_box_expand_ratio=0,
        )
        try:
            pixels = np.asarray(result)
            repaired = np.all(pixels == [255, 0, 0], axis=2)
            assert np.array_equal(repaired, expected_region > 0)
        finally:
            result.close()
    finally:
        source.close()
