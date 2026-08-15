"""DBNet contour extraction used by the current detector backends."""

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon
import torch


class SegDetectorRepresenter:
    def __init__(self, thresh=0.3, max_candidates=1000, unclip_ratio=1.5):
        self.thresh = thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio

    def __call__(self, pred, *, height, width):
        prediction = pred[:, 0, :, :]
        segmentation = prediction > self.thresh
        boxes_batch = []
        scores_batch = []
        batch_size = prediction.size(0) if isinstance(prediction, torch.Tensor) else prediction.shape[0]
        for batch_index in range(batch_size):
            boxes, scores = self.boxes_from_bitmap(
                prediction[batch_index],
                segmentation[batch_index],
                width,
                height,
            )
            boxes_batch.append(boxes)
            scores_batch.append(scores)
        return boxes_batch, scores_batch

    def boxes_from_bitmap(self, prediction, bitmap, dest_width, dest_height):
        if isinstance(prediction, torch.Tensor):
            bitmap = bitmap.cpu().numpy()
            prediction = prediction.detach().cpu().numpy()
        if bitmap.ndim != 2:
            raise ValueError("DBNet segmentation map must be two-dimensional")
        bitmap_height, bitmap_width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap * 255).astype(np.uint8),
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        boxes = []
        scores = []
        for contour in contours[: self.max_candidates]:
            contour = contour.squeeze(1)
            points, short_side = self.get_mini_box(contour)
            if short_side < 2:
                continue
            points = np.asarray(points)
            expanded = self.unclip(points)
            if len(expanded) != 1:
                continue
            box, _ = self.get_mini_box(
                np.asarray(expanded[0]).reshape(-1, 1, 2)
            )
            box = np.asarray(box)
            score = self.box_score(prediction, contour)
            box[:, 0] = np.clip(
                np.round(box[:, 0] / bitmap_width * dest_width),
                0,
                dest_width,
            )
            box[:, 1] = np.clip(
                np.round(box[:, 1] / bitmap_height * dest_height),
                0,
                dest_height,
            )
            boxes.append(box.astype(np.int64))
            scores.append(score)
        return (
            np.asarray(boxes, dtype=np.int64).reshape(-1, 4, 2),
            np.asarray(scores, dtype=np.float32),
        )

    def unclip(self, box):
        polygon = Polygon(box)
        if polygon.length <= 0:
            return []
        distance = polygon.area * self.unclip_ratio / polygon.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        return offset.Execute(distance)

    @staticmethod
    def get_mini_box(contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda value: value[0])
        if points[1][1] > points[0][1]:
            first, fourth = 0, 1
        else:
            first, fourth = 1, 0
        if points[3][1] > points[2][1]:
            second, third = 2, 3
        else:
            second, third = 3, 2
        return (
            [points[first], points[second], points[third], points[fourth]],
            min(bounding_box[1]),
        )

    @staticmethod
    def box_score(bitmap, box):
        height, width = bitmap.shape[:2]
        box = box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, width - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, width - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, height - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, height - 1)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] -= xmin
        box[:, 1] -= ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        if bitmap.dtype == np.float16:
            bitmap = bitmap.astype(np.float32)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]
