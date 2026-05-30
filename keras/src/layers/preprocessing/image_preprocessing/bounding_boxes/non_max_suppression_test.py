"""Tests for non_max_suppression."""

import numpy as np

from keras.src import backend
from keras.src import testing
from keras.src.layers.preprocessing.image_preprocessing.bounding_boxes.non_max_suppression import (  # noqa: E501
    non_max_suppression,
)


class NonMaxSuppressionTest(testing.TestCase):
    def _boxes_and_scores(self):
        # Boxes 0 and 2 are identical, box 1 overlaps them heavily, box 3 is
        # far away and overlaps nothing. Scores put box 2 on top.
        boxes = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.1],
                [0.0, 0.0, 1.0, 1.0],
                [10.0, 10.0, 11.0, 11.0],
            ],
            dtype="float32",
        )
        scores = np.array([0.9, 0.75, 0.95, 0.6], dtype="float32")
        return boxes, scores

    def _np(self, x):
        return np.asarray(backend.convert_to_numpy(x))

    def test_basic_suppression(self):
        boxes, scores = self._boxes_and_scores()
        indices, num_valid = non_max_suppression(
            boxes, scores, "xyxy", max_output_size=4, iou_threshold=0.5
        )
        self.assertEqual(tuple(self._np(indices).shape), (4,))
        self.assertEqual(int(self._np(num_valid)), 2)
        # Box 2 (0.95) wins and suppresses boxes 0 and 1, box 3 survives.
        self.assertAllEqual(self._np(indices)[:2], [2, 3])

    def test_iou_threshold_keeps_overlaps(self):
        boxes, scores = self._boxes_and_scores()
        _, num_valid = non_max_suppression(
            boxes, scores, "xyxy", max_output_size=4, iou_threshold=1.0
        )
        # Nothing overlaps another box with IoU strictly above 1, so all four
        # are kept.
        self.assertEqual(int(self._np(num_valid)), 4)

    def test_score_threshold_filters(self):
        boxes, scores = self._boxes_and_scores()
        _, num_valid = non_max_suppression(
            boxes,
            scores,
            "xyxy",
            max_output_size=4,
            iou_threshold=0.5,
            score_threshold=0.8,
        )
        # Only boxes 0 (0.9) and 2 (0.95) clear 0.8, and box 2 suppresses 0.
        self.assertEqual(int(self._np(num_valid)), 1)

    def test_max_output_size_caps_results(self):
        boxes, scores = self._boxes_and_scores()
        indices, num_valid = non_max_suppression(
            boxes, scores, "xyxy", max_output_size=1, iou_threshold=1.0
        )
        self.assertEqual(tuple(self._np(indices).shape), (1,))
        self.assertEqual(int(self._np(num_valid)), 1)
        # The single kept box is the highest scoring one (box 2).
        self.assertEqual(int(self._np(indices)[0]), 2)

    def test_matches_reference(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(2, 15))
            yx = rng.uniform(0, 40, size=(n, 2)).astype("float32")
            hw = rng.uniform(1, 15, size=(n, 2)).astype("float32")
            boxes = np.concatenate([yx, yx + hw], axis=1).astype("float32")
            scores = rng.uniform(0, 1, size=(n,)).astype("float32")
            mos = int(rng.integers(1, n + 1))
            iou_t = float(rng.uniform(0.2, 0.9))
            ref_idx, ref_nv = _reference_nms(boxes, scores, mos, iou_t)
            indices, num_valid = non_max_suppression(
                boxes, scores, "xyxy", mos, iou_t
            )
            nv = int(self._np(num_valid))
            self.assertEqual(nv, ref_nv)
            self.assertAllEqual(self._np(indices)[:nv], ref_idx[:ref_nv])


def _reference_nms(boxes, scores, max_output_size, iou_threshold):
    """Independent numpy greedy NMS for cross-checking, boxes as xyxy."""
    order = np.argsort(scores)[::-1]
    b = boxes[order]
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    active = np.ones(len(scores), dtype=bool)
    selected = np.zeros((max_output_size,), dtype="int32")
    num_valid = 0
    for k in range(max_output_size):
        if not active.any():
            break
        best = int(np.argmax(active))
        selected[k] = order[best]
        num_valid += 1
        ix1 = np.maximum(x1[best], x1)
        iy1 = np.maximum(y1[best], y1)
        ix2 = np.minimum(x2[best], x2)
        iy2 = np.minimum(y2[best], y2)
        inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
        union = areas[best] + areas - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1e-8), 0.0)
        active = active & (iou <= iou_threshold)
        active[best] = False
    return selected, num_valid
