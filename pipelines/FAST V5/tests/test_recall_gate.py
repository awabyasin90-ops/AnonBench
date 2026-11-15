import math
import sys
from pathlib import Path
import unittest
import numpy as np


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Compute IoU between boxes1 (N, 4) and boxes2 (M, 4). Returns (N, M)."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def wilson_lower_bound(k: int, n: int, confidence: float = 0.95) -> float:
    """Wilson score lower confidence bound for binomial proportion (default 95% CI)."""
    if n <= 0:
        return 1.0 if k >= n else 0.0
    z = 1.959963984540054  # 95% two-sided
    p = k / n
    denom = 1.0 + (z**2) / n
    center = p + (z**2) / (2.0 * n)
    spread = z * math.sqrt((p * (1.0 - p) + (z**2) / (4.0 * n)) / n)
    lower = (center - spread) / denom
    return max(0.0, min(1.0, lower))


def match_boxes_one_to_one(
    ref_boxes: np.ndarray,
    cand_boxes: np.ndarray,
    iou_thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    N = len(ref_boxes)
    M = len(cand_boxes)
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(M, dtype=bool)
    if M == 0:
        return np.zeros(N, dtype=bool), np.zeros(0, dtype=bool)

    ious = box_iou(ref_boxes, cand_boxes)  # (N, M)
    recalled_mask = np.zeros(N, dtype=bool)
    cand_matched_mask = np.zeros(M, dtype=bool)

    pairs = []
    for r in range(N):
        for c in range(M):
            if ious[r, c] >= iou_thresh:
                pairs.append((float(ious[r, c]), r, c))
    pairs.sort(key=lambda x: x[0], reverse=True)

    matched_refs = set()
    matched_cands = set()
    for iou_val, r, c in pairs:
        if r not in matched_refs and c not in matched_cands:
            matched_refs.add(r)
            matched_cands.add(c)
            recalled_mask[r] = True
            cand_matched_mask[c] = True

    return recalled_mask, cand_matched_mask


class TestRecallGateMath(unittest.TestCase):
    def test_box_iou_perfect_match(self):
        b1 = np.array([[10.0, 10.0, 50.0, 50.0]])
        b2 = np.array([[10.0, 10.0, 50.0, 50.0]])
        iou = box_iou(b1, b2)
        self.assertEqual(iou.shape, (1, 1))
        self.assertAlmostEqual(float(iou[0, 0]), 1.0, places=5)

    def test_box_iou_no_overlap(self):
        b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        b2 = np.array([[20.0, 20.0, 30.0, 30.0]])
        iou = box_iou(b1, b2)
        self.assertEqual(float(iou[0, 0]), 0.0)

    def test_box_iou_partial_overlap(self):
        # b1: 0,0 to 10,10 (area 100)
        # b2: 5,0 to 15,10 (area 100)
        # intersection: 5,0 to 10,10 (area 50)
        # union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
        b1 = np.array([[0.0, 0.0, 10.0, 10.0]])
        b2 = np.array([[5.0, 0.0, 15.0, 10.0]])
        iou = box_iou(b1, b2)
        self.assertAlmostEqual(float(iou[0, 0]), 1.0 / 3.0, places=5)

    def test_box_iou_empty_inputs(self):
        b1 = np.zeros((0, 4), dtype=np.float32)
        b2 = np.array([[0.0, 0.0, 10.0, 10.0]])
        self.assertEqual(box_iou(b1, b2).shape, (0, 1))
        self.assertEqual(box_iou(b2, b1).shape, (1, 0))
        self.assertEqual(box_iou(b1, b1).shape, (0, 0))

    def test_smallest_25_percentile_split(self):
        # 8 boxes with known areas: 100, 200, 300, 400, 500, 600, 700, 800
        areas = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0]
        thresh = float(np.percentile(areas, 25))
        # 25th percentile of 8 elements is 275.0
        small_mask = [a <= thresh for a in areas]
        # Exactly the first 2 (100, 200) are <= 275.0 -> exactly 25%
        self.assertEqual(sum(small_mask), 2)
        self.assertEqual(small_mask, [True, True, False, False, False, False, False, False])

    def test_recall_matching_threshold_05(self):
        # Ref boxes: 2 boxes
        ref = np.array([
            [10.0, 10.0, 50.0, 50.0],   # box 1
            [100.0, 100.0, 140.0, 140.0] # box 2
        ])
        # Candidate boxes:
        cand = np.array([
            [10.0, 10.0, 50.0, 45.0],   # high overlap with ref 1
            [100.0, 100.0, 115.0, 115.0] # tiny overlap with ref 2
        ])
        recalled, cand_matched = match_boxes_one_to_one(ref, cand, iou_thresh=0.5)
        self.assertEqual(recalled.tolist(), [True, False])
        self.assertEqual(cand_matched.tolist(), [True, False])

    def test_one_to_one_matching_prevents_candidate_reuse(self):
        """Reviewer Finding 6: verify single candidate box CANNOT recall multiple reference boxes."""
        # Two reference faces adjacent/overlapping slightly
        ref = np.array([
            [10.0, 10.0, 50.0, 50.0],  # ref 0
            [20.0, 10.0, 60.0, 50.0],  # ref 1
        ])
        # Single large candidate box that overlaps both ref 0 and ref 1 with IoU >= 0.5
        cand = np.array([
            [12.0, 10.0, 55.0, 50.0],  # cand 0
        ])
        recalled, cand_matched = match_boxes_one_to_one(ref, cand, iou_thresh=0.5)

        # Cand 0 has higher overlap with ref 0, so ref 0 is recalled, but ref 1 is NOT recalled
        self.assertTrue(recalled[0])
        self.assertFalse(recalled[1])
        self.assertEqual(int(recalled.sum()), 1, "Candidate box reuse was not prevented!")
        self.assertTrue(cand_matched[0])

    def test_wilson_score_lower_bound(self):
        # 1000/1000 successes
        lb_1000 = wilson_lower_bound(1000, 1000)
        self.assertGreater(lb_1000, 0.995)
        self.assertLessEqual(lb_1000, 1.0)

        # 995/1000 successes
        lb_995 = wilson_lower_bound(995, 1000)
        self.assertGreater(lb_995, 0.985)
        self.assertLess(lb_995, 0.995)

        # 0/1000 successes
        lb_0 = wilson_lower_bound(0, 1000)
        self.assertAlmostEqual(lb_0, 0.0, places=4)


if __name__ == "__main__":
    unittest.main()

