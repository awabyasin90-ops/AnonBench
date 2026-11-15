"""Blur functions. Runs on CPU worker threads (OpenCV releases the GIL).

method "fixed":    GaussianBlur with one fixed kernel (legacy behaviour, kernel 51).
                   Weak on large faces: a 51 px kernel does not hide an 800 px face.
method "adaptive": shrink the region so its short side is ~target_px, blur, enlarge back,
                   then a light full-size blur to remove blockiness. Strength scales with the
                   box size and cost stays constant no matter how large the box is.
"""
from __future__ import annotations

import cv2
import numpy as np

from .geometry import expand_box


def _odd(n: float) -> int:
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def blur_region(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                method: str, fixed_kernel: int, target_px: int) -> None:
    """Blur img[y1:y2, x1:x2] in place."""
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    h, w = roi.shape[:2]
    if method == "fixed":
        k = _odd(fixed_kernel)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return
    if method != "adaptive":
        raise ValueError(f"unknown blur method {method}")
    s = max(1.0, min(h, w) / float(target_px))
    sw, sh = max(1, int(round(w / s))), max(1, int(round(h / s)))
    small = cv2.resize(roi, (sw, sh), interpolation=cv2.INTER_AREA) if s > 1.0 else roi.copy()
    k_small = _odd(max(5, min(sw, sh) * 0.6))
    small = cv2.GaussianBlur(small, (k_small, k_small), 0)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) if s > 1.0 else small
    k_big = min(31, _odd(max(3, min(h, w) / 16)))
    img[y1:y2, x1:x2] = cv2.GaussianBlur(big, (k_big, k_big), 0)


def anonymize(img: np.ndarray, faces: np.ndarray, plates: np.ndarray, blur_cfg: dict) -> None:
    """faces / plates: float arrays (N,5) = x1,y1,x2,y2,conf in ORIGINAL pixel coords."""
    h0, w0 = img.shape[:2]
    method = blur_cfg["method"]
    fk = blur_cfg["fixed_kernel"]
    tp = blur_cfg["adaptive_target_px"]
    for boxes, pad in ((faces, blur_cfg["face_pad_ratio"]), (plates, blur_cfg["plate_pad_ratio"])):
        for b in boxes:
            x1, y1, x2, y2 = expand_box(b, pad, w0, h0)
            blur_region(img, x1, y1, x2, y2, method, fk, tp)
