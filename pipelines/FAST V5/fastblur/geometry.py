"""Geometry helpers: shape planning, letterbox (Ultralytics-compatible), inverse box mapping.

Coordinate convention everywhere: boxes are float xyxy = [x1, y1, x2, y2] in pixel-EDGE
coordinates (0 .. width, 0 .. height), exactly like Ultralytics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

PAD_VALUE = 114  # Ultralytics letterbox gray


def ceil_to(x: float, multiple: int) -> int:
    return int(math.ceil(x / multiple) * multiple)


def plan_shapes(max_hw_ratio_landscape: float, max_wh_ratio_portrait: float,
                long_side: int = 1280, multiple: int = 64) -> dict:
    """Return model input shapes [H, W] per orientation group.

    multiple=64 (not 32) so that the half-resolution vehicle input (H/2, W/2) is still
    divisible by the YOLO stride 32.
    """
    if long_side % multiple:
        raise ValueError(f"long_side {long_side} must be a multiple of {multiple}")
    land_h = min(long_side, ceil_to(long_side * max_hw_ratio_landscape, multiple))
    port_w = min(long_side, ceil_to(long_side * max_wh_ratio_portrait, multiple))
    return {
        "landscape": [land_h, long_side],
        "portrait": [long_side, port_w],
        "square": [long_side, long_side],
    }


def choose_group(h0: int, w0: int, shapes: dict, long_side: int = 1280) -> str:
    """Pick the planned shape for this image. Supports camera-group keys ('front'/'stereo')
    or orientation keys ('landscape'/'portrait'/'square').

    Raises ValueError if no configured shape can hold the image at full scale; the pipeline
    records that image as an error instead of silently shrinking it (which would hurt recall).
    """
    # 1. Camera-group keys (front vs stereo by aspect ratio: 1024/1040 ~ 0.985 vs 768/1024 ~ 0.75)
    for f_key in ("front", "front_camera"):
        if f_key in shapes and "stereo" in shapes:
            return f_key if (h0 / w0 >= 0.85) else "stereo"
    if "front" in shapes and (h0 / w0 >= 0.85):
        return "front"
    if "stereo" in shapes and (h0 / w0 < 0.85):
        return "stereo"

    # 2. Orientation keys fallback
    target_r = long_side / max(h0, w0)
    key = "landscape" if w0 >= h0 else "portrait"
    for k in (key, "square"):
        if k in shapes:
            H, W = shapes[k]
            if min(H / h0, W / w0) + 1e-9 >= target_r:
                return k
    raise ValueError(f"no configured shape fits a {w0}x{h0} image at full scale; add 'square' to shapes")


@dataclass(frozen=True)
class LetterboxParams:
    r: float        # scale factor original -> letterbox
    left: int       # pad pixels on the left
    top: int        # pad pixels on the top
    h0: int         # original height
    w0: int         # original width
    out_h: int
    out_w: int


def letterbox(img: np.ndarray, out_h: int, out_w: int,
              interp: int = cv2.INTER_LINEAR) -> tuple[np.ndarray, LetterboxParams]:
    """Resize keeping aspect ratio, pad to (out_h, out_w). Mirrors ultralytics LetterBox
    (center=True, scaleup=True, auto=False)."""
    h0, w0 = img.shape[:2]
    r = min(out_h / h0, out_w / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (out_w - new_w) / 2, (out_h - new_h) / 2
    if (new_w, new_h) != (w0, h0):
        img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    assert out.shape[:2] == (out_h, out_w), (out.shape, out_h, out_w)
    return out, LetterboxParams(r, left, top, h0, w0, out_h, out_w)


def boxes_to_original(boxes: np.ndarray, p: LetterboxParams) -> np.ndarray:
    """Map (N,4+) xyxy boxes from letterbox space back to original image space (clipped)."""
    b = boxes.astype(np.float32, copy=True)
    if len(b) == 0:
        return b
    b[:, [0, 2]] = (b[:, [0, 2]] - p.left) / p.r
    b[:, [1, 3]] = (b[:, [1, 3]] - p.top) / p.r
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, p.w0)
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, p.h0)
    return b


def expand_box(box, pad_ratio: float, w0: int, h0: int) -> tuple[int, int, int, int]:
    """Grow a box by pad_ratio of its size on every side, round OUTWARD, clip to image."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bw, bh = x2 - x1, y2 - y1
    x1 -= bw * pad_ratio
    x2 += bw * pad_ratio
    y1 -= bh * pad_ratio
    y2 += bh * pad_ratio
    return (max(0, int(math.floor(x1))), max(0, int(math.floor(y1))),
            min(w0, int(math.ceil(x2))), min(h0, int(math.ceil(y2))))
