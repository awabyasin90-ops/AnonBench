"""Decode raw YOLOv8 / YOLO11 detection-head output on the GPU.

Raw head output ("output0") shape: (B, 4 + nc + extra, A)
  - columns 0..3   : cx, cy, w, h  in MODEL-INPUT pixel coordinates
  - columns 4..4+nc: per-class scores, already sigmoid-ed (0..1)
  - extra columns  : keypoints for pose models (ignored here)
  - A = number of anchors = (H/8*W/8) + (H/16*W/16) + (H/32*W/32)

Semantics copied from ultralytics non_max_suppression (multi_label=False, agnostic=False):
best class per anchor -> conf filter -> class filter -> top max_nms by conf -> per-class NMS
-> keep max_det.
"""
from __future__ import annotations

import torch
from torchvision.ops import batched_nms

MAX_NMS = 30000


def decode_yolo(out: torch.Tensor, nc: int, conf: float, iou: float,
                classes: list[int] | None, max_det: int) -> list[torch.Tensor]:
    """Returns a list (len B) of float32 tensors (N, 6) = x1, y1, x2, y2, conf, cls."""
    if out.dim() != 3 or out.shape[1] < 4 + nc:
        raise ValueError(f"unexpected YOLO output shape {tuple(out.shape)} for nc={nc}; "
                         "end-to-end/NMS-embedded exports are not supported, export with nms=False")
    out = out.float()
    cls_filter = None
    if classes is not None:
        cls_filter = torch.tensor(classes, device=out.device, dtype=torch.long)
    results = []
    for p in out:                          # p: (C, A)
        p = p.transpose(0, 1)              # (A, C)
        scores, labels = p[:, 4:4 + nc].max(1)
        keep = scores > conf
        if cls_filter is not None:
            keep &= torch.isin(labels, cls_filter)
        if not bool(keep.any()):
            results.append(torch.zeros((0, 6), device=out.device))
            continue
        p, scores, labels = p[keep], scores[keep], labels[keep]
        if scores.numel() > MAX_NMS:
            top = scores.topk(MAX_NMS).indices
            p, scores, labels = p[top], scores[top], labels[top]
        cx, cy, w, h = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        boxes = torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), 1)
        idx = batched_nms(boxes, scores, labels, iou)[:max_det]
        results.append(torch.cat((boxes[idx], scores[idx, None], labels[idx, None].float()), 1))
    return results
