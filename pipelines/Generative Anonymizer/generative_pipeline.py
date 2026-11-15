#!/usr/bin/env python3
"""
generative_pipeline.py — Real Generative Anonymization Pipeline
==============================================================
Empirical Research Project:
"Anonymization Failure Patterns Across Real Drives and CARLA Simulation"

Paradigm:
  System 5: Generative Identity Synthesis & Deep Inpainting
  - Face Detection: RetinaFace (ResNet-18 FPN, deformable anchors, CVPR 2020)
  - Plate Detection: WPOD-NET (Warped Planar Object Detection Network, ECCV 2018)
  - Generative Inpainting: Big-LaMa (Fast Fourier Convolutions, WACV 2022) via TorchScript JIT

Architecturally distinct from:
  - Systems 1 & 3: Single-stage anchor-free YOLO (YOLOv8, YOLO11, NanoPlate)
  - Systems 2 & 4: Two-stage Region Proposal Networks (Meta AI & Understand.ai Faster R-CNN)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
import torch


def classify_bin(item: Dict[str, Any]) -> str:
    """Classifies bounding box into standard PP4AV distance bins."""
    x1, y1, x2, y2 = item["box"]
    diag = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if diag < 16.0:
        return "Bin 1 (<16px, Far 20m+)"
    elif diag < 32.0:
        return "Bin 2 (16-32px, Mid 10-20m)"
    elif diag < 64.0:
        return "Bin 3 (32-64px, Std 5-10m)"
    else:
        return "Bin 4 (>64px, Close <5m)"


class RetinaFaceDetector:
    """RetinaFace (ResNet-18) face detector running natively via cv2.dnn."""

    def __init__(self, model_path: Path, conf_thresh: float = 0.30, nms_thresh: float = 0.40):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = (640, 640)
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.priors = self._generate_priors(self.input_size)
        self.var = [0.1, 0.2]

    def _generate_priors(self, size: Tuple[int, int]) -> np.ndarray:
        w, h = size
        steps = [8, 16, 32]
        min_sizes = [[16, 32], [64, 128], [256, 512]]
        priors = []
        for step, min_size in zip(steps, min_sizes):
            feat_h, feat_w = h // step, w // step
            for i in range(feat_h):
                for j in range(feat_w):
                    for ms in min_size:
                        s_kx = ms / float(w)
                        s_ky = ms / float(h)
                        cx = (j + 0.5) * step / float(w)
                        cy = (i + 0.5) * step / float(h)
                        priors.append([cx, cy, s_kx, s_ky])
        return np.array(priors, dtype=np.float32)

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            img_bgr, 1.0, self.input_size, (104.0, 117.0, 123.0), False, False
        )
        self.net.setInput(blob)
        out_names = list(self.net.getUnconnectedOutLayersNames())
        outs = self.net.forward(out_names)
        loc = outs[out_names.index("loc")][0] if "loc" in out_names else outs[0][0]
        conf = outs[out_names.index("conf")][0] if "conf" in out_names else outs[1][0]

        scores = conf[:, 1]
        mask = scores > self.conf_thresh
        if not np.any(mask):
            return []

        filt_loc = loc[mask]
        filt_priors = self.priors[mask]
        filt_scores = scores[mask]

        boxes = np.zeros_like(filt_loc)
        boxes[:, 0] = filt_priors[:, 0] + filt_loc[:, 0] * self.var[0] * filt_priors[:, 2]
        boxes[:, 1] = filt_priors[:, 1] + filt_loc[:, 1] * self.var[0] * filt_priors[:, 3]
        boxes[:, 2] = filt_priors[:, 2] * np.exp(filt_loc[:, 2] * self.var[1])
        boxes[:, 3] = filt_priors[:, 3] * np.exp(filt_loc[:, 3] * self.var[1])

        x1 = np.clip((boxes[:, 0] - boxes[:, 2] / 2.0) * orig_w, 0, orig_w)
        y1 = np.clip((boxes[:, 1] - boxes[:, 3] / 2.0) * orig_h, 0, orig_h)
        x2 = np.clip((boxes[:, 0] + boxes[:, 2] / 2.0) * orig_w, 0, orig_w)
        y2 = np.clip((boxes[:, 1] + boxes[:, 3] / 2.0) * orig_h, 0, orig_h)
        decoded_boxes = np.stack([x1, y1, x2, y2], axis=1)

        indices = cv2.dnn.NMSBoxes(
            decoded_boxes.tolist(), filt_scores.tolist(), self.conf_thresh, self.nms_thresh
        )
        if len(indices) == 0:
            return []

        results = []
        for idx in indices:
            i = int(idx)
            b = decoded_boxes[i]
            s = float(filt_scores[i])
            results.append({
                "class": "face",
                "box": [round(float(b[0]), 1), round(float(b[1]), 1), round(float(b[2]), 1), round(float(b[3]), 1)],
                "conf": round(s, 4)
            })
        return results


class WPODNetPlateDetector:
    """WPOD-NET Warped Planar Object Detection Network for vehicle license plates."""

    def __init__(self, model_path: Path, conf_thresh: float = 0.30, nms_thresh: float = 0.40):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.net = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, img_bgr: np.ndarray) -> List[Dict[str, Any]]:
        orig_h, orig_w = img_bgr.shape[:2]
        ratio = float(max(orig_h, orig_w)) / min(orig_h, orig_w)
        side = int(ratio * 288.0)
        bound_dim = min(side + (side % 16), 608)

        scale = float(bound_dim) / max(orig_h, orig_w)
        new_w = max(16, int(orig_w * scale) - (int(orig_w * scale) % 16))
        new_h = max(16, int(orig_h * scale) - (int(orig_h * scale) % 16))

        resized = cv2.resize(img_bgr, (new_w, new_h))
        img_float = resized.astype(np.float32) / 255.0
        blob = np.expand_dims(img_float, axis=0)  # NHWC: (1, new_h, new_w, 3)

        self.net.setInput(blob)
        out = self.net.forward()  # (1, gh, gw, 8)
        if out is None or out.ndim != 4:
            return []

        probs = out[0, :, :, 0]
        affines = out[0, :, :, 2:]
        yy, xx = np.where(probs > self.conf_thresh)
        if len(yy) == 0:
            return []

        net_stride = 16.0
        unit_side = ((208.0 + 40.0) / 2.0) / net_stride
        MN = np.array([new_w / net_stride, new_h / net_stride], dtype=np.float32)

        base_corners = np.array([
            [-0.5, -0.5, 1.0],
            [ 0.5, -0.5, 1.0],
            [ 0.5,  0.5, 1.0],
            [-0.5,  0.5, 1.0]
        ], dtype=np.float32).T

        boxes = []
        scores = []
        for y, x in zip(yy, xx):
            p = float(probs[y, x])
            aff = affines[y, x].reshape((2, 3))
            aff[0, 0] = max(aff[0, 0], 0.0)
            aff[1, 1] = max(aff[1, 1], 0.0)

            pts = (aff @ base_corners) * unit_side
            pts[0, :] += float(x) + 0.5
            pts[1, :] += float(y) + 0.5

            pts_prop = pts / MN.reshape((2, 1))
            pts_orig_x = pts_prop[0, :] * orig_w
            pts_orig_y = pts_prop[1, :] * orig_h

            x1 = max(0.0, float(np.min(pts_orig_x)))
            y1 = max(0.0, float(np.min(pts_orig_y)))
            x2 = min(float(orig_w), float(np.max(pts_orig_x)))
            y2 = min(float(orig_h), float(np.max(pts_orig_y)))

            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                boxes.append([x1, y1, x2, y2])
                scores.append(p)

        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_thresh, self.nms_thresh)
        results = []
        for idx in indices:
            i = int(idx)
            b = boxes[i]
            results.append({
                "class": "plate",
                "box": [round(b[0], 1), round(b[1], 1), round(b[2], 1), round(b[3], 1)],
                "conf": round(scores[i], 4)
            })
        return results


class BigLaMaInpainter:
    """Big-LaMa Fast Fourier Convolution Inpainting Model via TorchScript JIT."""

    def __init__(self, model_path: Path, device: str = "cpu"):
        self.device = torch.device(device if (torch.cuda.is_available() and "cuda" in device) else "cpu")
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def inpaint_rois(self, img_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        if not detections:
            return img_bgr.copy()

        out_img = img_bgr.copy()
        h, w = out_img.shape[:2]

        for det in detections:
            bx1, by1, bx2, by2 = [int(v) for v in det["box"]]
            bw, bh = bx2 - bx1, by2 - by1
            if bw <= 0 or bh <= 0:
                continue

            # Context margin of 25% around the detected target
            margin_x = max(8, int(bw * 0.25))
            margin_y = max(8, int(bh * 0.25))

            rx1 = max(0, bx1 - margin_x)
            ry1 = max(0, by1 - margin_y)
            rx2 = min(w, bx2 + margin_x)
            ry2 = min(h, by2 + margin_y)

            patch = out_img[ry1:ry2, rx1:rx2]
            ph, pw = patch.shape[:2]
            if ph < 8 or pw < 8:
                continue

            # Build binary inpaint mask on the patch
            mask_patch = np.zeros((ph, pw), dtype=np.float32)
            px1 = bx1 - rx1
            py1 = by1 - ry1
            px2 = bx2 - rx1
            py2 = by2 - ry1

            if det["class"] == "face":
                cx = (px1 + px2) // 2
                cy = (py1 + py2) // 2
                ax = max(4, int((px2 - px1) * 0.55))
                ay = max(4, int((py2 - py1) * 0.65))
                cv2.ellipse(mask_patch, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
            else:
                cv2.rectangle(mask_patch, (px1, py1), (px2, py2), 1.0, -1)

            # Pad patch dimensions to multiples of 8 for LaMa FFC layers
            pad_h = (8 - ph % 8) % 8
            pad_w = (8 - pw % 8) % 8
            patch_pad = cv2.copyMakeBorder(patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            mask_pad = cv2.copyMakeBorder(mask_patch, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)

            img_t = torch.from_numpy(patch_pad).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0
            mask_t = torch.from_numpy(mask_pad).unsqueeze(0).unsqueeze(0).float().to(self.device)

            with torch.no_grad():
                inpainted_t = self.model(img_t, mask_t)

            inp_np = (inpainted_t[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            inp_unpad = inp_np[:ph, :pw]

            # Feather mask edges to seamlessly blend with background
            feather = cv2.GaussianBlur(mask_patch, (7, 7), 2.0)[:, :, None]
            blended = (inp_unpad * feather + patch * (1.0 - feather)).astype(np.uint8)
            out_img[ry1:ry2, rx1:rx2] = blended

        return out_img


class GenerativeAnonymizerPipeline:
    """Integrated System 5 Generative Pipeline."""

    def __init__(self, weights_dir: Path, conf_thresh: float = 0.30, device: str = "cpu"):
        self.weights_dir = weights_dir
        self.face_detector = RetinaFaceDetector(
            weights_dir / "retinaface_r18.onnx", conf_thresh=conf_thresh
        )
        self.plate_detector = WPODNetPlateDetector(
            weights_dir / "wpodnet.onnx", conf_thresh=conf_thresh
        )
        self.inpainter = BigLaMaInpainter(
            weights_dir / "big-lama.pt", device=device
        )

    def process_frame(self, img_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        # 1. Detect faces via RetinaFace
        faces = self.face_detector.detect(img_bgr)
        # 2. Detect license plates via WPOD-NET
        plates = self.plate_detector.detect(img_bgr)

        all_detections = faces + plates
        for d in all_detections:
            d["bin"] = classify_bin(d)

        # 3. Generative Inpainting via Big-LaMa
        anonymized_bgr = self.inpainter.inpaint_rois(img_bgr, all_detections)
        return anonymized_bgr, all_detections

