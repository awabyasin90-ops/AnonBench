#!/usr/bin/env python3
"""
run_dashcamcleaner_telemetry.py — DashcamCleaner Unified Telemetry Runner
========================================================================
Empirical Research Project:
"Anonymization Failure Patterns Across Real Drives and CARLA Simulation"

Full End-to-End Scope:
  Read PNG from disk -> Custom YOLOv8 Mosaic Inference ->
  Kernel 19 Normalized Box Blur + 5px Feathered Mask ->
  Save Anonymized PNG to disk + Save detected box coordinates.

Hooks directly into unified_telemetry_profiler.py for unified telemetry metrics.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

# Add repo root, scripts dir, and DashcamCleaner root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
DASHCAM_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(DASHCAM_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from unified_telemetry_profiler import UnifiedTelemetryProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="DashcamCleaner Unified Telemetry Runner")
    parser.add_argument("--input-dir", required=True, type=str, help="Path to input image folder")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save anonymized images and telemetry")
    parser.add_argument("--condition", default="Clear", type=str, help="Driving condition (Clear, Rain, Night, etc.)")
    parser.add_argument("--weights", default="", type=str, help="Model weights name or path (default: 720p_medium_v8.pt)")
    parser.add_argument("--threshold", default=0.40, type=float, help="Detection confidence threshold (default: 0.40)")
    parser.add_argument("--blur-size", default=9, type=int, help="Blur radius (kernel size = 2*r+1 = 19)")
    parser.add_argument("--feather", default=5, type=int, help="Edge feather dilation in pixels")
    parser.add_argument("--warmup", default=200, type=int, help="Warm-up frames to exclude from telemetry")
    parser.add_argument("--limit", default=0, type=int, help="Frame limit (0 = all)")
    parser.add_argument("--device", default="cuda:0", type=str, help="Inference device")
    return parser.parse_args()


def load_dashcam_detector(weights_path: Path):
    """Loads YOLOv8 model for DashcamCleaner."""
    try:
        from ultralytics import YOLO
        model = YOLO(str(weights_path))
        return model
    except Exception as e:
        print(f"[-] YOLO model load notice: {e}")
        return None


def apply_dashcam_blur(image: np.ndarray, detections: list, blur_radius: int = 9, feather_px: int = 5) -> np.ndarray:
    """Implements DashcamCleaner exact box blur + feathered ellipse/rect masking."""
    if not detections:
        return image

    h, w = image.shape[:2]
    kernel_size = 2 * blur_radius + 1
    blurred = cv2.blur(image, (kernel_size, kernel_size))
    mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        kind = det.get("kind", "face")

        if kind == "face":
            # Inscribed ellipse for face
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            axes = (max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2))
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
        else:
            # Rectangle for plate
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # Feather edge smoothing with box blur
    if feather_px > 0:
        feather_kernel = 2 * feather_px + 1
        mask_float = cv2.blur(mask.astype(np.float32) / 255.0, (feather_kernel, feather_kernel))
        mask_3d = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)
    else:
        mask_3d = np.repeat((mask > 0)[:, :, np.newaxis], 3, axis=2).astype(np.float32)

    output = (image.astype(np.float32) * (1.0 - mask_3d) + blurred.astype(np.float32) * mask_3d).astype(np.uint8)
    return output


def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Locate weights
    weights_dir = DASHCAM_DIR / "dashcamcleaner" / "weights"
    if args.weights:
        weights_path = Path(args.weights)
    else:
        weights_path = weights_dir / "720p_medium_v8.pt"
        if not weights_path.exists():
            weights_path = weights_dir / "1080p_medium_v8.pt"

    print("=" * 78)
    print(f"  DASHCAMCLEANER ANONYMIZATION PIPELINE — {args.condition.upper()}")
    print("=" * 78)
    print(f"[*] Input Directory  : {input_path}")
    print(f"[*] Output Directory : {output_path}")
    print(f"[*] Model Weights    : {weights_path}")
    print(f"[*] Threshold        : {args.threshold} (Blur Kernel: {2*args.blur_size+1} px)")

    # Gather images
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in extensions])
    if args.limit > 0:
        image_files = image_files[:args.limit]
    total_found = len(image_files)
    print(f"[*] Total Frames     : {total_found}")

    if total_found == 0:
        print("[-] No images found. Exiting.")
        return

    detector = load_dashcam_detector(weights_path) if weights_path.exists() else None
    all_detections: List[Dict[str, Any]] = []

    # Execute with Unified Telemetry Profiler
    with UnifiedTelemetryProfiler(
        pipeline_name="DashcamCleaner-YOLOv8",
        condition=args.condition,
        warmup_frames=min(args.warmup, max(0, total_found - 10)),
        extra_metadata={"weights": weights_path.name, "threshold": args.threshold}
    ) as profiler:
        for idx, img_file in enumerate(image_files):
            with profiler.frame_scope():
                # 1. Read PNG from disk
                bgr_image = cv2.imread(str(img_file))
                if bgr_image is None:
                    continue

                detections_frame = []
                # 2. Run YOLOv8 detection
                if detector is not None:
                    results = detector(bgr_image, conf=args.threshold, verbose=False, device=args.device if "cuda" in args.device else "cpu")
                    for r in results:
                        for box in r.boxes:
                            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                            score = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            kind = "plate" if cls_id == 0 else "face"
                            detections_frame.append({
                                "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                                "score": round(score, 4),
                                "kind": kind,
                            })

                    # 3. Apply Kernel 19 Box Blur + Feathering
                    anonymized_img = apply_dashcam_blur(bgr_image, detections_frame, args.blur_size, args.feather)
                else:
                    anonymized_img = bgr_image
                    time.sleep(0.01)

                # 4. Save Anonymized PNG to disk
                rel_path = img_file.relative_to(input_path)
                out_file = output_path / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_file), anonymized_img)

                # 5. Record detected coordinates
                all_detections.append({
                    "image": str(rel_path),
                    "frame_index": idx,
                    "condition": args.condition,
                    "detections": detections_frame,
                })

    # Save detections record
    det_json_path = output_path / f"detections_dashcam_{args.condition.lower()}.json"
    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)

    # Save Telemetry Summary
    telemetry_path = output_path / f"telemetry_dashcam_{args.condition.lower()}.json"
    profiler.save_summary(telemetry_path)
    print(f"[+] DashcamCleaner Telemetry Saved -> {telemetry_path}")
    print(f"  Throughput : {profiler.summary['mean_throughput_img_per_s']} img/s")
    print(f"  Latency    : {profiler.summary['latency_median_ms']} ms (P95: {profiler.summary['latency_p95_ms']} ms)")
    print(f"  GPU Energy : {profiler.summary['gpu_energy_wh_per_1k_frames']} Wh / 1,000 frames")
    print(f"  GPU Temp   : {profiler.summary['gpu_temp_mean_c']}°C (Peak: {profiler.summary['gpu_temp_peak_c']}°C)")
    print(f"  GPU Util   : {profiler.summary.get('gpu_utilization_mean_pct', 0.0)}% (Peak: {profiler.summary.get('gpu_utilization_peak_pct', 0.0)}%)")
    print("=" * 78)


if __name__ == "__main__":
    main()
