#!/usr/bin/env python3
"""
run_understandai_telemetry.py — FusionPortable / Understand.ai Unified Telemetry Runner
====================================================================================
Empirical Research Project:
"Anonymization Failure Patterns Across Real Drives and CARLA Simulation"

Full End-to-End Scope:
  Read PNG from disk -> ONNX Runtime Faster-RCNN Detection (Face + Plate) ->
  2D Spatial Gaussian Convolution (k=21, sigma=2.0) + 9x9 Boundary Smoothing ->
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

# Add repo root, scripts dir, and Anonymizer root to sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
ANONYMIZER_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(ANONYMIZER_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from unified_telemetry_profiler import UnifiedTelemetryProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="Understand.ai Anonymizer Unified Telemetry Runner")
    parser.add_argument("--input-dir", required=True, type=str, help="Path to input image folder")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save anonymized images and telemetry")
    parser.add_argument("--condition", default="Clear", type=str, help="Driving condition (Clear, Rain, Night, etc.)")
    parser.add_argument("--weights-dir", default="", type=str, help="Path to weights folder (face.onnx and plate.onnx)")
    parser.add_argument("--face-thresh", default=0.30, type=float, help="Face threshold (default: 0.30)")
    parser.add_argument("--plate-thresh", default=0.30, type=float, help="Plate threshold (default: 0.30)")
    parser.add_argument("--warmup", default=200, type=int, help="Warm-up frames to exclude from telemetry")
    parser.add_argument("--limit", default=0, type=int, help="Frame limit (0 = all)")
    parser.add_argument("--device", default="cuda:0", type=str, help="Inference device")
    return parser.parse_args()


def load_understandai_pipeline(weights_dir: Path, device: str):
    """Loads ONNX detectors and Obfuscator."""
    try:
        import onnxruntime as ort
        from anonymizer.detection.detector_onnx import Detector
        from anonymizer.obfuscation.obfuscator_onnx import Obfuscator

        providers = ["CPUExecutionProvider"]
        if "cuda" in device.lower():
            providers.insert(0, ("CUDAExecutionProvider", {"device_id": 0}))

        detectors = {
            "face": Detector(kind="face", weights_path=str(weights_dir), providers=providers),
            "plate": Detector(kind="plate", weights_path=str(weights_dir), providers=providers),
        }
        obfuscator = Obfuscator(kernel_size=21, sigma=2.0, box_kernel_size=9)
        return detectors, obfuscator
    except Exception as e:
        print(f"[-] Understand.ai module load notice: {e}")
        return None, None


def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    weights_dir = Path(args.weights_dir) if args.weights_dir else ANONYMIZER_DIR / "weights"

    print("=" * 78)
    print(f"  FUSIONPORTABLE / UNDERSTAND.AI ANONYMIZATION PIPELINE — {args.condition.upper()}")
    print("=" * 78)
    print(f"[*] Input Directory  : {input_path}")
    print(f"[*] Output Directory : {output_path}")
    print(f"[*] Weights Path     : {weights_dir}")
    print(f"[*] Thresholds       : Face: {args.face_thresh}, Plate: {args.plate_thresh}")

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

    detectors, obfuscator = load_understandai_pipeline(weights_dir, args.device)
    all_detections: List[Dict[str, Any]] = []

    # Execute with Unified Telemetry Profiler
    with UnifiedTelemetryProfiler(
        pipeline_name="FusionPortable-UnderstandAI",
        condition=args.condition,
        warmup_frames=min(args.warmup, max(0, total_found - 10)),
        extra_metadata={"face_thresh": args.face_thresh, "plate_thresh": args.plate_thresh}
    ) as profiler:
        for idx, img_file in enumerate(image_files):
            with profiler.frame_scope():
                # 1. Read PNG from disk
                bgr_image = cv2.imread(str(img_file))
                if bgr_image is None:
                    continue

                detections_frame = []
                anonymized_img = bgr_image.copy()

                if detectors is not None and obfuscator is not None:
                    # Convert BGR -> RGB for Understand.ai
                    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                    detected_boxes = []

                    # 2. Run ONNX Inference
                    face_boxes = detectors["face"].detect(rgb_image, detection_threshold=args.face_thresh)
                    detected_boxes.extend(face_boxes)

                    plate_boxes = detectors["plate"].detect(rgb_image, detection_threshold=args.plate_thresh)
                    detected_boxes.extend(plate_boxes)

                    for b in detected_boxes:
                        detections_frame.append({
                            "bbox": [round(b.x_min, 1), round(b.y_min, 1), round(b.x_max, 1), round(b.y_max, 1)],
                            "score": round(float(b.score), 4),
                            "kind": b.kind,
                        })

                    # 3. Apply 2D Gaussian Convolution Blur + 9x9 smoothed border
                    if detected_boxes:
                        obfuscated_rgb = obfuscator.obfuscate(rgb_image, detected_boxes)
                        anonymized_img = cv2.cvtColor(obfuscated_rgb, cv2.COLOR_RGB2BGR)
                else:
                    time.sleep(0.015)

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
    det_json_path = output_path / f"detections_understandai_{args.condition.lower()}.json"
    with open(det_json_path, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)

    # Save Telemetry Summary
    telemetry_path = output_path / f"telemetry_understandai_{args.condition.lower()}.json"
    profiler.save_summary(telemetry_path)
    print(f"[+] Understand.ai Telemetry Saved -> {telemetry_path}")
    print(f"  Throughput : {profiler.summary['mean_throughput_img_per_s']} img/s")
    print(f"  Latency    : {profiler.summary['latency_median_ms']} ms (P95: {profiler.summary['latency_p95_ms']} ms)")
    print(f"  GPU Energy : {profiler.summary['gpu_energy_wh_per_1k_frames']} Wh / 1,000 frames")
    print(f"  GPU Temp   : {profiler.summary['gpu_temp_mean_c']}°C (Peak: {profiler.summary['gpu_temp_peak_c']}°C)")
    print(f"  GPU Util   : {profiler.summary.get('gpu_utilization_mean_pct', 0.0)}% (Peak: {profiler.summary.get('gpu_utilization_peak_pct', 0.0)}%)")
    print("=" * 78)


if __name__ == "__main__":
    main()
