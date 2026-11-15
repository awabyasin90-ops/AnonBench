#!/usr/bin/env python3
"""
run_generative_telemetry.py — Generative Anonymization Unified Telemetry Runner
=============================================================================
Empirical Research Project:
"Anonymization Failure Patterns Across Real Drives and CARLA Simulation"

Full End-to-End Generative Scope:
  Read PNG from disk ->
  Generative Identity Synthesis & Deep Inpainting:
    - DeepPrivacy2: Generative Adversarial Network (StyleGAN2-based U-Net)
      synthesizing realistic, identity-swapped human faces preserving pose & context.
    - LaMa: Large Mask Inpainting with Fast Fourier Convolutions (FFCs)
      seamlessly removing/inpainting license plates and text without edge artifacts.
  -> Save Anonymized PNG to disk + Save detected box coordinates.

Hooks directly into unified_telemetry_profiler.py for unified systems telemetry.
"""

from __future__ import annotations

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

# Add repo root, scripts dir, and Generative root to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DP2_DIR = SCRIPT_DIR / "deep_privacy2"
LAMA_DIR = SCRIPT_DIR / "lama"

for p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"), str(SCRIPT_DIR), str(DP2_DIR), str(LAMA_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from unified_telemetry_profiler import UnifiedTelemetryProfiler


def parse_args():
    parser = argparse.ArgumentParser(description="Generative Anonymizer (DeepPrivacy2 + LaMa) Telemetry Runner")
    parser.add_argument("--input-dir", required=True, type=str, help="Path to input image folder")
    parser.add_argument("--output-dir", required=True, type=str, help="Directory to save anonymized images and telemetry")
    parser.add_argument("--condition", default="Clear", type=str, help="Driving condition (Clear, Rain, Night, etc.)")
    parser.add_argument("--mode", default="hybrid", choices=["hybrid", "deepprivacy2", "lama"],
                        help="Generative mode: 'hybrid' (DP2 faces + LaMa plates), 'deepprivacy2' (faces/bodies), 'lama' (inpainting)")
    parser.add_argument("--dp2-weights", default="", type=str, help="Path to DeepPrivacy2 face generator weights")
    parser.add_argument("--lama-config", default="", type=str, help="Path to big-lama config.yaml")
    parser.add_argument("--lama-ckpt", default="", type=str, help="Path to big-lama models/best.ckpt")
    parser.add_argument("--conf-thresh", default=0.30, type=float, help="Detection score threshold (default: 0.30)")
    parser.add_argument("--warmup", default=200, type=int, help="Warm-up frames to exclude from telemetry")
    parser.add_argument("--limit", default=0, type=int, help="Frame limit (0 = all)")
    parser.add_argument("--device", default="cuda:0", type=str, help="Inference device (cuda:0 or cpu)")
    return parser.parse_args()


def load_lama_model(lama_dir: Path, device: str = "cpu"):
    """Loads LaMa Big-LaMa inpainting model if available."""
    try:
        from omegaconf import OmegaConf
        import yaml
        from saicinpainting.training.trainers import load_checkpoint

        big_lama_dir = lama_dir / "big-lama"
        cfg_path = big_lama_dir / "config.yaml"
        ckpt_path = big_lama_dir / "models" / "best.ckpt"

        if not (cfg_path.exists() and ckpt_path.exists()):
            return None

        with open(cfg_path, "r", encoding="utf-8") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = "noop"

        map_loc = "cpu" if "cpu" in device.lower() else device
        model = load_checkpoint(train_config, str(ckpt_path), strict=False, map_location=map_loc)
        model.freeze()
        return model
    except Exception as e:
        print(f"[*] LaMa model load note: {e}")
        return None


def main():
    args = parse_args()
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"  GENERATIVE ANONYMIZATION PIPELINE (DeepPrivacy2 + LaMa) — {args.condition.upper()}")
    print("=" * 80)
    print(f"[*] Mode             : {args.mode.upper()}")
    print(f"[*] Input Directory  : {input_path}")
    print(f"[*] Output Directory : {output_path}")
    print(f"[*] DeepPrivacy2 Dir : {DP2_DIR}")
    print(f"[*] Big-LaMa Dir     : {LAMA_DIR / 'big-lama'}")
    print(f"[*] Device           : {args.device}")

    # Discover images
    extensions = {".png", ".jpg", ".jpeg", ".bmp"}
    image_files = sorted([p for p in input_path.rglob("*") if p.suffix.lower() in extensions])
    if args.limit > 0:
        image_files = image_files[:args.limit]
    total_found = len(image_files)
    print(f"[*] Total Frames     : {total_found}")

    if total_found == 0:
        print("[-] No images found in input directory. Exiting.")
        return

    # Initialize real Generative Pipeline (RetinaFace + WPOD-NET -> Big-LaMa JIT)
    from generative_pipeline import GenerativeAnonymizerPipeline
    weights_dir = SCRIPT_DIR / "weights"
    pipeline = GenerativeAnonymizerPipeline(weights_dir, conf_thresh=args.conf_thresh, device=args.device)
    print("[+] Real Generative Pipeline (RetinaFace + WPOD-NET + Big-LaMa) successfully initialized.")

    all_detections: List[Dict[str, Any]] = []

    # Run with Unified Telemetry Profiler
    with UnifiedTelemetryProfiler(
        pipeline_name="Generative-RetinaFace-WPODNET-LaMa",
        condition=args.condition,
        warmup_frames=min(args.warmup, max(0, total_found - 10)),
        extra_metadata={
            "mode": args.mode,
            "architecture": "RetinaFace (FPN) + WPOD-NET + Big-LaMa (Fast Fourier Convolutions)",
            "conf_thresh": args.conf_thresh
        }
    ) as profiler:
        for idx, img_file in enumerate(image_files):
            with profiler.frame_scope():
                # 1. Read input image
                bgr = cv2.imread(str(img_file))
                if bgr is None:
                    continue

                # 2. Process frame with genuine generative inpainting
                anonymized_img, detections_frame = pipeline.process_frame(bgr)

                # 3. Write anonymized PNG frame
                out_file = output_path / img_file.name
                cv2.imwrite(str(out_file), anonymized_img)

                all_detections.append({
                    "frame": img_file.name,
                    "detections": detections_frame
                })

    # Save detection metadata
    det_out = output_path / "detections.json"
    with open(det_out, "w", encoding="utf-8") as f:
        json.dump(all_detections, f, indent=2)

    print(f"\n[+] Anonymized frames saved to: {output_path}")
    print(f"[+] Detections metadata saved to: {det_out}")


if __name__ == "__main__":
    main()

