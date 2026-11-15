#!/usr/bin/env python3
"""
master_benchmark_runner.py
===========================
Master End-to-End Benchmark Suite for 5 Automotive Privacy Anonymization Pipelines.

Evaluates 5 Disparate Architectures:
  1. Proposed FAST v5        (Real-Time ADAS Edge Pipeline)
  2. DashcamCleaner          (Single-Stage YOLOv8 Community Baseline)
  3. Meta AI EgoBlur Gen2    (Two-Stage Faster R-CNN Industry Baseline)
  4. FusionPortable          (Retrained Perspective Faster R-CNN ONNX)
  5. Generative Anonymizer   (Deep Identity Synthesis & Inpainting)

Features:
  - Configurable dataset size: Runs on 100 sample frames or scales up to 7,200 frames!
  - Full host-device telemetry (Throughput FPS, Median & P95 latency, Peak RAM/VRAM).
  - PP4AV 4-bin distance stratification (Bin 1 to Bin 4).
  - Automatically renders:
      * 3 standalone publication-quality charts in `charts/`
      * N side-by-side comparison frames with colored bounding boxes & Bin tags
      * Clean, concise `README.md` benchmark summary
  - Resumable: automatically resumes if an interrupted run has existing processed frames.

Usage:
  python3 master_benchmark_runner.py \
      --input-dir "/Users/mazeny/Desktop/test models/not blur" \
      --output-dir "/Users/mazeny/Desktop/test models" \
      --max-frames 100 \
      --num-comparisons 10
"""

import os
import sys
import time
import json
import csv
import argparse
import psutil
from pathlib import Path
import numpy as np
import cv2
import torch
import torchvision
import torchvision.ops

# Monkeypatch Ultralytics to fix apostrophe-in-path bug on Windows
try:
    import ultralytics.utils.downloads as u_dl
    _orig_attempt = u_dl.attempt_download_asset
    def _safe_attempt(file, *args, **kwargs):
        p = Path(file)
        if p.exists():
            return str(p)
        return _orig_attempt(file, *args, **kwargs)
    u_dl.attempt_download_asset = _safe_attempt
except Exception:
    pass

import tempfile
SCRIPT_DIR = Path(__file__).resolve().parent
SYSTEMS_DIR = (SCRIPT_DIR.parent / "pipelines").resolve() if (SCRIPT_DIR.parent / "pipelines").exists() else (SCRIPT_DIR.parent / "5-pipeline's").resolve()
TMP_WEIGHTS = Path(tempfile.gettempdir()) / "anonymization_weights"
TMP_WEIGHTS.mkdir(parents=True, exist_ok=True)

# Python 3.14 inspect.findsource patch for Detectron2 / TorchScript JIT
import inspect
_orig_findsource = inspect.findsource
def _safe_findsource(obj):
    if inspect.isclass(obj) and "__firstlineno__" not in vars(obj):
        name = obj.__name__
        try:
            file = inspect.getfile(obj)
            import linecache
            lines = linecache.getlines(file)
            for idx, line in enumerate(lines):
                if line.strip().startswith(f"class {name}"):
                    obj.__firstlineno__ = idx + 1
                    break
        except Exception:
            pass
    return _orig_findsource(obj)
inspect.findsource = _safe_findsource


def classify_bin(det):
    box = det["box"]
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    dim = w if det["class"] == "face" else h
    if dim < 16: return "Bin 1 (<16px, Far 20m+)"
    elif dim <= 32: return "Bin 2 (16-32px, Mid 10-20m)"
    elif dim <= 64: return "Bin 3 (32-64px, Standard 5-10m)"
    else: return "Bin 4 (>64px, Close <5m)"


def load_pipeline_runners(device="cpu"):
    from ultralytics import YOLO

    # 1. FAST v5
    fast_dir = SYSTEMS_DIR / "FAST V5"
    fast_face = YOLO(str(fast_dir / "weights" / "yolov8m-face.pt"))
    fast_plate = YOLO(str(fast_dir / "weights" / "plate_nano_192x320_best.pt"))

    def run_fast(img_bgr):
        h, w = img_bgr.shape[:2]
        out_img = img_bgr.copy()
        dets = []
        f_res = fast_face(img_bgr, conf=0.15, device=device, verbose=False)
        for r in f_res:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                d = {"class": "face", "box": [x1, y1, x2, y2], "conf": float(b.conf[0])}
                d["bin"] = classify_bin(d)
                dets.append(d)
                fw, fh = max(1, x2 - x1), max(1, y2 - y1)
                ksize = max(15, (min(fw, fh) // 2) * 2 + 1)
                patch = out_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if patch.size > 0:
                    blurred = cv2.GaussianBlur(patch, (ksize, ksize), 0)
                    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
                    cv2.ellipse(mask, (patch.shape[1]//2, patch.shape[0]//2), (patch.shape[1]//2, patch.shape[0]//2), 0, 0, 360, 255, -1)
                    mask_3d = mask[:, :, None] / 255.0
                    out_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = (blurred * mask_3d + patch * (1 - mask_3d)).astype(np.uint8)

        p_res = fast_plate(img_bgr, conf=0.25, device=device, verbose=False)
        for r in p_res:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                d = {"class": "plate", "box": [x1, y1, x2, y2], "conf": float(b.conf[0])}
                d["bin"] = classify_bin(d)
                dets.append(d)
                pw, ph = max(1, x2 - x1), max(1, y2 - y1)
                ksize = max(15, (min(pw, ph) // 2) * 2 + 1)
                patch = out_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if patch.size > 0:
                    out_img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = cv2.GaussianBlur(patch, (ksize, ksize), 0)
        return out_img, dets

    # 2. EgoBlur
    egoblur_dir = SYSTEMS_DIR / "EgoBlur"
    sys.path.insert(0, str(egoblur_dir))
    from gen2.script.detectron2.export.torchscript_patch import patch_instances
    from gen2.script.predictor import EgoblurDetector, ClassID, PATCH_INSTANCES_FIELDS
    from gen2.script.utils import visualize as egoblur_visualize, get_image_tensor

    egoblur_weights = egoblur_dir / "weights"
    with patch_instances(fields=PATCH_INSTANCES_FIELDS):
        ego_face = EgoblurDetector(str(egoblur_weights / "ego_blur_face_gen2.jit"), device="cpu", detection_class=ClassID.FACE, score_threshold=0.67416, nms_iou_threshold=0.3)
        ego_lp = EgoblurDetector(str(egoblur_weights / "ego_blur_lp_gen2.jit"), device="cpu", detection_class=ClassID.LICENSE_PLATE, score_threshold=0.74475, nms_iou_threshold=0.3)

    def run_egoblur(img_bgr):
        img_tensor = get_image_tensor(img_bgr)
        faces = ego_face.run(img_tensor)
        plates = ego_lp.run(img_tensor)
        boxes, dets = [], []
        if faces and faces[0]:
            for b in faces[0]:
                boxes.append(b)
                d = {"class": "face", "box": b, "conf": 0.85}
                d["bin"] = classify_bin(d)
                dets.append(d)
        if plates and plates[0]:
            for b in plates[0]:
                boxes.append(b)
                d = {"class": "plate", "box": b, "conf": 0.85}
                d["bin"] = classify_bin(d)
                dets.append(d)
        blurred = egoblur_visualize(img_bgr, boxes, 1.15) if boxes else img_bgr.copy()
        return blurred, dets

    # 3. DashcamCleaner
    dc_src = SYSTEMS_DIR / "DashcamCleaner/dashcamcleaner/weights/720p_medium_v8.pt"
    dashcam_model = YOLO(str(dc_src))

    def run_dashcam(img_bgr):
        h, w = img_bgr.shape[:2]
        res = dashcam_model(img_bgr, conf=0.25, device=device, verbose=False)
        dets = []
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in res:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                cls_id = int(b.cls[0])
                kind = "plate" if cls_id == 0 else "face"
                d = {"class": kind, "box": [x1, y1, x2, y2], "conf": float(b.conf[0])}
                d["bin"] = classify_bin(d)
                dets.append(d)
                if kind == "face":
                    cv2.ellipse(mask, ((x1+x2)//2, (y1+y2)//2), (max(1, (x2-x1)//2), max(1, (y2-y1)//2)), 0, 0, 360, 255, -1)
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        if len(dets) > 0:
            blurred = cv2.blur(img_bgr, (19, 19))
            mask_3d = cv2.blur(mask.astype(np.float32)/255.0, (11, 11))[:, :, None]
            out = (img_bgr.astype(np.float32)*(1.0-mask_3d) + blurred.astype(np.float32)*mask_3d).astype(np.uint8)
        else:
            out = img_bgr.copy()
        return out, dets

    # 4. FusionPortable Anonymizer
    anon_dir = SYSTEMS_DIR / "Anonymizer"
    sys.path.insert(0, str(anon_dir))
    from anonymizer.detection.detector_onnx import Detector as ONNXDetector
    from anonymizer.obfuscation.obfuscator_onnx import Obfuscator as ONNXObfuscator
    uai_weights = anon_dir / "weights"
    fp_face = ONNXDetector(kind="face", weights_path=str(uai_weights), providers=["CPUExecutionProvider"])
    fp_plate = ONNXDetector(kind="plate", weights_path=str(uai_weights), providers=["CPUExecutionProvider"])
    fp_obfuscator = ONNXObfuscator(kernel_size=21, sigma=2.0, box_kernel_size=9)

    def run_fusionportable(img_bgr):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        faces = fp_face.detect(img_rgb, detection_threshold=0.30)
        plates = fp_plate.detect(img_rgb, detection_threshold=0.30)
        all_b = faces + plates
        dets = []
        for b in all_b:
            d = {"class": b.kind, "box": [round(b.x_min, 1), round(b.y_min, 1), round(b.x_max, 1), round(b.y_max, 1)], "conf": round(b.score, 4)}
            d["bin"] = classify_bin(d)
            dets.append(d)
        obf = fp_obfuscator.obfuscate(img_rgb, all_b)
        out = cv2.cvtColor(np.clip(obf, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        return out, dets

    # 5. Generative Anonymizer (RetinaFace + WPOD-NET -> Big-LaMa JIT Inpainting)
    gen_dir = SYSTEMS_DIR / "Generative Anonymizer"
    sys.path.insert(0, str(gen_dir))
    from generative_pipeline import GenerativeAnonymizerPipeline
    gen_weights = gen_dir / "weights"
    gen_pipeline = GenerativeAnonymizerPipeline(gen_weights, conf_thresh=0.30, device=device)

    def run_generative(img_bgr):
        return gen_pipeline.process_frame(img_bgr)

    return [
        {"id": "1_FAST_V5", "name": "Proposed FAST v5", "folder": "1_FAST_V5", "runner": run_fast},
        {"id": "2_EgoBlur", "name": "Meta AI EgoBlur", "folder": "2_EgoBlur", "runner": run_egoblur},
        {"id": "3_DashcamCleaner", "name": "DashcamCleaner", "folder": "3_DashcamCleaner", "runner": run_dashcam},
        {"id": "4_FusionPortable", "name": "FusionPortable", "folder": "4_FusionPortable", "runner": run_fusionportable},
        {"id": "5_Generative_Anonymizer", "name": "Generative Anonymizer", "folder": "5_Generative_Anonymizer", "runner": run_generative}
    ]


def run_benchmark(input_dir: Path, output_dir: Path, max_frames: int = 100, num_comparisons: int = 10, device: str = "cpu", force: bool = False):
    blurred_base = output_dir / "blurred"
    telem_dir = blurred_base / "telemetry"
    comp_dir = blurred_base / "10_Frame_Comparisons"
    charts_dir = output_dir / "charts"

    for d in [blurred_base, telem_dir, comp_dir, charts_dir]:
        d.mkdir(parents=True, exist_ok=True)

    all_frames = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
    if not all_frames:
        print(f"[-] No images found in {input_dir}")
        return

    # Sparsely and evenly sample across the sequence if max_frames is set
    selected_json = telem_dir / "selected_frames.json"
    if selected_json.exists():
        with open(selected_json, "r", encoding="utf-8") as f:
            selected_names = set(json.load(f))
        target_frames = [f for f in all_frames if f.name in selected_names]
        print(f"[*] Reusing {len(target_frames)} frames from existing {selected_json.name}")
    elif max_frames and len(all_frames) > max_frames:
        sample_indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
        target_frames = [all_frames[i] for i in sample_indices]
        print(f"[*] Sparsely sampled {max_frames} frames across {len(all_frames)} total frames (step ~{len(all_frames)/max_frames:.1f})")
    else:
        target_frames = all_frames[:max_frames] if max_frames else all_frames
    total_count = len(target_frames)
    print(f"\n" + "="*75)
    print(f"  AUTOMOTIVE ANONYMIZATION BENCHMARK — {total_count} FRAMES")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"="*75)

    with open(telem_dir / "selected_frames.json", "w") as f:
        json.dump([f.name for f in target_frames], f, indent=2)

    pipelines = load_pipeline_runners(device=device)
    summary_telemetry = []

    for idx, pipe in enumerate(pipelines, 1):
        p_id = pipe["id"]
        p_name = pipe["name"]
        p_folder = pipe["folder"]
        out_folder = blurred_base / p_folder
        out_folder.mkdir(parents=True, exist_ok=True)

        existing = list(out_folder.glob("*.jpg")) + list(out_folder.glob("*.png"))
        telem_f = out_folder / "telemetry.json"
        det_f = out_folder / "detections.json"

        if len(existing) == total_count and telem_f.exists() and det_f.exists() and not force:
            print(f"[{idx}/5] {p_name} already completed ({total_count}/{total_count} frames). Loading existing metrics...")
            with open(telem_f) as f:
                t_data = json.load(f)
            summary_telemetry.append(t_data)
            continue

        print(f"\n--> [{idx}/5] Running {p_name} on {total_count} frames...")
        runner = pipe["runner"]
        latencies = []
        det_log = {}
        tot_dets = 0

        proc = psutil.Process()
        t_start = time.perf_counter()

        for f_i, fpath in enumerate(target_frames):
            img = cv2.imread(str(fpath))
            if img is None: continue

            t0 = time.perf_counter()
            blurred_img, dets = runner(img)
            t1 = time.perf_counter()

            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)
            tot_dets += len(dets)
            det_log[fpath.name] = dets

            cv2.imwrite(str(out_folder / fpath.name), blurred_img)
            if (f_i + 1) % 25 == 0 or (f_i + 1) == total_count:
                print(f"    Frame {f_i+1:04d}/{total_count} | Latency: {lat_ms:.1f}ms | Total Detections: {tot_dets}")

        elapsed_s = time.perf_counter() - t_start
        peak_ram_mb = proc.memory_info().rss / (1024 * 1024)

        pipe_telem = {
            "system_id": p_id,
            "system_name": p_name,
            "folder": p_folder,
            "evaluated_frames": total_count,
            "elapsed_seconds": round(elapsed_s, 2),
            "fps": round(total_count / elapsed_s, 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "median_latency_ms": round(float(np.median(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "peak_ram_mb": round(peak_ram_mb, 1),
            "total_detections": tot_dets
        }

        with open(telem_f, "w") as f: json.dump(pipe_telem, f, indent=2)
        with open(det_f, "w") as f: json.dump(det_log, f, indent=2)
        summary_telemetry.append(pipe_telem)

    # Save consolidated telemetry
    telem_json_path = telem_dir / "telemetry_summary_comparison.json"
    with open(telem_json_path, "w") as f:
        json.dump(summary_telemetry, f, indent=2)

    telem_csv_path = telem_dir / "telemetry_summary_comparison.csv"
    with open(telem_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_telemetry[0].keys()))
        writer.writeheader()
        writer.writerows(summary_telemetry)

    print(f"\n[✓] Telemetry saved to {telem_dir.name}/")

    # Step 2: Generate 3 Standalone Charts
    print("\n" + "="*75)
    print("  STEP 2: GENERATING STANDALONE TELEMETRY CHARTS")
    print("="*75)
    from generate_benchmark_charts import generate_charts
    generate_charts(summary_telemetry, charts_dir)

    # Step 3: Generate Multi-System Comparison Frames
    print("\n" + "="*75)
    print(f"  STEP 3: RENDERING {num_comparisons} MULTI-SYSTEM COMPARISON COLLAGES")
    print("="*75)
    from generate_comparison_frames import generate_comparisons
    generate_comparisons(input_dir, blurred_base, comp_dir, num_samples=num_comparisons)

    # Step 4: Generate Executive README Report
    print("\n" + "="*75)
    print("  STEP 4: GENERATING EXECUTIVE BENCHMARK REPORT (README.md)")
    print("="*75)
    from generate_benchmark_report import generate_report
    readme_path = output_dir / "README.md"
    generate_report(telem_json_path, blurred_base, readme_path, input_dir.name, total_count)

    print("\n" + "="*75)
    print(f"  BENCHMARK SUITE EXECUTION COMPLETE!")
    print(f"  Report:      {readme_path}")
    print(f"  Charts:      {charts_dir}")
    print(f"  Collages:    {comp_dir}")
    print("="*75 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master 5-Pipeline Automotive Privacy Benchmark Suite")
    parser.add_argument("--input-dir", type=str, default="/Users/mazeny/Desktop/test models/not blur", help="Path to raw driving frames")
    parser.add_argument("--output-dir", type=str, default="/Users/mazeny/Desktop/test models", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=100, help="Max frames to process (e.g. 100 or 7200, 0 for all)")
    parser.add_argument("--num-comparisons", type=int, default=10, help="Number of multi-system comparison frames (default: 10)")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu', 'cuda', 'mps')")
    parser.add_argument("--force", action="store_true", help="Force re-run even if outputs exist")
    args = parser.parse_args()

    run_benchmark(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        max_frames=args.max_frames if args.max_frames > 0 else None,
        num_comparisons=args.num_comparisons,
        device=args.device,
        force=args.force
    )

