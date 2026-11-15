#!/usr/bin/env python3
"""
auto_tune.py  -  FAST v3 Automatic Pipeline Tuner
===================================================
Reads the hardware scan produced by scan_hardware.py and writes
a fully-tuned pipeline_v3.yaml plus a build_engines.ps1 script
with the exact trtexec commands for this machine.

  python tools/auto_tune.py [--scan run_facts/hw_scan.json]
                             [--out  config/pipeline_v3.yaml]

Output files
------------
  config/pipeline_v3.yaml    — ready-to-run pipeline config for this GPU
  config/pipeline_v3_compat.yaml — same settings but batch=4 (v2 baseline)
  build_engines.ps1          — PowerShell script that builds all 3 TRT engines
  run_facts/auto_tune.json   — full tune decision record
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
FACTS_DIR = ROOT / "run_facts"
CONFIG_DIR = ROOT / "config"


# ---------------------------------------------------------------------------
# Engine build commands for each model
# ---------------------------------------------------------------------------

def face_engine_cmd(batch: int, sm: str, weights_dir: Path) -> str:
    onnx  = weights_dir / "yolov8m-face.onnx"
    out   = weights_dir / f"yolov8m-face_fp16_b{batch}_{sm}.engine"
    cache = weights_dir / f"timing_{sm}.cache"
    return (
        f'trtexec `\n'
        f'  --onnx="{onnx}" `\n'
        f'  --saveEngine="{out}" `\n'
        f'  --fp16 `\n'
        f'  --minShapes=images:1x3x640x640 `\n'
        f'  --optShapes=images:{batch}x3x1280x1280 `\n'
        f'  --maxShapes=images:{batch}x3x1280x1280 `\n'
        f'  --timingCacheFile="{cache}" `\n'
        f'  --iterations=100\n'
    )


def vehicle_engine_cmd(batch: int, sm: str, weights_dir: Path) -> str:
    onnx  = weights_dir / "yolo11m.onnx"
    out   = weights_dir / f"yolo11m_fp16_b{batch}_{sm}.engine"
    cache = weights_dir / f"timing_{sm}.cache"
    return (
        f'trtexec `\n'
        f'  --onnx="{onnx}" `\n'
        f'  --saveEngine="{out}" `\n'
        f'  --fp16 `\n'
        f'  --minShapes=images:1x3x640x640 `\n'
        f'  --optShapes=images:{batch}x3x640x640 `\n'
        f'  --maxShapes=images:{batch}x3x640x640 `\n'
        f'  --timingCacheFile="{cache}" `\n'
        f'  --iterations=100\n'
    )


def plate_engine_cmd(batch: int, sm: str, weights_dir: Path) -> str:
    onnx  = weights_dir / "plate_nano_192x320_best.onnx"
    out   = weights_dir / f"plate_nano_fp16_b{batch}_{sm}.engine"
    cache = weights_dir / f"timing_{sm}.cache"
    return (
        f'trtexec `\n'
        f'  --onnx="{onnx}" `\n'
        f'  --saveEngine="{out}" `\n'
        f'  --fp16 `\n'
        f'  --minShapes=images:1x3x192x320 `\n'
        f'  --optShapes=images:{batch}x3x192x320 `\n'
        f'  --maxShapes=images:{batch}x3x192x320 `\n'
        f'  --timingCacheFile="{cache}" `\n'
        f'  --iterations=100\n'
    )


def onnx_export_snippet() -> str:
    return """
# ---- STEP 0: Export ONNX models (run if .onnx files don't exist yet) ----
Write-Host "Exporting ONNX models..."
python -c "from ultralytics import YOLO; YOLO('weights/yolov8m-face.pt').export(format='onnx', imgsz=1280, dynamic=True, simplify=True)"
python -c "from ultralytics import YOLO; YOLO('weights/yolo11m.pt').export(format='onnx', imgsz=640, dynamic=True, simplify=True)"
# plate_nano_192x320_best.onnx should already exist - no export needed
Write-Host "ONNX export done."
"""


# ---------------------------------------------------------------------------
# YAML writer (no external yaml dep needed for write)
# ---------------------------------------------------------------------------

def write_yaml(path: Path, scan: dict, tune: dict, compat: bool = False) -> None:
    """Write a complete pipeline_v3.yaml tuned for this hardware."""
    g    = scan["gpu"]
    cpu  = scan["cpu"]
    sm   = g["sm_str"]
    fb   = 4 if compat else tune["face_batch"]
    pb   = 24 if compat else tune["plate_batch"]
    sc   = 2 if compat else tune["stream_count"]
    shm  = 6 if compat else tune["shm_slots"]
    iq   = 16 if compat else tune["input_queue"]
    rq   = 24 if compat else tune["result_queue"]
    wt   = 5 if compat else tune["writer_threads"]
    dt   = 4 if compat else tune["decode_threads"]
    ps   = tune.get("pool_slots", max(4, shm - 1))
    w    = ROOT / "weights"
    label = "v2-compat baseline (batch=4)" if compat else "tuned for this GPU"

    input_val = '"<YOUR_DATA_DIR>"'
    output_val = '"<YOUR_OUTPUT_DIR>"' if not compat else '"<YOUR_OUTPUT_DIR_COMPAT>"'
    if path.exists():
        try:
            for l in path.read_text().splitlines():
                if l.startswith("input_dir:") and "<YOUR_DATA_DIR>" not in l:
                    input_val = l.split(":", 1)[1].strip()
                elif l.startswith("output_dir:") and "<YOUR_OUTPUT_DIR>" not in l:
                    output_val = l.split(":", 1)[1].strip()
        except Exception:
            pass

    lines = [
        f"# FAST v3 Pipeline Config — {label}",
        f"# GPU:  {g.get('name','?')}  ({g.get('vram_total_mib','?')} MiB  CC {g.get('cc_str','?')}  {g.get('sm_str','?')})",
        f"# CPU:  {cpu.get('logical_cores','?')} logical cores  RAM {cpu.get('ram_total_gib','?')} GiB",
        f"# Auto-generated by tools/auto_tune.py  at  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"# DO NOT hand-edit — re-run auto_tune.py after any hardware change.",
        "",
        f'input_dir:  {input_val}',
        f'output_dir: {output_val}',
        'run_dir:    "runs"',
        "recursive:  true",
        'extensions: [".jpg", ".jpeg"]',
        "skip_existing: true",
        "",
        "models:",
        "  face:",
        f'    engine: "{w / f"yolov8m-face_fp16_b{fb}_{sm}.engine"}"',
        f'    pt:     "{w / "yolov8m-face.pt"}"',
        "    long_side: 1280",
        "    conf: 0.15         # statutory privacy gate — do not raise",
        "    iou:  0.7",
        "    max_det: 300",
        "  vehicle:",
        f'    engine: "{w / f"yolo11m_fp16_b{fb}_{sm}.engine"}"',
        f'    pt:     "{w / "yolo11m.pt"}"',
        "    long_side: 640",
        "    conf: 0.25",
        "    iou:  0.7",
        "    classes: [2, 3, 5, 7]    # COCO: car motorcycle bus truck",
        "    max_det: 300",
        "  plate:",
        "    enabled: true",
        '    mode: "crops"',
        f'    engine: "{w / f"plate_nano_fp16_b{pb}_{sm}.engine"}"',
        f'    pt:     "{w / "plate_nano_192x320_best.pt"}"',
        "    imgsz: 320",
        "    conf: 0.30         # calibrated 98.47% recall — do not lower",
        "    iou:  0.7",
        "    max_det: 50",
        f"    max_batch: {pb}",
        "    crop_pad_ratio: 0.10",
        "    min_vehicle_px: 40 # eliminates grille false-positives",
        "    rect_192x320: false",
        "",
        f"# Generated tune: Tier {g.get('tier','?')} — {tune.get('desc','')}",
        "architecture:",
        f"  batch_size:           {fb}",
        f"  pool_slots:           {ps}",
        '  stream_strategy:      "fully_concurrent"',
        f"  compute_stream_count: {sc}",
        '  input_path:           "cpu_letterbox"',
        '  output_path:          "cpu_blur_cpu_encode"',
        '  jpeg_backend:         "turbojpeg"',
        "",
        f"# RAM budget: {cpu.get('ram_total_gib','?')} GiB system RAM",
        "threading:",
        f"  decode_threads:     {dt}",
        f"  writer_threads:     {wt}",
        f"  input_queue_depth:  {iq}",
        f"  result_queue_depth: {rq}",
        "  opencv_threads:     1",
        "  cpu_affinity:       null",
        "",
        "multiprocessing:",
        "  enabled:   true",
        f"  shm_slots: {shm}",
        "",
        "shapes:",
        "  front:     [1120, 1152]",
        "  stereo:    [864,  1152]",
        "  landscape: [1280, 1280]",
        "  portrait:  [1280,  640]",
        "  square:    [1280, 1280]",
        "",
        "blur:",
        '  method:            "fixed"',
        "  fixed_kernel:      51",
        "  adaptive_target_px: 24",
        "  face_pad_ratio:    0.0",
        "  plate_pad_ratio:   0.10",
        "",
        "jpeg_quality:           95",
        "copy_if_no_detections:  false",
        "telemetry_logging:      true",
        "telemetry_interval_ms:  100",
        "",
        "calibration:",
        "  warmup_frames:        100",
        "  rolling_window_frames: 50",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Engine build script writer
# ---------------------------------------------------------------------------

def write_build_script(path: Path, scan: dict, tune: dict) -> None:
    g   = scan["gpu"]
    sm  = g["sm_str"]
    fb  = tune["face_batch"]
    pb  = tune["plate_batch"]
    w   = ROOT / "weights"
    lines = [
        "# build_engines.ps1 — Auto-generated by tools/auto_tune.py",
        f"# GPU:  {g.get('name','?')}  ({g.get('sm_str','?')})",
        f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "# Run this script ON THE TARGET MACHINE with TensorRT in PATH",
        "",
        "Set-Location (Split-Path $MyInvocation.MyCommand.Path)",
        "",
        onnx_export_snippet(),
        "",
        "# ---- STEP 1: Build Face Engine ----",
        f"Write-Host 'Building face engine (batch={fb}, FP16) ...'",
        face_engine_cmd(fb, sm, w),
        "",
        "# ---- STEP 2: Build Vehicle Engine ----",
        f"Write-Host 'Building vehicle engine (batch={fb}, FP16) ...'",
        vehicle_engine_cmd(fb, sm, w),
        "",
        "# ---- STEP 3: Build Plate Engine ----",
        f"Write-Host 'Building plate engine (batch={pb}, FP16) ...'",
        plate_engine_cmd(pb, sm, w),
        "",
        "# ---- STEP 4: Verify VRAM ----",
        "Write-Host 'Checking VRAM after engine load ...'",
        "nvidia-smi --query-gpu=memory.used,memory.free --format=csv",
        "",
        "Write-Host 'All engines built. Now run:'",
        "Write-Host '  python run_blur_v3.py --dry-run'",
        "Write-Host '  python run_blur_v3.py --config config/pipeline_v3.yaml --frames 1000'",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="FAST v3 Auto-Tuner")
    ap.add_argument("--scan", default=str(FACTS_DIR / "hw_scan.json"),
                    help="Path to hw_scan.json from scan_hardware.py")
    ap.add_argument("--out",  default=str(CONFIG_DIR / "pipeline_v3.yaml"),
                    help="Output path for tuned pipeline YAML")
    a = ap.parse_args()

    scan_path = Path(a.scan)
    if not scan_path.exists():
        print(f"ERROR: hw_scan.json not found at {scan_path}")
        print("Run scan_hardware.py first:  python tools/scan_hardware.py")
        sys.exit(1)

    scan = json.loads(scan_path.read_text())

    if not scan.get("gate_pass"):
        print("ERROR: hw_scan.json reports GATE FAIL.")
        print("Fix the hardware/software issues first, then re-run scan_hardware.py.")
        sys.exit(1)

    tune = scan["tune"]
    g    = scan["gpu"]

    print("\n" + "="*60)
    print("  FAST v3  Auto-Tuner")
    print(f"  GPU: {g.get('name','?')}  Tier={g.get('tier','?')}  CC={g.get('cc_str','?')}")
    print("="*60)

    # Write tuned config
    out_path = Path(a.out)
    write_yaml(out_path, scan, tune, compat=False)
    print(f"\n  Written: {out_path}")

    # Write compat (v2-baseline) config
    compat_path = out_path.parent / "pipeline_v3_compat.yaml"
    write_yaml(compat_path, scan, tune, compat=True)
    print(f"  Written: {compat_path}")

    # Write engine build script
    build_path = ROOT / "build_engines.ps1"
    write_build_script(build_path, scan, tune)
    print(f"  Written: {build_path}")

    # Print tune summary
    vb = tune.get("vram_budget", {})
    print(f"""
  Tune summary
  ─────────────────────────────────────────────
  GPU tier:           {g.get('tier','?')} — {tune.get('desc','')}
  Architecture:       {g.get('architecture','?')}  ({g.get('sm_str','?')})
  face_batch:         {tune['face_batch']}
  plate_batch:        {tune['plate_batch']}
  CUDA streams:       {tune['stream_count']}
  shm_slots:          {tune['shm_slots']}
  input_queue_depth:  {tune['input_queue']}
  result_queue_depth: {tune['result_queue']}
  VRAM used (est.):   {vb.get('total_gb','?')} GB / {vb.get('available_gb','?')} GB
  VRAM headroom:      {vb.get('headroom_gb','?')} GB
  Speed ceiling est.: {tune.get('ceiling_estimate_img_s','?')} img/s
  ─────────────────────────────────────────────
""")

    # Save decision record
    decision = {
        "timestamp"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gpu"         : g.get("name"),
        "tier"        : g.get("tier"),
        "sm"          : g.get("sm_str"),
        "tune"        : tune,
        "output_files": {
            "pipeline_v3.yaml"        : str(out_path),
            "pipeline_v3_compat.yaml" : str(compat_path),
            "build_engines.ps1"       : str(build_path),
        },
    }
    rec = FACTS_DIR / "auto_tune.json"
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps(decision, indent=2, default=str))
    print(f"  Decision record: {rec}")

    print("\n  Next steps:")
    print("  1. Edit pipeline_v3.yaml — set input_dir and output_dir")
    print(f"  2. Run:  .\\build_engines.ps1           (builds TRT engines on this GPU)")
    print("  3. Run:  python run_blur_v3.py --dry-run")
    print("  4. Run:  python run_blur_v3.py --frames 1000  (benchmark)")
    print("  5. Run:  python run_blur_v3.py  (full production)\n")


if __name__ == "__main__":
    main()
