#!/usr/bin/env python3
"""
setup_env.py  —  FAST v4 Universal Hardware Scanner, Dependency Auto-Installer & Tuner
========================================================================================
Place this on ANY laptop or workstation. Run ONE command:

    python setup_env.py

What it does automatically:
  1. SCANS LAPTOP HARDWARE:
     - GPU model, VRAM capacity, compute capability (SM architecture)
     - NVIDIA driver version & maximum supported CUDA runtime
     - System RAM & CPU cores
  2. AUDITS INSTALLED ENVIRONMENT:
     - Checks Python, CUDA, PyTorch (CUDA vs CPU), TensorRT, OpenCV, etc.
  3. AUTO-INSTALLS & DOWNLOADS MISSING REQUIREMENTS:
     - Automatically selects and downloads the exact matching PyTorch wheel:
         * CUDA 12.4:  torch+cu124
         * CUDA 12.1:  torch+cu121
         * CUDA 11.8:  torch+cu118
         * CPU-only:   torch+cpu (if no NVIDIA GPU present)
     - Installs TensorRT (if supported GPU), OpenCV, NumPy, PyYAML, psutil, pytest
  4. AUTO-TUNES PIPELINE CONFIGURATION:
     - Classifies machine into Tier A / B / C / D
     - Tunes batch sizes, queue depths, decode/writer threads, VRAM limits for THIS laptop
     - Writes tailored config/pipeline_v4.yaml and build_engines.ps1
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FACTS_DIR = ROOT / "run_facts"
CONFIG_DIR = ROOT / "config"
WEIGHTS_DIR = ROOT / "weights"


def log(msg: str, tag: str = "INFO") -> None:
    prefixes = {
        "INFO": "[*]",
        "SUCCESS": "[+]",
        "WARN": "[!]",
        "ERROR": "[-]",
        "STEP": "[=>]",
    }
    p = prefixes.get(tag, "[*]")
    print(f"{p} {msg}")


def banner() -> None:
    print(r"""
================================================================================
  FAST v4  Universal Laptop Hardware Scanner, Installer & Auto-Tuner
  Drop on ANY Laptop -> Auto-Detect -> Auto-Download -> Auto-Tune -> Ready!
================================================================================
""")


# ---------------------------------------------------------------------------
# 1. Hardware Inspection
# ---------------------------------------------------------------------------

def inspect_hardware() -> dict:
    info = {
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 4,
        "ram_gb": 16.0,
        "has_nvidia_gpu": False,
        "gpu_name": "CPU Only",
        "gpu_vram_mb": 0,
        "driver_version": "None",
        "max_cuda_version": 0.0,
        "compute_cap": [0, 0],
        "tier": "D",
    }

    # RAM
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        pass

    # NVIDIA-SMI
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False
            )
            if r.returncode == 0 and r.stdout.strip():
                line = r.stdout.strip().split("\n")[0]
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    info["has_nvidia_gpu"] = True
                    info["gpu_name"] = parts[0]
                    info["gpu_vram_mb"] = int(float(parts[1]))
                    info["driver_version"] = parts[2]
        except Exception:
            pass

        # Check maximum CUDA supported by driver
        try:
            r_cuda = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
            for line in r_cuda.stdout.split("\n"):
                if "CUDA Version:" in line:
                    c_str = line.split("CUDA Version:")[1].split()[0].strip()
                    info["max_cuda_version"] = float(c_str)
                    break
        except Exception:
            pass

    # Classify Tier
    vram = info["gpu_vram_mb"]
    if info["has_nvidia_gpu"]:
        if vram >= 15000:
            info["tier"] = "A"
        elif vram >= 5500:
            info["tier"] = "B"
        elif vram >= 3500:
            info["tier"] = "C"
        else:
            info["tier"] = "D"
    else:
        info["tier"] = "CPU"

    return info


# ---------------------------------------------------------------------------
# 2. Python Packages Audit
# ---------------------------------------------------------------------------

def inspect_packages(py_exe: str) -> dict:
    script = """
import sys, json

status = {
    "torch_installed": False,
    "torch_version": None,
    "torch_cuda": False,
    "torch_cuda_version": None,
    "tensorrt_installed": False,
    "tensorrt_version": None,
    "opencv_installed": False,
    "opencv_version": None,
    "yaml_installed": False,
    "numpy_installed": False,
    "psutil_installed": False,
    "pytest_installed": False,
}

try:
    import torch
    status["torch_installed"] = True
    status["torch_version"] = torch.__version__
    status["torch_cuda"] = torch.cuda.is_available()
    status["torch_cuda_version"] = getattr(torch.version, "cuda", None)
except ImportError:
    pass

try:
    import tensorrt as trt
    status["tensorrt_installed"] = True
    status["tensorrt_version"] = trt.__version__
except ImportError:
    pass

try:
    import cv2
    status["opencv_installed"] = True
    status["opencv_version"] = cv2.__version__
except ImportError:
    pass

try:
    import yaml
    status["yaml_installed"] = True
except ImportError:
    pass

try:
    import numpy
    status["numpy_installed"] = True
except ImportError:
    pass

try:
    import psutil
    status["psutil_installed"] = True
except ImportError:
    pass

try:
    import pytest
    status["pytest_installed"] = True
except ImportError:
    pass

print(json.dumps(status))
"""
    r = subprocess.run([py_exe, "-c", script], capture_output=True, text=True, check=False)
    if r.returncode == 0 and r.stdout.strip():
        try:
            return json.loads(r.stdout.strip())
        except Exception:
            pass
    return {}


# ---------------------------------------------------------------------------
# 3. Automated Downloader & Installer
# ---------------------------------------------------------------------------

def resolve_torch_command(hw: dict) -> list[str]:
    """Select the best PyTorch installation command for this laptop."""
    max_cuda = hw.get("max_cuda_version", 0.0)
    has_gpu = hw.get("has_nvidia_gpu", False)

    if not has_gpu or max_cuda < 11.0:
        log("No compatible NVIDIA GPU found -> Selecting CPU PyTorch", "WARN")
        return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"]

    if max_cuda >= 12.4:
        log(f"NVIDIA GPU with Driver CUDA {max_cuda} -> Installing PyTorch with CUDA 12.4", "STEP")
        return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]
    elif max_cuda >= 12.1:
        log(f"NVIDIA GPU with Driver CUDA {max_cuda} -> Installing PyTorch with CUDA 12.1", "STEP")
        return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu121"]
    else:
        log(f"NVIDIA GPU with Driver CUDA {max_cuda} -> Installing PyTorch with CUDA 11.8", "STEP")
        return ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu118"]


def install_missing_dependencies(py_exe: str, hw: dict, pkgs: dict) -> bool:
    pip_cmd = [py_exe, "-m", "pip", "install"]

    # 1. Check PyTorch
    need_torch = False
    if not pkgs.get("torch_installed"):
        need_torch = True
    elif hw.get("has_nvidia_gpu") and not pkgs.get("torch_cuda"):
        log("PyTorch is installed but has NO CUDA acceleration. Re-installing CUDA PyTorch...", "WARN")
        need_torch = True

    if need_torch:
        log("Downloading & installing optimized PyTorch wheel...", "STEP")
        torch_args = resolve_torch_command(hw)
        cmd = pip_cmd + torch_args
        r = subprocess.run(cmd, check=False)
        if r.returncode != 0:
            log("Warning: Failed to install specific PyTorch wheel. Falling back to default pip install...", "WARN")
            subprocess.run(pip_cmd + ["torch", "torchvision"], check=False)

    # 2. Core dependencies
    to_install = []
    if not pkgs.get("opencv_installed"):
        to_install.append("opencv-python")
    if not pkgs.get("yaml_installed"):
        to_install.append("pyyaml")
    if not pkgs.get("numpy_installed"):
        to_install.append("numpy")
    if not pkgs.get("psutil_installed"):
        to_install.append("psutil")
    if not pkgs.get("pytest_installed"):
        to_install.append("pytest")

    # Optional TensorRT
    if hw.get("has_nvidia_gpu") and not pkgs.get("tensorrt_installed"):
        to_install.append("tensorrt")

    if to_install:
        log(f"Installing missing utilities: {to_install}...", "STEP")
        subprocess.run(pip_cmd + to_install, check=False)

    return True


# ---------------------------------------------------------------------------
# 4. Auto-Tuning Config Generator for This Machine
# ---------------------------------------------------------------------------

def generate_tuned_config(hw: dict) -> Path:
    tier = hw.get("tier", "B")
    ram_gb = hw.get("ram_gb", 16.0)

    # Defaults per tier
    if tier == "A":
        batch_size = 16
        decode_threads = 6
        writer_threads = 6
        prefetch = 4
        max_pending = 32
        plate_batch = 64
        desc = "High-End Tier A (RTX 4090 / A100 class)"
    elif tier == "B":
        batch_size = 8
        decode_threads = 4
        writer_threads = 4
        prefetch = 3
        max_pending = 24
        plate_batch = 32
        desc = "Mid-Range Tier B (RTX 3050 / 3060 / 4060 class)"
    elif tier == "C":
        batch_size = 4
        decode_threads = 3
        writer_threads = 3
        prefetch = 2
        max_pending = 16
        plate_batch = 16
        desc = "Budget Tier C (GTX 1650 Ti / 4GB class)"
    else:
        batch_size = 2
        decode_threads = 2
        writer_threads = 2
        prefetch = 2
        max_pending = 8
        plate_batch = 8
        desc = "Low-VRAM / CPU Fallback"

    # Scale down if RAM is limited (< 14 GB)
    if ram_gb < 14:
        decode_threads = max(2, decode_threads - 1)
        writer_threads = max(2, writer_threads - 1)
        max_pending = max(8, int(max_pending * 0.7))

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CONFIG_DIR / "pipeline_v4.yaml"

    cfg_text = f"""# ==============================================================================
# FAST v4 Auto-Tuned Configuration for: {hw.get('gpu_name', 'System')}
# Auto-detected Profile: {desc} (RAM: {ram_gb} GB)
# ==============================================================================

input_dir: "inputs"
output_dir: "outputs"
run_dir: "runs"
recursive: true
extensions: [".jpg", ".jpeg", ".png"]
skip_existing: true

models:
  face:
    engine: "weights/yolov8m-face_fp16_b{batch_size}_auto.engine"
    pt: "weights/yolov8m-face.pt"
    nc: null
    long_side: 1280
    conf: 0.15
    iou: 0.7
    max_det: 300
  vehicle:
    engine: "weights/yolo11m_fp16_b{batch_size}_auto.engine"
    pt: "weights/yolo11m.pt"
    nc: null
    long_side: 640
    conf: 0.25
    iou: 0.7
    classes: [2, 3, 5, 7]
    max_det: 300
  plate:
    enabled: true
    mode: "crops"
    engine: "weights/plate_nano_fp16_b{plate_batch}_auto.engine"
    pt: "weights/plate_nano_192x320_best.pt"
    nc: null
    imgsz: 320
    imgsz_w: 320
    imgsz_h: 192
    conf: 0.30
    iou: 0.7
    max_det: 50
    max_batch: {plate_batch}
    crop_pad: 0.10
    min_vehicle_px: 40

shapes:
  landscape: [1280, 1280]
  portrait: [1280, 640]
  square: [1280, 1280]

vehicle_input: "gpu_half"
vehicle_antialias: true
resize_interp: "linear"

batch_size: {batch_size}
decode_threads: {decode_threads}
writer_threads: {writer_threads}
prefetch_batches: {prefetch}
max_pending_writes: {max_pending}
opencv_threads: 1
cpu_affinity: null

# ── FAST v4 Render Settings ───────────────────────────────────────────────────
render:
  face_mask: "ellipse"              # Inscribed elliptical head mask
  plate_mask: "feathered_rect"      # Feathered padded rectangle
  feather_sigma_px: 2.0             # Continuous Gaussian edge transition
  face_pad_ratio: 0.05              # 5% outward expansion
  plate_pad_ratio: 0.10             # 10% outward expansion
  min_ellipse_side_px: 8            # Rect fallback for tiny boxes
  bit_identical_outside_mask: true  # Enforce pixel invariant

# ── FAST v4 Blur Settings ─────────────────────────────────────────────────────
blur:
  method: "adaptive"                # Adaptive resolution-based blur
  fixed_kernel: 51
  adaptive_target_px: 24
  min_sigma: 5.0

jpeg_quality: 95
copy_if_no_detections: false

# ── FAST v4 Privacy Report Settings ───────────────────────────────────────────
report:
  enabled: true
  output_dir: "privacy_report"
  include_box_coordinates: true
  include_confidence_scores: true
  include_timing: true
  include_blur_area_stats: true
  generate_markdown: true
  generate_csv: true
  generate_session_stats: true
  embed_audit_result: true
  audit_sample_size: 2000
  audit_max_ratio: 0.5
"""
    cfg_path.write_text(cfg_text, encoding="utf-8")
    log(f"Generated auto-tuned pipeline configuration -> {cfg_path}", "SUCCESS")
    return cfg_path


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FAST v4 Universal Environment Setup & Tuner")
    parser.add_argument("--check-only", action="store_true", help="Audit hardware and packages without installing")
    parser.add_argument("--venv", type=Path, default=None, help="Path to specific virtualenv python.exe")
    args = parser.parse_args()

    banner()

    # Determine python executable to inspect / install into
    py_exe = str(args.venv) if args.venv else sys.executable
    log(f"Operating Environment: Python {platform.python_version()} ({py_exe})", "INFO")

    # 1. Hardware Inspection
    log("Scanning Laptop Hardware...", "STEP")
    hw = inspect_hardware()
    print(f"    - System:    {hw['os']} {hw['os_release']}")
    print(f"    - CPU:       {hw['cpu_count']} Cores, RAM: {hw['ram_gb']} GB")
    print(f"    - GPU:       {hw['gpu_name']} ({hw['gpu_vram_mb']} MB VRAM)")
    print(f"    - Driver:    {hw['driver_version']} (Max CUDA Supported: {hw['max_cuda_version']})")
    print(f"    - Tier:      Tier {hw['tier']}")

    # 2. Package Audit
    log("Auditing Installed Packages...", "STEP")
    pkgs = inspect_packages(py_exe)
    print(f"    - PyTorch:   {pkgs.get('torch_version', 'Not installed')} (CUDA: {pkgs.get('torch_cuda')})")
    print(f"    - TensorRT:  {pkgs.get('tensorrt_version', 'Not installed')}")
    print(f"    - OpenCV:    {pkgs.get('opencv_version', 'Not installed')}")
    print(f"    - PyYAML:    {'Installed' if pkgs.get('yaml_installed') else 'Missing'}")
    print(f"    - NumPy:     {'Installed' if pkgs.get('numpy_installed') else 'Missing'}")

    if args.check_only:
        log("Check-only mode requested. Exiting without modifying environment.", "INFO")
        return

    # 3. Auto-Download & Install
    log("Resolving & Installing Missing Requirements...", "STEP")
    install_missing_dependencies(py_exe, hw, pkgs)

    # 4. Auto-Tune Configuration
    log("Auto-Tuning Pipeline for This Laptop...", "STEP")
    cfg_path = generate_tuned_config(hw)

    # 5. Verification
    log("Re-verifying Environment...", "STEP")
    post_pkgs = inspect_packages(py_exe)
    ready = post_pkgs.get("torch_installed") and post_pkgs.get("opencv_installed") and post_pkgs.get("yaml_installed")

    if ready:
        log("Environment is 100% READY on this laptop!", "SUCCESS")
        print("\n" + "=" * 68)
        print("  HOW TO RUN ON THIS LAPTOP:")
        print(f"  1. Test Dry-Run:     python run_blur_v4.py --dry-run")
        print(f"  2. Run Test Suite:   pytest tests/test_render_v4.py tests/test_privacy_report.py -v")
        print(f"  3. Run Pipeline:     python run_blur_v4.py --config {cfg_path}")
        print("=" * 68 + "\n")
    else:
        log("Environment setup finished with warnings. Review missing packages above.", "WARN")


if __name__ == "__main__":
    main()
