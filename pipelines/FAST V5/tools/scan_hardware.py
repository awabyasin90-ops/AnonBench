#!/usr/bin/env python3
"""
scan_hardware.py  -  FAST v3 Hardware Scanner & Tuning Advisor
==============================================================
Run this FIRST on any target machine before doing anything else.

  python tools/scan_hardware.py

What it does
------------
1. Scans GPU (VRAM, compute capability, architecture, Tensor Cores)
2. Scans CPU (cores, threads, clock)
3. Scans system RAM
4. Scans CUDA toolkit + driver version
5. Checks TensorRT availability and version
6. Checks TurboJPEG / PyTurboJPEG
7. Checks disk free space
8. Classifies the GPU into a capability tier (A/B/C/D)
9. Prints a concrete tune table: batch sizes, stream count, VRAM budget, RAM pool limits
10. Saves everything to run_facts/hw_scan.json  (machine-readable for auto_tune.py)
11. Prints a PASS/FAIL gate: if anything critical is missing, tells you exactly what to fix

Supported GPU tiers
-------------------
  Tier A  (>=16 GB VRAM, >=8.0 CC)   - A100, RTX 4090, A6000
  Tier B  ( 6-15 GB VRAM, 8.0+ CC)   - RTX 3050 6GB, RTX 3060, RTX 4060 <-- YOU ARE HERE
  Tier C  ( 4-5 GB VRAM, 7.5+ CC)    - GTX 1650 Ti (FAST v2 machine)
  Tier D  ( <4 GB or <7.5 CC)        - Not supported

Output: run_facts/hw_scan.json
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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


# ---------------------------------------------------------------------------
# GPU tier table:  (min_vram_mib, min_cc_major, name, face_batch, plate_batch,
#                   stream_count, shm_slots, queue_depth, writer_threads)
# ---------------------------------------------------------------------------
GPU_TIERS = [
    # tier, min_vram_mib, min_cc, label
    ("A", 15000, 8, "High-End (A100 / RTX 4090 class)"),
    ("B",  5500, 8, "Mid-Range 6GB+ Ampere/Ada (RTX 3050 / 3060 / 4060)"),
    ("C",  3500, 7, "Budget 4GB Turing/Volta (GTX 1650 Ti / RTX 2060 4GB)"),
    ("D",      0, 0, "Unsupported (<4GB or <SM 7.0)"),
]

TIER_TUNE = {
    "A": dict(face_batch=16, plate_batch=64, stream_count=4, shm_slots=8,
              input_queue=20, result_queue=32, writer_threads=6, decode_threads=6,
              desc="Maximum throughput. INT8 optional. GPU letterbox recommended."),
    "B": dict(face_batch=8,  plate_batch=32, stream_count=3, shm_slots=5,
              input_queue=12, result_queue=16, writer_threads=4, decode_threads=4,
              desc="RTX 3050 6GB - v3 production config. True FP16 Tensor Cores."),
    "C": dict(face_batch=4,  plate_batch=24, stream_count=2, shm_slots=6,
              input_queue=16, result_queue=24, writer_threads=5, decode_threads=4,
              desc="GTX 1650 Ti 4GB - FAST v2 config (use FAST v2 pipeline)."),
    "D": dict(face_batch=2,  plate_batch=8,  stream_count=1, shm_slots=3,
              input_queue=8,  result_queue=8,  writer_threads=2, decode_threads=2,
              desc="Unsupported. Use CPU-only mode or upgrade GPU."),
}

# RAM-based queue cap (16GB machines need tighter pools than 64GB)
RAM_QUEUE_CAPS = [
    (60,  dict(shm_mult=1.0, queue_mult=1.0)),   # >= 60 GB: no reduction
    (24,  dict(shm_mult=0.9, queue_mult=0.9)),   # 24-60 GB
    (14,  dict(shm_mult=0.7, queue_mult=0.7)),   # 14-24 GB  <-- RTX 3050 laptop
    ( 8,  dict(shm_mult=0.5, queue_mult=0.5)),   # 8-14 GB
    ( 0,  dict(shm_mult=0.3, queue_mult=0.3)),   # < 8 GB
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shell(cmd: list[str], timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception as e:
        return f"unavailable: {e}"


def try_import(name: str) -> tuple[bool, str]:
    try:
        m = __import__(name)
        v = getattr(m, "__version__", "present (no __version__)")
        return True, v
    except Exception as e:
        return False, str(e)


def get_gpu_architecture(cc_major: int, cc_minor: int) -> str:
    arch_map = {
        (7, 0): "Volta",
        (7, 5): "Turing",
        (8, 0): "Ampere (A100/A30/A40)",
        (8, 6): "Ampere (RTX 30xx / A2000-A5000)",
        (8, 7): "Ampere (Jetson Orin)",
        (8, 9): "Ada Lovelace (RTX 40xx)",
        (9, 0): "Hopper (H100)",
    }
    return arch_map.get((cc_major, cc_minor), f"Unknown (sm_{cc_major}{cc_minor})")


def classify_tier(vram_mib: int, cc_major: int, cc_minor: int) -> str:
    cc = cc_major + cc_minor / 10
    for tier, min_vram, min_cc, _ in GPU_TIERS:
        if vram_mib >= min_vram and cc >= min_cc:
            return tier
    return "D"


def ram_scale(ram_gib: float) -> dict:
    for threshold, caps in RAM_QUEUE_CAPS:
        if ram_gib >= threshold:
            return caps
    return RAM_QUEUE_CAPS[-1][1]


# ---------------------------------------------------------------------------
# Scan functions
# ---------------------------------------------------------------------------

def scan_gpu(info: dict) -> dict:
    """Scan GPU and return GPU sub-dict."""
    gpu = {"available": False}
    try:
        import torch
        gpu["torch_version"] = torch.__version__
        gpu["cuda_version_torch"] = torch.version.cuda
        gpu["available"] = torch.cuda.is_available()

        if not gpu["available"]:
            gpu["error"] = "torch.cuda.is_available() returned False"
            return gpu

        props = torch.cuda.get_device_properties(0)
        cap = torch.cuda.get_device_capability(0)
        free_mib, total_mib = torch.cuda.mem_get_info(0)
        free_mib //= 2**20
        total_mib //= 2**20

        gpu.update(
            name           = props.name,
            vram_total_mib = total_mib,
            vram_free_mib  = free_mib,
            vram_used_mib  = total_mib - free_mib,
            cc_major       = cap[0],
            cc_minor       = cap[1],
            cc_str         = f"{cap[0]}.{cap[1]}",
            sm_str         = f"sm_{cap[0]}{cap[1]}",
            architecture   = get_gpu_architecture(cap[0], cap[1]),
            cuda_cores     = props.multi_processor_count,
            has_tensor_cores = (cap[0] >= 7),
            has_real_fp16_tc = (cap[0] >= 8),
            has_int8_tc      = (cap[0] >= 7),
            has_bf16         = (cap[0] >= 8),
            cudnn_version    = torch.backends.cudnn.version(),
        )
        gpu["tier"] = classify_tier(total_mib, cap[0], cap[1])
    except Exception as e:
        gpu["error"] = str(e)
    return gpu


def scan_cpu(info: dict) -> dict:
    """Scan CPU cores and RAM."""
    cpu = {
        "logical_cores" : os.cpu_count() or 0,
        "platform"      : platform.processor(),
        "machine"       : platform.machine(),
        "os"            : platform.platform(),
    }
    try:
        import psutil
        cpu["physical_cores"] = psutil.cpu_count(logical=False)
        cpu["freq_mhz"]       = round(psutil.cpu_freq().max if psutil.cpu_freq() else 0)
        mem = psutil.virtual_memory()
        cpu["ram_total_gib"]  = round(mem.total / 2**30, 1)
        cpu["ram_free_gib"]   = round(mem.available / 2**30, 1)
        cpu["ram_used_gib"]   = round((mem.total - mem.available) / 2**30, 1)
    except ImportError:
        cpu["psutil"] = "MISSING - run: pip install psutil"
    return cpu


def scan_software(info: dict) -> dict:
    """Check all required and optional packages."""
    sw = {}
    required = {
        "torch"     : "PyTorch",
        "tensorrt"  : "TensorRT",
        "ultralytics": "Ultralytics YOLO",
        "cv2"       : "OpenCV",
        "numpy"     : "NumPy",
        "yaml"      : "PyYAML",
        "psutil"    : "psutil",
    }
    optional = {
        "turbojpeg" : "PyTurboJPEG (SIMD decode - strongly recommended)",
        "onnx"      : "onnx",
        "onnxslim"  : "onnxslim",
        "pynvml"    : "pynvml (GPU telemetry)",
    }
    sw["required"] = {}
    for mod, label in required.items():
        ok, ver = try_import(mod)
        sw["required"][mod] = {"ok": ok, "version": ver, "label": label}

    sw["optional"] = {}
    for mod, label in optional.items():
        ok, ver = try_import(mod)
        sw["optional"][mod] = {"ok": ok, "version": ver, "label": label}

    # nvcc
    sw["nvcc"] = shell(["nvcc", "--version"])

    # nvidia-smi
    sw["nvidia_smi_csv"] = shell([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,"
        "power.draw,clocks.sm,compute_cap",
        "--format=csv,noheader"
    ])

    # Power scheme (Windows)
    if os.name == "nt":
        sw["power_scheme"] = shell(["powercfg", "/getactivescheme"])

    return sw


def scan_disk() -> dict:
    """Check disk free space on the project drive."""
    usage = shutil.disk_usage(ROOT)
    return {
        "path"      : str(ROOT),
        "total_gib" : round(usage.total / 2**30, 1),
        "used_gib"  : round(usage.used / 2**30, 1),
        "free_gib"  : round(usage.free / 2**30, 1),
    }


# ---------------------------------------------------------------------------
# Tune advisor
# ---------------------------------------------------------------------------

def compute_tune(gpu: dict, cpu: dict) -> dict:
    """Return the recommended pipeline parameters for this hardware."""
    tier = gpu.get("tier", "D")
    base = dict(TIER_TUNE[tier])

    # Adjust for RAM
    ram_gib = cpu.get("ram_total_gib", 16.0)
    scale = ram_scale(ram_gib)
    base["shm_slots"]    = max(3, math.floor(base["shm_slots"]    * scale["shm_mult"]))
    base["input_queue"]  = max(6, math.floor(base["input_queue"]  * scale["queue_mult"]))
    base["result_queue"] = max(8, math.floor(base["result_queue"] * scale["queue_mult"]))

    # CPU thread budget
    logical = cpu.get("logical_cores", 8)
    base["decode_threads"]  = min(base["decode_threads"],  max(2, logical // 3))
    base["writer_threads"]  = min(base["writer_threads"],  max(2, logical // 4))

    # VRAM budget breakdown estimate
    vram = gpu.get("vram_total_mib", 4096)
    face_b = base["face_batch"]
    plate_b = base["plate_batch"]
    face_vram  = round(face_b  * 0.23, 1)   # ~0.23 GB per img at 1280px FP16
    veh_vram   = round(face_b  * 0.10, 1)   # ~0.10 GB per img at 640px  FP16
    plate_vram = round(plate_b * 0.019, 1)  # ~19 MB per crop at 192x320 FP16
    ipc_vram   = 0.3
    total_vram = face_vram + veh_vram + plate_vram + ipc_vram
    headroom   = round(vram / 1024 - total_vram, 1)

    base["vram_budget"] = {
        "face_gb"    : face_vram,
        "vehicle_gb" : veh_vram,
        "plate_gb"   : plate_vram,
        "ipc_gb"     : ipc_vram,
        "total_gb"   : round(total_vram, 1),
        "headroom_gb": headroom,
        "available_gb": round(vram / 1024, 1),
    }

    # Speed ceiling estimate (FP16 Tensor Core TFLOPS table)
    tflops_map = {
        "A": 77.0, "B": 8.0, "C": 4.2, "D": 2.0
    }
    tflops = tflops_map.get(tier, 4.0)
    # Very rough: 88.86 ms/frame ceiling on GTX 1650 Ti (4.2 TFLOPS)
    # Scale proportionally
    baseline_ms = 88.86
    baseline_tflops = 4.2
    est_ms = round(baseline_ms * baseline_tflops / tflops, 1)
    est_ceiling = round(1000 / est_ms, 1)
    base["ceiling_estimate_img_s"] = est_ceiling
    base["ceiling_estimate_ms"]    = est_ms

    return base


# ---------------------------------------------------------------------------
# PASS/FAIL gate
# ---------------------------------------------------------------------------

def print_gate(gpu: dict, sw: dict, tune: dict) -> bool:
    """Print PASS/FAIL gate. Returns True if all critical gates pass."""
    print("\n" + "="*70)
    print("  HARDWARE SCAN GATE RESULTS")
    print("="*70)

    failures = []
    warnings = []

    # Gate 1: GPU available
    if not gpu.get("available"):
        failures.append("CUDA GPU not detected. Install NVIDIA driver >= 527.41.")
    else:
        tier = gpu["tier"]
        cc = f"{gpu['cc_major']}.{gpu['cc_minor']}"
        vram = gpu["vram_total_mib"]
        print(f"  [{'PASS' if tier != 'D' else 'FAIL'}] GPU: {gpu['name']}  "
              f"Tier={tier}  CC={cc}  VRAM={vram}MiB")
        if tier == "D":
            failures.append(f"GPU {gpu['name']} is unsupported (Tier D). "
                            "Need >= 4GB VRAM and Compute >= 7.0.")
        elif tier == "C":
            warnings.append(f"GPU {gpu['name']} is Tier C (GTX 1650 Ti class). "
                            "Use FAST v2, not v3. v3 is optimised for >= 6GB Ampere cards.")

    # Gate 2: Required packages
    for mod, info in sw.get("required", {}).items():
        status = "PASS" if info["ok"] else "FAIL"
        print(f"  [{status}] {info['label']:30s} {info['version']}")
        if not info["ok"]:
            failures.append(f"Missing required package: {mod}  ({info['label']})")

    # Gate 3: TurboJPEG (optional but important)
    tj = sw.get("optional", {}).get("turbojpeg", {})
    if tj.get("ok"):
        print(f"  [PASS] PyTurboJPEG (SIMD decode)          {tj['version']}")
    else:
        print(f"  [WARN] PyTurboJPEG not installed — SIMD decode disabled (slower)")
        warnings.append("PyTurboJPEG missing. Run: pip install PyTurboJPEG")

    # Gate 4: VRAM headroom
    if gpu.get("available"):
        headroom = tune.get("vram_budget", {}).get("headroom_gb", -1)
        if headroom < 0.5:
            failures.append(f"VRAM headroom too tight ({headroom:.1f} GB). "
                            "Reduce face_batch or plate_batch.")
        else:
            print(f"  [PASS] VRAM headroom: {headroom:.1f} GB  "
                  f"({tune['vram_budget']['total_gb']:.1f} GB used / "
                  f"{tune['vram_budget']['available_gb']:.1f} GB total)")

    # Gate 5: Disk
    disk_free = scan_disk().get("free_gib", 0)
    if disk_free < 10:
        warnings.append(f"Low disk: {disk_free:.1f} GB free. Need >= 10 GB for output.")
    else:
        print(f"  [PASS] Disk free: {disk_free:.1f} GB")

    # Summary
    print()
    if failures:
        print(f"  RESULT: FAIL  ({len(failures)} critical issue(s))")
        for f in failures:
            print(f"    !! {f}")
    else:
        print(f"  RESULT: PASS  — Ready to build TRT engines and run v3 pipeline")

    if warnings:
        print(f"\n  Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"    >> {w}")

    print("="*70 + "\n")
    return len(failures) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("\n" + "="*70)
    print("  FAST v3  Hardware Scanner & Tuning Advisor")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  {platform.node()}")
    print("="*70 + "\n")

    info: dict[str, Any] = {
        "scan_time" : time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hostname"  : platform.node(),
        "python"    : sys.version,
    }

    print("[ 1/5 ] Scanning GPU ...")
    info["gpu"] = scan_gpu(info)
    g = info["gpu"]
    if g.get("available"):
        print(f"        Found: {g['name']}  VRAM={g['vram_total_mib']} MiB  "
              f"CC={g['cc_str']}  Arch={g['architecture']}  Tier={g['tier']}")
    else:
        print(f"        ERROR: {g.get('error', 'unknown')}")

    print("[ 2/5 ] Scanning CPU & RAM ...")
    info["cpu"] = scan_cpu(info)
    c = info["cpu"]
    print(f"        {c.get('logical_cores')} logical cores  |  "
          f"RAM {c.get('ram_total_gib','?')} GiB total  "
          f"({c.get('ram_free_gib','?')} GiB free)")

    print("[ 3/5 ] Scanning software stack ...")
    info["sw"] = scan_software(info)

    print("[ 4/5 ] Scanning disk ...")
    info["disk"] = scan_disk()
    print(f"        {info['disk']['free_gib']} GiB free on {info['disk']['path']}")

    print("[ 5/5 ] Computing recommended tune ...")
    info["tune"] = compute_tune(info["gpu"], info["cpu"])
    tune = info["tune"]
    print(f"        GPU Tier:          {g.get('tier','?')}  —  {tune.get('desc','')}")
    print(f"        face_batch:        {tune['face_batch']}")
    print(f"        plate_batch:       {tune['plate_batch']}")
    print(f"        CUDA streams:      {tune['stream_count']}")
    print(f"        shm_slots:         {tune['shm_slots']}")
    print(f"        input_queue_depth: {tune['input_queue']}")
    print(f"        Speed ceiling est: {tune['ceiling_estimate_img_s']} img/s  "
          f"({tune['ceiling_estimate_ms']} ms/frame)")

    # Print gate
    passed = print_gate(info["gpu"], info["sw"], info["tune"])
    info["gate_pass"] = passed

    # Save
    FACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = FACTS_DIR / "hw_scan.json"
    out.write_text(json.dumps(info, indent=2, default=str))
    print(f"Saved: {out}\n")

    if passed:
        print("Next step:")
        print("  python tools/auto_tune.py")
        print("  -> Reads hw_scan.json and writes the optimal pipeline_v3.yaml\n")
    else:
        print("Fix the FAIL items above before running auto_tune.py.\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
