#!/usr/bin/env python3
"""
pipeline_agent_manager.py
=========================
Autonomous AI Agent & Research Manager for Anonymization Failure Pattern Study:
- 2 Ground-Truth Baselines (InsightFace SCRFD-34G & RT-DETR v2 Plate)
- 4 Evaluated Anonymization Pipelines (FAST v5, Meta EgoBlur, DashcamCleaner, FusionPortable Anonymizer)
- 1 Sensor Simulation Platform (Pioneer 3AT ROS)

Features:
  1. --status / --check : Non-invasive audit of all dependencies and weights (NO DOWNLOADS).
  2. --json             : Emits machine-readable JSON status for automated AI agents.
  3. --export-reqs      : Exports requirements files per pipeline and unified requirements_all.txt.
  4. --download         : Safe, targeted downloader for missing weights (to be run ONLY when authorized).

Usage:
  python3 pipeline_agent_manager.py --status
  python3 pipeline_agent_manager.py --status --json
  python3 pipeline_agent_manager.py --export-reqs
  python3 pipeline_agent_manager.py --download <pipeline_id>   (Only when authorized!)
"""

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if SCRIPT_DIR.name == "tools" and SCRIPT_DIR.parent.name in ("FAST", "FAST V5"):
    BASE_DIR = SCRIPT_DIR.parents[2]
elif SCRIPT_DIR.name == "tools":
    BASE_DIR = SCRIPT_DIR.parents[1]
else:
    BASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = BASE_DIR
MANIFEST_PATH = SCRIPT_DIR / "PIPELINE_AGENT_MANIFEST.json"


def load_manifest() -> Dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found at: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def check_module_installed(module_name: str) -> bool:
    clean_name = module_name.split(">=")[0].split("==")[0].split("<")[0].split(";")[0].strip()
    # Normalize known package -> module mapping
    mapping = {
        "opencv-python": "cv2",
        "opencv-python-headless": "cv2",
        "pillow": "PIL",
        "pyyaml": "yaml",
        "scikit-image": "skimage",
        "imageio-ffmpeg": "imageio_ffmpeg",
        "faster-coco-eval": "faster_coco_eval",
        "huggingface-hub": "huggingface_hub",
        "openpyxl": "openpyxl"
    }
    import_target = mapping.get(clean_name.lower(), clean_name.replace("-", "_"))
    return importlib.util.find_spec(import_target) is not None


def resolve_path(p: str) -> Path:
    expanded = os.path.expanduser(p)
    path_obj = Path(expanded)
    if not path_obj.is_absolute():
        path_obj = BASE_DIR / path_obj
    if not path_obj.exists():
        posix = path_obj.as_posix()
        if "5-pipeline's/FAST/" in posix:
            alt = Path(posix.replace("5-pipeline's/FAST/", "5-pipeline's/FAST V5/"))
            if alt.exists():
                return alt
        elif posix.endswith("5-pipeline's/FAST"):
            alt = Path(posix.replace("5-pipeline's/FAST", "5-pipeline's/FAST V5"))
            if alt.exists():
                return alt
    return path_obj


def format_bytes(size_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(size_bytes) < 1024.0:
            return f"{size_bytes:3.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def audit_pipeline(pipeline_id: str, pinfo: Dict[str, Any]) -> Dict[str, Any]:
    # Check directory
    pdir = resolve_path(pinfo["directory"])
    dir_exists = pdir.exists()

    # Check dependencies
    dep_status = {}
    missing_deps = []
    for dep in pinfo.get("pip_dependencies", []):
        installed = check_module_installed(dep)
        dep_status[dep] = installed
        if not installed:
            missing_deps.append(dep)

    # Check weights
    weight_status = []
    missing_weights = []
    for w in pinfo.get("weights", []):
        vpath_str = w.get("verification_path", "")
        if not vpath_str:
            continue
        vpath = resolve_path(vpath_str)
        exists = vpath.exists()
        size_str = "0 B"
        if exists:
            try:
                size_str = format_bytes(vpath.stat().st_size)
            except Exception:
                pass
        else:
            missing_weights.append(w.get("model_name", w.get("filename", "unknown")))

        weight_status.append({
            "model_name": w.get("model_name"),
            "filename": w.get("filename"),
            "target_path": str(vpath),
            "exists": exists,
            "size": size_str
        })

    ready = (len(missing_deps) == 0 and len(missing_weights) == 0 and dir_exists)

    return {
        "pipeline_id": pipeline_id,
        "display_name": pinfo.get("display_name"),
        "category": pinfo.get("category"),
        "directory": str(pdir),
        "directory_exists": dir_exists,
        "is_ready": ready,
        "missing_dependencies": missing_deps,
        "missing_weights": missing_weights,
        "dependencies": dep_status,
        "weights": weight_status
    }


def run_audit(manifest: Dict[str, Any]) -> Dict[str, Any]:
    pipelines = manifest.get("pipelines", {})
    results = {}
    for pid, pinfo in pipelines.items():
        results[pid] = audit_pipeline(pid, pinfo)
    return results


def print_status_table(results: Dict[str, Any]) -> None:
    print("\n" + "=" * 90)
    print("  RESEARCH PIPELINES AUDIT REPORT (NO DOWNLOADS EXECUTED)")
    print("=" * 90)
    print(f"{'Pipeline ID':<28} | {'Category':<16} | {'Code':<6} | {'Weights':<14} | {'Status':<10}")
    print("-" * 90)

    for pid, res in results.items():
        cat = res["category"]
        code_str = "OK" if res["directory_exists"] else "MISSING"
        
        w_total = len(res["weights"])
        w_present = sum(1 for w in res["weights"] if w["exists"])
        w_str = f"{w_present}/{w_total} files" if w_total > 0 else "N/A (sensor)"

        if res["is_ready"]:
            status_str = "[READY]"
        elif not res["directory_exists"]:
            status_str = "[DIR MISS]"
        elif res["missing_weights"] and res["missing_dependencies"]:
            status_str = "[NEEDS W+D]"
        elif res["missing_weights"]:
            status_str = "[NEEDS WT]"
        else:
            status_str = "[NEEDS DEP]"

        print(f"{pid:<28} | {cat:<16} | {code_str:<6} | {w_str:<14} | {status_str:<10}")

    print("=" * 90)
    print("\nDetailed Breakdown:")
    for pid, res in results.items():
        print(f"\n--> [{pid}] {res['display_name']}")
        print(f"    Directory: {res['directory']} ({'FOUND' if res['directory_exists'] else 'NOT FOUND'})")
        
        if res["weights"]:
            print("    Weight Checkpoints:")
            for w in res["weights"]:
                icon = "[+]" if w["exists"] else "[-]"
                print(f"      {icon} {w['model_name']} ({w['filename']}) -> {w['size']} at {w['target_path']}")
        
        if res["missing_dependencies"]:
            print(f"    Missing Python Packages ({len(res['missing_dependencies'])}): {', '.join(res['missing_dependencies'])}")
        else:
            print("    Python Dependencies: All checked packages available.")

    print("\n" + "=" * 90 + "\n")


def export_requirements(manifest: Dict[str, Any]) -> None:
    # 1. Export per-pipeline requirements directly inside each pipeline directory
    for pid, pinfo in manifest.get("pipelines", {}).items():
        deps = pinfo.get("pip_dependencies", [])
        gpu_deps = pinfo.get("gpu_pip_dependencies", [])
        p_dir = BASE_DIR / pinfo.get("directory", "")
        if deps and p_dir.exists():
            p_file = p_dir / "requirements.txt"
            with open(p_file, "w", encoding="utf-8") as f:
                f.write(f"# Requirements for {pinfo.get('display_name')}\n")
                f.write(f"# Recommended Python: {pinfo.get('python_version_recommended', '>=3.8')}\n\n")
                f.write("# Core CPU / Standard Dependencies:\n")
                for d in deps:
                    f.write(f"{d}\n")
                if gpu_deps:
                    f.write("\n# Optional GPU Acceleration Dependencies (Select compatible CUDA version):\n")
                    for gd in gpu_deps:
                        f.write(f"# {gd}\n")
            print(f"[+] Exported {pinfo.get('directory')}/requirements.txt")

    # 2. Harmonized dependencies (resolving version overlaps and OpenCV conflict)
    # Never mix opencv-python and opencv-python-headless in the same environment.
    headless_curated = [
        "torch>=2.0.1",
        "torchvision>=0.15.2",
        "torchaudio>=0.8.1",
        "ultralytics>=8.0.59",
        "insightface>=0.7.3",
        "egoblur>=2.0.1",
        "onnx>=1.13.0",
        "onnxruntime>=1.14.0",
        "opencv-python-headless>=4.8.0,<5.0.0",  # Unified Headless OpenCV
        "numpy>=1.24.0,<3.0.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.2.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0,<5.0.0",
        "psutil>=5.9.1",
        "requests>=2.28.0",
        "huggingface_hub>=0.19.0",
        "datasets>=2.14.0",
        "faster-coco-eval>=1.6.6",
        "pycocotools>=2.0.6",
        "gdown>=4.7.0",
        "imageio>=2.9.0",
        "imageio-ffmpeg>=0.4.5",
        "moviepy>=1.0.3,<2.0.0",
        "av>=12.0.0,<14.0.0",
        "omegaconf>=2.3.0",
        "albumentations>=0.5.2",
        "kornia>=0.5.0",
        "openpyxl>=3.1.0",
        "pytest>=7.0.0",
    ]

    desktop_curated = [
        d.replace("opencv-python-headless>=4.8.0,<5.0.0", "opencv-python>=4.8.0")
        for d in headless_curated
    ] + ["pyside6>=6.5.1.1"]

    # Export Unified Headless (Default for 100K Batch Evaluation) inside FAST/tools
    all_file = SCRIPT_DIR / "requirements_all.txt"
    with open(all_file, "w", encoding="utf-8") as f:
        f.write("# =====================================================================\n")
        f.write("# UNIFIED BATCH/SERVER REQUIREMENTS (Clean, Conflict-Free)\n")
        f.write("# Optimized for batch evaluation across all 6 models on 100,000 frames.\n")
        f.write("# OpenCV is harmonized to 'opencv-python-headless' to avoid GUI/X11 conflicts.\n")
        f.write("# =====================================================================\n\n")
        for d in headless_curated:
            f.write(f"{d}\n")
        f.write("\n# GPU ACCELERATION NOTE:\n")
        f.write("# For CUDA-accelerated inference, install onnxruntime-gpu matching your CUDA driver:\n")
        f.write("#   CUDA 12.x: pip install onnxruntime-gpu>=1.16.0\n")
        f.write("#   CUDA 11.x: pip install onnxruntime-gpu==1.14.0\n")
        f.write("# For TensorRT engines (FAST v5 Pipeline), ensure TensorRT and pycuda/pynvml are installed.\n")
    print(f"[+] Exported unified conflict-free {all_file.name}")

    # Export Desktop/GUI variant inside FAST/tools
    desktop_file = SCRIPT_DIR / "requirements_all_desktop.txt"
    with open(desktop_file, "w", encoding="utf-8") as f:
        f.write("# =====================================================================\n")
        f.write("# UNIFIED DESKTOP/GUI REQUIREMENTS\n")
        f.write("# For local workstations needing PySide6 Qt GUI for DashcamCleaner.\n")
        f.write("# Uses 'opencv-python' (standard GUI backend).\n")
        f.write("# DO NOT install 'opencv-python-headless' in the same environment!\n")
        f.write("# =====================================================================\n\n")
        for d in sorted(desktop_curated):
            f.write(f"{d}\n")
    print(f"[+] Exported desktop variant {desktop_file.name}")


def download_pipeline_weights(pipeline_id: str, manifest: Dict[str, Any]) -> None:
    pipelines = manifest.get("pipelines", {})
    if pipeline_id not in pipelines:
        print(f"[-] Unknown pipeline: {pipeline_id}")
        print(f"    Available: {list(pipelines.keys())}")
        return

    pinfo = pipelines[pipeline_id]
    print(f"\n[=>] Initiating weight download for {pinfo.get('display_name')}...")
    weights = pinfo.get("weights", [])
    if not weights:
        print("    No standalone weights registered for this pipeline.")
        return

    for w in weights:
        dest_dir = resolve_path(w["destination_dir"])
        dest_dir.mkdir(parents=True, exist_ok=True)
        vpath = resolve_path(w["verification_path"])
        
        if vpath.exists() and vpath.stat().st_size > 0:
            print(f"    [*] Already present: {vpath.name} ({format_bytes(vpath.stat().st_size)}) -> Skipping.")
            continue

        method = w.get("download_method")
        print(f"    [+] Downloading {w.get('model_name')} via method: {method}...")

        if method == "http_direct":
            url = w.get("download_url")
            cmd = ["curl", "-L", url, "-o", str(vpath)]
            print(f"        Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        elif method in ("google_drive", "gdown_folder"):
            folder_url = w.get("gdrive_folder_url")
            cmd = ["gdown", "--folder", folder_url, "-O", str(dest_dir)]
            print(f"        Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        elif method in ("gdown_file", "gdown"):
            file_id = w.get("gdrive_file_id") or w.get("file_id")
            cmd = ["gdown", file_id, "-O", str(vpath)]
            print(f"        Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        elif method in ("huggingface_hub_or_portal", "huggingface_spaces"):
            url = w.get("download_url")
            cmd = ["curl", "-L", url, "-o", str(vpath)]
            print(f"        Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

        elif method == "insightface_api":
            cmd = [sys.executable, "-c", "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l'); app.prepare(ctx_id=-1)"]
            print(f"        Running InsightFace model zoo downloader...")
            subprocess.run(cmd, check=True)

        else:
            print(f"        [!] Please follow instruction: {w.get('download_command_note', 'See manifest')}")

        if vpath.exists():
            print(f"        [+] Verified: {vpath.name} ({format_bytes(vpath.stat().st_size)})")
        else:
            print(f"        [-] Warning: Expected output {vpath} not found after download step.")


def install_dependencies(pipeline_id: str, manifest: Dict[str, Any]) -> None:
    req_dir = BASE_DIR / "environments"
    if not (req_dir / "requirements_all.txt").exists():
        export_requirements(manifest)

    if pipeline_id == "all":
        req_file = req_dir / "requirements_all.txt"
        print(f"\n[=>] Installing unified conflict-free dependencies from {req_file.name}...")
        print("    [*] Tip: Standardized on 'opencv-python-headless' to eliminate GUI/X11 conflicts.")
        print("    [*] GPU Note: After this step, install 'onnxruntime-gpu' matching your CUDA version (CUDA 12: >=1.16.0).")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        print(f"    Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    elif pipeline_id in manifest.get("pipelines", {}):
        req_file = req_dir / f"requirements_{pipeline_id}.txt"
        if not req_file.exists():
            export_requirements(manifest)
        print(f"\n[=>] Installing dependencies for {pipeline_id} from {req_file.name}...")
        cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
        print(f"    Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print(f"[-] Unknown pipeline: {pipeline_id}")
        print(f"    Available: {list(manifest.get('pipelines', {}).keys())} or 'all'")


def main():
    parser = argparse.ArgumentParser(description="Master AI Agent Manager for Anonymization Pipelines")
    parser.add_argument("--status", "--check", action="store_true", help="Run a non-destructive audit of all pipelines, dependencies, and weights.")
    parser.add_argument("--json", action="store_true", help="Emit audit output in raw JSON format for machine parsing.")
    parser.add_argument("--export-reqs", action="store_true", help="Export individual and unified requirements.txt files.")
    parser.add_argument("--download", type=str, metavar="PIPELINE_ID", help="Download missing weights for a specific pipeline (or 'all_missing').")
    parser.add_argument("--install-deps", type=str, metavar="PIPELINE_ID_OR_ALL", help="Automatically install python dependencies for a pipeline or 'all'.")

    args = parser.parse_args()
    if args.install_deps == "proposed_fast_v3":
        args.install_deps = "proposed_fast_v5"
    if args.download == "proposed_fast_v3":
        args.download = "proposed_fast_v5"
    manifest = load_manifest()

    if args.export_reqs:
        export_requirements(manifest)
        return

    if args.install_deps:
        install_dependencies(args.install_deps, manifest)
        return

    if args.download:
        if args.download == "all_missing":
            for pid in manifest.get("pipelines", {}).keys():
                download_pipeline_weights(pid, manifest)
        else:
            download_pipeline_weights(args.download, manifest)
        return

    # Default to status audit
    results = run_audit(manifest)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_status_table(results)


if __name__ == "__main__":
    main()
