#!/usr/bin/env python3
"""
run_blur_v5.py — FAST v5.0-Research Master Pipeline
===================================================
Autonomous Vehicle Production Privacy Anonymization & Research Audit Suite.

Features:
  - Elliptical face masking & feathered anti-aliasing (FAST v4 core)
  - Single JPEG encode guarantee
  - Real-time Hardware Power & Energy Profiling (kW, kWh, Joules, Wh/1k frames)
  - Automated Condition Stratification (6 Research Tracker conditions)
  - Person-Anchored Ground Truth Model (PAG-Model: ISO/IEC 19794-5 biometric resolution)
  - License Plate Readability Model (OCR / stroke tracking test)
  - Sobel & Failure Taxonomy LaTeX Generation
  - Research Tracker Synchronization (Matrix 1, 2, 3 with # | tables & Research_Tracker.xlsx sync)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
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

VERSION = "5.0.0-Research"
ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "config" / "pipeline_config.yaml"
SCAN_JSON = ROOT / "run_facts" / "hw_scan.json"
AUTO_JSON = ROOT / "run_facts" / "auto_tune.json"
VAL_SCRIPT = ROOT / "tools" / "validate_report.py"

# Import research automation package
sys.path.insert(0, str(ROOT))
from fastblur.research.energy_monitor import PowerSampler, EnergyMonitor
from fastblur.research.run_research_audit import run as run_research_audit

def banner(gpu_name: str = "", tier: str = "") -> None:
    extra = f"  GPU: {gpu_name}  Tier {tier}" if gpu_name else ""
    print(f"""
{'='*72}
  FAST v{VERSION}  Autonomous Vehicle Privacy & Research Suite
  Elliptical Blur | Energy Telemetry (kW/kWh) | PAG Ground Truth | Tracker Sync
{extra}
{'='*72}
""")

def main() -> None:
    parser = argparse.ArgumentParser(description=f"FAST v{VERSION} Production & Research Pipeline")
    parser.add_argument("--config", "-c", type=Path, default=DEFAULT_CFG, help="Pipeline YAML config")
    parser.add_argument("--input-dir", "-i", type=Path, default=None, help="Input frames directory")
    parser.add_argument("--output-dir", "-o", type=Path, default=None, help="Output blurred frames directory")
    parser.add_argument("--frames", "-n", type=int, default=None, help="Limit frames to process")
    parser.add_argument("--session-id", default="R001", help="Session ID (e.g. R001, S001, T001)")
    parser.add_argument("--source", default="real_car", choices=["real_car", "carla", "tesla"], help="Data source")
    parser.add_argument("--tracker-path", type=Path, default=None, help="Path to Research_Tracker.xlsx")
    parser.add_argument("--original-dir", type=Path, default=None, help="Path to unblurred original frames (for pseudo-GT recall)")
    parser.add_argument("--audit-only", action="store_true", help="Skip blurring; run research audit on existing output")
    parser.add_argument("--no-audit", action="store_true", help="Run blur pipeline only, skip research audit")
    parser.add_argument("--force", "-f", action="store_true", help="Re-process all frames")
    parser.add_argument("--mode", choices=["draft", "release"], default="draft", help="Execution mode")
    args = parser.parse_args()

    banner()

    # Determine input/output paths
    from fastblur.config import load_config
    cfg = load_config(args.config) if args.config.exists() else {}
    
    if args.input_dir:
        input_dir = args.input_dir.resolve()
    else:
        input_dir = (ROOT / cfg.get("input_dir", "inputs")).resolve()

    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = (ROOT / cfg.get("output_dir", "outputs")).resolve()

    cfg["input_dir"] = input_dir
    cfg["output_dir"] = output_dir
    cfg["execution_mode"] = args.mode
    if args.force:
        cfg["skip_existing"] = False

    privacy_rep_dir = output_dir / "privacy_report"
    detections_csv = privacy_rep_dir / "detections_all.csv"

    # Start live hardware energy telemetry
    sampler = PowerSampler(interval_s=0.1)
    pipeline_summary = {}

    if not args.audit_only:
        print(f"[v5] Starting FAST v5 Blur Pipeline on: {input_dir}")
        print(f"[v5] Power monitor engaged -> sampling GPU Watts every 100ms...")
        sampler.start()

        from fastblur.pipeline import Pipeline
        pipeline = Pipeline(cfg, limit=args.frames)
        t0 = time.time()
        pipeline_summary = pipeline.run()
        t1 = time.time()

        sampler.stop()
        sampler.join(timeout=2.0)
        print(f"[v5] Blur stage finished in {t1 - t0:.2f}s ({pipeline_summary.get('images_per_s', 0):.2f} img/s).")
    else:
        print(f"[v5] --audit-only flag set. Skipping blur stage; auditing existing output at: {output_dir}")

    # Gather energy telemetry
    total_wh = sampler.total_energy_wh()
    avg_w = sampler.avg_gpu_power_w()
    n_frames = pipeline_summary.get("images_ok", 0)
    wh_per_1k = (total_wh / n_frames * 1000) if n_frames > 0 else 0.0

    energy_stats = {
        "device": "NVIDIA GPU (NVML)" if sampler._pynvml else "CPU / Baseline",
        "avg_power_w": round(avg_w, 2),
        "avg_power_kw": round(avg_w / 1000.0, 4),
        "total_energy_wh": round(total_wh, 4),
        "total_energy_kwh": round(total_wh / 1000.0, 6),
        "energy_per_1k_frames_wh": round(wh_per_1k, 4),
        "n_samples": len(sampler.samples),
    }

    print()
    print("[v5 Hardware & Energy Telemetry]")
    print(f"  GPU / Device     : {energy_stats.get('device', 'N/A')}")
    print(f"  Avg Power Draw   : {energy_stats.get('avg_power_w', 0.0):.2f} W ({energy_stats.get('avg_power_kw', 0.0):.4f} kW)")
    print(f"  Total Energy     : {energy_stats.get('total_energy_wh', 0.0):.4f} Wh ({energy_stats.get('total_energy_kwh', 0.0):.6f} kWh)")
    if energy_stats.get("energy_per_1k_frames_wh"):
        print(f"  Energy / 1k Img  : {energy_stats.get('energy_per_1k_frames_wh', 0.0):.4f} Wh / 1,000 frames")

    # Save energy summary to output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    energy_json_path = output_dir / "energy_summary.json"
    with open(energy_json_path, "w", encoding="utf-8") as f:
        json.dump(energy_stats, f, indent=2)
    print(f"  [+] Energy telemetry saved -> {energy_json_path}")

    # Execute Autonomous Research Audit Suite
    if not args.no_audit:
        print()
        print("=" * 72)
        print("  LAUNCHING FAST v5 AUTONOMOUS RESEARCH AUDIT SUITE")
        print("=" * 72)

        if not detections_csv.exists():
            # Check if detections_all.csv is in output_dir directly
            if (output_dir / "detections_all.csv").exists():
                detections_csv = output_dir / "detections_all.csv"

        research_out_dir = output_dir / "research_audit" / args.session_id
        research_out_dir.mkdir(parents=True, exist_ok=True)

        tracker_p = args.tracker_path
        if tracker_p is None:
            # Look for default Research_Tracker.xlsx in known locations
            candidates = [
                ROOT.parent.parent / "pendik" / "cameras_only-(2)_blurred" / "front_camera" / "privacy_report" / "03-Anonymization Failure Patterns Across Real Drives and CARLA Simulation" / "Research_Tracker.xlsx",
                ROOT / "Research_Tracker.xlsx",
            ]
            for c in candidates:
                if c.exists():
                    tracker_p = c
                    break

        print(f"[v5 Research] Blurred frames dir : {output_dir}")
        print(f"[v5 Research] Detections CSV     : {detections_csv}")
        print(f"[v5 Research] Output Audit dir   : {research_out_dir}")
        print(f"[v5 Research] Tracker Path       : {tracker_p}")

        if output_dir.exists() and detections_csv.exists():
            class AuditArgs:
                pass
            a = AuditArgs()
            a.blurred_dir = str(output_dir)
            a.detections_csv = str(detections_csv)
            a.session_id = str(args.session_id)
            a.source = str(args.source)
            a.output_dir = str(research_out_dir)
            a.tracker_xlsx = str(tracker_p) if tracker_p else None
            a.original_dir = str(args.original_dir) if args.original_dir else None
            a.energy_dir = str(output_dir)
            a.oracle_conf = 0.50
            a.attack_mode = "all"
            a.workers = 4
            a.max_boxes_per_frame = 0
            a.skip_recall = False if args.original_dir else True
            a.skip_attack = True  # Scope constraint: NO post-blur leakage/reconstruction audits
            a.skip_sobel = True   # Evaluation strictly detection-based

            run_research_audit(a)
            print()
            print("[v5 Research] Full Research Audit Completed successfully!")
            print(f"  -> Dashboard : {research_out_dir / 'RESEARCH_RESULTS_DASHBOARD.md'}")
            print(f"  -> Matrix 1  : {research_out_dir / 'tracker' / 'matrix1_dataset_tracking.md'}")
            print(f"  -> Matrix 2  : {research_out_dir / 'tracker' / 'matrix2_audit_matrix.md'}")
            print(f"  -> Matrix 3  : {research_out_dir / 'tracker' / 'matrix3_condition_performance.md'}")
        else:
            print(f"[v5 Research] Notice: detections_all.csv not found at {detections_csv}. Skipping post-audit.")

    print()
    print(f"[v5] All operations completed successfully! Mode: {args.mode.upper()}")

if __name__ == "__main__":
    main()
