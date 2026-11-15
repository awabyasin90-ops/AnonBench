# -*- coding: utf-8 -*-
"""
run_research_audit.py - OPINA Master Orchestrator
===================================================
Single command that runs the entire research automation pipeline:

  Step 1 — Condition Classifier     (C4 partial: labels current data)
  Step 2 — Sobel Gradient Analysis  (I1: failure pattern taxonomy)
  Step 3 — Recovery Attack          (I3: re-identification risk)
  Step 4 — Pseudo-GT Recall         (C1: recall estimation, if oracle available)
  Step 5 — Tracker Update           (auto-populates Research Tracker matrices)

Usage (minimal — blurred frames only):
    python run_research_audit.py
        --blurred-dir   "C:/...front_camera"
        --detections-csv "../detections_all.csv"
        --session-id    R001
        --source        real_car
        --output-dir    "./outputs/R001"

Usage (full — with original frames for true recall + similarity attack):
    python run_research_audit.py
        --blurred-dir    "C:/.../front_camera_blurred"
        --original-dir   "C:/.../front_camera_original"
        --detections-csv "../detections_all.csv"
        --session-id     R001
        --source         real_car
        --output-dir     "./outputs/R001"
        --oracle-conf    0.50

Skipping steps:
    --skip-recall       Skip pseudo-GT recall (no ultralytics needed)
    --skip-attack       Skip recovery attack
    --skip-sobel        Skip Sobel analysis
    --attack-mode       laplacian|ocr|similarity|all (default: laplacian)

Output tree:
    outputs/R001/
    ├── frame_conditions.csv               (Step 1)
    ├── sobel_detail.csv                   (Step 2)
    ├── failure_patterns_by_condition.csv  (Step 2)
    ├── failure_taxonomy_table.tex         (Step 2 — paste into paper)
    ├── failure_taxonomy_report.json       (Step 2)
    ├── recovery_attack_detail.csv         (Step 3)
    ├── recovery_attack_summary.csv        (Step 3)
    ├── recovery_attack_report.json        (Step 3)
    ├── pseudo_gt_boxes.csv                (Step 4)
    ├── per_frame_recall.csv               (Step 4)
    ├── recall_by_condition.csv            (Step 4)
    └── tracker/
        ├── matrix1_dataset_tracking.csv
        ├── matrix2_audit_matrix.csv
        └── matrix3_condition_performance.csv
"""

import sys
import argparse
import time
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# Imports from sibling modules
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import condition_classifier
import sobel_gradient_analysis
import recovery_attack      as ra
import tracker_updater      as tu

try:
    import pseudo_gt_recall as pg
    HAS_RECALL_MOD = True
except ImportError:
    HAS_RECALL_MOD = False


def banner(title: str):
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf_8'):
        import io, sys as _sys
        _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8', errors='replace')
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tracker_out = out / "tracker"

    total_start = time.monotonic()

    # ── STEP 1: Condition Classifier ─────────────────────────────
    banner("STEP 1 / 5 — Auto Condition Classification")
    cond_csv = str(out / "frame_conditions.csv")
    condition_classifier.run(
        image_dir  = args.blurred_dir,
        output_csv = cond_csv,
        source     = args.source,
        workers    = args.workers,
    )

    # ── STEP 2: Sobel Gradient Analysis ──────────────────────────
    if not args.skip_sobel:
        banner("STEP 2 / 5 — Sobel Gradient Failure Analysis  (I1)")
        sobel_gradient_analysis.run(
            blurred_dir    = args.blurred_dir,
            detections_csv = args.detections_csv,
            condition_csv  = cond_csv,
            output_dir     = str(out),
            original_dir   = args.original_dir,
        )
    else:
        print("[Orchestrator] Skipping Sobel analysis (--skip-sobel)")

    # ── STEP 3: Recovery Attack ───────────────────────────────────
    attack_csv = ""
    if not args.skip_attack:
        banner("STEP 3 / 5 — Recovery Attack Validation  (I3)")
        ra.run(
            blurred_dir          = args.blurred_dir,
            detections_csv       = args.detections_csv,
            condition_csv        = cond_csv,
            output_dir           = str(out),
            original_dir         = args.original_dir,
            mode                 = args.attack_mode,
            max_boxes_per_frame  = args.max_boxes_per_frame,
        )
        attack_csv = str(out / "recovery_attack_summary.csv")
        try:
            import hitl_review_pack
            hitl_review_pack.generate_review_pack(
                recovery_detail_csv = out / "recovery_attack_detail.csv",
                detections_csv      = Path(args.detections_csv),
                blurred_dir         = Path(args.blurred_dir),
                output_dir          = out,
                original_dir        = Path(args.original_dir) if args.original_dir else None
            )
        except Exception as e:
            print(f"[Orchestrator] Warning generating HITL review pack: {e}")
    else:
        print("[Orchestrator] Skipping recovery attack (--skip-attack)")

    # ── STEP 4: Pseudo-GT Recall ──────────────────────────────────
    recall_csv = ""
    if not args.skip_recall and HAS_RECALL_MOD:
        if args.original_dir:
            banner("STEP 4 / 5 — Pseudo-GT Recall Evaluation  (C1)")
            pg.run(
                original_dir   = args.original_dir,
                detections_csv = args.detections_csv,
                condition_csv  = cond_csv,
                output_dir     = str(out),
                oracle_conf    = args.oracle_conf,
            )
            recall_csv = str(out / "recall_by_condition.csv")
        else:
            print("[Orchestrator] Skipping recall — no --original-dir provided.")
            print("  Recall estimation requires unblurred source frames.")
    elif args.skip_recall:
        print("[Orchestrator] Skipping recall (--skip-recall)")

    # ── STEP 5: Update Research Tracker ───────────────────────────
    banner("STEP 5 / 5 — Update Research Tracker Matrices")
    import pandas as pd
    cond_df     = pd.read_csv(cond_csv)
    cond_counts = cond_df["condition_name"].value_counts().to_dict()
    recall_data = tu.load_recall_data(recall_csv)
    attack_data = tu.load_attack_data(attack_csv)
    energy_data = tu.load_energy_data(args.energy_dir or "")

    tu.update_tracker_csv(
        output_dir       = str(tracker_out),
        session_id       = args.session_id,
        source           = args.source,
        condition_counts = cond_counts,
        detections_csv   = args.detections_csv,
        condition_csv    = cond_csv,
        attack_data      = attack_data,
        energy_data      = energy_data,
        recall_data      = recall_data,
        tracker_xlsx     = args.tracker_xlsx,
    )

    # ── Done ──────────────────────────────────────────────────────
    elapsed = time.monotonic() - total_start
    banner(f"DONE — Total runtime: {elapsed:.1f}s")
    print(f"\n  All outputs in: {out.resolve()}")
    print(f"\n  Quick reference:")
    print(f"    Conditions:      {out / 'frame_conditions.csv'}")
    print(f"    Failure taxonomy:{out / 'failure_patterns_by_condition.csv'}")
    print(f"    LaTeX table:     {out / 'failure_taxonomy_table.tex'}")
    print(f"    Recovery attack: {out / 'recovery_attack_summary.csv'}")
    print(f"    Recall:          {out / 'recall_by_condition.csv'}")
    print(f"    Tracker:         {tracker_out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OPINA Research Audit — Master Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # ── Required ──────────────────────────────────────────────────
    parser.add_argument("--blurred-dir",    required=True,
                        help="Directory with blurred output frames from FAST v4")
    parser.add_argument("--detections-csv", required=True,
                        help="detections_all.csv from FAST v4 privacy_report")
    parser.add_argument("--session-id",     required=True,
                        help="Session ID tag (e.g. R001, S001, T001)")

    # ── Optional ──────────────────────────────────────────────────
    parser.add_argument("--original-dir",   default=None,
                        help="Original (unblurred) frames — enables similarity + recall")
    parser.add_argument("--output-dir",     default="./outputs",
                        help="Root output directory (default: ./outputs)")
    parser.add_argument("--source",         default="real_car",
                        choices=["real_car", "carla", "tesla"],
                        help="Data source for tracker (default: real_car)")
    parser.add_argument("--tracker-xlsx",   default=None,
                        help="Path to Research_Tracker.xlsx to update directly")
    parser.add_argument("--energy-dir",     default=None,
                        help="Directory with energy_result_*.json files")
    parser.add_argument("--oracle-conf",    type=float, default=0.50,
                        help="Oracle confidence for recall evaluation (default 0.50)")
    parser.add_argument("--attack-mode",    default="laplacian",
                        choices=["laplacian", "ocr", "similarity", "all"],
                        help="Recovery attack method (default: laplacian)")
    parser.add_argument("--workers",        type=int, default=4,
                        help="Threads for condition classifier (default 4)")
    parser.add_argument("--max-boxes-per-frame", type=int, default=0,
                        help="Max face boxes to test per frame in attack (0 = all detections)")

    # ── Skip flags ────────────────────────────────────────────────
    parser.add_argument("--skip-recall",  action="store_true",
                        help="Skip pseudo-GT recall step")
    parser.add_argument("--skip-attack",  action="store_true",
                        help="Skip recovery attack step")
    parser.add_argument("--skip-sobel",   action="store_true",
                        help="Skip Sobel gradient analysis step")

    args = parser.parse_args()

    # Auto-set output-dir to include session-id
    if args.output_dir == "./outputs":
        args.output_dir = f"./outputs/{args.session_id}"

    run(args)
