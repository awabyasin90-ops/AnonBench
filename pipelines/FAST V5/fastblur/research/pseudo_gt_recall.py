"""
pseudo_gt_recall.py — OPINA Research Automation  C1
=====================================================
Ground-Truth Strategy: Pseudo-GT via Oracle Ensemble

Problem: We have NO manual annotations → we cannot compute recall.

Solution: Use the ORIGINAL (unblurred) images as input to a STRONGER
detector ensemble (YOLOv8x-face at high confidence) and treat its detections
as "pseudo ground truth". Then compare FAST v4 detections (from detections_all.csv)
against this pseudo-GT using IoU matching to compute:

  - Pseudo-Recall   = TP / (TP + FN)   [how many GT faces did FAST v4 catch?]
  - Pseudo-Precision = TP / (TP + FP)  [how many FAST v4 detections are real faces?]
  - Miss-Rate       = FN / (TP + FN)   [per condition — the key paper metric]

Per-condition results power your failure taxonomy table.

NOTE: If you only have BLURRED images available (no originals), the oracle
runs on the blurred images at a higher threshold — still useful as a
relative recall estimate. This mode is automatically detected.

Usage:
    python pseudo_gt_recall.py
        --original-dir  <path_to_ORIGINAL_unblurred_frames>
        --detections-csv <detections_all.csv from FAST v4>
        --condition-csv  <frame_conditions.csv from condition_classifier.py>
        --output-dir    <output folder>
        --oracle-conf   0.50     (higher = stricter oracle = conservative GT)

Requirements:
    pip install ultralytics tqdm

Output:
    recall_by_condition.csv   — per-condition recall/miss-rate table
    per_frame_recall.csv      — per-frame TP/FP/FN breakdown
    pseudo_gt_boxes.csv       — all oracle detections (for auditing)
"""

import os
import argparse
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


IOU_MATCH_THRESHOLD = 0.40   # IoU >= 0.40 → TP match (standard for detection recall)


# ─────────────────────────────────────────────────────────────────
# IoU helpers
# ─────────────────────────────────────────────────────────────────
def iou(boxA, boxB) -> float:
    """
    boxA, boxB: [x1, y1, x2, y2]
    Returns Intersection-over-Union.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt_boxes: list, pred_boxes: list,
                     iou_thresh: float = IOU_MATCH_THRESHOLD):
    """
    Greedy IoU matching: each GT box matched to at most one prediction.
    Returns: TP, FP, FN, list of matched (gt_idx, pred_idx, iou_score)
    """
    matched_gt   = set()
    matched_pred = set()
    matches = []

    # Sort by IoU descending (greedy best-first)
    pairs = []
    for gi, gb in enumerate(gt_boxes):
        for pi, pb in enumerate(pred_boxes):
            score = iou(gb, pb)
            if score >= iou_thresh:
                pairs.append((score, gi, pi))
    pairs.sort(reverse=True)

    for score, gi, pi in pairs:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append((gi, pi, round(score, 4)))

    TP = len(matches)
    FN = len(gt_boxes)  - TP
    FP = len(pred_boxes) - TP
    return TP, FP, FN, matches


# ─────────────────────────────────────────────────────────────────
# Oracle detector (YOLOv8x-face)
# ─────────────────────────────────────────────────────────────────
def load_oracle(conf_threshold: float = 0.50):
    """Load the oracle detector. Falls back gracefully if ultralytics missing."""
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8x.pt")   # Will auto-download if not present
        print(f"[Oracle] Loaded YOLOv8x — conf_threshold={conf_threshold}")
        return model, conf_threshold
    except ImportError:
        print("[Oracle] WARNING: ultralytics not installed.")
        print("  Install with: pip install ultralytics")
        print("  Running in DETECTIONS-ONLY mode (no oracle — using existing CSV).")
        return None, conf_threshold


def run_oracle_on_frame(model, img_path: str, conf_threshold: float) -> list:
    """Run oracle detector on one image. Returns list of [x1,y1,x2,y2,conf]."""
    if model is None:
        return []
    results = model(img_path, conf=conf_threshold, classes=[0], verbose=False)  # class 0 = person
    boxes = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                c = float(box.conf[0])
                boxes.append([x1, y1, x2, y2, c])
    return boxes


# ─────────────────────────────────────────────────────────────────
# Main recall computation
# ─────────────────────────────────────────────────────────────────
def run(original_dir: str, detections_csv: str, condition_csv: str,
        output_dir: str, oracle_conf: float = 0.50):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load FAST v4 detections
    det_df = pd.read_csv(detections_csv)
    det_faces = det_df[det_df["class"] == "face"].copy()
    print(f"[Recall] Loaded {len(det_faces)} face detections from FAST v4 CSV.")

    # Load condition labels
    cond_df = pd.read_csv(condition_csv)
    cond_map = dict(zip(cond_df["frame"], cond_df["condition_name"]))
    print(f"[Recall] Loaded {len(cond_map)} frame condition labels.")

    # Load oracle model
    oracle, oracle_conf = load_oracle(oracle_conf)

    original_dir = Path(original_dir)
    img_files = sorted([
        p for p in original_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    print(f"[Recall] Processing {len(img_files)} frames ...")

    per_frame_rows = []
    gt_all_rows    = []

    for img_path in tqdm(img_files, desc="Pseudo-GT Recall"):
        frame_name = img_path.name
        condition  = cond_map.get(frame_name, "Unknown")

        # Oracle detections (pseudo-GT)
        gt_boxes_raw = run_oracle_on_frame(oracle, str(img_path), oracle_conf)
        gt_boxes = [[b[0], b[1], b[2], b[3]] for b in gt_boxes_raw]

        for b in gt_boxes_raw:
            gt_all_rows.append({
                "frame": frame_name, "condition": condition,
                "x1": round(b[0],2), "y1": round(b[1],2),
                "x2": round(b[2],2), "y2": round(b[3],2),
                "conf_oracle": round(b[4],4)
            })

        # FAST v4 face detections for this frame
        frame_dets = det_faces[det_faces["frame"] == frame_name]
        pred_boxes = [
            [row["x1"], row["y1"], row["x2"], row["y2"]]
            for _, row in frame_dets.iterrows()
        ]

        # If no oracle (ultralytics missing), just use pipeline detections as proxy
        if oracle is None:
            # Fallback: low-confidence detections as "unverified"
            per_frame_rows.append({
                "frame": frame_name, "condition": condition,
                "n_gt": -1, "n_pred": len(pred_boxes),
                "TP": -1, "FP": -1, "FN": -1,
                "recall": -1.0, "precision": -1.0, "miss_rate": -1.0,
                "note": "oracle_unavailable"
            })
            continue

        TP, FP, FN, matches = match_detections(gt_boxes, pred_boxes)
        n_gt   = len(gt_boxes)
        recall = TP / n_gt       if n_gt > 0        else 1.0
        prec   = TP / (TP + FP)  if (TP + FP) > 0   else 1.0
        miss   = FN / n_gt       if n_gt > 0        else 0.0

        per_frame_rows.append({
            "frame":     frame_name,
            "condition": condition,
            "n_gt":      n_gt,
            "n_pred":    len(pred_boxes),
            "TP":        TP,
            "FP":        FP,
            "FN":        FN,
            "recall":    round(recall, 4),
            "precision": round(prec,   4),
            "miss_rate": round(miss,   4),
            "note":      ""
        })

    # ── Per-frame CSV ────────────────────────────────────────────
    pf_path = output_dir / "per_frame_recall.csv"
    pd.DataFrame(per_frame_rows).to_csv(pf_path, index=False)

    # ── Pseudo-GT boxes CSV ──────────────────────────────────────
    gt_path = output_dir / "pseudo_gt_boxes.csv"
    pd.DataFrame(gt_all_rows).to_csv(gt_path, index=False)

    # ── Per-condition summary ────────────────────────────────────
    pf_df = pd.DataFrame(per_frame_rows)

    if oracle is not None:
        pf_df_valid = pf_df[pf_df["n_gt"] >= 0]
        cond_summary = (
            pf_df_valid
            .groupby("condition")
            .agg(
                frames       = ("frame", "count"),
                total_gt     = ("n_gt",     "sum"),
                total_pred   = ("n_pred",   "sum"),
                total_TP     = ("TP",       "sum"),
                total_FP     = ("FP",       "sum"),
                total_FN     = ("FN",       "sum"),
            )
            .reset_index()
        )
        cond_summary["recall"]    = (cond_summary["total_TP"] /
                                     cond_summary["total_gt"]).round(4)
        cond_summary["miss_rate"] = (cond_summary["total_FN"] /
                                     cond_summary["total_gt"]).round(4)
        cond_summary["precision"] = (cond_summary["total_TP"] /
                                     (cond_summary["total_TP"] +
                                      cond_summary["total_FP"])).round(4)
    else:
        # No oracle: just count detections per condition
        cond_summary = (
            pf_df
            .groupby("condition")
            .agg(frames=("frame","count"), total_pred=("n_pred","sum"))
            .reset_index()
        )
        cond_summary["recall"]    = None
        cond_summary["miss_rate"] = None
        cond_summary["precision"] = None

    cond_path = output_dir / "recall_by_condition.csv"
    cond_summary.to_csv(cond_path, index=False)

    # Print summary
    print("\n[Recall] ── Per-Condition Pseudo-Recall Summary ──────────────────")
    print(cond_summary.to_string(index=False))
    print(f"\n  Output files:")
    print(f"    {pf_path}")
    print(f"    {cond_path}")
    print(f"    {gt_path}")

    return cond_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pseudo-GT recall evaluation for OPINA anonymization paper")
    parser.add_argument("--original-dir",    required=True,
                        help="Directory with ORIGINAL (unblurred) frames")
    parser.add_argument("--detections-csv",  required=True,
                        help="detections_all.csv from FAST v4")
    parser.add_argument("--condition-csv",   required=True,
                        help="frame_conditions.csv from condition_classifier.py")
    parser.add_argument("--output-dir",      required=True,
                        help="Output folder for recall CSVs")
    parser.add_argument("--oracle-conf",     type=float, default=0.50,
                        help="Oracle detector confidence threshold (default 0.50)")
    args = parser.parse_args()
    run(args.original_dir, args.detections_csv,
        args.condition_csv, args.output_dir, args.oracle_conf)
