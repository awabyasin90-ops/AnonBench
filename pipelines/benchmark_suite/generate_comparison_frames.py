#!/usr/bin/env python3
"""
generate_comparison_frames.py
==============================
Generates publication-quality multi-system side-by-side comparison collages
across all 5 evaluated privacy pipelines using Ground Truth v.3 (and V.2 fallback)
as the Ground Truth Reference.

Visual Structure per Collage:
  - Card 1: RAW INPUT / GROUND TRUTH REFERENCE
      * Gold/Yellow boxes for all reference GT targets, numbered #1..#N.
      * Translucent Scoreboard HUD displaying verified detection counts per system.
  - Cards 2-6: The 5 Anonymization Systems
      * GREEN boxes for Detected targets (True Positives, #id DETECTED).
      * RED boxes for Missed targets (False Negatives / Privacy Leaks, #id MISSED).
      * BLUE boxes for False Positives (Over-blur / Hallucinations, FP #k).
      * Quadrant headers featuring distinct color-coded pill badges:
        [ X Detected ] (Green) | [ Y Missed ] (Red) | [ Z False+ ] (Blue)
        along with edge hardware telemetry (Peak RAM/VRAM, FPS throughput, Latency).
  - Bottom Legend Banner explaining the color coding and target mapping.

Outputs:
  Saved directly into `3_Systems/outputs/results_pendik_100/blurred/10_Frame_Comparisons/`
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np


SYSTEM_KEYS = [
    ("1_FAST_V5", "2. FAST v5", (30, 120, 60), (40, 180, 80)),
    ("2_EgoBlur", "3. Meta EgoBlur", (110, 40, 130), (160, 60, 190)),
    ("3_DashcamCleaner", "4. DashcamCleaner", (130, 80, 20), (190, 120, 30)),
    ("4_FusionPortable", "5. FusionPortable", (40, 110, 130), (60, 160, 190)),
    ("5_Generative_Anonymizer", "6. Generative Inpaint", (120, 40, 60), (180, 60, 90))
]

SYSTEM_DISPLAY_NAMES = {
    "1_FAST_V5": "FAST v5",
    "2_EgoBlur": "Meta AI EgoBlur",
    "3_DashcamCleaner": "DashcamCleaner",
    "4_FusionPortable": "FusionPortable",
    "5_Generative_Anonymizer": "Generative Inpaint"
}


def calculate_iou_and_iomin(boxA: List[float], boxB: List[float]) -> Tuple[float, float]:
    """Computes both IoU and IoMin between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])

    union_area = areaA + areaB - inter_area
    iou = inter_area / union_area if union_area > 0.0 else 0.0

    min_area = min(areaA, areaB)
    iomin = inter_area / min_area if min_area > 0.0 else 0.0

    return iou, iomin


def get_short_bin(box: List[float], cls_name: str = "face") -> str:
    bw = max(0.0, box[2] - box[0])
    bh = max(0.0, box[3] - box[1])
    sz = bw if "face" in cls_name.lower() else bh
    if sz < 16:
        return "Bin 1: >20m"
    elif sz < 32:
        return "Bin 2: 10-20m"
    elif sz < 64:
        return "Bin 3: 5-10m"
    else:
        return "Bin 4: <5m"


def match_system_detections(
    gt_targets: List[Dict[str, Any]],
    model_dets: List[Dict[str, Any]],
    iou_thresh: float = 0.40,
    iomin_thresh: float = 0.50
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Greedy bipartite matching of system detections against GT targets.
    Returns:
      tp_matches : list of dicts with gt_id, gt_box, model_box, label, iou
      fn_misses  : list of dicts with gt_id, box, label, bin (missed GT targets)
      fp_errors  : list of dicts with fp_id, box, label, bin (unmatched detections)
    """
    clean_model_dets = []
    for md in model_dets:
        cls_name = str(md.get("class", "face")).lower().strip()
        norm_cls = "plate" if ("plate" in cls_name or "license" in cls_name) else "face"
        box = md.get("box", [])
        if len(box) == 4:
            clean_model_dets.append({"label": norm_cls, "box": box, "conf": float(md.get("conf", 0.0))})

    candidate_pairs = []
    for m_idx, md in enumerate(clean_model_dets):
        for gt in gt_targets:
            if md["label"] == gt["class"]:
                iou, iomin = calculate_iou_and_iomin(md["box"], gt["box"])
                if iou >= iou_thresh or iomin >= iomin_thresh:
                    score = max(iou, iomin)
                    candidate_pairs.append((score, m_idx, gt["id"]))

    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    matched_m = set()
    matched_gt = set()
    tp_matches = []

    gt_by_id = {gt["id"]: gt for gt in gt_targets}

    for score, m_idx, gt_id in candidate_pairs:
        if m_idx not in matched_m and gt_id not in matched_gt:
            matched_m.add(m_idx)
            matched_gt.add(gt_id)
            gt_obj = gt_by_id[gt_id]
            tp_matches.append({
                "gt_id": gt_id,
                "gt_box": gt_obj["box"],
                "model_box": clean_model_dets[m_idx]["box"],
                "label": gt_obj["class"],
                "bin": get_short_bin(gt_obj["box"], gt_obj["class"]),
                "score": score
            })

    fn_misses = []
    for gt in gt_targets:
        if gt["id"] not in matched_gt:
            fn_misses.append({
                "gt_id": gt["id"],
                "box": gt["box"],
                "label": gt["class"],
                "bin": get_short_bin(gt["box"], gt["class"])
            })

    fp_errors = []
    fp_counter = 1
    for m_idx, md in enumerate(clean_model_dets):
        if m_idx not in matched_m:
            fp_errors.append({
                "fp_id": fp_counter,
                "box": md["box"],
                "label": md["label"],
                "bin": get_short_bin(md["box"], md["label"]),
                "conf": md["conf"]
            })
            fp_counter += 1

    return tp_matches, fn_misses, fp_errors


def draw_gt_reference_card(
    img: np.ndarray,
    gt_targets: List[Dict[str, Any]],
    systems_scores: Dict[str, Dict[str, int]],
    gt_source: str
) -> np.ndarray:
    """Renders Card 1 with all GT targets numbered #1..#N and a translucent HUD scoreboard."""
    canvas = img.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Gold GT boxes
    for gt in gt_targets:
        box = gt["box"]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cls_tag = "Face" if gt["class"] == "face" else "Plate"
        label = f"#{gt['id']} {cls_tag}"

        color_gold = (0, 215, 255) # BGR Yellow/Gold
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color_gold, 2)

        # Label badge
        font_scale = 0.36
        thickness = 1
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        lbl_y1 = max(0, y1 - lh - 4)
        lbl_y2 = y1
        if lbl_y1 < 2:
            lbl_y1 = min(h - lh - 6, y2 + 2)
            lbl_y2 = lbl_y1 + lh + 4

        cv2.rectangle(canvas, (x1, lbl_y1), (x1 + lw + 6, lbl_y2), (0, 140, 180), -1)
        cv2.putText(canvas, label, (x1 + 3, lbl_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 2. Draw Translucent HUD Scoreboard at bottom dashboard area (y = 815 to 1025)
    sb_x1, sb_y1 = 20, 810
    sb_x2, sb_y2 = w - 20, 1025

    overlay = canvas.copy()
    cv2.rectangle(overlay, (sb_x1, sb_y1), (sb_x2, sb_y2), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.rectangle(canvas, (sb_x1, sb_y1), (sb_x2, sb_y2), (0, 215, 255), 2)

    # Scoreboard Header
    hdr_title = f"GROUND TRUTH REFERENCE: {len(gt_targets)} TARGETS [{gt_source}]"
    cv2.putText(canvas, hdr_title, (sb_x1 + 16, sb_y1 + 26), cv2.FONT_HERSHEY_DUPLEX, 0.56, (0, 220, 255), 1, cv2.LINE_AA)
    cv2.line(canvas, (sb_x1 + 16, sb_y1 + 34), (sb_x2 - 16, sb_y1 + 34), (80, 80, 80), 1)

    # System breakdown lines
    cur_y = sb_y1 + 56
    row_gap = 29

    for s_key, s_short, _, _ in SYSTEM_KEYS:
        sc = systems_scores.get(s_key, {"tp": 0, "fn": len(gt_targets), "fp": 0})
        tp, fn, fp = sc["tp"], sc["fn"], sc["fp"]
        full_name = SYSTEM_DISPLAY_NAMES.get(s_key, s_short)
        rec = (tp / len(gt_targets) * 100.0) if len(gt_targets) > 0 else 0.0

        # System name
        cv2.putText(canvas, f"{full_name:<20}", (sb_x1 + 16, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (240, 240, 240), 1, cv2.LINE_AA)

        # TP pill (Green)
        tp_text = f"{tp:2d} Detected"
        cv2.rectangle(canvas, (sb_x1 + 230, cur_y - 14), (sb_x1 + 340, cur_y + 6), (0, 140, 40), -1)
        cv2.putText(canvas, tp_text, (sb_x1 + 236, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # FN pill (Red)
        fn_text = f"{fn:2d} Missed"
        cv2.rectangle(canvas, (sb_x1 + 355, cur_y - 14), (sb_x1 + 450, cur_y + 6), (20, 20, 180), -1)
        cv2.putText(canvas, fn_text, (sb_x1 + 361, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # FP pill (Blue)
        fp_text = f"{fp:2d} False+"
        cv2.rectangle(canvas, (sb_x1 + 465, cur_y - 14), (sb_x1 + 555, cur_y + 6), (180, 90, 20), -1)
        cv2.putText(canvas, fp_text, (sb_x1 + 471, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # Recall
        rec_text = f"Recall: {rec:5.1f}%"
        cv2.putText(canvas, rec_text, (sb_x1 + 575, cur_y), cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 215, 255), 1, cv2.LINE_AA)

        cur_y += row_gap

    return canvas


def draw_evaluated_system_card(
    img: np.ndarray,
    tp_matches: List[Dict[str, Any]],
    fn_misses: List[Dict[str, Any]],
    fp_errors: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Renders boxes on system blurred frame:
      - BLUE  : False Positives (Over-blur / Spurious)
      - RED   : Missed Ground Truth (False Negatives / Privacy Leaks)
      - GREEN : Detected Ground Truth (True Positives)
    """
    canvas = img.copy()
    h, w = canvas.shape[:2]

    drawn_labels = []

    def place_label(x1, y1, x2, y2, text, color_bg, font_scale=0.34):
        (lw, lh), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lbl_x1 = max(0, x1)
        lbl_x2 = min(w, lbl_x1 + lw + 4)
        lbl_y1 = max(0, y1 - lh - 4)
        lbl_y2 = y1

        collides = False
        for (px1, py1, px2, py2) in drawn_labels:
            if not (lbl_x2 < px1 or lbl_x1 > px2 or lbl_y2 < py1 or lbl_y1 > py2):
                collides = True
                break

        if collides or lbl_y1 <= 2:
            lbl_y1 = min(h - lh - 6, y2 + 2)
            lbl_y2 = lbl_y1 + lh + 4

        drawn_labels.append((lbl_x1, lbl_y1, lbl_x2, lbl_y2))
        cv2.rectangle(canvas, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color_bg, -1)
        cv2.putText(canvas, text, (lbl_x1 + 2, lbl_y2 - 3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    # 1. Draw False Positives in BLUE (BGR: 240, 100, 0)
    for fp in fp_errors:
        box = fp["box"]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (240, 100, 0), 2)
        place_label(x1, y1, x2, y2, f"FP #{fp['fp_id']}", (180, 70, 0))

    # 2. Draw Missed GT targets in RED (BGR: 0, 0, 240)
    for fn in fn_misses:
        box = fn["box"]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 240), 2)
        place_label(x1, y1, x2, y2, f"#{fn['gt_id']} MISSED", (0, 0, 180))

    # 3. Draw Detected GT targets in GREEN (BGR: 0, 220, 0)
    for tp in tp_matches:
        box = tp["model_box"]
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        place_label(x1, y1, x2, y2, f"#{tp['gt_id']} DETECTED", (0, 140, 30))

    # If completely missed with 0 detections
    if len(tp_matches) == 0 and len(fp_errors) == 0:
        cv2.rectangle(canvas, (w - 240, 8), (w - 8, 38), (0, 0, 180), -1)
        cv2.putText(canvas, "0 DETECTIONS (ALL MISSED)", (w - 232, 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def prepare_system_card(
    img: np.ndarray,
    system_name: str,
    tp: int,
    fn: int,
    fp: int,
    telem_info: Dict[str, Any],
    color_border: Tuple[int, int, int]
) -> np.ndarray:
    """Prepares an individual card with custom header bar and colored pill badges."""
    h, w = img.shape[:2]
    header_h = 46
    border_w = 3

    card = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    card[:header_h, :] = (26, 26, 26) # Dark sleek header background

    # 1. System Name (Left)
    cv2.putText(card, system_name, (12, 30), cv2.FONT_HERSHEY_DUPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. Pill 1: Detected (Green)
    x = 215
    s1 = f"{tp} Detected"
    (tw1, _), _ = cv2.getTextSize(s1, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
    cv2.rectangle(card, (x, 10), (x + tw1 + 14, 36), (0, 145, 40), -1)
    cv2.putText(card, s1, (x + 7, 27), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Pill 2: Missed (Red)
    x += tw1 + 22
    s2 = f"{fn} Missed"
    (tw2, _), _ = cv2.getTextSize(s2, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
    cv2.rectangle(card, (x, 10), (x + tw2 + 14, 36), (20, 20, 190), -1)
    cv2.putText(card, s2, (x + 7, 27), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # Pill 3: False+ (Blue)
    x += tw2 + 22
    s3 = f"{fp} False+"
    (tw3, _), _ = cv2.getTextSize(s3, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
    cv2.rectangle(card, (x, 10), (x + tw3 + 14, 36), (190, 95, 20), -1)
    cv2.putText(card, s3, (x + 7, 27), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    # 3. Telemetry (Right)
    ram_mb = telem_info.get("peak_ram_mb", 0.0)
    fps = telem_info.get("fps", telem_info.get("throughput_fps", 0.0))
    if ram_mb > 0:
        telem_str = f"RAM: {ram_mb:.0f}MB | {fps:.2f}FPS"
        (tw_t, _), _ = cv2.getTextSize(telem_str, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
        cv2.putText(card, telem_str, (w - tw_t - 14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

    card[header_h:, :] = img
    cv2.rectangle(card, (0, 0), (w - 1, h + header_h - 1), color_border, border_w)
    return card


def prepare_gt_header_card(
    img: np.ndarray,
    gt_count: int,
    gt_source: str,
    color_border: Tuple[int, int, int] = (60, 140, 210)
) -> np.ndarray:
    """Prepares Card 1 with Gold accent header."""
    h, w = img.shape[:2]
    header_h = 46
    border_w = 3

    card = np.zeros((h + header_h, w, 3), dtype=np.uint8)
    card[:header_h, :] = (35, 55, 95) # Dark gold-blue accent header
    hdr_text = f"1. RAW INPUT / GT REFERENCE ({gt_count} Targets: {gt_source})"
    cv2.putText(card, hdr_text, (14, 30), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

    card[header_h:, :] = img
    cv2.rectangle(card, (0, 0), (w - 1, h + header_h - 1), color_border, border_w)
    return card


def load_ground_truth_for_frame(frame_name: str, gt_dir: Path) -> Tuple[List[Dict[str, Any]], str]:
    """Loads reference GT directly from V.2 Ground Truth."""
    stem = Path(frame_name).stem
    gt_path = gt_dir / f"{stem}.json"

    if not gt_path.exists():
        # Fallback to desktop path if needed
        alt_path = Path("/Users/mazeny/Desktop/V.2") / f"{stem}.json"
        if alt_path.exists():
            gt_path = alt_path

    if not gt_path.exists():
        return [], "None"

    src = "V.2 (Ground Truth)"

    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_dets = []
    idx_counter = 1
    for sh in data.get("shapes", []):
        lbl = str(sh.get("label", "face")).lower().strip()
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1 = float(min(xs))
        y1 = float(min(ys))
        x2 = float(max(xs))
        y2 = float(max(ys))

        if x2 <= x1 or y2 <= y1:
            continue

        norm_cls = "plate" if ("plate" in lbl or "license" in lbl) else "face"
        gt_dets.append({
            "id": idx_counter,
            "class": norm_cls,
            "box": [x1, y1, x2, y2],
            "conf": 1.0
        })
        idx_counter += 1

    return gt_dets, src


def generate_comparisons(
    results_dir: Path,
    gt_dir: Path,
    output_dir: Path,
    num_samples: int = 10
):
    output_dir.mkdir(parents=True, exist_ok=True)
    blurred_dir = results_dir / "blurred" if (results_dir / "blurred").exists() else results_dir

    # 1. Load telemetry for RAM, FPS, and Latency
    telem_json = blurred_dir / "telemetry" / "telemetry_summary_comparison.json"
    telemetry_data = {}
    if telem_json.exists():
        with open(telem_json, "r", encoding="utf-8") as f:
            t_raw = json.load(f)
        if isinstance(t_raw, list):
            telemetry_data = {item["system_id"]: item for item in t_raw if "system_id" in item}

    # 2. Load model detections
    sys_dets = {}
    for folder_key, _, _, _ in SYSTEM_KEYS:
        det_path = blurred_dir / folder_key / "detections.json"
        if not det_path.exists() and "FusionPortable" in folder_key:
            det_path = blurred_dir / "4_UnderstandAI" / "detections.json"
        if det_path.exists():
            with open(det_path, "r", encoding="utf-8") as f:
                sys_dets[folder_key] = json.load(f)
        else:
            sys_dets[folder_key] = {}

    # 3. Load selected frames
    selected_json = blurred_dir / "telemetry" / "selected_frames.json"
    if selected_json.exists():
        with open(selected_json, "r", encoding="utf-8") as f:
            selected_names = json.load(f)
    else:
        fast_dir = blurred_dir / "1_FAST_V5"
        selected_names = sorted([p.name for p in fast_dir.glob("*.jpg")])

    total_frames = len(selected_names)
    if total_frames <= num_samples:
        sample_indices = list(range(total_frames))
    else:
        sample_indices = np.linspace(0, total_frames - 1, num_samples, dtype=int).tolist()

    sampled_frames = [selected_names[i] for i in sample_indices]

    print("=" * 80)
    print("  MULTI-SYSTEM 10-FRAME BENEFICIAL COMPARISON COLLAGE GENERATOR")
    print("=" * 80)
    print(f"[*] Ground Truth V.2 Dir   : {gt_dir}")
    print(f"[*] Comparison Output Dir  : {output_dir}")
    print(f"[*] Rendering {len(sampled_frames)} frames...")

    for comp_idx, (frame_idx, frame_name) in enumerate(zip(sample_indices, sampled_frames), 1):
        # Locate clean base image
        clean_img_path = gt_dir / frame_name
        if not clean_img_path.exists():
            clean_img_path = Path("/Users/mazeny/Desktop/V.2") / frame_name
        if not clean_img_path.exists():
            clean_img_path = blurred_dir / "2_EgoBlur" / frame_name
        if not clean_img_path.exists():
            clean_img_path = blurred_dir / "1_FAST_V5" / frame_name

        raw_img = cv2.imread(str(clean_img_path))
        if raw_img is None:
            print(f"[-] Warning: Could not read base image {clean_img_path}. Skipping.")
            continue
        h, w = raw_img.shape[:2]

        # Load GT for Card 1
        gt_dets, gt_source = load_ground_truth_for_frame(frame_name, gt_dir)

        # Evaluate each pipeline against GT
        evaluated_systems = {}
        systems_scores = {}

        for folder_key, display_name, col_bg, col_border in SYSTEM_KEYS:
            # Read blurred image
            img_path = blurred_dir / folder_key / frame_name
            if not img_path.exists() and "FusionPortable" in folder_key:
                img_path = blurred_dir / "4_UnderstandAI" / frame_name

            pipe_img = cv2.imread(str(img_path)) if img_path.exists() else raw_img.copy()

            # Retrieve detections
            m_dets = sys_dets.get(folder_key, {}).get(frame_name, [])

            # Run greedy bipartite matching
            tp_matches, fn_misses, fp_errors = match_system_detections(gt_dets, m_dets, iou_thresh=0.40, iomin_thresh=0.50)

            evaluated_systems[folder_key] = {
                "pipe_img": pipe_img,
                "display_name": display_name,
                "col_bg": col_bg,
                "col_border": col_border,
                "tp_matches": tp_matches,
                "fn_misses": fn_misses,
                "fp_errors": fp_errors,
                "tp": len(tp_matches),
                "fn": len(fn_misses),
                "fp": len(fp_errors)
            }
            systems_scores[folder_key] = {
                "tp": len(tp_matches),
                "fn": len(fn_misses),
                "fp": len(fp_errors)
            }

        # Render Card 1 (Raw Input / GT Reference) with HUD Scoreboard
        gt_boxed = draw_gt_reference_card(raw_img, gt_dets, systems_scores, gt_source)
        c_gt = prepare_gt_header_card(gt_boxed, len(gt_dets), gt_source, color_border=(60, 140, 210))

        # Render Cards 2-6 (The 5 Pipelines)
        cards = {}
        for folder_key, _, _, col_border in SYSTEM_KEYS:
            es = evaluated_systems[folder_key]
            # Draw evaluated boxes (Green: TP, Red: FN, Blue: FP)
            boxed_img = draw_evaluated_system_card(es["pipe_img"], es["tp_matches"], es["fn_misses"], es["fp_errors"])
            # Prepare card with header and pill badges
            card_obj = prepare_system_card(
                boxed_img,
                es["display_name"],
                es["tp"],
                es["fn"],
                es["fp"],
                telemetry_data.get(folder_key, {}),
                col_border
            )
            cards[folder_key] = card_obj

        # 2x3 Grid
        row1 = np.hstack([c_gt, cards["1_FAST_V5"], cards["2_EgoBlur"]])
        row2 = np.hstack([cards["3_DashcamCleaner"], cards["4_FusionPortable"], cards["5_Generative_Anonymizer"]])
        grid_img = np.vstack([row1, row2])

        banner_h = 56
        footer_h = 58
        final_canvas = np.zeros((grid_img.shape[0] + banner_h + footer_h, grid_img.shape[1], 3), dtype=np.uint8)

        # Top Banner
        final_canvas[:banner_h, :] = (20, 20, 20)
        title_text = f"BENEFICIAL EVALUATION COMPARISON #{comp_idx:02d} — Frame #{frame_idx+1:05d}: {frame_name} (Reference GT: {len(gt_dets)} Targets [{gt_source}])"
        cv2.putText(final_canvas, title_text, (25, 38), cv2.FONT_HERSHEY_DUPLEX, 0.82, (0, 220, 255), 2, cv2.LINE_AA)

        # Grid Images
        final_canvas[banner_h:banner_h + grid_img.shape[0], :] = grid_img

        # Bottom Color-Coded Legend Banner
        footer_y = banner_h + grid_img.shape[0]
        final_canvas[footer_y:, :] = (22, 22, 22)

        # Legend Line 1
        leg_line1_parts = [
            ("GROUND TRUTH REFERENCE: ", (0, 215, 255)),
            ("Gold Boxes (#1..#N) on Card 1   |   ", (220, 220, 220)),
            ("[GREEN BOX] DETECTED (True Positive)   |   ", (0, 230, 40)),
            ("[RED BOX] MISSED (False Negative / Leak)   |   ", (40, 40, 240)),
            ("[BLUE BOX] FALSE POSITIVE (Over-blur)", (240, 120, 20))
        ]
        cur_lx = 25
        for txt, col in leg_line1_parts:
            cv2.putText(final_canvas, txt, (cur_lx, footer_y + 24), cv2.FONT_HERSHEY_DUPLEX, 0.52, col, 1, cv2.LINE_AA)
            (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.52, 1)
            cur_lx += tw

        # Legend Line 2
        leg_line2 = "Box numbers (#1..#N) correspond directly across panels to identify specific pedestrian faces and license plates under high-speed automotive constraints."
        cv2.putText(final_canvas, leg_line2, (25, footer_y + 48), cv2.FONT_HERSHEY_DUPLEX, 0.44, (160, 160, 160), 1, cv2.LINE_AA)

        comp_file = output_dir / f"comparison_frame_{comp_idx:02d}_idx{frame_idx+1:05d}.jpg"
        cv2.imwrite(str(comp_file), final_canvas, [cv2.IMWRITE_JPEG_QUALITY, 94])
        print(f"  [+] Saved Beneficial Comparison #{comp_idx:02d} -> {comp_file.name} ({len(gt_dets)} GT targets from {gt_source})")

    print("-" * 80)
    print(f"[✓] Successfully generated all {len(sampled_frames)} beneficial comparison frames in:")
    print(f"    {output_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Generate 10 multi-system beneficial comparison frames using V.2 Ground Truth")
    parser.add_argument("--results-dir", type=str, default="3_Systems/outputs/results_pendik_100", help="Directory where model outputs reside")
    parser.add_argument("--gt-dir", type=str, default="3_Systems/outputs/V.2", help="Directory containing V.2 Ground Truth JSONs")
    parser.add_argument("--output-dir", type=str, default="3_Systems/outputs/results_pendik_100/blurred/10_Frame_Comparisons", help="Directory to save comparison frames")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of comparison frames (default: 10)")
    args = parser.parse_args()

    generate_comparisons(
        Path(args.results_dir).resolve(),
        Path(args.gt_dir).resolve(),
        Path(args.output_dir).resolve(),
        args.num_samples
    )


if __name__ == "__main__":
    main()
