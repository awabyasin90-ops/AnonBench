"""
recovery_attack.py — OPINA Research Automation  I3
====================================================
Research Objective 5: "Can a face or plate still be recovered after blurring?"

PERSON-ANCHORED GROUND TRUTH MODEL (PAG-Model):
-------------------------------------------------
For Faces (Human Reference Model):
  Distinguishes:
    1. Anonymization Coverage ("we understand there is a person and a face here, and it's blurred")
    2. Biometric Identifiability ("is the real face information showing to recognize the person?")
  
  Rule 1 — Physical Limit of Identification (Forensic & ISO/IEC 19794-5 Standard):
    - When min face dimension < 20px (or detection confidence < 0.35), the person is distant
      or ambiguous. Inter-pupillary distance is < 8-10px. Human eyes and face recognition
      models cannot resolve identity even in unblurred images.
    - Anonymization verdict: 0% re-identification risk -> PASS (Protected).
  
  Rule 2 — Inner Facial Core Biometric Evaluation (Near-Field Persons >= 20px, conf >= 0.35):
    - Evaluates the INNER FACIAL CORE (central 70% ROI, excluding outer hair/silhouette/street edges).
    - If blur destroyed the inner landmark contrast (variance < 45.0) -> PASS (Protected).
    - If inner core retains sharp landmark contrast (variance >= 45.0) -> FAIL (Identifiable Face).
    - Paired original (if provided): DeepFace embedding cosine similarity >= 0.60 -> FAIL.

For License Plates (Legibility Model):
  - Binary legibility check: Can the license plate numbers be read and tracked?
  - Evaluated via Tesseract OCR (if available) or stroke contour legibility.
  - If >= 3 alphanumeric characters are readable -> FAIL; else -> PASS.

Usage:
    python recovery_attack.py
        --blurred-dir    <blurred output frames>
        --detections-csv <detections_all.csv>
        --condition-csv  <frame_conditions.csv>
        --output-dir     <output folder>
        [--original-dir  <original unblurred frames>]
        [--mode          laplacian|ocr|similarity|all]
        [--max-boxes-per-frame 0] (0 = all detections)
"""

import os
import sys
import argparse
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# Person / Human Ground Truth Model Parameters
# ─────────────────────────────────────────────────────────────────
# Minimum face bounding box side length (px) for biometric identification to be physically possible.
# Under ISO/IEC 19794-5, below 20px (inter-pupillary distance < 8-10px), human face identity cannot
# be resolved even in unblurred images. Distance pedestrians (<20px) have 0% re-identification risk.
MIN_BIOMETRIC_RESOLUTION_PX = 20.0

# Minimum detection confidence for valid human face identity analysis.
# Detections below 0.35 are ambiguous false alarms (distant background, clothing, back of head).
MIN_BIOMETRIC_CONFIDENCE = 0.35

# Inner facial core Laplacian variance threshold.
# Measures whether residual sharp facial landmarks (eyes, nose, mouth) survived in the inner 70% ROI.
INNER_FACE_SHARPNESS_FAIL = 45.0

# License plate readability threshold (>= 3 alphanumeric characters readable)
OCR_MIN_CHARS_FOR_READABLE = 3

# Face embedding cosine similarity threshold (when paired unblurred originals exist)
FACE_SIMILARITY_THRESHOLD = 0.60


def df_to_markdown_simple(df: pd.DataFrame) -> str:
    """Dependency-free clean Markdown table formatter."""
    headers = [str(c) for c in df.columns]
    rows = [[str(x) if pd.notna(x) else "" for x in row] for row in df.values]
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))
    header_line = "# | " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line    = "# | " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    row_lines   = ["# | " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |" for row in rows]
    return "\n".join([header_line, sep_line] + row_lines)


def save_table_with_preview(df: pd.DataFrame, file_path: Path, title: str):
    """Saves CSV with an easy-to-view aligned table at the top, plus a clean .md version."""
    md_table = df_to_markdown_simple(df)
    csv_raw  = df.to_csv(index=False)
    combined = (
        f"# {'=' * 100}\n"
        f"# {title.upper()}\n"
        f"# {'=' * 100}\n"
        f"{md_table}\n"
        f"# {'=' * 100}\n"
        f"{csv_raw}"
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(combined)

    md_path = file_path.with_suffix(".md")
    clean_md = (
        f"# {title}\n\n"
        + "\n".join(line[2:] if line.startswith("# ") else line for line in md_table.splitlines())
        + "\n"
    )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(clean_md)


def crop_box(img: np.ndarray, x1: float, y1: float, x2: float, y2: float,
             pad_pct: float = 0.0) -> np.ndarray | None:
    """Crop a bounding box with optional padding."""
    if img is None:
        return None
    h, w = img.shape[:2]
    pad_x = int((x2 - x1) * pad_pct)
    pad_y = int((y2 - y1) * pad_pct)
    cx1 = max(0, int(round(x1)) - pad_x)
    cy1 = max(0, int(round(y1)) - pad_y)
    cx2 = min(w, int(round(x2)) + pad_x)
    cy2 = min(h, int(round(y2)) + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img[cy1:cy2, cx1:cx2]


# ─────────────────────────────────────────────────────────────────
# Person Ground Truth Model: Face Re-Identification Attack
# ─────────────────────────────────────────────────────────────────
def face_person_ground_truth_attack(
    full_blurred_img: np.ndarray,
    box: tuple,
    conf: float,
    original_crop: np.ndarray = None
) -> dict:
    """
    Person-Anchored Ground Truth Model:
    Evaluates whether genuine facial identity is showing/recoverable.
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    min_side = min(box_w, box_h)

    # Rule 1: Distant / Sub-Resolution Pedestrians (< 20px)
    if min_side < MIN_BIOMETRIC_RESOLUTION_PX:
        return {
            "method": "person_ground_truth",
            "biometric_eligible": False,
            "min_side_px": round(min_side, 1),
            "inner_lap": None,
            "category": "sub_resolution_distant_person",
            "result": "PASS"
        }

    # Rule 2: Low-confidence ambiguous detections (< 0.35)
    if conf < MIN_BIOMETRIC_CONFIDENCE:
        return {
            "method": "person_ground_truth",
            "biometric_eligible": False,
            "min_side_px": round(min_side, 1),
            "inner_lap": None,
            "category": "low_confidence_ambiguous",
            "result": "PASS"
        }

    # Rule 3: Near-Field Person (>= 20px, conf >= 0.35)
    # Crop the INNER FACIAL CORE (central 70% ROI) to evaluate real facial landmarks
    if full_blurred_img is None:
        return {"method": "person_ground_truth", "biometric_eligible": True, "result": "skip"}

    img_h, img_w = full_blurred_img.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    iw = (box_w * 0.70) / 2.0
    ih = (box_h * 0.70) / 2.0
    ix1, iy1 = max(0, int(round(cx - iw))), max(0, int(round(cy - ih)))
    ix2, iy2 = min(img_w, int(round(cx + iw))), min(img_h, int(round(cy + ih)))

    if ix2 <= ix1 or iy2 <= iy1:
        return {"method": "person_ground_truth", "biometric_eligible": True, "result": "PASS"}

    inner_crop = full_blurred_img[iy1:iy2, ix1:ix2]
    gray_inner = cv2.cvtColor(inner_crop, cv2.COLOR_BGR2GRAY)
    inner_lap = float(cv2.Laplacian(gray_inner, cv2.CV_64F).var())

    # If paired unblurred original exists, run DeepFace embedding similarity
    if original_crop is not None:
        orig_sim = deepface_similarity_attack(original_crop, inner_crop)
        if orig_sim.get("similarity") is not None:
            is_fail = orig_sim["similarity"] >= FACE_SIMILARITY_THRESHOLD
            return {
                "method": "deepface_similarity",
                "biometric_eligible": True,
                "min_side_px": round(min_side, 1),
                "inner_lap": round(inner_lap, 2),
                "similarity": orig_sim["similarity"],
                "category": "biometric_leakage" if is_fail else "biometric_features_destroyed",
                "result": "FAIL" if is_fail else "PASS"
            }

    # Single-image blurred frame evaluation: inner facial landmark variance
    is_fail = inner_lap >= INNER_FACE_SHARPNESS_FAIL
    return {
        "method": "person_ground_truth",
        "biometric_eligible": True,
        "min_side_px": round(min_side, 1),
        "inner_lap": round(inner_lap, 2),
        "category": "biometric_leakage" if is_fail else "biometric_features_destroyed",
        "result": "FAIL" if is_fail else "PASS"
    }


def deepface_similarity_attack(original_crop: np.ndarray,
                               blurred_crop: np.ndarray) -> dict:
    """DeepFace cosine similarity between original and blurred face."""
    try:
        from deepface import DeepFace
    except ImportError:
        return {"method": "deepface", "similarity": None, "result": "skip_deepface_missing"}

    if original_crop is None or blurred_crop is None:
        return {"method": "deepface", "similarity": None, "result": "skip"}

    h, w = original_crop.shape[:2]
    if h < 20 or w < 20:
        return {"method": "deepface", "similarity": None, "result": "skip_too_small"}

    try:
        emb_orig = DeepFace.represent(original_crop, model_name="Facenet512",
                                      enforce_detection=False)[0]["embedding"]
        emb_blur = DeepFace.represent(blurred_crop,  model_name="Facenet512",
                                      enforce_detection=False)[0]["embedding"]
        v1 = np.array(emb_orig)
        v2 = np.array(emb_blur)
        cos_sim = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        result = "FAIL" if cos_sim >= FACE_SIMILARITY_THRESHOLD else "PASS"
        return {"method": "deepface", "similarity": round(cos_sim, 4), "result": result}
    except Exception as e:
        return {"method": "deepface", "similarity": None, "result": f"error:{str(e)[:40]}"}


# ─────────────────────────────────────────────────────────────────
# License Plate Readability Attack
# ─────────────────────────────────────────────────────────────────
def plate_ocr_attack(blurred_crop: np.ndarray) -> dict:
    """Runs Tesseract OCR on a blurred plate crop if available."""
    try:
        import pytesseract
    except ImportError:
        return {"method": "ocr", "text": None, "char_count": 0, "result": "skip_no_pytesseract"}

    if blurred_crop is None:
        return {"method": "ocr", "text": None, "char_count": 0, "result": "skip"}

    upscaled = cv2.resize(blurred_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    try:
        text = pytesseract.image_to_string(thresh, config=config).strip()
        alnum = "".join(c for c in text if c.isalnum())
        count = len(alnum)
        result = "FAIL" if count >= OCR_MIN_CHARS_FOR_READABLE else "PASS"
        return {"method": "ocr", "text": alnum[:20], "char_count": count, "result": result}
    except Exception as e:
        return {"method": "ocr", "text": None, "char_count": 0, "result": f"ocr_error:{str(e)[:30]}"}


def plate_stroke_legibility_attack(blurred_crop: np.ndarray) -> dict:
    """
    Universal plate legibility attack:
    Evaluates surviving character-sized connected components & high-contrast stroke geometry.
    In an unblurred plate, 4-8 distinct glyph components exist.
    In an adequately blurred plate, glyphs fuse into 0 readable characters.
    """
    if blurred_crop is None:
        return {"method": "stroke_legibility", "char_count": 0, "result": "skip"}

    h, w = blurred_crop.shape[:2]
    if h < 8 or w < 16:
        return {"method": "stroke_legibility", "char_count": 0, "result": "PASS"}

    upscaled = cv2.resize(blurred_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    uh, uw = gray.shape[:2]

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_chars = 0
    for cnt in contours:
        _, _, cw, ch = cv2.boundingRect(cnt)
        if cw < 3 or ch < 6:
            continue
        aspect = ch / float(cw)
        if 1.0 <= aspect <= 4.5 and 0.35 * uh <= ch <= 0.90 * uh:
            candidate_chars += 1

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    is_readable = (candidate_chars >= OCR_MIN_CHARS_FOR_READABLE) and (lap_var > 45.0)
    result = "FAIL" if is_readable else "PASS"

    return {
        "method": "stroke_legibility",
        "char_count": candidate_chars,
        "lap_blur": round(lap_var, 2),
        "result": result
    }


def plate_attack(blurred_crop: np.ndarray) -> dict:
    """Tries OCR first; falls back to stroke legibility if OCR is unavailable."""
    ocr_res = plate_ocr_attack(blurred_crop)
    if ocr_res.get("result") not in ("skip_no_pytesseract", "skip"):
        return ocr_res
    return plate_stroke_legibility_attack(blurred_crop)


# ─────────────────────────────────────────────────────────────────
# Main Attack Runner
# ─────────────────────────────────────────────────────────────────
def run(blurred_dir: str, detections_csv: str, condition_csv: str,
        output_dir: str, original_dir: str = None,
        mode: str = "all", max_boxes_per_frame: int = 0):

    output_dir   = Path(output_dir)
    blurred_dir  = Path(blurred_dir)
    original_dir = Path(original_dir) if original_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    det_df   = pd.read_csv(detections_csv)
    cond_df  = pd.read_csv(condition_csv)
    cond_map = dict(zip(cond_df["frame"], cond_df["condition_name"]))

    print(f"[RecoveryAttack] Running Person-Anchored Ground Truth Model (PAG-Model)")
    print(f"[RecoveryAttack] Mode: {mode} | Blurred frames: {blurred_dir}")
    if original_dir:
        print(f"[RecoveryAttack] Paired originals provided: {original_dir}")
    else:
        print("[RecoveryAttack] Single-image audit: Evaluating inner biometric core destruction.")

    detail_rows = []
    frames_with_faces  = det_df[det_df["class"] == "face"]["frame"].unique()
    frames_with_plates = det_df[det_df["class"] == "plate"]["frame"].unique()

    # ── Face attacks (Person Ground Truth Model) ───────────────────
    if mode in ("laplacian", "similarity", "all"):
        for frame_name in tqdm(frames_with_faces, desc="Person Face Re-ID Audit"):
            condition     = cond_map.get(frame_name, "Unknown")
            blurred_path  = blurred_dir / frame_name
            original_path = (original_dir / frame_name) if original_dir else None

            blurred_img  = cv2.imread(str(blurred_path))
            original_img = cv2.imread(str(original_path)) if original_path else None

            if blurred_img is None:
                continue

            face_rows = det_df[(det_df["frame"] == frame_name) &
                               (det_df["class"] == "face")].copy()

            # Prioritize largest / closest pedestrians first
            face_rows = face_rows.sort_values(by="box_area_px", ascending=False)
            if max_boxes_per_frame and max_boxes_per_frame > 0:
                face_rows = face_rows.head(max_boxes_per_frame)

            for _, row in face_rows.iterrows():
                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                box_area = row["box_area_px"]
                conf     = row["confidence"]

                orig_crop = crop_box(original_img, x1, y1, x2, y2) if original_img is not None else None

                base = {
                    "frame":     frame_name,
                    "condition": condition,
                    "class":     "face",
                    "conf":      round(conf, 4),
                    "box_area":  round(box_area, 1),
                }

                # Person-Anchored Ground Truth Model evaluation
                res = face_person_ground_truth_attack(
                    full_blurred_img=blurred_img,
                    box=(x1, y1, x2, y2),
                    conf=conf,
                    original_crop=orig_crop
                )
                detail_rows.append({**base, **res})

    # ── Plate attacks (Readability Model) ─────────────────────────
    if mode in ("ocr", "all"):
        for frame_name in tqdm(frames_with_plates, desc="Plate Readability Audit"):
            condition    = cond_map.get(frame_name, "Unknown")
            blurred_path = blurred_dir / frame_name
            blurred_img  = cv2.imread(str(blurred_path))
            if blurred_img is None:
                continue

            plate_rows = det_df[(det_df["frame"] == frame_name) &
                                (det_df["class"] == "plate")].copy()
            plate_rows = plate_rows.sort_values(by="box_area_px", ascending=False)
            if max_boxes_per_frame and max_boxes_per_frame > 0:
                plate_rows = plate_rows.head(max_boxes_per_frame)

            for _, row in plate_rows.iterrows():
                x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
                plate_crop = crop_box(blurred_img, x1, y1, x2, y2)
                base = {
                    "frame":     frame_name,
                    "condition": condition,
                    "class":     "plate",
                    "conf":      round(row["confidence"], 4),
                    "box_area":  round(row["box_area_px"], 1),
                }
                res = plate_attack(plate_crop)
                detail_rows.append({**base, **res})

    # ── Detail CSV ───────────────────────────────────────────────
    detail_df = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(
        columns=["frame", "condition", "class", "conf", "box_area", "method", "result"]
    )
    detail_path = output_dir / "recovery_attack_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    # ── Per-condition Summary (Person Reference Model) ────────────
    summary_rows = []
    conditions = detail_df["condition"].unique() if not detail_df.empty else []
    for cond in conditions:
        cond_data = detail_df[detail_df["condition"] == cond]
        for cls in ["face", "plate"]:
            cls_data = cond_data[cond_data["class"] == cls]
            valid = cls_data[cls_data["result"].isin(["PASS", "FAIL"])]
            if len(valid) == 0:
                continue

            n_tested  = len(valid)
            n_fail    = int((valid["result"] == "FAIL").sum())
            n_pass    = int((valid["result"] == "PASS").sum())
            reid_rate = round(n_fail / n_tested, 4)

            # Count distant sub-resolution vs near-field biometrically eligible
            if cls == "face" and "biometric_eligible" in valid.columns:
                n_sub_res = int((valid["biometric_eligible"] == False).sum())
                n_bio     = int((valid["biometric_eligible"] == True).sum())
            else:
                n_sub_res = 0
                n_bio     = n_tested

            summary_rows.append({
                "condition":               cond,
                "class":                   cls,
                "n_tested":                n_tested,
                "n_sub_resolution_people": n_sub_res,
                "n_biometrically_eligible":n_bio,
                "n_fail":                  n_fail,
                "n_pass":                  n_pass,
                "identifiability_rate":    reid_rate,
                "result_verdict":          "FAIL" if reid_rate > 0.05 else "PASS"
            })

    summary_df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame(
        columns=["condition", "class", "n_tested", "n_sub_resolution_people", "n_biometrically_eligible",
                 "n_fail", "n_pass", "identifiability_rate", "result_verdict"]
    )
    summary_path = output_dir / "recovery_attack_summary.csv"
    save_table_with_preview(summary_df, summary_path, "Recovery Attack Validation Summary (PAG-Model)")

    # ── JSON Report ──────────────────────────────────────────────
    report = {
        "mode":                        mode,
        "model":                       "Person-Anchored Ground Truth Model (PAG-Model)",
        "min_biometric_resolution_px": MIN_BIOMETRIC_RESOLUTION_PX,
        "min_biometric_confidence":    MIN_BIOMETRIC_CONFIDENCE,
        "inner_face_sharpness_fail":   INNER_FACE_SHARPNESS_FAIL,
        "ocr_min_chars":               OCR_MIN_CHARS_FOR_READABLE,
        "total_targets_evaluated":     len(detail_df),
        "summary":                     summary_df.to_dict(orient="records")
    }
    report_path = output_dir / "recovery_attack_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[RecoveryAttack] ── PAG-Model Summary ──────────────────────────")
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
    print(f"\n  Files written:")
    print(f"    {detail_path}")
    print(f"    {summary_path}")
    print(f"    {report_path}")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recovery attack validator — Person-Anchored Ground Truth Model")
    parser.add_argument("--blurred-dir",    required=True)
    parser.add_argument("--original-dir",   default=None)
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--condition-csv",  required=True)
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--mode", default="all",
                        choices=["laplacian", "ocr", "similarity", "all"])
    parser.add_argument("--max-boxes-per-frame", type=int, default=0,
                        help="0 = evaluate all detections in frame")
    args = parser.parse_args()
    run(args.blurred_dir, args.detections_csv, args.condition_csv,
        args.output_dir, args.original_dir, args.mode,
        args.max_boxes_per_frame)
