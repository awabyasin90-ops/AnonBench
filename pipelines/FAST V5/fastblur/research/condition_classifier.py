"""
condition_classifier.py — OPINA Research Automation
=====================================================
Automatically classifies each frame into one of 6 research conditions:
  1. Clear Day
  2. Day Rain
  3. Clear Night
  4. Night + Rain/Glare
  5. Heavy Fog (Sim Only)
  6. Extreme Motion Blur / Glare (CARLA Only)

Method: Pure image-feature heuristics (luminance, saturation, Laplacian
sharpness, glare fraction, fog contrast ratio). No GPS or external metadata
required. Works on both real and CARLA frames.

For CARLA frames: source metadata tag overrides condition names where needed.

Usage:
    python condition_classifier.py
        --image-dir  <path_to_raw_or_blurred_frames>
        --output-csv <output_labels.csv>
        --source     [real_car | carla | tesla]   (default: real_car)

Output CSV columns:
    frame, source, condition_id, condition_name,
    lum_mean, sat_mean, lap_var, glare_pct, fog_contrast,
    top_lum, bot_lum, confidence
"""

import os
import argparse
import csv
import cv2
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# Condition IDs (match Research Tracker column order)
# ─────────────────────────────────────────────────────────────────
COND_CLEAR_DAY        = (1, "Clear Day")
COND_DAY_RAIN         = (2, "Day Rain")
COND_CLEAR_NIGHT      = (3, "Clear Night")
COND_NIGHT_RAIN_GLARE = (4, "Night + Rain/Glare")
COND_HEAVY_FOG        = (5, "Heavy Fog (Sim Only)")
COND_MOTION_BLUR      = (6, "Extreme Motion Blur / Glare (CARLA Only)")

ALL_CONDITIONS = [
    COND_CLEAR_DAY, COND_DAY_RAIN, COND_CLEAR_NIGHT,
    COND_NIGHT_RAIN_GLARE, COND_HEAVY_FOG, COND_MOTION_BLUR
]


# ─────────────────────────────────────────────────────────────────
# Thresholds (tunable — calibrated on Pendik front_camera data)
# ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    # Night: overall mean luminance below this
    "night_lum_max":          90.0,
    # Daytime: luminance above this
    "day_lum_min":            100.0,
    # Rain proxy: reduced saturation + texture softness
    "rain_sat_max":           45.0,        # rain washes out colour
    "rain_lap_max":           200.0,       # rain softens edges
    # Rain alternative: strong horizontal streaks — Sobel Y/X ratio
    "rain_streak_ratio_min":  1.25,
    # Glare: fraction of near-saturated pixels
    "glare_pct_min":          25.0,        # >25% of pixels > 230 lum
    # Fog: very low contrast (std_lum) AND low saturation
    "fog_contrast_max":       35.0,        # std of luminance
    "fog_sat_max":            30.0,
    # Motion blur: very low Laplacian variance
    "motion_blur_lap_max":    80.0,
}


def extract_features(img_path: str) -> dict:
    """Read one image and return a feature dict. Returns None on read error."""
    img = cv2.imread(img_path)
    if img is None:
        return None

    gray_u8 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # uint8
    hsv     = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w    = gray_u8.shape

    gray = gray_u8.astype(np.float32)   # float32 for stats
    lum_mean  = float(gray.mean())
    lum_std   = float(gray.std())
    sat_mean  = float(hsv[:, :, 1].mean())
    val_mean  = float(hsv[:, :, 2].mean())

    # Laplacian sharpness — must use uint8 input on this OpenCV build
    lap_var = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())

    # Glare: fraction of pixels with luminance > 230
    glare_pct = float((gray > 230).sum() / gray.size * 100.0)

    # Sky vs road luminance split (top 40% = sky, bottom 40% = road)
    top_lum = float(gray[:int(0.4 * h), :].mean())
    bot_lum = float(gray[int(0.6 * h):, :].mean())

    # Horizontal rain streaks: ratio of vertical to horizontal Sobel energy
    sobel_x = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
    energy_x = float(np.abs(sobel_x).mean()) + 1e-6
    energy_y = float(np.abs(sobel_y).mean()) + 1e-6
    rain_streak_ratio = energy_y / energy_x   # rain → more horizontal streaks → higher ratio

    # Fog contrast proxy: low luminance std + low saturation
    fog_contrast = lum_std

    return {
        "lum_mean":         lum_mean,
        "lum_std":          lum_std,
        "sat_mean":         sat_mean,
        "val_mean":         val_mean,
        "lap_var":          lap_var,
        "glare_pct":        glare_pct,
        "top_lum":          top_lum,
        "bot_lum":          bot_lum,
        "rain_streak_ratio":rain_streak_ratio,
        "fog_contrast":     fog_contrast,
    }


def classify_condition(feats: dict) -> tuple[int, str, float]:
    """
    Rule-based condition classifier.
    Returns (condition_id, condition_name, confidence_0_to_1).

    Decision tree (priority order):
      1. Heavy Fog   — very low contrast + very low saturation
      2. Motion Blur — very low Laplacian sharpness
      3. Night       — low luminance
         3a. Night + Rain/Glare — night AND (glare OR rain proxy)
         3b. Clear Night
      4. Day
         4a. Day Rain — day AND (rain proxy: low sat + soft texture)
         4b. Clear Day
    """
    t = THRESHOLDS

    is_night       = feats["lum_mean"] < t["night_lum_max"]
    is_day         = feats["lum_mean"] > t["day_lum_min"]
    has_glare      = feats["glare_pct"]  > t["glare_pct_min"]
    has_fog        = (feats["fog_contrast"] < t["fog_contrast_max"]
                      and feats["sat_mean"] < t["fog_sat_max"])
    has_motion_blur = feats["lap_var"]  < t["motion_blur_lap_max"]
    has_rain        = (feats["sat_mean"] < t["rain_sat_max"]
                       or feats["lap_var"]  < t["rain_lap_max"]
                       or feats["rain_streak_ratio"] > t["rain_streak_ratio_min"])

    # ── Priority 1: Heavy Fog ──────────────────────────────────────
    if has_fog:
        conf = _fog_confidence(feats, t)
        return (*COND_HEAVY_FOG, conf)

    # ── Priority 2: Extreme Motion Blur ───────────────────────────
    if has_motion_blur and not is_night:
        conf = _motion_blur_confidence(feats, t)
        return (*COND_MOTION_BLUR, conf)

    # ── Priority 3: Night ──────────────────────────────────────────
    if is_night:
        if has_glare or has_rain:
            conf = _night_rain_confidence(feats, t)
            return (*COND_NIGHT_RAIN_GLARE, conf)
        else:
            conf = _night_confidence(feats, t)
            return (*COND_CLEAR_NIGHT, conf)

    # ── Priority 4: Day ───────────────────────────────────────────
    if has_rain and not has_glare:
        conf = _day_rain_confidence(feats, t)
        return (*COND_DAY_RAIN, conf)

    # Default: Clear Day (also covers glare-only day)
    conf = _clear_day_confidence(feats, t)
    return (*COND_CLEAR_DAY, conf)


# ─────────────────────────────────────────────────────────────────
# Confidence helpers (sigmoid-style scores, 0–1)
# ─────────────────────────────────────────────────────────────────
def _sigmoid(x, center, scale=1.0):
    return 1.0 / (1.0 + np.exp(-scale * (x - center)))

def _clear_day_confidence(f, t):
    lum_score  = _sigmoid(f["lum_mean"],  130, 0.03)
    sat_score  = _sigmoid(f["sat_mean"],   50, 0.05)
    lap_score  = _sigmoid(f["lap_var"],   200, 0.005)
    return float(np.clip((lum_score + sat_score + lap_score) / 3, 0, 1))

def _day_rain_confidence(f, t):
    rain_soft  = _sigmoid(t["rain_lap_max"] - f["lap_var"], 0, 0.01)
    rain_desat = _sigmoid(t["rain_sat_max"] - f["sat_mean"], 0, 0.05)
    return float(np.clip((rain_soft + rain_desat) / 2, 0, 1))

def _night_confidence(f, t):
    return float(np.clip(_sigmoid(t["night_lum_max"] - f["lum_mean"], 0, 0.05), 0, 1))

def _night_rain_confidence(f, t):
    night_score = _sigmoid(t["night_lum_max"] - f["lum_mean"], 0, 0.05)
    glare_score = _sigmoid(f["glare_pct"] - t["glare_pct_min"], 0, 0.1)
    return float(np.clip((night_score + glare_score) / 2, 0, 1))

def _fog_confidence(f, t):
    contrast_score = _sigmoid(t["fog_contrast_max"] - f["fog_contrast"], 0, 0.1)
    sat_score      = _sigmoid(t["fog_sat_max"]      - f["sat_mean"],     0, 0.05)
    return float(np.clip((contrast_score + sat_score) / 2, 0, 1))

def _motion_blur_confidence(f, t):
    return float(np.clip(_sigmoid(t["motion_blur_lap_max"] - f["lap_var"], 0, 0.05), 0, 1))


# ─────────────────────────────────────────────────────────────────
# Per-frame worker
# ─────────────────────────────────────────────────────────────────
def process_frame(img_path: str, source: str) -> dict | None:
    feats = extract_features(img_path)
    if feats is None:
        return None

    cond_id, cond_name, confidence = classify_condition(feats)

    return {
        "frame":          os.path.basename(img_path),
        "source":         source,
        "condition_id":   cond_id,
        "condition_name": cond_name,
        "confidence":     round(confidence, 4),
        "lum_mean":       round(feats["lum_mean"],  2),
        "sat_mean":       round(feats["sat_mean"],  2),
        "lap_var":        round(feats["lap_var"],   2),
        "glare_pct":      round(feats["glare_pct"], 4),
        "fog_contrast":   round(feats["fog_contrast"], 2),
        "top_lum":        round(feats["top_lum"],   2),
        "bot_lum":        round(feats["bot_lum"],   2),
    }


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
FIELDNAMES = [
    "frame", "source", "condition_id", "condition_name", "confidence",
    "lum_mean", "sat_mean", "lap_var", "glare_pct",
    "fog_contrast", "top_lum", "bot_lum",
]


def run(image_dir: str, output_csv: str, source: str = "real_car",
        workers: int = 4):

    image_dir = Path(image_dir)
    img_paths = sorted([
        str(p) for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ])

    if not img_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print(f"[ConditionClassifier] Processing {len(img_paths)} frames "
          f"(source={source}, workers={workers}) ...")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_frame, p, source): p for p in img_paths}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Classifying conditions"):
            r = fut.result()
            if r:
                results.append(r)

    # Sort by frame filename (timestamp order)
    results.sort(key=lambda x: x["frame"])

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)

    # Summary
    from collections import Counter
    counts = Counter(r["condition_name"] for r in results)
    print(f"\n[ConditionClassifier] Done — {len(results)} frames labelled.")
    print("  Condition distribution:")
    for cond_id, cond_name in ALL_CONDITIONS:
        n = counts.get(cond_name, 0)
        pct = 100 * n / len(results) if results else 0
        print(f"    Cond {cond_id} | {cond_name:<40} {n:>5} frames ({pct:.1f}%)")

    print(f"\n  Output written to: {output_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-classify driving frames into research conditions")
    parser.add_argument("--image-dir",  required=True,
                        help="Folder containing .jpg/.png frames")
    parser.add_argument("--output-csv", required=True,
                        help="Output CSV with per-frame condition labels")
    parser.add_argument("--source", default="real_car",
                        choices=["real_car", "carla", "tesla"],
                        help="Data source tag")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel threads (default 4)")
    args = parser.parse_args()
    run(args.image_dir, args.output_csv, args.source, args.workers)
