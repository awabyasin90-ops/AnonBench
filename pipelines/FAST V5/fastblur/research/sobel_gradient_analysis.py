"""
sobel_gradient_analysis.py — OPINA Research Automation  I1
============================================================
Formalizes the I1 finding: "39.66% of boxes fail the Sobel gradient
energy attenuation check — publishable as Failure Pattern Type 3."

This script:
1. Reads existing detections_all.csv and blurred frames
2. Computes Sobel gradient energy ratio (blurred/original) per box
3. Produces a failure taxonomy breakdown:
   - Pattern Type 1: Low-confidence detections (<0.35)  — precision risk
   - Pattern Type 2: Tiny boxes (<15px)                 — sub-resolution
   - Pattern Type 3: Sobel gradient persistence          — blur insufficient
   - Pattern Type 4: Laplacian residual                  — texture leakage
4. Stratifies all patterns by condition (from condition_classifier output)
5. Produces paper-ready LaTeX table + CSV + JSON

Usage:
    python sobel_gradient_analysis.py
        --blurred-dir    <blurred output frames>
        --original-dir   <original frames>   (optional — needed for true ratio)
        --detections-csv <detections_all.csv>
        --condition-csv  <frame_conditions.csv>
        --output-dir     <output folder>

Output:
    failure_patterns_by_condition.csv   — per-condition pattern counts
    sobel_detail.csv                    — per-box Sobel/Laplacian metrics
    failure_taxonomy_table.tex          — LaTeX table for the paper
    failure_taxonomy_report.json        — machine-readable summary
"""

import os
import argparse
import csv
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────
# Thresholds matching FAST v4 audit logic
# ─────────────────────────────────────────────────────────────────
SOBEL_RATIO_FAIL      = 0.50   # gradient energy ratio blurred/orig > 0.5 → FAIL
LAPLACIAN_RATIO_FAIL  = 0.50   # Laplacian ratio > 0.5 → FAIL
LOW_CONF_THRESHOLD    = 0.35   # Pattern Type 1
MIN_BOX_SIZE_PX       = 15     # Pattern Type 2 (side length)


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


def compute_sobel_energy(img_gray: np.ndarray) -> float:
    """Mean absolute Sobel gradient magnitude (first-order derivative)."""
    sx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    return float(np.sqrt(sx**2 + sy**2).mean())


def compute_laplacian_var(img_gray: np.ndarray) -> float:
    """Laplacian variance (second-order derivative attenuation check)."""
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


def crop_tight(img: np.ndarray, x1, y1, x2, y2) -> np.ndarray | None:
    h, w = img.shape[:2]
    cx1 = max(0, int(x1))
    cy1 = max(0, int(y1))
    cx2 = min(w, int(x2))
    cy2 = min(h, int(y2))
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    return img[cy1:cy2, cx1:cx2]


def run(blurred_dir: str, detections_csv: str, condition_csv: str,
        output_dir: str, original_dir: str = None):

    blurred_dir  = Path(blurred_dir)
    output_dir   = Path(output_dir)
    original_dir = Path(original_dir) if original_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    det_df   = pd.read_csv(detections_csv)
    cond_df  = pd.read_csv(condition_csv)
    cond_map = dict(zip(cond_df["frame"], cond_df["condition_name"]))

    print(f"[SobelAnalysis] Loaded {len(det_df)} detections across "
          f"{det_df['frame'].nunique()} frames.")

    detail_rows = []
    unique_frames = det_df["frame"].unique()

    for frame_name in tqdm(unique_frames, desc="Sobel analysis"):
        condition    = cond_map.get(frame_name, "Unknown")
        blurred_path = blurred_dir / frame_name
        orig_path    = (original_dir / frame_name) if original_dir else None

        blurred_img = cv2.imread(str(blurred_path))
        orig_img    = cv2.imread(str(orig_path)) if orig_path else None

        if blurred_img is None:
            continue

        frame_rows = det_df[det_df["frame"] == frame_name]

        for _, row in frame_rows.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            cls      = row["class"]
            conf     = float(row["confidence"])
            box_area = float(row["box_area_px"])

            # Box side length (min dimension for size check)
            box_w = x2 - x1
            box_h = y2 - y1
            min_side = min(box_w, box_h)

            # Pattern Type 1: Low confidence
            pattern1_low_conf = conf < LOW_CONF_THRESHOLD

            # Pattern Type 2: Sub-resolution
            pattern2_tiny = min_side < MIN_BOX_SIZE_PX

            # Crop from blurred image
            blurred_crop = crop_tight(blurred_img, x1, y1, x2, y2)
            orig_crop    = crop_tight(orig_img, x1, y1, x2, y2) if orig_img else None

            # ── Compute gradient metrics on blurred crop ──────────
            sobel_ratio   = None
            lap_ratio     = None
            pattern3_sobel = False
            pattern4_lap   = False
            reliable       = (not pattern2_tiny) and (blurred_crop is not None)

            if reliable:
                gray_blur = cv2.cvtColor(blurred_crop, cv2.COLOR_BGR2GRAY).astype(np.float64)

                if orig_crop is not None:
                    gray_orig = cv2.cvtColor(orig_crop, cv2.COLOR_BGR2GRAY).astype(np.float64)
                    # True ratio (original available)
                    sobel_orig = compute_sobel_energy(gray_orig)
                    sobel_blur = compute_sobel_energy(gray_blur)
                    lap_orig   = compute_laplacian_var(gray_orig)
                    lap_blur   = compute_laplacian_var(gray_blur)

                    sobel_ratio = (sobel_blur / sobel_orig) if sobel_orig > 0.1 else None
                    lap_ratio   = (lap_blur   / lap_orig)   if lap_orig   > 1.0 else None
                else:
                    # Proxy when original is absent: compare blurred box against immediate background margin
                    sobel_blur = compute_sobel_energy(gray_blur)
                    lap_blur   = compute_laplacian_var(gray_blur)

                    # Extract context margin (pad box by 12px)
                    pad_x = max(6, int((x2 - x1) * 0.25))
                    pad_y = max(6, int((y2 - y1) * 0.25))
                    padded = crop_tight(blurred_img, x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)
                    if padded is not None and padded.shape[0] > gray_blur.shape[0] and padded.shape[1] > gray_blur.shape[1]:
                        gray_pad  = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY).astype(np.float64)
                        sobel_ctx = compute_sobel_energy(gray_pad)
                        lap_ctx   = compute_laplacian_var(gray_pad)
                        sobel_ratio = (sobel_blur / sobel_ctx) if sobel_ctx > 0.1 else None
                        lap_ratio   = (lap_blur   / lap_ctx)   if lap_ctx   > 1.0 else None
                    else:
                        sobel_ratio = round(sobel_blur / 20.0, 4)
                        lap_ratio   = round(lap_blur / 40.0, 4)

                if sobel_ratio is not None:
                    pattern3_sobel = bool(sobel_ratio >= SOBEL_RATIO_FAIL)
                if lap_ratio is not None:
                    pattern4_lap   = bool(lap_ratio   >= LAPLACIAN_RATIO_FAIL)

            detail_rows.append({
                "frame":           frame_name,
                "condition":       condition,
                "class":           cls,
                "confidence":      round(conf, 4),
                "box_area_px":     round(box_area, 1),
                "min_side_px":     round(min_side, 1),
                "reliable":        reliable,
                # Pattern flags
                "P1_low_conf":     pattern1_low_conf,
                "P2_tiny_box":     pattern2_tiny,
                "P3_sobel_fail":   pattern3_sobel,
                "P4_lap_fail":     pattern4_lap,
                # Raw metrics
                "sobel_ratio":     round(sobel_ratio, 4) if sobel_ratio else None,
                "lap_ratio":       round(lap_ratio,   4) if lap_ratio   else None,
            })

    detail_df = pd.DataFrame(detail_rows)
    detail_path = output_dir / "sobel_detail.csv"
    detail_df.to_csv(detail_path, index=False)

    # ─────────────────────────────────────────────────────────────
    # Per-condition failure pattern breakdown
    # ─────────────────────────────────────────────────────────────
    conditions = sorted(detail_df["condition"].unique())
    pattern_rows = []

    for cond in conditions:
        for cls in ["face", "plate"]:
            cdf = detail_df[(detail_df["condition"] == cond) &
                             (detail_df["class"]     == cls)]
            if len(cdf) == 0:
                continue

            n_total     = len(cdf)
            n_reliable  = cdf["reliable"].sum()
            n_p1        = cdf["P1_low_conf"].sum()
            n_p2        = cdf["P2_tiny_box"].sum()
            n_p3        = cdf[cdf["P3_sobel_fail"].notna()]["P3_sobel_fail"].sum()
            n_p4        = cdf[cdf["P4_lap_fail"].notna()]["P4_lap_fail"].sum()

            pattern_rows.append({
                "condition":           cond,
                "class":               cls,
                "n_total_detections":  int(n_total),
                "n_reliable":          int(n_reliable),
                "P1_low_conf_n":       int(n_p1),
                "P1_low_conf_pct":     round(100 * n_p1 / n_total, 2),
                "P2_tiny_box_n":       int(n_p2),
                "P2_tiny_box_pct":     round(100 * n_p2 / n_total, 2),
                "P3_sobel_fail_n":     int(n_p3),
                "P3_sobel_fail_pct":   round(100 * n_p3 / n_reliable, 2) if n_reliable > 0 else 0,
                "P4_lap_fail_n":       int(n_p4),
                "P4_lap_fail_pct":     round(100 * n_p4 / n_reliable, 2) if n_reliable > 0 else 0,
            })

    pattern_df = pd.DataFrame(pattern_rows)
    pattern_path = output_dir / "failure_patterns_by_condition.csv"
    save_table_with_preview(pattern_df, pattern_path, "Anonymization Failure Patterns by Driving Condition")

    # ─────────────────────────────────────────────────────────────
    # Global stats (paper numbers)
    # ─────────────────────────────────────────────────────────────
    global_stats = {
        "total_detections":      len(detail_df),
        "reliable_detections":   int(detail_df["reliable"].sum()),
        "P1_low_conf": {
            "n":   int(detail_df["P1_low_conf"].sum()),
            "pct": round(100 * detail_df["P1_low_conf"].mean(), 2)
        },
        "P2_tiny_box": {
            "n":   int(detail_df["P2_tiny_box"].sum()),
            "pct": round(100 * detail_df["P2_tiny_box"].mean(), 2)
        },
        "P3_sobel_fail": {
            "n":   int(detail_df["P3_sobel_fail"].sum()),
            "pct_of_reliable": round(
                100 * detail_df["P3_sobel_fail"].sum() /
                max(1, detail_df["reliable"].sum()), 2)
        },
        "P4_lap_fail": {
            "n":   int(detail_df["P4_lap_fail"].sum()),
            "pct_of_reliable": round(
                100 * detail_df["P4_lap_fail"].sum() /
                max(1, detail_df["reliable"].sum()), 2)
        }
    }

    report = {"global": global_stats, "by_condition": pattern_df.to_dict(orient="records")}
    report_path = output_dir / "failure_taxonomy_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ─────────────────────────────────────────────────────────────
    # LaTeX table for the paper
    # ─────────────────────────────────────────────────────────────
    _write_latex_table(pattern_df, output_dir / "failure_taxonomy_table.tex")

    # Print
    print("\n[SobelAnalysis] ── Global Failure Pattern Summary ─────────────")
    for k, v in global_stats.items():
        if isinstance(v, dict):
            print(f"  {k}: n={v.get('n', '?')}  pct={v.get('pct', v.get('pct_of_reliable','?'))}%")
        else:
            print(f"  {k}: {v}")
    print(f"\n[SobelAnalysis] Per-condition breakdown:")
    print(pattern_df.to_string(index=False))
    print(f"\n  Files written:")
    print(f"    {detail_path}")
    print(f"    {pattern_path}")
    print(f"    {report_path}")
    print(f"    {output_dir / 'failure_taxonomy_table.tex'}")

    return pattern_df


def _write_latex_table(df: pd.DataFrame, path: Path):
    """Generate a LaTeX booktabs table for the paper."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Anonymization Failure Pattern Taxonomy by Driving Condition}",
        r"\label{tab:failure_taxonomy}",
        r"\begin{tabular}{l l r r r r r r}",
        r"\toprule",
        r"Condition & Class & Total & "
        r"P1 Low-Conf (\%) & P2 Tiny-Box (\%) & "
        r"P3 Sobel Fail (\%) & P4 Lap Fail (\%) \\",
        r"\midrule",
    ]
    for _, row in df.iterrows():
        line = (
            f"{row['condition']} & {row['class']} & {row['n_total_detections']} & "
            f"{row['P1_low_conf_pct']:.1f} & {row['P2_tiny_box_pct']:.1f} & "
            f"{row['P3_sobel_fail_pct']:.1f} & {row['P4_lap_fail_pct']:.1f} \\\\"
        )
        lines.append(line)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sobel gradient failure analysis — OPINA paper I1")
    parser.add_argument("--blurred-dir",    required=True)
    parser.add_argument("--original-dir",   default=None)
    parser.add_argument("--detections-csv", required=True)
    parser.add_argument("--condition-csv",  required=True)
    parser.add_argument("--output-dir",     required=True)
    args = parser.parse_args()
    run(args.blurred_dir, args.detections_csv, args.condition_csv,
        args.output_dir, args.original_dir)
