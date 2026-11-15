"""
hitl_review_pack.py — Human-in-the-Loop (HITL) Manual Review Pack Generator
=============================================================================
Automatically extracts and formats side-by-side visual evidence for all flagged
failures (Identifiable Faces After Blur & Readable License Plates).

Eliminates the need for engineers or auditors to search through thousands of frames.
Produces:
  1. Side-by-side comparison image for every failure (Original vs Blurred, with 4x zoom and context).
  2. Unified Failure Contact Sheet (single image grid containing all edge cases).
  3. Interactive HTML Review Gallery (with side-by-side view and review status controls).
  4. Markdown Review Summary (HITL_MANUAL_REVIEW_PACK.md) with direct clickable file links.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd


def crop_with_padding(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_pct: float = 0.5) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Crops region with percentage padding, clamping to image borders."""
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1
    pad_x = int(bw * pad_pct)
    pad_y = int(bh * pad_pct)

    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)

    crop = img[cy1:cy2, cx1:cx2].copy()
    # Relative box coordinates inside the context crop
    rel_box = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    return crop, rel_box


def draw_bounding_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color=(0, 0, 255), thickness=2, label=""):
    """Draws bounding box with high-visibility outline and optional label."""
    res = img.copy()
    cv2.rectangle(res, (x1, y1), (x2, y2), color, thickness)
    if label:
        cv2.putText(res, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return res


def render_side_by_side_card(
    orig_img: Optional[np.ndarray],
    blur_img: np.ndarray,
    box: Tuple[int, int, int, int],
    meta: Dict[str, any],
    idx: int,
    card_width: int = 1200,
    card_height: int = 500
) -> np.ndarray:
    """
    Renders a side-by-side 4-panel visual audit card:
      Panel 1: Context (Original) with bounding box
      Panel 2: Context (Blurred) with bounding box
      Panel 3: Tight Crop 4x Zoom (Original)
      Panel 4: Tight Crop 4x Zoom (Blurred) showing 70% inner core
    """
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    # 1. Context crops (2.5x padding)
    blur_ctx, blur_rel = crop_with_padding(blur_img, x1, y1, x2, y2, pad_pct=2.0)
    blur_ctx_annotated = draw_bounding_box(blur_ctx, *blur_rel, color=(0, 255, 255), thickness=2, label=f"{bw}x{bh}")

    if orig_img is not None:
        orig_ctx, orig_rel = crop_with_padding(orig_img, x1, y1, x2, y2, pad_pct=2.0)
        orig_ctx_annotated = draw_bounding_box(orig_ctx, *orig_rel, color=(0, 200, 0), thickness=2, label="Original")
    else:
        # Fallback if no raw original available: produce edge gradient view
        gray_blur = cv2.cvtColor(blur_ctx, cv2.COLOR_BGR2GRAY)
        sobel_x = cv2.Sobel(gray_blur, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_blur, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.uint8(np.clip(np.sqrt(sobel_x**2 + sobel_y**2), 0, 255))
        orig_ctx_annotated = cv2.cvtColor(edge_mag, cv2.COLOR_GRAY2BGR)
        cv2.putText(orig_ctx_annotated, "Sobel Edge Map", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Tight crops with 4x zoom
    h_blur, w_blur = blur_img.shape[:2]
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w_blur, x2)
    cy2 = min(h_blur, y2)
    tight_blur = blur_img[cy1:cy2, cx1:cx2].copy() if cx2 > cx1 and cy2 > cy1 else np.zeros((30, 30, 3), np.uint8)

    # Inscribe the central 70% core
    tb_h, tb_w = tight_blur.shape[:2]
    mx = int(round(tb_w * 0.15))
    my = int(round(tb_h * 0.15))
    cv2.rectangle(tight_blur, (mx, my), (max(mx+1, tb_w - mx), max(my+1, tb_h - my)), (0, 255, 255), 1)

    zoom_target = (160, 180)
    tight_blur_zoomed = cv2.resize(tight_blur, zoom_target, interpolation=cv2.INTER_NEAREST)

    if orig_img is not None:
        tight_orig = orig_img[cy1:cy2, cx1:cx2].copy() if cx2 > cx1 and cy2 > cy1 else np.zeros((30, 30, 3), np.uint8)
        tight_orig_zoomed = cv2.resize(tight_orig, zoom_target, interpolation=cv2.INTER_NEAREST)
    else:
        tight_orig_zoomed = np.zeros((zoom_target[1], zoom_target[0], 3), np.uint8)
        cv2.putText(tight_orig_zoomed, "N/A", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)

    # Resize context panels to uniform size (e.g. 360x280)
    ctx_size = (380, 280)
    orig_ctx_resized = cv2.resize(orig_ctx_annotated, ctx_size, interpolation=cv2.INTER_AREA)
    blur_ctx_resized = cv2.resize(blur_ctx_annotated, ctx_size, interpolation=cv2.INTER_AREA)

    # Canvas assembly
    canvas = np.zeros((card_height, card_width, 3), dtype=np.uint8)
    canvas[:] = (24, 27, 33)  # Professional dark background

    # Header Bar
    cv2.rectangle(canvas, (0, 0), (card_width, 60), (36, 41, 51), -1)
    title_text = f"FAILURE #{idx:02d}: [{meta.get('class', 'face').upper()}] {meta.get('frame', '')}"
    cv2.putText(canvas, title_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    status_tag = f"FLAG: {meta.get('category', 'Identifiable Face')} | Conf: {meta.get('conf', 0.0)} | Size: {bw}x{bh} px"
    cv2.putText(canvas, status_tag, (card_width - 560, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 180, 255), 1, cv2.LINE_AA)

    # Panel Layout:
    # [Panel 1: Orig Context 380x280]  [Panel 2: Blur Context 380x280]  [Panel 3: Orig Zoom 160x180]  [Panel 4: Blur Zoom 160x180]
    y_panels = 80
    
    # 1. Orig Context
    x_p1 = 20
    canvas[y_panels:y_panels+ctx_size[1], x_p1:x_p1+ctx_size[0]] = orig_ctx_resized
    cv2.rectangle(canvas, (x_p1, y_panels), (x_p1+ctx_size[0], y_panels+ctx_size[1]), (60, 65, 75), 1)
    cv2.putText(canvas, "1. ORIGINAL CONTEXT (BEFORE)", (x_p1 + 10, y_panels + ctx_size[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 220, 100), 1, cv2.LINE_AA)

    # 2. Blur Context
    x_p2 = x_p1 + ctx_size[0] + 20
    canvas[y_panels:y_panels+ctx_size[1], x_p2:x_p2+ctx_size[0]] = blur_ctx_resized
    cv2.rectangle(canvas, (x_p2, y_panels), (x_p2+ctx_size[0], y_panels+ctx_size[1]), (60, 65, 75), 1)
    cv2.putText(canvas, "2. BLURRED OUTPUT (AFTER)", (x_p2 + 10, y_panels + ctx_size[1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # 3. Orig Zoom
    x_p3 = x_p2 + ctx_size[0] + 25
    y_zoom = y_panels + 40
    canvas[y_zoom:y_zoom+zoom_target[1], x_p3:x_p3+zoom_target[0]] = tight_orig_zoomed
    cv2.rectangle(canvas, (x_p3, y_zoom), (x_p3+zoom_target[0], y_zoom+zoom_target[1]), (0, 200, 0), 1)
    cv2.putText(canvas, "RAW FACE (4x)", (x_p3 + 12, y_zoom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1, cv2.LINE_AA)

    # 4. Blur Zoom
    x_p4 = x_p3 + zoom_target[0] + 15
    canvas[y_zoom:y_zoom+zoom_target[1], x_p4:x_p4+zoom_target[0]] = tight_blur_zoomed
    cv2.rectangle(canvas, (x_p4, y_zoom), (x_p4+zoom_target[0], y_zoom+zoom_target[1]), (0, 255, 255), 1)
    cv2.putText(canvas, "BLUR CORE (4x)", (x_p4 + 10, y_zoom - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 255), 1, cv2.LINE_AA)

    # Telemetry Footer Banner
    y_footer = card_height - 70
    cv2.rectangle(canvas, (20, y_footer), (card_width - 20, card_height - 15), (30, 34, 42), -1)
    cv2.rectangle(canvas, (20, y_footer), (card_width - 20, card_height - 15), (55, 60, 72), 1)

    inner_sharp = meta.get("inner_lap", "N/A")
    lap_text = f"Inner Core Sharpness: {inner_sharp} (Fail Thresh >= 45.0)" if inner_sharp != "N/A" else "Readability Test Failed"
    footer_text = (
        f"AUDIT DIAGNOSTIC: {lap_text} | Coordinates: [x1={x1}, y1={y1}, x2={x2}, y2={y2}] | "
        f"Condition: {meta.get('condition', 'Clear Day')} | Verdict: {meta.get('result', 'FAIL')}"
    )
    cv2.putText(canvas, footer_text, (35, y_footer + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 225, 235), 1, cv2.LINE_AA)

    return canvas


def generate_review_pack(
    recovery_detail_csv: Path,
    detections_csv: Path,
    blurred_dir: Path,
    output_dir: Path,
    original_dir: Optional[Path] = None
) -> Dict[str, any]:
    """
    Main entry point for generating the complete Human-in-the-Loop review package.
    """
    recovery_detail_csv = Path(recovery_detail_csv)
    detections_csv = Path(detections_csv)
    blurred_dir = Path(blurred_dir)
    output_dir = Path(output_dir)
    original_dir = Path(original_dir) if original_dir else None

    review_pack_dir = output_dir / "hitl_review_pack"
    cards_dir = review_pack_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    if not recovery_detail_csv.exists() or not detections_csv.exists():
        print(f"[HITL Review] Prerequisite CSVs missing: {recovery_detail_csv} or {detections_csv}")
        return {"status": "skipped", "failures_count": 0}

    atk_df = pd.read_csv(recovery_detail_csv)
    det_df = pd.read_csv(detections_csv)

    fails = atk_df[atk_df["result"] == "FAIL"].copy()
    num_fails = len(fails)
    print(f"\n[HITL Review] Identified {num_fails} flagged edge case(s) requiring human inspection.")

    if num_fails == 0:
        print("[HITL Review] Zero failures! 100% anonymization verified. No review pack needed.")
        return {"status": "clean", "failures_count": 0}

    card_paths = []
    card_records = []

    for idx, (_, r) in enumerate(fails.iterrows(), 1):
        frame_name = r["frame"]
        area = r.get("box_area", 0)
        
        # Match detection bounding box
        matched = det_df[(det_df["frame"] == frame_name) & (det_df["box_area_px"] == area)]
        if len(matched) == 0:
            matched = det_df[(det_df["frame"] == frame_name) & (det_df["class"] == r["class"])]
        if len(matched) == 0:
            continue
        m = matched.iloc[0]

        x1 = int(round(float(m["x1"])))
        y1 = int(round(float(m["y1"])))
        x2 = int(round(float(m["x2"])))
        y2 = int(round(float(m["y2"])))

        blur_path = blurred_dir / frame_name
        orig_path = (original_dir / frame_name) if original_dir else None

        blur_img = cv2.imread(str(blur_path))
        orig_img = cv2.imread(str(orig_path)) if (orig_path and orig_path.exists()) else None

        if blur_img is None:
            continue

        meta = {
            "frame": frame_name,
            "class": r["class"],
            "category": r.get("category", "Biometric Leakage"),
            "conf": m.get("confidence", 0.0),
            "inner_lap": r.get("inner_lap", "N/A"),
            "condition": r.get("condition", "Clear Day"),
            "result": r.get("result", "FAIL"),
        }

        # Render 4-panel visual card
        card = render_side_by_side_card(
            orig_img=orig_img,
            blur_img=blur_img,
            box=(x1, y1, x2, y2),
            meta=meta,
            idx=idx
        )

        card_filename = f"side_by_side_{idx:02d}_{r['class']}_{Path(frame_name).stem}.png"
        card_out_path = cards_dir / card_filename
        cv2.imwrite(str(card_out_path), card)
        card_paths.append(card_out_path)

        card_records.append({
            "idx": idx,
            "frame": frame_name,
            "class": r["class"],
            "conf": meta["conf"],
            "box": f"[{x1}, {y1}, {x2}, {y2}]",
            "size": f"{x2 - x1}x{y2 - y1} px",
            "inner_lap": meta["inner_lap"],
            "image_filename": card_filename,
            "image_path": str(card_out_path)
        })

    # Render unified contact sheet (stacking cards vertically)
    if card_paths:
        cards_imgs = [cv2.imread(str(p)) for p in card_paths if p.exists()]
        if cards_imgs:
            contact_sheet = np.vstack(cards_imgs)
            contact_sheet_path = review_pack_dir / "contact_sheet_all_failures.png"
            cv2.imwrite(str(contact_sheet_path), contact_sheet)
            print(f"[HITL Review] Generated Contact Sheet -> {contact_sheet_path}")

    # Generate Markdown review document
    md_path = review_pack_dir / "HITL_MANUAL_REVIEW_PACK.md"
    md_lines = [
        "# Human-in-the-Loop (HITL) Edge Case Review Pack",
        "",
        f"> **Generated:** Automated Pipeline Gate | **Flagged Edge Cases:** {num_fails}  ",
        f"> **Action Required:** Inspect side-by-side comparisons below to verify whether residual features present genuine biometric identity risk.",
        "",
        "---",
        "",
        "## Summary of Flagged Instances",
        "",
        "| ID | Frame | Class | Box Size | Confidence | Inner Sharpness | Reason | Comparison Card |",
        "|:---|:---|:---|:---|---:|---:|:---|:---|",
    ]
    for rec in card_records:
        rel_img = f"cards/{rec['image_filename']}"
        md_lines.append(
            f"| **#{rec['idx']}** | `{rec['frame']}` | `{rec['class']}` | {rec['size']} | {rec['conf']} | "
            f"**{rec['inner_lap']}** | Near-field core variance $\\ge 45.0$ | [View Comparison Image]({rel_img}) |"
        )
    md_lines.extend([
        "",
        "---",
        "",
        "## Visual Side-by-Side Review Cards",
        ""
    ])
    for rec in card_records:
        rel_img = f"cards/{rec['image_filename']}"
        md_lines.extend([
            f"### Case #{rec['idx']}: {rec['class'].upper()} in `{rec['frame']}`",
            f"- **Bounding Box:** `{rec['box']}` ({rec['size']}) | **Confidence:** `{rec['conf']}`",
            f"- **Inner Landmark Sharpness:** `{rec['inner_lap']}` (Threshold: 45.0)",
            "",
            f"![Comparison #{rec['idx']}]({rel_img})",
            "",
            "---",
            ""
        ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"[HITL Review] Generated Markdown Review Pack -> {md_path}")

    # Generate standalone interactive HTML review gallery
    html_path = review_pack_dir / "hitl_review_gallery.html"
    cards_html = ""
    for rec in card_records:
        cards_html += f"""
        <div class="review-card">
            <div class="card-header">
                <span class="card-title">Case #{rec['idx']}: {rec['class'].upper()} &bull; {rec['frame']}</span>
                <span class="badge badge-warning">Sharpness: {rec['inner_lap']}</span>
            </div>
            <div class="card-meta">
                <span><strong>Coordinates:</strong> {rec['box']}</span>
                <span><strong>Size:</strong> {rec['size']}</span>
                <span><strong>Confidence:</strong> {rec['conf']}</span>
                <span><strong>Threshold:</strong> &ge; 45.0</span>
            </div>
            <div class="image-wrapper">
                <img src="cards/{rec['image_filename']}" alt="Comparison #{rec['idx']}" />
            </div>
            <div class="action-bar">
                <button class="btn btn-approve" onclick="this.classList.toggle('active')">&#10004; Mark as Sufficiently Blurred (Pass)</button>
                <button class="btn btn-reject" onclick="this.classList.toggle('active')">&#10008; Confirm Identity Leakage (Flag)</button>
            </div>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HITL Privacy Review Gallery — {num_fails} Flagged Case(s)</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #0f1218;
            color: #e1e4e8;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            max-width: 1240px;
            margin: 0 auto 24px auto;
            padding: 20px 24px;
            background: #181c24;
            border-radius: 8px;
            border-left: 4px solid #3b82f6;
        }}
        h1 {{ margin: 0 0 8px 0; font-size: 24px; font-weight: 600; color: #fff; }}
        p {{ margin: 0; font-size: 14px; color: #94a3b8; }}
        .gallery-container {{
            max-width: 1240px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 28px;
        }}
        .review-card {{
            background: #181c24;
            border: 1px solid #262c38;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 16px rgba(0,0,0,0.3);
        }}
        .card-header {{
            padding: 16px 20px;
            background: #202632;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid #2d3444;
        }}
        .card-title {{ font-weight: 600; font-size: 16px; color: #f1f5f9; }}
        .badge {{
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 600;
        }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }}
        .card-meta {{
            padding: 12px 20px;
            background: #14171f;
            display: flex;
            gap: 24px;
            font-size: 13px;
            color: #94a3b8;
            border-bottom: 1px solid #262c38;
        }}
        .image-wrapper {{
            padding: 16px;
            background: #0f1218;
            display: flex;
            justify-content: center;
        }}
        .image-wrapper img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #2d3444;
        }}
        .action-bar {{
            padding: 14px 20px;
            background: #181c24;
            display: flex;
            gap: 12px;
            justify-content: flex-end;
            border-top: 1px solid #262c38;
        }}
        .btn {{
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }}
        .btn-approve {{
            background: #10b981;
            color: #fff;
        }}
        .btn-approve:hover {{ background: #059669; }}
        .btn-approve.active {{ background: #047857; border-color: #34d399; font-weight: bold; }}
        .btn-reject {{
            background: #ef4444;
            color: #fff;
        }}
        .btn-reject:hover {{ background: #dc2626; }}
        .btn-reject.active {{ background: #b91c1c; border-color: #f87171; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Human-in-the-Loop Privacy Inspection Gallery</h1>
        <p>Showing {num_fails} flagged edge case(s) from automated detection & recovery audit. Click buttons below each image to record auditor approval or confirm identity leakage.</p>
    </div>
    <div class="gallery-container">
        {cards_html}
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[HITL Review] Generated Interactive HTML Gallery -> {html_path}")

    return {
        "status": "success",
        "failures_count": num_fails,
        "review_pack_dir": str(review_pack_dir),
        "markdown_path": str(md_path),
        "html_path": str(html_path),
        "contact_sheet": str(contact_sheet_path) if card_paths else ""
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human-in-the-Loop Review Pack Generator")
    parser.add_argument("--recovery-detail", required=True, help="recovery_attack_detail.csv path")
    parser.add_argument("--detections-csv",  required=True, help="detections_all.csv path")
    parser.add_argument("--blurred-dir",    required=True, help="Path to blurred frames")
    parser.add_argument("--output-dir",     required=True, help="Output destination folder")
    parser.add_argument("--original-dir",   default=None,  help="Path to unblurred raw frames (optional)")
    args = parser.parse_args()

    generate_review_pack(
        recovery_detail_csv = Path(args.recovery_detail),
        detections_csv      = Path(args.detections_csv),
        blurred_dir         = Path(args.blurred_dir),
        output_dir          = Path(args.output_dir),
        original_dir        = Path(args.original_dir) if args.original_dir else None
    )
