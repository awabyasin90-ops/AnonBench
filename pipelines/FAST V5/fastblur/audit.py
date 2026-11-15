"""fastblur/audit.py — Audit Gate, Laplacian Blur Effectiveness, Pixel Integrity, and Metadata Scan.

Implements rigorous, verifiable quality and privacy checks:
- Stratified sampling across confidence and size buckets
- Laplacian variance ratio check (var(output) / var(input)) with UNRELIABLE tagging (< 8 px)
- Pixel integrity test outside dilated masks
- Output metadata sanitization scanner
"""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .render_v4 import RenderConfig, _create_feathered_mask
except ImportError:
    from fastblur.render_v4 import RenderConfig, _create_feathered_mask


def get_size_bucket(side_px: float) -> str:
    """Classifies bounding box side (sqrt(area)) into standard size bucket."""
    if side_px < 10.0:
        return "<10"
    elif side_px < 15.0:
        return "10-15"
    elif side_px < 20.0:
        return "15-20"
    elif side_px < 30.0:
        return "20-30"
    elif side_px < 50.0:
        return "30-50"
    else:
        return ">=50"


def get_conf_bucket(conf: float, min_thresh: float = 0.15) -> str:
    """Classifies confidence into 0.05 step buckets."""
    c = max(0.0, min(1.0, float(conf)))
    # Step intervals: 0.15-0.20, 0.20-0.25, ..., 0.95-1.00
    lower = math.floor(c * 20.0) / 20.0
    upper = min(1.0, lower + 0.05)
    return f"[{lower:.2f},{upper:.2f})" if upper < 1.0 else f"[{lower:.2f},1.00]"


def sample_frames_stratified(
    frame_records: list[Any],
    seed: int = 42,
    target_sample_size: int | None = None,
) -> list[Any]:
    """Stratified sampling of frames across confidence and size strata.
    
    Formula: sample_size = min(total_frames, max(200, ceil(0.05 * total_frames)))
    """
    total_frames = len(frame_records)
    if total_frames == 0:
        return []

    if target_sample_size is None:
        target_sample_size = min(total_frames, max(200, math.ceil(0.05 * total_frames)))
    else:
        target_sample_size = min(total_frames, target_sample_size)

    if target_sample_size >= total_frames:
        return list(frame_records)

    # Group frames into strata based on detections present
    strata: dict[str, list[Any]] = {}
    for r in frame_records:
        boxes = getattr(r, "face_boxes", []) + getattr(r, "plate_boxes", [])
        if not boxes:
            stratum = "empty"
        else:
            # Pick min confidence and min side
            min_c = min(float(b[4]) for b in boxes if len(b) >= 5)
            sides = [math.sqrt(max(1.0, (b[2] - b[0]) * (b[3] - b[1]))) for b in boxes if len(b) >= 4]
            min_s = min(sides) if sides else 10.0
            stratum = f"{get_conf_bucket(min_c)}_{get_size_bucket(min_s)}"
        strata.setdefault(stratum, []).append(r)

    # Proportional stratified sampling
    rng = random.Random(seed)
    sampled = []
    strata_keys = sorted(strata.keys())

    # First pass: allocate proportionally
    allocations = {}
    for k in strata_keys:
        count = len(strata[k])
        prop = count / total_frames
        alloc = int(round(prop * target_sample_size))
        allocations[k] = min(count, max(1 if count > 0 and target_sample_size >= len(strata_keys) else 0, alloc))

    current_sampled_count = sum(allocations.values())
    # Adjust to match target_sample_size exactly
    diff = target_sample_size - current_sampled_count
    if diff > 0:
        # Add to largest strata
        for k in sorted(strata_keys, key=lambda x: len(strata[x]), reverse=True):
            available = len(strata[k]) - allocations[k]
            add = min(available, diff)
            allocations[k] += add
            diff -= add
            if diff <= 0:
                break
    elif diff < 0:
        for k in sorted(strata_keys, key=lambda x: allocations[x], reverse=True):
            take = min(allocations[k] - 1, abs(diff)) if allocations[k] > 1 else 0
            allocations[k] -= take
            diff += take
            if diff >= 0:
                break

    for k in strata_keys:
        items = list(strata[k])
        rng.shuffle(items)
        sampled.extend(items[: allocations[k]])

    # Final shuffle
    rng.shuffle(sampled)
    return sampled[:target_sample_size]


def _summarize_metric(
    rows: list[dict[str, Any]],
    ratio_key: str,
    status_key: str,
    threshold: float,
    total_boxes: int,
) -> dict[str, Any]:
    """Generates comprehensive summary and breakdowns for an audit metric."""
    rel_rows = [r for r in rows if r["status"] != "UNRELIABLE"]
    unrel_rows = [r for r in rows if r["status"] == "UNRELIABLE"]
    
    ratios = [float(r[ratio_key]) for r in rel_rows if ratio_key in r]
    passed_rows = [r for r in rel_rows if r.get(status_key) == "PASS"]
    failed_rows = [r for r in rel_rows if r.get(status_key) == "FAIL"]
    
    median_val = round(float(statistics.median(ratios)), 4) if ratios else 0.0
    p95_val = round(float(np.percentile(ratios, 95)), 4) if ratios else 0.0
    max_val = round(float(max(ratios)), 4) if ratios else 0.0
    
    # Class breakdown
    by_class: dict[str, Any] = {}
    for c in ["face", "plate"]:
        c_rel = [r for r in rel_rows if r["class"] == c]
        c_unrel = [r for r in unrel_rows if r["class"] == c]
        c_pass = [r for r in c_rel if r.get(status_key) == "PASS"]
        c_fail = [r for r in c_rel if r.get(status_key) == "FAIL"]
        c_ratios = [float(r[ratio_key]) for r in c_rel]
        by_class[c] = {
            "total": len(c_rel) + len(c_unrel),
            "checked": len(c_rel),
            "passed": len(c_pass),
            "failed": len(c_fail),
            "unreliable": len(c_unrel),
            "pass_pct": round(len(c_pass) / max(1, len(c_rel)) * 100.0, 2),
            "median_ratio": round(float(statistics.median(c_ratios)), 4) if c_ratios else 0.0,
            "p95_ratio": round(float(np.percentile(c_ratios, 95)), 4) if c_ratios else 0.0,
        }
        
    # Size breakdown
    by_size: dict[str, Any] = {}
    for sb in ["<10", "10-15", "15-20", "20-30", "30-50", ">=50"]:
        s_rel = [r for r in rel_rows if r.get("size_bucket") == sb]
        s_unrel = [r for r in unrel_rows if r.get("size_bucket") == sb]
        s_pass = [r for r in s_rel if r.get(status_key) == "PASS"]
        s_fail = [r for r in s_rel if r.get(status_key) == "FAIL"]
        s_ratios = [float(r[ratio_key]) for r in s_rel]
        by_size[sb] = {
            "total": len(s_rel) + len(s_unrel),
            "checked": len(s_rel),
            "passed": len(s_pass),
            "failed": len(s_fail),
            "unreliable": len(s_unrel),
            "pass_pct": round(len(s_pass) / max(1, len(s_rel)) * 100.0, 2),
            "median_ratio": round(float(statistics.median(s_ratios)), 4) if s_ratios else 0.0,
            "p95_ratio": round(float(np.percentile(s_ratios, 95)), 4) if s_ratios else 0.0,
        }
        
    if total_boxes == 0:
        metric_status = "NOT_EXECUTED"
    else:
        metric_status = "PASS" if (len(failed_rows) == 0 and len(rel_rows) > 0) else "FAIL"

    return {
        "status": metric_status,
        "threshold": threshold,
        "boxes_total": total_boxes,
        "boxes_checked": len(rel_rows),
        "boxes_passed": len(passed_rows),
        "boxes_failed": len(failed_rows),
        "boxes_unreliable": len(unrel_rows),
        "unreliable_fraction": round(len(unrel_rows) / max(1, total_boxes), 4),
        "ratio_median": median_val,
        "ratio_p95": p95_val,
        "ratio_max": max_val,
        "breakdown_by_class": by_class,
        "breakdown_by_size": by_size,
    }


def audit_laplacian_effectiveness(
    input_dir: Path,
    output_dir: Path,
    sampled_records: list[Any],
    max_ratio_threshold: float = 0.5,
    min_side_reliable: float = 8.0,
    out_csv_path: Path | None = None,
    render_cfg: RenderConfig | dict | None = None,
) -> dict[str, Any]:
    """Computes dual blur effectiveness metrics across sampled frames:
    1. Primary: Laplacian variance ratio inside actual applied mask (alpha > 0.9)
    2. Secondary: High-frequency gradient energy ratio inside actual applied mask
    3. Comparison: Legacy full bounding box Laplacian variance ratio
    """
    if not sampled_records:
        return {
            "status": "NOT_EXECUTED",
            "reason": "sample_size=0: no frames sampled for audit",
            "sample_size": 0,
            "boxes_checked": 0,
            "boxes_failed": 0,
            "boxes_unreliable": 0,
            "ratio_median": 0.0,
            "ratio_p95": 0.0,
            "ratio_max": 0.0,
            "max_ratio_threshold": max_ratio_threshold,
            "PASS": False,
        }

    if render_cfg is None:
        r_cfg = RenderConfig()
    elif isinstance(render_cfg, dict):
        r_cfg = RenderConfig.from_dict(render_cfg)
    else:
        r_cfg = render_cfg

    total_boxes = 0
    all_audit_rows: list[dict[str, Any]] = []

    for r in sampled_records:
        filename = getattr(r, "filename", "")
        in_p = input_dir / filename
        out_p = output_dir / filename

        if not in_p.exists() or not out_p.exists():
            continue

        in_img = cv2.imread(str(in_p))
        out_img = cv2.imread(str(out_p))
        if in_img is None or out_img is None or in_img.shape != out_img.shape:
            continue

        h_img, w_img = in_img.shape[:2]
        gray_in = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
        gray_out = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)

        boxes_with_cls = []
        for idx, b in enumerate(getattr(r, "face_boxes", [])):
            boxes_with_cls.append((f"{Path(filename).stem}_face_{idx}", "face", b))
        for idx, b in enumerate(getattr(r, "plate_boxes", [])):
            boxes_with_cls.append((f"{Path(filename).stem}_plate_{idx}", "plate", b))

        for box_id, cls_name, b in boxes_with_cls:
            total_boxes += 1
            x1_raw = float(b[0])
            y1_raw = float(b[1])
            x2_raw = float(b[2])
            y2_raw = float(b[3])

            bw = max(0.0, x2_raw - x1_raw)
            bh = max(0.0, y2_raw - y1_raw)
            side = math.sqrt(max(1.0, bw * bh))
            size_bucket = get_size_bucket(side)
            conf = float(b[4]) if len(b) >= 5 else 1.0

            # Reconstruct exact render coordinates with padding
            pad_ratio = r_cfg.face_pad_ratio if cls_name == "face" else r_cfg.plate_pad_ratio
            pad_w = bw * pad_ratio
            pad_h = bh * pad_ratio
            cx = (x1_raw + x2_raw) * 0.5
            cy = (y1_raw + y2_raw) * 0.5

            x1_pad = max(0, int(math.floor(cx - (bw * 0.5 + pad_w))))
            y1_pad = max(0, int(math.floor(cy - (bh * 0.5 + pad_h))))
            x2_pad = min(w_img, int(math.ceil(cx + (bw * 0.5 + pad_w))))
            y2_pad = min(h_img, int(math.ceil(cy + (bh * 0.5 + pad_h))))

            roi_w = x2_pad - x1_pad
            roi_h = y2_pad - y1_pad

            if roi_w < 3 or roi_h < 3:
                all_audit_rows.append({
                    "frame": filename, "box_id": box_id, "class": cls_name,
                    "x1": round(x1_raw, 2), "y1": round(y1_raw, 2), "x2": round(x2_raw, 2), "y2": round(y2_raw, 2),
                    "box_w": round(bw, 2), "box_h": round(bh, 2), "side_px": round(side, 2),
                    "size_bucket": size_bucket, "confidence": round(conf, 4),
                    "mask_var_in": 0.0, "mask_var_out": 0.0, "mask_laplacian_ratio": 1.0, "mask_laplacian_status": "UNRELIABLE",
                    "mask_grad_in": 0.0, "mask_grad_out": 0.0, "mask_gradient_ratio": 1.0, "mask_gradient_status": "UNRELIABLE",
                    "box_var_in": 0.0, "box_var_out": 0.0, "box_laplacian_ratio": 1.0, "box_laplacian_status": "UNRELIABLE",
                    "var_input": 0.0, "var_output": 0.0, "ratio": 1.0, "status": "UNRELIABLE", "reason": "dimension_below_3px",
                })
                continue

            # Reconstruct exact applied mask
            use_ellipse = (cls_name == "face") and (min(roi_w, roi_h) >= r_cfg.min_ellipse_side_px)
            if use_ellipse:
                c_roi_x = int(round(cx - x1_pad))
                c_roi_y = int(round(cy - y1_pad))
                ax_w = max(1, int(round(bw * 0.5 + pad_w)))
                ax_h = max(1, int(round(bh * 0.5 + pad_h)))
                mask = _create_feathered_mask(
                    roi_h, roi_w, "ellipse", feather_sigma=r_cfg.feather_sigma_px, center=(c_roi_x, c_roi_y), axes=(ax_w, ax_h)
                )
            else:
                feather = r_cfg.feather_sigma_px if (cls_name == "face" or r_cfg.plate_mask == "feathered_rect") else 0.0
                mask = _create_feathered_mask(roi_h, roi_w, "rect", feather_sigma=feather)

            roi_in = gray_in[y1_pad:y2_pad, x1_pad:x2_pad]
            roi_out = gray_out[y1_pad:y2_pad, x1_pad:x2_pad]

            # 1. Laplacian computation
            lap_in = cv2.Laplacian(roi_in, cv2.CV_64F)
            lap_out = cv2.Laplacian(roi_out, cv2.CV_64F)

            # 2. Sobel Gradient Magnitude (High-Frequency Energy)
            gx_in = cv2.Sobel(roi_in, cv2.CV_64F, 1, 0, ksize=3)
            gy_in = cv2.Sobel(roi_in, cv2.CV_64F, 0, 1, ksize=3)
            mag_in = np.sqrt(gx_in**2 + gy_in**2)

            gx_out = cv2.Sobel(roi_out, cv2.CV_64F, 1, 0, ksize=3)
            gy_out = cv2.Sobel(roi_out, cv2.CV_64F, 0, 1, ksize=3)
            mag_out = np.sqrt(gx_out**2 + gy_out**2)

            # Full bounding box values
            box_var_in = float(lap_in.var())
            box_var_out = float(lap_out.var())
            box_lap_ratio = float(box_var_out / box_var_in) if box_var_in >= 1.0 else 0.0

            # Mask Core (alpha > 0.9)
            active_core = (mask > 0.9)
            core_px_count = int(np.sum(active_core))

            if core_px_count >= 4:
                mask_var_in = float(np.var(lap_in[active_core]))
                mask_var_out = float(np.var(lap_out[active_core]))
                mask_lap_ratio = float(mask_var_out / mask_var_in) if mask_var_in >= 1.0 else 0.0

                mask_grad_in = float(np.mean(mag_in[active_core]))
                mask_grad_out = float(np.mean(mag_out[active_core]))
                mask_grad_ratio = float(mask_grad_out / mask_grad_in) if mask_grad_in >= 1e-3 else 0.0
            else:
                mask_var_in = box_var_in
                mask_var_out = box_var_out
                mask_lap_ratio = box_lap_ratio
                mask_grad_in = float(np.mean(mag_in))
                mask_grad_out = float(np.mean(mag_out))
                mask_grad_ratio = float(mask_grad_out / mask_grad_in) if mask_grad_in >= 1e-3 else 0.0

            # Determine reliability honestly (do not fake var_in = 0.0)
            if min(bw, bh) < min_side_reliable or side < min_side_reliable:
                status = "UNRELIABLE"
                reason = "dimension_below_8px_threshold"
            elif core_px_count < 4:
                status = "UNRELIABLE"
                reason = "insufficient_core_mask_pixels"
            elif mask_var_in < 1.0:
                status = "UNRELIABLE"
                reason = "flat_input_texture"
            else:
                if mask_lap_ratio <= max_ratio_threshold:
                    status = "PASS"
                    reason = "ratio_within_threshold"
                else:
                    status = "FAIL"
                    reason = f"ratio_{mask_lap_ratio:.3f}_exceeds_{max_ratio_threshold}"

            # Per-metric pass/fail statuses
            if status == "UNRELIABLE":
                mask_lap_st = "UNRELIABLE"
                mask_grad_st = "UNRELIABLE"
                box_lap_st = "UNRELIABLE"
            else:
                mask_lap_st = "PASS" if mask_lap_ratio <= max_ratio_threshold else "FAIL"
                mask_grad_st = "PASS" if mask_grad_ratio <= max_ratio_threshold else "FAIL"
                box_lap_st = "PASS" if box_lap_ratio <= max_ratio_threshold else "FAIL"

            row = {
                "frame": filename,
                "box_id": box_id,
                "class": cls_name,
                "x1": round(x1_raw, 2),
                "y1": round(y1_raw, 2),
                "x2": round(x2_raw, 2),
                "y2": round(y2_raw, 2),
                "box_w": round(bw, 2),
                "box_h": round(bh, 2),
                "side_px": round(side, 2),
                "size_bucket": size_bucket,
                "confidence": round(conf, 4),
                # Core Mask Metrics (Primary)
                "mask_var_in": round(mask_var_in, 2),
                "mask_var_out": round(mask_var_out, 2),
                "mask_laplacian_ratio": round(mask_lap_ratio, 4),
                "mask_laplacian_status": mask_lap_st,
                "mask_grad_in": round(mask_grad_in, 2),
                "mask_grad_out": round(mask_grad_out, 2),
                "mask_gradient_ratio": round(mask_grad_ratio, 4),
                "mask_gradient_status": mask_grad_st,
                # Full Bounding Box Metrics (Secondary / Comparison)
                "box_var_in": round(box_var_in, 2),
                "box_var_out": round(box_var_out, 2),
                "box_laplacian_ratio": round(box_lap_ratio, 4),
                "box_laplacian_status": box_lap_st,
                # Compatibility fields for legacy consumers
                "var_input": round(mask_var_in, 2),
                "var_output": round(mask_var_out, 2),
                "ratio": round(mask_lap_ratio, 4),
                "status": status,
                "reason": reason,
            }
            all_audit_rows.append(row)

    if out_csv_path is not None:
        out_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
            if all_audit_rows:
                writer = csv.DictWriter(f, fieldnames=list(all_audit_rows[0].keys()))
                writer.writeheader()
                writer.writerows(all_audit_rows)
            else:
                writer = csv.writer(f)
                writer.writerow(["frame", "box_id", "class", "status", "ratio"])

    # Summarize all three metrics
    mask_lap_summary = _summarize_metric(
        all_audit_rows, "mask_laplacian_ratio", "mask_laplacian_status", max_ratio_threshold, total_boxes
    )
    mask_grad_summary = _summarize_metric(
        all_audit_rows, "mask_gradient_ratio", "mask_gradient_status", max_ratio_threshold, total_boxes
    )
    box_lap_summary = _summarize_metric(
        all_audit_rows, "box_laplacian_ratio", "box_laplacian_status", max_ratio_threshold, total_boxes
    )

    if total_boxes == 0:
        primary_pass = False
        primary_status = "NOT_EXECUTED"
    else:
        primary_pass = (mask_lap_summary["boxes_failed"] == 0 and mask_lap_summary["boxes_checked"] > 0)
        primary_status = "PASS" if primary_pass else "FAIL"

    return {
        "status": primary_status,
        "sample_size": len(sampled_records),
        "boxes_total": total_boxes,
        "boxes_checked": mask_lap_summary["boxes_checked"],
        "boxes_passed": mask_lap_summary["boxes_passed"],
        "boxes_failed": mask_lap_summary["boxes_failed"],
        "boxes_unreliable": mask_lap_summary["boxes_unreliable"],
        "unreliable_fraction": mask_lap_summary["unreliable_fraction"],
        "ratio_median": mask_lap_summary["ratio_median"],
        "ratio_p95": mask_lap_summary["ratio_p95"],
        "ratio_max": mask_lap_summary["ratio_max"],
        "max_ratio_threshold": max_ratio_threshold,
        "PASS": primary_pass,
        # Redesigned dual-metric blocks + legacy comparison
        "laplacian_check_mask_core": mask_lap_summary,
        "gradient_energy_check_mask_core": mask_grad_summary,
        "laplacian_check_full_box": box_lap_summary,
        "all_rows": all_audit_rows,
    }


def run_pixel_integrity_test(
    input_dir: Path,
    output_dir: Path,
    sampled_records: list[Any],
    feather_sigma_px: float = 2.0,
    max_test_frames: int = 50,
) -> dict[str, Any]:
    """Computes exact per-pixel differences outside masks dilated by (feather + 8 px)."""
    dilation_margin = int(math.ceil(feather_sigma_px + 8.0))
    test_recs = sampled_records[:max_test_frames]
    if not test_recs:
        return {
            "status": "NOT_EXECUTED",
            "reason": "no frames provided for pixel integrity test",
            "frames_tested": 0,
            "input_format": "JPEG (JFIF lossy source)",
            "output_format": "JPEG (quality 95 lossy export)",
            "max_abs_diff_levels": 0,
            "mean_abs_diff_levels": 0.0,
            "pct_unmasked_pixels_altered_by_compression": 0.0,
            "dilation_margin_px": dilation_margin,
            "findings_statement": "Pixel integrity test not executed (no frames provided).",
        }

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_margin * 2 + 1, dilation_margin * 2 + 1))

    max_diffs: list[int] = []
    mean_diffs: list[float] = []
    pct_changed_list: list[float] = []
    frames_tested = 0

    for r in test_recs:
        filename = getattr(r, "filename", "")
        in_p = input_dir / filename
        out_p = output_dir / filename
        if not in_p.exists() or not out_p.exists():
            continue

        in_img = cv2.imread(str(in_p))
        out_img = cv2.imread(str(out_p))
        if in_img is None or out_img is None or in_img.shape != out_img.shape:
            continue

        h, w = in_img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Draw detected face ellipses
        for b in getattr(r, "face_boxes", []):
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)
            cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 255, -1)

        # Draw plate rects
        for b in getattr(r, "plate_boxes", []):
            x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Dilate mask by feather + 8 px
        dilated_mask = cv2.dilate(mask, kernel)
        unmasked = (dilated_mask == 0)

        unmasked_count = int(unmasked.sum())
        if unmasked_count == 0:
            continue

        diff = np.abs(in_img.astype(np.int16) - out_img.astype(np.int16))
        unmasked_diff = diff[unmasked]  # (N, 3)

        frame_max_diff = int(unmasked_diff.max())
        frame_mean_diff = float(unmasked_diff.mean())
        pixels_changed = int((unmasked_diff > 0).any(axis=1).sum())
        pct_changed = round((pixels_changed / unmasked_count) * 100.0, 2)

        max_diffs.append(frame_max_diff)
        mean_diffs.append(frame_mean_diff)
        pct_changed_list.append(pct_changed)
        frames_tested += 1

    if frames_tested == 0:
        return {
            "status": "NOT_EXECUTED",
            "reason": "0 valid frames evaluated on disk",
            "frames_tested": 0,
            "input_format": "JPEG (JFIF lossy source)",
            "output_format": "JPEG (quality 95 lossy export)",
            "max_abs_diff_levels": 0,
            "mean_abs_diff_levels": 0.0,
            "pct_unmasked_pixels_altered_by_compression": 0.0,
            "dilation_margin_px": dilation_margin,
            "findings_statement": "Pixel integrity test not executed (0 frames on disk).",
        }

    overall_max_diff = max(max_diffs)
    overall_mean_diff = round(float(np.mean(mean_diffs)), 3)
    overall_pct_changed = round(float(np.mean(pct_changed_list)), 2)

    return {
        "status": "COMPLETED",
        "frames_tested": frames_tested,
        "input_format": "JPEG (JFIF lossy source)",
        "output_format": "JPEG (quality 95 lossy export)",
        "max_abs_diff_levels": overall_max_diff,
        "mean_abs_diff_levels": overall_mean_diff,
        "pct_unmasked_pixels_altered_by_compression": overall_pct_changed,
        "dilation_margin_px": dilation_margin,
        "findings_statement": (
            f"Input images are lossy JPEG. Re-encoding at quality 95 constitutes a second lossy encode. "
            f"Outside the dilated mask (+{dilation_margin}px), pixels are altered by JPEG quantization "
            f"with max abs diff {overall_max_diff} intensity levels (mean diff {overall_mean_diff}). "
            f"Zero double compression and bit-identical claims do not hold for JPEG inputs."
        ),
    }


def scan_metadata_sanitization(output_dir: Path) -> dict[str, Any]:
    """Scans all output images in output_dir for EXIF, XMP, ICC, COMMENTS, and GPS markers."""
    out_files = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.jpeg"))
    if not out_files:
        # Check subdirectories
        out_files = list(output_dir.rglob("*.jpg")) + list(output_dir.rglob("*.jpeg"))
        # Exclude report directory
        out_files = [f for f in out_files if "privacy_report" not in f.parts and "baseline" not in f.parts]

    total_scanned = len(out_files)
    exif_count = 0
    xmp_count = 0
    icc_count = 0
    gps_count = 0
    comment_count = 0

    for p in out_files:
        try:
            with open(p, "rb") as f:
                data = f.read(65536)  # Read first 64KB containing JPEG headers
            if b"Exif" in data:
                exif_count += 1
            if b"http://ns.adobe.com/xap/1.0/" in data or b"<x:xmpmeta" in data:
                xmp_count += 1
            if b"ICC_PROFILE" in data:
                icc_count += 1
            if b"GPS" in data:
                gps_count += 1
            # Check COM marker (0xFF, 0xFE)
            if b"\xff\xfe" in data:
                comment_count += 1
        except Exception:
            pass

    return {
        "status": "COMPLETED",
        "total_output_images_scanned": total_scanned,
        "metadata_segments_found": {
            "exif": exif_count,
            "xmp": xmp_count,
            "icc": icc_count,
            "gps": gps_count,
            "comments": comment_count,
        },
        "all_sanitized": (exif_count == 0 and xmp_count == 0 and gps_count == 0 and comment_count == 0),
        "notes": (
            "OpenCV cv2.imencode outputs standard JFIF streams omitting EXIF, XMP, GPS, and user comment tags. "
            "Output filenames retain nanosecond epoch timestamps (e.g. 1788953644458122.jpg) from input files."
        ),
    }
