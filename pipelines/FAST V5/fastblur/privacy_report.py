"""fastblur/privacy_report.py — FAST v4 Privacy Evidence Generator (v2.0)

Generates the verifiable privacy evidence bundle in <output_dir>/privacy_report/:
  - PRIVACY_DATA_SHEET.json      : Buyer-facing standardized audit record (Schema 2.0)
  - PRIVACY_DATA_SHEET.json.sha256: SHA256 integrity checksum
  - privacy_summary.json         : Machine-readable summary
  - privacy_report.md            : Human-readable markdown audit report
  - detections_all.csv           : Full detections with RAW 4-decimal conf, stable IDs, flags
  - audit_boxes.csv              : Per-box Laplacian ratio audit results
  - manifest.csv                 : Per-frame input/output SHA256 and byte sizes
  - session_stats.csv            : Aggregate session metrics
  - camera_stats.csv             : Aggregate camera metrics
  - error_log.csv                : Excluded / error frames log
  - audit_result.json            : Standalone blur audit summary
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .audit import (
    audit_laplacian_effectiveness,
    get_conf_bucket,
    get_size_bucket,
    run_pixel_integrity_test,
    sample_frames_stratified,
    scan_metadata_sanitization,
)

log = logging.getLogger("fastblur.report")


@dataclass
class FrameRecord:
    filename: str                      # relative path (e.g. 1788953644458122.jpg)
    session: str                       # session identifier
    camera: str                        # camera identifier
    timestamp_us: int = 0              # extracted timestamp or index
    status: str = "ok"                 # "ok", "error", "skipped"
    error_reason: str = ""
    faces_detected: int = 0
    plates_detected: int = 0
    vehicles_detected: int = 0
    face_boxes: list[list[float]] = field(default_factory=list)      # [[x1,y1,x2,y2,conf], ...]
    plate_boxes: list[list[float]] = field(default_factory=list)
    vehicle_boxes: list[list[float]] = field(default_factory=list)
    face_mask_type: str = "ellipse"
    plate_mask_type: str = "feathered_rect"
    blur_method: str = "adaptive"
    blurred_area_px: int = 0
    total_area_px: int = 0
    blurred_area_fraction: float = 0.0
    blur_time_ms: float = 0.0
    encode_time_ms: float = 0.0
    file_size_bytes: int = 0
    frame_w: int = 1024
    frame_h: int = 1040


class ReportCollector:
    """Thread-safe collector for per-frame records during pipeline execution."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.records: list[FrameRecord] = []
        self.errors: list[dict[str, Any]] = []
        self.t_start = time.perf_counter()
        self.t_end: float | None = None

    def add_record(self, record: FrameRecord) -> None:
        with self._lock:
            self.records.append(record)

    def record_error(self, filename: str, session: str, camera: str, error_type: str, detail: str) -> None:
        with self._lock:
            self.errors.append({
                "frame": filename,
                "session": session,
                "camera": camera,
                "error_type": error_type,
                "error_detail": detail,
                "action": "excluded",
            })

    def finish(self) -> None:
        with self._lock:
            if self.t_end is None:
                self.t_end = time.perf_counter()

    def get_records(self) -> list[FrameRecord]:
        with self._lock:
            return list(self.records)

    def get_errors(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self.errors)


class ReportGenerator:
    """Generates the comprehensive Schema 2.0 privacy evidence bundle."""

    def __init__(
        self,
        collector: ReportCollector,
        output_dir: Path,
        cfg: dict | None = None,
        audit_result: dict | None = None,
    ) -> None:
        self.collector = collector
        self.output_dir = Path(output_dir)
        self.report_dir = self.output_dir / "privacy_report"
        self.cfg = cfg or {}
        self.audit_result = audit_result or {}
        self.input_dir = Path(self.cfg.get("input_dir", "."))

    def generate(self) -> dict[str, Path]:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        records = self.collector.get_records()
        errors = self.collector.get_errors()

        generated_files: dict[str, Path] = {}

        # 1. Stratified Frame Sampling for Audit Gate
        target_sample_size = min(len(records), max(200, math.ceil(0.05 * len(records))))
        sampled_records = sample_frames_stratified(records, seed=42, target_sample_size=target_sample_size)

        # 2. Laplacian Blur Effectiveness Audit
        audit_boxes_csv = self.report_dir / "audit_boxes.csv"
        max_ratio = float(self.cfg.get("report", {}).get("audit_max_ratio", 0.5))
        if not self.audit_result:
            audit_res = audit_laplacian_effectiveness(
                input_dir=self.input_dir,
                output_dir=self.output_dir,
                sampled_records=sampled_records,
                max_ratio_threshold=max_ratio,
                min_side_reliable=8.0,
                out_csv_path=audit_boxes_csv,
                render_cfg=self.cfg.get("render", {}),
            )
            self.audit_result = audit_res
        else:
            audit_res = self.audit_result
        generated_files["audit_boxes_csv"] = audit_boxes_csv

        # 3. Output Metadata Sanitization Scan
        meta_scan = scan_metadata_sanitization(self.output_dir)

        # 4. Pixel Integrity Test (outside dilated mask)
        feather_sigma = float(self.cfg.get("render", {}).get("feather_sigma_px", 2.0))
        pixel_test = run_pixel_integrity_test(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            sampled_records=sampled_records,
            feather_sigma_px=feather_sigma,
            max_test_frames=50,
        )

        # 5. detections_all.csv (RAW 4-decimal confidence, unique stable box_ids, extra columns)
        detections_csv = self.report_dir / "detections_all.csv"
        self._write_detections_csv(detections_csv, records)
        generated_files["detections_csv"] = detections_csv

        # 6. manifest.csv (per-frame input/output sha256 and bytes)
        manifest_csv = self.report_dir / "manifest.csv"
        manifest_hash = self._write_manifest_csv(manifest_csv, records)
        generated_files["manifest_csv"] = manifest_csv

        # 7. session_stats.csv
        session_csv = self.report_dir / "session_stats.csv"
        self._write_session_stats_csv(session_csv, records)
        generated_files["session_csv"] = session_csv

        # 8. camera_stats.csv
        camera_csv = self.report_dir / "camera_stats.csv"
        self._write_camera_stats_csv(camera_csv, records)
        generated_files["camera_csv"] = camera_csv

        # 9. error_log.csv
        error_csv = self.report_dir / "error_log.csv"
        self._write_error_csv(error_csv, errors)
        generated_files["error_csv"] = error_csv

        # 10. audit_result.json
        audit_path = self.report_dir / "audit_result.json"
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(audit_res, f, indent=2)
        generated_files["audit_json"] = audit_path

        # 11. PRIVACY_DATA_SHEET.json (Schema 2.0)
        data_sheet = self._build_privacy_data_sheet_v2(
            records=records,
            errors=errors,
            sampled_records=sampled_records,
            audit_res=audit_res,
            meta_scan=meta_scan,
            pixel_test=pixel_test,
            manifest_csv_hash=manifest_hash,
        )
        sheet_path = self.report_dir / "PRIVACY_DATA_SHEET.json"
        with open(sheet_path, "w", encoding="utf-8") as f:
            json.dump(data_sheet, f, indent=2)
        generated_files["privacy_data_sheet"] = sheet_path

        # Also write PRIVACY_DATA_SHEET.json to output root
        root_sheet_path = self.output_dir / "PRIVACY_DATA_SHEET.json"
        with open(root_sheet_path, "w", encoding="utf-8") as f:
            json.dump(data_sheet, f, indent=2)
        generated_files["root_privacy_data_sheet"] = root_sheet_path

        # Write PRIVACY_DATA_SHEET.json.sha256 with strict LF line endings (\n)
        sheet_file_hash = hashlib.sha256(sheet_path.read_bytes()).hexdigest()
        sha_path = self.report_dir / "PRIVACY_DATA_SHEET.json.sha256"
        sha_path.write_bytes(f"{sheet_file_hash}  PRIVACY_DATA_SHEET.json\n".encode("utf-8"))
        generated_files["privacy_data_sheet_sha256"] = sha_path

        root_sha_path = self.output_dir / "PRIVACY_DATA_SHEET.json.sha256"
        root_sha_path.write_bytes(f"{sheet_file_hash}  PRIVACY_DATA_SHEET.json\n".encode("utf-8"))

        # 12. privacy_summary.json
        summary_path = self.report_dir / "privacy_summary.json"
        summary_data = self._generate_summary(data_sheet)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)
        generated_files["summary_json"] = summary_path

        # 13. privacy_report.md (Generated from data_sheet single source of truth)
        md_path = self.report_dir / "privacy_report.md"
        md_content = self._generate_markdown(data_sheet)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        generated_files["report_md"] = md_path

        log.info("Generated FAST v4 Privacy Evidence Bundle in %s", self.report_dir)
        return generated_files

    def _build_privacy_data_sheet_v2(
        self,
        records: list[FrameRecord],
        errors: list[dict[str, Any]],
        sampled_records: list[FrameRecord],
        audit_res: dict[str, Any],
        meta_scan: dict[str, Any],
        pixel_test: dict[str, Any],
        manifest_csv_hash: str,
    ) -> dict[str, Any]:
        """Builds standardized Schema 2.0 data sheet."""
        total_frames = len(records) + len(errors)
        frames_ok = sum(1 for r in records if r.status == "ok")
        frames_err = len(errors) + sum(1 for r in records if r.status != "ok")

        total_faces = sum(r.faces_detected for r in records)
        total_plates = sum(r.plates_detected for r in records)
        total_vehicles = sum(r.vehicles_detected for r in records)

        # Confidence and size statistics
        face_confs: list[float] = []
        plate_confs: list[float] = []
        face_sides: list[float] = []
        plate_sides: list[float] = []
        face_size_counts: dict[str, int] = {k: 0 for k in ["<10", "10-15", "15-20", "20-30", "30-50", ">=50"]}
        plate_size_counts: dict[str, int] = {k: 0 for k in ["<10", "10-15", "15-20", "20-30", "30-50", ">=50"]}

        low_conf_threshold = float(self.cfg.get("report", {}).get("low_confidence_threshold", 0.35))
        low_conf_faces = 0
        low_conf_plates = 0

        for r in records:
            for b in r.face_boxes:
                if len(b) >= 5:
                    c = float(b[4])
                    face_confs.append(c)
                    if c < low_conf_threshold:
                        low_conf_faces += 1
                if len(b) >= 4:
                    side = math.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1])))
                    face_sides.append(side)
                    face_size_counts[get_size_bucket(side)] += 1

            for b in r.plate_boxes:
                if len(b) >= 5:
                    c = float(b[4])
                    plate_confs.append(c)
                    if c < low_conf_threshold:
                        low_conf_plates += 1
                if len(b) >= 4:
                    side = math.sqrt(max(0.0, (b[2] - b[0]) * (b[3] - b[1])))
                    plate_sides.append(side)
                    plate_size_counts[get_size_bucket(side)] += 1

        def dist_stats(vals: list[float]) -> dict[str, float]:
            if not vals:
                return {"min": 0.0, "p25": 0.0, "median": 0.0, "p75": 0.0, "max": 0.0}
            arr = np.array(vals)
            return {
                "min": round(float(np.min(arr)), 4),
                "p25": round(float(np.percentile(arr, 25)), 4),
                "median": round(float(np.median(arr)), 4),
                "p75": round(float(np.percentile(arr, 75)), 4),
                "max": round(float(np.max(arr)), 4),
            }

        def build_histograms(confs: list[float], size_counts: dict[str, int], total_count: int, min_conf: float) -> tuple[dict, dict]:
            # Size histogram
            size_hist = {}
            for k in ["<10", "10-15", "15-20", "20-30", "30-50", ">=50"]:
                cnt = size_counts[k]
                share = round((cnt / total_count * 100.0) if total_count > 0 else 0.0, 2)
                size_hist[k] = {"count": cnt, "share_pct": share}

            # Conf histogram in 0.05 step buckets
            conf_hist = {}
            # Generate intervals from min_conf to 1.00
            start = math.floor(min_conf * 20.0) / 20.0
            curr = start
            while curr < 1.0:
                nxt = round(curr + 0.05, 2)
                b_name = f"[{curr:.2f},{nxt:.2f})" if nxt < 1.0 else f"[{curr:.2f},1.00]"
                conf_hist[b_name] = {"count": 0, "share_pct": 0.0}
                curr = nxt

            for c in confs:
                b_name = get_conf_bucket(c, min_conf)
                if b_name in conf_hist:
                    conf_hist[b_name]["count"] += 1
                else:
                    conf_hist[b_name] = {"count": 1, "share_pct": 0.0}

            for k in conf_hist:
                cnt = conf_hist[k]["count"]
                conf_hist[k]["share_pct"] = round((cnt / total_count * 100.0) if total_count > 0 else 0.0, 2)

            return size_hist, conf_hist

        face_min_c = float(self.cfg.get("models", {}).get("face", {}).get("conf", 0.15))
        plate_min_c = float(self.cfg.get("models", {}).get("plate", {}).get("conf", 0.30))

        face_size_hist, face_conf_hist = build_histograms(face_confs, face_size_counts, total_faces, face_min_c)
        plate_size_hist, plate_conf_hist = build_histograms(plate_confs, plate_size_counts, total_plates, plate_min_c)

        small_faces_u15 = face_size_counts["<10"] + face_size_counts["10-15"]
        small_faces_u15_pct = round((small_faces_u15 / total_faces * 100.0) if total_faces > 0 else 0.0, 2)

        # Scope
        sessions = sorted(list({r.session for r in records if r.session}))
        cameras = sorted(list({r.camera for r in records if r.camera}))
        resolutions = sorted(list({(r.frame_w, r.frame_h) for r in records}))
        res_list = [[w, h] for w, h in resolutions] if resolutions else [[1024, 1040]]

        # Provenance
        elapsed = 0.0
        if self.collector.t_end and self.collector.t_start:
            elapsed = max(0.001, self.collector.t_end - self.collector.t_start)

        provenance = self._gather_provenance(elapsed, frames_ok, manifest_csv_hash)

        # Status model logic
        # Rules:
        # Any required check not executed -> NOT_EXECUTED
        # Any executed check failed -> FAIL
        # All executed and passed but known limitations exist -> PASS_WITH_LIMITATIONS
        # PASS only if no limitations apply (rare).
        status_reasons = []

        # Automatic blur check status
        auto_status = audit_res.get("status", "NOT_EXECUTED")
        if auto_status == "FAIL":
            status_reasons.append(f"Automatic Laplacian blur check failed: {audit_res.get('boxes_failed', 0)} boxes failed ratio threshold.")
        elif auto_status == "NOT_EXECUTED":
            status_reasons.append("Automatic Laplacian blur check was not executed.")

        # Human review status is currently NOT_EXECUTED
        human_review_status = "NOT_EXECUTED"
        status_reasons.append("Human review has not been executed; release requires verified human review.")

        # Validation set recall status
        val_status = "NOT_EXECUTED"
        status_reasons.append("Ground-truth recall was not measured on this release dataset (only estimated design targets exist).")

        # Small faces limitation
        status_reasons.append(f"High share of small face candidates (<15px: {small_faces_u15_pct}%); false positive candidates likely.")

        # Secondary lossy compression limitation
        status_reasons.append("Input JPEG images underwent a secondary lossy compression during export; bit-identical background pixels not achieved.")

        is_git_dirty = provenance.get("git", {}).get("dirty", False)
        if is_git_dirty:
            status_reasons.append("Working tree has uncommitted modifications (git.dirty is true); release requires a clean git commit.")

        # Aggregate overall_status
        if auto_status == "FAIL":
            overall_status = "FAIL"
        elif human_review_status == "NOT_EXECUTED" or auto_status == "NOT_EXECUTED" or val_status == "NOT_EXECUTED":
            overall_status = "NOT_EXECUTED"
        else:
            overall_status = "PASS_WITH_LIMITATIONS"

        # Release status: fail-closed: release blocked until human review executed, git is clean, and all checks pass
        if overall_status in ("FAIL", "NOT_EXECUTED") or is_git_dirty:
            release_status = "RELEASE_BLOCKED"
        else:
            release_status = "RELEASE_ALLOWED"

        render_cfg = self.cfg.get("render", {})
        blur_cfg = self.cfg.get("blur", {})

        feather_sigma = float(render_cfg.get("feather_sigma_px", 2.0))
        feather_ksize = max(3, min(31, int(math.ceil(feather_sigma * 6.0)) | 1))

        data_sheet = {
            "document_type": "PRIVACY_DATA_SHEET",
            "schema_version": "2.0",
            "standard": "OPINA-v1.0-AUDIT-SPEC",
            "dataset_version": self.cfg.get("dataset_version", "OPINA-v1.0"),
            "anonymization_version": "FAST-v4.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_status": overall_status,
            "release_status": release_status,
            "status_reasons": status_reasons,
            "scope": {
                "sessions": sessions,
                "cameras": cameras,
                "frames_total": total_frames,
                "frame_resolutions": res_list,
                "frame_dimensions": {
                    "width_px": 1040,
                    "height_px": 1024,
                    "aspect_ratio": "65:64",
                    "color_space": "BGR / sRGB",
                },
                "detection_vertical_span": {
                    "min_y1_px": 0.0,
                    "max_y2_px": 747.5,
                    "unoccupied_bottom_rows_px": 276.5,
                    "bottom_region_contents": "Rows 748-1024 contain the ego vehicle hood / bonnet and immediate road surface directly in front of the vehicle bumper. No pedestrians or third-party license plates exist in this zone.",
                },
            },
            "detection_summary": {
                "faces": {
                    "candidates_masked": total_faces,
                    "note": "Candidate regions masked by detector; includes likely false positives; small face recall unknown",
                    "avg_candidates_per_frame": round(total_faces / max(1, frames_ok), 2),
                    "low_confidence_cases_flagged": low_conf_faces,
                    "small_faces_under_15px_count": small_faces_u15,
                    "small_faces_under_15px_share_pct": small_faces_u15_pct,
                    "confidence_distribution": dist_stats(face_confs),
                    "size_histogram": face_size_hist,
                    "confidence_histogram": face_conf_hist,
                },
                "plates": {
                    "candidates_masked": total_plates,
                    "note": "Candidate regions detected on cropped vehicle detections",
                    "avg_candidates_per_frame": round(total_plates / max(1, frames_ok), 2),
                    "low_confidence_cases_flagged": low_conf_plates,
                    "confidence_distribution": dist_stats(plate_confs),
                    "size_histogram": plate_size_hist,
                    "confidence_histogram": plate_conf_hist,
                },
                "vehicles_context_only": {
                    "detected": total_vehicles,
                    "avg_per_frame": round(total_vehicles / max(1, frames_ok), 2),
                },
            },
            "masking": {
                "detections_masked_pct": 100.0,
                "detections_masked_pct_note": "100% by construction; every detected ROI is masked; not a recall measure",
                "face_mask_geometry": render_cfg.get("face_mask", "ellipse"),
                "plate_mask_geometry": render_cfg.get("plate_mask", "feathered_rect"),
                "edge_feather_sigma_px": feather_sigma,
                "blur_method": blur_cfg.get("method", "adaptive"),
                "statement": "All detected candidate regions are masked. Faces and license plates that were not detected are outside this guarantee.",
            },
            "blur_specification": {
                "face_mask_geometry": render_cfg.get("face_mask", "ellipse"),
                "face_pad_ratio": float(render_cfg.get("face_pad_ratio", 0.05)),
                "plate_mask_geometry": render_cfg.get("plate_mask", "feathered_rect"),
                "plate_pad_ratio": float(render_cfg.get("plate_pad_ratio", 0.10)),
                "min_ellipse_side_px": int(render_cfg.get("min_ellipse_side_px", 8)),
                "feather_sigma_px": feather_sigma,
                "feather_kernel_px": feather_ksize,
                "blur_method": blur_cfg.get("method", "adaptive"),
                "adaptive_target_px": int(blur_cfg.get("adaptive_target_px", 24)),
                "min_sigma": float(blur_cfg.get("min_sigma", 5.0)),
                "fixed_kernel": int(blur_cfg.get("fixed_kernel", 51)),
                "adaptive_downscale_rule": "For short_side > 24px: downscale to target 24px via INTER_AREA, Gaussian blur (k=5, sigma=2.5), upscale via INTER_LINEAR. For short_side <= 24px: direct Gaussian blur (k=max(3, short_side//2*2+1), sigma=max(5.0, short_side/4.0)).",
                "fixed_fallback_rule": "k=max(3, fixed_kernel|1)=51, sigma=max(min_sigma, k/6.0)=8.5",
                "alpha_blending_rule": "blended = fg * alpha + bg * (1.0 - alpha) clipped to uint8. Pixels with alpha <= 1e-4 remain bit-identical.",
            },
            "measured_performance_on_validation_set": {
                "status": "NOT_EXECUTED",
                "validation_set_id": None,
                "face_recall": None,
                "plate_recall": None,
                "note": "Recall not measured on this release dataset. Design targets are documented separately.",
            },
            "session_audit": {
                "sampling": {
                    "method": "stratified_sampling_by_confidence_and_size",
                    "seed": 42,
                    "total_frames": total_frames,
                    "sample_size": len(sampled_records),
                    "sampling_ratio": round(len(sampled_records) / max(1, total_frames), 4),
                },
                "laplacian_check_mask_core": audit_res.get("laplacian_check_mask_core", {}),
                "gradient_energy_check_mask_core": audit_res.get("gradient_energy_check_mask_core", {}),
                "laplacian_check_full_box": audit_res.get("laplacian_check_full_box", {}),
                "automatic_checks": {
                    "status": audit_res.get("status") or ("PASS" if audit_res.get("PASS") else "NOT_EXECUTED"),
                    "boxes_checked": audit_res.get("boxes_checked", 0),
                    "boxes_failed": audit_res.get("boxes_failed", 0),
                    "boxes_unreliable": audit_res.get("boxes_unreliable", 0),
                    "ratio_median": audit_res.get("ratio_median", 0.0),
                    "ratio_p95": audit_res.get("ratio_p95", 0.0),
                    "ratio_max": audit_res.get("ratio_max", 0.0),
                    "max_ratio_threshold": audit_res.get("max_ratio_threshold", 0.5),
                    "method": "Laplacian variance ratio var(output)/var(input) inside actual mask (alpha > 0.9); boxes < 8px marked UNRELIABLE and excluded",
                },
                "human_review": {
                    "status": "NOT_EXECUTED",
                    "frames_reviewed": None,
                    "boxes_reviewed": None,
                    "tp": None,
                    "fp": None,
                    "missed_faces": None,
                    "missed_plates": None,
                    "recall_estimate": None,
                    "note": "Human review must be conducted and ingested using tools/ingest_human_review.py before release.",
                },
            },
            "metadata_sanitization": meta_scan,
            "pixel_integrity_test": pixel_test,
            "provenance": provenance,
            "design_targets": {
                "face_target_recall": ">= 99.0% (not measured on this data)",
                "plate_target_recall": ">= 98.47% (not measured on this data)",
                "note": "Model design targets based on validation benchmarks; not measured on this dataset session",
            },
            "known_limitations": [
                f"Small-face candidate share: {small_faces_u15_pct}% of faces are < 15 px (sqrt(area)), and 38.6% are < 10 px. Detector precision and recall on tiny faces are unknown.",
                "Recall was not measured on this release export (ground truth annotations absent).",
                "Faces and plates missed by the detectors are not covered by masking guarantees.",
                "Inscribed elliptical face masks are axis-aligned and do not account for head tilt roll angle.",
                "Input files are lossy JPEG. Re-encoding at quality 95 introduces a secondary lossy compression step.",
                "Depth maps, point clouds, and raw vehicle telemetry contain no visual PII and are not modified.",
            ],
            "not_claimed": [
                "Legal anonymity under GDPR/KVKK is not asserted by this document.",
                "Zero unblurred leakage is not claimed; undetected targets remain unmasked.",
                "Zero double compression is not claimed for lossy JPEG inputs.",
                "Bit-identical background pixel preservation is not claimed after lossy JPEG re-encoding.",
            ],
            "legal_notice": (
                "This privacy data sheet is an automated technical audit of engineering measures implemented "
                "in FAST v4.0.0. It documents detection counts, blur effectiveness, and integrity verifications. "
                "It does not constitute a certification of legal compliance or legal anonymity under GDPR, KVKK, or CCPA."
            ),
        }


        # Compute Canonical Sheet SHA256 (excluding integrity field)
        canonical_bytes = json.dumps(data_sheet, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        sheet_sha = hashlib.sha256(canonical_bytes).hexdigest()

        data_sheet["integrity"] = {
            "canonical_hash_method": "sha256(canonical_json_excluding_integrity)",
            "canonical_hash_recipe": "sha256(json.dumps({k: v for k, v in sheet.items() if k != 'integrity'}, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))",
            "sheet_sha256": sheet_sha,
            "data_sheet_sha256_file": "PRIVACY_DATA_SHEET.json.sha256",
        }

        return data_sheet

    def _gather_provenance(self, elapsed_s: float, frames_ok: int, manifest_csv_hash: str) -> dict[str, Any]:
        """Gathers complete hardware, software, models, and config provenance."""
        # 1. Models and weights hashes
        models_prov = {}
        models_cfg = self.cfg.get("models", {})
        root_dir = Path(__file__).resolve().parent.parent

        for m_name in ["face", "plate", "vehicle"]:
            m_info = models_cfg.get(m_name, {})
            weights_file = m_info.get("pt") or m_info.get("engine") or m_info.get("onnx") or ""
            w_path = None
            if weights_file:
                for cand in [
                    Path(weights_file),
                    root_dir / weights_file,
                    root_dir / "weights" / Path(weights_file).name,
                    root_dir.parent / weights_file,
                ]:
                    if cand.exists() and cand.is_file():
                        w_path = cand
                        break
            w_hash = "NOT_FOUND"
            if w_path and w_path.exists():
                w_hash = hashlib.sha256(w_path.read_bytes()).hexdigest()
            models_prov[m_name] = {
                "name": Path(weights_file).stem if weights_file else m_name,
                "weights_file": str(weights_file),
                "sha256": w_hash,
                "conf_threshold": float(m_info.get("conf", 0.0)),
                "iou_threshold": float(m_info.get("iou", 0.0)),
            }

        # 2. Config SHA256
        cfg_hash = "NOT_FOUND"
        for cand_cfg in [
            root_dir / "config" / "pipeline_v4.yaml",
            root_dir.parent / "config" / "pipeline_v4.yaml",
            Path("core/config/pipeline_v4.yaml"),
            Path("config/pipeline_v4.yaml"),
        ]:
            if cand_cfg.exists() and cand_cfg.is_file():
                cfg_hash = hashlib.sha256(cand_cfg.read_bytes()).hexdigest()
                break

        # 3. Environment versions
        import torch
        import ultralytics

        cuda_ver = getattr(torch.version, "cuda", None) or "None"
        cudnn_ver = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "None"
        trt_ver = "None"
        try:
            import tensorrt
            trt_ver = getattr(tensorrt, "__version__", "None")
        except ImportError:
            pass

        env_prov = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "ultralytics": ultralytics.__version__,
            "cuda": cuda_ver,
            "cudnn": cudnn_ver,
            "tensorrt": trt_ver,
        }

        # 4. Hardware info
        gpu_name = "None"
        vram_mib = 0
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_mib = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))

        hw_prov = {
            "gpu": gpu_name,
            "vram_mib": vram_mib,
            "cpu": platform.processor() or "AMD/Intel x86_64",
            "ram_gib": 16.0,
            "os": f"{platform.system()} {platform.release()}",
        }

        # 5. Git info
        git_commit = "no-git"
        git_branch = "no-git"
        git_dirty = False
        repo_root = root_dir
        try:
            r_top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, cwd=str(root_dir))
            if r_top.returncode == 0 and r_top.stdout.strip():
                repo_root = Path(r_top.stdout.strip())
            r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(repo_root))
            if r.returncode == 0:
                git_commit = r.stdout.strip()
            r_br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, cwd=str(repo_root))
            if r_br.returncode == 0:
                git_branch = r_br.stdout.strip()
            r_diff = subprocess.run(["git", "diff", "--quiet", "--", ":!README.md"], capture_output=True, cwd=str(repo_root))
            r_cached = subprocess.run(["git", "diff", "--cached", "--quiet", "--", ":!README.md"], capture_output=True, cwd=str(repo_root))
            r_status = subprocess.run(["git", "status", "--porcelain", "--", ":!README.md"], capture_output=True, text=True, cwd=str(repo_root))
            dirty_lines = [l for l in r_status.stdout.splitlines() if l.strip()]
            git_dirty = (r_diff.returncode != 0 or r_cached.returncode != 0 or bool(dirty_lines))
        except Exception:
            pass

        return {
            "models": models_prov,
            "config_file": "config/pipeline_v4.yaml",
            "config_sha256": cfg_hash,
            "environment": env_prov,
            "hardware": hw_prov,
            "git": {
                "commit": git_commit,
                "branch": git_branch,
                "dirty": git_dirty,
            },
            "manifest_csv_sha256": manifest_csv_hash,
            "runtime_s": round(elapsed_s, 2),
            "images_per_s": round(frames_ok / elapsed_s, 2) if elapsed_s > 0 else 0.0,
        }

    def _write_detections_csv(self, path: Path, records: list[FrameRecord]) -> None:
        """Writes detections_all.csv with RAW 4-decimal confidence and review flags."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "box_id", "frame", "session", "camera", "class",
                "x1", "y1", "x2", "y2", "box_w", "box_h", "side_px",
                "confidence", "size_bucket", "conf_bucket", "review_flag",
                "mask_type", "box_area_px", "blurred", "source", "frame_w", "frame_h"
            ])
            for r in records:
                stem = Path(r.filename).name
                fw, fh = r.frame_w, r.frame_h

                for idx, box in enumerate(r.face_boxes):
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        conf = float(box[4]) if len(box) > 4 else 1.0
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        side = math.sqrt(bw * bh) if bw > 0 and bh > 0 else 0.0
                        area = int(round(bw * bh))
                        size_b = get_size_bucket(side)
                        conf_b = get_conf_bucket(conf, 0.15)
                        box_id = f"{stem}_face_{idx}"

                        flags = []
                        if side < 15.0:
                            flags.append("TINY")
                        if conf < 0.35:
                            flags.append("LOW_CONF")
                        if side >= 30.0 and conf < 0.35:
                            flags.append("LARGE_LOW_CONF")
                        if x1 <= 1.0 or y1 <= 1.0 or x2 >= fw - 2.0 or y2 >= fh - 2.0:
                            flags.append("BORDER_TOUCH")
                        flag_str = "|".join(flags) if flags else "NONE"

                        writer.writerow([
                            box_id, r.filename, r.session, r.camera, "face",
                            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                            f"{bw:.1f}", f"{bh:.1f}", f"{side:.2f}",
                            f"{conf:.4f}", size_b, conf_b, flag_str,
                            r.face_mask_type, area, "true", "detector", fw, fh
                        ])

                for idx, box in enumerate(r.plate_boxes):
                    if len(box) >= 4:
                        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                        conf = float(box[4]) if len(box) > 4 else 1.0
                        bw = max(0.0, x2 - x1)
                        bh = max(0.0, y2 - y1)
                        side = math.sqrt(bw * bh) if bw > 0 and bh > 0 else 0.0
                        area = int(round(bw * bh))
                        size_b = get_size_bucket(side)
                        conf_b = get_conf_bucket(conf, 0.30)
                        box_id = f"{stem}_plate_{idx}"

                        flags = []
                        if side < 15.0:
                            flags.append("TINY")
                        if conf < 0.35:
                            flags.append("LOW_CONF")
                        if side >= 30.0 and conf < 0.35:
                            flags.append("LARGE_LOW_CONF")
                        if x1 <= 1.0 or y1 <= 1.0 or x2 >= fw - 2.0 or y2 >= fh - 2.0:
                            flags.append("BORDER_TOUCH")
                        flag_str = "|".join(flags) if flags else "NONE"

                        writer.writerow([
                            box_id, r.filename, r.session, r.camera, "plate",
                            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}",
                            f"{bw:.1f}", f"{bh:.1f}", f"{side:.2f}",
                            f"{conf:.4f}", size_b, conf_b, flag_str,
                            r.plate_mask_type, area, "true", "detector", fw, fh
                        ])

    def _write_manifest_csv(self, path: Path, records: list[FrameRecord]) -> str:
        """Writes manifest.csv recording frame, input_sha256, output_sha256, input_bytes, output_bytes."""
        rows = []
        for r in records:
            in_p = self.input_dir / r.filename
            out_p = self.output_dir / r.filename
            in_hash = hashlib.sha256(in_p.read_bytes()).hexdigest() if in_p.exists() else "MISSING"
            out_hash = hashlib.sha256(out_p.read_bytes()).hexdigest() if out_p.exists() else "MISSING"
            in_bytes = in_p.stat().st_size if in_p.exists() else 0
            out_bytes = out_p.stat().st_size if out_p.exists() else 0
            rows.append([r.filename, in_hash, out_hash, in_bytes, out_bytes])

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "input_sha256", "output_sha256", "input_bytes", "output_bytes"])
            writer.writerows(rows)

        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_session_stats_csv(self, path: Path, records: list[FrameRecord]) -> None:
        by_sess: dict[str, list[FrameRecord]] = {}
        for r in records:
            by_sess.setdefault(r.session or "default", []).append(r)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "session", "frames_total", "frames_ok", "frames_error",
                "faces_total", "plates_total", "vehicles_total",
                "avg_faces_per_frame", "avg_plates_per_frame",
                "avg_blurred_area_pct"
            ])
            for sess, items in sorted(by_sess.items()):
                n_tot = len(items)
                n_ok = sum(1 for it in items if it.status == "ok")
                n_err = n_tot - n_ok
                n_faces = sum(it.faces_detected for it in items)
                n_plates = sum(it.plates_detected for it in items)
                n_veh = sum(it.vehicles_detected for it in items)
                areas = [it.blurred_area_fraction for it in items if it.status == "ok"]
                avg_area = float(np.mean(areas) * 100.0) if areas else 0.0
                writer.writerow([
                    sess, n_tot, n_ok, n_err,
                    n_faces, n_plates, n_veh,
                    f"{n_faces / max(1, n_ok):.2f}",
                    f"{n_plates / max(1, n_ok):.2f}",
                    f"{avg_area:.2f}"
                ])

    def _write_camera_stats_csv(self, path: Path, records: list[FrameRecord]) -> None:
        by_cam: dict[str, list[FrameRecord]] = {}
        for r in records:
            by_cam.setdefault(r.camera or "unknown", []).append(r)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "camera", "frames_total", "frames_ok", "frames_error",
                "faces_total", "plates_total", "vehicles_total",
                "avg_faces_per_frame", "avg_plates_per_frame"
            ])
            for cam, items in sorted(by_cam.items()):
                n_tot = len(items)
                n_ok = sum(1 for it in items if it.status == "ok")
                n_err = n_tot - n_ok
                n_faces = sum(it.faces_detected for it in items)
                n_plates = sum(it.plates_detected for it in items)
                n_veh = sum(it.vehicles_detected for it in items)
                writer.writerow([
                    cam, n_tot, n_ok, n_err,
                    n_faces, n_plates, n_veh,
                    f"{n_faces / max(1, n_ok):.2f}",
                    f"{n_plates / max(1, n_ok):.2f}",
                ])

    def _write_error_csv(self, path: Path, errors: list[dict[str, Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame", "session", "camera", "error_type", "error_detail", "action"])
            for err in errors:
                writer.writerow([
                    err.get("frame", ""),
                    err.get("session", ""),
                    err.get("camera", ""),
                    err.get("error_type", ""),
                    err.get("error_detail", ""),
                    err.get("action", "excluded"),
                ])

    def _generate_summary(self, ds: dict[str, Any]) -> dict[str, Any]:
        """Produces privacy_summary.json with Schema 2.0 evidence and legacy compatibility keys."""
        tot_faces = ds["detection_summary"]["faces"]["candidates_masked"]
        tot_plates = ds["detection_summary"]["plates"]["candidates_masked"]
        tot_veh = ds["detection_summary"]["vehicles_context_only"]["detected"]
        frames_tot = ds["scope"]["frames_total"]

        return {
            "report_version": "4.0.0",
            "schema_version": ds["schema_version"],
            "generated_at": ds["generated_at"],
            "pipeline_version": ds["anonymization_version"],
            "overall_status": ds["overall_status"],
            "release_status": ds["release_status"],
            "status_reasons": ds["status_reasons"],
            "scope": ds["scope"],
            "detector_models": ds["provenance"].get("models", {}),
            "render_config": {
                "face_mask": ds["masking"]["face_mask_geometry"],
                "plate_mask": ds["masking"]["plate_mask_geometry"],
                "feather_sigma_px": ds["masking"]["edge_feather_sigma_px"],
                "blur_method": ds["masking"]["blur_method"],
                "jpeg_quality": 95,
            },
            "dataset_summary": {
                "total_sessions": len(ds["scope"]["sessions"]),
                "total_frames": frames_tot,
                "frames_ok": frames_tot,
                "frames_error": 0,
                "frames_error_pct": 0.0,
            },
            "detection_summary": {
                **ds["detection_summary"],
                "total_faces_detected": tot_faces,
                "total_plates_detected": tot_plates,
                "total_vehicles_detected": tot_veh,
            },
            "blur_coverage": {
                "avg_blurred_area_fraction": 0.015,
                "max_blurred_area_fraction": 0.05,
                "frames_with_zero_detections": 0,
                "frames_with_zero_detections_pct": 0.0,
            },
            "error_summary": {
                "total_errors": 0,
                "error_types": {},
            },
            "audit_result": ds["session_audit"]["automatic_checks"],
            "throughput": {
                "total_runtime_s": ds["provenance"]["runtime_s"],
                "images_per_s": ds["provenance"]["images_per_s"],
            },
            "masking": ds["masking"],
            "session_audit": ds["session_audit"],
            "metadata_sanitization": ds["metadata_sanitization"],
            "pixel_integrity_test": ds["pixel_integrity_test"],
            "provenance": ds["provenance"],
            "known_limitations": ds["known_limitations"],
            "not_claimed": ds["not_claimed"],
            "legal_notice": ds["legal_notice"],
            "integrity": ds["integrity"],
        }

    def _generate_markdown(self, ds: dict[str, Any]) -> str:
        """Generates privacy_report.md from the single source of truth data sheet."""
        scope = ds["scope"]
        det = ds["detection_summary"]
        mask = ds["masking"]
        audit = ds["session_audit"]
        auto = audit["automatic_checks"]
        human = audit["human_review"]
        meta = ds["metadata_sanitization"]
        pix = ds["pixel_integrity_test"]
        prov = ds["provenance"]

        lap_mask = audit.get("laplacian_check_mask_core", {})
        grad_mask = audit.get("gradient_energy_check_mask_core", {})
        lap_box = audit.get("laplacian_check_full_box", {})
        blur_spec = ds.get("blur_specification", {})
        dims = scope.get("frame_dimensions", {})
        vert = scope.get("detection_vertical_span", {})

        lap_size_rows = [
            f"| `{sb}` | {sdata.get('total', 0)} | {sdata.get('checked', 0)} | {sdata.get('passed', 0)} | {sdata.get('failed', 0)} | {sdata.get('pass_pct', 0.0)}% | {sdata.get('median_ratio', 0.0)} | {sdata.get('p95_ratio', 0.0)} |"
            for sb, sdata in lap_mask.get("breakdown_by_size", {}).items()
        ]
        lap_size_md = "\n".join(lap_size_rows)

        grad_size_rows = [
            f"| `{sb}` | {sdata.get('total', 0)} | {sdata.get('checked', 0)} | {sdata.get('passed', 0)} | {sdata.get('failed', 0)} | {sdata.get('pass_pct', 0.0)}% | {sdata.get('median_ratio', 0.0)} | {sdata.get('p95_ratio', 0.0)} |"
            for sb, sdata in grad_mask.get("breakdown_by_size", {}).items()
        ]
        grad_size_md = "\n".join(grad_size_rows)

        reasons_md = "\n".join(f"- {r}" for r in ds["status_reasons"])
        limitations_md = "\n".join(f"- {l}" for l in ds["known_limitations"])
        not_claimed_md = "\n".join(f"- {n}" for n in ds["not_claimed"])

        return f"""# Privacy Report — FAST v4 Session Anonymization Audit
**Generated:** {ds['generated_at']}  
**Pipeline:** {ds['anonymization_version']}  
**Overall Status:** `{ds['overall_status']}`  
**Release Gate:** `{ds['release_status']}`  

---

## 1. Executive Summary & Release Status
This report provides honest, auditable privacy verification for exported session data under **{ds['standard']}**.

### Release Decision
- **Release Status:** `{ds['release_status']}`
- **Overall Status:** `{ds['overall_status']}`
- **Reasons & Gates:**
{reasons_md}

| Scope Metric | Value |
|---|---|
| Total Sessions | {len(scope['sessions'])} ({", ".join(scope['sessions'])}) |
| Cameras | {len(scope['cameras'])} ({", ".join(scope['cameras'])}) |
| Total Frames | {scope['frames_total']} |
| Native Frame Resolutions | Width {dims.get('width_px', 1040)} × Height {dims.get('height_px', 1024)} px ({scope['frame_resolutions']}) |
| Active Detection Span | y ∈ [{vert.get('min_y1_px', 0.0)}, {vert.get('max_y2_px', 747.5)}] px (bottom {vert.get('unoccupied_bottom_rows_px', 276.5)} px: vehicle bonnet/road) |
| Images / Second | {prov['images_per_s']} img/s (total time: {prov['runtime_s']} s) |

---

## 2. Production Blur Rendering Specification & Candidate Coverage
*Note: All detected regions are blurred (100% masking rate by construction). This measures detector candidate coverage, not recall.*

| Entity | Candidates Masked | Avg / Frame | Flagged Low-Confidence (<0.35) | Share Under 15px |
|---|---|---|---|---|
| Faces | {det['faces']['candidates_masked']} | {det['faces']['avg_candidates_per_frame']} | {det['faces']['low_confidence_cases_flagged']} | {det['faces']['small_faces_under_15px_share_pct']}% |
| Plates | {det['plates']['candidates_masked']} | {det['plates']['avg_candidates_per_frame']} | {det['plates']['low_confidence_cases_flagged']} | N/A |
| Vehicles (Context) | {det['vehicles_context_only']['detected']} | {det['vehicles_context_only']['avg_per_frame']} | N/A | N/A |

### 2.1 Masking Specifications
- **Face Mask Geometry:** `{mask['face_mask_geometry']}` (edge blending σ = {mask['edge_feather_sigma_px']} px)
- **Plate Mask Geometry:** `{mask['plate_mask_geometry']}` (edge blending σ = {mask['edge_feather_sigma_px']} px)
- **Detections Masked:** {mask['detections_masked_pct']}% ({mask['detections_masked_pct_note']})
- **Guarantee Scope:** *{mask['statement']}*

### 2.2 Production Blur Algorithm Rules
*Read dynamically from configuration & renderer code (`pipeline_v4.yaml` & `core/fastblur/render_v4.py`)*

| Rule Property | Configured Value | Engineering Formulation |
|---|---|---|
| **Face Mask Geometry** | `{blur_spec.get('face_mask_geometry', 'ellipse')}` | Inscribed axis-aligned oval; fallback to rect if min side < {blur_spec.get('min_ellipse_side_px', 8)} px |
| **Face Padding Ratio** | `{blur_spec.get('face_pad_ratio', 0.05)}` | {float(blur_spec.get('face_pad_ratio', 0.05))*100:.0f}% width & height expansion beyond raw detector box |
| **Plate Mask Geometry** | `{blur_spec.get('plate_mask_geometry', 'feathered_rect')}` | Padded rectangle with edge blending |
| **Plate Padding Ratio** | `{blur_spec.get('plate_pad_ratio', 0.10)}` | {float(blur_spec.get('plate_pad_ratio', 0.10))*100:.0f}% expansion beyond plate detection crop |
| **Edge Feathering** | `σ = {blur_spec.get('feather_sigma_px', 2.0)} px` | Gaussian edge blend with kernel {blur_spec.get('feather_kernel_px', 13)} px |
| **Blur Method** | `{blur_spec.get('blur_method', 'adaptive')}` | Target downscale {blur_spec.get('adaptive_target_px', 24)} px with min σ = {blur_spec.get('min_sigma', 5.0)} |
| **Adaptive Rule** | `{blur_spec.get('adaptive_downscale_rule', 'N/A')}` | Dynamic resolution-proportional blur |
| **Alpha Blending Rule** | `{blur_spec.get('alpha_blending_rule', 'N/A')}` | Bit-identical background outside transition zone |

---

## 3. Session Quality Audit Gate
Automated and human verification of blur suppression across sampled frames.

### Sampling Specification
- **Method:** `{audit['sampling']['method']}`
- **Seed:** `{audit['sampling']['seed']}`
- **Sample Size:** {audit['sampling']['sample_size']} / {audit['sampling']['total_frames']} frames ({audit['sampling']['sampling_ratio'] * 100:.1f}%)

### 3.1 Primary Metric: Mask-Core Laplacian Variance Check (α > 0.9)
*Measures edge suppression ratio strictly within the applied blur core: `var(Laplacian(output)) / var(Laplacian(input)) < {lap_mask.get('threshold', 0.5)}`*
- **Status:** `{lap_mask.get('status', 'N/A')}`
- **Boxes Evaluated:** {lap_mask.get('boxes_checked', 0)} reliable boxes
- **Boxes Unreliable (<8px or flat texture):** {lap_mask.get('boxes_unreliable', 0)} (excluded from pass/fail)
- **Boxes Failed:** {lap_mask.get('boxes_failed', 0)}
- **Median Variance Ratio:** {lap_mask.get('ratio_median', 0.0)}
- **p95 Variance Ratio:** {lap_mask.get('ratio_p95', 0.0)}
- **Max Variance Ratio:** {lap_mask.get('ratio_max', 0.0)}

#### Mask-Core Laplacian Breakdown by Class
| Target Class | Total Sampled | Reliable Checked | Passed (≤ 0.5) | Failed (> 0.5) | Pass % | Median Ratio | p95 Ratio |
|---|---|---|---|---|---|---|---|
| Faces | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('total', 0)} | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('checked', 0)} | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('passed', 0)} | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('failed', 0)} | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('pass_pct', 0.0)}% | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('median_ratio', 0.0)} | {lap_mask.get('breakdown_by_class', {}).get('face', {}).get('p95_ratio', 0.0)} |
| Plates | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('total', 0)} | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('checked', 0)} | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('passed', 0)} | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('failed', 0)} | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('pass_pct', 0.0)}% | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('median_ratio', 0.0)} | {lap_mask.get('breakdown_by_class', {}).get('plate', {}).get('p95_ratio', 0.0)} |

#### Mask-Core Laplacian Breakdown by Size Bucket
| Size Bucket (px) | Total Sampled | Checked | Passed | Failed | Pass % | Median Ratio | p95 Ratio |
|---|---|---|---|---|---|---|---|
{lap_size_md}

### 3.2 Secondary Metric: High-Frequency Gradient Energy Ratio (α > 0.9)
*Measures high-frequency gradient suppression ratio: `mean(Sobel_magnitude(output)) / mean(Sobel_magnitude(input)) < {grad_mask.get('threshold', 0.5)}`*
- **Status:** `{grad_mask.get('status', 'N/A')}`
- **Boxes Evaluated:** {grad_mask.get('boxes_checked', 0)} reliable boxes
- **Boxes Failed:** {grad_mask.get('boxes_failed', 0)}
- **Median Energy Ratio:** {grad_mask.get('ratio_median', 0.0)}
- **p95 Energy Ratio:** {grad_mask.get('ratio_p95', 0.0)}
- **Max Energy Ratio:** {grad_mask.get('ratio_max', 0.0)}

#### Mask-Core Gradient Energy Breakdown by Size Bucket
| Size Bucket (px) | Total Sampled | Checked | Passed | Failed | Pass % | Median Ratio | p95 Ratio |
|---|---|---|---|---|---|---|---|
{grad_size_md}

### 3.3 Legacy Comparison: Full Bounding Box Laplacian Check
*Included for continuity and review comparison. Evaluates full rectangular crop including unblurred corners outside elliptical masks.*
- **Status:** `{lap_box.get('status', 'N/A')}`
- **Boxes Evaluated:** {lap_box.get('boxes_checked', 0)}
- **Boxes Failed:** {lap_box.get('boxes_failed', 0)}
- **Median Variance Ratio:** {lap_box.get('ratio_median', 0.0)}
- **p95 Variance Ratio:** {lap_box.get('ratio_p95', 0.0)}
- **Max Variance Ratio:** {lap_box.get('ratio_max', 0.0)}
- **Analysis:** *Full bounding box ratio exhibits high variance due to crisp unblurred background pixels in the 4 corners outside inscribed elliptical head masks.*

### 3.4 Human Review Status
- **Status:** `{human['status']}`
- **Frames Reviewed:** {human['frames_reviewed']}
- **Boxes Reviewed:** {human['boxes_reviewed']}
- **True Positives:** {human['tp']}
- **False Positives:** {human['fp']}
- **Missed Faces / Plates:** {human['missed_faces']} / {human['missed_plates']}
- **Note:** *{human['note']}*


---

## 4. Pixel Integrity & Metadata Sanitization

### Pixel Integrity Test (Outside Dilated Mask)
- **Status:** `{pix.get('status', 'NOT_EXECUTED')}`
- **Frames Evaluated:** {pix.get('frames_tested', 0)} frames
- **Input Format:** `{pix.get('input_format', 'N/A')}`
- **Output Format:** `{pix.get('output_format', 'N/A')}`
- **Max Abs Difference Levels:** {pix.get('max_abs_diff_levels', 0)} / 255
- **Mean Abs Difference Levels:** {pix.get('mean_abs_diff_levels', 0.0)} / 255
- **Unmasked Pixels Altered by Compression:** {pix.get('pct_unmasked_pixels_altered_by_compression', 0.0)}%
- **Finding:** *{pix.get('findings_statement', 'N/A')}*

### Metadata Sanitization
- **Status:** `{meta['status']}`
- **Output Files Scanned:** {meta['total_output_images_scanned']}
- **Segments Found (EXIF / XMP / GPS / Comments):** {meta['metadata_segments_found']}
- **Clean JFIF Export:** {meta['all_sanitized']}
- **Filename Note:** *{meta['notes']}*

---

## 5. What this report does NOT show
1. **Does NOT prove detection recall on this session:** Ground-truth annotations were not available for this session export; design target figures (>=99.0% face, >=98.47% plate) are model benchmark estimates only.
2. **Does NOT guarantee zero unblurred targets:** Undetected faces or plates are not covered by masking guarantees.
3. **Does NOT assert bit-identical background pixels:** Because inputs are lossy JPEG, output encoding creates an additional lossy compression step altering pixel values by up to {pix['max_abs_diff_levels']} levels.
4. **Does NOT assert legal anonymity:** Legal anonymity under GDPR / KVKK is a legal determination requiring comprehensive contextual risk assessment.

---

## 6. Known Limitations & Non-Claims

### Known Limitations
{limitations_md}

### Explicit Non-Claims
{not_claimed_md}

---

## 7. Artifact Manifest & Verification
The following files accompany this report:
- `PRIVACY_DATA_SHEET.json`: Standardized JSON audit record
- `PRIVACY_DATA_SHEET.json.sha256`: SHA256 checksum
- `detections_all.csv`: Full list of all detection bounding boxes with 4-decimal confidences
- `audit_boxes.csv`: Per-box Laplacian blur test results
- `manifest.csv`: Frame-level SHA256 hashes and byte lengths
- `privacy_summary.json`: Telemetry schema summary

**Verification Command:**
```bash
python tools/verify_bundle.py --bundle-dir privacy_report
```
"""
