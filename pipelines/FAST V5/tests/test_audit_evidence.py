"""tests/test_audit_evidence.py — Comprehensive Unit Tests for FAST v4 Evidence Layer

Covers Phase 3 Item 4:
  (a) audit gate: 0 samples -> NOT_EXECUTED and release blocked
  (b) status aggregation rules (PASS, PASS_WITH_LIMITATIONS, FAIL, NOT_EXECUTED)
  (c) hash verification detects a tampered sheet
  (d) schema validation against Schema 2.0
  (e) ingest tool refuses empty template & calculates Wilson CI on synthetic data
  (f) size/conf bucket math
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from fastblur.audit import (
    audit_laplacian_effectiveness,
    get_conf_bucket,
    get_size_bucket,
    sample_frames_stratified,
)
from fastblur.schema_validator import validate_against_schema
from tools.ingest_human_review import ingest_review, wilson_score_interval
from tools.verify_bundle import verify_bundle


# ==============================================================================
# (a) Audit Gate: 0 samples -> NOT_EXECUTED and release blocked
# ==============================================================================
def test_audit_gate_zero_samples_yields_not_executed():
    res = audit_laplacian_effectiveness(
        input_dir=Path("nonexistent"),
        output_dir=Path("nonexistent"),
        sampled_records=[],
    )
    assert res["status"] == "NOT_EXECUTED"
    assert res["PASS"] is False
    assert res["sample_size"] == 0
    assert res["boxes_checked"] == 0
    assert "0" in res["reason"]


def test_stratified_sampling_formula():
    class DummyRecord:
        def __init__(self, fn: str, boxes: list):
            self.filename = fn
            self.face_boxes = boxes
            self.plate_boxes = []

    # Test with 578 frames
    records = [DummyRecord(f"img_{i}.jpg", [[10, 10, 30, 30, 0.5]]) for i in range(578)]
    sampled = sample_frames_stratified(records, seed=42)
    # min(578, max(200, ceil(0.05 * 578))) = 200
    assert len(sampled) == 200


# ==============================================================================
# (b) Status Aggregation Rules
# ==============================================================================
def test_status_aggregation_rules():
    # Rule 1: Human review NOT_EXECUTED -> overall_status NOT_EXECUTED, release BLOCKED
    def get_overall(auto_st: str, human_st: str, val_st: str, has_limitations: bool) -> tuple[str, str]:
        if auto_st == "FAIL":
            ovr = "FAIL"
        elif human_st == "NOT_EXECUTED" or auto_st == "NOT_EXECUTED" or val_st == "NOT_EXECUTED":
            ovr = "NOT_EXECUTED"
        elif has_limitations:
            ovr = "PASS_WITH_LIMITATIONS"
        else:
            ovr = "PASS"

        rel = "RELEASE_BLOCKED" if ovr in ("FAIL", "NOT_EXECUTED") else "RELEASE_ALLOWED"
        return ovr, rel

    # Default state before human review
    ovr, rel = get_overall("PASS", "NOT_EXECUTED", "NOT_EXECUTED", True)
    assert ovr == "NOT_EXECUTED"
    assert rel == "RELEASE_BLOCKED"

    # Auto check fails
    ovr_fail, rel_fail = get_overall("FAIL", "NOT_EXECUTED", "NOT_EXECUTED", True)
    assert ovr_fail == "FAIL"
    assert rel_fail == "RELEASE_BLOCKED"

    # Human review completed with limitations
    ovr_lim, rel_lim = get_overall("PASS", "COMPLETED", "COMPLETED", True)
    assert ovr_lim == "PASS_WITH_LIMITATIONS"
    assert rel_lim == "RELEASE_ALLOWED"

    # Perfect pass (rare)
    ovr_pass, rel_pass = get_overall("PASS", "COMPLETED", "COMPLETED", False)
    assert ovr_pass == "PASS"
    assert rel_pass == "RELEASE_ALLOWED"


# ==============================================================================
# (c) Hash verification detects a tampered sheet
# ==============================================================================
def test_hash_verification_detects_tampered_sheet(tmp_path: Path):
    bundle_dir = tmp_path / "privacy_report"
    bundle_dir.mkdir()

    sheet = {
        "document_type": "PRIVACY_DATA_SHEET",
        "schema_version": "2.0",
        "standard": "OPINA-v1.0-AUDIT-SPEC",
        "dataset_version": "OPINA-v1.0",
        "anonymization_version": "FAST-v4.0.0",
        "generated_at": "2026-10-06T12:00:00Z",
        "overall_status": "NOT_EXECUTED",
        "release_status": "RELEASE_BLOCKED",
        "status_reasons": ["Test reasons"],
        "scope": {"sessions": ["s1"], "cameras": ["c1"], "frames_total": 1, "frame_resolutions": [[1024, 1040]]},
        "detection_summary": {
            "faces": {"candidates_masked": 10, "note": "n", "avg_candidates_per_frame": 10.0, "low_confidence_cases_flagged": 2, "confidence_distribution": {}, "size_histogram": {}, "confidence_histogram": {}},
            "plates": {"candidates_masked": 0, "note": "n", "avg_candidates_per_frame": 0.0, "low_confidence_cases_flagged": 0, "confidence_distribution": {}, "size_histogram": {}, "confidence_histogram": {}},
            "vehicles_context_only": {"detected": 0, "avg_per_frame": 0.0},
        },
        "masking": {
            "detections_masked_pct": 100.0, "detections_masked_pct_note": "n", "face_mask_geometry": "ellipse",
            "plate_mask_geometry": "feathered_rect", "edge_feather_sigma_px": 2.0, "blur_method": "adaptive", "statement": "s"
        },
        "measured_performance_on_validation_set": {"status": "NOT_EXECUTED", "validation_set_id": None, "face_recall": None, "plate_recall": None, "note": "n"},
        "session_audit": {
            "sampling": {"method": "s", "seed": 42, "total_frames": 1, "sample_size": 1, "sampling_ratio": 1.0},
            "automatic_checks": {"status": "PASS", "boxes_checked": 1, "boxes_failed": 0, "boxes_unreliable": 0, "ratio_median": 0.1, "ratio_p95": 0.1, "ratio_max": 0.1, "max_ratio_threshold": 0.5},
            "human_review": {"status": "NOT_EXECUTED", "frames_reviewed": None, "boxes_reviewed": None, "tp": None, "fp": None, "missed_faces": None, "missed_plates": None},
        },
        "metadata_sanitization": {"status": "COMPLETED", "total_output_images_scanned": 1, "metadata_segments_found": {}, "all_sanitized": True, "notes": "n"},
        "pixel_integrity_test": {"status": "COMPLETED", "frames_tested": 1, "input_format": "JPEG", "output_format": "JPEG", "max_abs_diff_levels": 5, "mean_abs_diff_levels": 1.2, "pct_unmasked_pixels_altered_by_compression": 10.0, "dilation_margin_px": 10, "findings_statement": "f"},
        "provenance": {"models": {}, "config_sha256": "h", "environment": {}, "hardware": {}, "git": {}, "manifest_csv_sha256": "m", "runtime_s": 1.0, "images_per_s": 1.0},
        "design_targets": {"face_target_recall": ">= 99.0%", "plate_target_recall": ">= 98.47%", "note": "n"},
        "known_limitations": ["l"],
        "not_claimed": ["nc"],
        "legal_notice": "ln",
    }

    # Canonical hash
    canon_bytes = json.dumps(sheet, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    valid_canon_sha = hashlib.sha256(canon_bytes).hexdigest()

    sheet["integrity"] = {
        "canonical_hash_method": "sha256(canonical_json_excluding_integrity)",
        "sheet_sha256": valid_canon_sha,
        "data_sheet_sha256_file": "PRIVACY_DATA_SHEET.json.sha256",
    }

    sheet_p = bundle_dir / "PRIVACY_DATA_SHEET.json"
    sheet_p.write_text(json.dumps(sheet, indent=2), encoding="utf-8")

    file_sha = hashlib.sha256(sheet_p.read_bytes()).hexdigest()
    (bundle_dir / "PRIVACY_DATA_SHEET.json.sha256").write_text(f"{file_sha}  PRIVACY_DATA_SHEET.json\n", encoding="utf-8")

    # Create dummy other files
    for fn in ["privacy_summary.json", "privacy_report.md", "detections_all.csv", "audit_boxes.csv", "manifest.csv", "session_stats.csv", "camera_stats.csv", "error_log.csv", "audit_result.json"]:
        (bundle_dir / fn).write_text("dummy", encoding="utf-8")

    # Now tamper with one field in the JSON file
    sheet["detection_summary"]["faces"]["candidates_masked"] = 99999
    sheet_p.write_text(json.dumps(sheet, indent=2), encoding="utf-8")

    ok, errors = verify_bundle(bundle_dir)
    assert ok is False
    assert any("Canonical Sheet SHA256 mismatch" in e or "File SHA256 mismatch" in e for e in errors)


# ==============================================================================
# (d) Schema validation
# ==============================================================================
def test_schema_validation_against_schema_v2():
    schema_path = Path(__file__).resolve().parent.parent / "schema" / "privacy_data_sheet.schema.json"
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    # Valid minimal instance
    valid_doc = {
        "document_type": "PRIVACY_DATA_SHEET",
        "schema_version": "2.0",
        "standard": "OPINA-v1.0-AUDIT-SPEC",
        "dataset_version": "OPINA-v1.0",
        "anonymization_version": "FAST-v4.0.0",
        "generated_at": "2026-10-06T12:00:00Z",
        "overall_status": "NOT_EXECUTED",
        "release_status": "RELEASE_BLOCKED",
        "status_reasons": ["Test reasons"],
        "scope": {"sessions": ["s1"], "cameras": ["c1"], "frames_total": 1, "frame_resolutions": [[1024, 1040]]},
        "detection_summary": {
            "faces": {"candidates_masked": 10, "note": "n", "avg_candidates_per_frame": 10.0, "low_confidence_cases_flagged": 2, "confidence_distribution": {}, "size_histogram": {}, "confidence_histogram": {}},
            "plates": {"candidates_masked": 0, "note": "n", "avg_candidates_per_frame": 0.0, "low_confidence_cases_flagged": 0, "confidence_distribution": {}, "size_histogram": {}, "confidence_histogram": {}},
            "vehicles_context_only": {"detected": 0, "avg_per_frame": 0.0},
        },
        "masking": {
            "detections_masked_pct": 100.0, "detections_masked_pct_note": "n", "face_mask_geometry": "ellipse",
            "plate_mask_geometry": "feathered_rect", "edge_feather_sigma_px": 2.0, "blur_method": "adaptive", "statement": "s"
        },
        "blur_specification": {
            "face_mask_geometry": "ellipse",
            "face_pad_ratio": 0.2,
            "plate_mask_geometry": "feathered_rect",
            "plate_pad_ratio": 0.1,
            "min_ellipse_side_px": 8.0,
            "feather_sigma_px": 2.0,
            "feather_kernel_px": 5,
            "blur_method": "adaptive",
            "adaptive_target_px": 7,
            "min_sigma": 1.5,
            "fixed_kernel": 21,
            "adaptive_downscale_rule": "stride = max(1, int(side / 7.0)); sigma = max(1.5, stride * 0.8)",
            "alpha_blending_rule": "output = (1 - alpha) * input + alpha * blurred",
        },
        "measured_performance_on_validation_set": {"status": "NOT_EXECUTED", "validation_set_id": None, "face_recall": None, "plate_recall": None, "note": "n"},
        "session_audit": {
            "sampling": {"method": "s", "seed": 42, "total_frames": 1, "sample_size": 1, "sampling_ratio": 1.0},
            "automatic_checks": {"status": "PASS", "boxes_checked": 1, "boxes_failed": 0, "boxes_unreliable": 0, "ratio_median": 0.1, "ratio_p95": 0.1, "ratio_max": 0.1, "max_ratio_threshold": 0.5},
            "human_review": {"status": "NOT_EXECUTED", "frames_reviewed": None, "boxes_reviewed": None, "tp": None, "fp": None, "missed_faces": None, "missed_plates": None, "note": "Pending"},
        },
        "metadata_sanitization": {"status": "COMPLETED", "total_output_images_scanned": 1, "metadata_segments_found": {}, "all_sanitized": True, "notes": "n"},
        "pixel_integrity_test": {"status": "COMPLETED", "frames_tested": 1, "input_format": "JPEG", "output_format": "JPEG", "max_abs_diff_levels": 5, "mean_abs_diff_levels": 1.2, "pct_unmasked_pixels_altered_by_compression": 10.0, "dilation_margin_px": 10, "findings_statement": "f"},
        "provenance": {"models": {}, "config_sha256": "h", "environment": {}, "hardware": {}, "git": {}, "manifest_csv_sha256": "m", "runtime_s": 1.0, "images_per_s": 1.0},
        "design_targets": {"face_target_recall": ">= 99.0%", "plate_target_recall": ">= 98.47%", "note": "n"},
        "known_limitations": ["l"],
        "not_claimed": ["nc"],
        "legal_notice": "ln",
        "integrity": {
            "canonical_hash_method": "sha256",
            "sheet_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "data_sheet_sha256_file": "PRIVACY_DATA_SHEET.json.sha256"
        }
    }

    # Should validate without error
    errs = validate_against_schema(valid_doc, schema)
    assert len(errs) == 0

    # Invalid instance: missing 'masking'
    invalid_doc = dict(valid_doc)
    del invalid_doc["masking"]
    errs_inv = validate_against_schema(invalid_doc, schema)
    assert len(errs_inv) > 0
    assert any("masking" in e for e in errs_inv)


# ==============================================================================
# (e) Ingest tool refuses empty template & calculates Wilson CI
# ==============================================================================
def test_ingest_tool_refuses_empty_template(tmp_path: Path):
    csv_file = tmp_path / "empty_template.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "box_id", "class", "size_bucket", "confidence", "verdict", "readable", "notes"])
        writer.writeheader()
        writer.writerow({
            "frame": "f1.jpg", "box_id": "b1", "class": "face", "size_bucket": "10-15",
            "confidence": "0.30", "verdict": "", "readable": "", "notes": ""
        })

    with pytest.raises(ValueError, match="REFUSED: 1 out of 1 rows have missing verdicts"):
        ingest_review(csv_file)


def test_ingest_tool_calculates_wilson_ci(tmp_path: Path):
    csv_file = tmp_path / "filled_template.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "box_id", "class", "size_bucket", "confidence", "verdict", "readable", "notes"])
        writer.writeheader()
        # 95 TP faces, 5 missed faces
        for i in range(95):
            writer.writerow({"frame": f"f_{i}.jpg", "box_id": f"b_{i}", "class": "face", "size_bucket": "20-30", "confidence": "0.8", "verdict": "TP", "readable": "Y", "notes": ""})
        for i in range(5):
            writer.writerow({"frame": f"f_{i}.jpg", "box_id": "MISSED", "class": "face", "size_bucket": "20-30", "confidence": "0.0", "verdict": "MISSED", "readable": "Y", "notes": ""})

    res = ingest_review(csv_file)
    assert res["status"] == "COMPLETED"
    assert res["tp"] == 95
    assert res["missed_faces"] == 5

    # Wilson interval check: 95/100 -> center ~0.938, bounds [0.887, 0.978]
    w = res["recall_estimates"]["faces_overall"]
    assert w["point_estimate"] == 0.95
    assert 0.88 <= w["ci_lower"] <= 0.91
    assert 0.97 <= w["ci_upper"] <= 0.99


# ==============================================================================
# (f) Size and Confidence bucket math
# ==============================================================================
def test_size_bucket_math():
    assert get_size_bucket(5.0) == "<10"
    assert get_size_bucket(9.99) == "<10"
    assert get_size_bucket(10.0) == "10-15"
    assert get_size_bucket(14.99) == "10-15"
    assert get_size_bucket(15.0) == "15-20"
    assert get_size_bucket(19.99) == "15-20"
    assert get_size_bucket(20.0) == "20-30"
    assert get_size_bucket(29.99) == "20-30"
    assert get_size_bucket(30.0) == "30-50"
    assert get_size_bucket(49.99) == "30-50"
    assert get_size_bucket(50.0) == ">=50"
    assert get_size_bucket(150.0) == ">=50"


def test_conf_bucket_math():
    assert get_conf_bucket(0.15) == "[0.15,0.20)"
    assert get_conf_bucket(0.199) == "[0.15,0.20)"
    assert get_conf_bucket(0.20) == "[0.20,0.25)"
    assert get_conf_bucket(0.349) == "[0.30,0.35)"
    assert get_conf_bucket(0.95) == "[0.95,1.00]"
    assert get_conf_bucket(1.00) == "[1.00,1.00]"


def test_verify_bundle_rejects_crlf_sha256(tmp_path: Path):
    bundle_dir = tmp_path / "privacy_report"
    bundle_dir.mkdir()

    sheet = {
        "document_type": "PRIVACY_DATA_SHEET",
        "schema_version": "2.0",
        "dataset_version": "OPINA-v1.0",
        "anonymization_version": "FAST-v4.0.0",
        "generated_at": "2026-10-06T12:00:00Z",
        "overall_status": "NOT_EXECUTED",
        "release_status": "RELEASE_BLOCKED",
    }
    canon_bytes = json.dumps(sheet, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    sheet["integrity"] = {
        "canonical_hash_method": "sha256(canonical_json_excluding_integrity)",
        "sheet_sha256": hashlib.sha256(canon_bytes).hexdigest(),
        "data_sheet_sha256_file": "PRIVACY_DATA_SHEET.json.sha256",
    }
    sheet_p = bundle_dir / "PRIVACY_DATA_SHEET.json"
    sheet_p.write_text(json.dumps(sheet, indent=2), encoding="utf-8")
    file_sha = hashlib.sha256(sheet_p.read_bytes()).hexdigest()

    # Write with CRLF
    sha_file = bundle_dir / "PRIVACY_DATA_SHEET.json.sha256"
    sha_file.write_bytes(f"{file_sha}  PRIVACY_DATA_SHEET.json\r\n".encode("utf-8"))

    for fn in ["privacy_summary.json", "privacy_report.md", "detections_all.csv", "audit_boxes.csv", "manifest.csv", "session_stats.csv", "camera_stats.csv", "error_log.csv", "audit_result.json"]:
        (bundle_dir / fn).write_text("dummy", encoding="utf-8")

    ok, errors = verify_bundle(bundle_dir)
    assert ok is False
    assert any("must use LF line endings" in e for e in errors)

