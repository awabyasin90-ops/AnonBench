"""tests/test_privacy_report.py — Unit tests for FAST v4 Privacy Report Module."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from fastblur.privacy_report import ReportCollector, ReportGenerator, FrameRecord


def test_summary_has_all_required_keys(tmp_path: Path):
    """R1: Verify privacy_summary.json contains all required schema sections."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(
        filename="s1/c1/001.jpg", session="s1", camera="c1", faces_detected=1,
        face_boxes=[[10, 10, 50, 50, 0.9]]
    ))
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    files = gen.generate()

    summary_file = files["summary_json"]
    assert summary_file.exists()
    data = json.loads(summary_file.read_text(encoding="utf-8"))

    expected = [
        "report_version", "generated_at", "pipeline_version", "detector_models",
        "render_config", "dataset_summary", "detection_summary", "blur_coverage",
        "error_summary", "audit_result", "throughput"
    ]
    for k in expected:
        assert k in data, f"Missing key {k}"


def test_csv_matches_detection_count(tmp_path: Path):
    """R2: detections_all.csv row count matches total detected objects in summary."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(
        filename="s1/c1/001.jpg", session="s1", camera="c1",
        faces_detected=2, plates_detected=1,
        face_boxes=[[10, 10, 50, 50, 0.9], [60, 60, 90, 90, 0.8]],
        plate_boxes=[[100, 100, 150, 120, 0.95]],
    ))
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    files = gen.generate()

    with open(files["detections_csv"], encoding="utf-8") as f:
        rows = list(csv.reader(f))
    # 1 header + 3 detection rows
    assert len(rows) - 1 == 3


def test_session_stats_covers_all_sessions(tmp_path: Path):
    """R3: session_stats.csv covers all sessions recorded."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(filename="s1/c1/001.jpg", session="s1", camera="c1"))
    collector.add_record(FrameRecord(filename="s2/c1/002.jpg", session="s2", camera="c1"))
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    files = gen.generate()

    with open(files["session_csv"], encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) - 1 == 2


def test_error_log_matches_errors(tmp_path: Path):
    """R4: error_log.csv accurately captures reported frame errors."""
    collector = ReportCollector()
    collector.record_error("s1/c1/err.jpg", "s1", "c1", "corrupt_input", "Corrupt JPEG header")
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    files = gen.generate()

    with open(files["error_csv"], encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert len(rows) - 1 == 1
    assert rows[1][3] == "corrupt_input"


def test_markdown_report_renders(tmp_path: Path):
    """R5: privacy_report.md renders valid markdown with core headers."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(filename="s1/c1/001.jpg", session="s1", camera="c1", status="ok"))
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    files = gen.generate()

    md = files["report_md"].read_text(encoding="utf-8")
    assert "# Privacy Report" in md
    assert "Executive Summary" in md
    assert "Production Blur Rendering Specification" in md


def test_report_deterministic(tmp_path: Path):
    """R6: Two identical generator runs on same data produce consistent structures."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(filename="s1/c1/001.jpg", session="s1", camera="c1", faces_detected=1, face_boxes=[[10, 10, 20, 20, 0.9]]))
    collector.finish()

    gen1 = ReportGenerator(collector=collector, output_dir=tmp_path / "out1")
    gen2 = ReportGenerator(collector=collector, output_dir=tmp_path / "out2")

    files1 = gen1.generate()
    files2 = gen2.generate()

    det1 = files1["detections_csv"].read_text(encoding="utf-8")
    det2 = files2["detections_csv"].read_text(encoding="utf-8")
    assert det1 == det2


def test_empty_run_produces_valid_report(tmp_path: Path):
    """R7: Zero processed frames generates valid empty report without errors."""
    collector = ReportCollector()
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    files = gen.generate()

    assert files["summary_json"].exists()
    summary = json.loads(files["summary_json"].read_text(encoding="utf-8"))
    assert summary["dataset_summary"]["total_frames"] == 0


def test_audit_result_embedded(tmp_path: Path):
    """R8: audit_result.json is embedded and preserved in the bundle."""
    collector = ReportCollector()
    collector.finish()

    audit_data = {"PASS": True, "sample_size": 500, "ratio_median": 0.02}
    gen = ReportGenerator(collector=collector, output_dir=tmp_path, audit_result=audit_data)
    files = gen.generate()

    assert files["audit_json"].exists()
    loaded_audit = json.loads(files["audit_json"].read_text(encoding="utf-8"))
    assert loaded_audit["PASS"] is True
    assert loaded_audit["sample_size"] == 500


def test_privacy_data_sheet_buyer_evidence(tmp_path: Path):
    """R9: PRIVACY_DATA_SHEET.json contains all required buyer-facing audit evidence."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(
        filename="s1/c1/001.jpg", session="s1", camera="c1",
        faces_detected=2, plates_detected=1,
        face_boxes=[[10, 10, 50, 50, 0.20], [60, 60, 90, 90, 0.85]],
        plate_boxes=[[100, 100, 150, 120, 0.95]],
        status="ok",
    ))
    collector.finish()

    gen = ReportGenerator(
        collector=collector,
        output_dir=tmp_path,
        cfg={"dataset_version": "OPINA-v1.0"},
        audit_result={"sample_size": 100, "boxes_checked": 250, "PASS": True},
    )
    files = gen.generate()

    assert "privacy_data_sheet" in files
    pds_path = files["privacy_data_sheet"]
    assert pds_path.exists()
    pds = json.loads(pds_path.read_text(encoding="utf-8"))

    # Verify Buyer-Facing Evidence requirements (Schema 2.0):
    assert pds["document_type"] == "PRIVACY_DATA_SHEET"
    assert pds["dataset_version"] == "OPINA-v1.0"
    assert pds["anonymization_version"] == "FAST-v4.0.0"

    # Schema 2.0 Detection Summary
    det_sum = pds["detection_summary"]
    assert det_sum["faces"]["candidates_masked"] == 2
    assert det_sum["plates"]["candidates_masked"] == 1
    assert det_sum["faces"]["low_confidence_cases_flagged"] == 1  # the 0.20 conf face

    # Blur Specification
    blur_spec = pds["blur_specification"]
    assert blur_spec["face_mask_geometry"] == "ellipse"
    assert blur_spec["blur_method"] == "adaptive"
    assert "short_side > 24px" in blur_spec["adaptive_downscale_rule"]

    # Session Audit
    session_audit = pds["session_audit"]
    assert session_audit["automatic_checks"]["boxes_checked"] == 250
    assert session_audit["automatic_checks"]["status"] == "PASS"
    assert session_audit["human_review"]["status"] == "NOT_EXECUTED"

    # Legacy blocks must be removed
    assert "detection_and_blur" not in pds
    assert "qa_and_traceability" not in pds
    # Must contain NO misleading blur_rate_pct or *_reviewed fields in detection_summary
    pds_str = json.dumps(pds)
    assert "blur_rate_pct" not in pds_str
    assert "target_recall" not in pds["detection_summary"]["faces"]
    assert "target_recall" not in pds["detection_summary"]["plates"]
    assert "low_confidence_cases_reviewed" not in pds_str

    # Check root copy also generated
    assert (tmp_path / "PRIVACY_DATA_SHEET.json").exists()
    assert (tmp_path / "PRIVACY_DATA_SHEET.json.sha256").exists()

