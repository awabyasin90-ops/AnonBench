"""tests/test_v4_integration.py — Integration tests for FAST v4 Pipeline and Report Validator."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import pytest

from fastblur.privacy_report import ReportCollector, ReportGenerator, FrameRecord
from tools.validate_report import validate_report


def test_validate_report_passes_on_good_report(tmp_path: Path):
    """I1: validate_report returns True and no errors on a properly formed report bundle."""
    collector = ReportCollector()
    collector.add_record(FrameRecord(
        filename="sessA/front/0001.jpg",
        session="sessA",
        camera="front",
        faces_detected=1,
        plates_detected=1,
        face_boxes=[[10, 10, 50, 50, 0.9]],
        plate_boxes=[[80, 80, 120, 100, 0.95]],
        status="ok",
    ))
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    gen.generate()

    rep_dir = tmp_path / "privacy_report"
    ok, errors = validate_report(rep_dir)
    assert ok, f"Expected validation to pass, got errors: {errors}"
    assert len(errors) == 0


def test_validate_report_fails_on_missing_file(tmp_path: Path):
    """I2: validate_report catches missing files and reports failure."""
    collector = ReportCollector()
    collector.finish()

    gen = ReportGenerator(collector=collector, output_dir=tmp_path)
    gen.generate()

    rep_dir = tmp_path / "privacy_report"
    # Delete detections_all.csv
    (rep_dir / "detections_all.csv").unlink()

    ok, errors = validate_report(rep_dir)
    assert not ok
    assert any("detections_all.csv" in e for e in errors)


def test_render_config_backward_compat():
    """I3: Config without render section defaults to rect masks without feather."""
    from fastblur.render_v4 import RenderConfig
    cfg = RenderConfig.from_dict({"blur": {"method": "fixed"}})
    assert cfg.face_mask == "ellipse"
    assert cfg.plate_mask == "feathered_rect"

    # Backward compat dict with rect explicitly:
    cfg_compat = RenderConfig.from_dict({"render": {"face_mask": "rect", "feather_sigma_px": 0.0}})
    assert cfg_compat.face_mask == "rect"
    assert cfg_compat.feather_sigma_px == 0.0


def test_single_encode_counter_in_task(tmp_path: Path):
    """I4: write_task returns mode == 'encode' when rendering and writing JPEG."""
    from fastblur.pipeline import write_task
    from fastblur.render_v4 import RenderConfig

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    src = tmp_path / "src.jpg"
    dst = tmp_path / "dst.jpg"
    cv2.imwrite(str(src), img)

    faces = np.array([[10, 10, 30, 30, 0.9]], dtype=np.float32)
    plates = np.zeros((0, 5), dtype=np.float32)
    cfg = RenderConfig()

    res = write_task(
        src=src,
        dst=dst,
        img=img,
        faces=faces,
        plates=plates,
        vehicles=None,
        blur_cfg={"method": "adaptive"},
        quality=95,
        copy_if_empty=False,
        render_cfg=cfg,
    )
    assert res["mode"] == "encode"
    assert dst.exists()
    assert res["bytes"] > 0
