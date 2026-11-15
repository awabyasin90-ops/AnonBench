"""Shared helpers for tools/ scripts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # blur_pipeline/ or FAST/
FACTS = ROOT / "run_facts"
sys.path.insert(0, str(ROOT))


def save_fact(name: str, data: dict) -> Path:
    FACTS.mkdir(exist_ok=True)
    p = FACTS / f"{name}.json"
    p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"[saved] {p}")
    return p


def parse_shape(s: str) -> tuple[int, ...]:
    """'4x3x960x1280' -> (4, 3, 960, 1280)"""
    return tuple(int(v) for v in s.lower().split("x"))
