"""fastblur/runtime.py: torch-free helpers shared by run_blur_v2.py and tools/sweep_v2.py.

Pure Python + PyYAML, so every function here can be unit-tested on any machine
(no GPU, no torch, no TensorRT).  Rules implemented here:
  v1 R1  unknown config keys fail loudly            -> apply_override / load_config_v2
  v1 R6  never write next to the originals          -> assert_safe_paths
  R0/R1  throughput is measured, not assumed        -> steady_state_rate / build_summary
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml

# sweep_v2.py uses flat names; the YAML nests them.  --set accepts both forms.
FLAT_KEY_ALIASES: dict[str, str] = {
    "batch_size": "architecture.batch_size",
    "pool_slots": "architecture.pool_slots",
    "stream_strategy": "architecture.stream_strategy",
    "compute_stream_count": "architecture.compute_stream_count",
    "jpeg_backend": "architecture.jpeg_backend",
    "decode_threads": "threading.decode_threads",
    "writer_threads": "threading.writer_threads",
    "opencv_threads": "threading.opencv_threads",
    "input_path": "architecture.input_path",
    "output_path": "architecture.output_path",
}

REQUIRED_TOP = ["input_dir", "output_dir", "run_dir", "extensions", "models", "architecture",
                "threading", "shapes", "blur", "jpeg_quality", "calibration"]
REQUIRED_ARCH = ["batch_size", "pool_slots", "stream_strategy", "compute_stream_count", "input_path", "output_path"]
REQUIRED_THREADING = ["decode_threads", "writer_threads", "opencv_threads"]
REQUIRED_CALIBRATION = ["warmup_frames", "rolling_window_frames"]
STREAM_STRATEGIES = ("fully_concurrent", "interleaved_serial")

# CPU/GPU work placement variants (plan Section 24).  A name may be valid in the config yet not implemented
# in run_blur_v2.py; assert_variants_implemented() makes such a run fail loudly instead of silently
# running the baseline path under another name (rule R10).
INPUT_PATHS = ("cpu_letterbox", "gpu_letterbox", "gpu_decode")
OUTPUT_PATHS = ("cpu_blur_cpu_encode", "gpu_blur_cpu_encode", "gpu_blur_gpu_encode")
IMPLEMENTED_VARIANTS: dict[str, set[str]] = {
    "input_path": {"cpu_letterbox"},
    "output_path": {"cpu_blur_cpu_encode"},
}


# --------------------------------------------------------------------------- config
def apply_override(cfg: dict, expr: str) -> None:
    """Apply one 'key=value' override in place.  Value is parsed as YAML (2 -> int, null -> None)."""
    key, sep, raw = expr.partition("=")
    if not sep or not key:
        raise ValueError(f"override must look like key=value, got {expr!r}")
    key = FLAT_KEY_ALIASES.get(key, key)
    parts = key.split(".")
    node: Any = cfg
    for p in parts[:-1]:
        if not isinstance(node, dict) or p not in node:
            raise KeyError(f"unknown config key {key!r}")
        node = node[p]
    if not isinstance(node, dict) or (parts[-1] not in node and parts[:-1] != ["shapes"]):
        raise KeyError(f"unknown config key {key!r}")
    node[parts[-1]] = yaml.safe_load(raw)


def load_config_v2(path: str | Path, overrides: Iterable[str] = ()) -> dict:
    """Load + validate config/pipeline_v2.yaml (the v2 schema is nested; v1's loader rejects it).

    Overrides are applied to the raw YAML first, then paths are resolved and everything is validated,
    so an override can never produce a config that skipped validation.
    """
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    for expr in overrides:
        apply_override(cfg, expr)

    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise KeyError(f"config missing keys: {missing}")
    for section, keys in (("architecture", REQUIRED_ARCH), ("threading", REQUIRED_THREADING),
                          ("calibration", REQUIRED_CALIBRATION)):
        miss = [k for k in keys if k not in cfg[section]]
        if miss:
            raise KeyError(f"config section '{section}' missing keys: {miss}")

    base = path.resolve().parent.parent  # blur_pipeline/ (config lives in blur_pipeline/config/)
    for key in ("input_dir", "output_dir", "run_dir"):
        p = Path(str(cfg[key]))
        cfg[key] = p if p.is_absolute() else base / p
    for name in ("face", "vehicle", "plate"):
        m = cfg["models"][name]
        for k in ("engine", "pt"):
            if m.get(k):
                p = Path(str(m[k]))
                m[k] = p if p.is_absolute() else base / p

    face, veh, plate = cfg["models"]["face"], cfg["models"]["vehicle"], cfg["models"]["plate"]
    arch, thr, cal = cfg["architecture"], cfg["threading"], cfg["calibration"]
    if face["long_side"] % 32:
        raise ValueError("models.face.long_side must be a multiple of 32")
    if veh["long_side"] % 32:
        raise ValueError("models.vehicle.long_side must be a multiple of 32")
    for g, (h, w) in cfg["shapes"].items():
        if h % 32 or w % 32:
            raise ValueError(f"shapes.{g}={(h, w)}: dimensions must be multiples of 32")
    if plate.get("enabled", True) and plate.get("mode") not in ("crops", "full_frame"):
        raise ValueError("models.plate.mode must be 'crops' or 'full_frame'")
    if cfg["blur"]["method"] not in ("fixed", "adaptive"):
        raise ValueError("blur.method must be 'fixed' or 'adaptive'")
    if not 1 <= int(cfg["jpeg_quality"]) <= 100:
        raise ValueError("jpeg_quality must be 1..100")
    if arch["stream_strategy"] not in STREAM_STRATEGIES:
        raise ValueError(f"architecture.stream_strategy must be one of {STREAM_STRATEGIES}")
    if int(arch["compute_stream_count"]) not in (1, 2, 3):
        raise ValueError("architecture.compute_stream_count must be 1, 2 or 3")
    if arch["input_path"] not in INPUT_PATHS:
        raise ValueError(f"architecture.input_path must be one of {INPUT_PATHS}")
    if arch["output_path"] not in OUTPUT_PATHS:
        raise ValueError(f"architecture.output_path must be one of {OUTPUT_PATHS}")
    for key, val in (("batch_size", arch["batch_size"]), ("pool_slots", arch["pool_slots"]),
                     ("decode_threads", thr["decode_threads"]), ("writer_threads", thr["writer_threads"])):
        if int(val) < 1:
            raise ValueError(f"{key} must be >= 1")
    if int(cal["warmup_frames"]) < 0 or int(cal["rolling_window_frames"]) < 2:
        raise ValueError("calibration.warmup_frames must be >= 0 and rolling_window_frames >= 2")
    return cfg


def assert_variants_implemented(cfg: dict) -> None:
    """Fail loudly when the config selects a work-placement variant the orchestrator does not implement yet."""
    for key, implemented in IMPLEMENTED_VARIANTS.items():
        value = cfg["architecture"][key]
        if value not in implemented:
            raise NotImplementedError(
                f"architecture.{key}={value!r} is a defined variant but is not implemented in run_blur_v2.py "
                f"(implemented: {sorted(implemented)}). Implement it, then add it to IMPLEMENTED_VARIANTS.")


# --------------------------------------------------------------------------- v1 rule R6
def assert_safe_paths(input_dir: str | Path, output_dir: str | Path) -> None:
    """Refuse to run when the output could overwrite (or be mistaken for) the originals."""
    src = Path(input_dir).resolve()
    dst = Path(output_dir).resolve()
    if src == dst or src in dst.parents or dst in src.parents:
        raise ValueError(f"output_dir ({dst}) and input_dir ({src}) must be separate folders, "
                         "and neither may contain the other")


# --------------------------------------------------------------------------- throughput
def rolling_rates(done_times: list[float], window: int) -> list[float]:
    """Images/s over each consecutive block of `window` finished frames."""
    ordered = sorted(done_times)
    rates: list[float] = []
    for i in range(0, len(ordered) - window + 1, window):
        span = ordered[i + window - 1] - ordered[i]
        if span > 0:
            rates.append((window - 1) / span)
    return rates


def steady_state_rate(done_times: list[float], warmup_frames: int) -> float | None:
    """Images/s over the frames that finished after the first `warmup_frames` (None if not measurable)."""
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be >= 0")
    ordered = sorted(done_times)
    if len(ordered) < warmup_frames + 2:
        return None
    window = ordered[warmup_frames:]
    span = window[-1] - window[0]
    if span <= 0:
        return None
    return (len(window) - 1) / span


def suggest_warmup_frames(done_times: list[float], window: int, tolerance: float = 0.05) -> int | None:
    """Smallest frame count after which every rolling rate stays within +-tolerance of the late-run median.

    This is how calibration.warmup_frames is MEASURED instead of guessed.  None = never stabilised.
    """
    rates = rolling_rates(done_times, window)
    if len(rates) < 4:
        return None
    tail = sorted(rates[len(rates) // 2:])
    ref = tail[len(tail) // 2]
    if ref <= 0:
        return None
    for i in range(len(rates) - 2):  # the stable stretch must span at least 3 blocks, one lucky block is not stability
        if all(abs(r - ref) / ref <= tolerance for r in rates[i:]):
            return i * window
    return None


def build_summary(*, run_name: str, planned: int, images_ok: int, errors: int, elapsed_s: float,
                  done_times: list[float], warmup_frames: int, window: int,
                  faces_total: int, plates_total: int, effective_cfg: dict,
                  corrupt: int = 0) -> dict:
    """The record tools/sweep_v2.py reads.  Every field is measured in the run, none is assumed."""
    return {
        "run_name": run_name,
        "planned_images": planned,
        "images_ok": images_ok,
        "errors": errors,
        "corrupt_input": corrupt,
        "elapsed_s": round(elapsed_s, 3),
        "images_per_s": (images_ok / elapsed_s) if elapsed_s > 0 else 0.0,
        "steady_state_images_per_s": steady_state_rate(done_times, warmup_frames),
        "warmup_frames": warmup_frames,
        "suggested_warmup_frames": suggest_warmup_frames(done_times, window),
        "rolling_window_frames": window,
        "rolling_rates_images_per_s": [round(r, 3) for r in rolling_rates(done_times, window)],
        "faces_total": faces_total,
        "plates_total": plates_total,
        "config": json.loads(json.dumps(effective_cfg, default=str)),
    }


def write_summary(path: str | Path, summary: dict) -> None:
    """Atomic write: readers never see a half-written summary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)
