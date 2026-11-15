"""Load and validate config/pipeline.yaml. Fails loudly on anything inconsistent."""
from __future__ import annotations

from pathlib import Path

import yaml

REQUIRED_TOP = [
    "input_dir", "output_dir", "run_dir", "extensions", "models", "shapes",
    "batch_size", "decode_threads", "writer_threads", "prefetch_batches",
    "max_pending_writes", "opencv_threads", "blur", "jpeg_quality"
]


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))

    # Support nested architecture and threading blocks by flattening if needed
    if "architecture" in cfg and isinstance(cfg["architecture"], dict):
        for k in ("batch_size", "pool_slots", "input_path", "output_path"):
            if k in cfg["architecture"] and k not in cfg:
                cfg[k] = cfg["architecture"][k]
    if "threading" in cfg and isinstance(cfg["threading"], dict):
        for k in ("decode_threads", "writer_threads", "opencv_threads", "cpu_affinity"):
            if k in cfg["threading"] and k not in cfg:
                cfg[k] = cfg["threading"][k]
        if "input_queue_depth" in cfg["threading"] and "prefetch_batches" not in cfg:
            cfg["prefetch_batches"] = max(1, cfg["threading"]["input_queue_depth"] // max(1, cfg.get("batch_size", 4)))
        if "result_queue_depth" in cfg["threading"] and "max_pending_writes" not in cfg:
            cfg["max_pending_writes"] = cfg["threading"]["result_queue_depth"]

    # Provide safe defaults for concurrency guards if missing
    if "prefetch_batches" not in cfg:
        cfg["prefetch_batches"] = 3
    if "max_pending_writes" not in cfg:
        cfg["max_pending_writes"] = 24
    if "opencv_threads" not in cfg:
        cfg["opencv_threads"] = 1

    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise KeyError(f"config missing keys: {missing}")

    base = path.parent.parent  # FAST_v.4/ (config lives in FAST_v.4/config/)
    for key in ("input_dir", "output_dir", "run_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (base / p)

    if cfg["input_dir"].resolve() == cfg["output_dir"].resolve():
        raise ValueError("input_dir and output_dir must be distinct directories (single encode guarantee)")

    for name in ("face", "vehicle", "plate"):
        m = cfg["models"][name]
        for k in ("engine", "pt"):
            if m.get(k):
                p = Path(m[k])
                m[k] = p if p.is_absolute() else (base / p)

    face, veh, plate = cfg["models"]["face"], cfg["models"]["vehicle"], cfg["models"]["plate"]
    if face["long_side"] % 64:
        raise ValueError("models.face.long_side must be a multiple of 64")
    if cfg.get("vehicle_input", "gpu_half") == "gpu_half" and veh["long_side"] * 2 != face["long_side"]:
        raise ValueError("vehicle_input=gpu_half requires vehicle.long_side == face.long_side / 2")

    for g, (h, w) in cfg["shapes"].items():
        if h % 64 or w % 64 or max(h, w) != face["long_side"]:
            raise ValueError(f"shapes.{g}={h,w}: both must be multiples of 64 and max must equal face.long_side")

    if plate.get("enabled") and plate["mode"] not in ("crops", "full_frame"):
        raise ValueError("models.plate.mode must be 'crops' or 'full_frame'")
    if plate.get("enabled") and plate["mode"] == "full_frame" and \
            plate["imgsz"] not in (face["long_side"], veh["long_side"]):
        raise ValueError("plate full_frame mode needs plate.imgsz equal to face or vehicle long_side")

    if cfg["blur"]["method"] not in ("fixed", "adaptive"):
        raise ValueError("blur.method must be 'fixed' or 'adaptive'")
    if not 1 <= int(cfg["jpeg_quality"]) <= 100:
        raise ValueError("jpeg_quality must be 1..100")

    # FAST v4 render section validation
    if "render" in cfg:
        rc = cfg["render"]
        if rc.get("face_mask") not in (None, "ellipse", "rect"):
            raise ValueError("render.face_mask must be 'ellipse' or 'rect'")
        if rc.get("plate_mask") not in (None, "feathered_rect", "rect"):
            raise ValueError("render.plate_mask must be 'feathered_rect' or 'rect'")
        if float(rc.get("feather_sigma_px", 2.0)) < 0.0:
            raise ValueError("render.feather_sigma_px must be >= 0.0")

    # FAST v4 report section validation
    if "report" in cfg:
        rep = cfg["report"]
        if "enabled" in rep and not isinstance(rep["enabled"], bool):
            raise ValueError("report.enabled must be a boolean")

    return cfg
