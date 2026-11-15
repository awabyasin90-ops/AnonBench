"""Phase 3b: build a TensorRT engine from an ONNX with an explicit shape profile.

Shapes are BxCxHxW. The engine allocates GPU memory for the MAX shape, so keep max tight.

python tools/build_engine.py --onnx weights/yolov8m-face.onnx --out weights/yolov8m-face_b4_fp16.engine ^
    --min 1x3x960x960 --opt 4x3x960x1280 --max 4x3x1280x1280 --precision fp16 --workspace-mib 1024
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from pathlib import Path

import torch  # noqa: F401  (load CUDA DLLs first on Windows)
import tensorrt as trt
import onnx

from _common import parse_shape


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min", required=True)
    ap.add_argument("--opt", required=True)
    ap.add_argument("--max", required=True)
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--workspace-mib", type=int, default=1024)
    ap.add_argument("--opt-level", type=int, default=3, help="0-5; 4-5 = longer build, sometimes faster engine")
    a = ap.parse_args()
    onnx_path, out = Path(a.onnx), Path(a.out)
    mn, opt, mx = parse_shape(a.min), parse_shape(a.opt), parse_shape(a.max)
    assert all(l <= o <= h for l, o, h in zip(mn, opt, mx)), "need min <= opt <= max in every dimension"

    major = int(trt.__version__.split(".")[0])
    meta = {p.key: p.value for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props}
    names = ast.literal_eval(meta.get("names", "{}"))
    parse_path = onnx_path
    if a.precision == "fp16" and major >= 11:
        # TensorRT 11 removed the FP16 builder flag: bake FP16 into the ONNX with NVIDIA ModelOpt AutoCast.
        import numpy as np
        from modelopt.onnx import autocast
        input_name = onnx.load(str(onnx_path), load_external_data=False).graph.input[0].name
        converted = autocast.convert_to_mixed_precision(
            str(onnx_path), low_precision_type="fp16", keep_io_types=True,
            calibration_data={input_name: np.random.rand(*opt).astype(np.float32)})
        parse_path = onnx_path.with_suffix(".fp16.onnx")
        onnx.save(converted, str(parse_path))

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(parse_path)):
        errs = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise SystemExit("ONNX parse failed:\n" + "\n".join(errs))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, a.workspace_mib << 20)
    config.builder_optimization_level = a.opt_level
    if a.precision == "fp16" and major < 11:
        config.set_flag(trt.BuilderFlag.FP16)
    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, mn, opt, mx)
    config.add_optimization_profile(profile)

    cache_path = out.parent / "timing.cache"
    cache = config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
    config.set_timing_cache(cache, ignore_mismatch=False)

    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise SystemExit("engine build failed (see log above). Try a smaller --max or --workspace-mib.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(blob))
    cache_path.write_bytes(bytes(cache.serialize()))
    side = {
        "engine": out.name, "onnx": onnx_path.name, "onnx_sha256": sha256(onnx_path),
        "precision": a.precision, "profile": {"min": mn, "opt": opt, "max": mx},
        "workspace_mib": a.workspace_mib, "opt_level": a.opt_level,
        "task": meta.get("task"), "nc": len(names), "names": names, "stride": meta.get("stride"),
        "tensorrt": trt.__version__, "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0), "build_seconds": round(time.time() - t0, 1),
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(str(out) + ".json").write_text(json.dumps(side, indent=2), encoding="utf-8")
    print(json.dumps(side, indent=2))


if __name__ == "__main__":
    main()
