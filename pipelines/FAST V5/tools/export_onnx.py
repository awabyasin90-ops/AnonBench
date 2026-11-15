"""Phase 3a: export each .pt to a dynamic-shape FP32 ONNX (batch, height, width all dynamic).

python tools/export_onnx.py weights/yolov8m-face.pt --imgsz 1280
python tools/export_onnx.py weights/yolo11m.pt --imgsz 640
python tools/export_onnx.py weights/license-plate.pt --imgsz 320
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import onnx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("--imgsz", type=int, required=True, help="tracing size; the ONNX stays dynamic")
    ap.add_argument("--opset", type=int, default=None, help="leave empty; try 17 if TensorRT cannot parse")
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.pt)
    kw = dict(format="onnx", imgsz=a.imgsz, dynamic=True, simplify=True, batch=1, device="cpu")
    if a.opset:
        kw["opset"] = a.opset
    out = Path(m.export(**kw))
    model = onnx.load(str(out), load_external_data=False)
    meta = {p.key: p.value for p in model.metadata_props}
    inp = model.graph.input[0]
    outs = model.graph.output
    dims = lambda t: [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]  # noqa: E731
    names = ast.literal_eval(meta.get("names", "{}"))
    print(f"ONNX: {out}")
    print(f"  input  {inp.name} {dims(inp)}")
    for o in outs:
        print(f"  output {o.name} {dims(o)}")
    print(f"  task={meta.get('task')} nc={len(names)} stride={meta.get('stride')}")
    ch = dims(outs[0])[1]
    if isinstance(ch, int) and ch < 4 + len(names):
        raise SystemExit("output channel count is too small: this is not a raw YOLO head (was NMS embedded?)")
    if len(outs) != 1 and meta.get("task") == "detect":
        print("  NOTE: more than one output; fastblur uses 'output0'")


if __name__ == "__main__":
    main()
