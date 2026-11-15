"""Phase 5: GPU-only throughput + VRAM headroom for the engines named in the config.

python tools/bench_engine.py --config config/pipeline.yaml --batches 1,2,4,8 --iters 30
Loads ALL enabled engines together (like production), reports free VRAM, then times each
face/vehicle batch size that the engine profile allows, for every planned shape.
"""
from __future__ import annotations

import argparse
import time

import torch

from _common import save_fact
from fastblur.config import load_config
from fastblur.trt_runner import TRTRunner


def gpu_temp() -> str:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return f"{pynvml.nvmlDeviceGetTemperature(h, 0)}C"
    except Exception:
        return "n/a (pip install nvidia-ml-py)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/pipeline.yaml")
    ap.add_argument("--batches", default="1,2,4,8")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = torch.device("cuda:0")
    free0 = torch.cuda.mem_get_info(dev)[0]
    m = cfg["models"]
    runners = {"face": TRTRunner(m["face"]["engine"], dev), "vehicle": TRTRunner(m["vehicle"]["engine"], dev)}
    if m["plate"].get("enabled"):
        runners["plate"] = TRTRunner(m["plate"]["engine"], dev)
    free1, total = torch.cuda.mem_get_info(dev)
    rows = []
    print(f"VRAM total {total >> 20} MiB | free before load {free0 >> 20} | free after loading engines {free1 >> 20}")
    for name in ("face", "vehicle"):
        r = runners[name]
        div = 1 if name == "face" else 2
        for g, (H, W) in cfg["shapes"].items():
            for b in [int(x) for x in a.batches.split(",")]:
                shape = (b, 3, H // div, W // div)
                try:
                    r.check_shape(shape)
                except ValueError:
                    continue
                x = torch.rand(shape, device=dev)
                for _ in range(5):
                    r(x)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(a.iters):
                    r(x)
                torch.cuda.synchronize()
                dt = (time.perf_counter() - t0) / a.iters
                free = torch.cuda.mem_get_info(dev)[0] >> 20
                row = {"model": name, "group": g, "batch": b, "shape": f"{shape[2]}x{shape[3]}",
                       "ms_per_batch": round(dt * 1000, 1), "img_per_s": round(b / dt, 2), "free_mib": free}
                rows.append(row)
                print(row)
    if "plate" in runners:
        r, S = runners["plate"], m["plate"]["imgsz"]
        for b in (1, r.max_batch):
            x = torch.rand((b, 3, S, S), device=dev)
            for _ in range(5):
                r(x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(a.iters):
                r(x)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / a.iters
            rows.append({"model": "plate", "batch": b, "shape": f"{S}x{S}",
                         "ms_per_batch": round(dt * 1000, 1), "crops_per_s": round(b / dt, 1)})
            print(rows[-1])
    min_free = min((rw.get("free_mib", 1 << 30) for rw in rows), default=free1 >> 20)
    verdict = "OK" if min_free >= 600 else "TOO TIGHT: rebuild with smaller max batch/shape"
    print(f"\nminimum free VRAM during benchmark: {min_free} MiB -> {verdict}   GPU temp {gpu_temp()}")
    save_fact("bench_engine", {"vram_total_mib": total >> 20, "free_after_load_mib": free1 >> 20,
                               "min_free_mib": min_free, "verdict": verdict, "rows": rows})


if __name__ == "__main__":
    main()
