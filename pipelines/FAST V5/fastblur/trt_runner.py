"""Minimal TensorRT runner (TensorRT 10.x / 11.x Python API, "tensor name" API).

Input and output memory are torch CUDA tensors, so there is no extra host<->device copy
and no pycuda dependency. Import torch BEFORE tensorrt (Windows DLL loading order).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import tensorrt as trt  # noqa: E402  (must come after torch)


def _dtype_map() -> dict:
    m = {trt.float32: torch.float32, trt.float16: torch.float16,
         trt.int32: torch.int32, trt.int8: torch.int8}
    for trt_name, torch_type in (("int64", torch.int64), ("bool", torch.bool),
                                 ("uint8", torch.uint8), ("bfloat16", torch.bfloat16)):
        if hasattr(trt, trt_name):
            m[getattr(trt, trt_name)] = torch_type
    return m


class TRTRunner:
    def __init__(self, engine_path: str | Path, device: torch.device):
        engine_path = Path(engine_path)
        self.device = device
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize {engine_path} (TensorRT version or GPU mismatch? rebuild it)")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"cannot create execution context for {engine_path} (out of GPU memory?)")
        dm = _dtype_map()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        ins = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        outs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        if len(ins) != 1:
            raise RuntimeError(f"expected exactly 1 input, got {ins}")
        self.input_name = ins[0]
        self.output_names = sorted(outs, key=lambda n: (n != "output0", n))  # output0 first
        self.input_dtype = dm[self.engine.get_tensor_dtype(self.input_name)]
        self.output_dtypes = {n: dm[self.engine.get_tensor_dtype(n)] for n in self.output_names}
        mn, opt, mx = self.engine.get_tensor_profile_shape(self.input_name, 0)
        self.min_shape, self.opt_shape, self.max_shape = tuple(mn), tuple(opt), tuple(mx)
        sidecar = engine_path.with_suffix(engine_path.suffix + ".json")
        self.meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}

    @property
    def max_batch(self) -> int:
        return int(self.max_shape[0])

    def check_shape(self, shape: tuple) -> None:
        for d, lo, hi in zip(shape, self.min_shape, self.max_shape):
            if not lo <= d <= hi:
                raise ValueError(f"input shape {shape} outside engine profile min={self.min_shape} max={self.max_shape}")

    def check_profile(self, B: int, H: int, W: int) -> None:
        return self.check_shape((B, 3, H, W))

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: CUDA tensor (B,3,H,W), RGB, 0..1. Returns output0 as a CUDA tensor."""
        shape = tuple(x.shape)
        self.check_shape(shape)
        x = x.to(self.input_dtype).contiguous()
        if not self.context.set_input_shape(self.input_name, shape):
            raise RuntimeError(f"set_input_shape failed for {shape}")
        self.context.set_tensor_address(self.input_name, x.data_ptr())
        outputs = []
        for n in self.output_names:
            t = torch.empty(tuple(self.context.get_tensor_shape(n)),
                            dtype=self.output_dtypes[n], device=self.device)
            self.context.set_tensor_address(n, t.data_ptr())
            outputs.append(t)
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return outputs[0]

    infer = __call__
