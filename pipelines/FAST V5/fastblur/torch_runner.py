"""fastblur/torch_runner.py — PyTorch runner for non-CUDA platforms.

Allows FAST v4 to run natively on Apple Silicon (Mac M1/M2/M3/M4) via MPS or CPU,
as well as any Linux/Windows machine without TensorRT or NVIDIA hardware.
"""
from __future__ import annotations

from pathlib import Path
import torch
from ultralytics import YOLO


class TorchRunner:
    def __init__(self, model_path: str | Path, device: torch.device):
        self.device = device
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"PyTorch model weights not found at: {self.model_path}")
        yolo = YOLO(str(self.model_path))
        self.model = yolo.model.to(device).eval()
        self.max_batch = 16
        nc = getattr(self.model, "nc", None)
        if nc is None and hasattr(self.model, "names"):
            nc = len(self.model.names)
        self.meta = {"nc": int(nc or 1)}

    def check_shape(self, shape: tuple) -> None:
        pass

    def check_profile(self, B: int, H: int, W: int) -> None:
        pass

    @torch.inference_mode()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: tensor (B, 3, H, W) normalized 0..1 RGB on self.device.
        Returns: raw detection tensor (B, 4 + nc, anchors).
        """
        x = x.to(self.device).float()
        out = self.model(x)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    infer = __call__
