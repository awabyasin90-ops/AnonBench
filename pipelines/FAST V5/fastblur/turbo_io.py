"""fastblur/turbo_io.py: SIMD-accelerated in-place JPEG decompression and compression."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from turbojpeg import TurboJPEG, TJPF_BGR, TJFLAG_FASTDCT
    HAS_TURBOJPEG = True
except ImportError:
    HAS_TURBOJPEG = False


class FastImageCodec:
    """Thread-safe codec wrapping native libjpeg-turbo with direct buffer target support,
    with automatic fallback to OpenCV if libjpeg-turbo native DLL is not present."""

    def __init__(self, lib_path: str | None = None):
        self.jpeg = None
        if HAS_TURBOJPEG:
            try:
                self.jpeg = TurboJPEG(lib_path)
            except Exception:
                self.jpeg = None

    def decode_direct(self, raw_bytes: bytes, target_slice: np.ndarray) -> tuple[int, int]:
        """Decodes raw JPEG bytes directly into a pre-allocated host slice."""
        if self.jpeg is not None:
            header = self.jpeg.decode_header(raw_bytes)
            w, h = header[0], header[1]
            if target_slice.shape[0] < h or target_slice.shape[1] < w:
                raise ValueError(f"Buffer slice {target_slice.shape} insufficient for image {w}x{h}")
            self.jpeg.decode(
                raw_bytes,
                target_slice[:h, :w, :],
                pixel_format=TJPF_BGR,
                flags=TJFLAG_FASTDCT,
            )
            return h, w
        else:
            img = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cannot decode image bytes")
            h, w = img.shape[:2]
            if target_slice.shape[0] < h or target_slice.shape[1] < w:
                raise ValueError(f"Buffer slice {target_slice.shape} insufficient for image {w}x{h}")
            target_slice[:h, :w, :] = img
            return h, w

    def encode_to_disk_atomic(self, path: Path, img: np.ndarray, quality: int = 95) -> int:
        """Compresses BGR array to JPEG and commits via atomic file replacement."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.part")
        if self.jpeg is not None:
            buf = self.jpeg.encode(img, quality=quality, pixel_format=TJPF_BGR, flags=TJFLAG_FASTDCT)
            with open(tmp_path, "wb") as f:
                f.write(buf)
            n_bytes = len(buf)
        else:
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
            if not ok:
                raise ValueError(f"cannot encode JPEG: {path}")
            with open(tmp_path, "wb") as f:
                f.write(buf.tobytes())
            n_bytes = int(buf.size)

        retries = 0
        for attempt in range(5):
            try:
                os.replace(tmp_path, path)
                break
            except (PermissionError, OSError) as e:
                retries += 1
                if attempt == 4:
                    raise
                import time
                time.sleep(0.100)
        return n_bytes, retries
