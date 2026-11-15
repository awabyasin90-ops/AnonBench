"""Unicode-safe, atomic image I/O (Windows friendly).

cv2.imread / cv2.imwrite fail silently on non-ASCII Windows paths, so we use
np.fromfile + cv2.imdecode and cv2.imencode + ndarray.tofile instead.
Writes go to "<name>.part" first and are renamed with os.replace, so a crash
never leaves a half-written JPEG that looks finished.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def list_images(root: Path, extensions: list[str], recursive: bool = True) -> list[Path]:
    exts = {e.lower() for e in extensions}
    out: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        with os.scandir(d) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    if recursive:
                        stack.append(Path(e.path))
                elif os.path.splitext(e.name)[1].lower() in exts:
                    out.append(Path(e.path))
    out.sort()
    return out


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)  # BGR uint8 HWC, EXIF orientation applied
    if img is None:
        raise IOError(f"cannot decode image: {path}")
    return img


def write_jpeg_atomic(path: Path, img: np.ndarray, quality: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise IOError(f"cannot encode JPEG: {path}")
    tmp = path.with_name(path.name + ".part")
    buf.tofile(str(tmp))
    os.replace(tmp, path)
    return int(buf.size)


def copy_atomic(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)
    return dst.stat().st_size
