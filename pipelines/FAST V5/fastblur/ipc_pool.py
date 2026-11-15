"""fastblur/ipc_pool.py: Zero-Copy SharedMemory ring buffer pool for multi-process FAST v2.

Allocates and manages named SharedMemory segments for raw frames and letterboxed detector
inputs, allowing Process A (Decoders), Process B (GPU TRT), and Process C (Writers) to share
huge image arrays with zero serialization and zero copy overhead.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging
import os
import uuid
from multiprocessing import shared_memory
from typing import Any

import numpy as np

logger = logging.getLogger("FAST_v2.IPC")

_CUDART: ctypes.CDLL | None = None


def _get_cudart() -> ctypes.CDLL | None:
    """Discovers and caches the CUDA runtime library from PyTorch or system path."""
    global _CUDART
    if _CUDART is not None:
        return _CUDART

    # 1. Try PyTorch's bundled cudart
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.exists(torch_lib):
            for fname in os.listdir(torch_lib):
                if fname.startswith("cudart64_") and fname.endswith(".dll"):
                    _CUDART = ctypes.CDLL(os.path.join(torch_lib, fname))
                    return _CUDART
    except Exception:
        pass

    # 2. Try standard system library lookup
    for name in ("cudart64_12.dll", "cudart64_110.dll", "cudart64_118.dll"):
        try:
            _CUDART = ctypes.CDLL(name)
            return _CUDART
        except Exception:
            continue
    return None


class SharedMemoryRingPool:
    """Pre-allocated ring pool of N slots stored in shared memory for zero-copy IPC."""

    def __init__(
        self,
        num_slots: int,
        batch_size: int,
        max_raw_shape: tuple[int, int] = (1040, 1040),
        letterbox_shape: tuple[int, int] = (1040, 1040),
        vehicle_shape: tuple[int, int] = (640, 640),
        prefix: str | None = None,
        create: bool = True,
    ):
        self.num_slots = num_slots
        self.batch_size = batch_size
        self.max_raw_shape = max_raw_shape
        self.letterbox_shape = letterbox_shape
        self.vehicle_shape = vehicle_shape
        self.is_owner = create
        self._pinned = False

        if prefix is None:
            prefix = f"fastblur_{uuid.uuid4().hex[:8]}"
        self.prefix = prefix

        self.name_raw = f"{prefix}_raw"
        self.name_face = f"{prefix}_face"
        self.name_veh = f"{prefix}_veh"

        self.shape_raw = (num_slots, batch_size, max_raw_shape[0], max_raw_shape[1], 3)
        self.shape_face = (num_slots, batch_size, letterbox_shape[0], letterbox_shape[1], 3)
        self.shape_veh = (num_slots, batch_size, vehicle_shape[0], vehicle_shape[1], 3)

        self.size_raw = int(np.prod(self.shape_raw))
        self.size_face = int(np.prod(self.shape_face))
        self.size_veh = int(np.prod(self.shape_veh))

        if create:
            # Clean up any stale segments with identical names if any exist
            for name in (self.name_raw, self.name_face, self.name_veh):
                try:
                    stale = shared_memory.SharedMemory(name=name)
                    stale.close()
                    stale.unlink()
                except Exception:
                    pass

            self.shm_raw = shared_memory.SharedMemory(name=self.name_raw, create=True, size=self.size_raw)
            self.shm_face = shared_memory.SharedMemory(name=self.name_face, create=True, size=self.size_face)
            self.shm_veh = shared_memory.SharedMemory(name=self.name_veh, create=True, size=self.size_veh)
        else:
            self.shm_raw = shared_memory.SharedMemory(name=self.name_raw, create=False)
            self.shm_face = shared_memory.SharedMemory(name=self.name_face, create=False)
            self.shm_veh = shared_memory.SharedMemory(name=self.name_veh, create=False)

        # Map shared memory buffers as contiguous NumPy arrays
        self.raw_arr = np.ndarray(self.shape_raw, dtype=np.uint8, buffer=self.shm_raw.buf)
        self.face_arr = np.ndarray(self.shape_face, dtype=np.uint8, buffer=self.shm_face.buf)
        self.veh_arr = np.ndarray(self.shape_veh, dtype=np.uint8, buffer=self.shm_veh.buf)

        if create:
            # Initialize letterbox backgrounds to standard YOLO gray padding value 114
            self.face_arr.fill(114)
            self.veh_arr.fill(114)

    @classmethod
    def attach(
        cls,
        prefix: str,
        num_slots: int,
        batch_size: int,
        max_raw_shape: tuple[int, int] = (1040, 1040),
        letterbox_shape: tuple[int, int] = (1040, 1040),
        vehicle_shape: tuple[int, int] = (640, 640),
    ) -> SharedMemoryRingPool:
        """Attaches to an existing shared memory ring pool from a child worker process."""
        return cls(
            num_slots=num_slots,
            batch_size=batch_size,
            max_raw_shape=max_raw_shape,
            letterbox_shape=letterbox_shape,
            vehicle_shape=vehicle_shape,
            prefix=prefix,
            create=False,
        )

    def pin_host_buffers(self) -> bool:
        """Page-locks the shared memory buffers in the GPU worker via cudaHostRegister.

        Enables truly non-blocking asynchronous DMA copies across the PCIe bus directly
        from the shared memory buffers to GPU VRAM without CPU staging buffers.
        """
        cudart = _get_cudart()
        if cudart is None:
            logger.warning("cudart DLL not found; skipping cudaHostRegister page-locking")
            return False

        try:
            # cudaHostRegister(void *ptr, size_t size, unsigned int flags=0)
            for arr in (self.face_arr, self.veh_arr):
                ptr = arr.ctypes.data
                nbytes = arr.nbytes
                ret = cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(0))
                if ret != 0:
                    logger.warning("cudaHostRegister returned error code %d", ret)
                    return False
            self._pinned = True
            logger.info("Successfully page-locked SharedMemory buffers for high-speed PCIe DMA")
            return True
        except Exception as e:
            logger.warning("Failed to pin SharedMemory buffers with cudaHostRegister: %s", e)
            return False

    def unpin_host_buffers(self) -> None:
        """Unregisters page-locked shared memory buffers before shutdown."""
        if not self._pinned:
            return
        cudart = _get_cudart()
        if cudart is not None:
            try:
                for arr in (self.face_arr, self.veh_arr):
                    cudart.cudaHostUnregister(ctypes.c_void_p(arr.ctypes.data))
                self._pinned = False
            except Exception:
                pass

    def close(self) -> None:
        """Closes local process handles to the shared memory segments."""
        self.unpin_host_buffers()
        for shm in (self.shm_raw, self.shm_face, self.shm_veh):
            try:
                shm.close()
            except Exception:
                pass

    def unlink(self) -> None:
        """Unlinks shared memory segments from the operating system (called by owner)."""
        if not self.is_owner:
            return
        for shm in (self.shm_raw, self.shm_face, self.shm_veh):
            try:
                shm.unlink()
            except Exception:
                pass
        logger.info("SharedMemory segments unlinked cleanly from OS")
