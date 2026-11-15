"""fastblur/ring_pool.py: Preallocated static memory pool with ring token dispatch."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class BufferSlot:
    slot_id: int
    # Host Pinned Memory views (C-contiguous, zero-copy DMA targets)
    raw_host: np.ndarray             # (B, H_max, W_max, 3) uint8 BGR
    letterbox_host: np.ndarray       # (B, H_lb, W_lb, 3) uint8 BGR padded (114)
    vehicle_host: np.ndarray         # (B, 640, 640, 3) uint8 BGR padded (114)
    # Device Tensors (Pre-allocated VRAM buffers)
    device_uint8_face: torch.Tensor  # (B, H_lb, W_lb, 3) uint8 on cuda:0 (DMA target)
    device_uint8_veh: torch.Tensor   # (B, 640, 640, 3) uint8 on cuda:0 (DMA target)
    device_input: torch.Tensor       # (B, 3, H_lb, W_lb) float16 on cuda:0
    device_downscale: torch.Tensor   # (B, 3, 640, 640) float16 on cuda:0
    # Synchronization Events (Per-slot lifecycle sentinels)
    event_h2d_done: torch.cuda.Event
    event_face_done: torch.cuda.Event
    event_vehicle_done: torch.cuda.Event
    event_plate_done: torch.cuda.Event
    event_d2h_done: torch.cuda.Event
    # Metadata container (Carries item paths, letterbox params, raw image references)
    metadata: list[dict[str, Any]]


class StaticRingPool:
    """Thread-safe ring pool of N pre-allocated slots."""

    def __init__(
        self,
        num_slots: int,
        batch_size: int,
        max_raw_shape: tuple[int, int],      # (H, W) e.g., (1040, 1024)
        letterbox_shape: tuple[int, int],    # (H, W) e.g., (1280, 1280)
        device: torch.device,
        precision: torch.dtype = torch.float16,
    ):
        self.num_slots = num_slots
        self.batch_size = batch_size
        self.device = device
        self.slots: list[BufferSlot] = []
        self._free_queue: queue.Queue[BufferSlot] = queue.Queue(maxsize=num_slots)

        h_raw, w_raw = max_raw_shape
        h_lb, w_lb = letterbox_shape

        for i in range(num_slots):
            # 1. Pinned Host Allocations
            t_raw = torch.empty((batch_size, h_raw, w_raw, 3), dtype=torch.uint8, pin_memory=True)
            t_lb = torch.full((batch_size, h_lb, w_lb, 3), 114, dtype=torch.uint8, pin_memory=True)
            t_veh = torch.full((batch_size, 640, 640, 3), 114, dtype=torch.uint8, pin_memory=True)

            # 2. Fixed Device VRAM Allocations
            d_u8_face = torch.empty((batch_size, h_lb, w_lb, 3), dtype=torch.uint8, device=device)
            d_u8_veh = torch.empty((batch_size, 640, 640, 3), dtype=torch.uint8, device=device)
            d_input = torch.empty((batch_size, 3, h_lb, w_lb), dtype=precision, device=device)
            d_down = torch.empty((batch_size, 3, 640, 640), dtype=precision, device=device)

            slot = BufferSlot(
                slot_id=i,
                raw_host=t_raw.numpy(),
                letterbox_host=t_lb.numpy(),
                vehicle_host=t_veh.numpy(),
                device_uint8_face=d_u8_face,
                device_uint8_veh=d_u8_veh,
                device_input=d_input,
                device_downscale=d_down,
                event_h2d_done=torch.cuda.Event(enable_timing=True),
                event_face_done=torch.cuda.Event(enable_timing=True),
                event_vehicle_done=torch.cuda.Event(enable_timing=True),
                event_plate_done=torch.cuda.Event(enable_timing=True),
                event_d2h_done=torch.cuda.Event(enable_timing=True),
                metadata=[],
            )
            self.slots.append(slot)
            self._free_queue.put(slot)

    def acquire(self, timeout: float | None = None) -> BufferSlot:
        """Acquire a free slot. Blocks if all slots are in-flight (Backpressure trigger)."""
        return self._free_queue.get(block=True, timeout=timeout)

    def release(self, slot: BufferSlot) -> None:
        """Re-enters slot into pool after output serialization finishes."""
        slot.metadata.clear()
        self._free_queue.put(slot)

    @property
    def free_count(self) -> int:
        return self._free_queue.qsize()
