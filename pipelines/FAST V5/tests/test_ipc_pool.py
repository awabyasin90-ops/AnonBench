"""Unit test for fastblur/ipc_pool.py: SharedMemoryRingPool lifecycle and IPC verification."""
import multiprocessing as mp
import numpy as np
import pytest
from fastblur.ipc_pool import SharedMemoryRingPool


def _child_reader(prefix: str, num_slots: int, batch_size: int, q_ready: mp.Queue, q_done: mp.Queue):
    pool = SharedMemoryRingPool.attach(
        prefix=prefix,
        num_slots=num_slots,
        batch_size=batch_size,
        max_raw_shape=(100, 100),
        letterbox_shape=(100, 100),
        vehicle_shape=(64, 64),
    )
    q_ready.put("ready")
    msg = q_ready.get()
    slot_id = msg["slot_id"]
    # Verify values written by parent
    assert pool.raw_arr[slot_id, 0, 10, 10, 0] == 42
    assert pool.face_arr[slot_id, 0, 5, 5, 1] == 84
    assert pool.veh_arr[slot_id, 0, 2, 2, 2] == 126
    pool.close()
    q_done.put("ok")


def test_shared_memory_ring_pool_lifecycle():
    num_slots = 2
    batch_size = 2
    pool = SharedMemoryRingPool(
        num_slots=num_slots,
        batch_size=batch_size,
        max_raw_shape=(100, 100),
        letterbox_shape=(100, 100),
        vehicle_shape=(64, 64),
        create=True,
    )

    try:
        # Check initial shapes and padding values
        assert pool.raw_arr.shape == (2, 2, 100, 100, 3)
        assert pool.face_arr.shape == (2, 2, 100, 100, 3)
        assert pool.veh_arr.shape == (2, 2, 64, 64, 3)
        assert pool.face_arr[0, 0, 0, 0, 0] == 114
        assert pool.veh_arr[0, 0, 0, 0, 0] == 114

        # Test cross-process reading
        ctx = mp.get_context("spawn")
        q_ready = ctx.Queue()
        q_done = ctx.Queue()

        p = ctx.Process(
            target=_child_reader,
            args=(pool.prefix, num_slots, batch_size, q_ready, q_done),
        )
        p.start()

        assert q_ready.get(timeout=5.0) == "ready"

        # Write data to slot 1
        pool.raw_arr[1, 0, 10, 10, 0] = 42
        pool.face_arr[1, 0, 5, 5, 1] = 84
        pool.veh_arr[1, 0, 2, 2, 2] = 126

        q_ready.put({"slot_id": 1})
        res = q_done.get(timeout=5.0)
        assert res == "ok"
        p.join(timeout=5.0)
        assert p.exitcode == 0

    finally:
        pool.close()
        pool.unlink()


def test_shared_memory_pinning():
    pool = SharedMemoryRingPool(
        num_slots=2,
        batch_size=1,
        max_raw_shape=(100, 100),
        letterbox_shape=(100, 100),
        vehicle_shape=(64, 64),
        create=True,
    )
    try:
        pinned = pool.pin_host_buffers()
        # Pinning may return True if CUDA is present or False if not, but should never crash
        assert isinstance(pinned, bool)
    finally:
        pool.close()
        pool.unlink()
