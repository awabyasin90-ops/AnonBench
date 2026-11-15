"""fastblur/scheduler.py: Non-blocking CUDA multi-stream inter-batch inference scheduler."""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .decode import decode_yolo
from .geometry import boxes_to_original, expand_box, letterbox
from .ring_pool import BufferSlot, StaticRingPool
from pathlib import Path

DEFAULT_PLATE_ENGINE = r"c:\Users\opina\Desktop\public_release\FAST_v2\weights\license-plate_fp16_b16.engine"


class GPUMultiStreamScheduler:
    def __init__(
        self,
        config: dict[str, Any],
        runners: dict[str, TRTRunner],
        device: torch.device,
        strategy: str = "fully_concurrent",
        compute_stream_count: int = 2,
    ):
        self.cfg = config
        self.runners = dict(runners)
        self.device = device
        self.strategy = strategy
        self.compute_stream_count = compute_stream_count

        # Plate staging buffer: pre-allocated on device for batch 16
        self.plate_on = bool(config.get("models", {}).get("plate", {}).get("enabled", True))
        if self.plate_on and "plate" not in self.runners:
            plate_eng = config.get("models", {}).get("plate", {}).get("engine", DEFAULT_PLATE_ENGINE)
            if plate_eng and Path(plate_eng).exists():
                self.runners["plate"] = TRTRunner(plate_eng, device)

        # 1. Initialize dedicated CUDA streams
        self.stream_h2d = torch.cuda.Stream(device=device)
        self.stream_d2h = torch.cuda.Stream(device=device)

        if compute_stream_count == 1 or strategy == "interleaved_serial":
            # Unified compute stream (Zero intra-batch SM contention)
            self.stream_face = torch.cuda.Stream(device=device)
            self.stream_vehicle = self.stream_face
            self.stream_plate = self.stream_face
        elif compute_stream_count == 2:
            # Face and Vehicle execute concurrently; Plate runs on Vehicle stream
            self.stream_face = torch.cuda.Stream(device=device)
            self.stream_vehicle = torch.cuda.Stream(device=device)
            self.stream_plate = self.stream_vehicle
        else:
            # 3 Compute Streams (Maximally concurrent)
            self.stream_face = torch.cuda.Stream(device=device)
            self.stream_vehicle = torch.cuda.Stream(device=device)
            self.stream_plate = torch.cuda.Stream(device=device)

        # Plate staging buffer: pre-allocated on device for batch 16
        self.plate_on = bool(config["models"]["plate"].get("enabled", True))
        self.plate_max_batch = int(config["models"]["plate"].get("max_batch", 16))
        # Plate crop input shape: toggleable between (192, 320) and (320, 320) in one line
        use_rect_192x320 = bool(config["models"]["plate"].get("rect_192x320", False))
        self.plate_shape = (192, 320) if use_rect_192x320 else (320, 320)
        self.plate_imgsz = self.plate_shape[0]
        self.plate_dtype = (
            self.runners["plate"].input_dtype if (self.plate_on and "plate" in runners) else torch.float32
        )

        self.device_plate_crops = torch.empty(
            (self.plate_max_batch, 3, self.plate_shape[0], self.plate_shape[1]),
            dtype=self.plate_dtype,
            device=device,
        )

        self.nc = {k: self._nc(k) for k in runners}
        self.aa = bool(config.get("vehicle_antialias", True))

    def _nc(self, k: str) -> int:
        n = self.cfg["models"][k].get("nc")
        if n is None:
            n = getattr(self.runners[k], "meta", {}).get("nc")
        if n is None:
            raise ValueError(f"number of classes unknown for '{k}': set models.{k}.nc or keep the engine .json sidecar")
        return int(n)

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return float(inter / union) if union > 0 else 0.0

    def dispatch_h2d_async(self, slot: BufferSlot) -> None:
        """Launches Host-to-Device memory copy and GPU normalization on stream_h2d."""
        B = len(slot.metadata)
        if B == 0:
            return
        H_in, W_in = slot.metadata[0]["lb"].shape[:2]
        for m in slot.metadata[1:]:
            assert m["lb"].shape[:2] == (H_in, W_in), (
                f"Batch shape homogenization error (Finding 1): item has shape {m['lb'].shape[:2]} "
                f"but batch slot expects {(H_in, W_in)}"
            )

        # -------------------------------------------------------------
        # STAGE 1: Host-to-Device Transfer (Stream 0)
        # -------------------------------------------------------------
        with torch.cuda.stream(self.stream_h2d):
            # 1. Direct Pinned Host-to-Device uint8 DMA (truly non-blocking, zero CPU conversion overhead)
            slot.device_uint8_face[:B, :H_in, :W_in, :].copy_(
                torch.from_numpy(slot.letterbox_host[:B, :H_in, :W_in, :]),
                non_blocking=True,
            )
            slot.device_uint8_veh[:B, :640, :640, :].copy_(
                torch.from_numpy(slot.vehicle_host[:B, :640, :640, :]),
                non_blocking=True,
            )

            # 2. Fast GPU Kernel Normalization on stream_h2d
            slot.device_input[:B, :, :H_in, :W_in].copy_(
                slot.device_uint8_face[:B, :H_in, :W_in, :].permute(0, 3, 1, 2).flip(1).to(slot.device_input.dtype).div_(255.0)
            )
            slot.device_downscale[:B, :, :640, :640].copy_(
                slot.device_uint8_veh[:B, :640, :640, :].permute(0, 3, 1, 2).flip(1).to(slot.device_downscale.dtype).div_(255.0)
            )
            slot.event_h2d_done.record(self.stream_h2d)

    def dispatch_compute_async(self, slot: BufferSlot) -> None:
        """Launches multi-stream compute (Face, Vehicle, Plate, D2H) for a slot awaiting its H2D event."""
        B = len(slot.metadata)
        if B == 0:
            return
        H_in, W_in = slot.metadata[0]["lb"].shape[:2]

        # -------------------------------------------------------------
        # STAGE 2A: Face Detection (Stream 1)
        # -------------------------------------------------------------
        with torch.cuda.stream(self.stream_face):
            self.stream_face.wait_event(slot.event_h2d_done)
            input_face = slot.device_input[:B, :, :H_in, :W_in]
            raw_face = self.runners["face"](input_face)
            mf = self.cfg["models"]["face"]
            face_dets = decode_yolo(
                raw_face,
                self.nc["face"],
                float(mf["conf"]),
                float(mf["iou"]),
                mf.get("classes"),
                int(mf["max_det"]),
            )
            slot.event_face_done.record(self.stream_face)

        # -------------------------------------------------------------
        # STAGE 2B: Vehicle Detection (Stream 2)
        # -------------------------------------------------------------
        with torch.cuda.stream(self.stream_vehicle):
            self.stream_vehicle.wait_event(slot.event_h2d_done)
            input_veh = slot.device_downscale[:B, :, :640, :640]
            raw_veh = self.runners["vehicle"](input_veh)
            mv = self.cfg["models"]["vehicle"]
            classes = [int(c) for c in mv["classes"]] if mv.get("classes") is not None else None
            veh_dets = decode_yolo(
                raw_veh,
                self.nc["vehicle"],
                float(mv["conf"]),
                float(mv["iou"]),
                classes,
                int(mv["max_det"]),
            )
            slot.event_vehicle_done.record(self.stream_vehicle)

        # -------------------------------------------------------------
        # STAGE 3: Plate Detection (Stream 3 / Dependent)
        # -------------------------------------------------------------
        # Synchronize vehicle detection on stream_vehicle
        slot.event_vehicle_done.synchronize()

        plate_jobs: list[tuple[int, int, int, np.ndarray, Any]] = []
        if self.plate_on and "plate" in self.runners:
            mp = self.cfg["models"]["plate"]
            pad = float(mp.get("crop_pad_ratio", 0.10))
            min_px = max(1, int(mp.get("min_vehicle_px", 40)))
            H_p, W_p = self.plate_shape

            for i, it in enumerate(slot.metadata):
                h0, w0 = it["img"].shape[:2]
                vd = veh_dets[i]
                if len(vd) == 0:
                    continue
                vd_np = vd[:, :4].cpu().numpy()
                lbp_v = it.get("lbp_v")
                if lbp_v is not None:
                    orig_vehs = boxes_to_original(vd_np, lbp_v)
                else:
                    vd_np *= 2.0
                    orig_vehs = boxes_to_original(vd_np, it["lbp"])
                for vb in orig_vehs:
                    x1, y1, x2, y2 = expand_box(vb, pad, w0, h0)
                    if min(x2 - x1, y2 - y1) < min_px:
                        continue
                    crop = it["img"][y1:y2, x1:x2]
                    lb, p = letterbox(crop, H_p, W_p)
                    plate_jobs.append((i, x1, y1, lb, p))

        plate_results: list[tuple[tuple[int, int, int, np.ndarray, Any], torch.Tensor]] = []
        with torch.cuda.stream(self.stream_plate):
            if self.plate_on and "plate" in self.runners and len(plate_jobs) > 0:
                mp = self.cfg["models"]["plate"]
                for s in range(0, len(plate_jobs), self.plate_max_batch):
                    chunk = plate_jobs[s : s + self.plate_max_batch]
                    n_c = len(chunk)
                    batch_arr = np.stack([j[3] for j in chunk], axis=0)  # (n_c, S, S, 3) uint8 BGR
                    t_raw = torch.from_numpy(batch_arr)
                    crop_input = t_raw.permute(0, 3, 1, 2).flip(1).to(
                        dtype=self.plate_dtype, device=self.device
                    ).div_(255.0)

                    raw_plate = self.runners["plate"](crop_input)
                    p_dets = decode_yolo(
                        raw_plate,
                        self.nc["plate"],
                        float(mp["conf"]),
                        float(mp["iou"]),
                        mp.get("classes"),
                        int(mp.get("max_det", 50)),
                    )
                    for job, pd in zip(chunk, p_dets):
                        plate_results.append((job, pd))
            slot.event_plate_done.record(self.stream_plate)

        # -------------------------------------------------------------
        # STAGE 4: Device-to-Host Box Transfer (Stream 4 / D2H)
        # -------------------------------------------------------------
        with torch.cuda.stream(self.stream_d2h):
            self.stream_d2h.wait_event(slot.event_face_done)
            self.stream_d2h.wait_event(slot.event_plate_done)
            slot.event_d2h_done.record(self.stream_d2h)

        slot.raw_face_dets = face_dets  # type: ignore[attr-defined]
        slot.raw_veh_dets = veh_dets  # type: ignore[attr-defined]
        slot.raw_plate_results = plate_results  # type: ignore[attr-defined]

    def dispatch_batch_async(self, slot: BufferSlot) -> None:
        """Launches the complete dependency execution graph for a slot (H2D + Compute)."""
        self.dispatch_h2d_async(slot)
        self.dispatch_compute_async(slot)

    def collect_detections(self, slot: BufferSlot) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return one (faces, plates) pair per image in the slot, in ORIGINAL pixel coordinates.

        Each array is float32 with rows [x1, y1, x2, y2, score]; shapes (N, 5) and (M, 5), empty is (0, 5).
        """
        if not hasattr(slot, "raw_face_dets"):
            raise RuntimeError("collect_detections called before dispatch_batch_async on this slot")

        B = len(slot.metadata)
        face_dets = slot.raw_face_dets  # type: ignore[attr-defined]
        plate_results = slot.raw_plate_results  # type: ignore[attr-defined]

        # 1. Postprocess Faces: map from letterbox back to original image space
        out_faces: list[np.ndarray] = []
        for i, item in enumerate(slot.metadata):
            fd = face_dets[i]
            if len(fd) == 0:
                out_faces.append(np.zeros((0, 5), dtype=np.float32))
            else:
                fd_np = fd[:, :5].cpu().numpy()
                fo = boxes_to_original(fd_np, item["lbp"])
                out_faces.append(fo)

        # 2. Postprocess Plates: map from crop space back to original image space and apply NMS
        per_img_plates: list[list[np.ndarray]] = [[] for _ in range(B)]
        for (img_idx, x1, y1, _, p), pd in plate_results:
            if len(pd) > 0:
                pd_np = pd[:, :5].cpu().numpy()
                b = boxes_to_original(pd_np, p)
                b[:, [0, 2]] += x1
                b[:, [1, 3]] += y1
                per_img_plates[img_idx].append(b)

        out_plates: list[np.ndarray] = []
        for v in per_img_plates:
            if not v:
                out_plates.append(np.zeros((0, 5), dtype=np.float32))
                continue
            cat = np.concatenate(v, axis=0)
            kept = []
            for b in sorted(cat, key=lambda x: -x[4]):
                if all(self._box_iou(b[:4], k[:4]) < 0.3 for k in kept):
                    kept.append(b)
            out_plates.append(np.array(kept, dtype=np.float32) if kept else np.zeros((0, 5), dtype=np.float32))

        del slot.raw_face_dets  # type: ignore[attr-defined]
        del slot.raw_plate_results  # type: ignore[attr-defined]

        return list(zip(out_faces, out_plates))
