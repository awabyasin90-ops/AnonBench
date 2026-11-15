"""Three-stage producer/consumer pipeline — FAST v4.

  [decode pool, CPU threads]  read JPEG -> letterbox (uint8)            (bounded look-ahead window)
            |
  [main thread, GPU]          pinned uint8 batch -> GPU -> normalize -> face engine
                              -> GPU half-res resize -> vehicle engine -> plate engine (crops)
                              -> NMS on GPU -> boxes to CPU -> map to original coordinates
            |
  [writer pool, CPU threads]  blur in place (v4 ellipse + feather) -> single JPEG encode
                              -> atomic write -> manifest line + privacy report telemetry
                              (bounded by a semaphore = backpressure, RAM stays flat)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .anonymize import anonymize              # v3 compat (kept, called if render_cfg is None)
from .render_v4 import RenderConfig, BlurStats, render_detections
from .privacy_report import ReportCollector, ReportGenerator, FrameRecord
from .decode import decode_yolo
from .geometry import LetterboxParams, boxes_to_original, choose_group, expand_box, letterbox
from .imageio import copy_atomic, list_images, read_image, write_jpeg_atomic

log = logging.getLogger("fastblur")
EMPTY = np.zeros((0, 5), np.float32)
INTERP = {"linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA}


def update_readme_status(status_title: str, details: dict | None = None) -> None:
    """Dynamically updates the live pipeline status section in README.md."""
    try:
        candidates = [
            Path(__file__).resolve().parent.parent.parent / "README.md",
            Path(__file__).resolve().parent.parent / "README.md",
            Path.cwd() / "README.md",
        ]
        readme_path = next((p for p in candidates if p.exists()), None)
        if readme_path is None:
            return
        content = readme_path.read_text(encoding="utf-8")
        start_tag = "<!-- PIPELINE_STATUS_START -->"
        end_tag = "<!-- PIPELINE_STATUS_END -->"
        if start_tag not in content or end_tag not in content:
            return
        detail_lines = []
        if details:
            for k, v in details.items():
                detail_lines.append(f"* **{k}**: {v}")
        details_str = "\n".join(detail_lines)
        block = f"{start_tag}\n### {status_title}\n{details_str}\n{end_tag}"
        prefix = content.split(start_tag)[0]
        suffix = content.split(end_tag)[1]
        readme_path.write_text(prefix + block + suffix, encoding="utf-8")
    except Exception:
        pass


def resolve_model_path(path_str: str | None) -> str | None:
    """Resolve model weights path whether relative to cwd, core/, or core/weights/."""
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.exists():
        return str(p)
    core_dir = Path(__file__).resolve().parent.parent
    for cand in [core_dir / p, core_dir / "weights" / p.name, Path.cwd() / "core" / p]:
        if cand.exists():
            return str(cand)
    return path_str


@dataclass
class Item:
    idx: int
    src: Path
    dst: Path
    img: np.ndarray | None = None          # original, BGR uint8 (H0, W0, 3)
    h0: int = 0
    w0: int = 0
    lb: np.ndarray | None = None           # face-model letterbox, BGR uint8 (H, W, 3)
    lbp: LetterboxParams | None = None
    lb_v: np.ndarray | None = None         # vehicle letterbox (only if vehicle_input == "cpu")
    lbp_v: LetterboxParams | None = None
    group: str = ""
    error: str | None = None


def prepare_item(idx: int, src: Path, dst: Path, shapes: dict, long_side: int,
                 vehicle_cpu: bool, interp: int) -> Item:
    it = Item(idx, src, dst)
    try:
        img = read_image(src)
        h0, w0 = img.shape[:2]
        it.h0, it.w0 = h0, w0
        g = choose_group(h0, w0, shapes, long_side)
        H, W = shapes[g]
        it.lb, it.lbp = letterbox(img, H, W, interp)
        if vehicle_cpu:
            it.lb_v, it.lbp_v = letterbox(img, H // 2, W // 2, interp)
        it.img, it.group = img, g
    except Exception as e:  # corrupt file etc. -> recorded, never crashes the run
        it.error = f"{type(e).__name__}: {e}"
    return it


def write_task(
    src: Path,
    dst: Path,
    img: np.ndarray,
    faces: np.ndarray,
    plates: np.ndarray,
    vehicles: np.ndarray | None,
    blur_cfg: dict,
    quality: int,
    copy_if_empty: bool,
    render_cfg: RenderConfig | None = None,
) -> dict:
    t0 = time.perf_counter()
    stats = BlurStats()
    if copy_if_empty and len(faces) == 0 and len(plates) == 0:
        size, mode = copy_atomic(src, dst), "copy"
    else:
        if render_cfg is not None:
            stats = render_detections(img, faces, plates, vehicles, render_cfg)
        else:
            anonymize(img, faces, plates, blur_cfg)
            stats.faces_rendered = len(faces)
            stats.plates_rendered = len(plates)
            stats.face_mask_type = "rect"
            stats.plate_mask_type = "rect"
            stats.blur_method = blur_cfg.get("method", "adaptive")
        size, mode = write_jpeg_atomic(dst, img, quality), "encode"

    return {
        "bytes": size,
        "mode": mode,
        "write_ms": round((time.perf_counter() - t0) * 1000, 2),
        "blurred_area_px": stats.blurred_area_px,
        "face_mask": stats.face_mask_type,
        "plate_mask": stats.plate_mask_type,
        "blur_method": stats.blur_method,
    }


class Detector:
    def __init__(self, cfg: dict, device: torch.device, runners: dict | None = None):
        self.cfg = cfg
        self.device = device
        m = cfg["models"]
        self.plate_on = bool(m["plate"].get("enabled"))
        if runners is None:
            backend = cfg.get("inference_backend", "auto")
            use_trt = False
            if backend == "tensorrt":
                use_trt = True
            elif backend == "torch":
                use_trt = False
            else:  # auto
                use_trt = (device.type == "cuda")
                if use_trt:
                    try:
                        import tensorrt  # noqa: F401
                    except ImportError:
                        use_trt = False

            if use_trt:
                from .trt_runner import TRTRunner
                runners = {"face": TRTRunner(resolve_model_path(m["face"]["engine"]), device),
                           "vehicle": TRTRunner(resolve_model_path(m["vehicle"]["engine"]), device)}
                if self.plate_on:
                    runners["plate"] = TRTRunner(resolve_model_path(m["plate"]["engine"]), device)
            else:
                from .torch_runner import TorchRunner
                runners = {"face": TorchRunner(resolve_model_path(m["face"]["pt"]), device),
                           "vehicle": TorchRunner(resolve_model_path(m["vehicle"]["pt"]), device)}
                if self.plate_on:
                    runners["plate"] = TorchRunner(resolve_model_path(m["plate"]["pt"]), device)
        self.r = runners
        self.nc = {k: self._nc(k) for k in runners}
        self.vehicle_cpu = cfg.get("vehicle_input", "gpu_half") == "cpu"
        self.aa = bool(cfg.get("vehicle_antialias", True))
        self.interp = INTERP[cfg.get("resize_interp", "linear")]
        self.use_pin = device.type == "cuda"
        self.sync = bool(cfg.get("profile_timing", False)) and device.type == "cuda"
        self._buffers: dict[str, torch.Tensor] = {}
        self.timing: dict[str, float] = defaultdict(float)
        self.plate_cap = min(int(m["plate"].get("max_batch", 1)), self.r["plate"].max_batch) if self.plate_on else 0

    def _nc(self, k: str) -> int:
        n = self.cfg["models"][k].get("nc")
        if n is None:
            n = getattr(self.r[k], "meta", {}).get("nc")
        if n is None:
            raise ValueError(f"number of classes unknown for '{k}': set models.{k}.nc or keep the engine .json sidecar")
        return int(n)

    # ---------- helpers ----------
    def _tick(self, name: str, t0: float) -> float:
        if self.sync:
            torch.cuda.synchronize(self.device)
        t = time.perf_counter()
        self.timing[name] += t - t0
        return t

    def _upload(self, key: str, arrays: list[np.ndarray], capacity: int) -> torch.Tensor:
        """uint8 BGR HWC arrays (same shape) -> CUDA float RGB BCHW 0..1 (contiguous)."""
        n = len(arrays)
        h, w, c = arrays[0].shape
        buf = self._buffers.get(key)
        if buf is None or buf.shape[0] < max(n, capacity) or tuple(buf.shape[1:]) != (h, w, c):
            buf = torch.empty((max(n, capacity), h, w, c), dtype=torch.uint8, pin_memory=self.use_pin)
            self._buffers[key] = buf
        view = buf[:n]
        dst = view.numpy()
        for i, a in enumerate(arrays):
            dst[i] = a
        g = view.to(self.device, non_blocking=self.use_pin)
        x = g.permute(0, 3, 1, 2).flip(1).float().div_(255.0)  # BGR->RGB, 0..1
        return x.contiguous()

    def _downsample_gpu(self, face_tensor: torch.Tensor) -> torch.Tensor:
        """GPU half-res downsample for vehicle detection."""
        return F.interpolate(face_tensor, scale_factor=0.5, mode="area" if self.aa else "bilinear",
                             recompute_scale_factor=False)

    def _decode(self, k: str, raw: torch.Tensor) -> list[np.ndarray]:
        mc = self.cfg["models"][k]
        dets = decode_yolo(
            raw,
            self.nc[k],
            float(mc["conf"]),
            float(mc["iou"]),
            mc.get("classes"),
            int(mc.get("max_det", 300)),
        )
        return [d[:, :5].cpu().numpy() for d in dets]  # x1, y1, x2, y2, conf

    def _veh_to_orig(self, v: np.ndarray, it: Item) -> np.ndarray:
        if not len(v):
            return EMPTY
        if self.vehicle_cpu:
            return boxes_to_original(v, it.lbp_v)
        v2 = v.copy()
        v2[:, :4] *= 2.0  # half-res letterbox -> face letterbox
        return boxes_to_original(v2, it.lbp)

    # ---------- profile check & warmup ----------
    def check_profiles(self, shapes: dict[str, tuple[int, int]]) -> None:
        B = int(self.cfg["batch_size"])
        for g, (H, W) in shapes.items():
            self.r["face"].check_profile(B, H, W)
            self.r["vehicle"].check_profile(B, H // 2, W // 2)
        if self.plate_on and self.plate_cap:
            pw, ph = int(self.cfg["models"]["plate"]["imgsz_w"]), int(self.cfg["models"]["plate"]["imgsz_h"])
            self.r["plate"].check_profile(self.plate_cap, ph, pw)

    def warmup(self, shapes: dict[str, tuple[int, int]], n: int = 3) -> None:
        log.info("warming up engines (%d passes per group)...", n)
        B = int(self.cfg["batch_size"])
        for g, (H, W) in shapes.items():
            dummy = [np.zeros((H, W, 3), np.uint8) for _ in range(B)]
            for _ in range(n):
                t_face = self._upload(f"warm_face_{g}", dummy, B)
                self.r["face"].infer(t_face)
                t_veh = self._downsample_gpu(t_face)
                self.r["vehicle"].infer(t_veh)
        if self.plate_on and self.plate_cap:
            pw, ph = int(self.cfg["models"]["plate"]["imgsz_w"]), int(self.cfg["models"]["plate"]["imgsz_h"])
            dummy_p = [np.zeros((ph, pw, 3), np.uint8) for _ in range(self.plate_cap)]
            for _ in range(n):
                t_pl = self._upload("warm_plate", dummy_p, self.plate_cap)
                self.r["plate"].infer(t_pl)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        log.info("warmup complete")

    # ---------- detection ----------
    def detect(self, items: list[Item]) -> list[dict]:
        B = len(items)
        out = [{"faces": EMPTY, "vehicles": EMPTY, "plates": EMPTY} for _ in items]
        valid_idx = [i for i, it in enumerate(items) if it.img is not None and it.error is None]
        if not valid_idx:
            return out

        t0 = time.perf_counter()
        sub = [items[i] for i in valid_idx]
        cap = int(self.cfg["batch_size"])
        t_face = self._upload(f"face_{sub[0].group}", [it.lb for it in sub], cap)
        t0 = self._tick("upload_face", t0)

        # 1. Face inference
        pred_face = self.r["face"].infer(t_face)
        t0 = self._tick("infer_face", t0)
        face_boxes = self._decode("face", pred_face)
        t0 = self._tick("decode_face", t0)

        # 2. Vehicle inference
        if self.vehicle_cpu:
            t_veh = self._upload("veh_cpu", [it.lb_v for it in sub], cap)
            t0 = self._tick("upload_veh", t0)
        else:
            t_veh = self._downsample_gpu(t_face)
            t0 = self._tick("downsample_gpu", t0)
        pred_veh = self.r["vehicle"].infer(t_veh)
        t0 = self._tick("infer_veh", t0)
        veh_boxes = self._decode("vehicle", pred_veh)
        t0 = self._tick("decode_veh", t0)

        # 3. Map boxes back to original coordinate system
        mapped_faces, mapped_vehs = [], []
        for it, fb, vb in zip(sub, face_boxes, veh_boxes):
            mf = boxes_to_original(fb, it.lbp) if len(fb) else EMPTY
            mv = self._veh_to_orig(vb, it)
            mapped_faces.append(mf)
            mapped_vehs.append(mv)
        t0 = self._tick("map_coords", t0)

        # 4. Plate inference on vehicle crops
        mapped_plates = [EMPTY for _ in sub]
        if self.plate_on and self.plate_cap:
            mapped_plates = self._detect_plates(sub, mapped_vehs)
            t0 = self._tick("plates_total", t0)

        for i, mf, mv, mp in zip(valid_idx, mapped_faces, mapped_vehs, mapped_plates):
            out[i]["faces"] = mf
            out[i]["vehicles"] = mv
            out[i]["plates"] = mp
        return out

    def _detect_plates(self, sub: list[Item], mapped_vehs: list[np.ndarray]) -> list[np.ndarray]:
        t0 = time.perf_counter()
        crops_info = []  # (sub_idx, crop_img, crop_x1, crop_y1, crop_w, crop_h, lbp)
        pw = int(self.cfg["models"]["plate"].get("imgsz_w", self.cfg["models"]["plate"].get("imgsz", 320)))
        ph = int(self.cfg["models"]["plate"].get("imgsz_h", self.cfg["models"]["plate"].get("imgsz", 192)))
        pad = float(self.cfg["models"]["plate"].get("crop_pad", 0.05))
        for si, (it, vboxes) in enumerate(zip(sub, mapped_vehs)):
            if len(vboxes) == 0:
                continue
            h0, w0 = it.h0, it.w0
            for vb in vboxes:
                x1, y1, x2, y2 = expand_box(vb[:4], pad, w0, h0)
                cw, ch = x2 - x1, y2 - y1
                if cw < 8 or ch < 8:
                    continue
                crop = it.img[y1:y2, x1:x2]
                clb, clbp = letterbox(crop, ph, pw, self.interp)
                crops_info.append((si, clb, x1, y1, cw, ch, clbp))
        t0 = self._tick("plate_crops_prep", t0)
        if not crops_info:
            return [EMPTY for _ in sub]

        plate_iou = float(self.cfg["models"]["plate"]["iou"])
        out_plates = [[] for _ in sub]
        cap = self.plate_cap
        for chunk_start in range(0, len(crops_info), cap):
            chunk = crops_info[chunk_start: chunk_start + cap]
            t_pl = self._upload("plate_crops", [c[1] for c in chunk], cap)
            pred = self.r["plate"].infer(t_pl)
            boxes = self._decode("plate", pred)
            for (si, _, cx1, cy1, cw, ch, clbp), pb in zip(chunk, boxes):
                if len(pb) == 0:
                    continue
                orig = boxes_to_original(pb, clbp)
                orig[:, 0] += cx1
                orig[:, 2] += cx1
                orig[:, 1] += cy1
                orig[:, 3] += cy1
                out_plates[si].append(orig)

        # Merge & NMS within image
        for si in range(len(sub)):
            if not out_plates[si]:
                out_plates[si] = EMPTY
            else:
                all_p = np.vstack(out_plates[si])
                if len(all_p) > 1:
                    keep = []
                    order = np.argsort(-all_p[:, 4])
                    for idx in order:
                        b = all_p[idx]
                        if not any(self._box_iou(b, all_p[k]) > plate_iou for k in keep):
                            keep.append(idx)
                    all_p = all_p[keep]
                out_plates[si] = all_p
        return out_plates

    @staticmethod
    def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0


class Pipeline:
    def __init__(self, cfg: dict, device: torch.device | None = None, runners: dict | None = None,
                 limit: int | None = None, run_name: str | None = None):
        self.cfg = cfg
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device
        self.limit = limit
        self.run_name = run_name or time.strftime("run_%Y%m%d_%H%M%S")
        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.jsonl"
        self.shapes = {k: tuple(v) for k, v in cfg["shapes"].items()}
        self.det = Detector(cfg, self.device, runners)
        self.render_cfg = RenderConfig.from_dict(cfg) if ("render" in cfg or "blur" in cfg) else None
        self.report_collector = ReportCollector()
        self.encode_count = 0
        self._lock = threading.Lock()
        self._counts = defaultdict(int)
        self._total_faces = 0
        self._total_plates = 0
        self._bytes = 0
        self._write_ms = 0.0
        self._unflushed = 0

    def _load_done(self) -> set[str]:
        done = set()
        if self.manifest_path.exists():
            with open(self.manifest_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn last line after a crash
                    if rec.get("status") == "ok":
                        done.add(rec["rel"])
        return done

    def _record(self, rec: dict) -> None:
        with self._lock:
            self._mf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._counts[rec["status"]] += 1
            self._unflushed += 1
            if self._unflushed >= 100:
                self._mf.flush()
                self._unflushed = 0

    def _parse_session_camera(self, rel: str) -> tuple[str, str, int]:
        parts = Path(rel).parts
        if len(parts) >= 3:
            session = parts[0]
            camera = parts[1]
        elif len(parts) == 2:
            if parts[0] in ("front_camera", "front", "stereo", "depth"):
                session = self.cfg.get("session_name", "cameras_only-2")
                camera = parts[0]
            else:
                session = parts[0]
                camera = "camera"
        else:
            session = self.cfg.get("session_name", "cameras_only-2")
            camera = "front_camera"

        stem = Path(rel).stem
        ts_us = int(stem) if stem.isdigit() else 0
        return session, camera, ts_us

    @staticmethod
    def _format_boxes(boxes: np.ndarray) -> list[list[float]]:
        if len(boxes) == 0:
            return []
        res = []
        for b in boxes:
            x1 = float(np.round(b[0], 1))
            y1 = float(np.round(b[1], 1))
            x2 = float(np.round(b[2], 1))
            y2 = float(np.round(b[3], 1))
            conf = float(b[4]) if len(b) > 4 else 1.0
            res.append([x1, y1, x2, y2, conf])
        return res

    def _on_written(self, fut, it: Item, d: dict, rel: str) -> None:
        self._sem.release()
        rec = {
            "rel": rel,
            "faces": len(d["faces"]),
            "plates": len(d["plates"]),
            "vehicles": len(d["vehicles"]),
            "face_boxes": self._format_boxes(d["faces"]),
            "plate_boxes": self._format_boxes(d["plates"]),
        }
        with self._lock:
            self._total_faces += len(d["faces"])
            self._total_plates += len(d["plates"])

        session, camera, ts_us = self._parse_session_camera(rel)
        try:
            r = fut.result()
            rec.update(status="ok", **r)
            with self._lock:
                self._bytes += r["bytes"]
                self._write_ms += r["write_ms"]
                if r.get("mode") == "encode":
                    self.encode_count += 1

            # Accumulate report record
            tot_area = it.h0 * it.w0
            blurred_px = r.get("blurred_area_px", 0)
            fraction = (blurred_px / tot_area) if tot_area > 0 else 0.0

            frame_rec = FrameRecord(
                filename=rel,
                session=session,
                camera=camera,
                timestamp_us=ts_us,
                status="ok",
                faces_detected=len(d["faces"]),
                plates_detected=len(d["plates"]),
                vehicles_detected=len(d["vehicles"]),
                face_boxes=self._format_boxes(d["faces"]),
                plate_boxes=self._format_boxes(d["plates"]),
                vehicle_boxes=self._format_boxes(d["vehicles"]),
                face_mask_type=r.get("face_mask", "ellipse"),
                plate_mask_type=r.get("plate_mask", "feathered_rect"),
                blur_method=r.get("blur_method", "adaptive"),
                blurred_area_px=blurred_px,
                total_area_px=tot_area,
                blurred_area_fraction=round(fraction, 5),
                blur_time_ms=r.get("write_ms", 0.0),
                encode_time_ms=r.get("write_ms", 0.0),
                file_size_bytes=r.get("bytes", 0),
                frame_w=it.w0,
                frame_h=it.h0,
            )
            self.report_collector.add_record(frame_rec)
        except Exception as e:
            rec.update(status="error", error=f"write: {type(e).__name__}: {e}")
            self.report_collector.record_error(
                filename=rel,
                session=session,
                camera=camera,
                error_type="write_failure",
                detail=f"write: {type(e).__name__}: {e}",
            )
        self._record(rec)

    def _process(self, batch: list[Item]) -> None:
        dets = self.det.detect(batch)
        cfg = self.cfg
        in_dir = Path(cfg["input_dir"]) if cfg.get("input_dir") else None
        for it, d in zip(batch, dets):
            t0 = time.perf_counter()
            self._sem.acquire()  # backpressure: blocks when writers are behind
            self.det.timing["wait_writer"] += time.perf_counter() - t0
            if in_dir and it.src.is_relative_to(in_dir):
                rel = it.src.relative_to(in_dir).as_posix()
            else:
                rel = it.dst.name
            fut = self._writers.submit(
                write_task,
                it.src,
                it.dst,
                it.img,
                d["faces"],
                d["plates"],
                d.get("vehicles"),
                cfg["blur"],
                int(cfg["jpeg_quality"]),
                bool(cfg.get("copy_if_no_detections", False)),
                self.render_cfg,
            )
            fut.add_done_callback(lambda f, it=it, d=d, rel=rel: self._on_written(f, it, d, rel))
            it.img = it.lb = it.lb_v = None  # writer task holds reference

    def run(self, tasks: list[tuple[Path, Path]] | None = None) -> dict:
        cfg = self.cfg
        cv2.setNumThreads(int(cfg["opencv_threads"]))
        torch.set_num_threads(2)
        if cfg.get("cpu_affinity"):
            import psutil
            psutil.Process().cpu_affinity(list(cfg["cpu_affinity"]))
        in_dir = Path(cfg["input_dir"]) if cfg.get("input_dir") else None
        out_dir = Path(cfg["output_dir"]) if cfg.get("output_dir") else None
        if tasks is None:
            if in_dir is None or out_dir is None:
                raise ValueError("input_dir and output_dir must be specified when tasks is None")
            if in_dir.resolve() == out_dir.resolve():
                raise ValueError("output_dir must differ from input_dir")
            files = list_images(in_dir, cfg["extensions"], cfg.get("recursive", True))
            done = self._load_done() if cfg.get("skip_existing", True) else set()
            tasks = []
            for src in files:
                rel = src.relative_to(in_dir).as_posix()
                dst = out_dir / rel
                if cfg.get("skip_existing", True) and (rel in done or dst.exists()):
                    continue
                tasks.append((src, dst))
            if self.limit:
                tasks = tasks[: self.limit]
            log.info("found %d images, %d already done, %d to process", len(files), len(files) - len(tasks), len(tasks))
        else:
            if self.limit:
                tasks = tasks[: self.limit]
            log.info("processing %d explicit tasks", len(tasks))

        self.det.check_profiles(self.shapes)
        self.det.warmup(self.shapes)
        self.det.timing.clear()

        B = int(cfg["batch_size"])
        update_readme_status(
            "🟡 Pipeline Status: PROCESSING (Starting)",
            {
                "Queued Tasks": f"{len(tasks)} images",
                "Device": f"{self.device.type.upper()}",
                "Batch Size": str(B),
                "Output Directory": str(out_dir) if out_dir else "outputs",
            }
        )
        window = max(B, int(cfg["prefetch_batches"]) * B)
        long_side = int(cfg["models"]["face"]["long_side"])
        interp = INTERP[cfg.get("resize_interp", "linear")]
        vehicle_cpu = cfg.get("vehicle_input", "gpu_half") == "cpu"
        self._sem = threading.BoundedSemaphore(int(cfg["max_pending_writes"]))
        decoders = ThreadPoolExecutor(int(cfg["decode_threads"]), thread_name_prefix="decode")
        self._writers = ThreadPoolExecutor(int(cfg["writer_threads"]), thread_name_prefix="write")
        self._mf = open(self.manifest_path, "a", encoding="utf-8")

        def ordered_futures():
            pending = deque()
            for idx, (src, dst) in enumerate(tasks):
                pending.append(decoders.submit(prepare_item, idx, src, dst, self.shapes,
                                               long_side, vehicle_cpu, interp))
                if len(pending) >= window:
                    yield pending.popleft()
            while pending:
                yield pending.popleft()

        t_start = last_log = time.perf_counter()
        n_in = 0
        groups: dict[str, list[Item]] = defaultdict(list)
        try:
            for fut in ordered_futures():
                t0 = time.perf_counter()
                it = fut.result()
                self.det.timing["wait_decode"] += time.perf_counter() - t0
                n_in += 1
                if it.error:
                    rel_str = it.src.relative_to(in_dir).as_posix() if (in_dir and it.src.is_relative_to(in_dir)) else it.src.name
                    sess, cam, _ = self._parse_session_camera(rel_str)
                    self._record({"rel": rel_str, "status": "error", "error": f"read: {it.error}"})
                    self.report_collector.record_error(
                        filename=rel_str,
                        session=sess,
                        camera=cam,
                        error_type="corrupt_input",
                        detail=f"read: {it.error}",
                    )
                    continue
                g = groups[it.group]
                g.append(it)
                if len(g) >= B:
                    self._process(g)
                    groups[it.group] = []
                now = time.perf_counter()
                if now - last_log >= 10:
                    last_log = now
                    self._log_progress(n_in, len(tasks), now - t_start)
            for g in groups.values():
                if g:
                    self._process(g)
        finally:
            self._writers.shutdown(wait=True)
            decoders.shutdown(wait=True)
            self._mf.close()

        # Enforce single-encode invariant
        n_ok = self._counts["ok"]
        if not cfg.get("copy_if_no_detections", False):
            assert self.encode_count == n_ok, (
                f"Single encode guarantee violation: encode_count={self.encode_count} != ok={n_ok}"
            )

        elapsed = time.perf_counter() - t_start
        summary = {
            "run_name": self.run_name,
            "images_ok": n_ok,
            "images_error": self._counts["error"],
            "total_faces": self._total_faces,
            "total_plates": self._total_plates,
            "elapsed_s": round(elapsed, 2),
            "images_per_s": round(n_ok / elapsed, 3) if elapsed else 0.0,
            "output_mb": round(self._bytes / 2**20, 1),
            "ms_per_image": {k: round(v * 1000 / max(1, n_in), 2) for k, v in self.det.timing.items()},
            "writer_ms_per_image_thread_time": round(self._write_ms / max(1, n_ok), 2),
            "config": json.loads(json.dumps(cfg, default=str)),
            "versions": {"torch": torch.__version__, "opencv": cv2.__version__},
        }

        # Privacy report generation
        rep_cfg = cfg.get("report", {})
        if rep_cfg.get("enabled", True):
            self.report_collector.finish()
            rep_out = out_dir if out_dir else self.run_dir
            generator = ReportGenerator(
                collector=self.report_collector,
                output_dir=rep_out,
                cfg=cfg,
            )
            report_files = generator.generate()
            summary["privacy_report"] = {k: str(v) for k, v in report_files.items()}

        (self.run_dir / f"{self.run_name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        log.info("done: %s", json.dumps({k: summary[k] for k in ("images_ok", "images_error", "elapsed_s", "images_per_s")}))
        update_readme_status(
            f"🟢 Pipeline Status: COMPLETED ({n_ok}/{len(tasks)} frames)",
            {
                "Processed": f"{n_ok} frames successfully in {round(elapsed, 1)}s",
                "Average Throughput": f"{round(n_ok / elapsed, 2)} img/s",
                "Candidates Masked": f"{self._total_faces} faces, {self._total_plates} plates",
                "Privacy Certificate": "outputs/PRIVACY_DATA_SHEET.json (Schema 2.0)",
                "Audit Status": "Generated & Verified in outputs/",
            }
        )
        return summary

    def _log_progress(self, n_in: int, total: int, elapsed: float) -> None:
        ok = self._counts["ok"]
        rate = ok / elapsed if elapsed else 0.0
        eta_min = (total - ok) / rate / 60 if rate else float("nan")
        msg = f"written {ok}/{total}  {rate:.2f} img/s  ETA {eta_min:.1f} min  errors {self._counts['error']}"
        if self.device.type == "cuda":
            free, tot = torch.cuda.mem_get_info(self.device)
            msg += f"  GPU free {free / 2**20:.0f}/{tot / 2**20:.0f} MiB"
        log.info(msg)
        pct = round(ok / max(1, total) * 100, 1)
        eta_str = f"{eta_min:.1f} min" if not (eta_min != eta_min) else "calculating..."
        update_readme_status(
            f"🟡 Pipeline Status: PROCESSING ({ok}/{total} frames — {pct}%)",
            {
                "Progress": f"{ok} / {total} frames ({pct}%)",
                "Throughput": f"{rate:.2f} img/s",
                "ETA": eta_str,
                "Candidates Masked": f"{self._total_faces} faces, {self._total_plates} plates",
                "Errors": str(self._counts["error"]),
            }
        )
