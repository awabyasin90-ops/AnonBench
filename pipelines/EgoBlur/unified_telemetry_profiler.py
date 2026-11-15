#!/usr/bin/env python3
"""
unified_telemetry_profiler.py
==============================
Empirical Research Project:
"Anonymization Failure Patterns Across Real Drives and CARLA Simulation"

Unified Host-Device (GPU + CPU) Runtime, Systems Energy, and Thermal Profiler.
Addresses Amdahl's Law and Host-Side Bottlenecks in Automotive Perception:
  - GPU Telemetry: pynvml NVIDIA GPU power (mW -> W), temperature (°C), utilization (%), and VRAM (MB)
  - CPU Telemetry: Linux RAPL CPU package energy (microjoules -> Wh / 1k frames), temperature (°C),
    overall and per-core CPU utilization (%), and Process RSS / System RAM (GB) via psutil
  - Total System Energy: E_sys = E_GPU + E_CPU (integrated via Trapezoidal Rule & hardware counters)
  - Latency Breakdown: Sub-stage profiling for Pre-processing, Inference, Post-processing/Blurring, and I/O
  - Cross-platform graceful fallback for environments without NVML or RAPL
  - Configurable warm-up frames (e.g. 200 frames) to eliminate JIT compilation spikes
  - Exact latency percentiles: mean, median (p50), p90, and p95 in ms
"""

from __future__ import annotations

import os
import sys
import time
import json
import threading
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
import numpy as np


# ─────────────────────────────────────────────────────────────────
# 1. Hardware Backends Initialization (NVML + RAPL + psutil)
# ─────────────────────────────────────────────────────────────────

def _init_nvml(device_index: int = 0):
    """Initializes pynvml and returns (pynvml_module, device_handle, device_name)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode("utf-8") if isinstance(name_raw, bytes) else str(name_raw)
        return pynvml, handle, name
    except Exception as e:
        return None, None, f"NVML Unavailable ({e})"


def _init_psutil():
    """Initializes psutil for CPU/RAM telemetry."""
    try:
        import psutil
        return psutil
    except ImportError:
        return None


def read_rapl_energy_uj() -> Optional[int]:
    """
    Reads cumulative CPU package energy in microjoules from Linux RAPL interface.
    Aggregates energy across all detected CPU package domains (e.g. intel-rapl:0, intel-rapl:1).
    """
    base_dir = Path("/sys/class/powercap/intel-rapl")
    if not base_dir.exists():
        return None
    total_uj = 0
    found = False
    try:
        for domain in base_dir.glob("intel-rapl:*"):
            energy_file = domain / "energy_uj"
            if energy_file.exists():
                val = int(energy_file.read_text().strip())
                total_uj += val
                found = True
    except Exception:
        return None
    return total_uj if found else None


def read_cpu_temperature_c(psutil_mod) -> float:
    """Reads current CPU package / die temperature in Celsius via psutil."""
    if not psutil_mod or not hasattr(psutil_mod, "sensors_temperatures"):
        return 0.0
    try:
        temps = psutil_mod.sensors_temperatures()
        if not temps:
            return 0.0
        for key in ("coretemp", "k10temp", "cpu_thermal", "soc_thermal", "cpu-thermal"):
            if key in temps:
                for entry in temps[key]:
                    label = (entry.label or "").lower()
                    if "package" in label or "tctl" in label or "die" in label or not entry.label:
                        if entry.current and entry.current > 0:
                            return float(entry.current)
                if temps[key] and temps[key][0].current:
                    return float(temps[key][0].current)
        for sensor_list in temps.values():
            for entry in sensor_list:
                if entry.current and entry.current > 0:
                    return float(entry.current)
    except Exception:
        pass
    return 0.0


# ─────────────────────────────────────────────────────────────────
# 2. Hardware Telemetry Data Structures & Background Sampler
# ─────────────────────────────────────────────────────────────────

@dataclass
class HardwareSample:
    timestamp_s: float
    gpu_power_w: float
    gpu_temp_c: float
    gpu_util_pct: float
    vram_used_mb: float
    cpu_percent: float
    cpu_temp_c: float
    cpu_per_core: List[float]
    ram_used_gb: float
    process_ram_gb: float


class TelemetrySampler(threading.Thread):
    """
    Background sampling thread running at high frequency (default: 100ms).
    Samples GPU power draw, temperatures, VRAM, CPU utilization, thermals, and RAM.
    """

    def __init__(self, sample_interval_s: float = 0.1, device_index: int = 0):
        super().__init__(daemon=True)
        self.interval_s = sample_interval_s
        self.device_index = device_index
        self.samples: List[HardwareSample] = []
        self._stop_event = threading.Event()
        self._nvml, self._handle, self.device_name = _init_nvml(device_index)
        self._psutil = _init_psutil()
        self._proc = self._psutil.Process() if self._psutil else None

        # Baseline hardware energy counters
        self.initial_gpu_energy_mj: Optional[int] = None
        if self._nvml and self._handle:
            try:
                self.initial_gpu_energy_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            except Exception:
                self.initial_gpu_energy_mj = None

        self.initial_rapl_uj: Optional[int] = read_rapl_energy_uj()

    def run(self):
        while not self._stop_event.is_set():
            ts = time.monotonic()
            gpu_w = 0.0
            gpu_temp = 0.0
            gpu_util = 0.0
            vram_mb = 0.0

            if self._nvml and self._handle:
                try:
                    gpu_w = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                except Exception:
                    pass
                try:
                    gpu_temp = float(self._nvml.nvmlDeviceGetTemperature(self._handle, 0))
                except Exception:
                    pass
                try:
                    gpu_util = float(self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu)
                except Exception:
                    pass
                try:
                    mem_info = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    vram_mb = float(mem_info.used) / (1024.0 * 1024.0)
                except Exception:
                    pass

            # CPU & Host System Telemetry
            cpu_pct = 0.0
            cpu_temp = 0.0
            cpu_cores: List[float] = []
            ram_gb = 0.0
            proc_ram_gb = 0.0

            if self._psutil:
                try:
                    cpu_pct = float(self._psutil.cpu_percent(interval=None))
                    cpu_cores = [float(c) for c in self._psutil.cpu_percent(interval=None, percpu=True)]
                    ram_gb = float(self._psutil.virtual_memory().used) / (1024.0 ** 3)
                    if self._proc:
                        proc_ram_gb = float(self._proc.memory_info().rss) / (1024.0 ** 3)
                except Exception:
                    pass
                cpu_temp = read_cpu_temperature_c(self._psutil)

            # Universal Power Calculation
            if gpu_w > 0.1:
                active_gpu_w = gpu_w
            else:
                active_gpu_w = 0.0

            self.samples.append(HardwareSample(
                timestamp_s=ts,
                gpu_power_w=active_gpu_w,
                gpu_temp_c=gpu_temp,
                gpu_util_pct=gpu_util,
                vram_used_mb=vram_mb,
                cpu_percent=cpu_pct,
                cpu_temp_c=cpu_temp,
                cpu_per_core=cpu_cores,
                ram_used_gb=ram_gb,
                process_ram_gb=proc_ram_gb,
            ))
            time.sleep(self.interval_s)

    def stop(self):
        self._stop_event.set()

    def clear(self):
        """Clears accumulated samples (used at the end of warm-up phase)."""
        self.samples.clear()
        if self._nvml and self._handle:
            try:
                self.initial_gpu_energy_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            except Exception:
                self.initial_gpu_energy_mj = None
        self.initial_rapl_uj = read_rapl_energy_uj()

    def total_gpu_energy_wh(self) -> float:
        """Integrates GPU power samples via Trapezoidal Rule -> Watt-hours."""
        if len(self.samples) < 2:
            return 0.0
        total_joules = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i].timestamp_s - self.samples[i - 1].timestamp_s
            avg_w = (self.samples[i].gpu_power_w + self.samples[i - 1].gpu_power_w) / 2.0
            total_joules += avg_w * dt
        return total_joules / 3600.0

    def query_gpu_counter_energy_wh(self) -> Optional[float]:
        """Queries hardware cumulative counter delta in mJ if supported -> Wh."""
        if self._nvml and self._handle and self.initial_gpu_energy_mj is not None:
            try:
                current_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
                delta_mj = current_mj - self.initial_gpu_energy_mj
                if delta_mj >= 0:
                    return (delta_mj / 1000.0) / 3600.0
            except Exception:
                pass
        return None

    def query_cpu_rapl_energy_wh(self) -> Optional[float]:
        """Queries Linux Intel/AMD RAPL hardware package energy delta -> Wh."""
        current_uj = read_rapl_energy_uj()
        if self.initial_rapl_uj is not None and current_uj is not None:
            delta_uj = current_uj - self.initial_rapl_uj
            if delta_uj >= 0:
                return (delta_uj / 1e6) / 3600.0
        return None


# ─────────────────────────────────────────────────────────────────
# 3. Latency Stage Profiling Context Managers
# ─────────────────────────────────────────────────────────────────

class _StageContext:
    def __init__(self, frame_ctx: "FrameRecordContext", stage_name: str):
        self.frame_ctx = frame_ctx
        self.stage_name = stage_name
        self.t0 = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        dt_ms = (time.perf_counter() - self.t0) * 1000.0
        self.frame_ctx.stages[self.stage_name] = self.frame_ctx.stages.get(self.stage_name, 0.0) + dt_ms


class FrameRecordContext:
    """Helper context manager to record per-frame wall-clock latency and sub-stages."""

    def __init__(self, profiler: "UnifiedTelemetryProfiler"):
        self.profiler = profiler
        self.t0 = 0.0
        self.stages: Dict[str, float] = {}

    def stage(self, stage_name: str):
        """Context manager for an individual sub-stage (e.g. 'pre', 'infer', 'post', 'io')."""
        return _StageContext(self, stage_name)

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        total_ms = (time.perf_counter() - self.t0) * 1000.0
        self.profiler.record_frame(
            latency_ms=total_ms,
            pre_ms=self.stages.get("pre", 0.0),
            infer_ms=self.stages.get("infer", 0.0),
            post_ms=self.stages.get("post", 0.0),
            io_ms=self.stages.get("io", 0.0),
        )


# ─────────────────────────────────────────────────────────────────
# 4. Unified Telemetry Profiler Core Class
# ─────────────────────────────────────────────────────────────────

class UnifiedTelemetryProfiler:
    """
    Standardized Host-Device (GPU + CPU) Telemetry Profiler.

    Usage:
        with UnifiedTelemetryProfiler(pipeline_name="FAST-v5", condition="Clear", warmup_frames=200) as profiler:
            for img_path in images:
                with profiler.frame_scope() as frame:
                    with frame.stage("pre"):
                        img = cv2.imread(img_path)
                    with frame.stage("infer"):
                        dets = detector.infer(img)
                    with frame.stage("post"):
                        anonymized = apply_blur(img, dets)
                    with frame.stage("io"):
                        cv2.imwrite(out_path, anonymized)
        profiler.save_summary("output/telemetry.json")
    """

    def __init__(
        self,
        pipeline_name: str,
        condition: str = "Clear",
        warmup_frames: int = 200,
        sample_interval_s: float = 0.1,
        device_index: int = 0,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.pipeline_name = pipeline_name
        self.condition = condition
        self.warmup_frames = warmup_frames
        self.sample_interval_s = sample_interval_s
        self.device_index = device_index
        self.extra_metadata = extra_metadata or {}

        # Internal state
        self._sampler = TelemetrySampler(sample_interval_s, device_index)
        self._warmup_count = 0
        self._in_warmup = True
        self._frame_latencies_ms: List[float] = []
        self._stage_pre_ms: List[float] = []
        self._stage_infer_ms: List[float] = []
        self._stage_post_ms: List[float] = []
        self._stage_io_ms: List[float] = []
        self._start_time_s: Optional[float] = None
        self._end_time_s: Optional[float] = None
        self.summary: Dict[str, Any] = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        """Starts the background telemetry sampler and marks start time."""
        self._sampler.start()
        self._start_time_s = time.monotonic()
        self._in_warmup = (self.warmup_frames > 0)
        self._warmup_count = 0
        self._frame_latencies_ms.clear()
        self._stage_pre_ms.clear()
        self._stage_infer_ms.clear()
        self._stage_post_ms.clear()
        self._stage_io_ms.clear()

    def record_frame(
        self,
        latency_ms: float,
        pre_ms: float = 0.0,
        infer_ms: float = 0.0,
        post_ms: float = 0.0,
        io_ms: float = 0.0,
    ):
        """Records a completed frame and optional stage breakdown."""
        if self._in_warmup:
            self._warmup_count += 1
            if self._warmup_count >= self.warmup_frames:
                self._in_warmup = False
                self._sampler.clear()
                self._frame_latencies_ms.clear()
                self._stage_pre_ms.clear()
                self._stage_infer_ms.clear()
                self._stage_post_ms.clear()
                self._stage_io_ms.clear()
                self._start_time_s = time.monotonic()
            return

        self._frame_latencies_ms.append(latency_ms)
        if pre_ms > 0:
            self._stage_pre_ms.append(pre_ms)
        if infer_ms > 0:
            self._stage_infer_ms.append(infer_ms)
        if post_ms > 0:
            self._stage_post_ms.append(post_ms)
        if io_ms > 0:
            self._stage_io_ms.append(io_ms)

    def frame_scope(self) -> FrameRecordContext:
        """Returns a context manager for timing an individual frame and sub-stages."""
        return FrameRecordContext(self)

    def stop(self) -> Dict[str, Any]:
        """Stops the sampler, computes all Host + Device metrics, and compiles summary."""
        self._end_time_s = time.monotonic()
        self._sampler.stop()
        self._sampler.join(timeout=2.0)

        elapsed_s = max(1e-6, self._end_time_s - (self._start_time_s or self._end_time_s))
        total_frames = len(self._frame_latencies_ms)

        # 1. Latency statistics
        if total_frames > 0:
            latencies = np.array(self._frame_latencies_ms, dtype=np.float64)
            mean_latency_ms = float(np.mean(latencies))
            median_latency_ms = float(np.median(latencies))
            p90_latency_ms = float(np.percentile(latencies, 90))
            p95_latency_ms = float(np.percentile(latencies, 95))
            throughput_img_s = float(total_frames / elapsed_s)
        else:
            mean_latency_ms = 0.0
            median_latency_ms = 0.0
            p90_latency_ms = 0.0
            p95_latency_ms = 0.0
            throughput_img_s = 0.0

        # Sub-stage means
        pre_mean_ms = float(np.mean(self._stage_pre_ms)) if self._stage_pre_ms else 0.0
        infer_mean_ms = float(np.mean(self._stage_infer_ms)) if self._stage_infer_ms else 0.0
        post_mean_ms = float(np.mean(self._stage_post_ms)) if self._stage_post_ms else 0.0
        io_mean_ms = float(np.mean(self._stage_io_ms)) if self._stage_io_ms else 0.0

        # 2. GPU Energy Calculation
        trapezoidal_gpu_wh = self._sampler.total_gpu_energy_wh()
        hw_gpu_wh = self._sampler.query_gpu_counter_energy_wh()
        gpu_energy_wh = hw_gpu_wh if hw_gpu_wh is not None else trapezoidal_gpu_wh
        gpu_wh_per_1k = (gpu_energy_wh / total_frames * 1000.0) if total_frames > 0 else 0.0

        # 3. CPU Package Energy Calculation (RAPL)
        cpu_energy_wh = self._sampler.query_cpu_rapl_energy_wh()
        is_rapl_active = bool(cpu_energy_wh is not None)
        if cpu_energy_wh is None:
            # Fallback estimation based on CPU load (idle 15W + load*45W)
            samples_cpu = [s.cpu_percent for s in self._sampler.samples] if self._sampler.samples else [0.0]
            avg_cpu_load = float(np.mean(samples_cpu)) / 100.0
            est_cpu_power_w = 15.0 + avg_cpu_load * 45.0
            cpu_energy_wh = (est_cpu_power_w * elapsed_s) / 3600.0

        cpu_wh_per_1k = (cpu_energy_wh / total_frames * 1000.0) if total_frames > 0 else 0.0
        total_sys_wh_per_1k = gpu_wh_per_1k + cpu_wh_per_1k

        # 4. Thermal and Utilization Samples
        samples = self._sampler.samples
        if samples:
            gpu_powers = [s.gpu_power_w for s in samples]
            gpu_temps = [s.gpu_temp_c for s in samples if s.gpu_temp_c > 0]
            gpu_utils = [s.gpu_util_pct for s in samples]
            vram_mbs = [s.vram_used_mb for s in samples]

            cpu_pcts = [s.cpu_percent for s in samples]
            cpu_temps = [s.cpu_temp_c for s in samples if s.cpu_temp_c > 0]
            ram_gbs = [s.ram_used_gb for s in samples]
            proc_ram_gbs = [s.process_ram_gb for s in samples]

            avg_power_w = float(np.mean(gpu_powers)) if gpu_powers else 0.0
            avg_temp_c = float(np.mean(gpu_temps)) if gpu_temps else 0.0
            peak_temp_c = float(np.max(gpu_temps)) if gpu_temps else 0.0
            avg_gpu_util = float(np.mean(gpu_utils)) if gpu_utils else 0.0
            peak_gpu_util = float(np.max(gpu_utils)) if gpu_utils else 0.0
            peak_vram_mb = float(np.max(vram_mbs)) if vram_mbs else 0.0

            avg_cpu_pct = float(np.mean(cpu_pcts)) if cpu_pcts else 0.0
            peak_cpu_pct = float(np.max(cpu_pcts)) if cpu_pcts else 0.0
            avg_cpu_temp = float(np.mean(cpu_temps)) if cpu_temps else 0.0
            peak_cpu_temp = float(np.max(cpu_temps)) if cpu_temps else 0.0
            peak_sys_ram_gb = float(np.max(ram_gbs)) if ram_gbs else 0.0
            peak_proc_ram_gb = float(np.max(proc_ram_gbs)) if proc_ram_gbs else 0.0

            # Per-core utilization
            per_core_samples = [s.cpu_per_core for s in samples if s.cpu_per_core]
            if per_core_samples:
                core_matrix = np.array(per_core_samples)
                per_core_means = [float(np.mean(core_matrix[:, c])) for c in range(core_matrix.shape[1])]
            else:
                per_core_means = []
        else:
            avg_power_w = 0.0
            avg_temp_c = 0.0
            peak_temp_c = 0.0
            avg_gpu_util = 0.0
            peak_gpu_util = 0.0
            peak_vram_mb = 0.0
            avg_cpu_pct = 0.0
            peak_cpu_pct = 0.0
            avg_cpu_temp = 0.0
            peak_cpu_temp = 0.0
            peak_sys_ram_gb = 0.0
            peak_proc_ram_gb = 0.0
            per_core_means = []

        self.summary = {
            "pipeline_name": self.pipeline_name,
            "condition": self.condition,
            "hardware_device": self._sampler.device_name,
            "is_nvml_active": bool(self._sampler._nvml is not None),
            "is_rapl_active": is_rapl_active,
            "warmup_frames_excluded": self.warmup_frames,
            "total_frames_processed": total_frames,
            "end_to_end_total_time_s": round(elapsed_s, 3),
            "mean_throughput_img_per_s": round(throughput_img_s, 2),

            # Latency Breakdown
            "latency_mean_ms": round(mean_latency_ms, 2),
            "latency_median_ms": round(median_latency_ms, 2),
            "latency_p90_ms": round(p90_latency_ms, 2),
            "latency_p95_ms": round(p95_latency_ms, 2),
            "latency_stage_pre_mean_ms": round(pre_mean_ms, 2),
            "latency_stage_infer_mean_ms": round(infer_mean_ms, 2),
            "latency_stage_post_mean_ms": round(post_mean_ms, 2),
            "latency_stage_io_mean_ms": round(io_mean_ms, 2),

            # Energy Metrics (E_sys = E_GPU + E_CPU)
            "gpu_energy_wh_per_1k_frames": round(gpu_wh_per_1k, 4),
            "cpu_energy_wh_per_1k_frames": round(cpu_wh_per_1k, 4),
            "total_system_energy_wh_per_1k_frames": round(total_sys_wh_per_1k, 4),
            "avg_gpu_power_w": round(avg_power_w, 2),

            # Thermals
            "gpu_temp_mean_c": round(avg_temp_c, 1),
            "gpu_temp_peak_c": round(peak_temp_c, 1),
            "cpu_temp_mean_c": round(avg_cpu_temp, 1),
            "cpu_temp_peak_c": round(peak_cpu_temp, 1),

            # Utilization & Memory
            "gpu_util_mean_pct": round(avg_gpu_util, 1),
            "gpu_util_peak_pct": round(peak_gpu_util, 1),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "cpu_util_mean_pct": round(avg_cpu_pct, 1),
            "cpu_util_peak_pct": round(peak_cpu_pct, 1),
            "cpu_per_core_mean_pct": [round(c, 1) for c in per_core_means],
            "process_ram_rss_peak_gb": round(peak_proc_ram_gb, 3),
            "system_ram_used_peak_gb": round(peak_sys_ram_gb, 3),

            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "extra_metadata": self.extra_metadata,
        }
        return self.summary

    def save_summary(self, output_path: Union[str, Path]) -> Path:
        """Saves telemetry summary to JSON file."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
        return out

    def to_markdown_row(self) -> str:
        """Formats the summary as a Markdown table row including CPU & GPU metrics."""
        s = self.summary
        return (
            f"| **{s['pipeline_name']}** | {s['condition']} | "
            f"**{s['mean_throughput_img_per_s']:.2f}** | "
            f"{s['latency_median_ms']:.1f} ms | "
            f"{s['gpu_energy_wh_per_1k_frames']:.2f} | "
            f"{s['cpu_energy_wh_per_1k_frames']:.2f} | "
            f"**{s['total_system_energy_wh_per_1k_frames']:.2f}** | "
            f"{s['cpu_temp_mean_c']:.1f}°C / {s['cpu_temp_peak_c']:.1f}°C | "
            f"{s['gpu_temp_mean_c']:.1f}°C / {s['gpu_temp_peak_c']:.1f}°C | "
            f"{s['cpu_util_mean_pct']:.1f}% | "
            f"{s['gpu_util_mean_pct']:.1f}% | "
            f"{s['process_ram_rss_peak_gb']:.2f} GB |"
        )


if __name__ == "__main__":
    print("[*] Testing Enhanced UnifiedTelemetryProfiler (Host-Device Suite)...")
    with UnifiedTelemetryProfiler(pipeline_name="SelfTest-HostDevice", condition="Clear", warmup_frames=3) as p:
        for i in range(10):
            with p.frame_scope() as frame:
                with frame.stage("pre"):
                    time.sleep(0.005)
                with frame.stage("infer"):
                    time.sleep(0.010)
                with frame.stage("post"):
                    time.sleep(0.005)
    res = p.summary
    print("[+] Test completed successfully. Summary keys:")
    for k, v in res.items():
        print(f"    - {k}: {v}")
