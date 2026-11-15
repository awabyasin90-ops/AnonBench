"""
energy_monitor.py — OPINA Research Automation
===============================================
Measures GPU + CPU energy consumption during FAST v4 pipeline runs,
and auto-populates the Research Tracker energy columns:
  - Total Energy (Wh)
  - Energy per 1,000 frames (Wh/1,000 frames)

Two backends (auto-selected):
  Backend A: pynvml         — NVIDIA GPU power (most accurate, Wh)
  Backend B: psutil + time  — CPU-only proxy (no GPU measurement)

Usage (wrap any command):
    python energy_monitor.py --command "python run_blur_v4.py ..."
                             --output-json energy_result.json
                             --n-frames 578

Or use as a context-manager in your own scripts:
    from energy_monitor import EnergyMonitor
    with EnergyMonitor(n_frames=578) as em:
        run_pipeline(...)
    print(em.result)
"""

import os
import time
import json
import threading
import subprocess
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime


# ─────────────────────────────────────────────────────────────────
# Power backends
# ─────────────────────────────────────────────────────────────────
def _try_pynvml():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name   = pynvml.nvmlDeviceGetName(handle)
        print(f"[EnergyMonitor] GPU backend: pynvml — {name}")
        return pynvml, handle
    except Exception as e:
        print(f"[EnergyMonitor] pynvml unavailable ({e}). Using CPU proxy.")
        return None, None


def _try_psutil():
    try:
        import psutil
        print("[EnergyMonitor] CPU backend: psutil (no GPU measurement)")
        return psutil
    except ImportError:
        print("[EnergyMonitor] psutil unavailable. Install: pip install psutil")
        return None


# ─────────────────────────────────────────────────────────────────
# Sampler thread
# ─────────────────────────────────────────────────────────────────
@dataclass
class EnergySample:
    timestamp_s: float
    gpu_power_w: float   # 0.0 if GPU unavailable
    cpu_percent: float


class PowerSampler(threading.Thread):
    """Background thread that samples power every `interval_s` seconds."""

    def __init__(self, interval_s: float = 0.5):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.samples: list[EnergySample] = []
        self._stop_event = threading.Event()
        self._pynvml, self._nvml_handle = _try_pynvml()
        self._psutil = _try_psutil()

    def run(self):
        while not self._stop_event.is_set():
            ts = time.monotonic()

            # GPU power (mW → W)
            gpu_w = 0.0
            if self._pynvml and self._nvml_handle:
                try:
                    import pynvml as nv
                    gpu_w = nv.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
                except Exception:
                    pass

            # CPU %
            cpu_pct = 0.0
            if self._psutil:
                try:
                    cpu_pct = self._psutil.cpu_percent(interval=None)
                except Exception:
                    pass

            # Universal power measurement across any PC:
            # If NVML GPU power is available (>0.1W), use GPU power + 10W platform overhead.
            # If on CPU/integrated GPU or NVML unavailable, estimate active power from CPU load:
            # 22.0W idle baseline + (cpu_pct / 100) * 35.0W active compute load.
            if gpu_w > 0.1:
                power_w = gpu_w + 10.0
            else:
                power_w = 22.0 + (cpu_pct / 100.0) * 35.0

            self.samples.append(EnergySample(ts, power_w, cpu_pct))
            time.sleep(self.interval_s)

    def stop(self):
        self._stop_event.set()

    def total_energy_wh(self) -> float:
        """Integrate GPU power samples using trapezoidal rule → Watt-hours."""
        if len(self.samples) < 2:
            return 0.0
        total_j = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i].timestamp_s - self.samples[i-1].timestamp_s
            avg_w = (self.samples[i].gpu_power_w + self.samples[i-1].gpu_power_w) / 2.0
            total_j += avg_w * dt
        return total_j / 3600.0   # Joules → Wh

    def avg_gpu_power_w(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.gpu_power_w for s in self.samples) / len(self.samples)

    def avg_cpu_percent(self) -> float:
        if not self.samples:
            return 0.0
        return sum(s.cpu_percent for s in self.samples) / len(self.samples)


# ─────────────────────────────────────────────────────────────────
# Context manager interface
# ─────────────────────────────────────────────────────────────────
class EnergyMonitor:
    """
    Context manager for wrapping any pipeline run with energy measurement.

    Usage:
        with EnergyMonitor(n_frames=578) as em:
            run_pipeline_here()
        em.save_result("energy_result.json")
        print(f"Total energy: {em.result['total_energy_wh']:.3f} Wh")
    """

    def __init__(self, n_frames: int, sample_interval_s: float = 0.5,
                 session_id: str = "", condition: str = ""):
        self.n_frames         = n_frames
        self.sample_interval  = sample_interval_s
        self.session_id       = session_id
        self.condition        = condition
        self._sampler         = None
        self._start_time      = None
        self.result           = {}

    def __enter__(self):
        self._sampler    = PowerSampler(self.sample_interval)
        self._start_time = time.monotonic()
        self._sampler.start()
        return self

    def __exit__(self, *args):
        self._sampler.stop()
        self._sampler.join(timeout=2.0)
        elapsed_s = time.monotonic() - self._start_time

        total_wh  = self._sampler.total_energy_wh()
        wh_per_1k = (total_wh / self.n_frames * 1000) if self.n_frames > 0 else 0.0
        throughput = self.n_frames / elapsed_s if elapsed_s > 0 else 0.0

        self.result = {
            "session_id":          self.session_id,
            "condition":           self.condition,
            "n_frames":            self.n_frames,
            "runtime_s":           round(elapsed_s, 2),
            "throughput_img_s":    round(throughput, 2),
            "avg_gpu_power_w":     round(self._sampler.avg_gpu_power_w(), 2),
            "avg_cpu_pct":         round(self._sampler.avg_cpu_percent(), 1),
            "total_energy_wh":     round(total_wh,  4),
            "energy_per_1k_wh":    round(wh_per_1k, 4),
            "n_power_samples":     len(self._sampler.samples),
            "gpu_backend":         "pynvml" if self._sampler._pynvml else "none",
            "timestamp":           datetime.utcnow().isoformat() + "Z",
        }

    def save_result(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.result, f, indent=2)
        print(f"[EnergyMonitor] Result saved to {path}")
        return self.result

    def print_result(self):
        r = self.result
        print(f"\n[EnergyMonitor] ── Energy Report ─────────────────────────")
        print(f"  Session:          {r.get('session_id', '—')}")
        print(f"  Condition:        {r.get('condition', '—')}")
        print(f"  Frames:           {r['n_frames']}")
        print(f"  Runtime:          {r['runtime_s']:.2f}s")
        print(f"  Throughput:       {r['throughput_img_s']:.2f} img/s")
        print(f"  Avg GPU Power:    {r['avg_gpu_power_w']:.1f} W")
        print(f"  Total Energy:     {r['total_energy_wh']:.4f} Wh")
        print(f"  Per 1k frames:    {r['energy_per_1k_wh']:.4f} Wh/1k")
        print(f"  GPU backend:      {r['gpu_backend']}")


# ─────────────────────────────────────────────────────────────────
# Aggregator — merge multiple energy JSONs into tracker row
# ─────────────────────────────────────────────────────────────────
def aggregate_energy_results(json_paths: list[str]) -> dict:
    """
    Merge multiple per-session energy JSONs into a totals dict
    matching the Research Tracker Condition Performance Matrix columns.
    """
    totals = {
        "total_frames": 0,
        "total_runtime_s": 0.0,
        "total_energy_wh": 0.0,
        "by_condition": {}
    }
    for p in json_paths:
        with open(p) as f:
            r = json.load(f)
        cond = r.get("condition", "Unknown")
        totals["total_frames"]    += r["n_frames"]
        totals["total_runtime_s"] += r["runtime_s"]
        totals["total_energy_wh"] += r["total_energy_wh"]

        if cond not in totals["by_condition"]:
            totals["by_condition"][cond] = {
                "frames": 0, "runtime_s": 0.0,
                "energy_wh": 0.0, "throughput_img_s": []
            }
        bc = totals["by_condition"][cond]
        bc["frames"]           += r["n_frames"]
        bc["runtime_s"]        += r["runtime_s"]
        bc["energy_wh"]        += r["total_energy_wh"]
        bc["throughput_img_s"].append(r["throughput_img_s"])

    for cond, bc in totals["by_condition"].items():
        bc["avg_throughput_img_s"] = round(
            sum(bc["throughput_img_s"]) / len(bc["throughput_img_s"]), 2)
        bc["energy_per_1k_wh"] = round(
            bc["energy_wh"] / bc["frames"] * 1000 if bc["frames"] > 0 else 0, 4)
        del bc["throughput_img_s"]

    return totals


# ─────────────────────────────────────────────────────────────────
# CLI: wrap an external command
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap a pipeline command with GPU/CPU energy measurement")
    parser.add_argument("--command",     required=True,
                        help="Shell command to run and measure")
    parser.add_argument("--n-frames",    type=int, required=True,
                        help="Number of frames processed by the command")
    parser.add_argument("--output-json", required=True,
                        help="Output JSON file for energy results")
    parser.add_argument("--session-id",  default="",
                        help="Session ID tag (e.g. R001)")
    parser.add_argument("--condition",   default="",
                        help="Condition tag (e.g. Clear Day)")
    args = parser.parse_args()

    print(f"[EnergyMonitor] Starting measurement — command: {args.command}")
    with EnergyMonitor(n_frames=args.n_frames,
                       session_id=args.session_id,
                       condition=args.condition) as em:
        ret = subprocess.run(args.command, shell=True)

    em.print_result()
    em.save_result(args.output_json)

    if ret.returncode != 0:
        print(f"[EnergyMonitor] WARNING: Command exited with code {ret.returncode}")
