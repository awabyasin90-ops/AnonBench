#!/usr/bin/env python3
"""
generate_benchmark_charts.py
=============================
Generates the 3 standalone publication-quality charts for the 5-system benchmark:
  1. chart_throughput_fps.png   - Throughput (FPS)
  2. chart_latency_summary.png  - Latency Breakdown (Median & P95 ms)
  3. chart_memory_vram.png      - Peak RAM / VRAM Footprint (MB)

Usage:
  python3 generate_benchmark_charts.py --telemetry-json <path_to_telemetry.json> --output-dir <charts_dir>
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

NAME_MAP = {
    "1_FAST_V5": "FAST v5",
    "2_EgoBlur": "EgoBlur",
    "3_DashcamCleaner": "DashcamCleaner",
    "4_UnderstandAI": "FusionPortable",
    "4_FusionPortable": "FusionPortable",
    "5_Generative_Anonymizer": "Generative"
}

def generate_charts(telemetry_data: list, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort order
    preferred_order = ["FAST v5", "DashcamCleaner", "EgoBlur", "FusionPortable", "Generative"]
    
    parsed = {}
    for item in telemetry_data:
        raw_name = item.get("system_id", item.get("folder", ""))
        clean_name = NAME_MAP.get(raw_name, item.get("system_name", raw_name))
        if "fast" in clean_name.lower(): clean_name = "FAST v5"
        elif "dashcam" in clean_name.lower(): clean_name = "DashcamCleaner"
        elif "egoblur" in clean_name.lower(): clean_name = "EgoBlur"
        elif "fusion" in clean_name.lower() or "understand" in clean_name.lower(): clean_name = "FusionPortable"
        elif "generative" in clean_name.lower(): clean_name = "Generative"
        
        parsed[clean_name] = {
            "fps": float(item.get("fps", 0.0)),
            "median_lat": float(item.get("median_latency_ms", 0.0)),
            "p95_lat": float(item.get("p95_latency_ms", 0.0)),
            "ram": float(item.get("peak_ram_mb", 0.0))
        }

    short_names = [n for n in preferred_order if n in parsed]
    if not short_names:
        short_names = list(parsed.keys())
        
    fps_values = [parsed[n]["fps"] for n in short_names]
    median_lat = [parsed[n]["median_lat"] for n in short_names]
    p95_lat = [parsed[n]["p95_lat"] for n in short_names]
    vram_values = [parsed[n]["ram"] for n in short_names]

    # -------------------------------------------------------------
    # CHART 1: THROUGHPUT (FPS) ONLY
    # -------------------------------------------------------------
    plt.figure(figsize=(8.5, 5), dpi=300)
    colors_fps = ['#27ae60', '#d35400', '#8e44ad', '#2980b9', '#16a085']
    bars = plt.bar(short_names, fps_values, color=colors_fps[:len(short_names)], width=0.5, edgecolor='#2c3e50', linewidth=1.2)
    plt.yscale('log')
    plt.title('Inference Throughput (FPS)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Frames Per Second (FPS) [Log Scale]', fontsize=10, fontweight='bold')
    plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)

    for bar, fps in zip(bars, fps_values):
        plt.annotate(f'{fps:.2f} FPS',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold', color='#1a252f')

    plt.tight_layout()
    fps_path = output_dir / "chart_throughput_fps.png"
    plt.savefig(fps_path)
    plt.close()
    print(f"[+] Saved FPS Chart -> {fps_path.name}")

    # -------------------------------------------------------------
    # CHART 2: LATENCY BREAKDOWN (MEDIAN & P95 ONLY)
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    x = np.arange(len(short_names))
    w = 0.32

    b1 = plt.bar(x - w/2, median_lat, width=w, label='Median Latency', color='#27ae60', edgecolor='black', linewidth=0.8)
    b2 = plt.bar(x + w/2, p95_lat, width=w, label='P95 Latency', color='#e74c3c', edgecolor='black', linewidth=0.8)

    plt.yscale('log')
    plt.xticks(x, short_names, fontsize=10, fontweight='bold')
    plt.title('Latency Breakdown: Median vs. P95 (ms)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Latency in Milliseconds (ms) [Log Scale]', fontsize=10, fontweight='bold')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    plt.grid(axis='y', which='both', linestyle='--', alpha=0.5)

    for bar, val in zip(b1, median_lat):
        txt = f"{val:.1f}ms" if val < 1000 else f"{val/1000:.1f}s"
        plt.annotate(txt,
                     (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1b4d2e')

    for bar, val in zip(b2, p95_lat):
        txt = f"{val:.1f}ms" if val < 1000 else f"{val/1000:.1f}s"
        plt.annotate(txt,
                     (bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 4), textcoords="offset points",
                     ha='center', va='bottom', fontsize=8, fontweight='bold', color='#7a1e16')

    plt.tight_layout()
    lat_path = output_dir / "chart_latency_summary.png"
    plt.savefig(lat_path)
    plt.close()
    print(f"[+] Saved Latency Chart -> {lat_path.name}")

    # -------------------------------------------------------------
    # CHART 3: PEAK RAM / VRAM MEMORY USAGE (MB)
    # -------------------------------------------------------------
    plt.figure(figsize=(8.5, 5), dpi=300)
    colors_vram = ['#2ecc71', '#f39c12', '#9b59b6', '#e74c3c', '#3498db']
    bars_v = plt.bar(short_names, vram_values, color=colors_vram[:len(short_names)], width=0.5, edgecolor='#2c3e50', linewidth=1.2)
    plt.title('Peak Memory / RAM Footprint (MB)', fontsize=13, fontweight='bold', pad=12)
    plt.ylabel('Peak RAM / VRAM (MB)', fontsize=10, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar, val in zip(bars_v, vram_values):
        plt.annotate(f'{val:.1f} MB',
                     (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                     xytext=(0, 5), textcoords="offset points",
                     ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold', color='#1a252f')

    plt.axhline(500, color='#27ae60', linestyle=':', linewidth=1.2, label='Embedded Edge Budget (500 MB)')
    plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5)

    plt.tight_layout()
    vram_path = output_dir / "chart_memory_vram.png"
    plt.savefig(vram_path)
    plt.close()
    print(f"[+] Saved VRAM / Memory Chart -> {vram_path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate benchmark telemetry charts")
    parser.add_argument("--telemetry-json", type=str, required=True, help="Path to telemetry_summary_comparison.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save generated charts")
    args = parser.parse_args()

    with open(args.telemetry_json, "r") as f:
        telem = json.load(f)

    generate_charts(telem, Path(args.output_dir))

