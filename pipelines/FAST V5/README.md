# FAST: Proposed Automotive Privacy Anonymization Pipeline

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20GPU%20%7C%20CPU%20Fallback-green.svg)]()
[![Inference](https://img.shields.io/badge/Inference-TensorRT%20FP16%20%7C%20ONNX%20%7C%20PyTorch-orange.svg)]()
[![Telemetry](https://img.shields.io/badge/Telemetry-NVML%20Standardized-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Research%20%26%20Commercial-purple.svg)]()

> **FAST (Fleet Anonymization Systems Telemetry)** is an end-to-end, high-throughput privacy anonymization pipeline engineered for autonomous driving fleets (nuScenes, Pandaset, OPINA, and CARLA driving datasets). It combines multi-scale face and license plate detection with adaptive Gaussian blurring, feathered boundary blending, and hardware-level telemetry profiling (speed, latency, GPU energy, temperature, and compute utilization).

---

## 1. Pipeline Architecture

The FAST pipeline processes high-resolution driving frames end-to-end:
```
[Input Frame (PNG/JPG)] 
       │
       ├──> [YOLOv8m/11m-Face @ 1280px] ──> Detected Face Bounding Boxes
       ├──> [YOLO11m Vehicle @ 640px]   ──> Vehicle Context Regions
       │                                            │
       │                                            ▼
       └──> [Plate-Nano @ 192x320px]    ──> License Plate Bounding Boxes
                                                    │
                                                    ▼
                       [Adaptive Gaussian Blur & Feathered Blending]
                                                    │
                                                    ▼
                       [Anonymized Frame Output & Detections JSON]
```

### Core Components
1. **Face Detection**: YOLOv8m/11m-Face operating at 1280px long-edge for robust detection of near-field and distant pedestrians.
2. **Vehicle Context**: YOLO11m operating at 640px to localize vehicles and provide candidate ROIs for plate identification.
3. **License Plate Detection**: Plate-Nano operating at 192x320px for high-throughput detection of vehicle registration plates.
4. **Adaptive Blurring**: Scale-dependent Gaussian kernel with 2px feathered elliptical/rectangular boundary blending.
5. **Telemetry Profiling**: High-frequency NVML hardware telemetry recording wall-clock speed (`img/s`), latency (`ms`), GPU energy (`Wh / 1k frames`), temperature (`°C`), and GPU utilization (`%`).

---

## 2. Directory Structure

```text
FAST/
├── run_fast_telemetry.py        # Master standalone unified telemetry runner
├── run_pipeline.py              # Native pipeline entry point
├── README.md                    # This documentation
├── config/
│   └── pipeline_config.yaml     # Pipeline thresholds, model paths, and batch sizes
├── weights/                     # Pre-trained deep learning weights & TensorRT engines
│   ├── yolov8m-face.pt / .engine
│   ├── yolo11m.pt / .engine
│   └── plate_nano_192x320_best.pt / .engine
├── fastblur/                    # Core pipeline package
│   ├── pipeline.py              # Pipeline orchestrator
│   ├── render_v4.py             # Feathered elliptical blur renderer
│   ├── imageio.py               # Image I/O encoder
│   └── research/                # Telemetry monitoring and condition stratification
└── tools/                       # Model export and engine compilation utilities
```

---

## 3. Quick Start & CLI Execution

### Environment Setup
Python **3.10 or newer** with PyTorch and optionally TensorRT:
```bash
pip install -r requirements.txt
```

### Standalone Telemetry Benchmark Execution
Run the standalone runner across any driving condition:
```bash
python run_fast_telemetry.py \
  --input-dir /path/to/dataset/Clear \
  --output-dir ./outputs/fast/Clear \
  --condition Clear \
  --warmup 200
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input-dir` | *(required)* | Path to folder containing raw driving frames |
| `--output-dir` | *(required)* | Directory to save anonymized images and telemetry |
| `--condition` | `Clear` | Driving scenario: `Clear`, `Rain`, `Night`, `Fog`, `Glare`, `Overcast` |
| `--config` | `config/pipeline_config.yaml` | Path to pipeline configuration file |
| `--warmup` | `200` | Warm-up frames to exclude from telemetry timers |
| `--limit` | `0` | Process first N frames (0 = all) |

---

## 4. Benchmark Specifications & Evaluation Scope

### Ground-Truth Oracle Alignment
Evaluation is conducted strictly against the repository's two standardized Ground-Truth Oracles:
1. **Faces**: InsightFace SCRFD-34G (Confirmed: width $\ge 10\text{ px}$, PP4AV standard)
2. **Plates**: RT-DETR-v2-L Plate (Confirmed: height $\ge 8\text{ px}$, PP4AV standard)

### Measured Benchmark Metrics
- **Detection Miss Rate (FN Rate)**: Percentage of confirmed ground truth targets missed by the detector.
- **Scale Stratification**: Evaluated across 4 scale bins: `<16px`, `16–32px`, `32–64px`, `>64px`.
- **Environmental Robustness**: Evaluated across 6 adverse driving conditions: `Clear`, `Rain`, `Night`, `Fog`, `Glare`, `Overcast`.
- **Systems Telemetry**:
  - Throughput (`img/s`) and Latency (`ms/img`).
  - Active GPU Power (`W`) and GPU Energy Consumption (`Wh / 1,000 frames`).
  - Mean & Peak GPU Temperature (`°C`).
  - Mean & Peak GPU Compute Utilization (`%`).
  - Peak VRAM (`MB`) and System RAM (`MB`).

### Strict Research Scope Rules Enforced
- **NO Post-Blur Leakage Audits**: Evaluation relies strictly on detection bounding boxes and Miss Rate. No post-blur facial recognition (ArcFace) or OCR reconstruction attacks are included.
- **NO Over-Blur Rate / Image Quality Metrics**: Background distortion metrics (SSIM/PSNR) and Over-Blur Rates are excluded; all models run strictly at their native as-shipped configurations.
- **Unified Hardware Conditions**: All models are benchmarked under identical hardware conditions, recording measured speed, power, and utilization per model.
