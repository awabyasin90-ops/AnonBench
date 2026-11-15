# Automotive Privacy Anonymization Benchmark Suite
## End-to-End Execution, Machine-Assisted Human Oracle (X-AnyLabeling), & Publication Telemetry

This directory contains the production-grade benchmark suite for evaluating the **5 vehicular privacy anonymization pipelines** across real driving datasets and CARLA simulations:

1. **Proposed FAST v5** (Real-Time ADAS Edge TensorRT / PyTorch Pipeline)
2. **DashcamCleaner** (Single-Stage YOLOv8 Community Baseline)
3. **Meta AI EgoBlur Gen2** (Two-Stage Industry TorchScript JIT Baseline)
4. **FusionPortable** (Retrained Perspective Faster R-CNN ONNX Runtime Baseline)
5. **Generative Anonymizer** (Dual Deep Identity Synthesis & Fourier Inpainting)

---

## 🏛️ The 3-Step Human Oracle Ground Truth Protocol

```
[5 Models Predictions] 
        │
        ▼  Step 1: prepare_xanylabeling_gt.py
[Merged Clean annotations.json + Fused Visual Frames]
        │
        ▼  Step 2: Human Oracle Review in X-AnyLabeling (You adjust & save)
[Human-Verified Ground Truth]
        │
        ▼  Step 3: evaluate_xanylabeling_gt.py
[Automated IoU Matching >= 0.50 + PP4AV Cutoff + Distance Bins 1-4]
        │
        ▼
[Final Q1 Tables (CSVs) + Analysis Report + Publication Charts]
```

### Step 1: Automated Multi-Model Fusion (`prepare_xanylabeling_gt.py`)
Fuses the predictions from all 5 engines into consensus candidate boxes ($\text{IoU} \ge 0.45$), renders them directly on each frame, and generates **ONE SINGLE MASTER JSON** (`annotations.json`):
```bash
python3 prepare_xanylabeling_gt.py \
    --results-dir "../outputs/results_pendik_100" \
    --output-dir "../outputs/xanylabeling_review" \
    --num-frames 10
```

* **Output Structure:**
  * Exactly $N$ image files (`<frame_name>.jpg`) with all candidate objects drawn directly on the frame.
  * Exactly ONE master JSON file (`annotations.json`) mapping each box location, dimensions, points, and class label for all frames.
  * Zero separate individual JSON files.

---

### Step 2: Human Review in X-AnyLabeling (Your Manual Step)
1. Open **X-AnyLabeling**.
2. Click **Open Dir** on `outputs/xanylabeling_review`.
3. Inspect each frame, delete false positives (FP), add any missing targets (FN), and press `Ctrl+S` (`Cmd+S`).

---

### Step 3: Automated IoU Evaluation & Q1 Metrics (`evaluate_xanylabeling_gt.py`)
Reads the human-reviewed `annotations.json` and evaluates all 5 models automatically:
```bash
python3 evaluate_xanylabeling_gt.py \
    --gt-dir "../outputs/xanylabeling_review" \
    --results-dir "../outputs/results_pendik_100" \
    --iou-thresh 0.50
```

* **Automated Computations:**
  * **PP4AV Recognizability:** Filters out faces $< 10\text{px}$ and plates $< 8\text{px}$ as `Ignore`.
  * **Distance Stratification:** Assigns objects to Bin 1 ($<16\text{px}$), Bin 2 ($16\text{--}32\text{px}$), Bin 3 ($32\text{--}64\text{px}$), Bin 4 ($>64\text{px}$).
  * **Greedy Bipartite IoU Matching:** Matches models' predicted boxes against human GT at $\text{IoU} \ge 0.50$.
  * **Mathematical Formulations:**
    $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} \times 100 \quad | \quad \text{Privacy Leak Rate} = \frac{\text{FN}}{\text{TP} + \text{FN}} \times 100$$
    $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} \times 100 \quad | \quad \text{Over-blur Rate} = \frac{\text{FP}}{\text{TP} + \text{FP}} \times 100$$
  * **Deliverables Generated:**
    * `oracle_overall_summary.csv`
    * `oracle_per_bin_metrics.csv`
    * `ORACLE_BENCHMARK_ANALYSIS.md`
    * 3 high-resolution publication charts in `charts/`.

---

## 📁 Benchmark Suite File Layout

```
3_Systems/benchmark_suite/
├── prepare_xanylabeling_gt.py     # Step 1: Fuses 5 models into 1 master annotations.json + visual frames
├── evaluate_xanylabeling_gt.py    # Step 3: Automated IoU evaluator, PP4AV binning & Q1 report generator
├── master_benchmark_runner.py     # End-to-end benchmark runner across 5 pipelines + telemetry
├── generate_comparison_frames.py  # Side-by-side 6-quadrant visual comparison generator
├── test_all_systems.py            # Smoke test for all 5 pipeline architectures
├── math.md                        # Theoretical statistical formulations & sampling mathematics
└── README.md                      # This architecture specification
```
