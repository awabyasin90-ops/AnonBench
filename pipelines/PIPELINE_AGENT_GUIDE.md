# Developer & AI Agent Execution Runbook
## Empirical Benchmark: Anonymization Failure Patterns Across Real Drives and CARLA Simulation

> **Scientific Classification: Empirical Benchmark Paper (Experimental Measurement Protocol & Failure Audit)**  
> This project executes an objective experimental measurement protocol and failure audit benchmarking **5 distinct production anonymization architectures** and **2 baseline detection oracles** across Real Drives (Research Car Rig & Tesla Fleet) and high-fidelity CARLA autonomous driving simulation under 6 standardized adverse environmental conditions and unified host-device telemetry.

---

## 1. Ground-Truth (GT) Architecture: The Two-Stage Hybrid Protocol

A foundational tenet of this benchmark is that the **Ground Truth (GT)** is not an unverified output of a single model, nor a naive manual drawing from scratch. It is constructed through a rigorous **Two-Stage Candidate Fusion & Human-in-the-Loop Audit**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: MULTI-SYSTEM CANDIDATE FUSION                          │
│                                                                                        │
│   All 5 Systems + 2 Reference Oracles Run at Low Threshold (≥ 0.15) to Maximize Recall │
│   • FAST v5 (TensorRT ADAS Edge)        • DashcamCleaner (YOLOv8)             │
│   • Meta AI EgoBlur Gen2 (Faster R-CNN)          • FusionPortable (Faster R-CNN ONNX)  │
│   • Generative Anonymizer (RetinaFace+WPOD)      • InsightFace SCRFD (Face Oracle)     │
│   • RT-DETR v2 (Plate Oracle)                                                          │
│                                      │                                                 │
│                                      ▼                                                 │
│   Candidate Fusion / Union: Aggregates all detected bounding boxes per frame           │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│              STAGE 2: HUMAN AUDIT VIA OPTIMIZED X-ANYLABELING (Desktop App)            │
│                                                                                        │
│   Human Expert Reviews & Refines Candidates Across the 7,200 Golden Subset:            │
│   • MobileSAM Auto-Snapping: Shrinks rough boxes to pixel-perfect physical boundaries  │
│   • Shift+F / F: One-key face tagging (Neon Green)                                     │
│   • Shift+P / P: One-key license plate tagging (Deep Sky Blue)                         │
│   • Delete / Backspace: Instant zero-modal removal of false positives                  │
│   • D / A: Navigation with automatic JSON disk persistence                             │
│   • Esc: Instant editor-mode exit and selection clear                                  │
│   • PP4AV Recognizability Tagging: Confirmed (≥10px face, ≥8px plate) vs. Ignore       │
└────────────────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       DEFINITIVE BENCHMARK GROUND TRUTH (GT)                           │
│   Gold-Standard Reference JSONs Used to Compute Empirical Miss Rates & PP4AV Bins 1–4  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Operating Principles for AI Agents

1. **100% Modular & Decoupled Execution**: Every pipeline is self-contained. There are no cross-imports. You can execute any pipeline independently, at any time, in any order.
2. **Standardized Data Contracts**:
   * Each system consumes raw image frames from `<INPUT_DIR>`.
   * Each system writes:
     * Anonymized (blurred/inpainted) output frames to `outputs/anonymized_frames/<system_id>/<condition>/`
     * Standardized bounding-box detection JSONs to `outputs/detections/<system_id>/<condition>/`
     * Fine-grained host-device systems telemetry to `outputs/telemetry/<system_id>_<condition>.json`
3. **Pristine As-Shipped Benchmarking**: Each system is benchmarked under its official shipping confidence thresholds and native input resolutions to capture off-the-shelf automotive performance.

---

## 3. Workspace Directory Map

```text
3_Systems/
├── 2-Gt-Pipelien's/               # High-Capacity Reference Detection Oracles
│   ├── insightface-face/          # InsightFace SCRFD-34G (Face Oracle)
│   └── RT-DETR-Plate/             # RT-DETR v2 (License Plate Oracle)
├── pipelines/                  # The 5 Benchmarked Production Architectures
│   ├── FAST V5/                   # Real-Time ADAS Edge Pipeline (TensorRT)
│   ├── EgoBlur/                   # Meta AI EgoBlur Gen2 (TorchScript JIT)
│   ├── DashcamCleaner/            # DashcamCleaner (YOLOv8 Automotive)
│   ├── Anonymizer/                # FusionPortable (Faster R-CNN ONNX)
│   └── Generative Anonymizer/     # RetinaFace + WPOD-NET + Big-LaMa Inpainting
├── X/                             # Custom-Instrumented Ground-Truth Annotation Suite
│   ├── X-AnyLabeling/             # Source code fork with micro-optimizations
│   └── README.md                  # Comprehensive optimization changelog & audit guide
├── benchmark_suite/               # Unified End-to-End Runner, Charts & Collages
│   ├── master_benchmark_runner.py
│   ├── run_benchmark.sh
│   ├── generate_benchmark_charts.py
│   ├── generate_comparison_frames.py
│   ├── generate_benchmark_report.py
│   └── README.md
├── BENCHMARK_SPECIFICATION_TABLES.md  # Formal measurement protocol & tables
└── PIPELINE_AGENT_GUIDE.md            # This agent execution runbook
```

---

## Step 0: Pre-Flight Scan & Model Verification (Run First!)

### 1. Model Loading & Weights Audit:
```bash
python3 "test_all_systems.py"
```
*(Validates in-memory model loading across all 5 pipelines and both baseline oracles).*

### 2. Check Weights Checklist:
```bash
python3 "pipelines/FAST V5/tools/pipeline_agent_manager.py" --status
```
Verify that all weight checkpoints are reported `[OK]`:
* `insightface_scrfd34g_gt`: 2/2 files `[OK]`
* `rtdetr_v2_plate_gt`: 4/4 files `[OK]`
* `proposed_fast_v5`: 7/7 files `[OK]`
* `meta_egoblur`: 2/2 files `[OK]`
* `dashcam_cleaner`: 8/8 files `[OK]`
* `fusionportable_anonymizer`: 2/2 files `[OK]`
* `generative_anonymizer`: 5/5 files `[OK]`

If any weight file is missing:
```bash
python3 "pipelines/FAST V5/tools/pipeline_agent_manager.py" --download <pipeline_id>
```

---

## Step 1: Environment & Dependency Setup

### Unified Environment (Recommended for GPU Server / Workstation):
```bash
pip install -r "pipelines/FAST V5/tools/requirements_all.txt"
pip install pytorch-lightning dill onnx onnxruntime omegaconf albumentations kornia psutil ultralytics
```

---

## Step 2: Canonical Pipeline Execution Commands

Replace `<INPUT_DIR>` with the path to the raw driving frames.

### A. High-Capacity Reference Oracles:
```bash
# 1. Face Reference Oracle (InsightFace SCRFD-34G)
python3 "2-Gt-Pipelien's/insightface-face/run_gt_oracle.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/gt_oracle/insightface" \
  --condition "Clear" --conf-thresh 0.50 --device "cuda:0"

# 2. Plate Reference Oracle (RT-DETR v2 Plate)
python3 "2-Gt-Pipelien's/RT-DETR-Plate/run_gt_oracle.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/gt_oracle/rtdetr" \
  --condition "Clear" --conf-thresh 0.25 --device "cuda:0"
```

### B. The 5 Evaluated Production Pipelines:
```bash
# 1. FAST v5 (Real-Time TensorRT ADAS Edge Pipeline)
python3 "pipelines/FAST V5/run_fast_telemetry.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/anonymized_frames/fast/Clear" \
  --condition "Clear" --config "pipelines/FAST V5/config/pipeline_config.yaml"

# 2. Meta AI EgoBlur Gen2 (Two-Stage Faster R-CNN TorchScript JIT)
python3 "pipelines/EgoBlur/run_egoblur_telemetry.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/anonymized_frames/egoblur/Clear" \
  --condition "Clear" --face-thresh 0.67416 --lp-thresh 0.74475 --device "cuda:0"

# 3. DashcamCleaner (YOLOv8 Medium 720p)
python3 "pipelines/DashcamCleaner/run_dashcamcleaner_telemetry.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/anonymized_frames/dashcamcleaner/Clear" \
  --condition "Clear" --threshold 0.40 --device "cuda:0"

# 4. FusionPortable Anonymizer (ONNX Runtime)
python3 "pipelines/Anonymizer/run_understandai_telemetry.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/anonymized_frames/understandai/Clear" \
  --condition "Clear" --face-thresh 0.30 --plate-thresh 0.30 --device "cuda:0"

# 5. Generative Anonymizer (RetinaFace + WPOD-NET + Big-LaMa Fast Fourier Inpainting)
python3 "pipelines/Generative Anonymizer/run_generative_telemetry.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/anonymized_frames/generative/Clear" \
  --condition "Clear" --conf-thresh 0.30 --device "cuda:0"
```

---

## Step 3: Candidate Fusion & Human-in-the-Loop Audit (Optimized X-AnyLabeling)

To construct the Gold-Standard Ground Truth across the 7,200 evaluation frames:

1. **Automated Multi-System Candidate Fusion**:
   Combine the bounding box detection JSONs generated from all 5 pipelines and both baseline oracles into an initial candidate proposal set:
   ```bash
   python3 "tools/fuse_system_candidates.py" \
     --detections-dirs "outputs/detections/fast" "outputs/detections/egoblur" "outputs/detections/dashcamcleaner" "outputs/detections/understandai" "outputs/detections/generative" "outputs/gt_oracle/insightface" "outputs/gt_oracle/rtdetr" \
     --output-dir "outputs/ground_truth_candidates"
   ```

2. **Launch the Custom-Optimized X-AnyLabeling Tool**:
   Open the application (via Desktop app, Applications, or terminal command):
   ```bash
   xanylabeling
   ```
   * Open the candidate folder: **Open Dir** $\rightarrow$ `outputs/ground_truth_candidates/`.
   * The tool automatically bootstraps **MobileSAM** for interactive auto-snapping.

3. **Audit Execution via Micro-Optimized Keystrokes**:
   * **`Shift+F` / `F`**: Instantly creates or re-labels a bounding box to **`face`** (Neon Green).
   * **`Shift+P` / `P`**: Instantly creates or re-labels a bounding box to **`plate`** (Deep Sky Blue).
   * **`Delete` / `Backspace`**: Immediately deletes selected false-positive boxes and flushes the updated `.json` to disk (zero dialog popups).
   * **`Esc`**: Instantly exits drawing/editor mode, resets the cursor to arrow, and deselects boxes.
   * **`D`**: Automatically saves the current `.json` and moves to the **Next Image** (0 ms RAM pre-fetched transition).
   * **`A`**: Automatically saves the current `.json` and moves to the **Previous Image**.
   * **Mouse Wheel**: Instant scroll-to-zoom centered on distant horizon targets (Bin 1 / $20+\text{ m}$) without holding Ctrl.

4. **PP4AV Recognizability Rule Enforcement**:
   * Confirm targets meeting recognizability thresholds: Face width $\ge 10\text{ px}$, License plate height $\ge 8\text{ px}$.
   * Sub-threshold targets are flagged as `status = Ignore` (logged in master accounts, excluded from privacy miss penalties).

---

## Step 4: Master 1-Click Automated Benchmark Runner

Once Ground Truth is established, run the automated suite to execute all systems, evaluate against GT, generate comparison collages, and compile telemetry:

```bash
# Rapid 100-Frame Validation Run:
python3 "benchmark_suite/master_benchmark_runner.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/results_100" \
  --max-frames 100 --num-comparisons 10

# Full 7,200-Frame Benchmark Run (GPU / CUDA):
python3 "benchmark_suite/master_benchmark_runner.py" \
  --input-dir "<INPUT_DIR>" --output-dir "outputs/results_7200" \
  --max-frames 7200 --num-comparisons 20 --device cuda
```

---

## Step 5: Post-Blur Adversarial Recovery Audit (Subset Analysis)

Audit residual biometric and alphanumeric privacy leakage on successfully blurred regions:

```bash
# 1. Face Biometric Recovery (ArcFace Feature Similarity):
python3 "tools/run_recovery_audit.py" \
  --blurred-dir "outputs/anonymized_frames/fast/Clear" \
  --gt-dir "outputs/ground_truth/Clear" \
  --subset-size 500 --model "arcface" \
  --output "outputs/telemetry/recovery_face_fast_clear.json"

# 2. Plate Character Recovery (EasyOCR Alphanumeric Recognition):
python3 "tools/run_recovery_audit.py" \
  --blurred-dir "outputs/anonymized_frames/fast/Clear" \
  --gt-dir "outputs/ground_truth/Clear" \
  --subset-size 500 --model "easyocr" \
  --output "outputs/telemetry/recovery_plate_fast_clear.json"
```

---

## Step 6: Empirical Failure Rates & PP4AV Distance Stratification

Compute empirical Miss Rates ($\text{MR}_{b,c}$) against the human-audited Ground Truth stratified across the 4 PP4AV scale bins:

```bash
python3 "pipelines/FAST V5/tools/calculate_failure_rates.py" \
  --gt-dir "outputs/ground_truth" \
  --pipeline-dir "outputs/anonymized_frames" \
  --detections-dir "outputs/detections" \
  --pp4av-thresholds "face:10,plate:8" \
  --output "outputs/reports/FINAL_FAILURE_MATRIX.csv"
```

---

## Step 7: Telemetry Aggregation & Benchmark Deliverables

Compile unified host-device telemetry into consolidated CSVs and Markdown executive summaries:

```bash
python3 "pipelines/FAST V5/tools/benchmark_telemetry_aggregator.py" \
  --telemetry-dir "outputs/telemetry" \
  --output-dir "outputs/reports"
```

### Generated Benchmark Deliverables:
* `outputs/reports/TELEMETRY_BENCHMARK_REPORT.md` — Consolidated GPU + CPU Executive Report
* `outputs/reports/telemetry_table_A_gt.csv` — Baseline Oracles Telemetry Profile
* `outputs/reports/telemetry_table_B_pipelines.csv` — 5 Evaluated Production Systems Telemetry Profile
* `outputs/reports/FINAL_FAILURE_MATRIX.csv` — Miss Rates across Bins 1–4 and 6 Environmental Conditions
* `outputs/reports/charts/` — Visual figures (Latency Breakdown, Energy vs. Throughput, Failure Heatmaps)

---

## Operational Guardrails & Agent FAQ

**Q1: How is the Ground Truth (GT) produced?**  
The Ground Truth is produced via **Multi-System Candidate Fusion** (aggregating candidate proposals from all 5 evaluated systems + InsightFace and RT-DETR oracles) followed by an exhaustive **Human-in-the-Loop Audit** using the **Optimized X-AnyLabeling** desktop tool. MobileSAM provides sub-pixel boundary snapping, and annotators apply single-key class assignments (`Shift+F`/`Shift+P`), instant deletion (`Delete`), and PP4AV recognizability thresholds.

**Q2: What is the purpose of the 4 PP4AV scale bins?**  
To uncover scale-dependent perception collapse. Bin 1 ($< 16\text{ px}$) represents extreme long range ($20+\text{ m}$), Bin 2 ($16\text{–}32\text{ px}$) represents standard tracking ($10\text{–}20\text{ m}$), Bin 3 ($32\text{–}64\text{ px}$) represents active lane perception ($5\text{–}10\text{ m}$), and Bin 4 ($> 64\text{ px}$) represents critical near-field safety ($< 5\text{ m}$).

**Q3: What if an evaluation run is interrupted?**  
All pipeline runners operate in streaming mode, writing detection JSONs, images, and telemetry incrementally. If interrupted, completed frames are preserved; running the command again will safely resume without re-running existing outputs.
