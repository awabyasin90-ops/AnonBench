# AnonBench: Empirical Benchmarking of Anonymization Failure Patterns in Autonomous Driving

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Status: Under Test](https://img.shields.io/badge/Status-Under%20Test-orange.svg)](#)
[![Evaluation IoU >= 0.50](https://img.shields.io/badge/IoU%20Matching-%E2%89%A5%200.50-brightgreen.svg)](#)
[![PP4AV Standard](https://img.shields.io/badge/Filtering-PP4AV%20Thresholds-orange.svg)](#)
[![Companion Repo](https://img.shields.io/badge/Companion%20Repo-FAST--Edge--Anonymization-purple.svg)](https://github.com/awabyasin90-ops/FAST-Edge-Anonymization)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official benchmarking suite, human oracle auditing interface, and automated statistical evaluation engine for empirical research investigating anonymization failure patterns, multi-scale distance collapse, and edge hardware trade-offs across real drives and autonomous driving simulation.

---

## 1. Study Overview & Research Questions

Visual anonymization in automated driving systems requires continuous privacy protection without compromising downstream perception utility or exceeding real-time computational budgets. This benchmark provides an end-to-end experimental failure audit across disparate algorithmic paradigms.

### Core Research Questions
- **RQ1 (Multi-Scale Degradation):** To what extent does target distance and pixel resolution (PP4AV Bins 1–4) cause privacy failure spikes across single-stage, two-stage, and generative models?
- **RQ2 (Algorithmic Trade-offs):** How do conservative high-precision detectors compare against high-recall ensembles in terms of privacy leak rate versus visual over-blur distortion?
- **RQ3 (Edge Deployment Feasibility):** Can high-recall multi-target anonymization satisfy real-time latency (<1000 ms) and embedded onboard memory bounds (<2000 MB RAM/VRAM)?

---

## 2. Repository Layout

```text
3_Systems/
├── README.md                      # Repository documentation & reproducibility guide
├── LICENSE                        # MIT License
├── CITATION.cff                   # Academic citation metadata
├── .gitignore                     # Repository exclusion rules
│
├── pipelines/                     # 5 Anonymization Pipelines & Evaluation Suite
│   ├── FAST V5/                   # Proposed Hardware-Adaptive ADAS Edge Pipeline
│   ├── EgoBlur/                   # Meta AI EgoBlur Baseline
│   ├── DashcamCleaner/            # Open-Source YOLOv8 Baseline
│   ├── Anonymizer/                # FusionPortable Baseline
│   ├── Generative Anonymizer/     # Dual Generative Synthesis (LaMa + DeepPrivacy2)
│   └── benchmark_suite/           # Evaluation scripts, math formulations & runner
│
├── outputs/                       # Empirical results, reports & publication charts
└── X/                             # Human oracle annotation tooling (X-AnyLabeling)
```

---

## 3. The Five Evaluated Anonymization Pipelines

To ensure fair benchmarking, each system was executed under its established operational configuration:

| System ID | Pipeline Name | Architectural Paradigm | Detection Thresholds | Primary Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **`1_FAST_V5`** | **Proposed FAST v5** | Hardware-Adaptive Single-Stage | Face: `0.25`, Plate: `0.20` | Real-time edge ADAS engine; elliptical Gaussian feathering; low memory footprint. |
| **`2_EgoBlur`** | **Meta AI EgoBlur** | Two-Stage Faster R-CNN | Face: `0.6742`, Plate: `0.7448` | Official conservative thresholds; high precision but drops small far-field targets. |
| **`3_DashcamCleaner`** | **DashcamCleaner** | Single-Stage YOLOv8 Medium | General: `0.40` | Balanced community baseline; box kernel blurring (size 19). |
| **`4_FusionPortable`** | **FusionPortable** | Multi-Scale Faster R-CNN ONNX | Face: `0.30`, Plate: `0.30` | High detection sensitivity; elevated false discovery rate and substantial memory footprint. |
| **`5_Generative_Anonymizer`** | **Generative Inpaint** | Detector + Inpainting (LaMa/GAN) | Pre-mask: `0.30` | Synthetic texture replacement; photorealistic inpainting for offline archival. |

---

## 4. Hardware Telemetry & Profiling Protocol

Host-device resource utilization is measured using an integrated profiler (`unified_telemetry_profiler.py`):

- **Throughput (FPS):** End-to-end frame processing speed including preprocessing, model forward passes, and image encoding.
- **Latency (ms):** Wall-clock execution measured with `time.perf_counter()`, recording Mean, Median, and 95th Percentile (P95) latency.
- **Memory (MB):** Process Resident Set Size (RSS) and system RAM tracking via `psutil`, paired with GPU memory counters via `pynvml` where available.
- **Variance Analysis:** Latency disparities among architectures are statistically verified using One-Way ANOVA ($F = 142.85, p < 10^{-15}$).

---

## 5. Human Oracle Ground-Truth Methodology

Ground truth evaluation avoids single-detector bias through a two-stage hybrid protocol:

1. **Candidate Consensus Pre-Labeling (`prepare_xanylabeling_gt.py`):** Combines candidate predictions across detectors using class-aware IoU deduplication ($\text{IoU} \ge 0.35$ or $\text{IoMin} \ge 0.50$), producing 102 initial proposals (86 faces, 16 plates) across 10 diverse audit frames.
2. **Human Oracle Audit (X-AnyLabeling):** Human adjudicators inspect every proposed region, adding missed targets and pruning spurious proposals.
3. **Standardized PP4AV Recognizability Filtering:** Following the established PP4AV protocol, targets with face width $<10\text{ px}$ or license plate height $<8\text{ px}$ lack identifiable biometrics and are tagged as `Ignore` (42 faces, 4 plates; 46 total). This yields exactly **56 Confirmed Ground Truth targets** (44 faces, 12 plates) for confusion matrix matching and statistical testing.
4. **Ergonomic Accelerations:** Custom source bindings enable atomic saving on frame navigation, instant scroll-wheel zooming for small scale verification, and direct keyboard shortcuts for face/plate toggling.

---

## 6. Automated Evaluation & Statistical Battery (`evaluate_xanylabeling_gt.py`)

The evaluation script carries out deterministic matching and statistical verification:

- **Multi-Scale Distance Strata:** Confirmed targets are partitioned into 4 resolution bins:
  - **Bin 1:** $<16\text{ px}$ (Far horizon, $>20\text{ m}$)
  - **Bin 2:** $16\text{–}32\text{ px}$ (Mid-range, $10\text{–}20\text{ m}$)
  - **Bin 3:** $32\text{–}64\text{ px}$ (Standard range, $5\text{–}10\text{ m}$)
  - **Bin 4:** $\ge 64\text{ px}$ (Close range, $<5\text{ m}$)
- **Greedy Bipartite Matching:** IoU bipartite matching ($\text{IoU} \ge 0.50$) computes TP, FN (privacy leaks), and FP (over-blurring).
- **Hypothesis Testing (`scipy.stats`):**
  - **Shapiro-Wilk Test:** Confirms departure from normality ($W = 0.8947, p = 3.23 \times 10^{-4}$), mandating non-parametric methods.
  - **Omnibus Friedman Test:** Confirms pipeline architecture significantly governs Privacy Leak Rates ($\chi^2(4) = 30.0482, p = 4.79 \times 10^{-6}$) and Over-blur Rates ($\chi^2(4) = 14.4468, p = 0.0060$).
  - **Wilcoxon Signed-Rank Test:** Pairwise post-hoc tests with Bonferroni correction ($\alpha = 0.0125$).
  - **Kruskal-Wallis & Mann-Whitney U Tests:** Quantifies scale-dependent degradation from Bin 2 to Bin 1 ($H = 3.9321, p = 0.0474$; $U = 1119.0, p = 0.0479$).

---

## 7. Key Results Summary

> [!NOTE]
> **Pilot Study Notice:** The quantitative benchmarks reported below represent the 10-frame high-density Human Oracle Audit (56 confirmed Ground Truth targets, 102 total evaluated regions across 5 pipelines) conducted to rigorously validate the evaluation battery and establish statistical significance. Full-scale large-scale benchmark results across 7,200 driving frames will be released upon manuscript publication.

### Confusion Matrix Across Reviewed Frames
| Pipeline | GT Targets | Detections | True Pos. | False Neg. | False Pos. | Privacy Leak Rate | Over-blur Rate | Recall | Precision | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed FAST v5** | 56 | 53 | **18** | 38 | 35 | **67.9%** | 66.0% | **32.1%** | 34.0% | **33.0** |
| **Meta AI EgoBlur** | 56 | 11 | **10** | 46 | 1 | **82.1%** | 9.1% | **17.9%** | 90.9% | **29.9** |
| **DashcamCleaner** | 56 | 45 | **23** | 33 | 22 | **58.9%** | 48.9% | **41.1%** | 51.1% | **45.5** |
| **FusionPortable** | 56 | 112 | **41** | 15 | 71 | **26.8%** | 63.4% | **73.2%** | 36.6% | **48.8** |
| **Generative Inpaint** | 56 | 16 | **7** | 49 | 9 | **87.5%** | 56.2% | **12.5%** | 43.8% | **19.4** |

### Publication Figures
- **Trade-off Analysis:** [Privacy Leak vs. Over-blur](outputs/results_pendik_100/charts/chart_oracle_privacy_leak_vs_overblur.png)
- **Detection Metrics:** [Precision, Recall & F1](outputs/results_pendik_100/charts/chart_oracle_precision_recall_f1.png)
- **Distance Collapse:** [Distance Degradation Across Bins](outputs/results_pendik_100/charts/chart_oracle_bin_degradation.png)

---

## 8. Reproducibility Workflow

```bash
# 1. Run multi-model consensus fusion to generate candidate labels
python3 pipelines/benchmark_suite/prepare_xanylabeling_gt.py

# 2. Execute automated evaluation and statistical significance battery on audited GT
python3 pipelines/benchmark_suite/evaluate_xanylabeling_gt.py \
    --gt-dir "outputs/V.2" \
    --results-dir "outputs/results_pendik_100" \
    --iou-thresh 0.50

# 3. Generate high-resolution visual comparison collages
python3 pipelines/benchmark_suite/generate_comparison_frames.py
```

---

## 9. Citation & Contact

If you use this benchmark suite, evaluation protocol, or dataset annotations in your research, please cite:

```bibtex
@article{anonbench2026,
  title={Empirical Benchmarking of Anonymization Failure Patterns in Real and Simulated Autonomous Driving Data},
  author={},
  journal={},
  year={2026}
}
```

**Contact:**
- For technical questions, bug reports, or benchmark extensions, please open an issue in this repository.

**License:** Distributed under the [MIT License](LICENSE).
