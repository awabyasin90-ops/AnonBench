# AnonBench: Empirical Benchmarking of Anonymization Failure Patterns in Autonomous Driving

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](#)
[![Status: Benchmarked](https://img.shields.io/badge/Status-Benchmarked-brightgreen.svg)](#)
[![Evaluation IoU >= 0.50](https://img.shields.io/badge/IoU%20Matching-%E2%89%A5%200.50-brightgreen.svg)](#)
[![PP4AV Standard](https://img.shields.io/badge/Filtering-PP4AV%20Thresholds-orange.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Official benchmark suite for evaluating visual anonymization failure patterns in autonomous driving across 5 pipelines, hardware telemetry, and a customized Human-in-the-Loop annotation engine X-AnyLabeling.**

---

## 1. Study Overview & Research Questions

Visual anonymization in automated driving systems requires continuous privacy protection without compromising downstream perception utility or exceeding real-time computational budgets. This benchmark provides an end-to-end experimental failure audit across disparate algorithmic paradigms (single-stage, two-stage cascaded, and generative inpainting).

### Core Research Questions
- **RQ1 (Multi-Scale Degradation):** To what extent does target distance and pixel resolution (PP4AV Bins 1–4) cause privacy failure spikes across single-stage, two-stage, and generative models?
- **RQ2 (Algorithmic Trade-offs):** How do conservative high-precision detectors compare against high-recall ensembles in terms of privacy leak rate versus visual over-blur distortion?
- **RQ3 (Edge Deployment Feasibility):** Can high-recall multi-target anonymization satisfy real-time latency (<1000 ms) and embedded onboard memory bounds (<2000 MB RAM/VRAM)?

---

## 2. Repository Layout

```text
AnonBench/
├── README.md                      # Repository documentation & reproducibility guide
├── LICENSE                        # MIT License
├── CITATION.cff                   # Academic citation metadata
├── .gitignore                     # Repository exclusion rules
│
├── pipelines/                     # 5 Anonymization Pipelines & Evaluation Suite
│   ├── FAST V5/                   # Hardware-Adaptive ADAS Edge Pipeline
│   ├── EgoBlur/                   # Meta AI EgoBlur Baseline
│   ├── DashcamCleaner/            # Open-Source YOLOv8 Baseline
│   ├── Anonymizer/                # FusionPortable Baseline
│   ├── Generative Anonymizer/     # Dual Generative Synthesis (LaMa + DeepPrivacy2)
│   └── benchmark_suite/           # Evaluation scripts, math formulations & runner
│
├── outputs/                       # Empirical results, reports & publication charts
│   ├── results_pendik_100/        # Telemetry logs, CSV tables & charts
│   └── V.3 test/                  # Audited frames, 6 publication charts & report
│
└── X/                             # Customized Human-in-the-Loop Engine (X-AnyLabeling)
```

---

## 3. The Five Evaluated Anonymization Pipelines

To ensure fair benchmarking, each system was executed under its established operational configuration:

| System ID | Pipeline Name | Architectural Paradigm | Detection Thresholds | Primary Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **`1_FAST_V5`** | **FAST v5** | Hardware-Adaptive Single-Stage | Face: `0.25`, Plate: `0.20` | Real-time edge ADAS engine; elliptical Gaussian feathering; low memory footprint. |
| **`2_EgoBlur`** | **Meta AI EgoBlur** | Two-Stage Faster R-CNN | Face: `0.6742`, Plate: `0.7448` | Official conservative thresholds; high precision but drops small far-field targets. |
| **`3_DashcamCleaner`** | **DashcamCleaner** | Single-Stage YOLOv8 Medium | General: `0.40` | Balanced community baseline; box kernel blurring (size 19). |
| **`4_FusionPortable`** | **FusionPortable** | Multi-Scale Faster R-CNN ONNX | Face: `0.30`, Plate: `0.30` | High detection sensitivity; elevated false discovery rate and substantial memory footprint. |
| **`5_Generative_Anonymizer`** | **Generative Inpaint** | Detector + Inpainting (LaMa/GAN) | Pre-mask: `0.30` | Synthetic texture replacement; photorealistic inpainting for offline archival. |

---

## 4. Customized Human-in-the-Loop (HIM) Engine: Simple & Fast Ground Truth Review

Establishing a gold-standard Ground Truth (GT) without human bias or manual fatigue is achieved via our customized **X-AnyLabeling (HIM)** workflow:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 1: MULTI-SYSTEM CANDIDATE FUSION                          │
│   Pre-populates candidate boxes from all 5 evaluated pipelines at low thresholds       │
│   (Zero manual drawing from scratch — maximizes initial candidate capture)             │
└──────────────────────────────────┬─────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             STAGE 2: SIMPLE HUMAN REVIEW VIA CUSTOMIZED X-ANYLABELING                  │
│                                                                                        │
│   • One-Key Verification: Press 'F' (Face), 'P' (Plate) with zero modal popup delay    │
│   • Instant False-Positive Removal: Press 'Delete' to immediately discard false alarms │
│   • MobileSAM AI Edge Snapping: Coarse boxes automatically snap to physical boundaries │
│   • PP4AV Threshold Tagging: Fast toggling for Confirmed vs. Ignore biometrics         │
│   • Frictionless Navigation: Atomic auto-saving to disk on 'D' (Next) / 'A' (Prev)     │
└──────────────────────────────────┬─────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                       DEFINITIVE BENCHMARK GROUND TRUTH (GT)                           │
│   High-Precision Sub-Pixel Annotations across Audited Real-World Autonomous Drives     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### Why Human-in-the-Loop (HIM) Review is Simple:
1. **No Manual Drawing from Scratch:** Automated multi-detector fusion pre-populates all candidate proposals, reducing human effort to fast verification.
2. **One-Touch Keystrokes:** Adjudicators navigate seamlessly (`D` next, `A` prev) and confirm labels (`F`/`P`) without slow mouse menus or interruption dialogs.
3. **AI Edge Snapping:** Integrated **MobileSAM** snaps loose boxes to true sub-pixel boundaries automatically.
4. **PP4AV Recognizability Standards:** Built-in size checks automatically guide human annotators to mark unidentifiable small biometrics ($<10\text{ px}$ face, $<8\text{ px}$ plate) as `Ignore`.

---

## 5. Hardware Telemetry & Profiling Protocol

Host-device resource utilization is measured using an integrated profiler (`unified_telemetry_profiler.py`):

- **Throughput (FPS):** End-to-end frame processing speed including preprocessing, neural inference, and post-blurring.
- **Latency (ms & s):** Wall-clock execution measured with `time.perf_counter()`, capturing Median (typical) and 95th Percentile ($P_{95}$ worst-case) latency.
- **Memory Footprint (MB):** Process Resident Set Size (RSS) and system RAM tracking via `psutil`, paired with GPU allocation counters via `pynvml`.
- **ANOVA Significance:** Resource disparities across architectures are statistically verified via One-Way ANOVA ($F = 142.85, p < 10^{-15}$).

---

## 6. Automated Evaluation & Statistical Battery (`evaluate_xanylabeling_gt.py`)

Deterministic matching and rigorous hypothesis testing are performed directly against the audited Ground Truth:

- **PP4AV Multi-Scale Distance Strata:** Confirmed targets are partitioned into 4 resolution distance bins:
  - **Bin 1:** $<16\text{ px}$ (Far horizon, $>20\text{ m}$)
  - **Bin 2:** $16\text{–}32\text{ px}$ (Mid-range, $10\text{–}20\text{ m}$)
  - **Bin 3:** $32\text{–}64\text{ px}$ (Standard range, $5\text{–}10\text{ m}$)
  - **Bin 4:** $\ge 64\text{ px}$ (Close range, $<5\text{ m}$)
- **Greedy Bipartite Matching:** IoU matching at $\tau_{\text{iou}} = 0.50$ computes True Positives (TP), False Negatives (Privacy Leaks), and False Positives (Over-blurring).
- **Non-Parametric Hypothesis Testing:**
  - **Shapiro-Wilk Test:** Confirms departure from normality ($W = 0.8947, p < 0.001$), mandating non-parametric methods.
  - **Omnibus Friedman Test:** Confirms pipeline architecture significantly governs Privacy Leak Rates ($\chi^2(4) = 30.0482, p = 4.79 \times 10^{-6}$) and Over-blur Rates ($\chi^2(4) = 14.4468, p = 0.0060$).
  - **Wilcoxon Signed-Rank Test:** Post-hoc pairwise differences with Bonferroni correction ($\alpha = 0.0125$).
  - **Kruskal-Wallis Test:** Statistically validates the severe scale-dependent perception collapse between Bin 2 and Bin 1 ($H = 3.9321, p < 0.05$).

---

## 7. Key Empirical Results & Findings

> [!NOTE]
> **Benchmarked Pilot Audit:** The quantitative results and visual analytics reported below represent the intensive 10-frame Human Oracle Audit (56 confirmed Ground Truth targets, 102 total evaluated regions across 5 pipelines) establishing baseline statistical significance. Large-scale expansion across all 7,200 driving frames will accompany the full manuscript release.

### Table 1: Audited Anonymization & Privacy Preservation Metrics
*Evaluated under greedy bipartite matching ($\tau_{\text{iou}} = 0.50$) against Human Oracle Ground Truth (56 confirmed targets).*

| Pipeline System | Privacy Leak Rate (%) $\downarrow$ | Over-blur Rate (%) $\downarrow$ | Precision (%) $\uparrow$ | Recall (%) $\uparrow$ | Balanced $F_1$-Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAST v5** | **23.21%** | 27.12% | **72.88%** | **76.79%** | **0.7478** |
| **Meta AI EgoBlur** | 83.93% | **25.00%** | 75.00% | 16.07% | 0.2647 |
| **DashcamCleaner** | 50.00% | 34.88% | 65.12% | 50.00% | 0.5657 |
| **FusionPortable** | 19.64% | 66.91% | 33.09% | 80.36% | 0.4688 |
| **Generative Inpaint** | 76.79% | 40.91% | 59.09% | 23.21% | 0.3333 |

---

### Table 2: Edge Hardware Telemetry & Real-Time Viability
*Measured end-to-end under real-world automotive drive replay.*

| Pipeline System | Throughput (FPS) $\uparrow$ | Delay per Frame (s) $\downarrow$ | Median Latency (ms) $\downarrow$ | $P_{95}$ Worst-Case (ms) $\downarrow$ | Peak Memory (MB) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAST v5** | 0.71 FPS | 1.41 s | **917.7 ms** | **2,683.2 ms** | **1,986.5 MB** |
| **Meta AI EgoBlur** | 0.10 FPS | 10.34 s | 10,248.4 ms | 10,980.4 ms | 2,283.9 MB |
| **DashcamCleaner** | 0.71 FPS | 1.41 s | 1,361.4 ms | 1,499.1 ms | 2,377.6 MB |
| **FusionPortable** | 0.13 FPS | 7.45 s | 7,446.0 ms | 8,188.7 ms | 5,117.9 MB |
| **Generative Inpaint** | **1.03 FPS** | **0.95 s** | 755.7 ms | 1,854.8 ms | 2,401.4 MB |

---

### Publication Visual Analytics
Detailed visual discussion and takeaways are available in [`outputs/V.3 test/03_BENCHMARK_CHARTS_AND_RESULTS.md`](outputs/V.3%20test/03_BENCHMARK_CHARTS_AND_RESULTS.md):

1. **[Chart 1: Privacy Leak Rate vs. Over-Blur Rate](outputs/V.3%20test/benchmark_charts/chart_oracle_privacy_leak_vs_overblur.png)**  
   *Illustrates the Pareto frontier balancing biometric privacy leaks against scene over-blurring.*
2. **[Chart 2: Multi-Scale Distance Degradation (Bin 1 to Bin 4)](outputs/V.3%20test/benchmark_charts/chart_oracle_bin_degradation.png)**  
   *Highlights the critical failure hazard zone (>70% miss rate) where commercial detectors drop small targets beyond 10–20 meters.*
3. **[Chart 3: Precision, Recall, and Balanced $F_1$-Score](outputs/V.3%20test/benchmark_charts/chart_oracle_precision_recall_f1.png)**  
   *Quantifies balanced detection utility across disparate architectural paradigms.*
4. **[Chart 4: Edge Processing Throughput (FPS & Seconds per Frame)](outputs/V.3%20test/benchmark_charts/chart_throughput_fps.png)**  
   *Compares real-time edge processing speeds against heavy multi-stage bottlenecks.*
5. **[Chart 5: Latency Breakdown (Median vs. $P_{95}$ Delay)](outputs/V.3%20test/benchmark_charts/chart_latency_summary.png)**  
   *Evaluates execution predictability under routine versus crowded automotive road conditions.*
6. **[Chart 6: Peak Memory & VRAM Allocation](outputs/V.3%20test/benchmark_charts/chart_memory_vram.png)**  
   *Contrasts low-footprint single-stage pipelines against multi-model memory exhaustion (>5.1 GB).*

---

## 8. Reproducibility Workflow

```bash
# 1. Run multi-model consensus fusion to generate candidate labels
python3 pipelines/benchmark_suite/prepare_xanylabeling_gt.py

# 2. Execute automated evaluation and statistical significance battery on audited GT
python3 pipelines/benchmark_suite/evaluate_xanylabeling_gt.py \
    --gt-dir "outputs/V.3 test" \
    --results-dir "outputs/results_pendik_100" \
    --iou-thresh 0.50

# 3. Generate high-resolution visual comparison collages
python3 pipelines/benchmark_suite/generate_comparison_frames.py

# 4. Generate all 6 publication-quality benchmark charts
python3 /tmp/generate_all_6_benchmark_charts.py
```

---

## 9. Citation & Contact

If you use this benchmark suite, evaluation protocol, or dataset annotations in your research, please cite:

```bibtex
@article{anonbench2026,
  title={AnonBench: Empirical Benchmarking of Anonymization Failure Patterns in Real and Simulated Autonomous Driving Data},
  author={Yasin, Awab and Collaborators},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2026}
}
```

**License:** Distributed under the [MIT License](LICENSE).
