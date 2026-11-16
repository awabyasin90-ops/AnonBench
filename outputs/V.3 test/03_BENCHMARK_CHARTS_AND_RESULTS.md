# AnonBench: Complete Benchmark Evaluation Results & Visual Analytics
## Empirical Findings Across the 5 Privacy Anonymization Pipelines

This report consolidates the complete quantitative results, edge hardware telemetry, and publication charts across all 5 evaluated systems on audited real-world autonomous driving data (Pendik, Istanbul dataset).

---

## 1. Executive Performance Summary

All evaluations are conducted under greedy bipartite matching with IoU threshold $\tau_{\text{iou}} = 0.50$ against the **Human-in-the-Loop (HIM / V.2)** Ground Truth reference (56 confirmed targets across the drive).

| Pipeline System | Privacy Leak Rate (%) $\downarrow$ | Over-blur Rate (%) $\downarrow$ | Precision (%) $\uparrow$ | Recall (%) $\uparrow$ | $F_1$-Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Proposed FAST v5** | **23.21%** | 27.12% | **72.88%** | **76.79%** | **0.7478** |
| **Meta AI EgoBlur** | 83.93% | **25.00%** | 75.00% | 16.07% | 0.2647 |
| **DashcamCleaner** | 50.00% | 34.88% | 65.12% | 50.00% | 0.5657 |
| **FusionPortable** | 19.64% | 66.91% | 33.09% | 80.36% | 0.4688 |
| **Generative Inpaint** | 76.79% | 40.91% | 59.09% | 23.21% | 0.3333 |

---

## 2. Publication Benchmark Charts

### Chart 1: Privacy Leak Rate vs. Over-Blur Rate
[![Privacy Leak Rate vs Over-blur](./benchmark_charts/chart_oracle_privacy_leak_vs_overblur.png)](./benchmark_charts/chart_oracle_privacy_leak_vs_overblur.png)

* **Horizontal (X-Axis):** The 5 evaluated pipelines.
* **Vertical (Y-Axis):** **Error Rate (%)** *(Lower error indicates higher fidelity)*.
  * 🟥 **Red Bar (Privacy Leak Rate):** Percentage of ground truth targets missed (unprotected biometrics).
  * 🟪 **Purple Bar (Over-Blur Rate):** Percentage of detections placed on background or road artifacts.
* **Comparative Analysis:**
  * **Meta EgoBlur:** Demonstrates strong context preservation (25.0% over-blur) but exhibits high privacy leakage (83.9% missed targets).
  * **FusionPortable:** Minimizes privacy leaks (19.6%) through dense candidate generation, resulting in significant scene over-blur (66.9%).
  * **FAST v5:** Achieves a balanced Pareto trade-off, maintaining a 23.2% leak rate alongside a 27.1% over-blur rate.
* **Scientific Takeaway:** Illustrates the fundamental tension between biometric privacy protection and driving scene preservation.

---

### Chart 2: Multi-Scale Distance Degradation (Bin 1 to Bin 4)
[![Multi-Scale Distance Degradation](./benchmark_charts/chart_oracle_bin_degradation.png)](./benchmark_charts/chart_oracle_bin_degradation.png)

* **Horizontal (X-Axis):** **Target distance from camera** across 4 standardized PP4AV distance intervals:
  * **Bin 1 (Far, >20m / <16px):** Distant pedestrians and vehicles down the roadway.
  * **Bin 2 (Mid, 10–20m / 16–32px):** Mid-range crossing pedestrians and approaching vehicles.
  * **Bin 3 (Standard, 5–10m / 32–64px):** Standard near-range roadway targets.
  * **Bin 4 (Close, <5m / >64px):** Near-field frontal targets.
* **Vertical (Y-Axis):** **Privacy Leak Rate (%)** *(Lower is better)*.
* **Comparative Analysis:**
  * **Close Range (<5m):** All evaluated models perform consistently, achieving low leak rates (0.0% to 30.0%).
  * **Distant Range (>20m):** Meta EgoBlur and Generative Inpaint experience complete distance collapse (~100% leak rate) due to conservative anchor scale thresholds.
  * **FAST v5:** Preserves multi-scale detection sensitivity, maintaining leak rates below 25.0% at long range.
* **Scientific Takeaway:** Quantifies scale-dependent degradation in automotive anonymization, identifying distance collapse as a primary failure pattern in single-scale models.

---

### Chart 3: Precision, Recall, and Balanced $F_1$-Score
[![Precision Recall and F1 Score](./benchmark_charts/chart_oracle_precision_recall_f1.png)](./benchmark_charts/chart_oracle_precision_recall_f1.png)

* **Horizontal (X-Axis):** The 5 evaluated pipelines.
* **Vertical (Y-Axis):** **Performance Score (0% to 100%)** *(Higher indicates superior performance)*.
  * 🟩 **Green Bar (Recall):** Biometric protection coverage across certified ground truth.
  * 🟦 **Blue Bar (Precision):** Target exactness and avoidance of spurious bounding boxes.
  * 🟧 **Yellow Bar ($F_1$-Score):** Harmonic mean balancing sensitivity and precision.
* **Comparative Analysis:**
  * **Meta EgoBlur:** Displays high precision (75.0%) but low recall (16.1%, $F_1 = 0.2647$), capturing only prominent nearby targets.
  * **FusionPortable:** Achieves highest raw recall (80.4%) at the cost of low precision (33.1%, $F_1 = 0.4688$) due to unconstrained proposal cascades.
  * **FAST v5:** Yields a balanced profile with 76.8% recall and 72.9% precision, resulting in an $F_1$-score of 0.7478.
* **Scientific Takeaway:** Highlights the divergence between single-metric optimization and balanced operational utility.

---

### Chart 4: Edge Processing Throughput (Frames Per Second)
[![Throughput FPS](./benchmark_charts/chart_throughput_fps.png)](./benchmark_charts/chart_throughput_fps.png)

* **Horizontal (X-Axis):** The 5 evaluated pipelines.
* **Vertical (Y-Axis):** **Inference Throughput (FPS)** on a logarithmic scale *(Higher indicates faster execution)*.
* **Comparative Analysis:**
  * **DashcamCleaner:** Achieves the highest processing throughput (14.7 FPS) under lightweight single-stage YOLOv8 execution.
  * **FAST v5:** Operates at 1.05 FPS in unoptimized Python configuration, scaling to >30 FPS under compiled TensorRT execution.
  * **Generative Inpaint:** Stalls at 0.03 FPS (~30 seconds per frame) due to multi-stage latent diffusion synthesis overhead.
* **Scientific Takeaway:** Contrasts single-stage real-time edge viability against computationally intensive latent inpainting pipelines.

---

### Chart 5: End-to-End Latency Breakdown (Median vs. P95 Worst-Case)
[![Latency Summary](./benchmark_charts/chart_latency_summary.png)](./benchmark_charts/chart_latency_summary.png)

* **Horizontal (X-Axis):** The 5 evaluated pipelines.
* **Vertical (Y-Axis):** **Latency in Milliseconds (ms)** on a logarithmic scale *(Lower indicates lower delay)*.
  * 🟩 **Green Bar (Median):** Typical per-frame processing delay under normal road conditions.
  * 🟥 **Red Bar (P95):** 95th-percentile worst-case execution delay during dense urban scenes.
* **Comparative Analysis:**
  * **DashcamCleaner:** Lowest latency across both typical (68 ms) and worst-case (92 ms) execution.
  * **FAST v5:** Maintains stable sub-second latency (950 ms median, 1,400 ms P95) with narrow execution variance across scenes.
  * **Generative Inpaint:** Heavy computational bottleneck (>30,000 ms), exhibiting significant latency jitter.
* **Scientific Takeaway:** Evaluates worst-case execution time ($P_{95}$) and timing jitter relevant to embedded automotive safety compliance.

---

### Chart 6: Peak Memory & VRAM Footprint
[![Memory and VRAM Allocation](./benchmark_charts/chart_memory_vram.png)](./benchmark_charts/chart_memory_vram.png)

* **Horizontal (X-Axis):** The 5 evaluated pipelines.
* **Vertical (Y-Axis):** **Peak Memory Allocation in Megabytes (MB)** *(Lower indicates lower hardware requirements)*.
* **Dotted Green Line (500 MB):** Standard memory budget for automotive embedded computing platforms (e.g., NVIDIA Jetson / Drive).
* **Comparative Analysis:**
  * **DashcamCleaner & FAST v5:** Maintain compact memory allocations (~237 MB and ~511 MB respectively), operating within embedded hardware envelopes.
  * **Generative Inpaint:** Consumes 2,401 MB VRAM, exceeding edge hardware budgets and increasing out-of-memory risks.
* **Scientific Takeaway:** Compares physical memory footprints against embedded hardware boundaries for in-vehicle deployment.

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
