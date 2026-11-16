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
  * 🟥 **Solid Red Line (Privacy Leak Rate):** Percentage of missed people and license plates left unblurred.
  * 🟪 **Dashed Purple Line (Over-Blur Rate):** Percentage of mistaken blur placed on empty roads, signs, or scenery.
* **Main Findings:**
  * 🟢 <span style="color:#15803d">**Best Balanced System:**</span> **FAST v5** — Optimal Pareto trade-off (**23.2%** leak, **27.1%** over-blur).
  * 🔴 <span style="color:#b91c1c">**Worst Privacy Leak:**</span> **Meta AI EgoBlur** — Critical privacy failure (**83.9%** leak; missed 47 of 56 targets).
  * 🔴 <span style="color:#b91c1c">**Worst Scene Destruction:**</span> **FusionPortable** — Excessive false blurring (**66.9%** over-blur), blinding driving perception.
* **Takeaway:** Overly aggressive models blind driving perception; overly conservative models leak private personal identities.

---

### Chart 2: Multi-Scale Distance Degradation (Bin 1 to Bin 4)
[![Multi-Scale Distance Degradation](./benchmark_charts/chart_oracle_bin_degradation.png)](./benchmark_charts/chart_oracle_bin_degradation.png)

* **Axes:** X = Target Distance (Far >20m to Close <5m) | Y = Privacy Leak Rate (%) *(Lower is better)*.
* **Main Findings:**
  * 🟢 <span style="color:#15803d">**Best Distance Resilience:**</span> **FAST v5** — Resilient across all scales (**24.3%** at Far >20m down to **2.5%** at Close <5m).
  * 🔴 <span style="color:#b91c1c">**Worst Distance Collapse:**</span> **Meta AI EgoBlur** (**89.2%** leak) and **Generative Inpaint** (**83.8%** leak) at Far range (>20m).
* **Takeaway:** Target distance is the #1 cause of visual privacy leaks; FAST v5 prevents distant perception collapse across the entire camera range.

---

### Chart 3: Precision, Recall, and Balanced $F_1$-Score
[![Precision Recall and F1 Score](./benchmark_charts/chart_oracle_precision_recall_f1.png)](./benchmark_charts/chart_oracle_precision_recall_f1.png)

* **Axes:** X = 5 Pipelines | Y = Score (%) *(Higher is better)*.
  * 🟩 **Recall:** Privacy coverage | 🟦 **Precision:** Target exactness | 🟧 **$F_1$-Score:** Balanced utility.
* **Main Findings:**
  * 🟢 <span style="color:#15803d">**Best Overall Performance:**</span> **FAST v5** — Highest balanced score (**$F_1 = 0.7478$**; **76.8%** recall, **72.9%** precision).
  * 🔴 <span style="color:#b91c1c">**Worst Overall Utility:**</span> **Meta AI EgoBlur** — Low score (**$F_1 = 0.2647$**) due to failure in recall (**16.1%**).
  * 🔴 <span style="color:#b91c1c">**Worst Target Exactness:**</span> **FusionPortable** — Low precision (**33.1%**, $F_1 = 0.4688$) due to unconstrained false alarms.
* **Takeaway:** Optimizing for a single metric fails; only balanced precision and recall ensure safe, practical automotive anonymization.

---

### Chart 4: Edge Processing Speed (FPS & Seconds per Frame)
[![Throughput FPS](./benchmark_charts/chart_throughput_fps.png)](./benchmark_charts/chart_throughput_fps.png)

* **Axes:** X = 5 Pipelines | Y = Processing Speed in FPS [Log Scale] *(Higher is faster)*.
* **Main Findings:**
  * 🟢 <span style="color:#15803d">**Best Edge Viability:**</span> **FAST v5 & DashcamCleaner** (**0.71 FPS**, **1.41s**/frame baseline; scales to **>30 FPS** with TensorRT) and **Generative Inpaint** (**1.03 FPS**, **0.95s**/frame in test mode).
  * 🔴 <span style="color:#b91c1c">**Worst Speed Bottleneck:**</span> **Meta AI EgoBlur** (**0.10 FPS**, **10.34s**/frame) and **FusionPortable** (**0.13 FPS**, **7.45s**/frame), unviable for live driving feeds.
* **Takeaway:** Heavy dual-stage and cascaded models (EgoBlur, FusionPortable) take 7–10 seconds per frame, making real-time automotive deployment impossible without edge-optimized architectures.

---

### Chart 5: End-to-End Latency Breakdown (Median vs. P95 Worst-Case Delay)
[![Latency Summary](./benchmark_charts/chart_latency_summary.png)](./benchmark_charts/chart_latency_summary.png)

* **Axes:** X = 5 Pipelines | Y = Latency in Seconds and Milliseconds [Log Scale] *(Lower is faster)*.
  * 🟩 **Median Delay (Typical)** | 🟥 **P95 Worst-Case Delay (Crowded Scenes)**.
* **Main Findings:**
  * 🟢 <span style="color:#15803d">**Best Sub-Second Response:**</span> **Generative Inpaint** (**0.76s** median) and **FAST v5** (**0.92s** / 918 ms median under routine driving).
  * 🔴 <span style="color:#b91c1c">**Worst Delay & Lag:**</span> **Meta AI EgoBlur** (**10.25s** median, **10.98s** P95 worst-case), creating hazardous processing delays.
* **Takeaway:** Safety-critical automotive applications require predictable sub-second latency; delays exceeding 5 seconds create hazardous situational blindness.

---

### Chart 6: Peak Memory & VRAM Allocation
[![Memory and VRAM Allocation](./benchmark_charts/chart_memory_vram.png)](./benchmark_charts/chart_memory_vram.png)

* **Axes:** X = 5 Pipelines | Y = Peak RAM/VRAM Memory in Megabytes (MB) *(Lower is better)*.
* **Main Findings:**
  * 🟢 <span style="color:#15803d">**Best Memory Efficiency:**</span> **FAST v5** — Lowest footprint at **1,986.5 MB**, fitting comfortably within edge envelopes.
  * 🔴 <span style="color:#b91c1c">**Worst Memory Hog:**</span> **FusionPortable** — Demands a massive **5,117.9 MB** (>5.1 GB), risking out-of-memory crashes on embedded hardware.
* **Takeaway:** Edge deployment requires a compact memory footprint so the vehicle has enough RAM to run autonomous driving and navigation stacks concurrently.

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
