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

> **Key Finding:** Meta EgoBlur misses 83.9% of targets; FusionPortable over-blurs 66.9% of the scene; FAST v5 achieves a balanced Pareto trade-off (23.2% leak, 27.1% over-blur).  
> **Takeaway:** Demonstrates the trade-off between biometric privacy protection and drivable scene preservation.

---

### Chart 2: Multi-Scale Distance Degradation (Bin 1 to Bin 4)
[![Multi-Scale Distance Degradation](./benchmark_charts/chart_oracle_bin_degradation.png)](./benchmark_charts/chart_oracle_bin_degradation.png)

> **Key Finding:** Beyond 20m, standard models collapse completely (EgoBlur: 89.2% leak; Generative: 83.8% leak), whereas FAST v5 maintains 24.3% leak down to 2.5% at close range via multi-scale anchors.  
> **Takeaway:** Target distance is the primary driver of anonymization failure; single-scale models suffer severe perception collapse beyond 10 meters.

---

### Chart 3: Precision, Recall, and Balanced $F_1$-Score
[![Precision Recall and F1 Score](./benchmark_charts/chart_oracle_precision_recall_f1.png)](./benchmark_charts/chart_oracle_precision_recall_f1.png)

> **Key Finding:** Meta EgoBlur captures only prominent targets (16.1% recall, $F_1 = 0.2647$); FusionPortable creates dense false alarms (33.1% precision, $F_1 = 0.4688$); FAST v5 achieves balanced utility (76.8% recall, 72.9% precision, $F_1 = 0.7478$).  
> **Takeaway:** Asymmetric metric optimization fails; automotive deployment demands joint optimization of precision and recall.

---

### Chart 4: Edge Processing Speed (FPS & Seconds per Frame)
[![Throughput FPS](./benchmark_charts/chart_throughput_fps.png)](./benchmark_charts/chart_throughput_fps.png)

> **Key Finding:** Single-stage models (Generative: 1.03 FPS / 0.95s; FAST v5: 0.71 FPS / 1.41s; DashcamCleaner: 0.71 FPS / 1.41s) run in near real-time, while cascaded models (FusionPortable: 7.45s; EgoBlur: 10.34s) introduce severe processing bottlenecks.  
> **Takeaway:** Heavy two-stage and cascaded architectures introduce compute overhead that precludes in-vehicle real-time execution.

---

### Chart 5: End-to-End Latency Breakdown (Median vs. P95 Worst-Case Delay)
[![Latency Summary](./benchmark_charts/chart_latency_summary.png)](./benchmark_charts/chart_latency_summary.png)

> **Key Finding:** FAST v5 delivers sub-second typical latency (0.92s / 918 ms) with predictable $P_{95}$ bounds (2.68s); Meta EgoBlur requires over 10.2s per frame regardless of scene complexity.  
> **Takeaway:** Safety compliance requires predictable latency bounds ($P_{95}$) to prevent hazardous situational perception lag.

---

### Chart 6: Peak Memory & VRAM Allocation
[![Memory and VRAM Allocation](./benchmark_charts/chart_memory_vram.png)](./benchmark_charts/chart_memory_vram.png)

> **Key Finding:** FAST v5 maintains the lowest memory footprint (1,986.5 MB), operating safely within embedded hardware limits, while FusionPortable consumes 5,117.9 MB (>5.1 GB) due to concurrent model loading.  
> **Takeaway:** Visual anonymizers must maintain compact memory footprints to avoid starving concurrent autonomous driving and navigation stacks.

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
