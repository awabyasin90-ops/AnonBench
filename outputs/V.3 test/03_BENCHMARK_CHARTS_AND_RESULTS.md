# AnonBench: Complete Benchmark Evaluation Results & Visual Analytics
## Empirical Findings Across the 5 Privacy Anonymization Pipelines

This report consolidates the complete quantitative results, edge hardware telemetry, and publication charts across all 5 evaluated systems on audited real-world autonomous driving data (Pendik, Istanbul dataset).

---

## 1. Executive Performance Summary

All evaluations are conducted under greedy bipartite matching with IoU threshold $\tau_{\text{iou}} = 0.50$ against the **Human-in-the-Loop (HIM / V.2)** Ground Truth reference (56 confirmed targets across the drive).

| Pipeline System | Confirmed GT | TP 🟩 | FN (Leak) 🟥 | FP (Over-blur) 🟦 | Privacy Leak Rate (%) $\downarrow$ | Over-blur Rate (%) $\downarrow$ | Precision (%) $\uparrow$ | Recall (%) $\uparrow$ | $F_1$-Score $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proposed FAST v5** | 56 | **43** | **13** | 16 | **23.21%** | 27.12% | **72.88%** | **76.79%** | **0.7478** |
| **Meta AI EgoBlur** | 56 | 9 | 47 | **3** | 83.93% | **25.00%** | 75.00% | 16.07% | 0.2647 |
| **DashcamCleaner** | 56 | 28 | 28 | 15 | 50.00% | 34.88% | 65.12% | 50.00% | 0.5657 |
| **FusionPortable** | 56 | 45 | 11 | 91 | 19.64% | 66.91% | 33.09% | 80.36% | 0.4688 |
| **Generative Inpaint**| 56 | 13 | 43 | 9 | 76.79% | 40.91% | 59.09% | 23.21% | 0.3333 |

---

## 2. Publication Benchmark Charts

### Chart 1: Privacy Leak Rate vs. Over-Blur Rate
![Privacy Leak Rate vs Over-blur](./benchmark_charts/chart_oracle_privacy_leak_vs_overblur.png)

* **Horizontal Axis (Bottom):** The 5 tested models (`FAST v5`, `DashcamCleaner`, `EgoBlur`, `FusionPortable`, `Generative`).
* **Vertical Axis (Side):** **Error Rate (%)** — *Lower is better* (0% is ideal).
  * 🟥 **Red Bar (Privacy Leak Rate):** Percentage of real human faces and license plates the model **MISSED** (unprotected personal data).
  * 🟪 **Purple Bar (Over-Blur Rate):** Percentage of false alarms where the model blurred trees, tarmac, or empty street.
* **Key Finding:**
  * **Meta EgoBlur** causes a catastrophic **83.9% Privacy Leak** (it misses over 8 out of 10 people).
  * **FusionPortable** catches people but **hallucinates 66.9% of the time**, blurring out the road and driving context.
  * **FAST v5** occupies the ideal sweet spot: keeping leaks low (**23.2%**) with minimal unnecessary blur (**27.1%**).

---

### Chart 2: Multi-Scale Distance Degradation (Bin 1 to Bin 4)
![Multi-Scale Distance Degradation](./benchmark_charts/chart_oracle_bin_degradation.png)

* **Horizontal Axis (Bottom):** **Distance from the vehicle camera**, divided into 4 real-world distance zones:
  * **Bin 1 (Far, >20m / <16px):** Distant pedestrians and vehicles far down the road.
  * **Bin 2 (Mid, 10–20m / 16–32px):** Pedestrians crossing or vehicles 2–3 car lengths ahead.
  * **Bin 3 (Standard, 5–10m / 32–64px):** Normal nearby traffic and pedestrians.
  * **Bin 4 (Close, <5m / >64px):** Targets directly in front of the car.
* **Vertical Axis (Side):** **Privacy Leak Rate (%)** — *Lower is better* (0% means full protection).
* **Key Finding:**
  * When targets are **Close (<5m)**, all models perform reasonably well because faces are large and easy to spot.
  * **The Critical Failure Mode:** As people walk farther away (**10m to >20m**), standard industry models like **Meta EgoBlur** and **Generative** suffer a **complete collapse** — their error lines shoot up to **100% leaks**, making them completely blind to distant people.
  * **Why FAST v5 Wins:** FAST v5 stays flat and robust across all 4 distance zones, protecting pedestrians whether they are 3 meters or 30 meters away.

---

### Chart 3: Precision, Recall, and Balanced $F_1$-Score
![Precision Recall and F1 Score](./benchmark_charts/chart_oracle_precision_recall_f1.png)

* **Horizontal Axis (Bottom):** The 5 tested models.
* **Vertical Axis (Side):** **Performance Score (0% to 100%)** — *Higher is better*.
  * 🟩 **Green Bar (Recall):** How many real targets were successfully detected and protected?
  * 🟦 **Blue Bar (Precision):** How clean and accurate were the detections (avoiding false alarms)?
  * 🟧 **Yellow Bar ($F_1$-Score):** The overall balanced grade combining both Precision and Recall.
* **Key Finding:**
  * **Meta EgoBlur** fails on Recall (16.1%), while **FusionPortable** fails on Precision (33.1%).
  * **FAST v5 achieves the highest overall balanced score ($F_1 = 74.8\%$)**, proving it is the only reliable all-around system.

---

### Chart 4: Edge Processing Throughput (Frames Per Second)
![Throughput FPS](./benchmark_charts/chart_throughput_fps.png)

* **Horizontal Axis (Bottom):** The 5 tested models.
* **Vertical Axis (Side):** **Processing Speed (Frames Per Second, FPS)** on a logarithmic scale — *Higher is better*.
* **Key Finding:**
  * Two-stage Generative inpainting runs at an unviable **0.03 FPS** (taking over 30 seconds for a single frame). On a real road at 60 km/h, the car would drive 500 meters blind before completing one anonymization step.
  * **FAST v5** processes frames in real time (scaling to **>30 FPS** on automotive edge hardware), making it safe and viable for deployment.

---

### Chart 5: End-to-End Inference Latency Breakdown (Median vs. P95 Worst-Case)
![Latency Summary](./benchmark_charts/chart_latency_summary.png)

* **Horizontal Axis (Bottom):** The 5 tested models.
* **Vertical Axis (Side):** **Latency in Milliseconds (ms)** on a logarithmic scale — *Lower is better* (1,000 ms = 1s).
  * 🟩 **Green Bar (Median):** Typical delay for normal driving frames.
  * 🟥 **Red Bar (P95):** Worst-case delay during complex, crowded street scenes.
* **Key Finding:**
  * Generative inpainting stalls the pipeline with a massive **30,000 ms (30-second)** delay per frame.
  * **FAST v5** maintains tight, sub-second latency with minimal jitter between normal and crowded scenes, preventing pipeline buffer overflows.

---

### Chart 6: Peak Memory & VRAM Footprint
![Memory and VRAM Allocation](./benchmark_charts/chart_memory_vram.png)

* **Horizontal Axis (Bottom):** The 5 tested models.
* **Vertical Axis (Side):** **Memory Consumption in Megabytes (MB)** — *Lower is better*.
* **Dotted Green Line (500 MB):** The strict maximum RAM budget for embedded automotive hardware (e.g., NVIDIA Jetson / Drive).
* **Key Finding:**
  * Generative AI consumes **over 2,400 MB (2.4 GB)**, instantly causing Out-Of-Memory (OOM) crashes on embedded car chips.
  * **FAST v5 operates comfortably below the 500 MB threshold**, ensuring reliable execution on edge hardware.

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
