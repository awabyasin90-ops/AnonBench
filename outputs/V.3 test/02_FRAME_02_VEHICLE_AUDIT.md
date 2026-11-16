# Frame #02 
**Frame ID:** `1788953691129285.jpg` | **Scene:** Urban Commercial Street | **GT Targets:** 6

---

## 1. Clean Raw Camera Input (Zero Detections)
[![01 Clean Raw Camera Input](./Frame_02_1788953691129285/01_clean_raw_image.jpg)](./Frame_02_1788953691129285/01_clean_raw_image.jpg)

---

## 2. Unassisted Model Detections (Before HIM Review)
[![02 Models Raw Detections Comparison](./Frame_02_1788953691129285/02_models_raw_detections_comparison.jpg)](./Frame_02_1788953691129285/02_models_raw_detections_comparison.jpg)

- **FAST v5:** 5 raw detections (4 faces, 1 plate)
- **Meta AI EgoBlur:** 1 raw detection (1 face, 0 plates)
- **DashcamCleaner:** 3 raw detections (2 faces, 1 plate)
- **FusionPortable:** 10 raw detections (6 faces, 4 plates)
- **Generative Inpaint:** 1 raw detection (1 face, 0 plates)

---

## 3. Consensus Fusion vs. Human-in-the-Loop (HIM) Review

| Stage A: Consensus Fusion (Pre-Review) | Stage B: Certified HIM Ground Truth (Post-Review) |
| :---: | :---: |
| [![03 Fused Candidate Proposals](./Frame_02_1788953691129285/03_fused_candidate_proposals.jpg)](./Frame_02_1788953691129285/03_fused_candidate_proposals.jpg) | [![04 HIM Ground Truth](./Frame_02_1788953691129285/04_HIM_human_reviewed_ground_truth.jpg)](./Frame_02_1788953691129285/04_HIM_human_reviewed_ground_truth.jpg) |
| **All ensemble candidates combined** | **6 Certified Ground Truth Targets** |

---

## 4. Final Benchmark Evaluation (Against HIM Reference)
[![05 Full Evaluation Comparison vs HIM](./Frame_02_1788953691129285/05_full_evaluation_comparison_vs_HIM.jpg)](./Frame_02_1788953691129285/05_full_evaluation_comparison_vs_HIM.jpg)

- 🟩 **Green Box:** True Positive (Detected / Blurred)
- 🟥 **Red Box:** False Negative (Missed / Privacy Leak)
- 🟦 **Blue Box:** False Positive (Over-blur / Hallucination)

### Performance Breakdown:
| Pipeline Architecture | Detected (TP) 🟩 | Missed (FN) 🟥 | False+ (FP) 🟦 | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **FAST v5** | **4** | **2** | **1** | **66.7%** |
| **Meta AI EgoBlur** | 1 | 5 | 0 | 16.7% |
| **DashcamCleaner** | 3 | 3 | 0 | 50.0% |
| **FusionPortable** | 5 | 1 | 5 | 83.3% |
| **Generative Inpaint** | 1 | 5 | 0 | 16.7% |
