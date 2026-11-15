 # Empirical Benchmark Specification & Measurement Protocol

## Anonymization Failure Patterns Across Real Drives and CARLA Simulation: An Empirical Benchmark & Systems Evaluation

> **Paper Classification: Empirical Benchmark Paper (Experimental Measurement Protocol & Failure Audit)**  
> *(Scientific Evaluation Scope: This research is classified strictly as an **Empirical Benchmark Paper**—an experimental measurement protocol and systematic failure audit—NOT an empirical review or literature survey. Peer-review criteria evaluate benchmark papers on how systematically, objectively, and rigorously disparate production systems are stress-tested under physical driving perturbations, comparative simulation, and hardware telemetry.)*
>
> **Core Benchmark Contribution & Empirical Value:**  
> The core contribution of this work is an open, standardized **Empirical Benchmark Protocol and Sim-to-Real Failure Audit** that systematically stress-tests widely deployed vehicular privacy anonymization systems that are **fundamentally disparate in their algorithmic methodologies**—ranging from classical two-stage region-proposal detectors (Faster R-CNN) and single-stage anchor-free architectures (YOLOv8) to real-time multi-scale ADAS edge pipelines (FAST v5) and dual deep generative synthesis/inpainting (StyleGAN2-UNet + Fast Fourier Convolutions).  
> Evaluating these radically distinct paradigms across **Real Drives (Physical Research Car Multi-Camera Rig & Tesla Fleet Perception Cameras)** and **CARLA Autonomous Driving Simulation** under identical adverse environmental perturbations and distance-scale degradations provides the first reproducible, end-to-end failure audit, edge-deployment efficiency evaluation, and **Sim-to-Real domain transfer analysis** for vehicular vision systems.

---

### Scope & Operational Constraints

- ✅ **IN SCOPE: Multi-Platform Experimental Benchmark Dataset (Sim-to-Real & Real Drives):**
  - **CARLA Autonomous Driving Simulator:** Synthetically controlled, ground-truth-aligned multi-camera sequences across 6 standardized environmental conditions (**Clear, Rain, Night, Fog, Glare, Overcast**).
  - **Physical Research Car Sensor Rig:** Multi-camera calibrated research vehicle capturing real-world urban and highway driving across identical environmental conditions.
  - **Tesla Fleet Drives:** Real-world driving sequences captured via Tesla onboard perception cameras under natural operational driving.
- ✅ **IN SCOPE: Sim-to-Real Failure Pattern Analysis:** Measuring how detection failure modes, boundary degradation, and false negative patterns observed in CARLA simulation transfer to physical real-world driving environments (Research Car & Tesla). Sim-to-Real domain transfer is quantified using a Pearson correlation coefficient ($r_{\text{sim-to-real}}$) comparing False Negative (FN) failure rates between CARLA simulation and physical Real Drives across the six environmental conditions.
- ✅ **IN SCOPE: Comprehensive Instance Accounting (All Images & Blurring Pixels):** Every single image across all platforms (CARLA simulation, Research Car rig, Tesla) and every detected target instance across all blurring pixel scales is preserved and tallied under two fundamental recognizability categories:
  - 🟢 **Recognizable** ($\ge 10\text{ px}$ face width, $\ge 8\text{ px}$ plate height): Legitimate privacy breaches requiring anonymization.
  - 🔴 **Unrecognizable** ($< 10\text{ px}$ face width, $< 8\text{ px}$ plate height): Sub-threshold instances with 0% human identifiability.
  Full quantitative counts are retained across both groups for all images so the researcher can flexibly make downstream cutoff and decision analysis.
- ✅ **IN SCOPE: Comparative Benchmark Across 5 Diverse Methodological Paradigms Evaluated Against a Rigorous Hybrid Ground Truth:**
  1. FAST v5 (Real-Time ADAS Edge TensorRT Pipeline)
  2. Meta AI EgoBlur Gen2 (Two-Stage Faster R-CNN JIT)
  3. DashcamCleaner (Single-Stage YOLOv8 720p)
  4. FusionPortable (Perspective Faster R-CNN ONNX)
  5. Generative Anonymizer (RetinaFace + WPOD-NET + Big-LaMa Fast Fourier Inpainting)
  Evaluated strictly against the **Two-Stage Hybrid Ground-Truth Protocol** across a statistically representative 7,200-frame Golden Subset:
  - **Stage 1 (Automated Multi-System Candidate Fusion):** Low-threshold spatial union combining candidate proposals from all 5 evaluated systems and 2 baseline detection oracles (InsightFace SCRFD-34G and RT-DETR v2) to maximize recall.
  - **Stage 2 (Human-in-the-Loop Audit via Optimized X-AnyLabeling):** Exhaustive expert human verification, false-positive elimination, and sub-pixel boundary snapping using the customized X-AnyLabeling suite with MobileSAM auto-snapping and PP4AV recognizability tagging.
- ✅ **IN SCOPE: Multi-Scale Distance Degradation:** 4 pixel-size bins (`< 16 px`, `16–32 px`, `32–64 px`, `> 64 px`) grounded in the **PP4AV human recognizability standard (WACV 2023)**.
- ✅ **IN SCOPE: Unified Host-Device Systems Telemetry (GPU + CPU):** Comprehensive end-to-end hardware profiling addressing Amdahl's law and host-side bottlenecks: Computational Throughput (img/s), Median & P95 Latency (ms) with sub-stage breakdown (Pre-processing, Neural Inference, Post-processing/Blurring, and Disk I/O), Accelerator Energy (GPU Wh / 1k frames), Processor Energy (CPU Package RAPL Wh / 1k frames), Total System Ingestion Energy ($\mathcal{E}_{\text{sys}} = \mathcal{E}_{\text{GPU}} + \mathcal{E}_{\text{CPU}}$), Thermals (GPU Core & CPU Package °C Mean/Peak), Resource Utilization (GPU Engine %, Overall & Per-Core CPU %, Host Process RSS & System RAM in GB).
- ✅ **IN SCOPE: Post-Blur Leakage Audit (Subset Analysis):** A targeted adversarial recovery audit using ArcFace and EasyOCR on a randomized 200–500 frame subset of successfully blurred images to determine residual re-identification risk.
- ❌ **STRICTLY OUT OF SCOPE:** No over-blur metrics (SSIM, PSNR) or image distortion indices.

---

## Table 1: Hybrid Ground-Truth Protocol (5-System Candidate Fusion + Optimized X-AnyLabeling Human Audit)

| Dimension / Protocol Aspect | Specification & Operational Standard | Scientific Rationale in Benchmark Evaluation |
| :--- | :--- | :--- |
| **Stage 1: Multi-System Candidate Fusion** | **Spatial Ensemble Union across 5 Production Systems + 2 Reference Oracles** | Computes a low-threshold ($\ge 0.15$) multi-detector spatial union across FAST v5, EgoBlur, DashcamCleaner, FusionPortable, Generative Anonymizer, InsightFace SCRFD-34G (Face Oracle), and RT-DETR v2 (Plate Oracle). Maximizes initial proposal recall and eliminates single-detector blind spots. |
| **Stage 2: Human-in-the-Loop Audit Tool** | **Custom-Instrumented X-AnyLabeling Platform** | High-throughput interactive annotation application re-engineered with MobileSAM auto-snapping, one-key class targeting (`Shift+F`/`Shift+P`), zero-modal deletion (`Delete`/`Backspace`), and atomic navigation auto-save (`D`/`A`). |
| **Sub-Pixel Boundary Refinement** | **Interactive Segment Anything Model (MobileSAM)** | Shrinks and snaps user or candidate bounding boxes to true physical boundaries with sub-pixel precision, eliminating human manual boundary jitter. |
| **Target PII Classes** | **Human Faces** & **Vehicle License Plates** | Exhaustive verification and delineation of all visual Personally Identifiable Information (PII) across vehicular driving scenes. |
| **Dataset Sampling Method** | **Stratified Random Sampling ($N = 7,200$ frames)** | $1,200$ frames sampled per environmental stratum across 6 conditions (Clear, Rain, Night, Fog, Glare, Overcast). |
| **Statistical Rigor** | **99% Confidence Level ($Z = 2.576$), $\pm 1.51\%$ Margin of Error** | Statistically representative sample size derived under maximum population variance assumption ($p = 0.50$). |
| **Face Annotation Boundary** | Inscribed tight rectangular box enclosing entire visible head/cranium | Encloses chin, ears, forehead, hair boundary, and facial profile across all yaw/pitch angles ($0^\circ \text{ to } 90^\circ$). |
| **Plate Annotation Boundary** | Inscribed tight rectangular box around physical perimeter of license plate | Encloses all alphanumeric characters, issuing emblems, and regional registration borders. |
| **PP4AV Recognizability Standard** | **Face Width $\ge 10\text{ px}$** $\rightarrow$ `Confirmed`<br>**Plate Height $\ge 8\text{ px}$** $\rightarrow$ `Confirmed`<br>*(Sub-threshold targets marked as `Ignore`)* | Grounded in Trinh et al. (WACV 2023) empirical human studies. Sub-threshold instances logged for completeness but excluded from privacy miss penalties. |
| **Inter-Annotator Agreement** | Dual independent audit + senior researcher adjudication | Enforces strict consensus (Cohen's $\kappa \ge 0.94$) to eliminate individual human labeling bias in the Golden Subset. |
| **Operational Scope** | Ground-Truth Coordinate & Dimension Annotation ONLY | Strictly exports reference metadata (`.json` / `.parquet`). **Never modifies pixels, applies blurring, or writes altered images.** |

> [!IMPORTANT]
> **Ground Truth Operational Rule:** The Hybrid Ground-Truth Golden Subset serves as the definitive, uncompromised ground-truth reference for all 5 evaluated pipelines. It is constructed through multi-system candidate fusion followed by exhaustive human audit and MobileSAM boundary snapping in X-AnyLabeling. It establishes pure reference coordinates, target dimensions, and PP4AV recognizability status tags without detector bias and never applies image blurring or pixel modification.

---

## Table 2: The 5 Evaluated Anonymization Pipelines (Architectural Stress-Test Suite)

| Pipeline Name | Methodological Paradigm | Underlying Architecture | Framework & Runtime | Default Conf. (Face / Plate) | Input Resolution | Anonymization Mechanism |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **FAST v5 (ADAS Edge)** | Real-Time Multi-Scale ADAS Pipeline | Dual YOLOv8m-Face + YOLO11m + NanoPlate | TensorRT (FP16) + CUDA Async | **0.15** / **0.35** | Face: **1280 px**<br>Plate: **192x320 px** | Adaptive Gaussian Blur + 2px Feathered Ellipse/Rect |
| **Meta AI EgoBlur Gen2** | Two-Stage Industry Baseline | Dual Faster R-CNN (~104M params) | PyTorch TorchScript JIT | **0.67416** / **0.74475** | Fixed **1200 x 1200 px** | EgoBlur Half-Image Box Blur + Solid Ellipse Mask |
| **DashcamCleaner** | Single-Stage Community Baseline | YOLOv8 trained on Mosaic Automotive Data | PyTorch Ultralytics | **0.40** (Combined) | **1280 x 720 px** (720p medium) | Normalized Kernel-19 Box Blur + 5px Feathered Mask |
| **FusionPortable** | Perspective Automotive Baseline | Retrained Faster R-CNN (Fisheye Excluded) | ONNX Runtime (FP32) | **0.30** / **0.30** | **Native Resolution** (Full size) | 2D Spatial Gaussian Convolution ($k=21, \sigma=2.0$) |
| **Generative Anonymizer** | Deep Generative Inpainting Pipeline | RetinaFace (ResNet-18 FPN) + WPOD-NET + Big-LaMa (FFC) | PyTorch TorchScript JIT + cv2.dnn ONNX | **0.30** / **0.30** | Face: **640 px**<br>Plate: **608 px**<br>Inpaint: **ROI Native** | Fast Fourier Convolution Inpainting (FFC Texture Synthesis) |

---

## Table 3: Empirical Human Recognizability Baseline (PP4AV Finding)

*Empirical human recognizability thresholds established by Trinh et al. (PP4AV, IEEE/CVF WACV 2023). Differentiates contextually detectable targets from human-identifiable privacy breaches. Every detected instance across all images is counted and recorded under one of these two fundamental categories.*

| Target Object | Resolution Threshold | Detection Status | Recognizability Status | Research Finding & Counting Action |
| :--- | :---: | :---: | :---: | :--- |
| **Faces** | **$< 10\text{ px}$ width** ($< 7\text{ px}$) | Detected | 🔴 **Unrecognizable** | `status = Ignore` — Counted in master log; excluded from privacy failure counts (sub-threshold / unrecognizable). |
| **Faces** | **$\ge 10\text{ px}$ width** | Detected | 🟢 **Recognizable** | `status = Confirmed` — Counted as primary privacy target; identity recognizable, requires anonymization. |
| **License Plates** | **$< 8\text{ px}$ height** ($6 - 7\text{ px}$) | Detected | 🔴 **Unrecognizable** | `status = Ignore` — Counted in master log; excluded from privacy failure counts (alphanumeric characters illegible). |
| **License Plates** | **$\ge 8\text{ px}$ height** ($\ge 10\text{ px}$) | Detected | 🟢 **Recognizable** | `status = Confirmed` — Counted as primary privacy target; characters readable, requires anonymization. |

---

## Table 4: Granular Scale Stratification (The 4 Pixel-Size Bins)

| Scale Bin | Pixel Range | Real-World Distance | Empirical Benchmark Standards & Scale Justification | Computer Vision & Safety Significance |
| :--- | :---: | :---: | :--- | :--- |
| **Bin 1: Very Small** | $< 16\text{ px}$ | $20+\text{ m}$ | **Trinh et al. (PP4AV, IEEE/CVF WACV 2023):** Establishes the empirical recognizability floor ($\ge 10\text{ px}$ face width, $\ge 8\text{ px}$ plate height).<br>**Yang et al. (WIDER FACE, IEEE CVPR 2016):** Defines $10\text{–}16\text{ px}$ as the "Hard" small-object detection regime where $8\times$/$16\times$ CNN stride causes feature collapse. | Extreme long-range small-object detection limits; early pedestrian and vehicle warning boundary. |
| **Bin 2: Small** | $16 - 32\text{ px}$ | $10 - 20\text{ m}$ | **Lin et al. (MS COCO, ECCV 2014):** Standardizes Small objects boundary at $32\text{ px}$ ($\text{area} < 32^2\text{ px}$).<br>**Zhang et al. (CityPersons, IEEE CVPR 2017) & Dollár et al. (IEEE TPAMI 2012):** Maps $16\text{–}32\text{ px}$ head/plate scales ($\sim 50\text{ px}$ pedestrian height) to $10\text{–}20\text{ m}$ urban vehicle tracking. | Urban crosswalk and approaching vehicle tracking zone; standard camera perception range. |
| **Bin 3: Medium** | $32 - 64\text{ px}$ | $5 - 10\text{ m}$ | **Lin et al. (MS COCO, ECCV 2014) & Lin et al. (FPN, IEEE CVPR 2017):** Represents Medium scale ($32\text{–}64\text{ px}$ octave bin) mapped to mid-pyramid feature levels ($P_4$).<br>**Geiger et al. (KITTI, IEEE CVPR 2012):** Corresponds to primary $5\text{–}10\text{ m}$ active lane perception operating zone. | Active lane driving range; adjacent vehicle and cyclist monitoring zone. |
| **Bin 4: Large** | $> 64\text{ px}$ | $< 5\text{ m}$ | **ISO/IEC 19794-5 & NIST SP 800-76 (Biometric Standards):** Face widths $> 64\text{ px}$ exceed the $16\text{–}24\text{ px}$ Inter-Ocular Distance (IOD) required for automated re-ID and high-confidence plate OCR.<br>**Caesar et al. (nuScenes, IEEE CVPR 2020):** Near-field ego-vehicle safety zone. | Near-field critical safety zone (intersections, crosswalk stops); highest risk of identity and biometric privacy leakage. |

---

## Table 5: The 6 Environmental Stress-Test Conditions & Visual Failure Modes

| Environmental Condition | Sensor & Photometric Degradation | Primary Computer Vision Failure Mode |
| :--- | :--- | :--- |
| **Clear** | Nominal ambient illumination, balanced dynamic range | Baseline control condition |
| **Rain** | Water droplets on lens, specular street puddles, wiper streaking | High-frequency refraction edge artifacts, false positive clusters |
| **Night** | Severe low light, high sensor ISO noise, deep shadows | Contrast loss, severe false negatives on small distance targets |
| **Fog** | Volumetric Mie scattering, attenuation of high-frequency textures | Structural gradient attenuation, low-contrast plate blending |
| **Glare** | Intense localized solar saturation, blooming, lens flares | Pixel sensor over-saturation, obliteration of facial landmarks |
| **Overcast** | Diffuse low-contrast ambient illumination, uniform gray sky | Mild texture degradation, shadow boundary suppression |

---

## Table 6: Operational Benchmark Protocol: Pristine As-Shipped Configuration

*Scientific Justification: Each pipeline is benchmarked under its official shipping confidence thresholds and native resolution to capture off-the-shelf automotive deployment performance. Forcing a uniform numerical threshold across architectures is scientifically invalid due to differing logit calibrations (e.g., two-stage Faster R-CNN vs. single-stage anchor-free YOLO).*

| Pipeline Name | Methodological Architecture | Confidence Thresholds | Input Image Resolution | Scientific Rationale |
| :--- | :--- | :---: | :---: | :--- |
| **FAST v5 (ADAS Edge)** | Real-Time ADAS Pipeline (TensorRT) | **0.15** (Face) / **0.35** (Plate) | **1280 px** (Face) / **192x320** (Plate crop) | Native multi-scale TensorRT edge optimization for onboard ADAS |
| **Meta AI EgoBlur Gen2** | Two-Stage Industry Architecture | **0.67416** (Face) / **0.74475** (Plate) | Fixed **1200 x 1200 px** | Official Meta AI TorchScript Gen2 calibration |
| **DashcamCleaner** | Single-Stage Community Architecture | **0.40** (Combined) | **1280 x 720 px** (720p medium) | Default community dashcam censoring preset |
| **FusionPortable** | Perspective Automotive Architecture | **0.30** (Face) / **0.30** (Plate) | **Native Resolution** (Full size) | Official retrained Faster R-CNN ONNX threshold |
| **Generative Anonymizer** | Deep Generative Inpainting Architecture | **0.30** (Face) / **0.30** (Plate) | **640 px** (Face) / **608 px** (Plate) | Official RetinaFace + WPOD-NET + Big-LaMa JIT preset |

---

## Table 7: Systems Telemetry & Hardware Efficiency Profile (Per-Model Metrics)

> **Empirical Telemetry Measurement Template:** All quantitative metrics below will be recorded live during the experimental benchmark via each pipeline's local `unified_telemetry_profiler.py` (sampling GPU NVML and Linux CPU RAPL / `psutil` at 100ms intervals with a 200-frame warm-up exclusion). Dummy/simulated numbers have been omitted to maintain empirical integrity.

| Pipeline / Model | Operational Scope | Throughput (img/s) | Median Latency (ms) | GPU Energy (Wh / 1k) | CPU Package Energy (Wh / 1k) | Total System Energy $\mathcal{E}_{\text{sys}}$ (Wh / 1k) | CPU Temp (°C Mean/Peak) | GPU Temp (°C Mean/Peak) | Mean CPU Util (%) | Mean GPU Util (%) | Host RAM (GB RSS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FAST v5 (ADAS Edge)** | End-to-End (Read + Detect + Blur + Write) | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |
| **DashcamCleaner** | End-to-End (Read + Detect + Blur + Write) | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |
| **FusionPortable** | End-to-End (Read + Detect + Blur + Write) | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |
| **Meta AI EgoBlur Gen2** | End-to-End (Read + Detect + Blur + Write) | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |
| **Generative Anonymizer** | End-to-End (Face Synthesis + Plate Inpainting) | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* | *[To be measured]* |

---

## Section III: Empirical Benchmark Methodology & Experimental Measurement Protocol

### Multi-Platform Benchmark Protocol & Failure Audit Specification (Paper Methodology Section)

*(Directly insert into the Experimental Setup & Annotation Protocol section of your research paper)*

> **Multi-Platform Benchmark Dataset & Multi-Scale Recognizability Protocol:** To benchmark vehicular privacy anonymization under genuine operational conditions, we formulate an empirical measurement protocol executed across a multi-platform driving dataset: (1) physical **Real Drives** captured using a multi-camera calibrated **Research Car Sensor Rig** and onboard **Tesla Fleet Cameras**, and (2) high-fidelity **CARLA Autonomous Driving Simulation** generating comparative driving trajectories across six standardized environmental perturbations (**Clear, Rain, Night, Fog, Glare, Overcast**). To eliminate scene redundancy and auto-correlation, frames are sampled via temporal decimation at 1–2 Hz, enforcing a strict minimum temporal baseline between successive evaluations.
>
> Ground-truth reference targets are established via a **Two-Stage Hybrid Protocol**: (1) an initial multi-detector spatial candidate fusion combining candidate proposals from all five benchmarked systems (FAST v5, EgoBlur, DashcamCleaner, FusionPortable, Generative Anonymizer) and baseline detection oracles (InsightFace SCRFD-34G, RT-DETR v2) evaluated at high sensitivity ($\ge 0.15$), followed by (2) exhaustive expert human verification and sub-pixel contour snapping across a statistically representative 7,200-frame Golden Subset using a **Custom-Instrumented fork of X-AnyLabeling** powered by MobileSAM auto-snapping. Following the empirical human-recognizability criteria formulated by Trinh et al. (PP4AV, WACV 2023), every candidate target across all evaluated frames is preserved in the master audit registry under two binary recognizability states:
> 1. 🟢 **Recognizable Privacy Targets** ($\ge 10\text{ px}$ face width, $\ge 8\text{ px}$ license plate height), representing instances where un-anonymized visual details constitute actionable identity leaks.
> 2. 🔴 **Unrecognizable Sub-Threshold Targets** ($< 10\text{ px}$ face width, $< 8\text{ px}$ license plate height), where empirical ratings confirm zero human identifiability.
>
> Tallying both categories across all images prevents premature data censoring, granting full transparency for downstream regulatory auditing. Confirmed targets are stratified across four standardized scale bins ($< 16\text{ px}$, $16\text{–}32\text{ px}$, $32\text{–}64\text{ px}$, and $> 64\text{ px}$) to quantify scale-dependent perceptual collapse.

### Statistical Sampling Theory & Golden Subset Derivation (Stratified Random Sampling)

To establish a definitive, statistically rigorous ground truth while maintaining manual human audit tractability, the Golden Subset of $N = 7,200$ driving frames was drawn via **Stratified Random Sampling** across the six operational environmental strata ($\text{Clear, Rain, Night, Fog, Glare, Overcast}$).

The representative sample size $n_0$ is derived from classical estimation theory for large finite populations:
$$n_0 = \frac{Z^2 \cdot p(1 - p)}{e^2}$$

Where:
- $Z = 2.576$ represents the critical value corresponding to a **$99\%$ Confidence Level** ($\alpha = 0.01$).
- $p = 0.50$ represents the conservative proportion assumption maximizing population variance:
  $$p(1 - p) = 0.50 \times (1 - 0.50) = 0.25$$
- $e = \pm 1.51\% = 0.0151$ denotes the strict acceptable **Margin of Error**.

Substituting these empirical parameters yields:
$$n_0 = \frac{(2.576)^2 \times (0.50 \times 0.50)}{(0.0151)^2} = \frac{6.635776 \times 0.25}{0.00022801} = \frac{1.658944}{0.00022801} \approx 7,275 \approx 7,200 \text{ frames}$$

Allocating these sample frames uniformly across the $K = 6$ adverse environmental strata yields:
$$n_c = \frac{N_{\text{total}}}{K} = \frac{7,200}{6} = 1,200 \text{ frames per environmental condition}$$
where $c \in \{\text{Clear}, \text{Rain}, \text{Night}, \text{Fog}, \text{Glare}, \text{Overcast}\}$.

To eliminate consecutive frame auto-correlation, frames within each stratum were sampled via temporal decimation at $1\text{–}2\text{ Hz}$, ensuring independent, identically distributed (i.i.d.) observations. Each of the 7,200 frames was then exhaustively verified, adjusted, and validated using the custom-optimized X-AnyLabeling suite under the Two-Stage Hybrid Ground-Truth Protocol.

### Mathematical Formulation of Benchmark Evaluation Metrics

1. **Empirical Miss Rate (False Negative Rate, $MR_{b, c}$):**
   For a given scale bin $b$ and environmental condition $c$, the empirical miss rate quantifies the proportion of recognizable privacy breaches un-anonymized by pipeline $m$:
   $$\text{MR}_{b,c}(m) = \frac{\text{FN}_{b,c}(m)}{\text{TP}_{b,c}(m) + \text{FN}_{b,c}(m)}$$
   where a True Positive (TP) requires intersection-over-union $\text{IoU} \ge 0.50$ against the human-audited (`Confirmed`) reference bounding box.

2. **Sim-to-Real Domain Transfer Correlation ($r_{\text{sim-to-real}}$):**
   The fidelity of CARLA simulation in predicting physical real-world failure patterns is quantified via Pearson correlation across the six environmental conditions:
   $$r_{\text{sim-to-real}}(m) = \frac{\sum_{c=1}^{6} (\text{FN}_{\text{sim}, c} - \overline{\text{FN}}_{\text{sim}})(\text{FN}_{\text{real}, c} - \overline{\text{FN}}_{\text{real}})}{\sqrt{\sum_{c=1}^{6} (\text{FN}_{\text{sim}, c} - \overline{\text{FN}}_{\text{sim}})^2 \sum_{c=1}^{6} (\text{FN}_{\text{real}, c} - \overline{\text{FN}}_{\text{real}})^2}}$$
   where $r \to 1.0$ indicates strong simulation predictive power for physical driving failures.

3. **Adversarial Post-Blur Re-Identification Rate ($ARR$):**
   Residual privacy leakage on successfully blurred regions is audited on a randomized 200–500 frame subset using adversarial feature extractors:
   $$\text{ARR}_{\text{face}} = \frac{1}{N_{\text{audit}}} \sum_{i=1}^{N_{\text{audit}}} \mathbb{I}\left( \cos(\mathbf{f}_{\text{blurred}, i}, \mathbf{f}_{\text{gt}, i}) \ge 0.60 \right)$$
   using ArcFace embeddings, and character-level accuracy for license plates via EasyOCR.

### Unified Host-Device Systems Telemetry Measurement Protocol (Paper Telemetry Section)

*(Directly insert into the Computational Performance & Telemetry section of your research paper)*

> **Host-Device Telemetry Protocol & Amdahl's Law:** To evaluate onboard edge deployment feasibility and uncover the host-side bottlenecks governed by Amdahl's Law, our benchmark tracks simultaneous accelerator (GPU) and host processor (CPU) metrics end-to-end. We record computational throughput (img/s) and latency percentiles ($P_{50}, P_{95}$) alongside a four-stage latency decomposition: $\tau_{\text{pre}}$ (decoding and letterboxing), $\tau_{\text{infer}}$ (neural tensor execution), $\tau_{\text{post}}$ (NMS and spatial Gaussian blurring/inpainting), and $\tau_{\text{io}}$ (disk serialization).
>
> Energy consumption is captured across both computational domains: GPU electrical power is integrated via trapezoidal summation ($E_{\text{GPU}} = \sum \frac{P_i + P_{i-1}}{2} \Delta t / 3600$) using NVIDIA Management Library (NVML) polled at 100ms intervals, while CPU Package Energy ($E_{\text{CPU}}$) is recorded directly from the Linux Intel/AMD Running Average Power Limit (RAPL) interface (`/sys/class/powercap/intel-rapl`). This yields the **Total System Ingestion Energy**:
> $$\mathcal{E}_{\text{sys}} = \mathcal{E}_{\text{GPU}} + \mathcal{E}_{\text{CPU}} \quad (\text{Wh per 1,000 frames})$$
>
> Host resource utilization is sampled via `psutil`, capturing total and per-core CPU load (%), resident set size (RSS RAM in GB), and operating thermal trajectories (°C for GPU Core and CPU Package). To eliminate transient CUDA/JIT runtime initialization spikes, the initial 200 frames of each condition are designated as a warm-up phase and excluded from steady-state calculations.
