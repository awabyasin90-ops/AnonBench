# Mathematical Formulations and Statistical Verification Framework
## AnonBench: Empirical Anonymization Failure Analysis in Autonomous Driving

This document presents the rigorous mathematical foundation, statistical sampling theory, evaluation metrics, and non-parametric hypothesis testing protocols employed across the **AnonBench** benchmark suite.

---

## 1. Mathematical Notation & Problem Formulation

Let an autonomous driving video sequence be represented as an ordered set of video frames:

$$\mathcal{F} = \{f_1, f_2, \dots, f_N\}, \quad f_t \in \mathbb{R}^{H \times W \times 3}$$

where $H$ and $W$ denote the frame height and width, and $N$ is the total frame population ($N = 100,000$).

### 1.1 Target Space & Ground Truth
Within any frame $f_t$, private identifiable biometric targets belong to the target space:

$$\mathcal{T} = \mathcal{T}_{\text{face}} \cup \mathcal{T}_{\text{plate}}$$

A target instance $T_i \in \mathcal{T}$ is parametrized by its spatial bounding box $b_i$ and semantic class $c_i$:

$$T_i = (b_i, c_i), \quad b_i = [x_{\min}, y_{\min}, x_{\max}, y_{\max}] \in \mathbb{R}^4, \quad c_i \in \{\text{face}, \text{plate}\}$$

The set of certified **Human-in-the-Loop (HIM)** reference targets in frame $f_t$ is denoted:

$$\mathcal{G}_t = \{G_1, G_2, \dots, G_{K_t}\}, \quad G_k = (b_k^*, c_k^*)$$

### 1.2 Model Perception Space
An arbitrary automated anonymization pipeline $\mathcal{M}$ (where $\mathcal{M} \in \{\text{FAST v5}, \text{Meta EgoBlur}, \text{DashcamCleaner}, \text{FusionPortable}, \text{Generative Inpaint}\}$) produces a set of candidate predictions:

$$\mathcal{D}_t^{\mathcal{M}} = \{D_1, D_2, \dots, D_{M_t}\}, \quad D_j = (\hat{b}_j, \hat{c}_j, s_j)$$

where $\hat{b}_j$ is the predicted bounding box, $\hat{c}_j$ is the predicted class, and $s_j \in [0, 1]$ is the detection confidence score.

---

## 2. Spatial Overlap & Bipartite Matching

### 2.1 Intersection-over-Union (IoU)
The spatial agreement between a predicted bounding box $\hat{b}$ and a ground truth bounding box $b^*$ is defined as:

$$\text{IoU}(\hat{b}, b^*) = \frac{\text{Area}(\hat{b} \cap b^*)}{\text{Area}(\hat{b} \cup b^*)} = \frac{|\hat{b} \cap b^*|}{|\hat{b}| + |b^*| - |\hat{b} \cap b^*|}$$

### 2.2 Intersection-over-Minimum-Area (IoMin)
To account for extreme scale disparities (e.g., small distant faces partially enclosed within broader contextual blurs), we define Intersection-over-Minimum-Area:

$$\text{IoMin}(\hat{b}, b^*) = \frac{|\hat{b} \cap b^*|}{\min(|\hat{b}|, |b^*|)}$$

### 2.3 Bipartite Matching Criterion
A prediction $D_j$ matches ground truth $G_k$ if and only if:

$$\hat{c}_j = c_k^* \quad \land \quad \left( \text{IoU}(\hat{b}_j, b_k^*) \ge \tau_{\text{iou}} \quad \lor \quad \text{IoMin}(\hat{b}_j, b_k^*) \ge \tau_{\text{iomin}} \right)$$

where the benchmark operates under the strict evaluation standard $\tau_{\text{iou}} = 0.50$ (with $\tau_{\text{iomin}} = 0.50$).

---

## 3. Multi-Model Consensus Fusion Operator $\Phi(\cdot)$

To prevent algorithmic bias toward any single architecture during ground-truth pre-annotation, candidate proposals are synthesized via a multi-model consensus fusion operator:

$$\mathcal{U}_t = \Phi\left(\mathcal{D}_t^{\mathcal{M}_1}, \mathcal{D}_t^{\mathcal{M}_2}, \dots, \mathcal{D}_t^{\mathcal{M}_5}\right) = \bigcup_{m=1}^5 \mathcal{D}_t^{\mathcal{M}_m}$$

### 3.1 Non-Duplication Aggregation
Pairs of overlapping proposals $(D_u, D_v) \in \mathcal{U}_t \times \mathcal{U}_t$ sharing identical class labels are merged if:

$$\max\left(\text{IoU}(\hat{b}_u, \hat{b}_v), \text{IoMin}(\hat{b}_u, \hat{b}_v)\right) \ge \tau_{\text{fusion}}, \quad \tau_{\text{fusion}} = 0.40$$

The fused box is computed via the spatial envelope:

$$\hat{b}_{\text{fused}} = \left[\min(x_1^u, x_1^v), \min(y_1^u, y_1^v), \max(x_2^u, x_2^v), \max(y_2^u, y_2^v)\right]$$

This produces the consensus candidate set ($\mathcal{U}_t^*$) presented to human oracle auditors.

---

## 4. Recognizability Resolution Thresholding (PP4AV Protocol)

Automotive sensors frequently capture sub-pixel pedestrian silhouettes and distant vehicles that lack identifiable biometrics. In compliance with the PP4AV standard, targets falling below recognizability thresholds are designated as `Ignore`:

$$\text{Recognizable}(b, c) = \begin{cases} 
\text{True}, & \text{if } c = \text{face} \land (x_2 - x_1) \ge 10\text{ px} \\
\text{True}, & \text{if } c = \text{plate} \land (y_2 - y_1) \ge 8\text{ px} \\
\text{False}, & \text{otherwise} \implies \text{Tag as `Ignore`}
\end{cases}$$

### 4.1 Multi-Scale Distance Discretization (PP4AV Bins)
Each confirmed target is partitioned into distance-proxy resolution bins:

$$\text{Bin}(b, c) = \begin{cases}
\text{Bin 1 (Far, } >20\text{m}): & \Delta(b, c) < 16\text{ px} \\
\text{Bin 2 (Mid, } 10\text{--}20\text{m}): & 16\text{ px} \le \Delta(b, c) < 32\text{ px} \\
\text{Bin 3 (Near, } 5\text{--}10\text{m}): & 32\text{ px} \le \Delta(b, c) < 64\text{ px} \\
\text{Bin 4 (Close, } <5\text{m}): & \Delta(b, c) \ge 64\text{ px}
\end{cases}$$

where $\Delta(b, c) = (x_2 - x_1)$ for faces, and $\Delta(b, c) = (y_2 - y_1)$ for license plates.

---

## 5. Statistical Sampling Theory & Finite Population Correction

Given the human annotation cost across $N = 100,000$ frames, stratified random sampling establishes the minimum audit subset $n = 7,200$ required for statistical validity.

### 5.1 Cochran's Sample Size Formula
For a finite population $N$, the required sample size $n$ under confidence level $(1 - \alpha)$ and margin of error $e$ is:

$$n = \frac{n_0}{1 + \frac{n_0 - 1}{N}}, \quad \text{where } n_0 = \frac{Z_{1 - \alpha/2}^2 \cdot p(1 - p)}{e^2}$$

### 5.2 Analytical Margin of Error Derivation
Substituting the benchmark parameters:
- Population size: $N = 100,000$
- Sample size: $n = 7,200$
- Confidence level: $99\%$ $\implies Z_{0.995} = 2.576$
- Maximum variance prior: $p = 0.50$

Solving for margin of error $e$:

$$e = Z_{1 - \alpha/2} \cdot \sqrt{\frac{p(1 - p)}{n} \cdot \left(\frac{N - n}{N - 1}\right)}$$

$$e = 2.576 \cdot \sqrt{\frac{0.25}{7200} \cdot \left(\frac{100000 - 7200}{99999}\right)} = 2.576 \cdot \sqrt{3.472 \times 10^{-5} \cdot 0.928} \approx \mathbf{\pm 1.46\%}$$

$$\therefore \text{Audited results generalize to the full 100K driving corpus at 99\% confidence within a } \pm 1.5\% \text{ error bound.}$$

---

## 6. Empirical Privacy & Perception Distortion Metrics

Evaluating model predictions $\mathcal{D}_t^{\mathcal{M}}$ against reference ground truth $\mathcal{G}_t$ yields four discrete outcomes:
- **True Positive ($\text{TP}$):** A valid private target was successfully identified and blurred.
- **False Negative ($\text{FN}$):** A valid private target was unblurred $\implies$ **Privacy Leak**.
- **False Positive ($\text{FP}$):** Non-sensitive pixels (trees, road signs, buildings) were blurred $\implies$ **Over-blur Distortion**.
- **True Negative ($\text{TN}$):** Background non-private regions correctly preserved without modification.

### 6.1 Privacy Leak Rate (PLR / Miss Rate)
The paramount metric for GDPR/KVKK compliance, quantifying the ratio of unanonymized private targets:

$$\text{PLR} = \frac{\text{FN}}{\text{TP} + \text{FN}} = 1 - \text{Recall} \in [0, 1] \quad (\text{Optimal} = 0.0)$$

### 6.2 Over-blur Distortion Rate (ODR / False Discovery Rate)
Quantifies downstream perceptual destruction inflicted on autonomous driving perception:

$$\text{ODR} = \frac{\text{FP}}{\text{TP} + \text{FP}} = 1 - \text{Precision} \in [0, 1] \quad (\text{Optimal} = 0.0)$$

### 6.3 Anonymization Utility Index (AUI / Balanced $F_1$-Score)
The harmonic mean reconciling compliance preservation against visual utility:

$$\text{AUI} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$$

### 6.4 Wilson Score 95% Confidence Intervals
For any observed proportion $\hat{p} \in \{\text{Recall}, \text{Precision}, \text{PLR}\}$ evaluated across $M$ total instances, the asymmetric 95% confidence interval is:

$$\text{CI}_{95\%}(\hat{p}) = \frac{\hat{p} + \frac{z^2}{2M} \pm z \sqrt{\frac{\hat{p}(1 - \hat{p})}{M} + \frac{z^2}{4M^2}}}{1 + \frac{z^2}{M}}, \quad z = 1.960$$

---

## 7. Non-Parametric Hypothesis Testing Battery

Because per-frame privacy failure distributions exhibit significant right-skewness (many frames have 0 leaks, while congested intersections have sharp spikes), parametric ANOVA assumptions are violated. We employ a non-parametric verification battery.

### 7.1 Shapiro-Wilk Normality Test
Tests the null hypothesis $H_0$ that per-frame failure arrays $\mathbf{x} = [x_1, \dots, x_K]$ follow a Gaussian distribution:

$$W = \frac{\left( \sum_{i=1}^K a_i x_{(i)} \right)^2}{\sum_{i=1}^K (x_i - \bar{x})^2}$$

$$\text{Empirical result: } p < 0.05 \implies \text{Reject } H_0 \implies \text{Non-parametric methods strictly required.}$$

### 7.2 Friedman Non-Parametric Repeated-Measures ANOVA
Tests the null hypothesis $H_0$ that all 5 anonymization systems perform with identical rank distributions across frames:

$$\chi_F^2 = \frac{12 K}{J(J + 1)} \sum_{j=1}^J \bar{R}_j^2 - 3 K (J + 1)$$

where $K$ is the number of evaluated frames ($K = 10$), $J$ is the number of systems ($J = 5$), and $\bar{R}_j$ is the average rank of system $j$.

$$\text{Empirical result: } \chi_F^2 = 20.89, \quad p = 0.00034 \implies \text{Statistically significant difference across pipelines.}$$

### 7.3 Wilcoxon Signed-Rank Pairwise Test
Performs paired rank comparisons between proposed FAST v5 and competing baselines:

$$W = \min(W^+, W^-), \quad W^+ = \sum_{d_i > 0} \text{Rank}(|d_i|), \quad d_i = x_{i, \text{baseline}} - x_{i, \text{FAST}}$$

$$\text{Empirical result: } p = 0.0156 \ (<0.05) \implies \text{FAST v5 failure rate reduction is statistically significant.}$$

---

## 8. Edge Hardware Telemetry Formulations

Edge deployment feasibility is evaluated on embedded automotive compute (NVIDIA Jetson AGX Orin / RTX 4090):

### 8.1 Throughput (Frames Per Second)
$$\text{FPS} = \frac{N_{\text{frames}}}{\sum_{i=1}^{N} \Delta t_i^{\text{e2e}}}$$

### 8.2 End-to-End Latency
$$\Delta t_i^{\text{e2e}} = t_i^{\text{pre}} + t_i^{\text{infer}} + t_i^{\text{blur}} + t_i^{\text{d2h}}$$

We report both the median ($P_{50}$) and worst-case execution time ($P_{95}$):

$$P_{95} = \inf \{t \in \mathbb{R} : F_{\Delta t}(t) \ge 0.95\}$$

### 8.3 Memory Allocation
$$\text{VRAM}_{\text{peak}} = \max_{t \in [0, T]} \left( \text{AllocatedMemory}_{\text{CUDA}}(t) \right) \quad [\text{MB}]$$
