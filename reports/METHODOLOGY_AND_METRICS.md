# Methodology & Evaluation Metrics: Hourly Stockout Forecasting

**Research Paper / Thesis Documentation**  
**Author:** BTP_2 Research Team  
**Subject:** Multi-Variate Hourly Trajectory Prediction in Fresh Retail & Quick Commerce

---

## 1. The Fundamental Problem: Point-Wise Evaluation & The "Double-Penalty" Flaw

### 1.1 The Failure of Point-Wise Metrics
Standard machine learning benchmarks evaluate binary time series classification using point-wise precision, recall, and F1 score:
$$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
where each hour $h \in \{0, \dots, 23\}$ is evaluated independently.

In real-world retail operations, **this evaluation paradigm breaks down**. Consider an actual stockout occurring at **14:00**:
- **Scenario A (Point-Wise Strict):** The model predicts stockout onset at **13:00** (1 hour earlier).
  - At $h = 13$: True = 0, Pred = 1 $\implies$ **False Positive (+1 FP)**.
  - At $h = 14$: True = 1, Pred = 0 $\implies$ **False Negative (+1 FN)**.
  - Calculated Point-Wise F1 = **0.00**.
- **Scenario B (Operational Reality):** A store replenishment truck requires 45–60 minutes to pack and transit. An alert arriving 1 hour early allows warehouse staff to dispatch replenishment *before* the shelf runs empty. The store avoids lost sales entirely.

Standard metrics penalize an early warning **twice** (as both a precision failure and a recall failure), creating a misalignment between ML loss optimization and supply chain value.

---

## 2. Realistic Supply Chain Operational Metrics

To resolve the Double-Penalty flaw, we established a four-pillar operational metric framework:

### 2.1 Window-Tolerant Precision, Recall, and F1 ($\pm \Delta$ Slack)
Let $y_{i, h} \in \{0, 1\}$ and $\hat{y}_{i, h} \in \{0, 1\}$ denote ground truth and predicted stockout state at hour $h$.
We define temporal dilation within slack $\Delta \in \{1, 2\}$ hours:
$$\tilde{y}_{i, h} = \max_{k \in [\max(0, h-\Delta), \min(H-1, h+\Delta)]} y_{i, k}$$
$$\hat{\tilde{y}}_{i, h} = \max_{k \in [\max(0, h-\Delta), \min(H-1, h+\Delta)]} \hat{y}_{i, k}$$

Then:
$$\text{Precision}_{W\Delta} = \frac{\sum_{i, h} \hat{y}_{i, h} \cdot \tilde{y}_{i, h}}{\max\left(1, \sum_{i, h} \hat{y}_{i, h}\right)}$$
$$\text{Recall}_{W\Delta} = \frac{\sum_{i, h} y_{i, h} \cdot \hat{\tilde{y}}_{i, h}}{\max\left(1, \sum_{i, h} y_{i, h}\right)}$$
$$F_{1, W\Delta} = \frac{2 \cdot \text{Precision}_{W\Delta} \cdot \text{Recall}_{W\Delta}}{\text{Precision}_{W\Delta} + \text{Recall}_{W\Delta}}$$

**Empirical Result:** Across our 32,774 holdout sequences, applying $\Delta = 1$h slack increases the iTransformer F1 score from **0.5431 to 0.5920 (~60%)**, demonstrating that a substantial portion of supposed "errors" are near-perfect early or late warnings.

### 2.2 First Stockout Onset Timing MAE
Store operations prioritize knowing *when* the stockout cycle begins:
$$h^*_{\text{true}} = \min \{h \mid y_h = 1\}, \quad h^*_{\text{pred}} = \min \{h \mid \hat{y}_h = 1\}$$
For all sequences where both truth and prediction experience a stockout:
$$\text{Onset Timing MAE} = \frac{1}{N_{\text{both}}} \sum_{i=1}^{N_{\text{both}}} |h^*_{i, \text{true}} - h^*_{i, \text{pred}}|$$

- **`iTransformer_Stock`:** **1.36 hours (~82 minutes)** average error.
- **Alert Timing $\le \pm 1$h:** **84.7%** of all generated onset alerts arrive within 1 hour of the true stockout event.

### 2.3 Trajectory Interval IoU (Intersection-over-Union)
Measures the shape fidelity of the daily stockout curve:
$$\text{IoU}_i = \begin{cases} 1.0, & \text{if } \sum_h y_{i, h} = 0 \text{ and } \sum_h \hat{y}_{i, h} = 0 \\ \frac{\sum_h (y_{i, h} \land \hat{y}_{i, h})}{\sum_h (y_{i, h} \lor \hat{y}_{i, h})}, & \text{otherwise} \end{cases}$$
$$\text{Mean Interval IoU} = \frac{1}{N} \sum_{i=1}^N \text{IoU}_i$$

### 2.4 Daily Stockout Volume MAE
Measures calibration of daily total stockout duration:
$$\text{Volume MAE} = \frac{1}{N} \sum_{i=1}^N \left| \sum_{h=0}^{23} \hat{y}_{i, h} - \sum_{h=0}^{23} y_{i, h} \right|$$

---

## 3. Ground Truth Definition & Empirical Dynamics

### 3.1 Class Labels
From the raw inventory data `hours_stock_status`:
- **Class 0 = In-Stock** (~71.4% overall, ~75.9% during business operating hours 06:00–22:00)
- **Class 1 = Stockout / Out-of-Stock** (~28.6% overall, ~24.1% during business operating hours)

### 3.2 Diurnal Distribution
Stockout frequency follows a clear diurnal curve:
- **07:00 AM (Morning restock):** Stockout rate is lowest at **4.9%** (95.1% in stock).
- **12:00 PM – 14:00 PM (Lunch rush):** Stockout rate steadily rises from 15% to 22%.
- **18:00 PM – 21:00 PM (Evening dinner peak):** Depletion accelerates rapidly.
- **22:00 PM (End of business day):** Stockout rate peaks at **50.4%** (more than half of all shelves depleted before overnight restocking).

### 3.3 Temporal Distribution Shift
In chronological testing:
- **Training Period (March – May):** Stockout rate = **32.2%**.
- **Holdout Test Period (Late May – June):** Stockout rate = **20.6%**.
This 11.6% drop in baseline stockout frequency reflects seasonal supply improvements and store efficiency gains, making standard unweighted cross-entropy susceptible to under-predicting stockouts.

---

## 4. Sequence Length Ablation Analysis ($T=14$ vs $T=21$ vs $T=28$)

To determine optimal historical lookback, we ablated lookback length $T$ using our baseline linear and DLinear models:

| Lookback Length ($T$) | Train Sequences | Holdout Test Sequences | Holdout F1 Score | Holdout Precision | Holdout Recall |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **$T = 14$ days (Champion)** | **72,571** | **32,774** | **0.5471** | **0.4932** | **0.6141** |
| $T = 21$ days | 54,923 (-24.3%) | 26,145 | 0.5152 | 0.4710 | 0.5684 |
| $T = 28$ days | 39,812 (-45.1%) | 20,419 | 0.5239 | 0.4851 | 0.5702 |

### Why Shorter Windows ($T=14$) Outperform Longer Windows ($T=28$):
1. **The Sample Attrition Penalty:** Fresh retail datasets exhibit frequent SKU introduction and store churn. Requiring 28 consecutive active days drops **45.1% of all training sequences**, starving deep models of diverse stockout training examples.
2. **Fresh Retail Recency Bias:** Perishable items (leafy vegetables, fruits, dairy) have a shelf life of 1–3 days. Demand patterns and supplier replenishment terms change bi-weekly. Historical stockout events from 4 weeks prior have negligible predictive relevance to tomorrow's intra-day depletion.

---

## 5. Loss Function Formulation: Dual-Objective Focal Loss

To address class imbalance without causing threshold distortion, we employ `DualObjectiveFocalLoss`:

$$\mathcal{L} = \mathcal{L}_{\text{slot-focal}} + \lambda_{\text{vol}} \cdot \mathcal{L}_{\text{volume-smooth-L1}}$$

### 5.1 Slot-Level Focal Loss
$$\mathcal{L}_{\text{slot-focal}} = - \frac{1}{B \cdot 24} \sum_{b=1}^B \sum_{h=0}^{23} \alpha_t (1 - p_{t, b, h})^\gamma \log(p_{t, b, h})$$
where:
- $\alpha = 0.70$ (providing 2.33x positive gradient weight to balance the 32/68 train imbalance).
- $\gamma = 2.0$ (suppressing gradients from easy, confident in-stock hours).

### 5.2 Daily Volume Regularization
$$\mathcal{L}_{\text{vol}} = \text{SmoothL1}\left(\sum_{h=0}^{23} p_{b, h}, \sum_{h=0}^{23} y_{b, h}\right), \quad \lambda_{\text{vol}} = 0.15$$
This acts as an inductive bias enforcing that the model's total predicted daily stockout duration matches the ground truth aggregate stockout duration.

---

## 6. Architecture Mechanics: Why iTransformer Wins

Traditional time-series transformers feed the sequence $[t_1, t_2, \dots, t_T]$ as tokens, applying self-attention across time. In high-frequency retail data, adjacent time steps are highly redundant, while cross-channel interactions (price discount $\leftrightarrow$ sales surge $\leftrightarrow$ rapid shelf depletion) are non-linear.

### The Inverted Transformation:
1. **Variate Tokenization:** The input $\mathbf{X} \in \mathbb{R}^{B \times T \times C}$ is transposed to $\mathbf{X}^T \in \mathbb{R}^{B \times C \times T}$. A linear projection maps the temporal history of each variable into an embedding token:
   $$\mathbf{e}_c = \mathbf{W}_{\text{proj}} \mathbf{x}_{:, c} \in \mathbb{R}^{d_{\text{model}}}$$
2. **Channel-Wise Attention:** The Transformer Encoder computes self-attention across variate tokens $\mathbf{E} \in \mathbb{R}^{B \times (C+1) \times d_{\text{model}}}$, discovering multi-variate dependencies invariant to time shifts.
3. **Diurnal Query Decoding:** 24 learned query vectors $\mathbf{Q}_{\text{hour}} \in \mathbb{R}^{24 \times d_{\text{model}}}$ attend over the multi-variate representation via Multi-Head Cross-Attention to generate the final 24-hour prediction curve.
