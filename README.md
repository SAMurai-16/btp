# Hourly Stockout Research & Operational Benchmark Suite

A modular, reproducible deep learning framework for **Day-Ahead Hourly Stockout Trajectory Prediction** in fresh retail and quick-commerce.

Evaluated on **2,341 store-product time series** (~140,400 daily trajectories, 3.37M hourly slots) with a **32,774 sequence holdout test set**.

---

## 📁 Repository Structure

```
hourly_stockout_research/
├── README.md                           <- This documentation and quickstart guide
├── benchmarks/                         <- Benchmark CSVs and automated runner script
│   ├── master_operational_benchmark_results.csv  <- Master operational evaluation across all 8 architectures
│   ├── recurrent_baselines_benchmark.csv         <- Detailed metrics for SimpleRNN, GRU, and LSTM
│   ├── sequence_length_comparison.csv            <- Sequence length ablation (T=14 vs T=21 vs T=28)
│   ├── balanced_itransformer_results.csv         <- Alpha class-balancing ablation (alpha=0.50 to 0.75)
│   └── run_recurrent_baselines.py                <- Automated baseline evaluation script
├── reports/                            <- In-depth technical and operational reports
│   ├── EXECUTIVE_SUMMARY.md            <- Master leaderboards, Dingdong paper comparison, ROI
│   └── METHODOLOGY_AND_METRICS.md      <- Double-penalty math, Window F1, loss functions
└── src/                                <- Clean, production-ready Python modules
    ├── __init__.py                     <- Package initialization
    ├── data_pipeline.py                <- Vectorized parquet loader, window extraction, scaling
    ├── models.py                       <- iTransformer, Entity-DLinear, PureLinear, DecompTransformer, Recurrent_Stock
    ├── losses.py                       <- FocalLoss & DualObjectiveFocalLoss
    ├── metrics.py                      <- Window-Tolerant F1 (+-1h), Onset MAE, IoU, Volume MAE
    └── train.py                        <- Unified training & evaluation CLI
```

---

## 🚀 Quickstart

### 1. Environment Setup
Activate your environment with PyTorch, Pandas, Scikit-learn, and PyArrow:
```bash
cd /home/samyak/code/Labs/BTP_2
source venv/bin/activate
```

### 2. Train Models with the Unified CLI

#### Train Champion `iTransformer_Stock` (126K parameters)
```bash
python hourly_stockout_research/src/train.py   --model itransformer   --epochs 10   --batch_size 256   --loss dual   --alpha 0.70   --lambda_vol 0.15   --lr 1e-3
```

#### Train Ultra-Fast `Entity-DLinear (GELU)` (34K parameters, ~20s train time)
```bash
python hourly_stockout_research/src/train.py   --model dlinear_gelu   --epochs 10   --batch_size 256   --loss dual   --alpha 0.70   --lr 1e-3
```

#### Train Recurrent Baselines (SimpleRNN, GRU, LSTM)
```bash
python hourly_stockout_research/src/train.py --model rnn --epochs 10
python hourly_stockout_research/src/train.py --model gru --epochs 10
python hourly_stockout_research/src/train.py --model lstm --epochs 10
```

---

## 📊 Summary Leaderboard (Holdout Test: 32,774 Sequences)

| Model Architecture | Parameters | Window F1 ($\pm 1$h) | Onset Timing MAE | Alerts $\le \pm 1$h | Interval IoU | Exact Match | Train Time |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`iTransformer_Stock` (Champion)** | **126K** | **0.5920 (~60%)** | **1.36 hours** | **84.7%** | **50.5%** | **30.3%** | **85s** |
| **`ContinuousHour DecompTransformer`** | **166K** | **0.5745** | **1.64 hours** | **82.7%** | **49.0%** | **28.4%** | **142s** |
| **`SimpleRNN (Elman 2-layer)`** | **25K** | **0.5700** | **1.27 hours** | **86.1%** | **51.2%** | **32.5%** | **44s** |
| **`GRU (2-layer)`** | **56K** | **0.5616** | **1.19 hours** | **87.3%** | **53.6%** | **35.5%** | **86s** |
| **`Entity-DLinear (GELU)`** | **34K** | **0.5598** | **1.70 hours** | **81.9%** | **48.2%** | **29.1%** | **22s** |
| **`PureLinear-HourlyHead`** | **11.5K** | **0.5518** | **3.13 hours** | **77.7%** | **50.1%** | **33.8%** | **18s** |
| **`LSTM (2-layer)`** | **72K** | **0.5393** | **1.23 hours** | **87.2%** | **53.9%** | **36.7%** | **68s** |
| `Persistence Baseline ("Tomorrow = Yesterday")` | 0 | 0.4691 | 8.71 hours | 45.7% | 44.1% | 31.6% | 0s |

---

## 📖 In-Depth Reports
- **Executive Summary:** [`reports/EXECUTIVE_SUMMARY.md`](reports/EXECUTIVE_SUMMARY.md)
  *Full leaderboard, Dingdong FreshRetailNet-50K (arXiv:2505.16319) alignment, operational implications.*
- **Methodology & Metrics:** [`reports/METHODOLOGY_AND_METRICS.md`](reports/METHODOLOGY_AND_METRICS.md)
  *Mathematical formulation of the Double-Penalty flaw, Window-Tolerant F1 ($\pm 1$h slack), Onset Timing MAE, and $T=14$ vs $T=28$ sequence length ablation.*
