"""
Recurrent Baselines Benchmark: Simple RNN, LSTM, and GRU
========================================================
Trains and evaluates:
1. Simple RNN (Elman RNN, 2 layers, hidden=64)
2. LSTM (2 layers, hidden=64)
3. GRU (2 layers, hidden=64)

Evaluates on the exact same 32,774 holdout test set using both:
- Rigid point-wise metrics
- Realistic supply chain operational metrics (+-1h slack, onset MAE, IoU, volume MAE)
"""

import os
import sys
import math
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Path setup for src imports
sys.path.append("/home/samyak/code/Labs/BTP_2/hourly_stockout_research/src")
from data_pipeline import get_dataloaders
from losses import DualObjectiveFocalLoss
from metrics import compute_comprehensive_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Recurrent_Stock(nn.Module):
    """
    Standard Recurrent baseline for hourly stockout forecasting.
    Supports cell_type in ['rnn', 'lstm', 'gru'].
    """
    def __init__(
        self,
        cell_type="lstm",
        seq_len=14,
        input_dim=14,
        num_products=68,
        num_cities=11,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        d_emb=32
    ):
        super().__init__()
        self.cell_type = cell_type.lower()
        self.prod_emb = nn.Embedding(num_products, d_emb // 2)
        self.city_emb = nn.Embedding(num_cities, d_emb // 2)

        rnn_in = input_dim + d_emb
        if self.cell_type == "rnn":
            self.rnn = nn.RNN(
                rnn_in, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == "gru":
            self.rnn = nn.GRU(
                rnn_in, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
        elif self.cell_type == "lstm":
            self.rnn = nn.LSTM(
                rnn_in, hidden_dim, num_layers=num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0.0
            )
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + 24 + d_emb, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 24)
        )

    def forward(self, x_seq, prev_stock, cat_meta):
        B, T, _ = x_seq.shape
        static_ctx = torch.cat([self.prod_emb(cat_meta[:, 0]), self.city_emb(cat_meta[:, 1])], dim=-1)
        static_expanded = static_ctx.unsqueeze(1).expand(-1, T, -1)
        rnn_input = torch.cat([x_seq, static_expanded], dim=-1)

        out, _ = self.rnn(rnn_input)
        last_hidden = out[:, -1, :]

        combined = torch.cat([last_hidden, prev_stock, static_ctx], dim=-1)
        return self.head(combined)


def train_and_eval(model, model_name, train_loader, val_loader, test_loader, device, epochs=10):
    set_seed(42)
    model = model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    criterion = DualObjectiveFocalLoss(gamma=2.0, alpha=0.70, lambda_vol=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    warmup_steps = 2 * len(train_loader)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(1, warmup_steps) if step < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    )

    print(f"\n" + "=" * 65)
    print(f"Training {model_name} ({params:,} params)...")
    print("=" * 65)

    t0 = time.time()
    best_val_f1 = 0.0
    best_tau = 0.50

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, bprev, bcat, by in train_loader:
            bx, bprev, bcat, by = bx.to(device), bprev.to(device), bcat.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx, bprev, bcat)
            loss = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, bprev, bcat, by in val_loader:
                bx, bprev, bcat = bx.to(device), bprev.to(device), bcat.to(device)
                logits = model(bx, bprev, bcat)
                val_preds.append(torch.sigmoid(logits).cpu().numpy())
                val_targets.append(by.numpy())

        vp = np.vstack(val_preds)
        vt = np.vstack(val_targets)

        epoch_best_f1, epoch_best_tau = 0.0, 0.50
        for tau in [0.40, 0.45, 0.50, 0.55, 0.60]:
            f1_candidate = f1_score(vt.ravel(), (vp.ravel() >= tau).astype(int), zero_division=0)
            if f1_candidate > epoch_best_f1:
                epoch_best_f1 = f1_candidate
                epoch_best_tau = tau

        if epoch_best_f1 > best_val_f1:
            best_val_f1 = epoch_best_f1
            best_tau = epoch_best_tau

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val F1 (tau={epoch_best_tau:.2f}): {epoch_best_f1:.4f}")

    total_time = time.time() - t0
    print(f"Completed in {total_time:.1f}s. Best Val F1: {best_val_f1:.4f} at tau={best_tau:.2f}")

    # Holdout Test Evaluation
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for bx, bprev, bcat, by in test_loader:
            bx, bprev, bcat = bx.to(device), bprev.to(device), bcat.to(device)
            logits = model(bx, bprev, bcat)
            test_preds.append(torch.sigmoid(logits).cpu().numpy())
            test_targets.append(by.numpy())

    tp = np.vstack(test_preds)
    tt = np.vstack(test_targets)
    bin_preds = (tp >= best_tau).astype(int)

    results = compute_comprehensive_metrics(tt, bin_preds, display_name=model_name, params=params)
    results["Optimal Tau"] = best_tau
    results["Train Time (s)"] = round(total_time, 1)

    print(f"Results for {model_name}:")
    print(f"  Rigid F1: {results['Rigid F1']} | Window F1 (+-1h): {results['Window F1 (+-1h)']}")
    print(f"  Onset Timing MAE: {results['Onset Timing MAE (h)']}h | Alert <= +-1h: {results['Timing <= +-1h (%)']}%")
    print(f"  Exact Match: {results['Exact Match 24h (%)']}% | IoU: {results['Interval IoU (%)']}%")

    return results


def main():
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("=== Recurrent Baselines Benchmark Suite ===")
    print(f"Device: {device}\n")

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_path="data/top15_train.parquet",
        seq_len=14,
        batch_size=256,
        verbose=True
    )

    models_to_test = [
        ("SimpleRNN (Elman 2-layer)", Recurrent_Stock("rnn", num_products=meta["num_products"], num_cities=meta["num_cities"])),
        ("GRU (2-layer)", Recurrent_Stock("gru", num_products=meta["num_products"], num_cities=meta["num_cities"])),
        ("LSTM (2-layer)", Recurrent_Stock("lstm", num_products=meta["num_products"], num_cities=meta["num_cities"])),
    ]

    all_results = []
    for name, model in models_to_test:
        res = train_and_eval(model, name, train_loader, val_loader, test_loader, device, epochs=10)
        all_results.append(res)

    res_df = pd.DataFrame(all_results)
    out_csv = "/home/samyak/code/Labs/BTP_2/hourly_stockout_research/benchmarks/recurrent_baselines_benchmark.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nAll recurrent baseline benchmarks saved to {out_csv}!")

    print("\n" + "=" * 80)
    print("=== RECURRENT BASELINES SUMMARY TABLE ===")
    print("=" * 80)
    summary_cols = ["Model", "Parameters", "Rigid F1", "Window F1 (+-1h)", "Onset Timing MAE (h)", "Timing <= +-1h (%)", "Interval IoU (%)", "Exact Match 24h (%)", "Train Time (s)"]
    print(res_df[summary_cols].to_string(index=False))
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
