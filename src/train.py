"""
Unified Training & Evaluation Pipeline
======================================
Usage Example:
  python src/train.py --model itransformer --epochs 10 --batch_size 256 --loss dual
  python src/train.py --model dlinear_gelu --epochs 10 --batch_size 256 --loss focal
"""

import os
import sys
import math
import time
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Package relative or direct imports
try:
    from .data_pipeline import get_dataloaders, CONTINUOUS_FEATURES
    from .models import (
        iTransformer_Stock,
        EntityNonLinearDLinear_GELU,
        PureLinear_HourlyHead_DLinear,
        ContinuousHour_DecompTransformer,
        Recurrent_Stock,
    )
    from .losses import FocalLoss, DualObjectiveFocalLoss
    from .metrics import compute_comprehensive_metrics
except ImportError:
    from data_pipeline import get_dataloaders, CONTINUOUS_FEATURES
    from models import (
        iTransformer_Stock,
        EntityNonLinearDLinear_GELU,
        PureLinear_HourlyHead_DLinear,
        ContinuousHour_DecompTransformer,
        Recurrent_Stock,
    )
    from losses import FocalLoss, DualObjectiveFocalLoss
    from metrics import compute_comprehensive_metrics


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate hourly stockout forecasting models.")
    parser.add_argument("--model", type=str, default="itransformer",
                        choices=["itransformer", "dlinear_gelu", "pure_linear", "decomp_transformer", "rnn", "lstm", "gru"],
                        help="Model architecture to train.")
    parser.add_argument("--data_path", type=str, default="data/top15_train.parquet",
                        help="Path to training parquet dataset.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--loss", type=str, default="dual", choices=["dual", "focal"],
                        help="Loss function: dual (Focal + SmoothL1 Volume) or focal.")
    parser.add_argument("--alpha", type=float, default=0.70, help="Focal loss alpha (positive weight).")
    parser.add_argument("--gamma", type=float, default=2.0, help="Focal loss gamma focusing parameter.")
    parser.add_argument("--lambda_vol", type=float, default=0.15, help="Dual loss volume penalty weight.")
    parser.add_argument("--seq_len", type=int, default=14, help="Lookback window length in days.")
    parser.add_argument("--save_path", type=str, default="", help="Optional path to save model weights.")
    parser.add_argument("--out_csv", type=str, default="benchmark_run.csv", help="CSV path to append results.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("=" * 65)
    print(f"Hourly Stockout Training Pipeline: Model = {args.model.upper()}")
    print(f"Device: {device} | Epochs: {args.epochs} | Loss: {args.loss} (alpha={args.alpha})")
    print("=" * 65)

    # 1. Load Data
    train_loader, val_loader, test_loader, meta = get_dataloaders(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        verbose=True
    )

    # 2. Build Model
    num_products = meta["num_products"]
    num_cities = meta["num_cities"]
    num_features = meta["num_features"]

    if args.model == "itransformer":
        model = iTransformer_Stock(
            seq_len=args.seq_len,
            num_features=num_features,
            num_products=num_products,
            num_cities=num_cities,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1
        )
        display_name = "iTransformer_Stock"
    elif args.model == "dlinear_gelu":
        model = EntityNonLinearDLinear_GELU(
            seq_len=args.seq_len,
            input_dim=num_features,
            num_products=num_products,
            num_cities=num_cities,
            hidden_dim=64,
            kernel_size=5,
            d_emb=32
        )
        display_name = "Entity-DLinear (GELU)"
    elif args.model == "pure_linear":
        model = PureLinear_HourlyHead_DLinear(
            seq_len=args.seq_len,
            input_dim=num_features,
            num_products=num_products,
            num_cities=num_cities,
            kernel_size=5,
            d_emb=32
        )
        display_name = "PureLinear_HourlyHead"
    elif args.model == "decomp_transformer":
        model = ContinuousHour_DecompTransformer(
            seq_len=args.seq_len,
            input_dim=num_features,
            num_products=num_products,
            num_cities=num_cities,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.1
        )
        display_name = "DecompTransformer (Continuous)"
    elif args.model in ["rnn", "lstm", "gru"]:
        model = Recurrent_Stock(
            cell_type=args.model,
            seq_len=args.seq_len,
            input_dim=num_features,
            num_products=num_products,
            num_cities=num_cities,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            d_emb=32
        )
        display_name = f"{args.model.upper()} (2-layer)"

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel initialized: {display_name} with {params:,} parameters.")

    # 3. Setup Loss and Optimizer
    if args.loss == "dual":
        criterion = DualObjectiveFocalLoss(gamma=args.gamma, alpha=args.alpha, lambda_vol=args.lambda_vol)
    else:
        criterion = FocalLoss(alpha=args.alpha, gamma=args.gamma)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    warmup_steps = 2 * len(train_loader)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: step / max(1, warmup_steps) if step < warmup_steps
        else 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / max(1, total_steps - warmup_steps)))
    )

    # 4. Training Loop
    t_start = time.time()
    best_val_f1 = 0.0
    best_tau = 0.50

    for epoch in range(1, args.epochs + 1):
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

        # Tune threshold tau on validation set
        epoch_best_f1, epoch_best_tau = 0.0, 0.50
        for tau in [0.40, 0.45, 0.50, 0.55, 0.60]:
            f1_candidate = f1_score(vt.ravel(), (vp.ravel() >= tau).astype(int), zero_division=0)
            if f1_candidate > epoch_best_f1:
                epoch_best_f1 = f1_candidate
                epoch_best_tau = tau

        if epoch_best_f1 > best_val_f1:
            best_val_f1 = epoch_best_f1
            best_tau = epoch_best_tau

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Train Loss: {train_loss:.4f} | Val F1 (tau={epoch_best_tau:.2f}): {epoch_best_f1:.4f}")

    total_time = time.time() - t_start
    print(f"\nTraining completed in {total_time:.1f}s. Best Val F1: {best_val_f1:.4f} at tau={best_tau:.2f}")

    # 5. Evaluate on Holdout Test Set
    print("\nRunning comprehensive evaluation on holdout test set...")
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

    results = compute_comprehensive_metrics(tt, bin_preds, display_name=display_name, params=params)
    results["Optimal Tau"] = best_tau
    results["Train Time (s)"] = round(total_time, 1)

    print("\n" + "=" * 65)
    print("=== FINAL HOLDOUT TEST EVALUATION RESULTS ===")
    print("=" * 65)
    for k, v in results.items():
        print(f"  {k:25s}: {v}")
    print("=" * 65 + "\n")

    # Optional save
    if args.save_path:
        torch.save(model.state_dict(), args.save_path)
        print(f"Saved model weights to {args.save_path}")

    # Append to CSV
    if args.out_csv:
        res_df = pd.DataFrame([results])
        if os.path.exists(args.out_csv):
            res_df.to_csv(args.out_csv, mode="a", header=False, index=False)
        else:
            res_df.to_csv(args.out_csv, index=False)
        print(f"Saved benchmark results to {args.out_csv}")


if __name__ == "__main__":
    main()
