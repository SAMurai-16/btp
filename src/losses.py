"""
Loss Functions for Hourly Stockout Forecasting
==============================================
Contains:
1. FocalLoss: Standard focal loss addressing class imbalance (alpha, gamma).
2. DualObjectiveFocalLoss: Multi-task loss combining slot-level focal loss with
   daily aggregate stockout volume regularization (Smooth L1).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.70, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * focal_weight * bce).mean()


class DualObjectiveFocalLoss(nn.Module):
    """
    Dual-Objective Loss:
    L = FocalLoss_slot(logits, targets) + lambda_vol * SmoothL1(pred_daily_volume, true_daily_volume)

    - Slot-level Focal Loss forces sharp detection of exact stockout hours.
    - Daily volume regularization guarantees the model predicts the correct total
      number of stockout hours across the day, preventing systematic under/over-prediction.
    """
    def __init__(self, gamma=2.0, alpha=0.70, lambda_vol=0.15):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda_vol = lambda_vol

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = bce * ((1 - p_t) ** self.gamma)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss_slot = (alpha_t * focal).mean()

        pred_sum = p.sum(dim=-1)
        true_sum = targets.sum(dim=-1)
        loss_vol = F.smooth_l1_loss(pred_sum, true_sum)

        return loss_slot + self.lambda_vol * loss_vol
