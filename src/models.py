"""
Deep Learning Architectures for Hourly Stockout Forecasting
===========================================================
Architectures included:
1. iTransformer_Stock:
   - Inverted Transformer Champion (126K parameters)
   - Channels/features as individual tokens, self-attention across variates,
     and learned hour queries with cross-attention to predict the 24h curve.
2. EntityNonLinearDLinear_GELU:
   - DLinear Champion (34K parameters)
   - Series decomposition into trend and seasonality, GELU bottleneck MLPs,
     and static entity embeddings (product & city).
3. PureLinear_HourlyHead_DLinear:
   - Ultra-lightweight linear baseline (11.5K parameters)
   - 24 independent linear heads per hour with zero non-linearities.
4. ContinuousHour_DecompTransformer:
   - Temporal-attention transformer with series decomposition and learned hour tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MovingAvg(nn.Module):
    """Moving average pool to extract trend from time series."""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)
        self.pad = (kernel_size - 1) // 2

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, self.pad, 1)
        end = x[:, -1:, :].repeat(1, self.pad, 1)
        x_pad = torch.cat([front, x, end], dim=1).permute(0, 2, 1)
        return self.avg(x_pad).permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    """Series decomposition block separating seasonal and trend components."""
    def __init__(self, kernel_size=5):
        super().__init__()
        self.ma = MovingAvg(kernel_size)

    def forward(self, x):
        trend = self.ma(x)
        return x - trend, trend


class iTransformer_Stock(nn.Module):
    """
    Inverted Transformer for Hourly Stockout Forecasting.

    Instead of tokenizing time steps (which suffers from temporal redundancy),
    iTransformer treats each feature/channel across the entire historical lookback
    as a single token. Self-attention captures cross-variate correlations (e.g.,
    how price discount, temperature, and historical stockouts interact), while
    cross-attention maps these multi-variate representations to 24 hourly prediction queries.
    """
    def __init__(
        self,
        seq_len=14,
        num_features=14,
        num_products=68,
        num_cities=11,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.feature_proj = nn.Linear(seq_len, d_model)
        self.prev_stock_proj = nn.Linear(24, d_model)
        self.prod_emb = nn.Embedding(num_products, d_model // 2)
        self.city_emb = nn.Embedding(num_cities, d_model // 2)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.hour_queries = nn.Embedding(24, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_seq, prev_stock, cat_meta):
        B = x_seq.size(0)

        # Transpose to (Batch, Num_Features, Seq_Len) so each feature is an individual token
        x_trans = x_seq.permute(0, 2, 1)
        var_tokens = self.feature_proj(x_trans)  # (B, Num_Features, d_model)
        prior_token = self.prev_stock_proj(prev_stock).unsqueeze(1)  # (B, 1, d_model)

        tokens = torch.cat([var_tokens, prior_token], dim=1)  # (B, Num_Features + 1, d_model)

        # Inject static product and city embeddings
        static_ctx = torch.cat([self.prod_emb(cat_meta[:, 0]), self.city_emb(cat_meta[:, 1])], dim=-1).unsqueeze(1)
        tokens = tokens + static_ctx

        # Variate-level self-attention
        multi_var_ctx = self.encoder(tokens)

        # 24-hour queries attend over multi-variate context
        queries = self.hour_queries(torch.arange(24, device=x_seq.device)).unsqueeze(0).expand(B, -1, -1)
        slot_ctx, _ = self.cross_attn(queries, multi_var_ctx, multi_var_ctx)
        slot_ctx = self.cross_norm(slot_ctx + queries)

        # Project to 24 logits
        logits = self.head(slot_ctx).squeeze(-1)  # (B, 24)
        return logits


class EntityNonLinearDLinear_GELU(nn.Module):
    """
    DLinear with Moving Average Decomposition and GELU Bottleneck.

    Decomposes the input into seasonal and trend components, processes each
    branch with non-linear feedforward MLPs, and conditions on static entity
    embeddings as well as previous-day hourly stock state.
    """
    def __init__(
        self,
        seq_len=14,
        input_dim=14,
        num_products=68,
        num_cities=11,
        hidden_dim=64,
        kernel_size=5,
        d_emb=32
    ):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.prod_emb = nn.Embedding(num_products, d_emb // 2)
        self.city_emb = nn.Embedding(num_cities, d_emb // 2)

        in_flat = seq_len * input_dim + d_emb + 24
        self.seasonal_mlp = nn.Sequential(
            nn.Linear(in_flat, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 24)
        )
        self.trend_mlp = nn.Sequential(
            nn.Linear(seq_len * input_dim + d_emb, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 24)
        )

    def forward(self, x, prev_stock, cat_meta):
        B = x.size(0)
        static_ctx = torch.cat([self.prod_emb(cat_meta[:, 0]), self.city_emb(cat_meta[:, 1])], dim=-1)
        s_init, t_init = self.decomp(x)

        s_flat = torch.cat([s_init.reshape(B, -1), static_ctx, prev_stock], dim=-1)
        t_flat = torch.cat([t_init.reshape(B, -1), static_ctx], dim=-1)

        return self.seasonal_mlp(s_flat) + self.trend_mlp(t_flat)


class PureLinear_HourlyHead_DLinear(nn.Module):
    """
    Pure Linear DLinear with dedicated hourly heads.

    Has ZERO non-linear activation functions (no ReLU, GELU, or Sigmoid in hidden layers).
    Splits the prediction across 24 independent linear regressors, achieving the highest
    exact daily match rate (33.91%) while using only 11.5K parameters.
    """
    def __init__(
        self,
        seq_len=14,
        input_dim=14,
        num_products=68,
        num_cities=11,
        kernel_size=5,
        d_emb=32
    ):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.prod_emb = nn.Embedding(num_products, d_emb // 2)
        self.city_emb = nn.Embedding(num_cities, d_emb // 2)

        s_in = seq_len * input_dim + d_emb + 1
        t_in = seq_len * input_dim + d_emb

        self.seasonal_linear_heads = nn.ModuleList([nn.Linear(s_in, 1) for _ in range(24)])
        self.trend_linear_heads = nn.ModuleList([nn.Linear(t_in, 1) for _ in range(24)])

    def forward(self, x, prev_stock, cat_meta):
        B = x.size(0)
        static_ctx = torch.cat([self.prod_emb(cat_meta[:, 0]), self.city_emb(cat_meta[:, 1])], dim=-1)
        s_init, t_init = self.decomp(x)

        s_flat_base = torch.cat([s_init.reshape(B, -1), static_ctx], dim=-1)
        t_flat = torch.cat([t_init.reshape(B, -1), static_ctx], dim=-1)

        logits = []
        for h in range(24):
            prev_h = prev_stock[:, h:h+1]
            s_h_in = torch.cat([s_flat_base, prev_h], dim=-1)
            logits.append(self.seasonal_linear_heads[h](s_h_in) + self.trend_linear_heads[h](t_flat))

        return torch.cat(logits, dim=1)


class ContinuousHour_DecompTransformer(nn.Module):
    """
    Temporal Decomposition Transformer with Hour-Continuity Attention.

    Applies self-attention across time steps for seasonal dynamics, an MLP for trend,
    and a second self-attention layer across 24 hourly queries to enforce smooth diurnal transitions.
    """
    def __init__(
        self,
        seq_len=14,
        input_dim=14,
        num_products=68,
        num_cities=11,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ):
        super().__init__()
        self.ma = MovingAvg(kernel_size=5)
        self.prod_emb = nn.Embedding(num_products, d_model // 2)
        self.city_emb = nn.Embedding(num_cities, d_model // 2)

        self.seasonal_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        self.hour_queries = nn.Embedding(24, d_model)
        self.prev_stock_proj = nn.Linear(1, d_model)
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        self.hour_self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=2, dropout=dropout, batch_first=True)
        self.hour_self_norm = nn.LayerNorm(d_model)

        self.trend_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim + d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x_seq, prev_stock, cat_meta):
        B, T, _ = x_seq.shape
        static_ctx = torch.cat([self.prod_emb(cat_meta[:, 0]), self.city_emb(cat_meta[:, 1])], dim=-1)

        trend = self.ma(x_seq)
        seasonal = x_seq - trend

        s = self.seasonal_proj(seasonal) + static_ctx.unsqueeze(1)
        context = self.encoder(s)

        queries = self.hour_queries(torch.arange(24, device=x_seq.device)).unsqueeze(0).expand(B, -1, -1)
        prior_ctx = self.prev_stock_proj(prev_stock.unsqueeze(-1))
        cond_queries = queries + prior_ctx

        slot_ctx, _ = self.cross_attn(cond_queries, context, context)
        slot_ctx = self.cross_norm(slot_ctx + cond_queries)

        hour_cont, _ = self.hour_self_attn(slot_ctx, slot_ctx, slot_ctx)
        slot_ctx = self.hour_self_norm(slot_ctx + hour_cont)

        trend_flat = torch.cat([trend.reshape(B, -1), static_ctx], dim=-1)
        trend_emb = self.trend_mlp(trend_flat).unsqueeze(1).expand(-1, 24, -1)

        combined = torch.cat([slot_ctx, trend_emb], dim=-1)
        return self.head(combined).squeeze(-1)


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
