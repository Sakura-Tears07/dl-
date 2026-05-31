#!/usr/bin/env python3
"""
train：Transformer 日线训练、验证 / 导出 pred_score（工作流编排见 workbench.py）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from data_preprocess import resolve_data_root, DataProcessor
from backtest import (
    format_metrics_line,
    metrics_dict_for_json,
    validation_metrics_bundle,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):  # type: ignore[no-untyped-def]
        t = x.size(1)
        return x + self.pe[:, :t, :]


class TsTransformer(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_ff: int = 256,
        dropout: float = 0.1,
        base_feature_idx: List[int] | None = None,
        sidecar_feature_idx: List[int] | None = None,
        base_head_weight: float = 0.8,
    ) -> None:
        super().__init__()
        self.base_head_weight = float(base_head_weight)
        if not (0.0 < self.base_head_weight < 1.0):
            raise ValueError(f"base_head_weight 必须在 (0,1) 内，当前={base_head_weight}")

        base_idx = list(base_feature_idx or [])
        side_idx = list(sidecar_feature_idx or [])
        if any(i < 0 or i >= feat_dim for i in (base_idx + side_idx)):
            raise ValueError("feature head 索引越界，请检查 feature_cols 分组")

        self.use_feature_heads = bool(base_idx) and bool(side_idx)
        if self.use_feature_heads:
            if d_model % 2 != 0:
                raise ValueError("启用双头特征注意力时，d_model 需为偶数（用于 num_heads=2）")
            self.register_buffer("base_index", torch.tensor(base_idx, dtype=torch.long), persistent=False)
            self.register_buffer("side_index", torch.tensor(side_idx, dtype=torch.long), persistent=False)
            self.base_proj = nn.Linear(len(base_idx), d_model)
            self.side_proj = nn.Linear(len(side_idx), d_model)
            self.feature_head_attn = nn.MultiheadAttention(
                d_model, num_heads=2, dropout=dropout, batch_first=True
            )
            self.feature_head_norm = nn.LayerNorm(d_model)
            self.input_proj = None
        else:
            self.input_proj = nn.Linear(feat_dim, d_model)
            self.register_buffer("base_index", torch.empty(0, dtype=torch.long), persistent=False)
            self.register_buffer("side_index", torch.empty(0, dtype=torch.long), persistent=False)
            self.base_proj = None
            self.side_proj = None
            self.feature_head_attn = None
            self.feature_head_norm = None

        self.pos_enc = PositionalEncoding(d_model, max_len=512)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def _project_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_feature_heads:
            assert self.input_proj is not None
            return self.input_proj(x)
        assert self.base_proj is not None
        assert self.side_proj is not None
        assert self.feature_head_attn is not None
        assert self.feature_head_norm is not None
        xb = x.index_select(2, self.base_index)
        xs = x.index_select(2, self.side_index)
        hb = self.base_proj(xb)
        hs = self.side_proj(xs)
        b, t, d = hb.shape
        # 每个交易日把「普通特征头」「侧车特征头」作为两个 token 做特征级多头注意力。
        pair = torch.stack((hb, hs), dim=2).reshape(b * t, 2, d)
        pair_out, _ = self.feature_head_attn(pair, pair, pair, need_weights=False)
        hb_out = pair_out[:, 0, :]
        hs_out = pair_out[:, 1, :]
        fused = self.base_head_weight * hb_out + (1.0 - self.base_head_weight) * hs_out
        return self.feature_head_norm(fused).reshape(b, t, d)

    def forward(self, x):  # type: ignore[no-untyped-def]
        h = self._project_inputs(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        return self.head(h.mean(dim=1))


def unwrap_model(m: nn.Module) -> nn.Module:
    return m.module if hasattr(m, "module") else m



def ddp_enabled() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def ddp_setup() -> tuple[bool, int, int]:
    if not ddp_enabled():
        return False, 0, 0
    if not torch.cuda.is_available():
        raise RuntimeError("WORLD_SIZE>1 时需要 CUDA + nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    # 大面板 + 多进程各跑一轮数据管线时，rank 间可能分差分钟级；首个 DDP collective 默认 600s 会误判超时。
    # 可用 DL_DDP_TIMEOUT_SEC 覆盖（秒），下限与 torch 默认值对齐。
    _sec = max(600, int(os.environ.get("DL_DDP_TIMEOUT_SEC", str(7200))))
    dist.init_process_group(backend="nccl", timeout=timedelta(seconds=_sec))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank


def default_load_end(val_end: str, buffer_days: int = 45) -> str:
    """未指定 ``--load-end`` 时：在 val-end 之后再尝试读若干自然日日线，便于 horizon 标签。"""
    dt = datetime.strptime(val_end.strip(), "%Y-%m-%d")
    return (dt + timedelta(days=buffer_days)).strftime("%Y-%m-%d")


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(os.path.abspath(file_path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _panel_last_trade_date_str(processor: DataProcessor) -> str:
    lv = processor.df.index.get_level_values("trade_date")
    cand: list[str] = []
    for x in lv:
        s = str(x).replace("-", "").strip()
        if len(s) >= 8:
            s8 = s[:8]
            if s8.isdigit():
                cand.append(s8)
    if not cand:
        raise RuntimeError("无法从面板推断最后交易日（索引为空或格式异常）")
    return max(cand)


def _ddp_barrier(on: bool) -> None:
    if on:
        dist.barrier()


def _ymd_digits(s: str) -> str:
    """文件名/列用：YYYY-MM-DD → YYYYMMDD。"""
    return str(s).strip().replace("-", "")[:8]


def _to_tensors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[TensorDataset, TensorDataset]:
    def clean(a: np.ndarray) -> np.ndarray:
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return np.ascontiguousarray(a, dtype=np.float32)

    Xt = torch.from_numpy(clean(X_train))
    yt = torch.from_numpy(clean(y_train))
    Xv = torch.from_numpy(clean(X_val))
    yv = torch.from_numpy(clean(y_val))
    return TensorDataset(Xt, yt), TensorDataset(Xv, yv)


def _pairwise_rank_hinge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    max_pairs: int = 2048,
    margin: float = 0.0,
) -> torch.Tensor:
    """
    截面排序辅助损失（pairwise hinge）。
    仅在一个 batch 内抽样若干样本对，鼓励 pred 的相对大小与 label_return 一致。
    """
    p = pred.reshape(-1)
    y = target.reshape(-1)
    n = int(p.numel())
    if n < 2 or max_pairs <= 0:
        return p.new_zeros(())
    n_pairs = int(min(max_pairs, n * (n - 1) // 2))
    i = torch.randint(0, n, (n_pairs,), device=p.device)
    j = torch.randint(0, n, (n_pairs,), device=p.device)
    valid = i != j
    if not torch.any(valid):
        return p.new_zeros(())
    i = i[valid]
    j = j[valid]
    dy = y[i] - y[j]
    s = torch.sign(dy)
    valid2 = s != 0
    if not torch.any(valid2):
        return p.new_zeros(())
    i = i[valid2]
    j = j[valid2]
    s = s[valid2]
    # s*(p_i-p_j) 越大越好；小于 margin 时产生惩罚
    logits = s * (p[i] - p[j])
    return torch.relu(float(margin) - logits).mean()


def _ddp_pipeline_cache_path(
    cache_dir: str,
    *,
    data_root: str,
    workflow: str,
    stock_pool: str,
    start_date: str,
    end_date: str,
    train_range: Tuple[str, str],
    val_range: Tuple[str, str],
    window_len: int,
    horizon: int,
    use_sidecars: bool,
) -> str:
    payload = {
        "data_root": os.path.abspath(data_root),
        "workflow": workflow,
        "stock_pool": stock_pool,
        "start_date": start_date,
        "end_date": end_date,
        "train_range": list(train_range),
        "val_range": list(val_range),
        "window_len": int(window_len),
        "horizon": int(horizon),
        "use_sidecars": bool(use_sidecars),
    }
    key = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"pipeline_{key}.npz")


def _save_pipeline_cache(
    path: str,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_cols: List[str],
    train_dates: np.ndarray,
    train_stocks: np.ndarray,
    val_dates: np.ndarray,
    val_stocks: np.ndarray,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    scaler_var: np.ndarray,
    scaler_n_samples_seen: np.ndarray,
    panel_last_trade_date: str,
) -> None:
    tmp = path + ".tmp.npz"
    np.savez_compressed(
        tmp,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_cols=np.asarray(feature_cols, dtype=object),
        train_dates=np.asarray(train_dates, dtype=object),
        train_stocks=np.asarray(train_stocks, dtype=object),
        val_dates=np.asarray(val_dates, dtype=object),
        val_stocks=np.asarray(val_stocks, dtype=object),
        scaler_mean=np.asarray(scaler_mean, dtype=np.float64),
        scaler_scale=np.asarray(scaler_scale, dtype=np.float64),
        scaler_var=np.asarray(scaler_var, dtype=np.float64),
        scaler_n_samples_seen=np.asarray(scaler_n_samples_seen),
        panel_last_trade_date=np.asarray([panel_last_trade_date], dtype=object),
    )
    os.replace(tmp, path)


def _load_pipeline_cache(path: str) -> Dict[str, Any]:
    d = np.load(path, allow_pickle=True)
    return {
        "X_train": d["X_train"],
        "y_train": d["y_train"],
        "X_val": d["X_val"],
        "y_val": d["y_val"],
        "feature_cols": [str(x) for x in d["feature_cols"].tolist()],
        "train_dates": d["train_dates"].astype(str),
        "train_stocks": d["train_stocks"].astype(str),
        "val_dates": d["val_dates"].astype(str),
        "val_stocks": d["val_stocks"].astype(str),
        "scaler_mean": d["scaler_mean"].astype(np.float64),
        "scaler_scale": d["scaler_scale"].astype(np.float64),
        "scaler_var": d["scaler_var"].astype(np.float64),
        "scaler_n_samples_seen": d["scaler_n_samples_seen"],
        "panel_last_trade_date": str(d["panel_last_trade_date"][0]),
    }


def _split_feature_heads(feature_cols: List[str]) -> Tuple[List[int], List[int]]:
    base_idx: List[int] = []
    side_idx: List[int] = []
    side_prefix = ("mf_", "mtr_", "idx_")
    for i, col in enumerate(feature_cols):
        if str(col).startswith(side_prefix):
            side_idx.append(i)
        else:
            base_idx.append(i)
    return base_idx, side_idx


def _is_sidecar_feature(col: str) -> bool:
    return str(col).startswith(("mf_", "mtr_", "idx_"))


def _compute_feature_scores_ic(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_dates: np.ndarray,
    *,
    min_names_per_day: int,
) -> np.ndarray:
    """
    基于训练集样本（每个锚定日的截面）计算单特征打分。
    使用窗口最后一个时点特征，对齐标签做按日截面 Pearson IC。
    """
    x_last = np.asarray(X_train[:, -1, :], dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    dates = pd.Series(np.asarray(train_dates).astype(str))
    by_day = dates.groupby(dates).indices
    n_feat = x_last.shape[1]
    daily_ics: List[np.ndarray] = []

    for idx in by_day.values():
        xv = x_last[idx, :]
        yv = y[idx]
        m = np.isfinite(yv)
        if int(m.sum()) < min_names_per_day:
            continue
        xv = xv[m, :]
        yv = yv[m]
        n = xv.shape[0]
        if n < min_names_per_day:
            continue
        y_std = float(np.std(yv, ddof=1))
        if y_std < 1e-12 or not np.isfinite(y_std):
            continue
        x_std = np.std(xv, axis=0, ddof=1)
        valid = np.isfinite(x_std) & (x_std > 1e-12)
        if not np.any(valid):
            continue
        xc = xv - np.mean(xv, axis=0, keepdims=True)
        yc = yv - float(np.mean(yv))
        cov = np.sum(xc * yc[:, None], axis=0) / max(n - 1, 1)
        corr = np.full(n_feat, np.nan, dtype=np.float64)
        corr[valid] = cov[valid] / (x_std[valid] * y_std)
        daily_ics.append(corr)

    out = np.full(n_feat, -1e9, dtype=np.float64)
    if not daily_ics:
        return out
    ic_mat = np.vstack(daily_ics)  # (n_days, n_feat)
    with np.errstate(all="ignore"):
        mean_ic = np.nanmean(ic_mat, axis=0)
        std_ic = np.nanstd(ic_mat, axis=0, ddof=1)
    valid_count = np.sum(np.isfinite(ic_mat), axis=0)
    std_ic = np.where(valid_count >= 2, std_ic, np.nan)
    with np.errstate(all="ignore"):
        icir = mean_ic / std_ic
    icir_pos = np.where(np.isfinite(icir), np.maximum(icir, 0.0), 0.0)
    score = 0.7 * np.abs(mean_ic) + 0.3 * icir_pos
    good = np.isfinite(score)
    out[good] = score[good]
    return out


def _select_feature_indices(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_dates: np.ndarray,
    feature_cols: List[str],
    *,
    mode: str,
    base_budget: int,
    sidecar_budget: int,
    corr_threshold: float,
    min_names_per_day: int,
    keep_all_base_features: bool,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    训练前特征削减：
    1) 单特征 IC 打分（训练集）；
    2) 相关性去冗余（贪心）；
    3) base/sidecar 分组预算保留（默认可配置为 base 全保留，仅筛 sidecar）。
    """
    n_feat = len(feature_cols)
    all_idx = np.arange(n_feat, dtype=np.int64)
    if mode == "none" or n_feat == 0:
        return all_idx, {"mode": mode, "selected_count": int(n_feat), "dropped_count": 0}

    scores = _compute_feature_scores_ic(
        X_train,
        y_train,
        train_dates,
        min_names_per_day=min_names_per_day,
    )

    x_last = np.asarray(X_train[:, -1, :], dtype=np.float64)
    # 相关性矩阵（无方差列视为 0 相关，靠 score 自动后排）
    if n_feat == 1:
        corr = np.ones((1, 1), dtype=np.float64)
    else:
        with np.errstate(all="ignore"):
            corr = np.corrcoef(x_last, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr)

    forced_names = {"close", "pct_chg", "vol"}
    forced_idx = [i for i, c in enumerate(feature_cols) if c in forced_names]
    ordered = sorted(range(n_feat), key=lambda i: float(scores[i]), reverse=True)

    base_all = [i for i, c in enumerate(feature_cols) if not _is_sidecar_feature(c)]
    side_all = [i for i, c in enumerate(feature_cols) if _is_sidecar_feature(c)]

    def _pick_group(cands: List[int], budget: int, forced: List[int]) -> List[int]:
        if budget <= 0:
            return list(cands)
        out_g: List[int] = []
        for i in forced:
            if i in cands and i not in out_g:
                out_g.append(i)
        remain = [i for i in cands if i not in out_g]
        remain = sorted(remain, key=lambda i: float(scores[i]), reverse=True)
        for i in remain:
            if len(out_g) >= budget:
                break
            out_g.append(i)
        return out_g

    if keep_all_base_features:
        sel_base = list(base_all)
        side_order = sorted(side_all, key=lambda i: float(scores[i]), reverse=True)
        sel_side: List[int] = []
        for i in side_order:
            if any(corr[i, j] >= corr_threshold for j in sel_side):
                continue
            sel_side.append(i)
        if sidecar_budget > 0:
            sel_side = sel_side[:sidecar_budget]
        selected = sorted(set(sel_base + sel_side))
    else:
        kept: List[int] = []
        for i in forced_idx:
            if i not in kept:
                kept.append(i)
        for i in ordered:
            if i in kept:
                continue
            if any(corr[i, k] >= corr_threshold for k in kept):
                continue
            kept.append(i)

        base_cands = [i for i in kept if not _is_sidecar_feature(feature_cols[i])]
        side_cands = [i for i in kept if _is_sidecar_feature(feature_cols[i])]
        base_forced = [i for i in forced_idx if i in base_cands]
        side_forced = [i for i in forced_idx if i in side_cands]

        sel_base = _pick_group(base_cands, base_budget, base_forced)
        sel_side = _pick_group(side_cands, sidecar_budget, side_forced)
        selected = sorted(set(sel_base + sel_side))
    if not selected:
        selected = sorted(ordered[: min(8, n_feat)])

    selected_set = set(selected)
    report = {
        "mode": mode,
        "total_features_before": int(n_feat),
        "selected_count": int(len(selected)),
        "dropped_count": int(n_feat - len(selected)),
        "base_selected": int(sum(1 for i in selected if not _is_sidecar_feature(feature_cols[i]))),
        "sidecar_selected": int(sum(1 for i in selected if _is_sidecar_feature(feature_cols[i]))),
        "keep_all_base_features": bool(keep_all_base_features),
        "base_budget": int(base_budget),
        "sidecar_budget": int(sidecar_budget),
        "corr_threshold": float(corr_threshold),
        "selected_features": [feature_cols[i] for i in selected],
        "dropped_features": [feature_cols[i] for i in range(n_feat) if i not in selected_set],
        "feature_scores": {feature_cols[i]: float(scores[i]) for i in range(n_feat)},
    }
    return np.asarray(selected, dtype=np.int64), report


def _early_stop_score(metric: str, bundle: Dict[str, Any] | None, val_loss: float) -> float:
    if metric == "val_loss":
        return float(-val_loss)
    if not bundle:
        return float("nan")
    if metric == "rank_ic":
        return float(bundle.get("rank_ic_daily_mean", np.nan))
    if metric == "ic_pearson":
        return float(bundle.get("ic_daily_pearson_mean", np.nan))
    if metric == "combo":
        r = float(bundle.get("rank_ic_daily_mean", np.nan))
        p = float(bundle.get("ic_daily_pearson_mean", np.nan))
        if not np.isfinite(r) or not np.isfinite(p):
            return float("nan")
        return float(0.7 * r + 0.3 * p)
    raise ValueError(f"未知 early-stop metric: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="日线 Transformer 基线训练（默认全市场池：排除 ST/北交所，见 --stock-pool）"
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DL_DATA_ROOT", ""),
        help="数据根目录；空则环境与 data_preprocess.resolve_data_root 默认（仓库上一级 data/）",
    )
    parser.add_argument(
        "--stock-pool",
        choices=("all", "hs300", "cyb", "kcb"),
        default="all",
        help="股票池：all=本地 daily 有行情的标的（排除 ST/北交所）；hs300 需 data/index_weight；cyb/kcb 按 basic.csv 市场列",
    )
    parser.add_argument(
        "--workflow",
        choices=("backtest", "predict-next"),
        default="backtest",
        help=(
            "backtest：训练+验证切分，可导出验证打分并做历史回测；"
            "predict-next：仅用监督样本全集训练（无验证 IC），必须配合 --export-scores，"
            "默认 --export-infer-anchor auto（面板末日推理打分 → 次日操作建议）"
        ),
    )
    parser.add_argument("--window-len", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dim-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.0,
        help="排序辅助损失权重；>0 时启用 batch 内 pairwise hinge rank loss",
    )
    parser.add_argument(
        "--rank-loss-max-pairs",
        type=int,
        default=2048,
        help="每个 batch 采样的 pairwise 对数上限（排序损失）",
    )
    parser.add_argument(
        "--rank-loss-margin",
        type=float,
        default=0.0,
        help="pairwise hinge margin（排序损失）",
    )
    parser.add_argument("--quick", action="store_true", help="缩短日期区间快速试跑")
    parser.add_argument(
        "--train-start",
        default="2016-01-01",
        help="训练集样本锚定日起始（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--train-end",
        default="2026-01-01",
        help="训练集样本锚定日结束（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--val-start",
        default="2026-01-02",
        help="验证集样本锚定日起始（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--val-end",
        default="2026-02-02",
        help="验证集样本锚定日结束（含），YYYY-MM-DD",
    )
    parser.add_argument(
        "--load-end",
        default=None,
        metavar="DATE",
        help="读取 daily/ 的结束日期（含）；默认 = val-end + 45 日历日",
    )
    parser.add_argument(
        "--export-scores",
        default=None,
        metavar="CSV",
        help="训练结束后写出打分 CSV：默认为验证集；若指定 --export-infer-anchor 则仅写出该锚定日的推理打分（label_return 为空）",
    )
    parser.add_argument(
        "--export-infer-anchor",
        default=None,
        metavar="DATE",
        help=(
            "无需 T+1 行情：在面板最后交易日即可导出锚定日 pred_score（YYYY-MM-DD）。"
            "可用关键字 auto（默认用于 workflow=predict-next）：自动取面板最后交易日。"
            "须同时指定 --export-scores；workflow=predict-next 时推理窗口会包含该锚定日当日特征。"
            "典型用法：日线只到 20260519 时，取 --val-* 为倒数第二个交易日作验证，"
            "本参数填面板末日用于次日决策。"
        ),
    )
    parser.add_argument(
        "--export-top-per-day",
        default=None,
        metavar="CSV",
        help="每日 pred_score 最高的 Top-N 标的（含 rank），便于对照作业「买入得分最高」策略",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="与 --export-top-per-day 配合：每个交易日保留前 N 只股票",
    )
    parser.add_argument(
        "--multi-gpu",
        action="store_true",
        help="单机多卡时使用 nn.DataParallel（单进程）。若已用 torchrun 多进程会自动走 DDP",
    )
    parser.add_argument(
        "--export-metrics-json",
        default=None,
        metavar="PATH",
        help="最后一个 epoch 的验证集指标写入 JSON（IC/ICIR/RankIC/方向胜率等）",
    )
    parser.add_argument(
        "--export-daily-ic-csv",
        default=None,
        metavar="PATH",
        help="导出按日 Pearson IC 时间序列 CSV（便于画图）",
    )
    parser.add_argument(
        "--min-names-per-day",
        type=int,
        default=10,
        help="计算截面 IC 时每个交易日至少需要的股票数量",
    )
    parser.add_argument(
        "--no-data-sidecars",
        action="store_true",
        help="关闭并入 data/moneyflow、metric、market（默认开启，与 README data/ 对齐）。",
    )
    parser.add_argument(
        "--base-head-weight",
        type=float,
        default=0.8,
        help="双头特征注意力中普通特征头权重（侧车头权重=1-该值），默认 0.8",
    )
    parser.add_argument(
        "--feature-select-mode",
        choices=("none", "ic_prune"),
        default="ic_prune",
        help="训练前特征削减策略：none=不筛选；ic_prune=按训练集截面 IC + 去相关 + 分组预算筛选",
    )
    parser.add_argument(
        "--keep-all-base-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="特征筛选时是否保留全部 base 特征（默认开启，仅筛 sidecar）",
    )
    parser.add_argument(
        "--base-feature-budget",
        type=int,
        default=24,
        help="feature-select-mode=ic_prune 且关闭 --keep-all-base-features 时生效；<=0 表示不限制",
    )
    parser.add_argument(
        "--sidecar-feature-budget",
        type=int,
        default=10,
        help="feature-select-mode=ic_prune 时保留的侧车特征数量预算；<=0 表示不限制",
    )
    parser.add_argument(
        "--feature-corr-threshold",
        type=float,
        default=0.95,
        help="ic_prune 去相关阈值（按训练样本窗口末日特征相关系数绝对值）",
    )
    parser.add_argument(
        "--feature-ic-min-names",
        type=int,
        default=20,
        help="计算单特征按日截面 IC 时每日至少样本数",
    )
    parser.add_argument(
        "--feature-report-path",
        default=None,
        metavar="PATH",
        help="写出特征筛选报告 JSON（保留列、剔除列、IC 打分等）",
    )
    parser.add_argument(
        "--ddp-pipeline-cache-dir",
        default=".cache/ddp_pipeline",
        metavar="DIR",
        help="DDP 多进程时的数据管线缓存目录（减少各 rank 重复预处理）",
    )
    parser.add_argument(
        "--disable-early-stop",
        action="store_true",
        help="关闭早停（默认开启；workflow=predict-next 自动禁用）",
    )
    parser.add_argument(
        "--early-stop-metric",
        choices=("rank_ic", "ic_pearson", "combo", "val_loss"),
        default="rank_ic",
        help="早停监控指标：rank_ic / ic_pearson / combo / val_loss",
    )
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=8,
        help="早停耐心轮数（超过该轮数无改进则停止）",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=12,
        help="最少训练轮数（在此之前不触发早停）",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=5e-4,
        help="判定改进所需最小增量（对监控分数而言）",
    )
    args = parser.parse_args()
    args.data_root = resolve_data_root(args.data_root)

    if not (0.0 < args.feature_corr_threshold < 1.0):
        raise ValueError("--feature-corr-threshold 必须在 (0,1)")
    if args.feature_ic_min_names < 2:
        raise ValueError("--feature-ic-min-names 必须 >=2")
    if args.rank_loss_weight < 0:
        raise ValueError("--rank-loss-weight 必须 >=0")
    if args.rank_loss_max_pairs < 0:
        raise ValueError("--rank-loss-max-pairs 必须 >=0")
    if args.early_stop_patience < 1:
        raise ValueError("--early-stop-patience 必须 >=1")
    if args.early_stop_min_epochs < 1:
        raise ValueError("--early-stop-min-epochs 必须 >=1")

    if args.export_infer_anchor and not args.export_scores:
        raise ValueError("使用 --export-infer-anchor 时必须指定 --export-scores（输出路径）")

    if args.quick and args.workflow == "predict-next":
        raise ValueError("--workflow predict-next 不能与 --quick 同时使用")

    if args.workflow == "predict-next":
        if not args.export_scores:
            raise ValueError("workflow=predict-next 时必须指定 --export-scores（写出用于次日决策的打分 CSV）")
        if args.export_infer_anchor is None:
            args.export_infer_anchor = "auto"

    if not args.quick:
        ta, tb = _ymd_digits(args.train_start), _ymd_digits(args.train_end)
        if ta > tb:
            raise ValueError("train-start 不能晚于 train-end")
        if args.workflow == "backtest":
            if _ymd_digits(args.train_end) >= _ymd_digits(args.val_start):
                raise ValueError(
                    "train-end 必须早于 val-start（样本锚定日的严格时间因果）；"
                    f"当前 train-end={args.train_end}, val-start={args.val_start}"
                )
            if _ymd_digits(args.val_start) > _ymd_digits(args.val_end):
                raise ValueError("val-start 不能晚于 val-end")

    use_ddp, rank, local_rank = ddp_setup()
    main_rank = (not use_ddp) or (rank == 0)

    # 降低 torch 中与正确性无关的 CLI 告警噪声（不改变训练/打分数值）
    import warnings as _warnings

    _warnings.filterwarnings("ignore", message=r".*enable_nested_tensor.*", category=UserWarning)
    _warnings.filterwarnings(
        "ignore", message=r".*barrier\(\): using the device.*", category=UserWarning
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.quick:
        start_date = "2024-01-01"
        end_date = "2025-06-30"
        train_range = ("2024-01-01", "2024-12-31")
        val_range = ("2025-01-01", "2025-06-30")
    elif args.workflow == "predict-next":
        start_date = args.train_start
        end_date = args.load_end or default_load_end(args.train_end)
        train_range = (args.train_start, args.train_end)
        val_range = ("2999-12-31", "2999-12-31")
    else:
        start_date = args.train_start
        end_date = args.load_end or default_load_end(args.val_end)
        train_range = (args.train_start, args.train_end)
        val_range = (args.val_start, args.val_end)

    if main_rank:
        val_note = (
            f"{val_range[0]}..{val_range[1]}"
            if args.workflow != "predict-next"
            else "(跳过，workflow=predict-next)"
        )
        print(
            f"[workflow={args.workflow}] load_daily: {start_date} .. {end_date} | "
            f"train_anchor: {train_range[0]}..{train_range[1]} | "
            f"val_anchor: {val_note} | stock_pool={args.stock_pool}"
        )

    if use_ddp:
        device = torch.device("cuda", local_rank)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if main_rank:
        if use_ddp:
            ws = int(os.environ.get("WORLD_SIZE", "1"))
            print(f"DDP: world_size={ws}, rank={rank}, device={device}")
        else:
            print(f"device={device}")

    use_sidecars = not bool(getattr(args, "no_data_sidecars", False))
    processor_main: DataProcessor | None = None
    pipeline_scaler: StandardScaler | None = None
    panel_last_trade_date = ""

    if use_ddp and args.ddp_pipeline_cache_dir:
        cache_path = _ddp_pipeline_cache_path(
            args.ddp_pipeline_cache_dir,
            data_root=args.data_root,
            workflow=args.workflow,
            stock_pool=args.stock_pool,
            start_date=start_date,
            end_date=end_date,
            train_range=train_range,
            val_range=val_range,
            window_len=args.window_len,
            horizon=args.horizon,
            use_sidecars=use_sidecars,
        )
        cache_payload: Dict[str, Any] | None = None
        if main_rank and os.path.isfile(cache_path):
            cache_payload = _load_pipeline_cache(cache_path)
            print(f"[ddp-cache] hit: {cache_path}")

        if main_rank and cache_payload is None:
            processor_main = DataProcessor(args.data_root, verbose=main_rank)
            X_train, y_train, X_val, y_val = processor_main.run_pipeline(
                start_date=start_date,
                end_date=end_date,
                stock_pool=args.stock_pool,
                window_len=args.window_len,
                horizon=args.horizon,
                train_range=train_range,
                val_range=val_range,
                add_ta=False,
                use_all_daily_columns=True,
                allow_empty_val=(args.workflow == "predict-next"),
                use_data_moneyflow_metric_index=use_sidecars,
            )
            panel_last_trade_date = _panel_last_trade_date_str(processor_main)
            pipeline_scaler = processor_main.scaler
            assert pipeline_scaler is not None
            _save_pipeline_cache(
                cache_path,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_cols=[str(c) for c in (processor_main.feature_cols or [])],
                train_dates=np.asarray(processor_main.train_dates).astype(str),
                train_stocks=np.asarray(processor_main.train_stocks).astype(str),
                val_dates=np.asarray(processor_main.val_dates).astype(str),
                val_stocks=np.asarray(processor_main.val_stocks).astype(str),
                scaler_mean=np.asarray(pipeline_scaler.mean_, dtype=np.float64),
                scaler_scale=np.asarray(pipeline_scaler.scale_, dtype=np.float64),
                scaler_var=np.asarray(pipeline_scaler.var_, dtype=np.float64),
                scaler_n_samples_seen=np.asarray(pipeline_scaler.n_samples_seen_),
                panel_last_trade_date=panel_last_trade_date,
            )
            print(f"[ddp-cache] wrote: {cache_path}")

        _ddp_barrier(use_ddp)
        if (not main_rank) or (cache_payload is None):
            cache_payload = _load_pipeline_cache(cache_path)
            if not main_rank:
                print(f"[ddp-cache] load on rank{rank}: {cache_path}")

        assert cache_payload is not None
        X_train = np.asarray(cache_payload["X_train"], dtype=np.float32)
        y_train = np.asarray(cache_payload["y_train"], dtype=np.float32)
        X_val = np.asarray(cache_payload["X_val"], dtype=np.float32)
        y_val = np.asarray(cache_payload["y_val"], dtype=np.float32)
        full_feature_cols = [str(c) for c in cache_payload["feature_cols"]]
        train_dates_arr = np.asarray(cache_payload["train_dates"]).astype(str)
        val_dates_arr = np.asarray(cache_payload["val_dates"]).astype(str)
        val_stocks_arr = np.asarray(cache_payload["val_stocks"]).astype(str)
        panel_last_trade_date = str(cache_payload["panel_last_trade_date"])
        if pipeline_scaler is None:
            pipeline_scaler = StandardScaler()
            pipeline_scaler.mean_ = np.asarray(cache_payload["scaler_mean"], dtype=np.float64)
            pipeline_scaler.scale_ = np.asarray(cache_payload["scaler_scale"], dtype=np.float64)
            pipeline_scaler.var_ = np.asarray(cache_payload["scaler_var"], dtype=np.float64)
            pipeline_scaler.n_samples_seen_ = np.asarray(cache_payload["scaler_n_samples_seen"])
            pipeline_scaler.n_features_in_ = int(pipeline_scaler.mean_.shape[0])
    else:
        processor_main = DataProcessor(args.data_root, verbose=main_rank)
        X_train, y_train, X_val, y_val = processor_main.run_pipeline(
            start_date=start_date,
            end_date=end_date,
            stock_pool=args.stock_pool,
            window_len=args.window_len,
            horizon=args.horizon,
            train_range=train_range,
            val_range=val_range,
            add_ta=False,
            use_all_daily_columns=True,
            allow_empty_val=(args.workflow == "predict-next"),
            use_data_moneyflow_metric_index=use_sidecars,
        )
        full_feature_cols = [str(c) for c in (processor_main.feature_cols or [])]
        train_dates_arr = np.asarray(processor_main.train_dates).astype(str)
        val_dates_arr = np.asarray(processor_main.val_dates).astype(str)
        val_stocks_arr = np.asarray(processor_main.val_stocks).astype(str)
        panel_last_trade_date = _panel_last_trade_date_str(processor_main)
        pipeline_scaler = processor_main.scaler

    if (
        args.export_infer_anchor is not None
        and str(args.export_infer_anchor).strip().lower() == "auto"
    ):
        args.export_infer_anchor = panel_last_trade_date
        if main_rank:
            print(f"[infer] export-infer-anchor=auto → 使用面板最后交易日 {args.export_infer_anchor}")

    selected_feature_idx = np.arange(len(full_feature_cols), dtype=np.int64)
    feature_select_report: Dict[str, Any] = {
        "mode": args.feature_select_mode,
        "total_features_before": len(full_feature_cols),
    }
    if len(full_feature_cols) == 0:
        raise RuntimeError("feature_cols 为空，无法训练。")
    if args.feature_select_mode != "none":
        if use_ddp:
            if main_rank:
                print("[feature_select] computing on rank0 ...")
                selected_feature_idx, feature_select_report = _select_feature_indices(
                    X_train,
                    y_train,
                    train_dates_arr,
                    full_feature_cols,
                    mode=args.feature_select_mode,
                    base_budget=args.base_feature_budget,
                    sidecar_budget=args.sidecar_feature_budget,
                    corr_threshold=args.feature_corr_threshold,
                    min_names_per_day=args.feature_ic_min_names,
                    keep_all_base_features=args.keep_all_base_features,
                )
                sel_size = int(len(selected_feature_idx))
            else:
                sel_size = 0
            sz_t = torch.tensor([sel_size], device=device, dtype=torch.int64)
            dist.broadcast(sz_t, src=0)
            recv_n = int(sz_t.item())
            if main_rank:
                idx_t = torch.tensor(selected_feature_idx.tolist(), device=device, dtype=torch.int64)
            else:
                idx_t = torch.zeros(recv_n, device=device, dtype=torch.int64)
            dist.broadcast(idx_t, src=0)
            selected_feature_idx = idx_t.detach().cpu().numpy().astype(np.int64)
        else:
            selected_feature_idx, feature_select_report = _select_feature_indices(
                X_train,
                y_train,
                train_dates_arr,
                full_feature_cols,
                mode=args.feature_select_mode,
                base_budget=args.base_feature_budget,
                sidecar_budget=args.sidecar_feature_budget,
                corr_threshold=args.feature_corr_threshold,
                min_names_per_day=args.feature_ic_min_names,
                keep_all_base_features=args.keep_all_base_features,
            )
        X_train = X_train[:, :, selected_feature_idx]
        X_val = X_val[:, :, selected_feature_idx]
    model_feature_cols = [full_feature_cols[int(i)] for i in selected_feature_idx.tolist()]

    if main_rank and args.feature_report_path:
        _ensure_parent_dir(args.feature_report_path)
        with open(args.feature_report_path, "w", encoding="utf-8") as f:
            json.dump(feature_select_report, f, indent=2, ensure_ascii=False)
        print(f"wrote feature report: {args.feature_report_path}")

    # 仅在 rank0 打印；避免把整个 feature_cols 列表刷 stdout（巨量 I/O 会拖慢 rank0，触发其他 rank DDP init 超时）
    if main_rank:
        cols = model_feature_cols
        n_f = len(cols)
        head = cols[:min(12, n_f)]
        print(f"feature_cols count={n_f}; head[:12]={head}")
        if n_f > 12:
            print(f"feature_cols tail (last 4) ... {cols[-4:]}")
        if args.feature_select_mode != "none":
            print(
                f"[feature_select] mode={args.feature_select_mode} "
                f"before={len(full_feature_cols)} after={len(model_feature_cols)} "
                f"(base={feature_select_report.get('base_selected')}, sidecar={feature_select_report.get('sidecar_selected')})"
            )
        print("shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape)
    _ddp_barrier(use_ddp)

    if len(X_train) == 0:
        raise RuntimeError(
            "训练样本数为 0：请检查本地 daily 是否覆盖所选日期区间，或放宽 train-start/train-end。"
        )
    if args.workflow == "backtest" and len(X_val) == 0:
        raise RuntimeError(
            "验证样本数为 0：请检查 train/val 日期与 horizon（详见报错上方的 create_sequences 提示）。"
        )

    _, _, feat_dim = X_train.shape
    full_feat_dim = len(full_feature_cols)
    feature_cols = model_feature_cols
    base_feature_idx, sidecar_feature_idx = _split_feature_heads(feature_cols)
    if main_rank:
        if base_feature_idx and sidecar_feature_idx:
            print(
                f"[feature_heads] base={len(base_feature_idx)}, sidecar={len(sidecar_feature_idx)}, "
                f"base_head_weight={args.base_head_weight:.2f}"
            )
        else:
            print(
                f"[feature_heads] sidecar 分组不足，退化为单输入投影（base={len(base_feature_idx)}, "
                f"sidecar={len(sidecar_feature_idx)}）"
            )
    train_ds, val_ds = _to_tensors(X_train, y_train, X_val, y_val)
    _ddp_barrier(use_ddp)
    train_sampler = None
    if use_ddp:
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    _ddp_barrier(use_ddp)

    model = TsTransformer(
        feat_dim=feat_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_ff=args.dim_ff,
        dropout=args.dropout,
        base_feature_idx=base_feature_idx,
        sidecar_feature_idx=sidecar_feature_idx,
        base_head_weight=args.base_head_weight,
    ).to(device)

    _ddp_barrier(use_ddp)
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    elif args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        if main_rank:
            print(f"DataParallel: using {torch.cuda.device_count()} GPUs")

    core = unwrap_model(model)
    opt = torch.optim.AdamW(core.parameters(), lr=args.lr)
    loss_fn = nn.SmoothL1Loss(beta=0.01)

    last_bundle = None
    # 兼容保留相关参数，但不再启用早停：固定训练到 args.epochs
    early_stop_enabled = False
    if main_rank and args.workflow == "backtest" and len(val_ds) > 0:
        print("[early_stop] disabled: fixed-epoch training is enabled")
    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        model.train()
        total = 0.0
        total_reg = 0.0
        total_rank = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            reg_loss = loss_fn(pred, yb)
            rank_loss = _pairwise_rank_hinge_loss(
                pred,
                yb,
                max_pairs=int(args.rank_loss_max_pairs),
                margin=float(args.rank_loss_margin),
            )
            loss = reg_loss + float(args.rank_loss_weight) * rank_loss
            loss.backward()
            nn.utils.clip_grad_norm_(core.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * xb.size(0)
            total_reg += float(reg_loss.item()) * xb.size(0)
            total_rank += float(rank_loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        # DDP：各 rank 只负责一部分 batch；必须用 all_reduce 得到全局加权平均，
        # 否则日志里的 train_loss 只是 rank0 分片均值，易产生误导。
        if use_ddp:
            stat = torch.tensor(
                [total, total_reg, total_rank, float(n_seen)],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(stat, op=dist.ReduceOp.SUM)
            denom = stat[3].clamp(min=1.0)
            train_loss = float((stat[0] / denom).item())
            train_reg_loss = float((stat[1] / denom).item())
            train_rank_loss = float((stat[2] / denom).item())
        else:
            train_loss = total / max(n_seen, 1)
            train_reg_loss = total_reg / max(n_seen, 1)
            train_rank_loss = total_rank / max(n_seen, 1)

        has_val = len(val_ds) > 0
        val_loss = float("nan")
        bundle = None
        if main_rank:
            # DDP 下不可仅在 rank0 对包装后的 model 做 forward（会触发 NCCL 死锁）；对裸模块 core 推理。
            core.eval()
            if has_val:
                v_total = 0.0
                v_n = 0
                preds = []
                targs = []
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
                        pred = core(xb)
                        v_total += float(loss_fn(pred, yb).item()) * xb.size(0)
                        v_n += xb.size(0)
                        preds.append(pred.detach().cpu().numpy())
                        targs.append(yb.detach().cpu().numpy())
                val_loss = v_total / max(v_n, 1)
                pv = np.concatenate(preds, axis=0)
                tv = np.concatenate(targs, axis=0)
                vd = val_dates_arr
                bundle = validation_metrics_bundle(
                    pv.reshape(-1),
                    tv.reshape(-1),
                    vd,
                    min_names_per_day=args.min_names_per_day,
                )
                last_bundle = bundle
                print(
                    f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}"
                    f" (reg={train_reg_loss:.6f}, rank={train_rank_loss:.6f}, w={args.rank_loss_weight:.3g})"
                    f"  val_loss={val_loss:.6f}  "
                    f"{format_metrics_line(bundle)}"
                )
            else:
                print(
                    f"epoch {epoch}/{args.epochs}  train_loss={train_loss:.6f}"
                    f" (reg={train_reg_loss:.6f}, rank={train_rank_loss:.6f}, w={args.rank_loss_weight:.3g})  "
                    f"(无验证集，跳过 val_loss / IC；见训练结束后的推理打分 CSV)"
                )
        else:
            core.eval()

        _ddp_barrier(use_ddp)

    _ddp_barrier(use_ddp)

    metrics_bundle_for_export = last_bundle

    if main_rank and metrics_bundle_for_export is not None and args.export_metrics_json:
        _ensure_parent_dir(args.export_metrics_json)
        with open(args.export_metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics_dict_for_json(metrics_bundle_for_export), f, indent=2, ensure_ascii=False)
        print(f"wrote metrics json: {args.export_metrics_json}")

    if main_rank and metrics_bundle_for_export is not None and args.export_daily_ic_csv:
        ser = metrics_bundle_for_export.get("_series_ic_pearson_daily")
        if ser is not None and len(ser) > 0:
            _ensure_parent_dir(args.export_daily_ic_csv)
            df_ic = ser.reset_index()
            df_ic.columns = ["trade_date", "ic_pearson_daily"]
            df_ic.to_csv(args.export_daily_ic_csv, index=False)
            print(f"wrote daily IC csv: {args.export_daily_ic_csv}")

    # -------- 打分导出（仅主进程）：验证集 或 末日推理 ----------
    if main_rank and (args.export_scores or args.export_top_per_day):
        core.eval()
        if args.export_infer_anchor:
            anchor_s = _ymd_digits(args.export_infer_anchor)
            pinf = DataProcessor(args.data_root, verbose=main_rank)
            pinf.load_data(start_date, end_date, args.stock_pool, exclude_st=True, exclude_bj=True)
            if use_sidecars:
                from data_preprocess import merge_moneyflow_metric_market_into_panel

                pinf.df = merge_moneyflow_metric_market_into_panel(
                    pinf.df, str(args.data_root), verbose=main_rank
                )
            pinf.select_features(add_ta=False, use_all_daily_columns=True)
            pinf.construct_labels(horizon=args.horizon)
            if args.workflow == "predict-next":
                X_inf, st_inf = pinf.build_inference_X_for_next_trade(anchor_s, args.window_len)
            else:
                X_inf, st_inf = pinf.build_inference_X_at_anchor(anchor_s, args.window_len)
            if tuple(pinf.feature_cols) != tuple(full_feature_cols):
                raise RuntimeError(
                    "推理分支特征列与训练不一致；请检查两次 load/select 参数是否相同。"
                )
            flat = X_inf.reshape(-1, full_feat_dim)
            if pipeline_scaler is None:
                raise RuntimeError("缺少训练阶段 scaler，无法执行推理分数导出。")
            X_inf_s = pipeline_scaler.transform(flat).astype(np.float32).reshape(X_inf.shape)
            X_inf_s = X_inf_s[:, :, selected_feature_idx]
            infer_preds: list[np.ndarray] = []
            inf_ds = TensorDataset(torch.from_numpy(X_inf_s), torch.zeros(len(X_inf_s), 1, dtype=torch.float32))
            inf_loader = DataLoader(
                inf_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
            )
            with torch.no_grad():
                for xb, _ in inf_loader:
                    xb = xb.to(device, non_blocking=True)
                    infer_preds.append(core(xb).detach().cpu().numpy())
            pv_all = np.concatenate(infer_preds, axis=0).reshape(-1)
            score_df = pd.DataFrame(
                {
                    "trade_date": anchor_s,
                    "ts_code": st_inf,
                    "pred_score": pv_all,
                    "label_return": np.nan,
                }
            )
            note = "（推理导出：无 T+1 标签，label_return 为空；predict-next 含锚定日当日输入）"
        else:
            val_preds: list[np.ndarray] = []
            with torch.no_grad():
                for xb, _ in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    val_preds.append(core(xb).detach().cpu().numpy())
            pv_all = np.concatenate(val_preds, axis=0).reshape(-1)
            vd = val_dates_arr
            vs = val_stocks_arr
            yv_all = y_val.reshape(-1).astype(np.float64)
            score_df = pd.DataFrame(
                {
                    "trade_date": vd,
                    "ts_code": vs,
                    "pred_score": pv_all,
                    "label_return": yv_all,
                }
            )
            note = ""

        if args.export_scores:
            _ensure_parent_dir(args.export_scores)
            score_df.to_csv(args.export_scores, index=False)
            print(f"wrote scores: {args.export_scores} ({len(score_df)} rows){note}")
        if args.export_top_per_day:
            _ensure_parent_dir(args.export_top_per_day)
            tops = []
            for d, g in score_df.groupby("trade_date", sort=True):
                g = g.nlargest(args.top_n, "pred_score").copy()
                g.insert(0, "rank", range(1, len(g) + 1))
                tops.append(g)
            top_df = pd.concat(tops, ignore_index=True)
            top_df.to_csv(args.export_top_per_day, index=False)
            print(
                f"wrote top-{args.top_n} per day: {args.export_top_per_day} "
                f"({len(top_df)} rows, {top_df['trade_date'].nunique()} days)"
            )

    _ddp_barrier(use_ddp)
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
