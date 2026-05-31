#!/usr/bin/env python3
"""
predict：下一交易日持仓推演（组合仿真）与操作建议 CLI。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from data_preprocess import resolve_data_root
from backtest import (
    fees_on_buy_turnover,
    fees_on_sell_turnover,
    load_scores,
    score_weights_from_picks_df,
    score_weights_from_picks_df,
    target_weights_from_panel,
    daily_rotation_trades,
    portfolio_has_holdings,
    _portfolio_mv_at_px,
)



@dataclass
class PortfolioState:
    cash: float
    lot_size: int = 100
    sellable: Dict[str, int] = field(default_factory=dict)
    locked: Dict[str, int] = field(default_factory=dict)
    commission_rate: float = 0.0002


def load_portfolio_json(path: str) -> PortfolioState:
    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = json.load(f)
    cash = float(raw.get("cash", 0.0))
    lot = int(raw.get("lot_size", 100))
    sellable = {str(k): int(v) for k, v in raw.get("sellable", {}).items() if int(v) > 0}
    locked = {str(k): int(v) for k, v in raw.get("locked", {}).items() if int(v) > 0}
    if "commission_rate" in raw:
        crate = float(raw.get("commission_rate", 0.0002))
    else:
        # 兼容旧字段 commission_bps
        crate = float(raw.get("commission_bps", 0.0)) / 10000.0
    return PortfolioState(cash=cash, lot_size=lot, sellable=sellable, locked=locked, commission_rate=crate)


def infer_position_mode_from_state_dict(raw: Dict[str, Any]) -> str:
    """
    依据 sellable/locked 推断 empty | holding；若存在 position_status / position_mode
    与推断不一致则在 stderr 打警告（以股数字段为准）。
    """
    explicit = raw.get("position_status") or raw.get("position_mode")
    sellable = raw.get("sellable") or {}
    locked = raw.get("locked") or {}
    has = sum(int(v) for v in sellable.values() if int(v) > 0) + sum(
        int(v) for v in locked.values() if int(v) > 0
    )
    inferred = "empty" if has == 0 else "holding"
    if explicit is None:
        return inferred
    ex = str(explicit).strip().lower()
    if ex in ("empty", "flat", "cash", "未建仓"):
        ex_norm = "empty"
    elif ex in ("holding", "持仓", "已建仓"):
        ex_norm = "holding"
    else:
        return inferred
    if ex_norm != inferred:
        print(
            f"[警告] JSON 中 position_status={explicit!r} 与 sellable/locked 推断的「{inferred}」不一致，以实际持仓字段为准。",
            file=sys.stderr,
        )
    return inferred


def save_portfolio_json(path: str, st: PortfolioState) -> None:
    payload = {
        "cash": float(st.cash),
        "lot_size": int(st.lot_size),
        "sellable": {k: int(v) for k, v in st.sellable.items() if v > 0},
        "locked": {k: int(v) for k, v in st.locked.items() if v > 0},
        "commission_rate": float(st.commission_rate),
        "_comment": "收盘后状态：当日买入在 locked，下一交易日早盘将并入 sellable（脚本推演已模拟解锁）。",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_trade_price_map_for_day(
    data_root: str,
    trade_date: str,
    *,
    price_col: str = "open",
) -> Dict[str, float]:
    fp = Path(data_root) / "daily" / f"{trade_date}.csv"
    if not fp.is_file():
        raise FileNotFoundError(f"缺少行情文件: {fp}")
    df = pd.read_csv(fp, usecols=["ts_code", price_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df[np.isfinite(df[price_col]) & (df[price_col] > 0)]
    return dict(zip(df["ts_code"].astype(str), df[price_col].astype(np.float64)))


def resolve_equity_trade_price_date(
    data_root: str,
    logical_trade_date_yyyymmdd: str,
    *,
    strict: bool = False,
    price_col: str = "open",
) -> Tuple[str, str]:
    """
    将「你希望标注的下一交易日」映射到本地实际用于读取成交价的 CSV 交易日。

    - 存在 daily/{{logical}}.csv → 用当日成交价列。
    - 否则在非 strict 下：取 daily/ 中 **<= logical** 的最近一个交易日的 CSV。
      适用于数据只到今天、但想把「语义上的次日」仍为 YYYYMMDD 的情境（占位成交近似）。
      若请求列为 open，则显式采用“前一可用交易日 close 近似次日 open”。

    返回 (pricing_date_yyyymmdd, stderr_note_or_empty)。
    strict=True → 必须由 logical 当天的 CSV；否则报错。
    """
    d = str(logical_trade_date_yyyymmdd).strip().replace("-", "")
    if len(d) != 8 or not d.isdigit():
        raise ValueError(f"非法交易日 logical_trade_date: {logical_trade_date_yyyymmdd!r}")

    fp = Path(data_root) / "daily" / f"{d}.csv"
    if fp.is_file():
        return d, ""
    if strict:
        raise FileNotFoundError(
            f"--strict-next-trade-csv：要求行情文件存在但缺失：{fp}"
        )

    daily_dir = Path(data_root) / "daily"
    if not daily_dir.is_dir():
        raise FileNotFoundError(f"daily 目录不存在: {daily_dir}")

    dated = sorted(
        p.stem for p in daily_dir.glob("*.csv") if len(p.stem) == 8 and p.stem.isdigit()
    )
    cand = [x for x in dated if x <= d]
    if not cand:
        raise FileNotFoundError(
            f"未找到不晚于 {d} 的 daily/*.csv（数据根目录：{data_root}）。请补行情或改用已有交易日。"
        )
    best = cand[-1]
    if best == d:
        return d, ""
    if str(price_col) == "open":
        note = (
            f"[占位价格] daily/{d}.csv 不存在 → 显式采用「{best} 的 close」近似语义日 {d} 的 open。"
            "（即“次日开盘≈前日收盘”的近似规则）"
        )
    else:
        note = (
            f"[占位价格] daily/{d}.csv 不存在 → 用「不晚于 {d}」的最近行情日 {best} 的 {price_col} 价模拟撮合。"
            "（数据尚未到车时，用你的历史末尾价近似语义上的次日）"
        )
    return best, note


def effective_trade_price_col(
    requested_price_col: str,
    logical_trade_date: str,
    pricing_trade_date: str,
) -> str:
    """
    当请求 open 且语义日 CSV 不存在时，按“次日开盘≈前日收盘”规则回退到 close。
    其余场景保持请求列不变。
    """
    req = str(requested_price_col)
    logical = str(logical_trade_date)
    pricing = str(pricing_trade_date)
    if req == "open" and logical and pricing and logical != pricing:
        return "close"
    return req


class OrderLog:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def sell(self, ts_code: str, shares: int, price: float, phase: str = "") -> None:
        if shares <= 0:
            return
        self.rows.append(
            {
                "side": "卖出",
                "ts_code": ts_code,
                "shares": int(shares),
                "price": float(price),
                "amount": float(shares * price),
                "phase": phase,
            }
        )

    def buy(self, ts_code: str, shares: int, price: float, phase: str = "") -> None:
        if shares <= 0:
            return
        self.rows.append(
            {
                "side": "买入",
                "ts_code": ts_code,
                "shares": int(shares),
                "price": float(price),
                "amount": float(shares * price),
                "phase": phase,
            }
        )


def _mv(px_map: Dict[str, float], sellable: Dict[str, int], locked: Dict[str, int]) -> float:
    mv = 0.0
    for c, sh in sellable.items():
        if sh <= 0:
            continue
        px = px_map.get(c, float("nan"))
        if np.isfinite(px):
            mv += sh * px
    for c, sh in locked.items():
        if sh <= 0:
            continue
        px = px_map.get(c, float("nan"))
        if np.isfinite(px):
            mv += sh * px
    return mv


def pick_picks_df(day_idx: pd.DataFrame, n: int, k: int) -> pd.DataFrame:
    """候选池 Top-n，持仓 Top-k（分数），与 backtest 一致。"""
    picks_df, _ = target_weights_from_panel(day_idx, n, k)
    return picks_df


def build_target_weight_rows(
    picks_df: pd.DataFrame,
    wmap: Dict[str, float],
    budget_cash: float,
) -> List[Dict[str, Any]]:
    """生成含 target_weight、target_amount（元）的目标持仓行。"""
    budget = max(float(budget_cash), 0.0)
    rows: List[Dict[str, Any]] = []
    for _, r in picks_df.iterrows():
        code = str(r["ts_code"])
        w = float(wmap.get(code, 0.0))
        rows.append(
            {
                "ts_code": code,
                "pred_score": float(r["pred_score"]),
                "target_weight": w,
                "target_amount": round(budget * w, 2),
            }
        )
    return rows


def planned_buy_rows_for_rotation(
    picks_df: pd.DataFrame,
    scores_map: Dict[str, float],
    sellable: Dict[str, int],
    locked: Dict[str, int],
    n: int,
    k: int,
    buy_budget_cash: float,
) -> List[Dict[str, Any]]:
    """有持仓时，按 trim + rotate-k 规则预估次日买入标的及参考金额（不含价格）。"""
    if not portfolio_has_holdings(sellable, locked):
        return []
    held = {c for c, sh in sellable.items() if sh > 0} | {c for c, sh in locked.items() if sh > 0}
    target_codes = [str(c) for c in picks_df["ts_code"].astype(str).tolist()]
    target_set = set(target_codes)
    out_of_pool = {c for c in held if c not in target_set}
    held_in_pool = held - out_of_pool
    trim_sold = len(out_of_pool)
    rotate_sold = min(max(0, int(k)), len(held_in_pool))
    held_after = len(held) - trim_sold - rotate_sold
    slots_to_n = max(0, int(n) - held_after)
    buy_names = min(slots_to_n, trim_sold + rotate_sold, len(target_codes))
    buy_pool = [
        c
        for c in sorted(
            target_codes,
            key=lambda c: (-scores_map.get(c, -np.inf), c),
        )
        if c not in held
    ]
    buy_codes = buy_pool[:buy_names]
    if not buy_codes or buy_budget_cash <= 1e-9:
        return []
    picks_buy = picks_df[picks_df["ts_code"].astype(str).isin(set(buy_codes))].copy()
    w_buy = score_weights_from_picks_df(picks_buy)
    budget = float(buy_budget_cash)
    out: List[Dict[str, Any]] = []
    for _, r in picks_buy.iterrows():
        code = str(r["ts_code"])
        w = float(w_buy.get(code, 0.0))
        out.append(
            {
                "ts_code": code,
                "pred_score": float(r["pred_score"]),
                "target_weight": w,
                "target_amount": round(budget * w, 2),
            }
        )
    return sorted(out, key=lambda x: (-x["pred_score"], x["ts_code"]))


def enrich_orders_with_targets(
    order_rows: List[Dict[str, Any]],
    full_weight_map: Dict[str, float],
    full_amount_map: Dict[str, float],
    buy_only_weight_map: Optional[Dict[str, float]] = None,
    buy_only_amount_map: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """为撮合指令补充目标权重与参考/目标金额。"""
    out: List[Dict[str, Any]] = []
    for row in order_rows:
        code = str(row["ts_code"])
        side = str(row["side"])
        item = dict(row)
        if side == "买入":
            bw = (buy_only_weight_map or {}).get(code)
            ba = (buy_only_amount_map or {}).get(code)
            if bw is not None:
                item["target_weight"] = float(bw)
            else:
                item["target_weight"] = float(full_weight_map.get(code, 0.0))
            if ba is not None:
                item["target_amount"] = round(float(ba), 2)
            else:
                item["target_amount"] = round(float(full_amount_map.get(code, 0.0)), 2)
        else:
            item["target_weight"] = float(full_weight_map.get(code, 0.0))
            item["target_amount"] = None
        item["amount"] = round(float(item.get("amount", 0.0)), 2)
        out.append(item)
    return out


def consolidate_orders_by_code(order_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """同一标的、同一方向的多笔整手委托合并为一行。"""
    if not order_rows:
        return []
    df = pd.DataFrame(order_rows)
    if df.empty:
        return []
    agg: Dict[str, Any] = {
        "shares": "sum",
        "amount": "sum",
    }
    for col in ("target_weight", "target_amount", "price", "phase"):
        if col in df.columns:
            agg[col] = "first"
    grouped = (
        df.groupby(["side", "ts_code"], as_index=False)
        .agg(agg)
        .sort_values(["side", "ts_code"])
        .reset_index(drop=True)
    )
    grouped["amount"] = grouped["amount"].astype(float).round(2)
    grouped["price"] = (grouped["amount"] / grouped["shares"].replace(0, np.nan)).astype(float)
    grouped.loc[~np.isfinite(grouped["price"]), "price"] = np.nan
    return grouped.to_dict(orient="records")


def build_target_weights_plan(
    panel: pd.DataFrame,
    n: int,
    k: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """仅基于 pred_score 生成 Top-n 目标权重（不含价格、不含股数）。"""
    day_idx = panel.drop_duplicates(subset=["ts_code"]).reset_index(drop=True)
    return target_weights_from_panel(day_idx, n, k)


def load_target_weights_plan(path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("target_weights") or raw.get("targets") or []
    if not rows:
        raise ValueError(f"计划文件缺少 target_weights: {path}")
    weights: Dict[str, float] = {}
    for row in rows:
        code = str(row["ts_code"])
        w = float(row.get("target_weight", row.get("weight", 0.0)))
        if w > 0:
            weights[code] = w
    if not weights:
        raise ValueError(f"计划文件 target_weights 为空: {path}")
    tot = sum(weights.values())
    if tot <= 1e-18:
        raise ValueError("target_weights 之和无效")
    weights = {c: w / tot for c, w in weights.items()}
    return weights, raw


def execute_rotation_at_open(
    picks_df: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
    *,
    full_weight_map: Optional[Dict[str, float]] = None,
    budget_cash: Optional[float] = None,
) -> Tuple[OrderLog, PortfolioState, float, float, str, List[Dict[str, Any]]]:
    """在开盘价下执行 Top-n 持仓 + 日度换 k 逻辑。"""
    log = OrderLog()
    wmap = full_weight_map or score_weights_from_picks_df(picks_df)
    budget = float(st.cash if budget_cash is None else budget_cash)
    full_amount_map = {c: budget * w for c, w in wmap.items()}

    sellable, locked, cash, _ts, _tb, order_rows = daily_rotation_trades(
        px_map,
        picks_df,
        scores_map,
        st.sellable,
        st.locked,
        st.cash,
        st.lot_size,
        commission_rate,
        n,
        k,
    )

    buy_only_amount_map: Dict[str, float] = {}
    buy_only_weight_map: Dict[str, float] = {}
    if portfolio_has_holdings(st.sellable, st.locked):
        buy_rows = planned_buy_rows_for_rotation(
            picks_df, scores_map, st.sellable, st.locked, n, k, cash if cash > 0 else budget
        )
        for row in buy_rows:
            code = str(row["ts_code"])
            buy_only_weight_map[code] = float(row["target_weight"])
            buy_only_amount_map[code] = float(row["target_amount"])

    enriched = enrich_orders_with_targets(
        order_rows,
        wmap,
        full_amount_map,
        buy_only_weight_map=buy_only_weight_map or None,
        buy_only_amount_map=buy_only_amount_map or None,
    )
    enriched = consolidate_orders_by_code(enriched)

    fee_day = 0.0
    for row in enriched:
        amt = float(row["amount"])
        sh = int(row["shares"])
        code = str(row["ts_code"])
        if row["side"] == "卖出":
            fee_day += fees_on_sell_turnover(code, amt, sh, commission_rate)
            log.sell(code, sh, float(row["price"]), str(row.get("phase", "")))
        else:
            fee_day += fees_on_buy_turnover(code, amt, sh, commission_rate)
            log.buy(code, sh, float(row["price"]), str(row.get("phase", "")))

    sellable_f = {c: int(sh) for c, sh in sellable.items() if sh > 0}
    locked_f = {c: int(sh) for c, sh in locked.items() if sh > 0}
    nav_after = cash + _portfolio_mv_at_px(px_map, sellable_f, locked_f)
    ps = PortfolioState(
        cash=cash,
        lot_size=st.lot_size,
        sellable=sellable_f,
        locked=locked_f,
        commission_rate=commission_rate,
    )
    if portfolio_has_holdings(st.sellable, st.locked):
        note = f"execute-at-open：持有 Top-{n}，卖出低分 {k} 只、买入高分 {k} 只"
    else:
        note = f"execute-at-open：空仓初始建仓 Top-{n}（忽略 k={k}）"
    return log, ps, fee_day, nav_after, note, enriched


def simulate_score_weighted_day(
    panel: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
) -> Tuple[OrderLog, PortfolioState, float, float, str]:
    """与 run_backtest 一致：Top-n 持仓，有仓日度换 k，空仓建 n。"""
    picks_df, _ = build_target_weights_plan(panel, n, k)
    if picks_df.empty:
        nav = float(st.cash) + _portfolio_mv_at_px(px_map, st.sellable, st.locked)
        ps = PortfolioState(st.cash, st.lot_size, dict(st.sellable), dict(st.locked), st.commission_rate)
        return OrderLog(), ps, 0.0, nav, "无候选标的"
    return execute_rotation_at_open(picks_df, px_map, scores_map, st, n, k, commission_rate)[:5]


def run_simulation(
    panel: pd.DataFrame,
    px_map: Dict[str, float],
    scores_map: Dict[str, float],
    st: PortfolioState,
    n: int,
    k: int,
    commission_rate: float,
) -> Tuple[OrderLog, PortfolioState, float, float, str]:
    """commission_rate 为券商佣金费率小数（万二=0.0002）；另含卖出印花税 0.1% 与仅上证 60* 双向过户费（按成交股数、单笔最低 1 元）。"""
    return simulate_score_weighted_day(panel, px_map, scores_map, st, n, k, commission_rate)



def load_holdings_csv(path: str) -> Dict[str, int]:
    df = pd.read_csv(path)
    if "ts_code" not in df.columns or "sellable_shares" not in df.columns:
        raise ValueError("持仓 CSV 至少需要列: ts_code, sellable_shares")
    out: Dict[str, int] = {}
    for _, r in df.iterrows():
        code = str(r["ts_code"])
        sh = int(pd.to_numeric(r["sellable_shares"], errors="coerce") or 0)
        if sh > 0:
            out[code] = sh
    return out


def infer_next_trade_date_from_daily(data_root: str, scores_last: str) -> Optional[str]:
    daily = Path(data_root) / "daily"
    if not daily.is_dir():
        return None
    names = sorted(p.stem for p in daily.glob("*.csv") if p.stem.isdigit() and len(p.stem) == 8)
    for d in names:
        if d > scores_last:
            return d
    return None


def codes_tradable_next_day(
    data_root: str,
    next_d: str,
    *,
    price_col: str = "open",
) -> Optional[Set[str]]:
    fp = Path(data_root) / "daily" / f"{next_d}.csv"
    if not fp.is_file():
        return None
    df = pd.read_csv(fp, usecols=["ts_code", price_col])
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=[price_col])
    df = df[np.isfinite(df[price_col]) & (df[price_col] > 0)]
    return set(df["ts_code"].astype(str))


def score_snapshot_date_for_day(
    sorted_score_dates: List[str],
    trade_day: str,
    score_lag: int,
) -> Tuple[str, str]:
    if not sorted_score_dates:
        raise ValueError("打分 CSV trade_date 列表为空")

    scores_last = sorted_score_dates[-1]

    if score_lag <= 0:
        if trade_day and trade_day in sorted_score_dates:
            return trade_day, "lag=0：使用当日截面（trade_date 与交易日对齐）"
        return scores_last, "lag=0：使用打分 CSV 中最后一个 trade_date 截面"

    # 语义上的执行日晚于打分文件中最晚截面：仅用历史已产生的 pred_score（典型：末日锚定 20260520 → 推演 20260521）
    if trade_day and len(trade_day) == 8 and trade_day.isdigit() and trade_day > scores_last:
        idx = max(0, len(sorted_score_dates) - score_lag)
        snap = sorted_score_dates[idx]
        return snap, (
            f"语义执行日 {trade_day} 晚于打分最后截面 {scores_last}；"
            f"lag={score_lag} → 使用 trade_date={snap}（与 predict-next 末日推理对齐）"
        )

    if trade_day and trade_day in sorted_score_dates:
        di = sorted_score_dates.index(trade_day)
        idx = max(0, di - score_lag)
        snap = sorted_score_dates[idx]
        return snap, f"lag={score_lag}：执行日在打分枚举内 → trade_date={snap}"

    di = len(sorted_score_dates)
    idx = max(0, di - score_lag)
    snap = sorted_score_dates[idx]
    label = trade_day if trade_day else "（未指定 / 推断）"
    return snap, (
        f"执行日线索 {label} 与打分 trade_date 未对齐 → lag={score_lag}，使用 trade_date={snap}"
    )


def print_advisory_summary(
    *,
    next_d: str,
    score_snap: str,
    snap_note: str,
    panel: pd.DataFrame,
    n: int,
    k: int,
    top_df: pd.DataFrame,
) -> None:
    print("=== 交易日与分数快照 ===")
    print(f"下一交易日（推断/指定）: {next_d}")
    print(f"score-lag → 使用的打分快照 trade_date = {score_snap}")
    print(f"说明: {snap_note}")
    print(f"快照股票数（去重后）: {len(panel)}")
    print(f"\n=== 目标持仓 Top-{n}；有仓时日换 {k} 只；权重 ∝ pred_score（Top-n 内归一）===")
    print("\n--- 目标持仓（代码 / 分数 / 目标权重）---")
    cols = [c for c in ("ts_code", "pred_score", "target_weight") if c in top_df.columns]
    print(top_df[cols].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下一交易日：plan 仅输出目标权重；execute 在开盘价下换算股数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "典型流程：\n"
            "  1) 盘后 plan：python predict.py --mode plan --scores ... --next-trade-date ...\n"
            "  2) 开盘 execute：python predict.py --mode execute --plan plan.json --state state.json "
            "--next-trade-date ... --strict-next-trade-csv\n"
            "未建仓 state 示例: examples/state_empty.json"
        ),
    )
    parser.add_argument("--mode", choices=("plan", "execute", "legacy"), default="plan")
    parser.add_argument("--scores", default="", help="plan / legacy 模式必填")
    parser.add_argument("--plan", default="", help="execute 模式：plan JSON（含 target_weights）")
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DL_DATA_ROOT", ""),
    )
    parser.add_argument("--next-trade-date", default="", help="下一交易日 YYYYMMDD")
    parser.add_argument("--n", type=int, default=20, help="同一时段目标持有股票只数 Top-n")
    parser.add_argument("--k", type=int, default=4, help="有持仓时日度换仓只数（空仓建仓忽略 k）")
    parser.add_argument("--score-lag", type=int, default=1)
    parser.add_argument("--lot-size", type=int, default=None, help="不传则用 state 或默认 100")
    parser.add_argument("--holdings", default="", help="execute：可在 state 空仓时合并简易持仓 CSV")
    parser.add_argument(
        "--state",
        default="",
        help="plan / execute / legacy：portfolio_state.json（plan 用于计算 budget_cash 与 target_amount）",
    )
    parser.add_argument(
        "--strict-next-trade-csv",
        action="store_true",
        help="必须为当日生成 daily/{{--next-trade-date}}.csv；execute 模式强烈建议开启",
    )
    parser.add_argument(
        "--trade-price-col",
        choices=("open", "close"),
        default="open",
        help="execute / legacy：撮合价格列（默认 open）",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=None,
        help="覆盖 state JSON 内 commission_rate（券商费率，小数）；不传则用 state",
    )
    parser.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="兼容旧参数：基点制（万三=3），若提供则覆盖 --commission-rate",
    )
    parser.add_argument("--out-plan", default="", help="plan 模式：写出目标权重 JSON")
    parser.add_argument("--out-csv", default="", help="plan 模式：写出目标权重 CSV")
    parser.add_argument("--out-orders", default="", help="execute / legacy：写出指令明细 CSV")
    parser.add_argument("--out-next-state", default="", help="execute / legacy：写出推演收盘后状态 JSON")
    args = parser.parse_args()

    args.data_root = resolve_data_root(args.data_root)

    if args.mode == "plan":
        if not args.scores:
            raise SystemExit("plan 模式需要 --scores")
        scores = load_scores(args.scores)
        dates = sorted(scores["trade_date"].unique())
        if not dates:
            raise SystemExit("scores 为空")

        last_s = dates[-1]
        next_d = args.next_trade_date.strip().replace("-", "")
        if not next_d:
            next_d = infer_next_trade_date_from_daily(args.data_root, last_s) or ""
        if not (next_d.isdigit() and len(next_d) == 8):
            raise SystemExit("plan 模式需要有效的 --next-trade-date YYYYMMDD（或确保 daily/ 可推断）")

        score_snap, snap_note = score_snapshot_date_for_day(dates, next_d, args.score_lag)
        panel = scores[scores["trade_date"] == score_snap].drop_duplicates(subset=["ts_code"]).copy()
        panel = panel.sort_values("pred_score", ascending=False).reset_index(drop=True)
        picks_df, wmap = build_target_weights_plan(panel, args.n, args.k)
        budget_cash = 0.0
        sellable: Dict[str, int] = {}
        locked: Dict[str, int] = {}
        if args.state:
            st_plan = load_portfolio_json(args.state)
            budget_cash = float(st_plan.cash)
            sellable = dict(st_plan.sellable)
            locked = dict(st_plan.locked)

        target_rows = build_target_weight_rows(picks_df, wmap, budget_cash)
        planned_buys = planned_buy_rows_for_rotation(
            picks_df,
            panel.set_index("ts_code")["pred_score"].astype(float).to_dict(),
            sellable,
            locked,
            args.n,
            args.k,
            budget_cash,
        )
        top_df = pd.DataFrame(target_rows)

        print("=== plan：目标权重与参考购买金额（不含股数 / 不含占位价格） ===")
        print(f"下一交易日: {next_d}")
        print(f"打分快照 trade_date = {score_snap}")
        print(f"说明: {snap_note}")
        print(f"预算现金 budget_cash = {budget_cash:.2f} 元")
        print(f"目标持仓 Top-{args.n}；有仓时日换 {args.k} 只；权重 ∝ pred_score（Top-n 内归一）")
        print("\n--- 目标持仓（权重 / 参考金额）---")
        print(top_df[["ts_code", "pred_score", "target_weight", "target_amount"]].to_string(index=False))
        if planned_buys:
            print(f"\n--- 预估买入 Top-{args.k}（按 rotate-k，参考金额=预算现金×买入权重）---")
            print(pd.DataFrame(planned_buys)[["ts_code", "pred_score", "target_weight", "target_amount"]].to_string(index=False))
        print("\n提示：target_amount 为参考金额；execute 阶段按 open 价换算整手股数。")

        plan_payload = {
            "workflow": "plan-target-weights",
            "next_trade_date": next_d,
            "score_snapshot_trade_date": score_snap,
            "score_snapshot_note": snap_note,
            "score_lag": int(args.score_lag),
            "hold_n": int(args.n),
            "rotate_k": int(args.k),
            "budget_cash": round(budget_cash, 2),
            "target_weights": target_rows,
            "planned_buys": planned_buys,
            "artifacts": {"scores_csv": args.scores},
            "notes": (
                "target_weights：Top-n 目标权重与参考购买金额（target_amount=budget_cash×target_weight）。"
                "planned_buys：有持仓时按 rotate-k 预估的次日买入标的及参考金额。"
                "execute-next 在 open 价下生成实际 orders（含 target_weight / target_amount / amount）。"
            ),
        }
        if args.out_plan:
            outp = Path(args.out_plan)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, "w", encoding="utf-8") as f:
                json.dump(plan_payload, f, ensure_ascii=False, indent=2)
            print(f"\n已写 plan JSON: {outp}")
        if args.out_csv:
            outp = Path(args.out_csv)
            outp.parent.mkdir(parents=True, exist_ok=True)
            top_df.assign(next_trade_date=str(next_d), score_snapshot_date=score_snap).to_csv(outp, index=False)
            print(f"已写 plan CSV: {outp}")
        return

    if args.mode == "execute":
        if not args.plan:
            raise SystemExit("execute 模式需要 --plan")
        if not args.state:
            raise SystemExit("execute 模式需要 --state")
        next_d = args.next_trade_date.strip().replace("-", "")
        if not (next_d.isdigit() and len(next_d) == 8):
            raise SystemExit("execute 模式需要有效的 --next-trade-date YYYYMMDD")

        target_weights, plan_raw = load_target_weights_plan(args.plan)
        hold_n = int(plan_raw.get("hold_n", plan_raw.get("candidate_pool_n", args.n)))
        rotate_k = int(plan_raw.get("rotate_k", plan_raw.get("hold_top_k", args.k)))
        rows = plan_raw.get("target_weights") or []
        picks_df = pd.DataFrame(rows)
        if picks_df.empty:
            raise SystemExit("plan 中 target_weights 为空")
        if "pred_score" not in picks_df.columns:
            raise SystemExit("plan.target_weights 缺少 pred_score")
        scores_map = picks_df.set_index("ts_code")["pred_score"].astype(float).to_dict()

        plan_trade_date = str(plan_raw.get("next_trade_date", "")).strip().replace("-", "")
        if plan_trade_date and plan_trade_date != next_d:
            print(
                f"[警告] plan.next_trade_date={plan_trade_date} 与 --next-trade-date={next_d} 不一致，"
                "以命令行 execute 日为准。",
                file=sys.stderr,
            )

        px_date, px_note = resolve_equity_trade_price_date(
            args.data_root,
            next_d,
            strict=True if args.strict_next_trade_csv else False,
            price_col=str(args.trade_price_col),
        )
        if args.strict_next_trade_csv and px_date != next_d:
            raise SystemExit(
                f"--strict-next-trade-csv：要求 {next_d} 当天 CSV，但实际映射到 {px_date}。"
            )
        price_col_used = effective_trade_price_col(str(args.trade_price_col), str(next_d), str(px_date))
        if px_note:
            print(px_note, flush=True)
        if price_col_used != args.trade_price_col:
            print(
                f"[警告] 请求列={args.trade_price_col}，实际使用={price_col_used}。"
                "execute 模式建议使用 --strict-next-trade-csv 且当日 CSV 已落盘。",
                file=sys.stderr,
            )

        px_map = load_trade_price_map_for_day(args.data_root, px_date, price_col=str(price_col_used))
        st = load_portfolio_json(args.state)
        if args.lot_size is not None:
            st.lot_size = int(args.lot_size)
        if args.holdings:
            extra = load_holdings_csv(args.holdings)
            for c, sh in extra.items():
                st.sellable[c] = st.sellable.get(c, 0) + sh

        if args.commission_bps is not None:
            crate = float(args.commission_bps) / 10000.0
        elif args.commission_rate is not None:
            crate = float(args.commission_rate)
        else:
            crate = float(st.commission_rate)

        log, ps_end, fee, nav_after, sim_note, enriched_orders = execute_rotation_at_open(
            picks_df, px_map, scores_map, st, hold_n, rotate_k, crate
        )

        print("=== execute-at-open：Top-n 持仓 + 日度换 k ===")
        print(f"执行交易日: {next_d}  |  成交价 CSV 日: {px_date}  |  价格列: {price_col_used}")
        print(f"hold_n={hold_n}  rotate_k={rotate_k}")
        print(f"推演说明: {sim_note}")
        print(f"交易费用估算合计 ≈ {fee:.2f} 元；推演净值 ≈ {nav_after:.2f} 元；现金 ≈ {ps_end.cash:.2f} 元")
        print("\n--- 指令明细（含目标权重 / 目标金额 / 实际金额）---")
        if not enriched_orders:
            print("（无成交指令）")
        else:
            od = pd.DataFrame(enriched_orders)
            cols = ["side", "ts_code", "shares", "price", "target_weight", "target_amount", "amount", "phase"]
            print(od[[c for c in cols if c in od.columns]].to_string(index=False))

        if args.out_orders:
            Path(args.out_orders).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(enriched_orders).to_csv(args.out_orders, index=False)
            print(f"\n已写指令: {args.out_orders}")
        if args.out_next_state:
            save_portfolio_json(
                args.out_next_state,
                PortfolioState(ps_end.cash, ps_end.lot_size, dict(ps_end.sellable), dict(ps_end.locked), crate),
            )
            print(f"已写下一状态: {args.out_next_state}")
        return

    # ---------- legacy：plan+execute 一步（兼容旧流程；无当日 CSV 时可能用占位价） ----------
    if not args.scores:
        raise SystemExit("legacy 模式需要 --scores")
    scores = load_scores(args.scores)
    dates = sorted(scores["trade_date"].unique())
    if not dates:
        raise SystemExit("scores 为空")

    last_s = dates[-1]
    next_d = args.next_trade_date.strip().replace("-", "")
    if not next_d:
        next_d = infer_next_trade_date_from_daily(args.data_root, last_s) or ""

    trade_day_for_snap = next_d if (next_d.isdigit() and len(next_d) == 8) else ""
    score_snap, snap_note = score_snapshot_date_for_day(dates, trade_day_for_snap, args.score_lag)

    panel = scores[scores["trade_date"] == score_snap].drop_duplicates(subset=["ts_code"]).copy()
    panel = panel.sort_values("pred_score", ascending=False).reset_index(drop=True)
    scores_map = panel.set_index("ts_code")["pred_score"].to_dict()

    if args.state:
        if not (next_d.isdigit() and len(next_d) == 8):
            raise SystemExit("细单模式需要有效的下一交易日 YYYYMMDD（请传 --next-trade-date 或确保 daily/ 可推断）")

        px_date, px_note = resolve_equity_trade_price_date(
            args.data_root,
            next_d,
            strict=args.strict_next_trade_csv,
            price_col=str(args.trade_price_col),
        )
        price_col_used = effective_trade_price_col(
            str(args.trade_price_col),
            str(next_d),
            str(px_date),
        )
        if px_note:
            print(px_note, flush=True)
        px_map = load_trade_price_map_for_day(
            args.data_root,
            px_date,
            price_col=str(price_col_used),
        )
        tradable = set(px_map.keys())
        panel = panel[panel["ts_code"].astype(str).isin(tradable)].reset_index(drop=True)
        scores_map = panel.set_index("ts_code")["pred_score"].to_dict()

        st = load_portfolio_json(args.state)
        if args.lot_size is not None:
            st.lot_size = int(args.lot_size)
        if args.holdings:
            extra = load_holdings_csv(args.holdings)
            for c, sh in extra.items():
                st.sellable[c] = st.sellable.get(c, 0) + sh

        if args.commission_bps is not None:
            crate = float(args.commission_bps) / 10000.0
        elif args.commission_rate is not None:
            crate = float(args.commission_rate)
        else:
            crate = float(st.commission_rate)

        log, ps_end, fee, nav_after, sim_note = run_simulation(
            panel,
            px_map,
            scores_map,
            st,
            args.n,
            args.k,
            crate,
        )

        print("=== 细单模式：下一交易日买卖指令（与回测规则对齐） ===")
        print(
            f"语义下一交易日: {next_d}  |  用于成交价的 CSV 交易日: {px_date}"
            + ("（与语义日相同）" if px_date == next_d else "（价格占位说明见上文）")
        )
        print(f"成交价列: 请求={args.trade_price_col}；实际使用={price_col_used}")
        print(f"使用打分快照: {score_snap}  |  {snap_note}")
        print(f"推演说明: {sim_note}")
        print(
            f"交易费用（印花税+过户+券商佣金）：commission_rate={crate}（仅券商项）"
            f"  →  当日估算合计 ≈ {fee:.2f} 元"
        )
        print(
            f"推演净值（按 {price_col_used} 价计价）≈ {nav_after:.2f} 元；"
            f"现金余额 ≈ {ps_end.cash:.2f} 元"
        )
        print("\n--- 指令明细（买入当日计入 locked，次日才可卖）---")
        if not log.rows:
            print("（无成交指令）")
        else:
            od = pd.DataFrame(log.rows)
            print(od.to_string(index=False))

        print("\n--- 收盘后账户状态（可写入 --out-next-state；次日加载时会自动早盘解锁）---")
        print(f"cash: {ps_end.cash:.6g}")
        print(f"sellable: {ps_end.sellable}")
        print(f"locked: {ps_end.locked}")

        if args.out_orders:
            outp = Path(args.out_orders)
            outp.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(log.rows).to_csv(outp, index=False)
            print(f"\n已写指令: {outp}")

        if args.out_next_state:
            ps_save = PortfolioState(
                cash=ps_end.cash,
                lot_size=ps_end.lot_size,
                sellable=dict(ps_end.sellable),
                locked=dict(ps_end.locked),
                commission_rate=crate,
            )
            save_portfolio_json(args.out_next_state, ps_save)
            print(f"已写下一状态: {args.out_next_state}")
            print(
                "提示：此为当日收盘后状态（新买入在 locked）。次日再用同一脚本加载时会自动先做 "
                "T+1 解锁（与回测一致），无需手工合并。"
            )

        return

    # ---------- 摘要模式 ----------
    if not next_d:
        print("=== 提示 ===")
        print(f"打分 CSV 中最后截面日 trade_date = {last_s}")
        print("未在 daily/ 中找到更晚交易日；请手动传入 --next-trade-date")
        next_d = "（请指定下一交易日）"

    tradable: Optional[Set[str]] = None
    if next_d.isdigit() and len(next_d) == 8:
        tradable_set: Optional[Set[str]] = None
        try:
            px_date, px_note = resolve_equity_trade_price_date(
                args.data_root,
                next_d,
                strict=args.strict_next_trade_csv,
                price_col=str(args.trade_price_col),
            )
            price_col_used = effective_trade_price_col(
                str(args.trade_price_col),
                str(next_d),
                str(px_date),
            )
            if px_note:
                print(px_note, flush=True)
            tradable_set = codes_tradable_next_day(
                args.data_root,
                px_date,
                price_col=str(price_col_used),
            )
        except FileNotFoundError as e:
            raise SystemExit(str(e))
        if tradable_set is not None:
            tradable = tradable_set
            panel = panel[panel["ts_code"].astype(str).isin(tradable)].reset_index(drop=True)
            scores_map = panel.set_index("ts_code")["pred_score"].to_dict()

    picks_df = pick_picks_df(panel, args.n, args.k)
    wmap = score_weights_from_picks_df(picks_df)
    top_df = picks_df.copy()
    top_df["target_weight"] = top_df["ts_code"].astype(str).map(lambda c: wmap.get(c, float("nan")))

    print_advisory_summary(
        next_d=str(next_d),
        score_snap=score_snap,
        snap_note=snap_note,
        panel=panel,
        n=args.n,
        k=args.k,
        top_df=top_df,
    )

    if args.out_csv:
        outp = Path(args.out_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        top_df.assign(next_trade_date=str(next_d), score_snapshot_date=score_snap, strategy="score_weighted").to_csv(
            outp, index=False
        )
        print(f"\n已写: {outp}")


if __name__ == "__main__":
    main()
