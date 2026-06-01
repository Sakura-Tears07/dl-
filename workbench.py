#!/usr/bin/env python3
"""
工作台统一入口：`data_preprocess` / `train` / `backtest` / `predict` 分列模块，本脚本负责编排子进程。

1) backtest：训练（train-end）→ 验证段打分 → 历史回测  
2) predict-next：末日推理打分 → 读持仓 JSON → 写出下一步操作 JSON

详见 README「workbench」小节。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from data_preprocess import (
    fmt_yyyy_mm_dd,
    load_open_trading_days,
    next_trading_day_strictly_after,
    normalize_trade_calendar_key,
    resolve_data_root,
)
from predict import (
    effective_trade_price_col,
    infer_position_mode_from_state_dict,
    infer_next_trade_date_from_daily,
    resolve_equity_trade_price_date,
)

_DL_DIR = Path(__file__).resolve().parent

_PASS_SKIP_TRAIN = "--skip-train"


def _train_command(launcher: str, nproc: int, inner: List[str]) -> List[str]:
    """拼装训练进程：torchrun 或 python + train.py。"""
    script_path = str(_DL_DIR / "train.py")
    if launcher == "torchrun":
        return ["torchrun", "--standalone", f"--nproc_per_node={nproc}", script_path, *inner]
    return [sys.executable, script_path, *inner]


def _sanitize_pass_through_train_argv(train_argv: List[str]) -> Tuple[List[str], bool]:
    """
    去掉误传给 train.py 的 --skip-train，并修复常见粘连写法
    「all--skip-train」（实为 --stock-pool all + 本应写在「--」前的 --skip-train）。
    返回（清洗后的 argv，是否检测到 skip-train 意图）。
    """
    want_skip = False
    out: List[str] = []
    for tok in train_argv:
        if tok == _PASS_SKIP_TRAIN:
            want_skip = True
            continue
        if _PASS_SKIP_TRAIN in tok and not tok.startswith("-"):
            head, sep, tail = tok.partition(_PASS_SKIP_TRAIN)
            if sep and tail.strip():
                raise SystemExit(
                    f"无法理解参数粘连: {tok!r}。"
                    f"请将 {_PASS_SKIP_TRAIN} 单独写在工作台的选项里，且置于「传给 train.py 的 --」之前。"
                )
            head_st = head.rstrip("-").strip()
            want_skip = True
            if head_st:
                out.append(head_st)
            continue
        out.append(tok)
    return out, want_skip


def _fmt_input_date(s: str) -> str:
    """训练脚本使用 YYYY-MM-DD。"""
    return fmt_yyyy_mm_dd(s)


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _infer_next_trade_date_from_scores_and_daily(data_root: str, scores_path: str) -> str:
    from backtest import load_scores

    scores = load_scores(scores_path)
    dates = sorted(scores["trade_date"].unique())
    if not dates:
        raise SystemExit("scores CSV 为空")
    last_s = dates[-1]
    next_d = infer_next_trade_date_from_daily(data_root, last_s)
    if not next_d:
        raise SystemExit(
            "无法从 daily/ 推断下一交易日：请显式传入 --next-trade-date YYYYMMDD。"
        )
    return next_d


def _get_trading_dates_between(data_root: str, start: str, end: str) -> List[str]:
    """获取两个日期之间的所有交易日列表（YYYYMMDD格式，升序）。"""
    open_days = load_open_trading_days(data_root)
    start_norm = normalize_trade_calendar_key(start)
    end_norm = normalize_trade_calendar_key(end)
    dates = [d for d in open_days if start_norm <= d <= end_norm]
    return sorted(dates)


def _run_single_day_training(
    data_root: str,
    train_start: str,
    train_end: str,
    val_date: str,
    scores_out: str,
    launcher: str,
    nproc: int,
    train_argv: List[str],
) -> None:
    """为单个交易日训练模型并导出该日的打分。"""
    inner: List[str] = [
        "--workflow",
        "backtest",
        "--data-root",
        data_root,
        "--train-start",
        _fmt_input_date(train_start),
        "--train-end",
        _fmt_input_date(train_end),
        "--val-start",
        fmt_yyyy_mm_dd(val_date),
        "--val-end",
        fmt_yyyy_mm_dd(val_date),
        "--export-scores",
        scores_out,
        *train_argv,
    ]
    train_cmd = _train_command(launcher, nproc, inner)
    print(f"[workbench] 训练 ({train_end} → {val_date}):", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True)


def _run_single_day_backtest(
    scores_path: str,
    data_root: str,
    out_curve: str,
    out_summary: str,
    cash: float,
    n_pool: int,
    k_hold: int,
    score_lag: int,
    trade_price_col: str,
    commission_rate: float,
    no_benchmark: bool,
    benchmark: str = "000300.SH.csv",
    no_panel_metrics: bool = False,
    panel_metrics_all_score_dates: bool = False,
    min_names_panel_ic: int = 10,
    out_panel_daily_ic: str = "",
) -> None:
    """为单个交易日运行回测。"""
    bt_cmd = [
        sys.executable,
        str(_DL_DIR / "backtest.py"),
        "--scores",
        scores_path,
        "--data-root",
        data_root,
        "--out-curve",
        out_curve,
        "--out-summary",
        out_summary,
        "--cash",
        str(cash),
        "--n",
        str(n_pool),
        "--k",
        str(k_hold),
        "--score-lag",
        str(score_lag),
        "--trade-price-col",
        str(trade_price_col),
        "--commission-rate",
        str(commission_rate),
        "--benchmark",
        str(benchmark),
        "--min-names-panel-ic",
        str(min_names_panel_ic),
    ]
    if no_benchmark:
        bt_cmd.append("--no-benchmark")
    if no_panel_metrics:
        bt_cmd.append("--no-panel-metrics")
    if panel_metrics_all_score_dates:
        bt_cmd.append("--panel-metrics-all-score-dates")
    if out_panel_daily_ic:
        bt_cmd.extend(["--out-panel-daily-ic", out_panel_daily_ic])
    print("[workbench] 回测:", " ".join(bt_cmd), flush=True)
    subprocess.run(bt_cmd, check=True)


def cmd_backtest(ns: argparse.Namespace, train_argv: List[str]) -> None:
    data_root = ns.data_root
    train_end_norm = normalize_trade_calendar_key(ns.train_end)
    backtest_end_norm = normalize_trade_calendar_key(ns.backtest_end)

    vs_raw = (getattr(ns, "val_start", "") or "").strip()
    if vs_raw:
        val_start_d = normalize_trade_calendar_key(vs_raw)
        if train_end_norm >= val_start_d:
            raise SystemExit(
                f"因果约束：`train-end` ({train_end_norm}) 必须严格早于 `--val-start` ({val_start_d})。"
            )
    else:
        val_start_d = next_trading_day_strictly_after(train_end_norm, data_root)

    if val_start_d > backtest_end_norm:
        raise SystemExit(
            f"`--backtest-end` ({backtest_end_norm}) 不能早于验证起点 ({val_start_d})；"
            "请核对 train-end / val-start（若手写）区间。"
        )

    # Walk-forward 模式：逐日重训（先逐日产出 scores，再做一次连续回测）
    if getattr(ns, "walk_forward", False):
        print(
            f"[workbench] Walk-forward 模式：逐日重训，区间 {fmt_yyyy_mm_dd(val_start_d)} .. {fmt_yyyy_mm_dd(backtest_end_norm)}",
            flush=True,
        )
        trading_dates = _get_trading_dates_between(data_root, val_start_d, backtest_end_norm)
        if not trading_dates:
            raise SystemExit(f"区间内无交易日：{val_start_d} 到 {backtest_end_norm}")

        print(f"[workbench] 共 {len(trading_dates)} 个交易日需要重训", flush=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="wf_backtest_"))
        daily_scores_parts: List[pd.DataFrame] = []
        merged_scores_path = str(ns.scores_out)

        try:
            for i, trade_date in enumerate(trading_dates):
                print(f"\n[workbench] ===== 第 {i + 1}/{len(trading_dates)} 日: {trade_date} =====", flush=True)

                # 该日的训练集终点是前一交易日（保证因果性）
                if i == 0:
                    current_train_end = train_end_norm
                else:
                    current_train_end = trading_dates[i - 1]

                daily_scores = str(temp_dir / f"scores_{trade_date}.csv")
                _run_single_day_training(
                    data_root,
                    ns.train_start,
                    current_train_end,
                    trade_date,
                    daily_scores,
                    ns.launcher,
                    ns.nproc,
                    train_argv,
                )

                if not os.path.isfile(daily_scores):
                    raise RuntimeError(f"walk-forward 当日打分文件不存在: {daily_scores}")
                df_day = pd.read_csv(daily_scores)
                if df_day.empty:
                    print(f"[workbench] 警告：{trade_date} 打分为空，已跳过", flush=True)
                    continue
                daily_scores_parts.append(df_day)

            if not daily_scores_parts:
                raise RuntimeError("walk-forward 未生成任何有效打分，无法回测。")

            merged_scores = pd.concat(daily_scores_parts, ignore_index=True)
            if "trade_date" in merged_scores.columns:
                merged_scores["trade_date"] = (
                    merged_scores["trade_date"].astype(str).str.replace("-", "", regex=False)
                )
            if "ts_code" in merged_scores.columns:
                merged_scores["ts_code"] = merged_scores["ts_code"].astype(str)
            if {"trade_date", "ts_code"}.issubset(set(merged_scores.columns)):
                merged_scores = merged_scores.drop_duplicates(
                    subset=["trade_date", "ts_code"], keep="last"
                )
            if "trade_date" in merged_scores.columns:
                merged_scores = merged_scores.sort_values(["trade_date", "ts_code"]).reset_index(
                    drop=True
                )

            out_parent = os.path.dirname(os.path.abspath(merged_scores_path))
            if out_parent:
                os.makedirs(out_parent, exist_ok=True)
            merged_scores.to_csv(merged_scores_path, index=False)
            print(
                f"[workbench] 已合并 walk-forward scores: {merged_scores_path} "
                f"（{len(merged_scores)} 行，{merged_scores['trade_date'].nunique() if 'trade_date' in merged_scores.columns else 'unknown'} 天）",
                flush=True,
            )

            # 用合并后的全区间 scores 做一次连续回测（仓位连续，score-lag 生效）
            _run_single_day_backtest(
                merged_scores_path,
                data_root,
                ns.out_curve,
                ns.out_summary,
                float(ns.cash),
                ns.n_pool,
                ns.k_hold,
                ns.score_lag,
                ns.trade_price_col,
                ns.commission_rate,
                ns.no_benchmark,
                getattr(ns, "benchmark", "000300.SH.csv"),
                getattr(ns, "no_panel_metrics", False),
                getattr(ns, "panel_metrics_all_score_dates", False),
                getattr(ns, "min_names_panel_ic", 10),
                getattr(ns, "out_panel_daily_ic", ""),
            )
            print(
                f"[workbench] Walk-forward 完成。净值: {ns.out_curve} ；摘要: {ns.out_summary}",
                flush=True,
            )
        finally:
            import shutil

            try:
                shutil.rmtree(temp_dir)
                print(f"[workbench] 已清理临时目录: {temp_dir}", flush=True)
            except OSError:
                pass

        return

    # 原有非 walk-forward 逻辑（单次训练）
    print(
        f"[workbench] 验证打分锚定: {fmt_yyyy_mm_dd(val_start_d)} .. {fmt_yyyy_mm_dd(backtest_end_norm)}",
        flush=True,
    )

    inner: List[str] = [
        "--workflow",
        "backtest",
        "--data-root",
        data_root,
        "--train-start",
        _fmt_input_date(ns.train_start),
        "--train-end",
        _fmt_input_date(ns.train_end),
        "--val-start",
        fmt_yyyy_mm_dd(val_start_d),
        "--val-end",
        fmt_yyyy_mm_dd(backtest_end_norm),
        "--export-scores",
        ns.scores_out,
        *train_argv,
    ]
    train_cmd = _train_command(ns.launcher, ns.nproc, inner)
    print("[workbench] 训练 + 验证导出:", " ".join(train_cmd), flush=True)
    subprocess.run(train_cmd, check=True)

    bt_cmd = [
        sys.executable,
        str(_DL_DIR / "backtest.py"),
        "--scores",
        ns.scores_out,
        "--data-root",
        data_root,
        "--out-curve",
        ns.out_curve,
        "--out-summary",
        ns.out_summary,
        "--cash",
        str(ns.cash),
        "--n",
        str(ns.n_pool),
        "--k",
        str(ns.k_hold),
        "--score-lag",
        str(ns.score_lag),
        "--trade-price-col",
        str(ns.trade_price_col),
        "--commission-rate",
        str(ns.commission_rate),
        "--benchmark",
        str(ns.benchmark),
        "--min-names-panel-ic",
        str(ns.min_names_panel_ic),
    ]
    if ns.no_benchmark:
        bt_cmd.append("--no-benchmark")
    if getattr(ns, "no_panel_metrics", False):
        bt_cmd.append("--no-panel-metrics")
    if getattr(ns, "panel_metrics_all_score_dates", False):
        bt_cmd.append("--panel-metrics-all-score-dates")
    if getattr(ns, "out_panel_daily_ic", ""):
        bt_cmd.extend(["--out-panel-daily-ic", ns.out_panel_daily_ic])
    print("[workbench] 历史回测:", " ".join(bt_cmd), flush=True)
    subprocess.run(bt_cmd, check=True)
    print(f"[workbench] 完成。净值: {ns.out_curve} ；摘要: {ns.out_summary}", flush=True)


def cmd_predict_next(ns: argparse.Namespace, train_argv: List[str]) -> None:
    """盘后：训练（可选）→ 仅输出目标权重 plan JSON（不含股数）。"""
    data_root = ns.data_root
    raw_state = _read_json(ns.state_in)
    pos = infer_position_mode_from_state_dict(raw_state)

    if not ns.skip_train:
        inner = [
            "--workflow",
            "predict-next",
            "--data-root",
            data_root,
            "--train-start",
            _fmt_input_date(ns.train_start),
            "--train-end",
            _fmt_input_date(ns.train_end),
            "--export-scores",
            ns.export_scores,
            *train_argv,
        ]
        train_cmd = _train_command(ns.launcher, ns.nproc, inner)
        print("[workbench] 训练（predict-next）:", " ".join(train_cmd), flush=True)
        subprocess.run(train_cmd, check=True)
    else:
        print("[workbench] 跳过训练，使用已有 scores:", ns.export_scores, flush=True)

    next_d = (ns.next_trade_date or "").strip().replace("-", "")
    if not next_d:
        next_d = _infer_next_trade_date_from_scores_and_daily(data_root, ns.export_scores)
    if len(next_d) != 8 or not next_d.isdigit():
        raise SystemExit(f"无效 next_trade_date: {next_d!r}")

    plan_cmd = [
        sys.executable,
        str(_DL_DIR / "predict.py"),
        "--mode",
        "plan",
        "--scores",
        ns.export_scores,
        "--data-root",
        data_root,
        "--next-trade-date",
        next_d,
        "--n",
        str(ns.n_pool),
        "--k",
        str(ns.k_hold),
        "--score-lag",
        str(ns.score_lag),
        "--state",
        ns.state_in,
        "--out-plan",
        ns.ops_out,
    ]
    print("[workbench] 盘后 plan（目标权重 + 参考金额）:", " ".join(plan_cmd), flush=True)
    subprocess.run(plan_cmd, check=True)

    plan_payload = _read_json(ns.ops_out)
    plan_payload["input_position_mode"] = pos
    plan_payload["state_in"] = ns.state_in
    plan_payload["rebalance_style"] = str(plan_payload.get("rebalance_style", "target_tracking"))
    plan_payload["budget_basis"] = str(plan_payload.get("budget_basis", "nav"))
    plan_payload["execute_trade_price_col"] = str(getattr(ns, "trade_price_col", "open"))
    if getattr(ns, "commission_rate", None) is not None:
        plan_payload["execute_commission_rate"] = float(ns.commission_rate)
    plan_payload["notes"] = (
        (plan_payload.get("notes") or "")
        + " execute-next 可读取 execute_trade_price_col / execute_commission_rate；"
        "若命令行未指定则沿用 plan 中的值。"
    )
    outp = Path(ns.ops_out)
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(plan_payload, f, ensure_ascii=False, indent=2)
    print(
        f"[workbench] 已写目标权重 plan: {outp}（共 {len(plan_payload.get('target_weights', []))} 只；"
        f"budget_nav={plan_payload.get('budget_nav', plan_payload.get('budget_cash'))}；"
        f"style={plan_payload.get('rebalance_style')}；不含 orders，请运行 execute-next 用开盘价换算股数）",
        flush=True,
    )


def cmd_execute_next(ns: argparse.Namespace) -> None:
    """开盘：读取 plan + 账户 state，用当日 open 价换算整手股数并输出 orders。"""
    data_root = ns.data_root
    raw_state = _read_json(ns.state_in)
    pos = infer_position_mode_from_state_dict(raw_state)
    plan_raw = _read_json(ns.plan_in)

    trade_d = (ns.trade_date or "").strip().replace("-", "")
    if not trade_d:
        trade_d = str(plan_raw.get("next_trade_date", "")).strip().replace("-", "")
    if len(trade_d) != 8 or not trade_d.isdigit():
        raise SystemExit("execute-next 需要有效的 --trade-date YYYYMMDD（或 plan 内含 next_trade_date）")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, encoding="utf-8"
    ) as tmp_o:
        orders_path = tmp_o.name
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp_s:
        next_state_tmp = tmp_s.name
    try:
        trade_price_col = str(ns.trade_price_col)
        if trade_price_col == "open" and plan_raw.get("execute_trade_price_col"):
            trade_price_col = str(plan_raw["execute_trade_price_col"])

        commission_rate = ns.commission_rate
        if commission_rate is None and plan_raw.get("execute_commission_rate") is not None:
            commission_rate = float(plan_raw["execute_commission_rate"])

        sim_cmd = [
            sys.executable,
            str(_DL_DIR / "predict.py"),
            "--mode",
            "execute",
            "--plan",
            ns.plan_in,
            "--data-root",
            data_root,
            "--state",
            ns.state_in,
            "--next-trade-date",
            trade_d,
            "--trade-price-col",
            trade_price_col,
            "--out-orders",
            orders_path,
            "--out-next-state",
            next_state_tmp,
        ]
        if commission_rate is not None:
            sim_cmd.extend(["--commission-rate", str(commission_rate)])
        if getattr(ns, "strict_next_trade_csv", True):
            sim_cmd.append("--strict-next-trade-csv")
        print("[workbench] 开盘 execute（目标权重→股数）:", " ".join(sim_cmd), flush=True)
        subprocess.run(sim_cmd, check=True)

        orders_rows: List[Dict[str, Any]] = []
        if os.path.isfile(orders_path) and os.path.getsize(orders_path) > 0:
            orders_rows = pd.read_csv(orders_path).to_dict(orient="records")

        next_state_raw: Optional[Dict[str, Any]] = None
        if os.path.isfile(next_state_tmp) and os.path.getsize(next_state_tmp) > 0:
            with open(next_state_tmp, "r", encoding="utf-8") as f:
                next_state_raw = json.load(f)

        px_date_used, px_note_used = resolve_equity_trade_price_date(
            data_root,
            trade_d,
            strict=True,
            price_col=trade_price_col,
        )
        price_col_used = effective_trade_price_col(
            trade_price_col,
            str(trade_d),
            str(px_date_used),
        )

        out_payload = {
            "workflow": "execute-at-open",
            "input_position_mode": pos,
            "plan_ref": ns.plan_in,
            "rebalance_style": plan_raw.get("rebalance_style", "target_tracking"),
            "budget_basis": plan_raw.get("budget_basis", "nav"),
            "trade_date": trade_d,
            "pricing_trade_date": px_date_used,
            "pricing_price_col": price_col_used,
            "pricing_is_placeholder": bool(str(px_date_used) != str(trade_d)),
            "pricing_trade_date_note": px_note_used or "",
            "score_snapshot_trade_date": plan_raw.get("score_snapshot_trade_date"),
            "score_snapshot_note": plan_raw.get("score_snapshot_note", ""),
            "score_lag": plan_raw.get("score_lag"),
            "hold_n": plan_raw.get("hold_n", plan_raw.get("candidate_pool_n")),
            "rotate_k": plan_raw.get("rotate_k", plan_raw.get("hold_top_k")),
            "target_weights": plan_raw.get("target_weights", []),
            "orders": orders_rows,
            "budget_nav": plan_raw.get("budget_nav", plan_raw.get("budget_cash")),
            "budget_cash": plan_raw.get("budget_cash"),
            "planned_buys": plan_raw.get("planned_buys", []),
            "portfolio_after_close": next_state_raw,
            "notes": (
                "orders 含 target_weight、target_amount（参考/目标金额）、amount（实际成交金额）。"
                "portfolio_after_close 为当日收盘后状态（当日买入在 locked）。"
            ),
        }

        outp = Path(ns.ops_out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=2)
        print(f"[workbench] 已写开盘执行 JSON: {outp}", flush=True)
    finally:
        try:
            os.unlink(orders_path)
        except OSError:
            pass
        try:
            os.unlink(next_state_tmp)
        except OSError:
            pass


def main() -> None:
    p = argparse.ArgumentParser(description="深度学习作业：工作台 CLI（编排 train / backtest / predict）")
    subs = p.add_subparsers(dest="cmd", required=True)

    pb = subs.add_parser(
        "backtest",
        help="训练 → 验证段打分 → 历史回测（默认验证起点=train-end 下一开市日，可用 --val-start 覆盖）",
    )
    pb.add_argument("--data-root", default=os.environ.get("DL_DATA_ROOT", ""))
    pb.add_argument("--train-start", required=True, help="训练锚定起始 YYYY-MM-DD / YYYYMMDD")
    pb.add_argument(
        "--train-end",
        required=True,
        help="训练锚定结束（含）；未写 --val-start 时验证起点为其下一开市日",
    )
    pb.add_argument(
        "--val-start",
        default="",
        help=(
            "可选。验证 / 导出打分首日锚定（YYYY-MM-DD / YYYYMMDD），须严格晚于 train-end。"
            "用于按「日历」指定回测窗口（如 2026-01-02）；实际有样本的日期仍受 trade_cal 与 daily/ 限制。"
        ),
    )
    pb.add_argument(
        "--backtest-end",
        required=True,
        help="回测 / 打分导出的最后锚定日（含），对应训练侧 --val-end",
    )
    pb.add_argument("--scores-out", default="outputs/workflow_backtest_scores.csv")
    pb.add_argument("--out-curve", default="outputs/workflow_equity_curve.csv")
    pb.add_argument("--out-summary", default="outputs/workflow_backtest_summary.csv")
    pb.add_argument("--cash", type=float, default=1_000_000.0)
    pb.add_argument("--n-pool", type=int, default=20, dest="n_pool")
    pb.add_argument("--k-hold", type=int, default=4, dest="k_hold")
    pb.add_argument("--score-lag", type=int, default=1)
    pb.add_argument(
        "--trade-price-col",
        choices=("open", "close"),
        default="open",
        help="回测撮合价格列（默认 open，满足每日开盘买卖）",
    )
    pb.add_argument("--commission-rate", type=float, default=0.0002)
    pb.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="兼容旧参数：基点制（万三=3），若提供则覆盖 --commission-rate",
    )
    pb.add_argument("--no-benchmark", action="store_true")
    pb.add_argument(
        "--benchmark",
        default="000300.SH.csv",
        help="相对于 data-root/market 下指数收益曲线（默认沪深300）；配合 pct_chg 列",
    )
    pb.add_argument(
        "--walk-forward",
        action="store_true",
        dest="walk_forward",
        help="启用逐日重训模式：每个交易日重新训练模型并打分，然后回测（更接近实盘流程）",
    )
    pb.add_argument(
        "--no-panel-metrics",
        action="store_true",
        help="不在 summary 中写入基于 label_return 的 panel_* IC/胜率（仍会跑组合回测）",
    )
    pb.add_argument(
        "--panel-metrics-all-score-dates",
        action="store_true",
        help="截面指标改用 scores 中带标签的全部行（默认仅用净值曲线出现的 trade_date 以对齐仿真区间）",
    )
    pb.add_argument(
        "--min-names-panel-ic",
        type=int,
        default=10,
        help="逐日截面 IC 时该日至少需要的股票数（与训练侧 --min-names-ic 一致）",
    )
    pb.add_argument(
        "--out-panel-daily-ic",
        default="",
        help="可选：写出按日 Pearson/Rank IC 的 CSV；留空则不写",
    )
    pb.add_argument("--launcher", choices=("python", "torchrun"), default="python")
    pb.add_argument("--nproc", type=int, default=1, help="torchrun 时每机进程数（GPU 数）")
    pb.add_argument(
        "train_argv",
        nargs=argparse.REMAINDER,
        help="传给 train.py 的额外参数，前置 -- ，例如: -- --epochs 3 --batch-size 512",
    )

    pn = subs.add_parser(
        "predict-next",
        help="盘后：训练（可选）+ 仅输出目标权重 plan JSON（不含股数/不含占位价）",
    )
    pn.add_argument("--data-root", default=os.environ.get("DL_DATA_ROOT", ""))
    pn.add_argument("--train-start", required=True)
    pn.add_argument(
        "--train-end",
        required=True,
        help="仍可计算标签的最后锚定日（通常为面板末日往前 horizon 个交易日，例如默认 horizon=3）",
    )
    pn.add_argument(
        "--export-scores",
        default="outputs/workflow_predict_next_scores.csv",
        help="训练写出 pred_score；predict-next 也用它喂给 predict.py",
    )
    pn.add_argument("--state-in", required=True, help="账户 JSON（示例见 examples/）")
    pn.add_argument("--ops-out", required=True, help="输出：目标权重 plan JSON")
    pn.add_argument(
        "--next-trade-date",
        default="",
        help="语义上的下一交易日 YYYYMMDD（plan 不含成交价；股数明早 execute-next 再算）",
    )
    pn.add_argument("--skip-train", action="store_true", help="跳过训练；勿写在「--」后，勿与 --stock-pool 等粘连")
    pn.add_argument("--n-pool", type=int, default=20, dest="n_pool")
    pn.add_argument("--k-hold", type=int, default=4, dest="k_hold")
    pn.add_argument("--score-lag", type=int, default=1)
    pn.add_argument(
        "--trade-price-col",
        choices=("open", "close"),
        default="open",
        help="写入 plan 供 execute-next 使用（plan 阶段不算股数；默认 open）",
    )
    pn.add_argument(
        "--commission-rate",
        type=float,
        default=None,
        help="写入 plan 供 execute-next 使用；不传则 execute 时用 state 内费率",
    )
    pn.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="兼容旧参数：基点制（万三=3），若提供则覆盖 --commission-rate",
    )
    pn.add_argument("--launcher", choices=("python", "torchrun"), default="python")
    pn.add_argument("--nproc", type=int, default=1)
    pn.add_argument(
        "train_argv",
        nargs=argparse.REMAINDER,
        help="传给 train.py；须以 -- 隔开。勿把 --skip-train 写在本段（predict-next 另有 --skip-train）",
    )

    ex = subs.add_parser(
        "execute-next",
        help="开盘：读取 plan + state，用当日 open 价换算整手 orders",
    )
    ex.add_argument("--data-root", default=os.environ.get("DL_DATA_ROOT", ""))
    ex.add_argument("--plan-in", required=True, help="predict-next 写出的 plan JSON")
    ex.add_argument("--state-in", required=True, help="账户 JSON（示例见 examples/）")
    ex.add_argument("--ops-out", required=True, help="输出：开盘执行 JSON（含 orders）")
    ex.add_argument(
        "--trade-date",
        default="",
        help="执行交易日 YYYYMMDD；默认取 plan.next_trade_date",
    )
    ex.add_argument(
        "--trade-price-col",
        choices=("open", "close"),
        default="open",
        help="撮合价格列（默认 open）",
    )
    ex.add_argument(
        "--no-strict-next-trade-csv",
        action="store_true",
        help="允许无当日 CSV 时回退（不推荐；默认必须存在当日 daily CSV）",
    )
    ex.add_argument("--commission-rate", type=float, default=None)
    ex.add_argument(
        "--commission-bps",
        type=float,
        default=None,
        help="兼容旧参数：基点制（万三=3），若提供则覆盖 --commission-rate",
    )

    args = p.parse_args()
    if getattr(args, "commission_bps", None) is not None:
        args.commission_rate = float(args.commission_bps) / 10000.0
    args.data_root = resolve_data_root(args.data_root)

    tv = getattr(args, "train_argv", None) or []
    if tv and tv[0] == "--":
        tv = tv[1:]

    tv, glued_skip_train = _sanitize_pass_through_train_argv(tv)
    if glued_skip_train and args.cmd == "predict-next":
        if not getattr(args, "skip_train", False):
            args.skip_train = True
            print(
                "[workbench] 提示：检测到将 --skip-train 粘在其它参数后面或写在「--」之后。"
                f"已对 argv 解压并启用跳过训练。推荐写法：`… {_PASS_SKIP_TRAIN} -- …`（{_PASS_SKIP_TRAIN} 在「--」前）。",
                flush=True,
            )
    elif glued_skip_train and args.cmd == "backtest":
        print(
            f"[workbench] 提示：{_PASS_SKIP_TRAIN} 仅用于 predict-next，已从你的训练额外参数里移除。"
            "backtest 工作流仍会照常训练后再回测。",
            flush=True,
        )

    if args.cmd == "backtest":
        cmd_backtest(args, tv)
    elif args.cmd == "predict-next":
        cmd_predict_next(args, tv)
    elif args.cmd == "execute-next":
        args.strict_next_trade_csv = not getattr(args, "no_strict_next_trade_csv", False)
        cmd_execute_next(args)
    else:
        raise SystemExit("unknown cmd")


if __name__ == "__main__":
    main()
