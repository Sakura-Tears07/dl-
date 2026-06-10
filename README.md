# 深度学习选股与组合调仓系统

## 概述

本目录提供基于 Transformer 的 A 股截面打分、历史回测与实盘调仓能力。系统以 `workbench.py` 为统一命令行入口，由数据预处理、模型训练、组合回测与交易计划生成等模块协同完成端到端流程。

| 能力 | 说明 |
|------|------|
| 截面打分 | 对股票池逐日输出 `pred_score`，用于排序与权重分配 |
| 历史回测 | 在固定训练窗或 Walk-forward 重训模式下评估组合表现 |
| 实盘调仓 | 盘后生成目标权重（plan），开盘按规则生成执行结果（execute） |

账户状态 JSON 的字段定义与示例见 `examples/` 及 `examples/README.md`。

---

## 模块组成

| 模块 | 源文件 | 职责 |
|------|--------|------|
| 数据 | `data_preprocess.py` | 读取日线及侧车特征，构造标签、时序样本与标准化参数 |
| 训练 | `train.py` | 模型训练；导出验证段或推理段 `pred_score` |
| 回测 | `backtest.py` | 基于打分 CSV 进行历史组合仿真与绩效统计 |
| 推理 | `predict.py` | 生成目标权重 plan 及开盘撮合指令 |
| 编排 | `workbench.py` | 串联 `backtest`、`predict-next`、`execute-next` 子流程 |

---

## 环境与依赖

### 安装

```bash
conda activate dl
cd /path/to/fundamentals_for_deep_learning/dl-
pip install -r requirements.txt
export DL_DATA_ROOT=/path/to/fundamentals_for_deep_learning/data
```

### 数据目录约定

`DL_DATA_ROOT` 须至少包含：

- `daily/`：按交易日划分的行情 CSV；
- `market/`：基准指数行情（默认沪深 300：`000300.SH.csv`）；
- 训练配置所需的侧车数据（资金流、指标等，见 `data_preprocess.py` 与数据仓库说明）。

---

## 分布式训练

训练阶段可通过 `workbench.py` 指定进程启动方式：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--launcher` | `python` | `python`：单进程；`torchrun`：多进程分布式（DDP） |
| `--nproc` | `4` | 仅当 `--launcher torchrun` 时生效，对应 `torchrun --nproc_per_node`（单机 GPU 进程数） |

单机多卡且仍使用单进程时，可在传给 `train.py` 的参数中启用 `--multi-gpu`（`nn.DataParallel`）；与 `torchrun` 的 DDP 路径互斥，不宜同时混用。

下文示例在需多卡训练时使用 `--launcher torchrun`；未显式指定 `--nproc` 时采用默认值 4。若本机 GPU 数量不同，请按需覆盖该参数。

---

## 策略与仿真约定

### 选股与持仓

| 符号 | 工作台参数 | 默认值 | 含义 |
|------|------------|--------|------|
| n | `--n-pool` | 20 | 同一时段目标持仓只数（`pred_score` Top-n） |
| k | `--k-hold` | 4 | 有持仓时，每个调仓日优先卖出的最低分持仓只数 |

目标组合内，参考权重与 `pred_score` 成正比（平移后归一化），写入 plan 的 `target_weight`。参考购买金额为 `target_amount = budget_nav × target_weight`，其中 `budget_nav = 现金 + 持仓市值`（按打分快照日收盘价估算）。

### 调仓规则

1. **空仓建仓**（如 `examples/state_empty.json`）：首次建仓一次性买入 Top-n；参数 `k` 不参与。
2. **有持仓调仓**（`predict-next` / `execute-next`）：采用 `target_tracking` 目标跟踪口径：
   - 按目标权重计算目标股数（整手约束）；
   - 先卖出持仓中得分最低的 `k` 只；
   - 若卖出后持仓只数低于 `n`，允许新增标的以补齐至 `n`；
   - 执行顺序为先卖后买：先处理超配与非目标仓位，再以现金及卖出回笼资金补足低配仓位；
   - 允许对已有持仓进行加仓或减仓（同名标的按 `pred_score` 再分配）；
   - 全程应用费用模型与可交易价格过滤。
3. **回测引擎**：`backtest.py` 保留 `rotate-k` 等历史口径；经工作台发起的实盘两阶段流程默认使用 `target_tracking`。

### 训练与回测的关系

默认 `workbench backtest` **不在每个交易日重新训练**，流程如下：

| 步骤 | 内容 |
|------|------|
| 1 | 在固定窗口 `[train-start, train-end]` 上训练一次模型 |
| 2 | 同一模型对 `[val-start, backtest-end]` 批量导出 `pred_score` |
| 3 | `backtest.py` 读取打分 CSV，按 `score-lag` 逐日调仓（不再训练） |

上述模式为**固定训练窗的样本外检验**（单模型、多日落盘分数）。与实盘「每个决策日盘后 `predict-next` 重新训练」的口径不同，评估时需区分场景。

#### Walk-forward 日度重训

指定 `--walk-forward` 后，对每个交易日 `T`：

| 步骤 | 内容 |
|------|------|
| 1 | 基于 `[train-start, T-1]` 重新训练 |
| 2 | 导出 `T` 日 `pred_score` |
| 3 | 以该日打分推进组合回测 |
| 4 | 重复直至回测区间结束 |

**说明**：该模式计算开销显著（回测 N 日约等于训练 N 次）；建议适当降低 `--epochs`。训练窗随 `T` 扩展，以利用全部可用历史。

### 时间因果

- `score-lag=1`（默认）：交易日 T 的调仓依据 T−1 日导出的打分截面；
- 训练集对验证起点执行 purge，避免标签区间与验证段重叠；
- 回测与实盘流程采用一致的因果约束。

### 实盘两阶段

| 阶段 | 子命令 | 说明 |
|------|--------|------|
| 权重规划 | `predict-next` | 盘后输出 Top-n 目标权重 plan（不含成交股数） |
| 开盘执行 | `execute-next` | 以执行日 `open` 撮合；须存在 `daily/{trade_date}.csv` |

### 费用模型

| 项目 | 规则 |
|------|------|
| 佣金 | 买卖双向；费率由 `--commission-rate` 指定（工作台默认 `0.0002`）；单笔最低 5 元 |
| 印花税 | 卖出成交金额 × 0.1% |
| 过户费 | 仅上证 60 开头标的，买卖双向，按成交股数计费；单笔最低 1 元 |

示例命令中若使用万三，可显式传入 `--commission-rate 0.0003`。

---

## 操作流程

### 1. 历史回测（固定训练窗）

训练模型，导出验证段打分，并完成组合回测：

```bash
python workbench.py backtest \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --val-start 2026-01-07 \
  --backtest-end 2026-04-07 \
  --scores-out outputs/wf_scores.csv \
  --out-curve outputs/wf_equity.csv \
  --out-summary outputs/wf_summary.csv \
  --n-pool 20 \
  --k-hold 4 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --launcher torchrun \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 3 \
  --stock-pool all \
  --base-head-weight 0.80 \
  --rank-loss-weight 0.05 \
  --rank-loss-max-pairs 2048 \
  --feature-select-mode ic_prune \
  --keep-all-base-features \
  --sidecar-feature-budget 10 \
  --feature-report-path outputs/feature_report.json \
  --export-metrics-json outputs/wf_metrics.json \
  --export-daily-ic-csv outputs/wf_daily_ic.csv \
  --early-stop-min-epochs 25 \
  --early-stop-patience 12
```

`--` 之后的参数原样传递给 `train.py`。

### 2. Walk-forward 回测

```bash
python workbench.py backtest \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-01-06 \
  --backtest-end 2026-04-07 \
  --walk-forward \
  --out-curve outputs/wf_equity_wf.csv \
  --out-summary outputs/wf_summary_wf.csv \
  --n-pool 20 \
  --k-hold 4 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --launcher torchrun \
  --nproc 4 \
  -- \
  --epochs 30 \
  --batch-size 1024 \
  --horizon 3
```

### 3. 实盘调仓

#### 3.1 权重规划（`predict-next`）

决策日收盘后训练（或复用已有打分），输出目标权重 plan：

```bash
python workbench.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-06-07 \
  --export-scores outputs/wf_scores.csv \
  --state-in examples/state_holding.json \
  --ops-out outputs/final_plan.json \
  --next-trade-date 20260608 \
  --n-pool 20 \
  --k-hold 4 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --launcher torchrun \
  --nproc 4 \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 3 \
  --stock-pool all \
  --export-metrics-json outputs/wf_metrics.json
```

若打分文件已存在，可跳过训练（`--skip-train` 须写在 `--` 之前，勿传入 `train.py`）：

```bash
python workbench.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-05-28 \
  --export-scores outputs/wf_scores.csv \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_plan.json \
  --next-trade-date 20260529 \
  --n-pool 20 \
  --k-hold 4 \
  --score-lag 1 \
  --trade-price-col open \
  --commission-rate 0.0003 \
  --skip-train
```

`final_plan.json` 主要字段：

| 字段 | 说明 |
|------|------|
| `budget_nav` | 参考预算总资产（现金 + 持仓市值） |
| `rebalance_style` | 调仓风格（默认 `target_tracking`） |
| `switch_cap_k` | 每轮优先卖出的最低分持仓只数（与 `--k-hold` 一致） |
| `planned_entries` / `planned_exits` | 计划层面的新进 / 退出标的 |
| `target_weights` | Top-n 目标持仓（含 `ts_code`、`pred_score`、`target_weight`、`target_amount`） |
| `planned_buys` | 有持仓时按缺口预估的加仓 / 建仓标的 |

plan 不含成交股数；`--trade-price-col` 与 `--commission-rate` 写入 plan，供执行阶段引用。

#### 3.2 开盘执行（`execute-next`）

```bash
python workbench.py execute-next \
  --data-root "$DL_DATA_ROOT" \
  --plan-in outputs/final_plan.json \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_ops.json \
  --trade-date 20260529
```

覆盖 plan 中的执行参数：

```bash
python workbench.py execute-next \
  --data-root "$DL_DATA_ROOT" \
  --plan-in outputs/final_plan.json \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_ops.json \
  --trade-date 20260529 \
  --trade-price-col open \
  --commission-rate 0.0003
```

#### 3.3 连续调仓

持仓期间，每个决策日重复「`predict-next` → `execute-next`」：`predict-next` 读取当前账户 JSON 生成 plan；`execute-next` 在下一执行日按开盘价撮合。

---

## 输出产物

| 路径（示例） | 内容 |
|--------------|------|
| `outputs/wf_scores.csv` | 逐日逐股 `pred_score` |
| `outputs/wf_equity.csv` | 回测净值曲线 |
| `outputs/wf_summary.csv` | 回测摘要指标 |
| `outputs/wf_metrics.json` | 训练验证指标 |
| `outputs/wf_daily_ic.csv` | 逐日截面 IC |
| `outputs/feature_report.json` | 特征筛选报告 |
| `outputs/final_plan.json` | 目标权重 plan |
| `outputs/final_ops.json` | 执行结果与收盘后账户状态 |

---

## 命令行参数参考

### `workbench.py backtest`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-start` / `--train-end` | （必填） | 训练锚定区间 |
| `--val-start` | 空 | 验证 / 打分首日；未指定时为 `train-end` 的下一开市日 |
| `--backtest-end` | （必填） | 回测与打分导出的最后锚定日 |
| `--scores-out` / `--out-curve` / `--out-summary` | 见 `--help` | 输出路径 |
| `--n-pool` | 20 | 目标持仓只数 n |
| `--k-hold` | 4 | 日度换仓只数 k（空仓建仓时不适用） |
| `--score-lag` | 1 | 打分滞后天数 |
| `--trade-price-col` | `open` | 回测撮合价格列 |
| `--commission-rate` | `0.0002` | 券商佣金费率 |
| `--benchmark` | `000300.SH.csv` | 基准指数文件名（相对 `market/`） |
| `--no-benchmark` | — | 不计算相对基准超额 |
| `--walk-forward` | 关闭 | 启用逐日重训回测 |
| `--launcher` | `python` | 训练启动方式 |
| `--nproc` | `4` | `torchrun` 时每机进程数（GPU 数） |

`--` 之后参数传递至 `train.py`。

### `workbench.py predict-next`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train-start` / `--train-end` | （必填） | 训练锚定区间 |
| `--export-scores` | 见 `--help` | 打分 CSV 路径 |
| `--state-in` | （必填） | 账户 JSON |
| `--ops-out` | （必填） | plan JSON 输出路径 |
| `--next-trade-date` | 空 | 目标执行日（`YYYYMMDD`）；可留空由数据推断 |
| `--n-pool` / `--k-hold` / `--score-lag` | 同上 | 组合与因果参数 |
| `--skip-train` | 关闭 | 跳过训练，复用已有打分 |
| `--launcher` / `--nproc` | `python` / `4` | 分布式训练配置 |

### `workbench.py execute-next`

| 参数 | 说明 |
|------|------|
| `--plan-in` | 权重 plan JSON |
| `--state-in` | 执行前账户 JSON |
| `--ops-out` | 执行结果 JSON |
| `--trade-date` | 执行日（默认取 plan 中 `next_trade_date`） |
| `--trade-price-col` / `--commission-rate` | 可覆盖 plan 中的对应字段 |
| `--no-strict-next-trade-csv` | 允许在缺少当日 CSV 时回退（默认须严格存在） |

### `train.py` 常用参数（经 `--` 传入）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window-len` | 60 | 输入序列长度 |
| `--horizon` | 3 | 预测收益 horizon |
| `--batch-size` | 1024 | 批大小 |
| `--feature-select-mode` | `ic_prune` | 特征筛选（`none` / `ic_prune`） |
| `--rank-loss-weight` | 0 | 排序辅助损失权重 |
| `--early-stop-min-epochs` / `--early-stop-patience` | 见 `--help` | 早停配置 |
| `--multi-gpu` | 关闭 | 单进程 DataParallel（与 `torchrun` DDP 二选一） |

完整参数列表请执行：`python train.py --help`。

---

## 底层脚本接口

可直接调用 `predict.py`，适用于自定义编排或与工作台解耦的流水线：

```bash
python predict.py --mode plan \
  --scores outputs/wf_scores.csv \
  --data-root "$DL_DATA_ROOT" \
  --next-trade-date 20260529 \
  --n 20 --k 4 --score-lag 1 \
  --out-plan outputs/final_plan.json

python predict.py --mode execute \
  --plan outputs/final_plan.json \
  --state examples/state_empty.json \
  --data-root "$DL_DATA_ROOT" \
  --next-trade-date 20260529 \
  --strict-next-trade-csv \
  --trade-price-col open \
  --out-orders outputs/orders.csv \
  --out-next-state outputs/next_state.json
```

`predict.py execute` 模式下，未指定 `--commission-rate` 时使用账户 JSON 中的 `commission_rate` 字段。