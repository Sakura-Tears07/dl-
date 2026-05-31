# 深度学习选股与组合调仓

本目录实现基于 Transformer 的截面打分、历史回测与实盘调仓流程。统一入口为 `workbench.py`；核心模块包括数据预处理、模型训练、回测仿真与交易计划生成。

---

## 目录结构

| 模块 | 文件 | 功能 |
|---|---|---|
| 数据 | `data_preprocess.py` | 读取日线及侧车特征，构造标签与序列，标准化 |
| 训练 | `train.py` | 训练模型，导出验证段或推理段 `pred_score` |
| 回测 | `backtest.py` | 基于打分进行历史组合仿真与指标统计 |
| 推理 | `predict.py` | 生成目标权重（plan）及开盘撮合指令（execute） |
| 编排 | `workbench.py` | 串联训练、回测、`predict-next` 与 `execute-next` |

账户状态 JSON 示例见 `examples/`（说明见 `examples/README.md`）。

---

## 环境配置

```bash
conda activate dl
cd /path/to/fundamentals_for_deep_learning/dl-
pip install -r requirements.txt
export DL_DATA_ROOT=/path/to/fundamentals_for_deep_learning/data
```

数据目录需包含 `daily/`（按交易日 CSV）、`market/`（基准指数）及训练所需侧车数据。

---

## 策略与仿真口径

### 选股与持仓

- **`n`（`--n-pool`）**：同一时段目标持有股票只数，取 pred_score 最高的 Top-n（默认 20）。
- **`k`（`--k-hold`）**：有持仓时的日度换仓只数（默认 4）。
- 目标 Top-n 内，参考权重与 `pred_score` 成正比（平移后归一化），写入 plan 的 `target_weight`；参考购买金额 `target_amount = budget_cash × target_weight`（`budget_cash` 取自账户 JSON 的现金余额）。

### 调仓规则

1. **空仓（如 `state_empty.json`）**：首次建仓一次性买入 Top-n，`k` 不参与。
2. **有持仓**：每个交易日
   - **强制卖出**已跌出 Top-n 的持仓；
   - 在剩余持仓中卖出 **分数最低的 k 只**（可卖整手）；
   - 从 Top-n 中 **尚未持有** 的标的里按分数从高到低买入，**买入只数 = 本轮成功卖出只数**（不超过 `n − 当前持仓只数`）；
   - 新买入资金按 pred_score 加权分配。
3. 组合目标规模维持 **≤ n 只**；正常日度最多换 **k 只**（跌出 Top-n 的 trim 不计入 k）。

### 训练与回测的关系（重要）

**默认 `workbench backtest` 不是「每个交易日重新训练」。** 实际流程是：

| 步骤 | 做什么 |
|---|---|
| 1 | **训练一次**：固定窗口 `[train-start, train-end]` 上拟合模型 |
| 2 | **批量导出打分**：同一模型对验证段 `[val-start, backtest-end]` 逐日输出 `pred_score` → `wf_scores.csv` |
| 3 | **组合回测**：`backtest.py` 读取上述 CSV，按 `score-lag=1` 每日调仓（不再训练） |

这与实盘 **「每个决策日盘后 `predict-next` 重新训练一次」** 不同：默认回测用的是 **单模型 + 多日落盘分数**，算是一种 **固定训练窗的样本外检验**。

#### Walk-forward 日度重训模式（已支持）

使用 `--walk-forward` 参数可启用 **逐日重训** 回测，更接近实盘流程：

| 步骤 | 做什么 |
|---|---|
| 1 | 对每个交易日 `T`，基于历史 `[train-start, T-1]` 重新训练模型 |
| 2 | 导出 `T` 日的 `pred_score` |
| 3 | 基于 `T` 日的打分进行回测，记录净值 |
| 4 | 重复 1-3 直到回测区间结束 |

使用方法：

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
  --nproc 8 \
  -- \
  --epochs 30 \
  --batch-size 1024 \
  --horizon 3
```

**注意**：
- Walk-forward 模式下，**每天都需要重新训练**，计算开销会显著增加（回测 60 天 ≈ 训练 60 次）。
- 建议配合 `--epochs 30` 或更少的 epoch 数以缩短单次训练时间。
- 训练窗口 `[train-start, T-1]` 会随日期 `T` 逐渐扩展，确保使用所有可用历史数据。

### 时间因果

- `score-lag=1`（默认）：T 日交易依据 T−1 日导出的打分截面。
- 训练集对验证起点执行 purge，保证标签区间不与验证段重叠。
- 回测与实盘均采用相同因果约束。

### 实盘两阶段

| 阶段 | 命令 | 说明 |
|---|---|---|
| 权重规划 | `predict-next` | 导出 Top-n 目标权重 plan（不含股数） |
| 开盘执行 | `execute-next` | 按开盘价执行上述调仓规则，生成 orders |

**开盘执行**使用执行日 `open` 价撮合，要求 `daily/{trade_date}.csv` 已存在；计入 A 股交易费用。

### 费用模型

- 佣金：买卖双向，费率由 `--commission-rate` 指定（如万三为 `0.0003`），单笔最低 5 元。
- 印花税：卖出成交金额 × 0.1%。
- 过户费：仅上证 60 开头标的，买卖双向，按成交股数计费，单笔最低 1 元。

---

## 使用流程

### 1. 历史回测

训练模型并在验证区间导出打分，随后进行组合回测：

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
  --nproc 8 \
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

### 2. 实盘调仓

#### 2.1 权重规划（`predict-next`）

在决策日收盘后，训练（或加载已有打分）并输出目标权重 plan：

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
  --launcher torchrun \
  --nproc 8 \
  -- \
  --epochs 70 \
  --batch-size 1024 \
  --horizon 3 \
  --stock-pool all \
  --export-metrics-json outputs/wf_metrics.json
```

若打分文件已存在，可省略训练步骤：

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

输出文件 `final_plan.json` 主要字段：

- `budget_cash`：参考预算现金（来自 `--state-in`）
- `target_weights`：Top-n 目标持仓，每行含 `ts_code`、`pred_score`、`target_weight`、`target_amount`
- `planned_buys`：有持仓时按 rotate-k 预估的次日买入标的（含权重与参考金额；空仓为空数组）

不含买卖股数。参数 `--trade-price-col` 与 `--commission-rate` 写入 plan，供执行阶段引用。

#### 2.2 开盘执行（`execute-next`）

在执行日获取当日行情后，依据 plan 与账户状态生成指令：

```bash
python workbench.py execute-next \
  --data-root "$DL_DATA_ROOT" \
  --plan-in outputs/final_plan.json \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_ops.json \
  --trade-date 20260529
```

如需覆盖 plan 中的执行参数：

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

#### 2.3 连续调仓

持仓期间，每个决策日重复上述两阶段：`predict-next` 读取当前账户 JSON 生成新 plan；`execute-next` 在下一执行日按开盘价撮合。账户 JSON 字段说明见 `examples/README.md`。

---

## 输出文件

| 路径 | 说明 |
|---|---|
| `outputs/wf_scores.csv` | 逐日逐股 `pred_score` |
| `outputs/wf_equity.csv` | 回测净值曲线 |
| `outputs/wf_summary.csv` | 回测摘要指标 |
| `outputs/wf_metrics.json` | 训练验证指标 |
| `outputs/wf_daily_ic.csv` | 逐日截面 IC |
| `outputs/feature_report.json` | 特征筛选报告 |
| `outputs/final_plan.json` | 目标权重 plan（含 `target_amount` / `planned_buys`） |
| `outputs/final_ops.json` | 执行指令（含 `target_weight` / `target_amount` / `amount`）与收盘后账户状态 |

---

## 命令行参数

### `workbench.py backtest`

| 参数 | 说明 |
|---|---|
| `--train-start` / `--train-end` | 训练区间 |
| `--val-start` / `--backtest-end` | 验证打分与回测区间 |
| `--scores-out` / `--out-curve` / `--out-summary` | 输出路径 |
| `--n-pool` | 目标持仓只数 n（Top-n，默认 20） |
| `--k-hold` | 日度换仓只数 k（默认 4；空仓建仓忽略） |
| `--score-lag` | 打分滞后天数 |
| `--trade-price-col` | 撮合价格列（`open` / `close`） |
| `--commission-rate` | 券商佣金费率 |
| `--benchmark` | 基准指数文件名（默认 `000300.SH.csv`） |
| `--no-benchmark` | 不计算相对基准的超额收益 |
| `--no-panel-metrics` | 不输出基于 `label_return` 的截面 IC/胜率指标 |
| `--panel-metrics-all-score-dates` | 截面指标使用全部带标签数据（默认只用净值曲线日期） |
| `--min-names-panel-ic` | 截面 IC 计算每日最少股票数（默认 10） |
| `--out-panel-daily-ic` | 输出逐日 IC CSV 路径 |
| `--walk-forward` | 启用逐日重训模式（见上文 Walk-forward 说明） |
| `--launcher` / `--nproc` | 分布式训练启动方式 |

`--` 之后参数传递至 `train.py`（如 `--epochs`、`--batch-size`、`--horizon`、`--stock-pool` 等）。

### `workbench.py predict-next`

| 参数 | 说明 |
|---|---|
| `--train-start` / `--train-end` | 训练锚定区间 |
| `--export-scores` | 打分 CSV 路径 |
| `--state-in` | 账户 JSON |
| `--ops-out` | plan JSON 输出路径 |
| `--next-trade-date` | 目标执行日（YYYYMMDD） |
| `--n-pool` | 目标持仓只数 n |
| `--k-hold` | 日度换仓只数 k（空仓建仓忽略） |
| `--score-lag` | 打分滞后天数 |
| `--trade-price-col` | 写入 plan 的执行价格列 |
| `--commission-rate` | 写入 plan 的佣金费率 |
| `--skip-train` | 跳过训练，复用已有打分 |

### `workbench.py execute-next`

| 参数 | 说明 |
|---|---|
| `--plan-in` | 权重 plan JSON |
| `--state-in` | 执行前账户 JSON |
| `--ops-out` | 执行结果 JSON |
| `--trade-date` | 执行日（默认取 plan 中 `next_trade_date`） |
| `--trade-price-col` | 撮合价格列（可覆盖 plan） |
| `--commission-rate` | 佣金费率（可覆盖 plan） |
| `--no-strict-next-trade-csv` | 允许在缺少当日 CSV 时回退（默认关闭） |

### `train.py` 常用参数（经 `--` 传入）

| 参数 | 说明 |
|---|---|
| `--window-len` | 输入序列长度（默认 60） |
| `--horizon` | 预测收益 horizon（默认 3） |
| `--feature-select-mode` | 特征筛选模式（`none` / `ic_prune`） |
| `--rank-loss-weight` | 排序辅助损失权重 |
| `--early-stop-min-epochs` / `--early-stop-patience` | 早停配置 |

---

## 底层脚本

可直接调用 `predict.py`：

```bash
python predict.py --mode plan \
  --scores outputs/wf_scores.csv \
  --data-root "$DL_DATA_ROOT" \
  --next-trade-date 20260529 \
  --n 30 --k 8 --score-lag 1 \
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

`predict.py execute` 模式下，未指定 `--commission-rate` 时使用账户 JSON 中的 `commission_rate`。
