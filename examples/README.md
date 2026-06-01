# 账户状态 JSON 说明

本目录提供 `workbench.py predict-next` 与 `execute-next` 所需的账户状态示例（`--state-in`）。

---

## 文件

| 文件 | 用途 |
|---|---|
| `state_empty.json` | 初始空仓：仅现金，无持仓 |
| `state_holding.json` | 已有持仓：含可卖与锁仓仓位 |

---

## 字段定义

| 字段 | 类型 | 说明 |
|---|---|---|
| `cash` | number | 可用现金 |
| `lot_size` | integer | 最小交易单位，A 股通常为 100 |
| `sellable` | object | 可卖持仓，`{ "ts_code": 股数 }` |
| `locked` | object | T+1 锁仓持仓，格式同上 |
| `commission_rate` | number | 券商佣金费率（小数，如万三为 0.0003） |
| `position_status` | string | 可选，`empty` / `holding`；最终以股数字段为准 |

---

## 映射规则

1. 将当前可用现金写入 `cash`。
2. 执行日可卖出的仓位写入 `sellable`。
3. 受 T+1 约束、执行日不可卖的仓位写入 `locked`。
4. 股数应为 `lot_size` 的整数倍。
5. 证券代码需含交易所后缀（如 `600000.SH`、`000001.SZ`）。

---

## 已有持仓下的执行口径（target_tracking）

- `predict-next` 会按 `budget_nav = cash + 持仓市值` 计算 `target_amount`，不是只看现金。
- `execute-next` 采用“先卖后买”的目标跟踪：
  0. 先卖出持仓中最低分 `k` 只（`--k-hold`，默认 4）；
  1. 若卖后持仓只数不足 `n`，可新增超过 `k` 的名称来补齐到 `n`；
  2. 在保留名称内按目标权重先减仓/清仓超配仓位（整手）；
  3. 再用“剩余现金 + 卖出回笼资金”买入/加仓低配仓位；
  4. 允许对已持有标的直接加仓或减仓。
- 因有整手与费用约束，最终成交股数会与理想目标股数有少量偏差。

---

## 调用示例

权重规划：

```bash
python workbench.py predict-next \
  --data-root "$DL_DATA_ROOT" \
  --train-start 2016-01-01 \
  --train-end 2026-05-28 \
  --export-scores outputs/wf_scores.csv \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_plan.json \
  --next-trade-date 20260529 \
  --n-pool 30 --k-hold 8 --score-lag 1 \
  --trade-price-col open --commission-rate 0.0003 \
  --skip-train
```

开盘执行：

```bash
python workbench.py execute-next \
  --data-root "$DL_DATA_ROOT" \
  --plan-in outputs/final_plan.json \
  --state-in examples/state_empty.json \
  --ops-out outputs/final_ops.json \
  --trade-date 20260529
```
