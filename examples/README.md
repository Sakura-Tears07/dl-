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
