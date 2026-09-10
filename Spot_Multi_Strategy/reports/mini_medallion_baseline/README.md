# Mini-Medallion 基线复现报告

生成时间（UTC）：见 `baseline_summary.json` 的 `generated_at_utc`。
本报告**未修改任何策略代码、参数或成本假设**，只是重跑现有 `main.py`，
把 2020-01-01 至最新已收盘K线（2026-09-10 16:00 UTC）的结果固化下来，
作为后续 alpha 研究平台的对照基线。

## 数据与成本假设（均为仓库现有默认值，未改动）

| 项目 | 值 |
|---|---|
| 品种 | BTCUSDT, ETHUSDT |
| 周期 | 4h |
| 起始 | 2020-01-01 |
| 结束（最后已收盘K线） | 2026-09-10 16:00 UTC |
| fee_rate | 0.001（单边0.10%） |
| slippage_rate | 0.0005（单边0.05%） |
| execution_delay_bars | 1（下一根K线开盘成交） |
| risk_per_trade | 0.005 |
| 初始资金 | 5000 USDT |

## 复现命令

```bash
python3 main.py --symbol BTCUSDT --interval 4h --start 2020-01-01
python3 main.py --symbol ETHUSDT --interval 4h --start 2020-01-01
python3 main.py --symbol BTCUSDT --interval 4h --start 2020-01-01 --strategy-config configs/regime_switching.json
python3 main.py --symbol ETHUSDT --interval 4h --start 2020-01-01 --strategy-config configs/regime_switching.json
```

不传 `--end` 时程序自动取最新已完全收盘的K线，所以未来重跑同样的命令，
"结束时间"会自动前移，属于预期行为，不代表基线被篡改。

## 结果总表

完整数值见 `baseline_summary.csv` / `baseline_summary.json`。

| 品种 | 配置 | 区间 | 最终资产 | 总收益 | 年化 | 最大回撤 | Sharpe | 利润因子 | 胜率 | 交易数 | 买入持有收益 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| BTCUSDT | default（趋势突破+回调） | 全历史 | 6248.25 | 24.97% | 3.39% | -2.63% | 1.343 | 2.114 | 38.96% | 154 | 969.17% |
| BTCUSDT | default | 样本外（最后30%） | 5171.34 | 3.43% | 1.69% | -2.25% | 0.829 | 1.560 | 40.00% | 45 | 42.94% |
| BTCUSDT | regime_switching（趋势+震荡自动切换） | 全历史 | 6077.55 | 21.55% | 2.96% | -3.38% | 1.140 | 2.215 | 39.22% | 102 | 969.17% |
| BTCUSDT | regime_switching | 样本外 | 5031.54 | 0.63% | 0.31% | -3.02% | 0.187 | 1.109 | 36.11% | 36 | 42.94% |
| ETHUSDT | default | 全历史 | 5809.37 | 16.19% | 2.27% | -4.37% | 0.836 | 1.590 | 32.00% | 175 | 1800.87% |
| ETHUSDT | default | 样本外 | 5280.61 | 5.61% | 2.76% | -3.11% | 1.069 | 1.796 | 25.53% | 47 | 9.68% |
| ETHUSDT | regime_switching | 全历史 | 5811.85 | 16.24% | 2.27% | -4.46% | 0.926 | 2.056 | 35.29% | 102 | 1800.87% |
| ETHUSDT | regime_switching | 样本外 | 5185.66 | 3.71% | 1.83% | -1.93% | 0.991 | 2.076 | 41.67% | 24 | 9.68% |

## 关键结论（对照，非评判）

- 所有权益曲线均无杠杆、无做空，历史最大回撤全部小于5%，符合"现货只做多、低回撤"的设计目标。
- 所有组合的历史总收益、年化收益都**远低于同期买入持有**（BTC/ETH本轮均为大牛市），
  这与仓库 README 中"当前版本定位为研究框架，未获准实盘"的结论一致，符合预期，不是bug。
- `regime_switching` 配置在BTC样本外区间Sharpe跌到0.187、利润因子跌破1.20验收门槛（对照
  仓库既有验收标准，未通过），`default`配置样本外两个品种都通过既有基础门槛。
- 这就是本次"迷你文艺复兴"研究要挑战的基线：**用低相关、可解释的弱alpha信号做分散化，
  在不提高历史CAGR幻觉的前提下，尝试提升稳健样本外Sharpe**，而不是在这两条曲线基础上
  继续调参。

## 测试状态

`python3 -m unittest discover -s tests -v` 全部 64 个用例通过（含期货/轮动/混合策略的历史用例），
详见运行日志。本次基线复现之前未对任何生产代码做改动。
