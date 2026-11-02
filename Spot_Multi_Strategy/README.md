# BTC/ETH 量化研究

此目录保留 BTC/ETH 策略、回测、模拟盘和多资产 Alpha 研究。扫链桌面应用已经独立到
[`Nu1LLL/WhaleWatcher`](https://github.com/Nu1LLL/WhaleWatcher)。本仓库的模拟盘不发送真实订单。

## 当前可运行的入口

| 入口 | 用途 |
|---|---|
| `main.py` | BTC/ETH 单品种历史回测，含最后 30% 样本外区间 |
| `paper_trader.py` | 使用公开行情的本地模拟盘 |
| `portfolio_backtest.py` | 多 Alpha 组合研究 |
| `walk_forward.py`、`rolling_validator.py` | 时间滚动验证 |
| `research/` | 数据来源、实验记录与统一研究工具 |

策略信号在 `strategies.py`，资金和风险参数在 `config.py`，成交与成本账目在
`engine.py`，配置文件在 `configs/`。`*_oos.py` 是独立的历史研究脚本，
对应的 `*_report.py` 生成各自报告；它们不等于已获准交易。

历史 4 小时基线 `configs/profit_first_simplified.json` 只计算 EMA200 趋势、
20 根已完成 K 线高点突破和 20 根低点结构止损。2020-01-01 至 2026-08-01
的当前缓存回放中，BTC 全区间 +21.68%、最后 30% +1.41%；ETH 全区间 +22.70%、
最后 30% +4.30%。BTC 最后 30% Sharpe 为 0.39，未过 0.50 门槛，故仅作研究和模拟对照。
该基线不是已验证的实盘策略。

## 运行

在本目录执行：

```bash
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -p 'test_*.py'
python3 main.py --symbol BTCUSDT --interval 4h --start 2020-01-01 --end 2026-08-01 \
  --strategy-config configs/profit_first_simplified.json
python3 paper_trader.py --symbols BTCUSDT ETHUSDT --interval 4h \
  --strategy-config configs/profit_first_simplified.json
```

旧模拟盘状态包含原配置哈希。升级后若提示配置不一致，请用新的 `--state-folder`
开始一份模拟记录；原状态文件保留供核对，不要强行混用两种配置。

默认成本与方向上限见 [`TRADING_COSTS.md`](TRADING_COSTS.md)。历史失败实验的数字和删除范围
记录在 [`RETIRED_STRATEGIES.md`](RETIRED_STRATEGIES.md)。完整旧实现可从 Git 历史恢复，
无需为复现而让失效策略继续出现在当前配置和运行入口。
