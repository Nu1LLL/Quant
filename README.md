# Quant

量化研究代码库。扫链提醒应用已迁至独立私有仓库
[`Nu1LLL/WhaleWatcher`](https://github.com/Nu1LLL/WhaleWatcher)。

## 从哪里开始

| 需求 | 位置 |
|---|---|
| BTC/ETH 策略回测与模拟盘 | [`Spot_Multi_Strategy/`](Spot_Multi_Strategy/) |
| 核心策略与风控 | `Spot_Multi_Strategy/strategies.py`、`config.py`、`engine.py` |
| Alpha、组合与滚动验证 | `Spot_Multi_Strategy/alphas/`、`research/`、`walk_forward.py` |
| 历史对照配置 | `Spot_Multi_Strategy/configs/` |
| 研究脚本 | `Spot_Multi_Strategy/*_oos.py` 和对应 `*_report.py` |
| 成本口径 | `Spot_Multi_Strategy/TRADING_COSTS.md` |
| 已淘汰实验与证据 | [`Spot_Multi_Strategy/RETIRED_STRATEGIES.md`](Spot_Multi_Strategy/RETIRED_STRATEGIES.md) |

行情缓存、回测生成文件和模拟盘状态为本地数据，不在 Git 中。完整运行命令见
[`Spot_Multi_Strategy/README.md`](Spot_Multi_Strategy/README.md)。
