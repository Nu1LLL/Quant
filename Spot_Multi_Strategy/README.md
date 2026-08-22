# 现货多策略量化 MVP

这是一个只做多、无杠杆的现货研究与模拟交易模型。

当前包含两套互补策略：

1. 趋势突破：上涨趋势中突破前20根K线最高价时进场。
2. 趋势回调：长期上涨趋势中出现短期超卖时进场。

另提供一份独立的市场状态自动切换配置：

3. 震荡均值回归：ADX较低、EMA较平时，价格跌破布林带下轨并超卖后进场。

自动切换配置位于 `configs/regime_switching.json`。ADX高于25时只允许趋势策略，ADX低于20且EMA平缓时只允许震荡策略，中间区域不建立新仓。

两套策略使用独立资金账户，然后汇总为一条组合权益曲线。模型包含：

- Binance公开K线分页下载和本地缓存
- 数据完整性检查
- EMA、Donchian通道、ATR、RSI和布林带指标
- 下一根K线开盘成交，防止使用未来数据
- 手续费和滑点
- ATR止损和趋势策略移动止损
- 单笔风险控制和最大仓位限制
- 完整交易记录、月度统计和样本外报告
- 按品种、周期和日期分别保存报告，不互相覆盖
- 自动研究验收门槛，不合格时明确禁止进入实盘
- 不联网的自动化测试

## 安装依赖

```bash
python3 -m pip install -r requirements.txt
```

## 运行测试

```bash
python3 -m unittest discover -s tests -v
```

## 运行回测

```bash
python3 main.py \
  --symbol BTCUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01
```

不传 `--end` 时，程序使用当前已经结束的K线。报告和交易记录会保存到 `reports/`。

报告文件名会包含 `default` 或参数文件名，每份报告同时保存完整参数快照，默认参数和优化参数不会互相覆盖。

可以修改两个策略的资金比例，例如只测试趋势突破策略：

```bash
python3 main.py \
  --symbol BTCUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --trend-weight 1 \
  --pullback-weight 0
```

## 当前验收状态

固定使用2020-01-01至2026-08-01的历史数据，同一套默认参数在BTC和ETH上都能运行且保持正收益，但最后30%样本外区间的Sharpe都低于0.50。

因此当前版本的定位是：

- 多策略研究框架：可用
- 历史回测和报告：可用
- 模拟盘候选策略：暂未通过
- 真实资金交易：禁止

2026-08-23完成第一轮稳健优化：60组候选参数、BTC与ETH各4个连续验证折，共480次开发区间回测。开发区间最佳参数为EMA100、30根突破、15根离场、2ATR初始止损、3ATR移动止损，并自动淘汰了回调策略。

最后20%隐藏测试结果：

- BTCUSDT：-3.10%，Sharpe -0.73，未通过。
- ETHUSDT：+7.28%，Sharpe 1.26，但只有26笔交易，未达到30笔门槛。

因此优化器正常工作，但当前策略结构仍然不够稳定。

## 稳健参数优化

优化器只使用前80%历史进行多时间折验证，最后20%在参数确定以后只验收一次：

```bash
python3 optimizer.py \
  --symbols BTCUSDT ETHUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --trials 60
```

候选排名、每个验证折、最佳参数和隐藏测试结果保存在 `optimization_reports/`。

注意：隐藏测试集一旦查看，就已经被“用过”。不能根据隐藏结果继续调参后又把同一区间称为隐藏测试；下一轮策略结构研究应使用滚动验证，并等待新的未来数据做最终验证。

使用优化后参数重新回测：

```bash
python3 main.py \
  --symbol BTCUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --strategy-config optimization_reports/optimized_strategy.json
```

## 市场状态自动切换与滚动验证

运行趋势/震荡自动切换版本：

```bash
python3 main.py \
  --symbol BTCUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --strategy-config configs/regime_switching.json
```

对固定参数执行BTC与ETH连续滚动验证：

```bash
python3 rolling_validator.py \
  --symbols BTCUSDT ETHUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --folds 6 \
  --strategy-config configs/regime_switching.json
```

这些区间已经被查看，因此滚动验证只能用于研究，不能重新称为隐藏测试。最终可靠性必须由未来新增数据和模拟盘提供。

2026-08-23第一次固定参数滚动验证结果：

- BTC与ETH各6个连续验证折，共12个“品种×时间折”。
- 组合盈利折比例75%，折收益中位数1.26%，Sharpe中位数0.96。
- 最差折收益-1.90%，最差最大回撤-2.37%。
- 趋势突破96笔，滚动累计盈利1059.27 USDT。
- 震荡均值回归只有11笔，滚动累计盈利11.08 USDT。

组合层指标通过，但震荡子策略未达到至少20笔的样本门槛，因此最终结论仍是未通过、禁止实盘。

## 稳健性与压力测试

针对只保留已验证趋势逻辑的保守候选参数：

```bash
python3 robustness.py \
  --symbols BTCUSDT ETHUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --strategy-config configs/trend_conservative.json \
  --simulations 5000
```

测试包括双倍成本、延迟成交、参数上下扰动20%、删除最佳两笔交易和交易回报蒙特卡洛。即使全部通过，也只能进入未来模拟盘，不能直接实盘。

对保守参数附近的11组参数执行同一套滚动门槛筛选：

```bash
python3 candidate_selector.py \
  --symbols BTCUSDT ETHUSDT \
  --interval 4h \
  --start 2020-01-01 \
  --end 2026-08-01 \
  --folds 6 \
  --base-config configs/trend_conservative.json
```

只有通过门槛时才会生成 `configs/trend_champion.json`，它最多只能成为模拟盘候选。

## 公开行情只读模拟盘

模拟盘不需要API密钥，代码中没有真实下单接口：

```bash
python3 paper_trader.py \
  --symbols BTCUSDT ETHUSDT \
  --interval 4h \
  --capital 5000 \
  --strategy-config configs/trend_champion.json
```

首次运行只建立起点，不会追补过去的信号。状态、事件和权益保存在 `paper_state/`。如果漏掉一根4小时K线，程序会停机，而不是利用历史价格伪造实时成交。

紧急停止：在 `paper_state/` 中创建名为 `EMERGENCY_STOP` 的空文件。

检查是否达到实盘准备门槛：

```bash
python3 release_gate.py
```

必须同时满足：历史滚动验证与压力测试通过、连续模拟至少90天、至少12笔完整模拟交易、扣费后净盈利、利润因子不低于1.10、最大回撤不超过10%、没有K线断档且状态持续更新。任意一项不满足，程序都会显示“禁止实盘”。通过也只代表可以开始编写和验证极小资金执行模块，不会自动连接真实账户。

## 重要说明

- 这是研究和模拟交易版本，不会向交易所发送真实订单。
- 默认手续费率为单边0.10%，滑点率为单边0.05%，实际费用需要按账户等级修改。
- 历史盈利不代表未来盈利。
- 不要使用房租、生活费、借款或其他急用资金交易。
- 实盘前必须完成滚动验证、压力测试和至少90天且12笔完整交易的未来模拟运行。
