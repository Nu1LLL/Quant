# 新 USDT 永续合约上市后做空：一次性实验封存

**状态：源时间戳/执行映射失败；没有策略收益结论。** 正式预注册 `d2b23e0`，执行细节补充 `894d2d4`，一次性结果 `f90cba1`。本规则不得在 2021–2025 的相同事件历史上修补、换方向、改参数或重跑。

## 源与样本

官方 Binance 公告目录 2,292 条中，价格盲规则及正文哈希确认 218 条单一 USDT 永续合约上市公告；按同一合约首次公告去重后 216 条，`CVCUSDT` 因开盘前一天已有合约档案而排除，最终 215 条事件，2021–2025 年分别为 24、18、82、49、42 条。开盘窗口所需的 Binance Vision 5 分钟 K 线 ZIP 与官方校验和通过，公告计划时间和实际首根 K 线不一致的 5 条已在预注册前按固定规则处理。资金费源为 223 份月度 ZIP 与 2 次官方历史 REST 查询，均有冻结哈希；每个持仓窗口均含资金费事件。[Binance 公共历史数据说明](https://github.com/binance/binance-public-data)、[USDⓈ-M 资金费历史接口](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History)。

## 一次性运行及失败原因

`2026-11-07T16:47:10Z` 启动正式运行，已验证完整价格、资金费与 BTC 状态源。`2026-11-07T16:51:16Z` 一次性运行写入 `research/results/binance_new_perpetual_post_launch_short_2026_11_08.json`，状态 `sealed_failed_run`，原因为 `LaunchStrategyFailure: funding event is not on the 5m grid`。结果 JSON 的 `metrics=null`、`gate.full=false`。**没有 CAGR、Sharpe、回撤、Profit Factor、交易盈亏或 Monte Carlo 结果；不能说该策略盈利或亏损。**

事后只读诊断在 215 个持仓窗口内找到 **341 条资金费记录**未精确落在 5 分钟 K 线的毫秒起点。例如 AKROUSDT 官方资金费档案的时间为 `2021-01-19T08:00:00.005Z`、`16:00:00.012Z`，而对应 5 分钟 K 线起点为整分钟。差异来自源时间戳的毫秒尾数；本实验的模拟器要求精确相等并按预注册停止。诊断未计算任何策略收益。

## 结论与后续边界

这是**数据时间映射失败**，不是可交易 Alpha 的正面或负面检验。该假说在这段历史上的一次性实验已经封存。未来不同经济机制的研究，应在其正式预注册前预检资金费与 K 线的时间容差和归属规则；不得把本次源修复伪装成同一假说的新 OOS。原有成熟策略、用户 futures/hybrid/rotation/paper_state 与既有正式报告未改动，也没有下单。
