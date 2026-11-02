# USDT+USDC 净供给四周变化 → 五币永续多空：预注册

注册日：2026-11-03。**本文件必须单独提交并推送，随后另行提交并推送数据/执行/测试代码，且工作树干净，才可读取两条稳定币供给数值、该规则的币价/资金费或盈亏。**此前仅核对官方接口文档、ID、JSON 字段、日期连续性和样本数量，未看供给数值或该规则收益。本轮只测试下述一条、此前未正式测试的经济假说；失败直接封存，不反向、调窗口或换资产。它与已测的价格形态、主动成交流、资金费/基差、BTC 挖矿/活跃地址、维基注意力和联储银行准备金在经济来源上不同，但统计独立性未知。

## 机制、先验反证与证据等级

待检验假说：USDT 与 USDC 合计流通数量四周净扩张，是加密美元一级市场流动性/风险需求增强的缓慢代理；净收缩则相反。[Ante–Fiedler–Strehle 原始事件研究](https://www.blockchainresearchlab.org/wp-content/uploads/2020/05/The_Influence_of_Stablecoin_Issuances_on_Cryptocurrency_Markets_BRL_Working_Paper_No_11.pdf)只在较短发行事件窗口发现关联，**没有证明本规则的周度延迟预测力**；[美联储关于一、二级市场的研究](https://www.federalreserve.gov/econres/notes/feds-notes/primary-and-secondary-markets-for-stablecoins-20240223.html)说明铸币不等于资金立即买入币；[Lyons–Viswanath-Natraj 原始研究](https://www.nber.org/papers/w27136)未找到 Tether 发行对 BTC/ETH 价格的系统效应，是重要反证。流通量也可能内生地响应币价或跨链迁移，不预设因果关系。

数据为今天可下载的 DefiLlama 回填历史，不含每个历史点的首次公布时刻或原始当时版本。即使人为滞后七天仍无法证明 2021 年当时可见；证据等级固定 `retrospective_backfilled_supply_proxy`，`independent_oos=false`。2021–2023 为开发期、2024–2025 仅为历史时间留出。该实验最多识别前瞻采集候选，**不能凭历史结果宣称完整 Gate 通过或进入自动交易**。

## 固定来源、样本、信号与执行

仅从 [DefiLlama 免费稳定币 API](https://github.com/DefiLlama/api-docs/blob/main/llms-free.txt) 读取 `https://stablecoins.llama.fi/stablecoins` 目录核对 ID 1=`USDT`、ID 2=`USDC`，以及 `https://stablecoins.llama.fi/stablecoincharts/all?stablecoin=1` 和 `?stablecoin=2` 两条历史序列。只取各 UTC 日期 `totalCirculating.peggedUSD` **原始流通数量**，不取 `totalCirculatingUSD` 或价格换算。2020-10-28 至 2025-12-31 须各有 **1,891** 个 UTC 零点日点、无缺失/重复、数值有限且正；HTTP/JSON/schema/日期/ID/单位任一不符封 `DATA_FAILURE`，不插值、不换源。逐响应存 URL、HTTP 状态、SHA-256、字节数、行数和覆盖范围；原始 JSON/每日供给不落盘。

固定取此区间每个周三 UTC 日期 D 的 `USDT+USDC` 数量，共 **271** 期；由于原始首次公布时间缺失，研究中**假定最早可见时间为 D+7 天 00:00 UTC**，明确这是保守代理而不是真实发布时间。每周一 M 00:00 UTC 决策，只用该代理可见时间 `<= M−4 天` 的最新周三期与其**往前第四期**；前者严格大于后者→下一周五币等权做多，严格小于→等权做空，相等→现金。例如 2021-01-04 首次周一只能用 2020-12-23 与 2020-11-25，不能用 2020-12-30。无足够前四期直接封数据失败，不制造数值。周一开盘调仓，2026-01-01 00:00 UTC 强制平仓。

**只交易加密货币**：固定 Binance USD-M USDT 永续 `BTCUSDT, ETHUSDT, DOGEUSDT, LTCUSDT, BCHUSDT`，2021-01-01 至 2026-01-01 UTC 整五年，2020-12 行情/波动率暖机。五币沿用先前冻结的候选池，不按稳定币信号结果筛选；不交易股票、外汇或稳定币本身。行情、mark、资金费仍用币安官方 4h 月度 `klines`、`markPriceKlines`、`fundingRate` 加 2026-01 终点仅开盘与 00:00 资金费；每包核对官方 `.CHECKSUM`。月档只在**恰缺整 UTC 日六根 bar**时可用官方同币 4h 日档补齐；零散缺口、时钟/资金费异常一律数据失败。原始 ZIP/CSV 不落盘，逐包指纹和确切补档日期入报告。

初始 10,000 USDT、现金零息。周一按**之前 30 个完整 UTC 日**五币等权篮子日收盘到收盘收益的样本标准差 `ddof=1×sqrt(365.25)` 计算年化波动，日收盘取 20:00 UTC 4h 交易 K close；`gross=min(3.0,0.30/vol)`，每腿名义权重 `gross/5`，多空对称，最多 3×，不以杠杆补救弱 Sharpe。固定对照为同一波动目标、同币、同成本的始终做多。旧仓资金费先于调仓，成交以 trade 4h 开盘、估值及不利压力以 mark 4h 开高低价。每笔手续费 10bp、基础每侧滑点 10bp，另报 25bp 压力；实际官方资金费按旧持仓符号计。BTC/ETH 向零取整至 0.001、DOGE 至 1、LTC/BCH 至 0.01；不足 5 USDT 不成交且记执行 Gate 失败。每单≤前 30 完整日成交额中位数的 1%，否则容量 Gate 失败；权益≤0 或 4h 不利权益低于名义仓位 5% 即风险失败。历史保证金阶梯、盘口冲击、稳定币/交易所信用风险未获充分覆盖。

## 统一 Gate 与一次性停止

主场景五年 CAGR≥40%（目标 40–60%，>60% 照实报告）、日净收益 Sharpe>1.5、保守 4h 最大回撤不超过 20%、日 Profit Factor>1.3；2021–2025 五个固定年度 walk-forward 折中至少四年同时净收益>0、日 PF>1、年内保守回撤≤20%；60 月正收益≥70% 且无月低于−10%；按前 30 完整日 BTC 涨跌 >+10% / <−10% / 其余的牛熊震荡三态全部列报；用种子 `20261103` 做 10,000 条 21 日循环区块 Monte Carlo，至少 5% 路径同时满足四个主数值门槛且 CAGR 第 5 百分位>0。成本、资金费、容量、最小订单、维持保证金、终点空仓、完整三态与独立 OOS 都是 Gate 条件；`independent_oos=false` 固定使**完整 Gate 不可能在这份历史上通过**。日 PF 是正负日收益绝对和之比，不是逐笔 PF。

本研究仅新增隔离的读取器、runner、测试和本目录报告；不得修改原有成熟策略基线、用户 futures/hybrid/rotation/paper_state 或既有正式实验文件。预注册与代码必须分开提交并推送。只能正式运行一次：源数据不合格封 `DATA_FAILURE.json`，意外代码错误封 `CODE_FAILURE.json`，有效结果无论好坏封 `RESULT.json`；随后只读审计、提交推送。不得对同一历史反向、改四期/币种/滞后/成本/波动目标、删年份、放松 Gate 或因结果改用另一个稳定币供给字段。若未来新鲜前瞻点时数据在所有 Gate 下真通过，也只可先以 10,000 USDT 账户小仓位模拟，未经新授权绝不实盘。
