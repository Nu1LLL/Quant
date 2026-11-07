# BTC MVRV 相对一年中位数 → 五币永续多空：预注册

注册日：2026-11-08。此文件必须**单独提交并推送**；之后才能实现并另行提交、推送代码；只有两次提交均已推送、工作树干净，才可首次读取本假说的历史 MVRV 数值、对应行情和盈亏。只检验这一条新规则一次，不因结果调整币种、窗口、方向、成本或时间段。原有成熟策略、旧正式报告和用户无关改动均只读。

## 机制、来源与证据等级

假说：BTC 的市值与链上已实现市值之比 MVRV，相对过去一年中位数偏低时，持币者的账面成本基础为未来风险资产需求提供支撑；偏高时，潜在获利了结压制未来收益。因此周度低于滚动中位数时做多固定五币篮子，高于时做空，相等时持现金。该机制属于链上持币成本基础/估值，不是已测的 BTC 手续费、活跃地址、价格动量、资金费或基差。其经济来源可区别，统计独立性未被证明。Coin Metrics [指标定义](https://docs.coinmetrics.io/network-data/network-data-overview/market/market-capitalization)明确 `CapMVRVCur = CapMrktCurUSD / CapRealUSD`；[免费目录](https://community-api.coinmetrics.io/v4/catalog-v2/asset-metrics?assets=btc&metrics=CapMVRVCur)在预注册前只用于确认 `btc`、`1d`、2010-07-18 至 2026-11-06 可用，**未读历史指标值或本策略收益**。

Coin Metrics 免费接口返回当前修订版，不能证明每个历史日当时发布的确切版本和时刻。固定假设日点 D 在 D+2 天 00:00 UTC 可见；周三 D 的信号仅在下周一 00:00 UTC 交易，仍然只是保守代理，不是已核验的历史发布记录。证据等级 `preregistered_retrospective_current_vintage_btc_mvrv`，`independent_executable_oos=false`，2024–2025 仅为回顾性时间留出；即使全部数值达标，完整 Gate 仍不通过，也不能实盘。免费数据与 Binance Vision 只用于相应条款允许的非商业研究；商用/实盘需另获合法数据权利。[Coin Metrics Community](https://docs.coinmetrics.io/packages/coin-metrics-community-data)、[Binance Vision 条款](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)。

## 冻结数据与信号

只读 `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics`，固定参数 `assets=btc&metrics=CapMVRVCur&frequency=1d&start_time=2019-10-30&end_time=2025-12-31&page_size=10000&paging_from=start`，若有官方 `next_page_url` 则完整分页，不换源、不补值。必须恰有 2019-10-30 至 2025-12-31 的 **2,255** 个唯一、递增 UTC 零点日点，值为有限且严格正的十进制数；缺失、重复、乱序、日期或 schema 异常均封 `DATA_FAILURE.json`。记录每页 URL、状态、次数、字节数、SHA-256 和覆盖日期；不落盘原始每日指标数值。

每个周三 D，从 2020-10-28 至 2025-12-31 共 **271** 个，计算以 D 为末日、含首尾的最近 **365** 个 UTC 日 `CapMVRVCur` 的精确中位数。若当日比值严格小于中位数，下周一固定五币等名义做多；严格大于则做空；相等则持现金。中位数只用当时及此前的值。无训练或阈值学习。执行账本旧接口以四周前周三值的差值决定 `-1/0/+1`，新增确定性内存编码：前四个 `V_i=0`，其后 `V_i=V_(i-4)+direction_i`，其中 `direction_i` 为上面的 `+1/-1/0`；以此保持账本原有仓位、成本和风控不变。报告保存方向位/三态编码指纹供核对，不保存原始 MVRV 值。合成测试覆盖严格不等、相等、中位数奇偶、首个交易周及未来日不可见。

交易仅限 Binance USD-M 五币 `BTCUSDT, ETHUSDT, DOGEUSDT, LTCUSDT, BCHUSDT`，回测 **2021-01-01 至 2026-01-01 UTC** 整五年；2020-12 行情预热。沿用已冻结的官方 4h `klines`、`markPriceKlines`、`fundingRate` 月档和 2026-01-01 终点开盘/资金费，逐包验证 `.CHECKSUM`。只有月档缺恰好完整 UTC 日六根 bar 时可用相同官方日档补齐，其他缺口即数据失败；ZIP、CSV 不落盘。初始权益 10,000 USDT，现金零息。每周一以此前 30 个完整 UTC 日五币等权篮子的样本日波动 `ddof=1 × sqrt(365.25)` 定杠杆 `min(3, 0.30/vol)`，每币等名义，绝不为补 Sharpe 加杠杆，且低于用户 5 倍硬上限。相同权重/波动目标/成本的始终做多为对照。

旧仓先结算真实官方资金费，再周一成交；成交价用 4h trade 开盘，估值和保守不利路径用 4h mark 开高低。每侧 taker 手续费 **10 bp**、基础滑点 **10 bp**，另跑每侧 **25 bp** 滑点压力；数量步长 BTC/ETH `0.001`、DOGE `1`、LTC/BCH `0.01`，订单低于 5 USDT 不成交并记 Gate 失败；单单不超此前 30 个完整日成交额中位数 1%，超限记容量 Gate 失败。权益≤0 或不利权益低于仓位名义 5% 为风险失败。历史真实盘口冲击、保证金阶梯和交易所信用风险未被完整复刻。

## 一次性 Gate 与停止规则

整五年主场景净 CAGR≥40%（目标 40–60%，超过照报）、日净收益 Sharpe>1.5、保守 4h 最大回撤≥−20%、日 Profit Factor>1.3；五个固定年度 walk-forward 折至少四年同时净收益>0、日 PF>1、年内回撤≥−20%；60 月正收益≥70%，无单月<−10%；按此前 30 完整日 BTC 涨跌 >+10% / <−10% / 其余列报牛熊震荡；种子 `20261103`、10,000 条 21 日循环区块 Monte Carlo 至少 5% 路径同时达到四个主数值门槛且 CAGR 第五百分位>0。另列报 2024–2025 回顾性留出、压力成本、做多对照、容量、最小订单、保证金代理、终点平仓、三态覆盖以及真正独立 OOS。任一失败原样封存；不在同历史反向、换币、删年份、降低成本或调中位数窗口。

只新增隔离读取器、适配器、runner、测试和本目录报告。注册与代码分开提交、推送。正式运行仅一次：源不合格封 `DATA_FAILURE.json`，非预期程序错封 `CODE_FAILURE.json`，有效数值封 `RESULT.json`。只有未来真正独立前瞻数据以及所有 Gate 均通过才可讨论 10,000 USDT 账户小仓位模拟；无另外授权绝不实盘。
