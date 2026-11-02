# H.4.1 储备余额五币永续：仅修复官方页面解析的数据版本 v2

注册日：2026-11-03。本文件须**单独提交并推送**，其后才能实现 v2 解析器；v2 代码和合成测试也须另行提交、推送，工作树干净后才能读取实际余额序列、币安行情或任何收益。这是 [v1 预注册](../mini_medallion_fed_reserves_crypto_oos/PREREGISTRATION.md) 的**同一经济假说、同一交易规则**的数据解析修复，**不是新经济假说，也不是独立样本外**。v1 在源解析处封存 `DATA_FAILURE`，未形成信号余额序列、未读取币安行情、未计算盈亏；其 [封存](../mini_medallion_fed_reserves_crypto_oos/DATA_FAILURE.json) 和 [复盘](../mini_medallion_fed_reserves_crypto_oos/POSTMORTEM.md) 不得改动。本版不以收益为依据修改任何模型、资产或阈值。

## 冻结的交易假说与数据范围

仅测试已发布的美联储 H.4.1 `Reserve balances with Federal Reserve Banks` 表 1 **末列 Wednesday 余额**四期变化能否指示下一周五币共同风险方向。余额增加→做多；减少→做空；相等→现金。宏观数据仅为信号；**只交易币圈**，固定 Binance USD-M USDT 永续 `BTCUSDT, ETHUSDT, DOGEUSDT, LTCUSDT, BCHUSDT`，不交易美股或外汇。五币不是依据本次结果筛出。

美联储数据仍只用官方 `releaseDates.json` 的 2020-11-01 至 2025-12-31 共 **269** 个实际发布日及对应 `https://www.federalreserve.gov/releases/h41/YYYYMMDD/h41.htm` 历史页；2021–2025 每年必须恰 52 期。禁止改用今日 FRED 数值、其他余额列、其他来源、插值或填补。每期页面、索引、币安 ZIP 和 `.CHECKSUM` 均记录 SHA-256、URL、状态及范围；原始内容和余额序列只在内存，不落盘。

v2 唯一允许的协议变化是 HTML **标题与发布日期的定位**，事先按官方页面结构固定如下：

1. 精确标签 `Reserve balances with Federal Reserve Banks` 必须在一张表中恰出现一次且该行恰五格。该表的**紧邻两个前置元素兄弟节点**必须分别是 `<p>`，规范化空白后文本精确为 `1. Factors Affecting Reserve Balances of Depository Institutions (continued)` 和 `Millions of dollars`。不得按“第 3 张表”硬选，不作全页宽松包含匹配，不接受其他标题、单位或表格。
2. 全页所有 `<title>` 中，须**恰有一个**同时含 `H.4.1` 和完整英文年月日；该日期须与官方索引及 URL 发布日一致。允许页面另有无关的 `<title>Lock</title>`；不再从表前截断 2,400 字符寻找发布日期。
3. 目标行末格的 `header` 或 `headers` ID 必须全部解析到该表内唯一单元格，其中**恰一个**所引表头含 `Wednesday` 加完整日期。所指观测日必须是周三且早于发布日；只读末格非负、有限、格式明确的百万美元余额。行、单位、表头、发布日期、完整性任一不符即 `DATA_FAILURE`。
4. 旧版、新版及节假日 HTML 布局须用**无真实余额数值的结构性夹具**覆盖，并保留负例；夹具不得把不存在的 `Table 1.` 或表前邻近发布日期当作有效页面。全 269 页正式检查仍一页不缺、不能按样本排除失败页面。

## 完全继承、不准变动的执行与 Gate

2020-12 行情/波动率暖机；正式窗口为 2021-01-01 00:00 至 2026-01-01 00:00 UTC，2021–2023 开发期、2024–2025 仅为**历史时间留出**，不是独立 OOS。每周一 00:00 UTC 仅使用 `release_date <= 周一 00:00 − 4 天` 的最新一期与其向前第四期；延期至周五/周一发布的值不能提前用于该周。周一 00:00 五币等权调仓；初始 10,000 USDT、现金利息零、终点强制平仓。仓位仅用以前 30 个完整 UTC 日五币等权篮子每日 20:00 4h 交易 K 收盘回报的 `ddof=1 × sqrt(365.25)` 波动率，`gross=min(3, 0.30/vol)`，每腿 `gross/5`。不以杠杆修补 Sharpe；无效波动率直接封存。同风险永远做多五币仅作固定对照。

币安仍用官方 USD-M 2020-12 至 2025-12 月度 4h `klines`、`markPriceKlines`、`fundingRate`，加 2026-01-01 首根**仅开盘**及终点资金费；每包严格校验官方 `.CHECKSUM`。只有月档恰缺整 UTC 日的六根 bar 才准用同 schema 官方日档补齐，其他缺口/异常及资金费缺失均 `DATA_FAILURE`。先按旧仓结算实际资金费，再交易；开仓和平仓每笔手续费 10bp、每侧滑点基础 10bp，另报 25bp 压力。数量向零取整：BTC/ETH 0.001、DOGE 1、LTC/BCH 0.01；单笔至少 5 USDT，且不超过此前 30 完整日成交额中位数的 1%，否则执行或容量 Gate 失败。按 4h mark 高低作有利先、不利后保守回撤；权益≤0 或最不利权益低于名义仓位 5% 即风险失败。历史盘口冲击和真实保证金阶梯仍未获证明。

完整 Gate 仍为：五年 CAGR ≥40%（目标 40–60%，超过 60% 如实披露）、日净收益 Sharpe **>1.5**、保守 4h 最大回撤不超过 20%、日 Profit Factor **>1.3**；2021–2025 固定年度 walk-forward 至少四年同时收益>0、日 PF>1、年内保守回撤≤20%；60 月正收益≥70% 且无月份低于−10%；BTC 前 30 完整日涨跌 >+10% / <−10% / 中间的牛熊震荡三态全部报告；对 1,826 日日收益用种子 `20261103` 做 10,000 条 21 日循环区块 Monte Carlo，≥5% 路径同时达四项主数值 Gate 且 CAGR 第 5 百分位>0。全部数据、成本、资金费、容量、执行、风险、年度折、市场状态和 MC 必须可核；`independent_oos=false`、证据等级固定 `preregistered_retrospective_historical_holdout`，所以本次即使历史数值全部过关也**不能称完整 Gate 通过或自动实盘**。

## 一次性停止规则

只新增隔离的 v2 页面读取器、测试和 runner，**不修改** v1 解析器、runner、封存报告或用户原有成熟策略、futures/hybrid/rotation/paper_state。数据无效封本目录 `DATA_FAILURE.json`，意外代码错误封 `CODE_FAILURE.json`，有效结果无论好坏封 `RESULT.json`，只运行一次并推送。再次失败不得顺手修补后重试本版，也不得根据数值反向、改窗口/币种/杠杆/成本或删年份；进一步修复须另注册、另定版本。真正前瞻独立 OOS 通过全部 Gate 后也只能先在 10,000 USDT 账户小仓位模拟，未经新授权不得实盘。
