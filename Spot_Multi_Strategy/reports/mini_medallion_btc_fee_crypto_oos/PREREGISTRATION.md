# BTC 链上手续费四周变化 → 五币永续多仓/现金：预注册

注册日：2026-11-03。**本文件须先独立提交并推送；数据、执行和测试代码再单独提交并推送，工作树干净后才能首次读取本信号的历史数值、对应行情/资金费或盈亏。**只做这一条此前未正式检验的经济假说；正式结果只能跑一次，不因失败反向、调窗、换币或修改成本。原有成熟策略和既有研究均只读。

## 机制与证据等级

假说：BTC 用户愿为有限区块空间支付的原生币手续费总额，反映可结算需求与市场参与强度；最近四周大于此前四周时，下周持有等权五币多仓，否则持现金。[Coin Metrics 指标定义](https://docs.coinmetrics.io/network-data/network-data-overview/fees-and-revenue/fees)是每日所有交易手续费的 BTC 总量，排除新发行；其[日指标时间语义](https://github.com/coinmetrics/docs-website/blob/master/asset-metrics/asset-metrics-overview.md)是 00:00 UTC 标记此前 24 小时。区块空间竞争也受拥堵、矿工供给、减半、铭文等非投资需求影响，手续费可能滞后价格；机制**不证明**下周收益可预测。它与此前价格、资金费、基差、稳定币发行、难度和地址数信号的经济来源不同，统计独立性尚未证实。

Coin Metrics 免费接口返回今天的历史版本，没有每个点的初次公开时间与当时版本。周三 D 的假设可见时刻固定为 **D+1 天 00:00 UTC**（周四），只是回顾性代理，并非已核实发布时点。证据等级固定 `retrospective_current_vintage_btc_fee`，`independent_oos=false`。2021–2023 为时间开发段，2024–2025 为回顾性留出；不是前瞻独立 OOS。完整 Gate 因此即使数值达标也不通过，不可直接部署实盘。[Coin Metrics Community 免费层](https://github.com/coinmetrics/product-docs/blob/master/docs/access-our-data/api/README.md)和[Binance Vision 数据条款](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)均限制非商业研究；未来前瞻模拟及商用/实盘须另获适当许可或独立合法数据，不能把本研究数据流直接转为赚钱交易流。

## 固定数据、信号与执行

只读 `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics`，参数 `assets=btc&metrics=FeeTotNtv&frequency=1d&start_time=2020-10-01&end_time=2025-12-31&page_size=10000&paging_from=start`，遇官方 `next_page_url` 完整分页；不得换字段、换源、补值。要求 2020-10-01 至 2025-12-31 共 **1,918** 个 UTC 零点日点，BTC 资产/时间唯一且连续，`FeeTotNtv` 有限、非负；HTTP、JSON、字段、单位、日期、分页任一不合封 `DATA_FAILURE`。每页只存 URL、HTTP 状态、SHA-256、字节数、行数和覆盖日，原始 JSON/每日手续费不落盘；报告须保存由 271 个周信号组成的紧凑方向位串以便复算执行路径。若来源后来修订，单靠哈希仍不能复原首次原始日值，这也是本研究可复现性的明确限制。

从 2020-10-28 至 2025-12-31 每个周三 D，共 **271** 个，固定计算 `value(D)=Σ FeeTotNtv[t]`，`t=D-27..D`，即以 D 00:00 UTC 结尾的最近 28 个完整日区间。每周一 M 00:00 UTC，只使用假设可见时间 `<=M-4天` 的最新周三 D 的 `value(D)` 与四周前 `value(D-28天)`；前者**严格大于**后者→五币等名义权重做多，等于或小于→现金。两段 28 日严格相邻、不重叠：后段 D-27..D，前段 D-55..D-28。2021-01-04 首次决策使用 2020-12-30 与 2020-12-02；不使用 2021-01-01 后才生成的点。预热或前四期不足即封数据失败。

为复用此前冻结的 4h 执行账本而不修改旧研究代码，新增一个**确定性、仅内存**的 long/cash 适配器。设连续周三原始滚动总量为 `R_i`，前四期 `V_i=0`，从第五期起 `V_i=V_(i-4)+1[R_i>R_(i-4)]`。旧账本只比较最新 `V_i` 与四期前 `V_(i-4)`，故其目标方向严格为 `1`（手续费上升）或 `0`（否则），绝不产生 `-1` 做空；`V` 只是执行接口编码，报告和经济解释始终以原始 `R` 的相邻 28 日比较为准。合成单元测试必须覆盖上升、下降、相等、首个周一和未来数据不可见。

**交易仅限加密货币**：沿用先前冻结的 Binance USD-M USDT 永续五币 `BTCUSDT, ETHUSDT, DOGEUSDT, LTCUSDT, BCHUSDT`；2021-01-01 至 2026-01-01 UTC 整五年，2020-12 行情/波动率预热。行情、mark 和真实资金费仍取币安官方 4h 月度 `klines`、`markPriceKlines`、`fundingRate` 加 2026-01 终点开盘及 00:00 资金费，每包核 `.CHECKSUM`；只在月档**恰缺完整 UTC 日六根 bar**时用同币官方 4h 日档补齐，零散缺口或时钟/资金费异常即失败。原始 ZIP/CSV 不落盘，来源指纹和补档日入报告。

初始 10,000 USDT，现金零息。每周一按**此前 30 个完整 UTC 日**五币等权篮子日收盘收益的样本标准差 `ddof=1×sqrt(365.25)` 估年化波动，日收盘为 20:00 UTC 4h 交易 K 收盘；`gross=min(3.0,0.30/vol)`，每币名义 `gross/5`；无信号全现金，最大杠杆 3×。固定对照为同币、同波动目标和同成本的始终做多。旧仓资金费先于周一调仓；成交以 trade 4h 开盘，估值与不利压力用 mark 4h 开高低。每侧手续费 10bp、基础滑点 10bp，另报 25bp 滑点压力；官方实际资金费按旧仓方向入账。BTC/ETH 向零取整至 0.001、DOGE 至 1、LTC/BCH 至 0.01；不足 5 USDT 不成交并记执行 Gate 失败。每单不得超此前 30 完整日成交额中位数的 1%；权益≤0 或 4h 不利权益低于名义仓位 5% 为风险失败。历史保证金阶梯、盘口冲击、交易所信用风险尚未充分覆盖。

## 一次性 Gate 与停止

主场景五年 CAGR≥40%（目标 40–60%，>60% 照实报告）、日净收益 Sharpe>1.5、保守 4h 最大回撤不超过 20%、日 Profit Factor>1.3；2021–2025 五个固定年度 walk-forward 折中至少四年同时净收益>0、日 PF>1、年内保守回撤≤20%；60 个月正收益≥70% 且无月低于−10%；按此前 30 完整日 BTC 涨跌 >+10% / <−10% / 其余列报牛熊震荡三态；种子 `20261103` 做 10,000 条 21 日循环区块 Monte Carlo，至少 5% 路径同时过四个主数值门槛且 CAGR 第 5 百分位>0。成本、资金费、容量、最小订单、维持保证金、终点空仓、三态齐全和独立 OOS 也均为 Gate；当前 `independent_oos=false` 使完整 Gate 不可能通过。日 PF 为正日收益绝对和/负日收益绝对和。

仅新增隔离的 Coin Metrics 读取器、runner、测试和本目录报告；不改成熟策略基线、用户 futures/hybrid/rotation/paper_state、旧报告或旧研究代码。预注册与代码必须分开提交并推送。正式只跑一次：源不合格封 `DATA_FAILURE.json`，非预期代码错误封 `CODE_FAILURE.json`，有效结果无论好坏封 `RESULT.json`；然后只读审计并推送。禁止在同一历史上优化或以杠杆掩盖 Sharpe。只有未来新鲜前瞻点在所有 Gate 真通过，才可讨论 10,000 USDT 账户小仓位模拟；未经另行授权绝不实盘。
