# CME 比特币机构期货持仓变化：预注册

注册日：2026-11-02。本文件必须**单独提交并推送**，随后才可读取本实验的资产管理人持仓数值、BTC 价格或交易结果。此前仅检查官方 CFTC 档案的文件存在性、列名、合约标识、报告日期/行数及特殊延迟公告；未读取持仓值或本规则盈亏。本轮只检验下列一套规则，失败即封存，不在同段历史调参或换方向。

## 假说、范围和证据等级

CFTC Traders in Financial Futures（TFF）中 CME 标准比特币期货 `CFTC_Contract_Market_Code=133741` 的 `Asset_Mgr_Positions_Long_All` 减 `Asset_Mgr_Positions_Short_All`，除以 `Open_Interest_All`，是资产管理人**期货净持仓占未平仓合约比率**，不是全市场净现金流。假说：该比例较四份周报前上升时，机构方向性需求可能持续，使后续 BTC 现货收益为正；下降或不变时持现金。资产管理人也可能套保或做基差套利，因此方向并无保证。信号来自 CFTC 参与者持仓，经济/数据来源不同于本仓库既有 Binance 价格趋势、成交量、永续资金费、现货—合约基差、活跃地址和挖矿难度假说，但统计正交性不能假定。[CFTC TFF 解释](https://www.cftc.gov/sites/default/files/idc/groups/public/%40commitmentsoftraders/documents/file/tfmexplanatorynotes.pdf)。

固定交易资产仅 Binance 现货 `BTCUSDT`，不把 ETH 或山寨币事后添加为赢家。本次考察 2019-01-01 00:00 UTC 至 2025-12-31 完整日历日，并在 2026-01-01 00:00 UTC 强制清仓；2018 年 COT/价格仅暖机。覆盖七个完整年度、84 个完整月；不会删 2022 年或其他坏年份。本次是 2026 年预注册的**回顾性历史留出**，不是 2019 年真正前瞻注册；且团队已看过其他 BTC 策略的历史表现。因此即使数字全部达标，`independent_oos=false`，不能宣称完整 Gate 通过或自动实盘。2022–2025 仅称预指定的历史时间留出，不称真正独立 OOS。

## 固定数据、可见时间与完整性

- COT 源：CFTC 官方 TFF **futures only** 年度压缩档案 `https://www.cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip`，`YYYY=2018…2025`，只取 `133741` 标准 CME BTC 行；微型 BTC `133742` 和合并期货/期权、ETH 均不纳入。记录每个 ZIP 的 URL、下载 SHA-256、文件大小、成员、表头、行数、首末报告日及异常；不把原始 ZIP/CSV 写入仓库。CFTC 无逐包官方校验和，本地 SHA 是复现指纹而非官方真实性证明。[官方历史档案](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm)、[字段定义](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalViewable/cotvariablestfm.html)。
- 只接受 `Report_Date_as_YYYY-MM-DD`、`CFTC_Contract_Market_Code`、`FutOnly_or_Combined`、`Asset_Mgr_Positions_Long_All`、`Asset_Mgr_Positions_Short_All`、`Open_Interest_All` 等准确列；合约、交易所、FutOnly 标志、日期、非负有限整数、`OI>0`、周序与重复行均严格校验。若官方行缺失或 schema 不符则 `DATA_FAILURE`，不插值、不替代合约。`as-of` 是报告持仓日，**不是公布日**。
- 普通行仅在报告日 `d+14` 日的 UTC 00:00 后视为可用，下一日 00:00 才交易。为避免事后偷看延迟公布：2018-12-24 至 2019-03-05、2023-01-31 至 2023-03-14、2025-09-30 至 2025-12-23（均含端点）的报告行一律**不产生信号**，不事后挑选其中盈亏好的周；上一份信号若超过 21 个日历日则持现金。此项统一的黑名单是源可用性屏蔽，不是市场择时。CFTC 对这些停摆/故障补发有公告，但未给所有历史周报的逐期发布日期；`d+14` 对普通节假日偏保守，却仍不能证明覆盖未公告异常。[历史特殊公告](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm)、[2019 年补发](https://www.cftc.gov/PressRoom/PressReleases/7864-19)。
- 价格源：Binance 官方现货 `BTCUSDT` 1d UTC 月归档及逐包 `.CHECKSUM`，需要 2018-12 至 2026-01；2026-01 只用首根日 K 开盘作清仓估值。逐包 SHA-256、ZIP 成员、12 列、ms/us 时间单位、UTC 每日网格、OHLC、非负成交额与缺口全部检验。原始价格仅在内存；若官方归档不可得或正式时段缺日、无终点开盘，封 `DATA_FAILURE`，不前填/换源。[Binance 官方归档](https://github.com/binance/binance-public-data)、[使用条款](https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md)。

## 固定信号、执行与风险

每日 00:00 UTC 决策，取届时已满足 `d+14` 可用且未屏蔽的**最新**标准 BTC 报告。只有该报告距决策日不超过 21 个日历日，且它与之前第 1–4 份未屏蔽报告按时间相邻、每相邻报告日期间隔均为 5–10 个日历日时，才计算 `x_d=(Long_d−Short_d)/OI_d`、`delta=x_d−x_(d−4)`。若 `delta>0`，目标为 BTC 现货 100% 账户权益；若 `delta<=0` 或信号不可用，目标为 USDT 现金 100%。不使用价格、后来的报告修正、机器学习或主观筛选来改信号。信号改变后在**下一日 00:00 UTC 的开盘**执行，期间不做日内交易；若仍同目标，不重复买卖。

初始账户 10,000 USDT，零利息，现货、不做空、不借贷、不加杠杆、不用合约，因此无资金费、借款费或爆仓；USDT/交易所信用风险仍存在。每笔买卖吃单手续费 10bp、逆向滑点 10bp；买价=日开盘×1.001、卖价=日开盘×0.999，费用另按成交名义额扣除。买入以可用现金和费用预算向下取整至 `0.00001 BTC`，剩余保留现金，卖出全部；小于 5 USDT 最低名义额时标为执行失败。每单名义额不能超过此前 30 个完整 UTC 日中位数成交额 1%，否则执行 Gate 失败但仍列原始诊断。额外报告滑点 25bp/边压力，不用压力值重选参数。终点 2026-01-01 00:00 强制卖出并计费。

账户按每日开盘执行后估值，`D` 日日收益为次日开盘权益比减一；2019-01-01 初始权益为 10,000。强制终点卖出计入 2025-12-31 收益。日内保守回撤把当日 high 估峰、low 估谷，并假设峰先于谷；不能解释为真实逐笔路径。同期 BTC 买入持有以同手续费/滑点列作基准，不可据其结果改变规则。若 COT 决策日恰好与价格开盘同刻，严格下一日交易。

## 统一 Gate 与停止规则

主场景：CAGR=`(E_end/10000)^(365.25/N_days)−1` 至少 40%（40–60% 为目标区间，超出 60% 仍如实列示），日收益 Sharpe=`sqrt(365.25)×mean/std(ddof=1)` 严格 >1.5，日内保守最大回撤 ≥−20%，**日收益** Profit Factor=`sum(正收益)/abs(sum(负收益))` 严格 >1.3。零波动或无亏损日要报未定义，不以无穷大闯关。任何一项失败即拒绝。

年度 walk-forward 固定 2019–2025 七折，无训练/重调，至少六折各自同时满足年度净收益>0、日 PF>1、折内保守回撤≥−20%；2022–2025 历史时间留出单列但不能升级为独立 OOS。市场状态只用日初之前 30 个**完整日** BTC 收盘：30 日回报 >+10% 牛、<−10% 熊、其余震荡；分别列日数、持仓日数、净收益、Sharpe、回撤、PF，不能用来实时改仓。84 个完整自然月中正收益比例至少 70%，任何月不低于 −10%；另报最差月与最长连续亏月。对完整日收益（含现金日），固定种子 `20261102`、10,000 条、21 日循环区块 bootstrap；至少 5% 路径**同时**满足 CAGR、Sharpe、日回撤、PF 四项数字 Gate，且 CAGR 5% 分位>0；MC 的日回撤不得替代主场景日内压力回撤。

数据完整、成本/滑点、容量、七完整年、年度折、不同状态、MC、月度稳定及真正独立 OOS 均是完整 Gate 的必要条件；本次历史回看已冻结 `independent_oos=false`。若数据不合格只封 `DATA_FAILURE`、不报告 PnL；若数字/稳健性不合格封 `RESULT`，不得同段历史微调方向、窗口、费用、删年、选资产或加杠杆补低 Sharpe。若将来在**新的前瞻样本**真正过全 Gate，也只能先以 10,000 USDT 账户小仓位模拟，未经用户另行授权绝不自动实盘。
