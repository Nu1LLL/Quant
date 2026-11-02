# Mini-Medallion V1.3 — 多资产 + 资金费率 + 按资产分别验收

## 这一轮做了什么

用户明确要求"放开了搞，不要求必须是BTC或者ETH"，具体做了三件事：

1. **扩大资产范围**：从BTC/ETH两个资产扩到BTC/ETH/SOL/BNB四个
   （SOL在Binance现货上市晚于其他三个，2020-08-11才有数据，如实
   保留这个较短历史，不为了对齐而截断其他资产或删除SOL）。
2. **引入真正的另类数据源**：新增A21/A22资金费率alpha，数据来自
   `futures_data.py`已经在抓的Binance永续合约资金费率（8小时结算
   一次），用`merge_asof(direction="backward")`因果对齐到4小时
   现货K线——这是本项目第一个不是"OHLCV数学变换"的信号来源。
3. **A20市场广度、A19相对强弱从占位符/两资产版本变成真正的N资产
   alpha**：新增A23（跨资产相对强弱的多资产泛化版），用outer join
   处理不同资产历史长度不一致的问题（SOL更晚上市不会污染其他资产
   或制造未来数据泄漏）。
4. **新增"按资产分别license"的Gate汇总方式**：承认一个alpha的
   有效性可能本来就是资产特定的（不同资产流动性、参与者结构不同），
   不再强制要求"必须在所有测试资产上都通过"才算数——但**用的是完全
   相同的6项walk-forward标准**，区别只在汇总规则，不是放宽验收本身。

## Alpha库规模

单资产品种×alpha实例数从58个增加到128个（BTCUSDT 31个、ETHUSDT
35个、SOLUSDT 31个、BNBUSDT 31个）。有效独立信号数量（相关矩阵
特征值估算）：BTCUSDT 4.96、ETHUSDT 5.43、SOLUSDT 4.93、BNBUSDT
5.15——比两资产版本的4.2-4.8略有提升，但提升幅度有限，说明新增的
广度/相对强弱/资金费率alpha虽然来自不同的计算逻辑甚至不同的数据源，
真正带来的"新增独立信息"依然有限（详见下文关于A21的具体分析）。

## 结果1：无条件门槛（必须4个资产都通过）——0/35通过

35个alpha名称，在BTC/ETH/SOL/BNB全部4个资产上都要求通过6项
walk-forward标准，**0个通过**（对照：两资产版本是3/31通过）。
这是资产数量增加后的直接数学后果——多一个必须同时满足的约束，
通过率只会更低不会更高，符合预期，不是bug。

之前在两资产版本上通过的A02_ema_distance_100、A07_atr_percentile、
A08_trend_quality这次具体是怎么失败的？答案很直接：**它们在
BTCUSDT和ETHUSDT上依然通过，只是在SOLUSDT和/或BNBUSDT上没通过**
（完整原因见`alpha_gate_results.csv`）。

## 结果2：regime条件门槛——仍然0/3个regime有专家alpha

延续V1.2的方法（因果regime分类、按激活位置切折），4个资产×35个
alpha×3个regime = 420个组合，同样要求"在所有测试资产上都通过"，
**0个通过**，原因和V1.2一致：折IC一致性和PnL集中度脱钩（趋势类
payoff天然lumpy），没有因为多了两个资产而改变。

## 结果3：按资产分别license——BTC/ETH照旧，SOL/BNB一个都没有

这是本轮最直接回应"多资产"要求的部分：不再要求跨资产统一，每个
资产用自己的6折walk-forward历史单独判定。结果（完整数据见
`per_asset_gate_summary.json`）：

| 资产 | 通过的alpha |
|---|---|
| BTCUSDT | A02_ema_distance_100, A07_atr_percentile, A08_trend_quality |
| ETHUSDT | A02_ema_distance_100, A07_atr_percentile, A08_trend_quality |
| SOLUSDT | （一个都没有） |
| BNBUSDT | （一个都没有） |

**关键发现：SOL和BNB不是"没有通过跨资产统一性检验"，而是35个
alpha候选里，没有一个能在SOL或BNB自己的历史上单独通过这6项
标准**——这是一个比"不能泛化"更强的结论。

这个发现和一个完全独立的数据来源互相印证：**既有的、完全没有改动
过的三策略引擎**，单独在4个资产上跑（用同样的`configs/regime_switching.json`，
未改动任何参数）：

| 资产 | 既有引擎Sharpe | 既有引擎最大回撤 | 交易数 |
|---|---|---|---|
| BTCUSDT | 1.14 | -3.38% | 102 |
| ETHUSDT | 0.93 | -4.46% | 102 |
| SOLUSDT | **0.21** | -3.49% | 95 |
| BNBUSDT | **0.46** | -4.46% | 99 |

既有引擎在SOL/BNB上的Sharpe远低于BTC/ETH，这是一个和本次alpha库
完全独立的验证——不是我的35个alpha候选设计得不够好，而是SOL/BNB
这两个资产在这段历史样本里，无论用既有的离散趋势策略还是用这次
新做的连续alpha框架，"趋势跟随"这条路径本身能提取的edge都明显更弱。
这可能和SOL历史更短（统计功效更低）、也可能和这两个资产的市场
微观结构、参与者结构有关，本报告不下定论，只如实记录这个跨方法论
一致的现象。

## 结果4：资金费率alpha（A21/A22）的具体表现——目前最接近"新发现"的信号

A21（资金费率拥挤度反向alpha）在BTC/ETH上的表现值得单独记录：

| 品种 | 折IC中位数 | 折IC为正比例 | 扣费后成本侵蚀比例 | 单折PnL占比 |
|---|---|---|---|---|
| BTCUSDT | 0.0169 | **83.3%**（6折里5折为正） | 14.4%（远低于70%上限） | 100%（唯一超标项） |
| ETHUSDT | 0.0293 | **83.3%** | 10.7% | 83.4%（唯一超标项） |

A21在BTC/ETH上折IC一致性很好（5/6折为正）、成本侵蚀也很低（费率
alpha换手天然不算频繁），**唯独卡在单折PnL占比这一项**——和V1.2
报告里A02/A08面临的问题是同一类（趋势/regime类信号payoff天然
lumpy，PnL集中度检验对这类信号可能系统性过严）。这不构成"通过"，
但作为唯一一个来自全新数据源、且折级方向一致性达到83%的候选，
比V1.2里那些纯OHLCV数学变换出来的候选更值得后续针对"PnL集中度
检验设计"问题解决后重新评估。

## 结果5：组合回测（A/C/F场景，B/D/E因为无条件门槛0通过而跳过）

| 场景 | CAGR | Sharpe | 最大回撤 | 年化换手 |
|---|---|---|---|---|
| A 既有多策略组合（4资产等权） | 1.70% | 0.183 | -1.98% | N/A（旧引擎口径不同） |
| C 买入持有（4资产等权） | 65.29% | 1.047 | -82.12% | 极低 |
| F 按资产分别license的相关性惩罚集成 | 2.24% | 0.385 | -18.21% | 6.33 |

F场景里SOL和BNB全程空仓（没有license到任何alpha），资金闲置在
现金里，实际上是"BTC/ETH corr_penalized集成的50%仓位稀释版"——
Sharpe（0.385）比V1.1/V1.2里纯BTC/ETH两资产版本的corr_penalized
（0.601）更低，正是因为一半资金没有产生任何收益，符合预期，
不是新的负面发现，只是稀释效应的直接体现。

**A场景（既有引擎的4资产等权组合）Sharpe从两资产版本的1.20跌到
0.18**——这印证了上面"SOL/BNB上趋势跟随失效"的结论：即使是完全
没有改动过的既有系统，一旦被迫在SOL/BNB上也分配资金，风险调整后
表现同样大幅下降。这不是新alpha框架的问题，是这两个资产本身在
这段样本里对趋势跟随类方法都不友好。

## 结论

| 检查项 | 结果 |
|---|---|
| 是否引入了真正的另类数据源 | **是**：资金费率（A21/A22），不是OHLCV数学变换 |
| 是否支持任意数量资产（不要求BTC/ETH） | **是**：A20/A23原生支持N个资产，SOL晚上市不会污染其他资产 |
| 是否支持按资产分别验收 | **是**：新增`summarize_per_asset_acceptance` + F场景 |
| 无条件4资产门槛通过数 | 0/35（数学上必然比2资产版本的3/31更严） |
| 按资产分别验收后SOL/BNB的通过数 | **0/35**（不是跨资产问题，是这两个资产自身没有alpha通过） |
| 独立数据源（既有引擎）是否印证SOL/BNB更难 | **是**：Sharpe从BTC的1.14/ETH的0.93降到SOL的0.21/BNB的0.46 |
| 资金费率alpha是否已经"发现新edge" | **否，但比其他候选更接近**：83%折IC一致性，只败在PnL集中度检验（和V1.2同一个已知的方法论局限） |
| 核心结论是否改变 | **不变**：仍不具备模拟盘资格 |

**"放开了搞"之后的诚实结果**：资产范围和数据源确实放开了（4个
资产、资金费率这个全新数据源、任意资产数量的架构），但样本内验证
标准没有放松，得到的还是一个总体上的负面结果——不过这次负面结果
的构成更清楚：SOL/BNB不是"因为要求太苛刻才没通过"，而是这两个
资产在这段历史上确实缺乏可靠的趋势类edge（有独立证据支持这一点），
而资金费率数据虽然还没有alpha正式通过，但目前看是最有希望的
下一个突破口。

## 附录：本报告引用的文件

- `alpha_gate_results.csv`、`alpha_gate_folds.csv`、
  `regime_conditional_gate_results.csv` — 完整4资产Gate明细
- `per_asset_gate_summary.json` — 按资产分别license的结果
- `scenario_comparison.csv`、`scenario_per_symbol_breakdown.csv`、
  `*_equity.csv` — A/C/F场景组合回测明细
- `alpha_reports_multi_asset/` — 128个alpha实例的完整IC/分桶/
  regime/逐年研究数据（`alpha_research.py`、`alpha_correlation.py`
  的输出，与旧的2资产`alpha_reports/`分开保存，不覆盖历史报告）
- `../../alphas/alternative_data.py`、`../../alphas/cross_asset.py` —
  新增的资金费率alpha和多资产广度/相对强弱alpha实现
- `../../tests/test_alternative_data.py`、
  `../../tests/test_alphas.py::MultiAssetCrossAssetAlphaTests`、
  `../../tests/test_portfolio_backtest.py::PerAssetLicensingTests` —
  对应的单元测试
