# Mini-Medallion 研究项目交接文档

写给接手的Codex（或任何后续agent）。这份文档只覆盖"Mini-Medallion"
这一条研究线（多个弱、低相关、可解释alpha信号组合的研究平台），
**不覆盖**本仓库里其他并行存在、由用户自己开发的独立工作——见
下方"不要碰的文件"一节，这点非常重要，请先读完再动手。

## 一句话现状

180个单元测试全部通过，代码全部已提交到本地git（未push），最新
一次诚实结论是：**这个研究框架能发现的、经过walk-forward验证的
alpha信号，风险调整后夏普比率的天花板大约在0.3-0.5之间**，用杠杆
或更换资产类别都无法把它推高到用户要求的1.5-2。完整过程见
`reports/mini_medallion_market_neutral/EXPANDED_SCOPE.md`（最新，
最应该先读的文件）。

## 用户要求 & 必须遵守的规则（不可协商）

用户最初的要求是把仓库里的`Spot_Multi_Strategy`（一个已有的量化
交易策略仓库）扩展成"迷你文艺复兴"式研究平台：许多弱的、低相关的、
经济上可解释的alpha信号，组合成一个投资组合，优化目标是**稳健的
样本外夏普比率/多样化程度**，不是历史CAGR，且有一整套反过拟合规则：

- **禁止lookahead**（用未来数据做决策）
- **禁止用隐藏/未来数据做优化**
- **禁止挑选表现最好的资产/时间段**（cherry-picking）
- **禁止为了让指标通过而放宽Gate阈值**——这是贯穿整个项目最重要
  的一条纪律。每次某个alpha"差一点点"通过验收，正确做法是如实
  报告"没通过"，绝不能因为差得少就调整阈值让它过
- 初期禁止杠杆/做空/实盘——后续用户逐步明确放开了这些限制（见下）

用户后来明确放开的范围（按时间顺序）：
1. 允许扩展到股票/外汇/美股/各国市场（不只是加密货币）
2. 允许做空、加杠杆，做真正的市场中性统计套利（用户原话："趋势
   跟随肯定是不行的，90年代就不行了"）——但要求**新建独立代码**，
   不能修改或影响用户自己正在跑的`futures_strategy.py`/
   `hybrid_strategy.py`/`paper_state/`
3. 提供了《缠中说禅：教你炒股票108课》PDF，要求参考缠论但不完全
   依靠
4. 最后一次明确放开了几乎所有限制（资产类别、期权期货、方向、
   杠杆），要求"继续优化，直到夏普1.5-2以上，年化40%，回撤越小
   越好……没做到前不要停"

**我（上一个agent）对最后这条要求的处理方式，接手时请延续**：
没有把"目标数字"当成继续搜索/调参的停止条件——在已经反复分析过的
历史数据上，搜索空间越大只会增加过拟合风险，命中一个预设目标的
结果如果被当真部署真实资金是会误导人的。正确做法是：测试用户
点名的具体机制（加杠杆是否有效、换资产类别是否有效），如实报告
真实结果，即使没达标也作为这个研究问题的最终答案，而不是继续
搜索的检查点。**如果用户后续还是要求"继续搜直到达标"，应该重复
这个立场，而不是默默开始调参数搜索。**

## 不要碰的文件（用户自己正在做的、无关的工作）

这个仓库同时存在用户自己独立开发的其他策略研究，**和Mini-Medallion
完全无关，绝对不要修改、不要当成本项目的一部分**：

- `futures_strategy.py`、`hybrid_strategy.py`、`rotation_strategy.py`、
  `rotation_research.py`、`rotation_paper_trader.py`、
  `hybrid_research.py`、`futures_data.py`
- `paper_state/equity.csv`、`paper_state/state.json`——**这两个文件
  会显示为git modified，是因为用户有一个真实在跑的模拟盘进程在
  持续更新它们，这是正常现象，不是需要调查或还原的异常**
- `ADAPTIVE_ROTATION_RESEARCH.md`、`FUTURES_RESEARCH.md`、
  `ROTATION_RESEARCH.md`——用户自己那条线的交接/研究文档
- `configs/rotation_*.json`、`configs/hybrid_spot_short.json`、
  `configs/futures_trend_research.json`
- `rotation_reports/`、`adaptive_rotation_reports/`

## Mini-Medallion属于这些文件（可以继续开发）

**核心研究基础设施**（`Spot_Multi_Strategy/`根目录）：
- `alphas/`（包）——alpha信号库，`base.py`（AlphaSignal数据类、
  工具函数）、`momentum.py`、`trend.py`、`mean_reversion.py`、
  `volatility.py`、`volume.py`、`cross_asset.py`、
  `alternative_data.py`（资金费率类）、`chan_theory.py`（缠论
  结构层）。`build_single_asset_alphas(df)`是主入口。
- `alpha_metrics.py`——IC计算、regime标签（**注意**：
  `compute_regime_labels`是全样本分位数，只能描述性使用，绝不能
  用于实盘/因果场景；因果安全版本是`compute_causal_regime_labels`）
- `alpha_research.py`——加密货币数据加载+alpha集合构建
  （`DEFAULT_SYMBOLS = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]`）
- `alpha_correlation.py`——相关性矩阵、冗余信号检测
- `walk_forward.py`——纯多头walk-forward Gate核心逻辑
  （`evaluate_alpha_walk_forward`，7项检验）
- `alpha_gate_report.py`——纯多头Gate的CLI封装
  （`run_gate`/`summarize_acceptance`/`run_regime_conditional_gate`）
- `ensemble.py`——信号组合（等权/IC加权/相关性惩罚加权）
- `risk_overlay.py`——波动率目标、回撤保护、不交易带
- `portfolio_metrics.py`——CAGR/Sharpe/Sortino/Calmar/回撤等扩展
  指标（和旧的`metrics.py`分开，不要混用）
- `portfolio_backtest.py`——A/B/C/D/E/F多场景组合回测
- `market_neutral.py`——**完全独立**的市场中性多空引擎（横截面
  去均值/排名、gross_cap缩放、不交易带、资金费率成本、7项检验的
  `evaluate_cross_sectional_alpha`）
- `market_neutral_gate_report.py`——市场中性Gate的CLI封装
- `yahoo_data.py`（最新新增）——美股/外汇数据源（Yahoo Finance
  公开chart接口，免费不需要API密钥），输出列名和`data.py`
  （Binance现货）完全一致，可以直接喂给上面所有管线

**测试**：`tests/test_alphas.py`、`test_alpha_metrics.py`、
`test_alpha_correlation.py`、`test_walk_forward.py`、`test_ensemble.py`、
`test_risk_overlay.py`、`test_portfolio_metrics.py`、
`test_portfolio_backtest.py`、`test_alternative_data.py`、
`test_market_neutral.py`、`test_chan_theory.py`、`test_yahoo_data.py`
——运行`python3 -m unittest discover -s tests`，当前180个全过。
仓库约定：**测试不联网**，`test_yahoo_data.py`用`unittest.mock`
模拟`requests.Session`。

**报告**（按时间顺序，每份都是独立、诚实的阶段性结论，后面的不会
删除或篡改前面的）：
1. `reports/mini_medallion_baseline/` ——原有策略的可复现基线
2. `reports/mini_medallion_v1/` ——V1完整报告（后来发现lookahead
   bug，有勘误说明指向V1.1）
3. `reports/mini_medallion_v1_1/` ——修复lookahead bug + 止损证伪
4. `reports/mini_medallion_v1_2/` ——regime条件架构，0/174通过，
   诊断出PnL集中度检验可能的问题
5. `reports/mini_medallion_v1_3/` ——4资产扩展+资金费率alpha+
   单资产准入制
6. `reports/mini_medallion_market_neutral/` ——市场中性引擎，
   0/31通过，`LITERATURE_FACTORS.md`记录了文献因子替换+周频再
   平衡+缠论A26的后续研究
7. `reports/mini_medallion_equities/` ——美股Gate完整结果（最新）
8. `reports/mini_medallion_market_neutral/leverage_sweep/` ——
   杠杆敏感性测试完整数据（最新）
9. **`reports/mini_medallion_market_neutral/EXPANDED_SCOPE.md`**
   ——**最新、最完整的诚实结论报告，建议接手时第一个读这份**

## 当前唯一悬而未决的方法论问题（三次独立触发，从未通过放宽阈值解决）

`walk_forward.py`和`market_neutral.py`里的**PnL集中度检验**（单折
/单年正贡献占比不能超过50%/60%）在V1.2、V1.3、市场中性周频再
平衡三次独立场景里，都成为"最接近通过但卡在这一项"的候选的唯一
拦路虎。怀疑：这个检验的设计初衷是抓"运气而非技能"，但对折IC本身
很稳定（比如A01_ts_momentum_120折IC 100%为正）、只是payoff天然
lumpy的趋势类信号，可能是系统性地误伤。**如果继续这条线，这是
一个值得认真设计新检验的方向**（比如按年份/折分别做IC显著性检验，
而不是检验金额集中度）——但明确禁止的做法是直接把50%/60%这两个
阈值调大来让它们通过，那是真正意义上的"放宽阈值"。

## 明确排除、说明了原因的范围

**期权/期货定价回测**——因为没有免费、可靠的历史期权数据源，用
现货价格模拟期权payoff会产生编造的虚假结果，所以没做，不是遗漏。
如果Codex要做这块，必须先找到真实、免费或用户愿意付费的期权历史
数据源，不能用合成数据代替。

**缠论的背驰判断和三类买卖点分类**——`alphas/chan_theory.py`只
实现了缠论里客观、可机械判定的结构层（K线合并、分型、笔、中枢）。
背驰判断需要对"用哪个级别的走势做比较"做主观判断，翻译成精确规则
风险上更接近"为了让它work而设计规则"，所以明确没做。如果要更充分
检验缠论，下一步应该是设计一个同样精确、同样经过walk-forward验证
的背驰判断规则。

## 已知的技术坑（如果照原样复用这些函数，不会再踩，但要知道历史）

- `.iloc[0]`取出的是`numpy.bool_`，和Python的`False`用`is`比较会
  静默失败——必须显式`bool(x) is False`（`market_neutral_gate_report.py`
  的`rejection_reasons`已经这样写了，`alpha_gate_report.py`用
  `.iterrows()`绕开了同一个坑）
- 跨资产truncation测试必须按**日历时间**切，不能按行数切（不同
  资产的K线数量可能不完全对齐，按行数切会导致测试对比的不是同一
  个时间点）
- `combine_portfolio_net_pnl`里的`weight`已经是"占组合总资金的
  比例"，不能再乘以1/N，否则会重复稀释总敞口
- `resample_to_rebalance_schedule`补上之前，市场中性组合默认按
  每根4小时K线跟踪目标分数，这不符合动量/carry类因子在文献里的
  实际再平衡频率（通常月度/周度），会人为制造过高的换手成本

## 如何验证一切仍然正常

```bash
cd "/Users/tanganheng/Library/Mobile Documents/com~apple~CloudDocs/Documents/商务/Quant/Spot_Multi_Strategy"
python3 -m unittest discover -s tests
```

应该看到`Ran 180 tests`、`OK`。

## Git状态

最新两次commit（都在本地，未push）：
- `76ba913` —— 新增`yahoo_data.py`+测试
- `9998d5f` —— `EXPANDED_SCOPE.md`报告+`reports/mini_medallion_equities/`
  +`reports/mini_medallion_market_neutral/leverage_sweep/`全部数据

在此之前的21次commit覆盖V1到V1.3、市场中性、文献因子、缠论的完整
历史，`git log --oneline`可以看到完整列表，每条message都写明了
那次改动的动机和诚实结果。
