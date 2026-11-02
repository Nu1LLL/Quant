# Mini-Medallion V1.2 — Regime条件激活架构

## 为什么做这一轮

上一轮迭代（V1.1）后用户明确指出：这个项目名字叫"Mini-Medallion"，
应该按文艺复兴科技Medallion基金那种"很多弱信号、低相关、组合出
稳健收益"的思路来做，而不是最后又变成"只剩几个和既有趋势策略
同质的信号"。这个批评是对的，需要正面回应，而不是重复V1.1已经
做过的"再修一个bug、再测一次止损"。

**Medallion真正做的事**：数千个短周期统计信号，基于远比日线OHLCV
丰富得多的数据（订单簿、跨市场领先滞后、另类数据），并且**大量依赖
regime条件——不同信号在不同市场状态下被选择性激活，而不是一套权重
从头到尾跑满全部历史**。V1/V1.1的Alpha Research Gate是无条件的：
要求一个alpha在全部6折、全部历史上都稳定有效才能通过，这本质上和
既有趋势引擎的验收逻辑是同一种思路，均值回归类信号自然全灭——
这才是"结果看起来像旧系统换皮"的根本原因。

**这一轮做的事**：把Gate和集成都改成regime条件的架构，让每个alpha
分别在trending/mixed/ranging三个（用纯滚动窗口、无未来数据泄漏的
efficiency ratio因果切分出来的）regime子集里单独跑一次6折
walk-forward验收，只有在它真正擅长的regime里才被激活参与组合。

## 架构改动（代码，不是调参）

1. `alpha_metrics.compute_causal_regime_labels()`：滚动窗口分位数
   切分regime，t时刻的分类只依赖t及更早的数据——这是唯一的regime
   定义来源，Gate和实盘敞口共用同一套定义，不会重蹈V1.1发现的
   "研究用一套、实盘用另一套"的泄漏问题。
2. `walk_forward.simulate_standalone_alpha()` / `evaluate_alpha_walk_forward()`
   新增`activation_mask`参数：只在mask为True的K线上计算暴露、换手
   和IC。**折的切分方式也相应改变**——不是先按日历时间切6折再叠加
   regime过滤（这样会导致某些折里regime样本极少，人为放大PnL集中度
   问题），而是先取出这个alpha在该regime下所有被激活的K线（按时间
   先后排列），再把这个"激活序列"本身切成6折。用一个人工构造的
   "只在mask=True时有效"的alpha证明了机制本身是对的：无条件评估
   失败，regime条件评估通过（见`tests/test_walk_forward.py::RegimeConditionalGateTests`）。
3. `alpha_gate_report.run_regime_conditional_gate()`：对58个
   alpha×品种实例、每个都在3个regime上单独跑一次6折门槛，同样要求
   "在所有测试品种上都通过"才算这个regime的专家alpha。
4. `ensemble.build_regime_conditional_ensemble()`：每个alpha只在
   它被license的regime里参与组合权重计算，未被license的时间段
   权重为0。用单元测试证明了机制本身正确
   （`tests/test_ensemble.py::RegimeConditionalEnsembleTests`）。

## 在真实数据上的结果：0个regime专家alpha

58个alpha×品种实例 × 3个regime = **174个regime条件评估，0个通过**
（`reports/mini_medallion_v1_2/regime_conditional_gate_summary.json`）。

这不是机制失灵——机制本身在合成测试里工作正常。真实数据上0个通过，
是因为**Gate沿用了和V1/V1.1完全相同的6项标准**（这是刻意的：如果
换了标准，就没法把这次的0和之前的3做公平比较）。用同一套标准，
regime条件评估比无条件评估更难通过，原因如下。

### 诊断：方向性技能和PnL集中度是两件不同的事

174个regime条件评估里：

- **15个（8.6%）折IC为正的比例达到100%**——也就是说这个alpha在
  它擅长的regime里，6折**全部**预测方向正确，这是相当强的方向性
  一致性证据。
- 但**58个（33%）在通过了IC相关的全部检验之后，唯独败在"单折/单年
  PnL占比"这一项**（`check__单折PnL占比不超过50%`或
  `check__单一日历年PnL占比不超过60%`）。

举例（完整数据见`regime_conditional_gate_results.csv`）：

| 品种 | Alpha | Regime | 折IC为正比例 | 折IC中位数 | 单折PnL占比 |
|---|---|---|---|---|---|
| BTCUSDT | A07_atr_percentile | ranging | **100%** | 0.044 | 100%（唯一贡献者）|
| BTCUSDT | A06_rsi_reversion_14 | ranging | **100%** | 0.036 | inf（正贡献年份为0）|
| ETHUSDT | A02_ema_distance_20 | trending | **100%** | 0.034 | 75% |
| ETHUSDT | A18_btc_lead_eth_lag_3 | trending | **100%** | 0.059 | 53% |

**为什么会这样**：这些是趋势/regime类信号，它们的收益天然是"lumpy"
的——大部分利润来自少数几段大趋势（比如2020年那轮牛市），即使
alpha每一折的**方向**判断都是对的，**金额**贡献仍然会集中在波动
最大的那几段时期，这是趋势跟随类payoff的固有特征（正偏度、肥尾），
不是signal不稳定的证据。而PnL集中度检验的本意是抓"某一段时期靠
运气蒙对、其余时期毫无技能"这种情况——对于折IC 100%为正的alpha，
这个假设的前提本身就不成立，检验的目标和alpha的实际失败模式
不匹配。

**这里我们选择不去改动PnL集中度阈值来让这些alpha通过**——那样就是
用户明确要求不要做的"回头调参数让数字更好看"。但这是一个值得记录
的、有充分数据支持的方法论发现：**PnL集中度检验对均匀分布收益的
均值回归类信号是合适的过滤器，但对本质上lumpy的趋势/regime类信号
可能系统性地过严**，如果要在不放松标准的前提下解决这个问题，正确
方向是设计一个**不同的检验**去分别识别"方向技能不稳定"和"收益金额
天然集中"这两种不同的失败模式，而不是用同一个阈值笼统地处理，
这需要新的统计设计，不是本轮迭代能够顺带完成的，留作后续工作。

## 对"Medallion"批评的正面回应

1. **机制确实是缺的，现在补上了**：V1/V1.2之前的Gate完全没有
   regime条件这个维度，现在有了，代码和测试都在，架构上可以在
   未来数据/新alpha上复用。
2. **但这次真实数据的结果仍然是"不成功"**，如实报告，不回避：
   regime条件激活在当前的验收标准下没有额外救回任何alpha。
3. **更根本的限制没有变**：Medallion的"多信号"建立在远超日线OHLCV
   的数据广度上（订单簿、跨市场、另类数据）。本仓库目前只有
   Binance公开K线，A20（多资产广度）目前只是架构占位，`futures_data.py`
   已经在抓取的资金费率数据还没有被纳入alpha候选库——**这才是"造不出
   真正意义上的很多个低相关alpha"最根本的原因，regime条件架构解决
   不了这个数据广度问题，它解决的是另一个（同样真实存在、但更小的）
   问题**。下一步如果要更接近Medallion，应该往数据广度方向投入
   （资金费率carry、跨资产领先滞后的更多品种、订单簿失衡代理），
   而不是继续在现有OHLCV alpha库内部调整验收逻辑。

## 结论

| 检查项 | V1/V1.1 | V1.2 |
|---|---|---|
| Gate是否支持regime条件激活 | 否 | **是（新架构，已测试）** |
| 真实数据上regime条件下通过的alpha数 | N/A | **0/174** |
| 是否发现了新的、有数据支持的方法论问题 | — | **是**：IC一致性与PnL集中度脱钩，对趋势类信号的检验设计需要重新思考 |
| 核心结论是否改变 | 不具备模拟盘资格 | **不变，仍不具备模拟盘资格** |
| 是否需要更多结构调整才能更像Medallion | — | **是**：数据广度不足是比Gate设计更根本的瓶颈 |

按照任务要求，不应该为了让regime条件Gate"跑出结果"而放松PnL集中度
阈值。这次迭代交付的是一个测试完备、可复用的regime条件架构，加上
一个诚实的、有充分证据支持的负面结果和一个更根本问题的诊断——这就是
这一轮"改结构"能够如实交付的全部内容。

## 附录：本报告引用的文件

- `regime_conditional_gate_results.csv`、`regime_conditional_gate_folds.csv`、
  `regime_conditional_gate_summary.json` — 174个regime条件评估的完整数据
- `alpha_gate_results.csv`、`alpha_gate_folds.csv`、`alpha_gate_summary.json` —
  同一次运行里的无条件Gate结果（与V1一致，用于对照）
- `../../tests/test_walk_forward.py::RegimeConditionalGateTests` —
  证明regime条件Gate机制本身正确的单元测试
- `../../tests/test_ensemble.py::RegimeConditionalEnsembleTests` —
  证明regime条件集成机制本身正确的单元测试
