# 任务三：首轮原始论文清单与可检验命题

检索日期：2026-09-16。只保留作者、NBER、期刊或论文预印本页面；以下是候选假说，不是收益保证。

## A. 状态切换与在线变化点

### Hamilton (1989), *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*

- 原始出处：Econometrica 57(2), DOI `10.2307/1912559`。
- 方法：离散状态 Markov 过程改变自回归参数，用非线性过滤器估计不可观察状态并预测。
- 可检验命题：两状态过滤概率能否在多资产中改善风险预算，而非直接预测涨跌。
- 实现纪律：只能使用 `P(S_t | y_1...y_t)` 的过滤概率，不能使用全样本平滑或事后 Viterbi 路径。

### Ang & Bekaert (2003/2004), *How Do Regimes Affect Asset Allocation?*

- 原始出处：NBER Working Paper 10080，后发表于 *Financial Analysts Journal*。
- 论文报告：纯股票集合中状态模型较难利用；允许现金、债券和股票切换时价值更明显。
- 对本项目的启示：首个 HMM 实验应做股票/债券/现金/商品/外汇的风险配置，不应只在一个股票指数上开关。
- 风险：论文的最优配置依赖参数估计；本项目只试固定 2、3 状态和保守概率阈值。

### Adams & MacKay (2007), *Bayesian Online Changepoint Detection*

- 原始出处：arXiv `0710.3742`，作者论文版。
- 方法：在线递推“距最近变化点的运行长度”后验分布，适合只看过去的实时检测。
- 可检验命题：收益方差、相关性或价差均值发生变化的后验概率，能否比固定波动率阈值更早降低风险。
- 研究价值：它天然是在线过滤，较少发生把全样本分段误当实盘信号的问题。

## B. 相对价值与残差均值回复

### Gatev, Goetzmann & Rouwenhorst (1998/2006), *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*

- 原始出处：NBER Working Paper 7032 / *Review of Financial Studies* 19(3)。
- 方法：形成期按标准化价格路径最小距离选对，随后六个月交易偏离；论文还用随机配对 bootstrap 区分普通反转。
- 可检验命题：在点时可交易、包含退市和借券成本的 ETF/双股类别/期现组合中，经济联系配对是否优于随机配对。
- 不照搬之处：原研究时代交易制度不同；必须使用现代成本、融券与短腿约束。

### Avellaneda & Lee (2010), *Statistical Arbitrage in the U.S. Equities Market*

- 原始出处：*Quantitative Finance* 10(7), DOI `10.1080/14697680903124632`；作者论文页提供工作论文。
- 方法：用 PCA 或行业 ETF 剔除共同成分，将个股残差拟合为均值回复过程。
- 关键警示：论文自己的结果在 2003 年后明显衰减；这不是可直接复用的高 Sharpe 配方。
- 可检验命题：现代点时股票池中，残差回复在成交成本后是否仍存在；概率状态只允许作为事前风险门控。

## C. 波动率与条件分布

### Bollerslev, Tauchen & Zhou (2009), *Expected Stock Returns and Variance Risk Premia*

- 原始出处：*Review of Financial Studies* 22(11)；作者 Duke 页面提供论文。
- 方法：以期权隐含方差和高频实现方差之差度量方差风险溢价，并研究中期股票收益预测。
- 可检验命题：方差风险溢价是否能为跨资产风险预算提供条件概率，而不是直接卖波动率。
- 数据门槛：需要可靠的历史期权隐含方差和日内实现方差；若只能用 VIX 减日线方差的粗代理，证据等级必须降低。

## 首轮优先级

1. Hamilton + Ang/Bekaert：多资产 2 状态过滤风险配置。
2. Adams/MacKay：在线波动率/相关性变化点防御层。
3. Gatev：有经济联系资产的相对价值。
4. Avellaneda/Lee：点时股票池残差回复；数据工程成本较高。
5. 方差风险溢价：等可靠期权与日内数据齐备再做。

未找到可靠公开来源证明“大奖章基金使用 HMM”。因此不把该说法写入任何实验依据。
