# 跨市场微观结构论文路线

## 第一优先：可解释的订单流与队列不平衡

1. Cont, Kukanov, Stoikov, *The Price Impact of Order Book Events*
   https://arxiv.org/abs/1011.6402
   50只美股研究中，短时间价格变化与最优买卖盘订单流不平衡近似线性，斜率与市场深度反向；该关系比仅用成交量更稳健。对应实验：OFI/深度缩放线性基准 + HMM 状态 + RBF 非线性增益。

2. Gould, Bonart, *Queue Imbalance as a One-Tick-Ahead Price Predictor in a Limit Order Book*
   https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2702117
   Nasdaq 10只流动性股票上，队列不平衡对下一中间价变动方向显著，大跳点股更强；局部逻辑回归略优于全局逻辑回归。对应实验：现有RBF核模型与简单逻辑基准比较，按 tick/price 和 spread/tick 分层。

## 第二优先：状态依赖事件模型

3. Morariu-Patrichi, Pakkanen, *State-dependent Hawkes processes and their application to limit order book modelling*
   https://arxiv.org/abs/1809.08060
   用 spread 和 queue imbalance 作为状态变量，研究订单流自激与盘口形状的反馈。对应实验：固定指数核多变量Hawkes，只在逐事件数据可用后实现，不用分钟K线伪造事件强度。

## 第三优先：深度学习基准

4. Zhang, Zohren, Roberts, *DeepLOB*
   https://arxiv.org/abs/1808.03668
   CNN提取盘口空间结构、LSTM提取时序关系，并报告向训练外品种迁移。对应实验：只在每市场有足量L2、purged walk-forward、真实成本和简单模型基准时作挑战者；预测准确率不等于可交易Alpha。

## 拒绝顺序

若 OFI/队列基准在真实成本后不成立，不直接用更深模型挖救同一数据；需先查数据粒度、成交延迟和市场结构。Hawkes和DeepLOB若不能在完全隔离OOS上净改善简单基准，即拒绝。
