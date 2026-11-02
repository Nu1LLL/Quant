# HMM + RBF 盘口模型v2：OFI增量预注册

状态：实现前冻结。无真实盘口数据前只允许无泄漏与逻辑验收。

## 新增特征

- `ofi_l1`：按 Cont–Kukanov–Stoikov 的最优档价格/队列变化规则计算单步 order-flow imbalance。
- `ofi_l1_depth_scaled`：`ofi_l1` 除以前后两个时点的平均L1深度。
- `ofi_5m_depth_scaled`：过去5个已完成时点的 OFI 和，除以同窗口的平均L1深度。

对于minute snapshot，上述只是快照近似，不冒充逐事件OFI；接入MBO/MBP后必须先按序列号重建订单簿再聚合。

## 比较

v2必须在同一滚动训练窗、同一OOS、同一交易Gate和同一成本下，同时报告：

1. 全局逻辑回归队列基准；
2. v1 HMM+RBF；
3. v2 HMM+RBF+OFI。

v2只有在独立OOS的净收益HAC下界、月度稳定性和成本压力下稳定优于v1才保留；仅AUC/准确率提高不足以通过。
