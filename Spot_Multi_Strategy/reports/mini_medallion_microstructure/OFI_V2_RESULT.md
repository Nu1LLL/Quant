# HMM + RBF 盘口模型v2：OFI实现结果

结论：OFI v2 的因果特征已实现，尚未获得真实多市场盘口，因此当前只通过结构与无泄漏验收，不报告 Alpha 收益。

## 实现

- 新增 Cont–Kukanov–Stoikov 最优档 OFI、相邻深度缩放 OFI 和过去5分钟深度缩放 OFI。
- 价格上移/下移与同价队列变化按论文符号规则计算，不用未来数据。
- v1 的13项特征仍可用 `feature_version=v1` 完整复现；v2比v1固定增加3项OFI特征，便于同窗口对照。
- 信号执行器默认版本升为 `microstructure-v2/hmm-rbf-minute-v2`，输出仍被 Alpha 生命周期门控。

## 限制

现有统一表是分钟快照，所以 OFI 是 snapshot proxy，会遗漏分钟内的新增、撤单和成交序列。真实MBO/MBP到位后，必须按序列号重建订单簿并另行生成逐事件OFI；在此之前不能声称复刻了论文结果。
