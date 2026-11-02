# 山寨币低活跃度隔日反转：正式结果

状态：**数据完整性失败；策略收益未计算，任何绩效 Gate 均不得标为通过。**

2026-11-02 严格按事先提交的 [预注册](PREREGISTRATION.md)与 [数据口径澄清](DATA_INTERPRETATION.md)运行一次。执行代码提交为 `1fc6d3e843b48f407527dc83b5d91f33380f14cd`；官方 Binance 月度 ZIP 逐档下载、校验 SHA-256 并解析后，在固定币种的整体时间网格审计处失败：`markPriceKlines/AAVEUSDT: coverage 0.998621 below 0.999`。按有效起点 2020-10-16 08:00 UTC 至 2026-10-01 00:00 UTC 的 13,054 根预期 4 小时 K 线计算，实际缺少 18 根，覆盖率为 99.8621%，低于预先设定的 99.9%。数据审计未产生可接受的完整归档清单。

依据冻结的停止规则，**没有删除或替换 AAVE、补零/插值、放宽完整性阈值、改样本期或读取该策略的正式收益**。因此本轮没有 CAGR、Sharpe、最大回撤、PF、月度盈利率或杠杆压力测试数字；不能声称目标已实现，也不能进入模拟交易。

本次只保存预注册、口径、代码、测试、本摘要与 [机器可读失败记录](DATA_FAILURE.json)；原始 ZIP 未落盘。复现命令：从仓库根目录运行 `python3 Spot_Multi_Strategy/research/run_altcoin_reversal.py --max-workers 12`。脚本检测到既有正式结果会拒绝重复运行；若需独立审计，应在隔离副本中复现，且不可用其结果调参本假说。

数据来源：[Binance 官方公开数据说明](https://github.com/binance/binance-public-data)；机制参考：[Bianchi、Babiak、Dickerson 原始论文](https://www.cerge-ei.cz/pdf/wp/Wp730.pdf)。这只是失败的历史数据资格审查，不是该经济机制有效或无效的证据。
