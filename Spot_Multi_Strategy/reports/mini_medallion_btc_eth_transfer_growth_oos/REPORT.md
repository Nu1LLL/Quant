# BTC/ETH 原生币转账增长轮动：数据失败封存

**结果：`DATA_FAILURE`。没有生成 CAGR、Sharpe、最大回撤、Profit Factor、月度盈利率或交易收益。**

## 顺序及唯一正式运行

- `4bcb280`：单独提交并推送经济假说、两币资产、28 天转账增长比、周一信号、成本、样本切分、杠杆、全套 Gate 与停止规则。此前只核对 Coin Metrics 资产/日期覆盖、字段、缺失掩码与响应哈希，没有读取转账次数值或本假说的价格/结果。
- `f223d03`：单独提交并推送冻结源校验、双币现货换仓、手续费/滑点、容量、年度/月度/状态/区块 Monte Carlo 检验。3 个本假说合成测试通过，工作区与 GitHub 在正式运行前一致。
- 唯一正式运行于 **2026-11-08 14:47:10 UTC** 开始，**14:47:43 UTC** 写入 `DATA_FAILURE.json`。失败点为币安官方 `BTCUSDT-1d-2023-05.zip.CHECKSUM` 的 `curl` TLS 握手（exit 35）。数据读取在组合模拟之前停止；没有写入 `RESULT.json`。

## 边界与下一步

该实验及其数据传输路径按预注册规则永久封存，不通过更换客户端、重试正式运行或改动币种和窗口来补做同一历史。Coin Metrics 的当前版本历史也无法证明每个历史周一的当时可用数值；即使数据通道成功，本设计的独立可执行 OOS 仍为 `false`。

下一轮应先对**所有**必需来源做只读传输和日期/校验文件可达性预检，再决定是否预注册另一条独立经济假说。此轮没有实盘或模拟盘订单，没有修改用户成熟策略、已封存报告或 WhaleWatcher 应用。

[Coin Metrics 指标定义](https://docs.coinmetrics.io/network-data/network-data-overview/transactions/transfers)；[Binance 官方公开数据说明](https://github.com/binance/binance-public-data)。
