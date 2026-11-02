# 分钟/盘口数据源核查（2026-09-16）

## 结论

真实多年盘口数据不是普通K线下载问题。当前机器没有专业数据服务密钥，因此已经完成模型与统一输入接口，但不能诚实地产生真实盘口 OOS 买卖记录。接入优先级如下。

## 推荐路径

1. **Databento**：优先覆盖 CME 等期货 MBO/MBP、美股逐笔/盘口及可用的期权数据。MBO 含订单ID、添加、撤单、修改、成交与每日恢复快照，可建队列位置。文档：https://databento.com/docs/schemas-and-data-formats/mbo
2. **Tardis**：优先覆盖加密现货和永续的 `incremental_book_L2`、trades、quotes、book snapshots。官方客户端：https://github.com/tardis-dev/tardis-python
3. **CoinAPI 或 Kaiko**：Tardis 的替代/交叉核验源，支持历史 L2 快照或完整订单簿文件。文档：https://www.coinapi.io/datasets/crypto-order-book-l2-data 和 https://docs.kaiko.com/explore-our-data/data-dictionary
4. **Nasdaq TotalView**：美股完整 Nasdaq 显示订单深度；历史研究需确认许可与供应方式。产品页：https://www.nasdaq.com/products/data/equities/nasdaq-totalview
5. **Cboe/OPRA DataShop**：期权报价、成交与历史深度；需按产品和许可购买。产品页：https://www.cboe.com/data/market-data-services/us/options 和 https://datashop.cboe.com/data-products
6. **Dukascopy tick**：外汇第一阶段免费/低门槛 tick 与报价历史；它是单一来源，不代表全市场深度。官方历史接口：https://www.dukascopy.com/swiss/docs/interfaces/IHistory.htm

## 免费数据的边界

Binance 官方公共归档提供 trades、aggTrades 与 1s/1m 等K线，可用于成交方向、量价和初步分钟模型；但它不等价于多年完整 L2 增量订单簿。官方说明：https://github.com/binance/binance-public-data

因此免费数据可先训练“成交流+分钟量价”模型，盘口不平衡、撤单、队列成交概率和真实冲击必须等 L2/MBO 数据接入后验证。

## 已完成接口

- `microstructure_blackbox.py`：统一盘口校验、13项因果特征、前向两状态HMM、RBF随机傅里叶核分类/回归、成本Gate、四分之一Kelly仓位、Alpha衰减停机与点时全品种筛选。
- `microstructure_signal_runner.py`：读取每个市场/场所/品种的标准分钟盘口CSV，输出完整和最新的 `BUY/SELL/FLAT` 信号文件。
- 当前只通过合成数据无泄漏测试；在真实数据、真实成本和独立OOS完成前，不能用于实盘。
