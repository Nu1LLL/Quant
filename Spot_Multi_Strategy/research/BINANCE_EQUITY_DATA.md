# 币安美股相关数据：临时数据源核验（2026-11-02）

用户指定现阶段只研究币圈与美股，并希望先用币安找美股数据。本文件只记录**数据源资格**，不预注册新收益假说，也不计算策略盈亏。

## 可以立即取得的币安数据

Binance Data Vision 的 USDⓈ-M 永续合约月档提供 `klines/4h`、`markPriceKlines/4h`、`fundingRate`，对应 ZIP 旁有 `.CHECKSUM`。这些品种映射美股或 ETF，但价格是 **USDT 结算的永续合约价格，不是美国交易所股票/ETF 原股成交价**。可用于 2026 年以来的短样本执行、资金费、价差和前瞻模拟；不能用来证明至少 5 年、更不能证明 10 年的美股策略稳定性。

| 映射品种 | 官方上线 / 归档首月 | 首月 4h 交易 K 线 | 首月 4h 标记 K 线 | 首月资金费记录 | 归档目录 |
| --- | --- | ---: | ---: | ---: | --- |
| AMZNUSDT | 2026-02 | 117 | 136 | 67 | `data/futures/um/monthly/` |
| SPYUSDT | 2026-04 | 147 | 168 | 83 | 同上 |
| QQQUSDT | 2026-04 | 147 | 168 | 83 | 同上 |

以上九个首月 ZIP 均已与币安对应 `.CHECKSUM` 作 SHA-256 校验并解析；行数只是数据存在性审计，**不是收益或完整历史验证**。上市首月标记价格早于可交易 K 线，不能把这段预启动价格当成可成交历史。2026-09 的 AMZN、SPY、QQQ 4h 交易档案端点也均返回可用；正式使用前仍须逐月校验完整性、资金费时间、成本和上市边界。

可复核的直接示例：[AMZN 2026-02 交易 K 线](https://data.binance.vision/data/futures/um/monthly/klines/AMZNUSDT/4h/AMZNUSDT-4h-2026-02.zip)、[SPY 2026-04 交易 K 线](https://data.binance.vision/data/futures/um/monthly/klines/SPYUSDT/4h/SPYUSDT-4h-2026-04.zip)、[QQQ 2026-04 交易 K 线](https://data.binance.vision/data/futures/um/monthly/klines/QQQUSDT/4h/QQQUSDT-4h-2026-04.zip)；各 URL 后加 `.CHECKSUM` 即为官方校验文件。[币安公开归档说明](https://github.com/binance/binance-public-data)与 [SPY/QQQ 上线公告](https://www.binance.com/en/support/announcement/detail/4cde981dc32d4268b7645e9d16e8d63a)说明来源和合约性质。币安说明股票永续在非美股交易时段采用不同的指数平滑机制，不能直接视作原股连续行情：[官方产品说明](https://www.binance.com/en/academy/articles/how-to-trade-stock-perpetual-contracts-on-binance)。

## 真实美股与长期 Gate

币安自 2026 年提供真实美股/ETF 交易，但目前[官方股票 API 市场数据目录](https://developers.binance.com/en/docs/catalog/advanced-trading-stocks-trading/api/rest-api/market-data)只列标的规则、代币映射和**当前报价**，没有历史 K 线接口；Data Vision 也没有独立的多年真实股票归档。[真实股票服务说明](https://www.binance.com/en-TR/support/faq/detail/a7469c7703524024b5bc2d492b03639d)明确其券商合作及市场适用性。2021 年旧股票代币已[停止支持](https://www.binance.com/en/support/announcement/detail/3a0304f3ee1c43668959c1b01f610d59)，不能与 2026 新合约拼接成一条虚假的长期历史。

因此目前的分工是：币安映射合约只作短历史补充与将来的前瞻监测；原定 CAGR/Sharpe/回撤/PF 的至少 5 年严格美股 Gate，必须另取**真实美股**历史数据。可评估与币安股票服务同一合作券商的 [Alpaca Historical Stock Data](https://docs.alpaca.markets/us/docs/about-market-data-api)（其官方说明覆盖 2016 年以来数据），但需要独立 Alpaca 凭证，不能拿 Binance API Key 代替。没有该历史数据前，不发布“美股长期 OOS 已通过”的结论。
