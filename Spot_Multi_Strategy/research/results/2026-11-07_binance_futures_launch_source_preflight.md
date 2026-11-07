# Binance USDT 永续合约上市事件：价格盲源数据预检

2026-11-07。此文件只记录公告源数据状态，不是策略结果。

- Binance 官方新上市目录 `catalogId=48` 的 46 页、2,292 条公告已完整读取并冻结：`research/preregistrations/2026-11-07_binance_listing_catalog_metadata.json`，提交 `28d2987`。目录包含现货、合约和其他公告，不能把所有行当作币种上市事件。
- 用预先写入的严格标题规则筛选 2021–2025 年、单一 USDT 本位永续合约、非盘前及非 TradFi 的公告，得到 222 条详情候选。事件取公告正文中的 UTC 开盘时间；目录与详情发布时间同时保留，较晚者作为可见起点，差异超过 60 秒的行排除。
- 官方详情接口的样本确实提供正文、合约代码和开盘时间；核验约 75 条后接口返回 HTTP 429。未生成完整事件清单；断点续取及减速代码已加入 `research/binance_futures_launch_preflight.py`，待源端限流恢复后继续。
- 本预检未读取行情、资金费或策略收益；未提交交易假说的正式预注册，也未运行回测。不能声称独立 OOS 或任何 Gate 通过。

源接口：`https://www.binance.com/bapi/composite/v1/public/cms/article/detail/query?articleCode=<code>`。公布的合约上市时间格式见 [Binance 2021 年 GTC 永续合约公告](https://www.binance.com/en/support/announcement/detail/c1a900363bd04ffd94032322f10de8ab)。
