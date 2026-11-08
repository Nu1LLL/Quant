# BTC/ETH spot–perpetual cash-and-carry: timestamp-only source rejection

This was **data preflight only**, before any cash-and-carry strategy was preregistered. No spot/perpetual price, funding rate, signal, fill, PnL or Gate metric was inspected. The intent was to see whether Binance's public BTCUSDT and ETHUSDT **spot 1h** archives could support a complete 2021–2025 hourly hedge and conservative intraday liquidation/drawdown replay. The [official archive format](https://github.com/binance/binance-public-data) supplies monthly ZIPs and `.CHECKSUM` files.

`research.basis_hourly_preflight.audit_one("spot", "spot", symbol, month)` verified official checksum, full ZIP SHA, sole expected CSV member, 12-column shape and then the UTC hourly open/close grid. Four preselected diagnostic months per symbol were checked; all eight passed the file hash and member checks but **failed the hourly timestamp grid**. A separate timestamp-only inspection of adjacent rows found:

| Both BTCUSDT and ETHUSDT spot archive | First failing row | UTC evidence |
| --- | ---: | --- |
| 2020-12 | 494 | Row opens 2020-12-21 14:00; recorded close is around **13:47**, earlier than its open. Next row opens **18:00**, leaving 15:00–17:00 absent. |
| 2021-02 | 243 | Row opens 2021-02-11 03:00 but closes around **03:40**; next row opens **05:00**, leaving 04:00 absent. |
| 2021-03 | 122 | After the 2021-03-06 01:00 row, next row opens **03:00**, leaving 02:00 absent. |
| 2023-03 | 564 | Row opens 2023-03-24 12:00 but closes around **12:39**; next row opens **14:00**, leaving 13:00 absent. |

These are direct timestamp observations from official-hash-matched ZIPs, not a claim that every hour of all five years was audited. One absent hour already violates a strict complete-grid preregistration; eight failures across the two most liquid spot pairs decisively prevent that design from using these archives as lossless hourly evidence. In particular, interpolating through exchange downtime or pretending the last observed bar was executable would understate basis and liquidation risk. A daily-bar substitute cannot verify intraday liquidation or the requested drawdown bound.

**Decision:** no formal cash-and-carry hypothesis was registered or tested on this source, no performance value was produced, and no account or order was touched. A later study would need a separately preregistered venue/source and explicit proof of quote, margin and outage handling. This result does not modify any protected strategy or sealed return report.
