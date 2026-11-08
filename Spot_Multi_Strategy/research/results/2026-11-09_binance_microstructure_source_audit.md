# Binance microstructure archive: source-only feasibility audit (2026-11-09)

This is a **source preflight**, not a registered alpha experiment or a backtest. Only HTTP `HEAD` status and `Content-Length` were read from the official archive. No ZIP body, trade, quote, price, return, signal, fee, funding, or strategy result was read. No market-data file was retained locally. Previous sealed results and the protected mature strategy were not changed.

## Reproducible checks

The archive template was `https://data.binance.vision/data/spot/monthly/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM}.zip`. For each of `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `DOGEUSDT`, all **60 monthly names from 2021-11 through 2026-10 inclusive** were checked with `curl -L -I`. Two transient transport errors were retried; both returned HTTP 200. Thus **240/240 monthly archive names returned HTTP 200**. The sum of reported compressed `Content-Length`, rounded to two GiB decimals, was approximately BTC 41.73 GiB, ETH 21.52 GiB, SOL 8.06 GiB, DOGE 6.22 GiB. These are transfer-size estimates, not disk allocation or proof of row continuity. The official [archive README](https://github.com/binance/binance-public-data/blob/master/README.md) defines Spot `aggTrades` as API-derived aggregate trades; a row can combine fills and is **not** a book quote or L2 snapshot. The same README notes Spot timestamps switch to microseconds from 2025-01-01. A future parser must handle that boundary explicitly and verify checksums, row continuity, and the actual timestamps before accepting an OOS period.

The free USD-M BTCUSDT daily archive templates were `https://data.binance.vision/data/futures/um/daily/{KIND}/BTCUSDT/BTCUSDT-{KIND}-{YYYY-MM-DD}.zip`. Sampled `HEAD` results:

| Kind | 2021-11-09 | 2022-01-01 | 2022-11-09 | 2023-06-01 | 2024-02-01 | 2024-06-01 | 2025-06-01 | 2026-10-01 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `bookDepth` | 404 | 404 | 404 | 200 | 200 | 200 | 200 | 200 |
| `bookTicker` | 404 | 404 | 404 | 200 | 200 | 404 | 404 | 404 |

One missing day already prevents a **continuous 2021-11–2026-10 five-year sample** from those particular free archives; these samples do not prove every other date's status or rule out a separate licensed source. The official [Binance collaborator's explanation](https://github.com/binance/binance-public-data/issues/447) says `bookDepth` records depth at percentage distances from mid, not price-level L2 updates. Whether those reported percentage bands are cumulative remains unclear in that issue; the audit does not assume an answer. An [open archive-defect report](https://github.com/binance/binance-public-data/issues/496) presents frozen near-ask bands from 2026-09-03 despite matching published checksums. That is a reported defect, not independently replicated here; it still requires a semantic quality gate before use. A [Binance collaborator](https://github.com/binance/binance-public-data/issues/268) said separate futures L2 construction had gaps before 2023-05-28 and used internal logs afterward. Its access, cost, symbol coverage, and present availability have not been verified in this audit.

## Research disposition

Five years of **historical executed-trade events** appear discoverable without an API key; five years of **continuous free true L2** have not been established. `aggTrades` can support an event-time hypothesis only after an economically distinct mechanism, fixed asset set and parameters, cost and latency model, five-year sample split, and Gate are preregistered in a separately pushed commit. Download only the needed monthly archive sequentially, verify its official checksum and contents, reduce it to bounded features, and discard the raw file; never pull all ~78 GiB into the user's disk. An HTTP 200 or checksum alone does not establish independent executable OOS. No strategy or live order is authorized by this audit.
