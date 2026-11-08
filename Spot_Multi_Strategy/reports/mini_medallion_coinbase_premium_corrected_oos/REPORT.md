# USD-corrected Coinbase BTC premium — sealed rejection

**Formal one-shot result:** preregistration and two price-blind source manifests committed/pushed at `e63231d`; implementation and synthetic checks committed/pushed at `a5ecd95`; the first official-price replay is [`RESULT.json`](RESULT.json). The read-only source audit matched all **16 Coinbase response hashes** and **63 Binance spot ZIPs** against their frozen hashes and current Binance checksums. Coinbase `BTC-USD` and `USDT-USD` each covered every UTC day from 2021-09-01 through 2026-09-30; the traded Binance spot series covered 2021-08-01 through 2026-09-30 plus only the 2026-10-01 terminal open. No real order was placed.

The fixed hypothesis interpreted a positive 7-day average of `ln(Coinbase BTC/USD ÷ [Binance BTC/USDT × Coinbase USDT/USD])` as USD-denominated venue demand and bought spot BTC on Tuesday, with one-day signal delay, 1x maximum exposure, a 10,000 USDT starting account, 10 bp taker fee and 5 bp adverse slippage on each fill. The 2024–2026 portion was a historical temporal holdout, **not independently executable OOS**. The cross-venue price-discovery papers motivate investigating segmentation but did not establish this next-week prediction.

| Preregistered Gate | Net official-history result | Required | Verdict |
|---|---:|---:|---|
| Full five-year CAGR | **7.62%** | ≥40% | Fail |
| Daily Sharpe | **0.384** | >1.5 | Fail |
| Conservative high-before-low max drawdown | **65.86%** | ≤20% | Fail |
| Daily-dollar Profit Factor | **1.035** | >1.3 | Fail |
| 2024-01 to 2026-09 holdout CAGR / Sharpe / drawdown / PF | 11.31% / 0.475 / 41.75% / 1.044 | all four numerical thresholds | Fail |
| Positive months | **28/60**; 6 consecutive nonpositive; worst −37.29% | ≥36/60; streak ≤3 | Fail |
| Annual fixed-rule folds | 2021 Q4 −10.4%; 2022 −51.7%; 2023 +148.4%; 2024 +69.9%; 2025 −16.4%; 2026 Jan–Sep −5.5% | every fold profitable with positive Sharpe | Fail |
| Four market states | bull/high only **20 days**; both bear states lost money | ≥90 days and positive dollar PnL in each | Fail |
| 5,000-path 21-day block Monte Carlo | **0/5,000** jointly pass | ≥80% | Fail |
| Independent executable OOS | unavailable | required | Fail |

The account ended at **14,437.50 USDT** after 58 fills (29 buys), 689.94 USDT in explicit fees and about 344.97 USDT in reference-price slippage drag. The no-cost comparator ended at 15,749.43 USDT. The frozen double-slippage and one-day-latency paths ended at 14,024.68 and 11,433.56 USDT, respectively, but they do not cure the main Gate failures. Monte Carlo 5th-percentile CAGR was **−22.38%** and median conservative drawdown **63.43%**.

**Decision:** reject. Do not reverse the premium direction, alter the averaging/holding period, select another venue, increase leverage, or reuse the better stress result on this history. Historical candles lack vintage availability, synchronized quotes and actual fill evidence; profitable-looking years cannot establish a deployable alpha. Protected mature strategy files and prior sealed reports remain unchanged.

Primary context: [Stalder and Cosenza, cross-venue Bitcoin price discovery](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5070964); [Alexander et al., order flow and price formation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4867599); [Coinbase historical-rate access guidance](https://help.coinbase.com/en/pro/other-topics/api/faq-on-api); [Binance public archive documentation](https://github.com/binance/binance-public-data).
