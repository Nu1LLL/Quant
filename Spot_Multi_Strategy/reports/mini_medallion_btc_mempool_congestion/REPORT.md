# BTC mempool congestion demand: sealed rejection

**Experiment:** `btc_mempool_congestion_demand_2026_11_08`

**Preregistration:** `fb8f486` (pushed alone before any indicator level or outcome was read)

**Implementation:** `87abf8f` (pushed before the one formal run)
**Formal record:** `RESULT.json`; `all_gate_passed=false`; no real order or paper deployment.

## What was tested

The one frozen rule bought Binance BTCUSDT spot at a Monday UTC open when the median of the latest seven complete daily Bitcoin mempool medians was at least 1.5 times the preceding 28-day baseline after a fixed 1 MB offset. The final mempool point was from Friday, more than 48 hours before entry. Otherwise the $10,000 account held USDT cash. Each changed target paid 10 bp taker fee and 5 bp adverse slippage, with 0.00001 BTC lot rounding and a 1% prior-volume capacity cap. No leverage, derivatives, funding, price-based signal or additional coin. Source gaps in 17 of 261 Mondays forced cash as preregistered; no year was dropped. There were 21 buy entries and 42 fills.

The fixed current-vintage Blockchain.com response matched its preregistered SHA-256, schema, unit and 315,432 timestamped rows. Official Binance monthly spot ZIPs matched their pinned checksums. The five-year account interval contains all 1,826 days and 60 months from 2021-01-01 through 2025-12-31. This is a **retrospective current-vintage study**, not independently prospective OOS; historic indicator values may have been revised and daily bars do not prove actual fills.

| Net account measure | Result | Frozen Gate |
| --- | ---: | ---: |
| Final equity | 11,843.83 USDT | — |
| CAGR | 3.44% | ≥40% |
| Daily Sharpe | 0.263 | >1.5 |
| Conservative max drawdown | 38.46% | ≤20% |
| Daily-dollar profit factor | 1.062 | >1.3 |
| Profitable months | 16/60 | ≥36/60 |
| Longest nonpositive monthly streak | 14 | ≤3 |
| Worst month | −12.67% | reported |
| 2024–2025 retrospective holdout CAGR / Sharpe | −6.83% / −0.431 | ≥40% / >1.5 |
| Holdout max drawdown / profit factor | 30.48% / 0.748 | ≤20% / >1.3 |
| 21-day block Monte Carlo joint passes | 0/5,000 | ≥80% |

The zero-cost version ended at 12,614.07 USDT; the net version incurred 455.10 USDT fees and 227.55 USDT reference-price slippage. Double-slippage stress ended at 11,597.72 USDT, but moving all fills one day later ended at **9,444.71 USDT**, below initial capital. The same-cost BTC buy-and-hold benchmark ended at 30,212.55 USDT but had 77.57% maximum drawdown and Sharpe 0.670, so it is not a passing replacement. Annual walk-forward, the four market-state Gate, monthly stability, the holdout and independent OOS all failed. In 2023 the rule gained 44.81%, but that one year cannot override the full-period and holdout failures.

**Decision:** reject and seal this exact demand/congestion direction and parameter set. No threshold search, signal inversion, year deletion, asset substitution or leverage rescue is permitted on these data. The protected mature strategy and previous reports were not modified.
