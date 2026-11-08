# US real-yield easing BTC/ETH: sealed rejection

Preregistration `639e477` was committed and pushed before any yield or price
value was opened. Frozen implementation `07ddb83b944337c9b13bbe7fbc082c46b744f3e3`
was separately pushed before the one formal run. Exact full audit and fills are
in [`RESULT.json`](RESULT.json). This experiment is permanently sealed; do not
reverse the rate sign, adjust the 20-observation lookback, replace the basket,
drop years or rerun on this history.

## Executed rule and source audit

Every Monday UTC, falling US 10-year TIPS real yield over 20 valid observations
targeted equal BTC/ETH spot; non-falling yield targeted USDT cash. There was no
leverage. Every fill included 10 bp fee and 5 bp adverse slippage. The account
started with 10,000 USDT on 2021-01-01 and ended at the 2026-01-01 opening.

The frozen FRED DFII10 CSV matched SHA-256
`35f372e026b173dfbf847cc2384f3d553ba82c049a0ba3ffc9c52ab5d0b7f9ef`
with 1,348 dated rows and 58 missing/holiday observations. All 125 monthly
BTC/ETH Binance spot ZIPs matched their preregistered official checksums. The
bounded transport used three retries across those files; no alternative
historical source was substituted. Raw source archives were not kept.

## Prespecified results

| Measure | Net strategy | Fixed requirement |
|---|---:|---:|
| Final account | 20,661.79 USDT | — |
| Net CAGR | 15.62% | ≥40% |
| Daily Sharpe | 0.550 | >1.5 |
| Conservative maximum drawdown | 58.18% | ≤20% |
| Daily-dollar Profit Factor | 1.059 | >1.3 |
| 2024–2025 holdout CAGR | −9.91% | ≥40% |
| Profitable months | 23 / 60 | ≥36 / 60 |
| Longest nonpositive monthly streak | 7 | ≤3 |
| 5,000 block paths jointly passing | 0 | ≥4,000 |

There were 29 basket entries. Taker fees totaled 1,158.00 USDT and reference
slippage drag 579.00 USDT. Gross zero-cost final equity was 22,539.46 USDT;
the same-cost static 50/50 buy-and-hold comparator ended at 35,222.16 USDT.
Double-slippage and one-day-latency stress ended at 20,071.01 and 10,678.08
USDT respectively, but neither repairs the failed core Gate. Annual net PnL
was positive in 2021–2023, negative by 5,135.55 USDT in 2024, and only
333.60 USDT positive in 2025. The bear/high-volatility regime lost 4,468.92
USDT; the bull/high-volatility cell contained only 88 days, below the 90-day
minimum. Monte Carlo median maximum drawdown was 59.42%; joint pass rate was
0%, with zero ruined or undefined paths.

The 2024–2025 test is retrospective and FRED history is current-vintage.
`independent_executable_oos=false` regardless of numerical outcome. **Final
decision: reject; no paper or real order.** The strategy misses CAGR, Sharpe,
drawdown, Profit Factor, annual, holdout, monthly, regime, bootstrap and OOS
requirements. Its apparent 2021 benefit did not persist in the untouched
2024–2025 window.
