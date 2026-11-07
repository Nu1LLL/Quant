# Sealed result: eight-coin weekly illiquidity premium

**Decision: REJECTED.** The [preregistration](../preregistrations/2026-11-07_eight_coin_weekly_illiquidity.md) was committed and pushed at `14cc42f` before any strategy return was read. The [daily-boundary clarification](../preregistrations/2026-11-07_eight_coin_weekly_illiquidity_amendment_1.md) was separately committed and pushed at `92211d6`, likewise before results. Frozen replay and tests were committed and pushed at `540a71a`. The one-shot machine-readable [formal result](eight_coin_weekly_illiquidity_2026_11_07.json) is immutable. No same-history sign reversal, asset removal, holding-time change, cost relaxation or leverage rescue is permitted.

## Hypothesis and data

[Zhang and Li (first published January 2021)](https://onlinelibrary.wiley.com/doi/10.1002/ijfe.2431) report lower-liquidity cryptocurrencies earning higher cross-sectional returns. This test used the precommitted 28-completed-day Amihud score, weekly long-two-high / short-two-low ranking, eight fixed Binance USD-M perpetuals, four 25%-equity legs, 1.0x gross, and seven-day holds. All 552 price and 552 funding monthly archives for 2021-01 through 2026-09 matched pinned SHA-256 and official checksums and passed grid/numeric/event validation before replay. The formal account contains 1,959 days from 2021-05-21 to 2026-09-30 and starts with $10,000. Five-minute candles cannot establish point-in-time bid/ask or actual executable fills; prospective executable OOS remains absent.

## Fixed-cost net account

| Measure | Required | Observed |
|---|---:|---:|
| CAGR | ≥40% | **−10.41%** |
| Daily Sharpe | >1.5 | **−0.647** |
| Conservative five-minute maximum drawdown | ≤20% | **55.95%** |
| Daily dollar-PnL Profit Factor | >1.3 | **0.841** |
| Final $10,000 account | profitable | **$5,545.01** |
| Positive months | ≥39/65 | **11/65** (34 flat, 20 negative) |
| Longest losing-month streak | ≤3 | **5** |
| Capacity-passing traded weeks | ≥150/279 | **90/279** |

The pre-entry 2% participation screen forced cash for 189 of 279 weeks. The terminal 133-day holdout had **zero** return, so it did not pass the positive-return rule. Cash dates remained in all 65 months and full-window metrics. Ex-post fill participation diagnostics are in the JSON result and were never used to select trades.

The annual fixed-rule forward folds returned **−27.10%, −12.73%, −9.45%, −5.10%, +1.43%**. Only the last fold passed its positive-return/Sharpe check, and it had just two nonflat daily records. Three of four BTC bull/bear × volatility states lost money; bear/high was positive but did not rescue the combined account. The precommitted 5,000-path circular 21-day block Monte Carlo had **0 joint Gate passes**. Its median CAGR was **−10.22%** and median maximum drawdown **52.43%**.

All fixed stress variants also lost: doubled slippage ended at **$5,063.22**, one extra five-minute entry delay at **$5,415.44**, and removal of favorable funding credits at **$4,921.16**. Baseline executed-price PnL including adverse slippage was **−$3,799.78**, taker fees **−$703.65**, and actual net funding **+$48.44**, exactly reconciling to **−$4,454.99** net account PnL. Independently, the product of all 65 monthly factors, sum of all four state PnLs, and sum of price/fee/funding PnLs each reconciled with the final equity.

Every full-Gate route is closed. This result is a failed test of this **specific** illiquidity score/universe/venue/capacity/cost rule, not a general proof that liquidity risk premiums never exist. No paper or live order follows from it. The original mature strategy and unrelated futures/hybrid/rotation/paper state remain untouched.
