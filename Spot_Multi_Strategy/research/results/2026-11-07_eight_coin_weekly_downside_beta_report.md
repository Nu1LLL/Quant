# Sealed result: eight-coin weekly downside beta

**Decision: REJECTED.** The [preregistration](../preregistrations/2026-11-07_eight_coin_weekly_downside_beta.md) was committed and pushed alone at `1e6cb62`, before any downside-beta rank, trade or return was read. The frozen replay and unit checks were committed and pushed at `9af365c`; the [one-shot result](eight_coin_weekly_downside_beta_2026_11_07.json) was then generated exactly once. Do not reverse the sign, change the BTC proxy, capacity/hold/lookback/cost rules, remove coins or years, or apply leverage to rescue this same-history test.

## Hypothesis and data

[Dobrynskaya (2020)](https://wp.hse.ru/data/2020/09/16/1583081932/79FE2020.pdf) found a positive historical premium for high-minus-low downside-market-beta crypto portfolios. This new **transfer test** used the fixed BTC proxy and eight verified Binance USD-M contracts, a 26-completed-week conditional OLS beta, weekly long-two-high/short-two-low positions, 1.0x gross, and seven-day holds. This differs materially from the paper's broad capitalization-weighted market proxy and large coin cross section. Its historical gross return is not this account's expected return. All 552 official five-minute price and 552 funding archives, 2021-01 through 2026-09, matched pinned SHA-256 and official checksums and passed full grid/numeric/event checks before replay. The formal $10,000 account has 1,910 consecutive daily records from 2021-07-09 through 2026-09-30. Five-minute candles lack historical executable BBO/depth, queue and fill evidence; prospective executable OOS is absent.

## Fixed-cost net account and Gate

| Measure | Required | Observed |
|---|---:|---:|
| CAGR | ≥40% | **+1.17%** |
| Daily Sharpe | >1.5 | **0.152** |
| Conservative five-minute maximum drawdown | ≤20% | **30.33%** |
| Daily dollar-PnL Profit Factor | >1.3 | **1.020** |
| Ending $10,000 account | profitable | **$10,625.88** |
| Positive calendar months | ≥38/63 | **18/63** (31 flat, 14 negative) |
| Traded weeks | ≥150/272 | **87/272** |

No week failed the precommitted five-negative/five-positive BTC-week requirement; **185/272** weeks failed the pre-entry 2% median-bar participation screen and stayed cash. The 84-day terminal holdout returned **+6.04%** and passed its isolated positive-return check, but two of five complete annual folds lost money: the five fold returns were **−8.23%, +7.48%, +3.70%, −8.71%, +7.31%**. Bear/low and bull/low BTC states lost money, while both high-volatility states made money. Only **18** of 63 calendar months were positive; longest losing-month streak was **3**. Ex-post, **17** simulated fills exceeded 2% of their actual five-minute bar quote volume despite the prior-day capacity screen; this diagnostic did not select or exclude trades.

The precommitted 5,000-path, 21-day circular block Monte Carlo produced **0** joint numerical-Gate passes. Its median CAGR was **+1.31%**, median Sharpe **0.161**, median max drawdown **34.52%**, and median Profit Factor **1.021**. Doubled slippage ended at **$9,736.92** and suppressing favorable funding credits at **$9,438.45**, both losses; one extra five-minute entry delay ended at **$10,673.29** but cannot rescue the primary Gate.

Executed-price PnL including adverse slippage was **+$1,387.22**, taker fees **−$798.29**, and actual net funding **+$36.96**, exactly reconciling to **+$625.88** account PnL. The product of all 63 monthly factors and sum of four state PnLs independently reconcile to the same ending equity. All 1,910 days, including cash days, were retained in the metrics and Monte Carlo.

The net gain is too small and too fragile for the requested return, Sharpe, drawdown, cost and monthly-stability Gates. This is a failure of this *specific* executable-candidate rule, not a general disproof of downside-risk pricing. The original mature strategy and unrelated futures/hybrid/rotation/paper state are untouched. No live or paper order is started from this rejected candidate.
