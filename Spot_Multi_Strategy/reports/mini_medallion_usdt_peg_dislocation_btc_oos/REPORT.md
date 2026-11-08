# USDT peg dislocation as a BTC risk gate: sealed rejection

**One formal run.** Preregistration `cceb076` was committed and pushed alone before reading the numeric source fields or this signal's outcome. Frozen implementation `8fac056` and four synthetic checks were committed and pushed before the run. The [immutable first result](RESULT.json) is `RESULT.json`; the official Coinbase `USDT-USD` eight response hashes and all 63 Binance BTCUSDT spot monthly ZIP/checksum pins matched. USDT/USD has 1,856 consecutive UTC daily candles from 2021-09-01 to 2026-09-30; BTC spot has 1,887 from 2021-08-01 to 2026-09-30 plus the 2026-10-01 first open. Coinbase BTC/USD was not decoded for this study. No real order was placed.

The frozen rule bought BTC spot when all seven USDT/USD closes dated `d−8` through `d−2` stayed in the **0.997–1.003** band, otherwise held USDT cash. A changed target filled at the Binance UTC daily open, with 10 bp taker fee plus 5 bp adverse slippage, 1× maximum BTC exposure and no short or funding. The entire account was marked in **USD using Coinbase USDT/USD**, including USDT cash, to avoid treating a discounted stablecoin as exactly one dollar. The initial 10,000 USDT marked at **$10,002**; the final account was **13,066.07 USDT / $13,057.57** after costs.

| Fixed Gate | Official-history result | Requirement | Verdict |
|---|---:|---:|---|
| Five-year USD net CAGR | **5.48%** | ≥40%; preferred 40–60% | Fail |
| Daily USD Sharpe | **0.359** | >1.5 | Fail |
| Conservative USD max drawdown | **78.90%** | ≤20% | Fail |
| Daily USD-dollar Profit Factor | **1.019** | >1.3 | Fail |
| BTC buys | **3** | ≥20 | Fail |
| Historical 2024–2026 holdout | CAGR 28.07%; Sharpe 0.757; drawdown 56.26%; PF 1.061 | all four numeric limits | Fail |
| Monthly stability | **32/60** positive; five consecutive nonpositive; worst −37.30% | ≥36/60; streak ≤3 | Fail |
| Calendar folds | 2022, 2025 and 2026 negative; no fills in 2024–2026 | every fold positive with a fill | Fail |
| Four market states | Bull/high only **20 days** and three of four states failed positive PnL | ≥90 days and positive USD PnL in each | Fail |
| 5,000 circular 21-day block paths | **0/5,000** joint passes | ≥80% | Fail |
| Independently executable OOS | unavailable | required | Fail |

The rule traded around only two historical dislocation windows: sold 2022-05-13, rebought 2022-05-20; sold 2023-03-12, rebought 2023-03-30. It then held BTC through the end. It incurred **45.16 USDT fees** plus about **22.58 USDT** reference-price slippage drag. Same-cost BTC buy-and-hold ended at **$19,013.72** with 13.71% CAGR; the frozen gate therefore did not improve either return or drawdown. The double-slippage and one-day-delay paths both ended positive ($13,018.48 and $13,542.19), but cannot repair the failed principal Gate. A zero-cost same-signal path ended at $13,175.77. Monte Carlo fifth-percentile CAGR was −28.86%, and median conservative drawdown 71.53%.

**Decision: reject and seal.** Do not reverse the dislocation signal, change the ±30 bp band or seven-day cooling period, add a price filter, raise leverage, delete 2022, or re-run this history under another configuration. Coinbase API candles are current-vintage and lack verified first-release timing or historical executable order books. The five-year backtest and 2024–2026 temporal slice are **retrospective**, not independent executable OOS. This result provides no basis for live or paper deployment under the required Gate.

Economic premise: [Lyons and Viswanath-Natraj, *What Keeps Stablecoins Stable?*](https://www.nber.org/papers/w27136). Source schema and history caveat: [Coinbase Exchange product-candle documentation](https://docs.cdp.coinbase.com/api-reference/exchange-api/rest-api/products/get-product-candles). Binance archive schema: [Binance Public Data](https://github.com/binance/binance-public-data).
