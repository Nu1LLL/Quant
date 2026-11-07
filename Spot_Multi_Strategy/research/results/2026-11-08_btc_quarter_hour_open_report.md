# BTC quarter-hour five-minute transfer: formal rejection

**Status:** sealed failure. The preregistration was committed as `24c62d9` before the formal price replay; the implementation was committed as `7108a32`. The first invocation had a calendar-constructor code error before reporting any strategy metric; the code-only repair and failure note were committed as `0de8d95` before the successful invocation. No asset, period, clock minute, cost, fill rule, Gate or model parameter changed. The immutable machine-readable first result is [`btc_quarter_hour_open_2026_11_08.json`](btc_quarter_hour_open_2026_11_08.json).

## Hypothesis and source

[Shanaev, Vasenin and Stepanov (2023)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10015199/) found a positive **first-minute** BTC return at 15-minute candle turns in their older multi-exchange sample. This experiment tested a distinct, harder transfer: buy Binance BTCUSDT USD-M perpetual at the 15/30/45-minute five-minute **open**, sell at the next five-minute open, and pay 5 bp taker fee plus 1 bp slippage on **each side**. Minute 0 was excluded in advance to avoid simultaneous funding settlement. It is a five-minute Binance futures result, **not** a one-minute Bitfinex replication or a claim that the original paper was wrong.

The registered asset and every one of the 69 official monthly five-minute source ZIPs, 2021-01 through 2026-09, were fixed before the strategy replay. Each ZIP matched its frozen SHA-256 and fresh official checksum and passed the complete UTC grid and numeric bar checks. The evaluation covered 2,099 UTC days and 151,128 scheduled trades. No price ZIP or CSV was saved locally.

## Net outcome against the frozen Gate

| Check | One-shot result | Required |
| --- | ---: | ---: |
| Geometric gross return per scheduled trade | -0.0623 bp | Economic direction positive |
| Geometric net return per scheduled trade | -12.0551 bp | Positive after costs |
| Net CAGR, theoretical continued replay | approximately -100% | at least 40% |
| Daily Sharpe | -105.44 | greater than 1.5 |
| Adverse intrabar maximum drawdown | 100% | at most 20% |
| Daily dollar Profit Factor | effectively 0 | greater than 1.3 |
| Positive months | 0 of 69 | at least 42 of 69 |
| Annual forward folds | all six negative | every full year positive; 2026 positive |
| 2024–2026 post-paper folds | all negative | each positive |
| 5,000-path, 21-day block Monte Carlo joint passes | 0 | at least 80% |
| Slippage and latency stress | both fail | both profitable |
| Independent executable OOS | unavailable | required |

The account fell below the preregistered **$100 practical-ruin threshold**. The machine-readable tiny terminal balance is a mathematical continuation of fixed percentage trades after that point, **not** an executable account forecast. All four classified market states lost money; one state had only 70 days against the 90-day minimum. The bootstrap recorded practical ruin on all 5,000 paths. Leverage cannot repair the negative gross edge and would worsen this failure.

## Decision and limits

Reject the registered five-minute quarter-hour rule. The result gives no basis for a live or paper order. The archived five-minute open is not a historical bid/ask or guaranteed fill, and the fixed 1 bp slippage is only an assumption. The one-minute effect discussed by the paper is a **different test** requiring one-minute data and an independently preregistered execution model; this result may not be reused to select a favorable minute, exchange, fee tier or date window on the same history. No follow-up parameter optimization is authorized on these 69 months.
