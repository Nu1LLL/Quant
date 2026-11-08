# Ethereum DeFi TVL relative-flow study — sealed rejection

**One formal run:** preregistration `ad00eed`, boundary amendment `110964e`, frozen implementation `6845108`, immutable machine result [`RESULT.json`](RESULT.json). The source audit checked 1,887 daily DeFiLlama TVL points against the preregistered subset hash and 63 official Binance spot monthly ZIPs against the committed hashes and live official checksums. The 2026 January ZIP contributed only the terminal first-row open. No real order was placed.

The rule used a Wednesday decision from the preceding Sunday's Ethereum DeFi TVL divided by ETH price, compared with 28 days earlier, and traded spot ETH or USDT at Thursday's open. It used 10,000 USDT, no leverage, 10 bp taker fee plus 5 bp adverse slippage per fill, and a forced terminal sale. It is a rough capital-commitment proxy, **not** observed net DeFi deposits. Historical TVL is backfilled without daily release vintages, so `independent_executable_oos=false` regardless of the backtest.

| Fixed check | Formal result | Required | Outcome |
|---|---:|---:|---|
| Five-year net CAGR | **26.95%** | ≥40% | Fail |
| Daily Sharpe | **0.713** | >1.5 | Fail |
| Conservative max drawdown | **60.51%** | ≤20% | Fail |
| Daily-dollar profit factor | **1.066** | >1.3 | Fail |
| 2024–2025 holdout CAGR / Sharpe / drawdown / PF | 14.31% / 0.516 / 60.51% / 1.034 | all four numerical Gate thresholds | Fail |
| Positive calendar months | **28/60**; longest nonpositive streak 5; worst month −32.21% | ≥36/60; streak ≤3 | Fail |
| Annual walk-forward | 2021 +135.6%; 2022 −13.9%; 2023 +24.4%; 2024 +97.3%; 2025 −33.9% | every year positive return and Sharpe | Fail |
| Four market states | bear/high −4,429 USDT; bear/low −22,486 USDT | all four positive, ≥90 days each | Fail |
| 5,000-path 21-day block Monte Carlo | **0/5,000** jointly pass | ≥80% | Fail |
| Independent executable OOS | unavailable | required | Fail |

The account ended at **32,965.25 USDT** before any tax, after 66 fills and 1,860.22 USDT in explicit fees; the reference-price slippage debit summed to about 930.12 USDT. The no-cost path ended at 36,395.82 USDT. Double-slippage and one-day-latency stress paths finished at 31,895.15 and 44,809.89 USDT respectively, but those two positive terminal values do not fix the Sharpe, drawdown, monthly, market-state, Monte Carlo or evidence failures. The 5th-percentile Monte Carlo CAGR was −14.78%; its median max drawdown was 65.56%.

**Decision:** reject and do not deploy or reselect the better latency path. The first official result is sealed; do not retune this economic hypothesis on 2021–2025. The protected mature strategy and its formal reports were untouched. Future work must use an economically independent hypothesis with a new preregistration, or gather prospective point-in-time evidence under a separate protocol.

Sources: [DeFiLlama free TVL API documentation](https://github.com/DefiLlama/api-docs/blob/main/llms-free.txt), [Binance public archive format](https://github.com/binance/binance-public-data), [DeFi loanable-funds paper](https://arxiv.org/abs/2006.13922).
