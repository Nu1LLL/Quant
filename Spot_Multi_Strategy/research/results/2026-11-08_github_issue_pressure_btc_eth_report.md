# GitHub core-issue pressure → BTC/ETH relative trade: sealed failure

**Preregistration:** pushed commit `8cb7cdb` before reading any selected issue count, price, funding, signal or PnL. **Code:** `bd05ac4`; one formal run, no parameter retry. Full machine-readable source and outcome are `github_issue_pressure_source_2026_11_08.json` and `github_issue_pressure_result_2026_11_08.json`. The 622 GitHub GraphQL queries were repeated independently; both reads produced SHA-256 `808bd4c4fc0cdb7c789c0e48fcf9891c591517d17a13fc04fcf15fc3273c99ee`. The official Binance pinned/checksum recheck passed all **138 five-minute price** and **138 funding** archives, 604,512 five-minute bars for two perpetuals. The time-varying GitHub issue index is still **not** a historical point-in-time vintage, and candles are not executable quotes.

The fixed rule used 299 Tuesday decisions from 2021-01-05 to 2026-09-22, with 108 active and 191 cash weeks; 108 entries passed the capacity gate. Its 2× gross, approximately market-neutral BTC/ETH account included 5-bp taker fee and 5-bp adverse slippage on each fill, actual published funding rates and a trade-open funding-notional proxy. Fee cost was **$900.30**, slippage **$900.30**, funding **+$9.68**, and mid-price PnL **−$4,079.45**; together they reconcile from $10,000 to **$4,129.63**.

| Fixed 2021-01-05–2026-09-30 test | Result | Required |
|---|---:|---:|
| Net CAGR | **−14.29%** | ≥40% |
| Daily Sharpe | **−0.400** | >1.5 |
| Conservative five-minute max drawdown | **84.59%** | ≤20% |
| Daily-dollar profit factor | **0.826** | >1.3 |
| Positive months | **24/69** | ≥42/69 |
| Longest nonpositive-month streak | **9** | ≤3 |
| 5,000-path 21-day block Monte Carlo joint pass | **0%** | ≥80% |

Calendar folds **2021, 2022, 2023 and 2024 each lost money**; 2025 and the partial 2026 fold were positive. The fixed retrospective 2025–2026 holdout had CAGR **44.25%**, Sharpe **1.602**, PF **1.546**, but max drawdown **29.40%**. This subset therefore still failed the holdout Gate; it is not a new strategy or independent OOS. The high-volatility bull regime contained only **70** days versus 90 required, and the bull-low regime lost money. The doubled-slippage, delayed-fill and no-favorable-funding stresses ended at **$3,288.60**, **$3,907.01** and **$2,866.07**, respectively, all below starting capital. The fixed 1× BTC-long/ETH-short comparator ended at **$1,199.83**; beating it does not rescue the candidate.

**Disposition:** `sealed_reject`; `all_gate_passed=false`; `independent_executable_oos=false`. Do not mine 2025–2026 alone, invert the sign, change the repository/threshold/asset, alter leverage or repeat this idea on the same history. No paper or live trade is authorized from this result. Existing mature strategy and prior sealed reports remain unchanged.
