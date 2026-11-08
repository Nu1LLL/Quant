# Sealed result: BSC stablecoin share and BNB spot

The [one-rule preregistration](../../research/preregistrations/2026-11-08_bsc_stablecoin_share_bnb.md) was committed and pushed alone in `de978ed`; implementation was pushed in `26d160a` before any stablecoin amount, price or return was read. The formal run occurred once on 2026-11-08. Full machine-readable evidence is [RESULT.json](RESULT.json). **Rejected: the complete Gate failed.** No paper or live order was placed. The mature baseline and previous sealed results were untouched.

The frozen test bought BNB spot when the BSC share of all-chain USD-pegged stablecoin token counts was higher than 28 days before, otherwise held USDT. The Wednesday decision used only the preceding Sunday chart; target changes filled Thursday. Cost assumptions were 10 bp taker fee and 10 bp adverse slippage per fill, with gross exposure ≤1x and a 10,000 USDT initial account. The exact **2021–2025** account had **1,826 daily factors, 60 months, 22 buys and 22 sells** including forced terminal liquidation.

| Net diagnostic | Frozen Gate | Result |
| --- | ---: | ---: |
| CAGR | ≥40% | **23.91%** |
| Daily Sharpe | >1.5 | **0.635** |
| Conservative maximum drawdown | ≤20% | **75.69%** |
| Daily-dollar Profit Factor | >1.3 | **1.072** |
| 2024–2025 retrospective holdout CAGR / Sharpe / drawdown / PF | All four pass | **18.29% / 0.663 / 36.88% / 1.104** |
| Strictly profitable months / longest nonpositive streak | ≥36/60 / ≤3 | **22/60 / 7** |
| Every annual fold positive with positive Sharpe | Required | **2022 and 2023 failed** |
| Four market states | Each ≥90 days and positive PnL | **Both bear states failed** |
| 5,000-path, 21-day block Monte Carlo joint pass | ≥80% | **0 / 5,000** |
| Better than same-cost BNB buy-and-hold in CAGR and Sharpe | Required | **Failed** |
| Independent executable OOS | Required | **Unavailable** |

The account ended at **29,202.86 USDT**, after **1,034.87 USDT in fees** and **1,034.89 USDT reference slippage drag**. Annual returns: 2021 **+242.31%**, 2022 **−36.13%**, 2023 **−4.53%**, 2024 **+12.26%**, 2025 **+24.67%**. Worst month: **−46.34%**. Doubled slippage ended at 27,946.11 USDT; a one-day delay ended at 31,602.26 USDT. Same-cost BNB buy-and-hold ended at 230,425.06 USDT with 87.30% CAGR, 1.144 Sharpe and 73.49% drawdown; that comparator also fails the full Gate.

Data integrity: both DeFiLlama current-vintage histories matched their preregistered selected-subset hashes and 1,857-day UTC grids. All 62 BNB monthly ZIPs matched official checksums and pinned hashes; the 61 historical months had complete daily grids and the January 2026 terminal archive was checked for all 31 daily timestamps while only its first open entered the account. Independent arithmetic checks reconciled 44 fills to total fees and slippage, 1,826 daily and regime records, terminal cash, and Thursday-only signal fills.

The evidence grade remains `preregistered_retrospective_current_bsc_stablecoin_share_vintage`: today's reconstructed chain totals do not establish what was published on historical decision dates, the holdout is retrospective, and OHLC data cannot verify executable fills. **`independent_executable_oos=false`** regardless of the historical returns. The rule, asset, sign, sample and costs are sealed; no parameter or chain selection will be tried on this same history to rescue it.
