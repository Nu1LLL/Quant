# Sealed result: ETH fee burden per transaction

Preregistered alone in commit `58da924`, implemented and pushed in `dae8537`, then run once on 2026-11-08. Full machine-readable evidence is [RESULT.json](RESULT.json). **Rejected; all Gate checks did not pass.** No paper or live order was placed. The protected mature strategy and earlier sealed results were not modified.

The frozen rule compared the ETH-denominated transaction-fee sum divided by transaction count over the most recent permitted 28 UTC days with the preceding 28 days. It bought ETH spot when the recent burden was higher, otherwise held USDT. Decisions were Mondays using data through Friday, with 10 bp taker fee and 5 bp adverse slippage per fill. Initial capital was 10,000 USDT; gross exposure never exceeded 1x. The account covered all **1,826 days / 60 months of 2021–2025** and had 23 buys and 23 sells, including terminal liquidation.

| Net account diagnostic | Frozen requirement | Result |
| --- | ---: | ---: |
| CAGR | ≥40% | **13.48%** |
| Daily Sharpe | >1.5 | **0.503** |
| Conservative maximum drawdown | ≤20% | **66.77%** |
| Daily-dollar Profit Factor | >1.3 | **1.043** |
| 2024–2025 retrospective holdout CAGR / Sharpe / drawdown / PF | All four pass | **8.23% / 0.393 / 45.44% / 1.047** |
| Strictly profitable months / longest nonpositive streak | ≥36/60 / ≤3 | **21/60 / 7** |
| Annual fold with positive return and Sharpe | All five | **2022 failed** |
| Four market states | Each ≥90 days and positive PnL | **Bear/low-volatility failed** |
| 5,000-path, 21-day block Monte Carlo joint pass | ≥80% | **0 / 5,000** |
| Independent executable OOS | Required | **Unavailable** |

The account ended at **18,816.67 USDT** after **771.19 USDT in fees** and **385.60 USDT reference slippage drag**. Annual returns were 2021 **+136.93%**, 2022 **−51.38%**, 2023 **+39.48%**, 2024 **+0.78%**, 2025 **+16.25%**. The worst month was **−28.78%**. Doubled slippage and a one-day execution delay ended at 18,388.75 and 17,849.02 USDT respectively, but neither rescues the required full-period and holdout Gate. Same-cost ETH buy-and-hold ended at 40,231.64 USDT with 32.11% CAGR and 81.89% drawdown; this comparator also fails the Gate.

Data integrity: Coin Metrics Community returned the exact preregistered 1,887-row `FeeTotNtv,TxCnt` response hash. All 63 official Binance ETHUSDT spot ZIPs matched their existing pinned checksums; 62 historical months had complete UTC grids, and only the preregistered first opening price was read from January 2026. The result contains source hashes, fill ledger, annual folds, monthly returns, regimes and Monte Carlo details. Independent checks reconciled 46 fills to the fee/slippage totals, 1,826 daily factors and regime days, final account cash, and Monday-only signal fills.

Evidence grade remains `preregistered_current_vintage_retrospective_eth_blockspace_fee_burden`: the on-chain series is a current reconstructed vintage, the 2024–2025 split was historical, and daily OHLC bars do not demonstrate executable order-book fills. **`independent_executable_oos=false` regardless of numerical performance.** The rule, asset, dates, sign and parameters are sealed and will not be optimized on this same history. This result is not a profitable monthly strategy or a trading authorization.
