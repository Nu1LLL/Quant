# Three-chain DEX activity rotation — failed one-shot result

**Decision:** reject and seal the fixed 2021–2025 rule. Pre-registration `09edb76` was pushed before any DEX amount or spot price was read; implementation/test commit `9d89600` was pushed before the single formal run. The sealed machine record is `research/results/three_chain_dex_volume_rotation_2026_11_08.json`. No parameter, asset, weekday, side, cost or leverage revision is permitted on this history. No paper or live order.

## Source and evidence

The fixed [DefiLlama DEX daily-volume endpoint](https://api-docs.defillama.com/) returned complete, hash-pinned **1,857-day** current-vintage histories for Ethereum, BSC and Tron from 2020-12-01 to 2025-12-31. The 248 consumed BTC/ETH/BNB/TRX spot monthly ZIPs matched committed pins and fresh official `.CHECKSUM` records; the [Binance public-data format](https://github.com/binance/binance-public-data) and UTC daily grids were validated. The rule used 2021–2025 as 1,826 account days, with 2024–2025 held out from rule selection. DefiLlama's original daily release vintages, historical best bid/ask and fills are unavailable, so `independent_executable_oos=false` regardless of numeric results. This is a retrospective historical transfer only.

## Fixed net account versus Gate

| Measure | Frozen target | One-shot result |
|---|---:|---:|
| Starting → ending account | $10,000 → growth | **$10,000 → $26,427.49** |
| Net CAGR | ≥40% (40–60% target) | **21.46%** |
| Daily Sharpe | >1.5 | **0.588** |
| Conservative max drawdown | ≤20% | **87.10%** |
| Daily-dollar Profit Factor | >1.3 | **1.030** |
| Profitable calendar months | ≥36/60 | **30/60**, longest nonpositive streak 4 |
| Completed buys | ≥20 | **84** (19 ETH, 26 BNB, 39 TRX) |
| 2024–2025 historical holdout | four numeric targets | CAGR **13.14%**, Sharpe **0.467**, MDD **63.75%**, PF **1.038** |
| 21-day block Monte Carlo joint pass | ≥80% of 5,000 | **0/5,000** |

Annual net CAGR was **81.73%, 32.30%, −14.10%, 116.62%, −41.01%** for 2021 through 2025. Two years lost money; the annual walk-forward Gate failed. Both bear/high and bear/low BTC states lost money; bull/high had only 88 days against the required 90, so the four-state Gate failed. The fixed equal-dollar ETH/BNB/TRX buy-and-hold comparator had **60.62% CAGR** and **1.014 Sharpe**; the DEX rule did not beat it on either measure and had a deeper drawdown (comparator 71.91%).

The identical-signal zero-cost path had 36.92% CAGR, below the 40% threshold even before fees/slippage. The baseline charged $5,139.00 in cumulative explicit fees; summed reference slippage was $10,278.13, which is an execution reference measure rather than an additive terminal-equity difference. Double-slip and Friday-entry stress paths ended above $10,000, but neither rescues the failed rule. The Friday path's higher return is **not** authorization to change the registered Thursday execution. There were 22 next-day protective-stop sales and the maximum drawdown remained far above target.

This result supplies no validated deployable alpha. The historical DEX-activity ranking is rejected as a standalone strategy and must not be inserted into the live or paper strategy merely because one stress path profited. The mature baseline and unrelated user systems were untouched.
