# Seven-alt daily idiosyncratic-volatility risk premium: rejected

**Formal experiment:** `seven_alt_daily_idiosyncratic_volatility_2026_11_07`

**Preregistration:** `ac0561c` committed and pushed alone before any formal signal/rank/PnL read.

**Frozen implementation:** `3cdf760`; the one-shot runner recorded that exact commit.

**Machine result:** `seven_alt_daily_idiosyncratic_volatility_2026_11_07.json` (write-once).
**Decision:** **reject**. Full Gate and every primary economic performance gate fail. No sign reversal, shortened hold, asset/period deletion, changed capacity, fee discount or leverage rescue on this history.

## What was tested

[Günther, Fieberg and Poddig (2020)](https://www.econstor.eu/bitstream/10419/270511/1/10_3790_vjh_89_4_07.pdf) observed a cost-free positive high-minus-low return spread for six-month CAPM-residual volatility in a much broader crypto cross section. This precommitted transfer used the BTC market factor, 182 completed daily returns and seven fixed Binance USD-M alts; each day it bought the two highest residual-volatility names and shorted the two lowest. $10,000 capital, four 25% legs, 1x gross exposure, one-day holding, taker fee, adverse slippage and actual archived funding were included. Prior-day quote-volume eligibility targeted ≤2% participation. The official pinned **552 five-minute price** and **552 funding** archives were rechecked, with no source failure. Formal dates were 2021-07-09 through 2026-09-30 (1,910 daily records); the last day was flat because its exit would exceed the pinned data.

## Net account result

| Measure | Precommitted Gate | Observed |
|---|---:|---:|
| Final equity | Above $10,000 | **$85.17** |
| CAGR | ≥40% (40–60% target band) | **−59.80%** |
| Daily Sharpe | >1.5 | **−3.357** |
| Conservative max drawdown | ≤20% | **99.17%** |
| Daily dollar-PnL Profit Factor | >1.3 | **0.610** |
| Trade days | ≥1,000/1,909 opportunities | **1,909/1,909** |
| Positive months | ≥38/63, losing streak ≤3 | **6/63; longest losing streak 16** |
| Annual walk-forward folds | All five positive | **0/5** |
| Final 84-day fold | Positive | **−11.42%** |
| Four bull/bear × high/low states | Each ≥90 days and positive net PnL | **All four negative** |
| 5,000 circular 21-day block Monte Carlo paths | ≥80% joint numeric pass | **0/5,000** |

The five annual fold net returns were −62.72%, −57.42%, −63.05%, −58.59% and −60.41%; each daily Sharpe was negative. All seven alts passed the prior-day capacity check on most days; there were **zero capacity-cash days**, so sparse trading does not explain this result. There were **31** individual ex-post entry/exit fills above the estimated 2% volume participation ceiling and no zero-volume fills, further limiting execution confidence. The worst month was April 2024 at −18.46%.

The baseline net loss of **$9,914.83** reconciles to executed-price PnL **−$6,071.81** (including modeled adverse slippage), taker fees **−$3,667.72**, and funding **−$175.31**. Thus the tested return did not overcome modeled execution even before adding fees and funding. Double-slippage, one-extra-bar entry latency and no-favorable-funding-credit scenarios each finished below $10,000 (respectively **$12.55**, **$80.93**, **$64.98**). Five-thousand Monte Carlo paths had zero joint numeric passes; its median resampled CAGR was −59.80%. The result JSON has 63 monthly returns, 1,910 market-state dates, 1,909 trades and exact PnL-to-equity reconciliation.

## Evidence limit

This is a post-2020-paper **historical transfer**, not independent program-level or prospective executable OOS. Multiple earlier hypotheses already used this 2021–2026 history. Five-minute candles do not supply contemporaneous bid/ask depth, queue or actual fills; the preregistered `independent_executable_oos` Gate remains false. Even under the modeled costs, the numerical results reject this mechanism. No paper trade or real-money order was opened.
