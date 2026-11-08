# Preregistration: U.S. initial-claims labor improvement and BTC spot/cash

**Experiment:** `us_weekly_claims_btc_2026_11_08`. Commit and push this
preregistration alone before parsing any `ICSA` values, BTC prices, positions,
returns or Gate outcomes. One fixed hypothesis and one formal run. Failure
seals the rule and data path; no sign flip, threshold search, year deletion,
asset switch, extra leverage, source substitution or second formal run.

## Mechanism, source and evidence

Falling U.S. initial unemployment claims may signal stronger labor demand and
risk appetite, potentially favoring BTC ownership over cash. Rising or flat
claims lead to cash. This is an employment-flow mechanism distinct from prior
CPI inflation, Federal Reserve balance-sheet, Treasury cash, VIX, on-chain,
funding, volume and price-trend hypotheses. The relationship may instead be
negative if strong labor data raises expected interest rates. The sign is
fixed here before viewing outcomes. The [U.S. Department of Labor archive](https://oui.doleta.gov/unemploy/claims_arch.asp) states that weekly releases
normally occur Thursday 08:30 Eastern, with holiday exceptions. [FRED ICSA](https://fred.stlouisfed.org/series/ICSA) is the seasonally adjusted, week-
ending-Saturday initial-claims series sourced from the U.S. Employment and
Training Administration; it is explicitly subject to revision.

Evidence grade:
`preregistered_retrospective_current_vintage_labor_proxy`;
`independent_executable_oos=false`. Today's FRED series is not the first
published vintage for each historical Monday, and the research program has
already seen broad BTC history. The 2024–2025 split is a fixed retrospective
holdout, not genuinely independent prospective OOS. Even excellent numbers
cannot clear the full Gate or authorize orders. This experiment does not
import WhaleWatcher AI advice or historical signals.

## Frozen sample and source audit

- Download exact CSV
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=ICSA` once in the formal
  run; require SHA-256
  `f84a9c9c6831eeb41b5b15125008f5d48d72c91453d75ad9e05118d40702f428`.
  Price-blind preflight found header `observation_date,ICSA`, 3,122 ascending
  weekly dates from 1967-01-07 through 2026-10-31, 52 weeks in each of
  2019–2021 and 2023–2025, 53 in 2022, and no missing-value markers in
  2019–2025. Require exactly these metadata and finite positive integer
  values for every week used. A changed response or gap seals `DATA_FAILURE`.
  No CSV is retained or committed.
- Reuse the immutable 2020-12 through 2026-01 Binance Vision BTCUSDT 1d spot
  monthly URL/checksum manifest, SHA-256
  `581c681e6d14065242ae8ff464b313fca02038fc16a6e158f37a203ae3cd4a5f`.
  Recheck official `.CHECKSUM`, ZIP SHA, sole CSV member and complete UTC daily
  OHLCV grid. January 2026 supplies only the first open. Do not save raw ZIPs.
- Start 10,000 USDT cash at 2021-01-01 00:00 UTC and force liquidate at the
  2026-01-01 00:00 UTC open. The 1,826 daily factors cover five full calendar
  years. 2021–2023 are fixed retrospective development/transfer;
  2024–2025 are the fixed retrospective holdout. No fitted parameter.

## Fixed signal, lag and execution

For each completed Saturday week `w`, compute the mean of its four most recent
ICSA values (`w` through `w−3` weeks) and the corresponding four values
13 weeks earlier (`w−13` through `w−16`). Target 100% BTC only if the recent
mean is **strictly lower**; otherwise target 100% USDT cash. Missing or
nonpositive source data is a source failure, not an interpolation.

Each `w` may first affect the **Monday UTC 00:00 open nine calendar days after
the week-ending Saturday**. This is after the normal Thursday release and
also after a holiday-shifted Wednesday release. Fixed stress delays execution
by another seven days. The source remains current-vintage, so the lag does
not cure revision hindsight. A target change trades at that Monday open;
unchanged targets incur no fill. No price, news, forecast or future return
enters the signal.

Unleveraged long spot/cash only; no funding, borrow, short, futures or yield.
Each buy/sell pays 10 bp taker fee and 5 bp adverse slippage, rounded down to
0.00001 BTC. Require at least 10 USDT notional, positive cash/equity, gross
notional ≤1x equity and order size ≤1% of the preceding 30 completed days'
median Binance quote volume. Conservative daily drawdown assumes high before
low. Reconcile fills, fees, daily equity and forced terminal cash. Run same
rule at zero cost, 10 bp slippage per fill, one-week extra lag, and same-cost
BTC buy-and-hold. Both cost/lag stress endpoints must exceed 10,000 USDT.

## Full Gate and stopping rule

Full net account: CAGR ≥40% (also note preferred 40–60%), daily Sharpe >1.5
using `sqrt(365)` and zero risk-free, conservative max drawdown ≤20%, and
daily-dollar Profit Factor >1.3. Require ≥20 BTC buys. Each 2021–2025 year
must include a fill, end above its start and have positive daily Sharpe.
The 2024–2025 holdout must independently meet all four numeric limits.
Require ≥36/60 strictly profitable months, no more than three consecutive
nonpositive months, and report worst month.

Validation-only regimes use the prior 31 completed BTC closes: positive
30-day log return is bull, otherwise bear; annualized sample return SD ≥80%
is high volatility, otherwise low. Each bull/bear × high/low cell needs at
least 90 days and positive net-dollar PnL. Exactly 5,000 circular 21-day
block-bootstrap paths, NumPy PCG64 seed `20261113`, jointly resample daily
net factors and normalized conservative high/low marks. At least 80% of
paths must jointly satisfy all four numeric limits. Report p05/p50/p95,
ruined and undefined paths.

Any data, execution, cost, numeric, annual, holdout, monthly, regime,
bootstrap, stress or independent-OOS failure rejects and seals. Full Gate
cannot pass while `independent_executable_oos=false`. No real orders.
