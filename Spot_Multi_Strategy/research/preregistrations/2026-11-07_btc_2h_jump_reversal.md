# Pre-registration: BTC two-hour extreme-move reversal

Registered on 2026-11-07 before reading any BTCUSDT perpetual price, return,
trade, or strategy-result record for this experiment. The only prior data
inspection was archive format, checksum, and one-minute timestamp-grid
integrity. This document must be committed alone before the formal result is
computed. Experiment ID: `btc_2h_jump_reversal_2026_11_07`.

## Economic hypothesis and evidence grade

An unusually large two-hour BTC move can reflect temporary overreaction or
leveraged liquidations; the next two hours may reverse enough to pay for
market-order costs. De Nicola, *On the Intraday Behavior of Bitcoin* (Ledger,
2021), studied Bitstamp BTC from 2015 to 2018 and reported a 0.74% **gross**
mean return per trade for the two-hour, four-standard-deviation variant. Its
table did not deduct fees. Source:
https://ledger.pitt.edu/ojs/ledger/article/download/213/212/1232 .

This is an economically distinct test from the repo's low-volume altcoin
overnight reversal, which invokes market-maker inventory. The exchange,
contract, sample, rolling volatility, and execution assumptions below are a
new, frozen implementation of the published overreaction mechanism. The
2021-2025 results are currently unread. Evidence grade is **retrospective
blind historical holdout**, never prospective live OOS. Its historical
publication timestamps and executable best bid/ask are not established by a
monthly K-line checksum. A passing simulated PnL cannot alone pass the full
independent execution Gate.

## Fixed data and splits

- One instrument only: Binance USD-M `BTCUSDT` perpetual. No spot, altcoins,
  other exchanges, or cross-asset selection for this experiment.
- Source: official Binance Vision monthly USD-M `1m` K-line ZIPs with the
  matching `.CHECKSUM`. The 2020-01 through 2025-12 archives were inspected
  for SHA-256 and complete UTC one-minute timestamp grids before registration;
  all 72 months passed. The contemporaneous checksums and expected row counts
  are pinned in `2026-11-07_btc_2h_jump_reversal_archive_manifest.csv`.
  No price or return field was inspected.
- Funding: official Binance Vision monthly USD-M `fundingRate/BTCUSDT` ZIPs,
  also SHA-256 checked. The 2021-01 and 2025-12 endpoints were confirmed to
  exist and to use `calc_time,funding_interval_hours,last_funding_rate`.
  Their first CSV rows, including one rate value each, were displayed only
  during this format check; no rate series or strategy outcome was inspected
  or used to choose a parameter. Missing months, bad
  checksums, duplicate timestamps, non-finite rates, or unexplained funding
  gaps over 12 hours fail the data Gate before PnL is accepted.
- 2020-01-01 through 2020-12-31 UTC is warm-up only. OOS is exactly
  2021-01-01 00:00 through 2025-12-31 23:59 UTC. Five full-calendar
  walk-forward folds are 2021, 2022, 2023, 2024, and 2025. No fitting,
  feature selection, weight optimization, or threshold revision is allowed;
  only the past 360 completed two-hour returns update the volatility estimate.
- Each monthly K-line must pass the existing archive integrity audit. A
  missing, duplicate, off-grid, or invalid bar fails the data Gate. No
  synthetic fill, silent interpolation, skipped year, or alternate market.
  Archive contents are streamed; raw ZIP/CSV files are not retained.

## Signal, execution, and risk

- UTC-aligned two-hour closes are the close of the last one-minute K-line in
  each interval. At a two-hour boundary `t`, compute
  `r_t = ln(C_t/C_(t-2h))`. Compute sample standard deviation, `ddof=1`,
  from the **previous 360** completed two-hour log returns, excluding
  `r_t`. If volatility is zero or history incomplete, do nothing.
- If `abs(r_t) >= 4*sigma_t`, take the opposite side: positive move ->
  short; negative move -> long. At most one position may be open. Signals
  occurring while a position is open are ignored. There are no other
  filters, stop-losses, take-profits, or discretionary overrides.
- Signal is treated as known only after the interval closes. Model a
  one-minute decision/transport delay. Reference entry is the open of the
  one-minute bar beginning at `t+1min`; reference exit is the open at
  `t+121min` (120-minute holding time). A missing reference bar fails the
  data Gate. These are bar-based proxies, **not** observed executable quotes.
- Initial equity is 10,000 USDT. Each entry uses 100% of then-current
  positive equity as USD notional at fixed 1x leverage, no pyramiding.
  If equity is nonpositive, stop and reject. 3x/5x are only separately
  labeled stress scenarios if the 1x cost-net signal first passes quality
  checks; leverage never repairs a failed Sharpe or creates a Gate pass.
- Market-order fee is fixed at 5 basis points of executed notional on
  **each** side, with no VIP/BNB discount. Adverse spread/slippage is 10
  basis points per side and additional adverse impact/latency is 5 basis
  points per side: total 20 basis points per side, 40 basis points
  round-trip at an unchanged reference price. Long buys / short covers
  execute above the reference price; long sells / short entries below it.
  A second, predeclared cost stress doubles the non-fee adverse component
  to 30 basis points per side while retaining the 5-basis-point fee.
- Apply each actual archived funding rate if a position is open at its
  `calc_time`: long funding PnL is `-rate * notional`, short is
  `+rate * notional`. Also report a conservative scenario that discards
  beneficial funding credits. Do not substitute a current rate for a
  historical one. No WhaleWatcher signal is used in this experiment.
- Calculate daily UTC equity with zero-return idle days, mark open positions
  to one-minute close, and measure intraminute drawdown conservatively using
  the low for longs and high for shorts. A 20% drawdown breach is a failed
  Gate even if later recovered. No order is sent.

## Frozen pass/fail checks

1. Data/causality: the complete OOS and warm-up grids and funding series
   pass; every feature and fill references information available at or
   before its declared decision time, plus the declared entry delay.
2. Net five-year portfolio metrics at 1x: CAGR at least 40% (report whether
   it lies in the target 40-60% band), daily Sharpe strictly above 1.5
   using `sqrt(365) * mean(daily return) / sample_sd(daily return)` and
   zero risk-free rate, max drawdown no worse than 20%, and daily profit
   factor strictly above 1.3 (sum positive daily PnL divided by absolute
   sum negative daily PnL; no-loss or zero-trade cases are insufficient
   evidence, not infinity). Report trade count, monthly positive/negative/
   flat counts, longest losing-month streak, and worst month.
3. Annual walk-forward: each of five full-year folds must have positive
   net return and positive daily Sharpe, with no parameter changes. Folds
   with no trade fail. Regime audit uses only lagged state labels: prior
   90-day BTC return sign and prior 30-day realized volatility versus the
   preceding 365-day median. Report the four bull/bear × high/low-volatility
   buckets. Each bucket must have at least one trade and positive net PnL;
   an empty bucket is insufficient evidence.
4. Monte Carlo: 5,000 circular 21-day block bootstrap paths from the full
   OOS daily **net** return series; deterministic seed `20261107`, each
   resampled path has the original day count. At least 80% of paths must
   jointly satisfy CAGR >=40%, Sharpe >1.5, max drawdown <=20%, and daily
   profit factor >1.3. The doubled adverse-cost stress must remain net
   profitable. Report all path distributions and joint pass count.
5. Independent evidence/execution: label this historical holdout honestly.
   Monthly trade bars do not establish the original historical data version,
   contemporaneous publication, bid/ask, queue, or exact $10,000 fill.
   Full Gate remains **unpassed** until a genuinely independent,
   replayable point-in-time source and executable cost evidence verify it.

Run this one specification once. On the first failed check, still publish the
predeclared diagnostics that can be computed without changing rules, then
seal the candidate as rejected or evidence-insufficient. Do not optimize
on these five years, shift sample dates, delete losing periods, change asset,
or cite the paper's gross return as this experiment's net performance.
Even a full future Gate pass would permit only small-position paper trading.
