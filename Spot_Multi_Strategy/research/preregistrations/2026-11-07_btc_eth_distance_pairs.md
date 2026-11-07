# Pre-registration — BTC/ETH hourly distance-pairs mean reversion

**Status: fixed before any BTC/ETH pairs result was read.** This registration
is to be committed separately **before** the BTC/ETH pair spread, paired funding accrual,
pair trades, or pair strategy results are computed. BTC historical prices
and funding rates were already processed, and their single-asset strategy
result seen, in a separate sealed BTC jump-reversal experiment. Thus this is only a
**pair-mechanism-internal blind historical holdout**, not untouched raw
prices or prospective out-of-sample evidence. Proposed experiment ID:
`btc_eth_distance_pairs_2026_11_07`.

## Independent mechanism and evidence

The hypothesis is that temporary **relative** dislocations between two
liquid, related crypto contracts converge within days even when their common
market direction is unpredictable. This differs from the sealed single-BTC
two-hour jump reversal, single-coin trend, and cross-sectional ranking of
individual Alpha scores. The pair, venue, and all parameters below
are fixed without looking at their 2021–2025 trading result.

The method is inspired by Fil and Kristoufek, *Pairs Trading in
Cryptocurrency Markets* (IEEE Access, 2020, DOI
10.1109/ACCESS.2020.3024619), and Fil's earlier
[source thesis](https://dspace.cuni.cz/bitstream/handle/20.500.11956/109634/130268190.pdf?sequence=1).
Their hourly baseline used 20-day formation, 10-day trading, 10-day jumps,
a two-standard-deviation entry, and a one-hour execution lag. The source
studied a historical Binance **spot** sample, did not report this fixed BTC/ETH USD-M
perpetual pair, and explicitly warned about costs and execution sensitivity.
We implement the published return-standardized, additive-index distance
method, but specify its ambiguous zero-cross and terminal-fill rules below.
This is **retrospective pair-mechanism-internal blind historical holdout**,
not prospective OOS.
Even favorable simulated PnL cannot pass the independent executable-data
Gate without point-in-time quotes and observed fills.

## Fixed assets, data, periods and completeness

- Exactly two contracts on Binance USD-M: `BTCUSDT` and `ETHUSDT`
  perpetuals. No other asset, pair search, exchange, or post-result
  selection. The only inputs to trade decisions are their historical
  price returns and this frozen rule. No WhaleWatcher or AI advice.
- Source: official Binance Vision monthly `1m` K-line ZIPs and matching
  `.CHECKSUM` for each symbol, 2020-01 through 2025-12. Before this
  draft, data-only audits checked 72/72 months for **each** symbol,
  matching SHA-256 and complete UTC minute grids, with zero missing,
  duplicate or off-grid minute. Those two audits did not inspect price
  or return fields; earlier single-BTC work did process BTC prices
  and funding rates.
  Exact month hashes and expected row counts must be pinned in
  `2026-11-07_btc_eth_pairs_kline_manifest.csv` before a result run;
  the measured ZIP SHA must match both the pinned manifest and the
  corresponding official `.CHECKSUM`. Failure to pin or later hash
  mismatch fails the data Gate.
- Hourly close at boundary `t` is the close of the minute
  `[t-1min,t)`. The minute archive also supplies only a **reference**
  open for delayed execution, not an executable bid/ask quote.
  Invalid OHLC, missing minute, month, or checksum is a data-Gate failure:
  no interpolation, synthetic fill, alternate archive, or skipped year.
  Process one month at a time; raw ZIP/CSV files are not retained.
- Actual archived USD-M `fundingRate` for **each** contract,
  2021-01 through 2025-12, with month CHECKSUM and parsed finite rate.
  Prior timestamp-only audits saw 60/60 months and 5,478 events for
  each contract; those timestamp-only
  audits did not inspect rate values, although the prior BTC-only
  experiment did process BTC funding rates. Missing/duplicate/unsorted
  rate, bad hash, or missing/extra event on the exact expected UTC
  timestamp grid fails the data Gate. A count or maximum-gap check
  alone is insufficient. The frozen expected set is every UTC day
  `2021-01-01` through `2025-12-31`, inclusive, at each anchor
  `00:00`, `08:00`, `16:00`, one actual `calc_time` in each
  half-open `[anchor,anchor+1s)` and no event elsewhere. The
  timestamp-only audit found exact coverage with actual offsets
  0–47 ms after the anchors; BTC and ETH each start at
  `2021-01-01 00:00:00.002` and end at
  `2025-12-31 16:00:00.001`. Use the **actual archived millisecond
  timestamp**, never a rounded anchor, for the holding interval,
  funding mark and daily attribution. Pin all funding ZIP SHA-256 values in
  `2026-11-07_btc_eth_pairs_funding_manifest.csv` before a result run
  and repeat them in the sealed result. Measured ZIP SHA must match
  both the pinned manifest and official `.CHECKSUM`; never substitute
  today's rate.
- `[2020-01-01,2021-01-01)` UTC is warm-up only; no entries.
  The fixed historical pair-test interval is exactly
  `[2021-01-01 00:00,2026-01-01 00:00)` UTC, 1,826 days.
  The five fixed annual walk-forward folds are 2021, 2022, 2023,
  2024, and 2025. No fitted parameter, pair, formation length, score
  threshold, cost or leverage is revised between folds.

## Hourly score and cycle rules

For each 10-day trading block `[S,T)`, where
`S = 2021-01-01 00:00 UTC + k*10d`, use only the prior 20-day
formation window of completed hourly bars `[S-20d,S)`. Its 480 return
endpoints are exactly `S-20d+1h, S-20d+2h, ..., S` and require 481
consecutive hourly closes from `S-20d` through `S`. Ordinary trading
score observations have endpoints `S+1h, ..., T-1h`; `z(T)` is never
used to generate an ordinary signal. There are 182 complete 10-day blocks
`k=0,...,181`; the final six OOS days are deliberately flat, with
no truncated or extra cycle.

1. For each asset `i` and hour ending `t`, set
   `r_i(t)=ln(P_i(t)/P_i(t-1h))`. From the 480 formation returns
   alone compute `mu_i` and sample `sd_i` (`ddof=1`). Hold both
   fixed throughout the ensuing 10-day trading block.
2. From the formation's first hourly return onward, set
   `u_i(t)=(r_i(t)-mu_i)/sd_i` and
   `R_i(t)=1+sum(u_i(s))` over all successive formation and
   trading hours through `t`. This is **additive** cumulative
   standardized return, not exponentiation or raw-price normalization.
   Define `d(t)=R_BTC(t)-R_ETH(t)`. Compute `mu_d` and sample
   `sd_d` from the 480 formation values of `d` only; hold them
   fixed in the trade block. The trade score is
   `z(t)=(d(t)-mu_d)/sd_d`. A zero or nonfinite `sd_i` or `sd_d`
   in any formation block fails signal-quality evidence for the
   candidate; do not omit that block, replace the score, or accept
   partial performance as a formal pass.
3. The formation's final `z(S)` is the previous observation for the
   first trading hour. A **new crossing** from `z_prev<+2` to
   `z_now>=+2` while flat schedules **short BTC / long ETH**.
   A crossing from `z_prev>-2` to `z_now<=-2` schedules
   **long BTC / short ETH**. Threshold equality triggers only
   on the crossing hour. No repeated entry merely because the score
   remains beyond a threshold. At most one pair position or pending
   order exists; signals while open/pending are ignored. Once scheduled,
   an entry or ordinary exit is not canceled or revised by later score
   observations before its fill, except that the known compulsory
   block exit takes precedence as specified below.
4. If short BTC/long ETH, the first observed `z<=0` after entry
   schedules an exit; if long BTC/short ETH, the first `z>=0`
   schedules an exit. This defines **reversion to the formation
   score mean** (`d=mu_d`), resolving the paper's ambiguity between
   `d=0` and `z=0`. Neither threshold nor exit target changes
   intrablock. No reversal is opened at the same observation as an exit.
5. Each ordinary signal is formed only after an hourly close at `t`.
   The first eligible fill proxy is the 1m open at `t+1h+1min`,
   a fixed one-hour-plus-one-minute delay. The order side is priced
   adversely as below. This extra minute is our conservative
   interpretation of the source's one-hour lag, not a claim of a
   historical market order fill.
6. The block end `T=S+10d` is known in advance. If still open,
   schedule compulsory exit at `T+1min`, using the same delayed
   fill proxy; this instruction is determined by `T-1h`, not
   future `z(T)`. Any entry signal whose scheduled fill is at
   or after `T` is ignored. A normal zero-cross exit scheduled
   later than the compulsory exit is replaced by that known
   compulsory exit. All positions are flat before the next
   block's first possible entry. No fill may be at or after
   2026-01-01 00:00 UTC. The holding interval is half-open at
   minute-bar resolution: include the **entire** minute OHLC whose
   open is the entry fill proxy in drawdown stress, and exclude
   the minute OHLC whose open is the exit fill proxy.

## Fixed account, fees, funding and drawdown

- Initial equity is 10,000 USDT. The fixed formal portfolio has **2x
  total gross entry notional**, with exactly the same pre-entry marked
  positive equity `E_pre` of USD notional on each leg, one long and one
  short. At the simultaneous fill proxy, first apply the declared
  adverse execution-price adjustment to each leg's delayed minute open;
  then set `q_BTC=E_pre/p_exec_BTC` and `q_ETH=E_pre/p_exec_ETH`
  (absolute quantities), and separately deduct each leg's 5 bps
  taker fee from cash. Both legs use `E_pre`; accounting order cannot
  change quantities. The same signals are also run at 1x total gross
  with `E_pre/2` notional per leg as a
  required cost-net **signal quality** comparator. This is fixed
  before results, not chosen from the better backtest. Never run or
  select 3x/5x based on these OOS results. Each entry fixes BTC/ETH
  quantities until exit; no rebalance or pyramiding.
- Each market-order leg pays 5 bps taker fee on its actual executed
  notional at entry and exit, assuming no VIP/BNB discount.
  Each fill also receives an adverse 15 bps spread/slippage and
  10 bps impact/latency adjustment versus the delayed 1m reference
  open: total 30 bps of that leg's notional per side. The two
  adverse price components are **added**, not compounded:
  `p_exec=p_ref*(1+0.0025)` for buy/cover and
  `p_exec=p_ref*(1-0.0025)` for sell/short. A predeclared
  cost stress uses `p_ref*(1+0.0050)` for buy/cover and
  `p_ref*(1-0.0050)` for sell/short, while keeping the
  5 bps fee. At 1x gross, base flat-price round trip is about
  60 bps of account equity across the two legs; at 2x it is
  about 120 bps. Actual spread/impact remains unverified.
- For each funding event whose `calc_time` falls in that leg's
  half-open holding interval `[entry_fill_time,exit_fill_time)`,
  use its archived signed rate and that leg's fixed quantity
  times the most recent **completed** one-minute trade-price close
  before `calc_time` as a notional proxy. Long pays a positive
  rate and short receives it. Funding-time trade close is not
  the exchange's actual mark price; a no-beneficial-credit
  stress (credits set to zero, charges kept) is reported.
- Calculate UTC daily equity including flat days and mark both
  fixed-quantity legs to minute close while open. Drawdown Gate
  takes the worse of minute-end marks and a conservative
  simultaneous adverse 1m mark: low on the long and high on
  the short, even though those extrema may occur at different
  seconds. Any minute marked equity <=0 is a strategy failure.
  A drawdown over 20% fails the Gate even if later recovered.
  Do not send any order. Historical liquidation/maintenance
  margin and $10,000 executable quotes are unverified.
- Marked equity is initial equity plus all realized leg PnL,
  received funding, minus paid funding and fees, plus each open
  leg's `sign*quantity*(current minute reference price - entry
  executed proxy price)`. On exit, realized leg PnL is
  `sign*quantity*(exit executed proxy price - entry executed proxy
  price)` and the corresponding open mark is removed. For a
  market-state bucket, a completed pair trade's net PnL includes
  both legs' realized PnL, both entry/exit fees, and all funding
  during that pair's holding interval.
- For daily, monthly and annual return attribution, use equity at
  each UTC `00:00` boundary immediately **before** any funding event
  stamped `00:00`; it includes marks through the completed preceding
  minute. The interval `[D,D+1d)` owns all funding with timestamp in
  that interval, including `D 00:00`, and ends just before funding at
  `D+1d 00:00`. A day/month/year return is the ratio of its ending
  and starting boundary equities minus one. Funding at midnight is
  never charged to the prior day or counted twice.

## Frozen Gate and stop rule

1. **Data and causal timing:** all 144 monthly BTC/ETH K-line ZIPs
   match their pinned SHA manifest, minute grids, and OHLC audits;
   120 funding ZIPs match their official checksums, have complete
   event coverage and finite rates. Every feature precedes its
   signal and every fill follows the declared lag. A failed
   data check stops the run before partial performance is accepted.
2. **1x signal quality prerequisite:** net daily Sharpe >1.5,
   net daily dollar-PnL profit factor >1.3, positive net
   five-year CAGR, and the same minute-stress max drawdown <=20%.
   If it fails, the
   fixed 2x metrics may be reported as diagnostics but can never
   promote this candidate or be used to claim leverage rescued it.
3. **2x formal portfolio:** net five-year CAGR >=40% (report
   whether it falls in the 40–60% target band), daily Sharpe
   >1.5, minute-stress max drawdown <=20%, and daily profit
   factor >1.3. Use 1,826 UTC days, initial 10,000 equity,
   `CAGR=(E_end/10000)^(365.25/1826)-1`,
   `Sharpe=sqrt(365)*mean(daily_return)/sample_sd(daily_return)`,
   zero risk-free rate; PF is positive **daily dollar PnL** /
   absolute negative daily dollar PnL. Zero-trade/no-loss
   cases give insufficient evidence, not infinite PF.
   Report number of completed pair round trips, each month's
   net return, positive/negative/flat month counts, worst month,
   and longest losing-month streak.
4. **Annual walk-forward and market states:** exactly five full
   calendar-year folds, each with >=1 entered pair trade,
   positive marked net annual return and positive daily Sharpe,
   with the same frozen mechanism and past-only formation
   recomputed every cycle. Cross-year positions allocate
   return from daily marked equity, not the exit year.
   For each trade, classify its market state at the **entry signal
   hour `t`**, using no observation after `t`: BTC's known close
   `P(t)/P(t-90d)-1` (`>=0` bull, `<0` bear), and sample SD of
   the 720 BTC hourly log returns with endpoints
   `t-719h,...,t` versus a median of 365 candidate past UTC
   daily snapshots. Let `D=floor_UTC_day(t)`. Snapshot times are
   `D-365d, D-364d, ..., D-1d` at 00:00 UTC, strictly excluding
   the entry day; each snapshot is the sample SD of 720 hourly
   returns ending at that snapshot. Require at least 300 finite
   snapshots after 2020-history truncation; the current SD
   `>=median` is high, otherwise low. A missing required 90-day
   close, current SD, or 300-snapshot median makes state evidence
   insufficient rather than assigning a bucket. Allocate all that pair's
   final net PnL to its entry bucket. Require at least one
   trade and positive net PnL in all four bull/bear × high/low
   buckets; missing state is insufficient evidence.
5. **Block Monte Carlo and stress:** use exactly 5,000 circular
   21-day block bootstrap paths from the formal 2x series of 1,826
   whole-day **net** records, using one
   `numpy.random.Generator(numpy.random.PCG64(20261108))` stream.
   For each path, independently draw 87 uniform integer block starts
   from `0,...,1825` with that stream; each block takes 21 successive
   days with circular index `(start+j) mod 1826`, concatenate in
   draw order, and truncate after exactly 1,826 days. Generate the
   5,000 paths in sequence without resetting the stream.
   Preserve within-day order of minute-end equity and the simultaneous
   adverse minute mark by storing, for each UTC day relative to its
   starting equity, (a) ending regular-equity factor, (b) maximum
   regular-equity factor including day start, (c) minimum adverse
   mark factor including day start, and (d) maximum within-day
   drawdown of an adverse mark from the prior within-day regular
   equity peak including day start. A bootstrap path rebases each
   sampled day on its new starting equity, updates the global
   regular-equity peak with (b), and calculates that day's exact
   minute-stress drawdown as the worse of (d) and
   `1 - start_equity*(c)/prior_global_regular_peak`.
   The path's maximum is its formal minute-stress drawdown, not a
   daily-close substitute. Verify this compressed replay reproduces
   the original, unresampled minute-stress maximum drawdown before
   accepting Monte Carlo results. Each path compounds 1,826 days
   from 10,000; compute CAGR, daily Sharpe and daily dollar-PnL PF
   from its daily endings. At least 80% of paths must jointly pass
   the four numerical formal metrics above. Report p05/p50/p95,
   undefined counts and joint pass count.
   Doubled non-fee cost and no-beneficial-funding-credit
   stresses must each end above 10,000 USDT at 2x.
6. **Evidence grade:** the internal historical holdout alone
   cannot prove contemporaneous archive publication/version,
   five years of executable BBO/L2, or actual two-leg
   simultaneous fills. These evidence gates remain
   **unpassed** until independent replayable market data
   and prospective small-size paper observation support them.

Run this exact experiment once after committing the final registration
and data hashes. If any check fails, seal the result as rejected or
evidence-insufficient and do not optimize thresholds, pair, years,
formation length, costs or leverage on this OOS. Even a fully
verified future pass would first enter small-position paper
trading; no automatic live order is authorized.
