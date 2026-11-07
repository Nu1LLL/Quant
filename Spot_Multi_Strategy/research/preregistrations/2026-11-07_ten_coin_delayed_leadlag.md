# Pre-registration: ten-coin delayed cross-return ridge model

**Experiment ID:** `ten_coin_delayed_leadlag_2026_11_07`. This document is the
sole economic/strategy hypothesis for the next formal run. Commit and push it
**alone before any price, model coefficient, predicted return, trade or PnL
from this experiment is read**. The price- and rate-blind archive audits and
their checksum manifests were committed earlier. No signal or economic result
was used to choose a symbol, sign, hyperparameter, trade interval or cost.

## Economic source and prior uncertainty

[Guo, Sang, Tu and Wang (2021)](https://ink.library.smu.edu.sg/lkcsb_research/6901/)
report minute-scale return predictability across cryptocurrencies, consistent
with information diffusing between coins. A later
[published version](https://www.sciencedirect.com/science/article/pii/S0165188924000551)
reports only 0.34 basis points per 10-minute rebalancing period after its own
4 bp futures-cost assumption. [Jia, Wu, Yan and Liu (2023)](https://www.sciencedirect.com/science/article/pii/S0927539823000956)
instead report a negative large-to-small lead–lag relation. These studies
support testing **cross-asset transfer** but do not justify fixing the sign
or claiming a stable tradable premium. Our new test asks whether a causally
fitted monthly linear model on *other coins' completed five-minute returns*
can identify rare ten-minute **long/short spreads above our larger execution
cost**. This is economically distinct from the prior price-jump reversal,
funding-spread carry and BTC/ETH distance-pair hypotheses. The paper results
reduce the prior probability of passing the cost Gate; they are not our
backtest results.

## Assets, source, period, evidence grade

- Fixed lexical universe: Binance USD-M perpetual `ADAUSDT`, `BCHUSDT`,
  `BNBUSDT`, `BTCUSDT`, `DOTUSDT`, `ETHUSDT`, `LINKUSDT`, `THETAUSDT`,
  `XMRUSDT`, `XTZUSDT`. This is the complete data-eligible subset of the
  non-stable, non-wrapped [2020-12-30 CoinMarketCap top 20](https://coinmarketcap.com/historical/20201230/)
  described in `research/results/2026-11-07_cross_crypto_leadlag_data_preflight.md`.
  Seven others failed timestamp-only boundary probes; no return was inspected.
  No coin may be added, dropped, reweighted by its realized performance or
  replaced after this registration.
- Official Binance Vision monthly USD-M `5m` K-line ZIPs from **2020-12
  through 2025-12**, 610 symbol-months, are pinned in
  `2026-11-07_leadlag_5m_timestamp_manifest.csv`. Official monthly USD-M
  `fundingRate` ZIPs from **2021-01 through 2025-12**, 600 symbol-months, are
  pinned in `2026-11-07_leadlag_funding_timestamp_manifest.csv`. Every archive
  must match both its committed SHA-256 and a newly fetched official checksum,
  contain one expected CSV member, and pass the complete 5-minute grid or
  full funding timestamp/event-count audit **before its economic fields are
  exposed**. No missing bar/settlement, interpolation, synthetic quote,
  deleted year or substituted symbol is permitted. Do not persist raw ZIP/CSV.
- Warm-up/model history begins 2020-12-01 00:00 UTC. The single formal
  evaluation interval is **[2021-01-01 00:00, 2026-01-01 00:00) UTC**, 1,826
  complete calendar days and 60 complete months; the fixed calendar-year
  walk-forward folds are 2021, 2022, 2023, 2024 and 2025. All five years,
  including flat days and cost losses, count.
- Published research overlaps the historical period, historical executable
  bid/ask/size and point-in-time venue revisions are absent, and monthly
  refitting consumes prior formal history. Therefore the evidence grade is
  **internal rolling historical holdout**, not five years of truly
  independent prospective executable OOS. The independent-data component
  of the complete Gate is fixed **false** in this run even if every numerical
  metric passes. No live trading follows directly from this archive.

## One frozen rolling model

Number the ten synchronous 5-minute bars `j` by UTC opening time. Let
`C[j,k]` and `O[j,k]` be the archived close and open for coin `k`. Define a
feature vector at decision boundary `d = open_time[j]+5m` by
`x[j,k] = ln(C[j,k]/C[j-1,k])`. It uses the newly completed bar and the prior
completed bar. Define a historical training label for target coin `i` as
`y[j,i] = ln(O[j+4,i]/O[j+2,i])`: the 10-minute trade return after **one full
5-minute bar of execution delay**. No current or future target return enters
`x[j]`.

At `T = 00:00 UTC` on each first calendar day from January 2021 through
December 2025, fit **ten separate ridge regressions** using only training
decisions `d` in `[T−30 days, T−20 minutes]`, complete feature/label rows,
and labels whose hypothetical exit is strictly before `T`. The last five
minutes are embargoed. There must be at least **8,000** training rows per
fit; otherwise the whole experiment fails. For each target `i`, the nine
features are the other coins' `x[j,k]` only; the target coin's own past
return, funding, volume, future price and event/news fields are excluded.
Calculate each feature's training mean and population SD; zero/nonfinite SD
or nonfinite data fails. Standardize both training and live feature values
with those fixed training statistics. Center the target label by its
training mean. Use **one fixed ridge penalty λ = 100** on the nine
standardized slopes, with an unpenalized label-mean intercept:
`beta = (ZᵀZ + 100 I)⁻¹ Zᵀ (y−mean(y))`. Predict
`p_i = mean(y_i) + z_i·beta_i`. Use the same ten fitted models for all
decisions in that calendar month. No cross-validation, coefficient sign
constraint, hyperparameter search, refit after loss, or model selection.

## Signal, holdings, funding and costs

- At every UTC boundary with minute **00, 15 or 30** in the formal interval,
  after the preceding completed bar is known, predict all ten future
  returns. This creates exactly **72 decision opportunities per full day**.
  Sort `(prediction, symbol)` ascending; candidate long is the highest,
  short the lowest. Enter only if `p_long − p_short > 0.004` (strictly above
  a 40 bp predicted coin spread), the projected 1x 20 bp account-level
  roundtrip cost break-even. Ties are resolved lexically. Otherwise remain
  USDT cash. No second signal, volume gate, stop, take-profit or intraday
  model change.
- Enter both USD-M legs at the **open five minutes after the decision**
  (`d+5m`), exit both at the **open 15 minutes after the decision**
  (`d+15m`), so holding time is exactly ten minutes. Decisions at minute 45
  are intentionally absent. All positions are flat across each top of the
  hour. The official Binance funding schedule is on UTC hour boundaries,
  with possible 1/4/8-hour frequency; the pinned rate-blind audit found zero
  off-hour settlement timestamps. No funding credit or debit is assigned to
  a position that is flat at settlement. If formal archive validation finds
  any off-hour event or held overlap, the data Gate fails instead of silently
  setting funding to zero. Binance warns that settlement processing may take
  up to about a minute; entry is no earlier than five minutes after an hour
  and the last exit is fifteen minutes before the next hour.
- Initial equity **10,000 USDT**. Freeze two simultaneous scenarios:
  **1x gross** for signal quality and **2x gross** for the formal target,
  each 50% of gross notional long and 50% short, quantities fixed during
  ten-minute holding. No borrowing yield or collateral interest is credited.
  Leverage never exceeds the user's **5x hard ceiling**.
- On each leg and each entry/exit pay **5 bp taker fee plus 5 bp adverse
  execution adjustment**: buy at `O×1.0005`, sell at `O×0.9995`, with the
  separate 5 bp fee charged on executed notional. Approximate full-roundtrip
  drag is 20 bp of 1x account equity or 40 bp at 2x. These are explicit
  assumed costs, **not** observed spread/size or guaranteed fills. Recompute
  fees on exit notional. Two fixed 2x stresses retain the exact signal
  schedule: (a) 10 bp adverse execution on each fill rather than 5 bp,
  same taker fee; (b) enter one additional five-minute bar later (`d+10m`)
  but exit at original `d+15m`, paying baseline costs. Both must end above
  10,000 USDT. No delayed fill may see a new signal or change the chosen pair.
- Mark at each completed held five-minute close. For conservative pressure
  drawdown, pair the long leg's low and short leg's high in the **same bar**,
  including every held bar and excluding bars after the scheduled exit.
  Update the global regular-equity peak with regular marks before comparing
  the adverse mark. Account for all entries, exits, fees and pressure at exact
  UTC day boundaries; no position spans a day boundary. A nonpositive regular
  or adverse equity in **any** predeclared scenario invalidates the whole
  run and is sealed without partial performance promotion.

## Frozen Gate and stopping

Compute 1,826 consecutive UTC daily and 60 monthly account returns. CAGR is
`(E_end/10000)^(365.25/1826)−1`. Daily Sharpe uses sample SD, `sqrt(365)`,
zero risk-free rate. Daily **dollar-PnL** Profit Factor is positive-day PnL
sum divided by absolute negative-day PnL sum; undefined Sharpe/PF fails.
Report 1x/2x final equity, CAGR, Sharpe, five-minute pressure max drawdown,
PF, gross price PnL, fees, funding (expected zero but validated), trade count,
all 60 monthly returns, positive/negative/flat month counts, worst month,
longest losing-month streak, and each 2021–2025 fold.

1. **1x prerequisite:** CAGR >0, Sharpe >1.5, pressure max drawdown ≤20%,
   daily dollar PF >1.3. Never increase leverage to rescue low Sharpe.
2. **2x numerical target:** CAGR ≥40% (also report whether within 40–60%),
   Sharpe >1.5, pressure max drawdown ≤20%, PF >1.3. Each of the five
   annual walk-forward folds must have ≥1 entered pair, positive net return
   and positive daily Sharpe under the frozen monthly retraining rule.
3. At entry decision classify market state from the equal-weight ten-coin
   30-day log-close return (`≥0` bull) and the sample SD of the latest
   **288** five-minute equal-weight returns (`≥` median of the **30 prior**
   UTC 00:00 daily SD snapshots means high volatility). December 2020 is
   warm-up. `unknown` fails. Allocate a pair's full net PnL to its entry
   `bull_high`, `bull_low`, `bear_high` or `bear_low` bucket. Every bucket
   must have ≥1 trade and positive summed net PnL.
4. On the fixed 2x daily record use **5,000** PCG64 paths seeded
   `20261111`, each with **87** uniform circular 21-day blocks truncated
   to 1,826 days. Store daily end factor `a`, maximum regular factor `b`,
   minimum adverse factor `c`, and worst within-day adverse drawdown `d`.
   Replay must exactly reproduce original five-minute pressure max drawdown
   before resampling. At least **80%** of paths must jointly pass all four
   2x numerical thresholds. Report p05/p50/p95 and undefined counts.
5. Both fixed execution stresses must finish above start. The complete Gate
   additionally requires historical executable BBO/size, actual fill and
   latency evidence, and five years of truly independent prospective OOS.
   Those are absent, so **full Gate is false** regardless of numerical
   success. A numerical pass could only advance to small-size, read-only
   paper simulation of a 10,000 USDT account, never automatic live orders.

Any missing/changed archive, model instability, scenario insolvency or failed
Gate is reported and sealed. Do not retune λ, return horizon, threshold,
costs, ten coins, years, model sign, leverage, folds or Monte Carlo seed on
this same historical sample. No later combination with a winning subset of
these outcomes can be called independent OOS.
