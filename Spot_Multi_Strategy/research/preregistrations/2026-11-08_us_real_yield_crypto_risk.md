# Preregistration: US real-yield easing and crypto risk exposure

**Experiment:** `us_10y_real_yield_easing_btc_eth_2026_11_08`.
Commit and push this document **alone** before reading any DFII10 number, BTC/ETH
price, signal, trade, return, or Gate result. One rule, one formal run. If the
data or any Gate fails, seal this exact direction, lookback, basket, timing,
cost, sample and Gate permanently. No same-history reversal, threshold search,
asset substitution, year deletion, leverage rescue, source replacement or
second formal run.

## Economic mechanism and evidence grade

When the US 10-year inflation-indexed Treasury yield falls over roughly one
trading month, the real discount rate and cost of carrying speculative risk
fall. The untested prediction is that a **decline** will favor an equal BTC/ETH
spot basket over USDT cash for the next week. This is a real-rate discounting
mechanism, distinct from this program's Fed balance-sheet, Treasury General
Account, CPI surprise, unemployment claims, VIX term structure, on-chain use,
exchange reserves, futures positioning and crypto price-only mechanisms. It
may still correlate with broad risk appetite; independence is economic, not a
claim of zero statistical correlation. The [Federal Reserve H.15 / FRED DFII10
series](https://fred.stlouisfed.org/series/DFII10) identifies a daily 10-year
inflation-indexed constant-maturity Treasury yield. [IMF research](https://www.imf.org/en/-/media/files/publications/wp/2023/english/wpiea2023163-print-pdf.pdf)
links US monetary contraction to the common crypto factor; this is motivation,
not evidence that this trading rule works.

Evidence grade `preregistered_retrospective_current_vintage_real_rate`;
`independent_executable_oos=false`. FRED supplies today's vintage rather than
an archived first-publication sequence, and Binance daily bars do not establish
historical executable depth. A two-day weekend delay reduces but cannot prove
away publication/revision lookahead. The 2024–2025 holdout is a fixed,
untouched retrospective test, **not** five years of prospective independent OOS.
Even a numerical pass cannot clear the full Gate or authorize an order.

## Frozen sources, sample and transport

- Exact FRED CSV GET: `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10&cosd=2020-11-01&coed=2025-12-31`.
  The value-blind preflight recorded **21,732 bytes**, raw SHA-256
  `35f372e026b173dfbf847cc2384f3d553ba82c049a0ba3ffc9c52ab5d0b7f9ef`,
  header `observation_date,DFII10`, 1,348 dated rows from 2020-11-02 through
  2025-12-31 and 58 dot/blank holiday observations. The preflight did not
  parse or display yield values. At formal read require the raw SHA, exact
  schema, unique increasing dates, finite numeric values or dot/blank for
  holidays; no interpolation or silent forward fill. An extra/missing row,
  revised byte, inaccessible source or abnormal date grid seals DATA_FAILURE.
- BTCUSDT Binance Vision spot `1d` December 2020–January 2026 uses committed
  manifest `2026-11-08_btc_exchange_supply_spot_manifest.csv`, SHA-256
  `581c681e6d14065242ae8ff464b313fca02038fc16a6e158f37a203ae3cd4a5f`.
  ETHUSDT spot November 2020–January 2026 uses committed manifest
  `2026-11-08_eth_defi_tvl_spot_manifest.csv`, SHA-256
  `80206c7d2fd9b5965d8beb37d8dcb73516d7f806dfec3f2195ff56f6b716f27e`.
  Verify fresh official `.CHECKSUM` and frozen ZIP SHA for every month, sole
  exact CSV member, full UTC daily OHLC/quote-volume grid and timestamp units.
  January 2026 supplies only its first day's opening prices. Raw sources are
  memory-only, not archived to disk.
- Every exact-URL GET (FRED CSV, official checksum and ZIP) uses up to **four**
  attempts on transport/HTTP errors, 25-second per-attempt limit, 0.25/0.5/1.0
  second waits; HTTPS and host allowlist. Once bytes arrive, a pinned SHA or
  content/schema disagreement is terminal with **no retry**. FRED body cap
  50 KB, checksum cap 500 B, ZIP cap 4 MB. Any exhausted transport error seals
  DATA_FAILURE. This policy was frozen before the experiment because a prior
  unrelated run hit a transient TLS error; it does not reopen that run.
- Initial account **10,000 USDT** at 2021-01-01 UTC; daily factors 2021-01-01
  through 2025-12-31 (1,826 days), forced liquidation at 2026-01-01 open.
  2021–2023 fixed development/transfer window, 2024–2025 fixed historical
  holdout. Fixed-rule annual folds 2021–2025 have no fitting.

## Frozen decision and fills

At every Monday **00:00 UTC**, only DFII10 observations dated through the
preceding **Friday** are eligible. Ignore official dot/blank non-observations;
take the most recent valid Friday-or-earlier yield `Y_new` and the yield
**20 valid observations before it** `Y_old`. Require 21 valid observations
within the last 45 calendar days; otherwise seal DATA_FAILURE. If
`Y_new < Y_old`, target invested: buy BTC and ETH spot with equal half of the
then-current account equity allocated to each. Otherwise target USDT cash.
Equality targets cash. The invested state does not rebalance BTC/ETH between
entry and exit; portfolio weights drift. A changed state executes at Monday's
BTC/ETH opening, selling both before any purchase. Before the first Monday,
hold cash. For the latency stress, execute the same already-frozen decision
at Tuesday's opening. No price, crypto feature, AI output or later yield enters
the decision.

Long spot/cash only, gross leverage ≤1x; no short, borrowing, futures,
funding or cash yield. Each buy/sell leg pays **10 bp taker fee** and **5 bp
adverse slippage**. Quantities round down to 0.00001 native coin, residual
cash stays in account, ≥10 USDT per fill. Each buy notional must be ≤1% of
that pair's preceding 30 completed UTC days' median quote volume. A failed
source, grid, capacity, cash, leverage, lot, reconciliation or valuation check
seals rather than dropping an asset/day. Record each fill's reference/open,
execution price, quantity, notional, fee and slippage; mark daily equity at
both closes and use high-before-low conservative intraday drawdown.

Prespecified comparisons: same signals at zero fees/slippage; same signals
with 10 bp slippage per fill; same signals delayed by one day; same-cost static
50/50 BTC/ETH buy and hold from 2021-01-01. Both stressed versions must finish
above 10,000 USDT.

## Full Gate and stopping

Full net account: CAGR **≥40%** (report preferred 40–60%), daily Sharpe
**>1.5** using `sqrt(365)` and zero risk-free, conservative maximum drawdown
**≤20%**, daily-dollar Profit Factor **>1.3**, at least **20 entries**.
Each annual fold 2021–2025 must have a fill, finish above its start and have
positive daily Sharpe. The 2024–2025 holdout independently must pass all four
numeric thresholds. Require ≥36 of 60 strictly profitable months and no more
than three consecutive nonpositive months; report worst month. Regimes, using
only the previous 31 completed BTC closes, are bull/bear by positive/otherwise
30-day log return and high/low volatility by ≥80%/below annualized sample SD;
each of four cells needs ≥90 days and positive net-dollar PnL. Exactly 5,000
circular 21-day block bootstrap paths with NumPy PCG64 seed **20261115** jointly
resample net daily factors and normalized high/low marks; ≥80% must jointly
pass the four numeric thresholds. Report p05/p50/p95, ruined and undefined
paths. Any numerical, annual, holdout, monthly, regime, Monte Carlo, cost,
stress, source, execution or independent-OOS failure rejects and seals. No
real or automatic paper order.
