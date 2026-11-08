# Preregistration: post-funding settlement shock fade

**Experiment ID:** `btc_eth_post_funding_settlement_shock_fade_2026_11_08`.
Commit and push this document **alone** before opening any registered K-line
price, funding-rate value, signal, trade, return or Gate result. One frozen
mechanism and one formal run. A source, code, numerical or evidence failure
seals this exact design; no same-history sign flip, threshold search, holding
period change, coin substitution, year deletion, fee reduction, leverage
rescue or rerun.

## Mechanism, prior work and novelty screen

The [Binance volatility-transmission study](https://www.tandfonline.com/doi/full/10.1080/1350486X.2022.2125885)
documents conspicuous five-minute trading-volume spikes at UTC 00:00, 08:00
and 16:00, suggesting portfolio adjustments around perpetual funding
settlements. The **untested transfer prediction here** is that an unusually
large price displacement in the *first complete five-minute bar after an
actual funding event* is partly temporary inventory/hedging pressure and
reverses over the following hour. The paper documents volume, **not** a
profitable reversal after our costs. The signal uses the settlement clock and
K-line prices, never the funding-rate value; positions open after settlement
and close before the next one, so the experiment does not seek funding carry.

Before registration, the tracked 310-document novelty screen used the
phrases `funding settlement`, `funding payments`, `post-funding`,
`settlement-driven`, `结算时点`, `资金费结算`, `结算后` and `funding timestamp`.
Fourteen documents matched. Full prior registrations for Binance spot/perp
basis, Bybit/alt funding carry, BTC two-hour generic jump reversal and BTC
non-hour quarter-candle drift were reviewed. Basis/carry earn or pay funding
over holding intervals; this rule is **flat at every settlement**. Generic
jump reversal fires on extreme two-hour returns regardless of external
event; this rule only fires after a funding-settlement hedging window and
uses the first five-minute bar. Quarter-candle drift trades minute 15/30/45
unconditionally and deliberately avoids the hour. Settlement-induced
inventory adjustment is a different proposed *cause*, though price response
may correlate with generic reversal and this economic distinction does not
prove statistical independence. No prior result for this exact mechanism was
read or used to set its sign, threshold or holding time.

Evidence grade `preregistered_retrospective_settlement_inventory_fade`;
`independent_executable_oos=false`. Existing archive bytes are current
versions without point-in-time first-publication or bid/ask snapshots. The
2024–2025 holdout is retrospective, not five years of prospective executable
OOS. Even a numerical pass cannot clear the full Gate or authorize an order.

## Frozen assets, sources and sample

- **BTCUSDT and ETHUSDT** Binance USD-M perpetuals, each with a maximum
  absolute notional of 0.5 × current positive account equity at an event.
  Both signals are evaluated independently and simultaneously; no ranking,
  post hoc coin choice or netting. Gross exposure ≤1×, below the user's 5×
  ceiling. No borrowing or cash yield. Start with **10,000 USDT**.
- Official Binance Vision monthly USD-M `5m klines` for both symbols from
  2020-12 through 2025-12. The committed
  `2026-11-07_leadlag_5m_timestamp_manifest.csv` has SHA-256
  `335ea7b79634756ee0cf82428fc32d6c95fa3477470984d00b26ce013a19328a`;
  select exactly 122 rows (2020-12 warm-up plus 60 full evaluation months
  per coin). Official `fundingRate` monthly ZIPs January 2021–December 2025
  use `2026-11-07_leadlag_funding_timestamp_manifest.csv`, SHA-256
  `09492df23bc57c88d620c506b134fe690f55672156fc468e642a02903a6e1389`;
  exactly 120 selected rows. Metadata-only preflight found **5,478** funding
  events per coin, i.e. three for every UTC date in the 1,826-day evaluation
  interval, with zero recorded off-hour events. No rate or price values were
  read in this preflight. Revalidate exact frozen manifest hashes, fresh
  official `.CHECKSUM` text and each ZIP SHA, sole expected CSV member,
  complete UTC five-minute grid, finite consistent OHLC, nonnegative quote
  volume, integer trade counts and unique settlement timestamps. Funding
  events must be exactly at 00/08/16 UTC plus at most 999 ms, with finite
  rates and eight-hour intervals. Any source mismatch or missing event/bar
  seals `DATA_FAILURE`; no interpolation, daily-file patch, alternate feed,
  coin deletion or shortened sample.
- Use the committed bounded HTTPS transport: same exact URL up to **four**
  attempts only for network/HTTP failures, 25 s each with 0.25/0.5/1.0 s
  waits. Pinned hash, format, range or schema disagreement is terminal with
  no retry. Each checksum cap 500 B, price ZIP cap 5 MB, funding ZIP cap
  500 KB. Raw archives remain memory-only; only compact source receipts and
  reports are stored.
- Trade/evaluate all UTC days 2021-01-01 through 2025-12-31, with 2020-12
  solely for warm-up. 2021–2023 is fixed historical transfer/development;
  2024–2025 is the untouched-rule historical holdout. Five annual folds
  2021–2025 have no fitting or parameter updates.

## One exact signal and account rule

For each verified funding event at UTC hour `T`, form the first K-line
`[T,T+5m)` log move `r = log(close/open)`. Before viewing that bar, compute
`m` as the median absolute `log(close/open)` over **all 8,640 completed
five-minute bars in `[T−30d,T)`** for the same coin. Require every bar,
`m>0`, and a nonzero `r`. If `|r| >= 3m`, target the **opposite** direction:
positive first move → short, negative → long. Otherwise remain flat. This
rolling volatility reference is strictly prior to the settlement bar. The
first bar ends at `T+5m`; model five additional minutes of data/decision
latency and enter at the K-line **open at `T+10m`**. Exit at **`T+70m`** open,
for a fixed 60-minute hold. There is at most one open position per coin and
no overlap across events. For the one-bar-latency stress, enter `T+15m` and
exit `T+75m` using the same already-frozen signal. The first 2021-01-01
event has a full preceding 30 days from 2020-12-02.

Each selected coin enters with absolute USD notional 0.5 × account equity
immediately before its `T+10m` entry; both coins at the same event reference
the same pre-entry equity. Quantities round **toward zero to 0.001 coin**;
require ≥10 USDT per fill. Both entry and exit pay **5 bp taker fee plus 5 bp
adverse slippage** on executed notional; long buys/short covers execute
above the reference open, long sells/short entries below. No spread rebate,
VIP fee or perfect queue fill is assumed. The fixed cost stress doubles only
the adverse slippage to 10 bp per side. Each order's reference notional must
be ≤1% of the preceding 30 completed UTC days' **median five-minute quote
volume** for that coin; if capacity fails, seal rather than selectively skip
the event. Record every fill's timestamp, side, reference, execution,
quantity, notional, fee, slip drag and account equity.

Require **no actual funding event** in any open position interval
`[entry,exit)`. The account is flat at the settlement event that triggers
it and closes hours before the next one, so modeled realized funding is
exactly zero; if the archived schedule contradicts this, seal a data/timing
failure rather than suppressing funding. Mark held positions at each five-
minute close, track the adverse intrabar low for longs/high for shorts and
use the conservative high-before-low equity path for maximum drawdown.
Require positive account equity, gross notional ≤1× at entry and a
conservative maintenance proxy: adverse intrabar equity must exceed 5% of
gross notional, otherwise seal an execution/risk failure. Actual Binance
mark-price liquidation, queue priority and historical margin tiers remain
unverified. Idle USDT earns zero. Every UTC-day endpoint is flat; include
zero-trade days in net daily PnL.

## Prespecified Gate, stress and stopping

Full net account and separately the 2024–2025 holdout must each have CAGR
**≥40%** (40–60% preferred), daily Sharpe **>1.5** (`sqrt(365)`, zero
risk-free), conservative maximum drawdown **≤20%**, daily-dollar Profit
Factor **>1.3**; at least **100** distinct settlement-event entries overall.
Each 2021–2025 annual fold must contain an entry, finish above its start,
have positive daily Sharpe and daily-dollar PF>1. Require ≥36/60 strictly
profitable months and ≤3 consecutive nonpositive months; report worst month.

Four bull/bear × high/low-volatility states use only the previous 31
completed BTC daily closes: 30-day log return positive/otherwise and
annualized sample SD ≥80%/below. Each state requires ≥90 days and positive
net-dollar PnL. Exactly **5,000** circular **21-day** block-bootstrap paths,
NumPy PCG64 seed **20261116**, jointly resample complete net daily factors
and normalized intrabar high/low marks; ≥80% must jointly pass the four
numeric targets. Report p05/p50/p95, ruined and undefined paths. Both
double-slippage and one-bar-delay paths must finish above 10,000 USDT.
Compare gross zero-cost account with the net account; no second strategy or
parameter variant is chosen from the result.

Any source, cash, fill, funding-time, capacity, maintenance, numerical,
annual, holdout, monthly, regime, bootstrap, stress or independent-OOS
failure rejects and seals. No real or automatic paper orders. Passing
numbers would only justify a separately registered prospective executable
study, and ultimately small paper positions after the full Gate truly passes.
