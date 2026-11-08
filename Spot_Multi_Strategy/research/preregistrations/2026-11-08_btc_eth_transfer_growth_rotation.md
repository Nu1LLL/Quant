# Preregistration: BTC/ETH native-transfer growth rotation

**Experiment:** `btc_eth_native_transfer_growth_rotation_2026_11_08`.
Commit and push this document **alone** before parsing any transfer counts,
BTC/ETH prices, signal, trade, return or Gate result. This is one economic
hypothesis and one formal run. Any failure permanently seals this exact
direction, window, assets, lag, cost, sample and Gate. No same-history sign
flip, asset expansion, threshold search, year deletion, leverage rescue,
alternate provider, cached data or second formal run.

## Mechanism and evidence grade

The fixed prediction is that **faster growth in native-coin transfer counts**
indicates relatively stronger settlement demand and should favor that chain's
coin in the following week. Relative growth is compared within each chain,
not raw BTC versus ETH count levels. [Coin Metrics defines `TxTfrCnt`](https://docs.coinmetrics.io/network-data/network-data-overview/transactions/transfers)
as daily positive-value transfers; it differs from transaction count, and
ETH's definition includes internal transfers but excludes failed transactions
and self-sends. Transfers can reflect custodial reshuffling or automation,
not final-user demand. BTC's UTXO and ETH's account structures are not
identical. This chain-settlement-intensity mechanism differs from previously
tested ETH fee burden, BTC exchange reserves, DEX trading volume, stablecoin
issuance, futures positioning and price momentum. Correlation with other
on-chain usage factors is possible; economic distinctness does not prove
statistical independence.

Evidence grade: `preregistered_retrospective_current_vintage_transfer_rotation`;
`independent_executable_oos=false`. Coin Metrics Community serves current
recomputed history, not archived first-publication vintages, and historical
daily availability is proxied by a two-day lag. Binance OHLCV does not prove
historical executable depth. The fixed 2024–2025 holdout is retrospective,
not genuinely independent prospective OOS. Thus even a numerical pass cannot
clear the full Gate or authorize any order.

## Frozen sources, account and sample

- Coin Metrics Community exact GET:
  `https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?assets=btc,eth&metrics=TxTfrCnt&frequency=1d&start_time=2020-11-01&end_time=2025-12-31&page_size=10000&paging_from=start`.
  Metadata-only, value-blind preflight via `curl` found raw response SHA-256
  `6059f1c775369dc4997498f48e581a158ba6df43c1e4bf4e49592b91c13bfe15`,
  287,971 bytes, one page, exactly **1,887** UTC-midnight dates per asset
  (3,774 rows), from 2020-11-01 through 2025-12-31, fields exactly
  `asset,time,TxTfrCnt`, no missing markers. At formal read require the exact
  full raw SHA before numeric parsing, unique complete dates, positive
  integer counts and no next page. A revision, missing day, malformed count,
  inaccessible service or changed hash seals `DATA_FAILURE`. No raw API
  response is saved. [Community access and terms](https://docs.coinmetrics.io/packages/coin-metrics-community-data).
- BTCUSDT Binance Vision spot `1d` monthly ZIPs December 2020–January 2026
  use the existing committed manifest SHA-256
  `581c681e6d14065242ae8ff464b313fca02038fc16a6e158f37a203ae3cd4a5f`.
  ETHUSDT spot November 2020–January 2026 use the existing committed
  `2026-11-08_eth_defi_tvl_spot_manifest.csv`, SHA-256
  `80206c7d2fd9b5965d8beb37d8dcb73516d7f806dfec3f2195ff56f6b716f27e`.
  For each month require a fresh official `.CHECKSUM`, frozen ZIP SHA, sole
  exact CSV member, complete UTC daily OHLC/quote-volume grid and correct
  millisecond/microsecond timestamp convention. January 2026 supplies only
  the first 2026-01-01 opening prices. Raw ZIPs are not retained.
- Begin with **10,000 USDT cash** on 2021-01-01 and liquidate at the
  2026-01-01 UTC opening. The 1,826 daily factors cover 60 full months.
  2021–2023 are the fixed retrospective development/transfer interval;
  2024–2025 are the fixed retrospective holdout. Fixed-rule annual folds
  2021–2025 have no fitting or adaptive parameters.

## Exactly one signal and execution rule

At each Monday 00:00 UTC decision `M`, the latest permitted transfer-count
date is **Friday `M−3`**, allowing the entire Friday and weekend for data
publication. For each coin `a`, compute recent 28-day sum over
`M−30..M−3`, previous 28-day sum over `M−58..M−31`, and ratio
`G_a = recent / previous`. All 56 daily values must exist and be positive.
If `G_BTC > G_ETH`, target **100% BTC spot**; if `G_ETH > G_BTC`, target
**100% ETH spot**; exact equality targets **USDT cash**. Before the first
Monday hold cash. No price, trend, volatility, fee, AI output or future
transfer value enters this decision. Execute any changed target at Monday's
BTC/ETH opening; for a switch sell the old coin before buying the new one.
Hold an unchanged target without turnover. A fixed one-day-latency stress
executes the same frozen Monday decision at Tuesday's open; terminal sale
still occurs at 2026-01-01 open.

Long spot/cash only, no short, margin, derivatives, funding, borrowing or
cash yield; gross BTC+ETH exposure ≤1x equity (under the user's 5x ceiling).
Each executed sell/buy pays **10 bp taker fee** plus **5 bp adverse slippage**.
Round quantities down to `0.00001` native coin, keep cash dust, require ≥10
USDT per fill, and require each buy notional ≤1% of the preceding 30
completed UTC days' median quote volume for that specific pair. Any failed
source, capacity, cash, leverage, positivity or reconciliation check seals
rather than silently skipping. Mark daily equity at both spot closes and
compute conservative high-before-low intraday drawdown. Record each leg's
reference, execution price, quantity, notional, fee and slippage. Compare
the same signal with zero costs, double slippage (10 bp/fill), extra one-day
latency, and a same-cost static 50/50 BTC/ETH buy-and-hold basket. Both
stress endpoints must exceed 10,000 USDT.

## Full Gate and stopping

Full net account requires CAGR **≥40%** (also report preferred 40–60%),
daily Sharpe **>1.5** with `sqrt(365)` and zero risk-free, conservative
max drawdown **≤20%**, daily-dollar Profit Factor **>1.3**, and ≥20
position entries. Each annual fold 2021–2025 must contain a fill, finish
above its start and have positive daily Sharpe. The 2024–2025 holdout must
separately pass all four numerical thresholds. Require ≥36/60 strictly
profitable months and no more than three consecutive nonpositive months;
report the worst month.

For validation only, classify days from prior 31 completed BTC closes:
positive 30-day log return = bull, otherwise bear; annualized sample SD
≥80% = high volatility, otherwise low. Require ≥90 days and positive
net-dollar PnL in each bull/bear × high/low state. Exactly **5,000** circular
**21-day** block-bootstrap paths, NumPy PCG64 seed **20261114**, jointly
resample net daily equity factors and normalized high/low marks. ≥80% must
jointly pass all four numerical limits; report p05/p50/p95, ruined and
undefined paths. Any numerical, annual, holdout, monthly, regime, bootstrap,
stress, cost, execution, source or independent-OOS failure rejects and seals.
No real orders or automatic live deployment.
