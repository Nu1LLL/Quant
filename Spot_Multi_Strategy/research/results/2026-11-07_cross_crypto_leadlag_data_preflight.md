# Cross-cryptocurrency lead-lag: data-only preflight

This is **not a strategy pre-registration or return result**. No price, volume,
funding rate, trading signal, fitted coefficient, return, or PnL was read.
The price-audit script `research/leadlag_5m_preflight.py` inspected only the
official ZIP checksum, CSV member, 5-minute open timestamp and close
timestamp; `research/leadlag_funding_preflight.py` inspected funding timestamps
and intervals without reading rate values. No raw ZIP or CSV was retained.

The economic source for a possible later test is [Guo, Sang, Tu and Wang
(2021)](https://ink.library.smu.edu.sg/lkcsb_research/6901/), who report
short-horizon cross-cryptocurrency return predictability. A later [published
study](https://www.sciencedirect.com/science/article/pii/S0165188924000551)
reports a much smaller post-cost futures return (0.34 bp per 10-minute
rebalance under its own 4 bp transaction-cost assumption). The 2023
[seesaw study](https://www.sciencedirect.com/science/article/pii/S0927539823000956)
even reports a negative large-to-small lead-lag sign. These opposing findings
make a fixed sign inappropriate and put execution costs at the center of any
future test. The published studies overlap the historical sample, so any
2021–2025 analysis must remain an internal historical holdout, never a claim
of five years of prospective independent OOS.

The asset screen began with the non-stable, non-wrapped cryptocurrencies in
the [2020-12-30 CoinMarketCap top 20 snapshot](https://coinmarketcap.com/historical/20201230/),
using their Binance USD-M `USDT` perpetual symbols. Three-date timestamp-only
probes (`2020-12`, `2022-02`, `2025-12`) rejected XRP, LTC, XLM, TRX and XEM:
their February 2022 monthly `5m` ZIPs have 7,200 rather than 8,064 bars, a
three-day terminal gap. BSV's December 2020 and February 2022 archives and
EOS's December 2025 archive were unavailable. They were excluded before any
economic value was inspected. No symbol was replaced after a return result.

The remaining fixed ten-symbol data-only screen was BTC, ETH, BCH, DOT, ADA,
BNB, LINK, XMR, THETA and XTZ. All **61 months** from 2020-12 through 2025-12
passed the full 5-minute timestamp grid, named-member and official-checksum
audit: **610 of 610 symbol-month ZIPs**. Their SHA-256 digests and bar counts
are in `research/preregistrations/2026-11-07_leadlag_5m_timestamp_manifest.csv`
(56,343 bytes). The full ZIP contents are not stored locally.

A separate **rate-blind** audit checked the same ten USD-M contracts' funding
history ZIPs for every month in 2021–2025. All **600 of 600** archive members
and official hashes passed. Across them, settlement counts vary from 84 to
186 per month, consistent with changing contract intervals, but **zero
settlement timestamps occur beyond the first second of a UTC hour**. The
600 SHA-256 pins, event counts and off-hour counts are in
`research/preregistrations/2026-11-07_leadlag_funding_timestamp_manifest.csv`
(60,844 bytes). The rate column was not read. This supports a later rule that
is flat around each top-of-hour funding event; it does not verify execution
timing or prove that an exchange settlement cannot overrun its published
schedule.

This check proves continuity and current archive integrity only. It does not
prove historical quote availability, funding settlement, fillability, absence
of archive revision before the checksum snapshot, or profitable alpha. Before
any price or signal outcome is read, a separate formal pre-registration must
freeze model, train/test boundaries, costs, funding handling, execution delay,
holding period, leverage, pressure marks, Monte Carlo and stopping rule. If
the expected short-horizon spread cannot exceed conservative roundtrip costs,
the hypothesis should be rejected without promoting a trading candidate.
