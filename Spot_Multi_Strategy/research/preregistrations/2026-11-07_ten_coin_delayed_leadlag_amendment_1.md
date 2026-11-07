# Boundary amendment 1: volatility snapshot availability

This amendment to `2026-11-07_ten_coin_delayed_leadlag.md` is made and
committed **before any price or funding-rate value, fitted model, trade or
return from this candidate is read**. The very first available 5-minute bar
begins 2020-12-01 00:00 UTC, so a 288-return volatility estimate at the
2020-12-02 00:00 boundary would require an unavailable 2020-11-30 close.

For the market-state test only, replace "30 prior UTC 00:00 daily SD
snapshots" with **"30 prior UTC 00:05 daily SD snapshots"**. At each 00:05
snapshot, use the 288 completed five-minute equal-weight log close-to-close
returns through the bar ending 00:05. The first valid snapshot is
2020-12-02 00:05 UTC. At a formal trade decision, exclude the current UTC
calendar day's snapshot even if it has occurred, and compare the current
288-return SD with the median of the preceding 30 *calendar days'* 00:05
snapshots. For 2021-01-01 decisions, those are exactly 2020-12-02 through
2020-12-31. Any missing or nonfinite snapshot fails the four-state Gate.

The 30-day trend, 30-day monthly model training window, 5-minute decision
features, one-bar entry delay, 10-minute holding period, trade threshold,
costs, scenario definitions, five-year sample, seed and every Gate threshold
are unchanged. This correction does not inspect a market result or add a
candidate asset or outcome filter.
