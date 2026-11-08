# WhaleWatcher prospective-source readiness guard

Read-only inspection found that Quant's WhaleWatcher event snapshot was fresh,
but BTC and ETH directional-claim polls became `unavailable` after 2026-11-08
11:05 UTC. The last checked poll at 15:05 UTC reported `events=fresh`,
`BTC=unavailable`, `ETH=unavailable`. The external-intelligence snapshot was
about four hours old. Its own source-health metadata said
`status=unconfigured`, `configured_source_count=0`, `ready_source_count=0`;
the only listed feed, `embercn`, was `waiting_for_authorized_export`.

Prior to this change, a newly refreshed **empty** external-intelligence
snapshot could have been imported as `fresh_no_claims` even with zero
configured sources. That would mistake lack of a data feed for a genuine
observation of no qualifying claims. The new fail-closed adapter requires at
least one configured source, all configured sources ready, mutually consistent
source-health and readiness summaries, and a fresh health timestamp before
it can label either nonempty or empty claims as fresh. An unconfigured,
degraded, malformed, or stale source remains `unavailable`. No historical
journal row is edited or retroactively promoted.

The new read-only `whalewatcher_readiness.py` checks both append-only Quant
poll journals. It finds the most recent continuous source interval, restarts
at a gap over two hours or an unavailable poll, and requires a full **five
calendar years**, including leap days, before reporting source continuity.
It never equates source continuity with an independent trading OOS result or
order eligibility. At this audit the event/claim stream has no current valid
interval, the macro schedule has about eight hours of continuous polls, and
their joint five-year source check is false. Laptop sleep previously caused
a 12.893-hour gap; the new guard does not stitch across it.

Focused synthetic tests pass for unconfigured/degraded sources, a real five
calendar-year synthetic stream, stale polls, unavailable polls, gap resets,
redaction and as-of behavior. No WhaleWatcher application code, market
returns, sealed strategy, live order or raw article data was changed. A
configured, authorized external feed and an always-on collector remain
necessary before the claims stream can begin a defensible five-year
prospective clock.
