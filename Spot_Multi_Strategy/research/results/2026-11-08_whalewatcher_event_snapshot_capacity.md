# WhaleWatcher event snapshot capacity check (research infrastructure only)

At 2026-11-08 14:15 UTC the local WhaleWatcher `event_state/latest.json` was
22,396,486 bytes with 622 top-level records. The prior Quant reader limit was
24 MiB (25,165,824 bytes), leaving under 2.8 MiB of headroom. A larger source
snapshot would have been marked unavailable even if its event metadata remained
valid. This is a collection continuity risk, not a strategy result.

The isolated Quant reader now accepts at most 64 MiB and 5,000 source rows.
It still parses and validates the complete snapshot, enforces a ten-minute
source age, copies only redacted event provenance into the append-only journal,
and leaves all event features ineligible for trading. It does not duplicate the
large source snapshot on disk. A direct read of the current snapshot returned
`fresh`, 170 eligible source-event metadata rows, and
`execution_eligible=false`. Seventeen focused synthetic tests passed.
The deployed local reader matched the repository SHA-256. A manual collector
run exited 0 and recorded `events/ALL: captured; items=170`; the separate
macro schedule was fresh with nine items and no changes.

The separate `external_intelligence_state/latest.json` is currently stale:
its last `generated_at_ms` was 2026-11-08 11:10:40 UTC and its source health
reported `unconfigured` with zero configured/ready sources. The 14:05 UTC
Quant source poll accordingly marked both BTC and ETH claim feeds
`unavailable`; no directional claim or backfilled OOS is inferred. The event
capacity change does not repair this independent source gap.

No economic hypothesis, returns, orders, WhaleWatcher application source, or
protected mature strategy files were changed. The prospective journal's
five-year continuous-source gate remains false.
