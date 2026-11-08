# WhaleWatcher prospective capture: continuity incident

**Status:** data-integrity failure for the current uninterrupted prospective window; no trading result.

The append-only source-poll ledger has a **12.893-hour gap** from 2026-11-07 18:11:49.879 UTC to 2026-11-08 07:05:24.969 UTC. The independent macro-schedule ledger has the same gap (18:11:49.881 to 07:05:24.979 UTC). Both exceed the preregistered two-hour maximum. The Mac power log records clamshell sleep at 2026-11-08 02:49 China time and user wake at 14:53; this explains the missed scheduled local captures. The resumed poll succeeded, so the observed problem is collection continuity rather than a source snapshot being marked unavailable.

`poll_coverage` already rejects this gap; historical backfill must not be labeled as contemporaneously observed. Any five-year uninterrupted OOS clock based on these ledgers must begin no earlier than the resumed poll, and later gaps reset it again. Neither journal is currently eligible for an independent OOS claim or trading signal. An always-on collector would be needed for reliable future hourly capture; changing the laptop's sleep settings or replaying missed polls cannot repair this evidence gap.

This incident audit reads only poll timestamps and local power-state logs. It does not modify the WhaleWatcher application, its source snapshots, the append-only journals, or any protected strategy or sealed experiment.
