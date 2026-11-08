# Prospective Hyperliquid source coverage audit

Run `PYTHONPATH=Spot_Multi_Strategy python3 -m research.crypto_blackbox.hyperliquid_flow_coverage` from the repository root. The audit opens the separate local source journal read-only; it does not create or update it. JSON timestamps are UTC epoch milliseconds. `--journal` and `--as-of-ms` support an explicit file and reproducible cutoff.

The fixed grid begins **2026-11-09 00:00 UTC** and includes both BTC and ETH for each whole UTC hour after the collector's five-minute delay. An absent journal, missed hour, missing coin, `late`, `unavailable`, `empty`, or `invalid` observation is reported as a bad hour. No later capture fills an earlier gap. The first five-calendar-year source window ends **2031-11-09 00:00 UTC**; `first_five_year_source_coverage_complete` can only be true after every hour in that window has two timely rows.

The status `awaiting_first_complete_hour` before 2026-11-09 01:05 UTC is expected. After that time, `journal_missing` is a collection failure. The digest summarizes the rows used by one audit so a later report can be compared against the same cutoff.

This checks **record continuity and stored row invariants**, not exchange-feed completeness. A `timely` row does not prove that the WebSocket delivered every trade. The audit contains no prices, fills, fees, PnL, alpha estimate, or decision to trade. `five_year_independent_oos_proven` stays false even if row coverage eventually reaches five years. Any strategy using this source needs its own preregistration, independently reconciled data, execution model, and full Gate before paper use.
