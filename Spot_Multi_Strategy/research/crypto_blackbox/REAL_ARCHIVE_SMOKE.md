# Binance Vision 1m archive integrity smoke check (2026-11-03 UTC)

This is a **data-format diagnostic only**. No strategy, signal labels, return series, or P&L were read. The two official monthly spot BTCUSDT ZIPs and their named `.CHECKSUM` files were fetched over HTTPS, checked in memory/bounded temporary spool, and not retained locally or committed.

| Archive | SHA-256 of ZIP | Observed / expected 1m rows | Outcome |
| --- | --- | ---: | --- |
| `BTCUSDT-1m-2020-02.zip` | `05aedf91c80f579da396951b1d9e84587d02c9684352e1ff17d4468f556c674e` | 41,346 / 41,760 | **Rejected**: UTC minute grid has two gaps. |
| `BTCUSDT-1m-2025-02.zip` | `26b3795072983a66bdbe8327e0d5e71069bdd5ca831d4da3850ba68ef7a991ef` | 40,320 / 40,320 | Passed whole-month schema, SHA-256, and UTC grid checks; spot timestamps were microseconds. |

The 2020-02 archive's first absent minute spans start at `2020-02-09T02:00:00Z` and resume at `03:00:00Z` (60 absent bars). The second spans start at `2020-02-19T11:36:00Z` and resume at `17:30:00Z` (354 absent bars). These are **archive observations, not a determination of why trading stopped**. The official [Binance notice](https://www.binance.com/en/support/announcement/detail/360039984971) confirms trading resumed at 17:30 UTC on February 19, but does not explain the February 9 gap. No bars were synthesized, forward-filled, or dropped to make the audit pass.

Only these two months were checked. This neither certifies a five-year dataset nor establishes point-in-time bar publication, executable bid/ask quotes, or independent OOS. Any future formal strategy must pre-register its own fixed data gate before results are read; a historical outage or missing bar may not be silently excused after seeing P&L.
