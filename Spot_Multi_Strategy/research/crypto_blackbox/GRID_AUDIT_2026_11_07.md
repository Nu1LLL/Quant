# Public archive integrity audit — 2026-11-07

This is a **data-only** audit. It did not read any price, return, or
strategy outcome, and retained no raw ZIP or CSV file. Each monthly Binance
Vision archive was streamed, matched to its official `.CHECKSUM`, and read
to the end so ZIP CRC was checked. CSV open timestamps alone were compared
with the expected UTC minute grid. This establishes the current archive
version's file and timestamp integrity; it does not establish point-in-time
publication, historical revisions, executable bid/ask, or an Alpha.

| Source and interval | Months | Timestamp rows | Grid result |
| --- | ---: | ---: | --- |
| BTCUSDT USD-M perpetual, 1m, 2020-01 to 2025-12 | 72/72 checksum pass | 3,156,480 | Every expected UTC minute present; zero duplicate or off-grid rows |
| BTCUSDT spot, 1m, 2021-01 to 2025-12 | 60/60 checksum pass | 2,628,367 | Six months contain 1,073 missing minutes in total; no duplicates |
| BTCUSDT USD-M fundingRate, 2021-01 to 2025-12 | 60/60 checksum pass | 5,478 | Eight-hour intervals, no duplicate or gap above 12 hours; rate values were not inspected in this audit |

Spot missing-minute counts: 2021-02 **79**, 2021-03 **90**,
2021-04 **434**, 2021-08 **270**, 2021-09 **120**, and 2023-03 **80**.
Separately, BTCUSDT spot 2020-06 has one 210-minute gap from
2020-06-28 02:00 through 05:29 UTC, despite a matching official
checksum. No missing bar is silently interpolated.

The fixed USD-M K-line archive hashes for the registered reversal experiment
are in
`../preregistrations/2026-11-07_btc_2h_jump_reversal_archive_manifest.csv`.
The official archive and format are documented at
https://github.com/binance/binance-public-data . A full strategy Gate still
needs fees, funding-rate values, slippage, independent OOS, and execution
evidence; this audit certifies none of those outcomes.
