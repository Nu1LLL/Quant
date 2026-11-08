# Binance Spot aggTrades reader: source validation only (2026-11-09)

The isolated `crypto_blackbox/binance_vision_aggtrades.py` reader now takes exactly one official monthly ZIP at a time, checks its adjacent official `.CHECKSUM`, enforces one expected CSV member, validates timestamp units/month bounds, positive price and quantity, contiguous aggregate IDs, ordered underlying trade-ID ranges and nondecreasing event times. It exposes rows only **after the complete archive passes**. Its default compressed-month cap is 512 MiB (hard cap 3 GiB on explicit override); the temporary spool is deleted on exit, and no raw archive was retained. It neither calculates a signal nor trades.

Two deliberately excluded development-only source months were audited with the official HTTP transport. They cannot be used as an independent OOS month for a later rule:

| Pair/month | Official SHA-256 | Rows | ZIP bytes | Timestamp unit | Source gate |
| --- | --- | ---: | ---: | --- | --- |
| DOGEUSDT 2020-07 | `ee8215747302ffadaa8956b9701676be1aa1f45b65eb7827c40e6fdafd4ef7c5` | 816,765 | 12,957,889 | milliseconds | pass |
| TUSDUSDT 2025-01 | `2b2b1b3ce1e4bc2fddc96e17d59551184e791283aaaa5df41da017e820798982` | 169,131 | 2,649,373 | microseconds | pass |

No price, notional, flow feature, return, or strategy outcome was printed or retained. The [official Binance public archive schema](https://github.com/binance/binance-public-data/blob/master/README.md) explains the 2025 timestamp change and the meaning of aggregate trades. Archive verification is not historical point-in-time publication proof, a full five-year coverage check, an executable spread/latency reconstruction, or independent OOS. Those remain formal experiment gates. The original mature strategy and all sealed experiment reports remain untouched.
