# Prospective source integrity: sleeping laptop and remote daily fallback

This is an operational **source audit**, not a strategy experiment. No new signal PnL, Gate statistic or order was computed.

## Local Binance WebSocket journal

The fixed 12-symbol listener activated on 2026-11-09 00:00 UTC. A read-only audit at **2026-11-11 07:18 UTC** found **55 due hours per symbol, zero complete all-symbol hours, zero missing database rows, and a longest complete run of zero**. BTCUSDT, like the other frozen symbols, had 53 `invalid` and two `unavailable` rows; the additive late-message table held four post-seal affected symbol/hours at that audit. Later, the stopped journal contained 660 hourly rows and five incidents. The journal was compacted by a lossless SQLite WAL checkpoint to about 72 KiB of main database with zero WAL bytes; no row or incident was deleted.

The macOS `pmset` power log showed repeated system sleep and brief maintenance wakeups while on battery during the same interval. The minute-level gaps are therefore real unavailable periods, not zero aggressor flow. The compatible late-message incident extension was deployed to the installed collector (file hash matched the committed version), and the audit now excludes any incident-marked hour without rewriting its source row. Because every observed hour was invalid anyway and the laptop kept sleeping, the local LaunchAgent was **disabled and booted out** on 2026-11-11. The database and plist remain for audit or a future deliberate restart. The local source cannot establish five years of continuous OOS.

## Always-on daily source, with a slower trading clock

An independent [GitHub Actions workflow](../../../.github/workflows/binance-vision-prospective.yml) and [source contract](../crypto_blackbox/BINANCE_VISION_DAILY_PROSPECTIVE_CONTRACT.md) were committed on the public repository before their 2026-11-13 00:00 UTC activation. They fetch the *next-day* [official Binance Vision spot `aggTrades` archives](https://github.com/binance/binance-public-data/blob/master/README.md), verify each `.CHECKSUM`, and commit only compressed hourly summaries plus their actual capture timestamps. The first manual preactivation workflow run `42971724294` finished successfully and explicitly wrote **no source file**. This path avoids laptop sleep, but GitHub scheduling can be delayed and historical ZIP publication is not a real-time order book. It supports a future daily or later signal clock; it is not a substitute for minute execution or five years of independent OOS today.

The full strategy Gate remains unpassed. The user would need a genuinely always-on host for a minute-level prospective feed. No paper or live trading has been activated.
