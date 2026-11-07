# Ten-coin lead-lag: sealed data failure addendum

The one-shot formal runner for pre-registration `0976483` and pre-result
boundary amendment `982e707` stopped at the first invalid numeric row. Its
immutable JSON outcome is `ten_coin_delayed_leadlag_2026_11_07.json` with
`status: sealed_failed_run` and `gate.full: false`. **No fitted model,
strategy trade, CAGR, Sharpe, drawdown, PF or Monte Carlo result was produced.**

The verified official Binance USD-M `XTZUSDT-5m-2023-09.zip` has a CSV record
opening **2023-09-21 07:05 UTC** (`1695279900000` ms) where the reported
*taker-buy base-asset volume* exceeds the reported *total base-asset volume*
by **6,135.3 XTZ**. This is far beyond rounding tolerance. The archive
matched the committed SHA-256 and the newly fetched official `.CHECKSUM`, and
the earlier timestamp-only preflight found a complete 5-minute grid; those
checks alone did not establish numeric data consistency. The row's other
numeric validity checks did not trigger the stop. The follow-up inspection
printed only the failure flags and difference, not price returns or strategy
outcomes.

The pre-registered universe and this exact historical test remain closed.
No volume repair, interpolation, row deletion, symbol replacement or
same-history rerun is permitted to promote this candidate. Future distinct
economic hypotheses should audit complete numeric source consistency before
locking an asset universe and should still treat current archive checksums
as distinct from point-in-time executable quotes.
