# Price-terminal and settlement clarification, before any strategy data read

Parent preregistration: `2026-11-08_eth_defi_tvl_relative_flow.md`, commit `ad00eed`. This clarification fixes two boundary ambiguities without changing the source, sign, asset, dates, risk, fees, timing, Gate or model.

The January 2026 spot ZIP is SHA-256 and member-name verified against the manifest and live official checksum, but **only its first CSV row is parsed**. Only that row's 2026-01-01 00:00 UTC timestamp and open price are used for the mandatory end sale; no later January price field is inspected, and the full-grid requirement applies only through December 2025. The terminal sale proceeds and fee are included in the **2025-12-31 daily end equity**, so the 1,826 daily account factors exactly reconcile to final cash at 2026-01-01 00:00. Its adverse equity is included in the final day's high-before-low drawdown record. All annual/monthly and bootstrap calculations use these reconciled factors.
