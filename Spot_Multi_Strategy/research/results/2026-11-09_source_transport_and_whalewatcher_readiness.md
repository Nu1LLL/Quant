# Source transport and WhaleWatcher readiness, 2026-11-08 UTC

This is a **source and integration audit**, not an Alpha experiment. No new hypothesis, signal return, PnL, option skew, or trading order was computed. The protected mature strategy and all prior sealed results remain unchanged.

## Deribit historical transport

- The prior preregistered BTC option-taker experiment remains sealed as `DATA_FAILURE`. Its additional numeric trade-ID monotonicity requirement failed; it was **not** rerun.
- A separate, generic `crypto_blackbox/deribit_history_archive.py` now fetches a bounded time interval from the [official Deribit history method](https://docs.deribit.com/api-reference/market-data/public-get_last_trades_by_currency_and_time), recursively bisects `has_more` pages, requires UTC-millisecond bounds and currency-unique trade IDs, and returns a canonical source hash. It does not compute a strategy or authorize execution. Current-vintage trades do not prove historical first-seen availability or executable option quotes.
- Four synthetic tests passed. A **source-only** smoke check of the already identified 2022-05-09 12:00–13:00 UTC interval returned 593 unique timestamp-bounded trades in one complete request, with zero splits. No direction, size, option pressure, price or return was printed or retained. This establishes that future preregistrations can state a source contract without the invalid numeric-ID assumption; it does not rehabilitate the sealed experiment.

## WhaleWatcher and stablecoin-data checks

- Read-only `whalewatcher_readiness` on the existing local journals at about 2026-11-08 17:33 UTC: `news_events=fresh` with 13 current continuous polls; `macro_schedule=fresh` with 13; `events_claims=unavailable`. The common news-plus-macro source interval is about 10.5 hours, far below five years. Both five-year flags and independent trading OOS remain false. A missing directional claim is **unavailable**, not a neutral or zero event. WhaleWatcher's other chat currently reports an OpenAI quota/billing error for new AI analysis; the public news and schedule streams remain separate. No account data or AI text was copied into this repo.
- A source-only query to Coin Metrics' **entitlement-aware** `/catalog-v2/asset-metrics` for USDT and USDC requested `TxTfrValAdjUSD`, `TxTfrValUSD`, `SplyCur`, and `VelCurAdj1yr`. For both assets the free catalog returned **only `SplyCur`**; the transfer-value and velocity metrics needed for a stablecoin-activity hypothesis were not entitled. No timeseries or market outcome was requested. Coin Metrics [distinguishes entitled catalog from catalog-all](https://docs.coinmetrics.io/api-reference/api-conventions) and [defines adjusted transfer value](https://docs.coinmetrics.io/network-data/network-data-overview/transactions/transfer-value). This candidate was not preregistered or tested.

## Research status

There is still **no candidate passing the full Gate**. Source continuity and a current-vintage archive are not independent OOS. The next formal experiment must be a new economic hypothesis with a separately committed preregistration and five-year reproducible history; neither this transport fix nor a minor variant of the sealed option-taker rule qualifies.
