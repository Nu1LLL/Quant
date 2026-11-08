# Independent-source preflight: Lightning, news, dollar strength (2026-11-09)

This is a **source and novelty audit only**. No asset return, signal direction, PnL, Gate statistic, or trading order was computed. It does not reopen any sealed hypothesis or imply independent executable OOS.

## Lightning Network public capacity

The [Aethereal producer page](https://dataset.aethereal.top/datasets/lightning-network-history) advertises a daily Lightning channel/capacity CSV over 2018-03-21–2026-10-21. A source-only read of its published CSV, with no price data, returned 2,073 rows, no duplicate dates and SHA-256 `6b0cd0af2277096ac404e19830791354ac50d95af039bbb0138501cb9c563aea` (85,948 bytes). The dates contain **15 gaps** longer than one day. The largest spans **2023-08-25 to 2024-12-14 (477 days)**; another spans 2026-03-09 to 2026-08-26 (170 days). There are **52 post-2024 rows with fewer than 100 public channels**, starting 2026-08-30; the producer's own preview shows 3 channels and 0.22 BTC capacity on 2026-10-20 followed by 22,680 channels and 2,530.94 BTC on 2026-10-21. These discontinuities cannot be silently interpolated or treated as adoption crashes. The series does not qualify as a reliable continuous five-year signal source. The [mempool.space Lightning statistics API](https://mempool.space/docs/api/rest) is a possible alternative, but a five-year timestamped reproducible history has not been established from it. **Decision: no preregistration, no backtest.**

## GDELT news attention

GDELT [documents `TimelineVolRaw`](https://blog.gdeltproject.org/gdelt-2-0-api-now-supports-raw-result-counts/) as article counts rather than normalized coverage. Two separate source-only requests for historical Bitcoin coverage to its DOC2 endpoint returned HTTP **429** (rate limit), even when separated in time. No complete five-year response, immutable vintage, or point-in-time news timing was obtained. News volume may also overlap the previously tested Wikipedia investor-attention family, so an independent mechanism would need more than a different publisher. **Decision: no preregistration, no backtest.**

## Dollar index

The official [Federal Reserve H.10](https://www.federalreserve.gov/Releases/H10/) and FRED broad dollar index are obtainable as source candidates, but `reports/mini_medallion_dxy_trend_oos/PREREGISTRATION.md` already tests dollar strength via ICE DXY futures. A new BTC position rule using a different dollar-index vendor would still draw on dollar strength. `research/novelty_inventory.py` now screens the `dollar_strength` family across legacy reports and new registrations; its regression test requires detection of that existing DXY preregistration. **Decision: duplicate economic source, no new Alpha claim.**

None of the three screens unlocks a new formal experiment. Continue the already frozen prospective Binance flow collection when its activation time arrives, and require a genuinely distinct, auditable five-year source before another formal preregistration. The overall strategy Gate remains unpassed; there is no live trading authorization.
