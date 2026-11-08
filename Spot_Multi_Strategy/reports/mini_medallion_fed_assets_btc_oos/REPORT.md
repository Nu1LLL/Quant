# Sealed data failure: Fed total assets BTC filter

**Verdict: DATA FAILURE; no strategy-performance inference and no order.** The one formal attempt for `fed_walcl_btc_liquidity_2026_11_08` began 2026-11-08 09:45 UTC and sealed 09:48 UTC. [DATA_FAILURE.json](DATA_FAILURE.json) is the machine-readable seal, SHA-256 `fd9b4011df087f5e8d9218f627f846c8e777b0b9cdfd2c116c61a09`. The fixed [preregistration](../../research/preregistrations/2026-11-08_fed_assets_btc_liquidity.md) was committed/pushed at `1512f73`; frozen implementation and synthetic tests were committed/pushed at `9c78bab` before this run.

The primary FRED WALCL CSV request timed out after the loader's bounded retries:

> `HTTPSConnectionPool(host='fred.stlouisfed.org', port=443): Read timed out. (read timeout=35)`

The request had passed metadata-only availability and hash preflight before preregistration, but the formal download did not return bytes. Thus **no WALCL levels, BTC price bars, signals, trades, account returns or Gate metrics were opened in this formal run**. This establishes neither a profitable nor an unprofitable rule. The fixed run's source Gate failed, so the same hypothesis and 2021–2025 sample are sealed without fallback vendor, changed timeouts, or a same-history second try. The generic risk limits and genuine OOS requirements remain unmet and no live or paper trade was placed.

The failure reflects access to the [FRED WALCL download](https://fred.stlouisfed.org/series/WALCL) on this attempt. It does not establish a change in the Federal Reserve's underlying [H.4.1 release](https://www.federalreserve.gov/releases/h41/). Future research can use a separately preregistered independent question or begin an append-only publication-time archive; this particular result is permanently a data failure.
