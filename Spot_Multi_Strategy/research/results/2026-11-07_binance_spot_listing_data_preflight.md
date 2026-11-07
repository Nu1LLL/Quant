# Binance spot-listing event source preflight (no strategy test)

**Date:** 2026-11-07. **Status:** event-source coverage unproven; no preregistration or return/PnL read. This note does not add an alpha candidate to the formal scorecard.

## Independent mechanism worth screening

Exchange cross-listing can expand market access and investor participation. [Ante's 2019 event study](https://www.blockchainresearchlab.org/de/portfolio/market-reaction-to-exchange-listings-of-cryptocurrencies-2/) reports abnormal returns around listings, but much of its observation window includes *pre-listing* days and exchange effects differ. Its gross event-study returns cannot establish a post-announcement, post-first-fill tradable return. A new Binance spot-listing experiment would need to test that narrower, executable claim without selecting winners after prices are seen.

## Read-only source checks, before seeing returns

- Binance's public [new-listing announcement catalog](https://www.binance.com/en/support/announcement/new-cryptocurrency-listing?c=48) is exposed through its public `bapi/composite/v1/public/cms/article/list/query` endpoint with `catalogId=48`. A page response reported **2,292** catalog articles at the time of inspection. Individual page checks reached articles dated 2017 through 2026; page 80 reached March 2021. The catalog contains spot, futures, margin and other announcements, so the whole category cannot be treated as a spot-listing universe.
- A [2021 ICP announcement](https://www.binance.com/en/support/announcement/detail/4fa91d953fd0484ab9a48cca0a41c192) shows both a publication time and a later scheduled spot trading time. A [2021 KEEP announcement](https://www.binance.com/en/support/announcement/detail/ed88b0ac99554ad7a5c7cae58fb0f492) likewise shows explicit opening pairs and trading time. These samples establish that event timing can be checked article by article, **not** that all five-plus years of eligible articles are retrievable.
- One page in the historical range mixed a 2025 `releaseDate` article among 2021 records. Pagination order therefore cannot be assumed to be a strict timestamp order. Article codes must be deduplicated and the complete catalog audited before making date-window or first-listing claims.
- A bounded batch attempt to enumerate the full 115-page catalog failed when the endpoint returned **HTTP 429**. A subsequent single-page check also returned 429. No complete official event manifest was produced. No price archive, signal, trade, return or PnL was accessed in this investigation.
- Binance's [public market-data archive](https://github.com/binance/binance-public-data) documents historical spot kline ZIPs. It could supply post-listing bars, but every eligible event would still need announcement-to-symbol mapping, official ZIP/checksum verification, delisting inclusion, candle-grid checks and explicit handling of missing/renamed pairs.

## Decision

**Defer formal testing.** The selected sample cannot be judged complete while official catalog enumeration is rate-limited. Later collection must be slow, resumable and price-blind, record all catalog pages and article codes, classify every article by predeclared text rules, then separately verify the spot bar archive. Any preregistration must freeze event selection, precise entry delay, hold period, sizing, fees/slippage, $10,000 account, 2021–2026 split and all standard Gates before loading the first formal-window return. If a complete timestamped event history remains unavailable, abandon this hypothesis without claiming an alpha result.
