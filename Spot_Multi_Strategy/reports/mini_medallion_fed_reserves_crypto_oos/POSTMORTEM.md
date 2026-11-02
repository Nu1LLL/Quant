# H.4.1 source-parser postmortem (2026-11-03)

The one-shot study is sealed in `DATA_FAILURE.json`. The run fetched H.4.1
HTML but extracted no reserve-balance value into a signal series, fetched no
Binance prices, and calculated no strategy P&L. The seal remains unchanged
and the same registered study will not be rerun.

A subsequent **structure-only** inspection of four official H.4.1 release
pages (2020-11-05, 2021-01-07, 2023-10-26, 2025-12-29) found that the target
row is in the third HTML table, headed `1. Factors Affecting Reserve Balances
of Depository Institutions (continued)`. The parser instead required the
nonexistent literal `Table 1. Factors Affecting Reserve Balances` in its
nearby text. It also expected the publication date in the last 2,400
characters before that table; in the inspected pages the unambiguous date is
in the page `<title>`. These two parser assumptions explain the immediate
rejection. The fixture had reproduced the assumptions rather than the actual
old/new official page structure.

Thus `status=data-ineligible` records the code's fail-closed source rejection,
**not** evidence that the source is missing, the signal lost money, or the
trading Gate failed. A later study may fix both structural checks only after
separate preregistration and a committed, tested implementation. This report
does not amend the old result or select a new asset, parameter, or direction.

Sources: [2020-11-05](https://www.federalreserve.gov/releases/h41/20201105/h41.htm),
[2021-01-07](https://www.federalreserve.gov/releases/h41/20210107/h41.htm),
[2023-10-26](https://www.federalreserve.gov/releases/h41/20231026/h41.htm),
[2025-12-29](https://www.federalreserve.gov/releases/h41/20251229/h41.htm).
