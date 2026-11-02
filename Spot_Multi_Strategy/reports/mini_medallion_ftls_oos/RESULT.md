# FTLS Live U.S. Equity Long/Short ETF — Strict OOS Result

## Decision

**Rejected.** The frozen first-read adjusted-price record contains 3,020
observations from 2014-09-09 through 2026-09-11, immediately after the official
2014-09-08 inception. The live fund has positive long-run returns and several
strong years, but no external leverage scenario approaches the joint return,
risk, profit-factor and walk-forward targets.

## Frozen result

| External leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | 8.99% | 0.677 | -29.27% | 1.169 | 8/11, fail | Fail |
| 2x | 10.13% | 0.545 | -58.56% | 1.134 | fail | Fail |
| 3x | 3.25% | 0.501 | -87.84% | 1.122 | fail | Fail |

At 1x, 81.8% of complete years are positive, but only 72.7% pass the complete
fold rule because 2020's positive return still contains a -20.56% drawdown.
This distinction prevents annual closing values from hiding survival risk.

## Robustness and regimes

- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 20140909. Median
  CAGR 9.09%, fifth-percentile CAGR 3.79%, median Sharpe 0.587,
  fifth-percentile Sharpe 0.301, and fifth-percentile max drawdown -41.11%.
  Joint target probability is 0.0%.
- Inception–2019: CAGR 7.74%, Sharpe 0.517, max drawdown -29.27%.
- 2020–2022: CAGR 4.96%, Sharpe 0.481, max drawdown -20.56%.
- 2023 onward: CAGR 14.29%, Sharpe 1.733, max drawdown -11.70%, but PF is only
  1.276 and CAGR remains far below 40%.
- Complete years 2017, 2019, 2021, 2023 and 2024 have Sharpe above 2, while
  2018 and 2022 lose money. The strong recent subperiod is reported but not
  selected as a replacement OOS window.

## Product and execution audit

First Trust confirms a 2014-09-08 inception and a mandate to establish both
long and short U.S. equity positions. As of 2026-09-10 it reported 94.56% long,
34.39% short and 60.17% net exposure: FTLS is lower-beta long/short, not market
neutral. Current net assets were approximately $2.53 billion.

The reported total expense ratio is 1.38%, including 0.43% of other expenses
consisting of margin interest and dividends on securities sold short. Those
costs and the fund's internal trading are already embedded in adjusted returns.
The current 30-day median bid-ask spread was 0.25%, versus the formal 5bp
single-side model, so the backtest is still an optimistic upper bound for a
$10,000 account. An SEC shareholder report also shows a 118% portfolio turnover
rate for the period reported, underscoring the value of using the live fund path.

Sources: [First Trust official FTLS page](https://www.ftportfolios.com/retail/etf/etfsummary.aspx?ticker=ftls),
[SEC April 2026 shareholder report](https://www.sec.gov/Archives/edgar/data/1424212/000144554626005042/etf3_ncsrs.htm),
[current prospectus](https://www.ftportfolios.com/LoadContent/8yhdgqbigr).

## Anti-overfit conclusion

No SPY hedge, trend, VIX, sector, valuation or net-exposure filter was added after
the first read. The 2023+ segment is not promoted to a standalone strategy, and
the cost or leverage assumptions were not changed. FTLS remains potentially
useful as a lower-beta equity allocation, but it is retired as a candidate for
the requested 40% CAGR, Sharpe-above-1.5 standalone system.
