# BDRY Dry-Bulk Freight Futures ETF — Strict OOS Result

## Decision

**Rejected.** The first-read adjusted-price record contains 2,130 observations
from 2018-03-22 through 2026-09-11, exactly matching the official commencement
date. Full-history performance is negative and drawdown is catastrophic despite
several isolated boom years.

## Frozen result

| Leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | -6.06% | 0.248 | -89.33% | 1.036 | 0/7, fail | Fail |
| 2x | -42.11% | 0.209 | -99.86% | 1.030 | 0/7, fail | Fail |
| 3x | -76.08% | 0.196 | -100.00% | 1.028 | 0/7, fail | Fail |

Every primary numeric target fails. No complete calendar year passes the joint
positive-return, profit-factor and maximum-20%-drawdown fold rule. Leverage does
not turn the intermittent freight spikes into a stable return stream; it makes
capital loss nearly total.

## Robustness and regimes

- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 20180322. Median
  CAGR -5.57%, fifth-percentile CAGR -36.89%, median Sharpe 0.213,
  fifth-percentile Sharpe -0.450, and fifth-percentile max drawdown -99.17%.
  Joint target probability is 0.0%.
- Inception–2019: CAGR -25.07%, Sharpe -0.440, max drawdown -63.94%.
- 2020–2022: CAGR -16.74%, Sharpe 0.181, max drawdown -83.16%.
- 2023 onward: CAGR 15.47%, Sharpe 0.645, but max drawdown still -69.94%.
- Calendar extremes expose the problem: 2021 returned 284.78% with Sharpe
  2.343, while 2020, 2022 and 2024 returned -50.71%, -69.54% and -47.86%.
  2026 YTD returned 137.76%, but even that partial year suffered a -21.63%
  drawdown. Isolated freight squeezes do not satisfy multi-year stability.

## Product and execution audit

SEC filings confirm that BDRY commenced investment operations and NYSE Arca
trading on 2018-03-22. It is a commodity pool holding a three-month strip of the
nearest calendar-quarter dry freight futures tied to Baltic Exchange Capesize,
Panamax and Supramax reference indexes. A March 2026 filing reported market
capitalization of approximately $42.6 million, adding meaningful liquidity and
closure risk for a small account even though ETF shares are nominally accessible.

Sources: [SEC 2026 product and inception disclosure](https://www.sec.gov/Archives/edgar/data/1610940/000121390026057535/ea0289571-10q_amplify.htm),
[SEC prospectus](https://www.sec.gov/Archives/edgar/data/1610940/000121390026016644/ea027613301-424b3_amplify.htm).

## Anti-overfit conclusion

No shipping-cycle timing, China macro filter, ship-class reweighting, year
exclusion, cost change or leverage selection was introduced after the first
read. BDRY is retired. This is a direct example of a high-convexity-looking asset
whose occasional triple-digit annual return coexists with negative long-run
compounding and unacceptable path risk.
