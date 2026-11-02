# KRBN Global Carbon Allowance ETF — Strict OOS Result

## Decision

**Rejected.** The first-read adjusted-price history has 1,536 observations from
2020-07-31 through 2026-09-11 and covers the fund's official 2020-07-29
inception closely enough to pass the preregistered inception gate. Performance,
risk, annual stability and transaction-cost realism do not support deployment.

## Frozen result

| Leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | 15.74% | 0.796 | -36.44% | 1.124 | 3/5 (60%), fail | Fail |
| 2x | 18.70% | 0.711 | -71.71% | 1.110 | 0/5, fail | Fail |
| 3x | 11.92% | 0.682 | -91.17% | 1.105 | 0/5, fail | Fail |

The primary 1x scenario fails CAGR, Sharpe, drawdown, profit factor and
walk-forward. Financing and volatility drag make 3x CAGR lower than both 1x and
2x while drawdown approaches total loss.

## Robustness and regimes

- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 28284271. Median
  CAGR 16.12%, fifth-percentile CAGR -2.49%, median Sharpe 0.677,
  fifth-percentile Sharpe 0.063, and fifth-percentile max drawdown -61.18%.
  Only 0.14% of resampled paths jointly meet the four numeric targets. A tiny
  favorable-tail frequency is not a pass and does not override the actual path.
- Inception–2022: CAGR 38.22%, Sharpe 1.296, max drawdown -36.44%.
- 2023 onward: CAGR 3.12%, Sharpe 0.302, max drawdown -27.95%. This large
  deterioration is inconsistent with stable 40% annual compounding.
- Calendar years: 2021 returned 109.06% with Sharpe 2.763, but 2022 and 2024
  returned -13.09% and -13.73%; 2026 YTD was also negative. The result is
  dominated by one early year rather than a repeatable annual edge.

## Product and implementation audit

The sponsor gives inception as 2020-07-29 and describes KRBN as tracking the S&P
Global Carbon Credit Index through liquid cap-and-trade allowance futures. The
current portfolio spans European, UK and North American programs, and annual
fund operating expense is 0.91%.

The formal test retains the preregistered 5 bp single-side external friction.
However, the sponsor currently reports a 30-day median bid/ask spread of 0.89%,
far above that assumption. This current snapshot is not substituted post hoc
into the historical test, but it makes the formal result optimistic for a
$10,000 account. Since the optimistic result already fails, a higher realistic
spread cannot rescue it.

Sources: [KraneShares KRBN fund page](https://kraneshares.com/etf/krbn/),
[SEC 2026 summary prospectus](https://www.sec.gov/Archives/edgar/data/1547576/000182912626008205/kraneshares-krbn_497k.htm).

## Anti-overfit conclusion

No trend filter, energy-price overlay, regional reweighting, cost change or year
exclusion was introduced after the first read. KRBN is retired. The sample shows
that a distinct policy-created asset can deliver a strong early surge without
providing stable high Sharpe or acceptable drawdown across subsequent regimes.
