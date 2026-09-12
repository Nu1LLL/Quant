# PFIX Long-Rate-Convexity ETF — Strict OOS Result

## Decision

**Rejected.** The frozen first-read adjusted-price record contains 1,341
observations from 2021-05-11 through 2026-09-11 and covers the official
2021-05-10 inception. PFIX captured the 2022 rate shock, but neither its full
path nor any leverage scenario meets the return, risk, profit-factor or
walk-forward targets.

## Frozen result

| Leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | 16.61% | 0.717 | -36.25% | 1.103 | 0/4, fail | Fail |
| 2x | 13.28% | 0.653 | -65.53% | 1.093 | 0/4, fail | Fail |
| 3x | -4.65% | 0.632 | -86.28% | 1.090 | 0/4, fail | Fail |

Every primary numeric target fails. All four complete calendar years fail the
joint positive-return, profit-factor and maximum-20%-drawdown fold rule. The
pre-registered four-fold exception therefore does not rescue the evidence.

## Robustness and regimes

- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 20210510. Median
  CAGR 16.46%, fifth-percentile CAGR -9.34%, median Sharpe 0.591,
  fifth-percentile Sharpe -0.078, and fifth-percentile max drawdown -73.07%.
  Joint target probability is 0.0%.
- 2021 partial period: CAGR -36.66%, Sharpe -2.201, max drawdown -26.25%.
- 2022: CAGR 91.74%, Sharpe 1.908, but max drawdown -30.22% and PF 1.282.
- 2023–2024: CAGR 19.08%, Sharpe 0.772, max drawdown -36.25%.
- 2025 onward: CAGR 7.34%, Sharpe 0.455, max drawdown -28.33%.
- Complete years 2023, 2024 and 2025 returned 5.02%, 35.09% and 0.02%; 2026
  YTD returned 18.90%. A successful rate-shock year does not establish a stable
  standalone return stream.

## Product and execution audit

The issuer reports a 2021-05-10 inception, 0.50% expense ratio and approximately
$181.1 million of assets as of 2026-09-09. It describes PFIX as holding a large
OTC interest-rate-option position, functionally similar to long-dated puts on
20-year US Treasury bonds. The SEC prospectus supplement confirms use of payer
swaptions, interest-rate options and Treasury futures with high rate sensitivity.

The issuer also reported a 0.54% 30-day median bid-ask spread, versus the formal
15bp single-side model. It warns that OTC-product spreads can cause persistent
premiums or discounts to NAV. The backtest is therefore an optimistic execution
upper bound for a $10,000 account; realistic friction cannot repair a failed result.

Sources: [issuer product page](https://www.simplify.us/etfs/pfix-simplify-interest-rate-hedge-etf),
[SEC 2025 prospectus supplement](https://www.sec.gov/Archives/edgar/data/1810747/000182912625002703/simplify-pfix_497.htm).

## Anti-overfit conclusion

No rate, CPI, MOVE, duration, trend or timing filter was added after the first
read. Costs and leverage were not changed, and no year was excluded. PFIX is
retired as a standalone return candidate. Its 2022 payoff supports the economic
hedging mechanism, but a crisis hedge can be useful in a portfolio without being
a high-CAGR, high-Sharpe strategy by itself.
