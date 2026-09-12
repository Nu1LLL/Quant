# FXA/FXB/FXC Currency Relative Momentum — Strict OOS Result

## Decision

**Rejected.** The frozen first-read common adjusted-price panel contains 5,085
observations from 2006-06-26 through 2026-09-11. The fixed 12–1 monthly
long-winner/short-loser rule has negative gross expectancy before the modeled
trading and borrow costs, so leverage cannot repair it.

## Frozen result

| Gross leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | -1.41% | -0.306 | -27.80% | 0.956 | 5/18, fail | Fail |
| 2x | -6.84% | -0.780 | -75.39% | 0.892 | fail | Fail |
| 3x | -12.21% | -0.938 | -92.20% | 0.871 | fail | Fail |

All primary metrics fail. Only 27.8% of the 18 complete annual folds pass the
joint positive-return, PF>1 and maximum-20%-drawdown rule, versus the frozen
80% requirement.

## Return and robustness audit

- At 1x, gross strategy returns average approximately -0.51% annualized before
  implementation. Trading friction averages 0.28% and short borrow 0.50%, taking
  the arithmetic net mean to approximately -1.29% annualized. Costs are not the
  cause of rejection; the pre-cost signal itself is negative.
- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 20070131. Median
  CAGR -1.40%, fifth-percentile CAGR -3.25%, median Sharpe -0.252,
  fifth-percentile Sharpe -0.619, and fifth-percentile max drawdown -50.47%.
  Joint target probability is 0.0%.
- Period CAGR is negative in 2008–2009, 2010–2014, 2015–2019 and 2020–2022.
  The 2023+ CAGR is only 0.05% with Sharpe 0.036.
- The best complete years were 2010 (+7.56%) and 2024 (+3.18%); no isolated
  high-return episode can explain away the full 19-year failure.
- When the equal-weight foreign-currency basket is above its lagged 200-day
  average, CAGR is 0.17% and Sharpe 0.115. Below it, CAGR is -1.68% and Sharpe
  -0.661. This state split is diagnostic only and was not traded.

## Product and execution audit

Issuer and SEC records confirm that the trusts began in June 2006 and that
shares started exchange trading on 2006-06-26, matching the first common market
observation. Each trust is designed to reflect the USD price of deposited AUD,
GBP or CAD plus accrued interest, if any, less expenses. The ordinary sponsor
fee is 0.40% annually. Thus adjusted market prices are materially more faithful
to a small brokerage account than bare spot-FX quotes with omitted interest.

The strategy still assumes short shares remain borrowable at a fixed 1% annual
rate and does not model tax differences or intraday premium/discount. Those are
optimistic limitations, but more realistic friction cannot rescue a negative
pre-cost result.

Sources: [Invesco FXA product page](https://www.invesco.com/us/en/financial-products/etfs/invesco-currencyshares-australian-dollar-trust.html),
[SEC FXA 2025 annual report](https://www.sec.gov/Archives/edgar/data/1353614/000119312526083544/fxa-20251231.htm),
[SEC FXB 2025 annual report](https://www.sec.gov/Archives/edgar/data/1353611/000119312526083570/fxb-20251231.htm),
[SEC FXC 2025 annual report](https://www.sec.gov/Archives/edgar/data/1353612/000119312526083543/fxc-20251231.htm).

## Anti-overfit conclusion

No currency, lookback, skip month, ranking direction, rebalance frequency or
dollar-regime filter was changed after first read. This exact three-currency
relative-momentum hypothesis is retired. A broader institutional FX-forward
panel with point-in-time carry and executable forward points would be a genuinely
new dataset; changing this revealed ETF history is not.
