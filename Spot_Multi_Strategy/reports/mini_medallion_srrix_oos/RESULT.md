# SRRIX Reinsurance Risk Premium Fund — Strict OOS Result

## Decision

**Rejected for performance, stability, NAV quality and $10,000 executability.**

The first-read adjusted-NAV record contains 3,207 observations from 2013-12-10
through 2026-09-11. SEC reports place commencement on December 9/10, 2013, so
the sample effectively covers the full operating history. That history does not
support the target.

## Frozen result

| Leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | 1.81% | 0.270 | -33.06% | 1.127 | 7/12 (58.3%), fail | Fail |
| 2x | -1.86% | 0.048 | -69.01% | 1.021 | 1/12 (8.3%), fail | Fail |
| 3x | -7.30% | -0.027 | -86.54% | 0.988 | 1/12 (8.3%), fail | Fail |

All scenarios use the same underlying NAV-return series for the valuation-quality
checks. Exactly zero NAV return occurs on 41.52% of observations, failing the
preregistered maximum of 10%. Lag-one autocorrelation is -0.001 and passes its
separate threshold, but that cannot override the excessive zero-return frequency.

## Robustness

- Monte Carlo: 5,000 circular 21-day block resamples, fixed seed 24494897; joint
  target pass rate 0.0%. Median CAGR was 2.10%, median Sharpe 0.252, fifth-
  percentile CAGR -3.64%, and fifth-percentile max drawdown -55.15%.
- Regimes: through 2019 CAGR -4.07% / Sharpe -0.519; 2020+ CAGR 7.44% /
  Sharpe 0.782. The sign reversal is large and the full-history result is weak.
- Events: 2017 and 2018 returned -11.62% and -8.31%; 2020 returned 3.90%, 2022
  5.14%, and 2023 20.57%. The heterogeneous disaster record does not establish a
  stable crisis hedge.
- Financing: 4% financing plus fixed maintenance friction makes 2x and 3x CAGR
  negative while expanding drawdown to -69% and -87%.

## Product and execution audit

SEC filings describe SRRIX as a continuously offered closed-end interval fund
investing in reinsurance-related securities. Shareholders do not have ordinary
redemption rights; liquidity is through periodic repurchase offers. The current
filing states a $15 million minimum initial investment, subject to exceptions.
Consequently, even a hypothetical statistical pass would not establish that the
user's $10,000 account can directly implement this instrument.

Sources: [SEC inception and early financial report](https://www.sec.gov/Archives/edgar/data/1581005/000119312514263815/d739450dncsrs.htm),
[SEC interval-fund and offering terms](https://www.sec.gov/Archives/edgar/data/1581005/000119312526071908/d106850d486bpos.htm),
[SEC quarterly repurchase schedule](https://www.sec.gov/Archives/edgar/data/1581005/000089418913006358/srtii-rrpif_497e.htm).

## Anti-overfit conclusion

No disaster filter, peer fund, NAV threshold, cost, financing rate, or sample
boundary was changed after the first performance read. SRRIX is retired. The
daily NAV is neither a freely tradable daily price nor credible evidence for a
high-frequency low-volatility Sharpe, and the actual long-horizon return is far
below the target even before considering the $15 million access barrier.
