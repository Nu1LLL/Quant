# MERFX Merger Arbitrage Mutual Fund — Strict OOS Result

## Decision

**Rejected.** The first-read adjusted-price record covers 9,473 observations from
1989-01-31 through 2026-09-11, exactly matching the official inception date at
the start. The 37.6-year history and annual walk-forward pass, but the primary
return, Sharpe and profit-factor targets fail by large margins.

## Frozen result

| Leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | 4.89% | 1.107 | -15.12% | 1.243 | 30/36 (83.3%), pass | Fail |
| 2x | 5.39% | 0.657 | -30.86% | 1.137 | 24/36 (66.7%), fail | Fail |
| 3x | 5.58% | 0.507 | -43.99% | 1.104 | 18/36 (50.0%), fail | Fail |

The 1x case is primary. It passes the sample, cost, drawdown and annual
walk-forward checks, but fails CAGR >=40%, Sharpe >1.5 and profit factor >1.3.
Its worst rolling three-year Sharpe is only 0.046. Synthetic leverage raises CAGR
by less than one percentage point because financing and maintenance friction
consume the low-volatility spread return, while drawdown roughly doubles/triples.

## Robustness

- Monte Carlo: 5,000 circular 21-day block resamples, fixed seed 22360679; joint
  target pass rate 0.0%. Median CAGR was 4.91%, median Sharpe 0.928, fifth-
  percentile CAGR 3.42%, fifth-percentile Sharpe 0.620, and fifth-percentile max
  drawdown -21.03%.
- Regimes: through 2007 CAGR 6.58% / Sharpe 1.324; 2008–2019 CAGR 3.09% /
  Sharpe 0.755; 2020+ CAGR 3.40% / Sharpe 1.082. The lower recent return argues
  against treating the attractive early period as a current forecast.
- Events: 2008 returned -2.31%, 2020 returned 4.86%, and 2022 returned 0.70%.
  This is useful diversification, but not a high-return crisis alpha.
- Costs: adjusted NAV embeds the fund's own fees and internal results; the model
  additionally applies the preregistered 5 bp external friction and 4% financing
  above 1x.

## Product audit

An SEC filing identifies MERFX as The Merger Fund Investor Class, gives inception
as 1989-01-31, and describes the fund as seeking capital growth through merger
arbitrage. The provider data begins on that exact date, so the formal sample has
full inception coverage. [SEC prospectus data](https://www.sec.gov/Archives/edgar/data/701804/000089418920003809/R7.htm)
and [SEC fund description](https://www.sec.gov/Archives/edgar/data/701804/000089853118000136/tmf-ncsra.htm).

## Anti-overfit conclusion

No parameter, filter, peer fund, cost assumption, or sample boundary was changed
after the first performance read. MERFX is retired from the candidate set. Its
record shows that an unusually long, relatively stable alternative-risk-premium
fund can improve drawdown behavior without remotely supplying the required 40%
CAGR; leverage cannot repair that mismatch.
