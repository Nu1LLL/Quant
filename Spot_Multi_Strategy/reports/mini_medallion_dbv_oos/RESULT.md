# DBV G10 Currency Carry ETF — Strict OOS Result

## Decision

**Rejected. Evidence is also incomplete for the preregistered full-lifetime test.**

The frozen Yahoo adjusted-close series contains 3,828 daily observations from
2008-01-02 through 2023-03-16. Official filings state that DBV began trading in
September 2006, so approximately the first fifteen months of the ETF's life are
missing. `complete_lifetime_data` is therefore a hard failure. The observed
2008–2023 path independently fails every investment target by a wide margin, so
the missing history is not backfilled or used to tune a replacement rule.

## Frozen primary result

| Leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | -0.49% | 0.023 | -34.12% | 1.004 | Fail | Fail |
| 2x | -6.25% | -0.176 | -72.16% | 0.972 | Fail | Fail |
| 3x | -13.00% | -0.243 | -91.40% | 0.961 | Fail | Fail |

The 1x case is primary. Higher leverage is reported only as preregistered stress
testing and makes the outcome worse.

## Robustness

- Walk-forward: 15 annual folds, of which 9 passed (60.0%); this fails the
  preregistered requirement of at least 15 folds with at least 80% passing.
- Monte Carlo: 5,000 circular 21-day block resamples, fixed seed 17320508; joint
  target pass rate 0.0%. Median CAGR was -0.49%, median Sharpe 0.019, fifth-
  percentile CAGR -4.67%, and fifth-percentile max drawdown -60.65%.
- Regimes: inception–2012 CAGR -0.81%, 2013–2019 CAGR -1.09%, and 2020+ CAGR
  1.32%. The observed 2008 event year lost 28.33% with Sharpe -1.781.
- Costs: the ETF adjusted price embeds distributions, fund expenses and index
  implementation; the simulation additionally includes the preregistered 5 bp
  leverage-maintenance friction and 4% financing on exposure above 1x.

## Data audit

Official SEC material describes DBV as tracking the Deutsche Bank G10 Currency
Future Harvest Index, with quarterly rebalancing and index leverage of up to 2:1.
An SEC filing reports inception of trading in September 2006. The sponsor later
approved liquidation in January 2023, and OCC documentation records cessation of
NYSE trading after 2023-03-03 and liquidation cash of $25.70093 per share.

Sources: [SEC 2020 fund description](https://www.sec.gov/Archives/edgar/data/1354730/000119312520124811/d903986dfwp.htm),
[SEC 2019 inception disclosure](https://www.sec.gov/Archives/edgar/data/1354730/000119312519220094/d778951ds1.htm),
[SEC 2023 liquidation filing](https://www.sec.gov/Archives/edgar/data/1354730/000119312523013236/d423170d8k.htm),
[OCC liquidation memo](https://infomemo.theocc.com/infomemos?number=52104).

The provider series extending to 2023-03-16 likely reflects post-trading
liquidation accounting rather than ordinary tradable closes. It is retained as
the immutable first-read dataset, but this survivorship/termination detail and
the missing inception segment both reduce evidentiary quality; neither can turn
the plainly negative observed result into a pass.

## Anti-overfit conclusion

No parameter, filter, cost assumption, or sample boundary was changed after the
first performance read. DBV currency carry is retired from the candidate set.
