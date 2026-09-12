# DIA Overnight-Long / Intraday-Short — Strict OOS Result

## Decision

**Rejected.** The repaired and frozen daily OHLC record contains 7,206
observations from 1998-01-20 through 2026-09-11, covering the official
1998-01-14 inception. The overnight equity premium is positive, but the fixed
intraday short loses money and four auction fills per day cost more than the
combined gross edge. Every leverage scenario fails.

## Frozen result

| Segment leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | -6.79% | -0.364 | -95.06% | 0.944 | 6/28, fail | Fail |
| 2x | -19.28% | -0.501 | -99.96% | 0.924 | fail | Fail |
| 3x | -32.45% | -0.549 | -100.00% | 0.917 | fail | Fail |

Only 21.4% of the 28 complete calendar folds pass the joint positive-return,
PF>1 and maximum-20%-drawdown rule. Monte Carlo joint target probability is 0%.

## Component attribution

At 1x, arithmetic mean components annualize as follows:

| Component | Annualized mean |
|---|---:|
| Long close-to-open | +7.67% |
| Short open-to-close | -2.71% |
| Four fills per day | -10.08% |
| Half-day short borrow | -0.25% |

The combined gross mean is only about +4.95% before nonlinear compounding and
costs. The result is therefore not close to the required 40% CAGR. Reducing the
pre-registered 1bp-per-fill cost after seeing this table would be a direct
post-result relaxation; even zero fills would leave gross return far below target.

## Robustness and regimes

- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 19980120. Median
  CAGR -6.78%, fifth-percentile CAGR -11.50%, median Sharpe -0.301,
  fifth-percentile Sharpe -0.587, and fifth-percentile max drawdown -97.59%.
- Period CAGR is negative in every fixed block: -0.74% in 1998–2002, -5.77% in
  2003–2007, -12.72% in 2008–2012, -4.43% in 2013–2019 and -9.71% since 2020.
- 2018 is the strongest named event year at +24.24%, Sharpe 1.460 and drawdown
  -9.48%, but still fails CAGR, Sharpe and PF targets. 2022 and 2025 lose 24.10%
  and 22.71%. No event window is substituted for the full history.

## Data-amendment audit

The preregistered Yahoo `range=max, interval=1d` call silently returned 345
monthly bars. The original ten-day gap gate stopped execution before any return
or performance metric was calculated. That rejected response is preserved as
`inputs/DIA_max_ohlc_adjusted_REJECTED_MONTHLY.csv`.

Before the first performance read, commit `f666ceb` documented and implemented
an explicit 1998-01-01 to 2026-09-14 request to the same endpoint. Asset, window,
OHLC adjustment, direction, costs, leverage and all gates remained unchanged.
The corrected response passed the original daily coverage rule.

## Product and $10,000 execution audit

State Street reports DIA's 1998-01-14 inception, $45.36 billion of assets,
0.16% expense ratio and 0.01% current median bid-ask spread. The formal 1bp fill
assumption equals the whole displayed spread on each execution and is not an
obviously punitive explanation for a result whose gross return is already low.

Direct daily ETF shorting is not reliably executable in a $10,000 U.S. margin
account. FINRA's current pattern-day-trader framework uses four day trades in
five business days and a $25,000 minimum, although a replacement intraday-margin
regime took effect in 2026 with a broker transition period through 2027. Account
location and broker implementation therefore remain material. Using MYM futures
would avoid an ETF day trade but introduces a distinct price, session, margin and
roll series that this DIA proxy does not validate. This practical failure is
additional to, not the cause of, the statistical rejection.

Sources: [State Street DIA product page](https://www.ssga.com/us/en/intermediary/etfs/state-street-spdr-dow-jones-industrial-average-etf-trust-dia),
[FINRA day-trading guidance](https://www.finra.org/investors/investing/investment-products/stocks/day-trading),
[SEC filing describing the 2026 rule transition](https://www.sec.gov/files/rules/sro/finra/2026/34-105226.pdf).

## Anti-overfit conclusion

No weekday, volatility, gap-size, trend or event filter was introduced, and the
intraday direction was not flipped after observing that it loses money. The
positive overnight leg is reported as attribution only; extracting it now would
be a revealed-history hypothesis rather than this experiment's independent OOS.
The exact DIA overnight-long/intraday-short rule is retired.
