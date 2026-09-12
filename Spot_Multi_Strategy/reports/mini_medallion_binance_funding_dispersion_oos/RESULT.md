# Binance Four-Coin Funding Dispersion — Strict OOS Result

## Decision

**Rejected.** The frozen first-read BCHUSDT/LTCUSDT/ADAUSDT/LINKUSDT common
panel has 7,245 consecutive eight-hour observations from 2020-02-01 through
2026-09-11. Coverage is complete, but the lagged long-lowest/short-highest
funding strategy fails every primary target and decays sharply after 2021.

## Frozen result

| Gross leverage | CAGR | Sharpe | Max drawdown | Profit factor | Walk-forward | Strict pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 1x | 6.97% | 0.367 | -83.07% | 1.058 | 0/6, fail | Fail |
| 2x | -5.25% | 0.324 | -98.29% | 1.051 | 0/6, fail | Fail |
| 3x | -27.77% | 0.315 | -99.88% | 1.050 | 0/6, fail | Fail |

No complete calendar year passes the joint positive-return, PF>1 and maximum
20% drawdown fold rule. Even 2021, which returned 216.33% with Sharpe 2.453,
had a -22.36% drawdown and therefore failed the frozen fold rule.

## Data and implementation audit

- Common-panel start 2020-02-01 00:00 UTC; end 2026-09-11 16:00 UTC.
- Expected-grid coverage 100%; maximum gap 8 hours; zero duplicate timestamps;
  every observation is at UTC 00:00, 08:00 or 16:00.
- Positions are shifted by one complete settlement interval. A mutation to any
  future funding or mark price cannot change prior weights or returns.
- Positive funding is credited to the short and charged to the long, matching
  Binance's published convention.

The 1x arithmetic component sums expose why the attractive early backtest did
not survive: gross cross-sectional price PnL summed to 3.108 and funding PnL to
0.595, but forced rank switching consumed 2.767 in modeled trading costs. On a
mean-rate basis, gross price PnL, funding and costs annualize to approximately
47.0%, 9.0% and 41.8%. This is not a low-turnover carry strategy.

## Robustness and regimes

- Monte Carlo: 5,000 circular 21-day block paths, fixed seed 20200201. Median
  CAGR 4.64%, fifth-percentile CAGR -12.82%, median Sharpe 0.301,
  fifth-percentile Sharpe -0.272, and fifth-percentile max drawdown -87.39%.
  Joint target probability is 0.0%.
- High funding-dispersion days: CAGR 19.43%, Sharpe 1.008, max drawdown -54.45%.
- Low funding-dispersion days: CAGR -10.53%, Sharpe -0.517, max drawdown -67.56%.
- Calendar CAGR falls from 65.18%, 216.33% and 21.82% in 2020–2022 to 5.85%,
  -45.95%, -18.73% and -57.54% in 2023–2026 YTD. The post-2021 degradation is
  incompatible with a stable six-year alpha claim.

Binance documents that positive funding is paid by longs to shorts, typically
at 00:00/08:00/16:00 UTC, while also warning that intervals, caps and floors can
change in extreme conditions. It further states that funding is exchanged only
by accounts holding the position at settlement. Those operational constraints,
plus liquidation, ADL and exchange-access risk, remain outside this optimistic
mark-price simulation.

Sources: [Binance funding-rate documentation](https://www.binance.com/en/support/faq/detail/360033525031),
[Binance futures fee and settlement documentation](https://www.binance.com/en/support/faq/detail/98488a516eb84e3eb34605683dffd554).

## Anti-overfit conclusion

No coin was removed, no direction was reversed, and no holding-period, spread,
trend or volatility filter was added after the first read. The especially strong
2021 fold is not selected as the strategy. This exact four-coin hypothesis is
retired; changing to a slower rank or filtering only high-spread observations
would be a new, already history-informed hypothesis rather than independent OOS.
