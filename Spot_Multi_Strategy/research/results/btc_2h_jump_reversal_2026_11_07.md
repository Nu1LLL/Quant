# BTC two-hour extreme-move reversal — sealed result

**Decision: reject.** This is the single fixed run of experiment
`btc_2h_jump_reversal_2026_11_07`, pre-registered in `32bff50`,
clarified before results in `4bfac91`, implemented in `a8ab3ea`, and
rerun with transport-only retry fix `c971bcc`. The first attempted
download stopped on a TLS timeout while fetching a funding CHECKSUM and
produced **no strategy result**. The successful run used the same
asset, years, signal, cost, leverage, and Gate; the full machine-readable
result is in the adjacent JSON file.

| Fixed 1x, cost-net metric | 2021–2025 result | Required |
| --- | ---: | ---: |
| CAGR | -12.03% | ≥40% (40–60% target) |
| Daily Sharpe | -1.347 | >1.5 |
| Minute-stress max drawdown | 53.40% | ≤20% |
| Daily dollar-PnL profit factor | 0.522 | >1.3 |
| Positive / negative / flat months | 13 / 45 / 2 | Stable monthly profit sought |
| 21-day block Monte Carlo joint pass | 0 / 5,000 | ≥80% |

There were 170 completed trades and 5,478 checked funding events.
Only 2021 had a positive annual fold (+2.18%); 2022–2025 were all
negative. Only the bear/high-volatility state had positive aggregate
trade PnL, so the registered four-state robustness test failed. Ending
equity under doubled adverse execution costs was 3,152.25 USDT.
No 3x/5x scenario was run because the 1x signal failed.

The official 2020–2025 BTCUSDT USD-M minute archives matched the
pre-result 72-month SHA manifest and complete minute grid. The 60
funding ZIP checksums and rates passed the runner's integrity checks;
their actual SHA values are included in the JSON result. This does
not establish historical quote availability, exact fills, or
prospective independent OOS. Those evidence Gates also remain
unpassed. No order was sent.

The older [published Bitstamp study](https://ledger.pitt.edu/ojs/ledger/article/download/213/212/1232)
reported gross results on 2015–2018 data without fees. This Binance
perpetual result is a different venue and period, under the frozen
cost assumptions. It does not support deploying the mechanism. The
five-year outcome is sealed and must not be used to retune the same
mechanism on these years.
