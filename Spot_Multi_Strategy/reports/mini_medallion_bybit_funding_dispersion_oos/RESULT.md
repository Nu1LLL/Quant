# Bybit Cross-Sectional Funding Dispersion — Strict OOS Result

## Decision

**Rejected before performance evaluation: preregistered OOS data coverage failed.**

The fixed universe was BTCUSDT, ETHUSDT, XRPUSDT and EOSUSDT, with an immutable
2021-01-01 OOS start and a complete common eight-hour grid required. The first
official API read found:

| Symbol | Merged funding/mark rows | First available | Covers OOS start? |
|---|---:|---|:---:|
| BTCUSDT | 6,604 | 2020-09-01 00:00 UTC | Yes |
| ETHUSDT | 6,018 | 2021-03-15 08:00 UTC | No |
| XRPUSDT | 5,840 | 2021-05-13 16:00 UTC | No |
| EOSUSDT | 0 | No mark-price history returned | No |

Because three of four fixed contracts fail the starting-coverage rule, there is
no valid common panel. No portfolio CAGR, Sharpe, drawdown, profit factor,
walk-forward result or Monte Carlo result is reported. Computing those metrics
on the surviving three assets or moving the start to May 2021 would be a post-hoc
change and is prohibited by the preregistration.

## Pipeline audit

The first attempt exposed two provider-format issues before any performance was
computed:

1. Bybit returns funding timestamps as millisecond strings; they are now
   explicitly converted to numeric before UTC parsing.
2. The official mark-price endpoint does not support 480-minute candles. It does
   support 240-minute candles, so the loader uses the close of each completed
   four-hour candle ending at 00/08/16 UTC. Adjacent selected closes still form
   the preregistered eight-hour mark-price return; strategy timing is unchanged.

Both fixes have dedicated tests. The signal remains lagged one settlement: the
rate observed at time t can only choose the position whose price and funding P&L
end at t+1.

Official API references: [funding-rate history](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)
and [mark-price kline intervals](https://bybit-exchange.github.io/docs/v5/market/mark-kline).

## Anti-overfit conclusion

The fixed symbols are not replaced, EOS is not dropped, and the OOS start is not
shifted. This hypothesis is retired as an evidence-availability failure. A future
cross-exchange replication must be preregistered with a universe whose historical
availability is established without exposing strategy performance, then tested
on data not used by the existing Binance A21/A22 or basis studies.
