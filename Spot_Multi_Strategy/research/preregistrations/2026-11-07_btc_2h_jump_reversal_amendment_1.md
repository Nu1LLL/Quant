# Pre-result clarification 1: BTC two-hour reversal

This amends only the implementation ambiguities in pre-registration commit
`32bff50`. It is committed before any 2021-2025 price, return, trade, or
strategy-result record is read. The original pre-registration remains intact.
No asset, signal threshold, hold time, cost, leverage, sample year, or Gate is
changed.

1. **OOS endpoints.** 2020 is warm-up with no entries. Start evaluating
   signals at 2021-01-01 00:00 UTC. OOS is the half-open interval
   [2021-01-01 00:00, 2026-01-01 00:00) UTC, exactly 1,826 days.
   A signal is ignored if its fixed `t+121min` exit open would be at or
   after 2026-01-01 00:00 UTC. No terminal position is force-closed or
   valued using 2026 data. Annual fold boundaries inside OOS may have an
   open trade; allocate each fold's return from its daily marked equity
   path, not by assigning the whole trade to its exit year.
2. **Funding notional.** Position BTC quantity is fixed at entry:
   `entry_USD_notional / adverse_entry_fill_price`. For a funding
   `calc_time` within the half-open holding interval
   [entry_time, exit_time), multiply the signed archived rate by this BTC
   quantity times the most recent **completed** one-minute trade-price
   close before `calc_time`. A long pays a positive rate; a short receives
   it. This trade close is a reproducible proxy, not Binance's exact
   funding-time mark price. Missing or invalid proxy data fail the data
   Gate. The separately registered no-beneficial-funding-credit stress
   still applies; exact exchange settlement remains unverified.
3. **Regime labels.** At a signal boundary `t`, the 90-day state uses
   the just-completed two-hour close divided by the two-hour close exactly
   90 days earlier: return >=0 is bull, <0 is bear. Current volatility
   is the prior-360-two-hour-return sample standard deviation already
   defined in the primary registration. At every past UTC midnight,
   compute that same trailing 30-day estimate from data completed before
   that midnight. Compare current volatility to the median of the most
   recent 365 **past daily** volatility snapshots, excluding the current
   day; at least 300 past daily snapshots are required. Volatility >= the
   median is high, otherwise low. If history is insufficient, a required
   regime label is unavailable and the regime Gate fails. Attribute the
   entire completed trade's net PnL to its entry-time regime bucket.
4. **Metrics.** Five-year CAGR uses
   `(ending_equity / 10000) ** (365.25 / 1826) - 1`. Each annual fold uses
   its own full-year UTC day count and marked fold-start/fold-end equity.
   Max drawdown includes the initial 10,000 peak and all minute-end marked
   equity values, plus the adverse intraminute low (long) or high (short)
   mark for every position-bearing minute; Gate uses the worst peak-to-
   trough value. For each Monte Carlo path, resampled daily net returns
   compound from 10,000 equity, and daily profit factor is recomputed
   from that path's signed **dollar PnL**, not from unweighted percentages.

This clarification does not convert a retrospective bar-data simulation
into true point-in-time executable evidence. Full execution and independent
prospective Gates remain unpassed without additional data.
