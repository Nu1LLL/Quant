# DIA OOS — Pre-result data-request amendment

The first request made after preregistration used Yahoo's `range=max` together
with `interval=1d`. Yahoo returned 345 monthly bars from 1998-02-01 through
2026-09-11, with repeated 28–31 day gaps. The preregistered ten-day gap gate
stopped execution before `return_components`, scenario metrics, walk-forward or
Monte Carlo were calculated.

The raw rejected response is frozen as
`inputs/DIA_max_ohlc_adjusted_REJECTED_MONTHLY.csv`. It must never be used to
estimate overnight or intraday returns.

Before any performance read, the transport request is amended to the same Yahoo
chart endpoint with explicit `period1=1998-01-01`, exclusive
`period2=2026-09-14`, and `interval=1d`. This preserves the original asset,
full-inception window, adjustment method, strategy, costs, leverage scenarios,
regimes, gates and stopping rule. It changes only the API form required to stop
Yahoo from silently downsampling a maximum-range request. If the explicit-date
response still fails the original daily coverage gate, the experiment ends as a
data failure without performance metrics.
