# BTC quarter-hour transfer: first formal invocation code failure

The first formal invocation of the preregistered simulator terminated with
`TypeError: function missing required argument 'day'` in the annual-fold date
constructor. Verified source prices and day factors had been loaded in memory,
but no fold metrics, result file, console PnL or Sharpe were generated or read.
The code-only repair supplies the missing day argument to the fixed calendar
boundary; it does not alter any registered asset, trade minute, cost, sample,
leverage, statistic or Gate. A synthetic fold-boundary test is added before
the next invocation. This is an implementation failure, not an observed
strategy result.
