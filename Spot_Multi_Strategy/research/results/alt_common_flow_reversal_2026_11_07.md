# Three-altcoin common-flow reversal: sealed data failure

Pre-registration: `690a7f5`. Code and synthetic tests were fixed and pushed before the formal read. The sole formal evaluation stopped at its predeclared complete-minute data Gate. It produced **no valid strategy performance statistics**.

The [official Binance XRPUSDT USD-M 1m ZIP for February 2022](https://data.binance.vision/data/futures/um/monthly/klines/XRPUSDT/1m/XRPUSDT-1m-2022-02.zip) matches both the pre-registered SHA-256 pin and the current official `.CHECKSUM` (`c2427ac93c91a589c62f10193b9672cd3e7cca23e219e444bc12c945d6a245e7`). Its first bar is 2022-02-01 00:00 UTC, its final bar is 2022-02-25 23:59 UTC, and it has 36,000 rows versus the required 40,320. The terminal 4,320 one-minute bars for February 26–28 are absent. The fixed ADAUSDT and DOGEUSDT archives for that month each have all 40,320 bars and pass their pinned and official hash checks.

The registration forbids interpolation, removing those days, substituting another symbol, or promoting a partial backtest. CAGR, Sharpe, drawdown, profit factor, annual walk-forward, market-state and Monte Carlo gates therefore remain **unevaluated**. This does not establish whether the flow-reversal idea has alpha.

The first download attempt encountered a local Python CA-bundle transport error before any performance output. The independently TLS-verified host `curl` retry reached the incomplete official archive. The original mature strategy baseline and existing sealed reports were not changed.
