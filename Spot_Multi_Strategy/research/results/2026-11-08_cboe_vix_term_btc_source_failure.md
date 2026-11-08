# Cboe VIX term structure to BTC: source failure, no return test

The one fixed hypothesis was preregistered and pushed as `b6526ac` before any selected Cboe level, BTC return, funding, signal, trade, performance or Gate result was read. The formal source audit stopped before simulation.

Official Binance USD-M monthly mark-price file `https://data.binance.vision/data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2021-07.zip` passed its adjacent official `.CHECKSUM` validation. Its ZIP SHA-256 is `569b41a00797933a3e2dd9ceaed3e2d4f7bc0526ce2dd06d1190835732958ca7` (19,625 bytes). The CSV contains **624** distinct hourly rows versus **744** required for July 2021, with **120 missing hours**, beginning 2021-07-01 00:00 UTC; no extra hours. The preregistration explicitly requires a complete 1h mark grid for funding valuation and does not authorize daily-archive repair, interpolation, dropping July, or price substitution.

**Disposition:** `source_integrity_fail`; `all_gate_passed=false`; `independent_executable_oos=false`. No strategy return, Sharpe, CAGR, drawdown, profit factor, walk-forward or Monte Carlo was computed. A later BTC/VIX-term idea on the same 2021–2025 history must not be presented as an independent retry. The protected mature strategy and prior sealed results were untouched.
