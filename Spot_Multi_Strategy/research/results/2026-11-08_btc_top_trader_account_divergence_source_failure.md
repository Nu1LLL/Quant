# BTC top-trader account divergence: source failure, no return test

Preregistration `551165d` was committed and pushed before any selected account-ratio, BTC return, funding, signal, trade, or outcome was read. The fixed official-source audit then found an unavoidable completeness failure, so **no backtest, profitability figure, Gate metric, tuning, or paper recommendation was produced**.

- Required file: `https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2021-12-31.zip`.
- ZIP SHA-256 after verifying its official adjacent `.CHECKSUM`: `d01ca93fc0c4e721f9d0b87ae226a692c3d33e2483b365da00b2874f389e82df` (5,678 bytes).
- The sole CSV has 287 data rows. All **287/287** entries in `count_toptrader_long_short_ratio` and all **287/287** entries in `count_long_short_ratio` are empty. The registered requirement was at least 231 matched strictly positive 5-minute slots for every one of the 60 decision files.
- A HEAD-only audit had found that all 60 named archives exist; this demonstrates that file presence does **not** establish field coverage. The 2021-12-31 failure is sufficient to seal the experiment. Other source dates, prices, funding, signals and outcomes need not be examined.

Disposition: `source_integrity_fail`; `all_gate_passed=false`; `independent_executable_oos=false`. No year/month deletion, imputation, alternate source, threshold edit, or change of trading rule is permitted on this hypothesis. The protected baseline and prior sealed reports were not touched.
