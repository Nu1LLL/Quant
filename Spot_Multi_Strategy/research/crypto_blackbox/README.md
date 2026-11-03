# Crypto black-box research scaffold (paper only)

This isolated package turns the *trading-related ideas* in the supplied screenshots into testable contracts. It does **not** reproduce any third-party account, claim its displayed profits, collect a user's private photographs, connect to an exchange, or authorize live/copy trading. The user's protected mature strategy and earlier sealed experiments are untouched.

## Causal path

1. A licensed, timestamped crypto source supplies complete historical windows and immutable point-in-time features. Source and license identifiers are audit metadata, **not** proof of permission.
2. `preprocessing.py` (when supplied with complete trade windows) derives five-minute aggressor-buy share and a prior-window volume baseline; `signal_intake.py` labels a first alert, continuation, non-alert, or stale observation. `signal_store.py` keeps these small records in an optional append-only SQLite store and retrieves them only after their observation **and** label became visible. An alert is not an Alpha or an order. Post-alert appreciation is a future outcome and is prohibited as an input.
3. `archetypes.py` constructs two explicitly unvalidated hypothesis proposals: `flow` (short-horizon opposing flow/dislocation) and `rising_tide` (stronger crypto themes with capped layered entries). Both are crypto-only, time-stamped, and non-deployable. Its assumed dislocation capture is a cost screen, not a forecast.
4. `regime_context.py` records BTC MVRV/weekly averages and two US macro observations only when each has a separately archived historical vintage. It emits descriptive relationships, not buy/sell calls.
5. `matching.py` joins exact market, venue, symbol, direction, and holding horizon to **one** prior registered result. It requires prospective preregistration before the five-year independent OOS interval, five annual walk-forward folds, fees/slippage/funding, and 5,000 21-day Monte Carlo paths. Walk-forward and Monte Carlo success are separate *reported* flags whose numerical thresholds must be frozen by each experiment's preregistration; the generic matcher does not impose an invented universal 80% or 5% success fraction. It then ranks only positive cost-and-uncertainty-adjusted lower bounds. Reported flags still require a separate data/code audit. Multiple result records fail closed to prevent winner selection.
6. `shadow_portfolio.py` combines non-conflicting shadow candidates across strategies and coins with fixed single-symbol, cluster, and gross exposure caps. A portfolio assembled this way must itself receive an independent OOS test; good constituent reports do not validate the combination.
7. `risk_accounting.py` is a $10,000 USDT **paper** account for linear crypto perpetuals. It applies a discounted Kelly cap, separate core/tactical/reserve budgets, a hard leverage ceiling of 5x, a 20% drawdown stop on new entries, FIFO realized/unrealized PnL, fees, funding, latency diagnostics, and a cash-flow-adjusted 30-day report. It has no network or order-sending code. It does not model liquidation or guarantee a 20% loss ceiling; severe mark-to-market losses fail closed. Reported 30-day drawdown is absent unless the book has hourly full-position marks, and even then is only an hourly sampled drawdown.

The existing `microstructure_blackbox.py` contains experimental HMM and RBF machinery. No new screenshot-derived feature is fed into that model or deployed until its own source, lag, cost, walk-forward, leakage, and independent-OOS checks are preregistered. The package's modules are separable so that a failed source or strategy can be removed without changing the others.

## Evidence and source boundaries

- The displayed 30-day returns, win rates, and 10x/20x positions in the VergeX-style screenshots are unverified observations, not replicable performance. Leverage above 5x is outside this project.
- FlashAI-style first-report and continuation labels are rebuilt from allowed raw market records; this package does not scrape or integrate an unlicensed third-party alert service. A 5-minute volume multiple must use completed **prior** windows; a current 24-hour number or later price cannot leak into it.
- MVRV, weekly moving averages, yields, and M2 may be logged as descriptive regime context only when a genuine point-in-time source/vintage is available. The one-off market post in the screenshots is not a time series or a fact source for a backtest.
- The screenshots about video-channel growth and a personal body-composition report are separate from crypto trading. The latter contains private health information and is deliberately neither copied nor stored in this public repository.
- A passing synthetic test proves software contract behavior only. No live market archive, genuine 5-year OOS PnL, monthly profitability, or Alpha has been produced by this package. Any future formal hypothesis must be preregistered in a separate commit **before** results are read, following `../SCOPE.md`; it can only enter small paper-trading after all gates pass.

## Test

Run the isolated synthetic suite from the repository root:

```sh
PYTHONPATH=Spot_Multi_Strategy python3 -m unittest discover -s Spot_Multi_Strategy/tests -p 'test_crypto_blackbox_*.py'
```
