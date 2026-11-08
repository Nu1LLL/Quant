# Deribit BTC option IV term structure: sealed source failure

**Decision: REJECT / DATA_FAILURE.** No strategy signal return, account PnL, Sharpe, drawdown, Profit Factor, walk-forward or Monte Carlo result was computed, and no real or paper order was placed.

- Frozen preregistration was committed and pushed as `7a31ecc`; the source, simulator and one-shot runner were separately committed and pushed as `5ba1448` before any formal IV or spot result was read. The worktree and remote were clean at launch. Synthetic source classification and the reused spot-account timing tests passed (12 relevant tests).
- The single formal run stopped during Deribit source validation of the **first Monday, 2021-01-04**. At least one trade in that hour failed the preregistered finite, positive strike/index/amount/IV requirement. The exact failed condition was `invalid option strike/index/amount/IV`; the source processor did not continue to Binance prices or account simulation. Machine-readable seal: [`DATA_FAILURE.json`](DATA_FAILURE.json). No partial weeks were substituted or dropped.
- The hypothesis required a complete 261-week source, fixed 2021–2025 account, costed BTC spot fills, annual folds, a historical 2024–2025 holdout, four regimes, monthly stability and 5,000 block Monte Carlo paths. None can be claimed from this incomplete source. In any event, the current-vintage historical tape could not prove independent executable OOS.

The frozen IV rule is closed on this history. The broad archive reader remains a generic future-source utility, but this specific experiment will not be repaired, rerun with IV omissions removed, moved to a later start date, or counted as an Alpha. The overall Mini-Medallion Gate remains unpassed and no live trading is authorized.
