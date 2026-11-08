# CME BTC asset-manager positioning growth — sealed retrospective failure

**Experiment:** `cme_btc_asset_manager_growth_hash_v2_2026_11_08`
**Preregistration:** `fd077d3` plus hash-only correction `627cda7`
**Frozen code:** `76b5355`
**Formal data and metrics:** [`RESULT.json`](RESULT.json)
**Decision:** **REJECT**. `all_gate_passed=false`. No real orders were placed.

The original first version was separately sealed as [`DATA_FAILURE`](../mini_medallion_cme_asset_manager_growth_oos/DATA_FAILURE.json) before any economic value or price was read. Its 2023 source digest had one missing character. Version 2 corrected only that exact digest; its hypothesis, report lag, signal sign, account, dates, costs, stress tests and Gate were unchanged and committed before the one formal data read.

## Frozen account outcome

Binance BTCUSDT spot or cash, 10,000 USDT initial capital, at most 1x exposure, 2021-01-01 through 2025-12-31 (1,826 days). CFTC CME standard Bitcoin futures asset-manager net/OI growth generated 36 buys. Each changed target was filled after the frozen 13-calendar-day report lag, with 10 bp taker fee plus 5 bp adverse slippage per fill. The source check passed all six pinned CFTC ZIP hashes, 313 unique weekly report dates and 62 pinned Binance daily archives.

| Measure | Sealed result | Required |
|---|---:|---:|
| Final account | 9,549.43 USDT | >10,000 USDT |
| Net CAGR | **−0.92%** | ≥40% |
| Daily Sharpe | **0.176** | >1.5 |
| Conservative maximum drawdown | **73.94%** | ≤20% |
| Daily-dollar profit factor | **0.995** | >1.3 |
| Profitable months | **23/60** | ≥36/60 |
| Longest nonpositive monthly streak | **5** | ≤3 |
| 2024–2025 historical holdout CAGR / Sharpe / drawdown / PF | 10.12% / 0.468 / 30.08% / 1.066 | All four numeric thresholds |
| 5,000-path 21-day block Monte Carlo joint passes | **0** | ≥80% |

Annual fixed-rule folds were +15.81%, −58.86%, +65.26%, +14.90% and +5.52% for 2021–2025 respectively; the negative 2022 fold fails annual walk-forward. The four-state check also fails (high-vol bull has 88 rather than 90 required days, and two states have negative net dollar PnL). Gross zero-cost final equity was 10,638.61 USDT; the costed account ended below initial capital. Double-slippage final equity was 9,211.78 USDT. The frozen Tuesday execution stress ended at 10,161.60 USDT, but this alternate schedule is a diagnostic, not a replacement strategy. The same-cost BTC buy-and-hold comparator ended at 30,212.55 USDT, with a 77.57% drawdown and Sharpe 0.670; it does not satisfy the Gate either.

Even if historical metrics had passed, the current-vintage CFTC archives have no preserved first-release snapshots, so independent executable OOS remains **false**. The result is permanently rejected. No same-history signal reversal, parameter adjustment, year deletion, asset selection or leverage rescue is allowed.
