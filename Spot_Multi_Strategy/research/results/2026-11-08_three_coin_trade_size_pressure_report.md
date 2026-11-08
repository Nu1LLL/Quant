# Three-coin abnormal trade size: sealed failure

**Precommitted experiment:** `08a63cd` was pushed alone before this rule's average-trade-size rankings, signals or returns were inspected. Code `d1d6160` fixed the single formal implementation. The earlier program had already processed some of the **same Binance price/volume history for other hypotheses**; this is signal-specific preregistration, **not** a wholly untouched market-data sample or a five-year point-in-time executable OOS. The 2025 [financial-uncertainty/trade-size paper](https://www.efmaefm.org/0EFMAMEETINGS/EFMA%20ANNUAL%20MEETINGS/2025-Greece/papers/Financial_Uncertainty___Names1.pdf) motivates a low-prior transfer but does not validate this weekly rule or identify actual investors from Binance K-line trade counts.

The one-shot reader verified official checksums and frozen pins for **210** five-minute price ZIPs and **201** funding ZIPs, 613,440 bars per coin for `ADAUSDT`, `BTCUSDT`, `ETHUSDT`. No raw ZIP/CSV was stored. Of **290** fixed Tuesday decisions, **159** produced an active long/short pair and all 159 passed the preregistered prior-volume capacity check. The formal account used 2× gross (1× equity per leg), 5-bp taker fee plus 5-bp adverse slippage **per fill**, actual archived funding with a trade-open notional proxy, and five-minute adverse equity marks. Seven positions were force-closed after the 3× gross/equity threshold; none was used to reselect a rule.

| Full account, 2021-03-09–2026-09-30 | Fixed result | Gate |
|---|---:|---:|
| Ending equity from $10,000 | **$629.76** | — |
| Net CAGR | **−39.17%** | ≥40% |
| Daily Sharpe | **−0.673** | >1.5 |
| Conservative max drawdown | **96.35%** | ≤20% |
| Daily-dollar profit factor | **0.776** | >1.3 |
| Positive months | **28/67** | ≥41/67 |
| Longest nonpositive-month streak | **9** | ≤3 |
| 5,000-path 21-day block Monte Carlo joint pass | **0%** | ≥80% |

The required **1× gross signal-quality comparator also failed**: $3,027.75 ending equity, CAGR −19.33%, Sharpe −0.624, drawdown 77.84%, PF 0.821. The 2× account's price PnL **−$7,751.21**, fees **−$784.08**, slippage **−$784.08** and funding **−$50.87** reconcile to the $629.76 ending balance. The fixed 2025–2026 retrospective holdout had CAGR **1.96%**, Sharpe **0.290**, drawdown **58.32%** and PF **1.009**. Only 2023 and 2025 annual folds were positive; the other four folds failed. All four market-state net PnLs were negative, and the bull/high-volatility state had only 35 days against 90 required. Doubled-slippage, delayed-fill and no-favorable-funding stresses ended at **$478.84**, **$662.84** and **$449.49**.

**Disposition:** `sealed_reject`, `all_gate_passed=false`, `independent_executable_oos=false`. Do not reverse the sign, select 2023/2025 alone, change assets/threshold/holding time or rescue the low Sharpe with leverage on this sample. No paper or live trade follows. The original mature strategy and prior sealed results were untouched. Full machine-readable result: `three_coin_trade_size_pressure_2026_11_08.json`.
