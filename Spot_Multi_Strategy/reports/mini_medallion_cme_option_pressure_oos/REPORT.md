# CME Bitcoin option nonmomentum pressure — sealed failure

**Experiment:** `cme_btc_option_nonmomentum_pressure_2026_11_08`
**Preregistration:** `f900f63`
**Frozen implementation:** `94da651`
**Formal data and full Gate:** [`RESULT.json`](RESULT.json)
**Decision:** **REJECT**; `all_gate_passed=false`. No real orders were placed.

One frozen BTCUSDT spot/cash rule tested whether the return-neutralized change in CME asset-manager option exposure predicted later BTC demand. The option exposure was reconstructed as futures-and-options combined minus futures-only positions; its 26-report rolling regression used only older reports. The decision used a Wednesday BTC spot open after an eight-day minimum report lag and the [CFTC's documented 2023 and 2025 delayed-release exceptions](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalSpecialAnnouncements/index.htm). It did **not** rerun or tune the earlier invalid futures-only result. The current-vintage archives still lack full first-release snapshots, so the evidence remains retrospective and `independent_executable_oos=false`.

The source audit passed six pinned CFTC futures-only ZIPs, six matching combined ZIPs, 313 joined weekly report dates, and 68 pinned Binance BTC spot daily ZIPs. The formal account held at most 1x BTC, started at 10,000 USDT, covered all 1,826 days in 2021–2025, executed 58 buys, and paid the frozen 10 bp taker fee plus 5 bp adverse slippage on each fill.

| Measure | Sealed result | Required |
|---|---:|---:|
| Final account | 8,482.26 USDT | >10,000 USDT |
| Net CAGR | **−3.24%** | ≥40% |
| Daily Sharpe | **0.080** | >1.5 |
| Conservative max drawdown | **71.47%** | ≤20% |
| Daily-dollar profit factor | **0.986** | >1.3 |
| Profitable months | **23/60** | ≥36/60 |
| Longest nonpositive monthly streak | **6** | ≤3 |
| 2024–2025 historical holdout CAGR / Sharpe / drawdown / PF | −22.97% / −0.690 / 55.86% / 0.870 | All four numeric thresholds |
| 5,000-path 21-day block Monte Carlo joint passes | **0** | ≥80% |

The fixed annual folds were +97.07%, −44.31%, +30.34%, +0.19% and −40.82% for 2021–2025. The 2021 result alone had 36.76% drawdown and cannot validate the five-year account. The four-state Gate failed, including negative net-dollar PnL in low-volatility bear days and only 88 high-volatility bull days. The zero-cost same-signal account ended at 10,094.36 USDT; costed execution fell below the initial 10,000. Double-slippage ended at 8,004.34 USDT. The frozen Thursday-latency stress ended at 10,375.01 USDT, but it is a diagnostic, not a selectable alternative. Same-cost BTC buy-and-hold ended at 30,212.55 USDT with a 77.57% drawdown and Sharpe 0.670; it also fails the full Gate.

The failed rule is sealed. No re-signing, execution-day switch, asset/date selection, additional leverage or model-window tuning on this same history is permitted. The 2026 paper motivating the mechanism overlaps this historical sample, and no independent executable OOS has been produced.
