# Sealed result: delayed crypto security losses as a BTC risk filter

**Verdict: REJECT. No live or paper order.** Frozen experiment `crypto_hack_loss_btc_risk_filter_2026_11_08`; preregistration `561cd36`, implementation `f76b569`, one formal replay sealed 2026-11-08 09:37 UTC. Machine-readable [RESULT.json](RESULT.json), SHA-256 `d4ea10e6edd96a7251114c1340e6a1ffc605e86e9243d225ee6a421d762d8ad9`. No data or code failure was retried; the rule, asset and 2021–2025 history are now spent for selection.

## Frozen account and data

The rule held Binance spot BTCUSDT or cash from 2021-01-01 through 2026-01-01 with 10,000 USDT initial equity, no leverage, 10 bp taker fee and 5 bp adverse slippage per fill. Each Monday it summed [DefiLlama hack-catalog](https://api.llama.fi/hacks) reported losses for the 30 incident dates ending eight days earlier; ≥100 million USD meant cash. The source returned the exact pinned 892-row 2020-11 through 2025-12 projection (`9fa5a45e806bf8718fab0c7e007f35311ed6280c92e69dae4e38344ebd812f8f`), and 62 official Binance ZIP hashes passed. The [preregistration](../../research/preregistrations/2026-11-08_crypto_security_shock_btc.md) fixed all thresholds and costs before the replay.

| Net account | CAGR | Sharpe | Max drawdown | Daily-dollar PF | Terminal USDT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen hack filter, 2021–2025 | 57.13% | 1.497 | 27.64% | 1.258 | 95,768.81 |
| Historical 2024–2025 holdout | 26.15% | 0.895 | 22.67% | 1.156 | 95,768.81 from 60,163.29 |
| Same-cost BTC buy-and-hold | 24.75% | 0.670 | 77.57% | 1.061 | 30,212.55 |

The base account made 28 buys and 56 fills including terminal liquidation; fees totaled 3,338.54 USDT and reference-price slippage drag 1,669.29 USDT. Zero-cost same-signal terminal equity was 104,161.19 USDT. Double-slippage terminal equity was 93,125.17; one-day delayed execution was 70,969.45. Both stress endpoints stayed positive but latency changed the result materially. The strategy's 2021–2025 annual returns were +186.98%, +26.89%, +65.41%, +47.64%, and +7.74%; all five annual folds were positive with an executed fill, but that does not make them independent trials.

## Gate audit

- **Numerical full account failed:** CAGR met ≥40% and was within the 40–60% target band, but Sharpe **1.497 ≤1.5**, drawdown **27.64% >20%**, and PF **1.258 ≤1.3**.
- **Historical 2024–2025 holdout failed all four numeric thresholds**: 26.15% CAGR, 0.895 Sharpe, 22.67% drawdown and 1.156 PF.
- **Monthly stability failed:** 32/60 strictly profitable months versus 36 required; longest nonpositive streak 5 versus 3 allowed. Worst month was 2025-11 at −14.13%.
- **Regime test failed:** bull/high-volatility state had 88 days, below the predeclared 90-day minimum. The other three states met their 90-day sample minimum and all four had positive aggregate account PnL, but this does not cure the missing state coverage.
- **Block Monte Carlo failed:** 8/5,000 circular 21-day paths jointly met the four numerical thresholds, **0.16%** versus the required ≥80%. Median bootstrapped drawdown was 32.48%; no path ruined or had undefined metrics.
- **Independent executable OOS failed by design:** the current-vintage event list has incident dates and revised reported loss amounts, but no first-publication timestamp or immutable historical vintage. A seven-day delay is only a proxy and cannot prevent later backfills from entering historical decisions. The 2024–2025 window is a retrospective holdout, not a genuinely untouched future live sample.

The larger retrospective terminal return is an observation, **not a deployable strategy**. No threshold, lag, window, asset or leverage will be adjusted against these same years. Any later experiment needs a new independent economic mechanism and preregistration, or a separately timestamped prospective security-event feed for genuine OOS.
