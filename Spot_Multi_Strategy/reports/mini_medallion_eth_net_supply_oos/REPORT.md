# Sealed rejection: ETH ledger-supply contraction

**Verdict: REJECT; no paper or live order.** One preregistered formal replay for `eth_net_supply_contraction_2026_11_08`, 2021-10-01 through 2026-10-01. Frozen [preregistration and price manifest](../../research/preregistrations/2026-11-08_eth_net_supply_contraction.md) were committed/pushed at `01c09fd`; code and synthetic checks at `b7e8e6a`. [RESULT.json](RESULT.json) is the machine-readable one-shot seal, SHA-256 `bc7b80d02c1841d44b1255d8192bfaaed1118540a2d33be3b7c42e0acead66e3`.

## Result

The rule bought Binance ETHUSDT spot only when [Coin Metrics `SplyCur`](https://docs.coinmetrics.io/network-data/network-data-overview/supply/current-supply) had declined over the preceding 28 completed days, evaluated on Sunday and traded Thursday. The 10,000 USDT account used no leverage, 10 bp taker fee and 5 bp adverse slippage each fill, and a 30-day capacity proxy. The current-vintage supply response passed its pinned 1,918-day SHA/grid/schema checks; 63 Binance monthly official ZIPs passed their frozen SHA and checksum checks. October 2026 contributed only its first opening price for terminal settlement. No raw data archives or actual orders were stored/executed.

| Costed account | CAGR | Daily Sharpe | Conservative max drawdown | Daily-dollar PF | Final USDT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen ETH supply rule, five years | **6.02%** | **0.347** | **36.10%** | **1.073** | **13,396.09** |
| Historical final two folds, 2024-10..2026-09 | 0.00% | undefined | 0.00% | undefined | 13,396.09 from 13,396.09 |
| Same-cost ETH spot buy-and-hold | −2.25% | 0.311 | 81.89% | 0.994 | 8,924.72 |

Only **three buys and six fills** occurred. Fees totaled 57.70 USDT and reference-price slippage drag 28.85 USDT. Zero-cost same-signal final equity was 13,517.13 USDT; doubled-slippage and one-day-latency final equity were 13,355.93 and 13,303.28 USDT, respectively. These positive stress endpoints do not establish alpha. Fixed Oct–Sep folds Y1..Y5 returned **0.00%, −11.74%, +51.64%, 0.00%, 0.00%**. Y1, Y4 and Y5 had no fills; the final two years were entirely cash.

## Gate audit

- **Full numerical Gate failed:** 6.02% CAGR versus ≥40%; 0.347 Sharpe versus >1.5; 36.10% drawdown versus ≤20%; 1.073 PF versus >1.3.
- **Trade sample and annual walk-forward failed:** 3 buys versus ≥20; Y1, Y4 and Y5 had zero fills and zero return; Y2 lost money. The Y4–Y5 historical holdout was flat with undefined Sharpe and PF, so it failed the four numeric thresholds.
- **Monthly stability failed:** 12/60 strictly positive months versus ≥36; longest nonpositive run **30 months** versus ≤3. The worst month returned −17.69%.
- **Market-state Gate failed:** bull/high-volatility and bear/high-volatility states lost 2,516.78 and 1,439.67 USDT, although all four states had ≥90 observations.
- **Block Monte Carlo failed:** **0/5,000** circular 21-day paths jointly met CAGR, Sharpe, drawdown and PF requirements; median path drawdown was 42.61%. No path ruined or had undefined metrics.
- **Independent executable OOS failed:** the supply series is today's reconstruction, without first-published vintages. The two-year temporal holdout cannot establish historical availability at each Thursday. `SplyCur` measures currently visible native units; it does not isolate base-fee burning from issuance or other supply accounting. [Ethereum's issuance/burn description](https://ethereum.org/roadmap/merge/issuance/) motivates the mechanism, not this rule's profitability.

The rule, direction, 28-day window and five-year sample are now spent. No same-history retuning, shorter window, alternative asset or leverage adjustment will be used to promote it. Future work must test an economically independent preregistered mechanism or collect genuine first-seen supply vintages prospectively.
