# Post-funding shock fade: sealed rejection

The rule was preregistered and pushed alone in 6f55f06; source loader,
simulation, and tests were pushed in c369433 before the formal Binance
price/funding archives were opened. The one formal run sealed
reports/mini_medallion_funding_settlement_fade_oos/RESULT.json (SHA-256
73d76284de44b5dd345d2a6a7668648e1472d21dda374d1edd7750c185b5a2b2).
Source verification passed: 122 frozen five-minute price ZIPs, 120 funding
ZIPs, and two bounded transport retries. The result stores compact receipts
and fills; raw archives were not retained.

The economic prediction failed. The first five-minute move after an actual
BTC/ETH funding settlement was faded for one hour, with positions flat at
settlement and the next funding event. Across 2021–2025, 1,082 settlement
events produced entries. After the frozen 5 bp taker fee and 5 bp adverse
slippage on each side, a 10,000 USDT account ended at **2,044.22 USDT**:
CAGR **−27.21%**, daily Sharpe **−2.131**, conservative maximum drawdown
**82.61%**, daily-dollar Profit Factor **0.622**. Even the zero-cost gross
scenario ended at **9,773.79 USDT**, so removing costs does not reveal a
profitable directional effect under this exact rule. Fees totaled
4,420.56 USDT in the path-dependent net simulation.

The fixed 2024–2025 historical holdout had CAGR **−26.02%**, Sharpe
**−2.580**, maximum drawdown **45.43%**, and Profit Factor **0.486**.
All five annual folds lost money. Only **9/60** months were profitable;
the longest nonprofitable streak was **17** months. The four regime Gate
failed, the one-bar delayed and double-slippage paths lost money, and
**0/5,000** circular 21-day Monte Carlo paths passed the four numeric targets
jointly.

The numerical Gate and the independent executable OOS Gate both failed.
Evidence grade remains preregistered_retrospective_settlement_inventory_fade;
the retrospective archive is not independently executable OOS. This design
is rejected and sealed. No same-history sign flip, threshold or hold-time
tuning, coin selection, leverage rescue, paper order, or real order is
authorized.
