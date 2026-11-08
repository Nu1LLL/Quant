# Correction: the FRED real-yield experiment repeated a prior mechanism

The sealed [FRED real-yield result](../../reports/mini_medallion_us_real_yield_crypto_oos/REPORT.md)
is numerically unchanged. Its preregistration was committed before its data
were opened, but its claim of being a **new economically independent
hypothesis was wrong**. A complete tracked-document screen found the earlier
[`tips_real_yield_crypto_2026_11_08`](../../reports/mini_medallion_tips_real_yield_crypto_oos/PREREGISTRATION.md)
preregistration (commit `6b35980`) and [sealed result](../../reports/mini_medallion_tips_real_yield_crypto_oos/RESULT.md).

Both rules use the **direction of the US ten-year TIPS real yield over exactly
20 valid observations** to predict the next week's common crypto exposure.
Both posit that a falling real discount rate favors crypto. The earlier rule
used official US Treasury data and five equal-weight USD-M perpetuals with
both long and short states; the later one used the FRED DFII10 series and a
BTC/ETH long-or-cash spot basket. Different provider, execution instrument,
coin basket and cash treatment do **not** create a new economic source of
Alpha. The first test sealed a maintenance-margin risk failure in April 2021;
the later spot test sealed numerical rejection (CAGR 15.62%, Sharpe 0.550,
maximum drawdown 58.18%, PF 1.059, negative 2024–2025 holdout). Neither
supports deployment, and neither provides genuine independent executable OOS.

The later run is retained as a transparent **implementation variant within
the already-tested real-yield family**, not counted as a new independent Alpha
family or evidence that source diversification was achieved. Its result must
not be rerun, reweighted or used for winner selection. The error arose because
the novelty search looked only in `research/preregistrations`, omitting the
older `reports/*/PREREGISTRATION.md` corpus. The new read-only
`novelty_inventory.py` searches both tracked corpora plus sealed result
summaries; the real-yield alias screen now finds the earlier registration.
The screen is a collision aid, not semantic proof of novelty.
