"""Run the preregistered independent currency-hedge-premium OOS study
(HEFA long / EFA short), using the standard 5-year strict_validation
gate since HEFA's ~12.6-year history doesn't clear the 15-year
enhanced threshold."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import currency_hedge_premium_oos as ch
import strict_validation


def run_experiment(hedged, unhedged, simulations=5000):
    if (hedged.index[-1] - hedged.index[0]).days / 365.25 < 5.0:
        raise ValueError("HEFA/EFA history must span at least five years")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        net, detail = ch.run_scenario(hedged, unhedged, leverage)
        results[leverage] = {
            "returns": net,
            "detail": detail,
            "validation": strict_validation.evaluate_strict_oos(
                net, independent_oos=True, costs_included=True
            ),
        }

    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20140214
    )

    period_label = pd.Series([
        "inception_2017" if year <= 2017 else
        "2018_2021" if year <= 2021 else "2022_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        str(year) if year in (2014, 2017, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)

    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_currency_hedge_premium_oos"
    inputs = output / "inputs"

    hedged = load_adjusted_close(
        "HEFA", "2014-02-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    unhedged = load_adjusted_close(
        "EFA", "2014-02-01", "2026-09-12", cache_folder=inputs, refresh=False
    )

    results, monte_carlo, regimes = run_experiment(hedged, unhedged)

    hedged.rename_axis("date").to_csv(output / "hefa_adjusted_close.csv")
    unhedged.rename_axis("date").to_csv(output / "efa_adjusted_close.csv")

    rows = []
    for leverage, result in results.items():
        result["detail"].rename_axis("date").to_csv(
            output / f"daily_detail_{int(leverage)}x.csv"
        )
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{int(leverage)}x.csv", index=False
        )
        rows.append({
            "leverage": leverage,
            **{f"oos__{k}": v for k, v in result["validation"]["metrics"].items()},
            **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)

    print("HEFA rows", len(hedged), hedged.index.min(), hedged.index.max())
    print("EFA rows", len(unhedged), unhedged.index.min(), unhedged.index.max())
    print(summary[[
        "leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
        "oos__profit_factor", "check__walk_forward_passed", "strict_oos_passed"
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
