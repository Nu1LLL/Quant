"""Run the preregistered independent developed international equity
OOS study (EFA long / SPY short), using DBV's 15-year enhanced gate
since EFA/SPY's ~25-year common history supports it."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import dbv_oos
import developed_intl_equity_oos as di
import strict_validation


def run_experiment(international, domestic, simulations=5000):
    if (international.index[-1] - international.index[0]).days / 365.25 < 15.0:
        raise ValueError("EFA/SPY history must span at least fifteen years")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        net, detail = di.run_scenario(international, domestic, leverage)
        results[leverage] = {
            "returns": net,
            "detail": detail,
            "validation": dbv_oos.evaluate_fifteen_year_oos(net),
        }

    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20010827
    )

    period_label = pd.Series([
        "inception_2008" if year <= 2008 else
        "2009_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        str(year) if year in (2008, 2011, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)

    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_developed_intl_equity_oos"
    inputs = output / "inputs"

    international = load_adjusted_close(
        "EFA", "2001-08-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    domestic = load_adjusted_close(
        "SPY", "2001-08-01", "2026-09-12", cache_folder=inputs, refresh=False
    )

    results, monte_carlo, regimes = run_experiment(international, domestic)

    international.rename_axis("date").to_csv(output / "efa_adjusted_close.csv")
    domestic.rename_axis("date").to_csv(output / "spy_adjusted_close.csv")

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

    print("EFA rows", len(international), international.index.min(), international.index.max())
    print("SPY rows", len(domestic), domestic.index.min(), domestic.index.max())
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
