"""Run the preregistered DBMF+GLD fixed-equal-weight diversification
OOS study. Loads the already-computed, already-cost-adjusted 1x
net-return series from the standalone DBMF and GLD experiments and
combines them 50/50 with no additional cost and no correlation-driven
weight optimization."""
from pathlib import Path

import pandas as pd

import dbmf_gold_combo_oos as combo
import strict_validation


def run_experiment(dbmf_returns, gold_returns, simulations=5000):
    combined, aligned = combo.combine_equal_weight(dbmf_returns, gold_returns)
    correlation = combo.pearson_correlation(dbmf_returns, gold_returns)

    validation = strict_validation.evaluate_strict_oos(
        combined, independent_oos=True, costs_included=True
    )
    monte_carlo = strict_validation.circular_block_monte_carlo(
        combined, simulations=simulations, block_length=21, seed=20190508
    )

    period_label = pd.Series([
        "inception_2020" if year <= 2020 else
        "2021_2022" if year <= 2022 else "2023_plus"
        for year in combined.index.year
    ], index=combined.index)
    event_label = pd.Series([
        str(year) if year in (2020, 2022) else "other_days"
        for year in combined.index.year
    ], index=combined.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(combined, period_label).assign(family="period"),
        strict_validation.regime_metrics(combined, event_label).assign(family="event"),
    ], ignore_index=True)

    standalone_metrics = {
        "DBMF": strict_validation.metrics_for_returns(aligned["DBMF"]),
        "GLD": strict_validation.metrics_for_returns(aligned["GLD"]),
        "combined": validation["metrics"],
    }

    return validation, monte_carlo, regimes, correlation, standalone_metrics, aligned


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_dbmf_gold_combo_oos"
    output.mkdir(parents=True, exist_ok=True)

    dbmf_detail = pd.read_csv(
        root / "reports/mini_medallion_dbmf_oos/daily_detail_1x.csv",
        parse_dates=["date"]
    ).set_index("date")
    gold_detail = pd.read_csv(
        root / "reports/mini_medallion_gold_oos/daily_detail_1x.csv",
        parse_dates=["date"]
    ).set_index("date")

    dbmf_returns = dbmf_detail["net_return"]
    gold_returns = gold_detail["net_return"]

    (validation, monte_carlo, regimes, correlation,
     standalone_metrics, aligned) = run_experiment(dbmf_returns, gold_returns)

    aligned.rename_axis("date").to_csv(output / "aligned_net_returns.csv")
    validation["walk_forward_folds"].to_csv(
        output / "walk_forward_combined.csv", index=False
    )
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_combined.csv", index=False)
    regimes.to_csv(output / "regime_metrics_combined.csv", index=False)

    summary_row = {
        "correlation_dbmf_gold": correlation,
        **{f"check__{k}": v for k, v in validation["checks"].items()},
        "strict_oos_passed": validation["passed"],
        **{f"combined__{k}": v for k, v in standalone_metrics["combined"].items()},
        **{f"dbmf_standalone__{k}": v for k, v in standalone_metrics["DBMF"].items()},
        **{f"gld_standalone__{k}": v for k, v in standalone_metrics["GLD"].items()},
    }
    pd.DataFrame([summary_row]).to_csv(output / "strict_oos_summary.csv", index=False)

    print("Sample:", len(aligned), aligned.index.min(), aligned.index.max())
    print("Correlation(DBMF, GLD) net daily returns:", correlation)
    print("Standalone DBMF   CAGR={:.4f} Sharpe={:.4f} MaxDD={:.4f}".format(
        standalone_metrics["DBMF"]["cagr"], standalone_metrics["DBMF"]["sharpe_ratio"],
        standalone_metrics["DBMF"]["max_drawdown"]
    ))
    print("Standalone GLD    CAGR={:.4f} Sharpe={:.4f} MaxDD={:.4f}".format(
        standalone_metrics["GLD"]["cagr"], standalone_metrics["GLD"]["sharpe_ratio"],
        standalone_metrics["GLD"]["max_drawdown"]
    ))
    print("Combined 50/50    CAGR={:.4f} Sharpe={:.4f} MaxDD={:.4f} PF={:.4f}".format(
        validation["metrics"]["cagr"], validation["metrics"]["sharpe_ratio"],
        validation["metrics"]["max_drawdown"], validation["metrics"]["profit_factor"]
    ))
    print("checks:", validation["checks"])
    print("strict_oos_passed:", validation["passed"])
    print("MONTE_CARLO", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
