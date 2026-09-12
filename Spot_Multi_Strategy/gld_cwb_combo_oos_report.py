"""Run the preregistered GLD+CWB fixed-equal-weight diversification
OOS study over their ~17.4-year common history, using dbv_oos's
15-year enhanced gate since this window is long enough to support it.
"""
from pathlib import Path

import pandas as pd

import dbv_oos
import gld_cwb_combo_oos as combo
import strict_validation


def run_experiment(gold_returns, cwb_returns, simulations=5000):
    combined, aligned = combo.combine_equal_weight(gold_returns, cwb_returns)
    if (combined.index[-1] - combined.index[0]).days / 365.25 < 15.0:
        raise ValueError("GLD/CWB combined history must span at least fifteen years")
    correlation = combo.pearson_correlation(gold_returns, cwb_returns)

    validation = dbv_oos.evaluate_fifteen_year_oos(combined)
    monte_carlo = strict_validation.circular_block_monte_carlo(
        combined, simulations=simulations, block_length=21, seed=20090416
    )

    period_label = pd.Series([
        "inception_2015" if year <= 2015 else
        "2016_2019" if year <= 2019 else "2020_plus"
        for year in combined.index.year
    ], index=combined.index)
    event_label = pd.Series([
        str(year) if year in (2018, 2020, 2022) else "other_days"
        for year in combined.index.year
    ], index=combined.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(combined, period_label).assign(family="period"),
        strict_validation.regime_metrics(combined, event_label).assign(family="event"),
    ], ignore_index=True)

    standalone_metrics = {
        "GLD": strict_validation.metrics_for_returns(aligned.iloc[:, 0]),
        "CWB": strict_validation.metrics_for_returns(aligned.iloc[:, 1]),
        "combined": validation["metrics"],
    }

    return validation, monte_carlo, regimes, correlation, standalone_metrics, aligned


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_gld_cwb_combo_oos"
    output.mkdir(parents=True, exist_ok=True)

    gold_detail = pd.read_csv(
        root / "reports/mini_medallion_gold_oos/daily_detail_1x.csv",
        parse_dates=["date"]
    ).set_index("date")
    cwb_detail = pd.read_csv(
        root / "reports/mini_medallion_convertible_bond_oos/daily_detail_1x.csv",
        parse_dates=["date"]
    ).set_index("date")

    gold_returns = gold_detail["net_return"].rename("GLD")
    cwb_returns = cwb_detail["net_return"].rename("CWB")

    (validation, monte_carlo, regimes, correlation,
     standalone_metrics, aligned) = run_experiment(gold_returns, cwb_returns)

    aligned.rename_axis("date").to_csv(output / "aligned_net_returns.csv")
    validation["walk_forward_folds"].to_csv(
        output / "walk_forward_combined.csv", index=False
    )
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_combined.csv", index=False)
    regimes.to_csv(output / "regime_metrics_combined.csv", index=False)

    summary_row = {
        "correlation_gld_cwb": correlation,
        **{f"check__{k}": v for k, v in validation["checks"].items()},
        "strict_oos_passed": validation["passed"],
        **{f"combined__{k}": v for k, v in standalone_metrics["combined"].items()},
        **{f"gld_standalone__{k}": v for k, v in standalone_metrics["GLD"].items()},
        **{f"cwb_standalone__{k}": v for k, v in standalone_metrics["CWB"].items()},
    }
    pd.DataFrame([summary_row]).to_csv(output / "strict_oos_summary.csv", index=False)

    print("Sample:", len(aligned), aligned.index.min(), aligned.index.max())
    print("Correlation(GLD, CWB) net daily returns:", correlation)
    print("Standalone GLD   CAGR={:.4f} Sharpe={:.4f} MaxDD={:.4f}".format(
        standalone_metrics["GLD"]["cagr"], standalone_metrics["GLD"]["sharpe_ratio"],
        standalone_metrics["GLD"]["max_drawdown"]
    ))
    print("Standalone CWB   CAGR={:.4f} Sharpe={:.4f} MaxDD={:.4f}".format(
        standalone_metrics["CWB"]["cagr"], standalone_metrics["CWB"]["sharpe_ratio"],
        standalone_metrics["CWB"]["max_drawdown"]
    ))
    print("Combined 50/50   CAGR={:.4f} Sharpe={:.4f} MaxDD={:.4f} PF={:.4f}".format(
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
