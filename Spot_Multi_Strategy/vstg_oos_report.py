"""Run the preregistered Cboe VSTG volatility-premium study."""
from pathlib import Path

import pandas as pd

import strict_validation
import vstg_oos as strategy


# Frozen as unknown until the post-performance methodology audit.
ACCOUNT_EXECUTABLE = False
LIVE_HISTORY_AT_LEAST_10Y = False


def run_experiment(
    levels, simulations=5000, account_executable=False,
    live_history_at_least_10y=False,
):
    audit = strategy.coverage_audit(levels)
    if not audit["passed"]:
        raise ValueError(f"VSTG coverage gate failed: {audit}")
    results = {}
    for leverage in strategy.FROZEN_LEVERAGES:
        returns, detail = strategy.run_scenario(levels, leverage)
        validation = strategy.evaluate(
            returns, bool(detail["solvent"].all()), account_executable,
            live_history_at_least_10y,
        )
        results[leverage] = {
            "returns": returns,
            "detail": detail,
            "validation": validation,
        }
    primary = results[1.0]["returns"]
    monte_carlo = strategy.circular_block_monte_carlo_chunked(
        primary, simulations=simulations
    )
    periods = pd.Series(index=primary.index, dtype="object")
    periods.loc[primary.index.year <= 2012] = "inception_2012"
    periods.loc[(primary.index.year >= 2013) & (primary.index.year <= 2017)] = "2013_2017"
    periods.loc[(primary.index.year >= 2018) & (primary.index.year <= 2022)] = "2018_2022"
    periods.loc[primary.index.year >= 2023] = "2023_plus"
    events = pd.Series([
        str(year) if year in (2008, 2011, 2018, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, periods).assign(family="period"),
        strict_validation.regime_metrics(primary, events).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, audit


def leverage_label(leverage):
    return str(leverage).rstrip("0").rstrip(".").replace(".", "p")


def main(refresh=False):
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_vstg_oos"
    inputs = output / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    levels = strategy.load_levels(inputs, refresh=refresh)
    results, monte_carlo, regimes, audit = run_experiment(
        levels, account_executable=ACCOUNT_EXECUTABLE,
        live_history_at_least_10y=LIVE_HISTORY_AT_LEAST_10Y,
    )
    levels.rename_axis("date").to_csv(output / "vstg_levels.csv")
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    rows = []
    for leverage, result in results.items():
        label = leverage_label(leverage)
        result["detail"].rename_axis("date").to_csv(
            output / f"daily_detail_{label}x.csv"
        )
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{label}x.csv", index=False
        )
        rows.append({
            "leverage": leverage,
            **{f"oos__{k}": v for k, v in result["validation"]["metrics"].items()},
            **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
            "ending_equity": float(result["detail"]["equity"].iloc[-1]),
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("COVERAGE", audit)
    print(summary[[
        "leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
        "oos__profit_factor", "oos__worst_rolling_3y_sharpe", "ending_equity",
        "check__walk_forward_passed", "check__account_executable_for_10000",
        "check__live_history_at_least_10y", "strict_oos_passed",
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
