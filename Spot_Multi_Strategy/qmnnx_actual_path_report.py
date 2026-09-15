"""Run the preregistered QMNNX equity-market-neutral path study."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import qmnnx_actual_path
import strict_validation


START = "2014-01-01"
END = "2026-09-15"
LATEST_START = pd.Timestamp("2015-12-31", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")
# The May 20, 2026 SEC supplement closes the fund to ordinary new investors
# from June 19, 2026. Existing holders and listed exceptions may still buy.
ACCOUNT_EXECUTABLE = False
# No material strategy discontinuity was disclosed for the observed lifetime.
STRATEGY_CONTINUITY = True


def coverage_checks(prices):
    prices = qmnnx_actual_path.validate_prices(prices)
    weekdays = pd.date_range(prices.index[0], prices.index[-1], freq="B", tz="UTC")
    coverage = len(prices.index.normalize().unique()) / len(weekdays)
    return {
        "start_by_2015_12_31": prices.index[0] <= LATEST_START,
        "end_by_2026_08_31": prices.index[-1] >= EARLIEST_END,
        "span_at_least_10_5y": (
            (prices.index[-1] - prices.index[0]).days / 365.25 >= 10.5
        ),
        "weekday_coverage_at_least_94pct": coverage >= 0.94,
    }, coverage


def run_experiment(
    prices, simulations=5000, account_executable=False,
    strategy_continuity=False,
):
    prices = qmnnx_actual_path.validate_prices(prices)
    coverage, coverage_ratio = coverage_checks(prices)
    if not all(coverage.values()):
        failed = [name for name, passed in coverage.items() if not passed]
        raise ValueError(f"QMNNX coverage gate failed: {', '.join(failed)}")
    results = {}
    for leverage in qmnnx_actual_path.FROZEN_LEVERAGES:
        returns, detail = qmnnx_actual_path.run_scenario(prices, leverage)
        validation = qmnnx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all()), account_executable,
            strategy_continuity,
        )
        validation["checks"].update(coverage)
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns,
            "detail": detail,
            "validation": validation,
        }

    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=63, seed=20141001
    )
    period = pd.Series([
        "inception_2018" if year <= 2018 else
        "2019_2021" if year <= 2021 else "2022_plus"
        for year in primary.index.year
    ], index=primary.index)
    event = pd.Series([
        str(year) if year in (2018, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period).assign(family="period"),
        strict_validation.regime_metrics(primary, event).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, coverage_ratio


def leverage_label(leverage):
    return str(leverage).rstrip("0").rstrip(".").replace(".", "p")


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_qmnnx_actual_path"
    inputs = output / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    prices = load_adjusted_close(
        "QMNNX", START, END, cache_folder=inputs, refresh=False
    )
    results, monte_carlo, regimes, coverage_ratio = run_experiment(
        prices, account_executable=ACCOUNT_EXECUTABLE,
        strategy_continuity=STRATEGY_CONTINUITY,
    )
    prices.rename_axis("date").to_csv(output / "adjusted_nav.csv")
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
            **{f"path__{k}": v for k, v in result["validation"]["metrics"].items()},
            **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
            "ending_equity": float(result["detail"]["equity"].iloc[-1]),
            "total_external_cost_pct_initial": float(
                result["detail"]["external_cost"].sum()
            ),
            "total_financing_cost_pct_initial": float(
                result["detail"]["financing_cost"].sum()
            ),
            "strict_gate_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_path_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("DATA", len(prices), prices.index.min(), prices.index.max(), coverage_ratio)
    print(summary[[
        "leverage", "path__cagr", "path__sharpe_ratio", "path__max_drawdown",
        "path__profit_factor", "path__worst_rolling_3y_sharpe", "ending_equity",
        "check__walk_forward_passed", "check__account_executable_for_10000",
        "check__same_strategy_for_full_sample", "strict_gate_passed",
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
