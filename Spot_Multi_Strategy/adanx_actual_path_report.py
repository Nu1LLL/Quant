"""Run the preregistered ADANX diversified-arbitrage path study."""
from pathlib import Path

import pandas as pd

import adanx_actual_path
from cross_asset_trend import load_adjusted_close
import strict_validation


START = "2009-01-01"
END = "2026-09-15"
LATEST_START = pd.Timestamp("2009-02-28", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")
# Frozen as unknown until the preregistered post-performance prospectus check.
ACCOUNT_EXECUTABLE = False


def coverage_checks(prices):
    prices = adanx_actual_path.validate_prices(prices)
    weekdays = pd.date_range(prices.index[0], prices.index[-1], freq="B", tz="UTC")
    coverage = len(prices.index.normalize().unique()) / len(weekdays)
    return {
        "start_by_2009_02_28": prices.index[0] <= LATEST_START,
        "end_by_2026_08_31": prices.index[-1] >= EARLIEST_END,
        "span_at_least_17_5y": (
            (prices.index[-1] - prices.index[0]).days / 365.25 >= 17.5
        ),
        "weekday_coverage_at_least_94pct": coverage >= 0.94,
    }, coverage


def run_experiment(prices, simulations=5000, account_executable=False):
    prices = adanx_actual_path.validate_prices(prices)
    coverage, coverage_ratio = coverage_checks(prices)
    if not all(coverage.values()):
        failed = [name for name, passed in coverage.items() if not passed]
        raise ValueError(f"ADANX coverage gate failed: {', '.join(failed)}")
    results = {}
    for leverage in adanx_actual_path.FROZEN_LEVERAGES:
        returns, detail = adanx_actual_path.run_scenario(prices, leverage)
        validation = adanx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all()), account_executable
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
        primary, simulations=simulations, block_length=63, seed=20090115
    )
    period = pd.Series([
        "inception_2013" if year <= 2013 else
        "2014_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event = pd.Series([
        str(year) if year in (2009, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period).assign(family="period"),
        strict_validation.regime_metrics(primary, event).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, coverage_ratio


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_adanx_actual_path"
    inputs = output / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    prices = load_adjusted_close(
        "ADANX", START, END, cache_folder=inputs, refresh=False
    )
    results, monte_carlo, regimes, coverage_ratio = run_experiment(
        prices, account_executable=ACCOUNT_EXECUTABLE
    )
    prices.rename_axis("date").to_csv(output / "adjusted_nav.csv")
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
        "strict_gate_passed",
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
