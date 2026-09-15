"""Run the preregistered ARBFX merger-arbitrage path study."""
from pathlib import Path

import pandas as pd

import arbfx_actual_path
from cross_asset_trend import load_adjusted_close
import strict_validation


START = "2000-01-01"
END = "2026-09-16"
LATEST_START = pd.Timestamp("2003-12-31", tz="UTC")
EARLIEST_END = pd.Timestamp("2026-08-31", tz="UTC")
# Resolved only after the first performance run from official documents.
ACCOUNT_EXECUTABLE = True
STRATEGY_CONTINUITY = True
ENTRY_SALES_LOAD = 0.0
EXIT_SALES_LOAD = 0.0


def coverage_checks(prices):
    prices = arbfx_actual_path.validate_prices(prices)
    weekdays = pd.date_range(prices.index[0], prices.index[-1], freq="B", tz="UTC")
    coverage = len(prices.index.normalize().unique()) / len(weekdays)
    return {
        "start_by_2003_12_31": prices.index[0] <= LATEST_START,
        "end_by_2026_08_31": prices.index[-1] >= EARLIEST_END,
        "span_at_least_15y": (
            (prices.index[-1] - prices.index[0]).days / 365.25 >= 15.0
        ),
        "weekday_coverage_at_least_94pct": coverage >= 0.94,
    }, coverage


def run_experiment(
    prices, simulations=5000, account_executable=False,
    strategy_continuity=False, entry_sales_load=0.0, exit_sales_load=0.0,
):
    prices = arbfx_actual_path.validate_prices(prices)
    coverage, coverage_ratio = coverage_checks(prices)
    if not all(coverage.values()):
        failed = [name for name, passed in coverage.items() if not passed]
        raise ValueError(f"ARBFX coverage gate failed: {', '.join(failed)}")
    quality = arbfx_actual_path.nav_quality(prices)
    results = {}
    for leverage in arbfx_actual_path.FROZEN_LEVERAGES:
        returns, detail = arbfx_actual_path.run_scenario(
            prices, leverage, entry_sales_load=entry_sales_load,
            exit_sales_load=exit_sales_load,
        )
        validation = arbfx_actual_path.evaluate_actual_path(
            returns, bool(detail["solvent"].all()), account_executable,
            strategy_continuity, quality["unsmoothed_daily_nav"],
        )
        validation["checks"].update(coverage)
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns,
            "detail": detail,
            "validation": validation,
        }
    primary = results[1.0]["returns"]
    monte_carlo = arbfx_actual_path.circular_block_monte_carlo_chunked(
        primary, simulations=simulations, block_length=63, seed=20000917
    )
    period = pd.Series([
        "inception_2007" if year <= 2007 else
        "2008_2009" if year <= 2009 else
        "2010_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event = pd.Series([
        str(year) if year in (2001, 2002, 2008, 2020, 2022, 2025)
        else "other_days" for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period).assign(family="period"),
        strict_validation.regime_metrics(primary, event).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, coverage_ratio, quality


def leverage_label(leverage):
    return str(leverage).rstrip("0").rstrip(".").replace(".", "p")


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_arbfx_actual_path"
    inputs = output / "inputs"
    output.mkdir(parents=True, exist_ok=True)
    prices = load_adjusted_close(
        "ARBFX", START, END, cache_folder=inputs, refresh=False
    )
    results, monte_carlo, regimes, coverage_ratio, quality = run_experiment(
        prices, account_executable=ACCOUNT_EXECUTABLE,
        strategy_continuity=STRATEGY_CONTINUITY,
        entry_sales_load=ENTRY_SALES_LOAD, exit_sales_load=EXIT_SALES_LOAD,
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
            "entry_sales_load": ENTRY_SALES_LOAD,
            "exit_sales_load": EXIT_SALES_LOAD,
            "strict_gate_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_path_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    pd.DataFrame([quality]).to_csv(output / "nav_quality.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("DATA", len(prices), prices.index.min(), prices.index.max(), coverage_ratio)
    print("NAV_QUALITY", quality)
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
