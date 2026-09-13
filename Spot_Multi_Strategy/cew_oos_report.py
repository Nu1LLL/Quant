"""Run the preregistered CEW emerging-currency ETF first-read OOS study."""
from pathlib import Path

import pandas as pd

import cew_oos
from cross_asset_trend import load_adjusted_close
import strict_validation


START = "2009-01-01"
END = "2026-09-13"
LATEST_ACCEPTABLE_START = pd.Timestamp("2009-06-30", tz="UTC")


def run_experiment(prices, simulations=5000):
    prices = cew_oos.validate_prices(prices)
    if (prices.index[-1] - prices.index[0]).days / 365.25 < 17.0:
        raise ValueError("CEW history must span at least seventeen years")
    inception_covered = prices.index[0] <= LATEST_ACCEPTABLE_START
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail = cew_oos.run_scenario(prices, leverage)
        validation = cew_oos.evaluate_seventeen_year_oos(returns)
        validation["checks"]["inception_covered"] = inception_covered
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns, "detail": detail, "validation": validation,
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20090506
    )
    periods = pd.Series([
        "2009_2012" if year <= 2012 else
        "2013_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    events = pd.Series([
        str(year) if year in (2013, 2018, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, periods).assign(family="period"),
        strict_validation.regime_metrics(primary, events).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_cew_oos"
    inputs = output / "inputs"
    prices = load_adjusted_close(
        "CEW", START, END, cache_folder=inputs, refresh=False
    )
    results, monte_carlo, regimes = run_experiment(prices)
    prices.rename_axis("date").to_csv(output / "adjusted_close.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        result["detail"].rename_axis("date").to_csv(output / f"daily_detail_{suffix}.csv")
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{suffix}.csv", index=False
        )
        rows.append({
            "leverage": leverage,
            **{f"oos__{key}": value for key, value in result["validation"]["metrics"].items()},
            **{f"check__{key}": value for key, value in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("DATA", len(prices), prices.index.min(), prices.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "check__inception_covered", "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
