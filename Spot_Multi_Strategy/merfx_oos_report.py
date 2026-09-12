"""Run the preregistered MERFX merger-arbitrage first-read OOS study."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import merfx_oos
import strict_validation


def run_experiment(prices, simulations=5000):
    prices = merfx_oos.validate_prices(prices)
    if (prices.index[-1] - prices.index[0]).days / 365.25 < 25.0:
        raise ValueError("MERFX history must span at least twenty-five years")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail = merfx_oos.run_scenario(prices, leverage)
        results[leverage] = {
            "returns": returns,
            "detail": detail,
            "validation": merfx_oos.evaluate_twenty_five_year_oos(returns),
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=22360679
    )
    period_label = pd.Series([
        "through_2007" if year <= 2007 else
        "2008_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        str(year) if year in (2008, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_merfx_oos"
    inputs = output / "inputs"
    prices = load_adjusted_close(
        "MERFX", "1988-01-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    results, monte_carlo, regimes = run_experiment(prices)
    prices.rename_axis("date").to_csv(output / "adjusted_close.csv")
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
    print("DATA", len(prices), prices.index.min(), prices.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
