"""Run the preregistered FXA/FXB/FXC relative-momentum OOS study."""
from pathlib import Path

import numpy as np
import pandas as pd

from cross_asset_trend import load_adjusted_close
import g10_fx_momentum_oos as strategy
import strict_validation


START = "2006-01-01"
END = "2026-09-12"


def run_experiment(prices, simulations=5000):
    prices = strategy.validate_prices(prices)
    span_years = (prices.index[-1] - prices.index[0]).days / 365.25
    inception_covered = prices.index[0] <= strategy.LATEST_ACCEPTABLE_START
    if span_years < 15.0:
        raise ValueError("Currency price panel must span at least fifteen years")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail, weights, scores = strategy.run_scenario(prices, leverage)
        validation = strict_validation.evaluate_strict_oos(
            returns, independent_oos=True, costs_included=True, minimum_years=15.0
        )
        validation["checks"]["inception_covered"] = inception_covered
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns, "detail": detail, "weights": weights,
            "scores": scores, "validation": validation,
        }
    primary = results[1.0]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary["returns"], simulations=simulations, block_length=21, seed=20070131
    )
    labels = strategy.diagnostic_labels(prices, primary["returns"])
    years = labels["calendar_year"].astype(int)
    period = pd.Series(np.select(
        [years <= 2009, years <= 2014, years <= 2019, years <= 2022],
        ["2008_2009", "2010_2014", "2015_2019", "2020_2022"],
        default="2023_plus",
    ), index=labels.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary["returns"], period).assign(family="period"),
        strict_validation.regime_metrics(
            primary["returns"], labels["usd_regime"]
        ).assign(family="usd_regime"),
        strict_validation.regime_metrics(
            primary["returns"], labels["calendar_year"]
        ).assign(family="calendar_year"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_g10_fx_momentum_oos"
    inputs = output / "inputs"
    prices = pd.concat({
        symbol: load_adjusted_close(
            symbol, START, END, cache_folder=inputs, refresh=False
        ) for symbol in strategy.SYMBOLS
    }, axis=1, join="inner")
    prices.columns = list(strategy.SYMBOLS)
    results, monte_carlo, regimes = run_experiment(prices)
    prices.rename_axis("date").to_csv(output / "adjusted_close.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        result["detail"].rename_axis("date").to_csv(output / f"daily_detail_{suffix}.csv")
        result["weights"].rename_axis("date").to_csv(output / f"weights_{suffix}.csv")
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{suffix}.csv", index=False
        )
        rows.append({
            "leverage": leverage,
            **{f"oos__{key}": value for key, value in result["validation"]["metrics"].items()},
            **{f"check__{key}": value for key, value in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    results[1.0]["scores"].rename_axis("signal_date").to_csv(output / "monthly_scores.csv")
    pd.DataFrame(rows).to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    summary = pd.DataFrame(rows)
    print("DATA", len(prices), prices.index.min(), prices.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "check__inception_covered", "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
