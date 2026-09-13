"""Run the preregistered ten-country ETF 12-1 momentum OOS study."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import country_momentum_oos as strategy
import strict_validation


START = "1996-01-01"
END = "2026-09-13"


def evaluate_long_oos(returns):
    validation = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=30.0
    )
    folds = validation["walk_forward_folds"]
    validation["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 28 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    validation["passed"] = all(validation["checks"].values())
    return validation


def run_experiment(prices, simulations=5000):
    prices = strategy.validate_prices(prices)
    if (prices.index[-1] - prices.index[0]).days / 365.25 < 30.0:
        raise ValueError("Country common panel must span at least thirty years")
    inception_covered = prices.index[0] <= strategy.LATEST_ACCEPTABLE_START
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail, weights, score, targets = strategy.run_scenario(prices, leverage)
        validation = evaluate_long_oos(returns)
        validation["checks"]["inception_covered"] = inception_covered
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns, "detail": detail, "weights": weights,
            "score": score, "targets": targets, "validation": validation,
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=19960318
    )
    periods = pd.Series([
        "1996_2005" if year <= 2005 else
        "2006_2015" if year <= 2015 else "2016_plus"
        for year in primary.index.year
    ], index=primary.index)
    events = pd.Series([
        str(year) if year in (1997, 2000, 2008, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, periods).assign(family="period"),
        strict_validation.regime_metrics(primary, events).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_country_momentum_oos"
    inputs = output / "inputs"
    prices = pd.concat({
        symbol: load_adjusted_close(symbol, START, END, cache_folder=inputs, refresh=False)
        for symbol in strategy.SYMBOLS
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
    summary = pd.DataFrame(rows)
    primary = results[1.0]
    holding = pd.DataFrame({
        "long_days": primary["weights"].gt(0).sum(),
        "short_days": primary["weights"].lt(0).sum(),
    }).rename_axis("symbol").reset_index()
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    holding.to_csv(output / "holding_diagnostics_1x.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("DATA", len(prices), prices.index.min(), prices.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "check__inception_covered", "strict_oos_passed"]].to_string(index=False))
    print("COST_ATTRIBUTION_1x", primary["detail"][["gross_return", "trading_cost",
          "short_borrow", "net_return"]].sum().to_dict())
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))
    print(holding.to_string(index=False))


if __name__ == "__main__":
    main()
