"""Run the preregistered GOOG/GOOGL share-class spread OOS study."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import goog_share_class_oos as strategy
import strict_validation


START = "2014-01-01"
END = "2026-09-13"


def run_experiment(prices, simulations=5000):
    prices = strategy.validate_prices(prices)
    span_years = (prices.index[-1] - prices.index[0]).days / 365.25
    if span_years < 12.0:
        raise ValueError("Alphabet common panel must span at least twelve years")
    inception_covered = prices.index[0] <= strategy.LATEST_ACCEPTABLE_START
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail, weights = strategy.run_scenario(prices, leverage)
        validation = strict_validation.evaluate_strict_oos(
            returns, independent_oos=True, costs_included=True, minimum_years=12.0
        )
        validation["checks"]["inception_covered"] = inception_covered
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns, "detail": detail, "weights": weights,
            "validation": validation,
        }
    primary = results[1.0]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary["returns"], simulations=simulations, block_length=21, seed=20140403
    )
    period = pd.Series([
        "2014_2019" if year <= 2019 else
        "2020_2022" if year <= 2022 else "2023_plus"
        for year in primary["returns"].index.year
    ], index=primary["returns"].index)
    year = pd.Series(
        primary["returns"].index.year.astype(str), index=primary["returns"].index
    )
    regimes = pd.concat([
        strict_validation.regime_metrics(primary["returns"], period).assign(family="period"),
        strict_validation.regime_metrics(primary["returns"], year).assign(family="calendar_year"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_goog_share_class_oos"
    inputs = output / "inputs"
    prices = pd.concat({
        symbol: load_adjusted_close(
            symbol, START, END, cache_folder=inputs, refresh=True
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
    primary = results[1.0]["detail"]
    transitions = primary["executed_state"].diff().fillna(0.0).ne(0.0)
    diagnostic = pd.DataFrame([{
        "active_day_ratio": primary["executed_state"].ne(0.0).mean(),
        "long_goog_days": int(primary["executed_state"].gt(0.0).sum()),
        "long_googl_days": int(primary["executed_state"].lt(0.0).sum()),
        "state_transitions": int(transitions.sum()),
        "gross_return_sum": primary["gross_return"].sum(),
        "trading_cost_sum": primary["trading_cost"].sum(),
        "short_borrow_sum": primary["short_borrow"].sum(),
    }])
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    diagnostic.to_csv(output / "position_diagnostics_1x.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("DATA", len(prices), prices.index.min(), prices.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "check__inception_covered", "strict_oos_passed"]].to_string(index=False))
    print("POSITION_DIAGNOSTICS_1x", diagnostic.iloc[0].to_dict())
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
