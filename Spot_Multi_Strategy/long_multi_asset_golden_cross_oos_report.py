"""Run preregistered long-history four-asset golden-cross replication."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import long_multi_asset_golden_cross_oos as strategy
import strict_validation


START, END = "2007-01-01", "2026-09-15"


def evaluate(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=18.0
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 12 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    result["checks"]["worst_rolling_3y_sharpe_at_least_1"] = bool(
        result["metrics"]["worst_rolling_3y_sharpe"] >= 1.0
    )
    result["checks"]["solvent"] = not pd.Series(returns).le(-1.0).any()
    result["passed"] = all(result["checks"].values())
    return result


def run_experiment(prices, simulations=5000):
    audit = strategy.coverage_audit(prices)
    if not audit["passed"]:
        raise ValueError(f"Long multi-asset coverage failed: {audit}")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        combined, sleeves, details = strategy.run_portfolio(prices, leverage)
        results[leverage] = {"returns": combined, "sleeves": sleeves,
                             "details": details, "validation": evaluate(combined)}
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=63, seed=20071217
    )
    periods = pd.Series(index=primary.index, dtype="object")
    periods.loc[primary.index.year <= 2012] = "2008_2012"
    periods.loc[(primary.index.year >= 2013) & (primary.index.year <= 2019)] = "2013_2019"
    periods.loc[primary.index.year >= 2020] = "2020_plus"
    events = pd.Series([str(y) if y in (2008, 2020, 2022) else "other_days"
                        for y in primary.index.year], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, periods).assign(family="period"),
        strict_validation.regime_metrics(primary, events).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, audit


def main(refresh=True):
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_long_multi_asset_golden_cross_oos"
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    raw = {symbol: load_adjusted_close(symbol, START, END, inputs, refresh=refresh)
           for symbol in strategy.SYMBOLS}
    prices = strategy.align_prices(raw)
    audit = strategy.coverage_audit(prices)
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    if not audit["passed"]:
        raise ValueError(f"Long multi-asset coverage failed before performance: {audit}")
    results, monte_carlo, regimes, _ = run_experiment(prices)
    prices.rename_axis("date").to_csv(output / "aligned_prices.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        pd.DataFrame({"net_return": result["returns"],
                      "equity": 10000.0 * (1.0 + result["returns"]).cumprod()}).to_csv(
            output / f"daily_{suffix}.csv"
        )
        result["sleeves"].rename_axis("date").to_csv(output / f"sleeve_returns_{suffix}.csv")
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{suffix}.csv", index=False
        )
        rows.append({"leverage": leverage,
                     **{f"oos__{k}": v for k, v in result["validation"]["metrics"].items()},
                     **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
                     "strict_oos_passed": result["validation"]["passed"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    results[1.0]["sleeves"].corr().to_csv(output / "sleeve_correlation_1x.csv")
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("COVERAGE", audit)
    print(summary[["leverage", "oos__final_value", "oos__cagr", "oos__sharpe_ratio",
                   "oos__max_drawdown", "oos__profit_factor",
                   "oos__positive_complete_year_ratio", "oos__worst_rolling_3y_sharpe",
                   "check__walk_forward_passed", "strict_oos_passed"]].to_string(index=False))
    print("SLEEVE METRICS")
    for symbol in strategy.SYMBOLS:
        metrics = strict_validation.metrics_for_returns(results[1.0]["sleeves"][symbol])
        print(symbol, {k: metrics[k] for k in ("cagr", "sharpe_ratio", "max_drawdown", "profit_factor")})
    print("CORRELATION\n", results[1.0]["sleeves"].corr().to_string())
    print("MONTE_CARLO", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
