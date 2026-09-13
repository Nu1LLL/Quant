"""Run the preregistered Cboe CNDR/BFLY blend OOS study."""
from pathlib import Path

import pandas as pd

import cboe_iron_blend_oos as strategy
import strict_validation


def evaluate(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=30.0
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 20 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    result["checks"]["worst_rolling_3y_sharpe_at_least_1"] = bool(
        result["metrics"]["worst_rolling_3y_sharpe"] >= 1.0
    )
    solvent = not pd.Series(returns).le(-1.0).any()
    result["checks"]["solvent"] = solvent
    if not solvent:
        result["metrics"].update(total_return=-1.0, final_value=0.0, cagr=-1.0)
    result["passed"] = all(result["checks"].values())
    return result


def run_experiment(levels, simulations=5000):
    audit = strategy.coverage_audit(levels)
    if not audit["passed"]:
        raise ValueError(f"Cboe iron blend coverage failed: {audit}")
    base, turnover = strategy.blend_returns(levels)
    results = {}
    for leverage in (1.0, 2.0, 3.0, 4.0, 5.0):
        returns = strategy.apply_leverage(base, leverage)
        results[leverage] = {"returns": returns, "validation": evaluate(returns)}
    monte_carlo = strict_validation.circular_block_monte_carlo(
        results[1.0]["returns"], simulations=simulations, block_length=63, seed=19870101
    )
    periods = pd.Series(index=base.index, dtype="object")
    periods.loc[base.index.year <= 1999] = "pre_2000"
    periods.loc[(base.index.year >= 2000) & (base.index.year <= 2009)] = "2000_2009"
    periods.loc[(base.index.year >= 2010) & (base.index.year <= 2019)] = "2010_2019"
    periods.loc[base.index.year >= 2020] = "2020_plus"
    events = pd.Series([str(y) if y in (2008, 2020, 2022) else "other_days"
                        for y in base.index.year], index=base.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(base, periods).assign(family="period"),
        strict_validation.regime_metrics(base, events).assign(family="event"),
    ], ignore_index=True)
    sleeve_returns = levels.loc[:, strategy.SYMBOLS].pct_change(fill_method=None).fillna(0.0)
    return results, turnover, monte_carlo, regimes, sleeve_returns, audit


def main(refresh=True):
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_cboe_iron_blend_oos"
    cache = output / "inputs"
    cache.mkdir(parents=True, exist_ok=True)
    levels = strategy.load_levels(cache, refresh=refresh)
    audit = strategy.coverage_audit(levels)
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    if not audit["passed"]:
        raise ValueError(f"Cboe iron blend coverage failed before performance: {audit}")
    results, turnover, monte_carlo, regimes, sleeves, _ = run_experiment(levels)
    levels.rename_axis("open_time").to_csv(output / "cboe_levels.csv")
    turnover.rename_axis("open_time").to_csv(output / "turnover.csv")
    sleeves.rename_axis("open_time").to_csv(output / "sleeve_returns.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        pd.DataFrame({"net_return": result["returns"],
                      "equity": 10000.0 * (1.0 + result["returns"]).cumprod()}).to_csv(
            output / f"daily_{suffix}.csv"
        )
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{suffix}.csv", index=False
        )
        rows.append({"leverage": leverage,
                     **{f"oos__{k}": v for k, v in result["validation"]["metrics"].items()},
                     **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
                     "strict_oos_passed": result["validation"]["passed"]})
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("COVERAGE", audit)
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "oos__positive_complete_year_ratio",
                   "oos__worst_rolling_3y_sharpe", "check__walk_forward_passed",
                   "check__solvent", "strict_oos_passed"]].to_string(index=False))
    print("SLEEVE SIMPLE RETURNS", sleeves.sum().to_dict())
    print("SLEEVE CORRELATION\n", sleeves.corr().to_string())
    print("TURNOVER", float(turnover.sum()))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
