"""Run the preregistered six-coin low-turnover crypto OOS portfolio."""
from pathlib import Path

import numpy as np
import pandas as pd

import multicrypto_lowturnover_oos as strategy
import strict_validation


START, END = "2020-01-01", "2026-09-13"


def evaluate_oos(returns):
    result = strict_validation.evaluate_strict_oos(
        returns, independent_oos=True, costs_included=True, minimum_years=5.75
    )
    folds = result["walk_forward_folds"]
    result["checks"]["walk_forward_passed"] = bool(
        len(folds) >= 4 and not folds.empty and folds["fold_pass"].mean() >= 0.80
    )
    solvent = not pd.Series(returns).le(-1.0).any()
    result["checks"]["solvent"] = solvent
    if not solvent:
        result["metrics"].update(total_return=-1.0, final_value=0.0, cagr=-1.0)
    result["passed"] = all(result["checks"].values())
    return result


def sleeve_diagnostics(detail):
    columns = ["basis_contribution", "funding_contribution", "trend_contribution"]
    sleeves = detail[columns]
    covariance = sleeves.cov() * 1095.0
    portfolio_variance = float(covariance.to_numpy().sum())
    rows = []
    for column in columns:
        cov_portfolio = float(covariance.loc[column].sum())
        rows.append({
            "sleeve": column, "simple_return_sum": sleeves[column].sum(),
            "annualized_volatility": sleeves[column].std(ddof=1) * np.sqrt(1095.0),
            "variance_contribution": cov_portfolio / portfolio_variance
            if portfolio_variance > 0 else np.nan,
        })
    return pd.DataFrame(rows), sleeves.corr()


def run_experiment(panel, simulations=5000):
    audit = strategy.coverage_audit(panel)
    if not audit["passed"]:
        raise ValueError(f"Low-turnover multicrypto coverage failed: {audit}")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        daily, detail, positions, sleeves, funding_mean, momentum, volatility = \
            strategy.run_scenario(panel, leverage)
        results[leverage] = {
            "returns": daily, "detail": detail, "positions": positions,
            "sleeves": sleeves, "funding_mean": funding_mean,
            "momentum": momentum, "volatility": volatility,
            "validation": evaluate_oos(daily),
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20201101
    )
    periods = pd.Series(np.where(primary.index.year <= 2022, "2020_2022", "2023_plus"),
                        index=primary.index)
    events = pd.Series([str(year) if year in (2021, 2022, 2024) else "other_days"
                        for year in primary.index.year], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, periods).assign(family="period"),
        strict_validation.regime_metrics(primary, events).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, audit


def main(refresh=True):
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_multicrypto_lowturnover_oos"
    inputs = output / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    frames = {symbol: strategy.load_symbol(symbol, START, END, inputs, refresh=refresh)
              for symbol in strategy.SYMBOLS}
    panel = strategy.build_common_panel(frames)
    audit = strategy.coverage_audit(panel)
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    for symbol, frame in frames.items():
        frame.to_csv(output / f"{symbol}_aligned_input.csv", index=False)
    if not audit["passed"]:
        raise ValueError(f"Low-turnover multicrypto coverage failed before performance: {audit}")
    results, monte_carlo, regimes, _ = run_experiment(panel)
    panel.rename_axis("funding_time").to_csv(output / "common_panel.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        result["detail"].rename_axis("funding_time").to_csv(output / f"interval_detail_{suffix}.csv")
        result["positions"].rename_axis("funding_time").to_csv(output / f"positions_{suffix}.csv")
        result["validation"]["walk_forward_folds"].to_csv(output / f"walk_forward_{suffix}.csv", index=False)
        rows.append({"leverage": leverage,
                     **{f"oos__{key}": value for key, value in result["validation"]["metrics"].items()},
                     **{f"check__{key}": value for key, value in result["validation"]["checks"].items()},
                     "strict_oos_passed": result["validation"]["passed"]})
    summary = pd.DataFrame(rows)
    primary = results[1.0]["detail"]
    attribution, correlation = sleeve_diagnostics(primary)
    costs = pd.DataFrame([primary[["gross_return", "trading_cost", "financing_cost",
                                  "net_return"]].sum().to_dict()])
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    costs.to_csv(output / "cost_attribution_1x.csv", index=False)
    attribution.to_csv(output / "sleeve_attribution_1x.csv", index=False)
    correlation.to_csv(output / "sleeve_correlation_1x.csv")
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print("COVERAGE", audit)
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed", "check__solvent",
                   "strict_oos_passed"]].to_string(index=False))
    print("COST_ATTRIBUTION_1x", costs.iloc[0].to_dict())
    print("SLEEVE_ATTRIBUTION_1x\n", attribution.to_string(index=False))
    print("SLEEVE_CORRELATION_1x\n", correlation.to_string())
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
