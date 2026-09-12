"""Run the preregistered Binance four-coin funding-dispersion OOS study."""
from pathlib import Path

import pandas as pd

import binance_funding_dispersion_oos as strategy
import strict_validation


SYMBOLS = ("BCHUSDT", "LTCUSDT", "ADAUSDT", "LINKUSDT")
START = "2020-02-01"
END = "2026-09-12"


def run_experiment(panel, simulations=5000):
    audit = strategy.coverage_audit(panel)
    if not audit["passed"]:
        raise ValueError(f"Binance common-panel coverage gate failed: {audit}")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail, weights = strategy.run_notional(panel, leverage)
        validation = strict_validation.evaluate_strict_oos(
            returns, independent_oos=True, costs_included=True, minimum_years=6.0
        )
        validation["checks"]["common_panel_covered"] = True
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns, "detail": detail, "weights": weights,
            "validation": validation,
        }
    primary = results[1.0]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary["returns"], simulations=simulations, block_length=21, seed=20200201
    )
    labels, spread_median = strategy.diagnostic_labels(primary["detail"])
    regimes = pd.concat([
        strict_validation.regime_metrics(
            primary["returns"], labels["funding_spread_regime"]
        ).assign(family="funding_spread"),
        strict_validation.regime_metrics(
            primary["returns"], labels["calendar_year"]
        ).assign(family="calendar_year"),
    ], ignore_index=True)
    return results, monte_carlo, regimes, audit, spread_median


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_binance_funding_dispersion_oos"
    inputs = output / "inputs"
    frames = {
        symbol: strategy.load_symbol(symbol, START, END, inputs, refresh=False)
        for symbol in SYMBOLS
    }
    panel = strategy.build_common_panel(frames)
    audit = strategy.coverage_audit(panel)
    pd.DataFrame([audit]).to_csv(output / "coverage_audit.csv", index=False)
    if not audit["passed"]:
        print("COVERAGE_FAILURE", audit)
        raise SystemExit(2)
    results, monte_carlo, regimes, audit, spread_median = run_experiment(panel)
    panel.rename_axis("funding_time").to_csv(output / "common_panel.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        result["detail"].rename_axis("funding_time").to_csv(
            output / f"interval_detail_{suffix}.csv"
        )
        result["weights"].rename_axis("funding_time").to_csv(
            output / f"weights_{suffix}.csv"
        )
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
    print("COVERAGE", audit)
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print("SPREAD_MEDIAN", spread_median)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
