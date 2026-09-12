"""Run the preregistered Binance multi-asset basis carry OOS study."""
from pathlib import Path

import pandas as pd

import binance_basis_oos as basis
import strict_validation


SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT")
DOWNLOAD_START = "2020-09-01"
DOWNLOAD_END = "2026-09-11"
DEV_START = pd.Timestamp("2020-10-01", tz="UTC")
DEV_END = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")
OOS_START = pd.Timestamp("2021-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-09-10 23:59:59", tz="UTC")


def run_experiment(frames, monte_carlo_simulations=5000):
    intervals = basis.build_interval_components(frames)
    if intervals.index.min() > pd.Timestamp("2020-10-01", tz="UTC"):
        raise ValueError("All four assets must cover the preregistered start")
    expected_oos = pd.date_range(
        OOS_START, OOS_END.floor("8h"), freq="8h", tz="UTC"
    )
    missing_oos = expected_oos.difference(intervals.index)
    if len(missing_oos):
        raise ValueError(
            f"Incomplete preregistered OOS funding coverage: "
            f"{len(missing_oos)} missing, first={missing_oos[0]}"
        )
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        daily, detail = basis.run_notional(intervals, leverage)
        dev_returns = daily.loc[DEV_START:DEV_END]
        oos_returns = daily.loc[OOS_START:OOS_END]
        results[leverage] = {
            "daily": daily, "detail": detail,
            "dev": strict_validation.metrics_for_returns(dev_returns),
            "validation": strict_validation.evaluate_strict_oos(
                oos_returns, independent_oos=True, costs_included=True
            ),
        }
    primary_oos = results[1.0]["daily"].loc[OOS_START:OOS_END]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary_oos, simulations=monte_carlo_simulations,
        block_length=21, seed=14930352
    )
    labels = basis.diagnostic_regimes(intervals).reindex(primary_oos.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary_oos, labels["funding_regime"]).assign(family="funding"),
        strict_validation.regime_metrics(primary_oos, labels["btc_trend_regime"]).assign(family="btc_trend"),
    ], ignore_index=True)
    return intervals, results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_binance_basis_oos"
    cache = output / "inputs"
    frames = {
        symbol: basis.load_symbol(
            symbol, DOWNLOAD_START, DOWNLOAD_END, cache, refresh=True
        )
        for symbol in SYMBOLS
    }
    intervals, results, monte_carlo, regimes = run_experiment(frames)
    intervals.rename_axis("funding_time").to_csv(output / "common_interval_components.csv")
    summary_rows = []
    for leverage, result in results.items():
        result["daily"].rename_axis("date").to_csv(output / f"daily_returns_{int(leverage)}x.csv")
        result["detail"].rename_axis("funding_time").to_csv(output / f"interval_detail_{int(leverage)}x.csv")
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{int(leverage)}x.csv", index=False
        )
        summary_rows.append({
            "leverage": leverage,
            **{f"dev__{k}": v for k, v in result["dev"].items()},
            **{f"oos__{k}": v for k, v in result["validation"]["metrics"].items()},
            **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
