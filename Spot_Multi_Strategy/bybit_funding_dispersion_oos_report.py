"""Run the preregistered Bybit cross-sectional funding-dispersion OOS study."""
from pathlib import Path

import pandas as pd

import bybit_funding_dispersion_oos as strategy
import strict_validation


SYMBOLS = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "EOSUSDT")
DOWNLOAD_START = "2020-09-01"
DOWNLOAD_END = "2026-09-11"
DEV_START = pd.Timestamp("2020-10-01", tz="UTC")
DEV_END = pd.Timestamp("2020-12-31 23:59:59", tz="UTC")
OOS_START = pd.Timestamp("2021-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-09-10 23:59:59", tz="UTC")


def run_experiment(frames, simulations=5000):
    panel = strategy.build_common_panel(frames)
    expected = pd.date_range(OOS_START, OOS_END.floor("8h"), freq="8h", tz="UTC")
    missing = expected.difference(panel.index)
    if len(missing):
        raise ValueError(
            f"Incomplete preregistered Bybit OOS grid: {len(missing)} missing, first={missing[0]}"
        )
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        daily, detail, weights = strategy.run_notional(panel, leverage)
        dev = daily.loc[DEV_START:DEV_END]
        oos = daily.loc[OOS_START:OOS_END]
        results[leverage] = {
            "daily": daily,
            "detail": detail,
            "weights": weights,
            "dev": strict_validation.metrics_for_returns(dev),
            "validation": strict_validation.evaluate_strict_oos(
                oos, independent_oos=True, costs_included=True
            ),
        }
    primary_oos = results[1.0]["daily"].loc[OOS_START:OOS_END]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary_oos, simulations=simulations, block_length=21, seed=26457513
    )
    labels, spread_median = strategy.diagnostic_labels(
        results[1.0]["detail"], OOS_START, OOS_END
    )
    regimes = pd.concat([
        strict_validation.regime_metrics(
            primary_oos, labels["funding_spread_regime"]
        ).assign(family="funding_spread"),
        strict_validation.regime_metrics(
            primary_oos, labels["calendar_year"]
        ).assign(family="calendar_year"),
    ], ignore_index=True)
    return panel, results, monte_carlo, regimes, spread_median


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_bybit_funding_dispersion_oos"
    cache = output / "inputs"
    frames = {
        symbol: strategy.load_symbol(
            symbol, DOWNLOAD_START, DOWNLOAD_END, cache, refresh=False
        )
        for symbol in SYMBOLS
    }
    panel, results, monte_carlo, regimes, spread_median = run_experiment(frames)
    panel.rename_axis("funding_time").to_csv(output / "common_panel.csv")
    rows = []
    for leverage, result in results.items():
        result["daily"].rename_axis("date").to_csv(
            output / f"daily_returns_{int(leverage)}x.csv"
        )
        result["detail"].rename_axis("funding_time").to_csv(
            output / f"interval_detail_{int(leverage)}x.csv"
        )
        result["weights"].rename_axis("funding_time").to_csv(
            output / f"weights_{int(leverage)}x.csv"
        )
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{int(leverage)}x.csv", index=False
        )
        rows.append({
            "leverage": leverage,
            **{f"dev__{key}": value for key, value in result["dev"].items()},
            **{f"oos__{key}": value for key, value in result["validation"]["metrics"].items()},
            **{f"check__{key}": value for key, value in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    pd.DataFrame([{"lagged_funding_spread_median": spread_median}]).to_csv(
        output / "diagnostic_threshold.csv", index=False
    )
    print("DATA", len(panel), panel.index.min(), panel.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print("SPREAD_MEDIAN", spread_median)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
