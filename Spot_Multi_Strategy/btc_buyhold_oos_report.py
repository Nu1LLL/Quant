"""Run the preregistered BTCUSDT buy-and-hold first-read OOS study.
Reuses qai_oos.validate_prices/run_scenario (same single-fund
leverage/financing model as QAI/DBV/SRRIX/MERFX/DBMF/GLD/WTMF) and
the standard 5-year strict_validation gate (Binance spot history only
goes back to 2017-08-17, ~9.1 years — doesn't support the 15-year
enhanced variant).
"""
from pathlib import Path

import pandas as pd

import strict_validation
from data import load_or_download_klines
from qai_oos import run_scenario, validate_prices

FEE_PLUS_SLIPPAGE = 0.0010 + 0.0005


def run_experiment(prices, simulations=5000):
    prices = validate_prices(prices)
    if (prices.index[-1] - prices.index[0]).days / 365.25 < 5.0:
        raise ValueError("BTCUSDT history must span at least five years")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail = run_scenario(prices, leverage, leg_cost=FEE_PLUS_SLIPPAGE)
        results[leverage] = {
            "returns": returns,
            "detail": detail,
            "validation": strict_validation.evaluate_strict_oos(
                returns, independent_oos=True, costs_included=True
            ),
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20170817
    )
    period_label = pd.Series([
        "inception_2019" if year <= 2019 else
        "2020_2022" if year <= 2022 else "2023_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        str(year) if year in (2018, 2022, 2025) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_btc_buyhold_oos"
    output.mkdir(parents=True, exist_ok=True)

    klines = load_or_download_klines(
        "BTCUSDT", "1d", "2017-08-17", "2026-09-12",
        cache_folder=root / "data_cache"
    )
    prices = klines.set_index("open_time")["close"]

    results, monte_carlo, regimes = run_experiment(prices)
    prices.rename_axis("date").to_csv(output / "close_prices.csv")

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
