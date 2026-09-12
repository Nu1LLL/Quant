"""Run the preregistered independent momentum-factor OOS study
(MTUM long / SPY short)."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import momentum_factor_oos as mf
import strict_validation


def run_experiment(momentum, market, simulations=5000):
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        net, detail = mf.run_scenario(momentum, market, leverage)
        results[leverage] = {
            "returns": net,
            "detail": detail,
            "validation": strict_validation.evaluate_strict_oos(
                net, independent_oos=True, costs_included=True
            ),
        }

    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20130418
    )

    period_label = pd.Series([
        "inception_2016" if year <= 2016 else
        "2017_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        str(year) if year in (2018, 2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)

    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_momentum_factor_oos"
    inputs = output / "inputs"

    momentum = load_adjusted_close(
        "MTUM", "2013-04-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    market = load_adjusted_close(
        "SPY", "2013-04-01", "2026-09-12", cache_folder=inputs, refresh=False
    )

    results, monte_carlo, regimes = run_experiment(momentum, market)

    momentum.rename_axis("date").to_csv(output / "momentum_adjusted_close.csv")
    market.rename_axis("date").to_csv(output / "market_adjusted_close.csv")

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

    print("MTUM rows", len(momentum), momentum.index.min(), momentum.index.max())
    print("SPY rows", len(market), market.index.min(), market.index.max())
    print(summary[[
        "leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
        "oos__profit_factor", "check__walk_forward_passed", "strict_oos_passed"
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
