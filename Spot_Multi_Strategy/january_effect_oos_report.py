"""Run the preregistered independent January-effect OOS study
(IWM long / SPY short, January only), using DBV's 15-year enhanced
gate since IWM/SPY's ~26.3-year common history supports it."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import dbv_oos
import january_effect_oos as je
import strict_validation


def run_experiment(small_cap, large_cap, simulations=5000):
    if (small_cap.index[-1] - small_cap.index[0]).days / 365.25 < 15.0:
        raise ValueError("IWM/SPY history must span at least fifteen years")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        net, detail = je.run_scenario(small_cap, large_cap, leverage)
        results[leverage] = {
            "returns": net,
            "detail": detail,
            "validation": dbv_oos.evaluate_fifteen_year_oos(net),
        }

    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20000526
    )

    period_label = pd.Series([
        "inception_2008" if year <= 2008 else
        "2009_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        str(year) if year in (2009, 2021) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)

    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_january_effect_oos"
    inputs = output / "inputs"

    small_cap = load_adjusted_close(
        "IWM", "2000-05-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    large_cap = load_adjusted_close(
        "SPY", "2000-05-01", "2026-09-12", cache_folder=inputs, refresh=False
    )

    results, monte_carlo, regimes = run_experiment(small_cap, large_cap)

    small_cap.rename_axis("date").to_csv(output / "iwm_adjusted_close.csv")
    large_cap.rename_axis("date").to_csv(output / "spy_adjusted_close.csv")

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

    print("IWM rows", len(small_cap), small_cap.index.min(), small_cap.index.max())
    print("SPY rows", len(large_cap), large_cap.index.min(), large_cap.index.max())
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
