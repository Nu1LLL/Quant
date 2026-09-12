"""Run the preregistered independent Betting-Against-Beta OOS study
(USMV long / SPHB short, beta-neutral to SPY)."""
from pathlib import Path

import pandas as pd

from cross_asset_trend import load_adjusted_close
import bab_oos
import strict_validation


def run_experiment(low_beta, high_beta, market, simulations=5000):
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        net, detail = bab_oos.run_scenario(low_beta, high_beta, market, leverage)
        results[leverage] = {
            "returns": net,
            "detail": detail,
            "validation": strict_validation.evaluate_strict_oos(
                net, independent_oos=True, costs_included=True
            ),
        }

    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20110505
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
    output = root / "reports/mini_medallion_bab_oos"
    inputs = output / "inputs"

    low_beta = load_adjusted_close(
        "USMV", "2011-10-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    high_beta = load_adjusted_close(
        "SPHB", "2011-05-01", "2026-09-12", cache_folder=inputs, refresh=False
    )
    market = load_adjusted_close(
        "SPY", "2011-05-01", "2026-09-12", cache_folder=inputs, refresh=False
    )

    results, monte_carlo, regimes = run_experiment(low_beta, high_beta, market)

    low_beta.rename_axis("date").to_csv(output / "usmv_adjusted_close.csv")
    high_beta.rename_axis("date").to_csv(output / "sphb_adjusted_close.csv")
    market.rename_axis("date").to_csv(output / "spy_adjusted_close.csv")

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

    print("USMV rows", len(low_beta), low_beta.index.min(), low_beta.index.max())
    print("SPHB rows", len(high_beta), high_beta.index.min(), high_beta.index.max())
    print(summary[[
        "leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
        "oos__profit_factor", "check__walk_forward_passed", "strict_oos_passed"
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor"
    ]].to_string(index=False))
    detail_1x = results[1.0]["detail"]
    valid_beta = detail_1x.dropna(subset=["beta_low", "beta_high"])
    print("beta_low median/min/max:", valid_beta["beta_low"].median(),
          valid_beta["beta_low"].min(), valid_beta["beta_low"].max())
    print("beta_high median/min/max:", valid_beta["beta_high"].median(),
          valid_beta["beta_high"].min(), valid_beta["beta_high"].max())
    print("gross_exposure median/max:", valid_beta["gross_exposure"].median(),
          valid_beta["gross_exposure"].max())


if __name__ == "__main__":
    main()
