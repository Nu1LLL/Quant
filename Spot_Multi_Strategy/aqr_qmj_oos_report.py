"""Run the preregistered AQR global QMJ post-publication OOS study."""
from pathlib import Path

import pandas as pd

import aqr_qmj_oos as study
import aqr_tsmom_oos as validation


OOS_START = pd.Timestamp("2013-09-01")


def run_experiment(frame, simulations=5000):
    raw = frame.loc[frame.index >= OOS_START, "QMJ_GLOBAL"]
    if len(raw) < 120 or raw.index.min().to_period("M") != pd.Period("2013-09", "M"):
        raise ValueError("Global QMJ OOS must start in 2013-09 and span ten years")
    expected = pd.period_range("2013-09", raw.index.max().to_period("M"), freq="M")
    missing = expected.difference(raw.index.to_period("M"))
    if len(missing):
        raise ValueError(f"Missing global QMJ OOS months: {list(missing[:3])}")
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns = study.apply_scenario(raw, leverage)
        results[leverage] = {
            "returns": returns,
            "validation": validation.evaluate(returns),
        }
    primary = results[1.0]["returns"]
    monte_carlo = validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=3, seed=16180339
    )
    period_label = pd.Series(
        ["2013_09_2019" if year <= 2019 else "2020_plus" for year in primary.index.year],
        index=primary.index,
    )
    event_label = pd.Series(
        [str(year) if year in (2020, 2022) else "other_months" for year in primary.index.year],
        index=primary.index,
    )
    regimes = pd.concat([
        validation.regime_metrics(primary, period_label).assign(family="period"),
        validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)
    return raw, results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_aqr_qmj_oos"
    frame = study.load_factor_csv(output / "inputs/aqr_qmj_global_monthly.csv")
    raw, results, monte_carlo, regimes = run_experiment(frame)
    raw.rename("gross_qmj_global").rename_axis("date").to_csv(
        output / "oos_factor_input.csv"
    )
    rows = []
    for leverage, result in results.items():
        result["returns"].rename_axis("date").to_csv(
            output / f"monthly_returns_{int(leverage)}x.csv"
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
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "strict_oos_passed"]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
