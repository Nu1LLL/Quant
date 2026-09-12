"""Run the preregistered AQR Value and Momentum Everywhere OOS study."""
from pathlib import Path

import pandas as pd

import aqr_tsmom_oos as validation
import aqr_vme_oos as study


OOS_START = pd.Timestamp("2013-07-01")


def run_experiment(frame, simulations=5000):
    factors = frame.loc[frame.index >= OOS_START, ["VAL", "MOM"]]
    if len(factors) < 120 or factors.index.min().to_period("M") != pd.Period("2013-07", "M"):
        raise ValueError("AQR VME OOS must start in 2013-07 and span ten years")
    expected = pd.period_range("2013-07", factors.index.max().to_period("M"), freq="M")
    missing = expected.difference(factors.index.to_period("M"))
    if len(missing):
        raise ValueError(f"Missing AQR VME OOS months: {list(missing[:3])}")
    gross = study.equal_value_momentum(factors)
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns = study.apply_scenario(gross, leverage)
        results[leverage] = {
            "returns": returns,
            "validation": validation.evaluate(returns),
        }
    primary = results[1.0]["returns"]
    monte_carlo = validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=3, seed=31415926
    )
    period_label = pd.Series(
        ["2013_07_2019" if year <= 2019 else "2020_plus" for year in primary.index.year],
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
    diagnostics = {
        "val_mom_correlation": float(factors["VAL"].corr(factors["MOM"])),
        "oos_months": int(len(factors)),
        "start": str(factors.index.min().date()),
        "end": str(factors.index.max().date()),
    }
    return factors, gross, results, monte_carlo, regimes, diagnostics


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_aqr_vme_oos"
    frame = study.load_factor_csv(output / "inputs/aqr_vme_everywhere_monthly.csv")
    factors, gross, results, monte_carlo, regimes, diagnostics = run_experiment(frame)
    pd.concat([factors, gross], axis=1).rename_axis("date").to_csv(
        output / "oos_factor_components.csv"
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
    pd.DataFrame([diagnostics]).to_csv(output / "diagnostics.csv", index=False)
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "strict_oos_passed"]].to_string(index=False))
    print("DIAGNOSTICS", diagnostics)
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
