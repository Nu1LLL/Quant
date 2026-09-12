"""Run the preregistered DIA overnight-long/intraday-short OOS study."""
from pathlib import Path

import pandas as pd

import dia_overnight_intraday_oos as strategy
import strict_validation


def run_experiment(frame, simulations=5000):
    frame = strategy.validate_ohlc(frame)
    span_years = (frame.index[-1] - frame.index[0]).days / 365.25
    if span_years < 20.0:
        raise ValueError("DIA history must span at least twenty years")
    inception_covered = frame.index[0] <= strategy.LATEST_ACCEPTABLE_START
    results = {}
    for leverage in (1.0, 2.0, 3.0):
        returns, detail = strategy.run_scenario(frame, leverage)
        validation = strict_validation.evaluate_strict_oos(
            returns, independent_oos=True, costs_included=True, minimum_years=20.0
        )
        validation["checks"]["inception_covered"] = inception_covered
        validation["passed"] = all(validation["checks"].values())
        results[leverage] = {
            "returns": returns, "detail": detail, "validation": validation,
        }
    primary = results[1.0]["returns"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=19980120
    )
    period_label = pd.Series([
        "1998_2002" if year <= 2002 else
        "2003_2007" if year <= 2007 else
        "2008_2012" if year <= 2012 else
        "2013_2019" if year <= 2019 else "2020_plus"
        for year in primary.index.year
    ], index=primary.index)
    event_label = pd.Series([
        "2000_2002" if year in (2000, 2001, 2002) else
        str(year) if year in (2008, 2018, 2020, 2022, 2025, 2026)
        else "other_days" for year in primary.index.year
    ], index=primary.index)
    regimes = pd.concat([
        strict_validation.regime_metrics(primary, period_label).assign(family="period"),
        strict_validation.regime_metrics(primary, event_label).assign(family="event"),
    ], ignore_index=True)
    return results, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_dia_overnight_intraday_oos"
    inputs = output / "inputs"
    frame = strategy.load_ohlc(inputs, refresh=True)
    results, monte_carlo, regimes = run_experiment(frame)
    frame.rename_axis("date").to_csv(output / "ohlc_adjusted.csv")
    rows = []
    for leverage, result in results.items():
        suffix = f"{int(leverage)}x"
        result["detail"].rename_axis("date").to_csv(output / f"daily_detail_{suffix}.csv")
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
    primary_detail = results[1.0]["detail"]
    component_summary = pd.DataFrame([{
        "gross_overnight_mean_annualized": primary_detail["overnight_return"].mean() * 252,
        "gross_short_intraday_mean_annualized": -primary_detail["intraday_return"].mean() * 252,
        "trading_cost_annualized": primary_detail["trading_cost"].mean() * 252,
        "short_borrow_annualized": primary_detail["short_borrow"].mean() * 252,
    }])
    component_summary.to_csv(output / "component_summary_1x.csv", index=False)
    print("DATA", len(frame), frame.index.min(), frame.index.max())
    print(summary[["leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
                   "oos__profit_factor", "check__walk_forward_passed",
                   "check__inception_covered", "strict_oos_passed"]].to_string(index=False))
    print("COMPONENTS_1x", component_summary.iloc[0].to_dict())
    print("MONTE_CARLO_1x", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio",
                   "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
