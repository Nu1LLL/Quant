"""Run the pre-registered BitMEX funding carry study."""
from pathlib import Path

import pandas as pd

import bitmex_funding
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def run_experiment(funding, repetitions=4999):
    rows = []
    artifacts = {}
    for policy in ("ALWAYS", "LAG_POSITIVE"):
        for leverage in (1.0, 2.0, 3.0, 4.0):
            intervals, daily, yearly, metrics = bitmex_funding.run_backtest(
                funding, policy, leverage
            )
            scenario = f"{policy}_{int(leverage)}x"
            rows.append({"scenario": scenario, **metrics})
            artifacts[scenario] = (intervals, daily, yearly)
    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][1]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=353711
        )
        for name in summary["scenario"]
    ]
    summary["bootstrap_bh_q"] = benjamini_hochberg(summary["bootstrap_p"])
    summary["check__sharpe_1_5"] = summary["sharpe_ratio"] >= 1.5
    summary["check__cagr_40pct"] = summary["cagr"] >= 0.40
    summary["check__max_drawdown_20pct"] = summary["max_drawdown"] >= -0.20
    summary["check__positive_years_80pct"] = (
        summary["positive_complete_year_ratio"] >= 0.80
    )
    summary["check__worst_rolling_3y_sharpe_1"] = (
        summary["worst_rolling_3y_sharpe"] >= 1.0
    )
    summary["check__mean_fdr_5pct"] = summary["bootstrap_bh_q"] <= 0.05
    checks = [column for column in summary if column.startswith("check__")]
    summary["passed"] = summary[checks].all(axis=1)
    return summary, artifacts


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_bitmex_funding"
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "xbtusd_funding.csv"
    if data_path.exists():
        funding = pd.read_csv(data_path, parse_dates=["timestamp"])
    else:
        funding = bitmex_funding.download_funding()
        funding.to_csv(data_path, index=False)
    summary, artifacts = run_experiment(funding)
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (intervals, daily, yearly) in artifacts.items():
        intervals.to_csv(output / f"intervals_{name}.csv", index=False)
        daily.to_csv(output / f"equity_{name}.csv", index=False)
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{name}.csv"
        )
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "active_interval_ratio", "final_value", "bootstrap_p",
        "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
