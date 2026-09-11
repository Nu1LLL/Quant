"""Run the pre-registered BTC 30-day trend replication."""
import argparse
from pathlib import Path

import pandas as pd

import btc_trend30
from cross_asset_trend import load_adjusted_close
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)

CANDIDATES = ("BUY_HOLD", "TSMOM30", "TSMOM30_DD")


def run_experiment(price, repetitions=4999):
    rows = []
    artifacts = {}
    for candidate in CANDIDATES:
        for leverage in (1.0, 2.0, 3.0):
            simulation, yearly, metrics = btc_trend30.run_backtest(
                price, candidate, leverage=leverage
            )
            scenario = f"{candidate}_{int(leverage)}x"
            rows.append({
                "scenario": scenario,
                "candidate": candidate,
                "leverage": leverage,
                **metrics,
                "bootstrap_p": circular_block_bootstrap_mean_pvalue(
                    simulation["net_pnl"], block_length=21,
                    repetitions=repetitions, seed=271828
                ),
            })
            artifacts[scenario] = (simulation, yearly)
    summary = pd.DataFrame(rows)
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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--repetitions", type=int, default=4999)
    parser.add_argument(
        "--output-folder", default="reports/mini_medallion_btc_trend30"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = Path(__file__).resolve().parent
    price = load_adjusted_close(
        "BTC-USD", "2016-08-01", "2026-09-11",
        root / "btc_trend30_data_cache", refresh=args.refresh
    )
    summary, artifacts = run_experiment(price, repetitions=args.repetitions)
    output = root / args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    price.rename("adjusted_close").rename_axis("open_time").to_csv(
        output / "btc_adjusted_close.csv"
    )
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for scenario, (simulation, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{scenario}.csv", index=False)
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{scenario}.csv"
        )
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].sort_values("sharpe_ratio", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
