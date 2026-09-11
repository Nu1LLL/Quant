"""Run the pre-registered cross-asset ETF trend experiment."""
import argparse
from pathlib import Path

import pandas as pd

import cross_asset_trend as trend
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def run_experiment(prices, repetitions=4999):
    rows = []
    artifacts = {}
    for leverage in (1.0, 2.0, 3.0):
        simulation, positions, yearly, metrics = trend.run_backtest(
            prices, leverage=leverage, initial_capital=10000.0
        )
        pvalue = circular_block_bootstrap_mean_pvalue(
            simulation["net_pnl"], block_length=21,
            repetitions=repetitions, seed=2718
        )
        rows.append({
            "leverage": leverage,
            **metrics,
            "bootstrap_p": pvalue,
        })
        artifacts[leverage] = {
            "simulation": simulation,
            "positions": positions,
            "yearly": yearly,
        }

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
    parser.add_argument("--start", default="2016-09-12")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--repetitions", type=int, default=4999)
    parser.add_argument(
        "--output-folder",
        default="reports/mini_medallion_cross_asset_trend",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = Path(__file__).resolve().parent
    prices = trend.load_price_matrix(
        trend.DEFAULT_SYMBOLS, args.start, args.end,
        root / "cross_asset_data_cache", refresh=args.refresh
    )
    summary, artifacts = run_experiment(prices, repetitions=args.repetitions)
    output = root / args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "scenario_summary.csv", index=False)
    prices.rename_axis("open_time").to_csv(output / "adjusted_close_input.csv")
    for leverage, artifact in artifacts.items():
        label = f"{int(leverage)}x"
        artifact["simulation"].to_csv(
            output / f"equity_{label}.csv", index=False
        )
        artifact["positions"].rename_axis("open_time").to_csv(
            output / f"positions_{label}.csv"
        )
        artifact["yearly"].rename("return").rename_axis("year").to_csv(
            output / f"yearly_{label}.csv"
        )
    print(f"Common observations: {len(prices)}")
    print(f"Common range: {prices.index.min()} -> {prices.index.max()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
