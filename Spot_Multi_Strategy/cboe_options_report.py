"""Run the pre-registered Cboe options benchmark experiment."""
import argparse
from pathlib import Path

import pandas as pd

import cboe_options_benchmark as options
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def run_experiment(levels, start, end, repetitions=4999):
    levels = levels.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    base_returns = {
        symbol: levels[symbol].pct_change(fill_method=None).fillna(0.0)
        for symbol in options.SYMBOLS
    }
    equal_returns, turnover = options.monthly_equal_weight_returns(levels)
    base_returns["OPTIONS_EQUAL"] = equal_returns

    rows = []
    artifacts = {}
    for candidate, returns in base_returns.items():
        for leverage in (1.0, 2.0, 3.0):
            scenario_returns = options.apply_leverage(returns, leverage)
            simulation, yearly, metrics = options.summarize_returns(
                scenario_returns
            )
            scenario = f"{candidate}_{int(leverage)}x"
            rows.append({
                "scenario": scenario,
                "candidate": candidate,
                "leverage": leverage,
                **metrics,
                "bootstrap_p": circular_block_bootstrap_mean_pvalue(
                    scenario_returns, block_length=21,
                    repetitions=repetitions, seed=31415
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
    return levels, summary, artifacts, turnover


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-09-12")
    parser.add_argument("--end", default="2026-09-10")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--repetitions", type=int, default=4999)
    parser.add_argument(
        "--output-folder", default="reports/mini_medallion_cboe_options"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = Path(__file__).resolve().parent
    levels = options.load_index_matrix(
        root / "cboe_options_data_cache", refresh=args.refresh
    )
    levels, summary, artifacts, turnover = run_experiment(
        levels, args.start, args.end, repetitions=args.repetitions
    )
    output = root / args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    levels.rename_axis("open_time").to_csv(output / "cboe_index_levels.csv")
    summary.to_csv(output / "scenario_summary.csv", index=False)
    turnover.rename_axis("open_time").to_csv(output / "equal_weight_turnover.csv")
    for scenario, (simulation, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{scenario}.csv", index=False)
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{scenario}.csv"
        )
    print(f"Common observations: {len(levels)}")
    print(f"Common range: {levels.index.min()} -> {levels.index.max()}")
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].sort_values("sharpe_ratio", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
