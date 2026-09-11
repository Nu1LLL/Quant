"""Run the pre-registered PPUT/BTC trend blend."""
from pathlib import Path

import pandas as pd

import pput_btc_blend as blend
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def run_experiment(component_returns, repetitions=4999):
    rows = []
    artifacts = {}
    for leverage in (1.0, 2.0, 3.0):
        simulation, positions, yearly, metrics = blend.run_scenario(
            component_returns, leverage
        )
        rows.append({
            "leverage": leverage,
            **metrics,
            "bootstrap_p": circular_block_bootstrap_mean_pvalue(
                simulation["net_pnl"], block_length=21,
                repetitions=repetitions, seed=161803
            ),
        })
        artifacts[leverage] = (simulation, positions, yearly)
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


def main():
    root = Path(__file__).resolve().parent
    pput_frame = pd.read_csv(
        root / "reports/mini_medallion_cboe_options/cboe_index_levels.csv",
        parse_dates=["open_time"],
    )
    pput = pput_frame.set_index("open_time")["PPUT"]
    btc = pd.read_csv(
        root / "reports/mini_medallion_btc_trend30/equity_TSMOM30_DD_1x.csv",
        parse_dates=["open_time"],
    )
    component_returns = blend.align_component_returns(pput, btc)
    summary, artifacts = run_experiment(component_returns)
    output = root / "reports/mini_medallion_pput_btc_blend"
    output.mkdir(parents=True, exist_ok=True)
    component_returns.rename_axis("open_time").to_csv(
        output / "component_returns.csv"
    )
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for leverage, (simulation, positions, yearly) in artifacts.items():
        label = f"{int(leverage)}x"
        simulation.to_csv(output / f"equity_{label}.csv", index=False)
        positions.rename_axis("open_time").to_csv(
            output / f"positions_{label}.csv"
        )
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{label}.csv"
        )
    correlation = component_returns.corr().iloc[0, 1]
    active_positions = artifacts[1.0][1]
    print(f"Component correlation: {correlation:.6f}")
    print("Average active weights:")
    print(active_positions[active_positions.sum(axis=1) > 0].mean())
    columns = [
        "leverage", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
