"""Run the pre-registered Cboe VXTH and fixed BTC trend blend."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import cboe_options_benchmark
import pput_btc_blend
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def run_experiment(component_returns, repetitions=4999):
    rows = []
    artifacts = {}
    for leverage in (1.0, 2.0, 3.0):
        simulation, positions, yearly, metrics = pput_btc_blend.run_scenario(
            component_returns, leverage
        )
        name = f"{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, positions, yearly)

    base_return, positions, _ = pput_btc_blend.run_base_blend(
        component_returns
    )
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(base_return)
    rows.append({"scenario": "RISK_OVERLAY", **metrics})
    artifacts["RISK_OVERLAY"] = (simulation, positions, yearly)

    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=223607
        )
        for name in summary["scenario"]
    ]
    summary["bootstrap_bh_q"] = benjamini_hochberg(
        summary["bootstrap_p"]
    )
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
    output = root / "reports/mini_medallion_vxth_btc"
    output.mkdir(parents=True, exist_ok=True)
    level_path = output / "vxth_index_levels.csv"
    if level_path.exists():
        vxth = pd.read_csv(
            level_path, parse_dates=["open_time"]
        ).set_index("open_time")["level"].rename("VXTH")
    else:
        vxth = cboe_options_benchmark.download_index("VXTH")
        vxth.rename("level").rename_axis("open_time").to_csv(level_path)

    btc = pd.read_csv(
        root / "reports/mini_medallion_btc_trend30/equity_TSMOM30_DD_1x.csv",
        parse_dates=["open_time"],
    )
    component_returns = pput_btc_blend.align_component_returns(vxth, btc)
    component_returns = component_returns.rename(columns={"PPUT": "VXTH"})
    component_returns = component_returns.loc[
        "2016-09-12":"2026-09-10"
    ]
    summary, artifacts = run_experiment(component_returns)

    component_returns.rename_axis("open_time").to_csv(
        output / "component_returns.csv"
    )
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (simulation, positions, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{name}.csv", index=False)
        positions.rename_axis("open_time").to_csv(
            output / f"positions_{name}.csv"
        )
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{name}.csv"
        )

    correlation = component_returns.corr().iloc[0, 1]
    active_positions = artifacts["1x"][1]
    active_positions = active_positions[active_positions.sum(axis=1) > 0]
    print(f"Component correlation: {correlation:.6f}")
    print("Average active weights:")
    print(active_positions.mean())
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
