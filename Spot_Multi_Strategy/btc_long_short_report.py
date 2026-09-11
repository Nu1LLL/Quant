"""Run the pre-registered BTC long/short and VXTH blend study."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import btc_long_short
import pput_btc_blend
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def add_gate(summary):
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
    return summary


def run_experiment(price, vxth_level, repetitions=4999):
    rows = []
    artifacts = {}
    standalone_one = None
    for leverage in (1.0, 2.0, 3.0):
        simulation, yearly, metrics = btc_long_short.run_backtest(
            price, leverage=leverage
        )
        name = f"BTC_LS_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, None, yearly)
        if leverage == 1.0:
            standalone_one = simulation

    component_returns = pput_btc_blend.align_component_returns(
        vxth_level, standalone_one
    ).rename(columns={
        "PPUT": "VXTH", "BTC_TSMOM30_DD": "BTC_TSMOM30_LS"
    })
    for leverage in (1.0, 2.0, 3.0):
        simulation, positions, yearly, metrics = pput_btc_blend.run_scenario(
            component_returns, leverage
        )
        name = f"BLEND_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, positions, yearly)
    base_return, positions, _ = pput_btc_blend.run_base_blend(
        component_returns
    )
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(base_return)
    name = "BLEND_RISK_OVERLAY"
    rows.append({"scenario": name, **metrics})
    artifacts[name] = (simulation, positions, yearly)

    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=316229
        )
        for name in summary["scenario"]
    ]
    return add_gate(summary), artifacts, component_returns


def main():
    root = Path(__file__).resolve().parent
    price = pd.read_csv(
        root / "reports/mini_medallion_btc_trend30/btc_adjusted_close.csv",
        parse_dates=["open_time"],
    ).set_index("open_time")["adjusted_close"]
    vxth = pd.read_csv(
        root / "reports/mini_medallion_vxth_btc/vxth_index_levels.csv",
        parse_dates=["open_time"],
    ).set_index("open_time")["level"]
    summary, artifacts, component_returns = run_experiment(price, vxth)

    output = root / "reports/mini_medallion_btc_long_short"
    output.mkdir(parents=True, exist_ok=True)
    component_returns.rename_axis("open_time").to_csv(
        output / "blend_component_returns.csv"
    )
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (simulation, positions, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{name}.csv", index=False)
        if positions is not None:
            positions.rename_axis("open_time").to_csv(
                output / f"positions_{name}.csv"
            )
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{name}.csv"
        )
    print(f"Blend component correlation: {component_returns.corr().iloc[0, 1]:.6f}")
    active = artifacts["BLEND_1x"][1]
    active = active[active.sum(axis=1) > 0]
    print("Average active blend weights:")
    print(active.mean())
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
