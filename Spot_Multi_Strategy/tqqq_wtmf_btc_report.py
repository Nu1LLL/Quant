"""Run the pre-registered TQQQ/WTMF regime switch and BTC blend."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import pput_btc_blend
import tqqq_trend_btc
import tqqq_wtmf_btc
from cross_asset_trend import load_adjusted_close
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def run_experiment(qqq, tqqq, wtmf, btc, repetitions=4999):
    sleeve = tqqq_wtmf_btc.run_switch_sleeve(qqq, tqqq, wtmf)
    rows = [{"scenario": "TQQQ_WTMF_SWITCH", **sleeve[3]}]
    artifacts = {
        "TQQQ_WTMF_SWITCH": (sleeve[0], sleeve[1], sleeve[2])
    }
    sleeve_return = sleeve[0].set_index("open_time")["net_pnl"]
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(sleeve_return)
    rows.append({"scenario": "SWITCH_RISK_OVERLAY", **metrics})
    artifacts["SWITCH_RISK_OVERLAY"] = (simulation, None, yearly)

    component_returns = tqqq_trend_btc.align_with_btc(sleeve[0], btc).rename(
        columns={"TACTICAL_TQQQ": "TQQQ_WTMF_SWITCH"}
    )
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
    rows.append({"scenario": "BLEND_RISK_OVERLAY", **metrics})
    artifacts["BLEND_RISK_OVERLAY"] = (simulation, positions, yearly)

    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=433494
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
    return summary, artifacts, component_returns


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_tqqq_wtmf_btc"
    output.mkdir(parents=True, exist_ok=True)
    prices = {}
    for symbol in ("QQQ", "TQQQ", "WTMF"):
        series = load_adjusted_close(
            symbol, "2015-09-01", "2026-09-11", output
        )
        series.index = series.index.normalize()
        prices[symbol] = series
    btc = pd.read_csv(
        root / "reports/mini_medallion_btc_trend30/equity_TSMOM30_DD_1x.csv",
        parse_dates=["open_time"],
    )
    summary, artifacts, component_returns = run_experiment(
        prices["QQQ"], prices["TQQQ"], prices["WTMF"], btc
    )
    pd.concat(prices, axis=1).rename_axis("open_time").to_csv(
        output / "adjusted_levels.csv"
    )
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
    print("Average switch positions:")
    print(artifacts["TQQQ_WTMF_SWITCH"][1].mean())
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
