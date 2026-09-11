"""Run the preregistered VIX curve filter on the frozen TQQQ/WTMF/BTC blend."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import pput_btc_blend
import vix_curve_filter
from cboe_options_benchmark import summarize_returns
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def evaluate(base_returns, vix, vix3m, repetitions=4999):
    artifacts = {}
    base_simulation, base_yearly, base_metrics = summarize_returns(base_returns)
    base_simulation["equity"] = (1.0 + base_returns).cumprod().values
    artifacts["BASE_BLEND_1x"] = (base_simulation, None, base_yearly)
    rows = [{"scenario": "BASE_BLEND_1x", **base_metrics}]

    filtered_1x = None
    filtered_position = None
    for leverage in (1.0, 2.0, 3.0):
        simulation, position, yearly, metrics = vix_curve_filter.run_filter(
            base_returns, vix, vix3m, leverage=leverage
        )
        name = f"CURVE_FILTER_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, position, yearly)
        if leverage == 1.0:
            filtered_1x = simulation.set_index("open_time")["net_pnl"]
            filtered_position = position

    simulation, yearly, metrics = blend_risk_overlay.run_overlay(filtered_1x)
    rows.append({"scenario": "CURVE_FILTER_RISK_OVERLAY", **metrics})
    artifacts["CURVE_FILTER_RISK_OVERLAY"] = (
        simulation, filtered_position, yearly
    )

    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=514229
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
    output = root / "reports/mini_medallion_vix_curve_filter"
    components = pd.read_csv(
        root / "reports/mini_medallion_tqqq_wtmf_btc/blend_component_returns.csv",
        parse_dates=["open_time"], index_col="open_time"
    )
    base_returns, _, _ = pput_btc_blend.run_base_blend(components)
    vix = vix_curve_filter.load_cboe_close("VIX", output)
    vix3m = vix_curve_filter.load_cboe_close("VIX3M", output)
    summary, artifacts = evaluate(base_returns, vix, vix3m)
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (simulation, position, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{name}.csv", index=False)
        if position is not None:
            position.rename_axis("open_time").to_csv(
                output / f"positions_{name}.csv"
            )
        yearly.rename("return").rename_axis("year").to_csv(
            output / f"yearly_{name}.csv"
        )
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
