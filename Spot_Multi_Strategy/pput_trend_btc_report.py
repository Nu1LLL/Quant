"""Run the pre-registered PPUT trend-filtered BTC blend."""
from pathlib import Path

import pandas as pd

import pput_btc_blend
import pput_trend_btc
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


def main():
    root = Path(__file__).resolve().parent
    pput_frame = pd.read_csv(
        root / "reports/mini_medallion_cboe_options/cboe_index_levels.csv",
        parse_dates=["open_time"],
    ).set_index("open_time")
    btc = pd.read_csv(
        root / "reports/mini_medallion_btc_trend30/equity_TSMOM30_DD_1x.csv",
        parse_dates=["open_time"],
    )
    raw_components = pput_btc_blend.align_component_returns(
        pput_frame["PPUT"], btc
    )
    pput_return, pput_position, pput_turnover = (
        pput_trend_btc.build_pput_trend_returns(pput_frame["PPUT"])
    )
    component_returns = raw_components.copy()
    component_returns["PPUT"] = pput_return.reindex(
        component_returns.index
    ).fillna(0.0)

    rows = []
    artifacts = {}
    for leverage in (1.0, 2.0, 3.0):
        simulation, positions, yearly, metrics = (
            pput_trend_btc.run_static_scenario(component_returns, leverage)
        )
        name = f"{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, positions, yearly)
    simulation, positions, yearly, metrics = (
        pput_trend_btc.run_overlay_scenario(component_returns)
    )
    rows.append({"scenario": "RISK_OVERLAY", **metrics})
    artifacts["RISK_OVERLAY"] = (simulation, positions, yearly)

    summary = pd.DataFrame(rows)
    pvalues = []
    for name in summary["scenario"]:
        pvalues.append(circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=4999, seed=173205
        ))
    summary["bootstrap_p"] = pvalues
    summary["bootstrap_bh_q"] = benjamini_hochberg(pvalues)
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

    output = root / "reports/mini_medallion_pput_trend_btc"
    output.mkdir(parents=True, exist_ok=True)
    component_returns.rename_axis("open_time").to_csv(
        output / "component_returns.csv"
    )
    pd.DataFrame({
        "open_time": pput_frame.index,
        "position": pput_position,
        "turnover": pput_turnover,
        "net_pnl": pput_return,
    }).to_csv(output / "pput_trend_sleeve.csv", index=False)
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (simulation, positions, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{name}.csv", index=False)
        positions.rename_axis("open_time").to_csv(
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
