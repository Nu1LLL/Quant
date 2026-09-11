"""Run the preregistered Cboe VPN and three-component blend study."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import cboe_options_benchmark as options
import pput_btc_blend
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)


START = pd.Timestamp("2016-09-12", tz="UTC")
END = pd.Timestamp("2026-09-10", tz="UTC")


def build_components(existing_components, vpn_level):
    existing = existing_components.sort_index().loc[START:END]
    vpn = vpn_level.sort_index().loc[START:END]
    common = existing.index.intersection(vpn.index)
    if common.empty or common[0] != START or common[-1] != END:
        raise ValueError("VPN does not cover the preregistered ten-year window")
    result = existing.reindex(common).copy()
    result["VPN"] = vpn.reindex(common).pct_change(fill_method=None).fillna(0.0)
    return result


def run_experiment(existing_components, vpn_level, repetitions=4999):
    components = build_components(existing_components, vpn_level)
    vpn_return = components["VPN"]
    rows = []
    artifacts = {}

    for leverage in (1.0, 2.0, 3.0):
        returns = options.apply_leverage(vpn_return, leverage)
        simulation, yearly, metrics = options.summarize_returns(returns)
        simulation["equity"] = (1.0 + returns).cumprod().values
        name = f"VPN_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, None, yearly)

    for leverage in (1.0, 2.0, 3.0):
        simulation, positions, yearly, metrics = pput_btc_blend.run_scenario(
            components, leverage
        )
        name = f"BLEND_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, positions, yearly)

    base_return, positions, _ = pput_btc_blend.run_base_blend(components)
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(base_return)
    rows.append({"scenario": "BLEND_RISK_OVERLAY", **metrics})
    artifacts["BLEND_RISK_OVERLAY"] = (simulation, positions, yearly)

    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=1346269
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
    return summary, artifacts, components


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_vpn_blend"
    output.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(
        root / "reports/mini_medallion_tqqq_wtmf_btc/blend_component_returns.csv",
        parse_dates=["open_time"], index_col="open_time"
    )
    vpn_level = options.download_index("VPN")
    vpn_level.rename("level").rename_axis("open_time").to_csv(
        output / "VPN_History.csv"
    )
    summary, artifacts, components = run_experiment(existing, vpn_level)
    components.rename_axis("open_time").to_csv(output / "component_returns.csv")
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (simulation, positions, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{name}.csv", index=False)
        if positions is not None:
            positions.rename_axis("open_time").to_csv(output / f"positions_{name}.csv")
        yearly.rename("return").rename_axis("year").to_csv(output / f"yearly_{name}.csv")
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
