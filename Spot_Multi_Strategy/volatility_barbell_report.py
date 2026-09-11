"""Evaluate the preregistered growth/SVRPO/VXTH volatility barbell."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import cboe_options_benchmark as options
import fixed_weight_blend
import pput_btc_blend
from stability_significance import benjamini_hochberg, circular_block_bootstrap_mean_pvalue


START = pd.Timestamp("2016-09-12", tz="UTC")
END = pd.Timestamp("2026-09-10", tz="UTC")


def build_levels(growth_returns, svrpo, vxth):
    common = growth_returns.index.intersection(svrpo.index).intersection(vxth.index)
    common = common[(common >= START) & (common <= END)]
    if common.empty or common[0] != START or common[-1] != END:
        raise ValueError("Components do not cover the preregistered window")
    return pd.DataFrame({
        "GROWTH": (1.0 + growth_returns.reindex(common)).cumprod(),
        "SVRPO": svrpo.reindex(common),
        "VXTH": vxth.reindex(common),
    }, index=common)


def run_experiment(growth_returns, svrpo, vxth, repetitions=4999):
    levels = build_levels(growth_returns, svrpo, vxth)
    barbell_return, barbell_positions, barbell_turnover = fixed_weight_blend.run_fixed_weight(
        levels[["SVRPO", "VXTH"]], {"SVRPO": 0.5, "VXTH": 0.5}
    )
    full_return, full_positions, full_turnover = fixed_weight_blend.run_fixed_weight(
        levels, {"GROWTH": 0.5, "SVRPO": 0.25, "VXTH": 0.25}
    )
    rows, artifacts = [], {}
    simulation, yearly, metrics = options.summarize_returns(barbell_return)
    simulation["equity"] = (1 + barbell_return).cumprod().values
    simulation["turnover"] = barbell_turnover.values
    rows.append({"scenario": "VOL_BARBELL_1x", **metrics})
    artifacts["VOL_BARBELL_1x"] = (simulation, barbell_positions, yearly)
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(barbell_return)
    rows.append({"scenario": "VOL_BARBELL_RISK_OVERLAY", **metrics})
    artifacts["VOL_BARBELL_RISK_OVERLAY"] = (simulation, barbell_positions, yearly)
    for leverage in (1.0, 2.0, 3.0):
        returns = options.apply_leverage(full_return, leverage)
        simulation, yearly, metrics = options.summarize_returns(returns)
        simulation["equity"] = (1 + returns).cumprod().values
        simulation["turnover"] = (full_turnover * leverage).values
        name = f"FULL_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, full_positions, yearly)
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(full_return)
    rows.append({"scenario": "FULL_RISK_OVERLAY", **metrics})
    artifacts["FULL_RISK_OVERLAY"] = (simulation, full_positions, yearly)
    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=3524578
        ) for name in summary["scenario"]
    ]
    summary["bootstrap_bh_q"] = benjamini_hochberg(summary["bootstrap_p"])
    summary["check__sharpe_1_5"] = summary["sharpe_ratio"] >= 1.5
    summary["check__cagr_40pct"] = summary["cagr"] >= 0.40
    summary["check__max_drawdown_20pct"] = summary["max_drawdown"] >= -0.20
    summary["check__positive_years_80pct"] = summary["positive_complete_year_ratio"] >= 0.80
    summary["check__worst_rolling_3y_sharpe_1"] = summary["worst_rolling_3y_sharpe"] >= 1.0
    summary["check__mean_fdr_5pct"] = summary["bootstrap_bh_q"] <= 0.05
    checks = [column for column in summary if column.startswith("check__")]
    summary["passed"] = summary[checks].all(axis=1)
    return summary, artifacts, levels


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_volatility_barbell"
    output.mkdir(parents=True, exist_ok=True)
    components = pd.read_csv(
        root / "reports/mini_medallion_tqqq_wtmf_btc/blend_component_returns.csv",
        parse_dates=["open_time"], index_col="open_time"
    )
    growth_returns, _, _ = pput_btc_blend.run_base_blend(components)
    svrpo = pd.read_csv(
        root / "reports/mini_medallion_svrpo_blend/SVRPO_History.csv",
        parse_dates=["open_time"], index_col="open_time"
    )["level"]
    vxth = pd.read_csv(
        root / "reports/mini_medallion_vxth_btc/vxth_index_levels.csv",
        parse_dates=["open_time"], index_col="open_time"
    )["level"]
    summary, artifacts, levels = run_experiment(growth_returns, svrpo, vxth)
    levels.rename_axis("open_time").to_csv(output / "aligned_levels.csv")
    summary.to_csv(output / "scenario_summary.csv", index=False)
    for name, (simulation, positions, yearly) in artifacts.items():
        simulation.to_csv(output / f"equity_{name}.csv", index=False)
        positions.rename_axis("open_time").to_csv(output / f"positions_{name}.csv")
        yearly.rename("return").rename_axis("year").to_csv(output / f"yearly_{name}.csv")
    columns = ["scenario", "cagr", "sharpe_ratio", "max_drawdown",
               "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
               "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
