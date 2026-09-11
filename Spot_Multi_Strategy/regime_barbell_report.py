"""Evaluate the preregistered growth/volatility-barbell regime switch."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import cboe_options_benchmark as options
import fixed_weight_blend
import pput_btc_blend
from stability_significance import benjamini_hochberg, circular_block_bootstrap_mean_pvalue


START = pd.Timestamp("2016-09-12", tz="UTC")
END = pd.Timestamp("2026-09-10", tz="UTC")


def build_switch_returns(growth, barbell, risk_on, transaction_cost=0.0005):
    frame = pd.concat({"GROWTH": growth, "BARBELL": barbell}, axis=1, join="inner")
    frame = frame.loc[START:END]
    state = risk_on.reindex(frame.index)
    if state.isna().any() or not state.isin([0.0, 1.0]).all():
        raise ValueError("Risk state must be complete and binary")
    positions = pd.DataFrame({"GROWTH": state, "BARBELL": 1.0-state}, index=frame.index)
    turnover = positions.diff().abs().sum(axis=1)
    turnover.iloc[0] = positions.iloc[0].abs().sum()
    returns = (positions * frame).sum(axis=1) - turnover * transaction_cost
    return returns.rename("switch_return"), positions, turnover


def run_experiment(growth, barbell, risk_on, repetitions=4999):
    base_return, positions, turnover = build_switch_returns(growth, barbell, risk_on)
    rows, artifacts = [], {}
    for leverage in (1.0, 2.0, 3.0):
        returns = options.apply_leverage(base_return, leverage)
        simulation, yearly, metrics = options.summarize_returns(returns)
        simulation["equity"] = (1+returns).cumprod().values
        simulation["turnover"] = (turnover*leverage).values
        name = f"SWITCH_{int(leverage)}x"
        rows.append({"scenario": name, **metrics})
        artifacts[name] = (simulation, positions, yearly)
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(base_return)
    rows.append({"scenario": "SWITCH_RISK_OVERLAY", **metrics})
    artifacts["SWITCH_RISK_OVERLAY"] = (simulation, positions, yearly)
    summary = pd.DataFrame(rows)
    summary["bootstrap_p"] = [
        circular_block_bootstrap_mean_pvalue(
            artifacts[name][0]["net_pnl"], block_length=21,
            repetitions=repetitions, seed=5702887
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
    return summary, artifacts


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_regime_barbell"
    output.mkdir(parents=True, exist_ok=True)
    components = pd.read_csv(
        root / "reports/mini_medallion_tqqq_wtmf_btc/blend_component_returns.csv",
        parse_dates=["open_time"], index_col="open_time"
    )
    growth, _, _ = pput_btc_blend.run_base_blend(components)
    levels = pd.read_csv(
        root / "reports/mini_medallion_volatility_barbell/aligned_levels.csv",
        parse_dates=["open_time"], index_col="open_time"
    )
    barbell, _, _ = fixed_weight_blend.run_fixed_weight(
        levels[["SVRPO", "VXTH"]], {"SVRPO": 0.5, "VXTH": 0.5}
    )
    state = pd.read_csv(
        root / "reports/mini_medallion_tqqq_wtmf_btc/positions_TQQQ_WTMF_SWITCH.csv",
        parse_dates=["open_time"], index_col="open_time"
    )["TQQQ"]
    summary, artifacts = run_experiment(growth, barbell, state)
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
