"""Run the pre-registered SG trend, VXTH, and fixed BTC trend blend."""
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

import blend_risk_overlay
import pput_btc_blend
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
)

SGTI_URL = (
    "https://wholesale.banking.societegenerale.com/fileadmin/indices_feeds/"
    "ti_screen/data/4.nav.csv"
)


def parse_sgti_csv(text):
    frame = pd.read_csv(StringIO(text))
    required = {"trade_date", "SG Trend Indicator"}
    if not required.issubset(frame.columns):
        raise ValueError("SG Trend Indicator CSV列格式不符合预期")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], utc=True)
    frame["SG Trend Indicator"] = pd.to_numeric(
        frame["SG Trend Indicator"], errors="coerce"
    )
    return frame.dropna(subset=["trade_date", "SG Trend Indicator"])


def load_sgti(path, timeout=20):
    if path.exists():
        return parse_sgti_csv(path.read_text())
    response = requests.get(SGTI_URL, timeout=timeout)
    response.raise_for_status()
    frame = parse_sgti_csv(response.text)
    frame.to_csv(path, index=False)
    return frame


def align_components(sgti_level, vxth_level, btc_simulation):
    btc = btc_simulation.copy()
    btc["open_time"] = pd.to_datetime(btc["open_time"], utc=True)
    btc_equity = btc.set_index("open_time")["equity"].sort_index()
    common = sgti_level.index.intersection(vxth_level.index)
    common = common.intersection(btc_equity.index)
    levels = pd.DataFrame({
        "SG_TREND_INDICATOR": sgti_level.reindex(common),
        "VXTH": vxth_level.reindex(common),
        "BTC_TSMOM30_DD": btc_equity.reindex(common),
    }, index=common)
    return levels.pct_change(fill_method=None).fillna(0.0)


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
            repetitions=repetitions, seed=244949
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
    output = root / "reports/mini_medallion_sgti_vxth_btc"
    output.mkdir(parents=True, exist_ok=True)
    sg_frame = load_sgti(output / "sg_official_nav.csv")
    sgti = sg_frame.set_index("trade_date")["SG Trend Indicator"]
    vxth = pd.read_csv(
        root / "reports/mini_medallion_vxth_btc/vxth_index_levels.csv",
        parse_dates=["open_time"],
    ).set_index("open_time")["level"]
    btc = pd.read_csv(
        root / "reports/mini_medallion_btc_trend30/equity_TSMOM30_DD_1x.csv",
        parse_dates=["open_time"],
    )
    component_returns = align_components(sgti, vxth, btc).loc[
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

    print("Component correlations:")
    print(component_returns.corr())
    active = artifacts["1x"][1]
    active = active[active.sum(axis=1) > 0]
    print("Average active weights:")
    print(active.mean())
    columns = [
        "scenario", "cagr", "sharpe_ratio", "max_drawdown",
        "positive_complete_year_ratio", "worst_rolling_3y_sharpe",
        "final_value", "bootstrap_p", "bootstrap_bh_q", "passed"
    ]
    print(summary[columns].to_string(index=False))


if __name__ == "__main__":
    main()
