"""Run the pre-registered risk overlay on the fixed PPUT/BTC blend."""
from pathlib import Path

import pandas as pd

import blend_risk_overlay
import pput_btc_blend
from stability_significance import circular_block_bootstrap_mean_pvalue


def main():
    root = Path(__file__).resolve().parent
    component_returns = pd.read_csv(
        root / "reports/mini_medallion_pput_btc_blend/component_returns.csv",
        parse_dates=["open_time"],
    ).set_index("open_time")
    base_returns, _, _ = pput_btc_blend.run_base_blend(component_returns)
    simulation, yearly, metrics = blend_risk_overlay.run_overlay(base_returns)
    pvalue = circular_block_bootstrap_mean_pvalue(
        simulation["net_pnl"], block_length=21,
        repetitions=4999, seed=141421
    )
    checks = {
        "sharpe_1_5": metrics["sharpe_ratio"] >= 1.5,
        "cagr_40pct": metrics["cagr"] >= 0.40,
        "max_drawdown_20pct": metrics["max_drawdown"] >= -0.20,
        "positive_years_80pct": metrics["positive_complete_year_ratio"] >= 0.80,
        "worst_rolling_3y_sharpe_1": metrics["worst_rolling_3y_sharpe"] >= 1.0,
        "mean_p_5pct": pvalue <= 0.05,
    }
    row = {**metrics, "bootstrap_p": pvalue}
    row.update({f"check__{name}": value for name, value in checks.items()})
    row["passed"] = all(checks.values())
    summary = pd.DataFrame([row])

    output = root / "reports/mini_medallion_pput_btc_risk_overlay"
    output.mkdir(parents=True, exist_ok=True)
    simulation.to_csv(output / "equity.csv", index=False)
    yearly.rename("return").rename_axis("year").to_csv(output / "yearly.csv")
    summary.to_csv(output / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
