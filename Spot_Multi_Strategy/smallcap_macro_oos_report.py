"""Run the preregistered small-cap macro defensive OOS validation."""
from pathlib import Path

import pandas as pd

import smallcap_macro_oos
import strict_validation
import vix_curve_filter
from cross_asset_trend import load_adjusted_close


DEV_START = pd.Timestamp("2010-01-04", tz="UTC")
DEV_END = pd.Timestamp("2020-12-31", tz="UTC")
OOS_START = pd.Timestamp("2021-01-04", tz="UTC")
OOS_END = pd.Timestamp("2026-09-10", tz="UTC")


def run_experiment(prices, vix, vix3m, monte_carlo_simulations=5000):
    dev = smallcap_macro_oos.run_strategy(
        prices["IWM"], prices["TNA"], prices["TBT"], prices["UGL"], DEV_START, DEV_END
    )
    oos = smallcap_macro_oos.run_strategy(
        prices["IWM"], prices["TNA"], prices["TBT"], prices["UGL"], OOS_START, OOS_END
    )
    oos_returns = oos[0].set_index("open_time")["net_pnl"]
    validation = strict_validation.evaluate_strict_oos(
        oos_returns, independent_oos=True, costs_included=True
    )
    monte_carlo = strict_validation.circular_block_monte_carlo(
        oos_returns, simulations=monte_carlo_simulations,
        block_length=21, seed=9227465
    )
    risk_regime = pd.Series(
        oos[1]["TNA"].map({1.0: "risk_on", 0.0: "defensive"}),
        index=oos[1].index
    )
    curve_position = vix_curve_filter.build_risk_position(vix, vix3m, oos_returns.index)
    curve_regime = curve_position.map({1.0: "curve_normal", 0.0: "curve_inverted"})
    regimes = pd.concat([
        strict_validation.regime_metrics(oos_returns, risk_regime).assign(family="trade_state"),
        strict_validation.regime_metrics(oos_returns, curve_regime).assign(family="vix_curve"),
    ], ignore_index=True)
    return dev, oos, validation, monte_carlo, regimes


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_smallcap_macro_oos"
    output.mkdir(parents=True, exist_ok=True)
    prices = {}
    for symbol in ("IWM", "TNA", "TBT", "UGL"):
        series = load_adjusted_close(symbol, "2009-01-01", "2026-09-11", output)
        series.index = series.index.normalize()
        prices[symbol] = series
    vix = pd.read_csv(
        root / "reports/mini_medallion_vix_curve_filter/VIX_History.csv",
        parse_dates=["DATE"], index_col="DATE"
    )["CLOSE"]
    vix3m = pd.read_csv(
        root / "reports/mini_medallion_vix_curve_filter/VIX3M_History.csv",
        parse_dates=["DATE"], index_col="DATE"
    )["CLOSE"]
    dev, oos, validation, monte_carlo, regimes = run_experiment(prices, vix, vix3m)
    pd.concat(prices, axis=1).rename_axis("open_time").to_csv(output / "adjusted_levels.csv")
    for label, artifact in (("DEV", dev), ("OOS", oos)):
        artifact[0].to_csv(output / f"equity_{label}.csv", index=False)
        artifact[1].rename_axis("open_time").to_csv(output / f"positions_{label}.csv")
        artifact[2].rename("return").rename_axis("year").to_csv(output / f"yearly_{label}.csv")
    validation["walk_forward_folds"].to_csv(output / "walk_forward_folds.csv", index=False)
    regimes.to_csv(output / "regime_metrics.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo.csv", index=False)
    summary = {
        **{f"metric__{k}": v for k, v in validation["metrics"].items()},
        **{f"check__{k}": v for k, v in validation["checks"].items()},
        "strict_oos_passed": validation["passed"],
    }
    pd.DataFrame([summary]).to_csv(output / "strict_oos_summary.csv", index=False)
    keys = ["cagr", "sharpe_ratio", "max_drawdown", "profit_factor"]
    print("DEV", {key: dev[3][key] for key in keys})
    print("OOS", {key: validation["metrics"][key] for key in keys})
    print("CHECKS", validation["checks"])
    print("MONTE_CARLO", monte_carlo)
    print(regimes[["family", "regime", "observations", "cagr", "sharpe_ratio", "max_drawdown", "profit_factor"]].to_string(index=False))


if __name__ == "__main__":
    main()
