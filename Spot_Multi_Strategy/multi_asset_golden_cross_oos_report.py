"""Run the preregistered multi-asset-class golden-cross portfolio OOS
study (QQQ equity / FXE currency / SLV commodity / ETHUSDT crypto),
using the standard 5-year strict_validation gate since the sample is
bound by ETHUSDT's ~9.1-year history."""
from pathlib import Path

import pandas as pd

import multi_asset_golden_cross_oos as mag
import strict_validation
from cross_asset_trend import load_adjusted_close
from data import load_or_download_klines


def run_experiment(prices_by_symbol, simulations=5000):
    aligned = mag.align_sleeves(prices_by_symbol)
    if (aligned.index[-1] - aligned.index[0]).days / 365.25 < 5.0:
        raise ValueError("Multi-asset sample must span at least five years")

    results = {}
    for leverage in (1.0, 2.0, 3.0):
        sleeve_returns, sleeve_details = mag.run_sleeve_returns(aligned, leverage)
        combined = mag.combine_equal_weight(sleeve_returns)
        results[leverage] = {
            "combined": combined,
            "sleeve_returns": sleeve_returns,
            "sleeve_details": sleeve_details,
            "validation": strict_validation.evaluate_strict_oos(
                combined, independent_oos=True, costs_included=True
            ),
        }

    primary = results[1.0]["combined"]
    monte_carlo = strict_validation.circular_block_monte_carlo(
        primary, simulations=simulations, block_length=21, seed=20170817
    )

    correlation = pd.DataFrame(results[1.0]["sleeve_returns"]).corr()

    event_label = pd.Series([
        str(year) if year in (2020, 2022) else "other_days"
        for year in primary.index.year
    ], index=primary.index)
    regimes = strict_validation.regime_metrics(primary, event_label).assign(family="event")

    return results, monte_carlo, regimes, correlation


def main():
    root = Path(__file__).resolve().parent
    output = root / "reports/mini_medallion_multi_asset_golden_cross_oos"
    inputs = output / "inputs"

    qqq = load_adjusted_close("QQQ", "2017-08-01", "2026-09-12", cache_folder=inputs, refresh=False)
    fxe = load_adjusted_close("FXE", "2017-08-01", "2026-09-12", cache_folder=inputs, refresh=False)
    slv = load_adjusted_close("SLV", "2017-08-01", "2026-09-12", cache_folder=inputs, refresh=False)
    eth_klines = load_or_download_klines(
        "ETHUSDT", "1d", "2017-08-17", "2026-09-12", cache_folder=root / "data_cache"
    )
    eth = eth_klines.set_index("open_time")["close"]

    prices_by_symbol = {"QQQ": qqq, "FXE": fxe, "SLV": slv, "ETHUSDT": eth}

    results, monte_carlo, regimes, correlation = run_experiment(prices_by_symbol)

    for symbol, series in prices_by_symbol.items():
        series.rename_axis("date").to_csv(output / f"{symbol.lower()}_adjusted_close.csv")

    rows = []
    for leverage, result in results.items():
        result["validation"]["walk_forward_folds"].to_csv(
            output / f"walk_forward_{int(leverage)}x.csv", index=False
        )
        pd.DataFrame(result["sleeve_returns"]).rename_axis("date").to_csv(
            output / f"sleeve_returns_{int(leverage)}x.csv"
        )
        rows.append({
            "leverage": leverage,
            **{f"oos__{k}": v for k, v in result["validation"]["metrics"].items()},
            **{f"check__{k}": v for k, v in result["validation"]["checks"].items()},
            "strict_oos_passed": result["validation"]["passed"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(output / "strict_oos_summary.csv", index=False)
    pd.DataFrame([monte_carlo]).to_csv(output / "monte_carlo_1x.csv", index=False)
    regimes.to_csv(output / "regime_metrics_1x.csv", index=False)
    correlation.to_csv(output / "sleeve_correlation_1x.csv")

    print("Sample:", len(mag.align_sleeves(prices_by_symbol)))
    print(summary[[
        "leverage", "oos__cagr", "oos__sharpe_ratio", "oos__max_drawdown",
        "oos__profit_factor", "check__walk_forward_passed", "strict_oos_passed"
    ]].to_string(index=False))
    print("MONTE_CARLO_1x", monte_carlo)
    print("CORRELATION\n", correlation.to_string())
    print(regimes[[
        "family", "regime", "observations", "cagr", "sharpe_ratio",
        "max_drawdown", "profit_factor"
    ]].to_string(index=False))
    for symbol in prices_by_symbol:
        m = strict_validation.metrics_for_returns(results[1.0]["sleeve_returns"][symbol])
        print(f"Standalone sleeve {symbol}: CAGR={m['cagr']:.4f} Sharpe={m['sharpe_ratio']:.4f} MaxDD={m['max_drawdown']:.4f}")


if __name__ == "__main__":
    main()
