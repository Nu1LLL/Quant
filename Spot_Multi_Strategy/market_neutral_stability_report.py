"""Experimental replacement diagnostic for the PnL-concentration checks.

The decision rule was fixed before inspecting these test outputs:
42-bar circular block bootstrap, 4,999 resamples, seed 1729, one-sided
mean-greater-than-zero tests for IC and net PnL, and BH FDR control at 5%.
Historical concentration Gate results remain unchanged.
"""
import argparse
from pathlib import Path

import pandas as pd

import market_neutral as mn
from alpha_research import (
    DEFAULT_SYMBOLS,
    build_alpha_sets,
    load_funding_frames,
    load_symbol_frames,
)
from market_neutral_gate_report import list_candidate_alpha_names
from stability_significance import (
    benjamini_hochberg,
    circular_block_bootstrap_mean_pvalue,
    cross_sectional_ic_series,
)


def build_diagnostic_series(
    alpha_name, frames, alpha_sets, fee, slippage,
    no_trade_band, method, rebalance_every_bars
):
    signal_matrix = mn.build_cross_sectional_signal(
        frames, alpha_sets, alpha_name, method=method
    )
    forward_matrix = mn.build_cross_sectional_forward_return(
        frames, delay=1, horizon=1
    )
    symbols = [c for c in signal_matrix.columns if c in forward_matrix.columns]
    signal_matrix = signal_matrix[symbols]
    forward_matrix = forward_matrix[symbols]

    capped = mn.scale_to_gross_cap(signal_matrix, gross_cap=1.0)
    scheduled = mn.resample_to_rebalance_schedule(
        capped, rebalance_every_bars
    )
    exposure, turnover = mn.apply_no_trade_band(
        scheduled, no_trade_band=no_trade_band
    )
    net_by_asset = (
        exposure * forward_matrix
        - turnover * (fee + slippage) * 2.0
    )
    net_pnl = net_by_asset.sum(axis=1, skipna=True).where(
        net_by_asset.notna().sum(axis=1) > 0
    )
    ic_series = cross_sectional_ic_series(signal_matrix, forward_matrix)
    return ic_series, net_pnl


def run_stability_gate(
    frames, alpha_sets, fee=0.001, slippage=0.0005,
    no_trade_band=0.05, method="demean", rebalance_every_bars=42,
    block_length=42, repetitions=4999, fdr=0.05
):
    names = list_candidate_alpha_names(alpha_sets, len(frames))
    records = []
    for alpha_name in names:
        historical = mn.evaluate_cross_sectional_alpha(
            alpha_name, frames, alpha_sets, fee_rate=fee,
            slippage_rate=slippage, no_trade_band=no_trade_band,
            method=method, rebalance_every_bars=rebalance_every_bars
        )
        ic_series, net_pnl = build_diagnostic_series(
            alpha_name, frames, alpha_sets, fee, slippage,
            no_trade_band, method, rebalance_every_bars
        )
        records.append({
            "alpha_name": alpha_name,
            "median_fold_ic": historical.summary["median_fold_ic"],
            "positive_fold_ratio": historical.summary["positive_fold_ratio"],
            "cost_drag_ratio": historical.summary["cost_drag_ratio"],
            "total_observations": historical.summary["total_observations"],
            "min_fold_observations": historical.summary["min_fold_observations"],
            "max_fold_pnl_ratio_diagnostic": historical.summary["max_fold_pnl_ratio"],
            "max_year_pnl_ratio_diagnostic": historical.summary["max_year_pnl_ratio"],
            "mean_timestamp_ic": float(ic_series.mean()),
            "mean_net_pnl": float(net_pnl.mean()),
            "ic_bootstrap_p": circular_block_bootstrap_mean_pvalue(
                ic_series, block_length, repetitions, seed=1729
            ),
            "pnl_bootstrap_p": circular_block_bootstrap_mean_pvalue(
                net_pnl, block_length, repetitions, seed=1729
            ),
        })

    result = pd.DataFrame(records)
    result["ic_bh_q"] = benjamini_hochberg(result["ic_bootstrap_p"])
    result["pnl_bh_q"] = benjamini_hochberg(result["pnl_bootstrap_p"])
    result["check__median_fold_ic_positive"] = result["median_fold_ic"] > 0
    result["check__positive_fold_ratio_60pct"] = (
        result["positive_fold_ratio"] >= 0.60
    )
    result["check__cost_drag_70pct"] = result["cost_drag_ratio"] <= 0.70
    result["check__total_observations_1000"] = (
        result["total_observations"] >= 1000
    )
    result["check__min_fold_observations_30"] = (
        result["min_fold_observations"] >= 30
    )
    result["check__ic_fdr_5pct"] = result["ic_bh_q"] <= fdr
    result["check__pnl_fdr_5pct"] = result["pnl_bh_q"] <= fdr
    checks = [c for c in result.columns if c.startswith("check__")]
    result["experimental_passed"] = result[checks].all(axis=1)
    return result


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-09-11")
    parser.add_argument("--block-length", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=4999)
    parser.add_argument(
        "--output-folder",
        default="reports/mini_medallion_stability_significance",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    root = Path(__file__).resolve().parent
    end = pd.Timestamp(args.end, tz="UTC")
    frames = load_symbol_frames(
        DEFAULT_SYMBOLS, "4h", args.start, end, root / "data_cache"
    )
    funding = load_funding_frames(
        DEFAULT_SYMBOLS, args.start, end, root / "futures_data_cache"
    )
    alpha_sets = build_alpha_sets(frames, funding_frames=funding)
    result = run_stability_gate(
        frames, alpha_sets, block_length=args.block_length,
        repetitions=args.repetitions
    )
    output = root / args.output_folder
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "stability_significance_results.csv", index=False)
    ordered = result.sort_values(
        ["experimental_passed", "pnl_bh_q"], ascending=[False, True]
    )
    print(ordered.to_string(index=False))
    print(f"\nExperimental passes: {int(result['experimental_passed'].sum())}")


if __name__ == "__main__":
    main()
