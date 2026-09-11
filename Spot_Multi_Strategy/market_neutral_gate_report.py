"""跑通所有alpha家族的横截面市场中性walk-forward Gate，产出
accepted/rejected列表——完全独立于alpha_gate_report.py（长仓专用），
不修改、不复用它的任何状态。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import market_neutral as mn
from alpha_research import (
    DEFAULT_SYMBOLS,
    build_alpha_sets,
    load_funding_frames,
    load_symbol_frames
)
from data import INTERVAL_TO_TIMEDELTA, to_utc_timestamp


def _resolve_end_time(interval, end):
    if end is not None:
        return to_utc_timestamp(end)
    now = pd.Timestamp.now(tz="UTC")
    interval_seconds = int(INTERVAL_TO_TIMEDELTA[interval].total_seconds())
    now_seconds = int(now.timestamp())
    closed_boundary_seconds = now_seconds // interval_seconds * interval_seconds
    return pd.Timestamp(closed_boundary_seconds, unit="s", tz="UTC")


def list_candidate_alpha_names(alpha_sets, min_symbol_coverage=3):
    """只测试至少在min_symbol_coverage个品种上都存在的alpha家族名称
    （比如A18只属于ETHUSDT，横截面对比没有意义，自动跳过）。
    """
    counts = {}
    for symbol, signals in alpha_sets.items():
        for name in signals:
            counts[name] = counts.get(name, 0) + 1
    return sorted(
        name for name, count in counts.items()
        if count >= min_symbol_coverage
    )


def run_cross_sectional_gate(
    frames, alpha_sets, fold_count=6, fee=0.001, slippage=0.0005,
    no_trade_band=0.05, method="demean", rebalance_every_bars=1
):
    alpha_names = list_candidate_alpha_names(alpha_sets, len(frames))
    gate_records = []
    fold_tables = []

    for alpha_name in alpha_names:
        result = mn.evaluate_cross_sectional_alpha(
            alpha_name, frames, alpha_sets,
            fold_count=fold_count, fee_rate=fee, slippage_rate=slippage,
            no_trade_band=no_trade_band, method=method,
            rebalance_every_bars=rebalance_every_bars
        )
        fold_table = result.fold_table.copy()
        fold_tables.append(fold_table)
        gate_records.append({
            "alpha_name": alpha_name,
            "passed": result.passed,
            **result.summary,
            **{
                f"check__{key}": value
                for key, value in result.checks.items()
            }
        })

    gate_df = pd.DataFrame(gate_records)
    fold_df = (
        pd.concat(fold_tables, ignore_index=True)
        if fold_tables else pd.DataFrame()
    )
    return gate_df, fold_df


def summarize_acceptance(gate_df):
    if gate_df.empty:
        return [], []
    passed = gate_df[gate_df["passed"]]
    accepted = sorted(passed["alpha_name"].tolist())
    rejected = sorted(
        gate_df[~gate_df["passed"]]["alpha_name"].tolist()
    )
    return accepted, rejected


def rejection_reasons(gate_df, alpha_name):
    row = gate_df[gate_df["alpha_name"] == alpha_name].iloc[0]
    check_columns = [c for c in gate_df.columns if c.startswith("check__")]
    # 用bool()显式转换再比较，不要用`is False`——.iloc取出的单元格是
    # numpy.bool_，和Python内建的False不是同一个对象，`is`比较会
    # 静默地永远判定不成立（这是实测踩到的坑，不是假设）。
    return [
        column.replace("check__", "")
        for column in check_columns
        if bool(row[column]) is False
    ]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="市场中性横截面Alpha Research Gate"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--no-trade-band", type=float, default=0.05)
    parser.add_argument(
        "--method", choices=["demean", "rank"], default="demean"
    )
    parser.add_argument(
        "--rebalance-every-bars", type=int, default=1,
        help="每隔多少根K线才更新一次目标（1=逐根K线跟踪，"
             "42=4小时K线的周频，更接近动量/carry类因子文献里"
             "的实际再平衡频率）"
    )
    parser.add_argument("--no-funding", action="store_true")
    parser.add_argument("--output-folder", default="reports/mini_medallion_market_neutral")
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    end_time = _resolve_end_time(args.interval, args.end)

    frames = load_symbol_frames(
        symbols=args.symbols, interval=args.interval,
        start=args.start, end=end_time,
        cache_folder=project_folder / "data_cache"
    )
    funding_frames = (
        {} if args.no_funding
        else load_funding_frames(
            symbols=args.symbols, start=args.start, end=end_time,
            cache_folder=project_folder / "futures_data_cache"
        )
    )
    alpha_sets = build_alpha_sets(frames, funding_frames=funding_frames)

    gate_df, fold_df = run_cross_sectional_gate(
        frames, alpha_sets, args.folds, args.fee, args.slippage,
        args.no_trade_band, method=args.method,
        rebalance_every_bars=args.rebalance_every_bars
    )
    accepted, rejected = summarize_acceptance(gate_df)

    output_folder = project_folder / args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)
    gate_df.to_csv(output_folder / "cross_sectional_gate_results.csv", index=False)
    fold_df.to_csv(output_folder / "cross_sectional_gate_folds.csv", index=False)

    rejection_summary = {name: rejection_reasons(gate_df, name) for name in rejected}
    with (output_folder / "cross_sectional_gate_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "fold_count": args.folds,
                "no_trade_band": args.no_trade_band,
                "symbols": args.symbols,
                "accepted_alphas": accepted,
                "rejected_alphas": rejected,
                "rejection_reasons": rejection_summary
            },
            file, ensure_ascii=False, indent=2
        )

    print(f"市场中性横截面Gate：{args.folds}折walk-forward验证，{len(gate_df)}个alpha家族")
    print(f"通过：{len(accepted)}个 -> {accepted}")
    print(f"未通过：{len(rejected)}个")
    for name in rejected:
        print(f"  未通过：{name} -> {rejection_summary[name]}")
    print(f"\n报告已保存到：{output_folder}")


if __name__ == "__main__":
    main()
