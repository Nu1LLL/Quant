"""跑通所有alpha的walk-forward Research Gate，产出accepted/rejected列表。

一个alpha必须在它被评估过的每一个品种上都通过门槛才算accepted——
不允许只挑表现好的品种（例如只在ETH上通过就宣称"通过"）。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import walk_forward
from alpha_research import build_alpha_sets, load_symbol_frames
from data import INTERVAL_TO_TIMEDELTA, to_utc_timestamp


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Alpha Research Gate（多折walk-forward验收）"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--output-folder", default="alpha_reports")
    return parser.parse_args()


def _resolve_end_time(interval, end):
    if end is not None:
        return to_utc_timestamp(end)

    now = pd.Timestamp.now(tz="UTC")
    interval_seconds = int(INTERVAL_TO_TIMEDELTA[interval].total_seconds())
    now_seconds = int(now.timestamp())
    closed_boundary_seconds = now_seconds // interval_seconds * interval_seconds
    return pd.Timestamp(closed_boundary_seconds, unit="s", tz="UTC")


def run_gate(frames, alpha_sets, folds, fee, slippage):
    gate_records = []
    fold_tables = []
    per_alpha_results = {}

    for symbol, alpha_signals in alpha_sets.items():
        df = frames[symbol]
        for name, alpha_signal in alpha_signals.items():
            result = walk_forward.evaluate_alpha_walk_forward(
                symbol, alpha_signal, df,
                fold_count=folds, fee_rate=fee, slippage_rate=slippage
            )
            per_alpha_results[(symbol, name)] = result
            fold_tables.append(result.fold_table)

            gate_records.append({
                "symbol": symbol,
                "alpha_name": name,
                "direction": alpha_signal.direction,
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
        if fold_tables
        else pd.DataFrame()
    )

    return gate_df, fold_df, per_alpha_results


def summarize_acceptance(gate_df):
    """一个alpha必须在所有被评估的品种上都通过才算accepted。"""
    per_alpha = gate_df.groupby("alpha_name")["passed"].all()
    accepted = sorted(per_alpha[per_alpha].index.tolist())
    rejected = sorted(per_alpha[~per_alpha].index.tolist())
    return accepted, rejected


def rejection_reasons(gate_df, alpha_name):
    subset = gate_df[gate_df["alpha_name"] == alpha_name]
    check_columns = [
        column for column in subset.columns
        if column.startswith("check__")
    ]

    reasons = []
    for _, row in subset.iterrows():
        failed_checks = [
            column.replace("check__", "")
            for column in check_columns
            if row[column] is False
        ]
        if failed_checks:
            reasons.append(f"{row['symbol']}: {', '.join(failed_checks)}")
    return reasons


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    end_time = _resolve_end_time(args.interval, args.end)

    frames = load_symbol_frames(
        symbols=args.symbols,
        interval=args.interval,
        start=args.start,
        end=end_time,
        cache_folder=project_folder / "data_cache"
    )
    alpha_sets = build_alpha_sets(frames)

    gate_df, fold_df, _ = run_gate(
        frames, alpha_sets, args.folds, args.fee, args.slippage
    )
    accepted, rejected = summarize_acceptance(gate_df)

    output_folder = project_folder / args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    gate_df.to_csv(output_folder / "alpha_gate_results.csv", index=False)
    fold_df.to_csv(output_folder / "alpha_gate_folds.csv", index=False)

    rejection_summary = {
        name: rejection_reasons(gate_df, name) for name in rejected
    }

    with (output_folder / "alpha_gate_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            {
                "fold_count": args.folds,
                "symbols": list(alpha_sets.keys()),
                "accepted_alphas": accepted,
                "rejected_alphas": rejected,
                "rejection_reasons": rejection_summary
            },
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"Alpha Research Gate：{args.folds}折walk-forward验证")
    print(f"通过（在所有测试品种上都通过）：{len(accepted)}个")
    for name in accepted:
        print(f"  通过：{name}")
    print(f"未通过：{len(rejected)}个")
    for name in rejected:
        print(f"  未通过：{name} -> {rejection_summary[name]}")

    print(f"\n报告已保存到：{output_folder}")


if __name__ == "__main__":
    main()
