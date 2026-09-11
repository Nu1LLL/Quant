"""跑通所有alpha的walk-forward Research Gate，产出accepted/rejected列表。

一个alpha必须在它被评估过的每一个品种上都通过门槛才算accepted——
不允许只挑表现好的品种（例如只在ETH上通过就宣称"通过"）。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import alpha_metrics
import walk_forward
from alpha_research import (
    DEFAULT_SYMBOLS,
    build_alpha_sets,
    load_funding_frames,
    load_symbol_frames
)
from data import INTERVAL_TO_TIMEDELTA, to_utc_timestamp

REGIMES = ("trending", "mixed", "ranging")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Alpha Research Gate（多折walk-forward验收）"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument(
        "--no-funding", action="store_true",
        help="跳过资金费率alpha"
    )
    parser.add_argument("--output-folder", default="alpha_reports")
    parser.add_argument(
        "--regime-conditional", action="store_true",
        help="额外跑一次regime条件门槛（每个alpha分别在trending/mixed/"
             "ranging子段上评估，只有子段内真正有效才被license）"
    )
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


def compute_regime_masks(frames):
    """{symbol: {regime: 布尔mask}}，全部用因果的
    compute_causal_regime_labels，可以安全用于regime条件评估。
    """
    masks = {}
    for symbol, df in frames.items():
        labels = alpha_metrics.compute_causal_regime_labels(df)
        masks[symbol] = {
            regime: (labels == regime) for regime in REGIMES
        }
    return masks


def run_regime_conditional_gate(frames, alpha_sets, folds, fee, slippage):
    """对每个alpha×品种×regime单独跑一次walk-forward门槛。

    一个alpha即使无条件评估失败，只要在它真正擅长的regime子集里
    单独评估能通过（且在所有被评估的品种上都通过），就会被记录为
    这个regime的"专家alpha"，供build_regime_conditional_ensemble
    在实际敞口只在对应regime里激活它。
    """
    regime_masks = compute_regime_masks(frames)

    gate_records = []
    fold_tables = []

    for symbol, alpha_signals in alpha_sets.items():
        df = frames[symbol]
        for regime in REGIMES:
            mask = regime_masks[symbol][regime]
            for name, alpha_signal in alpha_signals.items():
                result = walk_forward.evaluate_alpha_walk_forward(
                    symbol, alpha_signal, df,
                    fold_count=folds, fee_rate=fee, slippage_rate=slippage,
                    activation_mask=mask
                )
                fold_table = result.fold_table.copy()
                fold_table.insert(0, "regime", regime)
                fold_tables.append(fold_table)

                gate_records.append({
                    "symbol": symbol,
                    "alpha_name": name,
                    "regime": regime,
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

    return gate_df, fold_df


def summarize_regime_acceptance(regime_gate_df):
    """{regime: [通过该regime条件门槛、且在所有测试品种上都通过的alpha]}。"""
    regime_map = {}
    for regime in REGIMES:
        subset = regime_gate_df[regime_gate_df["regime"] == regime]
        per_alpha = subset.groupby("alpha_name")["passed"].all()
        regime_map[regime] = sorted(
            per_alpha[per_alpha].index.tolist()
        )
    return regime_map


def summarize_per_asset_acceptance(gate_df):
    """{品种: [只在这一个品种上通过walk-forward门槛的alpha]}。

    和summarize_acceptance()的"必须在所有测试品种上都通过"不同，
    这里承认一个alpha的有效性可能本来就是资产特定的（不同资产的
    流动性、参与者结构、波动率特征不一样），只要它在**这个资产自己
    的**6折walk-forward历史上通过了和其他alpha完全相同的6项标准，
    就在这个资产上被license——用的还是同一套门槛，不是放宽标准，
    只是不强制要求"必须放之四海而皆准"。
    """
    per_asset_map = {}
    for symbol, subset in gate_df.groupby("symbol"):
        passed = subset[subset["passed"]]
        per_asset_map[symbol] = sorted(passed["alpha_name"].tolist())
    return per_asset_map


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
    funding_frames = (
        {} if args.no_funding
        else load_funding_frames(
            symbols=args.symbols, start=args.start, end=end_time,
            cache_folder=project_folder / "futures_data_cache"
        )
    )
    alpha_sets = build_alpha_sets(frames, funding_frames=funding_frames)

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

    if args.regime_conditional:
        print("\n正在运行regime条件门槛（每个alpha分regime单独评估）...")
        regime_gate_df, regime_fold_df = run_regime_conditional_gate(
            frames, alpha_sets, args.folds, args.fee, args.slippage
        )
        regime_map = summarize_regime_acceptance(regime_gate_df)

        regime_gate_df.to_csv(
            output_folder / "regime_conditional_gate_results.csv",
            index=False
        )
        regime_fold_df.to_csv(
            output_folder / "regime_conditional_gate_folds.csv",
            index=False
        )
        with (output_folder / "regime_conditional_gate_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {"fold_count": args.folds, "regime_alpha_map": regime_map},
                file, ensure_ascii=False, indent=2
            )

        print("regime条件门槛结果（在所有测试品种上都通过才算license）：")
        for regime, names in regime_map.items():
            print(f"  {regime}：{len(names)}个 -> {names}")
        print(f"regime条件报告已保存到：{output_folder}")


if __name__ == "__main__":
    main()
