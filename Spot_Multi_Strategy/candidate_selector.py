import argparse
import json
from pathlib import Path

import pandas as pd

from config import BacktestConfig, validate_config
from config_io import load_strategy_config, save_strategy_config
from data import load_or_download_klines, to_utc_timestamp
from engine import run_backtest
from metrics import calculate_metrics
from optimizer import build_validation_folds
from robustness import build_parameter_variants
from rolling_validator import (
    evaluate_rolling_gate,
    summarize_rolling_folds,
    summarize_strategy_folds
)
from strategies import generate_signals


def calculate_candidate_score(summary):
    # 不追求最高总收益，优先考虑一般折、最差折和风险调整表现
    return float(
        summary["median_sharpe"]
        +
        4.0 * summary["median_fold_return"]
        +
        0.50 * summary["profitable_fold_ratio"]
        +
        2.0 * summary["worst_fold_return"]
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="从保守参数邻域选择滚动稳定候选"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument(
        "--base-config",
        default="configs/trend_conservative.json"
    )
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    base_config_path = Path(args.base_config)

    if not base_config_path.is_absolute():
        base_config_path = project_folder / base_config_path

    base_config = load_strategy_config(base_config_path)
    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk
    )
    variants = build_parameter_variants(base_config)
    raw_data_by_symbol = {
        symbol.upper(): load_or_download_klines(
            symbol=symbol,
            interval=args.interval,
            start_time=args.start,
            end_time=to_utc_timestamp(args.end),
            cache_folder=project_folder / "data_cache"
        )
        for symbol in args.symbols
    }

    candidate_records = []
    all_fold_records = []

    for variant_name, strategy_config in variants.items():
        validate_config(strategy_config, backtest_config)
        fold_records = []
        strategy_fold_records = []

        for symbol, raw_df in raw_data_by_symbol.items():
            signal_df, sleeves = generate_signals(
                raw_df,
                strategy_config
            )
            folds = build_validation_folds(
                data_length=len(raw_df),
                development_end=len(raw_df),
                fold_count=args.folds
            )

            for fold in folds:
                fold_data = signal_df.iloc[
                    fold["start"] - 1:fold["end"]
                ].reset_index(drop=True)
                result = run_backtest(
                    fold_data,
                    sleeves,
                    backtest_config
                )
                metrics = calculate_metrics(result)
                fold_record = {
                    "variant": variant_name,
                    "symbol": symbol,
                    "fold": fold["fold"],
                    **metrics
                }
                fold_records.append(fold_record)
                all_fold_records.append(fold_record)

                for sleeve in sleeves:
                    strategy_trades = (
                        result.trades[
                            result.trades["strategy"] == sleeve.name
                        ]
                        if not result.trades.empty
                        else result.trades
                    )
                    strategy_fold_records.append(
                        {
                            "strategy": sleeve.name,
                            "trade_count": len(strategy_trades),
                            "net_profit": (
                                float(
                                    strategy_trades["net_profit"].sum()
                                )
                                if not strategy_trades.empty
                                else 0.0
                            )
                        }
                    )

        fold_df = pd.DataFrame(fold_records)
        strategy_fold_df = pd.DataFrame(strategy_fold_records)
        summary = summarize_rolling_folds(fold_df)
        strategy_summary = summarize_strategy_folds(
            strategy_fold_df
        )
        gate_result = evaluate_rolling_gate(
            summary,
            strategy_summary
        )

        candidate_records.append(
            {
                "variant": variant_name,
                "score": calculate_candidate_score(summary),
                "gate_passed": gate_result["passed"],
                **summary
            }
        )

    candidate_df = pd.DataFrame(candidate_records).sort_values(
        ["gate_passed", "score"],
        ascending=[False, False]
    )
    fold_df = pd.DataFrame(all_fold_records)
    report_folder = project_folder / "candidate_reports"
    report_folder.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(
        report_folder / "trend_neighborhood_ranking.csv",
        index=False
    )
    fold_df.to_csv(
        report_folder / "trend_neighborhood_folds.csv",
        index=False
    )

    passing_candidates = candidate_df[candidate_df["gate_passed"]]

    if passing_candidates.empty:
        selected_variant = None
        selected_config = None
    else:
        selected_variant = passing_candidates.iloc[0]["variant"]
        selected_config = variants[selected_variant]
        save_strategy_config(
            selected_config,
            project_folder / "configs" / "trend_champion.json"
        )

    summary = {
        "warning": (
            "参数邻域和时间折均已被查看，选出的只是模拟盘候选，"
            "不是经过新隐藏数据验证的实盘策略。"
        ),
        "candidate_count": len(candidate_df),
        "passing_candidate_count": int(
            candidate_df["gate_passed"].sum()
        ),
        "selected_variant": selected_variant,
        "selected_config": (
            "configs/trend_champion.json"
            if selected_config is not None
            else None
        )
    }

    with (report_folder / "selection_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n保守参数邻域滚动筛选")
    print(
        candidate_df[
            [
                "variant",
                "gate_passed",
                "score",
                "profitable_fold_ratio",
                "median_fold_return",
                "median_sharpe",
                "worst_fold_return"
            ]
        ].to_string(index=False)
    )

    if selected_config is None:
        print("结论：没有候选通过滚动门槛，策略家族淘汰。")
    else:
        print(f"入选模拟盘候选：{selected_variant}")
        print("参数文件：configs/trend_champion.json")
        print("注意：仍须通过该参数自身的压力测试和未来模拟盘。")


if __name__ == "__main__":
    main()
