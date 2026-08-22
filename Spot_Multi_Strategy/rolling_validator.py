import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig, validate_config
from config_io import load_strategy_config
from data import load_or_download_klines, to_utc_timestamp
from engine import run_backtest
from metrics import calculate_metrics
from optimizer import build_validation_folds
from strategies import generate_signals


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="固定策略参数的滚动历史验证"
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
        "--strategy-config",
        default="configs/regime_switching.json"
    )
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    return parser.parse_args()


def summarize_rolling_folds(fold_df):
    return {
        "fold_count": int(len(fold_df)),
        "profitable_fold_ratio": float(
            (fold_df["total_return"] > 0).mean()
        ),
        "median_fold_return": float(
            fold_df["total_return"].median()
        ),
        "worst_fold_return": float(
            fold_df["total_return"].min()
        ),
        "median_sharpe": float(
            fold_df["sharpe_ratio"].median()
        ),
        "worst_max_drawdown": float(
            fold_df["max_drawdown"].min()
        ),
        "total_trades": int(fold_df["trade_count"].sum())
    }


def summarize_strategy_folds(strategy_fold_df):
    strategy_summary = (
        strategy_fold_df.groupby("strategy")
        .agg(
            trade_count=("trade_count", "sum"),
            net_profit=("net_profit", "sum"),
            profitable_fold_ratio=(
                "net_profit",
                lambda values: (values > 0).mean()
            )
        )
        .reset_index()
    )

    return strategy_summary


def evaluate_rolling_gate(summary, strategy_summary=None):
    checks = {
        "盈利折比例不低于67%": (
            summary["profitable_fold_ratio"] >= 0.67
        ),
        "滚动折收益中位数为正": (
            summary["median_fold_return"] > 0
        ),
        "滚动折Sharpe中位数不低于0.50": (
            summary["median_sharpe"] >= 0.50
        ),
        "最差最大回撤不超过20%": (
            summary["worst_max_drawdown"] >= -0.20
        ),
        "完整交易总数不少于60笔": (
            summary["total_trades"] >= 60
        )
    }

    if strategy_summary is not None:
        checks["每个启用策略至少20笔完整交易"] = bool(
            (strategy_summary["trade_count"] >= 20).all()
        )
        checks["每个启用策略累计净利润为正"] = bool(
            (strategy_summary["net_profit"] > 0).all()
        )

    return {
        "passed": all(checks.values()),
        "checks": checks
    }


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    strategy_config_path = Path(args.strategy_config)

    if not strategy_config_path.is_absolute():
        strategy_config_path = project_folder / strategy_config_path

    strategy_config = load_strategy_config(strategy_config_path)
    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk
    )
    validate_config(strategy_config, backtest_config)

    fold_records = []
    strategy_fold_records = []

    for symbol in args.symbols:
        raw_df = load_or_download_klines(
            symbol=symbol,
            interval=args.interval,
            start_time=args.start,
            end_time=to_utc_timestamp(args.end),
            cache_folder=project_folder / "data_cache"
        )
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
            fold_records.append(
                {
                    "symbol": symbol.upper(),
                    "fold": fold["fold"],
                    "start_time": fold_data.iloc[1]["open_time"],
                    "end_time": fold_data.iloc[-1]["open_time"],
                    **metrics
                }
            )

            for sleeve in sleeves:
                if result.trades.empty:
                    strategy_trades = result.trades
                else:
                    strategy_trades = result.trades[
                        result.trades["strategy"] == sleeve.name
                    ]

                strategy_fold_records.append(
                    {
                        "symbol": symbol.upper(),
                        "fold": fold["fold"],
                        "strategy": sleeve.name,
                        "trade_count": len(strategy_trades),
                        "net_profit": (
                            float(strategy_trades["net_profit"].sum())
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

    report_folder = project_folder / "rolling_reports"
    report_folder.mkdir(parents=True, exist_ok=True)
    config_name = strategy_config_path.stem
    fold_df.to_csv(
        report_folder / f"{config_name}_folds.csv",
        index=False
    )
    strategy_fold_df.to_csv(
        report_folder / f"{config_name}_strategy_folds.csv",
        index=False
    )
    strategy_summary.to_csv(
        report_folder / f"{config_name}_strategies.csv",
        index=False
    )

    report_summary = {
        "warning": (
            "这些历史区间已经被查看，只是滚动研究验证，"
            "不是新的隐藏测试。"
        ),
        "symbols": [symbol.upper() for symbol in args.symbols],
        "interval": args.interval,
        "start": args.start,
        "end": args.end,
        "strategy_config": str(strategy_config_path),
        **summary,
        "strategies": strategy_summary.to_dict("records"),
        "gate_passed": gate_result["passed"],
        "checks": gate_result["checks"]
    }

    with (report_folder / f"{config_name}_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(report_summary, file, ensure_ascii=False, indent=2)

    print("\n滚动验证结果（不是新的隐藏测试）")
    print(f"盈利折比例：{summary['profitable_fold_ratio'] * 100:.2f}%")
    print(f"折收益中位数：{summary['median_fold_return'] * 100:.2f}%")
    print(f"最差折收益：{summary['worst_fold_return'] * 100:.2f}%")
    print(f"Sharpe中位数：{summary['median_sharpe']:.2f}")
    print(
        "最差最大回撤："
        f"{summary['worst_max_drawdown'] * 100:.2f}%"
    )
    print(f"完整交易总数：{summary['total_trades']}")

    print("\n子策略滚动统计：")

    for row in strategy_summary.itertuples(index=False):
        print(
            f"{row.strategy}：{row.trade_count}笔，"
            f"累计盈亏{row.net_profit:.2f} USDT，"
            f"盈利折比例{row.profitable_fold_ratio * 100:.2f}%"
        )

    for name, passed in gate_result["checks"].items():
        status = "通过" if passed else "未通过"
        print(f"{status}：{name}")

    if gate_result["passed"]:
        print("结论：通过滚动研究门槛，仍须等待未来数据和模拟盘。")
    else:
        print("结论：未通过滚动研究门槛，禁止实盘。")

    print(f"报告位置：{report_folder}")


if __name__ == "__main__":
    main()
