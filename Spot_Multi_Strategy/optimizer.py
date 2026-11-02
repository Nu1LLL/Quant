import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig, StrategyConfig, validate_config
from config_io import save_strategy_config
from data import load_or_download_klines, to_utc_timestamp
from engine import run_backtest
from metrics import (
    calculate_metrics,
    evaluate_research_gates,
    format_metrics,
    format_research_gates
)
from strategies import generate_signals


@dataclass(frozen=True)
class OptimizationConfig:
    # 搜索多少组候选参数
    trial_count: int = 60

    # 固定随机种子，保证结果可以复现
    random_seed: int = 20260823

    # 最后20%数据完全不参与参数搜索
    development_fraction: float = 0.80

    # 开发区间分成几个连续验证折
    fold_count: int = 4

    # 每个品种、每个验证折至少需要的交易数
    minimum_fold_trades: int = 3


def build_validation_folds(
    data_length,
    development_end,
    fold_count,
    warmup_bars=500
):
    # 开发区间前40%只用于给指标预热，后60%切成连续验证折
    validation_start = max(
        warmup_bars,
        int(development_end * 0.40)
    )

    if validation_start >= development_end - fold_count:
        raise ValueError("数据太少，无法建立时间序列验证折")

    boundaries = np.linspace(
        validation_start,
        development_end,
        fold_count + 1,
        dtype=int
    )

    folds = []

    for fold_number in range(fold_count):
        start_position = int(boundaries[fold_number])
        end_position = int(boundaries[fold_number + 1])

        if end_position - start_position < 2:
            raise ValueError("验证折长度不足两根K线")

        folds.append(
            {
                "fold": fold_number + 1,
                "start": start_position,
                "end": end_position
            }
        )

    return folds


def generate_candidate_configs(trial_count, random_seed):
    # 从预先限制的合理范围内抽取候选参数
    if trial_count < 2:
        raise ValueError("候选参数数量至少为2")

    random_generator = random.Random(random_seed)
    baseline = StrategyConfig()
    candidates = [baseline]

    # 把只使用趋势突破的基线也固定加入候选
    candidates.append(
        replace(
            baseline,
            trend_weight=1.0,
            pullback_weight=0.0
        )
    )

    seen_parameters = {
        tuple(asdict(candidate).items())
        for candidate in candidates
    }

    while len(candidates) < trial_count:
        use_pullback = random_generator.choice([False, False, True])

        if use_pullback:
            trend_weight = random_generator.choice([0.70, 0.80, 0.90])
            pullback_weight = 1.0 - trend_weight
            pullback_entry_rsi = random_generator.choice([25.0, 30.0, 35.0])
            bollinger_std = random_generator.choice([1.5, 2.0, 2.5])
            pullback_stop = random_generator.choice([1.5, 2.0, 2.5])
        else:
            trend_weight = 1.0
            pullback_weight = 0.0
            pullback_entry_rsi = baseline.pullback_entry_rsi
            bollinger_std = baseline.bollinger_std_multiple
            pullback_stop = baseline.pullback_stop_atr_multiple

        candidate = StrategyConfig(
            trend_weight=trend_weight,
            pullback_weight=pullback_weight,
            ema_window=random_generator.choice([100, 150, 200, 250]),
            ema_slope_window=random_generator.choice([5, 10, 20]),
            breakout_entry_window=random_generator.choice([20, 30, 40, 55]),
            breakout_exit_window=random_generator.choice([5, 10, 15, 20]),
            atr_window=14,
            trend_stop_atr_multiple=random_generator.choice(
                [1.5, 2.0, 2.5, 3.0]
            ),
            trend_trailing_atr_multiple=random_generator.choice(
                [2.0, 2.5, 3.0, 3.5, 4.0]
            ),
            rsi_window=14,
            pullback_entry_rsi=pullback_entry_rsi,
            pullback_exit_rsi=55.0,
            bollinger_window=20,
            bollinger_std_multiple=bollinger_std,
            pullback_stop_atr_multiple=pullback_stop
        )

        parameter_key = tuple(asdict(candidate).items())

        if parameter_key not in seen_parameters:
            seen_parameters.add(parameter_key)
            candidates.append(candidate)

    return candidates


def calculate_fold_score(metrics, minimum_trades):
    # 同时奖励风险调整收益，并惩罚回撤和交易样本不足
    finite_profit_factor = min(
        float(metrics["profit_factor"]),
        3.0
    )

    score = (
        float(metrics["sharpe_ratio"])
        +
        2.0 * float(metrics["annualized_return"])
        +
        0.35 * (finite_profit_factor - 1.0)
        +
        2.0 * float(metrics["max_drawdown"])
    )

    if metrics["total_return"] <= 0:
        score -= 0.50

    if metrics["trade_count"] < minimum_trades:
        missing_trades = minimum_trades - metrics["trade_count"]
        score -= 0.20 * missing_trades

    return score


def aggregate_candidate_score(fold_records):
    scores = np.array(
        [record["fold_score"] for record in fold_records],
        dtype=float
    )
    profitable_ratio = np.mean(
        [record["total_return"] > 0 for record in fold_records]
    )

    # 中位数代表一般情况，最差折和波动惩罚不稳定参数
    return float(
        np.median(scores)
        +
        0.35 * np.min(scores)
        -
        0.25 * np.std(scores)
        +
        0.20 * profitable_ratio
    )


def evaluate_candidate(
    raw_data_by_symbol,
    strategy_config,
    backtest_config,
    optimization_config
):
    fold_records = []

    for symbol, raw_df in raw_data_by_symbol.items():
        development_end = int(
            len(raw_df) * optimization_config.development_fraction
        )
        development_df = raw_df.iloc[:development_end].copy()
        signal_df, sleeves = generate_signals(
            development_df,
            strategy_config
        )
        folds = build_validation_folds(
            data_length=len(raw_df),
            development_end=development_end,
            fold_count=optimization_config.fold_count
        )

        for fold in folds:
            # 多保留前一根K线，以便下一根开盘执行信号
            fold_df = signal_df.iloc[
                fold["start"] - 1:fold["end"]
            ].reset_index(drop=True)

            result = run_backtest(
                signal_df=fold_df,
                sleeves=sleeves,
                config=backtest_config
            )
            metrics = calculate_metrics(result)
            fold_score = calculate_fold_score(
                metrics,
                optimization_config.minimum_fold_trades
            )

            fold_records.append(
                {
                    "symbol": symbol,
                    "fold": fold["fold"],
                    "start_time": fold_df.iloc[1]["open_time"],
                    "end_time": fold_df.iloc[-1]["open_time"],
                    "fold_score": fold_score,
                    "total_return": metrics["total_return"],
                    "annualized_return": metrics["annualized_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "profit_factor": metrics["profit_factor"],
                    "trade_count": metrics["trade_count"]
                }
            )

    return aggregate_candidate_score(fold_records), fold_records


def run_hidden_test(
    raw_data_by_symbol,
    strategy_config,
    backtest_config,
    development_fraction
):
    hidden_records = []
    hidden_results = {}

    for symbol, raw_df in raw_data_by_symbol.items():
        hidden_start = int(len(raw_df) * development_fraction)
        signal_df, sleeves = generate_signals(raw_df, strategy_config)
        hidden_df = signal_df.iloc[
            hidden_start - 1:
        ].reset_index(drop=True)

        result = run_backtest(
            signal_df=hidden_df,
            sleeves=sleeves,
            config=backtest_config
        )
        metrics = calculate_metrics(result)
        gate_result = evaluate_research_gates(metrics)

        hidden_results[symbol] = result
        hidden_records.append(
            {
                "symbol": symbol,
                **metrics,
                "gate_passed": gate_result["passed"]
            }
        )

    return pd.DataFrame(hidden_records), hidden_results


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="现货多策略时间序列稳健优化"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--trials", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    end_time = to_utc_timestamp(args.end)

    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk
    )
    optimization_config = OptimizationConfig(
        trial_count=args.trials,
        random_seed=args.seed
    )

    raw_data_by_symbol = {}

    for symbol in args.symbols:
        raw_data_by_symbol[symbol.upper()] = load_or_download_klines(
            symbol=symbol,
            interval=args.interval,
            start_time=args.start,
            end_time=end_time,
            cache_folder=project_folder / "data_cache"
        )

    candidates = generate_candidate_configs(
        trial_count=optimization_config.trial_count,
        random_seed=optimization_config.random_seed
    )

    candidate_records = []
    all_fold_records = []

    print("\n开始稳健参数搜索")
    print(f"交易品种：{', '.join(raw_data_by_symbol)}")
    print(f"候选数量：{len(candidates)}")
    print("最后20%数据：锁定，不参与搜索")

    for candidate_number, strategy_config in enumerate(
        candidates,
        start=1
    ):
        validate_config(strategy_config, backtest_config)
        score, fold_records = evaluate_candidate(
            raw_data_by_symbol=raw_data_by_symbol,
            strategy_config=strategy_config,
            backtest_config=backtest_config,
            optimization_config=optimization_config
        )

        candidate_records.append(
            {
                "candidate": candidate_number,
                "score": score,
                **asdict(strategy_config)
            }
        )

        for record in fold_records:
            all_fold_records.append(
                {
                    "candidate": candidate_number,
                    **record
                }
            )

        if candidate_number == 1 or candidate_number % 5 == 0:
            print(
                f"已完成：{candidate_number}/{len(candidates)}，"
                f"当前得分：{score:.3f}",
                flush=True
            )

    candidate_df = pd.DataFrame(candidate_records).sort_values(
        "score",
        ascending=False
    )
    fold_df = pd.DataFrame(all_fold_records)
    best_candidate_number = int(candidate_df.iloc[0]["candidate"])
    best_config = candidates[best_candidate_number - 1]

    reports_folder = project_folder / "optimization_reports"
    reports_folder.mkdir(parents=True, exist_ok=True)
    candidate_df.to_csv(
        reports_folder / "candidate_ranking.csv",
        index=False
    )
    fold_df.to_csv(
        reports_folder / "validation_folds.csv",
        index=False
    )
    save_strategy_config(
        best_config,
        reports_folder / "optimized_strategy.json"
    )

    # 参数确定以后，隐藏测试集只在这里验收一次
    hidden_df, hidden_results = run_hidden_test(
        raw_data_by_symbol=raw_data_by_symbol,
        strategy_config=best_config,
        backtest_config=backtest_config,
        development_fraction=(
            optimization_config.development_fraction
        )
    )
    hidden_df.to_csv(
        reports_folder / "hidden_test.csv",
        index=False
    )

    summary = {
        "symbols": list(raw_data_by_symbol),
        "interval": args.interval,
        "start": args.start,
        "end": args.end,
        "trial_count": len(candidates),
        "random_seed": optimization_config.random_seed,
        "development_fraction": (
            optimization_config.development_fraction
        ),
        "fold_count": optimization_config.fold_count,
        "best_candidate": best_candidate_number,
        "best_score": float(candidate_df.iloc[0]["score"]),
        "all_hidden_gates_passed": bool(
            hidden_df["gate_passed"].all()
        )
    }

    with (reports_folder / "optimization_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n最佳开发区间参数：")

    for name, value in asdict(best_config).items():
        print(f"{name}：{value}")

    print("\n隐藏测试集结果：")

    for hidden_record in hidden_df.to_dict("records"):
        symbol = hidden_record["symbol"]
        metrics = {
            key: value
            for key, value in hidden_record.items()
            if key not in {"symbol", "gate_passed"}
        }
        print(format_metrics(symbol, metrics))
        print(format_research_gates(
            evaluate_research_gates(metrics)
        ))

    if summary["all_hidden_gates_passed"]:
        print("\n最终结论：通过基础研究门槛，仍需模拟盘。")
    else:
        print("\n最终结论：隐藏测试未全部通过，禁止实盘。")

    print(f"优化报告：{reports_folder}")


if __name__ == "__main__":
    main()
