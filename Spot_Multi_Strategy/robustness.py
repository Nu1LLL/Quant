import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig, validate_config
from config_io import load_strategy_config
from data import load_or_download_klines, to_utc_timestamp
from engine import run_backtest
from metrics import calculate_metrics
from strategies import generate_signals


def build_parameter_variants(base_config):
    # 每次只扰动一个重要参数，检查附近参数是否仍然有效
    variants = {
        "base": base_config,
        "ema_minus_20pct": replace(
            base_config,
            ema_window=max(20, round(base_config.ema_window * 0.80))
        ),
        "ema_plus_20pct": replace(
            base_config,
            ema_window=round(base_config.ema_window * 1.20)
        ),
        "entry_minus_20pct": replace(
            base_config,
            breakout_entry_window=max(
                5,
                round(base_config.breakout_entry_window * 0.80)
            )
        ),
        "entry_plus_20pct": replace(
            base_config,
            breakout_entry_window=round(
                base_config.breakout_entry_window * 1.20
            )
        ),
        "exit_minus_20pct": replace(
            base_config,
            breakout_exit_window=max(
                3,
                round(base_config.breakout_exit_window * 0.80)
            )
        ),
        "exit_plus_20pct": replace(
            base_config,
            breakout_exit_window=round(
                base_config.breakout_exit_window * 1.20
            )
        ),
        "stop_minus_20pct": replace(
            base_config,
            trend_stop_atr_multiple=(
                base_config.trend_stop_atr_multiple * 0.80
            )
        ),
        "stop_plus_20pct": replace(
            base_config,
            trend_stop_atr_multiple=(
                base_config.trend_stop_atr_multiple * 1.20
            )
        ),
        "trailing_minus_20pct": replace(
            base_config,
            trend_trailing_atr_multiple=(
                base_config.trend_trailing_atr_multiple * 0.80
            )
        ),
        "trailing_plus_20pct": replace(
            base_config,
            trend_trailing_atr_multiple=(
                base_config.trend_trailing_atr_multiple * 1.20
            )
        )
    }

    return variants


def build_cost_delay_scenarios(base_backtest_config):
    return {
        "base": base_backtest_config,
        "fee_x2": replace(
            base_backtest_config,
            fee_rate=base_backtest_config.fee_rate * 2
        ),
        "slippage_x2": replace(
            base_backtest_config,
            slippage_rate=(
                base_backtest_config.slippage_rate * 2
            )
        ),
        "all_costs_x2": replace(
            base_backtest_config,
            fee_rate=base_backtest_config.fee_rate * 2,
            slippage_rate=(
                base_backtest_config.slippage_rate * 2
            )
        ),
        "delay_2_bars": replace(
            base_backtest_config,
            execution_delay_bars=2
        ),
        "delay_3_bars": replace(
            base_backtest_config,
            execution_delay_bars=3
        ),
        "costs_x2_delay_2": replace(
            base_backtest_config,
            fee_rate=base_backtest_config.fee_rate * 2,
            slippage_rate=(
                base_backtest_config.slippage_rate * 2
            ),
            execution_delay_bars=2
        )
    }


def run_one_backtest(raw_df, strategy_config, backtest_config):
    signal_df, sleeves = generate_signals(
        raw_df,
        strategy_config
    )
    result = run_backtest(
        signal_df,
        sleeves,
        backtest_config
    )
    metrics = calculate_metrics(result)
    return result, metrics


def calculate_without_best_trades(result, remove_count=2):
    if result.trades.empty:
        return {
            "removed_trade_count": 0,
            "remaining_profit": 0.0,
            "remaining_return": 0.0
        }

    removed_count = min(remove_count, len(result.trades))
    best_profit = (
        result.trades["net_profit"]
        .nlargest(removed_count)
        .sum()
    )
    remaining_profit = (
        result.final_value
        -
        result.initial_capital
        -
        best_profit
    )

    return {
        "removed_trade_count": removed_count,
        "remaining_profit": float(remaining_profit),
        "remaining_return": float(
            remaining_profit / result.initial_capital
        )
    }


def run_trade_monte_carlo(
    trades,
    initial_capital,
    simulation_count=5000,
    random_seed=20260823
):
    if trades.empty:
        return {
            "simulation_count": simulation_count,
            "trade_count": 0,
            "median_final_value": initial_capital,
            "percentile_5_final_value": initial_capital,
            "loss_probability": 1.0,
            "percentile_5_max_drawdown": 0.0
        }

    account_returns = (
        trades["net_profit"].to_numpy(dtype=float)
        /
        trades["entry_account_value"].to_numpy(dtype=float)
    )
    account_returns = np.clip(account_returns, -0.99, None)
    random_generator = np.random.default_rng(random_seed)
    sampled_returns = random_generator.choice(
        account_returns,
        size=(simulation_count, len(account_returns)),
        replace=True
    )
    equity_paths = (
        initial_capital
        *
        np.cumprod(1 + sampled_returns, axis=1)
    )
    final_values = equity_paths[:, -1]
    running_high = np.maximum.accumulate(equity_paths, axis=1)
    path_drawdowns = equity_paths / running_high - 1
    maximum_drawdowns = path_drawdowns.min(axis=1)

    return {
        "simulation_count": simulation_count,
        "trade_count": len(account_returns),
        "median_final_value": float(np.median(final_values)),
        "percentile_5_final_value": float(
            np.percentile(final_values, 5)
        ),
        "loss_probability": float(
            np.mean(final_values < initial_capital)
        ),
        "percentile_5_max_drawdown": float(
            np.percentile(maximum_drawdowns, 5)
        )
    }


def evaluate_robustness_gate(
    scenario_df,
    parameter_df,
    removal_df,
    monte_carlo_df
):
    doubled_costs = scenario_df[
        scenario_df["scenario"] == "all_costs_x2"
    ]
    delayed = scenario_df[
        scenario_df["scenario"] == "delay_2_bars"
    ]

    parameter_symbol_summary = (
        parameter_df.groupby("symbol")
        .agg(
            profitable_ratio=(
                "total_return",
                lambda values: (values > 0).mean()
            ),
            median_sharpe=("sharpe_ratio", "median")
        )
    )

    checks = {
        "双倍成本下每个品种仍盈利": bool(
            (doubled_costs["total_return"] > 0).all()
        ),
        "延迟2根K线时每个品种仍盈利": bool(
            (delayed["total_return"] > 0).all()
        ),
        "每个品种至少70%参数扰动仍盈利": bool(
            (
                parameter_symbol_summary["profitable_ratio"]
                >=
                0.70
            ).all()
        ),
        "每个品种参数扰动Sharpe中位数至少0.50": bool(
            (
                parameter_symbol_summary["median_sharpe"]
                >=
                0.50
            ).all()
        ),
        "删除最佳两笔后每个品种仍盈利": bool(
            (removal_df["remaining_return"] > 0).all()
        ),
        "蒙特卡洛亏损概率每个品种不超过25%": bool(
            (monte_carlo_df["loss_probability"] <= 0.25).all()
        ),
        "蒙特卡洛5%情景回撤不超过20%": bool(
            (
                monte_carlo_df["percentile_5_max_drawdown"]
                >=
                -0.20
            ).all()
        )
    }

    return {
        "passed": all(checks.values()),
        "checks": checks
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="现货策略稳健性和压力测试"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument(
        "--strategy-config",
        default="configs/trend_conservative.json"
    )
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--simulations", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    strategy_config_path = Path(args.strategy_config)

    if not strategy_config_path.is_absolute():
        strategy_config_path = project_folder / strategy_config_path

    strategy_config = load_strategy_config(strategy_config_path)
    base_backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk
    )
    validate_config(strategy_config, base_backtest_config)

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

    scenario_records = []
    base_results = {}
    scenarios = build_cost_delay_scenarios(base_backtest_config)

    for symbol, raw_df in raw_data_by_symbol.items():
        for scenario_name, backtest_config in scenarios.items():
            result, metrics = run_one_backtest(
                raw_df,
                strategy_config,
                backtest_config
            )
            scenario_records.append(
                {
                    "symbol": symbol,
                    "scenario": scenario_name,
                    **metrics
                }
            )

            if scenario_name == "base":
                base_results[symbol] = result

    parameter_records = []
    variants = build_parameter_variants(strategy_config)

    for symbol, raw_df in raw_data_by_symbol.items():
        for variant_name, variant_config in variants.items():
            validate_config(variant_config, base_backtest_config)
            _, metrics = run_one_backtest(
                raw_df,
                variant_config,
                base_backtest_config
            )
            parameter_records.append(
                {
                    "symbol": symbol,
                    "variant": variant_name,
                    **asdict(variant_config),
                    **metrics
                }
            )

    removal_records = []
    monte_carlo_records = []

    for symbol, result in base_results.items():
        removal_records.append(
            {
                "symbol": symbol,
                **calculate_without_best_trades(result)
            }
        )
        monte_carlo_records.append(
            {
                "symbol": symbol,
                **run_trade_monte_carlo(
                    trades=result.trades,
                    initial_capital=result.initial_capital,
                    simulation_count=args.simulations,
                    random_seed=20260823
                )
            }
        )

    scenario_df = pd.DataFrame(scenario_records)
    parameter_df = pd.DataFrame(parameter_records)
    removal_df = pd.DataFrame(removal_records)
    monte_carlo_df = pd.DataFrame(monte_carlo_records)
    gate_result = evaluate_robustness_gate(
        scenario_df,
        parameter_df,
        removal_df,
        monte_carlo_df
    )

    report_folder = project_folder / "robustness_reports"
    report_folder.mkdir(parents=True, exist_ok=True)
    config_name = strategy_config_path.stem
    scenario_df.to_csv(
        report_folder / f"{config_name}_cost_delay.csv",
        index=False
    )
    parameter_df.to_csv(
        report_folder / f"{config_name}_parameters.csv",
        index=False
    )
    removal_df.to_csv(
        report_folder / f"{config_name}_remove_best.csv",
        index=False
    )
    monte_carlo_df.to_csv(
        report_folder / f"{config_name}_monte_carlo.csv",
        index=False
    )

    summary = {
        "warning": (
            "这些测试使用已经查看过的历史，只能检查脆弱性，"
            "不能代替未来模拟盘。"
        ),
        "symbols": list(raw_data_by_symbol),
        "interval": args.interval,
        "start": args.start,
        "end": args.end,
        "strategy_config": str(strategy_config_path),
        "simulation_count": args.simulations,
        "gate_passed": gate_result["passed"],
        "checks": gate_result["checks"]
    }

    with (report_folder / f"{config_name}_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n稳健性与压力测试")

    for name, passed in gate_result["checks"].items():
        status = "通过" if passed else "未通过"
        print(f"{status}：{name}")

    for row in monte_carlo_df.itertuples(index=False):
        print(
            f"{row.symbol}蒙特卡洛："
            f"亏损概率{row.loss_probability * 100:.2f}%，"
            f"5%最终资产{row.percentile_5_final_value:.2f} USDT，"
            f"5%回撤{row.percentile_5_max_drawdown * 100:.2f}%"
        )

    if gate_result["passed"]:
        print("结论：通过历史稳健性门槛，下一步只能进入模拟盘。")
    else:
        print("结论：未通过历史稳健性门槛，不能进入模拟盘。")

    print(f"报告位置：{report_folder}")


if __name__ == "__main__":
    main()
