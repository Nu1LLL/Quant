import argparse
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path

import pandas as pd

from data import load_or_download_klines, to_utc_timestamp
from engine import BacktestResult
from metrics import calculate_metrics, calculate_monthly_report
from rotation_strategy import (
    load_rotation_config,
    run_rotation_backtest
)


DEFAULT_CONFIG_FILE = (
    Path(__file__).resolve().parent
    /
    "configs"
    /
    "rotation_growth_20_25.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="BTC、ETH与现金之间的现货轮动研究"
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee-rate", type=float, default=0.001)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument(
        "--strategy-config",
        default=str(DEFAULT_CONFIG_FILE)
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--report-folder",
        default="rotation_reports"
    )

    return parser.parse_args()


def load_frames(arguments):
    # 读取两个交易品种完全相同时间范围的历史K线
    frames = {}

    for symbol in arguments.symbols:
        frames[symbol.upper()] = load_or_download_klines(
            symbol=symbol,
            interval=arguments.interval,
            start_time=arguments.start,
            end_time=arguments.end,
            cache_folder="data_cache",
            refresh=arguments.refresh
        )

    return frames


def run_case(
    frames,
    config,
    initial_capital,
    fee_rate,
    slippage_rate,
    execution_delay_bars
):
    result = run_rotation_backtest(
        original_frames=frames,
        config=config,
        initial_capital=initial_capital,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        execution_delay_bars=execution_delay_bars
    )

    return result, calculate_metrics(result)


def make_period_report(result, frequency):
    # 根据权益曲线计算每月或每年的连续收益
    equity_curve = result.equity_curve.copy()
    equity_curve["open_time"] = pd.to_datetime(
        equity_curve["open_time"],
        utc=True
    )
    equity_curve = equity_curve.set_index("open_time")

    ending_equity = equity_curve["equity"].resample(frequency).last()
    starting_equity = ending_equity.shift(1)

    if not starting_equity.empty:
        starting_equity.iloc[0] = result.initial_capital

    return pd.DataFrame(
        {
            "ending_equity": ending_equity,
            "profit": ending_equity - starting_equity,
            "return_rate": ending_equity / starting_equity - 1
        }
    ).dropna(subset=["ending_equity"])


def make_recent_frames(frames, recent_start="2022-01-01"):
    # 单独检查近年震荡和下跌市场，不把牛市成绩混在里面
    recent_timestamp = to_utc_timestamp(recent_start)

    return {
        symbol: df[
            pd.to_datetime(df["open_time"], utc=True)
            >=
            recent_timestamp
        ].reset_index(drop=True)
        for symbol, df in frames.items()
    }


def make_period_frames(frames, start, end):
    # 截取独立时间段，用相同初始资金重新起跑
    start_timestamp = to_utc_timestamp(start)
    end_timestamp = to_utc_timestamp(end)

    return {
        symbol: df[
            (
                pd.to_datetime(df["open_time"], utc=True)
                >=
                start_timestamp
            )
            &
            (
                pd.to_datetime(df["open_time"], utc=True)
                <
                end_timestamp
            )
        ].reset_index(drop=True)
        for symbol, df in frames.items()
    }


def run_recent_rolling_windows(
    frames,
    config,
    initial_capital,
    fee_rate,
    slippage_rate
):
    # 用多个近年窗口重新起跑，避免单一牛市区间抬高年化
    periods = [
        ("2022-2023", "2022-01-01", "2024-01-01"),
        ("2023-2024", "2023-01-01", "2025-01-01"),
        ("2024-2025", "2024-01-01", "2026-01-01")
    ]
    rows = []

    for name, start, end in periods:
        period_frames = make_period_frames(frames, start, end)
        _, metrics = run_case(
            frames=period_frames,
            config=config,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            execution_delay_bars=1
        )
        rows.append(
            {
                "period": name,
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "profit_factor": metrics["profit_factor"],
                "trade_count": metrics["trade_count"]
            }
        )

    return pd.DataFrame(rows)


def make_research_checks(
    base_metrics,
    recent_metrics,
    recent_double_cost_metrics,
    recent_delayed_metrics,
    recent_rolling_report,
    neighborhood_summary
):
    # 近年成绩是主门槛，完整历史只用于确认策略没有长期失效
    rolling_profitable = bool(
        (recent_rolling_report["annualized_return"] > 0).all()
    )
    rolling_drawdown_passed = bool(
        (recent_rolling_report["max_drawdown"] >= -0.15).all()
    )
    rolling_sharpe_passed = bool(
        (recent_rolling_report["sharpe_ratio"] >= 1.0).all()
    )

    return {
        "近年独立起算年化位于20%至25%": (
            0.20 <= recent_metrics["annualized_return"] <= 0.25
        ),
        "近年独立起算最大回撤不超过15%": (
            recent_metrics["max_drawdown"] >= -0.15
        ),
        "近年独立起算Sharpe不低于1": (
            recent_metrics["sharpe_ratio"] >= 1.0
        ),
        "近年双倍成本压力测试仍守住全部风险门槛": (
            recent_double_cost_metrics["annualized_return"] >= 0.10
            and
            recent_double_cost_metrics["max_drawdown"] >= -0.15
            and
            recent_double_cost_metrics["sharpe_ratio"] >= 1.0
        ),
        "近年延迟两根K线压力测试仍守住全部风险门槛": (
            recent_delayed_metrics["annualized_return"] >= 0.10
            and
            recent_delayed_metrics["max_drawdown"] >= -0.15
            and
            recent_delayed_metrics["sharpe_ratio"] >= 1.0
        ),
        "近年滚动两年窗口全部盈利": rolling_profitable,
        "近年滚动两年窗口回撤全部不超过15%": (
            rolling_drawdown_passed
        ),
        "近年滚动两年窗口Sharpe全部不低于1": (
            rolling_sharpe_passed
        ),
        "完整历史仍然盈利": base_metrics["total_return"] > 0,
        "相邻参数通过比例不低于20%": (
            neighborhood_summary["passed_ratio"] >= 0.20
        )
    }


def run_parameter_neighborhood(
    frames,
    recent_frames,
    config,
    initial_capital,
    fee_rate,
    slippage_rate
):
    # 检查相邻参数是否也有效，避免只选中一个孤立幸运点
    rows = []
    slow_windows = sorted(
        {
            config.slow_ema_window - 10,
            config.slow_ema_window,
            config.slow_ema_window + 10
        }
    )
    fast_windows = sorted(
        {
            config.fast_ema_window - 10,
            config.fast_ema_window,
            config.fast_ema_window + 10
        }
    )
    momentum_windows = sorted(
        {
            max(1, config.momentum_window - 20),
            config.momentum_window,
            config.momentum_window + 30
        }
    )

    for slow_window, fast_window, momentum_window in product(
        slow_windows,
        fast_windows,
        momentum_windows
    ):
        if fast_window >= slow_window:
            continue

        neighbor_config = type(config)(
            slow_ema_window=slow_window,
            fast_ema_window=fast_window,
            momentum_window=momentum_window,
            ema_slope_window=config.ema_slope_window,
            rebalance_bars=config.rebalance_bars,
            exposure=config.exposure,
            broad_market_exposure=(
                config.broad_market_exposure
            ),
            volatility_window=config.volatility_window,
            target_volatility=config.target_volatility,
            minimum_exposure=config.minimum_exposure
        )

        _, full_metrics = run_case(
            frames=frames,
            config=neighbor_config,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            execution_delay_bars=1
        )
        _, recent_metrics = run_case(
            frames=recent_frames,
            config=neighbor_config,
            initial_capital=initial_capital,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            execution_delay_bars=1
        )

        passed = (
            0.20 <= recent_metrics["annualized_return"] <= 0.25
            and
            recent_metrics["max_drawdown"] >= -0.15
            and
            recent_metrics["sharpe_ratio"] >= 1.0
            and
            full_metrics["total_return"] > 0
        )

        rows.append(
            {
                "slow_ema_window": slow_window,
                "fast_ema_window": fast_window,
                "momentum_window": momentum_window,
                "full_annualized_return": (
                    full_metrics["annualized_return"]
                ),
                "full_max_drawdown": full_metrics["max_drawdown"],
                "full_sharpe_ratio": full_metrics["sharpe_ratio"],
                "recent_annualized_return": (
                    recent_metrics["annualized_return"]
                ),
                "recent_max_drawdown": recent_metrics["max_drawdown"],
                "recent_sharpe_ratio": recent_metrics["sharpe_ratio"],
                "passed": passed
            }
        )

    report = pd.DataFrame(rows)

    return report, {
        "variant_count": len(report),
        "passed_count": int(report["passed"].sum()),
        "passed_ratio": float(report["passed"].mean()),
        "median_full_annualized_return": float(
            report["full_annualized_return"].median()
        ),
        "median_full_max_drawdown": float(
            report["full_max_drawdown"].median()
        ),
        "median_full_sharpe_ratio": float(
            report["full_sharpe_ratio"].median()
        ),
        "median_recent_annualized_return": float(
            report["recent_annualized_return"].median()
        ),
        "median_recent_max_drawdown": float(
            report["recent_max_drawdown"].median()
        ),
        "median_recent_sharpe_ratio": float(
            report["recent_sharpe_ratio"].median()
        )
    }


def json_safe(value):
    if isinstance(value, dict):
        return {
            key: json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    if value == float("inf"):
        return "Infinity"

    return value


def format_metrics(title, metrics):
    return "\n".join(
        [
            f"{title}：",
            f"年化收益率：{metrics['annualized_return'] * 100:.2f}%",
            f"最大回撤：{metrics['max_drawdown'] * 100:.2f}%",
            f"Sharpe：{metrics['sharpe_ratio']:.2f}",
            f"利润因子：{metrics['profit_factor']:.2f}",
            f"完整交易：{metrics['trade_count']}笔",
            f"最终资产：{metrics['final_value']:.2f} USDT"
        ]
    )


def save_reports(
    report_folder,
    result,
    monthly_report,
    yearly_report,
    summary,
    neighborhood_report,
    recent_rolling_report
):
    # 保存本次研究结果，不覆盖现有模拟盘状态
    output_folder = Path(report_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    prefix = output_folder / "rotation_growth_20_25"

    result.equity_curve.to_csv(
        f"{prefix}_equity.csv",
        index=False
    )
    result.trades.to_csv(
        f"{prefix}_trades.csv",
        index=False
    )
    monthly_report.to_csv(
        f"{prefix}_monthly.csv"
    )
    yearly_report.to_csv(
        f"{prefix}_yearly.csv"
    )
    neighborhood_report.to_csv(
        f"{prefix}_parameter_neighborhood.csv",
        index=False
    )
    recent_rolling_report.to_csv(
        f"{prefix}_recent_rolling_windows.csv",
        index=False
    )

    with Path(f"{prefix}_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            json_safe(summary),
            file,
            ensure_ascii=False,
            indent=2
        )

    return output_folder.resolve()


def main():
    arguments = parse_arguments()
    config = load_rotation_config(arguments.strategy_config)
    frames = load_frames(arguments)

    base_result, base_metrics = run_case(
        frames=frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate,
        slippage_rate=arguments.slippage_rate,
        execution_delay_bars=1
    )

    _, double_cost_metrics = run_case(
        frames=frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate * 2,
        slippage_rate=arguments.slippage_rate * 2,
        execution_delay_bars=1
    )

    _, delayed_metrics = run_case(
        frames=frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate,
        slippage_rate=arguments.slippage_rate,
        execution_delay_bars=2
    )

    recent_frames = make_recent_frames(frames)
    _, recent_metrics = run_case(
        frames=recent_frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate,
        slippage_rate=arguments.slippage_rate,
        execution_delay_bars=1
    )

    _, recent_double_cost_metrics = run_case(
        frames=recent_frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate * 2,
        slippage_rate=arguments.slippage_rate * 2,
        execution_delay_bars=1
    )

    _, recent_delayed_metrics = run_case(
        frames=recent_frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate,
        slippage_rate=arguments.slippage_rate,
        execution_delay_bars=2
    )

    recent_rolling_report = run_recent_rolling_windows(
        frames=frames,
        config=config,
        initial_capital=arguments.capital,
        fee_rate=arguments.fee_rate,
        slippage_rate=arguments.slippage_rate
    )

    monthly_report = calculate_monthly_report(base_result)
    yearly_report = make_period_report(base_result, "YE")
    neighborhood_report, neighborhood_summary = (
        run_parameter_neighborhood(
            frames=frames,
            recent_frames=recent_frames,
            config=config,
            initial_capital=arguments.capital,
            fee_rate=arguments.fee_rate,
            slippage_rate=arguments.slippage_rate
        )
    )

    research_checks = make_research_checks(
        base_metrics=base_metrics,
        recent_metrics=recent_metrics,
        recent_double_cost_metrics=recent_double_cost_metrics,
        recent_delayed_metrics=recent_delayed_metrics,
        recent_rolling_report=recent_rolling_report,
        neighborhood_summary=neighborhood_summary
    )

    summary = {
        "research_only": True,
        "symbols": [symbol.upper() for symbol in arguments.symbols],
        "interval": arguments.interval,
        "start": arguments.start,
        "end": arguments.end,
        "strategy_config": asdict(config),
        "cost_assumptions": {
            "fee_rate": arguments.fee_rate,
            "slippage_rate": arguments.slippage_rate
        },
        "base_metrics": base_metrics,
        "double_cost_metrics": double_cost_metrics,
        "two_bar_delay_metrics": delayed_metrics,
        "recent_2022_onward_metrics": recent_metrics,
        "recent_double_cost_metrics": recent_double_cost_metrics,
        "recent_two_bar_delay_metrics": recent_delayed_metrics,
        "recent_rolling_two_year_windows": (
            recent_rolling_report.to_dict(orient="records")
        ),
        "parameter_neighborhood_summary": neighborhood_summary,
        "research_checks": research_checks,
        "warning": (
            "历史区间和参数已经被查看，不能称为隐藏测试；"
            "该策略没有下单代码，不能直接进入实盘。"
        )
    }

    report_folder = save_reports(
        report_folder=arguments.report_folder,
        result=base_result,
        monthly_report=monthly_report,
        yearly_report=yearly_report,
        summary=summary,
        neighborhood_report=neighborhood_report,
        recent_rolling_report=recent_rolling_report
    )

    print("\n现货趋势动量轮动研究")
    print(format_metrics("完整历史", base_metrics))
    print()
    print(format_metrics("双倍成本压力测试", double_cost_metrics))
    print()
    print(format_metrics("延迟两根K线压力测试", delayed_metrics))
    print()
    print(format_metrics("2022年至今单独检查", recent_metrics))
    print()
    print(
        format_metrics(
            "2022年至今双倍成本压力测试",
            recent_double_cost_metrics
        )
    )
    print()
    print(
        format_metrics(
            "2022年至今延迟两根K线压力测试",
            recent_delayed_metrics
        )
    )

    print("\n近年两年滚动窗口：")

    for row in recent_rolling_report.to_dict(orient="records"):
        print(
            f"{row['period']}："
            f"年化{row['annualized_return'] * 100:.2f}%，"
            f"最大回撤{row['max_drawdown'] * 100:.2f}%，"
            f"Sharpe {row['sharpe_ratio']:.2f}"
        )

    print(
        "\n相邻参数检查："
        f"{neighborhood_summary['passed_count']}/"
        f"{neighborhood_summary['variant_count']}组通过，"
        f"通过比例"
        f"{neighborhood_summary['passed_ratio'] * 100:.2f}%"
    )

    print("\n研究门槛：")

    for name, passed in research_checks.items():
        status = "通过" if passed else "未通过"
        print(f"{status}：{name}")

    print(f"\n报告已保存到：{report_folder}")
    print("该模块只做历史研究，没有API密钥和真实下单接口。")
    print("结论：不能替换当前服务器模拟策略，必须另开未来模拟验收。")


if __name__ == "__main__":
    main()
