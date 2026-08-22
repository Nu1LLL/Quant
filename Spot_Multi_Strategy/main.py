import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from config import (
    BacktestConfig,
    StrategyConfig,
    validate_config
)
from config_io import load_strategy_config
from data import (
    INTERVAL_TO_TIMEDELTA,
    load_or_download_klines,
    to_utc_timestamp
)
from engine import run_backtest
from metrics import (
    calculate_buy_and_hold,
    calculate_metrics,
    calculate_monthly_report,
    calculate_monthly_summary,
    calculate_strategy_report,
    evaluate_research_gates,
    format_metrics,
    format_monthly_summary,
    format_research_gates,
    format_strategy_report
)
from strategies import generate_signals


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="只做多现货多策略回测模型"
    )

    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    parser.add_argument("--trend-weight", type=float, default=None)
    parser.add_argument("--pullback-weight", type=float, default=None)
    parser.add_argument("--strategy-config", default=None)
    parser.add_argument("--refresh", action="store_true")

    return parser.parse_args()


def get_effective_end_time(interval, end_time):
    if interval not in INTERVAL_TO_TIMEDELTA:
        raise ValueError(f"暂不支持K线周期：{interval}")

    if end_time is not None:
        return to_utc_timestamp(end_time)

    # 不传结束时间时，只获取已经结束的完整K线
    now = pd.Timestamp.now(tz="UTC")
    interval_delta = INTERVAL_TO_TIMEDELTA[interval]
    interval_seconds = int(interval_delta.total_seconds())
    now_seconds = int(now.timestamp())
    closed_boundary_seconds = (
        now_seconds // interval_seconds * interval_seconds
    )

    return pd.Timestamp(
        closed_boundary_seconds,
        unit="s",
        tz="UTC"
    )


def save_result_files(
    reports_folder,
    prefix,
    result,
    monthly_report,
    strategy_report,
    metrics,
    strategy_config,
    backtest_config
):
    reports_path = Path(reports_folder)
    reports_path.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(
        reports_path / f"{prefix}_equity.csv",
        index=False
    )

    result.trades.to_csv(
        reports_path / f"{prefix}_trades.csv",
        index=False
    )

    monthly_report.to_csv(
        reports_path / f"{prefix}_monthly.csv"
    )

    strategy_report.to_csv(
        reports_path / f"{prefix}_strategies.csv",
        index=False
    )

    pd.DataFrame([metrics]).to_csv(
        reports_path / f"{prefix}_metrics.csv",
        index=False
    )

    parameter_snapshot = {
        "strategy": asdict(strategy_config),
        "backtest": asdict(backtest_config)
    }

    with (reports_path / f"{prefix}_parameters.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            parameter_snapshot,
            file,
            ensure_ascii=False,
            indent=2
        )


def main():
    args = parse_arguments()

    if args.strategy_config is None:
        strategy_config = StrategyConfig()
    else:
        strategy_config = load_strategy_config(args.strategy_config)

    config_updates = {}

    if args.trend_weight is not None:
        config_updates["trend_weight"] = args.trend_weight

    if args.pullback_weight is not None:
        config_updates["pullback_weight"] = args.pullback_weight

    if config_updates:
        strategy_config = replace(strategy_config, **config_updates)
    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk
    )

    validate_config(strategy_config, backtest_config)

    effective_end = get_effective_end_time(
        args.interval,
        args.end
    )

    project_folder = Path(__file__).resolve().parent

    raw_df = load_or_download_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_time=args.start,
        end_time=effective_end,
        cache_folder=project_folder / "data_cache",
        refresh=args.refresh
    )

    signal_df, sleeves = generate_signals(
        raw_df,
        config=strategy_config
    )

    if len(signal_df) < strategy_config.ema_window + 50:
        raise ValueError(
            "K线数量太少，至少需要EMA周期再加50根K线"
        )

    # 最后30%数据作为完全不参与参数设计的样本外区间
    split_position = int(len(signal_df) * 0.70)

    if split_position <= strategy_config.ema_window:
        raise ValueError("训练区间太短，无法进行样本外验证")

    full_result = run_backtest(
        signal_df=signal_df,
        sleeves=sleeves,
        config=backtest_config
    )

    # 多保留一根K线，以便样本外第一根K线执行上一根收盘信号
    out_of_sample_df = (
        signal_df.iloc[split_position - 1:]
        .reset_index(drop=True)
    )

    out_of_sample_result = run_backtest(
        signal_df=out_of_sample_df,
        sleeves=sleeves,
        config=backtest_config
    )

    full_metrics = calculate_metrics(full_result)
    out_of_sample_metrics = calculate_metrics(
        out_of_sample_result
    )

    full_benchmark = calculate_buy_and_hold(
        df=signal_df,
        initial_capital=backtest_config.initial_capital,
        fee_rate=backtest_config.fee_rate,
        slippage_rate=backtest_config.slippage_rate
    )

    out_of_sample_benchmark = calculate_buy_and_hold(
        df=out_of_sample_df,
        initial_capital=backtest_config.initial_capital,
        fee_rate=backtest_config.fee_rate,
        slippage_rate=backtest_config.slippage_rate
    )

    print("\n现货多策略模型")
    print(f"交易品种：{args.symbol.upper()}")
    print(f"K线周期：{args.interval}")
    print(f"数据开始：{raw_df.iloc[0]['open_time']}")
    print(f"数据结束：{raw_df.iloc[-1]['open_time']}")
    print(f"K线数量：{len(raw_df)}")

    if "is_synthetic" in raw_df.columns:
        synthetic_count = int(raw_df["is_synthetic"].sum())
        print(f"修复的小缺口K线：{synthetic_count}")

    print(
        format_metrics(
            "全部历史区间",
            full_metrics,
            full_benchmark
        )
    )

    print(
        format_metrics(
            "样本外区间（最后30%）",
            out_of_sample_metrics,
            out_of_sample_benchmark
        )
    )

    full_monthly = calculate_monthly_report(full_result)
    out_of_sample_monthly = calculate_monthly_report(
        out_of_sample_result
    )

    full_strategy_report = calculate_strategy_report(full_result)
    out_of_sample_strategy_report = calculate_strategy_report(
        out_of_sample_result
    )
    gate_result = evaluate_research_gates(
        out_of_sample_metrics
    )

    print("\n" + format_monthly_summary(
        calculate_monthly_summary(full_monthly)
    ))
    print("\n" + format_strategy_report(full_strategy_report))

    print("\n样本外" + format_monthly_summary(
        calculate_monthly_summary(out_of_sample_monthly)
    ))
    print("\n样本外" + format_strategy_report(
        out_of_sample_strategy_report
    ))

    print("\n" + format_research_gates(gate_result))

    safe_start = pd.Timestamp(args.start).strftime("%Y%m%d")
    safe_end = effective_end.strftime("%Y%m%d")
    report_name = (
        f"{args.symbol.upper()}_{args.interval}_"
        f"{safe_start}_{safe_end}"
    )

    if args.strategy_config is None:
        config_label = "default"
    else:
        config_label = Path(args.strategy_config).stem

    report_name = f"{report_name}_{config_label}"

    save_result_files(
        reports_folder=project_folder / "reports",
        prefix=f"{report_name}_full",
        result=full_result,
        monthly_report=full_monthly,
        strategy_report=full_strategy_report,
        metrics=full_metrics,
        strategy_config=strategy_config,
        backtest_config=backtest_config
    )

    save_result_files(
        reports_folder=project_folder / "reports",
        prefix=f"{report_name}_out_of_sample",
        result=out_of_sample_result,
        monthly_report=out_of_sample_monthly,
        strategy_report=out_of_sample_strategy_report,
        metrics=out_of_sample_metrics,
        strategy_config=strategy_config,
        backtest_config=backtest_config
    )

    print(
        "\n报告已保存到："
        f"{project_folder / 'reports'}"
    )


if __name__ == "__main__":
    main()
