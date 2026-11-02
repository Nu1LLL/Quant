import argparse
from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from data import to_utc_timestamp
from futures_data import (
    load_or_download_funding_rates,
    load_or_download_futures_klines
)
from hybrid_strategy import (
    HybridConfig,
    load_hybrid_config,
    run_hybrid_backtest
)
from metrics import calculate_metrics


DEFAULT_CONFIG_FILE = (
    Path(__file__).resolve().parent
    /
    "configs"
    /
    "hybrid_spot_short.json"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="上涨持现货、明确熊市小仓位做空的独立研究模块"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"]
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-08-01")
    parser.add_argument("--recent-start", default="2022-01-01")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--spot-fee-rate", type=float, default=0.001)
    parser.add_argument("--futures-fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-rate", type=float, default=0.0005)
    parser.add_argument(
        "--strategy-config",
        default=str(DEFAULT_CONFIG_FILE)
    )
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--report-folder",
        default="hybrid_reports"
    )
    return parser.parse_args()


def load_public_data(arguments):
    # 只读取Binance公开合约K线和公开资金费率，不使用API密钥
    frames = {}
    funding_frames = {}

    for raw_symbol in arguments.symbols:
        symbol = raw_symbol.upper()
        frames[symbol] = load_or_download_futures_klines(
            symbol=symbol,
            interval=arguments.interval,
            start_time=arguments.start,
            end_time=arguments.end,
            cache_folder="futures_data_cache",
            refresh=arguments.refresh
        )
        funding_frames[symbol] = load_or_download_funding_rates(
            symbol=symbol,
            start_time=arguments.start,
            end_time=arguments.end,
            cache_folder="futures_data_cache",
            refresh=arguments.refresh
        )

    return frames, funding_frames


def slice_from(frames, start_time):
    start_timestamp = to_utc_timestamp(start_time)
    return {
        symbol: df[
            pd.to_datetime(df["open_time"], utc=True)
            >=
            start_timestamp
        ].reset_index(drop=True)
        for symbol, df in frames.items()
    }


def slice_funding_from(funding_frames, start_time):
    start_timestamp = to_utc_timestamp(start_time)
    return {
        symbol: df[
            pd.to_datetime(df["funding_time"], utc=True)
            >=
            start_timestamp
        ].reset_index(drop=True)
        for symbol, df in funding_frames.items()
    }


def run_case(
    frames,
    funding_frames,
    config,
    initial_capital,
    spot_fee_rate,
    futures_fee_rate,
    slippage_rate,
    execution_delay_bars
):
    result = run_hybrid_backtest(
        original_frames=frames,
        funding_frames=funding_frames,
        config=config,
        initial_capital=initial_capital,
        spot_fee_rate=spot_fee_rate,
        futures_fee_rate=futures_fee_rate,
        slippage_rate=slippage_rate,
        execution_delay_bars=execution_delay_bars
    )
    return result, calculate_metrics(result)


def trade_breakdown(result):
    if result.trades.empty:
        return {}

    rows = {}

    for instrument, trades in result.trades.groupby("instrument"):
        rows[str(instrument)] = {
            "trade_count": int(len(trades)),
            "net_profit": float(trades["net_profit"].sum()),
            "funding_cash_flow": float(
                trades.get(
                    "funding_cash_flow",
                    pd.Series(0.0, index=trades.index)
                ).sum()
            )
        }

    return rows


def passes_research_gate(metrics):
    # 20%到25%是目标区间，不是收益承诺
    return (
        metrics["annualized_return"] >= 0.20
        and metrics["max_drawdown"] >= -0.15
        and metrics["sharpe_ratio"] >= 1.0
    )


def metric_row(name, metrics, result):
    return {
        "case": name,
        **{
            key: float(value)
            for key, value in metrics.items()
        },
        "research_gate_passed": bool(passes_research_gate(metrics)),
        "trade_breakdown": trade_breakdown(result)
    }


def print_case(row):
    print(f"\n{row['case']}：")
    print(f"年化收益率：{row['annualized_return']:.2%}")
    print(f"最大回撤：{row['max_drawdown']:.2%}")
    print(f"Sharpe：{row['sharpe_ratio']:.2f}")
    print(f"利润因子：{row['profit_factor']:.2f}")
    print(f"完整交易：{int(row['trade_count'])}笔")
    print(f"研究门槛：{'通过' if row['research_gate_passed'] else '未通过'}")

    for instrument, summary in row["trade_breakdown"].items():
        print(
            f"{instrument}：{summary['trade_count']}笔，"
            f"净盈亏{summary['net_profit']:.2f} USDT，"
            f"资金费率现金流{summary['funding_cash_flow']:.2f} USDT"
        )


def main():
    arguments = parse_arguments()
    config = load_hybrid_config(arguments.strategy_config)
    frames, funding_frames = load_public_data(arguments)
    recent_frames = slice_from(frames, arguments.recent_start)
    recent_funding = slice_funding_from(
        funding_frames,
        arguments.recent_start
    )

    cases = []
    case_arguments = [
        (
            "全部历史",
            frames,
            funding_frames,
            arguments.spot_fee_rate,
            arguments.futures_fee_rate,
            arguments.slippage_rate,
            1
        ),
        (
            "双倍交易成本",
            frames,
            funding_frames,
            arguments.spot_fee_rate * 2,
            arguments.futures_fee_rate * 2,
            arguments.slippage_rate * 2,
            1
        ),
        (
            "延迟两根K线成交",
            frames,
            funding_frames,
            arguments.spot_fee_rate,
            arguments.futures_fee_rate,
            arguments.slippage_rate,
            2
        ),
        (
            f"{arguments.recent_start}以后独立起点",
            recent_frames,
            recent_funding,
            arguments.spot_fee_rate,
            arguments.futures_fee_rate,
            arguments.slippage_rate,
            1
        )
    ]

    report_folder = Path(arguments.report_folder)
    report_folder.mkdir(parents=True, exist_ok=True)

    for (
        name,
        case_frames,
        case_funding,
        spot_fee_rate,
        futures_fee_rate,
        slippage_rate,
        execution_delay_bars
    ) in case_arguments:
        result, metrics = run_case(
            frames=case_frames,
            funding_frames=case_funding,
            config=config,
            initial_capital=arguments.capital,
            spot_fee_rate=spot_fee_rate,
            futures_fee_rate=futures_fee_rate,
            slippage_rate=slippage_rate,
            execution_delay_bars=execution_delay_bars
        )
        row = metric_row(name, metrics, result)
        cases.append(row)
        safe_name = (
            name.replace("/", "-")
            .replace(" ", "_")
        )
        result.equity_curve.to_csv(
            report_folder / f"{safe_name}_equity.csv",
            index=False
        )
        result.trades.to_csv(
            report_folder / f"{safe_name}_trades.csv",
            index=False
        )
        print_case(row)

    summary = {
        "research_only": True,
        "approved_for_live_trading": False,
        "target": {
            "annualized_return": "20%-25%",
            "maximum_drawdown": "no worse than -15%",
            "sharpe_ratio": "at least 1.0"
        },
        "config": asdict(config),
        "cases": cases,
        "all_cases_passed": all(
            row["research_gate_passed"]
            for row in cases
        )
    }

    with (report_folder / "summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n结论：")

    if summary["all_cases_passed"]:
        print("仅通过研究门槛，仍需90天只读模拟和极小资金安全验收。")
    else:
        print("未通过研究门槛，禁止接入实盘。")

    print(f"报告位置：{report_folder.resolve()}")


if __name__ == "__main__":
    main()
