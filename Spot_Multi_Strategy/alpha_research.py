"""独立评估alphas/库里每一个alpha信号的预测质量（不是交易PnL）。

这个脚本只做研究分析：把每个alpha的normalized_signal与未来收益标签
（只在alpha_metrics.forward_return里产生，从不进入信号计算）配对，
计算IC、分桶收益、换手代理和成本敏感度。是否"通过"要交给
walk_forward.py里的Alpha Research Gate（多折样本外验证），
这里只负责把全样本研究事实摆出来。
"""
import argparse
from pathlib import Path

import pandas as pd

import alpha_metrics
import alphas
import alphas.alternative_data as alternative_data_alphas
from data import load_or_download_klines, to_utc_timestamp
from futures_data import load_or_download_funding_rates

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Alpha信号预测质量研究（不是交易回测）"
    )
    parser.add_argument(
        "--symbols", nargs="+", default=DEFAULT_SYMBOLS
    )
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 3, 6]
    )
    parser.add_argument("--delay", type=int, default=1)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--rolling-window", type=int, default=250)
    parser.add_argument(
        "--no-funding", action="store_true",
        help="跳过资金费率alpha（比如离线环境拿不到永续合约数据时）"
    )
    parser.add_argument("--output-folder", default="alpha_reports")
    return parser.parse_args()


def load_symbol_frames(symbols, interval, start, end, cache_folder):
    frames = {}
    for symbol in symbols:
        frames[symbol.upper()] = load_or_download_klines(
            symbol=symbol,
            interval=interval,
            start_time=start,
            end_time=end,
            cache_folder=cache_folder
        )
    return frames


def load_funding_frames(symbols, start, end, cache_folder="futures_data_cache"):
    """返回{symbol: funding_df}，某个品种的永续合约资金费率拿不到时
    直接跳过那个品种（不静默伪造数据），由调用方决定是否继续。
    """
    funding_frames = {}
    for symbol in symbols:
        try:
            funding_frames[symbol.upper()] = load_or_download_funding_rates(
                symbol=symbol,
                start_time=start,
                end_time=end,
                cache_folder=cache_folder
            )
        except Exception:
            continue
    return funding_frames


def build_alpha_sets(frames, funding_frames=None):
    """返回{symbol: {alpha_name: AlphaSignal}}。

    - 单资产alpha（A01-A17）：每个品种都有。
    - A18/A19（BTC领先/ETH滞后、BTC-ETH pairwise相对强弱）：只有当
      BTC和ETH都在frames里时才附加到ETHUSDT。
    - A20/A23（市场广度、跨资产相对强弱的N资产泛化版）：当frames里
      至少有3个品种时，附加到**每一个**品种（不要求恰好是BTC/ETH，
      不同品种历史长度不一致时用outer join对齐，缺失数据的品种
      自然被排除，不做静默填补）。
    - A21/A22（资金费率alpha）：某个品种在funding_frames里有对应的
      永续合约资金费率数据时才附加，没有就跳过那个品种，不报错。
    """
    alpha_sets = {}
    btc_df = frames.get("BTCUSDT")
    eth_df = frames.get("ETHUSDT")
    funding_frames = funding_frames or {}

    for symbol, df in frames.items():
        if symbol == "ETHUSDT" and btc_df is not None and eth_df is not None:
            alpha_sets[symbol] = alphas.build_alpha_library(
                df, btc_df=btc_df, eth_df=eth_df
            )
        else:
            alpha_sets[symbol] = alphas.build_single_asset_alphas(df)

    if len(frames) >= 3:
        breadth = alphas.cross_asset.build_market_breadth(frames)
        relative_strength = (
            alphas.cross_asset.build_cross_sectional_relative_strength(
                frames
            )
        )
        for symbol in frames:
            alpha_sets[symbol].update(breadth[symbol])
            alpha_sets[symbol].update(relative_strength[symbol])

    for symbol, funding_df in funding_frames.items():
        if symbol not in alpha_sets:
            continue
        alpha_sets[symbol].update(
            alternative_data_alphas.build_alphas(frames[symbol], funding_df)
        )

    return alpha_sets


def run_research(frames, alpha_sets, horizons, delay, fee, slippage, rolling_window):
    summary_rows = []
    yearly_frames = []

    for symbol, alpha_signals in alpha_sets.items():
        df = frames[symbol]
        regime_labels = alpha_metrics.compute_regime_labels(df)

        for alpha_signal in alpha_signals.values():
            summary_rows.extend(
                alpha_metrics.evaluate_alpha(
                    symbol=symbol,
                    alpha_signal=alpha_signal,
                    df=df,
                    horizons=horizons,
                    delay=delay,
                    fee_rate=fee,
                    slippage_rate=slippage,
                    rolling_window=rolling_window,
                    regime_labels=regime_labels
                )
            )
            yearly_frames.append(
                alpha_metrics.evaluate_alpha_years(
                    symbol=symbol,
                    alpha_signal=alpha_signal,
                    df=df,
                    horizons=horizons,
                    delay=delay
                )
            )

    summary_df = pd.DataFrame(summary_rows)
    yearly_df = (
        pd.concat(yearly_frames, ignore_index=True)
        if yearly_frames
        else pd.DataFrame()
    )

    return summary_df, yearly_df


def format_markdown_summary(summary_df, horizons):
    if summary_df.empty:
        return "# Alpha研究报告\n\n没有可用的alpha评估结果。\n"

    lines = [
        "# Alpha研究报告（预测质量，不是交易PnL）",
        "",
        (
            "本报告评估alphas/库中每个信号与未来收益标签的相关性，"
            "前瞻窗口只用于研究，从未进入信号计算。是否进入组合"
            "由walk_forward.py里的多折样本外Alpha Research Gate决定，"
            "这里的IC只是全样本描述性统计，不能单独作为入选依据。"
        ),
        ""
    ]

    primary_horizon = horizons[0]
    primary = summary_df[summary_df["horizon_bars"] == primary_horizon].copy()
    primary = primary.sort_values("pearson_ic", ascending=False)

    lines.append(f"## 按{primary_horizon}根K线前瞻窗口排序（Pearson IC）")
    lines.append("")
    lines.append(
        "| 品种 | Alpha | 方向 | 观测数 | Pearson IC | Spearman IC | "
        "t统计量 | 滚动IC为正比例 | 高低分桶价差 | 1x成本调整后价差 | "
        "3x成本调整后价差 | 换手代理 |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )

    for row in primary.itertuples(index=False):
        lines.append(
            f"| {row.symbol} | {row.alpha_name} | {row.direction} | "
            f"{row.observations} | {row.pearson_ic:.4f} | "
            f"{row.spearman_ic:.4f} | {row.ic_t_stat:.2f} | "
            f"{row.rolling_ic_positive_ratio:.2%} | "
            f"{row.top_minus_bottom_spread:.4f} | "
            f"{row.cost_adjusted_spread_1x:.4f} | "
            f"{row.cost_adjusted_spread_3x:.4f} | "
            f"{row.turnover_proxy:.4f} |"
        )

    lines.append("")
    lines.append(
        "完整数据（全部前瞻窗口、regime细分、逐年细分）见"
        "同目录下的CSV文件。"
    )

    return "\n".join(lines) + "\n"


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent

    end_time = (
        to_utc_timestamp(args.end) if args.end is not None else None
    )
    if end_time is None:
        now = pd.Timestamp.now(tz="UTC")
        from data import INTERVAL_TO_TIMEDELTA
        interval_seconds = int(
            INTERVAL_TO_TIMEDELTA[args.interval].total_seconds()
        )
        now_seconds = int(now.timestamp())
        closed_boundary_seconds = (
            now_seconds // interval_seconds * interval_seconds
        )
        end_time = pd.Timestamp(
            closed_boundary_seconds, unit="s", tz="UTC"
        )

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
            symbols=args.symbols,
            start=args.start,
            end=end_time,
            cache_folder=project_folder / "futures_data_cache"
        )
    )

    alpha_sets = build_alpha_sets(frames, funding_frames=funding_frames)
    summary_df, yearly_df = run_research(
        frames=frames,
        alpha_sets=alpha_sets,
        horizons=args.horizons,
        delay=args.delay,
        fee=args.fee,
        slippage=args.slippage,
        rolling_window=args.rolling_window
    )

    output_folder = project_folder / args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        output_folder / "alpha_summary.csv", index=False
    )
    yearly_df.to_csv(
        output_folder / "alpha_yearly_breakdown.csv", index=False
    )

    markdown = format_markdown_summary(summary_df, args.horizons)
    (output_folder / "README.md").write_text(markdown, encoding="utf-8")

    print(f"评估了{sum(len(v) for v in alpha_sets.values())}个alpha实例")
    print(f"品种：{list(alpha_sets.keys())}")
    print(f"报告已保存到：{output_folder}")


if __name__ == "__main__":
    main()
