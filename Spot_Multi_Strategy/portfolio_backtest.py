"""把通过Alpha Research Gate的alpha合成组合，跑真实的多资产回测，
并与A.既有多策略组合 / C.买入持有 / D.regime过滤版本做对比。

这是唯一一个真正"下单"的新回测路径：ensemble.py产生combined_alpha和
raw target exposure，risk_overlay.py施加波动率目标/回撤控制/换手控制，
这里再把BTC和ETH两个资产按各自资金份额汇总成一条组合权益曲线，
成本假设与仓库默认值完全一致（不允许静默修改）。
"""
import argparse
import json
from pathlib import Path

import pandas as pd

import alpha_metrics
import ensemble
import portfolio_metrics
import risk_overlay
from alpha_gate_report import run_gate, summarize_acceptance
from alpha_research import build_alpha_sets, load_symbol_frames
from config import BacktestConfig, StrategyConfig, validate_config
from config_io import load_strategy_config
from data import INTERVAL_TO_TIMEDELTA, to_utc_timestamp
from engine import run_backtest
from metrics import calculate_buy_and_hold
from strategies import generate_signals

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
DEFAULT_PER_ASSET_CAP = 0.60


def _resolve_end_time(interval, end):
    if end is not None:
        return to_utc_timestamp(end)

    now = pd.Timestamp.now(tz="UTC")
    interval_seconds = int(INTERVAL_TO_TIMEDELTA[interval].total_seconds())
    now_seconds = int(now.timestamp())
    closed_boundary_seconds = now_seconds // interval_seconds * interval_seconds
    return pd.Timestamp(closed_boundary_seconds, unit="s", tz="UTC")


def get_accepted_alphas(frames, alpha_sets, folds=6, fee=0.001, slippage=0.0005):
    gate_df, fold_df, _ = run_gate(frames, alpha_sets, folds, fee, slippage)
    accepted, rejected = summarize_acceptance(gate_df)
    return accepted, rejected, gate_df, fold_df


def build_symbol_exposure(
    symbol,
    df,
    alpha_sets,
    accepted_names,
    method,
    apply_regime_filter=False,
    horizon=3,
    window=500,
    min_periods=100
):
    available = {
        name: alpha_sets[symbol][name]
        for name in accepted_names
        if name in alpha_sets[symbol]
    }
    if not available:
        raise ValueError(f"{symbol}没有任何通过验收的alpha可用于组合")

    weights, combined_alpha, target_exposure = ensemble.build_ensemble(
        available, df, method=method,
        horizon=horizon, window=window, min_periods=min_periods
    )

    if apply_regime_filter:
        # 注意：compute_regime_labels()的三分位切分点用了全样本分位数，
        # 会把未来数据泄漏进regime分类，只适合alpha_research.py那种
        # 事后描述性统计，不能用来决定实际仓位。这里改用纯滚动窗口的
        # compute_raw_efficiency_ratio做连续缩放（regime越震荡、ER越
        # 低，敞口按比例调低，而不是硬切0/1），一是消除未来数据泄漏，
        # 二是避免硬切换带来的频繁开平仓换手。
        regime_scalar = alpha_metrics.compute_raw_efficiency_ratio(df)
        target_exposure = target_exposure * regime_scalar.reset_index(
            drop=True
        ).fillna(0.0)

    return weights, combined_alpha, target_exposure, available


def run_alpha_ensemble_portfolio(
    frames,
    alpha_sets,
    accepted_names,
    method,
    fee_rate,
    slippage_rate,
    per_asset_cap=DEFAULT_PER_ASSET_CAP,
    apply_regime_filter=False,
    vol_target_annualized=None,
    stop_loss_atr_multiple=None,
    stop_loss_atr_window=14,
    initial_capital=5000.0,
    symbols=None
):
    symbols = symbols or list(frames.keys())
    capital_per_symbol = initial_capital / len(symbols)

    per_symbol_results = {}
    dollar_equity_curves = []

    for symbol in symbols:
        df = frames[symbol]
        _, _, target_exposure, _ = build_symbol_exposure(
            symbol, df, alpha_sets, accepted_names, method,
            apply_regime_filter=apply_regime_filter
        )

        risk_config = risk_overlay.RiskConfig(
            vol_target_annualized=vol_target_annualized,
            exposure_cap=per_asset_cap,
            stop_loss_atr_multiple=stop_loss_atr_multiple,
            stop_loss_atr_window=stop_loss_atr_window
        )
        simulation = risk_overlay.apply_risk_overlay(
            df, target_exposure, config=risk_config,
            fee_rate=fee_rate, slippage_rate=slippage_rate
        )

        simulation["dollar_equity"] = capital_per_symbol * simulation["equity"]
        per_symbol_results[symbol] = simulation
        dollar_equity_curves.append(
            simulation.set_index("open_time")[
                ["dollar_equity", "position", "turnover", "cost", "rebalanced"]
            ]
        )

    weight = capital_per_symbol / initial_capital
    combined_equity = sum(
        curve["dollar_equity"] for curve in dollar_equity_curves
    )
    combined_position = sum(
        weight * curve["position"] for curve in dollar_equity_curves
    )
    combined_turnover = sum(
        weight * curve["turnover"] for curve in dollar_equity_curves
    )
    combined_cost_fraction = sum(
        weight * curve["cost"] for curve in dollar_equity_curves
    )
    combined_rebalance_count = sum(
        curve["rebalanced"].astype(int) for curve in dollar_equity_curves
    )

    combined_net_pnl = combined_equity.pct_change().fillna(0.0)
    combined_net_pnl.iloc[0] = combined_equity.iloc[0] / initial_capital - 1.0

    portfolio_df = pd.DataFrame({
        "open_time": combined_equity.index,
        "net_pnl": combined_net_pnl.values,
        "equity": (combined_equity / initial_capital).values,
        "position": combined_position.values,
        "turnover": combined_turnover.values,
        "cost": combined_cost_fraction.values,
        "rebalanced": (combined_rebalance_count > 0).values
    }).reset_index(drop=True)

    metrics = portfolio_metrics.calculate_extended_metrics(
        portfolio_df, initial_capital=initial_capital
    )
    metrics["rebalance_count"] = int(combined_rebalance_count.sum())

    return {
        "metrics": metrics,
        "portfolio_equity": portfolio_df,
        "per_symbol": per_symbol_results
    }


def run_legacy_portfolio(
    frames,
    strategy_config_path,
    fee_rate,
    slippage_rate,
    initial_capital=5000.0,
    symbols=None
):
    symbols = symbols or list(frames.keys())
    capital_per_symbol = initial_capital / len(symbols)

    strategy_config = load_strategy_config(strategy_config_path)
    backtest_config = BacktestConfig(
        initial_capital=capital_per_symbol,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate
    )
    validate_config(strategy_config, backtest_config)

    equity_curves = []
    for symbol in symbols:
        df = frames[symbol]
        signal_df, sleeves = generate_signals(df, config=strategy_config)
        result = run_backtest(
            signal_df=signal_df, sleeves=sleeves, config=backtest_config
        )
        equity = result.equity_curve.copy()
        equity["open_time"] = pd.to_datetime(equity["open_time"], utc=True)
        equity_curves.append(
            equity.set_index("open_time")["equity"].rename(symbol)
        )

    combined = pd.concat(equity_curves, axis=1).sum(axis=1)
    combined_net_pnl = combined.pct_change().fillna(0.0)
    combined_net_pnl.iloc[0] = combined.iloc[0] / initial_capital - 1.0

    portfolio_df = pd.DataFrame({
        "open_time": combined.index,
        "net_pnl": combined_net_pnl.values,
        "equity": (combined / initial_capital).values
    }).reset_index(drop=True)

    metrics = portfolio_metrics.calculate_extended_metrics(
        portfolio_df, initial_capital=initial_capital
    )
    return {"metrics": metrics, "portfolio_equity": portfolio_df}


def run_buy_and_hold_portfolio(
    frames, fee_rate, slippage_rate, initial_capital=5000.0, symbols=None
):
    symbols = symbols or list(frames.keys())
    capital_per_symbol = initial_capital / len(symbols)

    dollar_curves = []
    for symbol in symbols:
        df = frames[symbol].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        buy_price = float(df.iloc[0]["open"]) * (1 + slippage_rate)
        buy_fee = capital_per_symbol * fee_rate
        quantity = (capital_per_symbol - buy_fee) / buy_price
        dollar_value = quantity * df["close"] * (1 - slippage_rate)
        dollar_curves.append(
            pd.Series(dollar_value.values, index=df["open_time"], name=symbol)
        )

    combined = pd.concat(dollar_curves, axis=1).sum(axis=1)
    combined_net_pnl = combined.pct_change().fillna(0.0)
    combined_net_pnl.iloc[0] = combined.iloc[0] / initial_capital - 1.0

    position = pd.Series(1.0, index=combined.index)
    turnover = pd.Series(0.0, index=combined.index)
    turnover.iloc[0] = 1.0

    portfolio_df = pd.DataFrame({
        "open_time": combined.index,
        "net_pnl": combined_net_pnl.values,
        "equity": (combined / initial_capital).values,
        "position": position.values,
        "turnover": turnover.values,
        "rebalanced": (turnover.values > 0)
    }).reset_index(drop=True)

    metrics = portfolio_metrics.calculate_extended_metrics(
        portfolio_df, initial_capital=initial_capital
    )
    benchmark_details = {
        symbol: calculate_buy_and_hold(
            frames[symbol], capital_per_symbol, fee_rate, slippage_rate
        )
        for symbol in symbols
    }
    return {
        "metrics": metrics,
        "portfolio_equity": portfolio_df,
        "per_symbol": benchmark_details
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Mini-Medallion组合回测：A/B/C/D场景对比"
    )
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--per-asset-cap", type=float, default=DEFAULT_PER_ASSET_CAP)
    parser.add_argument(
        "--legacy-config", default="configs/regime_switching.json"
    )
    parser.add_argument(
        "--stop-loss-atr-multiple", type=float, default=2.5,
        help="E场景用的ATR止损倍数，默认借用既有引擎的trend_stop_atr_multiple"
    )
    parser.add_argument("--output-folder", default="reports/mini_medallion_v1")
    return parser.parse_args()


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    end_time = _resolve_end_time(args.interval, args.end)

    frames = load_symbol_frames(
        symbols=args.symbols, interval=args.interval,
        start=args.start, end=end_time,
        cache_folder=project_folder / "data_cache"
    )
    alpha_sets = build_alpha_sets(frames)

    print("正在运行Alpha Research Gate（walk-forward）...")
    accepted, rejected, gate_df, fold_df = get_accepted_alphas(
        frames, alpha_sets, fee=args.fee, slippage=args.slippage
    )
    print(f"通过：{accepted}")

    output_folder = project_folder / args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    if not accepted:
        print("没有alpha通过Research Gate，无法构建组合，据实报告。")
        with (output_folder / "gate_only_summary.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                {"accepted": accepted, "rejected": rejected},
                file, ensure_ascii=False, indent=2
            )
        return

    scenarios = {}

    legacy_config_path = project_folder / args.legacy_config
    scenarios["A_legacy_strategy_portfolio"] = run_legacy_portfolio(
        frames, legacy_config_path, args.fee, args.slippage,
        initial_capital=args.capital, symbols=args.symbols
    )

    for method in ("equal", "ic", "corr_penalized"):
        scenarios[f"B_alpha_ensemble_{method}"] = run_alpha_ensemble_portfolio(
            frames, alpha_sets, accepted, method,
            args.fee, args.slippage,
            per_asset_cap=args.per_asset_cap,
            initial_capital=args.capital, symbols=args.symbols
        )

    scenarios["C_buy_and_hold"] = run_buy_and_hold_portfolio(
        frames, args.fee, args.slippage,
        initial_capital=args.capital, symbols=args.symbols
    )

    scenarios["D_regime_filtered_ensemble_corr_penalized"] = (
        run_alpha_ensemble_portfolio(
            frames, alpha_sets, accepted, "corr_penalized",
            args.fee, args.slippage,
            per_asset_cap=args.per_asset_cap,
            apply_regime_filter=True,
            initial_capital=args.capital, symbols=args.symbols
        )
    )

    # E场景：结构性尝试——给组合层敞口加一个逐笔ATR止损，止损倍数
    # 直接借用既有引擎StrategyConfig.trend_stop_atr_multiple的默认值
    # (2.5)，不是从回测结果里挑出来的数字。
    scenarios["E_corr_penalized_with_atr_stop_loss"] = (
        run_alpha_ensemble_portfolio(
            frames, alpha_sets, accepted, "corr_penalized",
            args.fee, args.slippage,
            per_asset_cap=args.per_asset_cap,
            stop_loss_atr_multiple=args.stop_loss_atr_multiple,
            initial_capital=args.capital, symbols=args.symbols
        )
    )

    summary_rows = []
    for name, result in scenarios.items():
        summary_rows.append({"scenario": name, **result["metrics"]})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_folder / "scenario_comparison.csv", index=False)

    for name, result in scenarios.items():
        result["portfolio_equity"].to_csv(
            output_folder / f"{name}_equity.csv", index=False
        )

    gate_df.to_csv(output_folder / "alpha_gate_results.csv", index=False)
    fold_df.to_csv(output_folder / "alpha_gate_folds.csv", index=False)

    # BTC vs ETH单资产拆分（只针对alpha ensemble场景，因为per_symbol结果
    # 已经在risk_overlay模拟里保留了，不需要重新跑）
    per_symbol_rows = []
    for scenario_name in [
        "B_alpha_ensemble_equal", "B_alpha_ensemble_ic",
        "B_alpha_ensemble_corr_penalized",
        "D_regime_filtered_ensemble_corr_penalized",
        "E_corr_penalized_with_atr_stop_loss"
    ]:
        for symbol, simulation in scenarios[scenario_name]["per_symbol"].items():
            symbol_metrics = portfolio_metrics.calculate_extended_metrics(
                simulation, initial_capital=args.capital / len(args.symbols)
            )
            per_symbol_rows.append({
                "scenario": scenario_name, "symbol": symbol, **symbol_metrics
            })
    per_symbol_df = pd.DataFrame(per_symbol_rows)
    per_symbol_df.to_csv(
        output_folder / "scenario_per_symbol_breakdown.csv", index=False
    )

    # 成本敏感性：对corr_penalized集成方式做2倍/3倍成本压力测试
    cost_sensitivity_rows = []
    for multiplier in (1, 2, 3):
        stressed = run_alpha_ensemble_portfolio(
            frames, alpha_sets, accepted, "corr_penalized",
            args.fee * multiplier, args.slippage * multiplier,
            per_asset_cap=args.per_asset_cap,
            initial_capital=args.capital, symbols=args.symbols
        )
        cost_sensitivity_rows.append({
            "cost_multiplier": multiplier, **stressed["metrics"]
        })
    cost_sensitivity_df = pd.DataFrame(cost_sensitivity_rows)
    cost_sensitivity_df.to_csv(
        output_folder / "cost_sensitivity.csv", index=False
    )

    print(summary_df.to_string(index=False))
    print("\nBTC/ETH拆分：")
    print(per_symbol_df.to_string(index=False))
    print("\n成本敏感性（corr_penalized集成）：")
    print(cost_sensitivity_df.to_string(index=False))
    print(f"\n报告已保存到：{output_folder}")


if __name__ == "__main__":
    main()
