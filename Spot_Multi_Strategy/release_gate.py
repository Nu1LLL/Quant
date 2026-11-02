import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import BacktestConfig
from config_io import load_strategy_config
from paper_trader import calculate_config_hash


MINIMUM_PAPER_DAYS = 90
MINIMUM_CLOSED_TRADES = 12
MINIMUM_PROFIT_FACTOR = 1.10
MAXIMUM_DRAWDOWN = 0.10
MAXIMUM_STATE_AGE_HOURS = 6


def load_json(file_path):
    with Path(file_path).open("r", encoding="utf-8") as file:
        return json.load(file)


def calculate_paper_metrics(state, events_df, equity_df, now=None):
    current_time = (
        datetime.now(timezone.utc)
        if now is None
        else pd.Timestamp(now).to_pydatetime()
    )
    created_at = pd.Timestamp(state["created_at"])
    last_run_at = pd.Timestamp(state["last_run_at"])
    paper_days = (
        pd.Timestamp(current_time) - created_at
    ).total_seconds() / 86400
    state_age_hours = (
        pd.Timestamp(current_time) - last_run_at
    ).total_seconds() / 3600

    closed_trade_count = sum(
        account["closed_trades"]
        for account in state["accounts"].values()
    )
    initial_capital = sum(
        account["initial_capital"]
        for account in state["accounts"].values()
    )

    if events_df.empty or "net_profit" not in events_df.columns:
        net_profits = pd.Series(dtype=float)
    else:
        net_profits = pd.to_numeric(
            events_df["net_profit"],
            errors="coerce"
        ).dropna()

    gross_profit = net_profits[net_profits > 0].sum()
    gross_loss = -net_profits[net_profits < 0].sum()

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0

    if equity_df.empty:
        total_equity = 0.0
        maximum_drawdown = 1.0
    else:
        equity_data = equity_df.copy()
        equity_data["equity"] = pd.to_numeric(
            equity_data["equity"],
            errors="raise"
        )
        total_equity_curve = (
            equity_data.groupby("run_time", sort=False)["equity"]
            .sum()
        )
        total_equity = float(total_equity_curve.iloc[-1])
        equity_with_start = pd.concat(
            [
                pd.Series([initial_capital]),
                total_equity_curve.reset_index(drop=True)
            ],
            ignore_index=True
        )
        running_peak = equity_with_start.cummax()
        drawdown = equity_with_start / running_peak - 1
        maximum_drawdown = abs(float(drawdown.min()))
    net_profit = total_equity - initial_capital

    return {
        "paper_days": paper_days,
        "state_age_hours": state_age_hours,
        "closed_trade_count": closed_trade_count,
        "profit_factor": profit_factor,
        "maximum_drawdown": maximum_drawdown,
        "initial_capital": initial_capital,
        "total_equity": total_equity,
        "net_profit": net_profit
    }


def evaluate_release_gate(
    rolling_summary,
    robustness_summary,
    state,
    metrics,
    expected_config_hash,
    emergency_stop_exists=False
):
    checks = {
        "历史滚动验证通过": bool(
            rolling_summary.get("gate_passed", False)
        ),
        "历史压力测试通过": bool(
            robustness_summary.get("gate_passed", False)
        ),
        "模拟参数从未改变": (
            state["config_hash"] == expected_config_hash
        ),
        f"连续模拟不少于{MINIMUM_PAPER_DAYS}天": (
            metrics["paper_days"] >= MINIMUM_PAPER_DAYS
        ),
        f"完整模拟交易不少于{MINIMUM_CLOSED_TRADES}笔": (
            metrics["closed_trade_count"]
            >=
            MINIMUM_CLOSED_TRADES
        ),
        "模拟期扣费后净盈利": metrics["net_profit"] > 0,
        f"模拟期利润因子不低于{MINIMUM_PROFIT_FACTOR:.2f}": (
            metrics["profit_factor"] >= MINIMUM_PROFIT_FACTOR
        ),
        f"模拟期最大回撤不超过{MAXIMUM_DRAWDOWN:.0%}": (
            metrics["maximum_drawdown"] <= MAXIMUM_DRAWDOWN
        ),
        "没有发生K线断档": state["data_gap_count"] == 0,
        f"状态更新不超过{MAXIMUM_STATE_AGE_HOURS}小时": (
            metrics["state_age_hours"] <= MAXIMUM_STATE_AGE_HOURS
        ),
        "紧急停止开关未启用": not emergency_stop_exists
    }
    return checks, all(checks.values())


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="检查策略是否允许进入极小资金实盘准备阶段"
    )
    parser.add_argument(
        "--strategy-config",
        default="configs/trend_champion.json"
    )
    parser.add_argument(
        "--rolling-summary",
        default="rolling_reports/trend_champion_summary.json"
    )
    parser.add_argument(
        "--robustness-summary",
        default="robustness_reports/trend_champion_summary.json"
    )
    parser.add_argument("--state-folder", default="paper_state")
    parser.add_argument("--capital", type=float, default=5000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--risk", type=float, default=0.005)
    return parser.parse_args()


def resolve_path(project_folder, value):
    path = Path(value)
    return path if path.is_absolute() else project_folder / path


def main():
    args = parse_arguments()
    project_folder = Path(__file__).resolve().parent
    strategy_path = resolve_path(
        project_folder,
        args.strategy_config
    )
    state_folder = resolve_path(project_folder, args.state_folder)
    state = load_json(state_folder / "state.json")
    rolling_summary = load_json(
        resolve_path(project_folder, args.rolling_summary)
    )
    robustness_summary = load_json(
        resolve_path(project_folder, args.robustness_summary)
    )
    events_path = state_folder / "events.csv"
    equity_path = state_folder / "equity.csv"
    events_df = (
        pd.read_csv(events_path)
        if events_path.exists()
        else pd.DataFrame()
    )
    equity_df = pd.read_csv(equity_path)
    strategy_config = load_strategy_config(strategy_path)
    backtest_config = BacktestConfig(
        initial_capital=args.capital,
        fee_rate=args.fee,
        slippage_rate=args.slippage,
        risk_per_trade=args.risk
    )
    expected_config_hash = calculate_config_hash(
        strategy_config,
        backtest_config
    )
    metrics = calculate_paper_metrics(
        state,
        events_df,
        equity_df
    )
    checks, passed = evaluate_release_gate(
        rolling_summary,
        robustness_summary,
        state,
        metrics,
        expected_config_hash,
        emergency_stop_exists=(
            state_folder / "EMERGENCY_STOP"
        ).exists()
    )

    print("\n实盘放行检查：")

    for name, result in checks.items():
        marker = "通过" if result else "未通过"
        print(f"{marker}：{name}")

    print("\n模拟盘统计：")
    print(f"运行天数：{metrics['paper_days']:.1f}天")
    print(f"完整交易：{metrics['closed_trade_count']}笔")
    print(f"当前权益：{metrics['total_equity']:.2f} USDT")
    print(f"净盈亏：{metrics['net_profit']:.2f} USDT")
    print(f"利润因子：{metrics['profit_factor']:.2f}")
    print(f"最大回撤：{metrics['maximum_drawdown']:.2%}")

    if passed:
        print("\n结论：允许进入极小资金实盘准备阶段。")
        print("注意：这不代表自动下单，也不保证未来盈利。")
    else:
        print("\n结论：禁止实盘，继续只读模拟。")


if __name__ == "__main__":
    main()
