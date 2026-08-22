import math

import numpy as np
import pandas as pd


def calculate_metrics(result):
    # 复制并按时间排列权益曲线
    equity_curve = result.equity_curve.copy()
    equity_curve["open_time"] = pd.to_datetime(
        equity_curve["open_time"],
        utc=True
    )
    equity_curve = equity_curve.sort_values("open_time")

    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().dropna()

    total_return = (
        result.final_value / result.initial_capital - 1
    )

    elapsed_days = (
        equity_curve["open_time"].iloc[-1]
        -
        equity_curve["open_time"].iloc[0]
    ).total_seconds() / 86400

    if elapsed_days > 0 and result.final_value > 0:
        annualized_return = (
            result.final_value / result.initial_capital
        ) ** (365.25 / elapsed_days) - 1
    else:
        annualized_return = 0.0

    running_high = equity.cummax()
    drawdown = equity / running_high - 1
    max_drawdown = float(drawdown.min())

    if len(equity_curve) > 1:
        bar_seconds = (
            equity_curve["open_time"].diff().dropna().median()
            .total_seconds()
        )
        periods_per_year = (
            365.25 * 86400 / bar_seconds
            if bar_seconds > 0
            else 0
        )
    else:
        periods_per_year = 0

    if (
        not returns.empty
        and returns.std(ddof=0) > 0
        and periods_per_year > 0
    ):
        sharpe_ratio = (
            returns.mean()
            /
            returns.std(ddof=0)
            *
            math.sqrt(periods_per_year)
        )
    else:
        sharpe_ratio = 0.0

    trades = result.trades.copy()
    trade_count = len(trades)

    if trade_count > 0:
        winning_trades = trades[trades["net_profit"] > 0]
        losing_trades = trades[trades["net_profit"] < 0]

        win_rate = len(winning_trades) / trade_count
        gross_profit = winning_trades["net_profit"].sum()
        gross_loss = -losing_trades["net_profit"].sum()

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        average_trade = trades["net_profit"].mean()
    else:
        win_rate = 0.0
        profit_factor = 0.0
        average_trade = 0.0

    return {
        "initial_capital": result.initial_capital,
        "final_value": result.final_value,
        "profit": result.final_value - result.initial_capital,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "total_fees": result.total_fees,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "elapsed_days": elapsed_days
    }


def calculate_monthly_report(result):
    equity_curve = result.equity_curve.copy()
    equity_curve["open_time"] = pd.to_datetime(
        equity_curve["open_time"],
        utc=True
    )

    equity_curve = equity_curve.set_index("open_time")
    monthly_equity = equity_curve["equity"].resample("ME").last()

    monthly_start = monthly_equity.shift(1)

    if not monthly_start.empty:
        monthly_start.iloc[0] = result.initial_capital

    monthly_profit = monthly_equity - monthly_start
    monthly_return = monthly_equity / monthly_start - 1

    return pd.DataFrame(
        {
            "ending_equity": monthly_equity,
            "monthly_profit": monthly_profit,
            "monthly_return": monthly_return
        }
    ).dropna(subset=["ending_equity"])


def calculate_monthly_summary(monthly_report):
    if monthly_report.empty:
        return {
            "month_count": 0,
            "average_monthly_profit": 0.0,
            "median_monthly_profit": 0.0,
            "best_month_profit": 0.0,
            "worst_month_profit": 0.0,
            "profitable_month_ratio": 0.0,
            "average_monthly_return": 0.0
        }

    monthly_profit = monthly_report["monthly_profit"]
    monthly_return = monthly_report["monthly_return"]

    return {
        "month_count": len(monthly_report),
        "average_monthly_profit": monthly_profit.mean(),
        "median_monthly_profit": monthly_profit.median(),
        "best_month_profit": monthly_profit.max(),
        "worst_month_profit": monthly_profit.min(),
        "profitable_month_ratio": (monthly_profit > 0).mean(),
        "average_monthly_return": monthly_return.mean()
    }


def calculate_strategy_report(result):
    if result.trades.empty:
        return pd.DataFrame(
            columns=[
                "strategy",
                "trades",
                "net_profit",
                "win_rate",
                "average_trade",
                "total_fees"
            ]
        )

    trades = result.trades.copy()
    trades["total_trade_fee"] = (
        trades["entry_fee"] + trades["exit_fee"]
    )

    strategy_report = (
        trades.groupby("strategy")
        .agg(
            trades=("net_profit", "size"),
            net_profit=("net_profit", "sum"),
            win_rate=(
                "net_profit",
                lambda values: (values > 0).mean()
            ),
            average_trade=("net_profit", "mean"),
            total_fees=("total_trade_fee", "sum")
        )
        .reset_index()
    )

    return strategy_report


def format_monthly_summary(summary):
    return "\n".join(
        [
            "月度统计：",
            f"日历月份：{summary['month_count']}",
            (
                "平均每月利润："
                f"{summary['average_monthly_profit']:.2f} USDT"
            ),
            (
                "月利润中位数："
                f"{summary['median_monthly_profit']:.2f} USDT"
            ),
            (
                "最好月份利润："
                f"{summary['best_month_profit']:.2f} USDT"
            ),
            (
                "最差月份利润："
                f"{summary['worst_month_profit']:.2f} USDT"
            ),
            (
                "盈利月份比例："
                f"{summary['profitable_month_ratio'] * 100:.2f}%"
            ),
            (
                "平均月收益率："
                f"{summary['average_monthly_return'] * 100:.2f}%"
            )
        ]
    )


def format_strategy_report(strategy_report):
    if strategy_report.empty:
        return "子策略统计：没有完整交易"

    lines = ["子策略统计："]

    for row in strategy_report.itertuples(index=False):
        lines.append(
            f"{row.strategy}："
            f"{row.trades}笔，"
            f"盈亏{row.net_profit:.2f} USDT，"
            f"胜率{row.win_rate * 100:.2f}%，"
            f"平均每笔{row.average_trade:.2f} USDT，"
            f"手续费{row.total_fees:.2f} USDT"
        )

    return "\n".join(lines)


def evaluate_research_gates(metrics):
    # 这些门槛只用于筛掉明显不合格策略，不代表通过后一定赚钱
    checks = {
        "样本外收益为正": metrics["total_return"] > 0,
        "样本外完整交易不少于30笔": metrics["trade_count"] >= 30,
        "样本外利润因子不低于1.20": (
            metrics["profit_factor"] >= 1.20
        ),
        "样本外Sharpe不低于0.50": (
            metrics["sharpe_ratio"] >= 0.50
        ),
        "样本外最大回撤不超过20%": (
            metrics["max_drawdown"] >= -0.20
        )
    }

    return {
        "passed": all(checks.values()),
        "checks": checks
    }


def format_research_gates(gate_result):
    if gate_result["passed"]:
        title = "研究验收：通过基础门槛，下一步仍需模拟盘"
    else:
        title = "研究验收：未通过，不可连接实盘"

    lines = [title]

    for name, passed in gate_result["checks"].items():
        status = "通过" if passed else "未通过"
        lines.append(f"{status}：{name}")

    return "\n".join(lines)


def calculate_buy_and_hold(
    df,
    initial_capital,
    fee_rate,
    slippage_rate
):
    # 第一个开盘价买入，最后一个收盘价卖出
    buy_price = float(df.iloc[0]["open"]) * (1 + slippage_rate)
    buy_fee = initial_capital * fee_rate
    quantity = (initial_capital - buy_fee) / buy_price

    sell_price = float(df.iloc[-1]["close"]) * (1 - slippage_rate)
    gross_value = quantity * sell_price
    sell_fee = gross_value * fee_rate
    final_value = gross_value - sell_fee

    return {
        "final_value": final_value,
        "profit": final_value - initial_capital,
        "return_rate": final_value / initial_capital - 1,
        "total_fees": buy_fee + sell_fee
    }


def format_metrics(title, metrics, benchmark=None):
    profit_factor = metrics["profit_factor"]

    if np.isinf(profit_factor):
        profit_factor_text = "无限（没有亏损交易）"
    else:
        profit_factor_text = f"{profit_factor:.2f}"

    lines = [
        f"\n{title}",
        f"回测天数：{metrics['elapsed_days']:.1f}",
        f"初始资金：{metrics['initial_capital']:.2f} USDT",
        f"最终资产：{metrics['final_value']:.2f} USDT",
        f"总盈亏：{metrics['profit']:.2f} USDT",
        f"总收益率：{metrics['total_return'] * 100:.2f}%",
        f"折算年化：{metrics['annualized_return'] * 100:.2f}%",
        f"最大回撤：{metrics['max_drawdown'] * 100:.2f}%",
        f"Sharpe：{metrics['sharpe_ratio']:.2f}",
        f"总手续费：{metrics['total_fees']:.2f} USDT",
        f"完整交易：{metrics['trade_count']}",
        f"胜率：{metrics['win_rate'] * 100:.2f}%",
        f"利润因子：{profit_factor_text}",
        f"平均每笔：{metrics['average_trade']:.2f} USDT"
    ]

    if benchmark is not None:
        lines.extend(
            [
                "",
                "同期买入持有：",
                f"最终资产：{benchmark['final_value']:.2f} USDT",
                f"收益率：{benchmark['return_rate'] * 100:.2f}%",
                f"手续费：{benchmark['total_fees']:.2f} USDT"
            ]
        )

    return "\n".join(lines)
