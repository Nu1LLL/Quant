from dataclasses import dataclass

import pandas as pd

from config import BacktestConfig


@dataclass
class SleeveState:
    # 当前剩余现金
    cash: float

    # 当前持有的现货数量
    quantity: float = 0.0

    # 买入成交价格
    entry_price: float = 0.0

    # 买入时间
    entry_time: object = None

    # 买入手续费
    entry_fee: float = 0.0

    # 当前止损价格
    stop_price: float = 0.0

    # 持仓后出现的最高收盘价
    highest_close: float = 0.0

    # 本笔交易进场前的策略账户权益
    entry_account_value: float = 0.0


@dataclass
class BacktestResult:
    # 初始资金
    initial_capital: float

    # 最终资产
    final_value: float

    # 累计手续费
    total_fees: float

    # 组合权益曲线
    equity_curve: pd.DataFrame

    # 完整交易记录
    trades: pd.DataFrame


def _sell_position(
    state,
    strategy_name,
    exit_time,
    raw_exit_price,
    exit_reason,
    fee_rate,
    slippage_rate,
    trades
):
    # 卖出时按不利方向加入滑点
    exit_price = raw_exit_price * (1 - slippage_rate)

    gross_value = state.quantity * exit_price
    exit_fee = gross_value * fee_rate
    state.cash += gross_value - exit_fee

    gross_profit = (
        exit_price - state.entry_price
    ) * state.quantity

    net_profit = (
        gross_profit
        -
        state.entry_fee
        -
        exit_fee
    )

    entry_cost = (
        state.entry_price * state.quantity
        +
        state.entry_fee
    )

    trade_return = (
        net_profit / entry_cost
        if entry_cost > 0
        else 0.0
    )

    trades.append(
        {
            "strategy": strategy_name,
            "entry_time": state.entry_time,
            "exit_time": exit_time,
            "entry_price": state.entry_price,
            "exit_price": exit_price,
            "quantity": state.quantity,
            "entry_fee": state.entry_fee,
            "exit_fee": exit_fee,
            "entry_account_value": state.entry_account_value,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "return_rate": trade_return,
            "exit_reason": exit_reason
        }
    )

    state.quantity = 0.0
    state.entry_price = 0.0
    state.entry_time = None
    state.entry_fee = 0.0
    state.stop_price = 0.0
    state.highest_close = 0.0
    state.entry_account_value = 0.0

    return exit_fee


def run_backtest(
    signal_df,
    sleeves,
    config=None
):
    # 使用传入配置，未传入时使用默认回测参数
    backtest_config = config or BacktestConfig()
    df = signal_df.copy().reset_index(drop=True)

    required_columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "atr"
    ]

    for sleeve in sleeves:
        required_columns.extend(
            [
                sleeve.entry_column,
                sleeve.exit_column
            ]
        )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"回测数据缺少必要字段：{missing_columns}"
        )

    if len(df) < 2:
        raise ValueError("回测至少需要两根K线")

    total_weight = sum(sleeve.weight for sleeve in sleeves)

    if total_weight > 1.0 + 1e-12:
        raise ValueError("策略资金权重之和不能超过1")

    if any(sleeve.weight <= 0 for sleeve in sleeves):
        raise ValueError("每个策略的资金权重必须大于0")

    states = {
        sleeve.name: SleeveState(
            cash=(
                backtest_config.initial_capital
                *
                sleeve.weight
            )
        )
        for sleeve in sleeves
    }

    reserve_cash = (
        backtest_config.initial_capital
        *
        (1 - total_weight)
    )

    trades = []
    equity_records = [
        {
            "open_time": df.iloc[0]["open_time"],
            "equity": backtest_config.initial_capital
        }
    ]
    total_fees = 0.0

    # 信号按照设定的延迟，在未来某根K线开盘时执行
    for i in range(1, len(df)):
        signal_position = (
            i - backtest_config.execution_delay_bars
        )
        signal_row = (
            df.iloc[signal_position]
            if signal_position >= 0
            else None
        )
        current_bar = df.iloc[i]

        for sleeve in sleeves:
            state = states[sleeve.name]
            exited_this_bar = False
            # 同一根K线已经卖出后，不允许立刻再次买入

            if state.quantity > 0:
                # 离场信号在当前K线开盘时优先执行
                if (
                    signal_row is not None
                    and
                    bool(signal_row[sleeve.exit_column])
                ):
                    total_fees += _sell_position(
                        state=state,
                        strategy_name=sleeve.name,
                        exit_time=current_bar["open_time"],
                        raw_exit_price=float(current_bar["open"]),
                        exit_reason="signal",
                        fee_rate=backtest_config.fee_rate,
                        slippage_rate=(
                            backtest_config.slippage_rate
                        ),
                        trades=trades
                    )
                    exited_this_bar = True

                # 没有离场信号时，检查当前K线是否触发止损
                elif float(current_bar["low"]) <= state.stop_price:
                    if float(current_bar["open"]) <= state.stop_price:
                        raw_exit_price = float(current_bar["open"])
                    else:
                        raw_exit_price = state.stop_price

                    total_fees += _sell_position(
                        state=state,
                        strategy_name=sleeve.name,
                        exit_time=current_bar["open_time"],
                        raw_exit_price=raw_exit_price,
                        exit_reason="stop",
                        fee_rate=backtest_config.fee_rate,
                        slippage_rate=(
                            backtest_config.slippage_rate
                        ),
                        trades=trades
                    )
                    exited_this_bar = True

            if (
                state.quantity == 0
                and
                not exited_this_bar
                and
                signal_row is not None
                and
                bool(signal_row[sleeve.entry_column])
            ):
                signal_atr = float(signal_row["atr"])

                if signal_atr > 0 and pd.notna(signal_atr):
                    # 买入时按不利方向加入滑点
                    entry_price = (
                        float(current_bar["open"])
                        *
                        (1 + backtest_config.slippage_rate)
                    )

                    stop_price = (
                        entry_price
                        -
                        signal_atr
                        *
                        sleeve.stop_atr_multiple
                    )

                    if stop_price > 0:
                        estimated_stop_fill = (
                            stop_price
                            *
                            (1 - backtest_config.slippage_rate)
                        )

                        risk_per_unit = (
                            entry_price
                            -
                            estimated_stop_fill
                            +
                            entry_price * backtest_config.fee_rate
                            +
                            estimated_stop_fill
                            *
                            backtest_config.fee_rate
                        )

                        risk_budget = (
                            state.cash
                            *
                            backtest_config.risk_per_trade
                        )

                        quantity_by_risk = (
                            risk_budget / risk_per_unit
                            if risk_per_unit > 0
                            else 0.0
                        )

                        maximum_gross_cost = (
                            state.cash
                            *
                            backtest_config.max_sleeve_exposure
                            /
                            (1 + backtest_config.fee_rate)
                        )

                        quantity_by_cash = (
                            maximum_gross_cost / entry_price
                        )

                        quantity = min(
                            quantity_by_risk,
                            quantity_by_cash
                        )

                        if quantity > 0:
                            entry_account_value = state.cash
                            gross_cost = quantity * entry_price
                            entry_fee = (
                                gross_cost
                                *
                                backtest_config.fee_rate
                            )

                            state.cash -= gross_cost + entry_fee
                            state.quantity = quantity
                            state.entry_price = entry_price
                            state.entry_time = current_bar["open_time"]
                            state.entry_fee = entry_fee
                            state.entry_account_value = (
                                entry_account_value
                            )
                            state.stop_price = stop_price
                            state.highest_close = float(
                                current_bar["close"]
                            )

                            total_fees += entry_fee

                            # 开盘买入后，本根K线盘中止损立即生效
                            if (
                                float(current_bar["low"])
                                <= state.stop_price
                            ):
                                total_fees += _sell_position(
                                    state=state,
                                    strategy_name=sleeve.name,
                                    exit_time=current_bar["open_time"],
                                    raw_exit_price=state.stop_price,
                                    exit_reason="same_bar_stop",
                                    fee_rate=backtest_config.fee_rate,
                                    slippage_rate=(
                                        backtest_config.slippage_rate
                                    ),
                                    trades=trades
                                )

            # 收盘以后更新移动止损，新的止损从下一根K线开始生效
            if (
                state.quantity > 0
                and
                sleeve.trailing_atr_multiple is not None
                and
                pd.notna(current_bar["atr"])
            ):
                state.highest_close = max(
                    state.highest_close,
                    float(current_bar["close"])
                )

                trailing_stop = (
                    state.highest_close
                    -
                    float(current_bar["atr"])
                    *
                    sleeve.trailing_atr_multiple
                )

                state.stop_price = max(
                    state.stop_price,
                    trailing_stop
                )

        portfolio_equity = reserve_cash

        for state in states.values():
            portfolio_equity += (
                state.cash
                +
                state.quantity * float(current_bar["close"])
            )

        equity_records.append(
            {
                "open_time": current_bar["open_time"],
                "equity": portfolio_equity
            }
        )

    if backtest_config.liquidate_at_end:
        last_bar = df.iloc[-1]

        for sleeve in sleeves:
            state = states[sleeve.name]

            if state.quantity > 0:
                total_fees += _sell_position(
                    state=state,
                    strategy_name=sleeve.name,
                    exit_time=last_bar["open_time"],
                    raw_exit_price=float(last_bar["close"]),
                    exit_reason="end_of_backtest",
                    fee_rate=backtest_config.fee_rate,
                    slippage_rate=backtest_config.slippage_rate,
                    trades=trades
                )

        final_value = (
            reserve_cash
            +
            sum(state.cash for state in states.values())
        )

        equity_records[-1]["equity"] = final_value

    else:
        final_value = equity_records[-1]["equity"]

    equity_curve = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)

    return BacktestResult(
        initial_capital=backtest_config.initial_capital,
        final_value=final_value,
        total_fees=total_fees,
        equity_curve=equity_curve,
        trades=trades_df
    )
