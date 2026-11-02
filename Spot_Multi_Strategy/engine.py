from dataclasses import dataclass

import pandas as pd

from config import BacktestConfig


def _fee_rate_for_side(backtest_config, side):
    if side == "short" and backtest_config.short_fee_rate is not None:
        return float(backtest_config.short_fee_rate)
    return float(backtest_config.fee_rate)


@dataclass
class SleeveState:
    cash: float
    quantity: float = 0.0
    side: str = "flat"
    entry_price: float = 0.0
    raw_entry_price: float = 0.0
    entry_time: object = None
    entry_fee: float = 0.0
    entry_slippage: float = 0.0
    entry_account_value: float = 0.0
    locked_margin: float = 0.0
    stop_price: float = 0.0
    target_price: float = 0.0
    highest_close: float = 0.0
    lowest_close: float = 0.0
    funding_fee: float = 0.0
    entry_bar_index: int | None = None


@dataclass
class BacktestResult:
    initial_capital: float
    final_value: float
    total_fees: float
    total_trading_fees: float
    total_funding_fees: float
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    total_slippage_cost: float = 0.0


def _reset_position(state):
    state.quantity = 0.0
    state.side = "flat"
    state.entry_price = 0.0
    state.raw_entry_price = 0.0
    state.entry_time = None
    state.entry_fee = 0.0
    state.entry_slippage = 0.0
    state.entry_account_value = 0.0
    state.locked_margin = 0.0
    state.stop_price = 0.0
    state.target_price = 0.0
    state.highest_close = 0.0
    state.lowest_close = 0.0
    state.funding_fee = 0.0
    state.entry_bar_index = None


def _close_position(
    state,
    strategy_name,
    exit_time,
    raw_exit_price,
    exit_reason,
    fee_rate,
    slippage_rate,
    trades
):
    side = state.side
    if side == "long":
        exit_price = raw_exit_price * (1 - slippage_rate)
        gross_value = state.quantity * exit_price
        exit_fee = gross_value * fee_rate
        gross_profit = (exit_price - state.entry_price) * state.quantity
        gross_market_profit = (
            raw_exit_price - state.raw_entry_price
        ) * state.quantity
        state.cash += gross_value - exit_fee
        capital_base = state.entry_price * state.quantity + state.entry_fee
    elif side == "short":
        exit_price = raw_exit_price * (1 + slippage_rate)
        gross_value = state.quantity * exit_price
        exit_fee = gross_value * fee_rate
        gross_profit = (state.entry_price - exit_price) * state.quantity
        gross_market_profit = (
            state.raw_entry_price - raw_exit_price
        ) * state.quantity
        state.cash += state.locked_margin + gross_profit - exit_fee
        capital_base = state.locked_margin + state.entry_fee
    else:
        raise ValueError("无法平掉无方向的持仓")

    entry_slippage = state.entry_slippage
    exit_slippage = abs(exit_price - raw_exit_price) * state.quantity
    slippage_cost = entry_slippage + exit_slippage
    funding_fee = state.funding_fee
    net_profit = (
        gross_profit - state.entry_fee - exit_fee - funding_fee
    )
    trade_return = net_profit / capital_base if capital_base > 0 else 0.0
    trades.append(
        {
            "strategy": strategy_name,
            "side": side,
            "entry_time": state.entry_time,
            "exit_time": exit_time,
            "entry_price": state.entry_price,
            "exit_price": exit_price,
            "quantity": state.quantity,
            "entry_fee": state.entry_fee,
            "exit_fee": exit_fee,
            "trading_fee": state.entry_fee + exit_fee,
            "funding_fee": funding_fee,
            "entry_slippage": entry_slippage,
            "exit_slippage": exit_slippage,
            "slippage_cost": slippage_cost,
            "entry_account_value": state.entry_account_value,
            "gross_market_profit": gross_market_profit,
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "return_rate": trade_return,
            "exit_reason": exit_reason
        }
    )
    _reset_position(state)
    return exit_fee


def _position_equity(state, mark_price):
    if state.side == "long":
        return state.cash + state.quantity * mark_price
    if state.side == "short":
        unrealized_profit = (
            state.entry_price - mark_price
        ) * state.quantity
        return state.cash + state.locked_margin + unrealized_profit
    return state.cash


def _open_position(
    state,
    side,
    event_time,
    raw_entry_price,
    signal_atr,
    sleeve,
    backtest_config,
    structural_stop_price=None,
    entry_bar_index=None
):
    fee_rate = _fee_rate_for_side(backtest_config, side)
    adverse_entry_multiplier = (
        1 + backtest_config.slippage_rate
        if side == "long"
        else 1 - backtest_config.slippage_rate
    )
    entry_price = raw_entry_price * adverse_entry_multiplier
    if structural_stop_price is not None:
        stop_price = float(structural_stop_price)
    else:
        if signal_atr is None or sleeve.stop_atr_multiple is None:
            return 0.0
        stop_distance = signal_atr * sleeve.stop_atr_multiple
        stop_price = (
            entry_price - stop_distance
            if side == "long"
            else entry_price + stop_distance
        )
    valid_stop = (
        0 < stop_price < entry_price
        if side == "long"
        else stop_price > entry_price
    )
    if not valid_stop:
        return 0.0
    stop_distance = abs(entry_price - stop_price)
    stop_fraction = stop_distance / entry_price
    if (
        sleeve.minimum_stop_fraction is not None
        and stop_fraction < sleeve.minimum_stop_fraction
    ):
        return 0.0
    if (
        sleeve.maximum_stop_fraction is not None
        and stop_fraction > sleeve.maximum_stop_fraction
    ):
        return 0.0

    estimated_stop_fill = stop_price * (
        1 - backtest_config.slippage_rate
        if side == "long"
        else 1 + backtest_config.slippage_rate
    )
    price_risk = (
        entry_price - estimated_stop_fill
        if side == "long"
        else estimated_stop_fill - entry_price
    )
    risk_per_unit = (
        price_risk
        + entry_price * fee_rate
        + estimated_stop_fill * fee_rate
    )
    target_price = 0.0
    if sleeve.take_profit_r_multiple is not None:
        if sleeve.cost_adjusted_target:
            # target_price is the raw market trigger.  Solve it from the
            # desired reward after adverse exit slippage and both trading
            # fees, using the same all-in stop risk that sizes the position.
            # Funding remains separate because it is unknowable at entry.
            reward = risk_per_unit * sleeve.take_profit_r_multiple
            if side == "long":
                target_price = (
                    entry_price * (1 + fee_rate) + reward
                ) / (
                    (1 - backtest_config.slippage_rate) * (1 - fee_rate)
                )
            else:
                target_price = (
                    entry_price * (1 - fee_rate) - reward
                ) / (
                    (1 + backtest_config.slippage_rate) * (1 + fee_rate)
                )
        else:
            target_distance = stop_distance * sleeve.take_profit_r_multiple
            target_price = (
                entry_price + target_distance
                if side == "long"
                else entry_price - target_distance
            )
        if target_price <= 0:
            return 0.0
    risk_budget = state.cash * backtest_config.risk_per_trade
    quantity_by_risk = (
        risk_budget / risk_per_unit if risk_per_unit > 0 else 0.0
    )

    # 多头现货成本或空头保证金都不得超过策略权益，即最高1倍。
    exposure_notional = (
        state.cash
        * backtest_config.max_sleeve_exposure
        /
        (1 + fee_rate)
    )
    absolute_cap = (
        backtest_config.max_long_notional
        if side == "long"
        else backtest_config.max_short_notional
    )
    maximum_notional = min(exposure_notional, absolute_cap)
    quantity_by_cash = maximum_notional / entry_price
    quantity = min(quantity_by_risk, quantity_by_cash)
    if quantity <= 0:
        return 0.0

    entry_account_value = state.cash
    entry_notional = quantity * entry_price
    entry_fee = entry_notional * fee_rate
    state.cash -= entry_notional + entry_fee
    state.quantity = quantity
    state.side = side
    state.entry_price = entry_price
    state.raw_entry_price = raw_entry_price
    state.entry_time = event_time
    state.entry_fee = entry_fee
    state.entry_slippage = (
        abs(entry_price - raw_entry_price) * quantity
    )
    state.entry_account_value = entry_account_value
    state.locked_margin = entry_notional if side == "short" else 0.0
    state.stop_price = stop_price
    state.highest_close = entry_price
    state.lowest_close = entry_price
    state.funding_fee = 0.0
    state.entry_bar_index = entry_bar_index
    state.target_price = target_price
    return entry_fee


def _intrabar_exit(state, current_bar):
    bar_open = float(current_bar["open"])
    bar_high = float(current_bar["high"])
    bar_low = float(current_bar["low"])

    if state.side == "long":
        if bar_open <= state.stop_price:
            return bar_open, "stop"
        if state.target_price > 0 and bar_open >= state.target_price:
            return bar_open, "take_profit"
        # 同根K线同时触发时保守地按止损处理。
        if bar_low <= state.stop_price:
            return state.stop_price, "stop"
        if state.target_price > 0 and bar_high >= state.target_price:
            return state.target_price, "take_profit"
    elif state.side == "short":
        if bar_open >= state.stop_price:
            return bar_open, "stop"
        if state.target_price > 0 and bar_open <= state.target_price:
            return bar_open, "take_profit"
        if bar_high >= state.stop_price:
            return state.stop_price, "stop"
        if state.target_price > 0 and bar_low <= state.target_price:
            return state.target_price, "take_profit"
    return None


def _open_gap_exit(state, current_bar):
    """Evaluate only protection already crossed at the current bar open."""
    bar_open = float(current_bar["open"])
    if state.side == "long":
        if bar_open <= state.stop_price:
            return bar_open, "stop"
        if state.target_price > 0 and bar_open >= state.target_price:
            return bar_open, "take_profit"
    elif state.side == "short":
        if bar_open >= state.stop_price:
            return bar_open, "stop"
        if state.target_price > 0 and bar_open <= state.target_price:
            return bar_open, "take_profit"
    return None


def _within_bar_exit(state, current_bar):
    """Evaluate high/low after all actions executable at the bar open."""
    bar_high = float(current_bar["high"])
    bar_low = float(current_bar["low"])
    if state.side == "long":
        # Same-bar ambiguity is deliberately resolved against the strategy.
        if bar_low <= state.stop_price:
            return state.stop_price, "stop"
        if state.target_price > 0 and bar_high >= state.target_price:
            return state.target_price, "take_profit"
    elif state.side == "short":
        if bar_high >= state.stop_price:
            return state.stop_price, "stop"
        if state.target_price > 0 and bar_low <= state.target_price:
            return state.target_price, "take_profit"
    return None


def run_backtest(signal_df, sleeves, config=None):
    backtest_config = config or BacktestConfig()
    df = signal_df.copy().reset_index(drop=True)
    required_columns = ["open_time", "open", "high", "low", "close"]
    for sleeve in sleeves:
        required_columns.extend([sleeve.entry_column, sleeve.exit_column])
        if sleeve.long_stop_column is not None:
            required_columns.append(sleeve.long_stop_column)
        if sleeve.short_stop_column is not None:
            required_columns.append(sleeve.short_stop_column)
        if (
            sleeve.long_stop_column is None
            and sleeve.stop_atr_multiple is not None
            or sleeve.short_entry_column is not None
            and sleeve.short_stop_column is None
            and sleeve.stop_atr_multiple is not None
            or sleeve.trailing_atr_multiple is not None
        ):
            required_columns.append("atr")
        if sleeve.short_entry_column is not None:
            required_columns.append(sleeve.short_entry_column)
        if sleeve.short_exit_column is not None:
            required_columns.append(sleeve.short_exit_column)

    missing_columns = sorted(set(required_columns) - set(df.columns))
    if missing_columns:
        raise ValueError(f"回测数据缺少必要字段：{missing_columns}")
    if len(df) < 2:
        raise ValueError("回测至少需要两根K线")

    total_weight = sum(sleeve.weight for sleeve in sleeves)
    if total_weight > 1.0 + 1e-12:
        raise ValueError("策略资金权重之和不能超过1")
    if any(sleeve.weight <= 0 for sleeve in sleeves):
        raise ValueError("每个策略的资金权重必须大于0")

    states = {
        sleeve.name: SleeveState(
            cash=backtest_config.initial_capital * sleeve.weight
        )
        for sleeve in sleeves
    }
    reserve_cash = backtest_config.initial_capital * (1 - total_weight)
    trades = []
    equity_records = [{
        "open_time": df.iloc[0]["open_time"],
        "equity": backtest_config.initial_capital
    }]
    total_trading_fees = 0.0
    total_funding_fees = 0.0

    for i in range(1, len(df)):
        signal_position = i - backtest_config.execution_delay_bars
        signal_row = df.iloc[signal_position] if signal_position >= 0 else None
        current_bar = df.iloc[i]

        for sleeve in sleeves:
            state = states[sleeve.name]
            exited_this_bar = False

            # 资金费在结算时点只作用于已持有的空头。
            # funding_fee为正表示成本，为负表示空头收到资金费。
            if state.side == "short":
                funding_rate = float(current_bar.get("funding_rate", 0.0))
                if pd.notna(funding_rate) and funding_rate != 0:
                    funding_fee = (
                        -state.quantity
                        * float(current_bar["open"])
                        * funding_rate
                    )
                    state.cash -= funding_fee
                    state.funding_fee += funding_fee
                    total_funding_fees += funding_fee

            if state.side != "flat":
                exit_column = (
                    sleeve.exit_column
                    if state.side == "long"
                    else sleeve.short_exit_column
                )
                open_gap_exit = _open_gap_exit(state, current_bar)
                signal_exit = (
                    signal_row is not None
                    and exit_column is not None
                    and bool(signal_row[exit_column])
                )
                timeout_exit = (
                    sleeve.max_holding_bars is not None
                    and state.entry_bar_index is not None
                    and i - state.entry_bar_index >= sleeve.max_holding_bars
                )
                # Only opening-gap protection may precede an exit scheduled
                # for this open.  Looking at the current minute's high/low
                # before executing a signal/timeout would be lookahead.
                if open_gap_exit is not None:
                    raw_exit_price, exit_reason = open_gap_exit
                elif timeout_exit:
                    raw_exit_price = float(current_bar["open"])
                    exit_reason = "timeout"
                elif signal_exit:
                    raw_exit_price = float(current_bar["open"])
                    exit_reason = "signal"
                else:
                    within_bar_exit = _within_bar_exit(state, current_bar)
                    if within_bar_exit is not None:
                        raw_exit_price, exit_reason = within_bar_exit
                    else:
                        raw_exit_price = None
                        exit_reason = None

                if raw_exit_price is not None:
                    total_trading_fees += _close_position(
                        state,
                        sleeve.name,
                        current_bar["open_time"],
                        raw_exit_price,
                        exit_reason,
                        _fee_rate_for_side(backtest_config, state.side),
                        backtest_config.slippage_rate,
                        trades
                    )
                    exited_this_bar = True

            if state.side == "flat" and not exited_this_bar and signal_row is not None:
                long_entry = bool(signal_row[sleeve.entry_column])
                short_entry = (
                    sleeve.short_entry_column is not None
                    and bool(signal_row[sleeve.short_entry_column])
                )
                side = None
                if long_entry and not short_entry:
                    side = "long"
                elif short_entry and not long_entry:
                    side = "short"

                structural_stop = None
                if side == "long" and sleeve.long_stop_column is not None:
                    structural_stop = signal_row[sleeve.long_stop_column]
                elif side == "short" and sleeve.short_stop_column is not None:
                    structural_stop = signal_row[sleeve.short_stop_column]
                if structural_stop is not None and pd.isna(structural_stop):
                    structural_stop = None
                    side = None

                signal_atr = (
                    float(signal_row["atr"])
                    if structural_stop is None and "atr" in signal_row
                    else None
                )
                valid_risk_input = (
                    structural_stop is not None
                    or signal_atr is not None
                    and pd.notna(signal_atr)
                    and signal_atr > 0
                )
                if side is not None and valid_risk_input:
                    total_trading_fees += _open_position(
                        state,
                        side,
                        current_bar["open_time"],
                        float(current_bar["open"]),
                        signal_atr,
                        sleeve,
                        backtest_config,
                        structural_stop_price=structural_stop,
                        entry_bar_index=i
                    )
                    if state.side != "flat":
                        same_bar_exit = _intrabar_exit(state, current_bar)
                        if same_bar_exit is not None:
                            raw_exit_price, reason = same_bar_exit
                            total_trading_fees += _close_position(
                                state,
                                sleeve.name,
                                current_bar["open_time"],
                                raw_exit_price,
                                f"same_bar_{reason}",
                                _fee_rate_for_side(backtest_config, state.side),
                                backtest_config.slippage_rate,
                                trades
                            )

            if (
                state.side != "flat"
                and sleeve.trailing_atr_multiple is not None
                and pd.notna(current_bar["atr"])
            ):
                atr = float(current_bar["atr"])
                if state.side == "long":
                    state.highest_close = max(
                        state.highest_close, float(current_bar["close"])
                    )
                    trailing_stop = (
                        state.highest_close - atr * sleeve.trailing_atr_multiple
                    )
                    state.stop_price = max(state.stop_price, trailing_stop)
                else:
                    state.lowest_close = min(
                        state.lowest_close, float(current_bar["close"])
                    )
                    trailing_stop = (
                        state.lowest_close + atr * sleeve.trailing_atr_multiple
                    )
                    state.stop_price = min(state.stop_price, trailing_stop)

        portfolio_equity = reserve_cash + sum(
            _position_equity(state, float(current_bar["close"]))
            for state in states.values()
        )
        equity_records.append({
            "open_time": current_bar["open_time"],
            "equity": portfolio_equity
        })

    if backtest_config.liquidate_at_end:
        last_bar = df.iloc[-1]
        for sleeve in sleeves:
            state = states[sleeve.name]
            if state.side != "flat":
                total_trading_fees += _close_position(
                    state,
                    sleeve.name,
                    last_bar["open_time"],
                    float(last_bar["close"]),
                    "end_of_backtest",
                    _fee_rate_for_side(backtest_config, state.side),
                    backtest_config.slippage_rate,
                    trades
                )
        final_value = reserve_cash + sum(
            state.cash for state in states.values()
        )
        equity_records[-1]["equity"] = final_value
    else:
        final_value = equity_records[-1]["equity"]

    total_slippage_cost = sum(
        trade["slippage_cost"] for trade in trades
    ) + sum(
        state.entry_slippage
        for state in states.values()
        if state.side != "flat"
    )

    return BacktestResult(
        initial_capital=backtest_config.initial_capital,
        final_value=final_value,
        total_fees=total_trading_fees + total_funding_fees,
        total_trading_fees=total_trading_fees,
        total_funding_fees=total_funding_fees,
        equity_curve=pd.DataFrame(equity_records),
        trades=pd.DataFrame(trades),
        total_slippage_cost=total_slippage_cost
    )
