from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import BacktestResult
from futures_strategy import _funding_events_by_bar
from rotation_strategy import (
    REQUIRED_COLUMNS,
    RotationConfig,
    generate_rotation_targets
)


@dataclass(frozen=True)
class HybridConfig:
    # 现货上涨轮动使用的慢速EMA周期
    spot_slow_ema_window: int = 130

    # 现货上涨轮动使用的快速EMA周期
    spot_fast_ema_window: int = 100

    # 现货上涨轮动使用的动量周期
    spot_momentum_window: int = 120

    # 判断现货慢速EMA方向的回看周期
    spot_ema_slope_window: int = 10

    # 现货BTC和ETH重新比较的间隔
    spot_rebalance_bars: int = 12

    # 上涨阶段最多投入现货的资金比例
    spot_exposure: float = 0.33

    # 判断熊市趋势的慢速EMA周期
    short_slow_ema_window: int = 150

    # 判断熊市趋势的快速EMA周期
    short_fast_ema_window: int = 80

    # 判断熊市强度的动量周期
    short_momentum_window: int = 120

    # 判断熊市慢速EMA方向的回看周期
    short_ema_slope_window: int = 10

    # 合约做空品种重新比较的间隔
    short_rebalance_bars: int = 12

    # 是否要求全部观察品种同时进入熊市才允许做空
    require_all_symbols_short: bool = True

    # 合约空头名义价值占账户权益的比例
    short_exposure: float = 0.10

    # ATR计算周期
    atr_window: int = 14

    # 合约空头移动止损距离
    short_stop_atr_multiple: float = 10.0

    # 空头止损后等待多少根K线再允许进场
    short_cooldown_bars: int = 12

    # 研究强平价使用的保守维持保证金率
    maintenance_margin_rate: float = 0.01

    # 模拟强平时额外扣除的费用率
    liquidation_fee_rate: float = 0.01


def validate_hybrid_config(config):
    # 集中检查参数，避免错误配置放大合约风险
    positive_integer_parameters = {
        "现货慢速EMA周期": config.spot_slow_ema_window,
        "现货快速EMA周期": config.spot_fast_ema_window,
        "现货动量周期": config.spot_momentum_window,
        "现货EMA斜率周期": config.spot_ema_slope_window,
        "现货再平衡周期": config.spot_rebalance_bars,
        "空头慢速EMA周期": config.short_slow_ema_window,
        "空头快速EMA周期": config.short_fast_ema_window,
        "空头动量周期": config.short_momentum_window,
        "空头EMA斜率周期": config.short_ema_slope_window,
        "空头再平衡周期": config.short_rebalance_bars,
        "ATR周期": config.atr_window
    }

    for name, value in positive_integer_parameters.items():
        if value <= 0:
            raise ValueError(f"{name}必须大于0")

    if config.short_cooldown_bars < 0:
        raise ValueError("空头冷静期不能小于0")

    if config.spot_fast_ema_window >= config.spot_slow_ema_window:
        raise ValueError("现货快速EMA周期必须小于慢速EMA周期")

    if config.short_fast_ema_window >= config.short_slow_ema_window:
        raise ValueError("空头快速EMA周期必须小于慢速EMA周期")

    if not 0 < config.spot_exposure <= 1:
        raise ValueError("现货仓位比例必须大于0且不超过100%")

    if not 0 < config.short_exposure <= 1.5:
        raise ValueError("研究版合约空头名义仓位必须大于0且不超过1.5倍")

    if config.short_stop_atr_multiple <= 0:
        raise ValueError("空头止损ATR倍数必须大于0")

    if not 0 <= config.maintenance_margin_rate < 0.10:
        raise ValueError("维持保证金率必须位于0到10%之间")

    if not 0 <= config.liquidation_fee_rate < 0.10:
        raise ValueError("模拟强平费率必须位于0到10%之间")


def load_hybrid_config(config_file):
    # 从JSON文件读取现货加合约对冲参数
    with Path(config_file).open("r", encoding="utf-8") as file:
        config_values = json.load(file)

    config = HybridConfig(**config_values)
    validate_hybrid_config(config)
    return config


def _add_short_indicators(original_df, config):
    # 只使用当前和过去的K线识别明确熊市
    df = original_df.copy()
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    df["short_slow_ema"] = df["close"].ewm(
        span=config.short_slow_ema_window,
        adjust=False,
        min_periods=config.short_slow_ema_window
    ).mean()
    df["short_fast_ema"] = df["close"].ewm(
        span=config.short_fast_ema_window,
        adjust=False,
        min_periods=config.short_fast_ema_window
    ).mean()
    df["short_momentum"] = df["close"].pct_change(
        periods=config.short_momentum_window
    )
    df["atr"] = true_range.rolling(
        window=config.atr_window
    ).mean()

    slow_ema_change = (
        df["short_slow_ema"]
        -
        df["short_slow_ema"].shift(
            config.short_ema_slope_window
        )
    )

    df["short_eligible"] = (
        (df["close"] < df["short_slow_ema"])
        &
        (df["close"] < df["short_fast_ema"])
        &
        (df["short_fast_ema"] < df["short_slow_ema"])
        &
        (slow_ema_change < 0)
        &
        (df["short_momentum"] < 0)
    ).fillna(False)

    # 同样的负动量下，波动较小的下跌趋势更稳定
    recent_volatility = df["close"].pct_change().rolling(
        window=max(20, config.atr_window)
    ).std(ddof=0)
    df["short_score"] = (
        df["short_momentum"].abs()
        /
        recent_volatility.replace(0, np.nan)
    )

    return df


def generate_hybrid_targets(original_frames, config=None):
    # 上涨阶段选择现货，只有没有现货目标时才考虑合约做空
    hybrid_config = config or HybridConfig()
    validate_hybrid_config(hybrid_config)

    spot_config = RotationConfig(
        slow_ema_window=hybrid_config.spot_slow_ema_window,
        fast_ema_window=hybrid_config.spot_fast_ema_window,
        momentum_window=hybrid_config.spot_momentum_window,
        ema_slope_window=hybrid_config.spot_ema_slope_window,
        rebalance_bars=hybrid_config.spot_rebalance_bars,
        exposure=hybrid_config.spot_exposure
    )
    frames, spot_targets = generate_rotation_targets(
        original_frames,
        config=spot_config
    )
    symbols = list(frames)

    for symbol, df in frames.items():
        frames[symbol] = _add_short_indicators(
            df,
            hybrid_config
        )

    current_short_symbol = None
    target_rows = []

    for i in range(len(frames[symbols[0]])):
        spot_symbol = spot_targets.iloc[i]

        if pd.isna(spot_symbol):
            spot_symbol = None

        short_candidates = [
            symbol
            for symbol in symbols
            if bool(frames[symbol].iloc[i]["short_eligible"])
        ]

        if (
            hybrid_config.require_all_symbols_short
            and len(short_candidates) != len(symbols)
        ):
            short_candidates = []

        if current_short_symbol not in short_candidates:
            current_short_symbol = None

        if i % hybrid_config.short_rebalance_bars == 0:
            if short_candidates:
                current_short_symbol = max(
                    short_candidates,
                    key=lambda symbol: float(
                        frames[symbol].iloc[i]["short_score"]
                    )
                )
            else:
                current_short_symbol = None

        if spot_symbol is not None:
            target_type = "spot_long"
            target_symbol = spot_symbol
            target_exposure = hybrid_config.spot_exposure
        elif current_short_symbol is not None:
            target_type = "futures_short"
            target_symbol = current_short_symbol
            target_exposure = hybrid_config.short_exposure
        else:
            target_type = "cash"
            target_symbol = None
            target_exposure = 0.0

        target_rows.append(
            {
                "open_time": frames[symbols[0]].iloc[i]["open_time"],
                "target_type": target_type,
                "target_symbol": target_symbol,
                "target_exposure": target_exposure
            }
        )

    return frames, pd.DataFrame(target_rows)


def _empty_state(initial_capital):
    return {
        "balance": float(initial_capital),
        "position_type": "cash",
        "held_symbol": None,
        "quantity": 0.0,
        "signed_quantity": 0.0,
        "entry_price": 0.0,
        "entry_time": None,
        "entry_fee": 0.0,
        "entry_total_cost": 0.0,
        "entry_account_value": 0.0,
        "entry_exposure": 0.0,
        "position_funding_cash_flow": 0.0,
        "lowest_close": 0.0,
        "stop_price": 0.0,
        "liquidation_price": float("inf")
    }


def _reset_position(state):
    state.update(
        {
            "position_type": "cash",
            "held_symbol": None,
            "quantity": 0.0,
            "signed_quantity": 0.0,
            "entry_price": 0.0,
            "entry_time": None,
            "entry_fee": 0.0,
            "entry_total_cost": 0.0,
            "entry_account_value": 0.0,
            "entry_exposure": 0.0,
            "position_funding_cash_flow": 0.0,
            "lowest_close": 0.0,
            "stop_price": 0.0,
            "liquidation_price": float("inf")
        }
    )


def _open_spot_long(
    state,
    symbol,
    raw_price,
    entry_time,
    exposure,
    fee_rate,
    slippage_rate
):
    # 上涨阶段使用现货，避免长期支付永续合约资金费率
    entry_price = raw_price * (1 + slippage_rate)
    entry_account_value = state["balance"]
    cash_budget = entry_account_value * exposure
    quantity = cash_budget / (entry_price * (1 + fee_rate))
    gross_cost = quantity * entry_price
    entry_fee = gross_cost * fee_rate
    total_cost = gross_cost + entry_fee

    state["balance"] -= total_cost
    state["position_type"] = "spot_long"
    state["held_symbol"] = symbol
    state["quantity"] = quantity
    state["entry_price"] = entry_price
    state["entry_time"] = entry_time
    state["entry_fee"] = entry_fee
    state["entry_total_cost"] = total_cost
    state["entry_account_value"] = entry_account_value
    state["entry_exposure"] = exposure

    return entry_fee


def _close_spot_long(
    state,
    raw_price,
    exit_time,
    exit_reason,
    fee_rate,
    slippage_rate,
    trades
):
    exit_price = raw_price * (1 - slippage_rate)
    gross_value = state["quantity"] * exit_price
    exit_fee = gross_value * fee_rate
    net_value = gross_value - exit_fee
    state["balance"] += net_value
    gross_profit = (
        (exit_price - state["entry_price"])
        *
        state["quantity"]
    )
    net_profit = net_value - state["entry_total_cost"]

    trades.append(
        {
            "strategy": "hybrid_spot_futures",
            "instrument": "spot",
            "symbol": state["held_symbol"],
            "direction": "long",
            "entry_time": state["entry_time"],
            "exit_time": exit_time,
            "entry_price": state["entry_price"],
            "exit_price": exit_price,
            "quantity": state["quantity"],
            "entry_exposure": state["entry_exposure"],
            "entry_fee": state["entry_fee"],
            "exit_fee": exit_fee,
            "liquidation_fee": 0.0,
            "funding_cash_flow": 0.0,
            "entry_account_value": state["entry_account_value"],
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "return_rate": (
                net_profit / state["entry_account_value"]
                if state["entry_account_value"] > 0
                else 0.0
            ),
            "exit_reason": exit_reason
        }
    )
    _reset_position(state)
    return exit_fee


def _open_futures_short(
    state,
    symbol,
    raw_price,
    entry_time,
    exposure,
    atr,
    config,
    fee_rate,
    slippage_rate
):
    # 只在明确熊市使用低名义仓位永续合约做空
    entry_price = raw_price * (1 - slippage_rate)
    entry_account_value = state["balance"]
    notional = entry_account_value * exposure
    quantity = notional / entry_price
    entry_fee = notional * fee_rate

    state["balance"] -= entry_fee
    state["position_type"] = "futures_short"
    state["held_symbol"] = symbol
    state["signed_quantity"] = -quantity
    state["entry_price"] = entry_price
    state["entry_time"] = entry_time
    state["entry_fee"] = entry_fee
    state["entry_account_value"] = entry_account_value
    state["entry_exposure"] = exposure
    state["position_funding_cash_flow"] = 0.0
    state["lowest_close"] = entry_price
    state["stop_price"] = (
        entry_price
        +
        atr * config.short_stop_atr_multiple
    )

    if exposure > 1:
        state["liquidation_price"] = entry_price * (
            1
            +
            1 / exposure
            -
            config.maintenance_margin_rate
        )

    return entry_fee


def _close_futures_short(
    state,
    raw_price,
    exit_time,
    exit_reason,
    fee_rate,
    slippage_rate,
    liquidation_fee_rate,
    trades
):
    exit_price = raw_price * (1 + slippage_rate)
    quantity = abs(state["signed_quantity"])
    exit_notional = quantity * exit_price
    exit_fee = exit_notional * fee_rate
    liquidation_fee = (
        exit_notional * liquidation_fee_rate
        if exit_reason == "liquidation"
        else 0.0
    )
    gross_profit = (
        state["signed_quantity"]
        *
        (exit_price - state["entry_price"])
    )
    state["balance"] += (
        gross_profit - exit_fee - liquidation_fee
    )
    net_profit = (
        gross_profit
        -
        state["entry_fee"]
        -
        exit_fee
        -
        liquidation_fee
        +
        state["position_funding_cash_flow"]
    )

    trades.append(
        {
            "strategy": "hybrid_spot_futures",
            "instrument": "futures",
            "symbol": state["held_symbol"],
            "direction": "short",
            "entry_time": state["entry_time"],
            "exit_time": exit_time,
            "entry_price": state["entry_price"],
            "exit_price": exit_price,
            "quantity": quantity,
            "entry_exposure": state["entry_exposure"],
            "entry_fee": state["entry_fee"],
            "exit_fee": exit_fee,
            "liquidation_fee": liquidation_fee,
            "funding_cash_flow": state["position_funding_cash_flow"],
            "entry_account_value": state["entry_account_value"],
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "return_rate": (
                net_profit / state["entry_account_value"]
                if state["entry_account_value"] > 0
                else 0.0
            ),
            "exit_reason": exit_reason
        }
    )
    _reset_position(state)
    return exit_fee + liquidation_fee


def _apply_short_funding(state, events, fallback_price):
    # 正资金费率时空头收取，负资金费率时空头支付
    cash_flow = 0.0

    for event in events:
        mark_price = float(event["mark_price"])

        if mark_price <= 0:
            mark_price = fallback_price

        event_cash_flow = (
            -state["signed_quantity"]
            *
            mark_price
            *
            float(event["funding_rate"])
        )
        state["balance"] += event_cash_flow
        state["position_funding_cash_flow"] += event_cash_flow
        cash_flow += event_cash_flow

    return cash_flow


def _current_equity(state, raw_price):
    if state["position_type"] == "spot_long":
        return state["balance"] + state["quantity"] * raw_price

    if state["position_type"] == "futures_short":
        return (
            state["balance"]
            +
            state["signed_quantity"]
            *
            (raw_price - state["entry_price"])
        )

    return state["balance"]


def run_hybrid_backtest(
    original_frames,
    funding_frames,
    config=None,
    initial_capital=5000.0,
    spot_fee_rate=0.001,
    futures_fee_rate=0.0005,
    slippage_rate=0.0005,
    execution_delay_bars=1,
    liquidate_at_end=True
):
    # 回测上涨持现货、明确熊市小仓位做空的混合策略
    hybrid_config = config or HybridConfig()
    validate_hybrid_config(hybrid_config)

    if initial_capital <= 0:
        raise ValueError("初始资金必须大于0")

    for name, rate in {
        "现货手续费率": spot_fee_rate,
        "合约手续费率": futures_fee_rate,
        "滑点率": slippage_rate
    }.items():
        if not 0 <= rate < 1:
            raise ValueError(f"{name}必须位于0到1之间")

    if execution_delay_bars < 1:
        raise ValueError("成交延迟至少为1根K线")

    frames, targets = generate_hybrid_targets(
        original_frames,
        hybrid_config
    )
    symbols = list(frames)
    times = frames[symbols[0]]["open_time"]
    funding_events = _funding_events_by_bar(
        times,
        funding_frames
    )
    state = _empty_state(initial_capital)
    trades = []
    total_fees = 0.0
    cumulative_funding_cash_flow = 0.0
    short_cooldown_until = 0
    equity_records = [
        {
            "open_time": times.iloc[0],
            "equity": float(initial_capital),
            "balance": float(initial_capital),
            "position_type": "cash",
            "held_symbol": None,
            "cumulative_funding_cash_flow": 0.0
        }
    ]

    for i in range(1, len(times)):
        current_time = times.iloc[i]

        if state["position_type"] == "futures_short":
            held_symbol = state["held_symbol"]
            events = funding_events.get(held_symbol, {}).get(i, [])
            cumulative_funding_cash_flow += _apply_short_funding(
                state,
                events,
                float(frames[held_symbol].iloc[i]["open"])
            )

        signal_position = i - execution_delay_bars

        if signal_position >= 0:
            target_row = targets.iloc[signal_position]
            desired_type = str(target_row["target_type"])
            desired_symbol = target_row["target_symbol"]
            desired_exposure = float(target_row["target_exposure"])

            if pd.isna(desired_symbol):
                desired_symbol = None
        else:
            desired_type = "cash"
            desired_symbol = None
            desired_exposure = 0.0

        current_target = (
            state["position_type"],
            state["held_symbol"]
        )
        desired_target = (desired_type, desired_symbol)

        if (
            state["position_type"] != "cash"
            and
            desired_target != current_target
        ):
            held_symbol = state["held_symbol"]
            raw_exit_price = float(
                frames[held_symbol].iloc[i]["open"]
            )

            if state["position_type"] == "spot_long":
                total_fees += _close_spot_long(
                    state,
                    raw_exit_price,
                    current_time,
                    "target_changed",
                    spot_fee_rate,
                    slippage_rate,
                    trades
                )
            else:
                total_fees += _close_futures_short(
                    state,
                    raw_exit_price,
                    current_time,
                    "target_changed",
                    futures_fee_rate,
                    slippage_rate,
                    hybrid_config.liquidation_fee_rate,
                    trades
                )

        if state["position_type"] == "cash":
            if desired_type == "spot_long" and desired_symbol is not None:
                total_fees += _open_spot_long(
                    state,
                    desired_symbol,
                    float(frames[desired_symbol].iloc[i]["open"]),
                    current_time,
                    desired_exposure,
                    spot_fee_rate,
                    slippage_rate
                )
            elif (
                desired_type == "futures_short"
                and
                desired_symbol is not None
                and
                i >= short_cooldown_until
            ):
                signal_atr = float(
                    frames[desired_symbol].iloc[signal_position]["atr"]
                )
                total_fees += _open_futures_short(
                    state,
                    desired_symbol,
                    float(frames[desired_symbol].iloc[i]["open"]),
                    current_time,
                    desired_exposure,
                    signal_atr,
                    hybrid_config,
                    futures_fee_rate,
                    slippage_rate
                )

        if state["position_type"] == "futures_short":
            held_symbol = state["held_symbol"]
            row = frames[held_symbol].iloc[i]
            liquidation_hit = (
                float(row["open"]) >= state["liquidation_price"]
            )

            if liquidation_hit:
                total_fees += _close_futures_short(
                    state,
                    float(row["open"]),
                    current_time,
                    "liquidation",
                    futures_fee_rate,
                    slippage_rate,
                    hybrid_config.liquidation_fee_rate,
                    trades
                )
                short_cooldown_until = (
                    i + hybrid_config.short_cooldown_bars
                )
            elif float(row["high"]) >= state["stop_price"]:
                raw_stop_price = max(
                    float(row["open"]),
                    state["stop_price"]
                )
                total_fees += _close_futures_short(
                    state,
                    raw_stop_price,
                    current_time,
                    "trailing_stop",
                    futures_fee_rate,
                    slippage_rate,
                    hybrid_config.liquidation_fee_rate,
                    trades
                )
                short_cooldown_until = (
                    i + hybrid_config.short_cooldown_bars
                )

        if state["position_type"] == "futures_short":
            held_symbol = state["held_symbol"]
            row = frames[held_symbol].iloc[i]
            close_price = float(row["close"])
            atr = float(row["atr"])
            state["lowest_close"] = min(
                state["lowest_close"],
                close_price
            )
            state["stop_price"] = min(
                state["stop_price"],
                state["lowest_close"]
                +
                atr * hybrid_config.short_stop_atr_multiple
            )

        if state["position_type"] == "cash":
            equity = state["balance"]
        else:
            held_symbol = state["held_symbol"]
            equity = _current_equity(
                state,
                float(frames[held_symbol].iloc[i]["close"])
            )

        equity_records.append(
            {
                "open_time": current_time,
                "equity": equity,
                "balance": state["balance"],
                "position_type": state["position_type"],
                "held_symbol": state["held_symbol"],
                "cumulative_funding_cash_flow": (
                    cumulative_funding_cash_flow
                )
            }
        )

    if liquidate_at_end and state["position_type"] != "cash":
        held_symbol = state["held_symbol"]
        raw_exit_price = float(frames[held_symbol].iloc[-1]["close"])

        if state["position_type"] == "spot_long":
            total_fees += _close_spot_long(
                state,
                raw_exit_price,
                times.iloc[-1],
                "end_of_backtest",
                spot_fee_rate,
                slippage_rate,
                trades
            )
        else:
            total_fees += _close_futures_short(
                state,
                raw_exit_price,
                times.iloc[-1],
                "end_of_backtest",
                futures_fee_rate,
                slippage_rate,
                hybrid_config.liquidation_fee_rate,
                trades
            )

        equity_records[-1]["equity"] = state["balance"]
        equity_records[-1]["balance"] = state["balance"]
        equity_records[-1]["position_type"] = "cash"
        equity_records[-1]["held_symbol"] = None

    if state["position_type"] == "cash":
        final_value = state["balance"]
    else:
        held_symbol = state["held_symbol"]
        final_value = _current_equity(
            state,
            float(frames[held_symbol].iloc[-1]["close"])
        )

    trade_columns = [
        "strategy",
        "instrument",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "entry_exposure",
        "entry_fee",
        "exit_fee",
        "liquidation_fee",
        "funding_cash_flow",
        "entry_account_value",
        "gross_profit",
        "net_profit",
        "return_rate",
        "exit_reason"
    ]

    total_funding_fees = -cumulative_funding_cash_flow
    total_trading_fees = total_fees
    return BacktestResult(
        initial_capital=float(initial_capital),
        final_value=float(final_value),
        total_fees=float(total_trading_fees + total_funding_fees),
        total_trading_fees=float(total_trading_fees),
        total_funding_fees=float(total_funding_fees),
        equity_curve=pd.DataFrame(equity_records),
        trades=pd.DataFrame(trades, columns=trade_columns)
    )
