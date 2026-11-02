from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from engine import BacktestResult
from rotation_strategy import align_symbol_frames


@dataclass(frozen=True)
class FuturesRotationConfig:
    # 判断长期趋势的慢速EMA周期
    slow_ema_window: int = 130

    # 判断短期趋势的快速EMA周期
    fast_ema_window: int = 80

    # 比较不同品种趋势强弱的动量周期
    momentum_window: int = 120

    # 判断慢速EMA方向的回看周期
    ema_slope_window: int = 10

    # 每隔多少根K线重新比较一次交易品种
    rebalance_bars: int = 12

    # ATR计算周期
    atr_window: int = 14

    # 移动止损距离
    trailing_stop_atr_multiple: float = 4.0

    # 计算实际波动率使用的K线数量
    volatility_window: int = 42

    # 策略希望承担的年化波动率
    annual_volatility_target: float = 0.22

    # 名义仓位相对账户权益的最高倍数
    max_leverage: float = 1.5

    # 做空仓位相对于做多仓位的缩放比例
    short_exposure_multiplier: float = 0.80

    # 研究强平价时采用的保守维持保证金率
    maintenance_margin_rate: float = 0.01

    # 模拟强平时额外扣除的费用率
    liquidation_fee_rate: float = 0.01


def validate_futures_config(config):
    # 合约参数错误可能严重放大风险，因此集中检查
    positive_integer_parameters = {
        "慢速EMA周期": config.slow_ema_window,
        "快速EMA周期": config.fast_ema_window,
        "动量周期": config.momentum_window,
        "EMA斜率周期": config.ema_slope_window,
        "再平衡周期": config.rebalance_bars,
        "ATR周期": config.atr_window,
        "波动率周期": config.volatility_window
    }

    for name, value in positive_integer_parameters.items():
        if value <= 0:
            raise ValueError(f"{name}必须大于0")

    if config.fast_ema_window >= config.slow_ema_window:
        raise ValueError("快速EMA周期必须小于慢速EMA周期")

    if config.trailing_stop_atr_multiple <= 0:
        raise ValueError("移动止损ATR倍数必须大于0")

    if not 0 < config.annual_volatility_target <= 1:
        raise ValueError("目标年化波动率必须大于0且不超过100%")

    if not 0 < config.max_leverage <= 2:
        raise ValueError("研究版最高杠杆必须大于0且不超过2倍")

    if not 0 < config.short_exposure_multiplier <= 1:
        raise ValueError("做空仓位缩放比例必须大于0且不超过1")

    if not 0 <= config.maintenance_margin_rate < 0.10:
        raise ValueError("维持保证金率必须位于0到10%之间")

    if not 0 <= config.liquidation_fee_rate < 0.10:
        raise ValueError("模拟强平费率必须位于0到10%之间")


def load_futures_config(config_file):
    # 从JSON文件读取合约研究参数
    with Path(config_file).open("r", encoding="utf-8") as file:
        config_values = json.load(file)

    config = FuturesRotationConfig(**config_values)
    validate_futures_config(config)
    return config


def _add_futures_indicators(original_df, config):
    # 只使用当前及过去数据计算趋势、动量、ATR和波动率
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

    df["slow_ema"] = df["close"].ewm(
        span=config.slow_ema_window,
        adjust=False,
        min_periods=config.slow_ema_window
    ).mean()
    df["fast_ema"] = df["close"].ewm(
        span=config.fast_ema_window,
        adjust=False,
        min_periods=config.fast_ema_window
    ).mean()
    df["momentum"] = df["close"].pct_change(
        periods=config.momentum_window
    )
    df["atr"] = true_range.rolling(
        window=config.atr_window
    ).mean()

    # 4小时K线一年约有6乘365.25根
    periods_per_year = 6 * 365.25
    df["annualized_volatility"] = (
        df["close"].pct_change().rolling(
            window=config.volatility_window
        ).std(ddof=0)
        *
        np.sqrt(periods_per_year)
    )

    slow_ema_change = (
        df["slow_ema"]
        -
        df["slow_ema"].shift(config.ema_slope_window)
    )

    df["long_eligible"] = (
        (df["close"] > df["slow_ema"])
        &
        (df["close"] > df["fast_ema"])
        &
        (df["fast_ema"] > df["slow_ema"])
        &
        (slow_ema_change > 0)
        &
        (df["momentum"] > 0)
    ).fillna(False)

    df["short_eligible"] = (
        (df["close"] < df["slow_ema"])
        &
        (df["close"] < df["fast_ema"])
        &
        (df["fast_ema"] < df["slow_ema"])
        &
        (slow_ema_change < 0)
        &
        (df["momentum"] < 0)
    ).fillna(False)

    df["trend_score"] = (
        df["momentum"].abs()
        /
        df["annualized_volatility"].replace(0, np.nan)
    )

    return df


def generate_futures_targets(original_frames, config=None):
    # 在BTC、ETH多头、空头和现金之间选择目标
    futures_config = config or FuturesRotationConfig()
    validate_futures_config(futures_config)
    frames = align_symbol_frames(original_frames)
    symbols = list(frames)

    minimum_bars = max(
        futures_config.slow_ema_window,
        futures_config.momentum_window,
        futures_config.volatility_window
    ) + futures_config.ema_slope_window + 2

    if len(frames[symbols[0]]) < minimum_bars:
        raise ValueError(
            f"合约策略至少需要{minimum_bars}根共同K线"
        )

    for symbol, df in frames.items():
        frames[symbol] = _add_futures_indicators(
            df,
            futures_config
        )

    current_target = None
    target_rows = []

    for i in range(len(frames[symbols[0]])):
        candidates = []

        for symbol in symbols:
            row = frames[symbol].iloc[i]

            if bool(row["long_eligible"]):
                candidates.append(
                    (symbol, 1, float(row["trend_score"]))
                )

            if bool(row["short_eligible"]):
                candidates.append(
                    (symbol, -1, float(row["trend_score"]))
                )

        eligible_targets = {
            (symbol, direction)
            for symbol, direction, _ in candidates
        }

        # 当前方向失效时立即回到现金
        if current_target not in eligible_targets:
            current_target = None

        # 定期选择单位波动率趋势最强的品种和方向
        if i % futures_config.rebalance_bars == 0:
            if candidates:
                selected = max(candidates, key=lambda item: item[2])
                current_target = (selected[0], selected[1])
            else:
                current_target = None

        if current_target is None:
            target_symbol = None
            target_direction = 0
            target_leverage = 0.0
        else:
            target_symbol, target_direction = current_target
            row = frames[target_symbol].iloc[i]
            realized_volatility = float(
                row["annualized_volatility"]
            )

            if realized_volatility > 0:
                target_leverage = min(
                    futures_config.max_leverage,
                    futures_config.annual_volatility_target
                    /
                    realized_volatility
                )
            else:
                target_leverage = 0.0

            if target_direction < 0:
                target_leverage *= (
                    futures_config.short_exposure_multiplier
                )

        target_rows.append(
            {
                "open_time": frames[symbols[0]].iloc[i]["open_time"],
                "target_symbol": target_symbol,
                "target_direction": target_direction,
                "target_leverage": target_leverage
            }
        )

    return frames, pd.DataFrame(target_rows)


def _funding_events_by_bar(times, funding_frames):
    # 把每次资金费率结算映射到第一根不早于结算时间的K线
    events = {}
    time_index = pd.DatetimeIndex(times)

    for original_symbol, original_funding_df in funding_frames.items():
        symbol = original_symbol.upper()
        funding_df = original_funding_df.copy()

        if funding_df.empty:
            events[symbol] = {}
            continue

        funding_df["funding_time"] = pd.to_datetime(
            funding_df["funding_time"],
            utc=True
        )
        symbol_events = {}

        for row in funding_df.itertuples(index=False):
            bar_position = int(
                time_index.searchsorted(row.funding_time, side="left")
            )

            if 0 < bar_position < len(time_index):
                symbol_events.setdefault(bar_position, []).append(
                    {
                        "funding_time": row.funding_time,
                        "funding_rate": float(row.funding_rate),
                        "mark_price": float(row.mark_price)
                    }
                )

        events[symbol] = symbol_events

    return events


def _current_equity(state, raw_price):
    if state["held_symbol"] is None:
        return state["balance"]

    unrealized_profit = (
        state["signed_quantity"]
        *
        (raw_price - state["entry_price"])
    )
    return state["balance"] + unrealized_profit


def _calculate_liquidation_price(
    entry_price,
    direction,
    leverage,
    maintenance_margin_rate
):
    # 这是研究用近似值，真实强平价取决于账户保证金档位
    if leverage <= 1:
        return 0.0 if direction > 0 else float("inf")

    if direction > 0:
        return max(
            0.0,
            entry_price
            *
            (1 - 1 / leverage + maintenance_margin_rate)
        )

    return entry_price * (
        1 + 1 / leverage - maintenance_margin_rate
    )


def _open_futures_position(
    state,
    symbol,
    direction,
    target_leverage,
    raw_price,
    entry_time,
    atr,
    config,
    fee_rate,
    slippage_rate
):
    # 按账户权益和目标波动率确定名义仓位
    entry_price = raw_price * (
        1 + slippage_rate
        if direction > 0
        else 1 - slippage_rate
    )
    entry_account_value = state["balance"]
    notional = max(0.0, entry_account_value * target_leverage)
    quantity = notional / entry_price if entry_price > 0 else 0.0
    entry_fee = notional * fee_rate

    state["balance"] -= entry_fee
    state["signed_quantity"] = direction * quantity
    state["held_symbol"] = symbol
    state["direction"] = direction
    state["entry_price"] = entry_price
    state["entry_time"] = entry_time
    state["entry_fee"] = entry_fee
    state["entry_account_value"] = entry_account_value
    state["entry_leverage"] = target_leverage
    state["position_funding_cash_flow"] = 0.0
    state["highest_close"] = entry_price
    state["lowest_close"] = entry_price
    state["liquidation_price"] = _calculate_liquidation_price(
        entry_price=entry_price,
        direction=direction,
        leverage=target_leverage,
        maintenance_margin_rate=config.maintenance_margin_rate
    )

    stop_distance = atr * config.trailing_stop_atr_multiple

    if direction > 0:
        state["stop_price"] = entry_price - stop_distance
    else:
        state["stop_price"] = entry_price + stop_distance

    return entry_fee


def _close_futures_position(
    state,
    raw_price,
    exit_time,
    exit_reason,
    fee_rate,
    slippage_rate,
    liquidation_fee_rate,
    trades
):
    # 按持仓方向加入不利滑点并结算盈亏
    direction = state["direction"]
    exit_price = raw_price * (
        1 - slippage_rate
        if direction > 0
        else 1 + slippage_rate
    )
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

    trade_return = (
        net_profit / state["entry_account_value"]
        if state["entry_account_value"] > 0
        else 0.0
    )

    trades.append(
        {
            "strategy": "futures_trend_rotation",
            "symbol": state["held_symbol"],
            "direction": "long" if direction > 0 else "short",
            "entry_time": state["entry_time"],
            "exit_time": exit_time,
            "entry_price": state["entry_price"],
            "exit_price": exit_price,
            "quantity": quantity,
            "entry_leverage": state["entry_leverage"],
            "entry_fee": state["entry_fee"],
            "exit_fee": exit_fee,
            "liquidation_fee": liquidation_fee,
            "funding_cash_flow": (
                state["position_funding_cash_flow"]
            ),
            "entry_account_value": state["entry_account_value"],
            "gross_profit": gross_profit,
            "net_profit": net_profit,
            "return_rate": trade_return,
            "exit_reason": exit_reason
        }
    )

    total_cost = exit_fee + liquidation_fee
    state.update(
        {
            "signed_quantity": 0.0,
            "held_symbol": None,
            "direction": 0,
            "entry_price": 0.0,
            "entry_time": None,
            "entry_fee": 0.0,
            "entry_account_value": 0.0,
            "entry_leverage": 0.0,
            "position_funding_cash_flow": 0.0,
            "highest_close": 0.0,
            "lowest_close": 0.0,
            "stop_price": 0.0,
            "liquidation_price": 0.0
        }
    )

    return total_cost


def _apply_funding(state, funding_events, fallback_price):
    # 正资金费率时多头支付、空头收取；负费率方向相反
    cash_flow = 0.0

    for event in funding_events:
        mark_price = event["mark_price"]

        if mark_price <= 0:
            mark_price = fallback_price

        event_cash_flow = (
            -state["signed_quantity"]
            *
            mark_price
            *
            event["funding_rate"]
        )
        state["balance"] += event_cash_flow
        state["position_funding_cash_flow"] += event_cash_flow
        cash_flow += event_cash_flow

    return cash_flow


def run_futures_rotation_backtest(
    original_frames,
    funding_frames,
    config=None,
    initial_capital=5000.0,
    fee_rate=0.0005,
    slippage_rate=0.0005,
    execution_delay_bars=1,
    liquidate_at_end=True
):
    # 回测低倍U本位永续合约多空轮动策略
    futures_config = config or FuturesRotationConfig()
    validate_futures_config(futures_config)

    if initial_capital <= 0:
        raise ValueError("初始资金必须大于0")

    if not 0 <= fee_rate < 1:
        raise ValueError("手续费率必须位于0到1之间")

    if not 0 <= slippage_rate < 1:
        raise ValueError("滑点率必须位于0到1之间")

    if execution_delay_bars < 1:
        raise ValueError("成交延迟至少为1根K线")

    frames, targets = generate_futures_targets(
        original_frames,
        futures_config
    )
    symbols = list(frames)
    times = frames[symbols[0]]["open_time"]
    funding_events = _funding_events_by_bar(
        times,
        funding_frames
    )

    state = {
        "balance": float(initial_capital),
        "signed_quantity": 0.0,
        "held_symbol": None,
        "direction": 0,
        "entry_price": 0.0,
        "entry_time": None,
        "entry_fee": 0.0,
        "entry_account_value": 0.0,
        "entry_leverage": 0.0,
        "position_funding_cash_flow": 0.0,
        "highest_close": 0.0,
        "lowest_close": 0.0,
        "stop_price": 0.0,
        "liquidation_price": 0.0
    }

    trades = []
    total_fees = 0.0
    cumulative_funding_cash_flow = 0.0
    equity_records = [
        {
            "open_time": times.iloc[0],
            "equity": float(initial_capital),
            "balance": float(initial_capital),
            "held_symbol": None,
            "direction": 0,
            "effective_leverage": 0.0,
            "cumulative_funding_cash_flow": 0.0
        }
    ]

    for i in range(1, len(times)):
        current_time = times.iloc[i]

        # 先结算已经持有仓位在当前时点发生的资金费率
        if state["held_symbol"] is not None:
            held_symbol = state["held_symbol"]
            events = funding_events.get(held_symbol, {}).get(i, [])
            cumulative_funding_cash_flow += _apply_funding(
                state=state,
                funding_events=events,
                fallback_price=float(
                    frames[held_symbol].iloc[i]["open"]
                )
            )

        signal_position = i - execution_delay_bars

        if signal_position >= 0:
            target_row = targets.iloc[signal_position]
            desired_symbol = target_row["target_symbol"]
            desired_direction = int(target_row["target_direction"])
            desired_leverage = float(target_row["target_leverage"])

            # DataFrame会把None显示为NaN，这里统一还原为空目标
            if pd.isna(desired_symbol):
                desired_symbol = None
        else:
            desired_symbol = None
            desired_direction = 0
            desired_leverage = 0.0

        current_target = (
            state["held_symbol"],
            state["direction"]
        )
        desired_target = (desired_symbol, desired_direction)

        # 目标变化时在当前K线开盘平掉旧仓位
        if (
            state["held_symbol"] is not None
            and
            desired_target != current_target
        ):
            total_fees += _close_futures_position(
                state=state,
                raw_price=float(
                    frames[state["held_symbol"]].iloc[i]["open"]
                ),
                exit_time=current_time,
                exit_reason="target_changed",
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                liquidation_fee_rate=(
                    futures_config.liquidation_fee_rate
                ),
                trades=trades
            )

        # 平仓后可以在同一根K线开盘切换新方向
        if (
            state["held_symbol"] is None
            and
            desired_symbol is not None
            and
            desired_direction != 0
            and
            desired_leverage > 0
        ):
            signal_atr = float(
                frames[desired_symbol]
                .iloc[signal_position]["atr"]
            )
            total_fees += _open_futures_position(
                state=state,
                symbol=desired_symbol,
                direction=desired_direction,
                target_leverage=desired_leverage,
                raw_price=float(
                    frames[desired_symbol].iloc[i]["open"]
                ),
                entry_time=current_time,
                atr=signal_atr,
                config=futures_config,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate
            )

        # 检查强平保护和移动止损
        if state["held_symbol"] is not None:
            held_frame = frames[state["held_symbol"]]
            row = held_frame.iloc[i]
            direction = state["direction"]
            liquidation_hit = (
                direction > 0
                and
                float(row["open"]) <= state["liquidation_price"]
            ) or (
                direction < 0
                and
                float(row["open"]) >= state["liquidation_price"]
            )

            if liquidation_hit:
                total_fees += _close_futures_position(
                    state=state,
                    raw_price=float(row["open"]),
                    exit_time=current_time,
                    exit_reason="liquidation",
                    fee_rate=fee_rate,
                    slippage_rate=slippage_rate,
                    liquidation_fee_rate=(
                        futures_config.liquidation_fee_rate
                    ),
                    trades=trades
                )
            else:
                stop_hit = (
                    direction > 0
                    and
                    float(row["low"]) <= state["stop_price"]
                ) or (
                    direction < 0
                    and
                    float(row["high"]) >= state["stop_price"]
                )

                if stop_hit:
                    if direction > 0:
                        raw_stop_price = min(
                            float(row["open"]),
                            state["stop_price"]
                        )
                    else:
                        raw_stop_price = max(
                            float(row["open"]),
                            state["stop_price"]
                        )

                    total_fees += _close_futures_position(
                        state=state,
                        raw_price=raw_stop_price,
                        exit_time=current_time,
                        exit_reason="trailing_stop",
                        fee_rate=fee_rate,
                        slippage_rate=slippage_rate,
                        liquidation_fee_rate=(
                            futures_config.liquidation_fee_rate
                        ),
                        trades=trades
                    )

        # 收盘后更新下一根K线使用的移动止损
        if state["held_symbol"] is not None:
            held_frame = frames[state["held_symbol"]]
            row = held_frame.iloc[i]
            close_price = float(row["close"])
            atr = float(row["atr"])
            stop_distance = (
                atr * futures_config.trailing_stop_atr_multiple
            )

            if state["direction"] > 0:
                state["highest_close"] = max(
                    state["highest_close"],
                    close_price
                )
                state["stop_price"] = max(
                    state["stop_price"],
                    state["highest_close"] - stop_distance
                )
            else:
                state["lowest_close"] = min(
                    state["lowest_close"],
                    close_price
                )
                state["stop_price"] = min(
                    state["stop_price"],
                    state["lowest_close"] + stop_distance
                )

            equity = _current_equity(state, close_price)
        else:
            equity = state["balance"]

        equity_records.append(
            {
                "open_time": current_time,
                "equity": equity,
                "balance": state["balance"],
                "held_symbol": state["held_symbol"],
                "direction": state["direction"],
                "effective_leverage": state["entry_leverage"],
                "cumulative_funding_cash_flow": (
                    cumulative_funding_cash_flow
                )
            }
        )

    if liquidate_at_end and state["held_symbol"] is not None:
        held_frame = frames[state["held_symbol"]]
        total_fees += _close_futures_position(
            state=state,
            raw_price=float(held_frame.iloc[-1]["close"]),
            exit_time=times.iloc[-1],
            exit_reason="end_of_backtest",
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            liquidation_fee_rate=futures_config.liquidation_fee_rate,
            trades=trades
        )
        equity_records[-1]["equity"] = state["balance"]
        equity_records[-1]["balance"] = state["balance"]
        equity_records[-1]["held_symbol"] = None
        equity_records[-1]["direction"] = 0
        equity_records[-1]["effective_leverage"] = 0.0

    if state["held_symbol"] is None:
        final_value = state["balance"]
    else:
        final_price = float(
            frames[state["held_symbol"]].iloc[-1]["close"]
        )
        final_value = _current_equity(state, final_price)

    trade_columns = [
        "strategy",
        "symbol",
        "direction",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "entry_leverage",
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
