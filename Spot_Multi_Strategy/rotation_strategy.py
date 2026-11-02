from dataclasses import dataclass
from pathlib import Path
import json
import math

import pandas as pd

from engine import BacktestResult


REQUIRED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume"
]


@dataclass(frozen=True)
class RotationConfig:
    # 判断长期趋势的慢速EMA周期
    slow_ema_window: int = 130

    # 判断短期趋势的快速EMA周期
    fast_ema_window: int = 100

    # 比较BTC和ETH强弱的动量周期
    momentum_window: int = 120

    # 判断慢速EMA是否向上的回看周期
    ema_slope_window: int = 10

    # 每隔多少根K线重新比较一次BTC和ETH
    rebalance_bars: int = 12

    # 最多使用总资金的比例，其余资金保留为现金
    exposure: float = 0.33

    # BTC和ETH同时处于上升趋势时使用的仓位
    # None表示始终使用上面的固定仓位
    broad_market_exposure: float | None = None

    # 波动率仓位启用后，用过去多少根K线估计年化波动率
    volatility_window: int | None = None

    # 希望持仓承担的年化波动率
    target_volatility: float | None = None

    # 波动率很高时仍允许保留的最低仓位
    minimum_exposure: float | None = None


def validate_rotation_config(config):
    # 检查策略参数，避免无意义或危险的配置
    positive_integer_parameters = {
        "慢速EMA周期": config.slow_ema_window,
        "快速EMA周期": config.fast_ema_window,
        "动量周期": config.momentum_window,
        "EMA斜率周期": config.ema_slope_window,
        "再平衡周期": config.rebalance_bars
    }

    for name, value in positive_integer_parameters.items():
        if value <= 0:
            raise ValueError(f"{name}必须大于0")

    if config.fast_ema_window >= config.slow_ema_window:
        raise ValueError("快速EMA周期必须小于慢速EMA周期")

    if not 0 < config.exposure <= 1:
        raise ValueError("现货仓位比例必须大于0且不超过100%")

    if config.broad_market_exposure is not None:
        if not 0 < config.broad_market_exposure <= 1:
            raise ValueError("强趋势仓位必须大于0且不超过100%")

        if config.broad_market_exposure < config.exposure:
            raise ValueError("强趋势仓位不能低于普通趋势仓位")

    volatility_parameters = (
        config.volatility_window,
        config.target_volatility,
        config.minimum_exposure
    )
    enabled_count = sum(
        value is not None
        for value in volatility_parameters
    )

    if enabled_count not in {0, 3}:
        raise ValueError("波动率仓位的三个参数必须同时填写")

    if enabled_count == 3:
        if config.volatility_window <= 1:
            raise ValueError("波动率周期必须大于1")

        if config.target_volatility <= 0:
            raise ValueError("目标波动率必须大于0")

        if not 0 < config.minimum_exposure <= config.exposure:
            raise ValueError("最低仓位必须大于0且不超过最高仓位")


def load_rotation_config(config_file):
    # 从JSON文件读取轮动策略参数
    config_path = Path(config_file)

    with config_path.open("r", encoding="utf-8") as file:
        config_values = json.load(file)

    config = RotationConfig(**config_values)
    validate_rotation_config(config)
    return config


def align_symbol_frames(original_frames):
    # 按共同时间对齐不同交易品种的K线
    if len(original_frames) < 2:
        raise ValueError("轮动策略至少需要两个交易品种")

    cleaned_frames = {}
    common_times = None

    for original_symbol, original_df in original_frames.items():
        symbol = original_symbol.upper()
        df = original_df.copy()

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{symbol}的K线缺少必要字段：{missing_columns}"
            )

        if df.empty:
            raise ValueError(f"{symbol}没有K线数据")

        df["open_time"] = pd.to_datetime(
            df["open_time"],
            utc=True
        )

        df = (
            df[REQUIRED_COLUMNS]
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]

        df[numeric_columns] = df[numeric_columns].apply(
            pd.to_numeric,
            errors="raise"
        )

        if df[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(f"{symbol}的K线中存在空值")

        symbol_times = pd.Index(df["open_time"])
        common_times = (
            symbol_times
            if common_times is None
            else common_times.intersection(symbol_times)
        )

        cleaned_frames[symbol] = df

    common_times = common_times.sort_values()

    if len(common_times) < 2:
        raise ValueError("不同交易品种之间没有足够的共同K线")

    aligned_frames = {}

    for symbol, df in cleaned_frames.items():
        aligned_frames[symbol] = (
            df.set_index("open_time")
            .loc[common_times]
            .reset_index()
        )

    return aligned_frames


def generate_rotation_targets(original_frames, config=None):
    # 只使用当前及过去数据，生成持有BTC、ETH或现金的目标
    rotation_config = config or RotationConfig()
    validate_rotation_config(rotation_config)

    frames = align_symbol_frames(original_frames)
    symbols = list(frames)
    minimum_bars = max(
        rotation_config.slow_ema_window,
        rotation_config.momentum_window
    ) + rotation_config.ema_slope_window + 2

    if len(frames[symbols[0]]) < minimum_bars:
        raise ValueError(
            f"轮动策略至少需要{minimum_bars}根共同K线"
        )

    for symbol, df in frames.items():
        df["slow_ema"] = df["close"].ewm(
            span=rotation_config.slow_ema_window,
            adjust=False,
            min_periods=rotation_config.slow_ema_window
        ).mean()

        df["fast_ema"] = df["close"].ewm(
            span=rotation_config.fast_ema_window,
            adjust=False,
            min_periods=rotation_config.fast_ema_window
        ).mean()

        df["momentum"] = df["close"].pct_change(
            periods=rotation_config.momentum_window
        )

        if rotation_config.volatility_window is not None:
            time_steps = df["open_time"].diff().dropna()
            median_step = time_steps.median()

            if pd.isna(median_step) or median_step <= pd.Timedelta(0):
                raise ValueError("无法从K线时间计算波动率年化系数")

            bars_per_year = (
                pd.Timedelta(days=365.25) / median_step
            )
            df["realized_volatility"] = (
                df["close"]
                .pct_change()
                .rolling(rotation_config.volatility_window)
                .std()
                * math.sqrt(float(bars_per_year))
            )

        df["eligible"] = (
            (df["close"] > df["slow_ema"])
            &
            (df["close"] > df["fast_ema"])
            &
            (
                df["slow_ema"]
                >
                df["slow_ema"].shift(
                    rotation_config.ema_slope_window
                )
            )
            &
            (df["momentum"] > 0)
        ).fillna(False)

        frames[symbol] = df

    targets = []
    current_target = None

    for i in range(len(frames[symbols[0]])):
        eligible_symbols = [
            symbol
            for symbol in symbols
            if bool(frames[symbol].iloc[i]["eligible"])
        ]

        # 趋势失效时立即转回现金，不等待下次定期比较
        if current_target not in eligible_symbols:
            current_target = None

        # 定期选择正动量最强的交易品种
        if i % rotation_config.rebalance_bars == 0:
            if eligible_symbols:
                current_target = max(
                    eligible_symbols,
                    key=lambda symbol: (
                        frames[symbol].iloc[i]["momentum"]
                    )
                )
            else:
                current_target = None

        targets.append(current_target)

    target_series = pd.Series(
        targets,
        index=frames[symbols[0]]["open_time"],
        name="target_symbol",
        dtype="object"
    )

    return frames, target_series


def calculate_rotation_exposure(
    frames,
    desired_symbol,
    signal_position,
    config
):
    # 根据已经结束的信号K线计算本次入场仓位
    all_symbols_eligible = all(
        bool(frames[symbol].iloc[signal_position]["eligible"])
        for symbol in frames
    )

    maximum_exposure = config.exposure

    if (
        all_symbols_eligible
        and
        config.broad_market_exposure is not None
    ):
        maximum_exposure = config.broad_market_exposure

    if config.volatility_window is None:
        return maximum_exposure

    realized_volatility = float(
        frames[desired_symbol]
        .iloc[signal_position]["realized_volatility"]
    )

    if (
        not math.isfinite(realized_volatility)
        or
        realized_volatility <= 0
    ):
        return config.minimum_exposure

    raw_exposure = (
        config.target_volatility / realized_volatility
    )

    return min(
        maximum_exposure,
        max(config.minimum_exposure, raw_exposure)
    )


def _sell_position(
    state,
    raw_price,
    exit_time,
    exit_reason,
    fee_rate,
    slippage_rate,
    trades
):
    # 按不利滑点卖出当前现货仓位
    exit_price = raw_price * (1 - slippage_rate)
    gross_value = state["quantity"] * exit_price
    exit_fee = gross_value * fee_rate
    net_value = gross_value - exit_fee
    state["cash"] += net_value

    net_profit = net_value - state["entry_total_cost"]
    trade_return = (
        net_profit / state["entry_total_cost"]
        if state["entry_total_cost"] > 0
        else 0.0
    )

    trades.append(
        {
            "strategy": "trend_momentum_rotation",
            "symbol": state["held_symbol"],
            "entry_time": state["entry_time"],
            "exit_time": exit_time,
            "entry_price": state["entry_price"],
            "exit_price": exit_price,
            "quantity": state["quantity"],
            "entry_fee": state["entry_fee"],
            "exit_fee": exit_fee,
            "entry_account_value": state["entry_account_value"],
            "gross_profit": (
                (exit_price - state["entry_price"])
                *
                state["quantity"]
            ),
            "net_profit": net_profit,
            "return_rate": trade_return,
            "exit_reason": exit_reason
        }
    )

    state["quantity"] = 0.0
    state["held_symbol"] = None
    state["entry_price"] = 0.0
    state["entry_time"] = None
    state["entry_fee"] = 0.0
    state["entry_total_cost"] = 0.0
    state["entry_account_value"] = 0.0

    return exit_fee


def _buy_position(
    state,
    symbol,
    raw_price,
    entry_time,
    exposure,
    fee_rate,
    slippage_rate
):
    # 只投入设定比例的资金，其余资金继续保留为现金
    entry_price = raw_price * (1 + slippage_rate)
    entry_account_value = state["cash"]
    cash_budget = min(
        state["cash"],
        entry_account_value * exposure
    )

    quantity = cash_budget / (entry_price * (1 + fee_rate))
    gross_cost = quantity * entry_price
    entry_fee = gross_cost * fee_rate
    total_cost = gross_cost + entry_fee

    state["cash"] -= total_cost
    state["quantity"] = quantity
    state["held_symbol"] = symbol
    state["entry_price"] = entry_price
    state["entry_time"] = entry_time
    state["entry_fee"] = entry_fee
    state["entry_total_cost"] = total_cost
    state["entry_account_value"] = entry_account_value

    return entry_fee


def run_rotation_backtest(
    original_frames,
    config=None,
    initial_capital=5000.0,
    fee_rate=0.001,
    slippage_rate=0.0005,
    execution_delay_bars=1,
    liquidate_at_end=True
):
    # 回测BTC、ETH和现金之间的长仓轮动策略
    rotation_config = config or RotationConfig()
    validate_rotation_config(rotation_config)

    if initial_capital <= 0:
        raise ValueError("初始资金必须大于0")

    if not 0 <= fee_rate < 1:
        raise ValueError("手续费率必须位于0到1之间")

    if not 0 <= slippage_rate < 1:
        raise ValueError("滑点率必须位于0到1之间")

    if execution_delay_bars < 1:
        raise ValueError("成交延迟至少为1根K线")

    frames, targets = generate_rotation_targets(
        original_frames,
        config=rotation_config
    )
    symbols = list(frames)
    times = frames[symbols[0]]["open_time"]

    state = {
        "cash": float(initial_capital),
        "quantity": 0.0,
        "held_symbol": None,
        "entry_price": 0.0,
        "entry_time": None,
        "entry_fee": 0.0,
        "entry_total_cost": 0.0,
        "entry_account_value": 0.0
    }

    trades = []
    total_fees = 0.0
    equity_records = [
        {
            "open_time": times.iloc[0],
            "equity": float(initial_capital),
            "cash": float(initial_capital),
            "held_symbol": None
        }
    ]

    for i in range(1, len(times)):
        signal_position = i - execution_delay_bars
        desired_symbol = (
            targets.iloc[signal_position]
            if signal_position >= 0
            else None
        )
        current_time = times.iloc[i]

        # 目标发生变化时，先在当前K线开盘卖出旧仓位
        if (
            state["held_symbol"] is not None
            and
            desired_symbol != state["held_symbol"]
        ):
            held_frame = frames[state["held_symbol"]]
            total_fees += _sell_position(
                state=state,
                raw_price=float(held_frame.iloc[i]["open"]),
                exit_time=current_time,
                exit_reason="target_changed",
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
                trades=trades
            )

        # 卖出后可以在同一根K线开盘切换到新目标
        if (
            state["held_symbol"] is None
            and
            desired_symbol is not None
        ):
            entry_exposure = calculate_rotation_exposure(
                frames=frames,
                desired_symbol=desired_symbol,
                signal_position=signal_position,
                config=rotation_config
            )

            total_fees += _buy_position(
                state=state,
                symbol=desired_symbol,
                raw_price=float(
                    frames[desired_symbol].iloc[i]["open"]
                ),
                entry_time=current_time,
                exposure=entry_exposure,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate
            )

        if state["held_symbol"] is None:
            equity = state["cash"]
        else:
            close_price = float(
                frames[state["held_symbol"]].iloc[i]["close"]
            )
            equity = (
                state["cash"]
                +
                state["quantity"] * close_price
            )

        equity_records.append(
            {
                "open_time": current_time,
                "equity": equity,
                "cash": state["cash"],
                "held_symbol": state["held_symbol"]
            }
        )

    if liquidate_at_end and state["held_symbol"] is not None:
        held_frame = frames[state["held_symbol"]]
        total_fees += _sell_position(
            state=state,
            raw_price=float(held_frame.iloc[-1]["close"]),
            exit_time=times.iloc[-1],
            exit_reason="end_of_backtest",
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            trades=trades
        )
        equity_records[-1]["equity"] = state["cash"]
        equity_records[-1]["cash"] = state["cash"]
        equity_records[-1]["held_symbol"] = None

    if state["held_symbol"] is None:
        final_value = state["cash"]
    else:
        final_value = (
            state["cash"]
            +
            state["quantity"]
            *
            float(frames[state["held_symbol"]].iloc[-1]["close"])
        )

    trade_columns = [
        "strategy",
        "symbol",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "quantity",
        "entry_fee",
        "exit_fee",
        "entry_account_value",
        "gross_profit",
        "net_profit",
        "return_rate",
        "exit_reason"
    ]

    return BacktestResult(
        initial_capital=float(initial_capital),
        final_value=float(final_value),
        total_fees=float(total_fees),
        total_trading_fees=float(total_fees),
        total_funding_fees=0.0,
        equity_curve=pd.DataFrame(equity_records),
        trades=pd.DataFrame(trades, columns=trade_columns)
    )
