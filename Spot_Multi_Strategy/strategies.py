from dataclasses import dataclass

from config import StrategyConfig
from indicators import calculate_adx, calculate_atr, calculate_rsi


@dataclass(frozen=True)
class SleeveDefinition:
    # 策略名称
    name: str

    # 策略分配的初始资金比例
    weight: float

    # 做多进场信号列名
    entry_column: str

    # 做多离场信号列名
    exit_column: str

    # 初始ATR止损倍数
    stop_atr_multiple: float

    # 移动ATR止损倍数，None表示不使用
    trailing_atr_multiple: float | None = None


def generate_signals(
    original_df,
    config=None
):
    # 使用传入配置，未传入时使用默认参数
    strategy_config = config or StrategyConfig()

    df = original_df.copy()

    df["ema"] = df["close"].ewm(
        span=strategy_config.ema_window,
        adjust=False,
        min_periods=strategy_config.ema_window
    ).mean()

    df["ema_rising"] = (
        df["ema"]
        >
        df["ema"].shift(
            strategy_config.ema_slope_window
        )
    )

    df["range_ema"] = df["close"].ewm(
        span=strategy_config.range_ema_window,
        adjust=False,
        min_periods=strategy_config.range_ema_window
    ).mean()

    df["entry_high"] = (
        df["high"]
        .rolling(
            window=strategy_config.breakout_entry_window,
            min_periods=strategy_config.breakout_entry_window
        )
        .max()
        .shift(1)
    )

    df["exit_low"] = (
        df["low"]
        .rolling(
            window=strategy_config.breakout_exit_window,
            min_periods=strategy_config.breakout_exit_window
        )
        .min()
        .shift(1)
    )

    df["atr"] = calculate_atr(
        df,
        window=strategy_config.atr_window
    )

    adx_data = calculate_adx(
        df,
        window=strategy_config.adx_window
    )
    df[["adx", "plus_di", "minus_di"]] = adx_data

    df["rsi"] = calculate_rsi(
        df["close"],
        window=strategy_config.rsi_window
    )

    df["bollinger_middle"] = (
        df["close"]
        .rolling(
            window=strategy_config.bollinger_window,
            min_periods=strategy_config.bollinger_window
        )
        .mean()
    )

    bollinger_std = (
        df["close"]
        .rolling(
            window=strategy_config.bollinger_window,
            min_periods=strategy_config.bollinger_window
        )
        .std(ddof=0)
    )

    df["bollinger_lower"] = (
        df["bollinger_middle"]
        -
        strategy_config.bollinger_std_multiple
        *
        bollinger_std
    )

    range_ema_change = (
        df["range_ema"]
        -
        df["range_ema"].shift(
            strategy_config.range_ema_slope_window
        )
    ).abs()

    df["range_ema_slope_atr"] = (
        range_ema_change / df["atr"].replace(0, float("nan"))
    )
    df["range_price_distance_atr"] = (
        (df["close"] - df["range_ema"]).abs()
        /
        df["atr"].replace(0, float("nan"))
    )

    base_uptrend = (
        (df["close"] > df["ema"])
        &
        df["ema_rising"]
    )

    if strategy_config.use_regime_filter:
        df["trend_regime"] = (
            base_uptrend
            &
            (df["adx"] >= strategy_config.trend_adx_threshold)
            &
            (df["plus_di"] > df["minus_di"])
        )
    else:
        df["trend_regime"] = base_uptrend

    df["range_regime"] = (
        strategy_config.use_regime_filter
        &
        (df["adx"] <= strategy_config.range_adx_threshold)
        &
        (
            df["range_ema_slope_atr"]
            <=
            strategy_config.range_max_slope_atr
        )
        &
        (
            df["range_price_distance_atr"]
            <=
            strategy_config.range_max_distance_atr
        )
    )

    df["neutral_regime"] = ~(
        df["trend_regime"] | df["range_regime"]
    )

    # 趋势突破策略：长期趋势向上并突破前期高点
    df["trend_entry"] = (
        df["trend_regime"]
        &
        (df["close"] > df["entry_high"])
        &
        df["atr"].notna()
    )

    df["trend_exit"] = (
        (df["close"] < df["exit_low"])
        |
        (df["close"] < df["ema"])
    )

    if strategy_config.use_regime_filter:
        df["trend_exit"] = (
            df["trend_exit"]
            |
            (
                df["adx"]
                <
                strategy_config.range_adx_threshold
            )
        )

    # 趋势回调策略：长期趋势向上，短期价格跌到布林带下轨并超卖
    df["pullback_entry"] = (
        df["trend_regime"]
        &
        (df["close"] < df["bollinger_lower"])
        &
        (df["rsi"] < strategy_config.pullback_entry_rsi)
        &
        df["atr"].notna()
    )

    df["pullback_exit"] = (
        (df["close"] >= df["bollinger_middle"])
        |
        (df["rsi"] >= strategy_config.pullback_exit_rsi)
        |
        (df["close"] < df["ema"])
    )

    # 震荡策略：只在低ADX横盘中，价格跌破布林带下轨并超卖时买入
    df["range_entry"] = (
        df["range_regime"]
        &
        (df["close"] < df["bollinger_lower"])
        &
        (df["rsi"] < strategy_config.range_entry_rsi)
        &
        df["atr"].notna()
    )

    df["range_exit"] = (
        (df["close"] >= df["bollinger_middle"])
        |
        (df["rsi"] >= strategy_config.range_exit_rsi)
        |
        (~df["range_regime"])
    )

    # 缺少指标的初始K线不能产生进场信号
    df["trend_entry"] = df["trend_entry"].fillna(False)
    df["trend_exit"] = df["trend_exit"].fillna(False)
    df["pullback_entry"] = df["pullback_entry"].fillna(False)
    df["pullback_exit"] = df["pullback_exit"].fillna(False)
    df["range_entry"] = df["range_entry"].fillna(False)
    df["range_exit"] = df["range_exit"].fillna(False)
    df["trend_regime"] = df["trend_regime"].fillna(False)
    df["range_regime"] = df["range_regime"].fillna(False)
    df["neutral_regime"] = df["neutral_regime"].fillna(True)

    sleeves = []

    if strategy_config.trend_weight > 0:
        sleeves.append(
            SleeveDefinition(
                name="trend_breakout",
                weight=strategy_config.trend_weight,
                entry_column="trend_entry",
                exit_column="trend_exit",
                stop_atr_multiple=(
                    strategy_config.trend_stop_atr_multiple
                ),
                trailing_atr_multiple=(
                    strategy_config.trend_trailing_atr_multiple
                )
            )
        )

    if strategy_config.pullback_weight > 0:
        sleeves.append(
            SleeveDefinition(
                name="trend_pullback",
                weight=strategy_config.pullback_weight,
                entry_column="pullback_entry",
                exit_column="pullback_exit",
                stop_atr_multiple=(
                    strategy_config.pullback_stop_atr_multiple
                ),
                trailing_atr_multiple=None
            )
        )

    if strategy_config.range_weight > 0:
        sleeves.append(
            SleeveDefinition(
                name="range_mean_reversion",
                weight=strategy_config.range_weight,
                entry_column="range_entry",
                exit_column="range_exit",
                stop_atr_multiple=(
                    strategy_config.range_stop_atr_multiple
                ),
                trailing_atr_multiple=None
            )
        )

    return df, sleeves
