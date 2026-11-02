from dataclasses import dataclass

from config import StrategyConfig
from indicators import calculate_adx, calculate_atr, calculate_rsi


@dataclass(frozen=True)
class SleeveDefinition:
    name: str
    weight: float
    entry_column: str
    exit_column: str
    stop_atr_multiple: float | None
    trailing_atr_multiple: float | None = None
    short_entry_column: str | None = None
    short_exit_column: str | None = None
    take_profit_r_multiple: float | None = None
    cost_adjusted_target: bool = False
    long_stop_column: str | None = None
    short_stop_column: str | None = None
    max_holding_bars: int | None = None
    minimum_stop_fraction: float | None = None
    maximum_stop_fraction: float | None = None


def _uses_minimal_structure_trend(config):
    return bool(
        config.trend_weight > 0
        and config.pullback_weight == 0
        and config.range_weight == 0
        and not config.use_regime_filter
        and config.trend_structural_stop_window is not None
        and config.trend_trailing_atr_multiple is None
    )


def _generate_minimal_structure_trend(original_df, config):
    """Calculate only the price structure used by the retained 4h baseline."""
    df = original_df.copy()
    df["ema"] = df["close"].ewm(
        span=config.ema_window, adjust=False, min_periods=config.ema_window
    ).mean()
    df["ema_rising"] = df["ema"] > df["ema"].shift(config.ema_slope_window)
    df["entry_high"] = df["high"].rolling(
        config.breakout_entry_window,
        min_periods=config.breakout_entry_window,
    ).max().shift(1)
    df["exit_low"] = df["low"].rolling(
        config.breakout_exit_window,
        min_periods=config.breakout_exit_window,
    ).min().shift(1)
    df["trend_structural_stop"] = df["low"].rolling(
        config.trend_structural_stop_window,
        min_periods=config.trend_structural_stop_window,
    ).min().shift(1)
    df["donchian_high"] = df["entry_high"]
    df["donchian_low"] = df["trend_structural_stop"]
    df["trend_regime"] = (
        (df["close"] > df["ema"]) & df["ema_rising"]
    ).fillna(False)
    df["range_regime"] = False
    df["neutral_regime"] = ~df["trend_regime"]
    df["trend_entry"] = (
        df["trend_regime"]
        & (df["close"] > df["entry_high"])
        & df["trend_structural_stop"].notna()
    ).fillna(False)
    df["trend_exit"] = (
        (df["close"] < df["exit_low"]) | (df["close"] < df["ema"])
    ).fillna(False)
    sleeve = SleeveDefinition(
        name="profit_first_structure_trend",
        weight=config.trend_weight,
        entry_column="trend_entry",
        exit_column="trend_exit",
        stop_atr_multiple=None,
        long_stop_column="trend_structural_stop",
    )
    return df, [sleeve]


def generate_signals(original_df, config=None):
    strategy_config = config or StrategyConfig()
    if _uses_minimal_structure_trend(strategy_config):
        return _generate_minimal_structure_trend(original_df, strategy_config)

    df = original_df.copy()
    df["ema"] = df["close"].ewm(
        span=strategy_config.ema_window,
        adjust=False,
        min_periods=strategy_config.ema_window,
    ).mean()
    df["ema_rising"] = df["ema"] > df["ema"].shift(
        strategy_config.ema_slope_window
    )
    df["range_ema"] = df["close"].ewm(
        span=strategy_config.range_ema_window,
        adjust=False,
        min_periods=strategy_config.range_ema_window,
    ).mean()
    df["entry_high"] = df["high"].rolling(
        strategy_config.breakout_entry_window,
        min_periods=strategy_config.breakout_entry_window,
    ).max().shift(1)
    df["exit_low"] = df["low"].rolling(
        strategy_config.breakout_exit_window,
        min_periods=strategy_config.breakout_exit_window,
    ).min().shift(1)
    if strategy_config.trend_structural_stop_window is not None:
        df["trend_structural_stop"] = df["low"].rolling(
            strategy_config.trend_structural_stop_window,
            min_periods=strategy_config.trend_structural_stop_window,
        ).min().shift(1)

    df["atr"] = calculate_atr(df, window=strategy_config.atr_window)
    df[["adx", "plus_di", "minus_di"]] = calculate_adx(
        df, window=strategy_config.adx_window
    )
    df["rsi"] = calculate_rsi(df["close"], window=strategy_config.rsi_window)
    df["bollinger_middle"] = df["close"].rolling(
        strategy_config.bollinger_window,
        min_periods=strategy_config.bollinger_window,
    ).mean()
    bollinger_std = df["close"].rolling(
        strategy_config.bollinger_window,
        min_periods=strategy_config.bollinger_window,
    ).std(ddof=0)
    df["bollinger_lower"] = (
        df["bollinger_middle"]
        - strategy_config.bollinger_std_multiple * bollinger_std
    )
    df["bollinger_upper"] = (
        df["bollinger_middle"]
        + strategy_config.bollinger_std_multiple * bollinger_std
    )
    range_ema_change = (
        df["range_ema"]
        - df["range_ema"].shift(strategy_config.range_ema_slope_window)
    ).abs()
    df["range_ema_slope_atr"] = range_ema_change / df["atr"].replace(
        0, float("nan")
    )
    df["range_price_distance_atr"] = (
        (df["close"] - df["range_ema"]).abs()
        / df["atr"].replace(0, float("nan"))
    )

    base_uptrend = (df["close"] > df["ema"]) & df["ema_rising"]
    if strategy_config.use_regime_filter:
        df["trend_regime"] = (
            base_uptrend
            & (df["adx"] >= strategy_config.trend_adx_threshold)
            & (df["plus_di"] > df["minus_di"])
        )
    else:
        df["trend_regime"] = base_uptrend
    df["range_regime"] = (
        strategy_config.use_regime_filter
        & (df["adx"] <= strategy_config.range_adx_threshold)
        & (df["range_ema_slope_atr"] <= strategy_config.range_max_slope_atr)
        & (
            df["range_price_distance_atr"]
            <= strategy_config.range_max_distance_atr
        )
    )
    df["neutral_regime"] = ~(df["trend_regime"] | df["range_regime"])

    df["trend_entry"] = (
        df["trend_regime"]
        & (df["close"] > df["entry_high"])
        & df["atr"].notna()
    )
    df["trend_exit"] = (
        (df["close"] < df["exit_low"]) | (df["close"] < df["ema"])
    )
    if strategy_config.use_regime_filter:
        df["trend_exit"] = df["trend_exit"] | (
            df["adx"] < strategy_config.range_adx_threshold
        )
    df["pullback_entry"] = (
        df["trend_regime"]
        & (df["close"] < df["bollinger_lower"])
        & (df["rsi"] < strategy_config.pullback_entry_rsi)
        & df["atr"].notna()
    )
    df["pullback_exit"] = (
        (df["close"] >= df["bollinger_middle"])
        | (df["rsi"] >= strategy_config.pullback_exit_rsi)
        | (df["close"] < df["ema"])
    )
    df["range_entry"] = (
        df["range_regime"]
        & (df["close"] < df["bollinger_lower"])
        & (df["rsi"] < strategy_config.range_entry_rsi)
        & df["atr"].notna()
    )
    df["range_exit"] = (
        (df["close"] >= df["bollinger_middle"])
        | (df["rsi"] >= strategy_config.range_exit_rsi)
        | (~df["range_regime"])
    )
    for column in (
        "trend_entry", "trend_exit", "pullback_entry", "pullback_exit",
        "range_entry", "range_exit", "trend_regime", "range_regime",
    ):
        df[column] = df[column].fillna(False)
    df["neutral_regime"] = df["neutral_regime"].fillna(True)

    sleeves = []
    if strategy_config.trend_weight > 0:
        structural_stop = strategy_config.trend_structural_stop_window is not None
        sleeves.append(SleeveDefinition(
            name="trend_breakout",
            weight=strategy_config.trend_weight,
            entry_column="trend_entry",
            exit_column="trend_exit",
            stop_atr_multiple=(
                None if structural_stop else strategy_config.trend_stop_atr_multiple
            ),
            trailing_atr_multiple=strategy_config.trend_trailing_atr_multiple,
            long_stop_column="trend_structural_stop" if structural_stop else None,
        ))
    if strategy_config.pullback_weight > 0:
        sleeves.append(SleeveDefinition(
            name="trend_pullback",
            weight=strategy_config.pullback_weight,
            entry_column="pullback_entry",
            exit_column="pullback_exit",
            stop_atr_multiple=strategy_config.pullback_stop_atr_multiple,
        ))
    if strategy_config.range_weight > 0:
        sleeves.append(SleeveDefinition(
            name="range_mean_reversion",
            weight=strategy_config.range_weight,
            entry_column="range_entry",
            exit_column="range_exit",
            stop_atr_multiple=strategy_config.range_stop_atr_multiple,
        ))
    return df, sleeves
