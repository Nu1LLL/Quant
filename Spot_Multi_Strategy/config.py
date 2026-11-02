from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    """Parameters for the retained trend, pullback and range research paths."""

    trend_weight: float = 0.60
    pullback_weight: float = 0.40
    range_weight: float = 0.0
    use_regime_filter: bool = False

    ema_window: int = 200
    ema_slope_window: int = 10
    breakout_entry_window: int = 20
    breakout_exit_window: int = 10
    atr_window: int = 14
    trend_stop_atr_multiple: float | None = 2.5
    trend_trailing_atr_multiple: float | None = 3.0
    trend_structural_stop_window: int | None = None

    rsi_window: int = 14
    pullback_entry_rsi: float = 30.0
    pullback_exit_rsi: float = 55.0
    bollinger_window: int = 20
    bollinger_std_multiple: float = 2.0
    pullback_stop_atr_multiple: float = 2.0

    adx_window: int = 14
    trend_adx_threshold: float = 25.0
    range_adx_threshold: float = 20.0
    range_ema_window: int = 50
    range_ema_slope_window: int = 10
    range_max_slope_atr: float = 1.0
    range_max_distance_atr: float = 3.0
    range_entry_rsi: float = 30.0
    range_exit_rsi: float = 50.0
    range_stop_atr_multiple: float = 1.5


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 5000.0
    fee_rate: float = 0.001
    short_fee_rate: float | None = None
    slippage_rate: float = 0.0005
    risk_per_trade: float = 0.005
    max_sleeve_exposure: float = 1.0
    max_long_notional: float = 3000.0
    max_short_notional: float = 3000.0
    liquidate_at_end: bool = True
    execution_delay_bars: int = 1


def validate_config(strategy_config: StrategyConfig, backtest_config: BacktestConfig):
    weights = (
        strategy_config.trend_weight,
        strategy_config.pullback_weight,
        strategy_config.range_weight,
    )
    if any(weight < 0 for weight in weights):
        raise ValueError("策略资金权重不能小于0")
    total_weight = sum(weights)
    if total_weight > 1.0 + 1e-12:
        raise ValueError("策略资金权重之和不能超过1")
    if total_weight <= 0:
        raise ValueError("至少需要给一个策略分配资金")

    positive_integer_parameters = {
        "EMA周期": strategy_config.ema_window,
        "EMA斜率周期": strategy_config.ema_slope_window,
        "突破进场周期": strategy_config.breakout_entry_window,
        "突破离场周期": strategy_config.breakout_exit_window,
        "ATR周期": strategy_config.atr_window,
        "RSI周期": strategy_config.rsi_window,
        "布林带周期": strategy_config.bollinger_window,
        "ADX周期": strategy_config.adx_window,
        "震荡EMA周期": strategy_config.range_ema_window,
        "震荡EMA斜率周期": strategy_config.range_ema_slope_window,
    }
    if strategy_config.trend_structural_stop_window is not None:
        positive_integer_parameters["趋势结构止损周期"] = (
            strategy_config.trend_structural_stop_window
        )
    for name, value in positive_integer_parameters.items():
        if value <= 0:
            raise ValueError(f"{name}必须大于0")

    if strategy_config.trend_adx_threshold <= strategy_config.range_adx_threshold:
        raise ValueError("趋势ADX阈值必须大于震荡ADX阈值")
    if strategy_config.range_weight > 0 and not strategy_config.use_regime_filter:
        raise ValueError("启用震荡策略时必须同时启用市场状态过滤")

    positive_float_parameters = {
        "趋势止损倍数": strategy_config.trend_stop_atr_multiple,
        "趋势移动止损倍数": strategy_config.trend_trailing_atr_multiple,
        "回调止损倍数": strategy_config.pullback_stop_atr_multiple,
        "震荡斜率ATR上限": strategy_config.range_max_slope_atr,
        "震荡距离ATR上限": strategy_config.range_max_distance_atr,
        "震荡止损倍数": strategy_config.range_stop_atr_multiple,
    }
    for name, value in positive_float_parameters.items():
        if value is not None and value <= 0:
            raise ValueError(f"{name}必须大于0")
    if (
        strategy_config.trend_weight > 0
        and strategy_config.trend_structural_stop_window is None
        and strategy_config.trend_stop_atr_multiple is None
    ):
        raise ValueError("趋势策略必须配置ATR或结构初始止损")

    if backtest_config.initial_capital <= 0:
        raise ValueError("初始资金必须大于0")
    if not 0 <= backtest_config.fee_rate < 1:
        raise ValueError("手续费率必须位于0到1之间")
    if (
        backtest_config.short_fee_rate is not None
        and not 0 <= backtest_config.short_fee_rate < 1
    ):
        raise ValueError("空头手续费率必须位于0到1之间")
    if not 0 <= backtest_config.slippage_rate < 1:
        raise ValueError("滑点率必须位于0到1之间")
    if not 0 < backtest_config.risk_per_trade <= 0.05:
        raise ValueError("单笔风险率必须大于0且不超过5%")
    if not 0 < backtest_config.max_sleeve_exposure <= 1:
        raise ValueError("现货策略最大仓位必须大于0且不超过100%")
    if backtest_config.max_long_notional <= 0:
        raise ValueError("多头现货名义价值上限必须大于0")
    if backtest_config.max_short_notional <= 0:
        raise ValueError("空头名义价值上限必须大于0")
    if backtest_config.execution_delay_bars < 1:
        raise ValueError("成交延迟至少为1根K线")
