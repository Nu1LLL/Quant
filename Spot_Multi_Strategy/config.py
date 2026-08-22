from dataclasses import dataclass


@dataclass(frozen=True)
class StrategyConfig:
    # 趋势突破策略占组合资金的比例
    trend_weight: float = 0.60

    # 趋势回调策略占组合资金的比例
    pullback_weight: float = 0.40

    # 震荡均值回归策略占组合资金的比例
    range_weight: float = 0.0

    # 是否使用ADX市场状态自动切换
    use_regime_filter: bool = False

    # 判断长期趋势的EMA周期
    ema_window: int = 200

    # 判断EMA是否向上的回看周期
    ema_slope_window: int = 10

    # 趋势突破进场通道周期
    breakout_entry_window: int = 20

    # 趋势突破离场通道周期
    breakout_exit_window: int = 10

    # ATR计算周期
    atr_window: int = 14

    # 趋势策略初始止损距离
    trend_stop_atr_multiple: float = 2.5

    # 趋势策略移动止损距离
    trend_trailing_atr_multiple: float = 3.0

    # RSI计算周期
    rsi_window: int = 14

    # 回调策略进场RSI上限
    pullback_entry_rsi: float = 30.0

    # 回调策略离场RSI下限
    pullback_exit_rsi: float = 55.0

    # 布林带周期
    bollinger_window: int = 20

    # 布林带标准差倍数
    bollinger_std_multiple: float = 2.0

    # 回调策略初始止损距离
    pullback_stop_atr_multiple: float = 2.0

    # ADX计算周期
    adx_window: int = 14

    # ADX达到该值时允许趋势策略进场
    trend_adx_threshold: float = 25.0

    # ADX低于该值时允许震荡策略进场
    range_adx_threshold: float = 20.0

    # 震荡策略使用的中心EMA周期
    range_ema_window: int = 50

    # 判断中心EMA是否平缓的回看周期
    range_ema_slope_window: int = 10

    # 中心EMA变化不能超过多少个ATR
    range_max_slope_atr: float = 1.0

    # 价格距离中心EMA不能超过多少个ATR
    range_max_distance_atr: float = 3.0

    # 震荡策略进场RSI上限
    range_entry_rsi: float = 30.0

    # 震荡策略离场RSI下限
    range_exit_rsi: float = 50.0

    # 震荡策略初始止损距离
    range_stop_atr_multiple: float = 1.5


@dataclass(frozen=True)
class BacktestConfig:
    # 初始资金
    initial_capital: float = 5000.0

    # 单边手续费率，默认按0.10%保守计算
    fee_rate: float = 0.001

    # 单边滑点率，默认按0.05%计算
    slippage_rate: float = 0.0005

    # 每笔交易允许损失本策略账户权益的比例
    risk_per_trade: float = 0.005

    # 每个策略最多使用自己账户资金的比例
    max_sleeve_exposure: float = 1.0

    # 回测结束时是否强制平仓
    liquidate_at_end: bool = True

    # 信号产生后第几根K线开盘成交，1表示下一根
    execution_delay_bars: int = 1


def validate_config(
    strategy_config: StrategyConfig,
    backtest_config: BacktestConfig
):
    # 检查三个策略的资金权重是否合法
    total_weight = (
        strategy_config.trend_weight
        +
        strategy_config.pullback_weight
        +
        strategy_config.range_weight
    )

    if total_weight > 1.0 + 1e-12:
        raise ValueError("策略资金权重之和不能超过1")

    if total_weight <= 0:
        raise ValueError("至少需要给一个策略分配资金")

    if (
        strategy_config.trend_weight < 0
        or
        strategy_config.pullback_weight < 0
        or
        strategy_config.range_weight < 0
    ):
        raise ValueError("策略资金权重不能小于0")

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
        "震荡EMA斜率周期": (
            strategy_config.range_ema_slope_window
        )
    }

    for name, value in positive_integer_parameters.items():
        if value <= 0:
            raise ValueError(f"{name}必须大于0")

    if (
        strategy_config.trend_adx_threshold
        <=
        strategy_config.range_adx_threshold
    ):
        raise ValueError("趋势ADX阈值必须大于震荡ADX阈值")

    if (
        strategy_config.range_weight > 0
        and
        not strategy_config.use_regime_filter
    ):
        raise ValueError("启用震荡策略时必须同时启用市场状态过滤")

    positive_float_parameters = {
        "趋势止损倍数": strategy_config.trend_stop_atr_multiple,
        "趋势移动止损倍数": (
            strategy_config.trend_trailing_atr_multiple
        ),
        "回调止损倍数": strategy_config.pullback_stop_atr_multiple,
        "震荡斜率ATR上限": strategy_config.range_max_slope_atr,
        "震荡距离ATR上限": strategy_config.range_max_distance_atr,
        "震荡止损倍数": strategy_config.range_stop_atr_multiple
    }

    for name, value in positive_float_parameters.items():
        if value <= 0:
            raise ValueError(f"{name}必须大于0")

    if backtest_config.initial_capital <= 0:
        raise ValueError("初始资金必须大于0")

    if not 0 <= backtest_config.fee_rate < 1:
        raise ValueError("手续费率必须位于0到1之间")

    if not 0 <= backtest_config.slippage_rate < 1:
        raise ValueError("滑点率必须位于0到1之间")

    if not 0 < backtest_config.risk_per_trade <= 0.05:
        raise ValueError("单笔风险率必须大于0且不超过5%")

    if not 0 < backtest_config.max_sleeve_exposure <= 1:
        raise ValueError("现货策略最大仓位必须大于0且不超过100%")

    if backtest_config.execution_delay_bars < 1:
        raise ValueError("成交延迟至少为1根K线")
