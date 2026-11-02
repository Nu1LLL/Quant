"""组合层风控：与alpha生成完全独立的一层。

1. 波动率目标：用标的自身已实现波动率（不是策略自己在风控下的已实现
   波动率，避免循环依赖）按比例降低敞口，V1版本永远不加杠杆
   （scalar上限为1.0）。
2. 回撤控制：组合权益从峰值回撤超过软阈值后线性降低敞口，超过硬阈值
   封顶在一个最小敞口，不做激进的阈值调参。
3. 单资产敞口上限：由调用方在合成组合时传入cap。
4. Alpha集中度上限：cap_alpha_concentration，防止单个alpha主导组合。
5. 换手控制：no_trade_band，目标敞口变化不超过阈值就不换仓。
6. 逐笔ATR移动止损（可选，stop_loss_atr_multiple）：mini_medallion_v1
   报告发现新系统回撤远大于既有引擎（-18%~-27% vs -2.4%~-4.5%），
   根因是既有引擎有engine.py那样逐笔止损，这里的组合层风控只有
   软性的整体回撤打折，没有替代逐笔止损的机制。这里补上一个简化版：
   只在每根K线**收盘**检查（不是engine.py那种日内最低价检查，保护力度
   更弱，这是一个明确记录在案的简化，不是等价实现），用持仓期间
   收盘价的峰值回撤幅度（ATR标准化）触发强制平仓，且止损判定优先于
   换手不交易带——止损不应该因为"变化幅度不够"而被无视。

回撤控制、换手控制和止损天然是路径依赖的（今天的仓位取决于组合自己
过去的权益曲线和昨天的仓位），所以用一个显式的前向循环实现，不是
矢量化的简单一行代码——但循环里只使用当前及更早的数据，不引入未来
信息。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

import alpha_metrics
from indicators import calculate_atr


@dataclass
class RiskConfig:
    vol_target_annualized: float = None
    vol_window: int = 48
    exposure_cap: float = 1.0
    drawdown_soft_threshold: float = 0.08
    drawdown_hard_threshold: float = 0.20
    drawdown_min_scalar: float = 0.30
    no_trade_band: float = 0.05
    max_alpha_weight: float = 0.50
    stop_loss_atr_multiple: float = None
    stop_loss_atr_window: int = 14


def infer_periods_per_year(open_time):
    open_time = pd.to_datetime(open_time, utc=True)
    if len(open_time) < 2:
        return 0.0
    bar_seconds = open_time.diff().dropna().median().total_seconds()
    if bar_seconds <= 0:
        return 0.0
    return 365.25 * 86400 / bar_seconds


def realized_volatility_scalar(
    close,
    vol_target_annualized,
    vol_window,
    periods_per_year
):
    """标的自身已实现波动率相对目标波动率的缩放系数，上限1.0（不加杠杆）。"""
    if vol_target_annualized is None or periods_per_year <= 0:
        return pd.Series(1.0, index=close.index)

    realized = (
        close.pct_change().rolling(vol_window, min_periods=vol_window).std(ddof=0)
        * np.sqrt(periods_per_year)
    )
    scalar = (vol_target_annualized / realized.replace(0, np.nan)).clip(upper=1.0)
    return scalar.fillna(1.0).clip(lower=0.0)


def cap_alpha_concentration(weights, max_weight, max_iterations=20):
    """限制单个alpha在组合里的权重上限，超出部分按比例分给未封顶的alpha。

    用简单的"水位填充"迭代代替矩阵求解：每轮把超过上限的权重砍到上限，
    再把砍掉的部分按比例分给还没封顶的alpha，重复到没有权重超过上限
    为止（对少数几个alpha通常1-2轮就收敛）。
    """
    if max_weight >= 1.0:
        return weights

    row_totals = weights.sum(axis=1)
    valid_rows = row_totals > 0
    normalized = weights.div(
        row_totals.replace(0, np.nan), axis=0
    ).fillna(0.0)

    for _ in range(max_iterations):
        over_cap = normalized.gt(max_weight + 1e-12)
        if not over_cap.to_numpy().any():
            break

        normalized = normalized.clip(upper=max_weight)
        deficit = 1.0 - normalized.sum(axis=1)

        redistributable = (~over_cap) & (normalized < max_weight - 1e-12)
        redistributable_sum = normalized.where(
            redistributable, 0.0
        ).sum(axis=1)

        share = normalized.where(redistributable, 0.0).div(
            redistributable_sum.replace(0, np.nan), axis=0
        ).fillna(0.0)

        normalized = normalized + share.mul(deficit, axis=0)

    return normalized.where(valid_rows, weights).fillna(0.0)


def _drawdown_scalar(drawdown, soft_threshold, hard_threshold, min_scalar):
    depth = -drawdown
    if depth <= soft_threshold:
        return 1.0
    if depth >= hard_threshold:
        return min_scalar

    span = hard_threshold - soft_threshold
    progress = (depth - soft_threshold) / span
    return 1.0 - progress * (1.0 - min_scalar)


def apply_risk_overlay(
    df,
    raw_exposure,
    config=None,
    delay=1,
    fee_rate=0.001,
    slippage_rate=0.0005
):
    """把alpha合成产生的原始敞口，经过波动率目标、回撤控制、换手控制，
    转成实际持仓路径和逐根K线净收益。

    返回一个DataFrame：open_time, raw_exposure, vol_scalar,
    drawdown_scalar, position, bar_return, stopped_out, net_pnl, equity。
    """
    config = config or RiskConfig()

    periods_per_year = infer_periods_per_year(df["open_time"])
    vol_scalar = realized_volatility_scalar(
        df["close"], config.vol_target_annualized,
        config.vol_window, periods_per_year
    )
    bar_return = alpha_metrics.forward_return(df, horizon=1, delay=delay)

    scaled_target = (
        raw_exposure.fillna(0.0) * vol_scalar.fillna(1.0)
    ).clip(lower=0.0, upper=config.exposure_cap)

    scaled_target_values = scaled_target.to_numpy()
    bar_return_values = bar_return.to_numpy()
    close_values = df["close"].to_numpy()
    n = len(df)

    stop_loss_enabled = config.stop_loss_atr_multiple is not None
    if stop_loss_enabled:
        atr_values = calculate_atr(
            df, window=config.stop_loss_atr_window
        ).to_numpy()
    else:
        atr_values = None

    position = np.zeros(n)
    drawdown_scalar_values = np.ones(n)
    net_pnl = np.zeros(n)
    executed_turnover = np.zeros(n)
    cost_values = np.zeros(n)
    rebalance_flag = np.zeros(n, dtype=bool)
    stopped_out_flag = np.zeros(n, dtype=bool)

    equity = 1.0
    running_max_equity = 1.0
    current_position = 0.0
    peak_close_since_entry = None
    entry_atr = None
    position_epsilon = 1e-9

    for t in range(n):
        drawdown = equity / running_max_equity - 1.0 if running_max_equity > 0 else 0.0
        dd_scalar = _drawdown_scalar(
            drawdown,
            config.drawdown_soft_threshold,
            config.drawdown_hard_threshold,
            config.drawdown_min_scalar
        )
        drawdown_scalar_values[t] = dd_scalar

        target_t = scaled_target_values[t] * dd_scalar
        stop_triggered = False

        # 止损判定：只在已经持仓的情况下检查，用持仓期间收盘价峰值
        # 回撤（ATR标准化）——这是engine.py日内最低价止损的简化近似，
        # 保护力度更弱，只在收盘时刻（每4小时一次）才可能触发。
        if stop_loss_enabled and current_position > position_epsilon:
            close_t = close_values[t]
            if peak_close_since_entry is None:
                peak_close_since_entry = close_t
            else:
                peak_close_since_entry = max(peak_close_since_entry, close_t)

            # 如果开仓时ATR还没有完成预热（entry_atr是NaN），
            # 只要仍然持仓就每根K线用当前已知的ATR重试补齐，
            # 一旦补上就立刻开始生效——只用当前及更早的数据。
            if entry_atr is None and not np.isnan(atr_values[t]):
                entry_atr = atr_values[t]

            if entry_atr is not None and entry_atr > 0:
                stop_level = (
                    peak_close_since_entry
                    - config.stop_loss_atr_multiple * entry_atr
                )
                if close_t <= stop_level:
                    stop_triggered = True
                    target_t = 0.0

        if stop_triggered or abs(target_t - current_position) >= config.no_trade_band:
            turnover = abs(target_t - current_position)
            current_position = target_t
            rebalance_flag[t] = True
            stopped_out_flag[t] = stop_triggered

            if stop_loss_enabled:
                if current_position <= position_epsilon:
                    peak_close_since_entry = None
                    entry_atr = None
                elif entry_atr is None:
                    # 新开仓：记录入场时刻已知的ATR和收盘价作为止损基准
                    entry_atr = (
                        atr_values[t] if not np.isnan(atr_values[t]) else None
                    )
                    peak_close_since_entry = close_values[t]
        else:
            turnover = 0.0

        position[t] = current_position
        executed_turnover[t] = turnover

        ret = bar_return_values[t]
        cost = turnover * (fee_rate + slippage_rate) * 2.0
        cost_values[t] = cost

        if np.isnan(ret):
            bar_pnl = -cost
        else:
            bar_pnl = current_position * ret - cost

        equity *= (1.0 + bar_pnl)
        running_max_equity = max(running_max_equity, equity)
        net_pnl[t] = bar_pnl

    equity_curve = (1.0 + pd.Series(net_pnl, index=df.index)).cumprod()

    return pd.DataFrame({
        "open_time": pd.to_datetime(df["open_time"], utc=True),
        "raw_exposure": raw_exposure.reset_index(drop=True),
        "vol_scalar": vol_scalar.reset_index(drop=True),
        "drawdown_scalar": drawdown_scalar_values,
        "position": position,
        "bar_return": bar_return_values,
        "turnover": executed_turnover,
        "cost": cost_values,
        "rebalanced": rebalance_flag,
        "stopped_out": stopped_out_flag,
        "net_pnl": net_pnl,
        "equity": equity_curve.values
    })
