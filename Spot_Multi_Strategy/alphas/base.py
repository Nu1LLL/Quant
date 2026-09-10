"""Common interface for the OHLCV alpha library.

Every alpha function receives one or more OHLCV dataframes (columns:
open_time, open, high, low, close, volume) that already stop at the
timestamp being evaluated. All helpers here use only trailing
pandas rolling/ewm windows (never center=True) and .shift(positive),
so an AlphaSignal's value at row i must only depend on rows <= i.
See tests/test_alpha_no_lookahead.py for the invariant this buys us:
truncating a dataframe from the end must not change alpha values on
the rows that remain.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class AlphaSignal:
    name: str
    raw_signal: pd.Series
    normalized_signal: pd.Series
    direction: str
    lookback: int
    regime_compatibility: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.raw_signal.index.equals(self.normalized_signal.index):
            raise ValueError(
                f"{self.name}: raw_signal与normalized_signal索引不一致"
            )


def realized_volatility(close, window, min_periods=None):
    min_periods = min_periods or window
    returns = close.pct_change()
    return returns.rolling(window, min_periods=min_periods).std(ddof=0)


def rolling_zscore(series, window, min_periods=None):
    min_periods = min_periods or window
    mean = series.rolling(window, min_periods=min_periods).mean()
    std = series.rolling(window, min_periods=min_periods).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def squash(series, scale):
    """用tanh把无界的原始分数压缩到约(-1,1)之间，scale越大越不容易饱和。"""
    if scale <= 0:
        raise ValueError("squash的scale必须大于0")
    return np.tanh(series.astype(float) / scale)


def clip_unit(series):
    return series.clip(lower=-1.0, upper=1.0)


def make_signal(
    name,
    raw,
    normalized,
    direction,
    lookback,
    regime=None,
    **metadata
):
    return AlphaSignal(
        name=name,
        raw_signal=raw.astype(float),
        normalized_signal=clip_unit(normalized.astype(float)),
        direction=direction,
        lookback=int(lookback),
        regime_compatibility=regime,
        metadata=metadata
    )


def ensemble_signal(name, component_signals, direction, **metadata):
    """对多个AlphaSignal的normalized_signal做等权平均，用于多周期集成alpha。

    刻意不做历史表现加权，避免在alpha候选库内部先做一轮隐藏调参。
    """
    normalized = pd.concat(
        [signal.normalized_signal for signal in component_signals.values()],
        axis=1
    ).mean(axis=1)
    max_lookback = max(
        signal.lookback for signal in component_signals.values()
    )
    return make_signal(
        name=name,
        raw=normalized,
        normalized=normalized,
        direction=direction,
        lookback=max_lookback,
        component_alphas=list(component_signals.keys()),
        **metadata
    )
