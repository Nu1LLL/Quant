"""Frozen, causal five-minute common-flow signal for three altcoins.

This research module only labels completed windows. It never prices a trade,
routes an order, reads a future bar, or decides whether a signal is profitable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from statistics import median

from .alt_common_flow_data import AltMinute


_MINUTE = timedelta(minutes=1)
_PRIOR_WINDOWS = 12


@dataclass(frozen=True)
class FlowSignal:
    time: datetime  # End of the fully completed five-minute window.
    direction: int  # +1 means buy the basket; -1 means short the basket.
    imbalance: tuple[float, float, float]
    volume_multiple: tuple[float, float, float]
    qualifying_coins: int


class CommonFlowSignal:
    """Preserve a continuous minute grid and twelve *prior* 5m windows."""

    def __init__(self) -> None:
        self.expected_time: datetime | None = None
        self.minute_count = 0
        self.quote = [0.0, 0.0, 0.0]
        self.buy_quote = [0.0, 0.0, 0.0]
        self.prior: deque[tuple[float, float, float]] = deque(maxlen=_PRIOR_WINDOWS)
        self.windows_seen = 0

    def consume(self, bar: AltMinute) -> FlowSignal | None:
        if not isinstance(bar, AltMinute):
            raise TypeError("verified three-coin AltMinute required")
        t = bar.time
        if (t.tzinfo is None or t.utcoffset() != timedelta(0)
                or t.second or t.microsecond):
            raise ValueError("minute must be on the UTC grid")
        if self.expected_time is not None and t != self.expected_time:
            raise ValueError("missing, repeated, or out-of-order flow minute")
        if self.expected_time is None and t.minute % 5:
            raise ValueError("first minute must begin on a five-minute boundary")
        self.expected_time = t + _MINUTE
        for j in range(3):
            q, b = bar.quote[j], bar.taker_buy_quote[j]
            if (not math.isfinite(q) or not math.isfinite(b)
                    or not 0 <= b <= q):
                raise ValueError("invalid quote or taker-buy amount")
            self.quote[j] += q
            self.buy_quote[j] += b
        self.minute_count += 1
        if self.minute_count < 5:
            return None
        end = t + _MINUTE
        if end.minute % 5:
            raise ValueError("five-minute window is misaligned")
        q_now = tuple(self.quote)
        b_now = tuple(self.buy_quote)
        self.quote = [0.0, 0.0, 0.0]
        self.buy_quote = [0.0, 0.0, 0.0]
        self.minute_count = 0
        self.windows_seen += 1

        result = None
        if len(self.prior) == _PRIOR_WINDOWS and all(q > 0 for q in q_now):
            baseline = tuple(
                math.fsum(window[j] for window in self.prior) / _PRIOR_WINDOWS
                for j in range(3)
            )
            if all(math.isfinite(x) and x > 0 for x in baseline):
                oi = tuple(2 * b / q - 1 for b, q in zip(b_now, q_now))
                multiple = tuple(q / m for q, m in zip(q_now, baseline))
                if all(math.isfinite(x) for x in (*oi, *multiple)):
                    positive = sum(v >= 2 and x >= 0.20 for v, x in zip(multiple, oi))
                    negative = sum(v >= 2 and x <= -0.20 for v, x in zip(multiple, oi))
                    center = median(oi)
                    if positive >= 2 and center > 0:
                        result = FlowSignal(end, -1, oi, multiple, positive)
                    elif negative >= 2 and center < 0:
                        result = FlowSignal(end, 1, oi, multiple, negative)
        # The current window joins the baseline only after its signal is fixed.
        self.prior.append(q_now)
        return result

    def finish(self) -> int:
        if self.minute_count:
            raise ValueError("flow stream ended inside a five-minute window")
        return self.windows_seen
