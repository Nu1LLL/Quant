"""Immutable, crypto-only, point-in-time market-observation schema.

An observation records what a source made visible, not what happened after an
alert. A source/license identifier is provenance metadata, not a determination
that a particular commercial use is permitted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import math
from numbers import Real
import re
from types import MappingProxyType
from typing import Mapping


CRYPTO_MARKETS = frozenset({
    "crypto_spot", "crypto_perpetual", "crypto_futures", "crypto_options",
})
SIGNAL_FAMILIES = frozenset({
    "flow_volume_position", "funding", "onchain", "market_structure",
    "sentiment",
})
OBSERVED_FIELDS = frozenset({
    "aggressor_buy_share_5m",
    "quote_volume_5m_usd",
    "quote_volume_baseline_5m_usd",
    "open_interest_change_pct_5m",
    "spread_bps",
    "depth_5_usd",
    "trade_count_5m",
})
_OUTCOME_FRAGMENTS = (
    "post_alert", "after_alert", "future_", "forward_", "realized_pnl",
    "unrealized_pnl", "profit", "gain_since", "win_rate", "target_return",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/+-]{0,127}\Z")
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,63}\Z")


def require_utc(value: datetime, name: str) -> datetime:
    """Reject naive/local times; preserve the caller's UTC instant."""
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be a timezone-aware UTC datetime")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    return value


def _identity(value: str, name: str, pattern: re.Pattern[str] = _IDENTIFIER) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise ValueError(f"{name} must be a nonempty canonical identifier")
    return value


def _observed_fields(value: Mapping[str, float]) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("observed_fields must be a mapping")
    fields: dict[str, float] = {}
    for name, raw in value.items():
        if not isinstance(name, str):
            raise TypeError("observed field names must be strings")
        lowered = name.lower()
        if any(fragment in lowered for fragment in _OUTCOME_FRAGMENTS):
            raise ValueError(f"outcome-derived or post-alert field forbidden: {name}")
        if name not in OBSERVED_FIELDS:
            raise ValueError(f"unknown observed field: {name}")
        if isinstance(raw, bool) or not isinstance(raw, Real):
            raise TypeError(f"{name} must be numeric")
        number = float(raw)
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        if name == "aggressor_buy_share_5m" and not 0.0 <= number <= 1.0:
            raise ValueError("aggressor_buy_share_5m must be in [0, 1]")
        if name == "open_interest_change_pct_5m" and number < -100.0:
            raise ValueError("open_interest_change_pct_5m cannot be below -100%")
        if name not in {"open_interest_change_pct_5m"} and number < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        if name == "quote_volume_baseline_5m_usd" and number <= 0.0:
            raise ValueError("quote_volume_baseline_5m_usd must be positive")
        if name == "trade_count_5m" and not number.is_integer():
            raise ValueError("trade_count_5m must be an integer count")
        fields[name] = number
    return MappingProxyType(fields)


@dataclass(frozen=True)
class CryptoObservation:
    """One five-minute or longer observation, with explicit availability.

    The flow classifier only accepts an exact, completed five-minute window.
    Other families can share this source-agnostic schema but have no classifier
    in this package yet. ``horizon_seconds`` is a research label, not a holding
    instruction or permission to place an order.
    """

    event_id: str
    source: str
    source_version: str
    license_id: str
    market: str
    venue: str
    symbol: str
    family: str
    window_start: datetime
    event_time: datetime
    available_time: datetime
    expires_at: datetime
    horizon_seconds: int
    observed_fields: Mapping[str, float]

    def __post_init__(self) -> None:
        for name in ("event_id", "source", "source_version", "license_id", "venue"):
            _identity(getattr(self, name), name)
        _identity(self.symbol, "symbol", _SYMBOL)
        if self.market not in CRYPTO_MARKETS:
            raise ValueError("market must identify a crypto asset or crypto derivative")
        if self.family not in SIGNAL_FAMILIES:
            raise ValueError("unknown signal family")
        for name in ("window_start", "event_time", "available_time", "expires_at"):
            require_utc(getattr(self, name), name)
        if not self.window_start < self.event_time <= self.available_time < self.expires_at:
            raise ValueError("window/event/availability/expiry order is invalid")
        if self.family == "flow_volume_position" and (
            self.event_time - self.window_start != timedelta(minutes=5)
        ):
            raise ValueError("flow_volume_position requires one completed five-minute window")
        if type(self.horizon_seconds) is not int or self.horizon_seconds < 1:
            raise ValueError("horizon_seconds must be a positive integer")
        object.__setattr__(self, "observed_fields", _observed_fields(self.observed_fields))
        if (
            self.market not in {"crypto_perpetual", "crypto_futures"}
            and "open_interest_change_pct_5m" in self.observed_fields
        ):
            raise ValueError("open interest requires a crypto futures or perpetual instrument")

    @property
    def stream_key(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.source, self.source_version, self.market, self.venue,
            self.symbol, self.family,
        )

    @property
    def duplicate_key(self) -> tuple[str, str, str, str, str, str, datetime]:
        """One source/version may publish only one observation per window."""
        return (
            self.source, self.source_version, self.market, self.venue,
            self.symbol, self.family, self.event_time,
        )


@dataclass(frozen=True)
class ObservationClassification:
    """Auditable description, deliberately lacking action/size/PnL fields."""

    event_id: str
    market: str
    venue: str
    symbol: str
    family: str
    label: str
    reasons: tuple[str, ...]
    asof_time: datetime
    tradable: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.label not in {"FIRST_ALERT", "CONTINUATION", "NO_ALERT", "STALE"}:
            raise ValueError("unknown observation label")
        if not self.reasons or not all(isinstance(item, str) and item for item in self.reasons):
            raise ValueError("classification requires audit reasons")
        require_utc(self.asof_time, "asof_time")
