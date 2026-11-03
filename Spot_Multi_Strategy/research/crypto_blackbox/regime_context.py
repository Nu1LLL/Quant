"""Read-only, vintage-gated context for crypto research.

This module records Bitcoin valuation/weekly averages and two US macro
observations. It does not calculate returns, forecast markets, size positions,
or issue orders. A filled-in provenance record is necessary but not sufficient
proof that a vintage is genuine: archive digests/timestamps and licensing must
also be checked against the named source outside this pure module. In
particular, today's revised/backfilled macro series is not a historical
point-in-time vintage merely because its observation date is old.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isfinite
from numbers import Real
import re
from types import MappingProxyType
from typing import Iterable

from .schemas import CRYPTO_MARKETS, require_utc


BTC_SPOT_USD = "btc_spot_usd"
BTC_MVRV = "btc_mvrv"
BTC_MA_50W_USD = "btc_ma_50w_usd"
BTC_MA_200W_USD = "btc_ma_200w_usd"
US10Y_YIELD_PCT = "us10y_yield_pct"
US_M2_USD_BILLIONS = "us_m2_usd_billions"

METRIC_UNITS = MappingProxyType({
    BTC_SPOT_USD: "USD/BTC",
    BTC_MVRV: "ratio",
    BTC_MA_50W_USD: "USD/BTC",
    BTC_MA_200W_USD: "USD/BTC",
    US10Y_YIELD_PCT: "percent_per_year",
    US_M2_USD_BILLIONS: "USD_billions",
})
REQUIRED_METRICS = frozenset(METRIC_UNITS)
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_SYMBOL = re.compile(r"[A-Z0-9][A-Z0-9._-]{0,63}\Z")


def _identity(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier")


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class VintageEvidence:
    """Content-addressed record first archived no later than a decision.

    ``first_archived_at`` must be the *historical* first-seen timestamp, not
    the date on which a retrospective download was made. The digest binds the
    chosen published revision to its exact bytes, subject to external audit.
    """

    vintage_id: str
    published_at: datetime
    first_archived_at: datetime
    archive_uri: str
    sha256: str

    def __post_init__(self) -> None:
        _identity(self.vintage_id, "vintage_id")
        if self.vintage_id.casefold() in {"latest", "current", "backfill", "unknown"}:
            raise ValueError("a current/backfilled series is not a historical vintage")
        require_utc(self.published_at, "published_at")
        require_utc(self.first_archived_at, "first_archived_at")
        if self.published_at > self.first_archived_at:
            raise ValueError("archive cannot precede publication")
        if not isinstance(self.archive_uri, str) or not self.archive_uri.startswith(
                ("https://", "s3://")):
            raise ValueError("archive_uri must identify an auditable archive")
        if not isinstance(self.sha256, str) or not _DIGEST.fullmatch(self.sha256):
            raise ValueError("vintage requires a lowercase SHA-256 digest")


@dataclass(frozen=True)
class ContextPoint:
    """One value and the exact revision that was visible when it was used."""

    metric: str
    value: float
    unit: str
    observed_at: datetime
    available_at: datetime
    source: str
    source_version: str
    license_id: str
    vintage: VintageEvidence

    def __post_init__(self) -> None:
        if self.metric not in REQUIRED_METRICS:
            raise ValueError("unknown context metric")
        if self.unit != METRIC_UNITS[self.metric]:
            raise ValueError("metric unit mismatch")
        number = _finite(self.value, "context value")
        if self.metric != US10Y_YIELD_PCT and number <= 0:
            raise ValueError("this context metric must be positive")
        object.__setattr__(self, "value", number)
        require_utc(self.observed_at, "observed_at")
        require_utc(self.available_at, "available_at")
        for name in ("source", "source_version", "license_id"):
            _identity(getattr(self, name), name)
        if not isinstance(self.vintage, VintageEvidence):
            raise ValueError("a genuine historical vintage evidence object is required")
        if not (self.observed_at <= self.vintage.published_at
                <= self.vintage.first_archived_at <= self.available_at):
            raise ValueError("observation, publication, archive, and availability order is invalid")


@dataclass(frozen=True)
class RegimeContextConfig:
    """Maximum age since availability, fixed before evaluating history."""

    spot_max_age: timedelta = timedelta(hours=1)
    mvrv_max_age: timedelta = timedelta(days=3)
    weekly_ma_max_age: timedelta = timedelta(days=10)
    us10y_max_age: timedelta = timedelta(days=7)
    m2_max_age: timedelta = timedelta(days=75)

    def __post_init__(self) -> None:
        for name in ("spot_max_age", "mvrv_max_age", "weekly_ma_max_age",
                     "us10y_max_age", "m2_max_age"):
            value = getattr(self, name)
            if not isinstance(value, timedelta) or value <= timedelta(0):
                raise ValueError(f"{name} must be a positive timedelta")

    def max_age_for(self, metric: str) -> timedelta:
        return {
            BTC_SPOT_USD: self.spot_max_age,
            BTC_MVRV: self.mvrv_max_age,
            BTC_MA_50W_USD: self.weekly_ma_max_age,
            BTC_MA_200W_USD: self.weekly_ma_max_age,
            US10Y_YIELD_PCT: self.us10y_max_age,
            US_M2_USD_BILLIONS: self.m2_max_age,
        }[metric]


@dataclass(frozen=True)
class RegimeContext:
    """Auditable values and arithmetic relations, not a market regime call."""

    target_market: str
    target_symbol: str
    asof_time: datetime
    points: tuple[ContextPoint, ...]
    labels: tuple[str, ...]
    read_only: bool = field(default=True, init=False)
    tradable: bool = field(default=False, init=False)


def _relation(left: float, right: float, name: str) -> str:
    if left > right:
        return f"{name}_above"
    if left < right:
        return f"{name}_below"
    return f"{name}_equal"


def build_regime_context(points: Iterable[ContextPoint], asof_time: datetime,
                         target_market: str, target_symbol: str,
                         config: RegimeContextConfig = RegimeContextConfig()) -> RegimeContext:
    """Fail closed on absent, future, stale, or retrospectively archived data.

    The two US indicators are contextual observations only: they cannot turn
    an equity, bond, or FX instrument into an allowed trading target. No
    screenshot levels or 'MVRV top/bottom' heuristics are embedded here.
    """

    if target_market not in CRYPTO_MARKETS:
        raise ValueError("context may only attach to a crypto trading target")
    if not isinstance(target_symbol, str) or not _SYMBOL.fullmatch(target_symbol):
        raise ValueError("target_symbol must be a canonical crypto symbol")
    if not isinstance(config, RegimeContextConfig):
        raise TypeError("RegimeContextConfig is required")
    require_utc(asof_time, "asof_time")
    values = tuple(points)
    by_metric: dict[str, ContextPoint] = {}
    for point in values:
        if not isinstance(point, ContextPoint):
            raise TypeError("all context inputs must be ContextPoint")
        if point.metric in by_metric:
            raise ValueError("duplicate context metric")
        by_metric[point.metric] = point
        if point.vintage.first_archived_at > asof_time:
            raise ValueError("historical vintage was not archived at decision time")
        if point.available_at > asof_time:
            raise ValueError("context value was not available at decision time")
        if asof_time - point.available_at > config.max_age_for(point.metric):
            raise ValueError(f"{point.metric} is stale at decision time")
    if set(by_metric) != REQUIRED_METRICS:
        missing = sorted(REQUIRED_METRICS - set(by_metric))
        raise ValueError(f"missing required context metrics: {missing}")

    spot = by_metric[BTC_SPOT_USD].value
    ma50 = by_metric[BTC_MA_50W_USD].value
    ma200 = by_metric[BTC_MA_200W_USD].value
    mvrv = by_metric[BTC_MVRV].value
    labels = (
        "btc_mvrv_below_one" if mvrv < 1.0 else
        "btc_mvrv_equal_one" if mvrv == 1.0 else "btc_mvrv_above_one",
        _relation(spot, ma50, "btc_spot_vs_50w_ma"),
        _relation(spot, ma200, "btc_spot_vs_200w_ma"),
        _relation(ma50, ma200, "btc_50w_ma_vs_200w_ma"),
    )
    ordered = tuple(by_metric[metric] for metric in sorted(REQUIRED_METRICS))
    return RegimeContext(target_market, target_symbol, asof_time, ordered, labels)
