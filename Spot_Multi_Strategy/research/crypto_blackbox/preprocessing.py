"""Five-minute crypto trade preprocessing with an explicitly closed baseline.

Only current-window raw trades and 13 independently complete five-minute
volume/count attestations are accepted. The prior 12 attestations form an
unweighted, trailing volume baseline; there is no outcome label, order,
forecast or return calculation here. A provider adapter must verify that a
``CompletedTradeWindow`` really came from a complete point-in-time source.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
import math
from numbers import Real
from typing import Sequence

from .schemas import CRYPTO_MARKETS, CryptoObservation, _identity, require_utc


WINDOW = timedelta(minutes=5)
BASELINE_WINDOWS = 12


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return number


def _identity_fields(record: object) -> tuple[str, str, str, str, str, str]:
    fields = (
        record.source, record.source_version, record.license_id,
        record.market, record.venue, record.symbol,
    )
    for name, value in zip(
        ("source", "source_version", "license_id", "market", "venue", "symbol"),
        fields,
    ):
        _identity(value, name)
    if record.market not in CRYPTO_MARKETS:
        raise ValueError("only crypto markets are supported")
    return fields


def _aligned_boundary(value: datetime) -> bool:
    return value.minute % 5 == 0 and value.second == 0 and value.microsecond == 0


def binance_aggressor_side(is_buyer_maker: bool) -> str:
    """Binance buyer-is-maker flag: True => taker SELL, False => taker BUY."""
    if type(is_buyer_maker) is not bool:
        raise TypeError("is_buyer_maker must be an explicit boolean")
    return "SELL" if is_buyer_maker else "BUY"


@dataclass(frozen=True)
class TradeEvent:
    event_id: str
    source: str
    source_version: str
    license_id: str
    market: str
    venue: str
    symbol: str
    event_time: datetime
    available_time: datetime
    price: float
    quantity: float
    aggressor_side: str

    def __post_init__(self) -> None:
        _identity(self.event_id, "event_id")
        _identity_fields(self)
        require_utc(self.event_time, "event_time")
        require_utc(self.available_time, "available_time")
        if self.event_time > self.available_time:
            raise ValueError("trade availability precedes event")
        _number(self.price, "price", positive=True)
        _number(self.quantity, "quantity", positive=True)
        if self.aggressor_side not in {"BUY", "SELL"}:
            raise ValueError("aggressor_side must be explicit BUY or SELL")

    @property
    def quote_notional(self) -> float:
        amount = float(self.price) * float(self.quantity)
        if not math.isfinite(amount):
            raise ValueError("trade quote notional overflow")
        return amount


@dataclass(frozen=True)
class CompletedTradeWindow:
    source: str
    source_version: str
    license_id: str
    market: str
    venue: str
    symbol: str
    window_start: datetime
    window_end: datetime
    available_time: datetime
    trade_count: int
    quote_volume_usd: float
    complete: bool

    def __post_init__(self) -> None:
        _identity_fields(self)
        for name in ("window_start", "window_end", "available_time"):
            require_utc(getattr(self, name), name)
        if self.window_end - self.window_start != WINDOW:
            raise ValueError("completed trade window must span exactly five minutes")
        if not _aligned_boundary(self.window_start) or not _aligned_boundary(self.window_end):
            raise ValueError("completed trade window must align to the UTC five-minute grid")
        if self.available_time < self.window_end:
            raise ValueError("window cannot be available before it is complete")
        if self.complete is not True:
            raise ValueError("incomplete trade window is not an admissible baseline")
        if type(self.trade_count) is not int or self.trade_count < 0:
            raise ValueError("trade_count must be a nonnegative integer")
        _number(self.quote_volume_usd, "quote_volume_usd")
        if (self.trade_count == 0) != (self.quote_volume_usd == 0):
            raise ValueError("zero trade count and quote volume must agree")


@dataclass(frozen=True)
class OpenInterestSnapshot:
    source: str
    source_version: str
    license_id: str
    market: str
    venue: str
    symbol: str
    event_time: datetime
    available_time: datetime
    contracts: float

    def __post_init__(self) -> None:
        _identity_fields(self)
        if self.market not in {"crypto_perpetual", "crypto_futures"}:
            raise ValueError("open interest belongs to crypto futures or perpetuals")
        require_utc(self.event_time, "event_time")
        require_utc(self.available_time, "available_time")
        if self.event_time > self.available_time:
            raise ValueError("open-interest availability precedes event")
        _number(self.contracts, "contracts")


def build_flow_observation(
    trades: Sequence[TradeEvent],
    completed_windows: Sequence[CompletedTradeWindow],
    *,
    decision_time: datetime,
    horizon_seconds: int = 300,
    open_interest_start: OpenInterestSnapshot | None = None,
    open_interest_end: OpenInterestSnapshot | None = None,
) -> CryptoObservation:
    """Build one non-trading flow observation from available closed data.

    ``completed_windows`` must cover exactly the current window and 12 prior
    contiguous five-minute windows. The current count and quote volume must
    match the raw, unaggregated current-window trades. A zero-trade prior
    window is valid only when explicitly attested with count and volume zero.
    """
    require_utc(decision_time, "decision_time")
    windows = tuple(completed_windows)
    current_trades = tuple(trades)
    if len(windows) != BASELINE_WINDOWS + 1:
        raise ValueError("exactly 13 complete five-minute windows are required")
    if not all(isinstance(window, CompletedTradeWindow) for window in windows):
        raise TypeError("all completed_windows must be CompletedTradeWindow instances")
    if not current_trades or not all(isinstance(trade, TradeEvent) for trade in current_trades):
        raise ValueError("current window requires raw TradeEvent records")

    windows = tuple(sorted(windows, key=lambda window: window.window_start))
    current = windows[-1]
    identity = _identity_fields(current)
    expected_starts = tuple(
        current.window_end - WINDOW * offset
        for offset in range(BASELINE_WINDOWS + 1, 0, -1)
    )
    actual_starts = tuple(window.window_start for window in windows)
    if actual_starts != expected_starts:
        raise ValueError("missing, duplicate or noncontiguous completed windows")
    for window in windows:
        if _identity_fields(window) != identity:
            raise ValueError("completed windows mix source, license, venue or symbol")
        if window.available_time > decision_time:
            raise ValueError("completed window is not visible at decision_time")
    if current.window_end > decision_time:
        raise ValueError("current five-minute window has not ended")

    seen_ids: set[str] = set()
    previous_trade_time: datetime | None = None
    for trade in current_trades:
        if _identity_fields(trade) != identity:
            raise ValueError("trades mix source, license, venue or symbol")
        if trade.event_id in seen_ids:
            raise ValueError("duplicate trade event_id")
        seen_ids.add(trade.event_id)
        if not current.window_start <= trade.event_time < current.window_end:
            raise ValueError("trade event lies outside the completed current window")
        if previous_trade_time is not None and trade.event_time < previous_trade_time:
            raise ValueError("trades are not in event-time order")
        previous_trade_time = trade.event_time
        if trade.available_time > decision_time:
            raise ValueError("trade is not visible at decision_time")

    if len(current_trades) != current.trade_count:
        raise ValueError("raw trade count disagrees with current completed window")
    current_quote = math.fsum(trade.quote_notional for trade in current_trades)
    if not math.isfinite(current_quote) or not math.isclose(
        current_quote, float(current.quote_volume_usd), rel_tol=1e-8, abs_tol=1e-6,
    ):
        raise ValueError("raw trade quote amount disagrees with current completed window")
    baseline = math.fsum(float(window.quote_volume_usd) for window in windows[:-1]) / BASELINE_WINDOWS
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("prior 12-window quote volume baseline is invalid")
    buy_quote = math.fsum(
        trade.quote_notional for trade in current_trades if trade.aggressor_side == "BUY"
    )
    buy_share = buy_quote / current_quote
    if not math.isfinite(buy_share):
        raise ValueError("aggressor buy share is invalid")

    fields: dict[str, float] = {
        "aggressor_buy_share_5m": buy_share,
        "quote_volume_5m_usd": current_quote,
        "quote_volume_baseline_5m_usd": baseline,
        "trade_count_5m": len(current_trades),
    }
    visible_times = [window.available_time for window in windows]
    visible_times.extend(trade.available_time for trade in current_trades)
    if (open_interest_start is None) != (open_interest_end is None):
        raise ValueError("open interest requires both start and end snapshots")
    if open_interest_start is not None and open_interest_end is not None:
        if current.market not in {"crypto_perpetual", "crypto_futures"}:
            raise ValueError("spot/options flow cannot include open interest")
        for snapshot, expected_time in (
            (open_interest_start, current.window_start),
            (open_interest_end, current.window_end),
        ):
            if not isinstance(snapshot, OpenInterestSnapshot):
                raise TypeError("open interest must be OpenInterestSnapshot instances")
            if _identity_fields(snapshot) != identity:
                raise ValueError("open interest mixes source, license, venue or symbol")
            if snapshot.event_time != expected_time:
                raise ValueError("open interest snapshots must bracket the current window")
            if snapshot.available_time > decision_time:
                raise ValueError("open interest was not visible at decision_time")
            visible_times.append(snapshot.available_time)
        if open_interest_start.contracts <= 0:
            raise ValueError("initial open interest must be positive")
        change_pct = (
            float(open_interest_end.contracts) / float(open_interest_start.contracts) - 1.0
        ) * 100.0
        if not math.isfinite(change_pct):
            raise ValueError("open-interest change is invalid")
        fields["open_interest_change_pct_5m"] = change_pct

    event_id_material = "|".join((*identity, current.window_end.isoformat()))
    observation_id = sha256(event_id_material.encode("utf-8")).hexdigest()
    return CryptoObservation(
        event_id=observation_id,
        source=current.source,
        source_version=current.source_version,
        license_id=current.license_id,
        market=current.market,
        venue=current.venue,
        symbol=current.symbol,
        family="flow_volume_position",
        window_start=current.window_start,
        event_time=current.window_end,
        available_time=max(visible_times),
        expires_at=current.window_end + WINDOW,
        horizon_seconds=horizon_seconds,
        observed_fields=fields,
    )
