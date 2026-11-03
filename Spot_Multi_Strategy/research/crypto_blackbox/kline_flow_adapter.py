"""Research-only flow labels from fully verified Binance Vision 1m klines.

This is an aggregate-kline path, not a reconstruction of raw trades or L2
quotes. The observation schema's ``*_usd`` volume names contain unconverted
USDT quote amounts; they are a USD proxy only while the peg holds.
``availability_lag`` is an explicit simulation assumption: a monthly
historical archive cannot prove when each original bar was first published.
Only ``VerifiedKline`` records from a whole-month verified archive are intended
as input. No trading action, expected return, or order is produced here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from typing import Iterable, Iterator

from .binance_vision_5m import VerifiedKline
from .schemas import CryptoObservation, ObservationClassification
from .signal_intake import SignalIntake


MINUTE = timedelta(minutes=1)
FIVE_MINUTES = timedelta(minutes=5)
PRIOR_WINDOWS = 12
SOURCE = "binance_vision"
# Stable across monthly ZIPs: the individual ZIP fingerprints travel separately.
SOURCE_VERSION = "verified_1m_kline_flow_v1"
LICENSE_ID = "binance_public_data_terms_unverified"
ArchiveFingerprint = tuple[str, str]  # (official URL, verified ZIP SHA-256)


@dataclass(frozen=True)
class _FiveMinuteWindow:
    start: datetime
    end: datetime
    quote_volume: Decimal
    taker_buy_quote_volume: Decimal
    trade_count: int
    fingerprints: tuple[ArchiveFingerprint, ...]


@dataclass(frozen=True)
class KlineFlowResult:
    """One historical label and all source archives used by its 65-minute input."""

    observation: CryptoObservation
    classification: ObservationClassification
    archive_fingerprints: tuple[ArchiveFingerprint, ...]
    assumed_availability_lag: timedelta
    quote_asset: str = field(default="USDT", init=False)
    archive_publication_time_verified: bool = field(default=False, init=False)
    research_only: bool = field(default=True, init=False)
    deployable: bool = field(default=False, init=False)


class KlineFlowAdapter:
    """Consume a single continuous 1m Binance stream without month resets.

    A missing, repeated, out-of-order, wrong-month, or mixed-symbol minute
    halts the adapter. Zero-volume windows remain in the baseline but cannot
    themselves produce a buy-share observation.
    """

    def __init__(self, *, availability_lag: timedelta, horizon_seconds: int = 300) -> None:
        if (not isinstance(availability_lag, timedelta)
                or not timedelta(0) <= availability_lag < FIVE_MINUTES):
            raise ValueError("availability_lag must be in [0, 5 minutes)")
        if type(horizon_seconds) is not int or horizon_seconds <= 0:
            raise ValueError("horizon_seconds must be a positive integer")
        self.availability_lag = availability_lag
        self.horizon_seconds = horizon_seconds
        self._identity: tuple[str, str] | None = None
        self._archive_hash_by_url: dict[str, str] = {}
        self._last_end: datetime | None = None
        self._minute_bars: list[VerifiedKline] = []
        self._windows: deque[_FiveMinuteWindow] = deque(maxlen=PRIOR_WINDOWS + 1)
        self._intake = SignalIntake()
        self._halted = False
        self._emitted = 0

    @staticmethod
    def _fingerprints(bars: Iterable[VerifiedKline]) -> tuple[ArchiveFingerprint, ...]:
        return tuple(dict.fromkeys((bar.source_url, bar.archive_sha256) for bar in bars))

    def _validate_bar(self, bar: VerifiedKline) -> None:
        if not isinstance(bar, VerifiedKline):
            raise TypeError("adapter requires VerifiedKline records")
        if bar.request.interval != "1m":
            raise ValueError("only one-minute klines can feed this adapter")
        if not bar.request.symbol.endswith("USDT"):
            raise ValueError("quote_volume is treated as USDT and requires a USDT pair")
        if bar.open_time.strftime("%Y-%m") != bar.request.month:
            raise ValueError("bar timestamp disagrees with archive month")
        if (bar.open_time.second or bar.open_time.microsecond
                or bar.window_end - bar.open_time != MINUTE):
            raise ValueError("bar is not on the complete UTC one-minute grid")
        if (bar.trade_count == 0) != (bar.quote_volume == 0):
            raise ValueError("zero trade count and quote volume must agree")
        prior_hash = self._archive_hash_by_url.get(bar.source_url)
        if prior_hash is not None and prior_hash != bar.archive_sha256:
            raise ValueError("one archive URL has conflicting SHA-256 versions")
        identity = (bar.request.market, bar.request.symbol)
        if self._identity is None:
            if bar.open_time.minute % 5:
                raise ValueError("first minute must begin on the UTC five-minute grid")
            self._identity = identity
        elif identity != self._identity:
            raise ValueError("stream mixes market or symbol")
        if self._last_end is not None and bar.open_time != self._last_end:
            raise ValueError("missing, duplicate or out-of-order one-minute kline")
        self._archive_hash_by_url[bar.source_url] = bar.archive_sha256

    def _complete_window(self) -> _FiveMinuteWindow:
        bars = tuple(self._minute_bars)
        if len(bars) != 5 or bars[0].open_time.minute % 5:
            raise ValueError("five-minute window is incomplete or misaligned")
        return _FiveMinuteWindow(
            start=bars[0].open_time,
            end=bars[-1].window_end,
            quote_volume=sum((bar.quote_volume for bar in bars), Decimal(0)),
            taker_buy_quote_volume=sum(
                (bar.taker_buy_quote_volume for bar in bars), Decimal(0),
            ),
            trade_count=sum(bar.trade_count for bar in bars),
            fingerprints=self._fingerprints(bars),
        )

    def _make_result(self) -> KlineFlowResult | None:
        if len(self._windows) < PRIOR_WINDOWS + 1:
            return None
        windows = tuple(self._windows)
        current = windows[-1]
        baseline = sum((window.quote_volume for window in windows[:-1]), Decimal(0)) / PRIOR_WINDOWS
        if current.quote_volume == 0 or baseline <= 0:
            return None
        fingerprints = tuple(dict.fromkeys(
            fingerprint for window in windows for fingerprint in window.fingerprints
        ))
        buy_share = float(current.taker_buy_quote_volume / current.quote_volume)
        current_quote = float(current.quote_volume)
        baseline_quote = float(baseline)
        if not all(map(isfinite, (buy_share, current_quote, baseline_quote))):
            raise ValueError("aggregated quote amount cannot be represented as finite float")
        event_material = "|".join((
            SOURCE, SOURCE_VERSION, self._identity[0], self._identity[1],
            current.end.isoformat(), *(f"{url}:{digest}" for url, digest in fingerprints),
        ))
        observation = CryptoObservation(
            event_id=sha256(event_material.encode("utf-8")).hexdigest(),
            source=SOURCE,
            source_version=SOURCE_VERSION,
            license_id=LICENSE_ID,
            market="crypto_spot" if self._identity[0] == "spot" else "crypto_perpetual",
            venue="BINANCE",
            symbol=self._identity[1],
            family="flow_volume_position",
            window_start=current.start,
            event_time=current.end,
            available_time=current.end + self.availability_lag,
            expires_at=current.end + FIVE_MINUTES,
            horizon_seconds=self.horizon_seconds,
            observed_fields={
                "aggressor_buy_share_5m": buy_share,
                "quote_volume_5m_usd": current_quote,
                "quote_volume_baseline_5m_usd": baseline_quote,
                "trade_count_5m": current.trade_count,
            },
        )
        classification = self._intake.classify(observation, observation.available_time)
        self._emitted += 1
        return KlineFlowResult(
            observation=observation,
            classification=classification,
            archive_fingerprints=fingerprints,
            assumed_availability_lag=self.availability_lag,
        )

    def consume(self, bar: VerifiedKline) -> KlineFlowResult | None:
        """Take one verified minute; emit at most one completed-window label."""
        if self._halted:
            raise RuntimeError("adapter halted after an invalid input")
        try:
            self._validate_bar(bar)
            self._minute_bars.append(bar)
            self._last_end = bar.window_end
            if len(self._minute_bars) < 5:
                return None
            window = self._complete_window()
            self._minute_bars.clear()
            self._windows.append(window)
            return self._make_result()
        except Exception:
            self._halted = True
            raise

    def finish(self) -> int:
        """Reject a stream cut through a five-minute window."""
        if self._halted:
            raise RuntimeError("adapter halted after an invalid input")
        if self._minute_bars:
            self._halted = True
            raise ValueError("stream ends with an incomplete five-minute window")
        return self._emitted


def classify_verified_klines(
    bars: Iterable[VerifiedKline], *, availability_lag: timedelta,
    horizon_seconds: int = 300,
) -> Iterator[KlineFlowResult]:
    """Classify one continuous, preverified stream without keeping all bars."""
    adapter = KlineFlowAdapter(
        availability_lag=availability_lag, horizon_seconds=horizon_seconds,
    )
    for bar in bars:
        result = adapter.consume(bar)
        if result is not None:
            yield result
    adapter.finish()
