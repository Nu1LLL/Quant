"""Deterministic sentiment-to-ledger adapter for the frozen FNG preregistration.

Only complete, already-audited daily observations enter this module.  The
synthetic values are *not* market data: they encode a registered three-way
weekly decision into the older ledger's four-release comparison without
changing that ledger's sizing, fills, or risk accounting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from .fng_crypto_archive import FNGIntegrityError


FIRST_DAY = pd.Timestamp("2020-12-30", tz="UTC")
LAST_DAY = pd.Timestamp("2025-12-24", tz="UTC")
EXPECTED_DAYS = 1821
EXPECTED_WEEKS = 261
FIRST_ANCHOR = pd.Timestamp("2020-12-02", tz="UTC")
RELEASE_LAG = pd.Timedelta(days=1)
DAY = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)
ANCHOR_VALUE = 1000


@dataclass(frozen=True)
class ProxyObservation:
    """Observation accepted by the frozen H.4.1 4h execution ledger."""

    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: int


def _field(observation: object, name: str) -> object:
    if isinstance(observation, Mapping):
        if name not in observation:
            raise FNGIntegrityError(f"FNG observation lacks {name}")
        return observation[name]
    try:
        return getattr(observation, name)
    except AttributeError as exc:
        raise FNGIntegrityError(f"FNG observation lacks {name}") from exc


def _daily_signals(observations: Sequence[object]) -> tuple[tuple[pd.Timestamp, int, int], ...]:
    if (not isinstance(observations, Sequence) or
            isinstance(observations, (str, bytes, bytearray)) or
            len(observations) != EXPECTED_DAYS):
        raise FNGIntegrityError(f"expected exactly {EXPECTED_DAYS} daily FNG observations")

    signals: list[tuple[pd.Timestamp, int, int]] = []
    for index, observation in enumerate(observations):
        day = _field(observation, "day")
        expected_day = FIRST_DAY + index * DAY
        if (not isinstance(day, pd.Timestamp) or day.tzinfo is None or
                str(day.tz) != "UTC" or day != day.normalize() or
                day != expected_day):
            raise FNGIntegrityError("FNG daily dates must be the complete registered UTC grid")
        value = _field(observation, "value")
        if type(value) is not int or value < 0 or value > 100:
            raise FNGIntegrityError("FNG value must be an integer in 0..100")
        if day.weekday() == 2:
            direction = 1 if value <= 25 else -1 if value >= 75 else 0
            signals.append((day, value, direction))

    if (len(signals) != EXPECTED_WEEKS or
            signals[0][0] != FIRST_DAY or signals[-1][0] != LAST_DAY):
        raise FNGIntegrityError("FNG weekly signal grid violates registration")
    return tuple(signals)


def adapt_fng_observations(observations: Sequence[object]) -> tuple[ProxyObservation, ...]:
    """Encode 261 Wednesday decisions with four fixed, signal-free anchors.

    Anchor V[0:4] = 1000.  For the ith registered Wednesday (zero based),
    V[i+4] = V[i] + direction[i].  The frozen ledger compares V[i+4] with
    V[i] on the following Monday, so its sign exactly matches direction[i].
    """
    signals = _daily_signals(observations)
    output: list[ProxyObservation] = []
    values: list[int] = []
    for index in range(4):
        day = FIRST_ANCHOR + index * WEEK
        output.append(ProxyObservation(day + RELEASE_LAG, day, ANCHOR_VALUE))
        values.append(ANCHOR_VALUE)

    for index, (day, _, direction) in enumerate(signals):
        encoded = values[index] + direction
        values.append(encoded)
        output.append(ProxyObservation(day + RELEASE_LAG, day, encoded))
    return tuple(output)


def signal_fingerprint(observations: Sequence[object]) -> tuple[dict[str, object], ...]:
    """Return the registered 261 raw weekly scores and encoded directions."""
    return tuple({"day": day.strftime("%Y-%m-%d"),
                  "value": value, "direction": direction}
                 for day, value, direction in _daily_signals(observations))
