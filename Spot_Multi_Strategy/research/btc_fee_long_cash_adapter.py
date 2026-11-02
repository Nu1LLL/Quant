"""Encode the registered BTC-fee long/cash rule for the frozen 4h ledger.

The old ledger compares each observation with the fourth previous one and
interprets a decrease as a short.  The preregistered fee rule instead holds
cash on a decrease or tie.  This deterministic, in-memory encoding makes the
old comparison binary without modifying its execution or risk accounting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from .btc_fee_coinmetrics_data import BtcFeeDataError


FIRST_WEDNESDAY = pd.Timestamp("2020-10-28", tz="UTC")
LAST_WEDNESDAY = pd.Timestamp("2025-12-31", tz="UTC")
EXPECTED_WEEKS = 271
RELEASE_LAG = pd.Timedelta(days=1)
WEEK = pd.Timedelta(days=7)


@dataclass(frozen=True)
class ProxyObservation:
    """Old-ledger-compatible observation; ``value`` is not a fee amount."""

    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: int


def _field(observation: object, name: str) -> object:
    if isinstance(observation, Mapping):
        if name not in observation:
            raise BtcFeeDataError(f"weekly fee observation lacks {name}")
        return observation[name]
    try:
        return getattr(observation, name)
    except AttributeError as exc:
        raise BtcFeeDataError(f"weekly fee observation lacks {name}") from exc


def _utc_midnight(value: object, name: str) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise BtcFeeDataError(f"{name}: invalid timestamp") from exc
    if stamp.tzinfo is None or str(stamp.tz) != "UTC" or stamp != stamp.normalize():
        raise BtcFeeDataError(f"{name}: UTC midnight required")
    return stamp


def adapt_long_cash_observations(
    observations: Sequence[object],
) -> tuple[ProxyObservation, ...]:
    """Map raw 28-day BTC fee sums R to frozen-ledger values V.

    For i < 4, V_i = 0.  For i >= 4,
    V_i = V_(i-4) + 1[R_i > R_(i-4)].  Therefore the old ledger's
    sign(V_i - V_(i-4)) is one on a strict fee increase and zero otherwise.
    Raw values are compared as ``Decimal`` and are never stored in the output.
    """
    if (
        not isinstance(observations, Sequence)
        or isinstance(observations, (str, bytes, bytearray))
        or len(observations) != EXPECTED_WEEKS
    ):
        raise BtcFeeDataError(f"expected exactly {EXPECTED_WEEKS} weekly fee observations")

    raw: list[Decimal] = []
    encoded: list[int] = []
    output: list[ProxyObservation] = []
    for index, observation in enumerate(observations):
        observed = _utc_midnight(_field(observation, "observed_date"), "observed_date")
        release = _utc_midnight(_field(observation, "release_date"), "release_date")
        expected_day = FIRST_WEDNESDAY + index * WEEK
        if observed != expected_day or observed.weekday() != 2:
            raise BtcFeeDataError("weekly fee dates must be the complete registered Wednesday grid")
        if release != observed + RELEASE_LAG:
            raise BtcFeeDataError("weekly fee release must be the registered D+1 proxy")

        amount = _field(observation, "value")
        if not isinstance(amount, Decimal) or not amount.is_finite() or amount < 0:
            raise BtcFeeDataError("weekly BTC fee sum must be finite nonnegative Decimal")
        raw.append(amount)
        value = 0 if index < 4 else encoded[index - 4] + int(amount > raw[index - 4])
        encoded.append(value)
        output.append(ProxyObservation(release, observed, value))

    if output[-1].observed_date != LAST_WEDNESDAY:
        raise BtcFeeDataError("weekly fee series has wrong terminal Wednesday")
    return tuple(output)
