"""Synthetic-only causal and account-interface checks; no market reads."""

from decimal import Decimal

import pandas as pd
import pytest

from research.btc_mvrv_coinmetrics import (
    FIRST, LAST, MvrvDataError, directions_from_days,
)
from research.btc_mvrv_direction_adapter import encode
from research.fed_reserves_crypto_sim import _direction, _validate_observations


def _flat_history() -> dict[pd.Timestamp, Decimal]:
    return {day: Decimal("2") for day in pd.date_range(FIRST, LAST, freq="D", tz="UTC")}


def test_december_signal_executes_first_monday_and_ignores_future_days() -> None:
    days = _flat_history()
    target = pd.Timestamp("2020-12-30", tz="UTC")
    days[target] = Decimal("1")  # cheap relative to past-year median
    original, _ = encode(directions_from_days(days))
    releases, values = _validate_observations(original)
    monday = pd.Timestamp("2021-01-04", tz="UTC")
    assert _direction(releases, values, monday) == 1

    for day in pd.date_range("2020-12-31", "2021-01-03", freq="D", tz="UTC"):
        days[day] = Decimal("100")
    modified, _ = encode(directions_from_days(days))
    releases, values = _validate_observations(modified)
    assert _direction(releases, values, monday) == 1


def test_high_mvrv_short_and_exact_median_cash() -> None:
    days = _flat_history()
    expensive = pd.Timestamp("2020-12-30", tz="UTC")
    days[expensive] = Decimal("3")
    observations = directions_from_days(days)
    coded, fingerprint = encode(observations)
    releases, values = _validate_observations(coded)
    assert _direction(releases, values, pd.Timestamp("2021-01-04", tz="UTC")) == -1
    assert fingerprint["cash_count"] > 0
    assert fingerprint["short_count"] >= 1


def test_missing_daily_point_rejects_before_signal() -> None:
    days = _flat_history()
    del days[pd.Timestamp("2020-12-29", tz="UTC")]
    with pytest.raises(MvrvDataError, match="calendar incomplete"):
        directions_from_days(days)
