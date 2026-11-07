"""Encode preregistered MVRV directions for the frozen four-week ledger."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import pandas as pd

from research.btc_mvrv_coinmetrics import (
    DirectionObservation, FIRST_WED, MvrvDataError, WEEK_COUNT,
)


@dataclass(frozen=True)
class LedgerObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: int


def encode(observations: Sequence[DirectionObservation]) -> tuple[tuple[LedgerObservation, ...], dict]:
    if len(observations) != WEEK_COUNT:
        raise MvrvDataError("expected 271 MVRV directions")
    output = []
    values: list[int] = []
    symbols = []
    for index, item in enumerate(observations):
        expected = FIRST_WED + pd.Timedelta(days=7 * index)
        if (item.observed_date != expected or
                item.release_date != expected + pd.Timedelta(days=1) or
                type(item.direction) is not int or item.direction not in (-1, 0, 1)):
            raise MvrvDataError("MVRV direction observation violates frozen calendar")
        value = 100 if index < 4 else values[index - 4] + item.direction
        if value < 0:
            raise MvrvDataError("negative ledger encoding")
        values.append(value)
        output.append(LedgerObservation(item.release_date, item.observed_date, value))
        symbols.append({-1: "S", 0: "C", 1: "L"}[item.direction])
    direction_string = "".join(symbols)
    return tuple(output), {
        "weekly_direction_count": WEEK_COUNT,
        "first_four_are_warmup_only": True,
        "long_count": direction_string.count("L"),
        "short_count": direction_string.count("S"),
        "cash_count": direction_string.count("C"),
        "direction_codes": direction_string,
        "direction_codes_sha256": sha256(direction_string.encode("ascii")).hexdigest(),
        "direction_code_legend": "L=long, S=short, C=cash; chronological weekly Wednesday order",
    }
