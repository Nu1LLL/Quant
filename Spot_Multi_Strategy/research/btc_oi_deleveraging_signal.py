"""Frozen four-week OI contraction → long/cash ledger adapter."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import Sequence

import pandas as pd

from research.btc_oi_deleveraging_data import FIRST, LAST, OiDataError, OiObservation, WEEKS


@dataclass(frozen=True)
class LedgerObservation:
    release_date: pd.Timestamp
    observed_date: pd.Timestamp
    value: int


def encode(observations: Sequence[OiObservation]) -> tuple[tuple[LedgerObservation, ...], dict]:
    if len(observations) != WEEKS:
        raise OiDataError("expected exactly 271 weekly OI observations")
    raw: list[Decimal] = []
    coded: list[int] = []
    output: list[LedgerObservation] = []
    bits = []
    for index, item in enumerate(observations):
        day = FIRST + pd.Timedelta(days=index * 7)
        if (item.observed_date != day or item.release_date != day + pd.Timedelta(days=1)
                or not isinstance(item.value, Decimal) or not item.value.is_finite()
                or item.value <= 0):
            raise OiDataError("OI observation violates frozen weekly calendar/value")
        raw.append(item.value)
        long = index >= 4 and item.value < raw[index - 4]
        value = 0 if index < 4 else coded[index - 4] + int(long)
        coded.append(value)
        bits.append("1" if long else "0")
        output.append(LedgerObservation(item.release_date, item.observed_date, value))
    if output[-1].observed_date != LAST:
        raise OiDataError("OI observation terminal Wednesday changed")
    flags = "".join(bits)
    padding = (-len(flags)) % 8
    packed = int(flags + "0" * padding, 2).to_bytes((len(flags) + padding) // 8, "big").hex()
    return tuple(output), {
        "weekly_direction_count": WEEKS,
        "weekly_long_count": flags.count("1"),
        "weekly_cash_count": flags.count("0"),
        "weekly_direction_bits_hex": packed,
        "weekly_direction_bits_sha256": sha256(flags.encode("ascii")).hexdigest(),
        "bit_order": "chronological Wednesday, MSB first, final low bits zero-padded",
        "first_four_zero_warmup_bits": 4,
        "padding_bits": padding,
    }
