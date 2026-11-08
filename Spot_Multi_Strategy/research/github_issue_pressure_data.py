"""Load only the two preregistered, checksum-pinned five-minute futures tapes."""

from __future__ import annotations

from calendar import monthrange
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .weekday_seasonality_data import (
    FUNDING_FIELDS, FUNDING_MANIFESTS, MONTHS, PRICE_FIELDS,
    PRICE_MANIFESTS, _funding_bar_index, _funding_month, _pins,
    _price_month,
)


SYMBOLS = ("BTCUSDT", "ETHUSDT")


@dataclass(frozen=True, slots=True)
class PairTape:
    opening: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    funding: dict[int, tuple[tuple[int, float], ...]]
    price_hashes: dict[str, str]
    funding_hashes: dict[str, str]


def load_pair_tape() -> PairTape:
    prices = _pins(PRICE_MANIFESTS, PRICE_FIELDS, "5m")
    rates = _pins(FUNDING_MANIFESTS, FUNDING_FIELDS, "fundingRate")
    total = sum(monthrange(*map(int, m.split("-")))[1] * 288 for m in MONTHS)
    arrays = [np.empty((total, 2), dtype=np.float64) for _ in range(5)]
    funding: dict[int, list[tuple[int, float]]] = {}
    offset = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for month in MONTHS:
            expected = monthrange(*map(int, month.split("-")))[1] * 288
            jobs = {symbol: pool.submit(_price_month, symbol, month,
                                        prices[(month, symbol)])
                    for symbol in SYMBOLS}
            for column, symbol in enumerate(SYMBOLS):
                values = jobs[symbol].result()
                if values.shape != (expected, 5):
                    raise ValueError(f"registered price grid changed: {symbol} {month}")
                for k, array in enumerate(arrays):
                    array[offset:offset + expected, column] = values[:, k]
            offset += expected
        if offset != total:
            raise ValueError("registered tape length changed")
        for month in MONTHS:
            jobs = {symbol: pool.submit(_funding_month, symbol, month,
                                        rates[(month, symbol)])
                    for symbol in SYMBOLS}
            for column, symbol in enumerate(SYMBOLS):
                for stamp, rate in jobs[symbol].result():
                    index = _funding_bar_index(stamp, total)
                    funding.setdefault(index, []).append((column, rate))
    for array in arrays:
        if not np.all(np.isfinite(array)):
            raise ValueError("nonfinite price tape")
        array.setflags(write=False)
    return PairTape(
        *arrays,
        {index: tuple(rows) for index, rows in sorted(funding.items())},
        {f"{m}:{s}": prices[(m, s)]["sha256"] for m in MONTHS for s in SYMBOLS},
        {f"{m}:{s}": rates[(m, s)]["sha256"] for m in MONTHS for s in SYMBOLS},
    )
