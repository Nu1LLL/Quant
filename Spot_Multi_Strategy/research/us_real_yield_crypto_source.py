"""Frozen FRED real yields and verified Binance spot bars for one experiment."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
import csv
import io
import math
import re

import pandas as pd

from research.btc_eth_transfer_growth_source import (
    SourceFailure, _manifest, _parse_zip,
)
from research.frozen_source_transport import (
    IntegrityFailure, TransportFailure, fetch_pinned,
)


FRED_URL = ("https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10&"
            "cosd=2020-11-01&coed=2025-12-31")
FRED_SHA = "35f372e026b173dfbf847cc2384f3d553ba82c049a0ba3ffc9c52ab5d0b7f9ef"
START = date(2020, 11, 2)
LAST = date(2025, 12, 31)
ASSETS = ("BTC", "ETH")


def parse_fred(body: bytes) -> dict[date, float | None]:
    if len(body) != 21_732 or sha256(body).hexdigest() != FRED_SHA:
        raise SourceFailure("frozen FRED DFII10 bytes changed")
    try:
        reader = csv.reader(io.StringIO(body.decode("utf-8-sig")))
        if next(reader) != ["observation_date", "DFII10"]:
            raise SourceFailure("FRED CSV schema changed")
        rows = list(reader)
        if len(rows) != 1348:
            raise SourceFailure("FRED observation count changed")
        out: dict[date, float | None] = {}
        missing = 0
        previous = None
        for row in rows:
            if len(row) != 2:
                raise SourceFailure("FRED row width changed")
            day = date.fromisoformat(row[0])
            if previous is not None and day <= previous:
                raise SourceFailure("FRED dates not strictly increasing")
            previous = day
            if row[1] in ("", "."):
                value = None
                missing += 1
            else:
                value = float(row[1])
                if not math.isfinite(value):
                    raise SourceFailure("nonfinite FRED yield")
            out[day] = value
        if min(out) != START or max(out) != LAST or missing != 58:
            raise SourceFailure("FRED frozen date/missing grid changed")
        return out
    except SourceFailure:
        raise
    except (csv.Error, UnicodeError, ValueError, TypeError, StopIteration) as exc:
        raise SourceFailure(f"FRED CSV invalid: {exc}") from exc


def _price_month(asset: str, row: dict[str, str]):
    url, month, pin = row["url"], row["month"], row["official_zip_sha256"]
    try:
        receipt = fetch_pinned(url + ".CHECKSUM", max_bytes=500)
        text = receipt.body.decode("ascii").strip()
        if re.fullmatch(r"[0-9a-f]{64}\s+" + re.escape(url.rsplit('/', 1)[-1]),
                        text) is None or text.split()[0] != pin:
            raise SourceFailure(f"{asset} {month}: official checksum changed")
        zip_receipt = fetch_pinned(url, expected_sha256=pin,
                                   max_bytes=4_000_000)
        parsed = _parse_zip(zip_receipt.body, asset, month)
        return asset, month, parsed, {
            "asset": asset, "month": month, "sha256": pin,
            "bytes": len(zip_receipt.body),
            "checksum_attempts": receipt.attempts,
            "zip_attempts": zip_receipt.attempts,
        }
    except (TransportFailure, IntegrityFailure, UnicodeError) as exc:
        raise SourceFailure(f"{asset} {month}: frozen price source failed: {exc}") from exc


def load_sources() -> tuple[dict[date, float | None], dict[str, pd.DataFrame],
                            dict[str, float], dict]:
    manifests = {asset: _manifest(asset) for asset in ASSETS}
    try:
        fred_receipt = fetch_pinned(FRED_URL, expected_sha256=FRED_SHA,
                                    max_bytes=50_000)
    except (TransportFailure, IntegrityFailure) as exc:
        raise SourceFailure(f"frozen FRED source failed: {exc}") from exc
    real_yields = parse_fred(fred_receipt.body)
    jobs = [(asset, row) for asset in ASSETS for row in manifests[asset]]
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(lambda pair: _price_month(*pair), jobs))
    grouped = defaultdict(list)
    for asset, month, parsed, audit in parts:
        grouped[asset].append((month, parsed, audit))
    prices = {}
    terminal = {}
    for asset in ASSETS:
        ordered = sorted(grouped[asset], key=lambda item: item[0])
        if len(ordered) != len(manifests[asset]):
            raise SourceFailure(f"{asset}: price month count changed")
        terminal[asset] = ordered[-1][1]
        if not isinstance(terminal[asset], float):
            raise SourceFailure(f"{asset}: terminal open invalid")
        frames = [part for _, part, _ in ordered[:-1]]
        if not all(isinstance(frame, pd.DataFrame) for frame in frames):
            raise SourceFailure(f"{asset}: spot monthly type invalid")
        prices[asset] = pd.concat(frames)
        first = date(2020, 12, 1) if asset == "BTC" else date(2020, 11, 1)
        grid = pd.date_range(first, LAST, tz="UTC", freq="D")
        if not prices[asset].index.equals(grid):
            raise SourceFailure(f"{asset}: price day grid incomplete")
    return real_yields, prices, terminal, {
        "fred_url": FRED_URL, "fred_sha256": fred_receipt.sha256,
        "fred_bytes": len(fred_receipt.body), "fred_attempts": fred_receipt.attempts,
        "fred_rows": len(real_yields),
        "fred_missing": sum(value is None for value in real_yields.values()),
        "price_months": len(jobs),
        "price_transport_retries": sum(a["checksum_attempts"] - 1 +
                                       a["zip_attempts"] - 1 for *_, a in parts),
        "price_audit": [a for *_, a in parts],
    }
