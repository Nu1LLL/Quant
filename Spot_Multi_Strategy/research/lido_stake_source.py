"""Pinned Lido Sunday ETH stake and existing Binance ETH spot archive loader."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import pandas as pd

from research.eth_defi_tvl_source import SourceFailure, _load_month
from research.stablecoin_liquidity_source import _fetch


URL = "https://api.llama.fi/protocol/lido"
FIRST = date(2020, 12, 20)
LAST = date(2025, 12, 28)
STAKED_SHA = "004a5b99837e8b1584fe19c2932fe24165d50a90543f606dc19d41a331afa0e5"
MANIFEST = Path(__file__).parent / "preregistrations/2026-11-08_eth_defi_tvl_spot_manifest.csv"


def parse_stake(body: bytes) -> tuple[dict[date, float], dict]:
    try:
        payload = json.loads(body)
        if payload.get("name") != "Lido":
            raise SourceFailure("Lido protocol identity changed")
        rows = payload["chainTvls"]["Ethereum"]["tokens"]
        if not isinstance(rows, list):
            raise SourceFailure("Lido Ethereum token history absent")
        start = int(datetime.combine(FIRST, datetime.min.time(), timezone.utc).timestamp())
        end = int(datetime.combine(LAST, datetime.min.time(), timezone.utc).timestamp())
        selected = []
        for row in rows:
            if not isinstance(row, dict) or type(row.get("date")) is not int:
                raise SourceFailure("Lido token row schema invalid")
            stamp = row["date"]
            if start <= stamp <= end and datetime.fromtimestamp(stamp, timezone.utc).weekday() == 6:
                tokens = row.get("tokens")
                if not isinstance(tokens, dict) or "WETH" not in tokens:
                    raise SourceFailure("Lido WETH token entry missing")
                selected.append([stamp, tokens["WETH"]])
        digest = sha256(json.dumps(selected, separators=(",", ":"),
                                   allow_nan=False).encode()).hexdigest()
        if digest != STAKED_SHA:
            raise SourceFailure(f"Lido Sunday WETH historical subset revised: {digest}")
        expected = (LAST - FIRST).days // 7 + 1
        if len(selected) != expected or expected != 263:
            raise SourceFailure("Lido Sunday coverage incomplete")
        out = {}
        for i, (stamp, raw) in enumerate(selected):
            day = FIRST + timedelta(days=i * 7)
            if stamp != start + i * 7 * 86400:
                raise SourceFailure(f"Lido Sunday timestamp gap: {day}")
            if type(raw) not in (int, float) or not math.isfinite(raw) or raw <= 0:
                raise SourceFailure(f"Lido Sunday WETH units invalid: {day}")
            out[day] = float(raw)
        return out, {"source": URL, "ethereum_weth_sunday_sha256": digest,
                     "sundays": len(out), "first": str(FIRST), "last": str(LAST),
                     "raw_response_bytes": len(body), "first_published_vintage_unavailable": True}
    except SourceFailure:
        raise
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise SourceFailure("malformed Lido protocol history") from exc


def load_stake() -> tuple[dict[date, float], dict]:
    return parse_stake(_fetch(URL, limit=4_000_000))


def load_spot() -> tuple[pd.DataFrame, float, list[dict]]:
    try:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle, strict=True))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise SourceFailure("ETH spot manifest unavailable") from exc
    expected_months = pd.period_range("2020-11", "2026-01", freq="M").astype(str).tolist()
    if (len(records) != 63 or [row.get("month") for row in records] != expected_months
            or any(set(row) != {"month", "url", "official_zip_sha256"} for row in records)):
        raise SourceFailure("ETH spot manifest schema/grid changed")
    with ThreadPoolExecutor(max_workers=8) as pool:
        parts = list(pool.map(_load_month, records))
    frames = [part for month, part, _ in parts if month != "2026-01"]
    terminal = next(part for month, part, _ in parts if month == "2026-01")
    if not all(isinstance(frame, pd.DataFrame) for frame in frames) or not isinstance(terminal, float):
        raise SourceFailure("ETH spot source types invalid")
    prices = pd.concat(frames)
    expected = pd.date_range("2020-11-01", "2025-12-31", tz="UTC", freq="D")
    if not prices.index.equals(expected):
        raise SourceFailure("ETH spot complete daily grid invalid")
    return prices, terminal, [audit for _, _, audit in parts]
