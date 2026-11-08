"""Pinned USDT/USD signal and BTCUSDT spot history for one frozen study."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
import csv

import pandas as pd

from research.coinbase_premium_source import (
    BN_MANIFEST, CB_MANIFEST, SourceFailure, _load_bn_part, _load_cb_part,
)


CB_MANIFEST_SHA = "5033e6d3b2c1c83adf25280d1765a5536f78e2d17ae9ddde6a85f5e1793f1fe8"
BN_MANIFEST_SHA = "3b890f02883c2b81892d7e9c0ed398c343dca70865f555cef246e836693ac766"


def _manifest_rows(path, digest: str) -> list[dict[str, str]]:
    body = path.read_bytes()
    if sha256(body).hexdigest() != digest:
        raise SourceFailure(f"pinned source manifest changed: {path.name}")
    rows = list(csv.DictReader(body.decode("utf-8-sig").splitlines()))
    if not rows:
        raise SourceFailure(f"empty source manifest: {path.name}")
    return rows


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    cb_rows = _manifest_rows(CB_MANIFEST, CB_MANIFEST_SHA)
    cb_rows = [row for row in cb_rows if row["pair"] == "USDT-USD"]
    if len(cb_rows) != 8 or sum(int(row["days"]) for row in cb_rows) != 1856:
        raise SourceFailure("USDT-USD manifest coverage changed")
    bn_rows = _manifest_rows(BN_MANIFEST, BN_MANIFEST_SHA)
    months = pd.period_range("2021-08", "2026-10", freq="M").astype(str).tolist()
    if [row["month"] for row in bn_rows] != months:
        raise SourceFailure("BTCUSDT spot manifest month coverage changed")
    with ThreadPoolExecutor(max_workers=3) as pool:
        cb_parts = list(pool.map(_load_cb_part, cb_rows))
    if any(pair != "USDT-USD" for pair, _, _ in cb_parts):
        raise SourceFailure("unexpected Coinbase product in USDT study")
    coinbase = pd.concat([frame for _, frame, _ in cb_parts]).sort_index()
    expected_cb = pd.date_range("2021-09-01", "2026-09-30", tz="UTC", freq="D")
    if not coinbase.index.equals(expected_cb):
        raise SourceFailure("USDT-USD joined daily grid changed")
    with ThreadPoolExecutor(max_workers=4) as pool:
        bn_parts = list(pool.map(_load_bn_part, bn_rows))
    bn_parts.sort(key=lambda item: item[0])
    terminal = bn_parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("BTC terminal opening price missing")
    if any(not isinstance(part[1], pd.DataFrame) for part in bn_parts[:-1]):
        raise SourceFailure("BTC daily archive unexpected terminal frame")
    binance = pd.concat([part[1] for part in bn_parts[:-1]])
    expected_bn = pd.date_range("2021-08-01", "2026-09-30", tz="UTC", freq="D")
    if not binance.index.equals(expected_bn):
        raise SourceFailure("BTC joined daily grid changed")
    audit = {
        "coinbase_usdt_usd": [part[2] for part in cb_parts],
        "binance_btcusdt_spot": [part[2] for part in bn_parts],
        "coinbase_usdt_usd_days": len(coinbase),
        "binance_btc_spot_days": len(binance),
        "first_signal_day": str(date(2021, 9, 1)),
        "last_signal_day": str(date(2026, 9, 30)),
        "raw_source_files_persisted": False,
    }
    return coinbase, binance, terminal, audit
