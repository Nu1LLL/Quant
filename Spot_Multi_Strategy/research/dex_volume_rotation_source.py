"""Frozen DefiLlama DEX-volume vintages and checksum-verified spot candles."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
import math
from pathlib import Path

import pandas as pd

from research.stablecoin_liquidity_source import (
    LiquiditySourceFailure,
    _fetch, _load_month, _manifest, load_spot,
)


ROOT = Path(__file__).parent
DEX_MANIFEST = ROOT / "preregistrations/2026-11-08_three_chain_dex_volume_source_manifest.json"
ALT_MANIFEST = ROOT / "preregistrations/2026-11-08_bnb_trx_spot_daily_source_manifest.csv"
FIRST, LAST = date(2020, 12, 1), date(2025, 12, 31)
COUNT = (LAST - FIRST).days + 1
MONTHS = tuple(f"{year:04d}-{month:02d}" for year in range(2020, 2027)
               for month in range(1, 13)
               if (2020, 12) <= (year, month) <= (2026, 1))
CHAIN_SYMBOLS = {"BSC": "BNBUSDT", "Ethereum": "ETHUSDT", "Tron": "TRXUSDT"}


class DexSourceFailure(ValueError):
    pass


def _parse_dex(body: bytes, spec: dict) -> pd.Series:
    chain = spec["chain"]
    try:
        chart = json.loads(body)["totalDataChart"]
        selected = [x for x in chart if isinstance(x, list) and len(x) == 2
                    and FIRST <= datetime.fromtimestamp(x[0], timezone.utc).date() <= LAST]
        digest = sha256(json.dumps(selected, sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()
        if digest != spec["historical_subset_sha256"]:
            raise DexSourceFailure(f"{chain}: historical DEX volume snapshot changed: {digest}")
        if len(selected) != COUNT:
            raise DexSourceFailure(f"{chain}: incomplete DEX dates")
        values = []
        midnight = datetime(2020, 12, 1, tzinfo=timezone.utc)
        for i, (stamp, raw) in enumerate(selected):
            if type(stamp) is not int or stamp != int((midnight + timedelta(days=i)).timestamp()):
                raise DexSourceFailure(f"{chain}: DEX UTC timestamp gap")
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise DexSourceFailure(f"{chain}: nonpositive/nonfinite DEX volume")
            values.append(value)
        return pd.Series(values, index=pd.date_range(FIRST, LAST, tz="UTC", freq="D"),
                         name=spec["token_symbol"], dtype="float64")
    except DexSourceFailure:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as exc:
        raise DexSourceFailure(f"{chain}: malformed DEX source: {exc}") from exc


def _dex_one(spec: dict) -> tuple[str, pd.Series, dict]:
    body = _fetch(spec["url"], limit=1_000_000)
    series = _parse_dex(body, spec)
    return spec["token_symbol"], series, {
        "chain": spec["chain"], "url": spec["url"],
        "subset_sha256": spec["historical_subset_sha256"],
        "days": len(series), "current_vintage_not_point_in_time": True,
    }


def _new_price_manifest() -> tuple[list[dict], list[dict]]:
    with ALT_MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    if len(rows) != 124 or set(rows[0]) != {"symbol", "month", "url", "official_zip_sha256"}:
        raise DexSourceFailure("BNB/TRX price manifest shape changed")
    expected = [(symbol, month) for symbol in ("BNBUSDT", "TRXUSDT") for month in MONTHS]
    if [(r["symbol"], r["month"]) for r in rows] != expected:
        raise DexSourceFailure("BNB/TRX price manifest grid changed")
    existing = {(r["symbol"], r["month"]): r["sha256"] for r in _manifest()}
    if any(r["official_zip_sha256"] != existing[("BNBUSDT", r["month"])]
           for r in rows if r["symbol"] == "BNBUSDT"):
        raise DexSourceFailure("new BNB hashes disagree with protected existing pins")
    trx = []
    for row in rows:
        symbol, month = row["symbol"], row["month"]
        expected_url = ("https://data.binance.vision/data/spot/monthly/klines/"
                        f"{symbol}/1d/{symbol}-1d-{month}.zip")
        if row["url"] != expected_url or len(row["official_zip_sha256"]) != 64:
            raise DexSourceFailure("malformed new spot source URL/hash")
        if symbol == "TRXUSDT":
            trx.append({"symbol": symbol, "month": month,
                        "sha256": row["official_zip_sha256"]})
    return rows, trx


def load_sources() -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, float], dict]:
    manifest = json.loads(DEX_MANIFEST.read_text(encoding="utf-8"))
    specs = manifest["chains"]
    if (manifest.get("price_or_strategy_result_read") is not False
            or [(s["chain"], s["token_symbol"]) for s in specs]
            != [(c, CHAIN_SYMBOLS[c]) for c in sorted(CHAIN_SYMBOLS)]
            or any(s["days"] != COUNT or s["first"] != str(FIRST)
                   or s["last"] != str(LAST) for s in specs)):
        raise DexSourceFailure("frozen DEX source manifest changed")
    _, trx_manifest = _new_price_manifest()
    with ThreadPoolExecutor(max_workers=3) as pool:
        dex_parts = list(pool.map(_dex_one, specs))
    dex = pd.DataFrame({symbol: series for symbol, series, _ in dex_parts})
    prices, terminal, old_audit = load_spot()
    with ThreadPoolExecutor(max_workers=4) as pool:
        trx_parts = list(pool.map(_load_month, trx_manifest))
    trx_parts.sort(key=lambda x: x[0])
    frames = [part[2] for part in trx_parts[:-1]]
    if not all(isinstance(frame, pd.DataFrame) for frame in frames):
        raise DexSourceFailure("TRX history type changed")
    joined = pd.concat(frames)
    expected = pd.date_range(FIRST, LAST, tz="UTC", freq="D")
    if not joined.index.equals(expected) or not isinstance(trx_parts[-1][2], float):
        raise DexSourceFailure("TRX daily/terminal price grid changed")
    prices["TRXUSDT"] = joined
    terminal["TRXUSDT"] = trx_parts[-1][2]
    if (not dex.index.equals(expected) or set(prices) !=
            {"BNBUSDT", "BTCUSDT", "ETHUSDT", "TRXUSDT"}):
        raise DexSourceFailure("joined source date/symbol grid changed")
    audit = {"dex": [part[2] for part in dex_parts],
             "spot": old_audit + [part[3] for part in trx_parts]}
    return dex, prices, terminal, audit
