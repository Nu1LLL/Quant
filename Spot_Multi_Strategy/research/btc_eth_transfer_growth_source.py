"""Frozen Coin Metrics native-transfer counts and audited BTC/ETH spot archives."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import csv
import json
import math
import re
import subprocess
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parent
CM_URL = ("https://community-api.coinmetrics.io/v4/timeseries/asset-metrics?"
          "assets=btc,eth&metrics=TxTfrCnt&frequency=1d&start_time=2020-11-01&"
          "end_time=2025-12-31&page_size=10000&paging_from=start")
CM_SHA = "6059f1c775369dc4997498f48e581a158ba6df43c1e4bf4e49592b91c13bfe15"
METRIC_FIRST = date(2020, 11, 1)
PRICE_FIRST = {"BTC": date(2020, 12, 1), "ETH": date(2020, 11, 1)}
LAST = date(2025, 12, 31)
TERMINAL = date(2026, 1, 1)
METRIC_DAYS = 1887
MANIFESTS = {
    "BTC": (ROOT / "preregistrations/2026-11-08_btc_exchange_supply_spot_manifest.csv",
            "581c681e6d14065242ae8ff464b313fca02038fc16a6e158f37a203ae3cd4a5f"),
    "ETH": (ROOT / "preregistrations/2026-11-08_eth_defi_tvl_spot_manifest.csv",
            "80206c7d2fd9b5965d8beb37d8dcb73516d7f806dfec3f2195ff56f6b716f27e"),
}


class SourceFailure(ValueError):
    pass


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SourceFailure("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(value):
    raise SourceFailure(f"nonfinite JSON constant: {value}")


def _fetch(url: str, *, max_bytes: int = 4_000_000) -> bytes:
    try:
        process = subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error",
             "--max-time", "30", "--max-filesize", str(max_bytes), url],
            capture_output=True, timeout=35, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceFailure(f"source transport failed: {url}: {exc}") from exc
    if process.returncode or not process.stdout or len(process.stdout) > max_bytes:
        detail = process.stderr.decode("utf-8", "replace")[:200]
        raise SourceFailure(f"source request failed: {url}; exit={process.returncode}; {detail}")
    return process.stdout


def parse_transfer_counts(body: bytes) -> dict[str, dict[date, int]]:
    if sha256(body).hexdigest() != CM_SHA:
        raise SourceFailure("Coin Metrics transfer-count response SHA changed")
    try:
        payload = json.loads(body, object_pairs_hook=_unique, parse_constant=_nonfinite)
        if not isinstance(payload, dict) or set(payload) != {"data"}:
            raise SourceFailure("Coin Metrics transfer-count response paging/schema changed")
        rows = payload["data"]
        if not isinstance(rows, list) or len(rows) != 2 * METRIC_DAYS:
            raise SourceFailure("Coin Metrics transfer-count row count changed")
        out: dict[str, dict[date, int]] = {"BTC": {}, "ETH": {}}
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"asset", "time", "TxTfrCnt"}:
                raise SourceFailure("Coin Metrics transfer-count row schema changed")
            asset = row["asset"]
            if asset not in ("btc", "eth"):
                raise SourceFailure("unrecognized Coin Metrics transfer-count asset")
            stamp = row["time"]
            if not isinstance(stamp, str) or not re.fullmatch(
                    r"20\d{2}-\d{2}-\d{2}T00:00:00\.000000000Z", stamp):
                raise SourceFailure("Coin Metrics transfer-count timestamp invalid")
            day = date.fromisoformat(stamp[:10])
            number_text = row["TxTfrCnt"]
            if not isinstance(number_text, str) or not re.fullmatch(r"[0-9]+", number_text):
                raise SourceFailure("Coin Metrics transfer count is not a positive integer")
            number = int(number_text)
            if number <= 0 or day in out[asset.upper()]:
                raise SourceFailure("Coin Metrics transfer count nonpositive or duplicated")
            out[asset.upper()][day] = number
        expected = {METRIC_FIRST + timedelta(days=i) for i in range(METRIC_DAYS)}
        if expected != set(out["BTC"]) or expected != set(out["ETH"]) or max(expected) != LAST:
            raise SourceFailure("Coin Metrics transfer-count day grid changed")
        return out
    except SourceFailure:
        raise
    except (UnicodeError, TypeError, ValueError, OverflowError, KeyError) as exc:
        raise SourceFailure(f"Coin Metrics transfer-count payload invalid: {exc}") from exc


def _manifest(asset: str) -> list[dict[str, str]]:
    path, digest = MANIFESTS[asset]
    try:
        body = path.read_bytes()
    except OSError as exc:
        raise SourceFailure(f"{asset} spot manifest missing: {exc}") from exc
    if sha256(body).hexdigest() != digest:
        raise SourceFailure(f"{asset} spot manifest SHA changed")
    try:
        reader = csv.DictReader(body.decode("utf-8-sig").splitlines())
        if tuple(reader.fieldnames or ()) != ("month", "url", "official_zip_sha256"):
            raise SourceFailure(f"{asset} spot manifest schema changed")
        rows = list(reader)
        months = pd.period_range(PRICE_FIRST[asset].strftime("%Y-%m"),
                                 "2026-01", freq="M").astype(str).tolist()
        if [row["month"] for row in rows] != months:
            raise SourceFailure(f"{asset} spot manifest month grid changed")
        for row in rows:
            url = ("https://data.binance.vision/data/spot/monthly/klines/"
                   f"{asset}USDT/1d/{asset}USDT-1d-{row['month']}.zip")
            if row["url"] != url or not re.fullmatch(r"[0-9a-f]{64}", row["official_zip_sha256"]):
                raise SourceFailure(f"{asset} spot manifest URL/checksum changed")
        return rows
    except SourceFailure:
        raise
    except (csv.Error, UnicodeError, KeyError, TypeError, ValueError) as exc:
        raise SourceFailure(f"{asset} spot manifest invalid: {exc}") from exc


def _parse_zip(body: bytes, asset: str, month: str) -> pd.DataFrame | float:
    member = f"{asset}USDT-1d-{month}.csv"
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != [member]:
                raise SourceFailure(f"{asset} {month}: ZIP member changed")
            raw = archive.read(member)
        if month == "2026-01":
            first_line = raw.partition(b"\n")[0]
            row = next(csv.reader([first_line.decode("utf-8-sig")]))
            expected = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000
            opening = float(row[1])
            if len(row) != 12 or int(row[0]) != expected or not math.isfinite(opening) or opening <= 0:
                raise SourceFailure(f"{asset}: terminal opening price invalid")
            return opening
        rows = list(csv.reader(raw.decode("utf-8-sig").splitlines()))
        first = date.fromisoformat(month + "-01")
        next_month = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
        expected_rows = (next_month - first).days
        if len(rows) != expected_rows:
            raise SourceFailure(f"{asset} {month}: daily ZIP row count changed")
        unit = 1_000_000 if month >= "2025-01" else 1_000
        values = []
        for i, row in enumerate(rows):
            if len(row) != 12:
                raise SourceFailure(f"{asset} {month}: Binance CSV field count changed")
            stamp = int(row[0]); close_stamp = int(row[6])
            expected = int(datetime.combine(first + timedelta(days=i),
                                            datetime.min.time(), timezone.utc).timestamp()) * unit
            if stamp != expected or close_stamp != stamp + 86400 * unit - 1:
                raise SourceFailure(f"{asset} {month}: Binance UTC date grid changed")
            op, hi, lo, cl, quote = [float(row[k]) for k in (1, 2, 3, 4, 7)]
            if (not all(math.isfinite(v) for v in (op, hi, lo, cl, quote))
                    or lo <= 0 or quote < 0 or hi < max(op, cl)
                    or lo > min(op, cl) or lo > hi):
                raise SourceFailure(f"{asset} {month}: invalid OHLC/quote volume")
            values.append((op, hi, lo, cl, quote))
        dates = pd.date_range(first, next_month - timedelta(days=1), tz="UTC", freq="D")
        return pd.DataFrame(values, index=dates,
                            columns=["open", "high", "low", "close", "quote_volume"])
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, UnicodeError, csv.Error, ValueError,
            TypeError, IndexError, OverflowError, RuntimeError) as exc:
        raise SourceFailure(f"{asset} {month}: invalid Binance ZIP: {exc}") from exc


def _load_month(asset: str, row: dict[str, str]) -> tuple[str, str, pd.DataFrame | float, dict]:
    url = row["url"]
    month = row["month"]
    pin = row["official_zip_sha256"]
    checksum = _fetch(url + ".CHECKSUM", max_bytes=500).decode("ascii").strip()
    if re.fullmatch(r"[0-9a-f]{64}\s+" + re.escape(url.rsplit("/", 1)[-1]), checksum) is None:
        raise SourceFailure(f"{asset} {month}: Binance checksum format changed")
    if checksum.split()[0] != pin:
        raise SourceFailure(f"{asset} {month}: Binance official checksum changed")
    body = _fetch(url)
    if sha256(body).hexdigest() != pin:
        raise SourceFailure(f"{asset} {month}: Binance ZIP SHA changed")
    data = _parse_zip(body, asset, month)
    return asset, month, data, {"asset": asset, "month": month, "url": url,
                                "sha256": pin, "bytes": len(body),
                                "terminal_first_open_only": month == "2026-01"}


def load_sources() -> tuple[dict[str, dict[date, int]], dict[str, pd.DataFrame],
                            dict[str, float], dict]:
    manifests = {asset: _manifest(asset) for asset in ("BTC", "ETH")}
    body = _fetch(CM_URL)
    transfers = parse_transfer_counts(body)
    jobs = [(asset, row) for asset in ("BTC", "ETH") for row in manifests[asset]]
    with ThreadPoolExecutor(max_workers=6) as pool:
        parts = list(pool.map(lambda pair: _load_month(*pair), jobs))
    grouped: dict[str, list[tuple[str, pd.DataFrame | float, dict]]] = defaultdict(list)
    for asset, month, data, audit in parts:
        grouped[asset].append((month, data, audit))
    prices: dict[str, pd.DataFrame] = {}
    terminal: dict[str, float] = {}
    for asset in ("BTC", "ETH"):
        ordered = sorted(grouped[asset], key=lambda item: item[0])
        end = ordered[-1][1]
        if not isinstance(end, float) or not math.isfinite(end) or end <= 0:
            raise SourceFailure(f"{asset}: terminal open invalid")
        terminal[asset] = end
        frames = [item[1] for item in ordered[:-1]]
        if not all(isinstance(frame, pd.DataFrame) for frame in frames):
            raise SourceFailure(f"{asset}: nonterminal price frame invalid")
        frame = pd.concat(frames)
        if not frame.index.equals(pd.date_range(PRICE_FIRST[asset], LAST, tz="UTC", freq="D")):
            raise SourceFailure(f"{asset}: joined spot date grid invalid")
        prices[asset] = frame
    return transfers, prices, terminal, {
        "coin_metrics": {"url": CM_URL, "sha256": sha256(body).hexdigest(),
                         "bytes": len(body), "rows": 2 * METRIC_DAYS,
                         "current_vintage_not_point_in_time": True},
        "binance_spot": [part[3] for part in parts],
    }
