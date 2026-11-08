"""Frozen CFTC CME Bitcoin institutional-position source and reused spot audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from pathlib import Path
import csv
import math
import subprocess
import zipfile

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure as PriceSourceFailure
from research.btc_exchange_supply_source import _load_month, _manifest


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "preregistrations/2026-11-08_cme_asset_manager_manifest.csv"
CONTRACT = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
EXPECTED_COUNTS = {2020: 52, 2021: 52, 2022: 52, 2023: 52,
                   2024: 53, 2025: 52}
REQUIRED_FIELDS = {"Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
                   "Open_Interest_All", "Asset_Mgr_Positions_Long_All",
                   "Asset_Mgr_Positions_Short_All"}


class SourceFailure(ValueError):
    pass


@dataclass(frozen=True)
class Position:
    report_date: date
    net_oi_ratio: float


def _fetch(url: str) -> bytes:
    try:
        result = subprocess.run(["curl", "-fsS", "-m", "35", url],
                                capture_output=True, check=False, timeout=40)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SourceFailure(f"CFTC archive fetch failed: {url}: {exc}") from exc
    if result.returncode or not result.stdout:
        raise SourceFailure(f"CFTC archive fetch failed: {url}: {result.stderr.decode(errors='replace')[:160]}")
    return result.stdout


def _source_manifest() -> list[dict[str, str]]:
    try:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise SourceFailure(f"CFTC manifest unreadable: {exc}") from exc
    if [int(x["year"]) for x in rows] != list(EXPECTED_COUNTS):
        raise SourceFailure("CFTC manifest year grid changed")
    for row in rows:
        expected = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{row['year']}.zip"
        if row["url"] != expected or len(row["zip_sha256"]) != 64:
            raise SourceFailure("CFTC manifest URL/hash malformed")
    return rows


def _number(value: str) -> float:
    number = float(value.strip().replace(",", ""))
    if not math.isfinite(number):
        raise SourceFailure("CFTC nonfinite position number")
    return number


def parse_archive(body: bytes, year: int, pinned_sha: str) -> list[Position]:
    digest = sha256(body).hexdigest()
    if digest != pinned_sha:
        raise SourceFailure(f"{year}: CFTC archive SHA changed: {digest}")
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != ["FinFutYY.txt"]:
                raise SourceFailure(f"{year}: CFTC ZIP member changed")
            with TextIOWrapper(archive.open("FinFutYY.txt"), encoding="utf-8-sig",
                               newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or len(reader.fieldnames) != 87 or not REQUIRED_FIELDS.issubset(reader.fieldnames):
                    raise SourceFailure(f"{year}: CFTC CSV schema changed")
                selected = []
                for row in reader:
                    if row["Market_and_Exchange_Names"].strip() != CONTRACT:
                        continue
                    stamp = date.fromisoformat(row["Report_Date_as_YYYY-MM-DD"])
                    if stamp.year != year:
                        raise SourceFailure(f"{year}: wrong CFTC report year")
                    interest = _number(row["Open_Interest_All"])
                    long = _number(row["Asset_Mgr_Positions_Long_All"])
                    short = _number(row["Asset_Mgr_Positions_Short_All"])
                    if interest <= 0 or long < 0 or short < 0:
                        raise SourceFailure(f"{stamp}: invalid CFTC positions")
                    ratio = (long - short) / interest
                    if not math.isfinite(ratio) or not -1 <= ratio <= 1:
                        raise SourceFailure(f"{stamp}: invalid net/OI ratio")
                    selected.append(Position(stamp, ratio))
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, csv.Error, ValueError, TypeError, KeyError,
            UnicodeError, OSError) as exc:
        raise SourceFailure(f"{year}: invalid CFTC archive: {exc}") from exc
    selected.sort(key=lambda row: row.report_date)
    if len(selected) != EXPECTED_COUNTS[year] or len({x.report_date for x in selected}) != len(selected):
        raise SourceFailure(f"{year}: CME BTC weekly count or uniqueness failed")
    return selected


def _load_cftc(row: dict[str, str]) -> tuple[list[Position], dict]:
    year = int(row["year"])
    body = _fetch(row["url"])
    positions = parse_archive(body, year, row["zip_sha256"])
    return positions, {"year": year, "url": row["url"],
                       "sha256": row["zip_sha256"], "bytes": len(body),
                       "report_rows": len(positions)}


def load_sources() -> tuple[tuple[Position, ...], pd.DataFrame, float, dict]:
    with ThreadPoolExecutor(max_workers=4) as pool:
        cftc_parts = list(pool.map(_load_cftc, _source_manifest()))
    reports = tuple(sorted((item for rows, _ in cftc_parts for item in rows),
                           key=lambda item: item.report_date))
    if (len(reports) != 313 or reports[0].report_date != date(2020, 1, 7)
            or reports[-1].report_date != date(2025, 12, 30)
            or any((b.report_date - a.report_date).days not in (6, 7, 8)
                   for a, b in zip(reports, reports[1:]))):
        raise SourceFailure("joined CME BTC report-date grid changed")
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            price_parts = list(pool.map(_load_month, _manifest()))
    except PriceSourceFailure as exc:
        raise SourceFailure(f"Binance BTC spot source failure: {exc}") from exc
    price_parts.sort(key=lambda item: item[0])
    terminal = price_parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("BTC spot terminal open missing")
    frames = [item[1] for item in price_parts[:-1]]
    if not all(isinstance(item, pd.DataFrame) for item in frames):
        raise SourceFailure("BTC spot daily archive malformed")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(date(2020, 12, 1),
                                              date(2025, 12, 31), tz="UTC", freq="D")):
        raise SourceFailure("joined BTC spot daily grid changed")
    return reports, prices, terminal, {"cftc": [x[1] for x in cftc_parts],
                                       "binance_spot": [x[2] for x in price_parts]}
