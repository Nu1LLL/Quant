"""Frozen CFTC futures-vs-combined option exposure and Binance spot audit."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO, TextIOWrapper
from pathlib import Path
import csv
import math
import re
import zipfile

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure as PriceSourceFailure
from research.btc_exchange_supply_source import _load_month, _manifest as _price_manifest
from research.cme_asset_manager_source import SourceFailure, _fetch


ROOT = Path(__file__).resolve().parent
FUT_MANIFEST = ROOT / "preregistrations/2026-11-08_cme_asset_manager_manifest_v2.csv"
COM_MANIFEST = ROOT / "preregistrations/2026-11-08_cme_option_combined_manifest.csv"
WARMUP_MANIFEST = ROOT / "preregistrations/2026-11-08_cme_option_btc_warmup_manifest.csv"
CONTRACT = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
COUNTS = {2020: 52, 2021: 52, 2022: 52, 2023: 52, 2024: 53, 2025: 52}
FIELDS = {"Market_and_Exchange_Names", "Report_Date_as_YYYY-MM-DD",
          "Open_Interest_All", "Asset_Mgr_Positions_Long_All",
          "Asset_Mgr_Positions_Short_All"}


@dataclass(frozen=True)
class OptionReport:
    report_date: date
    option_net_ratio: float


def _manifest(path: Path, prefix: str) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise SourceFailure(f"{prefix} manifest unreadable: {exc}") from exc
    if [int(row["year"]) for row in rows] != list(COUNTS):
        raise SourceFailure(f"{prefix} year grid changed")
    for row in rows:
        expected = f"https://www.cftc.gov/files/dea/history/{prefix}_fin_txt_{row['year']}.zip"
        if row["url"] != expected or re.fullmatch("[0-9a-f]{64}", row["zip_sha256"]) is None:
            raise SourceFailure(f"{prefix} manifest URL or SHA malformed")
    return rows


def _parse_zip(body: bytes, year: int, prefix: str, expected_sha: str) -> dict[date, tuple[float, float, float]]:
    digest = sha256(body).hexdigest()
    if digest != expected_sha:
        raise SourceFailure(f"{prefix} {year} archive changed: {digest}")
    member = "FinFutYY.txt" if prefix == "fut" else "FinComYY.txt"
    try:
        with zipfile.ZipFile(BytesIO(body)) as archive:
            if archive.namelist() != [member]:
                raise SourceFailure(f"{prefix} {year} ZIP member changed")
            with TextIOWrapper(archive.open(member), encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames is None or len(reader.fieldnames) != 87 or not FIELDS.issubset(reader.fieldnames):
                    raise SourceFailure(f"{prefix} {year} CSV schema changed")
                selected: dict[date, tuple[float, float, float]] = {}
                for row in reader:
                    if row["Market_and_Exchange_Names"].strip() != CONTRACT:
                        continue
                    day = date.fromisoformat(row["Report_Date_as_YYYY-MM-DD"])
                    values = tuple(float(row[key].strip().replace(",", "")) for key in
                                   ("Open_Interest_All", "Asset_Mgr_Positions_Long_All",
                                    "Asset_Mgr_Positions_Short_All"))
                    if (day.year != year or day in selected or not all(math.isfinite(v) for v in values)
                            or values[0] <= 0 or values[1] < 0 or values[2] < 0):
                        raise SourceFailure(f"{prefix} {year} invalid CME BTC row {day}")
                    selected[day] = values
    except SourceFailure:
        raise
    except (zipfile.BadZipFile, csv.Error, ValueError, TypeError, KeyError,
            UnicodeError, OSError) as exc:
        raise SourceFailure(f"{prefix} {year} invalid archive: {exc}") from exc
    if len(selected) != COUNTS[year]:
        raise SourceFailure(f"{prefix} {year} CME BTC report count changed")
    return selected


def _load_cftc(row: dict[str, str], prefix: str) -> tuple[dict, dict]:
    year = int(row["year"])
    body = _fetch(row["url"])
    data = _parse_zip(body, year, prefix, row["zip_sha256"])
    return data, {"year": year, "url": row["url"], "sha256": row["zip_sha256"],
                  "bytes": len(body), "report_rows": len(data)}


def _early_spot_manifest() -> list[dict[str, str]]:
    try:
        with WARMUP_MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise SourceFailure(f"BTC warmup manifest unreadable: {exc}") from exc
    if [row["month"] for row in rows] != [f"2020-{m:02d}" for m in range(6, 12)]:
        raise SourceFailure("BTC warmup months changed")
    for row in rows:
        month = row["month"]
        expected = ("https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1d/"
                    f"BTCUSDT-1d-{month}.zip")
        if (row["url"] != expected or
                re.fullmatch("[0-9a-f]{64}", row["official_zip_sha256"]) is None):
            raise SourceFailure("BTC warmup URL/hash malformed")
    return rows


def load_sources() -> tuple[tuple[OptionReport, ...], pd.DataFrame, float, dict]:
    fut_manifest = _manifest(FUT_MANIFEST, "fut")
    com_manifest = _manifest(COM_MANIFEST, "com")
    jobs = [(row, "fut") for row in fut_manifest] + [(row, "com") for row in com_manifest]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(lambda item: _load_cftc(*item), jobs))
    futures: dict[date, tuple[float, float, float]] = {}
    combined: dict[date, tuple[float, float, float]] = {}
    for (data, _), (_, prefix) in zip(parts, jobs):
        target = futures if prefix == "fut" else combined
        if target.keys() & data.keys():
            raise SourceFailure(f"duplicate {prefix} CFTC report date")
        target.update(data)
    if futures.keys() != combined.keys() or len(futures) != 313:
        raise SourceFailure("CFTC futures/combined date join failed")
    dates = sorted(futures)
    if (dates[0] != date(2020, 1, 7) or dates[-1] != date(2025, 12, 30)
            or any((b - a).days not in (6, 7, 8) for a, b in zip(dates, dates[1:]))):
        raise SourceFailure("CFTC joined report-date grid changed")
    reports = []
    for day in dates:
        f_oi, f_long, f_short = futures[day]
        c_oi, c_long, c_short = combined[day]
        q = ((c_long - c_short) - (f_long - f_short)) / c_oi
        if not math.isfinite(q) or not -1 <= q <= 1:
            raise SourceFailure(f"{day}: invalid asset-manager option net ratio")
        reports.append(OptionReport(day, q))
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            price_parts = list(pool.map(_load_month, _early_spot_manifest() + _price_manifest()))
    except PriceSourceFailure as exc:
        raise SourceFailure(f"Binance BTC spot source failure: {exc}") from exc
    price_parts.sort(key=lambda item: item[0])
    terminal = price_parts[-1][1]
    if not isinstance(terminal, float):
        raise SourceFailure("BTC spot terminal open missing")
    frames = [item[1] for item in price_parts[:-1]]
    if not all(isinstance(item, pd.DataFrame) for item in frames):
        raise SourceFailure("BTC spot archive malformed")
    prices = pd.concat(frames)
    if not prices.index.equals(pd.date_range(date(2020, 6, 1),
                                              date(2025, 12, 31), tz="UTC", freq="D")):
        raise SourceFailure("joined BTC daily grid changed")
    return tuple(reports), prices, terminal, {
        "cftc_futures": [detail for _, detail in parts[:6]],
        "cftc_combined": [detail for _, detail in parts[6:]],
        "binance_spot": [item[2] for item in price_parts]}
