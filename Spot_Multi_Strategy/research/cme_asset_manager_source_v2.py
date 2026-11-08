"""Hash-corrected frozen CFTC source version; no strategy changes."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import csv
import re

import pandas as pd

from research.btc_exchange_supply_source import SourceFailure as PriceSourceFailure
from research.btc_exchange_supply_source import _load_month, _manifest
from research.cme_asset_manager_source import EXPECTED_COUNTS, Position, SourceFailure, _load_cftc


MANIFEST = Path(__file__).resolve().parent / "preregistrations/2026-11-08_cme_asset_manager_manifest_v2.csv"


def _source_manifest() -> list[dict[str, str]]:
    try:
        with MANIFEST.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise SourceFailure(f"CFTC v2 manifest unreadable: {exc}") from exc
    if [int(x["year"]) for x in rows] != list(EXPECTED_COUNTS):
        raise SourceFailure("CFTC v2 manifest year grid changed")
    for row in rows:
        expected = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{row['year']}.zip"
        if row["url"] != expected or re.fullmatch(r"[0-9a-f]{64}", row["zip_sha256"]) is None:
            raise SourceFailure("CFTC v2 manifest URL/hash malformed")
    return rows


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
