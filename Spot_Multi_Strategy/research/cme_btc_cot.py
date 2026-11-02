"""Frozen CFTC TFF futures-only input and positioning signal for CME BTC.

Only the official 2018--2025 annual archives are accepted. ZIP and CSV bytes
stay in memory; the manifests contain provenance and coverage, never positions.
An invalid or incomplete archive raises :class:`COTDataError` so callers can
seal a DATA_FAILURE before computing any trading results.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from hashlib import sha256
from io import BytesIO, TextIOWrapper
import re
import time as clock
from typing import Callable, Iterable
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile

import requests


UTC = timezone.utc
YEARS = tuple(range(2018, 2026))
CONTRACT_CODE = "133741"
MARKET_NAME = "BITCOIN - CHICAGO MERCANTILE EXCHANGE"
BASE_URL = "https://www.cftc.gov/files/dea/history"
MAX_ZIP_BYTES = 16 * 1024 * 1024
MAX_CSV_BYTES = 96 * 1024 * 1024
MAX_CSV_ROWS = 100_000
MAX_COLUMNS = 128
MIN_REPORTS_PER_YEAR = 40
BTC_FIRST_REPORT = date(2018, 4, 10)
REQUIRED_COLUMNS = (
    "Market_and_Exchange_Names",
    "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code",
    "Open_Interest_All",
    "Asset_Mgr_Positions_Long_All",
    "Asset_Mgr_Positions_Short_All",
    "FutOnly_or_Combined",
)
BLACKOUTS = (
    (date(2018, 12, 24), date(2019, 3, 5)),
    (date(2023, 1, 31), date(2023, 3, 14)),
    (date(2025, 9, 30), date(2025, 12, 23)),
)
_INTEGER = re.compile(r"(?:0|[1-9][0-9]*|[1-9][0-9]{0,2}(?:,[0-9]{3})+)")


class COTDataError(ValueError):
    """The frozen official COT input is unavailable, malformed or incomplete."""


@dataclass(frozen=True)
class COTReport:
    report_date: date
    long: int
    short: int
    open_interest: int


@dataclass(frozen=True)
class COTArchive:
    reports: tuple[COTReport, ...]
    manifest: dict


@dataclass(frozen=True)
class COTDataset:
    reports: tuple[COTReport, ...]
    manifest: dict


def archive_url(year: int) -> str:
    if type(year) is not int or year not in YEARS:
        raise ValueError("CFTC archive year must be in 2018--2025")
    return f"{BASE_URL}/fut_fin_txt_{year}.zip"


def _download(url: str) -> bytes:
    """Stream a bounded official ZIP, retrying only transient HTTP failures."""
    headers = {"User-Agent": "MiniMedallion-CFTC-TFF-Auditor/1.0"}
    for attempt in range(3):
        try:
            with requests.get(url, headers=headers, timeout=(10, 60), stream=True) as response:
                if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                    clock.sleep(0.5 * 2**attempt)
                    continue
                response.raise_for_status()
                output = BytesIO()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if output.tell() + len(chunk) > MAX_ZIP_BYTES:
                        raise COTDataError(f"{url}: ZIP exceeds byte cap")
                    output.write(chunk)
                return output.getvalue()
        except COTDataError:
            raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 2:
                raise COTDataError(f"{url}: download failed") from exc
            clock.sleep(0.5 * 2**attempt)
        except requests.RequestException as exc:
            raise COTDataError(f"{url}: HTTP failure") from exc
    raise RuntimeError("unreachable download retry state")


def _report_date(raw: str, year: int, where: str) -> date:
    value = raw.strip()
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise COTDataError(f"{where}: invalid report date")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise COTDataError(f"{where}: invalid report date") from exc
    if result.year != year:
        raise COTDataError(f"{where}: report date outside archive year")
    return result


def _nonnegative_int(raw: str, where: str) -> int:
    value = raw.strip()
    if _INTEGER.fullmatch(value) is None:
        raise COTDataError(f"{where}: expected nonnegative finite integer")
    return int(value.replace(",", ""))


def _validate_report(report: COTReport) -> None:
    if type(report.report_date) is not date:
        raise COTDataError("report date must be a date")
    for name, value in (("long", report.long), ("short", report.short),
                        ("open_interest", report.open_interest)):
        if type(value) is not int or value < 0:
            raise COTDataError(f"{report.report_date}: invalid {name}")
    if report.open_interest == 0:
        raise COTDataError(f"{report.report_date}: zero open interest")
    if report.long > report.open_interest or report.short > report.open_interest:
        raise COTDataError(f"{report.report_date}: position exceeds open interest")


def _is_blacked_out(report_date: date) -> bool:
    return any(start <= report_date <= end for start, end in BLACKOUTS)


def _sequence_anomalies(reports: tuple[COTReport, ...]) -> list[dict]:
    anomalies: list[dict] = []
    previous: date | None = None
    for report in reports:
        _validate_report(report)
        current = report.report_date
        if previous is not None:
            days = (current - previous).days
            if days <= 0:
                raise COTDataError(f"{current}: duplicate or nonmonotone report date")
            if days < 5:
                raise COTDataError(f"{current}: reports less than five days apart")
            if days > 10:
                anomalies.append({
                    "kind": "weekly_gap", "previous": previous.isoformat(),
                    "next": current.isoformat(), "days": days,
                })
        previous = current
    return anomalies


def parse_annual_zip(payload: bytes, year: int) -> COTArchive:
    """Validate one official annual TFF futures-only ZIP using bounded memory.

    This function is also the synthetic fixture entry point. It does not
    perform network I/O and never writes the original archive to disk.
    """
    url = archive_url(year)
    if not isinstance(payload, bytes) or not payload or len(payload) > MAX_ZIP_BYTES:
        raise COTDataError(f"{url}: invalid or oversized ZIP payload")
    digest = sha256(payload).hexdigest()
    try:
        with ZipFile(BytesIO(payload)) as zipped:
            members = zipped.infolist()
            if len(members) != 1:
                raise COTDataError(f"{url}: ZIP must contain exactly one data member")
            member = members[0]
            member_name = member.filename
            if (member.is_dir() or member_name != member_name.rsplit("/", 1)[-1]
                    or not member_name.lower().endswith((".txt", ".csv"))
                    or member.flag_bits & 0x1
                    or member.compress_type not in (ZIP_STORED, ZIP_DEFLATED)
                    or member.file_size == 0 or member.file_size > MAX_CSV_BYTES
                    or member.compress_size == 0):
                raise COTDataError(f"{url}: unsafe or unsupported ZIP member")
            with zipped.open(member) as raw, TextIOWrapper(
                raw, encoding="utf-8-sig", errors="strict", newline=""
            ) as text:
                reader = csv.reader(text, strict=True)
                header = next(reader, None)
                if (header is None or len(header) > MAX_COLUMNS
                        or len(header) != len(set(header))
                        or any(not name for name in header)):
                    raise COTDataError(f"{url}: invalid CSV header")
                missing = set(REQUIRED_COLUMNS).difference(header)
                if missing:
                    raise COTDataError(f"{url}: missing required columns: {sorted(missing)}")
                index = {name: header.index(name) for name in REQUIRED_COLUMNS}
                reports: list[COTReport] = []
                all_dates: set[date] = set()
                row_count = 0
                for row in reader:
                    row_count += 1
                    if row_count > MAX_CSV_ROWS:
                        raise COTDataError(f"{url}: CSV exceeds row cap")
                    where = f"{url} row {row_count}"
                    if len(row) != len(header):
                        raise COTDataError(f"{where}: inconsistent column count")
                    report_date = _report_date(
                        row[index["Report_Date_as_YYYY-MM-DD"]], year, where
                    )
                    all_dates.add(report_date)
                    if row[index["FutOnly_or_Combined"]].strip() != "FutOnly":
                        raise COTDataError(f"{where}: not TFF futures only")
                    if row[index["CFTC_Contract_Market_Code"]].strip() != CONTRACT_CODE:
                        continue
                    market = " ".join(row[index["Market_and_Exchange_Names"]].split())
                    if market != MARKET_NAME:
                        raise COTDataError(f"{where}: wrong BTC market or exchange")
                    report = COTReport(
                        report_date=report_date,
                        long=_nonnegative_int(row[index["Asset_Mgr_Positions_Long_All"]],
                                              f"{where} asset manager long"),
                        short=_nonnegative_int(row[index["Asset_Mgr_Positions_Short_All"]],
                                               f"{where} asset manager short"),
                        open_interest=_nonnegative_int(row[index["Open_Interest_All"]],
                                                       f"{where} open interest"),
                    )
                    _validate_report(report)
                    reports.append(report)
    except (BadZipFile, EOFError, OSError, UnicodeError, csv.Error,
            RuntimeError, StopIteration) as exc:
        raise COTDataError(f"{url}: malformed ZIP or CSV") from exc
    if not reports:
        raise COTDataError(f"{url}: standard CME BTC rows absent")
    reports.sort(key=lambda report: report.report_date)
    frozen = tuple(reports)
    anomalies = _sequence_anomalies(frozen)
    btc_dates = {report.report_date for report in frozen}
    covered_market_dates = {
        report_date for report_date in all_dates
        if frozen[0].report_date <= report_date <= frozen[-1].report_date
    }
    if covered_market_dates != btc_dates:
        missing_dates = sorted(covered_market_dates.difference(btc_dates))
        raise COTDataError(f"{url}: standard CME BTC row absent for {missing_dates[0]}")
    manifest = {
        "year": year,
        "url": url,
        "sha256": digest,
        "zip_bytes": len(payload),
        "member": member_name,
        "header": header,
        "csv_rows": row_count,
        "report_rows": len(frozen),
        "first_report_date": frozen[0].report_date.isoformat(),
        "last_report_date": frozen[-1].report_date.isoformat(),
        "anomalies": anomalies,
    }
    return COTArchive(frozen, manifest)


def fetch_cot(*, fetcher: Callable[[str], bytes] | None = None) -> COTDataset:
    """Fetch and audit exactly the eight frozen CFTC annual archives.

    A synthetic ``fetcher(url) -> bytes`` may be injected by tests. The
    default fetcher streams official ZIPs without caching raw data on disk.
    """
    get_bytes = _download if fetcher is None else fetcher
    archives: list[COTArchive] = []
    for year in YEARS:
        url = archive_url(year)
        try:
            payload = get_bytes(url)
        except (OSError, KeyError, requests.RequestException) as exc:
            raise COTDataError(f"{url}: archive unavailable") from exc
        archive = parse_annual_zip(payload, year)
        minimum = 39 if year == 2018 else MIN_REPORTS_PER_YEAR
        if len(archive.reports) < minimum:
            raise COTDataError(f"{url}: too few weekly CME BTC reports")
        if year == 2018 and archive.reports[0].report_date != BTC_FIRST_REPORT:
            raise COTDataError(f"{url}: unexpected first CME BTC report")
        if year != 2018 and archive.reports[0].report_date > date(year, 1, 10):
            raise COTDataError(f"{url}: start of year missing")
        if archive.reports[-1].report_date < date(year, 12, 21):
            raise COTDataError(f"{url}: end of year missing")
        archives.append(archive)
    headers = {tuple(archive.manifest["header"]) for archive in archives}
    if len(headers) != 1:
        raise COTDataError("CFTC TFF schema differs across annual archives")
    reports = tuple(report for archive in archives for report in archive.reports)
    anomalies = _sequence_anomalies(reports)
    for anomaly in anomalies:
        before = date.fromisoformat(anomaly["previous"])
        after = date.fromisoformat(anomaly["next"])
        if not any(before < end and after > start for start, end in BLACKOUTS):
            raise COTDataError(
                f"unexplained missing weekly CME BTC report: {before} to {after}"
            )
    manifest = {
        "source": "CFTC TFF futures only annual archives",
        "contract_market_code": CONTRACT_CODE,
        "archives": [archive.manifest for archive in archives],
        "archive_count": len(archives),
        "report_rows": len(reports),
        "first_report_date": reports[0].report_date.isoformat(),
        "last_report_date": reports[-1].report_date.isoformat(),
        "anomalies": anomalies,
    }
    return COTDataset(reports, manifest)


def desired_exposure(reports: Iterable[COTReport], decision_time: datetime) -> bool:
    """Return the preregistered BTC target at a UTC 00:00 decision.

    The chosen report is available at report date +14 days, never earlier.
    Execution belongs to the caller on the *next* daily 00:00 open.
    Structural corruption raises COTDataError; insufficient or stale signal
    history returns cash (False).
    """
    if (not isinstance(decision_time, datetime) or decision_time.tzinfo is None
            or decision_time.utcoffset() != timedelta(0)
            or decision_time.time() != time.min):
        raise ValueError("decision_time must be a UTC 00:00 datetime")
    sequence = tuple(reports)
    _sequence_anomalies(sequence)
    available = [
        report for report in sequence
        if not _is_blacked_out(report.report_date)
        and report.report_date + timedelta(days=14) <= decision_time.date()
    ]
    if len(available) < 5:
        return False
    recent = available[-5:]
    newest = recent[-1]
    if (decision_time.date() - newest.report_date).days > 21:
        return False
    if any(
        not 5 <= (right.report_date - left.report_date).days <= 10
        for left, right in zip(recent, recent[1:])
    ):
        return False
    oldest = recent[0]
    return (
        (newest.long - newest.short) * oldest.open_interest
        > (oldest.long - oldest.short) * newest.open_interest
    )
