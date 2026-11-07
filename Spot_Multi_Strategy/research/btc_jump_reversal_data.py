"""Bounded, opt-in official archive feed for one preregistered BTC study.

Price bars are yielded only after each whole monthly ZIP passes the existing
strict kline audit and its SHA-256 matches the pre-result manifest. Funding
archives are checksum-verified and small enough to retain only parsed rate
events. No raw ZIP or CSV is persisted by this module.
"""

from __future__ import annotations

from calendar import monthrange
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import io
from pathlib import Path
import re
from typing import Iterable, Iterator
from zipfile import BadZipFile, ZipFile

from .btc_jump_reversal_sim import (
    DAY_MS, OOS_END_MS, OOS_START_MS, DataGateError, FundingRate, MinuteBar,
)
from .crypto_blackbox.binance_vision_5m import (
    ArchiveRequest, ChunkFetcher, _checksum, _chunks, _opened_verified_archive,
    http_chunk_fetcher,
)


MANIFEST_PATH = (
    Path(__file__).parent / "preregistrations"
    / "2026-11-07_btc_2h_jump_reversal_archive_manifest.csv"
)
_MANIFEST_FIELDS = (
    "month", "market", "symbol", "interval", "sha256", "rows",
    "missing", "duplicate",
)
_FUNDING_FIELDS = (
    "calc_time", "funding_interval_hours", "last_funding_rate",
)
_FUNDING_ZIP_LIMIT = 64 * 1024
_FUNDING_CSV_LIMIT = 128 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _months(start: int, end: int) -> tuple[str, ...]:
    return tuple(f"{year}-{month:02d}" for year in range(start, end + 1)
                 for month in range(1, 13))


def read_pinned_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, str]]:
    """Reject a changed month list, checksum, expected count, or scope."""
    try:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != _MANIFEST_FIELDS:
                raise DataGateError("unexpected preregistered manifest fields")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise DataGateError("preregistered manifest unavailable") from exc
    expected_months = _months(2020, 2025)
    if len(rows) != len(expected_months):
        raise DataGateError("manifest does not cover 72 fixed months")
    result = {}
    for row, month in zip(rows, expected_months):
        year, month_number = map(int, month.split("-"))
        if (
            set(row) != set(_MANIFEST_FIELDS)
            or any(row[field] is None or row[field] == "" for field in _MANIFEST_FIELDS)
            or
            row["month"] != month or row["market"] != "um"
            or row["symbol"] != "BTCUSDT" or row["interval"] != "1m"
            or not _SHA.fullmatch(row["sha256"] or "")
            or row["rows"] != str(monthrange(year, month_number)[1] * 1440)
            or row["missing"] != "0" or row["duplicate"] != "0"
        ):
            raise DataGateError(f"invalid manifest pin for {month}")
        result[month] = row
    return result


def iter_price_bars(
    *, fetch_chunks: ChunkFetcher = http_chunk_fetcher,
    manifest_path: Path = MANIFEST_PATH,
) -> Iterator[MinuteBar]:
    """Stream all fixed months; an error invalidates the entire experiment."""
    manifest = read_pinned_manifest(manifest_path)
    for month, pin in manifest.items():
        request = ArchiveRequest("um", "BTCUSDT", month, "1m")
        with _opened_verified_archive(
            request, fetch_chunks=fetch_chunks,
        ) as (archive, member, audit):
            if audit["sha256"] != pin["sha256"] or audit["rows"] != int(pin["rows"]):
                raise DataGateError(f"archive differs from pinned version: {month}")
            with archive.open(member) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                for fields in reader:
                    if fields[0] == "open_time":
                        continue
                    if len(fields) != 12:
                        raise DataGateError("audited kline changed during second read")
                    try:
                        yield MinuteBar(
                            int(fields[0]), float(fields[1]), float(fields[2]),
                            float(fields[3]), float(fields[4]),
                        )
                    except (ValueError, OverflowError) as exc:
                        raise DataGateError("invalid audited kline price") from exc


def _funding_url(month: str) -> tuple[str, str]:
    name = f"BTCUSDT-fundingRate-{month}.zip"
    return (
        "https://data.binance.vision/data/futures/um/monthly/fundingRate/"
        f"BTCUSDT/{name}", name,
    )


def _read_funding_month(
    month: str, fetch_chunks: ChunkFetcher,
    *, audit_hashes: dict[str, str] | None = None,
) -> list[FundingRate]:
    url, name = _funding_url(month)
    expected_hash = _checksum(fetch_chunks, url, name)
    digest = sha256()
    payload = bytearray()
    for chunk in _chunks(fetch_chunks, url):
        if len(payload) + len(chunk) > _FUNDING_ZIP_LIMIT:
            raise DataGateError("funding archive exceeds byte limit")
        digest.update(chunk)
        payload.extend(chunk)
    actual_hash = digest.hexdigest()
    if actual_hash != expected_hash:
        raise DataGateError("funding ZIP differs from official CHECKSUM")
    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            if (
                len(members) != 1 or members[0].filename != name[:-4] + ".csv"
                or members[0].is_dir() or members[0].flag_bits & 1
                or members[0].file_size > _FUNDING_CSV_LIMIT
            ):
                raise DataGateError("unexpected or oversized funding member")
            with archive.open(members[0]) as raw:
                reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""))
                if tuple(next(reader, ())) != _FUNDING_FIELDS:
                    raise DataGateError("unexpected funding schema")
                events = []
                year, month_number = map(int, month.split("-"))
                start = int(datetime(
                    year, month_number, 1, tzinfo=timezone.utc
                ).timestamp() * 1000)
                end = start + monthrange(year, month_number)[1] * DAY_MS
                for fields in reader:
                    if len(fields) != 3:
                        raise DataGateError("malformed funding record")
                    try:
                        stamp = int(fields[0])
                        interval = Decimal(fields[1])
                        rate = Decimal(fields[2])
                    except (ValueError, InvalidOperation) as exc:
                        raise DataGateError("invalid funding number") from exc
                    if (
                        not start <= stamp < end
                        or not interval.is_finite() or interval <= 0
                        or not rate.is_finite()
                    ):
                        raise DataGateError("funding record outside its month or nonfinite")
                    events.append(FundingRate(stamp, float(rate)))
            if not events:
                raise DataGateError(f"empty funding month: {month}")
            if audit_hashes is not None:
                audit_hashes[month] = actual_hash
            return events
    except (BadZipFile, EOFError, RuntimeError, UnicodeError, csv.Error) as exc:
        raise DataGateError("invalid funding ZIP or CSV") from exc


def validate_funding_sequence(events: Iterable[FundingRate]) -> list[FundingRate]:
    result = list(events)
    previous = None
    for event in result:
        if not isinstance(event, FundingRate):
            raise DataGateError("invalid funding event")
        if (
            previous is not None and (
                event.calc_time_ms <= previous
                or event.calc_time_ms - previous > 12 * 60 * 60 * 1000
            )
        ):
            raise DataGateError("duplicate, unsorted, or >12h funding gap")
        previous = event.calc_time_ms
    if (
        not result or result[0].calc_time_ms > OOS_START_MS + 12 * 60 * 60 * 1000
        or result[-1].calc_time_ms < OOS_END_MS - 12 * 60 * 60 * 1000
    ):
        raise DataGateError("funding series does not span the fixed OOS")
    return result


def load_funding(
    *, fetch_chunks: ChunkFetcher = http_chunk_fetcher,
    audit_hashes: dict[str, str] | None = None,
) -> list[FundingRate]:
    """Load only 2021-2025 archived rates, never a current substitute."""
    all_events = []
    for month in _months(2021, 2025):
        all_events.extend(_read_funding_month(
            month, fetch_chunks, audit_hashes=audit_hashes,
        ))
    return validate_funding_sequence(all_events)
