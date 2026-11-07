"""Pinned Zenodo Bitcoin old-coin-spend source for one isolated study."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from io import BytesIO, StringIO
import tarfile
from typing import Callable
from urllib.parse import urlparse

import requests


URL = "https://zenodo.org/api/records/21156871/files/open_bitcoin_metrics-v0.1.0.tar.gz/content"
ARCHIVE_BYTES = 4_410_880
ARCHIVE_SHA256 = "d5caa010dd0dfc08b6a428fe733b15265b5a159ebfe8a38c24ad6c799641a26c"
MEMBER = ("open_bitcoin_metrics/metrics/obm_spent_value_ge365d_btc_daily/"
          "obm_spent_value_ge365d_btc_daily.csv")
MEMBER_BYTES = 502_363
HEADER = ["date", "series_id", "value", "unit", "frequency", "release_version"]
SERIES_ID = "obm_spent_value_ge365_btc_daily"
SOURCE_FIRST = date(2009, 1, 1)
SOURCE_LAST = date(2026, 6, 28)
FIRST = date(2020, 11, 1)
LAST = SOURCE_LAST
SOURCE_ROWS = 6388
ACCEPTED_ROWS = 2066


class OldCoinSourceError(RuntimeError):
    """The pinned source violates the preregistered data contract."""


@dataclass(frozen=True)
class OldCoinObservation:
    observed_date: date
    spent_btc: Decimal
    proxy_available_at: datetime


@dataclass(frozen=True)
class OldCoinBundle:
    observations: tuple[OldCoinObservation, ...]
    manifest: dict


def _download() -> tuple[bytes, str]:
    for attempt in range(3):
        try:
            with requests.get(URL, timeout=(10, 60), stream=True) as response:
                if (urlparse(response.url).hostname != "zenodo.org" or
                        response.url.split("?", 1)[0] != URL):
                    raise OldCoinSourceError("Zenodo archive redirected or URL changed")
                if response.status_code == 200:
                    body = bytearray()
                    for chunk in response.iter_content(chunk_size=65_536):
                        body.extend(chunk)
                        if len(body) > ARCHIVE_BYTES:
                            raise OldCoinSourceError("Zenodo archive exceeds pinned size")
                    if body:
                        return bytes(body), response.url
                elif response.status_code not in (429, 500, 502, 503, 504):
                    raise OldCoinSourceError(f"Zenodo HTTP {response.status_code}")
                if attempt == 2:
                    raise OldCoinSourceError(f"Zenodo HTTP {response.status_code} or empty body")
        except requests.RequestException as exc:
            if attempt == 2:
                raise OldCoinSourceError("Zenodo archive request failed") from exc
        import time as clock
        clock.sleep(0.5 * 2**attempt)
    raise AssertionError("unreachable")


def _parse(body: bytes, final_url: str) -> OldCoinBundle:
    if (not isinstance(body, bytes) or len(body) != ARCHIVE_BYTES or
            sha256(body).hexdigest() != ARCHIVE_SHA256):
        raise OldCoinSourceError("Zenodo archive byte count or SHA-256 changed")
    if (urlparse(final_url).hostname != "zenodo.org" or
            final_url.split("?", 1)[0] != URL):
        raise OldCoinSourceError("untrusted Zenodo source URL")
    try:
        with tarfile.open(fileobj=BytesIO(body), mode="r:gz") as archive:
            matches = [item for item in archive.getmembers() if item.name == MEMBER]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size != MEMBER_BYTES:
                raise OldCoinSourceError("pinned old-coin CSV member missing or changed")
            member_file = archive.extractfile(matches[0])
            if member_file is None:
                raise OldCoinSourceError("cannot extract old-coin CSV")
            data = member_file.read()
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise OldCoinSourceError("invalid Zenodo tar archive") from exc
    if len(data) != MEMBER_BYTES:
        raise OldCoinSourceError("old-coin CSV member byte count changed")
    try:
        decoded = data.decode("utf-8")
        rows = list(csv.reader(StringIO(decoded, newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise OldCoinSourceError("old-coin CSV encoding or quoting invalid") from exc
    if not rows or rows[0] != HEADER or len(rows) != SOURCE_ROWS + 1:
        raise OldCoinSourceError("old-coin CSV header or row count changed")
    observations = []
    for i, row in enumerate(rows[1:]):
        if len(row) != 6:
            raise OldCoinSourceError("old-coin CSV row schema changed")
        try:
            stamp = date.fromisoformat(row[0])
            amount = Decimal(row[2])
        except (ValueError, InvalidOperation) as exc:
            raise OldCoinSourceError("old-coin CSV date or value invalid") from exc
        if (stamp != SOURCE_FIRST + timedelta(days=i) or
                row[1] != SERIES_ID or row[3:] != ["BTC", "daily", "OBM v0.1.0"] or
                not amount.is_finite() or amount < 0):
            raise OldCoinSourceError("old-coin CSV coverage, metadata or value invalid")
        if stamp >= FIRST:
            observations.append(OldCoinObservation(
                stamp, amount,
                datetime.combine(stamp + timedelta(days=4), time.min,
                                 timezone.utc)))
    if len(observations) != ACCEPTED_ROWS or observations[-1].observed_date != LAST:
        raise OldCoinSourceError("registered old-coin daily coverage changed")
    manifest = {"url": final_url, "archive_bytes": len(body),
                "archive_sha256": sha256(body).hexdigest(),
                "member": MEMBER, "member_bytes": len(data),
                "member_sha256": sha256(data).hexdigest(),
                "source_rows": SOURCE_ROWS, "accepted_days": len(observations),
                "first_used_date": FIRST.isoformat(), "last_used_date": LAST.isoformat(),
                "series_id": SERIES_ID, "unit": "BTC", "release": "OBM v0.1.0",
                "historical_vintage_verified": False}
    return OldCoinBundle(tuple(observations), manifest)


def load_and_audit(*, fetcher: Callable[[], tuple[bytes, str]] | None = None) -> OldCoinBundle:
    body, url = (fetcher or _download)()
    return _parse(body, url)
