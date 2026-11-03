"""Fail-closed, single-request Alternative.me FNG snapshot loader.

The fixed historical grid and source are preregistered in b2db5a6. This
module does not calculate signals or returns. It keeps the raw response only
in memory and records its provenance without persisting the payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import re
from typing import Callable

import pandas as pd
import requests


FNG_URL = "https://api.alternative.me/fng/?limit=0"
START = pd.Timestamp("2020-12-30", tz="UTC")
END = pd.Timestamp("2025-12-24", tz="UTC")
EXPECTED_DAYS = 1821
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class FNGIntegrityError(ValueError):
    """The one allowed source response is unavailable or fails the contract."""

    def __init__(self, message: str, *, manifest: dict | None = None):
        super().__init__(message)
        self.manifest = manifest


@dataclass(frozen=True)
class Observation:
    day: pd.Timestamp
    value: int


@dataclass(frozen=True)
class DataBundle:
    observations: tuple[Observation, ...]
    manifest: dict


@dataclass(frozen=True)
class _Payload:
    body: bytes
    status_code: int
    content_type: str | None
    final_url: str


def _http_fetch(url: str) -> _Payload:
    """Make exactly one GET, with no retry and no alternate source."""
    try:
        response = requests.get(
            url, timeout=(10, 60), allow_redirects=False,
            headers={"User-Agent": "MiniMedallionResearch/1.0 (https://github.com/Nu1LLL/Quant)"},
        )
        try:
            return _Payload(
                body=response.content,
                status_code=response.status_code,
                content_type=response.headers.get("Content-Type"),
                final_url=response.url,
            )
        finally:
            response.close()
    except requests.RequestException as exc:
        raise FNGIntegrityError("FNG source download failed") from exc


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise FNGIntegrityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(token: str) -> None:
    raise FNGIntegrityError(f"nonstandard JSON number: {token}")


def _parse_observations(body: bytes) -> tuple[Observation, ...]:
    try:
        document = json.loads(
            body, object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FNGIntegrityError("malformed FNG JSON") from exc
    if not isinstance(document, dict):
        raise FNGIntegrityError("FNG JSON root must be an object")
    metadata = document.get("metadata")
    if not isinstance(metadata, dict) or "error" not in metadata or metadata["error"] is not None:
        raise FNGIntegrityError("FNG metadata.error must be null")
    records = document.get("data")
    if not isinstance(records, list):
        raise FNGIntegrityError("FNG data must be an array")

    within: dict[pd.Timestamp, Observation] = {}
    for record in records:
        if not isinstance(record, dict):
            raise FNGIntegrityError("malformed FNG observation")
        raw_timestamp = record.get("timestamp")
        raw_value = record.get("value")
        if not isinstance(raw_timestamp, str) or not re.fullmatch(r"[0-9]{10}", raw_timestamp):
            raise FNGIntegrityError("FNG timestamp must be a Unix-seconds string")
        if not isinstance(raw_value, str) or not re.fullmatch(r"[0-9]+", raw_value):
            raise FNGIntegrityError("FNG value must be a decimal integer string")
        value = int(raw_value)
        if value < 0 or value > 100:
            raise FNGIntegrityError("FNG value outside 0..100")
        try:
            moment = pd.Timestamp(int(raw_timestamp), unit="s", tz="UTC")
        except (ValueError, OverflowError) as exc:
            raise FNGIntegrityError("FNG timestamp outside Unix-seconds range") from exc
        day = moment.normalize()
        if START <= day <= END:
            if day in within:
                raise FNGIntegrityError(f"duplicate FNG UTC day: {day:%Y-%m-%d}")
            within[day] = Observation(day=day, value=value)

    expected = pd.date_range(START, END, freq="D", tz="UTC")
    if len(expected) != EXPECTED_DAYS:
        raise RuntimeError("preregistered FNG coverage constant disagrees with calendar")
    missing = [day for day in expected if day not in within]
    if missing:
        raise FNGIntegrityError(
            f"incomplete FNG UTC daily grid: {len(missing)} missing; first {missing[0]:%Y-%m-%d}"
        )
    return tuple(within[day] for day in expected)


def load_and_audit(*, fetcher: Callable[[str], bytes | _Payload] | None = None) -> DataBundle:
    """Fetch exactly the preregistered endpoint and validate 1,821 UTC days.

    ``fetcher`` is for deterministic synthetic tests. Production cannot
    change source, boundaries, or expected grid through this public API.
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    source = _http_fetch if fetcher is None else fetcher
    try:
        payload = source(FNG_URL)
    except FNGIntegrityError:
        raise
    except (requests.RequestException, OSError, KeyError) as exc:
        raise FNGIntegrityError("FNG source download failed") from exc
    if isinstance(payload, bytes):
        payload = _Payload(payload, 200, None, FNG_URL)
    if not isinstance(payload, _Payload) or not isinstance(payload.body, bytes):
        raise FNGIntegrityError("FNG fetcher returned invalid payload")
    manifest = {
        "url": FNG_URL,
        "http_status": payload.status_code,
        "http_content_type": payload.content_type,
        "http_final_url": payload.final_url,
        "fetched_at_utc": fetched_at,
        "response_bytes": len(payload.body),
        "sha256": sha256(payload.body).hexdigest(),
        "coverage_start": START.isoformat(),
        "coverage_end": END.isoformat(),
        "coverage_days": EXPECTED_DAYS,
        "source_attribution": "Alternative.me Crypto Fear & Greed Index",
    }
    json.dumps(manifest, allow_nan=False)
    if payload.status_code != 200:
        raise FNGIntegrityError(f"FNG HTTP status {payload.status_code}", manifest=manifest)
    if payload.final_url != FNG_URL:
        raise FNGIntegrityError("FNG final URL differs from preregistered source", manifest=manifest)
    if len(payload.body) > MAX_RESPONSE_BYTES:
        raise FNGIntegrityError("FNG response exceeds size limit", manifest=manifest)
    if payload.content_type is not None and "json" not in payload.content_type.lower():
        raise FNGIntegrityError("FNG response is not JSON content type", manifest=manifest)
    try:
        observations = _parse_observations(payload.body)
    except FNGIntegrityError as exc:
        raise FNGIntegrityError(str(exc), manifest=manifest) from exc
    return DataBundle(observations=observations, manifest=manifest)
